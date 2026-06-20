"""Integrity proofs and reasoning certification for ChimeraLang.

Generates cryptographic-style proofs that a reasoning path was followed,
including:
- Reasoning trace hashes (Merkle-like chain)
- Gate consensus certificates
- Confidence audit trail
- Integrity report generation
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass, field
from typing import Any

from chimera.detect import DetectionReport, HallucinationFlag
from chimera.vm import ExecutionResult


def _canonical_bytes(obj: Any) -> bytes:
    """Deterministic JSON encoding shared by producer and verifier.

    Both the certificate producer (here) and the independent verifier
    (chimera/verify.py) MUST serialize byte-for-byte identically, or hash
    bindings and signatures will never match. Keep this definition in sync.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _is_json_safe(value: Any) -> bool:
    """True when value can be embedded in the certificate without losing fidelity."""
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return True


@dataclass(frozen=True, slots=True)
class TraceLink:
    """One link in the reasoning chain."""
    index: int
    entry: str
    hash: str
    prev_hash: str


@dataclass
class ReasoningChain:
    """Merkle-like chain of reasoning steps."""
    links: list[TraceLink] = field(default_factory=list)

    @property
    def root_hash(self) -> str:
        if not self.links:
            return hashlib.sha256(b"empty").hexdigest()[:32]
        return self.links[-1].hash

    @property
    def length(self) -> int:
        return len(self.links)


class ChainBuilder:
    """Builds a reasoning chain from an execution trace."""

    @staticmethod
    def build(trace: list[str]) -> ReasoningChain:
        chain = ReasoningChain()
        prev = "genesis"
        for i, entry in enumerate(trace):
            data = f"{prev}:{entry}".encode()
            h = hashlib.sha256(data).hexdigest()[:32]
            chain.links.append(TraceLink(
                index=i,
                entry=entry,
                hash=h,
                prev_hash=prev,
            ))
            prev = h
        return chain

    @staticmethod
    def verify(chain: ReasoningChain) -> bool:
        """Verify chain integrity — every link references the correct predecessor."""
        if not chain.links:
            return True
        if chain.links[0].prev_hash != "genesis":
            return False
        for i in range(1, len(chain.links)):
            if chain.links[i].prev_hash != chain.links[i - 1].hash:
                return False
        return True


@dataclass
class GateCertificate:
    """Proof that a gate reached consensus."""
    gate_name: str
    branches: int
    collapse_strategy: str
    branch_confidences: list[float]
    result_confidence: float
    result_value: Any
    hash: str = ""

    def __post_init__(self) -> None:
        data = json.dumps({
            "gate": self.gate_name,
            "branches": self.branches,
            "collapse": self.collapse_strategy,
            "confs": self.branch_confidences,
            "result_conf": self.result_confidence,
        }, sort_keys=True).encode()
        self.hash = hashlib.sha256(data).hexdigest()[:32]


@dataclass
class IntegrityReport:
    """Full integrity report for a program execution."""
    timestamp: float = field(default_factory=time.time)
    program_hash: str = ""
    reasoning_chain: ReasoningChain = field(default_factory=ReasoningChain)
    chain_valid: bool = True
    gate_certificates: list[GateCertificate] = field(default_factory=list)
    assertions_passed: int = 0
    assertions_failed: int = 0
    hallucination_flags: list[HallucinationFlag] = field(default_factory=list)
    hallucination_clean: bool = True
    verdict: str = "UNKNOWN"
    duration_ms: float = 0.0

    def to_dict(self, full: bool = False) -> dict[str, Any]:
        chain: dict[str, Any] = {
            "length": self.reasoning_chain.length,
            "root_hash": self.reasoning_chain.root_hash,
            "valid": self.chain_valid,
        }
        gates: list[dict[str, Any]] = [
            {
                "name": gc.gate_name,
                "branches": gc.branches,
                "collapse": gc.collapse_strategy,
                "result_confidence": gc.result_confidence,
                "hash": gc.hash,
            }
            for gc in self.gate_certificates
        ]
        hallucination: dict[str, Any] = {
            "clean": self.hallucination_clean,
            "flags": len(self.hallucination_flags),
        }

        if full:
            # Complete chain links so a verifier can recompute every hash.
            chain["links"] = [
                {
                    "index": link.index,
                    "entry": link.entry,
                    "hash": link.hash,
                    "prev_hash": link.prev_hash,
                }
                for link in self.reasoning_chain.links
            ]
            # Complete gate fields so a verifier can recompute the gate hash.
            for gate_dict, gc in zip(gates, self.gate_certificates):
                gate_dict["branch_confidences"] = gc.branch_confidences
                if _is_json_safe(gc.result_value):
                    gate_dict["result_value"] = gc.result_value
            # Flag severities so a verifier can recompute the verdict exactly.
            hallucination["flags_detail"] = [
                {
                    "kind": f.kind.name,
                    "severity": f.severity,
                    "description": f.description,
                }
                for f in self.hallucination_flags
            ]

        return {
            "timestamp": self.timestamp,
            "program_hash": self.program_hash,
            "chain": chain,
            "gates": gates,
            "assertions": {
                "passed": self.assertions_passed,
                "failed": self.assertions_failed,
            },
            "hallucination": hallucination,
            "verdict": self.verdict,
            "duration_ms": self.duration_ms,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def to_certificate(
        self,
        *,
        hmac_key: bytes | None = None,
        sign_key: Any = None,
    ) -> dict[str, Any]:
        """Build a portable, self-verifying certificate.

        Guarantees, stated precisely:
          - certificate_hash (SHA-256 over the canonical report) = tamper-evidence.
          - hmac (HMAC-SHA256, optional) = authentication via a shared secret.
          - signature (Ed25519, optional) = asymmetric, third-party-verifiable
            signature; requires the optional `cryptography` package.
        """
        report = self.to_dict(full=True)
        report_bytes = _canonical_bytes(report)

        binding: dict[str, Any] = {
            "algo": "sha256",
            "certificate_hash": hashlib.sha256(report_bytes).hexdigest(),
            "hmac": None,
            "signature": None,
        }

        if hmac_key is not None:
            binding["hmac"] = hmac.new(hmac_key, report_bytes, hashlib.sha256).hexdigest()

        if sign_key is not None:
            binding["signature"] = _sign_ed25519(sign_key, report_bytes)

        return {
            "format": "chimeralang-cert/v1",
            "report": report,
            "binding": binding,
        }


def _sign_ed25519(sign_key: Any, message: bytes) -> dict[str, str]:
    """Sign `message` with Ed25519. Producing a signature requires `cryptography`."""
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import ed25519
    except ImportError as exc:  # pragma: no cover - exercised via env without crypto
        raise RuntimeError(
            "Ed25519 signing requires the 'cryptography' package. "
            "Install it with: pip install cryptography"
        ) from exc

    if isinstance(sign_key, (bytes, bytearray)):
        private_key = serialization.load_pem_private_key(bytes(sign_key), password=None)
    else:
        private_key = sign_key

    if not isinstance(private_key, ed25519.Ed25519PrivateKey):
        raise RuntimeError("sign_key must be an Ed25519 private key (object or PEM bytes)")

    signature = private_key.sign(message)
    public_raw = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return {
        "algo": "ed25519",
        "pubkey": public_raw.hex(),
        "sig": signature.hex(),
    }


class IntegrityEngine:
    """Generates integrity reports from execution results."""

    def certify(
        self,
        exec_result: ExecutionResult,
        detection_report: DetectionReport,
        source_code: str = "",
    ) -> IntegrityReport:
        report = IntegrityReport()
        report.duration_ms = exec_result.duration_ms

        # Program hash
        if source_code:
            report.program_hash = hashlib.sha256(source_code.encode()).hexdigest()[:32]

        # Build reasoning chain
        report.reasoning_chain = ChainBuilder.build(exec_result.trace)
        report.chain_valid = ChainBuilder.verify(report.reasoning_chain)

        # Gate certificates
        for gl in exec_result.gate_logs:
            report.gate_certificates.append(GateCertificate(
                gate_name=gl["gate"],
                branches=gl["branches"],
                collapse_strategy=gl["collapse"],
                branch_confidences=gl["branch_confidences"],
                result_confidence=gl["result_confidence"],
                result_value=gl["result_value"],
            ))

        # Assertion counts
        report.assertions_passed = exec_result.assertions_passed
        report.assertions_failed = exec_result.assertions_failed

        # Hallucination status
        report.hallucination_flags = detection_report.flags
        report.hallucination_clean = detection_report.clean

        # Verdict
        report.verdict = self._compute_verdict(report)

        return report

    @staticmethod
    def _compute_verdict(report: IntegrityReport) -> str:
        if report.assertions_failed > 0:
            return "FAIL — assertion failures"
        if not report.chain_valid:
            return "FAIL — reasoning chain corrupted"
        if not report.hallucination_clean:
            critical = [f for f in report.hallucination_flags if f.severity >= 0.8]
            if critical:
                return f"WARN — {len(critical)} critical hallucination flag(s)"
            return f"PASS_WITH_WARNINGS — {len(report.hallucination_flags)} flag(s)"
        return "PASS — all checks clean"
