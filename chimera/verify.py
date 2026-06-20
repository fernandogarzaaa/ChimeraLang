"""Independent, offline verifier for ChimeraLang certificates.

This module is deliberately self-contained: it imports ONLY the Python
standard library (plus a lazy, feature-detected `cryptography` import inside
the signature check). It does not import any part of the ChimeraLang
execution path (vm/parser/lexer/detect/integrity); all recomputation logic
(canonical encoding, chain hashing, gate hashing, verdict) is re-derived
here from the certificate alone. A test enforces this independence.

Guarantees, stated precisely:
  - certificate_hash binding = tamper-evidence (any edit is detected).
  - HMAC = authentication via a shared secret.
  - Ed25519 signature = asymmetric, third-party-verifiable signature.
  - Verifying against the certificate's *embedded* public key is a
    trust-on-first-use self-check, NOT proof of authorship.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass, field

# Verdict strings use an em dash (U+2014); these must match the producer
# byte-for-byte for the verdict-consistency check to succeed.
_EM = "—"


def _canonical_bytes(obj: object) -> bytes:
    """Byte-identical twin of the producer's canonical encoder."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


@dataclass
class VerificationResult:
    valid: bool
    failures: list[str] = field(default_factory=list)
    checks_run: int = 0
    verdict_recomputed: str = ""
    signature_status: str = "absent"  # valid|invalid|absent|unavailable|unverified


class CertificateVerifier:
    """Verifies a certificate dict produced by IntegrityReport.to_certificate."""

    @staticmethod
    def verify(
        certificate: dict,
        *,
        hmac_key: bytes | None = None,
        pubkey_hex: str | None = None,
    ) -> VerificationResult:
        failures: list[str] = []
        checks_run = 0
        verdict_recomputed = ""
        signature_status = "absent"

        # --- Check 1: format envelope -------------------------------------
        checks_run += 1
        report = certificate.get("report") if isinstance(certificate, dict) else None
        binding = certificate.get("binding") if isinstance(certificate, dict) else None
        fmt = certificate.get("format") if isinstance(certificate, dict) else None
        if fmt != "chimeralang-cert/v1":
            failures.append(f"format: expected 'chimeralang-cert/v1', got {fmt!r}")
        if not isinstance(report, dict):
            failures.append("format: missing or invalid 'report' object")
        if not isinstance(binding, dict):
            failures.append("format: missing or invalid 'binding' object")

        # Without report/binding nothing else can be checked meaningfully.
        if not isinstance(report, dict) or not isinstance(binding, dict):
            return VerificationResult(
                valid=False,
                failures=failures,
                checks_run=checks_run,
                verdict_recomputed=verdict_recomputed,
                signature_status=signature_status,
            )

        report_bytes = _canonical_bytes(report)

        # --- Check 2: certificate hash binding (tamper-evidence) ----------
        checks_run += 1
        recomputed_hash = hashlib.sha256(report_bytes).hexdigest()
        stored_hash = binding.get("certificate_hash")
        if recomputed_hash != stored_hash:
            failures.append(
                "binding: certificate_hash mismatch (report was modified) "
                f"expected {recomputed_hash}, stored {stored_hash}"
            )

        # --- Check 3: HMAC authentication (only when a key is supplied) ----
        if hmac_key is not None:
            checks_run += 1
            stored_hmac = binding.get("hmac")
            if not stored_hmac:
                failures.append("hmac: key supplied but certificate has no HMAC")
            else:
                recomputed_hmac = hmac.new(
                    hmac_key, report_bytes, hashlib.sha256
                ).hexdigest()
                if not hmac.compare_digest(recomputed_hmac, str(stored_hmac)):
                    failures.append("hmac: authentication failed (wrong key or tampered)")

        # --- Check 4: chain integrity -------------------------------------
        checks_run += 1
        chain = report.get("chain", {})
        links = chain.get("links")
        if links is None:
            failures.append("chain: certificate report is missing full chain links")
        elif not links:
            expected_empty = hashlib.sha256(b"empty").hexdigest()[:32]
            if chain.get("root_hash") != expected_empty:
                failures.append("chain: empty chain has incorrect root_hash")
        else:
            if links[0].get("prev_hash") != "genesis":
                failures.append("chain: first link prev_hash is not 'genesis'")
            prev_hash = None
            for i, link in enumerate(links):
                entry = link.get("entry", "")
                stored_link_hash = link.get("hash")
                link_prev = link.get("prev_hash")
                recomputed_link = hashlib.sha256(
                    f"{link_prev}:{entry}".encode()
                ).hexdigest()[:32]
                if recomputed_link != stored_link_hash:
                    failures.append(
                        f"chain: link {i} hash mismatch (entry tampered)"
                    )
                if i > 0 and link_prev != prev_hash:
                    failures.append(
                        f"chain: link {i} prev_hash does not match previous link"
                    )
                prev_hash = stored_link_hash
            if links[-1].get("hash") != chain.get("root_hash"):
                failures.append("chain: root_hash does not match last link hash")

        # --- Check 5: gate certificates -----------------------------------
        checks_run += 1
        gates = report.get("gates", [])
        for i, gate in enumerate(gates):
            if "branch_confidences" not in gate:
                failures.append(f"gate {i}: certificate missing branch_confidences")
                continue
            # Mirror GateCertificate.__post_init__ key names exactly.
            data = json.dumps(
                {
                    "gate": gate.get("name"),
                    "branches": gate.get("branches"),
                    "collapse": gate.get("collapse"),
                    "confs": gate.get("branch_confidences"),
                    "result_conf": gate.get("result_confidence"),
                },
                sort_keys=True,
            ).encode()
            recomputed_gate = hashlib.sha256(data).hexdigest()[:32]
            if recomputed_gate != gate.get("hash"):
                failures.append(
                    f"gate {i} ('{gate.get('name')}'): hash mismatch (gate tampered)"
                )

        # --- Check 6: verdict consistency ---------------------------------
        checks_run += 1
        verdict_recomputed = _recompute_verdict(report)
        stored_verdict = report.get("verdict")
        if verdict_recomputed != stored_verdict:
            failures.append(
                "verdict: recomputed verdict does not match stored verdict "
                f"(recomputed {verdict_recomputed!r}, stored {stored_verdict!r})"
            )

        # --- Check 7: signature -------------------------------------------
        checks_run += 1
        signature = binding.get("signature")
        if signature is None:
            if pubkey_hex is not None:
                signature_status = "absent"
                failures.append("signature: pubkey requested but certificate is unsigned")
            else:
                signature_status = "absent"
        else:
            signature_status, sig_failures = _verify_signature(
                signature, report_bytes, pubkey_hex
            )
            failures.extend(sig_failures)

        return VerificationResult(
            valid=(len(failures) == 0),
            failures=failures,
            checks_run=checks_run,
            verdict_recomputed=verdict_recomputed,
            signature_status=signature_status,
        )


def _recompute_verdict(report: dict) -> str:
    """Re-derive the verdict from report fields (mirrors _compute_verdict)."""
    assertions = report.get("assertions", {})
    if assertions.get("failed", 0) > 0:
        return f"FAIL {_EM} assertion failures"
    if not report.get("chain", {}).get("valid", True):
        return f"FAIL {_EM} reasoning chain corrupted"
    hallucination = report.get("hallucination", {})
    if not hallucination.get("clean", True):
        flags_detail = hallucination.get("flags_detail", [])
        critical = [f for f in flags_detail if f.get("severity", 0.0) >= 0.8]
        if critical:
            return f"WARN {_EM} {len(critical)} critical hallucination flag(s)"
        return f"PASS_WITH_WARNINGS {_EM} {hallucination.get('flags', 0)} flag(s)"
    return f"PASS {_EM} all checks clean"


def _verify_signature(
    signature: dict,
    message: bytes,
    pubkey_hex: str | None,
) -> tuple[str, list[str]]:
    """Verify an Ed25519 signature, degrading gracefully if crypto is absent."""
    failures: list[str] = []

    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric import ed25519
    except ImportError:
        # cryptography unavailable: only a failure if the caller demanded a check.
        if pubkey_hex is not None:
            failures.append(
                "signature: 'cryptography' not installed; cannot verify as requested"
            )
        return "unavailable", failures

    cert_pubkey_hex = signature.get("pubkey", "")
    sig_hex = signature.get("sig", "")

    try:
        sig_bytes = bytes.fromhex(sig_hex)
    except ValueError:
        failures.append("signature: 'sig' is not valid hex")
        return "invalid", failures

    if pubkey_hex is not None:
        # Third-party check against a key the caller already trusts.
        if cert_pubkey_hex != pubkey_hex:
            failures.append(
                "signature: embedded pubkey does not match the provided --pubkey"
            )
        try:
            verify_key = bytes.fromhex(pubkey_hex)
        except ValueError:
            failures.append("signature: provided pubkey is not valid hex")
            return "invalid", failures
        status_target = "valid"
    else:
        # Self-consistency only (trust-on-first-use, not authorship proof).
        try:
            verify_key = bytes.fromhex(cert_pubkey_hex)
        except ValueError:
            failures.append("signature: embedded pubkey is not valid hex")
            return "invalid", failures
        status_target = "unverified"

    try:
        public_key = ed25519.Ed25519PublicKey.from_public_bytes(verify_key)
        public_key.verify(sig_bytes, message)
    except InvalidSignature:
        failures.append("signature: cryptographic verification failed")
        return "invalid", failures
    except Exception as exc:  # malformed key bytes, etc.
        failures.append(f"signature: verification error ({exc})")
        return "invalid", failures

    # Signature validated. If the provided pubkey differed, that failure was
    # already recorded above; reflect it in the status.
    if pubkey_hex is not None and cert_pubkey_hex != pubkey_hex:
        return "invalid", failures
    return status_target, failures
