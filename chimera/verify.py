"""Independent, offline verifier for ChimeraLang certificates.

This module is self-contained at import time: it imports ONLY the Python
standard library (plus a lazy, feature-detected `cryptography` import inside
the signature check). For v2 CIR certificates it *attempts* to import the
chimera parser and lowering lazily, inside the verification function, to
re-derive the graph from cir.program_source. If that import fails, the
graph-source link is reported NOT RE-DERIVED and the guarantee degrades
(see trust boundary below); the module never requires chimera at import.

Two formats are understood, and anything else fails closed:
  - chimeralang-cert/v1: VM-path certificates (IntegrityReport).
  - chimeralang-cert/v2: CIR certificates (chimera.cir.certify). The v2
    verifier recomputes the dominance predicate from the embedded graph
    structure and never trusts a stored dominance flag.

Trust boundary for v2, stated exactly:
  - When the chimera package is importable, the verifier re-lowers
    cir.program_source with the real parser and lowering, and compares
    the canonical serialized graph byte for byte to the embedded graph
    (node ids are deterministic creation-order ids assigned by
    CIRLowering). A mismatch is a failure. The verifier then recomputes
    the dominance predicate from the re-derived graph and requires the
    stored claim to match. In this state the certificate proves: the
    source is unmodified (program_hash), the graph is unmodified
    (graph_hash), the graph is exactly the lowering of the source, and
    the dominance claim is correct.
  - When the chimera package cannot be imported, the graph-source link
    is NOT RE-DERIVED: the verifier checks internal consistency only
    (hashes, dominance recomputed from the embedded graph), reports
    "graph-source link: NOT RE-DERIVED", and never reports valid for a
    certificate whose stored dominance claim is "enforced".

Guarantees, stated precisely:
  - certificate_hash binding = tamper-evidence. It binds every report field
    to one SHA-256 digest, so corruption or modification is caught WHEN the
    expected digest is known through a trusted channel (e.g. pinned when the
    certificate was produced, or compared against an out-of-band value). The
    digest travels inside the certificate, so a bare hash on its own does NOT
    resist an adversary who edits the report and recomputes the digest — use
    HMAC or Ed25519 for authentication against untrusted parties.
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
    # v2 graph-source link: "RE-DERIVED" when the embedded graph was
    # shown to match a fresh lowering of cir.program_source,
    # "NOT RE-DERIVED" when the chimera package could not be imported
    # (v1 certificates report "N/A").
    link_status: str = "N/A"


class CertificateVerifier:
    """Verifies a certificate dict produced by IntegrityReport.to_certificate
    (v1) or chimera.cir.certify.certify_cir (v2)."""

    #: Certificate formats this verifier understands. Anything else fails
    #: closed: an unknown version is never treated as valid.
    KNOWN_FORMATS = frozenset({"chimeralang-cert/v1", "chimeralang-cert/v2"})

    @staticmethod
    def verify(
        certificate: dict,
        *,
        hmac_key: bytes | None = None,
        pubkey_hex: str | None = None,
    ) -> VerificationResult:
        fmt = certificate.get("format") if isinstance(certificate, dict) else None
        if fmt not in CertificateVerifier.KNOWN_FORMATS:
            return VerificationResult(
                valid=False,
                failures=[f"version: unknown certificate format {fmt!r}; "
                          f"failing closed"],
                checks_run=1,
                verdict_recomputed="",
                signature_status="absent",
            )
        if fmt == "chimeralang-cert/v2":
            return CertificateVerifier._verify_v2(
                certificate, hmac_key=hmac_key, pubkey_hex=pubkey_hex)
        return CertificateVerifier._verify_v1(
            certificate, hmac_key=hmac_key, pubkey_hex=pubkey_hex)

    @staticmethod
    def _verify_v1(
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
        chain = report.get("chain")
        if not isinstance(chain, dict):
            failures.append("chain: report.chain is missing or not an object")
            chain = {}
        links = chain.get("links")
        if links is None:
            failures.append("chain: certificate report is missing full chain links")
        elif not isinstance(links, list):
            failures.append("chain: report.chain.links is not a list")
        elif not links:
            expected_empty = hashlib.sha256(b"empty").hexdigest()[:32]
            if chain.get("root_hash") != expected_empty:
                failures.append("chain: empty chain has incorrect root_hash")
        elif not all(isinstance(link, dict) for link in links):
            failures.append("chain: one or more chain links are not objects")
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
        if not isinstance(gates, list):
            failures.append("gate: report.gates is not a list")
            gates = []
        for i, gate in enumerate(gates):
            if not isinstance(gate, dict):
                failures.append(f"gate {i}: certificate entry is not an object")
                continue
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
        elif not isinstance(signature, dict):
            signature_status = "invalid"
            failures.append("signature: binding.signature is not an object")
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

    # ------------------------------------------------------------------
    # v2: CIR certificates (chimera.cir.certify.certify_cir)
    # ------------------------------------------------------------------

    @staticmethod
    def _rederive_graph(source: str) -> tuple:
        """Re-lower cir.program_source with the real parser and lowering.

        Returns (canonical_bytes, error). canonical_bytes is the
        canonical JSON encoding of the serialized re-derived graph, or
        None when the chimera package cannot be imported or the source
        does not parse/lower. The import is attempted lazily so this
        module stays usable (with a degraded guarantee) where chimera
        is unavailable. Node ids are deterministic (creation order in
        CIRLowering), so the bytes are directly comparable to the
        embedded graph's canonical encoding.
        """
        try:
            from chimera.lexer import Lexer
            from chimera.parser import Parser
            from chimera.cir.lower import CIRLowering
            from chimera.cir.certify import serialize_graph, _canonical_bytes
        except ImportError as e:
            return None, f"chimera package not importable ({e})"
        try:
            program = Parser(Lexer(source).tokenize()).parse()
        except Exception as e:
            return None, f"program_source does not parse ({e})"
        try:
            graph = CIRLowering().lower(program)
        except Exception as e:
            return None, f"program_source does not lower ({e})"
        return _canonical_bytes(serialize_graph(graph)), ""

    @staticmethod
    def _recompute_dominance(graph: dict) -> tuple[bool, bool, list[dict]]:
        """Independently recompute dominance from the embedded graph.

        Standalone reimplementation operating only on the serialized
        structure. Never trusts a stored flag or stored evidence.
        Returns (dominated, all_strict, evidence). all_strict is True
        iff every ValidationNode dominating a consumer on any path
        carries strict=True. The cir.strict_guard flag is ignored.
        """
        kinds = {n["id"]: n["kind"] for n in graph.get("nodes", [])}
        strict_of = {n["id"]: bool(n.get("strict", False))
                     for n in graph.get("nodes", [])}
        preds: dict[str, list[str]] = {}
        for e in graph.get("edges", []):
            preds.setdefault(e["target_id"], []).append(e["source_id"])
        consensus = {nid for nid, k in kinds.items()
                     if k == "ConsensusNode"}

        def paths_to(consumer_id: str) -> list[list[str]]:
            paths: list[list[str]] = []
            stack = [(consumer_id, [consumer_id], {consumer_id})]
            while stack:
                nid, path, seen = stack.pop()
                ps = [p for p in preds.get(nid, []) if p not in seen]
                if not ps:
                    paths.append(list(reversed(path)))
                    continue
                for p in ps:
                    stack.append((p, path + [p], seen | {p}))
            return paths

        def dominating_guards(path: list[str]) -> list[str]:
            last_cons = -1
            for i, nid in enumerate(path):
                if nid in consensus:
                    last_cons = i
            return [nid for i, nid in enumerate(path)
                    if kinds.get(nid) == "ValidationNode" and i > last_cons]

        def dominated(consumer_id: str) -> bool:
            paths = paths_to(consumer_id)
            if not paths:
                return False
            for path in paths:
                if not dominating_guards(path):
                    return False
            return True

        evidence: list[dict] = []
        all_ok = True
        all_guards: set[str] = set()
        consumers = (
            [n["id"] for n in graph.get("nodes", [])
             if n["kind"] == "EvolutionNode"]
            + [eid for eid in graph.get("emit_ids", []) if eid in kinds]
        )
        for cid in consumers:
            ok = dominated(cid)
            if ok:
                for path in paths_to(cid):
                    all_guards.update(dominating_guards(path))
            evidence.append({"consumer": cid, "dominated": ok})
            all_ok = all_ok and ok
        all_strict = all(strict_of[gid] for gid in all_guards)
        return all_ok, all_strict, evidence

    @staticmethod
    def _expected_dominance_claim(dominated: bool, all_strict: bool) -> str:
        if dominated and all_strict:
            return "enforced"
        if dominated:
            return "non-blocking"
        return "absent"

    @staticmethod
    def _verify_v2(
        certificate: dict,
        *,
        hmac_key: bytes | None = None,
        pubkey_hex: str | None = None,
    ) -> VerificationResult:
        failures: list[str] = []
        checks_run = 0
        signature_status = "absent"

        # --- Check 1: envelope ------------------------------------------
        checks_run += 1
        cir = certificate.get("cir") if isinstance(certificate, dict) else None
        binding = (certificate.get("binding")
                   if isinstance(certificate, dict) else None)
        if not isinstance(cir, dict):
            failures.append("format: missing or invalid 'cir' object")
        if not isinstance(binding, dict):
            failures.append("format: missing or invalid 'binding' object")
        if not isinstance(cir, dict) or not isinstance(binding, dict):
            return VerificationResult(
                valid=False, failures=failures, checks_run=checks_run,
                verdict_recomputed="", signature_status=signature_status,
                link_status="N/A")

        cir_bytes = _canonical_bytes(cir)

        # --- Check 2: certificate hash binding ----------------------------
        checks_run += 1
        if (hashlib.sha256(cir_bytes).hexdigest()
                != binding.get("certificate_hash")):
            failures.append(
                "binding: certificate_hash mismatch (cir section modified)")

        # --- Check 3: program source binding ------------------------------
        checks_run += 1
        source = cir.get("program_source", "")
        if (hashlib.sha256(source.encode("utf-8")).hexdigest()[:32]
                != cir.get("program_hash")):
            failures.append(
                "cir: program_hash mismatch (program_source modified)")

        # --- Check 4: graph binding ---------------------------------------
        checks_run += 1
        graph = cir.get("graph")
        if not isinstance(graph, dict):
            failures.append("cir: missing or invalid 'graph' object")
            graph = {"nodes": [], "edges": [], "emit_ids": []}
        if (hashlib.sha256(_canonical_bytes(graph)).hexdigest()[:32]
                != cir.get("graph_hash")):
            failures.append("cir: graph_hash mismatch (graph modified)")

        # --- Check 5: dominance recomputation ------------------------------
        # Recomputed from the embedded graph structure. The stored claim,
        # the stored evidence, and the cir.strict_guard run flag are
        # never trusted: the expected claim is derived from the
        # source-level strict flags on the re-derived graph alone.
        checks_run += 1
        dominated, all_strict, _evidence = (
            CertificateVerifier._recompute_dominance(graph))
        expected_claim = CertificateVerifier._expected_dominance_claim(
            dominated, all_strict)
        stored_claim = (cir.get("dominance") or {}).get("claim")
        if stored_claim != expected_claim:
            failures.append(
                f"dominance: stored claim {stored_claim!r} does not match "
                f"recomputed {expected_claim!r} (dominated={dominated}, "
                f"all_strict={all_strict})")

        # --- Check 5b: graph-source link ----------------------------------
        # Re-lower cir.program_source with the real parser and lowering
        # and compare the canonical serialized graph byte for byte with
        # the embedded graph. Node ids are deterministic (creation
        # order), so equality is exact. This defeats a producer that
        # splices in a graph from a different program and fixes up the
        # hashes.
        checks_run += 1
        link_status = "RE-DERIVED"
        rederived_bytes, re_err = CertificateVerifier._rederive_graph(source)
        if rederived_bytes is None:
            link_status = "NOT RE-DERIVED"
            failures.append(f"graph-source link: NOT RE-DERIVED ({re_err})")
            if stored_claim == "enforced":
                failures.append(
                    "graph-source link: NOT RE-DERIVED, so an 'enforced' "
                    "claim can never be reported valid")
        elif rederived_bytes != _canonical_bytes(graph):
            failures.append(
                "graph-source link: re-derived graph does not match the "
                "embedded graph (program_source and graph are inconsistent)")

        # --- Check 6: HMAC --------------------------------------------------
        if hmac_key is not None:
            checks_run += 1
            stored_hmac = binding.get("hmac")
            if not stored_hmac:
                failures.append("hmac: key supplied but certificate has no HMAC")
            elif not hmac.compare_digest(
                    hmac.new(hmac_key, cir_bytes,
                             hashlib.sha256).hexdigest(),
                    str(stored_hmac)):
                failures.append("hmac: authentication failed")

        # --- Check 7: signature ----------------------------------------------
        checks_run += 1
        signature = binding.get("signature")
        if signature is None:
            if pubkey_hex is not None:
                failures.append(
                    "signature: pubkey requested but certificate is unsigned")
        elif not isinstance(signature, dict):
            signature_status = "invalid"
            failures.append("signature: binding.signature is not an object")
        else:
            signature_status, sig_failures = _verify_signature(
                signature, cir_bytes, pubkey_hex)
            failures.extend(sig_failures)

        return VerificationResult(
            valid=(len(failures) == 0),
            failures=failures,
            checks_run=checks_run,
            verdict_recomputed="",
            signature_status=signature_status,
            link_status=link_status,
        )


def _recompute_verdict(report: dict) -> str:
    """Re-derive the verdict from report fields (mirrors _compute_verdict).

    Tolerates malformed/missing nested objects: a bad shape simply does not
    match the stored verdict, surfacing as a verdict failure rather than a crash.
    """
    assertions = report.get("assertions")
    if not isinstance(assertions, dict):
        assertions = {}
    if (assertions.get("failed") or 0) > 0:
        return f"FAIL {_EM} assertion failures"
    chain = report.get("chain")
    if not isinstance(chain, dict):
        chain = {}
    if not chain.get("valid", True):
        return f"FAIL {_EM} reasoning chain corrupted"
    hallucination = report.get("hallucination")
    if not isinstance(hallucination, dict):
        hallucination = {}
    if not hallucination.get("clean", True):
        flags_detail = hallucination.get("flags_detail")
        if not isinstance(flags_detail, list):
            flags_detail = []
        critical = [
            f for f in flags_detail
            if isinstance(f, dict) and f.get("severity", 0.0) >= 0.8
        ]
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
