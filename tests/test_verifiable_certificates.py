"""Tests for the verifiable-proof layer: portable certificates + offline verifier."""

from __future__ import annotations

import copy
import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest

from chimera.detect import HallucinationDetector
from chimera.integrity import IntegrityEngine
from chimera.lexer import Lexer
from chimera.parser import Parser
from chimera.verify import CertificateVerifier
from chimera.vm import ChimeraVM

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = REPO_ROOT / "examples"
BELIEF = EXAMPLES / "belief_reasoning.chimera"
QUANTUM = EXAMPLES / "quantum_reasoning.chimera"  # has at least one gate


def _build_report(source_path: Path):
    source = source_path.read_text(encoding="utf-8")
    program = Parser(Lexer(source, str(source_path)).tokenize()).parse()
    vm = ChimeraVM()
    exec_result = vm.execute(program)
    detection = HallucinationDetector().full_scan(
        exec_result.gate_logs, exec_result.emitted
    )
    return IntegrityEngine().certify(exec_result, detection, source)


def _certificate(source_path: Path, **kwargs):
    return _build_report(source_path).to_certificate(**kwargs)


# ---------------------------------------------------------------------------
# 1. Round trip
# ---------------------------------------------------------------------------

def test_round_trip_valid():
    cert = _certificate(BELIEF)
    assert cert["format"] == "chimeralang-cert/v1"
    result = CertificateVerifier.verify(cert)
    assert result.valid is True
    assert result.failures == []
    assert result.signature_status == "absent"


def test_quantum_round_trip_has_gate():
    cert = _certificate(QUANTUM)
    assert len(cert["report"]["gates"]) >= 1
    result = CertificateVerifier.verify(cert)
    assert result.valid is True, result.failures


# ---------------------------------------------------------------------------
# 2-4. Tamper cases
# ---------------------------------------------------------------------------

def test_tampered_trace():
    cert = _certificate(BELIEF)
    assert cert["report"]["chain"]["links"], "need at least one chain link"
    cert["report"]["chain"]["links"][0]["entry"] += " (tampered)"
    result = CertificateVerifier.verify(cert)
    assert result.valid is False
    assert any("chain:" in f for f in result.failures)
    assert any("binding:" in f for f in result.failures)


def test_tampered_gate():
    cert = _certificate(QUANTUM)
    gate = cert["report"]["gates"][0]
    gate["result_confidence"] = (gate["result_confidence"] or 0.0) + 0.123
    result = CertificateVerifier.verify(cert)
    assert result.valid is False
    assert any("gate" in f and "hash mismatch" in f for f in result.failures)
    assert any("binding:" in f for f in result.failures)


def test_tampered_verdict():
    cert = _certificate(BELIEF)
    cert["report"]["verdict"] = "PASS — all checks clean"  # forced, likely wrong
    # Make sure we actually changed it to something inconsistent.
    if cert["report"]["verdict"] == _build_report(BELIEF).verdict:
        cert["report"]["verdict"] = "TOTALLY BOGUS VERDICT"
    result = CertificateVerifier.verify(cert)
    assert result.valid is False
    assert any("verdict:" in f for f in result.failures)
    assert any("binding:" in f for f in result.failures)


# ---------------------------------------------------------------------------
# 5. HMAC
# ---------------------------------------------------------------------------

def test_hmac_happy_and_wrong_key():
    key = b"super-secret-shared-key"
    cert = _certificate(BELIEF, hmac_key=key)
    assert cert["binding"]["hmac"] is not None

    ok = CertificateVerifier.verify(cert, hmac_key=key)
    assert ok.valid is True, ok.failures

    bad = CertificateVerifier.verify(cert, hmac_key=b"the-wrong-key")
    assert bad.valid is False
    assert any("hmac:" in f for f in bad.failures)


def test_hmac_key_supplied_but_absent_in_cert():
    cert = _certificate(BELIEF)  # no hmac in binding
    result = CertificateVerifier.verify(cert, hmac_key=b"any-key")
    assert result.valid is False
    assert any("hmac:" in f for f in result.failures)


# ---------------------------------------------------------------------------
# 6-9. Ed25519 signatures
# ---------------------------------------------------------------------------

def _make_signed_cert(source_path: Path):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ed25519

    private_key = ed25519.Ed25519PrivateKey.generate()
    pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    cert = _certificate(source_path, sign_key=pem)
    pubkey_hex = cert["binding"]["signature"]["pubkey"]
    return cert, pubkey_hex


def test_ed25519_happy_path():
    pytest.importorskip("cryptography")
    cert, pubkey_hex = _make_signed_cert(BELIEF)
    result = CertificateVerifier.verify(cert, pubkey_hex=pubkey_hex)
    assert result.valid is True, result.failures
    assert result.signature_status == "valid"


def test_ed25519_self_check_unverified():
    pytest.importorskip("cryptography")
    cert, _ = _make_signed_cert(BELIEF)
    result = CertificateVerifier.verify(cert)  # no pubkey -> trust-on-first-use
    assert result.valid is True, result.failures
    assert result.signature_status == "unverified"


def test_ed25519_tamper_after_sign():
    pytest.importorskip("cryptography")
    cert, pubkey_hex = _make_signed_cert(BELIEF)
    cert["report"]["chain"]["links"][0]["entry"] += " (tampered)"
    result = CertificateVerifier.verify(cert, pubkey_hex=pubkey_hex)
    assert result.valid is False
    assert result.signature_status == "invalid"


def test_ed25519_wrong_pubkey():
    pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ed25519

    cert, _ = _make_signed_cert(BELIEF)
    other_pub = ed25519.Ed25519PrivateKey.generate().public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    ).hex()
    result = CertificateVerifier.verify(cert, pubkey_hex=other_pub)
    assert result.valid is False
    assert any("signature:" in f for f in result.failures)
    assert result.signature_status == "invalid"


def test_signature_unavailable_degrades(monkeypatch):
    pytest.importorskip("cryptography")
    cert, pubkey_hex = _make_signed_cert(BELIEF)

    # Simulate cryptography being absent for the lazy import in verify.py.
    monkeypatch.setitem(sys.modules, "cryptography", None)
    monkeypatch.setitem(sys.modules, "cryptography.exceptions", None)
    monkeypatch.setitem(
        sys.modules, "cryptography.hazmat.primitives.asymmetric", None
    )

    # Without a demanded pubkey: degrade, do not fail.
    relaxed = CertificateVerifier.verify(cert)
    assert relaxed.signature_status == "unavailable"
    assert not any("signature:" in f for f in relaxed.failures)
    assert relaxed.valid is True, relaxed.failures

    # With a demanded pubkey: caller wanted a real check, so it must fail.
    strict = CertificateVerifier.verify(cert, pubkey_hex=pubkey_hex)
    assert strict.signature_status == "unavailable"
    assert strict.valid is False
    assert any("signature:" in f for f in strict.failures)


def test_pubkey_requested_but_unsigned():
    cert = _certificate(BELIEF)  # unsigned
    result = CertificateVerifier.verify(cert, pubkey_hex="00" * 32)
    assert result.valid is False
    assert result.signature_status == "absent"
    assert any("signature:" in f for f in result.failures)


# ---------------------------------------------------------------------------
# 10. Independence guard
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c["report"].__setitem__("chain", "not-an-object"),
        lambda c: c["report"]["chain"].__setitem__("links", "not-a-list"),
        lambda c: c["report"].__setitem__("gates", "not-a-list"),
        lambda c: c["report"].__setitem__("assertions", "not-an-object"),
        lambda c: c["report"].__setitem__("hallucination", 12345),
        lambda c: c["binding"].__setitem__("signature", "not-an-object"),
    ],
)
def test_malformed_certificate_fails_without_crashing(mutate):
    """Malformed/malicious nested types must yield a failed result, never a traceback."""
    cert = _certificate(BELIEF)
    mutate(cert)
    result = CertificateVerifier.verify(cert)  # must not raise
    assert result.valid is False
    assert result.failures


def test_verifier_is_independent():
    import chimera.verify

    src = inspect.getsource(chimera.verify)
    for banned in (
        "chimera.vm",
        "chimera.parser",
        "chimera.lexer",
        "chimera.detect",
        "chimera.integrity",
    ):
        assert banned not in src, f"verify.py must not reference {banned}"


def test_to_dict_default_unchanged():
    """Default to_dict must not gain new keys (back-compat with cmd_prove/tests)."""
    report = _build_report(BELIEF)
    d = report.to_dict()
    assert "links" not in d["chain"]
    assert "flags_detail" not in d["hallucination"]
    full = report.to_dict(full=True)
    assert "links" in full["chain"]
    assert "flags_detail" in full["hallucination"]


# ---------------------------------------------------------------------------
# 11. CLI smoke test
# ---------------------------------------------------------------------------

def _run_cli(args, **kwargs):
    return subprocess.run(
        [sys.executable, "-m", "chimera.cli", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=kwargs.pop("timeout", 30),
        **kwargs,
    )


def test_cli_prove_then_verify(tmp_path):
    cert_path = tmp_path / "cert.json"
    prove = _run_cli(["prove", str(BELIEF), f"--out={cert_path}"])
    assert prove.returncode == 0, prove.stderr
    assert cert_path.exists()

    verify = _run_cli(["verify", str(cert_path)])
    assert verify.returncode == 0, verify.stdout + verify.stderr
    assert "VERIFIED" in verify.stdout

    # Corrupt one byte of the report and confirm verification fails.
    data = json.loads(cert_path.read_text(encoding="utf-8"))
    data["report"]["program_hash"] = "0" * len(data["report"]["program_hash"])
    cert_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

    verify_bad = _run_cli(["verify", str(cert_path)])
    assert verify_bad.returncode == 1
    assert "VERIFICATION FAILED" in verify_bad.stdout
