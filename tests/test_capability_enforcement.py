"""Tests for static capability enforcement.

Adapted to the real ChimeraLang grammar (which the prompt's examples predate):

  * Only ``FnDecl`` carries ``allow``/``forbidden`` constraints; gates/reason/
    goals do not. So constraint *enforcement* is demonstrated on fns.
  * Agent inquiries (``InquireExpr``) are top-level only (inside ``belief``) and
    cannot appear in a fn body, so the network/model capability is exercised via
    a top-level inquiry (and attested in the certificate).
  * ``print`` is the only side-effecting builtin -> IO capability; it is the
    reachable effect used to demonstrate fn-level forbidden/allow enforcement.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from chimera.lexer import Lexer
from chimera.parser import Parser
from chimera.type_checker import TypeChecker
from chimera import capabilities as caps

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = REPO_ROOT / "examples"
BELIEF = EXAMPLES / "belief_reasoning.chimera"
QUANTUM = EXAMPLES / "quantum_reasoning.chimera"


def _check_src(src: str):
    program = Parser(Lexer(src, "<test>").tokenize()).parse()
    return TypeChecker().check(program)


def _run_cli(args, **kwargs):
    return subprocess.run(
        [sys.executable, "-m", "chimera.cli", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=kwargs.pop("timeout", 30),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 1. Forbidden violation
# ---------------------------------------------------------------------------

FORBIDDEN_IO_SRC = '''\
fn log_it(msg: Text) -> Void
  forbidden:
    "io"
  print(msg)
  return
end

val x: Text = "hi"
'''


def test_forbidden_violation_reported():
    result = _check_src(FORBIDDEN_IO_SRC)
    assert result.ok is False
    assert any("forbidden io" in e and "log_it" in e for e in result.errors)
    assert result.capability_errors  # tracked separately for the escape hatch


def test_forbidden_violation_cli(tmp_path):
    p = tmp_path / "f.chimera"
    p.write_text(FORBIDDEN_IO_SRC, encoding="utf-8")
    assert _run_cli(["check", str(p)]).returncode == 1
    run = _run_cli(["run", str(p)])
    assert run.returncode == 1
    assert "refusing to execute" in run.stderr


# ---------------------------------------------------------------------------
# 2. Forbidden clean
# ---------------------------------------------------------------------------

FORBIDDEN_CLEAN_SRC = '''\
fn pure_add(a: Int, b: Int) -> Int
  forbidden:
    "io"
    "network"
  val s: Int = a + b
  return s
end

val x: Int = 1
'''


def test_forbidden_clean_passes():
    result = _check_src(FORBIDDEN_CLEAN_SRC)
    assert result.ok is True
    assert result.errors == []


def test_forbidden_clean_runs_cli(tmp_path):
    p = tmp_path / "clean.chimera"
    p.write_text(FORBIDDEN_CLEAN_SRC, encoding="utf-8")
    assert _run_cli(["check", str(p)]).returncode == 0
    assert _run_cli(["run", str(p)]).returncode == 0


# ---------------------------------------------------------------------------
# 3. Allow whitelist hit
# ---------------------------------------------------------------------------
# Semantics: when an `allow` clause is present it is a whitelist. A capability
# the body uses that is NOT in the whitelist is an error. Here the fn allows
# only `model` but uses `io` (via print), so it must fail.

ALLOW_HIT_SRC = '''\
fn restricted(msg: Text) -> Void
  allow:
    "model"
  print(msg)
  return
end

val x: Text = "hi"
'''


def test_allow_whitelist_hit():
    result = _check_src(ALLOW_HIT_SRC)
    assert result.ok is False
    assert any("not in its 'allow' set" in e and "io" in e for e in result.errors)


def test_allow_whitelist_satisfied():
    # Allowing exactly the used capability passes.
    src = ALLOW_HIT_SRC.replace('"model"', '"io"')
    result = _check_src(src)
    assert result.ok is True, result.errors


def test_freeform_allow_is_not_an_empty_whitelist():
    # An `allow` clause with only non-canonical strings is a semantic annotation,
    # not a whitelist — it must not reject real capability use (here: io).
    src = '''\
fn annotated(msg: Text) -> Void
  allow:
    "external tool invocation"
  print(msg)
  return
end

val x: Text = "hi"
'''
    result = _check_src(src)
    assert result.ok is True, result.errors
    assert result.capability_errors == []


# ---------------------------------------------------------------------------
# 4. Transitivity
# ---------------------------------------------------------------------------

TRANSITIVE_SRC = '''\
fn helper(msg: Text) -> Void
  print(msg)
  return
end

fn caller(msg: Text) -> Void
  forbidden:
    "io"
  helper(msg)
  return
end

val x: Text = "hi"
'''


def test_transitive_violation_names_callee():
    result = _check_src(TRANSITIVE_SRC)
    assert result.ok is False
    offending = [e for e in result.errors if "caller" in e]
    assert offending
    assert any("helper" in e for e in offending)


# ---------------------------------------------------------------------------
# 5. Recursion safety
# ---------------------------------------------------------------------------

RECURSION_SRC = '''\
fn ping(n: Int) -> Void
  pong(n)
  return
end

fn pong(n: Int) -> Void
  ping(n)
  return
end

val x: Int = 1
'''


def test_mutual_recursion_terminates():
    # The fixed-point propagation must terminate on cyclic call graphs.
    result = _check_src(RECURSION_SRC)
    assert result.ok is True
    assert result.errors == []


def test_self_recursion_with_capability():
    src = '''\
fn loop_print(msg: Text) -> Void
  forbidden:
    "io"
  print(msg)
  loop_print(msg)
  return
end

val x: Text = "hi"
'''
    result = _check_src(src)
    assert result.ok is False
    assert any("loop_print" in e and "io" in e for e in result.errors)


# ---------------------------------------------------------------------------
# 6. Escape hatch
# ---------------------------------------------------------------------------

def test_escape_hatch_downgrades_capability_only(tmp_path):
    p = tmp_path / "viol.chimera"
    p.write_text(FORBIDDEN_IO_SRC, encoding="utf-8")
    run = _run_cli(["run", str(p), "--no-capability-check"])
    assert run.returncode == 0
    assert "capability check bypassed" in run.stderr


def test_escape_hatch_does_not_bypass_real_type_errors(tmp_path):
    # A non-capability type error (Explore -> Confident outside a gate) must
    # still block, even with --no-capability-check.
    src = '''\
val c: Confident<Text> = Explore("unverified")
'''
    result = _check_src(src)
    assert result.ok is False
    # This error is NOT a capability error.
    assert result.errors
    assert not result.capability_errors

    p = tmp_path / "typeerr.chimera"
    p.write_text(src, encoding="utf-8")
    run = _run_cli(["run", str(p), "--no-capability-check"])
    assert run.returncode == 1


# ---------------------------------------------------------------------------
# 7. Certificate attestation
# ---------------------------------------------------------------------------

def test_certificate_carries_capability_attestation(tmp_path):
    from chimera.verify import CertificateVerifier

    cert_path = tmp_path / "cert.json"
    prove = _run_cli(["prove", str(BELIEF), f"--out={cert_path}"])
    assert prove.returncode == 0, prove.stderr

    cert = json.loads(cert_path.read_text(encoding="utf-8"))
    cap_block = cert["report"]["capabilities"]
    assert cap_block["statically_checked"] is True
    # belief_reasoning performs a top-level inquiry -> model + network.
    used = cap_block["used"]
    assert any(
        set(v) >= {caps.MODEL, caps.NETWORK} for v in used.values()
    ), used

    # The existing hash binding covers the new field with no verifier change.
    result = CertificateVerifier.verify(cert)
    assert result.valid is True, result.failures


def test_prove_refuses_violating_program(tmp_path):
    # prove must not become a bypass: a program that fails the capability check
    # is refused (no certificate written), and never executed.
    p = tmp_path / "viol.chimera"
    p.write_text(FORBIDDEN_IO_SRC, encoding="utf-8")
    cert_path = tmp_path / "cert.json"
    prove = _run_cli(["prove", str(p), f"--out={cert_path}"])
    assert prove.returncode == 1
    assert "refusing to prove" in prove.stderr
    assert not cert_path.exists()


def test_capability_attestation_in_full_report_only():
    # Default to_dict stays clean; full mode carries the attestation.
    from chimera.integrity import IntegrityReport

    report = IntegrityReport(capabilities={"statically_checked": True,
                                           "declared": {}, "used": {}})
    assert "capabilities" not in report.to_dict()
    assert "capabilities" in report.to_dict(full=True)


# ---------------------------------------------------------------------------
# 8. No regression on existing examples
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("example", [BELIEF, QUANTUM])
def test_existing_examples_still_check_and_run(example, tmp_path):
    result = _check_src(example.read_text(encoding="utf-8"))
    assert result.ok is True, result.errors
    assert _run_cli(["run", str(example)]).returncode == 0


def test_quantum_freeform_constraints_not_flagged():
    # quantum_reasoning uses free-form allow/must strings (e.g. "multiple
    # interpretive frameworks") that are not canonical capabilities; they must
    # not be treated as capability constraints.
    result = _check_src(QUANTUM.read_text(encoding="utf-8"))
    assert result.capability_errors == []
