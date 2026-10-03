"""Red tests for guard dominance (Stage 2).

These tests specify the intended behavior from
docs/design/guard-dominance.md and the six product decisions.
They FAIL against the current code (no dominance pass exists yet).
"""
import pytest

from chimera.lexer import Lexer
from chimera.parser import Parser
from chimera.cir.lower import CIRLowering, LoweringError


def parse_src(src):
    return Parser(Lexer(src).tokenize()).parse()


EMIT_NO_GUARD = """belief x := inquire {
  prompt: "Is the sky blue?",
  agents: [claude]
}

emit x
"""

EVOLVE_NO_GUARD = """belief x := inquire {
  prompt: "Is the sky blue?",
  agents: [claude]
}

evolve x until stable { max_iter: 3 }

emit x
"""

GUARD_AFTER_EVOLVE = """belief x := inquire {
  prompt: "Is the sky blue?",
  agents: [claude]
}

evolve x until stable { max_iter: 3 }

guard x against hallucination { max_risk: 0.2 }

emit x
"""

FANOUT_RESOLVE_NO_GUARD = """belief x := inquire {
  prompt: "Is the sky blue?",
  agents: [a, b]
}

resolve x with consensus { threshold: 0.8 }

evolve x until stable { max_iter: 3 }

emit x
"""

CANONICAL = """belief x := inquire {
  prompt: "Is the sky blue?",
  agents: [claude]
}

resolve x with consensus { threshold: 0.8 }

guard x against hallucination { max_risk: 0.2, strategy: both }

evolve x until stable { max_iter: 3 }

emit x
"""

SINGLE_GUARD_EMIT = """belief x := inquire {
  prompt: "Is the sky blue?",
  agents: [claude]
}

guard x against hallucination { max_risk: 0.2 }

emit x
"""

SINGLE_GUARD_MEAN = """belief x := inquire {
  prompt: "Is the sky blue?",
  agents: [claude]
}

guard x against hallucination { max_risk: 0.2, strategy: mean }

emit x
"""

GUARD_BEFORE_RESOLVE = """belief x := inquire {
  prompt: "Is the sky blue?",
  agents: [a, b]
}

guard x against hallucination { max_risk: 0.2 }

resolve x with consensus { threshold: 0.8 }

emit x
"""

DOUBLE_GUARD = """belief x := inquire {
  prompt: "Is the sky blue?",
  agents: [claude]
}

guard x against hallucination { max_risk: 0.2 }

guard x against hallucination { max_risk: 0.1 }

emit x
"""


# ------------------------------------------------------------------
# Must be rejected under require_dominance=True
# ------------------------------------------------------------------

def test_emit_without_guard_raises_under_require():
    with pytest.raises(LoweringError):
        CIRLowering(require_dominance=True).lower(parse_src(EMIT_NO_GUARD))


def test_evolve_without_guard_raises_under_require():
    with pytest.raises(LoweringError):
        CIRLowering(require_dominance=True).lower(parse_src(EVOLVE_NO_GUARD))


def test_guard_after_evolve_rejects_evolve_under_require():
    # The evolve is undominated even though a later guard exists.
    with pytest.raises(LoweringError) as exc_info:
        CIRLowering(require_dominance=True).lower(parse_src(GUARD_AFTER_EVOLVE))
    assert "evolve" in str(exc_info.value).lower()


def test_fanout_resolve_without_guard_raises_under_require():
    with pytest.raises(LoweringError):
        CIRLowering(require_dominance=True).lower(
            parse_src(FANOUT_RESOLVE_NO_GUARD))


def test_guard_before_resolve_does_not_dominate():
    # Decision 3: a guard before resolve is not accepted as dominating
    # the consensus belief.
    with pytest.raises(LoweringError):
        CIRLowering(require_dominance=True).lower(parse_src(GUARD_BEFORE_RESOLVE))


# ------------------------------------------------------------------
# Default mode warns instead of raising
# ------------------------------------------------------------------

def test_emit_without_guard_warns_by_default():
    lowering = CIRLowering()
    lowering.lower(parse_src(EMIT_NO_GUARD))
    assert any("dominance" in w.lower() for w in lowering.warnings)


def test_evolve_without_guard_warns_by_default():
    lowering = CIRLowering()
    lowering.lower(parse_src(EVOLVE_NO_GUARD))
    assert any("dominance" in w.lower() for w in lowering.warnings)


def test_canonical_pipeline_no_warning_by_default():
    lowering = CIRLowering()
    lowering.lower(parse_src(CANONICAL))
    assert not any("dominance" in w.lower() for w in lowering.warnings)


# ------------------------------------------------------------------
# Must pass
# ------------------------------------------------------------------

def test_canonical_pipeline_passes_under_require():
    # Must not raise.
    CIRLowering(require_dominance=True).lower(parse_src(CANONICAL))


def test_single_source_guard_emit_passes_under_require():
    CIRLowering(require_dominance=True).lower(parse_src(SINGLE_GUARD_EMIT))


def test_double_guard_passes_under_require():
    CIRLowering(require_dominance=True).lower(parse_src(DOUBLE_GUARD))


# ------------------------------------------------------------------
# Decision 2: uncalibrated mean/both guards under require_dominance
# ------------------------------------------------------------------

def test_uncalibrated_mean_guard_rejected_under_require():
    with pytest.raises(LoweringError) as exc_info:
        CIRLowering(require_dominance=True, calibrator=None).lower(
            parse_src(SINGLE_GUARD_MEAN))
    assert "calibrator" in str(exc_info.value).lower()


def test_calibrated_mean_guard_accepted_under_require():
    from chimera.cir.calibration import LogisticCalibrator
    cal = LogisticCalibrator(a=0.0, b=0.0, n=30, fit_date="2026-10-03")
    # Must not raise.
    CIRLowering(require_dominance=True, calibrator=cal).lower(
        parse_src(SINGLE_GUARD_MEAN))


def test_variance_only_guard_accepted_without_calibrator():
    src = """belief x := inquire {
      prompt: "Is the sky blue?",
      agents: [claude]
    }

    guard x against hallucination { max_risk: 0.2, strategy: variance }

    emit x
    """
    # Variance strategy does not pretend to be a probability: allowed
    # without a calibrator even under require_dominance.
    CIRLowering(require_dominance=True, calibrator=None).lower(parse_src(src))


# ------------------------------------------------------------------
# run_cir wiring
# ------------------------------------------------------------------

def test_run_cir_require_dominance_raises():
    from chimera.cir import run_cir
    with pytest.raises(LoweringError):
        run_cir(parse_src(EVOLVE_NO_GUARD), require_dominance=True)


def test_run_cir_default_warns_not_raises():
    from chimera.cir import run_cir
    result = run_cir(parse_src(EVOLVE_NO_GUARD))
    warnings = result.meta.get("lowering_warnings", [])
    assert any("dominance" in w.lower() for w in warnings)


# ------------------------------------------------------------------
# Certificate v2: score_source, strict_guard, dominance claim
# (Decision 2 second half, decision 4, decision 5.)
# These fail on missing API until Stage 4.
# ------------------------------------------------------------------

def _make_certed_run(strict_guard=False, calibrator=None):
    from chimera.cir import run_cir
    from chimera.cir.certify import certify_cir
    prog = parse_src(CANONICAL)
    result = run_cir(prog, strict_guard=strict_guard, calibrator=calibrator)
    lowering = CIRLowering()
    graph = lowering.lower(prog)
    src = CANONICAL
    return certify_cir(
        src, graph, result, strict_guard=strict_guard, calibrator=calibrator)


def test_certificate_carries_score_source_and_strict_guard():
    cert = _make_certed_run(strict_guard=True)
    assert cert["format"] == "chimeralang-cert/v2"
    cir = cert["cir"]
    assert cir["strict_guard"] is True
    assert cir["validations"], "expected validation entries"
    for v in cir["validations"]:
        assert v["score_source"] in ("calibrated", "uncalibrated")


def test_dominance_claim_enforced_only_with_strict_guard():
    # Decision 4: "dominance: enforced" only when dominance holds AND
    # strict_guard was on.
    cert_strict = _make_certed_run(strict_guard=True)
    assert cert_strict["cir"]["dominance"]["claim"] == "enforced"
    cert_lax = _make_certed_run(strict_guard=False)
    assert cert_lax["cir"]["dominance"]["claim"] == "non-blocking"


def test_dominance_claim_absent_when_undominated():
    from chimera.cir import run_cir
    from chimera.cir.certify import certify_cir
    prog = parse_src(EVOLVE_NO_GUARD)
    result = run_cir(prog)
    graph = CIRLowering().lower(prog)
    cert = certify_cir(EVOLVE_NO_GUARD, graph, result, strict_guard=True,
                       calibrator=None)
    assert cert["cir"]["dominance"]["claim"] == "absent"


# ------------------------------------------------------------------
# Verifier tamper tests (Stage 4). Each must be rejected.
# ------------------------------------------------------------------

def _valid_cert():
    return _make_certed_run(strict_guard=True)


def test_verifier_rejects_flipped_dominance_flag():
    from chimera.verify import CertificateVerifier
    cert = _valid_cert()
    # Flip the stored claim to something the graph does not support.
    cert["cir"]["dominance"]["claim"] = "absent"
    result = CertificateVerifier.verify(cert)
    assert not result.valid


def test_verifier_rejects_removed_guard_node():
    from chimera.verify import CertificateVerifier
    cert = _valid_cert()
    nodes = cert["cir"]["graph"]["nodes"]
    cert["cir"]["graph"]["nodes"] = [
        n for n in nodes if n["kind"] != "ValidationNode"]
    result = CertificateVerifier.verify(cert)
    assert not result.valid


def test_verifier_rejects_rewired_edge():
    from chimera.verify import CertificateVerifier
    cert = _valid_cert()
    edges = cert["cir"]["graph"]["edges"]
    # Rewire the first edge to a nonsense target.
    edges[0] = dict(edges[0], target_id="deadbeef")
    result = CertificateVerifier.verify(cert)
    assert not result.valid


def test_verifier_rejects_changed_source():
    from chimera.verify import CertificateVerifier
    cert = _valid_cert()
    cert["cir"]["program_source"] += "\nemit x\n"
    result = CertificateVerifier.verify(cert)
    assert not result.valid


def test_verifier_fails_closed_on_unknown_version():
    from chimera.verify import CertificateVerifier
    cert = _valid_cert()
    cert["format"] = "chimeralang-cert/v99"
    result = CertificateVerifier.verify(cert)
    assert not result.valid
    assert any("version" in f.lower() for f in result.failures)


def test_verifier_still_accepts_v1():
    # v1 VM-path certificates must keep verifying unchanged.
    from chimera.integrity import IntegrityEngine
    from chimera.verify import CertificateVerifier
    from chimera.vm import ExecutionResult
    from chimera.detect import DetectionReport
    engine = IntegrityEngine()
    report = engine.certify(ExecutionResult(), DetectionReport(),
                            source_code="test")
    cert = report.to_certificate()
    assert cert["format"] == "chimeralang-cert/v1"
    result = CertificateVerifier.verify(cert)
    assert result.valid, result.failures
