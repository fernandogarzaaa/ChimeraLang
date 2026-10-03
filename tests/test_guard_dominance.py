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


def _test_calibrator():
    from chimera.cir.calibration import LogisticCalibrator
    return LogisticCalibrator(
        a=0.0, b=0.0, n=30, dataset_hash="test", fit_date="2026-10-03")


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
    # (Calibrator supplied to isolate the dominance rule from the
    # decision-2 calibrator rule.)
    with pytest.raises(LoweringError) as exc_info:
        CIRLowering(require_dominance=True,
                    calibrator=_test_calibrator()).lower(
                        parse_src(GUARD_AFTER_EVOLVE))
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
    # Must not raise (calibrator supplied per decision 2).
    CIRLowering(require_dominance=True,
                calibrator=_test_calibrator()).lower(parse_src(CANONICAL))


def test_single_source_guard_emit_passes_under_require():
    CIRLowering(require_dominance=True,
                calibrator=_test_calibrator()).lower(
                    parse_src(SINGLE_GUARD_EMIT))


def test_double_guard_passes_under_require():
    CIRLowering(require_dominance=True,
                calibrator=_test_calibrator()).lower(parse_src(DOUBLE_GUARD))


# ------------------------------------------------------------------
# Decision 2: uncalibrated mean/both guards under require_dominance
# ------------------------------------------------------------------

def test_uncalibrated_mean_guard_rejected_under_require():
    with pytest.raises(LoweringError) as exc_info:
        CIRLowering(require_dominance=True, calibrator=None).lower(
            parse_src(SINGLE_GUARD_MEAN))
    assert "calibrator" in str(exc_info.value).lower()


def test_calibrated_mean_guard_accepted_under_require():
    cal = _test_calibrator()
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
    from chimera.cir.executor import InquiryResponse

    def adapter(prompt, agents):
        return InquiryResponse(confidence=0.95, answer="yes")

    prog = parse_src(CANONICAL)
    result = run_cir(prog, strict_guard=strict_guard, calibrator=calibrator,
                     inquiry_adapter=adapter)
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
    # Option 2 (supersedes Decision 4): "dominance: enforced" only when
    # dominance holds AND every dominating guard is source-level strict.
    # The global strict_guard run flag does not affect the claim.
    # CANONICAL uses a non-strict guard, so the claim is non-blocking
    # regardless of the flag.
    cert_strict = _make_certed_run(strict_guard=True)
    assert cert_strict["cir"]["dominance"]["claim"] == "non-blocking"
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


def test_verifier_rejects_forged_graph_source_mismatch():
    """Forgery (trust-gap red test).

    Take certify_cir output for an UNGUARDED program, splice in the
    dominated graph from examples/guarded_pipeline.chimera, and fix up
    graph_hash, dominance, and certificate_hash so every
    internal-consistency check passes. The verifier must still reject
    it, by re-deriving the graph from cir.program_source.
    """
    import hashlib
    from chimera.cir import run_cir
    from chimera.cir.certify import (
        _canonical_bytes,
        certify_cir,
        check_dominance,
        serialize_graph,
    )
    from chimera.verify import CertificateVerifier

    # Honest cert for an UNGUARDED program (dominance: absent).
    prog = parse_src(EMIT_NO_GUARD)
    graph = CIRLowering().lower(prog)
    result = run_cir(prog)
    cert = certify_cir(EMIT_NO_GUARD, graph, result,
                       strict_guard=False, calibrator=None)
    assert cert["cir"]["dominance"]["claim"] == "absent"

    # Forge: splice in the dominated graph from guarded_pipeline.
    guarded_src = open("examples/guarded_pipeline.chimera",
                       encoding="utf-8").read()
    guarded_graph = CIRLowering().lower(parse_src(guarded_src))
    forged_graph = serialize_graph(guarded_graph)
    cert["cir"]["graph"] = forged_graph
    cert["cir"]["graph_hash"] = hashlib.sha256(
        _canonical_bytes(forged_graph)).hexdigest()[:32]
    dom = check_dominance(forged_graph)
    assert dom["dominated"]
    cert["cir"]["dominance"] = {
        "claim": "non-blocking",
        "evidence": dom["evidence"],
    }
    # Fix up the outer binding so internal-consistency checks pass.
    cert["binding"]["certificate_hash"] = hashlib.sha256(
        _canonical_bytes(cert["cir"])).hexdigest()

    vr = CertificateVerifier.verify(cert)
    assert not vr.valid, (
        "verifier accepted a forged graph-source link")
    assert any("graph-source" in f.lower() or "re-deriv" in f.lower()
               for f in vr.failures), vr.failures


def _dominance_verdicts(graph):
    """Overall dominated verdict from all three implementations."""
    from chimera.cir.nodes import EvolutionNode
    from chimera.cir.certify import check_dominance, serialize_graph
    from chimera.verify import CertificateVerifier

    lowering = CIRLowering()
    v_lower = True
    for nid, node in graph.nodes.items():
        if isinstance(node, EvolutionNode):
            v_lower = v_lower and lowering._dominates(graph, nid)
    for eid in graph.emit_ids:
        if eid in graph.nodes:
            v_lower = v_lower and lowering._dominates(graph, eid)

    d = serialize_graph(graph)
    dom_certify = check_dominance(d)
    v_certify = dom_certify["dominated"]
    v_verify, all_strict_verify, _ = CertificateVerifier._recompute_dominance(d)
    assert dom_certify["all_strict"] == all_strict_verify, \
        "certify and verify must agree on all_strict"
    return v_lower, v_certify, v_verify


def _random_cir_program(rng):
    """Seeded random CIR program for differential testing."""
    parts = []
    for i in range(rng.randint(1, 3)):
        name = f"b{i}"
        agents = [f"a{j}" for j in range(rng.randint(1, 3))]
        parts.append(
            f'belief {name} := inquire {{\n'
            f'  prompt: "Q{i}?",\n'
            f'  agents: [{", ".join(agents)}]\n}}')
        ops = []
        if len(agents) > 1 and rng.random() < 0.7:
            ops.append(
                f'resolve {name} with consensus {{ threshold: 0.8 }}')
        # Randomize guard/resolve order to hit guard-before-resolve.
        rng.shuffle(ops)
        if rng.random() < 0.6:
            strat = rng.choice(["mean", "both", "variance"])
            ops.append(
                f'guard {name} against hallucination '
                f'{{ max_risk: 0.2, strategy: {strat} }}')
        if rng.random() < 0.5:
            ops.append(f'evolve {name} until stable {{ max_iter: 3 }}')
        parts.extend(ops)
        if rng.random() < 0.7:
            parts.append(f'emit {name}')
    return "\n\n".join(parts) + "\n"


def test_dominance_implementations_agree():
    """Differential test: the lowering, certify, and verifier dominance
    implementations must agree on all examples, the fixture programs,
    and 200 seeded random programs."""
    import glob
    import random

    sources = []
    for f in sorted(glob.glob("examples/*.chimera")):
        sources.append((f, open(f, encoding="utf-8").read()))
    for name, src in [
        ("EMIT_NO_GUARD", EMIT_NO_GUARD),
        ("EVOLVE_NO_GUARD", EVOLVE_NO_GUARD),
        ("GUARD_AFTER_EVOLVE", GUARD_AFTER_EVOLVE),
        ("FANOUT_RESOLVE_NO_GUARD", FANOUT_RESOLVE_NO_GUARD),
        ("CANONICAL", CANONICAL),
        ("SINGLE_GUARD_EMIT", SINGLE_GUARD_EMIT),
        ("GUARD_BEFORE_RESOLVE", GUARD_BEFORE_RESOLVE),
        ("DOUBLE_GUARD", DOUBLE_GUARD),
    ]:
        sources.append((f"fixture:{name}", src))
    rng = random.Random(20261003)
    for i in range(200):
        sources.append((f"random:{i}", _random_cir_program(rng)))

    n_checked = 0
    for label, src in sources:
        try:
            prog = parse_src(src)
        except Exception:
            continue  # not a CIR program; skip
        graph = CIRLowering().lower(prog)
        # Only meaningful if the graph has effectful consumers.
        from chimera.cir.nodes import EvolutionNode
        has_effectful = any(
            isinstance(n, EvolutionNode) for n in graph.nodes.values()
        ) or bool(graph.emit_ids)
        if not has_effectful:
            continue
        v_lower, v_certify, v_verify = _dominance_verdicts(graph)
        assert (v_lower, v_certify, v_verify) == (v_lower, v_lower, v_lower), (
            f"dominance implementations disagree on {label}: "
            f"lower={v_lower} certify={v_certify} verify={v_verify}")
        n_checked += 1
    assert n_checked >= 150, f"too few programs checked: {n_checked}"


def test_not_rederived_never_valid_with_enforced():
    """When the chimera package cannot be imported, verification must
    report 'graph-source link: NOT RE-DERIVED' and must never report
    valid for a certificate claiming 'enforced'."""
    import sys
    from unittest import mock
    from chimera.verify import CertificateVerifier
    from chimera.cir import run_cir
    from chimera.cir.certify import certify_cir
    from chimera.cir.executor import InquiryResponse

    def adapter(prompt, agents):
        return InquiryResponse(confidence=0.95, answer="yes")

    # STRICT_CANONICAL has a source-level strict guard, so the claim is
    # genuinely "enforced" under Option 2.
    prog = parse_src(STRICT_CANONICAL)
    result = run_cir(prog, inquiry_adapter=adapter)
    graph = CIRLowering().lower(prog)
    cert = certify_cir(STRICT_CANONICAL, graph, result, strict_guard=False)
    assert cert["cir"]["dominance"]["claim"] == "enforced"

    blocked = {
        "chimera.lexer": None,
        "chimera.parser": None,
        "chimera.cir.lower": None,
        "chimera.cir.certify": None,
    }
    with mock.patch.dict(sys.modules, blocked):
        vr = CertificateVerifier.verify(cert)
    assert vr.link_status == "NOT RE-DERIVED", vr.link_status
    assert any("NOT RE-DERIVED" in f for f in vr.failures), vr.failures
    assert not vr.valid, "valid with 'enforced' while NOT RE-DERIVED"


def test_not_rederived_non_enforced_still_checked():
    """In NOT RE-DERIVED state the link failure is recorded even for
    non-'enforced' claims (the implementation is stricter than the
    minimum: any NOT RE-DERIVED v2 cert is invalid)."""
    import sys
    from unittest import mock
    from chimera.verify import CertificateVerifier

    cert = _make_certed_run(strict_guard=False)
    assert cert["cir"]["dominance"]["claim"] == "non-blocking"

    blocked = {
        "chimera.lexer": None,
        "chimera.parser": None,
        "chimera.cir.lower": None,
        "chimera.cir.certify": None,
    }
    with mock.patch.dict(sys.modules, blocked):
        vr = CertificateVerifier.verify(cert)
    assert vr.link_status == "NOT RE-DERIVED"
    assert any("NOT RE-DERIVED" in f for f in vr.failures)
    assert not vr.valid


def test_two_lowerings_produce_identical_graphs():
    """Red test: CIRLowering must assign deterministic node ids
    (creation order), so two lowerings of the same source serialize
    byte-for-byte identically. Required for exact graph comparison
    in the verifier."""
    from chimera.cir.certify import _canonical_bytes, serialize_graph

    sources = [
        open("examples/guarded_pipeline.chimera", encoding="utf-8").read(),
        CANONICAL,
        FANOUT_RESOLVE_NO_GUARD,
        DOUBLE_GUARD,
    ]
    for src in sources:
        g1 = CIRLowering().lower(parse_src(src))
        g2 = CIRLowering().lower(parse_src(src))
        b1 = _canonical_bytes(serialize_graph(g1))
        b2 = _canonical_bytes(serialize_graph(g2))
        assert b1 == b2, (
            "two lowerings of the same source must be byte-identical")


# ------------------------------------------------------------------
# Source-level strict guard modifier (Option 2 from
# docs/design/guard-dominance.md). Red tests: these FAIL until the
# strict modifier is implemented end to end (EBNF, lexer/parser,
# GuardStmt, ValidationNode.strict, lowering, executor, certificates,
# verifier).
# ------------------------------------------------------------------

STRICT_CANONICAL = """belief x := inquire {
  prompt: "Is the sky blue?",
  agents: [claude]
}

resolve x with consensus { threshold: 0.8 }

guard x against hallucination { max_risk: 0.2, strategy: both, strict: true }

evolve x until stable { max_iter: 3 }

emit x
"""

NONSTRICT_CANONICAL = """belief x := inquire {
  prompt: "Is the sky blue?",
  agents: [claude]
}

resolve x with consensus { threshold: 0.8 }

guard x against hallucination { max_risk: 0.2, strategy: both }

evolve x until stable { max_iter: 3 }

emit x
"""


def _low_confidence_adapter(prompt, agents):
    from chimera.cir.executor import InquiryResponse
    return InquiryResponse(confidence=0.75, answer="maybe")


def _high_confidence_adapter(prompt, agents):
    from chimera.cir.executor import InquiryResponse
    return InquiryResponse(confidence=0.95, answer="yes")


def test_strict_modifier_parses_to_guard_stmt_and_validation_node():
    """Red: the parser must accept `strict: true` on a guard and thread it
    through GuardStmt into ValidationNode.strict."""
    prog = parse_src(STRICT_CANONICAL)
    guards = [s for s in prog.declarations
              if type(s).__name__ == "GuardStmt"]
    assert guards, "expected a GuardStmt in the parsed program"
    assert guards[0].strict is True, "GuardStmt.strict must be True"

    graph = CIRLowering().lower(prog)
    from chimera.cir.nodes import ValidationNode
    val_nodes = [n for n in graph.nodes.values()
                 if isinstance(n, ValidationNode)]
    assert val_nodes, "expected a ValidationNode in the lowered graph"
    assert all(n.strict is True for n in val_nodes), \
        "ValidationNode.strict must be True"

    prog2 = parse_src(NONSTRICT_CANONICAL)
    guards2 = [s for s in prog2.declarations
               if type(s).__name__ == "GuardStmt"]
    assert guards2[0].strict is False, \
        "GuardStmt.strict must default to False"


def test_failing_strict_guard_halts_like_global_strict():
    """Red (a): a failing source-level strict guard prevents the dominated
    emit and evolve from executing, exactly like the global strict_guard
    flag does today, even when the global flag is off."""
    from chimera.cir import run_cir
    from chimera.cir.executor import GuardViolation

    prog = parse_src(STRICT_CANONICAL)
    with pytest.raises(GuardViolation):
        run_cir(prog, strict_guard=False,
                inquiry_adapter=_low_confidence_adapter)

    # Sanity: today's global-flag behavior still raises on a non-strict guard.
    prog2 = parse_src(NONSTRICT_CANONICAL)
    with pytest.raises(GuardViolation):
        run_cir(prog2, strict_guard=True,
                inquiry_adapter=_low_confidence_adapter)

    # And a non-strict guard with the global flag off still does not raise.
    prog3 = parse_src(NONSTRICT_CANONICAL)
    result = run_cir(prog3, strict_guard=False,
                     inquiry_adapter=_low_confidence_adapter)
    assert result.guard_violations, "expected recorded (non-fatal) violations"


def test_dominance_claim_enforced_only_when_all_dominating_guards_strict():
    """Red (b): the claim is 'enforced' only when every guard on every
    dominating path to each effectful node is source-level strict;
    otherwise 'non-blocking' (dominated) or 'absent' (not dominated)."""
    from chimera.cir import run_cir
    from chimera.cir.certify import certify_cir

    prog = parse_src(STRICT_CANONICAL)
    result = run_cir(prog, inquiry_adapter=_high_confidence_adapter)
    graph = CIRLowering().lower(prog)
    cert = certify_cir(STRICT_CANONICAL, graph, result, strict_guard=False)
    assert cert["cir"]["dominance"]["claim"] == "enforced", \
        "all dominating guards strict -> enforced"

    prog2 = parse_src(NONSTRICT_CANONICAL)
    result2 = run_cir(prog2, inquiry_adapter=_high_confidence_adapter)
    graph2 = CIRLowering().lower(prog2)
    cert2 = certify_cir(NONSTRICT_CANONICAL, graph2, result2,
                        strict_guard=False)
    assert cert2["cir"]["dominance"]["claim"] == "non-blocking", \
        "dominated but non-strict guard -> non-blocking, not enforced"

    prog3 = parse_src(EMIT_NO_GUARD)
    result3 = run_cir(prog3, inquiry_adapter=_high_confidence_adapter)
    graph3 = CIRLowering().lower(prog3)
    cert3 = certify_cir(EMIT_NO_GUARD, graph3, result3, strict_guard=False)
    assert cert3["cir"]["dominance"]["claim"] == "absent", \
        "undominated consumer -> absent"


def test_verifier_derives_claim_from_graph_ignoring_cir_strict_guard():
    """Red (c): the verifier derives the dominance claim from the
    re-lowered graph alone. Flipping cir.strict_guard on a non-strict
    source must not change the expected claim."""
    import copy
    import hashlib
    from chimera.cir import run_cir
    from chimera.cir.certify import certify_cir
    from chimera.verify import CertificateVerifier, _canonical_bytes

    prog = parse_src(NONSTRICT_CANONICAL)
    result = run_cir(prog, inquiry_adapter=_high_confidence_adapter)
    graph = CIRLowering().lower(prog)
    cert = certify_cir(NONSTRICT_CANONICAL, graph, result, strict_guard=False)
    assert cert["cir"]["dominance"]["claim"] == "non-blocking"

    tampered = copy.deepcopy(cert)
    tampered["cir"]["strict_guard"] = True
    cir_bytes = _canonical_bytes(tampered["cir"])
    tampered["binding"]["certificate_hash"] = hashlib.sha256(
        cir_bytes).hexdigest()

    res = CertificateVerifier().verify(tampered)
    assert res.valid, (
        "verifier must ignore cir.strict_guard and still accept the "
        f"non-blocking claim; failures: {res.failures}")
    assert tampered["cir"]["dominance"]["claim"] == "non-blocking"


def test_strict_guard_flag_cannot_force_enforced_claim():
    """Red (d): a certificate with strict_guard set but a non-strict source
    cannot claim 'enforced'; the producer must derive the claim from the
    source-level strict flags."""
    from chimera.cir import run_cir
    from chimera.cir.certify import certify_cir

    prog = parse_src(NONSTRICT_CANONICAL)
    result = run_cir(prog, inquiry_adapter=_high_confidence_adapter)
    graph = CIRLowering().lower(prog)
    cert = certify_cir(NONSTRICT_CANONICAL, graph, result, strict_guard=True)
    assert cert["cir"]["dominance"]["claim"] != "enforced", \
        "strict_guard=True with a non-strict source must not claim enforced"
    assert cert["cir"]["dominance"]["claim"] == "non-blocking"


# ------------------------------------------------------------------
# Guard strength: vacuous guards, guard_strength field, verifier
# warnings. Red tests: these FAIL until implemented.
# ------------------------------------------------------------------

VACUOUS_RISK_SOURCE = """belief x := inquire {
  prompt: "Is the sky blue?",
  agents: [claude]
}

resolve x with consensus { threshold: 0.8 }

guard x against hallucination { max_risk: 1.0, strategy: mean, strict: true }

evolve x until stable { max_iter: 3 }

emit x
"""

VACUOUS_VARIANCE_SOURCE = """belief x := inquire {
  prompt: "Is the sky blue?",
  agents: [claude]
}

resolve x with consensus { threshold: 0.8 }

guard x against hallucination { max_variance: 1.0, strategy: variance, strict: true }

evolve x until stable { max_iter: 3 }

emit x
"""


def test_lowering_warns_on_vacuous_guard():
    """Red (a): lowering must warn on guards that cannot fail.
    A guard cannot fail when none of its active checks can fire:
    - strategy mean/both with max_risk >= 1.0 (requires score >= 0, always true)
    - strategy variance/both with max_variance >= the largest variance any
      belief from that source can have (BetaDist.max_variance_for_strength)
    """
    prog = parse_src(VACUOUS_RISK_SOURCE)
    lowering = CIRLowering()
    lowering.lower(prog)
    assert any("vacuous" in w.lower() for w in lowering.warnings), \
        f"expected vacuous-guard warning, got: {lowering.warnings}"

    prog2 = parse_src(VACUOUS_VARIANCE_SOURCE)
    lowering2 = CIRLowering()
    lowering2.lower(prog2)
    assert any("vacuous" in w.lower() for w in lowering2.warnings), \
        f"expected vacuous-guard warning, got: {lowering2.warnings}"


def test_require_dominance_rejects_vacuous_guard():
    """Red (b): under require_dominance, a vacuous dominating guard is a
    LoweringError, not just a warning."""
    # Use the variance-vacuous source (strategy variance needs no calibrator).
    prog = parse_src(VACUOUS_VARIANCE_SOURCE)
    with pytest.raises(LoweringError) as exc_info:
        CIRLowering(require_dominance=True).lower(prog)
    assert "vacuous" in str(exc_info.value).lower(), \
        f"expected vacuous-guard LoweringError, got: {exc_info.value}"


def _cert_for_source(src, adapter=None, calibrator=None):
    from chimera.cir import run_cir
    from chimera.cir.certify import certify_cir
    from chimera.cir.executor import InquiryResponse

    def _adapter(prompt, agents):
        return InquiryResponse(confidence=0.95, answer="yes")

    prog = parse_src(src)
    result = run_cir(prog, inquiry_adapter=adapter or _adapter,
                     calibrator=calibrator)
    graph = CIRLowering().lower(prog)
    return certify_cir(src, graph, result, strict_guard=False,
                       calibrator=calibrator)


def test_certificate_lists_dominating_guards():
    """Red (c): the dominance section lists each dominating guard with
    strategy, thresholds, strict, score_source, and a vacuous flag."""
    cert = _cert_for_source(STRICT_CANONICAL)
    dom = cert["cir"]["dominance"]
    assert "guards" in dom, "dominance section must list dominating guards"
    guards = dom["guards"]
    assert len(guards) >= 1, "expected at least one dominating guard"
    g = guards[0]
    for field in ("strategy", "max_risk", "max_variance", "strict",
                  "score_source", "vacuous"):
        assert field in g, f"guard entry missing {field!r}: {g}"
    assert g["strict"] is True
    assert g["vacuous"] is False
    assert g["score_source"] in ("calibrated", "uncalibrated")
    assert "guard_strength" in dom
    # No calibrator supplied, so the guard is uncalibrated (but not vacuous).
    assert dom["guard_strength"] == "uncalibrated"


def test_certificate_marks_vacuous_guard():
    """Red (c2): a vacuous dominating guard is flagged and the strength
    is vacuous."""
    cert = _cert_for_source(VACUOUS_RISK_SOURCE)
    dom = cert["cir"]["dominance"]
    assert dom["claim"] == "enforced", \
        "structural claim is still enforced (vacuous but strict)"
    assert "guards" in dom
    assert any(g["vacuous"] for g in dom["guards"]), \
        "vacuous guard must be flagged"
    assert dom["guard_strength"] == "vacuous"


def test_verifier_recomputes_guard_list():
    """Red (d): verify.py recomputes the guard list from the re-lowered
    graph and rejects a certificate whose list differs."""
    import copy
    import hashlib
    from chimera.verify import CertificateVerifier, _canonical_bytes

    cert = _cert_for_source(STRICT_CANONICAL)
    tampered = copy.deepcopy(cert)
    # Flip the strict flag on the guard entry (list differs from graph).
    tampered["cir"]["dominance"]["guards"][0]["strict"] = False
    cir_bytes = _canonical_bytes(tampered["cir"])
    tampered["binding"]["certificate_hash"] = hashlib.sha256(
        cir_bytes).hexdigest()

    res = CertificateVerifier().verify(tampered)
    assert not res.valid, \
        "verifier must reject a certificate with a tampered guard list"
    assert any("guard" in f.lower() for f in res.failures), \
        f"expected guard-list failure, got: {res.failures}"


def test_verifier_rejects_hidden_vacuous_guard():
    """Red (e): a certificate that hides a vacuous guard by editing the
    list (flipping vacuous to False) is rejected."""
    import copy
    import hashlib
    from chimera.verify import CertificateVerifier, _canonical_bytes

    cert = _cert_for_source(VACUOUS_RISK_SOURCE)
    assert cert["cir"]["dominance"]["guard_strength"] == "vacuous"
    tampered = copy.deepcopy(cert)
    for g in tampered["cir"]["dominance"]["guards"]:
        g["vacuous"] = False
    tampered["cir"]["dominance"]["guard_strength"] = "nonvacuous"
    cir_bytes = _canonical_bytes(tampered["cir"])
    tampered["binding"]["certificate_hash"] = hashlib.sha256(
        cir_bytes).hexdigest()

    res = CertificateVerifier().verify(tampered)
    assert not res.valid, \
        "verifier must reject a certificate hiding a vacuous guard"


def test_verify_cli_warns_on_vacuous_or_uncalibrated():
    """Red (f): chimera verify prints a clear warning line when any
    dominating guard is vacuous or uncalibrated, and none when all are
    non-vacuous and calibrated."""
    import json
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    # Vacuous guard -> warning.
    cert_vac = _cert_for_source(VACUOUS_RISK_SOURCE)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                     delete=False) as f:
        json.dump(cert_vac, f)
        vac_path = f.name
    proc = subprocess.run(
        [sys.executable, "-m", "chimera.cli", "verify", vac_path],
        capture_output=True, text=True, cwd=str(Path(__file__).parent.parent))
    assert "WARNING" in proc.stdout, \
        f"expected WARNING for vacuous guard, got: {proc.stdout}"
    assert "vacuous" in proc.stdout.lower()

    # Uncalibrated (but non-vacuous) -> warning.
    cert_uncal = _cert_for_source(STRICT_CANONICAL)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                     delete=False) as f:
        json.dump(cert_uncal, f)
        uncal_path = f.name
    proc2 = subprocess.run(
        [sys.executable, "-m", "chimera.cli", "verify", uncal_path],
        capture_output=True, text=True, cwd=str(Path(__file__).parent.parent))
    assert "WARNING" in proc2.stdout, \
        f"expected WARNING for uncalibrated guard, got: {proc2.stdout}"
    assert "uncalibrated" in proc2.stdout.lower()
