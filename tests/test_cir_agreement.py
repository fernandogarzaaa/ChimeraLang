"""Red tests for agreement-based resolve and opt-in calibration (Stage 3).

Agreement resolve (chimera/cir/agreement.py):
- new `agreement` resolve strategy, the default when a belief has more
  than one source with answers;
- `pooled` keeps the Stage 2 Beta algebra; `dempster_shafer` stays
  accepted as an alias of `pooled` with a lowering warning;
- pluggable comparator (two answers -> bool) plus normalizer hook;
  default normalization is exactly experiments/h1_pooling/h1_common.py
  normalize_answer;
- winner = most common normalized answer, ties broken by earliest
  source in agent order;
- resolved belief carries raw agreement (votes/N), a Laplace-smoothed
  Beta posterior (unanimous 3/3 does not claim 1.0), and the winning
  answer text; the trace line prints both numbers and "uncalibrated";
- single-source resolve passes through unchanged.

Calibration (chimera/cir/calibration.py):
- LogisticCalibrator with deterministic pure-Python fit(scores,
  outcomes); to_json/from_json carry n, dataset hash, fit date;
  refuses to fit under MIN_FIT_N;
- run_cir accepts a calibrator; CLI gains --calibrator=PATH;
- calibrated_p appears only when a calibrator is supplied; guard and
  emit use it when present, else the uncalibrated posterior with a
  lowering warning.

All tests in this file are RED until the implementation lands.
"""
import importlib.util
import json
import math

import pytest

from chimera.ast_nodes import (
    BeliefDecl, EmitStmt, GuardStmt, Identifier, InquireExpr,
    Program, ResolveStmt,
)
from chimera.cir import run_cir
from chimera.cir.nodes import BetaDist


def _h1_normalize():
    """Load h1_common.normalize_answer straight from the experiment file."""
    spec = importlib.util.spec_from_file_location(
        "h1_common", "experiments/h1_pooling/h1_common.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.normalize_answer


# ---------------------------------------------------------------------------
# agreement.py unit tests
# ---------------------------------------------------------------------------

class TestAgreementModule:
    def test_normalization_matches_h1(self):
        from chimera.cir.agreement import normalize_answer
        h1 = _h1_normalize()
        cases = ["The Eiffel Tower", "  Paris! ", "A   cat", "An apple, red",
                 "U.S.A.", "", "the the the", " 42 ", "New\nYork"]
        for c in cases:
            assert normalize_answer(c) == h1(c), f"mismatch on {c!r}"

    def test_winner_is_most_common_normalized(self):
        from chimera.cir.agreement import resolve_agreement
        ar = resolve_agreement(["Paris", "paris!", "London"])
        assert ar.winner == "paris"
        assert ar.votes == 2 and ar.n == 3
        assert ar.agreement == pytest.approx(2 / 3)

    def test_tie_broken_by_earliest_source(self):
        from chimera.cir.agreement import resolve_agreement
        ar = resolve_agreement(["London", "Paris"])
        assert ar.winner == "london"
        ar2 = resolve_agreement(["Paris", "London"])
        assert ar2.winner == "paris"

    def test_laplace_posterior_unanimous_not_one(self):
        from chimera.cir.agreement import resolve_agreement
        ar = resolve_agreement(["a", "a", "a"])
        assert ar.posterior.alpha == pytest.approx(4.0)
        assert ar.posterior.beta == pytest.approx(1.0)
        assert ar.posterior.mean < 1.0
        assert ar.posterior.mean == pytest.approx(0.8)

    def test_laplace_posterior_partial(self):
        from chimera.cir.agreement import resolve_agreement
        ar = resolve_agreement(["a", "a", "b"])
        assert ar.posterior.mean == pytest.approx(3 / 5)

    def test_pluggable_comparator(self):
        from chimera.cir.agreement import resolve_agreement
        comp = lambda a, b: a[0].lower() == b[0].lower()
        ar = resolve_agreement(["Paris", "Peru", "London"], comparator=comp)
        assert ar.votes == 2 and ar.winner == "paris"

    def test_pluggable_normalizer(self):
        from chimera.cir.agreement import resolve_agreement
        norm = lambda s: s.strip().lower()
        ar = resolve_agreement(["The cat", "the cat"], normalizer=norm)
        assert ar.winner == "the cat" and ar.votes == 2


# ---------------------------------------------------------------------------
# resolve strategy integration (run_cir = the real path)
# ---------------------------------------------------------------------------

def _program(threshold=0.5, strategy=None, guard_risk=None):
    stmts = [BeliefDecl(name="x",
                        inquire_expr=InquireExpr(prompt="Q", agents=["a", "b", "c"],
                                               ttl=None))]
    if strategy is None:
        stmts.append(ResolveStmt(target="x", threshold=threshold))
    else:
        stmts.append(ResolveStmt(target="x", threshold=threshold, strategy=strategy))
    if guard_risk is not None:
        stmts.append(GuardStmt(target="x", max_risk=guard_risk, strategy="both"))
    stmts.append(EmitStmt(value=Identifier(name="x")))
    return Program(declarations=stmts)


def _adapter_2v1(prompt, agents):
    ans = {"a": "Paris", "b": "paris!", "c": "London"}[agents[0]]
    return {"confidence": 0.9, "answer": ans}


class TestAgreementResolve:
    def test_agreement_is_default_for_multi_source(self):
        result = run_cir(_program(), inquiry_adapter=_adapter_2v1)
        assert any("agreement=" in t and "uncalibrated" in t for t in result.trace), \
            result.trace

    def test_trace_prints_both_numbers_and_uncalibrated(self):
        result = run_cir(_program(), inquiry_adapter=_adapter_2v1)
        line = next(t for t in result.trace if "agreement=" in t)
        assert "uncalibrated" in line
        assert "2/3" in line  # votes/N
        assert "0.667" in line  # raw agreement
        assert "0.600" in line  # Laplace posterior mean Beta(3,2)

    def test_resolved_belief_carries_winner_and_agreement(self):
        result = run_cir(_program(), inquiry_adapter=_adapter_2v1)
        assert result.answers["x@a"] == "paris"
        assert result.beliefs["x@a"].mean == pytest.approx(0.6)

    def test_single_source_passthrough_unchanged(self):
        prog = Program(declarations=[
            BeliefDecl(name="x", inquire_expr=InquireExpr(prompt="Q", agents=["a"], ttl=None)),
            ResolveStmt(target="x", threshold=0.5),
            EmitStmt(value=Identifier(name="x")),
        ])
        result = run_cir(prog, inquiry_adapter=lambda p, a: 0.75)
        assert any("single source, no combination performed" in t for t in result.trace)
        assert not any("agreement=" in t for t in result.trace)

    def test_dempster_shafer_alias_warns_and_pools(self):
        result = run_cir(_program(strategy="dempster_shafer"), inquiry_adapter=_adapter_2v1)
        warnings = result.meta.get("lowering_warnings", [])
        assert any("dempster_shafer" in w and "alias" in w for w in warnings), warnings
        # pooled Beta algebra: (9+9+9, 1+1+1) mean 27/30 = 0.9
        assert result.beliefs["x@a"].mean == pytest.approx(0.9)
        assert not any("agreement=" in t for t in result.trace)

    def test_pooled_strategy_kept(self):
        result = run_cir(_program(strategy="pooled"), inquiry_adapter=_adapter_2v1)
        assert result.beliefs["x@a"].mean == pytest.approx(0.9)
        assert not any("agreement=" in t for t in result.trace)

    def test_agreement_falls_back_to_pooled_without_answers(self):
        result = run_cir(_program(), inquiry_adapter=lambda p, a: 0.8)
        assert any("falling back to pooled" in t for t in result.trace)
        assert not any("agreement=" in t for t in result.trace)


# ---------------------------------------------------------------------------
# calibration.py unit tests
# ---------------------------------------------------------------------------

def _fit_data():
    scores = [0.55, 0.60, 0.65] * 10 + [0.85, 0.90, 0.95] * 10
    outcomes = [0] * 30 + [1] * 30
    return scores, outcomes


class TestLogisticCalibrator:
    def test_fit_predict_sane(self):
        from chimera.cir.calibration import LogisticCalibrator
        scores, outcomes = _fit_data()
        cal = LogisticCalibrator.fit(scores, outcomes)
        assert cal.n == 60
        assert cal.predict(0.1) < 0.5 < cal.predict(0.9)
        assert 0.0 < cal.predict(0.6) < 1.0

    def test_fit_is_deterministic(self):
        from chimera.cir.calibration import LogisticCalibrator
        scores, outcomes = _fit_data()
        c1 = LogisticCalibrator.fit(scores, outcomes)
        c2 = LogisticCalibrator.fit(scores, outcomes)
        assert (c1.a, c1.b) == (c2.a, c2.b)

    def test_fit_refuses_small_n(self):
        from chimera.cir.calibration import LogisticCalibrator, MIN_FIT_N
        assert MIN_FIT_N > 0
        with pytest.raises(ValueError, match="[Mm]inimum"):
            LogisticCalibrator.fit([0.5] * (MIN_FIT_N - 1), [1] * (MIN_FIT_N - 1))

    def test_json_roundtrip_carries_metadata(self):
        from chimera.cir.calibration import LogisticCalibrator
        scores, outcomes = _fit_data()
        cal = LogisticCalibrator.fit(
            scores, outcomes, dataset_hash="deadbeef",
            metadata={"n_sources": 3, "source_type": "cross-model"})
        d = json.loads(cal.to_json())
        assert d["n"] == 60
        assert d["dataset_hash"] == "deadbeef"
        assert "fit_date" in d and d["fit_date"]
        assert d["metadata"] == {"n_sources": 3,
                                 "source_type": "cross-model"}
        cal2 = LogisticCalibrator.from_json(cal.to_json())
        assert (cal2.a, cal2.b, cal2.n, cal2.dataset_hash) == \
            (cal.a, cal.b, cal.n, cal.dataset_hash)
        assert cal2.metadata == {"n_sources": 3,
                                 "source_type": "cross-model"}
        assert cal2.predict(0.7) == pytest.approx(cal.predict(0.7))

    def test_from_json_without_metadata_still_loads(self):
        from chimera.cir.calibration import LogisticCalibrator
        old = json.dumps({"a": 1.0, "b": -0.5, "n": 60,
                          "dataset_hash": "abc", "fit_date": "2026-10-03"})
        cal = LogisticCalibrator.from_json(old)
        assert cal.metadata == {}
        import math
        assert cal.predict(0.7) == pytest.approx(1.0 / (1.0 + math.exp(-0.2)))

    def test_dataset_hash_defaults_to_data_hash(self):
        from chimera.cir.calibration import LogisticCalibrator
        scores, outcomes = _fit_data()
        cal = LogisticCalibrator.fit(scores, outcomes)
        assert len(cal.dataset_hash) == 64


# ---------------------------------------------------------------------------
# calibration integration (run_cir = the real path)
# ---------------------------------------------------------------------------

class TestCalibrationIntegration:
    def test_no_calibrator_warns_uncalibrated(self):
        result = run_cir(_program(), inquiry_adapter=_adapter_2v1)
        warnings = result.meta.get("lowering_warnings", [])
        assert any("uncalibrated" in w for w in warnings), warnings

    def test_calibrator_sets_calibrated_p(self):
        from chimera.cir.calibration import LogisticCalibrator
        cal = LogisticCalibrator(a=10.0, b=-4.0, n=100,
                                dataset_hash="test", fit_date="2026-10-03")
        result = run_cir(_program(), inquiry_adapter=_adapter_2v1, calibrator=cal)
        assert any("calibrated_p=" in t for t in result.trace)
        # uncalibrated posterior mean for 2/3 agreement is 0.6
        assert result.beliefs["x@a"].mean == pytest.approx(0.6)
        expected = 1 / (1 + math.exp(-(10.0 * 0.6 - 4.0)))
        assert result.calibrated["x@a"] == pytest.approx(expected)

    def test_no_calibrated_p_without_calibrator(self):
        result = run_cir(_program(), inquiry_adapter=_adapter_2v1)
        assert result.calibrated == {}

    def test_guard_uses_calibrated_p(self):
        from chimera.cir.calibration import LogisticCalibrator
        # 2/3 agreement -> posterior mean 0.6; guard max_risk=0.2 needs 0.8.
        # Calibrator maps 0.6 -> sigmoid(2) = 0.881 >= 0.8: passes only
        # when the calibrator is supplied.
        cal = LogisticCalibrator(a=10.0, b=-4.0, n=100,
                                dataset_hash="test", fit_date="2026-10-03")
        without = run_cir(_program(guard_risk=0.2), inquiry_adapter=_adapter_2v1)
        assert any("FAILED" in t for t in without.trace), without.trace
        with_cal = run_cir(_program(guard_risk=0.2), inquiry_adapter=_adapter_2v1,
                           calibrator=cal)
        assert any("PASSED" in t for t in with_cal.trace), with_cal.trace
        assert with_cal.guard_violations == []
