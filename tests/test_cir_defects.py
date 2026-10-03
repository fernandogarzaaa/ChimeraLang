"""Regression tests for four verified CIR belief-engine defects.

Audit date: 2026-10-02. Each test here fails on the pre-fix code and
passes after the corresponding defect fix. They are committed before
any fix (test-first, gated stages).

Defect 1: evolve after resolve/guard never calls the adapter.
Defect 2: guard has no usable variance limit (hardcoded 0.05 can never fire).
Defect 3: resolve over a single input traces a dishonest "combined mean".
Defect 4: seeded priors are overwritten, posteriors double-counted, and
    symbol prior strength grows without bound.
"""
import json

import pytest

from chimera.ast_nodes import (
    BeliefDecl,
    EmitStmt,
    EvolveStmt,
    GuardStmt,
    Identifier,
    InquireExpr,
    Program,
    ResolveStmt,
)
from chimera.cir import run_cir
from chimera.cir.executor import CIRExecutor
from chimera.cir.lower import CIRLowering
from chimera.cir.nodes import BetaDist
from chimera.cir.symbols import Symbol


def make_program(*stmts):
    return Program(declarations=list(stmts))


def counting_adapter(confidence):
    """Adapter returning a constant confidence; records every call."""
    calls = []

    def adapter(prompt, agents):
        calls.append(prompt)
        return confidence

    adapter.calls = calls
    return adapter


def _belief(name="c", prompt="Is the claim true?"):
    return BeliefDecl(
        name=name,
        inquire_expr=InquireExpr(prompt=prompt, agents=[], ttl=None),
    )


# ---------------------------------------------------------------------------
# Defect 1: evolve after resolve / guard
# ---------------------------------------------------------------------------

class TestEvolveAfterResolveOrGuard:
    def test_evolve_after_resolve_matches_direct_evolve(self):
        direct = make_program(
            _belief(),
            EvolveStmt(target="c", condition="stable", max_iter=3),
            EmitStmt(value=Identifier(name="c")),
        )
        with_resolve = make_program(
            _belief(),
            ResolveStmt(target="c", threshold=0.5, strategy="dempster_shafer"),
            EvolveStmt(target="c", condition="stable", max_iter=3),
            EmitStmt(value=Identifier(name="c")),
        )
        adapter_direct = counting_adapter(0.75)
        adapter_resolve = counting_adapter(0.75)
        r_direct = run_cir(direct, inquiry_adapter=adapter_direct)
        r_resolve = run_cir(with_resolve, inquiry_adapter=adapter_resolve)

        # Direct case: 1 inquiry call + 1 evolve iteration call, converged.
        assert len(adapter_direct.calls) == 2
        assert r_direct.converged is True
        # Resolve case must behave identically.
        assert len(adapter_resolve.calls) == 2
        assert r_resolve.converged is True

    def test_evolve_after_guard_matches_direct_evolve(self):
        direct = make_program(
            _belief(),
            EvolveStmt(target="c", condition="stable", max_iter=3),
            EmitStmt(value=Identifier(name="c")),
        )
        with_guard = make_program(
            _belief(),
            GuardStmt(target="c", max_risk=0.3, strategy="both"),
            EvolveStmt(target="c", condition="stable", max_iter=3),
            EmitStmt(value=Identifier(name="c")),
        )
        adapter_direct = counting_adapter(0.75)
        adapter_guard = counting_adapter(0.75)
        r_direct = run_cir(direct, inquiry_adapter=adapter_direct)
        r_guard = run_cir(with_guard, inquiry_adapter=adapter_guard)

        assert len(adapter_direct.calls) == 2
        assert r_direct.converged is True
        assert len(adapter_guard.calls) == 2
        assert r_guard.converged is True


# ---------------------------------------------------------------------------
# Defect 2: variance guard reachability
# ---------------------------------------------------------------------------

class TestVarianceGuard:
    def test_explicit_max_variance_records_violation(self):
        # Beta(7.5, 2.5) has variance 0.0170, above the 0.01 limit.
        prog = make_program(
            _belief(name="v"),
            GuardStmt(target="v", max_risk=0.0, strategy="variance",
                      max_variance=0.01),
            EmitStmt(value=Identifier(name="v")),
        )
        result = run_cir(prog, inquiry_adapter=lambda p, a: 0.75)
        assert any("variance" in v for v in result.guard_violations)

    def test_lowering_warns_on_unreachable_variance_limit(self):
        # BetaDist.from_confidence uses strength 10, which caps variance at
        # about 0.0227, so the hardcoded default 0.05 can never fire.
        prog = make_program(
            _belief(name="v"),
            GuardStmt(target="v", max_risk=0.2, strategy="variance"),
            EmitStmt(value=Identifier(name="v")),
        )
        lowering = CIRLowering()
        lowering.lower(prog)
        assert any("unreachable" in w for w in lowering.warnings)

    def test_reachable_variance_limit_produces_no_unreachable_warning(self):
        prog = make_program(
            _belief(name="v"),
            GuardStmt(target="v", max_risk=0.2, strategy="variance",
                      max_variance=0.01),
            EmitStmt(value=Identifier(name="v")),
        )
        lowering = CIRLowering()
        lowering.lower(prog)
        assert not any("unreachable" in w for w in lowering.warnings)


# ---------------------------------------------------------------------------
# Defect 3: honest resolve trace
# ---------------------------------------------------------------------------

class TestResolveTraceHonesty:
    def test_single_source_resolve_trace(self):
        prog = make_program(
            _belief(name="s"),
            ResolveStmt(target="s", threshold=0.5, strategy="dempster_shafer"),
            EmitStmt(value=Identifier(name="s")),
        )
        result = run_cir(prog, inquiry_adapter=lambda p, a: 0.75)
        assert any(
            "single source, no combination performed" in t
            for t in result.trace
        )
        assert not any("[consensus] combined mean" in t for t in result.trace)

    def test_combine_pseudocount_characterization(self):
        # Pins the actual arithmetic: pure pseudocount addition.
        # Beta(8,2) + Beta(8,2) -> Beta(16,4), mean 16/20 = 0.8.
        # Deliberately changed from the old Beta(15,3) / 15/18: the old
        # rule subtracted one pseudocount per combination
        # (alpha + alpha' - 1), which is what drove raw parameters to
        # zero and negative for agreeing high-confidence sources. The
        # new rule adds evidence counts without the subtract-one, so
        # two identical sources pool to exactly their shared mean.
        combined = BetaDist(8.0, 2.0).combine_pseudocount(BetaDist(8.0, 2.0))
        assert (combined.alpha, combined.beta) == (16.0, 4.0)
        assert combined.mean == pytest.approx(0.8, abs=1e-9)

    def test_combine_ds_alias_still_works(self):
        combined = BetaDist(8.0, 2.0).combine_ds(BetaDist(8.0, 2.0))
        assert combined.mean == pytest.approx(0.8, abs=1e-9)


# ---------------------------------------------------------------------------
# Defect 4: seeded priors
# ---------------------------------------------------------------------------

class TestSeededPriors:
    def test_seeded_prior_influences_posterior(self, tmp_path):
        prog = make_program(
            _belief(name="p", prompt="What is gravity?"),
            EmitStmt(value=Identifier(name="p")),
        )
        store_path = str(tmp_path / "symbols.json")
        adapter = lambda p, a: 0.9  # noqa: E731

        import os
        for _ in range(5):
            kwargs = {"save_symbols": store_path}
            if os.path.exists(store_path):
                kwargs["load_symbols"] = store_path
            run_cir(prog, inquiry_adapter=adapter, **kwargs)

        fresh = run_cir(prog, inquiry_adapter=adapter)
        seeded = run_cir(prog, inquiry_adapter=adapter,
                         load_symbols=store_path)

        assert seeded.meta["priors_seeded"], "expected a seeded prior"
        assert seeded.beliefs["p"].mean != pytest.approx(
            fresh.beliefs["p"].mean
        ), "seeded prior pseudocounts must shift the posterior"

    def test_symbol_prior_strength_stays_bounded(self):
        sym = Symbol(wl_hash="x", node_types=[], edge_types=[],
                     prompts=["q"])
        for _ in range(50):
            sym.record_observation(BetaDist.from_confidence(0.9))
        strength = sym.prior_alpha + sym.prior_beta
        assert strength <= Symbol.MAX_PRIOR_STRENGTH + 1e-9
        # Repeated positive evidence still concentrates the prior.
        assert 0.99 < sym.prior().mean < 1.0

    def test_prior_cap_preserves_mean_at_engagement(self):
        # Rescaling on cap engagement must not distort the calibrated mean.
        sym = Symbol(wl_hash="x", node_types=[], edge_types=[],
                     prompts=["q"], prior_alpha=90.0, prior_beta=10.0)
        sym.record_observation(BetaDist.from_confidence(0.9))  # adds (8, 0)
        assert sym.prior_alpha + sym.prior_beta == pytest.approx(100.0)
        assert sym.prior().mean == pytest.approx(98 / 108)
