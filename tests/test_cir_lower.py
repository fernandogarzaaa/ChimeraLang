"""Tests for AST → CIR lowering."""
import pytest
from chimera.ast_nodes import (
    BeliefDecl, InquireExpr, ResolveStmt, GuardStmt,
    EvolveStmt, EmitStmt, Identifier, Program,
)
from chimera.cir.lower import CIRLowering, LoweringError
from chimera.cir.nodes import (
    InquiryNode, ConsensusNode, ValidationNode, EvolutionNode,
    EdgeKind,
)


def make_program(*stmts):
    return Program(declarations=list(stmts))


class TestStructuralLowering:
    def test_belief_decl_creates_inquiry_node(self):
        prog = make_program(
            BeliefDecl(
                name="cause",
                inquire_expr=InquireExpr(prompt="Why?", agents=["claude"], ttl=None),
            )
        )
        graph = CIRLowering().lower(prog)
        inquiry_nodes = [n for n in graph.nodes.values() if isinstance(n, InquiryNode)]
        assert len(inquiry_nodes) == 1
        assert inquiry_nodes[0].prompt == "Why?"
        assert "cause" in graph.belief_store

    def test_resolve_creates_consensus_node(self):
        prog = make_program(
            BeliefDecl(name="x", inquire_expr=InquireExpr(prompt="Q", agents=[], ttl=None)),
            ResolveStmt(target="x", threshold=0.8, strategy="dempster_shafer"),
        )
        graph = CIRLowering().lower(prog)
        cons_nodes = [n for n in graph.nodes.values() if isinstance(n, ConsensusNode)]
        assert len(cons_nodes) == 1
        assert cons_nodes[0].threshold == 0.8

    def test_guard_creates_validation_node(self):
        prog = make_program(
            BeliefDecl(name="x", inquire_expr=InquireExpr(prompt="Q", agents=[], ttl=None)),
            GuardStmt(target="x", max_risk=0.2, strategy="both"),
        )
        graph = CIRLowering().lower(prog)
        val_nodes = [n for n in graph.nodes.values() if isinstance(n, ValidationNode)]
        assert len(val_nodes) == 1
        assert val_nodes[0].max_risk == 0.2

    def test_evolve_creates_evolution_node(self):
        prog = make_program(
            BeliefDecl(name="x", inquire_expr=InquireExpr(prompt="Q", agents=[], ttl=None)),
            EvolveStmt(target="x", condition="stable", max_iter=3),
        )
        graph = CIRLowering().lower(prog)
        evo_nodes = [n for n in graph.nodes.values() if isinstance(n, EvolutionNode)]
        assert len(evo_nodes) == 1
        assert evo_nodes[0].max_iter == 3


class TestDeadBeliefElimination:
    def test_unreachable_belief_removed(self):
        prog = make_program(
            BeliefDecl(name="used", inquire_expr=InquireExpr(prompt="A", agents=[], ttl=None)),
            BeliefDecl(name="unused", inquire_expr=InquireExpr(prompt="B", agents=[], ttl=None)),
            EmitStmt(value=Identifier(name="used")),
        )
        graph = CIRLowering().lower(prog)
        inquiry_prompts = [
            n.prompt for n in graph.nodes.values() if isinstance(n, InquiryNode)
        ]
        assert "B" not in inquiry_prompts

    def test_used_belief_kept(self):
        prog = make_program(
            BeliefDecl(name="used", inquire_expr=InquireExpr(prompt="A", agents=[], ttl=None)),
            EmitStmt(value=Identifier(name="used")),
        )
        graph = CIRLowering().lower(prog)
        inquiry_prompts = [
            n.prompt for n in graph.nodes.values() if isinstance(n, InquiryNode)
        ]
        assert "A" in inquiry_prompts


class TestBeliefFlowAnalysis:
    def test_high_variance_belief_flagged(self):
        prog = make_program(
            BeliefDecl(name="risky", inquire_expr=InquireExpr(prompt="X", agents=[], ttl=None)),
            GuardStmt(target="risky", max_risk=0.05, strategy="variance"),
        )
        lowering = CIRLowering()
        graph = lowering.lower(prog)
        assert any("variance" in w.lower() for w in lowering.warnings)

    def test_no_warnings_for_valid_graph(self):
        prog = make_program(
            BeliefDecl(name="x", inquire_expr=InquireExpr(prompt="Q", agents=[], ttl=None)),
        )
        lowering = CIRLowering()
        lowering.lower(prog)
        # No validation node present, so no variance warnings
        assert len(lowering.warnings) == 0


class TestSymbolStorePriors:
    """Lowering consults the SymbolStore to seed BetaDist priors."""

    def test_no_store_uses_uniform_prior(self):
        prog = make_program(
            BeliefDecl(name="x", inquire_expr=InquireExpr(
                prompt="What is gravity?", agents=[], ttl=None,
            )),
        )
        graph = CIRLowering().lower(prog)
        # No store, no priors seeded.
        assert graph.belief_store["x"].distribution.alpha == 1.0
        assert graph.belief_store["x"].distribution.beta == 1.0

    def test_store_with_observation_seeds_prior(self):
        from chimera.cir.symbols import SymbolStore
        from chimera.cir.nodes import BetaDist, CIRGraph, InquiryNode

        store = SymbolStore()
        seed_graph = CIRGraph()
        seed_graph.add_node(InquiryNode(prompt="What is gravity?"))
        store.register(seed_graph, prompts=["What is gravity?"])
        store.record_observation(
            "What is gravity?", BetaDist.from_confidence(0.9, strength=20.0)
        )

        prog = make_program(
            BeliefDecl(name="x", inquire_expr=InquireExpr(
                prompt="What is gravity?", agents=[], ttl=None,
            )),
        )
        lowering = CIRLowering(symbol_store=store)
        graph = lowering.lower(prog)
        # Prior should be pulled toward the observed mean of 0.9.
        assert graph.belief_store["x"].distribution.mean > 0.7
        assert "x" in lowering.priors_seeded

    def test_run_cir_round_trip_persists_observation(self, tmp_path):
        """End-to-end: run, save store, reload store, run again — second run
        should have its prior seeded from the first run's posterior."""
        import json
        from chimera.cir import run_cir
        from chimera.cir.executor import InquiryResponse

        prog = make_program(
            BeliefDecl(name="g", inquire_expr=InquireExpr(
                prompt="What is gravity?", agents=[], ttl=None,
            )),
            EmitStmt(value=Identifier(name="g")),
        )
        store_path = tmp_path / "symbols.json"

        # First run: confident answer, should accumulate evidence.
        run_cir(
            prog,
            inquiry_adapter=lambda p, a: InquiryResponse(confidence=0.9, answer="ok"),
            save_symbols=str(store_path),
        )

        # The persisted store must carry non-uniform prior_alpha/beta now.
        data = json.loads(store_path.read_text())
        syms = data["symbols"]
        assert any(s.get("prior_alpha", 1.0) > 1.0 for s in syms)

        # Second run: load the store and re-lower with a no-op adapter
        # that returns conf=0.5 with no real signal — the seeded prior
        # should still be near 0.9 because lowering happens BEFORE the
        # executor overwrites it. We assert via meta.
        result = run_cir(
            prog,
            inquiry_adapter=lambda p, a: 0.5,
            load_symbols=str(store_path),
        )
        assert "g" in result.meta["priors_seeded"]
