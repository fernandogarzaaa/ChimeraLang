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
