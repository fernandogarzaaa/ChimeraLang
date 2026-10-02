"""Red tests for CIR fan-out: one InquiryNode per agent.

Fan-out A: a belief with agents [a, b] lowers to two InquiryNodes
(agents=["a"] and agents=["b"]); resolve pools them with the shipped
combine_pseudocount chain. Sources are LESS CORRELATED, not
independent (models share training data).
"""
import pytest
from types import SimpleNamespace

from chimera.ast_nodes import (
    BeliefDecl, EmitStmt, GuardStmt, Identifier, InquireExpr,
    Program, ResolveStmt,
)
from chimera.cir import run_cir
from chimera.cir.executor import CIRExecutor
from chimera.cir.lower import CIRLowering
from chimera.cir.nodes import (
    BetaDist, ConsensusNode, EdgeKind, InquiryNode, ValidationNode,
)


def make_program(*stmts):
    return Program(declarations=list(stmts))


def fanout_program(threshold=0.5):
    return make_program(
        BeliefDecl(
            name="x",
            inquire_expr=InquireExpr(prompt="Q", agents=["a", "b"], ttl=None),
        ),
        ResolveStmt(target="x", threshold=threshold, strategy="dempster_shafer"),
        EmitStmt(value=Identifier(name="x")),
    )


class TestFanoutLowering:
    def test_multi_agent_belief_creates_one_inquiry_per_agent(self):
        graph = CIRLowering().lower(fanout_program())
        inqs = [n for n in graph.nodes.values() if isinstance(n, InquiryNode)]
        assert len(inqs) == 2
        assert sorted([inq.agents for inq in inqs]) == [["a"], ["b"]]
        assert "x@a" in graph.belief_store
        assert "x@b" in graph.belief_store

    def test_resolve_covers_all_source_inquiries(self):
        graph = CIRLowering().lower(fanout_program())
        inqs = [n for n in graph.nodes.values() if isinstance(n, InquiryNode)]
        cons = [n for n in graph.nodes.values() if isinstance(n, ConsensusNode)]
        assert len(cons) == 1
        assert sorted(cons[0].input_ids) == sorted(inq.id for inq in inqs)
        cons_edges = [e for e in graph.edges if e.kind == EdgeKind.CONSENSUS]
        assert len(cons_edges) == 2
        assert sorted(e.source_id for e in cons_edges) == sorted(inq.id for inq in inqs)
        assert all(e.target_id == cons[0].id for e in cons_edges)

    def test_adapter_called_once_per_agent_and_pooled(self):
        calls = []

        def adapter(prompt, agents):
            calls.append(list(agents))
            return {"confidence": 0.8 if agents == ["a"] else 0.6,
                    "answer": f"ans-{agents[0]}"}

        result = run_cir(fanout_program(), inquiry_adapter=adapter)
        assert sorted(calls) == [["a"], ["b"]]
        expected = BetaDist.from_confidence(0.8).combine_pseudocount(
            BetaDist.from_confidence(0.6))
        for name in ("x@a", "x@b"):
            got = result.beliefs[name]
            assert abs(got.mean - expected.mean) < 1e-9
            assert abs(got.variance - expected.variance) < 1e-9
        assert result.emitted[0][0] == "x@a"
        assert result.guard_violations == []


class TestSingleAgentUnchanged:
    def test_single_agent_belief_shape_unchanged(self):
        prog = make_program(
            BeliefDecl(name="x",
                       inquire_expr=InquireExpr(prompt="Q", agents=["a"], ttl=None)),
            ResolveStmt(target="x", threshold=0.8, strategy="dempster_shafer"),
            EmitStmt(value=Identifier(name="x")),
        )
        graph = CIRLowering().lower(prog)
        inqs = [n for n in graph.nodes.values() if isinstance(n, InquiryNode)]
        assert len(inqs) == 1
        assert inqs[0].agents == ["a"]
        assert list(graph.belief_store) == ["x"]
        assert graph.belief_store["x"].node_id == inqs[0].id


class TestDuplicateAgents:
    def test_duplicate_agent_names_warn_correlated(self):
        lowering = CIRLowering()
        prog = make_program(
            BeliefDecl(name="x", inquire_expr=InquireExpr(
                prompt="Q", agents=["claude", "claude"], ttl=None)),
            EmitStmt(value=Identifier(name="x")),
        )
        lowering.lower(prog)
        assert any("same model" in w and "correlated" in w
                   for w in lowering.warnings), lowering.warnings


class TestMultiAgentWithoutResolve:
    def test_warns_and_guards_first_agent_only(self):
        # Defined behavior (also documented in README): without resolve
        # there is no pooled belief, so guard/emit apply to the FIRST
        # agent's belief only. The warning names the remedy.
        lowering = CIRLowering()
        prog = make_program(
            BeliefDecl(name="x", inquire_expr=InquireExpr(
                prompt="Q", agents=["a", "b"], ttl=None)),
            GuardStmt(target="x", max_risk=0.5, strategy="mean"),
            EmitStmt(value=Identifier(name="x")),
        )
        graph = lowering.lower(prog)
        assert any("no resolve" in w for w in lowering.warnings), lowering.warnings
        vals = [n for n in graph.nodes.values() if isinstance(n, ValidationNode)]
        assert len(vals) == 1
        first_inq = next(n for n in graph.nodes.values()
                         if isinstance(n, InquiryNode) and n.agents == ["a"])
        assert vals[0].target_id == first_inq.id

    def test_first_agent_only_is_observable_at_runtime(self):
        def adapter(prompt, agents):
            return {"confidence": 0.9 if agents == ["a"] else 0.1}
        prog = make_program(
            BeliefDecl(name="x", inquire_expr=InquireExpr(
                prompt="Q", agents=["a", "b"], ttl=None)),
            GuardStmt(target="x", max_risk=0.5, strategy="mean"),
            EmitStmt(value=Identifier(name="x")),
        )
        result = run_cir(prog, inquiry_adapter=adapter)
        # "a" passes (0.9 >= 0.5); "b" (0.1) is never guarded.
        assert result.guard_violations == []
        assert result.emitted[0][0] == "x@a"


class TestAdapterRouting:
    def _fake_client(self, seen):
        class FakeMessages:
            def create(self, **kw):
                seen.update(kw)
                return SimpleNamespace(content=[SimpleNamespace(
                    text='{"answer": "Paris", "confidence": 0.9}')])
        return SimpleNamespace(messages=FakeMessages())

    def test_agent_names_route_to_model_ids(self):
        from chimera.cir.executor import _make_anthropic_adapter
        seen = {}
        adapter = _make_anthropic_adapter(
            self._fake_client(seen),
            {"claude": "claude-sonnet-4-6", "gpt": "gpt-x"})
        resp = adapter("Q?", ["gpt"])
        assert seen["model"] == "gpt-x"
        assert resp.confidence == 0.9
        assert resp.answer == "Paris"

    def test_default_mapping(self):
        from chimera.cir.executor import _make_anthropic_adapter
        seen = {}
        adapter = _make_anthropic_adapter(self._fake_client(seen), None)
        adapter("Q?", ["claude"])
        assert seen["model"] == "claude-sonnet-4-6"

    def test_unknown_agent_raises_naming_it(self):
        from chimera.cir.executor import _make_anthropic_adapter
        adapter = _make_anthropic_adapter(self._fake_client({}), {"claude": "claude-sonnet-4-6"})
        with pytest.raises(ValueError, match="nope"):
            adapter("Q?", ["nope"])

    def test_empty_agents_raises(self):
        from chimera.cir.executor import _make_anthropic_adapter
        adapter = _make_anthropic_adapter(self._fake_client({}), None)
        with pytest.raises(ValueError, match="[Aa]gent"):
            adapter("Q?", [])

    def test_executor_accepts_agent_models(self):
        ex = CIRExecutor(inquiry_adapter=lambda p, a: 0.5,
                         agent_models={"claude": "claude-sonnet-4-6"})
        assert ex._agent_models == {"claude": "claude-sonnet-4-6"}
