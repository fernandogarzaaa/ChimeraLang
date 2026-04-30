"""Tests for CIR executor."""
import pytest
from chimera.cir.nodes import (
    BetaDist, BeliefState, CIREdge, CIRGraph,
    ConsensusNode, EdgeKind, EvolutionNode, InquiryNode, ValidationNode,
)
from chimera.cir.executor import CIRExecutor, GuardViolation, CIRResult


def mock_adapter(prompt: str, agents: list[str]) -> float:
    return 0.85


def low_confidence_adapter(prompt: str, agents: list[str]) -> float:
    return 0.3


def _simple_graph(prompt: str = "Q") -> tuple[CIRGraph, str]:
    graph = CIRGraph()
    inq = InquiryNode(prompt=prompt, agents=[])
    graph.add_node(inq)
    graph.entry_id = inq.id
    graph.belief_store["x"] = BeliefState(
        name="x", distribution=BetaDist.uniform(), node_id=inq.id
    )
    return graph, inq.id


class TestInquiry:
    def test_inquiry_produces_belief(self):
        graph, inq_id = _simple_graph()
        graph.emit_ids = [inq_id]
        executor = CIRExecutor(inquiry_adapter=mock_adapter)
        result = executor.run(graph)
        assert "x" in result.beliefs
        assert result.beliefs["x"].mean > 0.5

    def test_inquiry_uses_adapter_confidence(self):
        graph, _ = _simple_graph()
        executor = CIRExecutor(inquiry_adapter=lambda p, a: 0.7)
        executor.run(graph)
        assert graph.belief_store["x"].distribution.mean == pytest.approx(0.7, abs=0.05)


class TestGuard:
    def test_guard_passes_high_confidence(self):
        graph, inq_id = _simple_graph()
        val = ValidationNode(max_risk=0.2, strategy="mean", target_id=inq_id)
        graph.add_node(val)
        graph.add_edge(CIREdge(source_id=inq_id, target_id=val.id, kind=EdgeKind.VALIDATION))
        executor = CIRExecutor(inquiry_adapter=lambda p, a: 0.9)
        result = executor.run(graph)
        assert not result.guard_violations

    def test_guard_raises_on_low_confidence(self):
        graph, inq_id = _simple_graph()
        val = ValidationNode(max_risk=0.2, strategy="mean", target_id=inq_id)
        graph.add_node(val)
        graph.add_edge(CIREdge(source_id=inq_id, target_id=val.id, kind=EdgeKind.VALIDATION))
        executor = CIRExecutor(inquiry_adapter=low_confidence_adapter)
        result = executor.run(graph)
        assert len(result.guard_violations) > 0


class TestEvolve:
    def test_evolve_runs_iterations(self):
        graph, inq_id = _simple_graph()
        evo = EvolutionNode(condition="stable", max_iter=3, subgraph_entry=inq_id)
        graph.add_node(evo)
        graph.add_edge(CIREdge(source_id=inq_id, target_id=evo.id, kind=EdgeKind.EVOLUTION))
        executor = CIRExecutor(inquiry_adapter=mock_adapter)
        result = executor.run(graph)
        assert result.evolution_iters <= 3

    def test_evolve_converges_with_stable_adapter(self):
        # Same adapter returns same confidence every time -> KL will be ~0 on second iter
        call_count = [0]
        def stable_adapter(p, a):
            call_count[0] += 1
            return 0.85
        graph, inq_id = _simple_graph()
        evo = EvolutionNode(condition="stable", max_iter=5, subgraph_entry=inq_id)
        graph.add_node(evo)
        graph.add_edge(CIREdge(source_id=inq_id, target_id=evo.id, kind=EdgeKind.EVOLUTION))
        executor = CIRExecutor(inquiry_adapter=stable_adapter)
        result = executor.run(graph)
        assert result.converged


class TestTemporalDecay:
    def test_stale_belief_decayed(self):
        import time
        graph, inq_id = _simple_graph()
        graph.belief_store["x"] = BeliefState(
            name="x",
            distribution=BetaDist(alpha=9.0, beta=1.0),
            ttl=0.001,
            node_id=inq_id,
        )
        time.sleep(0.02)
        executor = CIRExecutor(inquiry_adapter=mock_adapter)
        result = executor.run(graph)
        assert any("[decay]" in t for t in result.trace)


class TestReasoningTrace:
    def test_trace_captures_inquiry(self):
        graph, _ = _simple_graph("What is truth?")
        executor = CIRExecutor(inquiry_adapter=mock_adapter)
        result = executor.run(graph)
        assert any("[inquiry]" in t for t in result.trace)

    def test_meta_node_count(self):
        graph, _ = _simple_graph()
        executor = CIRExecutor(inquiry_adapter=mock_adapter)
        result = executor.run(graph)
        assert result.meta["node_count"] == 1


class TestInquiryResponse:
    """Adapter contract: float (legacy) and InquiryResponse (new) both work."""

    def test_legacy_float_adapter_still_works(self):
        graph, inq_id = _simple_graph()
        graph.emit_ids = [inq_id]
        executor = CIRExecutor(inquiry_adapter=lambda p, a: 0.82)
        result = executor.run(graph)
        assert result.beliefs["x"].mean == pytest.approx(0.82, abs=0.05)
        # No answer text means result.answers stays empty.
        assert "x" not in result.answers

    def test_inquiry_response_threads_answer_text(self):
        from chimera.cir.executor import InquiryResponse
        graph, inq_id = _simple_graph()
        graph.emit_ids = [inq_id]

        def adapter(prompt: str, agents: list[str]) -> InquiryResponse:
            return InquiryResponse(confidence=0.9, answer="42")

        executor = CIRExecutor(inquiry_adapter=adapter)
        result = executor.run(graph)
        assert result.answers["x"] == "42"
        assert graph.belief_store["x"].answer == "42"
        assert result.beliefs["x"].mean == pytest.approx(0.9, abs=0.05)

    def test_dict_adapter_response_is_normalized(self):
        graph, inq_id = _simple_graph()
        graph.emit_ids = [inq_id]
        executor = CIRExecutor(
            inquiry_adapter=lambda p, a: {"confidence": 0.6, "answer": "maybe"}
        )
        result = executor.run(graph)
        assert result.answers["x"] == "maybe"
        assert result.beliefs["x"].mean == pytest.approx(0.6, abs=0.05)

    def test_garbage_adapter_response_falls_back_to_prior(self):
        graph, inq_id = _simple_graph()
        executor = CIRExecutor(inquiry_adapter=lambda p, a: object())
        result = executor.run(graph)
        # Falls back to 0.5 prior, no answer captured.
        assert result.beliefs["x"].mean == pytest.approx(0.5, abs=0.05)
        assert "x" not in result.answers
