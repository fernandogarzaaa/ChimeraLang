"""Tests for CIR node types and BetaDist math."""
import math
import pytest
from chimera.cir.nodes import (
    BetaDist, BeliefState, InquiryNode, ConsensusNode,
    ValidationNode, EvolutionNode, CIREdge, EdgeKind, CIRGraph,
)


class TestBetaDist:
    def test_mean(self):
        b = BetaDist(alpha=8.0, beta=2.0)
        assert abs(b.mean - 0.8) < 1e-9

    def test_variance(self):
        b = BetaDist(alpha=2.0, beta=2.0)
        assert abs(b.variance - 0.05) < 1e-9

    def test_uniform_prior(self):
        b = BetaDist(alpha=1.0, beta=1.0)
        assert b.mean == 0.5
        assert b.variance == pytest.approx(1/12, abs=1e-9)

    def test_from_confidence(self):
        b = BetaDist.from_confidence(0.9)
        assert abs(b.mean - 0.9) < 0.01

    def test_ds_combination_no_conflict(self):
        b1 = BetaDist(alpha=9.0, beta=1.0)
        b2 = BetaDist(alpha=8.0, beta=2.0)
        combined = b1.combine_ds(b2)
        assert combined.mean > 0.8

    def test_ds_combination_high_conflict_raises(self):
        b1 = BetaDist(alpha=19.0, beta=1.0)
        b2 = BetaDist(alpha=1.0, beta=19.0)
        with pytest.raises(ValueError, match="conflict"):
            b1.combine_ds(b2, conflict_threshold=0.5)

    def test_kl_divergence_identical(self):
        b = BetaDist(alpha=5.0, beta=2.0)
        assert b.kl_divergence(b) == pytest.approx(0.0, abs=1e-9)

    def test_kl_divergence_different(self):
        b1 = BetaDist(alpha=9.0, beta=1.0)
        b2 = BetaDist(alpha=1.0, beta=1.0)
        assert b1.kl_divergence(b2) > 0.0

    def test_decay_toward_uniform(self):
        b = BetaDist(alpha=9.0, beta=1.0)
        decayed = b.decay(factor=0.5)
        assert decayed.mean < b.mean


class TestBeliefState:
    def test_creation(self):
        b = BeliefState(name="cause", distribution=BetaDist(8.0, 2.0))
        assert b.name == "cause"
        assert b.id is not None
        assert b.distribution.mean == pytest.approx(0.8, abs=0.01)

    def test_is_stale_with_ttl(self):
        import time
        b = BeliefState(name="x", distribution=BetaDist(5.0, 5.0), ttl=0.001)
        time.sleep(0.02)
        assert b.is_stale()

    def test_not_stale_without_ttl(self):
        b = BeliefState(name="x", distribution=BetaDist(5.0, 5.0), ttl=None)
        assert not b.is_stale()


class TestCIRGraph:
    def test_add_nodes_and_edges(self):
        g = CIRGraph()
        inq = InquiryNode(prompt="Why?", agents=["claude"])
        cons = ConsensusNode(threshold=0.8, strategy="dempster_shafer")
        g.add_node(inq)
        g.add_node(cons)
        g.add_edge(CIREdge(source_id=inq.id, target_id=cons.id, kind=EdgeKind.INQUIRY))
        assert len(g.nodes) == 2
        assert len(g.edges) == 1

    def test_graph_successors(self):
        g = CIRGraph()
        a = InquiryNode(prompt="A", agents=[])
        b = ConsensusNode(threshold=0.8, strategy="majority")
        g.add_node(a)
        g.add_node(b)
        g.add_edge(CIREdge(source_id=a.id, target_id=b.id, kind=EdgeKind.CONSENSUS))
        succs = g.successors(a.id)
        assert len(succs) == 1
        assert succs[0].id == b.id

    def test_topological_order(self):
        g = CIRGraph()
        a = InquiryNode(prompt="A", agents=[])
        b = ConsensusNode(threshold=0.8, strategy="majority")
        g.add_node(a)
        g.add_node(b)
        g.add_edge(CIREdge(source_id=a.id, target_id=b.id, kind=EdgeKind.CONSENSUS))
        order = g.topological_order()
        assert order.index(a.id) < order.index(b.id)

    def test_wl_hash_stable(self):
        g = CIRGraph()
        inq = InquiryNode(prompt="Q", agents=[])
        g.add_node(inq)
        assert g.wl_hash() == g.wl_hash()
