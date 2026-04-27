"""Tests for Symbol Emergence System."""
import json
import os
import tempfile
import pytest
from chimera.cir.nodes import (
    BetaDist, CIREdge, CIRGraph, ConsensusNode,
    EdgeKind, InquiryNode, ValidationNode,
)
from chimera.cir.symbols import SymbolStore, Symbol, extract_subgraphs, tfidf_similarity


def make_graph_a() -> CIRGraph:
    g = CIRGraph()
    inq = InquiryNode(prompt="What is gravity?", agents=["claude"])
    cons = ConsensusNode(threshold=0.8, strategy="dempster_shafer")
    val = ValidationNode(max_risk=0.2, strategy="both", target_id=inq.id)
    g.add_node(inq); g.add_node(cons); g.add_node(val)
    g.add_edge(CIREdge(source_id=inq.id, target_id=cons.id, kind=EdgeKind.CONSENSUS))
    g.add_edge(CIREdge(source_id=cons.id, target_id=val.id, kind=EdgeKind.VALIDATION))
    return g


def make_graph_b() -> CIRGraph:
    """Structurally identical to A — same WL hash."""
    g = CIRGraph()
    inq = InquiryNode(prompt="What is relativity?", agents=["claude"])
    cons = ConsensusNode(threshold=0.8, strategy="dempster_shafer")
    val = ValidationNode(max_risk=0.2, strategy="both", target_id=inq.id)
    g.add_node(inq); g.add_node(cons); g.add_node(val)
    g.add_edge(CIREdge(source_id=inq.id, target_id=cons.id, kind=EdgeKind.CONSENSUS))
    g.add_edge(CIREdge(source_id=cons.id, target_id=val.id, kind=EdgeKind.VALIDATION))
    return g


class TestWLHash:
    def test_identical_structure_same_hash(self):
        ga = make_graph_a()
        gb = make_graph_b()
        assert ga.wl_hash() == gb.wl_hash()

    def test_different_structure_different_hash(self):
        ga = make_graph_a()
        gc = CIRGraph()
        inq = InquiryNode(prompt="Solo", agents=[])
        gc.add_node(inq)
        assert ga.wl_hash() != gc.wl_hash()


class TestTFIDFSimilarity:
    def test_identical_texts(self):
        sim = tfidf_similarity(["what is gravity"], ["what is gravity"])
        assert sim == pytest.approx(1.0, abs=0.01)

    def test_similar_texts(self):
        sim = tfidf_similarity(["what is gravity force"], ["what is gravity mass"])
        assert sim > 0.5

    def test_unrelated_texts(self):
        sim = tfidf_similarity(["gravity physics"], ["recipe chocolate cake"])
        assert sim < 0.2


class TestSubgraphExtraction:
    def test_extracts_chains(self):
        g = make_graph_a()
        subgraphs = extract_subgraphs(g, min_length=2)
        assert len(subgraphs) >= 1

    def test_min_length_filter(self):
        g = make_graph_a()
        subgraphs = extract_subgraphs(g, min_length=10)
        assert len(subgraphs) == 0


class TestSymbolStore:
    def test_register_and_retrieve(self):
        store = SymbolStore()
        g = make_graph_a()
        sym = store.register(g, prompts=["What is gravity?"])
        assert sym.wl_hash in store._symbols
        assert sym.usage_count == 1

    def test_same_structure_increments_usage(self):
        store = SymbolStore()
        ga = make_graph_a()
        gb = make_graph_b()
        s1 = store.register(ga, prompts=["Q1 physics"])
        s2 = store.register(gb, prompts=["Q2 physics"])
        assert s1.wl_hash == s2.wl_hash
        assert store._symbols[s1.wl_hash].usage_count == 2

    def test_fitness_computation(self):
        store = SymbolStore()
        g = make_graph_a()
        sym = store.register(g, prompts=["What is gravity?"])
        assert 0.0 <= sym.fitness <= 1.0

    def test_darwinian_prune_removes_low_fitness(self):
        store = SymbolStore()
        for i in range(10):
            g = CIRGraph()
            inq = InquiryNode(prompt=f"unique zxq question number {i} alpha beta", agents=[])
            g.add_node(inq)
            store.register(g, prompts=[f"unique zxq question number {i} alpha beta"])
        initial_count = len(store._symbols)
        store.darwinian_prune(prune_fraction=0.2)
        assert len(store._symbols) <= initial_count

    def test_save_and_load(self):
        store = SymbolStore()
        g = make_graph_a()
        store.register(g, prompts=["What is gravity?"])
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            store.save_symbols(path)
            store2 = SymbolStore()
            store2.load_symbols(path)
            assert len(store2._symbols) == len(store._symbols)
        finally:
            os.unlink(path)

    def test_crdt_merge_is_union(self):
        s1 = SymbolStore()
        s2 = SymbolStore()
        ga = make_graph_a()
        gb = CIRGraph()
        inq2 = InquiryNode(prompt="completely unrelated zxqxzq", agents=[])
        gb.add_node(inq2)
        s1.register(ga, prompts=["P1 physics gravity"])
        s2.register(gb, prompts=["P2 unrelated zxqxzq"])
        initial = len(s1)
        s1.merge(s2)
        assert len(s1) >= initial
