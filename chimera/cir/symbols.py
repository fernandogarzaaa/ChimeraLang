"""Symbol Emergence System for ChimeraLang CIR.

Symbols are reusable CIR subgraphs identified by Weisfeiler-Lehman hash.
Semantically similar symbols (TF-IDF cosine > 0.7) are merged.
Multi-objective fitness drives Darwinian competition.
CRDT G-Set store enables conflict-free distributed persistence.
"""
from __future__ import annotations

import json
import math
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from chimera.cir.nodes import BetaDist, CIRGraph, InquiryNode


# ---------------------------------------------------------------------------
# TF-IDF similarity (stdlib only)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def tfidf_similarity(texts1: list[str], texts2: list[str]) -> float:
    """Cosine similarity between two bags of words."""
    bag1: Counter[str] = Counter(w for t in texts1 for w in _tokenize(t))
    bag2: Counter[str] = Counter(w for t in texts2 for w in _tokenize(t))
    if not bag1 or not bag2:
        return 0.0
    intersection = set(bag1) & set(bag2)
    dot = sum(bag1[w] * bag2[w] for w in intersection)
    norm1 = math.sqrt(sum(v * v for v in bag1.values()))
    norm2 = math.sqrt(sum(v * v for v in bag2.values()))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


# ---------------------------------------------------------------------------
# Subgraph extraction
# ---------------------------------------------------------------------------

def extract_subgraphs(graph: CIRGraph, min_length: int = 2) -> list[CIRGraph]:
    """Extract all linear chain subgraphs of length >= min_length."""
    subgraphs: list[CIRGraph] = []

    def _chains_from(start_id: str, path: list[str]) -> None:
        succs = graph.successors(start_id)
        if len(path) >= min_length:
            sub = CIRGraph()
            for nid in path:
                sub.add_node(graph.nodes[nid])
            for edge in graph.edges:
                if edge.source_id in path and edge.target_id in path:
                    sub.add_edge(edge)
            subgraphs.append(sub)
        for succ in succs:
            if succ.id not in path:
                _chains_from(succ.id, path + [succ.id])

    for nid in graph.nodes:
        _chains_from(nid, [nid])

    return subgraphs


# ---------------------------------------------------------------------------
# Symbol
# ---------------------------------------------------------------------------

@dataclass
class Symbol:
    wl_hash: str
    node_types: list[str]
    edge_types: list[str]
    prompts: list[str]
    usage_count: int = 1
    fitness: float = 0.0
    created_at: float = field(default_factory=time.time)
    edge_weight_mutation: float = 0.0
    # Cumulative belief observations attributed to this symbol. Each
    # successful CIR run feeds its posterior back here so future
    # lowerings can use a calibrated prior instead of Beta(1,1).
    prior_alpha: float = 1.0
    prior_beta: float = 1.0

    def compute_fitness(self, total_nodes_in_graph: int) -> float:
        node_count = len(self.node_types)
        compression_ratio = 1.0 - (node_count / max(total_nodes_in_graph, 1))
        structural_depth = min(node_count / 5.0, 1.0)
        semantic_coherence = (
            tfidf_similarity(self.prompts, self.prompts)
            if len(self.prompts) > 1
            else 0.5
        )
        usage_score = min(self.usage_count / 20.0, 1.0)
        self.fitness = (
            0.35 * compression_ratio
            + 0.25 * structural_depth
            + 0.20 * semantic_coherence
            + 0.20 * usage_score
        )
        return self.fitness

    def prior(self) -> BetaDist:
        """Best-guess prior for a fresh belief that matches this symbol."""
        return BetaDist(alpha=max(self.prior_alpha, 1e-6),
                        beta=max(self.prior_beta, 1e-6))

    def record_observation(self, dist: BetaDist) -> None:
        """Accumulate a posterior into this symbol's running prior.

        Uses pseudo-count addition: alpha += observed_alpha - 1,
        beta  += observed_beta  - 1 so that a Beta(1,1) observation
        (no information) is a no-op.
        """
        self.prior_alpha = max(self.prior_alpha + (dist.alpha - 1.0), 1e-6)
        self.prior_beta = max(self.prior_beta + (dist.beta - 1.0), 1e-6)

    def to_dict(self) -> dict[str, Any]:
        return {
            "wl_hash": self.wl_hash,
            "node_types": self.node_types,
            "edge_types": self.edge_types,
            "prompts": self.prompts,
            "usage_count": self.usage_count,
            "fitness": self.fitness,
            "created_at": self.created_at,
            "prior_alpha": self.prior_alpha,
            "prior_beta": self.prior_beta,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Symbol:
        return cls(
            wl_hash=d["wl_hash"],
            node_types=d["node_types"],
            edge_types=d["edge_types"],
            prompts=d["prompts"],
            usage_count=d.get("usage_count", 1),
            fitness=d.get("fitness", 0.0),
            created_at=d.get("created_at", time.time()),
            prior_alpha=d.get("prior_alpha", 1.0),
            prior_beta=d.get("prior_beta", 1.0),
        )


# ---------------------------------------------------------------------------
# SymbolStore — CRDT G-Set
# ---------------------------------------------------------------------------

class SymbolStore:
    """Grow-only set (G-Set CRDT) of symbols keyed by WL hash.

    Merge is union — no conflicts possible.
    Semantically similar symbols (TF-IDF > 0.7) are merged into one entry.
    """
    SEMANTIC_MERGE_THRESHOLD = 0.7

    def __init__(self) -> None:
        self._symbols: dict[str, Symbol] = {}

    def register(self, graph: CIRGraph, prompts: list[str]) -> Symbol:
        """Register a graph as a symbol (or increment usage if seen before)."""
        wl = graph.wl_hash()
        node_types = [n.__class__.__name__ for n in graph.nodes.values()]
        edge_types = [e.kind.name for e in graph.edges]
        total_nodes = len(graph.nodes)

        if wl in self._symbols:
            sym = self._symbols[wl]
            sym.usage_count += 1
            sym.prompts.extend(prompts)
            sym.compute_fitness(total_nodes)
            return sym

        # Check semantic similarity against existing symbols
        for existing_hash, existing in list(self._symbols.items()):
            if existing.wl_hash == wl:
                continue
            sim = tfidf_similarity(prompts, existing.prompts)
            if sim >= self.SEMANTIC_MERGE_THRESHOLD:
                existing.usage_count += 1
                existing.prompts.extend(prompts)
                existing.compute_fitness(total_nodes)
                self._symbols[wl] = existing
                return existing

        sym = Symbol(
            wl_hash=wl,
            node_types=node_types,
            edge_types=edge_types,
            prompts=list(prompts),
        )
        sym.compute_fitness(total_nodes)
        self._symbols[wl] = sym
        return sym

    def darwinian_prune(self, prune_fraction: float = 0.2) -> list[str]:
        """Prune lowest-fitness fraction of symbols. Returns pruned hashes."""
        if not self._symbols:
            return []
        unique_hashes = list({s.wl_hash: h for h, s in self._symbols.items()}.values())
        ranked = sorted(unique_hashes, key=lambda h: self._symbols[h].fitness)
        prune_count = max(1, int(len(ranked) * prune_fraction))
        to_prune = ranked[:prune_count]
        for h in to_prune:
            if h in self._symbols:
                del self._symbols[h]
        for sym in self._symbols.values():
            sym.edge_weight_mutation += random.uniform(-0.05, 0.05)
        return to_prune

    def merge(self, other: SymbolStore) -> None:
        """CRDT merge: union of symbol sets (G-Set). Conflict-free."""
        for h, sym in other._symbols.items():
            if h not in self._symbols:
                self._symbols[h] = sym
            else:
                self._symbols[h].usage_count = max(
                    self._symbols[h].usage_count, sym.usage_count
                )
                # Combine priors additively (sum of evidence) and
                # subtract one Beta(1,1) so the uninformative half
                # doesn't double-count.
                self._symbols[h].prior_alpha = max(
                    self._symbols[h].prior_alpha + sym.prior_alpha - 1.0, 1e-6
                )
                self._symbols[h].prior_beta = max(
                    self._symbols[h].prior_beta + sym.prior_beta - 1.0, 1e-6
                )

    def find_prior_for(
        self, prompt: str, *, min_similarity: float = 0.7,
    ) -> BetaDist | None:
        """Return the best-matching symbol's prior for ``prompt``, or None.

        Searches stored symbols by TF-IDF cosine similarity against their
        recorded prompts; only returns a prior when the best match clears
        ``min_similarity`` AND has at least one observation beyond the
        uninformative uniform default. Otherwise the caller should fall
        back to ``BetaDist.uniform()``.
        """
        if not self._symbols or not prompt.strip():
            return None
        best_sym: Symbol | None = None
        best_sim = 0.0
        for sym in self._symbols.values():
            sim = tfidf_similarity([prompt], sym.prompts)
            if sim > best_sim:
                best_sim = sim
                best_sym = sym
        if best_sym is None or best_sim < min_similarity:
            return None
        # Reject uninformative priors (still effectively Beta(1,1)).
        if abs(best_sym.prior_alpha - 1.0) < 1e-6 and abs(best_sym.prior_beta - 1.0) < 1e-6:
            return None
        return best_sym.prior()

    def record_observation(self, prompt: str, dist: BetaDist,
                           min_similarity: float = 0.7) -> Symbol | None:
        """Feed a posterior back to the symbol whose prompts best match.

        Returns the updated symbol, or None if no symbol clears the
        similarity floor (in which case the caller may want to register
        a new symbol for this graph instead).
        """
        if not self._symbols or not prompt.strip():
            return None
        best_sym: Symbol | None = None
        best_sim = 0.0
        for sym in self._symbols.values():
            sim = tfidf_similarity([prompt], sym.prompts)
            if sim > best_sim:
                best_sim = sim
                best_sym = sym
        if best_sym is None or best_sim < min_similarity:
            return None
        best_sym.record_observation(dist)
        return best_sym

    def save_symbols(self, path: str) -> None:
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for sym in self._symbols.values():
            if sym.wl_hash not in seen:
                unique.append(sym.to_dict())
                seen.add(sym.wl_hash)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"version": 1, "symbols": unique}, f, indent=2)

    def load_symbols(self, path: str) -> None:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for d in data.get("symbols", []):
            sym = Symbol.from_dict(d)
            if sym.wl_hash not in self._symbols:
                self._symbols[sym.wl_hash] = sym
            else:
                self._symbols[sym.wl_hash].usage_count = max(
                    self._symbols[sym.wl_hash].usage_count, sym.usage_count
                )

    def __len__(self) -> int:
        return len({s.wl_hash for s in self._symbols.values()})
