# ChimeraLang CIR + Belief System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend ChimeraLang with a Cognitive Intermediate Representation (CIR) layer adding `belief`/`inquire`/`resolve`/`guard`/`evolve` constructs, symbol emergence, and BFT consensus — all additive, zero breakage to existing programs.

**Architecture:** New `chimera/cir/` subpackage (nodes, lower, executor, symbols) sits between parser and output. AST containing `BeliefDecl` routes to CIR path; existing `fn`/`gate`/`goal`/`reason` programs use the existing VM unchanged. Beliefs are Beta distributions, resolve uses Dempster-Shafer combination, symbols use Weisfeiler-Lehman graph hashing.

**Tech Stack:** Python 3.11+, stdlib only (uuid, hashlib, math, collections, json, time). Optional `anthropic` SDK for live `inquire` calls; mocked via callback when absent.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `chimera/cir/__init__.py` | Create | Public API re-exports |
| `chimera/cir/nodes.py` | Create | BetaDist + all CIR node/edge types |
| `chimera/cir/lower.py` | Create | AST → CIR lowering (3 passes) |
| `chimera/cir/executor.py` | Create | DS combination, BFT, FE minimization, temporal decay |
| `chimera/cir/symbols.py` | Create | WL hashing, TF-IDF, Darwinian fitness, CRDT store |
| `chimera/tokens.py` | Modify | Add 9 new keyword tokens |
| `chimera/lexer.py` | Modify | Add `:=` two-char token |
| `chimera/ast_nodes.py` | Modify | Add 6 new AST node classes |
| `chimera/parser.py` | Modify | Add parsers for belief/inquire/resolve/guard/evolve |
| `chimera/cli.py` | Modify | CIR dispatch + `--save-symbols`/`--load-symbols` flags |
| `examples/belief_reasoning.chimera` | Create | End-to-end demo |
| `tests/test_cir_nodes.py` | Create | BetaDist math, DS combination, KL divergence |
| `tests/test_cir_lower.py` | Create | All 3 lowering passes |
| `tests/test_cir_executor.py` | Create | Inquiry mock, guard, evolve convergence |
| `tests/test_symbols.py` | Create | WL hash, TF-IDF, fitness, CRDT merge |

---

## Task 1: CIR Node Types (`chimera/cir/nodes.py`)

**Files:**
- Create: `chimera/cir/nodes.py`
- Create: `chimera/cir/__init__.py` (empty for now)

- [ ] **Step 1: Write failing test**

Create `tests/test_cir_nodes.py`:

```python
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
        # variance = (2*2) / (4*4*5) = 4/80 = 0.05
        assert abs(b.variance - 0.05) < 1e-9

    def test_uniform_prior(self):
        b = BetaDist(alpha=1.0, beta=1.0)
        assert b.mean == 0.5
        assert b.variance == pytest.approx(1/12, abs=1e-9)

    def test_from_confidence(self):
        b = BetaDist.from_confidence(0.9)
        assert abs(b.mean - 0.9) < 0.01

    def test_ds_combination_no_conflict(self):
        b1 = BetaDist(alpha=9.0, beta=1.0)   # mean=0.9
        b2 = BetaDist(alpha=8.0, beta=2.0)   # mean=0.8
        combined = b1.combine_ds(b2)
        assert combined.mean > 0.8  # combined should be high confidence

    def test_ds_combination_high_conflict_raises(self):
        b1 = BetaDist(alpha=19.0, beta=1.0)   # mean=0.95 — very high
        b2 = BetaDist(alpha=1.0, beta=19.0)   # mean=0.05 — very low
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
        assert decayed.mean < b.mean  # decays toward 0.5


class TestBeliefState:
    def test_creation(self):
        b = BeliefState(name="cause", distribution=BetaDist(8.0, 2.0))
        assert b.name == "cause"
        assert b.id is not None
        assert b.distribution.mean == pytest.approx(0.8, abs=0.01)

    def test_is_stale_with_ttl(self):
        import time
        b = BeliefState(name="x", distribution=BetaDist(5.0, 5.0), ttl=0.001)
        time.sleep(0.01)
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
```

- [ ] **Step 2: Run tests — expect ImportError**

```
cd C:/Users/garza/Downloads/ChimeraLang
python -m pytest tests/test_cir_nodes.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'chimera.cir'`

- [ ] **Step 3: Create `chimera/cir/__init__.py`**

```python
"""ChimeraLang CIR — Cognitive Intermediate Representation."""
```

- [ ] **Step 4: Create `chimera/cir/nodes.py`**

```python
"""CIR node types for ChimeraLang.

Beliefs are Beta distributions (not scalars). The graph is SSA-style
with typed directed edges. Each node has a UUID for identity.
"""
from __future__ import annotations

import hashlib
import math
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


# ---------------------------------------------------------------------------
# Beta Distribution (belief representation)
# ---------------------------------------------------------------------------

@dataclass
class BetaDist:
    """Beta distribution as a belief: alpha=successes, beta=failures.

    Unlike a scalar confidence, this captures both the estimate (mean)
    and the uncertainty (variance). Low alpha+beta = little evidence.
    """
    alpha: float
    beta: float

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        s = self.alpha + self.beta
        return (self.alpha * self.beta) / (s * s * (s + 1))

    @classmethod
    def from_confidence(cls, conf: float, strength: float = 10.0) -> BetaDist:
        """Convert scalar confidence [0,1] to Beta with given pseudocount strength."""
        conf = max(1e-6, min(1.0 - 1e-6, conf))
        return cls(alpha=conf * strength, beta=(1.0 - conf) * strength)

    @classmethod
    def uniform(cls) -> BetaDist:
        return cls(alpha=1.0, beta=1.0)

    def combine_ds(self, other: BetaDist, conflict_threshold: float = 0.8) -> BetaDist:
        """Dempster-Shafer-inspired evidence combination.

        Detects conflict (when one source is very high and the other very low)
        and raises ValueError rather than silently producing garbage.
        Combination: Bayesian update — add pseudocounts (conjugate prior).
        K = normalized cross-conflict term; if high, sources are irreconcilable.
        """
        # Cross-conflict: how much does each belief conflict with the other?
        # High alpha1 * high beta2 (or vice versa) = conflict
        total1 = self.alpha + self.beta
        total2 = other.alpha + other.beta
        m1_yes = self.alpha / total1    # belief mass on "true"
        m1_no = self.beta / total1      # belief mass on "false"
        m2_yes = other.alpha / total2
        m2_no = other.beta / total2

        # Conflict: one says very yes, other says very no
        K = m1_yes * m2_no + m1_no * m2_yes
        if K > conflict_threshold:
            raise ValueError(
                f"DS conflict K={K:.3f} exceeds threshold {conflict_threshold:.2f}. "
                f"Sources irreconcilable: mean1={self.mean:.3f}, mean2={other.mean:.3f}"
            )

        # Bayesian conjugate update (add pseudocounts, subtract shared prior)
        new_alpha = self.alpha + other.alpha - 1.0
        new_beta = self.beta + other.beta - 1.0
        return BetaDist(alpha=max(new_alpha, 1e-6), beta=max(new_beta, 1e-6))

    def kl_divergence(self, other: BetaDist) -> float:
        """KL(self || other) approximated via normal approximation for speed.

        For near-Gaussian Betas (alpha, beta > 1), the normal approx is accurate.
        Returns 0.0 for identical distributions.
        """
        if abs(self.alpha - other.alpha) < 1e-9 and abs(self.beta - other.beta) < 1e-9:
            return 0.0
        # Normal approximation: KL(N(μ1,σ1²) || N(μ2,σ2²))
        mu1, s1 = self.mean, math.sqrt(max(self.variance, 1e-12))
        mu2, s2 = other.mean, math.sqrt(max(other.variance, 1e-12))
        return (
            math.log(s2 / s1)
            + (s1 ** 2 + (mu1 - mu2) ** 2) / (2 * s2 ** 2)
            - 0.5
        )

    def decay(self, factor: float) -> BetaDist:
        """Regression toward uniform prior Beta(1,1) by factor in [0,1].

        factor=0 → unchanged, factor=1 → fully uniform.
        """
        factor = max(0.0, min(1.0, factor))
        new_alpha = self.alpha * (1 - factor) + 1.0 * factor
        new_beta = self.beta * (1 - factor) + 1.0 * factor
        return BetaDist(alpha=new_alpha, beta=new_beta)


# ---------------------------------------------------------------------------
# Edge kinds
# ---------------------------------------------------------------------------

class EdgeKind(Enum):
    INQUIRY = auto()
    CONSENSUS = auto()
    VALIDATION = auto()
    EVOLUTION = auto()
    CAUSAL = auto()


@dataclass
class CIREdge:
    source_id: str
    target_id: str
    kind: EdgeKind
    strength: float = 1.0


# ---------------------------------------------------------------------------
# CIR Node base
# ---------------------------------------------------------------------------

@dataclass
class CIRNode:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    belief: BetaDist | None = None
    provenance: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Concrete node types
# ---------------------------------------------------------------------------

@dataclass
class InquiryNode(CIRNode):
    prompt: str = ""
    agents: list[str] = field(default_factory=list)
    ttl: float | None = None


@dataclass
class ConsensusNode(CIRNode):
    threshold: float = 0.8
    strategy: str = "dempster_shafer"    # dempster_shafer | majority | bft
    input_ids: list[str] = field(default_factory=list)


@dataclass
class ValidationNode(CIRNode):
    max_risk: float = 0.2
    strategy: str = "both"              # mean | variance | both
    target_id: str = ""


@dataclass
class EvolutionNode(CIRNode):
    condition: str = "stable"
    max_iter: int = 3
    subgraph_entry: str = ""


@dataclass
class MetaNode(CIRNode):
    """Graph introspection node — emits ReasoningTrace."""


# ---------------------------------------------------------------------------
# BeliefState: named belief in the belief store
# ---------------------------------------------------------------------------

@dataclass
class BeliefState:
    name: str
    distribution: BetaDist
    ttl: float | None = None
    provenance: list[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = field(default_factory=time.time)
    node_id: str = ""

    def is_stale(self) -> bool:
        if self.ttl is None:
            return False
        return (time.time() - self.timestamp) > self.ttl

    def decayed(self) -> BeliefState:
        """Return a version of this belief decayed toward uniform if stale."""
        if not self.is_stale() or self.ttl is None:
            return self
        overshoot = (time.time() - self.timestamp - self.ttl) / max(self.ttl, 1e-6)
        factor = min(overshoot, 1.0)
        return BeliefState(
            name=self.name,
            distribution=self.distribution.decay(factor),
            ttl=self.ttl,
            provenance=[*self.provenance, f"decayed(factor={factor:.2f})"],
            id=self.id,
            timestamp=self.timestamp,
            node_id=self.node_id,
        )


# ---------------------------------------------------------------------------
# CIRGraph: the full graph
# ---------------------------------------------------------------------------

@dataclass
class CIRGraph:
    nodes: dict[str, CIRNode] = field(default_factory=dict)
    edges: list[CIREdge] = field(default_factory=list)
    entry_id: str = ""
    belief_store: dict[str, BeliefState] = field(default_factory=dict)
    emit_ids: list[str] = field(default_factory=list)   # node ids whose belief to emit

    def add_node(self, node: CIRNode) -> None:
        self.nodes[node.id] = node

    def add_edge(self, edge: CIREdge) -> None:
        self.edges.append(edge)

    def successors(self, node_id: str) -> list[CIRNode]:
        return [
            self.nodes[e.target_id]
            for e in self.edges
            if e.source_id == node_id and e.target_id in self.nodes
        ]

    def predecessors(self, node_id: str) -> list[CIRNode]:
        return [
            self.nodes[e.source_id]
            for e in self.edges
            if e.target_id == node_id and e.source_id in self.nodes
        ]

    def topological_order(self) -> list[str]:
        """Kahn's algorithm — raises on cycles."""
        in_degree: dict[str, int] = {nid: 0 for nid in self.nodes}
        for edge in self.edges:
            if edge.target_id in in_degree:
                in_degree[edge.target_id] += 1
        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        order: list[str] = []
        while queue:
            nid = queue.pop(0)
            order.append(nid)
            for succ in self.successors(nid):
                in_degree[succ.id] -= 1
                if in_degree[succ.id] == 0:
                    queue.append(succ.id)
        if len(order) != len(self.nodes):
            raise ValueError("CIRGraph contains a cycle")
        return order

    def wl_hash(self, rounds: int = 3) -> str:
        """Weisfeiler-Lehman graph hash for structural identity."""
        labels: dict[str, str] = {
            nid: node.__class__.__name__ for nid, node in self.nodes.items()
        }
        for _ in range(rounds):
            new_labels: dict[str, str] = {}
            for nid in self.nodes:
                neighbor_parts = sorted(
                    labels.get(e.target_id, "") + ":" + e.kind.name
                    for e in self.edges if e.source_id == nid
                )
                combined = labels[nid] + "|" + "|".join(neighbor_parts)
                new_labels[nid] = hashlib.md5(combined.encode()).hexdigest()[:8]
            labels = new_labels
        sorted_labels = "|".join(sorted(labels.values()))
        return hashlib.sha256(sorted_labels.encode()).hexdigest()[:16]
```

- [ ] **Step 5: Run tests**

```
python -m pytest tests/test_cir_nodes.py -v
```

Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git -C C:/Users/garza/Downloads/ChimeraLang add chimera/cir/__init__.py chimera/cir/nodes.py tests/test_cir_nodes.py
git -C C:/Users/garza/Downloads/ChimeraLang commit -m "feat: add CIR node types with BetaDist, DS combination, WL hash"
```

---

## Task 2: AST → CIR Lowering (`chimera/cir/lower.py`)

**Files:**
- Create: `chimera/cir/lower.py`
- Create: `tests/test_cir_lower.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_cir_lower.py`:

```python
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
                inquire_expr=InquireExpr(
                    prompt="Why?", agents=["claude"], ttl=None
                ),
            )
        )
        lowering = CIRLowering()
        graph = lowering.lower(prog)
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
        # belief y is declared but never resolved, guarded, evolved, or emitted
        prog = make_program(
            BeliefDecl(name="used", inquire_expr=InquireExpr(prompt="A", agents=[], ttl=None)),
            BeliefDecl(name="unused", inquire_expr=InquireExpr(prompt="B", agents=[], ttl=None)),
            EmitStmt(value=Identifier(name="used")),
        )
        graph = CIRLowering().lower(prog)
        # unused belief's InquiryNode should be eliminated
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
        # A belief with very low pseudocounts has high variance — should be flagged
        prog = make_program(
            BeliefDecl(name="risky", inquire_expr=InquireExpr(prompt="X", agents=[], ttl=None)),
            GuardStmt(target="risky", max_risk=0.05, strategy="variance"),
        )
        lowering = CIRLowering()
        graph = lowering.lower(prog)
        # Warnings should note that initial prior has high variance
        assert any("variance" in w.lower() for w in lowering.warnings)
```

- [ ] **Step 2: Run tests — expect ImportError**

```
python -m pytest tests/test_cir_lower.py -v 2>&1 | head -20
```

Expected: `ImportError: cannot import name 'BeliefDecl' from 'chimera.ast_nodes'`
(We haven't added the new AST nodes yet — that's ok, lowering tests will fail until Task 4.)

- [ ] **Step 3: Create `chimera/cir/lower.py`**

```python
"""AST → CIR Lowering for ChimeraLang.

Three sequential passes:
  1. Structural: map AST nodes to CIR nodes with typed edges
  2. Dead belief elimination: remove nodes with no emit/resolve downstream
  3. Belief flow analysis: forward-propagate BetaDist, flag high-variance paths
"""
from __future__ import annotations

from chimera.cir.nodes import (
    BeliefState, BetaDist, CIREdge, CIRGraph,
    ConsensusNode, EdgeKind, EvolutionNode, InquiryNode,
    MetaNode, ValidationNode,
)


class LoweringError(Exception):
    pass


class CIRLowering:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def lower(self, program: object) -> CIRGraph:  # program: chimera.ast_nodes.Program
        graph = CIRGraph()
        self.warnings = []

        # --- Pass 1: Structural lowering ---
        self._pass_structural(program, graph)

        # --- Pass 2: Dead belief elimination ---
        self._pass_dead_belief_elimination(program, graph)

        # --- Pass 3: Belief flow analysis ---
        self._pass_belief_flow_analysis(graph)

        return graph

    # ------------------------------------------------------------------
    # Pass 1: Structural
    # ------------------------------------------------------------------

    def _pass_structural(self, program: object, graph: CIRGraph) -> None:
        from chimera.ast_nodes import (
            BeliefDecl, EmitStmt, EvolveStmt, GuardStmt, ResolveStmt,
        )

        belief_node_map: dict[str, str] = {}  # belief_name → InquiryNode.id

        for decl in program.declarations:  # type: ignore[attr-defined]
            if isinstance(decl, BeliefDecl):
                inq = InquiryNode(
                    prompt=decl.inquire_expr.prompt,
                    agents=list(decl.inquire_expr.agents),
                    ttl=decl.inquire_expr.ttl,
                )
                graph.add_node(inq)
                belief_node_map[decl.name] = inq.id
                graph.belief_store[decl.name] = BeliefState(
                    name=decl.name,
                    distribution=BetaDist.uniform(),
                    ttl=decl.inquire_expr.ttl,
                    node_id=inq.id,
                )
                if not graph.entry_id:
                    graph.entry_id = inq.id

            elif isinstance(decl, ResolveStmt):
                source_id = belief_node_map.get(decl.target, "")
                cons = ConsensusNode(
                    threshold=decl.threshold,
                    strategy=decl.strategy,
                    input_ids=[source_id] if source_id else [],
                )
                graph.add_node(cons)
                if source_id:
                    graph.add_edge(CIREdge(
                        source_id=source_id, target_id=cons.id,
                        kind=EdgeKind.CONSENSUS,
                    ))
                belief_node_map[decl.target] = cons.id

            elif isinstance(decl, GuardStmt):
                source_id = belief_node_map.get(decl.target, "")
                val = ValidationNode(
                    max_risk=decl.max_risk,
                    strategy=decl.strategy,
                    target_id=source_id,
                )
                graph.add_node(val)
                if source_id:
                    graph.add_edge(CIREdge(
                        source_id=source_id, target_id=val.id,
                        kind=EdgeKind.VALIDATION,
                    ))
                belief_node_map[decl.target] = val.id

            elif isinstance(decl, EvolveStmt):
                source_id = belief_node_map.get(decl.target, "")
                evo = EvolutionNode(
                    condition=decl.condition,
                    max_iter=decl.max_iter,
                    subgraph_entry=source_id,
                )
                graph.add_node(evo)
                if source_id:
                    graph.add_edge(CIREdge(
                        source_id=source_id, target_id=evo.id,
                        kind=EdgeKind.EVOLUTION,
                    ))
                belief_node_map[decl.target] = evo.id

            elif isinstance(decl, EmitStmt):
                from chimera.ast_nodes import Identifier
                if isinstance(decl.value, Identifier):
                    node_id = belief_node_map.get(decl.value.name, "")
                    if node_id:
                        graph.emit_ids.append(node_id)

    # ------------------------------------------------------------------
    # Pass 2: Dead belief elimination
    # ------------------------------------------------------------------

    def _pass_dead_belief_elimination(self, program: object, graph: CIRGraph) -> None:
        """Remove InquiryNodes whose belief is never consumed downstream."""
        if not graph.emit_ids:
            return  # nothing emitted — skip (don't delete everything)

        # BFS backwards from emit nodes to find all live nodes
        live: set[str] = set(graph.emit_ids)
        queue = list(graph.emit_ids)
        while queue:
            nid = queue.pop(0)
            for edge in graph.edges:
                if edge.target_id == nid and edge.source_id not in live:
                    live.add(edge.source_id)
                    queue.append(edge.source_id)

        # Remove dead nodes
        dead = [nid for nid in list(graph.nodes) if nid not in live]
        for nid in dead:
            del graph.nodes[nid]
        graph.edges = [e for e in graph.edges if e.source_id in live and e.target_id in live]

        # Remove dead belief store entries
        dead_beliefs = [
            name for name, bs in graph.belief_store.items()
            if bs.node_id not in live
        ]
        for name in dead_beliefs:
            del graph.belief_store[name]

    # ------------------------------------------------------------------
    # Pass 3: Belief flow analysis
    # ------------------------------------------------------------------

    def _pass_belief_flow_analysis(self, graph: CIRGraph) -> None:
        """Forward-propagate BetaDist. Flag high-variance paths."""
        VARIANCE_WARN_THRESHOLD = 0.08

        for nid in graph.nodes:
            node = graph.nodes[nid]
            if isinstance(node, ValidationNode):
                # Find the belief that feeds this validation node
                preds = graph.predecessors(nid)
                for pred in preds:
                    bs = next(
                        (b for b in graph.belief_store.values() if b.node_id == pred.id),
                        None,
                    )
                    if bs is not None:
                        if bs.distribution.variance > VARIANCE_WARN_THRESHOLD:
                            self.warnings.append(
                                f"belief '{bs.name}' has high variance "
                                f"({bs.distribution.variance:.4f}) before validation — "
                                f"consider more inquiry agents or tighter prior"
                            )
```

- [ ] **Step 4: Add new AST nodes to `chimera/ast_nodes.py`** (required for lowering tests)

Append to `chimera/ast_nodes.py`:

```python
# ---------------------------------------------------------------------------
# CIR / Belief constructs (additive — do not modify existing nodes)
# ---------------------------------------------------------------------------

@dataclass
class InquireExpr(ASTNode):
    prompt: str = ""
    agents: list[str] = field(default_factory=list)
    ttl: float | None = None


@dataclass
class BeliefDecl(Statement):
    name: str = ""
    type_ann: TypeExpr | None = None
    inquire_expr: InquireExpr | None = None


@dataclass
class ResolveStmt(Statement):
    target: str = ""
    threshold: float = 0.8
    strategy: str = "dempster_shafer"


@dataclass
class GuardStmt(Statement):
    target: str = ""
    max_risk: float = 0.2
    strategy: str = "both"


@dataclass
class EvolveStmt(Statement):
    target: str = ""
    condition: str = "stable"
    max_iter: int = 3


@dataclass
class SymbolDecl(Declaration):
    name: str = ""
    body: list[Statement] = field(default_factory=list)
```

- [ ] **Step 5: Run lowering tests**

```
python -m pytest tests/test_cir_lower.py -v
```

Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git -C C:/Users/garza/Downloads/ChimeraLang add chimera/cir/lower.py chimera/ast_nodes.py tests/test_cir_lower.py
git -C C:/Users/garza/Downloads/ChimeraLang commit -m "feat: add CIR lowering (3 passes) + new AST nodes"
```

---

## Task 3: CIR Executor (`chimera/cir/executor.py`)

**Files:**
- Create: `chimera/cir/executor.py`
- Create: `tests/test_cir_executor.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_cir_executor.py`:

```python
"""Tests for CIR executor."""
import pytest
from chimera.cir.nodes import (
    BetaDist, BeliefState, CIREdge, CIRGraph,
    ConsensusNode, EdgeKind, EvolutionNode, InquiryNode, ValidationNode,
)
from chimera.cir.executor import CIRExecutor, GuardViolation, CIRResult


def mock_adapter(prompt: str, agents: list[str]) -> float:
    """Mock: always returns 0.85 confidence."""
    return 0.85


def low_confidence_adapter(prompt: str, agents: list[str]) -> float:
    return 0.3


class TestInquiry:
    def test_inquiry_produces_belief(self):
        graph = CIRGraph()
        inq = InquiryNode(prompt="Why?", agents=["claude"])
        graph.add_node(inq)
        graph.entry_id = inq.id
        graph.belief_store["cause"] = BeliefState(
            name="cause", distribution=BetaDist.uniform(), node_id=inq.id
        )
        graph.emit_ids = [inq.id]
        executor = CIRExecutor(inquiry_adapter=mock_adapter)
        result = executor.run(graph)
        assert "cause" in result.beliefs
        assert result.beliefs["cause"].mean > 0.5

    def test_inquiry_uses_adapter_confidence(self):
        graph = CIRGraph()
        inq = InquiryNode(prompt="Q", agents=[])
        graph.add_node(inq)
        graph.entry_id = inq.id
        graph.belief_store["x"] = BeliefState(
            name="x", distribution=BetaDist.uniform(), node_id=inq.id
        )
        executor = CIRExecutor(inquiry_adapter=lambda p, a: 0.7)
        executor.run(graph)
        assert graph.belief_store["x"].distribution.mean == pytest.approx(0.7, abs=0.05)


class TestGuard:
    def test_guard_passes_high_confidence(self):
        graph = CIRGraph()
        inq = InquiryNode(prompt="Q", agents=[])
        val = ValidationNode(max_risk=0.2, strategy="mean", target_id=inq.id)
        graph.add_node(inq)
        graph.add_node(val)
        graph.add_edge(CIREdge(source_id=inq.id, target_id=val.id, kind=EdgeKind.VALIDATION))
        graph.entry_id = inq.id
        graph.belief_store["x"] = BeliefState(
            name="x", distribution=BetaDist.uniform(), node_id=inq.id
        )
        executor = CIRExecutor(inquiry_adapter=lambda p, a: 0.9)
        result = executor.run(graph)
        assert not result.guard_violations

    def test_guard_raises_on_low_confidence(self):
        graph = CIRGraph()
        inq = InquiryNode(prompt="Q", agents=[])
        val = ValidationNode(max_risk=0.2, strategy="mean", target_id=inq.id)
        graph.add_node(inq)
        graph.add_node(val)
        graph.add_edge(CIREdge(source_id=inq.id, target_id=val.id, kind=EdgeKind.VALIDATION))
        graph.entry_id = inq.id
        graph.belief_store["x"] = BeliefState(
            name="x", distribution=BetaDist.uniform(), node_id=inq.id
        )
        executor = CIRExecutor(inquiry_adapter=low_confidence_adapter)
        result = executor.run(graph)
        assert len(result.guard_violations) > 0


class TestEvolve:
    def test_evolve_converges(self):
        graph = CIRGraph()
        inq = InquiryNode(prompt="Q", agents=[])
        evo = EvolutionNode(condition="stable", max_iter=3, subgraph_entry=inq.id)
        graph.add_node(inq)
        graph.add_node(evo)
        graph.add_edge(CIREdge(source_id=inq.id, target_id=evo.id, kind=EdgeKind.EVOLUTION))
        graph.entry_id = inq.id
        graph.belief_store["x"] = BeliefState(
            name="x", distribution=BetaDist.uniform(), node_id=inq.id
        )
        executor = CIRExecutor(inquiry_adapter=mock_adapter)
        result = executor.run(graph)
        assert result.evolution_iters <= 3
        assert result.converged


class TestTemporalDecay:
    def test_stale_belief_decayed(self):
        import time
        graph = CIRGraph()
        inq = InquiryNode(prompt="Q", agents=[])
        graph.add_node(inq)
        graph.entry_id = inq.id
        # TTL of 0.001 seconds — will be stale by the time executor runs
        graph.belief_store["x"] = BeliefState(
            name="x",
            distribution=BetaDist(alpha=9.0, beta=1.0),  # mean=0.9
            ttl=0.001,
            node_id=inq.id,
        )
        time.sleep(0.02)
        executor = CIRExecutor(inquiry_adapter=mock_adapter)
        result = executor.run(graph)
        # After decay, mean should be lower than 0.9
        # (decayed toward 0.5 due to staleness)
        assert result.beliefs.get("x", BetaDist(9, 1)).mean < 0.9


class TestReasoningTrace:
    def test_trace_captures_steps(self):
        graph = CIRGraph()
        inq = InquiryNode(prompt="What?", agents=[])
        graph.add_node(inq)
        graph.entry_id = inq.id
        graph.belief_store["x"] = BeliefState(
            name="x", distribution=BetaDist.uniform(), node_id=inq.id
        )
        executor = CIRExecutor(inquiry_adapter=mock_adapter)
        result = executor.run(graph)
        assert len(result.trace) > 0
```

- [ ] **Step 2: Run tests — expect ImportError**

```
python -m pytest tests/test_cir_executor.py -v 2>&1 | head -10
```

Expected: `ImportError: cannot import name 'CIRExecutor' from 'chimera.cir.executor'`

- [ ] **Step 3: Create `chimera/cir/executor.py`**

```python
"""CIR Executor for ChimeraLang.

Runs a CIRGraph in topological order:
  - InquiryNode  → calls inquiry_adapter (Claude or mock)
  - ConsensusNode → Dempster-Shafer combination + BFT validation
  - ValidationNode → guard: mean >= (1-max_risk) and/or variance <= 0.05
  - EvolutionNode → fixed-point loop minimizing KL divergence
  - Temporal decay → stale beliefs regressed toward uniform prior
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from chimera.cir.nodes import (
    BetaDist, BeliefState, CIRGraph, ConsensusNode,
    EdgeKind, EvolutionNode, InquiryNode, ValidationNode,
)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class CIRResult:
    beliefs: dict[str, BetaDist] = field(default_factory=dict)
    emitted: list[tuple[str, BetaDist]] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)
    guard_violations: list[str] = field(default_factory=list)
    evolution_iters: int = 0
    converged: bool = True
    duration_ms: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)


class GuardViolation(Exception):
    pass


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

InquiryAdapter = Callable[[str, list[str]], float]


def _default_mock_adapter(prompt: str, agents: list[str]) -> float:
    """Mock adapter: returns 0.75 for any prompt."""
    return 0.75


class CIRExecutor:
    """Execute a CIRGraph and produce beliefs + trace.

    inquiry_adapter: callable(prompt, agents) → confidence float [0,1].
    If None, tries the Anthropic SDK; falls back to mock.
    """

    def __init__(
        self,
        inquiry_adapter: InquiryAdapter | None = None,
        strict_guard: bool = False,
    ) -> None:
        self._adapter = inquiry_adapter or self._resolve_adapter()
        self._strict = strict_guard

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self, graph: CIRGraph) -> CIRResult:
        start = time.perf_counter()
        result = CIRResult()

        # Apply temporal decay to all stale beliefs before execution
        self._apply_temporal_decay(graph, result)

        # Execute in topological order
        try:
            order = graph.topological_order()
        except ValueError as e:
            result.trace.append(f"[error] {e}")
            result.converged = False
            return result

        for nid in order:
            node = graph.nodes[nid]
            if isinstance(node, InquiryNode):
                self._exec_inquiry(node, graph, result)
            elif isinstance(node, ConsensusNode):
                self._exec_consensus(node, graph, result)
            elif isinstance(node, ValidationNode):
                self._exec_validation(node, graph, result)
            elif isinstance(node, EvolutionNode):
                self._exec_evolution(node, graph, result)

        # Collect emitted beliefs
        for emit_id in graph.emit_ids:
            bs = next(
                (b for b in graph.belief_store.values() if b.node_id == emit_id),
                None,
            )
            if bs is not None:
                result.emitted.append((bs.name, bs.distribution))
                result.beliefs[bs.name] = bs.distribution

        # Populate all beliefs
        for name, bs in graph.belief_store.items():
            result.beliefs[name] = bs.distribution

        result.duration_ms = (time.perf_counter() - start) * 1000
        result.meta["node_count"] = len(graph.nodes)
        result.meta["edge_count"] = len(graph.edges)
        return result

    # ------------------------------------------------------------------
    # Node handlers
    # ------------------------------------------------------------------

    def _exec_inquiry(self, node: InquiryNode, graph: CIRGraph, result: CIRResult) -> None:
        result.trace.append(f"[inquiry] prompt={node.prompt!r} agents={node.agents}")
        try:
            conf = float(self._adapter(node.prompt, node.agents))
            conf = max(0.0, min(1.0, conf))
        except Exception as e:
            result.trace.append(f"[inquiry] adapter error: {e} — using 0.5")
            conf = 0.5

        dist = BetaDist.from_confidence(conf)
        result.trace.append(f"[inquiry] confidence={conf:.3f} → Beta({dist.alpha:.1f},{dist.beta:.1f})")

        # Update the belief store entry for this node
        bs = next(
            (b for b in graph.belief_store.values() if b.node_id == node.id), None
        )
        if bs is not None:
            bs.distribution = dist
            bs.provenance.append(f"inquired(conf={conf:.3f})")

    def _exec_consensus(self, node: ConsensusNode, graph: CIRGraph, result: CIRResult) -> None:
        result.trace.append(f"[consensus] strategy={node.strategy} threshold={node.threshold}")
        # Collect input beliefs from predecessors
        preds = graph.predecessors(node.id)
        input_beliefs: list[BetaDist] = []
        for pred in preds:
            bs = next(
                (b for b in graph.belief_store.values() if b.node_id == pred.id), None
            )
            if bs is not None:
                input_beliefs.append(bs.distribution)

        if len(input_beliefs) == 0:
            result.trace.append("[consensus] no input beliefs — skipping")
            return

        if len(input_beliefs) == 1:
            combined = input_beliefs[0]
        else:
            combined = input_beliefs[0]
            for other in input_beliefs[1:]:
                try:
                    combined = combined.combine_ds(other)
                except ValueError as e:
                    result.trace.append(f"[consensus] DS conflict: {e}")
                    result.guard_violations.append(f"consensus conflict: {e}")
                    return

        # BFT: check if threshold met
        if combined.mean < node.threshold:
            msg = (
                f"consensus mean {combined.mean:.3f} below threshold {node.threshold}"
            )
            result.trace.append(f"[consensus] BELOW THRESHOLD — {msg}")
            result.guard_violations.append(msg)

        result.trace.append(f"[consensus] combined mean={combined.mean:.3f} variance={combined.variance:.4f}")

        # Update belief store entry for this consensus node
        for bs in graph.belief_store.values():
            if bs.node_id in node.input_ids:
                bs.distribution = combined
                bs.node_id = node.id
                bs.provenance.append(f"consensus({node.strategy},mean={combined.mean:.3f})")

    def _exec_validation(self, node: ValidationNode, graph: CIRGraph, result: CIRResult) -> None:
        result.trace.append(f"[guard] max_risk={node.max_risk} strategy={node.strategy}")
        bs = next(
            (b for b in graph.belief_store.values() if b.node_id == node.target_id),
            None,
        )
        if bs is None:
            result.trace.append("[guard] no belief to validate — skipping")
            return

        dist = bs.distribution
        violations: list[str] = []

        if node.strategy in ("mean", "both"):
            required_mean = 1.0 - node.max_risk
            if dist.mean < required_mean:
                violations.append(
                    f"mean {dist.mean:.3f} < required {required_mean:.3f}"
                )

        if node.strategy in ("variance", "both"):
            max_variance = 0.05
            if dist.variance > max_variance:
                violations.append(
                    f"variance {dist.variance:.4f} > allowed {max_variance}"
                )

        if violations:
            msg = f"guard '{bs.name}' FAILED: {'; '.join(violations)}"
            result.trace.append(f"[guard] VIOLATION — {msg}")
            result.guard_violations.append(msg)
            if self._strict:
                raise GuardViolation(msg)
        else:
            result.trace.append(f"[guard] PASSED — mean={dist.mean:.3f} variance={dist.variance:.4f}")
            bs.node_id = node.id

    def _exec_evolution(self, node: EvolutionNode, graph: CIRGraph, result: CIRResult) -> None:
        result.trace.append(f"[evolve] condition={node.condition} max_iter={node.max_iter}")
        bs = next(
            (b for b in graph.belief_store.values() if b.node_id == node.subgraph_entry),
            None,
        )
        if bs is None:
            result.trace.append("[evolve] no belief to evolve — skipping")
            return

        prior = bs.distribution
        KL_THRESHOLD = 0.001
        converged = False

        for i in range(node.max_iter):
            result.evolution_iters += 1
            # Re-run inquiry to get updated belief
            inq_node = graph.nodes.get(node.subgraph_entry)
            if isinstance(inq_node, InquiryNode):
                try:
                    conf = float(self._adapter(inq_node.prompt, inq_node.agents))
                    conf = max(0.0, min(1.0, conf))
                except Exception:
                    conf = bs.distribution.mean

                posterior = BetaDist.from_confidence(conf)
                kl = posterior.kl_divergence(prior)
                result.trace.append(
                    f"[evolve] iter={i+1} KL={kl:.5f} mean={posterior.mean:.3f}"
                )
                if kl < KL_THRESHOLD:
                    converged = True
                    bs.distribution = posterior
                    bs.provenance.append(f"evolved(iters={i+1},converged=True)")
                    break
                prior = posterior
                bs.distribution = posterior
            else:
                break

        result.converged = converged
        bs.node_id = node.id
        if not converged:
            result.trace.append(f"[evolve] did not converge in {node.max_iter} iterations")
            bs.provenance.append(f"evolved(iters={node.max_iter},converged=False)")

    # ------------------------------------------------------------------
    # Temporal decay
    # ------------------------------------------------------------------

    def _apply_temporal_decay(self, graph: CIRGraph, result: CIRResult) -> None:
        for name, bs in graph.belief_store.items():
            if bs.is_stale():
                decayed = bs.decayed()
                graph.belief_store[name] = decayed
                result.trace.append(
                    f"[decay] belief '{name}' stale — decayed from "
                    f"mean={bs.distribution.mean:.3f} to {decayed.distribution.mean:.3f}"
                )

    # ------------------------------------------------------------------
    # Adapter resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_adapter() -> InquiryAdapter:
        """Try Anthropic SDK; fall back to mock."""
        try:
            import anthropic  # type: ignore[import]
            client = anthropic.Anthropic()

            def _anthropic_adapter(prompt: str, agents: list[str]) -> float:
                msg = client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=256,
                    messages=[{
                        "role": "user",
                        "content": (
                            f"{prompt}\n\nRespond with a single JSON object: "
                            '{"answer": "<brief answer>", "confidence": <0.0-1.0>}'
                        ),
                    }],
                )
                import json, re
                text = msg.content[0].text
                m = re.search(r'"confidence"\s*:\s*([0-9.]+)', text)
                return float(m.group(1)) if m else 0.7

            return _anthropic_adapter
        except (ImportError, Exception):
            return _default_mock_adapter
```

- [ ] **Step 4: Run executor tests**

```
python -m pytest tests/test_cir_executor.py -v
```

Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git -C C:/Users/garza/Downloads/ChimeraLang add chimera/cir/executor.py tests/test_cir_executor.py
git -C C:/Users/garza/Downloads/ChimeraLang commit -m "feat: CIR executor with DS consensus, guard, evolve, temporal decay"
```

---

## Task 4: Symbol Emergence (`chimera/cir/symbols.py`)

**Files:**
- Create: `chimera/cir/symbols.py`
- Create: `tests/test_symbols.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_symbols.py`:

```python
"""Tests for Symbol Emergence System."""
import json, os, tempfile
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
        s1 = store.register(ga, prompts=["Q1"])
        s2 = store.register(gb, prompts=["Q2"])
        assert s1.wl_hash == s2.wl_hash
        assert store._symbols[s1.wl_hash].usage_count == 2

    def test_fitness_computation(self):
        store = SymbolStore()
        g = make_graph_a()
        sym = store.register(g, prompts=["What is gravity?"])
        assert 0.0 <= sym.fitness <= 1.0

    def test_darwinian_prune_removes_low_fitness(self):
        store = SymbolStore()
        # Register many symbols; some will have low usage
        for i in range(10):
            g = CIRGraph()
            inq = InquiryNode(prompt=f"unique question {i} zxq", agents=[])
            g.add_node(inq)
            store.register(g, prompts=[f"unique question {i} zxq"])
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
        inq2 = InquiryNode(prompt="unrelated", agents=[])
        gb.add_node(inq2)
        s1.register(ga, prompts=["P1"])
        s2.register(gb, prompts=["P2"])
        s1.merge(s2)
        assert len(s1._symbols) == 2
```

- [ ] **Step 2: Run tests — expect ImportError**

```
python -m pytest tests/test_symbols.py -v 2>&1 | head -10
```

Expected: `ImportError: cannot import name 'SymbolStore' from 'chimera.cir.symbols'`

- [ ] **Step 3: Create `chimera/cir/symbols.py`**

```python
"""Symbol Emergence System for ChimeraLang CIR.

Symbols are reusable CIR subgraphs identified by Weisfeiler-Lehman hash.
Semantically similar symbols (TF-IDF cosine > 0.7) are merged.
Multi-objective fitness drives Darwinian competition.
CRDT G-Set store enables conflict-free distributed persistence.
"""
from __future__ import annotations

import json
import math
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from chimera.cir.nodes import CIRGraph, CIRNode, EdgeKind


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

    def compute_fitness(
        self,
        total_nodes_in_graph: int,
    ) -> float:
        node_count = len(self.node_types)
        compression_ratio = 1.0 - (node_count / max(total_nodes_in_graph, 1))
        structural_depth = min(node_count / 5.0, 1.0)
        semantic_coherence = tfidf_similarity(self.prompts, self.prompts) if len(self.prompts) > 1 else 0.5
        usage_score = min(self.usage_count / 20.0, 1.0)
        self.fitness = (
            0.35 * compression_ratio
            + 0.25 * structural_depth
            + 0.20 * semantic_coherence
            + 0.20 * usage_score
        )
        return self.fitness

    def to_dict(self) -> dict[str, Any]:
        return {
            "wl_hash": self.wl_hash,
            "node_types": self.node_types,
            "edge_types": self.edge_types,
            "prompts": self.prompts,
            "usage_count": self.usage_count,
            "fitness": self.fitness,
            "created_at": self.created_at,
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
            sim = tfidf_similarity(prompts, existing.prompts)
            if sim >= self.SEMANTIC_MERGE_THRESHOLD:
                existing.usage_count += 1
                existing.prompts.extend(prompts)
                existing.compute_fitness(total_nodes)
                # Also register under the new hash pointing to same symbol
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
        # Mutate survivors: edge weights shift ±0.05
        import random
        for sym in self._symbols.values():
            sym.edge_weight_mutation += random.uniform(-0.05, 0.05)
        return to_prune

    def merge(self, other: SymbolStore) -> None:
        """CRDT merge: union of symbol sets (G-Set). Conflict-free."""
        for h, sym in other._symbols.items():
            if h not in self._symbols:
                self._symbols[h] = sym
            else:
                # Existing: keep higher usage count
                self._symbols[h].usage_count = max(
                    self._symbols[h].usage_count, sym.usage_count
                )

    def save_symbols(self, path: str) -> None:
        """Serialize symbol store to JSON."""
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for sym in self._symbols.values():
            if sym.wl_hash not in seen:
                unique.append(sym.to_dict())
                seen.add(sym.wl_hash)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"version": 1, "symbols": unique}, f, indent=2)

    def load_symbols(self, path: str) -> None:
        """Load and merge symbols from JSON (CRDT merge)."""
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
```

- [ ] **Step 4: Run symbol tests**

```
python -m pytest tests/test_symbols.py -v
```

Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git -C C:/Users/garza/Downloads/ChimeraLang add chimera/cir/symbols.py tests/test_symbols.py
git -C C:/Users/garza/Downloads/ChimeraLang commit -m "feat: symbol emergence with WL hash, TF-IDF, Darwinian competition, CRDT store"
```

---

## Task 5: New Tokens + Lexer `:=`

**Files:**
- Modify: `chimera/tokens.py`
- Modify: `chimera/lexer.py`

- [ ] **Step 1: Add 9 new keywords to `chimera/tokens.py`**

In `TokenKind` enum, after `DETECT = auto()`, add:

```python
    # Keywords — CIR / belief constructs
    BELIEF = auto()
    INQUIRE = auto()
    RESOLVE = auto()
    GUARD = auto()
    EVOLVE = auto()
    SYMBOL = auto()
    WITH = auto()
    AGAINST = auto()
    UNTIL = auto()

    # New operator
    WALRUS = auto()    # :=
```

In `KEYWORDS` dict, add:

```python
    "belief": TokenKind.BELIEF,
    "inquire": TokenKind.INQUIRE,
    "resolve": TokenKind.RESOLVE,
    "guard": TokenKind.GUARD,
    "evolve": TokenKind.EVOLVE,
    "symbol": TokenKind.SYMBOL,
    "with": TokenKind.WITH,
    "against": TokenKind.AGAINST,
    "until": TokenKind.UNTIL,
```

- [ ] **Step 2: Add `:=` to lexer `TWO_CHAR` in `chimera/lexer.py`**

In `_read_symbol`, in the `TWO_CHAR` dict, add:

```python
            ":=": TokenKind.WALRUS,
```

- [ ] **Step 3: Verify lex works**

```
python -c "
from chimera.lexer import Lexer
tokens = Lexer('belief x := inquire { prompt: \"Why?\" }').tokenize()
print([t.kind.name for t in tokens if t.kind.name != 'NEWLINE'])
"
```

Expected output includes: `['BELIEF', 'IDENT', 'WALRUS', 'INQUIRE', 'LBRACE', 'IDENT', 'COLON', 'STRING_LIT', 'RBRACE', 'EOF']`

- [ ] **Step 4: Run existing tests to verify no breakage**

```
python -m pytest tests/ -v --ignore=tests/test_cir_nodes.py --ignore=tests/test_cir_lower.py --ignore=tests/test_cir_executor.py --ignore=tests/test_symbols.py
```

Expected: All existing tests still pass.

- [ ] **Step 5: Commit**

```bash
git -C C:/Users/garza/Downloads/ChimeraLang add chimera/tokens.py chimera/lexer.py
git -C C:/Users/garza/Downloads/ChimeraLang commit -m "feat: add 9 new CIR keyword tokens + := walrus operator"
```

---

## Task 6: Parser Extensions

**Files:**
- Modify: `chimera/parser.py`

- [ ] **Step 1: Add CIR imports to parser.py**

In parser.py, add to the `from chimera.ast_nodes import (...)` block:

```python
    BeliefDecl,
    EvolveStmt,
    GuardStmt,
    InquireExpr,
    ResolveStmt,
    SymbolDecl,
```

- [ ] **Step 2: Add CIR dispatch to `_parse_top_level`**

In `_parse_top_level`, before `return self._parse_statement()`, add:

```python
        if kind == TokenKind.BELIEF:
            return self._parse_belief()
        if kind == TokenKind.RESOLVE:
            return self._parse_resolve()
        if kind == TokenKind.GUARD:
            return self._parse_guard()
        if kind == TokenKind.EVOLVE:
            return self._parse_evolve()
        if kind == TokenKind.SYMBOL:
            return self._parse_symbol_decl()
```

Also add to `_parse_statement` before the final `self._parse_expr()` fallthrough:

```python
        if self._check(TokenKind.RESOLVE):
            return self._parse_resolve()
        if self._check(TokenKind.GUARD):
            return self._parse_guard()
        if self._check(TokenKind.EVOLVE):
            return self._parse_evolve()
```

- [ ] **Step 3: Add parser methods to `Parser` class**

Add these methods to the `Parser` class (before `_parse_expr`):

```python
    # ------------------------------------------------------------------
    # CIR / Belief constructs
    # ------------------------------------------------------------------

    def _parse_belief(self) -> BeliefDecl:
        """belief <name> := inquire { prompt: "...", agents: [...], ttl: N }"""
        span = self._expect(TokenKind.BELIEF).span
        name = self._expect(TokenKind.IDENT, "Expected belief name").value
        self._expect(TokenKind.WALRUS, "Expected ':=' after belief name")
        inquire_expr = self._parse_inquire_expr()
        self._expect_line_end()
        return BeliefDecl(name=name, inquire_expr=inquire_expr, span=span)

    def _parse_inquire_expr(self) -> InquireExpr:
        """inquire { prompt: "...", agents: [...], ttl: N }"""
        self._expect(TokenKind.INQUIRE, "Expected 'inquire'")
        self._expect(TokenKind.LBRACE, "Expected '{' after 'inquire'")
        self._skip_newlines()
        prompt = ""
        agents: list[str] = []
        ttl: float | None = None
        while not self._check(TokenKind.RBRACE) and not self._check(TokenKind.EOF):
            self._skip_newlines()
            if self._check(TokenKind.RBRACE):
                break
            key = self._advance().value
            self._expect(TokenKind.COLON, f"Expected ':' after '{key}'")
            if key == "prompt":
                prompt = self._expect(TokenKind.STRING_LIT, "Expected prompt string").value
            elif key == "agents":
                self._expect(TokenKind.LBRACKET, "Expected '[' for agents list")
                while not self._check(TokenKind.RBRACKET):
                    agents.append(self._advance().value)
                    self._match(TokenKind.COMMA)
                self._expect(TokenKind.RBRACKET, "Expected ']' after agents")
            elif key == "ttl":
                tok = self._advance()
                ttl = float(tok.value)
            self._match(TokenKind.COMMA)
            self._skip_newlines()
        self._expect(TokenKind.RBRACE, "Expected '}' to close inquire block")
        return InquireExpr(prompt=prompt, agents=agents, ttl=ttl)

    def _parse_resolve(self) -> ResolveStmt:
        """resolve <name> with consensus { threshold: 0.8, strategy: dempster_shafer }"""
        self._expect(TokenKind.RESOLVE)
        target = self._expect(TokenKind.IDENT, "Expected belief name").value
        self._expect(TokenKind.WITH, "Expected 'with' after belief name")
        self._advance()  # consume 'consensus' (treated as ident)
        threshold = 0.8
        strategy = "dempster_shafer"
        if self._check(TokenKind.LBRACE):
            self._advance()
            self._skip_newlines()
            while not self._check(TokenKind.RBRACE) and not self._check(TokenKind.EOF):
                self._skip_newlines()
                if self._check(TokenKind.RBRACE):
                    break
                key = self._advance().value
                self._expect(TokenKind.COLON)
                if key == "threshold":
                    threshold = float(self._advance().value)
                elif key == "strategy":
                    strategy = self._advance().value
                self._match(TokenKind.COMMA)
                self._skip_newlines()
            self._expect(TokenKind.RBRACE, "Expected '}' to close resolve block")
        self._expect_line_end()
        return ResolveStmt(target=target, threshold=threshold, strategy=strategy)

    def _parse_guard(self) -> GuardStmt:
        """guard <name> against hallucination { max_risk: 0.2, strategy: both }"""
        self._expect(TokenKind.GUARD)
        target = self._expect(TokenKind.IDENT, "Expected belief name").value
        self._expect(TokenKind.AGAINST, "Expected 'against' after belief name")
        self._advance()  # consume 'hallucination' (ident)
        max_risk = 0.2
        strategy = "both"
        if self._check(TokenKind.LBRACE):
            self._advance()
            self._skip_newlines()
            while not self._check(TokenKind.RBRACE) and not self._check(TokenKind.EOF):
                self._skip_newlines()
                if self._check(TokenKind.RBRACE):
                    break
                key = self._advance().value
                self._expect(TokenKind.COLON)
                if key == "max_risk":
                    max_risk = float(self._advance().value)
                elif key == "strategy":
                    strategy = self._advance().value
                self._match(TokenKind.COMMA)
                self._skip_newlines()
            self._expect(TokenKind.RBRACE, "Expected '}' to close guard block")
        self._expect_line_end()
        return GuardStmt(target=target, max_risk=max_risk, strategy=strategy)

    def _parse_evolve(self) -> EvolveStmt:
        """evolve <name> until stable { max_iter: 3 }"""
        self._expect(TokenKind.EVOLVE)
        target = self._expect(TokenKind.IDENT, "Expected belief name").value
        self._expect(TokenKind.UNTIL, "Expected 'until' after belief name")
        condition = self._advance().value  # e.g. 'stable'
        max_iter = 3
        if self._check(TokenKind.LBRACE):
            self._advance()
            self._skip_newlines()
            while not self._check(TokenKind.RBRACE) and not self._check(TokenKind.EOF):
                self._skip_newlines()
                if self._check(TokenKind.RBRACE):
                    break
                key = self._advance().value
                self._expect(TokenKind.COLON)
                if key == "max_iter":
                    max_iter = int(self._advance().value)
                self._match(TokenKind.COMMA)
                self._skip_newlines()
            self._expect(TokenKind.RBRACE, "Expected '}' to close evolve block")
        self._expect_line_end()
        return EvolveStmt(target=target, condition=condition, max_iter=max_iter)

    def _parse_symbol_decl(self) -> SymbolDecl:
        """symbol <name> { ... }"""
        self._expect(TokenKind.SYMBOL)
        name = self._expect(TokenKind.IDENT, "Expected symbol name").value
        self._expect(TokenKind.LBRACE, "Expected '{' after symbol name")
        self._expect_line_end()
        body: list[Statement] = []
        while not self._check(TokenKind.RBRACE) and not self._check(TokenKind.EOF):
            self._skip_newlines()
            if self._check(TokenKind.RBRACE):
                break
            body.append(self._parse_statement())
            self._skip_newlines()
        self._expect(TokenKind.RBRACE, "Expected '}' to close symbol block")
        self._expect_line_end()
        return SymbolDecl(name=name, body=body)
```

- [ ] **Step 4: Test parsing**

```
python -c "
from chimera.lexer import Lexer
from chimera.parser import Parser
src = '''
belief cause := inquire {
  prompt: \"What causes black holes?\",
  agents: [claude],
  ttl: 3600
}

resolve cause with consensus { threshold: 0.8, strategy: dempster_shafer }
guard cause against hallucination { max_risk: 0.2, strategy: both }
evolve cause until stable { max_iter: 3 }

emit cause
'''
tokens = Lexer(src).tokenize()
prog = Parser(tokens).parse()
print('Declarations:', [type(d).__name__ for d in prog.declarations])
"
```

Expected: `Declarations: ['BeliefDecl', 'ResolveStmt', 'GuardStmt', 'EvolveStmt', 'EmitStmt']`

- [ ] **Step 5: Run all tests**

```
python -m pytest tests/ -v
```

Expected: All tests pass (existing + new).

- [ ] **Step 6: Commit**

```bash
git -C C:/Users/garza/Downloads/ChimeraLang add chimera/parser.py
git -C C:/Users/garza/Downloads/ChimeraLang commit -m "feat: parser extensions for belief/inquire/resolve/guard/evolve/symbol"
```

---

## Task 7: CLI Dispatch + `chimera/cir/__init__.py` Public API

**Files:**
- Modify: `chimera/cli.py`
- Modify: `chimera/cir/__init__.py`

- [ ] **Step 1: Update `chimera/cir/__init__.py`**

```python
"""ChimeraLang CIR — Cognitive Intermediate Representation.

Public API:
    from chimera.cir import run_cir, CIRResult, SymbolStore
"""
from chimera.cir.executor import CIRExecutor, CIRResult, GuardViolation
from chimera.cir.lower import CIRLowering, LoweringError
from chimera.cir.nodes import (
    BetaDist, BeliefState, CIRGraph, CIRNode,
    ConsensusNode, EdgeKind, EvolutionNode,
    InquiryNode, MetaNode, ValidationNode,
)
from chimera.cir.symbols import SymbolStore, Symbol, extract_subgraphs

__all__ = [
    "CIRExecutor", "CIRResult", "GuardViolation",
    "CIRLowering", "LoweringError",
    "BetaDist", "BeliefState", "CIRGraph", "CIRNode",
    "ConsensusNode", "EdgeKind", "EvolutionNode",
    "InquiryNode", "MetaNode", "ValidationNode",
    "SymbolStore", "Symbol", "extract_subgraphs",
]


def run_cir(
    program: object,
    *,
    save_symbols: str | None = None,
    load_symbols: str | None = None,
    inquiry_adapter=None,
    strict_guard: bool = False,
) -> CIRResult:
    """Full CIR pipeline: lower + optionally load symbols + execute + optionally save symbols."""
    store = SymbolStore()
    if load_symbols:
        store.load_symbols(load_symbols)

    lowering = CIRLowering()
    graph = lowering.lower(program)

    executor = CIRExecutor(inquiry_adapter=inquiry_adapter, strict_guard=strict_guard)
    result = executor.run(graph)

    # Register the graph as a symbol after execution
    prompts = [
        n.prompt for n in graph.nodes.values() if isinstance(n, InquiryNode)
    ]
    if prompts:
        store.register(graph, prompts=prompts)

    if save_symbols:
        store.save_symbols(save_symbols)

    result.meta["lowering_warnings"] = lowering.warnings
    return result
```

- [ ] **Step 2: Add CIR dispatch to `cmd_run` in `chimera/cli.py`**

In `cmd_run`, replace:

```python
    vm = ChimeraVM()
    result = vm.execute(program)
```

With:

```python
    # Route to CIR path if program uses belief constructs
    from chimera.ast_nodes import BeliefDecl
    uses_cir = any(isinstance(d, BeliefDecl) for d in program.declarations)

    if uses_cir:
        from chimera.cir import run_cir
        save_sym = next((a.split("=")[1] for a in rest if a.startswith("--save-symbols=")), None)
        load_sym = next((a.split("=")[1] for a in rest if a.startswith("--load-symbols=")), None)
        cir_result = run_cir(program, save_symbols=save_sym, load_symbols=load_sym)
        if cir_result.emitted:
            for name, dist in cir_result.emitted:
                print(f"  emit: {name}  [mean={dist.mean:.3f} variance={dist.variance:.4f}]")
        if show_trace and cir_result.trace:
            print("\n— CIR Reasoning Trace —")
            for entry in cir_result.trace:
                print(f"  {entry}")
        if cir_result.guard_violations:
            print("\n— Guard Violations —")
            for v in cir_result.guard_violations:
                print(f"  {v}")
        print(f"\nchimera: {path} — CIR executed in {cir_result.duration_ms:.1f}ms")
        if cir_result.guard_violations:
            sys.exit(1)
        return
    else:
        vm = ChimeraVM()
        result = vm.execute(program)
```

Also update `USAGE` string to add:

```
  chimera run   <file.chimera> [--save-symbols=out.json] [--load-symbols=in.json]
```

- [ ] **Step 3: Verify CLI routing**

```
python -c "
from chimera.lexer import Lexer
from chimera.parser import Parser
from chimera.ast_nodes import BeliefDecl
src = 'belief x := inquire { prompt: \"Q\", agents: [] }'
tokens = Lexer(src).tokenize()
prog = Parser(tokens).parse()
uses_cir = any(isinstance(d, BeliefDecl) for d in prog.declarations)
print('uses_cir:', uses_cir)
"
```

Expected: `uses_cir: True`

- [ ] **Step 4: Run full test suite**

```
python -m pytest tests/ -v
```

Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git -C C:/Users/garza/Downloads/ChimeraLang add chimera/cir/__init__.py chimera/cli.py
git -C C:/Users/garza/Downloads/ChimeraLang commit -m "feat: CLI CIR dispatch + public run_cir API + --save/load-symbols flags"
```

---

## Task 8: Example + End-to-End Smoke Test

**Files:**
- Create: `examples/belief_reasoning.chimera`

- [ ] **Step 1: Create example file**

Create `examples/belief_reasoning.chimera`:

```chimera
// belief_reasoning.chimera
// Demonstrates the full CIR pipeline:
// belief → inquire → resolve → guard → evolve → emit

belief cause := inquire {
  prompt: "What are the primary causes of black hole formation?",
  agents: [claude],
  ttl: 3600
}

resolve cause with consensus { threshold: 0.75, strategy: dempster_shafer }
guard cause against hallucination { max_risk: 0.25, strategy: both }
evolve cause until stable { max_iter: 3 }

emit cause
```

- [ ] **Step 2: Run the example (mock mode — no API key needed)**

```
cd C:/Users/garza/Downloads/ChimeraLang
python -m chimera.cli run examples/belief_reasoning.chimera --trace
```

Expected output (approximately):
```
  emit: cause  [mean=0.750 variance=0.0167]

— CIR Reasoning Trace —
  [inquiry] prompt='What are the primary causes of black hole formation?'...
  [guard] PASSED — mean=0.750 ...
  [evolve] iter=1 KL=...

chimera: examples/belief_reasoning.chimera — CIR executed in X.Xms
```

- [ ] **Step 3: Run all tests one final time**

```
python -m pytest tests/ -v
```

Expected: All tests pass.

- [ ] **Step 4: Commit example**

```bash
git -C C:/Users/garza/Downloads/ChimeraLang add examples/belief_reasoning.chimera
git -C C:/Users/garza/Downloads/ChimeraLang commit -m "feat: add belief_reasoning.chimera example"
```

---

## Task 9: Push to GitHub

- [ ] **Step 1: Verify git log**

```
git -C C:/Users/garza/Downloads/ChimeraLang log --oneline -10
```

Expected: 8 commits since the spec was added.

- [ ] **Step 2: Push**

```
git -C C:/Users/garza/Downloads/ChimeraLang push origin main
```

- [ ] **Step 3: Verify on GitHub**

```
gh repo view fernandogarzaaa/ChimeraLang --json pushedAt,defaultBranchRef
```

Expected: `pushedAt` reflects today's date.

---

## Self-Review

**Spec coverage check:**
- ✅ CIR graph (nodes.py — InquiryNode, ConsensusNode, ValidationNode, EvolutionNode, MetaNode)
- ✅ BetaDist (not scalar confidence) with DS combination + KL divergence
- ✅ Dead belief elimination (lower.py pass 2)
- ✅ Belief flow analysis (lower.py pass 3)
- ✅ Temporal decay (executor.py `_apply_temporal_decay`)
- ✅ Dempster-Shafer combination with conflict detection (nodes.py `combine_ds`)
- ✅ BFT: consensus below threshold logged as violation (executor.py)
- ✅ Free energy minimization in evolve (KL convergence loop)
- ✅ Metacognitive layer (MetaNode defined; executor emits meta stats)
- ✅ WL hashing (nodes.py `CIRGraph.wl_hash`)
- ✅ TF-IDF semantic similarity (symbols.py `tfidf_similarity`)
- ✅ Multi-objective fitness (4 axes in Symbol.compute_fitness)
- ✅ Darwinian competition (symbols.py `darwinian_prune`)
- ✅ CRDT symbol store (symbols.py `SymbolStore.merge`)
- ✅ save_symbols / load_symbols (symbols.py + CLI flags)
- ✅ claude_adapter integration (executor.py `_resolve_adapter` uses anthropic SDK)
- ✅ CLI dispatch (cli.py routes on BeliefDecl presence)
- ✅ Zero breakage to existing programs (dispatch is additive)
- ✅ New example (belief_reasoning.chimera)
- ✅ 4 test files with 80%+ coverage targets

**Type consistency:**
- `BetaDist.combine_ds` → used in `executor.py _exec_consensus` ✅
- `BeliefState.node_id` → used in lower.py and executor.py ✅
- `CIRGraph.wl_hash()` → called in symbols.py `SymbolStore.register` ✅
- `CIRGraph.topological_order()` → called in executor.py `run` ✅
- `extract_subgraphs(graph, min_length)` → used in test_symbols.py ✅

**No placeholders:** All steps have complete code. ✅
