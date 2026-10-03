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

    def combine_pseudocount(
        self, other: BetaDist, conflict_threshold: float = 0.8
    ) -> BetaDist:
        """Pseudocount addition with a K conflict check.

        This is NOT Dempster-Shafer combination in the formal sense. It
        adds the two distributions' pseudocounts
        (alpha + alpha', beta + beta') after a Dempster-style
        conflict check. The conflict coefficient
        K = m1_yes*m2_no + m1_no*m2_yes measures how irreconcilable the
        sources are; when K exceeds ``conflict_threshold`` a ValueError
        is raised rather than silently producing a merged belief.

        There is deliberately no subtract-one and no max(., 1e-6)
        clamp: every BetaDist in the system has strictly positive
        parameters (``from_confidence`` clips to (1e-6, 1-1e-6) times
        a positive strength; ``uniform()`` is (1, 1)), and sums of
        positives are positive, so the raw parameters can never be
        negative and no clamp path is reachable. The pooled mean is a
        strength-weighted average of the input means (exactly the
        arithmetic mean for equal strengths); k identical sources at
        confidence c < 1 pool to exactly c, never 1.0. The operation
        is commutative and associative.

        Behavior change: previously
        (alpha + alpha' - 1, beta + beta' - 1) with a 1e-6 clamp. That
        rule drove the raw beta to zero and then negative for
        agreeing high-confidence sources (e.g. five 0.95 sources at
        strength 10: 0.5 + 0.5 - 1.0 = 0.0, then -0.5 three times),
        and the clamp hid it, so the pooled mean saturated at exactly
        1.0. See experiments/h1_pooling/runs/2026-10-02-nebius/ERRATUM.md.
        """
        total1 = self.alpha + self.beta
        total2 = other.alpha + other.beta
        m1_yes = self.alpha / total1
        m1_no = self.beta / total1
        m2_yes = other.alpha / total2
        m2_no = other.beta / total2

        K = m1_yes * m2_no + m1_no * m2_yes
        if K > conflict_threshold:
            raise ValueError(
                f"evidence conflict K={K:.3f} exceeds threshold {conflict_threshold:.2f}. "
                f"Sources irreconcilable: mean1={self.mean:.3f}, mean2={other.mean:.3f}"
            )

        return BetaDist(alpha=self.alpha + other.alpha, beta=self.beta + other.beta)

    def combine_ds(
        self, other: BetaDist, conflict_threshold: float = 0.8
    ) -> BetaDist:
        """Backward-compatible alias for :meth:`combine_pseudocount`.

        Kept so existing callers and tests keep working. The name
        reflects the original Dempster-Shafer-inspired labeling; the
        actual operation is pseudocount addition with a K conflict check.
        """
        return self.combine_pseudocount(other, conflict_threshold=conflict_threshold)

    @staticmethod
    def max_variance_for_strength(strength: float = 10.0) -> float:
        """Largest variance any Beta with total pseudocount ``strength`` can have.

        For Beta(alpha, beta) with alpha + beta = s, the variance
        alpha*beta / (s^2 * (s+1)) is maximized at alpha = beta = s/2,
        giving 1 / (4 * (s+1)). Inquiry-produced beliefs use
        ``from_confidence`` with the default strength of 10, so their
        variance can never exceed 1/44 = 0.0227.
        """
        return 1.0 / (4.0 * (strength + 1.0))

    def kl_divergence(self, other: BetaDist) -> float:
        """KL(self || other) via normal approximation."""
        if abs(self.alpha - other.alpha) < 1e-9 and abs(self.beta - other.beta) < 1e-9:
            return 0.0
        mu1, s1 = self.mean, math.sqrt(max(self.variance, 1e-12))
        mu2, s2 = other.mean, math.sqrt(max(other.variance, 1e-12))
        return (
            math.log(s2 / s1)
            + (s1 ** 2 + (mu1 - mu2) ** 2) / (2 * s2 ** 2)
            - 0.5
        )

    def decay(self, factor: float) -> BetaDist:
        """Regression toward uniform prior Beta(1,1) by factor in [0,1]."""
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
    # Resolve strategy: "agreement" (vote-share over normalized answers;
    # the default when a belief has more than one source with answers),
    # "pooled" (BetaDist.combine_pseudocount chain: pseudocount addition
    # with a K conflict check), or "dempster_shafer" (accepted as an
    # alias of "pooled" with a lowering warning; not formal
    # Dempster-Shafer combination).
    strategy: str = "agreement"
    input_ids: list[str] = field(default_factory=list)


@dataclass
class ValidationNode(CIRNode):
    max_risk: float = 0.2
    strategy: str = "both"
    target_id: str = ""
    # Optional explicit variance limit; None keeps the legacy 0.05 default.
    max_variance: float | None = None
    # Source-level strict modifier, lowered from GuardStmt.strict. When
    # True the executor raises GuardViolation on failure. This flag (not
    # the global strict_guard run flag) determines the certificate's
    # dominance claim.
    strict: bool = False


def is_vacuous_guard(strategy: str, max_risk: float,
                     max_variance: float | None) -> bool:
    """Whether a guard's checks can never fail.

    Derivation:
    - Mean check (strategy "mean" or "both"): violation if
      score < 1.0 - max_risk. Scores are Beta means in (0, 1]. If
      max_risk >= 1.0, the threshold 1.0 - max_risk <= 0, which no
      score can fall below. The check cannot fail.
    - Variance check (strategy "variance" or "both"): violation if
      variance > max_variance. Inquiry-produced beliefs are Beta
      distributions from BetaDist.from_confidence with strength 10;
      by BetaDist.max_variance_for_strength their variance cannot
      exceed 1/(4*11) ≈ 0.0227. If an explicit max_variance is at or
      above this cap, the check cannot fire. (The legacy default of
      0.05 is also above the cap; it gets its own "unreachable"
      warning in the lowering pass and is not flagged as vacuous
      here, to preserve backward compatibility.)
    - A guard is vacuous iff every check its strategy activates is
      vacuous. A "both" guard with only one vacuous check can still
      fail via the other.
    """
    cap = BetaDist.max_variance_for_strength(10.0)
    mean_vacuous = max_risk >= 1.0
    variance_vacuous = (max_variance is not None and max_variance >= cap)
    if strategy == "mean":
        return mean_vacuous
    if strategy == "variance":
        return variance_vacuous
    if strategy == "both":
        return mean_vacuous and variance_vacuous
    return False


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
    answer: str | None = None
    # Raw likelihood from the inquiry adapter, before any seeded-prior
    # combination. run_cir feeds only this (never the posterior) back to
    # the SymbolStore so the prior is not double-counted.
    observed: BetaDist | None = None
    # Agreement resolve outputs. `agreement` is the raw vote share
    # (votes for the winner / N); both are set only by the agreement
    # strategy, and `calibrated_p` only when run_cir was given a
    # calibrator (see chimera/cir/calibration.py).
    agreement: float | None = None
    calibrated_p: float | None = None

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
            answer=self.answer,
            observed=self.observed,
            agreement=self.agreement,
            calibrated_p=self.calibrated_p,
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
    emit_ids: list[str] = field(default_factory=list)

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
