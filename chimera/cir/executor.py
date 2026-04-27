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
    return 0.75


class CIRExecutor:
    """Execute a CIRGraph and produce beliefs + trace.

    inquiry_adapter: callable(prompt, agents) -> confidence float [0,1].
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

        self._apply_temporal_decay(graph, result)

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

        for emit_id in graph.emit_ids:
            bs = next(
                (b for b in graph.belief_store.values() if b.node_id == emit_id),
                None,
            )
            if bs is not None:
                result.emitted.append((bs.name, bs.distribution))
                result.beliefs[bs.name] = bs.distribution

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
        result.trace.append(
            f"[inquiry] confidence={conf:.3f} -> Beta({dist.alpha:.1f},{dist.beta:.1f})"
        )

        bs = next(
            (b for b in graph.belief_store.values() if b.node_id == node.id), None
        )
        if bs is not None:
            bs.distribution = dist
            bs.provenance.append(f"inquired(conf={conf:.3f})")

    def _exec_consensus(self, node: ConsensusNode, graph: CIRGraph, result: CIRResult) -> None:
        result.trace.append(f"[consensus] strategy={node.strategy} threshold={node.threshold}")
        preds = graph.predecessors(node.id)
        input_beliefs: list[tuple[str, BetaDist]] = []
        for pred in preds:
            bs = next(
                (b for b in graph.belief_store.values() if b.node_id == pred.id), None
            )
            if bs is not None:
                input_beliefs.append((bs.name, bs.distribution))

        if not input_beliefs:
            result.trace.append("[consensus] no input beliefs — skipping")
            return

        combined = input_beliefs[0][1]
        for _, other_dist in input_beliefs[1:]:
            try:
                combined = combined.combine_ds(other_dist)
            except ValueError as e:
                result.trace.append(f"[consensus] DS conflict: {e}")
                result.guard_violations.append(f"consensus conflict: {e}")
                return

        if combined.mean < node.threshold:
            msg = f"consensus mean {combined.mean:.3f} below threshold {node.threshold}"
            result.trace.append(f"[consensus] BELOW THRESHOLD — {msg}")
            result.guard_violations.append(msg)

        result.trace.append(
            f"[consensus] combined mean={combined.mean:.3f} variance={combined.variance:.4f}"
        )

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
                violations.append(f"mean {dist.mean:.3f} < required {required_mean:.3f}")

        if node.strategy in ("variance", "both"):
            max_variance = 0.05
            if dist.variance > max_variance:
                violations.append(f"variance {dist.variance:.4f} > allowed {max_variance}")

        if violations:
            msg = f"guard '{bs.name}' FAILED: {'; '.join(violations)}"
            result.trace.append(f"[guard] VIOLATION — {msg}")
            result.guard_violations.append(msg)
            if self._strict:
                raise GuardViolation(msg)
        else:
            result.trace.append(
                f"[guard] PASSED — mean={dist.mean:.3f} variance={dist.variance:.4f}"
            )
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
        for name, bs in list(graph.belief_store.items()):
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
        try:
            import anthropic  # type: ignore[import]
            client = anthropic.Anthropic()

            def _anthropic_adapter(prompt: str, agents: list[str]) -> float:
                import json
                import re
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
                text = msg.content[0].text
                m = re.search(r'"confidence"\s*:\s*([0-9.]+)', text)
                return float(m.group(1)) if m else 0.7

            return _anthropic_adapter
        except (ImportError, Exception):
            return _default_mock_adapter
