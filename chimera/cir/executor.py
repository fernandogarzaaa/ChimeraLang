"""CIR Executor for ChimeraLang.

Runs a CIRGraph in topological order:
  - InquiryNode  → calls inquiry_adapter (Claude or mock)
  - ConsensusNode → pseudocount addition with K conflict check + threshold check
  - ValidationNode → guard: mean >= (1-max_risk) and/or variance <= limit
  - EvolutionNode → fixed-point loop minimizing KL divergence
  - Temporal decay → stale beliefs regressed toward uniform prior
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Union

from chimera.cir.nodes import (
    BetaDist, BeliefState, CIRGraph, ConsensusNode,
    EdgeKind, EvolutionNode, InquiryNode, ValidationNode,
)
from chimera.cir.agreement import (
    Comparator as AgreementComparator,
    Normalizer as AnswerNormalizer,
    resolve_agreement,
)
from chimera.cir.calibration import LogisticCalibrator


# ---------------------------------------------------------------------------
# Result / response types
# ---------------------------------------------------------------------------

@dataclass
class InquiryResponse:
    """Structured adapter response — the answer text plus its confidence.

    Adapters may also return a bare ``float`` for backward compatibility;
    in that case ``answer`` is left as ``None``.
    """
    confidence: float
    answer: str | None = None


@dataclass
class CIRResult:
    beliefs: dict[str, BetaDist] = field(default_factory=dict)
    emitted: list[tuple[str, BetaDist]] = field(default_factory=list)
    answers: dict[str, str] = field(default_factory=dict)
    # Calibrated probabilities for emitted beliefs, present only when the
    # run was given a calibrator (see chimera/cir/calibration.py).
    calibrated: dict[str, float] = field(default_factory=dict)
    trace: list[str] = field(default_factory=list)
    guard_violations: list[str] = field(default_factory=list)
    # Structured per-guard record, one entry per ValidationNode executed:
    # {"belief", "strategy", "max_risk", "max_variance", "score_source",
    #  "score", "passed", "violations"}. score_source is "calibrated" when
    # the guard judged calibrated_p, else "uncalibrated".
    validations: list[dict] = field(default_factory=list)
    evolution_iters: int = 0
    converged: bool = True
    duration_ms: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)


class GuardViolation(Exception):
    pass


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

InquiryAdapter = Callable[[str, list[str]], Union[float, InquiryResponse]]


def _default_mock_adapter(prompt: str, agents: list[str]) -> InquiryResponse:
    return InquiryResponse(confidence=0.75, answer=f"<mock answer for {prompt!r}>")


# Default agent name -> model id mapping for the Anthropic adapter.
# Explicit and overridable; the adapter never silently falls back to
# one model for an unknown agent.
DEFAULT_AGENT_MODELS: dict[str, str] = {"claude": "claude-sonnet-4-6"}


def _make_anthropic_adapter(client: Any, agent_models: dict[str, str] | None) -> InquiryAdapter:
    """Build the default Anthropic inquiry adapter around an existing client.

    ``agent_models`` maps agent names to model ids and is merged over
    :data:`DEFAULT_AGENT_MODELS`. A call with an unknown or missing
    agent name raises ``ValueError`` naming the problem; the adapter
    never silently substitutes one model for another.
    """
    models = dict(DEFAULT_AGENT_MODELS)
    if agent_models:
        models.update(agent_models)

    def _anthropic_adapter(prompt: str, agents: list[str]) -> InquiryResponse:
        import json
        import re
        if not agents:
            raise ValueError(
                "default Anthropic adapter requires at least one agent name "
                f"(known agents: {sorted(models)}); pass agents=['claude'] "
                "or configure agent_models"
            )
        unknown = [a for a in agents if a not in models]
        if unknown:
            raise ValueError(
                f"unknown agent(s) {unknown}; known agents: {sorted(models)}; "
                "refusing to silently fall back to one model"
            )
        model = models[agents[0]]
        msg = client.messages.create(
            model=model,
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

        # Prefer a clean JSON parse; fall back to regex if the model
        # wrapped the JSON in prose. Capture both fields regardless.
        answer: str | None = None
        confidence: float = 0.7
        try:
            obj_match = re.search(r"\{.*\}", text, re.DOTALL)
            if obj_match:
                parsed = json.loads(obj_match.group(0))
                if isinstance(parsed, dict):
                    if "confidence" in parsed:
                        confidence = float(parsed["confidence"])
                    if "answer" in parsed and parsed["answer"] is not None:
                        answer = str(parsed["answer"])
                    return InquiryResponse(confidence=confidence, answer=answer)
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        conf_m = re.search(r'"confidence"\s*:\s*([0-9.]+)', text)
        if conf_m:
            confidence = float(conf_m.group(1))
        ans_m = re.search(r'"answer"\s*:\s*"([^"]*)"', text)
        if ans_m:
            answer = ans_m.group(1)
        return InquiryResponse(confidence=confidence, answer=answer)

    return _anthropic_adapter


def _normalize_response(raw: Any) -> InquiryResponse:
    """Coerce an adapter's return value into InquiryResponse.

    Accepts ``float``/``int``, ``InquiryResponse``, or a ``dict`` with at
    least a ``confidence`` key. Anything else falls back to a 0.5 prior
    so the run continues.
    """
    if isinstance(raw, InquiryResponse):
        return raw
    if isinstance(raw, (int, float)):
        return InquiryResponse(confidence=float(raw), answer=None)
    if isinstance(raw, dict) and "confidence" in raw:
        return InquiryResponse(
            confidence=float(raw["confidence"]),
            answer=raw.get("answer"),
        )
    return InquiryResponse(confidence=0.5, answer=None)


class CIRExecutor:
    """Execute a CIRGraph and produce beliefs + trace.

    inquiry_adapter: callable(prompt, agents) -> confidence float [0,1].
    If None, tries the Anthropic SDK; falls back to mock.
    """

    def __init__(
        self,
        inquiry_adapter: InquiryAdapter | None = None,
        strict_guard: bool = False,
        agent_models: dict[str, str] | None = None,
        calibrator: LogisticCalibrator | None = None,
        agreement_comparator: AgreementComparator | None = None,
        answer_normalizer: AnswerNormalizer | None = None,
    ) -> None:
        self._agent_models = dict(DEFAULT_AGENT_MODELS)
        if agent_models:
            self._agent_models.update(agent_models)
        self._adapter = inquiry_adapter or self._resolve_adapter()
        self._strict = strict_guard
        self._calibrator = calibrator
        self._agreement_comparator = agreement_comparator
        self._answer_normalizer = answer_normalizer

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
                if bs.answer is not None:
                    result.answers[bs.name] = bs.answer
                if bs.calibrated_p is not None:
                    result.calibrated[bs.name] = bs.calibrated_p

        for name, bs in graph.belief_store.items():
            result.beliefs[name] = bs.distribution
            if bs.answer is not None and name not in result.answers:
                result.answers[name] = bs.answer

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
            response = _normalize_response(self._adapter(node.prompt, node.agents))
        except Exception as e:
            result.trace.append(f"[inquiry] adapter error: {e} — using 0.5")
            response = InquiryResponse(confidence=0.5, answer=None)

        conf = max(0.0, min(1.0, response.confidence))
        observed = BetaDist.from_confidence(conf)
        result.trace.append(
            f"[inquiry] confidence={conf:.3f} -> Beta({observed.alpha:.1f},{observed.beta:.1f})"
        )

        bs = next(
            (b for b in graph.belief_store.values() if b.node_id == node.id), None
        )
        if bs is not None:
            # Combine a seeded prior (from the SymbolStore via lowering)
            # with the fresh observation by pseudocounts. A Beta(1,1)
            # prior leaves the observation unchanged, so unseeded runs
            # behave exactly as before.
            prior = bs.distribution
            bs.observed = observed
            if abs(prior.alpha - 1.0) > 1e-9 or abs(prior.beta - 1.0) > 1e-9:
                # Seeded prior: combine with the fresh observation using
                # the same pseudocount algebra as consensus (pure
                # addition, K conflict check, no subtract-one, no clamp).
                # A conflicting seeded prior is a guard violation, not a
                # silent merge; the prior is left in place.
                try:
                    posterior = prior.combine_pseudocount(observed)
                except ValueError as e:
                    result.trace.append(f"[inquiry] seeded prior conflict: {e}")
                    result.guard_violations.append(f"inquiry prior conflict: {e}")
                    return
                result.trace.append(
                    f"[inquiry] seeded prior combined -> "
                    f"Beta({posterior.alpha:.1f},{posterior.beta:.1f})"
                )
            else:
                posterior = observed
            bs.distribution = posterior
            bs.provenance.append(f"inquired(conf={conf:.3f})")
            if response.answer is not None:
                bs.answer = response.answer
                result.answers[bs.name] = response.answer
                result.trace.append(
                    f"[inquiry] answer recorded ({len(response.answer)} chars)"
                )

    def _exec_consensus(self, node: ConsensusNode, graph: CIRGraph, result: CIRResult) -> None:
        strategy = node.strategy
        if strategy == "dempster_shafer":
            # Directly-constructed nodes bypass lowering; keep the alias
            # working here too (lowering already warned and normalized).
            strategy = "pooled"
        result.trace.append(f"[consensus] strategy={strategy} threshold={node.threshold}")
        preds = graph.predecessors(node.id)
        input_beliefs: list[BeliefState] = []
        for pred in preds:
            bs = next(
                (b for b in graph.belief_store.values() if b.node_id == pred.id), None
            )
            if bs is not None:
                input_beliefs.append(bs)

        if not input_beliefs:
            result.trace.append("[consensus] no input beliefs — skipping")
            return

        winner_answer: str | None = None
        raw_agreement: float | None = None
        if len(input_beliefs) == 1:
            # A single input is not a combination; say so in the trace.
            result.trace.append("[consensus] single source, no combination performed")
            combined = input_beliefs[0].distribution
        else:
            answered = [bs for bs in input_beliefs if bs.answer is not None]
            if strategy == "agreement" and len(answered) >= 2:
                ar = resolve_agreement(
                    [bs.answer for bs in answered],  # type: ignore[misc]
                    comparator=self._agreement_comparator,
                    normalizer=self._answer_normalizer,
                )
                combined = ar.posterior
                winner_answer = ar.winner
                raw_agreement = ar.agreement
                result.trace.append(
                    f"[consensus] agreement={ar.agreement:.3f} "
                    f"({ar.votes}/{ar.n} votes) "
                    f"posterior_mean={combined.mean:.3f} "
                    f"winner={ar.winner!r} uncalibrated"
                )
            else:
                if strategy == "agreement":
                    result.trace.append(
                        "[consensus] agreement strategy needs 2+ answered "
                        "sources; falling back to pooled"
                    )
                combined = input_beliefs[0].distribution
                for other in input_beliefs[1:]:
                    try:
                        combined = combined.combine_pseudocount(other.distribution)
                    except ValueError as e:
                        result.trace.append(f"[consensus] combination conflict: {e}")
                        result.guard_violations.append(f"consensus conflict: {e}")
                        return
                result.trace.append(
                    f"[consensus] pooled mean={combined.mean:.3f} "
                    f"variance={combined.variance:.4f} uncalibrated"
                )

        # Opt-in calibration: map the uncalibrated posterior mean to a
        # calibrated probability. calibrated_p is set only here, so it
        # appears only when a calibrator was supplied.
        calibrated_p: float | None = None
        if self._calibrator is not None:
            calibrated_p = self._calibrator.predict(combined.mean)
            result.trace.append(
                f"[consensus] calibrated_p={calibrated_p:.3f} "
                f"(from uncalibrated {combined.mean:.3f}, "
                f"calibrator n={self._calibrator.n})"
            )

        if combined.mean < node.threshold:
            msg = f"consensus mean {combined.mean:.3f} below threshold {node.threshold}"
            result.trace.append(f"[consensus] BELOW THRESHOLD — {msg}")
            result.guard_violations.append(msg)

        for bs in graph.belief_store.values():
            if bs.node_id in node.input_ids:
                bs.distribution = combined
                bs.node_id = node.id
                bs.provenance.append(f"consensus({strategy},mean={combined.mean:.3f})")
                if winner_answer is not None:
                    bs.answer = winner_answer
                if raw_agreement is not None:
                    bs.agreement = raw_agreement
                if calibrated_p is not None:
                    bs.calibrated_p = calibrated_p

    def _exec_validation(self, node: ValidationNode, graph: CIRGraph, result: CIRResult) -> None:
        limit = (f"max_variance={node.max_variance}"
                 if node.max_variance is not None else "")
        result.trace.append(
            f"[guard] max_risk={node.max_risk} strategy={node.strategy}"
            + (f" {limit}" if limit else "")
        )
        bs = next(
            (b for b in graph.belief_store.values() if b.node_id == node.target_id),
            None,
        )
        if bs is None:
            result.trace.append("[guard] no belief to validate — skipping")
            return

        dist = bs.distribution
        violations: list[str] = []

        # The guard judges P(correct). When a calibrator was supplied the
        # resolved belief carries calibrated_p and it is used; otherwise
        # the uncalibrated posterior mean is used (lowering warns loudly
        # about the missing calibrator).
        score = bs.calibrated_p if bs.calibrated_p is not None else dist.mean
        score_label = "calibrated_p" if bs.calibrated_p is not None else "mean"
        score_source = ("calibrated" if bs.calibrated_p is not None
                        else "uncalibrated")

        if node.strategy in ("mean", "both"):
            required_mean = 1.0 - node.max_risk
            if score < required_mean:
                violations.append(
                    f"{score_label} {score:.3f} < required {required_mean:.3f}"
                )

        if node.strategy in ("variance", "both"):
            # Explicit per-guard limit when set; otherwise the legacy 0.05
            # default. Note the default can never fire on beliefs produced
            # by BetaDist.from_confidence (strength 10 caps variance at
            # about 0.0227); the lowering pass warns about this.
            max_variance = (node.max_variance if node.max_variance is not None
                            else 0.05)
            if dist.variance > max_variance:
                violations.append(f"variance {dist.variance:.4f} > allowed {max_variance}")

        if violations:
            msg = f"guard '{bs.name}' FAILED: {'; '.join(violations)}"
            result.trace.append(f"[guard] VIOLATION — {msg}")
            result.guard_violations.append(msg)
            result.validations.append({
                "belief": bs.name,
                "strategy": node.strategy,
                "max_risk": node.max_risk,
                "max_variance": node.max_variance,
                "score_source": score_source,
                "score": score,
                "passed": False,
                "violations": list(violations),
            })
            if self._strict:
                raise GuardViolation(msg)
        else:
            result.trace.append(
                f"[guard] PASSED — {score_label}={score:.3f} "
                f"variance={dist.variance:.4f} score_source={score_source}"
            )
            result.validations.append({
                "belief": bs.name,
                "strategy": node.strategy,
                "max_risk": node.max_risk,
                "max_variance": node.max_variance,
                "score_source": score_source,
                "score": score,
                "passed": True,
                "violations": [],
            })
            bs.node_id = node.id

    def _origin_inquiry(
        self, graph: CIRGraph, start_id: str
    ) -> tuple[InquiryNode | None, list[str]]:
        """Walk predecessors back to the originating InquiryNode.

        Returns the inquiry node (or None) plus the chain of node ids
        walked from ``start_id``. Consensus and validation nodes rewrite
        ``BeliefState.node_id`` to their own id, so an evolve entry may
        point at a non-inquiry node; the prompt and agents needed for
        re-inquiry live on the originating InquiryNode.
        """
        chain: list[str] = []
        seen: set[str] = set()
        nid: str | None = start_id
        while nid and nid not in seen:
            seen.add(nid)
            chain.append(nid)
            node = graph.nodes.get(nid)
            if isinstance(node, InquiryNode):
                return node, chain
            preds = graph.predecessors(nid)
            nid = preds[0].id if preds else None
        return None, chain

    def _exec_evolution(self, node: EvolutionNode, graph: CIRGraph, result: CIRResult) -> None:
        result.trace.append(f"[evolve] condition={node.condition} max_iter={node.max_iter}")
        inq_node, chain = self._origin_inquiry(graph, node.subgraph_entry)
        if inq_node is None:
            result.trace.append("[evolve] no originating inquiry found — skipping")
            return
        # Consensus and validation rewrite BeliefState.node_id to their own
        # node id; on guard failure the belief still points at the inquiry.
        # Accept any belief along the chain back to the inquiry.
        bs = next(
            (b for b in graph.belief_store.values() if b.node_id in chain),
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
            try:
                response = _normalize_response(
                    self._adapter(inq_node.prompt, inq_node.agents)
                )
                conf = max(0.0, min(1.0, response.confidence))
                if response.answer is not None:
                    bs.answer = response.answer
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

    def _resolve_adapter(self) -> InquiryAdapter:
        try:
            import anthropic  # type: ignore[import]
            client = anthropic.Anthropic()
            return _make_anthropic_adapter(client, self._agent_models)
        except (ImportError, Exception):
            return _default_mock_adapter
