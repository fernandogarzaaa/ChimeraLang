"""ChimeraLang CIR — Cognitive Intermediate Representation.

Public API:
    from chimera.cir import run_cir, CIRResult, SymbolStore
"""
from chimera.cir.executor import (
    CIRExecutor, CIRResult, GuardViolation, InquiryResponse,
)
from chimera.cir.lower import CIRLowering, LoweringError
from chimera.cir.nodes import (
    BetaDist, BeliefState, CIRGraph, CIRNode,
    ConsensusNode, EdgeKind, EvolutionNode,
    InquiryNode, MetaNode, ValidationNode,
)
from chimera.cir.symbols import SymbolStore, Symbol, extract_subgraphs
from chimera.cir.agreement import (
    AgreementResult, normalize_answer, resolve_agreement,
)
from chimera.cir.calibration import LogisticCalibrator, MIN_FIT_N

__all__ = [
    "CIRExecutor", "CIRResult", "GuardViolation", "InquiryResponse",
    "CIRLowering", "LoweringError",
    "BetaDist", "BeliefState", "CIRGraph", "CIRNode",
    "ConsensusNode", "EdgeKind", "EvolutionNode",
    "InquiryNode", "MetaNode", "ValidationNode",
    "SymbolStore", "Symbol", "extract_subgraphs",
    "AgreementResult", "normalize_answer", "resolve_agreement",
    "LogisticCalibrator", "MIN_FIT_N",
    "run_cir",
]


def run_cir(
    program: object,
    *,
    save_symbols: str | None = None,
    load_symbols: str | None = None,
    inquiry_adapter=None,
    strict_guard: bool = False,
    agent_models: dict[str, str] | None = None,
    calibrator: "LogisticCalibrator | None" = None,
    agreement_comparator=None,
    answer_normalizer=None,
) -> CIRResult:
    """Full CIR pipeline.

    1. Load symbols (if requested) so semantically-similar past prompts
       can seed Beta priors instead of starting from Beta(1,1).
    2. Lower the program with the loaded store wired in.
    3. Execute the graph.
    4. Feed each emitted raw observed likelihood back to its matching
       symbol so the store accumulates calibrated evidence across runs.
       Only the likelihood is fed, never the posterior, so a seeded prior
       is not double-counted.
    5. Register the graph as a (new or recurring) symbol and persist.

    ``calibrator`` is an optional
    :class:`chimera.cir.calibration.LogisticCalibrator`. When supplied,
    resolved beliefs carry ``calibrated_p`` (used by guard and emit);
    when omitted, guard and emit fall back to the uncalibrated
    posterior and a lowering warning says so. ``agreement_comparator``
    and ``answer_normalizer`` plug into the agreement resolve strategy
    (see :mod:`chimera.cir.agreement`).
    """
    store = SymbolStore()
    if load_symbols:
        store.load_symbols(load_symbols)

    lowering = CIRLowering(symbol_store=store)
    graph = lowering.lower(program)

    if calibrator is None and any(
        isinstance(n, ConsensusNode) and len(n.input_ids) > 1
        for n in graph.nodes.values()
    ):
        lowering.warnings.append(
            "consensus resolved without a calibrator: posteriors are "
            "uncalibrated (agreement vote shares and Beta means are not "
            "probabilities). Pass calibrator=... to run_cir (or "
            "--calibrator=PATH on the CLI), fit on a held-out calibration "
            "set, to get calibrated_p on resolved beliefs."
        )

    executor = CIRExecutor(inquiry_adapter=inquiry_adapter,
                           strict_guard=strict_guard,
                           agent_models=agent_models,
                           calibrator=calibrator,
                           agreement_comparator=agreement_comparator,
                           answer_normalizer=answer_normalizer)
    result = executor.run(graph)

    prompts = [
        n.prompt for n in graph.nodes.values() if isinstance(n, InquiryNode)
    ]
    if prompts:
        store.register(graph, prompts=prompts)

    # Feed raw likelihoods back AFTER registration so a fresh symbol can
    # receive its first observation. Only beliefs that originated from
    # an inquiry have an associated prompt to match against, and only the
    # raw observed likelihood is fed (never the posterior) so a seeded
    # prior is not double-counted.
    for bs in graph.belief_store.values():
        inq = graph.nodes.get(bs.node_id)
        prompt = getattr(inq, "prompt", "") if inq is not None else ""
        if prompt and bs.observed is not None:
            store.record_observation(prompt, bs.observed)

    if save_symbols:
        store.save_symbols(save_symbols)

    result.meta["lowering_warnings"] = lowering.warnings
    result.meta["priors_seeded"] = list(lowering.priors_seeded)
    return result
