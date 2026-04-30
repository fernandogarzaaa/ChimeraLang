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

__all__ = [
    "CIRExecutor", "CIRResult", "GuardViolation", "InquiryResponse",
    "CIRLowering", "LoweringError",
    "BetaDist", "BeliefState", "CIRGraph", "CIRNode",
    "ConsensusNode", "EdgeKind", "EvolutionNode",
    "InquiryNode", "MetaNode", "ValidationNode",
    "SymbolStore", "Symbol", "extract_subgraphs",
    "run_cir",
]


def run_cir(
    program: object,
    *,
    save_symbols: str | None = None,
    load_symbols: str | None = None,
    inquiry_adapter=None,
    strict_guard: bool = False,
) -> CIRResult:
    """Full CIR pipeline.

    1. Load symbols (if requested) so semantically-similar past prompts
       can seed Beta priors instead of starting from Beta(1,1).
    2. Lower the program with the loaded store wired in.
    3. Execute the graph.
    4. Feed each emitted posterior back to its matching symbol so the
       store accumulates calibrated evidence across runs.
    5. Register the graph as a (new or recurring) symbol and persist.
    """
    store = SymbolStore()
    if load_symbols:
        store.load_symbols(load_symbols)

    lowering = CIRLowering(symbol_store=store)
    graph = lowering.lower(program)

    executor = CIRExecutor(inquiry_adapter=inquiry_adapter, strict_guard=strict_guard)
    result = executor.run(graph)

    prompts = [
        n.prompt for n in graph.nodes.values() if isinstance(n, InquiryNode)
    ]
    if prompts:
        store.register(graph, prompts=prompts)

    # Feed posteriors back AFTER registration so a fresh symbol can
    # receive its first observation. Only beliefs that originated from
    # an inquiry have an associated prompt to match against.
    for bs in graph.belief_store.values():
        inq = graph.nodes.get(bs.node_id)
        prompt = getattr(inq, "prompt", "") if inq is not None else ""
        if prompt:
            store.record_observation(prompt, bs.distribution)

    if save_symbols:
        store.save_symbols(save_symbols)

    result.meta["lowering_warnings"] = lowering.warnings
    result.meta["priors_seeded"] = list(lowering.priors_seeded)
    return result
