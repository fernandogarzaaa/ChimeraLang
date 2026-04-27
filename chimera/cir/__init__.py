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
    """Full CIR pipeline: lower → optionally load symbols → execute → optionally save symbols."""
    store = SymbolStore()
    if load_symbols:
        store.load_symbols(load_symbols)

    lowering = CIRLowering()
    graph = lowering.lower(program)

    executor = CIRExecutor(inquiry_adapter=inquiry_adapter, strict_guard=strict_guard)
    result = executor.run(graph)

    prompts = [
        n.prompt for n in graph.nodes.values() if isinstance(n, InquiryNode)
    ]
    if prompts:
        store.register(graph, prompts=prompts)

    if save_symbols:
        store.save_symbols(save_symbols)

    result.meta["lowering_warnings"] = lowering.warnings
    return result
