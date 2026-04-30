"""AST → CIR Lowering for ChimeraLang.

Three sequential passes:
  1. Structural: map AST nodes to CIR nodes with typed edges
  2. Dead belief elimination: remove nodes with no emit/resolve downstream
  3. Belief flow analysis: forward-propagate BetaDist, flag high-variance paths

If a SymbolStore is provided, new BeliefDecls consult it for a prior
distribution derived from past observations of semantically-similar
prompts; otherwise the prior defaults to Beta(1,1).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from chimera.cir.nodes import (
    BeliefState, BetaDist, CIREdge, CIRGraph,
    ConsensusNode, EdgeKind, EvolutionNode, InquiryNode,
    ValidationNode,
)

if TYPE_CHECKING:
    from chimera.cir.symbols import SymbolStore


class LoweringError(Exception):
    pass


class CIRLowering:
    def __init__(self, symbol_store: "SymbolStore | None" = None) -> None:
        self.warnings: list[str] = []
        self._symbol_store = symbol_store
        self.priors_seeded: list[str] = []

    def lower(self, program: object) -> CIRGraph:
        graph = CIRGraph()
        self.warnings = []
        self.priors_seeded = []

        self._pass_structural(program, graph)
        self._pass_dead_belief_elimination(program, graph)
        self._pass_belief_flow_analysis(graph)

        return graph

    # ------------------------------------------------------------------
    # Pass 1: Structural
    # ------------------------------------------------------------------

    def _pass_structural(self, program: object, graph: CIRGraph) -> None:
        from chimera.ast_nodes import (
            BeliefDecl, EmitStmt, EvolveStmt, GuardStmt, ResolveStmt,
        )

        belief_node_map: dict[str, str] = {}

        for decl in program.declarations:  # type: ignore[attr-defined]
            if isinstance(decl, BeliefDecl):
                inq = InquiryNode(
                    prompt=decl.inquire_expr.prompt if decl.inquire_expr else "",
                    agents=list(decl.inquire_expr.agents) if decl.inquire_expr else [],
                    ttl=decl.inquire_expr.ttl if decl.inquire_expr else None,
                )
                graph.add_node(inq)
                belief_node_map[decl.name] = inq.id

                prior = BetaDist.uniform()
                if self._symbol_store is not None and inq.prompt:
                    seeded = self._symbol_store.find_prior_for(inq.prompt)
                    if seeded is not None:
                        prior = seeded
                        self.priors_seeded.append(decl.name)

                graph.belief_store[decl.name] = BeliefState(
                    name=decl.name,
                    distribution=prior,
                    ttl=inq.ttl,
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
        if not graph.emit_ids:
            return

        live: set[str] = set(graph.emit_ids)
        queue = list(graph.emit_ids)
        while queue:
            nid = queue.pop(0)
            for edge in graph.edges:
                if edge.target_id == nid and edge.source_id not in live:
                    live.add(edge.source_id)
                    queue.append(edge.source_id)

        dead = [nid for nid in list(graph.nodes) if nid not in live]
        for nid in dead:
            del graph.nodes[nid]
        graph.edges = [e for e in graph.edges if e.source_id in live and e.target_id in live]

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
        VARIANCE_WARN_THRESHOLD = 0.08

        for nid in graph.nodes:
            node = graph.nodes[nid]
            if isinstance(node, ValidationNode):
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
