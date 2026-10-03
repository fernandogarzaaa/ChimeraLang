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
    def __init__(
        self,
        symbol_store: "SymbolStore | None" = None,
        require_dominance: bool = False,
        calibrator: object = None,
    ) -> None:
        self.warnings: list[str] = []
        self._symbol_store = symbol_store
        self._require_dominance = require_dominance
        self._calibrator = calibrator
        self.priors_seeded: list[str] = []

    def lower(self, program: object) -> CIRGraph:
        graph = CIRGraph()
        self.warnings = []
        self.priors_seeded = []

        self._pass_structural(program, graph)
        self._pass_guard_dominance(graph)
        self._pass_dead_belief_elimination(program, graph)
        self._pass_belief_flow_analysis(graph)

        return graph

    # ------------------------------------------------------------------
    # Pass 1: Structural
    # ------------------------------------------------------------------

    def _first_source_or_warn(
        self, belief_name: str, source_ids: list[str], op: str
    ) -> str:
        """Pick the node id a guard/evolve/emit applies to.

        With several source ids and no intervening resolve there is no
        pooled belief; the op applies to the FIRST agent's belief only,
        loudly. resolve is the blessed path to a single pooled belief.
        """
        if len(source_ids) > 1:
            self.warnings.append(
                f"belief '{belief_name}' has {len(source_ids)} agents but "
                f"no resolve — '{op}' applies to the first agent's belief "
                f"only; add 'resolve {belief_name} ...' to pool all agents"
            )
        return source_ids[0] if source_ids else ""

    def _pass_structural(self, program: object, graph: CIRGraph) -> None:
        from chimera.ast_nodes import (
            BeliefDecl, EmitStmt, EvolveStmt, GuardStmt, ResolveStmt,
        )

        # Belief name -> ids of the nodes currently standing for it.
        # A multi-agent belief fans out to one id per agent; resolve
        # collapses the list back to a single consensus id.
        belief_node_map: dict[str, list[str]] = {}

        for decl in program.declarations:  # type: ignore[attr-defined]
            if isinstance(decl, BeliefDecl):
                agents = (list(decl.inquire_expr.agents)
                          if decl.inquire_expr and decl.inquire_expr.agents else [])
                if len(agents) != len(set(agents)):
                    dupes = sorted({a for a in agents if agents.count(a) > 1})
                    self.warnings.append(
                        f"belief '{decl.name}' lists agent(s) {dupes} more than "
                        f"once — sources are the same model, evidence is correlated"
                    )
                prompt = decl.inquire_expr.prompt if decl.inquire_expr else ""
                ttl = decl.inquire_expr.ttl if decl.inquire_expr else None
                # One InquiryNode per agent (fan-out A). Zero or one agent
                # keeps the historical single-node shape and belief name.
                per_agent = agents if len(agents) > 1 else [agents[0] if agents else ""]
                node_ids: list[str] = []
                seen_names: set[str] = set()
                for i, agent in enumerate(per_agent):
                    inq = InquiryNode(
                        prompt=prompt,
                        agents=[agent] if agent else [],
                        ttl=ttl,
                    )
                    graph.add_node(inq)
                    node_ids.append(inq.id)

                    if len(per_agent) == 1:
                        belief_name = decl.name
                    else:
                        belief_name = f"{decl.name}@{agent}"
                        n = 2
                        while belief_name in seen_names:
                            belief_name = f"{decl.name}@{agent}#{n}"
                            n += 1
                    seen_names.add(belief_name)

                    prior = BetaDist.uniform()
                    if self._symbol_store is not None and inq.prompt:
                        seeded = self._symbol_store.find_prior_for(inq.prompt)
                        if seeded is not None:
                            prior = seeded
                            self.priors_seeded.append(belief_name)

                    graph.belief_store[belief_name] = BeliefState(
                        name=belief_name,
                        distribution=prior,
                        ttl=inq.ttl,
                        node_id=inq.id,
                    )
                    if not graph.entry_id:
                        graph.entry_id = inq.id
                belief_node_map[decl.name] = node_ids

            elif isinstance(decl, ResolveStmt):
                source_ids = belief_node_map.get(decl.target, [])
                strategy = decl.strategy
                if strategy == "dempster_shafer":
                    # Accepted as an alias of "pooled" for compatibility;
                    # it was never formal Dempster-Shafer combination.
                    self.warnings.append(
                        f"resolve strategy 'dempster_shafer' on '{decl.target}' "
                        "is accepted as an alias of 'pooled' (pseudocount "
                        "addition with a K conflict check), not formal "
                        "Dempster-Shafer combination"
                    )
                    strategy = "pooled"
                cons = ConsensusNode(
                    threshold=decl.threshold,
                    strategy=strategy,
                    input_ids=list(source_ids),
                )
                graph.add_node(cons)
                for source_id in source_ids:
                    graph.add_edge(CIREdge(
                        source_id=source_id, target_id=cons.id,
                        kind=EdgeKind.CONSENSUS,
                    ))
                belief_node_map[decl.target] = [cons.id]

            elif isinstance(decl, GuardStmt):
                source_ids = belief_node_map.get(decl.target, [])
                source_id = self._first_source_or_warn(decl.target, source_ids, "guard")
                val = ValidationNode(
                    max_risk=decl.max_risk,
                    strategy=decl.strategy,
                    target_id=source_id,
                    max_variance=decl.max_variance,
                )
                graph.add_node(val)
                if source_id:
                    graph.add_edge(CIREdge(
                        source_id=source_id, target_id=val.id,
                        kind=EdgeKind.VALIDATION,
                    ))
                belief_node_map[decl.target] = [val.id]

            elif isinstance(decl, EvolveStmt):
                source_ids = belief_node_map.get(decl.target, [])
                source_id = self._first_source_or_warn(decl.target, source_ids, "evolve")
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
                belief_node_map[decl.target] = [evo.id]

            elif isinstance(decl, EmitStmt):
                from chimera.ast_nodes import Identifier
                if isinstance(decl.value, Identifier):
                    node_ids = belief_node_map.get(decl.value.name, [])
                    node_id = self._first_source_or_warn(
                        decl.value.name, node_ids, "emit")
                    if node_id:
                        graph.emit_ids.append(node_id)

    # ------------------------------------------------------------------
    # Pass 1b: Guard dominance (static check)
    # ------------------------------------------------------------------

    def _dominates(self, graph: CIRGraph, consumer_id: str) -> bool:
        """True iff every belief-flow path from a source to the consumer
        passes through a ValidationNode positioned after the last
        ConsensusNode on that path.

        A guard before a resolve does not dominate the consensus belief
        (intentional conservatism): the validation must validate the
        belief version the consumer actually consumes.
        """
        nodes = graph.nodes
        paths: list[list[str]] = []
        stack: list[tuple[str, list[str], frozenset]] = [
            (consumer_id, [consumer_id], frozenset({consumer_id}))]
        while stack:
            nid, path, seen = stack.pop()
            preds = [p for p in graph.predecessors(nid) if p.id not in seen]
            if not preds:
                paths.append(list(reversed(path)))
                continue
            for p in preds:
                stack.append((p.id, path + [p.id], seen | {p.id}))
        if not paths:
            return False
        for path in paths:
            last_cons = -1
            for i, nid in enumerate(path):
                if isinstance(nodes[nid], ConsensusNode):
                    last_cons = i
            if not any(
                isinstance(nodes[nid], ValidationNode) and i > last_cons
                for i, nid in enumerate(path)
            ):
                return False
        return True

    def _pass_guard_dominance(self, graph: CIRGraph) -> None:
        """Static guard-dominance check.

        Every effectful consumer (EvolutionNode, emit target) must be
        dominated along belief-flow edges by a ValidationNode on the
        same belief lineage, positioned after any resolve. Under
        require_dominance a violation is a LoweringError; otherwise it
        is a lowering warning. Under require_dominance a guard with
        strategy 'mean' or 'both' and no calibrator is also a
        LoweringError, because numeric thresholds on uncalibrated
        posterior means are unsound.
        """
        if self._require_dominance and self._calibrator is None:
            for nid, node in graph.nodes.items():
                if (isinstance(node, ValidationNode)
                        and node.strategy in ("mean", "both")):
                    name = self._belief_name_for(graph, nid) or nid
                    raise LoweringError(
                        f"guard on '{name}' uses strategy '{node.strategy}' "
                        f"with no calibrator: numeric thresholds on "
                        f"uncalibrated posterior means are unsound. Supply a "
                        f"calibrator or use strategy 'variance'."
                    )
        consumers: list[tuple[str, str]] = [
            (nid, "evolve")
            for nid, node in graph.nodes.items()
            if isinstance(node, EvolutionNode)
        ]
        consumers.extend(
            (eid, "emit") for eid in graph.emit_ids if eid in graph.nodes
        )
        for cid, kind in consumers:
            if not self._dominates(graph, cid):
                name = self._belief_name_for(graph, cid) or cid
                msg = (
                    f"guard dominance: effectful {kind} of belief '{name}' "
                    f"is not guard-dominated (no ValidationNode on every "
                    f"belief-flow path to node {cid}, after any resolve)"
                )
                if self._require_dominance:
                    raise LoweringError(msg)
                self.warnings.append(msg)

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

    def _belief_name_for(self, graph: CIRGraph, nid: str) -> str:
        """Find the belief name behind a node by walking predecessors back.

        At lowering time a belief's node_id still points at its InquiryNode
        (consensus/validation rewrite it only at execution), so a guard's
        direct predecessor may be a consensus node with no belief attached.
        """
        seen: set[str] = set()
        stack = [nid]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            bs = next(
                (b for b in graph.belief_store.values() if b.node_id == cur),
                None,
            )
            if bs is not None:
                return bs.name
            stack.extend(p.id for p in graph.predecessors(cur))
        return ""

    def _pass_belief_flow_analysis(self, graph: CIRGraph) -> None:
        VARIANCE_WARN_THRESHOLD = 0.08
        # Legacy default variance limit used by the executor when a guard
        # does not set max_variance explicitly.
        DEFAULT_VARIANCE_LIMIT = 0.05

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
                if node.strategy in ("variance", "both"):
                    limit = (node.max_variance if node.max_variance is not None
                             else DEFAULT_VARIANCE_LIMIT)
                    # Inquiry-produced beliefs use from_confidence with the
                    # default strength of 10, which caps variance at about
                    # 0.0227 (more evidence only lowers it). A limit above
                    # that cap can never fire, so the variance check is dead.
                    # Re-checked 2026-10-03 against the new pooling algebra
                    # (pure pseudocount addition): combined beliefs now
                    # carry the full sum of the input strengths, where the
                    # old rule subtracted 2 per combination and could clamp
                    # a parameter to ~1e-6, producing spuriously large
                    # variance. Combined beliefs therefore sit even further
                    # below this cap than before, and the cap itself is
                    # unchanged because from_confidence is unchanged.
                    cap = BetaDist.max_variance_for_strength(10.0)
                    if limit > cap:
                        target_name = self._belief_name_for(graph, nid)
                        self.warnings.append(
                            f"guard on '{target_name or node.target_id}' has an "
                            f"unreachable variance limit ({limit}); "
                            f"inquiry-produced beliefs cap variance at "
                            f"{cap:.4f} (strength 10), so the variance check "
                            f"can never fire"
                        )
