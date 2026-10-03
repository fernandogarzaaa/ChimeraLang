"""CIR certificate (v2) production.

Builds chimeralang-cert/v2 certificates with a cir section for CIR
program runs. The verifier (chimera/verify.py) re-derives the graph
from cir.program_source with the real parser and lowering, compares
the ID-insensitive canonical shape to the embedded graph, recomputes
the dominance predicate, and requires the stored claim to match.

Trust boundary, stated exactly:
  - When the verifier can import the chimera package, it proves the
    embedded graph is the canonical lowering of the embedded program
    source. A producer cannot substitute a graph from a different
    program.
  - When the verifier cannot import chimera, the graph-source link is
    NOT RE-DERIVED: only internal consistency is checked (hashes,
    dominance recomputed from the embedded graph), and a certificate
    claiming "enforced" is never reported valid.
"""
from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any


def _canonical_bytes(obj: Any) -> bytes:
    """Deterministic JSON encoding shared by producer and verifier."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def serialize_graph(graph: Any) -> dict[str, Any]:
    """Deterministic serialization of a CIRGraph for certificate embedding."""
    from chimera.cir.nodes import (
        ConsensusNode,
        EvolutionNode,
        InquiryNode,
        ValidationNode,
    )

    nodes: list[dict[str, Any]] = []
    for nid in sorted(graph.nodes):
        n = graph.nodes[nid]
        d: dict[str, Any] = {"id": nid, "kind": type(n).__name__}
        if isinstance(n, ValidationNode):
            d.update({
                "max_risk": n.max_risk,
                "strategy": n.strategy,
                "max_variance": n.max_variance,
                "target_id": n.target_id,
                "strict": n.strict,
            })
        elif isinstance(n, ConsensusNode):
            d.update({
                "threshold": n.threshold,
                "strategy": n.strategy,
                "input_ids": sorted(n.input_ids),
            })
        elif isinstance(n, EvolutionNode):
            d.update({
                "condition": n.condition,
                "max_iter": n.max_iter,
                "subgraph_entry": n.subgraph_entry,
            })
        elif isinstance(n, InquiryNode):
            d.update({"agents": list(n.agents), "ttl": n.ttl})
        nodes.append(d)
    edges = sorted(
        (
            {"source_id": e.source_id, "target_id": e.target_id,
             "kind": e.kind.name}
            for e in graph.edges
        ),
        key=lambda e: (e["source_id"], e["target_id"], e["kind"]),
    )
    return {
        "nodes": nodes,
        "edges": edges,
        "emit_ids": sorted(graph.emit_ids),
    }


def check_dominance(graph_dict: dict[str, Any]) -> dict[str, Any]:
    """Recompute dominance over a serialized graph dict.

    Standalone reimplementation of the lowering check, operating only
    on the serialized structure: every effectful consumer
    (EvolutionNode, emit target) is dominated iff every belief-flow
    path from a source to it passes through a ValidationNode
    positioned after the last ConsensusNode on that path.

    Also reports all_strict: True iff every ValidationNode that
    dominates a consumer on any path carries strict=True. Only
    source-level strict guards count; the global strict_guard run
    flag is not part of the graph and does not affect the claim.
    """
    kinds = {n["id"]: n["kind"] for n in graph_dict["nodes"]}
    strict_of = {n["id"]: bool(n.get("strict", False))
                 for n in graph_dict["nodes"]}
    preds: dict[str, list[str]] = {}
    for e in graph_dict["edges"]:
        preds.setdefault(e["target_id"], []).append(e["source_id"])
    consensus = {nid for nid, k in kinds.items() if k == "ConsensusNode"}

    def dominated(consumer_id: str) -> tuple[bool, list[list[str]]]:
        paths: list[list[str]] = []
        stack = [(consumer_id, [consumer_id], {consumer_id})]
        while stack:
            nid, path, seen = stack.pop()
            ps = [p for p in preds.get(nid, []) if p not in seen]
            if not ps:
                paths.append(list(reversed(path)))
                continue
            for p in ps:
                stack.append((p, path + [p], seen | {p}))
        if not paths:
            return False, []
        for path in paths:
            last_cons = -1
            for i, nid in enumerate(path):
                if nid in consensus:
                    last_cons = i
            if not any(kinds.get(nid) == "ValidationNode" and i > last_cons
                       for i, nid in enumerate(path)):
                return False, paths
        return True, paths

    def dominating_guards(consumer_id: str) -> set[str]:
        """ValidationNode ids positioned after the last ConsensusNode on
        any source-to-consumer path: the guards that dominate it."""
        _, paths = dominated(consumer_id)
        guards: set[str] = set()
        for path in paths:
            last_cons = -1
            for i, nid in enumerate(path):
                if nid in consensus:
                    last_cons = i
            for i, nid in enumerate(path):
                if kinds.get(nid) == "ValidationNode" and i > last_cons:
                    guards.add(nid)
        return guards

    evidence: list[dict[str, Any]] = []
    all_ok = True
    all_guards: set[str] = set()
    consumers = ([n["id"] for n in graph_dict["nodes"]
                  if n["kind"] == "EvolutionNode"]
                 + [eid for eid in graph_dict["emit_ids"] if eid in kinds])
    for cid in consumers:
        ok, paths = dominated(cid)
        guards = dominating_guards(cid) if ok else set()
        all_guards |= guards
        evidence.append({
            "consumer": cid,
            "consumer_kind": kinds[cid],
            "dominated": ok,
            "paths": paths,
        })
        all_ok = all_ok and ok
    all_strict = all(strict_of[gid] for gid in all_guards)
    return {"dominated": all_ok, "all_strict": all_strict,
            "dominating_guards": sorted(all_guards),
            "evidence": evidence}


def dominance_claim(dominated: bool, all_strict: bool) -> str:
    """Decision 4 (Option 2): 'enforced' only when dominance holds AND
    every guard on every dominating path to each effectful node is
    source-level strict. The global strict_guard run flag does not
    affect the claim."""
    if dominated and all_strict:
        return "enforced"
    if dominated:
        return "non-blocking"
    return "absent"


def certify_cir(
    source: str,
    graph: Any,
    result: Any,
    *,
    strict_guard: bool,
    calibrator: Any = None,
    hmac_key: bytes | None = None,
    sign_key: Any = None,
) -> dict[str, Any]:
    """Build a chimeralang-cert/v2 certificate for a CIR run.

    source: the program source text. graph: the lowered CIRGraph.
    result: the CIRResult (for the structured validations list).
    strict_guard: whether the run used strict guard mode. This is
    recorded for information only; it does not affect the dominance
    claim. The claim is derived solely from the source-level strict
    flags on the graph's ValidationNodes (Option 2).
    calibrator: the calibrator supplied to the run, if any.
    """
    graph_dict = serialize_graph(graph)
    graph_hash = hashlib.sha256(
        _canonical_bytes(graph_dict)).hexdigest()[:32]
    program_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()[:32]
    dom = check_dominance(graph_dict)

    # Build the dominating guard list for the certificate. Each entry
    # carries the guard's strategy, thresholds, strict flag, the
    # score_source from its validation record, and whether it is vacuous.
    from chimera.cir.nodes import is_vacuous_guard
    nodes_by_id = {n["id"]: n for n in graph_dict["nodes"]}
    # Map (strategy, max_risk, max_variance) -> score_source from validations.
    score_lookup: dict[tuple, str] = {}
    for v in result.validations:
        key = (v.get("strategy"), v.get("max_risk"), v.get("max_variance"))
        # Prefer the first occurrence; duplicates are rare.
        if key not in score_lookup:
            score_lookup[key] = v.get("score_source", "uncalibrated")
    guards_list: list[dict[str, Any]] = []
    for gid in dom["dominating_guards"]:
        gnode = nodes_by_id.get(gid, {})
        strategy = gnode.get("strategy", "both")
        max_risk = gnode.get("max_risk", 0.2)
        max_variance = gnode.get("max_variance")
        vacuous = is_vacuous_guard(strategy, max_risk, max_variance)
        key = (strategy, max_risk, max_variance)
        score_source = score_lookup.get(key, "uncalibrated")
        guards_list.append({
            "id": gid,
            "strategy": strategy,
            "max_risk": max_risk,
            "max_variance": max_variance,
            "strict": bool(gnode.get("strict", False)),
            "score_source": score_source,
            "vacuous": vacuous,
        })

    # guard_strength: worst case over dominating guards.
    # vacuous > uncalibrated > nonvacuous in severity.
    if any(g["vacuous"] for g in guards_list):
        guard_strength = "vacuous"
    elif any(g["score_source"] == "uncalibrated" for g in guards_list):
        guard_strength = "uncalibrated"
    else:
        guard_strength = "nonvacuous"

    cir: dict[str, Any] = {
        "program_source": source,
        "program_hash": program_hash,
        "graph": graph_dict,
        "graph_hash": graph_hash,
        "dominance": {
            "claim": dominance_claim(dom["dominated"], dom["all_strict"]),
            "evidence": dom["evidence"],
            "guards": guards_list,
            "guard_strength": guard_strength,
        },
        "validations": list(result.validations),
        "strict_guard": strict_guard,
        "calibrator_present": calibrator is not None,
    }
    cir_bytes = _canonical_bytes(cir)
    binding: dict[str, Any] = {
        "algo": "sha256",
        "certificate_hash": hashlib.sha256(cir_bytes).hexdigest(),
        "hmac": None,
        "signature": None,
    }
    if hmac_key is not None:
        binding["hmac"] = hmac.new(
            hmac_key, cir_bytes, hashlib.sha256).hexdigest()
    if sign_key is not None:
        from chimera.integrity import _sign_ed25519
        binding["signature"] = _sign_ed25519(sign_key, cir_bytes)
    return {"format": "chimeralang-cert/v2", "cir": cir, "binding": binding}
