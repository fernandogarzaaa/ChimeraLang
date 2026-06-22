"""Capability registry for ChimeraLang — the single source of truth for which
statically-identifiable operations require which capabilities.

Grounded in the constructs that actually exist in the language today:

  * An agent **inquiry** (`InquireExpr`, e.g. ``belief x := inquire { agents: [...] }``)
    contacts an external model over the network -> {MODEL, NETWORK}.
  * A host-side **tool call** (`chimera.claude_adapter.ToolCallSpec`) invokes an
    external tool over the network -> {TOOL, NETWORK}.
  * The ``print`` builtin writes to the console -> {IO}. It is the only
    side-effecting builtin; every other builtin (``confident``, ``explore``,
    ``consensus``, ``len``, ``sum``, ...) is pure and maps to no capability.

This is *not* a sandbox. It only describes the capabilities of operations the
type checker can see in the source AST (and the host adapter), so that declared
``allow``/``forbidden`` constraints can be enforced against them statically.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Iterator

from chimera.ast_nodes import ASTNode, CallExpr, Identifier, InquireExpr

# ---------------------------------------------------------------------------
# Canonical capability names. Keep this small and grounded in real operations.
# ---------------------------------------------------------------------------
NETWORK = "network"
MODEL = "model"        # LLM / agent inference
FILESYSTEM = "filesystem"
TOOL = "tool"          # external tool invocation
IO = "io"              # console / standard streams

#: The full set of names this checker understands. Only these names, when they
#: appear in an ``allow``/``forbidden`` list, are treated as capability
#: constraints; any other (free-form) string is left as a semantic annotation.
CANONICAL: frozenset[str] = frozenset({NETWORK, MODEL, FILESYSTEM, TOOL, IO})

#: Builtin / known call name -> capabilities it requires. Pure builtins are
#: simply absent from this map. ``filesystem`` is reserved: no builtin performs
#: filesystem I/O today, so nothing currently maps to it.
_CALL_CAPABILITIES: dict[str, frozenset[str]] = {
    "print": frozenset({IO}),
}


def capability_of_call(callee_name: str) -> frozenset[str]:
    """Capabilities required by calling the named builtin (empty if pure/unknown)."""
    return _CALL_CAPABILITIES.get(callee_name, frozenset())


def capability_of_inquiry() -> frozenset[str]:
    """An agent inquiry reaches an external model over the network."""
    return frozenset({MODEL, NETWORK})


def capability_of_tool_call(spec: object) -> frozenset[str]:  # noqa: ARG001
    """A host-side tool call (ToolCallSpec) invokes a tool over the network."""
    return frozenset({TOOL, NETWORK})


# ---------------------------------------------------------------------------
# AST walking
# ---------------------------------------------------------------------------

def _walk(node: object) -> Iterator[ASTNode]:
    """Yield ``node`` and every descendant ``ASTNode`` (skipping the ``span``)."""
    if isinstance(node, ASTNode):
        yield node
        if not is_dataclass(node):
            return
        for f in fields(node):
            if f.name == "span":
                continue
            yield from _walk(getattr(node, f.name, None))
    elif isinstance(node, (list, tuple)):
        for item in node:
            yield from _walk(item)


def _capability_sites(nodes: list[ASTNode]) -> dict[str, str]:
    """Map each directly-used capability to a human-readable source site.

    "Directly" means used by a construct physically present in ``nodes`` — this
    does not follow calls into other declarations (the checker does that, via
    the call graph). The first site found for a capability wins.
    """
    sites: dict[str, str] = {}
    for root in nodes:
        for sub in _walk(root):
            if isinstance(sub, InquireExpr):
                for cap in capability_of_inquiry():
                    sites.setdefault(cap, "agent inquiry")
            elif isinstance(sub, CallExpr) and isinstance(sub.callee, Identifier):
                for cap in capability_of_call(sub.callee.name):
                    sites.setdefault(cap, f"call to '{sub.callee.name}'")
    return sites


def capabilities_used_by_expr(expr: ASTNode) -> set[str]:
    """Directly-used capabilities within a single expression (non-transitive)."""
    return set(_capability_sites([expr]).keys())


def capabilities_used_by_stmt(stmt: ASTNode) -> set[str]:
    """Directly-used capabilities within a single statement (non-transitive)."""
    return set(_capability_sites([stmt]).keys())


def direct_capability_sites(nodes: list[ASTNode]) -> dict[str, str]:
    """Public accessor: capability -> source-site label for a body's nodes."""
    return _capability_sites(nodes)


def called_names(nodes: list[ASTNode]) -> set[str]:
    """Names of identifier-callees invoked within ``nodes`` (for the call graph)."""
    names: set[str] = set()
    for root in nodes:
        for sub in _walk(root):
            if isinstance(sub, CallExpr) and isinstance(sub.callee, Identifier):
                names.add(sub.callee.name)
    return names
