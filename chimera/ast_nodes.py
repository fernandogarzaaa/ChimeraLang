"""AST node definitions for ChimeraLang."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from chimera.tokens import SourceSpan


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

@dataclass
class ASTNode:
    span: SourceSpan | None = field(default=None, repr=False, kw_only=True)


# ---------------------------------------------------------------------------
# Type Expressions
# ---------------------------------------------------------------------------

@dataclass
class TypeExpr(ASTNode):
    """Base for all type expressions."""


@dataclass
class PrimitiveType(TypeExpr):
    name: str  # Int, Float, Bool, Text, Void


@dataclass
class ProbabilisticType(TypeExpr):
    """Confident<T>, Explore<T>, Converge<T>, Provisional<T>"""
    wrapper: str  # Confident | Explore | Converge | Provisional
    inner: TypeExpr


@dataclass
class MemoryType(TypeExpr):
    """Ephemeral<T>, Persistent<T>"""
    scope: str  # Ephemeral | Persistent
    inner: TypeExpr


@dataclass
class GenericType(TypeExpr):
    """List<T>, Map<K,V>, Option<T>, Result<T,E>"""
    name: str
    params: list[TypeExpr]


@dataclass
class NamedType(TypeExpr):
    """User-defined type reference."""
    name: str


# ---------------------------------------------------------------------------
# Expressions
# ---------------------------------------------------------------------------

@dataclass
class Expr(ASTNode):
    """Base for all expressions."""


@dataclass
class IntLiteral(Expr):
    value: int


@dataclass
class FloatLiteral(Expr):
    value: float


@dataclass
class StringLiteral(Expr):
    value: str


@dataclass
class BoolLiteral(Expr):
    value: bool


@dataclass
class ListLiteral(Expr):
    elements: list[Expr]


@dataclass
class Identifier(Expr):
    name: str


@dataclass
class BinaryOp(Expr):
    op: str
    left: Expr
    right: Expr


@dataclass
class UnaryOp(Expr):
    op: str
    operand: Expr


@dataclass
class CallExpr(Expr):
    callee: Expr
    args: list[Expr]


@dataclass
class MemberExpr(Expr):
    obj: Expr
    member: str


@dataclass
class IfExpr(Expr):
    condition: Expr
    then_body: list[Statement]
    else_body: list[Statement] | None = None


@dataclass
class CompareChain(Expr):
    """quality: security > performance > readability"""
    items: list[str]


# ---------------------------------------------------------------------------
# Statements
# ---------------------------------------------------------------------------

@dataclass
class Statement(ASTNode):
    """Base for all statements."""


@dataclass
class ValDecl(Statement):
    name: str
    type_ann: TypeExpr | None = None
    value: Expr | None = None


@dataclass
class ReturnStmt(Statement):
    value: Expr | None = None


@dataclass
class ExprStmt(Statement):
    expr: Expr


@dataclass
class AssertStmt(Statement):
    condition: Expr
    message: str | None = None


@dataclass
class EmitStmt(Statement):
    value: Expr


@dataclass
class ForStmt(Statement):
    """for x in collection ... end"""
    target: str
    iterable: Expr
    body: list[Statement]


@dataclass
class MatchArm(ASTNode):
    """One arm of a match expression: | pattern => body"""
    pattern: Expr | None  # None = wildcard (_)
    body: list[Statement]


@dataclass
class MatchExpr(Expr):
    """match subject | p1 => ... | p2 => ... end"""
    subject: Expr
    arms: list[MatchArm]


# ---------------------------------------------------------------------------
# Constraint Blocks (inside fn / gate / goal / reason)
# ---------------------------------------------------------------------------

@dataclass
class Constraint(ASTNode):
    """Base for must/allow/forbidden."""


@dataclass
class MustConstraint(Constraint):
    expr: Expr


@dataclass
class AllowConstraint(Constraint):
    capabilities: list[str]


@dataclass
class ForbiddenConstraint(Constraint):
    capabilities: list[str]


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass
class Param(ASTNode):
    name: str
    type_ann: TypeExpr


# ---------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------

@dataclass
class Declaration(ASTNode):
    """Base for top-level declarations."""


@dataclass
class FnDecl(Declaration):
    name: str
    params: list[Param]
    return_type: TypeExpr | None = None
    constraints: list[Constraint] = field(default_factory=list)
    body: list[Statement] = field(default_factory=list)


@dataclass
class GateDecl(Declaration):
    name: str
    params: list[Param]
    return_type: TypeExpr | None = None
    branches: int = 3
    collapse: str = "majority"
    threshold: float = 0.85
    fallback: str = "escalate"
    body: list[Statement] = field(default_factory=list)


@dataclass
class GoalDecl(Declaration):
    description: str
    constraints_list: list[str] = field(default_factory=list)
    quality_axes: list[str] = field(default_factory=list)
    explore_budget: float = 1.0
    body: list[Statement] = field(default_factory=list)


@dataclass
class ReasonDecl(Declaration):
    name: str
    params: list[Param]
    return_type: TypeExpr | None = None
    given: list[str] = field(default_factory=list)
    explore_expr: Expr | None = None
    evaluate_expr: Expr | None = None
    commit_strategy: str = "highest_consensus"
    body: list[Statement] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Program (root node)
# ---------------------------------------------------------------------------

@dataclass
class Program(ASTNode):
    declarations: list[Declaration | Statement] = field(default_factory=list)


# ---------------------------------------------------------------------------
# CIR / Belief constructs (additive — existing nodes unchanged)
# ---------------------------------------------------------------------------

@dataclass
class InquireExpr(ASTNode):
    prompt: str = ""
    agents: list[str] = field(default_factory=list)
    ttl: float | None = None


@dataclass
class BeliefDecl(Statement):
    name: str = ""
    type_ann: TypeExpr | None = None
    inquire_expr: "InquireExpr | None" = None


@dataclass
class ResolveStmt(Statement):
    target: str = ""
    threshold: float = 0.8
    strategy: str = "dempster_shafer"


@dataclass
class GuardStmt(Statement):
    target: str = ""
    max_risk: float = 0.2
    strategy: str = "both"


@dataclass
class EvolveStmt(Statement):
    target: str = ""
    condition: str = "stable"
    max_iter: int = 3


@dataclass
class SymbolDecl(Declaration):
    name: str = ""
    body: list[Statement] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ML Constructs — Phase 1 PyTorch compiler
# ---------------------------------------------------------------------------

@dataclass
class TensorType(TypeExpr):
    dtype: str = "Float"
    shape: list = field(default_factory=list)
    device: str = "cpu"
    causality: "str | None" = None
    privacy: "dict | None" = None

@dataclass
class VectorStoreType(TypeExpr):
    dim: int = 768
    capacity: int = 1_000_000

@dataclass
class SpikeTrainType(TypeExpr):
    dtype: str = "Float"
    neurons: int | str | None = None
    timesteps: int | str | None = None

@dataclass
class MultimodalType(TypeExpr):
    inner_type: "TypeExpr | None" = None

@dataclass
class MemPtrType(TypeExpr):
    inner_type: "TypeExpr | None" = None
    address: int | str | None = None

@dataclass
class ConstitutedType(TypeExpr):
    inner_type: "TypeExpr | None" = None

@dataclass
class GradType(TypeExpr):
    inner_type: "TypeExpr | None" = None

@dataclass
class OptimizerConfig(ASTNode):
    name: str = "Adam"
    params: dict = field(default_factory=dict)

@dataclass
class LossConfig(ASTNode):
    name: str = "CrossEntropy"

@dataclass
class LayerDecl(ASTNode):
    name: str = ""
    kind: str = ""
    in_dim: "int | None" = None
    out_dim: "int | None" = None
    repeat: int = 1
    config: dict = field(default_factory=dict)

@dataclass
class ForwardFn(ASTNode):
    params: list = field(default_factory=list)
    return_type: "TypeExpr | None" = None
    body: list = field(default_factory=list)

@dataclass
class EvolveConfig(ASTNode):
    metric: str = "val_loss"
    patience: int = 5
    max_iter: int = 50
    condition: str = "converged"

@dataclass
class ModelDecl(Declaration):
    name: str = ""
    layers: list = field(default_factory=list)
    forward_fn: "ForwardFn | None" = None
    retrieval: "RetrievalDecl | None" = None
    uncertainty: str = "none"
    constitution: "str | None" = None
    device: str = "cpu"
    config: dict = field(default_factory=dict)

@dataclass
class TrainStmt(Statement):
    model_name: str = ""
    dataset: str = ""
    optimizer: "OptimizerConfig | None" = None
    loss: "LossConfig | None" = None
    epochs: int = 10
    batch_size: int = 32
    guards: list = field(default_factory=list)
    evolve: "EvolveConfig | None" = None
    precision: str = "float32"
    config: dict = field(default_factory=dict)

@dataclass
class MoEBlock(ASTNode):
    name: str = ""
    n_experts: int = 8
    top_k: int = 2
    expert_dim: int = 4096
    config: dict = field(default_factory=dict)


@dataclass
class ExpertRoute(ASTNode):
    """ExpertRoute<n_experts, top_k> — MoE routing decision with uncertainty."""
    n_experts: int = 8
    top_k: int = 2
    uncertainty: float = 0.0  # BetaDist variance

@dataclass
class ConstitutionDecl(Declaration):
    name: str = ""
    principles: list = field(default_factory=list)
    critique_rounds: int = 2
    max_violation_score: float = 0.05

@dataclass
class ImportStmt(Statement):
    module: str = ""
    symbols: list = field(default_factory=list)
    alias: "str | None" = None


@dataclass
class RetrievalDecl(ASTNode):
    """RAG retrieval configuration attached to a model."""
    config: dict = field(default_factory=dict)


@dataclass
class CausalModelDecl(Declaration):
    name: str = ""
    config: dict = field(default_factory=dict)


@dataclass
class FederatedTrainStmt(Statement):
    name: str = ""
    config: dict = field(default_factory=dict)


@dataclass
class MetaTrainStmt(Statement):
    name: str = ""
    config: dict = field(default_factory=dict)


@dataclass
class SelfImproveDecl(Declaration):
    name: str = ""
    config: dict = field(default_factory=dict)


@dataclass
class SwarmDecl(Declaration):
    name: str = ""
    config: dict = field(default_factory=dict)


@dataclass
class ReplayBufferDecl(Declaration):
    name: str = ""
    config: dict = field(default_factory=dict)


@dataclass
class RewardSystemDecl(Declaration):
    name: str = ""
    config: dict = field(default_factory=dict)


@dataclass
class PredictiveCodingDecl(Declaration):
    name: str = ""
    config: dict = field(default_factory=dict)
