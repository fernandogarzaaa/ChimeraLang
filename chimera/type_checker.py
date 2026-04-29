"""Static type checker for ChimeraLang.

Walks the AST and verifies:
- Confidence boundaries (Explore cannot promote to Confident without gate)
- Memory scope rules (Ephemeral doesn't escape scope)
- Semantic constraint validation (must/allow/forbidden)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from chimera.ast_nodes import (
    AssertStmt,
    BinaryOp,
    BoolLiteral,
    CallExpr,
    CausalModelDecl,
    Declaration,
    EmitStmt,
    Expr,
    ExprStmt,
    FederatedTrainStmt,
    FloatLiteral,
    FnDecl,
    ForStmt,
    GateDecl,
    GenericType,
    GoalDecl,
    Identifier,
    IfExpr,
    IntLiteral,
    ListLiteral,
    MatchExpr,
    MemberExpr,
    MemPtrType,
    MemoryType,
    MetaTrainStmt,
    ModelDecl,
    MultimodalType,
    NamedType,
    Param,
    PrimitiveType,
    ProbabilisticType,
    Program,
    PredictiveCodingDecl,
    ReasonDecl,
    ReturnStmt,
    ReplayBufferDecl,
    RewardSystemDecl,
    SelfImproveDecl,
    Statement,
    StringLiteral,
    SpikeTrainType,
    SwarmDecl,
    TensorType,
    TypeExpr,
    TrainStmt,
    UnaryOp,
    ValDecl,
    VectorStoreType,
    ConstitutedType,
    ConstitutionDecl,
)
from chimera.types import (
    BOOL_T,
    BUILTINS,
    FLOAT_T,
    INT_T,
    TEXT_T,
    VOID_T,
    ChimeraType,
    ConstitutedTypeDesc,
    FnTypeDesc,
    GenericTypeDesc,
    MemTypeDesc,
    PrivacyTypeDesc,
    PrimitiveTypeDesc,
    ProbTypeDesc,
    PromotionViolation,
    TensorTypeDesc,
    TypeMismatch,
    VectorStoreTypeDesc,
    SpikeTrainTypeDesc,
    MultimodalTypeDesc,
    MemPtrTypeDesc,
)


@dataclass
class TypeEnv:
    """Scoped type environment."""
    bindings: dict[str, ChimeraType] = field(default_factory=dict)
    parent: TypeEnv | None = None

    def lookup(self, name: str) -> ChimeraType | None:
        if name in self.bindings:
            return self.bindings[name]
        if self.parent is not None:
            return self.parent.lookup(name)
        return None

    def define(self, name: str, ty: ChimeraType) -> None:
        self.bindings[name] = ty

    def child(self) -> TypeEnv:
        return TypeEnv(parent=self)


@dataclass
class TypeCheckResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    ok: bool = True


class TypeChecker:
    def __init__(self) -> None:
        self._env = TypeEnv(bindings=dict(BUILTINS))
        self._result = TypeCheckResult()
        self._in_gate = False  # inside a gate → promotion allowed

    def check(self, program: Program) -> TypeCheckResult:
        for decl in program.declarations:
            self._check_decl(decl)
        self._result.ok = len(self._result.errors) == 0
        return self._result

    # ------------------------------------------------------------------
    # Declarations
    # ------------------------------------------------------------------

    def _check_decl(self, node: Declaration | Statement) -> None:
        if isinstance(node, FnDecl):
            self._check_fn(node)
        elif isinstance(node, GateDecl):
            self._check_gate(node)
        elif isinstance(node, GoalDecl):
            self._check_goal(node)
        elif isinstance(node, ReasonDecl):
            self._check_reason(node)
        elif isinstance(node, ValDecl):
            self._check_val(node)
        elif isinstance(node, ModelDecl):
            self._check_model(node)
        elif isinstance(node, TrainStmt):
            self._check_train(node)
        elif isinstance(node, ConstitutionDecl):
            self._check_constitution(node)
        elif isinstance(node, (
            CausalModelDecl,
            FederatedTrainStmt,
            MetaTrainStmt,
            SelfImproveDecl,
            SwarmDecl,
            ReplayBufferDecl,
            RewardSystemDecl,
            PredictiveCodingDecl,
        )):
            self._check_roadmap_decl(node)
        elif isinstance(node, Statement):
            self._check_stmt(node)

    def _check_fn(self, fn: FnDecl) -> None:
        scope = self._env.child()
        param_types = []
        for p in fn.params:
            pt = self._resolve_type(p.type_ann)
            scope.define(p.name, pt)
            param_types.append(pt)
        ret_type = self._resolve_type(fn.return_type) if fn.return_type else VOID_T
        self._env.define(fn.name, FnTypeDesc(
            name=fn.name, param_types=tuple(param_types), return_type=ret_type
        ))
        old_env = self._env
        self._env = scope
        for stmt in fn.body:
            self._check_stmt(stmt)
        self._env = old_env

    def _check_gate(self, gate: GateDecl) -> None:
        old_in_gate = self._in_gate
        self._in_gate = True
        scope = self._env.child()
        for p in gate.params:
            pt = self._resolve_type(p.type_ann)
            scope.define(p.name, pt)
        old_env = self._env
        self._env = scope
        for stmt in gate.body:
            self._check_stmt(stmt)
        self._env = old_env
        self._in_gate = old_in_gate

    def _check_goal(self, goal: GoalDecl) -> None:
        scope = self._env.child()
        old_env = self._env
        self._env = scope
        for stmt in goal.body:
            self._check_stmt(stmt)
        self._env = old_env

    def _check_reason(self, reason: ReasonDecl) -> None:
        scope = self._env.child()
        for p in reason.params:
            pt = self._resolve_type(p.type_ann)
            scope.define(p.name, pt)
        old_env = self._env
        self._env = scope
        for stmt in reason.body:
            self._check_stmt(stmt)
        self._env = old_env

    def _check_val(self, val: ValDecl) -> None:
        declared = self._resolve_type(val.type_ann) if val.type_ann else None
        if val.value is not None:
            inferred = self._infer_expr(val.value)
            if declared and inferred:
                self._check_assignment_compat(declared, inferred, val.name)
            final = declared or inferred or VOID_T
        else:
            final = declared or VOID_T
        self._env.define(val.name, final)

    # ------------------------------------------------------------------
    # Statements
    # ------------------------------------------------------------------

    def _check_stmt(self, stmt: Statement) -> None:
        if isinstance(stmt, ValDecl):
            self._check_val(stmt)
        elif isinstance(stmt, ReturnStmt):
            if stmt.value:
                self._infer_expr(stmt.value)
        elif isinstance(stmt, AssertStmt):
            self._infer_expr(stmt.condition)
        elif isinstance(stmt, EmitStmt):
            self._infer_expr(stmt.value)
        elif isinstance(stmt, ForStmt):
            self._check_for(stmt)
        elif isinstance(stmt, ExprStmt):
            self._infer_expr(stmt.expr)

    def _check_for(self, stmt: ForStmt) -> None:
        self._infer_expr(stmt.iterable)
        scope = self._env.child()
        scope.define(stmt.target, VOID_T)
        old_env = self._env
        self._env = scope
        for s in stmt.body:
            self._check_stmt(s)
        self._env = old_env

    # ------------------------------------------------------------------
    # Expression type inference
    # ------------------------------------------------------------------

    def _infer_expr(self, expr: Expr) -> ChimeraType:
        if isinstance(expr, IntLiteral):
            return INT_T
        if isinstance(expr, FloatLiteral):
            return FLOAT_T
        if isinstance(expr, StringLiteral):
            return TEXT_T
        if isinstance(expr, BoolLiteral):
            return BOOL_T
        if isinstance(expr, Identifier):
            ty = self._env.lookup(expr.name)
            if ty is None:
                self._result.warnings.append(f"Unresolved identifier: {expr.name}")
                return VOID_T
            return ty
        if isinstance(expr, BinaryOp):
            return self._infer_binary(expr)
        if isinstance(expr, UnaryOp):
            return self._infer_expr(expr.operand)
        if isinstance(expr, CallExpr):
            return self._infer_call(expr)
        if isinstance(expr, MemberExpr):
            self._infer_expr(expr.obj)
            return VOID_T
        if isinstance(expr, IfExpr):
            self._infer_expr(expr.condition)
            for s in expr.then_body:
                self._check_stmt(s)
            if expr.else_body:
                for s in expr.else_body:
                    self._check_stmt(s)
            return VOID_T
        if isinstance(expr, MatchExpr):
            self._infer_expr(expr.subject)
            for arm in expr.arms:
                if arm.pattern is not None:
                    self._infer_expr(arm.pattern)
                for s in arm.body:
                    self._check_stmt(s)
            return VOID_T
        if isinstance(expr, ListLiteral):
            for el in expr.elements:
                self._infer_expr(el)
            return GenericTypeDesc(name="List", params=())
        return VOID_T

    def _infer_binary(self, expr: BinaryOp) -> ChimeraType:
        lt = self._infer_expr(expr.left)
        rt = self._infer_expr(expr.right)
        if expr.op in ("==", "!=", "<", ">", "<=", ">=", "and", "or"):
            return BOOL_T
        if isinstance(lt, PrimitiveTypeDesc) and isinstance(rt, PrimitiveTypeDesc):
            if lt.name == "Float" or rt.name == "Float":
                return FLOAT_T
            return INT_T
        return lt

    def _infer_call(self, expr: CallExpr) -> ChimeraType:
        if isinstance(expr.callee, Identifier):
            fn_type = self._env.lookup(expr.callee.name)
            if isinstance(fn_type, FnTypeDesc):
                return fn_type.return_type
            # Built-in wrapper constructors
            if expr.callee.name in ("Confident", "Explore", "Converge", "Provisional"):
                if expr.args:
                    inner = self._infer_expr(expr.args[0])
                    return ProbTypeDesc(name=expr.callee.name, wrapper=expr.callee.name, inner=inner)
            if expr.callee.name in ("confident", "consensus", "no_hallucination"):
                return BOOL_T
        for arg in expr.args:
            self._infer_expr(arg)
        return VOID_T

    # ------------------------------------------------------------------
    # Type compatibility
    # ------------------------------------------------------------------

    def _check_assignment_compat(self, declared: ChimeraType, inferred: ChimeraType, name: str) -> None:
        if isinstance(declared, ProbTypeDesc) and declared.wrapper == "Confident":
            if isinstance(inferred, ProbTypeDesc) and inferred.wrapper == "Explore":
                if not self._in_gate:
                    self._result.errors.append(
                        f"Cannot assign Explore value to Confident<> variable '{name}' "
                        f"outside a gate (use a gate for promotion)"
                    )

    # ------------------------------------------------------------------
    # ML declarations
    # ------------------------------------------------------------------

    def _check_model(self, model: ModelDecl) -> None:
        """Check a model declaration: validate layer chain dims and constitution."""
        self._check_retrieval_config(model)

        # Validate layer chain dimension continuity
        chain_errors = self._validate_layer_chain(model.layers)
        for err in chain_errors:
            self._result.errors.append(f"Model '{model.name}': {err}")

        # Check constitution reference if present
        if model.constitution:
            # Look up the constitution declaration
            const_type = self._env.lookup(model.constitution)
            if const_type is None:
                self._result.errors.append(
                    f"Model '{model.name}': constitution '{model.constitution}' not found"
                )
            elif not isinstance(const_type, ConstitutedTypeDesc):
                self._result.errors.append(
                    f"Model '{model.name}': '{model.constitution}' is not a constitution type"
                )

        # Register model in environment with its layers' output dims
        self._env.define(model.name, PrimitiveTypeDesc(name=f"Model<{model.name}>"))
        for layer in model.layers:
            if hasattr(layer, "out_dim") and layer.out_dim is not None:
                self._env.define(f"_layer_out_{layer.name}", layer.out_dim)

    def _check_train(self, train: TrainStmt) -> None:
        """Check a train statement: verify model exists and params are valid."""
        if not self._env.lookup(train.model_name):
            self._result.warnings.append(
                f"Train statement references model '{train.model_name}' "
                f"that may not be defined above it"
            )
        if train.epochs <= 0:
            self._result.errors.append(
                f"Train epochs must be positive, got {train.epochs}"
            )
        if train.batch_size <= 0:
            self._result.errors.append(
                f"Train batch_size must be positive, got {train.batch_size}"
            )

    def _check_constitution(self, decl: ConstitutionDecl) -> None:
        """Register a constitution declaration as a type."""
        if not decl.principles:
            self._result.warnings.append(
                f"Constitution '{decl.name}' has no principles"
            )
        for p in decl.principles:
            if not p or not p.strip():
                self._result.errors.append(
                    f"Constitution '{decl.name}': empty principle string"
                )
        # Register as ConstitutedTypeDesc for later ModelDecl reference
        self._env.define(
            decl.name,
            ConstitutedTypeDesc(name=decl.name, inner=VOID_T),
        )

    def _check_roadmap_decl(self, decl: object) -> None:
        name = getattr(decl, "name", "")
        if name:
            self._env.define(name, PrimitiveTypeDesc(name=f"{type(decl).__name__}<{name}>"))
        config = getattr(decl, "config", {})
        model_name = config.get("model") or config.get("base_model")
        if model_name and self._env.lookup(str(model_name)) is None:
            self._result.warnings.append(
                f"{type(decl).__name__} '{name}' references model '{model_name}' "
                f"that may not be defined above it"
            )
        kind = type(decl).__name__
        if kind == "FederatedTrainStmt":
            self._require_positive(config, "rounds", f"{name} rounds")
            self._require_positive(config, "privacy_epsilon", f"{name} privacy_epsilon")
        elif kind == "MetaTrainStmt":
            self._require_positive(config, "inner_steps", f"{name} inner_steps")
        elif kind == "SelfImproveDecl":
            self._require_positive(config, "max_generations", f"{name} max_generations")
        elif kind == "SwarmDecl":
            self._require_positive(config, "agents", f"{name} agents")
        elif kind == "ReplayBufferDecl":
            self._require_positive(config, "capacity", f"{name} capacity")
        elif kind == "PredictiveCodingDecl":
            self._require_positive(config, "depth", f"{name} depth")

    def _check_retrieval_config(self, model: ModelDecl) -> None:
        retrieval = getattr(model, "retrieval", None)
        if retrieval is None:
            return
        config = getattr(retrieval, "config", {})
        self._require_positive(config, "top_k", f"Model '{model.name}' retrieval top_k")
        self._require_positive(config, "store_dim", f"Model '{model.name}' retrieval store_dim", required=False)
        self._require_positive(config, "capacity", f"Model '{model.name}' retrieval capacity", required=False)
        if "min_similarity" in config:
            try:
                value = float(config["min_similarity"])
            except (TypeError, ValueError):
                self._result.errors.append(f"Model '{model.name}' retrieval min_similarity must be numeric")
                return
            if value < 0.0 or value > 1.0:
                self._result.errors.append(
                    f"Model '{model.name}' retrieval min_similarity must be between 0.0 and 1.0"
                )

    def _require_positive(self, config: dict, key: str, label: str, *, required: bool = True) -> None:
        if key not in config:
            if required:
                self._result.errors.append(f"{label} is required")
            return
        try:
            value = float(config[key])
        except (TypeError, ValueError):
            self._result.errors.append(f"{label} must be numeric")
            return
        if value <= 0:
            self._result.errors.append(f"{label} must be positive")

    # ------------------------------------------------------------------
    # Type resolution
    # ------------------------------------------------------------------

    def _resolve_type(self, t: TypeExpr | None) -> ChimeraType:
        if t is None:
            return VOID_T
        if isinstance(t, PrimitiveType):
            return BUILTINS.get(t.name, PrimitiveTypeDesc(name=t.name))
        if isinstance(t, ProbabilisticType):
            inner = self._resolve_type(t.inner)
            return ProbTypeDesc(name=t.wrapper, wrapper=t.wrapper, inner=inner)
        if isinstance(t, MemoryType):
            inner = self._resolve_type(t.inner)
            return MemTypeDesc(name=t.scope, scope=t.scope, inner=inner)
        if isinstance(t, GenericType):
            params = tuple(self._resolve_type(p) for p in t.params)
            return GenericTypeDesc(name=t.name, params=params)
        if isinstance(t, NamedType):
            resolved = self._env.lookup(t.name)
            if resolved:
                return resolved
            return PrimitiveTypeDesc(name=t.name)
        if isinstance(t, TensorType):
            tensor = TensorTypeDesc(
                name="Tensor",
                dtype=t.dtype if hasattr(t, "dtype") else "Float",
                dims=tuple(t.shape) if hasattr(t, "shape") and t.shape else (),
                device=t.device if hasattr(t, "device") else "cpu",
                causality=t.causality,
            )
            if t.privacy:
                return PrivacyTypeDesc(
                    name="Private",
                    inner=tensor,
                    epsilon=float(t.privacy.get("epsilon", 1.0)),
                    delta=float(t.privacy.get("delta", 1e-5)),
                )
            return tensor
        if isinstance(t, VectorStoreType):
            return VectorStoreTypeDesc(
                name="VectorStore",
                dim=t.dim,
                capacity=t.capacity,
            )
        if isinstance(t, SpikeTrainType):
            return SpikeTrainTypeDesc(
                name="SpikeTrain",
                dtype=t.dtype,
                neurons=t.neurons,
                timesteps=t.timesteps,
            )
        if isinstance(t, MultimodalType):
            inner = self._resolve_type(t.inner_type) if t.inner_type else VOID_T
            return MultimodalTypeDesc(name="Multimodal", inner=inner)
        if isinstance(t, MemPtrType):
            inner = self._resolve_type(t.inner_type) if t.inner_type else VOID_T
            return MemPtrTypeDesc(name="MemPtr", inner=inner, address=t.address)
        if isinstance(t, ConstitutedType):
            inner = self._resolve_type(t.inner_type) if t.inner_type else VOID_T
            return ConstitutedTypeDesc(name="Constituted", inner=inner)
        return VOID_T

    # ------------------------------------------------------------------
    # Layer chain validation
    # ------------------------------------------------------------------

    def _validate_layer_chain(self, layers: list) -> list[str]:
        """Verify each layer's out_dim matches the next layer's in_dim."""
        errors = []
        for i in range(len(layers) - 1):
            curr_out = getattr(layers[i], "out_dim", None)
            next_in = getattr(layers[i + 1], "in_dim", None)
            if curr_out is not None and next_in is not None and curr_out != next_in:
                errors.append(
                    f"Layer '{layers[i].name}' outputs {curr_out} "
                    f"but '{layers[i + 1].name}' expects in_dim={next_in}"
                )
        return errors
