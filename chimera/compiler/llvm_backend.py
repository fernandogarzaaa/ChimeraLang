"""LLVM IR backend for ChimeraLang (experimental scaffolding).

Emits LLVM IR text from ChimeraLang AST for the native compiler path
(Phase 2). The current emitter produces a structural skeleton — function
signatures, basic-block labels, and per-layer commentary — that captures
the model architecture but is NOT YET guaranteed to assemble cleanly under
``llvm-as``. Several layer bodies reference SSA values whose definitions
are pending. Treat the output as documentation-grade IR scaffolding until
the layer emitters are completed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import math

from chimera.ast_nodes import (
    LayerDecl,
    ModelDecl,
    TrainStmt,
    OptimizerConfig,
    LossConfig,
    ForwardFn,
    Statement,
    Expr,
    ValDecl,
    ReturnStmt,
    ForStmt,
    Identifier,
    CallExpr,
    BinaryOp,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize(name: str) -> str:
    """Mangle ChimeraLang identifier to be LLVM-safe."""
    return name.replace(".", "_").replace("-", "_").replace(":", "_")


def _ty(dtype: str) -> str:
    map_ = {"Float": "float", "Int": "i32", "Bool": "i1", "Double": "double"}
    return map_.get(dtype, "float")


# ---------------------------------------------------------------------------
# IR Builder
# ---------------------------------------------------------------------------

@dataclass
class BasicBlock:
    name: str
    instructions: list[str] = field(default_factory=list)
    terminator: str | None = None

    def add(self, line: str) -> None:
        self.instructions.append(line)

    def set_terminator(self, term: str) -> None:
        self.terminator = term


class LLVMEmitter:
    """Builds LLVM IR text from ChimeraLang AST."""

    def __init__(self) -> None:
        self._lines: list[str] = []
        self._indent = 0
        self._blocks: list[BasicBlock] = []
        self._block_id = 0
        self._local_id = 0
        self._fn_ret_type = "float"
        self._fn_param_types: list[str] = []
        self._model_name = ""
        self._layers: list[LayerDecl] = []
        self._tensor_shapes: dict[str, tuple[int, ...]] = {}
        self._train_stmts: list[TrainStmt] = []

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def emit(self, program: object) -> str:
        """Convert a ChimeraLang Program to LLVM IR text."""
        self._lines = []
        self._emit_header()

        # Handle standalone statements (TrainStmt, ModelDecl, etc.) passed directly
        decls: list
        if hasattr(program, "declarations"):
            decls = list(program.declarations)
        else:
            # Single declaration / statement passed directly (e.g. during testing)
            decls = [program]

        for decl in decls:
            self._dispatch(decl)
        return "\n".join(self._lines)

    # ------------------------------------------------------------------
    # Top-level dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, decl: object) -> None:
        cls = type(decl).__name__
        if cls == "ModelDecl" or cls == "_ModelDecl":
            self._gen_model(decl)
        elif cls == "TrainStmt" or cls == "_TrainStmt":
            self._gen_train(decl)
        elif cls == "ConstitutionDecl" or cls == "_ConstitutionDecl":
            self._gen_constitution(decl)

    # ------------------------------------------------------------------
    # Header / declarations
    # ------------------------------------------------------------------

    def _emit_header(self) -> None:
        self._emit('; Module ID "chimeralang.module"')
        self._emit("source_filename = \"chimeralang.module\"")
        self._emit("")
        self._emit("target datalayout = \"e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128\"")
        self._emit('target triple = "x86_64-unknown-linux-gnu"')
        self._emit("")
        self._emit("; === Declarations ===")
        self._emit("declare float @expf(float)")
        self._emit("declare float @sqrtf(float)")
        self._emit("declare float @tanhf(float)")
        self._emit("declare i32 @printf(i8*, ...)")
        self._emit("")

    # ------------------------------------------------------------------
    # Constitution declaration (no-op in pure LLVM — handled at runtime)
    # ------------------------------------------------------------------

    def _gen_constitution(self, decl: object) -> None:
        name = getattr(decl, "name", "Constitution")
        self._emit(f"; Constitution: {name}")
        for p in getattr(decl, "principles", []):
            self._emit(f";   principle: {p}")
        self._emit("")

    # ------------------------------------------------------------------
    # Model declaration → LLVM function
    # ------------------------------------------------------------------

    def _gen_model(self, model: ModelDecl | object) -> None:
        self._model_name = model.name
        self._layers = list(getattr(model, "layers", []))
        self._tensor_shapes = {}

        # Build shape map from layer dimensions
        self._infer_shapes()

        # Build parameter types for forward function
        # First layer's in_dim is the input shape[0], others inferred
        first_in = getattr(self._layers[0], "in_dim", None) if self._layers else None
        if first_in is not None:
            param_count = 1  # x input tensor
        else:
            param_count = 1

        ret_type = self._fn_ret_type
        self._emit(f"define dso_local {ret_type}* @{_sanitize(model.name)}_forward(")
        # Emit parameters
        params = self._model_param_sig()
        self._emit(f"  {params}")
        self._emit(") {")
        with self._indented():
            self._emit("entry:")
            self._gen_forward_body(model)
        self._emit("}")
        self._emit("")

    def _model_param_sig(self) -> str:
        """Return LLVM parameter signature for model forward."""
        parts = ["float* %x", "i64 %batch_size"]
        return ",\n  ".join(parts)

    def _infer_shapes(self) -> None:
        """Infer tensor shapes across the layer chain."""
        for i, layer in enumerate(self._layers):
            in_d = getattr(layer, "in_dim", None)
            out_d = getattr(layer, "out_dim", None)
            if in_d is not None and out_d is not None:
                self._tensor_shapes[layer.name] = (in_d, out_d)

    def _gen_forward_body(self, model: ModelDecl | object) -> None:
        layers = self._layers
        if not layers:
            self._emit("%0 = load float*, float** null")
            self._emit("ret float* %0")
            return

        # Allocate input %x
        self._emit("%x_ptr = alloca float*")
        self._emit("store float* %x, float** %x_ptr")

        prev_tensor = "%x"
        current_dim = getattr(layers[0], "in_dim", None) if layers else None
        for layer in layers:
            if getattr(layer, "in_dim", None) is None and current_dim is not None:
                layer.in_dim = current_dim
            if getattr(layer, "out_dim", None) is None and layer.kind in (
                "ReLU", "Softmax", "Dropout", "LayerNorm", "Sigmoid", "Tanh", "GELU"
            ):
                layer.out_dim = getattr(layer, "in_dim", None) or current_dim
            prev_tensor = self._gen_layer(layer, prev_tensor)
            self._tensor_shapes[f"h_{layer.name}"] = (getattr(layer, "out_dim", 0),)
            current_dim = getattr(layer, "out_dim", None) or current_dim

        # Store final result
        out_dim = (getattr(layers[-1], "out_dim", None) if layers else None) or current_dim or 0
        if out_dim > 0:
            self._emit(f"%result = alloca float*")
            self._emit(f"store float* {prev_tensor}, float** %result")

        # Return pointer
        self._emit("ret float* %result" if out_dim > 0 else "ret float* null")

    def _gen_layer(self, layer: LayerDecl, input_tensor: str) -> str:
        kind = layer.kind
        name = layer.name
        in_d = getattr(layer, "in_dim", None) or 0
        out_d = getattr(layer, "out_dim", None) or 0

        if kind in ("Dense", "Linear"):
            return self._gen_dense(layer, input_tensor, in_d, out_d)
        elif kind == "ReLU":
            return self._gen_relu(input_tensor, in_d, out_d)
        elif kind == "Softmax":
            return self._gen_softmax(input_tensor, out_d)
        elif kind == "Dropout":
            return self._gen_dropout(layer, input_tensor, out_d)
        elif kind == "LayerNorm":
            return self._gen_layernorm(layer, input_tensor, out_d)
        elif kind == "Sigmoid":
            return self._gen_sigmoid(input_tensor, out_d)
        elif kind == "Tanh":
            return self._gen_tanh(input_tensor, out_d)
        elif kind == "GELU":
            return self._gen_gelu(input_tensor, out_d)
        elif kind == "MoE":
            return self._gen_moe(layer, input_tensor)
        else:
            # Identity / passthrough for unknown layers
            self._emit(f"; unknown layer {kind} — passthrough")
            return input_tensor

    # ------------------------------------------------------------------
    # Layer implementations
    # ------------------------------------------------------------------

    def _gen_dense(self, layer: LayerDecl, input_tensor: str, in_d: int, out_d: int) -> str:
        """Dense/Linear layer: y = xW^T + b — emits structured LLVM IR."""
        self._emit(f"; Dense {in_d} -> {out_d}")

        # Allocate output
        result_ptr = self._fresh("%dense_out")
        self._emit(f"{result_ptr} = alloca float, i64 {out_d}")

        # Initialize output to zero
        self._emit(f"%zero = fmul float 0.0, 1.0")
        for j in range(min(out_d, 4)):
            idx_p = self._fresh("%idx")
            self._emit(f"{idx_p} = getelementptr float, float* {result_ptr}, i64 {j}")
            self._emit(f"store float %zero, float* {idx_p}")

        # Outer loop over output dimension
        j_loop = self._fresh_block("dense.j")
        j_done = self._fresh_block("dense.j.end")
        j_ptr = self._fresh("%j")
        self._emit(f"{j_ptr} = alloca i64")
        self._emit(f"store i64 0, i64* {j_ptr}")
        self._emit(f"br label %{j_loop.name}")

        self._blocks.append(j_loop)
        self._emit(f"{j_loop.name}:")
        self._emit(f"  %j_val = load i64, i64* {j_ptr}")
        self._emit(f"  %j_cmp = icmp slt i64 %j_val, {out_d}")
        self._emit(f"  br i1 %j_cmp, label %{self._fresh_block('dense.i').name}, label %{j_done.name}")

        # Inner loop over input dimension (dot product)
        i_loop = self._fresh_block("dense.i")
        i_done = self._fresh_block("dense.i.end")
        self._blocks.append(i_loop)
        self._emit(f"{i_loop.name}:")
        self._emit(f"  %i_val = load i64, i64* {self._fresh('%i')}")
        self._emit(f"  %i_cmp = icmp slt i64 %i_val, {in_d}")
        self._emit(f"  br i1 %i_cmp, label %{self._fresh_block('dense.body').name}, label %{i_done.name}")

        # Inner body: accumulator += input[i] * weight[i][j]
        body_bb = self._fresh_block("dense.body")
        self._blocks.append(body_bb)
        self._emit(f"{body_bb.name}:")
        acc_ptr = self._fresh("%acc")
        self._emit(f"{acc_ptr} = load float, float* {self._fresh('%out_p')}")
        x_i_ptr = self._fresh("%x_i")
        w_ij_ptr = self._fresh("%w_ij")
        self._emit(f"{x_i_ptr} = getelementptr float, float* {input_tensor}, i64 %i_val")
        self._emit(f"  ; weight index: i*{out_d}+j (row-major)")
        w_base = self._fresh("%w_base")
        self._emit(f"{w_base} = getelementptr float, float* {self._fresh('%w')}, i64 0")
        w_ij = self._fresh("%w_ij_val")
        self._emit(f"{w_ij} = load float, float* {w_ij_ptr}")
        x_i = self._fresh("%x_i_val")
        self._emit(f"{x_i} = load float, float* {x_i_ptr}")
        prod = self._fresh("%prod")
        self._emit(f"{prod} = fmul float {x_i}, {w_ij}")
        new_acc = self._fresh("%new_acc")
        self._emit(f"{new_acc} = fadd float {acc_ptr}, {prod}")
        self._emit(f"store float {new_acc}, float* {acc_ptr}")
        self._emit(f"br label %{i_done.name}")

        # i_done: increment i, back to i_loop
        self._blocks.append(i_done)
        self._emit(f"{i_done.name}:")
        self._emit(f"  %i_next = add i64 %i_val, 1")
        self._emit(f"  store i64 %i_next, i64* {self._fresh('%i')}")
        self._emit(f"  br label %{i_loop.name}")

        # Add bias and store final result for this j
        bias_j = self._fresh("%bias_j")
        self._emit(f"{bias_j} = getelementptr float, float* {self._fresh('%bias')}, i64 %j_val")
        bias_val = self._fresh("%bias_val")
        self._emit(f"{bias_val} = load float, float* {bias_j}")
        out_j = self._fresh("%out_j")
        self._emit(f"{out_j} = getelementptr float, float* {result_ptr}, i64 %j_val")
        final_val = self._fresh("%final")
        self._emit(f"{final_val} = fadd float {self._fresh('%acc_final')}, {bias_val}")
        self._emit(f"store float {final_val}, float* {out_j}")

        # j_done: increment j, back to j_loop
        self._blocks.append(j_done)
        self._emit(f"{j_done.name}:")
        self._emit(f"  %j_next = add i64 %j_val, 1")
        self._emit(f"  store i64 %j_next, i64* {j_ptr}")
        self._emit(f"  br label %{j_loop.name}")

        return result_ptr

    def _gen_relu(self, input_tensor: str, in_d: int, out_d: int) -> str:
        """ReLU: y = max(0, x) — emits element-wise loop with fcmp + select."""
        self._emit(f"; ReLU({in_d} -> {out_d})")
        result_ptr = self._fresh("%relu_out")
        self._emit(f"{result_ptr} = alloca float, i64 {out_d}")

        loop = self._fresh_block("relu.loop")
        done = self._fresh_block("relu.done")
        i_ptr = self._fresh("%i")
        self._emit(f"{i_ptr} = alloca i64")
        self._emit(f"store i64 0, i64* {i_ptr}")
        self._emit(f"br label %{loop.name}")

        self._blocks.append(loop)
        self._emit(f"{loop.name}:")
        self._emit(f"  %i_val = load i64, i64* {i_ptr}")
        self._emit(f"  %cmp = icmp slt i64 %i_val, {out_d}")
        self._emit(f"  br i1 %cmp, label %{self._fresh_block('relu.body').name}, label %{done.name}")

        body = self._fresh_block("relu.body")
        self._blocks.append(body)
        self._emit(f"{body.name}:")
        x_ptr = self._fresh("%x_ptr")
        self._emit(f"{x_ptr} = getelementptr float, float* {input_tensor}, i64 %i_val")
        x_val = self._fresh("%x")
        self._emit(f"{x_val} = load float, float* {x_ptr}")
        zero = self._fresh("%zero")
        self._emit(f"{zero} = fmul float 0.0, 1.0")
        y_val = self._fresh("%y")
        self._emit(f"{y_val} = fcmp oge float {x_val}, {zero}   ; x >= 0?")
        selected = self._fresh("%sel")
        self._emit(f"{selected} = select i1 {y_val}, float {x_val}, float {zero}")
        out_ptr = self._fresh("%out_p")
        self._emit(f"{out_ptr} = getelementptr float, float* {result_ptr}, i64 %i_val")
        self._emit(f"store float {selected}, float* {out_ptr}")
        self._emit(f"br label %{done.name}")

        done_bb = self._fresh_block("relu.done.cont")
        self._blocks.append(done_bb)
        self._emit(f"{done.name}:")
        self._emit(f"  %i_next = add i64 %i_val, 1")
        self._emit(f"  store i64 %i_next, i64* {i_ptr}")
        self._emit(f"  br label %{loop.name}")

        self._blocks.append(done_bb)
        self._emit(f"{done_bb.name}:")
        return result_ptr

    def _gen_softmax(self, input_tensor: str, dim: int) -> str:
        """Softmax: y[i] = exp(x[i]) / sum_j(exp(x[j])) — emits two-pass LLVM IR."""
        self._emit(f"; Softmax(dim={dim})")
        result_ptr = self._fresh("%softmax_out")
        self._emit(f"{result_ptr} = alloca float, i64 {dim}")

        # --- Pass 1: compute max of input (for numerical stability) ---
        max_ptr = self._fresh("%max")
        self._emit(f"{max_ptr} = alloca float")
        init_max = self._fresh("%init_max")
        self._emit(f"{init_max} = fmul float 0.0, 1.0")
        self._emit(f"store float {init_max}, float* {max_ptr}")

        max_loop = self._fresh_block("softmax.max.loop")
        max_done = self._fresh_block("softmax.max.done")
        i_ptr = self._fresh("%sm_i")
        self._emit(f"{i_ptr} = alloca i64")
        self._emit(f"store i64 0, i64* {i_ptr}")
        self._emit(f"br label %{max_loop.name}")

        self._blocks.append(max_loop)
        self._emit(f"{max_loop.name}:")
        self._emit(f"  %sm_i_val = load i64, i64* {i_ptr}")
        self._emit(f"  %sm_cmp = icmp slt i64 %sm_i_val, {dim}")
        self._emit(f"  br i1 %sm_cmp, label %{self._fresh_block('softmax.max.body').name}, label %{max_done.name}")

        max_body = self._fresh_block("softmax.max.body")
        self._blocks.append(max_body)
        self._emit(f"{max_body.name}:")
        x_ptr = self._fresh("%sm_x")
        self._emit(f"{x_ptr} = getelementptr float, float* {input_tensor}, i64 %sm_i_val")
        x_val = self._fresh("%sm_xv")
        self._emit(f"{x_val} = load float, float* {x_ptr}")
        cur_max = self._fresh("%cm")
        self._emit(f"{cur_max} = load float, float* {max_ptr}")
        new_max = self._fresh("%nm")
        self._emit(f"{new_max} = fcmp ogt float {x_val}, {cur_max}")
        sel_max = self._fresh("%smax")
        self._emit(f"{sel_max} = select i1 {new_max}, float {x_val}, float {cur_max}")
        self._emit(f"store float {sel_max}, float* {max_ptr}")
        self._emit(f"br label %{max_done.name}")

        self._blocks.append(max_done)
        self._emit(f"{max_done.name}:")
        self._emit(f"  %sm_i_next = add i64 %sm_i_val, 1")
        self._emit(f"  store i64 %sm_i_next, i64* {i_ptr}")
        self._emit(f"  br label %{max_loop.name}")

        max_cont = self._fresh_block("softmax.max.cont")
        self._blocks.append(max_cont)
        self._emit(f"{max_cont.name}:")

        # --- Pass 2: compute sum of exp(x_i - max) ---
        sum_ptr = self._fresh("%sum")
        self._emit(f"{sum_ptr} = alloca float")
        self._emit(f"store float %zero, float* {sum_ptr}")

        sum_loop = self._fresh_block("softmax.sum.loop")
        sum_done = self._fresh_block("softmax.sum.done")
        self._emit(f"store i64 0, i64* {i_ptr}")
        self._emit(f"br label %{sum_loop.name}")

        self._blocks.append(sum_loop)
        self._emit(f"{sum_loop.name}:")
        self._emit(f"  %si = load i64, i64* {i_ptr}")
        self._emit(f"  %sc = icmp slt i64 %si, {dim}")
        self._emit(f"  br i1 %sc, label %{self._fresh_block('softmax.sum.body').name}, label %{sum_done.name}")

        sum_body = self._fresh_block("softmax.sum.body")
        self._blocks.append(sum_body)
        self._emit(f"{sum_body.name}:")
        xs_ptr = self._fresh("%sxp")
        self._emit(f"{xs_ptr} = getelementptr float, float* {input_tensor}, i64 %si")
        xv = self._fresh("%sxv")
        self._emit(f"{xv} = load float, float* {xs_ptr}")
        mxv = self._fresh("%mxv")
        self._emit(f"{mxv} = load float, float* {max_ptr}")
        diff = self._fresh("%diff")
        self._emit(f"{diff} = fsub float {xv}, {mxv}")
        exp_diff = self._fresh("%expd")
        self._emit(f"{exp_diff} = call float @expf(float {diff})")
        cur_sum = self._fresh("%csum")
        self._emit(f"{cur_sum} = load float, float* {sum_ptr}")
        new_sum = self._fresh("%nsum")
        self._emit(f"{new_sum} = fadd float {cur_sum}, {exp_diff}")
        self._emit(f"store float {new_sum}, float* {sum_ptr}")
        self._emit(f"br label %{sum_done.name}")

        self._blocks.append(sum_done)
        self._emit(f"{sum_done.name}:")
        self._emit(f"  %ssi = add i64 %si, 1")
        self._emit(f"  store i64 %ssi, i64* {i_ptr}")
        self._emit(f"  br label %{sum_loop.name}")

        sum_cont = self._fresh_block("softmax.sum.cont")
        self._blocks.append(sum_cont)
        self._emit(f"{sum_cont.name}:")

        # --- Pass 3: compute exp(x_i - max) / sum ---
        out_loop = self._fresh_block("softmax.out.loop")
        out_done = self._fresh_block("softmax.out.done")
        self._emit(f"store i64 0, i64* {i_ptr}")
        self._emit(f"br label %{out_loop.name}")

        self._blocks.append(out_loop)
        self._emit(f"{out_loop.name}:")
        self._emit(f"  %oi = load i64, i64* {i_ptr}")
        self._emit(f"  %oc = icmp slt i64 %oi, {dim}")
        self._emit(f"  br i1 %oc, label %{self._fresh_block('softmax.out.body').name}, label %{out_done.name}")

        out_body = self._fresh_block("softmax.out.body")
        self._blocks.append(out_body)
        self._emit(f"{out_body.name}:")
        xo_ptr = self._fresh("%xop")
        self._emit(f"{xo_ptr} = getelementptr float, float* {input_tensor}, i64 %oi")
        xov = self._fresh("%xov")
        self._emit(f"{xov} = load float, float* {xo_ptr}")
        mxo = self._fresh("%mxo")
        self._emit(f"{mxo} = load float, float* {max_ptr}")
        diff_o = self._fresh("%difo")
        self._emit(f"{diff_o} = fsub float {xov}, {mxo}")
        exp_o = self._fresh("%expo")
        self._emit(f"{exp_o} = call float @expf(float {diff_o})")
        denom = self._fresh("%denom")
        self._emit(f"{denom} = load float, float* {sum_ptr}")
        fraction = self._fresh("%frac")
        self._emit(f"{fraction} = fdiv float {exp_o}, {denom}")
        out_j = self._fresh("%outj")
        self._emit(f"{out_j} = getelementptr float, float* {result_ptr}, i64 %oi")
        self._emit(f"store float {fraction}, float* {out_j}")
        self._emit(f"br label %{out_done.name}")

        self._blocks.append(out_done)
        self._emit(f"{out_done.name}:")
        self._emit(f"  %ooi = add i64 %oi, 1")
        self._emit(f"  store i64 %ooi, i64* {i_ptr}")
        self._emit(f"  br label %{out_loop.name}")

        out_cont = self._fresh_block("softmax.out.cont")
        self._blocks.append(out_cont)
        self._emit(f"{out_cont.name}:")
        return result_ptr

    def _gen_dropout(self, layer: LayerDecl, input_tensor: str, dim: int) -> str:
        """Dropout: at train time multiply by 1/p; at infer time pass through."""
        p = layer.config.get("p", 0.1) if hasattr(layer, "config") else 0.1
        self._emit(f"; Dropout(p={p}, dim={dim})")
        self._emit(f";   NOTE: runtime toggle — training mode applies mask and scales")
        result_ptr = self._fresh("%dropout_out")
        self._emit(f"{result_ptr} = alloca float, i64 {dim}")
        self._emit(f"  ; In training: generate uniform random [0,1) per element,")
        self._emit(f"  ;   multiply by 1/{p} if rand > {p}, else zero.")
        self._emit(f"  ; In inference: memcpy input to output (identity).")
        return result_ptr

    def _gen_layernorm(self, layer: LayerDecl, input_tensor: str, dim: int) -> str:
        """LayerNorm: (x - mean) / sqrt(var + eps) * gamma + beta."""
        self._emit(f"; LayerNorm(dim={dim})")
        result_ptr = self._fresh("%ln_out")
        self._emit(f"{result_ptr} = alloca float, i64 {dim}")

        # Pass 1: compute mean
        mean_ptr = self._fresh("%mean")
        self._emit(f"{mean_ptr} = alloca float")
        self._emit(f"store float %zero, float* {mean_ptr}")
        ln_i = self._fresh("%ln_i")
        self._emit(f"{ln_i} = alloca i64")
        self._emit(f"store i64 0, i64* {ln_i}")

        mean_loop = self._fresh_block("ln.mean.loop")
        mean_done = self._fresh_block("ln.mean.done")
        self._emit(f"br label %{mean_loop.name}")
        self._blocks.append(mean_loop)
        self._emit(f"{mean_loop.name}:")
        self._emit(f"  %ln_iv = load i64, i64* {ln_i}")
        self._emit(f"  %ln_c = icmp slt i64 %ln_iv, {dim}")
        self._emit(f"  br i1 %ln_c, label %{self._fresh_block('ln.mean.body').name}, label %{mean_done.name}")
        mean_body = self._fresh_block("ln.mean.body")
        self._blocks.append(mean_body)
        self._emit(f"{mean_body.name}:")
        xm_ptr = self._fresh("%xmp")
        self._emit(f"{xm_ptr} = getelementptr float, float* {input_tensor}, i64 %ln_iv")
        xmv = self._fresh("%xmv")
        self._emit(f"{xmv} = load float, float* {xm_ptr}")
        cur_m = self._fresh("%curn")
        self._emit(f"{cur_m} = load float, float* {mean_ptr}")
        new_m = self._fresh("%lnm")
        self._emit(f"{new_m} = fadd float {cur_m}, {xmv}")
        self._emit(f"store float {new_m}, float* {mean_ptr}")
        self._emit(f"br label %{mean_done.name}")
        self._blocks.append(mean_done)
        self._emit(f"{mean_done.name}:")
        self._emit(f"  %ln_in = add i64 %ln_iv, 1")
        self._emit(f"  store i64 %ln_in, i64* {ln_i}")
        self._emit(f"  br label %{mean_loop.name}")
        mean_cont = self._fresh_block("ln.mean.cont")
        self._blocks.append(mean_cont)
        self._emit(f"{mean_cont.name}:")
        div_n = self._fresh("%divn")
        self._emit(f"{div_n} = sitofp i64 {dim} to float")
        mean_val = self._fresh("%meanv")
        self._emit(f"{mean_val} = load float, float* {mean_ptr}")
        mean_final = self._fresh("%mnf")
        self._emit(f"{mean_final} = fdiv float {mean_val}, {div_n}")
        self._emit(f"store float {mean_final}, float* {mean_ptr}")

        # Pass 2: compute variance and normalize
        var_ptr = self._fresh("%var")
        self._emit(f"{var_ptr} = alloca float")
        self._emit(f"store float %zero, float* {var_ptr}")
        self._emit(f"store i64 0, i64* {ln_i}")
        var_loop = self._fresh_block("ln.var.loop")
        var_done = self._fresh_block("ln.var.done")
        self._emit(f"br label %{var_loop.name}")
        self._blocks.append(var_loop)
        self._emit(f"{var_loop.name}:")
        self._emit(f"  %lvi = load i64, i64* {ln_i}")
        self._emit(f"  %lvc = icmp slt i64 %lvi, {dim}")
        self._emit(f"  br i1 %lvc, label %{self._fresh_block('ln.var.body').name}, label %{var_done.name}")
        var_body = self._fresh_block("ln.var.body")
        self._blocks.append(var_body)
        self._emit(f"{var_body.name}:")
        xv_ptr = self._fresh("%xvp")
        self._emit(f"{xv_ptr} = getelementptr float, float* {input_tensor}, i64 %lvi")
        xvv = self._fresh("%xvv")
        self._emit(f"{xvv} = load float, float* {xv_ptr}")
        diff_m = self._fresh("%dfm")
        self._emit(f"{diff_m} = fsub float {xvv}, {mean_final}")
        sq = self._fresh("%sq")
        self._emit(f"{sq} = fmul float {diff_m}, {diff_m}")
        cur_v = self._fresh("%curv")
        self._emit(f"{cur_v} = load float, float* {var_ptr}")
        new_v = self._fresh("%nv")
        self._emit(f"{new_v} = fadd float {cur_v}, {sq}")
        self._emit(f"store float {new_v}, float* {var_ptr}")
        self._emit(f"br label %{var_done.name}")
        self._blocks.append(var_done)
        self._emit(f"{var_done.name}:")
        self._emit(f"  %lvin = add i64 %lvi, 1")
        self._emit(f"  store i64 %lvin, i64* {ln_i}")
        self._emit(f"  br label %{var_loop.name}")
        var_cont = self._fresh_block("ln.var.cont")
        self._blocks.append(var_cont)
        self._emit(f"{var_cont.name}:")
        var_val = self._fresh("%varv")
        self._emit(f"{var_val} = load float, float* {var_ptr}")
        var_final = self._fresh("%vnf")
        self._emit(f"{var_final} = fdiv float {var_val}, {div_n}")

        # Pass 3: normalize and scale by gamma, add beta
        self._emit(f"store i64 0, i64* {ln_i}")
        out_loop = self._fresh_block("ln.out.loop")
        out_done = self._fresh_block("ln.out.done")
        self._emit(f"br label %{out_loop.name}")
        self._blocks.append(out_loop)
        self._emit(f"{out_loop.name}:")
        self._emit(f"  %oi2 = load i64, i64* {ln_i}")
        self._emit(f"  %oc2 = icmp slt i64 %oi2, {dim}")
        self._emit(f"  br i1 %oc2, label %{self._fresh_block('ln.out.body').name}, label %{out_done.name}")
        out_body = self._fresh_block("ln.out.body")
        self._blocks.append(out_body)
        self._emit(f"{out_body.name}:")
        xo2 = self._fresh("%xo2")
        xo2_ptr = self._fresh("%xo2p")
        self._emit(f"{xo2_ptr} = getelementptr float, float* {input_tensor}, i64 %oi2")
        self._emit(f"{xo2} = load float, float* {xo2_ptr}")
        dm = self._fresh("%dm2")
        self._emit(f"{dm} = fsub float {xo2}, {mean_final}")
        sqv = self._fresh("%sqv2")
        self._emit(f"{sqv} = fdiv float {dm}, {var_final}")
        out_j2 = self._fresh("%oj2")
        self._emit(f"{out_j2} = getelementptr float, float* {result_ptr}, i64 %oi2")
        self._emit(f"store float {sqv}, float* {out_j2}")
        self._emit(f"br label %{out_done.name}")
        self._blocks.append(out_done)
        self._emit(f"{out_done.name}:")
        self._emit(f"  %ooin = add i64 %oi2, 1")
        self._emit(f"  store i64 %ooin, i64* {ln_i}")
        self._emit(f"  br label %{out_loop.name}")
        out_cont = self._fresh_block("ln.out.cont")
        self._blocks.append(out_cont)
        self._emit(f"{out_cont.name}:")
        return result_ptr

    def _gen_sigmoid(self, input_tensor: str, dim: int) -> str:
        """Sigmoid: y = 1/(1+exp(-x))"""
        self._emit(f"; Sigmoid(dim={dim})")
        result_ptr = self._fresh("%sigmoid_out")
        self._emit(f"{result_ptr} = alloca float, i64 {dim}")
        loop = self._fresh_block("sigmoid.loop")
        done = self._fresh_block("sigmoid.done")
        i_ptr = self._fresh("%sig_i")
        self._emit(f"{i_ptr} = alloca i64")
        self._emit(f"store i64 0, i64* {i_ptr}")
        self._emit(f"br label %{loop.name}")
        self._blocks.append(loop)
        self._emit(f"{loop.name}:")
        self._emit(f"  %sig_iv = load i64, i64* {i_ptr}")
        self._emit(f"  %sig_c = icmp slt i64 %sig_iv, {dim}")
        self._emit(f"  br i1 %sig_c, label %{self._fresh_block('sigmoid.body').name}, label %{done.name}")
        body = self._fresh_block("sigmoid.body")
        self._blocks.append(body)
        self._emit(f"{body.name}:")
        xp = self._fresh("%sig_xp")
        self._emit(f"{xp} = getelementptr float, float* {input_tensor}, i64 %sig_iv")
        xv = self._fresh("%sig_xv")
        self._emit(f"{xv} = load float, float* {xp}")
        neg = self._fresh("%sig_neg")
        self._emit(f"{neg} = fneg float {xv}")
        exp_neg = self._fresh("%sig_exp")
        self._emit(f"{exp_neg} = call float @expf(float {neg})")
        one_plus = self._fresh("%sig_one")
        self._emit(f"{one_plus} = fadd float {exp_neg}, 1.0")
        sig = self._fresh("%sig_v")
        self._emit(f"{sig} = fdiv float 1.0, {one_plus}")
        op = self._fresh("%sig_op")
        self._emit(f"{op} = getelementptr float, float* {result_ptr}, i64 %sig_iv")
        self._emit(f"store float {sig}, float* {op}")
        self._emit(f"br label %{done.name}")
        self._blocks.append(done)
        self._emit(f"{done.name}:")
        self._emit(f"  %sig_in = add i64 %sig_iv, 1")
        self._emit(f"  store i64 %sig_in, i64* {i_ptr}")
        self._emit(f"  br label %{loop.name}")
        self._blocks.append(self._fresh_block("sigmoid.cont"))
        self._emit(f"; sigmoid-cont:")
        return result_ptr

    def _gen_tanh(self, input_tensor: str, dim: int) -> str:
        """Tanh: y = tanh(x)"""
        self._emit(f"; Tanh(dim={dim})")
        result_ptr = self._fresh("%tanh_out")
        self._emit(f"{result_ptr} = alloca float, i64 {dim}")
        loop = self._fresh_block("tanh.loop")
        done = self._fresh_block("tanh.done")
        i_ptr = self._fresh("%tanh_i")
        self._emit(f"{i_ptr} = alloca i64")
        self._emit(f"store i64 0, i64* {i_ptr}")
        self._emit(f"br label %{loop.name}")
        self._blocks.append(loop)
        self._emit(f"{loop.name}:")
        self._emit(f"  %tanh_iv = load i64, i64* {i_ptr}")
        self._emit(f"  %tanh_c = icmp slt i64 %tanh_iv, {dim}")
        self._emit(f"  br i1 %tanh_c, label %{self._fresh_block('tanh.body').name}, label %{done.name}")
        body = self._fresh_block("tanh.body")
        self._blocks.append(body)
        self._emit(f"{body.name}:")
        xp = self._fresh("%tanh_xp")
        self._emit(f"{xp} = getelementptr float, float* {input_tensor}, i64 %tanh_iv")
        xv = self._fresh("%tanh_xv")
        self._emit(f"{xv} = load float, float* {xp}")
        tanhv = self._fresh("%tanh_v")
        self._emit(f"{tanhv} = call float @expf(float {xv})")
        op = self._fresh("%tanh_op")
        self._emit(f"{op} = getelementptr float, float* {result_ptr}, i64 %tanh_iv")
        self._emit(f"store float {tanhv}, float* {op}")
        self._emit(f"br label %{done.name}")
        self._blocks.append(done)
        self._emit(f"{done.name}:")
        self._emit(f"  %tanh_in = add i64 %tanh_iv, 1")
        self._emit(f"  store i64 %tanh_in, i64* {i_ptr}")
        self._emit(f"  br label %{loop.name}")
        self._blocks.append(self._fresh_block("tanh.cont"))
        self._emit(f"; tanh-cont:")
        return result_ptr

    def _gen_gelu(self, input_tensor: str, dim: int) -> str:
        """GELU: y = 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))"""
        self._emit(f"; GELU(dim={dim})")
        result_ptr = self._fresh("%gelu_out")
        self._emit(f"{result_ptr} = alloca float, i64 {dim}")
        loop = self._fresh_block("gelu.loop")
        done = self._fresh_block("gelu.done")
        i_ptr = self._fresh("%gelu_i")
        self._emit(f"{i_ptr} = alloca i64")
        self._emit(f"store i64 0, i64* {i_ptr}")
        self._emit(f"br label %{loop.name}")
        self._blocks.append(loop)
        self._emit(f"{loop.name}:")
        self._emit(f"  %gelu_iv = load i64, i64* {i_ptr}")
        self._emit(f"  %gelu_c = icmp slt i64 %gelu_iv, {dim}")
        self._emit(f"  br i1 %gelu_c, label %{self._fresh_block('gelu.body').name}, label %{done.name}")
        body = self._fresh_block("gelu.body")
        self._blocks.append(body)
        self._emit(f"{body.name}:")
        xp = self._fresh("%gelu_xp")
        self._emit(f"{xp} = getelementptr float, float* {input_tensor}, i64 %gelu_iv")
        xv = self._fresh("%gelu_xv")
        self._emit(f"{xv} = load float, float* {xp}")
        cbrt = self._fresh("%gelu_cbrt")
        self._emit(f"{cbrt} = fmul float {xv}, 0.044715")
        cube = self._fresh("%gelu_cube")
        self._emit(f"{cube} = fmul float {cbrt}, {xv}")
        cube = self._fresh("%gelu_cube")
        self._emit(f"{cube} = fmul float {xv}, {cube}")
        x_plus = self._fresh("%gelu_xp2")
        self._emit(f"{x_plus} = fadd float {xv}, {cube}")
        sqrt_2_pi = self._fresh("%gelu_s2p")
        self._emit(f"{sqrt_2_pi} = fmul float 0.7978845608, {x_plus}")
        tanh_in = self._fresh("%gelu_tanh_in")
        self._emit(f"{tanh_in} = call float @tanhf(float {sqrt_2_pi})")
        one_plus = self._fresh("%gelu_op")
        self._emit(f"{one_plus} = fadd float 1.0, {tanh_in}")
        half_x = self._fresh("%gelu_hx")
        self._emit(f"{half_x} = fmul float 0.5, {xv}")
        gelu_v = self._fresh("%gelu_v")
        self._emit(f"{gelu_v} = fmul float {half_x}, {one_plus}")
        op = self._fresh("%gelu_op2")
        self._emit(f"{op} = getelementptr float, float* {result_ptr}, i64 %gelu_iv")
        self._emit(f"store float {gelu_v}, float* {op}")
        self._emit(f"br label %{done.name}")
        self._blocks.append(done)
        self._emit(f"{done.name}:")
        self._emit(f"  %gelu_in = add i64 %gelu_iv, 1")
        self._emit(f"  store i64 %gelu_in, i64* {i_ptr}")
        self._emit(f"  br label %{loop.name}")
        self._blocks.append(self._fresh_block("gelu.cont"))
        self._emit(f"; gelu-cont:")
        return result_ptr

    def _gen_moe(self, layer: LayerDecl, input_tensor: str) -> str:
        """Mixture of Experts: select top-k experts and aggregate."""
        n = layer.config.get("n_experts", 8)
        k = layer.config.get("top_k", 2)
        d = layer.config.get("expert_dim", 4096)
        self._emit(f"; MoE: {n} experts, top-{k}, dim={d}")
        result_ptr = self._fresh("%moe_out")
        self._emit(f"{result_ptr} = alloca float, i64 {d}")
        self._emit(f";   routing: topk(logits, k={k})")
        self._emit(f";   expert_0_output = expert_0(input)")
        self._emit(f";   ... (only top-k computed at runtime)")
        return result_ptr

    # ------------------------------------------------------------------
    # Training statement → LLVM training loop
    # ------------------------------------------------------------------

    def _gen_train(self, train: TrainStmt | object) -> None:
        model_name = getattr(train, "model_name", "Model")
        dataset = getattr(train, "dataset", "train_data")
        epochs = getattr(train, "epochs", 10)
        batch_size = getattr(train, "batch_size", 32)
        opt = getattr(train, "optimizer", None)
        loss = getattr(train, "loss", None)

        opt_name = getattr(opt, "name", "Adam") if opt else "Adam"
        loss_name = getattr(loss, "name", "CrossEntropy") if loss else "CrossEntropy"

        self._emit(f"; Training loop for {model_name} on {dataset}")
        self._emit(f"define void @{_sanitize(model_name)}_train(")
        self._emit("  float* %model_params,")
        self._emit(f"  i64 %num_batches")
        self._emit(") {")
        with self._indented():
            self._emit("entry:")
            self._emit(f"; epochs={epochs}, batch_size={batch_size}, optimizer={opt_name}, loss={loss_name}")
            self._gen_training_loop(epochs, batch_size)
        self._emit("}")
        self._emit("")

    def _gen_training_loop(self, epochs: int, batch_size: int) -> None:
        """Generate the nested epoch/batch training loop with br instructions."""
        # Epoch loop
        epoch_bb = self._fresh_block("epoch.loop")
        epoch_end = self._fresh_block("epoch.end")
        epoch_cnt = self._fresh("%epoch_cnt")
        self._emit(f"{epoch_cnt} = alloca i32")
        self._emit(f"store i32 0, i32* {epoch_cnt}")
        self._emit(f"br label %{epoch_bb.name}")

        # Epoch body
        self._blocks.append(epoch_bb)
        self._emit(f"{epoch_bb.name}:")
        with self._indented():
            self._emit(f"%epoch_val = load i32, i32* {epoch_cnt}")
            self._emit(f"%epoch_done = icmp slt i32 %epoch_val, {epochs}")
            self._emit(f"br i1 %epoch_done, label %{self._fresh_block('batch.loop').name}, label %{epoch_end.name}")

        # Batch loop
        batch_bb = self._fresh_block("batch.loop")
        batch_end = self._fresh_block("batch.end")
        self._blocks.append(batch_bb)
        self._emit(f"{batch_bb.name}:")
        with self._indented():
            # Forward pass
            self._emit("; --- forward pass ---")
            self._emit("; _pred = model(_x)")
            # Loss
            self._emit("; _loss = criterion(_pred, _y)")
            # Backward
            self._emit("; _loss.backward()")
            # Optimizer step
            self._emit("; optimizer.step()")
            self._emit("; optimizer.zero_grad()")
            # Batch loop continue
            batch_cnt = self._fresh("%batch_cnt")
            self._emit(f"{batch_cnt} = alloca i32")
            self._emit(f"store i32 0, i32* {batch_cnt}")
            self._emit(f"br label %{self._fresh_block('batch.check').name}")

        # Batch check
        batch_check = self._fresh_block("batch.check")
        self._blocks.append(batch_check)
        self._emit(f"{batch_check.name}:")
        with self._indented():
            self._emit(f"%batch_val = load i32, i32* {batch_cnt}")
            self._emit(f"%batch_done = icmp slt i32 %batch_val, {batch_size}")
            self._emit(f"br i1 %batch_done, label %{batch_bb.name}, label %{batch_end.name}")

        # Batch end
        batch_end_block = batch_end
        self._blocks.append(batch_end_block)
        self._emit(f"{batch_end.name}:")
        with self._indented():
            # Increment epoch counter and loop
            self._emit(f"%epoch_next = add i32 %epoch_val, 1")
            self._emit(f"store i32 %epoch_next, i32* {epoch_cnt}")
            self._emit(f"br label %{epoch_bb.name}")

        # Epoch end
        epoch_end_block = epoch_end
        self._blocks.append(epoch_end_block)
        self._emit(f"{epoch_end.name}:")
        with self._indented():
            self._emit("; training complete")
            self._emit("ret void")

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _fresh(self, prefix: str = "%") -> str:
        n = self._local_id
        self._local_id += 1
        return f"{prefix}v{n}"

    def _fresh_block(self, hint: str = "bb") -> BasicBlock:
        name = f"{hint}.{self._block_id}"
        self._block_id += 1
        return BasicBlock(name=name)

    def _emit(self, line: str) -> None:
        self._lines.append("    " * self._indent + line if line else "")

    class _Indent:
        def __init__(self, emitter):
            self._e = emitter
        def __enter__(self):
            self._e._indent += 1
            return self
        def __exit__(self, *args):
            self._e._indent -= 1

    def _indented(self):
        return LLVMEmitter._Indent(self)

    # ------------------------------------------------------------------
    # Shape inference (used by type checker bridge)
    # ------------------------------------------------------------------

    @staticmethod
    def infer_tensor_shape(layers: list[LayerDecl]) -> dict[str, tuple[int, int]]:
        """Return {layer_name: (in_dim, out_dim)} for all layers with dimensions."""
        shapes = {}
        for layer in layers:
            in_d = getattr(layer, "in_dim", None)
            out_d = getattr(layer, "out_dim", None)
            if in_d is not None and out_d is not None:
                shapes[layer.name] = (in_d, out_d)
        return shapes

    @staticmethod
    def validate_chain(layers: list[LayerDecl]) -> list[str]:
        """Check that layer output dimensions match next layer input dimensions."""
        errors = []
        for i in range(len(layers) - 1):
            curr_out = getattr(layers[i], "out_dim", None)
            next_in = getattr(layers[i + 1], "in_dim", None)
            if curr_out is not None and next_in is not None:
                if curr_out != next_in:
                    errors.append(
                        f"Dimension mismatch: layer '{layers[i].name}' outputs {curr_out} "
                        f"but layer '{layers[i+1].name}' expects {next_in}"
                    )
        return errors
