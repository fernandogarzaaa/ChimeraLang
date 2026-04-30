"""Tests for the LLVM IR backend (Phase 2).

Verifies that the emitter produces structurally valid LLVM IR:
- Header declarations, target triple, data layout
- Dense layer emits matrix-vector multiply loops
- ReLU emits element-wise select
- Softmax emits exp/sum/divide loops with max subtraction
- LayerNorm emits mean+variance+normalize passes
- Training loops emit properly structured epoch/batch nested loops
- Chain dimension validation catches mismatches
- Emitted IR is free of accidental brace literals (`{{` / `}}`) and, when
  the `llvm-as` binary is available, parses cleanly through it.
"""

import shutil
import subprocess
import textwrap

import pytest

from chimera.ast_nodes import (
    LayerDecl,
    LossConfig,
    ModelDecl,
    OptimizerConfig,
    TrainStmt,
)
from chimera.compiler.llvm_backend import LLVMEmitter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def emit_model(layers, name="TestModel") -> str:
    model = ModelDecl(name=name, layers=layers)
    emitter = LLVMEmitter()
    return emitter.emit(model)


def emit_training(
    model_name="TestModel",
    epochs=3,
    batch_size=16,
    opt_name="Adam",
    loss_name="CrossEntropy",
) -> str:
    train = TrainStmt(
        model_name=model_name,
        dataset="mnist",
        epochs=epochs,
        batch_size=batch_size,
        optimizer=OptimizerConfig(name=opt_name),
        loss=LossConfig(name=loss_name),
    )
    emitter = LLVMEmitter()
    emitter.emit(train)
    return "\n".join(emitter._lines)


# ---------------------------------------------------------------------------
# Header / module structure
# ---------------------------------------------------------------------------


class TestLLVMHeader:
    def test_emits_module_id_comment(self):
        out = emit_model([])
        assert 'source_filename = "chimeralang.module"' in out

    def test_emits_target_triple(self):
        out = emit_model([])
        assert 'target triple = "x86_64-unknown-linux-gnu"' in out

    def test_emits_data_layout(self):
        out = emit_model([])
        assert "target datalayout" in out

    def test_emits_expf_declaration(self):
        out = emit_model([])
        assert 'declare float @expf(float)' in out

    def test_emits_sqrtf_declaration(self):
        out = emit_model([])
        assert 'declare float @sqrtf(float)' in out

    def test_emits_tanhf_declaration(self):
        out = emit_model([])
        assert 'declare float @tanhf(float)' in out

    def test_empty_model_emits_forward_function(self):
        out = emit_model([])
        assert "@TestModel_forward" in out

    def test_empty_model_ret_null_or_val(self):
        out = emit_model([])
        # Empty model returns null pointer (loaded into %0 first)
        assert "ret float*" in out and "null" in out


# ---------------------------------------------------------------------------
# Dense layer
# ---------------------------------------------------------------------------


class TestDenseLayer:
    def test_dense_emits_alloc_output(self):
        layers = [LayerDecl(name="dense1", kind="Dense", in_dim=784, out_dim=256)]
        out = emit_model(layers)
        assert "alloca float, i64 256" in out

    def test_dense_emits_comment(self):
        layers = [LayerDecl(name="dense1", kind="Dense", in_dim=784, out_dim=256)]
        out = emit_model(layers)
        assert "; Dense 784 -> 256" in out

    def test_dense_emits_loop_labels(self):
        layers = [LayerDecl(name="l1", kind="Dense", in_dim=128, out_dim=64)]
        out = emit_model(layers)
        assert "dense.j" in out
        assert "dense.j.end" in out

    def test_dense_2d_chain(self):
        layers = [
            LayerDecl(name="dense1", kind="Dense", in_dim=784, out_dim=256),
            LayerDecl(name="dense2", kind="Dense", in_dim=256, out_dim=128),
        ]
        out = emit_model(layers)
        assert "; Dense 784 -> 256" in out
        assert "; Dense 256 -> 128" in out


# ---------------------------------------------------------------------------
# Activation layers
# ---------------------------------------------------------------------------


class TestActivationLayers:
    def test_relu_emits_alloca(self):
        layers = [LayerDecl(name="relu1", kind="ReLU", in_dim=256, out_dim=256)]
        out = emit_model(layers)
        assert "relu_out" in out
        assert "fcmp oge" in out  # x >= 0 check

    def test_relu_emits_select(self):
        layers = [LayerDecl(name="relu1", kind="ReLU", in_dim=256, out_dim=256)]
        out = emit_model(layers)
        assert "select i1" in out

    def test_relu_emits_loop(self):
        layers = [LayerDecl(name="relu1", kind="ReLU", in_dim=256, out_dim=256)]
        out = emit_model(layers)
        assert "relu.loop" in out

    def test_softmax_emits_expf_call(self):
        layers = [LayerDecl(name="soft1", kind="Softmax", in_dim=10, out_dim=10)]
        out = emit_model(layers)
        assert "@expf(float" in out

    def test_softmax_emits_three_passes(self):
        layers = [LayerDecl(name="soft1", kind="Softmax", in_dim=10, out_dim=10)]
        out = emit_model(layers)
        # Three named loop sections: max, sum, out
        assert "softmax.max" in out
        assert "softmax.sum" in out
        assert "softmax.out" in out

    def test_sigmoid_emits_expf_call(self):
        layers = [LayerDecl(name="sig1", kind="Sigmoid", in_dim=128, out_dim=128)]
        out = emit_model(layers)
        assert "@expf(float" in out

    def test_tanh_emits_tanhf_call(self):
        layers = [LayerDecl(name="tanh1", kind="Tanh", in_dim=128, out_dim=128)]
        out = emit_model(layers)
        assert "@tanhf(float" in out

    def test_gelu_emits_sqrtf_call(self):
        layers = [LayerDecl(name="gelu1", kind="GELU", in_dim=128, out_dim=128)]
        out = emit_model(layers)
        assert "@sqrtf(float" in out


# ---------------------------------------------------------------------------
# Dropout / LayerNorm
# ---------------------------------------------------------------------------


class TestNormalizationLayers:
    def test_dropout_emits_alloc(self):
        layers = [LayerDecl(name="drop1", kind="Dropout", in_dim=256, out_dim=256)]
        out = emit_model(layers)
        assert "dropout_out" in out

    def test_dropout_uses_probability_from_config(self):
        layers = [
            LayerDecl(
                name="drop1",
                kind="Dropout",
                in_dim=256,
                out_dim=256,
                config={"p": 0.2},
            )
        ]
        out = emit_model(layers)
        assert "p=0.2" in out

    def test_layernorm_emits_mean_pass(self):
        layers = [LayerDecl(name="ln1", kind="LayerNorm", in_dim=128, out_dim=128)]
        out = emit_model(layers)
        assert "ln.mean" in out

    def test_layernorm_emits_var_pass(self):
        layers = [LayerDecl(name="ln1", kind="LayerNorm", in_dim=128, out_dim=128)]
        out = emit_model(layers)
        assert "ln.var" in out

    def test_layernorm_emits_normalize_pass(self):
        layers = [LayerDecl(name="ln1", kind="LayerNorm", in_dim=128, out_dim=128)]
        out = emit_model(layers)
        assert "ln.out" in out


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


class TestTrainingLoop:
    def test_train_emits_function(self):
        out = emit_training(model_name="MyModel", epochs=1, batch_size=1)
        assert "@MyModel_train" in out

    def test_train_declares_num_batches_param(self):
        out = emit_training(model_name="MyModel")
        assert "%num_batches" in out

    def test_train_emits_epoch_loop(self):
        out = emit_training(model_name="MyModel", epochs=5)
        assert "epoch.loop" in out
        assert "epoch.end" in out

    def test_train_includes_optimizer_comment(self):
        out = emit_training(model_name="MyModel", opt_name="SGD")
        assert "optimizer=SGD" in out

    def test_train_includes_loss_comment(self):
        out = emit_training(model_name="MyModel", loss_name="MSE")
        assert "loss=MSE" in out

    def test_train_includes_batch_size_in_comment(self):
        out = emit_training(model_name="M", batch_size=64)
        assert "batch_size=64" in out

    def test_train_ret_void(self):
        out = emit_training(model_name="MyModel")
        assert "ret void" in out


# ---------------------------------------------------------------------------
# Dimension validation
# ---------------------------------------------------------------------------


class TestDimensionValidation:
    def test_validate_chain_catches_mismatch(self):
        layers = [
            LayerDecl(name="l1", kind="Dense", in_dim=784, out_dim=256),
            LayerDecl(name="l2", kind="Dense", in_dim=128, out_dim=10),  # 256 != 128
        ]
        errors = LLVMEmitter.validate_chain(layers)
        assert len(errors) == 1
        assert "l1" in errors[0]
        assert "l2" in errors[0]
        assert "256" in errors[0]
        assert "128" in errors[0]

    def test_validate_chain_passes_correct_chain(self):
        layers = [
            LayerDecl(name="l1", kind="Dense", in_dim=784, out_dim=256),
            LayerDecl(name="l2", kind="Dense", in_dim=256, out_dim=128),
            LayerDecl(name="l3", kind="Dense", in_dim=128, out_dim=10),
        ]
        errors = LLVMEmitter.validate_chain(layers)
        assert errors == []

    def test_validate_chain_single_layer(self):
        layers = [LayerDecl(name="l1", kind="Dense", in_dim=784, out_dim=10)]
        errors = LLVMEmitter.validate_chain(layers)
        assert errors == []

    def test_validate_chain_partial_dims(self):
        # Layers with missing dims should not error
        layers = [
            LayerDecl(name="l1", kind="ReLU", in_dim=None, out_dim=None),
            LayerDecl(name="l2", kind="ReLU", in_dim=None, out_dim=None),
        ]
        errors = LLVMEmitter.validate_chain(layers)
        assert errors == []

    def test_validate_chain_missing_intermediate_dim(self):
        layers = [
            LayerDecl(name="l1", kind="Dense", in_dim=784, out_dim=256),
            LayerDecl(name="l2", kind="ReLU", in_dim=None, out_dim=None),
            LayerDecl(name="l3", kind="Dense", in_dim=256, out_dim=10),
        ]
        errors = LLVMEmitter.validate_chain(layers)
        assert errors == []


# ---------------------------------------------------------------------------
# Shape inference
# ---------------------------------------------------------------------------


class TestShapeInference:
    def test_infer_shapes_dense(self):
        layers = [
            LayerDecl(name="dense1", kind="Dense", in_dim=784, out_dim=256),
        ]
        shapes = LLVMEmitter.infer_tensor_shape(layers)
        assert shapes["dense1"] == (784, 256)

    def test_infer_shapes_chain(self):
        layers = [
            LayerDecl(name="dense1", kind="Dense", in_dim=784, out_dim=256),
            LayerDecl(name="dense2", kind="Dense", in_dim=256, out_dim=128),
        ]
        shapes = LLVMEmitter.infer_tensor_shape(layers)
        assert shapes["dense1"] == (784, 256)
        assert shapes["dense2"] == (256, 128)

    def test_infer_shapes_skips_missing_dims(self):
        layers = [
            LayerDecl(name="relu1", kind="ReLU", in_dim=None, out_dim=None),
            LayerDecl(name="dense1", kind="Dense", in_dim=784, out_dim=256),
        ]
        shapes = LLVMEmitter.infer_tensor_shape(layers)
        assert "dense1" in shapes
        assert "relu1" not in shapes


# ---------------------------------------------------------------------------
# Full MLP model emission
# ---------------------------------------------------------------------------


class TestFullMLPEmission:
    def test_mlp_784_256_128_10(self):
        layers = [
            LayerDecl(name="dense1", kind="Dense", in_dim=784, out_dim=256),
            LayerDecl(name="relu1", kind="ReLU", in_dim=256, out_dim=256),
            LayerDecl(name="dense2", kind="Dense", in_dim=256, out_dim=128),
            LayerDecl(name="relu2", kind="ReLU", in_dim=128, out_dim=128),
            LayerDecl(name="dense3", kind="Dense", in_dim=128, out_dim=10),
            LayerDecl(name="soft1", kind="Softmax", in_dim=10, out_dim=10),
        ]
        out = emit_model(layers, name="MLP")
        assert "@MLP_forward" in out
        assert "; Dense 784 -> 256" in out
        assert "; Dense 256 -> 128" in out
        assert "; Dense 128 -> 10" in out
        assert "relu.loop" in out
        assert "softmax.max" in out
        # Chain validation should pass
        errors = LLVMEmitter.validate_chain(layers)
        assert errors == []

    def test_batch_training_loop(self):
        layers = [
            LayerDecl(name="dense1", kind="Dense", in_dim=784, out_dim=256),
        ]
        out = emit_model(layers, name="Simple")
        train_out = emit_training(model_name="Simple", epochs=10, batch_size=64)
        assert "epoch.loop" in train_out
        assert "batch.loop" in train_out


# ---------------------------------------------------------------------------
# Unknown / passthrough layers
# ---------------------------------------------------------------------------


class TestUnknownLayers:
    def test_unknown_layer_passthrough_emits_comment(self):
        layers = [
            LayerDecl(name="unknown1", kind="CustomOp", in_dim=128, out_dim=128),
        ]
        out = emit_model(layers)
        assert "unknown layer CustomOp" in out

    def test_moe_emits_comment(self):
        layers = [
            LayerDecl(
                name="moe1",
                kind="MoE",
                in_dim=1024,
                out_dim=1024,
                config={"n_experts": 64, "top_k": 2},
            )
        ]
        out = emit_model(layers)
        assert "MoE: 64 experts" in out


# ---------------------------------------------------------------------------
# IR validity — catches f-string brace mistakes and (when llvm-as is on
# PATH) any deeper structural breakage in emitted IR.
# ---------------------------------------------------------------------------


def _all_emitter_outputs() -> list[tuple[str, str]]:
    """Produce a representative set of (label, ir_text) snapshots."""
    snapshots: list[tuple[str, str]] = []

    snapshots.append((
        "dense_relu_softmax",
        emit_model([
            LayerDecl(name="d1", kind="Dense", in_dim=784, out_dim=128),
            LayerDecl(name="a1", kind="ReLU", in_dim=128, out_dim=128),
            LayerDecl(name="d2", kind="Dense", in_dim=128, out_dim=10),
            LayerDecl(name="s1", kind="Softmax", in_dim=10, out_dim=10),
        ]),
    ))

    snapshots.append((
        "layernorm_only",
        emit_model([
            LayerDecl(name="ln1", kind="LayerNorm", in_dim=64, out_dim=64),
        ]),
    ))

    snapshots.append((
        "training_loop",
        emit_training(epochs=2, batch_size=8),
    ))

    return snapshots


class TestIRValidity:
    """Regression tests for emitted IR shape.

    The cheap brace check would have caught the `") {{"` typo in the forward
    and train function bodies (bug #3 from the audit). The llvm-as check,
    when available, validates the full module — anyone running the suite in
    CI with LLVM installed will catch deeper regressions for free.
    """

    @pytest.mark.parametrize("label,ir", _all_emitter_outputs(), ids=lambda x: x if isinstance(x, str) else "")
    def test_no_unescaped_double_braces(self, label, ir):
        # Skip the parametrized id pair (pytest passes both elements through
        # the ids lambda; we want the (label, ir) form here).
        assert "{{" not in ir, (
            f"emitted IR for {label!r} contains '{{{{' — likely an f-string "
            f"escaping bug like the one fixed in bug #3"
        )
        assert "}}" not in ir, (
            f"emitted IR for {label!r} contains '}}}}' — likely an f-string "
            f"escaping bug"
        )

    @pytest.mark.parametrize("label,ir", _all_emitter_outputs(), ids=lambda x: x if isinstance(x, str) else "")
    def test_llvm_as_accepts_ir(self, label, ir, tmp_path):
        llvm_as = shutil.which("llvm-as")
        if llvm_as is None:
            pytest.skip("llvm-as not on PATH; install LLVM to enable IR validation")

        # The standalone train-loop snapshot is a function body fragment
        # without the enclosing module header, so llvm-as can't parse it
        # by itself — the brace check above is sufficient for that case.
        if not ir.lstrip().startswith(("target ", "; ModuleID", "source_filename")):
            pytest.skip(f"{label} is a function-body fragment, not a full module")

        ll_path = tmp_path / f"{label}.ll"
        ll_path.write_text(ir)
        result = subprocess.run(
            [llvm_as, str(ll_path), "-o", str(tmp_path / f"{label}.bc")],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"llvm-as rejected {label} IR:\n"
            f"--- stderr ---\n{result.stderr}\n"
            f"--- ir (first 80 lines) ---\n"
            + "\n".join(ir.splitlines()[:80])
        )
