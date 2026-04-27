"""Tests for PyTorch code generation."""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Create minimal mock AST nodes for testing (avoids dependency on chimera/ast_nodes.py changes)
from dataclasses import dataclass, field

@dataclass
class _OptimizerConfig:
    name: str = "Adam"
    params: dict = field(default_factory=dict)

@dataclass
class _LossConfig:
    name: str = "CrossEntropy"

@dataclass
class _LayerDecl:
    name: str = ""
    kind: str = ""
    in_dim: object = None
    out_dim: object = None
    repeat: int = 1
    config: dict = field(default_factory=dict)

@dataclass
class _ModelDecl:
    name: str = ""
    layers: list = field(default_factory=list)
    forward_fn: object = None
    uncertainty: str = "none"
    constitution: object = None
    device: str = "cpu"
    config: dict = field(default_factory=dict)

@dataclass
class _TrainStmt:
    model_name: str = ""
    dataset: str = "train_data"
    optimizer: object = None
    loss: object = None
    epochs: int = 10
    batch_size: int = 32
    guards: list = field(default_factory=list)
    evolve: object = None
    precision: str = "float32"
    config: dict = field(default_factory=dict)

@dataclass
class _ConstitutionDecl:
    name: str = ""
    principles: list = field(default_factory=list)
    critique_rounds: int = 2
    max_violation_score: float = 0.05

@dataclass
class _Program:
    declarations: list = field(default_factory=list)

from chimera.compiler.codegen_pytorch import PyTorchCodegen


def test_simple_mlp_generates_nn_module():
    prog = _Program(declarations=[_ModelDecl(
        name="Classifier",
        layers=[
            _LayerDecl(name="fc1", kind="Dense", in_dim=784, out_dim=128),
            _LayerDecl(name="act1", kind="ReLU"),
            _LayerDecl(name="fc2", kind="Dense", in_dim=128, out_dim=10),
        ]
    )])
    code = PyTorchCodegen().generate(prog)
    assert "class Classifier(nn.Module)" in code
    assert "nn.Linear(784, 128)" in code
    assert "nn.ReLU()" in code
    assert "nn.Linear(128, 10)" in code


def test_train_generates_training_loop():
    prog = _Program(declarations=[
        _ModelDecl(name="Foo", layers=[_LayerDecl(name="fc1", kind="Dense", in_dim=10, out_dim=5)]),
        _TrainStmt(model_name="Foo", dataset="my_dataset",
                   optimizer=_OptimizerConfig("Adam", {"lr": 0.001}),
                   loss=_LossConfig("CrossEntropy"), epochs=5, batch_size=64),
    ])
    code = PyTorchCodegen().generate(prog)
    assert "torch.optim.Adam" in code
    assert "lr=0.001" in code
    assert "nn.CrossEntropyLoss" in code
    assert "for epoch in range(5)" in code
    assert "batch_size=64" in code
    assert "loss.backward()" in code


def test_moe_generates_expert_layer():
    prog = _Program(declarations=[_ModelDecl(
        name="MoEFoo",
        layers=[_LayerDecl(name="experts", kind="MoE", config={"n_experts": 8, "top_k": 2, "expert_dim": 256})]
    )])
    code = PyTorchCodegen().generate(prog)
    assert "n_experts=8" in code or "range(8)" in code
    assert "top_k" in code or "_top_k = 2" in code


def test_constitution_generates_critique_layer():
    prog = _Program(declarations=[
        _ConstitutionDecl(name="MySafety", principles=["Be helpful", "Be harmless"], critique_rounds=2),
        _ModelDecl(name="SafeModel", layers=[_LayerDecl(name="fc1", kind="Dense", in_dim=10, out_dim=5)],
                   constitution="MySafety"),
    ])
    code = PyTorchCodegen().generate(prog)
    assert "ConstitutionLayer" in code


def test_generated_code_is_valid_python():
    prog = _Program(declarations=[
        _ModelDecl(name="SimpleNet", layers=[
            _LayerDecl(name="fc1", kind="Dense", in_dim=784, out_dim=256),
            _LayerDecl(name="act1", kind="ReLU"),
            _LayerDecl(name="fc2", kind="Dense", in_dim=256, out_dim=10),
        ]),
        _TrainStmt(model_name="SimpleNet", dataset="train_data",
                   optimizer=_OptimizerConfig("Adam", {"lr": 0.001}),
                   loss=_LossConfig("CrossEntropy"), epochs=3, batch_size=32),
    ])
    code = PyTorchCodegen().generate(prog)
    try:
        compile(code, "<generated>", "exec")
    except SyntaxError as e:
        pytest.fail(f"Generated code has syntax error: {e}\n\nCode:\n{code}")


def test_header_always_present():
    code = PyTorchCodegen().generate(_Program(declarations=[]))
    assert "import torch" in code
    assert "import torch.nn as nn" in code
    assert "chimera_runtime" in code
