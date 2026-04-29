import pytest

from chimera.compiler.codegen_pytorch import PyTorchCodegen
from chimera.lexer import Lexer
from chimera.parser import Parser

torch = pytest.importorskip("torch")


def parse(src: str):
    return Parser(Lexer(src).tokenize()).parse()


def exec_generated(src: str) -> dict:
    code = PyTorchCodegen().generate(parse(src))
    namespace: dict = {}
    exec(compile(code, "<generated>", "exec"), namespace)
    return namespace


def test_generated_mlp_runs_forward_pass():
    ns = exec_generated(
        """
model TinyMLP {
  layer fc1: Dense(4 -> 3)
  layer act1: ReLU
  layer fc2: Dense(3 -> 2)
}
"""
    )

    model = ns["TinyMLP"]()
    y = model(torch.ones(2, 4))

    assert tuple(y.shape) == (2, 2)


def test_generated_moe_runs_forward_pass_and_records_aux_loss():
    ns = exec_generated(
        """
model TinyMoE {
  layer experts: MoE {n_experts: 4, top_k: 2, expert_dim: 8}
}
"""
    )

    model = ns["TinyMoE"]()
    model.train()
    y = model(torch.ones(3, 8))

    assert tuple(y.shape) == (3, 8)
    assert len(model._aux_losses) == 1


def test_generated_retrieval_model_initializes_vector_store():
    ns = exec_generated(
        """
model RAGChimera {
  retrieval {
    store_dim: 4
    top_k: 2
    min_similarity: 0.5
  }
  layer fc1: Dense(4 -> 2)
}
"""
    )

    model = ns["RAGChimera"]()

    assert model._retrieval_store is not None
    assert model._retrieval_store.dim == 4
    assert model._retrieval_top_k == 2
