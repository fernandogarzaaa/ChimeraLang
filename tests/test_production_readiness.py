import pytest

from chimera.compiler.codegen_pytorch import PyTorchCodegen
from chimera.lexer import Lexer
from chimera.parser import Parser
from chimera.type_checker import TypeChecker


def parse(src: str):
    return Parser(Lexer(src).tokenize()).parse()


def check(src: str):
    return TypeChecker().check(parse(src))


def test_type_checker_rejects_invalid_roadmap_configs():
    result = check(
        """
model Base {
  layer fc1: Dense(4 -> 2)
  retrieval {
    top_k: 0
    min_similarity: 1.5
  }
}

federated_train BadFed {
  model: Base
  rounds: 0
  privacy_epsilon: 0.0
}

meta_train BadMeta {
  base_model: Base
  inner_steps: 0
}

self_improve BadEvolver {
  model: Base
  max_generations: 0
}

swarm BadSwarm {
  agents: 0
}

replay_buffer BadReplay {
  capacity: 0
}

predictive_coding BadPredictor {
  depth: 0
}
"""
    )

    assert not result.ok
    joined = "\n".join(result.errors)
    assert "retrieval top_k" in joined
    assert "privacy_epsilon" in joined
    assert "inner_steps" in joined
    assert "max_generations" in joined
    assert "agents" in joined
    assert "capacity" in joined
    assert "depth" in joined


def test_vector_store_validates_dimensions_and_capacity():
    from chimera_runtime import VectorStore

    store = VectorStore(dim=2, capacity=1)
    store.write([1.0, 0.0], "alpha")

    with pytest.raises(ValueError, match="dimension"):
        store.write([1.0, 0.0, 0.0], "bad")

    with pytest.raises(OverflowError, match="capacity"):
        store.write([0.0, 1.0], "overflow")


def test_privacy_engine_rejects_invalid_budget_and_exhaustion():
    from chimera_runtime import DifferentialPrivacyEngine

    with pytest.raises(ValueError, match="epsilon"):
        DifferentialPrivacyEngine(epsilon=0.0)

    engine = DifferentialPrivacyEngine(epsilon=0.6, total_budget=1.0, seed=1)
    engine.privatize([1.0, 0.0])

    with pytest.raises(RuntimeError, match="privacy budget"):
        engine.privatize([1.0, 0.0])


def test_self_improver_rejects_forbidden_mutations():
    from chimera_runtime import SelfImprover

    improver = SelfImprover(model="AlignedChimera", forbidden_ops=["remove_constitution_layer"])

    with pytest.raises(ValueError, match="forbidden"):
        improver.propose({"op": "remove_constitution_layer"})


def test_moe_codegen_uses_stable_load_balance_accounting():
    program = parse(
        """
model MoEModel {
  layer experts: MoE {n_experts: 8, top_k: 2, expert_dim: 256}
}
"""
    )

    code = PyTorchCodegen().generate(program)

    assert "torch.bincount(_topk.indices.reshape(-1), minlength=8)" in code
    assert "self._aux_losses.append(_load_loss)" in code
