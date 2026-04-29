from chimera.ast_nodes import (
    CausalModelDecl,
    FederatedTrainStmt,
    MetaTrainStmt,
    PredictiveCodingDecl,
    ReplayBufferDecl,
    RewardSystemDecl,
    SelfImproveDecl,
    SwarmDecl,
)
from chimera.compiler.codegen_pytorch import PyTorchCodegen
from chimera.lexer import Lexer
from chimera.parser import Parser
from chimera.type_checker import TypeChecker


def parse(src: str):
    return Parser(Lexer(src).tokenize()).parse()


def test_parser_accepts_remaining_top_level_roadmap_blocks():
    program = parse(
        """
causal_model ChimeraCausal {
  variables: scm
  edges: X_to_Y
  adjustment: backdoor
}

federated_train ChimeraFederated {
  model: AlignedChimera
  rounds: 1000
  privacy_epsilon: 1.0
}

meta_train ChimeraFewShot {
  base_model: ChimeraMoE
  inner_steps: 5
}

self_improve ChimeraEvolver {
  model: AlignedChimera
  verifier: z3_solver
}

swarm ChimeraSwarm {
  agents: 16
  topology: ring
}

replay_buffer ChimeraHippocampus {
  capacity: 1000000
  strategy: prioritized
}

reward_system ChimeraDopamine {
  signal: prediction_error
}

predictive_coding ChimeraPredictor {
  depth: 12
  error_signal: residual
}
"""
    )

    assert isinstance(program.declarations[0], CausalModelDecl)
    assert isinstance(program.declarations[1], FederatedTrainStmt)
    assert isinstance(program.declarations[2], MetaTrainStmt)
    assert isinstance(program.declarations[3], SelfImproveDecl)
    assert isinstance(program.declarations[4], SwarmDecl)
    assert isinstance(program.declarations[5], ReplayBufferDecl)
    assert isinstance(program.declarations[6], RewardSystemDecl)
    assert isinstance(program.declarations[7], PredictiveCodingDecl)


def test_type_checker_registers_roadmap_declarations():
    checker = TypeChecker()
    result = checker.check(
        parse(
            """
model ChimeraMoE {
  layer fc1: Dense(4 -> 2)
}

causal_model ChimeraCausal {
  variables: scm
}

federated_train ChimeraFederated {
  model: ChimeraMoE
  rounds: 10
  privacy_epsilon: 1.0
}

meta_train ChimeraFewShot {
  base_model: ChimeraMoE
  inner_steps: 5
}

self_improve ChimeraEvolver {
  model: ChimeraMoE
  max_generations: 10
}
"""
        )
    )

    assert result.ok
    assert checker._env.lookup("ChimeraCausal") is not None
    assert checker._env.lookup("ChimeraFederated") is not None
    assert checker._env.lookup("ChimeraFewShot") is not None
    assert checker._env.lookup("ChimeraEvolver") is not None


def test_codegen_emits_rag_retrieval_runtime_hook():
    program = parse(
        """
model RAGChimera {
  retrieval {
    store_dim: 1536
    top_k: 10
    min_similarity: 0.75
  }
  layer fc1: Dense(4 -> 2)
}
"""
    )

    code = PyTorchCodegen().generate(program)

    assert "VectorStore" in code
    assert "self._retrieval_store" in code
    assert "self._retrieval_top_k = 10" in code
