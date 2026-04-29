from chimera.lexer import Lexer
from chimera.parser import Parser
from chimera.type_checker import TypeChecker


def parse(src: str):
    return Parser(Lexer(src).tokenize()).parse()


def check(src: str):
    return TypeChecker().check(parse(src))


def test_type_checker_accepts_model_with_constitution():
    result = check(
        """
constitution Safety {
  "Be helpful"
}

model SafeClassifier {
  layer fc1: Dense(784 -> 128)
  layer fc2: Dense(128 -> 10)
  constitution: Safety
}
"""
    )

    assert result.ok
    assert result.errors == []


def test_type_checker_reports_layer_dimension_mismatch():
    result = check(
        """
model BrokenClassifier {
  layer fc1: Dense(784 -> 128)
  layer fc2: Dense(64 -> 10)
}
"""
    )

    assert not result.ok
    assert any("outputs 128" in err and "expects in_dim=64" in err for err in result.errors)


def test_type_checker_resolves_vectorstore_and_constituted_types():
    result = check(
        """
val knowledge: VectorStore<768>[1000] = store
val answer: Constituted<Text> = response
"""
    )

    assert result.ok
