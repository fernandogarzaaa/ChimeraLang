"""Tests for the ChimeraVM (chimera/vm.py)."""

import pytest

from chimera.lexer import Lexer
from chimera.parser import Parser
from chimera.types import ConfidentValue, ConvergeValue
from chimera.vm import AssertionFailed, ChimeraVM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(source: str, seed: int = 42):
    tokens = Lexer(source).tokenize()
    program = Parser(tokens).parse()
    vm = ChimeraVM(seed=seed)
    return vm.execute(program)


# ---------------------------------------------------------------------------
# Fix 2: reason block registers under its actual name
# ---------------------------------------------------------------------------

class TestReasonBlock:
    def test_reason_registers_as_about(self):
        src = """
reason about(x: Int) -> Int
  return x
end

val result: Int = about(7)
emit result
"""
        result = run(src)
        assert result.errors == []
        assert result.emitted[0].raw == 7

    def test_reason_block_callable_in_goal(self):
        src = """
reason about(rate: Float) -> Float
  val decay: Float = 0.95
  return rate * decay
end

goal "test"
  val r: Float = about(0.1)
  emit r
end
"""
        result = run(src)
        assert result.errors == []
        assert abs(result.emitted[0].raw - 0.095) < 1e-9


# ---------------------------------------------------------------------------
# Fix 1: ConfidentValue constructor works
# ---------------------------------------------------------------------------

class TestConfidentValueVM:
    def test_confident_constructor_works(self):
        src = """
val x: Confident<Text> = Confident("hello")
emit x
"""
        result = run(src)
        assert result.errors == []
        assert result.emitted[0].raw == "hello"
        assert isinstance(result.emitted[0], ConfidentValue)

    def test_confident_constructor_in_function(self):
        src = """
fn greet(name: Text) -> Confident<Text>
  return Confident(name)
end

val msg: Confident<Text> = greet("world")
emit msg
"""
        result = run(src)
        assert result.errors == []
        assert result.emitted[0].raw == "world"


# ---------------------------------------------------------------------------
# Fix 3: Semantic constraints are wired in
# ---------------------------------------------------------------------------

class TestConstraints:
    def test_allow_constraint_logged_to_trace(self):
        src = """
fn explore_fn(x: Text) -> Text
  allow:
    "speculative reasoning"
  return x
end

val r: Text = explore_fn("test")
emit r
"""
        result = run(src)
        assert result.errors == []
        constraint_traces = [t for t in result.trace if "allow" in t]
        assert len(constraint_traces) >= 1
        assert "speculative reasoning" in constraint_traces[0]

    def test_forbidden_constraint_logged_to_trace(self):
        src = """
fn safe_fn(x: Text) -> Text
  forbidden:
    "fabricating citations"
    "inventing statistics"
  return x
end

val r: Text = safe_fn("test")
emit r
"""
        result = run(src)
        assert result.errors == []
        forbidden_traces = [t for t in result.trace if "forbidden" in t]
        assert len(forbidden_traces) >= 1
        assert "fabricating citations" in forbidden_traces[0]

    def test_must_constraint_passes_when_true(self):
        src = """
fn check_fn(x: Int) -> Int
  must:
    x > 0
  return x
end

val r: Int = check_fn(5)
emit r
"""
        result = run(src)
        assert result.errors == []
        assert result.emitted[0].raw == 5

    def test_must_constraint_fails_when_false(self):
        src = """
fn check_fn(x: Int) -> Int
  must:
    x > 100
  return x
end

val r: Int = check_fn(5)
emit r
"""
        result = run(src)
        # A must-constraint that evaluates to false raises AssertionFailed inside
        # the VM, which is caught and recorded as assertions_failed (not errors).
        assert result.assertions_failed > 0


# ---------------------------------------------------------------------------
# Fix 4: Gate branches expose branch_index / branch_seed
# ---------------------------------------------------------------------------

class TestGateBranchDivergence:
    def test_gate_branches_have_different_confidences(self):
        src = """
gate multi_branch(x: Text) -> Converge<Text>
  branches: 5
  collapse: majority
  threshold: 0.7
  val answer: Text = "result: " + x
  return answer
end

val out: Converge<Text> = multi_branch("input")
emit out
"""
        result = run(src, seed=42)
        assert result.errors == []
        assert len(result.gate_logs) == 1
        # Branches should have different confidence values due to Gaussian noise
        confs = result.gate_logs[0]["branch_confidences"]
        assert len(confs) == 5
        # Not all identical (Gaussian noise ensures divergence)
        assert len(set(round(c, 6) for c in confs)) > 1

    def test_gate_branch_index_accessible(self):
        """branch_index variable is injected into each branch scope."""
        src = """
gate indexed_gate(x: Int) -> Converge<Int>
  branches: 3
  collapse: majority
  threshold: 0.7
  val idx: Int = branch_index
  return idx
end

val out: Converge<Int> = indexed_gate(0)
emit out
"""
        result = run(src, seed=0)
        assert result.errors == []
        assert len(result.gate_logs) == 1
        assert result.gate_logs[0]["branches"] == 3

    def test_gate_result_is_converge_value(self):
        src = """
gate simple_gate(q: Text) -> Converge<Text>
  branches: 3
  collapse: majority
  threshold: 0.8
  val answer: Text = "answer"
  return answer
end

val r: Converge<Text> = simple_gate("question")
emit r
"""
        result = run(src)
        assert result.errors == []
        assert isinstance(result.emitted[0], ConvergeValue)


# ---------------------------------------------------------------------------
# Full example files
# ---------------------------------------------------------------------------

class TestExamples:
    def _run_example(self, filename: str):
        import pathlib
        src = (
            pathlib.Path(__file__).parent.parent / "examples" / filename
        ).read_text()
        return run(src)

    def test_hello_chimera(self):
        result = self._run_example("hello_chimera.chimera")
        assert result.errors == []
        assert result.emitted[0].raw == "Hello from ChimeraLang!"

    def test_goal_driven(self):
        result = self._run_example("goal_driven.chimera")
        assert result.errors == []
        assert len(result.emitted) >= 1

    def test_hallucination_guard(self):
        result = self._run_example("hallucination_guard.chimera")
        assert result.errors == []
        assert result.assertions_passed >= 2

    def test_quantum_reasoning(self):
        result = self._run_example("quantum_reasoning.chimera")
        assert result.errors == []
        assert result.assertions_passed >= 2
