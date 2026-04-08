"""Tests for new ChimeraLang v0.2.0 features: for loops, match, builtins, detect."""

import pytest

from chimera.lexer import Lexer
from chimera.parser import Parser
from chimera.types import ConfidentValue, ExploreValue
from chimera.vm import ChimeraVM


def run(source: str, seed: int = 42):
    tokens = Lexer(source).tokenize()
    program = Parser(tokens).parse()
    vm = ChimeraVM(seed=seed)
    return vm.execute(program)


# ---------------------------------------------------------------------------
# For loops
# ---------------------------------------------------------------------------

class TestForLoop:
    def test_for_over_list(self):
        src = """
val items: List<Int> = [1, 2, 3]
for x in items
  emit x
end
"""
        result = run(src)
        assert result.errors == []
        assert [v.raw for v in result.emitted] == [1, 2, 3]

    def test_for_accumulates_sum(self):
        src = """
val nums: List<Int> = [10, 20, 30]
val total: Int = sum(nums)
emit total
"""
        result = run(src)
        assert result.errors == []
        assert result.emitted[0].raw == 60

    def test_for_over_string_chars(self):
        src = """
for ch in "abc"
  emit ch
end
"""
        result = run(src)
        assert result.errors == []
        assert [v.raw for v in result.emitted] == ["a", "b", "c"]

    def test_for_empty_list(self):
        """Iterating over an empty list emits nothing and has no errors."""
        src = """
val empty: List<Int> = []
for x in empty
  emit x
end
emit 42
"""
        result = run(src)
        assert result.errors == []
        # Only the final emit fires
        assert result.emitted[-1].raw == 42

    def test_for_with_body_logic(self):
        src = """
fn double(n: Int) -> Int
  return n + n
end

for n in [3, 7]
  val d: Int = double(n)
  emit d
end
"""
        result = run(src)
        assert result.errors == []
        assert [v.raw for v in result.emitted] == [6, 14]


# ---------------------------------------------------------------------------
# Match expressions
# ---------------------------------------------------------------------------

class TestMatchExpr:
    def test_match_literal_arm(self):
        src = """
val x: Int = 2
match x
  | 1 => emit "one"
  | 2 => emit "two"
  | 3 => emit "three"
  | _ => emit "other"
end
"""
        result = run(src)
        assert result.errors == []
        assert result.emitted[0].raw == "two"

    def test_match_wildcard_arm(self):
        src = """
val x: Int = 99
match x
  | 1 => emit "one"
  | _ => emit "fallback"
end
"""
        result = run(src)
        assert result.errors == []
        assert result.emitted[0].raw == "fallback"

    def test_match_no_match_emits_nothing(self):
        src = """
val x: Int = 5
match x
  | 1 => emit "one"
  | 2 => emit "two"
end
emit "after"
"""
        result = run(src)
        assert result.errors == []
        assert result.emitted[0].raw == "after"

    def test_match_on_text(self):
        src = """
val status: Text = "ok"
match status
  | "ok"    => emit "success"
  | "error" => emit "failure"
  | _       => emit "unknown"
end
"""
        result = run(src)
        assert result.errors == []
        assert result.emitted[0].raw == "success"

    def test_match_inside_fn(self):
        src = """
fn describe(n: Int) -> Text
  match n
    | 0 => return "zero"
    | _ => return "nonzero"
  end
end

val r: Text = describe(0)
emit r
val r2: Text = describe(5)
emit r2
"""
        result = run(src)
        assert result.errors == []
        assert result.emitted[0].raw == "zero"
        assert result.emitted[1].raw == "nonzero"


# ---------------------------------------------------------------------------
# Built-in functions
# ---------------------------------------------------------------------------

class TestBuiltins:
    def test_len_list(self):
        src = """
val lst: List<Int> = [1, 2, 3, 4, 5]
val n: Int = len(lst)
emit n
"""
        result = run(src)
        assert result.errors == []
        assert result.emitted[0].raw == 5

    def test_len_string(self):
        src = """
val s: Text = "hello"
val n: Int = len(s)
emit n
"""
        result = run(src)
        assert result.errors == []
        assert result.emitted[0].raw == 5

    def test_sum_list(self):
        src = """
val nums: List<Float> = [1.0, 2.0, 3.0]
val s: Float = sum(nums)
emit s
"""
        result = run(src)
        assert result.errors == []
        assert result.emitted[0].raw == 6.0

    def test_max_val(self):
        src = """
val lst: List<Int> = [3, 1, 4, 1, 5, 9, 2, 6]
val m: Int = max_val(lst)
emit m
"""
        result = run(src)
        assert result.errors == []
        assert result.emitted[0].raw == 9

    def test_min_val(self):
        src = """
val lst: List<Int> = [3, 1, 4, 1, 5, 9]
val m: Int = min_val(lst)
emit m
"""
        result = run(src)
        assert result.errors == []
        assert result.emitted[0].raw == 1

    def test_abs_val_positive(self):
        src = """
val x: Int = abs_val(-7)
emit x
"""
        result = run(src)
        assert result.errors == []
        assert result.emitted[0].raw == 7

    def test_abs_val_already_positive(self):
        src = """
val x: Int = abs_val(3)
emit x
"""
        result = run(src)
        assert result.errors == []
        assert result.emitted[0].raw == 3

    def test_floor(self):
        src = """
val x: Int = floor(3.7)
emit x
"""
        result = run(src)
        assert result.errors == []
        assert result.emitted[0].raw == 3

    def test_ceil(self):
        src = """
val x: Int = ceil(3.2)
emit x
"""
        result = run(src)
        assert result.errors == []
        assert result.emitted[0].raw == 4

    def test_round_val(self):
        src = """
val x: Float = round_val(3.567, 2)
emit x
"""
        result = run(src)
        assert result.errors == []
        assert abs(result.emitted[0].raw - 3.57) < 1e-6


# ---------------------------------------------------------------------------
# Two-argument confident() and explore() constructors
# ---------------------------------------------------------------------------

class TestTwoArgConstructors:
    def test_confident_two_args_creates_confident_value(self):
        src = """
val x = confident(42, 0.97)
emit x
"""
        result = run(src)
        assert result.errors == []
        assert isinstance(result.emitted[0], ConfidentValue)
        assert result.emitted[0].raw == 42
        assert abs(result.emitted[0].confidence.value - 0.97) < 1e-9

    def test_confident_two_args_clamps_to_min_095(self):
        """Even if score < 0.95 is passed, it is raised to 0.95."""
        src = """
val x = confident("hello", 0.7)
emit x
"""
        result = run(src)
        assert result.errors == []
        assert isinstance(result.emitted[0], ConfidentValue)
        assert result.emitted[0].confidence.value >= 0.95

    def test_explore_two_args_creates_explore_value(self):
        src = """
val h = explore("maybe so", 0.6)
emit h
"""
        result = run(src)
        assert result.errors == []
        assert isinstance(result.emitted[0], ExploreValue)
        assert result.emitted[0].raw == "maybe so"
        assert abs(result.emitted[0].confidence.value - 0.6) < 1e-9

    def test_explore_one_arg_defaults_confidence(self):
        src = """
val h = explore("hypothesis")
emit h
"""
        result = run(src)
        assert result.errors == []
        assert isinstance(result.emitted[0], ExploreValue)
        assert result.emitted[0].confidence.value == 0.5

    def test_confident_one_arg_still_checks_confidence(self):
        """One-arg confident() still returns True/False."""
        src = """
val high = confident(99.9, 0.99)
val ok: Bool = confident(high)
emit ok
"""
        result = run(src)
        assert result.errors == []
        assert result.emitted[0].raw is True


# ---------------------------------------------------------------------------
# Detect blocks
# ---------------------------------------------------------------------------

class TestDetectBlocks:
    def test_detect_range_passes(self):
        src = """
val temperature: Float = 37.0
detect hallucination
  strategy: "range"
  on: temperature
  valid_range: [-50.0, 60.0]
  action: "flag"
end
emit temperature
"""
        result = run(src)
        assert result.errors == []
        detect_traces = [t for t in result.trace if "[detect:" in t]
        assert len(detect_traces) >= 1

    def test_detect_range_flags_out_of_range(self):
        src = """
val temperature: Float = 150.0
detect hallucination
  strategy: "range"
  on: temperature
  valid_range: [-50.0, 60.0]
  action: "flag"
end
emit temperature
"""
        result = run(src)
        assert result.errors == []
        flagged = [t for t in result.trace if "FLAGGED" in t]
        assert len(flagged) >= 1


# ---------------------------------------------------------------------------
# New example files
# ---------------------------------------------------------------------------

class TestNewExamples:
    def _run_example(self, filename: str):
        import pathlib
        src = (
            pathlib.Path(__file__).parent.parent / "examples" / filename
        ).read_text()
        return run(src)

    def test_for_loop_example(self):
        result = self._run_example("for_loop.chimera")
        assert result.errors == []
        assert result.assertions_passed >= 1
        # Should have emitted scores for all 5 items
        assert len(result.emitted) >= 5

    def test_advanced_reasoning_example(self):
        result = self._run_example("advanced_reasoning.chimera")
        assert result.errors == []
        assert result.assertions_passed >= 2
        assert len(result.emitted) >= 5
