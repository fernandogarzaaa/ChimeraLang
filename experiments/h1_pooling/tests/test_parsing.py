import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from h1_common import parse_response


def test_basic():
    a, c, ok = parse_response("Answer: Paris\nConfidence: 0.85\n")
    assert ok and a == "Paris" and abs(c - 0.85) < 1e-9


def test_case_insensitive_and_spaces():
    a, c, ok = parse_response("  answer:  42 \n  CONFIDENCE: 0.5\n")
    assert ok and a == "42" and abs(c - 0.5) < 1e-9


def test_missing_confidence():
    _, _, ok = parse_response("Answer: Paris\n")
    assert not ok


def test_bad_confidence():
    _, _, ok = parse_response("Answer: Paris\nConfidence: very sure\n")
    assert not ok


def test_out_of_range():
    _, _, ok = parse_response("Answer: Paris\nConfidence: 1.5\n")
    assert not ok


def test_extra_lines_ignored():
    a, c, ok = parse_response("Thinking...\nAnswer: Mars\nConfidence: 0.70\nDone.\n")
    assert ok and a == "Mars" and abs(c - 0.70) < 1e-9
