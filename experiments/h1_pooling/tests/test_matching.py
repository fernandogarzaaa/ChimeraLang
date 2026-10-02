import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from h1_common import is_correct, normalize_answer


def test_normalize_articles_punct():
    assert normalize_answer("The Pacific Ocean!") == "pacific ocean"
    assert normalize_answer("  a   cat ") == "cat"


def test_alias_match():
    assert is_correct("Shakespeare", ["William Shakespeare", "Shakespeare"])
    assert is_correct("the pacific ocean", ["Pacific Ocean", "Pacific"])


def test_no_match():
    assert not is_correct("London", ["Paris"])


def test_empty_prediction():
    assert not is_correct("   ", ["Paris"])
