import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analyze import icc_one_way, logit_clipped


def test_icc_perfect_agreement():
    m = [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [-1.0, -1.0, -1.0]]
    assert abs(icc_one_way(m) - 1.0) < 1e-9


def test_icc_no_agreement():
    # every question has the same spread -> between-question variance 0
    m = [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
    v = icc_one_way(m)
    assert v <= 0.0


def test_icc_unbalanced_nan():
    assert math.isnan(icc_one_way([[1.0, 2.0], [3.0]]))


def test_logit_clipped():
    assert logit_clipped(0.5) == 0.0
    assert logit_clipped(0.0) == -6.0
    assert logit_clipped(1.0) == 6.0
    assert abs(logit_clipped(0.9) - math.log(9)) < 1e-9
