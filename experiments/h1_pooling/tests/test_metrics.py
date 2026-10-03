import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analyze import auroc, brier, ece, fit_platt, percentile_ci, platt_apply


def test_brier_hand():
    # ((0.8-1)^2 + (0.2-0)^2) / 2 = (0.04 + 0.04)/2 = 0.04
    assert abs(brier([0.8, 0.2], [1, 0]) - 0.04) < 1e-12


def test_ece_perfect():
    # claimed == correctness in every bin -> ECE 0
    assert ece([0.0, 0.0, 1.0, 1.0], [0, 0, 1, 1]) < 1e-12


def test_ece_known():
    # two samples at 0.9, one right: bin acc 0.5, conf 0.9 -> 0.4
    assert abs(ece([0.9, 0.9], [1, 0]) - 0.4) < 1e-12


def test_auroc_perfect():
    assert abs(auroc([0.9, 0.1, 0.8, 0.2], [1, 0, 1, 0]) - 1.0) < 1e-12


def test_auroc_chance():
    assert abs(auroc([0.5, 0.5], [1, 0]) - 0.5) < 1e-12


def test_auroc_single_class_nan():
    import math
    assert math.isnan(auroc([0.9, 0.8], [1, 1]))


def test_bootstrap_ci_deterministic():
    import random
    vals = [random.Random(3).random() for _ in range(50)]
    lo1, hi1, d1 = percentile_ci(vals)
    lo2, hi2, d2 = percentile_ci(vals)
    assert (lo1, hi1, d1) == (lo2, hi2, d2)
    assert lo1 <= hi1


def test_bootstrap_ci_known_width():
    # uniform-ish spread: CI must bracket the median-ish region sanely
    lo, hi, _ = percentile_ci([float(i) for i in range(100)])
    assert lo < 50 < hi
    assert 0 <= lo <= 25 and 75 <= hi <= 99


def test_platt_deterministic_and_monotone():
    xs = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    ys = [0, 0, 0, 1, 1, 1]
    a1, b1 = fit_platt(xs, ys)
    a2, b2 = fit_platt(xs, ys)
    assert (a1, b1) == (a2, b2)
    assert platt_apply(a1, b1, 0.9) > platt_apply(a1, b1, 0.1)
    assert a1 > 0  # higher claimed -> higher calibrated


def test_platt_single_class_guarded():
    a, b = fit_platt([0.5, 0.6], [1, 1])
    assert b > 0
    a, b = fit_platt([0.5, 0.6], [0, 0])
    assert b < 0
