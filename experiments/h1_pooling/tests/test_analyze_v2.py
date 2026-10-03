"""Unit tests for analyze_v2.py: token-F1, cross-fitting, risk-coverage,
AURC, guard simulation, 2-D logistic regression, plus an end-to-end run
on the synthetic fixture."""
import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "..")
sys.path.insert(0, EXP)

from analyze_v2 import (  # noqa: E402
    aurc,
    cross_fit_scores,
    fit_logistic_2d,
    guard_sim,
    k_fold,
    risk_coverage,
    sigmoid,
    token_f1,
)

DATASET = os.path.join(EXP, "fixture", "questions.jsonl")


def test_token_f1_identical():
    assert token_f1("Michio Sugeno", ["Michio Sugeno"]) == 1.0


def test_token_f1_disjoint():
    assert token_f1("Paris", ["oxygen"]) == 0.0


def test_token_f1_partial():
    # pred tokens {michio, sugeno}, gold {sugeno} -> P=1/2, R=1/1
    assert token_f1("Michio Sugeno", ["Sugeno"]) == pytest.approx(2 / 3, abs=1e-9)


def test_token_f1_takes_max_over_aliases():
    assert token_f1("Paris", ["oxygen", "Paris"]) == 1.0


def test_k_fold_partitions():
    qids = [f"q{i}" for i in range(10)]
    folds = k_fold(qids, 5, 20261002)
    flat = sorted(q for f in folds for q in f)
    assert flat == sorted(qids)
    assert all(len(f) == 2 for f in folds)
    # deterministic
    assert k_fold(qids, 5, 20261002) == folds


def test_cross_fit_covers_every_question_once():
    qids = [f"q{i}" for i in range(10)]
    labels = {q: i % 2 for i, q in enumerate(qids)}
    oof = cross_fit_scores(qids, {}, labels, "K")
    assert sorted(oof) == sorted(qids)
    # each held-out score is the base rate of its 8 training questions
    folds = k_fold(qids, 5, 20261002)
    for fold in folds:
        train = [q for q in qids if q not in fold]
        base = sum(labels[q] for q in train) / len(train)
        for q in fold:
            assert oof[q] == pytest.approx(base)


def test_cross_fit_vc_monotone():
    qids = [f"q{i}" for i in range(20)]
    feats = {q: i / 19 for i, q in enumerate(qids)}
    labels = {q: 1 if i >= 10 else 0 for i, q in enumerate(qids)}
    oof = cross_fit_scores(qids, feats, labels, "Vc")
    lo = [oof[q] for q in qids[:10]]
    hi = [oof[q] for q in qids[10:]]
    assert sum(hi) / len(hi) > sum(lo) / len(lo)


def test_risk_coverage_known():
    scores = [0.9, 0.8, 0.2, 0.1]
    labels = [1, 1, 0, 0]
    rc = risk_coverage(scores, labels)
    assert rc[20] == 1.0   # top 1 of 4
    assert rc[40] == 1.0   # top 2 of 4
    assert rc[60] == pytest.approx(2 / 3)
    assert rc[100] == 0.5


def test_aurc_perfect_ranking():
    scores = [0.9, 0.8, 0.2, 0.1]
    labels = [1, 1, 0, 0]
    # risks: 0, 0, 1/3, 1/2 -> mean
    assert aurc(scores, labels) == pytest.approx((0 + 0 + 1 / 3 + 0.5) / 4)


def test_aurc_worst_ranking():
    scores = [0.1, 0.2, 0.8, 0.9]
    labels = [1, 1, 0, 0]
    # risks: 1, 1, 2/3, 1/2 -> mean
    assert aurc(scores, labels) == pytest.approx((1 + 1 + 2 / 3 + 0.5) / 4)


def test_guard_sim_known():
    scores = [0.9, 0.8, 0.4]
    labels = [1, 0, 1]
    gs = guard_sim(scores, labels, thresholds=[0.5, 0.85])
    assert gs[0.5] == (pytest.approx(2 / 3), pytest.approx(0.5))
    assert gs[0.85] == (pytest.approx(1 / 3), pytest.approx(1.0))


def test_fit_logistic_2d_deterministic_and_sane():
    xs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0),
          (0.2, 0.1), (0.9, 0.8), (0.8, 0.9), (0.1, 0.2)]
    ys = [0, 0, 1, 1, 0, 1, 1, 0]
    a1, b1, c1 = fit_logistic_2d(xs, ys)
    a2, b2, c2 = fit_logistic_2d(xs, ys)
    assert (a1, b1, c1) == (a2, b2, c2)
    p_hi = sigmoid(a1 * 1.0 + b1 * 1.0 + c1)
    p_lo = sigmoid(a1 * 0.0 + b1 * 0.0 + c1)
    assert p_hi > 0.5 > p_lo


def test_end_to_end_synthetic(tmp_path):
    out = str(tmp_path / "r.jsonl")
    cmd = [sys.executable, os.path.join(EXP, "collect.py"),
           "--dataset", DATASET, "--out", out, "--backend", "synthetic",
           "--mode", "B", "--models", "claude", "--k", "3", "--seed", "11"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    cmd = [sys.executable, os.path.join(EXP, "analyze_v2.py"),
           "--responses", out, "--bootstrap", "20"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "EXPECTATION E1_P10_saturation" in r.stdout
    assert "EXPECTATION E3_no_auc_above_06" in r.stdout
    assert "grading audit" in r.stdout
