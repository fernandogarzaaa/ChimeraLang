#!/usr/bin/env python3
"""Derive confirmatory decision thresholds from H1 questions 1-500.

For each mode (A: 3 models, B: 5 samples), computes via the redesigned
engine's agreement resolve:
  - agreement AUROC (posterior mean ranking correctness) + bootstrap CI
  - 5-fold cross-fit calibrated Brier vs constant-predictor Brier

No network. No model calls. Pure replay of runs/2026-10-02-nebius.
Prints the numbers the prereg thresholds are derived from.
"""
from __future__ import annotations

import json
import random
import sys

sys.path.insert(0, ".")

from chimera.cir.agreement import normalize_answer, resolve_agreement
from chimera.cir.calibration import LogisticCalibrator


def is_correct(predicted, gold_answers):
    pred = normalize_answer(predicted)
    if not pred:
        return False
    return any(pred == normalize_answer(g) for g in gold_answers)


def load(path):
    by_qa, by_qb = {}, {}
    gold = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            qi = r["question_index"]
            gold[qi] = r["gold_answers"]
            if r["mode"] == "A":
                by_qa.setdefault(qi, []).append(r)
            else:
                by_qb.setdefault(qi, []).append(r)
    return by_qa, by_qb, gold


def per_question(samples, gold_answers):
    ar = resolve_agreement([s["parsed_answer"] for s in samples])
    y = 1 if is_correct(ar.winner, gold_answers) else 0
    return ar.posterior.mean, y, ar.agreement


def auc(scores, labels):
    # Mann-Whitney U / rank AUC, ties averaged.
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    n1 = sum(labels)
    n0 = len(labels) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    s = sum(r for r, y in zip(ranks, labels) if y == 1)
    return (s - n1 * (n1 + 1) / 2) / (n1 * n0)


def bootstrap_ci(stat_fn, data, n_boot=2000, seed=20261002):
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        sample = [data[rng.randrange(len(data))] for _ in range(len(data))]
        vals.append(stat_fn(sample))
    vals.sort()
    return vals[int(0.025 * n_boot)], vals[int(0.975 * n_boot)]


def brier(scores, labels):
    return sum((s - y) ** 2 for s, y in zip(scores, labels)) / len(scores)


def main():
    path = sys.argv[1]
    by_qa, by_qb, gold = load(path)

    for mode, by_q in (("A", by_qa), ("B", by_qb)):
        rows = []
        for qi in sorted(by_q):
            score, y, agree = per_question(by_q[qi], gold[qi])
            rows.append((score, y))
        scores = [r[0] for r in rows]
        labels = [r[1] for r in rows]
        base_rate = sum(labels) / len(labels)
        a = auc(scores, labels)
        lo, hi = bootstrap_ci(lambda d: auc([x[0] for x in d], [x[1] for x in d]), rows)
        const_brier = brier([base_rate] * len(labels), labels)
        uncal_brier = brier(scores, labels)
        print(f"mode {mode}: n={len(rows)} base_rate={base_rate:.4f}")
        print(f"  agreement AUROC = {a:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
        print(f"  Brier(constant) = {const_brier:.4f}")
        print(f"  Brier(uncalibrated posterior mean) = {uncal_brier:.4f}")

        # 5-fold cross-fit calibrated Brier (what the calibrator can do)
        rng = random.Random(20261002)
        idx = list(range(len(rows)))
        rng.shuffle(idx)
        folds = [idx[i::5] for i in range(5)]
        cal_scores = [None] * len(rows)
        for k in range(5):
            train = [i for j in range(5) if j != k for i in folds[j]]
            cal = LogisticCalibrator.fit([scores[i] for i in train],
                                         [labels[i] for i in train])
            for i in folds[k]:
                cal_scores[i] = cal.predict(scores[i])
        cb = brier(cal_scores, labels)
        d = const_brier - cb
        dlo, dhi = bootstrap_ci(
            lambda dd: brier([base_rate] * len(dd), [x[1] for x in dd])
                       - brier([x[2] for x in dd], [x[1] for x in dd]),
            list(zip(scores, labels, cal_scores)))
        print(f"  Brier(cross-fit calibrated) = {cb:.4f}")
        print(f"  const - calibrated = {d:.4f}  95% CI [{dlo:.4f}, {dhi:.4f}]")

    # Fit the production calibrator on all 1000 points (both modes).
    all_scores, all_labels = [], []
    for by_q in (by_qa, by_qb):
        for qi in sorted(by_q):
            score, y, _ = per_question(by_q[qi], gold[qi])
            all_scores.append(score)
            all_labels.append(y)
    cal = LogisticCalibrator.fit(all_scores, all_labels)
    out = sys.argv[2] if len(sys.argv) > 2 else None
    if out:
        with open(out, "w", encoding="utf-8") as f:
            f.write(cal.to_json())
    print(f"production calibrator: n={cal.n} a={cal.a:.4f} b={cal.b:.4f} "
          f"hash={cal.dataset_hash[:16]}... date={cal.fit_date}")


if __name__ == "__main__":
    main()
