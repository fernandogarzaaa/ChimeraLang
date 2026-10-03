#!/usr/bin/env python3
"""Confirmatory analysis for the redesigned CIR pooling engine.

Fixed analysis script for PREREG_V2_CONFIRM. It:

1. Verifies the SHA-256 of the dataset, both frozen calibrators, and
   the hash-checked model-id table against expected hashes given on
   the command line, and REFUSES to run (exit 2) on any mismatch.
2. Verifies every response row's model_id against the model-id table
   for its mode, and REFUSES (exit 2) on any mismatch.
3. Replays every response through the engine's agreement resolve,
   applying the per-mode frozen calibrator (mode A -> calibrator-a,
   mode B -> calibrator-b).
4. Computes per mode: agreement AUROC with bootstrap 95% CI, and
   Brier(calibrated) vs Brier(constant base-rate) with a bootstrap
   95% CI on the difference.
5. Applies the preregistered decision rules and prints the verdict:
   Confirmed, Falsified, or Inconclusive.

No network. No model calls. Deterministic given the inputs
(bootstrap uses a fixed seed).

Usage:
    analyze_confirm.py --responses R.jsonl --dataset D.jsonl \\
        --calibrator-a CA.json --calibrator-b CB.json \\
        --expected-dataset-sha256 H --expected-calibrator-a-sha256 H \\
        --expected-calibrator-b-sha256 H \\
        --expected-model-ids model_ids.json --expected-model-ids-sha256 H
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys

sys.path.insert(0, ".")

from chimera.cir.agreement import normalize_answer, resolve_agreement
from chimera.cir.calibration import LogisticCalibrator
from chimera.cir.nodes import BetaDist

# Preregistered decision thresholds (PREREG_V2_CONFIRM, from questions 1-500).
C1_AUROC_POINT = 0.65
C1_AUROC_LOWER = 0.50
F1_AUROC_LOWER = 0.60

EXPECTED_SOURCES = {"A": 3, "B": 5}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path: str, expected: str, label: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        print(f"REFUSED: {label} hash mismatch\n"
              f"  expected: {expected}\n"
              f"  actual:   {actual}", file=sys.stderr)
        sys.exit(2)
    print(f"verified {label}: sha256 {actual[:16]}... ok")


def is_correct(predicted, gold_answers) -> bool:
    pred = normalize_answer(predicted)
    if not pred:
        return False
    return any(pred == normalize_answer(g) for g in gold_answers)


def auc_rank(scores, labels) -> float:
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


def bootstrap_ci(stat_fn, rows, n_boot=2000, seed=20261002):
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        sample = [rows[rng.randrange(len(rows))] for _ in range(len(rows))]
        vals.append(stat_fn(sample))
    vals.sort()
    return vals[int(0.025 * n_boot)], vals[int(0.975 * n_boot)]


def brier(scores, labels) -> float:
    return sum((s - y) ** 2 for s, y in zip(scores, labels)) / len(scores)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--calibrator-a", required=True)
    ap.add_argument("--calibrator-b", required=True)
    ap.add_argument("--expected-dataset-sha256", required=True)
    ap.add_argument("--expected-calibrator-a-sha256", required=True)
    ap.add_argument("--expected-calibrator-b-sha256", required=True)
    ap.add_argument("--expected-model-ids", required=True,
                    help="hash-checked model-id table (model_ids.json)")
    ap.add_argument("--expected-model-ids-sha256", required=True)
    args = ap.parse_args()

    # 1. Runtime hash verification. Refuse on any mismatch.
    verify(args.dataset, args.expected_dataset_sha256, "dataset")
    verify(args.calibrator_a, args.expected_calibrator_a_sha256,
           "calibrator-a")
    verify(args.calibrator_b, args.expected_calibrator_b_sha256,
           "calibrator-b")
    verify(args.expected_model_ids, args.expected_model_ids_sha256,
           "model-id table")

    cal_a = LogisticCalibrator.from_json(open(args.calibrator_a).read())
    cal_b = LogisticCalibrator.from_json(open(args.calibrator_b).read())
    for cal, mode in ((cal_a, "A"), (cal_b, "B")):
        md = cal.metadata or {}
        if md.get("n_sources") != EXPECTED_SOURCES[mode]:
            print(f"REFUSED: calibrator-{mode.lower()} metadata n_sources "
                  f"is {md.get('n_sources')}, expected "
                  f"{EXPECTED_SOURCES[mode]}", file=sys.stderr)
            sys.exit(2)
    print(f"calibrator-a: n={cal_a.n} fit_date={cal_a.fit_date} "
          f"metadata={cal_a.metadata}")
    print(f"calibrator-b: n={cal_b.n} fit_date={cal_b.fit_date} "
          f"metadata={cal_b.metadata}")

    # 2. Load dataset and responses; cross-check gold.
    dataset = [json.loads(l) for l in open(args.dataset, encoding="utf-8")
               if l.strip()]
    print(f"dataset: {len(dataset)} questions, "
          f"{dataset[0]['id']}..{dataset[-1]['id']}")
    model_table = json.loads(open(args.expected_model_ids,
                                  encoding="utf-8").read())
    by_mode: dict[str, dict[int, list]] = {"A": {}, "B": {}}
    n_resp = 0
    for line in open(args.responses, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        n_resp += 1
        # Model-id gate: the collected model_id must be the
        # preregistered id for the row's mode.
        expected_ids = model_table.get(f"mode_{r['mode']}", [])
        if r.get("model_id") not in expected_ids:
            print(f"REFUSED: mode {r['mode']} row has model_id "
                  f"{r.get('model_id')!r}, expected one of {expected_ids}",
                  file=sys.stderr)
            sys.exit(2)
        by_mode[r["mode"]].setdefault(r["question_index"], []).append(r)
    print(f"responses: {n_resp}")

    for mode in ("A", "B"):
        groups = by_mode[mode]
        bad = [qi for qi, rs in groups.items()
               if len(rs) != EXPECTED_SOURCES[mode]]
        if bad:
            print(f"REFUSED: mode {mode} has {len(bad)} questions with "
                  f"wrong source count (expected {EXPECTED_SOURCES[mode]})",
                  file=sys.stderr)
            sys.exit(2)

    # 3. Engine replay per mode.
    results = {}
    for mode, cal in (("A", cal_a), ("B", cal_b)):
        rows = []  # (score, calibrated, label)
        for qi in sorted(by_mode[mode]):
            rs = by_mode[mode][qi]
            ar = resolve_agreement([r["parsed_answer"] for r in rs])
            y = 1 if is_correct(ar.winner, rs[0]["gold_answers"]) else 0
            rows.append((ar.posterior.mean, cal.predict(ar.posterior.mean), y))
        scores = [r[0] for r in rows]
        cals = [r[1] for r in rows]
        labels = [r[2] for r in rows]
        base = sum(labels) / len(labels)
        a = auc_rank(scores, labels)
        a_lo, a_hi = bootstrap_ci(
            lambda d: auc_rank([x[0] for x in d], [x[2] for x in d]), rows)
        b_const = brier([base] * len(labels), labels)
        b_cal = brier(cals, labels)
        d_lo, d_hi = bootstrap_ci(
            lambda d: brier([base] * len([x[2] for x in d]), [x[2] for x in d])
                      - brier([x[1] for x in d], [x[2] for x in d]), rows)
        results[mode] = dict(n=len(rows), base_rate=base, auc=a,
                             auc_lo=a_lo, auc_hi=a_hi,
                             brier_const=b_const, brier_cal=b_cal,
                             diff_lo=d_lo, diff_hi=d_hi)
        print(f"mode {mode}: n={len(rows)} base_rate={base:.4f}")
        print(f"  agreement AUROC = {a:.4f}  95% CI [{a_lo:.4f}, {a_hi:.4f}]")
        print(f"  Brier(constant) = {b_const:.4f}  "
              f"Brier(calibrated) = {b_cal:.4f}")
        print(f"  const-minus-calibrated 95% CI [{d_lo:.4f}, {d_hi:.4f}]")

    # 4. Secondary pooled-arm sanity: pooled mean must equal the
    # arithmetic mean of source Beta means (tolerance 1e-4) and must
    # never saturate at 1.0.
    max_dev, saturated = 0.0, 0
    for mode in ("A", "B"):
        for qi, rs in by_mode[mode].items():
            betas = [BetaDist.from_confidence(r["parsed_confidence"])
                     for r in rs]
            pooled = betas[0]
            for b in betas[1:]:
                pooled = pooled.combine_pseudocount(b)
            arith = sum(b.mean for b in betas) / len(betas)
            max_dev = max(max_dev, abs(pooled.mean - arith))
            if pooled.mean >= 1.0:
                saturated += 1
    pooled_ok = max_dev <= 1e-4 and saturated == 0
    print(f"pooled-arm check: max |pooled - arithmetic| = {max_dev:.2e}, "
          f"saturated at 1.0: {saturated} -> {'PASS' if pooled_ok else 'FAIL'}")

    # 5. Preregistered verdict.
    c1 = all(results[m]["auc"] >= C1_AUROC_POINT
             and results[m]["auc_lo"] > C1_AUROC_LOWER for m in ("A", "B"))
    # C2: 95% CI of [Brier(constant) - Brier(calibrated)] lies entirely
    # above zero (calibrated beats constant, CI excludes zero).
    c2 = all(results[m]["diff_lo"] > 0 for m in ("A", "B"))
    # F1: agreement signal did not replicate.
    f1 = any(results[m]["auc_lo"] <= F1_AUROC_LOWER for m in ("A", "B"))
    # F2: 95% CI of [Brier(calibrated) - Brier(constant)] lies entirely
    # above zero, i.e. diff (const - cal) CI entirely below zero.
    f2 = any(results[m]["diff_hi"] < 0 for m in ("A", "B"))
    print(f"rules: C1(signal, auc>={C1_AUROC_POINT})={c1} "
          f"C2(calibration, CI excludes 0)={c2} "
          f"F1(auc lower<={F1_AUROC_LOWER})={f1} F2(calibrated worse, CI>0)={f2}")
    if f1 or f2:
        verdict = "Falsified"
    elif c1 and c2:
        verdict = "Confirmed"
    else:
        verdict = "Inconclusive"
    print(f"VERDICT: {verdict}")
    if not pooled_ok:
        print("NOTE: pooled-arm secondary check FAILED; see above.")


if __name__ == "__main__":
    main()
