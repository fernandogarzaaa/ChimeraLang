#!/usr/bin/env python3
"""Fit one LogisticCalibrator per source configuration on H1 questions 1-500.

Mode A: 3 sources, cross-model (3 models x 1 sample).
Mode B: 5 sources, single-model self-consistency (5 samples x 1 model).

Each calibrator records n_sources and source_type in its JSON metadata.
Training pairs come only from questions 1-500 replayed through the
redesigned engine's agreement resolve. No network. No model calls.

Usage: fit_per_mode_calibrators.py <responses.jsonl> <out_dir>
Writes agreement_v1_modeA.json and agreement_v1_modeB.json.
"""
from __future__ import annotations

import hashlib
import json
import sys

sys.path.insert(0, ".")

from chimera.cir.agreement import normalize_answer, resolve_agreement
from chimera.cir.calibration import LogisticCalibrator

CONFIGS = {
    "A": {"n_sources": 3, "source_type": "cross-model"},
    "B": {"n_sources": 5, "source_type": "single-model"},
}


def is_correct(predicted, gold_answers):
    pred = normalize_answer(predicted)
    if not pred:
        return False
    return any(pred == normalize_answer(g) for g in gold_answers)


def main():
    responses_path, out_dir = sys.argv[1], sys.argv[2]
    by_mode: dict[str, dict[int, list]] = {"A": {}, "B": {}}
    gold: dict[int, list] = {}
    with open(responses_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            qi = r["question_index"]
            gold[qi] = r["gold_answers"]
            by_mode[r["mode"]].setdefault(qi, []).append(r)

    for mode, cfg in CONFIGS.items():
        scores, outcomes = [], []
        for qi in sorted(by_mode[mode]):
            samples = by_mode[mode][qi]
            assert len(samples) == cfg["n_sources"], (
                f"mode {mode} q{qi}: {len(samples)} sources, "
                f"expected {cfg['n_sources']}")
            ar = resolve_agreement([s["parsed_answer"] for s in samples])
            scores.append(ar.posterior.mean)
            outcomes.append(1 if is_correct(ar.winner, gold[qi]) else 0)
        cal = LogisticCalibrator.fit(scores, outcomes, metadata=dict(cfg))
        path = f"{out_dir}/agreement_v1_mode{mode}.json"
        with open(path, "w", encoding="utf-8") as f:
            f.write(cal.to_json())
        raw = open(path, "rb").read()
        print(f"mode {mode}: n={cal.n} a={cal.a:.4f} b={cal.b:.4f} "
              f"metadata={cal.metadata}")
        print(f"  -> {path}")
        print(f"  file sha256: {hashlib.sha256(raw).hexdigest()}")


if __name__ == "__main__":
    main()
