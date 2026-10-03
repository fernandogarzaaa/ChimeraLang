#!/usr/bin/env python3
"""Smoke-test gate for the confirmatory run (PREREG_V2_CONFIRM).

Before the full 4,000-call confirmatory run, 10 questions per model are
collected (mode A: 3 models x 10 questions = 30 calls; mode B: 1 model x
10 questions x 5 samples = 50 calls; 80 calls total). This script
verifies the smoke responses and writes a run manifest:

- exactly 10 questions per model per mode, all parse_ok
- every collected model_id matches the hash-checked model-id table
- every question replays cleanly through the engine's agreement resolve
- manifest records model ids and timestamps

Correctness is deliberately NOT inspected at the smoke gate: grading
smoke responses would peek at confirmatory outcomes. The gate checks
pipeline health only (collection, parsing, engine replay).

Exit 0 and "SMOKE: PASS" when the gate passes; exit 1 and "SMOKE: FAIL"
otherwise. No network. No model calls.

Usage:
    smoke_test.py --responses smoke.jsonl --dataset D.jsonl \\
        --manifest manifest.json [--expected-dataset-sha256 H] \\
        --expected-model-ids model_ids.json \\
        --expected-model-ids-sha256 H
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone

sys.path.insert(0, ".")

from chimera.cir.agreement import resolve_agreement

QUESTIONS_PER_MODEL = 10


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--expected-dataset-sha256", default=None)
    ap.add_argument("--expected-model-ids", default=None,
                    help="hash-checked model-id table (model_ids.json)")
    ap.add_argument("--expected-model-ids-sha256", default=None)
    args = ap.parse_args()

    started = datetime.now(timezone.utc).isoformat()
    failures: list[str] = []

    dataset_hash = sha256_file(args.dataset)
    if args.expected_dataset_sha256 and \
            dataset_hash != args.expected_dataset_sha256:
        print(f"SMOKE: FAIL: dataset hash mismatch\n"
              f"  expected: {args.expected_dataset_sha256}\n"
              f"  actual:   {dataset_hash}")
        sys.exit(1)

    model_table: dict[str, list[str]] = {}
    model_table_hash = None
    if args.expected_model_ids:
        model_table_hash = sha256_file(args.expected_model_ids)
        if args.expected_model_ids_sha256 and \
                model_table_hash != args.expected_model_ids_sha256:
            print(f"SMOKE: FAIL: model-id table hash mismatch\n"
                  f"  expected: {args.expected_model_ids_sha256}\n"
                  f"  actual:   {model_table_hash}")
            sys.exit(1)
        with open(args.expected_model_ids, encoding="utf-8") as f:
            model_table = json.load(f)
        print(f"verified model-id table: sha256 {model_table_hash[:16]}... ok")

    by_model: dict[tuple[str, str], dict] = {}
    seen_bad_ids: set[tuple[str, str | None]] = set()
    for line in open(args.responses, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        key = (r["mode"], r.get("model_id") or r.get("model", "?"))
        by_model.setdefault(key, []).append(r)
        # Model-id gate: the collected model_id must be in the
        # hash-checked table for the row's mode.
        if model_table and (r["mode"], r.get("model_id")) not in seen_bad_ids:
            expected = model_table.get(f"mode_{r['mode']}", [])
            if r.get("model_id") not in expected:
                seen_bad_ids.add((r["mode"], r.get("model_id")))
                failures.append(
                    f"model-id mismatch: mode {r['mode']} collected "
                    f"model_id {r.get('model_id')!r}, expected one of "
                    f"{expected}")

    for (mode, model_id), rs in sorted(by_model.items()):
        qids = {r["question_index"] for r in rs}
        if len(qids) != QUESTIONS_PER_MODEL:
            failures.append(
                f"{mode}/{model_id}: {len(qids)} questions, "
                f"expected {QUESTIONS_PER_MODEL}")
        bad_parse = [r["question_index"] for r in rs if not r.get("parse_ok")]
        if bad_parse:
            failures.append(
                f"{mode}/{model_id}: parse failures on {bad_parse}")
        # Engine replay: agreement resolve must run cleanly per question.
        by_q: dict[int, list] = {}
        for r in rs:
            by_q.setdefault(r["question_index"], []).append(r)
        for qi, qs in by_q.items():
            try:
                resolve_agreement([q["parsed_answer"] for q in qs])
            except Exception as e:  # noqa: BLE001 - gate must catch all
                failures.append(
                    f"{mode}/{model_id} q{qi}: engine replay raised {e!r}")

    ended = datetime.now(timezone.utc).isoformat()
    manifest = {
        "smoke_test": True,
        "questions_per_model": QUESTIONS_PER_MODEL,
        "dataset": args.dataset,
        "dataset_sha256": dataset_hash,
        "model_id_table": args.expected_model_ids,
        "model_id_table_sha256": model_table_hash,
        "correctness_inspected": False,
        "started_at": started,
        "ended_at": ended,
        "models": [
            {"mode": mode, "model_id": model_id,
             "provider_model_ids": sorted({r.get("provider_model_id")
                                           for r in rs
                                           if r.get("provider_model_id")}),
             "n_questions": len({r["question_index"] for r in rs}),
             "n_responses": len(rs),
             "parse_ok_rate": sum(1 for r in rs if r.get("parse_ok")) / len(rs)}
            for (mode, model_id), rs in sorted(by_model.items())
        ],
        "failures": failures,
        "passed": not failures,
    }
    with open(args.manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    if failures:
        print("SMOKE: FAIL")
        for fl in failures:
            print(f"  - {fl}")
        print(f"manifest: {args.manifest}")
        sys.exit(1)
    print(f"SMOKE: PASS ({len(by_model)} model-configs, "
          f"{QUESTIONS_PER_MODEL} questions each)")
    print(f"manifest: {args.manifest} "
          f"(model ids and timestamps recorded)")


if __name__ == "__main__":
    main()
