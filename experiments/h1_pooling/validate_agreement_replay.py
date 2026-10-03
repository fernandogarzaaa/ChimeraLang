#!/usr/bin/env python3
"""Validate the agreement resolve against the H1 experiment's V arm.

Feeds the first N questions' mode-A records (3 models) from
runs/2026-10-02-nebius/responses.jsonl into run_cir via a replay
adapter (one agent per model, no network, no model calls), then
compares per question:

  - engine agreement vs the experiment's V-arm vote share
  - engine winner (normalized) vs the experiment's winner (normalized)

Also verifies the Stage-2 algebra property through the real path:
with strategy="pooled", the pooled mean equals the arithmetic mean of
the confidences for equal strengths.

Usage:
    validate_agreement_replay.py <responses.jsonl> [n_questions=100]
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter

sys.path.insert(0, ".")

from chimera.ast_nodes import (
    BeliefDecl, EmitStmt, Identifier, InquireExpr, Program, ResolveStmt,
)
from chimera.cir import run_cir
from chimera.cir.agreement import normalize_answer, resolve_agreement

AGENTS = ["qwen3-235b", "deepseek-v4pro", "gemma-3-27b"]


def load_mode_a(path):
    """question_index -> list of records in file order (mode A only)."""
    by_q: dict[int, list[dict]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["mode"] != "A":
                continue
            by_q.setdefault(r["question_index"], []).append(r)
    return by_q


def experiment_v(samples):
    """Replicate analyze.py majority_answer for one question.

    Returns (winner_normalized, vote_share). Ties -> earliest sample.
    """
    counts: Counter[str] = Counter()
    first_seen: dict[str, int] = {}
    for i, s in enumerate(samples):
        key = normalize_answer(s["parsed_answer"])
        counts[key] += 1
        if key not in first_seen:
            first_seen[key] = i
    best = sorted(counts, key=lambda k: (-counts[k], first_seen[k]))[0]
    return best, counts[best] / len(samples)


def make_program(strategy=None):
    stmts = [BeliefDecl(name="x", inquire_expr=InquireExpr(
        prompt="Q", agents=list(AGENTS), ttl=None))]
    if strategy is None:
        stmts.append(ResolveStmt(target="x", threshold=0.0))
    else:
        stmts.append(ResolveStmt(target="x", threshold=0.0, strategy=strategy))
    stmts.append(EmitStmt(value=Identifier(name="x")))
    return Program(declarations=stmts)


_AGREE_RE = re.compile(
    r"agreement=([0-9.]+) \((\d+)/(\d+) votes\) posterior_mean=([0-9.]+) winner='(.*)'")


def engine_agreement(records):
    """Run the agreement strategy on one question's records via run_cir.

    Returns the exact AgreementResult (same function the engine calls)
    plus a check that the real path emitted the agreement trace line.
    """
    by_model = {r["model"]: r for r in records}

    def adapter(prompt, agents):
        r = by_model[agents[0]]
        return {"confidence": r["parsed_confidence"], "answer": r["parsed_answer"]}

    result = run_cir(make_program(), inquiry_adapter=adapter)
    assert any("agreement=" in t and "uncalibrated" in t for t in result.trace), \
        "agreement strategy did not run through run_cir"
    ar = resolve_agreement([by_model[a]["parsed_answer"] for a in AGENTS])
    return {
        "agreement": ar.agreement,
        "votes": ar.votes,
        "n": ar.n,
        "posterior_mean": ar.posterior.mean,
        "winner": ar.winner,
        "conflicted": bool(result.guard_violations),
    }


def engine_pooled_mean(records):
    """Run the pooled strategy on one question's records via run_cir."""
    by_model = {r["model"]: r for r in records}

    def adapter(prompt, agents):
        r = by_model[agents[0]]
        return {"confidence": r["parsed_confidence"], "answer": r["parsed_answer"]}

    result = run_cir(make_program(strategy="pooled"), inquiry_adapter=adapter)
    if result.guard_violations:
        return None, result.guard_violations
    return result.beliefs[f"x@{AGENTS[0]}"].mean, None


def main():
    path = sys.argv[1]
    n_questions = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    by_q = load_mode_a(path)
    qids = sorted(by_q)[:n_questions]

    agree_match = winner_match = 0
    mismatches = []
    pooled_max_dev = 0.0
    pooled_conflicts = 0

    for qid in qids:
        samples = by_q[qid]
        assert [s["model"] for s in samples] == AGENTS, \
            f"unexpected model order on q{qid}"

        exp_winner, exp_share = experiment_v(samples)
        eng = engine_agreement(samples)

        a_ok = abs(eng["agreement"] - exp_share) < 1e-9
        w_ok = eng["winner"] == exp_winner
        agree_match += a_ok
        winner_match += w_ok
        if not (a_ok and w_ok):
            mismatches.append((qid, exp_winner, exp_share, eng))

        # Stage-2 algebra property through the real path.
        confs = [s["parsed_confidence"] for s in samples]
        arith = sum(confs) / len(confs)
        pooled_mean, violations = engine_pooled_mean(samples)
        if pooled_mean is None:
            pooled_conflicts += 1
        else:
            pooled_max_dev = max(pooled_max_dev, abs(pooled_mean - arith))

    print(f"questions replayed: {len(qids)}")
    print(f"agreement match rate: {agree_match}/{len(qids)}")
    print(f"winner match rate:    {winner_match}/{len(qids)}")
    print(f"mismatches: {len(mismatches)}")
    for qid, exp_winner, exp_share, eng in mismatches:
        print(f"  q{qid}: experiment winner={exp_winner!r} share={exp_share:.4f} "
              f"engine winner={eng['winner']!r} agreement={eng['agreement']:.4f}")
    print(f"pooled: max |pooled_mean - arithmetic_mean| = {pooled_max_dev:.2e}")
    print(f"pooled: K-conflict guard violations on {pooled_conflicts}/{len(qids)}")


if __name__ == "__main__":
    main()
