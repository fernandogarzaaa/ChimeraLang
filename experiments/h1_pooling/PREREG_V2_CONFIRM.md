# Confirmatory replication plan (DRAFT, no execution)

Status: proposal only. Nothing here has run. Do not execute any paid
calls under this plan without Inan's explicit approval. This plan is
written before the run and must be frozen (any edits after data
collection starts are labeled DEVIATION).

## What it confirms

The v1 experiment (N=500 SimpleQA, `runs/2026-10-02-nebius/`) and the
v2 exploratory follow-up produced three claims worth confirming on
fresh data at the shipped pooling strength (10.0):

- H1a/v1: the shipped pooling chain is overconfident on correlated
  samples (v1 gap 0.6692 at strength 2.0; v2 E1 shows saturation
  persists at strength 10).
- H1b/v1: the shipped pooling adds nothing over a plain mean on
  Brier (v1 and v2 agree, both modes).
- E2/v2: cross-fitted vote-share calibration (Vc, VC) beats the
  constant base-rate predictor on Brier in mode A.

## Data

Fresh SimpleQA questions 501 to 1000 (0-indexed rows 500-999 of
`simple_qa_test_set.csv`; rows 0-499 were the v1 set). Same
`{id, question, gold_answers}` conversion script as v1
(`id` = `simpleqa-0501` ... `simpleqa-1000`).

## Protocol (identical to v1 except the question set)

- Same prompt (`h1_common.build_prompt`, SHA-256 `c66766f5e266...`).
- Same models: mode B, k=5 at temperature 1.0 from
  `Qwen/Qwen3-235B-A22B-Instruct-2507`; mode A, one sample at
  temperature 0 from each of `Qwen/Qwen3-235B-A22B-Instruct-2507`,
  `deepseek-ai/DeepSeek-V4-Pro`, `google/gemma-3-27b-it`.
- Same collection code (`collect.py`, nebius backend, `--workers 8`,
  resumable/idempotent) and same analysis code (`analyze.py` for the
  confirmatory rules, `analyze_v2.py` for the exploratory arms).
- `analyze.py` reads only `responses.jsonl`.

Exact call count: mode B 500 x 1 x 5 = 2,500; mode A 500 x 3 x 1 =
1,500; total 4,000 calls. Estimated cost ~$0.15 based on measured v1
usage and live Token Factory pricing (DeepSeek-V4-Pro dominates).

## Pre-registered decision rules

1. **C1 (H1a at shipped strength):** mode B, accepted set = P10
   pooled mean >= 0.80. Supported iff mean claimed minus precision >
   0.05 **and** the 95 percent bootstrap CI (2000 resamples, seed
   20261002) excludes zero.
2. **C2 (H1b at shipped strength):** Brier(M) minus Brier(P10) on
   mode B eval. If the 95 percent CI excludes zero in P10's favor,
   pooling beats averaging; otherwise report that pooling adds
   nothing over averaging.
3. **C3 (E2 confirmation):** mode A, Brier(K) minus Brier(Vc) > 0
   with 95 percent CI excluding zero, **and** Brier(K) minus
   Brier(VC) > 0 with 95 percent CI excluding zero (out-of-fold
   scores, 5-fold cross-fitting, seed 20261002).

All three rules use the exact normalized-match grader from v1; the
token-F1>=0.8 sensitivity is reported alongside but decides nothing.

## Open dependency

Stage 4's defect investigation is unresolved: if Inan chooses a
pooling redesign (options (a)/(b)/(c) in
`docs/pooling-redesign.md` on branch `defect/cir-pooling-clamp`),
this confirmatory run should test the *new* pooling, and this plan
must be revised before freezing. Do not run the confirmatory set
against a pooling implementation that is about to change.
