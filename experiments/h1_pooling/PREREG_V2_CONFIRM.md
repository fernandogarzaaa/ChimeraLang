# PREREG V2 CONFIRM: confirmatory test of the redesigned CIR pooling engine

**Status: DRAFT. NOT FROZEN. DO NOT RUN until Inan approves.**
**Supersedes the earlier PREREG_V2_CONFIRM draft (written against the old engine).**

This is a confirmatory (not exploratory) experiment. Every number below is
fixed before any confirmatory data is collected. The confirmatory dataset
(questions 501-1000) has never been sent to any model. The calibrator is
frozen. Nothing here may be re-tuned after seeing confirmatory results.

## 1. Background in one paragraph

The H1 experiment (SimpleQA questions 1-500, 4,000 Nebius calls, 2026-10-02)
found: (a) verbalized confidence is near chance at ranking correctness
(mode-B AUROC 0.51 to 0.54); (b) multi-source pooling with the old Beta
algebra saturates: the pooled chain claims 1.0000 with 0.3308 precision on
the accepted set (overconfidence gap 0.6692, CI [0.6256, 0.7165]); (c)
majority-vote answer agreement carries the useful signal, and a calibrated
vote-share model restores ECE to about 0.04. The engine was then redesigned
(agreement resolve as default, Laplace-smoothed posterior, pure additive
Beta algebra for the pooled strategy, opt-in logistic calibration). This
preregistration tests the redesigned engine on fresh questions.

## 2. Engine under test (the redesigned path)

The confirmatory analysis runs every collected response through the real
engine path, not an offline reimplementation:

- Each question becomes a small CIR program with one agent per source
  (mode A: 3 model-agents; mode B: 5 sample-agents from one model).
- Sources carry their verbalized answer and confidence.
- Resolution uses the default strategy: `agreement`.
  Normalization is the H1 normalization (lowercase, strip punctuation,
  strip leading articles, collapse whitespace). Most-common normalized
  answer wins; ties break to the earliest source. The resolved belief
  carries raw agreement and the Laplace posterior
  Beta(1 + winner_votes, 1 + nonwinner_votes).
- The `pooled` strategy (pure additive Beta algebra) is exercised as a
  secondary arm to confirm the saturation defect is gone.
- Calibration: the frozen calibrator (section 5) is applied unchanged.
  `calibrated_p` is the only probability reported for decisions.
  The uncalibrated posterior mean is recorded for comparison only.

## 3. Confirmatory dataset (frozen)

- File: `experiments/h1_pooling/datasets/simpleqa_501_1000.jsonl`
- Contents: SimpleQA questions 501-1000 (500 records), ids
  `simpleqa-0501` through `simpleqa-1000`.
- SHA-256: `a170802b135c3a3d1409b5bdbe3e063a80ea213dc18e267f35c93d12bded5119`
- Provenance: rows 500-999 (0-indexed) of the public `basicv8vc/SimpleQA`
  mirror CSV (`simple_qa_test_set.csv`, 4,326 rows), which was verified to
  reproduce our questions 1-500 exactly (row 0 = simpleqa-0001,
  row 499 = simpleqa-0500, same questions and gold answers).
- Disjoint from the exploratory set by construction. No question in this
  file has been sent to any model by us.

## 4. Collection protocol (identical to H1)

Same models, same prompts, same temperatures, same modes, same parse and
grading code as the 2026-10-02 run. The only change is the question file.

- Mode B (self-consistency): Qwen3-235B-A22B-FP8-T, temperature 1.0,
  5 samples per question, 500 questions = **2,500 calls**.
- Mode A (cross-model): Qwen3-235B-A22B-FP8-T, DeepSeek-V3.1-Terminus,
  MiniMax-M2, temperature 0.0, 1 sample each per question,
  500 questions x 3 models = **1,500 calls**.
- Total: **4,000 calls**.
- Correctness grading: `normalize_answer` exact match against gold answers,
  the same function used in H1 and in the engine.

Estimated cost: about **$0.15** (the H1 run with the identical call mix
cost about $0.1515).

## 5. Frozen calibrator (fit on questions 1-500 only)

- File: `experiments/h1_pooling/calibrators/agreement_v1.json`
- SHA-256 of the JSON file:
  `8f269957df19d9d812f7fbc1e6ebeaef6a608b71e90533239138031549e1391f`
- Fit: logistic regression (deterministic Newton, pure Python) mapping the
  engine's Laplace posterior mean to P(winner correct).
- Training data: all 1,000 (posterior mean, correctness) pairs from
  questions 1-500 replayed through the redesigned engine
  (500 mode-A + 500 mode-B). n = 1000, a = 4.2315, b = -3.2915,
  fit date 2026-10-03, training-pair hash
  `52e7cb9f54808dfc77bebb5b634f726e1b1ab6798d4b7040` (embedded in the JSON).
- This file is applied UNCHANGED to questions 501-1000. No refit, no
  per-mode variants, no threshold tuning on confirmatory data.
- Rationale for one shared calibrator: the engine exposes a single
  `calibrator=` slot; the preregistered claim is about that slot.
  Per-mode calibrators are explicitly out of scope for this round.

## 6. Analysis plan (fixed)

For each mode separately, on questions 501-1000:

1. Replay every question through `run_cir` with the agreement strategy
   and the frozen calibrator (replay adapter, same as the validation
   script used on questions 1-500).
2. Primary metric: AUROC of the agreement score (Laplace posterior mean)
   ranking winner-correctness, with a 2,000-replicate bootstrap 95% CI
   (seed 20261002, same code as the derivation script).
3. Calibration metric: Brier(frozen calibrated_p) vs Brier(constant
   base-rate predictor), difference with 2,000-replicate bootstrap 95% CI.
4. Secondary: pooled-strategy arm, checking the pooled mean equals the
   arithmetic mean of source means (tolerance 1e-4, accounting for the
   documented 1e-6 from_confidence clip) and that no pooled belief
   saturates at 1.0.

## 7. Decision rules (fixed numeric thresholds, derived from questions 1-500)

Reference statistics from questions 1-500 (engine replay, derivation script
`experiments/h1_pooling/derive_confirm_thresholds.py`):

| mode | agreement AUROC [95% CI] | const-minus-calibrated Brier [95% CI] |
|------|--------------------------|---------------------------------------|
| A    | 0.7168 [0.6734, 0.7597]  | 0.0341 [0.0195, 0.0502]               |
| B    | 0.7647 [0.7279, 0.8002]  | 0.0421 [0.0271, 0.0560]               |

(For context: the UNCALIBRATED posterior-mean Brier was 0.2134 in mode A
vs constant 0.2183, and 0.2980 in mode B vs constant 0.2176. In mode B
the uncalibrated score is worse than the constant predictor; calibration
is doing real work, not decoration.)

CONFIRM rules (all must hold on questions 501-1000):

- **C1 (signal replicates):** agreement AUROC point estimate >= 0.65 in
  BOTH modes, AND the 95% CI lower bound > 0.50 in both modes.
  (0.65 sits about 1.5 standard errors below the 1-500 point estimates
  and 0.15 above chance: a real-signal bar, not a point-replication bar.)
- **C2 (calibration helps):** Brier(frozen calibrated_p) < Brier(constant
  predictor) as a point estimate in BOTH modes.

FALSIFY rules (any one fails the redesign; reported as prominently as
any confirmation):

- **F1:** the 95% CI lower bound of agreement AUROC is <= 0.60 in EITHER
  mode. (0.60 is below both 1-500 lower bounds, 0.6734 and 0.7279, so
  hitting it means the agreement signal did not replicate.)
- **F2:** the 95% CI of [Brier(calibrated) - Brier(constant)] excludes zero
  AND is positive in EITHER mode (the frozen calibrator is significantly
  worse than predicting the base rate).

Verdict logic: CONFIRM requires C1 and C2 with neither F1 nor F2
triggered. Any falsify rule triggered means the redesign fails this
confirmatory round, full stop, regardless of the confirm rules.

## 8. Call count and cost (exact)

- Mode B: 2,500 calls. Mode A: 1,500 calls. Total: **4,000 calls**.
- Estimated cost: **about $0.15** (same models and mix as H1: ~$0.1515).
- No other paid calls are part of this plan.

## 9. Freeze and run procedure (NOT YET DONE)

1. Inan approves this plan (explicit go).
2. Freeze: record the git SHA of this file, the dataset, and the
   calibrator in the run manifest. No edits after this point.
3. Run collection on questions 501-1000 only.
4. Run the fixed analysis. Report C1, C2, F1, F2 exactly as specified,
   with the contradictory outcomes (if any) as prominent as the
   supporting ones.
5. Publish the confirmatory report as a new PR (do not merge).
