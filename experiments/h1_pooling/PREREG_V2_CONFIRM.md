# PREREG V2 CONFIRM: confirmatory test of the redesigned CIR pooling engine

**Status: DRAFT. NOT FROZEN. DO NOT RUN until Inan approves.**
**Supersedes the earlier PREREG_V2_CONFIRM draft (written against the old engine).**

This is a confirmatory (not exploratory) experiment. Every number below is
fixed before any confirmatory data is collected. The confirmatory dataset
(questions 501-1000) has never been sent to any model. Both calibrators
are frozen. Nothing here may be re-tuned after seeing confirmatory
results.

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
- Calibration: one frozen calibrator per source configuration
  (section 4) is applied unchanged. Mode-A questions use the mode-A
  calibrator; mode-B questions use the mode-B calibrator. `calibrated_p`
  is the only probability reported for decisions. The uncalibrated
  posterior mean is recorded for comparison only.
- The fixed analysis is `experiments/h1_pooling/analyze_confirm.py`
  (SHA-256 `8655f0ce5935c2fe1acfad49e23d9f671241ebe33acca9dbca5d1375d1652ca5`).
  It verifies the dataset, calibrator, and model-id table hashes at run
  time and refuses to run on any mismatch, including a model id that
  differs from the hash-checked table in section 6.

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

## 4. Frozen calibrators (one per source configuration)

Fit ONLY on questions 1-500 replayed through the redesigned engine.
Each JSON records its source configuration in `metadata`.

- Mode A (3 sources, cross-model):
  `experiments/h1_pooling/calibrators/agreement_v1_modeA.json`
  SHA-256: `47684a2ac34537f0d4f1ae2b9eb87dbf59b07977c51a6cbf803800a996318bbf`
  n = 500, a = 7.4950, b = -4.4424,
  metadata `{"n_sources": 3, "source_type": "cross-model"}`.
- Mode B (5 sources, single-model self-consistency):
  `experiments/h1_pooling/calibrators/agreement_v1_modeB.json`
  SHA-256: `8ac4b2869a70093088102f8467d462f936e6ffe0375dcbde4f4d0ef35ce3ffa8`
  n = 500, a = 6.8160, b = -5.6773,
  metadata `{"n_sources": 5, "source_type": "single-model"}`.

Rationale for per-configuration calibrators: the vote dynamics of 3
cross-model sources and 5 single-model samples differ, and the engine
exposes one calibrator slot per resolve call, so each configuration
gets its own frozen mapping. No refit, no threshold tuning on
confirmatory data. A single shared calibrator is explicitly out of
scope for this round.

## 5. Smoke-test gate (10 questions per model)

Before the full run, 10 questions per model are collected and verified:

- Mode A: 3 models x 10 questions = 30 calls.
- Mode B: 1 model x 10 questions x 5 samples = 50 calls.
- 80 calls total, on questions simpleqa-0501 through simpleqa-0510.
- These 80 smoke responses are REUSED in the full run: they count
  toward the 4,000 total (section 9), and the full run collects only
  the remaining 3,920 calls on questions 0511-1000. No question is
  collected twice.
- Verification: `experiments/h1_pooling/smoke_test.py`
  (SHA-256 `ee19e031a64320c3f87eb8bda4a147868403b89dd78feb3eb3cdb55415da3197`).
  It checks exactly 10 questions per model per mode, 100 percent parse_ok,
  that every collected model_id matches the hash-checked model-id table
  in section 6 (SMOKE: FAIL on any mismatch), and a clean engine replay
  (agreement resolve) for every question.
- Correctness is deliberately NOT inspected at the smoke gate: no
  grading against gold answers happens there, so no confirmatory
  outcome can leak into the collection decision. The gate checks
  pipeline health only (collection, parsing, engine replay).
- It writes a run manifest recording the requested model ids, the
  provider-returned model id per row (`provider_model_id`), the
  model-id table SHA-256, and timestamps (`started_at`, `ended_at`),
  plus per-model question counts and parse_ok rates.
- The full run proceeds ONLY on `SMOKE: PASS`. Any failure
  stops the run before further spend.

## 6. Collection protocol (identical to H1)

Same models, same prompts, same temperatures, same modes, same parse and
grading code as the 2026-10-02 run. The only change is the question file.
Model ids below are the exact Nebius Token Factory ids from
`collect.py` `DEFAULT_NEBIUS_MODELS`, and match the H1 run's collected
rows verbatim (short agent names: qwen3-235b, deepseek-v4pro, gemma-3-27b).

- Mode B (self-consistency): `Qwen/Qwen3-235B-A22B-Instruct-2507`,
  temperature 1.0, 5 samples per question, 500 questions =
  **2,500 calls**.
- Mode A (cross-model): `Qwen/Qwen3-235B-A22B-Instruct-2507`,
  `deepseek-ai/DeepSeek-V4-Pro`, `google/gemma-3-27b-it`, temperature 0.0,
  1 sample each per question, 500 questions x 3 models = **1,500 calls**.
- Correctness grading: `normalize_answer` exact match against gold answers,
  the same function used in H1 and in the engine.
- Every collected row records both the requested `model_id` and the
  provider-returned model id (`provider_model_id`, from the Nebius API
  response's `model` field), so the smoke manifest and analysis can show
  exactly which model string the provider served for each call.

Model-id table (hash-checked). `experiments/h1_pooling/model_ids.json`
(SHA-256 `e5d9b8e330a024acd1463856d0230d57edb3cfc19a9aaf9f412b4076c6320a6b`):

| mode   | expected model_id(s)                                                                                                  |
|--------|-----------------------------------------------------------------------------------------------------------------------|
| mode_A | `Qwen/Qwen3-235B-A22B-Instruct-2507`, `deepseek-ai/DeepSeek-V4-Pro`, `google/gemma-3-27b-it`                           |
| mode_B | `Qwen/Qwen3-235B-A22B-Instruct-2507`                                                                                  |

Both `smoke_test.py` and `analyze_confirm.py` verify this table's
SHA-256 at run time and refuse if any collected row's `model_id` differs
from the table for its mode. The table may not be edited after freeze;
any model change would require a new preregistration.

## 7. Analysis plan (fixed)

`analyze_confirm.py` is the only analysis that counts. For each mode
separately, on questions 501-1000:

1. Verify SHA-256 of the dataset, both calibrators, and the model-id
   table against the hashes in sections 3, 4, and 6; refuse (exit 2)
   on any mismatch. Also refuse if a calibrator's `metadata.n_sources`
   does not match its mode, if any question has the wrong source count,
   or if any response row's `model_id` is not in the model-id table for
   its mode.
2. Replay every question through the engine's agreement resolve, applying
   the mode's frozen calibrator.
3. Primary metric: AUROC of the agreement score (Laplace posterior mean)
   ranking winner-correctness, with a 2,000-replicate bootstrap 95% CI
   (seed 20261002).
4. Calibration metric: Brier(frozen calibrated_p) vs Brier(constant
   base-rate predictor), with a 2,000-replicate bootstrap 95% CI on the
   difference (constant minus calibrated).
5. Secondary: pooled-strategy arm, checking the pooled mean equals the
   arithmetic mean of source means (tolerance 1e-4, accounting for the
   documented 1e-6 from_confidence clip) and that no pooled belief
   saturates at 1.0. Reported as PASS/FAIL; it does not change the
   verdict.

## 8. Decision rules and outcomes (fixed numeric thresholds)

Reference statistics from questions 1-500 (engine replay; derivation
script `experiments/h1_pooling/derive_confirm_thresholds.py`):

| mode | agreement AUROC [95% CI] | const-minus-calibrated Brier [95% CI] |
|------|--------------------------|---------------------------------------|
| A    | 0.7168 [0.6734, 0.7597]  | 0.0356 (in-sample; cross-fit 0.0341 [0.0195, 0.0502]) |
| B    | 0.7647 [0.7279, 0.8002]  | 0.0430 (in-sample; cross-fit 0.0421 [0.0271, 0.0560]) |

(In-sample uses the actual frozen per-mode calibrators; cross-fit is the
unbiased estimate. They agree closely: the two-parameter fit on 500
points barely overfits. For context, the UNCALIBRATED posterior-mean
Brier was 0.2134 in mode A vs constant 0.2183, and 0.2980 in mode B vs
constant 0.2176. In mode B the uncalibrated score is worse than the
constant predictor; calibration is doing real work, not decoration.)

Rules (all evaluated on questions 501-1000 by `analyze_confirm.py`):

- **C1 (signal replicates):** agreement AUROC point estimate >= 0.65 in
  BOTH modes, AND the 95% CI lower bound > 0.50 in both modes.
  (0.65 sits about 1.5 standard errors below the 1-500 point estimates
  and 0.15 above chance: a real-signal bar, not a point-replication bar.)
- **C2 (calibration helps):** the 95% CI of
  [Brier(constant) - Brier(calibrated)] lies entirely above zero
  (excludes zero) in BOTH modes.
- **F1 (signal did not replicate):** the 95% CI lower bound of agreement
  AUROC is <= 0.60 in EITHER mode. (0.60 is below both 1-500 lower
  bounds, 0.6734 and 0.7279.)
- **F2 (calibration hurts):** the 95% CI of
  [Brier(calibrated) - Brier(constant)] lies entirely above zero
  (excludes zero) in EITHER mode.

Outcomes:

- **Confirmed:** C1 and C2 hold, and neither F1 nor F2 is triggered.
- **Falsified:** F1 or F2 is triggered, full stop, regardless of the
  confirm rules. Reported as prominently as any confirmation.
- **Inconclusive:** anything else (for example C1 holds but C2's CI
  includes zero, or AUROC lands between the C1 bar and the F1 bar).
  An inconclusive outcome is reported as-is; it does not license
  re-tuning and re-running under this preregistration.

## 9. Call count and cost (exact)

- Smoke test: 80 calls (mode A 30 + mode B 50) on questions 0501-0510.
  These 80 responses are reused in the full run, not added on top.
- Confirmatory run: **4,000 calls total** (mode B 2,500 + mode A 1,500),
  of which 80 are the smoke responses and 3,920 are new calls on
  questions 0511-1000.
- Total: **4,000 calls** (unchanged from H1).
- Estimated cost: **about $0.15** (same models and mix as H1: ~$0.1515
  for 4,000 calls).
- No other paid calls are part of this plan.

## 10. Freeze and run procedure (NOT YET DONE)

1. Inan approves this plan (explicit go).
2. Freeze: record the git SHA of this file and the SHA-256 hashes of the
   dataset (section 3), both calibrators (section 4), `analyze_confirm.py`
   (section 2), `smoke_test.py` (section 5), and the model-id table
   `model_ids.json` (section 6) in the run manifest.
   No edits after this point.
3. Run the smoke test (section 5). Proceed only on SMOKE: PASS.
4. Run collection on questions 501-1000 only, reusing the 80 smoke
   responses from questions 0501-0510 (no re-collection of those).
5. Run `analyze_confirm.py` with the frozen hashes. Report the verdict
   (Confirmed / Falsified / Inconclusive) and every rule outcome exactly
   as specified, with contradictory outcomes (if any) as prominent as
   the supporting ones.
6. Publish the confirmatory report as a new PR (do not merge).
