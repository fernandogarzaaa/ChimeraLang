# H1 Pooling Experiment: Pre-registration

Status: PRE-REGISTERED before any real model run. 2026-10-02.
Code: `experiments/h1_pooling/` (collect.py, analyze.py).
This document fixes hypotheses, arms, metrics, and decision rules in
advance. Any deviation in the run report must be labeled as such.

## Hypotheses

**H1 (overconfidence under correlation):** Pooling verbalized
confidences with the shipped `combine_pseudocount` chain
(`chimera/cir/belief.py: BeliefState.observe` followed by
`combine_pseudocount`) is overconfident when the pooled sources are
correlated, and adds nothing over a plain mean of the confidences.

**H1a:** In mode B (k repeated samples from one model), the pooled
chain's claimed confidence on accepted answers exceeds its empirical
precision by a practically significant margin.

**H1b:** The pooled chain does not beat a plain mean of confidences on
Brier score; i.e., the belief algebra adds nothing over averaging.

Both outcomes (H1 supported / H1 not supported) are acceptable and
will be reported as observed.

## Arms (per question, per mode)

Each arm outputs one predicted answer and one claimed probability.

- **S (single):** the first sample's answer and confidence.
- **M (mean):** predicted answer = majority-vote winner (ties broken by
  earliest sample); claimed p = arithmetic mean of the samples'
  confidences.
- **P (shipped pooling):** predicted answer = majority-vote winner as
  in M. Each sample's confidence c becomes a Beta belief with mean c
  and strength 2.0, i.e. Beta(2c, 2(1-c)); the k beliefs are combined
  with the shipped `combine_pseudocount` in sample order; claimed p =
  pooled mean. (Strength 2.0 is the weakest symmetric prior that
  carries the verbalized claim; the combination order is fixed.)
- **V (majority-vote agreement):** predicted answer = majority-vote
  winner; claimed p = fraction of samples giving that answer.
- **C (recalibrated single):** predicted answer = first sample's answer;
  claimed p = Platt-scaled single-call confidence. The scaler is a
  logistic regression (correct ~ confidence) fit on the calibration
  split only (see Splits) and applied to the evaluation split.

## Modes

- **Mode B:** k=5 samples at temperature 1.0 from one model.
- **Mode A:** one sample (temperature 0) from each of M distinct models.

## Splits

Questions are used in dataset order. The first 20% form the
**calibration split** (used only to fit arm C's scaler); the remaining
80% form the **evaluation split**. All metrics below are computed on
the evaluation split.

## Correctness

Normalized exact match: lowercase, strip leading articles (a/an/the),
remove punctuation, collapse whitespace. A prediction is correct iff
its normalized form equals the normalized form of any gold alias.
No LLM judge.

## Metrics (per arm, per mode, on the evaluation split)

- **Brier:** mean((p - y)^2), y in {0,1} correctness of the arm's
  predicted answer.
- **ECE:** 10 equal-width bins over claimed p.
- **AUROC:** claimed p as the score for y.
- **Accepted-set precision and mean claimed confidence:** accept a
  question iff arm P's pooled mean >= 0.80; precision = fraction of
  accepted questions the arm answered correctly; mean claimed =
  mean of arm P's claimed p over accepted questions.

Each metric is reported with a 95% bootstrap confidence interval:
2000 resamples of questions with replacement, fixed seed 20261002,
percentile (2.5 / 97.5) intervals.

## Correlation diagnostic

Intraclass correlation ICC(1,1) (one-way random effects) of
logit(confidence) across the samples of each question, reported per
mode. High ICC in mode B confirms the repeated samples are
correlated, which is the condition H1 is about. (logit clipped to
[-6, 6] for numerical safety.)

## Decision rules

1. **H1a supported (mode B) iff** the overconfidence gap
   (mean claimed p on the accepted set minus accepted-set precision,
   arm P) exceeds 0.05 **and** its 95% bootstrap CI excludes zero.
2. **H1b evaluated iff** the Brier difference (Brier_M - Brier_P):
   if the 95% CI of the difference excludes zero in P's favor, the
   belief algebra beats averaging; otherwise report that the belief
   algebra adds nothing over averaging.

Secondary readouts (ECE, AUROC, mode A tables) are reported without
decision thresholds.

## Collection protocol

`collect.py` writes one JSON line per response to `responses.jsonl`
with: question id, prompt, model id, sample index, parsed confidence,
parsed answer, and SHA-256 of the raw response text. Collection is
resumable and idempotent: existing (question id, model id, sample
index) keys are skipped. `analyze.py` reads only `responses.jsonl`,
performs no network access, and is fully deterministic given the file.

Response format (fixed prompt): the model must reply with exactly
`Answer: <text>` and `Confidence: <0.00-1.00>` lines. Parsing is
case-insensitive on the labels. A sample whose confidence does not
parse as a float in [0, 1] is dropped; a question with fewer than 2
parseable samples (mode B) or from fewer than 2 models (mode A) is
excluded from all metric tables and counted in the exclusion table.

## Run manifest

The run report must record: dataset name and path, dataset SHA-256,
N, sampling method, model ids and versions, temperature settings,
prompt text hash, code commit, and the seed. Deviations from this
pre-registration must be labeled DEVIATION with a reason.
