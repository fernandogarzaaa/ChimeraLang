# H1 pooling experiment: PREREG_V2 (exploratory follow-up)

Status: EXPLORATORY. The 2026-10-02 Nebius data has been seen
(`runs/2026-10-02-nebius/responses.jsonl`), so nothing in this
document is confirmatory. It fixes the v2 arms, metrics, and
falsifiable expectations in advance of running `analyze_v2.py`, so
the follow-up analysis cannot be reshaped after seeing its own
output. Corrections to the v1 report are in
`runs/2026-10-02-nebius/ERRATUM.md`; `analyze.py`, `PREREG.md`, and
`REPORT.md` are not modified.

## Motivation

ERRATUM.md established three facts the v1 analysis mishandled: arm P
used strength 2.0 while the shipped default is 10.0
(`chimera/cir/nodes.py:41`, `chimera/cir/executor.py:240`); arm C's
ECE gain is base-rate prediction (mode B AUROC 0.5095, CI includes
chance); and verbalized confidence itself is the dominant failure
(mode B arm S Brier 0.6310, ECE 0.6483). V2 re-tests the pooling
claim at the shipped strength, adds honest calibration via
cross-fitting, and adds agreement-based arms that do not rely on
verbalized confidence.

## Arms (per question, per mode)

Predicted answer for every arm except K is the majority-vote winner
by exact normalized match (ties broken by earliest sample), as in v1.
Correctness `y` is correctness of that predicted answer against any
gold alias (exact normalized match), as in v1.

- **K (constant):** predicts the training-fold base rate (mean `y` of
  the majority-vote answer over training questions). The no-signal
  baseline every other arm must beat.
- **P2:** original v1 arm P, `combine_pseudocount` chain at strength
  2.0. Kept for comparison only.
- **P10:** the same chain at strength 10.0, the shipped default.
  Each confidence `c` becomes `Beta(10c, 10(1-c))` via
  `BetaDist.from_confidence`; combined in sample order with the
  shipped `combine_pseudocount`. Claimed p = pooled mean. On a
  `ValueError` conflict, falls back to the arithmetic mean (counted).
- **S:** first sample's answer and confidence (unchanged from v1).
- **M:** arithmetic mean of sample confidences (unchanged from v1).
- **V:** vote share for the majority answer (unchanged from v1).
- **Vc:** vote share passed through a logistic regression
  (correctness ~ vote share), cross-fitted (see Calibration).
- **VC:** logistic regression on two features, vote share and mean
  sample confidence, cross-fitted.
- **V-loose (mode A only):** agreement fraction where a sample counts
  as agreeing when token-F1(sample answer, majority answer) >= 0.8
  instead of exact normalized match. Same predicted answer as V.

## Calibration

5-fold cross-fitting over all 500 questions with fixed seed
`20261002`: split questions into 5 folds (seeded shuffle), fit each
logistic model (Vc, VC) on 4 folds, predict the held-out fold. Every
question gets an out-of-fold calibrated score; no question is scored
by a model fit on itself. K uses each training fold's own base rate.
For comparability, the original v1 20/80 split (first 100 questions
by dataset order as calibration, rest as eval) is also reported for
all arms.

## Metrics (95 percent question-level bootstrap CIs)

2000 resamples of questions with replacement, fixed seed 20261002,
percentile (2.5/97.5) intervals, computed on out-of-fold scores:

- **Brier**, **ECE** (10 equal-width bins), **AUROC**, as in v1.
- **Risk-coverage:** sort questions by claimed confidence descending;
  accuracy among the top 20, 40, 60, 80, 100 percent (coverage).
  Ties broken by question id (deterministic).
- **AURC:** area under the risk-coverage curve, risk = 1 - accuracy,
  trapezoidal rule over the per-question steps (lower is better).
- **Guard simulation:** for each arm, coverage (fraction accepted)
  and precision (accuracy among accepted) of the accepted set at
  thresholds 0.50, 0.55, ..., 0.95.

## Grading audit

Print 40 randomly chosen graded items (seed `20261003`): question
id, gold aliases, predicted answer, correctness under the exact
normalized-match grader. Report accuracy under a looser grader
(token-F1 >= 0.8 against any alias) as a sensitivity analysis, for
all arms, both modes. Token-F1: tokens are the normalized answer
split on whitespace; F1 = 2PR/(P+R) with P and R over token
multiset overlap; take the max over gold aliases.

## Stated expectations (falsifiable)

1. P10 still saturates for high confidences: among questions where
   every sample has confidence >= 0.9, the median P10 pooled mean is
   >= 0.99.
2. In mode A, Vc and VC beat K on Brier (point estimates).
3. In mode B, no verbalized-confidence arm (S, M, P2, P10) has AUROC
   above 0.6.

`V2_RESULTS.md` will list each expectation as met or not met, with
the numbers, including any that fail.

## What v2 does not do

No confirmatory decisions. No model calls. No changes to
`analyze.py`, `PREREG.md`, or `REPORT.md`. A confirmatory
replication on fresh questions is proposed separately (Stage 5) and
is not run here.
