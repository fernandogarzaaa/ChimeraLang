# Erratum to the H1 run report (2026-10-02 Nebius run)

This file corrects three statements in `REPORT.md`. `REPORT.md` itself
is left unedited. All numbers below are from the committed
`summary.json`, reproduced byte-identically by re-running `analyze.py`
on `responses.jsonl` (SHA-256 below).

## 1. Arm P did not test the shipped pooling strength

`REPORT.md` calls arm P "the shipped pooling" and states "no deviation
from PREREG.md". The prereg fixed `POOL_STRENGTH = 2.0`
(`analyze.py:36`, "weakest symmetric prior carrying the claim"), so
there was no deviation from the prereg. The error is the "shipped"
label: the shipped engine converts confidences with
`BetaDist.from_confidence(conf)` at its default `strength=10.0`
(`chimera/cir/nodes.py:41`), and `_exec_inquiry` calls it without
arguments (`chimera/cir/executor.py:240`). The experiment tested a
weaker-prior variant that clamps far harder than the shipped default.

Mechanism, verified against `chimera/cir/nodes.py`
(`combine_pseudocount` adds pseudocounts as
`(alpha + alpha' - 1, beta + beta' - 1)` and clamps each at `1e-6`):

- Five samples at confidence 0.7, strength 2.0: pooled mean **1.0000**.
  Raw beta before clamping per combine step: 0.2, -0.2, -0.4, -0.4.
  The subtract-one drives beta negative and the clamp pins it at
  1e-6, so the mean saturates at 1.0.
- Same five samples at strength 10.0: pooled mean **0.7381**. Raw
  betas stay positive (5.0, 7.0, 9.0, 11.0); no clamping.

Consequence: the reported H1a gap (0.6692, arm P claiming 1.0000 on
the accepted set) is partly an artifact of the strength-2.0 clamp,
not a pure measurement of the shipped engine. The qualitative
finding (overconfidence under correlation) still needs re-testing at
strength 10; see `PREREG_V2.md`.

## 2. Arm C's calibration gain is base-rate prediction, not signal

`REPORT.md` presents arm C (Platt recalibration) as evidence that
"the signal is there". The numbers do not support that reading:

- Mode B arm C AUROC: **0.5095**, 95% CI **[0.4739, 0.5485]**, which
  includes chance (0.5). The scaler has no discriminative power.
- Arm C scores on the 400 eval questions: mean **0.2900**, stdev
  **0.0520** (min 0.0003, max 0.4065), versus the eval base rate
  (majority-vote accuracy) of **0.3275**.
- Arm C ECE is 0.0392 because it predicts near the base rate with
  almost no variation, not because it ranks answers.

Recalibration fixes the ECE number while carrying no signal about
which answers are right.

## 3. The dominant failure is verbalized confidence itself

Pooling is not the main problem; the inputs to pooling are:

- Mode B arm S (single call): Brier **0.6310**, ECE **0.6483**.
- AUROC: S **0.5095**, M **0.5398**, P **0.5399**. No
  verbalized-confidence arm discriminates above chance in mode B.
- Pooling adds on top: P Brier **0.6671** vs S **0.6310** (plus
  0.0361) and vs M **0.6023** (plus 0.0648).
- The 0.80 guard barely filters: **396 of 400** eval questions pass
  it, so accepted-set precision (**0.3308**) equals overall
  majority-vote accuracy (**0.3275**) to three decimals. The guard
  threshold is non-binding at these confidence levels.

## Data manifest addition

- `responses.jsonl`: 4,000 rows, 4,318,677 bytes (under 5 MB).
- SHA-256:
  `7f238bd0335b073f8218b860ffacbc339f1c7a2bdc6f7f7d98104154a9cc14aa`
- Credential grep before commit: one hit for the pattern `secret`,
  verified to be the English word "secrets" inside a model answer
  ("uncover secrets about his past"); no `hsurr` values, no API
  keys, no auth headers, no `sk-` keys. Committed in this erratum's
  commit.
