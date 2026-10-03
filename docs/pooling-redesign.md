# Pooling redesign: options for the `combine_pseudocount` saturation defect

Status: investigation only. No fix implemented. Awaiting Inan's choice.

## The defect

`BetaDist.combine_pseudocount` (`chimera/cir/nodes.py:50`) merges two
beliefs as `(alpha + alpha' - 1, beta + beta' - 1)` and clamps each at
`1e-6`. Whenever `k` sources at confidence `c` and strength `s`
satisfy `k * (1 - c) * s < (k - 1)`, the raw beta goes to zero or
negative and the clamp pins it, so the pooled mean saturates at 1.0.
Example, all at the shipped default strength 10
(`chimera/cir/nodes.py:41`, used by `_exec_inquiry` at
`chimera/cir/executor.py:240`): five sources at 0.95 give raw betas
0.0, -0.5, -0.5, -0.5 and a pooled `Beta(43.5, 1e-6)` with mean
0.999999977. The H1 experiment measured the consequence on real
model outputs: on the accepted set (pooled mean >= 0.80) the chain
claims 1.0000 with 0.3308 precision (mode B, N=500 SimpleQA), and
loses to a plain mean on Brier in both modes.

Red test: `tests/test_cir_nodes.py::TestBetaDist::test_combine_pseudocount_no_saturation_clamp`
(currently failing).

## Option (a): evidence counts on Beta(1,1)

Represent each source as evidence counts added to a `Beta(1,1)`
prior: pooled = `Beta(1 + sum(alpha_evidence), 1 + sum(beta_evidence))`
where each source contributes its `(alpha, beta)` pseudocounts
without the subtract-one. The posterior can never clamp because both
parameters stay >= 1.

Public behavior changes:
- `BetaDist.combine_pseudocount` (and its alias `combine_ds`) return
  different distributions: for agreeing sources the mean moves less
  extremely toward 1.0/0.0, and the variance is larger (more
  pseudocounts retained). Any downstream threshold on
  `combined.mean` (e.g. `_exec_consensus` guard at
  `chimera/cir/executor.py:298-310`) fires differently.
- The `ValueError` conflict path is unchanged (still raised on high
  K), but K itself is computed from the same means, so conflict
  decisions are unchanged.
- `BeliefState.observe` callers that relied on saturation (none
  found in tests besides the new red test) would see lower
  posteriors.

## Option (b): vote-share agreement for resolve pooling

Replace the Beta pooling in `_exec_consensus`
(`chimera/cir/executor.py:275`) with vote-share agreement over
normalized answers: the consensus belief's mean becomes the fraction
of input beliefs whose normalized answer matches the majority
answer, with a fixed modest strength (e.g. 10). This matches the H1
finding that vote-share (arm V) was the best-calibrated arm in both
modes (Brier 0.3558/0.2040, AUROC 0.76/0.72).

Public behavior changes:
- `resolve` no longer uses verbalized confidences at all for the
  pooled mean; confidence only breaks ties. Programs that set
  per-agent confidences will see those numbers ignored at resolve
  time (still recorded in traces and per-agent beliefs).
- `ConsensusNode` outputs become coarser (multiples of 1/k), which
  changes guard thresholds tuned to fine-grained pooled means.
- `combine_pseudocount` itself stays as-is for other callers
  (`_exec_inquiry` observation combining at `executor.py:240-250`
  keeps current behavior, defect included).

## Option (c): both, with calibration only when a calibration set exists

Do (a) for the belief algebra and (b) for resolve pooling, and add an
optional calibration step: when the program provides labeled examples
(new `calibrate` source or a `calibration=` argument), fit a
per-deployment Platt scaler and apply it before pooling; without
labeled data, no calibration is applied and the trace says so.

Public behavior changes:
- Union of (a) and (b).
- New opt-in surface for calibration data (shape TBD with Inan);
  default behavior without it is (a)+(b) with a trace note that no
  calibration was applied. No silent behavior change for programs
  that provide no labels.
- The H1 v2 finding (Platt restores ECE to ~0.04 but carries no
  ranking signal, mode B AUROC 0.51) must be documented next to the
  calibration option so users do not mistake calibration for signal.

## Recommendation (not a decision)

Option (c) keeps the algebra sound, adopts the empirically best
pooling, and makes calibration explicit rather than magical. But it
is the largest change. Waiting on Inan's choice before touching
`nodes.py` or `executor.py`.
