# Pooling redesign: options for the `combine_pseudocount` saturation defect

Status: DECIDED 2026-10-03. Inan chose option (c): fix the algebra (a),
add agreement-based resolve (b), with opt-in calibration. Implemented
on `feat/cir-agreement-resolve` (Stages 1-4), docs in Stage 5.

## The defect

`BetaDist.combine_pseudocount` (`chimera/cir/nodes.py:50`) merged two
beliefs as `(alpha + alpha' - 1, beta + beta' - 1)` and clamped each at
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
(now passing).

## As built (option (c), chosen)

(a) The belief algebra is pure pseudocount addition:
`combine(a, b) = Beta(a.alpha + b.alpha, a.beta + b.beta)` after the
retained K conflict check. No subtract-one, no clamp. Deviation from
the note's literal `Beta(1 + sum alpha, 1 + sum beta)`: no `Beta(1,1)`
prior is added per combination, because it would accumulate `k - 1`
spurious priors over `k` chained sources and break the property the
task's Stage 4 verifies (pooled mean equals the arithmetic mean for
equal strengths; confirmed through `run_cir` within 6.67e-07, the
residual being the documented `1e-6` clip in `from_confidence` on
`1.0` inputs). Positivity, and hence clamp-unreachability, holds
without the `+1`: every `BetaDist` in the system has strictly positive
parameters and sums of positives are positive. The pooled mean is a
strength-weighted average of the input means, so it lies within
`[min, max]` of them; `k` identical sources at `c < 1` pool to exactly
`c`. `combine_ds` stays as an alias. The same old formula existed
inline in `_exec_inquiry` for seeded priors; it now delegates to
`combine_pseudocount`, and a conflicting seeded prior is a guard
violation, not a silent merge.

(b) `resolve` gained the `agreement` strategy (now the default when a
belief has more than one source with answers): vote-share over
normalized answers (`chimera/cir/agreement.py`), same normalization as
the H1 experiment, winner is the most common normalized answer with
ties broken by earliest source in agent order, comparator and
normalizer pluggable. Deviation from the note's sketch: the posterior
is Laplace-smoothed `Beta(1 + votes, 1 + (n - votes))` rather than a
fixed strength-10 Beta, so unanimous 3/3 yields mean 0.8, never 1.0.
Replay validation against the H1 data (first 100 mode-A questions
through `run_cir`): engine agreement matches the experiment V arm
100/100, engine winner matches 100/100, zero mismatches
(`experiments/h1_pooling/validate_agreement_replay.py`). `pooled`
keeps the (a) algebra; `dempster_shafer` is an alias of `pooled` with
a lowering warning. Single-source resolve passes through unchanged.

Calibration is opt-in (`chimera/cir/calibration.py`:
`LogisticCalibrator`, deterministic pure-Python fit, JSON carrying n,
dataset hash, fit date, refuses n < `MIN_FIT_N = 30`). `run_cir`
accepts `calibrator=...`; the CLI accepts `--calibrator=PATH`.
`calibrated_p` appears only when a calibrator is supplied; guard and
emit use it when present, else the uncalibrated posterior with a
lowering warning.

What the H1 data actually says, stated plainly: verbalized confidence
was near chance in mode B (AUROC 0.51 to 0.54); answer agreement
carried the signal. Calibration is opt-in and only meaningful with a
held-out calibration set: it rescales scores, it does not create
ranking signal (the H1 v2 finding was Platt restoring ECE to ~0.04
with mode-B AUROC still ~0.51).

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

## Decision (2026-10-03)

Inan chose option (c). Implemented as described in "As built" above.
The confirmatory experiment plan
(`experiments/h1_pooling/PREREG_V2_CONFIRM.md`) is being rewritten
against the redesigned engine; the old plan is superseded.
