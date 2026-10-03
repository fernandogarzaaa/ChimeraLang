# PREREG_V2_CONFIRM confirmatory report

**Verdict: Inconclusive.** (Not confirmed, not falsified. Per the
preregistration, inconclusive means inconclusive: no re-tuning, no
re-run under this preregistration.)

Run date: 2026-10-03. Freeze tag: `confirm-v2-freeze` on commit
`802e3c080e9ce91b1ae454f57bac628f913681f5`. All frozen file hashes
verified at run time by `analyze_confirm.py` (dataset, both calibrators,
model-id table) before any metric was computed.

## Rule outcomes

Evaluated from the analyzer's printed metrics (see DEVIATION below for
why the script's own verdict line is absent; the rules were applied
exactly as frozen in `analyze_confirm.py` lines 241-253):

| rule | result | detail |
|------|--------|--------|
| C1 (signal replicates) | **True** | A: AUROC 0.6714 >= 0.65, CI lower 0.6280 > 0.50. B: AUROC 0.7519 >= 0.65, CI lower 0.7121 > 0.50 |
| C2 (calibration helps, CI excludes 0, both modes) | **False** | A: const-minus-calibrated 95% CI [-0.0007, 0.0300] includes zero. B: [0.0156, 0.0468] excludes zero |
| F1 (signal did not replicate) | False | A lower 0.6280 > 0.60; B lower 0.7121 > 0.60 |
| F2 (calibration hurts) | False | neither mode's [calibrated - constant] CI lies above zero |

Confirmed requires C1 and C2 with neither F1 nor F2: fails on C2.
Falsified requires F1 or F2: neither triggered. Therefore Inconclusive.

## The contradictory result, stated plainly

The agreement signal replicated cleanly in both modes (C1 holds with
margin: AUROC lower bounds 0.6280 and 0.7121, both well above chance and
above the 0.60 falsification bar). What failed is the calibration claim:
in mode A the calibrated predictor beat the constant predictor by
0.0145 Brier points, but the 95% CI [-0.0007, 0.0300] grazes zero, so C2
is not met. Mode B's calibration benefit is solid ([0.0156, 0.0468]).
The preregistration requires both modes, so the outcome is Inconclusive,
not a partial confirmation. The mode-A miss is by 0.0007 on the CI
lower bound; that narrowness does not change the verdict.

## DEVIATION: pooled-arm check crashed the frozen analyzer

`analyze_confirm.py` exited 1 before printing its verdict line. The
primary metrics above were computed and printed before the crash; only
the secondary pooled-arm check and the verdict print did not run.

Cause: on one confirmatory question, two sources gave irreconcilable
confidences (means 0.050 vs 0.900, K=0.860 > 0.80), and the engine's
retained K-conflict guard raised `ValueError` inside
`combine_pseudocount`, exactly as designed. The frozen script's pooled-arm
loop does not catch the guard, so it crashed:

```text
File "experiments/h1_pooling/analyze_confirm.py", line 224, in main
    pooled = pooled.combine_pseudocount(b)
File "chimera/cir/nodes.py", line 91, in combine_pseudocount
    raise ValueError(
ValueError: evidence conflict K=0.860 exceeds threshold 0.80.
Sources irreconcilable: mean1=0.050, mean2=0.900
```

This is a robustness gap in the frozen analysis script, not an engine
defect: the guard firing is the engine refusing to pool irreconcilable
evidence, which is its specified behavior. The preregistration states the
pooled arm "does not change the verdict", so the verdict above stands.
No frozen file was edited after the freeze to work around this.

## Run provenance

- Preregistration: `experiments/h1_pooling/PREREG_V2_CONFIRM.md`
  (SHA-256 `474afe2bc6290d34d3cc857b0c8787d071fe64f6934d7cb659badd3c94a68316`),
  frozen at tag `confirm-v2-freeze`.
- Dataset: `simpleqa_501_1000.jsonl`, 500 questions
  (simpleqa-0501..simpleqa-1000), SHA-256 verified at run time.
- Calibrators: per-mode frozen, fit on questions 1-500 only, hashes
  verified at run time. Model-id table `model_ids.json`, hash verified
  at run time; every collected row's `model_id` matched the table.
- Smoke test: 80 calls on questions 0501-0510, `SMOKE: PASS`
  (100% parse_ok, 4 model-configs x 10 questions). Correctness was not
  inspected at the smoke gate. Provider-returned model ids matched the
  requested ids on all 80 calls. Smoke manifest:
  `runs/confirm-v2/smoke_manifest.json`.
- Full collection: 3,920 new calls on questions 0511-1000; the 80 smoke
  rows were reused (not re-collected). Total 4,000 calls: mode A 1,500
  (3 models x 500), mode B 2,500 (1 model x 500 x 5 samples).
  Parse failures: 0. `provider_model_id` differed from the requested
  `model_id` on 0 of 4,000 rows.
- Collection window: 2026-10-03 ~01:47 UTC to ~02:44 UTC
  (mode A finished 02:32 UTC, mode B 02:44 UTC).
- Responses: `runs/confirm-v2/responses.jsonl` (4,000 rows).
  Credential grep before commit: no secrets found (24 matches were the
  word "secret" inside SimpleQA question text).

## Literal analyzer output

```text
verified dataset: sha256 a170802b135c3a3d... ok
verified calibrator-a: sha256 47684a2ac34537f0... ok
verified calibrator-b: sha256 8ac4b2869a700930... ok
verified model-id table: sha256 e5d9b8e330a024ac... ok
calibrator-a: n=500 fit_date=2026-10-03 metadata={'n_sources': 3, 'source_type': 'cross-model'}
calibrator-b: n=500 fit_date=2026-10-03 metadata={'n_sources': 5, 'source_type': 'single-model'}
dataset: 500 questions, simpleqa-0501..simpleqa-1000
responses: 4000
mode A: n=500 base_rate=0.2720
  agreement AUROC = 0.6714  95% CI [0.6280, 0.7185]
  Brier(constant) = 0.1980  Brier(calibrated) = 0.1835
  const-minus-calibrated 95% CI [-0.0007, 0.0300]
mode B: n=500 base_rate=0.2740
  agreement AUROC = 0.7519  95% CI [0.7121, 0.7908]
  Brier(constant) = 0.1989  Brier(calibrated) = 0.1680
  const-minus-calibrated 95% CI [0.0156, 0.0468]
Traceback (most recent call last):
  ...
ValueError: evidence conflict K=0.860 exceeds threshold 0.80.
Sources irreconcilable: mean1=0.050, mean2=0.900
exit=1
```

(Full output saved at `runs/confirm-v2/analysis_output.txt`.)

## What happens next

Nothing under this preregistration. Inconclusive does not license
re-tuning the calibrator, re-running collection, or redefining the rules.
Any follow-up is a new preregistration.
