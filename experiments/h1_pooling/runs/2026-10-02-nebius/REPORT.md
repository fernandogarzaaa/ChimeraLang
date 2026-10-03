# H1 pooling experiment: run report (real models)

Date: 2026-10-02/03. Status: complete. Both pre-registered decision
rules evaluated; no deviation from PREREG.md (see notes on the model
set below, which the prereg leaves unspecified).

## Verdicts

- **H1a (mode B overconfidence gap): SUPPORTED.** Gap 0.6692, 95% CI
  [0.6256, 0.7165] (threshold: > 0.05 with CI excluding zero).
- **H1b (pooling vs plain mean, Brier): adds nothing over averaging.**
  Brier(M) - Brier(P) = -0.0648, 95% CI [-0.0712, -0.0588]: the CI
  excludes zero in the mean's favor, i.e. the shipped pooling is
  *worse* than a plain arithmetic mean, in both modes
  (mode A: -0.0738 [-0.0811, -0.0663]).

## What the numbers say

On the accepted set (arm P pooled mean >= 0.80), the shipped pooling
chain claims a mean confidence of **1.0000** while its empirical
precision is **0.3308** (mode B; mode A: 0.9996 claimed vs 0.3333
precision). The Beta-pseudocount combination saturates at the ceiling
when fed repeated high confidences, even though the model is wrong
two times out of three on those questions. Mean claimed confidence
across all 4,000 samples was 0.943.

Secondary readouts (no decision thresholds per prereg):

- Arm V (majority-vote agreement) is the best-calibrated arm in both
  modes: Brier 0.2042 (A) / 0.3468 (B), AUROC 0.7185 (A) / 0.7610 (B).
  Vote share beats verbalized-confidence pooling.
- Arm C (Platt recalibration on the 20% calibration split) restores
  calibration almost entirely: ECE 0.0436 (A) / 0.0392 (B), vs
  0.6663/0.6691 for the shipped pooling.
- Arm P is the worst arm on Brier in both modes.
- ICC(1,1) of logit(confidence): 0.7013 in mode B (repeated samples are
  strongly correlated, the condition H1 is about), -0.1544 in mode A
  (distinct models disagree, as expected).

Full tables: `summary.json` (deterministic; re-running `analyze.py`
on `responses.jsonl` reproduces it byte-identically).

## Run manifest (per PREREG)

- Dataset: `datasets/simpleqa_500.jsonl` (first 500 SimpleQA
  questions in file order), SHA-256
  `75f37d1c8932257e5e86da9790f4bdc7aa45eb9dd4c6061652a1a01378b09bd4`.
  N = 500.
- Sampling: mode B, k = 5 at temperature 1.0 from one model
  (2,500 calls); mode A, 1 sample at temperature 0 from each of
  3 distinct models (1,500 calls). 4,000 calls total, 0 parse
  failures, 0 excluded questions.
- Model ids: `Qwen/Qwen3-235B-A22B-Instruct-2507` (B + A),
  `deepseek-ai/DeepSeek-V4-Pro` (A), `google/gemma-3-27b-it` (A).
- Prompt SHA-256: `c66766f5e266...` (fixed prompt in `h1_common.py`:
  `Answer: <text>` / `Confidence: <0.00-1.00>` lines).
- Code: branch `exp/h1-pooling`. Mode B collected under the commit
  adding the Nebius backend; mode A rerun under the follow-up
  commit fixing the model set (only the model-id mapping changed;
  backend logic identical). Analysis code unchanged throughout.
- Seeds: collection `--seed 7` (server-side sampling for the real
  backend); analysis bootstrap seed 20261002, 2,000 resamples.

## Notes (not deviations)

- The prereg names no provider or model; the run used Nebius Token
  Factory open models instead of the originally sketched Anthropic
  setup (Inan approved the switch; his Claude quota was exhausted).
  This is a manifest entry, not a deviation: the hypotheses,
  arms, metrics, and decision rules are unchanged.
- Two planned mode-A models (`DeepSeek-V3.2-Exp`, `Kimi-K2.5`) return
  HTTP 404 on the chat endpoint (listed but not servable); three
  others (Kimi-K2.6, GLM-5.1, gpt-oss-120b) are reasoning-first and
  emit null content inside the 150-token budget. Replaced with the
  verified direct-answer models above. The 500 qwen3-235b mode-A rows
  were kept; only the missing 1,000 were recollected.

## Cost

4,000 Nebius chat-completion calls (max_tokens 150). Billed to Inan's
Nebius Token Factory credential; exact charge visible in his Nebius
console (not observable from here).
