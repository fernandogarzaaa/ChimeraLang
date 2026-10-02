# H1 pooling experiment

Tests whether pooling verbalized confidences with the shipped
`combine_pseudocount` chain (H1) is overconfident under correlation and
whether it adds anything over a plain mean.

Pre-registration: `PREREG.md` (read it first; it fixes hypotheses,
arms, metrics, and decision rules).

## Layout

- `PREREG.md` - pre-registered hypotheses, arms, metrics, decision rules
- `collect.py` - response collection (synthetic or anthropic backend)
- `analyze.py` - deterministic analysis; reads only `responses.jsonl`
- `h1_common.py` - shared prompt / parsing / matching helpers
- `fixture/questions.jsonl` - 40-question synthetic fixture
- `tests/` - unit tests for parsing, matching, bootstrap, ECE, ICC

## Dry run (no network, no cost)

From this directory:

```
python collect.py --dataset fixture/questions.jsonl \
  --out fixture/responses_b.jsonl --backend synthetic --mode B \
  --models claude --k 5 --seed 7
python collect.py --dataset fixture/questions.jsonl \
  --out fixture/responses_a.jsonl --backend synthetic --mode A \
  --models claude,gpt,gemini --seed 8
cat fixture/responses_b.jsonl fixture/responses_a.jsonl > fixture/responses.jsonl
python analyze.py --responses fixture/responses.jsonl \
  --out-json fixture/summary.json
```

Run the unit tests with `python -m pytest tests/ -q`.

## Real run (paid; requires explicit approval)

Do NOT run a paid backend without Inan's explicit approval. Approved
2026-10-02: Nebius Token Factory (open models; stored credential, no
raw keys). Then:

```
python collect.py --dataset datasets/simpleqa_500.jsonl \
  --out runs/<date>/responses_b.jsonl --backend nebius \
  --mode B --models qwen3-235b --k 5 --workers 8
python collect.py --dataset datasets/simpleqa_500.jsonl \
  --out runs/<date>/responses_a.jsonl --backend nebius \
  --mode A --models qwen3-235b,deepseek-v32,kimi-k2.5 --workers 8
cat runs/<date>/responses_b.jsonl runs/<date>/responses_a.jsonl \
  > runs/<date>/responses.jsonl
python analyze.py --responses runs/<date>/responses.jsonl \
  --out-json runs/<date>/summary.json
```

Mode B agent `qwen3-235b` maps to Qwen/Qwen3-235B-A22B-Instruct-2507;
mode A adds deepseek-ai/DeepSeek-V3.2-Exp and moonshotai/Kimi-K2.5
(three distinct model families). The prereg names no provider, so this
is not a deviation; model ids are recorded in every row and the run
manifest.

Every raw response is stored with its SHA-256; each row also carries
its gold answers, its position in the dataset, and the dataset
name/SHA-256, so the analysis reads only responses.jsonl. Collection
is resumable and idempotent. `analyze.py` performs no network access.
