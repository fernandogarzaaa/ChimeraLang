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
python analyze.py --dataset fixture/questions.jsonl \
  --responses fixture/responses.jsonl --out-json fixture/summary.json
```

Run the unit tests with `python -m pytest tests/ -q`.

## Real run (paid; requires explicit approval)

Do NOT run the anthropic backend without Inan's explicit approval and
his dataset path. Then:

```
export ANTHROPIC_API_KEY   # from the environment only, never committed
python collect.py --dataset <path> --out responses.jsonl \
  --backend anthropic --mode B --models claude --k 5
python analyze.py --dataset <path> --responses responses.jsonl
```

Every raw response is stored with its SHA-256; collection is resumable
and idempotent. `analyze.py` performs no network access.
