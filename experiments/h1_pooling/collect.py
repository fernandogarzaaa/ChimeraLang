"""Collect model responses for the H1 pooling experiment.

Writes one JSON line per response to responses.jsonl with the prompt,
model id, sample index, parsed confidence and answer, and SHA-256 of
the raw text. Resumable and idempotent: existing
(question id, model, sample index) keys are skipped.

Backends:
  synthetic  deterministic seeded generator, no network. Used for the
             fixture dry run and for testing the pipeline.
  anthropic  real API calls. Requires ANTHROPIC_API_KEY in the
             environment (never as an argument). Paid: do not run
             without explicit approval.

Response schema (one JSON object per line):
  id, question, prompt, prompt_sha256, model, sample_index, backend,
  mode, temperature, raw, raw_sha256, parsed_answer, parsed_confidence,
  parse_ok
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from h1_common import build_prompt, parse_response, sha256_hex, PROMPT_SHA256

DEFAULT_AGENT_MODELS = {"claude": "claude-sonnet-4-6"}

# Small fixed distractor pool for the synthetic backend.
_DISTRACTORS = [
    "Paris", "London", "42", "1776", "oxygen", "seven", "blue",
    "Mount Everest", "1914", "hydrogen",
]


def _seeded_rng(*parts: str) -> random.Random:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def synthetic_response(
    qid: str, question: str, gold_answers: list[str],
    model: str, sample_index: int, seed: int,
) -> str:
    """Deterministic synthetic response with correlated overconfidence.

    A per-question latent difficulty is shared across samples (so
    samples are correlated, like repeated draws from one model), and
    claimed confidence is systematically ~0.15 above the true
    correctness rate (so the fixture exercises the overconfidence
    decision rule).
    """
    latent = _seeded_rng(str(seed), qid)
    difficulty = latent.random()
    per_model_bias = (int(hashlib.sha256(model.encode()).hexdigest()[:8], 16) % 21 - 10) / 100.0
    rng = _seeded_rng(str(seed), qid, model, str(sample_index))
    correct_prob = max(0.02, min(0.98, 1.0 - difficulty))
    correct = rng.random() < correct_prob
    claimed = max(0.01, min(0.99, correct_prob + 0.15 + per_model_bias + rng.gauss(0, 0.05)))
    if correct:
        answer = gold_answers[0]
    else:
        answer = _DISTRACTORS[rng.randrange(len(_DISTRACTORS))]
    return f"Answer: {answer}\nConfidence: {claimed:.2f}\n"


def anthropic_response(prompt: str, model_id: str, temperature: float) -> str:
    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError(
            "anthropic backend needs the 'anthropic' package installed"
        ) from exc
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set in the environment; refusing to run"
        )
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model_id,
        max_tokens=150,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(
        block.text for block in message.content if getattr(block, "type", "") == "text"
    )


def load_dataset(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def existing_keys(path: str) -> set[tuple[str, str, int]]:
    keys: set[tuple[str, str, int]] = set()
    if not os.path.exists(path):
        return keys
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            keys.add((obj["id"], obj["model"], int(obj["sample_index"])))
    return keys


def main() -> int:
    ap = argparse.ArgumentParser(description="Collect H1 experiment responses")
    ap.add_argument("--dataset", required=True, help="JSONL of {id, question, gold_answers}")
    ap.add_argument("--out", required=True, help="responses.jsonl to write/extend")
    ap.add_argument("--backend", choices=["synthetic", "anthropic"], required=True)
    ap.add_argument("--mode", choices=["A", "B"], required=True)
    ap.add_argument("--models", required=True, help="comma-separated model/agent names")
    ap.add_argument("--k", type=int, default=None,
                    help="samples per model (default: 5 for mode B, 1 for mode A)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--agent-models", default=None,
                    help="JSON dict mapping agent names to model ids")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        raise SystemExit("no models given")
    k = args.k if args.k is not None else (5 if args.mode == "B" else 1)
    temperature = 1.0 if args.mode == "B" else 0.0

    agent_models = dict(DEFAULT_AGENT_MODELS)
    if args.agent_models:
        agent_models.update(json.loads(args.agent_models))

    if args.backend == "anthropic":
        unknown = [m for m in models if m not in agent_models]
        if unknown:
            raise SystemExit(
                f"unknown agent(s) {unknown}; known agents: {sorted(agent_models)}; "
                "refusing to silently fall back to one model"
            )
        model_ids = {m: agent_models[m] for m in models}
    else:
        model_ids = {m: m for m in models}

    dataset = load_dataset(args.dataset)
    done = existing_keys(args.out)
    wrote = 0
    skipped = 0
    with open(args.out, "a", encoding="utf-8") as out:
        for row in dataset:
            qid = row["id"]
            question = row["question"]
            gold = row["gold_answers"]
            prompt = build_prompt(question)
            for model in models:
                for idx in range(k):
                    key = (qid, model, idx)
                    if key in done:
                        skipped += 1
                        continue
                    if args.backend == "synthetic":
                        raw = synthetic_response(qid, question, gold, model, idx, args.seed)
                    else:
                        raw = anthropic_response(prompt, model_ids[model], temperature)
                    answer, confidence, ok = parse_response(raw)
                    out.write(json.dumps({
                        "id": qid,
                        "question": question,
                        "prompt": prompt,
                        "prompt_sha256": PROMPT_SHA256,
                        "model": model,
                        "model_id": model_ids[model],
                        "sample_index": idx,
                        "backend": args.backend,
                        "mode": args.mode,
                        "temperature": temperature,
                        "raw": raw,
                        "raw_sha256": sha256_hex(raw),
                        "parsed_answer": answer,
                        "parsed_confidence": confidence,
                        "parse_ok": ok,
                    }, sort_keys=True) + "\n")
                    done.add(key)
                    wrote += 1
    print(f"wrote {wrote} responses, skipped {skipped} existing -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
