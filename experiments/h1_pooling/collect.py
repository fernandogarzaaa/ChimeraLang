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
  nebius     real API calls to Nebius Token Factory (OpenAI-compatible
             chat completions over open models). Auth uses the stored
             connector credential via the skill's surrogate helper;
             never a raw key. Paid: do not run without explicit
             approval.

Response schema (one JSON object per line):
  id, question, question_index, gold_answers, dataset, dataset_sha256,
  prompt, prompt_sha256, model, model_id, sample_index, backend,
  mode, temperature, raw, raw_sha256, parsed_answer, parsed_confidence,
  parse_ok

question_index is the 0-based position of the question in the dataset
file, so the analysis can reconstruct dataset order (and the
calibration split) from responses.jsonl alone.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from h1_common import build_prompt, parse_response, sha256_hex, PROMPT_SHA256

DEFAULT_AGENT_MODELS = {"claude": "claude-sonnet-4-6"}

# Short agent names -> Nebius Token Factory model ids. Three different
# model families for mode A (distinct, less correlated sources).
# NOTE: DeepSeek-V3.2-Exp and Kimi-K2.5 404 on the chat endpoint, and
# several others (Kimi-K2.6, GLM-5.1, gpt-oss-120b) are reasoning-first
# and return null content within a small max_tokens budget, so the mode
# A set uses direct-answer models verified live on 2026-10-02.
DEFAULT_NEBIUS_MODELS = {
    "qwen3-235b": "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "deepseek-v4pro": "deepseek-ai/DeepSeek-V4-Pro",
    "gemma-3-27b": "google/gemma-3-27b-it",
}

NEBIUS_HOST = "api.tokenfactory.nebius.com"
NEBIUS_URL = f"https://{NEBIUS_HOST}/v1/chat/completions"
NEBIUS_CREDENTIAL = "custom.nebius"
NEBIUS_MAX_TOKENS = 150
NEBIUS_TIMEOUT = 180
NEBIUS_RETRIES = 3

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


def nebius_response(prompt: str, model_id: str, temperature: float) -> str:
    """One chat completion via Nebius Token Factory.

    Auth goes through authd's surrogate exchange (see the nebius skill):
    only hsurr:* values leave this machine, and only to
    api.tokenfactory.nebius.com. Raises RuntimeError on failure.
    """
    sys.path.insert(0, "/opt/hatch/skills/skill-creator/bin")
    from dynamic_credentials import (  # noqa: E402
        add_surrogate_to_request,
        read_json_response,
    )
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": NEBIUS_MAX_TOKENS,
    }
    last_exc: Exception | None = None
    for attempt in range(NEBIUS_RETRIES):
        try:
            req = urllib.request.Request(
                NEBIUS_URL,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            add_surrogate_to_request(
                req, NEBIUS_CREDENTIAL, allowed_hosts=[NEBIUS_HOST],
            )
            with urllib.request.urlopen(req, timeout=NEBIUS_TIMEOUT) as resp:
                data = read_json_response(resp)
            choices = data.get("choices") or []
            if not choices:
                raise RuntimeError("Nebius response missing choices")
            return choices[0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001 - retry transient failures
            last_exc = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(
        f"Nebius call failed after {NEBIUS_RETRIES} attempts: {last_exc}"
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
    ap.add_argument("--backend", choices=["synthetic", "anthropic", "nebius"],
                    required=True)
    ap.add_argument("--mode", choices=["A", "B"], required=True)
    ap.add_argument("--models", required=True, help="comma-separated model/agent names")
    ap.add_argument("--k", type=int, default=None,
                    help="samples per model (default: 5 for mode B, 1 for mode A)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--agent-models", default=None,
                    help="JSON dict mapping agent names to model ids")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel API workers (1 = sequential)")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        raise SystemExit("no models given")
    k = args.k if args.k is not None else (5 if args.mode == "B" else 1)
    temperature = 1.0 if args.mode == "B" else 0.0

    if args.backend == "anthropic":
        defaults = dict(DEFAULT_AGENT_MODELS)
    elif args.backend == "nebius":
        defaults = dict(DEFAULT_NEBIUS_MODELS)
    else:
        defaults = {}
    if args.backend in ("anthropic", "nebius"):
        agent_models = dict(defaults)
        if args.agent_models:
            agent_models.update(json.loads(args.agent_models))
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
    with open(args.dataset, "rb") as fh:
        dataset_sha256 = hashlib.sha256(fh.read()).hexdigest()
    dataset_name = os.path.basename(args.dataset)

    def make_raw(prompt: str, qid: str, question: str, gold: list,
                 model: str, idx: int) -> str:
        if args.backend == "synthetic":
            return synthetic_response(qid, question, gold, model, idx, args.seed)
        if args.backend == "anthropic":
            return anthropic_response(prompt, model_ids[model], temperature)
        return nebius_response(prompt, model_ids[model], temperature)

    def build_row(qi: int, row: dict, model: str, idx: int) -> dict:
        qid = row["id"]
        question = row["question"]
        gold = row["gold_answers"]
        prompt = build_prompt(question)
        raw = make_raw(prompt, qid, question, gold, model, idx)
        answer, confidence, ok = parse_response(raw)
        return {
            "id": qid,
            "question": question,
            "question_index": qi,
            "gold_answers": gold,
            "dataset": dataset_name,
            "dataset_sha256": dataset_sha256,
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
        }

    # (key, qi, row, model, idx) for everything not already collected.
    tasks = []
    for qi, row in enumerate(dataset):
        for model in models:
            for idx in range(k):
                key = (row["id"], model, idx)
                if key not in done:
                    tasks.append((key, qi, row, model, idx))
    n_skipped_existing = sum(
        1 for row in dataset for model in models for idx in range(k)
        if (row["id"], model, idx) in done
    )

    wrote = 0
    skipped = n_skipped_existing
    failed: list[tuple] = []
    lock = threading.Lock()

    def run_task(task):
        key, qi, row, model, idx = task
        try:
            return (key, build_row(qi, row, model, idx), None)
        except Exception as exc:  # noqa: BLE001 - record, keep going
            return (key, None, f"{type(exc).__name__}: {exc}")

    with open(args.out, "a", encoding="utf-8") as out:
        if args.workers > 1:
            pool = ThreadPoolExecutor(max_workers=args.workers)
            results = pool.map(run_task, tasks)
        else:
            results = map(run_task, tasks)
        for key, row_obj, err in results:
            with lock:
                if err is not None:
                    failed.append((key, err))
                elif key in done:
                    skipped += 1
                else:
                    out.write(json.dumps(row_obj, sort_keys=True) + "\n")
                    out.flush()
                    done.add(key)
                    wrote += 1
        if args.workers > 1:
            pool.shutdown()

    print(f"wrote {wrote} responses, skipped {skipped} existing, "
          f"failed {len(failed)} -> {args.out}")
    for key, err in failed[:20]:
        print(f"  FAILED {key}: {err}")
    if failed:
        raise SystemExit(
            f"{len(failed)} samples failed; re-run the same command to resume "
            "only the missing ones"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
