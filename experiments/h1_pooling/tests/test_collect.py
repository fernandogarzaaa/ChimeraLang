"""End-to-end: synthetic collection is deterministic, resumable, idempotent."""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "..")
DATASET = os.path.join(EXP, "fixture", "questions.jsonl")


def run_collect(out, *extra):
    cmd = [sys.executable, os.path.join(EXP, "collect.py"),
           "--dataset", DATASET, "--out", out, "--backend", "synthetic",
           *extra]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout


def read_rows(out):
    with open(out) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def test_synthetic_deterministic(tmp_path):
    o1 = str(tmp_path / "r1.jsonl")
    o2 = str(tmp_path / "r2.jsonl")
    run_collect(o1, "--mode", "B", "--models", "claude", "--k", "3", "--seed", "11")
    run_collect(o2, "--mode", "B", "--models", "claude", "--k", "3", "--seed", "11")
    assert read_rows(o1) == read_rows(o2)


def test_resumable_idempotent(tmp_path):
    out = str(tmp_path / "r.jsonl")
    run_collect(out, "--mode", "B", "--models", "claude", "--k", "3", "--seed", "11")
    n1 = len(read_rows(out))
    assert n1 == 40 * 3
    second = run_collect(out, "--mode", "B", "--models", "claude", "--k", "3", "--seed", "11")
    assert "skipped 120 existing" in second
    assert len(read_rows(out)) == n1


def test_row_schema(tmp_path):
    out = str(tmp_path / "r.jsonl")
    run_collect(out, "--mode", "A", "--models", "claude,gpt", "--seed", "11")
    rows = read_rows(out)
    assert len(rows) == 40 * 2
    required = {"id", "question", "prompt", "prompt_sha256", "model", "model_id",
                "sample_index", "backend", "mode", "temperature", "raw",
                "raw_sha256", "parsed_answer", "parsed_confidence", "parse_ok"}
    for row in rows:
        assert required <= set(row), required - set(row)
        assert row["parse_ok"] is True
        assert row["raw_sha256"] == __import__("hashlib").sha256(
            row["raw"].encode()).hexdigest()
