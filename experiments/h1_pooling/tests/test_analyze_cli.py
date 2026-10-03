"""Regression: analyze.py reads ONLY responses.jsonl (no --dataset)."""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "..")
DATASET = os.path.join(EXP, "fixture", "questions.jsonl")


def run(*args):
    r = subprocess.run([sys.executable, *args], capture_output=True, text=True)
    return r


def collect(out):
    r = run(os.path.join(EXP, "collect.py"),
            "--dataset", DATASET, "--out", out, "--backend", "synthetic",
            "--mode", "B", "--models", "claude", "--k", "3", "--seed", "11")
    assert r.returncode == 0, r.stderr


def analyze(out, *extra):
    return run(os.path.join(EXP, "analyze.py"),
               "--responses", out, "--bootstrap", "50", *extra)


def test_analyze_needs_only_responses(tmp_path):
    out = str(tmp_path / "r.jsonl")
    collect(out)
    r = analyze(out)
    assert r.returncode == 0, r.stderr
    assert "[H1a]" in r.stdout
    assert "[H1b]" in r.stdout
    assert "ICC(1,1)" in r.stdout


def test_analyze_rejects_dataset_flag(tmp_path):
    out = str(tmp_path / "r.jsonl")
    collect(out)
    r = analyze(out, "--dataset", DATASET)
    assert r.returncode != 0
    assert "unrecognized arguments: --dataset" in r.stderr


def test_analyze_deterministic(tmp_path):
    out = str(tmp_path / "r.jsonl")
    collect(out)
    a = analyze(out)
    b = analyze(out)
    assert a.returncode == 0 and b.returncode == 0
    assert a.stdout == b.stdout


def test_analyze_fails_loud_without_gold(tmp_path):
    out = str(tmp_path / "r.jsonl")
    collect(out)
    rows = []
    with open(out) as fh:
        for line in fh:
            if line.strip():
                obj = json.loads(line)
                del obj["gold_answers"]
                rows.append(obj)
    bad = str(tmp_path / "bad.jsonl")
    with open(bad, "w") as fh:
        for obj in rows:
            fh.write(json.dumps(obj) + "\n")
    r = analyze(bad)
    assert r.returncode != 0
    assert "gold_answers" in r.stderr
