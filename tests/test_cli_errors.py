"""CLI error paths should end in a one-line message and an exit code, not a traceback."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from chimera import cli

ROOT = Path(__file__).resolve().parent.parent


def _run_cli(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "chimera.cli", *args],
        capture_output=True, text=True, cwd=ROOT, timeout=60,
    )


def test_strict_guard_failure_is_clean_error_not_traceback():
    run = _run_cli(["run", "examples/strict_guarded_pipeline.chimera"])
    assert run.returncode == 1
    assert "chimera: guard violation: guard 'x' FAILED" in run.stderr
    assert "Traceback" not in run.stderr


def test_passing_guard_still_runs():
    run = _run_cli(["run", "examples/guarded_pipeline.chimera"])
    assert run.returncode == 0, run.stderr
    assert "Traceback" not in run.stderr


def test_bad_number_flag_is_usage_error(tmp_path):
    corpus = tmp_path / "c.json"
    corpus.write_text(json.dumps({"documents": []}), encoding="utf-8")
    run = _run_cli(["rag", str(corpus), "--query=x", "--top-k=abc"])
    assert run.returncode == 2
    assert "--top-k expects an integer, got 'abc'" in run.stderr
    assert "Traceback" not in run.stderr
    run = _run_cli(["rag", str(corpus), "--query=x", "--min-similarity=high"])
    assert run.returncode == 2
    assert "--min-similarity expects a number" in run.stderr


def test_compile_out_path_may_contain_equals(tmp_path):
    out = tmp_path / "model=v1.py"
    run = _run_cli(["compile", "examples/mnist_classifier.chimera", f"--out={out}"])
    assert run.returncode == 0, run.stderr
    assert out.exists()
    assert not (tmp_path / "model").exists()


def test_flag_helpers():
    flags = ["--out=a=b.py", "--top-k=5", "--min-similarity=0.5"]
    assert cli._flag(flags, "out") == "a=b.py"
    assert cli._flag(flags, "backend", "pytorch") == "pytorch"
    assert cli._flag(flags, "top") is None  # prefix of another flag is not a match
    assert cli._number_flag(flags, "top-k", "3", int) == 5
    assert cli._number_flag(flags, "min-similarity", "0.25", float) == 0.5
    assert cli._number_flag([], "top-k", "3", int) == 3
    with pytest.raises(SystemExit) as exc:
        cli._number_flag(["--top-k=x"], "top-k", "3", int)
    assert exc.value.code == 2
