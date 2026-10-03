"""Cross-process determinism: lowering must be byte-identical across PYTHONHASHSEED.

Stage 1 of the strict-guard-source work. Every example and fixture program
is lowered in a fresh subprocess under at least five distinct
PYTHONHASHSEED values; the serialized graph bytes must compare exactly.
A certificate produced under one seed must also verify under another.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PROGRAMS = sorted((REPO_ROOT / "examples").glob("*.chimera"))
SEEDS = [0, 1, 2, 42, 1337]

assert len(SEEDS) >= 5, "need at least five distinct PYTHONHASHSEED values"

LOWER_SNIPPET = r"""
import sys
from chimera.lexer import Lexer
from chimera.parser import Parser
from chimera.cir.lower import CIRLowering
from chimera.cir.certify import serialize_graph, _canonical_bytes

src = open(sys.argv[1], encoding="utf-8").read()
program = Parser(Lexer(src).tokenize()).parse()
graph = CIRLowering().lower(program)
sys.stdout.buffer.write(_canonical_bytes(serialize_graph(graph)))
"""

CERT_SNIPPET = r"""
import json
import sys
from chimera.cir import run_cir
from chimera.cir.certify import certify_cir
from chimera.cir.executor import InquiryResponse
from chimera.lexer import Lexer
from chimera.parser import Parser
from chimera.cir.lower import CIRLowering

def adapter(prompt, agents):
    return InquiryResponse(confidence=0.95, answer="yes")

src = open(sys.argv[1], encoding="utf-8").read()
program = Parser(Lexer(src).tokenize()).parse()
result = run_cir(program, inquiry_adapter=adapter)
graph = CIRLowering().lower(program)
cert = certify_cir(src, graph, result, strict_guard=True)
sys.stdout.buffer.write(json.dumps(cert).encode("utf-8"))
"""

VERIFY_SNIPPET = r"""
import json
import sys
from chimera.verify import CertificateVerifier

cert = json.loads(sys.stdin.read())
res = CertificateVerifier().verify(cert)
sys.stdout.write("VALID" if res.valid else "INVALID")
if not res.valid:
    sys.stdout.write("\n" + "\n".join(res.failures))
"""


def _run_snippet(snippet, seed, stdin_bytes=None, argv=()):
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = str(seed)
    proc = subprocess.run(
        [sys.executable, "-c", snippet, *argv],
        input=stdin_bytes,
        env=env,
        capture_output=True,
        cwd=str(REPO_ROOT),
        timeout=180,
    )
    return proc


def test_programs_found():
    assert PROGRAMS, "expected example .chimera programs under examples/"


@pytest.mark.parametrize("program", [p.name for p in PROGRAMS])
def test_lowering_byte_identical_across_hash_seeds(program):
    prog = REPO_ROOT / "examples" / program
    outputs = []
    for seed in SEEDS:
        proc = _run_snippet(LOWER_SNIPPET, seed, argv=(str(prog),))
        assert proc.returncode == 0, (
            f"{program}: lowering failed under seed {seed}: "
            f"{proc.stderr.decode('utf-8', 'replace')[:500]}"
        )
        outputs.append(proc.stdout)
    for seed, out in zip(SEEDS, outputs):
        assert out == outputs[0], (
            f"{program}: serialized graph differs under PYTHONHASHSEED={seed} "
            f"vs {SEEDS[0]} (ordering dependence)"
        )


def test_certificate_produced_under_one_seed_verifies_under_another():
    prog = REPO_ROOT / "examples" / "guarded_pipeline.chimera"
    assert prog.exists(), "expected examples/guarded_pipeline.chimera"
    produced = _run_snippet(CERT_SNIPPET, SEEDS[0], argv=(str(prog),))
    assert produced.returncode == 0, (
        f"cert production failed under seed {SEEDS[0]}: "
        f"{produced.stderr.decode('utf-8', 'replace')[:500]}"
    )
    for seed in SEEDS[1:]:
        verified = _run_snippet(
            VERIFY_SNIPPET, seed, stdin_bytes=produced.stdout)
        assert verified.returncode == 0, (
            f"verifier crashed under seed {seed}: "
            f"{verified.stderr.decode('utf-8', 'replace')[:500]}"
        )
        out = verified.stdout.decode("utf-8")
        assert out.startswith("VALID"), (
            f"certificate produced under seed {SEEDS[0]} failed verification "
            f"under seed {seed}: {out[:500]}"
        )
