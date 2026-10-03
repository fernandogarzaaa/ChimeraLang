"""Shared helpers for the H1 pooling experiment.

No network. No model calls. Pure stdlib.
"""
from __future__ import annotations

import hashlib
import re
import string

# Fixed prompt. Changing this invalidates comparability with earlier runs;
# record PROMPT_SHA256 in the run manifest.
PROMPT_TEMPLATE = (
    "Answer the following question. Reply with exactly two lines:\n"
    "Answer: <your answer, as short as possible>\n"
    "Confidence: <your probability (0.00 to 1.00) that your answer is correct>\n"
    "Question: {question}\n"
)

PROMPT_SHA256 = hashlib.sha256(PROMPT_TEMPLATE.encode("utf-8")).hexdigest()

_ARTICLES = {"a", "an", "the"}
_PUNCT_RE = re.compile("[" + re.escape(string.punctuation) + "]")
_WS_RE = re.compile(r"\s+")


def build_prompt(question: str) -> str:
    return PROMPT_TEMPLATE.format(question=question)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation, drop leading articles, collapse space."""
    text = _PUNCT_RE.sub(" ", text.lower())
    tokens = [t for t in _WS_RE.sub(" ", text).split(" ") if t]
    while tokens and tokens[0] in _ARTICLES:
        tokens = tokens[1:]
    return " ".join(tokens)


def is_correct(predicted: str, gold_answers: list[str]) -> bool:
    pred = normalize_answer(predicted)
    if not pred:
        return False
    return any(pred == normalize_answer(g) for g in gold_answers)


_ANSWER_RE = re.compile(r"^\s*answer\s*:\s*(.+?)\s*$", re.IGNORECASE)
_CONF_RE = re.compile(r"^\s*confidence\s*:\s*(.+?)\s*$", re.IGNORECASE)


def parse_response(raw: str) -> tuple[str | None, float | None, bool]:
    """Extract (answer, confidence) from a raw model response.

    Returns (answer, confidence, ok). ok is False when the confidence
    line is missing or does not parse as a float in [0, 1].
    """
    answer: str | None = None
    confidence: float | None = None
    for line in raw.splitlines():
        m = _ANSWER_RE.match(line)
        if m and answer is None:
            answer = m.group(1).strip()
            continue
        m = _CONF_RE.match(line)
        if m and confidence is None:
            try:
                confidence = float(m.group(1).strip().rstrip("%"))
            except ValueError:
                confidence = None
    ok = (
        answer is not None
        and len(answer) > 0
        and confidence is not None
        and 0.0 <= confidence <= 1.0
    )
    if not ok:
        return answer, None, False
    return answer, confidence, True
