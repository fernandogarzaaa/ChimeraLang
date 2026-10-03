"""Agreement-based resolve: vote-share over normalized answers.

The H1 experiment (experiments/h1_pooling) found that verbalized
confidence was near chance (mode-B AUROC 0.51 to 0.54) while answer
agreement carried the signal. This module implements the resolve
strategy built on that finding: sources vote with their answer text,
and the belief is the vote share, not the confidences.

The default answer normalization is exactly the one used by the H1
experiment (``experiments/h1_pooling/h1_common.py::normalize_answer``):
lowercase, punctuation removed, leading articles stripped, whitespace
collapsed. Both the comparator and the normalizer are pluggable.
"""
from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import Callable

from chimera.cir.nodes import BetaDist

_ARTICLES = {"a", "an", "the"}
_PUNCT_RE = re.compile("[" + re.escape(string.punctuation) + "]")
_WS_RE = re.compile(r"\s+")

#: Comparator: takes two (normalized) answers, returns True if they agree.
Comparator = Callable[[str, str], bool]
#: Normalizer hook: maps a raw answer to its canonical vote form.
Normalizer = Callable[[str], str]


def normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation, drop leading articles, collapse space.

    Identical to experiments/h1_pooling/h1_common.py::normalize_answer;
    pinned by tests/test_cir_agreement.py::test_normalization_matches_h1.
    """
    text = _PUNCT_RE.sub(" ", text.lower())
    tokens = [t for t in _WS_RE.sub(" ", text).split(" ") if t]
    while tokens and tokens[0] in _ARTICLES:
        tokens = tokens[1:]
    return " ".join(tokens)


def default_comparator(a: str, b: str) -> bool:
    """Two answers agree when their normalized forms are equal."""
    return a == b


@dataclass
class AgreementResult:
    """Outcome of an agreement vote."""
    winner: str
    """Winning normalized answer text."""
    votes: int
    """Votes for the winner."""
    n: int
    """Total voting sources."""
    agreement: float
    """Raw agreement: votes for the winner divided by N."""
    posterior: BetaDist
    """Laplace-smoothed posterior Beta(1 + votes, 1 + (n - votes)).

    The add-one smoothing means a unanimous 3-of-3 vote yields
    Beta(4, 1) with mean 0.8: strong evidence, but never a claim
    of 1.0.
    """


def resolve_agreement(
    answers: list[str],
    comparator: Comparator | None = None,
    normalizer: Normalizer | None = None,
) -> AgreementResult:
    """Vote over answers; winner is the most common normalized answer.

    Ties are broken by earliest source in input order: callers must
    pass sources in agent order (the executor passes consensus
    predecessors, which the lowering wires in agent order), so the
    earliest tied answer is the earliest agent's.

    The comparator receives normalized answer forms. The default
    comparator is equality; a custom comparator clusters greedily in
    input order (each answer joins the first cluster whose
    representative it agrees with, else starts a new one), which keeps
    the earliest-source tie-break.
    """
    if not answers:
        raise ValueError("resolve_agreement requires at least one answer")
    normalizer = normalizer or normalize_answer
    comparator = comparator or default_comparator

    keys = [normalizer(a) for a in answers]
    # Greedy clustering in input order: deterministic, and the earliest
    # cluster wins ties because max() returns the first maximal item.
    clusters: list[list[int]] = []
    for i, key in enumerate(keys):
        placed = False
        for cluster in clusters:
            if comparator(key, keys[cluster[0]]):
                cluster.append(i)
                placed = True
                break
        if not placed:
            clusters.append([i])

    winning = max(clusters, key=len)
    votes = len(winning)
    n = len(answers)
    return AgreementResult(
        winner=keys[winning[0]],
        votes=votes,
        n=n,
        agreement=votes / n,
        posterior=BetaDist(alpha=1.0 + votes, beta=1.0 + (n - votes)),
    )
