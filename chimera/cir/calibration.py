"""Opt-in logistic calibration for resolved beliefs.

A resolved belief carries an uncalibrated score (the posterior mean).
This module maps that score to a calibrated probability with a
two-parameter logistic (Platt-style) model::

    calibrated_p = sigmoid(a * score + b)

The fit is deterministic pure Python (Newton-Raphson / IRLS with a
fixed iteration count, no randomness), so the same training data
always yields the same calibrator.

Calibration is opt-in and only meaningful with a held-out calibration
set: fitting and evaluating on the same questions gives an optimistic
calibrator. ``LogisticCalibrator.fit`` refuses training sets smaller
than ``MIN_FIT_N``.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import date

#: Minimum training points for a logistic fit. Below this the two
#: parameters are unstable; fit() raises instead of returning noise.
MIN_FIT_N = 30

#: Fixed Newton-Raphson iterations. No early stopping, no randomness:
#: the fit is a pure function of the training data.
_FIT_ITERATIONS = 100

#: Ridge penalty keeping the Hessian positive definite on separable data.
_RIDGE = 1e-6


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


@dataclass
class LogisticCalibrator:
    """Two-parameter logistic map from uncalibrated score to probability."""
    a: float
    b: float
    n: int
    dataset_hash: str
    fit_date: str

    @classmethod
    def fit(
        cls,
        scores: list[float],
        outcomes: list[int],
        dataset_hash: str | None = None,
    ) -> "LogisticCalibrator":
        """Fit a and b on (score, 0/1 outcome) pairs, deterministically.

        Raises ValueError when fewer than MIN_FIT_N points are given.
        When dataset_hash is omitted it is computed as the SHA-256 of
        the training pairs, so the calibrator records exactly what it
        was fit on.
        """
        scores = [float(s) for s in scores]
        outcomes = [int(o) for o in outcomes]
        if len(scores) != len(outcomes):
            raise ValueError(
                f"scores ({len(scores)}) and outcomes ({len(outcomes)}) "
                "must have the same length"
            )
        n = len(scores)
        if n < MIN_FIT_N:
            raise ValueError(
                f"refusing to fit logistic calibrator on {n} points: "
                f"minimum is MIN_FIT_N = {MIN_FIT_N}"
            )
        if any(not 0.0 <= s <= 1.0 for s in scores):
            raise ValueError("calibration scores must be in [0, 1]")
        if any(o not in (0, 1) for o in outcomes):
            raise ValueError("calibration outcomes must be 0 or 1")

        if dataset_hash is None:
            payload = json.dumps(
                {"scores": scores, "outcomes": outcomes}, sort_keys=True
            )
            dataset_hash = hashlib.sha256(payload.encode()).hexdigest()

        # Newton-Raphson on the penalized log-likelihood. Fixed
        # iteration count from the fixed start (0, 0): deterministic.
        a, b = 0.0, 0.0
        for _ in range(_FIT_ITERATIONS):
            g_a = g_b = 0.0
            h_aa = h_ab = h_bb = 0.0
            for s, y in zip(scores, outcomes):
                p = _sigmoid(a * s + b)
                w = p * (1.0 - p)
                err = p - y
                g_a += err * s
                g_b += err
                h_aa += w * s * s
                h_ab += w * s
                h_bb += w
            h_aa += _RIDGE
            h_bb += _RIDGE
            det = h_aa * h_bb - h_ab * h_ab
            if det == 0.0:  # pragma: no cover - ridge makes this unreachable
                break
            a -= (h_bb * g_a - h_ab * g_b) / det
            b -= (h_aa * g_b - h_ab * g_a) / det

        return cls(a=a, b=b, n=n, dataset_hash=dataset_hash,
                   fit_date=date.today().isoformat())

    def predict(self, score: float) -> float:
        """Map an uncalibrated score in [0, 1] to a calibrated probability."""
        return _sigmoid(self.a * float(score) + self.b)

    def to_json(self) -> str:
        return json.dumps({
            "a": self.a,
            "b": self.b,
            "n": self.n,
            "dataset_hash": self.dataset_hash,
            "fit_date": self.fit_date,
        }, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "LogisticCalibrator":
        d = json.loads(text)
        return cls(a=float(d["a"]), b=float(d["b"]), n=int(d["n"]),
                   dataset_hash=str(d["dataset_hash"]),
                   fit_date=str(d["fit_date"]))
