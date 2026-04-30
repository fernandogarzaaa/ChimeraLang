"""Runtime primitives for ChimeraLang roadmap systems.

The classes below fall into two groups:

* DifferentialPrivacyEngine, ReplayBuffer, RewardSystem,
  PredictiveCodingRuntime, ConstitutionLayer (in constitution_layer.py)
  — minimum-viable but real implementations.

* CausalModel, MetaLearner, SelfImprover — surface-level facades. They
  validate constructor arguments and check structural invariants
  (e.g. SelfImprover refuses forbidden ops) but do NOT yet run real
  causal inference, gradient adaptation, or formal verification.
  Their results carry ``is_stub=True`` (or, for SelfImprover,
  ``verified=False`` plus ``verifier_invoked=False``) so callers can
  branch on stub status instead of trusting a false-positive payload.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CausalModel:
    """Structural facade — does NOT compute causal effects.

    Returns a record describing the requested intervention and adjustment
    set; ``is_stub=True`` is included so callers can detect that no
    estimation has actually been performed.
    """

    name: str
    adjustment: str = "backdoor"

    def causal_effect(self, intervention: Any, adjust_for: list[str] | None = None) -> dict:
        return {
            "model": self.name,
            "intervention": intervention,
            "adjustment": self.adjustment,
            "adjust_for": adjust_for or [],
            "effect_estimate": None,
            "is_stub": True,
        }


@dataclass
class DifferentialPrivacyEngine:
    epsilon: float = 1.0
    delta: float = 1e-5
    clip_norm: float = 1.0
    total_budget: float = 1.0
    seed: int | None = None
    consumed_epsilon: float = 0.0

    def __post_init__(self) -> None:
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if self.delta <= 0:
            raise ValueError("delta must be positive")
        if self.clip_norm <= 0:
            raise ValueError("clip_norm must be positive")
        if self.total_budget <= 0:
            raise ValueError("total_budget must be positive")

    @property
    def remaining_budget(self) -> float:
        return max(0.0, self.total_budget - self.consumed_epsilon)

    def privatize(self, gradients: list[float]) -> list[float]:
        if self.consumed_epsilon + self.epsilon > self.total_budget:
            raise RuntimeError("privacy budget exhausted")
        rng = random.Random(self.seed)
        norm = math.sqrt(sum(g * g for g in gradients)) or 1.0
        scale = min(1.0, self.clip_norm / norm)
        sigma = self.clip_norm / max(self.epsilon, 1e-9)
        self.consumed_epsilon += self.epsilon
        return [g * scale + rng.gauss(0.0, sigma) for g in gradients]


@dataclass
class MetaLearner:
    """Structural facade — does NOT perform real adaptation.

    Returns the adaptation configuration; ``is_stub=True`` and
    ``weights_updated=False`` mark that no inner-loop gradient steps
    were actually taken.
    """

    base_model: str
    inner_steps: int = 5
    lr: float = 0.01

    def adapt(self, new_task_examples: int, steps: int | None = None) -> dict:
        return {
            "base_model": self.base_model,
            "examples": new_task_examples,
            "inner_steps": steps or self.inner_steps,
            "lr": self.lr,
            "weights_updated": False,
            "is_stub": True,
        }


@dataclass
class SelfImprover:
    """Mutation gate — only the forbidden-ops check is real.

    The ``verifier`` field records the *intended* verifier (e.g.
    ``z3_solver``) but no verifier is actually invoked. The result
    therefore reports ``verified=False`` and ``verifier_invoked=False``
    alongside the real ``forbidden_op_check_passed`` signal. Callers
    that require formal verification should treat this as advisory
    until a verifier is wired in.
    """

    model: str
    verifier: str = "z3_solver"
    max_generations: int = 100
    forbidden_ops: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.max_generations <= 0:
            raise ValueError("max_generations must be positive")

    def propose(self, mutation: dict) -> dict:
        op = mutation.get("op")
        if op in self.forbidden_ops:
            raise ValueError(f"forbidden mutation operation: {op}")
        return {
            "model": self.model,
            "mutation": mutation,
            "verifier": self.verifier,
            "verifier_invoked": False,
            "forbidden_op_check_passed": True,
            "verified": False,
        }


@dataclass
class ReplayBuffer:
    capacity: int
    strategy: str = "prioritized"
    _items: list[tuple[float, Any]] = field(default_factory=list)

    def add(self, item: Any, priority: float = 0.0) -> None:
        self._items.append((priority, item))
        self._items.sort(key=lambda pair: pair[0], reverse=True)
        del self._items[self.capacity:]

    def sample(self, n: int) -> list[Any]:
        return [item for _, item in self._items[:n]]


class RewardSystem:
    def prediction_error(self, expected: float, actual: float) -> float:
        return actual - expected


@dataclass
class PredictiveCodingRuntime:
    depth: int
    error_signal: str = "residual"

    def propagate(self, values: list[float]) -> dict:
        return {
            "layers_evaluated": self.depth,
            "error_signal": self.error_signal,
            "prediction_error": values[-1] - values[0] if values else 0.0,
        }
