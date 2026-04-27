"""GuardLayer — applies CIR guard checks to model outputs."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class GuardResult:
    passed: bool
    violation: str = ""
    confidence: float = 0.0
    variance: float = 0.0


class GuardLayer:
    def __init__(self, max_risk: float = 0.2, strategy: str = "both") -> None:
        self.max_risk = max_risk
        self.strategy = strategy

    def check(self, confidence: float, variance: float = 0.0) -> GuardResult:
        violations = []
        if self.strategy in ("mean", "both"):
            required = 1.0 - self.max_risk
            if confidence < required:
                violations.append(f"mean {confidence:.3f} < required {required:.3f}")
        if self.strategy in ("variance", "both"):
            if variance > 0.05:
                violations.append(f"variance {variance:.4f} > allowed 0.05")
        if violations:
            return GuardResult(passed=False, violation="; ".join(violations),
                               confidence=confidence, variance=variance)
        return GuardResult(passed=True, confidence=confidence, variance=variance)

    def __call__(self, x: object, confidence: float | None = None) -> object:
        conf = confidence if confidence is not None else 0.8
        result = self.check(conf)
        if not result.passed:
            print(f"[chimera guard] VIOLATION: {result.violation}")
        return x
