"""ConstitutionLayer — self-critique loop for constitutional AI."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class ConstitutionResult:
    passed: bool
    text: str | None
    violation_score: float = 0.0
    violations: list = field(default_factory=list)


class ConstitutionLayer:
    def __init__(self, principles: list, rounds: int = 2, max_violation_score: float = 0.05) -> None:
        self.principles = principles
        self.rounds = rounds
        self.max_violation_score = max_violation_score
        self._forbidden = ["weapon", "bomb", "kill", "poison", "hack",
                           "illegal", "dangerous", "harmful"]

    def apply(self, text: object) -> ConstitutionResult:
        text_str = str(text) if not isinstance(text, str) else text
        violations = []
        score = 0.0
        for principle in self.principles:
            v = self._check_principle(text_str, principle)
            if v:
                violations.append(v)
                score += 1.0 / max(len(self.principles), 1)
        score = min(score, 1.0)
        return ConstitutionResult(passed=score <= self.max_violation_score,
                                  text=text_str, violation_score=round(score, 4),
                                  violations=violations)

    def _check_principle(self, text: str, principle: str) -> str | None:
        for marker in self._forbidden:
            if marker in text.lower():
                return f"Forbidden content '{marker}' (principle: {principle!r})"
        return None
