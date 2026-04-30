"""ConstitutionLayer — self-critique loop for constitutional AI."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class ConstitutionResult:
    passed: bool
    text: str | None
    violation_score: float = 0.0
    violations: list = field(default_factory=list)
    rounds_run: int = 0


class ConstitutionLayer:
    REDACTION = "[redacted]"

    def __init__(self, principles: list, rounds: int = 2, max_violation_score: float = 0.05) -> None:
        self.principles = principles
        self.rounds = max(1, int(rounds))
        self.max_violation_score = max_violation_score
        self._forbidden = ["weapon", "bomb", "kill", "poison", "hack",
                           "illegal", "dangerous", "harmful"]

    def apply(self, text: object) -> ConstitutionResult:
        current = str(text) if not isinstance(text, str) else text
        all_violations: list = []
        last_round_violations: list = []
        rounds_run = 0

        for _ in range(self.rounds):
            rounds_run += 1
            last_round_violations = []
            for principle in self.principles:
                v = self._check_principle(current, principle)
                if v:
                    last_round_violations.append(v)
            if not last_round_violations:
                break
            all_violations.extend(last_round_violations)
            current = self._redact(current)

        # Score reflects what's left after the final round — if redaction
        # cleaned everything, the layer "passes" even though intermediate
        # rounds saw violations. Callers can inspect `violations` for the
        # full audit trail.
        per_principle = 1.0 / max(len(self.principles), 1)
        score = min(per_principle * len(last_round_violations), 1.0)
        return ConstitutionResult(
            passed=score <= self.max_violation_score,
            text=current,
            violation_score=round(score, 4),
            violations=all_violations,
            rounds_run=rounds_run,
        )

    def _check_principle(self, text: str, principle: str) -> str | None:
        for marker in self._forbidden:
            if marker in text.lower():
                return f"Forbidden content '{marker}' (principle: {principle!r})"
        return None

    def _redact(self, text: str) -> str:
        out = text
        lower = out.lower()
        for marker in self._forbidden:
            start = 0
            while True:
                idx = lower.find(marker, start)
                if idx == -1:
                    break
                out = out[:idx] + self.REDACTION + out[idx + len(marker):]
                lower = out.lower()
                start = idx + len(self.REDACTION)
        return out
