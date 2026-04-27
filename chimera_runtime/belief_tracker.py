"""BeliefTracker — wraps BetaDist confidence tracking for PyTorch models."""
from __future__ import annotations
import math
import time
from dataclasses import dataclass, field


@dataclass
class BeliefTracker:
    mode: str = "epistemic"
    history: list = field(default_factory=list)
    _timestamps: list = field(default_factory=list)

    def record(self, value: object) -> None:
        conf = self._extract_confidence(value)
        self.history.append(conf)
        self._timestamps.append(time.time())

    def mean_confidence(self) -> float:
        if not self.history: return 0.0
        return sum(self.history) / len(self.history)

    def epistemic_uncertainty(self) -> float:
        if len(self.history) < 2: return 0.0
        mu = self.mean_confidence()
        variance = sum((x - mu) ** 2 for x in self.history) / len(self.history)
        return min(math.sqrt(variance), 1.0)

    def summary(self) -> dict:
        return {"mode": self.mode, "n_records": len(self.history),
                "mean_confidence": round(self.mean_confidence(), 4),
                "epistemic_uncertainty": round(self.epistemic_uncertainty(), 4)}

    @staticmethod
    def _extract_confidence(value: object) -> float:
        try:
            import torch
            if isinstance(value, torch.Tensor):
                probs = torch.softmax(value.float(), dim=-1) if value.dim() > 0 else value
                return float(probs.max().item())
        except ImportError:
            pass
        if isinstance(value, (int, float)):
            return float(min(max(value, 0.0), 1.0))
        return 0.5
