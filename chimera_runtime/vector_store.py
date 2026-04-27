"""VectorStore — first-class retrieval type."""
from __future__ import annotations
import math
import json
from dataclasses import dataclass, field


@dataclass
class VectorStore:
    dim: int = 768
    capacity: int = 1_000_000
    _vectors: list = field(default_factory=list)
    _texts: list = field(default_factory=list)

    def index(self, vectors: list, texts: list) -> None:
        for v, t in zip(vectors, texts):
            if len(self._vectors) < self.capacity:
                self._vectors.append(v); self._texts.append(t)

    def read(self, query: list, top_k: int = 10) -> list:
        if not self._vectors: return []
        scores = [(self._cosine(query, v), t) for v, t in zip(self._vectors, self._texts)]
        scores.sort(reverse=True)
        return [(t, s) for s, t in scores[:top_k]]

    def persist(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"dim": self.dim, "vectors": self._vectors, "texts": self._texts}, f)

    def load(self, path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        self._vectors = data["vectors"]; self._texts = data["texts"]

    @staticmethod
    def _cosine(a: list, b: list) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        return dot / (na * nb + 1e-9)
