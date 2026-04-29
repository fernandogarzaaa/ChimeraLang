"""VectorStore — first-class retrieval type with HNSW index."""
from __future__ import annotations
import math
import json
import os
from dataclasses import dataclass, field
from typing import Optional

# HNSW via faiss (preferred), fallback to numpy-based IVF otherwise
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


@dataclass
class VectorStore:
    """Persistent vector store with HNSW approximate nearest-neighbor search.

    Syntax: VectorStore<dim>[capacity]

    Operations:
        store.index(vectors, texts)  — add vectors to index
        store.read(query, top_k)      — cosine-similarity top-k retrieval
        store.persist(path)           — save index + texts to disk
        store.load(path)              — restore from disk
    """
    dim: int = 768
    capacity: int = 1_000_000
    _vectors: list = field(default_factory=list)
    _texts: list = field(default_factory=list)
    _index: Optional[object] = field(default=None, repr=False)

    # HNSW config
    m: int = 16          # connections per node
    ef_construction: int = 200  # build-time accuracy

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError("VectorStore dimension must be positive")
        if self.capacity <= 0:
            raise ValueError("VectorStore capacity must be positive")

    def index(self, vectors: list, texts: list) -> None:
        """Add vectors + associated texts to the store."""
        if len(vectors) != len(texts):
            raise ValueError("vectors and texts must have the same length")
        for v in vectors:
            self._validate_vector(v)
        if len(self._vectors) + len(vectors) > self.capacity:
            raise OverflowError("VectorStore capacity exceeded")
        for v, t in zip(vectors, texts):
            self._vectors.append(v)
            self._texts.append(t)
        self._build_index()

    def write(self, key: list, value: object) -> None:
        """Language-level memory write alias for adding one vector/value pair."""
        self.index([key], [value])

    def _build_index(self) -> None:
        if not self._vectors or self.dim <= 0:
            return
        if HAS_FAISS:
            vectors_np = self._ensure_numpy(self._vectors).astype('float32')
            d = vectors_np.shape[1]
            # Use HNSW for fast approximate k-NN
            index = faiss.IndexHNSWFlat(d, self.m)
            index.hnsw.efConstruction = self.ef_construction
            index.add(vectors_np)
            self._index = index
        # else: pure python fallback — no pre-built index; search scans all

    def read(self, query: list, top_k: int = 10) -> list:
        """Cosine-similarity top-k retrieval. Returns [(text, score), ...]."""
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self._validate_vector(query)
        if not self._vectors:
            return []
        if HAS_FAISS and self._index is not None:
            q = self._ensure_numpy([query]).astype('float32')
            # Search HNSW index
            ef_search = max(self.m, top_k * 2)
            self._index.hnsw.efSearch = ef_search
            D, I = self._index.search(q, top_k)
            results = []
            for scores, indices in zip(D, I):
                for score, idx in zip(scores, indices):
                    if idx >= 0 and idx < len(self._texts):
                        results.append((self._texts[int(idx)], float(score)))
            return results
        # Numpy fallback: brute-force cosine
        results = []
        for v, t in zip(self._vectors, self._texts):
            score = self._cosine(query, v)
            results.append((t, score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def persist(self, path: str) -> None:
        """Persist index + vectors + texts to disk as JSON."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "dim": self.dim,
            "capacity": self.capacity,
            "vectors": self._vectors,
            "texts": self._texts,
            "m": self.m,
            "ef_construction": self.ef_construction,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    def load(self, path: str) -> None:
        """Restore index + vectors + texts from disk."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.dim = data.get("dim", 768)
        self.capacity = data.get("capacity", 1_000_000)
        self._vectors = data.get("vectors", [])
        self._texts = data.get("texts", [])
        self.m = data.get("m", 16)
        self.ef_construction = data.get("ef_construction", 200)
        self._build_index()

    def _validate_vector(self, vector: list) -> None:
        if len(vector) != self.dim:
            raise ValueError(f"vector dimension {len(vector)} does not match store dimension {self.dim}")

    @staticmethod
    def _ensure_numpy(v: list) -> "numpy.ndarray":
        """Convert list-of-lists to numpy array (no-op if already array)."""
        try:
            import numpy as np
            return np.asarray(v)
        except Exception:
            return v

    @staticmethod
    def _cosine(a: list, b: list) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(max(sum(x * x for x in a), 1e-9))
        nb = math.sqrt(max(sum(x * x for x in b), 1e-9))
        return dot / (na * nb)

    def __len__(self) -> int:
        return len(self._vectors)

    def __repr__(self) -> str:
        return f"VectorStore<{self.dim}>[{len(self)}/{self.capacity}]"
