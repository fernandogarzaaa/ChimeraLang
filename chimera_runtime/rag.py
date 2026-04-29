"""Dependency-light retrieval-augmented generation runtime for ChimeraLang."""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Any

from chimera_runtime.belief_tracker import BeliefTracker
from chimera_runtime.constitution_layer import ConstitutionLayer
from chimera_runtime.guard_layer import GuardLayer
from chimera_runtime.vector_store import VectorStore


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "in", "is", "it", "of", "on", "or", "that", "the", "to", "what", "with",
}


@dataclass(frozen=True)
class RagDocument:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RagCitation:
    document_id: str
    score: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RagResult:
    answer: str
    passed: bool
    confidence: float
    variance: float
    citations: list[RagCitation] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "passed": self.passed,
            "confidence": round(self.confidence, 4),
            "variance": round(self.variance, 4),
            "citations": [
                {
                    "document_id": citation.document_id,
                    "score": round(citation.score, 4),
                    "text": citation.text,
                    "metadata": citation.metadata,
                }
                for citation in self.citations
            ],
            "violations": list(self.violations),
            "trace": list(self.trace),
        }


class HashingEmbedder:
    """Stable bag-of-words hashing embedder for local RAG demos and tests."""

    def __init__(self, dim: int = 128) -> None:
        if dim <= 0:
            raise ValueError("embedding dimension must be positive")
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in _tokens(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


class HallucinationGuardedRAG:
    """Extractive RAG engine with retrieval confidence and guard results."""

    def __init__(
        self,
        *,
        embedding_dim: int = 128,
        top_k: int = 3,
        min_similarity: float = 0.25,
        max_risk: float = 0.65,
        principles: list[str] | None = None,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not 0.0 <= min_similarity <= 1.0:
            raise ValueError("min_similarity must be between 0 and 1")

        self.embedder = HashingEmbedder(embedding_dim)
        self.store = VectorStore(dim=embedding_dim)
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.guard = GuardLayer(max_risk=max_risk, strategy="both")
        self.constitution = ConstitutionLayer(
            principles or [
                "Only answer from retrieved context.",
                "Decline when retrieved evidence is insufficient.",
                "Do not provide harmful instructions.",
            ],
            rounds=1,
            max_violation_score=0.0,
        )
        self.beliefs = BeliefTracker(mode="rag_grounding")
        self._documents: dict[str, RagDocument] = {}
        self._trace: list[str] = []

    def index(self, documents: list[RagDocument] | list[dict[str, Any]]) -> None:
        if not documents:
            raise ValueError("at least one document is required")

        vectors: list[list[float]] = []
        payloads: list[dict[str, Any]] = []
        for item in documents:
            doc = _coerce_document(item)
            if not doc.id.strip():
                raise ValueError("document id must not be empty")
            if not doc.text.strip():
                raise ValueError(f"document {doc.id!r} text must not be empty")
            self._documents[doc.id] = doc
            vectors.append(self.embedder.embed(doc.text))
            payloads.append({"id": doc.id, "text": doc.text, "metadata": doc.metadata})

        self.store.index(vectors, payloads)
        self._trace.append(f"indexed {len(documents)} document(s)")

    def answer(self, query: str, *, top_k: int | None = None) -> RagResult:
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        if len(self.store) == 0:
            raise ValueError("RAG index is empty")

        k = top_k or self.top_k
        retrieved = self._retrieve(query, k)
        trace = list(self._trace)
        trace.append(f"query: {query}")
        trace.append(f"retrieved {len(retrieved)} candidate(s)")

        citations = [citation for citation in retrieved if citation.score >= self.min_similarity]
        if not citations:
            best = retrieved[0].score if retrieved else 0.0
            violation = f"best similarity {best:.4f} below min_similarity {self.min_similarity:.4f}"
            trace.append(violation)
            return RagResult(
                answer="There is not enough retrieved evidence to answer that.",
                passed=False,
                confidence=best,
                variance=0.0,
                violations=[violation],
                trace=trace,
            )

        answer = self._synthesize_answer(query, citations)
        confidence = self._confidence(query, citations)
        variance = self._variance(citations)
        self.beliefs.record(confidence)

        guard_result = self.guard.check(confidence, variance)
        constitution_result = self.constitution.apply(answer)
        violations: list[str] = []
        if not guard_result.passed:
            violations.append(guard_result.violation)
        if not constitution_result.passed:
            violations.extend(constitution_result.violations)

        passed = not violations
        trace.append(f"confidence={confidence:.4f} variance={variance:.4f}")
        trace.append("guard passed" if guard_result.passed else f"guard failed: {guard_result.violation}")
        trace.append("constitution passed" if constitution_result.passed else "constitution failed")

        return RagResult(
            answer=answer if passed else "Retrieved evidence was blocked by safety guards.",
            passed=passed,
            confidence=confidence,
            variance=variance,
            citations=citations,
            violations=violations,
            trace=trace,
        )

    def _retrieve(self, query: str, top_k: int) -> list[RagCitation]:
        query_vec = self.embedder.embed(query)
        results = self.store.read(query_vec, top_k=top_k)
        citations = []
        for payload, score in results:
            citations.append(
                RagCitation(
                    document_id=str(payload["id"]),
                    score=float(max(score, 0.0)),
                    text=str(payload["text"]),
                    metadata=dict(payload.get("metadata", {})),
                )
            )
        return citations

    def _synthesize_answer(self, query: str, citations: list[RagCitation]) -> str:
        query_terms = set(_tokens(query))
        ranked: list[tuple[int, float, str]] = []
        for citation in citations:
            for sentence in _sentences(citation.text):
                overlap = len(query_terms.intersection(_tokens(sentence)))
                ranked.append((overlap, citation.score, sentence))
        ranked.sort(key=lambda item: (item[0], item[1], len(item[2])), reverse=True)
        selected = [sentence for overlap, _, sentence in ranked if overlap > 0][:2]
        if not selected:
            selected = [citations[0].text.strip()]
        return " ".join(selected)

    def _confidence(self, query: str, citations: list[RagCitation]) -> float:
        query_terms = set(_tokens(query))
        context_terms = set()
        for citation in citations:
            context_terms.update(_tokens(citation.text))
        coverage = len(query_terms.intersection(context_terms)) / max(len(query_terms), 1)
        top_score = citations[0].score
        return max(0.0, min(1.0, (0.65 * top_score) + (0.35 * coverage)))

    @staticmethod
    def _variance(citations: list[RagCitation]) -> float:
        if len(citations) < 2:
            return 0.0
        mean = sum(c.score for c in citations) / len(citations)
        return sum((c.score - mean) ** 2 for c in citations) / len(citations)


def _tokens(text: str) -> list[str]:
    return [token for token in (m.group(0).lower() for m in _TOKEN_RE.finditer(text)) if token not in _STOPWORDS]


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_RE.split(text.strip()) if part.strip()]


def _coerce_document(item: RagDocument | dict[str, Any]) -> RagDocument:
    if isinstance(item, RagDocument):
        return item
    return RagDocument(
        id=str(item.get("id", "")),
        text=str(item.get("text", "")),
        metadata=dict(item.get("metadata", {})),
    )
