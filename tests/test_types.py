"""Tests for the probabilistic type system (chimera/types.py)."""

import pytest

from chimera.types import (
    ChimeraValue,
    Confidence,
    ConfidenceLevel,
    ConfidenceViolation,
    ConfidentValue,
    ConvergeValue,
    ExploreValue,
    MemoryScope,
    ProvisionalValue,
)


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------

class TestConfidence:
    def test_level_high(self):
        assert Confidence(0.99, "test").level == ConfidenceLevel.HIGH

    def test_level_medium(self):
        assert Confidence(0.75, "test").level == ConfidenceLevel.MEDIUM

    def test_level_low(self):
        assert Confidence(0.3, "test").level == ConfidenceLevel.LOW

    def test_level_unknown(self):
        assert Confidence(0.0, "test").level == ConfidenceLevel.UNKNOWN

    def test_combine_product_rule(self):
        """combine() must use the product rule, not averaging."""
        c1 = Confidence(0.8, "a")
        c2 = Confidence(0.9, "b")
        combined = c1.combine(c2)
        assert abs(combined.value - 0.72) < 1e-9

    def test_combine_source_label(self):
        c1 = Confidence(0.8, "x")
        c2 = Confidence(0.9, "y")
        combined = c1.combine(c2)
        assert "x" in combined.source and "y" in combined.source


# ---------------------------------------------------------------------------
# ChimeraValue
# ---------------------------------------------------------------------------

class TestChimeraValue:
    def test_fingerprint_computed_on_access(self):
        val = ChimeraValue(raw="hello", confidence=Confidence(0.9, "s"), trace=[])
        fp1 = val.fingerprint
        assert isinstance(fp1, str) and len(fp1) == 16

    def test_fingerprint_reflects_confidence_mutation(self):
        """Fingerprint must NOT go stale when confidence.value is mutated."""
        val = ChimeraValue(raw="hello", confidence=Confidence(0.9, "s"), trace=[])
        fp_before = val.fingerprint
        val.confidence.value = 0.5
        fp_after = val.fingerprint
        assert fp_before != fp_after


# ---------------------------------------------------------------------------
# ConfidentValue
# ---------------------------------------------------------------------------

class TestConfidentValue:
    def test_creation_succeeds_above_threshold(self):
        val = ConfidentValue(
            raw="sure thing",
            confidence=Confidence(0.99, "test"),
            trace=[],
        )
        assert val.raw == "sure thing"
        assert val.confidence.value == 0.99

    def test_creation_fails_below_threshold(self):
        with pytest.raises(ConfidenceViolation):
            ConfidentValue(
                raw="unsure",
                confidence=Confidence(0.5, "test"),
                trace=[],
            )

    def test_creation_fails_at_exact_threshold_minus_epsilon(self):
        with pytest.raises(ConfidenceViolation):
            ConfidentValue(
                raw="borderline",
                confidence=Confidence(0.9499, "test"),
                trace=[],
            )

    def test_creation_succeeds_at_exact_threshold(self):
        val = ConfidentValue(
            raw="exactly",
            confidence=Confidence(0.95, "test"),
            trace=[],
        )
        assert val.raw == "exactly"

    def test_fingerprint_accessible(self):
        val = ConfidentValue(
            raw=42,
            confidence=Confidence(0.97, "test"),
            trace=[],
        )
        assert len(val.fingerprint) == 16


# ---------------------------------------------------------------------------
# ExploreValue
# ---------------------------------------------------------------------------

class TestExploreValue:
    def test_creation(self):
        val = ExploreValue(
            raw="hypothesis",
            confidence=Confidence(0.5, "explore"),
            trace=[],
            exploration_budget=0.7,
        )
        assert val.raw == "hypothesis"
        assert val.exploration_budget == 0.7


# ---------------------------------------------------------------------------
# ConvergeValue
# ---------------------------------------------------------------------------

class TestConvergeValue:
    def test_creation_with_branches(self):
        branches = [
            ChimeraValue(raw=1, confidence=Confidence(0.9, "b1"), trace=[]),
            ChimeraValue(raw=1, confidence=Confidence(0.8, "b2"), trace=[]),
        ]
        val = ConvergeValue(
            raw=1,
            confidence=Confidence(0.85, "converge"),
            trace=[],
            branch_values=branches,
        )
        assert len(val.branch_values) == 2


# ---------------------------------------------------------------------------
# ProvisionalValue
# ---------------------------------------------------------------------------

class TestProvisionalValue:
    def test_revoke(self):
        val = ProvisionalValue(
            raw="maybe",
            confidence=Confidence(0.6, "prov"),
            memory_scope=MemoryScope.PROVISIONAL,
            trace=["initial"],
        )
        revoked = val.revoke()
        assert revoked.revoked is True
        assert revoked.confidence.value == 0.0
        assert "REVOKED" in revoked.trace
