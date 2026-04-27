"""Tests for chimera_runtime package."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from chimera_runtime import BeliefTracker, GuardLayer, ConstitutionLayer

class TestBeliefTracker:
    def test_records_value(self):
        t = BeliefTracker(mode="epistemic"); t.record(0.85)
        assert len(t.history) == 1
    def test_mean_confidence(self):
        t = BeliefTracker(); t.record(0.8); t.record(0.9)
        assert abs(t.mean_confidence() - 0.85) < 0.01
    def test_epistemic_uncertainty(self):
        t = BeliefTracker()
        for v in [0.7, 0.8, 0.9, 0.6, 0.85]: t.record(v)
        assert 0.0 <= t.epistemic_uncertainty() <= 1.0

class TestGuardLayer:
    def test_passes_high_confidence(self):
        assert GuardLayer(max_risk=0.2, strategy="mean").check(confidence=0.9).passed
    def test_fails_low_confidence(self):
        r = GuardLayer(max_risk=0.2, strategy="mean").check(confidence=0.5)
        assert not r.passed and "mean" in r.violation.lower()
    def test_variance_check(self):
        assert not GuardLayer(max_risk=0.2, strategy="variance").check(confidence=0.9, variance=0.1).passed

class TestConstitutionLayer:
    def test_apply_passes_safe_text(self):
        r = ConstitutionLayer(["Be helpful", "Be harmless"], rounds=1).apply("Here is a helpful answer.")
        assert r.passed and r.text is not None
    def test_violation_score_range(self):
        r = ConstitutionLayer(["Never be harmful"], rounds=1).apply("Some output text here.")
        assert 0.0 <= r.violation_score <= 1.0

class TestVectorStore:
    def test_index_and_read(self):
        from chimera_runtime import VectorStore
        store = VectorStore(dim=3)
        store.index([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], ["doc_a", "doc_b"])
        results = store.read([1.0, 0.1, 0.0], top_k=1)
        assert results[0][0] == "doc_a"
    def test_persist_and_load(self):
        import tempfile, os
        from chimera_runtime import VectorStore
        store = VectorStore(dim=2)
        store.index([[1.0, 0.0]], ["hello"])
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            store.persist(path)
            store2 = VectorStore(dim=2)
            store2.load(path)
            assert len(store2._texts) == 1
        finally:
            os.unlink(path)
