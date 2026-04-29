def test_runtime_exports_neuromorphic_primitives():
    from chimera_runtime import LIFConfig, PoissonEncoder, RateDecoder, SpikeTrain

    spikes = PoissonEncoder(rate=1000, seed=1).encode([1.0, 0.0], timesteps=2)
    train = SpikeTrain(data=spikes)

    assert LIFConfig().threshold == 1.0
    assert train.shape == (2, 2)
    assert RateDecoder().decode(spikes) == [1.0, 0.0]


def test_runtime_exports_swarm_protocol():
    from chimera_runtime import SwarmProtocol

    swarm = SwarmProtocol()

    assert swarm.summary()["total_agents"] == 16
    assert len(swarm.get_domain_agents("code")) == 4


def test_vector_store_supports_language_level_write_alias():
    from chimera_runtime import VectorStore

    store = VectorStore(dim=2, capacity=2)
    store.write([1.0, 0.0], "alpha")

    assert store.read([1.0, 0.0], top_k=1) == [("alpha", 1.0)]


def test_vector_store_read_does_not_require_numpy(monkeypatch):
    import chimera_runtime.vector_store as vector_store

    monkeypatch.setattr(vector_store, "HAS_FAISS", False)
    monkeypatch.setattr(vector_store.VectorStore, "_ensure_numpy", staticmethod(lambda value: value))

    store = vector_store.VectorStore(dim=2, capacity=2)
    store.write([1.0, 0.0], "alpha")
    store.write([0.0, 1.0], "beta")

    assert store.read([1.0, 0.0], top_k=1) == [("alpha", 1.0)]
