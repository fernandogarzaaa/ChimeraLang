"""Tests for chimera.adapters: universal provider inquiry adapters.

Every provider implements chat(model, prompt) -> str. The adapter
maps agent names to (provider, model), parses verdict + confidence,
and returns InquiryResponse. No network calls in these tests.
"""

import pytest

from chimera.adapters import (
    AGENTS,
    PROVIDERS,
    make_adapter,
    parse_verdict,
)


def _fake_chat_factory(text):
    def _chat(model, prompt):
        _chat.calls.append((model, prompt))
        return text
    _chat.calls = []
    return _chat


def test_all_providers_registered():
    # Every major provider family must have a factory.
    from chimera.adapters import providers as _p
    for name in ["openai", "nebius", "openrouter", "together", "groq",
                 "deepseek", "mistral", "xai", "ollama"]:
        assert name in _p.PROVIDER_BASE_URLS, f"provider {name} not registered"
    for fn in [_p.openai_compat_provider, _p.anthropic_provider,
               _p.gemini_provider, _p.cohere_provider]:
        assert callable(fn)


def test_parse_verdict_true():
    answer, conf = parse_verdict("TRUE\nCONFIDENCE: 0.95")
    assert answer == "TRUE"
    assert conf == pytest.approx(0.95)


def test_parse_verdict_false():
    answer, conf = parse_verdict("FALSE\nCONFIDENCE: 0.98")
    assert answer == "FALSE"
    assert conf == pytest.approx(0.98)


def test_parse_verdict_missing_confidence_defaults_half():
    answer, conf = parse_verdict("TRUE")
    assert answer == "TRUE"
    assert conf == pytest.approx(0.5)


def test_parse_verdict_clamps_range():
    _, conf = parse_verdict("TRUE\nCONFIDENCE: 1.5")
    assert conf == pytest.approx(1.0)
    # Negative confidence is not parseable; defaults to 0.5.
    _, conf = parse_verdict("TRUE\nCONFIDENCE: -0.2")
    assert conf == pytest.approx(0.5)


def test_adapter_routes_to_provider():
    fake = _fake_chat_factory("TRUE\nCONFIDENCE: 0.9")
    adapter = make_adapter(
        providers={"fake": fake},
        agents={"test_agent": ("fake", "fake-model")},
    )
    resp = adapter("Is this true?", ["test_agent"])
    assert resp.confidence == pytest.approx(0.9)
    assert resp.answer == "TRUE"
    assert fake.calls[0][0] == "fake-model"


def test_adapter_unknown_agent_raises():
    adapter = make_adapter(
        providers={"fake": _fake_chat_factory("TRUE\nCONFIDENCE: 0.9")},
        agents={},
    )
    with pytest.raises(ValueError, match="unknown agent"):
        adapter("prompt", ["nope"])


def test_adapter_unknown_provider_raises():
    adapter = make_adapter(
        providers={},
        agents={"x": ("ghost", "model")},
    )
    with pytest.raises(ValueError, match="unknown provider"):
        adapter("prompt", ["x"])


def test_adapter_no_agents_raises():
    adapter = make_adapter(providers={}, agents={})
    with pytest.raises(ValueError, match="at least one agent"):
        adapter("prompt", [])


def test_openai_compat_provider_builds_request():
    from chimera.adapters import openai_compat_chat

    seen = {}

    class FakeResp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            import json
            return json.dumps({
                "choices": [{"message": {"content": "TRUE\nCONFIDENCE: 0.8"}}]
            }).encode()

    import urllib.request
    orig = urllib.request.urlopen

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["auth"] = req.headers.get("Authorization")
        return FakeResp()

    urllib.request.urlopen = fake_urlopen
    try:
        text = openai_compat_chat(
            base_url="https://example.com/v1",
            api_key="sk-test",
            model="m",
            prompt="p",
        )
    finally:
        urllib.request.urlopen = orig

    assert "TRUE" in text
    assert seen["url"] == "https://example.com/v1/chat/completions"
    assert seen["auth"] == "Bearer sk-test"
