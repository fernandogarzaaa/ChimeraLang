"""Built-in provider presets for ChimeraLang adapters.

Each preset is a factory: give it an API key, get back a
``chat(model, prompt) -> str`` function ready for make_adapter.

OpenAI-compatible providers share one transport; native APIs
(Anthropic, Gemini, Cohere) have their own.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Callable

from chimera.adapters.base import CONFIDENCE_SYSTEM, openai_compat_chat

#: Base URLs for OpenAI-compatible providers.
PROVIDER_BASE_URLS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "nebius": "https://api.tokenfactory.nebius.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "together": "https://api.together.xyz/v1",
    "groq": "https://api.groq.com/openai/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "mistral": "https://api.mistral.ai/v1",
    "xai": "https://api.x.ai/v1",
    "ollama": "http://localhost:11434/v1",
}


def openai_compat_provider(provider: str, api_key: str) -> Callable[[str, str], str]:
    """Build a chat function for an OpenAI-compatible provider."""
    base_url = PROVIDER_BASE_URLS.get(provider)
    if base_url is None:
        raise ValueError(
            f"unknown OpenAI-compatible provider {provider!r}; "
            f"known: {sorted(PROVIDER_BASE_URLS)}"
        )

    def _chat(model: str, prompt: str) -> str:
        return openai_compat_chat(base_url, api_key, model, prompt)

    return _chat


def anthropic_provider(api_key: str) -> Callable[[str, str], str]:
    """Build a chat function for the Anthropic Messages API."""

    def _chat(model: str, prompt: str) -> str:
        payload = {
            "model": model,
            "max_tokens": 256,
            "system": CONFIDENCE_SYSTEM,
            "messages": [{"role": "user", "content": prompt}],
        }
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Anthropic returned HTTP {resp.status}")
            body = json.loads(resp.read().decode("utf-8"))
        blocks = body.get("content") or []
        texts = [b.get("text", "") for b in blocks if b.get("type") == "text"]
        if not texts:
            raise RuntimeError("Anthropic response missing text")
        return "\n".join(texts).strip()

    return _chat


def gemini_provider(api_key: str) -> Callable[[str, str], str]:
    """Build a chat function for the Google Gemini API."""

    def _chat(model: str, prompt: str) -> str:
        payload = {
            "system_instruction": {"parts": [{"text": CONFIDENCE_SYSTEM}]},
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 256},
        }
        req = urllib.request.Request(
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={api_key}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Gemini returned HTTP {resp.status}")
            body = json.loads(resp.read().decode("utf-8"))
        candidates = body.get("candidates") or []
        if not candidates:
            raise RuntimeError("Gemini response missing candidates")
        parts = candidates[0].get("content", {}).get("parts") or []
        texts = [p.get("text", "") for p in parts]
        if not texts:
            raise RuntimeError("Gemini response missing text")
        return "\n".join(texts).strip()

    return _chat


def cohere_provider(api_key: str) -> Callable[[str, str], str]:
    """Build a chat function for the Cohere Chat API."""

    def _chat(model: str, prompt: str) -> str:
        payload = {
            "model": model,
            "preamble": CONFIDENCE_SYSTEM,
            "message": prompt,
            "temperature": 0.0,
            "max_tokens": 256,
        }
        req = urllib.request.Request(
            "https://api.cohere.com/v2/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Cohere returned HTTP {resp.status}")
            body = json.loads(resp.read().decode("utf-8"))
        content = (body.get("message") or {}).get("content") or []
        texts = [p.get("text", "") for p in content if isinstance(p, dict)]
        if not texts:
            raise RuntimeError("Cohere response missing text")
        return "\n".join(texts).strip()

    return _chat
