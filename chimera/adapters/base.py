"""Base machinery for universal inquiry adapters.

A provider is any callable ``chat(model, prompt) -> str``. This module
provides the shared OpenAI-compatible transport, verdict parsing, the
provider/agent registries, and the adapter factory.
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from typing import Callable

CONFIDENCE_SYSTEM = (
    "You are a fact checker. The user will give you a claim. "
    "Reply with your verdict (TRUE, FALSE, or UNCERTAIN) on the first line. "
    "On the second line, write exactly: CONFIDENCE: <number between 0 and 1> "
    "representing your confidence in your verdict. "
    "For example:\nFALSE\nCONFIDENCE: 0.98"
)

_CONF_RE = re.compile(r"CONFIDENCE:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)

#: Provider name -> chat(model, prompt) -> str.
PROVIDERS: dict[str, Callable[[str, str], str]] = {}

#: Agent name -> (provider, model). Hyphen-free names only; the
#: ChimeraLang parser splits `foo-bar` into separate agents.
AGENTS: dict[str, tuple[str, str]] = {}


def openai_compat_chat(base_url: str, api_key: str,
                       model: str, prompt: str) -> str:
    """Shared transport for OpenAI-compatible chat completions APIs.

    Covers OpenAI, Nebius, Together, Groq, DeepSeek, Mistral, xAI,
    Ollama, LM Studio, vLLM, OpenRouter, and any compatible endpoint.
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": CONFIDENCE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        if resp.status != 200:
            raise RuntimeError(
                f"provider returned HTTP {resp.status} for {base_url}")
        body = json.loads(resp.read().decode("utf-8"))
    choices = body.get("choices") or []
    if not choices:
        raise RuntimeError("provider response missing choices")
    return choices[0]["message"]["content"].strip()


def parse_verdict(text: str) -> tuple[str, float]:
    """Split verdict from the CONFIDENCE line.

    Returns (answer_text, confidence). Defaults to 0.5 when the model
    does not emit a parseable confidence; clamps to [0, 1].
    """
    m = _CONF_RE.search(text)
    confidence = 0.5
    if m:
        try:
            confidence = max(0.0, min(1.0, float(m.group(1))))
        except ValueError:
            pass
        answer = _CONF_RE.sub("", text).strip()
    else:
        answer = text
    return answer, confidence


def make_adapter(
    providers: dict[str, Callable[[str, str], str]] | None = None,
    agents: dict[str, tuple[str, str]] | None = None,
):
    """Build a ChimeraLang inquiry adapter over the given providers.

    providers: name -> chat(model, prompt) -> str. Defaults to PROVIDERS.
    agents: name -> (provider, model). Defaults to AGENTS.
    """
    provs = dict(PROVIDERS)
    if providers:
        provs.update(providers)
    agent_map = dict(AGENTS)
    if agents:
        agent_map.update(agents)

    from chimera.cir.executor import InquiryResponse

    def _adapter(prompt: str, agent_names: list[str]) -> InquiryResponse:
        if not agent_names:
            raise ValueError("adapter requires at least one agent name")
        name = agent_names[0]
        if name not in agent_map:
            raise ValueError(
                f"unknown agent {name!r}; known: {sorted(agent_map)}; "
                f"providers: {sorted(provs)}"
            )
        provider_name, model = agent_map[name]
        chat_fn = provs.get(provider_name)
        if chat_fn is None:
            raise ValueError(f"unknown provider {provider_name!r}")
        text = chat_fn(model, prompt)
        answer, confidence = parse_verdict(text)
        print(
            f"[adapter] provider={provider_name} agent={name} "
            f"confidence={confidence:.2f}",
            file=sys.stderr,
        )
        return InquiryResponse(confidence=confidence, answer=answer)

    return _adapter
