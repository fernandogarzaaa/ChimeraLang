"""Universal inquiry adapters for ChimeraLang.

Every provider implements one function::

    chat(model: str, prompt: str) -> str   # raw model text

The adapter maps agent names to (provider, model), parses the verdict
and confidence from the response text, and returns an InquiryResponse.

To add a provider: write a ``chat`` function, pass it to make_adapter.
Built-in presets live in chimera.adapters.providers.

Public API:
    from chimera.adapters import make_adapter, parse_verdict
    from chimera.adapters.providers import (
        openai_compat_provider, anthropic_provider,
        gemini_provider, cohere_provider,
    )

    adapter = make_adapter(
        providers={"nebius": openai_compat_provider("nebius", api_key)},
        agents={"nano": ("nebius", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B")},
    )
    run_cir(program, inquiry_adapter=adapter)
"""

from chimera.adapters.base import (
    AGENTS,
    PROVIDERS,
    make_adapter,
    openai_compat_chat,
    parse_verdict,
)

__all__ = [
    "AGENTS",
    "PROVIDERS",
    "make_adapter",
    "openai_compat_chat",
    "parse_verdict",
]
