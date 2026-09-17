"""Provider registry & factory.

``get_client`` builds a concrete :class:`~maestro.providers.base.LLMClient` from a
provider name plus overrides.  ``resolve_client`` adds the graceful-degradation
policy: if the requested backend is not available and stub fallback is enabled,
you get a deterministic :class:`EchoClient` (with the same persona) instead of a
hard failure — so a swarm always *runs*, even with no keys.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from maestro.config.settings import settings as default_settings
from maestro.providers.anthropic_provider import AnthropicClient
from maestro.providers.base import (
    EchoClient,
    FallbackClient,
    LLMClient,
    LLMResponse,
    NullClient,
    StreamEvent,
    collect_stream,
)
from maestro.providers.ollama_provider import OllamaClient
from maestro.providers.openai_provider import OpenAICompatClient

logger = logging.getLogger("maestro.providers")

#: Canonical provider identifiers users may name in a spec.
PROVIDERS = ("anthropic", "openai", "openai_compat", "ollama", "llamacpp", "echo", "null")


@dataclass
class ProviderSpec:
    """A named, reusable backend configuration referenced by agents."""

    provider: str = "echo"
    model: str = ""
    api_key: str = ""
    base_url: str = ""
    timeout: float | None = None
    persona: str = ""
    options: dict = field(default_factory=dict)


def _normalize_provider(raw: object) -> str:
    """Lower-case provider id only when it is a string (else fall back to echo)."""
    if isinstance(raw, str) and raw:
        return raw.lower().replace("-", "_")
    return "echo"


def get_client(spec: ProviderSpec, settings=None) -> LLMClient:
    """Instantiate the exact backend named by *spec* (no fallback applied)."""
    s = settings or default_settings
    provider = _normalize_provider(spec.provider)
    timeout = spec.timeout if spec.timeout is not None else s.request_timeout

    if provider == "echo":
        return EchoClient(model=spec.model or "echo-1", persona=spec.persona)
    if provider == "null":
        return NullClient()
    if provider == "anthropic":
        return AnthropicClient(
            model=spec.model,
            api_key=spec.api_key or s.anthropic_api_key,
            base_url=spec.base_url or s.anthropic_base_url,
            timeout=timeout,
            options=spec.options,
        )
    if provider == "openai":
        return OpenAICompatClient(
            model=spec.model,
            api_key=spec.api_key or s.openai_api_key,
            base_url=spec.base_url or s.openai_base_url,
            timeout=timeout,
            name="openai",
            require_key=True,
            options=spec.options,
        )
    if provider == "openai_compat":
        # Generic OpenAI-dialect gateway; key optional (local servers accept none).
        return OpenAICompatClient(
            model=spec.model,
            api_key=spec.api_key or s.openai_api_key,
            base_url=spec.base_url or s.openai_base_url,
            timeout=timeout,
            name="openai_compat",
            require_key=False,
            options=spec.options,
        )
    if provider == "llamacpp":
        return OpenAICompatClient(
            model=spec.model or "local",
            api_key=spec.api_key,
            base_url=spec.base_url or s.llamacpp_base_url,
            timeout=timeout,
            name="llamacpp",
            require_key=False,
            options=spec.options,
        )
    if provider == "ollama":
        return OllamaClient(
            model=spec.model,
            base_url=spec.base_url or s.ollama_base_url,
            timeout=timeout,
            options=spec.options,
        )
    raise ValueError(f"unknown provider {spec.provider!r}; choose one of {', '.join(PROVIDERS)}")


def resolve_client(spec: ProviderSpec, settings=None) -> LLMClient:
    """Build the client, degrading to ``echo`` when unavailable (if permitted)."""
    import sys

    s = settings or default_settings
    provider = _normalize_provider(spec.provider)
    client = get_client(spec, s)
    if client.available:
        if s.allow_stub_fallback and provider not in ("echo", "null"):
            logger.warning(
                "STUB fallback armed for provider %r — failures will degrade to 'echo'",
                spec.provider,
            )
            print(
                f"[MAESTRO] STUB fallback armed for {spec.provider!r} (failures degrade to 'echo')",
                file=sys.stderr,
            )
            return FallbackClient(
                client,
                EchoClient(model=f"echo:{spec.provider}", persona=spec.persona),
            )
        return client
    # Normalize the same way get_client() does, so "Null"/"NULL"/"null" are
    # recognized alike — otherwise a differently-cased "null" (which must stay
    # unavailable to force symbolic fallback) would be silently upgraded to a
    # working EchoClient here.
    provider = _normalize_provider(spec.provider)
    if s.allow_stub_fallback and provider not in ("echo", "null"):
        logger.warning(
            "STUB fallback active: provider %r unavailable — "
            "falling back to deterministic 'echo' backend",
            spec.provider,
        )
        print(
            f"[MAESTRO] STUB fallback active for {spec.provider!r} (unavailable — using 'echo')",
            file=sys.stderr,
        )
        return EchoClient(model=f"echo:{spec.provider}", persona=spec.persona)
    return client


__all__ = [
    "PROVIDERS",
    "AnthropicClient",
    "EchoClient",
    "FallbackClient",
    "LLMClient",
    "LLMResponse",
    "NullClient",
    "OllamaClient",
    "OpenAICompatClient",
    "ProviderSpec",
    "StreamEvent",
    "collect_stream",
    "get_client",
    "resolve_client",
]
