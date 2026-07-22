"""Provider layer — availability logic, determinism, factory & fallback."""

from __future__ import annotations

import pytest

from maestro.config.settings import Settings
from maestro.providers import (
    AnthropicClient,
    EchoClient,
    NullClient,
    OllamaClient,
    OpenAICompatClient,
    ProviderSpec,
    get_client,
    resolve_client,
)


def test_echo_is_available_and_deterministic():
    c = EchoClient()
    assert c.available
    r1 = c.complete([{"role": "user", "content": "hello world"}])
    r2 = c.complete([{"role": "user", "content": "hello world"}])
    assert r1.ok() and r1.text == r2.text
    assert "hello world" in r1.text


def test_null_is_unavailable():
    c = NullClient()
    assert not c.available
    assert c.complete([{"role": "user", "content": "x"}]).error


def test_factory_builds_each_provider_type():
    assert isinstance(get_client(ProviderSpec(provider="echo")), EchoClient)
    assert isinstance(get_client(ProviderSpec(provider="null")), NullClient)
    assert isinstance(get_client(ProviderSpec(provider="anthropic")), AnthropicClient)
    assert isinstance(get_client(ProviderSpec(provider="openai")), OpenAICompatClient)
    assert isinstance(get_client(ProviderSpec(provider="ollama")), OllamaClient)
    assert isinstance(get_client(ProviderSpec(provider="llamacpp")), OpenAICompatClient)


def test_unknown_provider_raises():
    with pytest.raises(ValueError):
        get_client(ProviderSpec(provider="does-not-exist"))


def test_cloud_providers_unavailable_without_keys():
    s = Settings(anthropic_api_key="", openai_api_key="")
    assert not get_client(ProviderSpec(provider="anthropic"), s).available
    assert not get_client(ProviderSpec(provider="openai"), s).available
    # Local / compat endpoints don't require a key.
    assert get_client(ProviderSpec(provider="ollama"), s).available
    assert get_client(ProviderSpec(provider="llamacpp"), s).available
    assert get_client(ProviderSpec(provider="openai_compat"), s).available


def test_resolve_falls_back_to_echo_when_unavailable():
    s = Settings(anthropic_api_key="", allow_stub_fallback=True)
    client = resolve_client(ProviderSpec(provider="anthropic", persona="lead"), s)
    assert isinstance(client, EchoClient)
    assert client.available


def test_resolve_no_fallback_returns_unavailable_client():
    s = Settings(anthropic_api_key="", allow_stub_fallback=False)
    client = resolve_client(ProviderSpec(provider="anthropic"), s)
    assert isinstance(client, AnthropicClient)
    assert not client.available


def test_available_check_makes_no_network_call():
    # Point at an unroutable host; .available must still return instantly.
    c = OllamaClient(base_url="http://10.255.255.1:1")
    assert c.available is True  # availability is config-only, never a round-trip
