"""Anthropic Messages API backend (stdlib transport)."""

from __future__ import annotations

from maestro.providers._http import post_json
from maestro.providers.base import LLMClient, LLMResponse

DEFAULT_MODEL = "claude-sonnet-5"
API_VERSION = "2023-06-01"


class AnthropicClient(LLMClient):
    """Talks to ``/v1/messages``.  Available iff an API key is present."""

    name = "anthropic"

    def __init__(
        self,
        model: str = "",
        api_key: str = "",
        base_url: str = "https://api.anthropic.com",
        timeout: float = 120.0,
    ) -> None:
        self.model = model or DEFAULT_MODEL
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def complete(self, messages, system="", max_tokens=None, temperature=None) -> LLMResponse:
        if not self.available:
            return LLMResponse(model=self.model, error="ANTHROPIC_API_KEY not set")
        payload: dict = {
            "model": self.model,
            "max_tokens": max_tokens or 1024,
            "messages": [
                {"role": m.get("role", "user"), "content": str(m.get("content", ""))}
                for m in messages
            ],
        }
        if system:
            payload["system"] = system
        if temperature is not None:
            payload["temperature"] = temperature
        res = post_json(
            f"{self.base_url}/v1/messages",
            payload,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": API_VERSION,
            },
            timeout=self.timeout,
        )
        if res.error:
            return LLMResponse(model=self.model, error=res.error)
        data = res.json()
        parts = data.get("content") or []
        text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
        usage = data.get("usage") or {}
        return LLMResponse(
            text=text,
            model=data.get("model", self.model),
            usage={
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
            },
            raw=data,
        )
