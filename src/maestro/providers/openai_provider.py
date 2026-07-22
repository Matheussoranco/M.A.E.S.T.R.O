"""OpenAI + OpenAI-compatible backend (stdlib transport).

One class covers a large slice of the ecosystem, because so many gateways speak
the ``/chat/completions`` dialect: OpenAI itself, Groq, Together, OpenRouter,
Fireworks, DeepInfra, vLLM, LM Studio, and llama.cpp's ``--api`` server.  Point
``base_url`` at any of them.
"""

from __future__ import annotations

from maestro.providers._http import post_json
from maestro.providers.base import LLMClient, LLMResponse

DEFAULT_MODEL = "gpt-4o-mini"


class OpenAICompatClient(LLMClient):
    """Chat-completions client for OpenAI and any compatible endpoint."""

    def __init__(
        self,
        model: str = "",
        api_key: str = "",
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 120.0,
        name: str = "openai",
        require_key: bool = True,
    ) -> None:
        self.model = model or DEFAULT_MODEL
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.name = name
        # Local servers (vLLM, LM Studio, llama.cpp) accept any/empty key.
        self._require_key = require_key

    @property
    def available(self) -> bool:
        return True if not self._require_key else bool(self.api_key)

    def complete(self, messages, system="", max_tokens=None, temperature=None) -> LLMResponse:
        if self._require_key and not self.api_key:
            return LLMResponse(model=self.model, error="OPENAI_API_KEY not set")
        chat: list[dict] = []
        if system:
            chat.append({"role": "system", "content": system})
        chat.extend(
            {"role": m.get("role", "user"), "content": str(m.get("content", ""))}
            for m in messages
        )
        payload: dict = {"model": self.model, "messages": chat}
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        headers = {"Authorization": f"Bearer {self.api_key or 'sk-none'}"}
        res = post_json(
            f"{self.base_url}/chat/completions", payload, headers=headers, timeout=self.timeout
        )
        if res.error:
            return LLMResponse(model=self.model, error=res.error)
        data = res.json()
        choices = data.get("choices") or []
        text = ""
        if choices:
            text = (choices[0].get("message") or {}).get("content", "") or ""
        usage = data.get("usage") or {}
        return LLMResponse(
            text=text,
            model=data.get("model", self.model),
            usage={
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
            },
            raw=data,
        )
