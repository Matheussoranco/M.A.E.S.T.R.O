"""Local Ollama backend (stdlib transport) — first-class local-LLM support."""

from __future__ import annotations

from maestro.providers._http import post_json
from maestro.providers.base import LLMClient, LLMResponse

DEFAULT_MODEL = "llama3.1"


class OllamaClient(LLMClient):
    """Talks to a local Ollama daemon's ``/api/chat`` endpoint.

    Reported as *available* whenever a base URL is configured — reachability is
    only known at call time, and the call fails gracefully (returns an error in
    the :class:`LLMResponse`) rather than raising, so a swarm keeps running.
    """

    name = "ollama"

    def __init__(
        self,
        model: str = "",
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
    ) -> None:
        self.model = model or DEFAULT_MODEL
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return bool(self.base_url)

    def complete(self, messages, system="", max_tokens=None, temperature=None) -> LLMResponse:
        chat: list[dict] = []
        if system:
            chat.append({"role": "system", "content": system})
        chat.extend(
            {"role": m.get("role", "user"), "content": str(m.get("content", ""))} for m in messages
        )
        options: dict = {}
        if max_tokens:
            options["num_predict"] = max_tokens
        if temperature is not None:
            options["temperature"] = temperature
        payload: dict = {"model": self.model, "messages": chat, "stream": False}
        if options:
            payload["options"] = options
        res = post_json(f"{self.base_url}/api/chat", payload, timeout=self.timeout)
        if res.error:
            return LLMResponse(model=self.model, error=res.error)
        data = res.json()
        text = (data.get("message") or {}).get("content", "") or ""
        return LLMResponse(
            text=text,
            model=data.get("model", self.model),
            usage={
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
            },
            raw=data,
        )
