"""Anthropic Messages API backend (stdlib transport)."""

from __future__ import annotations

import logging

from maestro.providers._http import post_json
from maestro.providers.base import LLMClient, LLMResponse

DEFAULT_MODEL = "claude-sonnet-5"
API_VERSION = "2023-06-01"

# Same logger family as the provider registry, so one handler covers both.
logger = logging.getLogger("maestro.providers")

#: The three sampling knobs the Messages API used to accept.
SAMPLING_PARAMS = ("temperature", "top_p", "top_k")

#: Model families that reject every entry of :data:`SAMPLING_PARAMS` with an
#: HTTP 400.  Sampling parameters were removed with the Claude Opus 4.7
#: generation; models from that generation onward steer through the prompt
#: instead.  Everything older — Sonnet 4.6, Opus 4.6, Haiku 4.5, and earlier —
#: still accepts them.  Matching is a substring test, so dated snapshots
#: ("claude-opus-4-8-20260401") and vendor-prefixed ids ("anthropic.claude-
#: sonnet-5", as used on Bedrock) are covered too.
#:
#: Add a line here when a new model family ships.
NO_SAMPLING_PARAM_MODELS = (
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-sonnet-5",
    "claude-fable-5",
    "claude-mythos-5",
    "claude-mythos-preview",
)

#: Models already warned about, so a long-running swarm logs once, not per call.
_WARNED_SAMPLING_MODELS: set[str] = set()


def accepts_sampling_params(model: str) -> bool:
    """True when *model* accepts ``temperature`` / ``top_p`` / ``top_k``."""
    name = (model or "").strip().lower()
    return not any(family in name for family in NO_SAMPLING_PARAM_MODELS)


def _warn_sampling_dropped(model: str, dropped: str) -> None:
    """Warn once per model that a spec's sampling parameter is being ignored."""
    if model in _WARNED_SAMPLING_MODELS:
        return
    _WARNED_SAMPLING_MODELS.add(model)
    logger.warning(
        "model %r rejects %s (HTTP 400); dropping %s from the request. "
        "Steer this model through its prompt instead.",
        model,
        "/".join(SAMPLING_PARAMS),
        dropped,
    )


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
        # `temperature` is the only sampling knob MAESTRO plumbs through; the
        # other two in SAMPLING_PARAMS are never sent, and would be gated by the
        # same predicate if they ever were.
        if temperature is not None:
            if accepts_sampling_params(self.model):
                payload["temperature"] = temperature
            else:
                _warn_sampling_dropped(self.model, "temperature")
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
        model = data.get("model", self.model)
        usage = data.get("usage") or {}
        tokens = {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
        }
        # A refusal is an HTTP 200 with an empty (or partial) `content` array.
        # Reported as an error so a swarm surfaces it instead of quietly
        # contributing nothing.
        if data.get("stop_reason") == "refusal":
            return LLMResponse(model=model, usage=tokens, raw=data, error=_refusal_message(data))
        parts = data.get("content") or []
        text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
        return LLMResponse(text=text, model=model, usage=tokens, raw=data)


def _refusal_message(data: dict) -> str:
    """Human-readable reason for a ``stop_reason: "refusal"`` response."""
    details = data.get("stop_details") or {}
    msg = "request refused by Anthropic safety classifiers"
    category = details.get("category")
    if category:
        msg += f" (category: {category})"
    explanation = details.get("explanation")
    if explanation:
        msg += f": {explanation}"
    return msg
