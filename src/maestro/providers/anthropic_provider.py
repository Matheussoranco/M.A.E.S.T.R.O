"""Anthropic Messages API backend (stdlib transport).

Implements all three halves of the provider contract natively: ``complete()``,
server-sent-event streaming, and native tool use (``tool_use`` blocks in,
``tool_result`` blocks back).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator

from maestro.providers._http import HttpStreamError, post_json, stream_lines, validate_url
from maestro.providers.base import LLMClient, LLMResponse, StreamEvent
from maestro.telemetry.usage import Usage
from maestro.tools.schema import ToolCall, to_anthropic

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
    "claude-opus-5",
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


def to_messages(messages: list[dict]) -> list[dict]:
    """Translate MAESTRO's neutral history into Anthropic content blocks.

    Two shapes need real translation:

    * an assistant turn carrying :class:`ToolCall` objects becomes a text block
      (when it had prose) plus one ``tool_use`` block per call;
    * ``{"role": "tool", …}`` results become ``tool_result`` blocks inside a
      *user* turn — and **consecutive** results are merged into a single turn,
      because the API expects every result for one assistant turn together.
    """
    out: list[dict] = []
    pending_results: list[dict] = []

    def flush() -> None:
        if pending_results:
            out.append({"role": "user", "content": list(pending_results)})
            pending_results.clear()

    for msg in messages:
        role = msg.get("role", "user")
        if role == "tool":
            pending_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": str(msg.get("tool_call_id", "")),
                    "content": str(msg.get("content", "")),
                }
            )
            continue
        flush()
        calls = msg.get("tool_calls") or []
        if role == "assistant" and calls:
            blocks: list[dict] = []
            text = str(msg.get("content", "") or "")
            if text.strip():
                blocks.append({"type": "text", "text": text})
            blocks.extend(
                {"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}
                for c in calls
            )
            out.append({"role": "assistant", "content": blocks})
            continue
        content = msg.get("content", "")
        # Pre-built block lists (rare, but legal) pass through untouched.
        out.append(
            {"role": role, "content": content if isinstance(content, list) else str(content)}
        )
    flush()
    return out


class AnthropicClient(LLMClient):
    """Talks to ``/v1/messages``.  Available iff an API key is present."""

    name = "anthropic"
    supports_streaming = True
    supports_tools = True

    def __init__(
        self,
        model: str = "",
        api_key: str = "",
        base_url: str = "https://api.anthropic.com",
        timeout: float = 120.0,
        options: dict | None = None,
    ) -> None:
        self.model = model or DEFAULT_MODEL
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.options = dict(options or {})

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    # -- request construction -------------------------------------------------
    def _payload(self, messages, system, max_tokens, temperature, tools) -> dict:
        payload: dict = {
            **self.options,
            "model": self.model,
            "max_tokens": max_tokens or 1024,
            "messages": to_messages(messages),
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = to_anthropic(tools)
        # `temperature` is the only sampling knob MAESTRO plumbs through; the
        # other two in SAMPLING_PARAMS are never sent, and would be gated by the
        # same predicate if they ever were.
        if temperature is not None:
            if accepts_sampling_params(self.model):
                payload["temperature"] = temperature
            else:
                _warn_sampling_dropped(self.model, "temperature")
        return payload

    def _headers(self) -> dict[str, str]:
        return {"x-api-key": self.api_key, "anthropic-version": API_VERSION}

    def _unavailable(self) -> LLMResponse:
        return LLMResponse(
            model=self.model,
            usage=Usage(provider=self.name, model=self.model),
            error="ANTHROPIC_API_KEY not set",
        )

    # -- completion -----------------------------------------------------------
    def complete(
        self, messages, system="", max_tokens=None, temperature=None, tools=None
    ) -> LLMResponse:
        if not self.available:
            return self._unavailable()
        url = f"{self.base_url}/v1/messages"
        try:
            validate_url(url)
        except ValueError as exc:
            return LLMResponse(
                model=self.model,
                usage=Usage(provider=self.name, model=self.model),
                error=f"blocked provider endpoint: {exc}",
            )
        res = post_json(
            url,
            self._payload(messages, system, max_tokens, temperature, tools),
            headers=self._headers(),
            timeout=self.timeout,
        )
        if res.error:
            return LLMResponse(
                model=self.model,
                usage=Usage(provider=self.name, model=self.model),
                error=res.error,
            )
        data = res.json()
        if data.get("_parse_error"):
            return LLMResponse(
                model=self.model,
                usage=Usage(provider=self.name, model=self.model),
                raw=data,
                error=f"invalid JSON response: {data['_parse_error']}",
            )
        model = data.get("model", self.model)
        usage = data.get("usage") or {}
        tokens = Usage(
            provider=self.name,
            model=model,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
        )
        stop_reason = data.get("stop_reason", "") or ""
        # A refusal is an HTTP 200 with an empty (or partial) `content` array.
        # Reported as an error so a swarm surfaces it instead of quietly
        # contributing nothing.
        if stop_reason == "refusal":
            return LLMResponse(
                model=model,
                usage=tokens,
                raw=data,
                stop_reason=stop_reason,
                error=_refusal_message(data),
            )
        parts = data.get("content") or []
        text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
        calls = [
            ToolCall(
                name=str(p.get("name", "")),
                arguments=p.get("input") if isinstance(p.get("input"), dict) else {},
                id=str(p.get("id", "")),
            )
            for p in parts
            if isinstance(p, dict) and p.get("type") == "tool_use"
        ]
        return LLMResponse(
            text=text,
            model=model,
            usage=tokens,
            raw=data,
            tool_calls=calls,
            stop_reason=stop_reason,
        )

    # -- streaming ------------------------------------------------------------
    def complete_stream(
        self, messages, system="", max_tokens=None, temperature=None, tools=None
    ) -> Iterator[StreamEvent]:
        if not self.available:
            response = self._unavailable()
            yield StreamEvent(done=True, response=response, error=response.error)
            return

        payload = self._payload(messages, system, max_tokens, temperature, tools)
        payload["stream"] = True
        state = _StreamState(provider=self.name, model=self.model)
        url = f"{self.base_url}/v1/messages"
        try:
            validate_url(url)
        except ValueError as exc:
            yield StreamEvent(
                done=True,
                error=f"blocked provider endpoint: {exc}",
                response=LLMResponse(
                    model=self.model,
                    usage=Usage(provider=self.name, model=self.model),
                    error=f"blocked provider endpoint: {exc}",
                ),
            )
            return
        try:
            for line in stream_lines(
                url,
                payload,
                headers=self._headers(),
                timeout=self.timeout,
            ):
                if not line.startswith("data:"):
                    # `event:` lines are redundant — every data payload carries
                    # its own "type" — and blank lines just separate frames.
                    continue
                try:
                    event = json.loads(line[5:].strip())
                except json.JSONDecodeError:
                    continue
                text = state.feed(event)
                if text:
                    yield StreamEvent(text=text)
        except HttpStreamError as exc:
            yield StreamEvent(
                done=True,
                error=str(exc),
                response=LLMResponse(
                    model=self.model,
                    usage=Usage(provider=self.name, model=self.model),
                    error=str(exc),
                ),
            )
            return
        response = state.finish()
        yield StreamEvent(done=True, response=response, error=response.error)


class _StreamState:
    """Accumulates Anthropic SSE frames into one :class:`LLMResponse`.

    Text arrives as ``text_delta`` fragments; a tool call arrives as a
    ``tool_use`` block start followed by ``input_json_delta`` fragments that must
    be concatenated before they parse as JSON.
    """

    def __init__(self, provider: str, model: str) -> None:
        self.provider = provider
        self.model = model
        self.text: list[str] = []
        self.blocks: dict[int, dict] = {}
        self.input_tokens: int | None = None
        self.output_tokens: int | None = None
        self.stop_reason = ""
        self.error = ""
        self.saw_event = False

    def feed(self, event: dict) -> str:
        """Consume one frame; return newly produced visible text (if any)."""
        self.saw_event = True
        kind = event.get("type", "")
        if kind == "message_start":
            usage = (event.get("message") or {}).get("usage") or {}
            self.model = (event.get("message") or {}).get("model", self.model)
            self.input_tokens = usage.get("input_tokens", self.input_tokens)
            self.output_tokens = usage.get("output_tokens", self.output_tokens)
        elif kind == "content_block_start":
            block = event.get("content_block") or {}
            if block.get("type") == "tool_use":
                self.blocks[int(event.get("index", 0))] = {
                    "id": str(block.get("id", "")),
                    "name": str(block.get("name", "")),
                    "json": "",
                }
        elif kind == "content_block_delta":
            delta = event.get("delta") or {}
            if delta.get("type") == "text_delta":
                chunk = str(delta.get("text", ""))
                self.text.append(chunk)
                return chunk
            if delta.get("type") == "input_json_delta":
                block = self.blocks.get(int(event.get("index", 0)))
                if block is not None:
                    block["json"] += str(delta.get("partial_json", ""))
        elif kind == "message_delta":
            self.stop_reason = (event.get("delta") or {}).get("stop_reason", self.stop_reason) or ""
            usage = event.get("usage") or {}
            if usage.get("output_tokens") is not None:
                self.output_tokens = usage["output_tokens"]
        elif kind == "error":
            err = event.get("error") or {}
            self.error = f"{err.get('type', 'error')}: {err.get('message', '')}".strip()
        return ""

    def finish(self) -> LLMResponse:
        calls: list[ToolCall] = []
        for index in sorted(self.blocks):
            block = self.blocks[index]
            try:
                arguments = json.loads(block["json"]) if block["json"].strip() else {}
            except json.JSONDecodeError:
                arguments = {"input": block["json"]}
            calls.append(
                ToolCall(
                    name=block["name"],
                    arguments=arguments if isinstance(arguments, dict) else {"input": arguments},
                    id=block["id"],
                )
            )
        error = self.error
        if not error and not self.saw_event:
            error = "stream contained no valid JSON response events"
        if not error and self.stop_reason == "refusal":
            error = "request refused by Anthropic safety classifiers"
        return LLMResponse(
            text="".join(self.text),
            model=self.model,
            usage=Usage(
                provider=self.provider,
                model=self.model,
                input_tokens=self.input_tokens,
                output_tokens=self.output_tokens,
            ),
            tool_calls=calls,
            stop_reason=self.stop_reason,
            error=error,
        )


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
