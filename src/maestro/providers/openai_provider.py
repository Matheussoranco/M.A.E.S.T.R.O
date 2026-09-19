"""OpenAI + OpenAI-compatible backend (stdlib transport).

One class covers a large slice of the ecosystem, because so many gateways speak
the ``/chat/completions`` dialect: OpenAI itself, Groq, Together, OpenRouter,
Fireworks, DeepInfra, vLLM, LM Studio, and llama.cpp's ``--api`` server.  Point
``base_url`` at any of them.

Native function calling and SSE streaming are both implemented here.  Gateways
vary in how faithfully they reproduce either; both paths degrade to "no tool
calls / one chunk" rather than failing, so a thin gateway still works.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

from maestro.providers._http import HttpStreamError, post_json, stream_lines, validate_url
from maestro.providers.base import LLMClient, LLMResponse, StreamEvent
from maestro.telemetry.usage import Usage
from maestro.tools.schema import ToolCall, parse_arguments, to_openai

DEFAULT_MODEL = "gpt-4o-mini"


def to_messages(messages: list[dict], system: str = "") -> list[dict]:
    """Translate MAESTRO's neutral history into chat-completions messages.

    Assistant tool calls become the ``tool_calls`` array (arguments serialized
    to a JSON *string*, as the dialect requires) and each result becomes its own
    ``{"role": "tool", "tool_call_id": …}`` message.
    """
    chat: list[dict] = []
    if system:
        chat.append({"role": "system", "content": system})
    for msg in messages:
        role = msg.get("role", "user")
        if role == "tool":
            chat.append(
                {
                    "role": "tool",
                    "tool_call_id": str(msg.get("tool_call_id", "")),
                    "content": str(msg.get("content", "")),
                }
            )
            continue
        calls = msg.get("tool_calls") or []
        if role == "assistant" and calls:
            chat.append(
                {
                    "role": "assistant",
                    "content": str(msg.get("content", "") or "") or None,
                    "tool_calls": [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {
                                "name": c.name,
                                "arguments": json.dumps(c.arguments, ensure_ascii=False),
                            },
                        }
                        for c in calls
                    ],
                }
            )
            continue
        chat.append({"role": role, "content": str(msg.get("content", ""))})
    return chat


def _parse_calls(raw_calls: list) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for item in raw_calls or []:
        if not isinstance(item, dict):
            continue
        fn = item.get("function") or {}
        calls.append(
            ToolCall(
                name=str(fn.get("name", "")),
                arguments=parse_arguments(fn.get("arguments")),
                id=str(item.get("id", "")),
            )
        )
    return calls


class OpenAICompatClient(LLMClient):
    """Chat-completions client for OpenAI and any compatible endpoint."""

    supports_streaming = True
    supports_tools = True

    def __init__(
        self,
        model: str = "",
        api_key: str = "",
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 120.0,
        name: str = "openai",
        require_key: bool = True,
        options: dict | None = None,
    ) -> None:
        self.model = model or DEFAULT_MODEL
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.name = name
        self.options = dict(options or {})
        # Local servers (vLLM, LM Studio, llama.cpp) accept any/empty key.
        self._require_key = require_key

    @property
    def available(self) -> bool:
        return True if not self._require_key else bool(self.api_key)

    # -- request construction -------------------------------------------------
    def _payload(self, messages, system, max_tokens, temperature, tools) -> dict:
        payload: dict = {
            **self.options,
            "model": self.model,
            "messages": to_messages(messages, system),
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        if tools:
            payload["tools"] = to_openai(tools)
        return payload

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key or 'sk-none'}"}

    def _unavailable(self) -> LLMResponse:
        return LLMResponse(
            model=self.model,
            usage=Usage(provider=self.name, model=self.model),
            error="OPENAI_API_KEY not set",
        )

    # -- completion -----------------------------------------------------------
    def complete(
        self, messages, system="", max_tokens=None, temperature=None, tools=None
    ) -> LLMResponse:
        if self._require_key and not self.api_key:
            return self._unavailable()
        url = f"{self.base_url}/chat/completions"
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
        choices = data.get("choices") or []
        text = ""
        calls: list[ToolCall] = []
        stop_reason = ""
        if choices:
            message = choices[0].get("message") or {}
            text = message.get("content", "") or ""
            calls = _parse_calls(message.get("tool_calls") or [])
            stop_reason = choices[0].get("finish_reason", "") or ""
        usage = data.get("usage") or {}
        model = data.get("model", self.model)
        return LLMResponse(
            text=text,
            model=model,
            usage=Usage(
                provider=self.name,
                model=model,
                input_tokens=usage.get("prompt_tokens"),
                output_tokens=usage.get("completion_tokens"),
            ),
            raw=data,
            tool_calls=calls,
            stop_reason=stop_reason,
        )

    # -- streaming ------------------------------------------------------------
    def complete_stream(
        self, messages, system="", max_tokens=None, temperature=None, tools=None
    ) -> Iterator[StreamEvent]:
        if self._require_key and not self.api_key:
            response = self._unavailable()
            yield StreamEvent(done=True, response=response, error=response.error)
            return

        payload = self._payload(messages, system, max_tokens, temperature, tools)
        payload["stream"] = True
        # Ask for the usage frame; gateways that don't know the option ignore it,
        # and we then honestly report usage as unknown rather than as zero.
        payload["stream_options"] = {"include_usage": True}

        parts: list[str] = []
        pending: dict[int, dict] = {}
        model = self.model
        stop_reason = ""
        input_tokens: int | None = None
        output_tokens: int | None = None
        saw_frame = False
        url = f"{self.base_url}/chat/completions"
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
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    break
                try:
                    frame = json.loads(body)
                except json.JSONDecodeError:
                    continue
                if not isinstance(frame, dict):
                    continue
                saw_frame = True
                model = frame.get("model", model)
                usage = frame.get("usage") or {}
                if usage:
                    input_tokens = usage.get("prompt_tokens", input_tokens)
                    output_tokens = usage.get("completion_tokens", output_tokens)
                for choice in frame.get("choices") or []:
                    stop_reason = choice.get("finish_reason") or stop_reason
                    delta = choice.get("delta") or {}
                    chunk = delta.get("content") or ""
                    if chunk:
                        parts.append(chunk)
                        yield StreamEvent(text=chunk)
                    # Tool calls stream in fragments keyed by `index`: the name
                    # and id arrive once, the arguments accumulate as a string.
                    for call in delta.get("tool_calls") or []:
                        slot = pending.setdefault(
                            int(call.get("index", 0)), {"id": "", "name": "", "args": ""}
                        )
                        if call.get("id"):
                            slot["id"] = str(call["id"])
                        fn = call.get("function") or {}
                        if fn.get("name"):
                            slot["name"] = str(fn["name"])
                        if fn.get("arguments"):
                            slot["args"] += str(fn["arguments"])
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

        calls = [
            ToolCall(
                name=pending[i]["name"],
                arguments=parse_arguments(pending[i]["args"]),
                id=pending[i]["id"],
            )
            for i in sorted(pending)
        ]
        response = LLMResponse(
            text="".join(parts),
            model=model,
            usage=Usage(
                provider=self.name,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ),
            tool_calls=calls,
            stop_reason=stop_reason,
        )
        if not saw_frame:
            response.error = "stream contained no valid JSON response frames"
        yield StreamEvent(done=True, response=response)
