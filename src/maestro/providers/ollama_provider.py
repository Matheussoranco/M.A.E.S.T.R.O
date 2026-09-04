"""Local Ollama backend (stdlib transport) — first-class local-LLM support.

Ollama speaks the OpenAI *function* shape for tools but streams newline-
delimited JSON rather than SSE, so both halves are implemented here rather than
inherited from the OpenAI client.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

from maestro.providers._http import HttpStreamError, post_json, stream_lines
from maestro.providers.base import LLMClient, LLMResponse, StreamEvent
from maestro.telemetry.usage import Usage
from maestro.tools.schema import ToolCall, parse_arguments, to_ollama

DEFAULT_MODEL = "llama3.1"


def to_messages(messages: list[dict], system: str = "") -> list[dict]:
    """Translate MAESTRO's neutral history into Ollama chat messages."""
    chat: list[dict] = []
    if system:
        chat.append({"role": "system", "content": system})
    for msg in messages:
        role = msg.get("role", "user")
        if role == "tool":
            # Ollama identifies a result by position and (optionally) name; it
            # has no tool_call_id, so the id is simply dropped here.
            entry = {"role": "tool", "content": str(msg.get("content", ""))}
            if msg.get("name"):
                entry["tool_name"] = str(msg["name"])
            chat.append(entry)
            continue
        calls = msg.get("tool_calls") or []
        if role == "assistant" and calls:
            chat.append(
                {
                    "role": "assistant",
                    "content": str(msg.get("content", "") or ""),
                    "tool_calls": [
                        {"function": {"name": c.name, "arguments": c.arguments}} for c in calls
                    ],
                }
            )
            continue
        chat.append({"role": role, "content": str(msg.get("content", ""))})
    return chat


def _parse_calls(raw_calls: list) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for i, item in enumerate(raw_calls or []):
        if not isinstance(item, dict):
            continue
        fn = item.get("function") or {}
        calls.append(
            ToolCall(
                name=str(fn.get("name", "")),
                arguments=parse_arguments(fn.get("arguments")),
                # Ollama sends no correlation id; synthesize a stable one so the
                # neutral history still pairs calls with their results.
                id=str(item.get("id") or f"ollama-{i}"),
            )
        )
    return calls


class OllamaClient(LLMClient):
    """Talks to a local Ollama daemon's ``/api/chat`` endpoint.

    Reported as *available* whenever a base URL is configured — reachability is
    only known at call time, and the call fails gracefully (returns an error in
    the :class:`LLMResponse`) rather than raising, so a swarm keeps running.
    """

    name = "ollama"
    supports_streaming = True
    supports_tools = True

    def __init__(
        self,
        model: str = "",
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
        options: dict | None = None,
    ) -> None:
        self.model = model or DEFAULT_MODEL
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.options = dict(options or {})

    @property
    def available(self) -> bool:
        return bool(self.base_url)

    def _payload(self, messages, system, max_tokens, temperature, tools, stream: bool) -> dict:
        options: dict = {}
        if max_tokens:
            options["num_predict"] = max_tokens
        if temperature is not None:
            options["temperature"] = temperature
        payload: dict = {
            **self.options,
            "model": self.model,
            "messages": to_messages(messages, system),
            "stream": stream,
        }
        if options:
            payload["options"] = options
        if tools:
            payload["tools"] = to_ollama(tools)
        return payload

    def _usage(self, data: dict, model: str) -> Usage:
        return Usage(
            provider=self.name,
            model=model,
            input_tokens=data.get("prompt_eval_count"),
            output_tokens=data.get("eval_count"),
        )

    # -- completion -----------------------------------------------------------
    def complete(
        self, messages, system="", max_tokens=None, temperature=None, tools=None
    ) -> LLMResponse:
        res = post_json(
            f"{self.base_url}/api/chat",
            self._payload(messages, system, max_tokens, temperature, tools, stream=False),
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
        message = data.get("message") or {}
        model = data.get("model", self.model)
        return LLMResponse(
            text=message.get("content", "") or "",
            model=model,
            usage=self._usage(data, model),
            raw=data,
            tool_calls=_parse_calls(message.get("tool_calls") or []),
            stop_reason=data.get("done_reason", "") or "",
        )

    # -- streaming ------------------------------------------------------------
    def complete_stream(
        self, messages, system="", max_tokens=None, temperature=None, tools=None
    ) -> Iterator[StreamEvent]:
        payload = self._payload(messages, system, max_tokens, temperature, tools, stream=True)
        parts: list[str] = []
        calls: list[ToolCall] = []
        model = self.model
        stop_reason = ""
        final: dict = {}
        saw_frame = False
        try:
            for line in stream_lines(f"{self.base_url}/api/chat", payload, timeout=self.timeout):
                # Ollama streams NDJSON: one complete JSON object per line.
                if not line.strip():
                    continue
                try:
                    frame = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(frame, dict):
                    continue
                saw_frame = True
                model = frame.get("model", model)
                message = frame.get("message") or {}
                chunk = message.get("content") or ""
                if chunk:
                    parts.append(chunk)
                    yield StreamEvent(text=chunk)
                if message.get("tool_calls"):
                    calls.extend(_parse_calls(message["tool_calls"]))
                if frame.get("done"):
                    stop_reason = frame.get("done_reason", "") or stop_reason
                    final = frame
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

        response = LLMResponse(
            text="".join(parts),
            model=model,
            usage=self._usage(final, model),
            tool_calls=calls,
            stop_reason=stop_reason,
        )
        if not saw_frame:
            response.error = "stream contained no valid JSON response frames"
        yield StreamEvent(done=True, response=response)
