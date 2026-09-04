"""Provider abstraction — the single interface every LLM backend implements.

The contract is intentionally tiny so a new backend stays a couple of dozen
lines.  It has three parts:

``complete()``
    One request, one :class:`LLMResponse`.  Every backend implements it.

``complete_stream()``
    The same request, yielded incrementally as :class:`StreamEvent` objects.
    **Opt-in and never mandatory**: the default implementation below runs
    ``complete()`` and yields its result as a single chunk, so a caller may
    always stream against any backend.  Providers with real server-sent-event
    support override it and set :attr:`LLMClient.supports_streaming`.

``tools=…``
    Native function calling, when :attr:`LLMClient.supports_tools` is set.
    Backends without it simply ignore the argument, and the caller falls back to
    the plain-text ReAct convention.

Two backends here need no network at all: :class:`NullClient` (always
unavailable — forces symbolic fallback) and :class:`EchoClient` (deterministic,
so swarm control-flow is testable offline).
"""

from __future__ import annotations

import abc
import hashlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from maestro.telemetry.usage import Usage
from maestro.tools.base import Tool
from maestro.tools.schema import ToolCall


@dataclass
class LLMResponse:
    """A single completion returned by a backend."""

    text: str = ""
    model: str = ""
    #: Tri-state token accounting — see :class:`maestro.telemetry.usage.Usage`.
    usage: Usage = field(default_factory=Usage)
    raw: Any = None
    error: str = ""
    #: Native function calls the model asked for (empty on the ReAct path).
    tool_calls: list[ToolCall] = field(default_factory=list)
    #: Provider's stop reason, normalized to its own vocabulary.
    stop_reason: str = ""

    def ok(self) -> bool:
        """True when the call produced usable, non-empty text."""
        return not self.error and bool(self.text.strip())

    def wants_tools(self) -> bool:
        """True when the model asked to run one or more tools."""
        return not self.error and bool(self.tool_calls)


@dataclass
class StreamEvent:
    """One step of a streamed completion.

    The contract, honoured by every backend including the fallback:

    * zero or more **text events** (``text`` non-empty, ``done`` false), each
      carrying only the *newly produced* fragment — concatenating them yields
      the full completion;
    * then exactly one **terminal event** (``done`` true) carrying the assembled
      :class:`LLMResponse` in :attr:`response`, errors included.

    A consumer that only cares about the final answer can therefore ignore
    everything until ``done`` — see :func:`collect_stream`.
    """

    text: str = ""
    done: bool = False
    response: LLMResponse | None = None
    error: str = ""


def collect_stream(events: Iterator[StreamEvent]) -> LLMResponse:
    """Drain a stream and return its final :class:`LLMResponse`.

    This is the bridge that lets streaming flow through code paths that need a
    whole answer (topology aggregation, tool loops): the caller may still watch
    the fragments go by, but the value handed onward is identical to what
    ``complete()`` would have returned.
    """
    parts: list[str] = []
    final: LLMResponse | None = None
    for event in events:
        if event.text:
            parts.append(event.text)
        if event.done:
            final = event.response
    if final is not None:
        return final
    # A provider that ended without a terminal event is misbehaving; salvage the
    # text rather than losing the whole turn.
    text = "".join(parts)
    return LLMResponse(text=text, error="" if text else "stream ended without a result")


class LLMClient(abc.ABC):
    """Minimal chat-completion interface shared by every provider."""

    #: Short, stable identifier for the backend (``"anthropic"``, ``"ollama"``…).
    name: str = "base"
    #: The concrete model this client talks to.
    model: str = ""
    #: True when :meth:`complete_stream` streams for real rather than falling back.
    supports_streaming: bool = False
    #: True when the backend accepts native tool/function definitions.
    supports_tools: bool = False

    @property
    @abc.abstractmethod
    def available(self) -> bool:
        """Cheap readiness check — must not perform a network round-trip."""

    @abc.abstractmethod
    def complete(
        self,
        messages: list[dict[str, Any]],
        system: str = "",
        max_tokens: int | None = None,
        temperature: float | None = None,
        tools: list[Tool] | None = None,
    ) -> LLMResponse:
        """Run one completion over ``[{'role': 'user'|'assistant', 'content': …}]``."""

    def complete_stream(
        self,
        messages: list[dict[str, Any]],
        system: str = "",
        max_tokens: int | None = None,
        temperature: float | None = None,
        tools: list[Tool] | None = None,
    ) -> Iterator[StreamEvent]:
        """Stream a completion.

        The default implementation is the graceful fallback: run the ordinary
        completion and emit it as one text event plus the terminal event.  A
        backend that cannot stream therefore still satisfies the interface, so
        callers never need to ask whether streaming is available before using
        it (:attr:`supports_streaming` says whether it is *incremental*).
        """
        # `tools` is only forwarded when actually requested: third-party
        # LLMClient implementations written against the 0.1 signature (which had
        # no `tools` parameter) keep working for every non-tool call.
        if tools:
            response = self.complete(messages, system, max_tokens, temperature, tools)
        else:
            response = self.complete(messages, system, max_tokens, temperature)
        if response.text:
            yield StreamEvent(text=response.text)
        yield StreamEvent(done=True, response=response, error=response.error)

    def ask(self, prompt: str, system: str = "", max_tokens: int | None = None) -> str:
        """Convenience one-shot; returns text only (empty string on failure)."""
        return self.complete([{"role": "user", "content": prompt}], system, max_tokens).text

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"{type(self).__name__}(name={self.name!r}, model={self.model!r})"


class NullClient(LLMClient):
    """No backend configured; consumers must fall back to non-LLM paths."""

    name = "null"

    @property
    def available(self) -> bool:
        return False

    def complete(
        self, messages, system="", max_tokens=None, temperature=None, tools=None
    ) -> LLMResponse:
        return LLMResponse(
            model="null",
            usage=Usage(provider=self.name, model="null"),
            error="no LLM backend configured",
        )


class EchoClient(LLMClient):
    """Deterministic offline backend.

    Produces a stable, human-readable stand-in for a completion.  This lets the
    *structure* of any swarm — routing, fan-out, aggregation, debate rounds — be
    exercised end-to-end with no keys and no network, and makes the whole test
    suite deterministic.  It is also the graceful-degradation target when a real
    provider is unavailable.

    It streams for real (word by word), so the streaming path is exercised
    offline exactly like a live backend.
    """

    name = "echo"
    supports_streaming = True

    def __init__(self, model: str = "echo-1", persona: str = "") -> None:
        self.model = model
        self.persona = persona

    @property
    def available(self) -> bool:
        return True

    def _render(self, messages, system: str) -> LLMResponse:
        last_user = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user = str(msg.get("content", ""))
                break
        digest = hashlib.sha1((system + "\n" + last_user).encode("utf-8", "replace")).hexdigest()[
            :8
        ]
        who = self.persona or "echo"
        # A compact, deterministic "answer" that quotes the request back so the
        # reader can see the request actually reached this agent.
        snippet = " ".join(last_user.split())[:280]
        text = f"[{who}#{digest}] {snippet}" if snippet else f"[{who}#{digest}] (no input)"
        return LLMResponse(
            text=text,
            model=self.model,
            usage=Usage(
                provider=self.name,
                model=self.model,
                input_tokens=len(last_user.split()),
                output_tokens=len(text.split()),
            ),
            stop_reason="end_turn",
        )

    def complete(
        self, messages, system="", max_tokens=None, temperature=None, tools=None
    ) -> LLMResponse:
        return self._render(messages, system)

    def complete_stream(
        self, messages, system="", max_tokens=None, temperature=None, tools=None
    ) -> Iterator[StreamEvent]:
        response = self._render(messages, system)
        words = response.text.split(" ")
        for i, word in enumerate(words):
            yield StreamEvent(text=word if i == len(words) - 1 else word + " ")
        yield StreamEvent(done=True, response=response)


class FallbackClient(LLMClient):
    """Use a deterministic client when a configured backend cannot answer.

    The primary client is attempted first.  Streaming falls back only when no
    text has reached the consumer, because replacing a partial stream would
    duplicate visible output.
    """

    def __init__(self, primary: LLMClient, fallback: LLMClient) -> None:
        self.primary = primary
        self.fallback = fallback
        self.name = primary.name
        self.model = primary.model
        self.supports_streaming = primary.supports_streaming
        self.supports_tools = primary.supports_tools

    @property
    def available(self) -> bool:
        return self.primary.available or self.fallback.available

    def complete(
        self, messages, system="", max_tokens=None, temperature=None, tools=None
    ) -> LLMResponse:
        try:
            if tools:
                response = self.primary.complete(messages, system, max_tokens, temperature, tools)
            else:
                response = self.primary.complete(messages, system, max_tokens, temperature)
        except Exception as exc:
            response = LLMResponse(
                model=self.primary.model,
                usage=Usage(provider=self.primary.name, model=self.primary.model),
                error=f"backend error: {exc}",
            )
        if not response.error:
            return response
        fallback = self.fallback.complete(messages, system, max_tokens, temperature)
        fallback.raw = {
            "fallback": True,
            "primary_error": response.error,
            "primary_raw": response.raw,
        }
        return fallback

    def complete_stream(
        self, messages, system="", max_tokens=None, temperature=None, tools=None
    ) -> Iterator[StreamEvent]:
        parts: list[str] = []
        final: LLMResponse | None = None
        try:
            if tools:
                events = self.primary.complete_stream(
                    messages, system, max_tokens, temperature, tools
                )
            else:
                events = self.primary.complete_stream(messages, system, max_tokens, temperature)
            for event in events:
                if event.text:
                    parts.append(event.text)
                if event.done:
                    final = event.response
                yield event
        except Exception as exc:
            final = LLMResponse(
                model=self.primary.model,
                usage=Usage(provider=self.primary.name, model=self.primary.model),
                error=f"backend error: {exc}",
            )
        if final is None:
            final = LLMResponse(
                model=self.primary.model,
                usage=Usage(provider=self.primary.name, model=self.primary.model),
                text="".join(parts),
                error="stream ended without a result" if not parts else "",
            )
        if final.error and not parts:
            yield from self.fallback.complete_stream(
                messages, system, max_tokens, temperature, tools=None
            )
