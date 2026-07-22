"""Provider abstraction — the single interface every LLM backend implements.

The contract is intentionally tiny (one method, :meth:`LLMClient.complete`) so a
new backend is a couple of dozen lines.  Two backends here need no network at
all: :class:`NullClient` (always unavailable — forces symbolic fallback) and
:class:`EchoClient` (deterministic, so swarm control-flow is testable offline).
"""

from __future__ import annotations

import abc
import hashlib
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMResponse:
    """A single completion returned by a backend."""

    text: str = ""
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    raw: Any = None
    error: str = ""

    def ok(self) -> bool:
        """True when the call produced usable, non-empty text."""
        return not self.error and bool(self.text.strip())


class LLMClient(abc.ABC):
    """Minimal chat-completion interface shared by every provider."""

    #: Short, stable identifier for the backend (``"anthropic"``, ``"ollama"``…).
    name: str = "base"
    #: The concrete model this client talks to.
    model: str = ""

    @property
    @abc.abstractmethod
    def available(self) -> bool:
        """Cheap readiness check — must not perform a network round-trip."""

    @abc.abstractmethod
    def complete(
        self,
        messages: list[dict[str, str]],
        system: str = "",
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        """Run one completion over ``[{'role': 'user'|'assistant', 'content': …}]``."""

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

    def complete(self, messages, system="", max_tokens=None, temperature=None) -> LLMResponse:
        return LLMResponse(model="null", error="no LLM backend configured")


class EchoClient(LLMClient):
    """Deterministic offline backend.

    Produces a stable, human-readable stand-in for a completion.  This lets the
    *structure* of any swarm — routing, fan-out, aggregation, debate rounds — be
    exercised end-to-end with no keys and no network, and makes the whole test
    suite deterministic.  It is also the graceful-degradation target when a real
    provider is unavailable.
    """

    name = "echo"

    def __init__(self, model: str = "echo-1", persona: str = "") -> None:
        self.model = model
        self.persona = persona

    @property
    def available(self) -> bool:
        return True

    def complete(self, messages, system="", max_tokens=None, temperature=None) -> LLMResponse:
        last_user = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user = str(msg.get("content", ""))
                break
        digest = hashlib.sha1(
            (system + "\n" + last_user).encode("utf-8", "replace")
        ).hexdigest()[:8]
        who = self.persona or "echo"
        # A compact, deterministic "answer" that quotes the request back so the
        # reader can see the request actually reached this agent.
        snippet = " ".join(last_user.split())[:280]
        text = f"[{who}#{digest}] {snippet}" if snippet else f"[{who}#{digest}] (no input)"
        return LLMResponse(
            text=text,
            model=self.model,
            usage={"prompt_tokens": len(last_user.split()), "completion_tokens": len(text.split())},
        )
