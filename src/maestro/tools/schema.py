"""One tool declaration, adapted to each provider's function-calling dialect.

A :class:`~maestro.tools.base.Tool` is declared **once** — name, description and
a JSON Schema for its arguments.  The adapters here translate that single
declaration into whatever shape a backend wants:

======================  ======================================================
Anthropic Messages API  ``{"name", "description", "input_schema"}``
OpenAI chat-completions ``{"type": "function", "function": {…, "parameters"}}``
Ollama ``/api/chat``    the OpenAI ``function`` shape
ReAct fallback          a plain-text catalogue in the system prompt
======================  ======================================================

Nothing is duplicated: add a tool and every provider gets it, including the
text-only fallback used by backends with no native support.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from maestro.tools.base import Tool


@dataclass
class ToolCall:
    """A provider-agnostic request from the model to run one tool.

    ``id`` is whatever correlation handle the backend uses (Anthropic's
    ``toolu_…``, OpenAI's ``call_…``); it is echoed back untouched with the
    result.  Ollama sends no id, so one is synthesized.
    """

    name: str
    arguments: dict = field(default_factory=dict)
    id: str = ""

    def summary(self, width: int = 80) -> str:
        """Short one-line form, for tracing."""
        try:
            args = json.dumps(self.arguments, ensure_ascii=False)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            args = str(self.arguments)
        return f"{self.name}({args})"[:width]


def parse_arguments(raw: object) -> dict:
    """Coerce a provider's argument payload into a dict.

    OpenAI streams arguments as a JSON *string*, Anthropic and Ollama send an
    object.  Anything unparseable is preserved under ``"input"`` so the tool
    still receives the model's intent instead of an exception.
    """
    if isinstance(raw, dict):
        return raw
    if raw is None:
        return {}
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {"input": raw}
        return parsed if isinstance(parsed, dict) else {"input": parsed}
    return {"input": raw}


def to_anthropic(tools: list[Tool]) -> list[dict]:
    """Anthropic Messages API ``tools`` array."""
    return [
        {"name": t.name, "description": t.description, "input_schema": t.schema()} for t in tools
    ]


def to_openai(tools: list[Tool]) -> list[dict]:
    """OpenAI (and OpenAI-compatible) ``tools`` array."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.schema(),
            },
        }
        for t in tools
    ]


#: Ollama's ``/api/chat`` accepts the OpenAI ``function`` shape verbatim.
to_ollama = to_openai


def to_react_catalogue(tools: list[Tool]) -> str:
    """The plain-text catalogue used by the ReAct fallback convention."""
    return "\n".join(f"- {t.name}: {t.description}" for t in tools)


__all__ = [
    "ToolCall",
    "parse_arguments",
    "to_anthropic",
    "to_ollama",
    "to_openai",
    "to_react_catalogue",
]
