"""Tool abstraction, usable from both tool-calling paths.

A tool is declared once and works two ways:

* **Native function calling** — :meth:`Tool.schema` yields the JSON Schema a
  provider advertises, and :meth:`Tool.call` receives the model's parsed
  arguments dict.
* **ReAct fallback** — :meth:`Tool.run` takes the single-line string argument of
  the plain-text protocol, which every backend can produce.

The default schema is a single required string called ``input``, so the simplest
tools need declare nothing extra and behave identically down both paths.  Give a
tool a richer :attr:`Tool.parameters` schema and override :meth:`Tool.call` when
it needs real structured arguments.
"""

from __future__ import annotations

import abc
import json
from collections.abc import Callable

#: The argument name used by the default single-string schema.
DEFAULT_ARG = "input"


class Tool(abc.ABC):
    name: str = "tool"
    description: str = ""
    #: JSON Schema for the arguments.  ``None`` ⇒ the single-string default.
    parameters: dict | None = None

    def schema(self) -> dict:
        """The JSON Schema advertised to native function-calling backends."""
        if self.parameters:
            return self.parameters
        return {
            "type": "object",
            "properties": {
                DEFAULT_ARG: {
                    "type": "string",
                    "description": self.description or "The argument for this tool.",
                }
            },
            "required": [DEFAULT_ARG],
        }

    @abc.abstractmethod
    def run(self, arg: str) -> str:
        """Execute the tool on *arg* and return an observation string."""

    def call(self, arguments: dict) -> str:
        """Execute the tool from a native function call's arguments.

        The default collapses the arguments to the single string
        :meth:`run` expects: the ``input`` key if present, else a lone value,
        else the whole payload as JSON.  Tools with a real multi-argument schema
        should override this instead of parsing JSON inside :meth:`run`.
        """
        if not isinstance(arguments, dict):
            return self.run(str(arguments))
        if DEFAULT_ARG in arguments:
            return self.run(str(arguments[DEFAULT_ARG]))
        if len(arguments) == 1:
            return self.run(str(next(iter(arguments.values()))))
        if not arguments:
            return self.run("")
        try:
            return self.run(json.dumps(arguments, ensure_ascii=False))
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return self.run(str(arguments))


class FunctionTool(Tool):
    """Wrap a plain ``str -> str`` callable as a tool."""

    def __init__(
        self,
        name: str,
        description: str,
        fn: Callable[[str], str],
        parameters: dict | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters
        self._fn = fn

    def run(self, arg: str) -> str:
        try:
            return str(self._fn(arg))
        except Exception as exc:
            return f"error: {exc}"
