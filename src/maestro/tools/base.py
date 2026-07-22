"""Tool abstraction for the LLM agent's ReAct loop.

Tools take a single-line string argument and return a string observation.  That
keeps them provider-agnostic and trivially serializable across the plain-text
tool protocol used by :class:`~maestro.agents.llm_agent.LLMAgent`.
"""

from __future__ import annotations

import abc
from collections.abc import Callable


class Tool(abc.ABC):
    name: str = "tool"
    description: str = ""

    @abc.abstractmethod
    def run(self, arg: str) -> str:
        """Execute the tool on *arg* and return an observation string."""


class FunctionTool(Tool):
    """Wrap a plain ``str -> str`` callable as a tool."""

    def __init__(self, name: str, description: str, fn: Callable[[str], str]) -> None:
        self.name = name
        self.description = description
        self._fn = fn

    def run(self, arg: str) -> str:
        try:
            return str(self._fn(arg))
        except Exception as exc:
            return f"error: {exc}"
