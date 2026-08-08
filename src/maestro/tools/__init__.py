"""Tool registry.

``build_tools(["calc", "now"])`` resolves tool names (from a spec) into live
:class:`Tool` instances.  Unknown names raise so typos surface early.
"""

from __future__ import annotations

from maestro.tools.base import FunctionTool, Tool
from maestro.tools.builtin import builtin_tools
from maestro.tools.schema import (
    ToolCall,
    parse_arguments,
    to_anthropic,
    to_ollama,
    to_openai,
    to_react_catalogue,
)


def build_tools(names: list[str] | None) -> list[Tool]:
    if not names:
        return []
    registry = builtin_tools()
    out: list[Tool] = []
    for name in names:
        if name not in registry:
            raise ValueError(f"unknown tool {name!r}; available: {', '.join(sorted(registry))}")
        out.append(registry[name])
    return out


def available_tools() -> list[str]:
    return sorted(builtin_tools())


__all__ = [
    "FunctionTool",
    "Tool",
    "ToolCall",
    "available_tools",
    "build_tools",
    "parse_arguments",
    "to_anthropic",
    "to_ollama",
    "to_openai",
    "to_react_catalogue",
]
