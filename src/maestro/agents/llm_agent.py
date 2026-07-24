"""A native LLM-backed agent, with an optional lightweight tool loop.

The tool protocol is a plain-text ReAct convention so it works across *every*
backend (no provider-specific function-calling required):

    ACTION: <tool_name> <single-line argument>
    ...model then receives...
    OBSERVATION: <tool result>

When the model stops emitting ``ACTION:`` lines, its text is the final answer.
Agents with no tools are a single completion — the common case.
"""

from __future__ import annotations

import re

from maestro.agents.base import Agent, AgentResult
from maestro.providers.base import LLMClient
from maestro.swarm.context import RunContext
from maestro.tools.base import Tool

_ACTION_RE = re.compile(r"^\s*ACTION:\s*(\S+)\s*(.*)$", re.IGNORECASE | re.MULTILINE)


class LLMAgent(Agent):
    """Role-typed agent backed by any :class:`LLMClient`."""

    kind = "llm"

    def __init__(
        self,
        name: str,
        client: LLMClient,
        role: str = "",
        system_prompt: str = "",
        description: str = "",
        max_tokens: int | None = None,
        temperature: float | None = None,
        tools: list[Tool] | None = None,
        max_tool_iters: int = 4,
        use_context: bool = True,
    ) -> None:
        super().__init__(name=name, role=role, description=description)
        self.client = client
        self.system_prompt = system_prompt or _default_system(role)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.tools = {t.name: t for t in (tools or [])}
        self.max_tool_iters = max_tool_iters
        self.use_context = use_context

    @property
    def available(self) -> bool:
        return self.client.available

    def _system(self) -> str:
        sys = self.system_prompt
        if self.tools:
            catalogue = "\n".join(f"- {t.name}: {t.description}" for t in self.tools.values())
            sys += (
                "\n\nYou may call tools.  To call one, emit a line exactly:\n"
                "ACTION: <tool_name> <single-line argument>\n"
                "You will then receive an OBSERVATION line.  When you have the "
                "answer, reply normally with no ACTION line.\n"
                f"Available tools:\n{catalogue}"
            )
        return sys

    def _build_messages(self, task: str, context: RunContext | None) -> list[dict[str, str]]:
        parts = []
        if self.use_context and context is not None:
            digest = context.context_digest()
            if digest:
                parts.append(digest)
        parts.append(f"Task: {task}")
        return [{"role": "user", "content": "\n\n".join(parts)}]

    def run(self, task: str, context: RunContext | None = None) -> AgentResult:
        if context is not None:
            context.tracer.emit("agent_start", name=self.name, detail=self.role)
        messages = self._build_messages(task, context)
        system = self._system()

        for _ in range(self.max_tool_iters if self.tools else 1):
            resp = self.client.complete(
                messages, system=system, max_tokens=self.max_tokens, temperature=self.temperature
            )
            if resp.error:
                return self._finish(context, AgentResult(self.name, self.role, error=resp.error))
            text = resp.text
            match = _ACTION_RE.search(text) if self.tools else None
            if not match:
                return self._finish(
                    context,
                    AgentResult(
                        self.name,
                        self.role,
                        output=text,
                        meta={"model": resp.model, "usage": resp.usage},
                    ),
                )
            # Execute the requested tool and feed the observation back.
            tool_name, arg = match.group(1), match.group(2).strip()
            tool = self.tools.get(tool_name)
            observation = tool.run(arg) if tool else f"error: unknown tool {tool_name!r}"
            if context is not None:
                context.tracer.emit("tool_call", name=tool_name, detail=arg[:80])
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": f"OBSERVATION: {observation}"})

        # Ran out of tool iterations — return whatever the last turn produced.
        return self._finish(
            context,
            AgentResult(self.name, self.role, output=text, meta={"note": "max_tool_iters reached"}),
        )


def _default_system(role: str) -> str:
    role = role or "assistant"
    return (
        f"You are a focused member of an agent swarm, acting as the {role}. "
        "Contribute your part concisely and defer coordination to the swarm. "
        "Build on prior contributions rather than repeating them."
    )
