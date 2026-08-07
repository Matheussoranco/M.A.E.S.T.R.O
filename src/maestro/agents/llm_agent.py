"""A native LLM-backed agent, with a two-track tool loop.

Tools are declared once (see :mod:`maestro.tools.schema`) and executed down
whichever track the backend supports:

**Native function calling** — used when the client sets ``supports_tools``.
The tool schemas are sent with the request, the model replies with structured
tool calls, and results go back as protocol-level tool results.

**ReAct fallback** — used for every other backend.  A plain-text convention that
works anywhere::

    ACTION: <tool_name> <single-line argument>
    ...model then receives...
    OBSERVATION: <tool result>

When the model stops asking for tools, its text is the final answer.  Agents
with no tools are a single completion — the common case.

Streaming is orthogonal to both: when the run context asks for it, fragments are
forwarded to the context's token sink as they arrive while the agent still
returns one complete :class:`AgentResult`, so topologies aggregate unchanged.
"""

from __future__ import annotations

import re

from maestro.agents.base import Agent, AgentResult
from maestro.providers.base import LLMClient, LLMResponse
from maestro.swarm.context import RunContext
from maestro.telemetry.usage import Usage
from maestro.tools.base import Tool
from maestro.tools.schema import to_react_catalogue

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
        native_tools: bool | None = None,
    ) -> None:
        super().__init__(name=name, role=role, description=description)
        self.client = client
        self.system_prompt = system_prompt or _default_system(role)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.tools = {t.name: t for t in (tools or [])}
        self.max_tool_iters = max_tool_iters
        self.use_context = use_context
        # None ⇒ follow the backend's capability; True/False ⇒ force a track,
        # which is useful for comparing them on a provider that supports both.
        self.native_tools = native_tools

    @property
    def available(self) -> bool:
        return self.client.available

    def uses_native_tools(self) -> bool:
        """Whether this agent will take the native function-calling track."""
        if not self.tools:
            return False
        if self.native_tools is not None:
            return self.native_tools
        return bool(getattr(self.client, "supports_tools", False))

    # -- prompt assembly ------------------------------------------------------
    def _system(self, native: bool) -> str:
        sys = self.system_prompt
        if self.tools and not native:
            sys += (
                "\n\nYou may call tools.  To call one, emit a line exactly:\n"
                "ACTION: <tool_name> <single-line argument>\n"
                "You will then receive an OBSERVATION line.  When you have the "
                "answer, reply normally with no ACTION line.\n"
                f"Available tools:\n{to_react_catalogue(list(self.tools.values()))}"
            )
        return sys

    def _build_messages(self, task: str, context: RunContext | None) -> list[dict]:
        parts = []
        if self.use_context and context is not None:
            digest = context.context_digest()
            if digest:
                parts.append(digest)
        parts.append(f"Task: {task}")
        return [{"role": "user", "content": "\n\n".join(parts)}]

    # -- backend call ---------------------------------------------------------
    def _call(
        self,
        messages: list[dict],
        system: str,
        context: RunContext | None,
        usage: list[Usage],
        tools: list[Tool] | None = None,
    ) -> LLMResponse:
        """One backend call, streamed or not, with its usage recorded."""
        kwargs: dict = {
            "system": system,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        # Only pass `tools` when there are some: clients written against the 0.1
        # signature (no `tools` parameter) must keep working untouched.
        if tools:
            kwargs["tools"] = tools

        if context is not None and context.stream:
            response: LLMResponse | None = None
            parts: list[str] = []
            for event in self.client.complete_stream(messages, **kwargs):
                if event.text:
                    parts.append(event.text)
                    context.emit_token(self.name, event.text)
                if event.done:
                    response = event.response
            if response is None:  # provider ended without a terminal event
                text = "".join(parts)
                response = LLMResponse(
                    text=text, error="" if text else "stream ended without a result"
                )
        else:
            response = self.client.complete(messages, **kwargs)

        usage.append(response.usage)
        if context is not None:
            # Also record against the run ledger: this agent's result may be
            # discarded by the topology (a supervisor's plan, a router's pick)
            # but the call was still billed.
            context.record_usage(self.name, response.usage)
        return response

    # -- run ------------------------------------------------------------------
    def run(self, task: str, context: RunContext | None = None) -> AgentResult:
        if context is not None:
            context.tracer.emit("agent_start", name=self.name, detail=self.role)
        messages = self._build_messages(task, context)
        usage: list[Usage] = []
        if self.uses_native_tools():
            return self._run_native(messages, context, usage)
        return self._run_react(messages, context, usage)

    def _meta(self, response: LLMResponse, protocol: str, **extra) -> dict:
        meta = {"model": response.model, "tool_protocol": protocol}
        meta.update(extra)
        return meta

    def _run_native(
        self, messages: list[dict], context: RunContext | None, usage: list[Usage]
    ) -> AgentResult:
        system = self._system(native=True)
        tools = list(self.tools.values())
        text = ""
        response: LLMResponse | None = None
        for _ in range(self.max_tool_iters):
            response = self._call(messages, system, context, usage, tools=tools)
            if response.error:
                return self._finish(
                    context,
                    AgentResult(self.name, self.role, error=response.error, usage=usage),
                )
            text = response.text
            if not response.tool_calls:
                return self._finish(
                    context,
                    AgentResult(
                        self.name,
                        self.role,
                        output=text,
                        meta=self._meta(response, "native"),
                        usage=usage,
                    ),
                )
            # Record the model's turn, then answer every call it made — all of
            # them, before the next request.
            messages.append(
                {"role": "assistant", "content": text, "tool_calls": response.tool_calls}
            )
            for call in response.tool_calls:
                tool = self.tools.get(call.name)
                observation = (
                    tool.call(call.arguments)
                    if tool is not None
                    else f"error: unknown tool {call.name!r}"
                )
                if context is not None:
                    context.tracer.emit("tool_call", name=call.name, detail=call.summary())
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.name,
                        "content": observation,
                    }
                )

        return self._finish(
            context,
            AgentResult(
                self.name,
                self.role,
                output=text,
                meta=self._meta(response, "native", note="max_tool_iters reached")
                if response
                else {"note": "max_tool_iters reached"},
                usage=usage,
            ),
        )

    def _run_react(
        self, messages: list[dict], context: RunContext | None, usage: list[Usage]
    ) -> AgentResult:
        system = self._system(native=False)
        text = ""
        response: LLMResponse | None = None
        for _ in range(self.max_tool_iters if self.tools else 1):
            response = self._call(messages, system, context, usage)
            if response.error:
                return self._finish(
                    context,
                    AgentResult(self.name, self.role, error=response.error, usage=usage),
                )
            text = response.text
            match = _ACTION_RE.search(text) if self.tools else None
            if not match:
                return self._finish(
                    context,
                    AgentResult(
                        self.name,
                        self.role,
                        output=text,
                        meta=self._meta(response, "react"),
                        usage=usage,
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
            AgentResult(
                self.name,
                self.role,
                output=text,
                meta=self._meta(response, "react", note="max_tool_iters reached")
                if response
                else {"note": "max_tool_iters reached"},
                usage=usage,
            ),
        )


def _default_system(role: str) -> str:
    role = role or "assistant"
    return (
        f"You are a focused member of an agent swarm, acting as the {role}. "
        "Contribute your part concisely and defer coordination to the swarm. "
        "Build on prior contributions rather than repeating them."
    )
