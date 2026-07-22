"""The agent contract every swarm member honours.

An *agent* is anything that can take a task string plus a shared
:class:`~maestro.swarm.context.RunContext` and return an :class:`AgentResult`.
That deliberately loose contract is what lets a native LLM agent, an external
CLI program, an MCP server, or a whole sibling project all sit side-by-side in
the same swarm.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

from maestro.swarm.context import RunContext


@dataclass
class AgentResult:
    name: str
    role: str = ""
    output: str = ""
    error: str = ""
    meta: dict = field(default_factory=dict)

    def ok(self) -> bool:
        return not self.error and bool(self.output.strip())


class Agent(abc.ABC):
    """Base class for all swarm members."""

    #: The blueprint type used to build this agent (``"llm"``, ``"cli"``, …).
    kind: str = "base"

    def __init__(self, name: str, role: str = "", description: str = "") -> None:
        self.name = name
        self.role = role
        self.description = description

    @property
    def available(self) -> bool:
        """Whether this agent can actually run right now (key present, cmd found…)."""
        return True

    @abc.abstractmethod
    def run(self, task: str, context: RunContext | None = None) -> AgentResult:
        """Execute *task* and return a result.  Must never raise for expected
        failures — capture them in :attr:`AgentResult.error` instead."""

    # Shared helper so every agent records itself on the blackboard uniformly.
    def _finish(self, context: RunContext | None, result: AgentResult) -> AgentResult:
        if context is not None:
            if result.ok():
                context.post(self.name, result.output, role="assistant", role_name=self.role)
            context.tracer.emit(
                "agent_end",
                name=self.name,
                detail=(result.error or f"{len(result.output)} chars"),
            )
        return result

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"{type(self).__name__}(name={self.name!r}, role={self.role!r})"
