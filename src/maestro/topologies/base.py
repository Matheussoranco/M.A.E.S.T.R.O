"""Topology abstraction — how a set of agents is composed into a swarm.

A *topology* owns the control-flow: who runs, in what order, who sees whose
output, and how the pieces are combined into a final answer.  Swapping the
topology (sequential → supervisor → debate) changes the swarm's behaviour
without touching the agents themselves.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

from maestro.agents.base import Agent, AgentResult
from maestro.swarm.context import RunContext
from maestro.telemetry.tracer import Tracer


@dataclass
class SwarmResult:
    task: str = ""
    final: str = ""
    topology: str = ""
    per_agent: list[AgentResult] = field(default_factory=list)
    error: str = ""
    tracer: Tracer | None = None

    def ok(self) -> bool:
        return not self.error and bool(self.final.strip())

    def summary(self) -> str:
        lines = [f"topology: {self.topology}", f"agents:   {len(self.per_agent)}"]
        for r in self.per_agent:
            status = "ok " if r.ok() else "ERR"
            preview = (r.output or r.error).replace("\n", " ")[:80]
            lines.append(f"  [{status}] {r.name:<14} {preview}")
        return "\n".join(lines)


class Topology(abc.ABC):
    name: str = "base"

    def __init__(self, **params) -> None:
        self.params = params

    @abc.abstractmethod
    def run(self, task: str, agents: list[Agent], context: RunContext) -> SwarmResult:
        """Execute the swarm over *agents* for *task* and return the result."""

    # Shared helper.
    @staticmethod
    def _worker_agents(agents: list[Agent], exclude: set[str]) -> list[Agent]:
        return [a for a in agents if a.name not in exclude]
