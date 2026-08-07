"""A ready-to-run swarm: agents + a topology + shared memory."""

from __future__ import annotations

from collections.abc import Callable

from maestro.agents.base import Agent
from maestro.swarm.context import RunContext
from maestro.telemetry.tracer import Tracer
from maestro.topologies.base import SwarmResult, Topology


class Swarm:
    """A named collection of agents conducted by a :class:`Topology`."""

    def __init__(self, name: str, agents: list[Agent], topology: Topology) -> None:
        if not agents:
            raise ValueError("a swarm needs at least one agent")
        self.name = name
        self.agents = agents
        self.topology = topology

    def run(
        self,
        task: str,
        trace: bool = True,
        on_token: Callable[[str, str], None] | None = None,
        stream: bool | None = None,
    ) -> SwarmResult:
        """Conduct the swarm over *task*.

        Pass *on_token* to receive streamed fragments as ``(agent_name, text)``
        while the run proceeds; the returned :class:`SwarmResult` is identical
        either way, so streaming never changes what a topology aggregates.
        Streaming is enabled implicitly by passing a sink, or explicitly with
        *stream* (useful to exercise the path with no consumer).
        """
        context = RunContext(
            task=task,
            tracer=Tracer(enabled=trace),
            stream=bool(on_token) if stream is None else stream,
            on_token=on_token,
        )
        context.tracer.emit("swarm_start", name=self.name, detail=self.topology.name)
        context.post("user", task, role="user")
        result = self.topology.run(task, self.agents, context)
        context.tracer.emit("swarm_end", name=self.name, detail=("ok" if result.ok() else "error"))
        result.tracer = context.tracer
        result.usage_log = context.usage_log
        return result

    def agent_names(self) -> list[str]:
        return [a.name for a in self.agents]

    def availability(self) -> dict[str, bool]:
        return {a.name: a.available for a in self.agents}

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"Swarm(name={self.name!r}, agents={self.agent_names()}, "
            f"topology={self.topology.name!r})"
        )
