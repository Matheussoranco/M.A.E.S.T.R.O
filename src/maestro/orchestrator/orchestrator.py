"""The Orchestrator — turn a spec into a live swarm and conduct it.

This is the top-level entry point most callers use::

    from maestro import Orchestrator
    orch = Orchestrator.from_file("examples/research_swarm.yaml")
    result = orch.run("Summarize the state of markerless biomechanics.")
    print(result.final)
"""

from __future__ import annotations

from collections.abc import Callable

from maestro.agents.registry import build_agent
from maestro.config.settings import settings as default_settings
from maestro.orchestrator.spec import SwarmSpec
from maestro.swarm.swarm import Swarm
from maestro.topologies import build_topology
from maestro.topologies.base import SwarmResult


class Orchestrator:
    """Builds and runs swarms from :class:`SwarmSpec` objects."""

    def __init__(self, spec: SwarmSpec, settings=None) -> None:
        self.spec = spec
        self.settings = settings or default_settings
        problems = spec.validate()
        if problems:
            raise ValueError("invalid swarm spec:\n  - " + "\n  - ".join(problems))
        self.swarm = self._build()

    # -- constructors ---------------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict, settings=None) -> Orchestrator:
        return cls(SwarmSpec.from_dict(data), settings=settings)

    @classmethod
    def from_file(cls, path: str, settings=None) -> Orchestrator:
        return cls(SwarmSpec.from_file(path), settings=settings)

    # -- build & run ----------------------------------------------------------
    def _build(self) -> Swarm:
        agents = [build_agent(ag, self.spec.providers, self.settings) for ag in self.spec.agents]
        topology = build_topology(self.spec.topology, **self.spec.topology_params)
        return Swarm(
            name=self.spec.name,
            agents=agents,
            topology=topology,
            prices=self.spec.prices,
        )

    def run(
        self,
        task: str,
        trace: bool = True,
        on_token: Callable[[str, str], None] | None = None,
        stream: bool | None = None,
    ) -> SwarmResult:
        return self.swarm.run(task, trace=trace, on_token=on_token, stream=stream)

    # -- introspection --------------------------------------------------------
    def describe(self) -> str:
        lines = [
            f"swarm:     {self.spec.name}",
            f"topology:  {self.spec.topology} {self.spec.topology_params or ''}".rstrip(),
            f"agents:    {len(self.swarm.agents)}",
        ]
        for a in self.swarm.agents:
            flag = "✓" if a.available else "✗"
            lines.append(f"  {flag} {a.name:<14} [{a.kind}] {a.role or a.description}")
        return "\n".join(lines)

    def availability(self) -> dict[str, bool]:
        return self.swarm.availability()
