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
from maestro.telemetry.usage import Usage, UsageTotals


@dataclass
class SwarmResult:
    task: str = ""
    final: str = ""
    topology: str = ""
    per_agent: list[AgentResult] = field(default_factory=list)
    error: str = ""
    tracer: Tracer | None = None
    #: Every backend call made during the run, as ``(agent_name, usage)``,
    #: attached by :meth:`Swarm.run` from the run context's ledger.
    usage_log: list[tuple[str, Usage]] = field(default_factory=list)
    #: Prices declared by the spec that produced this result.  Kept local to
    #: the result so constructing one swarm cannot mutate another swarm's bill.
    prices: dict[str, tuple[float, float]] = field(default_factory=dict)

    def ok(self) -> bool:
        return not self.error and bool(self.final.strip())

    # -- token & cost accounting ---------------------------------------------
    def usage_totals(self, prices: dict[str, tuple[float, float]] | None = None) -> UsageTotals:
        """Roll the run's backend calls up into one token/cost report.

        Reads the run ledger (:attr:`usage_log`) when there is one, so calls
        whose result the topology discarded — a supervisor's planning turn, a
        router's routing turn — are still counted; they were still billed.
        Falls back to the per-agent results for a :class:`SwarmResult` built by
        hand.  Pass *prices* to override or extend the built-in price table for
        this call only.
        """
        by_agent: dict[str, list[Usage]] = {}
        if self.usage_log:
            for name, usage in self.usage_log:
                by_agent.setdefault(name, []).append(usage)
        else:
            for result in self.per_agent:
                by_agent.setdefault(result.name, []).extend(result.usage)
        return UsageTotals.from_agent_usage(by_agent, self.prices if prices is None else prices)

    @property
    def usage(self) -> UsageTotals:
        """The run's token/cost totals, broken down per agent and per provider."""
        return self.usage_totals()

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
