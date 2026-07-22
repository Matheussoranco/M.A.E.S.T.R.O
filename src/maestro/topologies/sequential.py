"""Sequential (pipeline) topology.

Agents run one after another.  Each sees the accumulated blackboard, so agent N
builds on the output of agents 1…N-1.  The last agent's output is the final
answer.  Ideal for draft → critique → polish style pipelines.
"""

from __future__ import annotations

from maestro.agents.base import Agent
from maestro.swarm.context import RunContext
from maestro.topologies.base import SwarmResult, Topology


class SequentialTopology(Topology):
    name = "sequential"

    def run(self, task: str, agents: list[Agent], context: RunContext) -> SwarmResult:
        context.tracer.emit("topology", name=self.name, detail=f"{len(agents)} stages")
        results = []
        last_output = ""
        for i, agent in enumerate(agents):
            stage_task = task if i == 0 else self.params.get("relay_task", task)
            res = agent.run(stage_task, context)
            results.append(res)
            if res.ok():
                last_output = res.output
        final = last_output or (results[-1].error if results else "no agents ran")
        return SwarmResult(
            task=task, final=final, topology=self.name,
            per_agent=results, tracer=context.tracer,
            error="" if last_output else "pipeline produced no output",
        )
