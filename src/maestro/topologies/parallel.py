"""Parallel (fan-out / map-reduce) topology.

Every worker tackles the same task concurrently; an optional *aggregator* agent
then reduces their outputs into one answer.  With no aggregator the worker
outputs are concatenated, labelled by agent.  Good for ensembles, brainstorming,
and self-consistency.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from maestro.agents.base import Agent, AgentResult
from maestro.swarm.context import RunContext
from maestro.topologies.base import SwarmResult, Topology


class ParallelTopology(Topology):
    name = "parallel"

    def run(self, task: str, agents: list[Agent], context: RunContext) -> SwarmResult:
        aggregator_name = self.params.get("aggregator")
        aggregator = next((a for a in agents if a.name == aggregator_name), None)
        workers = self._worker_agents(agents, {aggregator_name} if aggregator else set())

        context.tracer.emit(
            "topology", name=self.name,
            detail=f"{len(workers)} workers" + (" + aggregator" if aggregator else ""),
        )

        max_workers = max(1, min(len(workers), self.params.get("max_workers", 8)))
        results: list[AgentResult] = []
        # Each worker gets an isolated child context view (shared blackboard) so
        # they don't prompt-contaminate each other, but still record results.
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(w.run, task, context.child()): w for w in workers
            }
            for fut in futures:
                results.append(fut.result())

        # Preserve declaration order for deterministic output.
        order = {w.name: i for i, w in enumerate(workers)}
        results.sort(key=lambda r: order.get(r.name, 0))

        if aggregator is not None:
            agg_task = self.params.get(
                "aggregator_task",
                f"Synthesize the swarm's contributions into one answer for: {task}",
            )
            agg_res = aggregator.run(agg_task, context)
            results.append(agg_res)
            final = agg_res.output or agg_res.error
            error = "" if agg_res.ok() else "aggregator produced no output"
        else:
            parts = [f"[{r.name}] {r.output}" for r in results if r.ok()]
            final = "\n\n".join(parts)
            error = "" if parts else "no worker produced output"

        return SwarmResult(
            task=task, final=final, topology=self.name,
            per_agent=results, tracer=context.tracer, error=error,
        )
