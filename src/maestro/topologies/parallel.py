"""Parallel (fan-out / map-reduce) topology.

Every worker tackles the same task concurrently; an optional *aggregator* agent
then reduces their outputs into one answer.  With no aggregator the worker
outputs are concatenated, labelled by agent.  Good for ensembles, brainstorming,
and self-consistency.
"""

from __future__ import annotations

import concurrent.futures
import time
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
            "topology",
            name=self.name,
            detail=f"{len(workers)} workers" + (" + aggregator" if aggregator else ""),
        )

        max_workers = max(1, min(len(workers), self.params.get("max_workers", 8)))
        results: list[AgentResult] = []
        # Every worker gets the same initial snapshot.  Their writes are merged
        # only after all workers finish, so fan-out remains independent and
        # reproducible while the aggregator still sees the complete transcript.
        initial = context.blackboard.snapshot()
        message_offset = len(initial.messages())
        completed_contexts = []
        pool = ThreadPoolExecutor(max_workers=max_workers)
        try:
            jobs = []
            for w in workers:
                worker_context = context.child(blackboard=initial.snapshot())
                timeout = float(self.params.get("worker_timeout", getattr(w, "timeout", 300.0)))
                jobs.append(
                    (
                        w,
                        worker_context,
                        pool.submit(w.run, task, worker_context),
                        time.monotonic() + timeout,
                    )
                )
            for worker, worker_context, fut, deadline in jobs:
                # Bound each worker by its own timeout so one hung backend
                # cannot stall the whole swarm; exceptions become error results.
                remaining = max(0.0, deadline - time.monotonic())
                try:
                    results.append(fut.result(timeout=remaining))
                    completed_contexts.append(worker_context)
                except concurrent.futures.TimeoutError:
                    fut.cancel()
                    results.append(
                        AgentResult(
                            name=worker.name,
                            role=getattr(worker, "role", ""),
                            error=f"worker timed out after {remaining:.1f}s",
                        )
                    )
                except Exception as exc:  # never let one worker kill the swarm
                    results.append(
                        AgentResult(
                            name=worker.name,
                            role=getattr(worker, "role", ""),
                            error=f"worker raised {type(exc).__name__}: {exc}",
                        )
                    )
        finally:
            # __exit__ waits for running workers, defeating result(timeout=...).
            pool.shutdown(wait=False, cancel_futures=True)
        for worker_context in completed_contexts:
            context.blackboard.merge_from(worker_context.blackboard, message_offset)

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
            task=task,
            final=final,
            topology=self.name,
            per_agent=results,
            tracer=context.tracer,
            error=error,
        )
