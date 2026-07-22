"""Supervisor (hierarchical) topology — the canonical agent-swarm pattern.

A supervisor agent decomposes the task into per-worker subtasks, the workers
execute (in parallel), and the supervisor synthesizes their results into the
final answer.  This mirrors the planner→workers→aggregator structure used by
production multi-agent systems.

Planning is *best-effort*: the supervisor is asked for a JSON assignment list.
If it returns one, it drives the swarm; if it cannot (e.g. offline echo backend),
the topology degrades to fan-out — every worker gets the original task — so the
swarm still runs and produces output.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor

from maestro.agents.base import Agent, AgentResult
from maestro.swarm.context import RunContext
from maestro.topologies.base import SwarmResult, Topology

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


class SupervisorTopology(Topology):
    name = "supervisor"

    def run(self, task: str, agents: list[Agent], context: RunContext) -> SwarmResult:
        supervisor = self._pick_supervisor(agents)
        workers = [a for a in agents if a.name != supervisor.name]
        if not workers:
            res = supervisor.run(task, context)
            return SwarmResult(task=task, final=res.output or res.error, topology=self.name,
                               per_agent=[res], tracer=context.tracer,
                               error="" if res.ok() else "supervisor produced no output")

        context.tracer.emit("topology", name=self.name,
                            detail=f"supervisor={supervisor.name}, {len(workers)} workers")

        assignments = self._plan(task, supervisor, workers, context)
        results: list[AgentResult] = []

        max_workers = max(1, min(len(assignments), self.params.get("max_workers", 8)))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = []
            for worker, subtask in assignments:
                futures.append((worker, pool.submit(worker.run, subtask, context.child())))
            for _worker, fut in futures:
                results.append(fut.result())

        # Supervisor synthesizes the workers' contributions.
        synth_prompt = (
            f"{context.context_digest(limit=len(results) + 2)}\n\n"
            f"As the supervisor, integrate the above worker contributions into a "
            f"single, coherent final answer for the original task:\n{task}"
        )
        synth = supervisor.run(synth_prompt, context)
        results.insert(0, synth)
        final = synth.output or "\n\n".join(f"[{r.name}] {r.output}" for r in results if r.ok())
        return SwarmResult(
            task=task, final=final, topology=self.name, per_agent=results,
            tracer=context.tracer, error="" if final.strip() else "no output produced",
        )

    # -- helpers --------------------------------------------------------------
    def _pick_supervisor(self, agents: list[Agent]) -> Agent:
        name = self.params.get("supervisor")
        if name:
            for a in agents:
                if a.name == name:
                    return a
        return agents[0]

    def _plan(self, task, supervisor, workers, context) -> list[tuple[Agent, str]]:
        roster = "\n".join(f"- {w.name}: {w.role or w.description or 'general'}" for w in workers)
        plan_prompt = (
            f"You are the supervisor of an agent swarm. Decompose this task into "
            f"subtasks, one per worker best suited to it. Reply with ONLY a JSON array "
            f'of objects like [{{"agent": "<name>", "subtask": "<what to do>"}}].\n\n'
            f"Workers:\n{roster}\n\nTask: {task}"
        )
        plan_res = supervisor.run(plan_prompt, context)
        by_name = {w.name.lower(): w for w in workers}
        assignments: list[tuple[Agent, str]] = []
        match = _JSON_ARRAY_RE.search(plan_res.output or "")
        if match:
            try:
                for item in json.loads(match.group(0)):
                    if not isinstance(item, dict):
                        continue
                    who = str(item.get("agent") or item.get("worker") or item.get("name") or "")
                    sub = str(item.get("subtask") or item.get("task") or task)
                    worker = by_name.get(who.lower())
                    if worker is not None:
                        assignments.append((worker, sub))
            except (json.JSONDecodeError, TypeError):
                assignments = []
        if assignments:
            context.tracer.emit("plan", name=supervisor.name, detail=f"{len(assignments)} subtasks")
            return assignments
        # Fallback — fan out the whole task to every worker.
        context.tracer.emit("plan", name=supervisor.name, detail="fan-out (no structured plan)")
        return [(w, task) for w in workers]
