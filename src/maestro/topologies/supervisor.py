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

import concurrent.futures
import json
from concurrent.futures import ThreadPoolExecutor

from maestro.agents.base import Agent, AgentResult
from maestro.swarm.context import RunContext
from maestro.topologies.base import SwarmResult, Topology


def _extract_json_array(text: str) -> list | None:
    """Extrai o primeiro array JSON válido via JSONDecoder.raw_decode.

    Substitui o antigo ``re.compile(r"\\[.*\\]", re.DOTALL)`` greedy, que
    casava do primeiro ``[`` até o ÚLTIMO ``]`` do texto — unindo arrays
    distintos, engolindo texto posterior e falhando no json.loads.
    Aqui cada candidato ``[`` é tentado com raw_decode (balanceado,
    string-aware); o primeiro que decodifica para list é retornado.
    """
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "[":
            continue
        try:
            obj, _ = decoder.raw_decode(text[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, list):
            return obj
    return None


class SupervisorTopology(Topology):
    name = "supervisor"

    def run(self, task: str, agents: list[Agent], context: RunContext) -> SwarmResult:
        supervisor = self._pick_supervisor(agents)
        workers = [a for a in agents if a.name != supervisor.name]
        if not workers:
            res = supervisor.run(task, context)
            return SwarmResult(
                task=task,
                final=res.output or res.error,
                topology=self.name,
                per_agent=[res],
                tracer=context.tracer,
                error="" if res.ok() else "supervisor produced no output",
            )

        context.tracer.emit(
            "topology",
            name=self.name,
            detail=f"supervisor={supervisor.name}, {len(workers)} workers",
        )

        assignments = self._plan(task, supervisor, workers, context)
        results: list[AgentResult] = []

        max_workers = max(1, min(len(assignments), self.params.get("max_workers", 8)))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            # Workers must all see the same post-planning snapshot.  Otherwise
            # completion order changes the prompts and therefore the synthesis.
            initial = context.blackboard.snapshot()
            message_offset = len(initial.messages())
            worker_contexts = []
            futures = []
            for worker, subtask in assignments:
                worker_context = context.child(blackboard=initial.snapshot())
                worker_contexts.append(worker_context)
                futures.append((worker, pool.submit(worker.run, subtask, worker_context)))
            for worker, fut in futures:
                # Bound each worker by its own timeout so one hung backend
                # cannot stall the whole swarm; exceptions become error results.
                timeout = getattr(worker, "timeout", None)
                if timeout is None:
                    timeout = self.params.get("worker_timeout", 300.0)
                try:
                    results.append(fut.result(timeout=timeout))
                except concurrent.futures.TimeoutError as exc:
                    results.append(
                        AgentResult(
                            name=worker.name,
                            role=getattr(worker, "role", ""),
                            error=f"worker timed out after {timeout}s: {exc}",
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
            for worker_context in worker_contexts:
                context.blackboard.merge_from(worker_context.blackboard, message_offset)

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
            task=task,
            final=final,
            topology=self.name,
            per_agent=results,
            tracer=context.tracer,
            error="" if final.strip() else "no output produced",
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
        items = _extract_json_array(plan_res.output or "")
        if items is not None:
            try:
                for item in items:
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
