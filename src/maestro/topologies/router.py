"""Router topology — pick the single best-suited agent and run only it.

Cheap and effective when a task belongs to one specialist.  Routing is by
keyword overlap against each agent's name/role/description by default; if a
``router`` agent is designated it is asked to choose (falling back to keyword
scoring when it cannot).
"""

from __future__ import annotations

from maestro.agents.base import Agent
from maestro.swarm.context import RunContext
from maestro.topologies.base import SwarmResult, Topology


class RouterTopology(Topology):
    name = "router"

    def run(self, task: str, agents: list[Agent], context: RunContext) -> SwarmResult:
        router_name = self.params.get("router")
        router = next((a for a in agents if a.name == router_name), None)
        candidates = [a for a in agents if a.name != router_name] if router else list(agents)
        if not candidates:
            return SwarmResult(task=task, topology=self.name, error="no candidate agents")

        chosen = None
        if router is not None:
            chosen = self._route_via_agent(task, router, candidates, context)
        if chosen is None:
            chosen = self._route_via_keywords(task, candidates)

        context.tracer.emit("route", name=self.name, detail=f"→ {chosen.name}")
        res = chosen.run(task, context)
        return SwarmResult(
            task=task, final=res.output or res.error, topology=self.name,
            per_agent=[res], tracer=context.tracer,
            error="" if res.ok() else f"{chosen.name} produced no output",
        )

    def _route_via_agent(self, task, router, candidates, context) -> Agent | None:
        roster = "\n".join(f"- {c.name}: {c.role or c.description}" for c in candidates)
        prompt = (
            f"Choose the single best agent for the task. Reply with ONLY the agent's "
            f"name.\n\nAgents:\n{roster}\n\nTask: {task}"
        )
        res = router.run(prompt, context)
        pick = (res.output or "").strip().lower()
        for c in candidates:
            if c.name.lower() in pick:
                return c
        return None

    def _route_via_keywords(self, task: str, candidates: list[Agent]) -> Agent:
        words = {w.lower() for w in task.split() if len(w) > 3}
        best, best_score = candidates[0], -1
        for c in candidates:
            profile = f"{c.name} {c.role} {c.description}".lower()
            score = sum(1 for w in words if w in profile)
            if score > best_score:
                best, best_score = c, score
        return best
