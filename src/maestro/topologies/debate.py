"""Debate topology — multi-agent debate with an optional judge.

Debaters each answer, then revise across one or more rounds while seeing the
others' arguments, and an optional judge delivers the verdict.  Improves
factuality and reasoning through cross-examination.
"""

from __future__ import annotations

from maestro.agents.base import Agent, AgentResult
from maestro.swarm.context import RunContext
from maestro.topologies.base import SwarmResult, Topology


class DebateTopology(Topology):
    name = "debate"

    def run(self, task: str, agents: list[Agent], context: RunContext) -> SwarmResult:
        rounds = max(1, int(self.params.get("rounds", 2)))
        judge_name = self.params.get("judge")
        judge = next((a for a in agents if a.name == judge_name), None)
        debaters = [a for a in agents if a.name != judge_name]
        if not debaters:
            debaters = agents
            judge = None

        context.tracer.emit("topology", name=self.name,
                            detail=f"{len(debaters)} debaters, {rounds} rounds"
                                   + (f", judge={judge.name}" if judge else ""))

        results: list[AgentResult] = []
        last_round: list[AgentResult] = []
        for r in range(rounds):
            context.tracer.emit("round", name=self.name, detail=f"round {r + 1}/{rounds}")
            round_results: list[AgentResult] = []
            for agent in debaters:
                if r == 0:
                    prompt = task
                else:
                    prompt = (
                        f"{context.context_digest(limit=len(debaters) + 1)}\n\n"
                        f"Consider the other arguments above. Refine or defend your "
                        f"answer to the task:\n{task}"
                    )
                res = agent.run(prompt, context)
                round_results.append(res)
                results.append(res)
            last_round = round_results

        if judge is not None:
            verdict_prompt = (
                f"{context.context_digest(limit=len(debaters) * rounds + 1)}\n\n"
                f"As the judge, weigh the arguments and deliver the single best final "
                f"answer to the task:\n{task}"
            )
            verdict = judge.run(verdict_prompt, context)
            results.append(verdict)
            final = verdict.output or verdict.error
            error = "" if verdict.ok() else "judge produced no verdict"
        else:
            parts = [f"[{r.name}] {r.output}" for r in last_round if r.ok()]
            final = "\n\n".join(parts)
            error = "" if parts else "no debater produced output"

        return SwarmResult(
            task=task, final=final, topology=self.name, per_agent=results,
            tracer=context.tracer, error=error,
        )
