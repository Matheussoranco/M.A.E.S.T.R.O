"""System 1 topology — parallel evaluation of typed decision questions.

A *System 1 topology* runs a decision-model agent (Laya, Jev, or Laya ONNX)
against a state with multiple typed questions (choice / score / noul) in a
single forward pass. The agent returns calibrated probabilities for every
question, which are combined into a structured DecisionResult.

This is the Jev / Laya / System One contract: state in, typed decisions out.
"""

from __future__ import annotations

from maestro.agents.base import Agent, AgentResult
from maestro.swarm.context import RunContext
from maestro.topologies.base import SwarmResult, Topology


class System1Topology(Topology):
    """System 1 decision-model topology.

    Takes exactly one decision-model agent (laya, laya-onnx, or jev) and
    evaluates all questions in a single forward pass. The task string is the
    state; questions come from the agent's default_questions or context.metadata.
    """

    name = "system1"

    def run(self, task: str, agents: list[Agent], context: RunContext) -> SwarmResult:
        if not agents:
            return SwarmResult(task=task, topology=self.name, error="no agents provided")

        # System 1 topology uses exactly one decision-model agent
        decision_agent = agents[0]

        # Verify it's a decision agent
        if decision_agent.kind not in ("decision", "laya", "laya-onnx", "jev"):
            return SwarmResult(
                task=task,
                topology=self.name,
                error=f"System 1 topology requires a decision-model agent (laya, laya-onnx, jev), got {decision_agent.kind}",
            )

        context.tracer.emit("system1_start", name=decision_agent.name, detail=decision_agent.role)

        # Run the decision agent
        result = decision_agent.run(task, context)

        context.tracer.emit("system1_end", name=decision_agent.name, detail="ok" if result.ok() else "error")

        # Return structured result
        if result.ok():
            return SwarmResult(
                task=task,
                final=result.output,  # JSON string of DecisionResult
                topology=self.name,
                per_agent=[result],
                tracer=context.tracer,
            )
        else:
            return SwarmResult(
                task=task,
                final="",
                topology=self.name,
                per_agent=[result],
                tracer=context.tracer,
                error=result.error,
            )