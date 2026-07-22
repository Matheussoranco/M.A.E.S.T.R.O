"""A zero-setup demo swarm.

Runs entirely on the deterministic ``echo`` backend, so ``maestro demo`` shows a
full supervisor→workers→synthesis flow with no keys, no network, no local model.
Swap the providers for ``anthropic`` / ``ollama`` / etc. to make it real.
"""

from __future__ import annotations


def demo_spec() -> dict:
    return {
        "name": "demo-research-swarm",
        "description": "A supervisor coordinating three specialists (offline echo demo).",
        "topology": "supervisor",
        "topology_params": {"supervisor": "lead"},
        "providers": {
            "stub": {"provider": "echo"},
        },
        "agents": [
            {
                "name": "lead",
                "type": "llm",
                "provider": "stub",
                "role": "supervisor",
                "system_prompt": "You plan the work and synthesize the team's findings.",
            },
            {
                "name": "researcher",
                "type": "llm",
                "provider": "stub",
                "role": "researcher",
                "system_prompt": "You gather facts and cite sources.",
            },
            {
                "name": "engineer",
                "type": "llm",
                "provider": "stub",
                "role": "engineer",
                "system_prompt": "You reason about implementation and feasibility.",
            },
            {
                "name": "critic",
                "type": "llm",
                "provider": "stub",
                "role": "critic",
                "system_prompt": "You find flaws and risks in the plan.",
            },
        ],
    }


def run_demo(
    task: str = "Design a plan to evaluate a new AI agent on ARC-AGI-2.",
    trace: bool = True,
):
    """Build and run the demo swarm; returns a :class:`SwarmResult`."""
    from maestro.orchestrator import Orchestrator

    return Orchestrator.from_dict(demo_spec()).run(task, trace=trace)
