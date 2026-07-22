"""M.A.E.S.T.R.O. — Multi-Agent Ensemble Swarm Task-Routing Orchestrator.

MAESTRO instantiates and conducts *agent swarms* — teams of cooperating agents
that decompose and solve a task together — in the spirit of the multi-agent
systems shipped by MiniMax and Kimi.  Its two design commitments:

* **Provider-agnostic.**  Every agent may be backed by any LLM: Anthropic,
  OpenAI, any OpenAI-compatible endpoint (Groq, Together, OpenRouter, vLLM,
  LM Studio, llama.cpp server), or a fully local Ollama model.  The whole
  provider layer is implemented on the Python standard library (``urllib``),
  so MAESTRO imports and runs with **zero third-party dependencies**.

* **Agent-agnostic.**  A swarm member can be a native LLM agent, an arbitrary
  external command-line agent, an MCP server, or a sibling project such as
  **I.S.A.A.C.** and **O.L.I.V.I.A.** enlisted as first-class swarm members.

Swarms are described declaratively (a dict / YAML / JSON *spec*) and brought to
life by the :class:`~maestro.orchestrator.orchestrator.Orchestrator`.  The whole
system degrades gracefully: with no API keys configured it falls back to a
deterministic offline ``echo`` backend so the control-flow of any swarm can be
exercised end-to-end without a network or credentials.
"""

from __future__ import annotations

__version__ = "0.1.0"


def __getattr__(name: str):
    """Lazily expose the public API without importing everything on ``import maestro``."""
    if name == "Orchestrator":
        from maestro.orchestrator import Orchestrator

        return Orchestrator
    if name == "SwarmSpec":
        from maestro.orchestrator import SwarmSpec

        return SwarmSpec
    if name == "Swarm":
        from maestro.swarm import Swarm

        return Swarm
    if name == "build_agent":
        from maestro.agents import build_agent

        return build_agent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Orchestrator", "Swarm", "SwarmSpec", "__version__", "build_agent"]

