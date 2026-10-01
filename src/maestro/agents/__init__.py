"""Swarm agents — native LLM agents plus external CLI/MCP/sibling adapters."""

from __future__ import annotations

from maestro.agents.base import Agent, AgentResult
from maestro.agents.cli_agent import CLIAgent
from maestro.agents.decision_agent import (
    DECISION_PRESETS,
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionAgent,
    DecisionResult,
    JevAgent,
    LayaAgent,
    LayaONNXAgent,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
    email_questions,
    guard_questions,
    moderation_questions,
    router_questions,
    triage_questions,
)
from maestro.agents.external import isaac_agent, olivia_agent
from maestro.agents.llm_agent import LLMAgent
from maestro.agents.mcp_agent import MCPAgent
from maestro.agents.registry import AGENT_TYPES, build_agent

__all__ = [
    "AGENT_TYPES",
    "DECISION_PRESETS",
    "Agent",
    "AgentResult",
    "CLIAgent",
    "ChoiceAnswer",
    "ChoiceQuestion",
    "DecisionAgent",
    "DecisionResult",
    "JevAgent",
    "LLMAgent",
    "LayaAgent",
    "LayaONNXAgent",
    "MCPAgent",
    "NoulAnswer",
    "NoulQuestion",
    "ScoreAnswer",
    "ScoreQuestion",
    "build_agent",
    "email_questions",
    "guard_questions",
    "isaac_agent",
    "moderation_questions",
    "olivia_agent",
    "router_questions",
    "triage_questions",
]
