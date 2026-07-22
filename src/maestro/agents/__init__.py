"""Swarm agents — native LLM agents plus external CLI/MCP/sibling adapters."""

from __future__ import annotations

from maestro.agents.base import Agent, AgentResult
from maestro.agents.cli_agent import CLIAgent
from maestro.agents.external import isaac_agent, olivia_agent
from maestro.agents.llm_agent import LLMAgent
from maestro.agents.mcp_agent import MCPAgent
from maestro.agents.registry import AGENT_TYPES, build_agent

__all__ = [
    "AGENT_TYPES",
    "Agent",
    "AgentResult",
    "CLIAgent",
    "LLMAgent",
    "MCPAgent",
    "build_agent",
    "isaac_agent",
    "olivia_agent",
]
