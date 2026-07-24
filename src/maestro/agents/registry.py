"""Build concrete agents from declarative spec dicts.

This is the bridge between a swarm *spec* (plain data) and live :class:`Agent`
objects.  The ``type`` field selects the blueprint; provider references are
resolved against the swarm's named provider table.
"""

from __future__ import annotations

from maestro.agents.base import Agent
from maestro.agents.cli_agent import CLIAgent
from maestro.agents.external import isaac_agent, olivia_agent
from maestro.agents.llm_agent import LLMAgent
from maestro.agents.mcp_agent import MCPAgent
from maestro.config.settings import settings as default_settings
from maestro.providers import ProviderSpec, resolve_client
from maestro.tools import build_tools

#: Agent ``type`` values understood by :func:`build_agent`.
AGENT_TYPES = ("llm", "cli", "mcp", "isaac", "olivia")


def _provider_spec(agent_spec: dict, providers: dict[str, ProviderSpec], settings) -> ProviderSpec:
    """Resolve an agent's provider reference into a concrete :class:`ProviderSpec`."""
    ref = agent_spec.get("provider")
    if isinstance(ref, dict):
        base = ProviderSpec(**{k: ref[k] for k in ref if k in ProviderSpec.__dataclass_fields__})
    elif isinstance(ref, str) and ref in providers:
        base = ProviderSpec(**vars(providers[ref]))
    elif isinstance(ref, str) and ref:
        base = ProviderSpec(provider=ref)
    else:
        base = ProviderSpec(provider=settings.default_provider, model=settings.default_model)
    # Per-agent overrides.
    if agent_spec.get("model"):
        base.model = agent_spec["model"]
    if agent_spec.get("persona"):
        base.persona = agent_spec["persona"]
    if not base.persona:
        base.persona = agent_spec.get("name", "")
    return base


def build_agent(
    spec: dict,
    providers: dict[str, ProviderSpec] | None = None,
    settings=None,
) -> Agent:
    """Instantiate a single agent from its spec dict."""
    settings = settings or default_settings
    providers = providers or {}
    spec = dict(spec)
    atype = (spec.get("type") or "llm").lower()
    name = spec.get("name")
    if not name:
        raise ValueError("agent spec is missing required field 'name'")
    role = spec.get("role", "")
    description = spec.get("description", "")

    if atype == "llm":
        pspec = _provider_spec(spec, providers, settings)
        client = resolve_client(pspec, settings)
        tools = build_tools(spec.get("tools", []))
        return LLMAgent(
            name=name,
            client=client,
            role=role,
            description=description,
            system_prompt=spec.get("system_prompt", ""),
            max_tokens=spec.get("max_tokens", settings.max_tokens),
            temperature=spec.get("temperature"),
            tools=tools,
            use_context=spec.get("use_context", True),
        )

    if atype == "cli":
        if not spec.get("command"):
            raise ValueError(f"cli agent {name!r} requires a 'command'")
        return CLIAgent(
            name=name,
            role=role,
            description=description,
            command=spec["command"],
            input_mode=spec.get("input_mode", "arg"),
            template=spec.get("template", "{task}"),
            cwd=spec.get("cwd"),
            env=spec.get("env"),
            timeout=spec.get("timeout", 300.0),
            include_context=spec.get("include_context", False),
        )

    if atype == "mcp":
        if not spec.get("server_cmd") or not spec.get("tool"):
            raise ValueError(f"mcp agent {name!r} requires 'server_cmd' and 'tool'")
        return MCPAgent(
            name=name,
            role=role,
            description=description,
            server_cmd=spec["server_cmd"],
            tool=spec["tool"],
            arg_key=spec.get("arg_key", "query"),
            extra_args=spec.get("extra_args"),
            cwd=spec.get("cwd"),
            env=spec.get("env"),
            timeout=spec.get("timeout", 300.0),
        )

    if atype == "isaac":
        return isaac_agent(
            name=name,
            role=role or "deep-reasoner",
            mode=spec.get("mode", "cli"),
            command=spec.get("command", ""),
            timeout=spec.get("timeout", 600.0),
            cwd=spec.get("cwd"),
            settings=settings,
        )

    if atype == "olivia":
        return olivia_agent(
            name=name,
            role=role or "researcher",
            mode=spec.get("mode", "cli"),
            subcommand=spec.get("subcommand", "ask"),
            command=spec.get("command", ""),
            timeout=spec.get("timeout", 600.0),
            cwd=spec.get("cwd"),
            settings=settings,
        )

    raise ValueError(f"unknown agent type {atype!r}; choose one of {', '.join(AGENT_TYPES)}")


__all__ = ["AGENT_TYPES", "build_agent"]
