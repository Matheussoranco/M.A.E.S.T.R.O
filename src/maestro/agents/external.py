"""Presets that enlist the sibling projects as first-class swarm members.

I.S.A.A.C. and O.L.I.V.I.A. ship non-interactive, task-as-argument entry points
and MCP servers, so both integrate cleanly.  Commands are read from settings
(``MAESTRO_ISAAC_CMD`` / ``MAESTRO_OLIVIA_CMD``) so a user whose projects live in
separate virtualenvs can point at the right interpreter without touching code.
"""

from __future__ import annotations

from maestro.agents.base import Agent
from maestro.agents.cli_agent import CLIAgent, split_command
from maestro.agents.mcp_agent import MCPAgent
from maestro.config.settings import settings as default_settings

#: Subcommands understood by O.L.I.V.I.A.'s CLI entry point.
OLIVIA_SUBCOMMANDS = ("ask", "solve")


def _olivia_cmd(base: list[str], subcommand: str) -> list[str]:
    """Merge a configured base command with the requested subcommand.

    Preserves the integral configured command (interpreter, ``-m`` flags,
    extra args) instead of truncating to ``[base[0], subcommand]``:

    * ``[]`` → ``["olivia", subcommand]``
    * ``["olivia"]`` → ``["olivia", subcommand]``
    * base already carries the wanted subcommand → ``base`` unchanged
    * base carries the *other* known subcommand → replace it, keep the rest
    * base carries the wanted subcommand deeper (e.g. ``python -m olivia ask``)
      → ``base`` unchanged
    * otherwise → ``base + [subcommand]``
    """
    if not base:
        return ["olivia", subcommand]
    if len(base) == 1:
        return [base[0], subcommand]
    if base[1] == subcommand:
        return list(base)
    if base[1] in OLIVIA_SUBCOMMANDS:
        return [base[0], subcommand, *base[2:]]
    if subcommand in base:
        return list(base)
    for i, tok in enumerate(base):
        if tok in OLIVIA_SUBCOMMANDS:
            return [*base[:i], subcommand, *base[i + 1 :]]
    return [*base, subcommand]


def isaac_agent(
    name: str = "isaac",
    role: str = "deep-reasoner",
    mode: str = "cli",
    command: str = "",
    timeout: float = 600.0,
    cwd: str | None = None,
    settings=None,
) -> Agent:
    """I.S.A.A.C. as a swarm member — the neuro-symbolic deep reasoner.

    ``mode="cli"`` runs ``isaac agent "<task>"``; ``mode="mcp"`` calls the
    ``isaac_ask`` tool over ``isaac mcp-serve``.
    """
    s = settings or default_settings
    description = "I.S.A.A.C. — neuro-symbolic autonomous agent (ARC-AGI, program synthesis)."
    if mode == "mcp":
        return MCPAgent(
            name=name,
            role=role,
            description=description,
            server_cmd="isaac mcp-serve",
            tool="isaac_ask",
            arg_key="question",
            timeout=timeout,
            cwd=cwd,
        )
    cmd = split_command(command) if command else split_command(s.isaac_cmd)
    return CLIAgent(
        name=name,
        role=role,
        description=description,
        command=cmd,
        input_mode="arg",
        timeout=timeout,
        cwd=cwd,
    )


def olivia_agent(
    name: str = "olivia",
    role: str = "researcher",
    mode: str = "cli",
    subcommand: str = "ask",  # "ask" | "solve"
    command: str = "",
    timeout: float = 600.0,
    cwd: str | None = None,
    settings=None,
) -> Agent:
    """O.L.I.V.I.A. as a swarm member — the study / research / discovery agent."""
    s = settings or default_settings
    description = "O.L.I.V.I.A. — study, learning & scientific-discovery agent."
    if mode == "mcp":
        return MCPAgent(
            name=name,
            role=role,
            description=description,
            server_cmd="olivia mcp-serve",
            tool="olivia_ask",
            arg_key="question",
            timeout=timeout,
            cwd=cwd,
        )
    if command:
        cmd = split_command(command)
    else:
        base = split_command(s.olivia_cmd)
        # Respect an explicit subcommand override (ask vs solve) while
        # preserving the integral configured command (interpreter, -m flags,
        # extra args) — see _olivia_cmd.
        cmd = _olivia_cmd(base, subcommand)
    return CLIAgent(
        name=name,
        role=role,
        description=description,
        command=cmd,
        input_mode="arg",
        timeout=timeout,
        cwd=cwd,
    )
