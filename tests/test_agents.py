"""Agents — LLM agent, tool loop, and the external CLI/MCP adapters."""

from __future__ import annotations

import sys

from maestro.agents import CLIAgent, LLMAgent, MCPAgent, isaac_agent, olivia_agent
from maestro.providers.base import EchoClient, LLMClient, LLMResponse
from maestro.swarm.context import RunContext
from maestro.tools.builtin import builtin_tools


class ScriptedClient(LLMClient):
    """Returns queued responses in order — for driving the tool loop in tests."""

    name = "scripted"

    def __init__(self, outputs):
        self._outputs = list(outputs)

    @property
    def available(self):
        return True

    def complete(self, messages, system="", max_tokens=None, temperature=None):
        text = self._outputs.pop(0) if self._outputs else "done"
        return LLMResponse(text=text, model="scripted")


def test_llm_agent_produces_output():
    agent = LLMAgent("writer", EchoClient(), role="writer")
    ctx = RunContext(task="t")
    res = agent.run("summarize the news", ctx)
    assert res.ok()
    assert "summarize" in res.output
    # It should have posted itself onto the blackboard.
    assert any(m.sender == "writer" for m in ctx.blackboard.messages())


def test_llm_agent_tool_loop_executes_tool():
    client = ScriptedClient(["ACTION: calc 2*(3+4)", "The result is 14."])
    agent = LLMAgent(
        "calculator",
        client,
        role="analyst",
        tools=[builtin_tools()["calc"]],
        max_tool_iters=3,
    )
    ctx = RunContext(task="t")
    res = agent.run("what is 2*(3+4)?", ctx)
    assert res.ok()
    assert "14" in res.output
    assert any(e.kind == "tool_call" for e in ctx.tracer.events)


def test_cli_agent_availability():
    assert not CLIAgent("nope", command=["definitely-not-a-real-binary-xyz"]).available
    assert CLIAgent("py", command=[sys.executable, "-c", "pass"]).available


def test_cli_agent_arg_mode_runs():
    agent = CLIAgent(
        "py",
        command=[sys.executable, "-c", "import sys; print('got ' + sys.argv[1])"],
        input_mode="arg",
    )
    res = agent.run("hello")
    assert res.ok()
    assert "got hello" in res.output


def test_cli_agent_stdin_mode_runs():
    agent = CLIAgent(
        "py",
        command=[sys.executable, "-c", "import sys; print(sys.stdin.read().strip().upper())"],
        input_mode="stdin",
    )
    res = agent.run("hello")
    assert res.ok()
    assert res.output == "HELLO"


def test_cli_agent_missing_command_reports_error_not_crash():
    res = CLIAgent("nope", command=["definitely-not-a-real-binary-xyz"]).run("x")
    assert not res.ok()
    assert "not found" in res.error


def test_mcp_agent_availability_and_graceful_failure():
    a = MCPAgent("m", server_cmd=["definitely-not-a-real-server-xyz"], tool="t")
    assert not a.available
    res = a.run("hi")
    assert not res.ok() and res.error


def test_external_presets_build():
    isaac = isaac_agent()
    olivia = olivia_agent()
    assert isaac.kind == "cli" and isaac.name == "isaac"
    assert olivia.kind == "cli" and olivia.name == "olivia"
    # mcp mode yields an MCP adapter.
    assert isaac_agent(mode="mcp").kind == "mcp"


def test_windows_cmd(monkeypatch):
    """Olivia must preserve the integral MAESTRO_OLIVIA_CMD; CLI must keep C:\\ paths."""
    import os as _os

    from maestro.agents.cli_agent import split_command
    from maestro.config.settings import Settings

    # Integral preservation (POSIX-safe base): extra args must survive,
    # and a subcommand override must replace — not drop — the rest.
    s = Settings()
    s.olivia_cmd = "python -m olivia ask --verbose"
    assert olivia_agent(subcommand="ask", settings=s).command == [
        "python",
        "-m",
        "olivia",
        "ask",
        "--verbose",
    ]
    assert olivia_agent(subcommand="solve", settings=s).command == [
        "python",
        "-m",
        "olivia",
        "solve",
        "--verbose",
    ]

    # Windows branch: backslashes in C:\ paths must survive shlex splitting.
    monkeypatch.setattr(_os, "name", "nt")
    parts = split_command(r"C:\venv\Scripts\python.exe -m olivia ask")
    assert parts[0] == r"C:\venv\Scripts\python.exe"
    assert "ask" in parts
    s2 = Settings()
    s2.olivia_cmd = r"C:\venv\Scripts\python.exe -m olivia ask --verbose"
    ag3 = olivia_agent(subcommand="ask", settings=s2)
    assert ag3.command[0] == r"C:\venv\Scripts\python.exe"
    assert "--verbose" in ag3.command
    c = CLIAgent("w", command=r"C:\tools\agent.exe run --flag")
    assert c.command[0] == r"C:\tools\agent.exe"
