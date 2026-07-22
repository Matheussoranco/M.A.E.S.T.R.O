"""Wrap any external command-line agent as a swarm member.

This is the most universal adapter: if a program takes a task and prints an
answer, it can join a MAESTRO swarm.  It is exactly how the sibling projects
**I.S.A.A.C.** (``isaac agent "<task>"``) and **O.L.I.V.I.A.**
(``olivia ask "<question>"``) are enlisted.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess

from maestro.agents.base import Agent, AgentResult
from maestro.swarm.context import RunContext


class CLIAgent(Agent):
    """Run an external command, passing the task as an argument or on stdin."""

    kind = "cli"

    def __init__(
        self,
        name: str,
        command: list[str] | str,
        role: str = "",
        description: str = "",
        input_mode: str = "arg",  # "arg" (append) | "stdin" | "template"
        template: str = "{task}",
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: float = 300.0,
        include_context: bool = False,
    ) -> None:
        super().__init__(name=name, role=role, description=description)
        self.command = shlex.split(command) if isinstance(command, str) else list(command)
        self.input_mode = input_mode
        self.template = template
        self.cwd = cwd
        self.env = env
        self.timeout = timeout
        self.include_context = include_context

    @property
    def available(self) -> bool:
        if not self.command:
            return False
        exe = self.command[0]
        # Absolute/relative path, or something resolvable on PATH.
        return os.path.exists(exe) or shutil.which(exe) is not None

    def _payload(self, task: str, context: RunContext | None) -> str:
        if self.include_context and context is not None:
            digest = context.context_digest()
            if digest:
                return f"{digest}\n\nTask: {task}"
        return task

    def run(self, task: str, context: RunContext | None = None) -> AgentResult:
        if context is not None:
            context.tracer.emit("agent_start", name=self.name, detail=f"cli:{self.command[0]}")
        if not self.available:
            return self._finish(
                context,
                AgentResult(
                    self.name, self.role,
                    error=f"command not found: {self.command[0]!r} "
                          f"(configure the path, e.g. MAESTRO_ISAAC_CMD)",
                ),
            )
        payload = self._payload(task, context)
        argv = list(self.command)
        stdin_data = None
        if self.input_mode == "stdin":
            stdin_data = payload
        elif self.input_mode == "template":
            argv = [a.replace("{task}", payload) for a in argv]
        else:  # "arg"
            argv.append(payload)

        run_env = None
        if self.env:
            run_env = {**os.environ, **self.env}
        try:
            proc = subprocess.run(
                argv,
                input=stdin_data,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=self.cwd,
                env=run_env,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            return self._finish(
                context,
                AgentResult(self.name, self.role, error=f"timed out after {self.timeout}s"),
            )
        except Exception as exc:
            return self._finish(
                context, AgentResult(self.name, self.role, error=f"launch failed: {exc}")
            )

        out = (proc.stdout or "").strip()
        if proc.returncode != 0 and not out:
            err = (proc.stderr or "").strip() or f"exited {proc.returncode}"
            return self._finish(context, AgentResult(self.name, self.role, error=err))
        return self._finish(
            context,
            AgentResult(
                self.name, self.role, output=out,
                meta={"returncode": proc.returncode, "stderr": (proc.stderr or "")[-400:]},
            ),
        )
