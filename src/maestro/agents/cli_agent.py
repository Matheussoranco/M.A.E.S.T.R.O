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

# Limite de payload para argv/stdin: evita estouro de linha de comando e
# vazamento acidental de dumps gigantes. Configurável via MAESTRO_CLI_MAX_PAYLOAD.
MAX_PAYLOAD_CHARS = int(os.environ.get("MAESTRO_CLI_MAX_PAYLOAD", "8000"))

# Chaves nunca herdadas do ambiente do pai para o subprocesso (secrets).
_SECRET_HINTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL")


def _filtered_env(extra: dict[str, str] | None) -> dict[str, str]:
    """Ambiente mínimo + extras explícitos, sem vazar secrets do pai."""
    keep = {
        "PATH",
        "PATHEXT",
        "SYSTEMROOT",
        "WINDIR",
        "LANG",
        "LC_ALL",
        "TEMP",
        "TMP",
        "HOME",
        "USERPROFILE",
    }
    base = {k: v for k, v in os.environ.items() if k in keep}
    for k, v in (extra or {}).items():
        upper = k.upper()
        if any(h in upper for h in _SECRET_HINTS):
            continue  # secret explícito não é herdado/injetado por padrão
        base[k] = v
    return base


def split_command(command: str) -> list[str]:
    """Split a command string without corrupting Windows paths.

    ``shlex.split`` defaults to ``posix=True``, which treats backslashes as
    escapes and mangles paths like ``C:\\venv\\...``.  On Windows (``os.name ==
    'nt'``) use ``posix=False`` so backslashes survive, stripping the single
    layer of quotes that non-POSIX mode preserves.  Either mode falls back to
    the other (then to a plain split) on unbalanced quotes.
    """
    if os.name == "nt":
        try:
            parts = shlex.split(command, posix=False)
        except ValueError:
            try:
                return shlex.split(command, posix=True)
            except ValueError:
                return command.split()
        out: list[str] = []
        for part in parts:
            if len(part) >= 2 and part[0] == part[-1] and part[0] in ("'", '"'):
                out.append(part[1:-1])
            else:
                out.append(part)
        return out
    try:
        return shlex.split(command)
    except ValueError:
        try:
            return shlex.split(command, posix=False)
        except ValueError:
            return command.split()


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
        self.command = split_command(command) if isinstance(command, str) else list(command)
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
                task = f"{digest}\n\nTask: {task}"
        # Limite rígido: trunca com marcador em vez de vazar argv gigante.
        if len(task) > MAX_PAYLOAD_CHARS:
            task = task[:MAX_PAYLOAD_CHARS] + "\n…[truncated]"
        return task

    def run(self, task: str, context: RunContext | None = None) -> AgentResult:
        if context is not None:
            context.tracer.emit("agent_start", name=self.name, detail=f"cli:{self.command[0]}")
        if not self.available:
            return self._finish(
                context,
                AgentResult(
                    self.name,
                    self.role,
                    error=f"command not found: {self.command[0]!r} "
                    f"(configure the path, e.g. MAESTRO_ISAAC_CMD)",
                ),
            )
        payload = self._payload(task, context)
        # Timeout configurável: self.timeout ou MAESTRO_CLI_TIMEOUT (segundos).
        try:
            timeout = float(os.environ.get("MAESTRO_CLI_TIMEOUT", str(self.timeout)))
        except ValueError:
            timeout = self.timeout
        argv = list(self.command)
        stdin_data = None
        if self.input_mode == "stdin":
            stdin_data = payload
        elif self.input_mode == "template":
            argv = [a.replace("{task}", payload) for a in argv]
        else:  # "arg"
            # argv leak: nunca passe payload gigante como argumento cru sem limite
            # (já truncado em _payload) nem secrets via env global.
            argv.append(payload)

        run_env = _filtered_env(self.env)
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
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return self._finish(
                context,
                AgentResult(self.name, self.role, error=f"timed out after {timeout}s"),
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
                self.name,
                self.role,
                output=out,
                meta={"returncode": proc.returncode, "stderr": (proc.stderr or "")[-400:]},
            ),
        )
