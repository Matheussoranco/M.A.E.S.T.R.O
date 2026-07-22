"""Drive an external MCP (Model Context Protocol) server as a swarm member.

MCP is the ecosystem's designated integration point — both I.S.A.A.C.
(``isaac mcp-serve``) and O.L.I.V.I.A. (``olivia mcp-serve``) expose one.  This
adapter speaks the stdio transport (newline-delimited JSON-RPC 2.0): it launches
the server, performs the ``initialize`` handshake, calls a single named tool,
and returns its text content.

The implementation is intentionally small and defensive — a fresh process per
call, hard timeouts, and graceful error capture — rather than a full persistent
MCP client.  For heavy use prefer :class:`~maestro.agents.cli_agent.CLIAgent`.
"""

from __future__ import annotations

import json
import os
import queue
import shlex
import shutil
import subprocess
import threading

from maestro.agents.base import Agent, AgentResult
from maestro.swarm.context import RunContext


class MCPAgent(Agent):
    kind = "mcp"

    def __init__(
        self,
        name: str,
        server_cmd: list[str] | str,
        tool: str,
        role: str = "",
        description: str = "",
        arg_key: str = "query",
        extra_args: dict | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: float = 300.0,
    ) -> None:
        super().__init__(name=name, role=role, description=description)
        self.server_cmd = (
            shlex.split(server_cmd) if isinstance(server_cmd, str) else list(server_cmd)
        )
        self.tool = tool
        self.arg_key = arg_key
        self.extra_args = extra_args or {}
        self.cwd = cwd
        self.env = env
        self.timeout = timeout

    @property
    def available(self) -> bool:
        if not self.server_cmd:
            return False
        exe = self.server_cmd[0]
        return os.path.exists(exe) or shutil.which(exe) is not None

    def run(self, task: str, context: RunContext | None = None) -> AgentResult:
        if context is not None:
            context.tracer.emit("agent_start", name=self.name, detail=f"mcp:{self.tool}")
        if not self.available:
            return self._finish(
                context,
                AgentResult(self.name, self.role,
                            error=f"MCP server command not found: {self.server_cmd[0]!r}"),
            )
        try:
            text = self._call_tool(task)
        except _MCPError as exc:
            return self._finish(context, AgentResult(self.name, self.role, error=str(exc)))
        except Exception as exc:
            return self._finish(
                context, AgentResult(self.name, self.role, error=f"mcp error: {exc}")
            )
        return self._finish(context, AgentResult(self.name, self.role, output=text))

    # -- stdio JSON-RPC plumbing ---------------------------------------------
    def _call_tool(self, task: str) -> str:
        run_env = {**os.environ, **self.env} if self.env else None
        proc = subprocess.Popen(
            self.server_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            cwd=self.cwd,
            env=run_env,
        )
        reader = _LineReader(proc.stdout)
        reader.start()
        try:
            self._send(proc, 1, "initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "maestro", "version": "0.1.0"},
            })
            self._await(reader, 1)
            self._notify(proc, "notifications/initialized")
            args = {self.arg_key: task, **self.extra_args}
            self._send(proc, 2, "tools/call", {"name": self.tool, "arguments": args})
            result = self._await(reader, 2)
        finally:
            with _suppress():
                proc.stdin.close()
            with _suppress():
                proc.terminate()
            reader.stop()

        content = (result or {}).get("content") or []
        texts = [
            c.get("text", "")
            for c in content
            if isinstance(c, dict) and c.get("type") == "text"
        ]
        joined = "\n".join(t for t in texts if t)
        if not joined and result is not None:
            joined = json.dumps(result)[:2000]
        return joined

    def _send(self, proc, mid: int, method: str, params: dict) -> None:
        proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": mid, "method": method,
                                     "params": params}) + "\n")
        proc.stdin.flush()

    def _notify(self, proc, method: str, params: dict | None = None) -> None:
        proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": method,
                                     "params": params or {}}) + "\n")
        proc.stdin.flush()

    def _await(self, reader: _LineReader, mid: int) -> dict | None:
        import time
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            line = reader.get(timeout=max(0.05, deadline - time.time()))
            if line is None:
                if reader.finished():
                    raise _MCPError("MCP server closed the connection")
                continue
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("id") == mid:
                if "error" in obj:
                    raise _MCPError(f"MCP error: {obj['error']}")
                return obj.get("result")
        raise _MCPError(f"timed out waiting for response to request {mid}")


class _MCPError(RuntimeError):
    pass


class _suppress:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return True


class _LineReader(threading.Thread):
    """Reads a pipe line-by-line into a queue so we can apply timeouts."""

    def __init__(self, stream) -> None:
        super().__init__(daemon=True)
        self._stream = stream
        self._q: queue.Queue[str | None] = queue.Queue()
        self._stop = threading.Event()
        self._done = threading.Event()

    def run(self) -> None:
        try:
            for line in self._stream:
                if self._stop.is_set():
                    break
                self._q.put(line)
        except Exception:
            pass
        finally:
            self._done.set()
            self._q.put(None)

    def get(self, timeout: float) -> str | None:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def finished(self) -> bool:
        return self._done.is_set()

    def stop(self) -> None:
        self._stop.set()
