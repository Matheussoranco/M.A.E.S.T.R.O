"""Expose MAESTRO itself over MCP (stdio, JSON-RPC 2.0, newline-delimited).

This lets Claude Code / Claude Desktop — or a *parent* MAESTRO swarm — instantiate
and run agent swarms as tool calls, keeping MAESTRO consistent with the rest of
the ecosystem (I.S.A.A.C. and O.L.I.V.I.A. both ship ``mcp-serve``).

Register with an ``.mcp.json`` entry::

    {"mcpServers": {"maestro": {"command": "maestro", "args": ["mcp-serve"]}}}
"""

from __future__ import annotations

import json
import sys

from maestro import __version__

PROTOCOL_VERSION = "2024-11-05"

TOOLS = [
    {
        "name": "maestro_run_demo",
        "description": "Run the built-in offline demo swarm (supervisor topology) on a task.",
        "inputSchema": {
            "type": "object",
            "properties": {"task": {"type": "string"}},
            "required": ["task"],
        },
    },
    {
        "name": "maestro_run_spec",
        "description": "Instantiate a swarm from a spec file (.yaml/.json) and run it on a task.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "spec_path": {"type": "string"},
                "task": {"type": "string"},
            },
            "required": ["spec_path", "task"],
        },
    },
    {
        "name": "maestro_validate",
        "description": "Validate a swarm spec file and return any problems.",
        "inputSchema": {
            "type": "object",
            "properties": {"spec_path": {"type": "string"}},
            "required": ["spec_path"],
        },
    },
    {
        "name": "maestro_list_topologies",
        "description": "List the available swarm topologies.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def _text(content: str) -> dict:
    return {"content": [{"type": "text", "text": content}], "isError": False}


def _error(content: str) -> dict:
    return {"content": [{"type": "text", "text": content}], "isError": True}


def _call_tool(name: str, args: dict) -> dict:
    try:
        if name == "maestro_run_demo":
            from maestro.demo import run_demo

            res = run_demo(args.get("task") or "Plan an evaluation.", trace=False)
            return _text(res.final or f"(no output — {res.error})")
        if name == "maestro_run_spec":
            from maestro.orchestrator import Orchestrator

            orch = Orchestrator.from_file(args["spec_path"])
            res = orch.run(args["task"], trace=False)
            return _text(res.final or f"(no output — {res.error})")
        if name == "maestro_validate":
            from maestro.orchestrator.spec import SwarmSpec

            spec = SwarmSpec.from_file(args["spec_path"])
            problems = spec.validate()
            return _text("valid" if not problems else "invalid:\n- " + "\n- ".join(problems))
        if name == "maestro_list_topologies":
            from maestro.topologies import topology_names

            return _text(", ".join(topology_names()))
    except Exception as exc:
        return _error(f"{type(exc).__name__}: {exc}")
    return _error(f"unknown tool {name!r}")


def _handle(msg: dict) -> dict | None:
    mid = msg.get("id")
    method = msg.get("method")
    if method == "initialize":
        return {
            "jsonrpc": "2.0", "id": mid,
            "result": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "maestro", "version": __version__},
            },
        }
    if method == "notifications/initialized":
        return None  # notification — no response
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS}}
    if method == "tools/call":
        params = msg.get("params") or {}
        result = _call_tool(params.get("name", ""), params.get("arguments") or {})
        return {"jsonrpc": "2.0", "id": mid, "result": result}
    if method == "ping":
        return {"jsonrpc": "2.0", "id": mid, "result": {}}
    if mid is not None:
        return {"jsonrpc": "2.0", "id": mid,
                "error": {"code": -32601, "message": f"method not found: {method}"}}
    return None


def serve(stdin=None, stdout=None) -> None:
    """Blocking stdio serve loop.  Reads newline-delimited JSON-RPC from stdin."""
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        response = _handle(msg)
        if response is not None:
            stdout.write(json.dumps(response) + "\n")
            stdout.flush()


if __name__ == "__main__":  # pragma: no cover
    serve()
