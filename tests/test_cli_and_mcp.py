"""CLI commands and the MCP server handler."""

from __future__ import annotations

import json

from maestro.cli import main
from maestro.mcp import server


def test_cli_demo_ok():
    assert main(["demo"]) == 0


def test_cli_version_topologies_providers_agents():
    assert main(["version"]) == 0
    assert main(["topologies"]) == 0
    assert main(["providers"]) == 0
    assert main(["agents"]) == 0


def test_cli_no_args_prints_help():
    assert main([]) == 0


def test_cli_validate_and_run_json_spec(tmp_path):
    spec = {
        "name": "s",
        "topology": "sequential",
        "providers": {"stub": {"provider": "echo"}},
        "agents": [
            {"name": "a", "type": "llm", "provider": "stub"},
            {"name": "b", "type": "llm", "provider": "stub"},
        ],
    }
    path = tmp_path / "s.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    assert main(["validate", str(path)]) == 0
    assert main(["run", str(path), "hello swarm"]) == 0
    assert main(["run", str(path), "hello", "--json"]) == 0


def test_cli_validate_bad_spec_returns_1(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"name": "s", "topology": "nope", "agents": []}), encoding="utf-8")
    assert main(["validate", str(path)]) == 1


# -- init -----------------------------------------------------------------


def test_cli_init_json_output_is_valid_json(tmp_path):
    # A .json target must contain actual JSON, not the YAML template — this
    # used to write YAML into a .json file, which then failed to parse.
    out = tmp_path / "swarm.json"
    assert main(["init", str(out), "--topology", "supervisor", "--agents", "lead,worker"]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["topology"] == "supervisor"
    assert data["topology_params"] == {"supervisor": "lead"}
    assert [a["name"] for a in data["agents"]] == ["lead", "worker"]


def test_cli_init_yaml_output_still_writes_yaml_template(tmp_path):
    out = tmp_path / "swarm.yaml"
    assert main(["init", str(out), "--topology", "sequential", "--agents", "a,b"]) == 0
    text = out.read_text(encoding="utf-8")
    assert text.startswith("# ")
    assert "topology: sequential" in text


# -- MCP server ---------------------------------------------------------------
def test_mcp_initialize_and_list():
    init = server._handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert init["result"]["serverInfo"]["name"] == "maestro"
    listed = server._handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    tool_names = {t["name"] for t in listed["result"]["tools"]}
    assert "maestro_run_demo" in tool_names


def test_mcp_tool_call_run_demo():
    resp = server._handle(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "maestro_run_demo", "arguments": {"task": "plan"}},
        }
    )
    result = resp["result"]
    assert result["isError"] is False
    assert result["content"][0]["text"]


def test_mcp_list_topologies_tool():
    resp = server._handle(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "maestro_list_topologies", "arguments": {}},
        }
    )
    assert "supervisor" in resp["result"]["content"][0]["text"]


def test_mcp_initialized_notification_has_no_response():
    assert server._handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
