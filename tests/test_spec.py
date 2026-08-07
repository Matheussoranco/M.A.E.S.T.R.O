"""Spec parsing & validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from maestro.orchestrator.spec import SwarmSpec

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _good() -> dict:
    return {
        "name": "s",
        "topology": "supervisor",
        "topology_params": {"supervisor": "lead"},
        "providers": {"stub": {"provider": "echo"}},
        "agents": [
            {"name": "lead", "type": "llm", "provider": "stub", "role": "supervisor"},
            {"name": "w1", "type": "llm", "provider": "stub"},
        ],
    }


def test_valid_spec_has_no_problems():
    assert SwarmSpec.from_dict(_good()).validate() == []


def test_unknown_topology_flagged():
    d = _good()
    d["topology"] = "bogus"
    problems = SwarmSpec.from_dict(d).validate()
    assert any("topology" in p for p in problems)


def test_duplicate_agent_names_flagged():
    d = _good()
    d["agents"].append({"name": "lead", "type": "llm", "provider": "stub"})
    assert any("duplicate" in p for p in SwarmSpec.from_dict(d).validate())


def test_cli_agent_requires_command():
    d = _good()
    d["agents"].append({"name": "x", "type": "cli"})
    assert any("command" in p for p in SwarmSpec.from_dict(d).validate())


def test_unknown_provider_reference_flagged():
    d = _good()
    d["agents"][0]["provider"] = "ghost"
    assert any("unknown provider" in p for p in SwarmSpec.from_dict(d).validate())


def test_dangling_supervisor_reference_flagged():
    d = _good()
    d["topology_params"]["supervisor"] = "nobody"
    assert any("supervisor" in p for p in SwarmSpec.from_dict(d).validate())


def test_from_file_json_roundtrip(tmp_path):
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(_good()), encoding="utf-8")
    spec = SwarmSpec.from_file(str(path))
    assert spec.name == "s"
    assert spec.validate() == []


# --------------------------------------------------------------------------- #
# The bundled examples are documentation people copy — keep them working.
# --------------------------------------------------------------------------- #


def _example_paths() -> list[Path]:
    if not EXAMPLES.is_dir():  # pragma: no cover - examples ship with the repo
        return []
    return sorted(p for p in EXAMPLES.iterdir() if p.suffix in {".yaml", ".yml", ".json"})


@pytest.mark.parametrize("path", _example_paths(), ids=lambda p: p.name)
def test_bundled_example_is_valid(path):
    if path.suffix in {".yaml", ".yml"}:
        pytest.importorskip("yaml", reason="YAML specs need the optional PyYAML extra")
    spec = SwarmSpec.from_file(str(path))
    assert spec.validate() == [], f"{path.name} is not a valid spec"
    assert spec.description, f"{path.name} should describe itself for `maestro examples`"


def test_examples_cover_every_topology():
    """Each topology needs at least one worked example to copy from."""
    pytest.importorskip("yaml", reason="most example specs are YAML")
    from maestro.topologies import TOPOLOGIES

    covered = {SwarmSpec.from_file(str(p)).topology for p in _example_paths()}
    assert set(TOPOLOGIES) <= covered, f"no example for: {set(TOPOLOGIES) - covered}"
