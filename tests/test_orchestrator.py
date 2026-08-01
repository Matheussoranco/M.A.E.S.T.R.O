"""Orchestrator — build from spec, run end-to-end, introspect."""

from __future__ import annotations

import pytest

from maestro.demo import demo_spec, run_demo
from maestro.orchestrator import Orchestrator


def test_orchestrator_builds_and_runs_demo():
    orch = Orchestrator.from_dict(demo_spec())
    res = orch.run("Plan an evaluation of an ARC-AGI-2 solver.")
    assert res.ok()
    names = [r.name for r in res.per_agent]
    assert "lead" in names
    # workers all present
    for w in ("researcher", "engineer", "critic"):
        assert w in names


def test_run_demo_helper():
    res = run_demo()
    assert res.ok()
    assert res.topology == "supervisor"


def test_describe_lists_agents():
    text = Orchestrator.from_dict(demo_spec()).describe()
    assert "lead" in text and "researcher" in text


def test_invalid_spec_raises():
    bad = {"name": "s", "topology": "supervisor", "agents": []}
    with pytest.raises(ValueError):
        Orchestrator.from_dict(bad)


def test_isaac_olivia_mixed_spec_builds():
    # Even without the sibling binaries installed, the swarm must *build*; the
    # external agents simply report unavailable until their CLIs are present.
    # The command is pinned to a guaranteed-nonexistent binary rather than the
    # default "isaac agent" / "olivia ask": on a machine that happens to have
    # the real I.S.A.A.C./O.L.I.V.I.A. console scripts on PATH (e.g. a dev box
    # with both sibling projects installed), the supervisor's fan-out would
    # otherwise shell out to the *real* agents and turn this offline unit test
    # into a slow, non-deterministic integration test.
    spec = {
        "name": "mix",
        "topology": "supervisor",
        "topology_params": {"supervisor": "lead"},
        "providers": {"stub": {"provider": "echo"}},
        "agents": [
            {"name": "lead", "type": "llm", "provider": "stub", "role": "supervisor"},
            {"name": "isaac", "type": "isaac", "command": "definitely-not-a-real-binary-xyz"},
            {"name": "olivia", "type": "olivia", "command": "definitely-not-a-real-binary-xyz"},
        ],
    }
    orch = Orchestrator.from_dict(spec)
    avail = orch.availability()
    assert "isaac" in avail and "olivia" in avail
    # The swarm still runs; the echo lead drives it regardless of sibling presence.
    res = orch.run("coordinate the experts")
    assert res.per_agent  # produced structure
