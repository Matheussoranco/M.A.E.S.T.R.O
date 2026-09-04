"""Topologies — every composition pattern runs and produces sane structure."""

from __future__ import annotations

from maestro.agents.llm_agent import LLMAgent
from maestro.providers.base import EchoClient
from maestro.swarm.swarm import Swarm
from maestro.topologies import build_topology


def _agent(name, role=""):
    return LLMAgent(name, EchoClient(persona=name), role=role)


def test_sequential_final_is_last_stage():
    agents = [_agent("a"), _agent("b"), _agent("c")]
    swarm = Swarm("s", agents, build_topology("sequential"))
    res = swarm.run("do the thing")
    assert res.ok()
    assert len(res.per_agent) == 3
    assert res.per_agent[-1].output == res.final


def test_parallel_without_aggregator_concatenates():
    agents = [_agent("a"), _agent("b")]
    res = Swarm("s", agents, build_topology("parallel")).run("task")
    assert res.ok()
    assert "[a]" in res.final and "[b]" in res.final


def test_parallel_workers_use_one_initial_blackboard_snapshot():
    agents = [_agent("a"), _agent("b"), _agent("c")]
    res = Swarm("s", agents, build_topology("parallel")).run("task")

    assert res.ok()
    # No worker should see another worker's output while fan-out is running.
    for agent_result in res.per_agent:
        assert "- a:" not in agent_result.output
        assert "- b:" not in agent_result.output
        assert "- c:" not in agent_result.output


def test_parallel_with_aggregator_uses_it():
    agents = [_agent("a"), _agent("b"), _agent("reducer")]
    res = Swarm("s", agents, build_topology("parallel", aggregator="reducer")).run("task")
    assert res.ok()
    # Final is the aggregator's own output (echo persona tag present).
    assert res.per_agent[-1].name == "reducer"
    assert res.per_agent[-1].output == res.final


def test_supervisor_fans_out_and_synthesizes_offline():
    agents = [_agent("lead", "supervisor"), _agent("w1"), _agent("w2")]
    res = Swarm("s", agents, build_topology("supervisor", supervisor="lead")).run("solve X")
    assert res.ok()
    # supervisor synth + two workers.
    names = [r.name for r in res.per_agent]
    assert "lead" in names and "w1" in names and "w2" in names


def test_debate_with_judge_returns_verdict():
    agents = [_agent("pro"), _agent("con"), _agent("judge")]
    res = Swarm("s", agents, build_topology("debate", rounds=2, judge="judge")).run("Q?")
    assert res.ok()
    assert res.per_agent[-1].name == "judge"
    assert res.per_agent[-1].output == res.final


def test_router_keyword_picks_matching_agent():
    agents = [
        _agent("mathbot", role="mathematics and arithmetic specialist"),
        _agent("bio", role="biology specialist"),
    ]
    res = Swarm("s", agents, build_topology("router")).run(
        "Please handle this arithmetic calculation problem"
    )
    assert res.ok()
    assert len(res.per_agent) == 1
    assert res.per_agent[0].name == "mathbot"


def test_unknown_topology_raises():
    import pytest

    with pytest.raises(ValueError):
        build_topology("nonsense")
