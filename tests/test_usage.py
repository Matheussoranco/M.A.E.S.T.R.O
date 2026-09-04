"""Token & cost accounting.

The load-bearing property throughout: **unknown is not zero**.  A backend that
reports nothing must never be presented as having consumed nothing.
"""

from __future__ import annotations

import pytest

from maestro.agents.llm_agent import LLMAgent
from maestro.orchestrator import Orchestrator
from maestro.providers.base import EchoClient, LLMClient, LLMResponse
from maestro.swarm.swarm import Swarm
from maestro.telemetry.usage import (
    Usage,
    UsageTotals,
    known_prices,
    price_for,
    register_price,
)
from maestro.topologies import build_topology
from maestro.topologies.base import SwarmResult


class SilentClient(LLMClient):
    """A backend that answers but reports no usage at all (many gateways do)."""

    name = "silent"

    def __init__(self, model: str = "mystery-1"):
        self.model = model

    @property
    def available(self):
        return True

    def complete(self, messages, system="", max_tokens=None, temperature=None, tools=None):
        return LLMResponse(text="an answer", model=self.model, usage=Usage(provider=self.name))


class MeteredClient(LLMClient):
    """A backend that reports usage, including a genuine zero."""

    name = "metered"

    def __init__(self, model="claude-opus-5", input_tokens=1000, output_tokens=0):
        self.model = model
        self._in = input_tokens
        self._out = output_tokens

    @property
    def available(self):
        return True

    def complete(self, messages, system="", max_tokens=None, temperature=None, tools=None):
        return LLMResponse(
            text="an answer",
            model=self.model,
            usage=Usage(
                provider=self.name,
                model=self.model,
                input_tokens=self._in,
                output_tokens=self._out,
            ),
        )


# --------------------------------------------------------------------------- #
# Usage: the tri-state
# --------------------------------------------------------------------------- #


def test_unreported_usage_is_unknown_not_zero():
    unknown = Usage(provider="p", model="m")
    assert unknown.measured is False
    assert unknown.input_tokens is None
    assert unknown.total_tokens is None
    assert unknown.as_dict()["input_tokens"] is None


def test_a_reported_zero_is_a_measurement():
    measured = Usage(provider="p", model="m", input_tokens=0, output_tokens=0)
    assert measured.measured is True
    assert measured.total_tokens == 0


def test_partially_reported_usage_still_counts_as_measured():
    partial = Usage(provider="p", model="m", output_tokens=7)
    assert partial.measured is True
    assert partial.total_tokens == 7


def test_unmeasured_usage_has_no_cost_even_for_a_priced_model():
    assert Usage(provider="anthropic", model="claude-opus-5").cost() is None


def test_unpriced_model_has_unknown_cost_not_free():
    usage = Usage(provider="x", model="some-local-model", input_tokens=1_000_000)
    assert usage.cost() is None


# --------------------------------------------------------------------------- #
# Prices
# --------------------------------------------------------------------------- #


def test_price_lookup_is_a_substring_match():
    assert price_for("claude-opus-5") == (5.0, 25.0)
    # Dated snapshots and vendor-prefixed ids resolve to the same entry.
    assert price_for("claude-opus-5-20260401") == (5.0, 25.0)
    assert price_for("anthropic.claude-sonnet-5") == (3.0, 15.0)
    assert price_for("CLAUDE-HAIKU-4-5") == (1.0, 5.0)


def test_unknown_model_has_no_price():
    assert price_for("llama3.1") is None
    assert price_for("") is None


def test_cost_arithmetic_is_per_million_tokens():
    usage = Usage(
        provider="anthropic", model="claude-opus-5", input_tokens=1_000_000, output_tokens=1_000_000
    )
    assert usage.cost() == pytest.approx(30.0)  # $5 in + $25 out


def test_register_price_teaches_a_new_model():
    register_price("test-local-model", 0.0, 0.0)
    try:
        usage = Usage(provider="x", model="test-local-model", input_tokens=5_000)
        assert usage.cost() == 0.0  # priced at zero — different from "unknown"
        assert "test-local-model" in known_prices()
    finally:
        known_prices()  # copy; the module table keeps the entry, which is fine


def test_per_call_prices_override_the_table():
    usage = Usage(provider="x", model="claude-opus-5", input_tokens=1_000_000)
    assert usage.cost() == pytest.approx(5.0)
    assert usage.cost({"claude-opus-5": (1.0, 1.0)}) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# UsageTotals: rollup
# --------------------------------------------------------------------------- #


def test_rollup_sums_and_breaks_down_by_agent_and_provider():
    totals = UsageTotals.from_agent_usage(
        {
            "lead": [
                Usage(
                    provider="anthropic", model="claude-opus-5", input_tokens=100, output_tokens=10
                ),
                Usage(
                    provider="anthropic", model="claude-opus-5", input_tokens=200, output_tokens=20
                ),
            ],
            "worker": [
                Usage(provider="ollama", model="llama3.1", input_tokens=50, output_tokens=5),
            ],
        }
    )

    assert totals.calls == 3
    assert totals.input_tokens == 350 and totals.output_tokens == 35
    assert totals.total_tokens == 385
    assert totals.per_agent["lead"].calls == 2
    assert totals.per_agent["lead"].input_tokens == 300
    assert totals.per_agent["worker"].input_tokens == 50
    assert set(totals.per_provider) == {"anthropic", "ollama"}
    assert totals.per_provider["anthropic"].output_tokens == 30


def test_rollup_counts_unmeasured_calls_separately():
    totals = UsageTotals.from_agent_usage(
        {
            "a": [
                Usage(provider="anthropic", model="claude-opus-5", input_tokens=10, output_tokens=1)
            ],
            "b": [Usage(provider="silent", model="mystery")],
        }
    )

    assert totals.calls == 2
    assert totals.unmeasured_calls == 1
    assert totals.tokens_complete is False  # the total is a floor, not a measurement
    assert totals.input_tokens == 10  # the silent call adds nothing, not zero


def test_rollup_cost_is_partial_when_some_models_are_unpriced():
    totals = UsageTotals.from_agent_usage(
        {
            "a": [
                Usage(
                    provider="anthropic",
                    model="claude-opus-5",
                    input_tokens=1_000_000,
                    output_tokens=0,
                )
            ],
            "b": [Usage(provider="ollama", model="llama3.1", input_tokens=500, output_tokens=100)],
        }
    )

    assert totals.cost == pytest.approx(5.0)
    assert totals.unpriced_calls == 1
    assert totals.cost_complete is False  # $5.00 is not the whole bill
    assert "partial" in totals.render()


def test_rollup_cost_is_none_when_nothing_can_be_priced():
    totals = UsageTotals.from_agent_usage(
        {"a": [Usage(provider="ollama", model="llama3.1", input_tokens=10, output_tokens=2)]}
    )
    assert totals.cost is None
    assert "cost unknown" in totals.render()


def test_empty_rollup_is_not_a_measurement():
    totals = UsageTotals.from_agent_usage({})
    assert totals.calls == 0
    assert totals.tokens_complete is False and totals.cost_complete is False
    assert totals.cost is None


def test_rollup_dict_view_is_json_safe():
    totals = UsageTotals.from_agent_usage(
        {"a": [Usage(provider="p", model="claude-opus-5", input_tokens=1, output_tokens=2)]}
    )
    data = totals.as_dict()
    assert data["total_tokens"] == 3
    assert data["per_agent"]["a"]["input_tokens"] == 1
    assert data["cost_complete"] is True


def test_render_warns_when_totals_are_incomplete():
    totals = UsageTotals.from_agent_usage({"a": [Usage(provider="silent", model="m")]})
    assert "reported no usage" in totals.render()


# --------------------------------------------------------------------------- #
# SwarmResult surface
# --------------------------------------------------------------------------- #


def test_swarm_result_surfaces_totals_from_a_real_run():
    agents = [
        LLMAgent("lead", MeteredClient(input_tokens=100, output_tokens=20), role="supervisor"),
        LLMAgent("worker", MeteredClient(input_tokens=50, output_tokens=10)),
    ]
    res = Swarm("s", agents, build_topology("sequential")).run("do it")

    assert res.usage.calls == 2
    assert res.usage.input_tokens == 150 and res.usage.output_tokens == 30
    assert set(res.usage.per_agent) == {"lead", "worker"}
    assert res.usage.per_provider["metered"].calls == 2
    assert res.usage.cost_complete  # claude-opus-5 is priced


def test_discarded_calls_are_still_billed_and_still_counted():
    """A supervisor's planning turn costs tokens even though its result is dropped."""
    agents = [
        LLMAgent("lead", MeteredClient(input_tokens=100, output_tokens=20), role="supervisor"),
        LLMAgent("w1", MeteredClient(input_tokens=10, output_tokens=1)),
    ]
    res = Swarm("s", agents, build_topology("supervisor", supervisor="lead")).run("solve X")

    # per_agent holds 2 results (synthesis + worker), but 3 calls were made:
    # the lead planned, the worker ran, the lead synthesized.
    assert len(res.per_agent) == 2
    assert res.usage.calls == 3
    assert res.usage.per_agent["lead"].calls == 2


def test_router_routing_call_is_counted_too():
    agents = [
        LLMAgent("router", MeteredClient(input_tokens=5, output_tokens=1)),
        LLMAgent("specialist", MeteredClient(input_tokens=10, output_tokens=2)),
    ]
    res = Swarm("s", agents, build_topology("router", router="router")).run("question")

    assert res.usage.calls == 2
    assert "router" in res.usage.per_agent


def test_mixed_measured_and_silent_backends_report_honestly():
    agents = [
        LLMAgent("metered", MeteredClient(input_tokens=100, output_tokens=20)),
        LLMAgent("silent", SilentClient()),
    ]
    res = Swarm("s", agents, build_topology("parallel")).run("task")

    totals = res.usage
    assert totals.calls == 2
    assert totals.unmeasured_calls == 1
    assert totals.tokens_complete is False
    assert totals.per_agent["silent"].unmeasured_calls == 1
    assert totals.per_agent["silent"].input_tokens == 0  # nothing added…
    assert totals.per_agent["silent"].tokens_complete is False  # …but not measured


def test_non_llm_agent_contributes_no_usage():
    result = SwarmResult(per_agent=[])
    assert result.usage.calls == 0


def test_hand_built_result_falls_back_to_per_agent_usage():
    from maestro.agents.base import AgentResult

    result = SwarmResult(
        per_agent=[
            AgentResult(
                "a",
                output="x",
                usage=[Usage(provider="p", model="claude-opus-5", input_tokens=4, output_tokens=2)],
            )
        ]
    )
    assert result.usage.calls == 1 and result.usage.input_tokens == 4


def test_usage_totals_accepts_a_price_override():
    agents = [LLMAgent("a", MeteredClient(model="local-thing", input_tokens=1_000_000))]
    res = Swarm("s", agents, build_topology("sequential")).run("task")

    assert res.usage.cost is None  # unpriced by default
    priced = res.usage_totals(prices={"local-thing": (2.0, 4.0)})
    assert priced.cost == pytest.approx(2.0)


def test_offline_echo_run_reports_tokens_but_unknown_cost():
    # The default offline backend counts words; nobody sells "echo-1", so the
    # honest cost is unknown rather than $0.00.
    agents = [LLMAgent("a", EchoClient(persona="a"))]
    res = Swarm("s", agents, build_topology("sequential")).run("task")

    assert res.usage.tokens_complete is True
    assert res.usage.cost is None


# --------------------------------------------------------------------------- #
# Spec-declared prices
# --------------------------------------------------------------------------- #


def test_spec_prices_flow_into_the_rollup():
    spec = {
        "name": "priced",
        "topology": "sequential",
        "providers": {"stub": {"provider": "echo", "model": "priced-echo"}},
        "agents": [{"name": "a", "type": "llm", "provider": "stub"}],
        "prices": {"priced-echo": {"input": 1.0, "output": 2.0}},
    }
    res = Orchestrator.from_dict(spec).run("hello there", trace=False)

    assert res.usage.cost is not None
    assert res.usage.cost_complete


def test_spec_prices_do_not_leak_into_other_swarms():
    first = {
        "name": "first",
        "topology": "sequential",
        "providers": {"stub": {"provider": "echo", "model": "isolated-price-model"}},
        "agents": [{"name": "a", "provider": "stub"}],
        "prices": {"isolated-price-model": {"input": 1.0, "output": 2.0}},
    }
    second = {
        "name": "second",
        "topology": "sequential",
        "providers": {"stub": {"provider": "echo", "model": "isolated-price-model"}},
        "agents": [{"name": "a", "provider": "stub"}],
    }

    priced = Orchestrator.from_dict(first).run("hello", trace=False)
    unpriced = Orchestrator.from_dict(second).run("hello", trace=False)

    assert priced.usage.cost_complete
    assert unpriced.usage.cost is None


def test_spec_prices_accept_the_sequence_form():
    from maestro.orchestrator.spec import SwarmSpec

    spec = SwarmSpec.from_dict(
        {
            "name": "s",
            "agents": [{"name": "a"}],
            "prices": {"m1": [3, 15], "m2": {"input": 1, "output": 5}},
        }
    )
    assert spec.prices == {"m1": (3.0, 15.0), "m2": (1.0, 5.0)}


@pytest.mark.parametrize("bad", ["nonsense", {"m": "free"}, {"m": [1]}, {"m": {"input": "x"}}])
def test_malformed_prices_are_rejected_early(bad):
    from maestro.orchestrator.spec import SwarmSpec

    with pytest.raises(ValueError):
        SwarmSpec.from_dict({"name": "s", "agents": [{"name": "a"}], "prices": bad})
