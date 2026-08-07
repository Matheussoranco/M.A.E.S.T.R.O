"""Streaming — the contract, the per-provider parsers, and topology flow-through.

Every test runs against a fake transport: no network, no keys.
"""

from __future__ import annotations

import json

import pytest

from maestro.agents.llm_agent import LLMAgent
from maestro.providers import anthropic_provider, ollama_provider, openai_provider
from maestro.providers._http import HttpStreamError
from maestro.providers.base import (
    EchoClient,
    LLMClient,
    LLMResponse,
    NullClient,
    StreamEvent,
    collect_stream,
)
from maestro.swarm.swarm import Swarm
from maestro.topologies import build_topology


def _fake_stream(monkeypatch, module, lines, capture=None):
    """Swap a provider's streaming transport for one that replays *lines*."""

    def fake(url, payload, headers=None, timeout=120.0):
        if capture is not None:
            capture["url"] = url
            capture["payload"] = payload
        yield from lines

    monkeypatch.setattr(module, "stream_lines", fake)


# --------------------------------------------------------------------------- #
# The contract itself
# --------------------------------------------------------------------------- #


class NonStreamingClient(LLMClient):
    """A 0.1-era backend: implements complete() only, unaware of streaming."""

    name = "legacy"

    def __init__(self):
        self.calls = []

    @property
    def available(self):
        return True

    def complete(self, messages, system="", max_tokens=None, temperature=None):
        # NB: the 0.1 signature — no `tools` parameter.  Passing one would be a
        # TypeError, which is exactly the regression this client guards.
        self.calls.append(messages)
        return LLMResponse(text="whole answer at once", model="legacy-1")


def test_non_streaming_backend_still_satisfies_the_stream_interface():
    client = NonStreamingClient()
    events = list(client.complete_stream([{"role": "user", "content": "hi"}]))

    # Fallback, not failure: text arrives in one chunk, then the terminal event.
    assert [e.text for e in events if e.text] == ["whole answer at once"]
    assert events[-1].done and events[-1].response is not None
    assert events[-1].response.text == "whole answer at once"
    assert client.supports_streaming is False


def test_streaming_never_passes_tools_to_a_legacy_client():
    # A 0.1-signature client must not receive a `tools` kwarg it cannot accept.
    client = NonStreamingClient()
    events = list(client.complete_stream([{"role": "user", "content": "hi"}], tools=None))
    assert events[-1].done


def test_terminal_event_is_emitted_even_for_an_error():
    events = list(NullClient().complete_stream([{"role": "user", "content": "x"}]))
    assert len(events) == 1
    assert events[0].done and events[0].error
    assert events[0].response is not None and events[0].response.error


def test_echo_streams_incrementally_and_reassembles():
    client = EchoClient(persona="tester")
    events = list(client.complete_stream([{"role": "user", "content": "hello world"}]))
    fragments = [e.text for e in events if e.text]

    assert client.supports_streaming is True
    assert len(fragments) > 1  # genuinely incremental, not one blob
    # Concatenating the fragments reproduces the completion exactly.
    assert "".join(fragments) == events[-1].response.text
    assert (
        events[-1].response.text
        == client.complete([{"role": "user", "content": "hello world"}]).text
    )


def test_collect_stream_returns_the_terminal_response():
    resp = collect_stream(iter(EchoClient().complete_stream([{"role": "user", "content": "hi"}])))
    assert resp.ok()


def test_collect_stream_salvages_a_stream_with_no_terminal_event():
    def broken():
        yield StreamEvent(text="half ")
        yield StreamEvent(text="an answer")

    resp = collect_stream(broken())
    assert resp.text == "half an answer"


# --------------------------------------------------------------------------- #
# Anthropic SSE
# --------------------------------------------------------------------------- #

ANTHROPIC_SSE = [
    "event: message_start",
    'data: {"type":"message_start","message":{"model":"claude-sonnet-5",'
    '"usage":{"input_tokens":11,"output_tokens":0}}}',
    "",
    "event: content_block_start",
    'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
    "",
    'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}',
    'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}',
    'data: {"type":"content_block_stop","index":0}',
    'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":7}}',
    'data: {"type":"message_stop"}',
]


def test_anthropic_stream_reassembles_text_and_usage(monkeypatch):
    seen: dict = {}
    _fake_stream(monkeypatch, anthropic_provider, ANTHROPIC_SSE, seen)
    client = anthropic_provider.AnthropicClient(api_key="k")

    events = list(client.complete_stream([{"role": "user", "content": "hi"}]))

    assert seen["payload"]["stream"] is True
    assert [e.text for e in events if e.text] == ["Hello", " world"]
    final = events[-1].response
    assert final.text == "Hello world"
    assert final.stop_reason == "end_turn"
    assert final.usage.input_tokens == 11 and final.usage.output_tokens == 7
    assert final.usage.provider == "anthropic"


def test_anthropic_stream_accumulates_partial_tool_json(monkeypatch):
    # Tool arguments arrive as JSON fragments that only parse once joined.
    lines = [
        'data: {"type":"content_block_start","index":0,'
        '"content_block":{"type":"tool_use","id":"toolu_1","name":"calc"}}',
        'data: {"type":"content_block_delta","index":0,'
        '"delta":{"type":"input_json_delta","partial_json":"{\\"inp"}}',
        'data: {"type":"content_block_delta","index":0,'
        '"delta":{"type":"input_json_delta","partial_json":"ut\\": \\"2+2\\"}"}}',
        'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},'
        '"usage":{"output_tokens":5}}',
    ]
    _fake_stream(monkeypatch, anthropic_provider, lines)

    final = list(anthropic_provider.AnthropicClient(api_key="k").complete_stream([]))[-1].response

    assert final.wants_tools()
    assert final.tool_calls[0].name == "calc"
    assert final.tool_calls[0].arguments == {"input": "2+2"}
    assert final.tool_calls[0].id == "toolu_1"


def test_anthropic_stream_reports_transport_failure_as_a_terminal_event(monkeypatch):
    def boom(url, payload, headers=None, timeout=120.0):
        raise HttpStreamError("HTTP 500: upstream exploded")
        yield  # pragma: no cover - unreachable, makes this a generator

    monkeypatch.setattr(anthropic_provider, "stream_lines", boom)

    events = list(anthropic_provider.AnthropicClient(api_key="k").complete_stream([]))

    # A failed stream must never raise into the swarm.
    assert len(events) == 1 and events[0].done
    assert "upstream exploded" in events[0].error
    assert events[0].response.error


def test_anthropic_stream_without_key_yields_an_error_event():
    events = list(anthropic_provider.AnthropicClient(api_key="").complete_stream([]))
    assert len(events) == 1 and events[0].done
    assert "ANTHROPIC_API_KEY" in events[0].error


# --------------------------------------------------------------------------- #
# OpenAI-compatible SSE
# --------------------------------------------------------------------------- #


def test_openai_stream_reassembles_text_and_usage(monkeypatch):
    lines = [
        'data: {"model":"gpt-4o-mini","choices":[{"delta":{"content":"Hel"}}]}',
        'data: {"model":"gpt-4o-mini","choices":[{"delta":{"content":"lo"},'
        '"finish_reason":"stop"}]}',
        'data: {"model":"gpt-4o-mini","choices":[],'
        '"usage":{"prompt_tokens":4,"completion_tokens":2}}',
        "data: [DONE]",
    ]
    seen: dict = {}
    _fake_stream(monkeypatch, openai_provider, lines, seen)

    events = list(
        openai_provider.OpenAICompatClient(api_key="k").complete_stream(
            [{"role": "user", "content": "hi"}]
        )
    )

    # Usage only arrives if we asked for it.
    assert seen["payload"]["stream_options"] == {"include_usage": True}
    assert [e.text for e in events if e.text] == ["Hel", "lo"]
    final = events[-1].response
    assert final.text == "Hello"
    assert final.usage.input_tokens == 4 and final.usage.output_tokens == 2
    assert final.stop_reason == "stop"


def test_openai_stream_joins_tool_call_argument_fragments(monkeypatch):
    lines = [
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1",'
        '"function":{"name":"calc","arguments":"{\\"inp"}}]}}]}',
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
        '"function":{"arguments":"ut\\": \\"7*6\\"}"}}]},"finish_reason":"tool_calls"}]}',
        "data: [DONE]",
    ]
    _fake_stream(monkeypatch, openai_provider, lines)

    final = list(openai_provider.OpenAICompatClient(api_key="k").complete_stream([]))[-1].response

    assert final.tool_calls[0].name == "calc"
    assert final.tool_calls[0].arguments == {"input": "7*6"}


def test_openai_stream_without_usage_frame_reports_unknown_not_zero(monkeypatch):
    # A gateway that ignores stream_options tells us nothing about tokens.
    _fake_stream(
        monkeypatch,
        openai_provider,
        ['data: {"choices":[{"delta":{"content":"hi"}}]}', "data: [DONE]"],
    )
    final = list(openai_provider.OpenAICompatClient(api_key="k").complete_stream([]))[-1].response

    assert final.text == "hi"
    assert final.usage.input_tokens is None
    assert final.usage.measured is False


# --------------------------------------------------------------------------- #
# Ollama NDJSON
# --------------------------------------------------------------------------- #


def test_ollama_stream_reassembles_ndjson(monkeypatch):
    lines = [
        json.dumps({"model": "llama3.1", "message": {"content": "Hel"}, "done": False}),
        json.dumps({"model": "llama3.1", "message": {"content": "lo"}, "done": False}),
        json.dumps(
            {
                "model": "llama3.1",
                "message": {"content": ""},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 12,
                "eval_count": 3,
            }
        ),
    ]
    seen: dict = {}
    _fake_stream(monkeypatch, ollama_provider, lines, seen)

    events = list(ollama_provider.OllamaClient().complete_stream([]))

    assert seen["payload"]["stream"] is True
    assert [e.text for e in events if e.text] == ["Hel", "lo"]
    final = events[-1].response
    assert final.text == "Hello"
    assert final.usage.input_tokens == 12 and final.usage.output_tokens == 3


def test_ollama_stream_collects_tool_calls(monkeypatch):
    lines = [
        json.dumps(
            {
                "message": {
                    "content": "",
                    "tool_calls": [{"function": {"name": "calc", "arguments": {"input": "1+1"}}}],
                },
                "done": True,
            }
        )
    ]
    _fake_stream(monkeypatch, ollama_provider, lines)

    final = list(ollama_provider.OllamaClient().complete_stream([]))[-1].response
    assert final.tool_calls[0].name == "calc"
    assert final.tool_calls[0].arguments == {"input": "1+1"}


# --------------------------------------------------------------------------- #
# Flow-through: agents, and the topologies that aggregate their output
# --------------------------------------------------------------------------- #


def _agent(name, role=""):
    return LLMAgent(name, EchoClient(persona=name), role=role)


def test_agent_forwards_fragments_and_still_returns_a_whole_result():
    seen: list[tuple[str, str]] = []
    swarm = Swarm("s", [_agent("writer")], build_topology("sequential"))

    streamed = swarm.run("summarize the news", on_token=lambda a, t: seen.append((a, t)))
    plain = Swarm("s", [_agent("writer")], build_topology("sequential")).run("summarize the news")

    assert [name for name, _ in seen] == ["writer"] * len(seen)
    assert "".join(text for _, text in seen) == streamed.final
    # Streaming is observational: the aggregated result is byte-identical.
    assert streamed.final == plain.final


@pytest.mark.parametrize(
    "topology,params",
    [
        ("sequential", {}),
        ("parallel", {"aggregator": "reducer"}),
        ("parallel", {}),
        ("supervisor", {"supervisor": "lead"}),
        ("debate", {"rounds": 2, "judge": "reducer"}),
        ("router", {}),
    ],
)
def test_streaming_does_not_change_what_a_topology_aggregates(topology, params):
    """The final answer must not depend on whether anyone was watching."""

    def build():
        return [_agent("lead", "supervisor"), _agent("worker"), _agent("reducer")]

    plain = Swarm("s", build(), build_topology(topology, **params)).run("solve X")

    seen: list[str] = []
    streamed = Swarm("s", build(), build_topology(topology, **params)).run(
        "solve X", on_token=lambda a, t: seen.append(t)
    )

    assert streamed.final == plain.final
    assert streamed.ok() == plain.ok()
    assert [r.name for r in streamed.per_agent] == [r.name for r in plain.per_agent]
    assert seen, "streaming produced no fragments"


def test_parallel_streaming_labels_every_fragment_with_its_agent():
    # Workers run concurrently; the sink must still be able to tell them apart.
    agents = [_agent("a"), _agent("b"), _agent("c")]
    seen: list[tuple[str, str]] = []
    Swarm("s", agents, build_topology("parallel")).run(
        "task", on_token=lambda a, t: seen.append((a, t))
    )

    by_agent: dict[str, str] = {}
    for name, text in seen:
        by_agent[name] = by_agent.get(name, "") + text

    assert set(by_agent) == {"a", "b", "c"}
    for name, text in by_agent.items():
        assert f"[{name}#" in text  # each agent's own persona-tagged output


def test_stream_flag_works_without_a_sink():
    # Exercising the streaming path with no consumer must still produce a result.
    res = Swarm("s", [_agent("solo")], build_topology("sequential")).run("task", stream=True)
    assert res.ok()
