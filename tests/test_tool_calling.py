"""Tool calling — one declaration, both tracks.

Covers the per-provider schema adaptation, the native function-calling loop, and
the ReAct fallback that keeps tools working on backends without native support.
"""

from __future__ import annotations

import json

from maestro.agents.llm_agent import LLMAgent
from maestro.providers import anthropic_provider, ollama_provider, openai_provider
from maestro.providers._http import HttpResult
from maestro.providers.base import EchoClient, LLMClient, LLMResponse
from maestro.swarm.context import RunContext
from maestro.telemetry.usage import Usage
from maestro.tools.base import FunctionTool, Tool
from maestro.tools.builtin import builtin_tools
from maestro.tools.schema import (
    ToolCall,
    parse_arguments,
    to_anthropic,
    to_ollama,
    to_openai,
    to_react_catalogue,
)

CALC = builtin_tools()["calc"]


class RichTool(Tool):
    """A tool with a real multi-argument schema."""

    name = "convert"
    description = "Convert a value between units."
    parameters = {
        "type": "object",
        "properties": {
            "value": {"type": "number"},
            "to": {"type": "string", "enum": ["c", "f"]},
        },
        "required": ["value", "to"],
    }

    def run(self, arg: str) -> str:
        return f"react:{arg}"

    def call(self, arguments: dict) -> str:
        return f"native:{arguments['value']}{arguments['to']}"


# --------------------------------------------------------------------------- #
# One declaration, adapted per provider
# --------------------------------------------------------------------------- #


def test_default_schema_is_a_single_string_argument():
    schema = CALC.schema()
    assert schema["type"] == "object"
    assert list(schema["properties"]) == ["input"]
    assert schema["required"] == ["input"]


def test_declared_schema_is_used_verbatim():
    assert RichTool().schema()["required"] == ["value", "to"]


def test_anthropic_adapter_shape():
    [entry] = to_anthropic([CALC])
    assert set(entry) == {"name", "description", "input_schema"}
    assert entry["name"] == "calc"
    assert entry["input_schema"] == CALC.schema()


def test_openai_adapter_shape():
    [entry] = to_openai([CALC])
    assert entry["type"] == "function"
    assert entry["function"]["name"] == "calc"
    assert entry["function"]["parameters"] == CALC.schema()


def test_ollama_reuses_the_openai_function_shape():
    assert to_ollama([CALC]) == to_openai([CALC])


def test_react_catalogue_lists_every_tool():
    text = to_react_catalogue([CALC, RichTool()])
    assert "- calc:" in text and "- convert:" in text


def test_every_adapter_sees_the_same_single_declaration():
    # The point of the layer: adding a tool once reaches all three dialects.
    tools = [CALC, RichTool()]
    names = {"calc", "convert"}
    assert {t["name"] for t in to_anthropic(tools)} == names
    assert {t["function"]["name"] for t in to_openai(tools)} == names
    assert {t["function"]["name"] for t in to_ollama(tools)} == names


def test_parse_arguments_handles_every_dialect():
    assert parse_arguments({"input": "x"}) == {"input": "x"}  # Anthropic/Ollama
    assert parse_arguments('{"input": "x"}') == {"input": "x"}  # OpenAI JSON string
    assert parse_arguments("") == {}
    assert parse_arguments(None) == {}
    # Unparseable arguments preserve the model's intent instead of raising.
    assert parse_arguments("not json") == {"input": "not json"}
    assert parse_arguments("[1, 2]") == {"input": [1, 2]}


def test_call_collapses_arguments_for_a_single_string_tool():
    assert CALC.call({"input": "2*(3+4)"}) == "14"
    assert CALC.call({"expression": "2*(3+4)"}) == "14"  # lone value, any key
    assert CALC.call({}) != ""  # no crash on empty arguments


def test_call_can_be_overridden_for_real_structured_arguments():
    assert RichTool().call({"value": 20, "to": "f"}) == "native:20f"


def test_function_tool_accepts_a_custom_schema():
    tool = FunctionTool("echo", "Echo it.", lambda s: s, parameters={"type": "object"})
    assert tool.schema() == {"type": "object"}


# --------------------------------------------------------------------------- #
# Provider request/response wiring
# --------------------------------------------------------------------------- #


def _capture(monkeypatch, module, body: str) -> dict:
    seen: dict = {}

    def fake_post(url, payload, headers=None, timeout=120.0, **kwargs):
        seen["payload"] = payload
        return HttpResult(status=200, body=body)

    monkeypatch.setattr(module, "post_json", fake_post)
    return seen


def test_anthropic_sends_tools_and_parses_tool_use(monkeypatch):
    body = json.dumps(
        {
            "model": "claude-sonnet-5",
            "stop_reason": "tool_use",
            "content": [
                {"type": "text", "text": "Let me compute that."},
                {"type": "tool_use", "id": "toolu_1", "name": "calc", "input": {"input": "6*7"}},
            ],
            "usage": {"input_tokens": 20, "output_tokens": 9},
        }
    )
    seen = _capture(monkeypatch, anthropic_provider, body)

    resp = anthropic_provider.AnthropicClient(api_key="k").complete([], tools=[CALC])

    assert seen["payload"]["tools"] == to_anthropic([CALC])
    assert resp.wants_tools()
    assert resp.tool_calls == [ToolCall(name="calc", arguments={"input": "6*7"}, id="toolu_1")]
    assert resp.stop_reason == "tool_use"


def test_anthropic_translates_tool_results_into_one_user_turn():
    history = [
        {"role": "user", "content": "what is 6*7?"},
        {
            "role": "assistant",
            "content": "computing",
            "tool_calls": [ToolCall(name="calc", arguments={"input": "6*7"}, id="toolu_1")],
        },
        {"role": "tool", "tool_call_id": "toolu_1", "name": "calc", "content": "42"},
        {"role": "tool", "tool_call_id": "toolu_2", "name": "calc", "content": "7"},
    ]
    out = anthropic_provider.to_messages(history)

    assert [m["role"] for m in out] == ["user", "assistant", "user"]
    assert out[1]["content"][0]["type"] == "text"
    assert out[1]["content"][1] == {
        "type": "tool_use",
        "id": "toolu_1",
        "name": "calc",
        "input": {"input": "6*7"},
    }
    # Both results must ride in a *single* user turn.
    assert len(out[2]["content"]) == 2
    assert out[2]["content"][0]["type"] == "tool_result"
    assert out[2]["content"][0]["tool_use_id"] == "toolu_1"


def test_openai_sends_tools_and_parses_tool_calls(monkeypatch):
    body = json.dumps(
        {
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "calc", "arguments": '{"input": "6*7"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 9},
        }
    )
    seen = _capture(monkeypatch, openai_provider, body)

    resp = openai_provider.OpenAICompatClient(api_key="k").complete([], tools=[CALC])

    assert seen["payload"]["tools"] == to_openai([CALC])
    assert resp.tool_calls[0].arguments == {"input": "6*7"}  # JSON string decoded
    assert resp.tool_calls[0].id == "call_1"


def test_openai_serializes_tool_history_in_its_own_dialect():
    history = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [ToolCall(name="calc", arguments={"input": "6*7"}, id="call_1")],
        },
        {"role": "tool", "tool_call_id": "call_1", "name": "calc", "content": "42"},
    ]
    out = openai_provider.to_messages(history)

    assert out[0]["tool_calls"][0]["function"]["arguments"] == '{"input": "6*7"}'
    assert out[1] == {"role": "tool", "tool_call_id": "call_1", "content": "42"}


def test_ollama_sends_tools_and_parses_tool_calls(monkeypatch):
    body = json.dumps(
        {
            "model": "llama3.1",
            "message": {
                "content": "",
                "tool_calls": [{"function": {"name": "calc", "arguments": {"input": "6*7"}}}],
            },
            "prompt_eval_count": 20,
            "eval_count": 9,
        }
    )
    seen = _capture(monkeypatch, ollama_provider, body)

    resp = ollama_provider.OllamaClient().complete([], tools=[CALC])

    assert seen["payload"]["tools"] == to_ollama([CALC])
    assert resp.tool_calls[0].name == "calc"
    # Ollama sends no correlation id, so one is synthesized to pair the result.
    assert resp.tool_calls[0].id == "ollama-0"


def test_no_tools_means_no_tools_key_in_the_payload(monkeypatch):
    seen = _capture(monkeypatch, anthropic_provider, json.dumps({"content": []}))
    anthropic_provider.AnthropicClient(api_key="k").complete([])
    assert "tools" not in seen["payload"]


# --------------------------------------------------------------------------- #
# The agent's two tracks
# --------------------------------------------------------------------------- #


class NativeToolClient(LLMClient):
    """A backend that answers with one native tool call, then with prose."""

    name = "native-fake"
    supports_tools = True

    def __init__(self):
        self.turns = 0
        self.seen_tools = None
        self.histories: list[list[dict]] = []

    @property
    def available(self):
        return True

    def complete(self, messages, system="", max_tokens=None, temperature=None, tools=None):
        self.turns += 1
        self.seen_tools = tools
        self.histories.append(list(messages))
        if self.turns == 1:
            return LLMResponse(
                model="native-1",
                usage=Usage(provider=self.name, model="native-1", input_tokens=10, output_tokens=5),
                tool_calls=[ToolCall(name="calc", arguments={"input": "2*(3+4)"}, id="t1")],
                stop_reason="tool_use",
            )
        return LLMResponse(
            text="The result is 14.",
            model="native-1",
            usage=Usage(provider=self.name, model="native-1", input_tokens=20, output_tokens=6),
        )


class ScriptedClient(LLMClient):
    """Returns queued responses in order — drives the ReAct loop."""

    name = "scripted"

    def __init__(self, outputs):
        self._outputs = list(outputs)
        self.systems: list[str] = []

    @property
    def available(self):
        return True

    def complete(self, messages, system="", max_tokens=None, temperature=None):
        self.systems.append(system)
        text = self._outputs.pop(0) if self._outputs else "done"
        return LLMResponse(text=text, model="scripted")


def test_native_track_runs_the_tool_and_feeds_the_result_back():
    client = NativeToolClient()
    agent = LLMAgent("calculator", client, role="analyst", tools=[CALC])
    ctx = RunContext(task="t")

    res = agent.run("what is 2*(3+4)?", ctx)

    assert agent.uses_native_tools()
    assert res.ok() and "14" in res.output
    assert res.meta["tool_protocol"] == "native"
    # The tool schemas were advertised, and the result came back as a tool turn.
    assert client.seen_tools == [CALC]
    second_turn = client.histories[1]
    assert second_turn[-2]["role"] == "assistant"
    assert second_turn[-2]["tool_calls"][0].name == "calc"
    assert second_turn[-1] == {
        "role": "tool",
        "tool_call_id": "t1",
        "name": "calc",
        "content": "14",
    }
    assert any(e.kind == "tool_call" for e in ctx.tracer.events)


def test_native_track_is_not_taken_when_the_backend_cannot_do_it():
    agent = LLMAgent("a", ScriptedClient(["done"]), tools=[CALC])
    assert agent.uses_native_tools() is False


def test_react_track_still_works_on_a_backend_without_native_tools():
    client = ScriptedClient(["ACTION: calc 2*(3+4)", "The result is 14."])
    agent = LLMAgent("calculator", client, role="analyst", tools=[CALC], max_tool_iters=3)
    ctx = RunContext(task="t")

    res = agent.run("what is 2*(3+4)?", ctx)

    assert res.ok() and "14" in res.output
    assert res.meta["tool_protocol"] == "react"
    # The text protocol has to be taught in the system prompt.
    assert "ACTION:" in client.systems[0] and "- calc:" in client.systems[0]


def test_native_track_omits_the_react_instructions_from_the_system_prompt():
    client = NativeToolClient()
    LLMAgent("a", client, tools=[CALC], system_prompt="Be terse.").run("x")
    assert client.turns  # sanity
    agent = LLMAgent("a", NativeToolClient(), tools=[CALC], system_prompt="Be terse.")
    assert "ACTION:" not in agent._system(native=True)
    assert "ACTION:" in agent._system(native=False)


def test_native_tools_can_be_forced_off_on_a_capable_backend():
    client = NativeToolClient()
    agent = LLMAgent("a", client, tools=[CALC], native_tools=False)
    assert agent.uses_native_tools() is False
    agent.run("x")
    assert client.seen_tools is None  # never advertised


def test_unknown_tool_is_reported_back_to_the_model_not_raised():
    class UnknownToolClient(NativeToolClient):
        def complete(self, messages, system="", max_tokens=None, temperature=None, tools=None):
            self.turns += 1
            self.histories.append(list(messages))
            if self.turns == 1:
                return LLMResponse(
                    tool_calls=[ToolCall(name="nope", arguments={}, id="t1")],
                    usage=Usage(provider="x", model="y"),
                )
            return LLMResponse(text="ok", usage=Usage(provider="x", model="y"))

    client = UnknownToolClient()
    res = LLMAgent("a", client, tools=[CALC]).run("x")

    assert res.ok()
    assert "unknown tool" in client.histories[1][-1]["content"]


def test_agent_without_tools_makes_exactly_one_call():
    client = ScriptedClient(["just an answer"])
    res = LLMAgent("a", client).run("x")
    assert res.output == "just an answer"
    assert len(client.systems) == 1


def test_echo_backend_keeps_the_react_track():
    # The offline default has no native tool support, so tools still work there.
    agent = LLMAgent("a", EchoClient(), tools=[CALC])
    assert agent.uses_native_tools() is False
