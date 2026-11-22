import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("fastapi")
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient

from maestro import system1_serve as server
from maestro.agents.base import AgentResult
from maestro.swarm.swarm import Swarm
from maestro.topologies.system1 import System1Topology


@pytest.mark.parametrize("key", [None, "", "  "])
def test_unconfigured_inference_fails_closed(key):
    with pytest.raises(HTTPException) as error:
        server._require_auth(key, "Bearer anything")
    assert error.value.status_code == 503


@pytest.mark.parametrize("header,status", [(None, 401), ("Basic key", 401), ("Bearer wrong", 403)])
def test_invalid_credentials(header, status):
    with pytest.raises(HTTPException) as error:
        server._require_auth("key", header)
    assert error.value.status_code == status


def test_endpoint_requires_auth_before_running_swarm(monkeypatch):
    agent = Mock(name="decision-agent")
    agent.name = "decision"
    agent.kind = "decision"
    agent.role = "decision"
    agent.run.return_value = AgentResult(name="decision", output='{"answers":{}}')
    swarm = Swarm("test", [agent], System1Topology())
    swarm.run = Mock(wraps=swarm.run)
    monkeypatch.setattr(
        server, "_orchestrator", SimpleNamespace(spec=SimpleNamespace(name="test"), swarm=swarm)
    )
    monkeypatch.setenv("MAESTRO_SYSTEM1_API_KEY", "key")

    async def exercise():
        async with AsyncClient(
            transport=ASGITransport(app=server.app), base_url="http://test"
        ) as client:
            payload = {
                "state": "hello",
                "questions": {"yes": {"type": "noul", "instructions": "Decide"}},
            }
            assert (await client.post("/v1/systemone", json=payload)).status_code == 401
            assert (
                await client.post(
                    "/v1/systemone", json=payload, headers={"Authorization": "Bearer wrong"}
                )
            ).status_code == 403
            swarm.run.assert_not_called()
            assert (
                await client.post(
                    "/v1/systemone", json=payload, headers={"Authorization": "Bearer key"}
                )
            ).status_code == 200
            swarm.run.assert_called_once()
            assert agent.run.call_args.args[0] == "hello"
            context = agent.run.call_args.args[1]
            assert context.task == "hello"
            assert context.scratch["questions"]["yes"]["instructions"] == "Decide"
            monkeypatch.delenv("MAESTRO_SYSTEM1_API_KEY")
            assert (await client.post("/v1/systemone", json=payload)).status_code == 503
            assert swarm.run.call_count == 1

    asyncio.run(exercise())


def test_asgi_startup_requires_key(monkeypatch):
    monkeypatch.delenv("MAESTRO_SYSTEM1_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="API_KEY"):

        async def start():
            async with server.lifespan(server.app):
                pass

        start().send(None)


def test_cli_binds_loopback(monkeypatch):
    import uvicorn

    runner = Mock()
    monkeypatch.setattr(uvicorn, "run", runner)
    monkeypatch.setenv("MAESTRO_SYSTEM1_API_KEY", "key")
    monkeypatch.delenv("MAESTRO_SYSTEM1_HOST", raising=False)
    server.run_server()
    assert runner.call_args.kwargs["host"] == "127.0.0.1"


def test_endpoint_keyless_direct_call(monkeypatch):
    monkeypatch.setattr(server, "_orchestrator", SimpleNamespace())
    monkeypatch.delenv("MAESTRO_SYSTEM1_API_KEY", raising=False)
    request = server.SystemOneRequest(state="test", questions={})
    with pytest.raises(HTTPException) as error:
        server.system_one(request, authorization="Bearer anything").send(None)
    assert error.value.status_code == 503


def test_typed_question_models_and_decision_result_serialization():
    from maestro.agents.decision_agent import (
        ChoiceAnswer,
        DecisionResult,
        NoulAnswer,
        ScoreAnswer,
    )
    from maestro.agents.registry import _parse_questions

    req = server.SystemOneRequest(
        state="status",
        questions={
            "c": {"type": "choice", "instructions": "Pick", "criteria": {"a": "A"}},
            "s": {"type": "score", "instructions": "Rate", "criteria": ["low", "high"]},
            "n": {"type": "noul", "instructions": "Check"},
        },
    )
    assert isinstance(req.questions["c"], server.ChoiceQuestion)
    assert isinstance(req.questions["s"], server.ScoreQuestion)
    assert isinstance(req.questions["n"], server.NoulQuestion)

    parsed = _parse_questions(
        {
            "c": {"type": "choice", "instructions": "Pick", "criteria": {"a": "A"}},
            "s": {"type": "score", "instructions": "Rate", "criteria": ["low", "high"]},
            "n": {"type": "noul", "instructions": "Check"},
        }
    )
    assert set(parsed) == {"c", "s", "n"}

    res = DecisionResult(
        model="m1",
        answers={
            "c": ChoiceAnswer(choice="a", probabilities={"a": 1.0}, confidence=1.0),
            "s": ScoreAnswer(score=1.0, probabilities={1: 1.0}, confidence=1.0, legend={1: "hi"}),
            "n": NoulAnswer(noul=0.9),
        },
        usage={"input_tokens": 1, "output_tokens": 2},
    )
    payload = res._to_dict()
    assert payload["model"] == "m1"
    assert set(payload["answers"]) == {"c", "s", "n"}
