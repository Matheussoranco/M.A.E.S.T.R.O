"""Jev-compatible HTTP server — expose MAESTRO decision models over POST /v1/systemone.

This server implements the TypeSafe Jev wire protocol, allowing any Jev client
(typesafe-sdk, @typesafe-ai/sdk, hs-jev, etc.) to point at a local MAESTRO
instance running a System 1 swarm.

Configuration via environment variables:
- MAESTRO_SYSTEM1_HOST: bind address (default 0.0.0.0)
- MAESTRO_SYSTEM1_PORT: bind port (default 8001)
- MAESTRO_SYSTEM1_SWARM: path to swarm spec (YAML/JSON) with a system1 topology
- MAESTRO_SYSTEM1_API_KEY: if set, require Authorization: Bearer <key>
- MAESTRO_SYSTEM1_LOG_LEVEL: uvicorn log level (default info)
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from maestro.orchestrator import Orchestrator
from maestro.swarm.context import RunContext
from maestro.telemetry.tracer import Tracer

# -----------------------------------------------------------------------------
# Request/Response models (match TypeSafe Jev /v1/systemone)
# -----------------------------------------------------------------------------


class NoulCriteria(BaseModel):
    true: Optional[str] = None
    false: Optional[str] = None


class NoulLabels(BaseModel):
    true: str
    false: str


class ChoiceQuestion(BaseModel):
    type: str = "choice"
    instructions: str
    criteria: Dict[str, Optional[str]]


class ScoreQuestion(BaseModel):
    type: str = "score"
    instructions: str
    criteria: list[str]


class NoulQuestion(BaseModel):
    type: str = "noul"
    instructions: str
    criteria: Optional[NoulCriteria] = None
    labels: Optional[NoulLabels] = None


Question = ChoiceQuestion | ScoreQuestion | NoulQuestion


class SystemOneRequest(BaseModel):
    model: Optional[str] = None  # Ignored — topology determines model
    state: Any  # string, dict, or list
    questions: Dict[str, Question]
    max_len: Optional[int] = None
    head_max_len: Optional[int] = None
    min_confidence: Optional[float] = None


class ChoiceAnswer(BaseModel):
    type: str = "choice"
    choice: str
    probabilities: Dict[str, float]
    confidence: float


class ScoreAnswer(BaseModel):
    type: str = "score"
    score: float
    probabilities: Dict[int, float]
    confidence: float
    legend: Dict[int, str]


class NoulAnswer(BaseModel):
    type: str = "noul"
    noul: float


Answer = ChoiceAnswer | ScoreAnswer | NoulAnswer


class SystemOneResponse(BaseModel):
    model: str
    answers: Dict[str, Answer]
    usage: Dict[str, int]
    routing: Optional[Dict[str, Any]] = None


# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------

_log = logging.getLogger("maestro.system1_serve")

# Global orchestrator (initialized at startup)
_orchestrator: Optional[Orchestrator] = None


def _require_auth(api_key: Optional[str], authorization: Optional[str]) -> None:
    if not api_key:
        return
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization must be Bearer token")
    token = authorization[7:]
    if not hmac.compare_digest(token, api_key):
        raise HTTPException(status_code=403, detail="Invalid API key")


import hmac


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _orchestrator
    swarm_path = os.environ.get("MAESTRO_SYSTEM1_SWARM")
    if not swarm_path:
        raise RuntimeError("MAESTRO_SYSTEM1_SWARM environment variable must be set")
    _orchestrator = Orchestrator.from_file(swarm_path)
    _log.info("Loaded System 1 swarm from %s", swarm_path)
    yield
    _orchestrator = None


app = FastAPI(
    title="MAESTRO System 1 Decision API",
    description="Jev-compatible /v1/systemone endpoint for MAESTRO decision-model swarms",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {"status": "ok", "swarm": _orchestrator.spec.name if _orchestrator else "not loaded"}


@app.get("/v1/models")
async def list_models():
    """List available models — mirrors Jev's GET /v1/models."""
    if not _orchestrator:
        raise HTTPException(status_code=503, detail="Swarm not loaded")
    # Return the decision agent's model info
    agent = _orchestrator.swarm.agents[0]
    model_name = getattr(agent, "model", "laya-router")
    return {
        "data": [
            {"id": model_name, "object": "model", "owned_by": "maestro"},
        ]
    }


@app.post("/v1/systemone", response_model=SystemOneResponse)
async def system_one(
    request: SystemOneRequest,
    authorization: Optional[str] = Header(None),
):
    """Jev-compatible decision endpoint.

    Accepts a state and typed questions, returns calibrated probabilities.
    """
    global _orchestrator

    if _orchestrator is None:
        raise HTTPException(status_code=503, detail="Swarm not loaded")

    api_key = os.environ.get("MAESTRO_SYSTEM1_API_KEY")
    _require_auth(api_key, authorization)

    # Convert request to MAESTRO task + context
    state = request.state
    questions = {}

    for qid, q in request.questions.items():
        if q.type == "choice":
            questions[qid] = {
                "type": "choice",
                "instructions": q.instructions,
                "criteria": q.criteria,
            }
        elif q.type == "score":
            questions[qid] = {
                "type": "score",
                "instructions": q.instructions,
                "criteria": q.criteria,
            }
        elif q.type == "noul":
            qdict = {"type": "noul", "instructions": q.instructions}
            if q.criteria:
                qdict["criteria"] = {"true": q.criteria.true, "false": q.criteria.false}
            if q.labels:
                qdict["labels"] = {"true": q.labels.true, "false": q.labels.false}
            questions[qid] = qdict

    # Run via orchestrator with questions in metadata
    tracer = Tracer()
    context = RunContext(
        swarm_name=_orchestrator.spec.name,
        tracer=tracer,
        stream=False,
        scratch={"questions": questions},
    )

    # State becomes the task
    task = json.dumps(state, ensure_ascii=False) if not isinstance(state, str) else state

    result = _orchestrator.swarm.run(task, trace=True, on_token=None, stream=False, context=context)

    # Parse the decision agent's JSON output
    if not result.per_agent:
        raise HTTPException(status_code=500, detail="No agent result")

    agent_result = result.per_agent[0]
    if agent_result.error:
        raise HTTPException(status_code=500, detail=f"Decision agent error: {agent_result.error}")

    try:
        decision_data = json.loads(agent_result.output)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Decision agent returned invalid JSON")

    # Build Jev-compatible response
    answers = {}
    for qid, ans in decision_data.get("answers", {}).items():
        atype = ans.get("type")
        if atype == "choice":
            answers[qid] = ChoiceAnswer(
                choice=ans.get("choice", ""),
                probabilities=ans.get("probabilities", {}),
                confidence=ans.get("confidence", 0.0),
            )
        elif atype == "score":
            probs = {int(k): v for k, v in ans.get("probabilities", {}).items()}
            legend = {int(k): v for k, v in ans.get("legend", {}).items()}
            answers[qid] = ScoreAnswer(
                score=ans.get("score", 0.0),
                probabilities=probs,
                confidence=ans.get("confidence", 0.0),
                legend=legend,
            )
        elif atype == "noul":
            answers[qid] = NoulAnswer(noul=ans.get("noul", 0.0))

    usage = decision_data.get("usage", {"input_tokens": 0, "output_tokens": 0})
    model = decision_data.get("model", "maestro-system1")
    routing = decision_data.get("routing")

    return SystemOneResponse(
        model=model,
        answers=answers,
        usage=usage,
        routing=routing,
    )


def run_server():
    """Entry point for `maestro system1-serve` command."""
    import uvicorn

    host = os.environ.get("MAESTRO_SYSTEM1_HOST", "0.0.0.0")
    port = int(os.environ.get("MAESTRO_SYSTEM1_PORT", "8001"))
    log_level = os.environ.get("MAESTRO_SYSTEM1_LOG_LEVEL", "info")

    uvicorn.run(
        "maestro.system1_serve:app",
        host=host,
        port=port,
        log_level=log_level,
        reload=False,
    )


if __name__ == "__main__":
    run_server()