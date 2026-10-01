"""Decision-model agents — System 1 models that return typed, calibrated decisions.

A *decision model* evaluates a state and typed questions (choice / score / noul)
in a single forward pass, returning structured answers with probabilities.
This is the Jev / Laya / System One contract.

Two backends are supported:
- **Laya** — local PyTorch or ONNX Runtime (no API key, self-hosted)
- **Jev** — TypeSafe's hosted API (requires API key, sub-500ms)
"""

from __future__ import annotations

import json
import os
import warnings
from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from maestro.agents.base import Agent, AgentResult
from maestro.swarm.context import RunContext
from maestro.telemetry.usage import Usage


# -----------------------------------------------------------------------------
# Decision primitives (match TypeSafe Jev / Laya wire protocol)
# -----------------------------------------------------------------------------

@dataclass
class ChoiceQuestion:
    """Pick one option from a defined set."""
    type: str = "choice"
    instructions: str = ""
    criteria: Dict[str, Optional[str]] = field(default_factory=dict)  # option -> description (None = no desc)

@dataclass
class ScoreQuestion:
    """Rate the state on an ordered rubric."""
    type: str = "score"
    instructions: str = ""
    criteria: List[str] = field(default_factory=list)  # level descriptions, 0..N

@dataclass
class NoulQuestion:
    """Is this statement true? Returns P(true)."""
    type: str = "noul"
    instructions: str = ""
    criteria: Optional[Dict[str, str]] = None  # {"true": "...", "false": "..."} optional
    labels: Optional[Dict[str, str]] = None    # {"true": "A", "false": "B"} optional


Question = Union[ChoiceQuestion, ScoreQuestion, NoulQuestion]


@dataclass
class ChoiceAnswer:
    choice: str
    probabilities: Dict[str, float]
    confidence: float

@dataclass
class ScoreAnswer:
    score: float
    probabilities: Dict[int, float]
    confidence: float
    legend: Dict[int, str]  # level -> description

@dataclass
class NoulAnswer:
    noul: float  # P(true)


Answer = Union[ChoiceAnswer, ScoreAnswer, NoulAnswer]


@dataclass
class DecisionResult:
    """Structured result from a decision-model call."""
    answers: Dict[str, Answer] = field(default_factory=dict)
    routing: Optional[Dict[str, Any]] = None  # Laya routing metadata
    usage: Dict[str, int] = field(default_factory=dict)  # input_tokens, output_tokens
    model: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(self._to_dict(), ensure_ascii=False, separators=(",", ":"))

    def _to_dict(self) -> Dict[str, Any]:
        out = {"answers": {}, "usage": self.usage}
        if self.model:
            out["model"] = self.model
        if self.routing:
            out["routing"] = self.routing
        for qid, ans in self.answers.items():
            if isinstance(ans, ChoiceAnswer):
                out["answers"][qid] = {
                    "type": "choice",
                    "choice": ans.choice,
                    "probabilities": ans.probabilities,
                    "confidence": ans.confidence,
                }
            elif isinstance(ans, ScoreAnswer):
                out["answers"][qid] = {
                    "type": "score",
                    "score": ans.score,
                    "probabilities": ans.probabilities,
                    "confidence": ans.confidence,
                    "legend": ans.legend,
                }
            elif isinstance(ans, NoulAnswer):
                out["answers"][qid] = {
                    "type": "noul",
                    "noul": ans.noul,
                }
        return out


# -----------------------------------------------------------------------------
# Abstract decision agent base
# -----------------------------------------------------------------------------

class DecisionAgent(Agent, ABC):
    """Base class for System 1 decision-model agents.

    Decision agents don't generate text — they evaluate typed questions against
    a state and return calibrated probabilities. The task string is the state;
    questions are passed via RunContext.metadata['questions'] or agent config.
    """

    kind = "decision"

    def __init__(
        self,
        name: str,
        role: str = "",
        description: str = "",
        default_questions: Optional[Dict[str, Question]] = None,
        timeout: float = 30.0,
    ) -> None:
        super().__init__(name=name, role=role, description=description)
        self.default_questions = default_questions or {}
        self.timeout = timeout

    def _get_questions(self, context: RunContext | None) -> Dict[str, Question]:
        """Extract questions from context or fall back to defaults."""
        if context and context.scratch.get("questions"):
            return context.scratch["questions"]
        return self.default_questions

    def _get_state(self, task: str, context: RunContext | None) -> Union[str, Dict, List]:
        """Extract state from task and context metadata."""
        # Allow state to be passed as structured data in metadata
        if context and "state" in context.scratch:
            return context.scratch["state"]
        # Otherwise task is the state
        return task


# -----------------------------------------------------------------------------
# Laya Agent (local PyTorch)
# -----------------------------------------------------------------------------

class LayaAgent(DecisionAgent):
    """Laya System 1 decision model via local PyTorch.

    Requires: `pip install laya` (pulls torch, transformers, huggingface_hub).
    Downloads ~400-800MB model weights on first use (cached in ~/.cache/huggingface).
    """

    kind = "laya"

    def __init__(
        self,
        name: str,
        role: str = "decision-model",
        description: str = "Laya — fast local System 1 decision engine",
        default_questions: Optional[Dict[str, Question]] = None,
        model: str = "router",  # "router", "english", "multilingual", "typed-decisions"
        device: Optional[str] = None,  # "cuda", "cpu", "mps", "xpu", None=auto
        preload: bool = False,
        max_loaded: int = 2,
        lang_guess: Optional[str] = None,
        default_model: str = "english",
        timeout: float = 30.0,
        preset: Optional[str] = None,
    ) -> None:
        # Apply preset if specified
        if preset and not default_questions:
            from .decision_agent import DECISION_PRESETS
            if preset in DECISION_PRESETS:
                default_questions = DECISION_PRESETS[preset]()
            else:
                warnings.warn(f"Unknown Laya preset: {preset}. Available: {list(DECISION_PRESETS.keys())}")

        super().__init__(name, role, description, default_questions, timeout)
        self.model = model
        self.device = device
        self.preload = preload
        self.max_loaded = max_loaded
        self.lang_guess = lang_guess
        self.default_model = default_model
        self._router = None
        self._agent = None

    @property
    def available(self) -> bool:
        try:
            import laya  # noqa: F401
            return True
        except ImportError:
            return False

    def _get_router(self):
        """Lazy-initialize the Laya Router."""
        if self._router is None:
            import laya
            self._router = laya.Router(
                preload=self.preload,
                device=self.device,
                max_loaded=self.max_loaded,
                lang_guess=self.lang_guess,
                default=self.default_model,
            )
        return self._router

    def run(self, task: str, context: RunContext | None = None) -> AgentResult:
        if context is not None:
            context.tracer.emit("agent_start", name=self.name, detail=self.role)

        if not self.available:
            return self._finish(
                context,
                AgentResult(
                    self.name,
                    self.role,
                    error="Laya not installed. Install with: pip install laya",
                ),
            )

        questions = self._get_questions(context)
        state = self._get_state(task, context)

        if not questions:
            return self._finish(
                context,
                AgentResult(
                    self.name,
                    self.role,
                    error="No questions provided for decision model. Pass via context.metadata['questions'] or agent default_questions.",
                ),
            )

        # Convert our Question dataclasses to Laya's dict format
        laya_questions = {}
        for qid, q in questions.items():
            if isinstance(q, ChoiceQuestion):
                laya_questions[qid] = {"type": "choice", "instructions": q.instructions, "criteria": q.criteria}
            elif isinstance(q, ScoreQuestion):
                laya_questions[qid] = {"type": "score", "instructions": q.instructions, "criteria": q.criteria}
            elif isinstance(q, NoulQuestion):
                qdict = {"type": "noul", "instructions": q.instructions}
                if q.criteria:
                    qdict["criteria"] = q.criteria
                if q.labels:
                    qdict["labels"] = q.labels
                laya_questions[qid] = qdict

        router = self._get_router()

        try:
            # Prepare predict kwargs
            predict_kwargs = {}
            if self.model != "router":
                predict_kwargs["model"] = self.model

            # Run prediction
            result = router.predict(state, laya_questions, **predict_kwargs)

            # Parse Laya result into our DecisionResult
            decision_result = self._parse_laya_result(result)

            # Build output as JSON for blackboard
            output = decision_result.to_json()

            return self._finish(
                context,
                AgentResult(
                    self.name,
                    self.role,
                    output=output,
                    meta={
                        "model": decision_result.model,
                        "routing": decision_result.routing,
                        "usage": decision_result.usage,
                        "answers": {k: v.__dict__ for k, v in decision_result.answers.items()},
                    },
                    usage=[Usage(
                        provider="laya-local",
                        model=decision_result.model or self.model,
                        input_tokens=decision_result.usage.get("input_tokens", 0),
                        output_tokens=decision_result.usage.get("output_tokens", 0),
                    )],
                ),
            )

        except Exception as exc:
            return self._finish(
                context,
                AgentResult(
                    self.name,
                    self.role,
                    error=f"Laya prediction failed: {exc}",
                ),
            )

    def _parse_laya_result(self, result: Dict) -> DecisionResult:
        """Convert Laya's raw dict result to DecisionResult."""
        answers = {}
        for qid, ans in result.get("answers", {}).items():
            atype = ans.get("type")
            if atype == "choice":
                answers[qid] = ChoiceAnswer(
                    choice=ans["choice"],
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

        return DecisionResult(
            answers=answers,
            routing=result.get("routing"),
            usage=result.get("usage", {}),
            model=result.get("model", ""),
        )


# -----------------------------------------------------------------------------
# Laya ONNX Agent (local ONNX Runtime)
# -----------------------------------------------------------------------------

class LayaONNXAgent(DecisionAgent):
    """Laya System 1 decision model via ONNX Runtime (no PyTorch at runtime).

    Requires: `pip install laya[onnx]` or `pip install onnxruntime transformers huggingface_hub`.
    Uses pre-exported ONNX model (~1.7GB fp32, can be INT8 quantized).
    """

    kind = "laya-onnx"

    def __init__(
        self,
        name: str,
        role: str = "decision-model",
        description: str = "Laya ONNX — fast CPU-optimized System 1 decisions",
        default_questions: Optional[Dict[str, Question]] = None,
        model_id: str = "convaiinnovations/laya",  # HF repo or local path
        onnx_path: str = "laya.onnx",
        token: Optional[str] = None,
        subfolder: Optional[str] = None,  # "multilingual", "typed-decisions"
        revision: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        super().__init__(name, role, description, default_questions, timeout)
        self.model_id = model_id
        self.onnx_path = onnx_path
        self.token = token
        self.subfolder = subfolder
        self.revision = revision
        self._agent = None

    @property
    def available(self) -> bool:
        try:
            import onnxruntime  # noqa: F401
            from transformers import AutoTokenizer  # noqa: F401
            return True
        except ImportError:
            return False

    def _get_agent(self):
        """Lazy-initialize the ONNX Agent."""
        if self._agent is None:
            from laya.onnx_agent import ONNXAgent
            self._agent = ONNXAgent(
                model_id_or_path=self.model_id,
                onnx_path=self.onnx_path,
                token=self.token,
                subfolder=self.subfolder,
                revision=self.revision,
            )
        return self._agent

    def run(self, task: str, context: RunContext | None = None) -> AgentResult:
        if context is not None:
            context.tracer.emit("agent_start", name=self.name, detail=self.role)

        if not self.available:
            return self._finish(
                context,
                AgentResult(
                    self.name,
                    self.role,
                    error="Laya ONNX dependencies not installed. Install with: pip install laya[onnx]",
                ),
            )

        questions = self._get_questions(context)
        state = self._get_state(task, context)

        if not questions:
            return self._finish(
                context,
                AgentResult(
                    self.name,
                    self.role,
                    error="No questions provided for decision model.",
                ),
            )

        # Convert to Laya ONNX format
        laya_questions = {}
        for qid, q in questions.items():
            if isinstance(q, ChoiceQuestion):
                laya_questions[qid] = {"type": "choice", "instructions": q.instructions, "criteria": q.criteria}
            elif isinstance(q, ScoreQuestion):
                laya_questions[qid] = {"type": "score", "instructions": q.instructions, "criteria": q.criteria}
            elif isinstance(q, NoulQuestion):
                qdict = {"type": "noul", "instructions": q.instructions}
                if q.criteria:
                    qdict["criteria"] = q.criteria
                if q.labels:
                    qdict["labels"] = q.labels
                laya_questions[qid] = qdict

        agent = self._get_agent()

        try:
            # ONNXAgent uses system_one API (matches Jev wire protocol)
            result = agent.system_one(state, laya_questions)

            # Parse result
            decision_result = self._parse_onnx_result(result)

            output = decision_result.to_json()

            return self._finish(
                context,
                AgentResult(
                    self.name,
                    self.role,
                    output=output,
                    meta={
                        "model": decision_result.model,
                        "usage": decision_result.usage,
                        "answers": {k: v.__dict__ for k, v in decision_result.answers.items()},
                    },
                    usage=[Usage(
                        provider="laya-onnx",
                        model=self.model_id,
                        input_tokens=decision_result.usage.get("input_tokens", 0),
                        output_tokens=decision_result.usage.get("output_tokens", 0),
                    )],
                ),
            )

        except Exception as exc:
            return self._finish(
                context,
                AgentResult(
                    self.name,
                    self.role,
                    error=f"Laya ONNX prediction failed: {exc}",
                ),
            )

    def _parse_onnx_result(self, result: Dict) -> DecisionResult:
        """ONNXAgent.system_one returns Jev-compatible format."""
        answers = {}
        for qid, ans in result.get("answers", {}).items():
            atype = ans.get("type")
            if atype == "choice":
                answers[qid] = ChoiceAnswer(
                    choice=ans["choice"],
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

        return DecisionResult(
            answers=answers,
            usage=result.get("usage", {}),
            model=result.get("model", ""),
        )


# -----------------------------------------------------------------------------
# Jev Agent (TypeSafe hosted API)
# -----------------------------------------------------------------------------

class JevAgent(DecisionAgent):
    """TypeSafe Jev — hosted System 1 decision model via HTTP API.

    Requires: `pip install typesafe-sdk` or direct HTTP calls.
    Needs TYPESAFE_API_KEY environment variable.
    ~70-500ms per call, no local model weights needed.
    """

    kind = "jev"

    def __init__(
        self,
        name: str,
        role: str = "decision-model",
        description: str = "Jev — TypeSafe hosted System 1 decision API",
        default_questions: Optional[Dict[str, Question]] = None,
        model: str = "jev-latest",  # "jev-latest", "jev-1.13.0", "jev-preview"
        api_key: Optional[str] = None,
        base_url: str = "https://api.typesafe.ai",
        timeout: float = 30.0,
    ) -> None:
        super().__init__(name, role, description, default_questions, timeout)
        self.model = model
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
        self.base_url = base_url.rstrip("/")
        self._client = None

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _get_client(self):
        """Lazy-initialize the Jev client."""
        if self._client is None:
            try:
                from typesafe_sdk import TypeSafeClient
                self._client = TypeSafeClient(
                    api_key=self.api_key,
                    default_model=self.model,
                    base_url=self.base_url,
                )
            except ImportError:
                # Fallback to raw HTTP
                self._client = "raw_http"
        return self._client

    def run(self, task: str, context: RunContext | None = None) -> AgentResult:
        if context is not None:
            context.tracer.emit("agent_start", name=self.name, detail=self.role)

        if not self.api_key:
            return self._finish(
                context,
                AgentResult(
                    self.name,
                    self.role,
                    error="Jev API key not configured. Set TYPESAFE_API_KEY or pass api_key to agent.",
                ),
            )

        questions = self._get_questions(context)
        state = self._get_state(task, context)

        if not questions:
            return self._finish(
                context,
                AgentResult(
                    self.name,
                    self.role,
                    error="No questions provided for decision model.",
                ),
            )

        # Convert to Jev SDK format
        jev_questions = {}
        for qid, q in questions.items():
            if isinstance(q, ChoiceQuestion):
                from typesafe_sdk import Choice
                jev_questions[qid] = Choice(instructions=q.instructions, criteria=q.criteria)
            elif isinstance(q, ScoreQuestion):
                from typesafe_sdk import Score
                jev_questions[qid] = Score(instructions=q.instructions, criteria=q.criteria)
            elif isinstance(q, NoulQuestion):
                from typesafe_sdk import Noul
                jev_questions[qid] = Noul(instructions=q.instructions, criteria=q.criteria, labels=q.labels)

        client = self._get_client()

        try:
            if client == "raw_http":
                result = self._call_raw_http(state, jev_questions)
            else:
                result = client.system_one(state=state, questions=jev_questions, model=self.model)

            # Parse Jev result
            decision_result = self._parse_jev_result(result)

            output = decision_result.to_json()

            return self._finish(
                context,
                AgentResult(
                    self.name,
                    self.role,
                    output=output,
                    meta={
                        "model": decision_result.model,
                        "usage": decision_result.usage,
                        "answers": {k: v.__dict__ for k, v in decision_result.answers.items()},
                    },
                    usage=[Usage(
                        provider="jev-api",
                        model=decision_result.model or self.model,
                        input_tokens=decision_result.usage.get("input_tokens", 0),
                        output_tokens=decision_result.usage.get("output_tokens", 0),
                    )],
                ),
            )

        except Exception as exc:
            return self._finish(
                context,
                AgentResult(
                    self.name,
                    self.role,
                    error=f"Jev API call failed: {exc}",
                ),
            )

    def _call_raw_http(self, state: Any, questions: Dict) -> Dict:
        """Fallback raw HTTP call when typesafe-sdk not available."""
        import urllib.request

        # Convert Jev SDK objects to plain dicts
        qdict = {}
        for qid, q in questions.items():
            if hasattr(q, "__dict__"):
                qdict[qid] = q.__dict__
            else:
                qdict[qid] = q

        payload = {
            "model": self.model,
            "state": state if isinstance(state, str) else json.dumps(state, ensure_ascii=False),
            "questions": qdict,
        }

        req = urllib.request.Request(
            f"{self.base_url}/v1/systemone",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _parse_jev_result(self, result: Any) -> DecisionResult:
        """Parse Jev SDK response or raw HTTP response."""
        # Handle both SDK response object and raw dict
        if hasattr(result, "answers"):
            answers_dict = result.answers
            usage = result.usage if hasattr(result, "usage") else {}
            model = result.model if hasattr(result, "model") else self.model
        else:
            answers_dict = result.get("answers", {})
            usage = result.get("usage", {})
            model = result.get("model", self.model)

        answers = {}
        for qid, ans in answers_dict.items():
            atype = ans.get("type") if isinstance(ans, dict) else getattr(ans, "type", None)
            if atype == "choice":
                choice = ans.get("choice") if isinstance(ans, dict) else getattr(ans, "choice", "")
                probabilities = ans.get("probabilities", {}) if isinstance(ans, dict) else getattr(ans, "probabilities", {})
                confidence = ans.get("confidence", 0.0) if isinstance(ans, dict) else getattr(ans, "confidence", 0.0)
                answers[qid] = ChoiceAnswer(choice=choice, probabilities=probabilities, confidence=confidence)
            elif atype == "score":
                score = ans.get("score", 0.0) if isinstance(ans, dict) else getattr(ans, "score", 0.0)
                probs_raw = ans.get("probabilities", {}) if isinstance(ans, dict) else getattr(ans, "probabilities", {})
                probs = {int(k): v for k, v in probs_raw.items()}
                legend_raw = ans.get("legend", {}) if isinstance(ans, dict) else getattr(ans, "legend", {})
                legend = {int(k): v for k, v in legend_raw.items()}
                confidence = ans.get("confidence", 0.0) if isinstance(ans, dict) else getattr(ans, "confidence", 0.0)
                answers[qid] = ScoreAnswer(score=score, probabilities=probs, confidence=confidence, legend=legend)
            elif atype == "noul":
                noul = ans.get("noul", 0.0) if isinstance(ans, dict) else getattr(ans, "noul", 0.0)
                answers[qid] = NoulAnswer(noul=noul)

        return DecisionResult(answers=answers, usage=usage, model=model)


# -----------------------------------------------------------------------------
# Decision presets (matching Laya's built-in presets)
# -----------------------------------------------------------------------------

def triage_questions() -> Dict[str, Question]:
    """Customer support triage preset."""
    return {
        "department": ChoiceQuestion(
            instructions="Which department should handle this request?",
            criteria={
                "billing": "invoices, payments, refunds",
                "technical": "bugs, outages, system errors",
                "sales": "pricing, new contracts",
                "other": "everything else",
            },
        ),
        "urgency": ScoreQuestion(
            instructions="How urgent is this request?",
            criteria=["not urgent", "soon", "critical deadline or blocking issue"],
        ),
        "churn_risk": NoulQuestion(
            instructions="Does the user threaten to cancel or leave?",
        ),
        "refund_requested": NoulQuestion(
            instructions="Does the user explicitly request a refund?",
        ),
    }


def email_questions() -> Dict[str, Question]:
    """Email classification preset."""
    return {
        "category": ChoiceQuestion(
            instructions="What type of email is this?",
            criteria={
                "inquiry": "general question or information request",
                "complaint": "expresses dissatisfaction or problem",
                "request": "asks for action or change",
                "spam": "unsolicited commercial or irrelevant",
                "other": "none of the above",
            },
        ),
        "needs_reply": NoulQuestion(
            instructions="Does this email require a reply?",
        ),
        "priority": ScoreQuestion(
            instructions="What is the priority of this email?",
            criteria=["low", "normal", "high", "urgent"],
        ),
    }


def guard_questions() -> Dict[str, Question]:
    """Content moderation / guardrail preset."""
    return {
        "violation": ChoiceQuestion(
            instructions="Does this content violate policy? If so, which category?",
            criteria={
                "none": "no violation",
                "hate": "hate speech or discrimination",
                "harassment": "targeted harassment or bullying",
                "violence": "threats or promotion of violence",
                "sexual": "explicit sexual content",
                "self_harm": "encouragement of self-harm",
                "pii": "personal identifiable information",
                "other": "other policy violation",
            },
        ),
        "severity": ScoreQuestion(
            instructions="How severe is the violation?",
            criteria=["none", "minor", "moderate", "severe", "critical"],
        ),
        "action_required": NoulQuestion(
            instructions="Does this require immediate moderation action?",
        ),
    }


def moderation_questions() -> Dict[str, Question]:
    """Detailed moderation preset."""
    return {
        "action": ChoiceQuestion(
            instructions="What moderation action is appropriate?",
            criteria={
                "allow": "content is fine",
                "flag": "flag for review",
                "remove": "remove content",
                "ban": "ban user",
            },
        ),
        "categories": ChoiceQuestion(
            instructions="Which policy categories apply?",
            criteria={
                "hate_speech": "hate speech",
                "harassment": "harassment",
                "violence": "violence",
                "sexual": "sexual content",
                "spam": "spam",
                "misinformation": "misinformation",
                "none": "no violation",
            },
        ),
    }


def router_questions() -> Dict[str, Question]:
    """Agent/task routing preset."""
    return {
        "route_to": ChoiceQuestion(
            instructions="Which agent or system should handle this task?",
            criteria={
                "coder": "code generation, debugging, refactoring",
                "researcher": "information gathering, analysis, synthesis",
                "analyst": "data analysis, metrics, reporting",
                "planner": "task decomposition, scheduling, strategy",
                "general": "general conversation, Q&A",
            },
        ),
        "complexity": ScoreQuestion(
            instructions="How complex is this task?",
            criteria=["trivial", "simple", "moderate", "complex", "very complex"],
        ),
        "requires_tools": NoulQuestion(
            instructions="Does this task require external tools (search, code execution, API)?",
        ),
    }


# Map of preset names to question builders
DECISION_PRESETS = {
    "triage": triage_questions,
    "email": email_questions,
    "guard": guard_questions,
    "moderation": moderation_questions,
    "router": router_questions,
}