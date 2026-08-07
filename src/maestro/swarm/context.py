"""The per-run context handed to every agent.

Bundles the root task, the shared :class:`Blackboard`, the run tracer, a
free-form ``scratch`` dict, and the optional token sink used for streaming.
Agents read prior outputs from the blackboard to stay coordinated; they never
need a direct reference to one another.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field

from maestro.swarm.blackboard import Blackboard
from maestro.swarm.message import Message
from maestro.telemetry.tracer import Tracer


@dataclass
class RunContext:
    task: str = ""
    blackboard: Blackboard = field(default_factory=Blackboard)
    tracer: Tracer = field(default_factory=Tracer)
    scratch: dict = field(default_factory=dict)
    depth: int = 0
    #: Ask streaming-capable agents to stream.  Off by default, so enabling it
    #: is always an explicit choice and never changes an existing run.
    stream: bool = False
    #: Sink for streamed fragments, called as ``on_token(agent_name, text)``.
    on_token: Callable[[str, str], None] | None = None
    #: Serializes the sink — parallel/supervisor topologies run agents in
    #: threads, and interleaved fragments must not corrupt the consumer.
    _token_lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def post(self, sender: str, content: str, role: str = "assistant", **meta) -> Message:
        msg = Message(sender=sender, content=content, role=role, meta=meta)
        self.blackboard.post(msg)
        return msg

    def emit_token(self, sender: str, text: str) -> None:
        """Forward one streamed fragment to the sink, if there is one.

        Streaming is *observational*: fragments go to the sink as they arrive
        while the agent still returns a complete :class:`AgentResult`, so every
        topology aggregates exactly the same text it always did.
        """
        if not text or self.on_token is None:
            return
        with self._token_lock:
            self.on_token(sender, text)

    def context_digest(self, limit: int = 6, width: int = 500) -> str:
        """A short digest of recent contributions, for prompting downstream agents."""
        msgs = self.blackboard.messages()[-limit:]
        if not msgs:
            return ""
        lines = ["Prior contributions from the swarm:"]
        lines.extend(f"- {m.short(width)}" for m in msgs)
        return "\n".join(lines)

    def child(self) -> RunContext:
        """A nested context that shares the blackboard/tracer but tracks depth."""
        return RunContext(
            task=self.task,
            blackboard=self.blackboard,
            tracer=self.tracer,
            scratch=self.scratch,
            depth=self.depth + 1,
            stream=self.stream,
            on_token=self.on_token,
            # Share the parent's lock so concurrent children serialize against
            # each other, not just against themselves.
            _token_lock=self._token_lock,
        )
