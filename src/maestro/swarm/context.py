"""The per-run context handed to every agent.

Bundles the root task, the shared :class:`Blackboard`, the run tracer, and a
free-form ``scratch`` dict.  Agents read prior outputs from the blackboard to
stay coordinated; they never need a direct reference to one another.
"""

from __future__ import annotations

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

    def post(self, sender: str, content: str, role: str = "assistant", **meta) -> Message:
        msg = Message(sender=sender, content=content, role=role, meta=meta)
        self.blackboard.post(msg)
        return msg

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
        )
