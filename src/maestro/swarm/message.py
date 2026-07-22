"""The unit of communication between swarm members."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Message:
    """A single message posted by an agent (or the user) during a run."""

    sender: str
    content: str
    role: str = "assistant"  # "user" | "assistant" | "system"
    meta: dict = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def as_chat(self) -> dict[str, str]:
        """Render as an LLM chat message (user vs. assistant)."""
        role = "user" if self.role == "user" else "assistant"
        return {"role": role, "content": self.content}

    def short(self, width: int = 100) -> str:
        body = " ".join(self.content.split())
        if len(body) > width:
            body = body[: width - 1] + "…"
        return f"{self.sender}: {body}"
