"""Swarm runtime primitives: messages, blackboard, run-context, the Swarm.

``Swarm`` is exposed lazily: it depends on the agent + topology layers, which in
turn import the light primitives below, so importing it eagerly here would form a
cycle.  The primitives (Message/Blackboard/RunContext) have no such dependency
and are imported directly.
"""

from __future__ import annotations

from maestro.swarm.blackboard import Blackboard
from maestro.swarm.context import RunContext
from maestro.swarm.message import Message

__all__ = ["Blackboard", "Message", "RunContext", "Swarm"]


def __getattr__(name: str):
    if name == "Swarm":
        from maestro.swarm.swarm import Swarm

        return Swarm
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
