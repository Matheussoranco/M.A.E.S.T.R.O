"""A minimal, allocation-light event tracer.

Every topology records structured events here so a run can be replayed, printed,
or serialized.  The tracer is deliberately passive: it only stores what it is
told and never touches the network or disk unless asked to dump.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field


@dataclass
class Event:
    kind: str
    name: str = ""
    detail: str = ""
    t: float = field(default_factory=time.time)
    data: dict = field(default_factory=dict)


class Tracer:
    """Collects :class:`Event` records for one swarm run."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.events: list[Event] = []
        self._t0 = time.time()

    def emit(self, kind: str, name: str = "", detail: str = "", **data) -> None:
        if self.enabled:
            self.events.append(Event(kind=kind, name=name, detail=detail, data=data))

    def elapsed(self) -> float:
        return time.time() - self._t0

    def render(self) -> str:
        """A compact, human-readable timeline."""
        lines = []
        for ev in self.events:
            rel = ev.t - self._t0
            head = f"  [{rel:6.2f}s] {ev.kind:<14}"
            if ev.name:
                head += f" {ev.name}"
            if ev.detail:
                head += f" — {ev.detail}"
            lines.append(head)
        return "\n".join(lines)

    def to_list(self) -> list[dict]:
        return [asdict(e) for e in self.events]

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_list(), indent=indent, ensure_ascii=False)
