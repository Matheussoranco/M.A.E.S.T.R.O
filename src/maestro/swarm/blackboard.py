"""A shared blackboard — the swarm's common working memory.

Agents post messages and read/write named facts.  It is thread-safe so a
parallel topology can fan out across worker threads and still share state.
"""

from __future__ import annotations

import threading

from maestro.swarm.message import Message


class Blackboard:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._facts: dict[str, object] = {}
        self._log: list[Message] = []

    # -- named facts ----------------------------------------------------------
    def set(self, key: str, value: object) -> None:
        with self._lock:
            self._facts[key] = value

    def get(self, key: str, default: object = None) -> object:
        with self._lock:
            return self._facts.get(key, default)

    def facts(self) -> dict[str, object]:
        with self._lock:
            return dict(self._facts)

    # -- message log ----------------------------------------------------------
    def post(self, message: Message) -> None:
        with self._lock:
            self._log.append(message)

    def messages(self) -> list[Message]:
        with self._lock:
            return list(self._log)

    def transcript(self, width: int = 100) -> str:
        with self._lock:
            return "\n".join(m.short(width) for m in self._log)
