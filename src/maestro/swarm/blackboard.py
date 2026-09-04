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

    def snapshot(self) -> Blackboard:
        """Return an isolated copy of the current working memory.

        A snapshot is used at the start of fan-out work: workers may read the
        same initial state, but one worker cannot change another worker's prompt
        halfway through a run.
        """
        copy = Blackboard()
        with self._lock:
            copy._facts = dict(self._facts)
            copy._log = list(self._log)
        return copy

    def merge_from(self, other: Blackboard, message_offset: int = 0) -> None:
        """Merge a worker's post-snapshot changes into this blackboard.

        ``message_offset`` is the number of messages present in the snapshot,
        so the initial transcript is not duplicated.  Facts are merged in the
        caller's chosen order, which lets a topology provide deterministic
        semantics even when workers finish out of order.
        """
        with other._lock:
            messages = list(other._log[message_offset:])
            facts = dict(other._facts)
        with self._lock:
            self._log.extend(messages)
            self._facts.update(facts)
