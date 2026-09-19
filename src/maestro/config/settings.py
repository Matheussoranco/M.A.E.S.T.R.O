"""Environment-driven settings.

All configuration is read from ``MAESTRO_*`` (and standard provider) environment
variables so nothing secret is ever committed.  Everything has a sensible
default; the object is cheap to construct and safe to import with no env at all.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger("maestro.config")


def _env(*names: str, default: str = "") -> str:
    """Return the first non-empty environment variable among *names*."""
    for name in names:
        val = os.environ.get(name)
        if val:
            return val
    return default


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _default_stub_fallback() -> bool:
    """Default for ``allow_stub_fallback``: False, except demo/test.

    Explicit ``MAESTRO_ALLOW_STUB`` always wins.  Otherwise True only when
    running under pytest (``PYTEST_CURRENT_TEST`` set or ``pytest`` already
    imported) or when ``MAESTRO_DEMO=1`` is set — every other context
    defaults to False so production failures surface instead of silently
    degrading to the ``echo`` stub.
    """
    import sys as _sys

    raw = os.environ.get("MAESTRO_ALLOW_STUB")
    if raw is not None:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if os.environ.get("MAESTRO_DEMO", "").strip().lower() in {"1", "true", "yes", "on"}:
        return True
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return True
    return "pytest" in _sys.modules


def _env_float(name: str, default: float) -> float:
    """Parse ``os.environ[name]`` as float, falling back to *default*."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.warning("invalid float for %s=%r — using default %r", name, raw, default)
        return default


def _env_int(name: str, default: int) -> int:
    """Parse ``os.environ[name]`` as int, falling back to *default*."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except (TypeError, ValueError):
        logger.warning("invalid int for %s=%r — using default %r", name, raw, default)
        return default


@dataclass
class Settings:
    """Resolved runtime configuration.

    Instances read the process environment once, at construction time.  Call
    :func:`reload` (or build a fresh :class:`Settings`) after mutating ``os.environ``
    in tests.
    """

    # Default provider used when a swarm/agent does not name one explicitly.
    default_provider: str = field(default_factory=lambda: _env("MAESTRO_PROVIDER", default="echo"))
    default_model: str = field(default_factory=lambda: _env("MAESTRO_MODEL", default=""))

    # Credentials (read from the conventional variable names as well).
    anthropic_api_key: str = field(
        default_factory=lambda: _env("MAESTRO_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY")
    )
    openai_api_key: str = field(
        default_factory=lambda: _env("MAESTRO_OPENAI_API_KEY", "OPENAI_API_KEY")
    )

    # Endpoints for local / self-hosted back-ends.
    ollama_base_url: str = field(
        default_factory=lambda: _env(
            "MAESTRO_OLLAMA_BASE_URL", "OLLAMA_HOST", default="http://localhost:11434"
        )
    )
    openai_base_url: str = field(
        default_factory=lambda: _env(
            "MAESTRO_OPENAI_BASE_URL", "OPENAI_BASE_URL", default="https://api.openai.com/v1"
        )
    )
    anthropic_base_url: str = field(
        default_factory=lambda: _env(
            "MAESTRO_ANTHROPIC_BASE_URL", default="https://api.anthropic.com"
        )
    )
    llamacpp_base_url: str = field(
        default_factory=lambda: _env(
            "MAESTRO_LLAMACPP_BASE_URL", default="http://localhost:8080/v1"
        )
    )

    # When a requested provider is unavailable (missing key, unreachable host)
    # fall back to the deterministic offline ``echo`` backend instead of failing,
    # so a swarm's structure can always be exercised.  Off by default outside
    # demo/test: silent stub use in production hides billing/config errors.
    # Opt in explicitly via ``MAESTRO_ALLOW_STUB=1`` (or ``--allow-stub`` on
    # the CLI).  Under pytest the default stays True so the offline suite
    # remains deterministic without extra env setup.
    allow_stub_fallback: bool = field(default_factory=_default_stub_fallback)

    # Per-request ceilings.
    request_timeout: float = field(default_factory=lambda: _env_float("MAESTRO_TIMEOUT", 120.0))
    max_tokens: int = field(default_factory=lambda: _env_int("MAESTRO_MAX_TOKENS", 1024))

    # Commands used to enlist the sibling projects as swarm members.  Overridable
    # so users with per-project virtualenvs can point at the right interpreter,
    # e.g. ``MAESTRO_ISAAC_CMD="C:/…/I.S.A.A.C/.venv/Scripts/python.exe -m isaac agent"``.
    isaac_cmd: str = field(default_factory=lambda: _env("MAESTRO_ISAAC_CMD", default="isaac agent"))
    olivia_cmd: str = field(
        default_factory=lambda: _env("MAESTRO_OLIVIA_CMD", default="olivia ask")
    )

    def redacted(self) -> dict[str, object]:
        """A dict view safe to print — secrets replaced by a presence flag."""
        return {
            "default_provider": self.default_provider,
            "default_model": self.default_model or "(provider default)",
            "anthropic_api_key": bool(self.anthropic_api_key),
            "openai_api_key": bool(self.openai_api_key),
            "ollama_base_url": self.ollama_base_url,
            "openai_base_url": self.openai_base_url,
            "allow_stub_fallback": self.allow_stub_fallback,
            "request_timeout": self.request_timeout,
            "max_tokens": self.max_tokens,
        }


settings = Settings()


def reload() -> Settings:
    """Re-read the environment and refresh the module-level ``settings``."""
    global settings
    settings = Settings()
    return settings
