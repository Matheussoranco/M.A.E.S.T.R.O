"""Environment-driven settings.

All configuration is read from ``MAESTRO_*`` (and standard provider) environment
variables so nothing secret is ever committed.  Everything has a sensible
default; the object is cheap to construct and safe to import with no env at all.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


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
        default_factory=lambda: _env("MAESTRO_OLLAMA_BASE_URL", "OLLAMA_HOST",
                                     default="http://localhost:11434")
    )
    openai_base_url: str = field(
        default_factory=lambda: _env("MAESTRO_OPENAI_BASE_URL", "OPENAI_BASE_URL",
                                     default="https://api.openai.com/v1")
    )
    anthropic_base_url: str = field(
        default_factory=lambda: _env("MAESTRO_ANTHROPIC_BASE_URL",
                                     default="https://api.anthropic.com")
    )
    llamacpp_base_url: str = field(
        default_factory=lambda: _env("MAESTRO_LLAMACPP_BASE_URL",
                                     default="http://localhost:8080/v1")
    )

    # When a requested provider is unavailable (missing key, unreachable host)
    # fall back to the deterministic offline ``echo`` backend instead of failing,
    # so a swarm's structure can always be exercised.  On by default.
    allow_stub_fallback: bool = field(
        default_factory=lambda: _bool("MAESTRO_ALLOW_STUB", True)
    )

    # Per-request ceilings.
    request_timeout: float = field(
        default_factory=lambda: float(_env("MAESTRO_TIMEOUT", default="120"))
    )
    max_tokens: int = field(default_factory=lambda: int(_env("MAESTRO_MAX_TOKENS", default="1024")))

    # Commands used to enlist the sibling projects as swarm members.  Overridable
    # so users with per-project virtualenvs can point at the right interpreter,
    # e.g. ``MAESTRO_ISAAC_CMD="C:/…/I.S.A.A.C/.venv/Scripts/python.exe -m isaac agent"``.
    isaac_cmd: str = field(
        default_factory=lambda: _env("MAESTRO_ISAAC_CMD", default="isaac agent")
    )
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
