"""Test bootstrap: put ``src`` on the path and pin a deterministic environment."""

from __future__ import annotations

import os
import sys

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Deterministic, offline defaults for the whole suite.
os.environ.setdefault("MAESTRO_PROVIDER", "echo")
os.environ.setdefault("MAESTRO_ALLOW_STUB", "1")
# Ensure no real keys leak in from the host and change availability under test.
os.environ.pop("ANTHROPIC_API_KEY", None)
os.environ.pop("OPENAI_API_KEY", None)
os.environ.pop("MAESTRO_ANTHROPIC_API_KEY", None)
os.environ.pop("MAESTRO_OPENAI_API_KEY", None)

import importlib  # noqa: E402

# NB: ``maestro.config`` re-exports the ``settings`` instance, which shadows the
# submodule attribute — use import_module to reach the real module object.
_settings_mod = importlib.import_module("maestro.config.settings")
_settings_mod.reload()
