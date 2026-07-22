"""Tiny JSON-over-HTTP helper built on the standard library only.

Keeping the transport dependency-free is what lets MAESTRO talk to Anthropic,
OpenAI, any OpenAI-compatible gateway, and a local Ollama server without pulling
in a single third-party package.
"""

from __future__ import annotations

import contextlib
import json
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass
class HttpResult:
    status: int = 0
    body: str = ""
    error: str = ""

    def json(self) -> dict:
        try:
            return json.loads(self.body) if self.body else {}
        except json.JSONDecodeError as exc:
            return {"_parse_error": str(exc), "_raw": self.body[:2000]}


def post_json(
    url: str,
    payload: dict,
    headers: dict[str, str] | None = None,
    timeout: float = 120.0,
) -> HttpResult:
    """POST ``payload`` as JSON, returning an :class:`HttpResult` (never raises)."""
    data = json.dumps(payload).encode("utf-8")
    hdrs = {"Content-Type": "application/json", "Accept": "application/json"}
    hdrs.update(headers or {})
    req = urllib.request.Request(url, data=data, headers=hdrs, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", "replace")
            return HttpResult(status=getattr(resp, "status", 200), body=body)
    except urllib.error.HTTPError as exc:
        detail = ""
        with contextlib.suppress(Exception):
            detail = exc.read().decode("utf-8", "replace")
        return HttpResult(status=exc.code, error=f"HTTP {exc.code}: {detail[:500]}")
    except urllib.error.URLError as exc:
        return HttpResult(error=f"connection error: {exc.reason}")
    except TimeoutError:
        return HttpResult(error=f"request timed out after {timeout}s")
    except Exception as exc:
        return HttpResult(error=f"transport error: {exc}")
