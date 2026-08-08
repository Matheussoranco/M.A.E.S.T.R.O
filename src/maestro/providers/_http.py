"""Tiny JSON-over-HTTP helper built on the standard library only.

Keeping the transport dependency-free is what lets MAESTRO talk to Anthropic,
OpenAI, any OpenAI-compatible gateway, and a local Ollama server without pulling
in a single third-party package.

Transient failures are retried *here* rather than in each provider, so every
backend inherits the same behaviour: HTTP 429 (rate limited) and 529 (server
overloaded) are retried with exponential backoff, honouring a ``retry-after``
header whenever the server sends one.

:func:`stream_lines` adds the streaming half of the transport — still stdlib
only, since ``urllib`` response objects are line-iterable as bytes arrive.
"""

from __future__ import annotations

import contextlib
import datetime
import email.utils
import json
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass, field

#: Status codes worth another attempt.  429 means "slow down" and 529 is
#: Anthropic's "overloaded"; both are explicitly transient.  Everything else —
#: ordinary 4xx client mistakes and 5xx faults — is returned to the caller as-is
#: so a bad request fails fast instead of being hammered three times.
RETRY_STATUSES = frozenset({429, 529})

#: Retries performed *after* the first attempt (so the default is 3 requests).
DEFAULT_MAX_RETRIES = 2
#: Seconds to wait before the first retry; doubled each time, clamped below.
DEFAULT_BACKOFF = 0.5
#: Upper bound on any single wait, including a server-supplied ``retry-after``.
MAX_BACKOFF = 30.0


@dataclass
class HttpResult:
    status: int = 0
    body: str = ""
    error: str = ""
    headers: dict[str, str] = field(default_factory=dict)

    def json(self) -> dict:
        try:
            return json.loads(self.body) if self.body else {}
        except json.JSONDecodeError as exc:
            return {"_parse_error": str(exc), "_raw": self.body[:2000]}

    def retry_after(self) -> float | None:
        """Seconds the server asked us to wait, or ``None`` if it did not say.

        RFC 9110 allows either a delay in seconds or an HTTP-date; both are
        accepted, and anything unparseable is treated as "no preference".
        """
        raw = ""
        for key, value in self.headers.items():
            if key.lower() == "retry-after":
                raw = str(value).strip()
                break
        if not raw:
            return None
        with contextlib.suppress(ValueError):
            return max(0.0, float(raw))
        with contextlib.suppress(Exception):
            when = email.utils.parsedate_to_datetime(raw)
            if when.tzinfo is None:  # HTTP-dates are GMT by definition.
                when = when.replace(tzinfo=datetime.timezone.utc)
            return max(0.0, when.timestamp() - time.time())
        return None


def _collect_headers(source: object) -> dict[str, str]:
    """Best-effort extraction of response headers from a response or HTTPError."""
    raw = getattr(source, "headers", None)
    if not raw:
        return {}
    with contextlib.suppress(Exception):
        return {str(k): str(v) for k, v in raw.items()}
    return {}


def _attempt(url: str, data: bytes, headers: dict[str, str], timeout: float) -> HttpResult:
    """One POST round-trip.  Never raises; failures come back as an HttpResult."""
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", "replace")
            return HttpResult(
                status=getattr(resp, "status", 200),
                body=body,
                headers=_collect_headers(resp),
            )
    except urllib.error.HTTPError as exc:
        detail = ""
        with contextlib.suppress(Exception):
            detail = exc.read().decode("utf-8", "replace")
        return HttpResult(
            status=exc.code,
            error=f"HTTP {exc.code}: {detail[:500]}",
            headers=_collect_headers(exc),
        )
    except urllib.error.URLError as exc:
        return HttpResult(error=f"connection error: {exc.reason}")
    except TimeoutError:
        return HttpResult(error=f"request timed out after {timeout}s")
    except Exception as exc:
        return HttpResult(error=f"transport error: {exc}")


def post_json(
    url: str,
    payload: dict,
    headers: dict[str, str] | None = None,
    timeout: float = 120.0,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff: float = DEFAULT_BACKOFF,
) -> HttpResult:
    """POST ``payload`` as JSON, returning an :class:`HttpResult` (never raises).

    Retries :data:`RETRY_STATUSES` up to *max_retries* times.  Pass
    ``max_retries=0`` to disable retrying entirely.
    """
    data = json.dumps(payload).encode("utf-8")
    hdrs = {"Content-Type": "application/json", "Accept": "application/json"}
    hdrs.update(headers or {})

    delay = max(0.0, backoff)
    result = HttpResult(error="no request attempted")
    for attempt in range(max(0, max_retries) + 1):
        if attempt:
            # Prefer the server's own guidance; fall back to our backoff curve.
            wait = result.retry_after()
            if wait is None:
                wait = delay
            time.sleep(min(wait, MAX_BACKOFF))
            delay = min(delay * 2, MAX_BACKOFF)
        result = _attempt(url, data, hdrs, timeout)
        if result.status not in RETRY_STATUSES:
            return result
    return result


class HttpStreamError(RuntimeError):
    """A streamed request failed.  Raised by :func:`stream_lines` only."""


def stream_lines(
    url: str,
    payload: dict,
    headers: dict[str, str] | None = None,
    timeout: float = 120.0,
) -> Iterator[str]:
    """POST *payload* as JSON and yield response lines as the server sends them.

    Unlike :func:`post_json` this **raises** :class:`HttpStreamError` on failure
    rather than returning a result object: a generator cannot hand back an error
    value before it has yielded anything, and half a stream is not a result.
    Providers catch it and turn it into a terminal
    :class:`~maestro.providers.base.StreamEvent`, so the caller still never sees
    an exception.

    Nothing is retried here.  A retried stream would have to re-emit tokens the
    consumer has already seen, so a failed stream is reported, not repeated.
    """
    data = json.dumps(payload).encode("utf-8")
    hdrs = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    hdrs.update(headers or {})
    req = urllib.request.Request(url, data=data, headers=hdrs, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                yield raw.decode("utf-8", "replace").rstrip("\r\n")
    except urllib.error.HTTPError as exc:
        detail = ""
        with contextlib.suppress(Exception):
            detail = exc.read().decode("utf-8", "replace")
        raise HttpStreamError(f"HTTP {exc.code}: {detail[:500]}") from exc
    except urllib.error.URLError as exc:
        raise HttpStreamError(f"connection error: {exc.reason}") from exc
    except TimeoutError as exc:
        raise HttpStreamError(f"request timed out after {timeout}s") from exc
    except HttpStreamError:
        raise
    except Exception as exc:
        raise HttpStreamError(f"transport error: {exc}") from exc
