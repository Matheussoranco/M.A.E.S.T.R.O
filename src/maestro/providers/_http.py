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
import ipaddress
import json
import os
import socket
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass, field
from urllib.parse import urlsplit

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
MAX_RESPONSE_BYTES = 16 * 1024 * 1024
MAX_LINE_BYTES = 1024 * 1024


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward API credentials or prompt bodies to another endpoint.
        return None


def _allow_private_urls() -> bool:
    """True only when ``MAESTRO_ALLOW_PRIVATE_URLS=true`` (Ollama/local dev).

    Cloud providers must never exfiltrate API keys to private/loopback
    endpoints; local Ollama (``http://localhost:11434``) opts in explicitly.
    """
    return os.getenv("MAESTRO_ALLOW_PRIVATE_URLS", "").strip().lower() == "true"


def validate_url(url: str) -> None:
    """Block SSRF endpoints before any API key is sent.

    Allows only ``http``/``https`` without userinfo/fragment; blocks
    private/loopback/link-local/reserved/multicast/unspecified (plus cloud
    metadata) unless :func:`_allow_private_urls` opts in — in which case
    private + loopback are allowed (Ollama) but link-local/multicast/
    unspecified/reserved/metadata stay blocked.
    """
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError("provider endpoint must be an HTTP(S) URL")
    if parts.username is not None or parts.password is not None or parts.fragment:
        raise ValueError("provider endpoint must not contain userinfo or a fragment")
    if parts.hostname.lower().rstrip(".") == "metadata.google.internal":
        raise ValueError("metadata endpoints are not model providers")
    allow_private = _allow_private_urls()
    try:
        infos = socket.getaddrinfo(
            parts.hostname,
            parts.port or (443 if parts.scheme == "https" else 80),
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as exc:
        raise ValueError(f"provider endpoint does not resolve: {exc}") from exc
    for info in infos:
        address = ipaddress.ip_address(info[4][0].split("%", 1)[0])
        address = getattr(address, "ipv4_mapped", None) or address
        # Always forbidden: exfiltration/redirect targets.
        if (
            address.is_multicast
            or address.is_unspecified
            or address.is_reserved
            or str(address) == "100.100.100.200"
        ):
            raise ValueError("provider endpoint resolves to a prohibited address")
        if address.is_link_local:
            raise ValueError("provider endpoint resolves to a prohibited address")
        if not allow_private and (address.is_private or address.is_loopback):
            raise ValueError(
                "provider endpoint resolves to a private address; "
                "set MAESTRO_ALLOW_PRIVATE_URLS=true to allow local endpoints (Ollama)"
            )


def _validate_endpoint(url: str) -> None:
    """Validate configured endpoints; local/LAN model servers remain supported.

    Specs are trusted executable configuration, not untrusted user input. This
    blocks metadata/link-local endpoints and credential-bearing redirects; it
    does not turn a configurable provider into a general-purpose URL sandbox.
    """
    validate_url(url)


def _open(req, timeout):
    _validate_endpoint(req.full_url)
    return urllib.request.build_opener(_NoRedirect()).open(req, timeout=timeout)


def _bounded_body(resp, limit=MAX_RESPONSE_BYTES):
    body = resp.read(limit + 1)
    if len(body) > limit:
        raise ValueError("provider response exceeds size limit")
    return body.decode("utf-8", "replace")


@dataclass
class HttpResult:
    status: int = 0
    body: str = ""
    error: str = ""
    headers: dict[str, str] = field(default_factory=dict)

    def json(self) -> dict:
        """Decode an object response, preserving malformed-body diagnostics."""
        if not self.body:
            return {"_parse_error": "empty response body", "_raw": ""}
        try:
            value = json.loads(self.body)
            if not isinstance(value, dict):
                return {
                    "_parse_error": f"expected a JSON object, got {type(value).__name__}",
                    "_raw": self.body[:2000],
                }
            return value
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
    try:
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with _open(req, timeout) as resp:
            body = _bounded_body(resp)
            return HttpResult(
                status=getattr(resp, "status", 200),
                body=body,
                headers=_collect_headers(resp),
            )
    except urllib.error.HTTPError as exc:
        detail = ""
        with contextlib.suppress(Exception):
            detail = _bounded_body(exc, 4096)
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
    try:
        req = urllib.request.Request(url, data=data, headers=hdrs, method="POST")
        with _open(req, timeout) as resp:
            total = 0
            while True:
                raw = resp.readline(MAX_LINE_BYTES + 1)
                if not raw:
                    break
                total += len(raw)
                if len(raw) > MAX_LINE_BYTES or total > MAX_RESPONSE_BYTES:
                    raise HttpStreamError("provider stream exceeds size limit")
                yield raw.decode("utf-8", "replace").rstrip("\r\n")
    except urllib.error.HTTPError as exc:
        detail = ""
        with contextlib.suppress(Exception):
            detail = _bounded_body(exc, 4096)
        raise HttpStreamError(f"HTTP {exc.code}: {detail[:500]}") from exc
    except urllib.error.URLError as exc:
        raise HttpStreamError(f"connection error: {exc.reason}") from exc
    except TimeoutError as exc:
        raise HttpStreamError(f"request timed out after {timeout}s") from exc
    except HttpStreamError:
        raise
    except Exception as exc:
        raise HttpStreamError(f"transport error: {exc}") from exc
