"""Anthropic backend: sampling-parameter gating, refusals, and transport retry.

Every test here runs against a fake transport — no network, no keys.
"""

from __future__ import annotations

import json
import logging

import pytest

from maestro.providers import _http, anthropic_provider
from maestro.providers._http import HttpResult, post_json
from maestro.providers.anthropic_provider import (
    NO_SAMPLING_PARAM_MODELS,
    AnthropicClient,
    accepts_sampling_params,
)

MESSAGES = [{"role": "user", "content": "hello"}]


def _forget_warning(model: str) -> None:
    """Reset the once-per-model warning latch so a test can observe it fire."""
    anthropic_provider._WARNED_SAMPLING_MODELS.discard(model)


def _ok_body(text: str = "hi", stop_reason: str = "end_turn", **extra) -> str:
    body = {
        "model": "test-model",
        "stop_reason": stop_reason,
        "content": [{"type": "text", "text": text}],
        "usage": {"input_tokens": 3, "output_tokens": 5},
    }
    body.update(extra)
    return json.dumps(body)


def _capture(monkeypatch, result: HttpResult) -> dict:
    """Swap the provider's transport for a recorder returning *result*."""
    seen: dict = {}

    def fake_post(url, payload, headers=None, timeout=120.0, **kwargs):
        seen["url"] = url
        seen["payload"] = payload
        seen["headers"] = headers
        return result

    monkeypatch.setattr("maestro.providers.anthropic_provider.post_json", fake_post)
    return seen


# --------------------------------------------------------------------------- #
# Sampling parameters
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model", NO_SAMPLING_PARAM_MODELS)
def test_current_models_reject_sampling_params(model):
    assert not accepts_sampling_params(model)
    # Dated snapshots and vendor-prefixed ids resolve to the same family.
    assert not accepts_sampling_params(f"{model}-20260401")
    assert not accepts_sampling_params(f"anthropic.{model}")


@pytest.mark.parametrize(
    "model", ["claude-sonnet-4-6", "claude-opus-4-6", "claude-haiku-4-5", "claude-3-opus"]
)
def test_older_models_accept_sampling_params(model):
    assert accepts_sampling_params(model)


def test_temperature_dropped_for_rejecting_model(monkeypatch, caplog):
    _forget_warning("claude-sonnet-5")
    seen = _capture(monkeypatch, HttpResult(status=200, body=_ok_body()))
    client = AnthropicClient(model="claude-sonnet-5", api_key="k")

    with caplog.at_level(logging.WARNING, logger="maestro.providers"):
        resp = client.complete(MESSAGES, temperature=0.7)

    assert resp.ok()
    assert "temperature" not in seen["payload"]
    assert "top_p" not in seen["payload"] and "top_k" not in seen["payload"]
    assert "rejects" in caplog.text and "claude-sonnet-5" in caplog.text


def test_temperature_warning_is_emitted_once_per_model(monkeypatch, caplog):
    _forget_warning("claude-opus-4-8")
    _capture(monkeypatch, HttpResult(status=200, body=_ok_body()))
    client = AnthropicClient(model="claude-opus-4-8", api_key="k")

    with caplog.at_level(logging.WARNING, logger="maestro.providers"):
        for _ in range(3):
            client.complete(MESSAGES, temperature=0.2)

    warnings = [r for r in caplog.records if "rejects" in r.getMessage()]
    assert len(warnings) == 1


def test_temperature_preserved_for_accepting_model(monkeypatch, caplog):
    seen = _capture(monkeypatch, HttpResult(status=200, body=_ok_body()))
    client = AnthropicClient(model="claude-sonnet-4-6", api_key="k")

    with caplog.at_level(logging.WARNING, logger="maestro.providers"):
        resp = client.complete(MESSAGES, temperature=0.3)

    assert resp.ok()
    assert seen["payload"]["temperature"] == 0.3
    assert "rejects" not in caplog.text


def test_no_temperature_key_when_none_requested(monkeypatch):
    seen = _capture(monkeypatch, HttpResult(status=200, body=_ok_body()))
    AnthropicClient(model="claude-sonnet-4-6", api_key="k").complete(MESSAGES)
    assert "temperature" not in seen["payload"]


# --------------------------------------------------------------------------- #
# Refusals
# --------------------------------------------------------------------------- #


def test_refusal_stop_reason_becomes_an_error(monkeypatch):
    body = json.dumps(
        {
            "model": "claude-sonnet-5",
            "stop_reason": "refusal",
            "stop_details": {"type": "refusal", "category": "cyber"},
            "content": [],
            "usage": {"input_tokens": 9, "output_tokens": 0},
        }
    )
    _capture(monkeypatch, HttpResult(status=200, body=body))

    resp = AnthropicClient(api_key="k").complete(MESSAGES)

    assert not resp.ok()
    assert "refused" in resp.error and "cyber" in resp.error
    assert resp.text == ""
    assert resp.usage["prompt_tokens"] == 9


def test_refusal_with_partial_content_still_errors(monkeypatch):
    _capture(
        monkeypatch,
        HttpResult(status=200, body=_ok_body("partial answer", stop_reason="refusal")),
    )
    resp = AnthropicClient(api_key="k").complete(MESSAGES)
    assert resp.error
    assert not resp.ok()


def test_normal_stop_reason_is_not_an_error(monkeypatch):
    _capture(monkeypatch, HttpResult(status=200, body=_ok_body("all good")))
    resp = AnthropicClient(api_key="k").complete(MESSAGES)
    assert resp.ok()
    assert resp.text == "all good"
    assert not resp.error


# --------------------------------------------------------------------------- #
# Transport retry (shared by every provider)
# --------------------------------------------------------------------------- #


def _stub_attempts(monkeypatch, results: list[HttpResult]) -> tuple[list, list]:
    """Feed *results* to post_json one per attempt; record calls and sleeps."""
    calls: list[str] = []
    slept: list[float] = []

    def fake_attempt(url, data, headers, timeout):
        calls.append(url)
        return results[min(len(calls) - 1, len(results) - 1)]

    monkeypatch.setattr(_http, "_attempt", fake_attempt)
    monkeypatch.setattr(_http.time, "sleep", slept.append)
    return calls, slept


def test_retries_429_and_honours_retry_after(monkeypatch):
    calls, slept = _stub_attempts(
        monkeypatch,
        [
            HttpResult(status=429, error="HTTP 429: slow down", headers={"Retry-After": "7"}),
            HttpResult(status=200, body="{}"),
        ],
    )
    res = post_json("http://x/v1/messages", {"a": 1})
    assert res.status == 200
    assert len(calls) == 2
    assert slept == [7.0]


def test_retries_529_with_exponential_backoff_then_gives_up(monkeypatch):
    calls, slept = _stub_attempts(
        monkeypatch, [HttpResult(status=529, error="HTTP 529: overloaded")]
    )
    res = post_json("http://x/v1/messages", {"a": 1})
    assert res.status == 529
    assert res.error  # the last failure is handed back to the caller
    assert len(calls) == 3  # 1 attempt + DEFAULT_MAX_RETRIES
    assert slept == [0.5, 1.0]  # doubling, no retry-after header present


def test_non_retryable_status_returns_immediately(monkeypatch):
    calls, slept = _stub_attempts(
        monkeypatch, [HttpResult(status=400, error="HTTP 400: bad request")]
    )
    res = post_json("http://x/v1/messages", {"a": 1})
    assert res.status == 400
    assert len(calls) == 1
    assert slept == []


def test_max_retries_zero_disables_retrying(monkeypatch):
    calls, _ = _stub_attempts(monkeypatch, [HttpResult(status=429, error="HTTP 429")])
    post_json("http://x", {}, max_retries=0)
    assert len(calls) == 1


def test_retry_after_parsing():
    assert HttpResult(headers={"retry-after": "3"}).retry_after() == 3.0
    assert HttpResult(headers={"Retry-After": "  2.5 "}).retry_after() == 2.5
    assert HttpResult(headers={"Retry-After": "not-a-date"}).retry_after() is None
    assert HttpResult().retry_after() is None
    # HTTP-date form resolves to a non-negative number of seconds.
    http_date = HttpResult(headers={"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"})
    assert http_date.retry_after() == 0.0


def test_retry_waits_are_capped(monkeypatch):
    _, slept = _stub_attempts(
        monkeypatch,
        [HttpResult(status=429, error="HTTP 429", headers={"Retry-After": "9999"})],
    )
    post_json("http://x", {}, max_retries=1)
    assert slept == [_http.MAX_BACKOFF]
