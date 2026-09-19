import io
import socket

import pytest

from maestro.providers import _http


@pytest.mark.parametrize(
    "url",
    [
        "file:///tmp/a",
        "https://user:secret@example.com",
        "http://169.254.169.254/latest",
        "http://[::ffff:169.254.169.254]/latest",
    ],
)
def test_unsafe_endpoint_rejected(url, monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **kw: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", 80))],
    )
    with pytest.raises(ValueError):
        _http._validate_endpoint(url)


def test_local_model_endpoint_allowed(monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **kw: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 11434))],
    )
    monkeypatch.setenv("MAESTRO_ALLOW_PRIVATE_URLS", "true")
    _http._validate_endpoint("http://localhost:11434/api/chat")


def test_local_model_endpoint_blocked_by_default(monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **kw: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 11434))],
    )
    monkeypatch.delenv("MAESTRO_ALLOW_PRIVATE_URLS", raising=False)
    with pytest.raises(ValueError, match="private"):
        _http._validate_endpoint("http://localhost:11434/api/chat")


def test_redirect_never_replays_credentials():
    assert (
        _http._NoRedirect().redirect_request(None, None, 302, "", {}, "https://evil.invalid")
        is None
    )


def test_oversized_response_rejected():
    with pytest.raises(ValueError, match="size limit"):
        _http._bounded_body(io.BytesIO(b"12345"), 4)
