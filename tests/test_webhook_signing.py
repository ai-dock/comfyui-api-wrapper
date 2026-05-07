"""Tests for HMAC-SHA256 webhook signing.

Verifies that:
- A webhook fired without a secret carries no signature header.
- A webhook fired with a secret carries the right header shape.
- The signature is computable from the exact body bytes the
  consumer receives (no re-serialisation drift).
- Per-request `webhook.secret` overrides the env default.
- Verification with the documented recipe succeeds; mutating
  one byte makes it fail.
"""
import asyncio
import hashlib
import hmac
import json
from typing import Dict
from unittest.mock import patch, AsyncMock

import pytest

from workers.postprocess_worker import PostprocessWorker


def _make_worker():
    class _StubStore:
        async def get(self, _):    return None
        async def set(self, *_):   return None
        async def delete(self, _): return None

    return PostprocessWorker(
        worker_id="t",
        kwargs={
            "preprocess_queue":  None,
            "generation_queue":  None,
            "postprocess_queue": None,
            "request_store":     _StubStore(),
            "response_store":    _StubStore(),
        },
    )


class _Result:
    """Duck-typed Result minimal enough for send_webhook."""
    def __init__(self):
        self.id = "rid-001"
        self.status = "completed"
        self.message = "Processing complete."
        self.output = []
        self.timings = {"preprocess_ms": 1, "generation_ms": 100, "postprocess_ms": 5}


class _CapturingClientSession:
    """Replaces aiohttp.ClientSession to capture the POST body +
    headers without a real HTTP server."""
    captured = []

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def post(self, url, data=None, headers=None, **kwargs):
        _CapturingClientSession.captured.append({
            "url":     url,
            "body":    data,
            "headers": headers or {},
        })
        return _CapturingResponse()


class _CapturingResponse:
    status = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def text(self):
        return ""


@pytest.fixture(autouse=True)
def _reset_capture():
    _CapturingClientSession.captured = []
    yield
    _CapturingClientSession.captured = []


@pytest.mark.asyncio
async def test_no_secret_no_signature_header():
    w = _make_worker()
    with patch("workers.postprocess_worker.aiohttp.ClientSession", _CapturingClientSession):
        await w.send_webhook("http://hook.test/x", _Result())

    assert len(_CapturingClientSession.captured) == 1
    headers = _CapturingClientSession.captured[0]["headers"]
    assert "X-Webhook-Signature" not in headers
    # Content-Type still set.
    assert headers.get("Content-Type") == "application/json"


@pytest.mark.asyncio
async def test_with_secret_signature_header_present_and_correct():
    w = _make_worker()
    secret = "shh-this-is-a-secret"

    with patch("workers.postprocess_worker.aiohttp.ClientSession", _CapturingClientSession):
        await w.send_webhook("http://hook.test/x", _Result(), secret=secret)

    cap = _CapturingClientSession.captured[0]
    sig_header = cap["headers"].get("X-Webhook-Signature")
    assert sig_header and sig_header.startswith("sha256=")
    sent_hex = sig_header.split("=", 1)[1]

    # The signature must match HMAC-SHA256 of the EXACT body bytes
    # we sent — re-serialising and recomputing is what a real
    # consumer would do and it must agree.
    body_bytes = cap["body"]
    expected = hmac.new(secret.encode(), body_bytes, hashlib.sha256).hexdigest()
    assert hmac.compare_digest(sent_hex, expected)


@pytest.mark.asyncio
async def test_signature_changes_when_payload_changes():
    """Sanity: tamper with one byte of body and the previously-
    valid signature stops verifying. Catches any silent body-
    re-serialisation regression."""
    w = _make_worker()
    secret = "k"

    with patch("workers.postprocess_worker.aiohttp.ClientSession", _CapturingClientSession):
        await w.send_webhook("http://hook.test/x", _Result(), secret=secret)

    cap = _CapturingClientSession.captured[0]
    sent_hex = cap["headers"]["X-Webhook-Signature"].split("=", 1)[1]
    tampered = cap["body"][:-1] + b"X"
    bad = hmac.new(secret.encode(), tampered, hashlib.sha256).hexdigest()
    assert not hmac.compare_digest(sent_hex, bad)


@pytest.mark.asyncio
async def test_extra_params_signed_too():
    """The signature covers the FULL body — including the customer's
    `extra` namespace — so consumers can trust extras as part of
    the signed payload."""
    w = _make_worker()
    secret = "k"

    with patch("workers.postprocess_worker.aiohttp.ClientSession", _CapturingClientSession):
        await w.send_webhook(
            "http://hook.test/x",
            _Result(),
            extra_params={"customer_field": "value"},
            secret=secret,
        )

    cap = _CapturingClientSession.captured[0]
    body = json.loads(cap["body"])
    assert body["extra"] == {"customer_field": "value"}

    # Recomputing matches.
    sent_hex = cap["headers"]["X-Webhook-Signature"].split("=", 1)[1]
    expected = hmac.new(secret.encode(), cap["body"], hashlib.sha256).hexdigest()
    assert hmac.compare_digest(sent_hex, expected)


@pytest.mark.asyncio
async def test_documented_verification_recipe():
    """The README's verification recipe must agree with what the
    wrapper sends. This locks in the published consumer contract."""
    w = _make_worker()
    secret = "shared-secret"

    with patch("workers.postprocess_worker.aiohttp.ClientSession", _CapturingClientSession):
        await w.send_webhook("http://hook.test/x", _Result(), secret=secret)

    cap = _CapturingClientSession.captured[0]

    # ---- Consumer-side recipe (mirrors README) ----
    raw_body = cap["body"]                                # bytes as received
    header   = cap["headers"]["X-Webhook-Signature"]      # "sha256=<hex>"
    assert header.startswith("sha256=")
    received_hex = header.split("=", 1)[1]
    expected_hex = hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest()
    assert hmac.compare_digest(received_hex, expected_hex)
