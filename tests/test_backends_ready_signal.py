"""Tests for the BACKENDS_READY / BACKENDS_READY_TIMEOUT log signals.

Background: pyworkers tail the api-wrapper log to drive their warm-up
state machine. The earlier ``Uvicorn running on …`` line only meant
"our HTTP listener is bound" — it didn't say anything about whether
ComfyUI behind us was actually up. Pods where ComfyUI hadn't loaded
(or had crashed between provisioning and uvicorn binding) ran benchmark
against an unreachable backend and the SDK's HTTP-status-only success
check counted the resulting fast 502s as a fast worker.

The wrapper now emits ``BACKENDS_READY`` only after every configured
backend passes the same HTTP+WS probe ``/health`` uses, and
``BACKENDS_READY_TIMEOUT`` if they don't come up within the deadline.
"""
import asyncio
import logging

import pytest

import main


@pytest.fixture
def fast_announcer(monkeypatch):
    """Tighten the announcer's loop so tests don't wait minutes."""
    monkeypatch.setattr(main, "_BACKENDS_READY_TIMEOUT_S", 1)
    monkeypatch.setattr(main, "_BACKENDS_READY_POLL_INTERVAL_S", 0.05)


@pytest.fixture
def two_backends(monkeypatch):
    monkeypatch.setattr(main, "COMFYUI_BACKENDS", [
        "http://backend-a:18188",
        "http://backend-b:18188",
    ])


async def _probe_returning(http_ok: bool, websocket_ok: bool):
    async def _impl(urls):
        return {
            "base": urls["base"],
            "http_ok": http_ok,
            "websocket_ok": websocket_ok,
        }
    return _impl


@pytest.mark.asyncio
async def test_emits_backends_ready_when_all_probes_pass(
    fast_announcer, two_backends, monkeypatch, caplog
):
    monkeypatch.setattr(main, "_probe_backend", await _probe_returning(True, True))

    with caplog.at_level(logging.INFO, logger=main.logger.name):
        await main._announce_backends_ready()

    msgs = [r.getMessage() for r in caplog.records]
    assert any("BACKENDS_READY:" in m for m in msgs), msgs
    assert not any("BACKENDS_READY_TIMEOUT" in m for m in msgs), msgs


@pytest.mark.asyncio
async def test_emits_timeout_when_backends_never_become_ready(
    fast_announcer, two_backends, monkeypatch, caplog
):
    monkeypatch.setattr(main, "_probe_backend", await _probe_returning(False, False))

    with caplog.at_level(logging.ERROR, logger=main.logger.name):
        await main._announce_backends_ready()

    msgs = [r.getMessage() for r in caplog.records]
    assert any("BACKENDS_READY_TIMEOUT" in m for m in msgs), msgs
    assert not any(m.startswith("BACKENDS_READY:") for m in msgs), msgs


@pytest.mark.asyncio
async def test_waits_for_late_backend_then_emits_ready(
    fast_announcer, two_backends, monkeypatch, caplog
):
    """One backend is slow to come up — announcer must keep polling
    until it does, then emit BACKENDS_READY exactly once."""
    calls = {"n": 0}

    async def _probe(urls):
        calls["n"] += 1
        # First two probe rounds (4 calls for 2 backends): backend-b
        # is down. After that, both up.
        backend_b_down = calls["n"] <= 4 and "backend-b" in urls["base"]
        return {
            "base":         urls["base"],
            "http_ok":      not backend_b_down,
            "websocket_ok": not backend_b_down,
        }

    monkeypatch.setattr(main, "_probe_backend", _probe)

    with caplog.at_level(logging.INFO, logger=main.logger.name):
        await main._announce_backends_ready()

    msgs = [r.getMessage() for r in caplog.records]
    ready = [m for m in msgs if "BACKENDS_READY:" in m and "TIMEOUT" not in m]
    assert len(ready) == 1, f"expected exactly one ready announcement, got {ready}"


def test_unrecoverable_log_carries_grep_token():
    """The pyworker matches MODEL_ERROR_LOG_MSGS as a substring; the
    fatal log line must contain BACKEND_UNRECOVERABLE so a fault
    reaches the autoscaler instead of stalling silently."""
    from workers.generation_worker import mark_gpu_unrecoverable, _GPU_STATE
    _GPU_STATE.clear()
    import logging as _l
    rec = []

    class _H(_l.Handler):
        def emit(self, r):
            rec.append(r.getMessage())

    h = _H()
    _l.getLogger("workers.generation_worker").addHandler(h)
    try:
        mark_gpu_unrecoverable("http://b:18188", "device-side assert")
    finally:
        _l.getLogger("workers.generation_worker").removeHandler(h)
        _GPU_STATE.clear()

    assert any("BACKEND_UNRECOVERABLE" in m for m in rec), rec
