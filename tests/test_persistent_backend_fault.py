"""Tests for the persistent-backend-fault classifier and counter.

Background: a CUDA fault is caught by ``_detect_cuda_unrecoverable_reason``
and immediately latches the backend. But ComfyUI also has plenty of
non-CUDA ways to die (process killed by OOM-killer, deadlock, hang),
which surface as ``Cannot connect to host`` / ``Failed to post
workflow after N attempts``. Those used to leave the worker
"healthy" from the SDK's view: requests fail with 502 but no fatal
log token fires, so the autoscaler never replaces the pod.

The fix counts connectivity-class failures per backend; once we've
seen BACKEND_FAILURE_THRESHOLD consecutive ones, the backend is
latched as unrecoverable (which emits ``BACKEND_UNRECOVERABLE`` for
the pyworker to pick up). A single forward-progress success resets
the counter so a brief restart blip doesn't accumulate.
"""
import logging

import pytest

from workers.generation_worker import (
    BACKEND_FAILURE_THRESHOLD,
    _BACKEND_FAILURE_COUNT,
    _GPU_STATE,
    _is_backend_connectivity_failure,
    _record_backend_failure,
    _record_backend_success,
    get_gpu_state,
)


@pytest.fixture(autouse=True)
def _reset_module_state():
    _BACKEND_FAILURE_COUNT.clear()
    _GPU_STATE.clear()
    yield
    _BACKEND_FAILURE_COUNT.clear()
    _GPU_STATE.clear()


_BACKEND = "http://b:18188"


@pytest.mark.parametrize("text", [
    "Cannot connect to host localhost:18188 ssl:default",
    "Failed to post workflow after 5 attempts: ClientConnectorError",
    "Network error getting result: ConnectionResetError",
    "Connection refused",
    "WebSocket timeout for job abc",
    "Operation timed out after 30s",
    # case insensitivity
    "FAILED TO POST WORKFLOW AFTER 5 attempts",
])
def test_classifies_connectivity_failure(text):
    assert _is_backend_connectivity_failure(text) is True


@pytest.mark.parametrize("text", [
    "ComfyUI node errors: KSampler.sampler_name not in list",
    "ComfyUI error: bad input",
    "torch.cuda.OutOfMemoryError",
    "device-side assert triggered",
    "",
    None,
])
def test_skips_non_connectivity_text(text):
    assert _is_backend_connectivity_failure(text) is False


def test_below_threshold_does_not_latch():
    """N-1 consecutive failures is not yet fatal."""
    for _ in range(BACKEND_FAILURE_THRESHOLD - 1):
        _record_backend_failure(_BACKEND, "Cannot connect")
    assert get_gpu_state(_BACKEND)["unrecoverable"] is False


def test_threshold_latches_backend_unrecoverable():
    """Hitting the threshold latches the backend and surfaces a
    BACKEND_UNRECOVERABLE log line."""
    for _ in range(BACKEND_FAILURE_THRESHOLD):
        _record_backend_failure(_BACKEND, "Cannot connect")
    state = get_gpu_state(_BACKEND)
    assert state["unrecoverable"] is True
    assert "consecutive" in state["reason"].lower()


def test_success_resets_counter():
    """A successful workflow post on the same backend means it's alive
    again — clear any prior connectivity-failure accumulation."""
    for _ in range(BACKEND_FAILURE_THRESHOLD - 1):
        _record_backend_failure(_BACKEND, "Cannot connect")
    _record_backend_success(_BACKEND)
    assert _BACKEND_FAILURE_COUNT.get(_BACKEND, 0) == 0
    # And one more failure now starts a fresh count, doesn't latch.
    _record_backend_failure(_BACKEND, "Cannot connect")
    assert get_gpu_state(_BACKEND)["unrecoverable"] is False


def test_threshold_latch_is_per_backend():
    """A failed backend doesn't drag healthy ones down at the latch
    level — but pod-level aggregation is still bad (any() in
    get_gpu_state()), which is the desired worker-replacement signal."""
    other = "http://other:18188"
    for _ in range(BACKEND_FAILURE_THRESHOLD):
        _record_backend_failure(_BACKEND, "Cannot connect")
    assert get_gpu_state(_BACKEND)["unrecoverable"] is True
    assert get_gpu_state(other)["unrecoverable"] is False
    # Pod-level aggregation reports unhealthy because any() is bad.
    assert get_gpu_state()["unrecoverable"] is True


def test_latch_emits_grep_token(caplog):
    """The latch path goes through mark_gpu_unrecoverable, which the
    earlier change tagged with BACKEND_UNRECOVERABLE so pyworkers'
    MODEL_ERROR_LOG_MSGS can match it."""
    with caplog.at_level(logging.ERROR, logger="workers.generation_worker"):
        for _ in range(BACKEND_FAILURE_THRESHOLD):
            _record_backend_failure(_BACKEND, "Cannot connect")
    msgs = [r.getMessage() for r in caplog.records]
    assert any("BACKEND_UNRECOVERABLE" in m for m in msgs), msgs
