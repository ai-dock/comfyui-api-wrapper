"""Tests for the OOM / CUDA-fault classifiers in generation_worker.

These run against the module-level pure functions; no asyncio, no
ComfyUI, no fixtures.
"""
import pytest

from workers.generation_worker import (
    _detect_oom_error,
    _detect_cuda_unrecoverable_reason,
    mark_gpu_unrecoverable,
    get_gpu_state,
    _GPU_STATE,
)


@pytest.fixture(autouse=True)
def _reset_gpu_state():
    """The GPU latch is module-global; reset between tests so they
    don't leak state."""
    _GPU_STATE.clear()
    yield
    _GPU_STATE.clear()


_BACKEND = "http://127.0.0.1:8188"


@pytest.mark.parametrize("text", [
    "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 24.00 GiB.",
    "RuntimeError: CUDA out of memory",
    "Out of memory",
    "ran out of memory",
    "memory allocation failed",
    "insufficient memory available",
    "OutOfMemoryError",
])
def test_detect_oom_positive(text):
    assert _detect_oom_error(text) is True


@pytest.mark.parametrize("text", [
    "",
    None,
    "RuntimeError: invalid argument",
    "CUDA error: device-side assert triggered",  # CUDA fault, not OOM
    "Connection refused",
])
def test_detect_oom_negative(text):
    assert _detect_oom_error(text) is False


@pytest.mark.parametrize("text,expected_substr", [
    ("CUDA error: an illegal memory access was encountered", "illegal memory access"),
    ("RuntimeError: CUDA error: misaligned address", "cuda error: misaligned address"),
    ("device-side assert triggered", "device-side assert triggered"),
    ("Triton Error [CUDA]: unspecified launch failure", "unspecified launch failure"),
])
def test_cuda_unrecoverable_positive(text, expected_substr):
    reason = _detect_cuda_unrecoverable_reason(text)
    assert reason and expected_substr in reason.lower()


def test_cuda_unrecoverable_does_not_match_oom():
    """OOM is recoverable, NOT a CUDA-unrecoverable. The classifier
    must explicitly exclude OOM patterns even though they technically
    contain 'cuda' / 'memory' substrings."""
    assert _detect_cuda_unrecoverable_reason("CUDA out of memory") == ""
    assert _detect_cuda_unrecoverable_reason("torch.cuda.OutOfMemoryError") == ""


def test_cuda_unrecoverable_negative_returns_empty_string():
    assert _detect_cuda_unrecoverable_reason("") == ""
    assert _detect_cuda_unrecoverable_reason("Connection refused") == ""
    assert _detect_cuda_unrecoverable_reason(None) == ""


def test_gpu_state_round_trip_per_backend():
    assert get_gpu_state(_BACKEND) == {"unrecoverable": False, "reason": ""}
    mark_gpu_unrecoverable(_BACKEND, "illegal memory access")
    assert get_gpu_state(_BACKEND) == {"unrecoverable": True, "reason": "illegal memory access"}


def test_mark_gpu_unrecoverable_is_idempotent_first_wins():
    """Once latched, the originating reason is preserved — a later
    fault must not overwrite the first one (which is the most
    informative for debugging)."""
    mark_gpu_unrecoverable(_BACKEND, "illegal memory access")
    mark_gpu_unrecoverable(_BACKEND, "device-side assert triggered")
    assert get_gpu_state(_BACKEND) == {
        "unrecoverable": True,
        "reason": "illegal memory access",
    }


def test_mark_gpu_unrecoverable_handles_empty_reason():
    """Defensive: an empty reason string still latches the flag,
    but records a sentinel so downstream consumers see something
    rather than blank."""
    mark_gpu_unrecoverable(_BACKEND, "")
    state = get_gpu_state(_BACKEND)
    assert state["unrecoverable"] is True
    assert state["reason"] == "unspecified"


def test_gpu_state_aggregate_no_backends():
    """No backends latched → aggregate is healthy."""
    s = get_gpu_state()
    assert s == {"unrecoverable": False, "reason": "", "by_backend": {}}


def test_gpu_state_aggregate_one_backend_latched():
    """One out of three latched → aggregate reports unrecoverable
    and surfaces the bad backend's reason."""
    mark_gpu_unrecoverable("http://127.0.0.1:8188", "illegal memory access")
    # Two healthy backends — register them by querying state.
    get_gpu_state("http://127.0.0.1:8189")
    get_gpu_state("http://127.0.0.1:8190")
    s = get_gpu_state()
    assert s["unrecoverable"] is True
    assert s["reason"] == "illegal memory access"
    assert s["by_backend"]["http://127.0.0.1:8188"]["unrecoverable"] is True


def test_gpu_state_isolates_per_backend():
    """A latch on backend A must not affect backend B."""
    mark_gpu_unrecoverable("http://a:8188", "illegal memory access")
    assert get_gpu_state("http://a:8188")["unrecoverable"] is True
    assert get_gpu_state("http://b:8188")["unrecoverable"] is False
