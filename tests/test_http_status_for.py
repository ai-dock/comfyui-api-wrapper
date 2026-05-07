"""Tests for the terminal-status -> HTTP-code mapping in main.py.

Background: `/generate/sync` and `/result/...` historically returned
HTTP 200 regardless of the body's `status` field. The Vast SDK's
benchmark counts a successful response by HTTP code, not body, so a
fast-failing 0.5s "Cannot connect to host" reply was being scored as
a 100-workload success — surfacing as a fake very-fast worker
(perf=200). The fix maps terminal status to a meaningful code.
"""
import pytest

from main import _http_status_for
from responses.result import Result


@pytest.mark.parametrize("message", [
    # post_workflow's after-N-attempts message
    "Failed to post workflow after 5 attempts: ClientConnectorError",
    # raw aiohttp connect-failed text
    "Cannot connect to host localhost:18188 ssl:default",
    "Network error getting result: ConnectionResetError",
    "Connection refused",
    "WebSocket timeout for job abc-123",
    "WebSocket failure mid-stream",
    "Operation timed out after 30s",
    # case insensitivity — match should be lower-case substring
    "TIMED OUT WAITING",
])
def test_failed_with_upstream_class_returns_502(message):
    r = Result(id="t", status="failed", message=message)
    assert _http_status_for(r) == 502


@pytest.mark.parametrize("message", [
    "ComfyUI node errors: KSampler.sampler_name not in list",
    "ComfyUI error: bad input",
    "torch.cuda.OutOfMemoryError: CUDA out of memory",
    "Generation failed: invalid workflow",
    "",  # no message at all — still a real failure
])
def test_failed_without_upstream_class_returns_500(message):
    r = Result(id="t", status="failed", message=message)
    assert _http_status_for(r) == 500


def test_completed_returns_200():
    r = Result(id="t", status="completed", message="Processing complete.")
    assert _http_status_for(r) == 200


def test_cancelled_returns_499():
    r = Result(id="t", status="cancelled", message="Request cancelled by client")
    assert _http_status_for(r) == 499


@pytest.mark.parametrize("status", ["pending", "generating", "processing", "generated"])
def test_non_terminal_falls_through_to_500(status):
    """Defensive: callers should only pass terminal results, but if
    something slips through, return 500 so the caller doesn't read it
    as a success. (`/result/{id}` and `/generate/sync` already gate on
    a terminal-status check before invoking this helper.)"""
    r = Result(id="t", status=status, message="")
    assert _http_status_for(r) == 500


def test_missing_status_attribute_returns_500():
    """A bare object without `.status` should still be treated as a
    failure, not silently mapped to 200."""
    class Bare:
        pass
    assert _http_status_for(Bare()) == 500
