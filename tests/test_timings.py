"""Tests for the ``Result.timings`` end-to-end shape.

The original goal is the queue→complete delta (``total_ms``) so SLO
tracking works against real elapsed time. The per-stage durations
(``preprocess_ms`` / ``generation_ms`` / ``postprocess_ms``) come
along because every worker already brackets its own work block —
they're useful when chasing where the time is going.

The end-to-end stamping cuts across three workers and an endpoint,
so most of the assertions here are unit-level: ``main._new_result``
stamps ``queued_at_ms``, and ``postprocess_worker._close_out_timings``
closes out ``postprocess_ms`` / ``completed_at_ms`` / ``total_ms`` on
both success and failure paths.
"""
import time

import pytest

import main
from responses.result import Result
from workers.postprocess_worker import _close_out_timings


def test_new_result_stamps_queued_at_ms():
    before = int(time.time() * 1000)
    r = main._new_result("req-x")
    after = int(time.time() * 1000)
    assert "queued_at_ms" in r.timings
    assert before <= r.timings["queued_at_ms"] <= after


def test_new_result_carries_only_queue_time_initially():
    """preprocess / generation / postprocess fields land later, as
    each worker fills its bracket. The initial Result must not lie
    about durations it hasn't measured."""
    r = main._new_result("req-x")
    assert set(r.timings.keys()) == {"queued_at_ms"}


def test_close_out_timings_populates_terminal_fields():
    """postprocess closing out a successful run produces the full
    timings dict the operator expects."""
    r = main._new_result("req-x")
    # Pretend each prior stage already measured itself.
    r.timings["preprocess_ms"] = 10
    r.timings["generation_ms"] = 1500
    # Now postprocess runs for ~50ms.
    pp_t0 = time.time() - 0.05
    _close_out_timings(r, pp_t0)

    assert "postprocess_ms" in r.timings
    assert 40 <= r.timings["postprocess_ms"] <= 200, r.timings
    assert "completed_at_ms" in r.timings
    assert r.timings["completed_at_ms"] >= r.timings["queued_at_ms"]
    assert "total_ms" in r.timings
    assert r.timings["total_ms"] == (
        r.timings["completed_at_ms"] - r.timings["queued_at_ms"]
    )


def test_close_out_timings_skips_total_when_queued_missing():
    """Defensive: if ``queued_at_ms`` was never stamped (e.g. a
    pre-timings Result deserialised from cache during a rolling
    upgrade), we should record ``postprocess_ms`` and
    ``completed_at_ms`` but not invent a bogus total."""
    r = Result(id="r", timings={})  # no queued_at_ms
    _close_out_timings(r, time.time() - 0.01)
    assert "postprocess_ms" in r.timings
    assert "completed_at_ms" in r.timings
    assert "total_ms" not in r.timings


def test_close_out_timings_runs_on_failure_path_too():
    """Failed jobs still get total_ms — knowing how long a doomed
    request was alive is just as useful as for a successful one."""
    r = main._new_result("req-fail")
    r.status = "failed"
    r.timings["queued_at_ms"] = int(time.time() * 1000) - 200  # 200ms ago
    pp_t0 = time.time() - 0.01
    _close_out_timings(r, pp_t0)
    assert r.timings["total_ms"] >= 150


def test_timings_are_integers():
    """Round to ms so downstream consumers don't have to deal with
    floating-point JSON serialisation noise."""
    r = main._new_result("req-x")
    _close_out_timings(r, time.time() - 0.01)
    for k, v in r.timings.items():
        assert isinstance(v, int), f"{k}={v} ({type(v).__name__})"
