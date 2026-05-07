"""Regression test: a postprocess job whose ``request_store`` entry has
disappeared between queue insertion and processing must not crash the
worker.

Failure mode this guards against (observed live, May 2026):

    ERROR:workers.postprocess_worker:PostprocessWorker N failed job X:
        Request X not found in store
    ERROR:asyncio:Task exception was never retrieved
        File "workers/postprocess_worker.py", line 124, in work
            webhook_config = await self.get_webhook_config(request.input)
                                                            ^^^^^^^^^^^^^
        AttributeError: 'NoneType' object has no attribute 'input'

The `try` block raises a clean ``"not found in store"`` exception; the
`except` logs it; then the `finally` dereferenced ``request.input``
unconditionally and the AttributeError propagated out of ``work()``'s
``while True`` loop. Each such crash removed one worker from the
``asyncio.gather`` pool with no auto-restart — after enough crashes
the postprocess queue had zero consumers and every subsequent request
hung forever waiting for a "completed" status.

The fix gates the webhook block on ``request is not None`` and wraps
it in its own try/except so any unexpected error stays inside the
loop iteration.
"""
import asyncio

import pytest

from workers.postprocess_worker import PostprocessWorker


class _FakeRequestStore:
    """request_store.get returns None — simulates a deleted entry."""
    async def get(self, _):    return None
    async def set(self, *_):   return None
    async def delete(self, _): return None


class _FakeResponseStore:
    """response_store also empty — both ``request`` and ``result`` are None
    when the worker enters its except/finally."""
    async def get(self, _):    return None
    async def set(self, *_):   return None
    async def delete(self, _): return None


def _make_worker(tmp_path, queue):
    w = PostprocessWorker(
        worker_id="t",
        kwargs={
            "preprocess_queue":  None,
            "generation_queue":  None,
            "postprocess_queue": queue,
            "request_store":     _FakeRequestStore(),
            "response_store":    _FakeResponseStore(),
        },
    )
    w.output_dir = tmp_path
    return w


@pytest.mark.asyncio
async def test_worker_survives_missing_request_store_entry(tmp_path):
    """Drive ``work()`` through one job whose stores are empty, then a
    sentinel ``None`` to terminate the loop. The worker must complete
    the loop without the AttributeError that previously escaped from
    the ``finally`` block."""
    queue = asyncio.Queue()
    await queue.put("missing-job")
    await queue.put(None)  # shutdown sentinel

    worker = _make_worker(tmp_path, queue)

    # If the bug regresses, this raises AttributeError instead of
    # returning normally.
    await asyncio.wait_for(worker.work(), timeout=2.0)

    assert queue.empty(), "queue should be drained after orderly shutdown"


@pytest.mark.asyncio
async def test_worker_keeps_processing_after_missing_request(tmp_path):
    """The cascade in production: one missing-store job followed by a
    real one. The real one must still be picked up — i.e. the worker
    keeps consuming the queue rather than dying after the first
    AttributeError."""
    processed = []

    class _RealRequestStore:
        """First lookup returns None (simulates the cleanup race),
        second returns a stub object so the second job runs normally."""
        def __init__(self):
            self._calls = 0

        async def get(self, _):
            self._calls += 1
            if self._calls == 1:
                return None
            return _StubRequest()

        async def set(self, *_):    return None
        async def delete(self, _):  return None

    class _StubRequest:
        class _Input:
            request_id = "ok-job"
            return_outputs_as_base64 = False
        input = _Input()

    class _RealResponseStore:
        def __init__(self):
            self._calls = 0

        async def get(self, _):
            self._calls += 1
            if self._calls <= 2:  # first job's two get() calls
                return None
            r = type("R", (), {})()
            r.status = "generated"
            r.message = ""
            r.comfyui_response = None
            r.output = []
            return r

        async def set(self, request_id, result):
            processed.append(request_id)

        async def delete(self, _): return None

    queue = asyncio.Queue()
    await queue.put("missing-job")
    await queue.put("ok-job")
    await queue.put(None)

    w = PostprocessWorker(
        worker_id="t",
        kwargs={
            "preprocess_queue":  None,
            "generation_queue":  None,
            "postprocess_queue": queue,
            "request_store":     _RealRequestStore(),
            "response_store":    _RealResponseStore(),
        },
    )
    w.output_dir = tmp_path

    await asyncio.wait_for(w.work(), timeout=2.0)

    assert "ok-job" in processed, (
        "worker should still process the second job after the first "
        "had a missing request_store entry"
    )
