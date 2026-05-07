"""Tests for the postprocess base64 inlining path.

Exercises `inline_outputs_as_base64` directly with a fake `result`
holding `output[*].local_path` entries pointing at real temp
files. No ComfyUI / queues / HTTP needed.
"""
import asyncio
import base64
import os

import pytest

from workers.postprocess_worker import PostprocessWorker


def _make_worker():
    """Stub-storage worker so we can call the inline method
    without a Redis / memory cache."""
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
    """Minimal duck-typed Result with a mutable output list."""
    def __init__(self, output):
        self.output = output


@pytest.mark.asyncio
async def test_inline_base64_happy_path(tmp_path):
    """A small PNG-ish payload gets read, base64-encoded, and
    attached as `data` + `mimetype`."""
    payload = b"\x89PNG\r\n\x1a\n" + b"X" * 256  # 264 bytes total
    f = tmp_path / "img.png"
    f.write_bytes(payload)

    w = _make_worker()
    result = _Result([{"filename": "img.png", "local_path": str(f)}])
    await w.inline_outputs_as_base64("rid", result)

    assert "data" in result.output[0]
    assert base64.b64decode(result.output[0]["data"]) == payload
    assert "mimetype" in result.output[0]
    assert "error" not in result.output[0]


@pytest.mark.asyncio
async def test_inline_base64_size_cap(tmp_path, monkeypatch):
    """A file larger than OUTPUT_BASE64_MAX_BYTES is skipped with
    an `error` field rather than crashing or silently truncating.
    Other entries still process."""
    big = tmp_path / "big.png"
    big.write_bytes(b"X" * 4096)
    small = tmp_path / "small.png"
    small.write_bytes(b"OK")

    # Cap at 1 KB so `big` is over and `small` is under. The
    # postprocess worker re-imports this at call time via
    # `from config import OUTPUT_BASE64_MAX_BYTES`, which binds
    # to the package-level attribute — patch there.
    import config as cfg
    monkeypatch.setattr(cfg, "OUTPUT_BASE64_MAX_BYTES", 1024)

    w = _make_worker()
    result = _Result([
        {"filename": "big.png",   "local_path": str(big)},
        {"filename": "small.png", "local_path": str(small)},
    ])
    await w.inline_outputs_as_base64("rid", result)

    assert "data"  not in result.output[0]
    assert "error" in     result.output[0]
    assert "exceeds"  in   result.output[0]["error"]

    assert "data"  in     result.output[1]
    assert "error" not in result.output[1]


@pytest.mark.asyncio
async def test_inline_base64_missing_file(tmp_path):
    """Local path that doesn't exist gets an `error`, no crash."""
    w = _make_worker()
    result = _Result([{"filename": "gone.png", "local_path": str(tmp_path / "gone.png")}])
    await w.inline_outputs_as_base64("rid", result)
    assert "data"  not in result.output[0]
    assert "error" in     result.output[0]
    assert "missing"   in result.output[0]["error"]


@pytest.mark.asyncio
async def test_inline_base64_empty_output_list_is_noop():
    """No outputs to inline → no-op, no crash."""
    w = _make_worker()
    result = _Result([])
    await w.inline_outputs_as_base64("rid", result)
    assert result.output == []


@pytest.mark.asyncio
async def test_inline_base64_skips_entries_without_local_path(tmp_path):
    """Output entries lacking `local_path` are ignored (e.g. items
    that came from move_assets but didn't resolve to a file)."""
    f = tmp_path / "ok.png"
    f.write_bytes(b"OK")

    w = _make_worker()
    result = _Result([
        {"filename": "no-path.png"},
        {"filename": "ok.png", "local_path": str(f)},
    ])
    await w.inline_outputs_as_base64("rid", result)

    assert "data"  not in result.output[0]
    assert "error" not in result.output[0]
    assert "data"  in     result.output[1]
