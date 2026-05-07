"""Regression test: postprocess must not raise `shutil.SameFileError`
when a request_id is reused AND ComfyUI's prompt cache hits with
the same outputs.

The failure shape: on the second run for the same request_id, the
top-level output filename is a symlink left behind by the first
run, pointing into the per-request directory. `original_path
.resolve()` follows the symlink, lands at exactly where
`dest_path` is, and `shutil.copy2(src, dst)` raises
`SameFileError: '...' and '...' are the same file`.

The fix detects same-file pre-copy and returns the result entry
unchanged.
"""
import asyncio
import os
import shutil

import pytest

from workers.postprocess_worker import PostprocessWorker


def _make_worker(tmp_path, monkeypatch):
    """Wire a PostprocessWorker against a tmp output dir.

    `self.output_dir` is constructed once at __init__ from the
    config-level `OUTPUT_DIR`, so we monkeypatch the worker's
    attribute directly after construction."""
    class _StubStore:
        async def get(self, _):    return None
        async def set(self, *_):   return None
        async def delete(self, _): return None

    w = PostprocessWorker(
        worker_id="t",
        kwargs={
            "preprocess_queue":  None,
            "generation_queue":  None,
            "postprocess_queue": None,
            "request_store":     _StubStore(),
            "response_store":    _StubStore(),
        },
    )
    w.output_dir = tmp_path
    return w


@pytest.mark.asyncio
async def test_duplicate_request_id_does_not_raise_same_file_error(tmp_path, monkeypatch):
    """Simulate the exact failure scenario:

    - Request 'abc' ran once. Output ended up at
      `<output>/abc/img.png`. The wrapper left a symlink at
      `<output>/img.png` pointing into the per-request dir.
    - Request 'abc' is being processed again (cached prompt
      returned the same output). ComfyUI's history reports
      `img.png` (top-level) as the output file.

    Without the fix, `_process_output_file` calls
    `shutil.copy2(<output>/abc/img.png, <output>/abc/img.png)`
    and raises SameFileError. With the fix, it short-circuits
    and returns the result entry.
    """
    w = _make_worker(tmp_path, monkeypatch)

    request_id = "abc"
    job_dir = tmp_path / request_id
    job_dir.mkdir()

    # Real file already in the per-request dir from the prior run.
    real_file = job_dir / "img.png"
    real_file.write_bytes(b"\x89PNG-fake")

    # Symlink at top-level OUTPUT_DIR pointing into the per-request
    # dir — mimics what the wrapper wrote on the first run.
    top_link = tmp_path / "img.png"
    os.symlink(str(real_file), str(top_link))

    # Sanity: the symlink resolves to the real file.
    assert top_link.resolve() == real_file.resolve()

    item = {"filename": "img.png", "subfolder": "", "type": "output"}

    result = await w._process_output_file(
        item, job_output_dir=job_dir, request_id=request_id,
        node_id="9", output_type="images",
    )

    assert result is not None, "should not return None on duplicate request_id"
    assert result["filename"]   == "img.png"
    assert result["local_path"] == str(real_file)
    # File still readable (we didn't accidentally delete or mangle it).
    assert real_file.read_bytes() == b"\x89PNG-fake"
    # Symlink at top-level still points into the request dir.
    assert top_link.is_symlink()
    assert top_link.resolve() == real_file.resolve()


@pytest.mark.asyncio
async def test_first_run_still_copies_and_symlinks(tmp_path, monkeypatch):
    """Regression guard for the happy path: a fresh run with no
    pre-existing per-request directory still copies the top-level
    file in and replaces it with a symlink pointing into the
    per-request dir."""
    w = _make_worker(tmp_path, monkeypatch)

    request_id = "fresh"
    job_dir = tmp_path / request_id
    job_dir.mkdir()

    # ComfyUI just wrote a file at the top of OUTPUT_DIR.
    src = tmp_path / "img.png"
    src.write_bytes(b"FRESH")

    item = {"filename": "img.png", "subfolder": "", "type": "output"}

    result = await w._process_output_file(
        item, job_output_dir=job_dir, request_id=request_id,
        node_id="9", output_type="images",
    )

    assert result is not None
    # File copied into the per-request dir.
    dest = job_dir / "img.png"
    assert dest.exists() and dest.read_bytes() == b"FRESH"
    # Top-level location is now a symlink to the copy.
    assert src.is_symlink()
    assert src.resolve() == dest.resolve()
