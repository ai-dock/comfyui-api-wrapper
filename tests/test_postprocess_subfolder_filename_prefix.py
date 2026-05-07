"""Regression test: postprocess must accept outputs under a workflow-author
subfolder (e.g. ``video/foo.mp4`` from ``filename_prefix: "video/foo"``).

Earlier the defensive stale-cache guard rejected *any* path whose first
segment under ``OUTPUT_DIR`` wasn't either top-level or the current
request id. That silently dropped legitimate outputs of workflows that
organise files into named subfolders, with the surface symptom
``Processed 0 output files for <request_id>`` despite the file being
present on disk.

The fix narrows the guard so it only rejects UUID-shaped subdirectories
that don't match the current request id (the real stale-cache case),
and leaves arbitrary workflow-author subfolders alone.
"""
import os
import uuid

import pytest

from workers.postprocess_worker import PostprocessWorker


def _make_worker(tmp_path):
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
async def test_workflow_subfolder_is_processed(tmp_path):
    """``SaveVideo`` with ``filename_prefix: "video/foo"`` writes to
    ``<output>/video/foo.mp4``. ComfyUI's history correctly reports it
    under ``images: [{filename: "foo.mp4", subfolder: "video"}]``. The
    file must be picked up, copied into the per-request directory, and
    returned in the result entry.
    """
    w = _make_worker(tmp_path)

    request_id = str(uuid.uuid4())
    job_dir = tmp_path / request_id
    job_dir.mkdir()

    subfolder_dir = tmp_path / "video"
    subfolder_dir.mkdir()
    src = subfolder_dir / "foo.mp4"
    src.write_bytes(b"VIDEO")

    item = {"filename": "foo.mp4", "subfolder": "video", "type": "output"}

    result = await w._process_output_file(
        item, job_output_dir=job_dir, request_id=request_id,
        node_id="75", output_type="images",
    )

    assert result is not None, (
        "workflow subfolder output should not be silently dropped"
    )
    assert result["filename"] == "foo.mp4"
    assert result["subfolder"] == "video"
    dest = job_dir / "foo.mp4"
    assert dest.exists() and dest.read_bytes() == b"VIDEO"


@pytest.mark.asyncio
async def test_other_request_subdir_still_rejected(tmp_path):
    """The stale-cache guard the original check was written to enforce.
    ComfyUI returned a path whose resolved location is inside *another*
    request's per-request directory — copying that file into ours would
    plagiarise a prior job's output. This must still be refused.
    """
    w = _make_worker(tmp_path)

    current_id = str(uuid.uuid4())
    other_id   = str(uuid.uuid4())
    assert current_id != other_id

    other_dir = tmp_path / other_id
    other_dir.mkdir()
    real_file = other_dir / "img.png"
    real_file.write_bytes(b"NOT MINE")

    # ComfyUI's history reports the file at top-level; the wrapper
    # symlinked it into the prior request's dir on that prior run.
    top_link = tmp_path / "img.png"
    os.symlink(str(real_file), str(top_link))

    job_dir = tmp_path / current_id
    job_dir.mkdir()

    item = {"filename": "img.png", "subfolder": "", "type": "output"}

    result = await w._process_output_file(
        item, job_output_dir=job_dir, request_id=current_id,
        node_id="9", output_type="images",
    )

    assert result is None, (
        "must not copy files that resolve into another request's directory"
    )
    # Per-request dir stays empty — we didn't accidentally claim the file.
    assert list(job_dir.iterdir()) == []


@pytest.mark.asyncio
async def test_self_request_subdir_is_processed(tmp_path):
    """Sanity: a file resolved into our *own* request directory (e.g.
    via a symlink we wrote on a prior run with the same request_id) is
    still accepted — covered by the same-file shortcut downstream."""
    w = _make_worker(tmp_path)

    request_id = str(uuid.uuid4())
    job_dir = tmp_path / request_id
    job_dir.mkdir()
    real_file = job_dir / "img.png"
    real_file.write_bytes(b"MINE")

    top_link = tmp_path / "img.png"
    os.symlink(str(real_file), str(top_link))

    item = {"filename": "img.png", "subfolder": "", "type": "output"}

    result = await w._process_output_file(
        item, job_output_dir=job_dir, request_id=request_id,
        node_id="9", output_type="images",
    )

    assert result is not None
    assert result["filename"] == "img.png"
