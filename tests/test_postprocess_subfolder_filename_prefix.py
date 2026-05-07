"""Regression tests for postprocess output discovery.

Two failure modes both surface as ``Processed 0 output files for
<request_id>`` despite the file being on disk:

1. Workflow-author subfolders (``SaveVideo`` with ``filename_prefix:
   "video/foo"`` writes to ``output/video/foo.mp4``) used to be
   silently dropped because an over-eager defensive guard refused
   anything outside the top-level or the current request's
   per-request directory.

2. Cache hits on a previously-seen prompt resolve into the *prior*
   request's per-request directory (that's where postprocess copied
   the file on the original run, and the top-level symlink still
   points there). The same defensive guard refused those too.

Both were the same broken assumption: "anything not under the current
request's directory is stale". In reality the stale-cache concern
doesn't exist — ComfyUI's history is the source of truth for which
file belongs to this job, and we should copy whatever it points at
into our per-request directory.
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
async def test_cache_hit_resolving_into_prior_request_subdir(tmp_path):
    """Cache hit on the same prompt as a prior job: ComfyUI's history
    reports the same filename, the top-level entry is a symlink into
    the prior request's per-request directory (where the wrapper
    parked the file on the original run), and the resolved real path
    therefore lives under another UUID-shaped subdirectory.

    This is **not** stale data — it's the same prompt's output, which
    is the whole point of ComfyUI's prompt cache. The wrapper must
    accept it and copy it into the current request's directory rather
    than silently dropping it (which is what an earlier over-eager
    "different request directory" guard did, surfacing as
    ``Processed 0 output files`` for every cache hit beyond the first).
    """
    w = _make_worker(tmp_path)

    current_id = str(uuid.uuid4())
    other_id   = str(uuid.uuid4())
    assert current_id != other_id

    # Prior run's per-request directory holds the actual file.
    other_dir = tmp_path / other_id
    other_dir.mkdir()
    real_file = other_dir / "img.png"
    real_file.write_bytes(b"CACHED")

    # Top-level entry is a symlink into that directory — what
    # postprocess wrote on the prior run.
    top_link = tmp_path / "img.png"
    os.symlink(str(real_file), str(top_link))

    job_dir = tmp_path / current_id
    job_dir.mkdir()

    item = {"filename": "img.png", "subfolder": "", "type": "output"}

    result = await w._process_output_file(
        item, job_output_dir=job_dir, request_id=current_id,
        node_id="9", output_type="images",
    )

    assert result is not None, (
        "cache-hit output must not be dropped just because it resolves "
        "into another request's per-request directory"
    )
    assert result["filename"] == "img.png"
    dest = job_dir / "img.png"
    assert dest.exists() and dest.read_bytes() == b"CACHED"


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
