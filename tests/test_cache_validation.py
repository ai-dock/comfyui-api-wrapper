"""Tests for the stale-cache empty-output detector.

ComfyUI's prompt cache assigns a NEW prompt_id on cache hit but
does NOT copy output references from the original execution — the
new history entry's `outputs` map is `{}`. The wrapper detected
this as 'completed' and produced empty Result envelopes.
`_validate_cached_outputs` is the gate: returns True iff there's
at least one usable output reference, so the caller can re-submit
when False.
"""
import pytest

from workers.generation_worker import GenerationWorker


def _make_worker():
    """Build a GenerationWorker with stub queues/stores. Only the
    pure validator method is exercised — none of the queue / HTTP
    paths run."""
    class _StubStore:
        async def get(self, _):    return None
        async def set(self, *_):   return None

    return GenerationWorker(
        worker_id="t",
        kwargs={
            "preprocess_queue":  None,
            "generation_queue":  None,
            "postprocess_queue": None,
            "request_store":     _StubStore(),
            "response_store":    _StubStore(),
        },
    )


@pytest.mark.asyncio
async def test_validate_cached_outputs_happy_path():
    w = _make_worker()
    response = {
        "abc-123": {
            "prompt": [],
            "outputs": {
                "9": {
                    "images": [
                        {"filename": "result_00001_.png", "subfolder": "", "type": "output"}
                    ]
                }
            },
            "status": {"completed": True},
        }
    }
    assert await w._validate_cached_outputs(response, "abc-123") is True


@pytest.mark.asyncio
async def test_validate_cached_outputs_empty_outputs_dict():
    """The exact bug shape: cache hit, prompt_id present, but
    outputs is `{}`. Wrapper must treat this as invalid so the
    caller can clear + re-submit."""
    w = _make_worker()
    response = {
        "abc-123": {
            "prompt": [],
            "outputs": {},
            "status": {"completed": True},
        }
    }
    assert await w._validate_cached_outputs(response, "abc-123") is False


@pytest.mark.asyncio
async def test_validate_cached_outputs_node_with_empty_lists():
    """Edge case: outputs map is non-empty but each node's output
    arrays are empty — also invalid."""
    w = _make_worker()
    response = {
        "abc-123": {
            "outputs": {
                "9": {"images": []},
                "10": {"gifs": []},
            }
        }
    }
    assert await w._validate_cached_outputs(response, "abc-123") is False


@pytest.mark.asyncio
async def test_validate_cached_outputs_non_dict_node():
    """If a node's outputs entry isn't a dict, skip it — but still
    count any sibling node with a populated list."""
    w = _make_worker()
    response = {
        "abc-123": {
            "outputs": {
                "9":  "garbage",
                "10": {"images": [{"filename": "x.png"}]},
            }
        }
    }
    assert await w._validate_cached_outputs(response, "abc-123") is True


@pytest.mark.asyncio
async def test_validate_cached_outputs_missing_prompt_id():
    w = _make_worker()
    response = {"some-other-id": {"outputs": {"9": {"images": [{"filename": "x"}]}}}}
    assert await w._validate_cached_outputs(response, "abc-123") is False


@pytest.mark.asyncio
async def test_validate_cached_outputs_none_or_garbage():
    w = _make_worker()
    assert await w._validate_cached_outputs(None,    "abc") is False
    assert await w._validate_cached_outputs({},      "abc") is False
    assert await w._validate_cached_outputs("nope",  "abc") is False
    assert await w._validate_cached_outputs({"abc": None}, "abc") is False
