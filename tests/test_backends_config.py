"""Tests for `_parse_backends` and `comfyui_urls` in config/config.py.

The backend-pool feature is opt-in via env (`COMFYUI_BACKENDS`);
existing single-backend deployments keep working without changes.
These tests pin the parser's behaviour so future edits don't drift.
"""
import pytest

from config.config import _parse_backends, comfyui_urls


def test_unset_falls_back_to_single_url():
    """When `COMFYUI_BACKENDS` is unset/empty, the parser yields a
    one-element list containing the fallback (typically
    `COMFYUI_API_BASE`). Existing deployments keep working."""
    assert _parse_backends(None,         "http://127.0.0.1:8188") == ["http://127.0.0.1:8188"]
    assert _parse_backends("",           "http://127.0.0.1:8188") == ["http://127.0.0.1:8188"]
    assert _parse_backends("   ",        "http://127.0.0.1:8188") == ["http://127.0.0.1:8188"]


def test_single_url_in_env():
    assert _parse_backends("http://127.0.0.1:8188", "ignored") == ["http://127.0.0.1:8188"]


def test_multiple_urls_comma_separated():
    raw = "http://127.0.0.1:8188,http://127.0.0.1:8189,http://127.0.0.1:8190"
    assert _parse_backends(raw, "ignored") == [
        "http://127.0.0.1:8188",
        "http://127.0.0.1:8189",
        "http://127.0.0.1:8190",
    ]


def test_whitespace_trimmed_and_trailing_slashes_stripped():
    raw = "  http://a:8188/  ,http://b:8189///"
    assert _parse_backends(raw, "ignored") == ["http://a:8188", "http://b:8189"]


def test_dedupes_duplicates_preserving_order():
    raw = "http://127.0.0.1:8188,http://127.0.0.1:8189,http://127.0.0.1:8188"
    assert _parse_backends(raw, "ignored") == [
        "http://127.0.0.1:8188",
        "http://127.0.0.1:8189",
    ]


def test_drops_empty_entries():
    raw = "http://a:8188,,http://b:8189,"
    assert _parse_backends(raw, "ignored") == ["http://a:8188", "http://b:8189"]


def test_falls_back_when_all_entries_invalid():
    """If every entry parses to empty (e.g. only commas), fall back."""
    assert _parse_backends(",,,", "http://127.0.0.1:8188") == ["http://127.0.0.1:8188"]


# ---- comfyui_urls -----------------------------------------------------


def test_comfyui_urls_http():
    u = comfyui_urls("http://127.0.0.1:8188")
    assert u["base"]         == "http://127.0.0.1:8188"
    assert u["prompt"]       == "http://127.0.0.1:8188/prompt"
    assert u["queue"]        == "http://127.0.0.1:8188/queue"
    assert u["history"]      == "http://127.0.0.1:8188/history"
    assert u["interrupt"]    == "http://127.0.0.1:8188/api/interrupt"
    assert u["free"]         == "http://127.0.0.1:8188/free"
    assert u["system_stats"] == "http://127.0.0.1:8188/system_stats"
    assert u["websocket"]    == "ws://127.0.0.1:8188/ws"


def test_comfyui_urls_https_to_wss():
    u = comfyui_urls("https://comfy.example.com")
    assert u["websocket"]    == "wss://comfy.example.com/ws"
    assert u["prompt"]       == "https://comfy.example.com/prompt"


def test_comfyui_urls_strips_trailing_slash():
    """Two URLs that differ only in a trailing slash should produce
    identical endpoint maps — so the parser-side stripping and the
    URL-builder-side stripping line up."""
    a = comfyui_urls("http://127.0.0.1:8188")
    b = comfyui_urls("http://127.0.0.1:8188/")
    assert a == b


def test_comfyui_urls_with_path_prefix():
    """Backends behind a reverse proxy at a sub-path (rare but
    legal). urljoin handles this correctly."""
    u = comfyui_urls("http://example.com/comfy")
    assert u["prompt"] == "http://example.com/comfy/prompt"
    assert u["websocket"] == "ws://example.com/comfy/ws"
