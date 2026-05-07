"""Tests for the `__RANDOM_INT__` placeholder substitution.

The placeholder mechanism is small but load-bearing — every
example payload uses it for the seed. Lock the contract:

- Whole-string match becomes a real int (KSampler needs an int).
- Embedded match stays a string, with the placeholder substituted.
- Each occurrence gets an independent random value.
- Range is bounded.
- Non-placeholder strings are untouched.
"""
from modifiers.basemodifier import BaseModifier


def test_whole_string_becomes_int():
    m = BaseModifier({})
    out = m.replace_random_ints({"seed": "__RANDOM_INT__"})
    assert isinstance(out["seed"], int)
    assert 0 <= out["seed"] <= 2**32 - 1


def test_embedded_in_string_stays_string():
    m = BaseModifier({})
    out = m.replace_random_ints({"filename_prefix": "run-__RANDOM_INT__-out"})
    val = out["filename_prefix"]
    assert isinstance(val, str)
    assert val.startswith("run-") and val.endswith("-out")
    middle = val[len("run-"):-len("-out")]
    assert middle.isdigit()


def test_distinct_occurrences_get_distinct_values():
    """Two `__RANDOM_INT__` placeholders in one workflow MUST
    resolve independently — otherwise multi-`SaveImage` workflows
    that want distinct seeds per branch would all collide."""
    m = BaseModifier({})
    out = m.replace_random_ints({"a": "__RANDOM_INT__", "b": "__RANDOM_INT__"})
    # Astronomically unlikely to collide on random 32-bit ints.
    assert out["a"] != out["b"], "expected independent random values"


def test_recurses_into_nested_dicts_and_lists():
    m = BaseModifier({})
    out = m.replace_random_ints({
        "nodes": {
            "3": {"inputs": {"seed": "__RANDOM_INT__"}},
            "list": ["__RANDOM_INT__", "literal", "__RANDOM_INT__"],
        }
    })
    assert isinstance(out["nodes"]["3"]["inputs"]["seed"], int)
    assert isinstance(out["nodes"]["list"][0], int)
    assert out["nodes"]["list"][1] == "literal"
    assert isinstance(out["nodes"]["list"][2], int)


def test_non_placeholder_strings_untouched():
    """Substring presence isn't enough — the literal token shape
    is what gets matched. A field happening to contain "RANDOM"
    or "INT" isn't affected."""
    m = BaseModifier({})
    out = m.replace_random_ints({
        "prompt": "draw a random integer",
        "name":   "RANDOMINT",
        "tag":    "_RANDOM_INT_",
    })
    assert out["prompt"] == "draw a random integer"
    assert out["name"]   == "RANDOMINT"
    assert out["tag"]    == "_RANDOM_INT_"


def test_non_string_values_untouched():
    """Numbers, bools, None — pass through unchanged."""
    m = BaseModifier({})
    out = m.replace_random_ints({"steps": 20, "denoise": 0.87, "cfg_present": True, "ph": None})
    assert out == {"steps": 20, "denoise": 0.87, "cfg_present": True, "ph": None}
