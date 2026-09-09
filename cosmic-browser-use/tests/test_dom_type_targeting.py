"""Regression tests for wrong-target typing.

The failure this module exists for: a job-application form reusing one
placeholder ("Type here...") across many fields, a DomType selector that
matched all of them, an executor that resolved to the first — and a salary
typed into the Legal Name field that nothing in the run ever noticed. The
guard refuses ambiguous selectors with the matched fields named; every
completed type echoes the label of the field that received the text and warns
when it replaced an existing value.

Run with:  python -m pytest tests/test_dom_type_targeting.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from browser_controller import (  # noqa: E402
    _ambiguous_type_error,
    _type_echo_description,
)

FIELDS = [
    {"label": "Legal Name", "value": "Test User"},
    {"label": "Preferred Name (if applicable)", "value": ""},
    {"label": "City", "value": ""},
    {"label": "Compensation expectations", "value": ""},
]


class TestAmbiguousSelectorGuard:
    def test_names_every_matched_field(self):
        error = _ambiguous_type_error("input[placeholder='Type here...']", 4, FIELDS)
        assert "matched 4 visible fields" in error
        for label in ("Legal Name", "Preferred Name (if applicable)", "City", "Compensation expectations"):
            assert label in error
        # The current value is the run's only memory of what it is about to
        # destroy — it must be in the refusal so the model can restore it.
        assert "Test User" in error

    def test_tells_the_model_how_to_recover(self):
        error = _ambiguous_type_error("input[type='text']", 2, FIELDS[:2])
        assert ":below(:text(" in error
        assert "exactly one visible field" in error

    def test_truncates_long_match_lists(self):
        many = [{"label": f"Field {i}", "value": ""} for i in range(11)]
        error = _ambiguous_type_error("input", 11, many)
        assert "... and 3 more" in error
        assert "Field 10" not in error  # capped at 8 listed fields

    def test_unlabeled_fields_still_get_listed(self):
        error = _ambiguous_type_error("input", 1, [{"label": "", "value": ""}])
        assert "(unlabeled)" in error


class TestTypeEchoDescription:
    def test_echoes_the_field_label(self):
        description, warning = _type_echo_description(
            "input[placeholder='Type here...']", "Legal Name", "$100,000 - $150,000", "Test User"
        )
        assert "field labeled 'Legal Name'" in description
        assert "'$100,000 - $150,000'" in description

    def test_overwrite_is_flagged_loudly(self):
        description, warning = _type_echo_description(
            "input[placeholder='Type here...']", "Legal Name", "$100,000 - $150,000", "Test User"
        )
        assert warning is not None and "Test User" in warning
        assert "WARNING" in description
        assert "Test User" in description

    def test_no_warning_for_a_fresh_field(self):
        description, warning = _type_echo_description(
            "input[placeholder='Type here...']", "Legal Name", "Test User", ""
        )
        assert warning is None
        assert "WARNING" not in description

    def test_no_warning_when_rewriting_the_same_value(self):
        description, warning = _type_echo_description(
            "input[placeholder='Type here...']", "Search", "hello", "hello"
        )
        assert warning is None
        assert "WARNING" not in description

    def test_unlabeled_target_falls_back_to_selector_only(self):
        description, warning = _type_echo_description(
            "input#q42", "", "hello", ""
        )
        assert "field labeled" not in description
        assert warning is None

    def test_long_labels_and_values_are_bounded(self):
        description, warning = _type_echo_description(
            "input", "L" * 500, "T" * 500, "P" * 500
        )
        assert len(description) < 800
        assert warning is not None and len(warning) < 120


class TestEchoOutputShape:
    """The output JSON is what TOOL_OUTPUT_DATA shows the model — it must
    carry the landing field, the replaced value, and the warning as data,
    not only as prose in the description."""

    def test_output_json_carries_the_echo(self):
        # Mirrors the payload _dom_type builds on success.
        label, previous_value, text = "Legal Name", "Test User", "$100,000 - $150,000"
        _, warning = _type_echo_description("sel", label, text, previous_value)
        payload = json.loads(json.dumps({
            "typed_into": {
                "label": label or None,
                "previous_value": previous_value[:60] or None,
                **({"warning": warning} if warning else {}),
            }
        }))
        assert payload["typed_into"]["label"] == "Legal Name"
        assert payload["typed_into"]["previous_value"] == "Test User"
        assert "Test User" in payload["typed_into"]["warning"]

    def test_no_warning_key_when_field_was_empty(self):
        _, warning = _type_echo_description("sel", "Legal Name", "Test User", "")
        payload = {"typed_into": {"label": "Legal Name", "previous_value": None}}
        if warning:
            payload["typed_into"]["warning"] = warning
        assert "warning" not in payload["typed_into"]
