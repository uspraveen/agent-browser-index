"""Regression tests for the DOMSnapshot / @ref tool set.

The snapshot is a perception shortcut, and shortcuts get blamed when they
guess. These pin the pieces that must never guess: password values never
leave the page as text, a ref resolves only when its fingerprint proves it
is the same element, and a value change alone never makes a ref stale
(typing between snapshot and act is the normal case, not the failure case).

Run with:  python -m pytest tests/test_snapshot_tools.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from browser_controller import (  # noqa: E402
    fingerprint_matches,
    format_snapshot_lines,
    mask_snapshot_value,
    parse_ref,
    snapshot_fingerprint,
)


class TestRefParsing:
    def test_accepts_wellformed_refs(self):
        assert parse_ref("@e5") == "@e5"
        assert parse_ref(" @e12 ") == "@e12"
        assert parse_ref("@e1") == "@e1"

    def test_rejects_malformed_refs(self):
        assert parse_ref("e5") is None
        assert parse_ref("@5") is None
        assert parse_ref("@e") is None
        assert parse_ref("@efive") is None
        assert parse_ref("") is None
        assert parse_ref(None) is None


class TestValueMasking:
    def test_passwords_never_appear_as_text(self):
        assert mask_snapshot_value("hunter2", True) == "********"
        assert "hunter2" not in mask_snapshot_value("hunter2", True)

    def test_empty_password_masks_to_empty(self):
        assert mask_snapshot_value("", True) == ""
        assert mask_snapshot_value(None, True) == ""

    def test_normal_values_pass_through_bounded(self):
        assert mask_snapshot_value("Test User", False) == "Test User"
        assert len(mask_snapshot_value("x" * 100, False)) == 40


class TestSnapshotLines:
    def test_line_carries_ref_role_and_name(self):
        lines = format_snapshot_lines([
            {"tag": "button", "role": "button", "name": "Apply for this Job"},
        ])
        assert lines == ['@e1 button "Apply for this Job"']

    def test_textbox_shows_value_and_checkbox_shows_state(self):
        lines = format_snapshot_lines([
            {"tag": "input", "role": "textbox", "name": "Legal Name", "value": "Test User"},
            {"tag": "input", "role": "checkbox", "name": "Remote, US", "checked": True},
            {"tag": "input", "role": "checkbox", "name": "Visa", "checked": False},
        ])
        assert "@e1 textbox \"Legal Name\" value='Test User'" in lines[0]
        assert "checked" in lines[1]
        assert "unchecked" in lines[2]

    def test_truncation_marker_ends_the_map(self):
        lines = format_snapshot_lines([
            {"tag": "a", "role": "link", "name": "one"},
            {"truncated": True},
        ])
        assert len(lines) == 2
        assert "truncated" in lines[1]

    def test_unnamed_elements_still_get_refs(self):
        lines = format_snapshot_lines([{"tag": "div", "role": "button", "name": ""}])
        assert lines == ["@e1 button"]


class TestFingerprints:
    def test_fingerprint_excludes_the_value(self):
        entry = {"tag": "input", "id": "name", "name": "Legal Name", "value": "Test User"}
        fp = snapshot_fingerprint(entry)
        assert "value" not in fp
        assert fp == {"tag": "input", "id": "name", "name": "Legal Name"}

    def test_same_element_matches_after_typing(self):
        before = {"tag": "input", "id": "name", "name": "Legal Name", "value": ""}
        after_snapshot = snapshot_fingerprint(before)
        after_typing = {"tag": "input", "id": "name", "name": "Legal Name", "value": "Test User"}
        assert fingerprint_matches(after_snapshot, after_typing)

    def test_different_element_is_caught(self):
        before = snapshot_fingerprint({"tag": "input", "id": "name", "name": "Legal Name"})
        rerendered = {"tag": "input", "id": "salary", "name": "Compensation expectations"}
        assert not fingerprint_matches(before, rerendered)

    def test_name_change_is_caught_even_with_same_tag_and_id(self):
        before = snapshot_fingerprint({"tag": "input", "id": "f1", "name": "Legal Name"})
        relabeled = {"tag": "input", "id": "f1", "name": "Preferred Name"}
        assert not fingerprint_matches(before, relabeled)

    def test_missing_fields_do_not_crash(self):
        assert fingerprint_matches({}, {}) is True
        assert fingerprint_matches({"tag": "input"}, {}) is False
