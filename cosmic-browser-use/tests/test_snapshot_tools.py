"""Regression tests for the DOMSnapshot / @ref tool set.

The snapshot is a perception shortcut, and shortcuts get blamed when they
guess. These pin the pieces that must never guess: password values never
leave the page as text, a ref resolves only when its fingerprint proves it
is the same element, and a value change alone never makes a ref stale
(typing between snapshot and act is the normal case, not the failure case).

Run with:  python -m pytest tests/test_snapshot_tools.py -q
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from browser_controller import (  # noqa: E402
    _SNAPSHOT_INTERACTIVE_CSS,
    BrowserController,
    fingerprint_matches,
    format_snapshot_lines,
    mask_snapshot_value,
    parse_ref,
    snapshot_fingerprint,
    typing_landed_value,
)


class TestInteractiveCss:
    """Pin the interactive-role surface so collector parity with reference
    implementations is not lost silently."""

    def test_menu_roles_are_collected(self):
        assert "[role='menuitem']" in _SNAPSHOT_INTERACTIVE_CSS
        assert "[role='menuitemradio']" in _SNAPSHOT_INTERACTIVE_CSS


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


class TestTypingLanded:
    """The verified-typing gate: every type must prove the text reached the
    field, and the proof must tolerate the ways sites legitimately transform
    values — otherwise the fill fallback fires pointlessly."""

    def test_exact_match_lands(self):
        assert typing_landed_value("Why do AI models", "Why do AI models")

    def test_case_and_padding_transformations_count(self):
        assert typing_landed_value("WHY DO AI MODELS", "Why do AI models")
        assert typing_landed_value("  Why do AI models  ", "Why do AI models")

    def test_mask_grouping_counts(self):
        # Card formatters insert spaces the raw text never had.
        assert typing_landed_value("4111 1111 1111 1111", "4111111111111111")

    def test_maxlength_truncation_counts_when_substantial(self):
        # Half the text surviving reads as a maxLength-enforced field.
        assert typing_landed_value("Why do AI models give such sim", "Why do AI models give such similar answers?")

    def test_swallowed_keystrokes_never_count_as_landed(self):
        # meta.ai's focus steal left 'Why' / 'W' behind — those are failures,
        # not truncation, and must trigger the fill fallback.
        assert not typing_landed_value("Why", "Why do AI models give such similar answers?")
        assert not typing_landed_value("W", "Why do AI models give such similar answers?")

    def test_empty_or_missing_value_is_not_landed(self):
        assert not typing_landed_value("", "Why do AI models")
        assert not typing_landed_value(None, "Why do AI models")

    def test_clearing_a_field_succeeds(self):
        assert typing_landed_value("", "")
        assert typing_landed_value(None, "")

    def test_unrelated_value_is_not_landed(self):
        assert not typing_landed_value("Search", "Why do AI models")
        assert not typing_landed_value("W", "Why do AI models give such similar answers?")


class _FakeSnapshotFrame:
    """Returns a canned _SNAPSHOT_COLLECT_JS result, recording the cap it got.
    Like the real collector JS, it truncates at the requested cap and marks
    the overflow."""

    def __init__(self, entries):
        self._entries = entries
        self.caps = []

    async def evaluate(self, _js, arg=None):
        cap = (arg or {}).get("cap")
        self.caps.append(cap)
        entries = self._entries[:cap]
        truncated = len(self._entries) > cap
        if truncated:
            entries = entries + [{"truncated": True}]
        return {"count": len(entries), "truncated": truncated, "entries": entries}


def _snapshot_controller(frames):
    controller = object.__new__(BrowserController)
    controller._frame_search_order = lambda: frames
    return controller


class TestCollectSnapshot:
    """The structured core _dom_snapshot renders from — and the Jev engine
    consumes directly. Refs must number sequentially across frames while nth
    stays per-frame (it indexes that frame's own filtered list at recheck
    time), and structured fields must survive into the entries."""

    def _frames(self):
        main = _FakeSnapshotFrame([
            {"tag": "input", "id": "name", "role": "textbox", "name": "Legal Name", "value": ""},
            {"tag": "button", "role": "button", "name": "Continue"},
        ])
        iframe = _FakeSnapshotFrame([
            {"tag": "button", "role": "button", "name": "Pay now"},
        ])
        return [main, iframe]

    def test_refs_are_sequential_and_nth_is_per_frame(self):
        controller = _snapshot_controller(self._frames())
        collected = asyncio.run(controller._collect_snapshot(120))
        refs = collected["refs"]
        assert list(refs) == ["@e1", "@e2", "@e3"]
        assert refs["@e3"]["frame_index"] == 1
        # The iframe element's nth is its own frame index (0), not the global
        # running count (2) — a global offset made iframe refs permanently
        # "stale" at recheck time.
        assert refs["@e3"]["nth"] == 0
        assert refs["@e2"]["nth"] == 1
        assert [entry["ref"] for entry in collected["entries"]] == ["@e1", "@e2", "@e3"]
        assert collected["total"] == 3

    def test_structured_fields_survive_into_entries(self):
        frame = _FakeSnapshotFrame([
            {
                "tag": "input", "id": "pw", "role": "textbox", "name": "Password",
                "value": "********", "secret": True, "expanded": "false",
            },
            {
                "tag": "select", "role": "combobox", "name": "Country", "selected": "Germany",
                "options": [{"label": "Germany", "value": "de"}, {"label": "India", "value": "in"}],
            },
        ])
        controller = _snapshot_controller([frame])
        collected = asyncio.run(controller._collect_snapshot(120))
        by_ref = {entry["ref"]: entry for entry in collected["entries"]}
        assert by_ref["@e1"]["secret"] is True
        assert by_ref["@e1"]["expanded"] == "false"
        assert by_ref["@e2"]["options"] == [
            {"label": "Germany", "value": "de"},
            {"label": "India", "value": "in"},
        ]

    def test_rendered_map_is_unchanged_by_structured_fields(self):
        frame = _FakeSnapshotFrame([
            {"tag": "input", "role": "textbox", "name": "Legal Name", "value": "Test User"},
            {"tag": "select", "role": "combobox", "name": "Country", "selected": "Germany",
             "options": [{"label": "Germany", "value": "de"}], "secret": False, "expanded": "true"},
        ])
        controller = _snapshot_controller([frame])

        class _Page:
            url = "https://example.com/form"

        controller.page = _Page()
        result = asyncio.run(controller._dom_snapshot(120))
        lines = result.output.split("\n")
        assert lines[0].startswith("2 interactive elements")
        assert lines[1] == '@e1 textbox "Legal Name" value=\'Test User\''
        assert lines[2] == "@e2 combobox \"Country\" selected='Germany'"

    def test_zero_elements_errors_for_the_model(self):
        controller = _snapshot_controller([_FakeSnapshotFrame([])])
        result = asyncio.run(controller._dom_snapshot(120))
        assert result.success is False
        assert "vision tools" in result.error

    def test_cap_bounds_each_frame_request(self):
        frame = _FakeSnapshotFrame([{"tag": "button", "role": "button", "name": f"b{i}"} for i in range(5)])
        controller = _snapshot_controller([frame])
        collected = asyncio.run(controller._collect_snapshot(3))
        assert frame.caps == [3]
        assert collected["total"] == 3
        # The collector JS flagged the overflow; the structured core honors it.
        assert collected["truncated"] is True
