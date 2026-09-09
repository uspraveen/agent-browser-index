"""Tests for the human-takeover input relay.

This is the security boundary of the takeover feature: the reason the desktop
gets a validated relay instead of the raw CDP URL a hosted product can afford
to hand out. Everything below is either "the event a person could have
produced is faithfully translated" or "anything else is dropped".

Run with:  python -m pytest tests/test_human_input.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from browser_controller import BrowserController  # noqa: E402


class _Page:
    viewport_size = {"width": 1280, "height": 720}


class _Config:
    screenshot_max_width = 1280


@pytest.fixture()
def controller():
    # The real class, without its heavy constructor: _normalize_human_input
    # only ever touches the page viewport and the configured width.
    instance = BrowserController.__new__(BrowserController)
    instance.page = _Page()
    instance.config = _Config()
    return instance


# ------------------------------------------------------------------ mouse


def test_normalized_coordinates_map_onto_the_viewport(controller):
    out = controller._normalize_human_input(
        {"kind": "mouse", "type": "mousePressed", "x": 0.5, "y": 0.25, "button": "left", "clickCount": 1}
    )
    assert out["method"] == "Input.dispatchMouseEvent"
    assert out["params"]["x"] == 640
    assert out["params"]["y"] == 180


def test_coordinates_are_clamped_to_the_page(controller):
    # A pointer outside the streamed image must not become a negative or
    # off-page CDP coordinate.
    out = controller._normalize_human_input({"kind": "mouse", "type": "mouseMoved", "x": 4.0, "y": -2.0})
    assert out["params"]["x"] == 1280
    assert out["params"]["y"] == 0


def test_wheel_deltas_are_bounded(controller):
    out = controller._normalize_human_input(
        {"kind": "mouse", "type": "mouseWheel", "x": 0.1, "y": 0.1, "deltaY": 99999}
    )
    assert out["params"]["deltaY"] == 2000.0


def test_click_count_is_capped(controller):
    out = controller._normalize_human_input(
        {"kind": "mouse", "type": "mousePressed", "x": 0, "y": 0, "clickCount": 99}
    )
    assert out["params"]["clickCount"] == 3


def test_unknown_mouse_button_falls_back_to_none(controller):
    out = controller._normalize_human_input(
        {"kind": "mouse", "type": "mousePressed", "x": 0, "y": 0, "button": "sneaky"}
    )
    assert out["params"]["button"] == "none"


def test_unknown_mouse_event_type_is_dropped(controller):
    assert controller._normalize_human_input({"kind": "mouse", "type": "mouseDragged"}) is None


def test_non_numeric_coordinates_are_dropped(controller):
    assert controller._normalize_human_input({"kind": "mouse", "type": "mouseMoved", "x": "left"}) is None


# -------------------------------------------------------------------- key


def test_key_event_carries_the_fields_chrome_needs(controller):
    out = controller._normalize_human_input(
        {"kind": "key", "type": "keyDown", "key": "a", "code": "KeyA", "text": "a", "windowsVirtualKeyCode": 65}
    )
    assert out["method"] == "Input.dispatchKeyEvent"
    assert out["params"]["key"] == "a"
    assert out["params"]["code"] == "KeyA"
    assert out["params"]["windowsVirtualKeyCode"] == 65
    assert out["params"]["nativeVirtualKeyCode"] == 65


def test_unknown_key_event_type_is_dropped(controller):
    assert controller._normalize_human_input({"kind": "key", "type": "keyPressed"}) is None


def test_oversized_key_strings_are_truncated(controller):
    out = controller._normalize_human_input({"kind": "key", "type": "char", "text": "x" * 500})
    assert len(out["params"]["text"]) == 32


def test_modifiers_outside_the_valid_range_are_discarded(controller):
    out = controller._normalize_human_input({"kind": "key", "type": "keyDown", "modifiers": 9999})
    assert out["params"]["modifiers"] == 0


# ------------------------------------------------------------------- text


def test_insert_text_is_length_capped(controller):
    out = controller._normalize_human_input({"kind": "text", "text": "y" * 99999})
    assert out["method"] == "Input.insertText"
    assert len(out["params"]["text"]) == 4096


def test_empty_text_is_dropped(controller):
    assert controller._normalize_human_input({"kind": "text", "text": ""}) is None


# --------------------------------------------------------------- the gate


def test_unknown_kinds_are_dropped(controller):
    assert controller._normalize_human_input({"kind": "navigate", "url": "file:///etc/passwd"}) is None
    assert controller._normalize_human_input({"kind": "evaluate", "js": "fetch('/x')"}) is None
    assert controller._normalize_human_input({}) is None
    assert controller._normalize_human_input(None) is None


def test_a_raw_cdp_method_cannot_be_smuggled_through(controller):
    # The relay takes a described intent, never a method name, so naming a
    # method in the payload reaches nothing.
    assert controller._normalize_human_input(
        {"method": "Runtime.evaluate", "params": {"expression": "1"}}
    ) is None
    assert controller._normalize_human_input(
        {"kind": "mouse", "type": "mousePressed", "x": 0, "y": 0, "method": "Page.navigate"}
    )["method"] == "Input.dispatchMouseEvent"


def test_every_producible_method_is_on_the_whitelist(controller):
    produced = set()
    for event in (
        {"kind": "mouse", "type": "mousePressed", "x": 0, "y": 0},
        {"kind": "key", "type": "keyDown", "key": "a"},
        {"kind": "text", "text": "hello"},
    ):
        produced.add(controller._normalize_human_input(event)["method"])
    assert produced == BrowserController._HUMAN_INPUT_METHODS


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
