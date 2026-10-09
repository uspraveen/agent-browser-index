"""Enter after a type action: never a half-filled form submit.

The 2026-10-08 Partiful run typed into the last question of a long form with
press_enter=true; Enter submitted the whole form with most required fields
empty. Inside a multi-field form Enter is now skipped (with a note the model
reads), a search form keeps it, and everything else still passes the commit
gate exactly as before.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from browser_controller import (  # noqa: E402
    _ENTER_FORM_SHAPE_JS,
    BrowserController,
    cap_indexed_db,
)
from cosmic_types import ActionType  # noqa: E402


class _Keyboard:
    def __init__(self):
        self.pressed: list = []

    async def press(self, key):
        self.pressed.append(key)


class _Page:
    url = "https://example.com/form"

    def __init__(self):
        self.keyboard = _Keyboard()

    async def wait_for_load_state(self, *_a, **_k):
        return None


def _controller(shape, commit=None, handler=None):
    controller = object.__new__(BrowserController)
    controller.commit_gate_handler = handler
    controller.commit_blocked_count = 0
    controller.page = _Page()
    controller._pending_dialogs = []
    controller.cursor_overlay = type("Overlay", (), {"show_key": staticmethod(lambda *a, **k: asyncio.sleep(0))})()

    async def fake_shape(frame=None):
        return shape

    async def fake_probe(frame=None):
        return commit

    controller._enter_form_shape = fake_shape
    controller._enter_commit_probe = fake_probe
    return controller


def _enter(controller):
    return asyncio.run(controller._enter_after_typing(
        None, action_type=ActionType.SNAPSHOT_TYPE, description="t", target="Enter in 'x'",
    ))


def test_multi_field_form_skips_enter_with_a_note():
    controller = _controller({"in_form": True, "text_fields": 3, "search": False})
    blocked, note = _enter(controller)
    assert blocked is None
    assert controller.page.keyboard.pressed == []
    assert "Enter was NOT pressed" in note and "3 text fields" in note


def test_single_field_form_still_presses_enter():
    controller = _controller({"in_form": True, "text_fields": 1, "search": False})
    blocked, note = _enter(controller)
    assert (blocked, note) == (None, "")
    assert controller.page.keyboard.pressed == ["Enter"]


def test_search_form_keeps_enter_even_with_several_fields():
    controller = _controller({"in_form": True, "text_fields": 2, "search": True})
    _enter(controller)
    assert controller.page.keyboard.pressed == ["Enter"]


def test_no_form_presses_enter():
    controller = _controller({"in_form": False, "text_fields": 0, "search": False})
    _enter(controller)
    assert controller.page.keyboard.pressed == ["Enter"]


def test_commit_gate_still_decides_a_single_field_submit():
    async def deny(_payload):
        return {"allowed": False, "reason": "not authorized"}

    controller = _controller(
        {"in_form": True, "text_fields": 1, "search": False},
        commit={"is_commit": True, "name": "Subscribe", "fields": []},
        handler=deny,
    )
    blocked, _ = _enter(controller)
    assert blocked is not None and "commit_blocked" in blocked.error
    assert controller.page.keyboard.pressed == []


# --- live shape probe -------------------------------------------------------

FORM = """
<form id="rsvp">
  <div><input id="name" placeholder="&nbsp;"><label>Your Name</label></div>
  <div><input id="tel" type="tel" name="phoneNumber"></div>
  <textarea id="msg"></textarea>
  <input type="hidden" name="token" value="x">
  <input type="checkbox" checked>
  <button type="submit">Continue</button>
</form>
<form role="search"><input id="q"><input id="where"><button>Go</button></form>
<input id="loose">
"""


async def _shape_for(element_id):
    try:
        from playwright.async_api import async_playwright
    except ImportError:  # pragma: no cover
        pytest.skip("playwright not installed")
    async with async_playwright() as p:
        try:
            browser = await p.chromium.launch(headless=True)
        except Exception as exc:  # pragma: no cover
            pytest.skip(f"chromium unavailable: {exc}")
        page = await browser.new_page()
        await page.set_content(FORM)
        await page.focus(f"#{element_id}")
        shape = await page.evaluate(_ENTER_FORM_SHAPE_JS)
        await browser.close()
        return shape


def test_live_shape_counts_visible_text_fields_only():
    shape = asyncio.new_event_loop().run_until_complete(_shape_for("name"))
    assert shape["in_form"] is True and shape["search"] is False
    assert shape["text_fields"] == 3  # name, tel, textarea — not hidden/checkbox/button


def test_live_shape_recognizes_a_search_form():
    shape = asyncio.new_event_loop().run_until_complete(_shape_for("q"))
    assert shape["search"] is True


def test_live_shape_outside_any_form():
    shape = asyncio.new_event_loop().run_until_complete(_shape_for("loose"))
    assert shape["in_form"] is False


# --- IndexedDB cap ------------------------------------------------------------

def test_small_indexed_db_survives_and_large_is_dropped():
    state = {
        "cookies": [{"name": "c"}],
        "origins": [
            {"origin": "https://partiful.com", "localStorage": [], "indexedDB": [{"name": "firebaseLocalStorageDb", "records": ["tok"]}]},
            {"origin": "https://big.example", "localStorage": [{"name": "k", "value": "v"}], "indexedDB": [{"blob": "x" * 5000}]},
        ],
    }
    capped = cap_indexed_db(state, per_origin_cap=1000)
    assert capped["origins"][0]["indexedDB"][0]["name"] == "firebaseLocalStorageDb"
    assert "indexedDB" not in capped["origins"][1]
    assert capped["origins"][1]["localStorage"] == [{"name": "k", "value": "v"}]
    assert capped["cookies"] == [{"name": "c"}]


def test_cap_tolerates_odd_shapes():
    assert cap_indexed_db(None) is None
    assert cap_indexed_db({"origins": [None, {"origin": "x"}]}) == {"origins": [None, {"origin": "x"}]}
