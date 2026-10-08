"""Rich-editor typing guard tests.

The Oct 2026 Hugging Face run typed a 13-line entry into a CodeMirror
editor on a 355-line file, and the typing verify-then-fill fallback
rewrote the whole document with just the entry: the landed-check read
the entire document (which never equals the typed text), judged the
typing failed, and fill replaced 86KB with 840 characters. The model
then reloaded the page and wiped its own remaining work. These tests pin
the three pieces that make that impossible again: documents verify by
shape-containment instead of literal prefixes, contenteditable targets
never get the fill fallback, and a navigation over unsaved editor typing
is rejected once so the discard is at least a decision.

Run with:  python -m pytest tests/test_rich_editor_typing.py -q
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from browser_controller import BrowserController, typing_landed_value  # noqa: E402
from cosmic_types import ActionType  # noqa: E402

ENTRY = (
    "{\n    cat: 'trained',\n    title: 'Jevify: open decision model',\n"
    "    date: '2026-10-08'\n}"
)


# ── landed verification: documents verify by shape ──────────────────────────


def test_autoclosed_brackets_still_count_as_landed():
    """CodeMirror auto-close inserts a closer after every '{' the agent
    types, so the document diverges from the typed stream by exactly the
    characters that must not count as failure."""
    value = "{\n}\n    cat: 'trained',\n    title: 'Jevify: open decision model',\n    date: '2026-10-08'\n}\n"
    assert typing_landed_value(value, ENTRY) is True


def test_swallowed_typing_still_reads_as_not_landed():
    assert typing_landed_value("W", ENTRY) is False
    assert typing_landed_value("", ENTRY) is False
    assert typing_landed_value(None, ENTRY) is False


def test_existing_semantics_unchanged():
    assert typing_landed_value(ENTRY, ENTRY) is True
    assert typing_landed_value(ENTRY[:40], ENTRY) is True, "substantial truncation still lands"
    assert typing_landed_value("junk", ENTRY) is False


# ── the fill fallback: banned on rich editors ────────────────────────────────


class _FakeLocator:
    def __init__(self, contenteditable: bool, value: str = ""):
        self._contenteditable = contenteditable
        self._value = value
        self.fill_calls: list[str] = []

    async def input_value(self):
        if self._contenteditable:
            raise RuntimeError("input_value: node is not an <input>")
        return self._value

    async def evaluate(self, script, arg=None):
        if "isContentEditable" in script:
            return self._contenteditable
        return self._value

    async def fill(self, text, timeout=None):
        self.fill_calls.append(text)
        self._value = text


def _controller() -> BrowserController:
    return object.__new__(BrowserController)


def test_fill_fallback_refused_on_contenteditable():
    locator = _FakeLocator(contenteditable=True)
    blocked = asyncio.run(_controller()._typing_fill_fallback(
        locator, ENTRY, ActionType.DOM_TYPE, ".cm-content"))
    assert blocked is not None
    assert blocked.success is False
    assert "rich text editor" in blocked.error
    assert "Click into the editor" in blocked.error
    assert locator.fill_calls == [], "a contenteditable document must never be filled"


def test_fill_fallback_still_works_on_plain_fields():
    locator = _FakeLocator(contenteditable=False, value="wrong")
    blocked = asyncio.run(_controller()._typing_fill_fallback(
        locator, "typed text", ActionType.SNAPSHOT_TYPE, "@e5"))
    assert blocked is None, "plain fields keep the deterministic fill"
    assert locator.fill_calls == ["typed text"]


# ── navigation over unsaved editor typing: reject once ───────────────────────


class _FakePage:
    def __init__(self):
        self.url = "https://huggingface.co/spaces/x/edit/main/news.html"
        self.goto_calls = 0
        self.reload_calls = 0

    async def goto(self, url, wait_until=None, timeout=None):
        self.goto_calls += 1

    async def reload(self, wait_until=None, timeout=None):
        self.reload_calls += 1

    async def wait_for_load_state(self, state=None, timeout=None):
        return True


def _nav_controller(page) -> BrowserController:
    controller = object.__new__(BrowserController)
    controller.page = page
    controller._pending_overlay_dismiss = False
    controller._unsaved_editor_typing = None
    return controller


def test_reload_over_unsaved_editor_typing_rejected_once():
    page = _FakePage()
    controller = _nav_controller(page)
    controller._unsaved_editor_typing = {"chars": 840, "warned": False}

    first = asyncio.run(controller._reload())
    assert first.success is False
    assert "discard 840 characters" in first.error
    assert "repeat the exact same navigation" in first.error
    assert page.reload_calls == 0, "the guarded reload must not touch the page"

    second = asyncio.run(controller._reload())
    assert second.success is True, "the explicit repeat is the decision to discard"
    assert page.reload_calls == 1
    assert controller._unsaved_editor_typing is None


def test_navigate_over_unsaved_editor_typing_rejected_once():
    page = _FakePage()
    controller = _nav_controller(page)
    controller._unsaved_editor_typing = {"chars": 840, "warned": False}

    first = asyncio.run(controller._navigate("https://huggingface.co/other"))
    assert first.success is False
    assert "rich text editor" in first.error
    assert page.goto_calls == 0

    second = asyncio.run(controller._navigate("https://huggingface.co/other"))
    assert second.success is True
    assert controller._unsaved_editor_typing is None


def test_navigation_unblocked_without_editor_typing():
    page = _FakePage()
    controller = _nav_controller(page)
    result = asyncio.run(controller._reload())
    assert result.success is True
    assert page.reload_calls == 1
