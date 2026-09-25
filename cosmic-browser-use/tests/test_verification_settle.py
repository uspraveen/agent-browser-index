"""Regression tests for the guarded verification settle deadline."""

import asyncio
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cosmic_types import ActionResult, ActionType  # noqa: E402
from main import _wait_for_verification_settle  # noqa: E402


class SettleBrowser:
    def __init__(self, canvas_frame=False):
        self.canvas_frame = canvas_frame
        self.canvas_calls = 0

    async def wait_for_canvas_input_frame(self):
        self.canvas_calls += 1
        return self.canvas_frame


def action(kind, success=True):
    return ActionResult(success=success, action_type=kind, description=kind.value)


def test_ordinary_page_action_and_failed_action_keep_the_old_deadline():
    async def scenario():
        browser = SettleBrowser(canvas_frame=True)
        for result in (action(ActionType.DOM_CLICK), action(ActionType.PRESS_KEY, success=False)):
            started = time.monotonic()
            mode, waited_ms = await _wait_for_verification_settle(
                browser, SimpleNamespace(url="https://example.test/"), result, 0.06,
                goal="Fill the application form", key="Space",
            )
            assert mode == "deadline" and waited_ms >= 55
            assert time.monotonic() - started >= 0.055
        assert browser.canvas_calls == 0
    asyncio.run(scenario())


def test_non_page_action_skips_only_the_settle_wait():
    async def scenario():
        browser = SettleBrowser()
        mode, waited_ms = await _wait_for_verification_settle(
            browser, SimpleNamespace(url="about:blank"), action(ActionType.SAVE_NOTE), 0.08,
        )
        assert mode == "action_complete"
        assert waited_ms == 0
        assert browser.canvas_calls == 0
    asyncio.run(scenario())


def test_canvas_frame_requires_explicit_game_goal_and_supported_key():
    async def scenario():
        browser = SettleBrowser(canvas_frame=True)
        for key in ("Space", " ", "ArrowUp"):
            mode, waited_ms = await _wait_for_verification_settle(
                browser, SimpleNamespace(url="https://chromedino.com/"),
                action(ActionType.PRESS_KEY), 0.06,
                goal="Play the Dino cactus game and beat my high score", key=key,
            )
            assert mode == "canvas_frame" and waited_ms < 40
        assert browser.canvas_calls == 3

        mode, waited_ms = await _wait_for_verification_settle(
            browser, SimpleNamespace(url="https://chromedino.com/"),
            action(ActionType.PRESS_KEY), 0.03,
            goal="Play the Dino cactus game", key="Enter",
        )
        assert mode == "deadline" and waited_ms >= 25
        assert browser.canvas_calls == 3
    asyncio.run(scenario())


def test_unavailable_canvas_frame_falls_back_to_original_deadline():
    async def scenario():
        browser = SettleBrowser(canvas_frame=False)
        mode, waited_ms = await _wait_for_verification_settle(
            browser, SimpleNamespace(url="https://chromedino.com/"),
            action(ActionType.PRESS_KEY), 0.05,
            goal="Play the Dino cactus game", key="Space",
        )
        assert mode == "deadline" and waited_ms >= 45
        assert browser.canvas_calls == 1
    asyncio.run(scenario())
