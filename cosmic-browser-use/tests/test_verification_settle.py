"""Regression tests for the guarded verification settle deadline."""

import asyncio
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cosmic_types import ActionResult, ActionType  # noqa: E402
from main import VerificationProfile, _wait_for_verification_settle  # noqa: E402
from orchestrator import Orchestrator  # noqa: E402


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


def test_model_profile_request_is_parsed_without_changing_the_action():
    orchestrator = Orchestrator.__new__(Orchestrator)
    for value, expected in (("realtime_canvas", "realtime_canvas"), ("standard", "standard"),
                            ("auto", "auto"), ("anything_else", None), ([], None)):
        import json
        response = orchestrator._parse_response(json.dumps({
            "action_type": "PressKey", "parameters": {"key": "Space"},
            "verification_profile_request": value,
        }))
        assert response.tool_call.action_type == ActionType.PRESS_KEY
        assert response.verification_profile_request == expected
        assert response.to_dict()["verification_profile_request"] == expected


def test_profile_scopes_to_page_and_url():
    profile = VerificationProfile()
    page_a, page_b = object(), object()
    profile.request("realtime_canvas", page_a, "https://game.test/")
    assert not profile.sync(page_a, "https://game.test/")
    assert profile.mode == "realtime_canvas"
    assert profile.sync(page_a, "https://other.test/")
    assert profile.mode == "auto"
    profile.request("standard", page_a, "https://game.test/")
    assert profile.sync(page_b, "https://game.test/")
    assert profile.mode == "auto"
    profile.request("realtime_canvas", page_a, "https://game.test/")
    assert profile.after_action(action(ActionType.RELOAD), page_a, "https://game.test/", None, True) == "navigation_or_tab_action"
    assert profile.mode == "auto"
    profile.request("realtime_canvas", page_a, "about:blank")
    assert profile.after_action(
        action(ActionType.NAVIGATE), page_a, "https://game.test/", "realtime_canvas", True,
    ) is None
    assert profile.mode == "realtime_canvas"
    assert profile.url == "https://game.test/"
    assert profile.after_action(
        action(ActionType.SWITCH_TAB), page_b, "https://game.test/", None, True,
    ) == "navigation_or_tab_action"
    assert profile.mode == "auto"


def test_requested_canvas_profile_keeps_all_runtime_guards():
    async def scenario():
        browser = SettleBrowser(canvas_frame=True)
        page = SimpleNamespace(url="https://game.test/")
        keypress = action(ActionType.PRESS_KEY)
        mode, _ = await _wait_for_verification_settle(
            browser, page, keypress, 0.05,
            goal="Reach the next obstacle", key="Space", profile="realtime_canvas",
        )
        assert mode == "canvas_frame"
        for result, key, profile in (
            (keypress, "Enter", "realtime_canvas"),
            (action(ActionType.DOM_CLICK), "Space", "realtime_canvas"),
            (keypress, "Space", "standard"),
        ):
            mode, waited_ms = await _wait_for_verification_settle(
                browser, page, result, 0.03,
                goal="Play the Dino cactus game", key=key, profile=profile,
            )
            assert mode == "deadline" and waited_ms >= 25
        assert browser.canvas_calls == 1
        browser.canvas_frame = False
        mode, waited_ms = await _wait_for_verification_settle(
            browser, page, keypress, 0.04,
            goal="Reach the next obstacle", key="Space", profile="realtime_canvas",
        )
        assert mode == "deadline" and waited_ms >= 35

    asyncio.run(scenario())
