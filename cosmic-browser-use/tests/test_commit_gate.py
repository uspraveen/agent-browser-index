"""Commit gate: every commit-like control is held before it fires.

The controller only enforces the decision; the policy (orchestrator
authorization or a user confirmation card) lives in the injected handler.
Without a handler the gate is off, preserving standalone CLI/demo runs.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from browser_controller import BrowserController  # noqa: E402
from cosmic_types import ActionType  # noqa: E402


class _FakePage:
    url = "https://example.com/checkout"


_COMMIT_INFO = {
    "is_commit": True,
    "name": "Submit application",
    "is_submit_control": True,
    "matched": ["submit", "apply"],
    "irreversible": False,
    "fields": [
        {"label": "Full name", "value": "Praveen Raj"},
        {"label": "Address", "value": "13500 Chenal Pkwy, Apt 2309"},
    ],
}


def _controller(handler):
    controller = object.__new__(BrowserController)
    controller.commit_gate_handler = handler
    controller.commit_blocked_count = 0
    controller.page = _FakePage()
    return controller


def _gate(controller, info=None):
    return asyncio.run(
        controller._gate_commit(
            action_type=ActionType.SNAPSHOT_CLICK,
            description="SnapshotClick @e7",
            info=info or _COMMIT_INFO,
            target="Submit application",
        )
    )


def test_gate_is_off_without_a_handler():
    assert _gate(_controller(None)) is None


def test_gate_allows_when_the_handler_allows():
    async def handler(payload):
        return {"allowed": True, "reason": "user asked for this"}

    assert _gate(_controller(handler)) is None


def test_gate_blocks_with_the_reason_and_counts():
    async def handler(payload):
        return {"allowed": False, "reason": "waiting on user confirmation"}

    controller = _controller(handler)
    result = _gate(controller)
    assert result is not None
    assert result.success is False
    assert "commit_blocked" in result.error
    assert "waiting on user confirmation" in result.error
    assert "NOT performed" in result.error
    assert controller.commit_blocked_count == 1


def test_gate_fails_closed_when_the_channel_raises():
    async def handler(payload):
        raise RuntimeError("gateway down")

    controller = _controller(handler)
    result = _gate(controller)
    assert result is not None
    assert "authorization channel failed" in result.error
    assert controller.commit_blocked_count == 1


def test_gate_payload_carries_fields_and_url_for_the_card():
    seen: dict = {}

    async def handler(payload):
        seen.update(payload)
        return {"allowed": True}

    _gate(_controller(handler))
    assert seen["url"] == "https://example.com/checkout"
    assert seen["target"] == "Submit application"
    assert seen["control"]["matched"] == ["submit", "apply"]
    assert seen["irreversible"] is False
    assert [field["label"] for field in seen["fields"]] == ["Full name", "Address"]


def test_enter_is_gated_when_the_active_form_submits():
    async def handler(payload):
        return {"allowed": False, "reason": "not authorized"}

    controller = _controller(handler)
    controller.cursor_overlay = type("Overlay", (), {"show_key": staticmethod(lambda *a, **k: asyncio.sleep(0))})()

    async def fake_probe(frame=None):
        return dict(_COMMIT_INFO, name="Submit")

    controller._enter_commit_probe = fake_probe
    result = asyncio.run(controller._press_key("Enter"))
    assert result.success is False
    assert "commit_blocked" in result.error
