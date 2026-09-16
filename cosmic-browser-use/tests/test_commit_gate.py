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


class _FakeFrame:
    """Records the apply-edits call so a test can assert what the form got."""

    def __init__(self):
        self.calls: list = []

    async def evaluate(self, _js, arg=None):
        self.calls.append(arg)
        applied = [entry["label"] for entry in (arg or []) if entry.get("label") != "Ghost field"]
        pending = [entry["label"] for entry in (arg or []) if entry["label"] == "Ghost field"]
        return {"applied": applied, "pending": pending}


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
    controller._pending_dialogs = []
    return controller


def _gate(controller, info=None, frame=None):
    return asyncio.run(
        controller._gate_commit(
            action_type=ActionType.SNAPSHOT_CLICK,
            description="SnapshotClick @e7",
            info=info or _COMMIT_INFO,
            target="Submit application",
            frame=frame,
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


def test_gate_payload_reports_how_many_fields_are_empty():
    seen: dict = {}

    async def handler(payload):
        seen.update(payload)
        return {"allowed": True}

    info = dict(
        _COMMIT_INFO,
        fields=[{"label": "Email", "value": "uspraveenraj@gmail.com"}],
        empty_field_count=14,
    )
    _gate(_controller(handler), info=info)
    assert seen["fields"] == [{"label": "Email", "value": "uspraveenraj@gmail.com"}]
    assert seen["empty_field_count"] == 14


def test_card_edits_are_applied_to_the_form_before_an_allowed_commit():
    async def handler(payload):
        return {
            "allowed": True,
            "field_edits": [{"label": "Full name", "value": "Praveen Raj U S"}],
        }

    controller = _controller(handler)
    frame = _FakeFrame()
    assert _gate(controller, frame=frame) is None
    assert frame.calls == [[{"label": "Full name", "value": "Praveen Raj U S"}]]
    assert any(d["type"] == "commit_edits_applied" for d in controller._pending_dialogs)


def test_masked_card_edits_never_become_real_values():
    async def handler(payload):
        return {
            "allowed": True,
            "field_edits": [{"label": "Password", "value": "********"}],
        }

    controller = _controller(handler)
    frame = _FakeFrame()
    assert _gate(controller, frame=frame) is None
    assert frame.calls == []


def test_edits_that_match_no_field_are_reported_not_fatal():
    async def handler(payload):
        return {
            "allowed": True,
            "field_edits": [{"label": "Ghost field", "value": "x"}],
        }

    controller = _controller(handler)
    frame = _FakeFrame()
    assert _gate(controller, frame=frame) is None
    notes = [d for d in controller._pending_dialogs if d["type"] == "commit_edits_applied"]
    assert notes and "could not be matched" in notes[0]["message"]


def test_denied_commit_ignores_any_edits():
    async def handler(payload):
        return {
            "allowed": False,
            "field_edits": [{"label": "Full name", "value": "nope"}],
        }

    controller = _controller(handler)
    frame = _FakeFrame()
    result = _gate(controller, frame=frame)
    assert result is not None and result.success is False
    assert frame.calls == []


def test_note_on_approval_becomes_a_dialog_for_the_next_step():
    async def handler(payload):
        return {"allowed": True, "user_note": "approved, but uncheck the newsletter box"}

    controller = _controller(handler)
    assert _gate(controller) is None
    notes = [d for d in controller._pending_dialogs if d["type"] == "user_note"]
    assert notes and "uncheck the newsletter box" in notes[0]["message"]


def test_note_on_denial_lands_in_the_error_the_model_reads():
    async def handler(payload):
        return {"allowed": False, "reason": "not authorized", "user_note": "use the other resume"}

    result = _gate(_controller(handler))
    assert result is not None and result.success is False
    assert "use the other resume" in result.error
    assert "commit_blocked" in result.error


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


def _request(controller, params):
    return asyncio.run(controller._request_commit_authorization(params))


def test_model_can_add_a_hold_through_the_same_gate():
    seen: dict = {}

    async def handler(payload):
        seen.update(payload)
        return {"allowed": True, "reason": "user asked"}

    result = _request(_controller(handler), {"target": "File return"})
    assert result.success is True
    assert seen["source"] == "model_request"
    assert seen["model_declared"] is True
    # No ref/selector resolved means the classifier did not flag it: a miss.
    assert seen["classifier_miss"] is True
    assert seen["target"] == "File return"


def test_model_request_denied_is_final_and_counted():
    async def handler(payload):
        return {"allowed": False, "reason": "confirm with the user first"}

    controller = _controller(handler)
    result = _request(controller, {"target": "Transmit"})
    assert result.success is False
    assert "commit_blocked" in result.error
    assert controller.commit_blocked_count == 1


def test_model_request_with_gate_off_succeeds_without_authorizing_anything():
    result = _request(_controller(None), {"target": "Finalize"})
    assert result.success is True
    assert "gate" in (result.output or "")


def test_model_request_resolves_a_ref_and_clears_the_miss_flag():
    seen: dict = {}

    async def handler(payload):
        seen.update(payload)
        return {"allowed": True}

    class _Locator:
        async def evaluate(self, _js):
            return {
                "is_commit": True,
                "name": "File return",
                "is_submit_control": False,
                "matched": ["file"],
                "irreversible": False,
                "fields": [{"label": "Tax year", "value": "2025"}],
            }

    async def fake_resolve(ref, who):
        return (_Locator(), None, {"name": "File return"})

    controller = _controller(handler)
    controller._snapshot_resolve = fake_resolve
    result = _request(controller, {"target": "file thing", "ref": "@e9"})
    assert result.success is True
    assert seen["classifier_miss"] is False
    assert seen["target"] == "File return"
    assert seen["fields"] == [{"label": "Tax year", "value": "2025"}]
