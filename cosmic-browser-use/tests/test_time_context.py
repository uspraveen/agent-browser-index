"""The browser planner and Jev use the same fresh clock as COSMIC's orchestrator."""

from datetime import datetime, timezone
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time_context  # noqa: E402
from orchestrator import Orchestrator  # noqa: E402


class _FrozenDateTime:
    @staticmethod
    def now(tz):
        return datetime(2026, 9, 24, 16, 30, tzinfo=timezone.utc).astimezone(tz)


def test_current_time_uses_utc_and_user_timezone(monkeypatch):
    monkeypatch.setattr(time_context, "datetime", _FrozenDateTime)
    lines = time_context.current_time_context("America/Chicago")
    assert lines == (
        "Current date and time (UTC): Thursday, September 24, 2026 at 16:30 UTC.\n"
        "User's local time: Thursday, September 24, 2026 at 11:30 AM CDT."
    )
    assert time_context.current_time_context("invalid/timezone") == (
        "Current date and time (UTC): Thursday, September 24, 2026 at 16:30 UTC."
    )


def test_planner_prompt_receives_fresh_clock(monkeypatch):
    monkeypatch.setattr(time_context, "datetime", _FrozenDateTime)
    prompt = Orchestrator.__new__(Orchestrator)._build_system_prompt({
        "goal": "Find a role", "estimated_progress": 0.0,
        "user_timezone": "America/Chicago",
    })
    assert "Current date and time (UTC): Thursday, September 24, 2026 at 16:30 UTC." in prompt
    assert "User's local time: Thursday, September 24, 2026 at 11:30 AM CDT." in prompt
