"""Regression tests for the agent's memory layers.

These cover the gap that let a run burn 24 steps re-reading its own notes:
the agent could remember *that* it had written something, but whatever it read
back was gone from context one action later, so it read it again, and again.

Run with:  python -m pytest tests/test_agent_memory.py -q
"""

from __future__ import annotations

import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cosmic_types import (  # noqa: E402
    ActionResult,
    ActionType,
    BrowserState,
    Step,
    TaskConfig,
    VerificationStatus,
)
from memory_manager import MemoryManager  # noqa: E402
from orchestrator import Orchestrator  # noqa: E402


def _state(url: str = "https://example.com/careers", notes=None, index=None) -> BrowserState:
    return BrowserState(
        url=url,
        title="Careers",
        viewport_width=1280,
        viewport_height=800,
        scroll_y=0,
        screenshot_hash="hash-a",
        timestamp=datetime.now(),
        notes=list(notes or []),
        large_notes_index=list(index or []),
    )


def _step(
    number: int,
    action_type: ActionType,
    *,
    output: str | None = None,
    params: dict | None = None,
    success: bool = True,
    verification: VerificationStatus = VerificationStatus.SUCCESS,
    url: str = "https://example.com/careers",
    screenshot_hash: str = "hash-a",
) -> Step:
    action = ActionResult(
        success=success,
        action_type=action_type,
        description=f"step {number}",
        output=output,
    )
    action.verification_status = verification
    state = _state(url=url)
    state.screenshot_hash = screenshot_hash
    return Step(
        step_number=number,
        timestamp=datetime.now(),
        screenshot_path=f"step_{number}.jpg",
        screenshot_hash=screenshot_hash,
        browser_state=state,
        action=action,
        thinking="",
        tool_call={"action_type": action_type.value, "parameters": params or {}},
    )


def _memory(steps, tmp_path=None) -> MemoryManager:
    memory = MemoryManager(
        TaskConfig(task_id="t_test", goal="find the first 3 engineering jobs and their salaries"),
        working_dir=Path(tempfile.mkdtemp()),
    )
    memory.steps = list(steps)
    return memory


# ── The working set: a read has to outlive the next action ───────────────

def test_read_output_survives_the_next_action():
    memory = _memory([
        _step(1, ActionType.DOM_EXTRACT, output="job list: A, B, C", params={"selector": ".jobs"}),
        _step(2, ActionType.SAVE_LARGE_NOTE, params={"title": "jobs"}),
        _step(3, ActionType.VISUAL_CLICK, params={"description": "next"}),
    ])
    working_set = memory._working_set_for_prompt()
    outputs = [entry["output"] for entry in working_set]
    assert "job list: A, B, C" in outputs, "an extraction two steps back must still be readable"


def test_latest_output_is_not_duplicated_into_the_working_set():
    # steps[-1] is already rendered as TOOL_OUTPUT_DATA.
    memory = _memory([
        _step(1, ActionType.DOM_EXTRACT, output="older", params={"selector": ".a"}),
        _step(2, ActionType.DOM_EXTRACT, output="newest", params={"selector": ".b"}),
    ])
    outputs = [entry["output"] for entry in memory._working_set_for_prompt()]
    assert outputs == ["older"]


def test_repeated_reads_of_one_note_are_kept_once():
    memory = _memory([
        _step(1, ActionType.READ_LARGE_NOTE, output="note body v1", params={"note_id": "ln_1"}),
        _step(2, ActionType.DOM_EXTRACT, output="something else", params={"selector": ".x"}),
        _step(3, ActionType.READ_LARGE_NOTE, output="note body v2", params={"note_id": "ln_1"}),
        _step(4, ActionType.VISUAL_CLICK, params={"description": "next"}),
    ])
    working_set = memory._working_set_for_prompt()
    note_entries = [e for e in working_set if e["label"] == "note_id=ln_1"]
    assert len(note_entries) == 1
    assert note_entries[0]["output"] == "note body v2", "the newest read wins"


def test_working_set_is_bounded_and_marks_truncation():
    big = "x" * (MemoryManager.WORKING_SET_ENTRY_CHARS + 5000)
    memory = _memory([
        _step(1, ActionType.DOM_EXTRACT, output=big, params={"selector": ".a"}),
        _step(2, ActionType.DOM_EXTRACT, output=big, params={"selector": ".b"}),
        _step(3, ActionType.DOM_EXTRACT, output=big, params={"selector": ".c"}),
        _step(4, ActionType.DOM_EXTRACT, output=big, params={"selector": ".d"}),
        _step(5, ActionType.VISUAL_CLICK, params={"description": "next"}),
    ])
    working_set = memory._working_set_for_prompt()
    assert len(working_set) <= MemoryManager.WORKING_SET_MAX_ENTRIES
    total = sum(len(entry["output"]) for entry in working_set)
    assert total <= MemoryManager.WORKING_SET_TOTAL_CHARS
    assert all(entry["truncated"] for entry in working_set)


def test_actions_that_change_the_page_are_not_working_set_material():
    memory = _memory([
        _step(1, ActionType.VISUAL_CLICK, output="clicked", params={"description": "apply"}),
        _step(2, ActionType.NAVIGATE, output="went", params={"url": "https://example.com"}),
        _step(3, ActionType.VISUAL_CLICK, params={"description": "next"}),
    ])
    assert memory._working_set_for_prompt() == []


# ── Loop detection: a successful read is not progress ────────────────────

def test_alternating_read_and_extract_is_detected_as_a_loop():
    # Exactly the shape of the Fireworks run: ReadLargeNote / DOMExtract,
    # over and over, each one individually "successful".
    memory = _memory([
        _step(21, ActionType.READ_LARGE_NOTE, params={"note_id": "ln_1"}),
        _step(22, ActionType.DOM_EXTRACT, params={"selector": ".jobs"}),
        _step(23, ActionType.READ_LARGE_NOTE, params={"note_id": "ln_1"}),
        _step(24, ActionType.DOM_EXTRACT, params={"selector": ".jobs"}),
    ])
    assert memory.detect_loop() is True


def test_a_real_action_that_worked_still_clears_the_loop_flag():
    # The original guard's intent: two bad attempts then a good click is
    # self-correction, not a loop. That must keep working.
    memory = _memory([
        _step(21, ActionType.READ_LARGE_NOTE, params={"note_id": "ln_1"}),
        _step(22, ActionType.DOM_EXTRACT, params={"selector": ".jobs"}),
        _step(23, ActionType.READ_LARGE_NOTE, params={"note_id": "ln_1"}),
        _step(24, ActionType.VISUAL_CLICK, params={"description": "open posting"}),
    ])
    assert memory.detect_loop() is False


def test_navigation_between_pages_is_not_a_loop():
    memory = _memory([
        _step(1, ActionType.DOM_EXTRACT, params={"selector": ".a"}, url="https://example.com/1", screenshot_hash="h1"),
        _step(2, ActionType.READ_LARGE_NOTE, params={"note_id": "ln_1"}, url="https://example.com/2", screenshot_hash="h2"),
        _step(3, ActionType.DOM_EXTRACT, params={"selector": ".b"}, url="https://example.com/3", screenshot_hash="h3"),
        _step(4, ActionType.READ_LARGE_NOTE, params={"note_id": "ln_1"}, url="https://example.com/4", screenshot_hash="h4"),
    ])
    assert memory.detect_loop() is False


def test_loop_detection_needs_enough_history():
    memory = _memory([_step(1, ActionType.DOM_EXTRACT, params={"selector": ".a"})])
    assert memory.detect_loop() is False


# ── The archive catalogue reaches the prompt ─────────────────────────────

def test_large_notes_index_is_rendered_not_stubbed():
    orchestrator = Orchestrator.__new__(Orchestrator)  # no LLM clients needed
    rendered = Orchestrator._format_large_notes_index(
        orchestrator,
        {
            "large_notes_index": [
                {
                    "id": "ln_20260909_020447_00001",
                    "title": "Fireworks engineering roles",
                    "contains": "3 job titles and links",
                    "summary": "AI Product Engineer, MTS Cloud Infra, Applied ML",
                    "lines": 40,
                    "chars": 2200,
                    "source_domain": "fireworks.ai",
                }
            ]
        },
    )
    assert "ln_20260909_020447_00001" in rendered
    assert "Fireworks engineering roles" in rendered
    assert "3 job titles and links" in rendered
    assert "ListLargeNotes" not in rendered, "the placeholder must be gone"


def test_large_notes_index_is_silent_when_there_is_no_archive():
    orchestrator = Orchestrator.__new__(Orchestrator)
    assert Orchestrator._format_large_notes_index(orchestrator, {}) == ""
    assert Orchestrator._format_large_notes_index(orchestrator, None) == ""


def test_working_set_region_renders_its_entries():
    orchestrator = Orchestrator.__new__(Orchestrator)
    rendered = Orchestrator._format_working_set(
        orchestrator,
        [{"step": 7, "action_type": "ReadLargeNote", "label": "note_id=ln_1",
          "output": "salary: $200K-$290K", "truncated": False}],
    )
    assert "WORKING SET" in rendered
    assert "salary: $200K-$290K" in rendered
    assert "note_id=ln_1" in rendered
    assert Orchestrator._format_working_set(orchestrator, []) == ""


# ── Tier selection: being stuck outranks the read-only fast path ─────────

def _tier_orchestrator() -> Orchestrator:
    return Orchestrator.__new__(Orchestrator)


def test_stuck_signal_escalates_even_after_a_successful_read():
    from cosmic_types import LLMTier

    context = {
        "stuck_signal": True,
        "enable_dom_fallback": True,
        "last_action": {
            "success": True,
            "output": "rows...",
            "action_type": ActionType.DOM_EXTRACT.value,
        },
        "recent_steps": [],
    }
    assert Orchestrator._select_tier(_tier_orchestrator(), context, 0.9) == LLMTier.SLOW


def test_a_successful_read_still_stays_cheap_when_not_stuck():
    from cosmic_types import LLMTier

    context = {
        "stuck_signal": False,
        "enable_dom_fallback": True,
        "last_action": {
            "success": True,
            "output": "rows...",
            "action_type": ActionType.DOM_EXTRACT.value,
        },
        "recent_steps": [],
    }
    assert Orchestrator._select_tier(_tier_orchestrator(), context, 0.9) == LLMTier.FAST


# ── Step-budget negotiation ──────────────────────────────────────────────
# Running out of steps used to end a run mid-thought, with the same
# "incomplete" status any other non-completion produced.

import asyncio  # noqa: E402

from main import _consider_step_extension  # noqa: E402


class _FakeOrchestrator:
    """Stands in for the escalation-tier adjudicator."""

    def __init__(self, grant: bool, reason: str = "because"):
        self._grant = grant
        self.reason = reason
        self.calls = 0

    async def decide_step_extension(self, report):
        self.calls += 1
        self.report = report
        return {"grant": self._grant, "reason": self.reason}


def _progressing_memory():
    return _memory([
        _step(1, ActionType.DOM_EXTRACT, output="rows", params={"selector": ".a"}),
        _step(2, ActionType.VISUAL_CLICK, params={"description": "open posting"}),
    ])


def _circling_memory():
    return _memory([
        _step(1, ActionType.READ_LARGE_NOTE, params={"note_id": "ln_1"}),
        _step(2, ActionType.DOM_EXTRACT, params={"selector": ".jobs"}),
        _step(3, ActionType.READ_LARGE_NOTE, params={"note_id": "ln_1"}),
        _step(4, ActionType.DOM_EXTRACT, params={"selector": ".jobs"}),
    ])


def _extend(memory, orchestrator, **overrides):
    kwargs = dict(
        orchestrator=orchestrator,
        memory=memory,
        goal="find 3 jobs and salaries",
        step_ceiling=30,
        extension_size=10,
        extension_limit=2,
        hard_step_cap=60,
        decision_log=[],
        elapsed_sec=100.0,
        time_budget_sec=840.0,
        browser_state=memory.steps[-1].browser_state if memory.steps else None,
    )
    kwargs.update(overrides)
    log = kwargs["decision_log"]
    result = asyncio.run(_consider_step_extension(**kwargs))
    return result, log


def test_extension_is_off_by_default():
    orch = _FakeOrchestrator(grant=True)
    result, log = _extend(_progressing_memory(), orch, extension_size=0, extension_limit=0)
    assert result is None
    assert orch.calls == 0, "the adjudicator must not be consulted when the policy is off"
    assert log == [], "a disabled feature should not log refusals"


def test_a_progressing_run_can_be_granted_more_room():
    orch = _FakeOrchestrator(grant=True, reason="two clicks from the salaries")
    result, log = _extend(_progressing_memory(), orch)
    assert result == 40
    assert log[0]["granted"] is True
    assert log[0]["steps_added"] == 10
    assert log[0]["reason"] == "two clicks from the salaries"


def test_a_circling_run_is_refused_without_asking():
    # The guard that matters: an extension must never buy more of a loop.
    orch = _FakeOrchestrator(grant=True)
    result, log = _extend(_circling_memory(), orch)
    assert result is None
    assert orch.calls == 0, "no verified progress means the model is never consulted"
    assert "no verified progress" in log[0]["reason"]


def test_the_adjudicator_can_still_say_no():
    orch = _FakeOrchestrator(grant=False, reason="what it has is already enough")
    result, log = _extend(_progressing_memory(), orch)
    assert result is None
    assert orch.calls == 1
    assert log[0]["granted"] is False
    assert log[0]["reason"] == "what it has is already enough"


def test_extensions_stop_at_the_limit():
    orch = _FakeOrchestrator(grant=True)
    spent = [{"granted": True}, {"granted": True}]
    result, log = _extend(_progressing_memory(), orch, decision_log=spent, extension_limit=2)
    assert result is None
    assert orch.calls == 0
    assert "extension limit reached" in log[-1]["reason"]


def test_the_hard_ceiling_wins():
    orch = _FakeOrchestrator(grant=True)
    result, log = _extend(_progressing_memory(), orch, step_ceiling=60, hard_step_cap=60)
    assert result is None
    assert orch.calls == 0
    assert "hard ceiling" in log[0]["reason"]


def test_a_grant_is_clipped_to_the_hard_ceiling():
    orch = _FakeOrchestrator(grant=True)
    result, log = _extend(_progressing_memory(), orch, step_ceiling=55, hard_step_cap=60)
    assert result == 60
    assert log[0]["steps_added"] == 5


def test_no_extension_when_there_is_no_time_to_spend_it():
    orch = _FakeOrchestrator(grant=True)
    result, log = _extend(_progressing_memory(), orch, elapsed_sec=800.0, time_budget_sec=840.0)
    assert result is None
    assert orch.calls == 0
    assert "time budget" in log[0]["reason"]


def test_an_unavailable_adjudicator_stops_the_run_rather_than_guessing():
    # decide_step_extension catches its own failures and returns grant=False,
    # so a model outage degrades to the old behaviour: stop with what you have.
    class Unavailable:
        calls = 0

        async def decide_step_extension(self, report):
            return {"grant": False, "reason": "adjudicator unavailable (RuntimeError)"}

    result, log = _extend(_progressing_memory(), Unavailable())
    assert result is None
    assert log[0]["granted"] is False
    assert "unavailable" in log[0]["reason"]
