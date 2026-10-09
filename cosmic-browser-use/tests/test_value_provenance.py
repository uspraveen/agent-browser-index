"""Type-value guard: invented personal values never get typed.

Pins the 2026-10-08 Partiful failures: a made-up LinkedIn URL / company / job
title (refused every time — nothing in the goal, notes or answers grounds
them), the user's email typed into "Your Name" (grounded, but a misfit — held
back once), and the guard's own safety: grounded values never cost a model
call, and a missing or failing decider never blocks a type.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cosmic_types import ActionResult, ActionType, ToolCall  # noqa: E402
from value_provenance import TypeValueGuard, user_answers, value_is_grounded  # noqa: E402

GOAL = (
    "Complete the RSVP. In the message box enter exactly: \"Hi — I'm Cosmic, Praveen's Life OS.\" "
    "If the form asks for an email address, use uspraveenraj@gmail.com. More: cosmic.thelearnchain.com"
)


class _Decider:
    def __init__(self, personal=0.0, fits=1.0, fail=False):
        self.personal, self.fits, self.fail = personal, fits, fail
        self.calls = []

    async def judge(self, state, questions):
        self.calls.append((state, questions))
        if self.fail:
            return None
        out = {"personal": {"noul": self.personal}}
        if "fits" in questions:
            out["fits"] = {"noul": self.fits}
        return out


def _type(text, ref="@e45"):
    return ToolCall(action_type=ActionType.SNAPSHOT_TYPE, parameters={"ref": ref, "text": text})


def _check(guard, tool_call, refs=None, answers=()):
    return asyncio.run(guard.check(
        tool_call=tool_call, goal=GOAL, notes=[], answers=list(answers),
        refs=refs or {"@e45": {"name": "What is your LinkedIn? *"}, "@e39": {"name": "Your Name"}},
    ))


def test_grounding_is_case_whitespace_and_url_tolerant():
    assert value_is_grounded("USPRAVEENRAJ@gmail.com", [GOAL])
    assert value_is_grounded("Hi — I'm  Cosmic, Praveen's Life OS.", [GOAL])
    assert value_is_grounded("https://www.cosmic.thelearnchain.com/", [GOAL])
    assert value_is_grounded("No", [])  # a choice, not an inventable fact
    assert not value_is_grounded("https://www.linkedin.com/in/praveenrajs/", [GOAL])


def test_invented_profile_url_is_refused_every_time():
    decider = _Decider(personal=0.97)
    guard = TypeValueGuard(decider)
    tool = _type("https://www.linkedin.com/in/praveenrajs/")
    first = _check(guard, tool)
    assert first and "Refused" in first and "AskUser" in first
    assert _check(guard, tool) == first  # insisting does not help
    assert guard.stats["refused_personal"] == 2


def test_a_value_the_user_gave_passes_without_asking_the_model():
    decider = _Decider(personal=0.97)
    guard = TypeValueGuard(decider)
    answers = user_answers([SimpleNamespace(action=ActionResult(
        success=True, action_type=ActionType.ASK_USER, description="Asked: LinkedIn?",
        output="User Answer: linkedin.com/in/uspraveen",
    ))])
    assert _check(guard, _type("https://www.linkedin.com/in/uspraveen"), answers=answers) is None
    assert decider.calls == []


def test_misfit_is_held_once_then_allowed():
    decider = _Decider(personal=0.0, fits=0.02)
    guard = TypeValueGuard(decider)
    # Ungrounded on purpose so the decider is asked; label is "Your Name".
    tool = _type("praveen@example.org", ref="@e39")
    first = _check(guard, tool)
    assert first and "Held back" in first and "Your Name" in first
    assert _check(guard, tool) is None


def test_non_personal_composed_text_passes():
    guard = TypeValueGuard(_Decider(personal=0.05, fits=0.9))
    assert _check(guard, _type("engineering jobs berlin", ref="@e45")) is None


def test_no_decider_or_failing_decider_never_blocks():
    assert _check(TypeValueGuard(None), _type("https://www.linkedin.com/in/x/")) is None
    failing = TypeValueGuard(_Decider(fail=True))
    assert _check(failing, _type("https://www.linkedin.com/in/x/")) is None
    assert failing.stats["judge_failures"] == 1


def test_only_type_actions_are_checked():
    guard = TypeValueGuard(_Decider(personal=1.0))
    click = ToolCall(action_type=ActionType.SNAPSHOT_CLICK, parameters={"ref": "@e3"})
    assert _check(guard, click) is None
    assert guard.stats["checked"] == 0


def test_unlabeled_field_skips_the_fit_question():
    decider = _Decider(personal=0.0)
    guard = TypeValueGuard(decider)
    _check(guard, _type("something new", ref="@e99"), refs={"@e99": {"name": ""}})
    assert "fits" not in decider.calls[0][1]
