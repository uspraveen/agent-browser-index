#!/usr/bin/env python3
"""
Type-value guard: the harness-side half of the Personal-Data Integrity Rule.

The planner prompt already says "fill personal fields only from values the
user or the task provided; never invent them". On 2026-10-08 the base model
typed a made-up LinkedIn URL, company and job title into a hackathon RSVP
anyway. A prompt rule the model can ignore is not a rule, so every type
action now passes this check before it runs:

1. Grounded values pass untouched. A value is grounded when it appears in the
   goal, the run's saved notes, or an answer the user gave this run. Most
   typing (the message the user dictated, an email they named) is grounded and
   costs nothing here.
2. An ungrounded value goes to the decision model (the same calibrated
   decider the fast path uses — no keyword lists) with two questions:
   - is this a fact about the person the form is for (identity, contact,
     profiles, employer, school, history)? If so it was invented, and the
     type is refused every time, with a pointer to AskUser.
   - does the value fit the field at all (an email in a "Your Name" field)?
     A misfit is refused once; if the model re-issues the identical type
     after reading the refusal, it goes through — the field may be mislabeled.
3. No decider, or the decider failing, never blocks typing: the guard is an
   extra check, not a new way for runs to die.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional

from cosmic_types import ActionType

TYPE_ACTIONS = {ActionType.SNAPSHOT_TYPE, ActionType.DOM_TYPE, ActionType.VISUAL_TYPE}

PERSONAL_THRESHOLD = 0.6
MISFIT_THRESHOLD = 0.15

_SPACE = re.compile(r"\s+")
_URL_PREFIX = re.compile(r"^(https?://)?(www\.)?", re.I)


def _norm(text: Any) -> str:
    return _SPACE.sub(" ", str(text or "")).strip().lower()


def _url_core(text: str) -> str:
    return _URL_PREFIX.sub("", text).rstrip("/")


def value_is_grounded(value: str, sources: Iterable[str]) -> bool:
    """Does `value` appear (whitespace/case-insensitively) in any source?

    Tiny values ("1", "No", "US") are treated as grounded: they are answers
    to choices, not facts that can be invented. URLs also match without their
    scheme/www/trailing slash, so `linkedin.com/in/x` grounds
    `https://www.linkedin.com/in/x/`."""
    v = _norm(value)
    if len(v) <= 3:
        return True
    haystack = " \n ".join(_norm(s) for s in sources if s)
    if not haystack:
        return False
    if v in haystack:
        return True
    core = _url_core(v)
    return len(core) > 3 and core != v and core in haystack


def user_answers(steps: Iterable[Any]) -> List[str]:
    """Every answer the user gave this run (AskUser outputs)."""
    answers: List[str] = []
    for step in steps or []:
        action = getattr(step, "action", None)
        if action is None or getattr(action, "action_type", None) != ActionType.ASK_USER:
            continue
        if not getattr(action, "success", False):
            continue
        out = str(getattr(action, "output", "") or "")
        if out:
            answers.append(out.split("User Answer:", 1)[-1])
    return answers


def type_target(tool_call: Any, refs: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """(value, field label) of a type action, best effort."""
    params = getattr(tool_call, "parameters", None) or {}
    value = str(params.get("text") or "")
    label = ""
    action = getattr(tool_call, "action_type", None)
    if action == ActionType.SNAPSHOT_TYPE:
        info = (refs or {}).get(str(params.get("ref") or "").strip()) or {}
        label = str(info.get("name") or "")
    elif action == ActionType.DOM_TYPE:
        label = str(params.get("selector") or "")
    elif action == ActionType.VISUAL_TYPE:
        label = str(params.get("description") or params.get("field_description") or "")
    return {"value": value, "label": label}


class TypeValueGuard:
    """Holds the refuse-once memory for misfits across one run."""

    def __init__(self, decider: Any = None):
        self.decider = decider
        self._misfit_warned: set = set()
        self.stats = {"checked": 0, "grounded": 0, "refused_personal": 0, "refused_misfit": 0, "judge_failures": 0}

    async def check(
        self,
        *,
        tool_call: Any,
        goal: str,
        notes: Iterable[str] = (),
        answers: Iterable[str] = (),
        refs: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """None to let the type run, or the refusal the model reads."""
        if getattr(tool_call, "action_type", None) not in TYPE_ACTIONS:
            return None
        target = type_target(tool_call, refs)
        value, label = target["value"], target["label"]
        if not value.strip():
            return None
        self.stats["checked"] += 1
        if value_is_grounded(value, [goal, *notes, *answers]):
            self.stats["grounded"] += 1
            return None
        judge = getattr(self.decider, "judge", None)
        if judge is None:
            return None
        field = label.strip() or "(unlabeled field)"
        questions = {
            "personal": {
                "type": "noul",
                "instructions": (
                    f"An agent filling a form is about to type {value!r} into the field {field!r}. "
                    "Is this value a fact about the specific person the form is for — their name, "
                    "contact details, an online profile or social URL, employer, job title, school, "
                    "graduation year, demographics or personal history — something only that person "
                    "can supply?"
                ),
            },
        }
        if label.strip():
            questions["fits"] = {
                "type": "noul",
                "instructions": (
                    f"Is {value!r} a sensible thing to enter in a form field labeled {field!r}? "
                    "(For example, an email address is not a sensible entry for a name field.)"
                ),
                # OpenAI's decider (the fallback) rated an email in a "First
                # name" field a fit at 1.00 under the wording above; asked the
                # other way round it caught every misfit in the 2026-10-09
                # bake-off. `invert` turns its answer back into "fits".
                "fallback": {
                    "invert": True,
                    "instructions": (
                        f"The value {value!r} is the wrong kind of data for a form field labeled "
                        f"{field!r} — for example an email address typed into a name or ZIP-code "
                        "field, a person's name typed into an email field, or free text typed into "
                        "a phone or card-number field."
                    ),
                },
            }
        try:
            answers_out = await judge({"goal": goal[:2000], "field": field, "value": value[:500]}, questions)
        except Exception:
            answers_out = None
        if not isinstance(answers_out, dict):
            self.stats["judge_failures"] += 1
            return None
        personal = _noul(answers_out.get("personal"))
        if personal is not None and personal >= PERSONAL_THRESHOLD:
            self.stats["refused_personal"] += 1
            return (
                f"Refused to type {value[:80]!r} into {field[:80]!r}: that is a fact about the user "
                "(identity, contact, profile, employer, school or history) and it appears nowhere in "
                "the goal, your saved notes or the user's answers, so it would be invented. Ask for "
                "it with AskUser (one question can cover every missing field), or leave the field "
                "empty and report it as missing. Never guess personal details."
            )
        fits = _noul(answers_out.get("fits"))
        key = (field, value)
        if fits is not None and fits < MISFIT_THRESHOLD and key not in self._misfit_warned:
            self._misfit_warned.add(key)
            self.stats["refused_misfit"] += 1
            return (
                f"Held back typing {value[:80]!r} into {field[:80]!r}: that value does not look like "
                "an answer to this field, so this is probably the wrong field. Re-check which field "
                "asks for it (the screenshot shows each field's label). If this really is the right "
                "field, issue the same action again and it will go through."
            )
        return None


def _noul(answer: Any) -> Optional[float]:
    try:
        value = float(answer.get("noul"))
    except (AttributeError, TypeError, ValueError):
        return None
    if value != value:  # NaN
        return None
    return min(max(value, 0.0), 1.0)
