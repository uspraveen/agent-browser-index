"""Offline tests for the Jev decision engine.

The fast path decides real browser steps, so its answers must be held to the
same contract as the LLM planner: a choice only from the offered ids, a full
probability set that sums to ~1, secrets never aimed at, escalations that
hand the step back to the normal planner, and DONE that never finishes a run
on Jev's word alone. The TypeSafe API is faked with httpx.MockTransport —
no network anywhere.

Run with:  python -m pytest tests/test_jev_engine.py -q
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from cosmic_types import ActionType, LLMTier, LLMResponse, ToolCall  # noqa: E402
from jev_engine import (  # noqa: E402
    JevEngine,
    JevEngineConfig,
    build_action_space,
    build_questions,
    validate_choice,
)


# --------------------------------------------------------------------- #
# Fakes

def _entries():
    return [
        {"ref": "@e1", "tag": "input", "role": "textbox", "name": "Legal Name", "value": "", "frame_index": 0},
        {
            "ref": "@e2", "tag": "select", "role": "combobox", "name": "Country", "value": "",
            "selected": "Germany",
            "options": [{"label": "Germany", "value": "de"}, {"label": "India", "value": "in"}],
            "frame_index": 0,
        },
        {"ref": "@e3", "tag": "button", "role": "button", "name": "Search", "frame_index": 0},
    ]


class _FakeBrowser:
    def __init__(self, entries=None, total=None):
        self._entries = entries if entries is not None else _entries()
        self._total = len(self._entries) if total is None else total
        self.snapshot_calls = []

    async def _collect_snapshot(self, max_elements=80):
        self.snapshot_calls.append(max_elements)
        return {"refs": {}, "entries": self._entries, "total": self._total,
                "truncated": False, "frames": 1}

    async def _safe_evaluate(self, _js, fallback=""):
        return "visible page text"


class _FakeProvider:
    def __init__(self, text="hello", key="text"):
        self.text = text
        self.key = key
        self.calls = []

    async def generate(self, messages=None, system_prompt=None, json_mode=False):
        self.calls.append({"messages": messages, "json_mode": json_mode})
        return {"content": json.dumps({self.key: self.text}), "raw_response": {}}


class _FakeOrchestrator:
    def __init__(self, text="hello", key="text", finalizer_response=None):
        self.models = {LLMTier.FAST: _FakeProvider(text, key)}
        self._finalizer_response = finalizer_response
        self.finalizer_calls = []

    async def force_visible_answer_note(self, context=None, screenshot_base64=None):
        self.finalizer_calls.append((context, screenshot_base64))
        return self._finalizer_response


def _context(**overrides):
    ctx = {
        "goal": "Search for engineering jobs in Berlin",
        "enable_dom_fallback": True,
        "browser_state": {"url": "https://example.com/jobs", "title": "Jobs", "scroll_y": 0},
        "recent_steps": [],
    }
    ctx.update(overrides)
    return ctx


def _engine(orchestrator=None, browser=None, config_overrides=None, handler=None):
    config = JevEngineConfig(api_key="test-key", breaker_limit=3, min_confidence=0.5)
    for key, value in (config_overrides or {}).items():
        setattr(config, key, value)
    engine = JevEngine(
        orchestrator=orchestrator or _FakeOrchestrator(),
        browser=browser or _FakeBrowser(),
        config=config,
    )
    if handler is not None:
        engine.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return engine


def _valid_choice(ids, choice, conf=0.9):
    ids = set(ids)
    if len(ids) <= 1:
        return {"choice": choice, "probabilities": {choice: 1.0}, "confidence": 1.0}
    rest = (1.0 - conf) / (len(ids) - 1)
    return {
        "choice": choice,
        "probabilities": {i: (conf if i == choice else rest) for i in ids},
        "confidence": conf,
    }


def _handler_script(script):
    """script: dict with operation/click/type/select/progress/needs_vision keys."""
    def handler(request):
        body = json.loads(request.content)
        questions = body["questions"]
        answers = {
            "operation": _valid_choice(
                set(questions["operation"]["criteria"]),
                script.get("operation", "CLICK"),
                script.get("operation_conf", 0.9),
            ),
            "goal_progress": {"score": script.get("progress", 0.4), "probabilities": {}, "confidence": 1.0},
            "needs_vision": {"noul": script.get("needs_vision", 0.0), "probabilities": {}, "confidence": 1.0},
        }
        if "click_target" in questions:
            ids = set(questions["click_target"]["criteria"])
            answers["click_target"] = _valid_choice(ids, script.get("click", "@e3"))
        if "type_text_target" in questions:
            ids = set(questions["type_text_target"]["criteria"])
            answers["type_text_target"] = _valid_choice(ids, script.get("type", "@e1"))
        if "select_target" in questions:
            ids = set(questions["select_target"]["criteria"])
            answers["select_target"] = _valid_choice(ids, script.get("select", "@e2:2"))
        return httpx.Response(200, json={"answers": answers, "model": "jev-test", "usage": {}})

    return handler


def _decide(engine, context=None):
    return asyncio.run(engine.decide_step(context or _context(), screenshot_b64="data:image/jpeg;base64,x"))


def _recording_handler(script):
    """Handler that also captures every request body for assertions."""
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return _handler_script(script)(request)

    return handler, bodies


# --------------------------------------------------------------------- #
# Action space

class TestActionSpace:
    def test_every_non_secret_element_is_clickable(self):
        space = build_action_space(_entries())
        assert set(space["click_targets"]) == {"@e1", "@e2", "@e3"}

    def test_type_targets_are_editable_fields_only(self):
        space = build_action_space(_entries())
        assert set(space["type_targets"]) == {"@e1"}

    def test_select_options_become_individual_targets(self):
        space = build_action_space(_entries())
        assert set(space["select_targets"]) == {"@e2:1", "@e2:2"}
        assert space["select_targets"]["@e2:2"]["option_label"] == "India"
        assert space["select_targets"]["@e2:2"]["entry"]["ref"] == "@e2"

    def test_secret_fields_are_excluded_everywhere(self):
        entries = _entries() + [
            {"ref": "@e4", "tag": "input", "role": "textbox", "name": "Password",
             "value": "********", "secret": True, "frame_index": 0},
        ]
        space = build_action_space(entries)
        assert "@e4" not in space["click_targets"]
        assert "@e4" not in space["type_targets"]
        assert all(item["index"] != "@e4" for item in space["elements"])

    def test_native_select_is_not_a_type_target(self):
        entries = [{"ref": "@e1", "tag": "select", "role": "combobox", "name": "Country",
                    "options": [{"label": "A", "value": "a"}], "frame_index": 0}]
        space = build_action_space(entries)
        assert "@e1" not in space["type_targets"]
        assert space["elements"][0]["operations"] == ["CLICK", "SELECT"]

    def test_one_element_keeps_one_index_across_operations(self):
        entries = [{"ref": "@e1", "tag": "input", "role": "combobox", "name": "City", "frame_index": 0}]
        space = build_action_space(entries)
        assert len(space["elements"]) == 1
        assert space["elements"][0]["operations"] == ["CLICK", "TYPE_TEXT"]

    def test_questions_only_offer_supported_heads(self):
        space = build_action_space(_entries())
        questions = build_questions("goal", space, allow_scroll_up=True)
        assert set(questions) == {
            "operation", "click_target", "type_text_target", "select_target",
            "goal_progress", "needs_vision",
        }
        assert "SCROLL_UP" in questions["operation"]["criteria"]

        bare = build_action_space([_entries()[2]])  # one button
        questions = build_questions("goal", bare, allow_scroll_up=False)
        assert set(questions) == {"operation", "click_target", "goal_progress", "needs_vision"}
        assert "SCROLL_UP" not in questions["operation"]["criteria"]

    def test_escalation_operations_are_always_offered(self):
        questions = build_questions("goal", build_action_space(_entries()), allow_scroll_up=False)
        criteria = questions["operation"]["criteria"]
        assert "ESCALATE_VISION" in criteria
        assert "ESCALATE_LLM" in criteria
        assert "BLOCKED" in criteria
        assert "GO_BACK" in criteria
        assert "RELOAD" in criteria


# --------------------------------------------------------------------- #
# Answer validation

class TestValidateChoice:
    def test_valid_answer_passes(self):
        answer = _valid_choice({"a", "b", "c"}, "b")
        assert validate_choice(answer, {"a", "b", "c"}) == answer

    def test_unknown_choice_rejected(self):
        assert validate_choice(_valid_choice({"a", "b"}, "a", conf=0.6), {"a", "b", "c"}) is None

    def test_missing_probability_ids_rejected(self):
        answer = {"choice": "a", "probabilities": {"a": 1.0}, "confidence": 0.9}
        assert validate_choice(answer, {"a", "b"}) is None

    def test_bad_sum_rejected(self):
        answer = {"choice": "a", "probabilities": {"a": 0.5, "b": 0.2}, "confidence": 0.9}
        assert validate_choice(answer, {"a", "b"}) is None

    def test_choice_must_be_the_argmax(self):
        answer = {"choice": "b", "probabilities": {"a": 0.8, "b": 0.2}, "confidence": 0.9}
        assert validate_choice(answer, {"a", "b"}) is None

    def test_non_finite_confidence_rejected(self):
        answer = {"choice": "a", "probabilities": {"a": 1.0}, "confidence": float("nan")}
        assert validate_choice(answer, {"a"}) is None

    def test_malformed_answer_rejected(self):
        assert validate_choice(None, {"a"}) is None
        assert validate_choice({}, {"a"}) is None


# --------------------------------------------------------------------- #
# Decision mapping

class TestDecideStep:
    def test_click_maps_to_snapshot_click(self):
        engine = _engine(handler=_handler_script({"operation": "CLICK", "click": "@e3"}))
        response = _decide(engine)
        assert isinstance(response, LLMResponse)
        assert response.tier_used == "jev"
        assert response.tool_call.action_type == ActionType.SNAPSHOT_CLICK
        assert response.tool_call.parameters == {"ref": "@e3"}
        assert response.confidence > engine.config.min_confidence
        assert response.estimated_completion == 0.4
        assert engine.stats["decisions"] == 1

    def test_type_text_uses_the_text_helper(self):
        orchestrator = _FakeOrchestrator(text="platform engineer")
        engine = _engine(orchestrator=orchestrator, handler=_handler_script({"operation": "TYPE_TEXT"}))
        response = _decide(engine)
        assert response.tool_call.action_type == ActionType.SNAPSHOT_TYPE
        assert response.tool_call.parameters == {"ref": "@e1", "text": "platform engineer"}
        # The helper gets json_mode requested and only the field context.
        provider = orchestrator.models[LLMTier.FAST]
        assert provider.calls and provider.calls[0]["json_mode"] is True
        field = json.loads(provider.calls[0]["messages"][0]["content"])["field"]
        assert field["label"] == "Legal Name"

    def test_type_text_helper_decline_falls_through(self):
        engine = _engine(
            orchestrator=_FakeOrchestrator(text=None),
            handler=_handler_script({"operation": "TYPE_TEXT"}),
        )
        assert _decide(engine) is None
        assert engine.stats["fallthroughs"] == 1

    def test_select_maps_to_option_label(self):
        engine = _engine(handler=_handler_script({"operation": "SELECT", "select": "@e2:2"}))
        response = _decide(engine)
        assert response.tool_call.action_type == ActionType.SNAPSHOT_SELECT
        assert response.tool_call.parameters == {"ref": "@e2", "label": "India"}

    def test_scroll_and_wait_map_to_visual_tools(self):
        engine = _engine(handler=_handler_script({"operation": "SCROLL_DOWN"}))
        response = _decide(engine)
        assert response.tool_call.action_type == ActionType.VISUAL_SCROLL
        assert response.tool_call.parameters["direction"] == "down"

        engine = _engine(handler=_handler_script({"operation": "WAIT"}))
        response = _decide(engine)
        assert response.tool_call.action_type == ActionType.VISUAL_WAIT

    def test_go_back_and_reload_map_to_navigation_tools(self):
        engine = _engine(handler=_handler_script({"operation": "GO_BACK"}))
        response = _decide(engine)
        assert response.tool_call.action_type == ActionType.GO_BACK
        assert response.tool_call.parameters == {}
        assert response.tier_used == "jev"

        engine = _engine(handler=_handler_script({"operation": "RELOAD"}))
        response = _decide(engine)
        assert response.tool_call.action_type == ActionType.RELOAD
        assert response.tool_call.parameters == {}

    def test_done_goes_through_the_finalizer_never_jev_tier(self):
        finalizer_response = LLMResponse(
            tool_call=ToolCall(action_type=ActionType.SAVE_NOTE, parameters={"note": "final answer"}),
            tier_used="fast",
        )
        orchestrator = _FakeOrchestrator(finalizer_response=finalizer_response)
        engine = _engine(orchestrator=orchestrator, handler=_handler_script({"operation": "DONE"}))
        response = _decide(engine)
        assert response is finalizer_response
        assert len(orchestrator.finalizer_calls) == 1
        assert engine.stats["done_attempts"] == 1
        # A DONE is not a fast-path action: the loop counts it by tier.
        assert response.tier_used != "jev"

    def test_done_without_finalizer_evidence_falls_through(self):
        engine = _engine(
            orchestrator=_FakeOrchestrator(finalizer_response=None),
            handler=_handler_script({"operation": "DONE"}),
        )
        assert _decide(engine) is None

    def test_blocked_and_llm_escalation_fall_through(self):
        for operation in ("BLOCKED", "ESCALATE_LLM"):
            engine = _engine(handler=_handler_script({"operation": operation}))
            assert _decide(engine) is None, operation
            assert engine.stats["llm_escalations"] == 1, operation

    def test_vision_escalation_sets_the_one_step_hint(self):
        engine = _engine(handler=_handler_script({"operation": "ESCALATE_VISION"}))
        context = _context()
        assert _decide(engine, context) is None
        assert "prefer_vision_hint" in context
        assert engine.stats["vision_escalations"] == 1

    def test_needs_vision_judgment_overrides_the_operation(self):
        engine = _engine(handler=_handler_script({"operation": "CLICK", "needs_vision": 0.9}))
        context = _context()
        assert _decide(engine, context) is None
        assert "prefer_vision_hint" in context

    def test_low_needs_vision_does_not_block_a_click(self):
        engine = _engine(handler=_handler_script({"operation": "CLICK", "needs_vision": 0.2}))
        response = _decide(engine)
        assert response is not None

    def test_low_confidence_falls_through(self):
        engine = _engine(handler=_handler_script({"operation": "CLICK", "operation_conf": 0.3}))
        assert _decide(engine) is None
        assert engine.stats["fallthroughs"] == 1

    def test_sparse_table_requests_vision(self):
        engine = _engine(browser=_FakeBrowser(total=2), handler=_handler_script({}))
        context = _context()
        assert _decide(engine, context) is None
        assert "prefer_vision_hint" in context
        # The TypeSafe API was never called.
        assert engine.stats["api_failures"] == 0

    def test_max_elements_bounds_the_snapshot(self):
        engine = _engine(config_overrides={"max_elements": 42}, handler=_handler_script({}))
        _decide(engine)
        assert engine.browser.snapshot_calls == [42]


# --------------------------------------------------------------------- #
# Gates and failure handling

class TestGates:
    def test_no_api_key_means_no_decision(self):
        engine = _engine(handler=_handler_script({}))
        engine.config.api_key = ""
        assert _decide(engine) is None

    def test_vision_mode_is_never_a_jev_step(self):
        engine = _engine(handler=_handler_script({}))
        assert _decide(engine, _context(enable_dom_fallback=False)) is None

    def test_stuck_signal_defers_to_the_frontier_brain(self):
        engine = _engine(handler=_handler_script({}))
        assert _decide(engine, _context(stuck_signal=True)) is None

    def test_transport_failures_fall_through_and_trip_the_breaker(self):
        def handler(request):
            return httpx.Response(500)

        engine = _engine(handler=handler)
        for _ in range(3):
            assert _decide(engine) is None
        assert not engine.available
        assert engine.stats["api_failures"] == 3
        # Tripped breaker short-circuits before even taking a snapshot.
        before = len(engine.browser.snapshot_calls)
        assert _decide(engine) is None
        assert len(engine.browser.snapshot_calls) == before

    def test_invalid_operation_answer_is_a_failure(self):
        def handler(request):
            return httpx.Response(200, json={"answers": {"operation": {"choice": "CLICK"}}, "model": "jev-test"})

        engine = _engine(handler=handler)
        assert _decide(engine) is None
        assert engine.stats["api_failures"] == 1

    def test_exception_inside_the_engine_never_fails_the_step(self):
        class _BrokenBrowser:
            async def _collect_snapshot(self, max_elements=80):
                raise RuntimeError("playwright exploded")

        engine = _engine(browser=_BrokenBrowser(), handler=_handler_script({}))
        assert _decide(engine) is None
        assert engine.stats["api_failures"] == 1


# --------------------------------------------------------------------- #
# Post-tour fixes: clamp, state, SAVE_NOTE, stand-down, guidance

class TestCompletionClamp:
    def test_jev_progress_never_reaches_the_completion_gate(self):
        engine = _engine(handler=_handler_script({"operation": "CLICK", "progress": 1.0}))
        response = _decide(engine)
        assert response.tool_call.action_type == ActionType.SNAPSHOT_CLICK
        # A per-page judgment must never end a cumulative-goal run; only the
        # DONE finalizer may finish one.
        assert response.estimated_completion == 0.9
        assert engine.last_debug["progress"] == 1.0

    def test_low_progress_is_untouched(self):
        engine = _engine(handler=_handler_script({"operation": "CLICK", "progress": 0.4}))
        assert _decide(engine).estimated_completion == 0.4


class TestStateAndBounding:
    def test_saved_notes_and_large_notes_index_reach_jev_state(self):
        handler, bodies = _recording_handler({"operation": "CLICK"})
        engine = _engine(handler=handler)
        ctx = _context()
        ctx["browser_state"]["notes"] = ["Restaurant 1: opens 09:00"]
        ctx["browser_state"]["large_notes_index"] = [
            {"note_id": "ln1", "title": "Hours", "contains": "hours", "summary": "s"}
        ]
        _decide(engine, ctx)
        state = bodies[0]["state"]
        assert state["notes"] == ["Restaurant 1: opens 09:00"]
        assert state["large_notes_index"][0]["note"] == "ln1"

    def test_recent_action_params_are_bounded(self):
        handler, bodies = _recording_handler({"operation": "CLICK"})
        engine = _engine(handler=handler)
        ctx = _context()
        ctx["recent_steps"] = [{
            "action_type": "SaveLargeNote",
            "description": "saved big extract",
            "verification_status": "success",
            "requested_parameters": {"content": "x" * 5000, "title": "t"},
        }]
        _decide(engine, ctx)
        row = bodies[0]["state"]["recent_actions"][0]
        assert len(row["params"]["content"]) == 200
        assert row["params"]["title"] == "t"
        assert len(row["detail"]) <= 200


class TestSaveNote:
    def test_save_note_composes_and_maps(self):
        orchestrator = _FakeOrchestrator(text="A Light in the Attic costs 51.77", key="note")
        engine = _engine(
            orchestrator=orchestrator,
            handler=_handler_script({"operation": "SAVE_NOTE"}),
        )
        response = _decide(engine)
        assert response.tool_call.action_type == ActionType.SAVE_NOTE
        assert response.tool_call.parameters == {"note": "A Light in the Attic costs 51.77"}
        assert response.tier_used == "jev"
        assert engine.stats["note_helper_calls"] == 1
        payload = json.loads(orchestrator.models[LLMTier.FAST].calls[0]["messages"][0]["content"])
        assert "already_recorded" in payload

    def test_save_note_helper_decline_falls_through(self):
        engine = _engine(
            orchestrator=_FakeOrchestrator(text=None, key="note"),
            handler=_handler_script({"operation": "SAVE_NOTE"}),
        )
        assert _decide(engine) is None
        assert engine.stats["fallthroughs"] == 1


class TestStandDown:
    def test_three_fallthroughs_stand_the_engine_down(self):
        handler, _bodies = _recording_handler({"operation": "ESCALATE_LLM"})
        engine = _engine(handler=handler)
        for _ in range(3):
            assert _decide(engine) is None
        assert engine.stats["standdowns"] == 1
        snapshots = len(engine.browser.snapshot_calls)
        assert _decide(engine) is None  # same page -> skipped before snapshot
        assert len(engine.browser.snapshot_calls) == snapshots
        assert "standing down" in engine.last_debug["skip"]

    def test_navigation_lifts_the_standdown(self):
        handler, _bodies = _recording_handler({"operation": "ESCALATE_LLM"})
        engine = _engine(handler=handler)
        for _ in range(3):
            _decide(engine)
        _decide(engine)  # stand-down skip on the same page
        ctx = _context()
        ctx["browser_state"]["url"] = "https://example.com/other-page"
        assert _decide(engine, ctx) is None
        assert len(engine.browser.snapshot_calls) == 4

    def test_a_verified_jev_decision_resets_the_streak(self):
        scripts = iter([
            {"operation": "ESCALATE_LLM"},
            {"operation": "ESCALATE_LLM"},
            {"operation": "CLICK", "click": "@e3"},
            {"operation": "ESCALATE_LLM"},
            {"operation": "ESCALATE_LLM"},
        ])

        def handler(request):
            return _handler_script(next(scripts))(request)

        engine = _engine(handler=handler)
        for _ in range(5):
            _decide(engine)
        assert engine.stats["decisions"] == 1
        assert engine.stats["standdowns"] == 0  # never three in a row


class TestPlannerHint:
    def test_hint_rides_the_rules_and_expires(self):
        handler, bodies = _recording_handler({"operation": "CLICK"})
        engine = _engine(handler=handler)
        engine.set_hint("set the sort dropdown before results count", steps=2)
        _decide(engine)
        _decide(engine)
        _decide(engine)
        rules = [b["questions"]["operation"]["instructions"]["rules"] for b in bodies]
        assert "set the sort dropdown before results count" in rules[0]
        assert "set the sort dropdown before results count" in rules[1]
        assert "set the sort dropdown before results count" not in rules[2]

    def test_blank_hints_are_ignored(self):
        engine = _engine(handler=_handler_script({"operation": "CLICK"}))
        engine.set_hint("   ")
        _decide(engine)
        assert engine._hint is None


class TestPlannerHintParsing:
    def test_parse_response_carries_a_bounded_hint(self):
        from orchestrator import Orchestrator, clean_fast_engine_hint

        o = Orchestrator.__new__(Orchestrator)
        raw = json.dumps({
            "action_type": "VisualClick",
            "parameters": {"description": "next"},
            "confidence": 0.9,
            "fast_engine_hint": "  avoid   the table  " + "x" * 400,
        })
        response = o._parse_response(raw)
        assert response.fast_engine_hint is not None
        assert response.fast_engine_hint.startswith("avoid the table")
        assert len(response.fast_engine_hint) <= 200
        assert clean_fast_engine_hint("") is None
        assert clean_fast_engine_hint(42) is None

    def test_parse_response_without_hint_is_none(self):
        from orchestrator import Orchestrator

        o = Orchestrator.__new__(Orchestrator)
        raw = json.dumps({
            "action_type": "VisualClick",
            "parameters": {"description": "next"},
            "confidence": 0.9,
        })
        assert o._parse_response(raw).fast_engine_hint is None
