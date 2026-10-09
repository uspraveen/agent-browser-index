"""Offline tests for decisions_client: retries on the primary decider, the
OpenAI Decisions fallback, and the request/answer translation between them.

Pins the 2026-10-09 decisions: Perplexity stays primary (it won the bake-off)
but allows 10 requests/second per organization, so a busy Perplexity is
retried and then answered by OpenAI. OpenAI rejects one-option choices,
wants string instructions and returns rounded answer lists; none of that may
leak to callers. No network anywhere (httpx.MockTransport).

Run with:  python -m pytest tests/test_decisions_client.py -q
"""
from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

import decisions_client  # noqa: E402
from decisions_client import (  # noqa: E402
    openai_answers,
    openai_request,
    post_decisions,
    primary_body,
)

PPLX = "https://api.perplexity.ai/v1/decisions"
OPENAI = "https://api.openai.com/v1/decisions"


@dataclass
class _Config:
    api_url: str = PPLX
    api_key: str = "pplx-test"
    provider: str = "pplx"
    attempts: int = 3
    retry_statuses: tuple = (429, 500, 502, 503, 529)
    retry_max_wait_ms: int = 1000
    fallback_api_key: str = "sk-test"
    fallback_url: str = OPENAI
    fallback_model: str = "gpt-6-luna"
    fallback_timeout_ms: int = 8000
    extra: dict = field(default_factory=dict)


BODY = {
    "model": "pplx-decider-v1.1-27b",
    "state": ['{"page": "rsvp"}', {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/AAAA"}}],
    "questions": {
        "operation": {
            "type": "choice",
            "criteria": {"CLICK": "Click something.", "TYPE_TEXT": "Type something."},
            "instructions": {"goal": "RSVP", "rules": ["rule one", "rule two"]},
        },
        "type_text_target": {"type": "choice", "criteria": {"@e1": {"element": "[@e1] Your Name"}}, "instructions": "pick"},
        "goal_progress": {"type": "score", "criteria": ["Nothing", "Half", "Done"], "instructions": "progress?"},
        "fits": {
            "type": "noul",
            "instructions": "Is it a fit?",
            "fallback": {"invert": True, "instructions": "It is the wrong kind of data."},
        },
    },
}


def _openai_reply(request_json, probability=0.9):
    answers = []
    for q in request_json["questions"]:
        if q["type"] == "choice":
            values = [c["value"] for c in q["choices"]]
            answers.append({"type": "choice", "name": q["name"], "choice": values[0], "confidence": 0.8,
                            "probabilities": [{"value": v, "probability": 0.67 if i == 0 else 0.34} for i, v in enumerate(values)]})
        elif q["type"] == "score":
            answers.append({"type": "score", "name": q["name"], "score": 1.0, "confidence": 0.7,
                            "probabilities": [{"value": i, "label": l["label"], "probability": 1 / 3} for i, l in enumerate(q["levels"])]})
        else:
            answers.append({"type": "predicate", "name": q["name"], "probability": probability})
    return {"model": "gpt-6-luna", "answers": answers, "usage": {"input_tokens": 10}}


PPLX_OK = {"model": "pplx-decider-v1.1-27b", "answers": {"operation": {"type": "choice", "choice": "CLICK"}}}


def _transport(pplx_statuses, openai_status=200, calls=None, pplx_exc=None):
    """pplx_statuses: statuses for successive primary calls (last one repeats)."""
    calls = calls if calls is not None else []

    def handler(request):
        url = str(request.url)
        calls.append((url, json.loads(request.content)))
        if url == PPLX:
            if pplx_exc is not None:
                raise pplx_exc
            n = sum(1 for u, _ in calls if u == PPLX)
            status = pplx_statuses[min(n - 1, len(pplx_statuses) - 1)]
            if status == 200:
                return httpx.Response(200, json=PPLX_OK)
            return httpx.Response(status, headers={"retry-after": "1"}, text="busy")
        if openai_status != 200:
            return httpx.Response(openai_status, text="nope")
        return httpx.Response(200, json=_openai_reply(json.loads(request.content)))

    return httpx.AsyncClient(transport=httpx.MockTransport(handler)), calls


@pytest.fixture(autouse=True)
def _no_sleep_no_cooldown(monkeypatch):
    waits = []

    async def fake_sleep(seconds):
        waits.append(seconds)

    monkeypatch.setattr(decisions_client.asyncio, "sleep", fake_sleep)
    decisions_client._COOLDOWN_UNTIL.clear()
    yield waits
    decisions_client._COOLDOWN_UNTIL.clear()


def _post(client, config=None, body=BODY):
    return asyncio.run(post_decisions(client, config or _Config(), body))


# --------------------------------------------------------------------- #
# Retries and fallback

def test_busy_primary_is_retried_and_answers_without_fallback(_no_sleep_no_cooldown):
    client, calls = _transport([429, 503, 200])
    data, meta = _post(client)
    assert data == PPLX_OK
    assert (meta["provider"], meta["attempts"], meta["retries"], meta["fallback"]) == ("pplx", 3, 2, False)
    assert [u for u, _ in calls] == [PPLX] * 3
    # Retry-After was 1s; every wait is bounded by retry_max_wait_ms (+ jitter).
    assert len(_no_sleep_no_cooldown) == 2 and all(0.1 <= w <= 1.1 for w in _no_sleep_no_cooldown)


def test_exhausted_primary_falls_back_to_openai_in_the_primary_shape():
    client, calls = _transport([429])
    data, meta = _post(client)
    assert [u for u, _ in calls] == [PPLX, PPLX, PPLX, OPENAI]
    assert (meta["provider"], meta["fallback"], meta["error"]) == ("openai", True, "HTTP 429")
    answers = data["answers"]
    assert answers["operation"]["choice"] == "CLICK"
    assert abs(sum(answers["operation"]["probabilities"].values()) - 1) < 1e-9  # rescaled from 0.67 + 0.34
    assert answers["type_text_target"] == {"type": "choice", "choice": "@e1", "confidence": 1.0, "probabilities": {"@e1": 1.0}}
    assert answers["fits"] == {"type": "noul", "noul": pytest.approx(0.1)}  # inverted 0.9
    assert answers["goal_progress"]["score"] == 1.0


def test_the_primary_never_sees_fallback_overrides():
    client, calls = _transport([200])
    _post(client)
    sent = calls[0][1]
    assert "fallback" not in sent["questions"]["fits"]
    assert sent["questions"]["fits"]["instructions"] == "Is it a fit?"
    assert "fallback" in BODY["questions"]["fits"]  # the caller's body is not mutated


def test_a_timeout_goes_straight_to_the_fallback():
    client, calls = _transport([200], pplx_exc=httpx.ReadTimeout("slow"))
    data, meta = _post(client)
    assert [u for u, _ in calls] == [PPLX, OPENAI]
    assert meta["error"] == "timeout" and meta["provider"] == "openai"


def test_a_non_retryable_status_is_not_retried():
    client, calls = _transport([400])
    _data, meta = _post(client)
    assert [u for u, _ in calls] == [PPLX, OPENAI]
    assert meta["retries"] == 0


def test_without_a_fallback_key_the_failure_is_returned():
    client, calls = _transport([429])
    data, meta = _post(client, _Config(fallback_api_key=""))
    assert data is None and meta["error"] == "HTTP 429"
    assert [u for u, _ in calls] == [PPLX] * 3
    assert decisions_client._COOLDOWN_UNTIL == {}  # nowhere else to go, so no cool-down


def test_both_providers_down_returns_none_with_both_reasons():
    client, _calls = _transport([503], openai_status=500)
    data, meta = _post(client)
    assert data is None
    assert meta["error"].startswith("HTTP 503") and "fallback HTTP 500" in meta["error"]


def test_a_rate_limited_primary_cools_down_then_is_tried_again(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(decisions_client.time, "monotonic", lambda: clock[0])
    client, calls = _transport([429, 429, 429, 200])
    _post(client)
    assert [u for u, _ in calls] == [PPLX, PPLX, PPLX, OPENAI]
    data, meta = _post(client)  # inside the cool-down: no primary call at all
    assert [u for u, _ in calls][4:] == [OPENAI]
    assert meta["attempts"] == 0 and meta["provider"] == "openai"
    clock[0] += decisions_client.COOLDOWN_MAX_SEC + 0.1
    data, meta = _post(client)
    assert data == PPLX_OK and meta["provider"] == "pplx"


def test_typesafe_defaults_keep_their_old_retry_policy():
    from jev_engine import JevEngineConfig
    config = JevEngineConfig(api_key="k")
    assert (config.attempts, config.retry_statuses, config.fallback_api_key) == (2, (429, 503, 529), "")


def test_pplx_from_env_picks_up_the_openai_fallback(monkeypatch):
    from jev_engine import JevEngineConfig
    monkeypatch.setenv("PERPLEXITY_API_KEY", "pplx-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("OPENAI_DECIDER_API_KEY", raising=False)
    monkeypatch.delenv("COSMIC_DECIDER_FALLBACK", raising=False)
    config = JevEngineConfig.from_env("pplx")
    assert (config.attempts, config.fallback_api_key, config.fallback_model) == (3, "sk-test", "gpt-6-luna")
    assert 500 in config.retry_statuses
    monkeypatch.setenv("COSMIC_DECIDER_FALLBACK", "off")
    assert JevEngineConfig.from_env("pplx").fallback_api_key == ""
    assert JevEngineConfig.from_env("typesafe").fallback_api_key == ""


# --------------------------------------------------------------------- #
# Translation

def test_openai_request_translates_every_question_type():
    request, local = openai_request(BODY, "gpt-6-luna")
    assert request["model"] == "gpt-6-luna"
    content = request["input"][0]["content"]
    assert content[0] == {"type": "input_text", "text": '{"page": "rsvp"}'}
    assert content[1] == {"type": "input_image", "image_url": "data:image/jpeg;base64,/9j/AAAA"}
    by_name = {q["name"]: q for q in request["questions"]}
    assert set(by_name) == {"operation", "goal_progress", "fits"}  # one-option choice answered locally
    assert local["type_text_target"]["choice"] == "@e1"
    op = by_name["operation"]
    assert op["instructions"] == "goal: RSVP\n\nrules: rule one\n\nrule two"
    assert op["choices"] == [{"value": "CLICK", "description": "Click something."},
                             {"value": "TYPE_TEXT", "description": "Type something."}]
    assert by_name["goal_progress"]["levels"] == [{"label": "Nothing"}, {"label": "Half"}, {"label": "Done"}]
    assert by_name["fits"] == {"type": "predicate", "name": "fits", "instructions": "It is the wrong kind of data."}


def test_noul_criteria_ride_along_and_plain_state_is_json():
    body = {"state": {"goal": "g"}, "questions": {"v": {"type": "noul", "instructions": "Needs vision?",
                                                        "criteria": {"true": "pixels", "false": "table"}}}}
    request, _local = openai_request(body, "gpt-6-luna")
    assert request["input"] == '{"goal": "g"}'
    assert request["questions"][0]["instructions"] == "Needs vision?\nTrue: pixels\nFalse: table"


def test_refusals_and_unasked_answers_are_left_out():
    data = {"answers": [
        {"type": "refusal", "name": "operation"},
        {"type": "predicate", "name": "not_asked", "probability": 0.5},
        {"type": "predicate", "name": "fits", "probability": 0.2},
    ]}
    out = openai_answers(data, BODY["questions"], {})
    assert out == {"fits": {"type": "noul", "noul": pytest.approx(0.8)}}


def test_primary_body_is_untouched_without_overrides():
    body = {"questions": {"a": {"type": "noul", "instructions": "x"}}}
    assert primary_body(body) is body
