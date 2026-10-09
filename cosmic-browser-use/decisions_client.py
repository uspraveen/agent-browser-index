#!/usr/bin/env python3
"""
One Decisions API call: the configured decider first, with retries, then
OpenAI's Decisions API as the fallback.

Perplexity's pplx-decider is the primary. It won the 2026-10-09 bake-off on
hand-labeled Cosmic browser states (97% vs 82% for gpt-6-luna; 0 vs 6 wrong
actions per 102 steps) but allows only 10 requests/second per organization,
shared by every Cosmic run. OpenAI's endpoint takes 5,000 requests/minute
and answers in ~200 ms, so it is the place to go when Perplexity is busy:

- 429 / 500 / 502 / 503 / 529 and transport errors are retried on the primary,
  waiting for the rate-limit window to reset (bounded by retry_max_wait_ms).
- A timeout, any other status or a non-JSON body skips straight to the
  fallback: the same request would not fare better a second time.
- After the retries end in a 429 the primary cools down briefly and calls
  in that window go straight to the fallback, so a burst does not add the
  full retry wait to every step.

The request is written once, in the primary's (Perplexity/TypeSafe) shape,
and translated here; the answers come back in that same shape, so callers
never see which provider answered. A question may carry a "fallback" dict
of OpenAI-specific overrides ({"instructions": ..., "invert": bool}); it is
removed before the primary sees the request. "invert" is for a question
OpenAI answers better phrased the other way round: its probability is
flipped back so the caller reads the original question's answer.
"""

import asyncio
import json
import random
import time
from typing import Any, Dict, Optional, Tuple

import httpx

OPENAI_DECISIONS_URL = "https://api.openai.com/v1/decisions"
OPENAI_DECIDER_MODEL = "gpt-6-luna"
FALLBACK_KEY = "fallback"

# Primary url -> monotonic time until which calls skip it (after a 429 run).
_COOLDOWN_UNTIL: Dict[str, float] = {}
COOLDOWN_MAX_SEC = 2.0


def primary_body(body: Dict[str, Any]) -> Dict[str, Any]:
    """The request without per-question fallback overrides."""
    questions = body.get("questions")
    if not isinstance(questions, dict):
        return body
    if not any(isinstance(q, dict) and FALLBACK_KEY in q for q in questions.values()):
        return body
    stripped = {
        name: ({k: v for k, v in q.items() if k != FALLBACK_KEY} if isinstance(q, dict) else q)
        for name, q in questions.items()
    }
    return {**body, "questions": stripped}


def _text(value: Any) -> str:
    """OpenAI takes instructions and descriptions as plain strings."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            if isinstance(item, list):
                item = "\n\n".join(str(x) for x in item)
            elif not isinstance(item, str):
                item = json.dumps(item, ensure_ascii=False)
            parts.append(f"{key}: {item}")
        return "\n\n".join(parts)
    return json.dumps(value, ensure_ascii=False)


def _openai_input(state: Any) -> Any:
    if isinstance(state, str):
        return state
    if not isinstance(state, list):
        return json.dumps(state, ensure_ascii=False)
    parts = []
    for item in state:
        if isinstance(item, str):
            parts.append({"type": "input_text", "text": item})
        elif isinstance(item, dict) and item.get("type") == "image_url":
            url = (item.get("image_url") or {}).get("url") if isinstance(item.get("image_url"), dict) else item.get("image_url")
            if url:
                parts.append({"type": "input_image", "image_url": url})
        else:
            parts.append({"type": "input_text", "text": json.dumps(item, ensure_ascii=False)})
    return [{"role": "user", "content": parts}]


def openai_request(body: Dict[str, Any], model: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """(OpenAI request, answers decided locally).

    OpenAI rejects a choice with fewer than two options; with one option the
    answer is not in doubt, so it is answered here as certain (which is what
    the primary returns for it) and left out of the request."""
    questions_out = []
    local: Dict[str, Any] = {}
    for name, question in (body.get("questions") or {}).items():
        if not isinstance(question, dict):
            continue
        override = question.get(FALLBACK_KEY) if isinstance(question.get(FALLBACK_KEY), dict) else {}
        instructions = _text(override.get("instructions", question.get("instructions", "")))
        kind = question.get("type")
        criteria = question.get("criteria")
        if kind == "choice":
            criteria = criteria or {}
            if len(criteria) < 2:
                if criteria:
                    only = next(iter(criteria))
                    local[name] = {"type": "choice", "choice": only, "confidence": 1.0, "probabilities": {only: 1.0}}
                continue
            choices = []
            for value, description in criteria.items():
                choice = {"value": value}
                if description is not None:
                    # Structured descriptions (an element and its current
                    # value) stay JSON: written as "key: value" text, OpenAI
                    # picked the wrong Wikipedia link in the live check.
                    choice["description"] = description if isinstance(description, str) else json.dumps(description, ensure_ascii=False)
                choices.append(choice)
            questions_out.append({"type": "choice", "name": name, "instructions": instructions, "choices": choices})
        elif kind == "score":
            levels = criteria if isinstance(criteria, list) else list((criteria or {}).values())
            questions_out.append({
                "type": "score", "name": name, "instructions": instructions,
                "levels": [{"label": _text(level)} for level in levels],
            })
        elif kind == "noul":
            if isinstance(criteria, dict) and criteria and not override.get("instructions"):
                instructions += f"\nTrue: {_text(criteria.get('true'))}\nFalse: {_text(criteria.get('false'))}"
            questions_out.append({"type": "predicate", "name": name, "instructions": instructions})
    request = {"model": model, "input": _openai_input(body.get("state")), "questions": questions_out}
    return request, local


def openai_answers(data: Dict[str, Any], questions: Dict[str, Any], local: Dict[str, Any]) -> Dict[str, Any]:
    """OpenAI's answer list in the primary's shape (answers keyed by name).

    Choice probabilities arrive rounded to two decimals; they are rescaled to
    sum to 1 so the fast path's answer contract holds. A refusal, or any
    answer type not asked for, is left out: the caller treats a missing
    answer exactly as it treats a missing answer from the primary."""
    out: Dict[str, Any] = dict(local)
    for answer in data.get("answers") or []:
        if not isinstance(answer, dict):
            continue
        name, kind = answer.get("name"), answer.get("type")
        question = questions.get(name) if isinstance(questions, dict) else None
        if not isinstance(question, dict):
            continue
        override = question.get(FALLBACK_KEY) if isinstance(question.get(FALLBACK_KEY), dict) else {}
        try:
            if kind == "predicate":
                p = float(answer.get("probability"))
                out[name] = {"type": "noul", "noul": (1.0 - p) if override.get("invert") else p}
            elif kind == "choice":
                probabilities = {str(p.get("value")): float(p.get("probability") or 0.0) for p in answer.get("probabilities") or []}
                total = sum(probabilities.values())
                if total > 0:
                    probabilities = {k: v / total for k, v in probabilities.items()}
                out[name] = {
                    "type": "choice",
                    "choice": str(answer.get("choice")),
                    "confidence": answer.get("confidence"),
                    "probabilities": probabilities,
                }
            elif kind == "score":
                out[name] = {
                    "type": "score",
                    "score": answer.get("score"),
                    "confidence": answer.get("confidence"),
                    "probabilities": {str(p.get("value")): p.get("probability") for p in answer.get("probabilities") or []},
                }
        except (TypeError, ValueError):
            continue
    return out


def _rate_limit_wait(response: httpx.Response, attempt: int, cap: float) -> float:
    """How long to wait before retrying: the server's Retry-After, else the
    time until x-ratelimit-reset (Unix seconds), else exponential backoff."""
    wait = None
    try:
        wait = float(response.headers.get("retry-after"))
    except (TypeError, ValueError):
        pass
    if wait is None:
        try:
            wait = float(response.headers.get("x-ratelimit-reset")) - time.time()
        except (TypeError, ValueError):
            pass
    if wait is None or wait <= 0:
        wait = 0.25 * (2 ** attempt)
    return min(max(wait, 0.1), cap) + random.uniform(0, 0.1)


async def post_decisions(client: httpx.AsyncClient, config: Any, body: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """(response body in the primary's shape, or None; meta).

    meta: provider (who answered), attempts (primary calls made), retries,
    fallback (bool), error (why the primary did not answer)."""
    meta: Dict[str, Any] = {"provider": getattr(config, "provider", ""), "attempts": 0, "retries": 0, "fallback": False, "error": ""}
    attempts = max(1, int(getattr(config, "attempts", 1) or 1))
    retry_statuses = set(getattr(config, "retry_statuses", ()) or ())
    cap = max(0.0, float(getattr(config, "retry_max_wait_ms", 1000) or 0) / 1000)
    fallback_key = str(getattr(config, "fallback_api_key", "") or "")
    url = config.api_url

    cooling = bool(fallback_key) and time.monotonic() < _COOLDOWN_UNTIL.get(url, 0.0)
    if cooling:
        meta["error"] = "primary rate-limited; cooling down"
    elif config.api_key:
        request = primary_body(body)
        for attempt in range(attempts):
            meta["attempts"] += 1
            last = attempt + 1 >= attempts
            try:
                response = await client.post(url, json=request, headers={"Authorization": f"Bearer {config.api_key}"})
            except httpx.TimeoutException:
                meta["error"] = "timeout"
                break
            except httpx.HTTPError as exc:
                meta["error"] = f"transport: {type(exc).__name__}"
                if last:
                    break
                meta["retries"] += 1
                await asyncio.sleep(min(0.25 * (2 ** attempt), cap))
                continue
            if not response.is_error:
                try:
                    return response.json(), meta
                except ValueError:
                    meta["error"] = "non-JSON response"
                    break
            meta["error"] = f"HTTP {response.status_code}"
            if response.status_code not in retry_statuses or last:
                if response.status_code == 429 and fallback_key:
                    _COOLDOWN_UNTIL[url] = time.monotonic() + min(_rate_limit_wait(response, 0, COOLDOWN_MAX_SEC), COOLDOWN_MAX_SEC)
                break
            meta["retries"] += 1
            # Never time.sleep here: it would freeze the event loop (live
            # frame relay, takeover listener) for the whole back-off.
            await asyncio.sleep(_rate_limit_wait(response, attempt, cap))
    else:
        meta["error"] = "no primary key"

    if not fallback_key:
        return None, meta
    request, local = openai_request(body, str(getattr(config, "fallback_model", "") or OPENAI_DECIDER_MODEL))
    if not request["questions"]:
        return ({"answers": local, "model": "local"}, {**meta, "provider": "local", "fallback": True}) if local else (None, meta)
    timeout = max(1.0, float(getattr(config, "fallback_timeout_ms", 8000) or 8000) / 1000)
    for attempt in range(2):
        try:
            response = await client.post(
                str(getattr(config, "fallback_url", "") or OPENAI_DECISIONS_URL),
                json=request,
                headers={"Authorization": f"Bearer {fallback_key}"},
                timeout=timeout,
            )
        except httpx.HTTPError as exc:
            meta["error"] += f"; fallback transport: {type(exc).__name__}"
            return None, meta
        if response.status_code in {429, 500, 502, 503} and attempt == 0:
            await asyncio.sleep(0.25)
            continue
        if response.is_error:
            meta["error"] += f"; fallback HTTP {response.status_code}"
            return None, meta
        try:
            data = response.json()
        except ValueError:
            meta["error"] += "; fallback non-JSON response"
            return None, meta
        meta.update(provider="openai", fallback=True)
        return {
            "answers": openai_answers(data, body.get("questions") or {}, local),
            "model": data.get("model"),
            "usage": data.get("usage"),
        }, meta
    meta["error"] += "; fallback busy"
    return None, meta
