"""
Local webhook server for AgentPhone (SMS + voice in webhook mode).

Run (after `provision.py` and exposing HTTPS, e.g. ngrok):
  cd AgentPhone
  uvicorn webhook_server:app --host 0.0.0.0 --port 8765

Voice: logs each STT segment from webhooks and starts one SSE transcript stream per call
(see https://docs.agentphone.ai/documentation/guides/calls#stream-transcript-sse ).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional

import requests
from agentphone import AgentPhone
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse

import config
from agent_resolve import resolve_agent_id, resolve_number_id
from security import verify_agentphone_webhook
from transcript_sse import start_transcript_stream_once

ROOT = Path(__file__).resolve().parent
COSMIC_DIR = (ROOT.parent / "cosmic-browser-use").resolve()
load_dotenv(ROOT / ".env")
load_dotenv(COSMIC_DIR / ".env", override=False)

app = FastAPI(title="AgentPhone standalone webhook")


@app.middleware("http")
async def _log_every_request(request: Request, call_next):
    """Log every incoming request so we can tell if traffic reaches the server at all."""
    try:
        client_host = request.client.host if request.client else "?"
    except Exception:
        client_host = "?"
    print(
        f"[req] {request.method} {request.url.path} from={client_host} "
        f"event={request.headers.get('x-webhook-event', '-')!r} "
        f"sig={'yes' if request.headers.get('x-webhook-signature') else 'no'} "
        f"len={request.headers.get('content-length', '?')}",
        flush=True,
    )
    response = await call_next(request)
    print(f"[req] -> {request.method} {request.url.path} status={response.status_code}", flush=True)
    return response

# ===========================================================================
# Code-defined agent reply (used in webhook voice mode, i.e. `make_call.py`
# without `--hosted`). This is what the AgentPhone line speaks back to the
# caller. Edit `DEFAULT_AGENT_RESPONSE` for a quick change, or rewrite
# `agent_response()` for richer logic (LLM call, RAG, scripted flow, etc.).
# It runs every time AgentPhone POSTs an `agent.message` voice event, so it
# sees the latest user STT transcript plus `recentHistory`.
# ===========================================================================

DEFAULT_AGENT_RESPONSE = (
    "Thanks — I'm listening. Tell me more whenever you're ready."
)

SMS_TRIGGER_PHRASES = ("text me", "send me a text", "sms me", "send me an sms", "send me a sms")
SMS_BODY = (
    "Hi from your AgentPhone agent — you asked me on the call to text you, "
    "so here it is. Sent from the standalone setup."
)


def _send_sms_async(to_number: str, body: str) -> None:
    """Fire-and-forget outbound SMS; runs in a thread so the voice reply isn't delayed."""

    def _run() -> None:
        try:
            client = AgentPhone(api_key=os.environ["AGENTPHONE_API_KEY"])
            agent_id = resolve_agent_id(client)
            number_id = resolve_number_id(client, agent_id) or None
            result = client.messages.send(
                agent_id=agent_id,
                to_number=to_number,
                body=body,
                number_id=number_id,
            )
            mid = (result or {}).get("id") if isinstance(result, dict) else None
            print(f"[sms/out] -> to={to_number} id={mid} body={body!r}", flush=True)
        except Exception as exc:  # noqa: BLE001 — log and move on; never crash the webhook
            print(f"[sms/out] failed to {to_number}: {exc!r}", flush=True)

    threading.Thread(target=_run, daemon=True).start()


def agent_response(
    user_transcript: str,
    history: list[dict[str, Any]],
    payload: dict[str, Any],
) -> str:
    """Return the text the agent should say back. Override in code to customize.

    Args:
        user_transcript: Latest STT segment for the current call turn.
        history: `recentHistory` from the webhook payload (recent user+agent turns).
        payload: Full webhook payload (useful for `callId`, `data.from`, etc.).
    """
    _ = history  # available for richer flows; not used in this minimal reply.

    text = (user_transcript or "").lower().strip()

    # If the caller asks to be texted, fire an SMS to the from-number in the background.
    if text and any(phrase in text for phrase in SMS_TRIGGER_PHRASES):
        data = payload.get("data") or {}
        caller = data.get("from") or data.get("fromNumber")
        if caller:
            _send_sms_async(caller, SMS_BODY)
            return f"Sure, I just sent you a text at {caller}. Check your messages."
        return "I'd love to text you, but I couldn't read your number from this call."

    return DEFAULT_AGENT_RESPONSE


# ===========================================================================
# Call-session bridge: lets an external process (e.g. AgentPhone/call_to_browse.py)
# drive what the agent says on a specific live call, and read what the caller
# says back. Used by the cosmic-browser-use AskUser tool to ask questions over
# voice and wait for the user's reply.
#
# All state is in-memory and only reachable through local control endpoints —
# AgentPhone itself still posts to /webhook with HMAC verification as before.
# ===========================================================================


class _CallSession:
    """One live call that an external script has claimed.

    The webhook handler reads `pending_speech` to decide what to say next, and
    pushes each STT transcript onto `reply_queue` so the external script can
    `next_reply()` it back to the asker.
    """

    HOLDING_PHRASE = "Still working on it, give me a moment."

    def __init__(self, call_id: str, to_number: Optional[str] = None) -> None:
        self.call_id = call_id
        self.to_number = to_number
        # asyncio primitives — must be created on the running loop.
        self.pending_speech: Optional[str] = None
        self.last_spoken: Optional[str] = None
        self.holding_phrase: str = self.HOLDING_PHRASE
        self.reply_queue: "asyncio.Queue[str]" = asyncio.Queue()
        self.closed: bool = False
        self.created_at: float = time.time()
        self.last_activity: float = time.time()
        self.status_text: str = "I’m starting the browser now."
        self.final_answer: Optional[str] = None
        self.run_dir: Optional[str] = None
        self.last_progress_step: int = 0

    def take_speech(self) -> Optional[str]:
        """Pop the staged agent line (if any), so it's only spoken once."""
        text = self.pending_speech
        self.pending_speech = None
        if text:
            self.last_spoken = text
        return text

    def stage_speech(self, text: str) -> None:
        cleaned = str(text or "").strip()
        if cleaned:
            self.pending_speech = cleaned
            self.status_text = cleaned
            self.last_activity = time.time()


_call_sessions: dict[str, _CallSession] = {}
_call_sessions_lock = threading.Lock()


def _get_session(call_id: str) -> Optional[_CallSession]:
    with _call_sessions_lock:
        return _call_sessions.get(call_id)


_SEEN_WEBHOOK_IDS: deque[str] = deque(maxlen=4096)
_SEEN_SET: set[str] = set()


def _duplicate(webhook_id: str | None) -> bool:
    if not webhook_id:
        return False
    if webhook_id in _SEEN_SET:
        return True
    if len(_SEEN_SET) >= _SEEN_WEBHOOK_IDS.maxlen:
        old = _SEEN_WEBHOOK_IDS.popleft()
        _SEEN_SET.discard(old)
    _SEEN_WEBHOOK_IDS.append(webhook_id)
    _SEEN_SET.add(webhook_id)
    return False


def _client() -> AgentPhone:
    return AgentPhone(api_key=os.environ["AGENTPHONE_API_KEY"])


def _voice_call_key(data: dict[str, Any]) -> tuple[str | None, str | None]:
    """Return (stable_session_key, real_agentphone_call_id).

    AgentPhone sometimes omits `data.callId` on live inbound `agent.message`
    webhooks even though the final `agent.call_ended` event includes it. The
    browse-on-call bridge only needs a stable key across turns, so fall back to
    a sanitized direction/from/to key when the real call id is absent.
    """
    real_call_id = (
        data.get("callId")
        or data.get("call_id")
        or data.get("callID")
        or data.get("id")
    )
    if real_call_id:
        return str(real_call_id), str(real_call_id)

    direction = str(data.get("direction") or "voice").lower()
    from_number = data.get("from") or data.get("fromNumber") or "unknown-from"
    to_number = data.get("to") or data.get("toNumber") or "unknown-to"
    raw = f"{direction}:{from_number}:{to_number}"
    synthetic = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_")
    return (synthetic or None), None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Call-session control plane (local-only — no AgentPhone signature on these).
# Designed to be hit by AgentPhone/call_to_browse.py from the same machine.
# ---------------------------------------------------------------------------


@app.post("/control/session")
async def control_register_session(request: Request) -> JSONResponse:
    body = await request.json()
    call_id = str(body.get("call_id") or "").strip()
    if not call_id:
        return JSONResponse({"error": "call_id required"}, status_code=400)
    to_number = body.get("to_number")
    with _call_sessions_lock:
        sess = _call_sessions.get(call_id)
        if sess is None:
            sess = _CallSession(call_id=call_id, to_number=to_number)
            _call_sessions[call_id] = sess
            created = True
        else:
            sess.to_number = to_number or sess.to_number
            sess.last_activity = time.time()
            created = False
    print(
        f"[control] session {'created' if created else 'reused'} call={call_id} to={to_number!r} "
        f"active_sessions={len(_call_sessions)}",
        flush=True,
    )
    return JSONResponse({"ok": True, "call_id": call_id, "created": created})


@app.post("/control/session/{call_id}/ask")
async def control_ask(call_id: str, request: Request) -> JSONResponse:
    sess = _get_session(call_id)
    if sess is None:
        return JSONResponse({"error": "unknown call_id"}, status_code=404)
    body = await request.json()
    question = str(body.get("question") or "").strip()
    if not question:
        return JSONResponse({"error": "question required"}, status_code=400)
    sess.pending_speech = question
    sess.last_activity = time.time()
    print(f"[control] ask call={call_id} question={question!r}", flush=True)
    return JSONResponse({"ok": True})


@app.post("/control/session/{call_id}/say")
async def control_say(call_id: str, request: Request) -> JSONResponse:
    """Stage a one-off line the agent will speak the next time the caller talks.

    Use this for status updates / the final answer when you DON'T need a reply.
    """
    sess = _get_session(call_id)
    if sess is None:
        return JSONResponse({"error": "unknown call_id"}, status_code=404)
    body = await request.json()
    text = str(body.get("text") or "").strip()
    if not text:
        return JSONResponse({"error": "text required"}, status_code=400)
    sess.pending_speech = text
    sess.last_activity = time.time()
    print(f"[control] say call={call_id} text={text!r}", flush=True)
    return JSONResponse({"ok": True})


@app.get("/control/session/{call_id}/next_reply")
async def control_next_reply(call_id: str, timeout: float = 120.0) -> JSONResponse:
    sess = _get_session(call_id)
    if sess is None:
        return JSONResponse({"error": "unknown call_id"}, status_code=404)
    try:
        reply = await asyncio.wait_for(sess.reply_queue.get(), timeout=max(0.1, float(timeout)))
    except asyncio.TimeoutError:
        return JSONResponse({"timeout": True, "reply": None}, status_code=204)
    sess.last_activity = time.time()
    print(f"[control] next_reply call={call_id} reply={reply!r}", flush=True)
    return JSONResponse({"reply": reply})


@app.post("/control/session/{call_id}/close")
async def control_close(call_id: str) -> JSONResponse:
    with _call_sessions_lock:
        sess = _call_sessions.pop(call_id, None)
    if sess is None:
        return JSONResponse({"ok": True, "existed": False})
    sess.closed = True
    print(f"[control] session closed call={call_id} active_sessions={len(_call_sessions)}", flush=True)
    return JSONResponse({"ok": True, "existed": True})


@app.get("/control/sessions")
async def control_list_sessions() -> JSONResponse:
    with _call_sessions_lock:
        items = [
            {
                "call_id": s.call_id,
                "to_number": s.to_number,
                "pending": s.pending_speech is not None,
                "queued_replies": s.reply_queue.qsize(),
                "age_sec": int(time.time() - s.created_at),
                "idle_sec": int(time.time() - s.last_activity),
            }
            for s in _call_sessions.values()
        ]
    return JSONResponse({"sessions": items})


@app.post("/webhook")
async def webhook(request: Request) -> Response:
    settings = config.get_settings()
    raw = await request.body()
    sig = request.headers.get("X-Webhook-Signature") or request.headers.get("x-webhook-signature")
    ts = request.headers.get("X-Webhook-Timestamp") or request.headers.get("x-webhook-timestamp")
    wid = request.headers.get("X-Webhook-ID") or request.headers.get("x-webhook-id")

    if not verify_agentphone_webhook(raw, sig, ts, settings.webhook_secret):
        return PlainTextResponse("invalid signature", status_code=403)

    try:
        payload: dict[str, Any] = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError:
        return PlainTextResponse("invalid json", status_code=400)

    event = payload.get("event")
    channel = payload.get("channel")

    if event == "agent.message" and channel == "voice":
        return await _handle_voice_webhook(payload)

    if _duplicate(wid):
        return Response(status_code=200)

    if event == "agent.message" and channel in ("sms", "mms", "imessage"):
        _log_sms_like(payload)
        return Response(status_code=200)

    if event == "agent.reaction":
        print(f"[reaction] {json.dumps(payload, ensure_ascii=False)}", flush=True)
        return Response(status_code=200)

    if event == "agent.call_ended":
        print(f"[call_ended] {json.dumps(payload, ensure_ascii=False)}", flush=True)
        ended_data = payload.get("data") or {}
        ended_real_id = ended_data.get("callId") or ""
        ended_key, _ = _voice_call_key(ended_data)
        cleanup_keys = {str(k) for k in (ended_real_id, ended_key) if k}
        # Also clear synthetic caller-pair keys for this completed call. Live
        # inbound speech events may have used the synthetic key because callId
        # was missing, while call_ended has the real callId.
        from_number = ended_data.get("from") or ended_data.get("fromNumber")
        to_number = ended_data.get("to") or ended_data.get("toNumber")
        direction = str(ended_data.get("direction") or "voice").lower()
        if from_number or to_number:
            raw = f"{direction}:{from_number or 'unknown-from'}:{to_number or 'unknown-to'}"
            cleanup_keys.add(re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_"))
        # Live inbound `agent.message` webhooks sometimes omit both callId and
        # phone numbers, so their bridge key is this unknown-pair fallback. Clear
        # it when the corresponding call_end arrives to avoid stale sessions.
        if direction == "inbound":
            cleanup_keys.add("inbound_unknown-from_unknown-to")
        for key in cleanup_keys:
            _drop_pending(key)
        with _call_sessions_lock:
            for key in cleanup_keys:
                _call_sessions.pop(key, None)
        return Response(status_code=200)

    print(f"[webhook] (unhandled) {json.dumps(payload, ensure_ascii=False)}", flush=True)
    return Response(status_code=200)


def _log_sms_like(payload: dict[str, Any]) -> None:
    data = payload.get("data") or {}
    print(
        f"[sms] agent={payload.get('agentId')} conv={data.get('conversationId')} "
        f"dir={data.get('direction')} from={data.get('from')} to={data.get('to')} "
        f"body={data.get('message')!r} media={data.get('mediaUrl')}",
        flush=True,
    )


# ===========================================================================
# Browse-on-call: when an inbound caller dials our AgentPhone line, treat the
# first thing they say as the browsing goal and auto-launch a cosmic-browser-use
# subprocess driven through the same session bridge that call_to_browse.py uses.
# Off by default — set BROWSE_ON_CALL_ENABLED=1 in .env to enable.
# ===========================================================================


def _browse_on_call_enabled() -> bool:
    return os.environ.get("BROWSE_ON_CALL_ENABLED", "").strip().lower() in {"1", "true", "yes", "y", "on"}


class _BrowsePending:
    """Per-call state while we're still gathering + confirming the user's goal."""

    def __init__(self, call_id: str, from_number: str) -> None:
        self.call_id = call_id
        self.from_number = from_number
        self.candidate_goal: Optional[str] = None
        self.last_transcript: Optional[str] = None
        self.created_at: float = time.time()
        self.confirm_attempts: int = 0
        self.turns: list[dict[str, str]] = []


_browse_pending: dict[str, _BrowsePending] = {}
_browse_pending_lock = threading.Lock()


def _get_or_create_pending(call_id: str, from_number: str) -> _BrowsePending:
    with _browse_pending_lock:
        pending = _browse_pending.get(call_id)
        if pending is None:
            pending = _BrowsePending(call_id=call_id, from_number=from_number)
            _browse_pending[call_id] = pending
        return pending


def _drop_pending(call_id: str) -> None:
    with _browse_pending_lock:
        _browse_pending.pop(call_id, None)


# Confirmation tokens — STRICT. We only treat very specific short utterances as
# "start the task". Things like "yeah" or "yes" are NOT enough because callers
# often say them as small talk ("Yeah. Can you hear me?") right after picking up.
# Caller has to literally say "go" / "start" / "do it" / etc.
_CONFIRM_PHRASES = {
    "go",
    "go.",
    "go ahead",
    "go ahead.",
    "just go",
    "good just go",
    "start",
    "start.",
    "begin",
    "begin.",
    "do it",
    "do it.",
    "let's go",
    "lets go",
    "kick it off",
    "fire away",
    "go for it",
    "yes go",
    "yes go ahead",
    "ok go",
    "okay go",
    "sure go ahead",
    "that's cool",
    "thats cool",
    "that is cool",
    "yeah that's cool",
    "yeah thats cool",
    "good",
    "all good",
    "looks good",
    "sounds good",
    "perfect",
    # Common STT mishears of "go to start" / "go, start".
    "go to start",
    "go to stat",
    "go start",
}


def _looks_like_confirm(text: str) -> bool:
    """True only for short, standalone start-the-task phrases. "Yeah" alone is NOT enough."""
    t = (text or "").lower().strip()
    t = re.sub(r"[^a-z0-9\s']", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return False
    if t in _CONFIRM_PHRASES:
        return True
    # Allow trailing " please" / " now" / " then"
    for suf in (" please", " now", " then"):
        if t.endswith(suf):
            inner = t[: -len(suf)].strip()
            if inner in _CONFIRM_PHRASES:
                return True
    return False


_FILLER_ONLY_PHRASES = {
    "so",
    "uh",
    "um",
    "hmm",
    "yeah",
    "yes",
    "yep",
    "ok",
    "okay",
    "cool",
    "right",
    "alright",
    "all right",
    "nothing",
    "oh nothing",
    "no nothing",
}


def _looks_like_filler_only(text: str) -> bool:
    """Short conversational crumbs that should not replace a pending goal."""
    t = (text or "").lower().strip()
    t = re.sub(r"[^a-z0-9\s']", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return True
    if t in _FILLER_ONLY_PHRASES:
        return True
    # Very short utterances with no task/action words are usually ASR fragments.
    words = t.split()
    if len(words) <= 3 and not any(
        w in t
        for w in (
            "youtube",
            "google",
            "search",
            "browse",
            "open",
            "find",
            "get",
            "description",
            "video",
            "website",
            "page",
        )
    ):
        return True
    return False


def _heuristic_normalize_spoken_goal(raw_goal: str) -> str:
    """Cheap fallback cleaner for conversational STT goals."""
    text = (raw_goal or "").strip()
    if not text:
        return text

    # Common STT drift for the specific Claude Code launch-video task.
    low = text.lower()
    if (
        "youtube" in low
        and "description" in low
        and "launch" in low
        and "code" in low
        and any(term in low for term in ("claude", "cloud", "clock", "blog", "plot"))
    ):
        return "Get the YouTube video description for the official Claude Code Launch Video"

    cleaned = text
    cleaned = re.sub(r"^(yeah|yes|yep|ok|okay|cool|right|alright|oh|uh|um|so)[,.\s]+", "", cleaned, flags=re.I)
    cleaned = re.sub(r"^(nothing|no nothing|oh nothing)[,.\s]+", "", cleaned, flags=re.I)
    cleaned = re.sub(r"^(can you|could you|would you)\s+", "", cleaned, flags=re.I)
    cleaned = re.sub(r"^just\s+", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\bfor me\b", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,")
    if cleaned and not cleaned.lower().startswith(("get ", "find ", "open ", "go ", "search ", "navigate ")):
        cleaned = f"Get {cleaned[0].lower() + cleaned[1:]}"
    return cleaned or text


def _postprocess_normalized_goal(goal: str) -> str:
    """Make LLM-normalized goals match the browser agent's proven-good style."""
    text = (goal or "").strip()
    if not text:
        return text
    low = text.lower()

    if (
        "youtube" in low
        and "description" in low
        and "launch" in low
        and "code" in low
        and any(term in low for term in ("claude", "cloud", "clock", "blog", "plot"))
    ):
        return "Get the YouTube video description for the official Claude Code Launch Video"

    # Convert "Go to YouTube and get the X description" into a query-like goal
    # that matches the workflow memory index better.
    match = re.search(
        r"(?:go to|open|visit)?\s*youtube\s*(?:and)?\s*(?:get|find|grab|retrieve)?\s*(?:me\s*)?(?:the\s*)?(?P<subject>.+?)\s*(?:video\s*)?description",
        text,
        flags=re.I,
    )
    if match:
        subject = match.group("subject").strip(" .,:;-")
        if subject:
            return f"Get the YouTube video description for {subject}"

    return text


def _normalize_spoken_goal(raw_goal: str) -> str:
    """Turn conversational phone STT into the concise task sent to cosmic-browser-use.

    The browser agent is excellent once its `--goal` is clean. Do not pass raw
    voice dictation like "can you go to YouTube..." as-is, because planners often
    use the goal text directly as a search query.
    """
    fallback = _postprocess_normalized_goal(_heuristic_normalize_spoken_goal(raw_goal))
    if os.environ.get("BROWSE_ON_CALL_LLM_NORMALIZE", "1").strip().lower() not in {"1", "true", "yes", "on"}:
        return fallback

    provider = os.environ.get("BROWSE_ON_CALL_NORMALIZER_PROVIDER", "kimi").strip().lower()
    if provider == "fireworks":
        provider = "kimi"
    if provider not in {"auto", "kimi", "openai", "gemini"}:
        provider = "kimi"

    fireworks_key = (
        os.environ.get("FIREWORKS_API_KEY")
        or os.environ.get("SLIDE_AGENT_FIREWORKS_API_KEY")
        or ""
    ).strip().strip('"')
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip().strip('"')
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip().strip('"')
    if not fireworks_key and not openai_key and not gemini_key:
        return fallback

    prompt = f"""
You normalize spoken browser-agent requests.

Return ONLY compact JSON: {{"goal": "..."}}

Rules:
- Remove conversational filler: "can you", "please", "for me", "oh nothing", "yeah".
- Preserve the user's actual browsing intent.
- Write an imperative browser task suitable for an automation agent.
- If speech recognition says cloud/clock/blog/plot code launch video in a YouTube-description request, interpret it as "Claude Code Launch Video" unless context strongly says otherwise.
- Do not include explanations.

Raw spoken transcript:
{raw_goal!r}
""".strip()

    def _parse_goal(text: str) -> Optional[str]:
        try:
            parsed = json.loads(text)
            goal = str(parsed.get("goal") or "").strip()
            return goal or None
        except Exception:
            match = re.search(r"\{.*\}", text, flags=re.S)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                    goal = str(parsed.get("goal") or "").strip()
                    return goal or None
                except Exception:
                    return None
        return None

    def _try_kimi() -> Optional[str]:
        if not fireworks_key:
            return None
        base_url = (os.environ.get("FIREWORKS_BASE_URL") or "https://api.fireworks.ai/inference/v1").rstrip("/")
        model = (
            os.environ.get("BROWSE_ON_CALL_KIMI_NORMALIZER_MODEL")
            or os.environ.get("FIREWORKS_KIMI_MODEL")
            or "accounts/fireworks/models/kimi-k2p6"
        ).strip()
        response = requests.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {fireworks_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "Return only compact JSON."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
                "max_tokens": 96,
                "response_format": {"type": "json_object"},
            },
            timeout=float(os.environ.get("BROWSE_ON_CALL_NORMALIZER_TIMEOUT", "6")),
        )
        response.raise_for_status()
        text = response.json()["choices"][0]["message"]["content"]
        return _parse_goal(text)

    def _try_openai() -> Optional[str]:
        if not openai_key:
            return None
        model = os.environ.get("BROWSE_ON_CALL_OPENAI_NORMALIZER_MODEL", "gpt-4o-mini").strip()
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {openai_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "Return only compact JSON."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
                "max_tokens": 96,
                "response_format": {"type": "json_object"},
            },
            timeout=float(os.environ.get("BROWSE_ON_CALL_NORMALIZER_TIMEOUT", "6")),
        )
        response.raise_for_status()
        text = response.json()["choices"][0]["message"]["content"]
        return _parse_goal(text)

    def _try_gemini() -> Optional[str]:
        if not gemini_key:
            return None
        model = os.environ.get("BROWSE_ON_CALL_GEMINI_NORMALIZER_MODEL", "gemini-2.0-flash").strip()
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            params={"key": gemini_key},
            json={
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0,
                    "maxOutputTokens": 96,
                    "responseMimeType": "application/json",
                },
            },
            timeout=float(os.environ.get("BROWSE_ON_CALL_NORMALIZER_TIMEOUT", "6")),
        )
        response.raise_for_status()
        data = response.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return _parse_goal(text)

    attempts = []
    if provider in {"auto", "kimi"}:
        attempts.append(("kimi", _try_kimi))
    if provider in {"auto", "openai"}:
        attempts.append(("openai", _try_openai))
    if provider in {"auto", "gemini"}:
        attempts.append(("gemini", _try_gemini))

    for name, fn in attempts:
        try:
            goal = fn()
            if goal:
                goal = _postprocess_normalized_goal(goal)
                print(f"[browse-on-call] normalized goal via {name}: raw={raw_goal!r} -> goal={goal!r}", flush=True)
                return goal
        except Exception as exc:  # noqa: BLE001
            print(f"[browse-on-call] {name} goal normalizer failed: {exc!r}", flush=True)

    if fallback != raw_goal:
        print(
            f"[browse-on-call] using heuristic-normalized fallback: raw={raw_goal!r} -> goal={fallback!r}",
            flush=True,
        )
    return fallback


def _call_kimi_json(prompt: str, *, max_tokens: int = 220, timeout: float = 8.0) -> dict[str, Any]:
    """Call Kimi via Fireworks and parse a compact JSON object."""
    fireworks_key = (
        os.environ.get("FIREWORKS_API_KEY")
        or os.environ.get("SLIDE_AGENT_FIREWORKS_API_KEY")
        or ""
    ).strip().strip('"')
    if not fireworks_key:
        raise RuntimeError("FIREWORKS_API_KEY / SLIDE_AGENT_FIREWORKS_API_KEY is not set")

    base_url = (os.environ.get("FIREWORKS_BASE_URL") or "https://api.fireworks.ai/inference/v1").rstrip("/")
    model = (
        os.environ.get("BROWSE_ON_CALL_KIMI_MODEL")
        or os.environ.get("FIREWORKS_KIMI_MODEL")
        or "accounts/fireworks/models/kimi-k2p6"
    ).strip()
    response = requests.post(
        f"{base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {fireworks_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are the voice intake brain for a browser automation assistant. "
                        "Return only compact JSON. No markdown."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        },
        timeout=timeout,
    )
    response.raise_for_status()
    text = response.json()["choices"][0]["message"]["content"]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def _browse_intake_decision(pending: _BrowsePending, utterance: str) -> dict[str, Any]:
    """Let Kimi decide what this phone turn means.

    This replaces the brittle rule cascade. Kimi sees the candidate goal and the
    recent mini-dialogue, then returns one of:
      chat | ask_goal | update_goal | start | cancel
    """
    history = pending.turns[-8:]
    prompt = f"""
You are handling an inbound phone call for a browser-use agent.

The caller may chat casually, correct speech-to-text mistakes, provide a browsing task,
confirm a proposed task, or cancel/restart.

Current proposed browser goal:
{pending.candidate_goal!r}

Recent turns:
{json.dumps(history, ensure_ascii=False)}

Latest caller transcript:
{utterance!r}

Return ONLY JSON:
{{
  "action": "chat" | "ask_goal" | "update_goal" | "start" | "cancel",
  "goal": string | null,
  "reply": string,
  "reason": string
}}

Decision policy:
- If caller is only greeting/checking the line/how you are: action="chat"; reply naturally and ask what to browse.
- If caller says no/nope/stop/scratch that/not that and does not give a new task: action="cancel"; clear any proposed goal; ask them to say the browsing task when ready.
- If caller gives or corrects a browsing task: action="update_goal"; produce a clean imperative browser goal in "goal".
- If a clean goal is already proposed and caller affirms it ("go", "start", "yes", "that's cool", "sounds good", "do it", etc.): action="start"; use the existing proposed goal unless the latest transcript contains a better complete goal.
- If unclear and no goal exists: action="ask_goal".
- If unclear and a goal exists: keep that goal, ask for either "go" or a correction.

Goal-writing rules:
- Remove filler like "can you", "please", "for me", "oh nothing", "yeah".
- Preserve the actual browsing intent.
- Prefer concise goals that work well as --goal for browser automation.
- For YouTube video-description requests about cloud/clock/blog/plot/Claude code launch, normalize to:
  "Get the YouTube video description for the official Claude Code Launch Video"
- Do NOT make a goal from pure chat like "how are you doing".
- Replies should be short, friendly, spoken text. Avoid repeating a broken phrase many times.
""".strip()

    data = _call_kimi_json(
        prompt,
        max_tokens=int(os.environ.get("BROWSE_ON_CALL_INTAKE_MAX_TOKENS", "260")),
        timeout=float(os.environ.get("BROWSE_ON_CALL_INTAKE_TIMEOUT", "8")),
    )
    action = str(data.get("action") or "ask_goal").strip().lower()
    if action not in {"chat", "ask_goal", "update_goal", "start", "cancel"}:
        action = "ask_goal"
    goal = data.get("goal")
    goal_text = _postprocess_normalized_goal(str(goal).strip()) if goal else None
    reply = str(data.get("reply") or "").strip()
    if not reply:
        if action == "start":
            reply = "Got it. Starting now. I’ll check in if I need anything."
        elif action == "update_goal" and goal_text:
            reply = f"I’ll browse for: {goal_text}. Say go to start, or correct me."
        elif action == "cancel":
            reply = "No problem. Tell me what you want me to browse when you're ready."
        else:
            reply = "What would you like me to browse for you?"
    return {
        "action": action,
        "goal": goal_text,
        "reply": reply,
        "reason": str(data.get("reason") or "").strip(),
    }


def _compact_for_voice(text: str, max_chars: int = 1200) -> str:
    """Trim long browser answers into something reasonable to speak on a call."""
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    cut = cleaned[:max_chars].rsplit(" ", 1)[0].rstrip()
    return cut + "..."


def _read_browser_log_safely(log_path: Path) -> list[dict[str, Any]]:
    try:
        if not log_path.exists():
            return []
        raw = log_path.read_text(encoding="utf-8")
        if not raw.strip():
            return []
        return json.loads(raw)
    except Exception:
        # The browser process may be writing the JSON file while we read it.
        return []


def _extract_note_from_steps(steps: list[dict[str, Any]]) -> Optional[str]:
    """Return the latest saved note / answer from a cosmic-browser-use log."""
    for step in reversed(steps or []):
        action = step.get("action") or {}
        if action.get("action_type") in {"SaveNote", "SAVE_NOTE"}:
            tool_call = step.get("tool_call") or {}
            params = tool_call.get("parameters") or {}
            note = params.get("note")
            if note:
                return str(note)
            after_state = step.get("after_browser_state") or step.get("browser_state") or {}
            notes = after_state.get("notes") or []
            if notes:
                return str(notes[-1])
        state = step.get("after_browser_state") or step.get("browser_state") or {}
        notes = state.get("notes") or []
        if notes:
            return str(notes[-1])
    return None


def _progress_text_from_step(step: dict[str, Any]) -> Optional[str]:
    action = step.get("action") or {}
    action_type = str(action.get("action_type") or "")
    desc = str(action.get("description") or "")
    state = step.get("after_browser_state") or step.get("browser_state") or {}
    title = str(state.get("title") or "").replace("[Tab 1/1]", "").strip()
    url = str(state.get("url") or "")

    if action_type == "Navigate" and "youtube" in url.lower():
        return "I’m on YouTube now."
    if action_type == "VisualType":
        return "I’m searching YouTube for the video."
    if action_type == "VisualClick" and "Claude Code" in title:
        return "I found the Claude Code video and I’m opening the description."
    if action_type == "VisualScroll":
        return "I’m scrolling to the description."
    if action_type == "SaveNote":
        return "I found and saved the answer."
    if desc:
        return f"I’m working: {desc[:120]}"
    return None


def _subprocess_line_is_failure(line: str) -> bool:
    low = (line or "").lower()
    if "critical error" in low and "mimo" in low:
        return True
    if "mimo vision server is unreachable" in low:
        return True
    if "ping failed" in low and "mimo" in low:
        return True
    return False


def _stage_on_call(call_id: str, text: str) -> None:
    sess = _get_session(call_id)
    if sess is not None:
        sess.stage_speech(text)


def _find_newest_run_dir(after_ts: float, known_names: set[str]) -> Optional[Path]:
    runs_dir = COSMIC_DIR / "runs"
    try:
        candidates = []
        for p in runs_dir.iterdir():
            if not p.is_dir():
                continue
            try:
                mtime = p.stat().st_mtime
            except OSError:
                continue
            if p.name not in known_names and mtime >= after_ts - 5:
                candidates.append((mtime, p))
        if not candidates:
            return None
        return sorted(candidates, key=lambda item: item[0], reverse=True)[0][1]
    except Exception:
        return None


def _monitor_browser_run_for_call(
    *,
    call_id: str,
    proc: subprocess.Popen,
    known_run_names: set[str],
    process_start_ts: float,
) -> None:
    """Watch cosmic-browser-use run history and stage call updates/final answer.

    This is the missing link: the call session should not have to wait for the
    browser subprocess to exit before it can answer "are you done?" or report the
    saved result. The log has the answer as soon as SaveNote runs.
    """
    run_dir: Optional[Path] = None
    final_reported = False

    while proc.poll() is None and not final_reported:
        sess = _get_session(call_id)
        if sess is None:
            return

        if run_dir is None:
            run_dir = _find_newest_run_dir(process_start_ts, known_run_names)
            if run_dir is not None:
                sess.run_dir = str(run_dir)
                sess.stage_speech("I’ve started the browser run and I’m working through the page now.")

        if run_dir is not None:
            steps = _read_browser_log_safely(run_dir / "log.json")
            if steps:
                latest_step_num = int(steps[-1].get("step") or len(steps))
                if latest_step_num > sess.last_progress_step:
                    sess.last_progress_step = latest_step_num
                    progress = _progress_text_from_step(steps[-1])
                    if progress:
                        sess.stage_speech(progress)

                note = _extract_note_from_steps(steps)
                if note:
                    answer = _compact_for_voice(note, max_chars=int(os.environ.get("BROWSE_ON_CALL_VOICE_MAX_CHARS", "1400")))
                    sess.final_answer = answer
                    sess.stage_speech(f"I found it. The video description says: {answer}")
                    final_reported = True
                    print(
                        f"[browse-on-call:{call_id}] final answer staged from {run_dir / 'log.json'}",
                        flush=True,
                    )
                    return

        time.sleep(0.8)


def _browse_subprocess_stall_watchdog(
    *,
    call_id: str,
    proc: subprocess.Popen,
    process_start_ts: float,
    known_run_names: set[str],
    pump_state: dict[str, Any],
) -> None:
    """Speak if the browser child is slow to start or never creates a run folder."""
    warned_loading = False
    warned_stuck = False

    while proc.poll() is None and not pump_state.get("failed"):
        sess = _get_session(call_id)
        if sess is None:
            return

        elapsed = time.time() - process_start_ts
        if not pump_state.get("first_line") and elapsed >= 20 and not warned_loading:
            warned_loading = True
            sess.stage_speech(
                "The browser software is still loading on my computer. Give me about half a minute."
            )

        run_dir = _find_newest_run_dir(process_start_ts, known_run_names)
        if run_dir is None and elapsed >= 45 and not warned_stuck:
            warned_stuck = True
            sess.stage_speech(
                "I'm having trouble starting the browser. My vision server may be offline — "
                "check that MiMo is running and reachable, then call again."
            )
        elif run_dir is not None and sess.run_dir is None:
            sess.run_dir = str(run_dir)

        time.sleep(2.0)


# Greeting-only / "are you there?" utterances we DON'T want to lock in as a goal.
_SMALLTALK_PATTERNS = (
    "can you hear",
    "are you there",
    "hi.",
    "hello.",
    "hello?",
    "hi?",
    "hey.",
    "hey?",
    "are you listening",
    "you there",
    "is this on",
)


def _looks_like_smalltalk(text: str) -> bool:
    """Recognize 'hi / can you hear me' style openers so we don't store them as the goal."""
    t = (text or "").lower().strip()
    if not t:
        return False
    if len(t.split()) <= 7:
        if any(p in t for p in _SMALLTALK_PATTERNS):
            return True
        # Pure greetings.
        bare = t.strip(".,!?'\"")
        if bare in {"hi", "hello", "hey", "yo"}:
            return True
    return False


def _format_readback(goal: str, attempt: int) -> str:
    """Phrase we say back so the caller can confirm or restate their goal."""
    quoted = goal.strip()
    if len(quoted) > 220:
        quoted = quoted[:217].rstrip() + "..."
    if attempt <= 1:
        return (
            f"Got it. You want me to: {quoted}. "
            "Say GO to start, or just say it again to fix it."
        )
    return (
        f"OK, updated. You want me to: {quoted}. "
        "Say GO to start, or just say it again."
    )


def _spawn_browse_on_call_subprocess(
    *,
    call_id: str,
    goal: str,
    local_base: str,
    provider: str,
    interaction_mode: str,
    memory_mode: str,
    demo_overlay: bool,
) -> None:
    """Spawn `python cosmic-browser-use/main.py ... --ask-user-bridge-url ...` for an inbound call.

    Runs in a background daemon thread so the webhook response is never delayed.
    Output is pumped into uvicorn's stdout with a [browse-on-call:<call_id>] prefix so
    you can watch the agent live in the same terminal that runs webhook_server.
    """

    bridge_url = f"{local_base.rstrip('/')}/control/session/{call_id}"
    runs_dir = COSMIC_DIR / "runs"
    try:
        known_run_names = {p.name for p in runs_dir.iterdir() if p.is_dir()}
    except Exception:
        known_run_names = set()

    cmd = [
        sys.executable,
        "-u",
        str(COSMIC_DIR / "main.py"),
        "--goal", goal,
        "--provider", provider,
        "--interaction-mode", interaction_mode,
        "--memory-mode", memory_mode,
        "--ask-user-bridge-url", bridge_url,
    ]
    if demo_overlay:
        cmd.append("--demo-overlay")

    def _run() -> None:
        print(
            f"[browse-on-call:{call_id}] spawning: {' '.join(cmd)}",
            flush=True,
        )
        # Force UTF-8 in the child so emojis in cosmic-browser-use/main.py don't blow
        # up under Windows cp1252 when stdout is piped (not a TTY).
        child_env = os.environ.copy()
        child_env["PYTHONIOENCODING"] = "utf-8"
        child_env["PYTHONUTF8"] = "1"
        process_start_ts = time.time()
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(COSMIC_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=child_env,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[browse-on-call:{call_id}] FAILED to spawn: {exc!r}", flush=True)
            try:
                # Try to apologize on the call so the caller doesn't sit in silence.
                import requests

                requests.post(
                    f"{local_base.rstrip('/')}/control/session/{call_id}/say",
                    json={"text": "Sorry, I couldn't start the browsing task on my end. I'll end the call now."},
                    timeout=5,
                )
            except Exception:
                pass
            return

        pump_state: dict[str, Any] = {"first_line": False, "failed": False}

        def _pump() -> None:
            try:
                assert proc.stdout is not None
                for line in iter(proc.stdout.readline, ""):
                    if not line:
                        break
                    pump_state["first_line"] = True
                    print(f"[browse-on-call:{call_id}] {line}", end="", flush=True)
                    if _subprocess_line_is_failure(line):
                        pump_state["failed"] = True
                        _stage_on_call(
                            call_id,
                            "Sorry — my vision server isn't responding, so I can't browse right now. "
                            "Please check that MiMo is online, then try again.",
                        )
                        break
            except Exception:
                pass

        pump = threading.Thread(target=_pump, daemon=True)
        pump.start()
        stall = threading.Thread(
            target=_browse_subprocess_stall_watchdog,
            kwargs={
                "call_id": call_id,
                "proc": proc,
                "process_start_ts": process_start_ts,
                "known_run_names": known_run_names,
                "pump_state": pump_state,
            },
            daemon=True,
        )
        stall.start()
        monitor = threading.Thread(
            target=_monitor_browser_run_for_call,
            kwargs={
                "call_id": call_id,
                "proc": proc,
                "known_run_names": known_run_names,
                "process_start_ts": process_start_ts,
            },
            daemon=True,
        )
        monitor.start()
        rc = proc.wait()
        pump.join(timeout=2.0)
        monitor.join(timeout=2.0)
        stall.join(timeout=2.0)
        print(f"[browse-on-call:{call_id}] subprocess exit_code={rc}", flush=True)

        # Speak a wrap-up line and release the session.
        try:
            import requests

            sess = _get_session(call_id)
            if sess and rc != 0 and not sess.final_answer and not pump_state.get("failed"):
                _stage_on_call(
                    call_id,
                    "The browser task stopped before it could finish. "
                    "MiMo or the browser may be unavailable on this machine.",
                )
            if sess and sess.final_answer:
                wrap = "That’s the result I found. I’ll keep the call open for a moment in case you need anything else."
            elif rc == 0:
                wrap = "The browser task finished. I’ll keep the call open for a moment in case you need anything else."
            else:
                wrap = "The browser task stopped before I could report a final answer."
            requests.post(
                f"{local_base.rstrip('/')}/control/session/{call_id}/say",
                json={"text": wrap},
                timeout=5,
            )
            # Do not immediately close: the caller may ask follow-up questions,
            # and the webhook session branch can now answer from final_answer/status.
            if os.environ.get("BROWSE_ON_CALL_AUTO_CLOSE", "0").strip().lower() in {"1", "true", "yes", "on"}:
                time.sleep(1.0)
                requests.post(
                    f"{local_base.rstrip('/')}/control/session/{call_id}/close",
                    timeout=5,
                )
        except Exception as exc:  # noqa: BLE001
            print(f"[browse-on-call:{call_id}] cleanup failed: {exc!r}", flush=True)

    threading.Thread(target=_run, name=f"browse-on-call-{call_id}", daemon=True).start()


async def _handle_voice_webhook(payload: dict[str, Any]) -> Response:
    data = payload.get("data") or {}
    call_id, real_call_id = _voice_call_key(data)
    transcript = data.get("transcript", "")
    confidence = data.get("confidence")
    status = data.get("status")
    direction = data.get("direction")
    history = payload.get("recentHistory") or []

    print(
        f"[voice/webhook] call={call_id} real_call={real_call_id} status={status} direction={direction} "
        f"confidence={confidence} transcript={transcript!r}",
        flush=True,
    )

    if real_call_id:
        start_transcript_stream_once(_client, real_call_id)

    # ----- Browse-on-call (inbound, env-gated) -----
    # State machine for inbound calls:
    #   no candidate yet  -> capture first non-empty transcript, read it back, ask "yes?"
    #   have candidate    -> short "yes" => spawn subprocess and switch to session bridge
    #                     -> anything else => replace candidate (user is restating), ask again
    # This avoids the failure mode where AgentPhone fires the first turn boundary
    # while the user is still speaking and we lock in a truncated goal.
    session = _get_session(call_id) if call_id else None
    if (
        call_id
        and session is None
        and _browse_on_call_enabled()
        and (direction or "").lower() == "inbound"
    ):
        utter = (transcript or "").strip()
        data_block = payload.get("data") or {}
        from_number = data_block.get("from") or data_block.get("fromNumber") or ""
        local_base = os.environ.get("WEBHOOK_LOCAL_BASE", "http://127.0.0.1:9876").rstrip("/")
        pending = _get_or_create_pending(call_id, from_number)

        def _stream(text: str, interim: str = "One second.") -> StreamingResponse:
            async def gen() -> Any:
                yield json.dumps({"text": interim, "interim": True}) + "\n"
                yield json.dumps({"text": text}) + "\n"

            return StreamingResponse(gen(), media_type="application/x-ndjson")

        # Empty transcript — re-prompt for the goal (don't burn a confirmation attempt).
        if not utter:
            if pending.candidate_goal:
                msg = _format_readback(pending.candidate_goal, pending.confirm_attempts + 1)
                print(
                    f"[browse-on-call] call={call_id} empty transcript while pending; re-asking",
                    flush=True,
                )
            else:
                msg = "Sorry, I didn't catch that. What would you like me to browse for you?"
                print(
                    f"[browse-on-call] call={call_id} empty first transcript; re-prompting",
                    flush=True,
                )
            return _stream(msg, interim="I'm listening.")

        # Preferred path: Kimi acts as the phone-intake assistant. It decides
        # whether this is chat, a goal, a correction, a confirmation, or cancel.
        if os.environ.get("BROWSE_ON_CALL_KIMI_INTAKE", "1").strip().lower() in {"1", "true", "yes", "on"}:
            try:
                decision = _browse_intake_decision(pending, utter)
                action = decision["action"]
                goal = decision.get("goal")
                reply = decision["reply"]
                print(
                    f"[browse-on-call] kimi intake call={call_id} action={action!r} "
                    f"goal={goal!r} reason={decision.get('reason')!r} utter={utter!r}",
                    flush=True,
                )

                pending.turns.append({"role": "user", "content": utter})

                if action == "start":
                    start_goal = goal or pending.candidate_goal
                    if not start_goal:
                        pending.turns.append({"role": "assistant", "content": "What should I browse?"})
                        return _stream("What would you like me to browse for you?", interim="I'm listening.")

                    _drop_pending(call_id)
                    with _call_sessions_lock:
                        sess = _CallSession(call_id=call_id, to_number=from_number)
                        _call_sessions[call_id] = sess

                    print(
                        f"[browse-on-call] call={call_id} from={from_number!r} KIMI START GOAL: {start_goal!r}",
                        flush=True,
                    )
                    _spawn_browse_on_call_subprocess(
                        call_id=call_id,
                        goal=start_goal,
                        local_base=local_base,
                        provider=os.environ.get("BROWSE_ON_CALL_PROVIDER", "fireworks_kimi"),
                        interaction_mode=os.environ.get("BROWSE_ON_CALL_INTERACTION_MODE", "vision"),
                        memory_mode=os.environ.get("BROWSE_ON_CALL_MEMORY_MODE", "recall"),
                        demo_overlay=os.environ.get("BROWSE_ON_CALL_DEMO_OVERLAY", "1").strip().lower()
                        in {"1", "true", "yes", "y", "on"},
                    )
                    return _stream(reply, interim="Got it.")

                if action == "update_goal" and goal:
                    pending.candidate_goal = goal
                    pending.confirm_attempts += 1
                    pending.last_transcript = utter
                    pending.turns.append({"role": "assistant", "content": reply})
                    return _stream(reply)

                if action == "cancel":
                    pending.candidate_goal = None
                    pending.confirm_attempts = 0
                    pending.last_transcript = utter
                    pending.turns.append({"role": "assistant", "content": reply})
                    return _stream(reply, interim="No problem.")

                # chat / ask_goal / unclear update without a goal
                pending.last_transcript = utter
                pending.turns.append({"role": "assistant", "content": reply})
                return _stream(reply, interim="Okay.")
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[browse-on-call] kimi intake failed call={call_id}: {exc!r}; falling back to rules",
                    flush=True,
                )

        # Drop literal-duplicate STT updates (AgentPhone re-fires the same line).
        if pending.last_transcript == utter:
            print(
                f"[browse-on-call] call={call_id} duplicate transcript; holding",
                flush=True,
            )
            if pending.candidate_goal:
                return _stream(
                    _format_readback(pending.candidate_goal, pending.confirm_attempts + 1),
                    interim="One sec.",
                )
            return _stream("Sorry, what would you like me to browse for you?", interim="I'm listening.")

        pending.last_transcript = utter

        # CASE A: caller confirmed the previous candidate -> spawn subprocess.
        if pending.candidate_goal and _looks_like_confirm(utter):
            goal = pending.candidate_goal
            _drop_pending(call_id)

            with _call_sessions_lock:
                sess = _CallSession(call_id=call_id, to_number=from_number)
                _call_sessions[call_id] = sess

            print(
                f"[browse-on-call] call={call_id} from={from_number!r} CONFIRMED GOAL: {goal!r}",
                flush=True,
            )

            _spawn_browse_on_call_subprocess(
                call_id=call_id,
                goal=goal,
                local_base=local_base,
                provider=os.environ.get("BROWSE_ON_CALL_PROVIDER", "fireworks_kimi"),
                interaction_mode=os.environ.get("BROWSE_ON_CALL_INTERACTION_MODE", "vision"),
                memory_mode=os.environ.get("BROWSE_ON_CALL_MEMORY_MODE", "recall"),
                demo_overlay=os.environ.get("BROWSE_ON_CALL_DEMO_OVERLAY", "1").strip().lower()
                in {"1", "true", "yes", "y", "on"},
            )

            return _stream(
                "Got it. Starting now — I'll check in if I need anything. Just stay on the line.",
                interim="Got it.",
            )

        # CASE B-smalltalk: caller said "hi / can you hear me / hello" — don't store
        # that as the goal; re-prompt for the actual task.
        if pending.candidate_goal is None and _looks_like_smalltalk(utter):
            print(
                f"[browse-on-call] call={call_id} small-talk ({utter!r}); re-prompting for goal",
                flush=True,
            )
            return _stream(
                "Yes, I can hear you. What would you like me to browse for you? "
                "Tell me, then say GO to start.",
                interim="Yes.",
            )

        # CASE B-filler: once we already have a goal, do not let short
        # conversational fragments ("so", "yeah", "nothing") wipe it out. This
        # was the source of the loop where the agent kept reading back "So" /
        # "Good. Just go." instead of starting.
        if pending.candidate_goal and _looks_like_filler_only(utter):
            print(
                f"[browse-on-call] call={call_id} filler ({utter!r}); keeping candidate={pending.candidate_goal!r}",
                flush=True,
            )
            return _stream(
                _format_readback(pending.candidate_goal, pending.confirm_attempts + 1),
                interim="One sec.",
            )

        # CASE B: new or replacement candidate goal -> normalize it, then read
        # back the *actual* browser goal we will send to cosmic-browser-use.
        normalized_goal = _normalize_spoken_goal(utter)
        pending.candidate_goal = normalized_goal
        pending.confirm_attempts += 1
        print(
            f"[browse-on-call] call={call_id} candidate (attempt {pending.confirm_attempts}): raw={utter!r} normalized={normalized_goal!r}",
            flush=True,
        )
        return _stream(_format_readback(normalized_goal, pending.confirm_attempts))

    # ----- Session bridge: external owner of this call (call_to_browse.py or above) -----
    if session is not None:
        user_text = (transcript or "").strip()
        if user_text:
            try:
                session.reply_queue.put_nowait(user_text)
            except asyncio.QueueFull:  # pragma: no cover — unbounded queue
                print(f"[voice/webhook] session reply queue full call={call_id}", flush=True)
            session.last_activity = time.time()

        staged = session.take_speech()
        if staged:
            reply_str = staged
        else:
            low = user_text.lower()
            if session.final_answer and any(p in low for p in ("done", "finish", "result", "answer", "what did", "found")):
                reply_str = f"Yes, I found it. The answer is: {session.final_answer}"
            elif any(p in low for p in ("done", "finish", "status", "where are", "progress", "are you")):
                reply_str = session.status_text or session.holding_phrase
            else:
                reply_str = session.status_text or session.holding_phrase
            # Avoid repeating the exact same line on every duplicate webhook turn.
            if not staged and reply_str and reply_str == session.last_spoken:
                reply_str = session.holding_phrase
        print(
            f"[voice/webhook] (session) call={call_id} "
            f"{'speaking-staged' if staged else 'holding'} -> reply={reply_str!r}",
            flush=True,
        )

        async def ndjson_session() -> Any:
            yield json.dumps({"text": "One second.", "interim": True}) + "\n"
            yield json.dumps({"text": reply_str}) + "\n"

        return StreamingResponse(ndjson_session(), media_type="application/x-ndjson")

    try:
        reply = agent_response(transcript, history, payload) or DEFAULT_AGENT_RESPONSE
    except Exception as exc:  # noqa: BLE001 — never let a coding error make the caller hear silence
        print(f"[voice/webhook] agent_response error: {exc!r}; using default", flush=True)
        reply = DEFAULT_AGENT_RESPONSE

    reply_str = str(reply)
    print(f"[voice/webhook] -> reply={reply_str!r}", flush=True)

    async def ndjson() -> Any:
        # Interim filler so the caller doesn't hear silence while we send the reply.
        yield json.dumps({"text": "Got it.", "interim": True}) + "\n"
        yield json.dumps({"text": reply_str}) + "\n"

    return StreamingResponse(ndjson(), media_type="application/x-ndjson")
