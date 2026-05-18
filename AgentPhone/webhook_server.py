"""
Local webhook server for AgentPhone (SMS + voice in webhook mode).

Run (after `provision.py` and exposing HTTPS, e.g. ngrok):
  cd AgentPhone
  uvicorn webhook_server:app --host 0.0.0.0 --port 8765

Voice: logs each STT segment from webhooks and starts one SSE transcript stream per call
(see https://docs.agentphone.ai/documentation/guides/calls#stream-transcript-sse ).
"""

from __future__ import annotations

import json
import os
import threading
from collections import deque
from typing import Any

from agentphone import AgentPhone
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, Response, StreamingResponse

import config
from agent_resolve import resolve_agent_id, resolve_number_id
from security import verify_agentphone_webhook
from transcript_sse import start_transcript_stream_once

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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


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


async def _handle_voice_webhook(payload: dict[str, Any]) -> Response:
    data = payload.get("data") or {}
    call_id = data.get("callId")
    transcript = data.get("transcript", "")
    confidence = data.get("confidence")
    status = data.get("status")
    direction = data.get("direction")
    history = payload.get("recentHistory") or []

    print(
        f"[voice/webhook] call={call_id} status={status} direction={direction} "
        f"confidence={confidence} transcript={transcript!r}",
        flush=True,
    )

    if call_id:
        start_transcript_stream_once(_client, call_id)

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
