"""
Place an outbound voice call.

  • FROM (caller ID) = your agent's AgentPhone line (+14786068471, etc.). It is chosen
    automatically from AGENTPHONE_NUMBER_ID / agentphone_state.json, or the agent's default line.
  • TO (who rings) = the number you pass on the command line — that is NOT the agent's number;
    it is whoever you want to simulate calling (often your own mobile in E.164 for a test).

  • Live words: in an interactive terminal this script subscribes to the call transcript SSE after
    placing the call, so [user] / [agent] turns print here as they happen. Note: the platform emits
    one SSE `turn` per finished speech segment (not per word), so a 60s call with 6 turns shows ~6
    lines spread across the call. For sub-turn STT chunks run webhook_server.py (uvicorn).
    Use --debug-sse to also print every raw line with timestamps.

Uses webhook mode when you omit --hosted (your webhook must be reachable for the conversation).

Usage:
  cd AgentPhone
  python make_call.py --list-from-numbers
  python make_call.py --dry-run +1YOURCELL...
  python make_call.py +1YOURCELL...
  python make_call.py +1YOURCELL... --no-stream   # JSON only, then exit
  python make_call.py +1YOURCELL... --listen-only # webhook reply is whatever agent_response() returns
  python make_call.py +15551234567 --greeting "Hi, this is a quick check-in call."
  python make_call.py +15551234567 --hosted --prompt "You are a friendly assistant. Keep replies under two sentences."
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

from agentphone import AgentPhone
from dotenv import load_dotenv

from agent_resolve import resolve_agent_id, resolve_number_id
from call_sse_transcript import iter_transcript_sse_events

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


def _number_id(client: AgentPhone, agent_id: str, override: str | None) -> str | None:
    if override:
        return override.strip() or None
    return resolve_number_id(client, agent_id)


def _ensure_webhook_voice_mode(client: AgentPhone, agent_id: str) -> bool:
    """Verify the agent uses webhook voice mode and a project webhook URL is set.

    Returns True if the call can proceed with code-defined replies; False (with a printed
    diagnostic) if AgentPhone has nowhere to deliver voice events. Flipping voiceMode is
    persistent on the AgentPhone side — future calls for this agent will keep using the
    webhook until you change it back (e.g. via the dashboard or `client.agents.update`).
    """
    agent = client.agents.get(agent_id)
    current = (agent.voice_mode or "").lower()
    if current != "webhook":
        print(
            f"[listen-only] Agent {agent_id} voice_mode={current or '(unset)'!r}; "
            "switching to 'webhook' so your code replies are used (persists for future calls).",
            flush=True,
        )
        try:
            client.agents.update(agent_id, voice_mode="webhook")
        except Exception as exc:  # noqa: BLE001
            print(f"[listen-only] Failed to switch voice_mode: {exc}", file=sys.stderr)
            return False
    else:
        print(f"[listen-only] Agent voice_mode='webhook' already.", flush=True)

    try:
        webhook = client.webhooks.get()
    except Exception as exc:  # noqa: BLE001
        print(f"[listen-only] Could not read project webhook: {exc}", file=sys.stderr)
        return False

    if webhook is None:
        print(
            "\n[listen-only] ERROR: no project webhook configured.\n"
            "  AgentPhone has nowhere to POST voice events, so agent_response() will never run.\n"
            "\n"
            "  One-time setup:\n"
            "    1) Run the webhook server:   uvicorn webhook_server:app --host 0.0.0.0 --port 8765\n"
            "    2) Expose it:                ngrok http 8765\n"
            "    3) Put the HTTPS URL in AgentPhone/.env as WEBHOOK_PUBLIC_BASE=https://...\n"
            "    4) Register the webhook:     python provision.py\n"
            "  Then retry: python make_call.py --listen-only +1...\n",
            file=sys.stderr,
        )
        return False

    print(f"[listen-only] Project webhook -> {webhook.url}", flush=True)
    return True


_T0 = time.monotonic()


def _ts() -> str:
    """Wall clock + offset since process start, e.g. '17:19:42.731 t+12.3s'."""
    now = time.time()
    lt = time.localtime(now)
    ms = int((now - int(now)) * 1000)
    offset = time.monotonic() - _T0
    return f"{time.strftime('%H:%M:%S', lt)}.{ms:03d} t+{offset:5.1f}s"


def _poll_transcript_rows(
    api_key: str,
    call_id: str,
    stop: threading.Event,
    seen_ids: set[str],
    interval: float = 0.6,
) -> None:
    """Poll GET /v1/calls/{id} for transcript rows on a separate HTTP client (does not share Session with SSE)."""
    poll_client = AgentPhone(api_key=api_key)
    while True:
        if stop.is_set():
            return
        try:
            call = poll_client.calls.get(call_id)
            for row in call.transcripts:
                if row.id in seen_ids:
                    continue
                seen_ids.add(row.id)
                bits: list[str] = []
                if row.transcript:
                    bits.append(f"transcript={row.transcript!r}")
                if row.response:
                    bits.append(f"response={row.response!r}")
                if bits:
                    print(f"{_ts()} [poll] id={row.id} {' | '.join(bits)}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"{_ts()} [poll] error: {exc}", file=sys.stderr, flush=True)
        if stop.wait(interval):
            return


def _follow_transcript(
    client: AgentPhone,
    call_id: str,
    *,
    use_poll: bool,
    api_key: str,
    debug_sse: bool,
) -> None:
    """SSE transcript (low-buffering) + optional REST polling, with timestamps on every line."""
    print(
        f"\n--- Live transcript (call {call_id}) — wall_clock t+elapsed_since_start ---",
        flush=True,
    )
    print(
        "    [sse]  = transcript SSE stream    [poll] = REST snapshot fallback",
        flush=True,
    )
    if debug_sse:
        print("    [raw]  = every line read from the SSE socket (debug)", flush=True)
    print("", flush=True)

    stop = threading.Event()
    seen_poll_ids: set[str] = set()
    poller: threading.Thread | None = None
    if use_poll:
        poller = threading.Thread(
            target=_poll_transcript_rows,
            args=(api_key, call_id, stop, seen_poll_ids),
            name="ap-transcript-poll",
            daemon=True,
        )
        poller.start()

    def _raw(t_mono: float, line: str) -> None:
        # Use _ts() rather than t_mono so output is consistent with the other lines.
        if debug_sse:
            print(f"{_ts()} [raw] {line}", flush=True)

    try:
        for ev, data in iter_transcript_sse_events(client, call_id, raw_callback=_raw if debug_sse else None):
            if ev == "connected":
                print(
                    f"{_ts()} [sse] connected status={data.get('status')} direction={data.get('direction')}",
                    flush=True,
                )
            elif ev == "turn" or ("role" in data and "content" in data):
                role = data.get("role", "?")
                content = data.get("content", "")
                print(f"{_ts()} [sse] [{role}] {content}", flush=True)
            elif ev == "ended":
                print(
                    f"{_ts()} [sse] ended status={data.get('status')} durationSeconds={data.get('durationSeconds')}",
                    flush=True,
                )
            else:
                print(f"{_ts()} [sse] event={ev!r} data={data!r}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"{_ts()} [sse] stream ended with error: {exc}", file=sys.stderr, flush=True)
    finally:
        stop.set()
        if poller is not None:
            poller.join(timeout=4.0)
        time.sleep(0.4)
        try:
            call = client.calls.get(call_id)
            for row in call.transcripts:
                if row.id in seen_poll_ids:
                    continue
                seen_poll_ids.add(row.id)
                bits: list[str] = []
                if row.transcript:
                    bits.append(f"transcript={row.transcript!r}")
                if row.response:
                    bits.append(f"response={row.response!r}")
                if bits:
                    print(f"{_ts()} [poll] id={row.id} {' | '.join(bits)} (final)", flush=True)
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Outbound AgentPhone voice call (FROM = agent line, TO = callee you pass).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Simulation: put YOUR phone in E.164 as CALLEE — your AgentPhone number will call you.\n"
            "Caller ID: AGENTPHONE_NUMBER_ID / provision / --list-from-numbers.\n"
            "Live speech: SSE transcript in this terminal; REST poll runs by default for extra turns.\n"
            "Use --no-poll for SSE only. Webhook STT chunks still only show on uvicorn (webhook_server.py)."
        ),
    )
    parser.add_argument(
        "callee",
        nargs="?",
        metavar="CALLEE_E164",
        help="Who receives the call (rings this number), e.g. +15551234567 or your own cell for a test",
    )
    parser.add_argument(
        "--from-number-id",
        dest="from_number_id",
        default=None,
        help="Override caller-id line (num_…). Default: AGENTPHONE_NUMBER_ID or state from provision",
    )
    parser.add_argument(
        "--list-from-numbers",
        action="store_true",
        help="Print this agent's AgentPhone lines (id + E.164) and exit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved FROM / TO and exit without placing a call",
    )
    parser.add_argument("--greeting", default=None, help="First thing spoken when they answer")
    parser.add_argument(
        "--hosted",
        action="store_true",
        help="Use built-in LLM with --prompt (no webhook for this call)",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="With --hosted: system prompt for the built-in model",
    )
    stream_g = parser.add_mutually_exclusive_group()
    stream_g.add_argument(
        "--stream",
        dest="stream_transcript",
        action="store_true",
        default=None,
        help="After placing the call, print live transcript lines (user + agent) until the call ends",
    )
    stream_g.add_argument(
        "--no-stream",
        dest="stream_transcript",
        action="store_false",
        default=None,
        help="Print call JSON only and exit (no live transcript in this terminal)",
    )
    poll_g = parser.add_mutually_exclusive_group()
    poll_g.add_argument(
        "--poll",
        dest="poll_transcript",
        action="store_true",
        default=None,
        help="While streaming, also poll GET /v1/calls/{id} for transcript rows (default when streaming)",
    )
    poll_g.add_argument(
        "--no-poll",
        dest="poll_transcript",
        action="store_false",
        default=None,
        help="SSE only; do not poll REST for extra transcript rows",
    )
    parser.add_argument(
        "--debug-sse",
        action="store_true",
        help="Also print every raw SSE line with timestamps (diagnose buffering vs platform pacing)",
    )
    parser.add_argument(
        "--listen-only",
        action="store_true",
        help=(
            "Listen mode: capture user speech here and let webhook_server.py decide what the agent "
            "says (via agent_response() in code; default is webhook_server.DEFAULT_AGENT_RESPONSE). "
            "Mutually exclusive with --hosted. Requires `uvicorn webhook_server:app` running and "
            "reachable via the project webhook URL set by provision.py."
        ),
    )
    args = parser.parse_args()

    api_key = os.environ.get("AGENTPHONE_API_KEY", "").strip()
    if not api_key:
        print("Set AGENTPHONE_API_KEY in .env", file=sys.stderr)
        return 1

    client = AgentPhone(api_key=api_key)
    try:
        agent_id = resolve_agent_id(client)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1

    number_id = _number_id(client, agent_id, args.from_number_id)

    if args.list_from_numbers:
        agent = client.agents.get(agent_id)
        print(f"Agent {agent_id} ({agent.name}) — lines usable as caller ID (FROM):")
        for n in agent.numbers:
            mark = "  <-- current default" if number_id and n.id == number_id else ""
            print(f"  {n.id}\t{n.phone_number}{mark}")
        if not agent.numbers:
            print("  (no numbers attached — run provision.py)")
        return 0

    if not args.callee:
        parser.error("CALLEE_E164 is required unless you use --list-from-numbers")

    if args.hosted and not args.prompt:
        print("--hosted requires --prompt", file=sys.stderr)
        return 1
    if args.listen_only and args.hosted:
        print("--listen-only and --hosted are mutually exclusive.", file=sys.stderr)
        return 1
    if args.listen_only:
        if not _ensure_webhook_voice_mode(client, agent_id):
            return 1
        try:
            from webhook_server import DEFAULT_AGENT_RESPONSE, agent_response

            sample = agent_response("(example user speech)", [], {})
        except Exception as exc:  # noqa: BLE001
            sample = f"(could not import webhook_server.agent_response: {exc})"
            DEFAULT_AGENT_RESPONSE = "(unknown — fallback is built into webhook_server.py)"  # type: ignore[assignment]
        print(
            "[listen-only] code-defined replies will run for this call:\n"
            "             User speech: streamed below as [sse] [user] / [poll] transcript=... lines.\n"
            "             Agent reply: returned by webhook_server.agent_response() (edit it in code).\n"
            f"             DEFAULT_AGENT_RESPONSE = {DEFAULT_AGENT_RESPONSE!r}\n"
            f"             agent_response('(example user speech)') -> {sample!r}\n"
            "             webhook_server.py must be running and reachable at the project webhook URL.",
            flush=True,
        )

    agent = client.agents.get(agent_id)
    from_lines = {n.id: n.phone_number for n in agent.numbers}
    resolved_from = from_lines.get(number_id) if number_id else None
    if number_id and resolved_from is None:
        print(
            f"Warning: --from-number-id / AGENTPHONE_NUMBER_ID {number_id!r} not on this agent; "
            "API may pick a default line.",
            file=sys.stderr,
        )
    elif not number_id and agent.numbers:
        resolved_from = agent.numbers[0].phone_number
        number_id = agent.numbers[0].id

    if args.dry_run:
        print("Dry run — no call placed.")
        print(f"  FROM (caller ID): {resolved_from or '(API default)'}  number_id={number_id!r}")
        print(f"  TO (rings):       {args.callee}")
        return 0

    kwargs: dict = {
        "agent_id": agent_id,
        "to_number": args.callee,
        "initial_greeting": args.greeting,
        "from_number_id": number_id,
    }
    if args.hosted:
        kwargs["system_prompt"] = args.prompt

    call = client.calls.make(**kwargs)
    out = {
        "id": call.id,
        "status": call.status,
        "direction": call.direction,
        "fromNumber": call.from_number,
        "toNumber": call.to_number,
    }
    print(json.dumps(out, indent=2))

    if args.stream_transcript is None:
        follow = sys.stdout.isatty()
    else:
        follow = args.stream_transcript
    if follow:
        if args.poll_transcript is None:
            use_poll = True
        else:
            use_poll = args.poll_transcript
        _follow_transcript(
            client,
            call.id,
            use_poll=use_poll,
            api_key=api_key,
            debug_sse=args.debug_sse,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
