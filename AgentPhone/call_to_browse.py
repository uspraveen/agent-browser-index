"""
call_to_browse — drive cosmic-browser-use from a live AgentPhone voice call.

How it works (high level):
  1. Place an outbound call to you (FROM = agent's AgentPhone line, TO = your cell).
  2. Register a "call session" with the local webhook server. While this session
     is active, the webhook server speaks whatever the browser agent stages and
     pushes every STT transcript onto a reply queue for the browser agent.
  3. Launch `python cosmic-browser-use/main.py --goal ... --ask-user-bridge-url ...`
     in a subprocess. Whenever the browser agent uses its AskUser tool, the
     question is POSTed to the bridge, spoken on the call the next time the user
     talks, and the next thing the user says becomes the AskUser reply.
  4. On finish, speak a completion phrase, close the session, end the call.

Prereqs:
  • `uvicorn webhook_server:app --host 127.0.0.1 --port 9876` running (this script
    talks to it over loopback for the control plane, and AgentPhone hits it via
    your ngrok / WEBHOOK_PUBLIC_BASE URL).
  • Project webhook configured (`provision.py`) and agent voiceMode = "webhook".
  • cosmic-browser-use deps installed (MiMo reachable, provider key set, etc.).

Usage:
  cd AgentPhone
  python call_to_browse.py --goal "Get the YouTube video description for the karuppu movie god more song offl video"
  python call_to_browse.py --goal "..." --to +12153079021 --provider fireworks_kimi --interaction-mode vision --memory-mode recall

The browser agent's own flags can be passed verbatim — anything after `--` is
forwarded as-is. Example:
  python call_to_browse.py --goal "..." -- --headless --steps 50
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import requests
from agentphone import AgentPhone
from dotenv import load_dotenv

from agent_resolve import resolve_agent_id, resolve_number_id
from make_call import _ensure_webhook_voice_mode  # reuse the persistent voice-mode + webhook check

ROOT = Path(__file__).resolve().parent
COSMIC_DIR = (ROOT.parent / "cosmic-browser-use").resolve()
load_dotenv(ROOT / ".env")


# --- knobs --------------------------------------------------------------------

DEFAULT_LOCAL_BASE = os.getenv("WEBHOOK_LOCAL_BASE", "http://127.0.0.1:9876").rstrip("/")
# Same number you've been testing with; override on the command line if needed.
DEFAULT_CALLEE = os.getenv("AGENTPHONE_DEFAULT_CALLEE", "+12153079021")
DEFAULT_GREETING = (
    "Hi! I'm your AgentPhone browsing assistant. I'll start the task now and "
    "check in with you on this call if I need anything. Just stay on the line."
)
COMPLETION_SAY = (
    "All done with the browsing task — I'll wrap up the call. Talk to you later."
)
ABORT_SAY = (
    "Something went wrong with the browsing task on my end. I'll end the call now."
)


# --- helpers ------------------------------------------------------------------


_T0 = time.monotonic()


def _ts() -> str:
    now = time.time()
    lt = time.localtime(now)
    ms = int((now - int(now)) * 1000)
    return f"{time.strftime('%H:%M:%S', lt)}.{ms:03d} t+{time.monotonic() - _T0:5.1f}s"


def _log(prefix: str, msg: str) -> None:
    print(f"{_ts()} [{prefix}] {msg}", flush=True)


def _control_health(local_base: str) -> bool:
    try:
        r = requests.get(f"{local_base}/health", timeout=3.0)
        return r.status_code == 200
    except Exception as exc:  # noqa: BLE001
        _log("control", f"health check failed @ {local_base}: {exc}")
        return False


def _register_session(local_base: str, call_id: str, to_number: str) -> None:
    r = requests.post(
        f"{local_base}/control/session",
        json={"call_id": call_id, "to_number": to_number},
        timeout=10.0,
    )
    r.raise_for_status()
    _log("control", f"session registered: {r.json()}")


def _close_session(local_base: str, call_id: str) -> None:
    try:
        r = requests.post(f"{local_base}/control/session/{call_id}/close", timeout=10.0)
        _log("control", f"session close: {r.status_code} {r.text.strip()}")
    except Exception as exc:  # noqa: BLE001
        _log("control", f"session close failed: {exc}")


def _say_on_call(local_base: str, call_id: str, text: str) -> None:
    try:
        r = requests.post(
            f"{local_base}/control/session/{call_id}/say",
            json={"text": text},
            timeout=10.0,
        )
        if r.status_code >= 400:
            _log("control", f"say failed: {r.status_code} {r.text.strip()}")
        else:
            _log("control", f"staged completion line: {text!r}")
    except Exception as exc:  # noqa: BLE001
        _log("control", f"say request failed: {exc}")


def _stream_subprocess(proc: subprocess.Popen) -> threading.Thread:
    """Tee the child's stdout/stderr to our terminal so the user can watch the agent live."""

    def _pump(pipe, prefix: str) -> None:
        try:
            for line in iter(pipe.readline, ""):
                if not line:
                    break
                sys.stdout.write(f"[{prefix}] {line}" if not line.startswith(f"[{prefix}]") else line)
                sys.stdout.flush()
        finally:
            try:
                pipe.close()
            except Exception:
                pass

    t_out = threading.Thread(target=_pump, args=(proc.stdout, "browser"), daemon=True)
    t_out.start()
    return t_out


def _build_browser_cmd(
    *,
    goal: str,
    bridge_url: str,
    extra_args: list[str],
    provider: Optional[str],
    interaction_mode: Optional[str],
    memory_mode: Optional[str],
    demo_overlay: bool,
) -> list[str]:
    """Build the `python main.py ...` command for the browser agent subprocess.

    Mirrors the user's known-good invocation:
      python main.py --provider fireworks_kimi --interaction-mode vision \
                     --memory-mode recall --goal "..." --demo-overlay
    """
    cmd: list[str] = [sys.executable, "-u", str(COSMIC_DIR / "main.py"), "--goal", goal]
    if provider:
        cmd += ["--provider", provider]
    if interaction_mode:
        cmd += ["--interaction-mode", interaction_mode]
    if memory_mode:
        cmd += ["--memory-mode", memory_mode]
    if demo_overlay:
        cmd += ["--demo-overlay"]
    cmd += ["--ask-user-bridge-url", bridge_url]
    if extra_args:
        cmd += extra_args
    return cmd


# --- main ---------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Place an AgentPhone call and let the cosmic-browser-use agent ask the caller questions over voice.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            '  python call_to_browse.py --goal "Get the YouTube video description for the karuppu movie god more song offl video"\n'
            "\n"
            "Everything after `--` is passed verbatim to cosmic-browser-use/main.py."
        ),
    )
    parser.add_argument("--goal", required=True, help="Task for the browser agent (forwarded as --goal).")
    parser.add_argument(
        "--to",
        default=DEFAULT_CALLEE,
        help=f"Phone number to call (E.164). Default {DEFAULT_CALLEE}.",
    )
    parser.add_argument(
        "--greeting",
        default=DEFAULT_GREETING,
        help="What the agent says when the call connects.",
    )
    parser.add_argument(
        "--provider",
        default="fireworks_kimi",
        help="Browser-agent LLM provider (default: fireworks_kimi).",
    )
    parser.add_argument(
        "--interaction-mode",
        default="vision",
        choices=["hybrid", "vision"],
        help="Browser-agent interaction mode (default: vision).",
    )
    parser.add_argument(
        "--memory-mode",
        default="recall",
        choices=["off", "learn", "recall", "auto"],
        help="COSMIC memory mode for the browser agent (default: recall).",
    )
    parser.add_argument(
        "--no-demo-overlay",
        action="store_true",
        help="Disable the glassy demo overlay (it's on by default for this flow).",
    )
    parser.add_argument(
        "--local-base",
        default=DEFAULT_LOCAL_BASE,
        help=f"Local URL of the running webhook server (default {DEFAULT_LOCAL_BASE}).",
    )
    parser.add_argument(
        "--end-call-on-finish",
        action="store_true",
        help="Hang up the call after the browser agent completes (default: leave the call open).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved plan (call + subprocess) without placing the call.",
    )
    parser.add_argument(
        "browser_args",
        nargs=argparse.REMAINDER,
        help="Anything after `--` is forwarded to cosmic-browser-use/main.py verbatim.",
    )
    args = parser.parse_args()

    # `nargs=REMAINDER` keeps the literal "--" as first element if present.
    extra_args = [a for a in (args.browser_args or []) if a != "--"]

    api_key = os.environ.get("AGENTPHONE_API_KEY", "").strip()
    if not api_key:
        print("Set AGENTPHONE_API_KEY in AgentPhone/.env", file=sys.stderr)
        return 1

    if not (COSMIC_DIR / "main.py").exists():
        print(
            f"cosmic-browser-use/main.py not found at {COSMIC_DIR}. "
            "Make sure the cosmic-browser-use directory is a sibling of AgentPhone/.",
            file=sys.stderr,
        )
        return 1

    # 1) Sanity: webhook server reachable on loopback?
    local_base = args.local_base.rstrip("/")
    if not _control_health(local_base):
        print(
            f"\n❌ Webhook server not reachable at {local_base}.\n"
            "   Start it first (in another terminal):\n"
            f"     cd {ROOT}\n"
            "     uvicorn webhook_server:app --host 127.0.0.1 --port 9876\n",
            file=sys.stderr,
        )
        return 1

    # 2) AgentPhone client + resolved IDs.
    client = AgentPhone(api_key=api_key)
    try:
        agent_id = resolve_agent_id(client)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1
    number_id = resolve_number_id(client, agent_id)

    # 3) Ensure the agent is in webhook voice mode and a project webhook is set.
    #    Without this, AgentPhone uses its own LLM and our `agent_response` never runs.
    if not _ensure_webhook_voice_mode(client, agent_id):
        return 1

    if args.dry_run:
        print("Dry run — no call placed, no subprocess launched.")
        print(f"  Local webhook base: {local_base}")
        print(f"  Calling FROM: agent_id={agent_id} number_id={number_id}")
        print(f"  Calling TO:   {args.to}")
        print(f"  Goal:         {args.goal!r}")
        print(f"  cosmic-browser-use dir: {COSMIC_DIR}")
        print(f"  Browser cmd:  {' '.join(_build_browser_cmd(goal=args.goal, bridge_url='<bridge>', extra_args=extra_args, provider=args.provider, interaction_mode=args.interaction_mode, memory_mode=args.memory_mode, demo_overlay=not args.no_demo_overlay))}")
        return 0

    # 4) Place the outbound call (webhook voice mode).
    _log("call", f"placing outbound call -> {args.to}")
    call = client.calls.make(
        agent_id=agent_id,
        to_number=args.to,
        initial_greeting=args.greeting,
        from_number_id=number_id,
    )
    call_id = call.id
    _log(
        "call",
        f"id={call_id} status={call.status} from={call.from_number} to={call.to_number}",
    )
    print(
        json.dumps(
            {
                "id": call.id,
                "status": call.status,
                "direction": call.direction,
                "fromNumber": call.from_number,
                "toNumber": call.to_number,
            },
            indent=2,
        )
    )

    # 5) Register the session BEFORE the call's first webhook lands.
    try:
        _register_session(local_base, call_id, args.to)
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to register session: {exc}", file=sys.stderr)
        return 1

    bridge_url = f"{local_base}/control/session/{call_id}"
    cmd = _build_browser_cmd(
        goal=args.goal,
        bridge_url=bridge_url,
        extra_args=extra_args,
        provider=args.provider,
        interaction_mode=args.interaction_mode,
        memory_mode=args.memory_mode,
        demo_overlay=not args.no_demo_overlay,
    )

    _log("browser", f"launching subprocess in {COSMIC_DIR}")
    _log("browser", f"$ {' '.join(cmd)}")

    # Force UTF-8 in the child so emojis in cosmic-browser-use/main.py don't blow
    # up under Windows cp1252 when stdout is piped (not a TTY).
    child_env = os.environ.copy()
    child_env["PYTHONIOENCODING"] = "utf-8"
    child_env["PYTHONUTF8"] = "1"
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
    pump = _stream_subprocess(proc)

    exit_code = 1
    interrupted = False
    try:
        exit_code = proc.wait()
    except KeyboardInterrupt:
        interrupted = True
        _log("call", "Ctrl+C — terminating browser subprocess and ending call...")
        try:
            proc.send_signal(signal.SIGINT)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass
        try:
            exit_code = proc.wait(timeout=10)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    finally:
        pump.join(timeout=2.0)
        _log("browser", f"subprocess exit_code={exit_code}")

        if exit_code == 0 and not interrupted:
            _say_on_call(local_base, call_id, COMPLETION_SAY)
        else:
            _say_on_call(local_base, call_id, ABORT_SAY)

        # Give the user a brief window to hear the final line before we tear down.
        time.sleep(2.0)
        _close_session(local_base, call_id)

        if args.end_call_on_finish or interrupted:
            try:
                client.calls.end(call_id)
                _log("call", f"call ended call_id={call_id}")
            except Exception as exc:  # noqa: BLE001
                _log("call", f"call end failed (already over?): {exc}")

    return 0 if exit_code == 0 else (exit_code or 1)


if __name__ == "__main__":
    raise SystemExit(main())
