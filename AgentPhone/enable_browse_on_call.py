"""
One-shot setup for "call the agent and tell it what to browse".

What it does:
  • Writes BROWSE_ON_CALL_ENABLED=1 into AgentPhone/.env (so webhook_server.py
    treats the first user utterance on an inbound call as the browsing goal).
  • Updates the agent's `begin_message` (the line spoken when you pick up) to
    "Hi! What would you like me to browse for you?"
  • Forces voice_mode='webhook' so your code (this server) controls replies.

After running this once, restart `uvicorn webhook_server:app` so it re-reads
.env, then just dial your AgentPhone number from your cell phone and say the
goal out loud.

Usage:
  cd AgentPhone
  python enable_browse_on_call.py
  python enable_browse_on_call.py --greeting "Hey! What can I browse for you?"
  python enable_browse_on_call.py --disable
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from agentphone import AgentPhone
from dotenv import load_dotenv

from agent_resolve import resolve_agent_id

ROOT = Path(__file__).resolve().parent
ENV_PATH = ROOT / ".env"
DEFAULT_GREETING = "Hi! What would you like me to browse for you?"


def _read_env_lines() -> list[str]:
    if not ENV_PATH.exists():
        return []
    return ENV_PATH.read_text(encoding="utf-8").splitlines()


def _write_env_lines(lines: list[str]) -> None:
    ENV_PATH.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _set_env_key(key: str, value: str) -> tuple[bool, str | None]:
    """Set KEY=VALUE in .env (preserving existing content). Returns (changed, previous_value)."""
    lines = _read_env_lines()
    prefix = f"{key}="
    new_line = f"{key}={value}"
    previous: str | None = None
    found = False
    out: list[str] = []
    for ln in lines:
        stripped = ln.lstrip()
        # Skip pure comments / blanks but keep them.
        if stripped.startswith("#") or not stripped:
            out.append(ln)
            continue
        if stripped.startswith(prefix):
            previous = stripped.split("=", 1)[1] if "=" in stripped else None
            out.append(new_line)
            found = True
        else:
            out.append(ln)
    if not found:
        out.append(new_line)
    _write_env_lines(out)
    return (previous != value, previous)


def main() -> int:
    parser = argparse.ArgumentParser(description="Enable browse-on-call (or disable with --disable).")
    parser.add_argument(
        "--greeting",
        default=DEFAULT_GREETING,
        help=f"Greeting the agent says when you pick up. Default: {DEFAULT_GREETING!r}",
    )
    parser.add_argument(
        "--disable",
        action="store_true",
        help="Set BROWSE_ON_CALL_ENABLED=0 in .env (greeting + voice_mode unchanged).",
    )
    args = parser.parse_args()

    load_dotenv(ENV_PATH)
    api_key = os.environ.get("AGENTPHONE_API_KEY", "").strip()
    if not api_key:
        print("Set AGENTPHONE_API_KEY in AgentPhone/.env first.", file=sys.stderr)
        return 1

    if args.disable:
        changed, prev = _set_env_key("BROWSE_ON_CALL_ENABLED", "0")
        print(f"BROWSE_ON_CALL_ENABLED -> 0 (was {prev!r}, changed={changed})")
        print("Done. Restart uvicorn for the change to take effect.")
        return 0

    client = AgentPhone(api_key=api_key)
    try:
        agent_id = resolve_agent_id(client)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1

    agent = client.agents.get(agent_id)
    current_voice = (agent.voice_mode or "").lower()
    current_begin = getattr(agent, "begin_message", None) or ""

    print(f"Agent: {agent_id} ({agent.name})")
    print(f"  voice_mode: {current_voice!r}")
    print(f"  begin_message: {current_begin!r}")

    needs_voice_update = current_voice != "webhook"
    needs_greeting_update = (current_begin or "").strip() != args.greeting.strip()

    if needs_voice_update or needs_greeting_update:
        update_kwargs: dict = {}
        if needs_voice_update:
            update_kwargs["voice_mode"] = "webhook"
        if needs_greeting_update:
            update_kwargs["begin_message"] = args.greeting
        print(f"Updating agent: {list(update_kwargs.keys())} ...")
        client.agents.update(agent_id, **update_kwargs)
        print("Agent updated.")
    else:
        print("Agent already in webhook mode with the requested greeting — nothing to update there.")

    changed, prev = _set_env_key("BROWSE_ON_CALL_ENABLED", "1")
    print(f"BROWSE_ON_CALL_ENABLED -> 1 (was {prev!r}, changed={changed})")

    print(
        "\nAll set. Next steps:\n"
        "  1) Restart uvicorn so it picks up the new env:\n"
        "       (Ctrl+C the running uvicorn) and run:\n"
        "       uvicorn webhook_server:app --host 127.0.0.1 --port 9876\n"
        "  2) Make sure ngrok is up and your project webhook is configured (`provision.py`).\n"
        "  3) Dial your AgentPhone number from your phone, hear the greeting,\n"
        "     and SAY THE GOAL OUT LOUD. The browser agent will start automatically.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
