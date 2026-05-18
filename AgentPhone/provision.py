"""
Provision an agent in voice webhook mode, attach a number, and register the project webhook.

Usage:
  cd AgentPhone
  copy .env.example .env   # then set AGENTPHONE_API_KEY and WEBHOOK_PUBLIC_BASE
  python provision.py

WEBHOOK_PUBLIC_BASE must be HTTPS with no trailing slash, e.g. https://abc.ngrok-free.app
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from agentphone import AgentPhone
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


def _digits(phone: str) -> str:
    return "".join(c for c in phone if c.isdigit())


def _pick_number(client: AgentPhone, agent_id: str, nums: list) -> object | None:
    """Resolve phone number resource: prefer AGENTPHONE_PHONE_NUMBER, then attached, then free, then buy."""
    preferred = os.environ.get("AGENTPHONE_PHONE_NUMBER", "").strip().strip('"').strip("'")
    if preferred:
        matches = [n for n in nums if _digits(n.phone_number) == _digits(preferred)]
        if not matches:
            print(
                f"Warning: AGENTPHONE_PHONE_NUMBER {preferred!r} not found on this account; "
                "using attach/buy logic instead.",
                file=sys.stderr,
            )
        else:
            n = matches[0]
            if n.agent_id == agent_id:
                print(f"Using configured number {n.phone_number} ({n.id}) already on this agent")
                return n
            if not n.agent_id:
                client.agents.attach_number(agent_id, n.id)
                refreshed = client.numbers.list(limit=50).data
                found = next(x for x in refreshed if x.id == n.id)
                print(f"Attached configured number {found.phone_number} ({found.id})")
                return found
            print(
                f"Error: {n.phone_number} is assigned to agent {n.agent_id}, not {agent_id}. "
                "Detach it in the dashboard or clear AGENTPHONE_PHONE_NUMBER.",
                file=sys.stderr,
            )
            sys.exit(1)

    attached = [n for n in nums if n.agent_id == agent_id]
    if attached:
        n = attached[0]
        print(f"Using attached number {n.phone_number} ({n.id})")
        return n

    free = [n for n in nums if not n.agent_id]
    if free:
        free_num = free[0]
        client.agents.attach_number(agent_id, free_num.id)
        refreshed = client.numbers.list(limit=50).data
        found = next(x for x in refreshed if x.id == free_num.id)
        print(f"Attached existing number {found.phone_number} ({found.id})")
        return found

    bought = client.numbers.buy(country="US", agent_id=agent_id)
    print(f"Purchased number {bought.phone_number} ({bought.id})")
    return bought


def main() -> int:
    api_key = os.environ.get("AGENTPHONE_API_KEY", "").strip()
    if not api_key:
        print("Set AGENTPHONE_API_KEY in .env", file=sys.stderr)
        return 1
    public_base = os.environ.get("WEBHOOK_PUBLIC_BASE", "").strip().rstrip("/")
    if not public_base or not public_base.startswith("https://"):
        print("Set WEBHOOK_PUBLIC_BASE to your public HTTPS origin (e.g. ngrok URL)", file=sys.stderr)
        return 1

    client = AgentPhone(api_key=api_key)

    preferred = os.environ.get("AGENTPHONE_PHONE_NUMBER", "").strip().strip('"').strip("'")
    nums = client.numbers.list(limit=100).data
    agent = None
    if preferred:
        want = _digits(preferred)
        for n in nums:
            if _digits(n.phone_number) == want and n.agent_id:
                agent = client.agents.get(n.agent_id)
                print(f"Using agent {agent.id} ({agent.name}) — already owns {n.phone_number}")
                break

    if agent is None:
        agents = client.agents.list(limit=50).data
        standalone = [a for a in agents if a.name == "Standalone AgentPhone demo"]
        if standalone:
            agent = client.agents.get(standalone[0].id)
            print(f"Using existing agent {agent.id} ({agent.name})")
        else:
            agent = client.agents.create(
                name="Standalone AgentPhone demo",
                voice_mode="webhook",
                begin_message="You reached the standalone demo line. Go ahead.",
                enable_messaging=True,
            )
            print(f"Created agent {agent.id}")

    agent = client.agents.update(agent.id, voice_mode="webhook", enable_messaging=True)

    # Clean up any empty demo agents left behind by earlier failed runs.
    try:
        all_agents = client.agents.list(limit=100).data
        number_owners = {n.agent_id for n in nums if n.agent_id}
        for a in all_agents:
            if (
                a.name == "Standalone AgentPhone demo"
                and a.id != agent.id
                and a.id not in number_owners
            ):
                try:
                    client.agents.delete(a.id)
                    print(f"Cleaned up empty demo agent {a.id} (no numbers attached)")
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"  warning: could not delete orphan agent {a.id}: {exc}",
                        file=sys.stderr,
                    )
    except Exception as exc:  # noqa: BLE001
        print(f"  warning: orphan cleanup skipped: {exc}", file=sys.stderr)

    number = _pick_number(client, agent.id, nums)

    webhook_url = f"{public_base}/webhook"
    wh = client.webhooks.set(url=webhook_url, context_limit=20, timeout=120)
    print(f"Webhook URL set to {webhook_url}")
    print(f"Webhook secret (store as AGENTPHONE_WEBHOOK_SECRET): {wh.secret}")

    state = {
        "agent_id": agent.id,
        "number_id": number.id,
        "phone_number": number.phone_number,
        "webhook_id": wh.id,
    }
    (ROOT / "agentphone_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(f"Wrote {ROOT / 'agentphone_state.json'} (gitignored)")

    env_hint = ROOT / ".env"
    print("\nAdd or update these lines in .env:")
    print(f"AGENTPHONE_WEBHOOK_SECRET={wh.secret}")
    print(f"AGENTPHONE_AGENT_ID={agent.id}")
    print(f"AGENTPHONE_NUMBER_ID={number.id}")
    print(f"\nThen start: uvicorn webhook_server:app --host 0.0.0.0 --port 8765")
    print(f"Expose port 8765 with ngrok: ngrok http 8765  -> match WEBHOOK_PUBLIC_BASE")
    _ = env_hint  # referenced for user message only
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
