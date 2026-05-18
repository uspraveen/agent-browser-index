"""
Resolve agent / number IDs from .env and the AgentPhone API when provision state is missing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentphone import AgentPhone

ROOT = Path(__file__).resolve().parent
DEMO_AGENT_NAME = "Standalone AgentPhone demo"


def _digits(phone: str) -> str:
    return "".join(c for c in phone if c.isdigit())


def _state_path() -> Path:
    return ROOT / "agentphone_state.json"


def _load_state() -> dict:
    path = _state_path()
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_agent_id(client: AgentPhone) -> str:
    aid = os.environ.get("AGENTPHONE_AGENT_ID", "").strip()
    if aid:
        return aid
    st = _load_state()
    if st.get("agent_id"):
        return str(st["agent_id"])

    agents = client.agents.list(limit=100).data
    for a in agents:
        if a.name == DEMO_AGENT_NAME:
            return a.id

    preferred = os.environ.get("AGENTPHONE_PHONE_NUMBER", "").strip().strip('"').strip("'")
    if preferred:
        want = _digits(preferred)
        for n in client.numbers.list(limit=100).data:
            if _digits(n.phone_number) == want and n.agent_id:
                return str(n.agent_id)

    if len(agents) == 1:
        return agents[0].id

    raise RuntimeError(
        "Could not determine which agent to use. Do one of:\n"
        f"  • Run: python provision.py\n"
        f"  • Or set AGENTPHONE_AGENT_ID in .env\n"
        f"  • Or set AGENTPHONE_PHONE_NUMBER to your AgentPhone line (e.g. +14786068471) so we can look it up\n"
        f"  • Or create an agent named {DEMO_AGENT_NAME!r} in the dashboard"
    )


def resolve_number_id(client: AgentPhone, agent_id: str | None = None) -> str | None:
    nid = os.environ.get("AGENTPHONE_NUMBER_ID", "").strip()
    if nid:
        return nid
    st = _load_state()
    if st.get("number_id"):
        return str(st["number_id"])

    preferred = os.environ.get("AGENTPHONE_PHONE_NUMBER", "").strip().strip('"').strip("'")
    if not preferred:
        return None
    want = _digits(preferred)
    for n in client.numbers.list(limit=100).data:
        if _digits(n.phone_number) != want:
            continue
        if agent_id is not None and n.agent_id != agent_id:
            continue
        return n.id
    return None
