"""
Send an outbound SMS via AgentPhone REST.

The /v1/messages endpoint expects snake_case keys (agent_id, to_number, body, number_id),
which matches what agentphone-python 0.7.0 already sends — so we just use the SDK helper.

Usage:
  cd AgentPhone
  python send_sms.py +15551234567 "Hello from standalone AgentPhone"
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from agentphone import AgentPhone
from dotenv import load_dotenv

from agent_resolve import resolve_agent_id, resolve_number_id

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


def main() -> int:
    api_key = os.environ.get("AGENTPHONE_API_KEY", "").strip()
    if not api_key:
        print("Set AGENTPHONE_API_KEY in .env", file=sys.stderr)
        return 1
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 1
    to_number, body = sys.argv[1], sys.argv[2]
    client = AgentPhone(api_key=api_key)
    try:
        agent_id = resolve_agent_id(client)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1
    nid = resolve_number_id(client, agent_id) or None
    result = client.messages.send(
        agent_id=agent_id,
        to_number=to_number,
        body=body,
        number_id=nid,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
