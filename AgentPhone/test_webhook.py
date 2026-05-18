"""
Diagnose webhook reachability.

  1) curl the /health endpoint via the ngrok URL (proves the tunnel works).
  2) Ask AgentPhone to deliver a synthetic webhook to the registered URL
     (POST /v1/webhooks/test — see https://docs.agentphone.ai/documentation/guides/webhooks#test-webhook).
  3) Print the AgentPhone-side result so we know whether it got a 2xx back.

Usage:
  cd AgentPhone
  python test_webhook.py
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

from agentphone import AgentPhone
from dotenv import load_dotenv

from agent_resolve import resolve_agent_id

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


def main() -> int:
    api_key = os.environ.get("AGENTPHONE_API_KEY", "").strip()
    if not api_key:
        print("Set AGENTPHONE_API_KEY in .env", file=sys.stderr)
        return 1

    client = AgentPhone(api_key=api_key)

    wh = client.webhooks.get()
    if wh is None:
        print("No project webhook is configured. Run `python provision.py` first.", file=sys.stderr)
        return 1
    print(f"Registered webhook URL: {wh.url}")

    base = wh.url.rsplit("/webhook", 1)[0] if wh.url.endswith("/webhook") else wh.url.rstrip("/")
    health_url = f"{base}/health"
    print(f"\n[1/2] GET {health_url}")
    try:
        with urllib.request.urlopen(health_url, timeout=10) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            print(f"  status={resp.status}  body={body!r}")
    except Exception as exc:  # noqa: BLE001
        print(f"  ERROR: {exc}")
        print(
            "  -> Tunnel is broken. Confirm ngrok / cloudflared is still running and that\n"
            "     WEBHOOK_PUBLIC_BASE in .env still matches your tunnel URL. Re-run provision.py\n"
            "     if the URL changed.",
            file=sys.stderr,
        )
        return 2

    try:
        agent_id = resolve_agent_id(client)
    except RuntimeError as exc:
        print(f"  ERROR resolving agent: {exc}", file=sys.stderr)
        return 3
    print(f"\n[2/2] POST /v1/webhooks/test?agentId={agent_id}")
    try:
        # SDK 0.7.0 sends agentId in the JSON body; the API wants it as a query string.
        result = client._request("POST", f"/v1/webhooks/test?agentId={agent_id}")
    except Exception as exc:  # noqa: BLE001
        print(f"  ERROR calling AgentPhone: {exc}", file=sys.stderr)
        return 3

    print(f"  AgentPhone reports: {json.dumps(result, indent=2)}")
    ok = bool(result.get("success"))
    status = result.get("httpStatus")
    err = result.get("errorMessage")
    if ok and status and 200 <= int(status) < 300:
        print("\n  -> Tunnel + signing secret both healthy. uvicorn should also show a [req] line above.")
        return 0
    print(
        f"\n  -> Delivery failed. status={status} error={err!r}\n"
        "     Check uvicorn output. A 403 means AGENTPHONE_WEBHOOK_SECRET in .env doesn't match\n"
        "     what AgentPhone has — re-run `python provision.py` then restart uvicorn.",
        file=sys.stderr,
    )
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
