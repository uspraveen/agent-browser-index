"""Subscribe to GET /v1/calls/{id}/transcript/stream for live user/agent turns (runs in a background thread)."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentphone import AgentPhone

_lock = threading.Lock()
_active: set[str] = set()


def start_transcript_stream_once(client_factory: Callable[[], "AgentPhone"], call_id: str) -> None:
    if not call_id:
        return
    with _lock:
        if call_id in _active:
            return
        _active.add(call_id)

    def _run() -> None:
        try:
            client = client_factory()
            for item in client.calls.stream_transcript(call_id):
                ev = item.get("event")
                data = item.get("data") or {}
                if ev == "connected":
                    print(
                        f"[transcript-sse] connected call={data.get('callId')} "
                        f"status={data.get('status')} direction={data.get('direction')}",
                        flush=True,
                    )
                elif ev == "turn":
                    role = data.get("role")
                    content = data.get("content", "")
                    print(f"[transcript-sse] [{role}] {content}", flush=True)
                elif ev == "ended":
                    print(
                        f"[transcript-sse] ended call={data.get('callId')} "
                        f"durationSeconds={data.get('durationSeconds')}",
                        flush=True,
                    )
        except Exception as exc:  # noqa: BLE001 — surface stream errors in the console
            print(f"[transcript-sse] error for {call_id}: {exc}", flush=True)
        finally:
            with _lock:
                _active.discard(call_id)

    threading.Thread(target=_run, name=f"ap-sse-{call_id[:12]}", daemon=True).start()
