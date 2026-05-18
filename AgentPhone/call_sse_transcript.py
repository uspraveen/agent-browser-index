"""
Low-buffering Server-Sent Events reader for GET /v1/calls/{id}/transcript/stream.

Notes:
- ``requests.Response.iter_lines`` reads bodies in 512-byte chunks, which can batch several SSE
  frames together. We read **bytes as they arrive** (`iter_content(chunk_size=1)`), decode utf-8
  incrementally, and split on ``\\n`` ourselves so each frame surfaces as soon as the socket has it.
- The SDK parser yielded one event per ``data:`` line and cleared ``event:`` each time, so a
  multi-line ``data:`` block could be mis-labeled. Here we accumulate ``data:`` lines until a
  blank line (SSE message boundary), per https://docs.agentphone.ai/documentation/guides/calls#stream-transcript-sse .
- A long read timeout keeps quiet calls alive between turns.
"""

from __future__ import annotations

import codecs
import json
import time
from collections.abc import Iterator
from typing import Any

from agentphone import AgentPhone

RawLine = tuple[float, str]
SseEvent = tuple[str | None, dict[str, Any]]


def iter_transcript_sse_events(
    client: AgentPhone,
    call_id: str,
    *,
    raw_callback=None,
) -> Iterator[SseEvent]:
    """
    Yield ``(event_name, data_dict)`` for each SSE message as soon as the socket has it.

    ``raw_callback(monotonic_seconds, raw_line)`` is called for every non-empty line
    (heartbeats, ``event:`` headers, ``data:`` chunks). Use this to debug timing.
    """
    url = f"{client.base_url}/v1/calls/{call_id}/transcript/stream"
    read_timeout = 600.0
    timeout: tuple[float, float] = (30.0, read_timeout)

    headers = {
        **client._session.headers,
        "Accept": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
    }

    with client._session.get(url, stream=True, timeout=timeout, headers=headers) as resp:
        resp.raise_for_status()

        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        text_buffer: str = ""
        event_name: str | None = None
        data_parts: list[str] = []

        def _flush_message() -> SseEvent | None:
            nonlocal event_name, data_parts
            if not data_parts:
                event_name = None
                return None
            raw = "\n".join(data_parts)
            data_parts = []
            ev = event_name
            event_name = None
            try:
                return ev, json.loads(raw)
            except json.JSONDecodeError:
                return None

        def _handle_line(line: str) -> SseEvent | None:
            nonlocal event_name
            line = line.rstrip("\r")
            if line == "":
                return _flush_message()
            if line.startswith(":"):
                return None
            if line.startswith("event:"):
                event_name = line[len("event:") :].strip() or None
                return None
            if line.startswith("data:"):
                data_parts.append(line[len("data:") :].lstrip())
                return None
            if line.startswith("{"):
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    return None
                ev = event_name
                event_name = None
                return ev, obj
            return None

        for byte_chunk in resp.iter_content(chunk_size=1, decode_unicode=False):
            if not byte_chunk:
                continue
            text_buffer += decoder.decode(byte_chunk)
            while "\n" in text_buffer:
                line, text_buffer = text_buffer.split("\n", 1)
                if raw_callback is not None and line.strip():
                    try:
                        raw_callback(time.monotonic(), line.rstrip("\r"))
                    except Exception:
                        pass
                emitted = _handle_line(line)
                if emitted is not None:
                    yield emitted

        if text_buffer:
            emitted = _handle_line(text_buffer)
            if emitted is not None:
                yield emitted
        tail = _flush_message()
        if tail is not None:
            yield tail
