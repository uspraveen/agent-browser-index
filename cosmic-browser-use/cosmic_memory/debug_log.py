"""Structured debug logs for COSMIC browser memory runs."""

from __future__ import annotations

import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


MAX_STRING_CHARS = 4000


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_for_log(value: Any, max_string_chars: int = MAX_STRING_CHARS) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= max_string_chars else value[:max_string_chars] + "...<truncated>"
    if isinstance(value, (list, tuple)):
        return [sanitize_for_log(item, max_string_chars) for item in value[:100]]
    if isinstance(value, dict):
        return {
            str(key): sanitize_for_log(item, max_string_chars)
            for key, item in list(value.items())[:100]
            if "api_key" not in str(key).lower() and "authorization" not in str(key).lower()
        }
    if hasattr(value, "to_dict"):
        return sanitize_for_log(value.to_dict(), max_string_chars)
    return sanitize_for_log(str(value), max_string_chars)


class CosmicDebugLogger:
    def __init__(self, run_dir: str | Path):
        self.root = Path(run_dir) / "cosmic_debug"
        self.root.mkdir(parents=True, exist_ok=True)
        self.events_path = self.root / "events.jsonl"
        self.steps_path = self.root / "steps.jsonl"
        self.memory_path = self.root / "memory.jsonl"
        self.replay_path = self.root / "replay.jsonl"
        self._start = time.time()

    def event(self, event: str, **payload: Any) -> None:
        self._write(self.events_path, event, payload)

    def memory(self, event: str, **payload: Any) -> None:
        self._write(self.memory_path, event, payload)

    def replay(self, event: str, **payload: Any) -> None:
        self._write(self.replay_path, event, payload)

    def step(self, step_number: int, phase: str, **payload: Any) -> None:
        self._write(self.steps_path, f"step.{phase}", {"step": step_number, **payload})

    def exception(self, event: str, exc: BaseException, **payload: Any) -> None:
        self._write(
            self.events_path,
            event,
            {
                **payload,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )

    def paths(self) -> Dict[str, str]:
        return {
            "events": str(self.events_path),
            "steps": str(self.steps_path),
            "memory": str(self.memory_path),
            "replay": str(self.replay_path),
        }

    def _write(self, path: Path, event: str, payload: Dict[str, Any]) -> None:
        row = {
            "ts": utc_now(),
            "elapsed_ms": int((time.time() - self._start) * 1000),
            "event": event,
            **sanitize_for_log(payload),
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
