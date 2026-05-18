from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    api_key: str
    webhook_secret: str
    agent_id: str | None
    number_id: str | None


def _state_path() -> Path:
    return ROOT / "agentphone_state.json"


def load_state() -> dict:
    path = _state_path()
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def get_settings() -> Settings:
    api_key = os.environ.get("AGENTPHONE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Set AGENTPHONE_API_KEY in AgentPhone/.env")
    secret = os.environ.get("AGENTPHONE_WEBHOOK_SECRET", "").strip()
    if not secret:
        raise RuntimeError("Set AGENTPHONE_WEBHOOK_SECRET (from webhook setup) in AgentPhone/.env")
    state = load_state()
    return Settings(
        api_key=api_key,
        webhook_secret=secret,
        agent_id=(os.environ.get("AGENTPHONE_AGENT_ID") or state.get("agent_id") or "").strip() or None,
        number_id=(os.environ.get("AGENTPHONE_NUMBER_ID") or state.get("number_id") or "").strip() or None,
    )
