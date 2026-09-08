"""Programmatic single-run API for the Cosmic-OS browser specialist.

Wraps main.run_task with:
- env-driven model configuration (Fireworks base brain, xAI frontier brain);
- vault credential provisioning (in-memory, per run);
- a step progress callback bridge for task.progress events;
- a normalized result contract:
    {status, answer, steps_taken, duration_sec, run_dir, needs_credentials}
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from . import get_repo_home  # noqa: F401 - also bootstraps sys.path


class BrowserRunError(RuntimeError):
    """Raised when a browser run cannot be started or fails hard."""


ProgressCallback = Callable[[Dict[str, Any]], Awaitable[None] | None]


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _resolve_model_configs():
    """Build fast/medium/slow LLMConfig triple from env, mirroring main.py's CLI defaults."""
    from cosmic_types import LLMConfig, LLMProvider, LLMTier
    from cli_labels import resolve_fireworks_default_model, resolve_escalation_model, XAI_BASE_URL

    api_key = _env("FIREWORKS_API_KEY") or _env("SLIDE_AGENT_FIREWORKS_API_KEY")
    if not api_key:
        raise BrowserRunError("FIREWORKS_API_KEY is not set — the browser agent cannot run without its base brain.")
    base_url = _env("FIREWORKS_BASE_URL") or "https://api.fireworks.ai/inference/v1"
    fast_model = _env("BROWSER_AGENT_MODEL") or _env("FIREWORKS_DEFAULT_MODEL") or resolve_fireworks_default_model()
    fast_timeout_ms = int(_env("BROWSER_AGENT_TIMEOUT_MS", "45000"))
    fast_max_tokens = int(_env("BROWSER_AGENT_MAX_TOKENS", "2048"))

    fast_config = LLMConfig(
        provider=LLMProvider.FIREWORKS_KIMI,
        model_id=fast_model,
        api_key=api_key,
        api_base=base_url,
        tier=LLMTier.FAST,
        timeout_ms=fast_timeout_ms,
        max_tokens=fast_max_tokens,
    )
    medium_config = None
    slow_config: Optional[LLMConfig] = None

    xai_key = _env("XAI_API_KEY")
    if xai_key:
        escalation_model = _env("XAI_MODEL") or resolve_escalation_model()
        slow_config = LLMConfig(
            provider=LLMProvider.XAI,
            model_id=escalation_model,
            api_key=xai_key,
            api_base=_env("XAI_BASE_URL") or XAI_BASE_URL,
            tier=LLMTier.SLOW,
            timeout_ms=int(_env("BROWSER_AGENT_SLOW_TIMEOUT_MS", "120000")),
            max_tokens=int(_env("BROWSER_AGENT_SLOW_MAX_TOKENS", "4096")),
        )
    return fast_config, medium_config, slow_config


def _normalize_needs_credentials(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except (TypeError, ValueError):
        pass
    return None


async def run_goal(
    goal: str,
    *,
    initial_url: Optional[str] = None,
    max_steps: int = 40,
    memory_mode: str = "off",
    headless: bool = True,
    credentials: Optional[Dict[str, Dict[str, str]]] = None,
    on_progress: Optional[ProgressCallback] = None,
    ask_user_handler=None,
    on_live_frame=None,
    working_dir_root: Optional[str] = None,
    run_timeout_sec: Optional[int] = None,
) -> Dict[str, Any]:
    """Run one browser goal to completion.

    credentials maps site domain (or URL) -> {username, password, totp_seed?}.
    Values stay in memory for this run only and are never logged or sent to
    any LLM; the model only learns the covered domains.

    on_live_frame, if given, receives each frame as a base64-encoded JPEG
    string (sync or async) from a CDP Page.startScreencast feed of the active
    tab — a true live view driven by actual repaints, not a fixed-interval
    poll. Caller is responsible for any throttling/transport; a slow
    on_live_frame only drops frame fidelity, it never blocks the browser
    (frames are ACKed immediately regardless of on_live_frame's own
    progress).

    Returns a normalized dict:
        status: success | incomplete | failed | credentials_needed
        answer: best final note (FINAL ANSWER preferred)
        needs_credentials: {site, reason} | None
        run_dir: artifacts/debug directory for this run
    """
    if not goal or not str(goal).strip():
        raise BrowserRunError("goal is required.")

    # Runs write artifacts relative to CWD (./runs/<ts>); pin it explicitly so
    # the specialist process controls where artifacts land.
    if working_dir_root:
        os.chdir(working_dir_root)

    fast_config, medium_config, slow_config = _resolve_model_configs()

    # Import late: main.py mutates the asyncio policy on win32 and pulls the
    # whole stack; do it inside the running loop like the CLI does.
    import main as browser_main

    started = time.time()

    async def _progress_bridge(info: Dict[str, Any]) -> None:
        if on_progress is None:
            return
        payload = dict(info)
        payload["elapsed_sec"] = round(time.time() - started, 1)
        result = on_progress(payload)
        if asyncio.iscoroutine(result):
            await result

    run_kwargs: Dict[str, Any] = dict(
        goal=str(goal).strip(),
        initial_url=initial_url,
        max_steps=max(1, int(max_steps)),
        fast_model_config=fast_config,
        medium_model_config=medium_config,
        slow_model_config=slow_config,
        memory_mode=memory_mode,
        headless=headless,
        ask_user_handler=ask_user_handler,
        step_callback=_progress_bridge,
        live_frame_callback=on_live_frame,
        supermemory_enabled=_env("SUPERMEMORY_API_KEY") != "" and _env("BROWSER_SUPERMEMORY_ENABLED", "true").lower() in {"1", "true", "yes", "on"},
    )
    mimo_url = _env("MIMO_API_URL")
    mimo_key = _env("MIMO_API_KEY")
    if mimo_url:
        run_kwargs["mimo_api_url"] = mimo_url
    if mimo_key:
        run_kwargs["mimo_api_key"] = mimo_key
    if credentials:
        run_kwargs["credentials"] = credentials

    timeout = int(run_timeout_sec or _env("BROWSER_AGENT_RUN_TIMEOUT_SEC", "840"))
    try:
        raw_result = await asyncio.wait_for(browser_main.run_task(**run_kwargs), timeout=timeout)
    except asyncio.TimeoutError as exc:
        raise BrowserRunError(f"Browser run exceeded {timeout}s and was cancelled.") from exc

    raw_result = raw_result or {}
    needs = _normalize_needs_credentials(raw_result.get("credentials_needed"))
    return {
        "status": str(raw_result.get("task_status") or "incomplete"),
        "answer": str(raw_result.get("final_answer") or ""),
        "steps_taken": int(raw_result.get("steps_taken") or 0),
        "duration_sec": round(time.time() - started, 1),
        "run_dir": str(raw_result.get("working_dir") or ""),
        "cosmic_replay": raw_result.get("cosmic_replay"),
        "llm_usage": raw_result.get("llm_usage") or {},
        "needs_credentials": needs,
    }
