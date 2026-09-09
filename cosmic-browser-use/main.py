#!/usr/bin/env python3
"""
Cosmic Browser Use Agent - Main execution loop with timing
"""
import asyncio
import base64
import argparse
import json
import time
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

# Windows Asyncio Fix
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from cosmic_types import (
    TaskConfig, LLMConfig, LLMProvider, LLMTier,
    VerificationStatus, ActionType, ActionResult
)
from memory_manager import MemoryManager
from orchestrator import Orchestrator, reset_fireworks_http2_preference
from browser_controller import BrowserController
from credentials import CredentialStore
from find_coordinates_mimo import check_mimo_health
from cli_labels import (
    cli_allowed_provider_labels,
    cli_provider_help,
    display_provider_label,
    display_provider_model,
    display_stat_value,
    normalize_cli_provider_arg,
    resolve_fireworks_default_model,
    resolve_escalation_model,
    XAI_BASE_URL,
)
from browser_memory.debug_log import CosmicDebugLogger
from browser_memory.demo_overlay import DemoOverlayManager
from browser_memory.replay import execute_indexed_replay_plan
from browser_memory.runtime import BrowserMemoryRuntime

import os
from dotenv import load_dotenv

load_dotenv()
# ==============================================================================

MIMO_DEFAULT_URL = "https://uspraveenraj--mimo-vl-7b-rl-serve.modal.run"


def _env_bool(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _is_information_goal(goal: str) -> bool:
    text = (goal or "").lower()
    terms = (
        "get",
        "find",
        "extract",
        "read",
        "what",
        "tell me",
        "description",
        "summarize",
        "summary",
        "list",
        "show me",
    )
    return any(term in text for term in terms)


def _should_try_visible_answer_governor(
    memory: MemoryManager,
    browser_state,
    config: TaskConfig,
    step_num: int,
    last_attempt_step: int,
) -> bool:
    """Detect same-page vision orbiting and trigger one final visible-answer check."""
    if not _env_bool("VISIBLE_ANSWER_GOVERNOR_ENABLED", True):
        return False
    if not _is_information_goal(config.goal):
        return False
    if browser_state.notes:
        return False

    min_step = int(os.getenv("VISIBLE_ANSWER_GOVERNOR_MIN_STEP", "12"))
    cooldown = int(os.getenv("VISIBLE_ANSWER_GOVERNOR_COOLDOWN_STEPS", "2"))
    window = int(os.getenv("VISIBLE_ANSWER_GOVERNOR_WINDOW", "4"))

    if step_num < min_step:
        return False
    if last_attempt_step and step_num - last_attempt_step < cooldown:
        return False
    if len(memory.steps) < window:
        return False

    recent = memory.steps[-window:]
    current_url = browser_state.url
    same_url_count = sum(
        1
        for step in recent
        if step.browser_state and step.browser_state.url == current_url
    )
    if same_url_count < max(2, window - 1):
        return False

    orbit_actions = {
        ActionType.VISUAL_SCROLL,
        ActionType.VISUAL_CLICK,
        ActionType.TIMED_WAIT,
        ActionType.VISUAL_WAIT,
        ActionType.PRESS_KEY,
    }
    orbit_count = sum(
        1 for step in recent if step.action and step.action.action_type in orbit_actions
    )
    if orbit_count < max(2, window - 1):
        return False

    if any(step.action and step.action.action_type == ActionType.SAVE_NOTE for step in recent):
        return False
    if any(
        step.action
        and step.action.action_type in {ActionType.NAVIGATE, ActionType.VISUAL_TYPE}
        for step in recent[-2:]
    ):
        return False

    has_expand_attempt = any(
        step.action
        and step.action.action_type in {ActionType.VISUAL_CLICK, ActionType.PRESS_KEY}
        and any(
            term in (step.action.description or "").lower()
            for term in ("more", "show more", "read more", "expand", "description")
        )
        for step in recent
    )
    best_progress = max(
        (step.action.estimated_completion or 0.0)
        for step in recent
        if step.action
    )
    min_progress = float(os.getenv("VISIBLE_ANSWER_GOVERNOR_MIN_PROGRESS", "0.65"))
    return best_progress >= min_progress or has_expand_attempt


def _is_search_results_page(browser_state) -> bool:
    url = (getattr(browser_state, "url", "") or "").lower()
    title = (getattr(browser_state, "title", "") or "").lower()
    return (
        "search_query=" in url
        or "/results?" in url
        or "/search?" in url
        or "?q=" in url
        or "&q=" in url
        or " search" in title
        or " - search" in title
    )


def _should_try_search_results_governor(
    memory: MemoryManager,
    browser_state,
    config: TaskConfig,
    step_num: int,
    last_attempt_step: int,
) -> bool:
    """Detect repeated search-results scanning and force a non-scroll decision."""
    if not _env_bool("SEARCH_RESULTS_GOVERNOR_ENABLED", True):
        return False
    if not _is_information_goal(config.goal):
        return False
    if not _is_search_results_page(browser_state):
        return False
    if browser_state.notes:
        return False

    min_step = int(os.getenv("SEARCH_RESULTS_GOVERNOR_MIN_STEP", "8"))
    cooldown = int(os.getenv("SEARCH_RESULTS_GOVERNOR_COOLDOWN_STEPS", "4"))
    window = int(os.getenv("SEARCH_RESULTS_GOVERNOR_WINDOW", "6"))
    if step_num < min_step:
        return False
    if last_attempt_step and step_num - last_attempt_step < cooldown:
        return False
    if len(memory.steps) < window:
        return False

    recent = memory.steps[-window:]
    same_url_count = sum(
        1
        for step in recent
        if step.browser_state and step.browser_state.url == browser_state.url
    )
    if same_url_count < max(3, window - 1):
        return False

    scroll_steps = [
        step for step in recent
        if step.action and step.action.action_type == ActionType.VISUAL_SCROLL
    ]
    if len(scroll_steps) < max(3, window - 2):
        return False

    directions = {
        (step.action.description or "").lower()
        for step in scroll_steps
    }
    has_zig_zag = any("down" in d for d in directions) and any("up" in d or "top" in d for d in directions)
    return has_zig_zag or len(scroll_steps) >= window - 1


async def _detect_credential_handoff_reason(browser: BrowserController) -> Optional[str]:
    """Cheap DOM probe: does the current page look like it's asking for the
    user's password or an MFA/OTP/verification code? The agent has no
    business attempting either itself (it doesn't have the user's secrets,
    and guessing at 2FA codes wastes steps at best). Returns a short reason
    string ("password" / "verification code") or None."""
    try:
        return await browser.page.evaluate(r"""
            () => {
                const visible = (el) => {
                    const r = el.getBoundingClientRect();
                    const s = window.getComputedStyle(el);
                    return r.width > 0 && r.height > 0 && s.visibility !== 'hidden' && s.display !== 'none';
                };
                const inputs = Array.from(document.querySelectorAll('input'));
                for (const el of inputs) {
                    if (el.disabled || !visible(el)) continue;
                    if ((el.type || '').toLowerCase() === 'password') return 'password';
                    const auto = (el.getAttribute('autocomplete') || '').toLowerCase();
                    // Normalize snake_case/kebab-case to spaces first — JS regex
                    // \b treats '_' as a word character, so "otp_field" would
                    // never match \botp\b without this (caught by testing).
                    const hint = ((el.name || '') + ' ' + (el.id || '') + ' ' + (el.getAttribute('aria-label') || '') + ' ' + (el.placeholder || ''))
                        .toLowerCase().replace(/[_-]/g, ' ');
                    if (auto.includes('one-time-code') || /\botp\b|\bmfa\b|\b2fa\b|verification.?code|security.?code|auth(entication)?.?code/.test(hint)) {
                        return 'verification code';
                    }
                }
                return null;
            }
        """)
    except Exception:
        return None


async def _should_try_credential_handoff_governor(
    browser: BrowserController,
    config: TaskConfig,
    step_num: int,
    last_attempt_step: int,
) -> Optional[str]:
    if not _env_bool("CREDENTIAL_HANDOFF_GOVERNOR_ENABLED", True):
        return None
    cooldown = int(os.getenv("CREDENTIAL_HANDOFF_GOVERNOR_COOLDOWN_STEPS", "3"))
    if last_attempt_step and step_num - last_attempt_step < cooldown:
        return None
    return await _detect_credential_handoff_reason(browser)


async def _try_finalize_after_replay_checkpoint(
    *,
    browser: BrowserController,
    memory: MemoryManager,
    orchestrator: Orchestrator,
    cosmic_log: CosmicDebugLogger,
) -> bool:
    """At a replay checkpoint, make one focused visible-answer save attempt."""
    step_num = len(memory.steps) + 1
    screenshot_path, screenshot_hash, browser_state = await browser.capture_state(
        f"step_{step_num:03d}_replay_checkpoint"
    )
    context = memory.get_context_for_llm(screenshot_path)
    with open(screenshot_path, "rb") as f:
        screenshot_b64 = f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode()}"

    cosmic_log.replay(
        "checkpoint_finalizer.attempt",
        step=step_num,
        screenshot_path=screenshot_path,
        browser_state=browser_state.to_dict(),
    )
    llm_response = await orchestrator.force_visible_answer_note(
        context=context,
        screenshot_base64=screenshot_b64,
    )
    if not llm_response:
        cosmic_log.replay("checkpoint_finalizer.no_visible_answer", step=step_num)
        return False

    execution_start = time.time()
    action_result = await browser.execute_tool(llm_response.tool_call, screenshot_path)
    action_result.execution_time_ms = action_result.execution_time_ms or ((time.time() - execution_start) * 1000)
    after_screenshot_path, after_screenshot_hash, after_state = await browser.capture_state(
        f"step_{step_num:03d}_replay_checkpoint_after"
    )
    verification_status = VerificationStatus.SUCCESS if action_result.success else VerificationStatus.ERROR
    action_result.verification_status = verification_status
    action_result.state_change_score = 0.0
    action_result.estimated_completion = llm_response.estimated_completion

    memory.add_step(
        screenshot_path=screenshot_path,
        screenshot_hash=screenshot_hash,
        browser_state=after_state,
        action=action_result,
        thinking=llm_response.reasoning,
        before_browser_state=browser_state,
        after_browser_state=after_state,
        after_screenshot_path=after_screenshot_path,
        after_screenshot_hash=after_screenshot_hash,
        tool_call={
            "action_type": llm_response.tool_call.action_type.value,
            "parameters": llm_response.tool_call.parameters,
            "source": "cosmic_replay_checkpoint_finalizer",
            "verification_hint": llm_response.tool_call.verification_hint,
        },
        llm_response=llm_response.to_dict(),
    )
    cosmic_log.replay(
        "checkpoint_finalizer.saved",
        step=step_num,
        llm_response=llm_response.to_dict(),
        action_result=action_result.to_dict(),
        after_browser_state=after_state.to_dict(),
    )
    return bool(action_result.success and llm_response.estimated_completion >= 0.95)


def _build_replay_handoff_note(replay_summary: dict) -> str:
    """Build a compact handoff note to prepend to cumulative_summary after indexed replay.

    Caps failure output to 3 items, 200 chars each, so a pathological replay never
    floods the LLM context. In practice replay breaks on the first failure so there
    is at most one item.
    """
    executed = replay_summary.get("executed") or []
    failures = replay_summary.get("failures") or []
    checkpoint_reason = replay_summary.get("checkpoint_reason") or ""

    n = len(executed)
    if not n:
        return "COSMIC indexed replay ran but executed no actions."

    if not failures:
        # Clean run or checkpoint
        parts = [f"COSMIC indexed replay completed {n} action(s) successfully."]
        if checkpoint_reason:
            parts.append(f"Checkpoint: {checkpoint_reason}")
        return " ".join(parts)

    # Failure path — cap at 3, truncate each to 200 chars
    capped = failures[-3:]
    failure_lines = []
    for f in capped:
        action = f.get("action") or {}
        err = str(f.get("error") or "unknown error")[:200]
        step_id = action.get("workflow_step_id") or action.get("action_type") or "unknown step"
        failure_lines.append(f"  - {step_id}: {err}")
    failure_block = "\n".join(failure_lines)
    return (
        f"COSMIC indexed replay ran {n} action(s) then failed. "
        f"Continue from the current page state.\nFailure(s):\n{failure_block}"
    )


def _should_try_replay_checkpoint_finalizer(replay_summary: dict) -> bool:
    if not replay_summary.get("completed_replay") or not replay_summary.get("checkpoint_reached"):
        return False
    reason = str(replay_summary.get("checkpoint_reason") or "").lower()
    if "answer extraction" in reason or "observed-target" in reason:
        return True
    executed = replay_summary.get("executed") or []
    if not executed:
        return False
    last = executed[-1]
    if last.get("action_type") in {ActionType.NAVIGATE.value, ActionType.VISUAL_TYPE.value}:
        return False
    if "search" in reason or "result" in reason:
        return False
    return last.get("action_type") in {
        ActionType.VISUAL_CLICK.value,
        ActionType.VISUAL_SCROLL.value,
        ActionType.PRESS_KEY.value,
    }


async def run_task(
    goal: str,
    initial_url: str = None, # Optional
    max_steps: int = 1000,   # Default increased to 1000
    fast_model_config: LLMConfig = None,
    medium_model_config: LLMConfig = None,
    slow_model_config: LLMConfig = None,
    mimo_api_url: str = None,
    mimo_api_key: str = None,
    headless: bool = None,
    # SDK-ready configuration overrides
    summary_interval: int = None,
    max_tabs: int = None,
    screenshot_quality: int = None,
    ask_user_timeout: int = None,
    large_notes_path: str = None,
    memory_mode: str = "off",
    memory_dir: str = None,
    cosmic_user_id: str = "demo_user",
    cosmic_container_tag: str = "cosmic-hackathon-demo",
    supermemory_enabled: bool = True,
    replay_max_actions: int = 8,
    interaction_mode: str = "hybrid",
    demo_overlay_enabled: bool = False,
    ask_user_handler=None,
    chrome_profile: str = None,
    restore_previous_tabs: bool = False,
    refresh_chrome_profile: bool = False,
    credentials: Optional[Dict[str, Dict[str, str]]] = None,
    step_callback=None,
    live_frame_callback=None,
):
    mimo_api_url = mimo_api_url or os.getenv("MIMO_API_URL", MIMO_DEFAULT_URL)
    mimo_api_key = mimo_api_key or os.getenv("MIMO_API_KEY")
    headless = headless if headless is not None else os.getenv("HEADLESS", "False").lower() == "true"
    summary_interval = summary_interval if summary_interval is not None else int(os.getenv("SUMMARY_INTERVAL_STEPS", "10"))
    max_tabs = max_tabs if max_tabs is not None else int(os.getenv("MAX_TABS", "5"))
    screenshot_quality = screenshot_quality if screenshot_quality is not None else int(os.getenv("SCREENSHOT_QUALITY", "50"))
    ask_user_timeout = ask_user_timeout if ask_user_timeout is not None else int(os.getenv("ASK_USER_TIMEOUT", "120"))
    interaction_mode = (interaction_mode or os.getenv("COSMIC_INTERACTION_MODE", "hybrid")).strip().lower()
    if interaction_mode not in {"hybrid", "vision"}:
        interaction_mode = "hybrid"
    enable_dom_fallback = interaction_mode != "vision"

    # Per-run vault credentials (provisioned by the Cosmic orchestrator or the
    # SDK caller). Values stay in memory only; the model learns just the
    # domains they cover.
    credential_store = CredentialStore()
    if credentials:
        if isinstance(credentials, str):
            credentials = json.loads(credentials)
        if isinstance(credentials, dict):
            for site, values in credentials.items():
                if isinstance(values, dict):
                    credential_store.add(
                        site,
                        username=values.get("username", ""),
                        password=values.get("password", ""),
                        totp_seed=values.get("totp_seed", ""),
                        notes=values.get("notes", ""),
                        site_url=values.get("site_url", ""),
                    )
    credentials_available_for = tuple(credential_store.available_domains())
    """
    Run a browser automation task with comprehensive timing.

    This function is the primary entry point for both CLI and SDK usage.
    All behavioral parameters are exposed as keyword arguments with sensible
    defaults from config.py, making it easy to override from an API layer.
    """

    reset_fireworks_http2_preference()

    # Setup working directory
    working_dir = Path(f"./runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    working_dir.mkdir(parents=True, exist_ok=True)
    resolved_large_notes_path = Path(large_notes_path).expanduser() if large_notes_path else (working_dir / "large_notes.jsonl")
    cosmic_log = CosmicDebugLogger(working_dir)

    # Resolve chrome_profile to an absolute path if just a profile name was given.
    resolved_chrome_profile = None
    if chrome_profile:
        p = Path(chrome_profile)
        if p.is_absolute() and p.is_dir():
            resolved_chrome_profile = str(p)
        else:
            # Treat as a profile directory name relative to Chrome's User Data folder.
            if sys.platform == "win32":
                user_data = Path(os.environ.get("LOCALAPPDATA", "")) / "Google" / "Chrome" / "User Data"
            elif sys.platform == "darwin":
                user_data = Path.home() / "Library" / "Application Support" / "Google" / "Chrome"
            else:
                user_data = Path.home() / ".config" / "google-chrome"
            candidate = user_data / chrome_profile
            if candidate.is_dir():
                resolved_chrome_profile = str(candidate)
            else:
                print(f"⚠️  Chrome profile '{chrome_profile}' not found at {candidate}. Falling back to default browser.")

    config = TaskConfig(
        task_id=f"task_{datetime.now().timestamp()}",
        goal=goal,
        max_steps=max_steps,
        screenshot_quality=screenshot_quality,
        summary_interval=summary_interval,
        max_tabs=max_tabs,
        ask_user_timeout=ask_user_timeout,
        enable_dom_fallback=enable_dom_fallback,
        chrome_profile=resolved_chrome_profile,
        restore_previous_tabs=restore_previous_tabs,
        refresh_chrome_profile=refresh_chrome_profile,
        credentials_available_for=credentials_available_for,
    )
    
    print(f"\n{'='*80}")
    print(f"COSMIC BROWSER USE AGENT - Task Started")
    print(f"{'='*80}")
    print(f"Goal: {goal}")
    base_model_id = getattr(fast_model_config, "model_id", "?")
    esc_model_id = getattr(slow_model_config, "model_id", "?") if slow_model_config else "?"
    print(f"Provider: {display_provider_label(fast_model_config.provider)}")
    print(f"Base brain: {base_model_id} | Escalation brain: {esc_model_id}")
    print(f"Initial URL: {initial_url if initial_url else 'about:blank'}")
    print(f"Max steps: {max_steps}")
    print(f"Working directory: {working_dir}")
    print(f"Large notes file: {resolved_large_notes_path}")
    print(f"Interaction mode: {interaction_mode} (DOM tools: {'on' if enable_dom_fallback else 'off'})")
    print(f"COSMIC memory mode: {memory_mode}")
    print(f"COSMIC debug logs: {working_dir / 'cosmic_debug'}")
    print(f"{'='*80}\n")
    cosmic_log.event(
        "run.start",
        goal=goal,
        initial_url=initial_url,
        provider=fast_model_config.provider.value,
        max_steps=max_steps,
        working_dir=str(working_dir),
        memory_mode=memory_mode,
        memory_dir=memory_dir,
        supermemory_enabled=supermemory_enabled,
        interaction_mode=interaction_mode,
        enable_dom_fallback=enable_dom_fallback,
    )
    
    # Initialize components
    # Fireworks may power the cheap summarizer even when it isn't the fast
    # tier (e.g. the "bu" model set puts browser-use cloud on fast and GLM/
    # Fireworks on slow as the escalation brain) — check both tiers so
    # compression doesn't silently fall back to an unconfigured OpenAI key.
    summary_provider = (
        os.getenv("SUMMARY_LLM_PROVIDER")
        or (
            "fireworks_kimi"
            if LLMProvider.FIREWORKS_KIMI in {
                getattr(fast_model_config, "provider", None),
                getattr(slow_model_config, "provider", None),
            }
            else "openai"
        )
    ).strip().lower()
    if summary_provider in {"fireworks", "fireworks_kimi", "kimi"}:
        summary_model = os.getenv("SUMMARY_LLM_MODEL") or resolve_fireworks_default_model()
        summary_api_key = os.getenv("FIREWORKS_API_KEY") or os.getenv("SLIDE_AGENT_FIREWORKS_API_KEY")
        summary_api_base = os.getenv("FIREWORKS_BASE_URL") or "https://api.fireworks.ai/inference/v1"
    else:
        summary_model = os.getenv("SUMMARY_LLM_MODEL", "gpt-5.6-luna")
        summary_api_key = os.getenv("OPENAI_API_KEY")
        summary_api_base = None

    memory = MemoryManager(
        config,
        working_dir,
        api_key=os.getenv("OPENAI_API_KEY"),
        summary_provider=summary_provider,
        summary_model=summary_model,
        summary_api_key=summary_api_key,
        summary_api_base=summary_api_base,
    )
    print(f"Summary model: {display_provider_model(memory.summary_provider, memory.summary_model)}")
    cosmic_log.event(
        "memory.summary_config",
        summary_provider=memory.summary_provider,
        summary_model=memory.summary_model,
        summary_enabled=memory.client is not None,
    )
    
    orchestrator = Orchestrator(
        fast_model=fast_model_config,
        medium_model=medium_model_config,
        slow_model=slow_model_config,
    )
    
    demo_overlay = DemoOverlayManager(enabled=demo_overlay_enabled)
    if demo_overlay.enabled:
        print("🎛️  Demo overlay enabled (hidden during every agent screenshot capture).")
        cosmic_log.event("demo_overlay.enabled")
        demo_overlay.set_state(
            mode=("Memory Mode" if memory_mode in {"recall", "auto"} else
                  "Learning Mode" if memory_mode == "learn" else "Live Mode"),
        )

    browser = BrowserController(
        config=config,
        mimo_api_url=mimo_api_url,
        mimo_api_key=mimo_api_key,
        working_dir=working_dir,
        large_notes_path=resolved_large_notes_path,
        headless=headless,
        demo_overlay=demo_overlay,
        ask_user_handler=ask_user_handler,
        credential_store=credential_store,
    )

    await browser.start(initial_url)
    if live_frame_callback is not None:
        await browser.start_live_screencast(live_frame_callback)
    task_start_time = time.time()

    cosmic_runtime = None
    retrieved_memory = None
    replay_summary = None
    if memory_mode in {"learn", "recall", "auto"}:
        cosmic_runtime = BrowserMemoryRuntime(
            data_dir=memory_dir,
            user_id=cosmic_user_id,
            container_tag=cosmic_container_tag,
            supermemory_enabled=supermemory_enabled,
        )
        cosmic_log.memory(
            "runtime.initialized",
            memory_dir=memory_dir,
            supermemory_enabled=cosmic_runtime.supermemory.enabled,
            supermemory_space_tag=cosmic_runtime.supermemory.space_tag,
            supermemory_task_type=cosmic_runtime.supermemory.task_type,
            supermemory_error=cosmic_runtime.supermemory.last_error,
        )
        if cosmic_runtime.supermemory.last_error:
            print(f"⚠️  COSMIC Supermemory: {cosmic_runtime.supermemory.last_error}")
        await demo_overlay.update(
            page=browser.page,
            supermemory=("Connected" if cosmic_runtime.supermemory.enabled else "Local only"),
        )
    else:
        await demo_overlay.update(page=browser.page, supermemory="Local only")

    if memory_mode in {"recall", "auto"} and cosmic_runtime:
        print("\n🧠 COSMIC Memory Recall: searching prior traversal workflows...")
        await demo_overlay.update(
            page=browser.page,
            phase="Retrieving memory",
            pulse_ms=1500,
            timeline_append={"kind": "recall", "label": "Supermemory recall"},
        )
        probe_screenshot_path, _, probe_state = await browser.capture_state("browser_memory_probe")
        cosmic_log.memory(
            "recall.probe_state",
            screenshot_path=probe_screenshot_path,
            browser_state=probe_state.to_dict(),
        )
        retrieved_memory = cosmic_runtime.retrieve(
            task=goal,
            domain=None,
            current_page_summary=f"{probe_state.title} {probe_state.url}",
        )
        best_workflow = retrieved_memory.get("best_workflow")
        best_score = float(retrieved_memory.get("best_score") or 0.0)
        cosmic_log.memory(
            "recall.result",
            domain=retrieved_memory.get("domain"),
            best_score=best_score,
            best_workflow_id=best_workflow.get("workflow_id") if best_workflow else None,
            candidate_count=len(retrieved_memory.get("candidate_workflows") or []),
            supermemory_result_count=len(retrieved_memory.get("supermemory_results") or []),
        )
        if best_workflow and best_score >= float(os.getenv("COSMIC_RECALL_MIN_SCORE", "0.18")):
            print(f"✅ COSMIC Memory matched workflow: {best_workflow.get('workflow_id')} (score={best_score:.2f})")
            await demo_overlay.update(
                page=browser.page,
                workflow_id=best_workflow.get("workflow_id"),
                recall_score=best_score,
                phase="Matched path",
                pulse_ms=1500,
                timeline_append={
                    "kind": "recall",
                    "label": f"Matched workflow ({best_score:.2f})",
                },
            )
            with open(probe_screenshot_path, "rb") as f:
                probe_screenshot_b64 = f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode()}"
            replay_plan = await orchestrator.plan_indexed_replay(
                goal=goal,
                memory_package=retrieved_memory,
                current_state=probe_state.to_dict(),
                screenshot_base64=probe_screenshot_b64,
                max_actions=replay_max_actions,
            )
            cosmic_log.replay(
                "planner.result",
                workflow_id=best_workflow.get("workflow_id"),
                best_score=best_score,
                replay_plan=replay_plan,
            )
            if replay_plan.get("use_workflow") and replay_plan.get("actions"):
                print(
                    "🚀 COSMIC Indexed Replay: "
                    f"{len(replay_plan.get('actions', []))} planned actions, "
                    f"checkpoint after {replay_plan.get('checkpoint_after_actions')}."
                )
                print(f"   Reason: {replay_plan.get('reason', '')[:240]}")
                replay_summary = await execute_indexed_replay_plan(
                    browser=browser,
                    memory=memory,
                    plan=replay_plan,
                    workflow=best_workflow,
                    max_actions=replay_max_actions,
                    debug_logger=cosmic_log,
                    demo_overlay=demo_overlay,
                )
                print(f"🧭 COSMIC Replay result: {replay_summary}")
                if (
                    _should_try_replay_checkpoint_finalizer(replay_summary)
                    and not replay_summary.get("goal_completed")
                    and _is_information_goal(goal)
                ):
                    print("🔎 COSMIC Replay checkpoint: trying one focused visible-answer finalizer...")
                    finalized = await _try_finalize_after_replay_checkpoint(
                        browser=browser,
                        memory=memory,
                        orchestrator=orchestrator,
                        cosmic_log=cosmic_log,
                    )
                    replay_summary["checkpoint_finalizer_saved"] = finalized
                    replay_summary["goal_completed"] = bool(finalized)
                    if finalized:
                        print("✅ COSMIC Replay checkpoint finalizer saved the answer.")
            else:
                print(f"ℹ️  COSMIC planner skipped replay: {replay_plan.get('reason', 'no reason')}")
                await demo_overlay.update(
                    page=browser.page,
                    phase="Live planner fallback",
                )
        else:
            print("ℹ️  COSMIC Memory: no strong workflow match found; running normal agent.")
            await demo_overlay.update(
                page=browser.page,
                phase="Live planner fallback",
            )
    
    # Main loop
    previous_confidence = 1.0
    pending_escalation = False  # set when the base brain requests the frontier model
    escalation_hold = 0  # consecutive frontier steps remaining (sticky escalation budget)
    escalation_cooldown = 0  # steps forced back onto the base brain after a frontier handback
    checkpoint_path = None
    task_status = "incomplete"
    credentials_request = None
    last_visible_answer_governor_step = 0
    last_search_results_governor_step = 0
    last_credential_governor_step = 0
    live_llm_decisions = 0
    
    try:
        if replay_summary and replay_summary.get("goal_completed"):
            print("\n🎉 GOAL ACHIEVED via COSMIC indexed replay!")
            task_status = "success"
            await demo_overlay.update(
                page=browser.page,
                phase="Saved answer",
                pulse_ms=2000,
                timeline_append={"kind": "saved", "label": "Answer saved"},
            )
            final_notes = []
            if memory.steps:
                last_state = memory.steps[-1].browser_state
                if last_state and last_state.notes:
                    final_notes = last_state.notes
            if final_notes:
                print("\n" + "="*80)
                print("📋 RESULT")
                print("="*80)
                for note in final_notes:
                    print(f"  {note}")
                print("="*80)

        if replay_summary and not replay_summary.get("goal_completed"):
            handoff_note = _build_replay_handoff_note(replay_summary)
            memory.cumulative_summary = handoff_note + " " + memory.cumulative_summary

        loop_start = max_steps + 1 if task_status == "success" else len(memory.steps) + 1
        for step_num in range(loop_start, max_steps + 1):
            # Track step execution time
            step_start_time = time.time()
            
            print(f"\n{'='*80}")
            print(f"STEP {step_num}/{max_steps}")
            print(f"{'='*80}")
            cosmic_log.step(step_num, "start", max_steps=max_steps)
            
            # 1. Capture current state
            capture_start = time.time()
            screenshot_path, screenshot_hash, browser_state = await browser.capture_state(
                f"step_{step_num:03d}"
            )
            capture_time_ms = (time.time() - capture_start) * 1000
            
            print(f"📸 Screenshot: {screenshot_path}")
            print(f"🌐 URL: {browser_state.url}")
            print(f"📄 Title: {browser_state.title}")
            print(f"⏱️  Capture time: {capture_time_ms:.0f}ms")
            cosmic_log.step(
                step_num,
                "capture.before",
                screenshot_path=screenshot_path,
                screenshot_hash=screenshot_hash,
                browser_state=browser_state.to_dict(),
                capture_time_ms=capture_time_ms,
            )
            
            # 2. Check loop detection
            stuck_signal = memory.detect_loop()
            if stuck_signal:
                print("\n⚠️  Stuck-signal detected — escalating to the frontier brain")
                previous_confidence = 0.0
            
            # 3. Get LLM decision
            llm_start = time.time()
            context = memory.get_context_for_llm(screenshot_path)
            # Routed to the tier selector AND shown to the model. Setting
            # previous_confidence alone was not enough: _select_tier's
            # read-only fast path sits above the confidence rule, and a
            # stuck agent's last action is almost always a read.
            context["stuck_signal"] = stuck_signal
            
            # Load screenshot as base64
            with open(screenshot_path, 'rb') as f:
                screenshot_b64 = f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode()}"

            llm_response = None

            credential_handoff_reason = await _should_try_credential_handoff_governor(
                browser=browser,
                config=config,
                step_num=step_num,
                last_attempt_step=last_credential_governor_step,
            )
            if credential_handoff_reason:
                last_credential_governor_step = step_num
                print(f"   [Credential governor] detected a visible {credential_handoff_reason} field — forcing AskUser handoff...")
                llm_response = await orchestrator.force_credential_handoff(context=context, reason=credential_handoff_reason)
                cosmic_log.step(
                    step_num,
                    "credential_governor.force_handoff",
                    reason=credential_handoff_reason,
                    llm_response=llm_response.to_dict(),
                )

            if llm_response is None and _should_try_visible_answer_governor(
                memory=memory,
                browser_state=browser_state,
                config=config,
                step_num=step_num,
                last_attempt_step=last_visible_answer_governor_step,
            ):
                last_visible_answer_governor_step = step_num
                print("   [Visible-answer governor] same-page orbit detected; forcing best visible answer note...")
                cosmic_log.step(
                    step_num,
                    "visible_answer_governor.force_attempt",
                    recent_steps=context.get("recent_steps", []),
                )
                llm_response = await orchestrator.force_visible_answer_note(
                    context=context,
                    screenshot_base64=screenshot_b64,
                )
                if llm_response:
                    print("   [Visible-answer governor] SaveNote forced from visible screen.")
                    cosmic_log.step(
                        step_num,
                        "visible_answer_governor.force_save",
                        llm_response=llm_response.to_dict(),
                    )
                else:
                    print("   [Visible-answer governor] current screen did not match a saveable answer; continuing normal planner.")
                    cosmic_log.step(
                        step_num,
                        "visible_answer_governor.no_save",
                    )

            if llm_response is None and _should_try_search_results_governor(
                memory=memory,
                browser_state=browser_state,
                config=config,
                step_num=step_num,
                last_attempt_step=last_search_results_governor_step,
            ):
                last_search_results_governor_step = step_num
                print("   [Search-results governor] repeated result scrolling detected; forcing a non-scroll decision...")
                cosmic_log.step(
                    step_num,
                    "search_results_governor.force_attempt",
                    recent_steps=context.get("recent_steps", []),
                )
                llm_response = await orchestrator.force_search_result_decision(
                    context=context,
                    screenshot_base64=screenshot_b64,
                )
                if llm_response:
                    print(f"   [Search-results governor] forced {llm_response.tool_call.action_type.value}.")
                    cosmic_log.step(
                        step_num,
                        "search_results_governor.force_action",
                        llm_response=llm_response.to_dict(),
                    )
                else:
                    print("   [Search-results governor] no safe forced action; continuing normal planner.")
                    cosmic_log.step(
                        step_num,
                        "search_results_governor.no_action",
                    )

            if llm_response is None:
                force_tier = None
                if pending_escalation:
                    # Base brain asked for help — give the frontier a small
                    # budget of consecutive steps to resolve the blocker.
                    force_tier = LLMTier.SLOW
                    escalation_hold = 3
                    pending_escalation = False
                elif escalation_hold > 0:
                    # Sticky: the frontier brain is mid-recovery.
                    force_tier = LLMTier.SLOW
                elif escalation_cooldown > 0:
                    # Frontier handed back — the base brain gets a fair chance
                    # before deterministic triggers may escalate again.
                    force_tier = LLMTier.FAST
                llm_response = await orchestrator.decide_action(
                    context=context,
                    screenshot_base64=screenshot_b64,
                    previous_confidence=previous_confidence,
                    force_tier=force_tier,
                )
                if llm_response.tier_used == "slow" and escalation_hold > 0:
                    escalation_hold -= 1
                if llm_response.tier_used == "fast" and escalation_cooldown > 0:
                    escalation_cooldown -= 1
                pending_escalation = False
            llm_time_ms = (time.time() - llm_start) * 1000

            # Live overlay: count this LLM-driven step. Replay-mode steps run
            # before this loop and aren't counted here, which is exactly the
            # "skipping exploration" story we want to show.
            live_llm_decisions += 1
            if demo_overlay.enabled:
                # First real agent activity in live mode — start the clock.
                demo_overlay.start_timer()
                demo_overlay.set_state(phase="Live planner active")
                demo_overlay.update_metrics(llm_calls=live_llm_decisions)
                try:
                    await demo_overlay.push(browser.page)
                except Exception:
                    pass
            
            print(f"\n🤖 LLM Decision:")
            print(f"   Action: {llm_response.tool_call.action_type.value}")
            print(f"   Params: {llm_response.tool_call.parameters}")
            print(f"   Confidence: {llm_response.confidence:.2f}")
            print(f"   Progress: {llm_response.estimated_completion:.0%}")
            print(f"   ⏱️  LLM time: {llm_time_ms:.0f}ms")

            if getattr(llm_response, "request_escalation", False):
                pending_escalation = True
                reason = (llm_response.escalation_reason or "base brain stuck").strip()
                print(f"   ⏫ Base brain requested escalation for the next step: {reason[:120]}")

            if llm_response.tier_used == "slow" and getattr(llm_response, "hand_back_to_base", False):
                escalation_hold = 0
                escalation_cooldown = 3
                print("   ⏬ Frontier brain reports the blocker is resolved — handing back to the base brain.")

            if llm_response.reasoning:
                print(f"   Reasoning: {llm_response.reasoning[:150]}...")
            cosmic_log.step(
                step_num,
                "llm.decision",
                llm_response=llm_response.to_dict(),
                llm_time_ms=llm_time_ms,
                previous_confidence=previous_confidence,
            )
            
            # 4. Execute action
            execution_start = time.time()
            action_result = None
            
            # Parse failure: no action ran. Record a teachable error step so
            # the model sees WHY nothing happened (the orchestrator already
            # retried once with a format correction).
            if getattr(llm_response, "parse_failed", False):
                print("   ⚠️  Model reply was not valid JSON — no action executed.")
                action_result = ActionResult(
                    success=False,
                    action_type=ActionType.PARSE_ERROR,
                    description="Decision was not valid JSON",
                    error=(
                        "Your previous decision was not valid JSON, so no action ran. Respond with ONLY "
                        "the JSON object per the output format (action_type, parameters, reasoning, "
                        "confidence) — no prose, no markdown fences — and re-issue your intended action."
                    ),
                    execution_time_ms=(time.time() - execution_start) * 1000,
                )
            # Special handling for ReadHistory
            elif llm_response.tool_call.action_type == ActionType.READ_HISTORY:
                try:
                    start_step = int(llm_response.tool_call.parameters.get("start_step", 1))
                    end_step = int(llm_response.tool_call.parameters.get("end_step", len(memory.steps)))
                    history_text = memory.read_history(start_step, end_step)
                    
                    action_result = ActionResult(
                        success=True,
                        action_type=ActionType.READ_HISTORY,
                        description=f"Read history from step {start_step} to {end_step}",
                        output=history_text,
                        execution_time_ms=(time.time() - execution_start) * 1000
                    )
                except Exception as e:
                     action_result = ActionResult(
                        success=False,
                        action_type=ActionType.READ_HISTORY,
                        description="Failed to read history",
                        error=str(e),
                        execution_time_ms=(time.time() - execution_start) * 1000
                    )

            # TimedWait circuit breaker: waiting is not the path forward when
            # recent steps show no progress. Hardened window — a stray failed
            # non-wait action (e.g. a malformed click) does NOT reset the
            # counter. Refuses the wait; the error text teaches the model.
            recent_three = memory.steps[-3:]
            timedwait_no_progress = sum(
                1 for s in recent_three
                if s.action
                and s.action.action_type == ActionType.TIMED_WAIT
                and s.action.verification_status in (VerificationStatus.NO_CHANGE, VerificationStatus.INCOMPLETE)
            )
            last_was_no_progress = bool(
                memory.steps
                and memory.steps[-1].action
                and memory.steps[-1].action.verification_status in (VerificationStatus.NO_CHANGE, VerificationStatus.INCOMPLETE)
            )
            if (
                llm_response.tool_call.action_type == ActionType.TIMED_WAIT
                and len(recent_three) >= 2
                and timedwait_no_progress >= 2
                and last_was_no_progress
            ):
                action_result = ActionResult(
                    success=False,
                    action_type=ActionType.TIMED_WAIT,
                    description="TimedWait refused",
                    error=(
                        "Recent waits already produced no change. Waiting again is blocked — "
                        "take a real action now: Navigate to a more specific search URL, click/type into the "
                        "page, DOMExtract the visible data, or SaveNote what you already have."
                    ),
                    execution_time_ms=0,
                )

            # ReadLargeNote circuit breaker: the whole point of the notes
            # system is that the agent writes something down once and can come
            # back to it. Re-reading a note whose text is already sitting in
            # WORKING SET is not coming back to it — it is the loop that burns
            # a run. Refuse and say where the content already is; the error
            # text teaches, exactly like the TimedWait breaker above.
            if (
                llm_response.tool_call.action_type == ActionType.READ_LARGE_NOTE
                and not llm_response.tool_call.parameters.get("start_line")
                and not llm_response.tool_call.parameters.get("end_line")
                and not action_result
            ):
                requested_note = str(llm_response.tool_call.parameters.get("note_id") or "").strip()
                already_visible = bool(requested_note) and any(
                    str(entry.get("label") or "") == f"note_id={requested_note}"
                    for entry in (context.get("working_set") or [])
                )
                if already_visible:
                    action_result = ActionResult(
                        success=False,
                        action_type=ActionType.READ_LARGE_NOTE,
                        description="ReadLargeNote refused",
                        error=(
                            f"{requested_note} is already shown in full under WORKING SET in this "
                            "prompt - read it there. Re-reading returns the same text and wastes a "
                            "step. Act on it now: SaveNote the answer it contains, navigate to what "
                            "it points at, or extract the one thing still missing. Use ReadLargeNote "
                            "again only with start_line/end_line for a part that was truncated."
                        ),
                        execution_time_ms=0,
                    )

            # Normal execution
            if not action_result:
                action_result = await browser.execute_tool(
                    llm_response.tool_call,
                    screenshot_path,
                )
            # execution_time_ms already tracked in action_result
            
            print(f"\n⚡ Execution: {'✓ Success' if action_result.success else '✗ Failed'}")
            if action_result.coordinates:
                print(f"   Coordinates: {action_result.coordinates}")
            if action_result.error:
                print(f"   Error: {action_result.error}")
            print(f"   ⏱️  Action execution time: {action_result.execution_time_ms:.0f}ms")
            cosmic_log.step(
                step_num,
                "action.executed",
                action_result=action_result.to_dict(),
            )
            
            # 5-7. Verify action. Read-only tools do not mutate page state, so avoid
            # the fixed settle delay plus an unnecessary after-screenshot capture.
            verification_start = time.time()
            read_only_action_types = {
                ActionType.DOM_EXTRACT,
                ActionType.BATCH_EXTRACT,
                ActionType.READ_HISTORY,
                ActionType.LIST_LARGE_NOTES,
                ActionType.SEARCH_LARGE_NOTES,
                ActionType.PARSE_ERROR,
            }
            if action_result.action_type in read_only_action_types:
                after_screenshot_path = screenshot_path
                after_screenshot_hash = screenshot_hash
                new_browser_state = browser_state
                verification_status = VerificationStatus.SUCCESS if action_result.success else VerificationStatus.ERROR
                change_score = 0.0
                verification_time_ms = (time.time() - verification_start) * 1000
            else:
                await asyncio.sleep(0.5)
                try:
                    after_screenshot_path, after_screenshot_hash, new_browser_state = await browser.capture_state(f"step_{step_num:03d}_after")
                    verification_status, change_score = await browser.verify_action(
                        before_state=browser_state,
                        after_state=new_browser_state,
                        verification_hint=llm_response.tool_call.verification_hint,
                        action_description=action_result.description,
                        action_type=llm_response.tool_call.action_type,
                    )
                except Exception as capture_err:
                    # The action already executed — a broken after-capture must
                    # not lose the whole run. Record the step against the last
                    # known state with an honest ERROR verification and move on.
                    print(f"⚠️  After-state capture failed ({capture_err}); recording step with last-known state and continuing.")
                    after_screenshot_path = screenshot_path
                    after_screenshot_hash = screenshot_hash
                    new_browser_state = browser_state
                    verification_status = VerificationStatus.ERROR
                    change_score = 0.0
                verification_time_ms = (time.time() - verification_start) * 1000
            
            action_result.verification_status = verification_status
            action_result.state_change_score = change_score
            action_result.estimated_completion = llm_response.estimated_completion

            # Auto-de-escalation: when the frontier brain's step verifies
            # success, the blocker it was summoned for is gone — hand control
            # back to the base brain (with a short cooldown so it gets a fair
            # chance before triggers may escalate again).
            if (
                llm_response.tier_used == "slow"
                and verification_status == VerificationStatus.SUCCESS
                and (escalation_hold > 0 or escalation_cooldown == 0)
            ):
                escalation_hold = 0
                escalation_cooldown = 2
                print("   ⏬ Frontier step verified success — handing control back to the base brain.")

            # Escalation episode discipline: if the frontier brain's step did
            # NOT verify success, the base brain must take a step before the
            # deterministic triggers may escalate again. Without this, a failing
            # escalation re-triggers itself forever (observed as 40+ straight
            # escalated steps on one stuck page).
            if llm_response.tier_used == "slow" and verification_status != VerificationStatus.SUCCESS:
                escalation_cooldown = max(escalation_cooldown, 1)
            
            print(f"\n✅ Verification:")
            print(f"   Status: {verification_status.value}")
            print(f"   State change: {change_score:.2f}")
            print(f"   ⏱️  Verification time: {verification_time_ms:.0f}ms")
            cosmic_log.step(
                step_num,
                "verification",
                after_screenshot_path=after_screenshot_path,
                after_screenshot_hash=after_screenshot_hash,
                after_browser_state=new_browser_state.to_dict(),
                verification_status=verification_status.value,
                change_score=change_score,
                verification_time_ms=verification_time_ms,
            )
            
            # 8. Add to memory
            memory.add_step(
                screenshot_path=screenshot_path,
                screenshot_hash=screenshot_hash,
                browser_state=new_browser_state,
                action=action_result,
                thinking=llm_response.reasoning,
                before_browser_state=browser_state,
                after_browser_state=new_browser_state,
                after_screenshot_path=after_screenshot_path,
                after_screenshot_hash=after_screenshot_hash,
                tool_call=llm_response.tool_call.to_dict() if hasattr(llm_response.tool_call, "to_dict") else {
                    "action_type": llm_response.tool_call.action_type.value,
                    "parameters": llm_response.tool_call.parameters,
                    "fallback": llm_response.tool_call.fallback,
                    "verification_hint": llm_response.tool_call.verification_hint,
                },
                llm_response=llm_response.to_dict(),
            )
            saved_step = memory.steps[-1] if memory.steps else None
            cosmic_log.step(
                step_num,
                "memory.persisted",
                log_path=str(memory.log_path),
                visual_index=saved_step.visual_index if saved_step else None,
            )

            # 8a. Progress hook (Cosmic orchestrator integration). Non-fatal by
            # design — a broken consumer must never kill a run.
            if step_callback is not None:
                try:
                    progress_info = {
                        "step": step_num,
                        "action_type": action_result.action_type.value if action_result else None,
                        "description": action_result.description if action_result else "",
                        "success": bool(action_result.success) if action_result else False,
                        "error": action_result.error if action_result else None,
                        "estimated_completion": llm_response.estimated_completion,
                        "tier_used": llm_response.tier_used,
                        "steps_taken": len(memory.steps),
                        "max_steps": config.max_steps,
                        # Live-view fields for the Cosmic desktop: the latest
                        # rendered screenshot plus where the page is now.
                        "screenshot_path": after_screenshot_path or screenshot_path,
                        "url": new_browser_state.url if new_browser_state else browser_state.url,
                        "page_title": new_browser_state.title if new_browser_state else browser_state.title,
                    }
                    hook_result = step_callback(progress_info)
                    if asyncio.iscoroutine(hook_result):
                        await hook_result
                except Exception as cb_err:
                    print(f"⚠️  step_callback error (non-fatal): {cb_err}")

            # 8b. Compress history if due. Run synchronously (not as a
            # background task) — see compress_if_due's docstring for why:
            # firing it concurrently with the next step's own LLM call caused
            # unpredictable 30-150s stalls under provider concurrency limits.
            await memory.compress_if_due()

            # 9. Calculate and print step execution time
            step_duration_ms = (time.time() - step_start_time) * 1000
            print(f"\n⏱️  STEP {step_num} TOTAL TIME: {step_duration_ms:.0f}ms ({step_duration_ms/1000:.2f}s)")
            print(f"   Breakdown: Capture={capture_time_ms:.0f}ms | LLM={llm_time_ms:.0f}ms | " +
                  f"Action={action_result.execution_time_ms:.0f}ms | Verify={verification_time_ms:.0f}ms")
            cosmic_log.step(
                step_num,
                "complete",
                step_duration_ms=step_duration_ms,
                capture_time_ms=capture_time_ms,
                llm_time_ms=llm_time_ms,
                action_time_ms=action_result.execution_time_ms,
                verification_time_ms=verification_time_ms,
            )
            
            # 10. Update confidence for next iteration
            previous_confidence = llm_response.confidence
            
            # 11. Check completion
            # 11. Check completion
            if action_result and action_result.action_type == ActionType.REQUEST_CREDENTIALS and action_result.success:
                print("\n🔑 Credentials needed - ending run for orchestrator follow-up")
                task_status = "credentials_needed"
                credentials_request = action_result
                break
            if llm_response.estimated_completion >= 0.95:
                if action_result and action_result.success:
                    print("\n🎉 GOAL ACHIEVED - Task complete!")
                    task_status = "success"
                    await demo_overlay.update(
                        page=browser.page,
                        phase="Saved answer",
                        pulse_ms=2000,
                        timeline_append={"kind": "saved", "label": "Answer saved"},
                    )
                    # Print the agent's saved notes as the final answer
                    final_notes = []
                    if memory.steps:
                        last_state = memory.steps[-1].browser_state
                        if last_state and last_state.notes:
                            final_notes = last_state.notes
                    if final_notes:
                        print("\n" + "="*80)
                        print("📋 RESULT")
                        print("="*80)
                        for note in final_notes:
                            print(f"  {note}")
                        print("="*80)
                    break
                else:
                    print("\n⚠️  LLM indicated completion but action FAILED - Continuing...")
                    previous_confidence = 0.0 # Force escalation
            
            # 12. Check failure conditions
            if verification_status == VerificationStatus.LOOP_DETECTED:
                print("\n⚠️  Loop detected - attempting recovery...")
                previous_confidence = 0.0  # Force escalation
                
            if verification_status == VerificationStatus.ERROR:
                if getattr(llm_response, "parse_failed", False):
                    # A format failure is recoverable — the orchestrator
                    # already retried once and the next step may parse fine.
                    # Never abort the whole run for it.
                    pass
                elif llm_response.confidence < 0.3:
                    print("\n❌ CRITICAL ERROR - Manual intervention needed")
                    task_status = "failed"
                    break
    finally:
        # Calculate total execution time
        total_duration_sec = time.time() - task_start_time
        
        # Final statistics
        print(f"\n{'='*80}")
        print(f"TASK COMPLETE")
        print(f"{'='*80}")
        print(f"\n📊 Execution Summary:")
        print(f"   Steps taken: {len(memory.steps)}")
        print(f"   ⏱️  Total execution time: {total_duration_sec:.2f}s ({total_duration_sec/60:.2f} minutes)")
        print(f"   ⏱️  Average time per step: {(total_duration_sec / len(memory.steps)):.2f}s" if memory.steps else "   N/A")
        
        print(f"\n💾 Memory Statistics:")
        mem_stats = memory.get_stats()
        summary_provider_hint = mem_stats.get("summary_provider")
        for key, value in mem_stats.items():
            print(f"   {key}: {display_stat_value(key, value, provider_hint=summary_provider_hint)}")
        
        print(f"\n🤖 Orchestrator Statistics:")
        orch_stats = orchestrator.get_stats()
        for key, value in orch_stats.items():
            print(f"   {key}: {value}")
        
        print(f"\n🌐 Browser Statistics:")
        browser_stats = browser.get_stats()
        for key, value in browser_stats.items():
            print(f"   {key}: {value}")
        
        # Save checkpoint
        checkpoint_path = memory.save_checkpoint()
        print(f"\n💾 Checkpoint saved: {checkpoint_path}")
        cosmic_log.event(
            "run.final_stats",
            task_status=task_status,
            steps_taken=len(memory.steps),
            total_duration_sec=total_duration_sec,
            checkpoint_path=str(checkpoint_path),
            memory_stats=memory.get_stats(),
            orchestrator_stats=orchestrator.get_stats(),
            browser_stats=browser.get_stats(),
        )

        if memory_mode in {"learn", "auto"} and cosmic_runtime:
            status = task_status if memory.steps else "empty"
            # compile_run blocks on two synchronous network calls (the indexer
            # LLM and the Supermemory write — the latter has no SDK timeout).
            # Run it off the event loop with a hard bound so end-of-run
            # finalization can never freeze the process for a minute.
            try:
                workflow = await asyncio.wait_for(
                    asyncio.to_thread(
                        cosmic_runtime.compile_run,
                        task=goal,
                        steps=memory.steps,
                        run_dir=str(working_dir),
                        status=status,
                    ),
                    timeout=120,
                )
            except asyncio.TimeoutError:
                print("⚠️  COSMIC indexing timed out after 120s — workflow not saved this run.")
                workflow = None
            if workflow:
                print(f"🧠 COSMIC workflow indexed: {workflow.get('workflow_id')}")
                cosmic_log.memory(
                    "workflow.indexed",
                    workflow_id=workflow.get("workflow_id"),
                    domain=workflow.get("domain"),
                    task_signature=workflow.get("task_signature"),
                    indexer=workflow.get("indexer"),
                    quality=workflow.get("quality"),
                    generalization_level=workflow.get("generalization_level"),
                    step_count=len(workflow.get("steps") or []),
                    discarded_step_count=len(workflow.get("discarded_steps") or []),
                    page_state_count=len(workflow.get("page_states") or []),
                    supermemory_enabled=cosmic_runtime.supermemory.enabled,
                    supermemory_last_error=cosmic_runtime.supermemory.last_error,
                )
                if cosmic_runtime.supermemory.enabled:
                    print("🧠 COSMIC semantic memory write requested via Supermemory.")
                elif cosmic_runtime.supermemory.last_error:
                    print(f"⚠️  COSMIC semantic write skipped: {cosmic_runtime.supermemory.last_error}")
        print(f"{'='*80}\n")

        # Cleanup — errors here are non-fatal (browser/LLM clients closing);
        # swallow them so they don't shadow the task result in main().
        try:
            await browser.stop_live_screencast()
        except Exception:
            pass
        try:
            await browser.close()
        except Exception as _e:
            print(f"⚠️  browser.close() error (non-fatal): {_e}")
        try:
            await orchestrator.close()
        except Exception as _e:
            print(f"⚠️  orchestrator.close() error (non-fatal): {_e}")

    final_answer = _extract_final_answer(memory)
    # One short recall-ledger line for the Cosmic-OS wrapper's session index —
    # reuses the same compression LLM already configured for cumulative_summary
    # (see MemoryManager.build_recall_summary). Fails soft: an empty string
    # here just means the caller falls back to final_answer/goal text itself.
    try:
        recall_summary = await memory.build_recall_summary(goal=goal, final_answer=final_answer)
    except Exception as _e:
        print(f"⚠️  recall summary build failed (non-fatal): {_e}")
        recall_summary = ""

    return {
        "success": True,
        "task_status": task_status,
        "steps_taken": len(memory.steps),
        "total_time_sec": total_duration_sec,
        "avg_time_per_step_sec": total_duration_sec / len(memory.steps) if memory.steps else 0,
        "checkpoint_path": str(checkpoint_path),
        "working_dir": str(working_dir),
        "cosmic_replay": replay_summary,
        "final_answer": final_answer,
        "recall_summary": recall_summary,
        "credentials_needed": credentials_request.output if credentials_request else None,
        "llm_usage": orchestrator.get_stats().get("llm_usage"),
    }

def _extract_final_answer(memory) -> str:
    """Best-effort extraction of the run's final answer from saved notes.

    Prefers the note the agent explicitly prefixed with 'FINAL ANSWER';
    falls back to the most recent note.
    """
    try:
        if not memory.steps:
            return ""
        last_state = memory.steps[-1].browser_state
        final_notes = last_state.notes if last_state else []
        if not final_notes:
            return ""
        preferred = [note for note in final_notes if "FINAL ANSWER" in note]
        return str(preferred[-1] if preferred else final_notes[-1])
    except Exception:
        return ""

def _print_chrome_profiles() -> None:
    """Print available Chrome profiles to stdout."""
    if sys.platform == "win32":
        user_data = Path(os.environ.get("LOCALAPPDATA", "")) / "Google" / "Chrome" / "User Data"
    elif sys.platform == "darwin":
        user_data = Path.home() / "Library" / "Application Support" / "Google" / "Chrome"
    else:
        user_data = Path.home() / ".config" / "google-chrome"

    if not user_data.is_dir():
        print(f"Chrome User Data directory not found at {user_data}")
        return

    print(f"Chrome profiles in: {user_data}\n")
    found = False
    for entry in sorted(user_data.iterdir()):
        prefs = entry / "Preferences"
        if not prefs.is_file():
            continue
        try:
            data = json.loads(prefs.read_text(encoding="utf-8", errors="replace"))
            name = data.get("profile", {}).get("name") or data.get("account_info", [{}])[0].get("full_name") or entry.name
        except Exception:
            name = entry.name
        print(f"  {entry.name:<20} — {name}")
        found = True

    if not found:
        print("  (no profiles found)")
    print(f"\nUsage: --chrome-profile 'Default'  or  --chrome-profile 'Profile 1'")


async def main():
    # 1. Argument Parsing
    parser = argparse.ArgumentParser(description="Cosmic Browser Use Agent")
    parser.add_argument("--goal", type=str, default=None, help="The task you want the agent to perform.")
    parser.add_argument("--provider", type=str, metavar="PROVIDER", default="fireworks_kimi", help=cli_provider_help())
    parser.add_argument("--url", type=str, default=None, help="Starting URL (optional).")
    parser.add_argument("--steps", type=int, default=1000, help="Max steps (default 1000).")
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode.")
    parser.add_argument("--mimo-url", type=str, default=os.getenv("MIMO_API_URL", MIMO_DEFAULT_URL), help="MiMo API URL.")
    parser.add_argument("--mimo-api-key", type=str, default=os.getenv("MIMO_API_KEY"), help="MiMo API Key for authentication.")

    # Model overrides (override defaults for the selected provider)
    parser.add_argument("--fast-model", type=str, default=None, help="Model ID for fast tier (overrides provider default).")
    parser.add_argument("--slow-model", type=str, default=None, help="Model ID for slow tier (overrides provider default).")
    parser.add_argument("--api-key", type=str, default=None, help="API key for the selected provider.")
    parser.add_argument("--temperature", type=float, default=None, help="LLM temperature (overrides provider default).")

    # Agent tuning
    parser.add_argument("--summary-interval", type=int, default=int(os.getenv("SUMMARY_INTERVAL_STEPS", "10")), help="Steps between memory compressions.")
    parser.add_argument("--max-tabs", type=int, default=int(os.getenv("MAX_TABS", "5")), help="Maximum open browser tabs.")
    parser.add_argument("--screenshot-quality", type=int, default=int(os.getenv("SCREENSHOT_QUALITY", "50")), help="Screenshot JPEG quality 1-100.")
    parser.add_argument("--ask-user-timeout", type=int, default=int(os.getenv("ASK_USER_TIMEOUT", "120")), help="Seconds to wait for user response.")
    parser.add_argument("--large-notes-path", type=str, default=None, help="Path to external large-notes JSONL file. Default: <run working dir>/large_notes.jsonl")
    parser.add_argument("--memory-mode", type=str, choices=["off", "learn", "recall", "auto"], default=os.getenv("BROWSER_MEMORY_MODE") or os.getenv("COSMIC_MEMORY_MODE", "off"), help="Browser traversal memory mode: off, learn, recall, or auto.")
    parser.add_argument("--memory-dir", type=str, default=os.getenv("BROWSER_MEMORY_DIR") or os.getenv("COSMIC_MEMORY_DIR"), help="Directory for local browser workflow memory (default: ./data/browser_memory, adopts legacy ./data/cosmic_memory when present).")
    parser.add_argument("--cosmic-user-id", type=str, default=os.getenv("COSMIC_USER_ID", "demo_user"), help="User/container identity for COSMIC memory.")
    parser.add_argument("--cosmic-container-tag", type=str, default=os.getenv("COSMIC_CONTAINER_TAG", "cosmic-hackathon-demo"), help="Supermemory container tag for COSMIC memories.")
    parser.add_argument("--disable-supermemory", action="store_true", help="Use only local workflow memory; skip Supermemory reads/writes.")
    parser.add_argument("--replay-max-actions", type=int, default=int(os.getenv("COSMIC_REPLAY_MAX_ACTIONS", "8")), help="Maximum indexed replay actions before returning to the normal agent loop.")
    parser.add_argument("--interaction-mode", type=str, choices=["hybrid", "vision"], default=os.getenv("COSMIC_INTERACTION_MODE", "hybrid"), help="Tool surface mode. 'vision' removes DOM tools/prompts; 'hybrid' keeps DOM fallback/extraction.")
    parser.add_argument("--demo-overlay", action="store_true", help="Demo-only: render a glassy COSMIC memory overlay inside the browser. Hidden during every agent/MiMo screenshot. Off by default.")
    parser.add_argument(
        "--ask-user-bridge-url",
        type=str,
        default=None,
        help=(
            "Route AskUser through an HTTP bridge instead of CLI input. "
            "Used by AgentPhone/call_to_browse.py: POSTs the question to <url>/ask and "
            "long-polls <url>/next_reply for the caller's spoken reply."
        ),
    )
    parser.add_argument(
        "--chrome-profile",
        type=str,
        default=None,
        metavar="PROFILE_DIR",
        help=(
            "Use an existing Chrome profile via CDP (e.g. 'Default', 'Profile 1'). "
            "Pass the profile directory name as it appears under Chrome's User Data folder. "
            "The agent launches your real Chrome binary against a persistent, dedicated "
            "agent copy of that profile (seeded with its logins and cookies), so your own "
            "Chrome windows are never closed or modified. Example: --chrome-profile 'Default'"
        ),
    )
    parser.add_argument(
        "--list-chrome-profiles",
        action="store_true",
        help="List available Chrome profiles and exit.",
    )
    parser.add_argument(
        "--restore-tabs",
        action="store_true",
        help=(
            "With --chrome-profile: best-effort reopen (in the agent's browser) of the tabs "
            "that were open in your live Chrome profile. This CANNOT preserve unsaved form "
            "text, scroll position, or other in-memory state — only the URLs, freshly "
            "reloaded. Off by default."
        ),
    )
    parser.add_argument(
        "--refresh-chrome-profile",
        action="store_true",
        help=(
            "With --chrome-profile: re-seed the agent's persistent Chrome data dir from your "
            "real profile (picks up logins you've done in your own Chrome since the first "
            "run). For a complete refresh, close Chrome first so no files are locked."
        ),
    )

    args = parser.parse_args()

    if args.list_chrome_profiles:
        _print_chrome_profiles()
        return

    if not args.goal:
        parser.error("the following arguments are required: --goal")

    args.provider = normalize_cli_provider_arg(args.provider)
    if args.provider not in {"openai", "anthropic", "gemini", "fireworks_kimi"}:
        parser.error(f"unsupported provider '{args.provider}'. Choose one of: {cli_allowed_provider_labels()}")

    print("\n" + "="*80)
    print("COSMIC BROWSER USE AGENT")
    print("Vision-Based Browser Automation with MiMo-VL")
    print("="*80)

    # 2. Setup Config based on Provider selection
    #    CLI flags --fast-model, --slow-model, --api-key, --temperature override defaults
    medium_config = None
    if args.provider == "openai":
        api_key = args.api_key or os.getenv("OPENAI_API_KEY")
        fast_model = args.fast_model or os.getenv("OPENAI_FAST_MODEL", "gpt-4o")
        slow_model = args.slow_model or os.getenv("OPENAI_SLOW_MODEL", "gpt-4o")
        default_temp = 0.3
        fast_config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model_id=fast_model,
            api_key=api_key,
            tier=LLMTier.FAST,
            timeout_ms=15000,
            temperature=args.temperature if args.temperature is not None else default_temp,
        )
        slow_config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model_id=slow_model,
            api_key=api_key,
            tier=LLMTier.SLOW,
            timeout_ms=90000,
            temperature=args.temperature if args.temperature is not None else default_temp,
        )
    elif args.provider == "anthropic":
        api_key = args.api_key or os.getenv("ANTHROPIC_API_KEY")
        fast_model = args.fast_model or os.getenv("CLAUDE_FAST_MODEL", "claude-haiku-4-5")
        slow_model = args.slow_model or os.getenv("CLAUDE_SLOW_MODEL", "claude-haiku-4-5")
        default_temp = 0.3
        fast_config = LLMConfig(
            provider=LLMProvider.CLAUDE,
            model_id=fast_model,
            api_key=api_key,
            tier=LLMTier.FAST,
            timeout_ms=15000,
            temperature=args.temperature if args.temperature is not None else default_temp,
        )
        slow_config = LLMConfig(
            provider=LLMProvider.CLAUDE,
            model_id=slow_model,
            api_key=api_key,
            tier=LLMTier.SLOW,
            timeout_ms=90000,
            temperature=args.temperature if args.temperature is not None else default_temp,
        )
    elif args.provider == "gemini":
        api_key = args.api_key or os.getenv("GEMINI_API_KEY")
        default_gemini_model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
        fast_model = args.fast_model or os.getenv("GEMINI_FAST_MODEL") or default_gemini_model
        medium_model = os.getenv("GEMINI_MEDIUM_MODEL") or fast_model
        slow_model = args.slow_model or os.getenv("GEMINI_SLOW_MODEL") or default_gemini_model
        default_temp = 1.0  # Gemini default - DO NOT change unless user explicitly overrides
        fast_config = LLMConfig(
            provider=LLMProvider.GEMINI,
            model_id=fast_model,
            api_key=api_key,
            tier=LLMTier.FAST,
            timeout_ms=15000,
            temperature=args.temperature if args.temperature is not None else default_temp,
        )
        medium_config = LLMConfig(
            provider=LLMProvider.GEMINI,
            model_id=medium_model,
            api_key=api_key,
            tier=LLMTier.MEDIUM,
            timeout_ms=45000,
            temperature=args.temperature if args.temperature is not None else default_temp,
        )
        slow_config = LLMConfig(
            provider=LLMProvider.GEMINI,
            model_id=slow_model,
            api_key=api_key,
            tier=LLMTier.SLOW,
            timeout_ms=90000,
            temperature=args.temperature if args.temperature is not None else default_temp,
        )
    elif args.provider == "fireworks_kimi":
        api_key = (
            args.api_key
            or os.getenv("FIREWORKS_API_KEY")
            or os.getenv("SLIDE_AGENT_FIREWORKS_API_KEY")
            or ""
        )
        api_key = api_key.strip() if api_key else ""
        if not api_key:
            print(
                "\n❌ Provider credentials are required for --provider fireworks_kimi "
                "(pass --api-key or set the configured provider API key env var)."
            )
            sys.exit(1)
        base_url = (os.getenv("FIREWORKS_BASE_URL") or "https://api.fireworks.ai/inference/v1").rstrip("/")
        default_model = resolve_fireworks_default_model()
        # Two-brain design: base brain (GLM 5.3 Flash on Fireworks) handles every
        # routine step; the escalation brain (frontier model on xAI) takes over
        # only on strong stuck-signals (see Orchestrator._select_tier) or when
        # the base brain itself requests escalation.
        fast_model = (args.fast_model or os.getenv("FIREWORKS_FAST_MODEL") or default_model).strip()
        default_temp = float(os.getenv("FIREWORKS_TEMPERATURE", "0.2"))
        # Headroom for reasoning-style models (GLM/Kimi emit thinking before the
        # action JSON): reasoning tokens count toward max_tokens, so generous
        # limits prevent truncated tool calls. Timeouts are env-tunable.
        max_fast = int(os.getenv("FIREWORKS_FAST_MAX_TOKENS", "2048"))
        max_slow = int(os.getenv("FIREWORKS_SLOW_MAX_TOKENS", "4096"))
        fast_timeout_ms = int(os.getenv("FIREWORKS_FAST_TIMEOUT_MS", "30000"))
        slow_timeout_ms = int(os.getenv("FIREWORKS_SLOW_TIMEOUT_MS", "90000"))
        fast_config = LLMConfig(
            provider=LLMProvider.FIREWORKS_KIMI,
            model_id=fast_model,
            api_key=api_key,
            api_base=base_url,
            tier=LLMTier.FAST,
            timeout_ms=fast_timeout_ms,
            max_tokens=max_fast,
            temperature=args.temperature if args.temperature is not None else default_temp,
        )

        # Escalation brain: frontier model (default: grok-4.6 on xAI).
        # A Fireworks-style --slow-model override stays on Fireworks; anything
        # else runs on xAI. Without an xAI key, escalation degrades gracefully.
        escalation_model = (args.slow_model or os.getenv("ESCALATION_MODEL") or resolve_escalation_model()).strip()
        if "fireworks" in escalation_model or "kimi" in escalation_model:
            slow_config = LLMConfig(
                provider=LLMProvider.FIREWORKS_KIMI,
                model_id=escalation_model,
                api_key=api_key,
                api_base=base_url,
                tier=LLMTier.SLOW,
                timeout_ms=slow_timeout_ms,
                max_tokens=max_slow,
                temperature=args.temperature if args.temperature is not None else default_temp,
            )
            print(f"Escalation brain: fireworks:{escalation_model}")
        elif (xai_key := (os.getenv("XAI_API_KEY") or "").strip()):
            escalation_base = (os.getenv("XAI_BASE_URL") or XAI_BASE_URL).rstrip("/")
            slow_config = LLMConfig(
                provider=LLMProvider.XAI,
                model_id=escalation_model,
                api_key=xai_key,
                api_base=escalation_base,
                tier=LLMTier.SLOW,
                timeout_ms=slow_timeout_ms,
                max_tokens=max_slow,
                temperature=args.temperature if args.temperature is not None else default_temp,
            )
            print(f"Escalation brain: xai:{escalation_model}")
        else:
            # No xAI key — degrade gracefully: escalation falls back to the base
            # brain with a longer timeout and bigger output budget. The run still
            # works; stuck-signal escalation just loses the frontier boost.
            print("⚠️  XAI_API_KEY not set — escalation tier falls back to the base brain (longer timeout, bigger budget).")
            slow_config = LLMConfig(
                provider=LLMProvider.FIREWORKS_KIMI,
                model_id=fast_model,
                api_key=api_key,
                api_base=base_url,
                tier=LLMTier.SLOW,
                timeout_ms=slow_timeout_ms,
                max_tokens=max_slow,
                temperature=args.temperature if args.temperature is not None else default_temp,
            )

    # 3. Pre-check MiMo Availability
    # Modal cold starts boot vLLM after the first request connects — the read
    # timeout must be long enough for the container to answer while booting,
    # and the total budget long enough for a full cold boot (~20-60s).
    mimo_health_timeout = int(os.getenv("MIMO_HEALTH_TIMEOUT", "25"))
    mimo_health_total_wait = int(os.getenv("MIMO_HEALTH_TOTAL_WAIT", "180"))
    print("\n[PRE-CHECK] verifying MiMo vision server...")
    if not check_mimo_health(
        args.mimo_url,
        timeout=mimo_health_timeout,
        api_key=args.mimo_api_key,
        total_wait=mimo_health_total_wait,
    ):
        print(f"\n❌ CRITICAL ERROR: MiMo Vision Server is unreachable at: {args.mimo_url}")
        print("   This is a vision-dominant system and cannot function without MiMo.")
        print("   Please ensure the server is running and accessible.")
        sys.exit(1)
    print("✅ MiMo-VL Server is online and ready.")

    ask_user_handler = None
    if args.ask_user_bridge_url:
        bridge_url = args.ask_user_bridge_url.rstrip("/")
        import httpx as _httpx  # local import keeps the default path dependency-free

        async def _bridge_ask_user(question: str, kind: str = "") -> str:
            poll_timeout = max(15.0, float(args.ask_user_timeout))
            async with _httpx.AsyncClient(timeout=poll_timeout + 5.0) as http:
                ask_resp = await http.post(f"{bridge_url}/ask", json={"question": question, "kind": kind})
                ask_resp.raise_for_status()
                while True:
                    try:
                        reply_resp = await http.get(
                            f"{bridge_url}/next_reply",
                            params={"timeout": poll_timeout},
                        )
                    except _httpx.ReadTimeout:
                        continue
                    if reply_resp.status_code == 204:
                        continue
                    reply_resp.raise_for_status()
                    payload = reply_resp.json() or {}
                    if payload.get("timeout"):
                        continue
                    reply_text = payload.get("reply")
                    if reply_text is None:
                        continue
                    return str(reply_text)

        ask_user_handler = _bridge_ask_user
        print(f"🎙️  AskUser bridge: {bridge_url} (questions will be spoken on a live call)")

    # 4. RunTask
    try:
        await run_task(
            goal=args.goal,
            initial_url=args.url,
            max_steps=args.steps,
            fast_model_config=fast_config,
            medium_model_config=medium_config,
            slow_model_config=slow_config,
            mimo_api_url=args.mimo_url,
            mimo_api_key=args.mimo_api_key,
            headless=args.headless,
            summary_interval=args.summary_interval,
            max_tabs=args.max_tabs,
            screenshot_quality=args.screenshot_quality,
            ask_user_timeout=args.ask_user_timeout,
            large_notes_path=args.large_notes_path,
            memory_mode=args.memory_mode,
            memory_dir=args.memory_dir,
            cosmic_user_id=args.cosmic_user_id,
            cosmic_container_tag=args.cosmic_container_tag,
            supermemory_enabled=not args.disable_supermemory,
            replay_max_actions=args.replay_max_actions,
            interaction_mode=args.interaction_mode,
            demo_overlay_enabled=args.demo_overlay,
            ask_user_handler=ask_user_handler,
            chrome_profile=args.chrome_profile,
            restore_previous_tabs=args.restore_tabs,
            refresh_chrome_profile=args.refresh_chrome_profile,
        )
    except Exception as e:
        print(f"\n❌ Execution Failed: {e}")
        # raise # Uncomment to see full traceback

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Task interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        raise
