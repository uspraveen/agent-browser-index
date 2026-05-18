#!/usr/bin/env python3
"""
Cosmic Browser Use Agent - Main execution loop with timing
"""
import asyncio
import base64
import argparse
import time
import sys
from pathlib import Path
from datetime import datetime

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
from find_coordinates_mimo import check_mimo_health
from cli_labels import (
    cli_allowed_provider_labels,
    cli_provider_help,
    display_provider_label,
    display_provider_model,
    display_stat_value,
    normalize_cli_provider_arg,
)
from cosmic_memory.debug_log import CosmicDebugLogger
from cosmic_memory.demo_overlay import DemoOverlayManager
from cosmic_memory.replay import execute_indexed_replay_plan
from cosmic_memory.runtime import CosmicMemoryRuntime

import os
from dotenv import load_dotenv

load_dotenv()
# ==============================================================================


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
    memory_dir: str = "./data/cosmic_memory",
    cosmic_user_id: str = "demo_user",
    cosmic_container_tag: str = "cosmic-hackathon-demo",
    supermemory_enabled: bool = True,
    replay_max_actions: int = 8,
    interaction_mode: str = "hybrid",
    demo_overlay_enabled: bool = False,
):
    mimo_api_url = mimo_api_url or os.getenv("MIMO_API_URL", "https://mbhl6tqhfyvdd3tu.us-east-1.aws.endpoints.huggingface.cloud/v1/")
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

    config = TaskConfig(
        task_id=f"task_{datetime.now().timestamp()}",
        goal=goal,
        max_steps=max_steps,
        screenshot_quality=screenshot_quality,
        summary_interval=summary_interval,
        max_tabs=max_tabs,
        ask_user_timeout=ask_user_timeout,
        enable_dom_fallback=enable_dom_fallback,
    )
    
    print(f"\n{'='*80}")
    print(f"COSMIC BROWSER USE AGENT - Task Started")
    print(f"{'='*80}")
    print(f"Goal: {goal}")
    print(f"Provider: {display_provider_label(fast_model_config.provider)}")
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
    summary_provider = (
        os.getenv("SUMMARY_LLM_PROVIDER")
        or ("fireworks_kimi" if fast_model_config.provider == LLMProvider.FIREWORKS_KIMI else "openai")
    ).strip().lower()
    if summary_provider in {"fireworks", "fireworks_kimi", "kimi"}:
        summary_model = os.getenv("SUMMARY_LLM_MODEL") or os.getenv("FIREWORKS_KIMI_MODEL", "accounts/fireworks/models/kimi-k2p6")
        summary_api_key = os.getenv("FIREWORKS_API_KEY") or os.getenv("SLIDE_AGENT_FIREWORKS_API_KEY")
        summary_api_base = os.getenv("FIREWORKS_BASE_URL") or "https://api.fireworks.ai/inference/v1"
    else:
        summary_model = os.getenv("SUMMARY_LLM_MODEL", "gpt-4o-mini")
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
    )

    await browser.start(initial_url)
    task_start_time = time.time()

    cosmic_runtime = None
    retrieved_memory = None
    replay_summary = None
    if memory_mode in {"learn", "recall", "auto"}:
        cosmic_runtime = CosmicMemoryRuntime(
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
        probe_screenshot_path, _, probe_state = await browser.capture_state("cosmic_memory_probe")
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
    checkpoint_path = None
    task_status = "incomplete"
    last_visible_answer_governor_step = 0
    last_search_results_governor_step = 0
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
            if memory.detect_loop():
                print("\n⚠️  LOOP DETECTED - Escalating to slow model")
                previous_confidence = 0.0
            
            # 3. Get LLM decision
            llm_start = time.time()
            context = memory.get_context_for_llm(screenshot_path)
            
            # Load screenshot as base64
            with open(screenshot_path, 'rb') as f:
                screenshot_b64 = f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode()}"

            llm_response = None
            if _should_try_visible_answer_governor(
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
                llm_response = await orchestrator.decide_action(
                    context=context,
                    screenshot_base64=screenshot_b64,
                    previous_confidence=previous_confidence,
                )
            llm_time_ms = (time.time() - llm_start) * 1000

            # Live overlay: count this LLM-driven step. Replay-mode steps run
            # before this loop and aren't counted here, which is exactly the
            # "skipping exploration" story we want to show.
            live_llm_decisions += 1
            if demo_overlay.enabled:
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
            
            # Special handling for ReadHistory
            if llm_response.tool_call.action_type == ActionType.READ_HISTORY:
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
                ActionType.READ_HISTORY,
                ActionType.LIST_LARGE_NOTES,
                ActionType.SEARCH_LARGE_NOTES,
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
                after_screenshot_path, after_screenshot_hash, new_browser_state = await browser.capture_state(f"step_{step_num:03d}_after")
                verification_status, change_score = await browser.verify_action(
                    before_state=browser_state,
                    after_state=new_browser_state,
                    verification_hint=llm_response.tool_call.verification_hint,
                )
                verification_time_ms = (time.time() - verification_start) * 1000
            
            action_result.verification_status = verification_status
            action_result.state_change_score = change_score
            action_result.estimated_completion = llm_response.estimated_completion
            
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
                if llm_response.confidence < 0.3:
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
            workflow = cosmic_runtime.compile_run(
                task=goal,
                steps=memory.steps,
                run_dir=str(working_dir),
                status=status,
            )
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
        
        # Cleanup
        await browser.close()
        await orchestrator.close()
    
    return {
        "success": True,
        "steps_taken": len(memory.steps),
        "total_time_sec": total_duration_sec,
        "avg_time_per_step_sec": total_duration_sec / len(memory.steps) if memory.steps else 0,
        "checkpoint_path": str(checkpoint_path),
        "working_dir": str(working_dir),
        "cosmic_replay": replay_summary,
    }

async def main():
    # 1. Argument Parsing
    parser = argparse.ArgumentParser(description="Cosmic Browser Use Agent")
    parser.add_argument("--goal", type=str, required=True, help="The task you want the agent to perform.")
    parser.add_argument("--provider", type=str, metavar="PROVIDER", default="openai", help=cli_provider_help())
    parser.add_argument("--url", type=str, default=None, help="Starting URL (optional).")
    parser.add_argument("--steps", type=int, default=1000, help="Max steps (default 1000).")
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode.")
    parser.add_argument("--mimo-url", type=str, default=os.getenv("MIMO_API_URL", "https://mbhl6tqhfyvdd3tu.us-east-1.aws.endpoints.huggingface.cloud/v1/"), help="MiMo API URL.")
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
    parser.add_argument("--memory-mode", type=str, choices=["off", "learn", "recall", "auto"], default=os.getenv("COSMIC_MEMORY_MODE", "off"), help="COSMIC traversal memory mode: off, learn, recall, or auto.")
    parser.add_argument("--memory-dir", type=str, default=os.getenv("COSMIC_MEMORY_DIR", "./data/cosmic_memory"), help="Directory for local COSMIC workflow memory.")
    parser.add_argument("--cosmic-user-id", type=str, default=os.getenv("COSMIC_USER_ID", "demo_user"), help="User/container identity for COSMIC memory.")
    parser.add_argument("--cosmic-container-tag", type=str, default=os.getenv("COSMIC_CONTAINER_TAG", "cosmic-hackathon-demo"), help="Supermemory container tag for COSMIC memories.")
    parser.add_argument("--disable-supermemory", action="store_true", help="Use only local workflow memory; skip Supermemory reads/writes.")
    parser.add_argument("--replay-max-actions", type=int, default=int(os.getenv("COSMIC_REPLAY_MAX_ACTIONS", "8")), help="Maximum indexed replay actions before returning to the normal agent loop.")
    parser.add_argument("--interaction-mode", type=str, choices=["hybrid", "vision"], default=os.getenv("COSMIC_INTERACTION_MODE", "hybrid"), help="Tool surface mode. 'vision' removes DOM tools/prompts; 'hybrid' keeps DOM fallback/extraction.")
    parser.add_argument("--demo-overlay", action="store_true", help="Demo-only: render a glassy COSMIC memory overlay inside the browser. Hidden during every agent/MiMo screenshot. Off by default.")

    args = parser.parse_args()
    args.provider = normalize_cli_provider_arg(args.provider)
    if args.provider not in {"openai", "anthropic", "gemini", "fireworks_kimi"}:
        parser.error(f"unsupported provider '{args.provider}'. Choose one of: {cli_allowed_provider_labels()}")

    print("\n" + "="*80)
    print("COSMIC BROWSER USE AGENT")
    print("Vision-Based Browser Automation with MiMo-VL")
    print("="*80)

    # 2. Setup Config based on Provider selection
    #    CLI flags --fast-model, --slow-model, --api-key, --temperature override defaults
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
        fast_model = args.fast_model or os.getenv("GEMINI_FAST_MODEL", "gemini-3-pro-preview")
        slow_model = args.slow_model or os.getenv("GEMINI_SLOW_MODEL", "gemini-3-pro-preview")
        default_temp = 1.0  # Gemini 3 default - DO NOT change unless user explicitly overrides
        fast_config = LLMConfig(
            provider=LLMProvider.GEMINI,
            model_id=fast_model,
            api_key=api_key,
            tier=LLMTier.FAST,
            timeout_ms=15000,
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
                "\n❌ Provider credentials are required for --provider google_gemini "
                "(pass --api-key or set the configured provider API key env var)."
            )
            sys.exit(1)
        base_url = (os.getenv("FIREWORKS_BASE_URL") or "https://api.fireworks.ai/inference/v1").rstrip("/")
        default_model = os.getenv("FIREWORKS_KIMI_MODEL", "accounts/fireworks/models/kimi-k2p6").strip()
        fast_model = (args.fast_model or os.getenv("FIREWORKS_FAST_MODEL") or default_model).strip()
        slow_model = (args.slow_model or os.getenv("FIREWORKS_SLOW_MODEL") or default_model).strip()
        default_temp = float(os.getenv("FIREWORKS_TEMPERATURE", "0.2"))
        max_fast = int(os.getenv("FIREWORKS_FAST_MAX_TOKENS", "1024"))
        max_slow = int(os.getenv("FIREWORKS_SLOW_MAX_TOKENS", "2048"))
        fast_config = LLMConfig(
            provider=LLMProvider.FIREWORKS_KIMI,
            model_id=fast_model,
            api_key=api_key,
            api_base=base_url,
            tier=LLMTier.FAST,
            timeout_ms=15000,
            max_tokens=max_fast,
            temperature=args.temperature if args.temperature is not None else default_temp,
        )
        slow_config = LLMConfig(
            provider=LLMProvider.FIREWORKS_KIMI,
            model_id=slow_model,
            api_key=api_key,
            api_base=base_url,
            tier=LLMTier.SLOW,
            timeout_ms=90000,
            max_tokens=max_slow,
            temperature=args.temperature if args.temperature is not None else default_temp,
        )

    # 3. Pre-check MiMo Availability
    print("\n[PRE-CHECK] verifying MiMo vision server...")
    if not check_mimo_health(args.mimo_url, api_key=args.mimo_api_key):
        print(f"\n❌ CRITICAL ERROR: MiMo Vision Server is unreachable at: {args.mimo_url}")
        print("   This is a vision-dominant system and cannot function without MiMo.")
        print("   Please ensure the server is running and accessible.")
        sys.exit(1)
    print("✅ MiMo-VL Server is online and ready.")

    # 4. RunTask
    try:
        await run_task(
            goal=args.goal,
            initial_url=args.url,
            max_steps=args.steps,
            fast_model_config=fast_config,
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
