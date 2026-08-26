#!/usr/bin/env python3
"""Record a human-driven browser session and compile it into a replayable
COSMIC workflow.

Launches the agent's persistent Chrome (seeded from the user's real profile)
via the same CDP machinery the agent uses — the user's own Chrome windows are
never touched — shows an in-page overlay with a "Stop & Save Workflow" button,
and records DOM clicks/typed text/scrolls/navigation while the human browses
normally. On stop, the recorded trace is run through the same indexing
pipeline `scripts/index_run.py` uses (WorkflowRunIndexer -> TraceCompiler ->
WorkflowStore) unless --no-auto-compile is passed.

Usage:
    python scripts/record_workflow.py \
        --workflow-name "Apply to LinkedIn job" \
        --goal "Apply to a software engineer job posting on LinkedIn" \
        --chrome-profile "Profile 9"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from cosmic_types import TaskConfig
from browser_controller import BrowserController
from cosmic_memory.recorder import WorkflowRecorder
from cosmic_memory.runtime import CosmicMemoryRuntime


async def run(args: argparse.Namespace) -> int:
    working_dir = Path(f"./runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}_recorded")
    working_dir.mkdir(parents=True, exist_ok=True)

    config = TaskConfig(
        task_id=f"recording_{datetime.now().timestamp()}",
        goal=args.goal,
        chrome_profile=args.chrome_profile,
        refresh_chrome_profile=args.refresh_chrome_profile,
        screenshot_max_width=int(os.getenv("SCREENSHOT_MAX_WIDTH", "1280")),
        screenshot_quality=int(os.getenv("SCREENSHOT_QUALITY", "50")),
    )

    browser = BrowserController(
        config=config,
        # MiMo is never called during recording — clicks are captured via DOM
        # selectors or raw coordinates from the real click event, not vision
        # grounding. This URL is structurally required (the constructor
        # validates it's non-empty) but never actually invoked.
        mimo_api_url=os.getenv("MIMO_API_URL", "https://uspraveenraj--mimo-vl-7b-rl-serve.modal.run"),
        working_dir=working_dir,
        # A human drives this session — disable the agent's tab janitoring
        # (don't close the user's existing tabs or their deliberate new tabs).
        human_driven=True,
    )

    print("\n" + "=" * 80)
    print("COSMIC WORKFLOW RECORDER")
    print("=" * 80)
    print(f"Workflow name: {args.workflow_name}")
    print(f"Goal: {args.goal}")
    print(f"Chrome profile: {args.chrome_profile}")
    print(f"Working directory: {working_dir}")
    print("=" * 80 + "\n")

    # Deliberately do NOT pass initial_url here — add_init_script only
    # applies to navigations that happen AFTER it's registered. Navigating
    # before the recorder's listeners are installed would mean the first
    # page the user actually interacts with never gets instrumented.
    await browser.start(initial_url=None)

    recorder = WorkflowRecorder(
        browser_controller=browser,
        workflow_name=args.workflow_name,
        goal=args.goal,
        working_dir=working_dir,
    )
    await recorder.start()

    if args.url:
        await browser.page.goto(args.url, wait_until="domcontentloaded")

    try:
        await recorder.wait_for_stop()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted — stopping recording.")

    print(f"\n🛑 Recording stopped. {len(recorder.steps)} action(s) captured.")
    await browser.close()

    if not recorder.steps:
        print("No actions were recorded — nothing to compile.")
        return 0

    if args.no_auto_compile:
        print(f"--no-auto-compile set. Raw trace saved at: {recorder.log_path}")
        print(f"Compile later with: python scripts/index_run.py --run-dir {working_dir}")
        return 0

    print("\n📦 Compiling recorded trace into a workflow...")
    runtime = CosmicMemoryRuntime(
        data_dir=args.memory_dir,
        user_id=args.cosmic_user_id,
        container_tag=args.cosmic_container_tag,
        supermemory_enabled=not args.disable_supermemory,
    )
    workflow = runtime.compile_run(
        task=args.goal,
        steps=recorder.steps,
        run_dir=str(working_dir),
        status="success",
    )
    if not workflow:
        print("⚠️  No workflow was compiled (no usable action steps found).")
        return 1

    report = {
        "workflow_id": workflow.get("workflow_id"),
        "name": workflow.get("name"),
        "domain": workflow.get("domain"),
        "intent": workflow.get("intent"),
        "generalization_level": workflow.get("generalization_level"),
        "quality": workflow.get("quality"),
        "indexer": workflow.get("indexer"),
        "step_count": len(workflow.get("steps") or []),
        "discarded_step_count": len(workflow.get("discarded_steps") or []),
        "workflow_path": str(runtime.store.workflows_dir / f"{workflow.get('workflow_id')}.json"),
        "supermemory_enabled": runtime.supermemory.enabled,
        "supermemory_last_error": runtime.supermemory.last_error,
        "indexer_plan_path": str(working_dir / "cosmic_debug" / "indexer_plan.json"),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return 0


def main() -> int:
    load_dotenv(dotenv_path=ROOT / ".env")
    parser = argparse.ArgumentParser(description="Record a human browser session into a replayable COSMIC workflow.")
    parser.add_argument("--workflow-name", required=True, help="Short name for this workflow, e.g. 'Apply to LinkedIn job'.")
    parser.add_argument("--goal", required=True, help="What this workflow accomplishes, used as the task intent for indexing.")
    parser.add_argument("--chrome-profile", required=True, metavar="PROFILE_DIR", help="Chrome profile directory name (e.g. 'Default', 'Profile 1') — see main.py --list-chrome-profiles.")
    parser.add_argument("--refresh-chrome-profile", action="store_true", help="Re-seed the agent's persistent Chrome data dir from the real profile before recording.")
    parser.add_argument("--url", default=None, help="Optional starting URL to navigate to once recording is active.")
    parser.add_argument("--memory-dir", default=os.getenv("COSMIC_MEMORY_DIR", "./data/cosmic_memory"), help="Local workflow memory directory.")
    parser.add_argument("--cosmic-user-id", default=os.getenv("COSMIC_USER_ID", "demo_user"), help="Supermemory user/container identity.")
    parser.add_argument("--cosmic-container-tag", default=os.getenv("COSMIC_CONTAINER_TAG", "cosmic-hackathon-demo"), help="Supermemory container tag.")
    parser.add_argument("--disable-supermemory", action="store_true", help="Write local workflow only; skip Supermemory upload.")
    parser.add_argument("--no-auto-compile", action="store_true", help="Keep the raw recorded trace only; don't run the indexer automatically.")
    args = parser.parse_args()

    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
