#!/usr/bin/env python3
"""Compile an existing COSMIC browser run into replay memory.

Input is a run directory containing log.json. The script writes the executable
workflow to the local workflow store and, unless disabled, uploads a semantic
workflow memory to Supermemory.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cosmic_memory.indexer import load_run_steps
from cosmic_memory.runtime import CosmicMemoryRuntime


def infer_goal(run_dir: Path) -> str:
    events_path = run_dir / "cosmic_debug" / "events.jsonl"
    if events_path.exists():
        for line in events_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event") == "run.start" and row.get("goal"):
                return str(row["goal"])
    checkpoint_files = sorted(run_dir.glob("checkpoint_*.json"), reverse=True)
    for path in checkpoint_files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        config = payload.get("config") or {}
        if config.get("goal"):
            return str(config["goal"])
    return ""


def infer_status(run_dir: Path) -> str:
    events_path = run_dir / "cosmic_debug" / "events.jsonl"
    if events_path.exists():
        for line in reversed(events_path.read_text(encoding="utf-8").splitlines()):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event") == "run.final_stats":
                return str(row.get("task_status") or "unknown")
    return "unknown"


def main() -> int:
    load_dotenv(dotenv_path=ROOT / ".env")
    parser = argparse.ArgumentParser(description="Index a completed COSMIC browser run into workflow memory.")
    parser.add_argument("--run-dir", required=True, help="Run directory containing log.json.")
    parser.add_argument("--goal", default=None, help="Goal for this run. Defaults to run.start goal when available.")
    parser.add_argument("--status", default=None, help="Run status. Defaults to run.final_stats task_status.")
    parser.add_argument("--memory-dir", default=os.getenv("COSMIC_MEMORY_DIR", "./data/cosmic_memory"), help="Local workflow memory directory.")
    parser.add_argument("--cosmic-user-id", default=os.getenv("COSMIC_USER_ID", "demo_user"), help="Supermemory user/container identity.")
    parser.add_argument("--cosmic-container-tag", default=os.getenv("COSMIC_CONTAINER_TAG", "cosmic-hackathon-demo"), help="Supermemory container tag.")
    parser.add_argument("--workflow-id", default=None, help="Optional explicit workflow ID.")
    parser.add_argument("--disable-supermemory", action="store_true", help="Write local workflow only.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}", file=sys.stderr)
        return 2
    if not (run_dir / "log.json").exists():
        print(f"Run log not found: {run_dir / 'log.json'}", file=sys.stderr)
        return 2

    goal = args.goal or infer_goal(run_dir)
    if not goal:
        print("Goal is required when it cannot be inferred from run logs.", file=sys.stderr)
        return 2
    status = args.status or infer_status(run_dir)

    runtime = CosmicMemoryRuntime(
        data_dir=args.memory_dir,
        user_id=args.cosmic_user_id,
        container_tag=args.cosmic_container_tag,
        supermemory_enabled=not args.disable_supermemory,
    )
    steps = load_run_steps(run_dir)
    workflow = runtime.compile_run(
        task=goal,
        steps=steps,
        run_dir=str(run_dir),
        status=status,
        workflow_id=args.workflow_id,
    )
    if not workflow:
        print("No workflow was compiled.", file=sys.stderr)
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
        "indexer_plan_path": str(run_dir / "cosmic_debug" / "indexer_plan.json"),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
