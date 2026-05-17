#!/usr/bin/env python3
"""Smoke test COSMIC's Supermemory integration.

This script intentionally uses an isolated container tag/user pair so it does
not pollute the demo memory namespace.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cosmic_memory.supermemory_client import SupermemoryMemoryClient


def now_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def make_workflow(marker: str) -> dict:
    return {
        "workflow_id": f"smoke_youtube_description_{marker}",
        "name": f"Supermemory smoke workflow {marker}",
        "domain": "youtube.com",
        "intent": f"Get YouTube video description smoke marker {marker}",
        "task_signature": f"youtube_description_smoke_{marker}",
        "summary": (
            "COSMIC Supermemory smoke test workflow. "
            f"Unique retrieval marker: {marker}. "
            "Stores reusable browser traversal route memory with visual indexes."
        ),
        "success_count": 1,
        "page_states": [
            {
                "state_id": "youtube_home_search",
                "name": "YouTube homepage search",
                "url_pattern": "youtube.com",
                "page_summary": "YouTube homepage with search box.",
                "visual_markers": ["youtube", "search"],
            }
        ],
        "steps": [
            {
                "step_id": "step_001",
                "from_state_id": "youtube_home_search",
                "action_type": "VisualType",
                "target_description": "YouTube search box",
                "parameters_template": {
                    "field_description": "YouTube search box",
                    "text": "<video search query>",
                    "press_enter": True,
                },
                "visual_index": {
                    "target_description": "YouTube search box",
                    "coordinate_source": "mimo_grounded",
                    "viewport": {"width": 1280, "height": 720},
                    "pixel_coordinates": {"x": 640, "y": 90},
                    "normalized_coordinates": {"x": 0.5, "y": 0.125},
                },
                "expected_result": "YouTube results page loads.",
                "to_state_id": "youtube_results",
                "delay_ms": 1000,
                "checkpoint": True,
                "success": True,
            }
        ],
        "failure_patches": [],
        "user_preferences": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test COSMIC Supermemory workflow writes/searches.")
    parser.add_argument("--container-tag", default="cosmic-smoke", help="Base Supermemory container tag.")
    parser.add_argument("--user-id", default=None, help="Smoke user id. Defaults to a timestamped id.")
    parser.add_argument("--retries", type=int, default=8, help="Search retries while indexing completes.")
    parser.add_argument("--sleep-sec", type=float, default=3.0, help="Delay between search retries.")
    parser.add_argument("--out", default="data/smoke_supermemory", help="Directory for smoke reports.")
    args = parser.parse_args()

    marker = now_slug()
    user_id = args.user_id or f"smoke_{marker}"
    report_dir = Path(args.out)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"supermemory_smoke_{marker}.json"

    api_key_present = bool(os.getenv("SUPERMEMORY_API_KEY"))
    client = SupermemoryMemoryClient(
        container_tag=args.container_tag,
        user_id=user_id,
        enabled=True,
        task_type="memory",
    )

    workflow = make_workflow(marker)
    report = {
        "marker": marker,
        "container_tag": args.container_tag,
        "space_tag": client.space_tag,
        "user_id": user_id,
        "task_type": client.task_type,
        "api_key_present": api_key_present,
        "sdk_enabled": client.enabled,
        "add_ok": False,
        "document_get_ok": False,
        "document_status": None,
        "extracted_memory_count": 0,
        "search_ok": False,
        "add_response": None,
        "search_attempts": [],
        "last_error": None,
    }

    if not client.enabled:
        report["last_error"] = client.last_error or "SUPERMEMORY_API_KEY is not set or SDK could not initialize."
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
        return 2

    add_response = client.remember_workflow(workflow)
    report["add_response"] = client._result_to_dict(add_response) if add_response is not None else None
    report["add_ok"] = add_response is not None and not client.last_error
    report["last_error"] = client.last_error
    doc_id = None
    if report["add_response"]:
        doc_id = report["add_response"].get("id")

    if doc_id and client.client:
        for _ in range(max(1, args.retries)):
            try:
                document = client.client.documents.get(doc_id)
                doc_payload = client._result_to_dict(document)
                report["document_get_ok"] = True
                report["document_status"] = doc_payload.get("status")
                report["extracted_memory_count"] = len(doc_payload.get("memories") or [])
                if report["document_status"] == "done":
                    break
            except Exception as exc:
                report["last_error"] = f"Document get failed: {exc}"
            time.sleep(args.sleep_sec)

    for attempt in range(1, args.retries + 1):
        results = client.search(
            task=f"Find COSMIC smoke marker {marker} for a YouTube video description workflow",
            domain="youtube.com",
            limit=5,
        )
        hit = any(marker in json.dumps(result, ensure_ascii=False, default=str) for result in results)
        report["search_attempts"].append(
            {
                "attempt": attempt,
                "result_count": len(results),
                "hit": hit,
                "last_error": client.last_error,
                "results": results[:3],
            }
        )
        if hit:
            report["search_ok"] = True
            break
        time.sleep(args.sleep_sec)

    report["last_error"] = client.last_error
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return 0 if report["add_ok"] and report["search_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
