"""Indexed replay execution."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

from cosmic_types import ActionResult, ActionType, ToolCall, VerificationStatus


def enrich_plan_actions(plan: Dict[str, Any], workflow: Dict[str, Any]) -> List[Dict[str, Any]]:
    steps_by_id = {step.get("step_id"): step for step in workflow.get("steps", [])}
    enriched = []
    for raw_action in plan.get("actions") or []:
        step = steps_by_id.get(raw_action.get("workflow_step_id")) or {}
        action = {
            "workflow_step_id": raw_action.get("workflow_step_id") or step.get("step_id"),
            "action_type": raw_action.get("action_type") or step.get("action_type"),
            "parameters": raw_action.get("parameters") or step.get("parameters_template") or {},
            "delay_ms": raw_action.get("delay_ms", step.get("delay_ms", 500)),
            "visual_index": raw_action.get("visual_index") or step.get("visual_index"),
            "expected_result": raw_action.get("expected_result") or step.get("expected_result"),
        }
        if action["action_type"]:
            enriched.append(action)
    return enriched


async def execute_indexed_replay_plan(
    *,
    browser: Any,
    memory: Any,
    plan: Dict[str, Any],
    workflow: Dict[str, Any],
    max_actions: int = 8,
    debug_logger: Any = None,
) -> Dict[str, Any]:
    actions = enrich_plan_actions(plan, workflow)[:max_actions]
    executed = []
    failures = []
    if debug_logger:
        debug_logger.replay(
            "execute.start",
            workflow_id=workflow.get("workflow_id"),
            planned_action_count=len(actions),
            checkpoint_after_actions=plan.get("checkpoint_after_actions"),
        )

    for action_index, action in enumerate(actions, start=1):
        step_num = len(memory.steps) + 1
        screenshot_path, screenshot_hash, before_state = await browser.capture_state(f"step_{step_num:03d}_indexed")
        if debug_logger:
            debug_logger.replay(
                "action.before",
                action_index=action_index,
                step=step_num,
                action=action,
                screenshot_path=screenshot_path,
                screenshot_hash=screenshot_hash,
                browser_state=before_state.to_dict(),
            )

        try:
            tool_call = ToolCall(
                action_type=ActionType(action["action_type"]),
                parameters=action.get("parameters") or {},
                verification_hint=None,
            )
        except Exception as exc:
            failures.append({"action": action, "error": f"invalid action type: {exc}"})
            if debug_logger:
                debug_logger.replay("action.invalid", action_index=action_index, action=action, error=str(exc))
            break

        start = time.time()
        if tool_call.action_type == ActionType.READ_HISTORY:
            action_result = ActionResult(
                success=False,
                action_type=ActionType.READ_HISTORY,
                description="ReadHistory is not allowed inside indexed replay",
                error="Replay cannot read live run history.",
                execution_time_ms=0,
            )
        else:
            action_result = await browser.execute_indexed_action(
                tool_call=tool_call,
                screenshot_path=screenshot_path,
                visual_index=action.get("visual_index"),
            )
        action_result.execution_time_ms = action_result.execution_time_ms or ((time.time() - start) * 1000)

        delay_ms = max(0, int(action.get("delay_ms") or 0))
        if delay_ms:
            await asyncio.sleep(delay_ms / 1000.0)

        after_screenshot_path, after_screenshot_hash, after_state = await browser.capture_state(f"step_{step_num:03d}_indexed_after")
        verification_status, change_score = await browser.verify_action(before_state, after_state, None)
        action_result.verification_status = verification_status
        action_result.state_change_score = change_score
        action_result.metadata.setdefault("cosmic_indexed_replay", {})
        action_result.metadata["cosmic_indexed_replay"].update(
            {
                "workflow_id": workflow.get("workflow_id"),
                "workflow_step_id": action.get("workflow_step_id"),
                "expected_result": action.get("expected_result"),
                "delay_ms": delay_ms,
            }
        )

        memory.add_step(
            screenshot_path=screenshot_path,
            screenshot_hash=screenshot_hash,
            browser_state=after_state,
            action=action_result,
            thinking=f"COSMIC indexed replay action {action_index}: {plan.get('reason', '')}",
            before_browser_state=before_state,
            after_browser_state=after_state,
            after_screenshot_path=after_screenshot_path,
            after_screenshot_hash=after_screenshot_hash,
            tool_call={
                "action_type": tool_call.action_type.value,
                "parameters": tool_call.parameters,
                "source": "cosmic_indexed_replay",
            },
            llm_response={
                "source": "cosmic_indexed_replay_plan",
                "workflow_id": workflow.get("workflow_id"),
                "workflow_step_id": action.get("workflow_step_id"),
            },
            visual_index=action.get("visual_index"),
        )

        executed.append(
            {
                "workflow_step_id": action.get("workflow_step_id"),
                "action_type": tool_call.action_type.value,
                "success": action_result.success,
                "verification_status": verification_status.value if verification_status else None,
            }
        )
        if debug_logger:
            debug_logger.replay(
                "action.after",
                action_index=action_index,
                step=step_num,
                action_result=action_result.to_dict(),
                after_screenshot_path=after_screenshot_path,
                after_screenshot_hash=after_screenshot_hash,
                after_browser_state=after_state.to_dict(),
                verification_status=verification_status.value if verification_status else None,
                change_score=change_score,
            )

        if not action_result.success or verification_status == VerificationStatus.ERROR:
            failures.append({"action": action, "error": action_result.error or "verification failed"})
            if debug_logger:
                debug_logger.replay(
                    "action.failed",
                    action_index=action_index,
                    action=action,
                    error=action_result.error or "verification failed",
                )
            break

        if action_index >= int(plan.get("checkpoint_after_actions") or len(actions)):
            break

    summary = {
        "executed": executed,
        "failures": failures,
        "completed_replay": not failures and bool(executed),
    }
    if debug_logger:
        debug_logger.replay("execute.complete", summary=summary)
    return summary
