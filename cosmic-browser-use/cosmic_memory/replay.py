"""Indexed replay execution."""

from __future__ import annotations

import asyncio
import copy
import re
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from cosmic_types import ActionResult, ActionType, ToolCall, VerificationStatus


_TEMPLATE_RE = re.compile(r"{{\s*([a-zA-Z0-9_]+)\s*}}")


def _workflow_variable_defaults(workflow: Dict[str, Any]) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    for variable in workflow.get("variables") or []:
        if isinstance(variable, dict) and variable.get("name"):
            values[str(variable["name"])] = variable.get("default", "")
    return values


def derive_goal_variable_values(goal: str, workflow: Dict[str, Any]) -> Dict[str, Any]:
    """Derive safe variable substitutions for template replay.

    This is intentionally conservative. It only fills common goal-sourced
    variables so a retrieved workflow can reuse stable mechanics (search box,
    submit, checkpoint) without replaying target-specific clicks.
    """
    goal_text = str(goal or "").strip()
    if not goal_text:
        return {}

    normalized_goal_text = re.sub(r"\boffl\b", "official", goal_text, flags=re.IGNORECASE)

    target = re.sub(r"^\s*(get|find|extract|read|show|tell\s+me)\s+", "", normalized_goal_text, flags=re.IGNORECASE)
    target = re.sub(r"\b(youtube\s+video\s+description|video\s+description|description)\b", " ", target, flags=re.IGNORECASE)
    target = re.sub(r"\b(for|from|of)\b", " ", target, flags=re.IGNORECASE)
    target = re.sub(r"\b(the|a|an)\b", " ", target, flags=re.IGNORECASE)
    target = re.sub(r"\byoutube\b", " ", target, flags=re.IGNORECASE)
    target = re.sub(r"[\"'“”‘’]", "", target)
    target = re.sub(r"\s+", " ", target).strip(" -:,.")
    if not target:
        target = goal_text

    values: Dict[str, Any] = {}
    variable_names = {
        str(variable.get("name"))
        for variable in workflow.get("variables") or []
        if isinstance(variable, dict) and variable.get("name")
    }

    if "search_query" in variable_names:
        values["search_query"] = target

    if "target_video_title_hint" in variable_names or "expected_title_contains" in variable_names:
        hint = target
        hint = re.sub(r"\b(offl|official)\b", " ", hint, flags=re.IGNORECASE)
        hint = re.sub(r"\b(launch|announcement|intro|introduction|video)\b", " ", hint, flags=re.IGNORECASE)
        hint = re.sub(r"\s+", " ", hint).strip(" -:,.")
        if "target_video_title_hint" in variable_names:
            values["target_video_title_hint"] = hint or target
        if "expected_title_contains" in variable_names:
            values["expected_title_contains"] = hint or target

    if "expected_channel" in variable_names:
        lowered = normalized_goal_text.lower()
        if "openai" in lowered or "gpt" in lowered:
            values["expected_channel"] = "OpenAI"
        elif "anthropic" in lowered or "claude" in lowered:
            values["expected_channel"] = "Anthropic"
        else:
            # Important: overwrite source-workflow defaults such as "OpenAI".
            # Unknown-channel tasks should use neutral live grounding prompts.
            values["expected_channel"] = ""

    values["goal"] = normalized_goal_text
    return values


def _clean_substituted_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    text = re.sub(r"\s+by\s*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+from\s*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+by\s+([,.;:])", r"\1", text, flags=re.IGNORECASE)
    return text.strip()


def _substitute_templates(value: Any, variables: Dict[str, Any]) -> Any:
    if isinstance(value, str):
        return _clean_substituted_text(
            _TEMPLATE_RE.sub(lambda match: str(variables.get(match.group(1), match.group(0))), value)
        )
    if isinstance(value, list):
        return [_substitute_templates(item, variables) for item in value]
    if isinstance(value, dict):
        return {key: _substitute_templates(item, variables) for key, item in value.items()}
    return value


def _has_unresolved_template(value: Any) -> bool:
    if isinstance(value, str):
        return bool(_TEMPLATE_RE.search(value))
    if isinstance(value, list):
        return any(_has_unresolved_template(item) for item in value)
    if isinstance(value, dict):
        return any(_has_unresolved_template(item) for item in value.values())
    return False


def _neutralize_dynamic_target_params(
    *,
    step: Dict[str, Any],
    action_type: Optional[str],
    params: Dict[str, Any],
    variables: Dict[str, Any],
) -> Dict[str, Any]:
    """Avoid carrying source-specific target hints into a new task.

    The indexed workflow may have been learned from a branded task such as an
    OpenAI video. When the new goal does not provide a channel/entity, a
    source default like "OpenAI" must not be used to ground a different target.
    """
    if not isinstance(params, dict):
        return params
    if step.get("role") != "open_target" or action_type != ActionType.VISUAL_CLICK.value:
        return params
    if variables.get("expected_channel"):
        return params

    target = (
        variables.get("expected_title_contains")
        or variables.get("target_video_title_hint")
        or variables.get("search_query")
        or "the best matching target"
    )
    params = dict(params)
    params["description"] = (
        f'best matching regular YouTube video result thumbnail or title for "{str(target).strip()}" '
        "(not Shorts, filters, search box, or navigation)"
    )
    return params


def _tokens(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", (value or "").lower()) if len(token) > 2}


def _goal_matches_observed_target(goal: str, workflow: Dict[str, Any]) -> bool:
    observed = workflow.get("observed_success") or {}
    final_url = observed.get("final_url") or ""
    if not final_url:
        return False
    try:
        final_host = urlparse(final_url).netloc.lower().removeprefix("www.")
    except Exception:
        final_host = ""
    if workflow.get("domain") and final_host and workflow["domain"] not in final_host:
        return False

    goal_tokens = _tokens(goal)
    variable_defaults = {
        str(var.get("name")): str(var.get("default") or "")
        for var in workflow.get("variables") or []
        if isinstance(var, dict)
    }
    query_tokens = _tokens(variable_defaults.get("search_query", ""))
    significant_query_tokens = query_tokens - {"youtube", "video", "description", "official"}
    if significant_query_tokens and len(goal_tokens.intersection(significant_query_tokens)) / max(1, len(significant_query_tokens)) >= 0.75:
        return True

    title_tokens = _tokens(variable_defaults.get("target_video_title_hint", ""))
    significant_title_tokens = title_tokens - {"youtube", "video", "official"}
    return bool(significant_title_tokens) and len(goal_tokens.intersection(significant_title_tokens)) >= max(1, min(2, len(significant_title_tokens)))


def _observed_final_url_fast_path(goal: str, workflow: Dict[str, Any], max_actions: int) -> Optional[Dict[str, Any]]:
    if not _goal_matches_observed_target(goal, workflow):
        return None
    observed = workflow.get("observed_success") or {}
    final_url = observed.get("final_url")
    if not final_url:
        return None

    actions: List[Dict[str, Any]] = [
        {
            "workflow_step_id": "__observed_final_url__",
            "action_type": ActionType.NAVIGATE.value,
            "parameters": {"url": final_url},
            "delay_ms": 1800,
            "expected_result": f"Navigate directly to observed successful target: {observed.get('final_title') or final_url}",
            "execution_mode": "browser_action",
        }
    ]
    variables = _workflow_variable_defaults(workflow)
    variables.update(derive_goal_variable_values(goal, workflow))
    checkpoint_reason = "Navigated to observed successful final URL."

    for step in workflow.get("steps") or []:
        if len(actions) >= max_actions:
            checkpoint_reason = "Reached replay action cap after observed final URL fast path."
            break
        if step.get("role") not in {"expand_content", "extract_answer"}:
            continue
        action_type = step.get("action_type")
        params = _substitute_templates(copy.deepcopy(step.get("parameters_template") or {}), variables)
        params = _neutralize_dynamic_target_params(
            step=step,
            action_type=action_type,
            params=params,
            variables=variables,
        )
        if action_type in {ActionType.SAVE_NOTE.value, "SaveNote"} or _has_unresolved_template(params):
            checkpoint_reason = "Stopped before answer extraction; live visible-answer finalizer should read the current page."
            break
        needs_grounding = bool(step.get("needs_grounding_on_replay"))
        replay_safe = bool((step.get("visual_index_replay_policy") or {}).get("replay_safe"))
        actions.append(
            {
                "workflow_step_id": step.get("step_id"),
                "action_type": action_type,
                "parameters": params,
                "delay_ms": step.get("delay_ms", 500),
                "expected_result": step.get("expected_result"),
                "execution_mode": "live_grounding"
                if needs_grounding
                else ("indexed_coordinates" if replay_safe else "browser_action"),
            }
        )
        if step.get("checkpoint"):
            checkpoint_reason = f"Stopped at observed-target workflow checkpoint after {step.get('step_id')}."
            break

    return {
        "use_workflow": True,
        "confidence": 0.68,
        "reason": "Deterministic observed-final-URL replay for a strongly matching target.",
        "checkpoint_after_actions": len(actions),
        "checkpoint_reason": checkpoint_reason,
        "actions": actions,
    }


def build_default_replay_plan(goal: str, workflow: Dict[str, Any], max_actions: int = 8) -> Dict[str, Any]:
    """Build a deterministic replay chain.

    This is a safety net for planner failures and a useful smoke-test target:
    execute known indexed/browser actions and live-grounded dynamic targets
    until answer extraction, a failure, or an unresolved template.
    """
    fast_path = _observed_final_url_fast_path(goal, workflow, max_actions)
    if fast_path:
        return fast_path

    variables = _workflow_variable_defaults(workflow)
    variables.update(derive_goal_variable_values(goal, workflow))
    actions: List[Dict[str, Any]] = []
    checkpoint_reason = "No replayable workflow steps."
    last_checkpoint_reason = ""

    for step in workflow.get("steps") or []:
        if len(actions) >= max_actions:
            checkpoint_reason = "Reached replay action cap."
            break

        action_type = step.get("action_type")
        params = _substitute_templates(copy.deepcopy(step.get("parameters_template") or {}), variables)
        params = _neutralize_dynamic_target_params(
            step=step,
            action_type=action_type,
            params=params,
            variables=variables,
        )
        if action_type in {ActionType.SAVE_NOTE.value, "SaveNote"} and _has_unresolved_template(params):
            checkpoint_reason = "Stopped before templated SaveNote; live finalizer must read the visible answer."
            break
        if _has_unresolved_template(params):
            checkpoint_reason = f"Stopped before unresolved template parameters in {step.get('step_id')}."
            break

        needs_grounding = bool(step.get("needs_grounding_on_replay"))
        replay_safe = bool((step.get("visual_index_replay_policy") or {}).get("replay_safe"))

        actions.append(
            {
                "workflow_step_id": step.get("step_id"),
                "action_type": action_type,
                "parameters": params,
                "delay_ms": step.get("delay_ms", 500),
                "expected_result": step.get("expected_result"),
                "execution_mode": "live_grounding"
                if needs_grounding
                else ("indexed_coordinates" if replay_safe else "browser_action"),
            }
        )

        if step.get("checkpoint"):
            last_checkpoint_reason = f"Crossed workflow checkpoint after {step.get('step_id')}."
            checkpoint_reason = last_checkpoint_reason

    if actions and checkpoint_reason == "No replayable workflow steps.":
        checkpoint_reason = last_checkpoint_reason or f"Executed {len(actions)} deterministic replay actions."

    return {
        "use_workflow": bool(actions),
        "confidence": 0.55 if actions else 0.0,
        "reason": "Deterministic replay chain built from workflow policies.",
        "variable_values": derive_goal_variable_values(goal, workflow),
        "checkpoint_after_actions": len(actions),
        "checkpoint_reason": checkpoint_reason,
        "actions": actions,
    }


def enrich_plan_actions(plan: Dict[str, Any], workflow: Dict[str, Any]) -> List[Dict[str, Any]]:
    steps_by_id = {step.get("step_id"): step for step in workflow.get("steps", [])}
    variables = _workflow_variable_defaults(workflow)
    if plan.get("goal"):
        variables.update(derive_goal_variable_values(str(plan.get("goal")), workflow))
    variables.update(plan.get("variable_values") or {})
    enriched = []
    for raw_action in plan.get("actions") or []:
        step = steps_by_id.get(raw_action.get("workflow_step_id")) or {}
        raw_params = raw_action.get("parameters") or step.get("parameters_template") or {}
        parameters = _substitute_templates(copy.deepcopy(raw_params), variables)
        parameters = _neutralize_dynamic_target_params(
            step=step,
            action_type=raw_action.get("action_type") or step.get("action_type"),
            params=parameters,
            variables=variables,
        )
        execution_mode = raw_action.get("execution_mode")
        visual_index = None if execution_mode == "live_grounding" else (raw_action.get("visual_index") or step.get("visual_index"))
        action = {
            "workflow_step_id": raw_action.get("workflow_step_id") or step.get("step_id"),
            "action_type": raw_action.get("action_type") or step.get("action_type"),
            "parameters": parameters,
            "delay_ms": raw_action.get("delay_ms", step.get("delay_ms", 500)),
            "visual_index": visual_index,
            "expected_result": raw_action.get("expected_result") or step.get("expected_result"),
            "target_description": raw_action.get("target_description") or step.get("target_description"),
            "role": step.get("role"),
            "checkpoint": raw_action.get("checkpoint", step.get("checkpoint")),
            "execution_mode": execution_mode,
            "needs_grounding_on_replay": step.get("needs_grounding_on_replay"),
            "visual_index_replay_policy": step.get("visual_index_replay_policy"),
            "has_unresolved_template": _has_unresolved_template(parameters),
        }
        if action["action_type"]:
            enriched.append(action)
    return enriched


def _timeline_label_for_action(action: Dict[str, Any]) -> str:
    action_type = str(action.get("action_type") or "Action")
    params = action.get("parameters") or {}
    if action_type == ActionType.NAVIGATE.value:
        return f"Navigate: {params.get('url', '')}"
    if action_type == ActionType.VISUAL_CLICK.value:
        return f"Indexed click: {params.get('description', '')}"
    if action_type == ActionType.VISUAL_TYPE.value:
        return f"Indexed type: {params.get('text', params.get('field_description', ''))}"
    if action_type == ActionType.VISUAL_SCROLL.value:
        return f"Scroll: {params.get('direction', '')}"
    if action_type == ActionType.PRESS_KEY.value:
        return f"Press: {params.get('key', '')}"
    if action_type == ActionType.SAVE_NOTE.value:
        return "Answer saved"
    return action_type


def _timeline_kind_for_action(action: Dict[str, Any]) -> str:
    action_type = str(action.get("action_type") or "")
    if action_type == ActionType.NAVIGATE.value:
        return "navigate"
    if action_type == ActionType.SAVE_NOTE.value:
        return "saved"
    if action.get("execution_mode") == "live_grounding":
        return "live"
    return "indexed"


def _state_url(state: Any) -> str:
    if isinstance(state, dict):
        return str(state.get("url") or "")
    return str(getattr(state, "url", "") or "")


def _state_title(state: Any) -> str:
    if isinstance(state, dict):
        return str(state.get("title") or "")
    return str(getattr(state, "title", "") or "")


def _same_url(left: str, right: str) -> bool:
    return (left or "").rstrip("/") == (right or "").rstrip("/")


def _is_youtube_url(url: str) -> bool:
    try:
        return "youtube.com" in urlparse(url or "").netloc.lower()
    except Exception:
        return False


def _is_youtube_search_url(url: str) -> bool:
    lowered = (url or "").lower()
    return "/results?" in lowered or "search_query=" in lowered


def _is_youtube_watch_url(url: str) -> bool:
    lowered = (url or "").lower()
    return "/watch?" in lowered or "/watch/" in lowered


def _replay_semantic_error(
    *,
    action: Dict[str, Any],
    tool_call: ToolCall,
    before_state: Any,
    after_state: Any,
) -> Optional[str]:
    """Validate replay checkpoints with page semantics, not just pixel diffs."""
    before_url = _state_url(before_state)
    after_url = _state_url(after_state)
    after_title = _state_title(after_state)
    expected = str(action.get("expected_result") or "").lower()
    role = str(action.get("role") or "").lower()
    params = tool_call.parameters or {}

    youtube_context = _is_youtube_url(before_url) or _is_youtube_url(after_url) or "youtube" in expected

    if (
        youtube_context
        and tool_call.action_type == ActionType.VISUAL_TYPE
        and params.get("press_enter")
        and "search result" in expected
        and not _is_youtube_search_url(after_url)
    ):
        return (
            "Replay expected YouTube search results after typing, but the page "
            f"remained at {after_url or after_title or 'an unknown page'}."
        )

    target_video_expected = role == "open_target" or "target video page" in expected or "video page loads" in expected
    if youtube_context and target_video_expected:
        if not _is_youtube_watch_url(after_url):
            return (
                "Replay expected a YouTube watch page after target selection, "
                f"but landed at {after_url or after_title or 'an unknown page'}."
            )
        if _same_url(before_url, after_url):
            return "Replay target selection did not change the YouTube URL."

    answer_page_step = role in {"expand_content", "extract_answer"} or "description" in expected
    if youtube_context and answer_page_step and not _is_youtube_watch_url(after_url):
        return (
            "Replay attempted an answer-page action before reaching a YouTube "
            f"watch page; current URL is {after_url or after_title or 'unknown'}."
        )

    return None


async def execute_indexed_replay_plan(
    *,
    browser: Any,
    memory: Any,
    plan: Dict[str, Any],
    workflow: Dict[str, Any],
    max_actions: int = 8,
    debug_logger: Any = None,
    demo_overlay: Any = None,
) -> Dict[str, Any]:
    actions = enrich_plan_actions(plan, workflow)[:max_actions]
    executed = []
    failures = []
    checkpoint_reached = False
    checkpoint_reason = plan.get("checkpoint_reason") or ""
    goal_completed = False
    if debug_logger:
        debug_logger.replay(
            "execute.start",
            workflow_id=workflow.get("workflow_id"),
            planned_action_count=len(actions),
            checkpoint_after_actions=plan.get("checkpoint_after_actions"),
        )
    if demo_overlay is not None:
        try:
            # First real agent activity — start the elapsed-time clock.
            if hasattr(demo_overlay, "start_timer"):
                demo_overlay.start_timer()
            await demo_overlay.update(
                page=getattr(browser, "page", None),
                mode="Replay Mode",
                phase="Executing known route",
                pulse_ms=1200,
            )
        except Exception:
            pass

    for action_index, action in enumerate(actions, start=1):
        if action.get("has_unresolved_template"):
            checkpoint_reached = True
            checkpoint_reason = (
                f"Stopped before unresolved template in {action.get('workflow_step_id')}; "
                "live planner/finalizer should continue."
            )
            if debug_logger:
                debug_logger.replay(
                    "action.skipped_unresolved_template",
                    action_index=action_index,
                    action=action,
                    checkpoint_reason=checkpoint_reason,
                )
            break

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
        semantic_error = None
        if action_result.success:
            semantic_error = _replay_semantic_error(
                action=action,
                tool_call=tool_call,
                before_state=before_state,
                after_state=after_state,
            )
            if semantic_error:
                action_result.success = False
                action_result.error = semantic_error
                verification_status = VerificationStatus.ERROR
                change_score = 0.0
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
                "execution_mode": action.get("execution_mode"),
            }
        )
        if demo_overlay is not None and action_result.success:
            try:
                await demo_overlay.update(
                    page=getattr(browser, "page", None),
                    timeline_append={
                        "kind": _timeline_kind_for_action(action),
                        "label": _timeline_label_for_action(action),
                    },
                    metrics={
                        "replay_actions": len(executed),
                        # Each replayed action with stored coordinates would
                        # otherwise have cost one MiMo grounding call.
                        "mimo_calls_avoided": sum(
                            1 for e in executed if e.get("execution_mode") == "indexed_coordinates"
                        ),
                    },
                    pulse_ms=900,
                )
            except Exception:
                pass
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
                semantic_error=semantic_error,
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

        if tool_call.action_type == ActionType.SAVE_NOTE:
            goal_completed = True
            checkpoint_reached = True
            checkpoint_reason = "Terminal SaveNote executed during indexed replay."
            break

        if action_index >= int(plan.get("checkpoint_after_actions") or len(actions)):
            checkpoint_reached = True
            break

    summary = {
        "executed": executed,
        "failures": failures,
        "completed_replay": not failures and bool(executed),
        "checkpoint_reached": checkpoint_reached,
        "checkpoint_reason": checkpoint_reason,
        "goal_completed": goal_completed,
    }
    if debug_logger:
        debug_logger.replay("execute.complete", summary=summary)
    if demo_overlay is not None:
        try:
            if goal_completed:
                phase = "Saved answer"
            elif failures:
                phase = "Live planner fallback"
            elif checkpoint_reached:
                phase = "Checkpoint"
            else:
                phase = "Matched path"
            await demo_overlay.update(
                page=getattr(browser, "page", None),
                phase=phase,
                pulse_ms=1500,
            )
        except Exception:
            pass
    return summary
