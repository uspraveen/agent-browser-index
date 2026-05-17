"""Compile browser run traces into reusable traversal workflows."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse

from .coordinates import VISUAL_ACTIONS
from .indexer import WorkflowRunIndexer
from .workflow_store import infer_domain, intent_tokens, now_iso, slugify, task_signature


def _state_id(domain: Optional[str], index: int, title: str) -> str:
    prefix = slugify(domain or "site")
    title_slug = slugify(title or "page", default="page")
    return f"{prefix}_state_{index:03d}_{title_slug[:28]}"


def _domain_from_url(url: str) -> Optional[str]:
    try:
        host = urlparse(url or "").netloc.lower().removeprefix("www.")
        return host or None
    except Exception:
        return None


def _visual_markers(*texts: str) -> List[str]:
    tokens = []
    for text in texts:
        tokens.extend(intent_tokens(text or ""))
    seen = []
    for token in tokens:
        if token not in seen:
            seen.append(token)
    return seen[:12]


class TraceCompiler:
    def __init__(self, store: Any, supermemory_client: Any = None):
        self.store = store
        self.supermemory = supermemory_client
        self.indexer = WorkflowRunIndexer.from_env()
        self.last_indexer_plan: Optional[Dict[str, Any]] = None

    def compile_from_steps(
        self,
        task: str,
        steps: Iterable[Any],
        run_dir: str,
        status: str = "success",
        workflow_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        step_dicts = [step.to_dict() if hasattr(step, "to_dict") else dict(step) for step in steps]
        action_steps = [step for step in step_dicts if step.get("action")]
        if not action_steps:
            return None

        indexer_plan = self.indexer.distill(task=task, steps=step_dicts, run_dir=run_dir, status=status)
        self.last_indexer_plan = indexer_plan
        if (indexer_plan.get("quality") or {}).get("usable") and indexer_plan.get("selected_steps"):
            workflow = self._compile_from_indexer_plan(
                task=task,
                step_dicts=step_dicts,
                run_dir=run_dir,
                status=status,
                indexer_plan=indexer_plan,
                workflow_id=workflow_id,
            )
            if workflow:
                return workflow

        first_before = action_steps[0].get("before_browser_state") or action_steps[0].get("browser_state") or {}
        first_url = first_before.get("url") or ""
        domain = infer_domain(task, first_url) or _domain_from_url(first_url)
        signature = task_signature(task, domain)
        workflow_id = workflow_id or f"{signature}_{slugify(status)}"
        existing_success_count = 0
        try:
            existing_success_count = int(self.store.load_workflow(workflow_id).get("success_count") or 0)
        except Exception:
            existing_success_count = 0

        page_states: List[Dict[str, Any]] = []
        compiled_steps: List[Dict[str, Any]] = []

        for idx, step in enumerate(action_steps, start=1):
            action = step.get("action") or {}
            before = step.get("before_browser_state") or step.get("browser_state") or {}
            after = step.get("after_browser_state") or step.get("browser_state") or {}
            tool_call = step.get("tool_call") or {}
            visual_index = step.get("visual_index") or action.get("metadata", {}).get("visual_index")
            from_state_id = _state_id(domain, idx, before.get("title", ""))
            to_state_id = _state_id(domain, idx + 1, after.get("title", ""))

            page_states.append(
                {
                    "state_id": from_state_id,
                    "name": before.get("title") or f"State {idx}",
                    "url_pattern": before.get("url"),
                    "page_summary": self._page_summary(before),
                    "visual_markers": _visual_markers(
                        before.get("title", ""),
                        action.get("description", ""),
                        (visual_index or {}).get("target_description", ""),
                    ),
                    "screenshot_path": step.get("screenshot_path"),
                    "screenshot_hash": step.get("screenshot_hash"),
                }
            )

            params = tool_call.get("parameters") or {}
            action_type = action.get("action_type")
            compiled_steps.append(
                {
                    "step_id": f"step_{idx:03d}",
                    "from_state_id": from_state_id,
                    "action_type": action_type,
                    "target_description": self._target_description(action, tool_call, visual_index),
                    "parameters_template": params,
                    "visual_index": visual_index,
                    "coordinates": action.get("coordinates"),
                    "expected_result": self._expected_result(after),
                    "to_state_id": to_state_id,
                    "delay_ms": self._default_delay_ms(action_type),
                    "checkpoint": action_type in {"Navigate", "VisualClick", "DOMClick"} or idx % 4 == 0,
                    "success": bool(action.get("success")),
                    "verification_status": action.get("verification_status"),
                    "error": action.get("error"),
                }
            )

        final = action_steps[-1].get("after_browser_state") or action_steps[-1].get("browser_state") or {}
        page_states.append(
            {
                "state_id": _state_id(domain, len(action_steps) + 1, final.get("title", "")),
                "name": final.get("title") or "Final state",
                "url_pattern": final.get("url"),
                "page_summary": self._page_summary(final),
                "visual_markers": _visual_markers(final.get("title", "")),
                "screenshot_path": action_steps[-1].get("after_screenshot_path"),
                "screenshot_hash": action_steps[-1].get("after_screenshot_hash"),
            }
        )

        failure_patches = self._failure_patches(compiled_steps, domain)
        success_count = existing_success_count + (1 if status == "success" else 0)
        workflow = {
            "workflow_id": workflow_id,
            "name": self._workflow_name(task, domain),
            "domain": domain,
            "intent": task,
            "task_signature": signature,
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "success_count": success_count,
            "avg_runtime_seconds": None,
            "summary": self._summary(task, domain, compiled_steps),
            "source_run_dir": run_dir,
            "page_states": page_states,
            "steps": compiled_steps,
            "failure_patches": failure_patches,
            "user_preferences": [],
        }

        saved_id = self.store.save_workflow(workflow)
        workflow["workflow_id"] = saved_id
        if self.supermemory:
            self.supermemory.remember_workflow(workflow)
        return workflow

    def _compile_from_indexer_plan(
        self,
        *,
        task: str,
        step_dicts: List[Dict[str, Any]],
        run_dir: str,
        status: str,
        indexer_plan: Dict[str, Any],
        workflow_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        steps_by_source = {step.get("step"): step for step in step_dicts}
        action_steps = [step for step in step_dicts if step.get("action")]
        if not action_steps:
            return None

        workflow_meta = indexer_plan.get("workflow") or {}
        quality = indexer_plan.get("quality") or {}
        first_before = action_steps[0].get("before_browser_state") or action_steps[0].get("browser_state") or {}
        final_state = self._final_state_from_steps(step_dicts)
        final_url = final_state.get("url") or ""
        domain = workflow_meta.get("domain") or infer_domain(task, final_url or first_before.get("url")) or _domain_from_url(final_url or first_before.get("url", ""))
        signature = task_signature(task, domain)
        workflow_id = workflow_id or f"{signature}_{slugify(status)}"
        existing_success_count = 0
        try:
            existing_success_count = int(self.store.load_workflow(workflow_id).get("success_count") or 0)
        except Exception:
            existing_success_count = 0

        page_states: List[Dict[str, Any]] = []
        compiled_steps: List[Dict[str, Any]] = []
        previous_after = first_before

        for idx, directive in enumerate(indexer_plan.get("selected_steps") or [], start=1):
            source_step = directive.get("source_step")
            source = steps_by_source.get(source_step) if source_step is not None else None
            action = (source or {}).get("action") or {}
            source_tool_call = (source or {}).get("tool_call") or {}
            before = (source or {}).get("before_browser_state") or (source or {}).get("browser_state") or previous_after or first_before
            after = (source or {}).get("after_browser_state") or (source or {}).get("browser_state") or before

            action_type = directive.get("action_type") or action.get("action_type")
            params = directive.get("parameters_template") or source_tool_call.get("parameters") or {}
            if source is None and action_type == "Navigate" and params.get("url"):
                after = dict(after or {})
                after["url"] = params["url"]
                if directive.get("expected_result"):
                    after["title"] = directive["expected_result"]

            visual_index = None
            coordinates = None
            if source is not None:
                visual_index = (source.get("visual_index") or action.get("metadata", {}).get("visual_index"))
                coordinates = action.get("coordinates")
            source_visual_index = visual_index
            replay_safe = bool(visual_index)
            replay_policy_reason = "safe_source_visual_index" if visual_index else "no_source_visual_index"
            if action_type not in VISUAL_ACTIONS:
                visual_index = None
                replay_safe = False
                replay_policy_reason = "not_a_visual_action"
            if (
                action_type in VISUAL_ACTIONS
                and directive.get("mutation") == "parameterized"
                and directive.get("role") not in {"search_query", "navigation"}
            ):
                # A parameterized target may not live at the same screen position
                # as the source run. Keep the semantic target but require live
                # grounding instead of replaying stale coordinates.
                visual_index = None
                coordinates = None
                replay_safe = False
                replay_policy_reason = "parameterized_dynamic_target_requires_live_grounding"
            route_gap = (
                action_type in VISUAL_ACTIONS
                and source is not None
                and previous_after
                and before.get("url")
                and previous_after.get("url")
                and before.get("url") != previous_after.get("url")
            )
            if route_gap:
                # The selected source step happened from a different page than
                # the selected replay route reaches. Its coordinates are not
                # replay-safe in this workflow.
                visual_index = None
                coordinates = None
                replay_safe = False
                replay_policy_reason = "source_page_route_gap_requires_live_grounding"

            from_state_id = _state_id(domain, idx, before.get("title", ""))
            to_state_id = _state_id(domain, idx + 1, after.get("title", ""))
            page_states.append(
                {
                    "state_id": from_state_id,
                    "name": before.get("title") or f"State {idx}",
                    "url_pattern": before.get("url"),
                    "page_summary": self._page_summary(before),
                    "visual_markers": _visual_markers(
                        before.get("title", ""),
                        directive.get("target_description", ""),
                        (source_visual_index or {}).get("target_description", ""),
                    ),
                    "screenshot_path": (source or {}).get("screenshot_path"),
                    "screenshot_hash": (source or {}).get("screenshot_hash"),
                }
            )

            compiled_steps.append(
                {
                    "step_id": f"step_{idx:03d}",
                    "source_step": source_step,
                    "source_evidence_steps": directive.get("source_evidence_steps") or ([source_step] if source_step is not None else []),
                    "role": directive.get("role"),
                    "from_state_id": from_state_id,
                    "action_type": action_type,
                    "target_description": directive.get("target_description") or self._target_description(action, source_tool_call, visual_index),
                    "parameters_template": params,
                    "parameter_slots": directive.get("parameter_slots") or {},
                    "visual_index": visual_index,
                    "source_visual_index": source_visual_index,
                    "visual_index_replay_policy": {
                        "stored": bool(source_visual_index),
                        "replay_safe": bool(replay_safe and visual_index),
                        "reason": replay_policy_reason,
                    },
                    "coordinates": coordinates,
                    "expected_result": directive.get("expected_result") or self._expected_result(after),
                    "to_state_id": to_state_id,
                    "delay_ms": int(directive.get("delay_ms") or self._default_delay_ms(action_type)),
                    "checkpoint": bool(directive.get("checkpoint", action_type in {"Navigate", "VisualClick", "DOMClick"}) or route_gap),
                    "success": True if source is None else bool(action.get("success")),
                    "verification_status": None if source is None else action.get("verification_status"),
                    "error": None if source is None else action.get("error"),
                    "mutation": "route_gap_requires_live_grounding" if route_gap else directive.get("mutation", "as_recorded"),
                    "needs_grounding_on_replay": action_type in VISUAL_ACTIONS and not bool(visual_index),
                }
            )
            previous_after = after

        page_states.append(
            {
                "state_id": _state_id(domain, len(compiled_steps) + 1, previous_after.get("title", "")),
                "name": previous_after.get("title") or "Final state",
                "url_pattern": previous_after.get("url"),
                "page_summary": self._page_summary(previous_after),
                "visual_markers": _visual_markers(previous_after.get("title", "")),
                "screenshot_path": compiled_steps[-1].get("after_screenshot_path") if compiled_steps else None,
                "screenshot_hash": None,
            }
        )

        success_count = existing_success_count + (1 if status == "success" else 0)
        workflow = {
            "workflow_id": workflow_id,
            "name": workflow_meta.get("name") or self._workflow_name(task, domain),
            "domain": domain,
            "intent": workflow_meta.get("intent") or task,
            "task_signature": signature,
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "success_count": success_count,
            "avg_runtime_seconds": None,
            "summary": workflow_meta.get("summary") or self._summary(task, domain, compiled_steps),
            "route_strategy": workflow_meta.get("route_strategy"),
            "generalization_level": quality.get("generalization_level"),
            "quality": quality,
            "variables": workflow_meta.get("variables") or [],
            "acceptance_criteria": workflow_meta.get("acceptance_criteria") or {},
            "source_run_dir": run_dir,
            "source_status": status,
            "indexer": indexer_plan.get("indexer") or {},
            "discarded_steps": indexer_plan.get("discarded_steps") or [],
            "trace_evidence": self._trace_evidence(step_dicts, indexer_plan),
            "page_states": page_states,
            "steps": compiled_steps,
            "failure_patches": self._plan_failure_patches(indexer_plan, compiled_steps, domain),
            "user_preferences": [],
        }

        saved_id = self.store.save_workflow(workflow)
        workflow["workflow_id"] = saved_id
        if self.supermemory:
            self.supermemory.remember_workflow(workflow)
        return workflow

    def _trace_evidence(self, step_dicts: List[Dict[str, Any]], indexer_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        selected = {
            step.get("source_step")
            for step in indexer_plan.get("selected_steps") or []
            if step.get("source_step") is not None
        }
        discarded = {
            row.get("source_step"): row.get("reason")
            for row in indexer_plan.get("discarded_steps") or []
            if row.get("source_step") is not None
        }
        rows: List[Dict[str, Any]] = []
        for step in step_dicts:
            action = step.get("action") or {}
            if not action:
                continue
            source_step = step.get("step")
            before = step.get("before_browser_state") or step.get("browser_state") or {}
            after = step.get("after_browser_state") or step.get("browser_state") or {}
            tool_call = step.get("tool_call") or {}
            visual_index = step.get("visual_index") or action.get("metadata", {}).get("visual_index") or {}
            rows.append(
                {
                    "source_step": source_step,
                    "selected": source_step in selected,
                    "discarded_reason": discarded.get(source_step),
                    "action_type": action.get("action_type"),
                    "success": action.get("success"),
                    "verification_status": action.get("verification_status"),
                    "error": action.get("error"),
                    "description": action.get("description"),
                    "parameters": tool_call.get("parameters") or {},
                    "coordinates": action.get("coordinates"),
                    "visual_index": {
                        "target_description": visual_index.get("target_description"),
                        "normalized_coordinates": visual_index.get("normalized_coordinates"),
                        "pixel_coordinates": visual_index.get("pixel_coordinates"),
                        "viewport": visual_index.get("viewport"),
                        "page_url": visual_index.get("page_url"),
                        "page_title": visual_index.get("page_title"),
                    }
                    if visual_index
                    else None,
                    "before": {
                        "url": before.get("url"),
                        "title": before.get("title"),
                        "scroll_y": before.get("scroll_y"),
                    },
                    "after": {
                        "url": after.get("url"),
                        "title": after.get("title"),
                        "scroll_y": after.get("scroll_y"),
                    },
                }
            )
        return rows

    def _final_state_from_steps(self, step_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        for step in reversed(step_dicts):
            action = step.get("action") or {}
            if action.get("action_type") == "SaveNote" and action.get("success"):
                return step.get("before_browser_state") or step.get("after_browser_state") or step.get("browser_state") or {}
        last = step_dicts[-1] if step_dicts else {}
        return last.get("after_browser_state") or last.get("browser_state") or {}

    def _plan_failure_patches(self, indexer_plan: Dict[str, Any], steps: List[Dict[str, Any]], domain: Optional[str]) -> List[Dict[str, Any]]:
        patches = []
        for idx, patch in enumerate(indexer_plan.get("failure_patches") or [], start=1):
            patches.append(
                {
                    "patch_id": f"patch_indexer_{idx:03d}",
                    "domain": domain,
                    "applies_to_source_step": patch.get("applies_to_source_step"),
                    "failure": patch.get("failure"),
                    "fix": patch.get("fix"),
                    "confidence": patch.get("confidence", 0.5),
                }
            )
        patches.extend(self._failure_patches(steps, domain))
        return patches

    def _target_description(self, action: Dict[str, Any], tool_call: Dict[str, Any], visual_index: Optional[Dict[str, Any]]) -> str:
        if visual_index and visual_index.get("target_description"):
            return str(visual_index["target_description"])
        params = tool_call.get("parameters") or {}
        for key in ("description", "field_description", "selector", "url", "query"):
            if params.get(key):
                return str(params[key])
        return str(action.get("description") or action.get("action_type") or "action")

    def _workflow_name(self, task: str, domain: Optional[str]) -> str:
        label = re.sub(r"\s+", " ", task.strip())[:80]
        return f"{domain or 'site'}: {label}"

    def _summary(self, task: str, domain: Optional[str], steps: List[Dict[str, Any]]) -> str:
        replayable = sum(1 for step in steps if step.get("action_type") in VISUAL_ACTIONS or step.get("action_type") in {"Navigate", "PressKey", "DOMClick", "DOMExtract"})
        return f"Workflow for {task!r} on {domain or 'unknown domain'} with {len(steps)} steps, {replayable} replayable actions, and visual indexes for MiMo-grounded targets."

    def _page_summary(self, state: Dict[str, Any]) -> str:
        return f"{state.get('title', '')} at {state.get('url', '')}".strip()

    def _expected_result(self, state: Dict[str, Any]) -> str:
        title = state.get("title") or "next page state"
        url = state.get("url") or ""
        return f"Browser reaches {title} {url}".strip()

    def _default_delay_ms(self, action_type: Optional[str]) -> int:
        if action_type == "Navigate":
            return 1500
        if action_type in {"VisualClick", "DOMClick", "PressKey"}:
            return 900
        if action_type == "VisualType":
            return 500
        return 300

    def _failure_patches(self, steps: List[Dict[str, Any]], domain: Optional[str]) -> List[Dict[str, Any]]:
        patches = []
        for step in steps:
            if step.get("success") is False or step.get("error"):
                patches.append(
                    {
                        "patch_id": f"patch_{step.get('step_id')}",
                        "domain": domain,
                        "applies_to_state_id": step.get("from_state_id"),
                        "failure": step.get("error") or f"{step.get('action_type')} did not succeed",
                        "fix": "Resume with current screenshot and let the browser agent repair the action before continuing indexed replay.",
                        "confidence": 0.5,
                    }
                )
        return patches
