"""Soft-fail Supermemory integration."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional


def _safe_tag(value: str, default: str = "cosmic") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_:-]+", "_", value or "").strip("_:-")
    return (cleaned or default)[:100]


def _metadata(value: Dict[str, Any]) -> Dict[str, Any]:
    clean = {}
    for key, item in value.items():
        if item is None:
            continue
        if isinstance(item, (str, int, float, bool)):
            clean[key] = item
        elif isinstance(item, list):
            clean[key] = ",".join(str(part) for part in item[:20])
        else:
            clean[key] = str(item)
    return clean


class SupermemoryMemoryClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        container_tag: str = "cosmic-hackathon-demo",
        user_id: str = "demo_user",
        enabled: bool = True,
        task_type: str = "memory",
    ):
        self.api_key = api_key or os.getenv("SUPERMEMORY_API_KEY")
        self.container_tag = container_tag
        self.user_id = user_id
        self.space_tag = _safe_tag(f"{container_tag}:{user_id}")
        self.task_type = task_type
        self.enabled = enabled and bool(self.api_key)
        self.client = None
        self.last_error: Optional[str] = None

        if not self.enabled:
            return

        try:
            from supermemory import Supermemory  # type: ignore

            self.client = Supermemory(api_key=self.api_key)
        except Exception as exc:
            self.enabled = False
            self.last_error = f"Supermemory unavailable: {exc}"

    @property
    def container_tags(self) -> List[str]:
        return [self.container_tag, self.user_id]

    def remember_workflow(self, workflow: Dict[str, Any]) -> Optional[Any]:
        if not self.enabled or not self.client:
            return None

        content = self._workflow_content(workflow)
        metadata = _metadata({
            "kind": "cosmic_browser_workflow",
            "workflow_id": workflow.get("workflow_id"),
            "domain": workflow.get("domain"),
            "task_signature": workflow.get("task_signature"),
            "generalization_level": workflow.get("generalization_level"),
            "indexer_source": (workflow.get("indexer") or {}).get("source"),
            "user_id": self.user_id,
        })
        try:
            return self.client.add(
                content=content,
                container_tag=self.space_tag,
                custom_id=_safe_tag(str(workflow.get("workflow_id") or "")),
                entity_context=(
                    "Browser-agent traversal memory for COSMIC. "
                    "The content describes reusable page states, visual indexes, replay actions, "
                    "failure patches, and user context for completing web workflows."
                ),
                metadata=metadata,
                task_type=self.task_type,
            )
        except Exception as exc:
            self.last_error = f"Supermemory add failed: {exc}"
            return None

    def search(self, task: str, domain: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
        if not self.enabled or not self.client:
            return []

        query = (
            "Browser traversal workflow memory.\n"
            f"Task: {task}\n"
            f"Domain: {domain or 'unknown'}\n"
            "Retrieve matching workflows, route memory, visual indexes, user preferences, and failure patches."
        )
        try:
            response = self.client.search.memories(
                q=query,
                container_tag=self.space_tag,
                limit=limit,
                rerank=True,
                rewrite_query=True,
                search_mode="hybrid",
            )
            raw_results = getattr(response, "results", response)
            if raw_results is None:
                return []
            if isinstance(raw_results, list):
                return [self._result_to_dict(item) for item in raw_results]
            return [self._result_to_dict(raw_results)]
        except TypeError:
            try:
                response = self.client.search.memories(
                    q=query,
                    container_tag=self.space_tag,
                    search_mode="hybrid",
                )
                raw_results = getattr(response, "results", response)
                return [self._result_to_dict(item) for item in (raw_results or [])]
            except Exception as exc:
                self.last_error = f"Supermemory search failed: {exc}"
                return []
        except Exception as exc:
            self.last_error = f"Supermemory search failed: {exc}"
            return []

    def profile(self, task: str, domain: Optional[str] = None) -> Optional[Any]:
        if not self.enabled or not self.client:
            return None
        try:
            return self.client.profile(
                container_tag=self.space_tag,
                q=f"Browser task: {task}\nDomain: {domain or 'unknown'}",
            )
        except Exception as exc:
            self.last_error = f"Supermemory profile failed: {exc}"
            return None

    def _workflow_content(self, workflow: Dict[str, Any]) -> str:
        slim_steps = []
        for step in workflow.get("steps", [])[:20]:
            visual = step.get("visual_index") or {}
            source_visual = step.get("source_visual_index") or {}
            slim_steps.append(
                {
                    "step_id": step.get("step_id"),
                    "from_state_id": step.get("from_state_id"),
                    "action_type": step.get("action_type"),
                    "target": step.get("target_description"),
                    "role": step.get("role"),
                    "checkpoint": step.get("checkpoint"),
                    "delay_ms": step.get("delay_ms"),
                    "visual_index": {
                        "normalized_coordinates": visual.get("normalized_coordinates"),
                        "target_description": visual.get("target_description"),
                    },
                    "source_visual_index": {
                        "normalized_coordinates": source_visual.get("normalized_coordinates"),
                        "target_description": source_visual.get("target_description"),
                        "page_url": source_visual.get("page_url"),
                        "page_title": source_visual.get("page_title"),
                    },
                    "visual_index_replay_policy": step.get("visual_index_replay_policy"),
                    "parameters_template": step.get("parameters_template"),
                    "expected_result": step.get("expected_result"),
                    "to_state_id": step.get("to_state_id"),
                }
            )

        return (
            "COSMIC browser traversal workflow memory\n"
            f"Workflow ID: {workflow.get('workflow_id')}\n"
            f"Domain: {workflow.get('domain')}\n"
            f"Intent: {workflow.get('intent')}\n"
            f"Task signature: {workflow.get('task_signature')}\n"
            f"Summary: {workflow.get('summary')}\n"
            f"Route strategy: {workflow.get('route_strategy')}\n"
            f"Generalization: {workflow.get('generalization_level')}\n"
            f"Quality: {json.dumps(workflow.get('quality', {}), ensure_ascii=False)}\n"
            f"Variables: {json.dumps(workflow.get('variables', []), ensure_ascii=False)}\n"
            f"Acceptance criteria: {json.dumps(workflow.get('acceptance_criteria', {}), ensure_ascii=False)}\n"
            f"Page states: {json.dumps(workflow.get('page_states', []), ensure_ascii=False)}\n"
            f"Replayable steps: {json.dumps(slim_steps, ensure_ascii=False)}\n"
            f"Discarded trace steps: {json.dumps(workflow.get('discarded_steps', []), ensure_ascii=False)}\n"
            f"Trace evidence: {json.dumps(workflow.get('trace_evidence', [])[:30], ensure_ascii=False)}\n"
            f"Failure patches: {json.dumps(workflow.get('failure_patches', []), ensure_ascii=False)}"
        )

    @staticmethod
    def _result_to_dict(item: Any) -> Dict[str, Any]:
        if isinstance(item, dict):
            return item
        if hasattr(item, "model_dump"):
            return item.model_dump()
        if hasattr(item, "__dict__"):
            return dict(item.__dict__)
        return {"value": str(item)}
