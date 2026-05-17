"""Workflow retrieval and scoring."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .workflow_store import WorkflowStore, infer_domain, intent_tokens


class MemoryRetriever:
    def __init__(self, store: WorkflowStore, supermemory_client: Any = None):
        self.store = store
        self.supermemory = supermemory_client

    def retrieve(
        self,
        task: str,
        domain: Optional[str] = None,
        current_page_summary: Optional[str] = None,
    ) -> Dict[str, Any]:
        inferred_domain = domain or infer_domain(task)
        semantic = self.supermemory.search(task, inferred_domain) if self.supermemory else []
        candidates = self.store.list_workflows(domain=inferred_domain) if inferred_domain else self.store.list_workflows()

        if not candidates:
            candidates = self.store.list_workflows(intent=task)

        scored = []
        for row in candidates:
            score = self._score(row, task, inferred_domain, current_page_summary)
            if score > 0:
                scored.append({"score": score, "workflow": row})
        scored.sort(key=lambda item: item["score"], reverse=True)

        best = None
        if scored:
            best = self.store.load_workflow(scored[0]["workflow"]["workflow_id"])

        return {
            "task": task,
            "domain": inferred_domain,
            "supermemory_results": semantic,
            "candidate_workflows": scored,
            "best_workflow": best,
            "best_score": scored[0]["score"] if scored else 0.0,
        }

    def _score(
        self,
        row: Dict[str, Any],
        task: str,
        domain: Optional[str],
        current_page_summary: Optional[str],
    ) -> float:
        task_tokens = set(intent_tokens(task))
        row_tokens = set(row.get("intent_tokens") or intent_tokens(row.get("intent", "")))
        overlap = len(task_tokens.intersection(row_tokens))
        union = max(1, len(task_tokens.union(row_tokens)))
        intent_similarity = overlap / union

        domain_match = 0.0
        if domain and row.get("domain"):
            domain_match = 1.0 if domain in row["domain"] or row["domain"] in domain else 0.0

        page_match = 0.0
        if current_page_summary:
            summary_tokens = set(intent_tokens(current_page_summary))
            page_match = len(summary_tokens.intersection(row_tokens)) / max(1, len(summary_tokens.union(row_tokens)))

        success_score = min(1.0, float(row.get("success_count") or 0) / 5.0)
        recency_score = self._recency(row.get("updated_at"))

        return (
            0.35 * intent_similarity
            + 0.25 * domain_match
            + 0.20 * page_match
            + 0.10 * success_score
            + 0.10 * recency_score
        )

    @staticmethod
    def _recency(value: Optional[str]) -> float:
        if not value:
            return 0.0
        try:
            updated = datetime.fromisoformat(value.replace("Z", "+00:00"))
            age_days = max(0.0, (datetime.now(timezone.utc) - updated).total_seconds() / 86400.0)
            return math.exp(-age_days / 30.0)
        except Exception:
            return 0.0
