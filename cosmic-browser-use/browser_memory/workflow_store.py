"""Local JSON workflow store for executable traversal memory."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse


KNOWN_DOMAINS = {
    "youtube": "youtube.com",
    "github": "github.com",
    "amazon": "amazon.com",
    "wikipedia": "wikipedia.org",
    "google flights": "google.com/travel/flights",
    "flights": "google.com/travel/flights",
    "opentable": "opentable.com",
    "arxiv": "arxiv.org",
    "pubmed": "pubmed.ncbi.nlm.nih.gov",
    "reddit": "reddit.com",
    "hacker news": "news.ycombinator.com",
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "for",
    "from",
    "get",
    "in",
    "into",
    "me",
    "of",
    "on",
    "open",
    "please",
    "the",
    "to",
    "with",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(value: str, default: str = "workflow") -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug[:80] or default


def infer_domain(text: str = "", url: Optional[str] = None) -> Optional[str]:
    candidates = []
    if url:
        candidates.append(url)
    if text:
        candidates.append(text)

    for candidate in candidates:
        for raw_url in re.findall(r"https?://[^\s)>\"]+", candidate):
            parsed = urlparse(raw_url)
            if parsed.netloc:
                return parsed.netloc.lower().removeprefix("www.")

    lower = (text or "").lower()
    for key, domain in KNOWN_DOMAINS.items():
        if key in lower:
            return domain
    return None


def intent_tokens(text: str) -> List[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(token) > 2 and token not in STOPWORDS
    ]


def task_signature(goal: str, domain: Optional[str] = None) -> str:
    tokens = intent_tokens(goal)
    if domain:
        tokens.insert(0, domain.split("/")[0].replace(".", "_"))
    return slugify("_".join(tokens[:10]), default="task")


class WorkflowStore:
    def __init__(self, data_dir: str | Path):
        self.root = Path(data_dir)
        self.workflows_dir = self.root / "workflows"
        self.index_path = self.root / "workflow_index.json"
        self.workflows_dir.mkdir(parents=True, exist_ok=True)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.index_path.exists():
            self._write_index([])

    def _read_index(self) -> List[Dict[str, Any]]:
        try:
            return json.loads(self.index_path.read_text(encoding="utf-8"))
        except Exception:
            return []

    def _write_index(self, rows: List[Dict[str, Any]]) -> None:
        self.index_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    def list_workflows(self, domain: Optional[str] = None, intent: Optional[str] = None) -> List[Dict[str, Any]]:
        rows = self._read_index()
        if domain:
            domain_lower = domain.lower()
            rows = [row for row in rows if domain_lower in str(row.get("domain", "")).lower()]
        if intent:
            tokens = set(intent_tokens(intent))
            rows = [
                row
                for row in rows
                if tokens.intersection(set(row.get("intent_tokens", [])))
            ]
        return rows

    def load_workflow(self, workflow_id: str) -> Dict[str, Any]:
        path = self.workflows_dir / f"{workflow_id}.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def save_workflow(self, workflow: Dict[str, Any]) -> str:
        workflow_id = workflow.get("workflow_id")
        if not workflow_id:
            domain = workflow.get("domain") or "unknown"
            signature = task_signature(workflow.get("intent", ""), domain)
            workflow_id = f"{signature}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            workflow["workflow_id"] = workflow_id

        workflow["updated_at"] = now_iso()
        if not workflow.get("created_at"):
            workflow["created_at"] = workflow["updated_at"]

        path = self.workflows_dir / f"{workflow_id}.json"
        path.write_text(json.dumps(workflow, indent=2, ensure_ascii=False), encoding="utf-8")

        rows = [row for row in self._read_index() if row.get("workflow_id") != workflow_id]
        rows.append(
            {
                "workflow_id": workflow_id,
                "name": workflow.get("name"),
                "domain": workflow.get("domain"),
                "intent": workflow.get("intent"),
                "intent_tokens": intent_tokens(workflow.get("intent", "")),
                "task_signature": workflow.get("task_signature"),
                "summary": workflow.get("summary", ""),
                "success_count": workflow.get("success_count", 0),
                "step_count": len(workflow.get("steps", [])),
                "updated_at": workflow["updated_at"],
                "path": str(path),
            }
        )
        rows.sort(key=lambda row: row.get("updated_at", ""), reverse=True)
        self._write_index(rows)
        return workflow_id
