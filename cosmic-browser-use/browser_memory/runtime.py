"""High-level browser workflow memory runtime wiring."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .retriever import MemoryRetriever
from .supermemory_client import SupermemoryMemoryClient
from .trace_compiler import TraceCompiler
from .workflow_store import WorkflowStore

LEGACY_MEMORY_DIR = "./data/cosmic_memory"
DEFAULT_MEMORY_DIR = "./data/browser_memory"


def resolve_memory_dir(data_dir: str | Path | None = None) -> Path:
    """Resolve the workflow-memory directory.

    Defaults to ./data/browser_memory, but transparently adopts the legacy
    ./data/cosmic_memory location when it exists and the new one does not, so
    existing workflow libraries keep working after the rename.
    """
    if data_dir:
        return Path(data_dir).expanduser()
    new_dir = Path(DEFAULT_MEMORY_DIR)
    legacy_dir = Path(LEGACY_MEMORY_DIR)
    if not new_dir.exists() and legacy_dir.exists():
        return legacy_dir
    return new_dir


class BrowserMemoryRuntime:
    def __init__(
        self,
        data_dir: str | Path | None = None,
        user_id: str = "demo_user",
        container_tag: str = "cosmic-hackathon-demo",
        supermemory_enabled: bool = True,
    ):
        self.store = WorkflowStore(resolve_memory_dir(data_dir))
        self.supermemory = SupermemoryMemoryClient(
            api_key=os.getenv("SUPERMEMORY_API_KEY"),
            container_tag=container_tag,
            user_id=user_id,
            enabled=supermemory_enabled,
            task_type="memory",
        )
        self.retriever = MemoryRetriever(self.store, self.supermemory)
        self.compiler = TraceCompiler(self.store, self.supermemory)

    def retrieve(self, task: str, domain: Optional[str] = None, current_page_summary: Optional[str] = None) -> Dict[str, Any]:
        return self.retriever.retrieve(task, domain, current_page_summary)

    def compile_run(
        self,
        task: str,
        steps: Iterable[Any],
        run_dir: str,
        status: str = "success",
        workflow_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        return self.compiler.compile_from_steps(
            task=task,
            steps=steps,
            run_dir=run_dir,
            status=status,
            workflow_id=workflow_id,
        )
