"""High-level COSMIC memory runtime wiring."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .retriever import MemoryRetriever
from .supermemory_client import SupermemoryMemoryClient
from .trace_compiler import TraceCompiler
from .workflow_store import WorkflowStore


class CosmicMemoryRuntime:
    def __init__(
        self,
        data_dir: str | Path = "./data/cosmic_memory",
        user_id: str = "demo_user",
        container_tag: str = "cosmic-hackathon-demo",
        supermemory_enabled: bool = True,
    ):
        self.store = WorkflowStore(data_dir)
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
