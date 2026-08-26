"""COSMIC cross-run browser traversal memory.

Submodule imports are intentionally lazy: importing ``cosmic_memory.coordinates``
must not pull Playwright, OpenAI, or Supermemory into every process (e.g. a
phone-triggered browser subprocess should start quickly).
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "build_visual_index",
    "replay_coordinates",
    "CosmicDebugLogger",
    "CosmicMemoryRuntime",
    "DemoOverlayManager",
    "MemoryRetriever",
    "SupermemoryMemoryClient",
    "TraceCompiler",
    "WorkflowRunIndexer",
    "WorkflowStore",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "build_visual_index": (".coordinates", "build_visual_index"),
    "replay_coordinates": (".coordinates", "replay_coordinates"),
    "CosmicDebugLogger": (".debug_log", "CosmicDebugLogger"),
    "DemoOverlayManager": (".demo_overlay", "DemoOverlayManager"),
    "WorkflowRunIndexer": (".indexer", "WorkflowRunIndexer"),
    "MemoryRetriever": (".retriever", "MemoryRetriever"),
    "CosmicMemoryRuntime": (".runtime", "CosmicMemoryRuntime"),
    "SupermemoryMemoryClient": (".supermemory_client", "SupermemoryMemoryClient"),
    "TraceCompiler": (".trace_compiler", "TraceCompiler"),
    "WorkflowStore": (".workflow_store", "WorkflowStore"),
}


def __getattr__(name: str) -> Any:
    spec = _LAZY_EXPORTS.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = spec
    import importlib

    module = importlib.import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
