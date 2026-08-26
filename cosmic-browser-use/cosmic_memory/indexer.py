"""LLM-assisted run indexing for replayable browser workflows."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse

from openai import OpenAI

from .coordinates import VISUAL_ACTIONS
from .workflow_store import infer_domain, intent_tokens


INDEXER_VERSION = "cosmic_run_indexer_v2"


def _strip_reasoning_markers(text: str) -> str:
    if not text:
        return ""
    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    if "```" in stripped:
        stripped = re.sub(r"```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```\s*", "", stripped).strip()
    return stripped.strip()


def _extract_json_object(text: str, required_keys: Optional[set[str]] = None) -> Optional[Dict[str, Any]]:
    cleaned = _strip_reasoning_markers(text)
    decoder = json.JSONDecoder()
    for idx, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(cleaned[idx:])
            if isinstance(obj, dict) and (not required_keys or required_keys.issubset(set(obj.keys()))):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def _host(url: str) -> Optional[str]:
    try:
        host = urlparse(url or "").netloc.lower().removeprefix("www.")
        return host or None
    except Exception:
        return None


def _short(value: Any, limit: int = 360) -> str:
    text = str(value or "")
    return text if len(text) <= limit else text[: limit - 3] + "..."


@dataclass
class IndexerConfig:
    enabled: bool = True
    provider: str = "fireworks_kimi"
    model: str = "accounts/fireworks/models/kimi-k2p6"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 4096
    timeout_sec: float = 90.0


class WorkflowRunIndexer:
    """Distill noisy browser traces into compact executable workflow plans."""

    def __init__(self, config: IndexerConfig):
        self.config = config
        self.last_error: Optional[str] = None
        self.last_plan: Optional[Dict[str, Any]] = None
        self.client: Optional[OpenAI] = None
        if self.config.enabled and self.config.api_key:
            kwargs: Dict[str, Any] = {
                "api_key": self.config.api_key,
                "timeout": self.config.timeout_sec,
            }
            if self.config.api_base:
                kwargs["base_url"] = self.config.api_base.rstrip("/")
            self.client = OpenAI(**kwargs)

    @classmethod
    def from_env(cls) -> "WorkflowRunIndexer":
        provider = os.getenv("COSMIC_INDEXER_PROVIDER", "fireworks_kimi").strip().lower()
        enabled = os.getenv("COSMIC_INDEXER_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}
        if provider in {"fireworks", "fireworks_kimi", "kimi"}:
            model = os.getenv("COSMIC_INDEXER_MODEL") or os.getenv("FIREWORKS_KIMI_MODEL", "accounts/fireworks/models/kimi-k2p6")
            api_key = os.getenv("COSMIC_INDEXER_API_KEY") or os.getenv("FIREWORKS_API_KEY") or os.getenv("SLIDE_AGENT_FIREWORKS_API_KEY")
            api_base = os.getenv("COSMIC_INDEXER_BASE_URL") or os.getenv("FIREWORKS_BASE_URL") or "https://api.fireworks.ai/inference/v1"
            provider = "fireworks_kimi"
        else:
            model = os.getenv("COSMIC_INDEXER_MODEL", "gpt-4o")
            api_key = os.getenv("COSMIC_INDEXER_API_KEY") or os.getenv("OPENAI_API_KEY")
            api_base = os.getenv("COSMIC_INDEXER_BASE_URL")
            provider = "openai"

        return cls(
            IndexerConfig(
                enabled=enabled,
                provider=provider,
                model=model.strip().strip('"'),
                api_key=api_key,
                api_base=api_base,
                temperature=float(os.getenv("COSMIC_INDEXER_TEMPERATURE", "0.1")),
                max_tokens=int(os.getenv("COSMIC_INDEXER_MAX_TOKENS", "4096")),
                timeout_sec=float(os.getenv("COSMIC_INDEXER_TIMEOUT", "90")),
            )
        )

    def distill(
        self,
        *,
        task: str,
        steps: Iterable[Dict[str, Any]],
        run_dir: str,
        status: str,
    ) -> Dict[str, Any]:
        trace = self.build_trace(task=task, steps=steps, run_dir=run_dir, status=status)
        plan: Optional[Dict[str, Any]] = None
        if self.config.enabled and self.client:
            plan = self._distill_with_llm(trace)
        if not plan:
            plan = self._fallback_plan(trace)
        plan = self._validate_plan(plan, trace)
        if not (plan.get("quality") or {}).get("usable") or not plan.get("selected_steps"):
            fallback = self._fallback_plan(trace)
            fallback.setdefault("indexer", {})
            fallback["indexer"]["llm_rejected_plan"] = {
                "quality": plan.get("quality"),
                "selected_step_count": len(plan.get("selected_steps") or []),
                "indexer": plan.get("indexer"),
            }
            plan = self._validate_plan(fallback, trace)
        self.last_plan = plan
        self._write_indexer_report(run_dir, trace, plan)
        return plan

    def build_trace(
        self,
        *,
        task: str,
        steps: Iterable[Dict[str, Any]],
        run_dir: str,
        status: str,
    ) -> Dict[str, Any]:
        step_rows = [dict(step) for step in steps]
        final_note = ""
        final_url = ""
        final_title = ""
        observed_urls: List[str] = []
        trace_steps: List[Dict[str, Any]] = []

        for step in step_rows:
            action = step.get("action") or {}
            before = step.get("before_browser_state") or step.get("browser_state") or {}
            after = step.get("after_browser_state") or step.get("browser_state") or {}
            tool_call = step.get("tool_call") or {}
            visual_index = step.get("visual_index") or action.get("metadata", {}).get("visual_index")
            for state in (before, after):
                url = state.get("url")
                if url and url not in observed_urls:
                    observed_urls.append(url)
            if action.get("action_type") == "SaveNote" and action.get("success"):
                params = tool_call.get("parameters") or {}
                final_note = params.get("note") or action.get("description") or final_note
                final_url = before.get("url") or after.get("url") or final_url
                final_title = before.get("title") or after.get("title") or final_title

            trace_steps.append(
                {
                    "source_step": step.get("step"),
                    "action_type": action.get("action_type"),
                    "success": action.get("success"),
                    "verification_status": action.get("verification_status"),
                    "error": action.get("error"),
                    "description": action.get("description"),
                    "reasoning": _short(step.get("thinking"), 500),
                    "parameters": tool_call.get("parameters") or {},
                    "coordinates": action.get("coordinates"),
                    "has_visual_index": bool(visual_index),
                    "visual_index": self._slim_visual_index(visual_index),
                    "before": self._state_summary(before),
                    "after": self._state_summary(after),
                    "screenshot_path": step.get("screenshot_path"),
                    "after_screenshot_path": step.get("after_screenshot_path"),
                }
            )

        domain = infer_domain(task, final_url or (observed_urls[0] if observed_urls else None)) or _host(final_url)
        return {
            "indexer_version": INDEXER_VERSION,
            "task": task,
            "status": status,
            "run_dir": run_dir,
            "domain": domain,
            "final": {
                "note": final_note,
                "url": final_url,
                "title": final_title,
            },
            "observed_urls": observed_urls[:60],
            "steps": trace_steps,
        }

    def _distill_with_llm(self, trace: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        system_prompt = """You are the COSMIC browser run indexer.

Your task: distill a noisy browser-agent run into a compact replayable workflow.

If you need to reason through the trace step-by-step first, do that ENTIRELY inside
a single <think>...</think> block. After the closing </think> tag, output ONLY the
JSON object described below — no prose, no markdown, no step-by-step narration
outside the think block. Analysis that leaks outside <think> tags breaks the
caller's parser and the whole run fails to index.

        Schema:
{
  "quality": {
    "usable": true,
    "confidence": 0.0,
    "generalization_level": "exact|template|partial",
    "reason": "short"
  },
  "workflow": {
    "name": "short workflow name",
    "intent": "canonical intent",
    "domain": "domain",
    "summary": "what this workflow does and why it is reusable",
    "route_strategy": "how replay should use this",
    "replay_instructions": {
      "do": ["specific positive instructions for the replay planner"],
      "avoid": ["mistakes from discarded trace steps that replay must not repeat"],
      "checkpoints": ["states where replay should pause or verify"],
      "stop_conditions": ["conditions that mean replay should hand back to the live agent"]
    },
    "variables": [
      {"name": "search_query", "default": "...", "source": "goal", "applies_to_source_steps": [3]}
    ],
    "acceptance_criteria": {
      "must_match": ["important visible words or URL hints"],
      "reject": ["wrong target words"]
    }
  },
  "selected_steps": [
    {
      "source_step": 3,
      "action_type": "VisualType",
      "role": "search_query|navigation|open_target|expand_content|extract_answer|checkpoint",
      "target_description": "clean target description",
      "parameters_template": {},
      "delay_ms": 900,
      "checkpoint": true,
      "expected_result": "observable result",
      "mutation": "as_recorded|parameterized|synthetic_observed_url",
      "source_evidence_steps": [3]
    }
  ],
  "discarded_steps": [
    {"source_step": 4, "reason": "wrong target or retry"}
  ],
  "failure_patches": [
    {"applies_to_source_step": 4, "failure": "...", "fix": "...", "confidence": 0.0}
  ]
}

Rules:
- Top-level JSON MUST contain keys: quality, workflow, selected_steps, discarded_steps, failure_patches.
- Do not invent visual coordinates. Visual indexes are attached later from source_step only.
- You may include synthetic Navigate actions only to URLs that appear in observed_urls or final.url.
- Prefer gold-path steps that move toward final success. Discard wrong pages, retries, waits, failed clicks, and detours.
- If the run found the right final page only through a detour, create a partial/template workflow with a checkpoint where the dynamic choice must be re-decided.
- For dynamic web search results, keep search-box typing and checkpoint after results unless the trace contains a reliable correct-result click.
- Put anti-detour guidance in workflow.replay_instructions.avoid and failure_patches so replay does not repeat the wrong path.
- SaveNote can be selected as the terminal extract_answer step, but use a parameterized note template if the answer text must be read live.
- Mark usable=false only if there is no successful final answer and no reusable route segment.
- If a typed-text value (DomType/VisualType parameters) looks like personal information — name, email, phone, password, address, resume/cover-letter content — ALWAYS create a `variables` entry for it instead of hardcoding the literal value in parameters_template. This trace may have been recorded by a human demonstrating a task (e.g. a job application) and could contain real personal data that must not leak into a shared, reusable workflow.
"""
        payload = {
            "task": trace.get("task"),
            "status": trace.get("status"),
            "domain": trace.get("domain"),
            "final": trace.get("final"),
            "observed_urls": trace.get("observed_urls"),
            "steps": trace.get("steps"),
        }
        def _call(extra_user_suffix: str = "") -> str:
            kwargs = {
                "model": self.config.model,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False) + extra_user_suffix},
                ],
                "response_format": {"type": "json_object"},
            }
            try:
                response = self.client.chat.completions.create(**kwargs)  # type: ignore[union-attr]
            except Exception as first_exc:
                if "response_format" not in str(first_exc):
                    raise
                kwargs.pop("response_format", None)
                response = self.client.chat.completions.create(**kwargs)  # type: ignore[union-attr]
            msg = response.choices[0].message
            content = msg.content or ""
            if not content.strip():
                # Same Fireworks Kimi quirk seen in the summarizer: the real
                # answer sometimes lands in reasoning_content instead of content.
                content = str(getattr(msg, "reasoning_content", None) or "")
            return content

        try:
            content = _call()
            parsed = _extract_json_object(content, required_keys={"quality", "workflow", "selected_steps"})
            if not parsed:
                # Kimi sometimes narrates ("The user wants me to distill...")
                # instead of emitting JSON, even with response_format set to
                # json_object — that mode isn't reliably enforced for this
                # model. One retry with an explicit reminder recovers most of
                # these cases; if it still fails, the caller falls back to
                # the deterministic plan rather than losing the run entirely.
                self.last_error = f"LLM indexer returned non-JSON, retrying once: {_strip_reasoning_markers(content)[:200]}"
                content = _call(
                    "\n\nYour last response put analysis outside <think> tags, which broke the parser. "
                    "Wrap ALL step-by-step reasoning inside <think>...</think>, then output ONLY the JSON "
                    "object after the closing tag."
                )
                parsed = _extract_json_object(content, required_keys={"quality", "workflow", "selected_steps"})
            if parsed:
                parsed.setdefault("indexer", {})
                parsed["indexer"].update(
                    {
                        "source": "llm",
                        "provider": self.config.provider,
                        "model": self.config.model,
                        "version": INDEXER_VERSION,
                    }
                )
                return parsed
            self.last_error = f"LLM indexer returned non-JSON after retry: {_strip_reasoning_markers(content)[:240]}"
        except Exception as exc:
            self.last_error = f"LLM indexer failed: {exc}"
        return None

    def _fallback_plan(self, trace: Dict[str, Any]) -> Dict[str, Any]:
        final = trace.get("final") or {}
        final_url = final.get("url") or ""
        final_host = _host(final_url)
        final_title = final.get("title") or ""
        selected: List[Dict[str, Any]] = []
        discarded: List[Dict[str, Any]] = []
        final_page_seen = False
        inserted_final_navigate = False

        for row in trace.get("steps", []):
            source_step = row.get("source_step")
            action_type = row.get("action_type")
            success = bool(row.get("success"))
            before_url = (row.get("before") or {}).get("url") or ""
            after_url = (row.get("after") or {}).get("url") or ""
            before_title = (row.get("before") or {}).get("title") or ""
            after_title = (row.get("after") or {}).get("title") or ""

            if not success:
                discarded.append({"source_step": source_step, "reason": "failed action"})
                continue
            if action_type in {"TimedWait", "VisualWait", "Screenshot", "ReadHistory"}:
                discarded.append({"source_step": source_step, "reason": "not useful for replay"})
                continue
            if action_type == "SaveNote":
                if final_page_seen or final_url in before_url:
                    selected.append(self._directive_from_trace_row(row, role="extract_answer", checkpoint=True))
                else:
                    discarded.append({"source_step": source_step, "reason": "terminal note not on final target"})
                continue

            on_wrong_watch = (
                final_host
                and final_host in before_url
                and "/watch" in before_url
                and final_url
                and final_url not in before_url
                and final_title not in before_title
            )
            leads_wrong_watch = (
                final_host
                and final_host in after_url
                and "/watch" in after_url
                and final_url
                and final_url not in after_url
            )
            if on_wrong_watch or leads_wrong_watch:
                discarded.append({"source_step": source_step, "reason": "wrong target page detour"})
                continue

            if final_url and final_url in before_url and not inserted_final_navigate:
                selected.append(self._synthetic_final_navigate(trace, final_url, final_title))
                inserted_final_navigate = True

            if final_url and (final_url in before_url or final_url in after_url):
                final_page_seen = True

            if action_type == "Navigate":
                selected.append(self._directive_from_trace_row(row, role="navigation", checkpoint=True))
            elif action_type in {"VisualClick", "VisualType", "VisualHover", "PressKey", "VisualScroll", "DOMClick", "DOMExtract"}:
                if action_type in VISUAL_ACTIONS and not row.get("has_visual_index"):
                    discarded.append({"source_step": source_step, "reason": "visual action had no visual index"})
                    continue
                if final_page_seen or "youtube.com" in before_url or source_step in {1, 2, 3}:
                    role = "search_query" if action_type == "VisualType" else "open_target"
                    if final_url in before_url:
                        role = "expand_content"
                    selected.append(self._directive_from_trace_row(row, role=role, checkpoint=action_type in {"Navigate", "VisualClick", "VisualType"}))

        if final_url and selected and not any((d.get("parameters_template") or {}).get("url") == final_url for d in selected):
            selected.append(self._synthetic_final_navigate(trace, final_url, final_title))

        return {
            "quality": {
                "usable": bool(selected),
                "confidence": 0.55 if selected else 0.0,
                "generalization_level": "partial",
                "reason": "Deterministic fallback selected successful replayable steps and observed final URL.",
            },
            "workflow": {
                "name": f"{trace.get('domain') or 'site'}: {trace.get('task')}",
                "intent": trace.get("task"),
                "domain": trace.get("domain"),
                "summary": "Fallback workflow from successful actions, visual indexes, and observed final URL.",
                "route_strategy": "Use indexed actions until checkpoint; dynamic result choices may require normal planner repair.",
                "replay_instructions": self._fallback_replay_instructions(trace, discarded),
                "variables": self._fallback_variables(trace),
                "acceptance_criteria": {
                    "must_match": intent_tokens(trace.get("task", ""))[:8],
                    "reject": ["wrong target", "shorts", "playlist", "demo clip"],
                },
            },
            "selected_steps": selected,
            "discarded_steps": discarded,
            "failure_patches": self._fallback_failure_patches(trace),
            "indexer": {
                "source": "deterministic_fallback",
                "version": INDEXER_VERSION,
                "last_error": self.last_error,
            },
        }

    def _validate_plan(self, plan: Dict[str, Any], trace: Dict[str, Any]) -> Dict[str, Any]:
        valid_steps = {row.get("source_step"): row for row in trace.get("steps", [])}
        observed_urls = set(trace.get("observed_urls") or [])
        final_url = (trace.get("final") or {}).get("url")
        if final_url:
            observed_urls.add(final_url)

        cleaned = []
        for item in plan.get("selected_steps") or []:
            if not isinstance(item, dict):
                continue
            source_step = item.get("source_step")
            action_type = item.get("action_type")
            if source_step is not None and source_step not in valid_steps:
                continue
            if action_type in VISUAL_ACTIONS and source_step is not None and not valid_steps[source_step].get("has_visual_index"):
                continue
            if source_step is None:
                params = item.get("parameters_template") or {}
                url = params.get("url")
                if action_type != "Navigate" or not url or url not in observed_urls:
                    continue
            cleaned.append(item)

        plan["selected_steps"] = cleaned
        quality = plan.setdefault("quality", {})
        quality["usable"] = bool(cleaned) and bool(quality.get("usable", True))
        plan.setdefault("discarded_steps", [])
        plan.setdefault("failure_patches", [])
        workflow = plan.setdefault("workflow", {})
        workflow["replay_instructions"] = self._normalize_replay_instructions(
            workflow.get("replay_instructions"),
            plan.get("discarded_steps") or [],
            plan.get("failure_patches") or [],
            trace,
        )
        plan.setdefault("indexer", {"source": "unknown", "version": INDEXER_VERSION})
        return plan

    def _normalize_replay_instructions(
        self,
        raw: Any,
        discarded_steps: List[Dict[str, Any]],
        failure_patches: List[Dict[str, Any]],
        trace: Dict[str, Any],
    ) -> Dict[str, List[str]]:
        if isinstance(raw, dict):
            instructions = {
                "do": list(raw.get("do") or []),
                "avoid": list(raw.get("avoid") or []),
                "checkpoints": list(raw.get("checkpoints") or []),
                "stop_conditions": list(raw.get("stop_conditions") or []),
            }
        else:
            instructions = {"do": [], "avoid": [], "checkpoints": [], "stop_conditions": []}

        final = trace.get("final") or {}
        task_tokens = intent_tokens(trace.get("task", ""))[:8]
        if task_tokens:
            instructions["do"].append(f"Preserve goal-specific target hints: {', '.join(task_tokens)}.")
        if final.get("url"):
            instructions["do"].append(f"Prefer the observed successful final URL when it still matches the goal: {final['url']}.")
        if final.get("title"):
            instructions["checkpoints"].append(f"Target page title should resemble: {final['title']}.")

        for row in discarded_steps:
            reason = str(row.get("reason") or "").strip()
            source_step = row.get("source_step")
            if reason:
                instructions["avoid"].append(f"Do not replay source step {source_step}: {reason}.")
        for patch in failure_patches:
            failure = str(patch.get("failure") or "").strip()
            fix = str(patch.get("fix") or "").strip()
            if failure and fix:
                instructions["avoid"].append(f"Avoid failure: {failure}.")
                instructions["do"].append(f"Patch: {fix}.")

        instructions["checkpoints"].append("Pause after dynamic search/result-selection states unless the next action is replay-safe.")
        instructions["stop_conditions"].append("If the current page no longer matches the workflow state, stop replay and hand back to the live planner.")
        instructions["stop_conditions"].append("If an answer text must be read from the current screenshot, stop before templated SaveNote and let the live answer finalizer read it.")

        normalized: Dict[str, List[str]] = {}
        for key, values in instructions.items():
            seen = set()
            cleaned = []
            for value in values:
                text = _short(value, 260).strip()
                if text and text not in seen:
                    cleaned.append(text)
                    seen.add(text)
            normalized[key] = cleaned[:12]
        return normalized

    def _fallback_replay_instructions(self, trace: Dict[str, Any], discarded_steps: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        return self._normalize_replay_instructions(
            None,
            discarded_steps,
            self._fallback_failure_patches(trace),
            trace,
        )

    def _directive_from_trace_row(self, row: Dict[str, Any], *, role: str, checkpoint: bool) -> Dict[str, Any]:
        return {
            "source_step": row.get("source_step"),
            "action_type": row.get("action_type"),
            "role": role,
            "target_description": row.get("description") or row.get("action_type"),
            "parameters_template": row.get("parameters") or {},
            "delay_ms": self._delay_for(row.get("action_type")),
            "checkpoint": checkpoint,
            "expected_result": self._expected_from_row(row),
            "mutation": "as_recorded",
            "source_evidence_steps": [row.get("source_step")],
        }

    def _fallback_variables(self, trace: Dict[str, Any]) -> List[Dict[str, Any]]:
        variables = []
        for row in trace.get("steps", []):
            if row.get("action_type") == "VisualType":
                params = row.get("parameters") or {}
                if params.get("text"):
                    variables.append(
                        {
                            "name": "search_query",
                            "default": params["text"],
                            "source": "goal",
                            "applies_to_source_steps": [row.get("source_step")],
                        }
                    )
        return variables[:5]

    def _synthetic_final_navigate(self, trace: Dict[str, Any], final_url: str, final_title: str) -> Dict[str, Any]:
        return {
            "source_step": None,
            "action_type": "Navigate",
            "role": "direct_final_target",
            "target_description": f"Observed final target URL: {final_url}",
            "parameters_template": {"url": final_url},
            "delay_ms": 1500,
            "checkpoint": True,
            "expected_result": f"Browser reaches {final_title or final_url}",
            "mutation": "synthetic_observed_url",
            "source_evidence_steps": [
                row.get("source_step")
                for row in trace.get("steps", [])
                if final_url in ((row.get("before") or {}).get("url") or "") or final_url in ((row.get("after") or {}).get("url") or "")
            ][:5],
        }

    def _fallback_failure_patches(self, trace: Dict[str, Any]) -> List[Dict[str, Any]]:
        patches = []
        for row in trace.get("steps", []):
            if row.get("success") is False or row.get("error"):
                patches.append(
                    {
                        "applies_to_source_step": row.get("source_step"),
                        "failure": row.get("error") or f"{row.get('action_type')} did not complete",
                        "fix": "Resume with current screenshot and let the normal browser agent repair before continuing indexed replay.",
                        "confidence": 0.5,
                    }
                )
        return patches[:12]

    def _slim_visual_index(self, visual_index: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not visual_index:
            return None
        grounding = visual_index.get("mimo_grounding") or {}
        return {
            "target_description": visual_index.get("target_description"),
            "viewport": visual_index.get("viewport"),
            "pixel_coordinates": visual_index.get("pixel_coordinates"),
            "normalized_coordinates": visual_index.get("normalized_coordinates"),
            "page_url": visual_index.get("page_url"),
            "page_title": visual_index.get("page_title"),
            "edge_guard": grounding.get("edge_guard"),
            "parse_retry": bool(grounding.get("parse_retry")),
        }

    def _state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "url": state.get("url"),
            "title": state.get("title"),
            "scroll_y": state.get("scroll_y"),
            "viewport": state.get("viewport"),
            "ready_state": state.get("ready_state"),
        }

    def _expected_from_row(self, row: Dict[str, Any]) -> str:
        after = row.get("after") or {}
        title = after.get("title") or "next state"
        url = after.get("url") or ""
        return f"Browser reaches {title} {url}".strip()

    def _delay_for(self, action_type: Optional[str]) -> int:
        if action_type == "Navigate":
            return 1500
        if action_type in {"VisualClick", "DOMClick", "PressKey"}:
            return 900
        if action_type == "VisualType":
            return 800
        if action_type == "VisualScroll":
            return 500
        return 300

    def _write_indexer_report(self, run_dir: str, trace: Dict[str, Any], plan: Dict[str, Any]) -> None:
        try:
            debug_dir = Path(run_dir) / "cosmic_debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            report_path = debug_dir / "indexer_plan.json"
            report_path.write_text(
                json.dumps(
                    {
                        "trace_summary": {
                            "task": trace.get("task"),
                            "status": trace.get("status"),
                            "domain": trace.get("domain"),
                            "final": trace.get("final"),
                            "step_count": len(trace.get("steps") or []),
                        },
                        "plan": plan,
                        "last_error": self.last_error,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass


def load_run_steps(run_dir: str | Path) -> List[Dict[str, Any]]:
    path = Path(run_dir) / "log.json"
    return json.loads(path.read_text(encoding="utf-8"))
