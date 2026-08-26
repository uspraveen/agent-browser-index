#!/usr/bin/env python3
"""
Memory and context management for Cosmic Browser Use Agent

Handles:
- Step history tracking with timestamps
- Cumulative summary with automatic compression
- Screenshot management
- Context assembly for LLM calls
"""
import asyncio
import hashlib
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from collections import deque
from openai import OpenAI
from cli_labels import display_provider_model

from cosmic_types import Step, BrowserState, ActionResult, TaskConfig
from cosmic_memory.coordinates import build_visual_index
import os
from dotenv import load_dotenv
load_dotenv()

class MemoryManager:
    """
    Manages conversation history, summaries, and context for the agent.
    """
    
    
    def __init__(
        self,
        config: TaskConfig,
        working_dir: Path,
        api_key: Optional[str] = None,
        summary_provider: Optional[str] = None,
        summary_model: Optional[str] = None,
        summary_api_key: Optional[str] = None,
        summary_api_base: Optional[str] = None,
    ):
        self.config = config
        self.working_dir = working_dir
        self.screenshots_dir = working_dir / "screenshots"
        self.screenshots_dir.mkdir(exist_ok=True)
        self.log_path = working_dir / "log.json"
        
        # Initialize storage
        if not self.log_path.exists():
            with open(self.log_path, 'w') as f:
                json.dump([], f)
        
        self.steps: List[Step] = []
        self.cumulative_summary: str = "Task Started."
        self.last_summarized_idx: int = 0
        self.recent_action_hashes: deque = deque(maxlen=config.loop_detection_window)
        self.total_tokens_saved: int = 0
        
        # Initialize compression LLM. It can use OpenAI Responses or Fireworks
        # Kimi through the OpenAI-compatible chat-completions API.
        self.summary_provider = (summary_provider or os.getenv("SUMMARY_LLM_PROVIDER") or "openai").strip().lower()
        self.summary_model = (
            summary_model
            or os.getenv("SUMMARY_LLM_MODEL")
            or (
                os.getenv("FIREWORKS_KIMI_MODEL", "accounts/fireworks/models/kimi-k2p6")
                if self.summary_provider in {"fireworks", "fireworks_kimi", "kimi"}
                else "gpt-4o-mini"
            )
        ).strip().strip('"')
        self.summary_temperature = float(os.getenv("SUMMARY_LLM_TEMPERATURE", "0.1"))
        self.summary_max_tokens = int(os.getenv("SUMMARY_LLM_MAX_TOKENS", "900"))

        if self.summary_provider in {"fireworks", "fireworks_kimi", "kimi"}:
            self.summary_provider = "fireworks_kimi"
            key = summary_api_key or os.getenv("FIREWORKS_API_KEY") or os.getenv("SLIDE_AGENT_FIREWORKS_API_KEY")
            base_url = (summary_api_base or os.getenv("FIREWORKS_BASE_URL") or "https://api.fireworks.ai/inference/v1").rstrip("/")
            self.client = OpenAI(api_key=key, base_url=base_url) if key else None
            if not summary_model and not os.getenv("SUMMARY_LLM_MODEL"):
                self.summary_model = os.getenv("FIREWORKS_KIMI_MODEL", "accounts/fireworks/models/kimi-k2p6").strip().strip('"')
        else:
            self.summary_provider = "openai"
            key = summary_api_key or api_key or os.getenv("OPENAI_API_KEY")
            self.client = OpenAI(api_key=key) if key else None
        
    def add_step(
        self, 
        screenshot_path: str,
        screenshot_hash: str,
        browser_state: BrowserState,
        action: Optional[ActionResult] = None,
        summary: str = "",
        thinking: Optional[str] = None,
        before_browser_state: Optional[BrowserState] = None,
        after_browser_state: Optional[BrowserState] = None,
        after_screenshot_path: Optional[str] = None,
        after_screenshot_hash: Optional[str] = None,
        tool_call: Optional[Dict[str, Any]] = None,
        llm_response: Optional[Dict[str, Any]] = None,
        visual_index: Optional[Dict[str, Any]] = None,
    ) -> Step:
        step_number = len(self.steps) + 1
        if visual_index is None and action and before_browser_state:
            visual_index = build_visual_index(
                action=action,
                tool_call=tool_call,
                before_state=before_browser_state,
                screenshot_path=screenshot_path,
            )
            if visual_index is not None:
                action.metadata.setdefault("visual_index", visual_index)
        
        step = Step(
            step_number=step_number,
            timestamp=datetime.now(),
            screenshot_path=screenshot_path,
            screenshot_hash=screenshot_hash,
            browser_state=browser_state,
            action=action,
            summary=summary,
            thinking=thinking,
            before_browser_state=before_browser_state,
            after_browser_state=after_browser_state,
            after_screenshot_path=after_screenshot_path,
            after_screenshot_hash=after_screenshot_hash,
            tool_call=tool_call,
            llm_response=llm_response,
            visual_index=visual_index,
        )
        
        self.steps.append(step)
        self._append_to_summary(step)
        
        if action:
            action_hash = self._hash_action(action)
            self.recent_action_hashes.append(action_hash)
            
        # Atomic Write to Disk
        try:
            with open(self.log_path, 'r') as f:
                log_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            log_data = []
        
        log_data.append(step.to_dict())
        
        with open(self.log_path, 'w') as f:
            json.dump(log_data, f, indent=2, default=str)
        
        return step

    def compression_due(self) -> bool:
        step_number = len(self.steps)
        return step_number % self.config.summary_interval == 0 and step_number > 5

    async def compress_if_due(self) -> None:
        """Run compression synchronously when due. Previously this fired as a
        fire-and-forget background asyncio task — but that ran CONCURRENTLY
        with the main loop's own LLM call to the same provider/account, and
        under Fireworks' per-account concurrency limits one of the two
        requests would queue behind the other, surfacing as unexplained
        30-150s stalls on ordinary steps. Awaiting it directly trades that
        unpredictable spike for one small, predictable pause every N steps."""
        if self.compression_due():
            await self._compress_summary()

    def _append_to_summary(self, step: Step) -> None:
        if step.summary:
            summary_text = step.summary
        elif step.action:
            status = ""
            if not step.action.success:
                status = f" (FAILED: {step.action.error})"
            summary_text = f"{step.action.action_type.value}: {step.action.description}{status}"
        else:
            summary_text = "Captured state"
        
        self.cumulative_summary += f" Step {step.step_number}: {summary_text}."
    
    async def _compress_summary(self) -> None:
        """Compress older steps in the summary using the configured summary LLM."""
        if len(self.steps) < 10 or not self.client:
           return
        
        # Define window
        SLIDING_WINDOW = 5
        # Only summarize what we haven't touched yet, minus the sliding window
        steps_to_summarize = self.steps[self.last_summarized_idx : -SLIDING_WINDOW]
        
        if not steps_to_summarize:
            return

        history_text = ""
        for s in steps_to_summarize:
            action_desc = s.action.description if s.action else "No Action"
            status = "FAILED" if s.action and not s.action.success else "SUCCESS"
            action_type = s.action.action_type.value if s.action else "None"
            verification = s.action.verification_status.value if s.action and s.action.verification_status else "unknown"
            progress = s.action.estimated_completion if s.action else 0.0
            params = {}
            if s.tool_call:
                params = s.tool_call.get("parameters", {}) or {}
            params_text = json.dumps(params, ensure_ascii=False, separators=(",", ":"))
            if len(params_text) > 300:
                params_text = params_text[:297] + "..."
            url = s.browser_state.url if s.browser_state else ""
            title = s.browser_state.title if s.browser_state else ""
            scroll_y = s.browser_state.scroll_y if s.browser_state else None
            err = f" error={s.action.error}" if s.action and s.action.error else ""
            history_text += (
                f"Step {s.step_number} ({status}): action={action_type}; desc={action_desc}; "
                f"params={params_text}; verification={verification}; progress={progress:.2f}; "
                f"url={url}; title={title}; scroll_y={scroll_y}{err}\n"
            )
            if s.browser_state.notes:
                history_text += f"  Notes: {s.browser_state.notes[-1]}\n"

        last_summarized_step = self.steps[self.last_summarized_idx].step_number - 1 if self.last_summarized_idx > 0 else 0
        prompt = f"""CURRENT SUMMARY (steps 1–{last_summarized_step}):
{self.cumulative_summary}

NEW STEPS TO ABSORB (steps {steps_to_summarize[0].step_number}–{steps_to_summarize[-1].step_number}):
{history_text}

Write an updated summary that absorbs the new steps into the current summary. Rules:
- Under 550 words. Plain prose only, no headers or bullets.
- Include every SAVED NOTE verbatim (e.g. "Step 5 saved: '...'").
- State key outcomes explicitly (e.g. "Signed in to LinkedIn — now on /feed/ at step 8").
- Preserve loop/wasted-motion patterns verbatim so the agent avoids repeating them.
- Preserve failures with specifics (what was tried, why it failed).
- Preserve wrong-page/wrong-target events exactly.
- Preserve URL/title transitions that show drift or recovery.

If you need to reason through the steps first, do that ENTIRELY inside a single
<think>...</think> block — then, after the closing </think> tag, return ONLY
this JSON object (no other text):
{{"updated_summary": "the prose summary as a single string"}}"""
        
        try:
            loop = asyncio.get_running_loop()
            new_summary = await loop.run_in_executor(None, lambda: self._call_summary_model(prompt))

            if self._looks_like_echoed_instructions(new_summary):
                # Before giving up, check if this is just a throwaway preamble
                # sentence followed by an actual usable summary — don't discard
                # good content just because it opened with "The user wants me
                # to..." before getting to the real answer.
                salvaged = self._salvage_after_preamble(new_summary)
                if salvaged:
                    new_summary = salvaged
                else:
                    print("⚠️  [Context Compression] Model echoed the instructions instead of summarizing — retrying once.")
                    retry_prompt = prompt + (
                        "\n\nYour last response put analysis outside <think> tags, which broke the parser. "
                        "Wrap ALL reasoning inside <think>...</think>, then write ONLY the updated summary "
                        "prose after the closing tag."
                    )
                    new_summary = await loop.run_in_executor(None, lambda: self._call_summary_model(retry_prompt))

            if self._looks_like_echoed_instructions(new_summary):
                # Do NOT overwrite cumulative_summary with garbage — that poisons
                # every subsequent main-decision prompt for the rest of the run
                # (this exact failure mode previously caused the orchestrator's
                # own reasoning to start talking about "updating a summary"
                # instead of the actual browser task). Keep the last-good
                # summary; just drop detail for this window of steps.
                print(f"⚠️  [Context Compression] Rejected echoed-instructions output after retry — keeping previous summary, dropping detail for steps {steps_to_summarize[0].step_number}-{steps_to_summarize[-1].step_number}.")
                print(f"   Rejected output ({len(new_summary)} chars): {new_summary[:500]!r}")
                self.last_summarized_idx += len(steps_to_summarize)
                return

            # Update stats
            saved = len(history_text)
            self.total_tokens_saved += saved
            self.cumulative_summary = new_summary
            self.last_summarized_idx += len(steps_to_summarize)

            print(f"\n🧠 [Context Compression] Summary Updated (Saved ~{saved} chars)")
            print(f"   Summary model: {display_provider_model(self.summary_provider, self.summary_model)}")
            print(f"   Consumed Range: Steps {steps_to_summarize[0].step_number}-{steps_to_summarize[-1].step_number}")
            print(f"   New Summary: {new_summary[:100]}...")

        except Exception as e:
            print(f"Compression failed: {e}")

    @staticmethod
    def _looks_like_echoed_instructions(text: str) -> bool:
        """Detect when the summary model echoed the compression PROMPT back
        instead of writing an actual summary — the failure mode that, left
        unchecked, corrupts cumulative_summary permanently for the rest of
        the run (every later prompt includes it)."""
        if not text or len(text.strip()) < 20:
            return True
        lowered = text.lower().strip()
        markers = (
            "the user wants me to",
            "i need to follow",
            "let me analyze",
            "i should write",
            "the task is to merge",
            "current summary (steps",
            "new steps to absorb",
            "specific rules",
            "let's analyze",
            "i'll write",
        )
        return any(m in lowered[:250] for m in markers)

    @classmethod
    def _salvage_after_preamble(cls, text: str) -> Optional[str]:
        """If the response is a short throwaway preamble ('The user wants me
        to...') followed by a paragraph break and then real content, recover
        the part after the break instead of discarding everything. Only
        salvages when the preamble is short (near the start) and what
        follows is substantial and doesn't itself look like more
        meta-commentary — otherwise return None and let the caller retry/reject."""
        break_idx = text.find("\n\n")
        if break_idx == -1 or break_idx > 300:
            return None
        remainder = text[break_idx:].strip()
        if len(remainder) < 100:
            return None
        if cls._looks_like_echoed_instructions(remainder):
            return None
        return remainder

    def _call_summary_model(self, prompt: str) -> str:
        if not self.client:
            raise RuntimeError(f"No summary LLM client configured for provider={self.summary_provider}")

        if self.summary_provider == "fireworks_kimi":
            kwargs = {
                "model": self.summary_model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a browser-agent memory compressor. When given a CURRENT SUMMARY and "
                            "NEW STEPS, return a JSON object {\"updated_summary\": \"...\"}. If you need to "
                            "reason first, do it entirely inside <think>...</think> before the JSON — "
                            "analysis or restated instructions outside the think block or outside the JSON "
                            "object breaks the caller's parser and the summary is discarded."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": self.summary_temperature,
                "max_tokens": self.summary_max_tokens,
                "extra_body": {"top_k": 40},
                "response_format": {"type": "json_object"},
            }
            try:
                response = self.client.chat.completions.create(**kwargs)
            except Exception as first_exc:
                if "response_format" not in str(first_exc):
                    raise
                kwargs.pop("response_format", None)
                response = self.client.chat.completions.create(**kwargs)

            msg = response.choices[0].message
            content = msg.content or ""
            if not content.strip():
                # Same Fireworks Kimi quirk as the main decision path: the
                # real answer sometimes lands in reasoning_content instead.
                content = str(getattr(msg, "reasoning_content", None) or "")
            cleaned_raw = self._strip_summary_thinking(content).strip()

            extracted = self._extract_updated_summary_json(cleaned_raw)
            if extracted is not None:
                return extracted
            # JSON mode wasn't honored (model/account doesn't support it, or
            # the model still wrote plain prose) — fall back to treating the
            # post-<think>-stripped text as the summary directly, same as
            # before JSON mode was added.
            return cleaned_raw

        request_kwargs = {
            "model": self.summary_model,
            "input": prompt,
        }
        if self.summary_model.startswith("gpt-5"):
            request_kwargs["reasoning"] = {"effort": "medium"}
            request_kwargs["text"] = {"verbosity": "low"}
        response = self.client.responses.create(**request_kwargs)
        return str(response.output_text or "").strip()

    @staticmethod
    def _strip_summary_thinking(text: str) -> str:
        if not text:
            return ""
        stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
        if "```" in stripped:
            stripped = re.sub(r"```(?:[a-zA-Z0-9_-]+)?\s*", "", stripped)
            stripped = re.sub(r"\s*```\s*", "", stripped).strip()
        return stripped

    @staticmethod
    def _extract_updated_summary_json(text: str) -> Optional[str]:
        """Pull {"updated_summary": "..."} out of text that may have extra
        characters around it (JSON mode isn't always honored exactly, or the
        object may be preceded by leftover stray text). Returns None if no
        valid object with that key is found, signalling the caller to fall
        back to treating the raw text as the summary."""
        if not text:
            return None
        decoder = json.JSONDecoder()
        for idx, char in enumerate(text):
            if char != "{":
                continue
            try:
                obj, _ = decoder.raw_decode(text[idx:])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and isinstance(obj.get("updated_summary"), str):
                return obj["updated_summary"].strip()
        return None

    def read_history(self, start_step: int, end_step: int) -> str:
        """Retrieve detailed history for a range of steps."""
        start_idx = max(0, start_step - 1)
        end_idx = min(len(self.steps), end_step)

        if start_idx >= end_idx:
            return f"Invalid range. max_step is {len(self.steps)}"

        output = f"HISTORY ({start_step}-{end_step}):\n"
        for i in range(start_idx, end_idx):
            step = self.steps[i]
            output += f"\n--- STEP {step.step_number} ---\n"
            output += f"URL: {step.browser_state.url}\n"
            if step.action:
                status = "SUCCESS" if step.action.success else "FAILED"
                output += f"Action: [{status}] {step.action.action_type.value}: {step.action.description}\n"
                if step.action.verification_status:
                    output += f"Verification: {step.action.verification_status.value}\n"
                if step.action.error:
                    output += f"Error: {step.action.error}\n"
            else:
                output += f"Action: None\n"
            if step.browser_state.notes:
                output += f"Notes: {step.browser_state.notes}\n"

        return output
    
    def get_context_for_llm(
        self,
        current_screenshot_path: str,
        include_screenshots: bool = True,
    ) -> Dict[str, Any]:
        context = {
            "task_id": self.config.task_id,
            "goal": self.config.goal,
            "current_step": len(self.steps) + 1,
            "max_steps": self.config.max_steps,
            "cumulative_summary": self.cumulative_summary,
            "enable_dom_fallback": self.config.enable_dom_fallback,
        }
        
        if self.steps:
            last_step = self.steps[-1]
            context["browser_state"] = last_step.browser_state.to_dict()
            context["last_action"] = last_step.action.to_dict() if last_step.action else None
            context["recent_steps"] = self._recent_steps_for_prompt(limit=8)
            
            if include_screenshots:
                context["screenshots"] = {
                    "current": current_screenshot_path,
                    "previous": last_step.screenshot_path,
                }
        else:
            context["browser_state"] = None
            context["last_action"] = None
            context["recent_steps"] = []
            if include_screenshots:
                context["screenshots"] = {"current": current_screenshot_path}
        
        # Progress estimate
        if self.steps:
            valid_actions = [s for s in self.steps if s.action]
            if valid_actions:
                avg_completion = sum(
                    getattr(s.action, "estimated_completion", 0.0) or 0.0
                    for s in valid_actions
                ) / len(valid_actions)
                context["estimated_progress"] = min(avg_completion, 0.99)
            else:
                context["estimated_progress"] = 0.0
        else:
            context["estimated_progress"] = 0.0
        
        return context

    def _recent_steps_for_prompt(self, limit: int = 8) -> List[Dict[str, Any]]:
        """Compact, uncompressed recent action trace for loop-aware planning."""
        rows: List[Dict[str, Any]] = []
        for step in self.steps[-max(1, limit):]:
            action = step.action
            state = step.browser_state
            tool = step.tool_call or {}
            rows.append(
                {
                    "step": step.step_number,
                    "url": state.url if state else None,
                    "title": state.title if state else None,
                    "scroll_y": state.scroll_y if state else None,
                    "action_type": action.action_type.value if action else None,
                    "description": action.description if action else None,
                    "success": action.success if action else None,
                    "error": action.error if action else None,
                    "verification_status": action.verification_status.value if action and action.verification_status else None,
                    "estimated_completion": action.estimated_completion if action else None,
                    "requested_parameters": tool.get("parameters", {}),
                }
            )
        return rows
    
    def detect_loop(self) -> bool:
        """Detect if agent is stuck in a loop.
        
        A loop is only detected if:
        1. Similar actions are repeated AND
        2. The page state hasn't changed (same URL + same screenshot)
        
        This prevents false positives when navigating forms or scrolling.
        """
        if len(self.steps) < self.config.loop_detection_window:
            return False
        
        # Get last N steps
        window = self.config.loop_detection_window
        recent_steps = self.steps[-window:]
        
        # Check if actions are similar
        action_types = [s.action.action_type for s in recent_steps if s.action]
        if len(action_types) < window:
            return False  # Not enough actions to compare
        
        # Pattern 0: Same exact action description repeated, even if each
        # attempt produced a slightly different screenshot hash (hover
        # rings, focus outlines, tiny scroll jiggle from MiMo's coordinate
        # noise). Requiring byte-identical hashes (old Pattern 1 below) missed
        # this — a button re-clicked 5 times in a row with state_change=1.00
        # every time (because the hash technically differed) was never
        # flagged, even though the URL never moved and nothing progressed.
        descriptions = []
        for s in recent_steps:
            if not s.action:
                continue
            desc = None
            if s.tool_call:
                desc = (s.tool_call.get("parameters", {}) or {}).get("description")
            desc = (desc or s.action.description or "").strip().lower()
            descriptions.append(desc)
        if len(descriptions) == window and descriptions[0] and len(set(descriptions)) == 1:
            urls = [s.browser_state.url for s in recent_steps]
            if len(set(urls)) == 1:
                return True

        # Pattern 1: All identical action TYPES (but possibly different targets)
        if len(set(action_types)) == 1:
            # But check if state changed
            urls = [s.browser_state.url for s in recent_steps]
            screenshots = [s.screenshot_hash for s in recent_steps]

            # If all URLs and screenshots are identical, it's a loop
            if len(set(urls)) == 1 and len(set(screenshots)) == 1:
                return True
            # If URLs or screenshots changed, it's legitimate progress
            return False
        
        # Pattern 2: Alternating actions (A-B-A-B)
        if len(set(action_types)) == 2 and len(action_types) >= 4:
            pattern = [at.value for at in action_types]
            # Check if it's alternating
            is_alternating = all(
                pattern[i] != pattern[i+1] 
                for i in range(len(pattern)-1)
            )
            
            if is_alternating:
                # Check if state is also stuck
                urls = [s.browser_state.url for s in recent_steps]
                screenshots = [s.screenshot_hash for s in recent_steps]
                
                if len(set(urls)) == 1 and len(set(screenshots)) <= 2:
                    return True  # Likely stuck alternating between two states
        
        return False
    
    @staticmethod
    def _hash_action(action: ActionResult) -> str:
        """Create hash for action that includes URL context.
        
        Note: This is now primarily used for tracking, not loop detection.
        Loop detection uses Step objects directly to check state changes.
        """
        key = f"{action.action_type.value}:{action.description}:{action.coordinates}"
        return hashlib.md5(key.encode()).hexdigest()[:8]
    
    def save_checkpoint(self, checkpoint_path: Optional[Path] = None) -> Path:
        """Save full history to disk."""
        if checkpoint_path is None:
            # FIX: Use safe string format for Windows filenames (no colons)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = self.working_dir / f"checkpoint_{timestamp}.json"
        
        data = {
            "config": {
                "task_id": self.config.task_id,
                "goal": self.config.goal,
                "max_steps": self.config.max_steps,
            },
            "steps": [step.to_dict() for step in self.steps],
            "cumulative_summary": self.cumulative_summary,
            "total_tokens_saved": self.total_tokens_saved,
        }
        
        with open(checkpoint_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return checkpoint_path
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_steps": len(self.steps),
            "summary_length": len(self.cumulative_summary),
            "tokens_saved_via_compression": self.total_tokens_saved,
            "recent_action_window": len(self.recent_action_hashes),
            "summary_provider": self.summary_provider,
            "summary_model": self.summary_model,
        }
