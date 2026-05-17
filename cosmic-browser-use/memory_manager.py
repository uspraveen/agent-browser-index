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
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from collections import deque
from openai import OpenAI

from cosmic_types import Step, BrowserState, ActionResult, TaskConfig
import os
from dotenv import load_dotenv
load_dotenv()

class MemoryManager:
    """
    Manages conversation history, summaries, and context for the agent.
    """
    
    
    def __init__(self, config: TaskConfig, working_dir: Path, api_key: Optional[str] = None):
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
        self._compression_task: Optional[asyncio.Task] = None
        self.total_tokens_saved: int = 0
        
        # Initialize OpenAI for compression
        self.client = OpenAI(api_key=api_key) if api_key else None
        
    def add_step(
        self, 
        screenshot_path: str,
        screenshot_hash: str,
        browser_state: BrowserState,
        action: Optional[ActionResult] = None,
        summary: str = "",
        thinking: Optional[str] = None,
    ) -> Step:
        step_number = len(self.steps) + 1
        
        step = Step(
            step_number=step_number,
            timestamp=datetime.now(),
            screenshot_path=screenshot_path,
            screenshot_hash=screenshot_hash,
            browser_state=browser_state,
            action=action,
            summary=summary,
            thinking=thinking,
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
        
        # Trigger compression if needed (every N steps, configurable via config.summary_interval)
        if step_number % self.config.summary_interval == 0 and step_number > 5:
            if self._compression_task is None or self._compression_task.done():
                self._compression_task = asyncio.create_task(self._compress_summary())
        
        return step
    
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
        """Compress older steps in the summary using GPT-5.2."""
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
            history_text += f"Step {s.step_number} ({status}): {action_desc}\n"
            if s.browser_state.notes:
                history_text += f"  Notes: {s.browser_state.notes[-1]}\n"

        prompt = f"""
        You are the Memory Manager for an autonomous browser agent.
        
        Current Summary (Up to Step {self.steps[self.last_summarized_idx].step_number - 1 if self.last_summarized_idx > 0 else 0}):
        {self.cumulative_summary}
        
        New Steps to Compress (Steps {steps_to_summarize[0].step_number} to {steps_to_summarize[-1].step_number}):
        {history_text}
        
        Task: Merge the 'New Steps' into the 'Current Summary' to create a concise narrative.
        
        CRITICAL RULES:
        1. Keep the summary under 550 words.
        2. EXPLICITLY retain any SAVED NOTES (e.g., "Step 5: Saved note '...'").
        3. Mention outcome of KEY GOALS (e.g., "Logged in successfully").
        4. Drop repetitive navigation details (e.g., "scrolled down 5 times" -> "browsed page").
        5. PRESERVE FAILURES: You MUST retain the specific details of any action that failed (e.g., "Tried simple-login-page.com but failed"). This is critical so the agent does not repeat mistakes.
        """
        
        try:
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: self.client.responses.create(
                    model=os.getenv("SUMMARY_LLM_MODEL", "gpt-5.2-mini"),
                    input=prompt,
                    reasoning={"effort": "medium"},
                    text={"verbosity": "low"}
                )
            )
            new_summary = response.output_text
            
            # Update stats
            saved = len(history_text)
            self.total_tokens_saved += saved
            self.cumulative_summary = new_summary
            self.last_summarized_idx += len(steps_to_summarize)
            
            print(f"\n🧠 [Context Compression] Summary Updated (Saved ~{saved} chars)")
            print(f"   Consumed Range: Steps {steps_to_summarize[0].step_number}-{steps_to_summarize[-1].step_number}")
            print(f"   New Summary: {new_summary[:100]}...")
            
        except Exception as e:
            print(f"Compression failed: {e}")
            
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
        }
        
        if self.steps:
            last_step = self.steps[-1]
            context["browser_state"] = last_step.browser_state.to_dict()
            context["last_action"] = last_step.action.to_dict() if last_step.action else None
            
            if include_screenshots:
                context["screenshots"] = {
                    "current": current_screenshot_path,
                    "previous": last_step.screenshot_path,
                }
        else:
            context["browser_state"] = None
            context["last_action"] = None
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
        
        # Pattern 1: All identical actions
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
        }
