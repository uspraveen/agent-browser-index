#!/usr/bin/env python3
"""
Cosmic Browser Use Agent - Main execution loop with timing
"""
import asyncio
import base64
import argparse
import time
import sys
from pathlib import Path
from datetime import datetime

# Windows Asyncio Fix
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from cosmic_types import (
    TaskConfig, LLMConfig, LLMProvider, LLMTier,
    VerificationStatus, ActionType, ActionResult
)
from memory_manager import MemoryManager
from orchestrator import Orchestrator
from browser_controller import BrowserController
from find_coordinates_mimo import check_mimo_health

import os
from dotenv import load_dotenv

load_dotenv()
# ==============================================================================

async def run_task(
    goal: str,
    initial_url: str = None, # Optional
    max_steps: int = 1000,   # Default increased to 1000
    fast_model_config: LLMConfig = None,
    medium_model_config: LLMConfig = None,
    slow_model_config: LLMConfig = None,
    mimo_api_url: str = None,
    mimo_api_key: str = None,
    headless: bool = None,
    # SDK-ready configuration overrides
    summary_interval: int = None,
    max_tabs: int = None,
    screenshot_quality: int = None,
    ask_user_timeout: int = None,
    large_notes_path: str = None,
):
    mimo_api_url = mimo_api_url or os.getenv("MIMO_API_URL", "https://mbhl6tqhfyvdd3tu.us-east-1.aws.endpoints.huggingface.cloud/v1/")
    mimo_api_key = mimo_api_key or os.getenv("MIMO_API_KEY")
    headless = headless if headless is not None else os.getenv("HEADLESS", "False").lower() == "true"
    summary_interval = summary_interval if summary_interval is not None else int(os.getenv("SUMMARY_INTERVAL_STEPS", "10"))
    max_tabs = max_tabs if max_tabs is not None else int(os.getenv("MAX_TABS", "5"))
    screenshot_quality = screenshot_quality if screenshot_quality is not None else int(os.getenv("SCREENSHOT_QUALITY", "50"))
    ask_user_timeout = ask_user_timeout if ask_user_timeout is not None else int(os.getenv("ASK_USER_TIMEOUT", "120"))
    """
    Run a browser automation task with comprehensive timing.

    This function is the primary entry point for both CLI and SDK usage.
    All behavioral parameters are exposed as keyword arguments with sensible
    defaults from config.py, making it easy to override from an API layer.
    """

    # Setup working directory
    working_dir = Path(f"./runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    working_dir.mkdir(parents=True, exist_ok=True)
    resolved_large_notes_path = Path(large_notes_path).expanduser() if large_notes_path else (working_dir / "large_notes.jsonl")

    config = TaskConfig(
        task_id=f"task_{datetime.now().timestamp()}",
        goal=goal,
        max_steps=max_steps,
        screenshot_quality=screenshot_quality,
        summary_interval=summary_interval,
        max_tabs=max_tabs,
        ask_user_timeout=ask_user_timeout,
    )
    
    print(f"\n{'='*80}")
    print(f"COSMIC BROWSER USE AGENT - Task Started")
    print(f"{'='*80}")
    print(f"Goal: {goal}")
    print(f"Provider: {fast_model_config.provider.value}")
    print(f"Initial URL: {initial_url if initial_url else 'about:blank'}")
    print(f"Max steps: {max_steps}")
    print(f"Working directory: {working_dir}")
    print(f"Large notes file: {resolved_large_notes_path}")
    print(f"{'='*80}\n")
    
    # Initialize components
    memory = MemoryManager(config, working_dir, api_key=os.getenv("OPENAI_API_KEY"))
    
    orchestrator = Orchestrator(
        fast_model=fast_model_config,
        medium_model=medium_model_config,
        slow_model=slow_model_config,
    )
    
    browser = BrowserController(
        config=config,
        mimo_api_url=mimo_api_url,
        mimo_api_key=mimo_api_key,
        working_dir=working_dir,
        large_notes_path=resolved_large_notes_path,
        headless=headless,
    )
    
    await browser.start(initial_url)
    
    # Track total task execution time
    task_start_time = time.time()
    
    # Main loop
    previous_confidence = 1.0
    checkpoint_path = None
    
    try:
        for step_num in range(1, max_steps + 1):
            # Track step execution time
            step_start_time = time.time()
            
            print(f"\n{'='*80}")
            print(f"STEP {step_num}/{max_steps}")
            print(f"{'='*80}")
            
            # 1. Capture current state
            capture_start = time.time()
            screenshot_path, screenshot_hash, browser_state = await browser.capture_state(
                f"step_{step_num:03d}"
            )
            capture_time_ms = (time.time() - capture_start) * 1000
            
            print(f"📸 Screenshot: {screenshot_path}")
            print(f"🌐 URL: {browser_state.url}")
            print(f"📄 Title: {browser_state.title}")
            print(f"⏱️  Capture time: {capture_time_ms:.0f}ms")
            
            # 2. Check loop detection
            if memory.detect_loop():
                print("\n⚠️  LOOP DETECTED - Escalating to slow model")
                previous_confidence = 0.0
            
            # 3. Get LLM decision
            llm_start = time.time()
            context = memory.get_context_for_llm(screenshot_path)
            
            # Load screenshot as base64
            with open(screenshot_path, 'rb') as f:
                screenshot_b64 = f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode()}"
            
            llm_response = await orchestrator.decide_action(
                context=context,
                screenshot_base64=screenshot_b64,
                previous_confidence=previous_confidence,
            )
            llm_time_ms = (time.time() - llm_start) * 1000
            
            print(f"\n🤖 LLM Decision:")
            print(f"   Action: {llm_response.tool_call.action_type.value}")
            print(f"   Params: {llm_response.tool_call.parameters}")
            print(f"   Confidence: {llm_response.confidence:.2f}")
            print(f"   Progress: {llm_response.estimated_completion:.0%}")
            print(f"   ⏱️  LLM time: {llm_time_ms:.0f}ms")
            
            if llm_response.reasoning:
                print(f"   Reasoning: {llm_response.reasoning[:150]}...")
            
            # 4. Execute action
            execution_start = time.time()
            action_result = None
            
            # Special handling for ReadHistory
            if llm_response.tool_call.action_type == ActionType.READ_HISTORY:
                try:
                    start_step = int(llm_response.tool_call.parameters.get("start_step", 1))
                    end_step = int(llm_response.tool_call.parameters.get("end_step", len(memory.steps)))
                    history_text = memory.read_history(start_step, end_step)
                    
                    action_result = ActionResult(
                        success=True,
                        action_type=ActionType.READ_HISTORY,
                        description=f"Read history from step {start_step} to {end_step}",
                        output=history_text,
                        execution_time_ms=(time.time() - execution_start) * 1000
                    )
                except Exception as e:
                     action_result = ActionResult(
                        success=False,
                        action_type=ActionType.READ_HISTORY,
                        description="Failed to read history",
                        error=str(e),
                        execution_time_ms=(time.time() - execution_start) * 1000
                    )

            # Normal execution
            if not action_result:
                action_result = await browser.execute_tool(
                    llm_response.tool_call,
                    screenshot_path,
                )
            # execution_time_ms already tracked in action_result
            
            print(f"\n⚡ Execution: {'✓ Success' if action_result.success else '✗ Failed'}")
            if action_result.coordinates:
                print(f"   Coordinates: {action_result.coordinates}")
            if action_result.error:
                print(f"   Error: {action_result.error}")
            print(f"   ⏱️  Action execution time: {action_result.execution_time_ms:.0f}ms")
            
            # 5. Wait for page to settle
            await asyncio.sleep(0.5)
            
            # 6. Capture new state for verification
            verification_start = time.time()
            _, _, new_browser_state = await browser.capture_state(f"step_{step_num:03d}_after")
            
            # 7. Verify action
            verification_status, change_score = await browser.verify_action(
                before_state=browser_state,
                after_state=new_browser_state,
                verification_hint=llm_response.tool_call.verification_hint,
            )
            verification_time_ms = (time.time() - verification_start) * 1000
            
            action_result.verification_status = verification_status
            action_result.state_change_score = change_score
            action_result.estimated_completion = llm_response.estimated_completion
            
            print(f"\n✅ Verification:")
            print(f"   Status: {verification_status.value}")
            print(f"   State change: {change_score:.2f}")
            print(f"   ⏱️  Verification time: {verification_time_ms:.0f}ms")
            
            # 8. Add to memory
            memory.add_step(
                screenshot_path=screenshot_path,
                screenshot_hash=screenshot_hash,
                browser_state=new_browser_state,
                action=action_result,
                thinking=llm_response.reasoning,
            )
            
            # 9. Calculate and print step execution time
            step_duration_ms = (time.time() - step_start_time) * 1000
            print(f"\n⏱️  STEP {step_num} TOTAL TIME: {step_duration_ms:.0f}ms ({step_duration_ms/1000:.2f}s)")
            print(f"   Breakdown: Capture={capture_time_ms:.0f}ms | LLM={llm_time_ms:.0f}ms | " +
                  f"Action={action_result.execution_time_ms:.0f}ms | Verify={verification_time_ms:.0f}ms")
            
            # 10. Update confidence for next iteration
            previous_confidence = llm_response.confidence
            
            # 11. Check completion
            # 11. Check completion
            if llm_response.estimated_completion >= 0.95:
                if action_result and action_result.success:
                    print("\n🎉 GOAL ACHIEVED - Task complete!")
                    # Print the agent's saved notes as the final answer
                    final_notes = []
                    if memory.steps:
                        last_state = memory.steps[-1].browser_state
                        if last_state and last_state.notes:
                            final_notes = last_state.notes
                    if final_notes:
                        print("\n" + "="*80)
                        print("📋 RESULT")
                        print("="*80)
                        for note in final_notes:
                            print(f"  {note}")
                        print("="*80)
                    break
                else:
                    print("\n⚠️  LLM indicated completion but action FAILED - Continuing...")
                    previous_confidence = 0.0 # Force escalation
            
            # 12. Check failure conditions
            if verification_status == VerificationStatus.LOOP_DETECTED:
                print("\n⚠️  Loop detected - attempting recovery...")
                previous_confidence = 0.0  # Force escalation
                
            if verification_status == VerificationStatus.ERROR:
                if llm_response.confidence < 0.3:
                    print("\n❌ CRITICAL ERROR - Manual intervention needed")
                    break
    finally:
        # Calculate total execution time
        total_duration_sec = time.time() - task_start_time
        
        # Final statistics
        print(f"\n{'='*80}")
        print(f"TASK COMPLETE")
        print(f"{'='*80}")
        print(f"\n📊 Execution Summary:")
        print(f"   Steps taken: {len(memory.steps)}")
        print(f"   ⏱️  Total execution time: {total_duration_sec:.2f}s ({total_duration_sec/60:.2f} minutes)")
        print(f"   ⏱️  Average time per step: {(total_duration_sec / len(memory.steps)):.2f}s" if memory.steps else "   N/A")
        
        print(f"\n💾 Memory Statistics:")
        mem_stats = memory.get_stats()
        for key, value in mem_stats.items():
            print(f"   {key}: {value}")
        
        print(f"\n🤖 Orchestrator Statistics:")
        orch_stats = orchestrator.get_stats()
        for key, value in orch_stats.items():
            print(f"   {key}: {value}")
        
        print(f"\n🌐 Browser Statistics:")
        browser_stats = browser.get_stats()
        for key, value in browser_stats.items():
            print(f"   {key}: {value}")
        
        # Save checkpoint
        checkpoint_path = memory.save_checkpoint()
        print(f"\n💾 Checkpoint saved: {checkpoint_path}")
        print(f"{'='*80}\n")
        
        # Cleanup
        await browser.close()
        await orchestrator.close()
    
    return {
        "success": True,
        "steps_taken": len(memory.steps),
        "total_time_sec": total_duration_sec,
        "avg_time_per_step_sec": total_duration_sec / len(memory.steps) if memory.steps else 0,
        "checkpoint_path": str(checkpoint_path),
        "working_dir": str(working_dir),
    }

async def main():
    # 1. Argument Parsing
    parser = argparse.ArgumentParser(description="Cosmic Browser Use Agent")
    parser.add_argument("--goal", type=str, required=True, help="The task you want the agent to perform.")
    parser.add_argument("--provider", type=str, choices=["openai", "anthropic", "gemini"], default="openai", help="LLM Provider to use.")
    parser.add_argument("--url", type=str, default=None, help="Starting URL (optional).")
    parser.add_argument("--steps", type=int, default=1000, help="Max steps (default 1000).")
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode.")
    parser.add_argument("--mimo-url", type=str, default=os.getenv("MIMO_API_URL", "https://mbhl6tqhfyvdd3tu.us-east-1.aws.endpoints.huggingface.cloud/v1/"), help="MiMo API URL.")
    parser.add_argument("--mimo-api-key", type=str, default=os.getenv("MIMO_API_KEY"), help="MiMo API Key for authentication.")

    # Model overrides (override defaults for the selected provider)
    parser.add_argument("--fast-model", type=str, default=None, help="Model ID for fast tier (overrides provider default).")
    parser.add_argument("--slow-model", type=str, default=None, help="Model ID for slow tier (overrides provider default).")
    parser.add_argument("--api-key", type=str, default=None, help="API key for the selected provider.")
    parser.add_argument("--temperature", type=float, default=None, help="LLM temperature (overrides provider default).")

    # Agent tuning
    parser.add_argument("--summary-interval", type=int, default=int(os.getenv("SUMMARY_INTERVAL_STEPS", "10")), help="Steps between memory compressions.")
    parser.add_argument("--max-tabs", type=int, default=int(os.getenv("MAX_TABS", "5")), help="Maximum open browser tabs.")
    parser.add_argument("--screenshot-quality", type=int, default=int(os.getenv("SCREENSHOT_QUALITY", "50")), help="Screenshot JPEG quality 1-100.")
    parser.add_argument("--ask-user-timeout", type=int, default=int(os.getenv("ASK_USER_TIMEOUT", "120")), help="Seconds to wait for user response.")
    parser.add_argument("--large-notes-path", type=str, default=None, help="Path to external large-notes JSONL file. Default: <run working dir>/large_notes.jsonl")

    args = parser.parse_args()

    print("\n" + "="*80)
    print("COSMIC BROWSER USE AGENT")
    print("Vision-Based Browser Automation with MiMo-VL")
    print("="*80)

    # 2. Setup Config based on Provider selection
    #    CLI flags --fast-model, --slow-model, --api-key, --temperature override defaults
    if args.provider == "openai":
        api_key = args.api_key or os.getenv("OPENAI_API_KEY")
        fast_model = args.fast_model or os.getenv("OPENAI_FAST_MODEL", "gpt-4o")
        slow_model = args.slow_model or os.getenv("OPENAI_SLOW_MODEL", "gpt-4o")
        default_temp = 0.3
        fast_config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model_id=fast_model,
            api_key=api_key,
            tier=LLMTier.FAST,
            timeout_ms=15000,
            temperature=args.temperature if args.temperature is not None else default_temp,
        )
        slow_config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model_id=slow_model,
            api_key=api_key,
            tier=LLMTier.SLOW,
            timeout_ms=90000,
            temperature=args.temperature if args.temperature is not None else default_temp,
        )
    elif args.provider == "anthropic":
        api_key = args.api_key or os.getenv("ANTHROPIC_API_KEY")
        fast_model = args.fast_model or os.getenv("CLAUDE_FAST_MODEL", "claude-haiku-4-5")
        slow_model = args.slow_model or os.getenv("CLAUDE_SLOW_MODEL", "claude-haiku-4-5")
        default_temp = 0.3
        fast_config = LLMConfig(
            provider=LLMProvider.CLAUDE,
            model_id=fast_model,
            api_key=api_key,
            tier=LLMTier.FAST,
            timeout_ms=15000,
            temperature=args.temperature if args.temperature is not None else default_temp,
        )
        slow_config = LLMConfig(
            provider=LLMProvider.CLAUDE,
            model_id=slow_model,
            api_key=api_key,
            tier=LLMTier.SLOW,
            timeout_ms=90000,
            temperature=args.temperature if args.temperature is not None else default_temp,
        )
    elif args.provider == "gemini":
        api_key = args.api_key or os.getenv("GEMINI_API_KEY")
        fast_model = args.fast_model or os.getenv("GEMINI_FAST_MODEL", "gemini-3-pro-preview")
        slow_model = args.slow_model or os.getenv("GEMINI_SLOW_MODEL", "gemini-3-pro-preview")
        default_temp = 1.0  # Gemini 3 default - DO NOT change unless user explicitly overrides
        fast_config = LLMConfig(
            provider=LLMProvider.GEMINI,
            model_id=fast_model,
            api_key=api_key,
            tier=LLMTier.FAST,
            timeout_ms=15000,
            temperature=args.temperature if args.temperature is not None else default_temp,
        )
        slow_config = LLMConfig(
            provider=LLMProvider.GEMINI,
            model_id=slow_model,
            api_key=api_key,
            tier=LLMTier.SLOW,
            timeout_ms=90000,
            temperature=args.temperature if args.temperature is not None else default_temp,
        )

    # 3. Pre-check MiMo Availability
    print("\n[PRE-CHECK] verifying MiMo vision server...")
    if not check_mimo_health(args.mimo_url, api_key=args.mimo_api_key):
        print(f"\n❌ CRITICAL ERROR: MiMo Vision Server is unreachable at: {args.mimo_url}")
        print("   This is a vision-dominant system and cannot function without MiMo.")
        print("   Please ensure the server is running and accessible.")
        sys.exit(1)
    print("✅ MiMo-VL Server is online and ready.")

    # 4. RunTask
    try:
        await run_task(
            goal=args.goal,
            initial_url=args.url,
            max_steps=args.steps,
            fast_model_config=fast_config,
            slow_model_config=slow_config,
            mimo_api_url=args.mimo_url,
            mimo_api_key=args.mimo_api_key,
            headless=args.headless,
            summary_interval=args.summary_interval,
            max_tabs=args.max_tabs,
            screenshot_quality=args.screenshot_quality,
            ask_user_timeout=args.ask_user_timeout,
            large_notes_path=args.large_notes_path,
        )
    except Exception as e:
        print(f"\n❌ Execution Failed: {e}")
        # raise # Uncomment to see full traceback

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Task interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        raise
