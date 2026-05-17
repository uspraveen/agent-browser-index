#!/usr/bin/env python3
"""
Browser controller with atomic vision-based tools

Integrates:
- Playwright for browser automation
- MiMo-VL for vision grounding (using robust parsing logic)
- Deterministic verification
"""
import asyncio
import base64
import hashlib
import imagehash
import time
import re
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
from urllib.parse import urlparse
from PIL import Image
from io import BytesIO

from playwright.async_api import async_playwright, Page, Browser, BrowserContext
import httpx

try:
    import tiktoken  # type: ignore
except Exception:
    tiktoken = None

from cosmic_types import ActionType, ActionResult, BrowserState, TabInfo, ToolCall, VerificationStatus, TaskConfig
import os
from dotenv import load_dotenv

load_dotenv()

# ==============================================================================
# 🧠 MiMo-VL PARSING LOGIC
# ==============================================================================

def extract_thinking(raw: str) -> Tuple[Optional[str], str]:
    """Extract thinking/reasoning content from model output."""
    think_match = re.search(r'<think>(.*?)</think>', raw, re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()
        remaining = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
        return thinking, remaining
    return None, raw

def parse_coordinates(raw: str, image_size: Tuple[int, int]) -> Tuple[int, int]:
    """
    Parse coordinates from model output.
    
    MiMo-VL typically outputs pixel coordinates directly.
    Also handles normalized coordinates (0-1 range) as fallback.
    """
    w, h = image_size
    
    # Remove thinking tags if present
    _, clean = extract_thinking(raw)
    
    # Try to find [x, y] pattern (most common)
    list_match = re.search(r'\[(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\]', clean)
    if list_match:
        x, y = float(list_match.group(1)), float(list_match.group(2))
        # Check if normalized (0-1 range) or pixel coordinates
        if x <= 1.0 and y <= 1.0 and x >= 0 and y >= 0:
            if x < 2 and y < 2:  # Almost certainly normalized
                return int(round(x * w)), int(round(y * h))
        return int(round(x)), int(round(y))
    
    # Try (x, y) pattern
    tuple_match = re.search(r'\((\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\)', clean)
    if tuple_match:
        x, y = float(tuple_match.group(1)), float(tuple_match.group(2))
        if x <= 1.0 and y <= 1.0 and x >= 0 and y >= 0 and x < 2 and y < 2:
            return int(round(x * w)), int(round(y * h))
        return int(round(x)), int(round(y))
        
    # Try x=... y=... pattern
    xy_match = re.search(r'x\s*[=:]\s*(\d+(?:\.\d+)?)[,\s]+y\s*[=:]\s*(\d+(?:\.\d+)?)', clean, re.I)
    if xy_match:
        x, y = float(xy_match.group(1)), float(xy_match.group(2))
        if x <= 1.0 and y <= 1.0 and x >= 0 and y >= 0 and x < 2 and y < 2:
            return int(round(x * w)), int(round(y * h))
        return int(round(x)), int(round(y))
    
    # Try JSON with position field
    json_match = re.search(r'"position"\s*:\s*\[(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\]', clean)
    if json_match:
        x, y = float(json_match.group(1)), float(json_match.group(2))
        if x <= 1.0 and y <= 1.0 and x >= 0 and y >= 0 and x < 2 and y < 2:
            return int(round(x * w)), int(round(y * h))
        return int(round(x)), int(round(y))
    
    # Fallback: Try regex finding any two numbers
    numbers = re.findall(r'(\d+(?:\.\d+)?)', clean)
    if len(numbers) >= 2:
        x, y = float(numbers[0]), float(numbers[1])
        if x <= 1.0 and y <= 1.0 and x >= 0 and y >= 0 and x < 2 and y < 2:
            return int(round(x * w)), int(round(y * h))
        elif 0 <= x <= w * 1.1 and 0 <= y <= h * 1.1:
             return int(round(min(x, w))), int(round(min(y, h)))
    
    raise ValueError(f"Could not parse coordinates from: {raw!r}")


# ==============================================================================
# 🎮 CONTROLLER CLASS
# ==============================================================================

class BrowserController:
    """Manages browser instance and executes atomic tools."""
    
    def __init__(
        self,
        config: TaskConfig,
        mimo_api_url: str,
        working_dir: Path,
        mimo_api_key: Optional[str] = None,
        large_notes_path: Optional[Path] = None,
        headless: bool = False,
    ):
        self.config = config
        self.mimo_api_url = mimo_api_url
        self.mimo_api_key = mimo_api_key
        self.mimo_chat_completions_url = self._resolve_mimo_chat_completions_url(mimo_api_url)
        self.working_dir = working_dir
        self.large_notes_path = Path(large_notes_path) if large_notes_path else (working_dir / "large_notes.jsonl")
        self.large_notes_index_path = self.large_notes_path.parent / "large_notes_index.json"
        self.headless = headless
        
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        
        # Tab management
        self.pages: List[Page] = []
        self.active_tab_index: int = 0
        
        # Scratchpad
        self.notes: List[str] = []

        # Dialog handling queue — auto-accepted dialogs are recorded here
        self._pending_dialogs: List[Dict[str, str]] = []

        self.http_client = httpx.AsyncClient(timeout=30.0)
        self.total_actions = 0
        self.mimo_calls = 0
        self.dom_calls = 0
        self.large_note_count = 0
        # Notes policy
        self.notes_token_budget = 2000
        self.large_note_min_tokens = 300
        self.large_notes_default_list_limit = 20
        self.large_notes_default_search_limit = 10
        self._token_encoder = self._init_token_encoder()
        self._initialize_large_notes_store()

    @staticmethod
    def _resolve_mimo_chat_completions_url(mimo_api_url: str) -> str:
        """Normalize MiMo URL to a chat-completions endpoint.

        Accepts either:
        - base URL (e.g., http://host:8098)
        - /v1 URL (e.g., http://host:8098/v1)
        - full endpoint (e.g., http://host:8098/v1/chat/completions)
        """
        url = (mimo_api_url or "").strip().rstrip("/")
        if not url:
            raise ValueError("mimo_api_url must not be empty")

        if url.endswith("/v1/chat/completions") or url.endswith("/chat/completions"):
            return url
        if url.endswith("/v1/models"):
            return url[: -len("/models")] + "/chat/completions"
        if url.endswith("/v1"):
            return f"{url}/chat/completions"
        return f"{url}/v1/chat/completions"

    def _initialize_large_notes_store(self):
        """Prepare the external large-notes store and initialize counters."""
        self.large_notes_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.large_notes_path.exists():
            self.large_notes_path.write_text("", encoding="utf-8")

        # Initialize or load index
        self.large_notes_index = {}
        if self.large_notes_index_path.exists():
            try:
                with open(self.large_notes_index_path, "r", encoding="utf-8") as f:
                    self.large_notes_index = json.load(f)
            except Exception as e:
                print(f"⚠️  Failed to load large notes index: {e}. Rebuilding...")
                self._rebuild_large_notes_index()
        else:
            # Build index from existing JSONL
            self._rebuild_large_notes_index()

        self.large_note_count = len(self.large_notes_index)

    def _next_large_note_id(self) -> str:
        self.large_note_count += 1
        return f"ln_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.large_note_count:05d}"

    def _rebuild_large_notes_index(self):
        """Rebuild index from large_notes.jsonl file."""
        self.large_notes_index = {}
        if not self.large_notes_path.exists():
            self._save_large_notes_index()
            return

        try:
            with open(self.large_notes_path, "r", encoding="utf-8") as f:
                for line_num, raw_line in enumerate(f, start=1):
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        note_id = entry.get("id")
                        if note_id:
                            self.large_notes_index[note_id] = {
                                "id": note_id,
                                "title": entry.get("title", ""),
                                "contains": entry.get("contains", ""),
                                "why": entry.get("why", ""),
                                "summary": entry.get("summary", ""),
                                "source_domain": entry.get("source_domain", ""),
                                "url": entry.get("url", ""),
                                "created_at": entry.get("created_at", ""),
                                "tokens": entry.get("content_tokens", 0),
                                "chars": entry.get("content_chars", 0),
                                "lines": entry.get("content_lines", 0),
                                "file_line_number": line_num,
                            }
                    except json.JSONDecodeError:
                        continue
            
            self._save_large_notes_index()
            print(f"✅ Rebuilt large notes index: {len(self.large_notes_index)} notes")
        except Exception as e:
            print(f"⚠️  Index rebuild failed: {e}")

    def _save_large_notes_index(self):
        """Save index to disk."""
        try:
            with open(self.large_notes_index_path, "w", encoding="utf-8") as f:
                json.dump(self.large_notes_index, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️  Failed to save large notes index: {e}")

    def _load_large_note_entries(self) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        if not self.large_notes_path.exists():
            return entries
        try:
            with open(self.large_notes_path, "r", encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        # Skip malformed lines; keep store resilient
                        continue
        except Exception:
            return []
        return entries

    def _get_large_note_by_id(self, note_id: str) -> Optional[Dict[str, Any]]:
        """Get large note by ID using index for fast O(1) lookup."""
        # Try index-based retrieval first (fast path)
        if note_id in self.large_notes_index:
            try:
                metadata = self.large_notes_index[note_id]
                line_number = metadata["file_line_number"]
                
                # Read specific line from JSONL (fast seeking)
                with open(self.large_notes_path, "r", encoding="utf-8") as f:
                    for current_line_num, raw_line in enumerate(f, start=1):
                        if current_line_num == line_number:
                            try:
                                return json.loads(raw_line.strip())
                            except json.JSONDecodeError:
                                print(f"⚠️  Corrupted JSONL line {line_number} for note {note_id}")
                                break
            except Exception as e:
                print(f"⚠️  Index-based retrieval failed for {note_id}: {e}. Falling back to scan.")
        
        # Fallback: Scan JSONL if index lookup failed
        entries = self._load_large_note_entries()
        for entry in reversed(entries):
            if str(entry.get("id", "")).strip() == str(note_id).strip():
                return entry
        return None


    def _init_token_encoder(self):
        if not tiktoken:
            return None
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None

    def _count_tokens(self, text: str) -> int:
        payload = str(text or "")
        if not payload:
            return 0
        if self._token_encoder is not None:
            try:
                return len(self._token_encoder.encode(payload))
            except Exception:
                pass
        # Fallback heuristic if tokenizer is unavailable
        return max(1, len(payload) // 4)

    def _notes_total_tokens(self) -> int:
        return self._count_tokens("\n".join(self.notes))

    def _enforce_notes_token_budget(self, protected_note: Optional[str] = None) -> Dict[str, Any]:
        removed: List[str] = []
        while self.notes and self._notes_total_tokens() > self.notes_token_budget:
            remove_idx = None

            # Prefer removing oldest non-pointer note first.
            for i, n in enumerate(self.notes):
                if protected_note is not None and n == protected_note:
                    continue
                if not str(n).startswith("[LargeNote:"):
                    remove_idx = i
                    break

            # Otherwise remove the oldest removable note.
            if remove_idx is None:
                for i, n in enumerate(self.notes):
                    if protected_note is not None and n == protected_note:
                        continue
                    remove_idx = i
                    break

            # If only protected notes remain, stop to avoid deleting required pointers.
            if remove_idx is None:
                break

            removed_note = self.notes.pop(remove_idx)
            removed.append(self._clip_single_line(removed_note, 80))

        return {
            "removed_count": len(removed),
            "removed_preview": removed[:3],
            "notes_total_tokens": self._notes_total_tokens(),
        }

    @staticmethod
    def _clip_single_line(text: str, max_len: int) -> str:
        clean = re.sub(r"\s+", " ", str(text or "")).strip()
        if len(clean) <= max_len:
            return clean
        return clean[: max(1, max_len - 3)] + "..."

    def _current_source_info(self) -> Tuple[str, str]:
        current_url = self.page.url if self.page else ""
        domain = ""
        if current_url:
            try:
                domain = urlparse(current_url).netloc
            except Exception:
                domain = ""
        return current_url, domain

    def _format_large_note_pointer(
        self,
        note_id: str,
        contains: str,
        source_domain: str,
        why: str,
        summary: str,
    ) -> str:
        contains_short = self._clip_single_line(contains, 44) or "N/A"
        source_short = self._clip_single_line(source_domain, 36) or "unknown"
        why_short = self._clip_single_line(why, 52) or "offloaded large extract"
        summary_short = self._clip_single_line(summary, 80) or "N/A"
        return (
            f"[LargeNote:{note_id}] contains={contains_short}; "
            f"source={source_short}; why={why_short}; summary={summary_short}"
        )

    def _large_note_metadata(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        content = str(entry.get("content", ""))
        return {
            "id": entry.get("id"),
            "title": entry.get("title"),
            "contains": entry.get("contains"),
            "why": entry.get("why"),
            "summary": entry.get("summary"),
            "url": entry.get("url"),
            "source_domain": entry.get("source_domain"),
            "content_chars": entry.get("content_chars", len(content)),
            "content_lines": entry.get("content_lines", content.count("\n") + 1 if content else 0),
            "content_tokens": entry.get("content_tokens", self._count_tokens(content)),
            "created_at": entry.get("created_at"),
        }
    
    def _register_dialog_handler(self, page: Page):
        """Register a dialog listener on a page to auto-accept native JS dialogs.

        Handles alert(), confirm(), prompt(), and beforeunload dialogs.
        Records each event in self._pending_dialogs so the agent is informed.
        """
        async def _on_dialog(dialog):
            try:
                self._pending_dialogs.append({
                    "type": dialog.type,
                    "message": dialog.message,
                })
                await dialog.accept()
            except Exception:
                # Never let dialog handling crash the agent
                pass

        page.on("dialog", _on_dialog)

    async def start(self, initial_url: Optional[str] = None):
        """Start browser instance."""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=self.headless,
            args=[
                "--disable-background-networking",
                "--disable-background-timer-throttling",
                "--disable-backgrounding-occluded-windows",
                "--disable-breakpad",
                "--disable-component-update",
                "--disable-default-apps",
                "--disable-dev-shm-usage",
                "--disable-extensions",
                "--disable-features=TranslateUI",
                "--disable-hang-monitor",
                "--disable-ipc-flooding-protection",
                "--disable-popup-blocking",
                "--disable-prompt-on-repost",
                "--disable-renderer-backgrounding",
                "--disable-sync",
                "--force-color-profile=srgb",
                "--metrics-recording-only",
                "--no-first-run",
                "--password-store=basic",
                "--use-mock-keychain",
                # STEALTH ARGS
                "--disable-blink-features=AutomationControlled",
            ]
        )
        self.context = await self.browser.new_context(
            viewport={"width": self.config.screenshot_max_width, "height": 720},
            locale="en-US",
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
        )
        
        # STEALTH: HIDE WEBDRIVER PROPERTY
        await self.context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
        """)
        
        # Create first page
        self.page = await self.context.new_page()
        self._register_dialog_handler(self.page)
        self.pages = [self.page]
        self.active_tab_index = 0
        self.page.set_default_timeout(10000)
        
        if initial_url:
            await self.page.goto(initial_url, wait_until="domcontentloaded")
    
    async def capture_state(self, screenshot_name: str) -> Tuple[str, str, BrowserState]:
        """Capture current browser state."""
        # Ensure we use the active page
        self.page = self.pages[self.active_tab_index]
        await self.page.bring_to_front()
        
        screenshot_path = self.working_dir / "screenshots" / f"{screenshot_name}.webp"
        await self.page.screenshot(path=screenshot_path, type="jpeg", quality=self.config.screenshot_quality)
        
        with Image.open(screenshot_path) as img:
            img_hash = str(imagehash.average_hash(img))
        
        viewport = self.page.viewport_size
        
        # Collect info for all tabs
        tabs_info = []
        for i, p in enumerate(self.pages):
            try:
                # Need to run basic js to get safe title/url if they failed to load
                p_url = p.url
                p_title = await p.title()
                tabs_info.append(TabInfo(page_id=i, url=p_url, title=p_title))
            except Exception:
                tabs_info.append(TabInfo(page_id=i, url="unknown", title="Error retreiving tab info"))

        # Add tab info to title
        # Add tab info to title
        raw_title = await self.page.title()
        
        # ZOMBIE TAB CLEANUP: Check for inactive about:blank tabs and close them
        # This prevents clutter from popups or empty target=_blank pages
        if len(self.pages) > 1:
            for i in range(len(self.pages) - 1, -1, -1): # Iterate backwards safe for removal
                # Skip current active tab
                if i == self.active_tab_index:
                    continue
                    
                p = self.pages[i]
                try:
                    if p.url == "about:blank":
                        print(f"   (Auto-closing inactive zombie tab {i}: about:blank)")
                        await p.close()
                        self.pages.pop(i)
                        # Adjust active index if we removed a tab before it
                        if i < self.active_tab_index:
                            self.active_tab_index -= 1
                except Exception:
                    pass

        tab_info_str = f"[Tab {self.active_tab_index + 1}/{len(self.pages)}]"
        full_title = f"{tab_info_str} {raw_title}"

        # Drain any auto-handled dialog events since last capture
        recent_dialogs = list(self._pending_dialogs)
        self._pending_dialogs.clear()

        state = BrowserState(
            url=self.page.url,
            title=full_title,
            viewport_width=viewport["width"],
            viewport_height=viewport["height"],
            scroll_y=await self.page.evaluate("window.scrollY"),
            screenshot_hash=img_hash,
            timestamp=datetime.now(),
            ready_state=await self.page.evaluate("document.readyState"),
            notes = list(self.notes),  # Shallow copy — prevents DeleteNote/EditNote from mutating historical states
            tabs = tabs_info,
            dialogs = recent_dialogs,
        )
        return str(screenshot_path), img_hash, state
    
    async def execute_tool(self, tool_call: ToolCall, screenshot_path: str) -> ActionResult:
        """Execute a tool call (atomic action)."""
        start_time = time.time()
        self.total_actions += 1
        
        # Sync self.page with active tab
        if self.pages:
            self.page = self.pages[self.active_tab_index]
        
        try:
            if tool_call.action_type == ActionType.VISUAL_CLICK:
                result = await self._visual_click(screenshot_path, tool_call.parameters["description"], tool_call.parameters.get("region_hint"))
            elif tool_call.action_type == ActionType.VISUAL_TYPE:
                result = await self._visual_type(
                    screenshot_path,
                    tool_call.parameters["field_description"],
                    tool_call.parameters["text"],
                    tool_call.parameters.get("press_enter", False),
                )
            elif tool_call.action_type == ActionType.VISUAL_SCROLL:
                result = await self._visual_scroll(tool_call.parameters["direction"], tool_call.parameters.get("amount", 500))
            elif tool_call.action_type == ActionType.DOM_CLICK:
                result = await self._dom_click(tool_call.parameters["selector"])
            elif tool_call.action_type == ActionType.DOM_EXTRACT:
                result = await self._dom_extract(tool_call.parameters["query"], tool_call.parameters.get("schema"), tool_call.parameters.get("max_results", 10))
            elif tool_call.action_type == ActionType.NAVIGATE:
                result = await self._navigate(tool_call.parameters["url"])
            elif tool_call.action_type == ActionType.GO_BACK:
                result = await self._go_back()
            elif tool_call.action_type == ActionType.GO_FORWARD:
                result = await self._go_forward()
            elif tool_call.action_type == ActionType.RELOAD:
                result = await self._reload()
            elif tool_call.action_type == ActionType.TIMED_WAIT:
                if os.getenv("TIMED_WAIT_ENABLED", "True").lower() == "true":
                    result = await self._wait(tool_call.parameters.get("seconds", 1))
                else:
                    result = ActionResult(success=False, action_type=ActionType.TIMED_WAIT, description="TimedWait disabled by config", error="Tool disabled")
            elif tool_call.action_type == ActionType.VISUAL_WAIT:
                if os.getenv("VISUAL_WAIT_ENABLED", "True").lower() == "true":
                    result = await self._visual_wait(tool_call.parameters.get("timeout", int(os.getenv("VISUAL_WAIT_TIMEOUT", "30"))))
                else:
                    result = ActionResult(success=False, action_type=ActionType.VISUAL_WAIT, description="VisualWait disabled by config", error="Tool disabled")
            elif tool_call.action_type == ActionType.PRESS_KEY:
                result = await self._press_key(tool_call.parameters["key"])
            elif tool_call.action_type == ActionType.SCREENSHOT:
                result = await self._screenshot(tool_call.parameters.get("name"))
            elif tool_call.action_type == ActionType.NEW_TAB:
                result = await self._new_tab(tool_call.parameters["url"])
            elif tool_call.action_type == ActionType.SWITCH_TAB:
                result = await self._switch_tab(tool_call.parameters["index"])
            elif tool_call.action_type == ActionType.CLOSE_TAB:
                result = await self._close_tab(tool_call.parameters.get("index"))
            elif tool_call.action_type == ActionType.VISUAL_HOVER:
                result = await self._visual_hover(screenshot_path, tool_call.parameters["description"], tool_call.parameters.get("region_hint"))
            elif tool_call.action_type == ActionType.SAVE_NOTE:
                result = await self._save_note(tool_call.parameters["note"])
            elif tool_call.action_type == ActionType.SAVE_LARGE_NOTE:
                result = await self._save_large_note(
                    content=tool_call.parameters["content"],
                    title=tool_call.parameters.get("title"),
                    summary=tool_call.parameters.get("summary"),
                    contains=tool_call.parameters.get("contains"),
                    why=tool_call.parameters.get("why"),
                )
            elif tool_call.action_type == ActionType.READ_LARGE_NOTE:
                full_param = tool_call.parameters.get("full", False)
                is_full = full_param if isinstance(full_param, bool) else str(full_param).strip().lower() in {"1", "true", "yes", "y"}
                result = await self._read_large_note(
                    note_id=tool_call.parameters.get("note_id"),
                    start_line=tool_call.parameters.get("start_line"),
                    end_line=tool_call.parameters.get("end_line"),
                    full=is_full,
                )
            elif tool_call.action_type == ActionType.LIST_LARGE_NOTES:
                result = await self._list_large_notes(
                    limit=tool_call.parameters.get("limit", self.large_notes_default_list_limit),
                    newest_first=tool_call.parameters.get("newest_first", True),
                )
            elif tool_call.action_type == ActionType.SEARCH_LARGE_NOTES:
                result = await self._search_large_notes(
                    query=tool_call.parameters["query"],
                    limit=tool_call.parameters.get("limit", self.large_notes_default_search_limit),
                )
            elif tool_call.action_type == ActionType.DELETE_NOTE:
                result = await self._delete_note(tool_call.parameters["index"])
            elif tool_call.action_type == ActionType.EDIT_NOTE:
                result = await self._edit_note(tool_call.parameters["index"], tool_call.parameters["new_note"])
            elif tool_call.action_type == ActionType.ASK_USER:
                result = await self._ask_user(tool_call.parameters["question"])
            else:
                result = ActionResult(success=False, action_type=tool_call.action_type, description="Unknown action", error=f"Unsupported: {tool_call.action_type}")
            
            result.execution_time_ms = (time.time() - start_time) * 1000
            return result
        except Exception as e:
            return ActionResult(success=False, action_type=tool_call.action_type, description=str(tool_call.parameters), error=str(e), execution_time_ms=(time.time() - start_time) * 1000)

    # --- Tab Actions ---
    async def _new_tab(self, url: str) -> ActionResult:
        # Enforce tab limit from config
        if len(self.pages) >= self.config.max_tabs:
             return ActionResult(
                 success=False,
                 action_type=ActionType.NEW_TAB,
                 description=f"Open new tab: {url}",
                 error=f"TAB LIMIT REACHED. You have {len(self.pages)} open tabs (limit is {self.config.max_tabs}). You must use CloseTab(index) to free up space before opening a new one."
             )

        # Try to clean up initial blank tab if we are opening a real one
        if len(self.pages) == 1 and self.pages[0].url == "about:blank":
            try:
                await self.pages[0].close()
                self.pages.pop(0)
            except Exception:
                pass

        try:
            new_page = await self.context.new_page()
            self._register_dialog_handler(new_page)
            await new_page.goto(url, wait_until="domcontentloaded")
            self.pages.append(new_page)
            self.active_tab_index = len(self.pages) - 1
            self.page = new_page
            await self.page.bring_to_front()
            return ActionResult(success=True, action_type=ActionType.NEW_TAB, description=f"Opened new tab: {url}")
        except Exception as e: return ActionResult(success=False, action_type=ActionType.NEW_TAB, description=f"Open tab {url}", error=str(e))

    async def _switch_tab(self, index: int) -> ActionResult:
        try:
            if 0 <= index < len(self.pages):
                self.active_tab_index = index
                self.page = self.pages[index]
                await self.page.bring_to_front()
                return ActionResult(success=True, action_type=ActionType.SWITCH_TAB, description=f"Switched to tab {index}")
            else:
                return ActionResult(success=False, action_type=ActionType.SWITCH_TAB, description=f"Switch to tab {index}", error="Invalid tab index")
        except Exception as e: return ActionResult(success=False, action_type=ActionType.SWITCH_TAB, description=f"Switch to tab {index}", error=str(e))

    async def _close_tab(self, index: Optional[int] = None) -> ActionResult:
        try:
            target_index = index if index is not None else self.active_tab_index
            
            if 0 <= target_index < len(self.pages):
                page_to_close = self.pages[target_index]
                await page_to_close.close()
                self.pages.pop(target_index)
                
                # Adjust active index if needed
                if not self.pages:
                    # No pages left, open a blank one
                    self.page = await self.context.new_page()
                    self.pages = [self.page]
                    self.active_tab_index = 0
                elif target_index <= self.active_tab_index:
                    # If we closed current or previous tab, shift left
                    self.active_tab_index = max(0, self.active_tab_index - 1)
                    self.page = self.pages[self.active_tab_index]
                
                await self.page.bring_to_front()
                return ActionResult(success=True, action_type=ActionType.CLOSE_TAB, description=f"Closed tab {target_index}")
            else:
                return ActionResult(success=False, action_type=ActionType.CLOSE_TAB, description=f"Close tab {target_index}", error="Invalid tab index")
        except Exception as e: return ActionResult(success=False, action_type=ActionType.CLOSE_TAB, description=f"Close tab {index}", error=str(e))
        
    async def _save_note(
        self,
        note: str,
        bypass_policy: bool = False,
        rerouted_from: Optional[str] = None,
    ) -> ActionResult:
        """Save a persistent note. Auto-reroutes to large-note storage if policy requires."""
        try:
            note_text = str(note or "").strip()
            if not note_text:
                return ActionResult(
                    success=False,
                    action_type=ActionType.SAVE_NOTE,
                    description="Save note",
                    error="note must not be empty.",
                )

            note_tokens = self._count_tokens(note_text)
            current_total = self._notes_total_tokens()
            projected_total = current_total + note_tokens

            if not bypass_policy:
                reroute_reason = None
                if note_tokens >= self.large_note_min_tokens:
                    reroute_reason = (
                        f"SaveNote policy reroute: note has {note_tokens} tokens (>= {self.large_note_min_tokens} threshold)."
                    )
                elif projected_total > self.notes_token_budget:
                    reroute_reason = (
                        f"SaveNote policy reroute: note budget exceeded ({projected_total}>{self.notes_token_budget} tokens total)."
                    )

                if reroute_reason:
                    contains = self._clip_single_line(note_text, 64)
                    summary = self._clip_single_line(note_text, 220)
                    rerouted = await self._save_large_note(
                        content=note_text,
                        title="Auto-offloaded note",
                        summary=summary,
                        contains=contains,
                        why=reroute_reason,
                        bypass_policy=True,
                        rerouted_from="SaveNote",
                    )
                    if rerouted.success:
                        rerouted.description = f"SaveNote rerouted to large note: {rerouted.description}"
                    return rerouted

            self.notes.append(note_text)
            budget_info = self._enforce_notes_token_budget()
            output = json.dumps(
                {
                    "note_tokens": note_tokens,
                    "notes_total_tokens": budget_info["notes_total_tokens"],
                    "notes_token_budget": self.notes_token_budget,
                    "notes_pruned_count": budget_info["removed_count"],
                    "notes_pruned_preview": budget_info["removed_preview"],
                    "rerouted_from": rerouted_from,
                },
                ensure_ascii=False,
            )
            return ActionResult(
                success=True,
                action_type=ActionType.SAVE_NOTE,
                description=f"Saved note ({note_tokens} tokens)",
                output=output,
            )
        except Exception as e:
            return ActionResult(success=False, action_type=ActionType.SAVE_NOTE, description="Save note", error=str(e))

    async def _save_large_note(
        self,
        content: str,
        title: Optional[str] = None,
        summary: Optional[str] = None,
        contains: Optional[str] = None,
        why: Optional[str] = None,
        bypass_policy: bool = False,
        rerouted_from: Optional[str] = None,
    ) -> ActionResult:
        """Persist large extracts to external storage and add a compact pointer note."""
        try:
            text = str(content or "").strip()
            if not text:
                return ActionResult(
                    success=False,
                    action_type=ActionType.SAVE_LARGE_NOTE,
                    description="Save large note",
                    error="content must not be empty.",
                )

            content_tokens = self._count_tokens(text)
            if not bypass_policy and content_tokens < self.large_note_min_tokens:
                reroute_reason = (
                    f"SaveLargeNote policy reroute: content has {content_tokens} tokens (< {self.large_note_min_tokens} threshold)."
                )
                rerouted = await self._save_note(
                    note=text,
                    bypass_policy=False,
                    rerouted_from="SaveLargeNote",
                )
                if rerouted.success:
                    rerouted.description = f"SaveLargeNote rerouted to note: {rerouted.description} ({reroute_reason})"
                return rerouted

            title_text = str(title).strip() if title is not None else "Large Extract"
            if not title_text:
                title_text = "Large Extract"

            summary_text = str(summary).strip() if summary is not None else ""
            if not summary_text:
                summary_text = self._clip_single_line(text, 220)

            contains_text = str(contains).strip() if contains is not None else ""
            if not contains_text:
                contains_text = title_text

            why_text = str(why).strip() if why is not None else ""
            if not why_text:
                why_text = "Large extract offloaded to external notes store."

            current_url, source_domain = self._current_source_info()
            note_id = self._next_large_note_id()
            line_count = text.count("\n") + 1

            entry = {
                "id": note_id,
                "title": title_text,
                "contains": contains_text,
                "why": why_text,
                "summary": summary_text,
                "content": text,
                "content_chars": len(text),
                "content_lines": line_count,
                "content_tokens": content_tokens,
                "url": current_url,
                "source_domain": source_domain,
                "created_at": datetime.now().isoformat(),
                "rerouted_from": rerouted_from,
            }

            with open(self.large_notes_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            # Update index with line number for fast retrieval
            file_line_number = len(self.large_notes_index) + 1
            self.large_notes_index[note_id] = {
                "id": note_id,
                "title": title_text,
                "contains": contains_text,
                "why": why_text,
                "summary": summary_text,
                "source_domain": source_domain,
                "url": current_url,
                "created_at": entry["created_at"],
                "tokens": content_tokens,
                "chars": len(text),
                "lines": line_count,
                "file_line_number": file_line_number,
            }
            self._save_large_notes_index()

            pointer = self._format_large_note_pointer(
                note_id=note_id,
                contains=contains_text,
                source_domain=source_domain,
                why=why_text,
                summary=summary_text,
            )
            self.notes.append(pointer)
            budget_info = self._enforce_notes_token_budget(protected_note=pointer)
            pointer_note_index = None
            for idx, n in enumerate(self.notes, start=1):
                if n == pointer:
                    pointer_note_index = idx
                    break

            output = json.dumps(
                {
                    "note_id": note_id,
                    "path": str(self.large_notes_path),
                    "content_chars": len(text),
                    "content_lines": line_count,
                    "content_tokens": content_tokens,
                    "pointer_note_index": pointer_note_index,
                    "pointer": pointer,
                    "notes_total_tokens": budget_info["notes_total_tokens"],
                    "notes_token_budget": self.notes_token_budget,
                    "notes_pruned_count": budget_info["removed_count"],
                    "notes_pruned_preview": budget_info["removed_preview"],
                    "rerouted_from": rerouted_from,
                },
                ensure_ascii=False,
            )
            return ActionResult(
                success=True,
                action_type=ActionType.SAVE_LARGE_NOTE,
                description=f"Saved large note {note_id}",
                output=output,
            )
        except Exception as e:
            return ActionResult(
                success=False,
                action_type=ActionType.SAVE_LARGE_NOTE,
                description="Save large note",
                error=str(e),
            )

    async def _read_large_note(
        self,
        note_id: Optional[str] = None,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        full: bool = False,
    ) -> ActionResult:
        """Read either a specific large note (recommended) or sections of the raw store file."""
        max_output_chars = 120000

        try:
            if note_id:
                entry = self._get_large_note_by_id(note_id)
                if not entry:
                    return ActionResult(
                        success=False,
                        action_type=ActionType.READ_LARGE_NOTE,
                        description=f"Read large note {note_id}",
                        error=f"Large note '{note_id}' not found.",
                    )

                content = str(entry.get("content", ""))
                lines = content.splitlines()
                total_lines = len(lines)

                if full:
                    s_line = 1
                    e_line = total_lines if total_lines > 0 else 1
                    selected = content
                else:
                    s_line = max(1, int(start_line) if start_line is not None else 1)
                    default_end = s_line + 199
                    e_line = int(end_line) if end_line is not None else default_end
                    if total_lines > 0:
                        e_line = min(total_lines, max(s_line, e_line))
                        selected = "\n".join(lines[s_line - 1:e_line])
                    else:
                        e_line = 1
                        selected = content

                truncated = False
                if len(selected) > max_output_chars:
                    selected = selected[:max_output_chars] + "\n... [truncated]"
                    truncated = True

                header = (
                    f"LARGE_NOTE id={entry.get('id', note_id)} "
                    f"title={entry.get('title', 'N/A')} "
                    f"contains={entry.get('contains', 'N/A')} "
                    f"source={entry.get('source_domain', 'unknown')} "
                    f"why={entry.get('why', 'N/A')} "
                    f"lines={s_line}-{e_line}/{max(total_lines, 1)} "
                    f"chars={len(content)} tokens={entry.get('content_tokens', self._count_tokens(content))} "
                    f"path={self.large_notes_path}"
                )
                output = f"{header}\n\n{selected}"
                if truncated:
                    output += f"\n\n[Output truncated to {max_output_chars} chars]"

                return ActionResult(
                    success=True,
                    action_type=ActionType.READ_LARGE_NOTE,
                    description=f"Read large note {entry.get('id', note_id)}",
                    output=output,
                )

            # File mode: read raw large-notes file by line range or full file.
            if not self.large_notes_path.exists():
                return ActionResult(
                    success=False,
                    action_type=ActionType.READ_LARGE_NOTE,
                    description="Read large notes file",
                    error=f"Large notes file not found: {self.large_notes_path}",
                )

            with open(self.large_notes_path, "r", encoding="utf-8") as f:
                file_lines = f.readlines()

            total_file_lines = len(file_lines)
            if full:
                s_line = 1
                e_line = total_file_lines if total_file_lines > 0 else 1
            else:
                s_line = max(1, int(start_line) if start_line is not None else 1)
                default_end = s_line + 199
                e_line = int(end_line) if end_line is not None else default_end
                if total_file_lines > 0:
                    e_line = min(total_file_lines, max(s_line, e_line))
                else:
                    e_line = 1

            if total_file_lines > 0:
                selected = "".join(file_lines[s_line - 1:e_line])
            else:
                selected = ""

            truncated = False
            if len(selected) > max_output_chars:
                selected = selected[:max_output_chars] + "\n... [truncated]"
                truncated = True

            header = (
                f"LARGE_NOTES_FILE path={self.large_notes_path} "
                f"lines={s_line}-{e_line}/{max(total_file_lines, 1)}"
            )
            output = f"{header}\n\n{selected}"
            if truncated:
                output += f"\n\n[Output truncated to {max_output_chars} chars]"

            return ActionResult(
                success=True,
                action_type=ActionType.READ_LARGE_NOTE,
                description=f"Read large notes file lines {s_line}-{e_line}",
                output=output,
            )
        except Exception as e:
            return ActionResult(
                success=False,
                action_type=ActionType.READ_LARGE_NOTE,
                description="Read large note",
                error=str(e),
            )

    async def _list_large_notes(self, limit: Any = 20, newest_first: Any = True) -> ActionResult:
        """List metadata for large notes using index (fast)."""
        try:
            safe_limit = max(1, min(200, int(limit)))
            if isinstance(newest_first, bool):
                is_newest_first = newest_first
            else:
                is_newest_first = str(newest_first).strip().lower() in {"1", "true", "yes", "y"}

            # Use index instead of scanning JSONL
            notes_list = list(self.large_notes_index.values())
            if is_newest_first:
                notes_list = list(reversed(notes_list))
            
            selected = notes_list[:safe_limit]

            output = json.dumps(
                {
                    "path": str(self.large_notes_path),
                    "index_path": str(self.large_notes_index_path),
                    "total_notes": len(self.large_notes_index),
                    "returned": len(selected),
                    "newest_first": is_newest_first,
                    "notes": selected,
                },
                ensure_ascii=False,
                indent=2,
            )
            return ActionResult(
                success=True,
                action_type=ActionType.LIST_LARGE_NOTES,
                description=f"Listed {len(selected)} large notes",
                output=output,
            )
        except Exception as e:
            return ActionResult(
                success=False,
                action_type=ActionType.LIST_LARGE_NOTES,
                description="List large notes",
                error=f"Failed to list large notes from index: {str(e)}",
            )


    async def _search_large_notes(self, query: str, limit: Any = 10) -> ActionResult:
        """Search large notes using index for metadata, with fallback to content search."""
        try:
            q = str(query or "").strip()
            if not q:
                return ActionResult(
                    success=False,
                    action_type=ActionType.SEARCH_LARGE_NOTES,
                    description="Search large notes",
                    error="query must not be empty.",
                )

            safe_limit = max(1, min(100, int(limit)))
            q_lower = q.lower()

            # Phase 1: Search index metadata (fast)
            matches = []
            for note_id, metadata in self.large_notes_index.items():
                match_score = 0
                if q_lower in metadata.get("title", "").lower():
                    match_score += 10
                if q_lower in metadata.get("contains", "").lower():
                    match_score += 8
                if q_lower in metadata.get("summary", "").lower():
                    match_score += 5
                if q_lower in metadata.get("source_domain", "").lower():
                    match_score += 3
                if q_lower in metadata.get("why", "").lower():
                    match_score += 2
                
                if match_score > 0:
                    matches.append((match_score, metadata))

            # Sort by relevance
            matches.sort(reverse=True, key=lambda x: x[0])
            results = [m[1] for m in matches[:safe_limit]]

            # Phase 2: If metadata search yields few results, search content
            if len(results) < safe_limit // 2:
                print(f"   Metadata search found {len(results)} matches, searching content...")
                entries = self._load_large_note_entries()
                for entry in entries:
                    if len(results) >= safe_limit:
                        break
                    note_id = entry.get("id")
                    # Skip if already matched
                    if any(r["id"] == note_id for r in results):
                        continue
                        
                    if q_lower in str(entry.get("content", "")).lower():
                        if note_id in self.large_notes_index:
                            results.append(self.large_notes_index[note_id])

            output = json.dumps(
                {
                    "path": str(self.large_notes_path),
                    "query": q,
                    "total_matches": len(results),
                    "returned": len(results),
                    "notes": results,
                },
                ensure_ascii=False,
                indent=2,
            )
            return ActionResult(
                success=True,
                action_type=ActionType.SEARCH_LARGE_NOTES,
                description=f"Found {len(results)} matching notes",
                output=output,
            )
        except Exception as e:
            return ActionResult(
                success=False,
                action_type=ActionType.SEARCH_LARGE_NOTES,
                description="Search large notes",
                error=f"Search failed: {str(e)}",
            )


    async def _delete_note(self, index: int) -> ActionResult:
        """Delete a note by 1-based index."""
        try:
            idx = int(index) - 1  # Convert to 0-based
            if idx < 0 or idx >= len(self.notes):
                return ActionResult(success=False, action_type=ActionType.DELETE_NOTE, description=f"Delete note {index}", error=f"Invalid index {index}. You have {len(self.notes)} notes (1-{len(self.notes)}).")
            removed = self.notes.pop(idx)
            return ActionResult(success=True, action_type=ActionType.DELETE_NOTE, description=f"Deleted note {index}: {removed}")
        except Exception as e:
            return ActionResult(success=False, action_type=ActionType.DELETE_NOTE, description=f"Delete note {index}", error=str(e))

    async def _edit_note(self, index: int, new_note: str) -> ActionResult:
        """Edit a note by 1-based index with new content."""
        try:
            if not new_note or not str(new_note).strip():
                return ActionResult(success=False, action_type=ActionType.EDIT_NOTE, description=f"Edit note {index}", error="new_note must not be empty.")
            idx = int(index) - 1  # Convert to 0-based
            if idx < 0 or idx >= len(self.notes):
                return ActionResult(success=False, action_type=ActionType.EDIT_NOTE, description=f"Edit note {index}", error=f"Invalid index {index}. You have {len(self.notes)} notes (1-{len(self.notes)}).")
            old_note = self.notes[idx]
            self.notes[idx] = str(new_note).strip()
            return ActionResult(success=True, action_type=ActionType.EDIT_NOTE, description=f"Edited note {index}: '{old_note}' -> '{new_note}'")
        except Exception as e:
            return ActionResult(success=False, action_type=ActionType.EDIT_NOTE, description=f"Edit note {index}", error=str(e))

    async def _ask_user(self, question: str) -> ActionResult:
        """Ask the user a question via CLI input.

        Handles:
        - Headless mode: returns graceful error (no terminal available)
        - Timeout: configurable via config.ask_user_timeout (default 120s)
        - Non-blocking: runs input() in executor to avoid blocking event loop
        """
        # In headless mode, no interactive terminal is available
        if self.headless:
            return ActionResult(
                success=False,
                action_type=ActionType.ASK_USER,
                description=f"Asked: {question}",
                error="Cannot ask user in headless mode. No interactive terminal available. Try a different approach.",
            )

        try:
            print(f"\n❓ [Agent Asks]: {question}")
            loop = asyncio.get_running_loop()
            try:
                response = await asyncio.wait_for(
                    loop.run_in_executor(None, input, "> "),
                    timeout=self.config.ask_user_timeout,
                )
            except asyncio.TimeoutError:
                print(f"\n⏰ [AskUser] No response after {self.config.ask_user_timeout}s - moving on.")
                return ActionResult(
                    success=False,
                    action_type=ActionType.ASK_USER,
                    description=f"Asked: {question}",
                    error=f"User did not respond within {self.config.ask_user_timeout}s timeout.",
                )

            return ActionResult(
                success=True,
                action_type=ActionType.ASK_USER,
                description=f"Asked: {question}",
                output=f"User Answer: {response.strip() if response.strip() else '(empty response)'}",
            )
        except EOFError:
            return ActionResult(
                success=False,
                action_type=ActionType.ASK_USER,
                description=f"Asked: {question}",
                error="No interactive terminal available (stdin closed).",
            )
        except Exception as e:
            return ActionResult(success=False, action_type=ActionType.ASK_USER, description=f"Ask user: {question}", error=str(e))

    # --- Internal Actions ---
    async def _visual_click(self, screenshot_path: str, description: str, region_hint: Optional[str] = None) -> ActionResult:
        coords = await self._call_mimo_grounding(screenshot_path, description)
        if not coords:
            return ActionResult(success=False, action_type=ActionType.VISUAL_CLICK, description=description, error="MiMo failed to find element")
        x, y = coords
        await self.page.mouse.click(x, y)
        try: await self.page.wait_for_load_state("domcontentloaded", timeout=2000)
        except: pass
        return ActionResult(success=True, action_type=ActionType.VISUAL_CLICK, description=description, coordinates=(x, y))

    async def _visual_hover(self, screenshot_path: str, description: str, region_hint: Optional[str] = None) -> ActionResult:
        """Hover over an element without clicking (for dropdowns, tooltips, menus)."""
        coords = await self._call_mimo_grounding(screenshot_path, description)
        if not coords:
            return ActionResult(success=False, action_type=ActionType.VISUAL_HOVER, description=description, error="MiMo failed to find element")
        x, y = coords
        await self.page.mouse.move(x, y)
        await asyncio.sleep(0.3)  # Wait for hover effects to render
        return ActionResult(success=True, action_type=ActionType.VISUAL_HOVER, description=f"Hovered over: {description}", coordinates=(x, y))

    async def _visual_type(self, screenshot_path: str, field_description: str, text: str, press_enter: bool = False) -> ActionResult:
        coords = await self._call_mimo_grounding(screenshot_path, field_description)
        if not coords:
            return ActionResult(success=False, action_type=ActionType.VISUAL_TYPE, description=f"Type '{text}'", error="MiMo failed to find field")
        x, y = coords
        await self.page.mouse.click(x, y)
        await asyncio.sleep(0.2)
        await self.page.keyboard.press("Control+A")
        await self.page.keyboard.press("Backspace")
        await self.page.keyboard.type(text, delay=30)
        if press_enter:
            await self.page.keyboard.press("Enter")
            try: await self.page.wait_for_load_state("domcontentloaded", timeout=2000)
            except: pass
        return ActionResult(success=True, action_type=ActionType.VISUAL_TYPE, description=f"Typed '{text}'", coordinates=(x, y))

    async def _visual_scroll(self, direction: str, amount: Any) -> ActionResult:
        direction_lower = direction.lower()
        
        # 1. Only calculate pixels if needed (for relative scrolling)
        pixels = 0
        if direction_lower in ["up", "down"]:
            # Map string descriptions to pixel values
            scroll_map = {
                "small": 300,
                "medium": 600,
                "large": 1000,
                "page": 800
            }
            
            # Resolve amount
            pixels = 500  # Default
            if isinstance(amount, int):
                pixels = amount
            elif isinstance(amount, str) and amount.lower() in scroll_map:
                pixels = scroll_map[amount.lower()]
            elif isinstance(amount, str) and amount.isdigit():
                pixels = int(amount)
            
        # 2. Get initial scroll position
        start_y = await self.page.evaluate("window.scrollY")
            
        # 3. Try Standard Window Scroll
        if direction_lower == "top":
            await self.page.evaluate("window.scrollTo(0, 0)")
        elif direction_lower == "bottom":
            await self.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        elif direction_lower == "down":
            await self.page.evaluate(f"window.scrollBy(0, {pixels})")
        elif direction_lower == "up":
            await self.page.evaluate(f"window.scrollBy(0, -{pixels})")
            
        await asyncio.sleep(0.5)
        
        # 4. Check if scroll actually happened
        end_y = await self.page.evaluate("window.scrollY")
        
        # Smart Fallback: If window didn't move, try to find a scrollable container
        if start_y == end_y:
            # JS to find largest scrollable element
            fallback_js = """
            (args) => {
                const pixels = args[0];
                const direction = args[1];
                
                // Find potential scroll containers
                const elements = Array.from(document.querySelectorAll('*')).filter(el => {
                    const style = window.getComputedStyle(el);
                    const isScrollable = (el.scrollHeight > el.clientHeight) && 
                                         (style.overflowY === 'auto' || style.overflowY === 'scroll');
                    return isScrollable && (el.clientHeight > 50); 
                });
                
                if (elements.length === 0) return "no_containers";
                
                // Sort by area (approximation for 'main content')
                elements.sort((a, b) => (b.clientWidth * b.clientHeight) - (a.clientWidth * a.clientHeight));
                const target = elements[0];

                const startTop = target.scrollTop;
                
                // Handle absolute positioning (top/bottom) vs relative (up/down)
                if (direction === 'top') {
                    target.scrollTop = 0;
                } else if (direction === 'bottom') {
                    target.scrollTop = target.scrollHeight;
                } else if (direction === 'down') {
                    target.scrollBy(0, pixels);
                } else {
                    target.scrollBy(0, -pixels);
                }
                
                return target.scrollTop !== startTop ? "scrolled" : "at_limit";
            }
            """
            
            result = await self.page.evaluate(fallback_js, [pixels, direction_lower])
            
            if result == "scrolled":
                if direction_lower in ["top", "bottom"]:
                    return ActionResult(success=True, action_type=ActionType.VISUAL_SCROLL, 
                                      description=f"Scrolled container to {direction_lower}")
                else:
                    return ActionResult(success=True, action_type=ActionType.VISUAL_SCROLL, 
                                      description=f"Scrolled container {direction_lower} by {pixels}px")
            elif result == "at_limit":
                return ActionResult(success=True, action_type=ActionType.VISUAL_SCROLL, 
                                  description=f"Already at {direction_lower} (limit reached)")
            # If no containers or window didn't move, just report success (maybe at bottom)
            pass

        # 5. Build appropriate success message
        if direction_lower in ["top", "bottom"]:
            description = f"Scrolled to {direction_lower} of page"
        else:
            description = f"Scrolled {direction_lower} by {pixels}px"
            
        return ActionResult(success=True, action_type=ActionType.VISUAL_SCROLL, description=description)
    
    async def _dom_click(self, selector: str) -> ActionResult:
        self.dom_calls += 1
        try:
            await self.page.click(selector, timeout=5000)
            return ActionResult(success=True, action_type=ActionType.DOM_CLICK, description=f"Clicked {selector}")
        except Exception as e: return ActionResult(success=False, action_type=ActionType.DOM_CLICK, description=f"Click {selector}", error=str(e))

    async def _dom_extract(self, query: str, schema: Optional[Dict], max_results: int) -> ActionResult:
        self.dom_calls += 1
        try:
            # Execute extraction entirely in browser context for stability and speed
            # This avoids "Node is not an HTMLElement" errors and stale handles
            js_script = """
            (args) => {
                const query = args.query;
                const schema = args.schema;
                const max_results = args.max_results;
                
                const elements = Array.from(document.querySelectorAll(query)).slice(0, max_results);
                
                return elements.map(el => {
                    if (schema) {
                        const item = {};
                        for (const key in schema) {
                            const selector = schema[key];
                            const child = el.querySelector(selector);
                            item[key] = child ? child.innerText.trim() : null;
                        }
                        return item;
                    } else {
                        return el.innerText.trim();
                    }
                });
            }
            """
            
            data = await self.page.evaluate(js_script, {
                "query": query, 
                "schema": schema, 
                "max_results": max_results
            })
            
            # Limit output size to prevent context overflow
            import json
            output_str = json.dumps(data, indent=2)
            if len(output_str) > 100000:
                output_str = output_str[:100000] + "... (truncated)"
            
            return ActionResult(
                success=True, 
                action_type=ActionType.DOM_EXTRACT, 
                description=f"Extracted {len(data)} items",
                output=output_str
            )
        except Exception as e: return ActionResult(success=False, action_type=ActionType.DOM_EXTRACT, description=f"Extract {query}", error=str(e))

    async def _navigate(self, url: str) -> ActionResult:
        try:
            await self.page.goto(url, wait_until="domcontentloaded", timeout=15000)
            return ActionResult(success=True, action_type=ActionType.NAVIGATE, description=f"Navigated to {url}")
        except Exception as e: return ActionResult(success=False, action_type=ActionType.NAVIGATE, description=f"Navigate {url}", error=str(e))

    async def _go_back(self) -> ActionResult:
        try:
            response = await self.page.go_back(wait_until="domcontentloaded", timeout=15000)
            if response is None:
                return ActionResult(success=False, action_type=ActionType.GO_BACK, description="Go back", error="No previous page in history")
            return ActionResult(success=True, action_type=ActionType.GO_BACK, description=f"Went back to {self.page.url}")
        except Exception as e:
            return ActionResult(success=False, action_type=ActionType.GO_BACK, description="Go back", error=str(e))

    async def _go_forward(self) -> ActionResult:
        try:
            response = await self.page.go_forward(wait_until="domcontentloaded", timeout=15000)
            if response is None:
                return ActionResult(success=False, action_type=ActionType.GO_FORWARD, description="Go forward", error="No forward page in history")
            return ActionResult(success=True, action_type=ActionType.GO_FORWARD, description=f"Went forward to {self.page.url}")
        except Exception as e:
            return ActionResult(success=False, action_type=ActionType.GO_FORWARD, description="Go forward", error=str(e))

    async def _reload(self) -> ActionResult:
        try:
            await self.page.reload(wait_until="domcontentloaded", timeout=15000)
            return ActionResult(success=True, action_type=ActionType.RELOAD, description=f"Reloaded {self.page.url}")
        except Exception as e:
            return ActionResult(success=False, action_type=ActionType.RELOAD, description="Reload page", error=str(e))

    async def _wait(self, seconds: float) -> ActionResult:
        # Enforce config limits
        safe_seconds = max(float(os.getenv("WAIT_MIN_SECONDS", "0.5")), min(float(seconds), float(os.getenv("WAIT_MAX_SECONDS", "60"))))
        await asyncio.sleep(safe_seconds)
        return ActionResult(success=True, action_type=ActionType.TIMED_WAIT, description=f"Waited {safe_seconds}s")


    async def _visual_wait(self, timeout: int = 30) -> ActionResult:
        """Wait until screen content stabilizes (useful for streaming generation)."""
        start_time = time.time()
        timeout = min(timeout, 60) # Global cap hard safety
        
        last_hash = None
        stable_count = 0
        
        # Initial wait to let things start moving
        await asyncio.sleep(1.0) 
        
        while (time.time() - start_time) < timeout:
            # Quick screenshot for hashing (low quality fine for diff detection)
            # Use current page reference
            if self.pages: self.page = self.pages[self.active_tab_index]
            
            # We use a cheaper buffer-based screenshot for speed
            try:
                screenshot_bytes = await self.page.screenshot(type="jpeg", quality=40)
                with Image.open(BytesIO(screenshot_bytes)) as img:
                    current_hash = imagehash.average_hash(img)
            except Exception as e:
                # If screenshot fails, assume not stable or browser issue; wait and retry
                await asyncio.sleep(1)
                continue
                
            if last_hash and current_hash == last_hash:
                stable_count += 1
            else:
                stable_count = 0
                last_hash = current_hash
            
            if stable_count >= int(os.getenv("VISUAL_STABILITY_THRESHOLD", "3")):
                duration = time.time() - start_time
                return ActionResult(
                    success=True, 
                    action_type=ActionType.VISUAL_WAIT, 
                    description=f"Screen stabilized after {duration:.1f}s"
                )
            
            await asyncio.sleep(1.0)
            
        return ActionResult(
            success=True, 
            action_type=ActionType.VISUAL_WAIT, 
            description=f"Wait timed out after {timeout}s (Screen might still be moving)"
        )

    async def _press_key(self, key: str) -> ActionResult:
        await self.page.keyboard.press(key)
        return ActionResult(success=True, action_type=ActionType.PRESS_KEY, description=f"Pressed {key}")

    async def _screenshot(self, name: Optional[str] = None) -> ActionResult:
        """Capture an explicit screenshot and return the saved file path."""
        try:
            screenshots_dir = self.working_dir / "screenshots"
            screenshots_dir.mkdir(parents=True, exist_ok=True)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            if name:
                safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("._")
            else:
                safe_name = ""
            prefix = safe_name if safe_name else "manual"
            screenshot_path = screenshots_dir / f"{prefix}_{ts}.jpg"

            await self.page.screenshot(path=screenshot_path, type="jpeg", quality=self.config.screenshot_quality)
            return ActionResult(
                success=True,
                action_type=ActionType.SCREENSHOT,
                description=f"Saved screenshot: {screenshot_path.name}",
                output=str(screenshot_path),
            )
        except Exception as e:
            return ActionResult(success=False, action_type=ActionType.SCREENSHOT, description="Capture screenshot", error=str(e))

    async def _call_mimo_grounding(self, screenshot_path: str, instruction: str) -> Optional[Tuple[int, int]]:
        """Call MiMo-VL to find element coordinates with robust parsing."""
        self.mimo_calls += 1

        # --- Encode ---
        t0 = time.time()
        with Image.open(screenshot_path) as img:
            w, h = img.size
            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=85)
            img_b64 = base64.b64encode(buffered.getvalue()).decode()
        encode_ms = (time.time() - t0) * 1000

        # CORRECTED QUERY FORMAT matching find_coordinates_mimo.py
        query = f"Image size: {w}x{h} pixels\n\nFind the element: {instruction}\n\nOutput the center coordinates as [x, y] in pixels."
        
        payload = {
            "model": "XiaomiMiMo/MiMo-VL-7B-RL",
            "messages": [
                {
                    "role": "system", 
                    "content": "You are a GUI grounding assistant. Given a screenshot and an element description, output the pixel coordinates (x, y) of the center of that element.\n\nOutput format: Return ONLY the coordinates as [x, y] where x and y are pixel values.\nDo not include any other text, explanation, or formatting."
                },
                {
                    "role": "user", 
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}, 
                        {"type": "text", "text": query}
                    ]
                }
            ],
            "temperature": 0.3,   # Updated to 0.3
            "max_tokens": 1024,   # Updated to 1024 to allow for reasoning tags
        }
        
        try:
            headers = {}
            if getattr(self, 'mimo_api_key', None):
                headers["Authorization"] = f"Bearer {self.mimo_api_key}"

            # --- HTTP inference ---
            t1 = time.time()
            response = await self.http_client.post(self.mimo_chat_completions_url, json=payload, headers=headers)
            response.raise_for_status()
            infer_ms = (time.time() - t1) * 1000

            content = response.json()["choices"][0]["message"]["content"]
            
            # --- Parse ---
            t2 = time.time()
            try:
                coords = parse_coordinates(content, (w, h))
                parse_ms = (time.time() - t2) * 1000
                print(f"   ⏱️  MiMo latency: encode={encode_ms:.0f}ms | infer={infer_ms:.0f}ms | parse={parse_ms:.0f}ms | total={encode_ms+infer_ms+parse_ms:.0f}ms")
                return coords
            except ValueError:
                parse_ms = (time.time() - t2) * 1000
                print(f"⚠️ MiMo parse failed. Raw Output: '{content}'")
                print(f"   ⏱️  MiMo latency: encode={encode_ms:.0f}ms | infer={infer_ms:.0f}ms | parse={parse_ms:.0f}ms")
                return None
            
        except Exception as e:
            print(f"MiMo call failed: {e}")
            return None


    async def verify_action(self, before_state: BrowserState, after_state: BrowserState, verification_hint: Optional[str]) -> Tuple[VerificationStatus, float]:
        if before_state.screenshot_hash == after_state.screenshot_hash:
            return VerificationStatus.NO_CHANGE, 0.0
        return VerificationStatus.SUCCESS, 1.0

    async def close(self):
        if self.context: await self.context.close()
        if self.browser: await self.browser.close()
        if self.playwright: await self.playwright.stop()
        await self.http_client.aclose()

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_actions": self.total_actions,
            "mimo_calls": self.mimo_calls,
            "dom_calls": self.dom_calls,
            "large_notes_path": str(self.large_notes_path),
            "large_notes_count": self.large_note_count,
            "notes_token_budget": self.notes_token_budget,
            "notes_total_tokens": self._notes_total_tokens(),
        }
