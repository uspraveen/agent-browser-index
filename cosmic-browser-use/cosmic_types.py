#!/usr/bin/env python3
"""
Shared types and schemas for Cosmic Browser Use Agent
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from enum import Enum


class ActionType(str, Enum):
    """Available browser actions"""
    VISUAL_CLICK = "VisualClick"
    VISUAL_TYPE = "VisualType"
    VISUAL_SCROLL = "VisualScroll"
    VISUAL_HOVER = "VisualHover"
    DOM_CLICK = "DOMClick"
    DOM_EXTRACT = "DOMExtract"
    NAVIGATE = "Navigate"
    GO_BACK = "GoBack"
    GO_FORWARD = "GoForward"
    RELOAD = "Reload"
    TIMED_WAIT = "TimedWait"
    VISUAL_WAIT = "VisualWait"
    PRESS_KEY = "PressKey"
    SCREENSHOT = "Screenshot"
    NEW_TAB = "NewTab"
    SWITCH_TAB = "SwitchTab"
    CLOSE_TAB = "CloseTab"
    SAVE_NOTE = "SaveNote"
    SAVE_LARGE_NOTE = "SaveLargeNote"
    READ_LARGE_NOTE = "ReadLargeNote"
    LIST_LARGE_NOTES = "ListLargeNotes"
    SEARCH_LARGE_NOTES = "SearchLargeNotes"
    DELETE_NOTE = "DeleteNote"
    EDIT_NOTE = "EditNote"
    READ_HISTORY = "ReadHistory"
    ASK_USER = "AskUser"


class VerificationStatus(str, Enum):
    """Result of action verification"""
    SUCCESS = "success"
    NO_CHANGE = "no_change"
    WRONG_STATE = "wrong_state"
    INCOMPLETE = "incomplete"
    LOOP_DETECTED = "loop_detected"
    ERROR = "error"


class LLMProvider(str, Enum):
    """Supported LLM providers"""
    GEMINI = "gemini"
    CLAUDE = "claude"
    OPENAI = "openai"
    # Fireworks OpenAI-compatible API (e.g. Kimi K2.6 — accounts/fireworks/models/kimi-k2p6)
    FIREWORKS_KIMI = "fireworks_kimi"
    VLLM = "vllm"
    OLLAMA = "ollama"


class LLMTier(str, Enum):
    """LLM performance tiers"""
    FAST = "fast"      # 100-300ms, 70% of cases
    MEDIUM = "medium"  # 300-800ms, 25% of cases
    SLOW = "slow"      # 800ms+, 5% of cases


@dataclass
class TabInfo:
    """Information about an open tab"""
    page_id: int
    url: str
    title: str


@dataclass
class BrowserState:
    """Snapshot of browser state"""
    url: str
    title: str
    viewport_width: int
    viewport_height: int
    scroll_y: int
    screenshot_hash: str
    timestamp: datetime
    ready_state: str = "complete"
    notes: List[str] = field(default_factory=list)  # Persistent knowledge base
    tabs: List[TabInfo] = field(default_factory=list)  # List of all open tabs
    dialogs: List[Dict[str, str]] = field(default_factory=list)  # Auto-handled browser dialogs since last capture
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "viewport": {"width": self.viewport_width, "height": self.viewport_height},
            "scroll_y": self.scroll_y,
            "screenshot_hash": self.screenshot_hash,
            "timestamp": self.timestamp.isoformat(),
            "ready_state": self.ready_state,
            "notes": self.notes,
            "tabs": [
                {"page_id": t.page_id, "url": t.url, "title": t.title}
                for t in self.tabs
            ],
            "dialogs": self.dialogs
        }


@dataclass
class ActionResult:
    """Result of executing an action"""
    success: bool
    action_type: ActionType
    description: str
    coordinates: Optional[tuple[int, int]] = None
    output: Optional[str] = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    verification_status: Optional[VerificationStatus] = None
    state_change_score: float = 0.0  # 0-1, how much the page changed
    estimated_completion: float = 0.0  # 0-1, copied from LLM decision for progress tracking
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "action_type": self.action_type.value,
            "description": self.description,
            "coordinates": self.coordinates,
            "output": self.output,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "verification_status": self.verification_status.value if self.verification_status else None,
            "state_change_score": self.state_change_score,
            "estimated_completion": self.estimated_completion,
            "metadata": self.metadata,
        }


@dataclass
class Step:
    """A single step in the workflow"""
    step_number: int
    timestamp: datetime
    screenshot_path: str
    screenshot_hash: str
    browser_state: BrowserState
    action: Optional[ActionResult] = None
    summary: str = ""
    thinking: Optional[str] = None  # LLM reasoning
    before_browser_state: Optional[BrowserState] = None
    after_browser_state: Optional[BrowserState] = None
    after_screenshot_path: Optional[str] = None
    after_screenshot_hash: Optional[str] = None
    tool_call: Optional[Dict[str, Any]] = None
    llm_response: Optional[Dict[str, Any]] = None
    visual_index: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step_number,
            "timestamp": self.timestamp.isoformat(),
            "screenshot_path": self.screenshot_path,
            "screenshot_hash": self.screenshot_hash,
            "browser_state": self.browser_state.to_dict(),
            "action": self.action.to_dict() if self.action else None,
            "summary": self.summary,
            "thinking": self.thinking,
            "before_browser_state": self.before_browser_state.to_dict() if self.before_browser_state else None,
            "after_browser_state": self.after_browser_state.to_dict() if self.after_browser_state else None,
            "after_screenshot_path": self.after_screenshot_path,
            "after_screenshot_hash": self.after_screenshot_hash,
            "tool_call": self.tool_call,
            "llm_response": self.llm_response,
            "visual_index": self.visual_index,
        }


@dataclass
class ToolCall:
    """LLM tool call request"""
    action_type: ActionType
    parameters: Dict[str, Any]
    fallback: Optional[Dict[str, Any]] = None
    verification_hint: Optional[str] = None  # e.g., "url_contains('/checkout')"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ToolCall':
        return cls(
            action_type=ActionType(data["action_type"]),
            parameters=data.get("parameters", {}),
            fallback=data.get("fallback"),
            verification_hint=data.get("verification_hint"),
        )


@dataclass
class LLMResponse:
    """Structured LLM decision output"""
    tool_call: ToolCall
    reasoning: Optional[str] = None
    confidence: float = 1.0
    requires_escalation: bool = False  # True if uncertain, needs more powerful model
    estimated_completion: float = 0.0  # 0-1, how close to goal
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_call": {
                "action_type": self.tool_call.action_type.value,
                "parameters": self.tool_call.parameters,
                "fallback": self.tool_call.fallback,
                "verification_hint": self.tool_call.verification_hint,
            },
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "requires_escalation": self.requires_escalation,
            "estimated_completion": self.estimated_completion,
        }


@dataclass
class LLMConfig:
    """Configuration for a specific LLM"""
    provider: LLMProvider
    model_id: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    timeout_ms: int = 5000
    max_tokens: int = 2048
    temperature: float = 0.3
    supports_vision: bool = True
    tier: LLMTier = LLMTier.MEDIUM


@dataclass
class TaskConfig:
    """Configuration for a browser automation task"""
    task_id: str
    goal: str
    max_steps: int = 50
    timeout_seconds: int = 300
    enable_thinking: bool = True
    enable_dom_fallback: bool = True
    screenshot_quality: int = 80
    screenshot_max_width: int = 1280
    retry_limit: int = 2
    loop_detection_window: int = 3
    summary_interval: int = 10       # Steps between memory compressions
    max_tabs: int = 5                # Maximum open browser tabs
    ask_user_timeout: int = 120      # Seconds to wait for user response before timing out
