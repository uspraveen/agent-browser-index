"""Records a human-driven browser session into the same Step/log.json shape
the AI agent already produces, so the existing indexing pipeline
(WorkflowRunIndexer -> TraceCompiler -> WorkflowStore) can distill it into a
replayable workflow without any changes to that pipeline.

Design: prefer DOM selectors over visual coordinates wherever a selector can
be generated (consistent with this project's broader finding that DOM-based
actions replay more reliably than vision-grounded ones whenever a stable
selector exists). Visual coordinates + a real visual_index are only recorded
as a fallback when no usable selector exists for the clicked element (e.g.
canvas, deeply anonymous shadow DOM).
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import imagehash
from PIL import Image

from cosmic_types import ActionResult, ActionType, BrowserState, Step
from .coordinates import build_visual_index
from .recorder_overlay import RECORDER_OVERLAY_ROOT_ID, RecorderOverlay

EVENT_BINDING_NAME = "__cosmicRecorderEvent"

# Injected once per page via context.add_init_script. Generates a compact,
# reasonably stable CSS selector for click targets (prefer id -> common
# test/aria attributes -> nth-of-type chain), and reports click / committed
# input value / Enter keypress / scroll-idle events back to Python via an
# exposed binding. Explicitly ignores anything inside the recorder overlay's
# own root element so clicking "Stop & Save Workflow" doesn't get recorded as
# a workflow step.
RECORDER_INIT_SCRIPT = r"""
(() => {
  if (window.__cosmicRecorderInstalled) return;
  window.__cosmicRecorderInstalled = true;

  const OVERLAY_ROOT_ID = '__OVERLAY_ROOT_ID_PLACEHOLDER__';
  const BINDING = '__BINDING_NAME_PLACEHOLDER__';

  function inOverlay(el) {
    return !!(el && el.closest && el.closest('#' + OVERLAY_ROOT_ID));
  }

  function cssEscape(s) {
    try {
      if (window.CSS && window.CSS.escape) return window.CSS.escape(s);
    } catch (_e) {}
    return String(s).replace(/[^a-zA-Z0-9_-]/g, '\\$&');
  }

  function getSelector(el) {
    if (!(el instanceof Element)) return null;
    if (el.id) return '#' + cssEscape(el.id);
    for (const attr of ['data-testid', 'data-test', 'aria-label', 'name']) {
      const v = el.getAttribute(attr);
      if (v) return el.tagName.toLowerCase() + '[' + attr + '="' + String(v).replace(/"/g, '\\"') + '"]';
    }
    // nth-of-type chain fallback, capped depth so selectors stay short-ish.
    let path = [];
    let node = el;
    let depth = 0;
    while (node && node.nodeType === Node.ELEMENT_NODE && depth < 6) {
      if (node.id) { path.unshift('#' + cssEscape(node.id)); break; }
      let selector = node.tagName.toLowerCase();
      let sibling = node, nth = 1;
      while ((sibling = sibling.previousElementSibling)) {
        if (sibling.tagName === node.tagName) nth++;
      }
      selector += ':nth-of-type(' + nth + ')';
      path.unshift(selector);
      node = node.parentElement;
      depth++;
    }
    return path.join(' > ') || null;
  }

  function describe(el) {
    return (
      el.getAttribute && (el.getAttribute('aria-label') || el.getAttribute('placeholder'))
    ) || (el.innerText ? el.innerText.trim().slice(0, 80) : '') || (el.value ? String(el.value).slice(0, 80) : '') || '';
  }

  function emit(payload) {
    try {
      if (window[BINDING]) window[BINDING](payload);
    } catch (_e) {}
  }

  document.addEventListener('click', (ev) => {
    const el = ev.target;
    if (inOverlay(el)) return;
    const rect = el.getBoundingClientRect ? el.getBoundingClientRect() : {left: 0, top: 0, width: 0, height: 0};
    emit({
      type: 'click',
      selector: getSelector(el),
      tag: el.tagName ? el.tagName.toLowerCase() : '',
      description: describe(el),
      x: Math.round(rect.left + rect.width / 2),
      y: Math.round(rect.top + rect.height / 2),
    });
  }, true);

  document.addEventListener('change', (ev) => {
    const el = ev.target;
    if (inOverlay(el)) return;
    const tag = el.tagName ? el.tagName.toLowerCase() : '';
    if (tag !== 'input' && tag !== 'textarea' && tag !== 'select') return;
    const isPassword = (el.type || '').toLowerCase() === 'password';
    const rect = el.getBoundingClientRect ? el.getBoundingClientRect() : {left: 0, top: 0, width: 0, height: 0};
    emit({
      type: 'input',
      selector: getSelector(el),
      value: isPassword ? '***REDACTED***' : String(el.value || ''),
      masked: isPassword,
      x: Math.round(rect.left + rect.width / 2),
      y: Math.round(rect.top + rect.height / 2),
    });
  }, true);

  document.addEventListener('keydown', (ev) => {
    const el = ev.target;
    if (inOverlay(el)) return;
    if (ev.key !== 'Enter') return;
    const tag = el.tagName ? el.tagName.toLowerCase() : '';
    if (tag !== 'input' && tag !== 'textarea') return;
    emit({ type: 'keydown_enter', selector: getSelector(el) });
  }, true);

  let scrollTimer = null;
  let scrollAccum = 0;
  let lastScrollY = window.scrollY;
  window.addEventListener('scroll', () => {
    scrollAccum += (window.scrollY - lastScrollY);
    lastScrollY = window.scrollY;
    if (scrollTimer) clearTimeout(scrollTimer);
    scrollTimer = setTimeout(() => {
      const delta = scrollAccum;
      scrollAccum = 0;
      if (Math.abs(delta) >= 30) emit({ type: 'scroll', deltaY: Math.round(delta) });
    }, 400);
  }, true);
})();
"""
RECORDER_INIT_SCRIPT = (
    RECORDER_INIT_SCRIPT
    .replace("__OVERLAY_ROOT_ID_PLACEHOLDER__", RECORDER_OVERLAY_ROOT_ID)
    .replace("__BINDING_NAME_PLACEHOLDER__", EVENT_BINDING_NAME)
)


class WorkflowRecorder:
    """Captures a human browsing session into Step-shaped records compatible
    with log.json / the existing indexing pipeline."""

    def __init__(self, browser_controller, workflow_name: str, goal: str, working_dir: Path):
        self.browser = browser_controller
        self.workflow_name = workflow_name
        self.goal = goal
        self.working_dir = Path(working_dir)
        (self.working_dir / "screenshots").mkdir(parents=True, exist_ok=True)
        self.log_path = self.working_dir / "log.json"
        with open(self.log_path, "w") as f:
            json.dump([], f)

        self.steps: List[Step] = []
        self._step_counter = 0
        self._last_state: Optional[BrowserState] = None
        self._last_click_time: float = 0.0
        self._stop_event = asyncio.Event()
        self._instrumented_pages: set = set()
        self.overlay = RecorderOverlay(workflow_name=workflow_name, goal=goal)

    async def start(self) -> None:
        context = self.browser.context
        await context.add_init_script(RECORDER_INIT_SCRIPT)
        await context.expose_binding(EVENT_BINDING_NAME, self._on_event)
        await self.overlay.install(context, on_stop=self._stop_event.set)

        # add_init_script only runs on FUTURE navigations, so the page that's
        # already open right now (the about:blank tab opened during CDP
        # startup) would have no overlay and no event listeners until the user
        # navigates. Inject into every currently-open page directly so the
        # overlay is visible and recording is live immediately. Also follow
        # any new tab the human opens (Ctrl+T), instrumenting it the same way.
        for page in list(context.pages):
            await self._instrument_page(page)
        context.on("page", self._on_new_page)

        _, self._last_state = await self._capture_state()
        await self.overlay.push(self.browser.page, actions_captured=0)
        print(f"🔴 Recording started — workflow '{self.workflow_name}'. Browse normally; click 'Stop & Save Workflow' in the overlay when done.")

    async def _instrument_page(self, page) -> None:
        """Make a live page recordable right now: attach the navigation
        listener and evaluate the listener + overlay scripts directly (rather
        than waiting for the next navigation to trigger add_init_script).
        Safe to call repeatedly — the injected scripts self-guard against
        double-install, and we de-dupe the framenavigated listener."""
        if page not in self._instrumented_pages:
            self._instrumented_pages.add(page)
            page.on("framenavigated", self._on_frame_navigated)
        try:
            await page.evaluate(RECORDER_INIT_SCRIPT)
        except Exception:
            # chrome:// / extension pages block evaluation — that's fine, the
            # init script will run when the user navigates to a real site.
            pass
        await self.overlay.inject_live(page)
        await self.overlay.push(page, actions_captured=self._step_counter)

    def _on_new_page(self, page) -> None:
        # A human opened a new tab. Instrument it once it has settled.
        async def _go():
            try:
                await page.wait_for_load_state("domcontentloaded", timeout=3000)
            except Exception:
                pass
            await self._instrument_page(page)
        asyncio.create_task(_go())

    async def wait_for_stop(self) -> None:
        await self._stop_event.wait()

    # ------------------------------------------------------------------
    # State capture — deliberately lighter than BrowserController.capture_state():
    # no Escape-key press (would interfere with the human's real interaction,
    # e.g. dismissing a dropdown they just opened) and no zombie-tab cleanup.
    # ------------------------------------------------------------------
    async def _capture_state(self) -> tuple[str, BrowserState]:
        page = self.browser.page
        screenshot_path = self.working_dir / "screenshots" / f"step_{self._step_counter:03d}.webp"
        # fast_screenshot uses CDP (no font/stability wait) so capturing on a
        # live, mid-load page during active browsing never hangs — the agent's
        # _safe_page_screenshot blocks on document.fonts.ready, which stalled
        # every recorded event for the full 30s timeout.
        await self.browser.fast_screenshot(
            page, path=screenshot_path, quality=self.browser.config.screenshot_quality
        )
        with Image.open(screenshot_path) as img:
            img_hash = str(imagehash.average_hash(img))
        viewport = page.viewport_size or {"width": self.browser.config.screenshot_max_width, "height": 720}
        raw_title = await page.title()
        state = BrowserState(
            url=page.url,
            title=raw_title,
            viewport_width=viewport["width"],
            viewport_height=viewport["height"],
            scroll_y=await self.browser._safe_evaluate("window.scrollY", fallback=0),
            screenshot_hash=img_hash,
            timestamp=datetime.now(),
        )
        return str(screenshot_path), state

    def _write_log(self) -> None:
        try:
            with open(self.log_path, "r") as f:
                log_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            log_data = []
        log_data.append(self.steps[-1].to_dict())
        with open(self.log_path, "w") as f:
            json.dump(log_data, f, indent=2, default=str)

    async def _append_step(
        self,
        action: ActionResult,
        tool_call: Dict[str, Any],
        before_state: BrowserState,
        screenshot_path: str,
        after_state: BrowserState,
        visual_index: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._step_counter += 1
        if visual_index:
            action.metadata["visual_index"] = visual_index
        step = Step(
            step_number=self._step_counter,
            timestamp=datetime.now(),
            screenshot_path=screenshot_path,
            screenshot_hash=after_state.screenshot_hash,
            browser_state=after_state,
            action=action,
            before_browser_state=before_state,
            after_browser_state=after_state,
            after_screenshot_path=screenshot_path,
            after_screenshot_hash=after_state.screenshot_hash,
            tool_call=tool_call,
            visual_index=visual_index,
        )
        self.steps.append(step)
        self._write_log()
        self._last_state = after_state
        try:
            await self.overlay.push(self.browser.page, actions_captured=self._step_counter)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------
    async def _on_event(self, source: Dict[str, Any], payload: Dict[str, Any]) -> None:
        try:
            event_type = payload.get("type")
            before_state = self._last_state
            screenshot_path, after_state = await self._capture_state()

            if event_type == "click":
                self._last_click_time = time.time()
                selector = payload.get("selector")
                description = payload.get("description") or selector or payload.get("tag") or "clicked element"
                x, y = int(payload.get("x", 0)), int(payload.get("y", 0))
                visual_index = None
                if selector:
                    action = ActionResult(
                        success=True, action_type=ActionType.DOM_CLICK,
                        description=f"Clicked: {description}", coordinates=(x, y),
                    )
                    tool_call = {"action_type": "DOMClick", "parameters": {"selector": selector}}
                else:
                    # No usable selector (e.g. canvas, anonymous shadow DOM) —
                    # fall back to a visual click with a real visual_index so
                    # replay can still scale the coordinates to any viewport.
                    action = ActionResult(
                        success=True, action_type=ActionType.VISUAL_CLICK,
                        description=description or "clicked element", coordinates=(x, y),
                    )
                    tool_call = {"action_type": "VisualClick", "parameters": {"description": description or "clicked element"}}
                    if before_state:
                        visual_index = build_visual_index(action, tool_call, before_state, screenshot_path)
                await self._append_step(action, tool_call, before_state, screenshot_path, after_state, visual_index)

            elif event_type == "input":
                selector = payload.get("selector")
                text = payload.get("value", "")
                if not selector:
                    return  # can't replay typed text without a target selector
                action = ActionResult(
                    success=True, action_type=ActionType.DOM_TYPE,
                    description=f"Typed into {selector}",
                    coordinates=(int(payload.get("x", 0)), int(payload.get("y", 0))),
                )
                tool_call = {"action_type": "DomType", "parameters": {"selector": selector, "text": text, "press_enter": False}}
                await self._append_step(action, tool_call, before_state, screenshot_path, after_state)

            elif event_type == "keydown_enter":
                action = ActionResult(success=True, action_type=ActionType.PRESS_KEY, description="Pressed Enter")
                tool_call = {"action_type": "PressKey", "parameters": {"key": "Enter"}}
                await self._append_step(action, tool_call, before_state, screenshot_path, after_state)

            elif event_type == "scroll":
                delta = int(payload.get("deltaY", 0))
                direction = "down" if delta > 0 else "up"
                action = ActionResult(success=True, action_type=ActionType.VISUAL_SCROLL, description=f"Scrolled {direction}")
                tool_call = {"action_type": "VisualScroll", "parameters": {"direction": direction, "amount": abs(delta)}}
                await self._append_step(action, tool_call, before_state, screenshot_path, after_state)
        except Exception as e:
            print(f"⚠️  Recorder event handling failed ({payload.get('type') if isinstance(payload, dict) else '?'}): {e}")

    def _on_frame_navigated(self, frame) -> None:
        # Only the main frame, and only if this navigation wasn't likely
        # caused by a click we already recorded (clicking a link produces its
        # own DOMClick/VisualClick step whose after-state already shows the
        # new URL — recording a second synthetic Navigate step for the same
        # human action would be redundant). This heuristic only catches
        # address-bar navigation, back/forward, and bookmarks.
        if frame is not self.browser.page.main_frame:
            return
        if time.time() - self._last_click_time < 0.8:
            return
        asyncio.create_task(self._handle_explicit_navigation())

    async def _handle_explicit_navigation(self) -> None:
        try:
            before_state = self._last_state
            screenshot_path, after_state = await self._capture_state()
            if before_state and after_state.url == before_state.url:
                return
            action = ActionResult(success=True, action_type=ActionType.NAVIGATE, description=f"Navigated to {after_state.url}")
            tool_call = {"action_type": "Navigate", "parameters": {"url": after_state.url}}
            await self._append_step(action, tool_call, before_state, screenshot_path, after_state)
        except Exception as e:
            print(f"⚠️  Recorder navigation handling failed: {e}")


__all__ = ["WorkflowRecorder", "RECORDER_INIT_SCRIPT", "EVENT_BINDING_NAME"]
