"""In-page overlay for the workflow recorder.

Unlike `demo_overlay.py` (deliberately read-only, pointer-events:none, must
never appear in agent screenshots), this overlay is meant to be SEEN and
CLICKED by the human operator — it shows what's being recorded and exposes a
"Stop & Save Workflow" button. Kept as a separate module so nothing here can
affect the existing agent demo-overlay path.

The root element's id is exported as RECORDER_OVERLAY_ROOT_ID so the
recorder's own click/input listeners can exclude it — without that, clicking
"Stop" would itself get recorded as a workflow step.
"""

from __future__ import annotations

from typing import Optional

RECORDER_OVERLAY_ROOT_ID = "__cosmic_recorder_overlay_root__"
STOP_BINDING_NAME = "__cosmicRecorderStop"

RECORDER_OVERLAY_INIT_SCRIPT = r"""
(() => {
  // Render ONLY in the top-level frame. This script is injected via
  // context.add_init_script, which runs it in EVERY frame — including
  // cross-origin iframes (LinkedIn's "Sign in with Google" button, its
  // "Sign in with email" widget, the messaging panel, etc.). Inside an
  // iframe, position:fixed is relative to that iframe's OWN box, so the panel
  // renders — clipped to the iframe — at each one's top-left. That is exactly
  // the duplicated, truncated "RECORDING WORKFLOW" panels appearing inside
  // page widgets. Bail out in any subframe.
  try { if (window.top !== window.self) return; } catch (_e) { return; }
  if (window.__cosmicRecorderOverlayInstalled) return;
  window.__cosmicRecorderOverlayInstalled = true;

  const ROOT_ID = '__ROOT_ID_PLACEHOLDER__';
  const STOP_BINDING = '__STOP_BINDING_PLACEHOLDER__';

  const HOST_STYLES = [
    ['position', 'fixed'],
    ['top', '14px'],
    ['left', '14px'],
    ['right', 'auto'],
    ['bottom', 'auto'],
    ['z-index', '2147483647'],
    // The host itself must NOT capture clicks — only the Stop button (which
    // explicitly re-enables pointer-events) should. Otherwise this panel
    // would physically block clicks on whatever happens to render underneath
    // it on the real page (caught by the end-to-end test: a button placed
    // near top-left became unclickable through the overlay).
    ['pointer-events', 'none'],
    ['width', '300px'],
    ['box-sizing', 'border-box'],
    ['margin', '0'],
    ['padding', '12px 14px'],
    // Glassy translucent black: low-alpha black tint so the page shows
    // through, with a strong backdrop blur for the frosted-glass look. A faint
    // white hairline border + soft red glow keeps the "recording" identity
    // without the old near-opaque reddish panel.
    ['border', '1px solid rgba(255,255,255,0.16)'],
    ['border-radius', '14px'],
    ['background', 'rgba(0,0,0,0.42)'],
    ['backdrop-filter', 'blur(22px) saturate(160%)'],
    ['-webkit-backdrop-filter', 'blur(22px) saturate(160%)'],
    ['box-shadow', '0 16px 50px rgba(0,0,0,0.45), 0 0 0 1px rgba(255,80,80,0.10)'],
    ['color', '#f3e9e9'],
    ['font', '12px/1.45 -apple-system, "Segoe UI", Arial, sans-serif'],
    ['transform', 'none'],
    ['opacity', '1'],
    ['visibility', 'visible'],
  ];

  function applyHostStyles(host) {
    try {
      for (const [prop, value] of HOST_STYLES) host.style.setProperty(prop, value, 'important');
    } catch (_e) {}
  }

  function el(tag, style, text) {
    const node = document.createElement(tag);
    if (style) node.setAttribute('style', style);
    if (text != null) node.textContent = String(text);
    return node;
  }

  function mount() {
    if (!document || !document.documentElement) return null;
    const existing = document.getElementById(ROOT_ID);
    if (existing && window.__cosmicRecorderOverlay) return window.__cosmicRecorderOverlay;
    if (existing) { try { existing.remove(); } catch (_e) {} }

    const host = document.createElement('div');
    host.id = ROOT_ID;
    applyHostStyles(host);

    const dot = el('span', 'display:inline-block !important;width:8px !important;height:8px !important;border-radius:999px !important;background:#ff5050 !important;margin-right:7px !important;box-shadow:0 0 8px rgba(255,80,80,0.7) !important;animation:cosmicRecPulse 1.2s infinite !important;');
    const styleTag = document.createElement('style');
    styleTag.textContent = '@keyframes cosmicRecPulse {0%,100%{opacity:1} 50%{opacity:0.35}}';
    document.documentElement.appendChild(styleTag);

    const title = el('div', 'display:flex !important;align-items:center !important;font-weight:600 !important;font-size:12px !important;letter-spacing:0.06em !important;text-transform:uppercase !important;margin-bottom:8px !important;color:#ffb3b3 !important;');
    title.appendChild(dot);
    title.appendChild(document.createTextNode('Recording Workflow'));

    const nameRow = el('div', 'font-size:13px !important;font-weight:600 !important;color:#fff !important;margin-bottom:2px !important;overflow:hidden !important;text-overflow:ellipsis !important;white-space:nowrap !important;', '—');
    const goalRow = el('div', 'font-size:11px !important;color:#cbb;margin-bottom:8px !important;max-height:48px !important;overflow:hidden !important;', '—');
    const counterRow = el('div', 'font-size:11px !important;color:#e8c8c8 !important;margin-bottom:10px !important;', 'Actions captured: 0');

    const button = el('button', 'pointer-events:auto !important;width:100% !important;padding:8px 10px !important;border-radius:8px !important;border:1px solid rgba(255,255,255,0.18) !important;background:#ff5050 !important;color:#fff !important;font-weight:600 !important;font-size:12px !important;cursor:pointer !important;', 'Stop & Save Workflow');
    button.addEventListener('click', (ev) => {
      ev.stopPropagation();
      try { button.disabled = true; button.textContent = 'Saving…'; } catch (_e) {}
      try {
        if (window[STOP_BINDING]) window[STOP_BINDING]('stop');
      } catch (_e) {}
    }, true);

    host.appendChild(title);
    host.appendChild(nameRow);
    host.appendChild(goalRow);
    host.appendChild(counterRow);
    host.appendChild(button);
    document.documentElement.appendChild(host);

    function update(patch) {
      try {
        if (!host.isConnected) {
          document.documentElement.appendChild(host);
          applyHostStyles(host);
        }
        if (!patch || typeof patch !== 'object') return;
        if (patch.workflow_name != null) nameRow.textContent = String(patch.workflow_name);
        if (patch.goal != null) goalRow.textContent = String(patch.goal);
        if (patch.actions_captured != null) counterRow.textContent = 'Actions captured: ' + String(patch.actions_captured);
      } catch (_e) {}
    }

    window.__cosmicRecorderOverlay = { update, _host: host };
    return window.__cosmicRecorderOverlay;
  }

  function boot() { try { mount(); } catch (_e) {} }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot, { once: true });
  } else {
    boot();
  }
  let tries = 0;
  const iv = setInterval(() => {
    tries += 1;
    if (window.__cosmicRecorderOverlay || tries > 30) { clearInterval(iv); return; }
    boot();
  }, 80);
})();
"""
RECORDER_OVERLAY_INIT_SCRIPT = (
    RECORDER_OVERLAY_INIT_SCRIPT
    .replace("__ROOT_ID_PLACEHOLDER__", RECORDER_OVERLAY_ROOT_ID)
    .replace("__STOP_BINDING_PLACEHOLDER__", STOP_BINDING_NAME)
)


class RecorderOverlay:
    """Owns the recorder's in-page status panel and Stop button."""

    def __init__(self, workflow_name: str, goal: str):
        self.workflow_name = workflow_name
        self.goal = goal
        self._actions_captured = 0

    async def install(self, context, on_stop) -> None:
        """Inject the overlay script and wire the Stop button to on_stop().

        on_stop: a zero-arg callable (sync or async) invoked when the human
        clicks "Stop & Save Workflow".
        """
        await context.add_init_script(RECORDER_OVERLAY_INIT_SCRIPT)

        async def _handle_stop(source, *_args) -> None:
            result = on_stop()
            if hasattr(result, "__await__"):
                await result

        await context.expose_binding(STOP_BINDING_NAME, _handle_stop)

    async def inject_live(self, page) -> bool:
        """Run the overlay script on an ALREADY-loaded page.

        add_init_script (used by install()) only fires on future navigations,
        so the page that's already open when recording starts would show no
        overlay until the user navigates somewhere. Evaluating the script
        directly makes the overlay appear immediately. Returns False (without
        raising) on pages where evaluation is blocked, e.g. chrome:// /
        chrome-extension:// internal pages.
        """
        try:
            await page.evaluate(RECORDER_OVERLAY_INIT_SCRIPT)
            return True
        except Exception:
            return False

    async def push(self, page, *, actions_captured: Optional[int] = None) -> None:
        if actions_captured is not None:
            self._actions_captured = actions_captured
        snapshot = {
            "workflow_name": self.workflow_name,
            "goal": self.goal,
            "actions_captured": self._actions_captured,
        }
        try:
            await page.evaluate(
                "(s) => { if (window.__cosmicRecorderOverlay) window.__cosmicRecorderOverlay.update(s); }",
                snapshot,
            )
        except Exception:
            pass


__all__ = ["RecorderOverlay", "RECORDER_OVERLAY_ROOT_ID", "STOP_BINDING_NAME", "RECORDER_OVERLAY_INIT_SCRIPT"]
