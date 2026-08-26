"""Visible action cues (cursor, clicks, typing, scrolling, keys) for watching
the agent work live.

Playwright dispatches input via CDP (Input.dispatchMouseEvent etc.) — this
does NOT move the OS-level mouse cursor a human sees on screen, so watching a
real (non-headless) agent-controlled Chrome window gives no visual signal of
what the agent is doing. This module injects a blue-themed effect layer:

  - a glowing blue cursor dot that GLIDES to each interaction point
    (distance-scaled duration, capped ~380ms) instead of teleporting;
  - an expanding ring + soft splash pulse on clicks;
  - a "typing…" pill with bouncing dots anchored near the focused field,
    plus a pulsing halo on the cursor dot, while text is being typed;
  - drifting chevrons at the right edge showing scroll direction;
  - a bottom-center key badge (e.g. "Enter") on key presses.

Every effect lives inside ONE fixed, pointer-events:none container element.
That container is what hide/show toggles — so the hide-for-capture path
covers transient effects (rings mid-animation, typing pills) too, not just
the dot. (Previously the pulse ring lived outside the hide mechanism and
could leak into a MiMo screenshot taken during its 600ms lifetime.)

Like demo_overlay.py, none of this must EVER appear in a screenshot consumed
by MiMo/the orchestrator — a stray blue mark near the target could confuse
vision grounding. Hidden via the same hide-for-capture / restore-after-capture
pattern, wired into BrowserController._safe_page_screenshot, the single
chokepoint all agent screenshot consumers go through.

Timing contract: moveTo/pulseAt return promises that resolve when the glide
lands (or immediately on any error), and page.evaluate awaits them — so the
physical click fires the moment the visual cursor arrives. Both the JS
(unconditional setTimeout resolve) and the Python side (asyncio.wait_for cap)
guarantee an animation can never stall a real action.
"""

from __future__ import annotations

import asyncio

CURSOR_OVERLAY_INIT_SCRIPT = r"""
(() => {
  if (window.__cosmicCursorInstalled) return;
  window.__cosmicCursorInstalled = true;

  const ROOT_ID = '__cosmic_fx__';
  const DOT_ID = '__cosmic_cursor_dot__';
  // Blue palette: #93C5FD light / #3B82F6 primary / #2563EB deep
  const B_LIGHT = '147,197,253';
  const B_MAIN  = '59,130,246';
  const B_DEEP  = '37,99,235';

  function ensureRoot() {
    let root = document.getElementById(ROOT_ID);
    if (root && root.isConnected) return root;
    root = document.createElement('div');
    root.id = ROOT_ID;
    root.setAttribute('aria-hidden', 'true');
    root.style.cssText = 'position:fixed;inset:0;pointer-events:none;z-index:2147483647;'
      + (window.__cosmicCursorVisible === false ? 'display:none;' : 'display:block;');
    const style = document.createElement('style');
    style.textContent = `
      @keyframes __cfx_ring   { from { transform:translate(-50%,-50%) scale(.4);  opacity:.95; }
                                to   { transform:translate(-50%,-50%) scale(3.4); opacity:0; } }
      @keyframes __cfx_splash { from { transform:translate(-50%,-50%) scale(.3);  opacity:.4; }
                                to   { transform:translate(-50%,-50%) scale(2.3); opacity:0; } }
      @keyframes __cfx_badge  { 0%  { opacity:0; transform:translate(-50%,8px); }
                                15% { opacity:1; transform:translate(-50%,0); }
                                75% { opacity:1; } 100% { opacity:0; } }
      @keyframes __cfx_bounce { 0%,80%,100% { transform:translateY(0);    opacity:.45; }
                                40%         { transform:translateY(-4px); opacity:1; } }
      @keyframes __cfx_chev   { 0%  { opacity:0; transform:rotate(var(--cfx-rot)) translate(0,0); }
                                25% { opacity:.95; }
                                100%{ opacity:0; transform:rotate(var(--cfx-rot)) translate(var(--cfx-dx),var(--cfx-dy)); } }
      @keyframes __cfx_typepulse {
        0%,100% { box-shadow:0 0 0 3px rgba(${B_MAIN},.35), 0 0 12px rgba(${B_MAIN},.75); }
        50%     { box-shadow:0 0 0 8px rgba(${B_MAIN},.12), 0 0 18px rgba(${B_MAIN},.9); } }
    `;
    root.appendChild(style);
    document.documentElement.appendChild(root);
    return root;
  }

  function ensureDot() {
    const root = ensureRoot();
    let dot = document.getElementById(DOT_ID);
    if (dot && dot.isConnected && dot.parentElement === root) return dot;
    if (dot) { try { dot.remove(); } catch (_e) {} }
    dot = document.createElement('div');
    dot.id = DOT_ID;
    dot.setAttribute('aria-hidden', 'true');
    dot.style.cssText = [
      'position:fixed', 'width:16px', 'height:16px', 'border-radius:50%',
      `background:radial-gradient(circle at 35% 35%, rgba(${B_LIGHT},.95), rgba(${B_MAIN},.95) 55%, rgba(${B_DEEP},.95))`,
      'border:2px solid rgba(255,255,255,.9)',
      `box-shadow:0 0 0 3px rgba(${B_MAIN},.35), 0 0 12px rgba(${B_MAIN},.75)`,
      'pointer-events:none',
      'transform:translate(-50%,-50%)',
      'left:-9999px', 'top:-9999px',
    ].join(';');
    root.appendChild(dot);
    return dot;
  }

  // Resolves when the glide lands. From offscreen it jumps instantly (no
  // glide from nowhere); otherwise duration scales with distance, capped so
  // it can never meaningfully delay the real click that follows.
  function moveTo(x, y) {
    return new Promise((resolve) => {
      try {
        const dot = ensureDot();
        const px = parseFloat(dot.style.left);
        const py = parseFloat(dot.style.top);
        const offscreen = !isFinite(px) || px < -1000;
        const dist = offscreen ? 0 : Math.hypot(x - px, y - py);
        const ms = (offscreen || dist < 2) ? 0 : Math.max(90, Math.min(380, Math.round(dist * 0.9)));
        dot.style.transition = ms
          ? `left ${ms}ms cubic-bezier(.3,.7,.3,1), top ${ms}ms cubic-bezier(.3,.7,.3,1)`
          : 'none';
        void dot.offsetWidth; // commit transition before moving
        dot.style.left = x + 'px';
        dot.style.top = y + 'px';
        setTimeout(resolve, ms + 20);
      } catch (_e) { resolve(); }
    });
  }

  // Glide to the point, then fire ring + splash. Resolves at arrival (the
  // pulse plays out on its own), so the caller clicks exactly on arrival.
  function pulseAt(x, y) {
    return moveTo(x, y).then(() => {
      try {
        const root = ensureRoot();
        const ring = document.createElement('div');
        ring.style.cssText = [
          'position:fixed', 'width:18px', 'height:18px', 'border-radius:50%',
          `border:2.5px solid rgba(${B_MAIN},.95)`, 'pointer-events:none',
          `left:${x}px`, `top:${y}px`,
          'animation:__cfx_ring .55s ease-out forwards',
        ].join(';');
        const splash = document.createElement('div');
        splash.style.cssText = [
          'position:fixed', 'width:26px', 'height:26px', 'border-radius:50%',
          `background:radial-gradient(circle, rgba(${B_LIGHT},.55), rgba(${B_MAIN},.25) 60%, transparent 75%)`,
          'pointer-events:none', `left:${x}px`, `top:${y}px`,
          'animation:__cfx_splash .45s ease-out forwards',
        ].join(';');
        root.appendChild(ring);
        root.appendChild(splash);
        setTimeout(() => { try { ring.remove(); splash.remove(); } catch (_e) {} }, 650);
      } catch (_e) {}
    });
  }

  let typingPill = null;
  let typingTimer = null;

  function stopTyping() {
    try {
      if (typingTimer) { clearTimeout(typingTimer); typingTimer = null; }
      if (typingPill) { try { typingPill.remove(); } catch (_e) {} typingPill = null; }
      const dot = document.getElementById(DOT_ID);
      if (dot) dot.style.animation = '';
    } catch (_e) {}
  }

  function startTyping(x, y) {
    try {
      stopTyping();
      const root = ensureRoot();
      const pill = document.createElement('div');
      const left = Math.max(70, Math.min((window.innerWidth || 1280) - 70, x));
      const top = Math.max(14, y - 42);
      pill.style.cssText = [
        'position:fixed', `left:${left}px`, `top:${top}px`,
        'transform:translate(-50%,0)',
        `background:rgba(${B_DEEP},.92)`, 'color:#fff',
        'border-radius:999px', 'padding:5px 12px',
        'font:600 12px/1 system-ui,Segoe UI,sans-serif', 'letter-spacing:.02em',
        `box-shadow:0 2px 10px rgba(${B_DEEP},.45)`,
        'pointer-events:none', 'display:flex', 'align-items:center', 'gap:6px',
      ].join(';');
      const label = document.createElement('span');
      label.textContent = 'typing';
      pill.appendChild(label);
      for (let i = 0; i < 3; i++) {
        const d = document.createElement('span');
        d.style.cssText = [
          'width:4px', 'height:4px', 'border-radius:50%', 'background:#fff',
          `animation:__cfx_bounce 1s ease-in-out ${i * 0.15}s infinite`,
        ].join(';');
        pill.appendChild(d);
      }
      root.appendChild(pill);
      typingPill = pill;
      const dot = ensureDot();
      dot.style.animation = '__cfx_typepulse 1s ease-in-out infinite';
      // Failsafe: if the caller never reaches stopTyping (error mid-type,
      // navigation raced us), the pill must not linger forever.
      typingTimer = setTimeout(stopTyping, 25000);
    } catch (_e) {}
  }

  // Drifting chevrons at the right edge showing scroll direction.
  function scrollCue(direction) {
    try {
      const root = ensureRoot();
      const dir = String(direction || 'down').toLowerCase();
      const down = (dir === 'down' || dir === 'bottom');
      const big = (dir === 'top' || dir === 'bottom');
      const baseY = Math.round((window.innerHeight || 720) / 2);
      const x = (window.innerWidth || 1280) - 38;
      for (let i = 0; i < 3; i++) {
        const c = document.createElement('div');
        const offset = (i - 1) * 16 * (down ? 1 : -1);
        c.style.cssText = [
          'position:fixed', 'width:14px', 'height:14px', 'pointer-events:none',
          `left:${x}px`, `top:${baseY + offset}px`,
          `border-right:3px solid rgba(${B_MAIN},.95)`,
          `border-bottom:3px solid rgba(${B_MAIN},.95)`,
          `filter:drop-shadow(0 0 6px rgba(${B_MAIN},.7))`,
          `--cfx-rot:${down ? '45deg' : '-135deg'}`,
          `--cfx-dx:${big ? '30px' : '18px'}`,
          `--cfx-dy:${big ? '30px' : '18px'}`,
          `transform:rotate(${down ? '45deg' : '-135deg'})`,
          `animation:__cfx_chev .7s ease-out ${i * 0.09}s forwards`,
          'opacity:0',
        ].join(';');
        root.appendChild(c);
        setTimeout(() => { try { c.remove(); } catch (_e) {} }, 1200);
      }
    } catch (_e) {}
  }

  // Bottom-center badge naming the key just pressed (e.g. "Enter").
  function keyBadge(label) {
    try {
      const root = ensureRoot();
      const badge = document.createElement('div');
      badge.textContent = String(label || '');
      badge.style.cssText = [
        'position:fixed', 'left:50%', 'bottom:42px', 'transform:translate(-50%,0)',
        `background:rgba(${B_DEEP},.92)`, 'color:#fff',
        'border:1px solid rgba(255,255,255,.25)',
        'border-radius:8px', 'padding:6px 14px',
        'font:600 13px/1 system-ui,Segoe UI,sans-serif', 'letter-spacing:.04em',
        `box-shadow:0 2px 12px rgba(${B_DEEP},.5)`,
        'pointer-events:none',
        'animation:__cfx_badge .9s ease-out forwards',
      ].join(';');
      root.appendChild(badge);
      setTimeout(() => { try { badge.remove(); } catch (_e) {} }, 950);
    } catch (_e) {}
  }

  // Container-level hide/show: covers the dot AND every transient effect
  // (rings, pills, chevrons, badges) in one switch.
  function hide() {
    window.__cosmicCursorVisible = false;
    const root = document.getElementById(ROOT_ID);
    if (root) root.style.display = 'none';
  }

  function show() {
    window.__cosmicCursorVisible = true;
    const root = document.getElementById(ROOT_ID);
    if (root) root.style.display = 'block';
  }

  window.__cosmicCursor = { moveTo, pulseAt, startTyping, stopTyping, scrollCue, keyBadge, hide, show };
})();
"""


class CursorOverlayManager:
    """Owns the in-page action-effects layer. Cheap no-op when disabled.

    Every public method swallows all errors: a visual cue must never be able
    to fail (or delay, beyond the small capped glide) a real agent action.
    """

    INIT_SCRIPT = CURSOR_OVERLAY_INIT_SCRIPT

    # Hard cap on how long any effect evaluate may take. The JS promises
    # resolve via setTimeout in ≤ ~400ms; this guards the pathological cases
    # (page busy-looping, navigation racing the evaluate).
    _EVAL_TIMEOUT_S = 2.0

    def __init__(self, enabled: bool = True):
        self.enabled = bool(enabled)

    async def _ensure_installed(self, page) -> None:
        # Re-evaluated (idempotently, guarded by window.__cosmicCursorInstalled)
        # on every call rather than relying solely on add_init_script — that
        # only applies to FUTURE navigations, not whatever page is already
        # loaded when the overlay is first used (same gap discovered and
        # fixed for the workflow recorder earlier this session).
        try:
            await page.evaluate(self.INIT_SCRIPT)
        except Exception:
            pass

    async def _fire(self, page, js: str, arg=None) -> None:
        """Install-then-evaluate with the standard guards and time cap."""
        if not self.enabled or page is None:
            return
        await self._ensure_installed(page)
        try:
            if arg is None:
                await asyncio.wait_for(page.evaluate(js), timeout=self._EVAL_TIMEOUT_S)
            else:
                await asyncio.wait_for(page.evaluate(js, arg), timeout=self._EVAL_TIMEOUT_S)
        except Exception:
            pass

    async def show_click(self, page, x: int, y: int) -> None:
        """Glide the dot to (x, y) and pulse. Resolves at arrival, so callers
        should await this immediately before the physical click."""
        await self._fire(
            page,
            "(p) => window.__cosmicCursor ? window.__cosmicCursor.pulseAt(p.x, p.y) : null",
            {"x": x, "y": y},
        )

    async def show_move(self, page, x: int, y: int) -> None:
        await self._fire(
            page,
            "(p) => window.__cosmicCursor ? window.__cosmicCursor.moveTo(p.x, p.y) : null",
            {"x": x, "y": y},
        )

    async def show_typing_start(self, page, x: int, y: int) -> None:
        """Show the typing pill near the field at (x, y) and pulse the dot.
        Auto-expires in-page after 25s even if show_typing_stop never runs."""
        await self._fire(
            page,
            "(p) => { if (window.__cosmicCursor) window.__cosmicCursor.startTyping(p.x, p.y); }",
            {"x": x, "y": y},
        )

    async def show_typing_stop(self, page) -> None:
        await self._fire(
            page,
            "() => { if (window.__cosmicCursor) window.__cosmicCursor.stopTyping(); }",
        )

    async def show_scroll(self, page, direction: str) -> None:
        """Drifting chevrons showing scroll direction (up/down/top/bottom)."""
        await self._fire(
            page,
            "(d) => { if (window.__cosmicCursor) window.__cosmicCursor.scrollCue(d); }",
            str(direction or "down"),
        )

    async def show_key(self, page, label: str) -> None:
        """Bottom-center badge naming the key being pressed (e.g. 'Enter')."""
        await self._fire(
            page,
            "(k) => { if (window.__cosmicCursor) window.__cosmicCursor.keyBadge(k); }",
            str(label or ""),
        )

    async def hide_for_agent_capture(self, page) -> None:
        if not self.enabled or page is None:
            return
        try:
            await page.evaluate("() => { if (window.__cosmicCursor) window.__cosmicCursor.hide(); }")
        except Exception:
            pass

    async def restore_after_agent_capture(self, page) -> None:
        if not self.enabled or page is None:
            return
        try:
            await page.evaluate("() => { if (window.__cosmicCursor) window.__cosmicCursor.show(); }")
        except Exception:
            pass


__all__ = ["CursorOverlayManager", "CURSOR_OVERLAY_INIT_SCRIPT"]
