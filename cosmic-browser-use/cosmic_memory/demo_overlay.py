"""Demo-only visual overlay for COSMIC Browser Memory.

This module is purely a presentation layer. It must never:
  * affect agent decisions or coordinates
  * intercept page input (pointer-events stays `none`)
  * appear in any screenshot consumed by MiMo / orchestrator

The Python `DemoOverlayManager` owns local mirror state and pushes it into
each Playwright page. The browser-side script lives inside an injected init
script and renders a fixed, pointer-events-none DOM island. It deliberately
avoids `innerHTML` so Trusted Types / CSP-heavy sites cannot blank the panel.

Activation is gated by the `--demo-overlay` CLI flag in main.py; when the
flag is off, every method becomes an inexpensive no-op.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional


# JavaScript injected once per page (via context.add_init_script). It is
# defensive: if executed before <html> exists, it retries until the
# documentElement is available. The overlay uses real DOM nodes and textContent
# rather than HTML strings so it survives Trusted Types / CSP-heavy sites.
OVERLAY_INIT_SCRIPT = r"""
(() => {
  if (window.__cosmicOverlayInstalled) return;
  window.__cosmicOverlayInstalled = true;

  const ROOT_ID = '__cosmic_demo_overlay_root__';

  // Critical host styles. Using setProperty(..., 'important') so page CSS
  // cannot override us via tag/attribute/* selectors.
  //  - No `all: initial`: it resets `display` to `inline`, and on some pages
  //    that interacted poorly with our later `display: block` declaration.
  //  - No `contain: size`: that forces the element to ignore its intrinsic
  //    content size and collapses to 0 height when no explicit height is set.
  const HOST_STYLES = [
    ['position', 'fixed'],
    ['top', '14px'],
    ['right', '14px'],
    ['left', 'auto'],
    ['bottom', 'auto'],
    ['z-index', '2147483647'],
    ['pointer-events', 'none'],
    ['width', '320px'],
    ['min-width', '320px'],
    ['max-width', '320px'],
    ['height', 'auto'],
    ['max-height', '92vh'],
    ['display', 'block'],
    ['box-sizing', 'border-box'],
    ['margin', '0'],
    ['padding', '0'],
    ['border', '0'],
    ['float', 'none'],
    ['transform', 'none'],
    // NOTE: visibility/opacity are *not* in this list — they are toggled at
    // runtime by show()/hide() and must not be re-asserted by the watchdog.
    ['contain', 'layout style'],
    // Match the COSMIC sign-in wordmark font stack (Bahnschrift on Windows,
    // Avenir Next Condensed on macOS, Segoe UI Variable Display fallback).
    ['font', '12px/1.45 Bahnschrift, "Avenir Next Condensed", "Segoe UI Variable Display", "Helvetica Neue", Arial, sans-serif'],
    ['color', '#eef2f9'],
    // Asserted once so page CSS can't make us invisible while still
    // letting show()/hide() override them later in the same priority bucket.
    ['opacity', '1'],
    ['visibility', 'visible'],
    // Defensive: even if the panel loses its computed height, the host itself
    // stays clearly visible.
    ['min-height', '230px'],
  ];

  function applyHostStyles(host) {
    try {
      for (const [prop, value] of HOST_STYLES) {
        host.style.setProperty(prop, value, 'important');
      }
    } catch (_e) {}
  }

  function mount() {
    if (!document || !document.documentElement) return null;
    const existing = document.getElementById(ROOT_ID);
    if (existing && window.__cosmicOverlay) return window.__cosmicOverlay;
    if (existing) { try { existing.remove(); } catch (_e) {} }

    const host = document.createElement('div');
    host.id = ROOT_ID;
    host.setAttribute('aria-hidden', 'true');
    applyHostStyles(host);

    // The panel lives directly under the host with inline !important styles.
    // No shadow DOM: previous shadow-based renders had their content go blank
    // on YouTube once navigation churn started. Plain DOM + inline styles
    // can't lose their stylesheet because they don't depend on one.
    const root = document.createElement('div');
    root.id = ROOT_ID + '__panel';
    root.setAttribute(
      'style',
      [
        'position:relative !important',
        'display:block !important',
        'box-sizing:border-box !important',
        'width:100% !important',
        'min-height:220px !important',
        'max-height:92vh !important',
        'margin:0 !important',
        'padding:14px 16px !important',
        'border:1px solid rgba(255,255,255,0.10) !important',
        'border-radius:14px !important',
        'background:linear-gradient(180deg, rgba(8,8,10,0.78) 0%, rgba(0,0,0,0.82) 100%) !important',
        'backdrop-filter:blur(22px) saturate(140%) !important',
        '-webkit-backdrop-filter:blur(22px) saturate(140%) !important',
        'box-shadow:0 18px 48px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.06), inset 0 0 0 1px rgba(255,255,255,0.03) !important',
        'color:#eef2f9 !important',
        'font:12px/1.45 Bahnschrift, "Avenir Next Condensed", "Segoe UI Variable Display", "Helvetica Neue", Arial, sans-serif !important',
        'overflow:hidden !important',
        'transform:translateZ(0) !important',
        'transition:none !important',
        'opacity:1 !important',
        'visibility:visible !important',
        'pointer-events:none !important',
        'text-align:left !important',
        'text-transform:none !important',
        'letter-spacing:normal !important',
        'line-height:1.45 !important',
        'word-spacing:normal !important',
      ].join(';'),
    );
    host.appendChild(root);

    document.documentElement.appendChild(host);

    const state = {
      mode: 'Idle',
      supermemory: 'Initializing',
      workflow_id: null,
      recall_score: null,
      phase: 'Waiting',
      timeline: [],
      metrics: {
        elapsed_sec: 0,
        replay_actions: 0,
        mimo_calls: 0,
        dom_calls: 0,
        llm_calls: 0,
        mimo_calls_avoided: 0,
      },
      failure_patch: null,
      pulse_until: 0,
    };

    function fmtTime(s) {
      s = Math.max(0, Math.round(Number(s) || 0));
      const m = Math.floor(s / 60);
      const r = s - m * 60;
      return m + ':' + String(r).padStart(2, '0');
    }
    function shortWorkflow(wf) {
      if (!wf) return '—';
      const s = String(wf);
      return s.length > 22 ? s.slice(0, 10) + '…' + s.slice(-8) : s;
    }

    // ------------------------------------------------------------------
    // Inline styles for every rendered child. Why inline? On busy SPAs
    // (YouTube/Polymer) we saw the panel chrome painting correctly while
    // child content vanished. Keeping styles and text pinned on real DOM
    // nodes avoids both missing stylesheets and Trusted Types innerHTML
    // failures.
    // ------------------------------------------------------------------
    const FONT_STACK = 'Bahnschrift, "Avenir Next Condensed", "Segoe UI Variable Display", "Helvetica Neue", Arial, sans-serif';
    // Without shadow DOM, page CSS can reach our nodes — every property we
    // care about uses !important so site-wide tag selectors can't override.
    const FONT_INH = 'font-family:' + FONT_STACK + ' !important;font-style:normal !important;';
    const S = {
      header:      'display:flex !important;align-items:center !important;gap:8px !important;margin:0 0 12px 0 !important;padding:0 !important;color:#eef2f9 !important;' + FONT_INH,
      brand:       FONT_INH + 'font-weight:600 !important;font-size:11px !important;letter-spacing:0.32em !important;color:#eef2f9 !important;text-transform:uppercase !important;display:inline-flex !important;align-items:center !important;line-height:1.2 !important;',
      brandDot:    'display:inline-block !important;width:6px !important;height:6px !important;border-radius:999px !important;background:#ffffff !important;margin-right:8px !important;box-shadow:0 0 8px rgba(255,255,255,0.45) !important;',
      badge:       'margin-left:auto !important;' + FONT_INH + 'font-size:10px !important;padding:2px 8px !important;border-radius:999px !important;background:rgba(255,255,255,0.08) !important;color:#eef2f9 !important;border:1px solid rgba(255,255,255,0.16) !important;letter-spacing:0.12em !important;text-transform:uppercase !important;line-height:1.4 !important;',
      row:         'display:flex !important;justify-content:space-between !important;gap:10px !important;padding:3px 0 !important;margin:0 !important;' + FONT_INH,
      rowK:        'color:#9aa3b2 !important;font-size:12px !important;font-weight:400 !important;' + FONT_INH,
      rowV:        'color:#eef2f9 !important;font-weight:500 !important;font-size:12px !important;max-width:62% !important;text-align:right !important;overflow:hidden !important;text-overflow:ellipsis !important;white-space:nowrap !important;' + FONT_INH,
      rowVOk:      'color:#9be8b6 !important;font-weight:500 !important;font-size:12px !important;max-width:62% !important;text-align:right !important;overflow:hidden !important;text-overflow:ellipsis !important;white-space:nowrap !important;' + FONT_INH,
      rowVMuted:   'color:#9aa3b2 !important;font-weight:500 !important;font-size:12px !important;max-width:62% !important;text-align:right !important;overflow:hidden !important;text-overflow:ellipsis !important;white-space:nowrap !important;' + FONT_INH,
      section:     'margin:10px 0 0 0 !important;padding:8px 0 0 0 !important;border:0 !important;border-top:1px dashed rgba(255,255,255,0.08) !important;' + FONT_INH,
      sectionTitle:'font-size:10px !important;letter-spacing:0.18em !important;text-transform:uppercase !important;color:#9aa3b2 !important;margin:0 0 6px 0 !important;padding:0 !important;font-weight:500 !important;' + FONT_INH,
      phase:       'display:inline-flex !important;align-items:center !important;gap:6px !important;padding:4px 9px !important;border-radius:8px !important;background:rgba(255,255,255,0.06) !important;border:1px solid rgba(255,255,255,0.14) !important;color:#ffffff !important;font-weight:500 !important;font-size:12px !important;line-height:1.3 !important;' + FONT_INH,
      phaseDot:    'width:6px !important;height:6px !important;border-radius:999px !important;background:#ffffff !important;display:inline-block !important;',
      timeline:    'display:flex !important;flex-direction:column !important;gap:4px !important;padding:0 !important;margin:0 !important;',
      tlItem:      'display:flex !important;align-items:center !important;gap:8px !important;min-height:18px !important;font-size:12px !important;' + FONT_INH,
      tlMarker:    'width:6px !important;height:6px !important;border-radius:999px !important;flex:none !important;',
      tlLabel:     'color:#d8dde6 !important;white-space:nowrap !important;overflow:hidden !important;text-overflow:ellipsis !important;font-size:12px !important;' + FONT_INH,
      tlLabelMuted:'color:#9aa3b2 !important;white-space:nowrap !important;overflow:hidden !important;text-overflow:ellipsis !important;font-size:12px !important;' + FONT_INH,
      metrics:     'display:grid !important;grid-template-columns:1fr 1fr !important;gap:6px 12px !important;padding:0 !important;margin:0 !important;',
      metricK:     'font-size:10px !important;color:#9aa3b2 !important;text-transform:uppercase !important;letter-spacing:0.18em !important;margin:0 !important;padding:0 !important;font-weight:500 !important;' + FONT_INH,
      metricV:     'font-size:14px !important;color:#eef2f9 !important;font-weight:600 !important;margin:0 !important;padding:0 !important;' + FONT_INH,
      metricVOk:   'font-size:14px !important;color:#9be8b6 !important;font-weight:600 !important;margin:0 !important;padding:0 !important;' + FONT_INH,
      patch:       'margin:8px 0 0 0 !important;padding:6px 8px !important;border-radius:8px !important;background:rgba(255,200,120,0.08) !important;border:1px solid rgba(255,200,120,0.22) !important;color:#ffd9a8 !important;font-size:11px !important;' + FONT_INH,
    };
    const MARKER_COLORS = {
      recall:    '#c8b8ff',
      indexed:   '#ffffff',
      navigate:  '#b5e3c4',
      checkpoint:'#ffd99a',
      live:      '#ffb0bf',
      saved:     '#9be8b6',
    };

    function el(tag, style, text) {
      const node = document.createElement(tag);
      if (style) node.setAttribute('style', style);
      if (text != null) node.textContent = String(text);
      return node;
    }

    function append(parent, children) {
      for (const child of children) {
        if (child) parent.appendChild(child);
      }
      return parent;
    }

    function row(label, value, valueStyle, title) {
      const wrap = el('div', S.row);
      const left = el('span', S.rowK, label);
      const right = el('span', valueStyle || S.rowV, value == null || value === '' ? '—' : value);
      if (title) right.setAttribute('title', String(title));
      return append(wrap, [left, right]);
    }

    function metric(label, value, ok) {
      const wrap = el('div', '');
      return append(wrap, [
        el('div', S.metricK, label),
        el('div', ok ? S.metricVOk : S.metricV, value == null || value === '' ? 0 : value),
      ]);
    }

    function marker(color) {
      return el('span', S.tlMarker + 'background:' + color + ' !important;');
    }

    function timelineItem(kind, label, muted) {
      const item = el('div', S.tlItem);
      const color = muted ? 'rgba(255,255,255,0.5)' : (MARKER_COLORS[String(kind || 'indexed').toLowerCase()] || '#ffffff');
      return append(item, [
        marker(color),
        el('span', muted ? S.tlLabelMuted : S.tlLabel, label || ''),
      ]);
    }

    function section(title, body) {
      const wrap = el('div', S.section);
      return append(wrap, [el('div', S.sectionTitle, title), body]);
    }

    function render() {
      const s = state;
      const supermemoryRaw = String(s.supermemory || '');
      const supermemoryOk = /^connected/i.test(supermemoryRaw);
      const wf = shortWorkflow(s.workflow_id);
      const score = (s.recall_score == null) ? '—' : Number(s.recall_score).toFixed(2);
      const m = s.metrics || {};

      const tlItems = (s.timeline || []).slice(-5);
      const supermemoryValueStyle = supermemoryOk ? S.rowVOk : S.rowVMuted;

      const header = el('div', S.header);
      const brand = el('span', S.brand);
      append(brand, [el('span', S.brandDot), document.createTextNode('Cosmic Browser Memory')]);
      append(header, [brand, el('span', S.badge, s.mode || 'Idle')]);

      const phase = el('div', S.phase);
      append(phase, [el('span', S.phaseDot), document.createTextNode(String(s.phase || '—'))]);

      const timeline = el('div', S.timeline);
      if (tlItems.length) {
        for (const item of tlItems) {
          timeline.appendChild(timelineItem(item.kind, item.label, false));
        }
      } else {
        timeline.appendChild(timelineItem('indexed', 'No replay actions yet', true));
      }

      const metrics = el('div', S.metrics);
      append(metrics, [
        metric('Elapsed', fmtTime(m.elapsed_sec), false),
        metric('Replay actions', m.replay_actions || 0, false),
        metric('MiMo avoided', m.mimo_calls_avoided || 0, true),
        metric('MiMo calls', m.mimo_calls || 0, false),
        metric('DOM calls', m.dom_calls || 0, false),
        metric('LLM calls', m.llm_calls || 0, false),
      ]);

      const children = [
        header,
        row('Supermemory', supermemoryRaw || '—', supermemoryValueStyle),
        row('Workflow', wf, S.rowV, s.workflow_id || ''),
        row('Recall score', score, S.rowV),
        section('Replay phase', phase),
        section('Action timeline', timeline),
        section('Metrics', metrics),
      ];
      if (s.failure_patch) {
        children.push(el('div', S.patch, '⚡ ' + String(s.failure_patch)));
      }
      root.replaceChildren(...children);
    }

    function update(patch) {
      try {
        // Defensive: if the host got detached by a SPA re-render, reattach
        // and re-apply our pinned styles. We do NOT call applyHostStyles on
        // every update — that thrashes inline styles and triggers expensive
        // mutation cycles on busy pages like YouTube.
        if (!host.isConnected) {
          try { document.documentElement.appendChild(host); } catch (_e) {}
          applyHostStyles(host);
        }
        if (!patch || typeof patch !== 'object') return;
        for (const [k, v] of Object.entries(patch)) {
          if (k === 'metrics' && v && typeof v === 'object') {
            state.metrics = Object.assign({}, state.metrics, v);
          } else if (k === 'timeline_append' && v) {
            state.timeline = (state.timeline || []).concat([v]).slice(-12);
          } else if (k === 'timeline_replace' && Array.isArray(v)) {
            state.timeline = v.slice(-12);
          } else if (k === 'pulse_ms') {
            const ms = Math.max(0, Math.min(8000, Number(v) || 0));
            if (ms) state.pulse_until = Date.now() + ms;
          } else {
            state[k] = v;
          }
        }
        render();
      } catch (_e) { /* never throw into page */ }
    }

    window.__cosmicOverlay = {
      update,
      show() {
        try {
          host.style.setProperty('display', 'block', 'important');
          host.style.setProperty('visibility', 'visible', 'important');
          host.style.setProperty('opacity', '1', 'important');
        } catch (_e) {}
      },
      hide() {
        try {
          // Use visibility/opacity so our `!important` `display: block` keeps
          // us out of the layout flow but the element stays in the DOM.
          host.style.setProperty('visibility', 'hidden', 'important');
          host.style.setProperty('opacity', '0', 'important');
        } catch (_e) {}
      },
      _state: state,
      _host: host,
      _reapply() { applyHostStyles(host); },
    };

    render();

    // Self-healing watcher: SPA frameworks (YouTube/Polymer, React) can
    // rewrite documentElement's children. If our host disappears we re-mount
    // and re-pin styles. We deliberately do NOT observe the host's own style
    // attribute — re-asserting on every style mutation triggered glitchy
    // re-renders on busy pages and was the root cause of the "distorts when
    // the agent starts working" symptom.
    try {
      const reattachObserver = new MutationObserver(() => {
        if (!host.isConnected) {
          try { document.documentElement.appendChild(host); } catch (_e) {}
          applyHostStyles(host);
        }
      });
      reattachObserver.observe(document.documentElement, { childList: true, subtree: false });
    } catch (_e) {}

    return window.__cosmicOverlay;
  }

  function boot() { try { mount(); } catch (_e) {} }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot, { once: true });
  } else {
    boot();
  }
  // Retry briefly in case init-script ran before <html> existed.
  let tries = 0;
  const iv = setInterval(() => {
    tries += 1;
    if (window.__cosmicOverlay || tries > 30) { clearInterval(iv); return; }
    boot();
  }, 80);
})();
"""


_DEFAULT_METRICS = {
    "elapsed_sec": 0,
    "replay_actions": 0,
    "mimo_calls": 0,
    "dom_calls": 0,
    "llm_calls": 0,
    "mimo_calls_avoided": 0,
}


class DemoOverlayManager:
    """Owns mirror state for the in-page overlay and drives Playwright pages.

    Every method becomes a cheap no-op when `enabled` is False so callers can
    invoke this manager unconditionally.
    """

    INIT_SCRIPT = OVERLAY_INIT_SCRIPT

    def __init__(self, enabled: bool = False):
        self.enabled = bool(enabled)
        # Timer is intentionally deferred — it starts the first time the agent
        # actually acts on the browser (call start_timer()), not when the
        # browser process boots. Until then `elapsed_sec` reports 0.
        self._start_time: Optional[float] = None
        self._state: Dict[str, Any] = {
            "mode": "Idle",
            "supermemory": "Initializing",
            "workflow_id": None,
            "recall_score": None,
            "phase": "Waiting",
            "timeline": [],
            "metrics": dict(_DEFAULT_METRICS),
            "failure_patch": None,
        }
        self._pulse_request_ms: Optional[int] = None
        self._last_push_key: Optional[str] = None
        self._last_push_time: float = 0.0
        self._min_push_interval_sec: float = 0.75

    # ------------------------------------------------------------------
    # State mutation (Python-side mirror)
    # ------------------------------------------------------------------
    def set_state(self, **kwargs: Any) -> None:
        if not self.enabled:
            return
        for key, value in kwargs.items():
            if key == "metrics" and isinstance(value, dict):
                self._state["metrics"].update(value)
            else:
                self._state[key] = value

    def add_timeline(self, kind: str, label: str) -> None:
        if not self.enabled:
            return
        item = {"kind": str(kind or "indexed"), "label": str(label or "")}
        timeline: List[Dict[str, Any]] = self._state.get("timeline") or []
        timeline.append(item)
        self._state["timeline"] = timeline[-12:]

    def update_metrics(self, **metrics: Any) -> None:
        if not self.enabled:
            return
        for key, value in metrics.items():
            try:
                self._state["metrics"][key] = int(value)
            except (TypeError, ValueError):
                self._state["metrics"][key] = value

    def start_timer(self) -> None:
        """Begin counting elapsed time. Idempotent — only the first call wins."""
        if not self.enabled:
            return
        if self._start_time is None:
            self._start_time = time.time()

    def request_pulse(self, ms: int = 1500) -> None:
        if not self.enabled:
            return
        try:
            self._pulse_request_ms = max(0, int(ms))
        except (TypeError, ValueError):
            self._pulse_request_ms = 1500

    # ------------------------------------------------------------------
    # Browser-side pushing
    # ------------------------------------------------------------------
    async def install(self, context) -> None:
        if not self.enabled or context is None:
            return
        try:
            await context.add_init_script(self.INIT_SCRIPT)
        except Exception:
            # Overlay must never break the agent.
            pass

    def _snapshot(self) -> Dict[str, Any]:
        snapshot = {
            "mode": self._state.get("mode"),
            "supermemory": self._state.get("supermemory"),
            "workflow_id": self._state.get("workflow_id"),
            "recall_score": self._state.get("recall_score"),
            "phase": self._state.get("phase"),
            "timeline_replace": list(self._state.get("timeline") or []),
            "metrics": {
                **dict(_DEFAULT_METRICS),
                **(self._state.get("metrics") or {}),
                "elapsed_sec": (
                    int(time.time() - self._start_time)
                    if self._start_time is not None
                    else 0
                ),
            },
            "failure_patch": self._state.get("failure_patch"),
        }
        if self._pulse_request_ms:
            snapshot["pulse_ms"] = self._pulse_request_ms
            self._pulse_request_ms = None
        return snapshot

    def _snapshot_key(self, snapshot: Dict[str, Any]) -> str:
        key_snapshot = dict(snapshot)
        metrics = dict(key_snapshot.get("metrics") or {})
        # Elapsed time is intentionally not part of the dedupe key; otherwise
        # every capture repaint rewrites the overlay and looks jittery live.
        metrics.pop("elapsed_sec", None)
        key_snapshot["metrics"] = metrics
        return json.dumps(key_snapshot, sort_keys=True, ensure_ascii=False, default=str)

    async def push(self, page) -> None:
        if not self.enabled or page is None:
            return
        snapshot = self._snapshot()
        push_key = self._snapshot_key(snapshot)
        now = time.time()
        if push_key == self._last_push_key and (now - self._last_push_time) < self._min_push_interval_sec:
            return
        try:
            await page.evaluate(
                "(s) => { if (window.__cosmicOverlay) window.__cosmicOverlay.update(s); }",
                snapshot,
            )
            self._last_push_key = push_key
            self._last_push_time = now
        except Exception:
            pass

    async def update(self, page=None, *, pulse_ms: Optional[int] = None, **kwargs: Any) -> None:
        """Apply state mutations and (optionally) push to a page."""
        if not self.enabled:
            return
        timeline_append = kwargs.pop("timeline_append", None)
        metrics = kwargs.pop("metrics", None)
        for key, value in kwargs.items():
            self._state[key] = value
        if metrics:
            self._state["metrics"].update(metrics)
        if timeline_append:
            self.add_timeline(timeline_append.get("kind", "indexed"), timeline_append.get("label", ""))
        if pulse_ms is not None:
            self.request_pulse(pulse_ms)
        if page is not None:
            await self.push(page)

    # ------------------------------------------------------------------
    # Hide/restore around agent screenshots
    # ------------------------------------------------------------------
    async def hide_for_agent_capture(self, page) -> None:
        if not self.enabled or page is None:
            return
        try:
            await page.evaluate(
                "() => { if (window.__cosmicOverlay) window.__cosmicOverlay.hide(); }"
            )
        except Exception:
            pass

    async def restore_after_agent_capture(self, page) -> None:
        if not self.enabled or page is None:
            return
        try:
            await page.evaluate(
                "() => { if (window.__cosmicOverlay) window.__cosmicOverlay.show(); }"
            )
        except Exception:
            pass

    # Aliases for the spec wording
    async def show(self, page) -> None:
        await self.restore_after_agent_capture(page)

    async def hide(self, page) -> None:
        await self.hide_for_agent_capture(page)


__all__ = ["DemoOverlayManager", "OVERLAY_INIT_SCRIPT"]
