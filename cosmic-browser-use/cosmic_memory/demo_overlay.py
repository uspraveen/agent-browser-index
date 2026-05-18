"""Demo-only visual overlay for COSMIC Browser Memory.

This module is purely a presentation layer. It must never:
  * affect agent decisions or coordinates
  * intercept page input (pointer-events stays `none`)
  * appear in any screenshot consumed by MiMo / orchestrator

The Python `DemoOverlayManager` owns local mirror state and pushes it into
each Playwright page. The browser-side script lives inside an injected init
script and renders into a closed-style shadow DOM so page CSS cannot reach it
and so it cannot mutate the host document's layout.

Activation is gated by the `--demo-overlay` CLI flag in main.py; when the
flag is off, every method becomes an inexpensive no-op.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional


# JavaScript injected once per page (via context.add_init_script). It is
# defensive: if executed before <html> exists, it retries until the
# documentElement is available. The overlay attaches into a shadow root so
# page styles can't bleed in.
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
    // Defensive: even if the panel inside the shadow loses its computed
    // height, the host itself stays clearly visible.
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

    const shadow = host.attachShadow({ mode: 'open' });
    const style = document.createElement('style');
    style.textContent = `
      :host {
        /* Reassert inherited properties inside the shadow so even if the
           page's CSS or our own watchdog races, text and fonts stay set. */
        color: #eef2f9 !important;
        font: 12px/1.45 Bahnschrift, "Avenir Next Condensed", "Segoe UI Variable Display", "Helvetica Neue", Arial, sans-serif !important;
        line-height: 1.45;
      }
      * { box-sizing: border-box; }
      .panel {
        position: relative;
        /* True black frosted glass — neutral tint, no blue cast. */
        backdrop-filter: blur(22px) saturate(140%);
        -webkit-backdrop-filter: blur(22px) saturate(140%);
        background:
          linear-gradient(180deg, rgba(8,8,10,0.62) 0%, rgba(0,0,0,0.66) 100%);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 14px;
        padding: 14px 16px;
        box-shadow:
          0 18px 48px rgba(0,0,0,0.45),
          inset 0 1px 0 rgba(255,255,255,0.06),
          inset 0 0 0 1px rgba(255,255,255,0.03);
        color: #eef2f9;
        /* COSMIC sign-in font stack (see Cosmic-OS/src/settings.css). */
        font: 12px/1.45 Bahnschrift, "Avenir Next Condensed", "Segoe UI Variable Display", "Helvetica Neue", Arial, sans-serif;
        min-height: 220px;
        max-height: 92vh;
        overflow: hidden;
        transform: translateZ(0);
        transition: none;
      }
      /* Soft highlight strip at the top — sells the glassy reflection. */
      .panel::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 36%;
        background: linear-gradient(180deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0) 100%);
        pointer-events: none;
        border-top-left-radius: 14px;
        border-top-right-radius: 14px;
      }
      .panel.dim { opacity: 0.92; }
      .header { display: flex; align-items: center; gap: 8px; margin-bottom: 12px; }
      .brand {
        /* Mirror the COSMIC wordmark: heavy tracking + small caps look. */
        font-family: Bahnschrift, "Avenir Next Condensed", "Segoe UI Variable Display", "Helvetica Neue", Arial, sans-serif;
        font-weight: 600;
        font-size: 11px;
        letter-spacing: 0.32em;
        color: #eef2f9;
        text-transform: uppercase;
      }
      .brand .dot {
        display: inline-block; width: 6px; height: 6px; border-radius: 999px;
        background: #ffffff;
        margin-right: 8px; vertical-align: middle;
        box-shadow: 0 0 8px rgba(255,255,255,0.45);
      }
      .badge {
        font-family: Bahnschrift, "Avenir Next Condensed", "Segoe UI Variable Display", sans-serif;
        font-size: 10px; padding: 2px 8px; border-radius: 999px;
        background: rgba(255,255,255,0.08); color: #eef2f9;
        border: 1px solid rgba(255,255,255,0.16);
        letter-spacing: 0.12em; text-transform: uppercase;
        margin-left: auto;
      }
      .pulse-ring { position: relative; }
      .pulse-ring::after {
        content: ''; position: absolute; inset: -3px;
        border-radius: inherit; pointer-events: none;
        box-shadow: 0 0 0 1px rgba(255,255,255,0.20), 0 0 14px rgba(255,255,255,0.10);
      }
      .row { display: flex; justify-content: space-between; gap: 10px; padding: 3px 0; }
      .row .k { color: #9aa3b2; }
      .row .v {
        color: #eef2f9; font-weight: 500; max-width: 62%;
        text-align: right; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
      }
      .section { margin-top: 10px; padding-top: 8px; border-top: 1px dashed rgba(255,255,255,0.08); }
      .section-title {
        font-size: 10px; letter-spacing: 0.18em; text-transform: uppercase;
        color: #9aa3b2; margin-bottom: 6px;
      }
      .phase {
        display: inline-flex; align-items: center; gap: 6px;
        padding: 4px 9px; border-radius: 8px;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.14);
        color: #ffffff; font-weight: 500;
      }
      .phase .dot { width: 6px; height: 6px; border-radius: 999px; background: #ffffff; }
      .timeline { display: flex; flex-direction: column; gap: 4px; }
      .tl-item { display: flex; align-items: center; gap: 8px; min-height: 18px; }
      .tl-item .marker { width: 6px; height: 6px; border-radius: 999px; background: rgba(255,255,255,0.7); opacity: 0.9; flex: none; }
      .tl-item .label { color: #d8dde6; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
      .tl-item.recall   .marker { background: #c8b8ff; }
      .tl-item.indexed  .marker { background: #ffffff; }
      .tl-item.navigate .marker { background: #b5e3c4; }
      .tl-item.checkpoint .marker { background: #ffd99a; }
      .tl-item.live     .marker { background: #ffb0bf; }
      .tl-item.saved    .marker { background: #9be8b6; box-shadow: 0 0 6px rgba(155,232,182,0.45); }
      .metrics { display: grid; grid-template-columns: 1fr 1fr; gap: 6px 12px; }
      .metric .k { font-size: 10px; color: #9aa3b2; text-transform: uppercase; letter-spacing: 0.18em; }
      .metric .v { font-size: 14px; color: #eef2f9; font-weight: 600; }
      .metric .v.ok { color: #9be8b6; }
      .patch {
        margin-top: 8px; padding: 6px 8px; border-radius: 8px;
        background: rgba(255,200,120,0.08);
        border: 1px solid rgba(255,200,120,0.22);
        color: #ffd9a8; font-size: 11px;
      }
      .ok { color: #9be8b6; }
      .warn { color: #ffd99a; }
      .muted { color: #9aa3b2; }
    `;
    shadow.appendChild(style);

    const root = document.createElement('div');
    root.className = 'panel';
    // Inline panel chrome so the glassy shell doesn't depend on the shadow
    // stylesheet being applied. Matches the .panel CSS rule above.
    root.setAttribute(
      'style',
      [
        'position:relative',
        'backdrop-filter:blur(22px) saturate(140%)',
        '-webkit-backdrop-filter:blur(22px) saturate(140%)',
        'background:linear-gradient(180deg, rgba(8,8,10,0.62) 0%, rgba(0,0,0,0.66) 100%)',
        'border:1px solid rgba(255,255,255,0.10)',
        'border-radius:14px',
        'padding:14px 16px',
        'box-shadow:0 18px 48px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.06), inset 0 0 0 1px rgba(255,255,255,0.03)',
        'color:#eef2f9',
        'font:12px/1.45 Bahnschrift, "Avenir Next Condensed", "Segoe UI Variable Display", "Helvetica Neue", Arial, sans-serif',
        'min-height:220px',
        'max-height:92vh',
        'overflow:hidden',
        'transform:translateZ(0)',
        'transition:none',
        'box-sizing:border-box',
        'width:100%',
      ].join(';'),
    );
    shadow.appendChild(root);

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

    function escapeHtml(s) {
      return String(s == null ? '' : s)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
    }
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
    // child content vanished — almost certainly a cascade race against
    // the shadow <style> element. Inline element styles are immune to
    // that and cannot be overridden from outside the shadow tree.
    // ------------------------------------------------------------------
    const FONT_STACK = 'Bahnschrift, "Avenir Next Condensed", "Segoe UI Variable Display", "Helvetica Neue", Arial, sans-serif';
    const S = {
      header:      'display:flex;align-items:center;gap:8px;margin:0 0 12px 0;color:#eef2f9;font-family:' + FONT_STACK + ';',
      brand:       'font-family:' + FONT_STACK + ';font-weight:600;font-size:11px;letter-spacing:0.32em;color:#eef2f9;text-transform:uppercase;display:inline-flex;align-items:center;',
      brandDot:    'display:inline-block;width:6px;height:6px;border-radius:999px;background:#ffffff;margin-right:8px;box-shadow:0 0 8px rgba(255,255,255,0.45);',
      badge:       'margin-left:auto;font-family:' + FONT_STACK + ';font-size:10px;padding:2px 8px;border-radius:999px;background:rgba(255,255,255,0.08);color:#eef2f9;border:1px solid rgba(255,255,255,0.16);letter-spacing:0.12em;text-transform:uppercase;',
      row:         'display:flex;justify-content:space-between;gap:10px;padding:3px 0;font-family:' + FONT_STACK + ';',
      rowK:        'color:#9aa3b2;font-size:12px;',
      rowV:        'color:#eef2f9;font-weight:500;font-size:12px;max-width:62%;text-align:right;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;',
      rowVOk:      'color:#9be8b6;font-weight:500;font-size:12px;max-width:62%;text-align:right;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;',
      rowVMuted:   'color:#9aa3b2;font-weight:500;font-size:12px;max-width:62%;text-align:right;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;',
      section:     'margin-top:10px;padding-top:8px;border-top:1px dashed rgba(255,255,255,0.08);font-family:' + FONT_STACK + ';',
      sectionTitle:'font-size:10px;letter-spacing:0.18em;text-transform:uppercase;color:#9aa3b2;margin:0 0 6px 0;',
      phase:       'display:inline-flex;align-items:center;gap:6px;padding:4px 9px;border-radius:8px;background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.14);color:#ffffff;font-weight:500;font-size:12px;font-family:' + FONT_STACK + ';',
      phaseDot:    'width:6px;height:6px;border-radius:999px;background:#ffffff;display:inline-block;',
      timeline:    'display:flex;flex-direction:column;gap:4px;',
      tlItem:      'display:flex;align-items:center;gap:8px;min-height:18px;font-size:12px;',
      tlMarker:    'width:6px;height:6px;border-radius:999px;flex:none;',
      tlLabel:     'color:#d8dde6;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;',
      tlLabelMuted:'color:#9aa3b2;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;',
      metrics:     'display:grid;grid-template-columns:1fr 1fr;gap:6px 12px;',
      metricK:     'font-size:10px;color:#9aa3b2;text-transform:uppercase;letter-spacing:0.18em;font-family:' + FONT_STACK + ';',
      metricV:     'font-size:14px;color:#eef2f9;font-weight:600;font-family:' + FONT_STACK + ';',
      metricVOk:   'font-size:14px;color:#9be8b6;font-weight:600;font-family:' + FONT_STACK + ';',
      patch:       'margin-top:8px;padding:6px 8px;border-radius:8px;background:rgba(255,200,120,0.08);border:1px solid rgba(255,200,120,0.22);color:#ffd9a8;font-size:11px;font-family:' + FONT_STACK + ';',
    };
    const MARKER_COLORS = {
      recall:    '#c8b8ff',
      indexed:   '#ffffff',
      navigate:  '#b5e3c4',
      checkpoint:'#ffd99a',
      live:      '#ffb0bf',
      saved:     '#9be8b6',
    };

    function render() {
      const s = state;
      const supermemoryRaw = String(s.supermemory || '');
      const supermemoryOk = /^connected/i.test(supermemoryRaw);
      const wf = shortWorkflow(s.workflow_id);
      const score = (s.recall_score == null) ? '—' : Number(s.recall_score).toFixed(2);
      const m = s.metrics || {};

      const tlItems = (s.timeline || []).slice(-5);
      const timelineHtml = tlItems.length
        ? tlItems.map(item => {
            const kind = String(item.kind || 'indexed').toLowerCase();
            const markerColor = MARKER_COLORS[kind] || '#ffffff';
            const markerStyle = S.tlMarker + 'background:' + markerColor + ';';
            return '<div style="' + S.tlItem + '">'
              +   '<span style="' + markerStyle + '"></span>'
              +   '<span style="' + S.tlLabel + '">' + escapeHtml(item.label || '') + '</span>'
              + '</div>';
          }).join('')
        : '<div style="' + S.tlItem + '"><span style="' + S.tlMarker + 'background:rgba(255,255,255,0.5);"></span><span style="' + S.tlLabelMuted + '">No replay actions yet</span></div>';

      const patchHtml = s.failure_patch
        ? '<div style="' + S.patch + '">⚡ ' + escapeHtml(s.failure_patch) + '</div>'
        : '';

      const supermemoryValueStyle = supermemoryOk ? S.rowVOk : S.rowVMuted;

      root.innerHTML =
        '<div style="' + S.header + '">'
        +   '<span style="' + S.brand + '"><span style="' + S.brandDot + '"></span>Cosmic Browser Memory</span>'
        +   '<span style="' + S.badge + '">' + escapeHtml(s.mode || 'Idle') + '</span>'
        + '</div>'
        + '<div style="' + S.row + '"><span style="' + S.rowK + '">Supermemory</span>'
        +   '<span style="' + supermemoryValueStyle + '">' + escapeHtml(supermemoryRaw || '—') + '</span></div>'
        + '<div style="' + S.row + '"><span style="' + S.rowK + '">Workflow</span>'
        +   '<span style="' + S.rowV + '" title="' + escapeHtml(s.workflow_id || '') + '">' + escapeHtml(wf) + '</span></div>'
        + '<div style="' + S.row + '"><span style="' + S.rowK + '">Recall score</span>'
        +   '<span style="' + S.rowV + '">' + escapeHtml(score) + '</span></div>'
        + '<div style="' + S.section + '">'
        +   '<div style="' + S.sectionTitle + '">Replay phase</div>'
        +   '<div style="' + S.phase + '"><span style="' + S.phaseDot + '"></span>' + escapeHtml(s.phase || '—') + '</div>'
        + '</div>'
        + '<div style="' + S.section + '">'
        +   '<div style="' + S.sectionTitle + '">Action timeline</div>'
        +   '<div style="' + S.timeline + '">' + timelineHtml + '</div>'
        + '</div>'
        + '<div style="' + S.section + '">'
        +   '<div style="' + S.sectionTitle + '">Metrics</div>'
        +   '<div style="' + S.metrics + '">'
        +     '<div><div style="' + S.metricK + '">Elapsed</div><div style="' + S.metricV + '">' + fmtTime(m.elapsed_sec) + '</div></div>'
        +     '<div><div style="' + S.metricK + '">Replay actions</div><div style="' + S.metricV + '">' + escapeHtml(m.replay_actions || 0) + '</div></div>'
        +     '<div><div style="' + S.metricK + '">MiMo avoided</div><div style="' + S.metricVOk + '">' + escapeHtml(m.mimo_calls_avoided || 0) + '</div></div>'
        +     '<div><div style="' + S.metricK + '">MiMo calls</div><div style="' + S.metricV + '">' + escapeHtml(m.mimo_calls || 0) + '</div></div>'
        +     '<div><div style="' + S.metricK + '">DOM calls</div><div style="' + S.metricV + '">' + escapeHtml(m.dom_calls || 0) + '</div></div>'
        +     '<div><div style="' + S.metricK + '">LLM calls</div><div style="' + S.metricV + '">' + escapeHtml(m.llm_calls || 0) + '</div></div>'
        +   '</div>'
        + '</div>'
        + patchHtml;
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
