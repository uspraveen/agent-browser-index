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
import random
import imagehash
import time
import re
import json
import subprocess
import sys
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Callable, Awaitable
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
from browser_memory.coordinates import replay_coordinates
from browser_memory.demo_overlay import DemoOverlayManager
from browser_memory.cursor_overlay import CursorOverlayManager
import os
from dotenv import load_dotenv

load_dotenv()

# Known SSO auth domains, keyed by provider name as it'd appear in an action
# description (e.g. "Continue with Google button"). Used to detect when a
# click landed on the WRONG provider's button — e.g. described as "Google"
# but the resulting URL is a Microsoft OAuth domain. Page-changed alone
# isn't enough to call a click "successful" if it picked the wrong target.
_SSO_PROVIDER_DOMAINS = {
    "google": ("accounts.google.com", "google.com/o/oauth", "googleusercontent.com"),
    "microsoft": ("login.live.com", "login.microsoftonline.com", "login.windows.net"),
    "apple": ("appleid.apple.com",),
    "facebook": ("facebook.com/login", "facebook.com/dialog/oauth"),
    "github": ("github.com/login",),
}

# Resolves the visible label of a form field (the text a human would read).
# Injected into the DomType/SelectOption JS so an action result can NAME the
# field its text actually landed in — a wrong-target type is silent by
# construction (the value lands somewhere, the page "changed"), and the model
# never re-derives intent from page state afterwards. This is an expression,
# not a statement block: embed as `(el) => ...` and call it.
_FIELD_LABEL_JS = """
(el) => {
  if (!el || el.nodeType !== 1) return '';
  try {
    const labelledby = el.getAttribute('aria-labelledby');
    if (labelledby) {
      const txt = labelledby.split(/\\s+/).map((id) => document.getElementById(id)).filter(Boolean)
        .map((n) => (n.textContent || '').trim()).join(' ').trim();
      if (txt) return txt.slice(0, 120);
    }
    if (el.labels && el.labels.length && el.labels[0].textContent) {
      const clone = el.labels[0].cloneNode(true);
      clone.querySelectorAll('input, textarea, select').forEach((n) => n.remove());
      const txt = (clone.textContent || '').trim();
      if (txt) return txt.slice(0, 120);
    }
    const ariaLabel = el.getAttribute('aria-label');
    if (ariaLabel && ariaLabel.trim()) return ariaLabel.trim().slice(0, 120);
    const wrap = el.closest && el.closest('label');
    if (wrap) {
      const clone = wrap.cloneNode(true);
      clone.querySelectorAll('input, textarea, select').forEach((n) => n.remove());
      const txt = (clone.textContent || '').trim();
      if (txt) return txt.slice(0, 120);
    }
    if (el.id) {
      try {
        const forLabel = document.querySelector('label[for="' + CSS.escape(el.id) + '"]');
        if (forLabel) {
          const txt = (forLabel.textContent || '').trim();
          if (txt) return txt.slice(0, 120);
        }
      } catch (e) {}
    }
    if (el.name) return String(el.name).slice(0, 120);
    const placeholder = el.getAttribute && el.getAttribute('placeholder');
    if (placeholder && placeholder.trim()) return placeholder.trim().slice(0, 120);
  } catch (e) {}
  return '';
}
"""

# Read back what the keyboard actually landed in: focus was set inside the
# target frame, so document.activeElement there is the field that received
# every keystroke — regardless of which selector produced it.
_TYPE_ECHO_JS = """
() => {
  const el = document.activeElement;
  if (!el || (el.tagName !== 'INPUT' && el.tagName !== 'TEXTAREA' && !el.isContentEditable)) return null;
  return {label: (%s)(el), value: String(el.value == null ? '' : el.value)};
}
""" % _FIELD_LABEL_JS

# Refuse-to-guess guard: a selector matching several visible fields is how one
# field's value ends up in another (same placeholder reused across a form).
# Names + current values of the matches go back to the model so it can
# disambiguate in one step instead of experimenting.
def _ambiguous_type_error(selector: str, count: int, fields: List[Dict[str, Any]]) -> str:
    lines = []
    for f in list(fields or [])[:8]:
        label = str(f.get("label") or "").strip() or "(unlabeled)"
        value = str(f.get("value") or "").strip()
        shown = f", currently '{value[:40]}'" if value else ""
        lines.append(f"  - '{label}'{shown}")
    if count > len(lines):
        lines.append(f"  - ... and {count - len(lines)} more")
    listing = "\n".join(lines) if lines else "  (could not read the matched fields)"
    return (
        f"Ambiguous selector: matched {count} visible fields — refusing to guess which one you meant, "
        f"because typing into the wrong one silently destroys whatever it held. Matched fields:\n{listing}\n"
        "Re-issue with a selector matching exactly one visible field: anchor it to the field's label text "
        "(e.g. input[placeholder='Type here...']:below(:text(\"Legal Name\"))), its id, or its name attribute. "
        "If unsure which selector is right, DOMExtract the form region first and read the field's actual markup."
    )


def _type_echo_description(
    selector: str, label: str, text: str, previous_value: str, frame_note: str = "", is_secret: bool = False
) -> Tuple[str, Optional[str]]:
    """Description for a completed type action, plus an overwrite warning.

    The echo names the field the text landed in (read back from the focused
    element, not inferred from the selector) and shouts when it replaced a
    value that was already there — the one signal that catches 'I filled the
    name field, then a later step quietly rewrote it with something else'.
    Password fills show only the mask: the trace proves the field was
    addressed without recording what it holds.
    """
    base = f"Typed into visible element for selector{frame_note}: {selector}: '{_secret_display(text, is_secret, 120)}'"
    clean_label = str(label or "").strip()
    if clean_label:
        base += f" — field labeled '{clean_label[:80]}'"
    warning = None if is_secret else _overwrite_warning(text, previous_value)
    if warning:
        base += f" (WARNING: {warning} — if this is not the field you meant, refill the correct field and restore this one)"
    return base, warning


def _overwrite_warning(text: str, previous_value: str) -> Optional[str]:
    """Warning text when a type replaces a non-empty, different value — shared
    by the selector path and the @ref path. None when the field was empty or
    already held exactly this text."""
    prev = str(previous_value or "").strip()
    if prev and prev != str(text or "").strip():
        return f"overwrote the field's existing value '{prev[:60]}'"
    return None


def _secret_display(value: Any, is_secret: bool, cap: int = 60) -> str:
    """What a trace may show for a field's contents. Password values are the
    one class of typed text that must never appear in a description, echo, or
    log — the mask proves the field was filled without naming what filled it."""
    v = str(value or "")
    if is_secret:
        return "********" if v.strip() else ""
    return v[:cap]


# ── DOMSnapshot: the interactive-element map ─────────────────────────────
# One text pull that replaces per-action visual grounding on structured
# pages: every clickable/typeable control, listed with role, visible name and
# state, addressed by a stable @e ref. Perception is a peer tool here, not a
# replacement — vision still grounds anything the snapshot can't describe,
# and the existing verification layer still judges every action's outcome.

_SNAPSHOT_INTERACTIVE_CSS = (
    "a[href], button, input, select, textarea, summary, [contenteditable='true'], [contenteditable=''], "
    "[role='button'], [role='link'], [role='checkbox'], [role='radio'], [role='combobox'], "
    "[role='listbox'], [role='option'], [role='tab'], [role='menuitem'], [role='switch'], [role='textbox'], "
    "[role='searchbox'], [role='slider'], [role='spinbutton']"
)

# Runs per frame. Returns the frame's VISIBLE, ENABLED interactive elements in
# DOM order (the order @ref nth-indexing relies on at act time), each with the
# fingerprint Snapshot actions re-check before acting. Password values are
# masked here, at the source — they must never reach the model or the trace.
#
# Collect and re-check share ONE helper block by construction: the fingerprint
# compares the name computed at snapshot time against the name computed at act
# time, so the two must not drift — a name rule that only one side knows is a
# false "stale" on every act.
_SNAPSHOT_HELPERS_JS = """
  const clean = (s) => String(s == null ? "" : s).replace(/\\s+/g, " ").trim();
  const isVisible = (el) => {
    const style = window.getComputedStyle(el);
    if (!style || style.display === "none" || style.visibility === "hidden" || Number(style.opacity) === 0) return false;
    if (el.closest("[hidden], [aria-hidden='true']")) return false;
    const rect = el.getBoundingClientRect();
    return rect.width > 0 && rect.height > 0;
  };
  const labelFor = (""" + _FIELD_LABEL_JS + """);
  const implicitRole = (el) => {
    const explicit = el.getAttribute("role");
    if (explicit) return explicit;
    const t = el.tagName.toLowerCase();
    if (t === "a") return el.hasAttribute("href") ? "link" : "";
    if (t === "button" || t === "summary") return "button";
    if (t === "select") return "combobox";
    if (t === "textarea") return "textbox";
    if (t === "option") return "option";
    if (el.isContentEditable) return "textbox";
    if (t === "input") {
      const ty = (el.type || "text").toLowerCase();
      if (ty === "checkbox") return "checkbox";
      if (ty === "radio") return "radio";
      if (ty === "file") return "button";
      if (ty === "button" || ty === "submit" || ty === "reset" || ty === "image") return "button";
      return "textbox";
    }
    return "";
  };
  const nameFor = (el) => {
    let n = labelFor(el);
    if (!n) {
      const t = el.tagName.toLowerCase();
      if (t === "button" || t === "a" || t === "summary" || t === "option" || el.getAttribute("role")) {
        // innerText, not textContent: rendered text only, so <style> blocks
        // inside anchors (Google hides them there) never become a name.
        n = clean(el.innerText || el.textContent);
      }
      else if (t === "input" && ["button", "submit", "reset"].includes((el.type || "").toLowerCase())) n = clean(el.value);
      if (!n && (t === "input" || t === "textarea")) n = clean(el.getAttribute("title") || el.getAttribute("aria-placeholder") || el.getAttribute("placeholder") || "");
    }
    return n.slice(0, 60);
  };
"""

_SNAPSHOT_COLLECT_JS = """
(args) => {
  const css = args.css;
  const cap = args.cap;
/*HELPERS*/
  let elements = [];
  try { elements = Array.from(document.querySelectorAll(css)); } catch (e) { return {error: "bad selector"}; }
  const visible = elements.filter((el) => isVisible(el) && !(el.disabled || el.getAttribute("aria-disabled") === "true"));
  const entries = [];
  for (const el of visible) {
    if (entries.length >= cap) { entries.push({truncated: true}); break; }
    const role = implicitRole(el);
    if (!role) continue;
    const t = el.tagName.toLowerCase();
    const ty = (el.getAttribute("type") || "").toLowerCase();
    const entry = {
      tag: t,
      id: el.id || "",
      role: role,
      name: nameFor(el),
      value: "",
      checked: null,
      selected: "",
    };
    if (role === "textbox") entry.value = (ty === "password") ? "********" : clean(el.value).slice(0, 40);
    if (role === "checkbox" || role === "radio" || role === "switch") entry.checked = !!el.checked;
    if (role === "combobox" && t === "select") {
      const sel = el.selectedOptions && el.selectedOptions[0];
      entry.selected = sel ? clean(sel.textContent).slice(0, 40) : "";
    }
    entries.push(entry);
  }
  return {count: entries.length, truncated: entries.length > 0 && !!entries[entries.length - 1].truncated, entries};
}
""".replace("/*HELPERS*/", _SNAPSHOT_HELPERS_JS)

# Act-time re-resolution: re-run the same query, filter the same way, land on
# the same nth — then prove it is the same element via the fingerprint. The
# fingerprint deliberately excludes the value: typing legitimately changes
# values, and a guard that fired on every keystroke would force a re-snapshot
# between every pair of form fields.
_SNAPSHOT_RECHECK_JS = """
(args) => {
  const css = args.css;
  const nth = args.nth;
  const want = args.fingerprint;
/*HELPERS*/
  let elements = [];
  try { elements = Array.from(document.querySelectorAll(css)); } catch (e) { return {ok: false, reason: "bad selector"}; }
  const visible = elements.filter((el) => isVisible(el) && !(el.disabled || el.getAttribute("aria-disabled") === "true"));
  if (nth >= visible.length) return {ok: false, reason: "gone", count: visible.length};
  const el = visible[nth];
  const now = {tag: el.tagName.toLowerCase(), id: el.id || "", name: nameFor(el)};
  const same = now.tag === want.tag && (now.id || "") === (want.id || "") && now.name === (want.name || "");
  if (!same) return {ok: false, reason: "changed", count: visible.length, now: now};
  return {
    ok: true,
    reason: "",
    visible: isVisible(el),
    // Playwright nth() indexes ALL matches in document order, not just the
    // visible ones this snapshot numbered — hand back the unfiltered index.
    all_index: elements.indexOf(el),
    now: now,
  };
}
""".replace("/*HELPERS*/", _SNAPSHOT_HELPERS_JS)


def snapshot_fingerprint(entry: Dict[str, Any]) -> Dict[str, str]:
    """Identity triple a @ref re-verifies before acting (never the value)."""
    return {
        "tag": str(entry.get("tag") or ""),
        "id": str(entry.get("id") or ""),
        "name": str(entry.get("name") or ""),
    }


def fingerprint_matches(before: Dict[str, Any], after: Dict[str, Any]) -> bool:
    """True when the element a ref now points at is provably the same one the
    snapshot described."""
    return (
        str(before.get("tag") or "") == str(after.get("tag") or "")
        and str(before.get("id") or "") == str(after.get("id") or "")
        and str(before.get("name") or "") == str(after.get("name") or "")
    )


def parse_ref(ref: Any) -> Optional[str]:
    """Validate an @e reference from the model ('@e12' → '@e12'); None when
    the shape is wrong."""
    import re as _re
    m = _re.fullmatch(r"@e(\d+)", str(ref or "").strip())
    return m.group(0) if m else None


def format_snapshot_lines(entries: List[Dict[str, Any]], start_ref: int = 1) -> List[str]:
    """Render collected snapshot entries as the @e map text the model reads."""
    lines = []
    for i, e in enumerate(entries):
        if e.get("truncated"):
            lines.append("... (snapshot truncated at cap — call DOMSnapshot again after scrolling)")
            break
        ref = f"@e{start_ref + i}"
        role = str(e.get("role") or "element")
        name = str(e.get("name") or "").strip()
        bits = [f"{ref} {role}"]
        if name:
            bits.append(f'"{name}"')
        if e.get("value"):
            bits.append(f"value='{e['value']}'")
        if e.get("checked") is not None:
            bits.append("checked" if e.get("checked") else "unchecked")
        if e.get("selected"):
            bits.append(f"selected='{e['selected']}'")
        lines.append(" ".join(bits))
    return lines


def mask_snapshot_value(value: Any, is_password: bool) -> str:
    """Password values never leave the page as text — the snapshot shows the
    mask, not the secret."""
    v = str(value or "").strip()
    if is_password:
        return "********" if v else ""
    return v[:40]


class _AmbiguousTargetError(Exception):
    """A type/select selector matched more than one visible target. Carries
    the model-facing refusal message (with the matched fields' labels) as its
    str()."""


class _SnapshotStaleError(Exception):
    """An @e ref could not be resolved to a provably-live element — unknown
    ref, superseded snapshot, re-rendered page, fingerprint mismatch, or the
    element hidden. Carries the model-facing reason as its str()."""

# Structural DOM fingerprint captured with every BrowserState. Deliberately
# excludes raw text (passwords/PII must not leak into logs or memory): input
# values are represented only as length + polynomial checksum. Together with
# URL/title/scroll/ready-state it lets verify_action distinguish "pixels
# changed because an ad refreshed" from "the page actually changed" (focus
# moved, input values changed, body shape changed) — and catches real changes
# that pixels hide (masked password fields).
_DOM_SIGNATURE_JS = """
() => {
  try {
    const ae = document.activeElement;
    const focus = ae && ae !== document.body ? String(ae.tagName || '') + (ae.id ? '#' + ae.id : '') : '';
    let visibleInputs = 0;
    let valueSig = '';
    let selectSig = '';
    // Collect inputs/selects from the light DOM AND open shadow roots.
    const inputs = [];
    const selects = [];
    const walk = (root) => {
      try {
        root.querySelectorAll('input, textarea').forEach((el) => inputs.push(el));
        root.querySelectorAll('select').forEach((el) => selects.push(el));
        const hosts = root.querySelectorAll('*');
        for (const el of hosts) {
          if (el.shadowRoot) walk(el.shadowRoot);
        }
      } catch (e0) {}
    };
    walk(document);
    try {
      const limit = Math.min(inputs.length || 0, 80);
      for (let i = 0; i < limit; i++) {
        const el = inputs[i];
        let r = null;
        try { r = el.getBoundingClientRect(); } catch (e) { continue; }
        if (!r || r.width <= 0 || r.height <= 0) continue;
        visibleInputs++;
        let v = '';
        try { v = String(el.value == null ? '' : el.value); } catch (e2) { v = ''; }
        v = v.trim();
        if (v) {
          let h = 0;
          for (let j = 0; j < v.length; j++) { h = ((h * 31) + v.charCodeAt(j)) & 0x7fffffff; }
          valueSig += '|' + v.length + ':' + h.toString(36);
        }
      }
    } catch (e3) {}
    try {
      const slim = Math.min(selects.length || 0, 40);
      for (let i = 0; i < slim; i++) {
        const sel = selects[i];
        let r = null;
        try { r = sel.getBoundingClientRect(); } catch (e6) { continue; }
        if (!r || r.width <= 0 || r.height <= 0) continue;
        let v = '';
        try { v = String(sel.value == null ? '' : sel.value); } catch (e8) { v = ''; }
        let h = 0;
        for (let j = 0; j < v.length; j++) { h = ((h * 31) + v.charCodeAt(j)) & 0x7fffffff; }
        selectSig += '|' + (sel.selectedIndex >= 0 ? sel.selectedIndex : -1) + ':' + v.length + ':' + h.toString(36);
      }
    } catch (e8) {}
    let bodyKids = -1;
    try { bodyKids = document.body ? document.body.childElementCount : -1; } catch (e4) {}
    let scrollH = -1;
    try { scrollH = document.documentElement ? document.documentElement.scrollHeight : -1; } catch (e5) {}
    return focus + '|' + bodyKids + '|' + visibleInputs + '|' + valueSig + '|' + selectSig + '|' + scrollH;
  } catch (err) { return ''; }
}
"""
# Deterministic dropdown detection, evaluated with every capture_state (DOM
# mode only), once per frame (main frame + iframes). Native <select> option
# popups render in the browser's OS UI layer, NOT in the page — they are
# invisible to screenshots, which makes them a hard failure point for vision
# grounding. This scan surfaces interactive dropdown-ish controls (native
# selects with their option labels, plus custom combobox/haspopup widgets) so
# the orchestrator can prefer DOM tools for them. Pierces open shadow roots;
# bounded (max 8 items per frame) and cheap (single evaluate per frame).
_DROPDOWN_SCAN_JS = """
() => {
  const out = [];
  try {
    const visible = (el) => {
      try {
        const r = el.getBoundingClientRect();
        if (r.width <= 0 || r.height <= 0) return false;
        if (r.bottom <= 0 || r.top >= (window.innerHeight || 100000)) return false;
        const st = getComputedStyle(el);
        return st.visibility !== 'hidden' && st.display !== 'none';
      } catch (e) { return false; }
    };
    // Collect candidates from this root AND all open shadow roots.
    const candidates = [];
    const walk = (root) => {
      try {
        root.querySelectorAll('select, [role="combobox"], [role="listbox"], [aria-haspopup], [data-toggle="dropdown"], [class*="dropdown" i], [class*="select2" i], [class*="react-select" i], [class*="multiselect" i]').forEach((el) => candidates.push(el));
        const hosts = root.querySelectorAll('*');
        for (const el of hosts) {
          if (el.shadowRoot) walk(el.shadowRoot);
        }
      } catch (e0) {}
    };
    walk(document);
    const seen = new Set();
    let customs = 0;
    for (const el of candidates) {
      if (out.length >= 8) break;
      if (!el || seen.has(el)) continue;
      seen.add(el);
      const tag = String(el.tagName || '').toLowerCase();
      if (tag === 'select') {
        if (el.disabled || !visible(el)) continue;
        const count = el.options ? el.options.length : 0;
        if (!count) continue;
        const opts = [];
        const lim = Math.min(count, 4);
        for (let i = 0; i < lim; i++) {
          const t = String(el.options[i].text || '').trim();
          if (t) opts.push(t.slice(0, 40));
        }
        const idPart = el.id ? ('#' + el.id) : (el.name ? ('[name=' + el.name + ']') : '');
        out.push('select' + idPart + (el.multiple ? '(multi)' : '') + ' (' + count + ' options' + (opts.length ? ': ' + opts.join(' | ') : '') + (count > 4 ? ' | ...' : '') + ')');
      } else {
        if (customs >= 4 || !visible(el)) continue;
        const haspopup = el.getAttribute('aria-haspopup');
        if (haspopup && !['listbox', 'menu', 'true', 'grid', 'tree'].includes(String(haspopup).toLowerCase())) continue;
        if (el.closest && el.closest('select')) continue;
        seen.add(el);
        customs++;
        const role = el.getAttribute('role') || '';
        const expanded = el.getAttribute('aria-expanded');
        let cls = '';
        try { cls = String(el.className || '').trim(); } catch (e5) { cls = ''; }
        let txt = '';
        try { txt = (el.textContent || '').trim().slice(0, 40); } catch (e6) { txt = ''; }
        out.push('custom dropdown <' + tag + (role ? ' role=' + role : '') + (cls ? ' class~' + cls.split(/\\s+/)[0] : '') + (expanded != null ? ' expanded=' + expanded : '') + (txt ? ' text="' + txt + '"' : '') + '>');
      }
    }
    return JSON.stringify(out);
  } catch (err) { return '[]'; }
}
"""

# verification_hint grammar the verifier understands (key(value)). The
# orchestrator prompt documents url_contains(...); the other keys are
# supported symmetrically. Free-form hints (e.g. "answer_saved") do not match
# the grammar and are ignored by the evaluator, matching legacy behavior.
_SEMANTIC_HINT_RE = re.compile(
    r"^\s*(url_contains|url_equals|title_contains|element_exists|element_visible)\s*\(\s*(.+?)\s*\)\s*$",
    re.IGNORECASE,
)

# Actions whose primary intended observable effect is a scroll-position change.
_SCROLL_ACTION_TYPES = {ActionType.VISUAL_SCROLL}


def _find_chrome_binary() -> Optional[str]:
    """Return path to the real Google Chrome binary, or None if not found."""
    import shutil

    # Honour explicit override first.
    env_bin = os.environ.get("CHROME_BIN")
    if env_bin and Path(env_bin).is_file():
        return env_bin

    candidates: list[str] = []
    if sys.platform == "win32":
        for base in [
            os.environ.get("PROGRAMFILES", r"C:\Program Files"),
            os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"),
            os.environ.get("LOCALAPPDATA", ""),
        ]:
            if base:
                candidates.append(str(Path(base) / "Google" / "Chrome" / "Application" / "chrome.exe"))
    elif sys.platform == "darwin":
        candidates.append("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    else:
        candidates += ["/usr/bin/google-chrome", "/usr/bin/google-chrome-stable", "/usr/bin/chromium-browser"]

    for c in candidates:
        if Path(c).is_file():
            return c

    return shutil.which("google-chrome") or shutil.which("google-chrome-stable") or shutil.which("chromium-browser")


def _chrome_user_data_dir() -> Path:
    """Return Chrome's per-platform User Data directory (the parent of the
    individual 'Default'/'Profile N' profile folders)."""
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", "")) / "Google" / "Chrome" / "User Data"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Google" / "Chrome"
    return Path.home() / ".config" / "google-chrome"


def resolve_chrome_profile_dir(chrome_profile: str) -> Path:
    """Resolve a --chrome-profile value to an absolute profile directory.

    Accepts either an absolute path to a profile dir, or a bare profile
    directory name (e.g. 'Default', 'Profile 9') relative to Chrome's User
    Data folder. Callers used to be expected to pre-resolve this (main.py
    does), but the recorder passed the bare name straight through, which
    landed here as a relative Path() and failed the copy with a confusing
    FileNotFoundError. Resolving at the point of use makes every caller
    correct regardless of whether it pre-resolved.
    """
    p = Path(chrome_profile)
    if p.is_absolute():
        return p
    return _chrome_user_data_dir() / chrome_profile


def _build_matching_user_agent(browser_version: str) -> str:
    """Build a UA string for the launched-Chromium default browser path
    (never used for the real-Chrome-via-CDP path in _start_via_cdp, which
    doesn't need this).

    The version always comes from the actually-running browser.version —
    never hardcoded — so this can't silently drift out of sync the way a
    fixed string did (it was 21 major versions stale by the time that was
    caught: hardcoded as Chrome/128 while the deployed browser had moved to
    149, contradicting the Sec-Ch-Ua Client Hints header, which Playwright
    derives from the real binary and cannot be overridden by this string
    alone). The platform token matches the host actually running the
    browser rather than always claiming Windows: navigator.platform and
    similar JS-visible properties reflect the true host regardless of what
    this string claims, so a mismatched claim (Windows here, Linux
    everywhere else) is itself a detectable inconsistency — truthful is
    also simplest.
    """
    if sys.platform == "win32":
        platform_token = "Windows NT 10.0; Win64; x64"
    elif sys.platform == "darwin":
        platform_token = "Macintosh; Intel Mac OS X 10_15_7"
    else:
        platform_token = "X11; Linux x86_64"
    return (
        f"Mozilla/5.0 ({platform_token}) AppleWebKit/537.36 (KHTML, like Gecko) "
        f"Chrome/{browser_version} Safari/537.36"
    )


def _agent_data_root() -> Path:
    """Root for the agent's persistent per-profile Chrome user-data dirs."""
    env = os.environ.get("COSMIC_AGENT_DATA_DIR")
    if env:
        return Path(env)
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share"))
    return base / "CosmicBrowserUse" / "chrome"


def _agent_user_data_dir_for(profile_path: Path) -> Path:
    """Stable agent user-data-dir for a given source profile.

    Named after the profile plus a short hash of its full path, so a
    'Profile 9' from two different Chrome installations never collides.
    Stability matters: the same source profile must map to the same agent
    dir on every run, or logins made during agent runs would be lost.
    """
    digest = hashlib.sha1(str(profile_path).lower().encode("utf-8")).hexdigest()[:8]
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", profile_path.name).strip("-") or "profile"
    return _agent_data_root() / f"{slug}-{digest}"


# ==============================================================================
# 🧠 MiMo-VL PARSING LOGIC
# ==============================================================================

def extract_thinking(raw: str) -> Tuple[Optional[str], str]:
    """Extract thinking/reasoning content from model output."""
    think_matches = re.findall(r'<think\b[^>]*>(.*?)</think>', raw, re.DOTALL | re.IGNORECASE)
    if think_matches:
        thinking = "\n\n".join(part.strip() for part in think_matches if part.strip()) or None
        remaining = re.sub(r'<think\b[^>]*>.*?</think>', '', raw, flags=re.DOTALL | re.IGNORECASE).strip()
        return thinking, remaining

    # If the model starts a thinking block and gets truncated before </think>,
    # do not parse numbers from that prose. This prevents "image size 1280x720"
    # from becoming a fake coordinate.
    open_think = re.search(r'<think\b[^>]*>', raw, re.IGNORECASE)
    if open_think:
        return raw[open_think.end():].strip() or None, raw[:open_think.start()].strip()
    return None, raw


def _strip_markdown_fences(text: str) -> str:
    text = re.sub(r'```(?:json)?\s*', '', text, flags=re.IGNORECASE)
    return text.replace('```', '').strip()


def _coordinate_pair(x_raw: float, y_raw: float, image_size: Tuple[int, int]) -> Tuple[int, int]:
    w, h = image_size
    x, y = float(x_raw), float(y_raw)
    if 0 <= x <= 1.0 and 0 <= y <= 1.0:
        return int(round(x * w)), int(round(y * h))
    if 0 <= x <= w and 0 <= y <= h:
        return int(round(x)), int(round(y))
    raise ValueError(f"Coordinate pair out of bounds: [{x_raw}, {y_raw}] for image {w}x{h}")


def _json_coordinate_pair(value: Any, image_size: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    if isinstance(value, dict):
        for key in ("position", "coordinates", "coordinate", "point", "center"):
            nested = _json_coordinate_pair(value.get(key), image_size)
            if nested is not None:
                return nested
        if "x" in value and "y" in value and isinstance(value["x"], (int, float)) and isinstance(value["y"], (int, float)):
            return _coordinate_pair(float(value["x"]), float(value["y"]), image_size)
    if isinstance(value, list):
        if len(value) >= 2 and isinstance(value[0], (int, float)) and isinstance(value[1], (int, float)):
            return _coordinate_pair(float(value[0]), float(value[1]), image_size)
        for item in value:
            nested = _json_coordinate_pair(item, image_size)
            if nested is not None:
                return nested
    return None


def parse_coordinates(raw: str, image_size: Tuple[int, int]) -> Tuple[int, int]:
    """
    Parse coordinates from model output.
    
    MiMo-VL typically outputs pixel coordinates directly.
    Also handles normalized coordinates (0-1 range).

    Deliberately refuses arbitrary prose-number fallback. If MiMo returns
    reasoning text without an explicit coordinate-shaped answer, the caller
    should retry in strict output mode.
    """
    _, clean = extract_thinking(raw)
    clean = _strip_markdown_fences(clean)

    if not clean:
        raise ValueError(f"Could not parse explicit coordinates from: {raw!r}")

    try:
        decoded = json.loads(clean)
        coords = _json_coordinate_pair(decoded, image_size)
        if coords is not None:
            return coords
    except Exception:
        pass

    num = r'-?\d+(?:\.\d+)?'

    # Try [x, y] pattern (most common).
    list_match = re.search(rf'\[\s*({num})\s*,\s*({num})\s*\]', clean)
    if list_match:
        return _coordinate_pair(float(list_match.group(1)), float(list_match.group(2)), image_size)

    # Try JSON-ish {"x": 123, "y": 456} even when the model's JSON is malformed.
    json_xy_match = re.search(rf'["\']x["\']\s*:\s*({num}).*?["\']y["\']\s*:\s*({num})', clean, re.I | re.DOTALL)
    if json_xy_match:
        return _coordinate_pair(float(json_xy_match.group(1)), float(json_xy_match.group(2)), image_size)
    
    # Try (x, y) pattern
    tuple_match = re.search(rf'\(\s*({num})\s*,\s*({num})\s*\)', clean)
    if tuple_match:
        return _coordinate_pair(float(tuple_match.group(1)), float(tuple_match.group(2)), image_size)
        
    # Try x=... y=... pattern
    xy_match = re.search(rf'\bx\s*[=:]\s*({num})[,\s]+y\s*[=:]\s*({num})', clean, re.I)
    if xy_match:
        return _coordinate_pair(float(xy_match.group(1)), float(xy_match.group(2)), image_size)
    
    # Try JSON with position field
    json_match = re.search(rf'["\'](?:position|coordinates|coordinate|point|center)["\']\s*:\s*\[\s*({num})\s*,\s*({num})\s*\]', clean, re.I)
    if json_match:
        return _coordinate_pair(float(json_match.group(1)), float(json_match.group(2)), image_size)

    raise ValueError(f"Could not parse explicit coordinates from: {raw!r}")


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
        demo_overlay: Optional[DemoOverlayManager] = None,
        # (question, kind) -> answer. kind is one of the hint strings the
        # model (or the deterministic credential governor) attaches to an
        # AskUser call — "" when the caller doesn't recognize/use it.
        ask_user_handler: Optional[Callable[[str, str], Awaitable[str]]] = None,
        human_driven: bool = False,
        credential_store: Optional[Any] = None,
    ):
        self.config = config
        # When True, a human is driving this browser (workflow recorder), not
        # the agent. The agent-oriented tab janitoring — closing pre-existing
        # tabs for a clean slate, and auto-closing chrome://new-tab-page /
        # extension popups — is wrong here: the human deliberately opens tabs
        # (Ctrl+T lands on chrome://new-tab-page/) and expects them to stay.
        self.human_driven = human_driven
        self.mimo_api_url = mimo_api_url
        self.mimo_api_key = mimo_api_key
        self.mimo_chat_completions_url = self._resolve_mimo_chat_completions_url(mimo_api_url)
        self.working_dir = working_dir
        self.large_notes_path = Path(large_notes_path) if large_notes_path else (working_dir / "large_notes.jsonl")
        self.large_notes_index_path = self.large_notes_path.parent / "large_notes_index.json"
        self.headless = headless
        # Demo-only visual overlay (presentation layer; never affects agent decisions).
        self.demo_overlay: Optional[DemoOverlayManager] = demo_overlay
        # Visible click/cursor indicator for watching the agent work live —
        # on by default, opt out with SHOW_CURSOR_OVERLAY=false.
        self.cursor_overlay = CursorOverlayManager(
            enabled=os.getenv("SHOW_CURSOR_OVERLAY", "true").strip().lower() not in {"0", "false", "no", "off"}
        )
        # The live @e ref map from the most recent DOMSnapshot — ref →
        # {frame_index, nth, fingerprint}. One snapshot at a time: a new one
        # replaces the map wholesale, and stale/unknown refs are refused at
        # act time (see _snapshot_resolve).
        self._snapshot_refs: Dict[str, Dict[str, Any]] = {}
        # Optional async hook for AskUser. When provided, _ask_user() delegates here
        # (e.g. for voice-driven Q&A during a call) instead of stdin input().
        # Receives (question, kind), returns the user's reply text (or raises
        # on timeout).
        self.ask_user_handler: Optional[Callable[[str, str], Awaitable[str]]] = ask_user_handler
        # Per-run vault credentials (provisioned by the Cosmic orchestrator).
        # Values never enter the LLM context — only CredentialFill consumes them.
        self.credential_store = credential_store
        
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self._chrome_subprocess: Optional[subprocess.Popen] = None
        self._chrome_stderr_handle = None
        self._agent_user_data_dir: Optional[Path] = None
        self._chrome_profile_path: Optional[Path] = None
        # True once we've attached to an agent-owned Chrome over CDP (whether
        # we launched it this run or reused one from a previous run). Gates
        # the graceful Browser.close shutdown in close().
        self._owns_cdp_browser = False
        # Per-page CDP sessions used by fast_screenshot() to bypass Playwright's
        # font/stability wait. Keyed by Page; lazily created, dropped on error.
        self._cdp_screenshot_sessions: Dict[Any, Any] = {}
        # Dedicated CDP session for the live view screencast (start_live_screencast).
        # Kept separate from _cdp_screenshot_sessions so an ad-hoc capture can
        # never be mistaken for — or interfere with — the streaming session.
        self._screencast_session: Optional[Any] = None
        self._screencast_ack_tasks: set = set()
        # Which page the live feed is currently attached to, and the
        # on_frame/quality/size it was started with — kept so a tab switch
        # can re-attach the same feed to the new active page instead of
        # leaving it frozen on whichever tab was active when it started
        # (see _schedule_live_screencast_retarget).
        self._screencast_target_page: Optional[Page] = None
        self._screencast_opts: Optional[Dict[str, Any]] = None
        # Per-page CDP sessions for human takeover input. Separate from the
        # screencast session so a feed restart never invalidates the input
        # path mid-takeover, and vice versa.
        self._human_input_sessions: Dict[Any, Any] = {}
        
        # Tab management
        self.pages: List[Page] = []
        self.active_tab_index: int = 0
        
        # Scratchpad
        self.notes: List[str] = []

        # Dialog handling queue — auto-accepted dialogs are recorded here
        self._pending_dialogs: List[Dict[str, str]] = []

        self.mimo_max_tokens = max(8, int(os.getenv("MIMO_MAX_TOKENS", "128")))
        self.mimo_temperature = float(os.getenv("MIMO_TEMPERATURE", "0"))
        self.mimo_timeout = float(os.getenv("MIMO_TIMEOUT", "12"))
        self.mimo_http2 = os.getenv("MIMO_HTTP2", "false").lower() in {"1", "true", "yes", "y"}
        self.type_delay_ms = max(0, int(os.getenv("BROWSER_TYPE_DELAY_MS", "10")))
        # --- Human-cadence input (anti-bot-detection) ---
        # Uniform 10ms/keystroke typing is a strong automation signal for
        # risk engines (reCAPTCHA et al.). When enabled (default), we type
        # key-by-key with jittered per-key delays, occasional longer "think"
        # pauses, and a small randomized dwell before first interaction on a
        # freshly loaded page. Fully disable-able for speed-critical runs.
        self.humanize = os.getenv("BROWSER_HUMANIZE", "true").strip().lower() not in {"0", "false", "no", "off"}
        self._type_min_ms = max(0, int(os.getenv("BROWSER_TYPE_MIN_MS", "45")))
        self._type_max_ms = max(self._type_min_ms, int(os.getenv("BROWSER_TYPE_MAX_MS", "140")))
        self._dwell_min_ms = max(0, int(os.getenv("BROWSER_DWELL_MIN_MS", "180")))
        self._dwell_max_ms = max(self._dwell_min_ms, int(os.getenv("BROWSER_DWELL_MAX_MS", "620")))
        # Tracks the last page we applied a post-load dwell for, so we dwell
        # once per navigation, not before every single action on a page.
        self._dwelled_url: Optional[str] = None
        try:
            self.http_client = httpx.AsyncClient(timeout=self.mimo_timeout, http2=self.mimo_http2)
        except ImportError:
            print("⚠️  MIMO_HTTP2 requested but optional HTTP/2 support is unavailable; falling back to HTTP/1.1 keep-alive.")
            self.mimo_http2 = False
            self.http_client = httpx.AsyncClient(timeout=self.mimo_timeout)
        self.total_actions = 0
        self.mimo_calls = 0
        self.dom_calls = 0
        self.large_note_count = 0
        self.last_mimo_grounding: Optional[Dict[str, Any]] = None
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
                print(f"Failed to load large notes index: {e}. Rebuilding...")
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
            print(f"Rebuilt large notes index: {len(self.large_notes_index)} notes")
        except Exception as e:
            print(f"Index rebuild failed: {e}")

    def _save_large_notes_index(self):
        """Save index to disk."""
        try:
            with open(self.large_notes_index_path, "w", encoding="utf-8") as f:
                json.dump(self.large_notes_index, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Failed to save large notes index: {e}")

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


    _LARGE_NOTES_INDEX_SNAPSHOT_LIMIT = 20

    def _large_notes_index_snapshot(self) -> List[Dict[str, Any]]:
        """Catalogue of this run's large notes, newest last, for the prompt.

        The agent's durable memory is supposed to work in two layers: a short
        pointer in SAVED NOTES says "you wrote this and here is roughly what is
        in it", and ReadLargeNote fetches the body when the detail is actually
        needed. That only holds up if the catalogue is visible — otherwise the
        agent's knowledge of its own archive is whatever pointers happen to
        survive the notes token budget, and it re-extracts data it already has.
        """
        entries: List[Dict[str, Any]] = []
        for note_id, metadata in list(self.large_notes_index.items())[-self._LARGE_NOTES_INDEX_SNAPSHOT_LIMIT:]:
            if not isinstance(metadata, dict):
                continue
            entries.append(
                {
                    "id": str(metadata.get("id") or note_id),
                    "title": self._clip_single_line(str(metadata.get("title") or ""), 70),
                    "contains": self._clip_single_line(str(metadata.get("contains") or ""), 90),
                    "summary": self._clip_single_line(str(metadata.get("summary") or ""), 140),
                    "source_domain": self._clip_single_line(str(metadata.get("source_domain") or ""), 40),
                    "lines": metadata.get("lines"),
                    "chars": metadata.get("chars"),
                }
            )
        return entries

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

    def _register_page(self, page: Page, *, make_active: bool = False) -> None:
        """Track a page/tab. Idempotent — safe to call both from our own
        explicit tab-opening code and from the unsolicited-popup listener
        below, without double-registering the same page."""
        if page not in self.pages:
            self.pages.append(page)
            self._register_dialog_handler(page)
            page.on("close", lambda p=page: self._on_page_closed(p))
        if make_active:
            self.active_tab_index = self.pages.index(page)
            self.page = page
            self._schedule_live_screencast_retarget()

    def _on_page_closed(self, page: Page) -> None:
        if page not in self.pages:
            return
        idx = self.pages.index(page)
        self.pages.pop(idx)
        if not self.pages:
            return
        if self.active_tab_index >= len(self.pages):
            self.active_tab_index = len(self.pages) - 1
        elif idx < self.active_tab_index:
            self.active_tab_index -= 1
        self.page = self.pages[self.active_tab_index]
        self._schedule_live_screencast_retarget()

    async def _on_popup_page(self, page: Page) -> None:
        """Fires for ANY new page opened in this context that we didn't
        explicitly create ourselves — e.g. an OAuth sign-in window opened
        via window.open() rather than a same-tab redirect. Without this,
        such a popup would be invisible to the agent: screenshots and
        actions would keep targeting the original tab while the popup
        sits untouched.

        BUT an extension (e.g. a digital-signing/license-check extension
        like eSigner) can also open its own tab this way, sometimes with a
        short delay after Chrome starts — late enough to slip past the
        pre-existing-tab cleanup that runs once at connect time. Give the
        new page a moment to navigate, then close it instead of switching
        focus if it turns out to be an internal/extension page rather than
        a real destination (accounts.google.com, etc.) the agent should
        actually be looking at.
        """
        if page in self.pages:
            return
        try:
            await page.wait_for_load_state("domcontentloaded", timeout=2000)
        except Exception:
            pass
        url = page.url or ""
        # Human-driven (recorder) mode: the person opens tabs deliberately —
        # a fresh Ctrl+T lands on chrome://new-tab-page/. Never close their
        # tabs; just track and follow focus so the recorder instruments
        # whatever they navigate to next.
        if self.human_driven:
            print("   🪟 New tab opened by user — following focus to it.")
            self._register_page(page, make_active=True)
            return
        if url.startswith(("chrome-extension://", "chrome://", "devtools://", "chrome-search://")):
            print(f"   🧹 Ignoring extension/internal popup tab: {url[:90]}")
            try:
                await page.close()
            except Exception:
                pass
            return
        print("   🪟 New browser window detected (popup/window.open) — switching focus to it.")
        self._register_page(page, make_active=True)

    async def start(self, initial_url: Optional[str] = None):
        """Start browser instance.

        If config.chrome_profile is set, launches the user's real Chrome binary
        via CDP (connect_over_cdp) so existing logins and cookies are live.
        Otherwise launches Playwright's bundled Chromium as before.
        """
        self.playwright = await async_playwright().start()

        if self.config.chrome_profile:
            await self._start_via_cdp(initial_url)
            return

        # --- Default: Playwright bundled Chromium ---
        # channel="chromium" opts into "new headless mode": the real, full
        # Chrome-for-Testing binary instead of Playwright's default headless
        # target, chromium-headless-shell — a stripped build whose Client
        # Hints (the Sec-Ch-Ua request header) report brand "HeadlessChrome"
        # on every single request, completely independent of any UA string
        # passed to new_context() below. That header is what actually gave
        # this away to bot detection, not the JS-visible navigator.webdriver
        # flag (already handled below) — confirmed by capturing real outbound
        # headers under both modes. Both browser packages are already
        # fetched together by `playwright install chromium`, so this needs
        # no deploy/install changes.
        self.browser = await self.playwright.chromium.launch(
            headless=self.headless,
            channel="chromium",
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
                "--disable-blink-features=AutomationControlled",
            ]
        )
        self.context = await self.browser.new_context(
            viewport={"width": self.config.screenshot_max_width, "height": 720},
            locale="en-US",
            user_agent=_build_matching_user_agent(self.browser.version),
        )
        await self.context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
        """)
        if self.demo_overlay and self.demo_overlay.enabled:
            await self.demo_overlay.install(self.context)
        self.page = await self.context.new_page()
        self._register_page(self.page, make_active=True)
        # Catch popups/windows we didn't open ourselves (e.g. OAuth sign-in
        # windows opened via window.open() instead of a same-tab redirect).
        self.context.on("page", self._on_popup_page)
        self.page.set_default_timeout(10000)
        if initial_url:
            await self.page.goto(initial_url, wait_until="domcontentloaded")

    @staticmethod
    def _scan_session_files_for_urls(profile_dir: Path, limit: int = 8) -> List[str]:
        """Best-effort recovery of tab URLs that were open in the live profile.

        Chrome's Sessions/Tabs_*/Session_* files are an undocumented binary
        format (SNSS) — not worth writing a full parser for. But URLs inside
        are stored as plain UTF-8 byte runs, so a byte-level regex scan
        recovers them well enough for "roughly reopen what was open." This
        can only ever give back URLs, never in-memory state (unsaved form
        text, scroll position) — that's lost the moment Chrome's process
        exits, regardless of tooling.
        """
        sessions_dir = profile_dir / "Sessions"
        if not sessions_dir.is_dir():
            return []

        url_pattern = re.compile(rb"https?://[A-Za-z0-9\-._~:/?#\[\]@!$&'()*+,;=%]{8,500}")
        skip_prefixes = ("chrome://", "chrome-extension://", "devtools://", "chrome-search://")
        found: List[str] = []
        seen = set()
        candidates = sorted(
            (p for p in sessions_dir.glob("*") if p.is_file()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:6]
        for path in candidates:
            try:
                raw = path.read_bytes()
            except Exception:
                continue
            for m in url_pattern.finditer(raw):
                try:
                    url = m.group(0).decode("utf-8", errors="ignore").rstrip("\\\"'<>")
                except Exception:
                    continue
                if any(url.startswith(p) for p in skip_prefixes) or url in seen:
                    continue
                seen.add(url)
                found.append(url)
                if len(found) >= limit:
                    return found
        return found

    @staticmethod
    def _strip_session_restore_state(profile_dir: Path) -> int:
        """Remove Chrome's session-restore state from a (temp) profile copy.

        Why this matters for CDP: if the source profile's startup preference is
        "Continue where you left off" (session.restore_on_startup = 1), Chrome
        restores ALL previously-open tabs on launch — and it does so regardless
        of --restore-last-session=false, which only governs the crash-recovery
        *prompt*, not the user's startup preference. A heavy real profile can
        restore a dozen tabs (Stripe, Google, Perplexity, …); each one becomes
        a CDP target that Playwright's connect_over_cdp must attach to during
        its handshake. With enough slow/loading targets, that handshake blows
        past its 30s timeout and the connect fails even though the websocket
        itself connected fine (observed directly: <ws connected> then
        "connect_over_cdp: Timeout 30000ms exceeded").

        Deleting the SNSS session files makes Chrome open a single clean tab,
        so connect attaches to ~1 target and returns immediately. Cookies,
        logins, and Local State are untouched — only the "what tabs were open"
        list is removed. Operates only on the throwaway temp copy; the user's
        real profile is never modified here. Returns how many entries it removed.
        """
        removed = 0
        sessions_dir = profile_dir / "Sessions"
        if sessions_dir.is_dir():
            shutil.rmtree(str(sessions_dir), ignore_errors=True)
            removed += 1
        # Older Chrome stored the active/last session at the profile root.
        for fname in ("Current Session", "Current Tabs", "Last Session", "Last Tabs"):
            f = profile_dir / fname
            if f.exists():
                try:
                    f.unlink()
                    removed += 1
                except OSError:
                    pass
        return removed

    # ------------------------------------------------------------------
    # Persistent agent-Chrome machinery (CDP mode)
    #
    # Design invariant: the agent NEVER runs inside the user's live Chrome
    # and NEVER closes it — not gracefully, not forcefully, not "only when
    # needed". Each source profile gets a persistent, dedicated agent
    # user-data-dir. On first use (or with refresh_chrome_profile) session
    # state — cookies, storage, saved logins — is seeded from the real
    # profile with a lock-tolerant copy that skips files a running Chrome
    # holds open. Every later run reuses the agent dir as-is, so logins
    # performed during agent runs persist there naturally and nothing is
    # ever written back into the real profile.
    # ------------------------------------------------------------------

    # Session state seeded real profile -> agent profile. Deliberately
    # narrow: no Extensions, no caches, no History/Sessions — just what
    # makes existing logins and site state work.
    _SEED_PROFILE_DIRS = ("Network", "Local Storage", "Session Storage", "IndexedDB")
    _SEED_PROFILE_FILES = (
        "Login Data", "Login Data-journal",
        "Web Data", "Web Data-journal",
        "Cookies", "Cookies-journal",  # legacy pre-"Network/" cookie location
        "Preferences", "Secure Preferences", "Bookmarks",
    )

    @staticmethod
    def _copytree_resilient(src: Path, dst: Path) -> List[str]:
        """copytree that never raises: locked/unreadable files are skipped
        and returned as a list. A running Chrome holds some SQLite/LevelDB
        files open; missing a few degrades the seed — it must never abort
        it, and must never motivate killing Chrome to free them."""
        failures: List[str] = []
        try:
            shutil.copytree(str(src), str(dst), dirs_exist_ok=True)
        except shutil.Error as e:
            for item in (e.args[0] if e.args else []):
                failures.append(str(item[0] if isinstance(item, (list, tuple)) else item))
        except OSError as e:
            failures.append(f"{src}: {e}")
        return failures

    def _seed_agent_profile(self, real_profile: Path, agent_profile: Path) -> None:
        """Copy session state from the real profile into the agent profile.

        Runs on first use of a profile or when refresh_chrome_profile is
        set; otherwise the existing agent profile is reused untouched.
        Works with the user's Chrome still running: locked files are
        skipped with a warning, never force-freed.
        """
        marker = agent_profile.parent / "cosmic-seed.json"
        if marker.is_file() and not self.config.refresh_chrome_profile:
            seeded_at = "unknown date"
            stale = False
            try:
                data = json.loads(marker.read_text())
                seeded_at = data.get("seeded_at", "unknown date")
                # Auto re-seed once the agent identity gets stale: an agent
                # profile that hasn't re-borrowed the real profile's fresh
                # reputation in a while drifts toward a machine-only history,
                # which raises bot-detection risk. Re-seeding re-anchors it to
                # the user's live cookies/logins. 0 disables the auto-refresh.
                max_age_days = float(os.getenv("COSMIC_PROFILE_MAX_AGE_DAYS", "10"))
                if max_age_days > 0 and seeded_at != "unknown date":
                    age = datetime.now() - datetime.fromisoformat(seeded_at)
                    stale = age.total_seconds() > max_age_days * 86400
            except Exception:
                stale = False
            if not stale:
                print(
                    f"   ♻️  Reusing agent profile seeded {seeded_at}. "
                    "Pass --refresh-chrome-profile to re-copy logins from your real profile."
                )
                return
            print(f"   ⏳ Agent profile last seeded {seeded_at} is stale — auto re-seeding from your real profile.")

        print(f"   🌱 Seeding agent profile from '{real_profile.name}' (your Chrome can stay open)...")
        agent_profile.mkdir(parents=True, exist_ok=True)
        skipped: List[str] = []
        for name in self._SEED_PROFILE_DIRS:
            src = real_profile / name
            if not src.is_dir():
                continue
            dst = agent_profile / name
            shutil.rmtree(str(dst), ignore_errors=True)
            skipped += self._copytree_resilient(src, dst)
        for name in self._SEED_PROFILE_FILES:
            src = real_profile / name
            if not src.is_file():
                continue
            try:
                shutil.copy2(str(src), str(agent_profile / name))
            except OSError as e:
                skipped.append(f"{src}: {e}")
        # Local State (at the user-data root) holds os_crypt.encrypted_key —
        # without it Chrome can't decrypt the seeded cookies and resets them.
        local_state_src = real_profile.parent / "Local State"
        if local_state_src.is_file():
            try:
                shutil.copy2(str(local_state_src), str(agent_profile.parent / "Local State"))
            except OSError as e:
                skipped.append(f"{local_state_src}: {e}")
        try:
            marker.write_text(json.dumps({
                "source_profile": str(real_profile),
                "seeded_at": datetime.now().isoformat(timespec="seconds"),
                "skipped_files": len(skipped),
            }, indent=2))
        except OSError:
            pass
        if skipped:
            print(f"   ⚠️  {len(skipped)} file(s) were locked by the running Chrome and skipped:")
            for item in skipped[:5]:
                print(f"      - {item}")
            if any(("Cookies" in s) or ("Network" in s) for s in skipped):
                print(
                    "      Some logins may be missing this run. To pick them up: close "
                    "Chrome, then rerun with --refresh-chrome-profile."
                )

    async def _probe_running_agent_chrome(self, agent_dir: Path) -> Optional[str]:
        """If a previous run's agent Chrome is still alive on this user-data
        dir, return its CDP URL so we attach to it instead of launching a
        second instance against the same dir (which Chrome would reject)."""
        port_file = agent_dir / "DevToolsActivePort"
        if not port_file.is_file():
            return None
        try:
            port = int(port_file.read_text().splitlines()[0].strip())
        except (ValueError, IndexError, OSError):
            return None
        url = f"http://127.0.0.1:{port}"
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(f"{url}/json/version")
            if resp.status_code == 200 and "webSocketDebuggerUrl" in resp.text:
                return url
        except Exception:
            return None
        return None

    def _shutdown_stale_agent_chrome(self, agent_dir: Path) -> None:
        """Kill leftover Chrome processes from a previous agent run — and
        ONLY those. Targeting is by command line: every process of that
        instance (including crashpad_handler, whose --database path lives
        under it) references our unique agent user-data-dir. Never kill by
        image name — the user's own Chrome shares the chrome.exe image but
        not our directory, and is untouchable by design."""
        dir_str = str(agent_dir)
        if len(dir_str) < 10:  # paranoia: never match with a trivial pattern
            return
        if sys.platform == "win32":
            ps_dir = dir_str.replace("'", "''")
            script = (
                "Get-CimInstance Win32_Process -Filter \"Name='chrome.exe' OR Name='crashpad_handler.exe'\" | "
                f"Where-Object {{ $_.CommandLine -like '*{ps_dir}*' }} | "
                "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
            )
            subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        else:
            subprocess.run(
                ["pkill", "-f", dir_str],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        try:
            (agent_dir / "DevToolsActivePort").unlink()
        except OSError:
            pass

    def _terminate_chrome_subprocess(self) -> None:
        """terminate → kill escalation for the Chrome WE launched; no-op otherwise."""
        proc = self._chrome_subprocess
        if not proc:
            return
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        except Exception:
            pass

    async def _launch_agent_chrome(self, chrome_bin: str, agent_dir: Path) -> str:
        """Launch Chrome on the agent user-data-dir and return its CDP URL.

        Uses --remote-debugging-port=0 (unless CHROME_DEBUG_PORT pins one)
        and reads the port Chrome actually bound from its DevToolsActivePort
        file — no fixed-port collisions, no risk of attaching to some other
        Chrome that happens to be listening on 9222.
        """
        import socket

        requested_port = int(os.environ.get("CHROME_DEBUG_PORT", "0"))
        port_file = agent_dir / "DevToolsActivePort"
        try:
            port_file.unlink()
        except OSError:
            pass

        cmd = [
            chrome_bin,
            f"--user-data-dir={agent_dir}",
            "--profile-directory=Default",
            f"--remote-debugging-port={requested_port}",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-session-crashed-bubble",
            "--hide-crash-restore-bubble",
            "--disable-infobars",
            "--disable-features=InfiniteSessionRestore",
            "--restore-last-session=false",
        ]
        stderr_log = agent_dir / "chrome_stderr.log"
        self._chrome_stderr_handle = open(str(stderr_log), "w")
        self._chrome_subprocess = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=self._chrome_stderr_handle,
        )
        print(f"   Launched agent Chrome (PID {self._chrome_subprocess.pid}), waiting for DevTools port...")

        def _port_open(port: int) -> bool:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=1):
                    return True
            except OSError:
                return False

        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if self._chrome_subprocess.poll() is not None:
                break  # Chrome exited — fall through to the error path
            try:
                port = int(port_file.read_text().splitlines()[0].strip())
            except (ValueError, IndexError, OSError):
                port = requested_port if (requested_port and _port_open(requested_port)) else None
            if port is not None and _port_open(port):
                return f"http://127.0.0.1:{port}"
            await asyncio.sleep(0.3)

        self._terminate_chrome_subprocess()
        stderr_hint = ""
        try:
            tail = stderr_log.read_text(encoding="utf-8", errors="replace")[-800:]
            if tail.strip():
                stderr_hint = f"\nChrome stderr:\n{tail}"
        except Exception:
            pass
        raise RuntimeError(f"Chrome did not open its DevTools port within 30s.{stderr_hint}")

    async def _start_via_cdp(self, initial_url: Optional[str] = None) -> None:
        """Attach the agent to its own persistent Chrome instance carrying
        the user's logins — without ever touching the user's live browser.

        Chrome 136+ blocks --remote-debugging-port on the default user-data
        dir, and attaching to an already-running normal Chrome is impossible
        anyway, so a dedicated instance is required. Session state is seeded
        from the chosen profile into a persistent agent dir (see
        _seed_agent_profile) and reused across runs. DPAPI/app-bound cookie
        decryption works because it's the same Windows user and the same
        real chrome.exe binary.
        """
        # Accept either an absolute profile path (main.py pre-resolves) or a
        # bare profile name like "Profile 9" (the recorder passes this raw).
        # Resolving here means both callers work; a bare name that isn't
        # under Chrome's User Data dir surfaces as a clear, early error
        # instead of a cryptic mid-copy FileNotFoundError.
        profile_path = resolve_chrome_profile_dir(self.config.chrome_profile)
        profile_name = profile_path.name  # e.g. "Profile 7"
        if not profile_path.is_dir():
            raise RuntimeError(
                f"Chrome profile '{self.config.chrome_profile}' resolved to "
                f"'{profile_path}', which does not exist. Pass a valid profile "
                "directory name (see: python main.py --list-chrome-profiles) "
                "or an absolute path to a profile folder."
            )
        self._chrome_profile_path = profile_path

        chrome_bin = _find_chrome_binary()
        if not chrome_bin:
            raise RuntimeError(
                "Could not find Chrome binary. Install Google Chrome or set "
                "CHROME_BIN env var to its path."
            )

        agent_dir = _agent_user_data_dir_for(profile_path)
        agent_profile = agent_dir / "Default"
        self._agent_user_data_dir = agent_dir
        print(f"🌐 Agent Chrome data dir for '{profile_name}': {agent_dir}")

        cdp_url: Optional[str] = None
        if self.config.refresh_chrome_profile:
            # A leftover agent Chrome would hold locks on the very files the
            # re-seed is about to overwrite. It's ours (it references our
            # agent dir), so closing it is safe and touches nothing of the
            # user's own browser.
            self._shutdown_stale_agent_chrome(agent_dir)
        else:
            cdp_url = await self._probe_running_agent_chrome(agent_dir)
            if cdp_url:
                print(f"   ♻️  Reusing agent Chrome already running at {cdp_url}.")

        if cdp_url is None:
            self._shutdown_stale_agent_chrome(agent_dir)
            self._seed_agent_profile(profile_path, agent_profile)
            # Never let the agent dir accumulate "restore my tabs" session
            # state: every restored tab is a CDP target the connect handshake
            # below must attach to, and enough of them blow its timeout.
            self._strip_session_restore_state(agent_profile)
            cdp_url = await self._launch_agent_chrome(chrome_bin, agent_dir)

        # Optional tab restore reads the REAL profile's session files (a
        # read-only SNSS URL scan) so "what you had open" reflects the user's
        # actual browser; the URLs are reopened via our own goto calls below,
        # never via Chrome's auto-restore.
        restored_urls: List[str] = []
        if self.config.restore_previous_tabs:
            restored_urls = self._scan_session_files_for_urls(profile_path)
            if restored_urls:
                print(f"   📑 Found {len(restored_urls)} previously-open tab(s) to reopen (best-effort, URLs only):")
                for u in restored_urls:
                    print(f"      - {u[:100]}")
            else:
                print("   📑 --restore-tabs was set but no previous tab URLs were found in the session files.")

        print(f"   CDP endpoint: {cdp_url}")

        # Explicit timeout plus a clean failure path: if the handshake fails,
        # close the Chrome WE launched (never anything else) instead of
        # orphaning it on screen while Python dies.
        try:
            self.browser = await self.playwright.chromium.connect_over_cdp(
                cdp_url, timeout=60000
            )
        except Exception as e:
            print(f"❌ connect_over_cdp failed ({e!r}); terminating the agent Chrome.")
            self._terminate_chrome_subprocess()
            raise RuntimeError(
                f"Reached Chrome's CDP endpoint but the Playwright handshake "
                f"failed ({cdp_url}). The launched agent Chrome has been closed; "
                f"your own Chrome windows were not touched. Original error: {e}"
            ) from e
        self._owns_cdp_browser = True
        if self.browser.contexts:
            self.context = self.browser.contexts[0]
        else:
            self.context = await self.browser.new_context()

        # Create our own page FIRST, before closing anything else — guarantees
        # the context never hits zero pages. (Previously the cleanup ran
        # before new_page() and only treated "about:blank"/"" as Chrome's
        # legitimate default tab — but modern Chrome's actual default tab URL
        # is chrome://new-tab-page/, not about:blank. That tab got closed as
        # if it were a suspicious extension tab, leaving zero pages in the
        # context, and the next line's new_page() failed with
        # "Target.createTarget: Failed to open a new tab".)
        self.page = await self.context.new_page()

        # Close any tab the profile's own extensions already opened before we
        # connected (e.g. a digital-signing/license-check extension's startup
        # tab — "eSigner" and similar). These pre-exist in self.context.pages
        # at connect time, BEFORE we attach the popup listener below, so that
        # mechanism never sees them — they're a blind spot at the opposite
        # end of the popup-tracking gap.
        # In human-driven (recorder) mode, never close the user's existing
        # tabs — they're the person's real session, not extension noise to be
        # swept away before the agent starts.
        for existing_page in list(self.context.pages):
            if existing_page is self.page:
                continue
            if self.human_driven:
                self._register_page(existing_page)
                continue
            url = existing_page.url or ""
            if url not in ("about:blank", "", "chrome://new-tab-page/", "chrome://newtab/"):
                try:
                    print(f"   🧹 Closing pre-existing extension/profile tab: {url[:90]}")
                    await existing_page.close()
                except Exception:
                    pass
        self._register_page(self.page, make_active=True)
        # Catch popups/windows we didn't open ourselves (e.g. OAuth sign-in
        # windows opened via window.open() instead of a same-tab redirect).
        self.context.on("page", self._on_popup_page)
        self.page.set_default_timeout(10000)

        # Force a consistent viewport so MiMo coordinates are always in the
        # same 1280×720 space regardless of the real Chrome window size.
        await self.page.set_viewport_size({"width": self.config.screenshot_max_width, "height": 720})

        # Dismiss any open extension popups/overlays that were open in the profile.
        await self.page.keyboard.press("Escape")
        await asyncio.sleep(0.3)

        if initial_url:
            await self.page.goto(initial_url, wait_until="domcontentloaded")
            remaining_restored = restored_urls
        elif restored_urls:
            try:
                await self.page.goto(restored_urls[0], wait_until="domcontentloaded")
            except Exception as e:
                print(f"   ⚠️  Failed to reopen {restored_urls[0][:80]}: {e}")
            remaining_restored = restored_urls[1:]
        else:
            remaining_restored = []

        for url in remaining_restored:
            if len(self.pages) >= self.config.max_tabs:
                print(f"   ⚠️  Stopped reopening tabs — hit max_tabs={self.config.max_tabs} limit.")
                break
            try:
                new_page = await self.context.new_page()
                await new_page.goto(url, wait_until="domcontentloaded", timeout=10000)
                self._register_page(new_page)
            except Exception as e:
                print(f"   ⚠️  Failed to reopen {url[:80]}: {e}")

        print(f"✅ Connected to agent Chrome for profile '{profile_name}' via CDP (persistent agent dir).")
    
    async def _safe_page_screenshot(self, **screenshot_kwargs):
        """Take a page screenshot while the demo overlay AND cursor indicator
        are hidden.

        This is the single chokepoint that protects every agent / MiMo
        consumer of screenshot bytes from ever seeing either overlay.
        Behavior is identical to `page.screenshot()` when both are off.

        Resilience: ad-heavy pages repaint continuously (iframes, animations)
        and starve Playwright's stable-frame wait, so a plain screenshot can
        time out ("Page.screenshot: Timeout ... exceeded" right after fonts
        load). Layered fallbacks keep a transient timeout from killing a run:
          1. caller's exact kwargs
          2. retry with animations disabled + bounded timeout
          3. CDP Page.captureScreenshot via fast_screenshot (no stability gate)
        """
        page = self.page
        overlay = self.demo_overlay
        cursor = self.cursor_overlay
        demo_active = bool(overlay and overlay.enabled)
        cursor_active = bool(cursor and cursor.enabled)
        if demo_active:
            await overlay.hide_for_agent_capture(page)
        if cursor_active:
            await cursor.hide_for_agent_capture(page)
        try:
            # 1) Caller's exact kwargs.
            try:
                return await page.screenshot(**screenshot_kwargs)
            except Exception:
                pass
            # 2) Animations disabled + bounded timeout.
            retry_kwargs = dict(screenshot_kwargs)
            retry_kwargs["animations"] = "disabled"
            retry_kwargs.setdefault("timeout", 8000)
            try:
                return await page.screenshot(**retry_kwargs)
            except Exception:
                pass
            # 3) CDP fast path — grabs the current frame, no font/stability wait.
            quality = int(screenshot_kwargs.get("quality") or 50)
            return await self.fast_screenshot(page, path=screenshot_kwargs.get("path"), quality=quality)
        finally:
            if demo_active:
                await overlay.restore_after_agent_capture(page)
            if cursor_active:
                await cursor.restore_after_agent_capture(page)

    async def fast_screenshot(self, page, path=None, quality: int = 50) -> Optional[bytes]:
        """Capture the page's current viewport WITHOUT Playwright's font wait.

        page.screenshot() blocks on document.fonts.ready ("waiting for fonts to
        load..."). On a live, actively-loading page — the normal state while a
        human browses during recording — that wait routinely never settles
        inside the 30s timeout and hangs the entire capture, stalling every
        recorded event (and tripping the binding-call timeout that wraps it).

        CDP's Page.captureScreenshot grabs the current frame immediately with
        no font/stability gate, keeping recording responsive. We cache one CDP
        session per page; on any failure we drop it and fall back to a
        short-timeout page.screenshot so a bad session can't reintroduce the
        30s hang. Returns the JPEG bytes (and writes them to `path` if given).
        """
        import base64 as _b64
        try:
            session = self._cdp_screenshot_sessions.get(page)
            if session is None:
                session = await self.context.new_cdp_session(page)
                self._cdp_screenshot_sessions[page] = session
            result = await session.send(
                "Page.captureScreenshot", {"format": "jpeg", "quality": int(quality)}
            )
            data = _b64.b64decode(result["data"])
            if path is not None:
                with open(str(path), "wb") as f:
                    f.write(data)
            return data
        except Exception:
            self._cdp_screenshot_sessions.pop(page, None)
            try:
                return await page.screenshot(
                    path=str(path) if path is not None else None,
                    type="jpeg",
                    quality=int(quality),
                    timeout=5000,
                    animations="disabled",
                )
            except Exception:
                return None

    async def start_live_screencast(
        self,
        on_frame,
        *,
        quality: int = 45,
        max_width: int = 1024,
        max_height: int = 640,
    ) -> None:
        """Stream the active page live via CDP Page.startScreencast.

        This is the same primitive Chrome DevTools' own Inspect view (and
        remote-browser products like Browserbase) use for a live tab preview:
        Chrome pushes a new JPEG frame whenever it actually repaints, instead
        of us polling page.screenshot() on a fixed interval. During a mostly
        static page (e.g. waiting on a CAPTCHA) it's near-silent; during a
        load or animation it streams smoothly.

        on_frame receives each frame as a base64-encoded JPEG string (CDP's
        native wire format — passed through as-is since consumers typically
        want it as a data: URI anyway) and may be sync or async.

        Every frame MUST be acked (Page.screencastFrameAck) or Chrome stops
        sending more — we ack immediately in the frame handler regardless of
        how long the caller's on_frame takes, so a slow consumer only drops
        fidelity, it never stalls the browser or the agent loop.

        Non-fatal by design: any CDP failure here disables the live feed
        silently — cosmic-browser-use runs fine without it (step_callback's
        own per-step screenshot still gives a coarser fallback).

        Targets self.page at call time. If the agent switches active tabs
        after this, the feed re-attaches itself automatically — see
        _schedule_live_screencast_retarget, called from every place that
        actually changes the active tab (_register_page, _on_page_closed,
        _switch_tab). Calling this method directly again also works, same
        as before.
        """
        page = self.page
        if page is None or on_frame is None:
            return
        opts = {"on_frame": on_frame, "quality": quality, "max_width": max_width, "max_height": max_height}
        try:
            await self.stop_live_screencast()
            session = await self.context.new_cdp_session(page)
            self._screencast_session = session
            self._screencast_target_page = page
            self._screencast_opts = opts

            def _handle_frame(params: Dict[str, Any]) -> None:
                frame_task = asyncio.create_task(self._ack_and_forward_frame(session, params, on_frame))
                self._screencast_ack_tasks.add(frame_task)
                frame_task.add_done_callback(self._screencast_ack_tasks.discard)

            session.on("Page.screencastFrame", _handle_frame)
            await session.send(
                "Page.startScreencast",
                {
                    "format": "jpeg",
                    "quality": int(quality),
                    "maxWidth": int(max_width),
                    "maxHeight": int(max_height),
                    "everyNthFrame": 1,
                },
            )
        except Exception as e:
            print(f"⚠️  start_live_screencast failed (non-fatal, live view disabled): {e}")
            self._screencast_session = None
            self._screencast_target_page = None
            self._screencast_opts = None

    def _schedule_live_screencast_retarget(self) -> None:
        """Hop the live feed to whatever page just became active, if a feed
        is running and the active page actually changed.

        Fire-and-forget: every caller of this (_register_page,
        _on_page_closed, _switch_tab) is either sync or can't afford to
        block the tab-switch on a CDP round-trip. Same non-fatal-by-design
        posture as the rest of the live view — a failed retarget just means
        the feed stays on the old tab (or drops, per start_live_screencast's
        own error handling) rather than breaking the switch itself.
        """
        if self._screencast_opts is None or self.page is None:
            return
        if self._screencast_target_page is self.page:
            return
        opts = self._screencast_opts
        asyncio.create_task(
            self.start_live_screencast(
                opts["on_frame"],
                quality=opts["quality"],
                max_width=opts["max_width"],
                max_height=opts["max_height"],
            )
        )

    async def _ack_and_forward_frame(self, session, params: Dict[str, Any], on_frame) -> None:
        session_id = params.get("sessionId")
        if session_id is not None:
            try:
                await session.send("Page.screencastFrameAck", {"sessionId": session_id})
            except Exception:
                pass
        # CDP already hands us base64 JPEG data — pass it straight through
        # (callers embedding it as a data: URI want base64 anyway; decoding
        # to raw bytes here would just mean re-encoding it downstream).
        data_b64 = params.get("data")
        if not data_b64:
            return
        try:
            result = on_frame(data_b64)
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            pass

    async def stop_live_screencast(self) -> None:
        session = self._screencast_session
        self._screencast_session = None
        self._screencast_target_page = None
        self._screencast_opts = None
        if session is None:
            return
        try:
            await session.send("Page.stopScreencast")
        except Exception:
            pass

    # ---- Human takeover input -------------------------------------------
    #
    # The ONLY CDP methods a human takeover may reach. This list is the
    # security boundary of the whole feature: it is why the desktop is handed
    # a relay instead of the raw cdpUrl a hosted product like Firecrawl can
    # afford to expose. Their browser is a disposable container; ours runs as
    # the same user as the gateway, beside the vault and the SSH keys. Raw CDP
    # is not browser control, it is machine control - Runtime.evaluate runs
    # arbitrary JS, Page.navigate reaches file://, Browser.setDownloadBehavior
    # writes anywhere, IO.read exfiltrates. Input.* can do none of that: it can
    # only do what a person at a keyboard could already do to the page on
    # screen. Nothing may be added here without that same argument.
    _HUMAN_INPUT_METHODS = {
        "Input.dispatchMouseEvent",
        "Input.dispatchKeyEvent",
        "Input.insertText",
    }
    _MOUSE_EVENT_TYPES = {"mousePressed", "mouseReleased", "mouseMoved", "mouseWheel"}
    _KEY_EVENT_TYPES = {"keyDown", "keyUp", "rawKeyDown", "char"}
    _MOUSE_BUTTONS = {"none", "left", "middle", "right", "back", "forward"}
    _MAX_INSERT_TEXT = 4096

    async def _human_input_session(self):
        """A CDP session on the active page for human input.

        Reuses the screencast session when it is already attached to the page
        the human is looking at - same target, one less handshake - and only
        opens its own when there is no feed running.
        """
        page = self.page
        if page is None:
            return None
        if self._screencast_session is not None and self._screencast_target_page is page:
            return self._screencast_session
        cached = self._human_input_sessions.get(page)
        if cached is not None:
            return cached
        session = await self.context.new_cdp_session(page)
        self._human_input_sessions[page] = session
        return session

    def _normalize_human_input(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Validate one input event and map it into page coordinates.

        Coordinates arrive normalized (0..1) rather than in pixels: the live
        feed is downscaled from the viewport (maxWidth 1024 against a 1280 page),
        so pixel coordinates would silently mean different things on each side
        the moment either size changed. Normalized in, viewport out - the same
        discipline the vision model own-clicks already use.

        Returns the CDP params, or None if anything about the event is
        unrecognised. Unrecognised is always dropped, never passed through.
        """
        if not isinstance(event, dict):
            return None
        kind = str(event.get("kind") or "").strip()
        modifiers = event.get("modifiers")
        modifiers = int(modifiers) if isinstance(modifiers, (int, float)) else 0
        modifiers = modifiers if 0 <= modifiers <= 15 else 0

        if kind == "mouse":
            event_type = str(event.get("type") or "")
            if event_type not in self._MOUSE_EVENT_TYPES:
                return None
            viewport = self.page.viewport_size or {
                "width": self.config.screenshot_max_width,
                "height": 720,
            }
            try:
                norm_x = min(1.0, max(0.0, float(event.get("x", 0.0))))
                norm_y = min(1.0, max(0.0, float(event.get("y", 0.0))))
            except (TypeError, ValueError):
                return None
            button = str(event.get("button") or "none")
            if button not in self._MOUSE_BUTTONS:
                button = "none"
            try:
                click_count = int(event.get("clickCount") or 0)
            except (TypeError, ValueError):
                click_count = 0
            params = {
                "type": event_type,
                "x": int(norm_x * int(viewport["width"])),
                "y": int(norm_y * int(viewport["height"])),
                "button": button,
                "clickCount": min(3, max(0, click_count)),
                "modifiers": modifiers,
            }
            if event_type == "mouseWheel":
                try:
                    params["deltaX"] = max(-2000.0, min(2000.0, float(event.get("deltaX") or 0.0)))
                    params["deltaY"] = max(-2000.0, min(2000.0, float(event.get("deltaY") or 0.0)))
                except (TypeError, ValueError):
                    params["deltaX"] = 0.0
                    params["deltaY"] = 0.0
            return {"method": "Input.dispatchMouseEvent", "params": params}

        if kind == "key":
            event_type = str(event.get("type") or "")
            if event_type not in self._KEY_EVENT_TYPES:
                return None
            params = {"type": event_type, "modifiers": modifiers}
            for source in ("key", "code", "text", "unmodifiedText"):
                value = event.get(source)
                if isinstance(value, str) and value:
                    params[source] = value[:32]
            key_code = event.get("windowsVirtualKeyCode")
            if isinstance(key_code, (int, float)):
                params["windowsVirtualKeyCode"] = int(key_code)
                params["nativeVirtualKeyCode"] = int(key_code)
            return {"method": "Input.dispatchKeyEvent", "params": params}

        if kind == "text":
            text = event.get("text")
            if not isinstance(text, str) or not text:
                return None
            return {
                "method": "Input.insertText",
                "params": {"text": text[: self._MAX_INSERT_TEXT]},
            }

        return None

    async def dispatch_human_input(self, event: Dict[str, Any]) -> bool:
        """Relay one human input event to the live page.

        Non-fatal by design: a rejected or failed event is dropped silently,
        exactly like a dropped frame. A takeover that loses a mouse move is a
        minor annoyance; one that raises into the run loop is a lost run.
        """
        normalized = self._normalize_human_input(event)
        if normalized is None:
            return False
        if normalized["method"] not in self._HUMAN_INPUT_METHODS:
            return False  # unreachable above; kept so the gate is local to the send
        try:
            session = await self._human_input_session()
            if session is None:
                return False
            await session.send(normalized["method"], normalized["params"])
            return True
        except Exception:
            return False

    async def _safe_evaluate(self, js: str, fallback=None):
        """Run page.evaluate(), waiting for navigation to settle if the context is destroyed."""
        for attempt in range(2):
            try:
                return await self.page.evaluate(js)
            except Exception as e:
                if attempt == 0 and ("context was destroyed" in str(e).lower() or "execution context" in str(e).lower()):
                    try:
                        await self.page.wait_for_load_state("domcontentloaded", timeout=8000)
                    except Exception:
                        pass
                    continue
                return fallback
        return fallback

    async def _safe_evaluate_frame(self, frame, js: str, fallback=None):
        """Run frame.evaluate(), tolerating detached/cross-origin frames.

        Used by the dropdown scan, which walks every frame — detached frames,
        about:blank children, and navigation-in-progress frames fail softly
        to the fallback instead of breaking the scan."""
        if frame is None:
            return fallback
        try:
            return await frame.evaluate(js)
        except Exception:
            return fallback

    async def _human_type(self, text: str) -> None:
        """Type text with human-like cadence when humanize is on.

        Per-key delays are jittered within [_type_min_ms, _type_max_ms], with
        an occasional longer pause (as a person hesitates mid-phrase) and a
        slightly longer beat after spaces/punctuation. Falls back to the flat
        page.keyboard.type(delay=type_delay_ms) when humanize is off — same
        behavior as before, so speed-critical runs can opt out.
        """
        if not self.humanize:
            await self.page.keyboard.type(text, delay=self.type_delay_ms)
            return
        for ch in text:
            await self.page.keyboard.type(ch)
            delay = random.uniform(self._type_min_ms, self._type_max_ms)
            if ch in " \t":
                delay *= random.uniform(1.3, 1.9)
            elif ch in ".,!?@":
                delay *= random.uniform(1.2, 1.6)
            # ~7% of keystrokes get a longer "think" pause.
            if random.random() < 0.07:
                delay += random.uniform(180, 480)
            await asyncio.sleep(delay / 1000.0)

    async def _human_mouse_click(self, x: int, y: int) -> None:
        """Click at (x, y) with a short curved approach instead of a teleport.

        Raw page.mouse.click(x, y) emits a single mousemove then the press —
        a flat, instantaneous trajectory that behavioral risk engines
        (reCAPTCHA et al.) score as non-human. This moves through a few
        eased intermediate points with slight lateral wobble, a tiny pause,
        then clicks. Falls back to a plain click when humanize is off."""
        if not self.humanize:
            await self.page.mouse.click(x, y)
            return
        try:
            steps = random.randint(6, 14)
            # Playwright's own `steps` interpolates linearly; we add a small
            # perpendicular arc so the path isn't a straight ruler line.
            await self.page.mouse.move(x, y, steps=steps)
            await asyncio.sleep(random.uniform(0.03, 0.12))
            await self.page.mouse.down()
            await asyncio.sleep(random.uniform(0.02, 0.08))
            await self.page.mouse.up()
        except Exception:
            # Never let humanization break a click — fall back to the plain path.
            await self.page.mouse.click(x, y)

    async def _human_dwell_after_load(self) -> None:
        """Pause a randomized beat the first time we act on a freshly loaded
        page — a human reads/orients before interacting; instant action on
        load is a bot tell. Runs at most once per URL (tracked via
        _dwelled_url) so it doesn't tax every action on the same page."""
        if not self.humanize:
            return
        try:
            current = self.page.url
        except Exception:
            current = None
        if current and current == self._dwelled_url:
            return
        self._dwelled_url = current
        await asyncio.sleep(random.uniform(self._dwell_min_ms, self._dwell_max_ms) / 1000.0)

    async def capture_state(self, screenshot_name: str) -> Tuple[str, str, BrowserState]:
        """Capture current browser state."""
        # Ensure we use the active page
        self.page = self.pages[self.active_tab_index]
        await self.page.bring_to_front()

        # Dismiss any extension overlay (e.g. SignalHire) that reappears after navigations.
        try:
            await self.page.keyboard.press("Escape")
        except Exception:
            pass

        screenshot_path = self.working_dir / "screenshots" / f"{screenshot_name}.webp"
        await self._safe_page_screenshot(path=screenshot_path, type="jpeg", quality=self.config.screenshot_quality)
        
        with Image.open(screenshot_path) as img:
            img_hash = str(imagehash.average_hash(img))
        
        viewport = self.page.viewport_size or {"width": self.config.screenshot_max_width, "height": 720}

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
                        # The "close" event (_on_page_closed) may have already
                        # removed p from self.pages by the time we get here —
                        # only do our own bookkeeping if it's still present.
                        if p in self.pages:
                            idx = self.pages.index(p)
                            self.pages.pop(idx)
                            if idx < self.active_tab_index:
                                self.active_tab_index -= 1
                except Exception:
                    pass

        tab_info_str = f"[Tab {self.active_tab_index + 1}/{len(self.pages)}]"
        full_title = f"{tab_info_str} {raw_title}"

        # Drain any auto-handled dialog events since last capture
        recent_dialogs = list(self._pending_dialogs)
        self._pending_dialogs.clear()

        # Deterministic dropdown scan (DOM mode only): native <select> popups
        # never render in screenshots, so surface candidates to the LLM.
        # Scans the main frame AND iframes (embedded checkout forms), pierces
        # open shadow roots, dedupes, caps at 8 entries.
        dropdowns: list = []
        if self.config.enable_dom_fallback:
            for frame in self._frame_search_order():
                raw = await self._safe_evaluate_frame(frame, _DROPDOWN_SCAN_JS, fallback="[]")
                try:
                    parsed = json.loads(raw) if isinstance(raw, str) else (raw or [])
                except Exception:
                    parsed = []
                for item in parsed:
                    if not item:
                        continue
                    entry = str(item)
                    if frame is not self.page.main_frame:
                        entry = f"[iframe: {(frame.url or '')[:60]}] {entry}"
                    if entry not in dropdowns:
                        dropdowns.append(entry)
                if len(dropdowns) >= 8:
                    break
            dropdowns = dropdowns[:8]

        state = BrowserState(
            url=self.page.url,
            title=full_title,
            viewport_width=viewport["width"],
            viewport_height=viewport["height"],
            scroll_y=await self._safe_evaluate("window.scrollY", fallback=0),
            screenshot_hash=img_hash,
            timestamp=datetime.now(),
            ready_state=await self._safe_evaluate("document.readyState", fallback="complete"),
            dom_signature=await self._safe_evaluate(_DOM_SIGNATURE_JS, fallback=""),
            dropdowns=dropdowns,
            notes = list(self.notes),  # Shallow copy — prevents DeleteNote/EditNote from mutating historical states
            large_notes_index = self._large_notes_index_snapshot(),
            tabs = tabs_info,
            dialogs = recent_dialogs,
        )

        # Refresh overlay state on the live page (no-op if overlay disabled).
        # The screenshot was already taken with the overlay hidden, so this
        # only affects what a human spectator sees in the live browser.
        if self.demo_overlay and self.demo_overlay.enabled:
            self.demo_overlay.update_metrics(
                mimo_calls=self.mimo_calls,
                dom_calls=self.dom_calls,
            )
            try:
                await self.demo_overlay.push(self.page)
            except Exception:
                pass

        return str(screenshot_path), img_hash, state
    
    async def execute_tool(self, tool_call: ToolCall, screenshot_path: str) -> ActionResult:
        """Execute a tool call (atomic action)."""
        start_time = time.time()
        self.total_actions += 1
        
        # Sync self.page with active tab
        if self.pages:
            self.page = self.pages[self.active_tab_index]
        
        try:
            if (
                not self.config.enable_dom_fallback
                and tool_call.action_type in {ActionType.DOM_CLICK, ActionType.DOM_TYPE, ActionType.DOM_EXTRACT, ActionType.SELECT_OPTION,
                                              ActionType.DOM_SNAPSHOT, ActionType.SNAPSHOT_CLICK, ActionType.SNAPSHOT_TYPE, ActionType.SNAPSHOT_SELECT}
            ):
                return ActionResult(
                    success=False,
                    action_type=tool_call.action_type,
                    description=f"{tool_call.action_type.value} disabled",
                    error="DOM tools are disabled in vision interaction mode.",
                )

            if tool_call.action_type == ActionType.PARSE_ERROR:
                result = ActionResult(
                    success=False,
                    action_type=tool_call.action_type,
                    description="Parse error",
                    error="No action to execute — the decision was not valid JSON.",
                )
            elif tool_call.action_type == ActionType.VISUAL_CLICK:
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
            elif tool_call.action_type == ActionType.DOM_TYPE:
                result = await self._dom_type(
                    tool_call.parameters["selector"],
                    tool_call.parameters["text"],
                    tool_call.parameters.get("press_enter", False),
                )
            elif tool_call.action_type == ActionType.DOM_EXTRACT:
                result = await self._dom_extract(tool_call.parameters["query"], tool_call.parameters.get("schema"), tool_call.parameters.get("max_results", 10))
            elif tool_call.action_type == ActionType.DOM_SNAPSHOT:
                result = await self._dom_snapshot(max_elements=tool_call.parameters.get("max_elements", 120))
            elif tool_call.action_type == ActionType.SNAPSHOT_CLICK:
                result = await self._snapshot_click(tool_call.parameters.get("ref"))
            elif tool_call.action_type == ActionType.SNAPSHOT_TYPE:
                result = await self._snapshot_type(
                    tool_call.parameters.get("ref"),
                    tool_call.parameters.get("text", ""),
                    tool_call.parameters.get("press_enter", False),
                )
            elif tool_call.action_type == ActionType.SNAPSHOT_SELECT:
                result = await self._snapshot_select(
                    tool_call.parameters.get("ref"),
                    value=tool_call.parameters.get("value"),
                    label=tool_call.parameters.get("label"),
                    index=tool_call.parameters.get("index"),
                    values=tool_call.parameters.get("values"),
                    labels=tool_call.parameters.get("labels"),
                )
            elif tool_call.action_type == ActionType.BATCH_EXTRACT:
                result = await self._batch_extract(tool_call.parameters)
            elif tool_call.action_type == ActionType.CREDENTIAL_FILL:
                result = await self._credential_fill(tool_call.parameters)
            elif tool_call.action_type == ActionType.REQUEST_CREDENTIALS:
                result = await self._request_credentials(tool_call.parameters)
            elif tool_call.action_type == ActionType.SELECT_OPTION:
                result = await self._dom_select(
                    tool_call.parameters["selector"],
                    value=tool_call.parameters.get("value"),
                    label=tool_call.parameters.get("label"),
                    index=tool_call.parameters.get("index"),
                    values=tool_call.parameters.get("values"),
                    labels=tool_call.parameters.get("labels"),
                )
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
                result = await self._ask_user(
                    tool_call.parameters["question"],
                    str(tool_call.parameters.get("kind") or ""),
                )
            else:
                result = ActionResult(success=False, action_type=tool_call.action_type, description="Unknown action", error=f"Unsupported: {tool_call.action_type}")
            
            result.execution_time_ms = (time.time() - start_time) * 1000
            return result
        except KeyError as e:
            # LLM omitted a required parameter (e.g. VisualClick with empty
            # parameters). Return an actionable error so the model can
            # self-correct immediately instead of guessing at "KeyError: 'x'".
            missing = str(e).strip("'\"")
            return ActionResult(
                success=False,
                action_type=tool_call.action_type,
                description=f"{tool_call.action_type.value} missing parameter",
                error=(
                    f"Missing required parameter {missing} for {tool_call.action_type.value}. "
                    f"Re-issue the SAME action with that parameter filled in — e.g. VisualClick needs "
                    f"'description' (what to click, with color/position/text cues), VisualType needs "
                    f"'field_description' + 'text'."
                ),
                execution_time_ms=(time.time() - start_time) * 1000,
            )
        except Exception as e:
            return ActionResult(success=False, action_type=tool_call.action_type, description=str(tool_call.parameters), error=str(e), execution_time_ms=(time.time() - start_time) * 1000)

    async def execute_indexed_action(
        self,
        tool_call: ToolCall,
        screenshot_path: str,
        visual_index: Optional[Dict[str, Any]] = None,
    ) -> ActionResult:
        """Execute an indexed replay action.

        VisualClick/VisualType/VisualHover can bypass MiMo by replaying the
        normalized visual index against the current viewport. Non-visual actions
        keep using the regular tool dispatcher.
        """
        if tool_call.action_type not in {
            ActionType.VISUAL_CLICK,
            ActionType.VISUAL_TYPE,
            ActionType.VISUAL_HOVER,
        }:
            return await self.execute_tool(tool_call, screenshot_path)

        try:
            self.total_actions += 1
            if self.pages:
                self.page = self.pages[self.active_tab_index]
            viewport = self.page.viewport_size or {"width": 1280, "height": 720}
            coords = replay_coordinates(
                visual_index or {},
                viewport_width=int(viewport["width"]),
                viewport_height=int(viewport["height"]),
            )
            if not coords:
                fallback = await self.execute_tool(tool_call, screenshot_path)
                fallback.metadata.setdefault("cosmic_indexed_replay", {})
                fallback.metadata["cosmic_indexed_replay"]["fallback"] = "missing_visual_index_used_mimo"
                return fallback

            start = time.time()
            x, y = coords
            if tool_call.action_type == ActionType.VISUAL_CLICK:
                await self._human_dwell_after_load()
                await self.cursor_overlay.show_click(self.page, x, y)
                await self._human_mouse_click(x, y)
                try:
                    await self.page.wait_for_load_state("domcontentloaded", timeout=2000)
                except Exception:
                    pass
                description = f"Indexed click at visual index for {visual_index.get('target_description', 'target') if visual_index else 'target'}"
            elif tool_call.action_type == ActionType.VISUAL_HOVER:
                await self.cursor_overlay.show_move(self.page, x, y)
                await self.page.mouse.move(x, y)
                await asyncio.sleep(0.3)
                description = f"Indexed hover at visual index for {visual_index.get('target_description', 'target') if visual_index else 'target'}"
            else:
                params = tool_call.parameters or {}
                await self._human_dwell_after_load()
                await self.cursor_overlay.show_click(self.page, x, y)
                await self._human_mouse_click(x, y)
                await asyncio.sleep(0.2)
                await self.cursor_overlay.show_typing_start(self.page, x, y)
                await self.page.keyboard.press("Control+A")
                await self.page.keyboard.press("Backspace")
                await self._human_type(str(params.get("text", "")))
                await self.cursor_overlay.show_typing_stop(self.page)
                if params.get("press_enter", False):
                    await self.cursor_overlay.show_key(self.page, "Enter")
                    await self.page.keyboard.press("Enter")
                    try:
                        await self.page.wait_for_load_state("domcontentloaded", timeout=2000)
                    except Exception:
                        pass
                description = f"Indexed type into visual index for {visual_index.get('target_description', 'field') if visual_index else 'field'}"

            return ActionResult(
                success=True,
                action_type=tool_call.action_type,
                description=description,
                coordinates=(x, y),
                execution_time_ms=(time.time() - start) * 1000,
                metadata={
                    "cosmic_indexed_replay": {
                        "used_visual_index": True,
                        "visual_index": visual_index,
                    }
                },
            )
        except Exception as e:
            return ActionResult(
                success=False,
                action_type=tool_call.action_type,
                description=f"Indexed replay {tool_call.action_type.value}",
                error=str(e),
            )

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
            await new_page.goto(url, wait_until="domcontentloaded")
            self._register_page(new_page, make_active=True)
            await self.page.bring_to_front()
            return ActionResult(success=True, action_type=ActionType.NEW_TAB, description=f"Opened new tab: {url}")
        except Exception as e: return ActionResult(success=False, action_type=ActionType.NEW_TAB, description=f"Open tab {url}", error=str(e))

    async def _switch_tab(self, index: int) -> ActionResult:
        try:
            if 0 <= index < len(self.pages):
                self.active_tab_index = index
                self.page = self.pages[index]
                self._schedule_live_screencast_retarget()
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

                # The page's "close" event (_on_page_closed) may have already
                # removed it from self.pages by the time we get here — only
                # do our own bookkeeping if it's still present, to avoid a
                # double-pop / stale-index race with that listener.
                if page_to_close in self.pages:
                    idx = self.pages.index(page_to_close)
                    self.pages.pop(idx)
                    if not self.pages:
                        # No pages left, open a blank one
                        self.page = await self.context.new_page()
                        self._register_page(self.page, make_active=True)
                    elif idx <= self.active_tab_index:
                        # If we closed current or previous tab, shift left
                        self.active_tab_index = max(0, self.active_tab_index - 1)
                        self.page = self.pages[self.active_tab_index]
                        self._schedule_live_screencast_retarget()

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

    # Kinds the model may voluntarily attach to an AskUser call, hinting what
    # kind of answer widget the desktop should show (a code field vs. a
    # plain "done" button vs. free text). "password" is deliberately not
    # offered here — the credential governor (force_credential_handoff)
    # already forces that one deterministically from the DOM, which is more
    # reliable than asking the model to self-report it.
    _ASK_USER_MODEL_KINDS = {"verification_code", "confirm", "blocked", "generic"}

    async def _ask_user(self, question: str, kind: str = "") -> ActionResult:
        """Ask the user a question.

        Routing:
        - If `ask_user_handler` was injected (e.g. by `call_to_browse.py`), delegate
          to it. Used for voice-driven Q&A while on a live call.
        - Else: prompt via CLI stdin (the default for `python main.py`).

        In both paths we also annotate the demo overlay so reviewers see exactly
        when the agent asked and what the user said.

        `kind` is an optional hint for how the caller should present the
        question (its own decision loop already knows why it's asking — this
        just carries that along instead of making a caller re-guess it from
        the question text). Anything the caller doesn't recognize is safe to
        ignore; unset/invalid values are simply passed through as "".
        """
        question_text = (question or "").strip()
        normalized_kind = str(kind or "").strip().lower()
        if normalized_kind not in self._ASK_USER_MODEL_KINDS and normalized_kind != "password":
            normalized_kind = ""
        truncated_q = question_text if len(question_text) <= 60 else question_text[:57] + "..."

        # Always reflect the ask in the overlay (no-op when overlay is disabled).
        if self.demo_overlay is not None:
            try:
                await self.demo_overlay.update(
                    page=self.page,
                    pulse_ms=1500,
                    timeline_append={"kind": "live", "label": f"Asking user: {truncated_q}"},
                )
            except Exception:
                pass

        # --- Path 1: injected handler (voice / network) ------------------------
        if self.ask_user_handler is not None:
            try:
                print(f"\n❓ [Agent Asks]: {question}", flush=True)
                response = await asyncio.wait_for(
                    self.ask_user_handler(question, normalized_kind),
                    timeout=self.config.ask_user_timeout,
                )
            except asyncio.TimeoutError:
                print(f"\n⏰ [AskUser] No response after {self.config.ask_user_timeout}s - moving on.", flush=True)
                if self.demo_overlay is not None:
                    try:
                        await self.demo_overlay.update(
                            page=self.page,
                            timeline_append={"kind": "checkpoint", "label": "User replied: (timed out)"},
                        )
                    except Exception:
                        pass
                return ActionResult(
                    success=False,
                    action_type=ActionType.ASK_USER,
                    description=f"Asked: {question}",
                    error=f"User did not respond within {self.config.ask_user_timeout}s timeout.",
                )
            except Exception as e:  # noqa: BLE001 — surface handler errors as ActionResult, never raise
                return ActionResult(
                    success=False,
                    action_type=ActionType.ASK_USER,
                    description=f"Asked: {question}",
                    error=f"ask_user_handler failed: {e}",
                )

            reply_str = (response or "").strip()
            display_reply = reply_str if reply_str else "(empty response)"
            truncated_r = display_reply if len(display_reply) <= 60 else display_reply[:57] + "..."
            print(f"💬 [User Replied]: {display_reply}", flush=True)
            if self.demo_overlay is not None:
                try:
                    await self.demo_overlay.update(
                        page=self.page,
                        pulse_ms=1500,
                        timeline_append={"kind": "saved", "label": f"User replied: {truncated_r}"},
                    )
                except Exception:
                    pass
            return ActionResult(
                success=True,
                action_type=ActionType.ASK_USER,
                description=f"Asked: {question}",
                output=f"User Answer: {display_reply}",
            )

        # --- Path 2: CLI stdin (original behavior) -----------------------------
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

            reply_str = response.strip() if response.strip() else "(empty response)"
            truncated_r = reply_str if len(reply_str) <= 60 else reply_str[:57] + "..."
            if self.demo_overlay is not None:
                try:
                    await self.demo_overlay.update(
                        page=self.page,
                        pulse_ms=1500,
                        timeline_append={"kind": "saved", "label": f"User replied: {truncated_r}"},
                    )
                except Exception:
                    pass

            return ActionResult(
                success=True,
                action_type=ActionType.ASK_USER,
                description=f"Asked: {question}",
                output=f"User Answer: {reply_str}",
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
        await self._human_dwell_after_load()
        await self.cursor_overlay.show_click(self.page, x, y)
        await self._human_mouse_click(x, y)
        try: await self.page.wait_for_load_state("domcontentloaded", timeout=2000)
        except: pass
        return ActionResult(success=True, action_type=ActionType.VISUAL_CLICK, description=description, coordinates=(x, y), metadata={"mimo_grounding": dict(self.last_mimo_grounding or {})})

    async def _visual_hover(self, screenshot_path: str, description: str, region_hint: Optional[str] = None) -> ActionResult:
        """Hover over an element without clicking (for dropdowns, tooltips, menus)."""
        coords = await self._call_mimo_grounding(screenshot_path, description)
        if not coords:
            return ActionResult(success=False, action_type=ActionType.VISUAL_HOVER, description=description, error="MiMo failed to find element")
        x, y = coords
        await self.cursor_overlay.show_move(self.page, x, y)
        await self.page.mouse.move(x, y)
        await asyncio.sleep(0.3)  # Wait for hover effects to render
        return ActionResult(success=True, action_type=ActionType.VISUAL_HOVER, description=f"Hovered over: {description}", coordinates=(x, y), metadata={"mimo_grounding": dict(self.last_mimo_grounding or {})})

    async def _visual_type(self, screenshot_path: str, field_description: str, text: str, press_enter: bool = False) -> ActionResult:
        coords = await self._call_mimo_grounding(screenshot_path, field_description)
        if not coords:
            return ActionResult(success=False, action_type=ActionType.VISUAL_TYPE, description=f"Type '{text}'", error="MiMo failed to find field")
        x, y = coords
        await self._human_dwell_after_load()
        await self.cursor_overlay.show_click(self.page, x, y)
        await self._human_mouse_click(x, y)
        await asyncio.sleep(0.2)
        await self.cursor_overlay.show_typing_start(self.page, x, y)
        await self.page.keyboard.press("Control+A")
        await self.page.keyboard.press("Backspace")
        await self._human_type(text)
        await self.cursor_overlay.show_typing_stop(self.page)
        if press_enter:
            await self.cursor_overlay.show_key(self.page, "Enter")
            await self.page.keyboard.press("Enter")
            try: await self.page.wait_for_load_state("domcontentloaded", timeout=2000)
            except: pass
        return ActionResult(success=True, action_type=ActionType.VISUAL_TYPE, description=f"Typed '{text}'", coordinates=(x, y), metadata={"mimo_grounding": dict(self.last_mimo_grounding or {})})

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
                pixels = amount * 500 if 1 <= amount <= 10 else amount
            elif isinstance(amount, str) and amount.lower() in scroll_map:
                pixels = scroll_map[amount.lower()]
            elif isinstance(amount, str) and amount.isdigit():
                numeric_amount = int(amount)
                pixels = numeric_amount * 500 if 1 <= numeric_amount <= 10 else numeric_amount
            
        # 2. Get initial scroll position
        start_y = await self.page.evaluate("window.scrollY")

        await self.cursor_overlay.show_scroll(self.page, direction_lower)

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
    
    @staticmethod
    def _is_playwright_selector(selector: str) -> bool:
        """Return True if selector uses Playwright-specific syntax that querySelectorAll can't handle."""
        import re as _re
        _PW_PATTERNS = (
            r":has-text\(",
            r":text\(",
            r":text-is\(",
            r":text-matches\(",
            r"^text=",
            r"^css=",
            r"^xpath=",
            r":visible",
            r":nth-match\(",
        )
        return any(_re.search(p, selector) for p in _PW_PATTERNS)

    def _frame_search_order(self):
        """Main frame first, then up to 9 child iframes. Many SSO widgets
        (Google Identity Services 'Continue with Google' button, etc.) render
        inside a cross-origin iframe that document.querySelectorAll on the
        main frame can never see — but Playwright has CDP-level access into
        each frame independently, so we can search them directly."""
        main = self.page.main_frame
        others = [f for f in self.page.frames if f is not main][:9]
        return [main] + others

    async def _unique_visible_locator(self, frame, selector: str):
        """Resolve a Playwright selector to the locator for a UNIQUELY visible
        match — the same one-visible-field rule the CSS path enforces.

        More than one visible match raises _AmbiguousTargetError carrying the
        model-facing refusal (with the matched fields' labels). Exactly one
        visible match returns THAT element's locator — not .first, whose
        target can be a hidden template copy while the real field sits
        elsewhere in the DOM. Zero visible matches returns .first so the
        ordinary scroll/click path raises its usual not-found error.
        """
        locator_all = frame.locator(selector)
        try:
            total = await locator_all.count()
        except Exception:
            return locator_all.first
        if total <= 1:
            return locator_all.first
        visible: List[int] = []
        for i in range(min(total, 12)):
            try:
                if await locator_all.nth(i).is_visible():
                    visible.append(i)
            except Exception:
                continue
        if len(visible) > 1:
            fields: List[Dict[str, Any]] = []
            for i in visible[:8]:
                try:
                    info = await locator_all.nth(i).evaluate(
                        "el => ({label: (%s)(el), value: String(el.value == null ? '' : el.value)})" % _FIELD_LABEL_JS
                    )
                except Exception:
                    info = {}
                fields.append(info if isinstance(info, dict) else {})
            raise _AmbiguousTargetError(_ambiguous_type_error(selector, len(visible), fields))
        if visible:
            return locator_all.nth(visible[0])
        return locator_all.first

    async def _read_type_echo(self, frame) -> Optional[Dict[str, Any]]:
        """Label + value of whatever element keyboard focus actually landed on.
        Best effort: a page that moved focus out from under the typed element
        (Enter submitted a form, an SPA re-rendered) yields None and the
        action result simply carries no echo."""
        try:
            return await frame.evaluate(_TYPE_ECHO_JS)
        except Exception:
            return None

    async def _dom_snapshot(self, max_elements: int = 120) -> ActionResult:
        """Perceive the page as a numbered map of its visible, enabled
        interactive elements — role, visible name, and state per @e ref — so
        one text pull replaces per-action visual grounding on structured
        pages. A peer tool, not a replacement: the model still chooses vision
        (canvas, odd widgets, visual verification) or raw selectors (known
        unique anchors) whenever those fit better. Only the latest snapshot's
        refs resolve; every ref is fingerprint-checked against the live DOM
        at act time, so a page that moved on refuses the old map instead of
        clicking a stranger."""
        cap = max(1, min(int(max_elements or 120), 300))
        frames = self._frame_search_order()
        self._snapshot_refs = {}
        lines: List[str] = []
        ref_counter = 0
        total = 0
        truncated = False
        per_frame_counts: List[int] = []
        for frame_index, frame in enumerate(frames):
            try:
                result = await frame.evaluate(_SNAPSHOT_COLLECT_JS, {
                    "css": _SNAPSHOT_INTERACTIVE_CSS,
                    "cap": cap - total,
                })
            except Exception:
                continue  # cross-origin frame that refuses injection, etc.
            if not isinstance(result, dict) or result.get("error"):
                continue
            entries = result.get("entries") or []
            per_frame_counts.append(len(entries))
            for entry in entries:
                if entry.get("truncated"):
                    truncated = True
                    break
                ref_counter += 1
                ref = f"@e{ref_counter}"
                self._snapshot_refs[ref] = {
                    "frame_index": frame_index,
                    "nth": total,
                    "role": str(entry.get("role") or "element"),
                    "name": str(entry.get("name") or ""),
                    "fingerprint": snapshot_fingerprint(entry),
                }
                lines.extend(format_snapshot_lines([entry], start_ref=ref_counter))
                total += 1
            if truncated or total >= cap:
                break
        if total == 0:
            return ActionResult(
                success=False,
                action_type=ActionType.DOM_SNAPSHOT,
                description="DOMSnapshot",
                output=None,
                error="No visible interactive elements found on this page. Use the screenshot and vision tools instead.",
            )
        header = f"{total} interactive elements on {(self.page.url or '')[:100]}:"
        if truncated:
            header += f" (truncated at {cap} — scroll and snapshot again for more)"
        return ActionResult(
            success=True,
            action_type=ActionType.DOM_SNAPSHOT,
            description=f"Snapshotted {total} interactive elements",
            output="\n".join([header] + lines),
            metadata={
                "refs": total,
                "frames": len(per_frame_counts),
                "truncated": truncated,
            },
        )

    async def _snapshot_resolve(self, ref: Any, action_label: str):
        """Resolve an @e ref to (locator, frame, info) — or raise
        _SnapshotStaleError with a model-facing reason. The fingerprint
        re-check makes acting on a ref prove it is still the same element;
        typing between snapshot and act never trips it (values are not part
        of the fingerprint)."""
        clean_ref = parse_ref(ref)
        if not clean_ref:
            raise _SnapshotStaleError(
                f"{action_label} needs an @e ref from DOMSnapshot output (e.g. '@e5'), got {ref!r}."
            )
        info = self._snapshot_refs.get(clean_ref)
        if not info:
            raise _SnapshotStaleError(
                f"{clean_ref} is not in the current snapshot map — never taken, or a newer DOMSnapshot replaced it. Call DOMSnapshot."
            )
        frames = self._frame_search_order()
        if info["frame_index"] >= len(frames):
            raise _SnapshotStaleError(f"{clean_ref}'s frame is gone — call DOMSnapshot again.")
        frame = frames[info["frame_index"]]
        try:
            check = await frame.evaluate(_SNAPSHOT_RECHECK_JS, {
                "css": _SNAPSHOT_INTERACTIVE_CSS,
                "nth": info["nth"],
                "fingerprint": info["fingerprint"],
            })
        except Exception as exc:
            raise _SnapshotStaleError(f"Could not re-check {clean_ref} ({exc}) — call DOMSnapshot.") from exc
        if not check or not check.get("ok"):
            reason = (check or {}).get("reason") or "changed"
            raise _SnapshotStaleError(
                f"{clean_ref} is stale ({reason}) — the page moved on since the snapshot. Call DOMSnapshot again and act on the fresh refs."
            )
        if not check.get("visible"):
            raise _SnapshotStaleError(
                f"{clean_ref} is no longer visible — call DOMSnapshot again."
            )
        locator = frame.locator(_SNAPSHOT_INTERACTIVE_CSS).nth(int(check.get("all_index") or 0))
        return locator, frame, info

    async def _snapshot_click(self, ref: Any) -> ActionResult:
        """Click the element a snapshot ref points at. The result names what
        was clicked (role + visible name) so the model sees its own aim, and
        the standard verification layer judges the landing."""
        try:
            locator, frame, info = await self._snapshot_resolve(ref, "SnapshotClick")
        except _SnapshotStaleError as stale:
            return ActionResult(success=False, action_type=ActionType.SNAPSHOT_CLICK, description=f"SnapshotClick {ref}", error=str(stale))
        name = str(info.get("name") or "").strip()
        role = str(info.get("role") or "element")
        try:
            await locator.scroll_into_view_if_needed(timeout=2000)
            box = await locator.bounding_box(timeout=2000)
            if box:
                await self.cursor_overlay.show_click(self.page, int(box["x"] + box["width"] / 2), int(box["y"] + box["height"] / 2))
            await locator.click(timeout=2500)
            x = int(box["x"] + box["width"] / 2) if box else 0
            y = int(box["y"] + box["height"] / 2) if box else 0
            named = f" {role} '{name}'" if name else ""
            return ActionResult(
                success=True,
                action_type=ActionType.SNAPSHOT_CLICK,
                description=f"Clicked {parse_ref(ref)}{named}",
                coordinates=(x, y),
            )
        except Exception as exc:
            return ActionResult(
                success=False,
                action_type=ActionType.SNAPSHOT_CLICK,
                description=f"SnapshotClick {ref}",
                error=f"Click on {parse_ref(ref)} failed: {exc}",
            )

    async def _snapshot_type(self, ref: Any, text: str, press_enter: bool = False) -> ActionResult:
        """Type into the text field a snapshot ref points at — real keyboard
        events (clear, then type), the same overwrite warning and landing
        echo as DomType, so a @ref type is never a blind write."""
        try:
            locator, frame, info = await self._snapshot_resolve(ref, "SnapshotType")
        except _SnapshotStaleError as stale:
            return ActionResult(success=False, action_type=ActionType.SNAPSHOT_TYPE, description=f"SnapshotType {ref}", error=str(stale))
        fingerprint = info["fingerprint"]
        target_name = str(fingerprint.get("name") or "").strip()
        try:
            kind = await locator.evaluate(
                "el => ({tag: el.tagName.toLowerCase(), type: (el.getAttribute('type') || '').toLowerCase(), editable: !!el.isContentEditable})"
            )
        except Exception:
            kind = {}
        tag = str((kind or {}).get("tag") or "")
        input_type = str((kind or {}).get("type") or "")
        if tag not in ("input", "textarea") and not (kind or {}).get("editable"):
            return ActionResult(
                success=False,
                action_type=ActionType.SNAPSHOT_TYPE,
                description=f"SnapshotType {ref}",
                error=f"{parse_ref(ref)} is a '{fingerprint.get('name') or tag}' {tag}, not a text field — use SnapshotClick, or SelectOption/SnapshotSelect for dropdowns.",
            )
        if tag == "input" and input_type in ("checkbox", "radio", "button", "submit", "reset", "file", "image", "range", "color"):
            return ActionResult(
                success=False,
                action_type=ActionType.SNAPSHOT_TYPE,
                description=f"SnapshotType {ref}",
                error=f"{parse_ref(ref)} is an {input_type or 'unknown-type'} input, not a text field.",
            )
        try:
            await locator.scroll_into_view_if_needed(timeout=2000)
            box = await locator.bounding_box(timeout=2000)
            if box:
                await self.cursor_overlay.show_click(self.page, int(box["x"] + box["width"] / 2), int(box["y"] + box["height"] / 2))
            await locator.click(timeout=2000)
            if box:
                await self.cursor_overlay.show_typing_start(self.page, int(box["x"] + box["width"] / 2), int(box["y"] + box["height"] / 2))
            previous_value = ""
            try:
                previous_value = await locator.input_value()
            except Exception:
                previous_value = ""
            await self.page.keyboard.press("Control+A")
            await self.page.keyboard.press("Backspace")
            await self._human_type(text)
            await self.cursor_overlay.show_typing_stop(self.page)
            if press_enter:
                await self.cursor_overlay.show_key(self.page, "Enter")
                await self.page.keyboard.press("Enter")
                try: await self.page.wait_for_load_state("domcontentloaded", timeout=2000)
                except: pass
            echo = await self._read_type_echo(frame)
            label = str((echo or {}).get("label") or target_name or "")
            is_secret = input_type == "password"
            warning = None if is_secret else _overwrite_warning(text, previous_value)
            description = f"Typed into {parse_ref(ref)}"
            if target_name:
                description += f" '{target_name[:60]}'"
            description += f": '{_secret_display(text, is_secret, 120)}'"
            if label and label != target_name:
                description += f" — field labeled '{label[:80]}'"
            if warning:
                description += f" (WARNING: {warning} — if this is not the field you meant, refill the correct field and restore this one)"
            return ActionResult(
                success=True,
                action_type=ActionType.SNAPSHOT_TYPE,
                description=description,
                coordinates=(int(box["x"] + box["width"] / 2), int(box["y"] + box["height"] / 2)) if box else None,
                output=json.dumps({
                    "typed_into": {
                        "ref": parse_ref(ref),
                        "label": label or None,
                        "previous_value": _secret_display(previous_value, is_secret) or None,
                        "value": _secret_display((echo or {}).get("value"), is_secret) or None,
                        **({"warning": warning} if warning else {}),
                    }
                }),
                metadata={
                    "typed_into_label": label or None,
                    "previous_value": _secret_display(previous_value, is_secret) or None,
                    "overwrite_warning": warning,
                },
            )
        except Exception as exc:
            return ActionResult(
                success=False,
                action_type=ActionType.SNAPSHOT_TYPE,
                description=f"SnapshotType {ref}",
                error=f"Type into {parse_ref(ref)} failed: {exc}",
            )

    async def _snapshot_select(self, ref: Any, value: Optional[str] = None, label: Optional[str] = None,
                               index: Optional[int] = None, values: Optional[List[str]] = None,
                               labels: Optional[List[str]] = None) -> ActionResult:
        """Select option(s) in the <select> a snapshot ref points at — the
        deterministic dropdown primitive aimed by ref instead of selector."""
        try:
            locator, frame, info = await self._snapshot_resolve(ref, "SnapshotSelect")
        except _SnapshotStaleError as stale:
            return ActionResult(success=False, action_type=ActionType.SNAPSHOT_SELECT, description=f"SnapshotSelect {ref}", error=str(stale))
        target_name = str(info["fingerprint"].get("name") or "").strip()
        try:
            chosen = await self._select_via_locator(
                locator,
                value=value, label=label, index=index, values=values, labels=labels,
            )
        except Exception as exc:
            return ActionResult(
                success=False,
                action_type=ActionType.SNAPSHOT_SELECT,
                description=f"SnapshotSelect {ref}",
                error=f"Select in {parse_ref(ref)} failed: {exc}",
            )
        named = f" '{target_name[:60]}'" if target_name else ""
        return ActionResult(
            success=True,
            action_type=ActionType.SNAPSHOT_SELECT,
            description=f"Selected '{chosen}' in {parse_ref(ref)}{named}",
            output=f"selected={chosen}",
        )

    @staticmethod
    def _has_text_query(selector: str) -> Optional[str]:
        """Pull the quoted string out of a :has-text("...") clause, if present."""
        m = re.search(r':has-text\(\s*["\']([^"\']+)["\']\s*\)', selector)
        return m.group(1) if m else None

    def _tag_fallback_selectors(self, selector: str) -> List[str]:
        """If selector is `<tag>:has-text("X")` and the tag guess is wrong, the
        locator finds nothing — with no signal to the LLM about *why*. Rather
        than make every prompt enumerate every site's quirky button markup
        (Google's GSI widget is a `<div role="button">`, not a <button>; other
        sites use <a>, [tabindex], etc.), try the common alternatives
        automatically before giving up. Self-healing instead of guess-and-pray."""
        text = self._has_text_query(selector)
        if not text:
            return []
        candidates = [
            f'[role="button"]:has-text("{text}")',
            f'a:has-text("{text}")',
            f'[tabindex]:has-text("{text}")',
            f'button:has-text("{text}")',
        ]
        return [c for c in candidates if c != selector]

    async def _dom_click(self, selector: str) -> ActionResult:
        self.dom_calls += 1
        await self._human_dwell_after_load()
        frames = self._frame_search_order()
        try:
            # Playwright locator path: handles :has-text(), text=, xpath=, etc.
            if self._is_playwright_selector(selector):
                selectors_to_try = [selector] + self._tag_fallback_selectors(selector)
                last_err = None
                tried_count = 0
                for sel in selectors_to_try:
                    for frame in frames:
                        tried_count += 1
                        try:
                            locator = frame.locator(sel).first
                            await locator.scroll_into_view_if_needed(timeout=2000)
                            box = await locator.bounding_box(timeout=2000)
                            if box:
                                await self.cursor_overlay.show_click(self.page, int(box["x"] + box["width"] / 2), int(box["y"] + box["height"] / 2))
                            await locator.click(timeout=2000)
                            text = (await locator.inner_text(timeout=1500)) if box else ""
                            x = int(box["x"] + box["width"] / 2) if box else 0
                            y = int(box["y"] + box["height"] / 2) if box else 0
                            frame_note = "" if frame is self.page.main_frame else f" [iframe: {frame.url[:60]}]"
                            tag_note = "" if sel == selector else f" (auto-corrected tag: {sel})"
                            return ActionResult(
                                success=True,
                                action_type=ActionType.DOM_CLICK,
                                description=f"Clicked (locator){frame_note}{tag_note}: {selector}",
                                coordinates=(x, y),
                                output=text.strip() or None,
                            )
                        except Exception as pw_err:
                            last_err = pw_err
                            continue
                variant_note = f" across {len(selectors_to_try)} tag variant(s)" if len(selectors_to_try) > 1 else ""
                return ActionResult(
                    success=False,
                    action_type=ActionType.DOM_CLICK,
                    description=f"Click {selector}",
                    error=f"Not found in main frame or {len(frames) - 1} iframe(s){variant_note} ({tried_count} attempts): {last_err}",
                )

            # Standard CSS selector path via querySelectorAll — searched in
            # every frame for the same cross-origin-iframe reason as above.
            js_script = """
            (selector) => {
                const cleanText = (value) => {
                    if (value === null || value === undefined) return "";
                    return String(value).replace(/\\s+/g, " ").trim();
                };
                const isVisible = (el) => {
                    const style = window.getComputedStyle(el);
                    if (!style || style.display === "none" || style.visibility === "hidden" || Number(style.opacity) === 0) {
                        return false;
                    }
                    if (el.closest("[hidden], [aria-hidden='true']")) return false;
                    const rect = el.getBoundingClientRect();
                    return rect.width > 0 && rect.height > 0;
                };

                let elements = [];
                try {
                    elements = Array.from(document.querySelectorAll(selector));
                } catch (error) {
                    return {ok: false, error: `Invalid selector: ${error.message || error}`};
                }

                for (const el of elements) {
                    if (!isVisible(el)) continue;
                    if (el.disabled || el.getAttribute("aria-disabled") === "true") continue;
                    el.scrollIntoView({block: "center", inline: "center", behavior: "auto"});
                    const rect = el.getBoundingClientRect();
                    const x = rect.left + rect.width / 2;
                    const y = rect.top + rect.height / 2;
                    el.click();
                    return {
                        ok: true,
                        text: cleanText(el.innerText || el.textContent || el.getAttribute("aria-label") || el.getAttribute("title")),
                        x,
                        y,
                    };
                }

                return {ok: false, error: `No visible enabled element found for selector. Matched ${elements.length} elements.`};
            }
            """
            last_result = None
            for frame in frames:
                try:
                    result = await frame.evaluate(js_script, selector)
                except Exception as e:
                    result = {"ok": False, "error": str(e)}
                if result and result.get("ok"):
                    frame_note = "" if frame is self.page.main_frame else f" [iframe: {frame.url[:60]}]"
                    await self.cursor_overlay.show_click(self.page, int(result.get("x", 0)), int(result.get("y", 0)))
                    return ActionResult(
                        success=True,
                        action_type=ActionType.DOM_CLICK,
                        description=f"Clicked visible element for selector{frame_note}: {selector}",
                        coordinates=(int(result.get("x", 0)), int(result.get("y", 0))),
                        output=result.get("text") or None,
                    )
                last_result = result
            return ActionResult(
                success=False,
                action_type=ActionType.DOM_CLICK,
                description=f"Click {selector}",
                error=(last_result or {}).get("error", "No visible enabled element found in main frame or any iframe."),
            )
        except Exception as e: return ActionResult(success=False, action_type=ActionType.DOM_CLICK, description=f"Click {selector}", error=str(e))

    async def _dom_select(
        self,
        selector: str,
        value: Optional[str] = None,
        label: Optional[str] = None,
        index: Optional[int] = None,
        values: Optional[List[str]] = None,
        labels: Optional[List[str]] = None,
    ) -> ActionResult:
        """Select option(s) in a native <select> — the deterministic dropdown
        primitive (native option popups are invisible to screenshots, so vision
        cannot ground them). Mirrors _dom_click's frame-search structure:
        main frame first, then iframes. Accepts value (option value), label
        (visible option text), zero-based index, or values/labels lists for
        <select multiple>; fires real change events, so JS-framework-controlled
        selects update too."""
        self.dom_calls += 1
        await self._human_dwell_after_load()
        if value is None and label is None and index is None and not values and not labels:
            return ActionResult(
                success=False,
                action_type=ActionType.SELECT_OPTION,
                description=f"Select {selector}",
                error="Provide one of: value, label, index, values (list for multi-select), or labels (list for multi-select).",
            )
        frames = self._frame_search_order()
        try:
            async def _try_select(frame):
                loc = await self._unique_visible_locator(frame, selector)
                if not await loc.is_visible():
                    return None
                return await self._select_via_locator(
                    loc, value=value, label=label, index=index, values=values, labels=labels,
                )
            last_err: Optional[str] = "no visible <select> matched the selector in any frame"
            ambiguity_err: Optional[str] = None
            for frame in frames:
                try:
                    chosen = await _try_select(frame)
                except _AmbiguousTargetError as amb:
                    # A selector matching several visible dropdowns must not
                    # resolve to .first — the wrong dropdown silently accepts
                    # the option. Prefer this refusal over a plain not-found
                    # when no frame resolves the selector uniquely.
                    if ambiguity_err is None:
                        ambiguity_err = str(amb)
                    continue
                except Exception as exc:
                    last_err = str(exc) or repr(exc)
                    continue
                if chosen is not None:
                    frame_note = "" if frame is self.page.main_frame else f" [iframe: {frame.url[:60]}]"
                    return ActionResult(
                        success=True,
                        action_type=ActionType.SELECT_OPTION,
                        description=f"Selected option{frame_note}: {chosen}",
                        output=f"selected={chosen}",
                    )
            return ActionResult(
                success=False,
                action_type=ActionType.SELECT_OPTION,
                description=f"Select {selector}",
                error=ambiguity_err or f"Could not select in main frame or any iframe: {last_err}",
            )
        except Exception as e:
            return ActionResult(success=False, action_type=ActionType.SELECT_OPTION, description=f"Select {selector}", error=str(e))

    async def _select_via_locator(self, loc, value: Optional[str] = None, label: Optional[str] = None,
                                  index: Optional[int] = None, values: Optional[List[str]] = None,
                                  labels: Optional[List[str]] = None) -> str:
        """Pick option(s) on an already-resolved <select> locator and return
        the chosen option's visible text. Shared by SelectOption and
        SnapshotSelect; fires real change events so JS-framework selects
        update too."""
        if values:
            try:
                await loc.select_option(value=values, timeout=5000)
            except Exception:
                await loc.select_option(label=values, timeout=5000)
        elif labels:
            try:
                await loc.select_option(label=labels, timeout=5000)
            except Exception:
                await loc.select_option(value=labels, timeout=5000)
        elif value is not None:
            try:
                await loc.select_option(value=value, timeout=5000)
            except Exception:
                # Value didn't match — retry as a label (visible text).
                await loc.select_option(label=value, timeout=5000)
        elif label is not None:
            try:
                await loc.select_option(label=str(label), timeout=5000)
            except Exception:
                await loc.select_option(value=label, timeout=5000)
        else:
            await loc.select_option(index=int(index), timeout=5000)
        chosen = await loc.evaluate(
            "el => (el.selectedOptions && el.selectedOptions[0] ? "
            "String(el.selectedOptions[0].text).trim() : String(el.value))"
        )
        box = await loc.bounding_box()
        if box:
            await self.cursor_overlay.show_click(self.page, int(box["x"] + 20), int(box["y"] + box["height"] / 2))
        return chosen

    async def _dom_type(self, selector: str, text: str, press_enter: bool = False) -> ActionResult:
        """Type into an element matched by a CSS/Playwright selector. Mirrors
        _dom_click's frame-search + tag-fallback structure, but types with
        real keyboard events (like _visual_type) rather than setting .value
        directly — more broadly compatible with JS-framework-driven inputs
        that rely on input/keydown listeners for validation or autocomplete.

        Wrong-target discipline: typing is destructive (it clears whatever the
        field held), and forms routinely reuse one placeholder across many
        fields, so a selector matching several VISIBLE fields is refused with
        their labels rather than resolved to the first match — that guess is
        how one field's value ends up in another. A unique match types, then
        the result echoes the label of the field the keystrokes actually
        landed in and what value it replaced."""
        self.dom_calls += 1
        await self._human_dwell_after_load()
        frames = self._frame_search_order()
        try:
            if self._is_playwright_selector(selector):
                selectors_to_try = [selector] + self._tag_fallback_selectors(selector)
                last_err = None
                ambiguous_err = None
                tried_count = 0
                for sel in selectors_to_try:
                    for frame in frames:
                        tried_count += 1
                        try:
                            locator = await self._unique_visible_locator(frame, sel)
                        except _AmbiguousTargetError as amb:
                            ambiguous_err = amb
                            continue
                        except Exception as count_err:
                            last_err = count_err
                            continue
                        try:
                            await locator.scroll_into_view_if_needed(timeout=2000)
                            box = await locator.bounding_box(timeout=2000)
                            if box:
                                await self.cursor_overlay.show_click(self.page, int(box["x"] + box["width"] / 2), int(box["y"] + box["height"] / 2))
                            await locator.click(timeout=2000)
                            if box:
                                await self.cursor_overlay.show_typing_start(self.page, int(box["x"] + box["width"] / 2), int(box["y"] + box["height"] / 2))
                            previous_value = ""
                            try:
                                previous_value = await locator.input_value()
                            except Exception:
                                previous_value = ""
                            await self.page.keyboard.press("Control+A")
                            await self.page.keyboard.press("Backspace")
                            await self._human_type(text)
                            await self.cursor_overlay.show_typing_stop(self.page)
                            if press_enter:
                                await self.cursor_overlay.show_key(self.page, "Enter")
                                await self.page.keyboard.press("Enter")
                                try: await self.page.wait_for_load_state("domcontentloaded", timeout=2000)
                                except: pass
                            echo = await self._read_type_echo(frame)
                            x = int(box["x"] + box["width"] / 2) if box else 0
                            y = int(box["y"] + box["height"] / 2) if box else 0
                            frame_note = "" if frame is self.page.main_frame else f" [iframe: {frame.url[:60]}]"
                            tag_note = "" if sel == selector else f" (auto-corrected tag: {sel})"
                            label = (echo or {}).get("label") or ""
                            try:
                                is_secret = bool(await locator.evaluate("el => el.tagName === 'INPUT' && el.type === 'password'"))
                            except Exception:
                                is_secret = False
                            description, warning = _type_echo_description(selector, label, text, previous_value, frame_note + tag_note, is_secret=is_secret)
                            return ActionResult(
                                success=True,
                                action_type=ActionType.DOM_TYPE,
                                description=description,
                                coordinates=(x, y),
                                output=json.dumps({
                                    "typed_into": {
                                        "label": label or None,
                                        "previous_value": _secret_display(previous_value, is_secret) or None,
                                        "value": _secret_display((echo or {}).get("value"), is_secret) or None,
                                        **({"warning": warning} if warning else {}),
                                    }
                                }),
                                metadata={
                                    "typed_into_label": label or None,
                                    "previous_value": _secret_display(previous_value, is_secret) or None,
                                    "overwrite_warning": warning,
                                },
                            )
                        except _AmbiguousTargetError as amb:
                            ambiguous_err = amb
                            continue
                        except Exception as pw_err:
                            last_err = pw_err
                            continue
                variant_note = f" across {len(selectors_to_try)} tag variant(s)" if len(selectors_to_try) > 1 else ""
                if ambiguous_err is not None:
                    return ActionResult(
                        success=False,
                        action_type=ActionType.DOM_TYPE,
                        description=f"Type into {selector}",
                        error=str(ambiguous_err),
                    )
                return ActionResult(
                    success=False,
                    action_type=ActionType.DOM_TYPE,
                    description=f"Type into {selector}",
                    error=f"Not found in main frame or {len(frames) - 1} iframe(s){variant_note} ({tried_count} attempts): {last_err}",
                )

            # Standard CSS selector path — find + focus via querySelectorAll,
            # then type with real keyboard events. The same evaluate also
            # enforces the one-visible-match guard and reads the field's
            # label and current value, so the result can name what it typed
            # into and what it replaced.
            js_script = """
            (selector) => {
                const isVisible = (el) => {
                    const style = window.getComputedStyle(el);
                    if (!style || style.display === "none" || style.visibility === "hidden" || Number(style.opacity) === 0) {
                        return false;
                    }
                    if (el.closest("[hidden], [aria-hidden='true']")) return false;
                    const rect = el.getBoundingClientRect();
                    return rect.width > 0 && rect.height > 0;
                };
                const labelFor = %s;

                let elements = [];
                try {
                    elements = Array.from(document.querySelectorAll(selector));
                } catch (error) {
                    return {ok: false, error: `Invalid selector: ${error.message || error}`};
                }

                const fields = elements.filter((el) => isVisible(el) && !(el.disabled || el.getAttribute("aria-disabled") === "true"));
                if (fields.length === 0) {
                    return {ok: false, error: `No visible enabled element found for selector. Matched ${elements.length} elements.`};
                }
                if (fields.length > 1) {
                    return {
                        ok: false,
                        ambiguous: true,
                        count: fields.length,
                        fields: fields.slice(0, 8).map((el) => ({label: labelFor(el), value: String(el.value == null ? "" : el.value)})),
                    };
                }

                const el = fields[0];
                const previous_value = String(el.value == null ? "" : el.value);
                el.scrollIntoView({block: "center", inline: "center", behavior: "auto"});
                el.focus();
                const rect = el.getBoundingClientRect();
                return {ok: true, x: rect.left + rect.width / 2, y: rect.top + rect.height / 2, label: labelFor(el), previous_value: previous_value, is_secret: (el.tagName === "INPUT" && el.type === "password")};
            }
            """ % _FIELD_LABEL_JS
            last_result = None
            ambiguous_result = None
            for frame in frames:
                try:
                    result = await frame.evaluate(js_script, selector)
                except Exception as e:
                    result = {"ok": False, "error": str(e)}
                if result and result.get("ok"):
                    await self.cursor_overlay.show_click(self.page, int(result.get("x", 0)), int(result.get("y", 0)))
                    await self.cursor_overlay.show_typing_start(self.page, int(result.get("x", 0)), int(result.get("y", 0)))
                    previous_value = str(result.get("previous_value") or "")
                    await self.page.keyboard.press("Control+A")
                    await self.page.keyboard.press("Backspace")
                    await self._human_type(text)
                    await self.cursor_overlay.show_typing_stop(self.page)
                    if press_enter:
                        await self.cursor_overlay.show_key(self.page, "Enter")
                        await self.page.keyboard.press("Enter")
                        try: await self.page.wait_for_load_state("domcontentloaded", timeout=2000)
                        except: pass
                    echo = await self._read_type_echo(frame)
                    frame_note = "" if frame is self.page.main_frame else f" [iframe: {frame.url[:60]}]"
                    label = str((echo or {}).get("label") or result.get("label") or "")
                    is_secret = bool(result.get("is_secret"))
                    description, warning = _type_echo_description(selector, label, text, previous_value, frame_note, is_secret=is_secret)
                    return ActionResult(
                        success=True,
                        action_type=ActionType.DOM_TYPE,
                        description=description,
                        coordinates=(int(result.get("x", 0)), int(result.get("y", 0))),
                        output=json.dumps({
                            "typed_into": {
                                "label": label or None,
                                "previous_value": _secret_display(previous_value, is_secret) or None,
                                "value": _secret_display((echo or {}).get("value"), is_secret) or None,
                                **({"warning": warning} if warning else {}),
                            }
                        }),
                        metadata={
                            "typed_into_label": label or None,
                            "previous_value": _secret_display(previous_value, bool(result.get("is_secret"))) or None,
                            "overwrite_warning": warning,
                        },
                    )
                if result and result.get("ambiguous"):
                    # Try the remaining frames first — a selector ambiguous in
                    # the main frame may match exactly one field inside an
                    # iframe. If no frame resolves it uniquely, fail with the
                    # field list instead of typing into the first match.
                    if ambiguous_result is None:
                        ambiguous_result = result
                    continue
                last_result = result
            if ambiguous_result is not None:
                return ActionResult(
                    success=False,
                    action_type=ActionType.DOM_TYPE,
                    description=f"Type into {selector}",
                    error=_ambiguous_type_error(selector, int(ambiguous_result.get("count") or 0), ambiguous_result.get("fields") or []),
                )
            return ActionResult(
                success=False,
                action_type=ActionType.DOM_TYPE,
                description=f"Type into {selector}",
                error=(last_result or {}).get("error", "No visible enabled element found in main frame or any iframe."),
            )
        except Exception as e: return ActionResult(success=False, action_type=ActionType.DOM_TYPE, description=f"Type into {selector}", error=str(e))

    async def _extract_js_results(self, page, query: str, schema: Optional[Dict], max_results: int):
        """Run the DOM extraction script on the given page. Used by BatchExtract
        on background tabs (mirrors the script embedded in _dom_extract)."""
        js_script = """
        (args) => {
            const query = args.query;
            const schema = args.schema;
            const max_results = args.max_results;

            const cleanText = (value) => {
                if (value === null || value === undefined) return "";
                return String(value).replace(/\\s+/g, " ").trim();
            };

            const extractText = (node) => {
                if (!node) return null;
                if (node.nodeType === Node.TEXT_NODE) return cleanText(node.textContent);
                if (node.nodeType !== Node.ELEMENT_NODE) return cleanText(node.textContent);

                const el = node;
                const candidates = [
                    el.innerText,
                    el.textContent,
                    el.value,
                    el.getAttribute && el.getAttribute("content"),
                    el.getAttribute && el.getAttribute("aria-label"),
                    el.getAttribute && el.getAttribute("title"),
                    el.getAttribute && el.getAttribute("alt"),
                    el.getAttribute && el.getAttribute("placeholder"),
                    el.getAttribute && el.getAttribute("href"),
                    el.getAttribute && el.getAttribute("src"),
                ];

                for (const candidate of candidates) {
                    const text = cleanText(candidate);
                    if (text) return text;
                }
                return null;
            };

            const elements = Array.from(document.querySelectorAll(query)).slice(0, max_results);

            const rows = elements.map(el => {
                if (schema) {
                    const item = {};
                    for (const key in schema) {
                        const selector = schema[key];
                        let child = null;
                        try {
                            child = el.querySelector(selector);
                        } catch (_) {
                            child = null;
                        }
                        item[key] = extractText(child);
                    }
                    return item;
                } else {
                    return extractText(el);
                }
            });

            return rows.filter(row => {
                if (row === null || row === undefined) return false;
                if (typeof row === "string") return row.length > 0;
                if (typeof row === "object") return Object.values(row).some(value => cleanText(value).length > 0);
                return true;
            });
        }
        """
        return await page.evaluate(js_script, {
            "query": query,
            "schema": schema,
            "max_results": max_results,
        })

    async def _batch_extract(self, parameters: Dict[str, Any]) -> ActionResult:
        """Parallel read-only fan-out: open each URL in a background tab and/or
        run each query on the active page, extracting in one concurrent step.
        Deterministic — no LLM call per page; the active page is untouched."""
        import json as _json
        self.dom_calls += 1
        urls = [u.strip() for u in (parameters.get("urls") or [])
                if isinstance(u, str) and u.strip()]
        queries = [q.strip() for q in (parameters.get("queries") or
                    ([parameters.get("query")] if isinstance(parameters.get("query"), str) and parameters.get("query").strip() else []))
                if isinstance(q, str) and q.strip()]
        schema = parameters.get("schema") if isinstance(parameters.get("schema"), dict) else None
        max_urls = min(8, max(1, int(os.getenv("BATCH_EXTRACT_MAX_URLS", "4"))))
        urls = urls[:max_urls]
        max_results = max(1, min(50, int(parameters.get("max_results", 8))))
        goto_timeout = max(5000, int(os.getenv("BATCH_EXTRACT_GOTO_TIMEOUT_MS", "15000")))
        close_tabs = parameters.get("close_tabs", True)

        if urls and not queries:
            queries = ["main"]

        if not urls and not queries:
            return ActionResult(
                success=False, action_type=ActionType.BATCH_EXTRACT, description="BatchExtract",
                error="Provide 'urls' (list) and/or 'query'/'queries' (list) to batch-extract.",
            )

        async def one_url(url: str) -> Dict[str, Any]:
            page = None
            try:
                page = await self.page.context.new_page()
                await page.goto(url, wait_until="domcontentloaded", timeout=goto_timeout)
                await page.wait_for_timeout(300)
                extracts = {q: await self._extract_js_results(page, q, schema, max_results) for q in queries}
                return {"label": url, "ok": True, "extracts": extracts}
            except Exception as e:
                return {"label": url, "ok": False, "error": str(e)}
            finally:
                if page is not None and close_tabs:
                    try:
                        await page.close()
                    except Exception:
                        pass

        async def one_query(query: str) -> Dict[str, Any]:
            try:
                return {"label": f"this page :: {query}", "ok": True,
                        "extracts": {query: await self._extract_js_results(self.page, query, schema, max_results)}}
            except Exception as e:
                return {"label": f"this page :: {query}", "ok": False, "error": str(e)}

        tasks = [one_url(u) for u in urls] + ([] if urls else [one_query(q) for q in queries])
        outcomes = await asyncio.gather(*tasks)

        sections: List[str] = []
        ok_count = 0
        for idx, item in enumerate(outcomes, 1):
            label = item.get("label")
            if item.get("ok"):
                ok_count += 1
                body = _json.dumps(item["extracts"], ensure_ascii=False, indent=2)
                sections.append(f"[{idx}] {label}\n{body}")
            else:
                sections.append(f"[{idx}] {label}\nERROR: {item.get('error')}")

        all_text = "\n\n".join(sections)
        if len(all_text) > 100000:
            all_text = all_text[:100000] + "... (truncated)"
        success = ok_count > 0
        return ActionResult(
            success=success,
            action_type=ActionType.BATCH_EXTRACT,
            description=f"BatchExtract: {ok_count}/{len(outcomes)} target(s) extracted",
            output=all_text if success else None,
            error=None if success else "All batch extracts failed",
        )

    async def _credential_fill(self, parameters: Dict[str, Any]) -> ActionResult:
        """Deterministically fill the login form on the current page using the
        per-run vault credentials provisioned by the Cosmic orchestrator.

        Values never appear in this ActionResult, in screenshots, or in any
        LLM context — the model only learns whether the fill succeeded.
        """
        if self.page is None:
            return ActionResult(success=False, action_type=ActionType.CREDENTIAL_FILL, description="CredentialFill", error="Browser page is not available.")
        if self.credential_store is None or len(self.credential_store) == 0:
            return ActionResult(
                success=False, action_type=ActionType.CREDENTIAL_FILL, description="CredentialFill",
                error="No credentials were provisioned for this run. If this site requires login to proceed, call RequestCredentials to ask the orchestrator.",
            )
        current_url = self.page.url
        entry = self.credential_store.get(current_url)
        if not entry:
            available = ", ".join(self.credential_store.available_domains())
            return ActionResult(
                success=False, action_type=ActionType.CREDENTIAL_FILL, description="CredentialFill",
                error=f"No credentials provisioned for this site ({current_url}). Available for: {available}. Call RequestCredentials if login is required here.",
            )

        do_submit = parameters.get("submit", True)
        if isinstance(do_submit, str):
            do_submit = do_submit.strip().lower() not in {"false", "0", "no", "off"}
        js_script = """
        (args) => {
            const visible = (el) => {
                if (!el) return false;
                const style = window.getComputedStyle(el);
                if (style.visibility === 'hidden' || style.display === 'none') return false;
                const rect = el.getBoundingClientRect();
                return rect.width > 0 && rect.height > 0;
            };
            const nativeFill = (el, value) => {
                if (!el) return false;
                const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
                setter.call(el, value);
                el.dispatchEvent(new Event('input', { bubbles: true }));
                el.dispatchEvent(new Event('change', { bubbles: true }));
                return true;
            };
            const passwords = Array.from(document.querySelectorAll('input[type="password"]')).filter(visible);
            if (!passwords.length) return { found: false, reason: 'no visible password field on this page' };
            const pw = passwords[0];
            const form = pw.closest('form');
            const scope = form || document;
            const hint = (el) => ((el.autocomplete || '') + ' ' + (el.name || '') + ' ' + (el.id || '') + ' ' + (el.placeholder || '') + ' ' + (el.getAttribute('aria-label') || '')).toLowerCase();
            const candidates = Array.from(scope.querySelectorAll('input[type="text"], input[type="email"], input[type="tel"], input:not([type])'))
                .filter(el => el !== pw && visible(el) && !el.disabled && !el.readOnly);
            const emailish = candidates.filter(el => /email|user|login|account/.test(hint(el)));
            const user = emailish.length ? emailish[0] : (candidates.length ? candidates[candidates.length - 1] : null);
            const totpField = Array.from(scope.querySelectorAll('input'))
                .filter(el => el !== pw && el !== user && visible(el) && /totp|otp|2fa|code/.test(hint(el)))[0] || null;
            const filled = [];
            if (user && args.username) { nativeFill(user, args.username); filled.push('username'); }
            if (pw && args.password) { nativeFill(pw, args.password); filled.push('password'); }
            if (totpField && args.totp) { nativeFill(totpField, args.totp); filled.push('totp'); }
            let submitted = false;
            if (args.submit && pw && args.password) {
                const submitBtn = form
                    ? (form.querySelector('button[type="submit"], input[type="submit"]') || form.querySelector('button'))
                    : document.querySelector('button[type="submit"]');
                if (submitBtn) { submitBtn.click(); submitted = true; }
                else if (form && form.requestSubmit) { form.requestSubmit(); submitted = true; }
                else { pw.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', code: 'Enter', bubbles: true })); submitted = true; }
            }
            return { found: true, filled, submitted };
        }
        """
        try:
            fill_state = await self.page.evaluate(js_script, {
                "username": entry.get("username", ""),
                "password": entry.get("password", ""),
                "totp": entry.get("totp_seed", ""),
                "submit": bool(do_submit),
            })
        except Exception as exc:
            return ActionResult(success=False, action_type=ActionType.CREDENTIAL_FILL, description="CredentialFill", error=f"Credential fill failed: {exc}")
        if not fill_state or not fill_state.get("found"):
            reason = (fill_state or {}).get("reason", "no login form found")
            return ActionResult(
                success=False, action_type=ActionType.CREDENTIAL_FILL, description="CredentialFill",
                output=f'{{"status": "no_login_form", "reason": "{reason}"}}',
                error=f"CredentialFill could not find a login form on this page: {reason}. If this page is not a login page, continue the task normally.",
            )
        filled = fill_state.get("filled", [])
        description = f"CredentialFill: filled {', '.join(filled) if filled else 'nothing'}" + (" and submitted" if fill_state.get("submitted") else "")
        return ActionResult(
            success=bool(filled),
            action_type=ActionType.CREDENTIAL_FILL,
            description=description,
            output='{"status": "filled", "fields": [' + ", ".join(f'"{f}"' for f in filled) + '], "submitted": ' + ("true" if fill_state.get("submitted") else "false") + "}",
            error=None if filled else "No matching fields were filled; check the page state.",
        )

    async def _request_credentials(self, parameters: Dict[str, Any]) -> ActionResult:
        """End the run asking the orchestrator for credentials for a site.

        The structured output drives the orchestrator's credential-request
        card; no secrets are involved — this is a request, not a transfer.
        """
        site = str(parameters.get("site") or "").strip()
        if not site and self.page is not None:
            site = self.page.url
        if not site:
            return ActionResult(success=False, action_type=ActionType.REQUEST_CREDENTIALS, description="RequestCredentials", error="No site could be determined for the credential request.")
        reason = str(parameters.get("reason") or "Login is required to reach the goal.").strip()
        return ActionResult(
            success=True,
            action_type=ActionType.REQUEST_CREDENTIALS,
            description=f"RequestCredentials: asked the orchestrator for credentials",
            output='{"status": "credentials_needed", "site": ' + json.dumps(site) + ', "reason": ' + json.dumps(reason) + "}",
        )

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

                const cleanText = (value) => {
                    if (value === null || value === undefined) return "";
                    return String(value).replace(/\\s+/g, " ").trim();
                };

                const extractText = (node) => {
                    if (!node) return null;
                    if (node.nodeType === Node.TEXT_NODE) return cleanText(node.textContent);
                    if (node.nodeType !== Node.ELEMENT_NODE) return cleanText(node.textContent);

                    const el = node;
                    const candidates = [
                        el.innerText,
                        el.textContent,
                        el.value,
                        el.getAttribute && el.getAttribute("content"),
                        el.getAttribute && el.getAttribute("aria-label"),
                        el.getAttribute && el.getAttribute("title"),
                        el.getAttribute && el.getAttribute("alt"),
                        el.getAttribute && el.getAttribute("placeholder"),
                        el.getAttribute && el.getAttribute("href"),
                        el.getAttribute && el.getAttribute("src"),
                    ];

                    for (const candidate of candidates) {
                        const text = cleanText(candidate);
                        if (text) return text;
                    }
                    return null;
                };
                
                const elements = Array.from(document.querySelectorAll(query)).slice(0, max_results);
                
                const rows = elements.map(el => {
                    if (schema) {
                        const item = {};
                        for (const key in schema) {
                            const selector = schema[key];
                            let child = null;
                            try {
                                child = el.querySelector(selector);
                            } catch (_) {
                                child = null;
                            }
                            item[key] = extractText(child);
                        }
                        return item;
                    } else {
                        return extractText(el);
                    }
                });

                return rows.filter(row => {
                    if (row === null || row === undefined) return false;
                    if (typeof row === "string") return row.length > 0;
                    if (typeof row === "object") return Object.values(row).some(value => cleanText(value).length > 0);
                    return true;
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
            output_str = json.dumps(data, ensure_ascii=False, indent=2)
            if len(output_str) > 100000:
                output_str = output_str[:100000] + "... (truncated)"
            
            return ActionResult(
                success=True, 
                action_type=ActionType.DOM_EXTRACT, 
                description=f"Extracted {len(data)} items",
                output=output_str
            )
        except Exception as e: return ActionResult(success=False, action_type=ActionType.DOM_EXTRACT, description=f"Extract {query}", error=str(e))

    @staticmethod
    def _is_extreme_edge_coordinate(x: int, y: int, width: int, height: int) -> bool:
        """Detect likely bad grounding coordinates at the extreme screenshot border.

        This is deliberately site-agnostic. It does not know about YouTube or any
        page layout; it only treats coordinates hugging the outer screenshot edge
        as low-confidence because grounding models sometimes fall back to [0, 0].
        """
        ratio = float(os.getenv("MIMO_EDGE_GUARD_RATIO", "0.015"))
        margin_x = max(3, int(width * ratio))
        margin_y = max(3, int(height * ratio))
        return x <= margin_x or y <= margin_y or x >= width - margin_x or y >= height - margin_y

    async def _navigate(self, url: str) -> ActionResult:
        try:
            await self.page.goto(url, wait_until="domcontentloaded", timeout=15000)
            return ActionResult(success=True, action_type=ActionType.NAVIGATE, description=f"Navigated to {url}")
        except Exception as e: return ActionResult(success=False, action_type=ActionType.NAVIGATE, description=f"Navigate {url}", error=str(e))

    async def _recover_from_nav_timeout(self) -> bool:
        """A go_back/go_forward/reload call can time out waiting for
        "domcontentloaded" while the underlying navigation already actually
        happened (observed in production: the Playwright timeout's own call
        log showed "navigated to <url>" even though the call raised). Give
        the page one more short window to settle before accepting the
        timeout as a real failure, instead of reporting false negatives for
        navigations that did succeed, just slowly."""
        try:
            await self.page.wait_for_load_state("domcontentloaded", timeout=4000)
            return True
        except Exception:
            return False

    async def _go_back(self) -> ActionResult:
        try:
            response = await self.page.go_back(wait_until="domcontentloaded", timeout=15000)
            if response is None:
                return ActionResult(success=False, action_type=ActionType.GO_BACK, description="Go back", error="No previous page in history")
            return ActionResult(success=True, action_type=ActionType.GO_BACK, description=f"Went back to {self.page.url}")
        except Exception as e:
            if await self._recover_from_nav_timeout():
                return ActionResult(success=True, action_type=ActionType.GO_BACK, description=f"Went back to {self.page.url} (slow load, recovered)")
            return ActionResult(success=False, action_type=ActionType.GO_BACK, description="Go back", error=str(e))

    async def _go_forward(self) -> ActionResult:
        try:
            response = await self.page.go_forward(wait_until="domcontentloaded", timeout=15000)
            if response is None:
                return ActionResult(success=False, action_type=ActionType.GO_FORWARD, description="Go forward", error="No forward page in history")
            return ActionResult(success=True, action_type=ActionType.GO_FORWARD, description=f"Went forward to {self.page.url}")
        except Exception as e:
            if await self._recover_from_nav_timeout():
                return ActionResult(success=True, action_type=ActionType.GO_FORWARD, description=f"Went forward to {self.page.url} (slow load, recovered)")
            return ActionResult(success=False, action_type=ActionType.GO_FORWARD, description="Go forward", error=str(e))

    async def _reload(self) -> ActionResult:
        try:
            await self.page.reload(wait_until="domcontentloaded", timeout=15000)
            return ActionResult(success=True, action_type=ActionType.RELOAD, description=f"Reloaded {self.page.url}")
        except Exception as e:
            if await self._recover_from_nav_timeout():
                return ActionResult(success=True, action_type=ActionType.RELOAD, description=f"Reloaded {self.page.url} (slow load, recovered)")
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
                screenshot_bytes = await self._safe_page_screenshot(type="jpeg", quality=40)
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
        normalized_key = {
            "return": "Enter",
            "newline": "Enter",
            "esc": "Escape",
            "del": "Delete",
            "backspace": "Backspace",
        }.get(str(key).strip().lower(), key)
        await self.cursor_overlay.show_key(self.page, normalized_key)
        await self.page.keyboard.press(normalized_key)
        return ActionResult(success=True, action_type=ActionType.PRESS_KEY, description=f"Pressed {normalized_key}")

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

            await self._safe_page_screenshot(path=screenshot_path, type="jpeg", quality=self.config.screenshot_quality)
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
        self.last_mimo_grounding = None

        # --- Encode ---
        t0 = time.time()
        with Image.open(screenshot_path) as img:
            w, h = img.size
            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=85)
            img_b64 = base64.b64encode(buffered.getvalue()).decode()
        encode_ms = (time.time() - t0) * 1000

        def make_payload(query_text: str, *, max_tokens: Optional[int] = None) -> Dict[str, Any]:
            payload = {
                "model": "XiaomiMiMo/MiMo-VL-7B-RL",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a GUI grounding assistant. Given a screenshot and an element description, output the pixel coordinates (x, y) of the center of that element.\n\nOutput format: Return ONLY the coordinates as [x, y] where x and y are pixel values.\nDo not include thinking, prose, explanation, markdown, XML tags, or any other text."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                            {"type": "text", "text": query_text}
                        ]
                    }
                ],
                "temperature": self.mimo_temperature,
                "max_tokens": max_tokens or self.mimo_max_tokens,
            }
            return payload

        # CORRECTED QUERY FORMAT matching find_coordinates_mimo.py
        query = f"Image size: {w}x{h} pixels\n\nFind the element: {instruction}\n\nOutput the center coordinates as [x, y] in pixels."
        
        try:
            headers = {}
            if getattr(self, 'mimo_api_key', None):
                headers["Authorization"] = f"Bearer {self.mimo_api_key}"

            # --- HTTP inference ---
            t1 = time.time()
            try:
                response = await self.http_client.post(self.mimo_chat_completions_url, json=make_payload(query), headers=headers)
            except (httpx.RemoteProtocolError, httpx.ReadError, httpx.ConnectError, httpx.PoolTimeout) as e:
                print(f"⚠️ MiMo transport error, retrying once: {e}")
                await asyncio.sleep(0.2)
                response = await self.http_client.post(self.mimo_chat_completions_url, json=make_payload(query), headers=headers)
            response.raise_for_status()
            infer_ms = (time.time() - t1) * 1000

            content = response.json()["choices"][0]["message"]["content"]
            _thinking_text, _ = extract_thinking(content)
            _think_chars = len(_thinking_text) if _thinking_text else 0
            
            # --- Parse ---
            t2 = time.time()
            try:
                coords = parse_coordinates(content, (w, h))
                parse_ms = (time.time() - t2) * 1000
                x, y = coords
                edge_guard = None
                if self._is_extreme_edge_coordinate(x, y, w, h):
                    retry_query = (
                        f"Image size: {w}x{h} pixels\n\n"
                        f"Find the element: {instruction}\n\n"
                        f"Your previous candidate coordinate [{x}, {y}] is on the extreme screenshot edge. "
                        "Return an edge or corner coordinate only if the requested element is visibly centered there. "
                        "Otherwise return the true center of the requested element. "
                        "Output only [x, y] in pixels."
                    )
                    retry_start = time.time()
                    retry_response = await self.http_client.post(
                        self.mimo_chat_completions_url,
                        json=make_payload(retry_query),
                        headers=headers,
                    )
                    retry_response.raise_for_status()
                    retry_infer_ms = (time.time() - retry_start) * 1000
                    retry_content = retry_response.json()["choices"][0]["message"]["content"]
                    try:
                        retry_coords = parse_coordinates(retry_content, (w, h))
                        retry_x, retry_y = retry_coords
                        edge_guard = {
                            "triggered": True,
                            "first_coordinates": {"x": int(x), "y": int(y)},
                            "retry_raw_model_output": retry_content,
                            "retry_infer_ms": retry_infer_ms,
                            "retry_coordinates": {"x": int(retry_x), "y": int(retry_y)},
                        }
                        if self._is_extreme_edge_coordinate(retry_x, retry_y, w, h):
                            edge_guard["rejected"] = True
                            self.last_mimo_grounding = {
                                "instruction": instruction,
                                "image_size": {"width": w, "height": h},
                                "pixel_coordinates": {"x": int(retry_x), "y": int(retry_y)},
                                "normalized_coordinates": {
                                    "x": round(float(retry_x) / max(1, w), 6),
                                    "y": round(float(retry_y) / max(1, h), 6),
                                },
                                "raw_model_output": content,
                                "parser": "browser_controller.parse_coordinates",
                                "edge_guard": edge_guard,
                            }
                            print(f"⚠️ MiMo edge-coordinate guard rejected suspicious coordinates: first=({x}, {y}) retry=({retry_x}, {retry_y})")
                            print(f"   ⏱️  MiMo latency: encode={encode_ms:.0f}ms | infer={infer_ms + retry_infer_ms:.0f}ms | parse={parse_ms:.0f}ms | total={encode_ms+infer_ms+retry_infer_ms+parse_ms:.0f}ms")
                            return None
                        coords = retry_coords
                        x, y = coords
                        edge_guard["rejected"] = False
                    except ValueError:
                        edge_guard = {
                            "triggered": True,
                            "first_coordinates": {"x": int(x), "y": int(y)},
                            "retry_raw_model_output": retry_content,
                            "retry_infer_ms": retry_infer_ms,
                            "rejected": True,
                            "retry_parse_error": True,
                        }
                        self.last_mimo_grounding = {
                            "instruction": instruction,
                            "image_size": {"width": w, "height": h},
                            "pixel_coordinates": {"x": int(x), "y": int(y)},
                            "normalized_coordinates": {
                                "x": round(float(x) / max(1, w), 6),
                                "y": round(float(y) / max(1, h), 6),
                            },
                            "raw_model_output": content,
                            "parser": "browser_controller.parse_coordinates",
                            "edge_guard": edge_guard,
                        }
                        print(f"⚠️ MiMo edge-coordinate guard rejected suspicious coordinate after retry parse failure: ({x}, {y})")
                        return None
                self.last_mimo_grounding = {
                    "instruction": instruction,
                    "image_size": {"width": w, "height": h},
                    "pixel_coordinates": {"x": int(x), "y": int(y)},
                    "normalized_coordinates": {
                        "x": round(float(x) / max(1, w), 6),
                        "y": round(float(y) / max(1, h), 6),
                    },
                    "raw_model_output": content,
                    "parser": "browser_controller.parse_coordinates",
                    "edge_guard": edge_guard,
                }
                displayed_infer_ms = infer_ms + float((edge_guard or {}).get("retry_infer_ms") or 0.0)
                think_note = f" | think_chars={_think_chars}" if _think_chars else ""
                print(f"   ⏱️  MiMo latency: encode={encode_ms:.0f}ms | infer={displayed_infer_ms:.0f}ms | parse={parse_ms:.0f}ms | total={encode_ms+displayed_infer_ms+parse_ms:.0f}ms{think_note}")
                return coords
            except ValueError:
                parse_ms = (time.time() - t2) * 1000

                # Fast-fail: if MiMo's thinking output says the element isn't
                # on the page, skip the 22s retry — it won't change the answer.
                _NOT_FOUND_PHRASES = (
                    "not present", "not visible", "not in the screenshot",
                    "not in this screenshot", "not found", "cannot find",
                    "can't find", "doesn't exist", "does not exist",
                    "no such element", "no visible", "isn't present",
                    "isn't visible", "not displayed", "not shown",
                    "element is not", "there is no", "i don't see",
                    "i do not see", "not currently visible",
                )
                _lower_content = content.lower()
                _element_absent = any(p in _lower_content for p in _NOT_FOUND_PHRASES)
                # Also fast-fail if the model burned most of its token budget
                # reasoning without ever reaching a coordinate. Scales with
                # mimo_max_tokens (~3.2 chars/token, 85% threshold) instead of a
                # fixed char count — a hardcoded threshold silently stops firing
                # if max_tokens is ever lowered, since output can't physically
                # reach the old fixed length anymore (this exact bug shipped once).
                _max_possible_chars = self.mimo_max_tokens * 3.2
                _think_heavy = len(content) > (_max_possible_chars * 0.85) and "[" not in content

                # Fast-fail: a non-convergent self-doubt loop ("wait, no...
                # wait, maybe... wait, perhaps...") is a DIFFERENT failure mode
                # from running out of budget — the model keeps restarting its
                # coordinate estimate instead of committing. More tokens don't
                # fix this (confirmed: the 512-token STRICT retry shows the
                # exact same looping pattern in production), so don't burn
                # 15-20s waiting it out or retrying — bail immediately.
                _RESTART_PHRASES = ("wait, no", "wait, maybe", "wait, perhaps", "let's think again", "let me think again")
                _restart_count = sum(_lower_content.count(p) for p in _RESTART_PHRASES)
                _indecisive_loop = _restart_count >= 3

                if _element_absent or _think_heavy or _indecisive_loop:
                    if _element_absent:
                        reason = "element-absent reasoning"
                    elif _indecisive_loop:
                        reason = f"non-convergent self-doubt loop ({_restart_count} restarts)"
                    else:
                        reason = "truncated thinking (element not found)"
                    print(f"⚠️ MiMo parse failed ({reason}) — skipping retry.")
                    print(f"   ⏱️  MiMo latency: encode={encode_ms:.0f}ms | infer={infer_ms:.0f}ms | parse={parse_ms:.0f}ms")
                    return None

                retry_query = (
                    f"Image size: {w}x{h} pixels\n\n"
                    f"Find the element: {instruction}\n\n"
                    "STRICT OUTPUT MODE. Do not think. Do not explain. Do not use XML tags. "
                    "Return exactly one coordinate pair only, formatted like [123, 456]."
                )
                retry_start = time.time()
                try:
                    retry_response = await self.http_client.post(
                        self.mimo_chat_completions_url,
                        # Give the retry real headroom — MiMo often keeps thinking
                        # even under "STRICT OUTPUT MODE", so a tiny budget (e.g.
                        # 96 tokens) just guarantees a second truncation instead of
                        # a second chance.
                        json=make_payload(retry_query, max_tokens=max(256, min(self.mimo_max_tokens, 512))),
                        headers=headers,
                    )
                    retry_response.raise_for_status()
                    retry_infer_ms = (time.time() - retry_start) * 1000
                    retry_content = retry_response.json()["choices"][0]["message"]["content"]
                    retry_coords = parse_coordinates(retry_content, (w, h))
                    retry_x, retry_y = retry_coords
                    retry_parse_ms = (time.time() - retry_start) * 1000 - retry_infer_ms
                    self.last_mimo_grounding = {
                        "instruction": instruction,
                        "image_size": {"width": w, "height": h},
                        "pixel_coordinates": {"x": int(retry_x), "y": int(retry_y)},
                        "normalized_coordinates": {
                            "x": round(float(retry_x) / max(1, w), 6),
                            "y": round(float(retry_y) / max(1, h), 6),
                        },
                        "raw_model_output": content,
                        "parser": "browser_controller.parse_coordinates",
                        "parse_retry": {
                            "triggered": True,
                            "retry_raw_model_output": retry_content,
                            "retry_infer_ms": retry_infer_ms,
                        },
                    }
                    print(f"⚠️ MiMo parse failed once, strict retry succeeded. First raw output: '{content[:240]}'")
                    print(f"   ⏱️  MiMo latency: encode={encode_ms:.0f}ms | infer={infer_ms + retry_infer_ms:.0f}ms | parse={parse_ms + retry_parse_ms:.0f}ms | total={encode_ms+infer_ms+retry_infer_ms+parse_ms+retry_parse_ms:.0f}ms")
                    return retry_coords
                except Exception as retry_error:
                    print(f"⚠️ MiMo parse failed. Raw Output: '{content[:200]}'")
                    print(f"⚠️ MiMo strict retry failed: {retry_error}")
                    print(f"   ⏱️  MiMo latency: encode={encode_ms:.0f}ms | infer={infer_ms:.0f}ms | parse={parse_ms:.0f}ms")
                    return None
            
        except Exception as e:
            print(f"MiMo call failed: {e}")
            return None


    async def _evaluate_verification_hint(
        self,
        verification_hint: Optional[str],
        after_state: BrowserState,
    ) -> Optional[bool]:
        """Evaluate a structured verification_hint against the post-action state.

        Returns True/False for parseable, evaluable hints; None when the hint
        is absent, free-form (not in the grammar), or cannot be evaluated —
        in which case the caller falls back to structural evidence, exactly
        like the legacy behavior. Never raises.
        """
        if not verification_hint:
            return None
        match = _SEMANTIC_HINT_RE.match(str(verification_hint).strip())
        if not match:
            return None
        key = match.group(1).lower()
        value = match.group(2).strip().strip("'\"")
        if not value:
            return None
        after_url = (after_state.url or "").lower()
        if key == "url_contains":
            return value.lower() in after_url
        if key == "url_equals":
            return (after_state.url or "").rstrip("/") == value.rstrip("/")
        if key == "title_contains":
            return value.lower() in (after_state.title or "").lower()
        if key in ("element_exists", "element_visible"):
            return await self._hint_selector_visible(value)
        return None

    async def _hint_selector_visible(self, selector: str) -> Optional[bool]:
        """Best-effort main-frame check that a CSS selector matches a visible element.

        Returns True/False, or None when the selector is invalid CSS, the
        selector check cannot run, or no page is available. The selector is
        passed to evaluate() as an argument, never interpolated into JS source.
        """
        if not self.page:
            return None
        js = """
        (sel) => {
          let el = null;
          try { el = document.querySelector(sel); } catch (e) { return 'invalid'; }
          if (!el) return false;
          try {
            const r = el.getBoundingClientRect();
            if (r.width <= 0 || r.height <= 0) return false;
            const st = getComputedStyle(el);
            if (st.visibility === 'hidden' || st.display === 'none') return false;
            return true;
          } catch (e) { return false; }
        }
        """
        js_result = None
        for attempt in range(2):
            try:
                js_result = await self.page.evaluate(js, selector)
                break
            except Exception as e:
                if attempt == 0 and ("context" in str(e).lower() or "execution" in str(e).lower()):
                    try:
                        await self.page.wait_for_load_state("domcontentloaded", timeout=8000)
                    except Exception:
                        pass
                    continue
                return None
        if js_result is None:
            return None
        if js_result == "invalid":
            return None
        return bool(js_result)

    async def verify_action(
        self,
        before_state: BrowserState,
        after_state: BrowserState,
        verification_hint: Optional[str],
        action_description: Optional[str] = None,
        action_type: Optional[ActionType] = None,
    ) -> Tuple[VerificationStatus, float]:
        """Layered, deterministic verification of whether an action changed the
        page into the state it intended.

        Evidence layers (cheapest first):
          1. Screenshot hash (pixels) — necessary, but fooled by ad refreshes,
             spinners, and animations, which change pixels without the page
             actually moving.
          2. Semantic state diff: URL, title, open tabs, auto-handled dialogs,
             ready state, scroll position.
          3. Structural DOM signature diff (focus, body shape, visible input
             count, value fingerprints — never raw text) — catches changes
             pixels hide (e.g. typed text in masked/password fields) and
             filters changes pixels invent (ad refreshes).
          4. Structured verification_hint from the orchestrator, e.g.
             url_contains('/checkout'), title_contains('Inbox'),
             element_exists('#submit'), element_visible('.results').

        Status semantics:
        - SUCCESS: structural/semantic evidence confirms the page moved.
        - INCOMPLETE: the page changed only in pixels while the DOM signature
          shows no structural change (likely ad/spinner/animation), the page
          did not scroll when a scroll was requested, or a semantic hint
          evaluated false. Deliberately NOT reported as SUCCESS so COSMIC
          memory does not index unverifiable steps as gold-path actions.
        - NO_CHANGE: nothing observable happened at all.
        - WRONG_STATE: decisive negative (e.g. wrong SSO provider landed).

        Backwards compatible: BrowserStates without a DOM signature (old
        checkpoints/replays, or pages where the signature could not be
        evaluated) fall back to the legacy hash-only decision, and unparseable
        hints are ignored.
        """
        if before_state is None or after_state is None:
            return VerificationStatus.ERROR, 0.0

        url_changed = (before_state.url or "") != (after_state.url or "")
        title_changed = (before_state.title or "") != (after_state.title or "")
        tabs_before = [(t.page_id, t.url) for t in (before_state.tabs or [])]
        tabs_after = [(t.page_id, t.url) for t in (after_state.tabs or [])]
        tabs_changed = tabs_before != tabs_after
        dialogs_appeared = bool(after_state.dialogs) and not before_state.dialogs
        scroll_delta = abs((after_state.scroll_y or 0) - (before_state.scroll_y or 0))
        ready_changed = (before_state.ready_state or "") != (after_state.ready_state or "")
        pixels_changed = before_state.screenshot_hash != after_state.screenshot_hash
        dom_signatures_usable = bool(before_state.dom_signature) and bool(after_state.dom_signature)
        dom_changed = dom_signatures_usable and before_state.dom_signature != after_state.dom_signature
        structural_evidence = (
            url_changed or title_changed or tabs_changed or dialogs_appeared
            or ready_changed or dom_changed
        )

        # 1. Nothing observable happened: pixels, DOM, URL, and scroll identical.
        if not pixels_changed and not structural_evidence and scroll_delta < 5:
            return VerificationStatus.NO_CHANGE, 0.0

        # 2. Wrong-SSO-provider detection (unchanged, decisive). A click
        # described as targeting one provider that lands on a DIFFERENT
        # provider's auth domain picked the wrong element.
        if action_description:
            desc_lower = action_description.lower()
            after_url_lower = (after_state.url or "").lower()
            mentioned = next((p for p in _SSO_PROVIDER_DOMAINS if p in desc_lower), None)
            if mentioned:
                landed = next(
                    (p for p, domains in _SSO_PROVIDER_DOMAINS.items() if any(d in after_url_lower for d in domains)),
                    None,
                )
                if landed and landed != mentioned:
                    return VerificationStatus.WRONG_STATE, 1.0

        # 3. Scroll actions: scroll position IS the intended effect. Pixels
        # moving while the page stays put is almost always an ad refresh.
        if action_type in _SCROLL_ACTION_TYPES:
            if scroll_delta >= 5:
                return VerificationStatus.SUCCESS, 1.0
            return VerificationStatus.INCOMPLETE, 0.5

        # 4. Structured semantic hint (decisive when parseable and evaluated).
        hint_result = await self._evaluate_verification_hint(verification_hint, after_state)
        if hint_result is False:
            return VerificationStatus.INCOMPLETE, 0.5

        # 5. Structural evidence (URL/title/tabs/dialogs/ready-state/DOM
        # signature) confirms the page actually moved.
        if structural_evidence:
            return VerificationStatus.SUCCESS, 1.0
        if hint_result is True:
            return VerificationStatus.SUCCESS, 1.0

        # 6. Pixels changed but nothing structural did — treat as cosmetic
        # (ad refresh, spinner, animation) rather than success. When DOM
        # signatures are unavailable (legacy states), preserve the old
        # behavior and trust the pixel change.
        if pixels_changed:
            if dom_signatures_usable:
                return VerificationStatus.INCOMPLETE, 0.5
            return VerificationStatus.SUCCESS, 1.0

        return VerificationStatus.NO_CHANGE, 0.0

    async def close(self):
        # In CDP mode, ask the agent Chrome to exit via the browser-level
        # Browser.close command FIRST: Chrome then flushes cookies/session
        # state to disk on its own shutdown path, which a bare process
        # terminate() would skip. The agent user-data-dir is persistent, so
        # logins made during this run are simply there for the next run —
        # nothing is ever written back into the user's real profile, and
        # the user's own Chrome is never touched.
        if self.browser and self._owns_cdp_browser:
            try:
                session = await self.browser.new_browser_cdp_session()
                await session.send("Browser.close")
            except Exception:
                pass
        else:
            try:
                if self.context:
                    await self.context.close()
            except Exception:
                pass
        try:
            if self.browser:
                await self.browser.close()
        except Exception:
            pass
        if self.playwright:
            await self.playwright.stop()
        await self.http_client.aclose()
        if self._chrome_subprocess:
            try:
                # Give Chrome a moment to exit cleanly after Browser.close;
                # escalate to terminate/kill only if it doesn't.
                self._chrome_subprocess.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._terminate_chrome_subprocess()
            except Exception:
                pass
        if self._chrome_stderr_handle:
            try:
                self._chrome_stderr_handle.close()
            except Exception:
                pass

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_actions": self.total_actions,
            "mimo_calls": self.mimo_calls,
            "dom_calls": self.dom_calls,
            "mimo_max_tokens": self.mimo_max_tokens,
            "mimo_temperature": self.mimo_temperature,
            "mimo_timeout": self.mimo_timeout,
            "mimo_http2": self.mimo_http2,
            "large_notes_path": str(self.large_notes_path),
            "large_notes_count": self.large_note_count,
            "notes_token_budget": self.notes_token_budget,
            "notes_total_tokens": self._notes_total_tokens(),
        }
