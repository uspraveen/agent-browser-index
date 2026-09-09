"""Human takeover: pause the agent, hand the live browser to the user, resume.

Three ideas hold this together.

**The pause is a step boundary, never an interruption.** A run is only ever
parked between steps, after the action executed and memory was persisted.
Pausing mid-action would leave a half-typed field or an un-awaited navigation
and no record of either.

**The handover is recorded as a step, not narrated into a prompt.** Whatever
the human does is invisible to the agent otherwise: it resumes with a working
set describing a page that no longer exists, and re-does the sign-in the human
just completed. A takeover therefore produces a real Step with before/after
browser state and screenshots, exactly the shape every other step has, so every
mechanism that reads history - the working set, loop detection, the progress
test - sees it without having to know it is special.

**The interesting delta is in storage, not in the DOM.** A DOM diff of a real
page is mostly ad refreshes and framework class churn: enormous, and noise in
precisely the working memory we keep small on purpose. The fact that actually
matters after a takeover is usually "the human authenticated somewhere", which
neither a screenshot nor a DOM diff shows. storage_state() shows it in a
handful of bytes, so that is what gets captured and summarised.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse


# How long a takeover may hold the run before it is force-resumed. A paused run
# is holding a browser, a CDP session and a step budget; it must not hold them
# forever because a laptop went to sleep mid-handover.
DEFAULT_TAKEOVER_TIMEOUT_SEC = 900.0

# Input is drained in small batches so a burst of mouse-move events can never
# starve the resume check.
_INPUT_BATCH = 32
_IDLE_POLL_SEC = 0.05


@dataclass
class TakeoverRecord:
    """What happened while the human had the wheel."""

    started_at: float
    ended_at: float = 0.0
    url_before: str = ""
    url_after: str = ""
    title_before: str = ""
    title_after: str = ""
    screenshot_before: str = ""
    screenshot_after: str = ""
    # Origins present in storage state afterwards but not before. This is the
    # signal that says "they logged in", and the reason we snapshot storage
    # rather than the DOM.
    new_origins: List[str] = field(default_factory=list)
    dropped_origins: List[str] = field(default_factory=list)
    input_events: int = 0
    human_note: str = ""
    summary: str = ""
    timed_out: bool = False

    @property
    def duration_sec(self) -> float:
        return max(0.0, (self.ended_at or time.time()) - self.started_at)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_sec": round(self.duration_sec, 1),
            "url_before": self.url_before,
            "url_after": self.url_after,
            "title_before": self.title_before,
            "title_after": self.title_after,
            "new_origins": list(self.new_origins),
            "dropped_origins": list(self.dropped_origins),
            "input_events": self.input_events,
            "human_note": self.human_note,
            "summary": self.summary,
            "timed_out": self.timed_out,
        }


class TakeoverSession:
    """Pause/resume signalling and the input mailbox for one run.

    Lives in the library so the run loop can be tested without a gateway, and
    is driven from outside by whoever owns the user connection (in Cosmic, the
    browser agent process relaying from the desktop).
    """

    def __init__(self, *, timeout_sec: float = DEFAULT_TAKEOVER_TIMEOUT_SEC) -> None:
        self.timeout_sec = max(30.0, float(timeout_sec or DEFAULT_TAKEOVER_TIMEOUT_SEC))
        self._pause_requested = False
        self._resume_event = asyncio.Event()
        self._inputs: "asyncio.Queue[Dict[str, Any]]" = asyncio.Queue(maxsize=512)
        self._active = False
        self._active_since = 0.0
        self._human_note = ""
        self.records: List[TakeoverRecord] = []
        # Total seconds the run spent parked. The caller subtracts this from
        # elapsed time so a human's coffee break is not charged against the
        # agent's own time and step budgets.
        self.paused_sec = 0.0

    # -- driven from outside -------------------------------------------------

    def request_pause(self) -> None:
        """Ask the run to park at its next step boundary."""
        self._pause_requested = True
        self._resume_event.clear()

    def resume(self, note: str = "") -> None:
        """Hand control back. Safe to call when not paused."""
        self._human_note = str(note or "").strip()
        self._pause_requested = False
        self._resume_event.set()

    def submit_input(self, event: Dict[str, Any]) -> bool:
        """Queue one input event for the paused page.

        Drops rather than blocks when the mailbox is full: input is realtime,
        and a stale mouse-move is worth less than a fresh one.
        """
        if not self._active or not isinstance(event, dict):
            return False
        try:
            self._inputs.put_nowait(event)
            return True
        except asyncio.QueueFull:
            return False

    # -- read by the run loop ------------------------------------------------

    @property
    def pause_requested(self) -> bool:
        return self._pause_requested

    @property
    def active(self) -> bool:
        return self._active

    @property
    def total_paused_sec(self) -> float:
        """Paused seconds including a takeover that is happening right now.

        `paused_sec` only grows when a takeover ends, which is too late for
        anything watching a deadline: the run timeout would fire in the middle
        of a long handover and cancel the run the human is still working in.
        """
        live = time.time() - self._active_since if (self._active and self._active_since) else 0.0
        return self.paused_sec + max(0.0, live)

    def _begin(self) -> None:
        self._active = True
        self._active_since = time.time()
        self._human_note = ""
        self._resume_event.clear()
        while not self._inputs.empty():  # discard anything queued before we parked
            try:
                self._inputs.get_nowait()
            except asyncio.QueueEmpty:
                break

    def _end(self) -> str:
        self._active = False
        self._active_since = 0.0
        self._pause_requested = False
        return self._human_note

    def _drain(self) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        while len(events) < _INPUT_BATCH:
            try:
                events.append(self._inputs.get_nowait())
            except asyncio.QueueEmpty:
                break
        return events

    async def _wait_for_resume_or_input(self) -> None:
        """Sleep until there is something to do, without busy-waiting."""
        if not self._inputs.empty() or self._resume_event.is_set():
            return
        waiter = asyncio.ensure_future(self._resume_event.wait())
        try:
            await asyncio.wait({waiter}, timeout=_IDLE_POLL_SEC)
        finally:
            if not waiter.done():
                waiter.cancel()
                try:
                    await waiter
                except (asyncio.CancelledError, Exception):
                    pass


def _origins(storage_state: Optional[Dict[str, Any]]) -> set:
    """Origins represented in a Playwright storage state.

    Cookies are keyed by domain and localStorage by origin; both are folded to
    a bare host so "logged into google" reads the same either way.
    """
    found = set()
    if not isinstance(storage_state, dict):
        return found
    for cookie in storage_state.get("cookies") or []:
        if isinstance(cookie, dict):
            domain = str(cookie.get("domain") or "").strip().lstrip(".")
            if domain:
                found.add(domain.lower())
    for entry in storage_state.get("origins") or []:
        if isinstance(entry, dict):
            host = urlparse(str(entry.get("origin") or "")).netloc
            if host:
                found.add(host.lower())
    return found


def diff_storage_origins(
    before: Optional[Dict[str, Any]],
    after: Optional[Dict[str, Any]],
) -> Dict[str, List[str]]:
    """What sites the human gained or lost session state for."""
    before_set = _origins(before)
    after_set = _origins(after)
    return {
        "new": sorted(after_set - before_set),
        "dropped": sorted(before_set - after_set),
    }


def describe_takeover(record: TakeoverRecord) -> str:
    """A one-line, deterministic account of the handover.

    Deliberately not model-written: this is the line that goes into the agent's
    working set, it has to exist even when every LLM call is failing, and
    everything in it is observed fact. The human's own note carries the intent
    that observation cannot supply.
    """
    parts: List[str] = []
    if record.url_after and record.url_before != record.url_after:
        parts.append("left the page at " + record.url_after)
    else:
        parts.append("stayed on the same page")
    if record.new_origins:
        shown = ", ".join(record.new_origins[:4])
        extra = len(record.new_origins) - 4
        more = " (+" + str(extra) + " more)" if extra > 0 else ""
        parts.append("gained signed-in state for " + shown + more)
    if record.human_note:
        parts.append("said: " + record.human_note)
    tail = "; ".join(parts)
    timed = " The takeover timed out rather than being handed back." if record.timed_out else ""
    return (
        "A human paused the run for {:.0f}s and drove the browser: {}. "
        "Treat the current page as their work, not yours - re-read it before "
        "acting, and do not repeat what they just did.{}"
    ).format(record.duration_sec, tail, timed)


async def run_takeover(
    *,
    session: TakeoverSession,
    browser: Any,
    on_state: Optional[Callable[[str, TakeoverRecord], Any]] = None,
    screenshot_dir: Optional[str] = None,
) -> TakeoverRecord:
    """Park the run, relay the human's input, then record what changed.

    Never raises: a takeover that fails half-way still hands control back and
    still records what it managed to observe. Losing the run because the
    handover glitched would be a far worse outcome than a thin record.
    """
    record = TakeoverRecord(started_at=time.time())
    session._begin()
    before_storage = None
    try:
        state = await _safe(browser.capture_state())
        if state is not None:
            record.url_before = getattr(state, "url", "") or ""
            record.title_before = getattr(state, "title", "") or ""
        record.screenshot_before = await _safe_screenshot(browser, screenshot_dir, "takeover_before")
        before_storage = await _safe(_storage_state(browser))
        await _notify(on_state, "paused", record)

        deadline = record.started_at + session.timeout_sec
        while True:
            if session._resume_event.is_set():
                break
            if time.time() >= deadline:
                record.timed_out = True
                break
            for event in session._drain():
                if await _safe(browser.dispatch_human_input(event)):
                    record.input_events += 1
            await session._wait_for_resume_or_input()
    finally:
        record.human_note = session._end()
        record.ended_at = time.time()
        session.paused_sec += record.duration_sec
        try:
            state = await _safe(browser.capture_state())
            if state is not None:
                record.url_after = getattr(state, "url", "") or ""
                record.title_after = getattr(state, "title", "") or ""
            record.screenshot_after = await _safe_screenshot(browser, screenshot_dir, "takeover_after")
            after_storage = await _safe(_storage_state(browser))
            delta = diff_storage_origins(before_storage, after_storage)
            record.new_origins = delta["new"]
            record.dropped_origins = delta["dropped"]
        except Exception:
            pass
        record.summary = describe_takeover(record)
        session.records.append(record)
        await _notify(on_state, "resumed", record)
    return record


async def _storage_state(browser: Any) -> Optional[Dict[str, Any]]:
    context = getattr(browser, "context", None)
    if context is None:
        return None
    return await context.storage_state()


async def _safe(awaitable_or_value: Any) -> Any:
    """Await if awaitable, swallow anything that goes wrong."""
    try:
        if asyncio.iscoroutine(awaitable_or_value) or isinstance(awaitable_or_value, asyncio.Future):
            return await awaitable_or_value
        return awaitable_or_value
    except Exception:
        return None


async def _safe_screenshot(browser: Any, screenshot_dir: Optional[str], name: str) -> str:
    if not screenshot_dir:
        return ""
    try:
        path = "{}/{}_{}.jpg".format(screenshot_dir.rstrip("/"), name, int(time.time() * 1000))
        result = await browser.fast_screenshot(path)
        return path if result is not False else ""
    except Exception:
        return ""


async def _notify(
    callback: Optional[Callable[[str, TakeoverRecord], Any]],
    phase: str,
    record: TakeoverRecord,
) -> None:
    if callback is None:
        return
    try:
        result = callback(phase, record)
        if asyncio.iscoroutine(result):
            await result
    except Exception:
        pass
