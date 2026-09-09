"""Regression tests for human takeover.

The behaviours that matter here are the ones that are silent when they break:
a takeover that leaves no trace in memory (so the agent redoes the human's
work), a paused clock that keeps running (so a handover quietly shrinks the
agent's own budget), and an input relay that forwards something it should have
dropped.

Run with:  python -m pytest tests/test_takeover.py -q
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from takeover import (  # noqa: E402
    TakeoverRecord,
    TakeoverSession,
    describe_takeover,
    diff_storage_origins,
    run_takeover,
)


# ---------------------------------------------------------------- fakes


class FakeState:
    def __init__(self, url: str, title: str) -> None:
        self.url = url
        self.title = title


class FakeContext:
    def __init__(self, states):
        self._states = list(states)

    async def storage_state(self):
        return self._states.pop(0) if self._states else {}


class FakeBrowser:
    """Stands in for BrowserController.

    capture_state mirrors the real signature deliberately: it takes a
    screenshot name and returns (path, hash, state), not a bare state. An
    earlier version of this fake accepted no arguments and returned the state
    directly, which made the tests pass against a contract BrowserController
    does not have - every real capture would have raised, been swallowed, and
    recorded a takeover with no evidence in it.
    """

    def __init__(self, states, storage=None, fail_input=False):
        self._states = list(states)
        self.context = FakeContext(storage or [{}, {}])
        self.dispatched = []
        self.fail_input = fail_input
        self.captures = []

    async def capture_state(self, screenshot_name: str):
        self.captures.append(screenshot_name)
        state = self._states.pop(0) if self._states else FakeState("", "")
        return (f"/runs/screenshots/{screenshot_name}.webp", f"hash-{screenshot_name}", state)

    async def dispatch_human_input(self, event):
        if self.fail_input:
            raise RuntimeError("cdp is gone")
        self.dispatched.append(event)
        return True


# ---------------------------------------------------------- storage delta


def test_storage_diff_detects_a_new_login():
    before = {"cookies": [{"domain": "example.com"}], "origins": []}
    after = {
        "cookies": [{"domain": "example.com"}, {"domain": ".accounts.google.com"}],
        "origins": [],
    }
    delta = diff_storage_origins(before, after)
    assert delta["new"] == ["accounts.google.com"]
    assert delta["dropped"] == []


def test_storage_diff_folds_cookie_domains_and_origins_together():
    before = {"cookies": [], "origins": []}
    after = {
        "cookies": [{"domain": ".github.com"}],
        "origins": [{"origin": "https://github.com"}],
    }
    # Same host reached two different ways must not read as two logins.
    assert diff_storage_origins(before, after)["new"] == ["github.com"]


def test_storage_diff_reports_a_signed_out_origin():
    before = {"cookies": [{"domain": "example.com"}]}
    after = {"cookies": []}
    assert diff_storage_origins(before, after)["dropped"] == ["example.com"]


def test_storage_diff_survives_missing_state():
    assert diff_storage_origins(None, None) == {"new": [], "dropped": []}


# ------------------------------------------------------------- narration


def _record(**kwargs) -> TakeoverRecord:
    base = dict(started_at=time.time() - 30, ended_at=time.time())
    base.update(kwargs)
    return TakeoverRecord(**base)


def test_description_names_the_page_the_human_left_behind():
    text = describe_takeover(
        _record(url_before="https://a.test/1", url_after="https://a.test/2")
    )
    assert "https://a.test/2" in text
    assert "do not repeat what they just did" in text


def test_description_reports_a_login_the_screenshot_would_not_show():
    text = describe_takeover(_record(new_origins=["accounts.google.com"]))
    assert "signed-in state for accounts.google.com" in text


def test_description_truncates_a_long_origin_list():
    text = describe_takeover(_record(new_origins=[f"s{i}.test" for i in range(7)]))
    assert "+3 more" in text


def test_description_carries_the_human_note():
    text = describe_takeover(_record(human_note="I solved the captcha"))
    assert "I solved the captcha" in text


def test_description_says_so_when_nothing_moved():
    assert "stayed on the same page" in describe_takeover(_record())


def test_description_flags_a_timeout():
    assert "timed out" in describe_takeover(_record(timed_out=True))


# --------------------------------------------------------------- session


def test_input_is_refused_while_the_session_is_not_paused():
    session = TakeoverSession()
    # Nothing is parked, so there is no page to drive: accepting here would
    # queue events that later fire at whatever the agent had navigated to.
    assert session.submit_input({"kind": "mouse"}) is False


def test_input_queued_before_the_pause_is_discarded():
    session = TakeoverSession()
    session._active = True
    session.submit_input({"kind": "mouse", "seq": 1})
    session._begin()
    assert session._drain() == []


def test_drain_batches_rather_than_returning_everything():
    session = TakeoverSession()
    session._begin()
    for i in range(100):
        session.submit_input({"kind": "mouse", "seq": i})
    first = session._drain()
    assert len(first) == 32
    assert first[0]["seq"] == 0


def test_full_mailbox_drops_instead_of_blocking():
    session = TakeoverSession()
    session._begin()
    accepted = sum(1 for i in range(600) if session.submit_input({"kind": "mouse", "seq": i}))
    assert accepted == 512


def test_timeout_has_a_floor():
    assert TakeoverSession(timeout_sec=1).timeout_sec == 30.0


# --------------------------------------------------------------- the run


def test_takeover_relays_input_then_records_what_changed():
    session = TakeoverSession()
    browser = FakeBrowser(
        states=[FakeState("https://a.test/login", "Sign in"), FakeState("https://a.test/home", "Home")],
        storage=[{"cookies": []}, {"cookies": [{"domain": "a.test"}]}],
    )

    async def scenario():
        session.request_pause()
        task = asyncio.ensure_future(run_takeover(session=session, browser=browser))
        await asyncio.sleep(0.01)
        session.submit_input({"kind": "mouse", "type": "mousePressed"})
        await asyncio.sleep(0.12)
        session.resume(note="signed in for you")
        return await task

    record = asyncio.run(scenario())
    assert record.input_events == 1
    assert browser.dispatched[0]["type"] == "mousePressed"
    assert record.url_before == "https://a.test/login"
    assert record.url_after == "https://a.test/home"
    assert record.new_origins == ["a.test"]
    assert record.human_note == "signed in for you"
    assert "signed in for you" in record.summary


def test_takeover_captures_evidence_through_the_real_capture_signature():
    # capture_state(name) -> (path, hash, state). Calling it any other way
    # raises into _safe and yields a takeover record with no evidence, which
    # is silent: the run continues and the agent simply learns nothing about
    # what the human did.
    session = TakeoverSession()
    browser = FakeBrowser(states=[FakeState("https://a.test/1", "One"), FakeState("https://a.test/2", "Two")])

    async def scenario():
        session.request_pause()
        task = asyncio.ensure_future(run_takeover(session=session, browser=browser))
        await asyncio.sleep(0.03)
        session.resume()
        return await task

    record = asyncio.run(scenario())
    assert len(browser.captures) == 2
    assert browser.captures[0].startswith("takeover_before_")
    assert browser.captures[1].startswith("takeover_after_")
    assert record.screenshot_before.endswith(".webp")
    assert record.screenshot_after_hash.startswith("hash-takeover_after_")
    assert record.state_before is not None and record.state_after is not None
    assert record.url_before == "https://a.test/1"
    assert record.url_after == "https://a.test/2"


def test_a_browser_that_cannot_capture_still_hands_control_back():
    class Unavailable(FakeBrowser):
        async def capture_state(self, screenshot_name: str):
            raise RuntimeError("page is gone")

    session = TakeoverSession()
    browser = Unavailable(states=[])

    async def scenario():
        session.request_pause()
        task = asyncio.ensure_future(run_takeover(session=session, browser=browser))
        await asyncio.sleep(0.03)
        session.resume()
        return await task

    record = asyncio.run(scenario())
    assert record.ended_at > 0
    assert session.active is False
    assert record.summary  # a thin record, but a record


def test_a_synchronously_raising_browser_call_cannot_kill_the_run():
    # The original _safe took the *result* of a call, so the call happened
    # outside its try. A signature mismatch (or a closed page) therefore raised
    # straight through run_takeover - which has no except - and out into the
    # step loop, killing a run that was only trying to hand over.
    class WrongArity(FakeBrowser):
        async def capture_state(self):  # no screenshot_name: raises at call time
            return None

    session = TakeoverSession()
    browser = WrongArity(states=[])

    async def scenario():
        session.request_pause()
        task = asyncio.ensure_future(run_takeover(session=session, browser=browser))
        await asyncio.sleep(0.03)
        session.resume()
        return await task

    record = asyncio.run(scenario())  # must not raise
    assert record.ended_at > 0
    assert session.active is False


def test_takeover_ends_on_timeout_without_losing_the_run():
    session = TakeoverSession()
    session.timeout_sec = 0.05  # bypasses the constructor floor deliberately
    browser = FakeBrowser(states=[FakeState("https://a.test", "A"), FakeState("https://a.test", "A")])

    async def scenario():
        session.request_pause()
        return await run_takeover(session=session, browser=browser)

    record = asyncio.run(scenario())
    assert record.timed_out is True
    assert session.active is False
    assert session.pause_requested is False


def test_a_broken_input_path_does_not_break_the_takeover():
    session = TakeoverSession()
    browser = FakeBrowser(
        states=[FakeState("https://a.test", "A"), FakeState("https://a.test", "A")],
        fail_input=True,
    )

    async def scenario():
        session.request_pause()
        task = asyncio.ensure_future(run_takeover(session=session, browser=browser))
        await asyncio.sleep(0.01)
        session.submit_input({"kind": "mouse", "type": "mousePressed"})
        await asyncio.sleep(0.12)
        session.resume()
        return await task

    record = asyncio.run(scenario())
    assert record.input_events == 0  # dropped, not counted, not raised
    assert session.active is False


def test_paused_time_accumulates_across_takeovers():
    session = TakeoverSession()
    browser = FakeBrowser(states=[FakeState("u", "t")] * 8)

    async def once():
        session.request_pause()
        task = asyncio.ensure_future(run_takeover(session=session, browser=browser))
        await asyncio.sleep(0.06)
        session.resume()
        await task

    async def scenario():
        await once()
        first = session.paused_sec
        await once()
        return first, session.paused_sec

    first, total = asyncio.run(scenario())
    assert first > 0
    assert total > first
    assert len(session.records) == 2


def test_state_callback_sees_both_edges():
    session = TakeoverSession()
    browser = FakeBrowser(states=[FakeState("u", "t")] * 4)
    seen = []

    async def scenario():
        session.request_pause()
        task = asyncio.ensure_future(
            run_takeover(
                session=session,
                browser=browser,
                on_state=lambda phase, record: seen.append(phase),
            )
        )
        await asyncio.sleep(0.03)
        session.resume()
        await task

    asyncio.run(scenario())
    assert seen == ["paused", "resumed"]


def test_a_failing_state_callback_never_reaches_the_run():
    session = TakeoverSession()
    browser = FakeBrowser(states=[FakeState("u", "t")] * 4)

    def explode(phase, record):
        raise RuntimeError("desktop went away")

    async def scenario():
        session.request_pause()
        task = asyncio.ensure_future(
            run_takeover(session=session, browser=browser, on_state=explode)
        )
        await asyncio.sleep(0.03)
        session.resume()
        return await task

    record = asyncio.run(scenario())
    assert record.ended_at > 0


# ------------------------------------------------------- the run watchdog


def test_live_pause_counts_before_the_takeover_ends():
    # paused_sec only grows when a takeover finishes. Anything watching a
    # deadline needs the in-progress pause too, or it fires mid-handover.
    session = TakeoverSession()
    session._begin()
    time.sleep(0.05)
    assert session.paused_sec == 0.0
    assert session.total_paused_sec >= 0.05
    session._end()


def test_watchdog_does_not_cancel_a_run_that_is_paused():
    from cosmic_browser_use.api import _await_run_with_budget

    session = TakeoverSession()

    async def slow_run():
        await asyncio.sleep(0.4)
        return "finished"

    async def scenario():
        session._begin()  # the human has the wheel for the whole run
        try:
            # A 0.2s budget against a 0.4s run: only the paused clock saves it.
            return await _await_run_with_budget(
                slow_run(), timeout=0.2, takeover_session=session
            )
        finally:
            session._end()

    assert asyncio.run(scenario()) == "finished"


def test_watchdog_still_enforces_the_budget_when_nobody_is_paused():
    from cosmic_browser_use.api import _await_run_with_budget

    async def slow_run():
        await asyncio.sleep(5)
        return "finished"

    async def scenario():
        return await _await_run_with_budget(slow_run(), timeout=0.2, takeover_session=None)

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(scenario())


def test_watchdog_returns_the_result_immediately_when_the_run_finishes():
    from cosmic_browser_use.api import _await_run_with_budget

    async def quick():
        return {"task_status": "success"}

    result = asyncio.run(_await_run_with_budget(quick(), timeout=30, takeover_session=None))
    assert result["task_status"] == "success"


def test_watchdog_propagates_a_run_failure_unchanged():
    from cosmic_browser_use.api import _await_run_with_budget

    async def boom():
        raise ValueError("run blew up")

    with pytest.raises(ValueError, match="run blew up"):
        asyncio.run(_await_run_with_budget(boom(), timeout=30, takeover_session=None))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
