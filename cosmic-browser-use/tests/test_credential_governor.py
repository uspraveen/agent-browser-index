"""Governor stand-down and human-wait accounting.

The PetScreening night: one login produced thirteen password cards in 39
minutes. Three deterministic causes are pinned here — the governor re-firing
on its cooldown while the field is still visible, the governor ignoring that
the run already holds vault credentials for the site, and the run watchdog
charging human wait time against the run's budget (720 of 840 seconds, then
cancelled mid-prompt).
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main as browser_main  # noqa: E402


class _FakePage:
    def __init__(self, url: str) -> None:
        self.url = url

    async def evaluate(self, *_args, **_kwargs):
        return "password"


class _FakeCredentialStore:
    def __init__(self, urls=()) -> None:
        self.urls = tuple(urls)

    def __len__(self) -> int:
        return len(self.urls)

    def get(self, url: str):
        for known in self.urls:
            if known == url:
                return {"site_domain": "petscreening.com"}
        return None


class _FakeBrowser:
    def __init__(self, url: str, *, shown: bool = False, creds=()) -> None:
        self.page = _FakePage(url)
        self.credential_store = _FakeCredentialStore(creds) if creds else None
        self._shown = shown

    def credential_prompt_shown_for(self, *_args, **_kwargs) -> bool:
        return self._shown


def _governor(browser, step_num: int = 5, last: int = 0):
    return asyncio.run(
        browser_main._should_try_credential_handoff_governor(
            browser=browser,
            config=None,  # unused by the governor
            step_num=step_num,
            last_attempt_step=last,
        )
    )


def test_governor_fires_once_on_a_visible_password_field():
    browser = _FakeBrowser("https://thewatersatchenal.petscreening.com/users/sign_in")
    assert _governor(browser, step_num=5) == "password"


def test_governor_stands_down_after_this_site_was_already_asked():
    browser = _FakeBrowser(
        "https://thewatersatchenal.petscreening.com/users/sign_in",
        shown=True,
    )
    assert _governor(browser, step_num=8) is None


def test_governor_stands_down_when_vault_credentials_cover_the_site():
    browser = _FakeBrowser(
        "https://thewatersatchenal.petscreening.com/users/sign_in",
        creds=("https://thewatersatchenal.petscreening.com/users/sign_in",),
    )
    assert _governor(browser, step_num=5) is None


def test_governor_still_covers_other_sites_after_one_site_was_asked():
    browser = _FakeBrowser("https://accounts.google.com/signin", shown=False)
    # credential_prompt_shown_for is per-site in the real controller; the fake
    # reports the flag it was built with, so a fresh site still fires.
    assert _governor(browser, step_num=5) == "password"


def test_paused_sec_adds_human_wait_to_takeover_time():
    class _Takeover:
        total_paused_sec = 10.0

    assert browser_main._paused_sec(_Takeover()) == 10.0
    assert browser_main._paused_sec(_Takeover(), lambda: 720.0) == 730.0
    assert browser_main._paused_sec(_Takeover(), lambda: (_ for _ in ()).throw(RuntimeError())) == 10.0


def test_human_wait_clock_counts_live():
    from cosmic_browser_use.api import HumanWaitClock

    clock = HumanWaitClock()
    assert clock.total() == 0.0
    clock.start()
    time.sleep(0.25)
    # Counts mid-wait — the watchdog polls while the human is still typing.
    assert clock.total() >= 0.25
    clock.stop()
    frozen = clock.total()
    time.sleep(0.05)
    assert clock.total() == frozen
    clock.start()
    time.sleep(0.05)
    assert clock.total() > frozen


def test_run_watchdog_gives_human_wait_time_back():
    from cosmic_browser_use.api import HumanWaitClock, _await_run_with_budget

    clock = HumanWaitClock()

    async def _run() -> str:
        clock.start()
        await asyncio.sleep(0.6)
        clock.stop()
        return "done"

    class _NoTakeover:
        total_paused_sec = 0.0

    # 0.3s budget: without the wait credit the watchdog cancels; with it the
    # run completes because the 0.6s it spent waiting on a human is free.
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(
            _await_run_with_budget(
                _run(),
                timeout=0.3,
                takeover_session=_NoTakeover(),
            )
        )

    clock2 = HumanWaitClock()

    async def _run2() -> str:
        clock2.start()
        await asyncio.sleep(0.6)
        clock2.stop()
        return "done"

    assert (
        asyncio.run(
            _await_run_with_budget(
                _run2(),
                timeout=0.3,
                takeover_session=_NoTakeover(),
                human_wait_getter=clock2.total,
            )
        )
        == "done"
    )
