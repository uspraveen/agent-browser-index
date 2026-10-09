"""Live-Chromium pins for how the snapshot names form fields.

The 2026-10-08 Partiful RSVP run typed an email address into the name field
because the field reached every decision maker as a bare `textbox`: its
visible label was a floating sibling <label> with no `for`, and the follow-up
page's questions were plain text above unlabeled inputs. These run the real
collector JS in a real page, because a fake DOM would only prove the fake.

Skipped when no Chromium is installed.

Run with:  python -m pytest tests/test_field_labels_live.py -q
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from browser_controller import (  # noqa: E402
    _FIELD_LABEL_JS,
    _SNAPSHOT_COLLECT_JS,
    _SNAPSHOT_INTERACTIVE_CSS,
    format_disabled_lines,
)

PARTIFUL_LIKE = """
<form>
  <div class="row"><input placeholder="&nbsp;" autocomplete="name"><label>Your Name</label></div>
  <div class="row"><span><button type="button" tabindex="-1">US</button></span>
    <input type="tel" name="phoneNumber" placeholder="Phone Number"></div>
  <div class="msg"><textarea placeholder="Write something fun!"></textarea><label> </label></div>
  <button type="button">Cancel</button>
  <button type="submit" aria-disabled="true">Continue</button>
</form>
"""

HOST_QUESTIONS = """
<section>
  <h3>Questions from the hosts</h3><p>Only the hosts can see your answers</p>
  <div class="q"><div>First name *</div><div><input value="Praveen Raj"></div></div>
  <div class="q"><div>What is your LinkedIn? *</div><div><input></div></div>
  <div class="q"><div>What is the name of your company? *</div><div><input></div></div>
  <div class="q"><div>Price</div><div><span>$</span><input id="price"></div></div>
  <div class="q"><div>Are you currently a student? *</div><button type="button" aria-haspopup="listbox">Select…</button></div>
</section>
"""

EXPLICIT = """
<label for="a">Email address</label><input id="a" name="contact_field_7">
<input id="lonely" autocomplete="email">
"""


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


async def _collect(html: str):
    try:
        from playwright.async_api import async_playwright
    except ImportError:  # pragma: no cover
        pytest.skip("playwright not installed")
    async with async_playwright() as p:
        try:
            browser = await p.chromium.launch(headless=True)
        except Exception as exc:  # pragma: no cover
            pytest.skip(f"chromium unavailable: {exc}")
        page = await browser.new_page()
        await page.set_content(html)
        result = await page.evaluate(_SNAPSHOT_COLLECT_JS, {"css": _SNAPSHOT_INTERACTIVE_CSS, "cap": 100})
        lonely = await page.evaluate(f"(() => ({_FIELD_LABEL_JS})(document.getElementById('lonely')))()") \
            if "lonely" in html else None
        await browser.close()
        return result, lonely


def _names(result):
    return [(e["role"], e["name"]) for e in result["entries"]]


def test_floating_sibling_label_names_the_field():
    result, _ = _run(_collect(PARTIFUL_LIKE))
    names = _names(result)
    assert ("textbox", "Your Name") in names
    # A labelled-by-name-attribute field keeps its old name: only unlabeled
    # fields gain one, so existing fingerprints do not churn.
    assert ("textbox", "phoneNumber") in names
    assert ("textbox", "Write something fun!") in names


def test_disabled_submit_is_reported_without_a_ref():
    result, _ = _run(_collect(PARTIFUL_LIKE))
    assert all(e.get("name") != "Continue" for e in result["entries"])
    assert {"role": "button", "name": "Continue"} in result["disabled"]
    lines = format_disabled_lines(result["disabled"])
    assert lines[0].startswith("Disabled right now")
    assert '- button "Continue"' in lines


def test_question_text_names_unlabeled_inputs():
    result, _ = _run(_collect(HOST_QUESTIONS))
    names = [n for r, n in _names(result) if r == "textbox"]
    assert names[:3] == ["First name *", "What is your LinkedIn? *", "What is the name of your company? *"]


def test_section_heading_is_not_glued_onto_the_first_question():
    result, _ = _run(_collect(HOST_QUESTIONS))
    first = [n for r, n in _names(result) if r == "textbox"][0]
    assert "Questions from the hosts" not in first


def test_a_prefix_glyph_is_not_the_whole_label():
    # "$" alone says nothing; the walk continues out to the question and keeps
    # the unit beside it ("Price $").
    result, _ = _run(_collect(HOST_QUESTIONS))
    price = [n for r, n in _names(result) if r == "textbox"][3]
    assert price.startswith("Price")


def test_buttons_keep_their_own_text():
    result, _ = _run(_collect(HOST_QUESTIONS))
    assert ("button", "Select…") in _names(result)


def test_explicit_label_wins_and_autocomplete_is_the_last_resort():
    result, lonely = _run(_collect(EXPLICIT))
    assert ("textbox", "Email address") in _names(result)
    assert lonely == "autocomplete: email"


def test_form_values_reader_labels_values_and_masks_passwords():
    from browser_controller import _FORM_VALUES_JS

    async def go():
        try:
            from playwright.async_api import async_playwright
        except ImportError:  # pragma: no cover
            pytest.skip("playwright not installed")
        async with async_playwright() as p:
            try:
                browser = await p.chromium.launch(headless=True)
            except Exception as exc:  # pragma: no cover
                pytest.skip(f"chromium unavailable: {exc}")
            page = await browser.new_page()
            await page.set_content(PARTIFUL_LIKE + '<div><input type="password" id="pw"><label>Password</label></div>')
            await page.fill("input[autocomplete=name]", "Praveen Raj")
            await page.fill("#pw", "hunter2")
            rows = await page.evaluate(_FORM_VALUES_JS)
            await browser.close()
            return rows

    rows = _run(go())
    assert {"label": "Your Name", "value": "Praveen Raj"} in rows
    assert {"label": "Password", "value": "********"} in rows
    assert all("hunter2" not in r["value"] for r in rows)


def test_commit_card_fields_use_the_shared_labeler():
    """The Enter-commit probe lists the form's valued fields for the card and
    the orchestrator's check; they must carry the same label the snapshot
    shows, or "email under Your Name" is invisible to the check."""
    from browser_controller import _ENTER_COMMIT_PROBE_JS

    html = PARTIFUL_LIKE.replace(' aria-disabled="true"', "")

    async def go():
        try:
            from playwright.async_api import async_playwright
        except ImportError:  # pragma: no cover
            pytest.skip("playwright not installed")
        async with async_playwright() as p:
            try:
                browser = await p.chromium.launch(headless=True)
            except Exception as exc:  # pragma: no cover
                pytest.skip(f"chromium unavailable: {exc}")
            page = await browser.new_page()
            await page.set_content(html)
            await page.fill("input[autocomplete=name]", "uspraveenraj@gmail.com")
            await page.focus("input[autocomplete=name]")
            info = await page.evaluate(_ENTER_COMMIT_PROBE_JS)
            await browser.close()
            return info

    info = _run(go())
    assert info and info["is_commit"] is True
    assert {"label": "Your Name", "value": "uspraveenraj@gmail.com"} in info["fields"]
