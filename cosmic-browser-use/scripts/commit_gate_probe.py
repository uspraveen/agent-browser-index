#!/usr/bin/env python3
"""Live probe for the commit-gate JavaScript: Enter parity, required-field
skip, valued-fields-only payloads, and card-edit application.

Runs the REAL probe/classifier/apply JS (imported from browser_controller)
against fixture pages in a real Chromium, pinning the behaviours the gate
must hold everywhere:

  1. Enter parity — a visible type=submit control gates Enter regardless of
     its name ("Continue" submits a login just like "Submit" does). This is
     the exact bypass that once sent a magic-link email with no gate at all.
  2. Required-empty forms skip the gate: the browser itself will block that
     submission, so Enter is navigation, not a commit.
  3. Search boxes and textareas (chat) stay free — Enter there never needs
     authorization.
  4. A single-field form with no submit button still submits on Enter, so it
     gates (email captures) unless it reads like a search field.
  5. The card payload lists only fields that hold a value and reports how
     many are empty.
  6. Card edits land in the labelled fields (native setter + input/change
     events), masked values never become real values, and unmatched labels
     are reported, not guessed.

Run on a machine with the engine's deps + Playwright browsers:
    python scripts/commit_gate_probe.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from playwright.async_api import async_playwright  # noqa: E402

from browser_controller import (  # noqa: E402
    _APPLY_FIELD_EDITS_JS,
    _COMMIT_CLASSIFY_JS,
    _ENTER_COMMIT_PROBE_JS,
)

LOGIN_CONTINUE = """
<form>
  <label for="email">Email</label>
  <input id="email" name="email" type="email" value="uspraveenraj@gmail.com">
  <button type="submit">Continue</button>
</form>
"""

LOGIN_CONTINUE_REQUIRED_GAP = """
<form>
  <label for="email">Email</label>
  <input id="email" name="email" type="email" value="uspraveenraj@gmail.com">
  <label for="company">Company</label>
  <input id="company" name="company" required>
  <button type="submit">Continue</button>
</form>
"""

SEARCH_FORM = """
<form action="/search">
  <label for="q">Search</label>
  <input id="q" name="q" type="text" value="playwright">
  <button type="submit">Search</button>
</form>
"""

CHAT_TEXTAREA = """
<form>
  <textarea id="msg" name="message">hello there</textarea>
  <button type="submit">Send</button>
</form>
"""

SINGLE_FIELD_CAPTURE = """
<form>
  <label for="nl">Join the newsletter</label>
  <input id="nl" name="newsletter_email" type="email" value="uspraveenraj@gmail.com">
</form>
"""

SINGLE_FIELD_SEARCH = """
<form>
  <input id="f" name="filter_text" type="text" placeholder="Filter results">
</form>
"""

VERB_BUTTON = """
<form>
  <label for="email">Email</label>
  <input id="email" name="email" type="text" value="x@y.z">
  <button type="submit">Submit application</button>
</form>
"""

BIG_FORM = """
<form>
  <label for="email">Email</label>
  <input id="email" name="email" type="text" value="uspraveenraj@gmail.com">
  <input id="f1" name="f1" type="text">
  <input id="f2" name="f2" type="text">
  <input id="f3" name="f3" type="text">
  <label for="tos">Accept terms</label>
  <input id="tos" name="tos" type="checkbox" checked>
  <label for="plan">Plan</label>
  <select id="plan" name="plan"><option value="pro">Pro</option><option value="team" selected>Team</option></select>
  <input id="f4" name="f4" type="text">
  <button type="submit">Submit</button>
</form>
"""

EDIT_FORM = """
<form>
  <label for="name">Full name</label>
  <input id="name" name="name" type="text" value="Praveen Raj">
  <label for="company">Company</label>
  <input id="company" name="company" type="text" value="Acme">
  <label for="pw">Password</label>
  <input id="pw" name="password" type="password" value="hunter2">
  <button id="submit" type="submit">Submit</button>
</form>
<script>
  window.__events = [];
  document.getElementById('name').addEventListener('input', () => window.__events.push('name:input'));
  document.getElementById('name').addEventListener('change', () => window.__events.push('name:change'));
  document.getElementById('company').addEventListener('input', () => window.__events.push('company:input'));
</script>
"""


class Probe:
    def __init__(self) -> None:
        self.results: list[tuple[str, str, str]] = []

    def record(self, name: str, ok: bool, note: str) -> None:
        self.results.append((name, "PASS" if ok else "FAIL", note))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {note}")

    def report(self) -> int:
        fails = [r for r in self.results if r[1] == "FAIL"]
        print("\n=== SUMMARY ===")
        for name, status, note in self.results:
            print(f"  {status:4} {name}: {note}")
        if fails:
            print(f"\n{len(fails)} FAILURE(S)")
            return 1
        print("\nAll scenarios passed.")
        return 0


async def enter_probe(page, focus_id: str):
    await page.evaluate(f"document.getElementById('{focus_id}').focus()")
    return await page.evaluate(_ENTER_COMMIT_PROBE_JS)


async def main() -> int:
    probe = Probe()
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page()

        async def load(html: str) -> None:
            await page.set_content(html, wait_until="domcontentloaded")

        # 1. Enter parity: type=submit "Continue" gates Enter (the old probe
        #    demanded a verb-named control and let this submission through).
        await load(LOGIN_CONTINUE)
        info = await enter_probe(page, "email")
        probe.record(
            "enter_parity_continue_button",
            bool(info and info.get("is_commit")) and info.get("name") == "Continue",
            f"is_commit={bool(info and info.get('is_commit'))} name={info and info.get('name')!r}",
        )

        # 2. Required field still empty: the browser blocks this submission,
        #    Enter is navigation — no gate.
        await load(LOGIN_CONTINUE_REQUIRED_GAP)
        info = await enter_probe(page, "email")
        probe.record(
            "enter_free_when_required_empty",
            info is None,
            f"probe={info!r}",
        )

        # 3. Search boxes stay free.
        await load(SEARCH_FORM)
        info = await enter_probe(page, "q")
        probe.record("enter_free_search_form", info is None, f"probe={info!r}")

        # 4. Textareas never implicit-submit, so chat is never gated.
        await load(CHAT_TEXTAREA)
        info = await enter_probe(page, "msg")
        probe.record("enter_free_chat_textarea", info is None, f"probe={info!r}")

        # 5. Single-field, no submit button: Enter still submits — gate it.
        await load(SINGLE_FIELD_CAPTURE)
        info = await enter_probe(page, "nl")
        ok = bool(info and info.get("is_commit")) and any(
            f.get("label") == "Join the newsletter" for f in (info or {}).get("fields", [])
        )
        probe.record("enter_gates_single_field_capture", ok, f"probe_fields={(info or {}).get('fields')!r}")

        # 6. A search-named single field stays free.
        await load(SINGLE_FIELD_SEARCH)
        info = await enter_probe(page, "f")
        probe.record("enter_free_single_field_search", info is None, f"probe={info!r}")

        # 7. Verb-named submit buttons still gate (unchanged legacy rule).
        await load(VERB_BUTTON)
        info = await enter_probe(page, "email")
        probe.record(
            "enter_still_gates_verb_buttons",
            bool(info and info.get("is_commit")),
            f"name={info and info.get('name')!r}",
        )

        # 8. Payload: valued fields only + an honest empty count. The card
        #    reads "email + terms + plan, 4 empty" — not a wall of dashes.
        await load(BIG_FORM)
        info = await page.evaluate(
            "(function () { const el = document.querySelector('button[type=submit]');"
            " return (" + _COMMIT_CLASSIFY_JS + ")(el); })()"
        )
        labels = [f.get("label") for f in (info or {}).get("fields", [])]
        ok = (
            (info or {}).get("is_commit") is True
            and labels == ["Email", "Accept terms", "Plan"]
            and (info or {}).get("empty_field_count") == 4
        )
        probe.record(
            "payload_valued_fields_and_empty_count",
            ok,
            f"fields={labels} empty={(info or {}).get('empty_field_count')}",
        )

        # 9. Card edits land in the labelled field (native setter so React
        #    accepts it), fire input/change, never touch passwords, and
        #    report unmatched labels instead of guessing.
        await load(EDIT_FORM)
        await page.evaluate(
            "document.getElementById('submit').setAttribute('data-cosmic-commit-gate', 'm1')"
        )
        outcome = await page.evaluate(
            _APPLY_FIELD_EDITS_JS,
            [
                {"label": "Full name", "value": "Praveen Raj U S"},
                {"label": "Password", "value": "********"},
                {"label": "Ghost field", "value": "x"},
            ],
        )
        name_value = await page.evaluate("document.getElementById('name').value")
        pw_value = await page.evaluate("document.getElementById('pw').value")
        events = await page.evaluate("window.__events")
        ok = (
            name_value == "Praveen Raj U S"
            and pw_value == "hunter2"
            and outcome.get("applied") == ["Full name"]
            and outcome.get("pending") == ["Ghost field"]
            and "name:input" in events
            and "name:change" in events
        )
        probe.record(
            "card_edits_apply_to_labels_only",
            ok,
            f"name={name_value!r} pw_touched={pw_value != 'hunter2'} outcome={outcome}",
        )

        await browser.close()
    return probe.report()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
