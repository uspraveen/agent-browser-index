"""Regression tests for the CredentialFill contract.

CredentialFill is the one action that writes vault values into a page, and
the HN login failure (Oct 2026) showed exactly how that goes wrong when the
action is opaque: the model typed a handle, the fill silently replaced it
with the vault username, submitted, and the model then diagnosed the login
failure as "the site rejects that handle" — a conclusion that was never
tested. These tests pin the three properties that prevent that story from
repeating: the caller can direct the username field, the result reports what
was actually written and what it overwrote, and secrets plus unintended
submits stay out of the action by default.

Run with:  python -m pytest tests/test_credential_fill.py -q
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from browser_controller import BrowserController  # noqa: E402
from credentials import CredentialStore  # noqa: E402

ENTRY_USERNAME = "usp@alumni.upenn.edu"
ENTRY_PASSWORD = "vault-secret-value-42"


class _FakePage:
    """Stands in for the live page: records the args the JS would receive
    and returns a canned fill_state, mirroring the real script's contract
    (a field is only reported filled when its arg value is non-empty)."""

    def __init__(self, url: str, fill_state: dict | None = None):
        self.url = url
        self.fill_state = fill_state
        self.evaluated_args: dict | None = None

    async def evaluate(self, _js, args=None):
        self.evaluated_args = dict(args or {})
        state = dict(self.fill_state or {})
        state.setdefault("found", True)
        filled = list(state.get("filled") or [])
        if not filled:
            if self.evaluated_args.get("username"):
                filled.append("username")
            if self.evaluated_args.get("password"):
                filled.append("password")
            if self.evaluated_args.get("totp"):
                filled.append("totp")
        state["filled"] = filled
        state.setdefault("submitted", bool(self.evaluated_args.get("submit")) and "password" in filled)
        return state


def _controller(page: _FakePage) -> BrowserController:
    store = CredentialStore()
    store.add(
        "news.ycombinator.com",
        username=ENTRY_USERNAME,
        password=ENTRY_PASSWORD,
        site_url="https://news.ycombinator.com/login",
    )
    controller = object.__new__(BrowserController)
    controller.page = page
    controller.credential_store = store
    return controller


def _fill(page: _FakePage, parameters: dict | None = None):
    controller = _controller(page)
    result = asyncio.run(controller._credential_fill(parameters or {}))
    output = json.loads(result.output) if result.output else {}
    return result, output, page.evaluated_args or {}


def test_submit_defaults_to_false():
    """An omitted submit flag must never submit; read-only verification
    goals used to submit the form because the default was true."""
    result, output, args = _fill(_FakePage("https://news.ycombinator.com/login"))
    assert args["submit"] is False
    assert output["submitted"] is False
    assert "submitted" not in result.description


def test_explicit_submit_true_still_submits():
    result, output, args = _fill(
        _FakePage("https://news.ycombinator.com/login"), {"submit": True}
    )
    assert args["submit"] is True
    assert output["submitted"] is True
    assert result.description.endswith(" and submitted")


def test_string_submit_false_is_honored():
    _result, output, args = _fill(
        _FakePage("https://news.ycombinator.com/login"), {"submit": "false"}
    )
    assert args["submit"] is False
    assert output["submitted"] is False


def test_default_username_comes_from_the_vault():
    _result, output, args = _fill(_FakePage("https://news.ycombinator.com/login"))
    assert args["username"] == ENTRY_USERNAME
    assert output["username_source"] == "vault"


def test_username_override_replaces_the_vault_value():
    result, output, args = _fill(
        _FakePage("https://news.ycombinator.com/login"),
        {"username": "praveenrajus", "submit": True},
    )
    assert args["username"] == "praveenrajus"
    assert output["username_source"] == "action"
    assert args["password"] == ENTRY_PASSWORD, "the password must still come from the vault"
    assert "the value you passed" in result.description


def test_non_string_username_override_is_coerced():
    _result, output, args = _fill(
        _FakePage("https://news.ycombinator.com/login"), {"username": 12345}
    )
    assert args["username"] == "12345"
    assert output["username_source"] == "action"


def test_empty_username_keeps_the_typed_value():
    """username:\"\" is the password-only fill: whatever the model typed
    into the field survives, which is the property whose absence caused the
    Oct 2026 HN wrong-conclusion loop."""
    result, output, args = _fill(
        _FakePage("https://news.ycombinator.com/login"),
        {"username": "", "submit": True},
    )
    assert args["username"] == ""
    assert output["username_source"] == "skipped"
    assert "username" not in output["fields"]
    assert "username field left untouched" in result.description


def test_report_names_the_value_it_overwrote():
    """The exact HN failure: the fill replaced a typed handle. The result
    must say so instead of leaving the model to assume its text survived."""
    page = _FakePage(
        "https://news.ycombinator.com/login",
        {"found": True, "previous_username": "praveenrajus", "password_had_value": False},
    )
    result, output, _args = _fill(page, {"submit": True})
    assert "overwrote 'praveenrajus'" in result.description
    assert output["overwrote_username"] == "praveenrajus"


def test_no_overwrite_is_reported_as_none():
    page = _FakePage(
        "https://news.ycombinator.com/login",
        {"found": True, "previous_username": "", "password_had_value": False},
    )
    result, output, _args = _fill(page)
    assert "overwrote" not in result.description
    assert output["overwrote_username"] is None


def test_report_flags_a_replaced_password_without_its_value():
    page = _FakePage(
        "https://news.ycombinator.com/login",
        {"found": True, "previous_username": "", "password_had_value": True},
    )
    result, output, _args = _fill(page)
    assert "replaced what was in the field" in result.description
    assert output["password_field_had_value"] is True
    assert ENTRY_PASSWORD not in result.description
    assert ENTRY_PASSWORD not in result.output


def test_secrets_never_appear_in_result_text():
    result, output, _args = _fill(_FakePage("https://news.ycombinator.com/login"))
    assert ENTRY_PASSWORD not in result.description
    assert ENTRY_PASSWORD not in result.output
    assert ENTRY_PASSWORD not in json.dumps(output)
    assert result.error is None


def test_output_carries_the_site_and_username_source():
    _result, output, _args = _fill(
        _FakePage("https://news.ycombinator.com/login"), {"username": "praveenrajus"}
    )
    assert output["status"] == "filled"
    assert output["site"] == "ycombinator.com"  # normalized registrable host
    assert output["username_source"] == "action"


def test_no_password_field_still_reports_cleanly():
    page = _FakePage(
        "https://news.ycombinator.com/login",
        {"found": False, "reason": "no visible password field on this page"},
    )
    result, output, _args = _fill(page, {"submit": True})
    assert result.success is False
    assert output["status"] == "no_login_form"
    assert "no visible password field" in result.error
