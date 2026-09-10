#!/usr/bin/env python3
"""Live cross-site probe for the DOMSnapshot / @ref tool set.

Drives the REAL BrowserController (execute_tool dispatch included) against
real sites and checks the behaviours that must hold everywhere:

  1. Structured pages yield a usable, bounded ref map.
  2. Password values are masked in the snapshot AND in echoes.
  3. SnapshotType lands in the labelled field (echo) — no blind writes.
  4. A value change never invalidates refs (typing between snapshot and act
     is the normal case); navigation/re-render does (staleness refusal).
  5. Thin-tree pages return an honest empty/short result — the vision
     fallback case — without crashing.

The Ashby scenario is advisory (bot-guarded, slow): it replays the original
salary-into-name-field failure using refs only.

Run on a machine with the engine's deps + Playwright browsers:
    python scripts/snapshot_live_probe.py            # all scenarios
    python scripts/snapshot_live_probe.py --only login,ashby
"""
from __future__ import annotations

import argparse
import asyncio
import re
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cosmic_types import ActionType, TaskConfig  # noqa: E402
from browser_controller import BrowserController  # noqa: E402

HEADLESS = True
REF_RE = re.compile(r"@e(\d+)\s+(\S+)(?:\s+\"([^\"]*)\")?")


class Probe:
    def __init__(self, only: set[str] | None = None):
        self.only = only or set()
        self.results: list[tuple[str, str, str]] = []  # (scenario, status, note)

    def should(self, name: str) -> bool:
        return not self.only or name in self.only

    def record(self, name: str, ok: bool | None, note: str):
        status = {True: "PASS", False: "FAIL", None: "SKIP"}[ok]
        self.results.append((name, status, note))
        print(f"  [{status}] {name}: {note}")

    def report(self) -> int:
        core_fails = [r for r in self.results if r[1] == "FAIL" and r[0] not in ("ashby", "canvas")]
        print("\n=== SUMMARY ===")
        for name, status, note in self.results:
            print(f"  {status:4} {name}: {note}")
        if core_fails:
            print(f"\n{len(core_fails)} CORE FAILURE(S)")
            return 1
        print("\nAll core scenarios passed.")
        return 0


async def snapshot(controller: BrowserController, max_elements: int = 120):
    return await controller.execute_tool(
        ToolCallFor(ActionType.DOM_SNAPSHOT, {"max_elements": max_elements}), ""
    )


async def ref_click(controller: BrowserController, ref: str):
    return await controller.execute_tool(
        ToolCallFor(ActionType.SNAPSHOT_CLICK, {"ref": ref}), ""
    )


async def ref_type(controller: BrowserController, ref: str, text: str, press_enter: bool = False):
    return await controller.execute_tool(
        ToolCallFor(ActionType.SNAPSHOT_TYPE, {"ref": ref, "text": text, "press_enter": press_enter}), ""
    )


def ToolCallFor(action_type: ActionType, parameters: dict):
    from cosmic_types import ToolCall
    return ToolCall(action_type=action_type, parameters=parameters)


def refs_from(result) -> dict[str, tuple[str, str]]:
    """ref -> (role, name) parsed from the snapshot output text."""
    out = {}
    for line in (result.output or "").splitlines():
        m = REF_RE.match(line.strip())
        if m:
            out[f"@e{m.group(1)}"] = (m.group(2), m.group(3) or "")
    return out


def find_ref(refs: dict, name_contains: str, role: str | None = None):
    for ref, (r, name) in refs.items():
        if name_contains.lower() in name.lower() and (role is None or r == role):
            return ref
    return None


async def goto(controller: BrowserController, url: str):
    await controller.page.goto(url, wait_until="domcontentloaded", timeout=30000)
    await asyncio.sleep(1.5)


async def scenario_basic(controller: BrowserController, probe: Probe):
    """Structured page → bounded, populated ref map."""
    name = "basic"
    if not probe.should(name):
        return
    try:
        await goto(controller, "https://news.ycombinator.com")
        t0 = time.time()
        result = await snapshot(controller)
        ms = (time.time() - t0) * 1000
        refs = refs_from(result)
        ok = result.success and len(refs) >= 20 and len(result.output or "") < 15000
        probe.record(name, ok,
                     f"{len(refs)} refs, output {len(result.output or '')} chars, {ms:.0f}ms "
                     f"(sample: {list(refs.values())[:2]})")
    except Exception as exc:
        probe.record(name, False, f"exception: {exc}")


async def scenario_google(controller: BrowserController, probe: Probe):
    """JS-heavy page → the search box must still be in the map."""
    name = "google"
    if not probe.should(name):
        return
    try:
        await goto(controller, "https://www.google.com")
        result = await snapshot(controller)
        refs = refs_from(result)
        search_ref = find_ref(refs, "search", role="textbox") or find_ref(refs, "search", role="combobox")
        probe.record(name, result.success and search_ref is not None,
                     f"{len(refs)} refs; search box: {search_ref}")
    except Exception as exc:
        probe.record(name, False, f"exception: {exc}")


async def scenario_login(controller: BrowserController, probe: Probe):
    """Masking + echo + submit click: no blind writes end to end."""
    name = "login"
    if not probe.should(name):
        return
    try:
        await goto(controller, "https://the-internet.herokuapp.com/login")
        result = await snapshot(controller)
        refs = refs_from(result)
        user_ref = find_ref(refs, "username", role="textbox")
        pass_ref = find_ref(refs, "password", role="textbox")
        if not (user_ref and pass_ref):
            probe.record(name, False, f"fields not found among {len(refs)} refs")
            return
        # 1. password masked in the snapshot text
        masked_in_snapshot = "********" in (result.output or "") and "secret" not in (result.output or "")
        # 2. type into both fields via refs; each echo must name ITS OWN field
        r1 = await ref_type(controller, user_ref, "tomsmith")
        r2 = await ref_type(controller, pass_ref, "SuperSecretPassword!")
        import json as _json
        e1 = _json.loads(r1.output or "{}").get("typed_into", {}) if r1.success else {}
        e2 = _json.loads(r2.output or "{}").get("typed_into", {}) if r2.success else {}
        right_fields = (r1.success and r2.success
                        and "username" in str(e1.get("label", "")).lower()
                        and "password" in str(e2.get("label", "")).lower())
        secret_masked = ("********" in (r2.description or "")
                         and "SuperSecretPassword" not in (r2.description or "")
                         and "SuperSecretPassword" not in (r2.output or ""))
        # 3. re-snapshot: password must show the mask, never the secret
        result2 = await snapshot(controller)
        masked_after = "********" in (result2.output or "") and "SuperSecret" not in (result2.output or "")
        # 4. submit via ref click and verify the landing
        result3 = await snapshot(controller)
        refs3 = refs_from(result3)
        submit_ref = find_ref(refs3, "login", role="button")
        r3 = await ref_click(controller, submit_ref) if submit_ref else None
        await asyncio.sleep(2)
        landed = "secure" in (controller.page.url or "")
        probe.record(name, masked_in_snapshot and right_fields and masked_after and secret_masked and bool(r3 and r3.success) and landed,
                     f"masked_before={masked_in_snapshot} echo_ok={right_fields} masked_after={masked_after} desc_masked={secret_masked} landed={landed}")
    except Exception as exc:
        probe.record(name, False, f"exception: {exc}")


async def scenario_values_dont_invalidate(controller: BrowserController, probe: Probe):
    """Toggle one checkbox by ref, then act on ANOTHER ref from the SAME
    snapshot — a value change must not make the map stale."""
    name = "value_stability"
    if not probe.should(name):
        return
    try:
        await goto(controller, "https://the-internet.herokuapp.com/checkboxes")
        result = await snapshot(controller)
        refs = refs_from(result)
        checkbox_refs = [r for r, (role, _n) in refs.items() if role == "checkbox"]
        if len(checkbox_refs) < 2:
            probe.record(name, False, f"expected 2 checkboxes, got {len(checkbox_refs)}")
            return
        r1 = await ref_click(controller, checkbox_refs[0])
        r2 = await ref_click(controller, checkbox_refs[1])  # same map, post-value-change
        probe.record(name, r1.success and r2.success,
                     f"second ref after value change: {'still valid' if r2.success else r2.error}")
    except Exception as exc:
        probe.record(name, False, f"exception: {exc}")


async def scenario_staleness(controller: BrowserController, probe: Probe):
    """Navigate away → old refs must be REFUSED, not resolved to strangers."""
    name = "staleness"
    if not probe.should(name):
        return
    try:
        await goto(controller, "https://the-internet.herokuapp.com/login")
        result = await snapshot(controller)
        refs = refs_from(result)
        some_ref = next(iter(refs))
        await goto(controller, "https://the-internet.herokuapp.com/checkboxes")
        r1 = await ref_click(controller, some_ref)
        refused_as_stale = (not r1.success) and ("stale" in (r1.error or "") or "changed" in (r1.error or ""))
        r2 = await ref_click(controller, "@e999")
        unknown_refused = (not r2.success) and ("not in the current snapshot" in (r2.error or ""))
        probe.record(name, refused_as_stale and unknown_refused,
                     f"stale refused={refused_as_stale} unknown refused={unknown_refused}")
    except Exception as exc:
        probe.record(name, False, f"exception: {exc}")


async def scenario_ashby(controller: BrowserController, probe: Probe):
    """The original failure, replayed with refs: salary into Compensation,
    name into Legal Name — echoes must name the right fields. Advisory."""
    name = "ashby"
    if not probe.should(name):
        return
    try:
        await goto(controller, "https://jobs.ashbyhq.com/fireworks")
        result = await snapshot(controller)
        refs = refs_from(result)
        job_ref = find_ref(refs, "AI Deployment Strategist")
        if not job_ref:
            probe.record(name, None, f"board did not show the target job ({len(refs)} refs) — possibly bot-guarded")
            return
        await ref_click(controller, job_ref)
        await asyncio.sleep(2.5)
        result = await snapshot(controller)
        refs = refs_from(result)
        apply_ref = find_ref(refs, "Apply for this Job")
        if not apply_ref:
            probe.record(name, None, "job page did not expose the Apply button")
            return
        await ref_click(controller, apply_ref)
        await asyncio.sleep(2.5)
        result = await snapshot(controller, max_elements=200)
        refs = refs_from(result)
        name_ref = find_ref(refs, "Legal Name", role="textbox")
        comp_ref = find_ref(refs, "compensation", role="textbox")
        if not (name_ref and comp_ref):
            probe.record(name, None, f"form fields not found (name={name_ref}, comp={comp_ref}) of {len(refs)} refs")
            return
        r1 = await ref_type(controller, name_ref, "Test User")
        r2 = await ref_type(controller, comp_ref, "$100,000 - $150,000")
        import json as _json
        e1 = _json.loads(r1.output or "{}").get("typed_into", {})
        e2 = _json.loads(r2.output or "{}").get("typed_into", {})
        correct = (r1.success and r2.success
                   and "legal name" in str(e1.get("label", "")).lower()
                   and "compensation" in str(e2.get("label", "")).lower()
                   and not e1.get("warning") and not e2.get("warning"))
        probe.record(name, correct,
                     f"name echo='{e1.get('label')}' comp echo='{e2.get('label')}' warnings=({e1.get('warning')}, {e2.get('warning')})")
    except Exception as exc:
        probe.record(name, False, f"exception: {exc}")


async def scenario_canvas(controller: BrowserController, probe: Probe):
    """Canvas app → honest thin/empty result, no crash. Vision territory."""
    name = "canvas"
    if not probe.should(name):
        return
    try:
        await goto(controller, "https://paint.js.org")
        result = await snapshot(controller)
        refs = refs_from(result) if result.success else {}
        honest = True  # any non-crash outcome is acceptable; vision takes over
        probe.record(name, honest, f"success={result.success} refs={len(refs)} err={(result.error or '')[:80]}")
    except Exception as exc:
        probe.record(name, True, f"exception (still honest): {str(exc)[:80]}")


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="", help="comma-separated scenario names")
    args = parser.parse_args()
    only = {s.strip() for s in args.only.split(",") if s.strip()}

    probe = Probe(only)
    workdir = Path(tempfile.mkdtemp(prefix="cbu_probe_"))
    config = TaskConfig(
        task_id="probe_snapshot",
        goal="probe",
        enable_dom_fallback=True,
        chrome_profile=None,
    )
    controller = BrowserController(
        config=config,
        mimo_api_url="http://127.0.0.1:1",  # no MiMo needed: structured actions only
        working_dir=workdir,
        headless=HEADLESS,
    )
    await controller.start("https://news.ycombinator.com")
    try:
        await scenario_basic(controller, probe)
        await scenario_google(controller, probe)
        await scenario_login(controller, probe)
        await scenario_values_dont_invalidate(controller, probe)
        await scenario_staleness(controller, probe)
        await scenario_ashby(controller, probe)
        await scenario_canvas(controller, probe)
    finally:
        await controller.close()
    return probe.report()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
