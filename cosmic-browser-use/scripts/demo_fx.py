#!/usr/bin/env python3
"""Watchable demo of the blue action-FX overlay (cursor glide, click pulse,
typing pill, scroll chevrons, key badge).

Launches the agent's persistent Chrome for the given profile (your own Chrome
windows are never touched), then slowly performs a scripted sequence of real
agent actions on a local demo page so you can watch every effect.

Usage:
    python scripts/demo_fx.py --chrome-profile "Default"
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from cosmic_types import TaskConfig
from browser_controller import BrowserController

DEMO_HTML = """
<!doctype html><html><head><title>Cosmic FX demo</title></head>
<body style="margin:0;font-family:system-ui;background:#f6f8fb;height:2400px">
  <div style="padding:48px;max-width:720px">
    <h1 style="color:#1e3a8a">Cosmic FX demo page</h1>
    <p>Watch the blue cursor glide, pulse on clicks, and announce typing,
       scrolling and key presses.</p>
    <input id="field" style="font-size:16px;padding:10px;width:340px;margin-top:16px"
           placeholder="the agent will type here">
    <div style="margin-top:240px">
      <button id="btn" style="padding:12px 22px;font-size:15px">A button to click</button>
    </div>
  </div>
</body></html>
"""


async def run(profile: str) -> None:
    ctrl = BrowserController(
        config=TaskConfig(task_id="fx_demo", goal="fx demo", chrome_profile=profile),
        mimo_api_url="https://example.invalid/never-called",  # required by ctor, never used here
        working_dir=ROOT / "runs" / "fx_demo",
    )
    await ctrl.start()
    page = ctrl.page
    await page.goto("data:text/html," + urllib.parse.quote(DEMO_HTML))
    cur = ctrl.cursor_overlay
    print("Overlay enabled:", cur.enabled, "(set SHOW_CURSOR_OVERLAY=false to disable)")

    print("1) cursor glide + click pulses...")
    await cur.show_move(page, 200, 160)
    await asyncio.sleep(1.0)
    await cur.show_click(page, 900, 300)
    await asyncio.sleep(1.0)
    await cur.show_click(page, 300, 520)
    await asyncio.sleep(1.2)

    print("2) real type action into the input (typing pill + Enter badge)...")
    await ctrl._dom_type("#field", "hello from the cosmic agent", press_enter=True)
    await asyncio.sleep(1.2)

    print("3) real scroll actions (chevrons at the right edge)...")
    await ctrl._visual_scroll("down", "medium")
    await asyncio.sleep(1.0)
    await ctrl._visual_scroll("up", "medium")
    await asyncio.sleep(1.0)

    print("4) key press badge...")
    await ctrl._press_key("Escape")
    await asyncio.sleep(1.2)

    print("5) screenshot-safety check: capturing via the MiMo chokepoint —")
    await cur.show_click(page, 640, 360)
    data = await ctrl._safe_page_screenshot(type="jpeg", quality=60)
    print(f"   captured {len(data)} bytes with all effects hidden, overlay restored after.")

    print("\nDemo done — closing in 3s.")
    await asyncio.sleep(3)
    await ctrl.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Watchable demo of the agent's blue action effects.")
    parser.add_argument("--chrome-profile", default="Default", metavar="PROFILE_DIR",
                        help="Chrome profile directory name (default: Default)")
    args = parser.parse_args()
    asyncio.run(run(args.chrome_profile))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
