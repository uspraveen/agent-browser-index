"""The deterministic CAPTCHA-wall cap.

The model's rule is "never more than three attempts on the same challenge
without progress"; this counter is the harness-side teeth. It counts
consecutive decision steps parked on the same challenge URL — Google's /sorry/
or an explicit captcha path — and flips to exhausted so the prompt cuts
attempts off. Any non-wall URL resets it completely: browsing away is progress.

Run with:  python -m pytest tests/test_captcha_wall.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import _captcha_wall_url, update_captcha_wall  # noqa: E402

GOOGLE_SORRY = "https://www.google.com/sorry/index?continue=x"
CAPTCHA_PATH = "https://example.com/captcha?next=/dash"


def test_known_challenge_signatures():
    assert _captcha_wall_url(GOOGLE_SORRY)
    assert _captcha_wall_url(CAPTCHA_PATH)
    assert not _captcha_wall_url("https://www.google.com/search?q=cats")
    # A normal page whose slug merely mentions the word is not a wall.
    assert not _captcha_wall_url("https://example.com/blog/our-challenge-2026")
    assert not _captcha_wall_url("")
    assert not _captcha_wall_url(None)


def test_allows_exactly_three_steps_then_exhausts():
    url, steps, exhausted = "", 0, False
    for i in range(1, 6):
        url, steps, exhausted = update_captcha_wall(GOOGLE_SORRY, url, steps)
        if i <= 3:
            assert not exhausted, f"attempt {i} must still be allowed"
        else:
            assert exhausted, f"attempt {i} must be cut off"


def test_browsing_away_resets_the_tracker():
    url, steps, _ = update_captcha_wall(GOOGLE_SORRY, "", 0)
    url, steps, _ = update_captcha_wall(GOOGLE_SORRY, url, steps)
    url, steps, _ = update_captcha_wall(GOOGLE_SORRY, url, steps)
    assert steps == 2
    url, steps, exhausted = update_captcha_wall("https://wellfound.com/role/x", url, steps)
    assert (url, steps, exhausted) == ("", 0, False)


def test_returning_to_a_wall_starts_a_fresh_count():
    url, steps, _ = update_captcha_wall(GOOGLE_SORRY, "", 0)
    url, steps, _ = update_captcha_wall(GOOGLE_SORRY, url, steps)
    url, steps, _ = update_captcha_wall("https://wellfound.com/role/x", url, steps)
    url, steps, exhausted = update_captcha_wall(GOOGLE_SORRY, url, steps)
    assert steps == 0
    assert not exhausted


def test_a_different_wall_is_its_own_count():
    url, steps, _ = update_captcha_wall(GOOGLE_SORRY, "", 0)
    url, steps, _ = update_captcha_wall(GOOGLE_SORRY, url, steps)
    url, steps, exhausted = update_captcha_wall(CAPTCHA_PATH, url, steps)
    assert url == CAPTCHA_PATH
    assert steps == 0
    assert not exhausted
