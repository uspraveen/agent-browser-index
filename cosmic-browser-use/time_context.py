"""Current clock context for browser decisions, matching COSMIC orchestrator prompts."""

from datetime import datetime, timezone
from zoneinfo import ZoneInfo


def current_time_context(user_timezone: str | None = None) -> str:
    """Build a fresh UTC clock line and optional user-local line for each decision."""
    now_utc = datetime.now(timezone.utc)
    date_line = f"Current date and time (UTC): {now_utc.strftime('%A, %B %d, %Y at %H:%M UTC')}."
    tz_name = (user_timezone or "").strip()
    if tz_name:
        try:
            local_now = now_utc.astimezone(ZoneInfo(tz_name))
            date_line += f"\nUser's local time: {local_now.strftime('%A, %B %d, %Y at %I:%M %p %Z')}."
        except Exception:
            pass
    return date_line
