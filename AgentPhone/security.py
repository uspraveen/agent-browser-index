"""Webhook verification per https://docs.agentphone.ai/documentation/guides/webhooks"""

from __future__ import annotations

import hashlib
import hmac
import time


def verify_agentphone_webhook(raw_body: bytes, signature: str | None, timestamp: str | None, secret: str) -> bool:
    if not signature or not timestamp or not secret:
        return False
    try:
        ts = int(timestamp)
    except ValueError:
        return False
    if abs(time.time() - ts) > 300:
        return False
    signed_string = timestamp.encode("utf-8") + b"." + raw_body
    expected = hmac.new(secret.encode("utf-8"), signed_string, hashlib.sha256).hexdigest()
    return hmac.compare_digest(signature, f"sha256={expected}")
