"""User-facing CLI labels.

These helpers keep provider names consistent across terminal output, stats, and
docs without changing internal provider routing.
"""

from __future__ import annotations

import os
from typing import Any


FIREWORKS_KIMI_LABEL = "fireworks_kimi"
_KIMI_PROVIDER_ALIASES = {"fireworks", "kimi", "glm", "fireworks_glm", "fireworks_glm5p3"}

# Default brain model: GLM 5.3 Flash on Fireworks (1040k context, function calling).
FIREWORKS_DEFAULT_MODEL_ID = "accounts/fireworks/models/glm-5p3-flash"
# Opt-in alternative: Kimi K2.6 on Fireworks.
FIREWORKS_KIMI_MODEL_ID = "accounts/fireworks/models/kimi-k2p6"
# Escalation brain: Grok 4.6 on xAI (frontier model, vision-capable, OpenAI-compatible).
XAI_ESCALATION_MODEL_ID = "grok-4.6"
XAI_BASE_URL = "https://api.x.ai/v1"


def resolve_escalation_model() -> str:
    """Frontier/escalation model for the SLOW tier.

    Priority: ESCALATION_MODEL env -> XAI_ESCALATION_MODEL_ID.
    """
    return (os.getenv("ESCALATION_MODEL") or XAI_ESCALATION_MODEL_ID).strip().strip('"')


def resolve_fireworks_default_model() -> str:
    """Default Fireworks brain model.

    Priority: FIREWORKS_DEFAULT_MODEL env -> legacy FIREWORKS_KIMI_MODEL env
    (backwards compatibility) -> GLM 5.3 Flash.
    """
    return (
        os.getenv("FIREWORKS_DEFAULT_MODEL")
        or os.getenv("FIREWORKS_KIMI_MODEL")
        or FIREWORKS_DEFAULT_MODEL_ID
    ).strip().strip('"')


def normalize_cli_provider_arg(value: str) -> str:
    provider = (value or "").strip().lower()
    if provider in _KIMI_PROVIDER_ALIASES:
        return "fireworks_kimi"
    return provider


def cli_provider_help() -> str:
    return (
        "LLM provider to use: openai, anthropic, gemini, fireworks_kimi "
        f"(default; runs {FIREWORKS_DEFAULT_MODEL_ID} by default, Kimi K2.6 opt-in)."
    )


def cli_allowed_provider_labels() -> str:
    return "openai, anthropic, gemini, fireworks_kimi"


def display_provider_label(value: Any) -> str:
    provider = str(getattr(value, "value", value) or "").strip().lower()
    if provider in _KIMI_PROVIDER_ALIASES:
        return FIREWORKS_KIMI_LABEL
    return provider or "unknown"


def display_model_label(provider: Any, model: Any) -> str:
    model_text = str(model or "")
    return model_text


def display_provider_model(provider: Any, model: Any) -> str:
    provider_label = display_provider_label(provider)
    model_label = display_model_label(provider, model)
    if model_label == provider_label:
        return provider_label
    return f"{provider_label}:{model_label}"


def display_stat_value(key: str, value: Any, *, provider_hint: Any = None) -> Any:
    if key.endswith("provider"):
        return display_provider_label(value)
    if key.endswith("model"):
        return display_model_label(provider_hint, value)
    return value
