"""User-facing CLI labels.

These helpers keep provider names consistent across terminal output, stats, and
docs without changing internal provider routing.
"""

from __future__ import annotations

from typing import Any


FIREWORKS_KIMI_LABEL = "fireworks_kimi"
_KIMI_PROVIDER_ALIASES = {"fireworks", "kimi"}


def normalize_cli_provider_arg(value: str) -> str:
    provider = (value or "").strip().lower()
    if provider in _KIMI_PROVIDER_ALIASES:
        return "fireworks_kimi"
    return provider


def cli_provider_help() -> str:
    return "LLM provider to use: openai, anthropic, gemini, fireworks_kimi."


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
