"""User-facing CLI labels.

These helpers intentionally do not change internal provider names, env vars,
model IDs, stored memory, or routing. They only mask demo-facing labels printed
to the terminal.
"""

from __future__ import annotations

from typing import Any


GOOGLE_GEMINI_LABEL = "google gemini"
GOOGLE_GEMINI_PROVIDER_ARG = "google_gemini"
_KIMI_INTERNAL_PROVIDER_VALUES = {"fireworks", "fireworks_kimi", "kimi"}
_GOOGLE_GEMINI_ALIASES = {
    GOOGLE_GEMINI_PROVIDER_ARG,
    "google-gemini",
    "google gemini",
}


def normalize_cli_provider_arg(value: str) -> str:
    provider = (value or "").strip().lower()
    if provider in _GOOGLE_GEMINI_ALIASES:
        return "fireworks_kimi"
    return provider


def cli_provider_help() -> str:
    return "LLM provider to use: openai, anthropic, gemini, google_gemini."


def cli_allowed_provider_labels() -> str:
    return "openai, anthropic, gemini, google_gemini"


def display_provider_label(value: Any) -> str:
    provider = str(getattr(value, "value", value) or "").strip().lower()
    if provider in _KIMI_INTERNAL_PROVIDER_VALUES:
        return GOOGLE_GEMINI_LABEL
    return provider or "unknown"


def display_model_label(provider: Any, model: Any) -> str:
    provider_text = str(getattr(provider, "value", provider) or "").strip().lower()
    model_text = str(model or "")
    model_lower = model_text.lower()
    if provider_text in _KIMI_INTERNAL_PROVIDER_VALUES or "fireworks" in model_lower or "kimi" in model_lower:
        return GOOGLE_GEMINI_LABEL
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
