"""Base-brain providers for the "claude" model set.

Claude Haiku 5.5 (official SDK) as the base brain with GPT-6 Luna as its
same-tier fallback and grok-4.7 as escalation. These pin the request shapes
the live APIs require — Haiku 5.5 400s on non-default temperature, thinking
budgets and prefill; GPT-5/6-family models 400 on temperature/max_tokens —
and that a fallback is reported against the model that actually ran.

Run with:  python -m pytest tests/test_model_providers.py -q
"""
from __future__ import annotations

import asyncio
import base64
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cosmic_types import LLMConfig, LLMProvider, LLMTier  # noqa: E402
from orchestrator import (  # noqa: E402
    ClaudeProvider,
    FallbackProvider,
    ModelRefusalError,
    OpenAIProvider,
    Orchestrator,
    sniff_image_media_type,
)

WEBP_B64 = "UklGRiQAAABXRUJQVlA4IBgAAAAwAQCdASoBAAEAAwA0JaQAA3AA/vuUAAA="
PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="


class _FakeMessages:
    def __init__(self, response):
        self.response = response
        self.calls: list = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _claude_response(text="{\"action_type\": \"Wait\"}", stop_reason="end_turn", usage=(1000, 50)):
    return SimpleNamespace(
        content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        stop_details=SimpleNamespace(category="cyber") if stop_reason == "refusal" else None,
        usage=SimpleNamespace(input_tokens=usage[0], output_tokens=usage[1],
                              cache_read_input_tokens=0, cache_creation_input_tokens=0),
    )


def _claude(monkeypatch, response=None, **env):
    for key in ("ANTHROPIC_WORKSPACE_ID", "BROWSER_AGENT_CLAUDE_EFFORT", "BROWSER_AGENT_CLAUDE_THINKING"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    provider = ClaudeProvider(LLMConfig(
        provider=LLMProvider.CLAUDE, model_id="claude-haiku-5-5", api_key="k",
        timeout_ms=30000, max_tokens=8192, temperature=0.3,
    ))
    fake = _FakeMessages(response if response is not None else _claude_response())
    provider.sdk = SimpleNamespace(messages=fake, close=lambda: asyncio.sleep(0))
    return provider, fake


def _messages():
    return [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{WEBP_B64}"}},
        {"type": "text", "text": "next action?"},
    ]}]


def test_media_type_comes_from_the_bytes():
    assert sniff_image_media_type(WEBP_B64, "image/jpeg") == "image/webp"
    assert sniff_image_media_type(PNG_B64, "image/jpeg") == "image/png"
    assert sniff_image_media_type("!!!", "image/gif") == "image/gif"


def test_haiku_request_shape(monkeypatch):
    provider, fake = _claude(monkeypatch)
    result = asyncio.run(provider.generate(messages=_messages(), system_prompt="sys"))
    assert result["content"] == "{\"action_type\": \"Wait\"}"
    request = fake.calls[0]
    assert request["model"] == "claude-haiku-5-5"
    assert "temperature" not in request and "thinking" not in request
    assert request["output_config"] == {"effort": "low"}
    assert request["system"] == "sys" and request["max_tokens"] == 8192
    image = request["messages"][0]["content"][0]
    assert image["type"] == "image" and image["source"]["media_type"] == "image/webp"
    assert provider.usage_totals == {"requests": 1, "prompt_tokens": 1000, "completion_tokens": 50, "total_tokens": 1050}


def test_effort_and_thinking_are_configurable(monkeypatch):
    provider, fake = _claude(monkeypatch, BROWSER_AGENT_CLAUDE_EFFORT="medium", BROWSER_AGENT_CLAUDE_THINKING="disabled")
    asyncio.run(provider.generate(messages=_messages(), json_mode=True))
    request = fake.calls[0]
    assert request["output_config"] == {"effort": "medium"}
    assert request["thinking"] == {"type": "disabled"}


def test_workspace_id_becomes_a_header(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_test")
    provider = ClaudeProvider(LLMConfig(provider=LLMProvider.CLAUDE, model_id="claude-haiku-5-5", api_key="k"))
    headers = getattr(provider.sdk, "_custom_headers", None) or getattr(provider.sdk, "default_headers", {})
    assert headers.get("anthropic-workspace-id") == "wrkspc_test"


def test_refusal_raises_so_the_fallback_runs(monkeypatch):
    provider, _ = _claude(monkeypatch, response=_claude_response(stop_reason="refusal"))
    with pytest.raises(ModelRefusalError):
        asyncio.run(provider.generate(messages=_messages()))


def _openai(handler, model="gpt-6-luna"):
    provider = OpenAIProvider(LLMConfig(provider=LLMProvider.OPENAI, model_id=model, api_key="k", max_tokens=8192))
    provider.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return provider


def _openai_ok(bodies):
    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "{\"action_type\": \"Wait\"}"}}],
            "usage": {"prompt_tokens": 900, "completion_tokens": 20, "total_tokens": 920},
        })
    return handler


def test_gpt6_payload_has_no_temperature_and_caps_completion_tokens(monkeypatch):
    monkeypatch.delenv("BROWSER_AGENT_OPENAI_REASONING_EFFORT", raising=False)
    bodies: list = []
    asyncio.run(_openai(_openai_ok(bodies)).generate(messages=_messages(), system_prompt="sys", json_mode=True))
    body = bodies[0]
    assert "temperature" not in body and "max_tokens" not in body
    assert body["max_completion_tokens"] == 8192 and body["reasoning_effort"] == "low"
    assert body["response_format"] == {"type": "json_object"}
    assert body["messages"][0] == {"role": "system", "content": "sys"}


def test_older_openai_models_keep_their_payload():
    bodies: list = []
    asyncio.run(_openai(_openai_ok(bodies), model="gpt-4o").generate(messages=_messages()))
    assert bodies[0]["temperature"] == 0.3 and bodies[0]["max_tokens"] == 8192


def test_fallback_serves_the_step_and_usage_stays_separate(monkeypatch):
    primary, _ = _claude(monkeypatch, response=RuntimeError("anthropic 529 overloaded"))
    bodies: list = []
    fallback = _openai(_openai_ok(bodies))
    wrapped = FallbackProvider(primary, fallback)
    result = asyncio.run(wrapped.generate(messages=_messages(), system_prompt="sys", json_mode=True))
    assert result["content"] == "{\"action_type\": \"Wait\"}"
    assert wrapped.fallback_count == 1 and "overloaded" in wrapped.last_error
    assert wrapped.usage_totals["requests"] == 0  # primary billed nothing
    assert fallback.usage_totals["requests"] == 1


def test_both_failing_propagates_to_the_frontier_failover(monkeypatch):
    primary, _ = _claude(monkeypatch, response=RuntimeError("down"))

    def broken(_request):
        return httpx.Response(500, json={"error": {"message": "also down"}})

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(FallbackProvider(primary, _openai(broken)).generate(messages=_messages()))


def test_orchestrator_builds_the_fallback_and_reports_it(monkeypatch):
    for key in ("ANTHROPIC_WORKSPACE_ID",):
        monkeypatch.delenv(key, raising=False)
    fast = LLMConfig(
        provider=LLMProvider.CLAUDE, model_id="claude-haiku-5-5", api_key="k", tier=LLMTier.FAST,
        fallback=LLMConfig(provider=LLMProvider.OPENAI, model_id="gpt-6-luna", api_key="k", tier=LLMTier.FAST),
    )
    slow = LLMConfig(provider=LLMProvider.XAI, model_id="grok-4.7", api_key="x", tier=LLMTier.SLOW,
                     api_base="https://api.x.ai/v1")
    orchestrator = Orchestrator(fast_model=fast, slow_model=slow)
    assert isinstance(orchestrator.models[LLMTier.FAST], FallbackProvider)
    stats = orchestrator.get_stats()
    assert stats["llm_usage"]["base"]["provider"] == "claude"
    assert stats["llm_usage"]["base"]["model"] == "claude-haiku-5-5"
    assert stats["llm_usage"]["base_fallback"]["model"] == "gpt-6-luna"
    assert stats["llm_usage"]["frontier"]["model"] == "grok-4.7"
    assert stats["base_fallbacks"]["count"] == 0


def test_without_a_fallback_nothing_extra_is_reported():
    fast = LLMConfig(provider=LLMProvider.OPENAI, model_id="gpt-6-luna", api_key="k", tier=LLMTier.FAST)
    stats = Orchestrator(fast_model=fast).get_stats()
    assert "base_fallback" not in stats["llm_usage"] and "base_fallbacks" not in stats


# --- model set resolution ---------------------------------------------------

def _resolve(monkeypatch, **env):
    for key in ("BROWSER_AGENT_MODEL_SET", "ANTHROPIC_API_KEY", "BROWSER_AGENT_ANTHROPIC_API_KEY", "OPENAI_API_KEY",
                "BROWSER_AGENT_OPENAI_API_KEY", "BROWSER_USE_API_KEY", "XAI_API_KEY", "XAI_MODEL", "ESCALATION_MODEL",
                "FIREWORKS_API_KEY", "SLIDE_AGENT_FIREWORKS_API_KEY", "BROWSER_AGENT_ESCALATION",
                "BROWSER_AGENT_CLAUDE_MODEL", "BROWSER_AGENT_FALLBACK_MODEL"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    from cosmic_browser_use.api import _resolve_model_configs
    return _resolve_model_configs()


def test_claude_set_is_haiku_with_luna_fallback_and_grok_47(monkeypatch):
    fast, medium, slow = _resolve(monkeypatch, BROWSER_AGENT_MODEL_SET="claude", ANTHROPIC_API_KEY="a",
                                  OPENAI_API_KEY="o", XAI_API_KEY="x")
    assert (fast.provider, fast.model_id) == (LLMProvider.CLAUDE, "claude-haiku-5-5")
    assert fast.max_tokens == 8192
    assert (fast.fallback.provider, fast.fallback.model_id) == (LLMProvider.OPENAI, "gpt-6-luna")
    assert medium is None
    assert (slow.provider, slow.model_id) == (LLMProvider.XAI, "grok-4.7")


def test_claude_set_without_openai_key_has_no_fallback(monkeypatch):
    fast, _, _ = _resolve(monkeypatch, BROWSER_AGENT_MODEL_SET="claude", ANTHROPIC_API_KEY="a", XAI_API_KEY="x")
    assert fast.fallback is None


def test_claude_set_without_anthropic_key_degrades_to_bu(monkeypatch):
    fast, _, slow = _resolve(monkeypatch, BROWSER_AGENT_MODEL_SET="claude", BROWSER_USE_API_KEY="b", XAI_API_KEY="x")
    assert fast.provider == LLMProvider.BROWSER_USE
    assert slow.model_id == "grok-4.7"


def test_bu_set_escalates_to_grok_47_by_default(monkeypatch):
    _, _, slow = _resolve(monkeypatch, BROWSER_USE_API_KEY="b", XAI_API_KEY="x")
    assert slow.model_id == "grok-4.7"
