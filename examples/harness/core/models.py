"""
Model factory: turn a `provider:name` spec into a toki model.

    openrouter:anthropic/claude-sonnet-4.5   (default provider if omitted)
    anthropic:claude-sonnet-4-5
    openai:gpt-5.4-mini
    openai-responses:gpt-5.4-mini
    google:gemini-2.5-flash
    ollama:qwen3:8b
    bedrock:global.anthropic.claude-sonnet-4-6
    demo                                     (scripted, no API key needed)

API keys come from the usual environment variables (`OPENROUTER_API_KEY`,
`ANTHROPIC_API_KEY`, ...). The spec can also be set with `HARNESS_MODEL`.
"""

from __future__ import annotations

import os

from toki import BaseModel, ReasoningEffort

DEFAULT_MODEL = "openrouter:anthropic/claude-sonnet-4.5"
PROVIDERS = ["openrouter", "anthropic", "openai", "openai-responses", "google", "ollama", "bedrock", "demo"]


def default_spec() -> str:
    return os.environ.get("HARNESS_MODEL", DEFAULT_MODEL)


def parse_spec(spec: str) -> tuple[str, str]:
    spec = spec.strip()
    if spec == "demo":
        return "demo", "demo"
    provider, sep, name = spec.partition(":")
    if not sep or provider not in PROVIDERS:
        # bare OpenRouter id such as `anthropic/claude-sonnet-4.5` or `ollama:qwen3:8b` style leftovers
        return "openrouter", spec
    return provider, name


def make_model(spec: str, *, reasoning_effort: ReasoningEffort | None = None) -> BaseModel:
    provider, name = parse_spec(spec)
    effort = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}

    if provider == "demo":
        from ..testing import demo_model
        return demo_model()
    if provider == "openrouter":
        from toki import OpenRouterModel, get_openrouter_api_key
        return OpenRouterModel(name, api_key=get_openrouter_api_key(), cache="rolling", **effort)  # type: ignore[arg-type]
    if provider == "anthropic":
        from toki import AnthropicModel, get_anthropic_api_key
        return AnthropicModel(name, api_key=get_anthropic_api_key(), cache="rolling", **effort)
    if provider == "openai":
        from toki import OpenAIModel, get_openai_api_key
        return OpenAIModel(name, api_key=get_openai_api_key(), **effort)
    if provider == "openai-responses":
        from toki import OpenAIResponsesModel, get_openai_api_key
        return OpenAIResponsesModel(name, api_key=get_openai_api_key(), **effort)
    if provider == "google":
        from toki import GoogleModel, get_google_api_key
        return GoogleModel(name, api_key=get_google_api_key(), **effort)
    if provider == "ollama":
        from toki import OllamaModel
        return OllamaModel(name)
    if provider == "bedrock":
        from toki import BedrockModel
        return BedrockModel(name, region_name=os.environ.get("AWS_REGION"), **effort)
    raise ValueError(f"unknown provider {provider!r}; expected one of {', '.join(PROVIDERS)}")


def describe(model: BaseModel) -> str:
    name = getattr(model, "model", None) or type(model).__name__
    provider = type(model).__name__.removesuffix("Model").lower() or "model"
    return f"{provider}:{name}"
