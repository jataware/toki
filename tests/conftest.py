"""Shared fixtures and helpers for the toki test suite.

The `MODELS` table at the top is the single editable source of truth for which
models each provider uses. `default` is a tools-capable model; `reasoning` is
additionally a reasoning-capable model used whenever a test sets
`capture_thinking=True`. Setting `reasoning=None` causes those tests to be
skipped (rather than failing) for that provider.
"""

import importlib
import os
from datetime import datetime
from pathlib import Path

from toki import (
    StreamingToolSchema,
    TokiArgStream,
    TokiThinking,
    TokiToolCall,
    TokiToolCallStream,
    ToolSchema,
)

import pytest


here = Path(__file__).parent


def pytest_configure(config: pytest.Config) -> None:
    """Auto-emit a timestamped HTML report under `tests/results/` when the user
    runs plain `pytest`. If they pass `--html=...` explicitly, respect it."""
    if config.getoption("htmlpath", None):
        return
    results_dir = here / "results"
    results_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    config.option.htmlpath = str(results_dir / f"{stamp}.html")



# Pick smallest/cheapest models that satisfy each test profile. Verified against
# each provider's generated `models.py` snapshot. Qwen3 natively supports tool
# calls and reasoning via its chat template, so the same Qwen entry serves as
# both default and reasoning for the local backend.
#
# OpenAI's `reasoning` is intentionally `None`: even via the Responses API
# bridge with `reasoning_summary='detailed'`, gpt-5.4-nano/mini emit
# `reasoning_content` only sporadically (especially when a tool call is the
# response). Server-side reasoning still works (`OpenAIModel(...,
# reasoning_effort=...)` is exposed), but we can't reliably assert on captured
# thoughts so the corresponding `capture_thinking=True` tests are skipped.
MODELS: dict[str, dict[str, str | None]] = {
    "openrouter": {"default": "anthropic/claude-haiku-4-5",   "reasoning": "anthropic/claude-sonnet-4-5"},
    "openai":     {"default": "gpt-5.4-nano",                 "reasoning": None},
    "anthropic":  {"default": "claude-haiku-4-5",             "reasoning": "claude-sonnet-4-5"},
    "google":     {"default": "gemini-2.5-flash",             "reasoning": "gemini-2.5-flash"},
    "local":      {"default": "Qwen/Qwen3-1.7B",              "reasoning": "Qwen/Qwen3-1.7B"},
}


SENTINEL_VALUE = "apple"
SENTINEL_TEXT = "banana"


# Mapping from provider name to (env var, top-level toki class symbol) for the
# four hosted providers. `local` is handled separately because it needs no key.
_HOSTED_PROVIDER_CONFIG: dict[str, tuple[str, str]] = {
    "openrouter": ("OPENROUTER_API_KEY", "OpenRouterModel"),
    "openai":     ("OPENAI_API_KEY",     "OpenAIModel"),
    "anthropic":  ("ANTHROPIC_API_KEY",  "AnthropicModel"),
    "google":     ("GEMINI_API_KEY",     "GoogleModel"),
}


# litellm-backed backends accept a `reasoning_effort` init param. The
# OpenRouter and local backends drive reasoning through `capture_thinking` alone,
# so we don't forward it to them.
_REASONING_EFFORT_PROVIDERS: set[str] = {"openai", "anthropic", "google"}


def make_model(provider: str, *, reasoning: bool):
    """Construct the configured backend for `provider`.

    Hard-fails (not skips) when the required env var is missing — per the
    project's testing philosophy, a missing key is a configuration error not a
    silent gap. Skips cleanly when no model is configured for the requested
    profile (e.g. a provider with no reasoning model). When `reasoning=True`
    and the provider's frontend accepts a `reasoning_effort` init param, the
    model is constructed with `reasoning_effort='medium'`.
    """
    name = MODELS[provider]["reasoning" if reasoning else "default"]
    if name is None:
        pytest.skip(f"{provider!r} has no configured {'reasoning' if reasoning else 'default'} model")

    if provider == "local":
        toki_mod = importlib.import_module("toki")
        LocalModel = toki_mod.LocalModel
        return LocalModel(name, allow_parallel_tool_calls=True)

    env_var, ctor_name = _HOSTED_PROVIDER_CONFIG[provider]
    if not os.getenv(env_var):
        pytest.fail(f"{env_var} is not set; required to run {provider} tests")

    Cls = getattr(importlib.import_module("toki"), ctor_name)

    if provider == "openrouter":
        return Cls(name, os.environ[env_var], allow_parallel_tool_calls=True)

    kwargs: dict = {"api_key": os.environ[env_var], "allow_parallel_tool_calls": True}
    if reasoning and provider in _REASONING_EFFORT_PROVIDERS:
        kwargs["reasoning_effort"] = "medium"
    return Cls(name, **kwargs)


# ----- tool schema helpers --------------------------------------------------

def make_static_schema(name: str, arg_name: str, description: str = "Record a value for testing.") -> ToolSchema:
    """Build a single-string-arg static tool schema."""
    return ToolSchema(schema={
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    arg_name: {"type": "string", "description": f"The {arg_name} to record."},
                },
                "required": [arg_name],
                "additionalProperties": False,
            },
        },
    })


def make_streaming_schema(name: str, arg_name: str, description: str = "Record a value for testing.") -> StreamingToolSchema:
    """Build a single-string-arg streaming tool schema."""
    return StreamingToolSchema(schema={
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    arg_name: {"type": "string", "description": f"The {arg_name} to record."},
                },
                "required": [arg_name],
                "additionalProperties": False,
            },
        },
    })


# ----- stream draining ------------------------------------------------------

def drain_stream(generator, *, expected_arg_names: dict[str, str] | None = None) -> dict[str, list]:
    """Walk a `complete(stream=True)` generator and bucket its yielded items.

    `expected_arg_names` maps tool name -> the name of the single argument that
    streaming tool exposes (e.g. {"record_streaming": "text"}). Each
    `TokiToolCallStream` encountered is consumed via `expect_arg(...)` so its
    `.arguments` is fully populated by the time this returns.
    """
    expected_arg_names = expected_arg_names or {}
    out: dict[str, list] = {
        "strings": [],
        "thoughts": [],
        "tool_calls": [],
        "tool_streams": [],
    }
    for item in generator:
        if isinstance(item, str):
            out["strings"].append(item)
        elif isinstance(item, TokiThinking):
            out["thoughts"].append(item)
        elif isinstance(item, TokiToolCall):
            out["tool_calls"].append(item)
        elif isinstance(item, TokiToolCallStream):
            arg_name = expected_arg_names.get(item.name)
            if arg_name is not None:
                arg_stream: TokiArgStream = item.expect_arg(arg_name)
                # consume to completion — value will be cached on the parent
                _ = arg_stream.value
            # final drain to ensure `.arguments` is materialized
            _ = item.arguments
            out["tool_streams"].append(item)
        else:
            raise AssertionError(f"unexpected item from stream: {type(item).__name__}: {item!r}")
    return out
