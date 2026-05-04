"""Per-backend coverage for `count_tokens` / `acount_tokens`.

Each backend exposes a different `kind` Literal (see the README's
"Token counting" section). This file asserts:

  - the supported `kind` values return the right *type* (`int` for exact paths,
    `TokenCountEstimate` for offline-estimate paths)
  - unsupported `kind` values raise `ValueError`
  - `tools=` flows through and changes the count
  - `safety_factor` is reflected in `prompt_tokens` for estimate paths
  - the async sibling matches the sync result shape

Live-network paths on hosted backends (Anthropic / Google / OpenRouter
`'exact'` and `'online'`) are gated behind the `cost_integration` marker so
they aren't run on the default suite.
"""

import pytest

from toki import TokenCountEstimate

from .conftest import make_model, make_static_schema


SHORT_MESSAGES = [{"role": "user", "content": "Hello, how many tokens am I?"}]
LONG_MESSAGES = [
    {"role": "system", "content": "You are a careful assistant. " * 30},
    {"role": "user", "content": "Tell me a joke about token counting. " * 5},
    {"role": "assistant", "content": "Why did the LLM cross the road? " * 5},
    {"role": "user", "content": "I'm not sure, why?"},
]


# ----- LocalModel -----------------------------------------------------------

def test_local_count_tokens_exact():
    model = make_model("local", reasoning=False)
    n = model.count_tokens(SHORT_MESSAGES)
    assert isinstance(n, int)
    assert n > 0


def test_local_count_tokens_with_tools_increases_count():
    model = make_model("local", reasoning=False)
    base = model.count_tokens(SHORT_MESSAGES)
    with_tools = model.count_tokens(
        SHORT_MESSAGES,
        tools=[make_static_schema("record_value", "value")],
    )
    assert with_tools > base


def test_local_count_tokens_unsupported_kind():
    model = make_model("local", reasoning=False)
    with pytest.raises(ValueError):
        model.count_tokens(SHORT_MESSAGES, kind="offline")  # type: ignore[arg-type]


async def test_local_acount_tokens_exact():
    model = make_model("local", reasoning=False)
    n = await model.acount_tokens(SHORT_MESSAGES)
    assert isinstance(n, int)
    assert n > 0


# ----- OllamaModel ----------------------------------------------------------

def test_ollama_count_tokens_exact():
    model = make_model("ollama", reasoning=False)
    n = model.count_tokens(SHORT_MESSAGES)
    assert isinstance(n, int)
    assert n > 0


def test_ollama_count_tokens_with_tools_increases_count():
    model = make_model("ollama", reasoning=False)
    base = model.count_tokens(SHORT_MESSAGES)
    with_tools = model.count_tokens(
        SHORT_MESSAGES,
        tools=[make_static_schema("record_value", "value")],
    )
    assert with_tools > base


def test_ollama_count_tokens_unsupported_kind():
    model = make_model("ollama", reasoning=False)
    with pytest.raises(ValueError):
        model.count_tokens(SHORT_MESSAGES, kind="offline")  # type: ignore[arg-type]


async def test_ollama_acount_tokens_exact():
    model = make_model("ollama", reasoning=False)
    n = await model.acount_tokens(SHORT_MESSAGES)
    assert isinstance(n, int)
    assert n > 0


# ----- OpenAIModel ----------------------------------------------------------

def test_openai_count_tokens_exact():
    model = make_model("openai", reasoning=False)
    n = model.count_tokens(SHORT_MESSAGES)
    assert isinstance(n, int)
    assert n > 0


def test_openai_count_tokens_with_tools_increases_count():
    model = make_model("openai", reasoning=False)
    base = model.count_tokens(SHORT_MESSAGES)
    with_tools = model.count_tokens(
        SHORT_MESSAGES,
        tools=[make_static_schema("record_value", "value")],
    )
    assert with_tools > base


def test_openai_count_tokens_unsupported_kind():
    model = make_model("openai", reasoning=False)
    with pytest.raises(ValueError):
        model.count_tokens(SHORT_MESSAGES, kind="online")  # type: ignore[arg-type]


# ----- AnthropicModel -------------------------------------------------------

def test_anthropic_count_tokens_offline():
    model = make_model("anthropic", reasoning=False)
    result = model.count_tokens(SHORT_MESSAGES, kind="offline")
    assert isinstance(result, TokenCountEstimate)
    assert result.raw_prompt_tokens > 0
    assert result.safety_factor == pytest.approx(1.15)
    assert result.prompt_tokens == round(result.raw_prompt_tokens * result.safety_factor)


def test_anthropic_count_tokens_offline_safety_factor_override():
    model = make_model("anthropic", reasoning=False)
    a = model.count_tokens(SHORT_MESSAGES, kind="offline", safety_factor=1.0)
    b = model.count_tokens(SHORT_MESSAGES, kind="offline", safety_factor=2.0)
    assert isinstance(a, TokenCountEstimate) and isinstance(b, TokenCountEstimate)
    assert a.safety_factor == pytest.approx(1.0)
    assert b.safety_factor == pytest.approx(2.0)
    assert a.prompt_tokens == a.raw_prompt_tokens
    assert b.prompt_tokens == round(b.raw_prompt_tokens * 2.0)
    assert a.raw_prompt_tokens == b.raw_prompt_tokens


def test_anthropic_count_tokens_unsupported_kind():
    model = make_model("anthropic", reasoning=False)
    with pytest.raises(ValueError):
        model.count_tokens(SHORT_MESSAGES, kind="banana")  # type: ignore[arg-type]


@pytest.mark.cost_integration
def test_anthropic_count_tokens_exact_online():
    model = make_model("anthropic", reasoning=False)
    n = model.count_tokens(LONG_MESSAGES)
    assert isinstance(n, int)
    assert n > 0


@pytest.mark.cost_integration
async def test_anthropic_acount_tokens_exact_online():
    model = make_model("anthropic", reasoning=False)
    n = await model.acount_tokens(LONG_MESSAGES)
    assert isinstance(n, int)
    assert n > 0


# ----- GoogleModel ----------------------------------------------------------

def test_google_count_tokens_offline():
    model = make_model("google", reasoning=False)
    result = model.count_tokens(SHORT_MESSAGES, kind="offline")
    assert isinstance(result, TokenCountEstimate)
    assert result.raw_prompt_tokens > 0


def test_google_count_tokens_unsupported_kind():
    model = make_model("google", reasoning=False)
    with pytest.raises(ValueError):
        model.count_tokens(SHORT_MESSAGES, kind="banana")  # type: ignore[arg-type]


@pytest.mark.cost_integration
def test_google_count_tokens_exact_online():
    model = make_model("google", reasoning=False)
    n = model.count_tokens(LONG_MESSAGES)
    assert isinstance(n, int)
    assert n > 0


@pytest.mark.cost_integration
async def test_google_acount_tokens_exact_online():
    model = make_model("google", reasoning=False)
    n = await model.acount_tokens(LONG_MESSAGES)
    assert isinstance(n, int)
    assert n > 0


# ----- OpenRouterModel ------------------------------------------------------

def test_openrouter_count_tokens_offline():
    model = make_model("openrouter", reasoning=False)
    result = model.count_tokens(SHORT_MESSAGES, kind="offline")
    assert isinstance(result, TokenCountEstimate)
    assert result.raw_prompt_tokens > 0


def test_openrouter_count_tokens_offline_with_tools_increases_count():
    model = make_model("openrouter", reasoning=False)
    base = model.count_tokens(SHORT_MESSAGES, kind="offline")
    with_tools = model.count_tokens(
        SHORT_MESSAGES,
        kind="offline",
        tools=[make_static_schema("record_value", "value")],
    )
    assert isinstance(base, TokenCountEstimate) and isinstance(with_tools, TokenCountEstimate)
    assert with_tools.raw_prompt_tokens > base.raw_prompt_tokens


def test_openrouter_count_tokens_unsupported_kind():
    model = make_model("openrouter", reasoning=False)
    with pytest.raises(ValueError):
        model.count_tokens(SHORT_MESSAGES, kind="banana")  # type: ignore[arg-type]


@pytest.mark.cost_integration
def test_openrouter_count_tokens_exact_online():
    model = make_model("openrouter", reasoning=False)
    n = model.count_tokens(LONG_MESSAGES)
    assert isinstance(n, int)
    assert n > 0


@pytest.mark.cost_integration
async def test_openrouter_acount_tokens_exact_online():
    model = make_model("openrouter", reasoning=False)
    n = await model.acount_tokens(LONG_MESSAGES)
    assert isinstance(n, int)
    assert n > 0
