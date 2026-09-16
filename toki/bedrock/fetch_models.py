"""Developer utility for refreshing Toki's public Bedrock model snapshot."""

import json
from os import PathLike
from pathlib import Path
from urllib.request import urlopen

from .models import Attr, normalize_bedrock_model_id

here = Path(__file__).parent
_MODELS_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
)


def _model_id(key: str) -> str:
    for prefix in ("bedrock/converse/", "bedrock/"):
        if key.startswith(prefix):
            return key.removeprefix(prefix)
    return key


def _optional_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _reasoning_family(model_id: str, info: dict):
    if info.get("supports_adaptive_thinking") is True:
        return "claude_adaptive"
    if info.get("supports_legacy_thinking") is True:
        return "claude_budget"
    normalized = normalize_bedrock_model_id(model_id)
    if normalized.startswith("amazon.nova") and info.get("supports_reasoning") is True:
        return "nova"
    if (
        normalized.startswith("openai.")
        and info.get("supports_max_reasoning_effort") is True
    ):
        return "openai"
    return None


def _attributes(model_id: str, info: dict) -> Attr:
    context_size = info.get("max_input_tokens") or info.get("max_tokens")
    if not isinstance(context_size, int):
        context_size = None

    supports_tools = _optional_bool(info.get("supports_function_calling"))
    supports_thinking = _optional_bool(info.get("supports_reasoning"))
    normalized = normalize_bedrock_model_id(model_id)
    if supports_thinking is None and normalized.startswith("amazon.nova-"):
        supports_thinking = normalized.startswith("amazon.nova-2-")
    if supports_thinking is None and "anthropic.claude-3-5-" in normalized:
        supports_thinking = False

    public_caching = _optional_bool(info.get("supports_prompt_caching"))
    supports_caching = public_caching is True
    cache_min_tokens = info.get("prompt_cache_min_tokens")
    if not isinstance(cache_min_tokens, int):
        cache_min_tokens = 1_000 if normalized.startswith("amazon.nova") else 1_024
        if not supports_caching:
            cache_min_tokens = None

    if supports_caching:
        cache_ttls = (
            ("5m", "1h")
            if info.get("cache_creation_input_token_cost_above_1hr") is not None
            else ("5m",)
        )
        if normalized.startswith("amazon.nova"):
            cache_locations = ("system", "messages")
        elif normalized.startswith("anthropic."):
            cache_locations = ("system", "messages", "tools")
        else:
            cache_locations = ("messages",)
    else:
        cache_ttls = ()
        cache_locations = ()

    return Attr(
        context_size=context_size,
        supports_tools=supports_tools,
        supports_thinking=supports_thinking,
        supports_streaming=True,
        supports_explicit_caching=supports_caching,
        cache_min_tokens=cache_min_tokens,
        cache_ttls=cache_ttls,
        cache_locations=cache_locations,
        reasoning_family=_reasoning_family(model_id, info),
    )


def _fetch_models() -> dict[str, Attr]:
    with urlopen(_MODELS_URL) as response:
        metadata = json.load(response)
    models: dict[str, Attr] = {}
    for key, info in metadata.items():
        if not isinstance(info, dict):
            continue
        if info.get("litellm_provider") != "bedrock_converse":
            continue
        if info.get("mode") != "chat":
            continue
        model_id = _model_id(key)
        if "@" in model_id:
            continue
        models[model_id] = _attributes(model_id, info)
    return dict(sorted(models.items()))


def _format_attributes(model_id: str, attributes: Attr) -> str:
    def string(value: str | None) -> str:
        return "None" if value is None else json.dumps(value)

    def string_tuple(values: tuple[str, ...]) -> str:
        if not values:
            return "()"
        items = ", ".join(json.dumps(value) for value in values)
        if len(values) == 1:
            items += ","
        return f"({items})"

    return f'''    "{model_id}": Attr(
        context_size={attributes.context_size!r},
        supports_tools={attributes.supports_tools!r},
        supports_thinking={attributes.supports_thinking!r},
        supports_streaming={attributes.supports_streaming!r},
        supports_explicit_caching={attributes.supports_explicit_caching!r},
        cache_min_tokens={attributes.cache_min_tokens!r},
        cache_ttls={string_tuple(attributes.cache_ttls)},
        cache_locations={string_tuple(attributes.cache_locations)},
        reasoning_family={string(attributes.reasoning_family)},
    ),'''


def _create_models_types_file(file: PathLike = here / "models.py") -> None:
    """Regenerate the Bedrock snapshot from LiteLLM's public metadata."""
    file = Path(file)
    models = _fetch_models()
    ids = list(models)
    literal_lines = "\n".join(f'    "{model_id}",' for model_id in ids)
    attr_lines = "\n".join(
        _format_attributes(model_id, attributes)
        for model_id, attributes in models.items()
    )
    print(f"Writing {file} with {len(models)} models")
    file.write_text(
        f'''# DO NOT EDIT THIS FILE MANUALLY
# This file is generated via the `toki-fetch-bedrock-models` script.

from dataclasses import dataclass
from typing import Literal

BedrockModelName = Literal[
{literal_lines}
]

ReasoningFamily = Literal["claude_adaptive", "claude_budget", "nova", "openai"]
CacheLocation = Literal["system", "messages", "tools"]
CacheTTL = Literal["5m", "1h"]


@dataclass(frozen=True)
class Attr:
    context_size: int | None
    supports_tools: bool | None
    supports_thinking: bool | None
    supports_streaming: bool
    supports_explicit_caching: bool
    cache_min_tokens: int | None = None
    cache_ttls: tuple[CacheTTL, ...] = ()
    cache_locations: tuple[CacheLocation, ...] = ()
    reasoning_family: ReasoningFamily | None = None


attributes_map: dict[BedrockModelName, Attr] = {{
{attr_lines}
}}


def list_bedrock_models() -> list[str]:
    """List model and system-profile ids in Toki's generated snapshot."""
    return list(attributes_map)


def normalize_bedrock_model_id(model_id: str) -> str:
    """Return the foundation-model-like portion of a Bedrock id or ARN."""
    normalized = model_id.rsplit("/", 1)[-1]
    parts = normalized.split(".", 1)
    geographic_prefixes = {{
        "apac",
        "au",
        "eu",
        "global",
        "jp",
        "us",
        "us-gov",
    }}
    if len(parts) == 2 and parts[0] in geographic_prefixes:
        normalized = parts[1]
    return normalized


def get_bedrock_model_attributes(model_id: str) -> Attr | None:
    """Look up static capabilities for a model, profile id, or model ARN."""
    direct = attributes_map.get(model_id)  # type: ignore[arg-type]
    if direct is not None:
        return direct
    normalized = normalize_bedrock_model_id(model_id)
    for known_id, attributes in attributes_map.items():
        if normalize_bedrock_model_id(known_id) == normalized:
            return attributes
    return None
''',
        encoding="utf-8",
    )
