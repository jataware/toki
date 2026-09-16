import importlib
from typing import TYPE_CHECKING

from .models import BedrockModelName, attributes_map, list_bedrock_models
from .reasoning import (
    BedrockReasoningConfig,
    ClaudeAdaptiveReasoning,
    ClaudeBudgetReasoning,
    NovaReasoning,
    OpenAIReasoning,
)
from .utils import get_bedrock_api_key

if TYPE_CHECKING:
    from .discovery import (
        BedrockModelInfo,
        adiscover_bedrock_models,
        discover_bedrock_models,
    )
    from .model import BedrockModel

_LAZY: dict[str, tuple[str, str]] = {
    "BedrockModel": ("toki.bedrock.model", "BedrockModel"),
    "BedrockModelInfo": ("toki.bedrock.discovery", "BedrockModelInfo"),
    "discover_bedrock_models": (
        "toki.bedrock.discovery",
        "discover_bedrock_models",
    ),
    "adiscover_bedrock_models": (
        "toki.bedrock.discovery",
        "adiscover_bedrock_models",
    ),
}


def __getattr__(name: str):
    if name in _LAZY:
        module_path, attribute = _LAZY[name]
        return getattr(importlib.import_module(module_path), attribute)
    raise AttributeError(f"module 'toki.bedrock' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_LAZY.keys()))


__all__ = [
    "BedrockModel",
    "BedrockModelInfo",
    "BedrockModelName",
    "BedrockReasoningConfig",
    "ClaudeAdaptiveReasoning",
    "ClaudeBudgetReasoning",
    "NovaReasoning",
    "OpenAIReasoning",
    "adiscover_bedrock_models",
    "attributes_map",
    "discover_bedrock_models",
    "get_bedrock_api_key",
    "list_bedrock_models",
]
