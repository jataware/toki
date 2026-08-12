import importlib
from typing import TYPE_CHECKING

from .models import AnthropicModelName, attributes_map
from .utils import get_anthropic_api_key, list_anthropic_models

if TYPE_CHECKING:
    from .model import AnthropicModel

# `AnthropicModel` is loaded lazily so that importing this package's dep-free
# helpers (`utils`, cache-marker functions used by OpenRouter) does not pull
# in litellm via `.model`.
_LAZY: dict[str, tuple[str, str]] = {
    'AnthropicModel': ('toki.anthropic.model', 'AnthropicModel'),
}


def __getattr__(name: str):
    if name in _LAZY:
        mod_path, attr = _LAZY[name]
        return getattr(importlib.import_module(mod_path), attr)
    raise AttributeError(f"module 'toki.anthropic' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_LAZY.keys()))


__all__ = ['AnthropicModel', 'AnthropicModelName', 'attributes_map', 'get_anthropic_api_key', 'list_anthropic_models']
