import importlib
from typing import TYPE_CHECKING

from .models import OllamaModelName, attributes_map
from .utils import list_ollama_models

if TYPE_CHECKING:
    from .model import OllamaModel

# `OllamaModel` is loaded lazily so that fetch-models / utils can be imported
# without the `ollama` runtime extra.
_LAZY: dict[str, tuple[str, str]] = {
    'OllamaModel': ('toki.ollama.model', 'OllamaModel'),
}


def __getattr__(name: str):
    if name in _LAZY:
        mod_path, attr = _LAZY[name]
        return getattr(importlib.import_module(mod_path), attr)
    raise AttributeError(f"module 'toki.ollama' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_LAZY.keys()))


__all__ = ['OllamaModel', 'OllamaModelName', 'attributes_map', 'list_ollama_models']
