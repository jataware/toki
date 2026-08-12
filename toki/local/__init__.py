import importlib
from typing import TYPE_CHECKING

from .models import LocalModelName, attributes_map
from .utils import list_huggingface_models

if TYPE_CHECKING:
    from .transformers import LocalModel

# `LocalModel` is loaded lazily so that fetch-models / utils can be imported
# without the `local` runtime extra (torch + transformers).
_LAZY: dict[str, tuple[str, str]] = {
    'LocalModel': ('toki.local.transformers', 'LocalModel'),
}


def __getattr__(name: str):
    if name in _LAZY:
        mod_path, attr = _LAZY[name]
        return getattr(importlib.import_module(mod_path), attr)
    raise AttributeError(f"module 'toki.local' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_LAZY.keys()))


__all__ = ['LocalModel', 'LocalModelName', 'attributes_map', 'list_huggingface_models']
