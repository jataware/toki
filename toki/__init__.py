import importlib

from .agent import Agent
from .model import (
    BaseModel,
    Role,
    TokiChatResponse,
    TokiMessage,
    TokiThinking,
    TokiToolCall,
    TokiToolFunction,
    TokiToolResponse,
    TokiUsageMetadata,
    pretty_tool_call,
)
from .statemachine import ClassStateMachine, END_STATE, EndState, StateMachine, on


# Per-provider symbols exposed at the top level. Loaded lazily via
# module-level `__getattr__` so that backends with optional dependencies don't
# get imported on a bare `import toki`.
_LAZY: dict[str, tuple[str, str]] = {
    'OpenRouterModel': ('toki.openrouter', 'OpenRouterModel'),
    'OpenRouterModelName': ('toki.openrouter', 'OpenRouterModelName'),
    'get_openrouter_api_key': ('toki.openrouter', 'get_openrouter_api_key'),
    'LocalModel': ('toki.local', 'LocalModel'),
    'LocalModelName': ('toki.local', 'LocalModelName'),
}


def __getattr__(name: str):
    if name in _LAZY:
        mod_path, attr = _LAZY[name]
        return getattr(importlib.import_module(mod_path), attr)
    raise AttributeError(f"module 'toki' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_LAZY.keys()))


__all__ = [
    'Agent',
    'BaseModel',
    'Role',
    'TokiChatResponse',
    'TokiMessage',
    'TokiThinking',
    'TokiToolCall',
    'TokiToolFunction',
    'TokiToolResponse',
    'TokiUsageMetadata',
    'pretty_tool_call',
    'StateMachine',
    'ClassStateMachine',
    'on',
    'EndState',
    'END_STATE',
    'OpenRouterModel',
    'OpenRouterModelName',
    'get_openrouter_api_key',
    'LocalModel',
    'LocalModelName',
]
