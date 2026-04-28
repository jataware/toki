"""Internal litellm-backed shared core. Not exposed via `toki.<name>`.

The user-facing per-provider frontends (`toki.openai`, `toki.anthropic`,
`toki.google`) all subclass `_LiteLLMModel` from `toki.litellm.model`.
"""

from .model import _LiteLLMModel

__all__ = ['_LiteLLMModel']
