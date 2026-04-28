from ..litellm.model import _LiteLLMModel
from .models import OpenAIModelName


class OpenAIModel(_LiteLLMModel):
    """Toki frontend for OpenAI's chat-completions API, dispatched via litellm."""

    def __init__(self, model: OpenAIModelName | str, *, api_key: str, allow_parallel_tool_calls: bool = False, cache: bool = False):
        super().__init__(
            wire_model=f"openai/{model}",
            api_key=api_key,
            allow_parallel_tool_calls=allow_parallel_tool_calls,
            cache=cache,
        )
        self.model = model
