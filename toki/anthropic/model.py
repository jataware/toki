from ..litellm.model import ReasoningEffort, _LiteLLMModel
from .models import AnthropicModelName


class AnthropicModel(_LiteLLMModel):
    """Toki frontend for Anthropic's Claude models, dispatched via litellm."""

    def __init__(
        self,
        model: AnthropicModelName | str,
        *,
        api_key: str,
        reasoning_effort: ReasoningEffort | None = None,
        allow_parallel_tool_calls: bool = False,
        cache: bool = False,
    ):
        super().__init__(
            wire_model=f"anthropic/{model}",
            api_key=api_key,
            reasoning_effort=reasoning_effort,
            allow_parallel_tool_calls=allow_parallel_tool_calls,
            cache=cache,
        )
        self.model = model
