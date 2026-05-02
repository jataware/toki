from ..litellm.model import ReasoningEffort, _LiteLLMModel
from .models import OpenAIModelName


class OpenAIModel(_LiteLLMModel):
    """Toki frontend for OpenAI's chat-completions API, dispatched via litellm.

    No `cache=` kwarg: OpenAI's prompt-prefix caching is fully automatic for
    prompts >= 1024 tokens and cannot be disabled or controlled. See the README
    Caching section for details.
    """

    def __init__(
        self,
        model: OpenAIModelName | str,
        *,
        api_key: str,
        reasoning_effort: ReasoningEffort | None = None,
        allow_parallel_tool_calls: bool = False,
    ):
        super().__init__(
            wire_model=f"openai/{model}",
            api_key=api_key,
            reasoning_effort=reasoning_effort,
            allow_parallel_tool_calls=allow_parallel_tool_calls,
        )
        self.model = model
