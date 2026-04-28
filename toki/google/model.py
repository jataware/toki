from ..litellm.model import _LiteLLMModel
from .models import GoogleModelName


class GoogleModel(_LiteLLMModel):
    """Toki frontend for Google's Gemini models via AI Studio, dispatched through litellm.

    Uses litellm's `gemini/*` provider with `GEMINI_API_KEY`. Vertex AI is a
    separate provider in litellm and is not wired up here.
    """

    def __init__(self, model: GoogleModelName | str, *, api_key: str, allow_parallel_tool_calls: bool = False, cache: bool = False):
        super().__init__(
            wire_model=f"gemini/{model}",
            api_key=api_key,
            allow_parallel_tool_calls=allow_parallel_tool_calls,
            cache=cache,
        )
        self.model = model
