from typing import Literal

from ..litellm.model import ReasoningEffort, _LiteLLMModel
from ..model import TokiMessage, ToolsArg
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

    def count_tokens(
        self,
        messages: list[TokiMessage | dict],
        *,
        tools: ToolsArg = None,
        kind: Literal['exact'] = 'exact',
    ) -> int:
        """
        Count the prompt tokens for the given messages (and tools).
        
        Produces the EXACT prompt-token count for the OpenAI wire payload, computed
        purely offline via `litellm.token_counter` (which routes to `tiktoken`
        for OpenAI models). No network round-trip.

        Only `kind='exact'` is exposed; passing any other value raises
        `ValueError`. There is no `'offline'` / `'online'` distinction because
        tiktoken is already exact and offline.
        """
        if kind != 'exact':
            raise ValueError(f"OpenAIModel only supports kind='exact'; got {kind!r}")
        wire_messages, wire_tools = self._normalize_for_count(messages, tools)
        return self._litellm_offline_count(wire_messages, wire_tools)
