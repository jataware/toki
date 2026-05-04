import json
import warnings
from typing import Any, AsyncIterator, Iterator, Literal, Sequence
from uuid import uuid4

from ollama import AsyncClient, Client, ProgressResponse
from tqdm import tqdm

from ..model import (
    BaseModel,
    StreamingToolSchema,
    TokiMessage,
    TokiToolCall,
    TokiToolFunction,
    TokiUsageMetadata,
    ToolsArg,
    _RawChunk,
    _RawContentChunk,
    _RawThoughtChunk,
    _RawToolCallChunk,
    _RawTurn,
    _RawUsage,
    _unwrap_tools,
)
from .models import OllamaModelName


def _normalize_model_name(model: str) -> str:
    """Ollama treats a bare family name as `<family>:latest` when pulling/chatting; do the same for our local existence check."""
    return model if ':' in model else f"{model}:latest"


def _messages_to_wire(messages: list[TokiMessage]) -> list[dict]:
    """Translate toki's `TokiMessage` history into ollama's wire shape.

    The two notable differences from openai-style:
    - tool-call `arguments` is a dict (already parsed), not a JSON string
    - tool responses identify their tool via `tool_name` rather than `tool_call_id`,
      so we walk forward through prior assistant tool_calls to map ids -> names.
    """
    id_to_name: dict[str, str] = {}
    out: list[dict] = []
    for m in messages:
        wire: dict = {"role": m.role, "content": m.content}
        if m.tool_calls:
            wire_tool_calls: list[dict] = []
            for tc in m.tool_calls:
                id_to_name[tc.id] = tc.function.name
                wire_tool_calls.append({
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                })
            wire["tool_calls"] = wire_tool_calls
        if m.tool_call_id is not None:
            wire["tool_name"] = id_to_name.get(m.tool_call_id, "")
        out.append(wire)
    return out


def _prepare_for_count(
    messages: list[TokiMessage | dict],
    tools: ToolsArg,
) -> tuple[list[dict], list[dict] | None]:
    """Normalize messages and tools into the wire shapes the daemon expects for a count request."""
    normalized = [TokiMessage.from_dict(m) for m in messages]
    wire_tools, _ = _unwrap_tools(tools)
    return _messages_to_wire(normalized), wire_tools


def _build_usage(prompt_eval_count: int | None, eval_count: int | None) -> TokiUsageMetadata | None:
    if prompt_eval_count is None and eval_count is None:
        return None
    pt = prompt_eval_count or 0
    ct = eval_count or 0
    return TokiUsageMetadata(prompt_tokens=pt, completion_tokens=ct, total_tokens=pt + ct)


class OllamaModel(BaseModel):
    """Toki frontend for a locally-running Ollama daemon.

    On construction, checks whether `model` is already pulled and pulls it (with a
    tqdm progress bar) if not. Subsequent `complete()` / `acomplete()` calls go to
    the daemon's `/api/chat` endpoint via the official `ollama` python client.
    """

    def __init__(self, model: OllamaModelName | str, *, host: str | None = None):
        super().__init__()
        self.model = model
        client_kwargs: dict = {"host": host} if host is not None else {}
        self._client = Client(**client_kwargs)
        self._async_client = AsyncClient(**client_kwargs)
        self._warned_streaming_tools = False
        self._ensure_pulled()

    # ----- pull -------------------------------------------------------------

    def _ensure_pulled(self) -> None:
        listed = self._client.list()
        present = {entry.model for entry in listed.models if entry.model}
        target = _normalize_model_name(self.model)
        if self.model in present or target in present:
            return
        self._pull_with_progress()

    def _pull_with_progress(self) -> None:
        bar: tqdm | None = None
        current_digest: str | None = None
        try:
            for ev in self._client.pull(self.model, stream=True):
                ev: ProgressResponse
                total = ev.total
                completed = ev.completed or 0
                digest = ev.digest
                status = ev.status or ''
                if total and digest:
                    if digest != current_digest:
                        if bar is not None:
                            bar.close()
                        bar = tqdm(
                            total=total,
                            unit='B',
                            unit_scale=True,
                            unit_divisor=1024,
                            desc=f"{status[:32]} ({digest[:12]})",
                        )
                        current_digest = digest
                    assert bar is not None
                    delta = completed - bar.n
                    if delta > 0:
                        bar.update(delta)
                else:
                    if bar is not None:
                        bar.close()
                        bar = None
                        current_digest = None
                    if status:
                        print(f"[ollama pull] {status}")
        finally:
            if bar is not None:
                bar.close()

    # ----- streaming-tools warning ------------------------------------------

    def _maybe_warn_streaming_tools(self, stream: bool, tools: Sequence | None) -> None:
        if stream and tools and not self._warned_streaming_tools:
            if any(isinstance(t, StreamingToolSchema) for t in tools):
                warnings.warn(
                    "OllamaModel emits each tool call as a single batch (id+name+arguments together) "
                    "rather than as per-character argument deltas. StreamingToolSchema still works, "
                    "but TokiArgStream values arrive in one shot.",
                    stacklevel=3,
                )
                self._warned_streaming_tools = True

    def complete(self, messages, *, stream: bool = False, tools: Sequence | None = None, capture_thinking: bool = False, **kwargs):
        self._maybe_warn_streaming_tools(stream, tools)
        return super().complete(messages, stream=stream, tools=tools, capture_thinking=capture_thinking, **kwargs)

    def acomplete(self, messages, *, stream: bool = False, tools: Sequence | None = None, capture_thinking: bool = False, **kwargs):
        self._maybe_warn_streaming_tools(stream, tools)
        return super().acomplete(messages, stream=stream, tools=tools, capture_thinking=capture_thinking, **kwargs)

    # ----- token counting ---------------------------------------------------

    def count_tokens(
        self,
        messages: list[TokiMessage | dict],
        *,
        tools: ToolsArg = None,
        kind: Literal['exact'] = 'exact',
    ) -> int:
        if kind != 'exact':
            raise ValueError(f"OllamaModel only supports kind='exact'; got {kind!r}")
        wire_messages, wire_tools = _prepare_for_count(messages, tools)
        response = self._client.chat(
            model=self.model,
            messages=wire_messages,
            tools=wire_tools,
            stream=False,
            options={"num_predict": 0},
        )
        count = response.prompt_eval_count
        if count is None:
            raise RuntimeError("ollama daemon did not return prompt_eval_count for token-count request")
        return count

    async def acount_tokens(
        self,
        messages: list[TokiMessage | dict],
        *,
        tools: ToolsArg = None,
        kind: Literal['exact'] = 'exact',
    ) -> int:
        if kind != 'exact':
            raise ValueError(f"OllamaModel only supports kind='exact'; got {kind!r}")
        wire_messages, wire_tools = _prepare_for_count(messages, tools)
        response = await self._async_client.chat(
            model=self.model,
            messages=wire_messages,
            tools=wire_tools,
            stream=False,
            options={"num_predict": 0},
        )
        count = response.prompt_eval_count
        if count is None:
            raise RuntimeError("ollama daemon did not return prompt_eval_count for token-count request")
        return count

    # ----- raw I/O ----------------------------------------------------------

    def _raw_blocking(
        self,
        messages: list[TokiMessage],
        tools: list[dict] | None,
        *,
        capture_thinking: bool,
        **kwargs,
    ) -> _RawTurn:
        response = self._client.chat(
            model=self.model,
            messages=_messages_to_wire(messages),
            tools=tools,
            think=capture_thinking or None,
            stream=False,
            **kwargs,
        )
        return _turn_from_blocking_response(response, capture_thinking=capture_thinking)

    def _raw_streaming(
        self,
        messages: list[TokiMessage],
        tools: list[dict] | None,
        *,
        capture_thinking: bool,
        **kwargs,
    ) -> Iterator[_RawChunk]:
        stream = self._client.chat(
            model=self.model,
            messages=_messages_to_wire(messages),
            tools=tools,
            think=capture_thinking or None,
            stream=True,
            **kwargs,
        )
        # ollama hands tool_calls as fully-formed objects per chunk (not arg-fragment deltas).
        # Synthesize per-call a (id+name, then complete arguments_fragment) pair so the
        # base StreamCore / JsonStreamParser machinery can consume it normally.
        next_tool_index = 0
        last_prompt_eval = None
        last_eval = None
        for chunk in stream:
            msg = chunk.message
            if capture_thinking and msg.thinking:
                yield _RawThoughtChunk(text=msg.thinking)
            if msg.content:
                yield _RawContentChunk(text=msg.content)
            for tc in (msg.tool_calls or []):
                idx = next_tool_index
                next_tool_index += 1
                yield _RawToolCallChunk(index=idx, id=f"ollama-{uuid4().hex}", name=tc.function.name)
                yield _RawToolCallChunk(index=idx, arguments_fragment=json.dumps(dict(tc.function.arguments or {})))
            if chunk.prompt_eval_count is not None:
                last_prompt_eval = chunk.prompt_eval_count
            if chunk.eval_count is not None:
                last_eval = chunk.eval_count
        usage = _build_usage(last_prompt_eval, last_eval)
        if usage is not None:
            yield _RawUsage(usage=usage)

    async def _raw_blocking_async(
        self,
        messages: list[TokiMessage],
        tools: list[dict] | None,
        *,
        capture_thinking: bool,
        **kwargs,
    ) -> _RawTurn:
        response = await self._async_client.chat(
            model=self.model,
            messages=_messages_to_wire(messages),
            tools=tools,
            think=capture_thinking or None,
            stream=False,
            **kwargs,
        )
        return _turn_from_blocking_response(response, capture_thinking=capture_thinking)

    async def _raw_streaming_async(
        self,
        messages: list[TokiMessage],
        tools: list[dict] | None,
        *,
        capture_thinking: bool,
        **kwargs,
    ) -> AsyncIterator[_RawChunk]:
        stream = await self._async_client.chat(
            model=self.model,
            messages=_messages_to_wire(messages),
            tools=tools,
            think=capture_thinking or None,
            stream=True,
            **kwargs,
        )
        next_tool_index = 0
        last_prompt_eval = None
        last_eval = None
        async for chunk in stream:
            msg = chunk.message
            if capture_thinking and msg.thinking:
                yield _RawThoughtChunk(text=msg.thinking)
            if msg.content:
                yield _RawContentChunk(text=msg.content)
            for tc in (msg.tool_calls or []):
                idx = next_tool_index
                next_tool_index += 1
                yield _RawToolCallChunk(index=idx, id=f"ollama-{uuid4().hex}", name=tc.function.name)
                yield _RawToolCallChunk(index=idx, arguments_fragment=json.dumps(dict(tc.function.arguments or {})))
            if chunk.prompt_eval_count is not None:
                last_prompt_eval = chunk.prompt_eval_count
            if chunk.eval_count is not None:
                last_eval = chunk.eval_count
        usage = _build_usage(last_prompt_eval, last_eval)
        if usage is not None:
            yield _RawUsage(usage=usage)


def _turn_from_blocking_response(response: Any, *, capture_thinking: bool) -> _RawTurn:
    """Translate a non-streaming ollama `ChatResponse` into a `_RawTurn`."""
    msg = response.message
    content = msg.content or ''
    thought = (msg.thinking or '') if capture_thinking else ''
    tool_calls = [_to_toki_tool_call(tc) for tc in (msg.tool_calls or [])]
    usage = _build_usage(response.prompt_eval_count, response.eval_count)
    return _RawTurn(content=content, tool_calls=tool_calls, thought=thought, usage=usage)


def _to_toki_tool_call(tc: Any) -> TokiToolCall:
    return TokiToolCall(
        id=f"ollama-{uuid4().hex}",
        function=TokiToolFunction(
            name=tc.function.name,
            arguments=dict(tc.function.arguments or {}),
        ),
    )
