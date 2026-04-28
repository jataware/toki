import json
import warnings
from typing import Any, Iterator, Sequence
from uuid import uuid4

from ollama import Client, ProgressResponse
from tqdm import tqdm

from ..model import (
    BaseModel,
    StreamingToolSchema,
    TokiMessage,
    TokiToolCall,
    TokiToolFunction,
    TokiUsageMetadata,
    _RawChunk,
    _RawContentChunk,
    _RawThoughtChunk,
    _RawToolCallChunk,
    _RawTurn,
    _RawUsage,
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


def _build_usage(prompt_eval_count: int | None, eval_count: int | None) -> TokiUsageMetadata | None:
    if prompt_eval_count is None and eval_count is None:
        return None
    pt = prompt_eval_count or 0
    ct = eval_count or 0
    return TokiUsageMetadata(prompt_tokens=pt, completion_tokens=ct, total_tokens=pt + ct)


class OllamaModel(BaseModel):
    """Toki frontend for a locally-running Ollama daemon.

    On construction, checks whether `model` is already pulled and pulls it (with a
    tqdm progress bar) if not. Subsequent `complete()` calls go to the daemon's
    `/api/chat` endpoint via the official `ollama` python client.
    """

    def __init__(self, model: OllamaModelName | str, *, host: str | None = None):
        super().__init__()
        self.model = model
        self._client = Client(host=host) if host is not None else Client()
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

    # ----- complete override (one-shot streaming-tools warning) -------------

    def complete(self, messages, *, stream: bool = False, tools: Sequence | None = None, capture_thinking: bool = False, **kwargs):
        if stream and tools and not self._warned_streaming_tools:
            if any(isinstance(t, StreamingToolSchema) for t in tools):
                warnings.warn(
                    "OllamaModel emits each tool call as a single batch (id+name+arguments together) "
                    "rather than as per-character argument deltas. StreamingToolSchema still works, "
                    "but TokiArgStream values arrive in one shot.",
                    stacklevel=2,
                )
                self._warned_streaming_tools = True
        return super().complete(messages, stream=stream, tools=tools, capture_thinking=capture_thinking, **kwargs)

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
        msg = response.message
        content = msg.content or ''
        thought = (msg.thinking or '') if capture_thinking else ''
        tool_calls = [_to_toki_tool_call(tc) for tc in (msg.tool_calls or [])]
        usage = _build_usage(response.prompt_eval_count, response.eval_count)
        return _RawTurn(content=content, tool_calls=tool_calls, thought=thought, usage=usage)

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
        # base StreamDriver / JsonStreamParser machinery can consume it normally.
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
                name = tc.function.name
                args = dict(tc.function.arguments or {})
                yield _RawToolCallChunk(index=idx, id=f"ollama-{uuid4().hex}", name=name)
                yield _RawToolCallChunk(index=idx, arguments_fragment=json.dumps(args))
            if chunk.prompt_eval_count is not None:
                last_prompt_eval = chunk.prompt_eval_count
            if chunk.eval_count is not None:
                last_eval = chunk.eval_count
        usage = _build_usage(last_prompt_eval, last_eval)
        if usage is not None:
            yield _RawUsage(usage=usage)


def _to_toki_tool_call(tc: Any) -> TokiToolCall:
    return TokiToolCall(
        id=f"ollama-{uuid4().hex}",
        function=TokiToolFunction(
            name=tc.function.name,
            arguments=dict(tc.function.arguments or {}),
        ),
    )
