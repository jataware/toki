import asyncio
import json
from threading import Thread
from typing import AsyncIterator, Iterator, Literal
from uuid import uuid4

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, TextStreamer

from ..helpers._jsonstream import JsonStreamParser
from ..model import (
    BaseModel,
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


_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_TOOL_OPEN = "<tool_call>"
_TOOL_CLOSE = "</tool_call>"


class _ThinkingSplitter:
    """Streaming state machine: feed text chunks, yields ('content', s) or ('thought', s) tuples.

    Buffers boundary-straddling tokens like '<thi' until enough characters arrive to disambiguate
    a tag from regular text.
    """

    def __init__(self) -> None:
        self.buf: str = ""
        self.in_thought: bool = False

    def feed(self, chunk: str) -> Iterator[tuple[Literal['content', 'thought'], str]]:
        self.buf += chunk
        while self.buf:
            target, kind_after = (_THINK_OPEN, True) if not self.in_thought else (_THINK_CLOSE, False)
            kind: Literal['content', 'thought'] = 'thought' if self.in_thought else 'content'

            idx = self.buf.find(target)
            if idx != -1:
                emit = self.buf[:idx]
                if emit:
                    yield (kind, emit)
                self.buf = self.buf[idx + len(target):]
                self.in_thought = kind_after
                continue

            hold = len(target) - 1
            safe_end = _safe_emit_end(self.buf, target, hold)
            if safe_end > 0:
                emit = self.buf[:safe_end]
                self.buf = self.buf[safe_end:]
                yield (kind, emit)
            return

    def flush(self) -> Iterator[tuple[Literal['content', 'thought'], str]]:
        if self.buf:
            kind: Literal['content', 'thought'] = 'thought' if self.in_thought else 'content'
            yield (kind, self.buf)
            self.buf = ""


def _safe_emit_end(buf: str, target: str, hold: int) -> int:
    # find the largest k <= hold such that the last k chars of buf are a prefix of target
    max_check = min(hold, len(buf))
    for k in range(max_check, 0, -1):
        if target.startswith(buf[-k:]):
            return len(buf) - k
    return len(buf)


# Output events from `_ToolEnvelopeSplitter.feed`:
#   ('content', text)                     - regular content chunk
#   ('tool_first', id, name, index)       - first event for a new tool call (after parsing the name)
#   ('tool_args', chunk)                  - raw JSON-text fragment of the args object value
#   ('tool_close', None)                  - end of one tool-call envelope
class _ToolEnvelopeSplitter:
    """Streaming splitter that recognizes `<tool_call>{json}</tool_call>` envelopes inline.

    Outside an envelope, emits content chunks. Inside, drives an internal
    `JsonStreamParser` over the envelope JSON to detect the `name` key (after
    which the first tool event is emitted) and stream the args-object text
    fragments. Assumes the envelope JSON puts `name` before `arguments`.
    """

    def __init__(self) -> None:
        self.buf: str = ''
        self.state: Literal['content', 'envelope_json', 'envelope_after_json'] = 'content'
        self.parser: JsonStreamParser = JsonStreamParser()
        self.tool_id: str = ''
        self.tool_name: str | None = None
        self.tool_index: int = -1
        self.in_args_value: bool = False
        self.first_emitted: bool = False
        self.current_key: str | None = None

    def feed(self, chunk: str) -> Iterator[tuple]:
        self.buf += chunk
        while True:
            before_state = self.state
            before_buf_len = len(self.buf)
            if self.state == 'content':
                yield from self._step_content()
            elif self.state == 'envelope_json':
                yield from self._step_envelope_json()
            else:
                yield from self._step_envelope_after()
            if self.state == before_state and len(self.buf) == before_buf_len:
                return

    def flush(self) -> Iterator[tuple]:
        if self.state != 'content':
            raise ValueError(f"incomplete tool-call envelope at end of stream (state={self.state!r})")
        if self.buf:
            yield ('content', self.buf)
            self.buf = ''

    def _step_content(self) -> Iterator[tuple]:
        target = _TOOL_OPEN
        idx = self.buf.find(target)
        if idx != -1:
            emit = self.buf[:idx]
            if emit:
                yield ('content', emit)
            self.buf = self.buf[idx + len(target):]
            self.state = 'envelope_json'
            self.parser = JsonStreamParser()
            self.tool_id = f"local-tool-{uuid4().hex}"
            self.tool_name = None
            self.tool_index += 1
            self.in_args_value = False
            self.first_emitted = False
            self.current_key = None
            return
        hold = len(target) - 1
        safe_end = _safe_emit_end(self.buf, target, hold)
        if safe_end > 0:
            emit = self.buf[:safe_end]
            self.buf = self.buf[safe_end:]
            yield ('content', emit)

    def _step_envelope_json(self) -> Iterator[tuple]:
        while self.buf and self.state == 'envelope_json':
            ch = self.buf[0]
            self.buf = self.buf[1:]
            for ev in self.parser.feed(ch):
                kind = ev[0]
                if kind == 'arg_start':
                    self.current_key = ev[1]
                    if ev[1] == 'arguments':
                        self.in_args_value = True
                        if self.tool_name is None:
                            raise ValueError("local-model envelope put 'arguments' before 'name'; not supported")
                elif kind == 'arg_chunk':
                    if self.in_args_value:
                        yield ('tool_args', ev[1])
                elif kind == 'arg_end':
                    if self.current_key == 'name':
                        self.tool_name = ev[1]
                        if not self.first_emitted:
                            yield ('tool_first', self.tool_id, self.tool_name, self.tool_index)
                            self.first_emitted = True
                    elif self.current_key == 'arguments':
                        self.in_args_value = False
                    self.current_key = None
                elif kind == 'done':
                    # the envelope JSON ended with no 'arguments' key — emit an empty args object
                    if not self.first_emitted:
                        # also no name was seen — malformed envelope
                        raise ValueError("tool-call envelope missing 'name' field")
                    if not self.in_args_value and 'arguments' not in ev[1]:
                        yield ('tool_args', '{}')
                    self.state = 'envelope_after_json'
                    break
            if self.state != 'envelope_json':
                break

    def _step_envelope_after(self) -> Iterator[tuple]:
        target = _TOOL_CLOSE
        # skip leading whitespace
        i = 0
        while i < len(self.buf) and self.buf[i].isspace():
            i += 1
        self.buf = self.buf[i:]
        if not self.buf:
            return
        if self.buf.startswith(target):
            self.buf = self.buf[len(target):]
            yield ('tool_close', None)
            self.state = 'content'
            return
        if not target.startswith(self.buf):
            raise ValueError(f"expected '{target}' after tool-call envelope JSON, got {self.buf!r}")
        # buf is a prefix of target; wait for more


class _AsyncQueueStreamer(TextStreamer):
    """Bridge from the (sync) generation worker thread to an asyncio.Queue.

    Subclasses `TextStreamer` so it plugs straight into `model.generate(streamer=...)`,
    but redirects each finalized chunk into an `asyncio.Queue` on the calling
    event loop. A `None` sentinel is enqueued when generation ends, signalling
    the consumer to stop.
    """

    def __init__(self, tokenizer, loop: asyncio.AbstractEventLoop, queue: 'asyncio.Queue[str | None]', **decode_kwargs):
        super().__init__(tokenizer, **decode_kwargs)
        self._loop = loop
        self._queue = queue

    def on_finalized_text(self, text: str, stream_end: bool = False) -> None:
        if text:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, text)
        if stream_end:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, None)


class LocalModel(BaseModel):
    """Local `transformers`-backed model with the same interface as other toki backends."""

    def __init__(self, model: str, allow_parallel_tool_calls: bool = False):
        super().__init__()
        self.model = model
        self.allow_parallel_tool_calls = allow_parallel_tool_calls
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        if self.tokenizer.chat_template is None:
            raise ValueError(
                f"'{model}' has no chat template; toki targets instruction-tuned chat models. "
                f"Try the '-it' / '-Instruct' variant of this model if one exists."
            )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(model, torch_dtype="auto")
        self._model.to(self.device)
        self._model.eval()

    def count_tokens(
        self,
        messages: list[TokiMessage | dict],
        *,
        tools: ToolsArg = None,
        kind: Literal['exact'] = 'exact',
    ) -> int:
        if kind != 'exact':
            raise ValueError(f"LocalModel only supports kind='exact'; got {kind!r}")
        normalized = [TokiMessage.from_dict(m) for m in messages]
        wire_tools, _ = _unwrap_tools(tools)
        # mirror what `_raw_blocking` does: render the chat template to text first,
        # then tokenize. `apply_chat_template(tokenize=True, ...)` returns a
        # `BatchEncoding` (whose `len` is the number of fields, not the token
        # count) on fast tokenizers, so going through the prompt-string path is
        # both more portable and exactly consistent with the eventual generation call.
        prompt = self._build_prompt(normalized, tools=wire_tools)
        return len(self.tokenizer.encode(prompt))

    def _raw_blocking(
        self,
        messages: list[TokiMessage],
        tools: list[dict] | None,
        *,
        capture_thinking: bool,
        **kwargs,
    ) -> _RawTurn:
        prompt = self._build_prompt(messages, tools=tools)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        generation_kwargs = self._build_generate_kwargs(
            input_token_count=inputs["input_ids"].shape[-1],
            **kwargs,
        )

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                **generation_kwargs,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated_tokens = outputs[0, inputs["input_ids"].shape[-1]:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        usage = self._build_usage(prompt_tokens=inputs["input_ids"].shape[-1], completion_text=response)

        content_text, thought_text = self._split_thinking(response)
        tool_calls = self._extract_tool_calls(content_text) if tools else []
        # strip the envelope text out of the content for clean storage
        clean_content = self._strip_tool_envelopes(content_text)

        return _RawTurn(
            content=clean_content,
            tool_calls=tool_calls,
            thought=thought_text if capture_thinking else '',
            usage=usage,
        )

    def _raw_streaming(
        self,
        messages: list[TokiMessage],
        tools: list[dict] | None,
        *,
        capture_thinking: bool,
        **kwargs,
    ) -> Iterator[_RawChunk]:
        prompt = self._build_prompt(messages, tools=tools)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = self._build_generate_kwargs(
            input_token_count=inputs["input_ids"].shape[-1],
            **kwargs,
        )
        errors: list[BaseException] = []

        def run_generation() -> None:
            try:
                with torch.no_grad():
                    self._model.generate(
                        **inputs,
                        **generation_kwargs,
                        pad_token_id=self.tokenizer.pad_token_id,
                        streamer=streamer,
                    )
            except BaseException as exc:
                errors.append(exc)

        worker = Thread(target=run_generation, daemon=True)
        worker.start()

        thinking_splitter = _ThinkingSplitter()
        envelope_splitter = _ToolEnvelopeSplitter()
        # tracks the current open tool call's index so subsequent args fragments map back
        current_tool_index: int | None = None
        completion_text_chunks: list[str] = []

        def drive_envelope(text: str) -> Iterator[_RawChunk]:
            nonlocal current_tool_index
            for ev in envelope_splitter.feed(text):
                kind = ev[0]
                if kind == 'content':
                    if ev[1]:
                        yield _RawContentChunk(text=ev[1])
                elif kind == 'tool_first':
                    _, tid, tname, tindex = ev
                    current_tool_index = tindex
                    yield _RawToolCallChunk(index=tindex, id=tid, name=tname)
                elif kind == 'tool_args':
                    assert current_tool_index is not None
                    yield _RawToolCallChunk(index=current_tool_index, arguments_fragment=ev[1])
                elif kind == 'tool_close':
                    current_tool_index = None

        try:
            for raw in streamer:
                completion_text_chunks.append(raw)
                for kind, s in thinking_splitter.feed(raw):
                    if kind == 'thought':
                        if capture_thinking and s:
                            yield _RawThoughtChunk(text=s)
                    else:
                        yield from drive_envelope(s)
            for kind, s in thinking_splitter.flush():
                if kind == 'thought':
                    if capture_thinking and s:
                        yield _RawThoughtChunk(text=s)
                else:
                    yield from drive_envelope(s)
            for ev in envelope_splitter.flush():
                if ev[0] == 'content' and ev[1]:
                    yield _RawContentChunk(text=ev[1])
        finally:
            worker.join()
            if errors:
                raise errors[0]

        full_completion = "".join(completion_text_chunks)
        # also account for stripped thinking in the usage tally so callers see total work done
        usage = self._build_usage(prompt_tokens=inputs["input_ids"].shape[-1], completion_text=full_completion)
        yield _RawUsage(usage=usage)

    async def _raw_blocking_async(
        self,
        messages: list[TokiMessage],
        tools: list[dict] | None,
        *,
        capture_thinking: bool,
        **kwargs,
    ) -> _RawTurn:
        # `model.generate(stream=False)` is one big blocking GPU call with no incremental
        # output; the right primitive is to_thread (frees the event loop) rather than a
        # bespoke async re-implementation.
        return await asyncio.to_thread(
            self._raw_blocking,
            messages,
            tools,
            capture_thinking=capture_thinking,
            **kwargs,
        )

    async def _raw_streaming_async(
        self,
        messages: list[TokiMessage],
        tools: list[dict] | None,
        *,
        capture_thinking: bool,
        **kwargs,
    ) -> AsyncIterator[_RawChunk]:
        prompt = self._build_prompt(messages, tools=tools)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        streamer = _AsyncQueueStreamer(self.tokenizer, loop, queue, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = self._build_generate_kwargs(
            input_token_count=inputs["input_ids"].shape[-1],
            **kwargs,
        )
        errors: list[BaseException] = []

        def run_generation() -> None:
            try:
                with torch.no_grad():
                    self._model.generate(
                        **inputs,
                        **generation_kwargs,
                        pad_token_id=self.tokenizer.pad_token_id,
                        streamer=streamer,
                    )
            except BaseException as exc:
                errors.append(exc)
                # wake the consumer if generation died before stream_end
                loop.call_soon_threadsafe(queue.put_nowait, None)

        worker = Thread(target=run_generation, daemon=True)
        worker.start()

        thinking_splitter = _ThinkingSplitter()
        envelope_splitter = _ToolEnvelopeSplitter()
        current_tool_index: int | None = None
        completion_text_chunks: list[str] = []

        def drive_envelope(text: str) -> Iterator[_RawChunk]:
            nonlocal current_tool_index
            for ev in envelope_splitter.feed(text):
                kind = ev[0]
                if kind == 'content':
                    if ev[1]:
                        yield _RawContentChunk(text=ev[1])
                elif kind == 'tool_first':
                    _, tid, tname, tindex = ev
                    current_tool_index = tindex
                    yield _RawToolCallChunk(index=tindex, id=tid, name=tname)
                elif kind == 'tool_args':
                    assert current_tool_index is not None
                    yield _RawToolCallChunk(index=current_tool_index, arguments_fragment=ev[1])
                elif kind == 'tool_close':
                    current_tool_index = None

        try:
            while True:
                text = await queue.get()
                if text is None:
                    break
                completion_text_chunks.append(text)
                for kind, s in thinking_splitter.feed(text):
                    if kind == 'thought':
                        if capture_thinking and s:
                            yield _RawThoughtChunk(text=s)
                    else:
                        for ev in drive_envelope(s):
                            yield ev
            for kind, s in thinking_splitter.flush():
                if kind == 'thought':
                    if capture_thinking and s:
                        yield _RawThoughtChunk(text=s)
                else:
                    for ev in drive_envelope(s):
                        yield ev
            for ev in envelope_splitter.flush():
                if ev[0] == 'content' and ev[1]:
                    yield _RawContentChunk(text=ev[1])
        finally:
            # generation has signaled stream_end (or an error) — join is near-instant
            await asyncio.to_thread(worker.join)
            if errors:
                raise errors[0]

        full_completion = "".join(completion_text_chunks)
        usage = self._build_usage(prompt_tokens=inputs["input_ids"].shape[-1], completion_text=full_completion)
        yield _RawUsage(usage=usage)

    def _build_prompt(self, messages: list[TokiMessage], *, tools: list[dict] | None = None) -> str:
        template_kwargs: dict = {}
        if tools:
            template_kwargs["tools"] = tools
        wire_messages = [self._msg_to_wire(m) for m in messages]
        return self.tokenizer.apply_chat_template(
            wire_messages,
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )

    @staticmethod
    def _msg_to_wire(m: TokiMessage) -> dict:
        out: dict = {"role": m.role, "content": m.content}
        if m.tool_calls is not None:
            # mirror the OpenAI-style wire shape that most HF chat templates expect
            out["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {"name": tc.function.name, "arguments": json.dumps(tc.function.arguments)},
                }
                for tc in m.tool_calls
            ]
        if m.tool_call_id is not None:
            out["tool_call_id"] = m.tool_call_id
        return out

    @staticmethod
    def _split_thinking(text: str) -> tuple[str, str]:
        """Run the splitter eagerly over a complete string. Returns (content, thought)."""
        splitter = _ThinkingSplitter()
        content_parts: list[str] = []
        thought_parts: list[str] = []
        for kind, s in splitter.feed(text):
            (thought_parts if kind == 'thought' else content_parts).append(s)
        for kind, s in splitter.flush():
            (thought_parts if kind == 'thought' else content_parts).append(s)
        return "".join(content_parts), "".join(thought_parts)

    def _build_generate_kwargs(self, *, input_token_count: int, **kwargs) -> dict:
        temperature = kwargs.pop("temperature", 0.7)
        top_p = kwargs.pop("top_p", 0.95)
        max_new_tokens = kwargs.pop("max_new_tokens", None)

        generate_kwargs = {
            "do_sample": temperature > 0,
            **kwargs,
        }
        if max_new_tokens is not None:
            generate_kwargs["max_new_tokens"] = max_new_tokens
        else:
            generate_kwargs["max_new_tokens"] = max(1, self.tokenizer.model_max_length - input_token_count)
        if temperature > 0:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = top_p
        return generate_kwargs

    def _build_usage(self, *, prompt_tokens: int, completion_text: str) -> TokiUsageMetadata:
        completion_tokens = len(self.tokenizer.encode(completion_text, add_special_tokens=False))
        return TokiUsageMetadata(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

    def _extract_tool_calls(self, content_text: str) -> list[TokiToolCall]:
        """Eagerly extract tool calls from a full-content string by running the
        envelope splitter once across the whole text."""
        splitter = _ToolEnvelopeSplitter()
        calls: list[TokiToolCall] = []
        cur_id: str | None = None
        cur_name: str | None = None
        cur_args_chunks: list[str] = []

        def emit(events: Iterator[tuple]) -> None:
            nonlocal cur_id, cur_name, cur_args_chunks
            for ev in events:
                kind = ev[0]
                if kind == 'tool_first':
                    _, cur_id, cur_name, _ = ev
                    cur_args_chunks = []
                elif kind == 'tool_args':
                    cur_args_chunks.append(ev[1])
                elif kind == 'tool_close':
                    args_text = ''.join(cur_args_chunks) or '{}'
                    args = json.loads(args_text)
                    assert cur_id is not None and cur_name is not None
                    calls.append(TokiToolCall(id=cur_id, function=TokiToolFunction(name=cur_name, arguments=args)))
                    cur_id = None
                    cur_name = None
                    cur_args_chunks = []

        emit(splitter.feed(content_text))
        emit(splitter.flush())
        return calls

    @staticmethod
    def _strip_tool_envelopes(text: str) -> str:
        """Remove `<tool_call>...</tool_call>` envelopes from text for clean content storage."""
        out: list[str] = []
        idx = 0
        while True:
            open_idx = text.find(_TOOL_OPEN, idx)
            if open_idx == -1:
                out.append(text[idx:])
                break
            out.append(text[idx:open_idx])
            close_idx = text.find(_TOOL_CLOSE, open_idx + len(_TOOL_OPEN))
            if close_idx == -1:
                # malformed envelope; leave the rest in place
                out.append(text[open_idx:])
                break
            idx = close_idx + len(_TOOL_CLOSE)
        return ''.join(out)
