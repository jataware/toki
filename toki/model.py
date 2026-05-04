import json
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, AsyncIterator, Coroutine, Generator, Generic, Iterator, Literal, Sequence, TypeVar, overload
from uuid import uuid4

from .helpers._jsonstream import JsonStreamParser


Role = Literal["user", "assistant", "system", "tool"]


# --- Wire / message / tool-call types ----------------------------------------

@dataclass
class TokiToolFunction:
    name: str
    arguments: dict

    @classmethod
    def from_dict(cls, x: 'TokiToolFunction | dict') -> 'TokiToolFunction':
        if isinstance(x, cls):
            return x
        # provider wire formats (e.g. OpenAI/OpenRouter) deliver arguments as a JSON-encoded string;
        # local backends already build a dict. Normalize to dict here.
        args = x["arguments"]
        if isinstance(args, str):
            args = json.loads(args) if args else {}
        return cls(name=x["name"], arguments=args)


@dataclass
class TokiToolCall:
    id: str
    function: TokiToolFunction
    type: Literal["function"] = "function"

    @classmethod
    def from_dict(cls, x: 'TokiToolCall | dict') -> 'TokiToolCall':
        if isinstance(x, cls):
            return x
        return cls(
            id=x["id"],
            type=x.get("type", "function"),
            function=TokiToolFunction.from_dict(x["function"]),
        )


@dataclass
class TokiMessage:
    role: Role
    content: str
    tool_calls: list[TokiToolCall] | None = None
    tool_call_id: str | None = None

    @classmethod
    def from_dict(cls, x: 'TokiMessage | dict') -> 'TokiMessage':
        if isinstance(x, cls):
            return x
        tcs = x.get("tool_calls")
        return cls(
            role=x["role"],
            content=x["content"],
            tool_calls=[TokiToolCall.from_dict(t) for t in tcs] if tcs else None,
            tool_call_id=x.get("tool_call_id"),
        )


@dataclass
class TokiUsageMetadata:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class TokenCountEstimate:
    """Returned by `count_tokens(kind='offline')` (or any path that can't produce
    an exact count). `prompt_tokens` is the post-safety-factor figure callers
    should budget against; `raw_prompt_tokens` is what the underlying estimator
    actually returned, and `safety_factor` is the multiplier already applied.
    """
    prompt_tokens: int
    raw_prompt_tokens: int
    safety_factor: float


# --- Tool-schema wrappers ----------------------------------------------------

@dataclass
class ToolSchema:
    """Wrapper around a static (non-streaming) OpenAI-style tool schema.

    Equivalent to passing a raw dict in the tools list; only used by callers that
    want the explicit type annotation. Backends only ever see the unwrapped dict.
    """
    schema: dict


@dataclass
class StreamingToolSchema:
    """Wrapper around a tool schema marking the tool as streaming.

    When the model invokes this tool during `complete(stream=True)`, the call is
    surfaced as a live `TokiToolCallStream` whose argument values can be consumed
    incrementally via `expect_arg()` or `items()`. Required to opt into per-arg
    streaming (a raw dict or `ToolSchema` is treated as static).
    """
    schema: dict


# --- Streaming chunk types ---------------------------------------------------

@dataclass
class TokiThinking:
    """Streaming chunk carrying a piece of the model's reasoning text. Only seen when `capture_thinking=True`."""
    text: str


# --- Per-arg / per-tool-call live streams ------------------------------------

class _ArgStreamState:
    """Shared state for sync/async ArgStream wrappers.

    Holds the chunk queue, completion flag, parsed value, and never-appeared
    flag. The two iteration protocols live on the subclasses.
    """

    def __init__(self, name: str, parent: '_ToolCallStreamStateBase') -> None:
        self.name = name
        self._parent = parent
        self._chunks: deque[str] = deque()
        self._done: bool = False
        self._value: Any = None
        self._never_appeared: bool = False

    def _push_chunk(self, chunk: str) -> None:
        self._chunks.append(chunk)

    def _close(self, value: Any) -> None:
        self._done = True
        self._value = value

    def _mark_never_appeared(self) -> None:
        self._never_appeared = True
        self._done = True


class TokiArgStream(_ArgStreamState):
    """A sync stream of one tool-call argument's value as it arrives.

    Iterating yields decoded characters for top-level string values, and raw JSON
    text fragments for everything else (numbers, booleans, null, arrays, objects).
    `.value` returns the parsed Python value once the stream is exhausted (and
    drains the stream as a side effect if it isn't already).

    Single-shot: a `TokiArgStream` returned from `expect_arg(name)` or `items()`
    is intended to be iterated once. Iterating after the underlying value has
    finished arriving is a no-op.
    """

    def __iter__(self) -> Iterator[str]:
        while True:
            while self._chunks:
                yield self._chunks.popleft()
            if self._never_appeared:
                raise RuntimeError(f"arg {self.name!r} never appeared in tool call")
            if self._done:
                return
            advanced = self._parent._driver.advance()
            if not advanced:
                if self._chunks:
                    continue
                if self._done:
                    return
                raise RuntimeError(f"stream ended before arg {self.name!r} completed")

    @property
    def value(self) -> Any:
        for _ in self:
            pass
        return self._value


class AsyncTokiArgStream(_ArgStreamState):
    """Async sibling of `TokiArgStream`. Same semantics, async iteration.

    Iterate with `async for piece in arg_stream`. `await arg_stream.value()`
    drains and returns the parsed Python value.
    """

    def __aiter__(self) -> 'AsyncTokiArgStream':
        return self

    async def __anext__(self) -> str:
        while True:
            if self._chunks:
                return self._chunks.popleft()
            if self._never_appeared:
                raise RuntimeError(f"arg {self.name!r} never appeared in tool call")
            if self._done:
                raise StopAsyncIteration
            advanced = await self._parent._driver.advance()
            if not advanced:
                if self._chunks:
                    continue
                if self._done:
                    raise StopAsyncIteration
                raise RuntimeError(f"stream ended before arg {self.name!r} completed")

    async def value(self) -> Any:
        async for _ in self:
            pass
        return self._value


class _ToolCallStreamStateBase:
    """Shared state and event handling for sync/async tool-call streams.

    Subclasses bind `_ARG_STREAM_CLS` (sync vs async ArgStream) and add
    iteration-protocol methods (`expect_arg`, `items`, `arguments`) on top of
    this shared state.
    """

    _ARG_STREAM_CLS: type

    def __init__(self, *, driver: Any, id: str, name: str) -> None:
        self.id = id
        self.name = name
        self._driver = driver
        # finalized parsed dict (filled when the underlying parser emits 'done')
        self._dict: dict = {}
        # arg names in the order they arrived (committed entries are those with a value in _args_values)
        self._args_order: list[str] = []
        # for each completed arg: chunks that were emitted (raw or decoded)
        self._args_chunks: dict[str, list[str]] = {}
        # for each completed arg: parsed Python value
        self._args_values: dict[str, Any] = {}
        # currently-streaming arg state
        self._current_name: str | None = None
        self._current_chunks: list[str] = []
        # if the user has claimed the current arg, the live ArgStream
        self._current_stream: _ArgStreamState | None = None
        # claims made before the arg arrived
        self._pending_claims: dict[str, _ArgStreamState] = {}
        # one-shot enforcement
        self._used: str | None = None  # 'expect_arg' | 'items'
        self._expected_names: set[str] = set()
        self._items_started: bool = False
        # source/finalization
        self._done: bool = False

    # ----- driver-side wiring ------------------------------------------------

    def _handle_event(self, ev: tuple) -> None:
        kind = ev[0]
        if kind == 'arg_start':
            name: str = ev[1]
            self._current_name = name
            self._current_chunks = []
            self._args_order.append(name)
            claimed = self._pending_claims.pop(name, None)
            self._current_stream = claimed
        elif kind == 'arg_chunk':
            chunk: str = ev[1]
            self._current_chunks.append(chunk)
            if self._current_stream is not None:
                self._current_stream._push_chunk(chunk)
        elif kind == 'arg_end':
            value = ev[1]
            assert self._current_name is not None
            name = self._current_name
            self._args_chunks[name] = list(self._current_chunks)
            self._args_values[name] = value
            self._dict[name] = value
            if self._current_stream is not None:
                self._current_stream._close(value)
            self._current_name = None
            self._current_chunks = []
            self._current_stream = None
        elif kind == 'done':
            self._done = True
            self._dict = ev[1]
            # any never-appeared pending claims raise on iteration
            for stream in self._pending_claims.values():
                stream._mark_never_appeared()
            self._pending_claims.clear()

    def _mark_source_ended(self) -> None:
        # underlying source went away before we got 'done'; signal pending claims
        for stream in self._pending_claims.values():
            stream._mark_never_appeared()
        self._pending_claims.clear()
        if self._current_stream is not None:
            self._current_stream._mark_never_appeared()
        self._done = True

    # ----- shared expect_arg setup ------------------------------------------

    def _expect_arg_setup(self, name: str) -> _ArgStreamState | None:
        """Common state checks and the three replay/live/done branches that don't
        require pumping the driver. Returns a fully-prepared stream for the
        first three branches, or None if the caller should register a pending
        claim (the only branch that actually waits for new data).
        """
        if self._used == 'items':
            raise RuntimeError("expect_arg() cannot be called after items()")
        self._used = 'expect_arg'
        if name in self._expected_names:
            raise RuntimeError(f"expect_arg({name!r}) already called")
        self._expected_names.add(name)

        if name in self._args_values:
            stream = self._ARG_STREAM_CLS(name, self)
            for chunk in self._args_chunks[name]:
                stream._push_chunk(chunk)
            stream._close(self._args_values[name])
            return stream

        if self._current_name == name:
            stream = self._ARG_STREAM_CLS(name, self)
            for chunk in self._current_chunks:
                stream._push_chunk(chunk)
            self._current_chunks = []
            self._current_stream = stream
            return stream

        if self._done:
            raise RuntimeError(f"arg {name!r} never appeared in tool call")

        return None

    def _items_setup(self) -> None:
        if self._used == 'expect_arg':
            raise RuntimeError("items() cannot be called after expect_arg()")
        if self._items_started:
            raise RuntimeError("items() can only be called once")
        self._items_started = True
        self._used = 'items'


class TokiToolCallStream(_ToolCallStreamStateBase):
    """A live (sync) stream of a single tool call.

    Yielded once per streaming-tool invocation, as soon as the model has emitted
    the call's id and name. Argument values can be consumed live via
    `expect_arg(name)` or by iterating `items()`. Both are one-shot per stream
    and mutually exclusive. After the stream is fully drained, `.arguments`
    returns the parsed dict of all arguments.
    """

    _ARG_STREAM_CLS = TokiArgStream

    def expect_arg(self, name: str) -> TokiArgStream:
        prepared = self._expect_arg_setup(name)
        if prepared is not None:
            return prepared  # type: ignore[return-value]
        stream = TokiArgStream(name, self)
        self._pending_claims[name] = stream
        return stream

    def items(self) -> Iterator[tuple[str, TokiArgStream]]:
        self._items_setup()
        emitted_idx = 0
        # walk args in arrival order; new ones may be appended as we advance the driver
        while True:
            if emitted_idx < len(self._args_order):
                name = self._args_order[emitted_idx]
                emitted_idx += 1
                if name in self._args_values:
                    # already complete — single-shot replay
                    stream = TokiArgStream(name, self)
                    for chunk in self._args_chunks[name]:
                        stream._push_chunk(chunk)
                    stream._close(self._args_values[name])
                    yield name, stream
                    continue
                # currently streaming — claim it live
                stream = TokiArgStream(name, self)
                for chunk in self._current_chunks:
                    stream._push_chunk(chunk)
                self._current_chunks = []
                self._current_stream = stream
                yield name, stream
                # keep advancing until this arg completes (auto-drain on next iteration)
                while not stream._done and not self._done:
                    if not self._driver.advance():
                        break
                continue
            # no pending arg; pull more from driver until a new arg starts or we're done
            if self._done:
                return
            if not self._driver.advance():
                return

    @property
    def arguments(self) -> dict:
        # idempotent drain
        while not self._done:
            if not self._driver.advance():
                break
        # if there's a live current_stream still mid-arrival, drain it too
        if self._current_stream is not None and not self._current_stream._done:
            assert isinstance(self._current_stream, TokiArgStream)
            for _ in self._current_stream:
                pass
        return self._dict


class AsyncTokiToolCallStream(_ToolCallStreamStateBase):
    """Async sibling of `TokiToolCallStream`.

    Same lifecycle and one-shot semantics as the sync version. `expect_arg(name)`
    returns an `AsyncTokiArgStream`. `items()` is an async generator. `arguments`
    is `await stream.arguments()` (a coroutine, since draining can suspend).
    """

    _ARG_STREAM_CLS = AsyncTokiArgStream

    def expect_arg(self, name: str) -> AsyncTokiArgStream:
        prepared = self._expect_arg_setup(name)
        if prepared is not None:
            return prepared  # type: ignore[return-value]
        stream = AsyncTokiArgStream(name, self)
        self._pending_claims[name] = stream
        return stream

    async def items(self) -> AsyncIterator[tuple[str, AsyncTokiArgStream]]:
        self._items_setup()
        emitted_idx = 0
        while True:
            if emitted_idx < len(self._args_order):
                name = self._args_order[emitted_idx]
                emitted_idx += 1
                if name in self._args_values:
                    stream = AsyncTokiArgStream(name, self)
                    for chunk in self._args_chunks[name]:
                        stream._push_chunk(chunk)
                    stream._close(self._args_values[name])
                    yield name, stream
                    continue
                stream = AsyncTokiArgStream(name, self)
                for chunk in self._current_chunks:
                    stream._push_chunk(chunk)
                self._current_chunks = []
                self._current_stream = stream
                yield name, stream
                while not stream._done and not self._done:
                    if not await self._driver.advance():
                        break
                continue
            if self._done:
                return
            if not await self._driver.advance():
                return

    async def arguments(self) -> dict:
        while not self._done:
            if not await self._driver.advance():
                break
        if self._current_stream is not None and not self._current_stream._done:
            assert isinstance(self._current_stream, AsyncTokiArgStream)
            async for _ in self._current_stream:
                pass
        return self._dict


# --- Blocking response shapes ------------------------------------------------

@dataclass
class TokiThoughtResponse:
    """Returned (blocking) when `capture_thinking=True` and the model invoked zero tools."""
    content: str
    thought: str


T = TypeVar('T')


@dataclass
class TokiToolsResponse(Generic[T]):
    """Returned (blocking) when the model invoked at least one tool, `capture_thinking=False`.

    Element type `T` of `tool_calls` reflects the agent's tools shape:
    - `WithStaticTools`     -> `T = TokiToolCall`
    - `WithStreamingTools`  -> `T = TokiToolCallStream` (or `AsyncTokiToolCallStream` for `acomplete`)
    - `WithMixedTools`      -> `T = TokiToolCall | TokiToolCallStream` (async: `... | AsyncTokiToolCallStream`)
    """
    content: str
    tool_calls: list[T]


@dataclass
class TokiToolsThoughtResponse(TokiToolsResponse[T]):
    """Same as `TokiToolsResponse[T]` plus a `thought` field. Returned when `capture_thinking=True`."""
    thought: str = ''


def pretty_tool_call(tool_call: TokiToolCall) -> str:
    """Return a string representation of the tool call, i.e. `tool_name(arg1=value1, arg2=value2, ...)`"""
    args_str = ", ".join(f"{k}={v}" for k, v in tool_call.function.arguments.items())
    return f'{tool_call.function.name}({args_str})'


# --- Internal raw-event types backends produce -------------------------------

@dataclass
class _RawTurn:
    """Full assistant turn from a non-streaming API call. Backends build this in `_raw_blocking`."""
    content: str
    tool_calls: list[TokiToolCall]
    thought: str
    usage: TokiUsageMetadata | None = None


@dataclass
class _RawContentChunk:
    text: str


@dataclass
class _RawThoughtChunk:
    text: str


@dataclass
class _RawToolCallChunk:
    """A streaming delta for one tool call. `id` and `name` are populated only on the
    first chunk for a given `index`; `arguments_fragment` carries args JSON text as it
    arrives."""
    index: int
    id: str | None = None
    name: str | None = None
    arguments_fragment: str | None = None


@dataclass
class _RawUsage:
    usage: TokiUsageMetadata


_RawChunk = _RawContentChunk | _RawThoughtChunk | _RawToolCallChunk | _RawUsage


# --- Stream core + drivers ---------------------------------------------------

@dataclass
class _ToolState:
    index: int
    id: str
    name: str
    is_streaming: bool
    parser: JsonStreamParser = field(default_factory=JsonStreamParser)
    stream: _ToolCallStreamStateBase | None = None
    finalized: bool = False


class _StreamCore:
    """I/O-agnostic translation from `_RawChunk` events into user-facing items.

    Owns the outer-event queue, per-tool-call parsing state, and the
    finalization logic. A driver (`_SyncStreamDriver` / `_AsyncStreamDriver`)
    pumps raw chunks into `feed()` and reads items off `_outer`. The driver is
    `attach()`ed after construction so the core can pass a reference to it when
    instantiating `TokiToolCallStream` / `AsyncTokiToolCallStream` for streaming
    tool calls.
    """

    def __init__(
        self,
        *,
        streaming_names: set[str],
        capture_thinking: bool,
        on_usage: Any,  # callable(TokiUsageMetadata) -> None
    ) -> None:
        self._streaming_names = streaming_names
        self._capture_thinking = capture_thinking
        self._on_usage = on_usage
        self._outer: deque = deque()
        self._tool_state: dict[int, _ToolState] = {}
        # set in attach()
        self._driver: Any = None
        self._tool_call_stream_cls: type = TokiToolCallStream

    def attach(self, driver: Any, tool_call_stream_cls: type) -> None:
        self._driver = driver
        self._tool_call_stream_cls = tool_call_stream_cls

    def feed(self, chunk: _RawChunk) -> None:
        if isinstance(chunk, _RawContentChunk):
            if chunk.text:
                self._outer.append(chunk.text)
            return
        if isinstance(chunk, _RawThoughtChunk):
            if self._capture_thinking and chunk.text:
                self._outer.append(TokiThinking(text=chunk.text))
            return
        if isinstance(chunk, _RawToolCallChunk):
            self._process_tool_chunk(chunk)
            return
        if isinstance(chunk, _RawUsage):
            self._on_usage(chunk.usage)
            return

    def _process_tool_chunk(self, chunk: _RawToolCallChunk) -> None:
        idx = chunk.index
        state = self._tool_state.get(idx)
        if state is None:
            if chunk.name is None:
                raise ValueError(f"first tool-call chunk for index {idx} missing name")
            is_streaming = chunk.name in self._streaming_names
            state = _ToolState(
                index=idx,
                id=chunk.id or f"toki-tool-{uuid4().hex}",
                name=chunk.name,
                is_streaming=is_streaming,
            )
            if is_streaming:
                state.stream = self._tool_call_stream_cls(driver=self._driver, id=state.id, name=state.name)
                self._outer.append(state.stream)
            self._tool_state[idx] = state

        if chunk.arguments_fragment:
            self._feed_args(state, chunk.arguments_fragment)

    def _feed_args(self, state: _ToolState, fragment: str) -> None:
        for ev in state.parser.feed(fragment):
            if state.is_streaming and state.stream is not None:
                state.stream._handle_event(ev)
            if ev[0] == 'done':
                state.finalized = True
                if not state.is_streaming:
                    tc = TokiToolCall(
                        id=state.id,
                        function=TokiToolFunction(name=state.name, arguments=ev[1]),
                    )
                    self._outer.append(tc)

    def finalize(self) -> None:
        """Called by the driver when the source iterator is exhausted; flushes
        any partially-parsed tool calls and signals streams whose source ended."""
        for state in self._tool_state.values():
            if state.finalized:
                continue
            try:
                # if the parser already saw the final '}', flush is a no-op; otherwise it raises
                for ev in state.parser.flush():
                    if state.is_streaming and state.stream is not None:
                        state.stream._handle_event(ev)
                    if ev[0] == 'done':
                        state.finalized = True
                        if not state.is_streaming:
                            tc = TokiToolCall(
                                id=state.id,
                                function=TokiToolFunction(name=state.name, arguments=ev[1]),
                            )
                            self._outer.append(tc)
            except ValueError:
                pass
            if state.is_streaming and state.stream is not None and not state.finalized:
                state.stream._mark_source_ended()


class _SyncStreamDriver:
    """Sync transport: pulls from an `Iterator[_RawChunk]` and feeds the core."""

    def __init__(self, source: Iterator[_RawChunk], core: _StreamCore) -> None:
        self._source = source
        self._core = core
        self._exhausted = False
        core.attach(self, TokiToolCallStream)

    def advance(self) -> bool:
        if self._exhausted:
            return False
        try:
            chunk = next(self._source)
        except StopIteration:
            self._core.finalize()
            self._exhausted = True
            return False
        self._core.feed(chunk)
        return True

    def outer_generator(self) -> Generator:
        last_stream: _ToolCallStreamStateBase | None = None
        while True:
            # auto-drain any previously-yielded streaming tool call
            if last_stream is not None and not last_stream._done:
                while not last_stream._done:
                    if not self.advance():
                        break
            last_stream = None
            if self._core._outer:
                item = self._core._outer.popleft()
                if isinstance(item, _ToolCallStreamStateBase):
                    last_stream = item
                yield item
                continue
            if self._exhausted:
                return
            if not self.advance():
                # may have produced events on final advance via finalize()
                continue


class _AsyncStreamDriver:
    """Async transport: awaits an `AsyncIterator[_RawChunk]` and feeds the core."""

    def __init__(self, source: AsyncIterator[_RawChunk], core: _StreamCore) -> None:
        self._source = source
        self._core = core
        self._exhausted = False
        core.attach(self, AsyncTokiToolCallStream)

    async def advance(self) -> bool:
        if self._exhausted:
            return False
        try:
            chunk = await self._source.__anext__()
        except StopAsyncIteration:
            self._core.finalize()
            self._exhausted = True
            return False
        self._core.feed(chunk)
        return True

    async def outer_agen(self) -> AsyncGenerator:
        last_stream: _ToolCallStreamStateBase | None = None
        while True:
            if last_stream is not None and not last_stream._done:
                while not last_stream._done:
                    if not await self.advance():
                        break
            last_stream = None
            if self._core._outer:
                item = self._core._outer.popleft()
                if isinstance(item, _ToolCallStreamStateBase):
                    last_stream = item
                yield item
                continue
            if self._exhausted:
                return
            if not await self.advance():
                continue


# --- Schema helpers ----------------------------------------------------------

# Shared `tools=` annotation used by `complete()` / `acomplete()` overloads and
# by every backend's `count_tokens` / `acount_tokens`. Mirrors the four shapes
# the overload table covers (none, all-streaming, all-static, mixed).
ToolsArg = (
    Sequence[StreamingToolSchema]
    | Sequence[ToolSchema | dict]
    | Sequence[StreamingToolSchema | ToolSchema | dict]
    | None
)


def _unwrap_tools(tools: Sequence | None) -> tuple[list[dict] | None, set[str]]:
    """Translate a list of `ToolSchema` / `StreamingToolSchema` / raw dicts into a
    plain `list[dict]` (the wire format) and the set of streaming-tool names.
    """
    if tools is None:
        return None, set()
    raw_tools: list[dict] = []
    streaming_names: set[str] = set()
    for t in tools:
        if isinstance(t, StreamingToolSchema):
            raw_tools.append(t.schema)
            streaming_names.add(t.schema["function"]["name"])
        elif isinstance(t, ToolSchema):
            raw_tools.append(t.schema)
        elif isinstance(t, dict):
            raw_tools.append(t)
        else:
            raise TypeError(f"unknown tool schema entry: {type(t).__name__}")
    return raw_tools, streaming_names


# --- BaseModel ---------------------------------------------------------------

class BaseModel(ABC):
    """Abstract base for all toki model backends.

    Backends implement four raw-I/O methods: `_raw_blocking` / `_raw_streaming`
    for sync, and `_raw_blocking_async` / `_raw_streaming_async` for async. The
    base class owns:
    - schema unwrapping (`ToolSchema` / `StreamingToolSchema` / raw dict -> wire dict)
    - typed-response construction for both blocking paths
    - the `_StreamCore` + sync/async drivers that turn raw `_RawChunk` events
      into the public yield types
    - all 16 typing overloads on the public `complete()` and `acomplete()`
      entry points
    """

    def __init__(self) -> None:
        # updated after every completion
        self._usage_metadata: TokiUsageMetadata | None = None

    # ----- abstract raw-I/O methods ------------------------------------------

    @abstractmethod
    def _raw_blocking(
        self,
        messages: list[TokiMessage],
        tools: list[dict] | None,
        *,
        capture_thinking: bool,
        **kwargs,
    ) -> _RawTurn: ...

    @abstractmethod
    def _raw_streaming(
        self,
        messages: list[TokiMessage],
        tools: list[dict] | None,
        *,
        capture_thinking: bool,
        **kwargs,
    ) -> Iterator[_RawChunk]: ...

    @abstractmethod
    async def _raw_blocking_async(
        self,
        messages: list[TokiMessage],
        tools: list[dict] | None,
        *,
        capture_thinking: bool,
        **kwargs,
    ) -> _RawTurn: ...

    @abstractmethod
    def _raw_streaming_async(
        self,
        messages: list[TokiMessage],
        tools: list[dict] | None,
        *,
        capture_thinking: bool,
        **kwargs,
    ) -> AsyncIterator[_RawChunk]: ...

    # ----- token counting ----------------------------------------------------

    @abstractmethod
    def count_tokens(
        self,
        messages: list[TokiMessage | dict],
        *,
        tools: ToolsArg = None,
        kind: Literal['exact'] = 'exact',
    ) -> int | TokenCountEstimate:
        """Count the prompt tokens for the given messages (and tools).

        Returns a plain `int` for exact counts and a `TokenCountEstimate` for
        backends that can only estimate (e.g. an offline path that runs a
        heuristic tokenizer). The abstract signature only advertises
        `kind='exact'`; backends widen the literal to expose any additional
        modes (`'offline'`, `'online'`) they support.
        """
        ...

    async def acount_tokens(
        self,
        messages: list[TokiMessage | dict],
        *,
        tools: ToolsArg = None,
        kind: Literal['exact'] = 'exact',
    ) -> int | TokenCountEstimate:
        """Async sibling of `count_tokens`. Default implementation just calls
        the sync version; backends with a real async path override."""
        return self.count_tokens(messages, tools=tools, kind=kind)

    # ----- public `complete` overloads ---------------------------------------

    # blocking, no thinking
    @overload
    def complete(self, messages: list[TokiMessage | dict], *, stream: Literal[False] = False, tools: None = None, capture_thinking: Literal[False] = False, **kwargs) -> str: ...
    @overload
    def complete(self, messages: list[TokiMessage | dict], *, stream: Literal[False] = False, tools: Sequence[StreamingToolSchema], capture_thinking: Literal[False] = False, **kwargs) -> str | TokiToolsResponse[TokiToolCallStream]: ...
    @overload
    def complete(self, messages: list[TokiMessage | dict], *, stream: Literal[False] = False, tools: Sequence[ToolSchema | dict], capture_thinking: Literal[False] = False, **kwargs) -> str | TokiToolsResponse[TokiToolCall]: ...
    @overload
    def complete(self, messages: list[TokiMessage | dict], *, stream: Literal[False] = False, tools: Sequence[StreamingToolSchema | ToolSchema | dict], capture_thinking: Literal[False] = False, **kwargs) -> str | TokiToolsResponse[TokiToolCall | TokiToolCallStream]: ...
    # blocking, capture_thinking
    @overload
    def complete(self, messages: list[TokiMessage | dict], *, stream: Literal[False] = False, tools: None = None, capture_thinking: Literal[True], **kwargs) -> TokiThoughtResponse: ...
    @overload
    def complete(self, messages: list[TokiMessage | dict], *, stream: Literal[False] = False, tools: Sequence[StreamingToolSchema], capture_thinking: Literal[True], **kwargs) -> TokiThoughtResponse | TokiToolsThoughtResponse[TokiToolCallStream]: ...
    @overload
    def complete(self, messages: list[TokiMessage | dict], *, stream: Literal[False] = False, tools: Sequence[ToolSchema | dict], capture_thinking: Literal[True], **kwargs) -> TokiThoughtResponse | TokiToolsThoughtResponse[TokiToolCall]: ...
    @overload
    def complete(self, messages: list[TokiMessage | dict], *, stream: Literal[False] = False, tools: Sequence[StreamingToolSchema | ToolSchema | dict], capture_thinking: Literal[True], **kwargs) -> TokiThoughtResponse | TokiToolsThoughtResponse[TokiToolCall | TokiToolCallStream]: ...
    # streaming, no thinking
    @overload
    def complete(self, messages: list[TokiMessage | dict], *, stream: Literal[True], tools: None = None, capture_thinking: Literal[False] = False, **kwargs) -> Generator[str, None, None]: ...
    @overload
    def complete(self, messages: list[TokiMessage | dict], *, stream: Literal[True], tools: Sequence[StreamingToolSchema], capture_thinking: Literal[False] = False, **kwargs) -> Generator[str | TokiToolCallStream, None, None]: ...
    @overload
    def complete(self, messages: list[TokiMessage | dict], *, stream: Literal[True], tools: Sequence[ToolSchema | dict], capture_thinking: Literal[False] = False, **kwargs) -> Generator[str | TokiToolCall, None, None]: ...
    @overload
    def complete(self, messages: list[TokiMessage | dict], *, stream: Literal[True], tools: Sequence[StreamingToolSchema | ToolSchema | dict], capture_thinking: Literal[False] = False, **kwargs) -> Generator[str | TokiToolCall | TokiToolCallStream, None, None]: ...
    # streaming, capture_thinking
    @overload
    def complete(self, messages: list[TokiMessage | dict], *, stream: Literal[True], tools: None = None, capture_thinking: Literal[True], **kwargs) -> Generator[str | TokiThinking, None, None]: ...
    @overload
    def complete(self, messages: list[TokiMessage | dict], *, stream: Literal[True], tools: Sequence[StreamingToolSchema], capture_thinking: Literal[True], **kwargs) -> Generator[str | TokiThinking | TokiToolCallStream, None, None]: ...
    @overload
    def complete(self, messages: list[TokiMessage | dict], *, stream: Literal[True], tools: Sequence[ToolSchema | dict], capture_thinking: Literal[True], **kwargs) -> Generator[str | TokiThinking | TokiToolCall, None, None]: ...
    @overload
    def complete(self, messages: list[TokiMessage | dict], *, stream: Literal[True], tools: Sequence[StreamingToolSchema | ToolSchema | dict], capture_thinking: Literal[True], **kwargs) -> Generator[str | TokiThinking | TokiToolCall | TokiToolCallStream, None, None]: ...

    def complete(
        self,
        messages: list[TokiMessage | dict],
        *,
        stream: bool = False,
        tools: Sequence | None = None,
        capture_thinking: bool = False,
        **kwargs,
    ):
        normalized = [TokiMessage.from_dict(m) for m in messages]
        wire_tools, streaming_names = _unwrap_tools(tools)
        if stream:
            source = self._raw_streaming(normalized, wire_tools, capture_thinking=capture_thinking, **kwargs)
            core = _StreamCore(
                streaming_names=streaming_names,
                capture_thinking=capture_thinking,
                on_usage=self._record_usage,
            )
            driver = _SyncStreamDriver(source, core)
            return driver.outer_generator()
        return self._build_blocking_response(
            self._raw_blocking(normalized, wire_tools, capture_thinking=capture_thinking, **kwargs),
            streaming_names=streaming_names,
            capture_thinking=capture_thinking,
            async_mode=False,
        )

    # ----- public `acomplete` overloads --------------------------------------

    # blocking, no thinking
    @overload
    def acomplete(self, messages: list[TokiMessage | dict], *, stream: Literal[False] = False, tools: None = None, capture_thinking: Literal[False] = False, **kwargs) -> Coroutine[Any, Any, str]: ...
    @overload
    def acomplete(self, messages: list[TokiMessage | dict], *, stream: Literal[False] = False, tools: Sequence[StreamingToolSchema], capture_thinking: Literal[False] = False, **kwargs) -> Coroutine[Any, Any, str | TokiToolsResponse[AsyncTokiToolCallStream]]: ...
    @overload
    def acomplete(self, messages: list[TokiMessage | dict], *, stream: Literal[False] = False, tools: Sequence[ToolSchema | dict], capture_thinking: Literal[False] = False, **kwargs) -> Coroutine[Any, Any, str | TokiToolsResponse[TokiToolCall]]: ...
    @overload
    def acomplete(self, messages: list[TokiMessage | dict], *, stream: Literal[False] = False, tools: Sequence[StreamingToolSchema | ToolSchema | dict], capture_thinking: Literal[False] = False, **kwargs) -> Coroutine[Any, Any, str | TokiToolsResponse[TokiToolCall | AsyncTokiToolCallStream]]: ...
    # blocking, capture_thinking
    @overload
    def acomplete(self, messages: list[TokiMessage | dict], *, stream: Literal[False] = False, tools: None = None, capture_thinking: Literal[True], **kwargs) -> Coroutine[Any, Any, TokiThoughtResponse]: ...
    @overload
    def acomplete(self, messages: list[TokiMessage | dict], *, stream: Literal[False] = False, tools: Sequence[StreamingToolSchema], capture_thinking: Literal[True], **kwargs) -> Coroutine[Any, Any, TokiThoughtResponse | TokiToolsThoughtResponse[AsyncTokiToolCallStream]]: ...
    @overload
    def acomplete(self, messages: list[TokiMessage | dict], *, stream: Literal[False] = False, tools: Sequence[ToolSchema | dict], capture_thinking: Literal[True], **kwargs) -> Coroutine[Any, Any, TokiThoughtResponse | TokiToolsThoughtResponse[TokiToolCall]]: ...
    @overload
    def acomplete(self, messages: list[TokiMessage | dict], *, stream: Literal[False] = False, tools: Sequence[StreamingToolSchema | ToolSchema | dict], capture_thinking: Literal[True], **kwargs) -> Coroutine[Any, Any, TokiThoughtResponse | TokiToolsThoughtResponse[TokiToolCall | AsyncTokiToolCallStream]]: ...
    # streaming, no thinking
    @overload
    def acomplete(self, messages: list[TokiMessage | dict], *, stream: Literal[True], tools: None = None, capture_thinking: Literal[False] = False, **kwargs) -> AsyncGenerator[str, None]: ...
    @overload
    def acomplete(self, messages: list[TokiMessage | dict], *, stream: Literal[True], tools: Sequence[StreamingToolSchema], capture_thinking: Literal[False] = False, **kwargs) -> AsyncGenerator[str | AsyncTokiToolCallStream, None]: ...
    @overload
    def acomplete(self, messages: list[TokiMessage | dict], *, stream: Literal[True], tools: Sequence[ToolSchema | dict], capture_thinking: Literal[False] = False, **kwargs) -> AsyncGenerator[str | TokiToolCall, None]: ...
    @overload
    def acomplete(self, messages: list[TokiMessage | dict], *, stream: Literal[True], tools: Sequence[StreamingToolSchema | ToolSchema | dict], capture_thinking: Literal[False] = False, **kwargs) -> AsyncGenerator[str | TokiToolCall | AsyncTokiToolCallStream, None]: ...
    # streaming, capture_thinking
    @overload
    def acomplete(self, messages: list[TokiMessage | dict], *, stream: Literal[True], tools: None = None, capture_thinking: Literal[True], **kwargs) -> AsyncGenerator[str | TokiThinking, None]: ...
    @overload
    def acomplete(self, messages: list[TokiMessage | dict], *, stream: Literal[True], tools: Sequence[StreamingToolSchema], capture_thinking: Literal[True], **kwargs) -> AsyncGenerator[str | TokiThinking | AsyncTokiToolCallStream, None]: ...
    @overload
    def acomplete(self, messages: list[TokiMessage | dict], *, stream: Literal[True], tools: Sequence[ToolSchema | dict], capture_thinking: Literal[True], **kwargs) -> AsyncGenerator[str | TokiThinking | TokiToolCall, None]: ...
    @overload
    def acomplete(self, messages: list[TokiMessage | dict], *, stream: Literal[True], tools: Sequence[StreamingToolSchema | ToolSchema | dict], capture_thinking: Literal[True], **kwargs) -> AsyncGenerator[str | TokiThinking | TokiToolCall | AsyncTokiToolCallStream, None]: ...

    def acomplete(
        self,
        messages: list[TokiMessage | dict],
        *,
        stream: bool = False,
        tools: Sequence | None = None,
        capture_thinking: bool = False,
        **kwargs,
    ):
        normalized = [TokiMessage.from_dict(m) for m in messages]
        wire_tools, streaming_names = _unwrap_tools(tools)
        if stream:
            source = self._raw_streaming_async(normalized, wire_tools, capture_thinking=capture_thinking, **kwargs)
            core = _StreamCore(
                streaming_names=streaming_names,
                capture_thinking=capture_thinking,
                on_usage=self._record_usage,
            )
            driver = _AsyncStreamDriver(source, core)
            return driver.outer_agen()
        return self._acomplete_blocking(normalized, wire_tools, streaming_names=streaming_names, capture_thinking=capture_thinking, **kwargs)

    async def _acomplete_blocking(
        self,
        messages: list[TokiMessage],
        wire_tools: list[dict] | None,
        *,
        streaming_names: set[str],
        capture_thinking: bool,
        **kwargs,
    ):
        turn = await self._raw_blocking_async(messages, wire_tools, capture_thinking=capture_thinking, **kwargs)
        return self._build_blocking_response(
            turn,
            streaming_names=streaming_names,
            capture_thinking=capture_thinking,
            async_mode=True,
        )

    # ----- internals --------------------------------------------------------

    def _record_usage(self, usage: TokiUsageMetadata) -> None:
        self._usage_metadata = usage

    def _build_blocking_response(
        self,
        turn: _RawTurn,
        *,
        streaming_names: set[str],
        capture_thinking: bool,
        async_mode: bool,
    ):
        if turn.usage is not None:
            self._usage_metadata = turn.usage

        if not turn.tool_calls:
            if capture_thinking:
                return TokiThoughtResponse(content=turn.content, thought=turn.thought)
            return turn.content

        # at least one tool was invoked. wrap streaming-flagged ones into pre-drained
        # `TokiToolCallStream`s (or async siblings) so the surface matches the streaming case.
        builder = _async_prebuilt_stream_from_tool_call if async_mode else _prebuilt_stream_from_tool_call
        tool_calls_out: list = []
        for tc in turn.tool_calls:
            if tc.function.name in streaming_names:
                tool_calls_out.append(builder(tc))
            else:
                tool_calls_out.append(tc)

        if capture_thinking:
            return TokiToolsThoughtResponse(content=turn.content, tool_calls=tool_calls_out, thought=turn.thought)
        return TokiToolsResponse(content=turn.content, tool_calls=tool_calls_out)


# --- helpers used internally -------------------------------------------------

class _PrebuiltSyncDriver:
    """Stand-in sync driver for pre-drained `TokiToolCallStream`s in blocking responses.
    Has no live source, so `advance()` is a no-op returning False."""

    def advance(self) -> bool:
        return False


class _PrebuiltAsyncDriver:
    """Async sibling of `_PrebuiltSyncDriver`."""

    async def advance(self) -> bool:
        return False


_PREBUILT_SYNC = _PrebuiltSyncDriver()
_PREBUILT_ASYNC = _PrebuiltAsyncDriver()


def _prebuilt_stream_from_tool_call(tc: TokiToolCall) -> TokiToolCallStream:
    """Build a `TokiToolCallStream` whose state is pre-populated from a finished
    `TokiToolCall`. All `expect_arg` / `items()` calls become single-shot replays.
    """
    s = TokiToolCallStream(driver=_PREBUILT_SYNC, id=tc.id, name=tc.function.name)
    args = tc.function.arguments
    s._args_order = list(args.keys())
    for k, v in args.items():
        s._args_chunks[k] = [json.dumps(v) if not isinstance(v, str) else v]
        s._args_values[k] = v
    s._dict = dict(args)
    s._done = True
    return s


def _async_prebuilt_stream_from_tool_call(tc: TokiToolCall) -> AsyncTokiToolCallStream:
    """Async sibling of `_prebuilt_stream_from_tool_call`."""
    s = AsyncTokiToolCallStream(driver=_PREBUILT_ASYNC, id=tc.id, name=tc.function.name)
    args = tc.function.arguments
    s._args_order = list(args.keys())
    for k, v in args.items():
        s._args_chunks[k] = [json.dumps(v) if not isinstance(v, str) else v]
        s._args_values[k] = v
    s._dict = dict(args)
    s._done = True
    return s
