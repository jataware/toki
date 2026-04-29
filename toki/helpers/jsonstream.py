"""
Streaming JSON parser, pull-based.

Hand `streaming_parse_json` an iterable of string chunks; it identifies the
type of the next JSON value and either:
- returns the parsed primitive (number, bool, null) directly, or
- returns a stream object (`JsonStrStream`, `JsonArrStream`, `JsonDictStream`)
  that you iterate to consume the value's pieces as they arrive.

Stream objects expose:
- iteration — yields decoded chunks (strings), elements (arrays), or
  (key, value) pairs (dicts). A nested value is itself a primitive or a
  sub-stream, recursively.
- `.value` — full parsed Python value. Auto-drains anything you haven't
  consumed yet. After `.value` is accessed, the stream cannot be iterated
  further (RuntimeError).
- `.done` — True once the closing token has been consumed.

If a parent stream advances past an unfinished child stream, the child is
silently auto-drained (no error) and its iteration is then exhausted.

Not thread-safe: the cursor is shared mutable state, so all iteration of a
single top-level call's streams must happen on one thread.
"""

from __future__ import annotations

import json
from typing import Iterable, Iterator, TypeAlias


Primitive: TypeAlias = int | float | bool | None
JsonValue: TypeAlias = "Primitive | JsonStrStream | JsonArrStream | JsonDictStream"


_WS = frozenset(' \t\n\r')
_NUMBER_CHARS = frozenset('0123456789-+.eE')
_STR_STOPS = frozenset('"\\')
_SIMPLE_ESCAPES = {
    '"': '"', '\\': '\\', '/': '/',
    'b': '\b', 'f': '\f', 'n': '\n', 'r': '\r', 't': '\t',
}


# ─── public entry ────────────────────────────────────────────────────────────


def streaming_parse_json(stream: Iterable[str]) -> JsonValue:
    """Parse one JSON value from an iterable of string chunks.

    The iterable is consumed lazily; the returned value (or stream) holds a
    cursor into the same iterable, so child streams keep reading more chunks
    as they're iterated.
    """
    cur = _Cursor(stream)
    cur.skip_ws()
    return _parse_value(cur)


# ─── cursor over a chunked source ────────────────────────────────────────────


class _Cursor:
    """Lazy chunk buffer over an iterable of string chunks."""

    def __init__(self, source: Iterable[str]) -> None:
        self._src: Iterator[str] = iter(source)
        self._buf: str = ''
        self._exhausted: bool = False

    def _pull(self) -> bool:
        if self._exhausted:
            return False
        try:
            chunk = next(self._src)
        except StopIteration:
            self._exhausted = True
            return False
        if chunk:
            self._buf += chunk
        return True

    def peek(self, n: int = 1) -> str:
        while len(self._buf) < n and self._pull():
            pass
        return self._buf[:n]

    def consume(self, n: int) -> str:
        s = self.peek(n)
        self._buf = self._buf[len(s):]
        return s

    def consume_while(self, allowed: frozenset[str]) -> str:
        out: list[str] = []
        while True:
            i = 0
            while i < len(self._buf) and self._buf[i] in allowed:
                i += 1
            if i:
                out.append(self._buf[:i])
                self._buf = self._buf[i:]
            if i < len(self._buf) or not self._pull():
                break
        return ''.join(out)

    def consume_until(self, stops: frozenset[str]) -> str:
        out: list[str] = []
        while True:
            idx = -1
            for s in stops:
                i = self._buf.find(s)
                if i >= 0 and (idx == -1 or i < idx):
                    idx = i
            if idx >= 0:
                out.append(self._buf[:idx])
                self._buf = self._buf[idx:]
                return ''.join(out)
            if self._buf:
                out.append(self._buf)
                self._buf = ''
            if not self._pull():
                return ''.join(out)

    def skip_ws(self) -> None:
        self.consume_while(_WS)

    def expect(self, ch: str) -> None:
        got = self.consume(1)
        if got != ch:
            raise ValueError(f"expected {ch!r}, got {got!r}")


# ─── stream classes ──────────────────────────────────────────────────────────


class _BaseStream:
    """Common state for streamable JSON values (string, array, object)."""

    _value: object  # filled on drain by subclasses

    def __init__(self, cur: _Cursor) -> None:
        self._cur = cur
        self._done = False
        self._value_accessed = False

    @property
    def done(self) -> bool:
        return self._done

    @property
    def value(self):
        if not self._done:
            self._drain_remaining()
        self._value_accessed = True
        return self._value

    def _drain_remaining(self) -> None:
        for _ in self:  # type: ignore[attr-defined]
            pass

    def _check_iter_allowed(self) -> None:
        if self._value_accessed:
            raise RuntimeError(
                f"{type(self).__name__} was drained via .value; cannot continue iterating"
            )


class JsonStrStream(_BaseStream):
    """Iterator over decoded chunks of a JSON string.

    Each `__next__` returns a run of decoded characters (plain content runs
    arrive in one chunk; each escape sequence emits its decoded char as a
    separate chunk). After drain, `.value` is the full decoded `str`.
    """

    def __init__(self, cur: _Cursor) -> None:
        super().__init__(cur)
        cur.expect('"')
        self._chunks: list[str] = []

    def __iter__(self) -> JsonStrStream:
        return self

    def __next__(self) -> str:
        self._check_iter_allowed()
        if self._done:
            raise StopIteration

        run = self._cur.consume_until(_STR_STOPS)
        if run:
            self._chunks.append(run)
            return run

        delim = self._cur.consume(1)
        if not delim:
            raise ValueError("unexpected end of stream while reading string")

        if delim == '"':
            self._value = ''.join(self._chunks)
            self._done = True
            raise StopIteration

        # delim == '\\'
        ec = self._cur.consume(1)
        if not ec:
            raise ValueError("unexpected end of stream after '\\'")
        if ec in _SIMPLE_ESCAPES:
            decoded = _SIMPLE_ESCAPES[ec]
        elif ec == 'u':
            decoded = self._consume_unicode_escape()
        else:
            raise ValueError(f"invalid escape sequence '\\{ec}'")
        self._chunks.append(decoded)
        return decoded

    def _consume_unicode_escape(self) -> str:
        hex4 = self._cur.consume(4)
        if len(hex4) != 4:
            raise ValueError("unexpected end of stream in unicode escape")
        try:
            code = int(hex4, 16)
        except ValueError:
            raise ValueError(f"invalid unicode escape '\\u{hex4}'") from None
        # surrogate-pair handling: a high surrogate must be followed by '\uXXXX'
        # forming a low surrogate; combine into one codepoint
        if 0xD800 <= code <= 0xDBFF:
            if self._cur.consume(2) != '\\u':
                raise ValueError(f"high surrogate \\u{hex4} not followed by low surrogate")
            hex4b = self._cur.consume(4)
            if len(hex4b) != 4:
                raise ValueError("unexpected end of stream in unicode escape")
            try:
                code2 = int(hex4b, 16)
            except ValueError:
                raise ValueError(f"invalid unicode escape '\\u{hex4b}'") from None
            if not (0xDC00 <= code2 <= 0xDFFF):
                raise ValueError(f"high surrogate \\u{hex4} not followed by low surrogate")
            code = 0x10000 + ((code - 0xD800) << 10) + (code2 - 0xDC00)
        return chr(code)


class JsonArrStream(_BaseStream):
    """Iterator over the elements of a JSON array.

    Each yielded element is a `JsonValue` — primitive or a sub-stream.
    Sub-streams are auto-drained when this stream advances past them.
    """

    def __init__(self, cur: _Cursor) -> None:
        super().__init__(cur)
        cur.expect('[')
        self._items: list[JsonValue] = []
        self._first = True
        self._child: _BaseStream | None = None

    def __iter__(self) -> JsonArrStream:
        return self

    def __next__(self) -> JsonValue:
        self._check_iter_allowed()
        if self._done:
            raise StopIteration

        if self._child is not None and not self._child._done:
            self._child._drain_remaining()
        self._child = None

        self._cur.skip_ws()
        nxt = self._cur.peek(1)
        if not nxt:
            raise ValueError("unexpected end of stream while reading array")

        if not self._first:
            if nxt == ']':
                self._cur.consume(1)
                self._finalize()
                raise StopIteration
            if nxt != ',':
                raise ValueError(f"expected ',' or ']' in array, got {nxt!r}")
            self._cur.consume(1)
            self._cur.skip_ws()
        else:
            if nxt == ']':
                self._cur.consume(1)
                self._finalize()
                raise StopIteration
            self._first = False

        item = _parse_value(self._cur)
        if isinstance(item, _BaseStream):
            self._child = item
        self._items.append(item)
        return item

    def _finalize(self) -> None:
        out: list = []
        for it in self._items:
            out.append(it.value if isinstance(it, _BaseStream) else it)
        self._value = out
        self._done = True


class JsonDictStream(_BaseStream):
    """Iterator over the (key, value) pairs of a JSON object.

    Iteration yields `(str, JsonValue)` tuples, where the value is either a
    primitive or a sub-stream. Sub-streams are auto-drained when this stream
    advances past them. `.items()` is provided as a dict-like alias for the
    direct iterator.
    """

    def __init__(self, cur: _Cursor) -> None:
        super().__init__(cur)
        cur.expect('{')
        self._pairs: list[tuple[str, JsonValue]] = []
        self._first = True
        self._child: _BaseStream | None = None

    def __iter__(self) -> JsonDictStream:
        return self

    def items(self) -> JsonDictStream:
        return self

    def __next__(self) -> tuple[str, JsonValue]:
        self._check_iter_allowed()
        if self._done:
            raise StopIteration

        if self._child is not None and not self._child._done:
            self._child._drain_remaining()
        self._child = None

        self._cur.skip_ws()
        nxt = self._cur.peek(1)
        if not nxt:
            raise ValueError("unexpected end of stream while reading object")

        if not self._first:
            if nxt == '}':
                self._cur.consume(1)
                self._finalize()
                raise StopIteration
            if nxt != ',':
                raise ValueError(f"expected ',' or '}}' in object, got {nxt!r}")
            self._cur.consume(1)
            self._cur.skip_ws()
        else:
            if nxt == '}':
                self._cur.consume(1)
                self._finalize()
                raise StopIteration
            self._first = False

        if self._cur.peek(1) != '"':
            raise ValueError(f"expected '\"' for object key, got {self._cur.peek(1)!r}")
        key = JsonStrStream(self._cur).value
        self._cur.skip_ws()
        if self._cur.consume(1) != ':':
            raise ValueError("expected ':' after object key")
        self._cur.skip_ws()
        value = _parse_value(self._cur)
        if isinstance(value, _BaseStream):
            self._child = value
        self._pairs.append((key, value))
        return key, value

    def _finalize(self) -> None:
        out: dict = {}
        for k, v in self._pairs:
            out[k] = v.value if isinstance(v, _BaseStream) else v
        self._value = out
        self._done = True


# ─── primitive consumers and dispatch ────────────────────────────────────────


def _parse_value(cur: _Cursor) -> JsonValue:
    nxt = cur.peek(1)
    if not nxt:
        raise ValueError("unexpected end of stream looking for json value")
    if nxt == '"':
        return JsonStrStream(cur)
    if nxt == '[':
        return JsonArrStream(cur)
    if nxt == '{':
        return JsonDictStream(cur)
    if nxt == 't' or nxt == 'f':
        return _consume_boolean(cur)
    if nxt == 'n':
        return _consume_null(cur)
    if nxt == '-' or nxt.isdigit():
        return _consume_number(cur)
    raise ValueError(f"unexpected character {nxt!r} starting json value")


def _consume_boolean(cur: _Cursor) -> bool:
    s = cur.consume(4)
    if s == 'true':
        return True
    if s == 'fals' and cur.consume(1) == 'e':
        return False
    raise ValueError(f"invalid boolean literal at {s!r}")


def _consume_null(cur: _Cursor) -> None:
    s = cur.consume(4)
    if s != 'null':
        raise ValueError(f"invalid null literal at {s!r}")
    return None


def _consume_number(cur: _Cursor) -> int | float:
    raw = cur.consume_while(_NUMBER_CHARS)
    if not raw:
        raise ValueError("expected number")
    try:
        return json.loads(raw)
    except ValueError as e:
        raise ValueError(f"invalid number literal: {raw!r}") from e


# ─── trash_skipper ───────────────────────────────────────────────────────────


_LITERALS_BY_START = {'t': 'true', 'f': 'false', 'n': 'null'}


def trash_skipper(
    source: Iterable[str],
    look_for: type | tuple[type, ...] = (dict, list),
) -> Iterator[str]:
    """Wrap an iterable of string chunks, yielding only the substrings that form
    the first complete top-level JSON value whose Python type is in `look_for`.
    Leading content before the value's start, and trailing content after the
    value's end, are dropped.

    Heuristic: the wrapper starts at the first character that looks like one of
    the requested JSON types. If the trash itself contains such a character
    (e.g. a stray '{' in prose before the real object), the wrapper will start
    there and likely produce invalid JSON downstream — this is intentional and
    cheap; the parser will surface the malformed input.

    Useful for cleaning LLM responses that wrap a JSON answer in prose or
    markdown fences:

        clean = trash_skipper(response_chunks, look_for=(dict, list))
        result = streaming_parse_json(clean)
    """
    if isinstance(look_for, type):
        look_for = (look_for,)
    starts = _start_chars_for(look_for)
    src = iter(source)

    pending = _skip_to_valid_start(src, starts)
    if not pending:
        return
    first = pending[0]
    if first == '{' or first == '[':
        yield from _stream_balanced(pending, src)
    elif first == '"':
        yield from _stream_string(pending, src)
    elif first in '-0123456789':
        yield from _stream_number(pending, src)
    elif first in _LITERALS_BY_START:
        yield from _stream_literal(pending, src, _LITERALS_BY_START[first])


def _skip_to_valid_start(src: Iterator[str], starts: frozenset[str]) -> str | None:
    """Pull chunks until we find a position that begins a valid JSON value.

    `{`, `[`, `"`, and number-start chars are accepted on first sight (the
    downstream consumer will validate). Bool / null candidates (`t` / `f` /
    `n`) require enough lookahead to confirm the literal — pull more chunks
    if needed, and skip past the candidate if it doesn't actually spell out
    `true` / `false` / `null`.
    """
    pending = ''
    scan = 0
    while True:
        while scan < len(pending):
            c = pending[scan]
            if c not in starts:
                scan += 1
                continue
            if c not in _LITERALS_BY_START:
                return pending[scan:]
            lit = _LITERALS_BY_START[c]
            if len(pending) - scan < len(lit):
                break  # not enough lookahead yet; pull more
            if pending[scan:scan + len(lit)] == lit:
                return pending[scan:]
            scan += 1
        try:
            pending += next(src)
        except StopIteration:
            return None


def _start_chars_for(look_for: tuple[type, ...]) -> frozenset[str]:
    starts: set[str] = set()
    for t in look_for:
        if t is dict:
            starts.add('{')
        elif t is list:
            starts.add('[')
        elif t is str:
            starts.add('"')
        elif t is int or t is float:
            starts |= set('-0123456789')
        elif t is bool:
            starts |= {'t', 'f'}
        elif t is type(None):
            starts.add('n')
        else:
            raise ValueError(f"unsupported look_for type: {t!r}")
    return frozenset(starts)


def _stream_balanced(pending: str, src: Iterator[str]) -> Iterator[str]:
    depth = 0
    in_str = False
    in_escape = False
    while True:
        for i, c in enumerate(pending):
            if in_escape:
                in_escape = False
                continue
            if in_str:
                if c == '\\':
                    in_escape = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == '{' or c == '[':
                depth += 1
            elif c == '}' or c == ']':
                depth -= 1
                if depth == 0:
                    yield pending[:i + 1]
                    return
        yield pending
        try:
            pending = next(src)
        except StopIteration:
            return


def _stream_string(pending: str, src: Iterator[str]) -> Iterator[str]:
    in_escape = False
    skip = 1  # skip the opening quote on the first chunk
    while True:
        for i in range(skip, len(pending)):
            c = pending[i]
            if in_escape:
                in_escape = False
                continue
            if c == '\\':
                in_escape = True
                continue
            if c == '"':
                yield pending[:i + 1]
                return
        yield pending
        skip = 0
        try:
            pending = next(src)
        except StopIteration:
            return


def _stream_number(pending: str, src: Iterator[str]) -> Iterator[str]:
    while True:
        for i, c in enumerate(pending):
            if c not in _NUMBER_CHARS:
                if i:
                    yield pending[:i]
                return
        yield pending
        try:
            pending = next(src)
        except StopIteration:
            return


def _stream_literal(pending: str, src: Iterator[str], lit: str) -> Iterator[str]:
    consumed = 0
    while True:
        need = len(lit) - consumed
        if len(pending) >= need:
            yield pending[:need]
            return
        consumed += len(pending)
        yield pending
        try:
            pending = next(src)
        except StopIteration:
            return
