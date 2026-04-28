"""Streaming JSON parser for top-level argument objects (tool-call args).

Yields tagged events as input is fed via `feed()`. Top-level string values stream
as post-decoded characters via `arg_chunk` events; other top-level value types
(numbers, booleans, null, arrays, objects) stream as raw JSON text via
`arg_chunk`. The parsed Python value of each arg is delivered in `arg_end`, and
the full parsed object in `done`.

Strict mode only: any malformed input raises `ValueError` at the offending feed.
"""

import json
from typing import Any, Iterator, Literal


_StreamEvent = (
    tuple[Literal['arg_start'], str]
    | tuple[Literal['arg_chunk'], str]
    | tuple[Literal['arg_end'], Any]
    | tuple[Literal['done'], dict]
)


_SIMPLE_ESCAPES = {
    '"': '"', '\\': '\\', '/': '/',
    'b': '\b', 'f': '\f', 'n': '\n', 'r': '\r', 't': '\t',
}


# all states: start | key_or_close | key_expecting | key | key_escape | key_unicode |
#             colon | value_start | string_value | string_value_escape | string_value_unicode |
#             nested_value | simple_value | value_end | done
class JsonStreamParser:
    """Streaming parser for one top-level JSON object (the args dict)."""

    def __init__(self) -> None:
        self._mode: str = 'start'
        # current key chars (decoded)
        self._key: list[str] = []
        # raw chars of the current value (used for json.loads at arg_end)
        self._raw: list[str] = []
        # accumulated hex chars for a pending \uXXXX escape
        self._unicode: list[str] = []
        # nested-value tracking
        self._depth: int = 0
        self._n_str: bool = False
        self._n_esc: bool = False
        # accumulated parsed result
        self._result: dict = {}
        # name of the key whose value we are currently consuming
        self._current_key: str = ''

    def feed(self, chunk: str) -> Iterator[_StreamEvent]:
        for ch in chunk:
            yield from self._step(ch)

    def flush(self) -> Iterator[_StreamEvent]:
        if self._mode != 'done':
            raise ValueError(f"unexpected end of JSON stream in state {self._mode!r}")
        return
        yield  # pragma: no cover  (generator marker)

    @property
    def done(self) -> bool:
        return self._mode == 'done'

    def _step(self, ch: str) -> Iterator[_StreamEvent]:
        m = self._mode

        if m == 'start':
            if ch.isspace():
                return
            if ch == '{':
                self._mode = 'key_or_close'
                return
            raise ValueError(f"expected '{{' at start, got {ch!r}")

        if m == 'key_or_close':
            if ch.isspace():
                return
            if ch == '"':
                self._mode = 'key'
                self._key = []
                return
            if ch == '}':
                self._mode = 'done'
                yield ('done', self._result)
                return
            raise ValueError(f"expected '\"' or '}}' at key position, got {ch!r}")

        if m == 'key_expecting':
            if ch.isspace():
                return
            if ch == '"':
                self._mode = 'key'
                self._key = []
                return
            raise ValueError(f"expected '\"' at key position, got {ch!r}")

        if m == 'key':
            if ch == '\\':
                self._mode = 'key_escape'
                return
            if ch == '"':
                self._current_key = ''.join(self._key)
                self._key = []
                self._mode = 'colon'
                return
            self._key.append(ch)
            return

        if m == 'key_escape':
            if ch == 'u':
                self._unicode = []
                self._mode = 'key_unicode'
                return
            if ch in _SIMPLE_ESCAPES:
                self._key.append(_SIMPLE_ESCAPES[ch])
                self._mode = 'key'
                return
            raise ValueError(f"invalid escape sequence '\\{ch}' in key")

        if m == 'key_unicode':
            self._unicode.append(ch)
            if len(self._unicode) == 4:
                try:
                    code = int(''.join(self._unicode), 16)
                except ValueError:
                    raise ValueError(f"invalid unicode escape '\\u{''.join(self._unicode)}'") from None
                self._key.append(chr(code))
                self._unicode = []
                self._mode = 'key'
            return

        if m == 'colon':
            if ch.isspace():
                return
            if ch == ':':
                self._mode = 'value_start'
                yield ('arg_start', self._current_key)
                return
            raise ValueError(f"expected ':' after key, got {ch!r}")

        if m == 'value_start':
            if ch.isspace():
                return
            self._raw = []
            if ch == '"':
                self._raw.append(ch)
                self._mode = 'string_value'
                return
            if ch == '[' or ch == '{':
                self._raw.append(ch)
                self._depth = 1
                self._n_str = False
                self._n_esc = False
                self._mode = 'nested_value'
                yield ('arg_chunk', ch)
                return
            self._raw.append(ch)
            self._mode = 'simple_value'
            yield ('arg_chunk', ch)
            return

        if m == 'string_value':
            self._raw.append(ch)
            if ch == '\\':
                self._mode = 'string_value_escape'
                return
            if ch == '"':
                value = json.loads(''.join(self._raw))
                self._result[self._current_key] = value
                self._mode = 'value_end'
                yield ('arg_end', value)
                return
            yield ('arg_chunk', ch)
            return

        if m == 'string_value_escape':
            self._raw.append(ch)
            if ch == 'u':
                self._unicode = []
                self._mode = 'string_value_unicode'
                return
            if ch in _SIMPLE_ESCAPES:
                yield ('arg_chunk', _SIMPLE_ESCAPES[ch])
                self._mode = 'string_value'
                return
            raise ValueError(f"invalid escape sequence '\\{ch}' in string value")

        if m == 'string_value_unicode':
            self._raw.append(ch)
            self._unicode.append(ch)
            if len(self._unicode) == 4:
                try:
                    code = int(''.join(self._unicode), 16)
                except ValueError:
                    raise ValueError(f"invalid unicode escape '\\u{''.join(self._unicode)}'") from None
                self._unicode = []
                self._mode = 'string_value'
                yield ('arg_chunk', chr(code))
            return

        if m == 'nested_value':
            self._raw.append(ch)
            yield ('arg_chunk', ch)
            if self._n_str:
                if self._n_esc:
                    self._n_esc = False
                    return
                if ch == '\\':
                    self._n_esc = True
                    return
                if ch == '"':
                    self._n_str = False
                    return
                return
            if ch == '"':
                self._n_str = True
                return
            if ch == '[' or ch == '{':
                self._depth += 1
                return
            if ch == ']' or ch == '}':
                self._depth -= 1
                if self._depth == 0:
                    value = json.loads(''.join(self._raw))
                    self._result[self._current_key] = value
                    self._mode = 'value_end'
                    yield ('arg_end', value)
                return
            return

        if m == 'simple_value':
            if ch in ' \t\n\r,}':
                value = json.loads(''.join(self._raw).strip())
                self._result[self._current_key] = value
                self._mode = 'value_end'
                yield ('arg_end', value)
                yield from self._step(ch)
                return
            self._raw.append(ch)
            yield ('arg_chunk', ch)
            return

        if m == 'value_end':
            if ch.isspace():
                return
            if ch == ',':
                self._mode = 'key_expecting'
                return
            if ch == '}':
                self._mode = 'done'
                yield ('done', self._result)
                return
            raise ValueError(f"expected ',' or '}}' after value, got {ch!r}")

        if m == 'done':
            if ch.isspace():
                return
            raise ValueError(f"unexpected character {ch!r} after end of object")

        raise AssertionError(f"unhandled state {m!r}")
