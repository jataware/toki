"""Tests for the internal tool-call envelope parser (`toki.helpers._jsonstream`).

This parser is the push-fed state machine that backs `TokiToolCallStream` /
`TokiArgStream`. Tests cover:

  - top-level structural rules (single object only, empty object, multiple args)
  - string-arg streaming (decoded chars via `arg_chunk`, escape handling)
  - non-string-arg streaming (raw JSON text via `arg_chunk`, finalized via `arg_end`)
  - whitespace tolerance
  - error reporting (strict mode)
  - chunk-size invariance (events identical regardless of how input is split)
  - the composition contract that load-bears the streaming-tool-call API: the
    raw chunks emitted for a nested-value arg form valid JSON suitable for
    feeding into the public `streaming_parse_json`.

The public, recursive streaming parser is tested separately in
`test_jsonstream.py`.
"""

import pytest

from toki.helpers._jsonstream import JsonStreamParser
from toki.helpers.jsonstream import (
    JsonArrStream,
    JsonDictStream,
    streaming_parse_json,
)


def all_events(parser: JsonStreamParser, text: str) -> list[tuple]:
    """Drain `parser.feed(text)` into a list."""
    return list(parser.feed(text))


def feed_chunked(text: str, size: int) -> list[tuple]:
    """Feed `text` as `size`-char chunks through a fresh parser, return all events."""
    parser = JsonStreamParser()
    events: list[tuple] = []
    for i in range(0, len(text), size):
        events.extend(parser.feed(text[i:i + size]))
    return events


# ─── basic shapes ────────────────────────────────────────────────────────────


def test_empty_object():
    p = JsonStreamParser()
    events = all_events(p, '{}')
    assert events == [('done', {})]
    assert p.done


def test_single_string_arg():
    events = all_events(JsonStreamParser(), '{"city": "Paris"}')
    assert events == [
        ('arg_start', 'city'),
        ('arg_chunk', 'P'),
        ('arg_chunk', 'a'),
        ('arg_chunk', 'r'),
        ('arg_chunk', 'i'),
        ('arg_chunk', 's'),
        ('arg_end', 'Paris'),
        ('done', {'city': 'Paris'}),
    ]


def test_single_number_arg():
    events = all_events(JsonStreamParser(), '{"n": 42}')
    assert events == [
        ('arg_start', 'n'),
        ('arg_chunk', '4'),
        ('arg_chunk', '2'),
        ('arg_end', 42),
        ('done', {'n': 42}),
    ]


@pytest.mark.parametrize("text,expected", [
    ('{"x": true}', True),
    ('{"x": false}', False),
    ('{"x": null}', None),
    ('{"x": 0}', 0),
    ('{"x": -1.5}', -1.5),
    ('{"x": 1e3}', 1000.0),
])
def test_simple_value_types(text, expected):
    events = all_events(JsonStreamParser(), text)
    assert events[0] == ('arg_start', 'x')
    assert events[-2] == ('arg_end', expected)
    assert events[-1] == ('done', {'x': expected})


def test_multiple_args_in_order():
    events = all_events(JsonStreamParser(), '{"a": 1, "b": "two", "c": null}')
    starts = [e for e in events if e[0] == 'arg_start']
    ends = [e for e in events if e[0] == 'arg_end']
    assert starts == [('arg_start', 'a'), ('arg_start', 'b'), ('arg_start', 'c')]
    assert ends == [('arg_end', 1), ('arg_end', 'two'), ('arg_end', None)]
    assert events[-1] == ('done', {'a': 1, 'b': 'two', 'c': None})


# ─── strings: arg_chunk yields decoded characters ───────────────────────────


def test_string_arg_chunks_are_decoded_chars():
    events = all_events(JsonStreamParser(), r'{"s": "a\nb\tc"}')
    chunks = [e[1] for e in events if e[0] == 'arg_chunk']
    # decoded: 'a', '\n', 'b', '\t', 'c'
    assert chunks == ['a', '\n', 'b', '\t', 'c']
    end_event = next(e for e in events if e[0] == 'arg_end')
    assert end_event == ('arg_end', 'a\nb\tc')


def test_string_arg_unicode_escape_decoded():
    events = all_events(JsonStreamParser(), r'{"s": "x\u00e9y"}')
    chunks = [e[1] for e in events if e[0] == 'arg_chunk']
    assert chunks == ['x', 'é', 'y']
    end_event = next(e for e in events if e[0] == 'arg_end')
    assert end_event == ('arg_end', 'xéy')


def test_empty_string_arg():
    events = all_events(JsonStreamParser(), '{"s": ""}')
    chunks = [e for e in events if e[0] == 'arg_chunk']
    assert chunks == []
    assert any(e == ('arg_end', '') for e in events)


def test_string_with_embedded_quote_and_brace():
    # the string contains characters that look like JSON delimiters; the parser must
    # not get confused. embedded chars come through as decoded chunks.
    events = all_events(JsonStreamParser(), r'{"s": "a\"{b}\""}')
    chunks = [e[1] for e in events if e[0] == 'arg_chunk']
    assert ''.join(chunks) == 'a"{b}"'
    end_event = next(e for e in events if e[0] == 'arg_end')
    assert end_event == ('arg_end', 'a"{b}"')


# ─── nested values: arg_chunk yields raw JSON text ──────────────────────────


def test_array_arg_chunks_are_raw_json_text():
    events = all_events(JsonStreamParser(), '{"items": [1, 2, 3]}')
    chunks = [e[1] for e in events if e[0] == 'arg_chunk']
    assert ''.join(chunks) == '[1, 2, 3]'
    end_event = next(e for e in events if e[0] == 'arg_end')
    assert end_event == ('arg_end', [1, 2, 3])


def test_object_arg_chunks_are_raw_json_text():
    events = all_events(JsonStreamParser(), '{"obj": {"a": 1, "b": 2}}')
    chunks = [e[1] for e in events if e[0] == 'arg_chunk']
    assert ''.join(chunks) == '{"a": 1, "b": 2}'
    end_event = next(e for e in events if e[0] == 'arg_end')
    assert end_event == ('arg_end', {'a': 1, 'b': 2})


def test_deeply_nested_value():
    text = '{"x": [1, [2, [3, [4]]]]}'
    events = all_events(JsonStreamParser(), text)
    chunks = [e[1] for e in events if e[0] == 'arg_chunk']
    assert ''.join(chunks) == '[1, [2, [3, [4]]]]'
    end_event = next(e for e in events if e[0] == 'arg_end')
    assert end_event == ('arg_end', [1, [2, [3, [4]]]])


def test_nested_object_with_strings_containing_braces():
    # the depth tracker must skip braces / brackets that appear inside strings.
    text = r'{"x": {"a": "}{][", "b": "\""}}'
    events = all_events(JsonStreamParser(), text)
    end_event = next(e for e in events if e[0] == 'arg_end')
    assert end_event == ('arg_end', {'a': '}{][', 'b': '"'})


# ─── whitespace tolerance ────────────────────────────────────────────────────


def test_whitespace_around_structural_tokens():
    text = '  {  "a"  :  1  ,  "b"  :  "x"  }  '
    events = all_events(JsonStreamParser(), text)
    assert events[-1] == ('done', {'a': 1, 'b': 'x'})


def test_trailing_whitespace_after_done_is_tolerated():
    p = JsonStreamParser()
    list(p.feed('{"a": 1}'))
    assert p.done
    extra = list(p.feed('   \n\t '))
    assert extra == []


# ─── keys ────────────────────────────────────────────────────────────────────


def test_key_with_escapes():
    events = all_events(JsonStreamParser(), r'{"a\nb": 1}')
    starts = [e for e in events if e[0] == 'arg_start']
    assert starts == [('arg_start', 'a\nb')]
    assert events[-1] == ('done', {'a\nb': 1})


def test_key_with_unicode_escape():
    events = all_events(JsonStreamParser(), r'{"caf\u00e9": 1}')
    starts = [e for e in events if e[0] == 'arg_start']
    assert starts == [('arg_start', 'café')]


def test_empty_key_allowed():
    # JSON permits an empty key string, so the envelope parser does too.
    events = all_events(JsonStreamParser(), '{"": 1}')
    assert events[0] == ('arg_start', '')
    assert events[-1] == ('done', {'': 1})


# ─── errors (strict mode) ────────────────────────────────────────────────────


@pytest.mark.parametrize("text,reason", [
    ('[]', 'top-level array'),
    ('42', 'top-level primitive'),
    ('"x"', 'top-level string'),
    ('{1: 2}', 'non-string key'),
    ('{"a" 1}', 'missing colon'),
    ('{"a": 1 "b": 2}', 'missing comma'),
    ('{"a": 1,}', 'trailing comma'),
    (r'{"a": "\q"}', 'invalid escape'),
])
def test_malformed_input_raises(text, reason):
    p = JsonStreamParser()
    with pytest.raises(ValueError):
        list(p.feed(text))


def test_unexpected_char_after_done_raises():
    p = JsonStreamParser()
    list(p.feed('{}'))
    with pytest.raises(ValueError):
        list(p.feed('x'))


# ─── flush / done ────────────────────────────────────────────────────────────


def test_flush_after_done_is_noop():
    p = JsonStreamParser()
    list(p.feed('{"a": 1}'))
    assert p.done
    assert list(p.flush()) == []


def test_flush_before_done_raises():
    p = JsonStreamParser()
    list(p.feed('{"a": 1'))   # mid-value
    assert not p.done
    with pytest.raises(ValueError):
        list(p.flush())


def test_done_starts_false():
    p = JsonStreamParser()
    assert p.done is False
    list(p.feed('{'))
    assert p.done is False


# ─── chunk-size invariance ──────────────────────────────────────────────────


@pytest.mark.parametrize("text", [
    '{"a": 1}',
    '{"city": "Paris", "n": 42}',
    r'{"s": "a\nb\u00e9c"}',
    '{"items": [1, 2, [3, 4]], "obj": {"k": "v"}}',
    '{"x": true, "y": false, "z": null}',
])
@pytest.mark.parametrize("size", [1, 3, 7, 100])
def test_chunk_size_invariance(text, size):
    expected = all_events(JsonStreamParser(), text)
    got = feed_chunked(text, size)
    assert got == expected


# ─── composition with public parser ─────────────────────────────────────────


def test_nested_object_arg_chunks_pipe_into_public_parser():
    # the contract that load-bears the "stream a big nested arg" use case:
    # for a nested-structure arg, the concatenated arg_chunks form valid JSON
    # that streaming_parse_json can consume directly.
    text = '{"items": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]}'
    events = all_events(JsonStreamParser(), text)
    chunks = [e[1] for e in events if e[0] == 'arg_chunk']

    value = streaming_parse_json(iter(chunks))
    assert isinstance(value, JsonArrStream)
    items = []
    for item in value:
        assert isinstance(item, JsonDictStream)
        items.append(item.value)
    assert items == [{'id': 1, 'name': 'a'}, {'id': 2, 'name': 'b'}]


def test_nested_dict_arg_chunks_pipe_into_public_parser():
    text = '{"obj": {"city": "Paris", "n": 42, "tags": ["a", "b"]}}'
    events = all_events(JsonStreamParser(), text)
    chunks = [e[1] for e in events if e[0] == 'arg_chunk']

    value = streaming_parse_json(iter(chunks))
    assert isinstance(value, JsonDictStream)
    assert value.value == {'city': 'Paris', 'n': 42, 'tags': ['a', 'b']}
