"""Tests for the streaming JSON parser (`toki.helpers.jsonstream`).

Each test exercises one design point: primitive parsing, string streaming,
recursive nesting, `.value` semantics, auto-drain behavior, error reporting,
and the `trash_skipper` extraction wrapper.
"""

import pytest

from toki.helpers.jsonstream import (
    JsonArrStream,
    JsonDictStream,
    JsonStrStream,
    streaming_parse_json,
    trash_skipper,
)


def chunked(s: str, size: int = 4):
    """Yield `s` as `size`-char chunks (lazy generator)."""
    for i in range(0, len(s), size):
        yield s[i:i + size]


def by_char(s: str):
    """Yield `s` one character per chunk (worst-case granularity)."""
    return chunked(s, 1)


def whole(s: str):
    """Yield `s` as a single chunk."""
    yield s


# ─── primitives ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("text,expected", [
    ("0", 0),
    ("42", 42),
    ("-42", -42),
    ("1.5", 1.5),
    ("-1.5e2", -150.0),
    ("1E+5", 100000.0),
    ("0.5", 0.5),
    ("true", True),
    ("false", False),
    ("null", None),
])
def test_primitives(text, expected):
    assert streaming_parse_json(whole(text)) == expected


@pytest.mark.parametrize("text,expected", [
    ("42", 42),
    ("-1.5e2", -150.0),
    ("true", True),
    ("false", False),
    ("null", None),
])
def test_primitives_chunked_byte_by_byte(text, expected):
    assert streaming_parse_json(by_char(text)) == expected


# ─── strings ─────────────────────────────────────────────────────────────────


def test_empty_string():
    s = streaming_parse_json(whole('""'))
    assert isinstance(s, JsonStrStream)
    assert s.value == ''


def test_simple_string():
    s = streaming_parse_json(whole('"hello"'))
    chunks = list(s)
    assert ''.join(chunks) == 'hello'
    assert s.value == 'hello'


def test_string_simple_escapes():
    s = streaming_parse_json(whole(r'"a\nb\tc\"d\\e\/f\rg\bh\fi"'))
    assert s.value == 'a\nb\tc"d\\e/f\rg\bh\fi'


def test_string_unicode_escape():
    s = streaming_parse_json(whole(r'"\u00e9"'))
    assert s.value == '\u00e9'


def test_string_surrogate_pair():
    """High + low surrogate must combine into one supplementary-plane char."""
    s = streaming_parse_json(whole(r'"\uD834\uDD1E"'))
    assert s.value == '\U0001D11E'  # musical G clef


def test_string_chunked_byte_by_byte():
    s = streaming_parse_json(by_char('"hello world"'))
    assert ''.join(list(s)) == 'hello world'


# ─── arrays ──────────────────────────────────────────────────────────────────


def test_empty_array():
    a = streaming_parse_json(whole('[]'))
    assert isinstance(a, JsonArrStream)
    assert a.value == []


def test_array_of_primitives():
    a = streaming_parse_json(whole('[1, 2.0, true, null, "hi"]'))
    assert a.value == [1, 2.0, True, None, 'hi']


def test_nested_arrays():
    a = streaming_parse_json(whole('[[1, 2], [3, [4, 5]]]'))
    assert a.value == [[1, 2], [3, [4, 5]]]


@pytest.mark.parametrize("size", [1, 2, 3, 5, 8, 13, 50])
def test_array_chunk_size_invariance(size):
    text = '[1, "two", [3, 4], null, true, -1.5e2]'
    a = streaming_parse_json(chunked(text, size))
    assert a.value == [1, 'two', [3, 4], None, True, -150.0]


# ─── dicts ───────────────────────────────────────────────────────────────────


def test_empty_dict():
    d = streaming_parse_json(whole('{}'))
    assert isinstance(d, JsonDictStream)
    assert d.value == {}


def test_simple_dict():
    d = streaming_parse_json(whole('{"a": 1, "b": "two"}'))
    assert d.value == {'a': 1, 'b': 'two'}


def test_nested_dict():
    d = streaming_parse_json(whole('{"a": {"b": [1, {"c": "d"}]}}'))
    assert d.value == {'a': {'b': [1, {'c': 'd'}]}}


def test_dict_iteration_via_items():
    d = streaming_parse_json(whole('{"a": 1, "b": 2}'))
    pairs = list(d.items())
    assert [k for k, _ in pairs] == ['a', 'b']
    assert [v for _, v in pairs] == [1, 2]


def test_dict_iteration_direct():
    d = streaming_parse_json(whole('{"a": 1, "b": 2}'))
    pairs = list(d)
    assert pairs == [('a', 1), ('b', 2)]


@pytest.mark.parametrize("size", [1, 3, 7, 13, 100])
def test_dict_chunk_size_invariance(size):
    text = '{"city": "Paris", "n": 42, "arr": [1, 2], "deep": {"k": null}}'
    d = streaming_parse_json(chunked(text, size))
    assert d.value == {'city': 'Paris', 'n': 42, 'arr': [1, 2], 'deep': {'k': None}}


# ─── whitespace ──────────────────────────────────────────────────────────────


def test_whitespace_everywhere():
    text = '   {\n  "a"  :  1 ,\n  "b"  :  [ 2 ,  3 ]  }  '
    d = streaming_parse_json(whole(text))
    assert d.value == {'a': 1, 'b': [2, 3]}


# ─── .value semantics ────────────────────────────────────────────────────────


def test_value_drains_string_mid_iteration():
    s = streaming_parse_json(whole('"abcdef"'))
    next(iter(s))                  # consume one chunk
    assert s.value == 'abcdef'      # forces drain of the rest


def test_value_locks_iteration_after_access():
    s = streaming_parse_json(whole('"abc"'))
    s.value
    with pytest.raises(RuntimeError, match="drained via .value"):
        next(s)


def test_value_can_be_called_repeatedly():
    s = streaming_parse_json(whole('"abc"'))
    assert s.value == 'abc'
    assert s.value == 'abc'         # idempotent on a drained stream


def test_done_flag_transitions():
    s = streaming_parse_json(whole('"abc"'))
    assert s.done is False
    list(s)
    assert s.done is True


def test_value_of_parent_drains_unfinished_children():
    d = streaming_parse_json(whole('{"a": [1, 2], "b": "xyz"}'))
    assert d.value == {'a': [1, 2], 'b': 'xyz'}


# ─── auto-drain on parent advance ────────────────────────────────────────────


def test_parent_auto_drains_string_child():
    arr = streaming_parse_json(whole('["abc", 42]'))
    it = iter(arr)
    s = next(it)
    assert isinstance(s, JsonStrStream)
    n = next(it)                    # parent advance auto-drains s
    assert n == 42
    assert s.value == 'abc'


def test_parent_auto_drains_nested_array_child():
    arr = streaming_parse_json(whole('[[1, 2, 3], "x"]'))
    it = iter(arr)
    inner = next(it)
    assert isinstance(inner, JsonArrStream)
    s = next(it)
    assert isinstance(s, JsonStrStream)
    assert inner.value == [1, 2, 3]


def test_iteration_after_parent_auto_drain_yields_stop_iteration():
    """Auto-drain by parent is silent — the dead child returns StopIteration cleanly."""
    arr = streaming_parse_json(whole('["abc", 42]'))
    it = iter(arr)
    s = next(it)
    next(it)                        # auto-drains s
    with pytest.raises(StopIteration):
        next(s)


def test_partial_child_iteration_then_parent_advance():
    arr = streaming_parse_json(whole('["abcdef", 1]'))
    it = iter(arr)
    s = next(it)
    next(iter(s))                   # consume only the first chunk of s
    n = next(it)                    # parent advances; auto-drain s
    assert n == 1
    assert s.value == 'abcdef'


# ─── errors ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("text,frag", [
    ("tru", "boolean"),
    ("nul", "null"),
    (r'"oops\x"', "escape"),
    ("[1, 2", "array"),
    ('{"a": 1', "object"),
    ("xyz", "starting json"),
    ('{1: 2}', "object key"),
    ('{"a"; 1}', "':'"),
    ('[1; 2]', "array"),
    ('"unterminated', "string"),
])
def test_error_messages(text, frag):
    with pytest.raises(ValueError, match=frag):
        result = streaming_parse_json(whole(text))
        if hasattr(result, 'value'):
            result.value


# ─── trash_skipper: leading-trash extraction ─────────────────────────────────


def test_trash_skipper_strips_prefix_and_suffix():
    text = "junk before {\"a\": 1} junk after"
    out = ''.join(trash_skipper(whole(text)))
    assert out == '{"a": 1}'


def test_trash_skipper_with_markdown_fence():
    text = "Sure! ```json\n{\n  \"key\": \"value\"\n}\n```\nLet me know!"
    parsed = streaming_parse_json(trash_skipper(whole(text)))
    assert parsed.value == {'key': 'value'}


def test_trash_skipper_typical_llm_response_chunked():
    """The user's example response, chunked at 20-char boundaries."""
    response = (
        "Certainly I can do that for you. Here is your json containing "
        "the items you mentioned:\n\n```json\n"
        '{"key1": "value1", "key2": "value2", "key25": "value25"}'
        "\n```\nlet me know if you need anything else!"
    )
    chunks = [response[i:i + 20] for i in range(0, len(response), 20)]
    parsed = streaming_parse_json(trash_skipper(chunks, look_for=(dict, list)))
    assert parsed.value == {'key1': 'value1', 'key2': 'value2', 'key25': 'value25'}


def test_trash_skipper_braces_inside_strings_not_counted():
    text = 'noise {"k": "v with } and { in it"} junk'
    out = ''.join(trash_skipper(whole(text)))
    assert out == '{"k": "v with } and { in it"}'


def test_trash_skipper_escaped_quote_inside_string():
    """A `\\"` inside a JSON string should not be treated as ending the string."""
    text = 'preamble {"k": "before \\" after"} tail'
    out = ''.join(trash_skipper(whole(text)))
    assert out == '{"k": "before \\" after"}'


@pytest.mark.parametrize("size", [1, 3, 7, 17])
def test_trash_skipper_chunk_size_invariance(size):
    text = "preamble " + '{"city": "Paris", "n": 42}' + " trailing"
    out = ''.join(trash_skipper(chunked(text, size)))
    assert out == '{"city": "Paris", "n": 42}'


def test_trash_skipper_no_match_returns_empty():
    out = list(trash_skipper(whole("just plain text"), look_for=(dict,)))
    assert out == []


def test_trash_skipper_accepts_single_type_arg():
    out = ''.join(trash_skipper(whole("noise [1] trail"), look_for=list))
    assert out == '[1]'


# ─── trash_skipper: per-type look_for ────────────────────────────────────────


def test_trash_skipper_list_only():
    text = 'before [1, 2, 3] {"not": "this"} after'
    out = ''.join(trash_skipper(whole(text), look_for=(list,)))
    assert out == '[1, 2, 3]'


def test_trash_skipper_string_only():
    text = 'noise "the string" {"x": 1} tail'
    out = ''.join(trash_skipper(whole(text), look_for=(str,)))
    assert out == '"the string"'


def test_trash_skipper_number_only():
    text = "answer is -1.5e2 grams"
    out = ''.join(trash_skipper(whole(text), look_for=(int, float)))
    assert out == '-1.5e2'


def test_trash_skipper_bool_only_validates_literal():
    """`t` in `truthfully` must NOT trigger a match — only the actual `true` literal does."""
    text = "truthfully no, the result is false yes"
    out = ''.join(trash_skipper(whole(text), look_for=(bool,)))
    assert out == 'false'


def test_trash_skipper_null_only_validates_literal():
    """`n` in `nine` must NOT trigger a match — only the actual `null` literal does."""
    text = "number nine, then null at the end"
    out = ''.join(trash_skipper(whole(text), look_for=(type(None),)))
    assert out == 'null'


def test_trash_skipper_literal_split_across_chunks():
    """Literal validation must pull more chunks if the lookahead spans a boundary."""
    out = ''.join(trash_skipper(iter(['hi tr', 'ue end']), look_for=(bool,)))
    assert out == 'true'


def test_trash_skipper_picks_first_of_allowed_types():
    text = "noise [1, 2] more {\"k\": 1} done"
    out = ''.join(trash_skipper(whole(text), look_for=(dict, list)))
    assert out == '[1, 2]'         # list comes first


def test_trash_skipper_ignores_disallowed_types():
    text = '"a string" then {"actual": "json"}'
    out = ''.join(trash_skipper(whole(text), look_for=(dict,)))
    assert out == '{"actual": "json"}'


# ─── trash_skipper round-trip ────────────────────────────────────────────────


def test_trash_skipper_pipes_into_streaming_parse_json():
    """End-to-end: trashy LLM response → trash_skipper → streaming_parse_json."""
    response = "Here you go:\n```json\n[1, 2, [3, {\"k\": \"v\"}]]\n```"
    chunks = chunked(response, 7)
    parsed = streaming_parse_json(trash_skipper(chunks, look_for=(dict, list)))
    assert parsed.value == [1, 2, [3, {'k': 'v'}]]
