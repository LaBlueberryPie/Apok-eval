"""apok_eval.jsonstrict: one strict JSON parser over bytes, and its deterministic writer.

Every refusal asserts the precise problem string, because callers put it into their own refusals
as the named cause.
"""

from __future__ import annotations

import json
import math
import sys

import numpy as np
import pytest

from apok_eval.canonical import canonical_dumps
from apok_eval.jsonstrict import MAX_DEPTH, StrictJsonError, dumps, loads


def nested(depth: int) -> bytes:
    return b"[" * depth + b"]" * depth


def nested_list(depth: int) -> list:
    value: list = []
    for _ in range(depth - 1):
        value = [value]
    return value


def problem(data: bytes, **kwargs) -> str:
    with pytest.raises(StrictJsonError) as info:
        loads(data, **kwargs)
    return str(info.value)


# --- what is accepted ------------------------------------------------------------------------------


VALID = [
    b'{"a": [1, 2.5, -0.0, true, false, null, "\xc3\xa9"], "b": {"c": {}}, "d": []}',
    b'  [0.1, 1e-300, 5e-324, 1.7976931348623157e308, -2, 1E5, 3.0]\n',
    b'"\\ud83d\\ude00 a valid pair is one character"',
    b'["\\\\ud800 is an escaped backslash, not a surrogate", "\\u00e9", "\\u0000"]',
    b'{"n": 1' + b"0" * 400 + b', "m": -12}',
    b"17",
    b"null",
]


@pytest.mark.parametrize("data", VALID)
def test_a_strict_document_reads_exactly_as_python_json_reads_it(data):
    value = loads(data)
    assert value == json.loads(data)
    # ...with the same types: 1 stays int, 1.0 float, and long integers stay exact.
    assert canonical_dumps(value) == canonical_dumps(json.loads(data))


def test_a_valid_surrogate_pair_is_one_character_and_long_integers_stay_exact():
    assert loads(b'"\\ud83d\\ude00"') == "\U0001f600"
    assert loads(b'{"n": 1' + b"0" * 400 + b"}") == {"n": 10 ** 400}
    assert loads(bytearray(b"[1]")) == [1]


def test_the_error_is_a_value_error():
    assert issubclass(StrictJsonError, ValueError)


# --- what is refused, and the problem named -----------------------------------------------------------


REFUSED = [
    (b"\xff", "not UTF-8 (invalid start byte at byte 0)"),
    (b'{"a": "\xc3"}', "not UTF-8 (invalid continuation byte at byte 7)"),
    (b'["\xed\xa0\x80"]', "not UTF-8"),  # a surrogate encoded in UTF-8 is not UTF-8
    (b"\xef\xbb\xbf{}", "starts with a byte-order mark (U+FEFF), which JSON text must not carry"),
    (b"{", "invalid JSON: Expecting property name enclosed in double quotes at line 1 column 2"),
    (b"[1,]", "invalid JSON: Illegal trailing comma before end of array at line 1 column 3"
     if sys.version_info >= (3, 13) else "invalid JSON: Expecting value at line 1 column 4"),
    (b"", "invalid JSON: Expecting value at line 1 column 1"),
    (b"{'a': 1}", "invalid JSON"),
    (b'{"a": 1, "a": 2}', "duplicate key 'a'"),
    (b'{"a": {"b": 1, "b": 1}}', "duplicate key 'b'"),  # nested, and with an equal value
    (b'[{"x": [{"k": 1, "k": 2}]}]', "duplicate key 'k'"),
    (b"NaN", "non-standard constant NaN (NaN, Infinity and -Infinity are not JSON)"),
    (b'{"x": [Infinity]}', "non-standard constant Infinity"),
    (b"[-Infinity]", "non-standard constant -Infinity"),
    (b"1e400", "number 1e400 overflows a double"),
    (b"[-1e999]", "number -1e999 overflows a double"),
    (b"[1" + b"0" * 400 + b".5]", "number 10000000000000000000... (403 characters) overflows a double"),
    (b'{"n": ' + b"1" * 5000 + b"}", "integer 11111111111111111111... (5000 characters) has 5000 digits, more "
                                     f"than int() converts ({sys.get_int_max_str_digits()})"),
    (b"-" + b"9" * 4400, "has 4400 digits, more than int() converts"),
    (b'["\\ud800"]', "lone surrogate '\\ud800' in the string at $[0]: not Unicode text"),
    (b'{"a": {"\\uDFFF": 1}}', "lone surrogate '\\udfff' in a key of the object at $.a"),
    (b'{"a": [1, {"k": "ok"}, "x\\udc00"]}', "lone surrogate '\\udc00' in the string at $.a[2]"),
    (b'["\\ud83d\\ude00", "\\ude00\\ud83d"]', "lone surrogate '\\ude00' in the string at $[1]"),  # reversed pair
    (b'["\\uDBFF"]', "lone surrogate '\\udbff'"),
    (b'{"\\ud800": 1, "\\ud800": 2}', "duplicate key '\\ud800'"),  # the message itself stays encodable
]


@pytest.mark.parametrize("data, expected", REFUSED)
def test_a_document_that_is_not_strict_json_is_refused_naming_the_problem(data, expected):
    message = problem(data)
    assert expected in message
    message.encode("utf-8")  # callers print it as JSON


def test_the_first_lone_surrogate_in_document_order_is_named():
    assert "in a key of the object at $" in problem(b'{"\\udc01": "\\ud801"}')
    assert "'\\ud801' in the string at $.a" in problem(b'{"a": "\\ud801", "\\udc01": 1}')
    assert "'\\ud801' in the string at $[0]" in problem(b'["\\ud801", "\\udc02"]')


@pytest.mark.parametrize("data, where", [
    # Dotted, "a.b" -> c and a -> b -> c (which holds the integer 1) would both be $.a.b.c.
    (b'{"a.b": {"c": "\\ud800"}, "a": {"b": {"c": 1}}}', '$["a.b"].c'),
    (b'{"x[0]": "\\ud800"}', '$["x[0]"]'),
    (b'{"0": "\\ud800"}', '$["0"]'),  # a member, not the array index $[0]
    (b'{"": ["\\ud800"]}', '$[""][0]'),
    (b'{"a\\"\\n": "\\ud800"}', '$["a\\"\\n"]'),  # quote and newline escaped as JSON escapes them
    (b'{"gr\xc3\xb6\xc3\x9fe": "\\ud800"}', '$["gr\u00f6\u00dfe"]'),  # plain means ASCII; the rest is raw
    (b'{"_a1": {"B_2": "\\ud800"}}', "$._a1.B_2"),
])
def test_the_path_to_a_lone_surrogate_names_one_place(data, where):
    assert problem(data) == f"lone surrogate '\\ud800' in the string at {where}: not Unicode text (it cannot be " \
                            "encoded as UTF-8)"


# --- nesting --------------------------------------------------------------------------------------


def test_nesting_is_bounded_at_max_depth_levels():
    assert MAX_DEPTH == 256
    assert loads(nested(MAX_DEPTH)) == nested_list(MAX_DEPTH)
    assert loads(b'{"a":' + nested(MAX_DEPTH - 1) + b"}")  # an object is a level too
    assert problem(nested(MAX_DEPTH + 1)) == "arrays and objects nested deeper than 256 levels"
    assert problem(b'{"a":' + nested(MAX_DEPTH) + b"}") == "arrays and objects nested deeper than 256 levels"
    assert problem(b'[1, {"a": [2, ' + nested(MAX_DEPTH - 2) + b"]}]") == \
        "arrays and objects nested deeper than 256 levels"  # the deepest branch counts, not the first


def test_whatever_parses_can_be_re_serialised():
    deepest = loads(nested(MAX_DEPTH))
    canonical_dumps(deepest)
    assert loads(dumps(deepest)) == deepest


@pytest.mark.parametrize("data", [nested(100_000), b'{"a":' * 100_000])
def test_nesting_beyond_the_parser_is_refused_not_a_recursion_error(data):
    assert problem(data) == "arrays and objects nested deeper than the parser can recurse (the limit is 256 levels)"


def test_the_depth_limit_is_declared_by_the_caller():
    assert loads(nested(3), max_depth=3) == [[[]]]
    assert loads(b"[1]", max_depth=1) == [1] and loads(b"1", max_depth=1) == 1
    assert problem(nested(4), max_depth=3) == "arrays and objects nested deeper than 3 levels"
    assert problem(b'{"a": {}}', max_depth=1) == "arrays and objects nested deeper than 1 levels"


# --- non-standard constants -------------------------------------------------------------------------


def test_constants_are_collected_in_document_order_when_asked():
    value, tokens = loads(b'[NaN, {"a": Infinity}, -Infinity, NaN, 1.5]', constants="collect")
    assert math.isnan(value[0]) and value[1] == {"a": math.inf} and value[2] == -math.inf and value[4] == 1.5
    assert tokens == ("NaN", "Infinity", "-Infinity", "NaN")
    assert loads(b'{"a": 1}', constants="collect") == ({"a": 1}, ())


def test_collecting_constants_relaxes_nothing_else():
    assert problem(b"[1e400]", constants="collect") == "number 1e400 overflows a double"
    assert problem(b'{"a": NaN, "b": 1, "b": 2}', constants="collect") == "duplicate key 'b'"
    assert "lone surrogate" in problem(b'[NaN, "\\ud800"]', constants="collect")


def test_constants_are_rejected_by_default():
    assert problem(b"[NaN]") == problem(b"[NaN]", constants="reject")


# --- arguments ------------------------------------------------------------------------------------


@pytest.mark.parametrize("data", ['{"a": 1}', None, memoryview(b"{}")])
def test_only_bytes_are_parsed(data):
    with pytest.raises(TypeError, match="parsed from bytes"):
        loads(data)


@pytest.mark.parametrize("kwargs", [{"constants": "accept"}, {"constants": None}, {"max_depth": 0},
                                    {"max_depth": -1}, {"max_depth": True}, {"max_depth": 2.0}])
def test_a_bad_argument_is_a_programming_error_not_a_document_problem(kwargs):
    with pytest.raises(ValueError) as info:
        loads(b"[]", **kwargs)
    assert not isinstance(info.value, StrictJsonError)


# --- the writer ---------------------------------------------------------------------------------------


def test_the_writer_sorts_keys_indents_by_four_writes_raw_utf8_and_ends_with_a_newline():
    value = {"b": [1, "é"], "a": {}, "B": 1.5, "c": []}
    assert dumps(value) == ('{\n    "B": 1.5,\n    "a": {},\n    "b": [\n        1,\n        "é"\n    ],\n'
                            '    "c": []\n}\n').encode("utf-8")
    assert loads(dumps(value)) == value


@pytest.mark.parametrize("value, expected", [
    ({"a": (1, 2)}, "not plain finite JSON: TypeError: tuple is not canonical JSON (at .a)"),
    ({1: "x"}, "not plain finite JSON: TypeError: non-string key 1"),
    ({"a": math.nan}, "not plain finite JSON: ValueError: non-finite float at .a"),
    ([math.inf], "not plain finite JSON: ValueError"),
    ({"a": np.float32(1)}, "not plain finite JSON: TypeError: float32 is not canonical JSON"),
    ({"a": "\ud800"}, "not plain finite JSON: UnicodeEncodeError"),
    ({"\udfff": 1}, "not plain finite JSON: UnicodeEncodeError"),
    ([10 ** 5000], "not plain finite JSON: ValueError: Exceeds the limit"),
    (nested_list(MAX_DEPTH + 1), "would not read back as strict JSON: arrays and objects nested deeper than 256 "
                                 "levels"),
    (nested_list(100_000), "not plain finite JSON: RecursionError"),
])
def test_the_writer_refuses_what_the_reader_would_refuse(value, expected):
    with pytest.raises(StrictJsonError) as info:
        dumps(value)
    assert expected in str(info.value)


def test_the_writer_accepts_the_deepest_document_the_reader_accepts():
    assert dumps(nested_list(MAX_DEPTH)).count(b"[") == MAX_DEPTH
