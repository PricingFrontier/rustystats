"""Source fingerprints hash canonical, tagged key bytes (``specs/out-of-core-fit.md`` §9).

``fingerprint_update`` feeds every value to the 128-bit XXH3 hash itself: a kind byte, a
validity byte, then the value's 64-bit pattern or its length-prefixed bytes. It never
goes through Polars' 64-bit hashes, so two keys that collide there stay distinct, and the
hash is streamed, so the digest does not depend on how a column is cut into slices.
"""

from __future__ import annotations

import datetime
import os
import subprocess
import sys
from decimal import Decimal

import polars as pl
import pytest
from rustystats._rustystats import ColumnFingerprint
from rustystats.out_of_core import FINGERPRINT_ROWS, fingerprint_update


def _digest(*slices: pl.Series) -> tuple[int, int]:
    fingerprint = ColumnFingerprint()
    for piece in slices:
        fingerprint_update(fingerprint, piece)
    return fingerprint.digest(), fingerprint.rows


def _columns() -> dict[str, pl.Series]:
    """One column of every dtype class a model input can have, each with a null."""
    labels = ["a", "", None, "ab", "é€😀", "a", "x" * 9000, None, "bc", "c"]
    n = len(labels)
    columns = {
        "float": pl.Series([1.5, -0.0, None, 0.0, float("nan"), 2.0, 1e300, -3.0, 0.5, 1.0]),
        "float32": pl.Series(
            [1.5, None, 0.0, -2.5, 3.0, 1.0, 2.0, 4.0, 5.0, 6.0], dtype=pl.Float32
        ),
        "int": pl.Series([2**53 + 1, 1, None, -5, 0, 7, 2**62, -(2**62), 3, 4], dtype=pl.Int64),
        "int8": pl.Series([1, -1, None, 0, 5, 6, 7, 8, 9, 10], dtype=pl.Int8),
        "uint": pl.Series([2**64 - 1, 0, None, 1, 2, 3, 4, 5, 6, 7], dtype=pl.UInt64),
        "bool": pl.Series([True, False, None, True, True, False, False, True, None, True]),
        "date": pl.Series([datetime.date(2020, 1, i + 1) if i != 2 else None for i in range(n)]),
        "datetime": pl.Series(
            [datetime.datetime(2020, 1, 1, i) if i != 3 else None for i in range(n)]
        ),
        "duration": pl.Series(
            [datetime.timedelta(seconds=i) if i != 4 else None for i in range(n)]
        ),
        "time": pl.Series([datetime.time(i, 0) if i != 5 else None for i in range(n)]),
        "string": pl.Series(labels, dtype=pl.String),
        "categorical": pl.Series(labels, dtype=pl.Categorical),
        "enum": pl.Series(
            labels, dtype=pl.Enum(sorted({label for label in labels if label is not None}))
        ),
        "binary": pl.Series([b"\x00\xff", b"", None, b"ab", b"\x00", b"a", b"b", None, b"c", b"d"]),
        "decimal": pl.Series(
            [Decimal("1.10"), None, Decimal("-2.50"), Decimal("0.00")] * 2 + [None, Decimal("3")],
            dtype=pl.Decimal(10, 2),
        ),
    }
    columns["int128"] = pl.Series([2**100, None, -(2**90), 0, 1, 2, 3, 4, 5, 6], dtype=pl.Int128)
    return columns


COLUMNS = _columns()


@pytest.mark.parametrize("name", sorted(COLUMNS))
def test_the_digest_does_not_depend_on_how_the_column_is_sliced(name: str) -> None:
    column = COLUMNS[name]
    whole = _digest(column)
    assert whole[1] == column.len()
    for size in (1, 2, 3, column.len() - 1):
        pieces = [column.slice(start, size) for start in range(0, column.len(), size)]
        assert _digest(*pieces) == whole, f"slices of {size}"
    # Empty slices change nothing.
    assert _digest(column.slice(0, 0), column, column.slice(0, 0)) == whole


@pytest.mark.parametrize("name", sorted(COLUMNS))
def test_each_value_change_and_each_null_is_detected(name: str) -> None:
    column = COLUMNS[name]
    digest = _digest(column)[0]
    first_valid = next(i for i, valid in enumerate(column.is_not_null()) if valid)
    # Built anew: `scatter` would change the shared column in place.
    nulled = pl.concat(
        [
            column.head(first_valid),
            pl.Series(column.name, [None], dtype=column.dtype),
            column.slice(first_valid + 1),
        ]
    )
    assert nulled.len() == column.len()
    assert _digest(nulled)[0] != digest, "a value replaced by null"
    assert _digest(column.reverse())[0] != digest, "the row order"
    assert _digest(column.head(column.len() - 1))[0] != digest, "the row count"


def test_polars_hashes_are_not_used(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keys whose Polars hashes collide (simulated: every hash is 0) stay distinct."""
    calls = []

    def colliding_hash(self: pl.Series, *args: object, **kwargs: object) -> pl.Series:
        calls.append(self.name)
        return pl.Series(self.name, [0] * self.len(), dtype=pl.UInt64)

    monkeypatch.setattr(pl.Series, "hash", colliding_hash)
    for dtype in (pl.String, pl.Categorical):
        left = pl.Series(["brand01", "brand02", None], dtype=dtype)
        right = pl.Series(["brand02", "brand01", None], dtype=dtype)
        assert _digest(left)[0] != _digest(right)[0]
    assert _digest(pl.Series([b"\x01"]))[0] != _digest(pl.Series([b"\x02"]))[0], "binary keys"
    assert calls == []


def test_keys_are_length_delimited_and_tagged() -> None:
    def digest(values: list, dtype: pl.DataType) -> int:
        return _digest(pl.Series(values, dtype=dtype))[0]

    # Length prefixes: the same bytes split differently are different keys.
    assert digest(["ab", "c"], pl.String) != digest(["a", "bc"], pl.String)
    assert digest(["abc", ""], pl.String) != digest(["", "abc"], pl.String)
    # A null is neither the empty string nor the text "null"; nor 0, nor NaN.
    assert digest([None], pl.String) != digest([""], pl.String)
    assert digest([None], pl.String) != digest(["null"], pl.String)
    assert digest([None], pl.Int64) != digest([0], pl.Int64)
    assert digest([None], pl.Float64) != digest([float("nan")], pl.Float64)
    assert digest([0.0], pl.Float64) != digest([-0.0], pl.Float64)
    # Kinds: the value 1 as an integer, a boolean and a date (one day), and the float
    # whose bit pattern is 1, all differ.
    one = {
        digest([1], pl.Int64),
        digest([True], pl.Boolean),
        digest([datetime.date(1970, 1, 2)], pl.Date),
        digest([5e-324], pl.Float64),
    }
    assert len(one) == 4
    # A label and its own bytes as a binary value differ by kind.
    assert digest(["ab"], pl.String) != digest([b"ab"], pl.Binary)
    # A String and a Categorical with the same labels are the same keys.
    assert digest(["a", None, "b"], pl.String) == digest(["a", None, "b"], pl.Categorical)


def test_the_digest_is_the_same_in_another_process() -> None:
    script = (
        "import sys; sys.path.insert(0, 'tests/python');"
        "from test_fingerprint_keys import COLUMNS, _digest;"
        "print(sorted((name, _digest(column)) for name, column in COLUMNS.items()))"
    )
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True, cwd=root
    )
    assert completed.stdout.strip() == str(
        sorted((name, _digest(column)) for name, column in COLUMNS.items())
    )


def test_slices_longer_than_one_conversion_chunk() -> None:
    """A call converts FINGERPRINT_ROWS rows at a time; the digest does not show it."""
    n = 2 * FINGERPRINT_ROWS + 123
    labels = pl.Series([None if i % 97 == 0 else f"k{i % 1000}" for i in range(n)])
    numbers = pl.Series([None if i % 89 == 0 else i * 0.5 for i in range(n)])
    for column in (labels, numbers, labels.cast(pl.Categorical), labels.cast(pl.Binary)):
        whole = _digest(column)
        for size in (1000, FINGERPRINT_ROWS - 1, FINGERPRINT_ROWS, FINGERPRINT_ROWS + 1):
            pieces = [column.slice(start, size) for start in range(0, n, size)]
            assert _digest(*pieces) == whole, (column.dtype, size)


def _peak_rss_after(call) -> int:
    """Bytes the peak RSS rose by during ``call`` (Linux: VmHWM reset by clear_refs)."""

    def status(field: str) -> int:
        with open("/proc/self/status", encoding="ascii") as handle:
            for line in handle:
                if line.startswith(field):
                    return int(line.split()[1]) * 1024
        raise AssertionError(field)

    with open("/proc/self/clear_refs", "w", encoding="ascii") as handle:
        handle.write("5")
    base = status("VmRSS:")
    call()
    return status("VmHWM:") - base


@pytest.mark.skipif(
    not os.access("/proc/self/clear_refs", os.W_OK), reason="needs Linux peak-RSS reset"
)
@pytest.mark.parametrize("dtype", [pl.Binary, pl.String, pl.Categorical])
def test_a_large_value_costs_one_copy_of_its_bytes(dtype) -> None:
    """A 40 MiB value is hashed from its bytes: no hex text, no joined string."""
    size = 40 * 1024 * 1024
    value = b"\xff" * size if dtype == pl.Binary else "x" * size
    column = pl.Series(
        [value, None, value[:10]], dtype=pl.Binary if dtype == pl.Binary else pl.String
    )
    column = column.cast(dtype)
    del value
    rise = _peak_rss_after(lambda: fingerprint_update(ColumnFingerprint(), column))
    assert rise <= 1.25 * size, f"{rise / size:.2f} × the value"
