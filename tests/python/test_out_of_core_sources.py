"""Sources, slicing and fingerprints for out-of-core fits (``specs/out-of-core-fit.md`` §4, §9)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from rustystats._rustystats import ColumnFingerprint
from rustystats.exceptions import BatchSourceChangedError, ValidationError
from rustystats.out_of_core import (
    BatchSource,
    OutOfCoreOptions,
    fingerprint_update,
    slice_rows,
)


def _frame(n: int = 50) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "x": np.arange(n, dtype=float),
            "big": pl.Series([2**53 + i for i in range(n)], dtype=pl.Int64),
            "s": [("a" * (i % 5)) if i % 7 else None for i in range(n)],
        }
    )


def test_source_projects_and_closes_its_iterable():
    closed = []

    def open_(columns):
        def generate():
            try:
                for start in range(0, 50, 20):
                    yield _frame().slice(start, 20)
            finally:
                closed.append(True)

        return generate()

    source = BatchSource(open_, _frame().schema)
    iterator = source.batches(["x"])
    first = next(iterator)
    assert first.columns == ["x"]
    iterator.close()
    assert closed == [True]


class _OwningReader:
    """A reader whose ``__iter__`` returns a generator: closing the generator does not
    release the reader, only the reader's own ``close()`` does."""

    def __init__(self, fail_on_iter: bool = False) -> None:
        self.closed = 0
        self.fail_on_iter = fail_on_iter

    def __iter__(self):
        if self.fail_on_iter:
            raise RuntimeError("cannot start reading")
        return (_frame().slice(start, 20) for start in range(0, 50, 20))

    def close(self) -> None:
        self.closed += 1


@pytest.mark.parametrize("how", ["closed early", "consumed", "error in a batch"])
def test_source_closes_an_owning_reader_and_its_iterator(how):
    reader = _OwningReader()
    schema = dict(_frame().schema)
    if how == "error in a batch":
        schema["x"] = pl.Int64  # the first batch's dtype is refused
    source = BatchSource(lambda columns: reader, schema)
    iterator = source.batches(["x"])
    if how == "closed early":
        next(iterator)
        iterator.close()
    elif how == "consumed":
        assert sum(batch.height for batch in iterator) == 50
    else:
        with pytest.raises(BatchSourceChangedError):
            next(iterator)
    assert reader.closed == 1


def test_source_closes_a_reader_whose_iteration_fails_to_start():
    reader = _OwningReader(fail_on_iter=True)
    source = BatchSource(lambda columns: reader, _frame().schema)
    with pytest.raises(RuntimeError, match="cannot start reading"):
        next(source.batches(["x"]))
    assert reader.closed == 1


def test_a_failing_close_does_not_hide_the_error_on_its_way_out():
    class Stop(Exception):
        pass

    class Reader(_OwningReader):
        def close(self) -> None:
            super().close()
            raise OSError("close failed")

    reader = Reader()
    iterator = BatchSource(lambda columns: reader, _frame().schema).batches(["x"])
    next(iterator)
    with pytest.raises(Stop):
        iterator.throw(Stop())
    assert reader.closed == 1
    # Consumed to its end, the close error is the caller's to see.
    with pytest.raises(OSError, match="close failed"):
        list(BatchSource(lambda columns: Reader(), _frame().schema).batches(["x"]))


def test_source_refuses_a_dtype_that_differs_from_its_schema():
    schema = dict(_frame().schema)
    schema["x"] = pl.Int64
    source = BatchSource(lambda columns: [_frame()], schema)
    with pytest.raises(BatchSourceChangedError, match="source changed"):
        list(source.batches(["x"]))


def test_source_refuses_unknown_columns():
    source = BatchSource(lambda columns: [_frame()], _frame().schema)
    with pytest.raises(ValidationError, match="no column"):
        list(source.batches(["nope"]))


def test_slices_are_bounded_by_rows_and_by_actual_bytes():
    frame = pl.DataFrame({"s": ["x" * 16_384] + ["y"] * 99})
    slices = list(slice_rows(frame, max_rows=1_000, max_bytes=1_024))
    # The oversized first row forms its own slice; the rest fit together.
    assert [len(s) for s in slices] == [1, 99]
    assert sum(len(s) for s in slice_rows(frame, max_rows=7)) == 100
    assert max(len(s) for s in slice_rows(frame, max_rows=7)) == 7


def _reference_slices(sizes: list[int], max_rows: int, max_bytes: int) -> list[int]:
    """§4.3's rule, row by row: a slice ends before the row that would take it past
    max_bytes or max_rows; a single larger row is a slice of its own."""
    lengths, start = [], 0
    while start < len(sizes):
        stop, total = start, 0
        while stop < len(sizes) and stop - start < max_rows:
            if stop > start and total + sizes[stop] > max_bytes:
                break
            total += sizes[stop]
            stop += 1
        lengths.append(stop - start)
        start = stop
    return lengths


@pytest.mark.parametrize("seed", range(6))
def test_slices_follow_the_row_rule_while_sizing_a_bounded_window(seed, monkeypatch):
    """Row sizes are computed for at most max_rows rows at a time (§4.3), and slices
    still follow the row-by-row rule across the windows' edges."""
    from rustystats import out_of_core

    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 400))
    lengths = rng.integers(0, 60, n)
    lengths[rng.random(n) < 0.03] = 700  # rows larger than a slice
    frame = pl.DataFrame({"s": ["x" * int(k) for k in lengths], "v": np.arange(n)})
    max_rows = int(rng.integers(1, 40))
    max_bytes = int(rng.integers(30, 300))
    sized: list[int] = []
    sizer = out_of_core._row_payload

    def spy(batch, start, length, fixed_bytes=8):
        sizes, widest = sizer(batch, start, length, fixed_bytes)
        sized.append(len(sizes))
        return sizes, widest

    monkeypatch.setattr(out_of_core, "_row_payload", spy)
    slices = list(slice_rows(frame, max_rows=max_rows, max_bytes=max_bytes))
    expected = _reference_slices([int(k) + 8 for k in lengths], max_rows, max_bytes)
    assert [len(piece) for piece in slices] == expected
    assert pl.concat(slices).equals(frame)
    assert max(sized) <= max_rows
    assert sum(sized) == n, "every row is sized once"


def test_an_oversized_row_is_admitted_before_it_is_sliced_out():
    frame = pl.DataFrame({"x": [1.0, 2.0, 3.0], "b": [b"a", b"\xff" * 5_000, b"c"]})
    admitted: list[tuple[str, int]] = []

    def admit(column, row_bytes):
        admitted.append((column, row_bytes))
        raise MemoryError("refused")

    pieces = slice_rows(frame, max_rows=10, max_bytes=100, admit=admit)
    assert len(next(pieces)) == 1  # the first row; the second is larger than a slice
    with pytest.raises(MemoryError):
        next(pieces)
    assert admitted == [("b", 5_008)]


def test_slices_can_count_only_variable_width_payload():
    frame = pl.DataFrame({"s": ["ab"] * 10, "x": np.arange(10.0)})
    assert [len(s) for s in slice_rows(frame, max_rows=100, max_bytes=5, fixed_bytes=0)] == [2] * 5
    assert [len(s) for s in slice_rows(frame, max_rows=100, max_bytes=20)] == [2] * 5


@pytest.mark.parametrize("column", ["x", "big", "s"])
def test_fingerprint_does_not_depend_on_batching(column):
    frame = _frame()
    whole = ColumnFingerprint()
    fingerprint_update(whole, frame[column])
    parts = ColumnFingerprint()
    for piece in slice_rows(frame, max_rows=3):
        fingerprint_update(parts, piece[column])
    assert whole.digest() == parts.digest()
    assert whole.rows == parts.rows == frame.height


def test_fingerprint_detects_changed_values_order_and_large_integers():
    def digest(series: pl.Series) -> int:
        fingerprint = ColumnFingerprint()
        fingerprint_update(fingerprint, series)
        return fingerprint.digest()

    big = _frame()["big"]
    assert digest(big) != digest(big.scatter([3], [2**53 + 4]))  # differs only above 2**53
    s = _frame()["s"]
    assert digest(s) != digest(s.reverse())
    assert digest(s) != digest(s.fill_null("z"))


def test_options_refuse_non_positive_limits():
    with pytest.raises(ValidationError, match="max_bytes"):
        OutOfCoreOptions(max_bytes=0)
    with pytest.raises(ValidationError, match="encode_rows"):
        OutOfCoreOptions(encode_rows=-1)
