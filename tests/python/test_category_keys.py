"""Categorical labels learned from slices equal today's whole-column labels (spec §6.3).

Today's training labels are ``[str(v) for v in column.to_numpy()]`` factorised
in sorted order (``interactions.py``), and explicit levels match against
``column.to_numpy().astype(str)``. Learning the same column from slices, in any
split, must give exactly those labels and codes.
"""

from __future__ import annotations

import datetime as dt
import decimal

import numpy as np
import polars as pl
import pytest
from rustystats._rustystats import factorize_strings_py
from rustystats.category_keys import (
    ENTRY_OVERHEAD_BYTES,
    CategoryKeyLearner,
    LearnedStateBudget,
    observed_category_labels,
    training_keys,
)
from rustystats.exceptions import (
    EncodingError,
    LearnedStateTooLargeError,
    RustyStatsError,
    UnseenLabelError,
    ValidationError,
)


def _nan_with_bits(bits: int) -> float:
    return float(np.array([bits], dtype=np.uint64).view(np.float64)[0])


def _london(*moments: dt.datetime | None) -> pl.Series:
    return pl.Series("x", moments, dtype=pl.Datetime("us")).dt.replace_time_zone("Europe/London")


# Nulls sit mid-column and at the end, so a split into threes has null-free slices.
COLUMNS = {
    "string_with_nulls": pl.Series("x", ["b", "a", "é", None, "None", "a\x00", "b", None]),
    "categorical": pl.Series("x", ["b", "a", "c", None, "a", "b"], dtype=pl.Categorical),
    "enum": pl.Series("x", ["b", "a", "b", None, "a"], dtype=pl.Enum(["c", "b", "a"])),
    "boolean_with_nulls": pl.Series("x", [True, False, True, None, False]),
    "boolean_without_nulls": pl.Series("x", [True, False, True, True]),
    "int64_with_nulls": pl.Series("x", [1, 2, 10, None, 2, -3, None]),
    "int64_without_nulls": pl.Series("x", [1, 2, 10, 2, -3]),
    "int64_above_2_53_with_nulls": pl.Series(
        "x", [2**53, 2**53 + 1, 3, None, 2**53 + 1, 2**62 + 1, 2**62]
    ),
    "int8_with_nulls": pl.Series("x", [1, -2, 3, None], dtype=pl.Int8),
    "uint64_with_nulls": pl.Series("x", [2**64 - 1, 3, 4, None], dtype=pl.UInt64),
    "float64_nan_and_signed_zeros": pl.Series(
        "x",
        [
            1.5,
            -0.0,
            0.0,
            None,
            float("nan"),
            _nan_with_bits(0xFFF8_0000_0000_0000),
            _nan_with_bits(0x7FF0_0000_0000_0001),
            1e16,
            -0.0,
        ],
    ),
    "float32_with_nulls": pl.Series("x", [1.1, 2.5, float("nan"), None], dtype=pl.Float32),
    "date_with_nulls": pl.Series("x", [dt.date(2020, 1, 1), dt.date(1999, 3, 4), None]),
    "datetime_with_tz": _london(
        dt.datetime(2020, 7, 1, 12), dt.datetime(2020, 1, 1), dt.datetime(2020, 7, 1, 12), None
    ),
    "datetime_ms_naive": pl.Series(
        "x", [dt.datetime(2020, 1, 1, 0, 0, 0, 1000), None], dtype=pl.Datetime("ms")
    ),
    "duration_with_nulls": pl.Series(
        "x", [dt.timedelta(seconds=3), dt.timedelta(days=-1), dt.timedelta(seconds=3), None]
    ),
    "time_with_nulls": pl.Series("x", [dt.time(1, 2, 3), dt.time(0, 0), None]),
    "decimal_with_nulls": pl.Series(
        "x",
        [decimal.Decimal("1.50"), decimal.Decimal("-2"), decimal.Decimal("1.5"), None],
        dtype=pl.Decimal(10, 2),
    ),
    "binary_with_nulls": pl.Series("x", [b"a", b"\x00b", b"a", None, b""]),
}


def _today(col: pl.Series) -> tuple[list[str], np.ndarray]:
    """Training's labels and codes: interactions.py's str() then sorted factorisation."""
    levels, codes = factorize_strings_py([str(v) for v in col.to_numpy()])
    return list(levels), np.asarray(codes, dtype=np.int32)


def _haute_observed(col: pl.Series) -> list[str]:
    """Haute's helper (``_rustystats.py::_observed_category_labels``)."""
    strings = np.asarray(col.to_numpy()).astype(str)
    present = ~col.is_null().to_numpy()
    return sorted(set(strings[present].tolist()))


def _split(col: pl.Series, size: int) -> list[pl.Series]:
    return [col.slice(start, size) for start in range(0, len(col), size)]


SPLITS = {
    "whole": lambda col: [col],
    "single_rows": lambda col: _split(col, 1),
    "threes": lambda col: _split(col, 3),
}


@pytest.mark.parametrize("split", SPLITS)
@pytest.mark.parametrize("case", COLUMNS)
def test_slices_reproduce_todays_labels_and_codes(case: str, split: str) -> None:
    col = COLUMNS[case]
    slices = SPLITS[split](col)
    levels, codes = _today(col)

    learner = CategoryKeyLearner("x")
    for part in slices:
        learner.observe(part)
    frozen = learner.finish()

    assert frozen.labels == levels
    encoded = np.concatenate([frozen.encode(part) for part in slices])
    assert encoded.dtype == np.int32
    np.testing.assert_array_equal(encoded, codes)
    # Encoding does not depend on how the column is sliced.
    np.testing.assert_array_equal(frozen.encode(col), codes)


@pytest.mark.parametrize("split", SPLITS)
@pytest.mark.parametrize("case", COLUMNS)
def test_slices_reproduce_explicit_level_labels(case: str, split: str) -> None:
    col = COLUMNS[case]
    slices = SPLITS[split](col)
    learner = CategoryKeyLearner("x")
    for part in slices:
        learner.observe(part)
    frozen = learner.finish()

    explicit = np.asarray(frozen.explicit_labels)[frozen.encode(col, explicit=True)]
    assert explicit.tolist() == np.asarray(col.to_numpy()).astype(str).tolist()
    assert frozen.observed_labels() == _haute_observed(col)
    assert observed_category_labels(slices) == _haute_observed(col)


def test_renderers_differ_for_binary() -> None:
    col = pl.Series("x", [b"a", None])
    learner = CategoryKeyLearner("x")
    learner.observe(col)
    frozen = learner.finish()
    assert frozen.labels == ["None", "b'a'"]
    assert frozen.observed_labels() == ["a"]


def test_binary_without_an_ascii_label_refuses_explicit_levels_only() -> None:
    col = pl.Series("bin", [b"\xff", b"a"])
    learner = CategoryKeyLearner("bin")
    learner.observe(col)
    frozen = learner.finish()
    assert frozen.labels == _today(col)[0]
    with pytest.raises(EncodingError, match="'bin'"):
        frozen.observed_labels()


def test_keys_keep_every_distinction_training_makes() -> None:
    floats = COLUMNS["float64_nan_and_signed_zeros"]
    # 1.5, -0.0, 0.0, null, one NaN for three payloads, 1e16.
    assert training_keys(floats).n_unique() == 6
    assert training_keys(floats).null_count() == 1

    big = COLUMNS["int64_above_2_53_with_nulls"]
    learner = CategoryKeyLearner("x")
    learner.observe(big)
    frozen = learner.finish()
    # 2**53 and 2**53 + 1 are distinct keys that both render as float64 2**53.
    assert frozen.n_keys == 6
    assert len(frozen.labels) == 4
    codes = frozen.encode(big)
    assert codes[0] == codes[1]


def test_many_keys_and_a_long_key_render_in_chunks() -> None:
    ints = pl.Series("x", [*range(700), None])
    learner = CategoryKeyLearner("x")
    for part in _split(ints, 97):
        learner.observe(part)
    frozen = learner.finish()
    levels, codes = _today(ints)
    assert frozen.labels == levels
    np.testing.assert_array_equal(frozen.encode(ints), codes)

    strings = pl.Series("x", ["z" * 70_000, "a", None, "b"])
    learner = CategoryKeyLearner("x")
    learner.observe(strings)
    assert learner.finish().labels == _today(strings)[0]


def test_all_null_column() -> None:
    col = pl.Series("x", [None, None], dtype=pl.Int64)
    learner = CategoryKeyLearner("x")
    for part in _split(col, 1):
        learner.observe(part)
    frozen = learner.finish()
    levels, codes = _today(col)
    assert frozen.labels == levels == ["nan"]
    np.testing.assert_array_equal(frozen.encode(col), codes)
    assert frozen.observed_labels() == _haute_observed(col) == []


@pytest.mark.parametrize(
    "col",
    [
        pl.Series("x", [[1], [2]]),
        pl.Series("x", [[1, 2]], dtype=pl.Array(pl.Int64, 2)),
        pl.Series("x", [{"a": 1}]),
        pl.Series("x", [object()], dtype=pl.Object),
        pl.Series("x", [None, None]),
        pl.Series("x", [1], dtype=pl.Int128),
    ],
    ids=["list", "array", "struct", "object", "null", "int128"],
)
def test_nested_and_unconvertible_dtypes_are_refused(col: pl.Series) -> None:
    with pytest.raises(ValidationError, match="'x'"):
        training_keys(col)
    with pytest.raises(ValidationError, match="'x'"):
        CategoryKeyLearner("x").observe(col)


def test_unseen_key_raises() -> None:
    learner = CategoryKeyLearner("region")
    learner.observe(pl.Series("region", ["a", "b"]))
    frozen = learner.finish()
    with pytest.raises(UnseenLabelError, match=r"'region'.*'c'"):
        frozen.encode(pl.Series("region", ["a", "c"]))
    # A null the learner never saw is unseen too.
    with pytest.raises(UnseenLabelError, match="'None'"):
        frozen.encode(pl.Series("region", ["a", None]))
    assert issubclass(UnseenLabelError, RustyStatsError)


def test_unseen_float_signed_zero_raises() -> None:
    learner = CategoryKeyLearner("x")
    learner.observe(pl.Series("x", [0.0, float("nan")]))
    frozen = learner.finish()
    # Any NaN payload is the learned NaN; -0.0 is not 0.0.
    assert frozen.encode(pl.Series("x", [_nan_with_bits(0xFFF8_0000_0000_0000)])).tolist() == [1]
    with pytest.raises(UnseenLabelError, match=r"'-0\.0'"):
        frozen.encode(pl.Series("x", [-0.0]))


def test_slice_with_another_dtype_is_refused() -> None:
    learner = CategoryKeyLearner("x")
    learner.observe(pl.Series("x", [1, 2]))
    with pytest.raises(ValidationError, match="changed dtype"):
        learner.observe(pl.Series("x", [1.0]))
    frozen = learner.finish()
    with pytest.raises(ValidationError, match="changed dtype"):
        frozen.encode(pl.Series("x", ["1"]))
    with pytest.raises(ValidationError, match="frozen"):
        learner.observe(pl.Series("x", [3]))


def test_budget_refuses_a_slice_before_inserting_it() -> None:
    entry = 2 + ENTRY_OVERHEAD_BYTES  # a two-byte string key
    budget = LearnedStateBudget(3 * entry)
    learner = CategoryKeyLearner("x", budget=budget)

    learner.observe(pl.Series("x", ["aa", "bb", "aa"]))
    assert (learner.n_keys, learner.nbytes, budget.used) == (2, 2 * entry, 2 * entry)
    # Keys already in the table cost nothing.
    learner.observe(pl.Series("x", ["bb", "aa"]))
    assert budget.used == 2 * entry

    with pytest.raises(LearnedStateTooLargeError, match=r"'x' \(4 distinct values\)"):
        learner.observe(pl.Series("x", ["cc", "dd"]))
    assert (learner.n_keys, learner.nbytes, budget.used) == (2, 2 * entry, 2 * entry)

    # A null is its own key and costs one entry.
    learner.observe(pl.Series("x", [None, "aa"], dtype=pl.String))
    assert (learner.n_keys, learner.has_null, budget.used) == (
        3,
        True,
        2 * entry + ENTRY_OVERHEAD_BYTES,
    )


def test_budget_is_shared_across_columns() -> None:
    budget = LearnedStateBudget(2 * (8 + ENTRY_OVERHEAD_BYTES))
    first = CategoryKeyLearner("a", budget=budget)
    first.observe(pl.Series("a", [1, 2]))
    second = CategoryKeyLearner("b", budget=budget)
    with pytest.raises(LearnedStateTooLargeError, match="'b'"):
        second.observe(pl.Series("b", [1]))
    assert second.n_keys == 0


def test_label_reservation_refuses_before_forming_labels() -> None:
    col = pl.Series("x", ["aa", "bb"])
    table_bytes = 2 * (2 + ENTRY_OVERHEAD_BYTES)
    budget = LearnedStateBudget(table_bytes + 100)
    learner = CategoryKeyLearner("x", budget=budget)
    learner.observe(col)
    with pytest.raises(LearnedStateTooLargeError, match="'x'"):
        learner.finish()
    assert budget.used == table_bytes

    roomy = LearnedStateBudget(1 << 20)
    learner = CategoryKeyLearner("x", budget=roomy)
    learner.observe(col)
    assert learner.finish().labels == ["aa", "bb"]
    assert roomy.used > table_bytes  # the labels stay charged


def test_unbounded_budget_and_errors() -> None:
    with pytest.raises(ValidationError, match="non-negative"):
        LearnedStateBudget(-1)
    with pytest.raises(ValidationError, match="No slices"):
        CategoryKeyLearner("x").finish()
    assert observed_category_labels([]) == []
    assert observed_category_labels(pl.Series("x", [3, None, 1])) == ["1.0", "3.0"]
