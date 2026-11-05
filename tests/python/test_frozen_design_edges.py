"""Frozen-design edge cases: refusals, unusual term shapes and fold designs (§6).

Each test pins an observable outcome: the error a refusal raises (type and message),
bitwise parity with today's ``InteractionBuilder`` for a term shape the golden and
term-kind cases do not include, or the values a fold design writes for rows its
training rows never showed.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import polars as pl
import pytest
import rustystats.frozen_design as frozen_design
from _design_cases import CAT, LIN, _frame, _spec
from rustystats.exceptions import (
    DesignMatrixError,
    DesignTooWideError,
    SpillError,
    ValidationError,
)
from rustystats.frozen_design import (
    FoldRows,
    FrozenDesignUnsupported,
    TargetInputs,
    build_design_in_memory,
    build_design_matrix,
    check_design_width,
    check_key_dtypes,
    design_term_widths,
    in_memory_target_inputs,
    learn_frozen_design,
    learn_frozen_designs,
    learn_target_values,
)
from rustystats.interactions import InteractionBuilder
from test_frozen_design import _assert_bitwise, _assert_parity, _legacy, _parse


@pytest.fixture(scope="module")
def frame() -> pl.DataFrame:
    return _frame(n=600, seed=11)


def _encode(design: Any, data: pl.DataFrame, slice_rows: int = 128) -> np.ndarray:
    """Every row of ``data`` encoded slice by slice, each with its first row."""
    X = np.empty((data.height, design.p), dtype=np.float64)
    for start in range(0, data.height, slice_rows):
        rows = data.slice(start, slice_rows)
        design.encode_into(rows, X[start : start + rows.height], first_row=start)
    return X


# ── How the slices are given ────────────────────────────────────────────────


def test_a_one_shot_iterator_is_refused_before_learning(frame) -> None:
    parsed = _parse(_spec({"x1": LIN, "cat_a": CAT}))
    with pytest.raises(ValidationError, match="re-iterable"):
        learn_frozen_design(parsed, iter([frame]))


def test_a_callable_opening_each_pass_learns_as_a_list_does(frame) -> None:
    spec = _spec({"x2": {"type": "bs", "df": 5}, "cat_a": CAT})
    reference = learn_frozen_design(_parse(spec), [frame])
    opened: list[str] = []

    def by_kind(kind: str) -> list[pl.DataFrame]:
        opened.append(kind)
        return [frame]

    class Unreadable:
        """A callable whose signature cannot be read: it is called with no pass kind."""

        __signature__ = "unreadable"

        def __call__(self) -> list[pl.DataFrame]:
            opened.append("unnamed")
            return [frame]

    for slices in (lambda: [frame], by_kind, Unreadable()):
        design = learn_frozen_design(_parse(spec), slices)
        assert design.p == reference.p
        _assert_bitwise(_encode(design, frame), _encode(reference, frame), "X")
    assert opened[0] == "freeze" and "knots" in opened and "unnamed" in opened


def test_empty_slices_change_nothing(frame) -> None:
    spec = _spec({"x2": {"type": "bs", "df": 5}, "cat_a": CAT, "x1": LIN})
    empty = frame.head(0)
    reference = learn_frozen_design(_parse(spec), [frame])
    design = learn_frozen_design(_parse(spec), [empty, frame.head(300), empty, frame.tail(300)])
    _assert_bitwise(_encode(design, frame), _encode(reference, frame), "X")


# ── Refusals before anything is learned ─────────────────────────────────────


def test_learning_refusals(frame) -> None:
    parsed = _parse(_spec({"x1": LIN}))
    with pytest.raises(FrozenDesignUnsupported, match="encodes float64, not float32"):
        learn_frozen_design(parsed, [frame], np.float32)
    with pytest.raises(ValidationError, match="one fold \\(or None\\) per formula"):
        learn_frozen_designs([parsed], [frame], [None, None])
    with pytest.raises(FrozenDesignUnsupported, match="no rows to learn from"):
        learn_frozen_design(_parse(_spec({"x1": LIN})), [frame.head(0)])
    te = _parse(_spec({"brand": {"type": "target_encoding"}, "x1": LIN}))
    with pytest.raises(FrozenDesignUnsupported, match="learn from TargetInputs"):
        learn_frozen_design(te, [frame])


def test_a_fold_with_no_training_rows_is_refused(frame) -> None:
    rows = frame.head(1)
    fold = int(frozen_design.cv_fold_ids_py(0, 1, 2, 5)[0])
    parsed = [_parse(_spec({"x1": LIN})), _parse(_spec({"x1": LIN}))]
    with pytest.raises(ValidationError, match=f"fold {fold} has no training rows"):
        learn_frozen_designs(parsed, [rows], [None, FoldRows(fold, 2, 5)])


def test_key_dtypes_are_judged_from_the_schema_it_has(frame) -> None:
    parsed = _parse(_spec({"cat_a": CAT, "x1": LIN}))
    # A column the schema does not hold is judged when its first slice arrives.
    check_key_dtypes(parsed, {"x1": pl.Float64})
    check_key_dtypes(parsed, {"cat_a": pl.String})
    with pytest.raises(ValidationError, match="'cat_a' has dtype"):
        check_key_dtypes(parsed, {"cat_a": pl.List(pl.String)})


@pytest.mark.parametrize(
    ("terms", "message"),
    [
        ({"x2": {"type": "bs", "df": 4.5}}, "has a df or degree knots cannot use"),
        ({"x2": {"type": "bs", "df": 5, "boundary_knots": [1, 2, 3]}}, "unusable boundary"),
        ({"x2": {"type": "ns", "df": 4, "boundary_knots": ["a", "b"]}}, "unusable boundary"),
    ],
)
def test_splines_the_knot_functions_cannot_take_are_left_to_todays_builder(
    frame, terms, message
) -> None:
    spec = _spec({**terms, "x1": LIN})
    with pytest.raises(FrozenDesignUnsupported, match=message):
        learn_frozen_design(_parse(spec), [frame])
    try:
        expected = _legacy(frame, spec)
    except Exception as exc:  # today's builder refuses it too: the same error
        with pytest.raises(type(exc)):
            build_design_matrix(frame, _parse(spec))
        return
    _builder, y, X, names = build_design_matrix(frame, _parse(spec))
    _assert_bitwise(expected[1], y, "y")
    _assert_bitwise(expected[2], X, "X")
    assert names == expected[3]


def test_a_spline_with_knots_already_placed_keeps_them(frame) -> None:
    spec = _spec({"x2": {"type": "bs", "df": 6}, "x1": LIN})
    parsed = _parse(spec)
    _b, _y, full, _n = build_design_in_memory(frame, parsed)
    # The same formula again, from a third of the rows: its knots are not relearned.
    head = frame.head(200)
    _b, _y, again, _n = build_design_in_memory(head, parsed)
    _assert_bitwise(again, full[:200], "X from the placed knots")
    fresh = build_design_in_memory(head, _parse(spec))[2]
    assert not np.array_equal(fresh, full[:200]), "a third of the rows has other knots"


def test_boundary_knots_without_interior_knots_are_refused(frame) -> None:
    parsed = _parse(_spec({"x2": {"type": "bs", "df": 6}}))
    spline = parsed.spline_terms[0]
    spline._computed_boundary_knots = (18.0, 80.0)
    spline._computed_internal_knots = None
    with pytest.raises(FrozenDesignUnsupported, match="boundary but no interior knots"):
        learn_frozen_design(parsed, [frame])


def test_a_spline_basis_that_changed_after_the_layout_is_refused(frame) -> None:
    parsed = _parse(_spec({"x2": {"type": "bs", "df": 6}}))
    design = learn_frozen_design(parsed, [frame])
    spline = parsed.spline_terms[0]
    spline._computed_internal_knots = list(spline._computed_internal_knots[:-1])
    with pytest.raises(DesignMatrixError, match=r"evaluated to \d+ columns; its layout declares"):
        design.encode_into(frame, np.empty((frame.height, design.p)), first_row=0)


def test_slice_rows_must_be_positive(frame) -> None:
    with pytest.raises(ValidationError, match="slice_rows must be at least 1, got 0"):
        build_design_in_memory(frame, _parse(_spec({"x1": LIN})), slice_rows=0)


# ── Target-encoding settings (§6.6) ─────────────────────────────────────────


@pytest.mark.parametrize(
    ("setting", "message"),
    [
        ({"n_permutations": "many"}, "n_permutations must be a positive integer"),
        ({"n_permutations": 0}, "n_permutations must be a positive integer"),
        ({"prior_weight": "heavy"}, "numeric, non-negative, or 'auto'"),
        ({"prior_weight": [1.0]}, "numeric, non-negative, or 'auto'"),
        ({"prior_weight": -1.0}, "finite and non-negative"),
        ({"prior_weight": float("nan")}, "finite and non-negative"),
    ],
)
def test_target_encoding_settings_are_validated(frame, setting, message) -> None:
    spec = _spec({"brand": {"type": "target_encoding", **setting}, "x1": LIN})
    with pytest.raises(ValidationError, match=message):
        build_design_in_memory(frame, _parse(spec), seed=1)


def test_an_unseeded_build_draws_its_seed(frame, monkeypatch) -> None:
    spec = _spec({"brand": {"type": "target_encoding"}, "x1": LIN})
    monkeypatch.setattr(frozen_design, "draw_target_encoding_seed", lambda: 1234)
    drawn = build_design_in_memory(frame, _parse(spec))[2]
    _assert_bitwise(drawn, build_design_in_memory(frame, _parse(spec), seed=1234)[2], "X")


def test_target_encoded_designs_need_the_first_row(frame) -> None:
    spec = _spec({"brand": {"type": "target_encoding"}, "x1": LIN})
    parsed = _parse(spec)
    target = in_memory_target_inputs("y_pois", None, 3)
    design = learn_frozen_design(parsed, [frame], target=target)
    with pytest.raises(DesignMatrixError, match="encode_into needs first_row"):
        design.encode_into(frame, np.empty((frame.height, design.p)))


def test_a_design_learned_without_its_values_cannot_encode_until_they_are_sorted(
    frame,
) -> None:
    spec = _spec({"brand": {"type": "target_encoding"}, "x1": LIN})
    target = in_memory_target_inputs("y_pois", None, 3)
    (design,) = learn_frozen_designs([_parse(spec)], [frame], [None], target=target, values=[False])
    with pytest.raises(DesignMatrixError, match="has no training values yet"):
        design.encode_into(frame, np.empty((frame.height, design.p)), first_row=0)
    learn_target_values([design], [frame], target)
    expected = build_design_in_memory(frame, _parse(spec), seed=3)[2]
    _assert_bitwise(_encode(design, frame), expected, "X after the values are sorted")


def _spilling(tmp_path, keep) -> TargetInputs:
    base = in_memory_target_inputs("y_pois", None, 3)
    return TargetInputs(
        target=base.target,
        exposure=None,
        seed=3,
        scratch_dir=os.fspath(tmp_path),
        run_records=50,
        keep=keep,
    )


def test_spilled_training_values_equal_the_in_memory_ones(frame, tmp_path) -> None:
    spec = _spec({"brand": {"type": "target_encoding"}, "x1": LIN})
    kept: list[str] = []
    design = learn_frozen_design(_parse(spec), [frame], target=_spilling(tmp_path, kept.append))
    assert kept and all(os.path.exists(path) for path in kept)
    expected = build_design_in_memory(frame, _parse(spec), seed=3)[2]
    try:
        _assert_bitwise(_encode(design, frame, slice_rows=97), expected, "X from the spill")
    finally:
        design.close()
    with pytest.raises(DesignMatrixError, match="needs an owner"):
        learn_frozen_design(_parse(spec), [frame], target=_spilling(tmp_path, None))


def test_a_spill_file_that_vanishes_or_shrinks_is_refused(frame, tmp_path) -> None:
    spec = _spec({"brand": {"type": "target_encoding"}, "x1": LIN})
    kept: list[str] = []
    design = learn_frozen_design(_parse(spec), [frame], target=_spilling(tmp_path, kept.append))
    (path,) = kept
    with open(path, "r+b") as handle:
        handle.truncate(8 * (frame.height // 2))
    try:
        with pytest.raises(SpillError, match="ends before row"):
            _encode(design, frame)
    finally:
        design.close()
    design = learn_frozen_design(_parse(spec), [frame], target=_spilling(tmp_path, kept.append))
    os.remove(kept[-1])
    try:
        with pytest.raises(SpillError, match="Cannot read the spill file"):
            _encode(design, frame)
    finally:
        design.close()


# ── The training values of a fold, read from any row ────────────────────────


def test_training_values_read_from_an_unexpected_row_skip_the_rows_before() -> None:
    fold = FoldRows(1, 3, 17)
    n = 1000
    training = fold.training(0, n)
    stored = np.arange(int(training.sum()), dtype=np.float64)
    values = frozen_design._TrainingValues(fold, array=stored)
    # A pass that starts at row 600 (not where the last read ended) reads the
    # training rows after the 600 rows' own training rows.
    got = values.read(600, 100, training[600:700])
    first = int(training[:600].sum())
    np.testing.assert_array_equal(got, stored[first : first + int(training[600:700].sum())])
    with pytest.raises(DesignMatrixError, match="go past the"):
        values.read(n, 50, None)


# ── Term shapes the golden and term-kind cases leave out (parity with today) ─


def test_a_zero_maximum_is_positive_zero_whenever_one_occurs() -> None:
    """The maximum's sign of zero, like the minimum's, does not depend on the slices:
    +0.0 when any +0.0 occurs, −0.0 when every zero is −0.0 (today's SIMD fold picks
    by lane, so this boundary is not compared with today's builder)."""
    rng = np.random.default_rng(8)
    x = -rng.uniform(0.0, 3.0, 800)
    x[rng.random(800) < 0.2] = -0.0
    with_positive = x.copy()
    with_positive[rng.random(800) < 0.2] = 0.0
    spec = {"response": "y", "terms": {"x": {"type": "bs", "df": 6}}}
    for column, positive in ((with_positive, True), (x, False)):
        data = pl.DataFrame({"y": rng.poisson(1.0, 800).astype(float), "x": column})
        builds = [build_design_in_memory(data, _parse(spec), slice_rows=r) for r in (3, 64, 800)]
        bounds = [b[0]._fitted_splines["x"].get_knot_info()["boundary_knots"] for b in builds]
        for (_builder, _y, X, _names), bound in zip(builds, bounds):
            _assert_bitwise(builds[0][2], X, "X")
            assert bound[1] == 0.0 and np.signbit(bound[1]) != positive
            assert bound[0] == column.min()


SHAPES: dict[str, dict[str, Any]] = {
    "categorical_by_three_continuous": _spec(
        {"x1": LIN}, [{"cat_a": CAT, "x1": LIN, "x2": LIN, "x3": LIN}]
    ),
    "one_level_main_effect": _spec({"one": CAT, "x1": LIN}),
    "empty_levels": _spec({"cat_a": {"type": "categorical", "levels": []}, "x1": LIN}),
    "no_terms": _spec({}, intercept=False),
    "smooth_by_one_level_categorical": _spec(
        {"x1": LIN, "cat_a": CAT}, [{"x4": {"type": "s", "k": 6}, "one": CAT}]
    ),
    # A factor of no recognised type stays a raw continuous column, as in today's
    # build; under it a main-effect spline of the same variable is the factor's basis.
    "untyped_factor_with_a_local_spline": _spec(
        {"x1": LIN}, [{"x2": {"type": "bs", "df": 4}, "x1": {"type": "raw"}}]
    ),
    "untyped_factor_by_categorical_spline": _spec(
        {"x1": LIN, "cat_a": CAT},
        [{"x2": {"type": "bs", "df": 4}, "cat_a": CAT, "x1": {"type": "raw"}}],
    ),
    "untyped_factor_takes_the_main_spline": _spec(
        {"x2": {"type": "bs", "df": 4}, "x1": LIN}, [{"x2": {"type": "raw"}, "x1": LIN}]
    ),
}


@pytest.fixture(scope="module")
def shapes_frame(frame) -> pl.DataFrame:
    rng = np.random.default_rng(3)
    neg = -rng.uniform(0.0, 5.0, frame.height)
    neg[::7] = 0.0
    neg[::11] = -0.0
    return frame.with_columns(pl.Series("neg", neg))


@pytest.mark.parametrize("name", list(SHAPES))
def test_term_shape_design_equals_todays_builder(shapes_frame, name: str) -> None:
    _assert_parity(shapes_frame, SHAPES[name], slice_rows=97)


TARGET_SHAPES: dict[str, dict[str, Any]] = {
    # With a spline factor, today's build leaves a target-encoded factor out (D14).
    "spline_by_target_encoded": _spec(
        {"x1": LIN}, [{"x2": {"type": "bs", "df": 4}, "brand": {"type": "target_encoding"}}]
    ),
    "target_encoded_by_target_encoded": _spec(
        {"x1": LIN},
        [{"brand": {"type": "target_encoding"}, "cat_b": {"type": "target_encoding"}}],
    ),
}


@pytest.mark.parametrize("name", list(TARGET_SHAPES))
def test_target_encoded_interaction_shapes_equal_todays_builder(frame, name: str) -> None:
    spec = TARGET_SHAPES[name]
    y, X, names = InteractionBuilder(frame).build_design_matrix_from_parsed(_parse(spec), seed=5)
    _builder, new_y, new_X, new_names = build_design_in_memory(frame, _parse(spec), seed=5)
    _assert_bitwise(X, new_X, "X")
    _assert_bitwise(y, new_y, "y")
    assert new_names == names


# ── Design widths before the layout (§6.4) ──────────────────────────────────


WIDTH_SPECS: dict[str, dict[str, Any]] = {
    **SHAPES,
    **TARGET_SHAPES,
    "spline_with_a_zero_maximum": _spec({"neg": {"type": "bs", "df": 5}, "x1": LIN}),
    "given_knots_no_intercept": _spec(
        {"x2": {"type": "bs", "knots": [30.0, 50.0, 65.0]}, "cat_a": CAT}, intercept=False
    ),
    "target_encoded_by_continuous": _spec(
        {"x1": LIN}, [{"brand": {"type": "target_encoding"}, "x3": LIN}]
    ),
    "linear_only_interaction": _spec({"x1": LIN}, [{"x1": LIN, "x2": LIN}]),
    "spline_by_one_level_categorical": _spec(
        {"x1": LIN}, [{"x3": {"type": "bs", "df": 4}, "one": CAT}]
    ),
}


@pytest.mark.parametrize("name", list(WIDTH_SPECS))
def test_term_widths_from_the_labels_equal_the_learned_design(shapes_frame, name) -> None:
    spec = WIDTH_SPECS[name]
    design = build_design_in_memory(shapes_frame, _parse(spec), seed=5)
    p = design[2].shape[1]
    learned = learn_frozen_design(
        _parse(spec), [shapes_frame], target=in_memory_target_inputs("y_pois", None, 5)
    )

    def labels(column: str) -> int:
        return len(learned.categories[column].labels)

    widths = design_term_widths(_parse(spec), labels)
    # An empty model is one column of ones, which has no term.
    assert sum(width for _term, width in widths) == (p if widths else 0)
    if widths:
        check_design_width(_parse(spec), labels, p)
        with pytest.raises(DesignTooWideError, match=f"at least p = {p:,} columns"):
            check_design_width(_parse(spec), labels, p - 1)


def test_designs_learned_together_equal_designs_learned_alone(frame) -> None:
    """One pass set serves designs with and without knots to place."""
    with_spline = _spec({"x2": {"type": "bs", "df": 6}, "cat_a": CAT})
    without = _spec({"x1": LIN, "cat_b": CAT})
    together = learn_frozen_designs([_parse(without), _parse(with_spline)], [frame], [None, None])
    for design, spec in zip(together, (without, with_spline)):
        alone = learn_frozen_design(_parse(spec), [frame])
        _assert_bitwise(_encode(design, frame), _encode(alone, frame), "X")


# ── A fold's design meets labels its training rows never showed ─────────────


@pytest.fixture(scope="module")
def rare(frame) -> tuple[pl.DataFrame, FoldRows, int]:
    """The frame with a column whose label ``Z`` is on one row only, and the fold
    whose validation rows hold that row."""
    labels = np.array(["A", "B", "C"])[np.arange(frame.height) % 3]
    row = 101
    labels[row] = "Z"
    data = frame.with_columns(pl.Series("rare", labels))
    fold = FoldRows(int(frozen_design.cv_fold_ids_py(row, 1, 3, 9)[0]), 3, 9)
    return data, fold, row


def _fold_design(data: pl.DataFrame, fold: FoldRows, spec: dict) -> Any:
    _full, design = learn_frozen_designs([_parse(spec), _parse(spec)], [data], [None, fold])
    return design


def test_a_fold_counts_an_unseen_label_or_combination_as_zero(rare) -> None:
    data, fold, row = rare
    spec = _spec(
        {"rare": {"type": "frequency_encoding"}, "x1": LIN},
        [{"rare": CAT, "cat_a": CAT, "frequency_encoding": True}],
    )
    design = _fold_design(data, fold, spec)
    X = _encode(design, data)
    names = design.names
    single, combined = names.index("FE(rare)"), len(names) - 1
    assert X[row, single] == 0.0 and X[row, combined] == 0.0
    training = fold.training(0, data.height)
    counts = data.filter(pl.Series(training))["rare"].value_counts()
    a = counts.filter(pl.col("rare") == "A")["count"][0]
    first_a = int(np.flatnonzero(data["rare"].to_numpy() == "A")[0])
    assert X[first_a, single] == a / counts["count"].max()


def test_a_full_design_refuses_a_combination_it_did_not_learn(rare) -> None:
    data, _fold, _row = rare
    spec = _spec({"x1": LIN}, [{"rare": CAT, "cat_a": CAT, "frequency_encoding": True}])
    # Learned without the pair (A, B): both labels are known, the pair is not.
    unseen = (pl.col("rare") == "A") & (pl.col("cat_a") == "B")
    design = learn_frozen_design(_parse(spec), [data.filter(~unseen)])
    rows = data.filter(unseen).head(1)
    with pytest.raises(DesignMatrixError, match="met a combination of labels that was not learned"):
        design.encode_into(rows, np.empty((1, design.p)), first_row=0)


def test_a_fold_writes_no_indicator_or_curve_for_an_unseen_label(rare) -> None:
    data, fold, row = rare
    spec = _spec(
        {"rare": {"type": "categorical", "levels": ["A", "Z"]}, "x1": LIN, "cat_b": CAT},
        [{"x2": {"type": "bs", "k": 6, "monotonicity": "increasing"}, "cat_b": CAT}],
    )
    # The monotone curve per category reads the row's own level: give row ``row`` a
    # level of its own too.
    data = data.with_columns(
        pl.when(pl.int_range(pl.len()) == row)
        .then(pl.lit("only_here"))
        .otherwise(pl.col("cat_b"))
        .alias("cat_b")
    )
    design = _fold_design(data, fold, spec)
    X = _encode(design, data)
    names = design.names
    assert X[row, names.index("rare[A]")] == 0.0 and X[row, names.index("rare[Z]")] == 0.0
    curve = [i for i, name in enumerate(names) if name.startswith("cat_b[") and ":" in name]
    assert curve and not X[row, curve].any(), "no category's curve for an unseen level"
    seen = (row + 1) % data.height
    assert X[seen, curve].any(), "a seen level has its curve"
    assert X[seen, names.index("rare[A]")] == float(data["rare"][seen] == "A")


def test_a_fold_reads_its_training_values_one_row_at_a_time(frame) -> None:
    spec = _spec({"brand": {"type": "target_encoding"}, "x1": LIN})
    fold = FoldRows(0, 3, 21)
    target = in_memory_target_inputs("y_pois", None, 4)
    design = learn_frozen_designs([_parse(spec)], [frame], [fold], target=target)[0]
    by_row = _encode(design, frame, slice_rows=1)
    _assert_bitwise(by_row, _encode(design, frame, slice_rows=128), "X")
