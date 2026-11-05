"""Frozen-design encoder parity (``specs/out-of-core-fit.md`` §6.8, §14 T1).

A model without target encoding builds its design through a frozen design learned from
slices, whose terms write each slice straight into one preallocated X. These tests pin
that the result equals ``InteractionBuilder.build_design_matrix_from_parsed`` bit for
bit: X, y, the names, and every piece of builder state that fitting, prediction and
serialisation read. They also pin the build's memory: X and the declared scratch.
"""

from __future__ import annotations

import json
import struct
import tracemalloc
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
import rustystats as rs
import rustystats._rustystats as native
import rustystats.frozen_design as frozen_design
import rustystats.interactions as interactions
import rustystats.splines as splines
from _design_cases import (
    CAT,
    DIGEST_SCHEMA,
    LIN,
    TERM_KINDS,
    _frame,
    _spec,
    design_record,
    design_specs,
)
from _golden_cases import CASES
from rustystats.category_keys import ENTRY_OVERHEAD_BYTES, LearnedStateBudget
from rustystats.exceptions import (
    DesignMatrixError,
    EncodingError,
    LearnedStateTooLargeError,
    ValidationError,
)
from rustystats.formula import dict_to_parsed_formula
from rustystats.frozen_design import (
    E_LEARN,
    FrozenDesignUnsupported,
    build_design_in_memory,
    build_design_matrix,
    learn_frozen_design,
)
from rustystats.interactions import FrequencyEncodingTermSpec, InteractionBuilder

# ── Comparison ──────────────────────────────────────────────────────────────


def _assert_bitwise(a: np.ndarray, b: np.ndarray, what: str) -> None:
    assert a.shape == b.shape, f"{what}: shape {a.shape} != {b.shape}"
    assert a.dtype == b.dtype, f"{what}: dtype {a.dtype} != {b.dtype}"
    a_bits = np.ascontiguousarray(a).view(np.uint8)
    b_bits = np.ascontiguousarray(b).view(np.uint8)
    if a.dtype == np.float64:
        a_bits = np.ascontiguousarray(a).view(np.uint64)
        b_bits = np.ascontiguousarray(b).view(np.uint64)
    assert np.array_equal(a_bits, b_bits), f"{what}: values differ"


def _assert_same(a: Any, b: Any, path: str, seen: set[tuple[int, int]] | None = None) -> None:
    """Deep equality: arrays and floats bitwise, objects by their attributes."""
    seen = set() if seen is None else seen
    if (id(a), id(b)) in seen:
        return
    seen.add((id(a), id(b)))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        assert isinstance(a, np.ndarray) and isinstance(b, np.ndarray), path
        _assert_bitwise(a, b, path)
    elif isinstance(a, float) or isinstance(b, float):
        assert type(a) is type(b), f"{path}: {type(a)} != {type(b)}"
        assert struct.pack("<d", a) == struct.pack("<d", b), f"{path}: {a!r} != {b!r}"
    elif isinstance(a, pl.DataFrame):
        assert a is b, path
    elif isinstance(a, dict):
        assert type(a) is type(b), path
        assert set(a) == set(b), f"{path}: keys {sorted(map(str, set(a) ^ set(b)))}"
        for key in a:
            _assert_same(a[key], b[key], f"{path}[{key!r}]", seen)
    elif isinstance(a, list | tuple):
        assert type(a) is type(b) and len(a) == len(b), f"{path}: {a!r} != {b!r}"
        for index, (left, right) in enumerate(zip(a, b)):
            _assert_same(left, right, f"{path}[{index}]", seen)
    elif hasattr(a, "__dict__") and not isinstance(a, type):
        assert type(a) is type(b), f"{path}: {type(a)} != {type(b)}"
        _assert_same(vars(a), vars(b), path, seen)
    else:
        assert type(a) is type(b) and a == b, f"{path}: {a!r} != {b!r}"


def _parse(spec: dict[str, Any]) -> Any:
    return dict_to_parsed_formula(
        response=spec["response"],
        terms=spec["terms"],
        interactions=spec.get("interactions"),
        intercept=spec.get("intercept", True),
    )


def _legacy(data: pl.DataFrame, spec: dict[str, Any]) -> tuple:
    builder = InteractionBuilder(data)
    y, X, names = builder.build_design_matrix_from_parsed(_parse(spec))
    return builder, y, X, names


def _assert_parity(data: pl.DataFrame, spec: dict[str, Any], **build: Any) -> np.ndarray:
    """The frozen build equals today's builder: X, y, names and builder state."""
    old_builder, old_y, old_X, old_names = _legacy(data, spec)
    parsed = _parse(spec)
    assert not parsed.target_encoding_terms
    new_builder, new_y, new_X, new_names = build_design_in_memory(data, parsed, **build)

    _assert_bitwise(old_X, new_X, "X")
    assert new_X.flags.c_contiguous
    _assert_bitwise(old_y, new_y, "y")
    assert new_names == old_names
    assert type(new_builder) is InteractionBuilder
    assert new_builder.data is data

    # State downstream code reads before fitting: levels, names, slots, splines.
    assert list(new_builder._cat_encoding_cache) == list(old_builder._cat_encoding_cache)
    for key, cached in old_builder._cat_encoding_cache.items():
        assert new_builder._cat_encoding_cache[key].levels == cached.levels
        assert new_builder._cat_encoding_cache[key].names == cached.names
    _assert_same(old_builder._term_slots, new_builder._term_slots, "term_slots")
    _assert_same(old_builder.get_smooth_terms(), new_builder.get_smooth_terms(), "smooth")
    _assert_same(old_builder.get_all_spline_terms(), new_builder.get_all_spline_terms(), "all")
    _assert_same(old_builder.get_spline_info(), new_builder.get_spline_info(), "spline_info")

    # Everything a fitted model keeps (fitting clears the caches before storing it),
    # which prediction and to_bytes read.
    old_builder.clear_caches()
    new_builder.clear_caches()
    _assert_same(vars(old_builder), vars(new_builder), "builder")
    return new_X


# ── Golden cases ────────────────────────────────────────────────────────────


def _non_te_golden_cases() -> list[str]:
    names = []
    for case in CASES:
        kwargs, _fit, _holdout = case.build()
        if not _parse(kwargs).target_encoding_terms:
            names.append(case.name)
    return names


_GOLDEN = {case.name: case for case in CASES}


@pytest.mark.parametrize("name", _non_te_golden_cases())
def test_golden_case_design_equals_todays_builder(name: str) -> None:
    kwargs, _fit, _holdout = _GOLDEN[name].build()
    _assert_parity(kwargs["data"], kwargs)


# ── Every term and interaction kind ─────────────────────────────────────────


# The frame and every term kind live in ``_design_cases`` (shared with the digest capture).


@pytest.fixture(scope="module")
def frame() -> pl.DataFrame:
    return _frame()


@pytest.mark.parametrize("name", list(TERM_KINDS))
def test_term_kind_design_equals_todays_builder(frame: pl.DataFrame, name: str) -> None:
    _assert_parity(frame, TERM_KINDS[name], slice_rows=1024)


# ── Pinned digests of the v0.9.0 builder's designs (T1) ─────────────────────

_DIGESTS = Path(__file__).resolve().parents[1] / "golden" / "design_digests_v0_9_0.json"


def _pinned() -> dict[str, Any]:
    document = json.loads(_DIGESTS.read_text(encoding="utf-8"))
    assert document["schema"] == DIGEST_SCHEMA
    assert document["rustystats_version"] == "0.9.0"
    return document["designs"]


_SPECS = design_specs()


def test_every_design_case_has_a_pinned_digest() -> None:
    assert set(_pinned()) == set(_SPECS)


@pytest.mark.parametrize("name", sorted(_SPECS))
@pytest.mark.parametrize("slice_rows", [None, 1024])
def test_frozen_design_matches_the_pinned_v0_9_0_digest(name: str, slice_rows) -> None:
    """X, y and the names hash to what v0.9.0's builder gave (not the retained builder)."""
    data, spec = _SPECS[name]
    build = {} if slice_rows is None else {"slice_rows": slice_rows}
    _builder, y, X, names = build_design_in_memory(data, _parse(spec), **build)
    assert X.dtype == np.float64 and X.flags.c_contiguous
    assert design_record(y, X, names) == _pinned()[name]


def _rich_spec() -> dict[str, Any]:
    """One model with every term kind, for the slicing tests."""
    return _spec(
        {
            "cat_a": CAT,
            "cat_int": CAT,
            "f_special": CAT,
            "x1": LIN,
            "x2": {"type": "bs", "df": 5},
            "x3": {"type": "ns", "k": 5},
            "x4": {"type": "bs", "knots": [-0.5, 0.5]},
            "brand": {"type": "frequency_encoding"},
            "bool_null": {"type": "categorical", "levels": ["True", "None"]},
            "x1_sq": {"type": "expression", "expr": "x1 ** 2"},
            "exposure": {"type": "linear", "monotonicity": "increasing"},
        },
        [
            {"cat_a": CAT, "cat_b": CAT},
            {"cat_a": CAT, "flag": CAT, "x1": LIN},
            {"weight": {"type": "bs", "k": 5, "monotonicity": "increasing"}, "cat_a": CAT},
            {"x3": {"type": "bs", "df": 4}, "x1": LIN},
            {"cat_b": CAT, "str_null": CAT, "frequency_encoding": True},
        ],
    )


# Each slice costs a fixed overhead (Polars queries, one encoder), so the tiny slice
# sizes run over fewer rows; every learning and encoding step still meets each slice.
_SLICE_FRAME_ROWS = {1: 150, 7: 600}


@pytest.mark.parametrize("slice_rows", [1, 7, 4096, None])
def test_slice_size_does_not_change_the_design(slice_rows: int | None) -> None:
    data = _frame(n=4500, seed=3).head(_SLICE_FRAME_ROWS.get(slice_rows, 4500))
    _assert_parity(data, _rich_spec(), slice_rows=slice_rows or data.height)


def test_default_slicing_crosses_the_learning_slice() -> None:
    data = _frame(n=E_LEARN + 4321, seed=11)
    _assert_parity(data, _rich_spec())


def test_encode_into_writes_a_slice_in_place(frame: pl.DataFrame) -> None:
    spec = _rich_spec()
    _builder, _y, X, names = _legacy(frame, spec)
    design = learn_frozen_design(_parse(spec), [frame.slice(0, 1000), frame.slice(1000)])
    out = np.full((700, X.shape[1]), np.nan)
    assert design.encode_into(frame.slice(1500, 700), out) == names
    _assert_bitwise(X[1500:2200], out, "encode_into")


def test_layout_is_declared_before_any_row_is_encoded(
    frame: pl.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Names, slots and p follow from the learned state; no slice is built as term blocks."""
    spec = _rich_spec()
    old_builder, _y, old_X, old_names = _legacy(frame, spec)

    def no_blocks(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the frozen design built a slice as stacked term blocks")

    monkeypatch.setattr(InteractionBuilder, "_build_design_matrix_core", no_blocks)
    monkeypatch.setattr(InteractionBuilder, "_stack_columns", staticmethod(no_blocks))

    parsed = _parse(spec)
    design = learn_frozen_design(parsed, [frame.slice(0, 1000), frame.slice(1000)])
    assert list(design.names) == old_names
    assert design.p == old_X.shape[1]
    _assert_same(old_builder._term_slots, list(design.term_slots), "term_slots")
    # Every spline the layout reaches holds its knots; none is left waiting for one.
    for spline in parsed.spline_terms:
        assert spline._computed_boundary_knots is not None
        assert not hasattr(spline, "_frozen_knots")

    _builder, _y, X, _names = build_design_in_memory(frame, _parse(spec), slice_rows=700)
    _assert_bitwise(old_X, X, "X")


def test_encode_into_needs_a_c_order_float64_block(frame: pl.DataFrame) -> None:
    design = learn_frozen_design(_parse(_rich_spec()), [frame])
    rows = frame.slice(0, 10)
    for out in (
        np.empty((10, design.p), order="F"),
        np.empty((10, design.p), dtype=np.float32),
        np.empty((9, design.p)),
    ):
        with pytest.raises(DesignMatrixError, match="C-order float64 block"):
            design.encode_into(rows, out)


def test_frequency_counts_are_charged_before_the_table_grows() -> None:
    """§6.4: a slice's new key tuples are charged first, and refused without merging."""
    # A two-byte string key, its representative value, and its count.
    entry = 2 * 2 + ENTRY_OVERHEAD_BYTES + 8
    budget = LearnedStateBudget(3 * entry)
    learner = frozen_design._FrequencyLearner(FrequencyEncodingTermSpec("x"), budget)

    learner.observe(pl.DataFrame({"x": ["aa", "bb", "aa"]}))
    assert budget.used == 2 * entry
    learner.observe(pl.DataFrame({"x": ["bb", "aa"]}))  # known tuples only add counts
    assert budget.used == 2 * entry

    table = learner._table
    with pytest.raises(LearnedStateTooLargeError, match=r"'x' \(4 distinct values\)"):
        learner.observe(pl.DataFrame({"x": ["cc", "dd", "aa"]}))
    assert learner._table is table
    assert budget.used == 2 * entry

    # A null key has no payload; a tuple with nulls is known once learned.
    nulls = pl.DataFrame({"x": [None, "aa"]}, schema={"x": pl.String})
    learner.observe(nulls)
    assert budget.used == 2 * entry + ENTRY_OVERHEAD_BYTES + 8
    pair = frozen_design._FrequencyLearner(
        FrequencyEncodingTermSpec("x:z", ["x", "z"]), LearnedStateBudget()
    )
    with_nulls = pl.DataFrame(
        {"x": ["a", None, None], "z": [None, 1, None]}, schema={"x": pl.String, "z": pl.Int64}
    )
    pair.observe(with_nulls)
    used = pair._budget.used
    pair.observe(with_nulls)
    assert pair._budget.used == used
    assert pair._table is not None and pair._table["count"].to_list() == [2, 2, 2]


# ── Memory (§6.8, §11) ──────────────────────────────────────────────────────

#: Traced bytes beyond X and the declared scratch: names, slots, labels and the Python
#: objects of each slice's Polars calls.
_ALLOWANCE = 1 << 20


def _kernel_outputs_in_numpy(monkeypatch: pytest.MonkeyPatch) -> Counter[str]:
    """Copy every Rust kernel's arrays into NumPy memory, so tracemalloc counts them.

    An array a Rust kernel returns owns a Rust allocation, which tracemalloc does not
    trace. Replacing it with a NumPy copy makes every block and basis a kernel builds
    count toward the traced peak. Returns the calls made, by kernel.
    """
    calls: Counter[str] = Counter()

    def traced(kernel: Callable[..., Any]) -> Callable[..., Any]:
        def copy_arrays(value: Any) -> Any:
            return np.array(value) if isinstance(value, np.ndarray) else value

        def call(*args: Any, **kwargs: Any) -> Any:
            calls[kernel.__name__] += 1
            result = kernel(*args, **kwargs)
            if isinstance(result, tuple):
                return tuple(copy_arrays(item) for item in result)
            return copy_arrays(result)

        return call

    for module in (interactions, splines):
        for name, value in list(vars(module).items()):
            if getattr(native, getattr(value, "__name__", ""), None) is value:
                monkeypatch.setattr(module, name, traced(value))
    return calls


def _traced_peak(build: Callable[[], Any]) -> tuple[Any, int]:
    """``build()``, and the peak of the memory it traced."""
    assert not tracemalloc.is_tracing()
    tracemalloc.start()
    try:
        start = tracemalloc.get_traced_memory()[0]
        return build(), tracemalloc.get_traced_memory()[1] - start
    finally:
        tracemalloc.stop()


def _memory_frame(n: int) -> pl.DataFrame:
    rng = np.random.default_rng(17)

    def categorical(levels: int) -> np.ndarray:
        # Every slice holds every level, so today's builder gives each slice one width.
        codes = rng.permutation(np.arange(n) % levels)
        return np.array([f"l{level:02d}" for level in range(levels)])[codes]

    return pl.DataFrame(
        {
            "y": rng.poisson(1.0, n).astype(np.float64),
            "a": categorical(6),
            "b": categorical(10),
            "c": categorical(16),
            "x1": rng.normal(size=n),
            "x2": rng.uniform(0.0, 10.0, n),
            "x3": rng.uniform(1.0, 5.0, n),
        }
    )


_MEMORY_SPEC = {
    "response": "y",
    "terms": {
        "a": CAT,
        "b": CAT,
        "c": CAT,
        "x1": LIN,
        "x2": {"type": "bs", "df": 6},
        "x3": {"type": "ns", "df": 5},
    },
    "interactions": [
        {"a": CAT, "x2": {"type": "bs", "df": 4}, "include_main": False},
        {"b": CAT, "x1": LIN, "include_main": False},
    ],
}


def test_in_memory_build_holds_x_and_the_declared_scratch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Peak traced memory is X plus one slice's declared scratch, not slice-sized blocks."""
    n, slice_rows = 250_000, E_LEARN
    data = _memory_frame(n)
    calls = _kernel_outputs_in_numpy(monkeypatch)

    (_builder, _y, X, _names), peak = _traced_peak(
        lambda: build_design_in_memory(data, _parse(_MEMORY_SPEC), slice_rows=slice_rows)
    )
    # Spline bases came through the kernels, now traced.
    assert calls["bs_knots_py"] and calls["ns_with_knots_py"]
    slices = [data.slice(start, slice_rows) for start in range(0, n, slice_rows)]
    design = learn_frozen_design(_parse(_MEMORY_SPEC), slices)
    assert design.p == X.shape[1]
    # The scratch is a fraction of a row of X: no term holds a slice-sized block.
    assert design.scratch_bytes_per_row < X.shape[1] * X.itemsize / 2
    assert peak <= X.nbytes + slice_rows * design.scratch_bytes_per_row + _ALLOWANCE

    def blocks_per_slice() -> np.ndarray:
        """The encoding this replaces: each slice built as term blocks, stacked, copied."""
        parsed = _parse(_MEMORY_SPEC)
        out = np.empty_like(X)
        for start in range(0, n, slice_rows):
            rows = data.slice(start, slice_rows)
            _y, block, _names = InteractionBuilder(rows).build_design_matrix_from_parsed(parsed)
            out[start : start + rows.height] = block
        return out

    _out, old_peak = _traced_peak(blocks_per_slice)
    assert old_peak > peak + slice_rows * X.shape[1] * X.itemsize


# ── Routing and fallback ────────────────────────────────────────────────────


def test_glm_dict_learns_a_frozen_design(
    frame: pl.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[object] = []
    learn = frozen_design.learn_frozen_design

    def spy(*args: Any, **kwargs: Any) -> Any:
        calls.append(args[0])
        return learn(*args, **kwargs)

    monkeypatch.setattr(frozen_design, "learn_frozen_design", spy)
    spec = TERM_KINDS["spline_cat"]
    model = rs.glm_dict(
        response="y_pois",
        terms=spec["terms"],
        interactions=spec["interactions"],
        data=frame,
        family="poisson",
    )
    assert len(calls) == 1
    _builder, _y, old_X, old_names = _legacy(frame, spec)
    _assert_bitwise(old_X, model.X, "glm_dict X")
    assert model.feature_names == old_names

    # A target-encoded model too (§6.6); InteractionBuilder computes the same values.
    te_spec = {"response": "y_pois", "terms": {"brand": {"type": "target_encoding"}, "x1": LIN}}
    model = rs.glm_dict(**te_spec, data=frame, family="poisson", seed=1)
    assert len(calls) == 2
    _y, old_X, old_names = InteractionBuilder(frame).build_design_matrix_from_parsed(
        _parse(te_spec), seed=1
    )
    _assert_bitwise(old_X, model.X, "target-encoded X")
    assert model.feature_names == old_names


@pytest.mark.parametrize(
    ("reason", "data", "terms"),
    [
        (
            "non-finite spline input",
            pl.DataFrame({"y": [1.0, 2.0, 3.0, 4.0, 5.0], "x": [1.0, np.nan, 3.0, 4.0, 2.0]}),
            {"x": {"type": "bs", "df": 4}},
        ),
    ],
)
def test_unsupported_models_keep_todays_builder(
    reason: str, data: pl.DataFrame, terms: dict[str, Any]
) -> None:
    spec = {"response": "y", "terms": terms}
    parsed = _parse(spec)
    with pytest.raises(FrozenDesignUnsupported):
        learn_frozen_design(parsed, [data])
    # Nothing was applied to the formula before the refusal.
    for spline in parsed.spline_terms:
        assert spline._computed_boundary_knots is None
        assert not hasattr(spline, "_frozen_knots")

    _old_builder, old_y, old_X, old_names = _legacy(data, spec)
    _builder, y, X, names = build_design_matrix(data, parsed)
    _assert_bitwise(old_X, X, reason)
    _assert_bitwise(old_y, y, reason)
    assert names == old_names


def test_nested_key_dtypes_are_refused_in_both_builds() -> None:
    """DEC-5: a nested categorical input is refused, not handed to today's builder."""
    data = pl.DataFrame({"y": [1.0, 2.0, 3.0], "c": [[1], [2, 3], [1]]})
    parsed = _parse({"response": "y", "terms": {"c": CAT}})
    with pytest.raises(ValidationError, match="cannot be a categorical"):
        learn_frozen_design(parsed, [data])
    with pytest.raises(ValidationError, match="cannot be a categorical"):
        build_design_matrix(data, parsed)


def test_zero_boundary_does_not_depend_on_slicing() -> None:
    """A zero extreme with both signs: −0.0 below +0.0, whatever the slices.

    Today's minimum is a SIMD fold whose sign of zero depends on which lane each value
    lands in, so this is the one boundary that is not compared with today's builder.
    """
    rng = np.random.default_rng(5)
    x = rng.uniform(0.0, 3.0, 800)
    x[rng.random(800) < 0.2] = 0.0
    x[rng.random(800) < 0.2] = -0.0
    x[rng.random(800) < 0.1] = 3.0
    data = pl.DataFrame({"y": rng.poisson(1.0, 800).astype(float), "x": x})
    spec = {"response": "y", "terms": {"x": {"type": "bs", "df": 6}}}
    designs = [build_design_in_memory(data, _parse(spec), slice_rows=rows) for rows in (3, 64, 800)]
    knots = [design[0]._fitted_splines["x"].get_knot_info() for design in designs]
    for info, (_builder, _y, X, _names) in zip(knots, designs):
        _assert_same(knots[0], info, "knots")
        _assert_bitwise(designs[0][2], X, "X")
    assert np.signbit(knots[0]["boundary_knots"][0])
    assert knots[0]["boundary_knots"][1] == 3.0


def test_levels_on_non_ascii_binary_raise_encoding_error(frame: pl.DataFrame) -> None:
    """Today ``astype(str)`` raises ``UnicodeDecodeError``; §6.3 names it ``EncodingError``."""
    spec = _spec({"bin": {"type": "categorical", "levels": ["a"]}, "x1": LIN})
    with pytest.raises(UnicodeDecodeError):
        _legacy(frame, spec)
    with pytest.raises(EncodingError, match="'bin'"):
        build_design_in_memory(frame, _parse(spec))
