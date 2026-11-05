"""Design cases for frozen-design parity and the pinned design digests (§14 T1).

The frame has one column per categorical dtype class of §6.3 beside the golden frame's
columns, and ``TERM_KINDS`` holds a model of every term and interaction kind. The module
needs only NumPy and Polars, so ``scripts/capture_design_digests.py`` can build these
designs with another RustyStats build (v0.9.0) to pin their digests.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from typing import Any

import numpy as np
import polars as pl
from _golden_cases import CASES, design_frame


def _frame(n: int = 3000, seed: int = 7) -> pl.DataFrame:
    """The golden frame plus one column per categorical dtype class (§6.3)."""
    rng = np.random.default_rng(seed)
    base = design_frame(n, seed)

    def pick(values: list[Any], null_rate: float = 0.0) -> list[Any]:
        chosen = [values[i] for i in rng.integers(0, len(values), n)]
        if null_rate:
            chosen = [None if rng.random() < null_rate else v for v in chosen]
        return chosen

    big = 2**53
    return base.with_columns(
        pl.Series("bool_null", pick([True, False], 0.1), dtype=pl.Boolean),
        pl.Series("bin", pick([b"a", b"b", b"c\x00", b"\xff"]), dtype=pl.Binary),
        pl.Series("bin_ascii", pick([b"a", b"b", b"c\x00"]), dtype=pl.Binary),
        pl.Series("f_special", pick([np.nan, -0.0, 0.0, 1.5, 2.5]), dtype=pl.Float64),
        pl.Series("big_int_null", pick([big, big + 1, big + 2, 7], 0.1), dtype=pl.Int64),
        pl.Series("str_null", pick(["s1", "s2", "s3"], 0.1), dtype=pl.String),
        pl.Series("cat_dtype", pick(["p", "q", "r"]), dtype=pl.Categorical),
        pl.Series("enum_dtype", pick(["e1", "e2"]), dtype=pl.Enum(["e2", "e1", "e3"])),
        pl.Series("day", pick([date(2020, 1, d) for d in (1, 2, 3)]), dtype=pl.Date),
        pl.Series("int_plain", pick([1, 2, 3, 4, 5, 6]), dtype=pl.Int64),
        pl.Series("colon", pick(["a:b", "a", "b"]), dtype=pl.String),
        pl.Series("colon2", pick(["c", "b:c", "x"]), dtype=pl.String),
        pl.Series("one", ["only"] * n, dtype=pl.String),
    )


CAT = {"type": "categorical"}
LIN = {"type": "linear"}
CAT_DTYPES = (
    "cat_int",
    "flag",
    "bool_null",
    "bin",
    "f_special",
    "big_int_null",
    "str_null",
    "cat_dtype",
    "enum_dtype",
    "day",
    "int_plain",
    "colon",
)


def _spec(terms: dict[str, Any], interactions: list[dict[str, Any]] | None = None, **kw: Any):
    return {"response": "y_pois", "terms": terms, "interactions": interactions, **kw}


TERM_KINDS: dict[str, dict[str, Any]] = {
    "discovered_categorical_dtypes": _spec({column: CAT for column in CAT_DTYPES}),
    "levels": _spec(
        {
            "cat_int": {"type": "categorical", "levels": ["1.0", "3.0", "nan"]},
            "flag": {"type": "categorical", "levels": ["True"]},
            "bool_null": {"type": "categorical", "levels": ["False", "None"]},
            "bin_ascii": {"type": "categorical", "levels": ["a", "c"]},
            "f_special": {"type": "categorical", "levels": ["nan", "-0.0", "0.0", "2.5"]},
            "big_int_null": {"type": "categorical", "levels": ["9007199254740992.0", "nan"]},
            "str_null": {"type": "categorical", "levels": ["s1", "None"]},
            "cat_dtype": {"type": "categorical", "levels": ["p"]},
            "day": {"type": "categorical", "levels": ["2020-01-02"]},
            "int_plain": {"type": "categorical", "levels": ["2", 3]},
            "cat_b": {"type": "categorical", "levels": ["b1", "b9"]},
            "x1": LIN,
        },
        # cat_int is also a discovered factor here, so one learner serves both renderers.
        [{"cat_int": CAT, "x1": LIN}],
    ),
    "cat_cat_two_way": _spec({"x1": LIN}, [{"cat_a": CAT, "cat_b": CAT, "include_main": True}]),
    "cat_n_way": _spec(
        {"x1": LIN},
        [
            {"cat_a": CAT, "flag": CAT, "cat_int": CAT},
            {"cat_a": CAT, "cat_b": CAT, "bool_null": CAT, "str_null": CAT},
            {"one": CAT, "cat_a": CAT, "flag": CAT},
        ],
    ),
    "cat_cont": _spec(
        {"x2": LIN},
        [
            {"cat_a": CAT, "x1": LIN, "include_main": True},
            {"cat_int": CAT, "x1": LIN, "x2": LIN},
            {"cat_a": CAT, "cat_b": CAT, "x1": LIN},
            {"cat_a": CAT, "flag": CAT, "bin": CAT, "x3": LIN},
            {"one": CAT, "x1": LIN},
        ],
    ),
    "cont_cont": _spec(
        {"x1": LIN},
        [{"x1": LIN, "x2": LIN}, {"x1": LIN, "x2": LIN, "x3": LIN}, {"int_plain": LIN, "x4": LIN}],
    ),
    "spline_cont": _spec(
        {"x1": LIN},
        [
            {"x2": {"type": "bs", "df": 5}, "x1": LIN},
            {"x3": {"type": "ns", "df": 4}, "x1": LIN, "x4": LIN},
            {"x2": {"type": "bs", "k": 6}, "x1": LIN},
            {"x2": {"type": "bs", "df": 4}, "x3": {"type": "ns", "df": 3}},
            {"x2": {"type": "bs", "df": 4}, "x3": {"type": "ns", "df": 3}, "x1": LIN},
        ],
    ),
    "spline_cat": _spec(
        {"x1": LIN, "cat_a": CAT},
        [
            {"x2": {"type": "bs", "df": 5}, "cat_a": CAT},
            {"x3": {"type": "ns", "df": 4}, "cat_b": CAT},
            {"x2": {"type": "bs", "k": 6}, "cat_a": CAT},
            {"x2": {"type": "bs", "df": 4}, "cat_a": CAT, "flag": CAT},
            {"x2": {"type": "bs", "df": 4}, "cat_a": CAT, "x1": LIN},
            {"x4": {"type": "s", "k": 6}, "cat_int": CAT},
            # A one-level categorical gives no columns, so today never places these knots.
            {"x3": {"type": "bs", "df": 4}, "one": CAT},
        ],
    ),
    "monotone_smooth_cat": _spec(
        {"x1": LIN, "cat_a": CAT, "cat_b": CAT},
        [
            {"x2": {"type": "bs", "k": 6, "monotonicity": "increasing"}, "cat_a": CAT},
            {"x3": {"type": "ms", "k": 5}, "cat_b": CAT},
            {"x4": {"type": "bs", "k": 5, "monotonicity": "decreasing"}, "cat_a": CAT, "x1": LIN},
        ],
    ),
    "frequency_encoding": _spec(
        {
            "brand": {"type": "frequency_encoding"},
            "cat_int": {"type": "frequency_encoding"},
            "bin": {"type": "frequency_encoding"},
            "f_special": {"type": "frequency_encoding"},
            "cat_dtype": {"type": "frequency_encoding"},
            "alias": {"type": "frequency_encoding", "variable": "str_null"},
            "cat_a": CAT,
        },
        [
            {"cat_a": CAT, "brand": CAT, "frequency_encoding": True},
            # "a:b"+"c" and "a"+"b:c" join to one label and share its count.
            {"colon": CAT, "colon2": CAT, "frequency_encoding": True},
            {"cat_a": CAT, "flag": CAT, "big_int_null": CAT, "frequency_encoding": True},
        ],
    ),
    "expressions_and_constraints": _spec(
        {
            "x1_sq": {"type": "expression", "expr": "x1 ** 2"},
            "x12": {"type": "expression", "expr": "x1 + x2"},
            "x23": {"type": "expression", "expr": "x2 / x3"},
            "x4m": {"type": "expression", "expr": "x4 - 1.5"},
            "x13": {"type": "expression", "expr": "x1 * x3"},
            "isq": {"type": "expression", "expr": "int_plain ** 2"},
            "x1_cube": {"type": "expression", "expr": "x1 ** 3", "monotonicity": "increasing"},
            "x3": {"type": "linear", "monotonicity": "decreasing"},
            "x4": {"type": "linear", "monotonicity": "increasing"},
            "x1": LIN,
        }
    ),
    "splines_learned": _spec(
        {
            "x2": {"type": "bs", "df": 5},
            "x3": {"type": "ns", "df": 4},
            "x4": {"type": "ms", "df": 5},
            "x1": {"type": "bs", "k": 8},
            "int_plain": {"type": "bs", "df": 4},
            "exposure": {"type": "ns", "k": 6},
            "weight": {"type": "bs", "df": 6, "degree": 2},
            "offset": {"type": "ms", "df": 3},
        }
    ),
    "splines_given_knots": _spec(
        {
            "x2": {"type": "bs", "knots": [30.0, 50.0, 65.0]},
            "x3": {"type": "ns", "knots": [0.8, 1.2], "boundary_knots": [0.01, 10.0]},
            "x1": {"type": "ms", "knots": [-1.0, 0.0, 1.0]},
            "x4": {"type": "bs", "knots": [-0.5, 0.5], "boundary_knots": [-5.0, 5.0]},
            "exposure": {"type": "ns", "knots": [0.4, 0.6]},
        }
    ),
    "splines_given_boundary": _spec(
        {
            "x2": {"type": "bs", "df": 5, "boundary_knots": [10.0, 90.0]},
            "x3": {"type": "ns", "df": 4, "boundary_knots": [0.0, 20.0]},
            "x1": {"type": "ms", "df": 4, "boundary_knots": [-6.0, 6.0]},
        }
    ),
    "no_intercept": _spec({"cat_a": CAT, "x1": LIN}, intercept=False),
}


# ── Pinned digests (T1) ──────────────────────────────────────────────────────
#
# A design's digest can only be pinned when every platform builds the same bits. The
# inputs come from generators that call libm (``np.exp``, the normal and gamma
# samplers), whose last bits differ by CPU and compiler, so each T1 input frame is
# first rounded onto a dyadic grid, exactly, and the result is pinned too. Building a
# design is then IEEE arithmetic, identical everywhere, except where it evaluates a
# transcendental function itself (``**`` in an expression term goes through ``pow``):
# those designs are compared with stored arrays within a few ulps instead.

#: The digest record's format.
DIGEST_SCHEMA = 2
#: T1's float inputs are multiples of 2⁻²⁰: x·2²⁰, rounded half to even, then ÷2²⁰.
#: Each step is exact, so the result depends on nothing but x's grid cell.
PORTABLE_GRID = 2.0**20
#: The relative tolerance for the designs that evaluate a transcendental function.
TRANSCENDENTAL_RTOL = 4 * np.finfo(np.float64).eps


def portable(frame: pl.DataFrame) -> pl.DataFrame:
    """``frame`` with every float column rounded onto the dyadic grid (nulls, NaN,
    infinities and the sign of zero kept), so its bits are the same on every platform."""
    columns = []
    for name, dtype in frame.schema.items():
        column = frame[name]
        if dtype.is_float():
            values = column.cast(pl.Float64).to_numpy()
            rounded = np.round(values * PORTABLE_GRID) / PORTABLE_GRID
            column = (
                pl.Series(name, rounded, dtype=pl.Float64)
                .scatter(np.flatnonzero(column.is_null().to_numpy()), None)
                .cast(dtype)
            )
        columns.append(column)
    return pl.DataFrame(columns)


def frame_digest(frame: pl.DataFrame) -> str:
    """A digest of a frame's names, dtype classes and values (floats by their exact
    shortest repr), independent of the Polars version's memory layout."""
    hasher = hashlib.sha256()
    for name in frame.columns:
        column = frame[name]
        hasher.update(f"{name}\0{type(column.dtype).__name__}\0".encode())
        hasher.update(repr(column.to_list()).encode("utf-8"))
    return hasher.hexdigest()


def evaluates_transcendentals(spec: dict[str, Any]) -> bool:
    """Whether building ``spec``'s design calls a libm function: an expression term
    with a power (Polars evaluates ``**`` with ``pow``)."""
    terms = list(spec["terms"].values())
    for interaction in spec.get("interactions") or []:
        terms += [term for term in interaction.values() if isinstance(term, dict)]
    return any(
        term.get("type") == "expression" and "**" in str(term.get("expr", "")) for term in terms
    )


def digest(array: np.ndarray) -> dict[str, Any]:
    """The digest of an array's float64 bytes in C order, with its shape."""
    values = np.ascontiguousarray(array, dtype=np.float64)
    return {"shape": list(values.shape), "sha256": hashlib.sha256(values.tobytes()).hexdigest()}


def names_digest(names: list[str]) -> str:
    return hashlib.sha256(json.dumps(list(names)).encode("utf-8")).hexdigest()


def array_key(name: str) -> str:
    """A design's key in the stored-array archive."""
    return name.replace("/", "__")


def design_record(
    data: pl.DataFrame, spec: dict[str, Any], name: str, y: np.ndarray, x: np.ndarray, names
) -> dict[str, Any]:
    """What T1 pins for one design: its input frame's digest, y's and the names'
    digests, and X's digest, or, for a design that evaluates a transcendental function,
    X's shape and its key in the stored-array archive."""
    if evaluates_transcendentals(spec):
        x_record: dict[str, Any] = {"shape": list(np.shape(x)), "array": array_key(name)}
    else:
        x_record = digest(x)
    return {
        "input": frame_digest(data),
        "X": x_record,
        "y": digest(y),
        "names": names_digest(names),
    }


def design_specs(*, raw: bool = False) -> dict[str, tuple[pl.DataFrame, dict[str, Any]]]:
    """Every pinned design: its portable input frame (with ``raw``, the generated frame
    before rounding) and its dict specification."""
    from rustystats.formula import dict_to_parsed_formula

    def prepare(frame: pl.DataFrame) -> pl.DataFrame:
        return frame if raw else portable(frame)

    frame = prepare(_frame())
    specs = {f"term_kind/{name}": (frame, spec) for name, spec in TERM_KINDS.items()}
    for case in CASES:
        kwargs, _fit, _holdout = case.build()
        parsed = dict_to_parsed_formula(
            response=kwargs["response"],
            terms=kwargs["terms"],
            interactions=kwargs.get("interactions"),
            intercept=kwargs.get("intercept", True),
        )
        if not parsed.target_encoding_terms:
            specs[f"golden/{case.name}"] = (prepare(kwargs["data"]), kwargs)
    return specs
