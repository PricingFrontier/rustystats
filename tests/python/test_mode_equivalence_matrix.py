"""Mode equivalence over the batching matrix (``specs/out-of-core-fit.md`` §14 T3).

An out-of-core fit gives the in-memory fit bit for bit whatever the source's batch
size, the encoder's block size and the thread count: every §8 statistic, the iteration
count, the status, the solver trace, the robust standard errors and the predictions.
B is ``GRAM_BLOCK_ROWS``. The full matrix (source batches of 1,000, B-1, B, B+1 and n
rows; ``encode_rows`` B and 4B; 1 and 4 threads) runs for Gaussian and Poisson fits, a
spread of it for the other families, and single-row batches on a small source.
"""

from __future__ import annotations

import functools
from typing import Any

import numpy as np
import polars as pl
import pytest
import rustystats as rs
from _golden_cases import RESULT_FIELDS, _read, design_frame
from rustystats._rustystats import solver_constants_py
from rustystats.exceptions import OutOfCoreResultError

B = solver_constants_py()[1]
N = 4 * B + B // 2 + 3
N_SMALL = 48
ROBUST = ["HC0", "HC1", "HC2", "HC3"]

TERMS = {
    "x1": {"type": "linear"},
    "x2": {"type": "bs", "df": 4},
    "cat_a": {"type": "categorical"},
    "cat_b": {"type": "categorical"},
    "flag": {"type": "categorical"},
}
#: A smaller model for the small source, whose fit stays well determined.
SMALL_TERMS = {"x1": {"type": "linear"}, "cat_a": {"type": "categorical"}}
MODELS: dict[str, dict[str, Any]] = {
    "gaussian": {"response": "y_gauss", "weights": "weight", "offset": "offset"},
    "poisson": {"response": "y_pois", "exposure": "exposure", "weights": "weight"},
    "gamma": {"response": "y_gamma", "weights": "weight"},
    "tweedie": {"response": "y_tweedie", "exposure": "exposure", "var_power": 1.5},
    "binomial": {"response": "y_bin", "weights": "weight"},
    "negbinomial": {"response": "y_nb", "exposure": "exposure", "theta": "estimate"},
}

FULL = [
    (family, batch, encode, threads)
    for family in ("gaussian", "poisson")
    for batch in (1000, B - 1, B, B + 1, N)
    for encode in (B, 4 * B)
    for threads in (1, 4)
]
SPREAD = [
    (family, batch, encode, threads)
    for family in ("gamma", "tweedie", "binomial", "negbinomial")
    for batch, encode, threads in (
        (1000, B, 1),
        (B - 1, 4 * B, 4),
        (B, B, 4),
        (B + 1, 4 * B, 1),
        (N, B, 4),
    )
]
SINGLE_ROWS = [
    ("gaussian", 1, B, 1),
    ("gaussian", 1, B, 4),
    ("poisson", 1, 4 * B, 1),
    ("binomial", 1, B, 4),
]


@functools.cache
def _frame(n: int) -> pl.DataFrame:
    return design_frame(n, 17)


def _holdout() -> pl.DataFrame:
    return design_frame(64, 99)


def _bits(value: Any) -> Any:
    if isinstance(value, (float, np.floating)):
        return float(value).hex()
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "f":
            return [float(v).hex() for v in value.ravel()]
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_bits(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _bits(v) for k, v in value.items()}
    if isinstance(value, (bool, int, str)) or value is None:
        return value
    return repr(value)


def _fields(result: Any) -> dict[str, Any]:
    fields = {name: _bits(_read(result, name)) for name in RESULT_FIELDS}
    fields["solver_trace"] = _bits(_read(result, "solver_trace"))
    for hc_type in ROBUST:
        fields[f"bse_robust_{hc_type}"] = _bits(np.asarray(result.bse_robust(hc_type)))
    fields["predict"] = _bits(np.asarray(result.predict(_holdout())))
    return fields


def _glm(family: str, data: Any, n: int) -> Any:
    terms = SMALL_TERMS if n == N_SMALL else TERMS
    return rs.glm_dict(data=data, terms=terms, family=family, **MODELS[family])


@functools.cache
def _in_memory(family: str, n: int) -> dict[str, Any]:
    return _fields(_glm(family, _frame(n), n).fit(robust=ROBUST))


def _source(frame: pl.DataFrame, rows: int) -> rs.BatchSource:
    def open_batches(columns):
        selected = frame.select(columns)
        for start in range(0, selected.height, rows):
            yield selected.slice(start, rows)

    return rs.BatchSource(open_batches, frame.schema)


#: What §4.5 refuses on an out-of-core result (it needs per-row training values), by
#: name; none of it is among the compared quantities, which both modes must serve alike.
REFUSED_OUT_OF_CORE = {
    "fittedvalues": lambda result: result.fittedvalues,
    "linear_predictor": lambda result: result.linear_predictor,
    "resid_response": lambda result: result.resid_response(),
    "resid_pearson": lambda result: result.resid_pearson(),
    "resid_deviance": lambda result: result.resid_deviance(),
    "resid_working": lambda result: result.resid_working(),
    "fit_diagnostics": lambda result: result.fit_diagnostics(),
    "diagnostics": lambda result: result.diagnostics(),
}


def _compare(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    """Every quantity equal, its availability included: one that raises in one mode
    must raise the same exception type in the other."""
    assert actual.keys() == expected.keys()
    assert not REFUSED_OUT_OF_CORE.keys() & actual.keys()
    for name, value in expected.items():
        assert actual[name] == value, name


def _fit_out_of_core(family: str, n: int, batch: int, encode: int, threads: int, scratch):
    return _glm(family, _source(_frame(n), batch), n).fit(
        robust=ROBUST,
        out_of_core=rs.OutOfCoreOptions(encode_rows=encode, threads=threads, scratch_dir=scratch),
    )


def _check(family: str, n: int, batch: int, encode: int, threads: int, scratch) -> None:
    result = _fit_out_of_core(family, n, batch, encode, threads, scratch)
    assert result.memory_plan.encode_rows == encode
    _compare(_fields(result), _in_memory(family, n))
    for name, refused in REFUSED_OUT_OF_CORE.items():
        with pytest.raises(OutOfCoreResultError, match=name):
            refused(result)
    assert list(scratch.iterdir()) == [], "spill files are removed"


class _RaisingLlf:
    """An out-of-core result whose ``llf()`` is broken."""

    def __init__(self, result: Any) -> None:
        self._result = result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._result, name)

    def llf(self) -> float:
        raise RuntimeError("a broken llf")


def test_the_comparison_rejects_a_quantity_that_raises_in_one_mode(tmp_path) -> None:
    result = _fit_out_of_core("gaussian", N_SMALL, 16, B, 1, tmp_path)
    expected = _in_memory("gaussian", N_SMALL)
    _compare(_fields(result), expected)
    broken = _fields(_RaisingLlf(result))
    assert broken["llf"] == {"unavailable": "RuntimeError"}
    with pytest.raises(AssertionError, match="llf"):
        _compare(broken, expected)


def _id(case: tuple[str, int, int, int]) -> str:
    family, batch, encode, threads = case
    return f"{family}-batch{batch}-encode{encode}-threads{threads}"


@pytest.mark.slow
@pytest.mark.parametrize("case", FULL + SPREAD, ids=_id)
def test_out_of_core_equals_in_memory_across_the_batching_matrix(case, tmp_path) -> None:
    family, batch, encode, threads = case
    _check(family, N, batch, encode, threads, tmp_path)


@pytest.mark.slow
@pytest.mark.parametrize("case", SINGLE_ROWS, ids=_id)
def test_single_row_batches_equal_in_memory(case, tmp_path) -> None:
    family, batch, encode, threads = case
    _check(family, N_SMALL, batch, encode, threads, tmp_path)
