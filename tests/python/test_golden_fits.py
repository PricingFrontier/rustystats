"""The current fit matches the v0.9.0 golden record (``specs/out-of-core-fit.md`` §14 T2).

``scripts/capture_golden_fits.py`` recorded every case in ``_golden_cases.CASES`` from
v0.9.0 before the design build, solver loops and row reductions were replaced. The
rework may change results at rounding level (fixed reduction order, Gram-form kernels),
so floating-point fields are compared with per-field relative tolerances; discrete fields
(statuses, names, availability) are compared exactly, and the IRLS iteration count may
move by one when a rounding-level difference crosses the convergence threshold.

A case that an applied owner decision changes on purpose (``changes_under``) was
re-captured from the build that applies it (the record's ``recaptured`` table says when
and why) and is compared with that record the same way, except that its iteration counts
must match exactly: the re-captured values come from the current arithmetic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from _golden_cases import CASES, GOLDEN_SCHEMA, GoldenCase, decode, encode, record

GOLDEN = Path(__file__).resolve().parents[1] / "golden" / "fits_v0_9_0.json"

#: Relative tolerance per field class.
_RTOL: dict[str, float] = {
    "params": 1e-7,
    "statistic": 1e-9,
    "smooth": 1e-6,
    "path": 1e-7,
    "theta": 1e-7,
}
_FIELD_CLASS: dict[str, str] = {
    "params": "params",
    "bse": "params",
    "tvalues": "params",
    "pvalues": "params",
    "conf_int": "params",
    "predict_holdout": "params",
    "deviance": "statistic",
    "null_deviance": "statistic",
    "scale": "statistic",
    "pearson_chi2": "statistic",
    "llf": "statistic",
    "aic": "statistic",
    "bic": "statistic",
    "response_scale": "statistic",
    "alpha": "path",
    "l1_ratio": "path",
    "cv_deviance": "path",
    "cv_deviance_se": "path",
    "regularization_path": "path",
    "total_edf": "smooth",
    "df_model": "smooth",
    "df_resid": "smooth",
    "gcv": "smooth",
    "smooth_terms": "smooth",
    "theta": "theta",
}


def _document() -> dict[str, Any]:
    document = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert document["schema"] == GOLDEN_SCHEMA
    return document


def _golden() -> dict[str, Any]:
    return _document()["cases"]


def _rtol(case: GoldenCase, field: str) -> float:
    field_class = "params" if field.startswith("bse_robust_") else _FIELD_CLASS.get(field)
    if field_class is None:
        return 0.0
    return case.rtol_overrides.get(field_class, _RTOL[field_class])


def _assert_close(
    actual: Any, expected: Any, rtol: float, where: str, *, discrete_only: bool = False
) -> None:
    if isinstance(expected, dict) and set(expected) == {"unavailable"}:
        assert actual == expected, f"{where}: availability changed"
        return
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and set(actual) == set(expected), f"{where}: keys"
        for key in expected:
            _assert_close(
                actual[key], expected[key], rtol, f"{where}.{key}", discrete_only=discrete_only
            )
        return
    if isinstance(expected, list) and not all(isinstance(v, float) for v in expected):
        assert isinstance(actual, list) and len(actual) == len(expected), f"{where}: length"
        for index, (a, e) in enumerate(zip(actual, expected)):
            _assert_close(a, e, rtol, f"{where}[{index}]", discrete_only=discrete_only)
        return
    if isinstance(expected, (float, np.ndarray, list)):
        expected_array = np.asarray(expected, dtype=np.float64)
        actual_array = np.asarray(actual, dtype=np.float64)
        assert actual_array.shape == expected_array.shape, f"{where}: shape"
        finite = np.isfinite(expected_array)
        assert np.array_equal(np.isfinite(actual_array), finite), f"{where}: finiteness"
        if discrete_only:
            return
        if rtol == 0.0:
            assert np.array_equal(actual_array[finite], expected_array[finite]), where
            return
        scale = float(np.max(np.abs(expected_array[finite]), initial=0.0))
        np.testing.assert_allclose(
            actual_array[finite],
            expected_array[finite],
            rtol=rtol,
            atol=rtol * scale,
            err_msg=where,
        )
        return
    assert actual == expected, f"{where}: {actual!r} != {expected!r}"


def _normalise(value: Any) -> Any:
    """Round-trip a freshly recorded value through the golden encoding."""

    return decode(encode(value))


#: Owner decisions (spec §17) this build applies. A case recorded as changing under one
#: is compared with its re-capture from a build that applies it, never left unchecked.
APPLIED_DECISIONS = frozenset({"DEC-1", "DEC-2"})


def _recaptured_for_decision(case: GoldenCase) -> bool:
    return bool(set(case.changes_under) & APPLIED_DECISIONS)


def test_decided_cases_are_recaptured() -> None:
    """Every case an applied decision changes has a re-captured numerical record."""
    recaptured = _document().get("recaptured", {})
    for case in CASES:
        if _recaptured_for_decision(case):
            assert case.name in recaptured, f"{case.name} has no re-captured record"
            note = recaptured[case.name]
            assert all(decision in note for decision in case.changes_under), (
                f"{case.name}: the re-capture does not name {case.changes_under}"
            )


def test_comparison_rejects_nonfinite_and_changed_values() -> None:
    """The oracle refuses NaN or Inf in place of a recorded finite value, a value moved
    beyond its tolerance, and a changed iteration count on a re-captured case."""
    case = next(case for case in CASES if _recaptured_for_decision(case))
    expected = {name: decode(encoded) for name, encoded in _golden()[case.name].items()}
    params = np.asarray(expected["params"], dtype=np.float64)
    assert np.isfinite(params).all() and np.any(params != 0.0)
    rtol = _rtol(case, "params")
    _assert_close(params.copy(), expected["params"], rtol, "params")
    for bad in (np.full_like(params, np.nan), np.full_like(params, np.inf), params * 1.001):
        with pytest.raises(AssertionError):
            _assert_close(bad, expected["params"], rtol, "params")
    iterations = expected["iterations"]
    assert isinstance(iterations, int)
    with pytest.raises(AssertionError):
        _assert_close(iterations + 1, iterations, _rtol(case, "iterations"), "iterations")


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_fit_matches_golden_record(case: GoldenCase) -> None:
    expected_fields = _golden()[case.name]
    actual_fields = {name: _normalise(value) for name, value in record(case).items()}
    assert set(actual_fields) == set(expected_fields), "recorded field set changed"
    recaptured = _recaptured_for_decision(case)
    for name, encoded in expected_fields.items():
        expected = decode(encoded)
        actual = actual_fields[name]
        if name == "iterations" and isinstance(expected, int) and not recaptured:
            assert isinstance(actual, int) and abs(actual - expected) <= 1, (
                f"iterations: {actual} vs {expected}"
            )
            continue
        _assert_close(
            actual,
            expected,
            _rtol(case, name),
            f"{case.name}.{name}",
            discrete_only=case.discrete_only,
        )
