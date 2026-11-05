"""``--check`` of the oracle fixture generators compares regenerated fixtures
numerically (``scripts/oracle_fixture_compare.py``): floats within a tight relative
tolerance, everything else exactly."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "oracle_fixture_compare.py"
_spec = importlib.util.spec_from_file_location("oracle_fixture_compare", SCRIPT)
assert _spec is not None and _spec.loader is not None
compare = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compare)

FIXTURE = {
    "case_id": "poisson_log",
    "n": 96,
    "converged": True,
    "params": [0.125, -1.5, 3.0e-7],
    "deviance": 81.25,
    "missing": None,
    "nested": {"bse": [0.5, math.inf], "nan": math.nan},
}


def _with(path: list, value) -> dict:
    import copy

    fixture = copy.deepcopy(FIXTURE)
    target = fixture
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return fixture


def test_last_digit_float_differences_pass() -> None:
    assert compare.fixture_differences(FIXTURE, FIXTURE) == []
    nudged = _with(["params", 0], 0.125 * (1 + 1e-13))
    nudged["deviance"] = 81.25 * (1 - 4e-11)
    assert compare.fixture_differences(FIXTURE, nudged) == []


def test_real_differences_fail() -> None:
    for path, value in (
        (["params", 0], 0.125 * (1 + 1e-9)),  # beyond the tolerance
        (["params", 2], 0.0),  # zero against a nonzero value
        (["nested", "bse", 1], 1e308),  # infinity against a finite value
        (["nested", "nan"], 0.0),  # NaN against a number
        (["n"], 97),  # integers exactly
        (["converged"], False),
        (["case_id"], "poisson"),
        (["missing"], 0.0),
    ):
        assert compare.fixture_differences(FIXTURE, _with(path, value)), path
    shorter = _with(["params"], [0.125, -1.5])
    assert compare.fixture_differences(FIXTURE, shorter) == ["$.params: length 2 != 3"]
    extra = dict(FIXTURE, extra=1)
    assert "keys differ" in compare.fixture_differences(FIXTURE, extra)[0]


def test_convergence_residuals_compare_absolutely() -> None:
    """A KKT residual at rounding level is noise across platforms (glum's gaussian
    fixtures regenerate 1.83e-14 against 9.33e-15), but a residual that has grown
    to a real size still fails."""
    expected = {"expected": {"kkt_max_abs": 9.325873406851315e-15, "deviance": 1.5}}
    noisy = {"expected": {"kkt_max_abs": 1.831867990631508e-14, "deviance": 1.5}}
    grown = {"expected": {"kkt_max_abs": 1e-6, "deviance": 1.5}}
    assert compare.fixture_differences(expected, noisy) == []
    assert compare.fixture_differences(expected, grown) == [
        "$.expected.kkt_max_abs: 1e-06 != 9.325873406851315e-15 (absolute tolerance 1e-09)"
    ]
    # Other near-zero floats keep the relative tolerance.
    assert compare.fixture_differences({"x": 9.3e-15}, {"x": 1.8e-14})
