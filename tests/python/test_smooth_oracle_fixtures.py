"""Exact fixed-penalty smooth oracle fixture tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from rustystats._rustystats import fit_smooth_glm_unified_py
from rustystats.smooth import penalty_matrix

ROOT = Path(__file__).resolve().parents[2]
ORACLE_DIR = ROOT / "tests" / "oracles" / "exact"


def _fixture_paths() -> list[Path]:
    return sorted(ORACLE_DIR.glob("*.json"))


@pytest.mark.assurance
@pytest.mark.parametrize("path", _fixture_paths(), ids=lambda p: p.stem)
def test_exact_fixed_penalty_smooth_oracle_fixture(path: Path) -> None:
    fixture = json.loads(path.read_text(encoding="utf-8"))
    model = fixture["model"]
    columns = fixture["data"]["columns"]
    y = np.asarray(columns["y"], dtype=np.float64)
    basis = np.asarray(columns["basis"], dtype=np.float64)
    x_full = np.column_stack([np.ones(len(y)), basis])
    expected = fixture["expected"]
    tol = fixture["tolerances"]
    smooth_penalty = penalty_matrix(basis.shape[1], order=model["penalty_order"])

    np.testing.assert_allclose(
        smooth_penalty,
        np.asarray(expected["penalty_matrix"], dtype=np.float64),
        atol=0.0,
        rtol=0.0,
    )
    result, _meta = fit_smooth_glm_unified_py(
        y,
        x_full,
        [tuple(model["smooth_col_ranges"][0])],
        [smooth_penalty],
        model["family"],
        model["link"],
        None,
        None,
        200,
        1e-12,
        model["lambda"],
        model["lambda"],
        None,
        False,
    )

    np.testing.assert_allclose(
        result.params,
        np.asarray(expected["params"], dtype=np.float64),
        atol=tol["params_atol"],
        rtol=tol["params_rtol"],
    )
    np.testing.assert_allclose(
        x_full @ result.params,
        np.asarray(expected["fittedvalues"], dtype=np.float64),
        atol=tol["prediction_atol"],
        rtol=tol["prediction_rtol"],
    )
    full_penalty = np.zeros((x_full.shape[1], x_full.shape[1]), dtype=np.float64)
    full_penalty[1:, 1:] = model["lambda"] * smooth_penalty
    gradient = x_full.T @ (x_full @ result.params - y) + full_penalty @ result.params
    assert float(np.max(np.abs(gradient))) <= tol["kkt_atol"]
    assert expected["kkt_max_abs"] <= tol["kkt_atol"]
    edf = np.trace(x_full @ np.linalg.solve(x_full.T @ x_full + full_penalty, x_full.T))
    assert edf == pytest.approx(expected["edf"], abs=tol["edf_atol"])
