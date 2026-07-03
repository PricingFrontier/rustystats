"""Archived glum oracle fixture tests for regularized GLMs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import rustystats as rs

ROOT = Path(__file__).resolve().parents[2]
ORACLE_DIR = ROOT / "tests" / "oracles" / "glum"


def _fixture_paths() -> list[Path]:
    return sorted(ORACLE_DIR.glob("*.json"))


def _design_matrix(columns: dict[str, list[float]]) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(columns["y"], dtype=np.float64)
    x = np.column_stack(
        [
            np.ones(len(y)),
            np.asarray(columns["x1"], dtype=np.float64),
            np.asarray(columns["x2"], dtype=np.float64),
            np.asarray(columns["x3"], dtype=np.float64),
        ]
    )
    return y, x


def _kkt_residual(
    x: np.ndarray, y: np.ndarray, beta: np.ndarray, alpha: float, l1_ratio: float
) -> float:
    gradient = x.T @ (x @ beta - y)
    l1 = alpha * l1_ratio
    l2 = alpha * (1.0 - l1_ratio)
    residuals = [abs(float(gradient[0]))]
    for j in range(1, len(beta)):
        if abs(beta[j]) > 1e-10:
            residuals.append(abs(float(gradient[j] + l2 * beta[j] + l1 * np.sign(beta[j]))))
        else:
            residuals.append(max(0.0, abs(float(gradient[j])) - l1))
    return max(residuals)


@pytest.mark.assurance
@pytest.mark.parametrize("path", _fixture_paths(), ids=lambda p: p.stem)
def test_archived_glum_regularized_oracle_fixture(path: Path) -> None:
    fixture = json.loads(path.read_text(encoding="utf-8"))
    model = fixture["model"]
    columns = fixture["data"]["columns"]
    data = pl.DataFrame(columns)
    result = rs.glm_dict(
        response=model["response"],
        terms=model["terms"],
        data=data,
        family=model["family"],
    ).fit(alpha=model["alpha"], l1_ratio=model["l1_ratio"], **model["fit_kwargs"])

    expected = fixture["expected"]
    tol = fixture["tolerances"]
    np.testing.assert_allclose(
        result.params,
        np.asarray(expected["params"], dtype=np.float64),
        atol=tol["params_atol"],
        rtol=tol["params_rtol"],
    )
    np.testing.assert_allclose(
        result.predict(data),
        np.asarray(expected["fittedvalues"], dtype=np.float64),
        atol=tol["prediction_atol"],
        rtol=tol["prediction_rtol"],
    )
    y, x = _design_matrix(columns)
    assert _kkt_residual(x, y, result.params, model["alpha"], model["l1_ratio"]) <= tol["kkt_atol"]
    assert expected["kkt_max_abs"] <= tol["kkt_atol"]
