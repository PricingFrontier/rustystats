"""Archived external-oracle fixture tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import rustystats as rs

ROOT = Path(__file__).resolve().parents[2]
ORACLE_DIR = ROOT / "tests" / "oracles" / "statsmodels"


def _fixture_paths() -> list[Path]:
    return sorted(ORACLE_DIR.glob("*.json"))


def _fit_fixture(fixture: dict):
    model = fixture["model"]
    data = pl.DataFrame(fixture["data"]["columns"])
    kwargs = {
        "response": model["response"],
        "terms": model["terms"],
        "data": data,
        "family": model["family"],
    }
    if model.get("exposure") is not None:
        kwargs["exposure"] = model["exposure"]
    if model.get("var_power") is not None:
        kwargs["var_power"] = model["var_power"]
    if model.get("theta") is not None:
        kwargs["theta"] = model["theta"]
    fit_kwargs = dict(model.get("fit_kwargs") or {})
    return rs.glm_dict(**kwargs).fit(**fit_kwargs), data


@pytest.mark.assurance
@pytest.mark.parametrize("path", _fixture_paths(), ids=lambda p: p.stem)
def test_archived_statsmodels_oracle_fixture(path):
    fixture = json.loads(path.read_text(encoding="utf-8"))
    result, data = _fit_fixture(fixture)
    expected = fixture["expected"]
    tol = fixture["tolerances"]

    np.testing.assert_allclose(
        result.params,
        np.asarray(expected["params"], dtype=np.float64),
        atol=tol["params_atol"],
        rtol=tol["params_rtol"],
    )
    np.testing.assert_allclose(
        result.deviance,
        float(expected["deviance"]),
        atol=tol["deviance_atol"],
        rtol=tol["deviance_rtol"],
    )
    np.testing.assert_allclose(
        result.predict(data),
        np.asarray(expected["fittedvalues"], dtype=np.float64),
        atol=tol["prediction_atol"],
        rtol=tol["prediction_rtol"],
    )
