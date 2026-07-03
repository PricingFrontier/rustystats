"""Archived statsmodels MNLogit oracle fixture tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import rustystats as rs

ROOT = Path(__file__).resolve().parents[2]
ORACLE_DIR = ROOT / "tests" / "oracles" / "statsmodels_multinomial"


def _fixture_paths() -> list[Path]:
    return sorted(ORACLE_DIR.glob("*.json"))


@pytest.mark.assurance
@pytest.mark.parametrize("path", _fixture_paths(), ids=lambda p: p.stem)
def test_archived_statsmodels_multinomial_oracle_fixture(path: Path) -> None:
    fixture = json.loads(path.read_text(encoding="utf-8"))
    model = fixture["model"]
    data = pl.DataFrame(fixture["data"]["columns"])
    result = rs.multinomial_dict(
        response=model["response"],
        terms=model["terms"],
        data=data,
        classes=model["classes"],
        reference=model["reference"],
    ).fit(**model["fit_kwargs"])

    expected = fixture["expected"]
    tol = fixture["tolerances"]
    assert result.reference_ == model["reference"]
    assert result.reference_ != result.classes_[0]
    probabilities = result.predict_proba(data)
    np.testing.assert_allclose(
        probabilities,
        np.asarray(expected["probabilities"], dtype=np.float64),
        atol=tol["probability_atol"],
        rtol=tol["probability_rtol"],
    )
    np.testing.assert_allclose(
        probabilities.sum(axis=1),
        np.ones(probabilities.shape[0]),
        atol=tol["row_sum_atol"],
        rtol=0.0,
    )
