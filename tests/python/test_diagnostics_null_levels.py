"""diagnostics() reports a null categorical value as a level of its own."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import rustystats as rs


@pytest.fixture(autouse=True)
def _in_tmp_cwd(tmp_path, monkeypatch):
    """diagnostics() writes analysis/diagnostics.json to the working directory."""
    monkeypatch.chdir(tmp_path)


def _null_frame(n: int = 2000) -> pl.DataFrame:
    rng = np.random.default_rng(3)
    return pl.DataFrame(
        {
            "x": rng.normal(size=n),
            "e": rng.uniform(0.2, 1.0, n),
            "y": rng.poisson(0.5, n).astype(float),
            "blanks": [None if i % 7 == 0 else "a" if i % 3 else "b" for i in range(n)],
        }
    )


def test_diagnostics_report_null_categorical_values_as_a_level():
    df = _null_frame()
    train, test = df[:1500], df[1500:]
    model = rs.glm_dict(
        response="y", terms={"x": {"type": "linear"}}, data=train, family="poisson", exposure="e"
    ).fit()
    diag = model.diagnostics(
        train_data=train,
        test_data=test,
        categorical_factors=["blanks"],
        interactions=[("x", "blanks")],
    )
    levels = {b.level: b for b in diag.train_test.train.factor_diagnostics["blanks"]}
    assert levels["None"].n == train["blanks"].null_count()
    tested = {b.level: b for b in diag.train_test.test.factor_diagnostics["blanks"]}
    assert tested["None"].n == test["blanks"].null_count()
    grid = next(i for i in diag.interactions if i.name == "x:blanks").train_surface_grid
    assert "None" in (grid.row_levels or grid.col_levels)
