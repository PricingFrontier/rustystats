"""Metamorphic high-assurance contracts for public GLM APIs."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import rustystats as rs


def _gaussian_frame(n: int = 80) -> pl.DataFrame:
    x1 = np.linspace(-2.0, 2.0, n)
    x2 = np.cos(np.linspace(0.0, 2.0, n))
    y = 1.2 + 0.35 * x1 - 0.2 * x2 + 0.03 * np.sin(np.arange(n))
    return pl.DataFrame({"y": y, "x1": x1, "x2": x2})


def _poisson_exposure_frame(n: int = 90) -> pl.DataFrame:
    x = np.linspace(-1.2, 1.2, n)
    exposure = np.linspace(0.3, 1.8, n)
    rate = np.exp(-0.5 + 0.25 * x)
    y = np.floor(exposure * rate + (np.arange(n) % 4 == 0)).astype(float)
    return pl.DataFrame({"y": y, "x": x, "exposure": exposure})


def _fit_gaussian(data: pl.DataFrame, *, weights: str | None = None):
    return rs.glm_dict(
        response="y",
        terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
        data=data,
        family="gaussian",
        weights=weights,
    ).fit(max_iter=100, tol=1e-11)


@pytest.mark.assurance
def test_row_permutation_preserves_unpenalized_fit_and_predictions():
    data = _gaussian_frame()
    permuted = data.with_row_index("_idx").sort(pl.col("_idx") * 37 % len(data)).drop("_idx")

    base = _fit_gaussian(data)
    shuffled = _fit_gaussian(permuted)

    np.testing.assert_allclose(shuffled.params, base.params, atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(shuffled.predict(data), base.predict(data), atol=1e-9, rtol=1e-9)


@pytest.mark.assurance
def test_duplicated_rows_match_integer_case_weights():
    data = _gaussian_frame(60)
    duplicated = pl.concat([data, data], how="vertical")
    weighted = data.with_columns(pl.lit(2.0).alias("w"))

    fit_duplicate = _fit_gaussian(duplicated)
    fit_weighted = _fit_gaussian(weighted, weights="w")

    np.testing.assert_allclose(fit_duplicate.params, fit_weighted.params, atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(
        fit_duplicate.predict(data), fit_weighted.predict(data), atol=1e-9, rtol=1e-9
    )


@pytest.mark.assurance
def test_global_weight_scaling_preserves_coefficients():
    data = _gaussian_frame(70)
    w = 0.4 + np.linspace(0.0, 2.0, len(data))
    fit_w = _fit_gaussian(data.with_columns(pl.Series("w", w)), weights="w")
    fit_scaled = _fit_gaussian(data.with_columns(pl.Series("w", 17.0 * w)), weights="w")

    np.testing.assert_allclose(fit_w.params, fit_scaled.params, atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(fit_w.predict(data), fit_scaled.predict(data), atol=1e-9, rtol=1e-9)


@pytest.mark.assurance
def test_exposure_scaling_with_offset_adjustment_preserves_rate_model():
    data = _poisson_exposure_frame()
    scaled = data.with_columns(
        (pl.col("exposure") * 10.0).alias("exposure"),
        pl.lit(-np.log(10.0)).alias("offset_adj"),
    )

    base = rs.glm_dict(
        response="y",
        terms={"x": {"type": "linear"}},
        data=data,
        family="poisson",
        exposure="exposure",
    ).fit(max_iter=100, tol=1e-10)
    adjusted = rs.glm_dict(
        response="y",
        terms={"x": {"type": "linear"}},
        data=scaled,
        family="poisson",
        exposure="exposure",
        offset="offset_adj",
    ).fit(max_iter=100, tol=1e-10)

    np.testing.assert_allclose(base.params, adjusted.params, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(base.predict(data), adjusted.predict(scaled), atol=1e-8, rtol=1e-8)


@pytest.mark.assurance
def test_serialization_roundtrip_preserves_predictions_and_metadata():
    data = _poisson_exposure_frame()
    result = rs.glm_dict(
        response="y",
        terms={"x": {"type": "linear"}},
        data=data,
        family="poisson",
        exposure="exposure",
    ).fit(max_iter=100, tol=1e-10)

    loaded = rs.GLMModel.from_bytes(result.to_bytes())

    assert loaded.family == result.family
    assert loaded.link == result.link
    assert loaded.feature_names == result.feature_names
    np.testing.assert_allclose(loaded.params, result.params, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(loaded.predict(data), result.predict(data), atol=1e-12, rtol=1e-12)


@pytest.mark.assurance
def test_chunked_prediction_matches_single_shot_prediction(monkeypatch):
    import rustystats.formula as formula

    data = _poisson_exposure_frame(75)
    result = rs.glm_dict(
        response="y",
        terms={"x": {"type": "linear"}},
        data=data,
        family="poisson",
        exposure="exposure",
    ).fit(max_iter=100, tol=1e-10)

    monkeypatch.setattr(formula, "_PREDICT_ROW_CHUNK_DEFAULT", 1_000_000)
    single = result.predict(data)
    monkeypatch.setattr(formula, "_PREDICT_ROW_CHUNK_DEFAULT", 7)
    chunked = result.predict(data)

    np.testing.assert_allclose(chunked, single, atol=1e-12, rtol=1e-12)
