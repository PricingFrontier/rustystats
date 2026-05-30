"""RS-ACT-012: internal standardization for regularized GLMs."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import rustystats as rs
from rustystats._rustystats import fit_glm_py
from rustystats.regularization_path import (
    MIN_WEIGHTED_STD,
    compute_alpha_max,
    compute_standardization,
)

# Two-sided 95% normal quantile, matching the core's CI construction.
Z95 = 1.959963984540054


def _standardization_matrix(center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Linear map ``A`` with ``beta = A @ beta_tilde`` (intercept at index 0)."""
    p = len(scale)
    a = np.zeros((p, p))
    a[0, 0] = 1.0
    for j in range(1, p):
        a[0, j] = -center[j] / scale[j]
        a[j, j] = 1.0 / scale[j]
    return a


def _gaussian_frame(scale_x1: float = 1.0, *, intercept: bool = True) -> pl.DataFrame:
    rng = np.random.default_rng(123)
    n = 180
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    base = 1.0 if intercept else 0.0
    y = base + 1.5 * x1 - 0.8 * x2 + rng.normal(scale=0.2, size=n)
    return pl.DataFrame(
        {
            "y": y,
            "x1": x1 * scale_x1,
            "x2": x2,
        }
    )


def _fit_regularized(data: pl.DataFrame, *, standardize: bool = True, intercept: bool = True):
    return rs.glm_dict(
        response="y",
        terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
        data=data,
        family="gaussian",
        intercept=intercept,
    ).fit(
        alpha=80.0,
        l1_ratio=0.5,
        max_iter=100,
        tol=1e-10,
        standardize=standardize,
    )


def test_standardized_regularization_is_affine_invariant():
    base = _gaussian_frame()
    scaled = _gaussian_frame(scale_x1=1000.0)

    fit_base = _fit_regularized(base)
    fit_scaled = _fit_regularized(scaled)

    np.testing.assert_allclose(fit_base.predict(base), fit_scaled.predict(scaled), atol=1e-8)
    assert fit_scaled.params[1] * 1000.0 == pytest.approx(fit_base.params[1], rel=1e-8)


def test_standardize_false_preserves_raw_scale_penalty_behavior():
    base = _gaussian_frame()
    scaled = _gaussian_frame(scale_x1=1000.0)

    fit_base = _fit_regularized(base, standardize=False)
    fit_scaled = _fit_regularized(scaled, standardize=False)

    max_delta = float(np.max(np.abs(fit_base.predict(base) - fit_scaled.predict(scaled))))
    assert max_delta > 1e-2


def test_standardize_is_noop_without_penalty():
    data = _gaussian_frame()
    fit_std = rs.glm_dict(
        response="y",
        terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
        data=data,
        family="gaussian",
    ).fit(alpha=0.0, standardize=True, max_iter=100, tol=1e-10)
    fit_raw = rs.glm_dict(
        response="y",
        terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
        data=data,
        family="gaussian",
    ).fit(alpha=0.0, standardize=False, max_iter=100, tol=1e-10)

    np.testing.assert_allclose(fit_std.params, fit_raw.params, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(fit_std.predict(data), fit_raw.predict(data), atol=0.0, rtol=0.0)


def test_no_intercept_standardization_uses_scale_only_invariance():
    base = _gaussian_frame(intercept=False)
    scaled = _gaussian_frame(scale_x1=1000.0, intercept=False)

    fit_base = _fit_regularized(base, intercept=False)
    fit_scaled = _fit_regularized(scaled, intercept=False)

    np.testing.assert_allclose(fit_base.predict(base), fit_scaled.predict(scaled), atol=1e-8)
    assert fit_scaled.params[0] * 1000.0 == pytest.approx(fit_base.params[0], rel=1e-8)


def test_alpha_max_uses_standardized_scores_when_scales_are_supplied():
    data = _gaussian_frame()
    X = np.column_stack(
        [
            np.ones(data.height),
            data["x1"].to_numpy(),
            data["x2"].to_numpy(),
        ]
    )
    y = data["y"].to_numpy()
    X_scaled = X.copy()
    X_scaled[:, 1] *= 1000.0

    pen_mask = np.array([False, True, True])
    center, scale = compute_standardization(X, None, pen_mask)
    center_scaled, scale_scaled = compute_standardization(X_scaled, None, pen_mask)

    a = compute_alpha_max(
        X,
        y,
        l1_ratio=0.5,
        family="gaussian",
        link="identity",
        center=center,
        scale=scale,
        pen_mask=pen_mask,
    )
    a_scaled = compute_alpha_max(
        X_scaled,
        y,
        l1_ratio=0.5,
        family="gaussian",
        link="identity",
        center=center_scaled,
        scale=scale_scaled,
        pen_mask=pen_mask,
    )

    assert a_scaled == pytest.approx(a, rel=1e-10)


@pytest.mark.parametrize("l1_ratio", [0.0, 0.5, 1.0])
def test_se_ci_and_robust_back_transform_match_external_standardization(l1_ratio):
    """Internal standardization must reproduce an external-standardization fit
    plus the documented back-transform — for coefficients, model covariance,
    SEs, confidence intervals and the robust sandwich (RS-ACT-012 §8.5). This
    pins the intercept cross-term covariance map and robust-SE consistency,
    which the affine-invariance test alone does not exercise.
    """
    rng = np.random.default_rng(7)
    n = 300
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = rng.poisson(np.exp(0.4 + 0.9 * x1 - 0.6 * x2)).astype(float)
    w = rng.uniform(0.5, 2.0, size=n)
    off = rng.normal(scale=0.1, size=n)
    # Intercept + a heavy-scale column (×1000) + an O(1) column.
    X = np.column_stack([np.ones(n), x1 * 1000.0, x2])
    pen_mask = np.array([False, True, True])
    center, scale = compute_standardization(X, w, pen_mask, fit_intercept=True)

    def _fit(design, c, s):
        # store_design_matrix=True so the robust sandwich is available.
        return fit_glm_py(
            y,
            design,
            "poisson",
            "log",
            1.5,
            1.0,
            off,
            w,
            5.0,
            l1_ratio,
            300,
            1e-12,
            None,
            None,
            True,
            False,
            True,
            c,
            s,
        )

    internal = _fit(X, center, scale)  # standardize internally, report original scale
    external = _fit((X - center) / scale, None, None)  # pre-standardized, no transform

    a = _standardization_matrix(center, scale)
    beta_back = a @ np.asarray(external.params)
    cov_back = a @ np.asarray(external.cov_params_unscaled) @ a.T
    se_back = np.sqrt(np.diag(cov_back) * internal.scale())
    cov_robust_back = a @ np.asarray(external.cov_robust()) @ a.T

    np.testing.assert_allclose(internal.params, beta_back, atol=1e-8)
    np.testing.assert_allclose(internal.cov_params_unscaled, cov_back, atol=1e-8)
    np.testing.assert_allclose(internal.bse(), se_back, atol=1e-8)
    np.testing.assert_allclose(internal.cov_robust(), cov_robust_back, atol=1e-8)
    np.testing.assert_allclose(internal.bse_robust(), np.sqrt(np.diag(cov_robust_back)), atol=1e-8)

    ci = np.asarray(internal.conf_int())
    np.testing.assert_allclose(ci[:, 0], beta_back - Z95 * se_back, atol=1e-8)
    np.testing.assert_allclose(ci[:, 1], beta_back + Z95 * se_back, atol=1e-8)

    # The linear predictor is invariant to the reparameterization.
    np.testing.assert_allclose(internal.fittedvalues, external.fittedvalues, atol=1e-10)


def test_monotonicity_sign_preserved_after_back_transform():
    """nonneg/nonpos sign constraints survive the standardization back-transform
    (dividing by a positive scale preserves sign) — RS-ACT-012 §8.7.
    """
    data = _gaussian_frame(scale_x1=1000.0)
    model = rs.glm_dict(
        response="y",
        terms={
            "x1": {"type": "linear", "monotonicity": "increasing"},
            "x2": {"type": "linear", "monotonicity": "decreasing"},
        },
        data=data,
        family="gaussian",
    ).fit(alpha=40.0, l1_ratio=0.4, standardize=True, max_iter=200, tol=1e-10)

    names = list(model.feature_names)
    params = np.asarray(model.params)
    pos = params[next(i for i, nm in enumerate(names) if nm.startswith("pos("))]
    neg = params[next(i for i, nm in enumerate(names) if nm.startswith("neg("))]
    assert pos >= -1e-10
    assert neg <= 1e-10
    # Not a degenerate intercept-only fit.
    assert float(np.ptp(model.predict(data))) > 0.1


def _smooth_model(data: pl.DataFrame, *, standardize: bool):
    return rs.glm_dict(
        response="y",
        terms={"x1": {"type": "bs"}, "x2": {"type": "linear"}},
        data=data,
        family="gaussian",
        seed=0,
    ).fit(standardize=standardize)


def test_smooth_fit_unaffected_by_standardize_toggle():
    """standardize is a structural no-op for smooth (P-spline) fits: they take
    the GCV path before the parametric standardization branch — RS-ACT-012 §7/§8.8.
    """
    data = _gaussian_frame()
    fit_std = _smooth_model(data, standardize=True)
    fit_raw = _smooth_model(data, standardize=False)
    np.testing.assert_array_equal(np.asarray(fit_std.params), np.asarray(fit_raw.params))


def test_standardized_fit_serialization_and_export_parity():
    """Round-trip and PMML/ONNX export use original-scale coefficients (§8.6)."""
    data = _gaussian_frame(scale_x1=1000.0)
    model = rs.glm_dict(
        response="y",
        terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
        data=data,
        family="gaussian",
    ).fit(alpha=50.0, l1_ratio=0.0, standardize=True, max_iter=200, tol=1e-10)

    reloaded = rs.GLMModel.from_bytes(model.to_bytes())
    np.testing.assert_array_equal(np.asarray(model.params), np.asarray(reloaded.params))
    np.testing.assert_allclose(model.predict(data), reloaded.predict(data), atol=1e-10)

    assert len(model.to_pmml()) > 0
    assert len(model.to_onnx(mode="full")) > 0
    assert len(model.to_onnx(mode="scoring")) > 0


def test_cv_standardized_path_is_deterministic_and_sane():
    """The standardized CV path (warm-started fold fits) is deterministic to
    bytes and selects a sane, non-astronomical alpha (RS-ACT-012 §8.10/§8.12).
    """
    data = _gaussian_frame(scale_x1=1000.0)

    def _cv():
        return rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            data=data,
            family="gaussian",
            seed=99,
        ).fit(regularization="lasso", cv=4, standardize=True)

    first, second = _cv(), _cv()
    assert first.to_bytes() == second.to_bytes()
    assert np.isfinite(first.alpha) and first.alpha >= 0.0
    assert first.alpha < 1e6


def test_compute_standardization_is_stable_for_high_magnitude_columns():
    """The two-pass weighted variance keeps an accurate scale for very
    high-magnitude columns. The one-pass ``E[x²] - E[x]²`` lost ~1.6% here and
    could clamp a genuinely varying column to "constant", letting it escape the
    penalty — the exact RS-ACT-012 defect, reintroduced numerically.
    """
    col = 1e10 + np.arange(2000, dtype=float)
    X = np.column_stack([np.ones(2000), col])
    _center, scale = compute_standardization(X, None, np.array([False, True]))
    # Matches numpy's stable (two-pass) std and is not mis-flagged as constant.
    np.testing.assert_allclose(scale[1], np.std(col), rtol=1e-9)
    assert scale[1] > MIN_WEIGHTED_STD
