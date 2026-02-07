"""
Tests for smooth (penalized spline) GLM fitting.

Covers:
- Single smooth term with various families
- Multi-term smooth fitting
- Monotonic smooth terms
- Diagnostics (scale, pvalues, bse, conf_int, summary)
- EDF properties
- Serialization round-trip
- Smooth + parametric interactions
"""

import numpy as np
import polars as pl
import pytest
import rustystats as rs


def _make_poisson_data(n=500, seed=42):
    """Generate Poisson data with a smooth effect."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 10, n)
    # True effect: sin curve
    eta = 0.5 + 0.3 * np.sin(x)
    mu = np.exp(eta)
    y = rng.poisson(mu)
    return pl.DataFrame({"y": y, "x": x.round(4)})


def _make_gamma_data(n=500, seed=42):
    """Generate Gamma data with a smooth effect."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 10, n)
    eta = 1.0 + 0.5 * np.sin(x)
    mu = np.exp(eta)
    y = rng.gamma(shape=5.0, scale=mu / 5.0)
    return pl.DataFrame({"y": y.round(6), "x": x.round(4)})


def _make_gaussian_data(n=500, seed=42):
    """Generate Gaussian data with a smooth effect."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 10, n)
    y = 2.0 + np.sin(x) + rng.normal(0, 0.3, n)
    return pl.DataFrame({"y": y.round(6), "x": x.round(4)})


def _make_multi_smooth_data(n=500, seed=42):
    """Generate data with two smooth effects."""
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0, 10, n)
    x2 = rng.uniform(0, 5, n)
    eta = 0.5 + 0.3 * np.sin(x1) + 0.2 * np.cos(x2)
    mu = np.exp(eta)
    y = rng.poisson(mu)
    return pl.DataFrame({"y": y, "x1": x1.round(4), "x2": x2.round(4)})


class TestSingleSmoothPoisson:
    """Test single smooth term with Poisson family."""

    def test_basic_fit(self):
        data = _make_poisson_data()
        result = rs.glm("y ~ bs(x, k=10)", data, family="poisson").fit(max_iter=100)
        assert result.converged
        assert result.has_smooth_terms()
        assert len(result.smooth_terms) == 1

    def test_edf_less_than_k(self):
        data = _make_poisson_data()
        result = rs.glm("y ~ bs(x, k=10)", data, family="poisson").fit(max_iter=100)
        st = result.smooth_terms[0]
        # EDF should be less than k (penalty shrinks complexity)
        assert st.edf < 10.0
        # But should be > 1 (non-trivial smooth)
        assert st.edf > 1.0

    def test_scale_is_one(self):
        """Poisson has fixed dispersion = 1."""
        data = _make_poisson_data()
        result = rs.glm("y ~ bs(x, k=10)", data, family="poisson").fit(max_iter=100)
        assert result.scale() == 1.0

    def test_diagnostics_available(self):
        data = _make_poisson_data()
        result = rs.glm("y ~ bs(x, k=10)", data, family="poisson").fit(max_iter=100)
        # All standard diagnostics should work
        assert len(result.bse()) == len(result.params)
        assert len(result.tvalues()) == len(result.params)
        assert len(result.pvalues()) == len(result.params)
        ci = result.conf_int()
        assert ci.shape == (len(result.params), 2)

    def test_summary_works(self):
        data = _make_poisson_data()
        result = rs.glm("y ~ bs(x, k=10)", data, family="poisson").fit(max_iter=100)
        summary = result.summary()
        assert isinstance(summary, str)
        assert "poisson" in summary.lower() or "Poisson" in summary

    def test_lambda_and_gcv(self):
        data = _make_poisson_data()
        result = rs.glm("y ~ bs(x, k=10)", data, family="poisson").fit(max_iter=100)
        st = result.smooth_terms[0]
        assert st.lambda_ > 0
        assert result.gcv is not None
        assert result.gcv > 0

    def test_total_edf(self):
        data = _make_poisson_data()
        result = rs.glm("y ~ bs(x, k=10)", data, family="poisson").fit(max_iter=100)
        # total_edf = parametric (intercept=1) + smooth edf
        assert result.total_edf is not None
        assert result.total_edf > 1.0

    def test_df_resid_uses_edf(self):
        """df_resid should use n - total_edf for smooth models."""
        data = _make_poisson_data()
        result = rs.glm("y ~ bs(x, k=10)", data, family="poisson").fit(max_iter=100)
        expected = result.nobs - result.total_edf
        assert abs(result.df_resid - expected) < 1e-10


class TestSingleSmoothGamma:
    """Test single smooth term with Gamma family (estimated dispersion)."""

    def test_basic_fit(self):
        data = _make_gamma_data()
        result = rs.glm("y ~ bs(x, k=10)", data, family="gamma").fit(max_iter=100)
        assert result.converged
        assert result.has_smooth_terms()

    def test_scale_estimated(self):
        """Gamma should estimate dispersion, not hardcode 1.0."""
        data = _make_gamma_data()
        result = rs.glm("y ~ bs(x, k=10)", data, family="gamma").fit(max_iter=100)
        scale = result.scale()
        assert scale != 1.0
        # For Gamma with shape=5, dispersion ≈ 1/5 = 0.2
        assert 0.05 < scale < 1.0

    def test_diagnostics(self):
        data = _make_gamma_data()
        result = rs.glm("y ~ bs(x, k=10)", data, family="gamma").fit(max_iter=100)
        assert len(result.bse()) == len(result.params)
        assert len(result.pvalues()) == len(result.params)


class TestSingleSmoothGaussian:
    """Test single smooth term with Gaussian family."""

    def test_basic_fit(self):
        data = _make_gaussian_data()
        result = rs.glm("y ~ bs(x, k=10)", data, family="gaussian").fit(max_iter=100)
        assert result.converged
        assert result.has_smooth_terms()

    def test_scale_estimated(self):
        """Gaussian should estimate dispersion from Pearson residuals."""
        data = _make_gaussian_data()
        result = rs.glm("y ~ bs(x, k=10)", data, family="gaussian").fit(max_iter=100)
        scale = result.scale()
        # True noise variance is 0.3^2 = 0.09
        assert 0.01 < scale < 0.5


class TestMultiTermSmooth:
    """Test multiple smooth terms in one model."""

    def test_two_smooth_terms(self):
        data = _make_multi_smooth_data()
        result = rs.glm(
            "y ~ bs(x1, k=10) + bs(x2, k=8)", data, family="poisson"
        ).fit(max_iter=100)
        assert result.converged
        assert result.has_smooth_terms()
        assert len(result.smooth_terms) == 2

    def test_separate_lambdas(self):
        """Each smooth term should get its own lambda."""
        data = _make_multi_smooth_data()
        result = rs.glm(
            "y ~ bs(x1, k=10) + bs(x2, k=8)", data, family="poisson"
        ).fit(max_iter=100)
        lambdas = [st.lambda_ for st in result.smooth_terms]
        assert len(lambdas) == 2
        # Lambdas can differ
        assert all(l > 0 for l in lambdas)

    def test_separate_edfs(self):
        """Each smooth term should get its own EDF."""
        data = _make_multi_smooth_data()
        result = rs.glm(
            "y ~ bs(x1, k=10) + bs(x2, k=8)", data, family="poisson"
        ).fit(max_iter=100)
        edfs = [st.edf for st in result.smooth_terms]
        assert len(edfs) == 2
        assert all(e > 1.0 for e in edfs)
        # EDF should be less than k for each term
        assert edfs[0] < 10.0
        assert edfs[1] < 8.0

    def test_total_edf_is_sum(self):
        data = _make_multi_smooth_data()
        result = rs.glm(
            "y ~ bs(x1, k=10) + bs(x2, k=8)", data, family="poisson"
        ).fit(max_iter=100)
        smooth_sum = sum(st.edf for st in result.smooth_terms)
        # total_edf = intercept (1) + smooth EDFs
        assert abs(result.total_edf - (1.0 + smooth_sum)) < 0.1

    def test_diagnostics(self):
        data = _make_multi_smooth_data()
        result = rs.glm(
            "y ~ bs(x1, k=10) + bs(x2, k=8)", data, family="poisson"
        ).fit(max_iter=100)
        n_params = len(result.params)
        assert len(result.bse()) == n_params
        assert len(result.pvalues()) == n_params


class TestMonotonicSmooth:
    """Test monotonic smooth terms."""

    def test_increasing(self):
        rng = np.random.default_rng(42)
        n = 300
        x = rng.uniform(0, 10, n)
        # Monotonically increasing true effect
        eta = 0.5 + 0.3 * x
        y = rng.poisson(np.exp(eta))
        data = pl.DataFrame({"y": y, "x": x.round(4)})
        result = rs.glm(
            "y ~ bs(x, k=8, monotonicity='increasing')",
            data, family="poisson"
        ).fit(max_iter=100)
        assert result.converged

    def test_decreasing(self):
        rng = np.random.default_rng(42)
        n = 300
        x = rng.uniform(0, 10, n)
        eta = 2.0 - 0.3 * x
        y = rng.poisson(np.exp(eta))
        data = pl.DataFrame({"y": y, "x": x.round(4)})
        result = rs.glm(
            "y ~ bs(x, k=8, monotonicity='decreasing')",
            data, family="poisson"
        ).fit(max_iter=100)
        assert result.converged


class TestSmoothWithParametric:
    """Test smooth + parametric terms together."""

    def test_smooth_plus_continuous(self):
        rng = np.random.default_rng(42)
        n = 400
        x1 = rng.uniform(0, 10, n)
        x2 = rng.normal(0, 1, n)
        eta = 0.5 + 0.3 * np.sin(x1) + 0.5 * x2
        y = rng.poisson(np.exp(eta))
        data = pl.DataFrame({
            "y": y, "x1": x1.round(4), "x2": x2.round(4)
        })
        result = rs.glm(
            "y ~ bs(x1, k=10) + x2", data, family="poisson"
        ).fit(max_iter=100)
        assert result.converged
        assert result.has_smooth_terms()
        # Should have intercept + smooth columns + x2
        assert len(result.params) > 2

    def test_smooth_plus_categorical(self):
        rng = np.random.default_rng(42)
        n = 400
        x = rng.uniform(0, 10, n)
        cat = rng.choice(["A", "B", "C"], n)
        cat_effect = np.where(cat == "A", 0.0, np.where(cat == "B", 0.3, -0.2))
        eta = 0.5 + 0.3 * np.sin(x) + cat_effect
        y = rng.poisson(np.exp(eta))
        data = pl.DataFrame({
            "y": y, "x": x.round(4), "cat": cat
        })
        result = rs.glm(
            "y ~ bs(x, k=10) + C(cat)", data, family="poisson"
        ).fit(max_iter=100)
        assert result.converged
        assert result.has_smooth_terms()


class TestFixedVsSmooth:
    """Compare fixed spline (df=) vs penalized smooth (k=)."""

    def test_fixed_has_no_smooth_terms(self):
        data = _make_poisson_data()
        result = rs.glm("y ~ bs(x, df=5)", data, family="poisson").fit()
        assert not result.has_smooth_terms()

    def test_smooth_has_smooth_terms(self):
        data = _make_poisson_data()
        result = rs.glm("y ~ bs(x, k=10)", data, family="poisson").fit(max_iter=100)
        assert result.has_smooth_terms()

    def test_both_converge(self):
        data = _make_poisson_data()
        r_fixed = rs.glm("y ~ bs(x, df=5)", data, family="poisson").fit()
        r_smooth = rs.glm("y ~ bs(x, k=10)", data, family="poisson").fit(max_iter=100)
        assert r_fixed.converged
        assert r_smooth.converged


class TestSmoothSerialization:
    """Test to_bytes/from_bytes round-trip for smooth models."""

    def test_round_trip_predict(self):
        data = _make_poisson_data(n=300)
        result = rs.glm("y ~ bs(x, k=10)", data, family="poisson").fit(max_iter=100)

        # Serialize and deserialize
        model_bytes = result.to_bytes()
        loaded = rs.GLMModel.from_bytes(model_bytes)

        # Predict on same data
        pred_orig = result.predict(data)
        pred_loaded = loaded.predict(data)
        np.testing.assert_allclose(pred_orig, pred_loaded, rtol=1e-10)

    def test_smooth_metadata_preserved(self):
        data = _make_poisson_data(n=300)
        result = rs.glm("y ~ bs(x, k=10)", data, family="poisson").fit(max_iter=100)

        model_bytes = result.to_bytes()
        loaded = rs.GLMModel.from_bytes(model_bytes)

        assert loaded.has_smooth_terms() == result.has_smooth_terms()
        if result.has_smooth_terms():
            assert len(loaded.smooth_terms) == len(result.smooth_terms)


class TestNaturalSplineSmooth:
    """Test smooth terms with natural splines."""

    def test_ns_smooth(self):
        data = _make_poisson_data()
        result = rs.glm("y ~ ns(x, k=10)", data, family="poisson").fit(max_iter=100)
        assert result.converged
        assert result.has_smooth_terms()

    def test_ns_fixed(self):
        data = _make_poisson_data()
        result = rs.glm("y ~ ns(x, df=4)", data, family="poisson").fit()
        assert result.converged
        assert not result.has_smooth_terms()


class TestSmoothRecovery:
    """Verify the smooth actually recovers the true underlying curve."""

    def test_poisson_sin_recovery(self):
        """Smooth fit should approximate the true sin(x) curve, not just converge."""
        rng = np.random.default_rng(42)
        n = 1000
        x = rng.uniform(0, 10, n)
        eta_true = 0.5 + 0.3 * np.sin(x)
        mu_true = np.exp(eta_true)
        y = rng.poisson(mu_true)
        data = pl.DataFrame({"y": y, "x": x.round(6)})

        result = rs.glm("y ~ bs(x, k=10)", data, family="poisson").fit(max_iter=100)

        # Predictions should be close to true mu
        pred = result.fittedvalues
        # RMSE between predicted and true mu should be small relative to mean(mu)
        rmse = np.sqrt(np.mean((pred - mu_true) ** 2))
        mean_mu = np.mean(mu_true)
        relative_rmse = rmse / mean_mu
        assert relative_rmse < 0.15, f"Relative RMSE {relative_rmse:.3f} too large"

    def test_gaussian_sin_recovery(self):
        """Gaussian smooth should recover sin(x) with tight tolerance."""
        rng = np.random.default_rng(42)
        n = 1000
        x = rng.uniform(0, 10, n)
        mu_true = 2.0 + np.sin(x)
        y = mu_true + rng.normal(0, 0.2, n)
        data = pl.DataFrame({"y": y.round(6), "x": x.round(6)})

        result = rs.glm("y ~ bs(x, k=15)", data, family="gaussian").fit(max_iter=100)
        pred = result.fittedvalues

        # With low noise (sigma=0.2), predictions should track truth closely
        rmse = np.sqrt(np.mean((pred - mu_true) ** 2))
        assert rmse < 0.1, f"RMSE {rmse:.4f} too large for low-noise Gaussian"

    def test_smooth_beats_linear(self):
        """Smooth model should have lower deviance than linear on sinusoidal data."""
        data = _make_poisson_data(n=500)
        r_linear = rs.glm("y ~ x", data, family="poisson").fit()
        r_smooth = rs.glm("y ~ bs(x, k=10)", data, family="poisson").fit(max_iter=100)

        assert r_smooth.deviance < r_linear.deviance, (
            f"Smooth deviance {r_smooth.deviance:.2f} >= linear {r_linear.deviance:.2f}"
        )


class TestMonotonicCorrectness:
    """Verify monotonic constraints actually produce monotone predictions."""

    def test_increasing_predictions_are_monotone(self):
        """Predictions on a grid should be non-decreasing."""
        rng = np.random.default_rng(42)
        n = 500
        x = rng.uniform(0, 10, n)
        eta = 0.5 + 0.3 * x
        y = rng.poisson(np.exp(eta))
        data = pl.DataFrame({"y": y, "x": x.round(4)})
        result = rs.glm(
            "y ~ bs(x, k=8, monotonicity='increasing')",
            data, family="poisson"
        ).fit(max_iter=100)

        # Predict on an ordered grid
        x_grid = np.linspace(0.1, 9.9, 50)
        grid_data = pl.DataFrame({"x": x_grid})
        pred = result.predict(grid_data)

        # Predictions must be non-decreasing
        diffs = np.diff(pred)
        assert np.all(diffs >= -1e-10), (
            f"Monotone increasing violated: min diff = {diffs.min():.6f}"
        )

    def test_decreasing_predictions_are_monotone(self):
        """Predictions on a grid should be non-increasing."""
        rng = np.random.default_rng(42)
        n = 500
        x = rng.uniform(0, 10, n)
        eta = 2.0 - 0.3 * x
        y = rng.poisson(np.exp(eta))
        data = pl.DataFrame({"y": y, "x": x.round(4)})
        result = rs.glm(
            "y ~ bs(x, k=8, monotonicity='decreasing')",
            data, family="poisson"
        ).fit(max_iter=100)

        x_grid = np.linspace(0.1, 9.9, 50)
        grid_data = pl.DataFrame({"x": x_grid})
        pred = result.predict(grid_data)

        diffs = np.diff(pred)
        assert np.all(diffs <= 1e-10), (
            f"Monotone decreasing violated: max diff = {diffs.max():.6f}"
        )


class TestCoefficientReordering:
    """Verify _ReorderedResult correctly maps coefficients to feature names."""

    def test_parametric_coef_with_smooth(self):
        """Parametric coefficient should be similar with and without smooth term."""
        rng = np.random.default_rng(42)
        n = 500
        x1 = rng.uniform(0, 10, n)
        x2 = rng.normal(0, 1, n)
        # Strong linear x2 effect, weak smooth x1 effect
        eta = 0.5 + 0.05 * np.sin(x1) + 0.8 * x2
        y = rng.poisson(np.exp(eta))
        data = pl.DataFrame({"y": y, "x1": x1.round(4), "x2": x2.round(4)})

        # Fit with smooth
        r_smooth = rs.glm(
            "y ~ bs(x1, k=10) + x2", data, family="poisson"
        ).fit(max_iter=100)

        # Fit linear only on x2
        r_linear = rs.glm("y ~ x2", data, family="poisson").fit()

        # Find x2 coefficient in the smooth model
        x2_idx = r_smooth.feature_names.index("x2")
        x2_coef_smooth = r_smooth.params[x2_idx]

        # Find x2 coefficient in the linear model
        x2_idx_lin = r_linear.feature_names.index("x2")
        x2_coef_linear = r_linear.params[x2_idx_lin]

        # Should be in same ballpark (both recovering true β=0.8)
        assert abs(x2_coef_smooth - x2_coef_linear) < 0.3, (
            f"x2 coef smooth={x2_coef_smooth:.3f} vs linear={x2_coef_linear:.3f}"
        )
        # Both should be close to the true value 0.8
        assert abs(x2_coef_smooth - 0.8) < 0.2, (
            f"x2 coef smooth={x2_coef_smooth:.3f}, expected ~0.8"
        )

    def test_reordered_bse_matches_params_length(self):
        """BSE, tvalues, pvalues must align with params after reordering."""
        rng = np.random.default_rng(42)
        n = 300
        x1 = rng.uniform(0, 10, n)
        x2 = rng.normal(0, 1, n)
        y = rng.poisson(np.exp(0.5 + 0.1 * x1 + 0.3 * x2))
        data = pl.DataFrame({"y": y, "x1": x1.round(4), "x2": x2.round(4)})

        result = rs.glm(
            "y ~ bs(x1, k=8) + x2", data, family="poisson"
        ).fit(max_iter=100)

        n_params = len(result.params)
        assert len(result.bse()) == n_params
        assert len(result.tvalues()) == n_params
        assert len(result.pvalues()) == n_params
        assert result.conf_int().shape == (n_params, 2)

        # tvalue = params / bse (verify the relationship holds)
        tvals = result.tvalues()
        bse = result.bse()
        params = result.params
        for i in range(n_params):
            if bse[i] > 1e-10:
                expected_t = params[i] / bse[i]
                assert abs(tvals[i] - expected_t) < 1e-6, (
                    f"tvalue[{i}] = {tvals[i]:.6f} != params/bse = {expected_t:.6f}"
                )


class TestDiagnosticsValues:
    """Verify diagnostic values are numerically valid, not just the right shape."""

    def test_bse_finite_positive(self):
        data = _make_poisson_data()
        result = rs.glm("y ~ bs(x, k=10)", data, family="poisson").fit(max_iter=100)
        bse = result.bse()
        assert np.all(np.isfinite(bse)), "BSE contains NaN/Inf"
        assert np.all(bse >= 0), "BSE contains negative values"

    def test_pvalues_in_unit_interval(self):
        data = _make_poisson_data()
        result = rs.glm("y ~ bs(x, k=10)", data, family="poisson").fit(max_iter=100)
        pvals = result.pvalues()
        assert np.all(np.isfinite(pvals)), "pvalues contain NaN/Inf"
        assert np.all(pvals >= 0) and np.all(pvals <= 1), (
            f"pvalues outside [0,1]: min={pvals.min():.6f}, max={pvals.max():.6f}"
        )

    def test_conf_int_lower_less_than_upper(self):
        data = _make_poisson_data()
        result = rs.glm("y ~ bs(x, k=10)", data, family="poisson").fit(max_iter=100)
        ci = result.conf_int()
        assert np.all(ci[:, 0] <= ci[:, 1]), "Lower CI > Upper CI for some params"

    def test_residuals_sum_to_zero_gaussian(self):
        """For Gaussian, response residuals should sum to approximately zero."""
        data = _make_gaussian_data()
        result = rs.glm("y ~ bs(x, k=10)", data, family="gaussian").fit(max_iter=100)
        resid = result.resid_response()
        assert abs(np.sum(resid)) < 1.0, f"Residuals sum = {np.sum(resid):.4f}"


class TestOutOfSamplePrediction:
    """Verify smooth models produce sensible predictions on new data."""

    def test_predict_new_data_poisson(self):
        """Predictions on new x values should be in reasonable range."""
        rng = np.random.default_rng(42)
        n = 500
        x = rng.uniform(0, 10, n)
        eta = 0.5 + 0.3 * np.sin(x)
        y = rng.poisson(np.exp(eta))
        train = pl.DataFrame({"y": y, "x": x.round(4)})

        result = rs.glm("y ~ bs(x, k=10)", train, family="poisson").fit(max_iter=100)

        # Predict on new data within training range
        x_new = np.linspace(0.5, 9.5, 20)
        new_data = pl.DataFrame({"x": x_new})
        pred = result.predict(new_data)

        # True mu at these points
        mu_true = np.exp(0.5 + 0.3 * np.sin(x_new))

        # Predictions should be close to true mu
        relative_error = np.abs(pred - mu_true) / mu_true
        assert np.mean(relative_error) < 0.15, (
            f"Mean relative prediction error {np.mean(relative_error):.3f} too large"
        )

    def test_predict_matches_fittedvalues_on_train(self):
        """predict(train_data) should match fittedvalues."""
        data = _make_poisson_data(n=200)
        result = rs.glm("y ~ bs(x, k=10)", data, family="poisson").fit(max_iter=100)

        pred = result.predict(data)
        fitted = result.fittedvalues
        np.testing.assert_allclose(pred, fitted, rtol=1e-6)


class TestEDFBehavior:
    """Verify EDF responds correctly to data complexity."""

    def test_linear_data_gives_low_edf(self):
        """When true effect is linear, EDF should be close to 2 (intercept + slope)."""
        rng = np.random.default_rng(42)
        n = 500
        x = rng.uniform(0, 10, n)
        # Purely linear effect
        y = rng.poisson(np.exp(0.5 + 0.1 * x))
        data = pl.DataFrame({"y": y, "x": x.round(4)})

        result = rs.glm("y ~ bs(x, k=10)", data, family="poisson").fit(max_iter=100)
        st = result.smooth_terms[0]
        # For linear data, EDF should be small — closer to 2 than to 10
        assert st.edf < 5.0, f"EDF={st.edf:.2f} too high for linear data"

    def test_wiggly_data_gives_higher_edf(self):
        """When true effect is wiggly, EDF should be higher."""
        rng = np.random.default_rng(42)
        n = 1000
        x = rng.uniform(0, 10, n)
        # Wiggly effect: sum of sines at different frequencies
        eta = 0.5 + 0.3 * np.sin(x) + 0.2 * np.sin(2 * x) + 0.1 * np.sin(3 * x)
        y = rng.poisson(np.exp(eta))
        data = pl.DataFrame({"y": y, "x": x.round(6)})

        result = rs.glm("y ~ bs(x, k=15)", data, family="poisson").fit(max_iter=100)
        st = result.smooth_terms[0]
        # Wiggly data should need more EDF than linear data
        assert st.edf > 3.0, f"EDF={st.edf:.2f} too low for wiggly data"
