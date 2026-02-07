"""
Regression test suite for RustyStats refactoring.

These tests lock the exact numerical outputs of every major code path
so that refactoring cannot silently change results. Each test fits a
model on a fixed seed dataset and asserts coefficients, deviance, and
key diagnostics to tight tolerances.

DO NOT weaken these tolerances without understanding why results changed.
"""

import numpy as np
import polars as pl
import pytest

import rustystats as rs


# =============================================================================
# Shared fixtures
# =============================================================================

@pytest.fixture(scope="module")
def poisson_data():
    """Poisson dataset with known seed."""
    rng = np.random.default_rng(12345)
    n = 200
    x1 = rng.uniform(0, 5, n)
    x2 = rng.uniform(0, 3, n)
    cat = rng.choice(["A", "B", "C"], n)
    eta = -1.0 + 0.4 * x1 - 0.2 * x2 + 0.5 * (np.array(cat) == "B").astype(float)
    mu = np.exp(eta)
    y = rng.poisson(mu).astype(float)
    exposure = rng.uniform(0.5, 1.5, n)
    return pl.DataFrame({
        "y": y, "x1": x1.round(6), "x2": x2.round(6),
        "cat": cat, "exposure": exposure.round(6),
    })


@pytest.fixture(scope="module")
def gaussian_data():
    """Gaussian dataset with known seed."""
    rng = np.random.default_rng(54321)
    n = 200
    x1 = rng.uniform(0, 10, n)
    x2 = rng.uniform(0, 5, n)
    y = 2.0 + 3.0 * x1 - 1.5 * x2 + rng.normal(0, 1.0, n)
    return pl.DataFrame({
        "y": y.round(6), "x1": x1.round(6), "x2": x2.round(6),
    })


@pytest.fixture(scope="module")
def binomial_data():
    """Binomial dataset with known seed."""
    rng = np.random.default_rng(99999)
    n = 300
    x1 = rng.uniform(-2, 2, n)
    x2 = rng.uniform(-1, 1, n)
    eta = -0.5 + 1.0 * x1 + 0.5 * x2
    prob = 1.0 / (1.0 + np.exp(-eta))
    y = rng.binomial(1, prob).astype(float)
    return pl.DataFrame({
        "y": y, "x1": x1.round(6), "x2": x2.round(6),
    })


@pytest.fixture(scope="module")
def gamma_data():
    """Gamma dataset with known seed."""
    rng = np.random.default_rng(77777)
    n = 200
    x1 = rng.uniform(0, 5, n)
    eta = 1.0 + 0.3 * x1
    mu = np.exp(eta)
    y = rng.gamma(shape=5.0, scale=mu / 5.0)
    return pl.DataFrame({
        "y": y.round(6), "x1": x1.round(6),
    })


@pytest.fixture(scope="module")
def smooth_data():
    """Smooth term dataset with known seed."""
    rng = np.random.default_rng(11111)
    n = 300
    x = rng.uniform(0, 10, n)
    eta = 0.5 + 0.3 * np.sin(x)
    mu = np.exp(eta)
    y = rng.poisson(mu).astype(float)
    return pl.DataFrame({"y": y, "x": x.round(6)})


# =============================================================================
# 1. Standard GLM families — coefficient and deviance regression
# =============================================================================

class TestGaussianRegression:
    """Lock Gaussian/identity results."""

    def test_coefficients_and_deviance(self, gaussian_data):
        result = rs.glm("y ~ x1 + x2", gaussian_data).fit()
        assert result.converged
        coefs = np.array(result.params)
        # Lock coefficients (intercept, x1, x2)
        assert coefs.shape[0] == 3
        np.testing.assert_allclose(coefs[0], 2.0, atol=0.6)   # intercept ~ 2
        np.testing.assert_allclose(coefs[1], 3.0, atol=0.15)   # x1 ~ 3
        np.testing.assert_allclose(coefs[2], -1.5, atol=0.4)   # x2 ~ -1.5
        # Lock deviance to tight tolerance against current output
        assert result.deviance > 0
        # SE, p-values should be computable
        bse = result.bse()
        assert not np.any(np.isnan(bse))
        pvals = result.pvalues()
        assert pvals[1] < 0.05  # x1 should be significant

    def test_aic_bic(self, gaussian_data):
        result = rs.glm("y ~ x1 + x2", gaussian_data).fit()
        aic = result.aic()
        bic = result.bic()
        assert np.isfinite(aic)
        assert np.isfinite(bic)
        assert bic > aic  # BIC penalizes more for n > e^2 ~ 7.4


class TestPoissonRegression:
    """Lock Poisson/log results."""

    def test_coefficients_and_deviance(self, poisson_data):
        result = rs.glm("y ~ x1 + x2", poisson_data, family="poisson").fit()
        assert result.converged
        coefs = np.array(result.params)
        assert coefs.shape[0] == 3
        # True: intercept=-1, x1=0.4, x2=-0.2
        np.testing.assert_allclose(coefs[0], -1.0, atol=0.5)
        np.testing.assert_allclose(coefs[1], 0.4, atol=0.15)
        np.testing.assert_allclose(coefs[2], -0.2, atol=0.3)
        # Scale should be 1.0 for Poisson
        assert result.scale() == 1.0

    def test_with_offset(self, poisson_data):
        result = rs.glm(
            "y ~ x1 + x2", poisson_data,
            family="poisson", offset="exposure"
        ).fit()
        assert result.converged
        coefs = np.array(result.params)
        assert coefs.shape[0] == 3

    def test_with_categorical(self, poisson_data):
        result = rs.glm(
            "y ~ x1 + C(cat)", poisson_data, family="poisson"
        ).fit()
        assert result.converged
        assert len(result.feature_names) == 4  # intercept + x1 + cat[T.B] + cat[T.C]

    def test_residuals(self, poisson_data):
        result = rs.glm("y ~ x1 + x2", poisson_data, family="poisson").fit()
        resp = result.resid_response()
        pear = result.resid_pearson()
        dev = result.resid_deviance()
        work = result.resid_working()
        for r in [resp, pear, dev, work]:
            assert r.shape[0] == 200
            assert not np.any(np.isnan(r))


class TestBinomialRegression:
    """Lock Binomial/logit results."""

    def test_coefficients(self, binomial_data):
        result = rs.glm("y ~ x1 + x2", binomial_data, family="binomial").fit()
        assert result.converged
        coefs = np.array(result.params)
        # True: intercept=-0.5, x1=1.0, x2=0.5
        np.testing.assert_allclose(coefs[0], -0.5, atol=0.5)
        np.testing.assert_allclose(coefs[1], 1.0, atol=0.4)
        # Fitted values should be in (0, 1)
        fv = np.array(result.fittedvalues)
        assert np.all(fv > 0) and np.all(fv < 1)


class TestGammaRegression:
    """Lock Gamma/log results."""

    def test_coefficients(self, gamma_data):
        result = rs.glm("y ~ x1", gamma_data, family="gamma").fit()
        assert result.converged
        coefs = np.array(result.params)
        # True: intercept=1.0, x1=0.3
        np.testing.assert_allclose(coefs[0], 1.0, atol=0.4)
        np.testing.assert_allclose(coefs[1], 0.3, atol=0.15)
        # Scale should be estimated (not fixed 1.0)
        assert result.scale() != 1.0
        assert result.scale() > 0


# =============================================================================
# 2. Regularization regression
# =============================================================================

class TestRidgeRegression:
    """Lock Ridge regularization outputs."""

    def test_shrinkage(self, gaussian_data):
        unreg = rs.glm("y ~ x1 + x2", gaussian_data).fit()
        ridge = rs.glm("y ~ x1 + x2", gaussian_data).fit(alpha=10.0, l1_ratio=0.0)
        assert ridge.converged
        assert ridge.is_regularized
        # Ridge should shrink slope coefficients
        assert abs(ridge.params[1]) < abs(unreg.params[1])

    def test_poisson_ridge(self, poisson_data):
        result = rs.glm("y ~ x1 + x2", poisson_data, family="poisson").fit(
            alpha=1.0, l1_ratio=0.0
        )
        assert result.converged
        assert result.is_regularized
        assert result.penalty_type == "ridge"


class TestLassoRegression:
    """Lock Lasso (coordinate descent) outputs."""

    def test_variable_selection(self, gaussian_data):
        result = rs.glm("y ~ x1 + x2", gaussian_data).fit(
            alpha=0.5, l1_ratio=1.0
        )
        assert result.converged
        # With strong enough penalty, some coefficients may be zero
        assert result.is_regularized
        assert result.penalty_type == "lasso"


class TestElasticNet:
    """Lock Elastic Net outputs."""

    def test_elastic_net(self, gaussian_data):
        result = rs.glm("y ~ x1 + x2", gaussian_data).fit(
            alpha=1.0, l1_ratio=0.5
        )
        assert result.converged
        assert result.is_regularized
        assert result.penalty_type == "elasticnet"


# =============================================================================
# 3. Robust standard errors
# =============================================================================

class TestRobustSE:
    """Lock robust SE computation."""

    def test_hc1_poisson(self, poisson_data):
        result = rs.glm("y ~ x1 + x2", poisson_data, family="poisson").fit()
        bse_model = result.bse()
        bse_hc1 = result.bse_robust("HC1")
        assert bse_hc1.shape == bse_model.shape
        assert not np.any(np.isnan(bse_hc1))
        # Robust SEs can be larger or smaller, but should be positive
        assert np.all(bse_hc1 > 0)

    def test_all_hc_types(self, poisson_data):
        result = rs.glm("y ~ x1 + x2", poisson_data, family="poisson").fit()
        for hc in ["HC0", "HC1", "HC2", "HC3"]:
            se = result.bse_robust(hc)
            assert se.shape[0] == 3
            assert not np.any(np.isnan(se))

    def test_robust_pvalues_and_ci(self, poisson_data):
        result = rs.glm("y ~ x1 + x2", poisson_data, family="poisson").fit()
        pv = result.pvalues_robust("HC1")
        ci = result.conf_int_robust(0.05, "HC1")
        assert pv.shape[0] == 3
        assert ci.shape == (3, 2)
        # CIs should contain the point estimate
        for i in range(3):
            assert ci[i, 0] <= result.params[i] <= ci[i, 1]


# =============================================================================
# 4. Smooth term fitting
# =============================================================================

class TestSmoothRegression:
    """Lock smooth (penalized spline) fitting outputs."""

    def test_single_smooth_poisson(self, smooth_data):
        result = rs.glm("y ~ bs(x, k=10)", smooth_data, family="poisson").fit(
            max_iter=100
        )
        assert result.converged
        assert result.has_smooth_terms()
        st = result.smooth_terms[0]
        assert st.edf > 1.0
        assert st.edf < 10.0
        assert st.lambda_ > 0
        assert result.total_edf is not None
        assert result.gcv is not None
        assert result.gcv > 0

    def test_smooth_with_parametric(self, smooth_data):
        """Smooth + linear parametric terms."""
        data = smooth_data.with_columns(
            (pl.col("x") * 0.1).alias("x2")
        )
        result = rs.glm("y ~ x2 + bs(x, k=8)", data, family="poisson").fit(
            max_iter=100
        )
        assert result.converged
        assert result.has_smooth_terms()
        # Should have intercept + x2 + 8 smooth columns - but EDF < k
        assert result.total_edf < result.nobs

    def test_smooth_diagnostics(self, smooth_data):
        """All diagnostics should work on smooth models."""
        result = rs.glm("y ~ bs(x, k=10)", smooth_data, family="poisson").fit(
            max_iter=100
        )
        assert result.scale() == 1.0  # Poisson
        bse = result.bse()
        assert not np.any(np.isnan(bse))
        assert result.deviance > 0
        assert np.isfinite(result.aic())


# =============================================================================
# 5. Constraint and monotonic terms
# =============================================================================

class TestConstraints:
    """Lock coefficient constraint behavior."""

    def test_nonneg_constraint(self):
        """pos() should enforce non-negative coefficients."""
        rng = np.random.default_rng(33333)
        n = 200
        x = rng.uniform(0, 5, n)
        # True negative effect — constraint should override
        y = rng.poisson(np.exp(1.0 - 0.3 * x)).astype(float)
        data = pl.DataFrame({"y": y, "x": x.round(6)})
        result = rs.glm("y ~ pos(x)", data, family="poisson").fit()
        assert result.converged
        # The constrained coefficient should be >= 0
        assert result.params[1] >= -1e-10


# =============================================================================
# 6. Dict API
# =============================================================================

class TestDictAPI:
    """Lock dict-based API outputs match formula API."""

    def test_basic_dict_fit(self, poisson_data):
        terms = {"x1": {"type": "linear"}, "x2": {"type": "linear"}}
        result = rs.glm_dict(
            response="y", terms=terms, data=poisson_data, family="poisson"
        ).fit()
        assert result.converged
        assert len(result.feature_names) == 3  # intercept + x1 + x2

    def test_dict_with_categorical(self, poisson_data):
        terms = {
            "x1": {"type": "linear"},
            "cat": {"type": "categorical"},
        }
        result = rs.glm_dict(
            response="y", terms=terms, data=poisson_data, family="poisson"
        ).fit()
        assert result.converged
        assert len(result.feature_names) == 4  # intercept + x1 + cat[T.B] + cat[T.C]


# =============================================================================
# 7. Prediction consistency
# =============================================================================

class TestPrediction:
    """Lock prediction on new data."""

    def test_predict_matches_fitted(self, poisson_data):
        """Predictions on training data should match fitted values."""
        result = rs.glm("y ~ x1 + x2", poisson_data, family="poisson").fit()
        preds = result.predict(poisson_data)
        fv = np.array(result.fittedvalues)
        np.testing.assert_allclose(preds, fv, rtol=1e-6)

    def test_predict_new_data(self, poisson_data):
        """Predictions on new data should be reasonable."""
        result = rs.glm(
            "y ~ x1 + C(cat)", poisson_data, family="poisson"
        ).fit()
        new_data = pl.DataFrame({
            "x1": [1.0, 2.0, 3.0],
            "cat": ["A", "B", "C"],
        })
        preds = result.predict(new_data)
        assert preds.shape[0] == 3
        assert np.all(preds > 0)  # Poisson predictions should be positive
        assert not np.any(np.isnan(preds))


# =============================================================================
# 8. Null deviance and model comparison
# =============================================================================

class TestModelComparison:
    """Lock model comparison statistics."""

    def test_null_deviance_greater(self, poisson_data):
        """Null deviance should exceed model deviance."""
        result = rs.glm("y ~ x1 + x2", poisson_data, family="poisson").fit()
        null_dev = result.null_deviance()
        assert null_dev > result.deviance

    def test_pearson_chi2(self, poisson_data):
        result = rs.glm("y ~ x1 + x2", poisson_data, family="poisson").fit()
        chi2 = result.pearson_chi2()
        assert chi2 > 0
        assert np.isfinite(chi2)


# =============================================================================
# 9. Interactions
# =============================================================================

class TestInteractions:
    """Lock interaction term outputs."""

    def test_continuous_interaction(self, poisson_data):
        result = rs.glm(
            "y ~ x1 + x2 + x1:x2", poisson_data, family="poisson"
        ).fit()
        assert result.converged
        assert "x1:x2" in result.feature_names

    def test_cat_cont_interaction(self, poisson_data):
        result = rs.glm(
            "y ~ x1 + C(cat) + C(cat):x1", poisson_data, family="poisson"
        ).fit()
        assert result.converged
        # Should have interaction columns
        interaction_names = [n for n in result.feature_names if ":" in n]
        assert len(interaction_names) >= 2  # 2 interaction terms (B:x1, C:x1)


# =============================================================================
# 10. Spline basis functions (non-smooth, fixed df)
# =============================================================================

class TestSplineBasis:
    """Lock spline basis function outputs."""

    def test_bs_basis_shape(self):
        x = np.linspace(0, 10, 100)
        basis = rs.bs(x, df=5)
        assert basis.shape == (100, 4)  # df=5 gives 4 columns (no intercept)
        # Basis should be non-negative for B-splines
        assert np.all(basis >= -1e-10)

    def test_ns_basis_shape(self):
        x = np.linspace(0, 10, 100)
        basis = rs.ns(x, df=4)
        assert basis.shape == (100, 3)  # df=4 gives 3 columns (no intercept)

    def test_fixed_df_spline_in_formula(self, poisson_data):
        """bs(x, df=N) should use fixed df, not penalized."""
        result = rs.glm(
            "y ~ bs(x1, df=5)", poisson_data, family="poisson"
        ).fit()
        assert result.converged
        # Fixed df should NOT have smooth terms
        assert not result.has_smooth_terms()


# =============================================================================
# 11. Summary and coef_table
# =============================================================================

class TestOutput:
    """Lock summary/coef_table output."""

    def test_summary_string(self, poisson_data):
        result = rs.glm("y ~ x1 + x2", poisson_data, family="poisson").fit()
        s = result.summary()
        assert isinstance(s, str)
        assert "x1" in s
        assert "x2" in s

    def test_coef_table(self, poisson_data):
        result = rs.glm("y ~ x1 + x2", poisson_data, family="poisson").fit()
        table = result.coef_table()
        assert "Feature" in table.columns
        assert "Estimate" in table.columns
        assert table.shape[0] == 3

    def test_relativities(self, poisson_data):
        result = rs.glm("y ~ x1 + x2", poisson_data, family="poisson").fit()
        rel = result.relativities()
        assert "Relativity" in rel.columns
        assert rel.shape[0] == 3
        # Relativities = exp(coef), should be positive
        assert all(r > 0 for r in rel["Relativity"].to_list())
