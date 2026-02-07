"""
Pinned numerical regression tests.

These tests lock EXACT numerical outputs (coefficients, deviance, AIC, etc.)
to tight tolerances. They will fail if refactoring changes fitted values,
even slightly. This is intentional — any change must be understood and
the pinned values updated deliberately.

DO NOT weaken tolerances without understanding why results changed.
"""

import numpy as np
import polars as pl
import pytest

import rustystats as rs


# =============================================================================
# Shared fixtures — deterministic data with fixed seeds
# =============================================================================

@pytest.fixture(scope="module")
def poisson_data():
    rng = np.random.default_rng(12345)
    n = 200
    x1 = rng.uniform(0, 5, n)
    x2 = rng.uniform(0, 3, n)
    cat = rng.choice(["A", "B", "C"], n)
    eta = -1.0 + 0.4 * x1 - 0.2 * x2 + 0.5 * (np.array(cat) == "B").astype(float)
    mu = np.exp(eta)
    y = rng.poisson(mu).astype(float)
    return pl.DataFrame({
        "y": y, "x1": x1.round(6), "x2": x2.round(6), "cat": cat,
    })


@pytest.fixture(scope="module")
def gaussian_data():
    rng = np.random.default_rng(54321)
    n = 200
    x1 = rng.uniform(0, 10, n)
    x2 = rng.uniform(0, 5, n)
    y = 2.0 + 3.0 * x1 - 1.5 * x2 + rng.normal(0, 1.0, n)
    return pl.DataFrame({
        "y": y.round(6), "x1": x1.round(6), "x2": x2.round(6),
    })


@pytest.fixture(scope="module")
def smooth_data():
    rng = np.random.default_rng(11111)
    n = 300
    x = rng.uniform(0, 10, n)
    eta = 0.5 + 0.3 * np.sin(x)
    mu = np.exp(eta)
    y = rng.poisson(mu).astype(float)
    return pl.DataFrame({"y": y, "x": x.round(6)})


# =============================================================================
# Poisson regression — pinned values
# =============================================================================

class TestPoissonPinned:
    """Exact numerical lock for Poisson GLM."""

    def test_coefficients(self, poisson_data):
        result = rs.glm("y ~ x1 + x2", poisson_data, family="poisson").fit()
        params = np.array(result.params)
        np.testing.assert_allclose(params[0], -0.8871353436072057, rtol=1e-8)
        np.testing.assert_allclose(params[1],  0.416298083133042,  rtol=1e-8)
        np.testing.assert_allclose(params[2], -0.1941620413155981, rtol=1e-8)

    def test_deviance(self, poisson_data):
        result = rs.glm("y ~ x1 + x2", poisson_data, family="poisson").fit()
        np.testing.assert_allclose(result.deviance, 218.72232983646379, rtol=1e-8)

    def test_scale(self, poisson_data):
        result = rs.glm("y ~ x1 + x2", poisson_data, family="poisson").fit()
        assert result.scale() == 1.0  # Exact for Poisson

    def test_iterations(self, poisson_data):
        result = rs.glm("y ~ x1 + x2", poisson_data, family="poisson").fit()
        assert result.iterations == 5

    def test_bse(self, poisson_data):
        result = rs.glm("y ~ x1 + x2", poisson_data, family="poisson").fit()
        bse = result.bse()
        np.testing.assert_allclose(bse[0], 0.20754325554066552, rtol=1e-8)
        np.testing.assert_allclose(bse[1], 0.05128829669131667, rtol=1e-8)
        np.testing.assert_allclose(bse[2], 0.08024857183824942, rtol=1e-8)

    def test_aic_bic(self, poisson_data):
        result = rs.glm("y ~ x1 + x2", poisson_data, family="poisson").fit()
        np.testing.assert_allclose(result.aic(), 497.56113066707337, rtol=1e-8)
        np.testing.assert_allclose(result.bic(), 507.4560827667175,  rtol=1e-8)

    def test_pearson_chi2(self, poisson_data):
        result = rs.glm("y ~ x1 + x2", poisson_data, family="poisson").fit()
        np.testing.assert_allclose(result.pearson_chi2(), 207.0163514748195, rtol=1e-8)

    def test_null_deviance(self, poisson_data):
        result = rs.glm("y ~ x1 + x2", poisson_data, family="poisson").fit()
        np.testing.assert_allclose(result.null_deviance(), 294.07918329160407, rtol=1e-8)

    def test_df(self, poisson_data):
        result = rs.glm("y ~ x1 + x2", poisson_data, family="poisson").fit()
        assert result.nobs == 200
        assert result.df_resid == 197


# =============================================================================
# Gaussian regression — pinned values
# =============================================================================

class TestGaussianPinned:
    """Exact numerical lock for Gaussian GLM."""

    def test_coefficients(self, gaussian_data):
        result = rs.glm("y ~ x1 + x2", gaussian_data).fit()
        params = np.array(result.params)
        np.testing.assert_allclose(params[0], 2.0543500369910856, rtol=1e-8)
        np.testing.assert_allclose(params[1], 2.9822761593212435, rtol=1e-8)
        np.testing.assert_allclose(params[2], -1.5144258395436492, rtol=1e-8)

    def test_deviance(self, gaussian_data):
        result = rs.glm("y ~ x1 + x2", gaussian_data).fit()
        np.testing.assert_allclose(result.deviance, 191.05103776261143, rtol=1e-8)

    def test_scale(self, gaussian_data):
        result = rs.glm("y ~ x1 + x2", gaussian_data).fit()
        np.testing.assert_allclose(result.scale(), 0.9698022221452357, rtol=1e-8)

    def test_aic_bic(self, gaussian_data):
        result = rs.glm("y ~ x1 + x2", gaussian_data).fit()
        np.testing.assert_allclose(result.aic(), 564.4427886879308, rtol=1e-6)
        np.testing.assert_allclose(result.bic(), 574.337740787575, rtol=1e-6)


# =============================================================================
# Smooth Poisson regression — pinned values
# =============================================================================

class TestSmoothPoissonPinned:
    """Exact numerical lock for smooth Poisson GAM."""

    def test_deviance(self, smooth_data):
        result = rs.glm("y ~ bs(x, k=10)", smooth_data, family="poisson").fit(max_iter=100)
        np.testing.assert_allclose(result.deviance, 329.8268924721029, rtol=1e-6)

    def test_total_edf(self, smooth_data):
        result = rs.glm("y ~ bs(x, k=10)", smooth_data, family="poisson").fit(max_iter=100)
        np.testing.assert_allclose(result.total_edf, 6.554292645590362, rtol=1e-4)

    def test_gcv(self, smooth_data):
        result = rs.glm("y ~ bs(x, k=10)", smooth_data, family="poisson").fit(max_iter=100)
        np.testing.assert_allclose(result.gcv, 1.1490840489275542, rtol=1e-4)

    def test_smooth_edf(self, smooth_data):
        result = rs.glm("y ~ bs(x, k=10)", smooth_data, family="poisson").fit(max_iter=100)
        st = result.smooth_terms[0]
        np.testing.assert_allclose(st.edf, 5.554292645590362, rtol=1e-4)

    def test_smooth_lambda(self, smooth_data):
        result = rs.glm("y ~ bs(x, k=10)", smooth_data, family="poisson").fit(max_iter=100)
        st = result.smooth_terms[0]
        np.testing.assert_allclose(st.lambda_, 6.043386590071252, rtol=1e-3)

    def test_first_coefficients(self, smooth_data):
        result = rs.glm("y ~ bs(x, k=10)", smooth_data, family="poisson").fit(max_iter=100)
        np.testing.assert_allclose(result.params[0], 0.5425650612276607, rtol=1e-4)
