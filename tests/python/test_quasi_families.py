"""
Tests for Quasi-Families (QuasiPoisson, QuasiBinomial)
======================================================

These tests verify that:
1. Quasi-families produce the same point estimates as their base families
2. Standard errors are inflated by √φ where φ is the estimated dispersion
3. Dispersion is correctly estimated from Pearson residuals
4. All API methods work correctly
"""

import numpy as np
import pytest
import rustystats as rs


# =============================================================================
# Test Data Fixtures
# =============================================================================

@pytest.fixture
def overdispersed_count_data():
    """Generate overdispersed count data for QuasiPoisson testing."""
    np.random.seed(42)
    n = 500
    
    # Generate predictors
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    
    # Generate overdispersed counts using negative binomial
    # (which has more variance than Poisson)
    mu = np.exp(0.5 + 0.3 * x1 - 0.2 * x2)
    
    # Add extra variance by using a mixture
    overdispersion = 2.0
    y = np.random.negative_binomial(
        n=1/overdispersion,  # r parameter
        p=1/(1 + overdispersion * mu)  # success probability
    )
    
    # Design matrix with intercept
    X = np.column_stack([np.ones(n), x1, x2])
    
    return y.astype(float), X


@pytest.fixture
def overdispersed_binary_data():
    """Generate overdispersed binary data for QuasiBinomial testing."""
    np.random.seed(123)
    n = 500
    
    # Generate predictors
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    
    # True probabilities
    logit = 0.5 + 0.8 * x1 - 0.5 * x2
    p = 1 / (1 + np.exp(-logit))
    
    # Generate binary outcomes
    y = (np.random.rand(n) < p).astype(float)
    
    # Design matrix with intercept
    X = np.column_stack([np.ones(n), x1, x2])
    
    return y, X


# =============================================================================
# QuasiPoisson Tests
# =============================================================================

class TestQuasiPoisson:
    """Tests for QuasiPoisson family."""
    
    def test_quasipoisson_basic_fit(self, overdispersed_count_data):
        """QuasiPoisson should fit without errors."""
        y, X = overdispersed_count_data
        result = rs.fit_glm(y, X, family="quasipoisson")
        
        assert result.converged
        assert len(result.params) == 3
        assert result.deviance > 0
    
    def test_quasipoisson_same_coefficients_as_poisson(self, overdispersed_count_data):
        """QuasiPoisson and Poisson should have identical point estimates."""
        y, X = overdispersed_count_data
        
        result_poisson = rs.fit_glm(y, X, family="poisson")
        result_quasi = rs.fit_glm(y, X, family="quasipoisson")
        
        # Coefficients should be identical
        np.testing.assert_allclose(
            result_quasi.params,
            result_poisson.params,
            rtol=1e-10
        )
        
        # Deviance should be identical
        np.testing.assert_allclose(
            result_quasi.deviance,
            result_poisson.deviance,
            rtol=1e-10
        )
        
        # Fitted values should be identical
        np.testing.assert_allclose(
            result_quasi.fittedvalues,
            result_poisson.fittedvalues,
            rtol=1e-10
        )
    
    def test_quasipoisson_dispersion_estimated(self, overdispersed_count_data):
        """QuasiPoisson should estimate dispersion > 1 for overdispersed data."""
        y, X = overdispersed_count_data
        
        result_poisson = rs.fit_glm(y, X, family="poisson")
        result_quasi = rs.fit_glm(y, X, family="quasipoisson")
        
        # Poisson has fixed dispersion = 1
        assert result_poisson.scale() == 1.0
        
        # QuasiPoisson should estimate dispersion
        # For our overdispersed data, it should be > 1
        assert result_quasi.scale() > 1.0
    
    def test_quasipoisson_larger_standard_errors(self, overdispersed_count_data):
        """QuasiPoisson SE should be larger than Poisson SE for overdispersed data."""
        y, X = overdispersed_count_data
        
        result_poisson = rs.fit_glm(y, X, family="poisson")
        result_quasi = rs.fit_glm(y, X, family="quasipoisson")
        
        se_poisson = np.array(result_poisson.bse())
        se_quasi = np.array(result_quasi.bse())
        
        dispersion = result_quasi.scale()
        
        # QuasiPoisson SE should be √φ times Poisson SE
        expected_se_ratio = np.sqrt(dispersion)
        actual_se_ratio = se_quasi / se_poisson
        
        np.testing.assert_allclose(
            actual_se_ratio,
            expected_se_ratio * np.ones_like(actual_se_ratio),
            rtol=1e-6
        )
    
    def test_quasipoisson_wider_confidence_intervals(self, overdispersed_count_data):
        """QuasiPoisson CI should be wider than Poisson CI."""
        y, X = overdispersed_count_data
        
        result_poisson = rs.fit_glm(y, X, family="poisson")
        result_quasi = rs.fit_glm(y, X, family="quasipoisson")
        
        ci_poisson = np.array(result_poisson.conf_int())
        ci_quasi = np.array(result_quasi.conf_int())
        
        # CI width
        width_poisson = ci_poisson[:, 1] - ci_poisson[:, 0]
        width_quasi = ci_quasi[:, 1] - ci_quasi[:, 0]
        
        # QuasiPoisson should have wider CIs
        assert np.all(width_quasi >= width_poisson)
    
    def test_quasipoisson_conservative_pvalues(self, overdispersed_count_data):
        """QuasiPoisson p-values should be larger (more conservative)."""
        y, X = overdispersed_count_data
        
        result_poisson = rs.fit_glm(y, X, family="poisson")
        result_quasi = rs.fit_glm(y, X, family="quasipoisson")
        
        pvals_poisson = np.array(result_poisson.pvalues())
        pvals_quasi = np.array(result_quasi.pvalues())
        
        # QuasiPoisson p-values should be >= Poisson p-values
        # (more conservative due to larger SE)
        assert np.all(pvals_quasi >= pvals_poisson - 1e-10)
    
    def test_quasipoisson_family_name(self, overdispersed_count_data):
        """QuasiPoisson should report correct family name."""
        y, X = overdispersed_count_data
        result = rs.fit_glm(y, X, family="quasipoisson")
        assert result.family == "QuasiPoisson"
    
    def test_quasipoisson_alternative_spellings(self, overdispersed_count_data):
        """QuasiPoisson should accept various spellings."""
        y, X = overdispersed_count_data
        
        # All these should work
        r1 = rs.fit_glm(y, X, family="quasipoisson")
        r2 = rs.fit_glm(y, X, family="quasi-poisson")
        r3 = rs.fit_glm(y, X, family="quasi_poisson")
        
        # All should give same results
        np.testing.assert_allclose(r1.params, r2.params, rtol=1e-10)
        np.testing.assert_allclose(r1.params, r3.params, rtol=1e-10)
    
    def test_quasipoisson_residuals(self, overdispersed_count_data):
        """QuasiPoisson should compute all residual types."""
        y, X = overdispersed_count_data
        result = rs.fit_glm(y, X, family="quasipoisson")
        
        # All residual methods should work
        resid_response = result.resid_response()
        resid_pearson = result.resid_pearson()
        resid_deviance = result.resid_deviance()
        resid_working = result.resid_working()
        
        assert len(resid_response) == len(y)
        assert len(resid_pearson) == len(y)
        assert len(resid_deviance) == len(y)
        assert len(resid_working) == len(y)


# =============================================================================
# QuasiBinomial Tests
# =============================================================================

class TestQuasiBinomial:
    """Tests for QuasiBinomial family."""
    
    def test_quasibinomial_basic_fit(self, overdispersed_binary_data):
        """QuasiBinomial should fit without errors."""
        y, X = overdispersed_binary_data
        result = rs.fit_glm(y, X, family="quasibinomial")
        
        assert result.converged
        assert len(result.params) == 3
        assert result.deviance > 0
    
    def test_quasibinomial_same_coefficients_as_binomial(self, overdispersed_binary_data):
        """QuasiBinomial and Binomial should have identical point estimates."""
        y, X = overdispersed_binary_data
        
        result_binomial = rs.fit_glm(y, X, family="binomial")
        result_quasi = rs.fit_glm(y, X, family="quasibinomial")
        
        # Coefficients should be identical
        np.testing.assert_allclose(
            result_quasi.params,
            result_binomial.params,
            rtol=1e-10
        )
        
        # Deviance should be identical
        np.testing.assert_allclose(
            result_quasi.deviance,
            result_binomial.deviance,
            rtol=1e-10
        )
    
    def test_quasibinomial_dispersion_estimated(self, overdispersed_binary_data):
        """QuasiBinomial should estimate dispersion."""
        y, X = overdispersed_binary_data
        
        result_binomial = rs.fit_glm(y, X, family="binomial")
        result_quasi = rs.fit_glm(y, X, family="quasibinomial")
        
        # Binomial has fixed dispersion = 1
        assert result_binomial.scale() == 1.0
        
        # QuasiBinomial estimates dispersion
        dispersion = result_quasi.scale()
        assert dispersion > 0  # Should be positive
    
    def test_quasibinomial_se_relationship(self, overdispersed_binary_data):
        """QuasiBinomial SE should be √φ times Binomial SE."""
        y, X = overdispersed_binary_data
        
        result_binomial = rs.fit_glm(y, X, family="binomial")
        result_quasi = rs.fit_glm(y, X, family="quasibinomial")
        
        se_binomial = np.array(result_binomial.bse())
        se_quasi = np.array(result_quasi.bse())
        
        dispersion = result_quasi.scale()
        
        # QuasiBinomial SE should be √φ times Binomial SE
        expected_se_ratio = np.sqrt(dispersion)
        actual_se_ratio = se_quasi / se_binomial
        
        np.testing.assert_allclose(
            actual_se_ratio,
            expected_se_ratio * np.ones_like(actual_se_ratio),
            rtol=1e-6
        )
    
    def test_quasibinomial_family_name(self, overdispersed_binary_data):
        """QuasiBinomial should report correct family name."""
        y, X = overdispersed_binary_data
        result = rs.fit_glm(y, X, family="quasibinomial")
        assert result.family == "QuasiBinomial"
    
    def test_quasibinomial_alternative_spellings(self, overdispersed_binary_data):
        """QuasiBinomial should accept various spellings."""
        y, X = overdispersed_binary_data
        
        r1 = rs.fit_glm(y, X, family="quasibinomial")
        r2 = rs.fit_glm(y, X, family="quasi-binomial")
        r3 = rs.fit_glm(y, X, family="quasi_binomial")
        
        np.testing.assert_allclose(r1.params, r2.params, rtol=1e-10)
        np.testing.assert_allclose(r1.params, r3.params, rtol=1e-10)
    
    def test_quasibinomial_residuals(self, overdispersed_binary_data):
        """QuasiBinomial should compute all residual types."""
        y, X = overdispersed_binary_data
        result = rs.fit_glm(y, X, family="quasibinomial")
        
        resid_response = result.resid_response()
        resid_pearson = result.resid_pearson()
        resid_deviance = result.resid_deviance()
        resid_working = result.resid_working()
        
        assert len(resid_response) == len(y)
        assert len(resid_pearson) == len(y)
        assert len(resid_deviance) == len(y)
        assert len(resid_working) == len(y)


# =============================================================================
# Family Object Tests
# =============================================================================

class TestQuasiFamilyObjects:
    """Tests for QuasiPoisson and QuasiBinomial family objects."""
    
    def test_quasipoisson_family_object(self):
        """Test QuasiPoisson family object methods."""
        family = rs.families.QuasiPoisson()
        
        assert family.name() == "QuasiPoisson"
        
        mu = np.array([0.5, 1.0, 2.0, 5.0])
        variance = np.array(family.variance(mu))
        
        # Variance = μ (same as Poisson)
        np.testing.assert_allclose(variance, mu, rtol=1e-10)
    
    def test_quasibinomial_family_object(self):
        """Test QuasiBinomial family object methods."""
        family = rs.families.QuasiBinomial()
        
        assert family.name() == "QuasiBinomial"
        
        mu = np.array([0.2, 0.5, 0.8])
        variance = np.array(family.variance(mu))
        
        # Variance = μ(1-μ) (same as Binomial)
        expected_variance = mu * (1 - mu)
        np.testing.assert_allclose(variance, expected_variance, rtol=1e-10)
    
    def test_quasipoisson_deviance_equals_poisson(self):
        """QuasiPoisson deviance should match Poisson deviance."""
        quasi = rs.families.QuasiPoisson()
        poisson = rs.families.Poisson()
        
        y = np.array([0.0, 1.0, 3.0, 5.0])
        mu = np.array([1.0, 1.5, 2.5, 4.0])
        
        dev_quasi = np.array(quasi.unit_deviance(y, mu))
        dev_poisson = np.array(poisson.unit_deviance(y, mu))
        
        np.testing.assert_allclose(dev_quasi, dev_poisson, rtol=1e-10)
    
    def test_quasibinomial_deviance_equals_binomial(self):
        """QuasiBinomial deviance should match Binomial deviance."""
        quasi = rs.families.QuasiBinomial()
        binomial = rs.families.Binomial()
        
        y = np.array([0.0, 0.3, 0.7, 1.0])
        mu = np.array([0.2, 0.4, 0.6, 0.8])
        
        dev_quasi = np.array(quasi.unit_deviance(y, mu))
        dev_binomial = np.array(binomial.unit_deviance(y, mu))
        
        np.testing.assert_allclose(dev_quasi, dev_binomial, rtol=1e-10)


# =============================================================================
# Edge Cases and Special Scenarios
# =============================================================================

class TestQuasiFamiliesEdgeCases:
    """Tests for edge cases and special scenarios."""
    
    def test_quasipoisson_no_overdispersion(self):
        """When data is truly Poisson, dispersion should be close to 1."""
        np.random.seed(999)
        n = 1000
        
        x = np.random.randn(n)
        mu = np.exp(0.5 + 0.3 * x)
        y = np.random.poisson(mu).astype(float)
        
        X = np.column_stack([np.ones(n), x])
        
        result = rs.fit_glm(y, X, family="quasipoisson")
        
        # Dispersion should be close to 1 for true Poisson data
        assert 0.8 < result.scale() < 1.2
    
    def test_quasipoisson_with_offset(self):
        """QuasiPoisson should work with offset."""
        np.random.seed(42)
        n = 200
        
        x = np.random.randn(n)
        exposure = np.random.uniform(0.5, 2.0, n)
        mu = exposure * np.exp(0.5 + 0.3 * x)
        y = np.random.poisson(mu).astype(float)
        
        X = np.column_stack([np.ones(n), x])
        offset = np.log(exposure)
        
        result = rs.fit_glm(y, X, family="quasipoisson", offset=offset)
        
        assert result.converged
        assert len(result.params) == 2
    
    def test_quasipoisson_with_weights(self):
        """QuasiPoisson should work with weights."""
        np.random.seed(42)
        n = 200
        
        x = np.random.randn(n)
        mu = np.exp(0.5 + 0.3 * x)
        y = np.random.poisson(mu).astype(float)
        weights = np.random.uniform(0.5, 2.0, n)
        
        X = np.column_stack([np.ones(n), x])
        
        result = rs.fit_glm(y, X, family="quasipoisson", weights=weights)
        
        assert result.converged
    
    def test_quasibinomial_with_weights(self):
        """QuasiBinomial should work with weights."""
        np.random.seed(42)
        n = 200
        
        x = np.random.randn(n)
        p = 1 / (1 + np.exp(-(0.5 + 0.5 * x)))
        y = (np.random.rand(n) < p).astype(float)
        weights = np.random.uniform(0.5, 2.0, n)
        
        X = np.column_stack([np.ones(n), x])
        
        result = rs.fit_glm(y, X, family="quasibinomial", weights=weights)
        
        assert result.converged
    
    def test_dispersion_pearson_vs_deviance(self):
        """Compare Pearson-based and deviance-based dispersion estimates."""
        np.random.seed(42)
        n = 500
        
        x = np.random.randn(n)
        mu = np.exp(0.5 + 0.3 * x)
        # Add overdispersion
        y = np.random.negative_binomial(n=2, p=2/(2+mu))
        
        X = np.column_stack([np.ones(n), x])
        
        result = rs.fit_glm(y, X, family="quasipoisson")
        
        # scale() uses Pearson chi-squared
        phi_pearson = result.scale()
        
        # scale_pearson() should give the same value
        phi_pearson_direct = result.scale_pearson()
        
        # They should be close (scale() uses Pearson for quasi-families)
        np.testing.assert_allclose(phi_pearson, phi_pearson_direct, rtol=1e-6)


# =============================================================================
# Diagnostics and Statistics Tests
# =============================================================================

class TestQuasiFamiliesDiagnostics:
    """Tests for diagnostic methods with quasi-families."""
    
    def test_quasipoisson_pearson_chi2(self, overdispersed_count_data):
        """Pearson chi-squared should be computed correctly."""
        y, X = overdispersed_count_data
        
        result = rs.fit_glm(y, X, family="quasipoisson")
        
        chi2 = result.pearson_chi2()
        df_resid = result.df_resid
        dispersion = result.scale()
        
        # scale = chi2 / df_resid for quasi-families
        expected_dispersion = chi2 / df_resid
        np.testing.assert_allclose(dispersion, expected_dispersion, rtol=1e-10)
    
    def test_quasipoisson_aic_bic(self, overdispersed_count_data):
        """AIC/BIC should be computed."""
        y, X = overdispersed_count_data
        
        result = rs.fit_glm(y, X, family="quasipoisson")
        
        # Should return values (interpretation for quasi-families is different)
        aic = result.aic()
        bic = result.bic()
        
        assert np.isfinite(aic)
        assert np.isfinite(bic)
    
    def test_quasipoisson_null_deviance(self, overdispersed_count_data):
        """Null deviance should be computed."""
        y, X = overdispersed_count_data
        
        result = rs.fit_glm(y, X, family="quasipoisson")
        
        null_dev = result.null_deviance()
        resid_dev = result.deviance
        
        # Null deviance should be >= residual deviance
        assert null_dev >= resid_dev - 1e-10
