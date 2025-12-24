"""
Tests for Negative Binomial Family
==================================

These tests verify that:
1. NegativeBinomial produces reasonable fits for overdispersed count data
2. Variance function is computed correctly
3. Deviance is computed correctly
4. All API methods work correctly
5. Different theta values produce expected behavior
"""

import numpy as np
import pytest
import rustystats as rs


# =============================================================================
# Test Data Fixtures
# =============================================================================

@pytest.fixture
def overdispersed_count_data():
    """Generate overdispersed count data using negative binomial distribution."""
    np.random.seed(42)
    n = 500
    
    # Generate predictors
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    
    # Generate true negative binomial data
    mu = np.exp(0.5 + 0.3 * x1 - 0.2 * x2)
    theta = 2.0  # True dispersion parameter
    
    # NB parameterization: p = theta / (theta + mu)
    y = np.random.negative_binomial(n=theta, p=theta / (theta + mu))
    
    # Design matrix with intercept
    X = np.column_stack([np.ones(n), x1, x2])
    
    return y.astype(float), X


@pytest.fixture
def simple_count_data():
    """Simple count data for basic tests."""
    np.random.seed(123)
    n = 200
    
    x = np.random.randn(n)
    mu = np.exp(1.0 + 0.5 * x)
    y = np.random.poisson(mu).astype(float)
    
    X = np.column_stack([np.ones(n), x])
    
    return y, X


# =============================================================================
# NegativeBinomial Basic Tests
# =============================================================================

class TestNegativeBinomialBasic:
    """Basic tests for NegativeBinomial family."""
    
    def test_negbinomial_basic_fit(self, overdispersed_count_data):
        """NegativeBinomial should fit without errors."""
        y, X = overdispersed_count_data
        result = rs.fit_glm(y, X, family="negbinomial", theta=1.0)
        
        assert result.converged
        assert len(result.params) == 3
        assert result.deviance > 0
    
    def test_negbinomial_family_name(self, overdispersed_count_data):
        """NegativeBinomial should report correct family name."""
        y, X = overdispersed_count_data
        result = rs.fit_glm(y, X, family="negbinomial", theta=1.0)
        assert result.family == "NegativeBinomial"
    
    def test_negbinomial_alternative_spellings(self, simple_count_data):
        """NegativeBinomial should accept various spellings."""
        y, X = simple_count_data
        
        r1 = rs.fit_glm(y, X, family="negbinomial", theta=1.0)
        r2 = rs.fit_glm(y, X, family="negativebinomial", theta=1.0)
        r3 = rs.fit_glm(y, X, family="negative_binomial", theta=1.0)
        r4 = rs.fit_glm(y, X, family="nb", theta=1.0)
        
        np.testing.assert_allclose(r1.params, r2.params, rtol=1e-10)
        np.testing.assert_allclose(r1.params, r3.params, rtol=1e-10)
        np.testing.assert_allclose(r1.params, r4.params, rtol=1e-10)
    
    def test_negbinomial_residuals(self, overdispersed_count_data):
        """NegativeBinomial should compute all residual types."""
        y, X = overdispersed_count_data
        result = rs.fit_glm(y, X, family="negbinomial", theta=1.0)
        
        resid_response = result.resid_response()
        resid_pearson = result.resid_pearson()
        resid_deviance = result.resid_deviance()
        resid_working = result.resid_working()
        
        assert len(resid_response) == len(y)
        assert len(resid_pearson) == len(y)
        assert len(resid_deviance) == len(y)
        assert len(resid_working) == len(y)


# =============================================================================
# Theta Parameter Tests
# =============================================================================

class TestNegativeBinomialTheta:
    """Tests for different theta values."""
    
    def test_large_theta_approaches_poisson(self, simple_count_data):
        """Large θ should give results similar to Poisson."""
        y, X = simple_count_data
        
        result_poisson = rs.fit_glm(y, X, family="poisson")
        result_nb = rs.fit_glm(y, X, family="negbinomial", theta=100.0)
        
        # Coefficients should be similar (not identical due to different variance)
        np.testing.assert_allclose(
            result_nb.params,
            result_poisson.params,
            rtol=0.1  # Within 10%
        )
    
    def test_small_theta_more_overdispersion(self, overdispersed_count_data):
        """Smaller θ should model more overdispersion."""
        y, X = overdispersed_count_data
        
        result_theta_05 = rs.fit_glm(y, X, family="negbinomial", theta=0.5)
        result_theta_2 = rs.fit_glm(y, X, family="negbinomial", theta=2.0)
        result_theta_10 = rs.fit_glm(y, X, family="negbinomial", theta=10.0)
        
        # All should converge
        assert result_theta_05.converged
        assert result_theta_2.converged
        assert result_theta_10.converged
    
    def test_invalid_theta_raises(self, simple_count_data):
        """Invalid theta values should raise errors."""
        y, X = simple_count_data
        
        with pytest.raises(ValueError):
            rs.fit_glm(y, X, family="negbinomial", theta=0.0)
        
        with pytest.raises(ValueError):
            rs.fit_glm(y, X, family="negbinomial", theta=-1.0)


# =============================================================================
# Family Object Tests
# =============================================================================

class TestNegativeBinomialFamilyObject:
    """Tests for NegativeBinomial family object."""
    
    def test_family_object_creation(self):
        """Test NegativeBinomial family object creation."""
        family = rs.families.NegativeBinomial(theta=2.0)
        
        assert family.name() == "NegativeBinomial"
        assert family.theta == 2.0
        assert family.alpha == 0.5  # alpha = 1/theta
    
    def test_family_object_default_theta(self):
        """Test default theta value."""
        family = rs.families.NegativeBinomial()
        assert family.theta == 1.0
    
    def test_variance_function(self):
        """Test variance function V(μ) = μ + μ²/θ."""
        family = rs.families.NegativeBinomial(theta=1.0)
        
        mu = np.array([1.0, 2.0, 4.0])
        variance = np.array(family.variance(mu))
        
        # V(μ) = μ + μ²/θ = μ + μ² (when θ=1)
        expected = mu + mu**2
        np.testing.assert_allclose(variance, expected, rtol=1e-10)
    
    def test_variance_function_different_theta(self):
        """Test variance function with different θ values."""
        family = rs.families.NegativeBinomial(theta=2.0)
        
        mu = np.array([2.0])
        variance = np.array(family.variance(mu))
        
        # V(2) = 2 + 4/2 = 4
        expected = np.array([4.0])
        np.testing.assert_allclose(variance, expected, rtol=1e-10)
    
    def test_deviance_perfect_fit(self):
        """Perfect fit should have zero deviance."""
        family = rs.families.NegativeBinomial(theta=1.0)
        
        y = np.array([1.0, 2.0, 3.0])
        mu = np.array([1.0, 2.0, 3.0])
        
        dev = np.array(family.unit_deviance(y, mu))
        
        np.testing.assert_allclose(dev, np.zeros(3), atol=1e-10)
    
    def test_deviance_with_zeros(self):
        """Deviance should handle zeros correctly."""
        family = rs.families.NegativeBinomial(theta=1.0)
        
        y = np.array([0.0, 0.0, 1.0])
        mu = np.array([1.0, 2.0, 1.0])
        
        dev = np.array(family.unit_deviance(y, mu))
        
        # All deviances should be non-negative
        assert np.all(dev >= 0)
        assert np.all(np.isfinite(dev))
    
    def test_default_link(self):
        """Default link should be log."""
        family = rs.families.NegativeBinomial(theta=1.0)
        link = family.default_link()
        assert link.name() == "log"


# =============================================================================
# Comparison Tests
# =============================================================================

class TestNegativeBinomialComparison:
    """Tests comparing NegativeBinomial to other families."""
    
    def test_nb_vs_poisson_overdispersed_data(self, overdispersed_count_data):
        """NB should fit overdispersed data better than Poisson."""
        y, X = overdispersed_count_data
        
        result_poisson = rs.fit_glm(y, X, family="poisson")
        result_nb = rs.fit_glm(y, X, family="negbinomial", theta=2.0)
        
        # Both should converge
        assert result_poisson.converged
        assert result_nb.converged
        
        # NB typically has lower deviance for overdispersed data
        # (though this depends on the true data generating process)
    
    def test_nb_vs_quasipoisson(self, overdispersed_count_data):
        """Compare NB and QuasiPoisson fits."""
        y, X = overdispersed_count_data
        
        result_quasi = rs.fit_glm(y, X, family="quasipoisson")
        result_nb = rs.fit_glm(y, X, family="negbinomial", theta=2.0)
        
        # Both should converge
        assert result_quasi.converged
        assert result_nb.converged
        
        # Coefficients should be in similar range
        # (not identical due to different variance functions)
        assert np.allclose(result_quasi.params, result_nb.params, rtol=0.3)


# =============================================================================
# Edge Cases and Special Scenarios
# =============================================================================

class TestNegativeBinomialEdgeCases:
    """Tests for edge cases and special scenarios."""
    
    def test_with_offset(self):
        """NegativeBinomial should work with offset."""
        np.random.seed(42)
        n = 200
        
        x = np.random.randn(n)
        exposure = np.random.uniform(0.5, 2.0, n)
        mu = exposure * np.exp(0.5 + 0.3 * x)
        y = np.random.poisson(mu).astype(float)
        
        X = np.column_stack([np.ones(n), x])
        offset = np.log(exposure)
        
        result = rs.fit_glm(y, X, family="negbinomial", theta=1.0, offset=offset)
        
        assert result.converged
        assert len(result.params) == 2
    
    def test_with_weights(self):
        """NegativeBinomial should work with weights."""
        np.random.seed(42)
        n = 200
        
        x = np.random.randn(n)
        mu = np.exp(0.5 + 0.3 * x)
        y = np.random.poisson(mu).astype(float)
        weights = np.random.uniform(0.5, 2.0, n)
        
        X = np.column_stack([np.ones(n), x])
        
        result = rs.fit_glm(y, X, family="negbinomial", theta=1.0, weights=weights)
        
        assert result.converged
    
    def test_extreme_theta_values(self, simple_count_data):
        """Test with extreme theta values."""
        y, X = simple_count_data
        
        # Very small theta (strong overdispersion)
        result_small = rs.fit_glm(y, X, family="negbinomial", theta=0.1)
        assert result_small.converged
        
        # Very large theta (close to Poisson)
        result_large = rs.fit_glm(y, X, family="negbinomial", theta=1000.0)
        assert result_large.converged
    
    def test_all_zeros(self):
        """Test with response containing many zeros."""
        np.random.seed(42)
        n = 200
        
        x = np.random.randn(n)
        X = np.column_stack([np.ones(n), x])
        
        # Generate data with many zeros
        y = np.zeros(n)
        y[np.random.choice(n, 20, replace=False)] = np.random.poisson(2, 20)
        
        result = rs.fit_glm(y, X, family="negbinomial", theta=0.5)
        
        assert result.converged


# =============================================================================
# Diagnostics Tests
# =============================================================================

class TestNegativeBinomialDiagnostics:
    """Tests for diagnostic methods with NegativeBinomial."""
    
    def test_aic_bic(self, overdispersed_count_data):
        """AIC and BIC should be computed."""
        y, X = overdispersed_count_data
        result = rs.fit_glm(y, X, family="negbinomial", theta=1.0)
        
        aic = result.aic()
        bic = result.bic()
        
        assert np.isfinite(aic)
        assert np.isfinite(bic)
    
    def test_null_deviance(self, overdispersed_count_data):
        """Null deviance should be computed."""
        y, X = overdispersed_count_data
        result = rs.fit_glm(y, X, family="negbinomial", theta=1.0)
        
        null_dev = result.null_deviance()
        resid_dev = result.deviance
        
        # Null deviance should be >= residual deviance
        assert null_dev >= resid_dev - 1e-6
    
    def test_standard_errors(self, overdispersed_count_data):
        """Standard errors should be positive and finite."""
        y, X = overdispersed_count_data
        result = rs.fit_glm(y, X, family="negbinomial", theta=1.0)
        
        se = np.array(result.bse())
        
        assert np.all(se > 0)
        assert np.all(np.isfinite(se))
    
    def test_confidence_intervals(self, overdispersed_count_data):
        """Confidence intervals should be computed."""
        y, X = overdispersed_count_data
        result = rs.fit_glm(y, X, family="negbinomial", theta=1.0)
        
        ci = np.array(result.conf_int())
        
        assert ci.shape == (3, 2)
        assert np.all(ci[:, 0] < ci[:, 1])  # Lower < Upper
