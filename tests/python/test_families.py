"""
Tests for Distribution Families
===============================

These tests verify that our family implementations are correct.
We test both the Rust implementations (via Python bindings) and
compare results with expected values.

Run with: pytest tests/python/test_families.py -v
"""

import numpy as np
import pytest

# Import our library
import rustystats as rs


class TestGaussianFamily:
    """Tests for Gaussian (Normal) family."""
    
    def test_variance_is_constant(self):
        """Gaussian variance function should return 1 for all μ values."""
        family = rs.families.Gaussian()
        mu = np.array([1.0, 10.0, 100.0, 0.001])
        
        variance = family.variance(mu)
        
        expected = np.ones_like(mu)
        np.testing.assert_array_almost_equal(variance, expected)
    
    def test_unit_deviance_is_squared_residual(self):
        """Gaussian unit deviance should be (y - μ)²."""
        family = rs.families.Gaussian()
        y = np.array([1.0, 2.0, 3.0])
        mu = np.array([1.5, 2.0, 2.5])
        
        unit_dev = family.unit_deviance(y, mu)
        
        expected = (y - mu) ** 2
        np.testing.assert_array_almost_equal(unit_dev, expected)
    
    def test_deviance_is_sum_of_squares(self):
        """Total deviance should be sum of squared residuals."""
        family = rs.families.Gaussian()
        y = np.array([1.0, 2.0, 3.0, 4.0])
        mu = np.array([1.1, 2.2, 2.8, 4.1])
        
        deviance = family.deviance(y, mu)
        
        expected = np.sum((y - mu) ** 2)
        np.testing.assert_almost_equal(deviance, expected)
    
    def test_default_link_is_identity(self):
        """Gaussian default link should be identity."""
        family = rs.families.Gaussian()
        link = family.default_link()
        
        assert link.name() == "identity"


class TestPoissonFamily:
    """Tests for Poisson family."""
    
    def test_variance_equals_mean(self):
        """Poisson variance function: V(μ) = μ."""
        family = rs.families.Poisson()
        mu = np.array([0.5, 1.0, 2.0, 10.0])
        
        variance = family.variance(mu)
        
        np.testing.assert_array_almost_equal(variance, mu)
    
    def test_unit_deviance_perfect_fit(self):
        """When y = μ, deviance should be zero."""
        family = rs.families.Poisson()
        y = np.array([1.0, 2.0, 5.0])
        mu = y.copy()
        
        unit_dev = family.unit_deviance(y, mu)
        
        np.testing.assert_array_almost_equal(unit_dev, np.zeros_like(y))
    
    def test_unit_deviance_with_zeros(self):
        """Test deviance calculation when y = 0."""
        family = rs.families.Poisson()
        y = np.array([0.0])
        mu = np.array([1.0])
        
        unit_dev = family.unit_deviance(y, mu)
        
        # When y=0: 2 × [0 - (0 - μ)] = 2μ
        expected = 2.0 * mu
        np.testing.assert_array_almost_equal(unit_dev, expected)
    
    def test_default_link_is_log(self):
        """Poisson default link should be log."""
        family = rs.families.Poisson()
        link = family.default_link()
        
        assert link.name() == "log"


class TestBinomialFamily:
    """Tests for Binomial family."""
    
    def test_variance_function(self):
        """Binomial variance: V(μ) = μ(1-μ)."""
        family = rs.families.Binomial()
        mu = np.array([0.2, 0.5, 0.8])
        
        variance = family.variance(mu)
        
        expected = mu * (1 - mu)
        np.testing.assert_array_almost_equal(variance, expected)
    
    def test_variance_is_symmetric(self):
        """V(p) should equal V(1-p)."""
        family = rs.families.Binomial()
        
        mu1 = np.array([0.3])
        mu2 = np.array([0.7])
        
        var1 = family.variance(mu1)
        var2 = family.variance(mu2)
        
        np.testing.assert_array_almost_equal(var1, var2)
    
    def test_variance_maximum_at_half(self):
        """Variance should be maximum at μ = 0.5."""
        family = rs.families.Binomial()
        mu = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        
        variance = family.variance(mu)
        
        max_idx = np.argmax(variance)
        assert max_idx == 2  # μ = 0.5
    
    def test_unit_deviance_perfect_fit(self):
        """When y = μ, deviance should be zero."""
        family = rs.families.Binomial()
        y = np.array([0.3, 0.5, 0.7])
        mu = y.copy()
        
        unit_dev = family.unit_deviance(y, mu)
        
        np.testing.assert_array_almost_equal(unit_dev, np.zeros_like(y))
    
    def test_default_link_is_logit(self):
        """Binomial default link should be logit."""
        family = rs.families.Binomial()
        link = family.default_link()
        
        assert link.name() == "logit"


class TestGammaFamily:
    """Tests for Gamma family."""
    
    def test_variance_is_mean_squared(self):
        """Gamma variance: V(μ) = μ²."""
        family = rs.families.Gamma()
        mu = np.array([10.0, 100.0, 1000.0])
        
        variance = family.variance(mu)
        
        expected = mu ** 2
        np.testing.assert_array_almost_equal(variance, expected)
    
    def test_constant_coefficient_of_variation(self):
        """
        Gamma has constant CV = √φ.
        
        The "unit CV²" (before dispersion scaling) should be 1.
        """
        family = rs.families.Gamma()
        mu = np.array([10.0, 100.0, 1000.0])
        
        variance = family.variance(mu)
        
        # Unit CV² = V(μ) / μ² = μ² / μ² = 1
        unit_cv_squared = variance / (mu ** 2)
        expected = np.ones_like(mu)
        np.testing.assert_array_almost_equal(unit_cv_squared, expected)
    
    def test_unit_deviance_perfect_fit(self):
        """When y = μ, deviance should be zero."""
        family = rs.families.Gamma()
        y = np.array([100.0, 500.0, 1000.0])
        mu = y.copy()
        
        unit_dev = family.unit_deviance(y, mu)
        
        np.testing.assert_array_almost_equal(unit_dev, np.zeros_like(y))
    
    def test_default_link_is_log(self):
        """Gamma default link should be log (not canonical inverse)."""
        family = rs.families.Gamma()
        link = family.default_link()
        
        assert link.name() == "log"


class TestFamilyComparisons:
    """
    Cross-family comparison tests.
    
    These tests highlight the differences between families and help
    users understand which family is appropriate for their data.
    """
    
    def test_variance_scaling_comparison(self):
        """
        Compare how variance scales with mean across families.
        
        This demonstrates the key difference in the heteroscedasticity
        assumptions of each family.
        """
        mu = np.array([1.0, 10.0, 100.0])
        
        gaussian_var = rs.families.Gaussian().variance(mu)
        poisson_var = rs.families.Poisson().variance(mu)
        gamma_var = rs.families.Gamma().variance(mu)
        
        # Gaussian: constant (all 1s)
        np.testing.assert_array_almost_equal(gaussian_var, [1.0, 1.0, 1.0])
        
        # Poisson: linear in μ
        np.testing.assert_array_almost_equal(poisson_var, [1.0, 10.0, 100.0])
        
        # Gamma: quadratic in μ
        np.testing.assert_array_almost_equal(gamma_var, [1.0, 100.0, 10000.0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
