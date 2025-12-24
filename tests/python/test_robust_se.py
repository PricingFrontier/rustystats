"""
Tests for robust standard errors (HC0-HC3 sandwich estimators).

These tests verify that the robust standard error implementation:
1. Returns valid (finite, positive) standard errors
2. Different HC types give different results (HC0 < HC1 < HC2/HC3 typically)
3. Robust SEs are generally larger than model-based SEs when there's heteroscedasticity
4. Works with different families (Gaussian, Poisson, Binomial, Gamma)
"""

import numpy as np
import pytest
import rustystats as rs


class TestRobustSEBasic:
    """Basic tests for robust standard errors."""
    
    @pytest.fixture
    def gaussian_model(self):
        """Simple Gaussian model for testing."""
        np.random.seed(42)
        n = 100
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = X @ np.array([1.0, 0.5]) + np.random.randn(n)
        return rs.fit_glm(y, X, family="gaussian")
    
    @pytest.fixture
    def poisson_model(self):
        """Simple Poisson model for testing."""
        np.random.seed(42)
        n = 200
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        eta = X @ np.array([1.0, 0.3])
        mu = np.exp(eta)
        y = np.random.poisson(mu)
        return rs.fit_glm(y, X, family="poisson")
    
    def test_bse_robust_returns_array(self, gaussian_model):
        """bse_robust should return a numpy array."""
        se = gaussian_model.bse_robust()
        assert isinstance(se, np.ndarray)
        assert len(se) == 2  # intercept + 1 predictor
    
    def test_bse_robust_positive(self, gaussian_model):
        """Robust standard errors should be positive."""
        se = gaussian_model.bse_robust()
        assert np.all(se > 0)
    
    def test_bse_robust_finite(self, gaussian_model):
        """Robust standard errors should be finite."""
        se = gaussian_model.bse_robust()
        assert np.all(np.isfinite(se))
    
    def test_hc_types_work(self, gaussian_model):
        """All HC types should work without error."""
        for hc_type in ["HC0", "HC1", "HC2", "HC3"]:
            se = gaussian_model.bse_robust(hc_type)
            assert np.all(se > 0)
            assert np.all(np.isfinite(se))
    
    def test_hc_types_case_insensitive(self, gaussian_model):
        """HC types should be case-insensitive."""
        se_lower = gaussian_model.bse_robust("hc1")
        se_upper = gaussian_model.bse_robust("HC1")
        np.testing.assert_array_equal(se_lower, se_upper)
    
    def test_invalid_hc_type_raises(self, gaussian_model):
        """Invalid HC type should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown cov_type"):
            gaussian_model.bse_robust("HC4")
    
    def test_hc1_larger_than_hc0(self, gaussian_model):
        """HC1 should give larger SE than HC0 (degrees of freedom correction)."""
        se_hc0 = gaussian_model.bse_robust("HC0")
        se_hc1 = gaussian_model.bse_robust("HC1")
        # HC1 multiplies by n/(n-p), so should be larger
        assert np.all(se_hc1 >= se_hc0)


class TestRobustSEInference:
    """Tests for robust inference methods."""
    
    @pytest.fixture
    def model(self):
        """Model for testing inference."""
        np.random.seed(123)
        n = 150
        X = np.column_stack([np.ones(n), np.random.randn(n), np.random.randn(n)])
        y = X @ np.array([2.0, 0.5, -0.3]) + np.random.randn(n)
        return rs.fit_glm(y, X, family="gaussian")
    
    def test_tvalues_robust(self, model):
        """tvalues_robust should return z-statistics."""
        t = model.tvalues_robust()
        assert len(t) == 3
        assert np.all(np.isfinite(t))
        
        # t-values should equal coef / robust_se
        se = model.bse_robust()
        expected_t = model.params / se
        np.testing.assert_array_almost_equal(t, expected_t)
    
    def test_pvalues_robust(self, model):
        """pvalues_robust should return valid p-values."""
        p = model.pvalues_robust()
        assert len(p) == 3
        # P-values should be between 0 and 1
        assert np.all(p >= 0)
        assert np.all(p <= 1)
    
    def test_conf_int_robust(self, model):
        """conf_int_robust should return confidence intervals."""
        ci = model.conf_int_robust()
        assert ci.shape == (3, 2)
        
        # Lower bound should be less than upper bound
        assert np.all(ci[:, 0] < ci[:, 1])
        
        # Coefficient should be within CI
        coefs = model.params
        assert np.all(coefs >= ci[:, 0])
        assert np.all(coefs <= ci[:, 1])
    
    def test_conf_int_robust_alpha(self, model):
        """Different alpha should give different CI widths."""
        ci_95 = model.conf_int_robust(alpha=0.05)
        ci_99 = model.conf_int_robust(alpha=0.01)
        
        # 99% CI should be wider than 95% CI
        width_95 = ci_95[:, 1] - ci_95[:, 0]
        width_99 = ci_99[:, 1] - ci_99[:, 0]
        assert np.all(width_99 > width_95)
    
    def test_cov_robust(self, model):
        """cov_robust should return a covariance matrix."""
        cov = model.cov_robust()
        assert cov.shape == (3, 3)
        
        # Should be symmetric
        np.testing.assert_array_almost_equal(cov, cov.T)
        
        # Diagonal should be non-negative (variances)
        assert np.all(np.diag(cov) >= 0)


class TestRobustSEWithHeteroscedasticity:
    """Test robust SE with heteroscedastic data."""
    
    def test_robust_vs_model_based_with_heteroscedasticity(self):
        """Robust SE should differ from model-based when there's heteroscedasticity."""
        np.random.seed(42)
        n = 300
        x = np.random.randn(n)
        X = np.column_stack([np.ones(n), x])
        
        # Heteroscedastic errors: variance increases with x
        errors = np.random.randn(n) * (1 + np.abs(x))
        y = 1.0 + 0.5 * x + errors
        
        result = rs.fit_glm(y, X, family="gaussian")
        
        se_model = result.bse()
        se_robust = result.bse_robust("HC1")
        
        # Robust and model-based should differ
        assert not np.allclose(se_model, se_robust)
        
        # Both should be positive and finite
        assert np.all(se_model > 0) and np.all(np.isfinite(se_model))
        assert np.all(se_robust > 0) and np.all(np.isfinite(se_robust))


class TestRobustSEDifferentFamilies:
    """Test robust SE works with all families."""
    
    def test_gaussian(self):
        """Robust SE should work with Gaussian family."""
        np.random.seed(42)
        n = 100
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = X @ np.array([1.0, 0.5]) + np.random.randn(n)
        
        result = rs.fit_glm(y, X, family="gaussian")
        se = result.bse_robust("HC1")
        assert np.all(se > 0) and np.all(np.isfinite(se))
    
    def test_poisson(self):
        """Robust SE should work with Poisson family."""
        np.random.seed(42)
        n = 200
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        mu = np.exp(X @ np.array([1.0, 0.3]))
        y = np.random.poisson(mu)
        
        result = rs.fit_glm(y, X, family="poisson")
        se = result.bse_robust("HC1")
        assert np.all(se > 0) and np.all(np.isfinite(se))
    
    def test_binomial(self):
        """Robust SE should work with Binomial family."""
        np.random.seed(42)
        n = 200
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        eta = X @ np.array([0.0, 0.5])
        p = 1 / (1 + np.exp(-eta))
        y = np.random.binomial(1, p)
        
        result = rs.fit_glm(y, X, family="binomial")
        se = result.bse_robust("HC1")
        assert np.all(se > 0) and np.all(np.isfinite(se))
    
    def test_gamma(self):
        """Robust SE should work with Gamma family."""
        np.random.seed(42)
        n = 200
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        mu = np.exp(X @ np.array([2.0, 0.3]))
        y = np.random.gamma(shape=2, scale=mu/2)
        
        result = rs.fit_glm(y, X, family="gamma")
        se = result.bse_robust("HC1")
        assert np.all(se > 0) and np.all(np.isfinite(se))


class TestRobustSEFormulaAPI:
    """Test robust SE with formula API."""
    
    @pytest.fixture
    def formula_model(self):
        """Model fitted with formula API."""
        import polars as pl
        
        np.random.seed(42)
        n = 100
        data = pl.DataFrame({
            "y": np.random.randn(n) + 2,
            "x1": np.random.randn(n),
            "x2": np.random.randn(n),
        })
        
        return rs.glm("y ~ x1 + x2", data=data, family="gaussian").fit()
    
    def test_bse_robust_formula(self, formula_model):
        """bse_robust should work with formula API."""
        se = formula_model.bse_robust("HC1")
        assert len(se) == 3  # intercept + 2 predictors
        assert np.all(se > 0)
    
    def test_all_robust_methods_formula(self, formula_model):
        """All robust methods should work with formula API."""
        assert formula_model.bse_robust() is not None
        assert formula_model.tvalues_robust() is not None
        assert formula_model.pvalues_robust() is not None
        assert formula_model.conf_int_robust() is not None
        assert formula_model.cov_robust() is not None
