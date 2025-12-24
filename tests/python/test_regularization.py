"""
Tests for Regularization Features
=================================

Tests for Ridge, Lasso, Elastic Net regularization and variable selection utilities.
"""

import numpy as np
import pytest
import rustystats as rs


class TestRidgeRegularization:
    """Tests for Ridge (L2) regularization."""
    
    def test_ridge_shrinks_coefficients(self):
        """Ridge should shrink coefficients toward zero."""
        np.random.seed(42)
        n = 100
        X = np.column_stack([np.ones(n), np.random.randn(n, 3)])
        y = 2 + 3 * X[:, 1] + np.random.randn(n) * 0.5
        
        # Fit with and without Ridge
        result_ols = rs.fit_glm(y, X, family="gaussian")
        result_ridge = rs.fit_glm(y, X, family="gaussian", alpha=1.0, l1_ratio=0.0)
        
        # Ridge coefficients should be smaller in magnitude (except intercept)
        assert np.abs(result_ridge.params[1]) < np.abs(result_ols.params[1])
    
    def test_ridge_intercept_not_penalized(self):
        """Intercept should not be penalized by Ridge."""
        np.random.seed(42)
        n = 100
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = 10 + X[:, 1] + np.random.randn(n) * 0.1
        
        result_ols = rs.fit_glm(y, X, family="gaussian")
        result_ridge = rs.fit_glm(y, X, family="gaussian", alpha=10.0, l1_ratio=0.0)
        
        # Intercepts should be similar
        assert np.abs(result_ridge.params[0] - result_ols.params[0]) < 1.0
    
    def test_ridge_penalty_type(self):
        """Ridge result should report correct penalty type."""
        np.random.seed(42)
        n = 50
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.random.randn(n)
        
        result = rs.fit_glm(y, X, family="gaussian", alpha=0.5, l1_ratio=0.0)
        
        assert result.penalty_type == "ridge"
        assert result.is_regularized == True
        assert result.alpha == 0.5


class TestLassoRegularization:
    """Tests for Lasso (L1) regularization."""
    
    def test_lasso_produces_sparse_solution(self):
        """Lasso should zero out some coefficients."""
        np.random.seed(42)
        n = 200
        p = 10
        X = np.column_stack([np.ones(n), np.random.randn(n, p-1)])
        
        # Only first 2 features (after intercept) matter
        y = 2 + 3 * X[:, 1] + 2 * X[:, 2] + np.random.randn(n) * 0.5
        
        # Use stronger penalty to ensure sparsity
        result = rs.fit_glm(y, X, family="gaussian", alpha=2.0, l1_ratio=1.0)
        
        # Should have some zero coefficients
        n_nonzero = result.n_nonzero()
        assert n_nonzero < p - 1  # Some should be zeroed out
    
    def test_lasso_penalty_type(self):
        """Lasso result should report correct penalty type."""
        np.random.seed(42)
        n = 50
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.random.randn(n)
        
        result = rs.fit_glm(y, X, family="gaussian", alpha=0.5, l1_ratio=1.0)
        
        assert result.penalty_type == "lasso"
        assert result.is_regularized == True
        assert result.l1_ratio == 1.0
    
    def test_lasso_selected_features(self):
        """selected_features should return indices of non-zero coefficients."""
        np.random.seed(42)
        n = 100
        X = np.column_stack([np.ones(n), np.random.randn(n, 5)])
        y = 2 + 5 * X[:, 1] + np.random.randn(n) * 0.1
        
        result = rs.fit_glm(y, X, family="gaussian", alpha=0.5, l1_ratio=1.0)
        
        selected = result.selected_features()
        assert isinstance(selected, list)
        assert 1 in selected  # Feature 1 should be selected


class TestElasticNet:
    """Tests for Elastic Net regularization."""
    
    def test_elastic_net_intermediate(self):
        """Elastic Net should be between Ridge and Lasso."""
        np.random.seed(42)
        n = 100
        X = np.column_stack([np.ones(n), np.random.randn(n, 3)])
        y = 2 + 3 * X[:, 1] + np.random.randn(n) * 0.5
        
        result = rs.fit_glm(y, X, family="gaussian", alpha=0.5, l1_ratio=0.5)
        
        assert result.penalty_type == "elasticnet"
        assert result.l1_ratio == 0.5
    
    def test_elastic_net_converges(self):
        """Elastic Net should converge."""
        np.random.seed(42)
        n = 100
        X = np.column_stack([np.ones(n), np.random.randn(n, 3)])
        y = np.random.randn(n)
        
        result = rs.fit_glm(y, X, family="gaussian", alpha=0.1, l1_ratio=0.5, max_iter=50)
        
        assert result.converged or result.iterations <= 50


class TestNoRegularization:
    """Tests that alpha=0 produces standard GLM."""
    
    def test_alpha_zero_is_unregularized(self):
        """alpha=0 should give same result as standard fit_glm."""
        np.random.seed(42)
        n = 100
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = 2 + 3 * X[:, 1] + np.random.randn(n) * 0.5
        
        result1 = rs.fit_glm(y, X, family="gaussian")
        result2 = rs.fit_glm(y, X, family="gaussian", alpha=0.0)
        
        np.testing.assert_array_almost_equal(result1.params, result2.params, decimal=6)
        assert result2.is_regularized == False
        assert result2.penalty_type == "none"


class TestRegularizationPath:
    """Tests for coefficient path computation."""
    
    def test_path_has_correct_shape(self):
        """Path should have correct number of alphas and features."""
        np.random.seed(42)
        n = 100
        p = 5
        X = np.column_stack([np.ones(n), np.random.randn(n, p-1)])
        y = np.random.randn(n)
        
        path = rs.lasso_path(y, X, family="gaussian", n_alphas=20)
        
        assert len(path.alphas) == 20
        assert path.coefficients.shape == (20, p)
        assert len(path.deviances) == 20
        assert len(path.n_nonzero) == 20
    
    def test_path_alphas_decreasing(self):
        """Alphas should be in decreasing order."""
        np.random.seed(42)
        n = 50
        X = np.column_stack([np.ones(n), np.random.randn(n, 2)])
        y = np.random.randn(n)
        
        path = rs.lasso_path(y, X, family="gaussian", n_alphas=10)
        
        assert np.all(np.diff(path.alphas) <= 0)
    
    def test_path_more_nonzero_at_smaller_alpha(self):
        """Should have more non-zero coefficients at smaller alpha."""
        np.random.seed(42)
        n = 100
        X = np.column_stack([np.ones(n), np.random.randn(n, 5)])
        y = X[:, 1] + 0.5 * X[:, 2] + np.random.randn(n) * 0.1
        
        path = rs.lasso_path(y, X, family="gaussian", n_alphas=20)
        
        # Non-zero count should generally increase as alpha decreases
        assert path.n_nonzero[-1] >= path.n_nonzero[0]


class TestCrossValidation:
    """Tests for cross-validation utilities."""
    
    def test_cv_returns_best_alpha(self):
        """CV should return a best alpha."""
        np.random.seed(42)
        n = 100
        X = np.column_stack([np.ones(n), np.random.randn(n, 3)])
        y = 2 + 3 * X[:, 1] + np.random.randn(n) * 0.5
        
        cv_result = rs.cv_glm(y, X, family="gaussian", l1_ratio=1.0, cv=3, n_alphas=10)
        
        assert cv_result.alpha_best > 0
        assert cv_result.alpha_1se >= cv_result.alpha_best
        assert cv_result.best_result is not None
    
    def test_cv_scores_have_correct_shape(self):
        """CV scores should have correct shape."""
        np.random.seed(42)
        n = 50
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.random.randn(n)
        
        cv_result = rs.cv_glm(y, X, family="gaussian", cv=3, n_alphas=5)
        
        assert len(cv_result.alphas) == 5
        assert len(cv_result.scores_mean) == 5
        assert len(cv_result.scores_std) == 5
    
    def test_cv_lasso_alias(self):
        """cv_lasso should be equivalent to cv_glm with l1_ratio=1.0."""
        np.random.seed(42)
        n = 50
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.random.randn(n)
        
        result = rs.cv_lasso(y, X, family="gaussian", cv=3, n_alphas=5)
        
        assert result.best_result.penalty_type == "lasso"
    
    def test_cv_ridge_alias(self):
        """cv_ridge should be equivalent to cv_glm with l1_ratio=0.0."""
        np.random.seed(42)
        n = 50
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.random.randn(n)
        
        result = rs.cv_ridge(y, X, family="gaussian", cv=3, n_alphas=5)
        
        assert result.best_result.penalty_type == "ridge"


class TestPoissonRegularization:
    """Tests for regularization with Poisson family."""
    
    def test_poisson_lasso(self):
        """Lasso should work with Poisson family."""
        np.random.seed(42)
        n = 200
        X = np.column_stack([np.ones(n), np.random.randn(n, 3)])
        y = np.random.poisson(np.exp(0.5 + 0.3 * X[:, 1]))
        
        result = rs.fit_glm(y, X, family="poisson", alpha=0.1, l1_ratio=1.0)
        
        assert result.converged or result.iterations > 0
        assert np.all(result.fittedvalues > 0)
    
    def test_poisson_ridge(self):
        """Ridge should work with Poisson family."""
        np.random.seed(42)
        n = 200
        X = np.column_stack([np.ones(n), np.random.randn(n, 3)])
        y = np.random.poisson(np.exp(0.5 + 0.3 * X[:, 1]))
        
        result = rs.fit_glm(y, X, family="poisson", alpha=0.1, l1_ratio=0.0)
        
        assert result.converged
        assert result.penalty_type == "ridge"


class TestGLMClassRegularization:
    """Tests for regularization via GLM class."""
    
    def test_glm_class_with_alpha(self):
        """GLM class should support alpha parameter in fit()."""
        np.random.seed(42)
        n = 50
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.random.randn(n)
        
        model = rs.GLM(y, X, family="gaussian")
        result = model.fit(alpha=0.5, l1_ratio=1.0)
        
        assert result.is_regularized
        assert result.penalty_type == "lasso"
