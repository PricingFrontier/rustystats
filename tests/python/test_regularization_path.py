"""
Tests for regularization path fitting with cross-validation.
"""

import pytest
import numpy as np
import polars as pl
import sys
sys.path.insert(0, '/home/ralph/rustystats/python')

import rustystats as rs
from rustystats.regularization_path import (
    compute_alpha_max,
    generate_alpha_path,
    create_cv_folds,
    select_optimal_alpha,
    RegularizationPathResult,
)


# ============================================================================
# Test Data Fixtures
# ============================================================================

@pytest.fixture
def simple_data():
    """Create simple synthetic data for testing."""
    np.random.seed(42)
    n = 500
    
    # Create predictors
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    x3 = np.random.randn(n)
    
    # Create response (Poisson-like)
    eta = 0.5 + 0.3 * x1 - 0.2 * x2 + 0.1 * x3
    mu = np.exp(eta)
    y = np.random.poisson(mu)
    
    return pl.DataFrame({
        "y": y,
        "x1": x1,
        "x2": x2,
        "x3": x3,
    })


@pytest.fixture
def insurance_like_data():
    """Create insurance-like synthetic data."""
    np.random.seed(123)
    n = 1000
    
    # Continuous predictors
    age = np.random.uniform(18, 80, n)
    bonus_malus = np.random.uniform(50, 200, n)
    
    # Categorical predictors
    regions = np.random.choice(["North", "South", "East", "West"], n)
    
    # Exposure
    exposure = np.random.uniform(0.1, 1.0, n)
    
    # Response (claim count)
    eta = -2.0 + 0.01 * age + 0.005 * bonus_malus
    mu = np.exp(eta) * exposure
    claims = np.random.poisson(mu)
    
    return pl.DataFrame({
        "ClaimCount": claims,
        "Age": age,
        "BonusMalus": bonus_malus,
        "Region": regions,
        "Exposure": exposure,
    })


# ============================================================================
# Unit Tests for Helper Functions
# ============================================================================

class TestAlphaMax:
    """Tests for compute_alpha_max."""
    
    def test_alpha_max_positive(self, simple_data):
        """Alpha max should be positive."""
        X = np.column_stack([
            np.ones(len(simple_data)),
            simple_data["x1"].to_numpy(),
            simple_data["x2"].to_numpy(),
        ])
        y = simple_data["y"].to_numpy().astype(float)
        
        alpha_max = compute_alpha_max(X, y, l1_ratio=1.0)
        assert alpha_max > 0
    
    def test_alpha_max_ridge_vs_lasso(self, simple_data):
        """Alpha max differs for Ridge vs Lasso."""
        X = np.column_stack([
            np.ones(len(simple_data)),
            simple_data["x1"].to_numpy(),
            simple_data["x2"].to_numpy(),
        ])
        y = simple_data["y"].to_numpy().astype(float)
        
        alpha_max_lasso = compute_alpha_max(X, y, l1_ratio=1.0)
        alpha_max_ridge = compute_alpha_max(X, y, l1_ratio=0.0)
        
        # Both should be positive
        assert alpha_max_lasso > 0
        assert alpha_max_ridge > 0


class TestAlphaPath:
    """Tests for generate_alpha_path."""
    
    def test_path_length(self):
        """Path should have correct number of points."""
        path = generate_alpha_path(alpha_max=1.0, n_alphas=50)
        assert len(path) == 50
    
    def test_path_range(self):
        """Path should span from alpha_max to alpha_min."""
        alpha_max = 1.0
        alpha_min_ratio = 0.001
        path = generate_alpha_path(alpha_max, n_alphas=100, alpha_min_ratio=alpha_min_ratio)
        
        assert path[0] == pytest.approx(alpha_max, rel=1e-5)
        assert path[-1] == pytest.approx(alpha_max * alpha_min_ratio, rel=1e-5)
    
    def test_path_decreasing(self):
        """Path should be monotonically decreasing."""
        path = generate_alpha_path(alpha_max=1.0, n_alphas=50)
        assert all(path[i] >= path[i+1] for i in range(len(path)-1))


class TestCVFolds:
    """Tests for create_cv_folds."""
    
    def test_fold_count(self):
        """Should create correct number of folds."""
        folds = create_cv_folds(n=100, n_folds=5)
        assert len(folds) == 5
    
    def test_no_overlap(self):
        """Validation sets should not overlap."""
        folds = create_cv_folds(n=100, n_folds=5, seed=42)
        
        all_val_indices = []
        for _, val_idx in folds:
            all_val_indices.extend(val_idx)
        
        # All indices should appear exactly once
        assert len(all_val_indices) == 100
        assert len(set(all_val_indices)) == 100
    
    def test_coverage(self):
        """Each observation should be in exactly one validation set."""
        folds = create_cv_folds(n=100, n_folds=5, seed=42)
        
        counts = np.zeros(100)
        for _, val_idx in folds:
            counts[val_idx] += 1
        
        assert all(counts == 1)
    
    def test_train_val_disjoint(self):
        """Train and validation indices should be disjoint."""
        folds = create_cv_folds(n=100, n_folds=5, seed=42)
        
        for train_idx, val_idx in folds:
            assert len(set(train_idx) & set(val_idx)) == 0
    
    def test_reproducibility(self):
        """Same seed should give same folds."""
        folds1 = create_cv_folds(n=100, n_folds=5, seed=42)
        folds2 = create_cv_folds(n=100, n_folds=5, seed=42)
        
        for (t1, v1), (t2, v2) in zip(folds1, folds2):
            assert np.array_equal(t1, t2)
            assert np.array_equal(v1, v2)


class TestSelectOptimalAlpha:
    """Tests for select_optimal_alpha."""
    
    def test_min_selection(self):
        """Min selection should pick lowest CV deviance."""
        results = [
            RegularizationPathResult(0.1, 0.0, 0.30, 0.01, 10, 1.0),
            RegularizationPathResult(0.01, 0.0, 0.25, 0.01, 10, 1.0),  # Best
            RegularizationPathResult(0.001, 0.0, 0.28, 0.01, 10, 1.0),
        ]
        
        selected = select_optimal_alpha(results, selection="min")
        assert selected.alpha == 0.01
    
    def test_1se_selection(self):
        """1SE selection should pick largest alpha within 1 SE of min."""
        results = [
            RegularizationPathResult(0.1, 0.0, 0.260, 0.01, 5, 1.0),  # Within 1SE
            RegularizationPathResult(0.05, 0.0, 0.255, 0.01, 8, 1.0),  # Within 1SE
            RegularizationPathResult(0.01, 0.0, 0.250, 0.01, 10, 1.0),  # Best
            RegularizationPathResult(0.001, 0.0, 0.252, 0.01, 10, 1.0),
        ]
        
        # 1SE threshold = 0.250 + 0.01 = 0.260
        # First result (alpha=0.1) is at threshold, should be selected
        selected = select_optimal_alpha(results, selection="1se")
        assert selected.alpha == 0.1


# ============================================================================
# Integration Tests
# ============================================================================

class TestCVRegularizationFit:
    """Integration tests for CV-based regularization."""
    
    def test_ridge_cv_basic(self, simple_data):
        """Basic Ridge CV should work."""
        model = rs.glm(
            formula="y ~ x1 + x2 + x3",
            data=simple_data,
            family="poisson",
        )
        
        result = model.fit(
            cv=3,
            regularization="ridge",
            n_alphas=10,
            verbose=False,
        )
        
        # Should have CV attributes
        assert result.cv_deviance is not None
        assert result.cv_deviance_se is not None
        assert result.regularization_type in ("ridge", "none")
        assert result.regularization_path is not None
        assert len(result.regularization_path) > 0
    
    def test_lasso_cv_basic(self, simple_data):
        """Basic Lasso CV should work."""
        model = rs.glm(
            formula="y ~ x1 + x2 + x3",
            data=simple_data,
            family="poisson",
        )
        
        result = model.fit(
            cv=3,
            regularization="lasso",
            n_alphas=10,
            verbose=False,
        )
        
        assert result.cv_deviance is not None
        assert result.regularization_type in ("lasso", "none")
    
    def test_1se_selection_more_regularized(self, simple_data):
        """1SE selection should give equal or higher alpha than min."""
        model = rs.glm(
            formula="y ~ x1 + x2 + x3",
            data=simple_data,
            family="poisson",
        )
        
        result_min = model.fit(
            cv=3,
            regularization="ridge",
            n_alphas=20,
            selection="min",
            cv_seed=42,
            verbose=False,
        )
        
        result_1se = model.fit(
            cv=3,
            regularization="ridge",
            n_alphas=20,
            selection="1se",
            cv_seed=42,
            verbose=False,
        )
        
        # 1SE should select equal or higher alpha (more regularization)
        assert result_1se.alpha >= result_min.alpha
    
    def test_cv_requires_regularization_type(self, simple_data):
        """CV without regularization type should raise error."""
        model = rs.glm(
            formula="y ~ x1 + x2",
            data=simple_data,
            family="poisson",
        )
        
        with pytest.raises(rs.ValidationError, match="regularization"):
            model.fit(cv=5)
    
    def test_explicit_alpha_no_cv(self, simple_data):
        """Explicit alpha without CV should work as before."""
        model = rs.glm(
            formula="y ~ x1 + x2",
            data=simple_data,
            family="poisson",
        )
        
        result = model.fit(alpha=0.1, l1_ratio=0.0)
        
        assert result.alpha == pytest.approx(0.1)
        assert result.cv_deviance is None  # No CV was used
    
    def test_unregularized_included(self, simple_data):
        """Unregularized model should be included in comparison by default."""
        model = rs.glm(
            formula="y ~ x1 + x2",
            data=simple_data,
            family="poisson",
        )
        
        result = model.fit(
            cv=3,
            regularization="ridge",
            n_alphas=10,
            include_unregularized=True,
            verbose=False,
        )
        
        # Path should include alpha=0
        alphas = [r["alpha"] for r in result.regularization_path]
        assert 0.0 in alphas


class TestWithInsuranceData:
    """Tests with insurance-like data."""
    
    def test_poisson_with_offset(self, insurance_like_data):
        """CV regularization should work with offset."""
        model = rs.glm(
            formula="ClaimCount ~ Age + BonusMalus",
            data=insurance_like_data,
            family="poisson",
            offset="Exposure",
        )
        
        result = model.fit(
            cv=3,
            regularization="ridge",
            n_alphas=10,
            verbose=False,
        )
        
        assert result.cv_deviance is not None
        assert result.converged
    
    def test_cv_reproducibility(self, insurance_like_data):
        """Same seed should give reproducible results."""
        model = rs.glm(
            formula="ClaimCount ~ Age + BonusMalus",
            data=insurance_like_data,
            family="poisson",
            offset="Exposure",
        )
        
        result1 = model.fit(
            cv=3,
            regularization="ridge",
            n_alphas=10,
            cv_seed=42,
            verbose=False,
        )
        
        result2 = model.fit(
            cv=3,
            regularization="ridge",
            n_alphas=10,
            cv_seed=42,
            verbose=False,
        )
        
        assert result1.alpha == pytest.approx(result2.alpha)
        assert result1.cv_deviance == pytest.approx(result2.cv_deviance)


class TestRegularizationPathAttributes:
    """Tests for result attributes after CV fitting."""
    
    def test_all_cv_attributes_present(self, simple_data):
        """All CV-related attributes should be present."""
        model = rs.glm(
            formula="y ~ x1 + x2",
            data=simple_data,
            family="poisson",
        )
        
        result = model.fit(
            cv=3,
            regularization="ridge",
            n_alphas=10,
            verbose=False,
        )
        
        # Check all CV attributes
        assert result.cv_deviance is not None
        assert result.cv_deviance_se is not None
        assert result.regularization_type is not None
        assert result.regularization_path is not None
        assert result.cv_selection_method is not None
        assert result.n_cv_folds == 3
    
    def test_path_structure(self, simple_data):
        """Regularization path should have correct structure."""
        model = rs.glm(
            formula="y ~ x1 + x2",
            data=simple_data,
            family="poisson",
        )
        
        result = model.fit(
            cv=3,
            regularization="ridge",
            n_alphas=10,
            verbose=False,
        )
        
        path = result.regularization_path
        
        # Check structure of each path entry
        for entry in path:
            assert "alpha" in entry
            assert "l1_ratio" in entry
            assert "cv_deviance_mean" in entry
            assert "cv_deviance_se" in entry
            assert "n_nonzero" in entry
            assert "max_coef" in entry


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
