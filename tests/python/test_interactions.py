"""
Tests for interaction term support in RustyStats.

Tests cover:
- Continuous × Continuous interactions
- Categorical × Continuous interactions
- Categorical × Categorical interactions
- Pure interaction terms
- InteractionTerm dataclass properties
- Design matrix construction via dict API
"""

import numpy as np
import polars as pl
import pytest

import rustystats as rs
from rustystats.interactions import (
    InteractionBuilder,
    InteractionTerm,
    ParsedFormula,
)
from rustystats.splines import SplineTerm


# =============================================================================
# Design Matrix Construction Tests
# =============================================================================

class TestInteractionBuilder:
    """Test the InteractionBuilder class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample test data."""
        np.random.seed(42)
        n = 100
        return pl.DataFrame({
            'y': np.random.poisson(1, n),
            'x1': np.random.uniform(0, 10, n),
            'x2': np.random.uniform(0, 10, n),
            'cat1': np.random.choice(['A', 'B', 'C'], n),
            'cat2': np.random.choice(['X', 'Y'], n),
        })
    
    def test_continuous_continuous(self, sample_data):
        """Test continuous × continuous interaction."""
        parsed = ParsedFormula(
            response="y",
            main_effects=["x1", "x2"],
            interactions=[InteractionTerm(factors=["x1", "x2"], categorical_flags=[False, False])],
            categorical_vars=set(),
            has_intercept=True,
        )
        builder = InteractionBuilder(sample_data)
        y, X, names = builder.build_design_matrix_from_parsed(parsed)
        
        assert 'Intercept' in names
        assert 'x1' in names
        assert 'x2' in names
        assert 'x1:x2' in names
        assert X.shape[1] == 4
        
        # Verify interaction column is product of x1 and x2
        x1_col = names.index('x1')
        x2_col = names.index('x2')
        int_col = names.index('x1:x2')
        np.testing.assert_allclose(X[:, int_col], X[:, x1_col] * X[:, x2_col])
    
    def test_categorical_continuous(self, sample_data):
        """Test categorical × continuous interaction."""
        parsed = ParsedFormula(
            response="y",
            main_effects=["cat1", "x1"],
            interactions=[InteractionTerm(factors=["cat1", "x1"], categorical_flags=[True, False])],
            categorical_vars={"cat1"},
            has_intercept=True,
        )
        builder = InteractionBuilder(sample_data)
        y, X, names = builder.build_design_matrix_from_parsed(parsed)
        
        assert 'Intercept' in names
        assert 'cat1[T.B]' in names
        assert 'cat1[T.C]' in names
        assert 'x1' in names
        assert 'cat1[T.B]:x1' in names
        assert 'cat1[T.C]:x1' in names
        assert X.shape[1] == 6
        
        # Verify interaction: cat1[T.B]:x1 should be cat1[T.B] * x1
        cat_b_col = names.index('cat1[T.B]')
        x1_col = names.index('x1')
        int_col = names.index('cat1[T.B]:x1')
        np.testing.assert_allclose(X[:, int_col], X[:, cat_b_col] * X[:, x1_col])
    
    def test_categorical_categorical(self, sample_data):
        """Test categorical × categorical interaction."""
        parsed = ParsedFormula(
            response="y",
            main_effects=["cat1", "cat2"],
            interactions=[InteractionTerm(factors=["cat1", "cat2"], categorical_flags=[True, True])],
            categorical_vars={"cat1", "cat2"},
            has_intercept=True,
        )
        builder = InteractionBuilder(sample_data)
        y, X, names = builder.build_design_matrix_from_parsed(parsed)
        
        # cat1 has 3 levels (2 dummies), cat2 has 2 levels (1 dummy)
        # Interaction: 2 × 1 = 2 columns
        # Total: 1 + 2 + 1 + 2 = 6
        assert X.shape[1] == 6
        
        # Check interaction column names
        assert 'cat1[T.B]:cat2[T.Y]' in names
        assert 'cat1[T.C]:cat2[T.Y]' in names
    
    def test_pure_interaction(self, sample_data):
        """Test pure interaction without main effects for some variables."""
        parsed = ParsedFormula(
            response="y",
            main_effects=["x1"],
            interactions=[InteractionTerm(factors=["cat1", "x2"], categorical_flags=[True, False])],
            categorical_vars={"cat1"},
            has_intercept=True,
        )
        builder = InteractionBuilder(sample_data)
        y, X, names = builder.build_design_matrix_from_parsed(parsed)
        
        # Should have: Intercept, x1, cat1[T.B]:x2, cat1[T.C]:x2
        assert 'Intercept' in names
        assert 'x1' in names
        assert 'cat1[T.B]:x2' in names
        assert 'cat1[T.C]:x2' in names
        assert 'cat1[T.B]' not in names
        assert 'cat1[T.C]' not in names
        assert 'x2' not in names


# =============================================================================
# GLM Fitting with Interactions Tests
# =============================================================================

class TestGLMInteractions:
    """Test GLM fitting with interaction terms."""
    
    @pytest.fixture
    def insurance_data(self):
        """Create simulated insurance data."""
        np.random.seed(42)
        n = 1000
        
        age = np.random.uniform(20, 70, n)
        power = np.random.uniform(50, 200, n)
        area = np.random.choice(['Urban', 'Suburban', 'Rural'], n)
        
        # Create claims with some interaction effects
        log_rate = -3.0 + 0.02 * age + 0.01 * power - 0.0001 * age * power
        log_rate += np.where(area == 'Urban', 0.3, np.where(area == 'Suburban', 0.1, 0.0))
        
        claims = np.random.poisson(np.exp(log_rate))
        exposure = np.random.uniform(0.5, 1.0, n)
        
        return pl.DataFrame({
            'claims': claims,
            'age': age,
            'power': power,
            'area': area,
            'exposure': exposure,
        })
    
    def test_fit_continuous_interaction(self, insurance_data):
        """Fit GLM with continuous × continuous interaction."""
        result = rs.glm_dict(
            response='claims',
            terms={'age': {'type': 'linear'}, 'power': {'type': 'linear'}},
            interactions=[{'age': {'type': 'linear'}, 'power': {'type': 'linear'}}],
            data=insurance_data,
            family='poisson',
            offset='exposure',
        ).fit()
        
        assert len(result.params) == 4  # Intercept, age, power, age:power
        assert result.converged
    
    def test_fit_categorical_continuous_interaction(self, insurance_data):
        """Fit GLM with categorical × continuous interaction."""
        result = rs.glm_dict(
            response='claims',
            terms={'area': {'type': 'categorical'}, 'age': {'type': 'linear'}},
            interactions=[{'area': {'type': 'categorical'}, 'age': {'type': 'linear'}}],
            data=insurance_data,
            family='poisson',
            offset='exposure',
        ).fit()
        
        # Intercept + 2 area dummies + age + 2 interactions = 6
        assert len(result.params) == 6
        assert result.converged
    
    def test_fit_categorical_categorical_interaction(self, insurance_data):
        """Fit GLM with categorical × categorical interaction."""
        insurance_data = insurance_data.with_columns(
            pl.Series('fuel', np.random.choice(['Petrol', 'Diesel'], len(insurance_data)))
        )
        
        result = rs.glm_dict(
            response='claims',
            terms={'area': {'type': 'categorical'}, 'fuel': {'type': 'categorical'}},
            interactions=[{'area': {'type': 'categorical'}, 'fuel': {'type': 'categorical'}}],
            data=insurance_data,
            family='poisson',
            offset='exposure',
        ).fit()
        
        # area: 2 dummies, fuel: 1 dummy
        # Total: 1 + 2 + 1 + 2×1 = 6
        assert len(result.params) == 6
        assert result.converged
    
    def test_regularized_interaction_model(self, insurance_data):
        """Fit regularized model with interactions."""
        result = rs.glm_dict(
            response='claims',
            terms={
                'age': {'type': 'linear'},
                'power': {'type': 'linear'},
                'area': {'type': 'categorical'},
            },
            interactions=[{'age': {'type': 'linear'}, 'power': {'type': 'linear'}}],
            data=insurance_data,
            family='poisson',
            offset='exposure',
        ).fit(alpha=0.1, l1_ratio=0.0)  # Ridge
        
        assert result.is_regularized
        assert result.converged
    
    def test_predictions_with_interactions(self, insurance_data):
        """Verify predictions work with interaction models."""
        result = rs.glm_dict(
            response='claims',
            terms={'age': {'type': 'linear'}, 'power': {'type': 'linear'}},
            interactions=[{'age': {'type': 'linear'}, 'power': {'type': 'linear'}}],
            data=insurance_data,
            family='poisson',
            offset='exposure',
        ).fit()
        
        # Check fitted values are reasonable
        fv = result.fittedvalues
        assert np.all(fv >= 0)  # Poisson predictions should be non-negative
        assert len(fv) == len(insurance_data)


# =============================================================================
# Performance Tests
# =============================================================================

class TestInteractionPerformance:
    """Test performance of interaction handling."""
    
    def test_large_categorical_interaction(self):
        """Test performance with high-cardinality categorical interaction."""
        np.random.seed(42)
        n = 50_000
        
        df = pl.DataFrame({
            'y': np.random.poisson(1, n),
            'cat1': np.random.choice([f'A{i}' for i in range(10)], n),
            'cat2': np.random.choice([f'B{i}' for i in range(8)], n),
            'exposure': np.random.uniform(0.5, 1.0, n),
        })
        
        import time
        
        t0 = time.time()
        result = rs.glm_dict(
            response='y',
            terms={'cat1': {'type': 'categorical'}, 'cat2': {'type': 'categorical'}},
            interactions=[{'cat1': {'type': 'categorical'}, 'cat2': {'type': 'categorical'}}],
            data=df,
            family='poisson',
            offset='exposure',
        ).fit()
        t_opt = time.time() - t0
        
        # Should complete in reasonable time
        assert t_opt < 30.0, f"Optimized backend took {t_opt:.1f}s (expected < 30s)"
        
        # Verify correct number of features
        # cat1: 9 dummies, cat2: 7 dummies
        # Total: 1 + 9 + 7 + 63 = 80
        assert len(result.params) == 80
        assert result.converged


# =============================================================================
# Edge Cases
# =============================================================================

class TestInteractionEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_level_categorical(self):
        """Handle categorical with single level (no variation)."""
        df = pl.DataFrame({
            'y': [1, 2, 3, 4],
            'x': [1.0, 2.0, 3.0, 4.0],
            'cat': ['A', 'A', 'A', 'A'],  # Only one level
        })
        
        parsed = ParsedFormula(
            response="y",
            main_effects=["x", "cat"],
            interactions=[],
            categorical_vars={"cat"},
            has_intercept=True,
        )
        builder = InteractionBuilder(df)
        y, X, names = builder.build_design_matrix_from_parsed(parsed)
        
        # Only Intercept and x (no cat dummies since it's constant)
        assert 'x' in names


# =============================================================================
# InteractionTerm Property Tests
# =============================================================================

class TestInteractionTermProperties:
    """Test InteractionTerm dataclass properties."""
    
    def test_order(self):
        """Test order property returns number of factors."""
        term = InteractionTerm(factors=["x1", "x2"], categorical_flags=[False, False])
        assert term.order == 2
        
        term3 = InteractionTerm(factors=["x1", "x2", "x3"], categorical_flags=[False, False, False])
        assert term3.order == 3
    
    def test_is_pure_continuous(self):
        """Test pure continuous detection."""
        term = InteractionTerm(factors=["x1", "x2"], categorical_flags=[False, False])
        assert term.is_pure_continuous
        assert not term.is_pure_categorical
        assert not term.is_mixed
    
    def test_is_pure_categorical(self):
        """Test pure categorical detection."""
        term = InteractionTerm(factors=["cat1", "cat2"], categorical_flags=[True, True])
        assert term.is_pure_categorical
        assert not term.is_pure_continuous
        assert not term.is_mixed
    
    def test_is_mixed(self):
        """Test mixed detection."""
        term = InteractionTerm(factors=["cat1", "x1"], categorical_flags=[True, False])
        assert term.is_mixed
        assert not term.is_pure_continuous
        assert not term.is_pure_categorical


# =============================================================================
# InteractionBuilder Additional Tests
# =============================================================================

class TestInteractionBuilderAdvanced:
    """Additional tests for InteractionBuilder."""
    
    @pytest.fixture
    def spline_data(self):
        """Create data for spline tests."""
        np.random.seed(42)
        n = 100
        return pl.DataFrame({
            'y': np.random.poisson(1, n),
            'age': np.random.uniform(20, 70, n),
            'income': np.random.uniform(30000, 150000, n),
            'region': np.random.choice(['North', 'South', 'East', 'West'], n),
        })
    
    def test_build_with_bspline(self, spline_data):
        """Build design matrix with B-spline term."""
        spline = SplineTerm(var_name="age", spline_type="bs", df=5, degree=3)
        parsed = ParsedFormula(
            response="y",
            main_effects=[],
            interactions=[],
            categorical_vars=set(),
            spline_terms=[spline],
            has_intercept=True,
        )
        builder = InteractionBuilder(spline_data)
        y, X, names = builder.build_design_matrix_from_parsed(parsed)
        
        # Intercept + spline columns
        assert X.shape[1] >= 5
        assert any('bs(age' in name for name in names)
    
    def test_build_with_natural_spline(self, spline_data):
        """Build design matrix with natural spline term."""
        spline = SplineTerm(var_name="age", spline_type="ns", df=4)
        parsed = ParsedFormula(
            response="y",
            main_effects=[],
            interactions=[],
            categorical_vars=set(),
            spline_terms=[spline],
            has_intercept=True,
        )
        builder = InteractionBuilder(spline_data)
        y, X, names = builder.build_design_matrix_from_parsed(parsed)
        
        assert X.shape[1] >= 4
        assert any('ns(age' in name for name in names)
    
    def test_get_spline_info(self, spline_data):
        """Test get_spline_info returns knot information."""
        spline = SplineTerm(var_name="age", spline_type="ns", df=4)
        parsed = ParsedFormula(
            response="y",
            main_effects=[],
            interactions=[],
            categorical_vars=set(),
            spline_terms=[spline],
            has_intercept=True,
        )
        builder = InteractionBuilder(spline_data)
        y, X, names = builder.build_design_matrix_from_parsed(parsed)
        
        info = builder.get_spline_info()
        assert 'age' in info
        assert 'type' in info['age']
        assert 'df' in info['age']
    
    def test_no_intercept(self, spline_data):
        """Test design matrix without intercept."""
        parsed = ParsedFormula(
            response="y",
            main_effects=["age", "income"],
            interactions=[],
            categorical_vars=set(),
            has_intercept=False,
        )
        builder = InteractionBuilder(spline_data)
        y, X, names = builder.build_design_matrix_from_parsed(parsed)
        
        assert 'Intercept' not in names
        assert 'age' in names
        assert 'income' in names


# =============================================================================
# Combined Workflow Tests
# =============================================================================

class TestIntegrationWorkflows:
    """Test complete workflows combining multiple features."""
    
    def test_spline_with_categorical(self):
        """Fit model with spline and categorical terms."""
        np.random.seed(42)
        n = 200
        df = pl.DataFrame({
            'y': np.random.poisson(1, n),
            'age': np.random.uniform(20, 70, n),
            'region': np.random.choice(['A', 'B'], n),
            'exposure': np.random.uniform(0.5, 1.5, n),
        })
        
        result = rs.glm_dict(
            response='y',
            terms={
                'age': {'type': 'ns', 'df': 3},
                'region': {'type': 'categorical'},
            },
            data=df,
            family='poisson',
            offset='exposure',
        ).fit()
        
        assert result.converged
        # 1 intercept + 2 spline + 1 region dummy = 4
        assert len(result.params) >= 4
    
    def test_multiple_interactions(self):
        """Fit model with multiple different interaction types."""
        np.random.seed(42)
        n = 200
        df = pl.DataFrame({
            'y': np.random.poisson(1, n),
            'x1': np.random.uniform(0, 10, n),
            'x2': np.random.uniform(0, 10, n),
            'cat': np.random.choice(['A', 'B', 'C'], n),
            'exposure': np.ones(n),
        })
        
        result = rs.glm_dict(
            response='y',
            terms={
                'x1': {'type': 'linear'},
                'x2': {'type': 'linear'},
                'cat': {'type': 'categorical'},
            },
            interactions=[
                {'x1': {'type': 'linear'}, 'x2': {'type': 'linear'}, 'include_main': False},
                {'cat': {'type': 'categorical'}, 'x1': {'type': 'linear'}, 'include_main': False},
            ],
            data=df,
            family='poisson',
            offset='exposure',
        ).fit()
        
        assert result.converged


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
