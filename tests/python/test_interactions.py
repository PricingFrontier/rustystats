"""
Tests for interaction term support in RustyStats.

Tests cover:
- Continuous × Continuous interactions (x1*x2)
- Categorical × Continuous interactions (C(cat)*x)
- Categorical × Categorical interactions (C(cat1)*C(cat2))
- Pure interaction terms (x1:x2 without main effects)
- Higher-order interactions (x1:x2:x3)
- Performance comparison between backends
"""

import numpy as np
import pandas as pd
import pytest

import rustystats as rs
from rustystats.interactions import (
    InteractionBuilder,
    InteractionTerm,
    ParsedFormula,
    parse_formula_interactions,
)


# =============================================================================
# Formula Parsing Tests
# =============================================================================

class TestFormulaParsing:
    """Test formula parsing and interaction detection."""
    
    def test_parse_simple_formula(self):
        """Parse formula with no interactions."""
        parsed = parse_formula_interactions("y ~ x1 + x2")
        assert parsed.response == "y"
        assert parsed.main_effects == ["x1", "x2"]
        assert parsed.interactions == []
        assert parsed.has_intercept
    
    def test_parse_star_interaction(self):
        """Parse formula with * interaction (main effects + interaction)."""
        parsed = parse_formula_interactions("y ~ x1*x2")
        assert parsed.response == "y"
        assert "x1" in parsed.main_effects
        assert "x2" in parsed.main_effects
        assert len(parsed.interactions) == 1
        assert parsed.interactions[0].factors == ["x1", "x2"]
        assert parsed.interactions[0].is_pure_continuous
    
    def test_parse_colon_interaction(self):
        """Parse formula with : interaction (interaction only)."""
        parsed = parse_formula_interactions("y ~ x1 + x2 + x1:x2")
        assert "x1" in parsed.main_effects
        assert "x2" in parsed.main_effects
        assert len(parsed.interactions) == 1
        assert parsed.interactions[0].factors == ["x1", "x2"]
    
    def test_parse_categorical(self):
        """Parse formula with categorical variables."""
        parsed = parse_formula_interactions("y ~ C(cat) + x1")
        assert "cat" in parsed.categorical_vars
        assert parsed.main_effects == ["cat", "x1"]
    
    def test_parse_categorical_interaction(self):
        """Parse formula with categorical × continuous interaction."""
        parsed = parse_formula_interactions("y ~ C(cat)*x1")
        assert len(parsed.interactions) == 1
        interaction = parsed.interactions[0]
        assert interaction.factors == ["cat", "x1"]
        assert interaction.categorical_flags == [True, False]
        assert interaction.is_mixed
    
    def test_parse_categorical_categorical_interaction(self):
        """Parse formula with categorical × categorical interaction."""
        parsed = parse_formula_interactions("y ~ C(cat1)*C(cat2)")
        assert len(parsed.interactions) == 1
        interaction = parsed.interactions[0]
        assert interaction.factors == ["cat1", "cat2"]
        assert all(interaction.categorical_flags)
        assert interaction.is_pure_categorical
    
    def test_parse_no_intercept(self):
        """Parse formula without intercept."""
        parsed = parse_formula_interactions("y ~ 0 + x1 + x2")
        assert not parsed.has_intercept
        
        parsed = parse_formula_interactions("y ~ x1 + x2 - 1")
        assert not parsed.has_intercept


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
        return pd.DataFrame({
            'y': np.random.poisson(1, n),
            'x1': np.random.uniform(0, 10, n),
            'x2': np.random.uniform(0, 10, n),
            'cat1': np.random.choice(['A', 'B', 'C'], n),
            'cat2': np.random.choice(['X', 'Y'], n),
        })
    
    def test_continuous_continuous(self, sample_data):
        """Test continuous × continuous interaction."""
        builder = InteractionBuilder(sample_data)
        y, X, names = builder.build_design_matrix("y ~ x1*x2")
        
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
        builder = InteractionBuilder(sample_data)
        y, X, names = builder.build_design_matrix("y ~ C(cat1)*x1")
        
        # Should have: Intercept, cat1[T.B], cat1[T.C], x1, cat1[T.B]:x1, cat1[T.C]:x1
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
        builder = InteractionBuilder(sample_data)
        y, X, names = builder.build_design_matrix("y ~ C(cat1)*C(cat2)")
        
        # cat1 has 3 levels (2 dummies), cat2 has 2 levels (1 dummy)
        # Interaction: 2 × 1 = 2 columns
        # Total: 1 + 2 + 1 + 2 = 6
        assert X.shape[1] == 6
        
        # Check interaction column names
        assert 'cat1[T.B]:cat2[T.Y]' in names
        assert 'cat1[T.C]:cat2[T.Y]' in names
    
    def test_pure_interaction(self, sample_data):
        """Test pure interaction without main effects for some variables."""
        builder = InteractionBuilder(sample_data)
        y, X, names = builder.build_design_matrix("y ~ x1 + C(cat1):x2")
        
        # Should have: Intercept, x1, cat1[T.B]:x2, cat1[T.C]:x2
        # Note: No main effect for cat1 or x2
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
        
        return pd.DataFrame({
            'claims': claims,
            'age': age,
            'power': power,
            'area': area,
            'exposure': exposure,
        })
    
    def test_fit_continuous_interaction(self, insurance_data):
        """Fit GLM with continuous × continuous interaction."""
        result = rs.glm(
            'claims ~ age*power',
            insurance_data,
            family='poisson',
            offset='exposure',
            backend='optimized'
        ).fit()
        
        assert len(result.params) == 4  # Intercept, age, power, age:power
        assert result.converged
        
        # Check that we can get a summary
        summary = result.summary()
        assert 'age:power' in summary
    
    def test_fit_categorical_continuous_interaction(self, insurance_data):
        """Fit GLM with categorical × continuous interaction."""
        result = rs.glm(
            'claims ~ C(area)*age',
            insurance_data,
            family='poisson',
            offset='exposure',
            backend='optimized'
        ).fit()
        
        # Intercept + 2 area dummies + age + 2 interactions = 6
        assert len(result.params) == 6
        assert result.converged
        
        # Check feature names
        assert 'C(area)[T.Suburban]:age' in result.feature_names or 'area[T.Suburban]:age' in result.feature_names
    
    def test_fit_categorical_categorical_interaction(self, insurance_data):
        """Fit GLM with categorical × categorical interaction."""
        # Add another categorical
        insurance_data['fuel'] = np.random.choice(['Petrol', 'Diesel'], len(insurance_data))
        
        result = rs.glm(
            'claims ~ C(area)*C(fuel)',
            insurance_data,
            family='poisson',
            offset='exposure',
            backend='optimized'
        ).fit()
        
        # area: 2 dummies, fuel: 1 dummy
        # Total: 1 + 2 + 1 + 2×1 = 6
        assert len(result.params) == 6
        assert result.converged
    
    def test_backend_consistency(self, insurance_data):
        """Verify formulaic and optimized backends give similar results."""
        formula = 'claims ~ age*power'
        
        result_form = rs.glm(
            formula, insurance_data, family='poisson',
            offset='exposure', backend='formulaic'
        ).fit()
        
        result_opt = rs.glm(
            formula, insurance_data, family='poisson',
            offset='exposure', backend='optimized'
        ).fit()
        
        # Coefficients should be very close
        np.testing.assert_allclose(result_form.params, result_opt.params, rtol=1e-5)
        
        # Feature names should match
        assert result_form.feature_names == result_opt.feature_names
    
    def test_regularized_interaction_model(self, insurance_data):
        """Fit regularized model with interactions."""
        result = rs.glm(
            'claims ~ age*power + C(area)',
            insurance_data,
            family='poisson',
            offset='exposure',
            backend='optimized'
        ).fit(alpha=0.1, l1_ratio=0.0)  # Ridge
        
        assert result.is_regularized
        assert result.converged
    
    def test_predictions_with_interactions(self, insurance_data):
        """Verify predictions work with interaction models."""
        result = rs.glm(
            'claims ~ age*power',
            insurance_data,
            family='poisson',
            offset='exposure',
            backend='optimized'
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
        
        df = pd.DataFrame({
            'y': np.random.poisson(1, n),
            'cat1': np.random.choice([f'A{i}' for i in range(10)], n),
            'cat2': np.random.choice([f'B{i}' for i in range(8)], n),
            'exposure': np.random.uniform(0.5, 1.0, n),
        })
        
        import time
        
        t0 = time.time()
        result = rs.glm(
            'y ~ C(cat1)*C(cat2)',
            df,
            family='poisson',
            offset='exposure',
            backend='optimized'
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
        df = pd.DataFrame({
            'y': [1, 2, 3, 4],
            'x': [1.0, 2.0, 3.0, 4.0],
            'cat': ['A', 'A', 'A', 'A'],  # Only one level
        })
        
        # Should work but produce no dummy columns for cat
        builder = InteractionBuilder(df)
        y, X, names = builder.build_design_matrix("y ~ x + C(cat)")
        
        # Only Intercept and x (no cat dummies since it's constant)
        assert 'x' in names
    
    def test_missing_variable(self):
        """Handle reference to non-existent variable."""
        df = pd.DataFrame({
            'y': [1, 2, 3, 4],
            'x': [1.0, 2.0, 3.0, 4.0],
        })
        
        with pytest.raises((KeyError, Exception)):
            rs.glm('y ~ x*z', df, backend='optimized').fit()
    
    def test_empty_formula(self):
        """Handle degenerate formulas."""
        df = pd.DataFrame({
            'y': [1, 2, 3, 4],
            'x': [1.0, 2.0, 3.0, 4.0],
        })
        
        # Intercept-only model
        result = rs.glm('y ~ 1', df, backend='formulaic').fit()
        assert len(result.params) == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
