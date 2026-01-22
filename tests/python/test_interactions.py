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
import polars as pl
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
        return pl.DataFrame({
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
        
        return pl.DataFrame({
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
            offset='exposure'
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
            offset='exposure'
        ).fit()
        
        # Intercept + 2 area dummies + age + 2 interactions = 6
        assert len(result.params) == 6
        assert result.converged
        
        # Check feature names
        assert 'C(area)[T.Suburban]:age' in result.feature_names or 'area[T.Suburban]:age' in result.feature_names
    
    def test_fit_categorical_categorical_interaction(self, insurance_data):
        """Fit GLM with categorical × categorical interaction."""
        # Add another categorical
        insurance_data = insurance_data.with_columns(
            pl.Series('fuel', np.random.choice(['Petrol', 'Diesel'], len(insurance_data)))
        )
        
        result = rs.glm(
            'claims ~ C(area)*C(fuel)',
            insurance_data,
            family='poisson',
            offset='exposure'
        ).fit()
        
        # area: 2 dummies, fuel: 1 dummy
        # Total: 1 + 2 + 1 + 2×1 = 6
        assert len(result.params) == 6
        assert result.converged
    
    
    def test_regularized_interaction_model(self, insurance_data):
        """Fit regularized model with interactions."""
        result = rs.glm(
            'claims ~ age*power + C(area)',
            insurance_data,
            family='poisson',
            offset='exposure'
        ).fit(alpha=0.1, l1_ratio=0.0)  # Ridge
        
        assert result.is_regularized
        assert result.converged
    
    def test_predictions_with_interactions(self, insurance_data):
        """Verify predictions work with interaction models."""
        result = rs.glm(
            'claims ~ age*power',
            insurance_data,
            family='poisson',
            offset='exposure'
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
        result = rs.glm(
            'y ~ C(cat1)*C(cat2)',
            df,
            family='poisson',
            offset='exposure'
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
        
        # Should work but produce no dummy columns for cat
        builder = InteractionBuilder(df)
        y, X, names = builder.build_design_matrix("y ~ x + C(cat)")
        
        # Only Intercept and x (no cat dummies since it's constant)
        assert 'x' in names
    
    def test_missing_variable(self):
        """Handle reference to non-existent variable."""
        df = pl.DataFrame({
            'y': [1, 2, 3, 4],
            'x': [1.0, 2.0, 3.0, 4.0],
        })
        
        with pytest.raises((KeyError, Exception)):
            rs.glm('y ~ x*z', df).fit()
    
    def test_empty_formula(self):
        """Handle degenerate formulas."""
        df = pl.DataFrame({
            'y': [1, 2, 3, 4],
            'x': [1.0, 2.0, 3.0, 4.0],
        })
        
        # Intercept-only model
        result = rs.glm('y ~ 1', df).fit()
        assert len(result.params) == 1


# =============================================================================
# Categorical Level Selection Tests
# =============================================================================

class TestCategoricalLevelSelection:
    """Test C(var, level='value') syntax for single-level indicators."""
    
    def test_parse_single_level(self):
        """Parse C(var, level='value') syntax."""
        parsed = parse_formula_interactions("y ~ C(Region, level='Paris') + age")
        assert len(parsed.categorical_terms) == 1
        assert parsed.categorical_terms[0].var_name == "Region"
        assert parsed.categorical_terms[0].levels == ["Paris"]
        assert parsed.main_effects == ["age"]
        # Should NOT be in categorical_vars (that's for full C())
        assert "Region" not in parsed.categorical_vars
    
    def test_parse_multiple_levels(self):
        """Parse C(var, levels=['a', 'b']) syntax."""
        parsed = parse_formula_interactions("y ~ C(Region, levels=['Paris', 'Lyon'])")
        assert len(parsed.categorical_terms) == 1
        assert parsed.categorical_terms[0].var_name == "Region"
        assert parsed.categorical_terms[0].levels == ["Paris", "Lyon"]
    
    def test_regular_c_still_works(self):
        """Regular C(var) should still work as before."""
        parsed = parse_formula_interactions("y ~ C(Region) + age")
        assert len(parsed.categorical_terms) == 0
        assert "Region" in parsed.main_effects
        assert "Region" in parsed.categorical_vars
    
    def test_build_single_level_indicator(self):
        """Build 0/1 indicator for single level."""
        data = pl.DataFrame({
            'y': [1.0, 2.0, 3.0, 4.0, 5.0],
            'Region': ['Paris', 'Lyon', 'Paris', 'Marseille', 'Lyon'],
            'age': [25.0, 30.0, 35.0, 40.0, 45.0]
        })
        
        builder = InteractionBuilder(data)
        y, X, names = builder.build_design_matrix("y ~ C(Region, level='Paris') + age")
        
        assert "Region[Paris]" in names
        paris_idx = names.index("Region[Paris]")
        expected = np.array([1.0, 0.0, 1.0, 0.0, 0.0])
        np.testing.assert_array_equal(X[:, paris_idx], expected)
    
    def test_build_multiple_level_indicators(self):
        """Build indicators for multiple specific levels."""
        data = pl.DataFrame({
            'y': [1.0, 2.0, 3.0, 4.0, 5.0],
            'Region': ['Paris', 'Lyon', 'Paris', 'Marseille', 'Lyon'],
            'age': [25.0, 30.0, 35.0, 40.0, 45.0]
        })
        
        builder = InteractionBuilder(data)
        y, X, names = builder.build_design_matrix("y ~ C(Region, levels=['Paris', 'Lyon']) + age")
        
        assert "Region[Paris]" in names
        assert "Region[Lyon]" in names
        
        paris_idx = names.index("Region[Paris]")
        lyon_idx = names.index("Region[Lyon]")
        
        np.testing.assert_array_equal(X[:, paris_idx], [1.0, 0.0, 1.0, 0.0, 0.0])
        np.testing.assert_array_equal(X[:, lyon_idx], [0.0, 1.0, 0.0, 0.0, 1.0])
    
    def test_fit_model_with_single_level(self):
        """Fit a GLM with single-level categorical indicator."""
        np.random.seed(42)
        data = pl.DataFrame({
            'claims': np.random.poisson(0.1, 100),
            'Region': np.random.choice(['Paris', 'Lyon', 'Marseille'], 100),
            'age': np.random.uniform(20, 60, 100),
            'exposure': np.random.uniform(0.5, 1.5, 100)
        })
        
        result = rs.glm(
            "claims ~ C(Region, level='Paris') + age",
            data=data,
            family='poisson',
            offset='exposure'
        ).fit()
        
        assert result.converged
        assert "Region[Paris]" in result.feature_names
        assert len(result.params) == 3  # Intercept + Paris + age


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
# Additional Formula Parsing Tests  
# =============================================================================

class TestFormulaParsingAdvanced:
    """Additional formula parsing tests for edge cases."""
    
    def test_parse_three_way_interaction(self):
        """Parse formula with three-way interaction."""
        parsed = parse_formula_interactions("y ~ x1:x2:x3")
        assert len(parsed.interactions) == 1
        assert parsed.interactions[0].factors == ["x1", "x2", "x3"]
        assert parsed.interactions[0].order == 3
    
    def test_parse_spline_term(self):
        """Parse formula with B-spline term."""
        parsed = parse_formula_interactions("y ~ bs(age, df=5)")
        assert len(parsed.spline_terms) == 1
        assert parsed.spline_terms[0].var_name == "age"
        assert parsed.spline_terms[0].df == 5
    
    def test_parse_natural_spline(self):
        """Parse formula with natural spline term."""
        parsed = parse_formula_interactions("y ~ ns(age, df=4)")
        assert len(parsed.spline_terms) == 1
        assert parsed.spline_terms[0].var_name == "age"
        assert parsed.spline_terms[0].spline_type == "ns"
    
    def test_parse_monotonic_spline(self):
        """Parse formula with monotonic spline term."""
        parsed = parse_formula_interactions("y ~ ms(age, df=4)")
        assert len(parsed.spline_terms) == 1
        assert parsed.spline_terms[0].spline_type == "ms"
    
    def test_parse_target_encoding(self):
        """Parse formula with target encoding term."""
        parsed = parse_formula_interactions("y ~ TE(Region)")
        assert len(parsed.target_encoding_terms) == 1
        assert parsed.target_encoding_terms[0].var_name == "Region"
    
    def test_parse_target_encoding_with_params(self):
        """Parse formula with target encoding with custom parameters."""
        parsed = parse_formula_interactions("y ~ TE(Region, pw=2, nperm=5)")
        assert len(parsed.target_encoding_terms) == 1
        assert parsed.target_encoding_terms[0].var_name == "Region"
        assert parsed.target_encoding_terms[0].prior_weight == 2.0
        assert parsed.target_encoding_terms[0].n_permutations == 5
    
    def test_parse_identity_term(self):
        """Parse formula with identity term."""
        parsed = parse_formula_interactions("y ~ I(x ** 2)")
        assert len(parsed.identity_terms) == 1
        assert parsed.identity_terms[0].expression == "x ** 2"
    
    def test_parse_constraint_terms(self):
        """Parse formula with constraint terms."""
        parsed = parse_formula_interactions("y ~ pos(x) + neg(z)")
        assert len(parsed.constraint_terms) == 2
        assert parsed.constraint_terms[0].var_name == "x"
        assert parsed.constraint_terms[0].constraint == "pos"
        assert parsed.constraint_terms[1].var_name == "z"
        assert parsed.constraint_terms[1].constraint == "neg"
    
    def test_parse_complex_formula(self):
        """Parse complex formula with multiple term types."""
        parsed = parse_formula_interactions(
            "y ~ x1*x2 + C(cat) + bs(age, df=5) + C(region):x3"
        )
        assert "x1" in parsed.main_effects
        assert "x2" in parsed.main_effects
        assert "cat" in parsed.categorical_vars
        assert len(parsed.spline_terms) == 1
        # Should have x1:x2 and region:x3 interactions
        assert len(parsed.interactions) >= 1


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
        builder = InteractionBuilder(spline_data)
        y, X, names = builder.build_design_matrix("y ~ bs(age, df=5)")
        
        # Intercept + 4 spline columns (df=5 with intercept absorbed)
        assert X.shape[1] >= 5
        assert any('bs(age' in name for name in names)
    
    def test_build_with_natural_spline(self, spline_data):
        """Build design matrix with natural spline term."""
        builder = InteractionBuilder(spline_data)
        y, X, names = builder.build_design_matrix("y ~ ns(age, df=4)")
        
        assert X.shape[1] >= 4
        assert any('ns(age' in name for name in names)
    
    def test_build_with_spline_and_categorical(self, spline_data):
        """Build design matrix with spline and categorical."""
        builder = InteractionBuilder(spline_data)
        y, X, names = builder.build_design_matrix("y ~ ns(age, df=3) + C(region)")
        
        # Intercept + 2 spline + 3 region dummies
        assert X.shape[1] >= 6
    
    def test_get_spline_info(self, spline_data):
        """Test get_spline_info returns knot information."""
        builder = InteractionBuilder(spline_data)
        y, X, names = builder.build_design_matrix("y ~ ns(age, df=4)")
        
        info = builder.get_spline_info()
        assert 'age' in info
        assert 'type' in info['age']
        assert 'df' in info['age']
    
    def test_no_intercept_formula(self, spline_data):
        """Test formula without intercept."""
        builder = InteractionBuilder(spline_data)
        y, X, names = builder.build_design_matrix("y ~ 0 + age + income")
        
        assert 'Intercept' not in names
        assert 'age' in names
        assert 'income' in names


# =============================================================================
# Combined Workflow Tests
# =============================================================================

class TestIntegrationWorkflows:
    """Test complete workflows combining multiple features."""
    
    def test_spline_with_categorical_interaction(self):
        """Fit model with spline × categorical interaction."""
        np.random.seed(42)
        n = 200
        df = pl.DataFrame({
            'y': np.random.poisson(1, n),
            'age': np.random.uniform(20, 70, n),
            'region': np.random.choice(['A', 'B'], n),
            'exposure': np.random.uniform(0.5, 1.5, n),
        })
        
        result = rs.glm(
            'y ~ ns(age, df=3) + C(region)',
            df,
            family='poisson',
            offset='exposure'
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
        
        result = rs.glm(
            'y ~ x1:x2 + C(cat):x1',
            df,
            family='poisson',
            offset='exposure'
        ).fit()
        
        assert result.converged
        assert 'x1:x2' in result.feature_names


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
