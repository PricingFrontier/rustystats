"""Tests for Frequency Encoding and Target Encoding Interactions.

These features are inspired by CatBoost's categorical encoding strategies:
- Frequency Encoding (Counter CTR): Encodes by category frequency
- TE Interactions: Target encoding for categorical combinations

Test coverage:
- Core function behavior with exact value verification
- Formula API: FE(var) and TE(var1:var2) syntax
- Dictionary API: type='frequency_encoding' and type='target_encoding' with interaction
- Edge cases and error handling
"""

import numpy as np
import polars as pl
import pytest
import rustystats as rs
from rustystats.interactions import (
    InteractionBuilder,
    parse_formula_interactions,
    FrequencyEncodingTermSpec,
    TargetEncodingTermSpec,
)
from rustystats.formula import dict_to_parsed_formula


# =============================================================================
# FREQUENCY ENCODING CORE TESTS
# =============================================================================


class TestFrequencyEncode:
    """Tests for the frequency_encode function."""
    
    def test_basic_encoding(self):
        """Test basic frequency encoding."""
        categories = ["A", "B", "A", "A", "B", "C"]
        
        encoded, name, counts, max_count, n_obs = rs.frequency_encode(categories, "cat")
        
        # Check metadata
        assert name == "FE(cat)"
        assert max_count == 3  # A appears 3 times
        assert n_obs == 6
        
        # Check counts
        assert counts["A"] == 3
        assert counts["B"] == 2
        assert counts["C"] == 1
        
        # Check encoded values: count / max_count
        # [A, B, A, A, B, C] -> [3/3, 2/3, 3/3, 3/3, 2/3, 1/3]
        expected = np.array([1.0, 2/3, 1.0, 1.0, 2/3, 1/3])
        np.testing.assert_allclose(encoded, expected)
    
    def test_single_category(self):
        """All same category should encode to 1.0."""
        categories = ["X", "X", "X", "X"]
        
        encoded, name, counts, max_count, n_obs = rs.frequency_encode(categories, "var")
        
        assert max_count == 4
        np.testing.assert_allclose(encoded, [1.0, 1.0, 1.0, 1.0])
    
    def test_uniform_categories(self):
        """Equal frequency categories should all get same encoding."""
        categories = ["A", "B", "C", "A", "B", "C"]
        
        encoded, _, counts, max_count, _ = rs.frequency_encode(categories, "x")
        
        # All have count 2, max is 2, so all encode to 1.0
        assert max_count == 2
        np.testing.assert_allclose(encoded, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    
    def test_numpy_array_input(self):
        """Should handle numpy array input."""
        categories = np.array(["A", "B", "A", "C"])
        
        encoded, name, _, _, _ = rs.frequency_encode(categories, "test")
        
        assert name == "FE(test)"
        assert len(encoded) == 4


class TestApplyFrequencyEncoding:
    """Tests for apply_frequency_encoding on new data."""
    
    def test_apply_to_new_data(self):
        """Apply encoding to new data with seen categories."""
        train_categories = ["A", "B", "A", "A", "B"]
        
        encoded, _, counts, max_count, _ = rs.frequency_encode(train_categories, "cat")
        
        # Apply to new data
        test_categories = ["A", "B", "A"]
        test_encoded = rs.apply_frequency_encoding(test_categories, counts, max_count)
        
        # A: 3/3 = 1.0, B: 2/3
        expected = np.array([1.0, 2/3, 1.0])
        np.testing.assert_allclose(test_encoded, expected)
    
    def test_unseen_categories_get_zero(self):
        """Unseen categories should encode to 0.0."""
        train_categories = ["A", "B", "A"]
        
        _, _, counts, max_count, _ = rs.frequency_encode(train_categories, "cat")
        
        # Apply with unseen categories
        test_categories = ["A", "B", "C", "D"]  # C, D are unseen
        test_encoded = rs.apply_frequency_encoding(test_categories, counts, max_count)
        
        # A: 2/2 = 1.0, B: 1/2 = 0.5, C: 0, D: 0
        expected = np.array([1.0, 0.5, 0.0, 0.0])
        np.testing.assert_allclose(test_encoded, expected)


class TestFrequencyEncoder:
    """Tests for the FrequencyEncoder sklearn-style class."""
    
    def test_fit_transform(self):
        """Test fit_transform method."""
        encoder = rs.FrequencyEncoder()
        
        categories = ["A", "B", "A", "A", "B", "C"]
        encoded = encoder.fit_transform(categories)
        
        assert len(encoded) == 6
        assert encoder.max_count_ == 3
        assert encoder.level_counts_["A"] == 3
    
    def test_fit_then_transform(self):
        """Test separate fit and transform calls."""
        encoder = rs.FrequencyEncoder()
        
        train_categories = ["A", "B", "A", "A"]
        encoder.fit(train_categories)
        
        test_categories = ["A", "B", "C"]
        test_encoded = encoder.transform(test_categories)
        
        # A: 3/3, B: 1/3, C: 0
        expected = np.array([1.0, 1/3, 0.0])
        np.testing.assert_allclose(test_encoded, expected)
    
    def test_transform_before_fit_raises(self):
        """Transform before fit should raise error."""
        encoder = rs.FrequencyEncoder()
        
        with pytest.raises(ValueError, match="not fitted"):
            encoder.transform(["A", "B"])


# =============================================================================
# TARGET ENCODING INTERACTION TESTS
# =============================================================================


class TestTargetEncodeInteraction:
    """Tests for target_encode_interaction function."""
    
    def test_basic_interaction(self):
        """Test basic two-way categorical interaction encoding."""
        cat1 = ["A", "A", "B", "B"]
        cat2 = ["X", "Y", "X", "Y"]
        target = np.array([1.0, 2.0, 3.0, 4.0])
        
        encoded, name, prior, stats = rs.target_encode_interaction(
            cat1, cat2, target, "c1", "c2", seed=42
        )
        
        # Check structure
        assert name == "TE(c1:c2)"
        assert len(encoded) == 4
        
        # Prior should be mean of target
        assert abs(prior - 2.5) < 1e-10
        
        # Should have 4 unique combinations
        assert len(stats) == 4
        assert "A:X" in stats
        assert "A:Y" in stats
        assert "B:X" in stats
        assert "B:Y" in stats
    
    def test_repeated_combinations(self):
        """Test with repeated category combinations."""
        cat1 = ["A", "A", "A", "B", "B", "B"]
        cat2 = ["X", "X", "Y", "X", "Y", "Y"]
        target = np.array([1.0, 1.0, 0.0, 1.0, 0.0, 0.0])
        
        encoded, name, prior, stats = rs.target_encode_interaction(
            cat1, cat2, target, "brand", "region", 
            prior_weight=1.0, n_permutations=4, seed=42
        )
        
        # Check level stats
        assert stats["A:X"][1] == 2  # count for A:X is 2
        assert abs(stats["A:X"][0] - 2.0) < 1e-10  # sum for A:X is 2.0
        
        assert stats["B:Y"][1] == 2  # count for B:Y is 2
        assert abs(stats["B:Y"][0] - 0.0) < 1e-10  # sum for B:Y is 0.0
    
    def test_numpy_array_input(self):
        """Should handle numpy array inputs."""
        cat1 = np.array(["A", "B", "A", "B"])
        cat2 = np.array(["X", "X", "Y", "Y"])
        target = np.array([1.0, 2.0, 3.0, 4.0])
        
        encoded, name, prior, stats = rs.target_encode_interaction(
            cat1, cat2, target, "a", "b"
        )
        
        assert len(encoded) == 4
        assert name == "TE(a:b)"
    
    def test_no_leakage(self):
        """Unique combinations should get prior (no leakage)."""
        # Each combination is unique
        cat1 = ["A", "B", "C", "D"]
        cat2 = ["W", "X", "Y", "Z"]
        target = np.array([1.0, 0.0, 1.0, 0.0])
        
        encoded, _, prior, _ = rs.target_encode_interaction(
            cat1, cat2, target, "c1", "c2",
            prior_weight=1.0, n_permutations=1, seed=42
        )
        
        # With ordered statistics, unique combinations get only prior
        np.testing.assert_allclose(encoded, [prior] * 4)
    
    def test_deterministic_with_seed(self):
        """Results should be reproducible with same seed."""
        cat1 = ["A", "B", "A", "B", "A", "B"]
        cat2 = ["X", "Y", "X", "Y", "X", "Y"]
        target = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        
        enc1, _, _, _ = rs.target_encode_interaction(
            cat1, cat2, target, "a", "b", seed=12345
        )
        enc2, _, _, _ = rs.target_encode_interaction(
            cat1, cat2, target, "a", "b", seed=12345
        )
        
        np.testing.assert_allclose(enc1, enc2)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestEncodingComparison:
    """Compare different encoding strategies."""
    
    def test_fe_vs_te_different_signals(self):
        """FE and TE capture fundamentally different information."""
        # Create scenario where frequency and target mean are inversely related
        # Rare category C has highest target, frequent category A has lowest
        categories = ["A", "A", "A", "A", "B", "B", "C"]
        target = np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 1.0])
        
        # Frequency encoding: A=1.0 (4/4), B=0.5 (2/4), C=0.25 (1/4)
        fe_encoded, _, counts, max_count, _ = rs.frequency_encode(categories, "cat")
        
        # Verify FE values exactly
        assert counts["A"] == 4
        assert counts["B"] == 2
        assert counts["C"] == 1
        assert max_count == 4
        
        # Most frequent (A) gets 1.0, rare (C) gets 0.25
        np.testing.assert_allclose(fe_encoded[0], 1.0)  # A
        np.testing.assert_allclose(fe_encoded[4], 0.5)  # B
        np.testing.assert_allclose(fe_encoded[6], 0.25)  # C
        
        # Target encoding: captures target relationship
        te_encoded, _, prior, stats = rs.target_encode(categories, target, "cat", seed=42)
        
        # Prior is mean of target
        expected_prior = target.mean()
        np.testing.assert_allclose(prior, expected_prior)
        
        # Stats should reflect sum and count correctly
        np.testing.assert_allclose(stats["A"][0], 0.0)  # sum for A
        assert stats["A"][1] == 4  # count for A
        np.testing.assert_allclose(stats["C"][0], 1.0)  # sum for C
        assert stats["C"][1] == 1  # count for C
        
        # TE and FE must differ - they capture different signals
        assert not np.allclose(fe_encoded, te_encoded)
    
    def test_te_interaction_vs_main_effects(self):
        """TE interaction captures joint effects not in main effects."""
        # Setup: interaction effect where brand X region matters
        # Same brand/region individually but different combined
        cat1 = ["A", "A", "B", "B", "A", "A", "B", "B"]
        cat2 = ["X", "Y", "X", "Y", "X", "Y", "X", "Y"]
        # A:X and B:Y are high, A:Y and B:X are low (interaction effect)
        target = np.array([1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0])
        
        # Main effect for A: mean([1,0,1,0]) = 0.5
        # Main effect for B: mean([0,1,0,1]) = 0.5
        # But interaction: A:X=1.0, A:Y=0.0, B:X=0.0, B:Y=1.0
        
        encoded, name, prior, stats = rs.target_encode_interaction(
            cat1, cat2, target, "c1", "c2", seed=42
        )
        
        # Prior is global mean
        np.testing.assert_allclose(prior, 0.5)
        
        # Check interaction statistics
        assert stats["A:X"][1] == 2  # count
        np.testing.assert_allclose(stats["A:X"][0], 2.0)  # sum (1+1)
        assert stats["A:Y"][1] == 2
        np.testing.assert_allclose(stats["A:Y"][0], 0.0)  # sum (0+0)
        assert stats["B:X"][1] == 2
        np.testing.assert_allclose(stats["B:X"][0], 0.0)  # sum (0+0)
        assert stats["B:Y"][1] == 2
        np.testing.assert_allclose(stats["B:Y"][0], 2.0)  # sum (1+1)


# =============================================================================
# FORMULA API INTEGRATION TESTS
# =============================================================================


class TestFormulaAPIFrequencyEncoding:
    """Test FE() in formula strings."""
    
    @pytest.fixture
    def sample_data(self):
        return pl.DataFrame({
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "brand": ["A", "B", "A", "B", "A", "B"],
            "region": ["N", "N", "S", "S", "N", "S"],
            "age": [25.0, 30.0, 35.0, 40.0, 45.0, 50.0],
        })
    
    def test_parse_fe_term(self):
        """FE() should be parsed correctly."""
        parsed = parse_formula_interactions("y ~ FE(brand) + age")
        
        assert len(parsed.frequency_encoding_terms) == 1
        assert parsed.frequency_encoding_terms[0].var_name == "brand"
        assert "age" in parsed.main_effects
        assert parsed.response == "y"
    
    def test_parse_multiple_fe_terms(self):
        """Multiple FE() terms should parse correctly."""
        parsed = parse_formula_interactions("y ~ FE(brand) + FE(region)")
        
        assert len(parsed.frequency_encoding_terms) == 2
        var_names = {t.var_name for t in parsed.frequency_encoding_terms}
        assert var_names == {"brand", "region"}
    
    def test_fe_in_design_matrix(self, sample_data):
        """FE() should produce correct design matrix column."""
        builder = InteractionBuilder(sample_data)
        y, X, names = builder.build_design_matrix("y ~ FE(brand) + age", seed=42)
        
        # Check structure
        assert "FE(brand)" in names
        assert "age" in names
        assert "Intercept" in names
        
        # FE(brand) should be in column after intercept and age
        fe_idx = names.index("FE(brand)")
        fe_col = X[:, fe_idx]
        
        # brand has A:3, B:3, so both should encode to 1.0 (equal frequency)
        np.testing.assert_allclose(fe_col, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    
    def test_fe_transform_new_data(self, sample_data):
        """FE() should apply correctly to new data."""
        builder = InteractionBuilder(sample_data)
        y, X, names = builder.build_design_matrix("y ~ FE(brand) + age", seed=42)
        
        # New data with same and unseen categories
        new_data = pl.DataFrame({
            "brand": ["A", "B", "C"],  # C is unseen
            "age": [30.0, 40.0, 50.0],
        })
        
        X_new = builder.transform_new_data(new_data)
        fe_idx = names.index("FE(brand)")
        fe_col_new = X_new[:, fe_idx]
        
        # A and B: 1.0 (both had max count), C: 0.0 (unseen)
        np.testing.assert_allclose(fe_col_new, [1.0, 1.0, 0.0])


class TestFormulaAPITargetEncodingInteraction:
    """Test TE(var1:var2) in formula strings."""
    
    @pytest.fixture
    def sample_data(self):
        return pl.DataFrame({
            "y": [1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
            "brand": ["A", "A", "B", "B", "A", "B"],
            "region": ["N", "S", "N", "S", "N", "S"],
            "age": [25.0, 30.0, 35.0, 40.0, 45.0, 50.0],
        })
    
    def test_parse_te_interaction(self):
        """TE(a:b) should parse with interaction_vars."""
        parsed = parse_formula_interactions("y ~ TE(brand:region) + age")
        
        assert len(parsed.target_encoding_terms) == 1
        te_term = parsed.target_encoding_terms[0]
        assert te_term.var_name == "brand:region"
        assert te_term.interaction_vars == ["brand", "region"]
    
    def test_parse_three_way_te_interaction(self):
        """TE(a:b:c) should parse correctly."""
        parsed = parse_formula_interactions("y ~ TE(brand:region:year)")
        
        te_term = parsed.target_encoding_terms[0]
        assert te_term.var_name == "brand:region:year"
        assert te_term.interaction_vars == ["brand", "region", "year"]
    
    def test_te_interaction_in_design_matrix(self, sample_data):
        """TE(a:b) should produce correct design matrix column."""
        builder = InteractionBuilder(sample_data)
        y, X, names = builder.build_design_matrix(
            "y ~ TE(brand:region) + age", seed=42
        )
        
        # Check structure
        assert "TE(brand:region)" in names
        assert X.shape[0] == 6
        
        te_idx = names.index("TE(brand:region)")
        te_col = X[:, te_idx]
        
        # Values should be finite and in reasonable range
        assert np.all(np.isfinite(te_col))
        assert np.all(te_col >= 0) and np.all(te_col <= 1)
    
    def test_te_interaction_with_options(self):
        """TE(a:b, prior_weight=X) should parse options."""
        parsed = parse_formula_interactions(
            "y ~ TE(brand:region, prior_weight=2.0, n_permutations=8)"
        )
        
        te_term = parsed.target_encoding_terms[0]
        assert te_term.interaction_vars == ["brand", "region"]
        np.testing.assert_allclose(te_term.prior_weight, 2.0)
        assert te_term.n_permutations == 8


# =============================================================================
# DICTIONARY API INTEGRATION TESTS
# =============================================================================


class TestDictAPIFrequencyEncoding:
    """Test frequency_encoding type in dict API."""
    
    def test_dict_frequency_encoding_parsed(self):
        """type='frequency_encoding' should create FE term."""
        parsed = dict_to_parsed_formula(
            response="y",
            terms={
                "brand": {"type": "frequency_encoding"},
                "age": {"type": "linear"},
            },
        )
        
        assert len(parsed.frequency_encoding_terms) == 1
        assert parsed.frequency_encoding_terms[0].var_name == "brand"
        assert "age" in parsed.main_effects


class TestDictAPITargetEncodingInteraction:
    """Test target_encoding with interaction in dict API via interactions list."""
    
    def test_dict_te_interaction_parsed(self):
        """target_encoding: True in interactions should work."""
        parsed = dict_to_parsed_formula(
            response="y",
            terms={
                "age": {"type": "linear"},
            },
            interactions=[
                {
                    "brand": {"type": "categorical"},
                    "region": {"type": "categorical"},
                    "target_encoding": True,
                },
            ],
        )
        
        assert len(parsed.target_encoding_terms) == 1
        te_term = parsed.target_encoding_terms[0]
        assert te_term.var_name == "brand:region"
        assert te_term.interaction_vars == ["brand", "region"]
    
    def test_dict_te_interaction_with_options(self):
        """TE interaction with prior_weight option."""
        parsed = dict_to_parsed_formula(
            response="y",
            terms={},
            interactions=[
                {
                    "brand": {"type": "categorical"},
                    "region": {"type": "categorical"},
                    "target_encoding": True,
                    "prior_weight": 2.5,
                    "n_permutations": 10,
                },
            ],
        )
        
        te_term = parsed.target_encoding_terms[0]
        np.testing.assert_allclose(te_term.prior_weight, 2.5)
        assert te_term.n_permutations == 10
    
    def test_dict_te_interaction_requires_two_vars(self):
        """TE interaction must have at least 2 variables."""
        with pytest.raises(ValueError, match="at least 2 variables"):
            dict_to_parsed_formula(
                response="y",
                terms={},
                interactions=[
                    {
                        "brand": {"type": "categorical"},
                        "target_encoding": True,
                    },
                ],
            )
    
    def test_dict_te_interaction_with_include_main(self):
        """TE interaction with include_main adds main effects."""
        parsed = dict_to_parsed_formula(
            response="y",
            terms={},
            interactions=[
                {
                    "brand": {"type": "categorical"},
                    "region": {"type": "categorical"},
                    "target_encoding": True,
                    "include_main": True,
                },
            ],
        )
        
        # Should have TE interaction
        assert len(parsed.target_encoding_terms) == 1
        te_term = parsed.target_encoding_terms[0]
        assert te_term.var_name == "brand:region"
        
        # Should also have categorical vars tracked
        assert "brand" in parsed.categorical_vars
        assert "region" in parsed.categorical_vars
    
    def test_dict_te_interaction_continuous_categorical(self):
        """TE interaction with continuous variable (type=linear)."""
        parsed = dict_to_parsed_formula(
            response="y",
            terms={},
            interactions=[
                {
                    "age": {"type": "linear"},
                    "region": {"type": "categorical"},
                    "target_encoding": True,
                },
            ],
        )
        
        te_term = parsed.target_encoding_terms[0]
        assert te_term.var_name == "age:region"
        assert te_term.interaction_vars == ["age", "region"]


# =============================================================================
# EDGE CASES AND ERROR HANDLING
# =============================================================================


class TestEdgeCases:
    """Edge cases and error handling."""
    
    def test_fe_empty_string_category(self):
        """Empty string as category should work."""
        categories = ["", "A", "", "B"]
        encoded, _, counts, _, _ = rs.frequency_encode(categories, "x")
        
        assert counts[""] == 2
        assert len(encoded) == 4
    
    def test_fe_special_characters(self):
        """Special characters in categories should work."""
        categories = ["A:B", "C:D", "A:B"]
        encoded, _, counts, _, _ = rs.frequency_encode(categories, "x")
        
        assert counts["A:B"] == 2
        assert counts["C:D"] == 1
    
    def test_te_interaction_mismatched_lengths(self):
        """Mismatched category lengths should raise error."""
        cat1 = ["A", "B", "A"]
        cat2 = ["X", "Y"]  # Wrong length
        target = np.array([1.0, 2.0, 3.0])
        
        # Rust panics are converted to BaseException subclasses
        with pytest.raises(BaseException, match="same length"):
            rs.target_encode_interaction(cat1, cat2, target, "c1", "c2")
    
    def test_fe_large_cardinality(self):
        """High cardinality should work efficiently."""
        n_categories = 1000
        n_obs = 10000
        
        # Create skewed distribution
        categories = []
        for i in range(n_categories):
            count = max(1, n_categories - i)  # Decreasing frequency
            categories.extend([f"cat_{i}"] * count)
        categories = categories[:n_obs]
        
        encoded, _, counts, max_count, _ = rs.frequency_encode(categories, "x")
        
        assert len(encoded) == n_obs
        assert max_count == n_categories  # cat_0 has highest count


class TestTEInteractionPredict:
    """Test TE interaction prediction on new data."""
    
    def test_te_interaction_predict_new_data(self):
        """TE interaction should work with predict() on new data.
        
        Regression test: _encode_target_new was trying to access 
        new_data["VehBrand:Region"] instead of combining the columns.
        """
        import polars as pl
        
        # Create train data
        train_data = pl.DataFrame({
            "y": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0],
            "brand": ["A", "A", "B", "B", "A", "B", "A", "B"],
            "region": ["X", "Y", "X", "Y", "X", "Y", "Y", "X"],
            "age": [25, 30, 35, 40, 45, 50, 55, 60],
        })
        
        # Create test data (separate columns, no "brand:region" column)
        test_data = pl.DataFrame({
            "brand": ["A", "B", "A"],
            "region": ["X", "Y", "X"],
            "age": [28, 42, 52],
        })
        
        # Fit model with TE interaction
        result = rs.glm_dict(
            response="y",
            terms={"age": {"type": "linear"}},
            interactions=[{
                "brand": {"type": "categorical"},
                "region": {"type": "categorical"},
                "target_encoding": True,
            }],
            data=train_data,
            family="gaussian",
        ).fit()
        
        # This should NOT raise ColumnNotFoundError
        predictions = result.predict(test_data)
        
        assert len(predictions) == 3
        assert all(np.isfinite(predictions))
    
    def test_te_interaction_diagnostics(self):
        """TE interaction should work with diagnostics().
        
        Regression test: diagnostics() calls predict() internally.
        """
        import polars as pl
        
        train_data = pl.DataFrame({
            "y": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0],
            "brand": ["A", "A", "B", "B", "A", "B", "A", "B"],
            "region": ["X", "Y", "X", "Y", "X", "Y", "Y", "X"],
            "age": [25, 30, 35, 40, 45, 50, 55, 60],
        })
        
        result = rs.glm_dict(
            response="y",
            terms={"age": {"type": "linear"}},
            interactions=[{
                "brand": {"type": "categorical"},
                "region": {"type": "categorical"},
                "target_encoding": True,
            }],
            data=train_data,
            family="gaussian",
        ).fit()
        
        # This should NOT raise ColumnNotFoundError
        diag = result.diagnostics(train_data=train_data)
        assert diag is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
