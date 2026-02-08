"""
Tests for CatBoost-style target encoding.

Tests cover:
- Basic target encoding functionality
- Prevention of target leakage
- Multiple permutations for variance reduction
- Prediction on new data
- TargetEncoder class (sklearn-style)
- Formula integration with TE()
- Edge cases (rare categories, unseen levels)
"""

import pytest
import numpy as np


class TestTargetEncodeBasic:
    """Tests for the target_encode function."""
    
    def test_target_encode_returns_correct_shape(self):
        """Encoded values should have same length as input."""
        import rustystats as rs
        
        categories = ["A", "B", "A", "B", "A", "B"]
        target = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        
        encoded, name, prior, stats = rs.target_encode(categories, target, "cat")
        
        assert len(encoded) == 6
        assert name == "TE(cat)"
        assert isinstance(prior, float)
        assert isinstance(stats, dict)
    
    def test_target_encode_prior_is_mean(self):
        """Prior should be the mean of the target."""
        import rustystats as rs
        
        categories = ["A", "B", "C", "A", "B", "C"]
        target = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        
        encoded, name, prior, stats = rs.target_encode(categories, target, "cat")
        
        assert abs(prior - 3.5) < 1e-10  # mean of 1,2,3,4,5,6
    
    def test_target_encode_level_stats_correct(self):
        """Level statistics should reflect the data."""
        import rustystats as rs
        
        categories = ["A", "B", "A", "B"]
        target = np.array([1.0, 0.0, 1.0, 0.0])
        
        encoded, name, prior, stats = rs.target_encode(categories, target, "cat")
        
        # A: sum=2, count=2
        # B: sum=0, count=2
        assert stats["A"] == (2.0, 2)
        assert stats["B"] == (0.0, 2)
    
    def test_target_encode_deterministic_with_seed(self):
        """Same seed should produce same results."""
        import rustystats as rs
        
        categories = ["A", "B", "C", "A", "B", "C"]
        target = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        
        enc1, _, _, _ = rs.target_encode(categories, target, "cat", seed=42)
        enc2, _, _, _ = rs.target_encode(categories, target, "cat", seed=42)
        
        np.testing.assert_array_almost_equal(enc1, enc2)
    
    def test_target_encode_different_seeds_differ(self):
        """Different seeds should produce different results (usually)."""
        import rustystats as rs
        
        categories = ["A", "B", "C", "A", "B", "C"] * 10  # More data
        target = np.random.randn(60)
        
        enc1, _, _, _ = rs.target_encode(categories, target, "cat", seed=42)
        enc2, _, _, _ = rs.target_encode(categories, target, "cat", seed=123)
        
        # They should be different (with high probability)
        assert not np.allclose(enc1, enc2)


class TestTargetLeakage:
    """Tests that target encoding prevents target leakage."""
    
    def test_unique_categories_get_prior(self):
        """Each unique category should get the prior (no past data)."""
        import rustystats as rs
        
        # Each category is unique - no information from other observations
        categories = [f"cat_{i}" for i in range(10)]
        target = np.array([1.0 if i % 2 == 0 else 0.0 for i in range(10)])
        
        encoded, _, prior, _ = rs.target_encode(
            categories, target, "cat", 
            prior_weight=1.0, n_permutations=1, seed=42
        )
        
        # Each category appears exactly once, so each gets the prior
        for val in encoded:
            assert abs(val - prior) < 1e-10
    
    def test_no_perfect_split(self):
        """Target encoding should not create a perfect split for random data."""
        import rustystats as rs
        
        # Random assignment - should not be predictable
        np.random.seed(42)
        n = 100
        categories = [f"cat_{i}" for i in range(n)]  # Each unique
        target = np.random.randint(0, 2, n).astype(float)
        
        encoded, _, prior, _ = rs.target_encode(
            categories, target, "cat",
            prior_weight=1.0, n_permutations=1, seed=42
        )
        
        # All values should be equal to prior (unique categories)
        assert np.allclose(encoded, prior)


class TestApplyTargetEncoding:
    """Tests for applying target encoding to new data."""
    
    def test_apply_to_seen_categories(self):
        """Applying to seen categories should use training statistics."""
        import rustystats as rs
        
        # Train
        categories = ["A", "B", "A", "B"]
        target = np.array([1.0, 0.0, 1.0, 0.0])
        
        _, _, prior, stats = rs.target_encode(categories, target, "cat")
        
        # Apply to new data with same categories
        new_cats = ["A", "B"]
        new_encoded = rs.apply_target_encoding(new_cats, stats, prior)
        
        # A: (2 + 0.5*1) / (2 + 1) = 2.5/3 ≈ 0.833
        # B: (0 + 0.5*1) / (2 + 1) = 0.5/3 ≈ 0.167
        assert len(new_encoded) == 2
        assert abs(new_encoded[0] - 2.5/3) < 1e-10
        assert abs(new_encoded[1] - 0.5/3) < 1e-10
    
    def test_apply_to_unseen_category(self):
        """Unseen categories should get the prior."""
        import rustystats as rs
        
        # Train
        categories = ["A", "B", "A", "B"]
        target = np.array([1.0, 0.0, 1.0, 0.0])
        
        _, _, prior, stats = rs.target_encode(categories, target, "cat")
        
        # Apply to new data with unseen category
        new_cats = ["A", "C", "D"]  # C and D are unseen
        new_encoded = rs.apply_target_encoding(new_cats, stats, prior)
        
        assert abs(new_encoded[1] - prior) < 1e-10  # C gets prior
        assert abs(new_encoded[2] - prior) < 1e-10  # D gets prior


class TestTargetEncoder:
    """Tests for the TargetEncoder class (sklearn-style API)."""
    
    def test_fit_transform(self):
        """fit_transform should work correctly."""
        import rustystats as rs
        
        categories = ["A", "B", "A", "B", "A", "B"]
        target = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        
        encoder = rs.TargetEncoder(prior_weight=1.0, n_permutations=4, seed=42)
        encoded = encoder.fit_transform(categories, target)
        
        assert len(encoded) == 6
        assert encoder.prior_ is not None
        assert encoder.level_stats_ is not None
    
    def test_transform_after_fit(self):
        """transform should use fitted statistics."""
        import rustystats as rs
        
        # Train
        train_cats = ["A", "B", "A", "B"]
        train_target = np.array([1.0, 0.0, 1.0, 0.0])
        
        encoder = rs.TargetEncoder(prior_weight=1.0)
        encoder.fit(train_cats, train_target)
        
        # Transform new data
        test_cats = ["A", "B", "C"]
        test_encoded = encoder.transform(test_cats)
        
        assert len(test_encoded) == 3
        # C is unseen, should get prior
        assert abs(test_encoded[2] - encoder.prior_) < 1e-10
    
    def test_transform_before_fit_raises(self):
        """transform before fit should raise error."""
        import rustystats as rs
        
        encoder = rs.TargetEncoder()
        
        with pytest.raises(rs.EncodingError, match="not fitted"):
            encoder.transform(["A", "B"])


class TestMultiplePermutations:
    """Tests for multiple permutation averaging."""
    
    def test_more_permutations_reduces_variance(self):
        """More permutations should reduce variance in encoded values."""
        import rustystats as rs
        
        np.random.seed(42)
        categories = ["A", "B", "C"] * 100
        target = np.random.randn(300)
        
        # Run with 1 permutation multiple times
        variances_1perm = []
        for seed in range(10):
            enc, _, _, _ = rs.target_encode(
                categories, target, "cat",
                n_permutations=1, seed=seed
            )
            variances_1perm.append(np.var(enc))
        
        # Run with 10 permutations multiple times
        variances_10perm = []
        for seed in range(10):
            enc, _, _, _ = rs.target_encode(
                categories, target, "cat",
                n_permutations=10, seed=seed
            )
            variances_10perm.append(np.var(enc))
        
        # Variance of variances should be lower with more permutations
        # (more stable estimates)
        assert np.var(variances_10perm) <= np.var(variances_1perm) * 1.5


class TestPriorWeight:
    """Tests for prior weight regularization."""
    
    def test_higher_prior_weight_shrinks_toward_mean(self):
        """Higher prior weight should shrink estimates toward global mean."""
        import rustystats as rs
        
        categories = ["A", "A", "B"]
        target = np.array([1.0, 1.0, 0.0])
        prior = 2/3  # (1+1+0)/3
        
        # Low prior weight
        _, _, _, stats_low = rs.target_encode(
            categories, target, "cat", prior_weight=0.1
        )
        
        # High prior weight
        _, _, _, stats_high = rs.target_encode(
            categories, target, "cat", prior_weight=10.0
        )
        
        # Compute expected values for A
        # A: sum=2, count=2
        # Low: (2 + prior*0.1) / (2 + 0.1) ≈ (2 + 0.0667) / 2.1 ≈ 0.984
        # High: (2 + prior*10) / (2 + 10) ≈ (2 + 6.67) / 12 ≈ 0.722
        
        val_low = (stats_low["A"][0] + prior * 0.1) / (stats_low["A"][1] + 0.1)
        val_high = (stats_high["A"][0] + prior * 10) / (stats_high["A"][1] + 10)
        
        # High prior weight should be closer to prior
        assert abs(val_high - prior) < abs(val_low - prior)


class TestFormulaIntegration:
    """Tests for TE() in formulas."""
    
    def test_parse_te_term(self):
        """TE() should be parsed correctly."""
        from rustystats.interactions import parse_formula_interactions
        
        parsed = parse_formula_interactions("y ~ TE(brand) + age")
        
        assert len(parsed.target_encoding_terms) == 1
        assert parsed.target_encoding_terms[0].var_name == "brand"
        assert parsed.main_effects == ["age"]
    
    def test_parse_te_with_options(self):
        """TE() with options should be parsed correctly."""
        from rustystats.interactions import parse_formula_interactions
        
        parsed = parse_formula_interactions("y ~ TE(brand, prior_weight=2.0, n_permutations=8)")
        
        assert len(parsed.target_encoding_terms) == 1
        assert parsed.target_encoding_terms[0].var_name == "brand"
        assert abs(parsed.target_encoding_terms[0].prior_weight - 2.0) < 1e-10
        assert parsed.target_encoding_terms[0].n_permutations == 8
    
    def test_te_in_design_matrix(self):
        """TE() should work in design matrix construction."""
        import polars as pl
        from rustystats.interactions import InteractionBuilder
        
        # Create test data
        data = pl.DataFrame({
            "y": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            "brand": ["A", "B", "A", "B", "A", "B"],
            "age": [25.0, 30.0, 35.0, 40.0, 45.0, 50.0],
        })
        
        builder = InteractionBuilder(data)
        y, X, names = builder.build_design_matrix("y ~ TE(brand) + age")
        
        # Should have: Intercept, TE(brand), age
        assert X.shape == (6, 3)
        assert "TE(brand)" in names
        assert "age" in names
        assert "Intercept" in names
    
    def test_te_combined_with_splines_and_categorical(self):
        """TE() should work with other term types."""
        import polars as pl
        from rustystats.interactions import InteractionBuilder
        
        data = pl.DataFrame({
            "y": np.random.randn(20),
            "brand": np.random.choice(["A", "B", "C"], 20),
            "region": np.random.choice(["X", "Y"], 20),
            "age": np.random.uniform(20, 60, 20),
        })
        
        builder = InteractionBuilder(data)
        y, X, names = builder.build_design_matrix("y ~ TE(brand) + C(region) + age")
        
        assert "TE(brand)" in names
        assert any("region" in n for n in names)
        assert "age" in names


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_single_category(self):
        """Single category should work."""
        import rustystats as rs
        
        categories = ["A", "A", "A"]
        target = np.array([1.0, 2.0, 3.0])
        
        encoded, _, prior, stats = rs.target_encode(categories, target, "cat")
        
        assert len(encoded) == 3
        assert "A" in stats
    
    def test_empty_category_name(self):
        """Empty string category should work."""
        import rustystats as rs
        
        categories = ["", "A", "", "A"]
        target = np.array([1.0, 0.0, 1.0, 0.0])
        
        encoded, _, prior, stats = rs.target_encode(categories, target, "cat")
        
        assert len(encoded) == 4
        assert "" in stats
    
    def test_numeric_categories(self):
        """Numeric categories should be converted to strings."""
        import rustystats as rs
        
        categories = np.array([1, 2, 1, 2, 1, 2])
        target = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        
        encoded, _, prior, stats = rs.target_encode(categories, target, "cat")
        
        assert len(encoded) == 6
        assert "1" in stats
        assert "2" in stats
    
    def test_large_number_of_categories(self):
        """Many categories should work efficiently."""
        import rustystats as rs
        
        n = 10000
        n_cats = 1000
        categories = [f"cat_{i % n_cats}" for i in range(n)]
        target = np.random.randn(n)
        
        encoded, _, _, stats = rs.target_encode(categories, target, "cat")
        
        assert len(encoded) == n
        assert len(stats) == n_cats
