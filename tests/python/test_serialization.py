"""
Tests for model serialization (to_bytes/from_bytes).
"""

import numpy as np
import polars as pl
import pytest

import rustystats as rs


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n = 500
    
    return pl.DataFrame({
        "y": np.random.poisson(2, n).astype(float),
        "x1": np.random.randn(n),
        "x2": np.random.randn(n),
        "cat": np.random.choice(["A", "B", "C"], n),
        "exposure": np.random.uniform(0.5, 2.0, n),
    })


class TestBasicSerialization:
    """Test basic serialization roundtrip."""
    
    def test_simple_model_roundtrip(self, sample_data):
        """Test that a simple model can be serialized and loaded."""
        result = rs.glm("y ~ x1 + x2", sample_data, family="poisson").fit()
        
        # Serialize
        model_bytes = result.to_bytes()
        assert isinstance(model_bytes, bytes)
        assert len(model_bytes) > 0
        
        # Deserialize
        loaded = rs.FormulaGLMResults.from_bytes(model_bytes)
        
        # Check basic properties
        assert loaded.family == result.family
        assert loaded.formula == result.formula
        assert loaded.link == result.link
        assert len(loaded.params) == len(result.params)
        np.testing.assert_array_almost_equal(loaded.params, result.params)
    
    def test_categorical_model_roundtrip(self, sample_data):
        """Test serialization with categorical variables."""
        result = rs.glm("y ~ x1 + C(cat)", sample_data, family="poisson").fit()
        
        model_bytes = result.to_bytes()
        loaded = rs.FormulaGLMResults.from_bytes(model_bytes)
        
        np.testing.assert_array_almost_equal(loaded.params, result.params)
        assert loaded.feature_names == result.feature_names
    
    def test_offset_model_roundtrip(self, sample_data):
        """Test serialization with offset."""
        result = rs.glm(
            "y ~ x1 + C(cat)", 
            sample_data, 
            family="poisson",
            offset="exposure"
        ).fit()
        
        model_bytes = result.to_bytes()
        loaded = rs.FormulaGLMResults.from_bytes(model_bytes)
        
        np.testing.assert_array_almost_equal(loaded.params, result.params)


class TestPredictionAfterLoad:
    """Test that predictions work after loading."""
    
    def test_predict_after_load(self, sample_data):
        """Test that predictions match after serialization."""
        train = sample_data.head(400)
        test = sample_data.tail(100)
        
        result = rs.glm("y ~ x1 + x2 + C(cat)", train, family="poisson").fit()
        original_pred = result.predict(test)
        
        # Serialize and load
        loaded = rs.FormulaGLMResults.from_bytes(result.to_bytes())
        loaded_pred = loaded.predict(test)
        
        np.testing.assert_array_almost_equal(original_pred, loaded_pred)
    
    def test_predict_with_offset_after_load(self, sample_data):
        """Test predictions with offset after serialization."""
        train = sample_data.head(400)
        test = sample_data.tail(100)
        
        result = rs.glm(
            "y ~ x1 + C(cat)", 
            train, 
            family="poisson",
            offset="exposure"
        ).fit()
        original_pred = result.predict(test)
        
        loaded = rs.FormulaGLMResults.from_bytes(result.to_bytes())
        loaded_pred = loaded.predict(test)
        
        np.testing.assert_array_almost_equal(original_pred, loaded_pred)


class TestSplineSerialization:
    """Test serialization with spline terms."""
    
    def test_spline_model_roundtrip(self, sample_data):
        """Test serialization with B-splines."""
        result = rs.glm(
            "y ~ bs(x1, df=4) + x2", 
            sample_data, 
            family="poisson"
        ).fit()
        
        model_bytes = result.to_bytes()
        loaded = rs.FormulaGLMResults.from_bytes(model_bytes)
        
        np.testing.assert_array_almost_equal(loaded.params, result.params)
    
    def test_spline_prediction_after_load(self, sample_data):
        """Test spline predictions after serialization."""
        train = sample_data.head(400)
        test = sample_data.tail(100)
        
        result = rs.glm(
            "y ~ bs(x1, df=4) + C(cat)", 
            train, 
            family="poisson"
        ).fit()
        original_pred = result.predict(test)
        
        loaded = rs.FormulaGLMResults.from_bytes(result.to_bytes())
        loaded_pred = loaded.predict(test)
        
        np.testing.assert_array_almost_equal(original_pred, loaded_pred)


class TestInteractionSerialization:
    """Test serialization with interaction terms."""
    
    def test_interaction_model_roundtrip(self, sample_data):
        """Test serialization with interactions."""
        result = rs.glm("y ~ x1*x2", sample_data, family="poisson").fit()
        
        model_bytes = result.to_bytes()
        loaded = rs.FormulaGLMResults.from_bytes(model_bytes)
        
        np.testing.assert_array_almost_equal(loaded.params, result.params)
    
    def test_cat_interaction_prediction(self, sample_data):
        """Test categorical interaction predictions after load."""
        train = sample_data.head(400)
        test = sample_data.tail(100)
        
        result = rs.glm("y ~ C(cat)*x1", train, family="poisson").fit()
        original_pred = result.predict(test)
        
        loaded = rs.FormulaGLMResults.from_bytes(result.to_bytes())
        loaded_pred = loaded.predict(test)
        
        np.testing.assert_array_almost_equal(original_pred, loaded_pred)


class TestModelProperties:
    """Test that model properties are preserved."""
    
    def test_properties_preserved(self, sample_data):
        """Test that key properties are preserved."""
        result = rs.glm("y ~ x1 + x2 + C(cat)", sample_data, family="poisson").fit()
        loaded = rs.FormulaGLMResults.from_bytes(result.to_bytes())
        
        assert loaded.deviance == result.deviance
        assert loaded.converged == result.converged
        assert loaded.iterations == result.iterations
        assert loaded.nobs == result.nobs
        assert loaded.df_resid == result.df_resid
        assert loaded.df_model == result.df_model
    
    def test_different_families(self, sample_data):
        """Test serialization with different families."""
        for family in ["gaussian", "poisson", "gamma"]:
            # Ensure positive values for gamma
            data = sample_data.with_columns(pl.col("y").abs() + 0.1)
            
            result = rs.glm("y ~ x1 + x2", data, family=family).fit()
            loaded = rs.FormulaGLMResults.from_bytes(result.to_bytes())
            
            assert loaded.family == family
            np.testing.assert_array_almost_equal(loaded.params, result.params)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_invalid_bytes_raises(self):
        """Test that invalid bytes raise an error."""
        with pytest.raises(Exception):
            rs.FormulaGLMResults.from_bytes(b"invalid data")
    
    def test_empty_bytes_raises(self):
        """Test that empty bytes raise an error."""
        with pytest.raises(Exception):
            rs.FormulaGLMResults.from_bytes(b"")
    
    def test_intercept_only_model(self, sample_data):
        """Test serialization of intercept-only model."""
        result = rs.glm("y ~ 1", sample_data, family="poisson").fit()
        loaded = rs.FormulaGLMResults.from_bytes(result.to_bytes())
        
        np.testing.assert_array_almost_equal(loaded.params, result.params)
