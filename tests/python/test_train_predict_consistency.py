"""
Regression tests for train/predict consistency.

These tests verify that transformations (splines, encodings, etc.) produce
consistent results between training and prediction on new data.

This was added after discovering a critical bug where natural splines
would recompute knots from new data instead of using stored training knots,
causing catastrophic prediction errors.
"""

import pytest
import numpy as np
import polars as pl
import rustystats as rs


class TestSplineTrainPredictConsistency:
    """Verify spline basis is consistent between training and prediction."""
    
    def test_ns_knots_reused_on_new_data(self):
        """Natural spline knots should be stored and reused on new data."""
        np.random.seed(42)
        
        # Training data with specific distribution
        train_x = np.random.uniform(0, 100, 1000)
        train_y = 5 + 0.1 * train_x + np.random.normal(0, 1, 1000)
        train_data = pl.DataFrame({"x": train_x, "y": train_y})
        
        # Test data with DIFFERENT distribution (but within range)
        test_x = np.random.uniform(20, 80, 500)  # Narrower range
        test_y = 5 + 0.1 * test_x + np.random.normal(0, 1, 500)
        test_data = pl.DataFrame({"x": test_x, "y": test_y})
        
        # Fit model
        result = rs.glm("y ~ ns(x, df=5)", train_data, family="gaussian").fit()
        
        # Get predictions
        train_pred = result.fittedvalues
        test_pred = result.predict(test_data)
        
        # Key check: predictions should be in similar range
        # If knots were recomputed, test predictions would be wildly different
        assert test_pred.min() > 0, "Test predictions should be positive"
        assert test_pred.max() < 20, "Test predictions should be reasonable"
        assert abs(test_pred.mean() - train_pred.mean()) < 5, \
            "Train and test mean predictions should be similar"
        
        # Verify no extreme values that would indicate knot recomputation
        assert np.all(np.isfinite(test_pred)), "All predictions should be finite"
        assert np.std(test_pred) < 10 * np.std(train_pred), \
            "Test prediction variance should not explode"
    
    def test_bs_knots_reused_on_new_data(self):
        """B-spline knots should be stored and reused on new data."""
        np.random.seed(42)
        
        train_x = np.random.uniform(0, 100, 1000)
        train_y = 5 + 0.1 * train_x + np.random.normal(0, 1, 1000)
        train_data = pl.DataFrame({"x": train_x, "y": train_y})
        
        test_x = np.random.uniform(20, 80, 500)
        test_y = 5 + 0.1 * test_x + np.random.normal(0, 1, 500)
        test_data = pl.DataFrame({"x": test_x, "y": test_y})
        
        result = rs.glm("y ~ bs(x, df=5)", train_data, family="gaussian").fit()
        
        train_pred = result.fittedvalues
        test_pred = result.predict(test_data)
        
        assert test_pred.min() > 0
        assert test_pred.max() < 20
        assert abs(test_pred.mean() - train_pred.mean()) < 5
        assert np.all(np.isfinite(test_pred))
    
    def test_monotonic_spline_knots_reused(self):
        """Monotonic spline knots should be stored and reused."""
        np.random.seed(42)
        
        train_x = np.random.uniform(0, 100, 1000)
        # Monotonically increasing relationship
        train_y = 5 + 0.1 * train_x + np.random.normal(0, 0.5, 1000)
        train_data = pl.DataFrame({"x": train_x, "y": train_y})
        
        test_x = np.random.uniform(20, 80, 500)
        test_y = 5 + 0.1 * test_x + np.random.normal(0, 0.5, 500)
        test_data = pl.DataFrame({"x": test_x, "y": test_y})
        
        # Formula parser uses 'increasing=True' syntax
        result = rs.glm("y ~ bs(x, df=5, increasing=True)", 
                       train_data, family="gaussian").fit()
        
        train_pred = result.fittedvalues
        test_pred = result.predict(test_data)
        
        assert test_pred.min() > 0
        assert test_pred.max() < 20
        assert abs(test_pred.mean() - train_pred.mean()) < 5
        assert np.all(np.isfinite(test_pred))


class TestGLMDictTrainPredictConsistency:
    """Verify glm_dict transformations are consistent."""
    
    def test_glm_dict_ns_consistency(self):
        """glm_dict with ns() should produce consistent predictions."""
        np.random.seed(42)
        
        train_x = np.random.uniform(0, 100, 1000)
        train_y = np.exp(5 + 0.01 * train_x + np.random.normal(0, 0.5, 1000))
        train_data = pl.DataFrame({"x": train_x, "y": train_y})
        
        test_x = np.random.uniform(20, 80, 500)
        test_y = np.exp(5 + 0.01 * test_x + np.random.normal(0, 0.5, 500))
        test_data = pl.DataFrame({"x": test_x, "y": test_y})
        
        terms = {"x": {"type": "ns", "df": 5}}
        result = rs.glm_dict(response="y", terms=terms, data=train_data, 
                            family="gamma").fit()
        
        train_pred = result.fittedvalues
        test_pred = result.predict(test_data)
        
        # Gamma predictions should be positive and reasonable
        assert test_pred.min() > 50, f"Min test pred too low: {test_pred.min()}"
        assert test_pred.max() < 500, f"Max test pred too high: {test_pred.max()}"
        assert np.all(np.isfinite(test_pred))
        
        # Mean predictions should be similar
        ratio = test_pred.mean() / train_pred.mean()
        assert 0.5 < ratio < 2.0, f"Prediction ratio out of range: {ratio}"
    
    def test_glm_dict_target_encoding_consistency(self):
        """Target encoding should use training statistics on new data."""
        np.random.seed(42)
        
        n_train, n_test = 1000, 500
        
        # Create categorical with different levels having different means
        train_cat = np.random.choice(["A", "B", "C"], n_train)
        train_y = np.where(train_cat == "A", 100, 
                   np.where(train_cat == "B", 200, 300))
        train_y = train_y + np.random.normal(0, 10, n_train)
        train_data = pl.DataFrame({"cat": train_cat, "y": train_y})
        
        # Test data - same categories
        test_cat = np.random.choice(["A", "B", "C"], n_test)
        test_y = np.where(test_cat == "A", 100,
                  np.where(test_cat == "B", 200, 300))
        test_y = test_y + np.random.normal(0, 10, n_test)
        test_data = pl.DataFrame({"cat": test_cat, "y": test_y})
        
        terms = {"cat": {"type": "target_encoding"}}
        result = rs.glm_dict(response="y", terms=terms, data=train_data,
                            family="gaussian").fit()
        
        train_pred = result.fittedvalues
        test_pred = result.predict(test_data)
        
        # Predictions should reflect the category means
        assert test_pred.min() > 50
        assert test_pred.max() < 350
        assert np.all(np.isfinite(test_pred))
        
        # Check predictions cluster around category means
        test_pred_A = test_pred[test_cat == "A"]
        test_pred_B = test_pred[test_cat == "B"]
        test_pred_C = test_pred[test_cat == "C"]
        
        assert test_pred_A.mean() < test_pred_B.mean() < test_pred_C.mean(), \
            "Target encoding should preserve category ordering"


class TestCategoricalEncodingConsistency:
    """Verify categorical encoding is consistent."""
    
    def test_categorical_levels_preserved(self):
        """Categorical levels from training should be used on new data."""
        np.random.seed(42)
        
        train_cat = np.random.choice(["A", "B", "C", "D"], 1000)
        train_y = np.random.normal(10, 1, 1000)
        train_data = pl.DataFrame({"cat": train_cat, "y": train_y})
        
        # Test data with subset of categories
        test_cat = np.random.choice(["A", "B"], 500)  # Only A and B
        test_y = np.random.normal(10, 1, 500)
        test_data = pl.DataFrame({"cat": test_cat, "y": test_y})
        
        result = rs.glm("y ~ C(cat)", train_data, family="gaussian").fit()
        
        train_pred = result.fittedvalues
        test_pred = result.predict(test_data)
        
        assert np.all(np.isfinite(test_pred))
        assert abs(test_pred.mean() - train_pred.mean()) < 1
    
    def test_unseen_categorical_level_handled(self):
        """Unseen categorical levels should map to reference level."""
        np.random.seed(42)
        
        train_cat = np.random.choice(["A", "B", "C"], 1000)
        train_y = np.random.normal(10, 1, 1000)
        train_data = pl.DataFrame({"cat": train_cat, "y": train_y})
        
        # Test data with an unseen level
        test_cat = np.array(["A", "B", "X", "Y", "Z"])  # X, Y, Z are new
        test_y = np.random.normal(10, 1, 5)
        test_data = pl.DataFrame({"cat": test_cat, "y": test_y})
        
        result = rs.glm("y ~ C(cat)", train_data, family="gaussian").fit()
        test_pred = result.predict(test_data)
        
        # Should not crash and should produce finite predictions
        assert np.all(np.isfinite(test_pred))


class TestInteractionConsistency:
    """Verify interaction terms are consistent."""
    
    def test_spline_categorical_interaction(self):
        """Spline × categorical interactions should be consistent."""
        np.random.seed(42)
        
        n_train, n_test = 1000, 500
        
        train_x = np.random.uniform(0, 100, n_train)
        train_cat = np.random.choice(["A", "B"], n_train)
        train_y = 5 + 0.1 * train_x + np.where(train_cat == "A", 0, 5)
        train_y += np.random.normal(0, 1, n_train)
        train_data = pl.DataFrame({"x": train_x, "cat": train_cat, "y": train_y})
        
        test_x = np.random.uniform(20, 80, n_test)
        test_cat = np.random.choice(["A", "B"], n_test)
        test_y = 5 + 0.1 * test_x + np.where(test_cat == "A", 0, 5)
        test_y += np.random.normal(0, 1, n_test)
        test_data = pl.DataFrame({"x": test_x, "cat": test_cat, "y": test_y})
        
        result = rs.glm("y ~ ns(x, df=3):C(cat)", train_data, 
                       family="gaussian").fit()
        
        train_pred = result.fittedvalues
        test_pred = result.predict(test_data)
        
        assert np.all(np.isfinite(test_pred))
        ratio = test_pred.mean() / train_pred.mean()
        assert 0.5 < ratio < 2.0


class TestDiagnosticsConsistency:
    """Verify diagnostics compute consistent predictions."""
    
    def test_diagnostics_train_test_predictions_sensible(self):
        """Diagnostics should produce sensible predictions for both train and test."""
        np.random.seed(42)
        
        # Create realistic severity-like data
        n = 2000
        x1 = np.random.uniform(0, 100, n)
        x2 = np.random.uniform(50, 150, n)
        cat = np.random.choice(["A", "B", "C"], n)
        
        # Log-linear response for Gamma
        log_mu = 6 + 0.01 * x1 + 0.005 * x2
        log_mu += np.where(cat == "A", 0, np.where(cat == "B", 0.2, 0.4))
        y = np.random.gamma(shape=5, scale=np.exp(log_mu)/5, size=n)
        
        data = pl.DataFrame({"x1": x1, "x2": x2, "cat": cat, "y": y})
        
        # Split into train/test
        train_data = data.head(1500)
        test_data = data.tail(500)
        
        # Fit on train only
        terms = {
            "x1": {"type": "ns", "df": 4},
            "x2": {"type": "ns", "df": 4},
            "cat": {"type": "target_encoding"}
        }
        result = rs.glm_dict(response="y", terms=terms, data=train_data,
                            family="gamma").fit()
        
        # Get predictions
        train_pred = result.fittedvalues
        test_pred = result.predict(test_data)
        
        # Key assertions
        assert train_pred.min() > 100, f"Train min too low: {train_pred.min()}"
        assert test_pred.min() > 100, f"Test min too low: {test_pred.min()}"
        
        train_ae = train_data["y"].sum() / train_pred.sum()
        test_ae = test_data["y"].sum() / test_pred.sum()
        
        assert 0.8 < train_ae < 1.2, f"Train A/E out of range: {train_ae}"
        assert 0.5 < test_ae < 2.0, f"Test A/E out of range: {test_ae}"
        
        # No extreme predictions
        assert np.all(test_pred > 50), "Test predictions too low"
        assert np.all(test_pred < 10000), "Test predictions too high"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
