"""
Tests for formula.py and GLM fitting functionality.

Tests cover:
- FormulaGLM class construction
- Offset and weight processing
- Multiple GLM families (Gaussian, Poisson, Binomial, Gamma, NegBinomial)
- GLMModel attributes and methods
- Regularization (Ridge, Lasso, Elastic Net)
- Cross-validation based regularization
- Model diagnostics and residuals
- Prediction on new data
"""

import numpy as np
import polars as pl
import pytest

import rustystats as rs


# =============================================================================
# FormulaGLM Construction Tests
# =============================================================================

class TestFormulaGLMConstruction:
    """Test FormulaGLM class construction."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample test data."""
        np.random.seed(42)
        n = 100
        return pl.DataFrame({
            'y': np.random.poisson(1, n),
            'x1': np.random.uniform(0, 10, n),
            'x2': np.random.uniform(0, 10, n),
            'cat': np.random.choice(['A', 'B', 'C'], n),
            'exposure': np.random.uniform(0.5, 1.5, n),
            'weight': np.random.uniform(0.5, 2.0, n),
        })
    
    def test_basic_construction(self, sample_data):
        """Test basic model construction."""
        model = rs.glm('y ~ x1 + x2', sample_data)
        
        assert model.formula == 'y ~ x1 + x2'
        assert model.family == 'gaussian'
        assert model.n_obs == 100
        assert model.n_params == 3  # Intercept + x1 + x2
    
    def test_poisson_construction(self, sample_data):
        """Test Poisson model construction."""
        model = rs.glm('y ~ x1', sample_data, family='poisson')
        
        assert model.family == 'poisson'
    
    def test_binomial_construction(self):
        """Test Binomial model construction."""
        np.random.seed(42)
        data = pl.DataFrame({
            'y': np.random.binomial(1, 0.5, 100),
            'x1': np.random.uniform(0, 10, 100),
        })
        model = rs.glm('y ~ x1', data, family='binomial')
        
        assert model.family == 'binomial'
    
    def test_gamma_construction(self):
        """Test Gamma model construction."""
        np.random.seed(42)
        data = pl.DataFrame({
            'y': np.random.gamma(2, 2, 100),
            'x1': np.random.uniform(0, 10, 100),
        })
        model = rs.glm('y ~ x1', data, family='gamma')
        
        assert model.family == 'gamma'
    
    def test_offset_as_column_name(self, sample_data):
        """Test offset specified as column name."""
        model = rs.glm('y ~ x1', sample_data, family='poisson', offset='exposure')
        
        assert model.offset is not None
        assert len(model.offset) == 100
    
    def test_offset_as_array(self, sample_data):
        """Test offset specified as array."""
        offset_arr = np.log(sample_data['exposure'].to_numpy())
        model = rs.glm('y ~ x1', sample_data, family='poisson', offset=offset_arr)
        
        assert model.offset is not None
        np.testing.assert_array_almost_equal(model.offset, offset_arr)
    
    def test_weights_as_column_name(self, sample_data):
        """Test weights specified as column name."""
        model = rs.glm('y ~ x1', sample_data, weights='weight')
        
        assert model.weights is not None
        assert len(model.weights) == 100
    
    def test_weights_as_array(self, sample_data):
        """Test weights specified as array."""
        weights_arr = sample_data['weight'].to_numpy()
        model = rs.glm('y ~ x1', sample_data, weights=weights_arr)
        
        assert model.weights is not None
        np.testing.assert_array_almost_equal(model.weights, weights_arr)
    
    def test_df_model_df_resid(self, sample_data):
        """Test degrees of freedom properties."""
        model = rs.glm('y ~ x1 + x2', sample_data)
        
        assert model.df_model == 2  # x1 + x2 (excluding intercept)
        assert model.df_resid == 97  # 100 - 3


# =============================================================================
# GLM Fitting Tests
# =============================================================================

class TestGLMFitting:
    """Test GLM fitting for various families."""
    
    def test_fit_gaussian(self):
        """Fit Gaussian GLM."""
        np.random.seed(42)
        x = np.random.uniform(0, 10, 100)
        y = 2 + 3 * x + np.random.normal(0, 1, 100)
        data = pl.DataFrame({'y': y, 'x': x})
        
        result = rs.glm('y ~ x', data, family='gaussian').fit()
        
        assert result.converged
        assert len(result.params) == 2
        # Check coefficients are close to true values
        assert abs(result.params[0] - 2) < 1.0  # Intercept
        assert abs(result.params[1] - 3) < 0.5  # Slope
    
    def test_fit_poisson(self):
        """Fit Poisson GLM."""
        np.random.seed(42)
        n = 200
        x = np.random.uniform(0, 2, n)
        mu = np.exp(0.5 + 0.5 * x)
        y = np.random.poisson(mu)
        data = pl.DataFrame({'y': y, 'x': x})
        
        result = rs.glm('y ~ x', data, family='poisson').fit()
        
        assert result.converged
        assert len(result.params) == 2
    
    def test_fit_binomial(self):
        """Fit Binomial GLM."""
        np.random.seed(42)
        n = 200
        x = np.random.uniform(-2, 2, n)
        p = 1 / (1 + np.exp(-(0.5 + x)))
        y = np.random.binomial(1, p)
        data = pl.DataFrame({'y': y.astype(float), 'x': x})
        
        result = rs.glm('y ~ x', data, family='binomial').fit()
        
        assert result.converged
        assert len(result.params) == 2
    
    def test_fit_gamma(self):
        """Fit Gamma GLM."""
        np.random.seed(42)
        n = 200
        x = np.random.uniform(1, 5, n)
        mu = np.exp(1 + 0.3 * x)
        y = np.random.gamma(2, mu / 2, n)
        data = pl.DataFrame({'y': y, 'x': x})
        
        result = rs.glm('y ~ x', data, family='gamma').fit()
        
        assert result.converged
        assert len(result.params) == 2
    
    def test_fit_quasipoisson(self):
        """Fit QuasiPoisson GLM."""
        np.random.seed(42)
        n = 100
        x = np.random.uniform(0, 5, n)
        y = np.random.poisson(np.exp(0.5 + 0.3 * x))
        data = pl.DataFrame({'y': y, 'x': x})
        
        result = rs.glm('y ~ x', data, family='quasipoisson').fit()
        
        assert result.converged
    
    def test_fit_negbinomial(self):
        """Fit Negative Binomial GLM."""
        np.random.seed(42)
        n = 200
        x = np.random.uniform(0, 2, n)
        mu = np.exp(0.5 + 0.3 * x)
        # Simulate NegBin using numpy
        y = np.random.negative_binomial(2, 2 / (2 + mu))
        data = pl.DataFrame({'y': y.astype(float), 'x': x})
        
        result = rs.glm('y ~ x', data, family='negbinomial').fit()
        
        assert result.converged
        assert 'NegativeBinomial' in result.family
    
    def test_fit_with_offset(self):
        """Fit model with offset."""
        np.random.seed(42)
        n = 100
        exposure = np.random.uniform(0.5, 2, n)
        x = np.random.uniform(0, 5, n)
        y = np.random.poisson(exposure * np.exp(0.5 + 0.2 * x))
        data = pl.DataFrame({'y': y, 'x': x, 'exposure': exposure})
        
        result = rs.glm('y ~ x', data, family='poisson', offset='exposure').fit()
        
        assert result.converged
    
    def test_fit_with_weights(self):
        """Fit model with weights."""
        np.random.seed(42)
        n = 100
        x = np.random.uniform(0, 10, n)
        y = 2 + 3 * x + np.random.normal(0, 1, n)
        weights = np.random.uniform(0.5, 2, n)
        data = pl.DataFrame({'y': y, 'x': x, 'w': weights})
        
        result = rs.glm('y ~ x', data, weights='w').fit()
        
        assert result.converged


# =============================================================================
# GLMModel Tests
# =============================================================================

class TestGLMModel:
    """Test GLMModel attributes and methods."""
    
    @pytest.fixture
    def fitted_result(self):
        """Create a fitted GLM result."""
        np.random.seed(42)
        n = 100
        x = np.random.uniform(0, 10, n)
        y = 2 + 3 * x + np.random.normal(0, 1, n)
        data = pl.DataFrame({'y': y, 'x': x})
        return rs.glm('y ~ x', data).fit()
    
    def test_params_shape(self, fitted_result):
        """Test params array shape."""
        assert len(fitted_result.params) == 2
    
    def test_feature_names(self, fitted_result):
        """Test feature names."""
        assert 'Intercept' in fitted_result.feature_names
        assert 'x' in fitted_result.feature_names
    
    def test_fittedvalues_shape(self, fitted_result):
        """Test fitted values shape."""
        assert len(fitted_result.fittedvalues) == 100
    
    def test_linear_predictor(self, fitted_result):
        """Test linear predictor."""
        assert len(fitted_result.linear_predictor) == 100
    
    def test_deviance(self, fitted_result):
        """Test deviance is positive."""
        assert fitted_result.deviance >= 0
    
    def test_bse(self, fitted_result):
        """Test standard errors."""
        bse = fitted_result.bse()
        assert len(bse) == 2
        assert np.all(bse > 0)
    
    def test_tvalues(self, fitted_result):
        """Test t-values."""
        tvals = fitted_result.tvalues()
        assert len(tvals) == 2
        assert np.all(np.isfinite(tvals))
    
    def test_pvalues(self, fitted_result):
        """Test p-values."""
        pvals = fitted_result.pvalues()
        assert len(pvals) == 2
        assert np.all(pvals >= 0)
        assert np.all(pvals <= 1)
    
    def test_conf_int(self, fitted_result):
        """Test confidence intervals."""
        ci = fitted_result.conf_int(alpha=0.05)
        assert ci.shape == (2, 2)
        # Lower bound should be less than upper bound
        assert np.all(ci[:, 0] < ci[:, 1])
    
    def test_significance_codes(self, fitted_result):
        """Test significance codes."""
        codes = fitted_result.significance_codes()
        assert len(codes) == 2
        # Codes should be valid significance markers
        valid_codes = ['***', '**', '*', '.', '']
        for code in codes:
            assert code in valid_codes
    
    def test_summary(self, fitted_result):
        """Test summary generation."""
        summary = fitted_result.summary()
        assert isinstance(summary, str)
        assert 'Intercept' in summary
        assert 'x' in summary


# =============================================================================
# Robust Standard Errors Tests
# =============================================================================

class TestRobustStandardErrors:
    """Test robust (sandwich) standard errors."""
    
    @pytest.fixture
    def heteroscedastic_result(self):
        """Create a result with heteroscedastic errors."""
        np.random.seed(42)
        n = 200
        x = np.random.uniform(1, 10, n)
        # Heteroscedastic: variance increases with x
        y = 2 + 3 * x + np.random.normal(0, x, n)
        data = pl.DataFrame({'y': y, 'x': x})
        return rs.glm('y ~ x', data).fit()
    
    def test_bse_robust_hc1(self, heteroscedastic_result):
        """Test HC1 robust standard errors."""
        bse_robust = heteroscedastic_result.bse_robust("HC1")
        assert len(bse_robust) == 2
        assert np.all(bse_robust > 0)
    
    def test_bse_robust_hc0(self, heteroscedastic_result):
        """Test HC0 robust standard errors."""
        bse_robust = heteroscedastic_result.bse_robust("HC0")
        assert len(bse_robust) == 2
        assert np.all(bse_robust > 0)
    
    def test_bse_robust_hc3(self, heteroscedastic_result):
        """Test HC3 robust standard errors."""
        bse_robust = heteroscedastic_result.bse_robust("HC3")
        assert len(bse_robust) == 2
        assert np.all(bse_robust > 0)
    
    def test_robust_vs_model_se(self, heteroscedastic_result):
        """Robust SE should differ from model-based SE under heteroscedasticity."""
        bse_model = heteroscedastic_result.bse()
        bse_robust = heteroscedastic_result.bse_robust("HC1")
        
        # They should be different (not identical)
        assert not np.allclose(bse_model, bse_robust, rtol=0.01)
    
    def test_tvalues_robust(self, heteroscedastic_result):
        """Test robust t-values."""
        tvals = heteroscedastic_result.tvalues_robust("HC1")
        assert len(tvals) == 2
        assert np.all(np.isfinite(tvals))
    
    def test_pvalues_robust(self, heteroscedastic_result):
        """Test robust p-values."""
        pvals = heteroscedastic_result.pvalues_robust("HC1")
        assert len(pvals) == 2
        assert np.all(pvals >= 0)
        assert np.all(pvals <= 1)
    
    def test_conf_int_robust(self, heteroscedastic_result):
        """Test robust confidence intervals."""
        ci = heteroscedastic_result.conf_int_robust(alpha=0.05, cov_type="HC1")
        assert ci.shape == (2, 2)
        assert np.all(ci[:, 0] < ci[:, 1])


# =============================================================================
# Residual Tests
# =============================================================================

class TestResiduals:
    """Test residual methods."""
    
    @pytest.fixture
    def fitted_result(self):
        """Create a fitted GLM result."""
        np.random.seed(42)
        n = 100
        x = np.random.uniform(0, 10, n)
        y = 2 + 3 * x + np.random.normal(0, 2, n)
        data = pl.DataFrame({'y': y, 'x': x})
        return rs.glm('y ~ x', data).fit()
    
    def test_resid_response(self, fitted_result):
        """Test response residuals."""
        resid = fitted_result.resid_response()
        assert len(resid) == 100
        # Mean of residuals should be close to 0
        assert abs(np.mean(resid)) < 1.0
    
    def test_resid_pearson(self, fitted_result):
        """Test Pearson residuals."""
        resid = fitted_result.resid_pearson()
        assert len(resid) == 100
        assert np.all(np.isfinite(resid))
    
    def test_resid_deviance(self, fitted_result):
        """Test deviance residuals."""
        resid = fitted_result.resid_deviance()
        assert len(resid) == 100
        assert np.all(np.isfinite(resid))


# =============================================================================
# Regularization Tests
# =============================================================================

class TestRegularization:
    """Test regularization options."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data with many predictors."""
        np.random.seed(42)
        n = 200
        data = {'y': np.random.poisson(2, n)}
        for i in range(10):
            data[f'x{i}'] = np.random.uniform(0, 10, n)
        data['exposure'] = np.ones(n)
        return pl.DataFrame(data)
    
    def test_ridge_regularization(self, sample_data):
        """Test Ridge (L2) regularization."""
        formula = 'y ~ ' + ' + '.join([f'x{i}' for i in range(10)])
        result = rs.glm(formula, sample_data, family='poisson', offset='exposure').fit(
            alpha=0.1, l1_ratio=0.0
        )
        
        assert result.converged
        assert result.is_regularized
    
    def test_lasso_regularization(self, sample_data):
        """Test Lasso (L1) regularization."""
        formula = 'y ~ ' + ' + '.join([f'x{i}' for i in range(10)])
        result = rs.glm(formula, sample_data, family='poisson', offset='exposure').fit(
            alpha=0.1, l1_ratio=1.0
        )
        
        assert result.converged
        assert result.is_regularized
        # Lasso should produce some zero coefficients
        # (with strong enough penalty)
    
    def test_elastic_net_regularization(self, sample_data):
        """Test Elastic Net regularization."""
        formula = 'y ~ ' + ' + '.join([f'x{i}' for i in range(10)])
        result = rs.glm(formula, sample_data, family='poisson', offset='exposure').fit(
            alpha=0.1, l1_ratio=0.5
        )
        
        assert result.converged
        assert result.is_regularized
    
    def test_no_regularization(self, sample_data):
        """Test unregularized fit."""
        result = rs.glm('y ~ x0 + x1', sample_data, family='poisson', offset='exposure').fit(
            alpha=0.0
        )
        
        assert result.converged
        assert not result.is_regularized


# =============================================================================
# Prediction Tests
# =============================================================================

class TestPrediction:
    """Test prediction functionality."""
    
    def test_predict_on_training_data(self):
        """Predict on training data."""
        np.random.seed(42)
        n = 100
        x = np.random.uniform(0, 10, n)
        y = 2 + 3 * x + np.random.normal(0, 1, n)
        data = pl.DataFrame({'y': y, 'x': x})
        
        result = rs.glm('y ~ x', data).fit()
        predictions = result.fittedvalues
        
        assert len(predictions) == n
        # Correlation with y should be high
        assert np.corrcoef(predictions, y)[0, 1] > 0.9
    
    def test_predict_on_new_data(self):
        """Predict on new data."""
        np.random.seed(42)
        n = 100
        x = np.random.uniform(0, 10, n)
        y = 2 + 3 * x + np.random.normal(0, 1, n)
        train_data = pl.DataFrame({'y': y, 'x': x})
        
        result = rs.glm('y ~ x', train_data).fit()
        
        # Create new data
        new_x = np.array([1.0, 5.0, 9.0])
        new_data = pl.DataFrame({'x': new_x})
        
        predictions = result.predict(new_data)
        
        assert len(predictions) == 3
        # Should be close to 2 + 3*x
        expected = 2 + 3 * new_x
        np.testing.assert_array_almost_equal(predictions, expected, decimal=0)


# =============================================================================
# Model Validation Tests
# =============================================================================

class TestModelValidation:
    """Test model validation functionality."""
    
    def test_validate_good_model(self):
        """Validate a well-specified model."""
        np.random.seed(42)
        n = 100
        x = np.random.uniform(0, 10, n)
        y = 2 + 3 * x + np.random.normal(0, 1, n)
        data = pl.DataFrame({'y': y, 'x': x})
        
        model = rs.glm('y ~ x', data)
        validation = model.validate(verbose=False)
        
        assert 'valid' in validation
        # A simple model should be valid
        assert validation['valid']


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_intercept_only_model(self):
        """Fit intercept-only model."""
        np.random.seed(42)
        data = pl.DataFrame({'y': np.random.normal(5, 1, 100)})
        
        result = rs.glm('y ~ 1', data).fit()
        
        assert result.converged
        assert len(result.params) == 1
        # Intercept should be close to mean
        assert abs(result.params[0] - 5) < 0.5
    
    def test_no_intercept_model(self):
        """Fit model without intercept."""
        np.random.seed(42)
        x = np.random.uniform(1, 10, 100)
        y = 3 * x + np.random.normal(0, 1, 100)
        data = pl.DataFrame({'y': y, 'x': x})
        
        result = rs.glm('y ~ 0 + x', data).fit()
        
        assert result.converged
        assert 'Intercept' not in result.feature_names
    
    def test_single_observation_fails(self):
        """Single observation should fail gracefully."""
        data = pl.DataFrame({'y': [1.0], 'x': [1.0]})
        
        with pytest.raises(Exception):
            rs.glm('y ~ x', data).fit()
    
    def test_missing_variable_fails(self):
        """Missing variable should raise error."""
        data = pl.DataFrame({'y': [1.0, 2.0], 'x': [1.0, 2.0]})
        
        with pytest.raises((KeyError, Exception)):
            rs.glm('y ~ z', data).fit()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
