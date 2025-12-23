"""
Tests for GLM Fitting
=====================

These tests verify that GLM fitting produces correct results.
We compare against known values and check that the algorithm converges.

Run with: pytest tests/python/test_glm.py -v
"""

import numpy as np
import pytest

import rustystats as rs


class TestGaussianGLM:
    """Tests for Gaussian (OLS) GLM."""
    
    def test_simple_linear_regression(self):
        """Gaussian GLM with identity link should match OLS."""
        np.random.seed(42)
        n = 100
        
        # True model: y = 2 + 3*x + noise
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = 2 + 3 * X[:, 1] + np.random.randn(n) * 0.5
        
        result = rs.fit_glm(y, X, family="gaussian")
        
        assert result.converged
        # Coefficients should be close to true values
        np.testing.assert_almost_equal(result.params[0], 2.0, decimal=0)
        np.testing.assert_almost_equal(result.params[1], 3.0, decimal=0)
    
    def test_perfect_fit(self):
        """Perfect linear relationship should have near-zero deviance."""
        X = np.column_stack([np.ones(5), [1, 2, 3, 4, 5]])
        y = np.array([3, 5, 7, 9, 11])  # y = 1 + 2*x
        
        result = rs.fit_glm(y, X, family="gaussian")
        
        assert result.converged
        np.testing.assert_almost_equal(result.params[0], 1.0, decimal=10)
        np.testing.assert_almost_equal(result.params[1], 2.0, decimal=10)
        np.testing.assert_almost_equal(result.deviance, 0.0, decimal=10)
    
    def test_fitted_values(self):
        """Fitted values should equal X @ beta."""
        np.random.seed(123)
        n = 50
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.random.randn(n)
        
        result = rs.fit_glm(y, X, family="gaussian")
        
        expected_fitted = X @ result.params
        np.testing.assert_array_almost_equal(result.fittedvalues, expected_fitted)


class TestPoissonGLM:
    """Tests for Poisson GLM."""
    
    def test_poisson_convergence(self):
        """Poisson GLM should converge."""
        np.random.seed(42)
        n = 100
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        mu = np.exp(0.5 + 0.3 * X[:, 1])
        y = np.random.poisson(mu)
        
        result = rs.fit_glm(y, X, family="poisson")
        
        assert result.converged
        assert result.iterations < 25
    
    def test_poisson_fitted_positive(self):
        """Poisson fitted values should all be positive."""
        np.random.seed(42)
        n = 100
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.random.poisson(lam=2, size=n)
        
        result = rs.fit_glm(y, X, family="poisson")
        
        assert np.all(result.fittedvalues > 0)
    
    def test_poisson_coefficient_recovery(self):
        """Poisson should recover true coefficients with large sample."""
        np.random.seed(42)
        n = 1000  # Large sample for better recovery
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        true_beta = [0.5, 0.3]
        mu = np.exp(X @ true_beta)
        y = np.random.poisson(mu)
        
        result = rs.fit_glm(y, X, family="poisson")
        
        # Should be within 0.1 of true values
        assert abs(result.params[0] - true_beta[0]) < 0.1
        assert abs(result.params[1] - true_beta[1]) < 0.1


class TestBinomialGLM:
    """Tests for Binomial (logistic) GLM."""
    
    def test_logistic_convergence(self):
        """Logistic regression should converge."""
        np.random.seed(42)
        n = 200
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        prob = 1 / (1 + np.exp(-(0 + 0.5 * X[:, 1])))
        y = np.random.binomial(1, prob)
        
        result = rs.fit_glm(y, X, family="binomial")
        
        assert result.converged
    
    def test_logistic_fitted_in_unit_interval(self):
        """Logistic fitted values should be in (0, 1)."""
        np.random.seed(42)
        n = 100
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.random.binomial(1, 0.5, size=n).astype(float)
        
        result = rs.fit_glm(y, X, family="binomial")
        
        assert np.all(result.fittedvalues > 0)
        assert np.all(result.fittedvalues < 1)
    
    def test_balanced_data(self):
        """With balanced data (50/50), intercept should be near 0."""
        np.random.seed(42)
        n = 200
        X = np.ones((n, 1))  # Intercept only
        y = np.array([0] * 100 + [1] * 100, dtype=float)
        np.random.shuffle(y)
        
        result = rs.fit_glm(y, X, family="binomial")
        
        # Intercept should be close to 0 (log-odds of 50% = 0)
        assert abs(result.params[0]) < 0.3


class TestGammaGLM:
    """Tests for Gamma GLM."""
    
    def test_gamma_convergence(self):
        """Gamma GLM should converge."""
        np.random.seed(42)
        n = 100
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        mu = np.exp(5 + 0.2 * X[:, 1])
        shape = 2.0
        y = np.random.gamma(shape, mu / shape)
        
        result = rs.fit_glm(y, X, family="gamma")
        
        assert result.converged
    
    def test_gamma_fitted_positive(self):
        """Gamma fitted values should all be positive."""
        np.random.seed(42)
        n = 100
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.random.gamma(2, 50, size=n)
        
        result = rs.fit_glm(y, X, family="gamma")
        
        assert np.all(result.fittedvalues > 0)


class TestGLMResults:
    """Tests for GLMResults object."""
    
    def test_results_attributes(self):
        """GLMResults should have all expected attributes."""
        np.random.seed(42)
        n = 50
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.random.randn(n)
        
        result = rs.fit_glm(y, X, family="gaussian")
        
        # Check attributes exist
        assert hasattr(result, 'params')
        assert hasattr(result, 'fittedvalues')
        assert hasattr(result, 'deviance')
        assert hasattr(result, 'converged')
        assert hasattr(result, 'iterations')
        assert hasattr(result, 'nobs')
        assert hasattr(result, 'df_resid')
        assert hasattr(result, 'df_model')
    
    def test_results_shapes(self):
        """Results should have correct shapes."""
        np.random.seed(42)
        n = 100
        p = 3
        X = np.column_stack([np.ones(n)] + [np.random.randn(n) for _ in range(p-1)])
        y = np.random.randn(n)
        
        result = rs.fit_glm(y, X, family="gaussian")
        
        assert result.params.shape == (p,)
        assert result.fittedvalues.shape == (n,)
        assert result.bse().shape == (p,)
        assert result.tvalues().shape == (p,)
        assert result.nobs == n
        assert result.df_resid == n - p
        assert result.df_model == p - 1
    
    def test_standard_errors_positive(self):
        """Standard errors should be positive."""
        np.random.seed(42)
        n = 100
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.random.randn(n)
        
        result = rs.fit_glm(y, X, family="gaussian")
        
        assert np.all(result.bse() > 0)


class TestGLMClass:
    """Tests for the GLM class interface."""
    
    def test_glm_class_basic(self):
        """GLM class should work like fit_glm."""
        np.random.seed(42)
        n = 100
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.random.randn(n)
        
        # Fit using class
        model = rs.GLM(y, X, family="gaussian")
        result_class = model.fit()
        
        # Fit using function
        result_func = rs.fit_glm(y, X, family="gaussian")
        
        # Should give same results
        np.testing.assert_array_almost_equal(
            result_class.params, result_func.params
        )
    
    def test_glm_class_properties(self):
        """GLM class should have correct properties."""
        n = 100
        p = 3
        X = np.random.randn(n, p)
        y = np.random.randn(n)
        
        model = rs.GLM(y, X, family="gaussian")
        
        assert model.nobs == n
        assert model.df_model == p - 1
        assert model.df_resid == n - p


class TestInputValidation:
    """Tests for input validation."""
    
    def test_dimension_mismatch(self):
        """Should raise error if X and y have different n."""
        X = np.ones((10, 2))
        y = np.ones(5)  # Wrong length
        
        with pytest.raises(ValueError):
            rs.fit_glm(y, X, family="gaussian")
    
    def test_invalid_family(self):
        """Should raise error for unknown family."""
        X = np.ones((10, 2))
        y = np.ones(10)
        
        with pytest.raises(ValueError, match="Unknown family"):
            rs.fit_glm(y, X, family="invalid_family")
    
    def test_invalid_link(self):
        """Should raise error for invalid link."""
        X = np.ones((10, 2))
        y = np.ones(10)
        
        with pytest.raises(ValueError, match="Unknown link"):
            rs.fit_glm(y, X, family="gaussian", link="invalid_link")


class TestStatisticalInference:
    """Tests for p-values, confidence intervals, and summary."""
    
    def test_pvalues_shape(self):
        """P-values should have correct shape."""
        np.random.seed(42)
        n = 100
        p = 3
        X = np.column_stack([np.ones(n)] + [np.random.randn(n) for _ in range(p-1)])
        y = np.random.randn(n)
        
        result = rs.fit_glm(y, X, family="gaussian")
        
        assert result.pvalues().shape == (p,)
    
    def test_pvalues_in_range(self):
        """P-values should be between 0 and 1."""
        np.random.seed(42)
        n = 100
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.random.randn(n)
        
        result = rs.fit_glm(y, X, family="gaussian")
        pvals = result.pvalues()
        
        assert np.all(pvals >= 0)
        assert np.all(pvals <= 1)
    
    def test_significant_effect_low_pvalue(self):
        """Strong effect should have low p-value."""
        np.random.seed(42)
        n = 200
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        # Strong effect: y = 3*x + small noise
        y = 3 * X[:, 1] + np.random.randn(n) * 0.1
        
        result = rs.fit_glm(y, X, family="gaussian")
        
        # Slope p-value should be very small
        assert result.pvalues()[1] < 0.001
    
    def test_confidence_interval_shape(self):
        """Confidence intervals should have shape (p, 2)."""
        np.random.seed(42)
        n = 100
        p = 3
        X = np.column_stack([np.ones(n)] + [np.random.randn(n) for _ in range(p-1)])
        y = np.random.randn(n)
        
        result = rs.fit_glm(y, X, family="gaussian")
        ci = result.conf_int()
        
        assert ci.shape == (p, 2)
    
    def test_confidence_interval_contains_estimate(self):
        """Point estimate should be inside confidence interval."""
        np.random.seed(42)
        n = 100
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.random.randn(n)
        
        result = rs.fit_glm(y, X, family="gaussian")
        ci = result.conf_int()
        coefs = result.params
        
        for i in range(len(coefs)):
            assert ci[i, 0] <= coefs[i] <= ci[i, 1]
    
    def test_wider_ci_with_higher_alpha(self):
        """90% CI should be narrower than 99% CI."""
        np.random.seed(42)
        n = 100
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.random.randn(n)
        
        result = rs.fit_glm(y, X, family="gaussian")
        ci_90 = result.conf_int(alpha=0.10)
        ci_99 = result.conf_int(alpha=0.01)
        
        # 99% CI should be wider
        width_90 = ci_90[:, 1] - ci_90[:, 0]
        width_99 = ci_99[:, 1] - ci_99[:, 0]
        
        assert np.all(width_99 > width_90)
    
    def test_significance_codes(self):
        """Significance codes should match p-values."""
        np.random.seed(42)
        n = 200
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = 3 * X[:, 1] + np.random.randn(n) * 0.1  # Strong effect
        
        result = rs.fit_glm(y, X, family="gaussian")
        codes = result.significance_codes()
        
        assert len(codes) == 2
        # Strong effect should be highly significant
        assert codes[1] in ["***", "**", "*"]


class TestOffsetAndWeights:
    """Tests for offset and prior weights functionality."""
    
    def test_poisson_with_offset(self):
        """Poisson with exposure offset should work."""
        np.random.seed(42)
        n = 200
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        exposure = np.random.uniform(0.5, 2.0, n)
        
        # True model with exposure
        mu = exposure * np.exp(0.5 + 0.3 * X[:, 1])
        y = np.random.poisson(mu)
        
        result = rs.fit_glm(y, X, family="poisson", offset=np.log(exposure))
        
        assert result.converged
        # Coefficients should be close to true values
        assert abs(result.params[0] - 0.5) < 0.3
        assert abs(result.params[1] - 0.3) < 0.2
    
    def test_weighted_regression(self):
        """Weighted regression should converge."""
        np.random.seed(42)
        n = 100
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.random.randn(n)
        weights = np.random.uniform(1, 10, n)
        
        result = rs.fit_glm(y, X, family="gaussian", weights=weights)
        
        assert result.converged
    
    def test_offset_in_glm_class(self):
        """GLM class should support offset."""
        np.random.seed(42)
        n = 100
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        exposure = np.ones(n)
        y = np.random.poisson(2, n)
        
        model = rs.GLM(y, X, family="poisson", offset=np.log(exposure))
        result = model.fit()
        
        assert result.converged
    
    def test_weights_in_glm_class(self):
        """GLM class should support weights."""
        np.random.seed(42)
        n = 100
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.random.randn(n)
        weights = np.ones(n) * 2
        
        model = rs.GLM(y, X, family="gaussian", weights=weights)
        result = model.fit()
        
        assert result.converged


class TestPredict:
    """Tests for prediction functionality."""
    
    def test_predict_identity_link(self):
        """Prediction with identity link should be X @ beta."""
        np.random.seed(42)
        n = 100
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = 2 + 3 * X[:, 1] + np.random.randn(n) * 0.1
        
        result = rs.fit_glm(y, X, family="gaussian")
        
        # Predict on training data
        predictions = rs.glm.predict(result, X, link="identity")
        
        # Should match fitted values
        np.testing.assert_array_almost_equal(predictions, result.fittedvalues, decimal=5)
    
    def test_predict_log_link(self):
        """Prediction with log link should be exp(X @ beta)."""
        np.random.seed(42)
        n = 100
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.random.poisson(np.exp(0.5 + 0.3 * X[:, 1]))
        
        result = rs.fit_glm(y, X, family="poisson")
        
        # Predict on new data
        X_new = np.array([[1.0, 0.0], [1.0, 1.0]])
        predictions = rs.glm.predict(result, X_new, link="log")
        
        # Should be positive
        assert np.all(predictions > 0)
    
    def test_predict_logit_link(self):
        """Prediction with logit link should be in (0, 1)."""
        np.random.seed(42)
        n = 100
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        prob = 1 / (1 + np.exp(-X[:, 1]))
        y = np.random.binomial(1, prob).astype(float)
        
        result = rs.fit_glm(y, X, family="binomial")
        
        # Predict on new data
        X_new = np.array([[1.0, -2.0], [1.0, 0.0], [1.0, 2.0]])
        predictions = rs.glm.predict(result, X_new, link="logit")
        
        assert np.all(predictions > 0)
        assert np.all(predictions < 1)
    
    def test_predict_with_offset(self):
        """Prediction should support offset."""
        np.random.seed(42)
        n = 100
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.random.poisson(2, n)
        
        result = rs.fit_glm(y, X, family="poisson")
        
        # Predict with exposure
        X_new = np.array([[1.0, 0.0]])
        exposure_new = np.array([2.0])
        pred = rs.glm.predict(result, X_new, link="log", offset=np.log(exposure_new))
        
        # Prediction should be scaled by exposure
        pred_no_offset = rs.glm.predict(result, X_new, link="log")
        np.testing.assert_almost_equal(pred[0], pred_no_offset[0] * 2.0, decimal=5)


class TestSummaryFunctions:
    """Tests for summary output functions."""
    
    def test_summary_returns_string(self):
        """Summary should return a string."""
        np.random.seed(42)
        n = 50
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.random.randn(n)
        
        result = rs.fit_glm(y, X, family="gaussian")
        output = rs.summary(result)
        
        assert isinstance(output, str)
        assert len(output) > 0
    
    def test_summary_contains_key_info(self):
        """Summary should contain important information."""
        np.random.seed(42)
        n = 50
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.random.randn(n)
        
        result = rs.fit_glm(y, X, family="gaussian")
        output = rs.summary(result)
        
        assert "Observations" in output
        assert "Deviance" in output
        assert "Coef" in output
        assert "P>|z|" in output
    
    def test_summary_with_feature_names(self):
        """Summary should use provided feature names."""
        np.random.seed(42)
        n = 50
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.random.randn(n)
        
        result = rs.fit_glm(y, X, family="gaussian")
        output = rs.summary(result, feature_names=["Intercept", "MyVar"])
        
        assert "Intercept" in output
        assert "MyVar" in output
    
    def test_summary_relativities(self):
        """Relativity summary should show exp(coef)."""
        np.random.seed(42)
        n = 100
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.random.poisson(np.exp(0.5 + 0.3 * X[:, 1]))
        
        result = rs.fit_glm(y, X, family="poisson")
        output = rs.summary_relativities(result)
        
        assert "Relativity" in output
        assert "exp(Coef)" in output


class TestDiagnostics:
    """Tests for model diagnostics (residuals, dispersion, information criteria)."""
    
    def test_residual_types(self):
        """Test all residual types are computed correctly."""
        np.random.seed(42)
        n = 100
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.random.poisson(np.exp(0.5 + 0.3 * X[:, 1]))
        
        result = rs.fit_glm(y, X, family="poisson")
        
        # All residual types should be arrays of correct length
        assert len(result.resid_response()) == n
        assert len(result.resid_pearson()) == n
        assert len(result.resid_deviance()) == n
        assert len(result.resid_working()) == n
    
    def test_deviance_residuals_sum_to_deviance(self):
        """Sum of squared deviance residuals should equal model deviance."""
        np.random.seed(42)
        n = 100
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.random.poisson(np.exp(0.5 + 0.3 * X[:, 1]))
        
        result = rs.fit_glm(y, X, family="poisson")
        
        resid_dev = result.resid_deviance()
        sum_sq = np.sum(resid_dev ** 2)
        
        np.testing.assert_almost_equal(sum_sq, result.deviance, decimal=10)
    
    def test_response_residuals(self):
        """Response residuals should be y - fitted."""
        np.random.seed(42)
        n = 50
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.random.randn(n) + 5
        
        result = rs.fit_glm(y, X, family="gaussian")
        
        expected = y - result.fittedvalues
        np.testing.assert_array_almost_equal(result.resid_response(), expected)
    
    def test_log_likelihood(self):
        """Log-likelihood should be a finite negative number."""
        np.random.seed(42)
        n = 100
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.random.poisson(np.exp(0.5 + 0.3 * X[:, 1]))
        
        result = rs.fit_glm(y, X, family="poisson")
        
        llf = result.llf()
        assert np.isfinite(llf)
        assert llf < 0  # Log-likelihood is typically negative
    
    def test_aic_bic(self):
        """AIC and BIC should be finite positive numbers."""
        np.random.seed(42)
        n = 100
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.random.poisson(np.exp(0.5 + 0.3 * X[:, 1]))
        
        result = rs.fit_glm(y, X, family="poisson")
        
        assert np.isfinite(result.aic())
        assert np.isfinite(result.bic())
        assert result.aic() > 0
        assert result.bic() > 0
        # BIC penalizes more for large n, so BIC > AIC typically
        # AIC = -2*llf + 2*p, BIC = -2*llf + p*log(n)
        # For n=100, log(100) ≈ 4.6 > 2, so BIC > AIC
        assert result.bic() > result.aic()
    
    def test_null_deviance(self):
        """Null deviance should be >= residual deviance."""
        np.random.seed(42)
        n = 100
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.random.poisson(np.exp(0.5 + 0.3 * X[:, 1]))
        
        result = rs.fit_glm(y, X, family="poisson")
        
        null_dev = result.null_deviance()
        resid_dev = result.deviance
        
        assert np.isfinite(null_dev)
        assert null_dev >= resid_dev  # Adding predictors should reduce deviance
    
    def test_pearson_chi2(self):
        """Pearson chi-squared should be finite and positive."""
        np.random.seed(42)
        n = 100
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.random.poisson(np.exp(0.5 + 0.3 * X[:, 1]))
        
        result = rs.fit_glm(y, X, family="poisson")
        
        chi2 = result.pearson_chi2()
        assert np.isfinite(chi2)
        assert chi2 > 0
    
    def test_scale_estimates(self):
        """Scale estimates should be reasonable."""
        np.random.seed(42)
        n = 100
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.random.poisson(np.exp(0.5 + 0.3 * X[:, 1]))
        
        result = rs.fit_glm(y, X, family="poisson")
        
        scale_dev = result.scale()
        scale_pear = result.scale_pearson()
        
        assert np.isfinite(scale_dev)
        assert np.isfinite(scale_pear)
        assert scale_dev > 0
        assert scale_pear > 0
        # For Poisson with correct specification, scale should be near 1
        # But with real data it might be > 1 (overdispersion)
    
    def test_family_attribute(self):
        """Family attribute should return correct family name."""
        np.random.seed(42)
        n = 50
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        
        # Test each family
        for family_name, expected in [
            ("gaussian", "Gaussian"),
            ("poisson", "Poisson"),
            ("binomial", "Binomial"),
            ("gamma", "Gamma"),
        ]:
            if family_name == "binomial":
                y = np.random.binomial(1, 0.5, size=n).astype(float)
            elif family_name == "gamma":
                y = np.random.gamma(2, 2, size=n)
            elif family_name == "poisson":
                y = np.random.poisson(2, size=n).astype(float)
            else:
                y = np.random.randn(n)
            
            result = rs.fit_glm(y, X, family=family_name)
            assert result.family == expected
    
    def test_gaussian_diagnostics(self):
        """Test diagnostics for Gaussian family specifically."""
        np.random.seed(42)
        n = 100
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = 2 + 3 * X[:, 1] + np.random.randn(n)
        
        result = rs.fit_glm(y, X, family="gaussian")
        
        # For Gaussian with identity link:
        # - Response residuals = Pearson residuals (since V(mu) = 1)
        # - Deviance residuals = response residuals (since d = (y-mu)^2)
        np.testing.assert_array_almost_equal(
            result.resid_response(), 
            result.resid_pearson()
        )
    
    def test_enhanced_summary_contains_diagnostics(self):
        """Summary should include diagnostic information."""
        np.random.seed(42)
        n = 100
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.random.poisson(np.exp(0.5 + 0.3 * X[:, 1]))
        
        result = rs.fit_glm(y, X, family="poisson")
        output = rs.summary(result)
        
        # Check for diagnostic metrics in summary
        assert "Log-Likelihood" in output
        assert "AIC" in output
        assert "BIC" in output
        assert "Null Deviance" in output
        assert "Pearson chi2" in output
        assert "Scale" in output
        assert "Poisson" in output  # Family name


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
