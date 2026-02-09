// =============================================================================
// GLM Results
// =============================================================================
//
// This class holds the results of fitting a GLM.
// It provides access to coefficients, fitted values, and diagnostic info.
// =============================================================================

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{IntoPyArray, PyArray1, PyArray2};
use ndarray::{Array1, Array2};

use rustystats_core::families::Family;
use rustystats_core::inference::{pvalue_z, confidence_interval_z, HCType, robust_covariance, robust_standard_errors};
use rustystats_core::diagnostics::{
    resid_response, resid_pearson, resid_deviance, resid_working,
    estimate_dispersion_pearson, pearson_chi2,
    aic, bic, null_deviance_for_family,
};
use rustystats_core::regularization::Penalty;

use crate::families_py::family_from_name;

/// Results from fitting a GLM.
///
/// Contains coefficients, fitted values, deviance, and diagnostic information.
/// Use this to make predictions and assess model fit.
#[pyclass(name = "GLMResults")]
#[derive(Clone)]
pub struct PyGLMResults {
    /// Fitted coefficients
    pub(crate) coefficients: Array1<f64>,
    /// Fitted values (predictions on response scale)
    pub(crate) fitted_values: Array1<f64>,
    /// Linear predictor η = Xβ
    pub(crate) linear_predictor: Array1<f64>,
    /// Model deviance
    pub(crate) deviance: f64,
    /// Number of IRLS iterations
    pub(crate) iterations: usize,
    /// Did the algorithm converge?
    pub(crate) converged: bool,
    /// Unscaled covariance matrix (X'WX)⁻¹
    pub(crate) covariance_unscaled: Array2<f64>,
    /// Number of observations
    pub(crate) n_obs: usize,
    /// Number of parameters
    pub(crate) n_params: usize,
    /// Original response variable (for residuals)
    pub(crate) y: Array1<f64>,
    /// Family name (for diagnostics)
    pub(crate) family_name: String,
    /// Prior weights
    pub(crate) prior_weights: Array1<f64>,
    /// Regularization penalty applied (if any)
    pub(crate) penalty: Penalty,
    /// Design matrix X (for robust standard errors).
    /// Optional to allow lean mode — omit to save memory at scale.
    pub(crate) design_matrix: Option<Array2<f64>>,
    /// IRLS weights (for robust standard errors)
    pub(crate) irls_weights: Array1<f64>,
    /// Offset values (e.g., log(exposure) for count models)
    pub(crate) offset: Option<Array1<f64>>,
}

// =============================================================================
// Helper Methods (not exposed to Python)
// =============================================================================

impl PyGLMResults {
    /// Get the appropriate Family trait object based on family_name.
    /// Used internally by diagnostics and robust SE methods.
    /// Note: family_name is validated at model creation, so this should never fail.
    fn get_family(&self) -> Box<dyn Family> {
        family_from_name(&self.family_name, 1.5, 1.0)
            .expect("Invalid family name stored in results - this is a bug")
    }
    
    /// Get prior weights as Option, returning None if all weights are 1.0.
    /// Many functions accept Option<&Array1<f64>> for weights.
    fn maybe_weights(&self) -> Option<&Array1<f64>> {
        if self.prior_weights.iter().all(|&w| (w - 1.0).abs() < rustystats_core::constants::ZERO_TOL) {
            None
        } else {
            Some(&self.prior_weights)
        }
    }
    
    /// Compute robust covariance matrix (internal helper).
    /// Factored out to avoid repeating the same logic in cov_robust, bse_robust, etc.
    /// Returns Err if design_matrix was not stored (lean mode).
    fn compute_robust_cov(&self, hc_type: HCType) -> PyResult<Array2<f64>> {
        let dm = self.design_matrix.as_ref().ok_or_else(|| {
            PyValueError::new_err(
                "Design matrix not stored (lean mode). Robust standard errors require \
                 store_design_matrix=True when fitting, or pass the design matrix via \
                 the Python diagnostics API."
            )
        })?;
        let family = self.get_family();
        let pearson_resid = resid_pearson(&self.y, &self.fitted_values, family.as_ref());
        
        Ok(robust_covariance(
            dm,
            &pearson_resid,
            &self.irls_weights,
            &self.prior_weights,
            &self.covariance_unscaled,
            hc_type,
        ))
    }
}

#[pymethods]
impl PyGLMResults {
    /// Get the fitted coefficients (β).
    ///
    /// These are the parameter estimates from the model.
    /// For log link: exp(β) gives multiplicative effects (relativities).
    /// For logit link: exp(β) gives odds ratios.
    #[getter]
    fn params<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.coefficients.clone().into_pyarray_bound(py)
    }

    /// Alias for params (statsmodels compatibility).
    #[getter]
    fn coefficients<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.coefficients.clone().into_pyarray_bound(py)
    }

    /// Get the fitted values μ = g⁻¹(Xβ).
    ///
    /// These are the predicted means on the response scale.
    #[getter]
    fn fittedvalues<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.fitted_values.clone().into_pyarray_bound(py)
    }

    /// Get the linear predictor η = Xβ.
    #[getter]
    fn linear_predictor<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.linear_predictor.clone().into_pyarray_bound(py)
    }

    /// Get the model deviance.
    ///
    /// Lower deviance indicates better fit.
    /// Use for comparing nested models (likelihood ratio test).
    #[getter]
    fn deviance(&self) -> f64 {
        self.deviance
    }

    /// Number of iterations until convergence.
    #[getter]
    fn iterations(&self) -> usize {
        self.iterations
    }

    /// Did the fitting algorithm converge?
    #[getter]
    fn converged(&self) -> bool {
        self.converged
    }

    /// Number of observations.
    #[getter]
    fn nobs(&self) -> usize {
        self.n_obs
    }

    /// Degrees of freedom for residuals (n - p).
    #[getter]
    fn df_resid(&self) -> usize {
        self.n_obs.saturating_sub(self.n_params)
    }

    /// Degrees of freedom for model (p - 1, excluding intercept).
    #[getter]
    fn df_model(&self) -> usize {
        self.n_params.saturating_sub(1)
    }

    /// Get the unscaled covariance matrix (X'WX)⁻¹.
    ///
    /// Multiply by dispersion φ to get Var(β̂).
    /// For Poisson/Binomial, φ = 1.
    /// For Gaussian/Gamma, estimate φ from residuals.
    #[getter]
    fn cov_params_unscaled<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.covariance_unscaled.clone().into_pyarray_bound(py)
    }

    /// Get the design matrix X used in fitting.
    ///
    /// Returns None if store_design_matrix=False was used (lean mode).
    /// This is useful for computing score tests for unfitted factors.
    #[getter]
    fn get_design_matrix<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        self.design_matrix.as_ref().map(|dm| dm.clone().into_pyarray_bound(py))
    }

    /// Get the IRLS working weights from the final iteration.
    ///
    /// These are the diagonal elements of the weight matrix W in the
    /// weighted least squares problem: (X'WX)β = X'Wz.
    /// Useful for computing score tests for unfitted factors.
    #[getter]
    fn get_irls_weights<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.irls_weights.clone().into_pyarray_bound(py)
    }

    /// Get the dispersion parameter φ for standard error computation.
    ///
    /// For Poisson/Binomial: φ = 1 (fixed by assumption)
    /// For QuasiPoisson/QuasiBinomial: φ = Pearson χ² / df_resid (estimated)
    /// For Gamma/Gaussian/Tweedie: φ = Pearson χ² / df_resid (estimated)
    ///
    /// This matches statsmodels default behavior for SE computation.
    /// Note: For log-likelihood/AIC, deviance-based scale is used separately.
    fn scale(&self) -> f64 {
        let family = self.get_family();
        if family.fixed_dispersion() {
            1.0
        } else {
            estimate_dispersion_pearson(
                &self.y,
                &self.fitted_values,
                family.as_ref(),
                self.df_resid(),
                self.maybe_weights(),
            )
        }
    }

    /// Get standard errors of coefficients.
    ///
    /// SE(β̂) = sqrt(diag(φ × (X'WX)⁻¹))
    fn bse<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let scale = self.scale();
        let se: Array1<f64> = (0..self.n_params)
            .map(|i| (scale * self.covariance_unscaled[[i, i]]).sqrt())
            .collect();
        se.into_pyarray_bound(py)
    }

    /// Get z-statistics (or t-statistics) for coefficients.
    ///
    /// z = β̂ / SE(β̂)
    fn tvalues<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let scale = self.scale();
        let t: Array1<f64> = (0..self.n_params)
            .map(|i| {
                let se = (scale * self.covariance_unscaled[[i, i]]).sqrt();
                if se > 1e-10 {
                    self.coefficients[i] / se
                } else {
                    0.0
                }
            })
            .collect();
        t.into_pyarray_bound(py)
    }

    /// Get p-values for coefficients.
    ///
    /// Tests the null hypothesis that each coefficient equals zero.
    /// Uses the z-distribution (appropriate for GLMs with known dispersion
    /// or large samples).
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     Array of p-values, one for each coefficient.
    ///
    /// Interpretation
    /// --------------
    /// - p < 0.05: Coefficient is significantly different from zero at 5% level
    /// - p < 0.01: Highly significant
    /// - p < 0.001: Very highly significant
    ///
    /// Note: Small p-values indicate statistical significance, not practical
    /// importance. Always consider the magnitude of coefficients too!
    fn pvalues<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let scale = self.scale();
        let pvals: Array1<f64> = (0..self.n_params)
            .map(|i| {
                let se = (scale * self.covariance_unscaled[[i, i]]).sqrt();
                if se > 1e-10 {
                    let z = self.coefficients[i] / se;
                    pvalue_z(z)
                } else {
                    f64::NAN
                }
            })
            .collect();
        pvals.into_pyarray_bound(py)
    }

    /// Get confidence intervals for coefficients.
    ///
    /// Parameters
    /// ----------
    /// alpha : float, optional
    ///     Significance level. Default 0.05 gives 95% CI.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     2D array of shape (n_params, 2) with [lower, upper] bounds.
    ///
    /// Interpretation
    /// --------------
    /// We are (1-alpha)% confident that the true parameter value
    /// lies within this interval.
    ///
    /// For log link models, use np.exp(conf_int) to get relativities.
    #[pyo3(signature = (alpha=0.05))]
    fn conf_int<'py>(&self, py: Python<'py>, alpha: f64) -> Bound<'py, PyArray2<f64>> {
        let scale = self.scale();
        let confidence = 1.0 - alpha;
        
        let mut ci = Array2::zeros((self.n_params, 2));
        for i in 0..self.n_params {
            let se = (scale * self.covariance_unscaled[[i, i]]).sqrt();
            let (lower, upper) = confidence_interval_z(self.coefficients[i], se, confidence);
            ci[[i, 0]] = lower;
            ci[[i, 1]] = upper;
        }
        ci.into_pyarray_bound(py)
    }

    /// Get significance codes for p-values.
    ///
    /// Returns a list of significance codes:
    /// - "***" : p < 0.001
    /// - "**"  : p < 0.01
    /// - "*"   : p < 0.05
    /// - "."   : p < 0.1
    /// - ""    : p >= 0.1
    fn significance_codes(&self) -> Vec<String> {
        let scale = self.scale();
        (0..self.n_params)
            .map(|i| {
                let se = (scale * self.covariance_unscaled[[i, i]]).sqrt();
                if se > 1e-10 {
                    let z = self.coefficients[i] / se;
                    let p = pvalue_z(z);
                    if p < 0.001 {
                        "***".to_string()
                    } else if p < 0.01 {
                        "**".to_string()
                    } else if p < 0.05 {
                        "*".to_string()
                    } else if p < 0.1 {
                        ".".to_string()
                    } else {
                        "".to_string()
                    }
                } else {
                    "".to_string()
                }
            })
            .collect()
    }

    // =========================================================================
    // Robust Standard Errors (Sandwich Estimators)
    // =========================================================================

    /// Get robust (HC) covariance matrix.
    ///
    /// Uses the sandwich estimator which is valid even when the variance
    /// function is misspecified (heteroscedasticity).
    ///
    /// Parameters
    /// ----------
    /// cov_type : str, optional
    ///     Type of robust covariance. Options:
    ///     - "HC0": No small-sample correction (default)
    ///     - "HC1": Degrees of freedom correction (n/(n-p))
    ///     - "HC2": Leverage-adjusted
    ///     - "HC3": Jackknife-like (most conservative)
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     Robust covariance matrix (p × p)
    #[pyo3(signature = (cov_type="HC1"))]
    fn cov_robust<'py>(&self, py: Python<'py>, cov_type: &str) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let hc_type = HCType::from_str(cov_type).ok_or_else(|| {
            PyValueError::new_err(format!(
                "Unknown cov_type '{}'. Use 'HC0', 'HC1', 'HC2', or 'HC3'.", cov_type
            ))
        })?;
        
        let cov = self.compute_robust_cov(hc_type)?;
        Ok(cov.into_pyarray_bound(py))
    }

    /// Get robust standard errors of coefficients (HC/sandwich estimator).
    ///
    /// Unlike model-based standard errors that assume correct variance
    /// specification, robust standard errors are valid under
    /// heteroscedasticity.
    ///
    /// Parameters
    /// ----------
    /// cov_type : str, optional
    ///     Type of robust covariance. Options:
    ///     - "HC0": No small-sample correction
    ///     - "HC1": Degrees of freedom correction (default, recommended)
    ///     - "HC2": Leverage-adjusted
    ///     - "HC3": Jackknife-like (most conservative)
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     Array of robust standard errors, one for each coefficient.
    ///
    /// Notes
    /// -----
    /// HC1 is the default as it applies the standard n/(n-p) degrees of
    /// freedom correction, which is what most users expect.
    ///
    /// HC3 gives larger standard errors and is often recommended for
    /// small samples or when there are influential observations.
    ///
    /// Example
    /// -------
    /// >>> result = rs.glm("y ~ x1 + x2", data, family="poisson").fit()
    /// >>> se_model = result.bse()       # Model-based SE
    /// >>> se_robust = result.bse_robust("HC1")  # Robust SE
    #[pyo3(signature = (cov_type="HC1"))]
    fn bse_robust<'py>(&self, py: Python<'py>, cov_type: &str) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let hc_type = HCType::from_str(cov_type).ok_or_else(|| {
            PyValueError::new_err(format!(
                "Unknown cov_type '{}'. Use 'HC0', 'HC1', 'HC2', or 'HC3'.", cov_type
            ))
        })?;
        
        let cov = self.compute_robust_cov(hc_type)?;
        let se = robust_standard_errors(&cov);
        Ok(se.into_pyarray_bound(py))
    }

    /// Get z/t statistics using robust standard errors.
    ///
    /// Parameters
    /// ----------
    /// cov_type : str, optional
    ///     Type of robust covariance. Default "HC1".
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     Array of t/z statistics (coefficient / robust SE).
    #[pyo3(signature = (cov_type="HC1"))]
    fn tvalues_robust<'py>(&self, py: Python<'py>, cov_type: &str) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let hc_type = HCType::from_str(cov_type).ok_or_else(|| {
            PyValueError::new_err(format!(
                "Unknown cov_type '{}'. Use 'HC0', 'HC1', 'HC2', or 'HC3'.", cov_type
            ))
        })?;
        
        let cov = self.compute_robust_cov(hc_type)?;
        let se = robust_standard_errors(&cov);
        let t: Array1<f64> = self.coefficients.iter()
            .zip(se.iter())
            .map(|(&c, &s)| if s > 1e-10 { c / s } else { 0.0 })
            .collect();
        
        Ok(t.into_pyarray_bound(py))
    }

    /// Get p-values using robust standard errors.
    ///
    /// Parameters
    /// ----------
    /// cov_type : str, optional
    ///     Type of robust covariance. Default "HC1".
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     Array of p-values.
    #[pyo3(signature = (cov_type="HC1"))]
    fn pvalues_robust<'py>(&self, py: Python<'py>, cov_type: &str) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let hc_type = HCType::from_str(cov_type).ok_or_else(|| {
            PyValueError::new_err(format!(
                "Unknown cov_type '{}'. Use 'HC0', 'HC1', 'HC2', or 'HC3'.", cov_type
            ))
        })?;
        
        let cov = self.compute_robust_cov(hc_type)?;
        let se = robust_standard_errors(&cov);
        let pvals: Array1<f64> = self.coefficients.iter()
            .zip(se.iter())
            .map(|(&c, &s)| {
                if s > 1e-10 {
                    let z = c / s;
                    pvalue_z(z)
                } else {
                    f64::NAN
                }
            })
            .collect();
        
        Ok(pvals.into_pyarray_bound(py))
    }

    /// Get confidence intervals using robust standard errors.
    ///
    /// Parameters
    /// ----------
    /// alpha : float, optional
    ///     Significance level. Default 0.05 gives 95% CI.
    /// cov_type : str, optional
    ///     Type of robust covariance. Default "HC1".
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     2D array of shape (n_params, 2) with [lower, upper] bounds.
    #[pyo3(signature = (alpha=0.05, cov_type="HC1"))]
    fn conf_int_robust<'py>(&self, py: Python<'py>, alpha: f64, cov_type: &str) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let hc_type = HCType::from_str(cov_type).ok_or_else(|| {
            PyValueError::new_err(format!(
                "Unknown cov_type '{}'. Use 'HC0', 'HC1', 'HC2', or 'HC3'.", cov_type
            ))
        })?;
        
        let cov = self.compute_robust_cov(hc_type)?;
        let se = robust_standard_errors(&cov);
        let confidence = 1.0 - alpha;
        
        let mut ci = Array2::zeros((self.n_params, 2));
        for (i, (&coef, &se_i)) in self.coefficients.iter().zip(se.iter()).enumerate() {
            let (lower, upper) = confidence_interval_z(coef, se_i, confidence);
            ci[[i, 0]] = lower;
            ci[[i, 1]] = upper;
        }
        
        Ok(ci.into_pyarray_bound(py))
    }

    // =========================================================================
    // Residuals (statsmodels-compatible)
    // =========================================================================

    /// Get response residuals: y - μ
    ///
    /// Simple difference between observed and predicted values.
    /// Not standardized.
    fn resid_response<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let resid = resid_response(&self.y, &self.fitted_values);
        resid.into_pyarray_bound(py)
    }

    /// Get Pearson residuals: (y - μ) / √V(μ)
    ///
    /// Standardized residuals that account for the variance function.
    /// For a well-specified model, should have approximately constant variance.
    fn resid_pearson<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let family = self.get_family();
        let resid = resid_pearson(&self.y, &self.fitted_values, family.as_ref());
        resid.into_pyarray_bound(py)
    }

    /// Get deviance residuals: sign(y - μ) × √d_i
    ///
    /// Based on the unit deviance contributions. Often more normally
    /// distributed than Pearson residuals for non-Gaussian families.
    /// sum(resid_deviance²) = model deviance
    fn resid_deviance<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let family = self.get_family();
        let resid = resid_deviance(&self.y, &self.fitted_values, family.as_ref());
        resid.into_pyarray_bound(py)
    }

    /// Get working residuals: (y - μ) × g'(μ)
    ///
    /// Used internally by IRLS. On the scale of the linear predictor.
    /// Useful for understanding the fitting process.
    fn resid_working<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let family = self.get_family();
        let link = family.default_link();
        let resid = resid_working(&self.y, &self.fitted_values, link.as_ref());
        resid.into_pyarray_bound(py)
    }

    // =========================================================================
    // Dispersion and Scale
    // =========================================================================

    /// Get Pearson chi-squared statistic.
    ///
    /// X² = Σ(y-μ)²/V(μ)
    ///
    /// For a well-specified model with known dispersion φ=1,
    /// X² should be approximately chi-squared with (n-p) df.
    fn pearson_chi2(&self) -> f64 {
        let family = self.get_family();
        pearson_chi2(&self.y, &self.fitted_values, family.as_ref(), self.maybe_weights())
    }

    /// Get dispersion estimated from Pearson residuals.
    ///
    /// φ_pearson = X² / (n - p)
    fn scale_pearson(&self) -> f64 {
        let family = self.get_family();
        estimate_dispersion_pearson(
            &self.y, 
            &self.fitted_values, 
            family.as_ref(), 
            self.df_resid(),
            self.maybe_weights(),
        )
    }

    // =========================================================================
    // Log-Likelihood and Information Criteria
    // =========================================================================

    /// Get the log-likelihood value.
    ///
    /// This is the log of the probability of observing the data given
    /// the fitted model. Higher (less negative) is better.
    fn llf(&self) -> f64 {
        let family = self.get_family();
        let scale = self.scale();
        let weights = self.maybe_weights();
        family.log_likelihood(&self.y, &self.fitted_values, scale, weights)
    }

    /// Get the Akaike Information Criterion.
    ///
    /// AIC = -2ℓ + 2p
    ///
    /// Lower is better. Use for model comparison.
    fn aic(&self) -> f64 {
        aic(self.llf(), self.n_params)
    }

    /// Get the Bayesian Information Criterion.
    ///
    /// BIC = -2ℓ + p×log(n)
    ///
    /// Lower is better. Penalizes complexity more than AIC for large n.
    fn bic(&self) -> f64 {
        bic(self.llf(), self.n_params, self.n_obs)
    }

    /// Get the null deviance (deviance of intercept-only model).
    ///
    /// Measures total variation in y before accounting for predictors.
    /// Compare to residual deviance to assess explanatory power.
    /// Accounts for offset if present (e.g., exposure in count models).
    fn null_deviance(&self) -> f64 {
        let family = self.get_family();
        null_deviance_for_family(&self.y, family.as_ref(), self.maybe_weights(), self.offset.as_ref())
    }

    /// Get the family name.
    #[getter]
    fn family(&self) -> &str {
        &self.family_name
    }

    // =========================================================================
    // Regularization Information
    // =========================================================================

    /// Get the regularization strength (alpha/lambda).
    ///
    /// Returns 0.0 for unregularized models.
    #[getter]
    fn alpha(&self) -> f64 {
        self.penalty.lambda()
    }

    /// Get the L1 ratio for Elastic Net.
    ///
    /// Returns:
    /// - None: unregularized or pure Ridge
    /// - 1.0: pure Lasso
    /// - 0.0-1.0: Elastic Net mix
    #[getter]
    fn l1_ratio(&self) -> Option<f64> {
        match &self.penalty {
            Penalty::None => None,
            Penalty::Ridge(_) => Some(0.0),
            Penalty::Lasso(_) => Some(1.0),
            Penalty::ElasticNet { l1_ratio, .. } => Some(*l1_ratio),
            Penalty::Smooth(_) => None,  // Smooth penalties don't have L1 ratio
        }
    }

    /// Get the penalty type as a string.
    ///
    /// Returns "none", "ridge", "lasso", or "elasticnet".
    #[getter]
    fn penalty_type(&self) -> &str {
        match &self.penalty {
            Penalty::None => "none",
            Penalty::Ridge(_) => "ridge",
            Penalty::Lasso(_) => "lasso",
            Penalty::ElasticNet { .. } => "elasticnet",
            Penalty::Smooth(_) => "smooth",
        }
    }

    /// Check if this is a regularized model.
    #[getter]
    fn is_regularized(&self) -> bool {
        !self.penalty.is_none()
    }

    /// Get the number of non-zero coefficients.
    ///
    /// Useful for Lasso/Elastic Net to see how many variables were selected.
    /// Excludes the intercept (first coefficient) from the count.
    fn n_nonzero(&self) -> usize {
        self.coefficients.iter().skip(1).filter(|&&c| c.abs() > 1e-10).count()
    }

    /// Get indices of non-zero coefficients (selected variables).
    ///
    /// For Lasso/Elastic Net, this shows which variables were retained.
    fn selected_features(&self) -> Vec<usize> {
        self.coefficients
            .iter()
            .enumerate()
            .skip(1)  // Skip intercept
            .filter(|(_, &c)| c.abs() > 1e-10)
            .map(|(i, _)| i)
            .collect()
    }
}
