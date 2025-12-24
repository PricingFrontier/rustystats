// =============================================================================
// RustyStats Python Bindings
// =============================================================================
//
// This module creates the bridge between Rust and Python using PyO3.
// It wraps the pure Rust code from `rustystats-core` and exposes it as
// a Python module that can be imported with `import rustystats`.
//
// HOW THIS WORKS:
// ---------------
// 1. PyO3 lets us define Python classes and functions in Rust
// 2. When Python imports the module, it loads the compiled Rust code
// 3. Python objects get converted to/from Rust types automatically
//
// STRUCTURE:
// ----------
// - `#[pymodule]` marks the main entry point for Python
// - `#[pyclass]` marks Rust structs that become Python classes
// - `#[pymethods]` marks methods that Python can call
// - `#[pyfunction]` marks standalone functions
//
// FOR MAINTAINERS:
// ----------------
// When adding new functionality:
// 1. Implement the logic in `rustystats-core` first
// 2. Create a Python wrapper here that calls the Rust code
// 3. Add it to the module in the `rustystats` function at the bottom
//
// =============================================================================

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use ndarray::{Array1, Array2};

// Import our core library
use rustystats_core::families::{Family, GaussianFamily, PoissonFamily, BinomialFamily, GammaFamily, TweedieFamily};
use rustystats_core::links::{Link, IdentityLink, LogLink, LogitLink};
use rustystats_core::solvers::{fit_glm_full, fit_glm_regularized, fit_glm_coordinate_descent, IRLSConfig, IRLSResult};
use rustystats_core::regularization::{Penalty, RegularizationConfig};
use rustystats_core::inference::{pvalue_z, confidence_interval_z, HCType, robust_covariance, robust_standard_errors};
use rustystats_core::diagnostics::{
    resid_response, resid_pearson, resid_deviance, resid_working,
    estimate_dispersion_pearson, pearson_chi2,
    log_likelihood_gaussian, log_likelihood_poisson, log_likelihood_binomial, log_likelihood_gamma,
    aic, bic, null_deviance,
};

// =============================================================================
// Link Function Wrappers
// =============================================================================
//
// These wrap the Rust link functions so Python can use them.
// Each class provides the same interface: link(), inverse(), derivative()
// =============================================================================

/// Identity link function: η = μ
/// 
/// The simplest link - no transformation at all.
/// Default for Gaussian family (linear regression).
#[pyclass(name = "IdentityLink")]
#[derive(Clone)]
pub struct PyIdentityLink {
    inner: IdentityLink,
}

#[pymethods]
impl PyIdentityLink {
    #[new]
    fn new() -> Self {
        Self { inner: IdentityLink }
    }
    
    /// Get the name of this link function
    fn name(&self) -> &str {
        self.inner.name()
    }
    
    /// Apply link function: η = g(μ)
    fn link<'py>(&self, py: Python<'py>, mu: PyReadonlyArray1<f64>) -> Bound<'py, PyArray1<f64>> {
        let mu_array = mu.as_array().to_owned();
        let result = self.inner.link(&mu_array);
        result.into_pyarray_bound(py)
    }
    
    /// Apply inverse link: μ = g⁻¹(η)
    fn inverse<'py>(&self, py: Python<'py>, eta: PyReadonlyArray1<f64>) -> Bound<'py, PyArray1<f64>> {
        let eta_array = eta.as_array().to_owned();
        let result = self.inner.inverse(&eta_array);
        result.into_pyarray_bound(py)
    }
    
    /// Compute derivative: dη/dμ
    fn derivative<'py>(&self, py: Python<'py>, mu: PyReadonlyArray1<f64>) -> Bound<'py, PyArray1<f64>> {
        let mu_array = mu.as_array().to_owned();
        let result = self.inner.derivative(&mu_array);
        result.into_pyarray_bound(py)
    }
}

/// Log link function: η = log(μ)
/// 
/// Ensures predictions are always positive.
/// Default for Poisson (counts) and Gamma (severity) families.
#[pyclass(name = "LogLink")]
#[derive(Clone)]
pub struct PyLogLink {
    inner: LogLink,
}

#[pymethods]
impl PyLogLink {
    #[new]
    fn new() -> Self {
        Self { inner: LogLink }
    }
    
    fn name(&self) -> &str {
        self.inner.name()
    }
    
    fn link<'py>(&self, py: Python<'py>, mu: PyReadonlyArray1<f64>) -> Bound<'py, PyArray1<f64>> {
        let mu_array = mu.as_array().to_owned();
        let result = self.inner.link(&mu_array);
        result.into_pyarray_bound(py)
    }
    
    fn inverse<'py>(&self, py: Python<'py>, eta: PyReadonlyArray1<f64>) -> Bound<'py, PyArray1<f64>> {
        let eta_array = eta.as_array().to_owned();
        let result = self.inner.inverse(&eta_array);
        result.into_pyarray_bound(py)
    }
    
    fn derivative<'py>(&self, py: Python<'py>, mu: PyReadonlyArray1<f64>) -> Bound<'py, PyArray1<f64>> {
        let mu_array = mu.as_array().to_owned();
        let result = self.inner.derivative(&mu_array);
        result.into_pyarray_bound(py)
    }
}

/// Logit link function: η = log(μ/(1-μ))
/// 
/// Transforms probabilities to log-odds scale.
/// Default for Binomial family (logistic regression).
#[pyclass(name = "LogitLink")]
#[derive(Clone)]
pub struct PyLogitLink {
    inner: LogitLink,
}

#[pymethods]
impl PyLogitLink {
    #[new]
    fn new() -> Self {
        Self { inner: LogitLink }
    }
    
    fn name(&self) -> &str {
        self.inner.name()
    }
    
    fn link<'py>(&self, py: Python<'py>, mu: PyReadonlyArray1<f64>) -> Bound<'py, PyArray1<f64>> {
        let mu_array = mu.as_array().to_owned();
        let result = self.inner.link(&mu_array);
        result.into_pyarray_bound(py)
    }
    
    fn inverse<'py>(&self, py: Python<'py>, eta: PyReadonlyArray1<f64>) -> Bound<'py, PyArray1<f64>> {
        let eta_array = eta.as_array().to_owned();
        let result = self.inner.inverse(&eta_array);
        result.into_pyarray_bound(py)
    }
    
    fn derivative<'py>(&self, py: Python<'py>, mu: PyReadonlyArray1<f64>) -> Bound<'py, PyArray1<f64>> {
        let mu_array = mu.as_array().to_owned();
        let result = self.inner.derivative(&mu_array);
        result.into_pyarray_bound(py)
    }
}

// =============================================================================
// Family Wrappers
// =============================================================================
//
// These wrap the Rust distribution families for Python.
// Each provides: variance(), unit_deviance(), deviance()
// =============================================================================

/// Gaussian (Normal) family for continuous response data.
/// 
/// Use for standard linear regression.
/// Variance function: V(μ) = 1 (constant variance)
#[pyclass(name = "GaussianFamily")]
#[derive(Clone)]
pub struct PyGaussianFamily {
    inner: GaussianFamily,
}

#[pymethods]
impl PyGaussianFamily {
    #[new]
    fn new() -> Self {
        Self { inner: GaussianFamily }
    }
    
    fn name(&self) -> &str {
        self.inner.name()
    }
    
    /// Compute variance function V(μ). Returns array of 1s.
    fn variance<'py>(&self, py: Python<'py>, mu: PyReadonlyArray1<f64>) -> Bound<'py, PyArray1<f64>> {
        let mu_array = mu.as_array().to_owned();
        let result = self.inner.variance(&mu_array);
        result.into_pyarray_bound(py)
    }
    
    /// Compute unit deviance: (y - μ)² for each observation.
    fn unit_deviance<'py>(
        &self,
        py: Python<'py>,
        y: PyReadonlyArray1<f64>,
        mu: PyReadonlyArray1<f64>,
    ) -> Bound<'py, PyArray1<f64>> {
        let y_array = y.as_array().to_owned();
        let mu_array = mu.as_array().to_owned();
        let result = self.inner.unit_deviance(&y_array, &mu_array);
        result.into_pyarray_bound(py)
    }
    
    /// Compute total deviance (sum of squared residuals).
    fn deviance(&self, y: PyReadonlyArray1<f64>, mu: PyReadonlyArray1<f64>) -> f64 {
        let y_array = y.as_array().to_owned();
        let mu_array = mu.as_array().to_owned();
        self.inner.deviance(&y_array, &mu_array, None)
    }
    
    /// Get the default link function (Identity for Gaussian).
    fn default_link(&self) -> PyIdentityLink {
        PyIdentityLink::new()
    }
}

/// Poisson family for count data.
/// 
/// Use for claim frequency, event counts, etc.
/// Variance function: V(μ) = μ (variance equals mean)
#[pyclass(name = "PoissonFamily")]
#[derive(Clone)]
pub struct PyPoissonFamily {
    inner: PoissonFamily,
}

#[pymethods]
impl PyPoissonFamily {
    #[new]
    fn new() -> Self {
        Self { inner: PoissonFamily }
    }
    
    fn name(&self) -> &str {
        self.inner.name()
    }
    
    fn variance<'py>(&self, py: Python<'py>, mu: PyReadonlyArray1<f64>) -> Bound<'py, PyArray1<f64>> {
        let mu_array = mu.as_array().to_owned();
        let result = self.inner.variance(&mu_array);
        result.into_pyarray_bound(py)
    }
    
    fn unit_deviance<'py>(
        &self,
        py: Python<'py>,
        y: PyReadonlyArray1<f64>,
        mu: PyReadonlyArray1<f64>,
    ) -> Bound<'py, PyArray1<f64>> {
        let y_array = y.as_array().to_owned();
        let mu_array = mu.as_array().to_owned();
        let result = self.inner.unit_deviance(&y_array, &mu_array);
        result.into_pyarray_bound(py)
    }
    
    fn deviance(&self, y: PyReadonlyArray1<f64>, mu: PyReadonlyArray1<f64>) -> f64 {
        let y_array = y.as_array().to_owned();
        let mu_array = mu.as_array().to_owned();
        self.inner.deviance(&y_array, &mu_array, None)
    }
    
    fn default_link(&self) -> PyLogLink {
        PyLogLink::new()
    }
}

/// Binomial family for binary/proportion data.
/// 
/// Use for logistic regression (yes/no outcomes).
/// Variance function: V(μ) = μ(1-μ)
#[pyclass(name = "BinomialFamily")]
#[derive(Clone)]
pub struct PyBinomialFamily {
    inner: BinomialFamily,
}

#[pymethods]
impl PyBinomialFamily {
    #[new]
    fn new() -> Self {
        Self { inner: BinomialFamily }
    }
    
    fn name(&self) -> &str {
        self.inner.name()
    }
    
    fn variance<'py>(&self, py: Python<'py>, mu: PyReadonlyArray1<f64>) -> Bound<'py, PyArray1<f64>> {
        let mu_array = mu.as_array().to_owned();
        let result = self.inner.variance(&mu_array);
        result.into_pyarray_bound(py)
    }
    
    fn unit_deviance<'py>(
        &self,
        py: Python<'py>,
        y: PyReadonlyArray1<f64>,
        mu: PyReadonlyArray1<f64>,
    ) -> Bound<'py, PyArray1<f64>> {
        let y_array = y.as_array().to_owned();
        let mu_array = mu.as_array().to_owned();
        let result = self.inner.unit_deviance(&y_array, &mu_array);
        result.into_pyarray_bound(py)
    }
    
    fn deviance(&self, y: PyReadonlyArray1<f64>, mu: PyReadonlyArray1<f64>) -> f64 {
        let y_array = y.as_array().to_owned();
        let mu_array = mu.as_array().to_owned();
        self.inner.deviance(&y_array, &mu_array, None)
    }
    
    fn default_link(&self) -> PyLogitLink {
        PyLogitLink::new()
    }
}

/// Gamma family for positive continuous data.
/// 
/// Use for claim severity, amounts, durations.
/// Variance function: V(μ) = μ² (constant CV)
#[pyclass(name = "GammaFamily")]
#[derive(Clone)]
pub struct PyGammaFamily {
    inner: GammaFamily,
}

#[pymethods]
impl PyGammaFamily {
    #[new]
    fn new() -> Self {
        Self { inner: GammaFamily }
    }
    
    fn name(&self) -> &str {
        self.inner.name()
    }
    
    fn variance<'py>(&self, py: Python<'py>, mu: PyReadonlyArray1<f64>) -> Bound<'py, PyArray1<f64>> {
        let mu_array = mu.as_array().to_owned();
        let result = self.inner.variance(&mu_array);
        result.into_pyarray_bound(py)
    }
    
    fn unit_deviance<'py>(
        &self,
        py: Python<'py>,
        y: PyReadonlyArray1<f64>,
        mu: PyReadonlyArray1<f64>,
    ) -> Bound<'py, PyArray1<f64>> {
        let y_array = y.as_array().to_owned();
        let mu_array = mu.as_array().to_owned();
        let result = self.inner.unit_deviance(&y_array, &mu_array);
        result.into_pyarray_bound(py)
    }
    
    fn deviance(&self, y: PyReadonlyArray1<f64>, mu: PyReadonlyArray1<f64>) -> f64 {
        let y_array = y.as_array().to_owned();
        let mu_array = mu.as_array().to_owned();
        self.inner.deviance(&y_array, &mu_array, None)
    }
    
    fn default_link(&self) -> PyLogLink {
        PyLogLink::new()
    }
}

/// Tweedie family for mixed zeros and positive continuous data.
/// 
/// Essential for insurance pure premium modeling (frequency × severity in one model).
/// Variance function: V(μ) = μ^p where p is the variance power.
///
/// Parameters
/// ----------
/// var_power : float
///     The variance power parameter p. Must be <= 0 or >= 1.
///     - p = 0: Gaussian
///     - p = 1: Poisson  
///     - 1 < p < 2: Compound Poisson-Gamma (insurance use case)
///     - p = 2: Gamma
///     - p = 3: Inverse Gaussian
///
/// Examples
/// --------
/// >>> import rustystats as rs
/// >>> # Fit Tweedie with p=1.5 for pure premium
/// >>> result = rs.fit_glm(y, X, family="tweedie", var_power=1.5)
#[pyclass(name = "TweedieFamily")]
#[derive(Clone)]
pub struct PyTweedieFamily {
    inner: TweedieFamily,
}

#[pymethods]
impl PyTweedieFamily {
    #[new]
    #[pyo3(signature = (var_power=1.5))]
    fn new(var_power: f64) -> PyResult<Self> {
        if var_power > 0.0 && var_power < 1.0 {
            return Err(PyValueError::new_err(
                format!("var_power must be <= 0 or >= 1, got {}", var_power)
            ));
        }
        Ok(Self { inner: TweedieFamily::new(var_power) })
    }
    
    fn name(&self) -> &str {
        self.inner.name()
    }
    
    /// Get the variance power parameter
    #[getter]
    fn var_power(&self) -> f64 {
        self.inner.var_power
    }
    
    fn variance<'py>(&self, py: Python<'py>, mu: PyReadonlyArray1<f64>) -> Bound<'py, PyArray1<f64>> {
        let mu_array = mu.as_array().to_owned();
        let result = self.inner.variance(&mu_array);
        result.into_pyarray_bound(py)
    }
    
    fn unit_deviance<'py>(
        &self,
        py: Python<'py>,
        y: PyReadonlyArray1<f64>,
        mu: PyReadonlyArray1<f64>,
    ) -> Bound<'py, PyArray1<f64>> {
        let y_array = y.as_array().to_owned();
        let mu_array = mu.as_array().to_owned();
        let result = self.inner.unit_deviance(&y_array, &mu_array);
        result.into_pyarray_bound(py)
    }
    
    fn deviance(&self, y: PyReadonlyArray1<f64>, mu: PyReadonlyArray1<f64>) -> f64 {
        let y_array = y.as_array().to_owned();
        let mu_array = mu.as_array().to_owned();
        self.inner.deviance(&y_array, &mu_array, None)
    }
    
    fn default_link(&self) -> PyLogLink {
        PyLogLink::new()
    }
}

// =============================================================================
// GLM Results
// =============================================================================
//
// This class holds the results of fitting a GLM.
// It provides access to coefficients, fitted values, and diagnostic info.
// =============================================================================

/// Results from fitting a GLM.
///
/// Contains coefficients, fitted values, deviance, and diagnostic information.
/// Use this to make predictions and assess model fit.
#[pyclass(name = "GLMResults")]
#[derive(Clone)]
pub struct PyGLMResults {
    /// Fitted coefficients
    coefficients: Array1<f64>,
    /// Fitted values (predictions on response scale)
    fitted_values: Array1<f64>,
    /// Linear predictor η = Xβ
    linear_predictor: Array1<f64>,
    /// Model deviance
    deviance: f64,
    /// Number of IRLS iterations
    iterations: usize,
    /// Did the algorithm converge?
    converged: bool,
    /// Unscaled covariance matrix (X'WX)⁻¹
    covariance_unscaled: Array2<f64>,
    /// Number of observations
    n_obs: usize,
    /// Number of parameters
    n_params: usize,
    /// Original response variable (for residuals)
    y: Array1<f64>,
    /// Family name (for diagnostics)
    family_name: String,
    /// Prior weights
    prior_weights: Array1<f64>,
    /// Regularization penalty applied (if any)
    penalty: Penalty,
    /// Design matrix X (for robust standard errors)
    design_matrix: Array2<f64>,
    /// IRLS weights (for robust standard errors)
    irls_weights: Array1<f64>,
}

// =============================================================================
// Helper Methods (not exposed to Python)
// =============================================================================

impl PyGLMResults {
    /// Get the appropriate Family trait object based on family_name.
    /// Used internally by diagnostics and robust SE methods.
    fn get_family(&self) -> Box<dyn Family> {
        match self.family_name.as_str() {
            "Gaussian" => Box::new(GaussianFamily),
            "Poisson" => Box::new(PoissonFamily),
            "Binomial" => Box::new(BinomialFamily),
            "Gamma" => Box::new(GammaFamily),
            // Tweedie falls back to Gaussian for now (would need var_power stored)
            _ => Box::new(GaussianFamily),
        }
    }
    
    /// Get prior weights as Option, returning None if all weights are 1.0.
    /// Many functions accept Option<&Array1<f64>> for weights.
    fn maybe_weights(&self) -> Option<&Array1<f64>> {
        if self.prior_weights.iter().all(|&w| (w - 1.0).abs() < 1e-10) {
            None
        } else {
            Some(&self.prior_weights)
        }
    }
    
    /// Compute robust covariance matrix (internal helper).
    /// Factored out to avoid repeating the same logic in cov_robust, bse_robust, etc.
    fn compute_robust_cov(&self, hc_type: HCType) -> Array2<f64> {
        let family = self.get_family();
        let pearson_resid = resid_pearson(&self.y, &self.fitted_values, family.as_ref());
        
        robust_covariance(
            &self.design_matrix,
            &pearson_resid,
            &self.irls_weights,
            &self.prior_weights,
            &self.covariance_unscaled,
            hc_type,
        )
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

    /// Get the dispersion parameter φ.
    ///
    /// For Poisson/Binomial: φ = 1 (fixed by assumption)
    /// For Gaussian/Gamma/Tweedie: φ = deviance / df_resid (estimated)
    ///
    /// This matches statsmodels behavior.
    fn scale(&self) -> f64 {
        // Poisson and Binomial have fixed dispersion = 1
        match self.family_name.as_str() {
            "Poisson" | "Binomial" => 1.0,
            _ => {
                // Estimate dispersion for Gaussian, Gamma, Tweedie
                let df = self.df_resid() as f64;
                if df > 0.0 {
                    self.deviance / df
                } else {
                    1.0
                }
            }
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
        
        let cov = self.compute_robust_cov(hc_type);
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
    /// >>> result = rs.fit_glm(y, X, family="poisson")
    /// >>> se_model = result.bse()       # Model-based SE
    /// >>> se_robust = result.bse_robust("HC1")  # Robust SE
    #[pyo3(signature = (cov_type="HC1"))]
    fn bse_robust<'py>(&self, py: Python<'py>, cov_type: &str) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let hc_type = HCType::from_str(cov_type).ok_or_else(|| {
            PyValueError::new_err(format!(
                "Unknown cov_type '{}'. Use 'HC0', 'HC1', 'HC2', or 'HC3'.", cov_type
            ))
        })?;
        
        let cov = self.compute_robust_cov(hc_type);
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
        
        let cov = self.compute_robust_cov(hc_type);
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
        
        let cov = self.compute_robust_cov(hc_type);
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
        
        let cov = self.compute_robust_cov(hc_type);
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
        // Determine link from family (using default links)
        let link: Box<dyn Link> = match self.family_name.as_str() {
            "Gaussian" => Box::new(IdentityLink),
            "Poisson" => Box::new(LogLink),
            "Binomial" => Box::new(LogitLink),
            "Gamma" => Box::new(LogLink),
            _ => Box::new(IdentityLink),
        };
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
        let scale = self.scale();
        let weights = self.maybe_weights();
        
        match self.family_name.as_str() {
            "Gaussian" => log_likelihood_gaussian(&self.y, &self.fitted_values, scale, weights),
            "Poisson" => log_likelihood_poisson(&self.y, &self.fitted_values, weights),
            "Binomial" => log_likelihood_binomial(&self.y, &self.fitted_values, weights),
            "Gamma" => log_likelihood_gamma(&self.y, &self.fitted_values, scale, weights),
            _ => log_likelihood_gaussian(&self.y, &self.fitted_values, scale, weights),
        }
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
    fn null_deviance(&self) -> f64 {
        null_deviance(&self.y, &self.family_name, self.maybe_weights())
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

// =============================================================================
// GLM Fitting Function
// =============================================================================

/// Fit a Generalized Linear Model.
///
/// This is the main entry point for GLM fitting. It uses IRLS
/// (Iteratively Reweighted Least Squares) to find the MLE.
///
/// Parameters
/// ----------
/// y : array-like
///     Response variable (1D array of length n)
/// X : array-like
///     Design matrix (2D array of shape n × p)
///     Should include a column of 1s for intercept if desired
/// family : str
///     Distribution family: "gaussian", "poisson", "binomial", "gamma", "tweedie"
/// link : str, optional
///     Link function: "identity", "log", "logit"
///     If None, uses the canonical link for the family
/// var_power : float, optional
///     Variance power for Tweedie family (default: 1.5)
///     Must be <= 0 or >= 1. Common values: 1.5-1.9 for insurance
/// offset : array-like, optional
///     Offset term added to linear predictor (e.g., log(exposure))
/// weights : array-like, optional
///     Prior weights for each observation
/// alpha : float, optional
///     Regularization strength (default: 0.0 = no regularization)
///     Higher values = stronger regularization = more shrinkage
/// l1_ratio : float, optional
///     Elastic Net mixing parameter (default: 0.0 = pure Ridge)
///     - 0.0: Ridge (L2) penalty only
///     - 1.0: Lasso (L1) penalty only  
///     - 0.0-1.0: Elastic Net (mix of L1 and L2)
/// max_iter : int, optional
///     Maximum IRLS iterations (default: 25)
/// tol : float, optional
///     Convergence tolerance (default: 1e-8)
///
/// Returns
/// -------
/// GLMResults
///     Object containing fitted coefficients, deviance, etc.
///
/// Examples
/// --------
/// >>> import rustystats as rs
/// >>> # Unregularized GLM
/// >>> result = rs.fit_glm(y, X, family="poisson")
/// >>> # Ridge regression
/// >>> result = rs.fit_glm(y, X, family="gaussian", alpha=0.1, l1_ratio=0.0)
/// >>> # Lasso regression  
/// >>> result = rs.fit_glm(y, X, family="gaussian", alpha=0.1, l1_ratio=1.0)
/// >>> # Elastic Net
/// >>> result = rs.fit_glm(y, X, family="gaussian", alpha=0.1, l1_ratio=0.5)
#[pyfunction]
#[pyo3(signature = (y, x, family, link=None, var_power=1.5, offset=None, weights=None, alpha=0.0, l1_ratio=0.0, max_iter=25, tol=1e-8))]
fn fit_glm_py(
    y: PyReadonlyArray1<f64>,
    x: PyReadonlyArray2<f64>,
    family: &str,
    link: Option<&str>,
    var_power: f64,
    offset: Option<PyReadonlyArray1<f64>>,
    weights: Option<PyReadonlyArray1<f64>>,
    alpha: f64,
    l1_ratio: f64,
    max_iter: usize,
    tol: f64,
) -> PyResult<PyGLMResults> {
    // Convert numpy arrays to ndarray
    let y_array: Array1<f64> = y.as_array().to_owned();
    let x_array: Array2<f64> = x.as_array().to_owned();

    let n_obs = y_array.len();
    let n_params = x_array.ncols();

    // Convert optional offset and weights
    let offset_array: Option<Array1<f64>> = offset.map(|o| o.as_array().to_owned());
    let weights_array: Option<Array1<f64>> = weights.map(|w| w.as_array().to_owned());

    // Create IRLS config
    let irls_config = IRLSConfig {
        max_iterations: max_iter,
        tolerance: tol,
        min_weight: 1e-10,
        verbose: false,
    };

    // Determine regularization type
    let use_regularization = alpha > 0.0;
    let use_coordinate_descent = use_regularization && l1_ratio > 0.0;

    // Create regularization config
    let reg_config = if use_regularization {
        if l1_ratio >= 1.0 {
            RegularizationConfig::lasso(alpha)
        } else if l1_ratio <= 0.0 {
            RegularizationConfig::ridge(alpha)
        } else {
            RegularizationConfig::elastic_net(alpha, l1_ratio)
        }
    } else {
        RegularizationConfig::none()
    };

    // Helper macro to reduce repetition - fits with appropriate solver
    macro_rules! fit_model {
        ($fam:expr, $link:expr) => {
            if use_coordinate_descent {
                fit_glm_coordinate_descent(
                    &y_array, &x_array, $fam, $link, &irls_config, &reg_config,
                    offset_array.as_ref(), weights_array.as_ref()
                )
            } else if use_regularization {
                fit_glm_regularized(
                    &y_array, &x_array, $fam, $link, &irls_config, &reg_config,
                    offset_array.as_ref(), weights_array.as_ref()
                )
            } else {
                fit_glm_full(
                    &y_array, &x_array, $fam, $link, &irls_config,
                    offset_array.as_ref(), weights_array.as_ref()
                )
            }
        };
    }

    // Match family and link, then fit
    let result: IRLSResult = match family.to_lowercase().as_str() {
        "gaussian" | "normal" => {
            let fam = GaussianFamily;
            match link.unwrap_or("identity") {
                "identity" => fit_model!(&fam, &IdentityLink),
                "log" => fit_model!(&fam, &LogLink),
                other => return Err(PyValueError::new_err(format!(
                    "Unknown link '{}' for Gaussian family. Use 'identity' or 'log'.", other
                ))),
            }
        }
        "poisson" => {
            let fam = PoissonFamily;
            match link.unwrap_or("log") {
                "log" => fit_model!(&fam, &LogLink),
                "identity" => fit_model!(&fam, &IdentityLink),
                other => return Err(PyValueError::new_err(format!(
                    "Unknown link '{}' for Poisson family. Use 'log' or 'identity'.", other
                ))),
            }
        }
        "binomial" => {
            let fam = BinomialFamily;
            match link.unwrap_or("logit") {
                "logit" => fit_model!(&fam, &LogitLink),
                "log" => fit_model!(&fam, &LogLink),
                other => return Err(PyValueError::new_err(format!(
                    "Unknown link '{}' for Binomial family. Use 'logit' or 'log'.", other
                ))),
            }
        }
        "gamma" => {
            let fam = GammaFamily;
            match link.unwrap_or("log") {
                "log" => fit_model!(&fam, &LogLink),
                "identity" => fit_model!(&fam, &IdentityLink),
                other => return Err(PyValueError::new_err(format!(
                    "Unknown link '{}' for Gamma family. Use 'log' or 'identity'.", other
                ))),
            }
        }
        "tweedie" => {
            if var_power > 0.0 && var_power < 1.0 {
                return Err(PyValueError::new_err(
                    format!("var_power must be <= 0 or >= 1, got {}", var_power)
                ));
            }
            let fam = TweedieFamily::new(var_power);
            match link.unwrap_or("log") {
                "log" => fit_model!(&fam, &LogLink),
                "identity" => fit_model!(&fam, &IdentityLink),
                other => return Err(PyValueError::new_err(format!(
                    "Unknown link '{}' for Tweedie family. Use 'log' or 'identity'.", other
                ))),
            }
        }
        other => return Err(PyValueError::new_err(format!(
            "Unknown family '{}'. Use 'gaussian', 'poisson', 'binomial', 'gamma', or 'tweedie'.", other
        ))),
    }.map_err(|e| PyValueError::new_err(format!("GLM fitting failed: {}", e)))?;

    Ok(PyGLMResults {
        coefficients: result.coefficients,
        fitted_values: result.fitted_values,
        linear_predictor: result.linear_predictor,
        deviance: result.deviance,
        iterations: result.iterations,
        converged: result.converged,
        covariance_unscaled: result.covariance_unscaled,
        n_obs,
        n_params,
        y: result.y,
        family_name: result.family_name,
        prior_weights: result.prior_weights,
        penalty: result.penalty,
        design_matrix: x_array,  // Use the X we already have (no extra copy)
        irls_weights: result.irls_weights,
    })
}

// =============================================================================
// Spline Basis Functions
// =============================================================================
//
// B-splines and natural splines for non-linear continuous effects in GLMs.
// These are computed in Rust for maximum performance.
// =============================================================================

use rustystats_core::splines;

/// Compute B-spline basis matrix.
///
/// B-splines are flexible piecewise polynomial bases commonly used for
/// modeling non-linear continuous effects in regression models.
///
/// Parameters
/// ----------
/// x : numpy.ndarray
///     Data points (1D array of length n)
/// df : int
///     Degrees of freedom (number of basis functions)
/// degree : int, optional
///     Spline degree. Default 3 (cubic splines).
/// boundary_knots : tuple, optional
///     (min, max) boundary knots. If None, uses data range.
/// include_intercept : bool, optional
///     Whether to include an intercept column. Default False.
///
/// Returns
/// -------
/// numpy.ndarray
///     Basis matrix of shape (n, df) or (n, df-1) if include_intercept=False
///
/// Examples
/// --------
/// >>> import rustystats as rs
/// >>> import numpy as np
/// >>> x = np.linspace(0, 10, 100)
/// >>> basis = rs.bs(x, df=5)
/// >>> print(basis.shape)
/// (100, 4)
#[pyfunction]
#[pyo3(signature = (x, df, degree=3, boundary_knots=None, include_intercept=false))]
fn bs_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    df: usize,
    degree: usize,
    boundary_knots: Option<(f64, f64)>,
    include_intercept: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let x_array = x.as_array().to_owned();
    let result = splines::bs_basis(&x_array, df, degree, boundary_knots, include_intercept);
    Ok(result.into_pyarray_bound(py))
}

/// Compute natural cubic spline basis matrix.
///
/// Natural splines are cubic splines with the additional constraint that
/// the second derivative is zero at the boundaries. This makes extrapolation
/// linear beyond the data range, which is often more sensible for prediction.
///
/// Parameters
/// ----------
/// x : numpy.ndarray
///     Data points (1D array of length n)
/// df : int
///     Degrees of freedom (number of basis functions)
/// boundary_knots : tuple, optional
///     (min, max) boundary knots. If None, uses data range.
/// include_intercept : bool, optional
///     Whether to include an intercept column. Default False.
///
/// Returns
/// -------
/// numpy.ndarray
///     Basis matrix of shape (n, df) or (n, df-1)
///
/// Notes
/// -----
/// Natural splines are recommended when extrapolation beyond the data
/// range is needed, as they provide more sensible linear extrapolation
/// compared to B-splines which can have erratic behavior at boundaries.
///
/// Examples
/// --------
/// >>> import rustystats as rs
/// >>> import numpy as np
/// >>> x = np.linspace(0, 10, 100)
/// >>> basis = rs.ns(x, df=5)
/// >>> print(basis.shape)
/// (100, 4)
#[pyfunction]
#[pyo3(signature = (x, df, boundary_knots=None, include_intercept=false))]
fn ns_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    df: usize,
    boundary_knots: Option<(f64, f64)>,
    include_intercept: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let x_array = x.as_array().to_owned();
    let result = splines::ns_basis(&x_array, df, boundary_knots, include_intercept);
    Ok(result.into_pyarray_bound(py))
}

/// Compute B-spline basis with explicit knots.
///
/// For cases where you want to specify interior knots directly rather
/// than having them computed from the data.
///
/// Parameters
/// ----------
/// x : numpy.ndarray
///     Data points (1D array)
/// knots : list
///     Interior knot positions
/// degree : int, optional
///     Spline degree. Default 3.
/// boundary_knots : tuple, optional
///     (min, max) boundary knots.
///
/// Returns
/// -------
/// numpy.ndarray
///     Basis matrix
#[pyfunction]
#[pyo3(signature = (x, knots, degree=3, boundary_knots=None))]
fn bs_knots_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    knots: Vec<f64>,
    degree: usize,
    boundary_knots: Option<(f64, f64)>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let x_array = x.as_array().to_owned();
    let result = splines::bs_with_knots(&x_array, &knots, degree, boundary_knots);
    Ok(result.into_pyarray_bound(py))
}

/// Get column names for B-spline basis.
#[pyfunction]
#[pyo3(signature = (var_name, df, include_intercept=false))]
fn bs_names_py(var_name: &str, df: usize, include_intercept: bool) -> Vec<String> {
    splines::bs_names(var_name, df, include_intercept)
}

/// Get column names for natural spline basis.
#[pyfunction]
#[pyo3(signature = (var_name, df, include_intercept=false))]
fn ns_names_py(var_name: &str, df: usize, include_intercept: bool) -> Vec<String> {
    splines::ns_names(var_name, df, include_intercept)
}

// =============================================================================
// Design Matrix Functions
// =============================================================================
//
// Fast categorical encoding and interaction construction in Rust.
// =============================================================================

use rustystats_core::design_matrix;

/// Encode categorical variable from string values.
///
/// Parameters
/// ----------
/// values : list[str]
///     String values for each observation
/// var_name : str
///     Variable name (for column naming)
/// drop_first : bool
///     Whether to drop the first level (reference category)
///
/// Returns
/// -------
/// tuple[numpy.ndarray, list[str], list[int], list[str]]
///     (dummy_matrix, column_names, indices, levels)
#[pyfunction]
#[pyo3(signature = (values, var_name, drop_first=true))]
fn encode_categorical_py<'py>(
    py: Python<'py>,
    values: Vec<String>,
    var_name: &str,
    drop_first: bool,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Vec<String>, Vec<i32>, Vec<String>)> {
    let enc = design_matrix::encode_categorical(&values, var_name, drop_first);
    Ok((
        enc.matrix.into_pyarray_bound(py),
        enc.names,
        enc.indices,
        enc.levels,
    ))
}

/// Encode categorical from pre-computed indices.
///
/// Use when indices are already computed (e.g., from pandas factorize).
///
/// Parameters
/// ----------
/// indices : numpy.ndarray
///     Pre-computed level indices (0-indexed, int32)
/// n_levels : int
///     Total number of levels
/// level_names : list[str]
///     Names for each level
/// var_name : str
///     Variable name
/// drop_first : bool
///     Drop first level
///
/// Returns
/// -------
/// tuple[numpy.ndarray, list[str]]
///     (dummy_matrix, column_names)
#[pyfunction]
#[pyo3(signature = (indices, n_levels, level_names, var_name, drop_first=true))]
fn encode_categorical_indices_py<'py>(
    py: Python<'py>,
    indices: PyReadonlyArray1<i32>,
    n_levels: usize,
    level_names: Vec<String>,
    var_name: &str,
    drop_first: bool,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Vec<String>)> {
    let indices_vec: Vec<i32> = indices.as_array().to_vec();
    let enc = design_matrix::encode_categorical_from_indices(
        &indices_vec, n_levels, &level_names, var_name, drop_first
    );
    Ok((enc.matrix.into_pyarray_bound(py), enc.names))
}

/// Build categorical × categorical interaction matrix.
///
/// Parameters
/// ----------
/// idx1 : numpy.ndarray
///     Level indices for first categorical (0 = reference)
/// n_levels1 : int
///     Number of non-reference levels for first
/// idx2 : numpy.ndarray
///     Level indices for second categorical
/// n_levels2 : int
///     Number of non-reference levels for second
/// names1 : list[str]
///     Column names for first categorical dummies
/// names2 : list[str]
///     Column names for second categorical dummies
///
/// Returns
/// -------
/// tuple[numpy.ndarray, list[str]]
///     (interaction_matrix, column_names)
#[pyfunction]
fn build_cat_cat_interaction_py<'py>(
    py: Python<'py>,
    idx1: PyReadonlyArray1<i32>,
    n_levels1: usize,
    idx2: PyReadonlyArray1<i32>,
    n_levels2: usize,
    names1: Vec<String>,
    names2: Vec<String>,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Vec<String>)> {
    let idx1_vec: Vec<i32> = idx1.as_array().to_vec();
    let idx2_vec: Vec<i32> = idx2.as_array().to_vec();
    let (matrix, names) = design_matrix::build_categorical_categorical_interaction(
        &idx1_vec, n_levels1, &idx2_vec, n_levels2, &names1, &names2
    );
    Ok((matrix.into_pyarray_bound(py), names))
}

/// Build categorical × continuous interaction matrix.
///
/// Parameters
/// ----------
/// cat_indices : numpy.ndarray
///     Level indices for categorical (0 = reference)
/// n_levels : int
///     Number of non-reference levels
/// continuous : numpy.ndarray
///     Continuous variable values
/// cat_names : list[str]
///     Column names for categorical dummies
/// cont_name : str
///     Name of continuous variable
///
/// Returns
/// -------
/// tuple[numpy.ndarray, list[str]]
///     (interaction_matrix, column_names)
#[pyfunction]
fn build_cat_cont_interaction_py<'py>(
    py: Python<'py>,
    cat_indices: PyReadonlyArray1<i32>,
    n_levels: usize,
    continuous: PyReadonlyArray1<f64>,
    cat_names: Vec<String>,
    cont_name: &str,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Vec<String>)> {
    let idx_vec: Vec<i32> = cat_indices.as_array().to_vec();
    let cont_array = continuous.as_array().to_owned();
    let (matrix, names) = design_matrix::build_categorical_continuous_interaction(
        &idx_vec, n_levels, &cont_array, &cat_names, cont_name
    );
    Ok((matrix.into_pyarray_bound(py), names))
}

/// Build continuous × continuous interaction.
///
/// Simple element-wise multiplication.
#[pyfunction]
fn build_cont_cont_interaction_py<'py>(
    py: Python<'py>,
    x1: PyReadonlyArray1<f64>,
    x2: PyReadonlyArray1<f64>,
    name1: &str,
    name2: &str,
) -> PyResult<(Bound<'py, PyArray1<f64>>, String)> {
    let x1_array = x1.as_array().to_owned();
    let x2_array = x2.as_array().to_owned();
    let (result, name) = design_matrix::build_continuous_continuous_interaction(
        &x1_array, &x2_array, name1, name2
    );
    Ok((result.into_pyarray_bound(py), name))
}

/// Multiply each column of a matrix by a continuous vector.
///
/// Used for multi-categorical × continuous interactions.
#[pyfunction]
fn multiply_matrix_by_continuous_py<'py>(
    py: Python<'py>,
    matrix: PyReadonlyArray2<f64>,
    continuous: PyReadonlyArray1<f64>,
    matrix_names: Vec<String>,
    cont_name: &str,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Vec<String>)> {
    let matrix_array = matrix.as_array().to_owned();
    let cont_array = continuous.as_array().to_owned();
    let (result, names) = design_matrix::multiply_matrix_by_continuous(
        &matrix_array, &cont_array, &matrix_names, cont_name
    );
    Ok((result.into_pyarray_bound(py), names))
}

// =============================================================================
// Formula Parsing
// =============================================================================

use rustystats_core::formula;

/// Parse a formula string into structured components.
///
/// Parameters
/// ----------
/// formula_str : str
///     R-style formula like "y ~ x1*x2 + C(cat) + bs(age, df=5)"
///
/// Returns
/// -------
/// dict
///     Parsed formula with keys:
///     - response: str
///     - main_effects: list[str]
///     - interactions: list[dict] with 'factors' and 'categorical_flags'
///     - categorical_vars: list[str]
///     - spline_terms: list[dict] with 'var_name', 'spline_type', 'df', 'degree'
///     - has_intercept: bool
#[pyfunction]
fn parse_formula_py(formula_str: &str) -> PyResult<std::collections::HashMap<String, pyo3::PyObject>> {
    use pyo3::types::PyDict;
    
    let parsed = formula::parse_formula(formula_str)
        .map_err(|e| PyValueError::new_err(e))?;
    
    Python::with_gil(|py| {
        let mut result = std::collections::HashMap::new();
        
        result.insert("response".to_string(), parsed.response.into_py(py));
        result.insert("main_effects".to_string(), parsed.main_effects.into_py(py));
        result.insert("has_intercept".to_string(), parsed.has_intercept.into_py(py));
        result.insert("categorical_vars".to_string(), 
            parsed.categorical_vars.into_iter().collect::<Vec<_>>().into_py(py));
        
        // Convert interactions
        let interactions: Vec<_> = parsed.interactions
            .into_iter()
            .map(|i| {
                let dict = PyDict::new_bound(py);
                dict.set_item("factors", i.factors).unwrap();
                dict.set_item("categorical_flags", i.categorical_flags).unwrap();
                dict.into_py(py)
            })
            .collect();
        result.insert("interactions".to_string(), interactions.into_py(py));
        
        // Convert spline terms
        let splines: Vec<_> = parsed.spline_terms
            .into_iter()
            .map(|s| {
                let dict = PyDict::new_bound(py);
                dict.set_item("var_name", s.var_name).unwrap();
                dict.set_item("spline_type", s.spline_type).unwrap();
                dict.set_item("df", s.df).unwrap();
                dict.set_item("degree", s.degree).unwrap();
                dict.into_py(py)
            })
            .collect();
        result.insert("spline_terms".to_string(), splines.into_py(py));
        
        Ok(result)
    })
}

// =============================================================================
// Module Registration
// =============================================================================
//
// This is where we tell Python what's available when you import the module.
// Everything added here with `m.add_class` or `m.add_function` becomes
// accessible from Python.
// =============================================================================

/// RustyStats: Fast GLM fitting with a Rust backend
/// 
/// This is the internal Rust module. Users should import from the
/// Python wrapper: `import rustystats`
#[pymodule]
fn _rustystats(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add link functions
    m.add_class::<PyIdentityLink>()?;
    m.add_class::<PyLogLink>()?;
    m.add_class::<PyLogitLink>()?;
    
    // Add families
    m.add_class::<PyGaussianFamily>()?;
    m.add_class::<PyPoissonFamily>()?;
    m.add_class::<PyBinomialFamily>()?;
    m.add_class::<PyGammaFamily>()?;
    m.add_class::<PyTweedieFamily>()?;
    
    // Add GLM fitting
    m.add_class::<PyGLMResults>()?;
    m.add_function(wrap_pyfunction!(fit_glm_py, m)?)?;
    
    // Add spline functions
    m.add_function(wrap_pyfunction!(bs_py, m)?)?;
    m.add_function(wrap_pyfunction!(ns_py, m)?)?;
    m.add_function(wrap_pyfunction!(bs_knots_py, m)?)?;
    m.add_function(wrap_pyfunction!(bs_names_py, m)?)?;
    m.add_function(wrap_pyfunction!(ns_names_py, m)?)?;
    
    // Add design matrix functions
    m.add_function(wrap_pyfunction!(encode_categorical_py, m)?)?;
    m.add_function(wrap_pyfunction!(encode_categorical_indices_py, m)?)?;
    m.add_function(wrap_pyfunction!(build_cat_cat_interaction_py, m)?)?;
    m.add_function(wrap_pyfunction!(build_cat_cont_interaction_py, m)?)?;
    m.add_function(wrap_pyfunction!(build_cont_cont_interaction_py, m)?)?;
    m.add_function(wrap_pyfunction!(multiply_matrix_by_continuous_py, m)?)?;
    
    // Add formula parsing
    m.add_function(wrap_pyfunction!(parse_formula_py, m)?)?;
    
    Ok(())
}
