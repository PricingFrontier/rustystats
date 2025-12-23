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
use rustystats_core::families::{Family, GaussianFamily, PoissonFamily, BinomialFamily, GammaFamily};
use rustystats_core::links::{Link, IdentityLink, LogLink, LogitLink};
use rustystats_core::solvers::{fit_glm_full, IRLSConfig, IRLSResult};
use rustystats_core::inference::{pvalue_z, confidence_interval_z};
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

    /// Estimate the dispersion parameter φ.
    ///
    /// For Gaussian: φ = deviance / df_resid
    /// For Poisson/Binomial: φ = 1 (by assumption)
    /// For Gamma: φ = deviance / df_resid (approximately)
    fn scale(&self) -> f64 {
        let df = self.df_resid() as f64;
        if df > 0.0 {
            self.deviance / df
        } else {
            1.0
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
        let family: Box<dyn Family> = match self.family_name.as_str() {
            "Gaussian" => Box::new(GaussianFamily),
            "Poisson" => Box::new(PoissonFamily),
            "Binomial" => Box::new(BinomialFamily),
            "Gamma" => Box::new(GammaFamily),
            _ => Box::new(GaussianFamily),
        };
        let resid = resid_pearson(&self.y, &self.fitted_values, family.as_ref());
        resid.into_pyarray_bound(py)
    }

    /// Get deviance residuals: sign(y - μ) × √d_i
    ///
    /// Based on the unit deviance contributions. Often more normally
    /// distributed than Pearson residuals for non-Gaussian families.
    /// sum(resid_deviance²) = model deviance
    fn resid_deviance<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let family: Box<dyn Family> = match self.family_name.as_str() {
            "Gaussian" => Box::new(GaussianFamily),
            "Poisson" => Box::new(PoissonFamily),
            "Binomial" => Box::new(BinomialFamily),
            "Gamma" => Box::new(GammaFamily),
            _ => Box::new(GaussianFamily),
        };
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
        let family: Box<dyn Family> = match self.family_name.as_str() {
            "Gaussian" => Box::new(GaussianFamily),
            "Poisson" => Box::new(PoissonFamily),
            "Binomial" => Box::new(BinomialFamily),
            "Gamma" => Box::new(GammaFamily),
            _ => Box::new(GaussianFamily),
        };
        
        let weights = if self.prior_weights.iter().all(|&w| (w - 1.0).abs() < 1e-10) {
            None
        } else {
            Some(&self.prior_weights)
        };
        
        pearson_chi2(&self.y, &self.fitted_values, family.as_ref(), weights)
    }

    /// Get dispersion estimated from Pearson residuals.
    ///
    /// φ_pearson = X² / (n - p)
    fn scale_pearson(&self) -> f64 {
        let family: Box<dyn Family> = match self.family_name.as_str() {
            "Gaussian" => Box::new(GaussianFamily),
            "Poisson" => Box::new(PoissonFamily),
            "Binomial" => Box::new(BinomialFamily),
            "Gamma" => Box::new(GammaFamily),
            _ => Box::new(GaussianFamily),
        };
        
        let weights = if self.prior_weights.iter().all(|&w| (w - 1.0).abs() < 1e-10) {
            None
        } else {
            Some(&self.prior_weights)
        };
        
        estimate_dispersion_pearson(
            &self.y, 
            &self.fitted_values, 
            family.as_ref(), 
            self.df_resid(),
            weights,
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
        let weights = if self.prior_weights.iter().all(|&w| (w - 1.0).abs() < 1e-10) {
            None
        } else {
            Some(&self.prior_weights)
        };
        
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
        let weights = if self.prior_weights.iter().all(|&w| (w - 1.0).abs() < 1e-10) {
            None
        } else {
            Some(&self.prior_weights)
        };
        
        null_deviance(&self.y, &self.family_name, weights)
    }

    /// Get the family name.
    #[getter]
    fn family(&self) -> &str {
        &self.family_name
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
///     Distribution family: "gaussian", "poisson", "binomial", "gamma"
/// link : str, optional
///     Link function: "identity", "log", "logit"
///     If None, uses the canonical link for the family
/// offset : array-like, optional
///     Offset term added to linear predictor (e.g., log(exposure))
/// weights : array-like, optional
///     Prior weights for each observation
/// max_iter : int, optional
///     Maximum IRLS iterations (default: 25)
/// tol : float, optional
///     Convergence tolerance (default: 1e-8)
///
/// Returns
/// -------
/// GLMResults
///     Object containing fitted coefficients, deviance, etc.
#[pyfunction]
#[pyo3(signature = (y, x, family, link=None, offset=None, weights=None, max_iter=25, tol=1e-8))]
fn fit_glm_py(
    y: PyReadonlyArray1<f64>,
    x: PyReadonlyArray2<f64>,
    family: &str,
    link: Option<&str>,
    offset: Option<PyReadonlyArray1<f64>>,
    weights: Option<PyReadonlyArray1<f64>>,
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

    // Create config
    let config = IRLSConfig {
        max_iterations: max_iter,
        tolerance: tol,
        min_weight: 1e-10,
        verbose: false,
    };

    // Match family and link, then fit
    let result: IRLSResult = match family.to_lowercase().as_str() {
        "gaussian" | "normal" => {
            let fam = GaussianFamily;
            match link.unwrap_or("identity") {
                "identity" => fit_glm_full(&y_array, &x_array, &fam, &IdentityLink, &config, 
                    offset_array.as_ref(), weights_array.as_ref()),
                "log" => fit_glm_full(&y_array, &x_array, &fam, &LogLink, &config,
                    offset_array.as_ref(), weights_array.as_ref()),
                other => return Err(PyValueError::new_err(format!(
                    "Unknown link '{}' for Gaussian family. Use 'identity' or 'log'.", other
                ))),
            }
        }
        "poisson" => {
            let fam = PoissonFamily;
            match link.unwrap_or("log") {
                "log" => fit_glm_full(&y_array, &x_array, &fam, &LogLink, &config,
                    offset_array.as_ref(), weights_array.as_ref()),
                "identity" => fit_glm_full(&y_array, &x_array, &fam, &IdentityLink, &config,
                    offset_array.as_ref(), weights_array.as_ref()),
                other => return Err(PyValueError::new_err(format!(
                    "Unknown link '{}' for Poisson family. Use 'log' or 'identity'.", other
                ))),
            }
        }
        "binomial" => {
            let fam = BinomialFamily;
            match link.unwrap_or("logit") {
                "logit" => fit_glm_full(&y_array, &x_array, &fam, &LogitLink, &config,
                    offset_array.as_ref(), weights_array.as_ref()),
                "log" => fit_glm_full(&y_array, &x_array, &fam, &LogLink, &config,
                    offset_array.as_ref(), weights_array.as_ref()),
                other => return Err(PyValueError::new_err(format!(
                    "Unknown link '{}' for Binomial family. Use 'logit' or 'log'.", other
                ))),
            }
        }
        "gamma" => {
            let fam = GammaFamily;
            match link.unwrap_or("log") {
                "log" => fit_glm_full(&y_array, &x_array, &fam, &LogLink, &config,
                    offset_array.as_ref(), weights_array.as_ref()),
                "identity" => fit_glm_full(&y_array, &x_array, &fam, &IdentityLink, &config,
                    offset_array.as_ref(), weights_array.as_ref()),
                other => return Err(PyValueError::new_err(format!(
                    "Unknown link '{}' for Gamma family. Use 'log' or 'identity'.", other
                ))),
            }
        }
        other => return Err(PyValueError::new_err(format!(
            "Unknown family '{}'. Use 'gaussian', 'poisson', 'binomial', or 'gamma'.", other
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
    
    // Add GLM fitting
    m.add_class::<PyGLMResults>()?;
    m.add_function(wrap_pyfunction!(fit_glm_py, m)?)?;
    
    Ok(())
}
