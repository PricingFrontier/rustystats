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
use rustystats_core::families::{Family, GaussianFamily, PoissonFamily, BinomialFamily, GammaFamily, TweedieFamily, QuasiPoissonFamily, QuasiBinomialFamily, NegativeBinomialFamily};
use rustystats_core::links::{Link, IdentityLink, LogLink, LogitLink};
use rustystats_core::solvers::{fit_glm_full, fit_glm_regularized, fit_glm_coordinate_descent, IRLSConfig, IRLSResult};
use rustystats_core::regularization::{Penalty, RegularizationConfig};
use rustystats_core::inference::{pvalue_z, confidence_interval_z, HCType, robust_covariance, robust_standard_errors};
use rustystats_core::diagnostics::{
    resid_response, resid_pearson, resid_deviance, resid_working,
    estimate_dispersion_pearson, pearson_chi2,
    log_likelihood_gaussian, log_likelihood_poisson, log_likelihood_binomial, log_likelihood_gamma,
    aic, bic, null_deviance,
    estimate_theta_profile, estimate_theta_moments,
    // Calibration and discrimination
    compute_calibration_curve, compute_discrimination_stats, compute_lorenz_curve,
    hosmer_lemeshow_test,
    // Factor diagnostics
    compute_ae_continuous, compute_ae_categorical,
    compute_residual_pattern_continuous,
    // Interactions
    detect_interactions, InteractionConfig, FactorData,
    // Loss
    mse, rmse, mae, compute_family_loss,
};

// =============================================================================
// Family and Link Helper Functions
// =============================================================================
//
// These helper functions consolidate the repeated family/link dispatch logic
// that was previously duplicated across multiple functions.
// =============================================================================

/// Get a Family trait object from a family name string.
/// 
/// Handles case-insensitive matching and common aliases.
/// Returns an error for unknown family names instead of silently defaulting.
fn family_from_name(name: &str) -> PyResult<Box<dyn Family>> {
    let lower = name.to_lowercase();
    
    // Handle negativebinomial with optional theta parameter like "negativebinomial(theta=1.38)"
    if lower.starts_with("negativebinomial") || lower.starts_with("negbinomial") || lower.starts_with("negbin") {
        // Parse theta if present
        let theta = if let Some(start) = lower.find("theta=") {
            let rest = &lower[start + 6..];
            let end = rest.find(')').unwrap_or(rest.len());
            rest[..end].parse::<f64>().unwrap_or(1.0)
        } else {
            1.0
        };
        return Ok(Box::new(NegativeBinomialFamily::new(theta)));
    }
    
    match lower.as_str() {
        "gaussian" | "normal" => Ok(Box::new(GaussianFamily)),
        "poisson" => Ok(Box::new(PoissonFamily)),
        "binomial" => Ok(Box::new(BinomialFamily)),
        "gamma" => Ok(Box::new(GammaFamily)),
        "quasipoisson" => Ok(Box::new(QuasiPoissonFamily)),
        "quasibinomial" => Ok(Box::new(QuasiBinomialFamily)),
        _ => Err(PyValueError::new_err(format!(
            "Unknown family '{}'. Use 'gaussian', 'poisson', 'binomial', 'gamma', \
             'quasipoisson', 'quasibinomial', or 'negativebinomial'.", name
        ))),
    }
}

/// Get the default Link function for a given family name.
/// Panics on unknown family - this should only be called with validated family names.
fn default_link_from_family(family_name: &str) -> Box<dyn Link> {
    match family_name.to_lowercase().as_str() {
        "gaussian" | "normal" => Box::new(IdentityLink),
        "poisson" | "quasipoisson" | "negativebinomial" | "negbinomial" | "negbin" | "gamma" | "tweedie" => Box::new(LogLink),
        "binomial" | "quasibinomial" => Box::new(LogitLink),
        other => panic!("default_link_from_family called with unknown family '{}' - this is a bug", other),
    }
}

/// Get a Link trait object from a link name string.
/// Returns an error for unknown link names instead of silently defaulting.
fn link_from_name(name: &str) -> PyResult<Box<dyn Link>> {
    match name.to_lowercase().as_str() {
        "identity" => Ok(Box::new(IdentityLink)),
        "log" => Ok(Box::new(LogLink)),
        "logit" => Ok(Box::new(LogitLink)),
        _ => Err(PyValueError::new_err(format!(
            "Unknown link '{}'. Use 'identity', 'log', or 'logit'.", name
        ))),
    }
}

// =============================================================================
// Link Function Wrappers (Macro-Generated)
// =============================================================================
//
// These wrap the Rust link functions so Python can use them.
// Each class provides the same interface: link(), inverse(), derivative()
// =============================================================================

/// Macro to generate PyO3 link function wrappers.
/// Eliminates ~40 lines of boilerplate per link type.
macro_rules! impl_py_link {
    ($py_name:ident, $py_str:literal, $inner_type:ty, $inner_expr:expr) => {
        #[pyclass(name = $py_str)]
        #[derive(Clone)]
        pub struct $py_name {
            inner: $inner_type,
        }

        #[pymethods]
        impl $py_name {
            #[new]
            fn new() -> Self {
                Self { inner: $inner_expr }
            }

            fn name(&self) -> &str {
                self.inner.name()
            }

            fn link<'py>(&self, py: Python<'py>, mu: PyReadonlyArray1<f64>) -> Bound<'py, PyArray1<f64>> {
                self.inner.link(&mu.as_array().to_owned()).into_pyarray_bound(py)
            }

            fn inverse<'py>(&self, py: Python<'py>, eta: PyReadonlyArray1<f64>) -> Bound<'py, PyArray1<f64>> {
                self.inner.inverse(&eta.as_array().to_owned()).into_pyarray_bound(py)
            }

            fn derivative<'py>(&self, py: Python<'py>, mu: PyReadonlyArray1<f64>) -> Bound<'py, PyArray1<f64>> {
                self.inner.derivative(&mu.as_array().to_owned()).into_pyarray_bound(py)
            }
        }
    };
}

// Generate all link wrappers (3 types × ~40 lines = ~120 lines → ~3 lines each)
impl_py_link!(PyIdentityLink, "IdentityLink", IdentityLink, IdentityLink);
impl_py_link!(PyLogLink, "LogLink", LogLink, LogLink);
impl_py_link!(PyLogitLink, "LogitLink", LogitLink, LogitLink);

// =============================================================================
// Family Wrappers (Macro-Generated)
// =============================================================================
//
// These wrap the Rust distribution families for Python.
// Each provides: variance(), unit_deviance(), deviance(), default_link()
// =============================================================================

/// Macro to generate PyO3 family wrappers for simple (no-parameter) families.
/// Eliminates ~50 lines of boilerplate per family type.
macro_rules! impl_py_family {
    ($py_name:ident, $py_str:literal, $inner_type:ty, $inner_expr:expr, $default_link:ty) => {
        #[pyclass(name = $py_str)]
        #[derive(Clone)]
        pub struct $py_name {
            inner: $inner_type,
        }

        #[pymethods]
        impl $py_name {
            #[new]
            fn new() -> Self {
                Self { inner: $inner_expr }
            }

            fn name(&self) -> &str {
                self.inner.name()
            }

            fn variance<'py>(&self, py: Python<'py>, mu: PyReadonlyArray1<f64>) -> Bound<'py, PyArray1<f64>> {
                self.inner.variance(&mu.as_array().to_owned()).into_pyarray_bound(py)
            }

            fn unit_deviance<'py>(&self, py: Python<'py>, y: PyReadonlyArray1<f64>, mu: PyReadonlyArray1<f64>) -> Bound<'py, PyArray1<f64>> {
                self.inner.unit_deviance(&y.as_array().to_owned(), &mu.as_array().to_owned()).into_pyarray_bound(py)
            }

            fn deviance(&self, y: PyReadonlyArray1<f64>, mu: PyReadonlyArray1<f64>) -> f64 {
                self.inner.deviance(&y.as_array().to_owned(), &mu.as_array().to_owned(), None)
            }

            fn default_link(&self) -> $default_link {
                <$default_link>::new()
            }
        }
    };
}

// Generate simple family wrappers (6 types × ~50 lines = ~300 lines → ~6 lines each)
impl_py_family!(PyGaussianFamily, "GaussianFamily", GaussianFamily, GaussianFamily, PyIdentityLink);
impl_py_family!(PyPoissonFamily, "PoissonFamily", PoissonFamily, PoissonFamily, PyLogLink);
impl_py_family!(PyBinomialFamily, "BinomialFamily", BinomialFamily, BinomialFamily, PyLogitLink);
impl_py_family!(PyGammaFamily, "GammaFamily", GammaFamily, GammaFamily, PyLogLink);
impl_py_family!(PyQuasiPoissonFamily, "QuasiPoissonFamily", QuasiPoissonFamily, QuasiPoissonFamily, PyLogLink);
impl_py_family!(PyQuasiBinomialFamily, "QuasiBinomialFamily", QuasiBinomialFamily, QuasiBinomialFamily, PyLogitLink);

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
/// >>> result = rs.glm("y ~ x1 + x2", data, family="tweedie", var_power=1.5).fit()
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

// Parameterized families (Tweedie, NegativeBinomial) are defined manually below
// as they require constructor arguments.

/// Negative Binomial family for overdispersed count data.
///
/// Uses the NB2 parameterization where variance is quadratic in the mean:
///   Var(Y) = μ + μ²/θ
///
/// This is an alternative to QuasiPoisson that models overdispersion explicitly
/// with a proper probability distribution, enabling valid likelihood-based inference.
///
/// Parameters
/// ----------
/// theta : float, optional
///     Dispersion parameter (default: 1.0). Larger θ = less overdispersion.
///     - θ = 0.5: Strong overdispersion (variance = μ + 2μ²)
///     - θ = 1.0: Moderate overdispersion (variance = μ + μ²)
///     - θ = 10: Mild overdispersion (close to Poisson)
///     - θ → ∞: Approaches Poisson
///
/// Examples
/// --------
/// >>> import rustystats as rs
/// >>> # Fit Negative Binomial with θ=1.0
/// >>> result = rs.glm("y ~ x1 + x2", data, family="negbinomial", theta=1.0).fit()
/// >>> # Or use the family object directly
/// >>> family = rs.families.NegativeBinomial(theta=2.0)
#[pyclass(name = "NegativeBinomialFamily")]
#[derive(Clone)]
pub struct PyNegativeBinomialFamily {
    inner: NegativeBinomialFamily,
}

#[pymethods]
impl PyNegativeBinomialFamily {
    #[new]
    #[pyo3(signature = (theta=1.0))]
    fn new(theta: f64) -> PyResult<Self> {
        if theta <= 0.0 {
            return Err(PyValueError::new_err(
                format!("theta must be > 0, got {}", theta)
            ));
        }
        Ok(Self { inner: NegativeBinomialFamily::new(theta) })
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    /// Get the theta (dispersion) parameter
    #[getter]
    fn theta(&self) -> f64 {
        self.inner.theta
    }

    /// Get alpha = 1/theta (alternative parameterization)
    #[getter]
    fn alpha(&self) -> f64 {
        self.inner.alpha()
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
    /// Note: family_name is validated at model creation, so this should never fail.
    fn get_family(&self) -> Box<dyn Family> {
        family_from_name(&self.family_name)
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
    /// For QuasiPoisson/QuasiBinomial: φ = Pearson χ² / df_resid (estimated)
    /// For Gaussian/Gamma/Tweedie: φ = deviance / df_resid (estimated)
    ///
    /// This matches statsmodels behavior.
    fn scale(&self) -> f64 {
        match self.family_name.as_str() {
            // True Poisson and Binomial have fixed dispersion = 1
            "Poisson" | "Binomial" => 1.0,
            // Quasi-families estimate dispersion from Pearson residuals
            // This is the key difference: SE's are inflated by √φ
            "QuasiPoisson" | "QuasiBinomial" => {
                let family = self.get_family();
                estimate_dispersion_pearson(
                    &self.y,
                    &self.fitted_values,
                    family.as_ref(),
                    self.df_resid(),
                    self.maybe_weights(),
                )
            }
            _ => {
                // Estimate dispersion for Gaussian, Gamma, Tweedie using deviance
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
        let link = default_link_from_family(&self.family_name);
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
///     Distribution family: "gaussian", "poisson", "binomial", "gamma", "tweedie",
///     "quasipoisson", "quasibinomial", or "negbinomial"
/// link : str, optional
///     Link function: "identity", "log", "logit"
///     If None, uses the canonical link for the family
/// var_power : float, optional
///     Variance power for Tweedie family (default: 1.5)
///     Must be <= 0 or >= 1. Common values: 1.5-1.9 for insurance
/// theta : float, optional
///     Dispersion parameter for Negative Binomial (default: 1.0)
///     Larger θ = less overdispersion. As θ → ∞, approaches Poisson.
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
/// This is an internal function. Use the formula API instead:
/// >>> import rustystats as rs
/// >>> result = rs.glm("y ~ x1 + x2", data, family="poisson").fit()
/// >>> # With regularization
/// >>> result = rs.glm("y ~ x1 + x2", data, family="gaussian").fit(alpha=0.1, l1_ratio=0.5)
#[pyfunction]
#[pyo3(signature = (y, x, family, link=None, var_power=1.5, theta=1.0, offset=None, weights=None, alpha=0.0, l1_ratio=0.0, max_iter=25, tol=1e-8))]
fn fit_glm_py(
    y: PyReadonlyArray1<f64>,
    x: PyReadonlyArray2<f64>,
    family: &str,
    link: Option<&str>,
    var_power: f64,
    theta: f64,
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
            // Gamma requires y > 0 (strictly positive)
            let n_invalid = y_array.iter().filter(|&&v| v <= 0.0).count();
            if n_invalid > 0 {
                return Err(PyValueError::new_err(format!(
                    "Gamma family requires strictly positive response values (y > 0). \
                     Found {} values <= 0 out of {} observations. \
                     For severity modeling, filter to only records with claims: \
                     data.filter(pl.col('ClaimAmount') > 0)",
                    n_invalid, y_array.len()
                )));
            }
            
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
        "quasipoisson" | "quasi-poisson" | "quasi_poisson" => {
            let fam = QuasiPoissonFamily;
            match link.unwrap_or("log") {
                "log" => fit_model!(&fam, &LogLink),
                "identity" => fit_model!(&fam, &IdentityLink),
                other => return Err(PyValueError::new_err(format!(
                    "Unknown link '{}' for QuasiPoisson family. Use 'log' or 'identity'.", other
                ))),
            }
        }
        "quasibinomial" | "quasi-binomial" | "quasi_binomial" => {
            let fam = QuasiBinomialFamily;
            match link.unwrap_or("logit") {
                "logit" => fit_model!(&fam, &LogitLink),
                "log" => fit_model!(&fam, &LogLink),
                other => return Err(PyValueError::new_err(format!(
                    "Unknown link '{}' for QuasiBinomial family. Use 'logit' or 'log'.", other
                ))),
            }
        }
        "negbinomial" | "negativebinomial" | "negative_binomial" | "neg-binomial" | "nb" => {
            if theta <= 0.0 {
                return Err(PyValueError::new_err(
                    format!("theta must be > 0 for Negative Binomial, got {}", theta)
                ));
            }
            let fam = NegativeBinomialFamily::new(theta);
            match link.unwrap_or("log") {
                "log" => fit_model!(&fam, &LogLink),
                "identity" => fit_model!(&fam, &IdentityLink),
                other => return Err(PyValueError::new_err(format!(
                    "Unknown link '{}' for NegativeBinomial family. Use 'log' or 'identity'.", other
                ))),
            }
        }
        other => return Err(PyValueError::new_err(format!(
            "Unknown family '{}'. Use 'gaussian', 'poisson', 'binomial', 'gamma', 'tweedie', 'quasipoisson', 'quasibinomial', or 'negbinomial'.", other
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

/// Fit Negative Binomial GLM with automatic theta estimation.
///
/// This function uses profile likelihood to automatically estimate the
/// optimal dispersion parameter θ, alternating between:
/// 1. Fitting the GLM with current θ to get β and μ
/// 2. Optimizing θ given μ using profile likelihood
/// 3. Repeating until θ converges
///
/// Parameters
/// ----------
/// y : array-like
///     Response variable (non-negative counts)
/// X : array-like
///     Design matrix (include intercept column if desired)
/// link : str, optional
///     Link function: "log" (default) or "identity"
/// init_theta : float, optional
///     Initial θ value (default: 1.0, uses method-of-moments if None)
/// theta_tol : float, optional
///     Convergence tolerance for θ (default: 1e-5)
/// max_theta_iter : int, optional
///     Maximum θ iterations (default: 10)
/// offset : array-like, optional
///     Offset term added to linear predictor
/// weights : array-like, optional
///     Prior weights for each observation
/// max_iter : int, optional
///     Maximum IRLS iterations per GLM fit (default: 25)
/// tol : float, optional
///     IRLS convergence tolerance (default: 1e-8)
///
/// Returns
/// -------
/// GLMResults
///     Fitted model with estimated theta accessible via result.theta
///
/// Examples
/// --------
/// This is an internal function. Use the formula API instead:
/// >>> import rustystats as rs
/// >>> # Auto-estimate theta (default when theta not supplied)
/// >>> result = rs.glm("y ~ x1 + x2", data, family="negbinomial").fit()
/// >>> print(f"Estimated theta: {result.theta:.3f}")
#[pyfunction]
#[pyo3(signature = (y, x, link=None, init_theta=None, theta_tol=1e-5, max_theta_iter=10, offset=None, weights=None, max_iter=25, tol=1e-8))]
fn fit_negbinomial_py(
    y: PyReadonlyArray1<f64>,
    x: PyReadonlyArray2<f64>,
    link: Option<&str>,
    init_theta: Option<f64>,
    theta_tol: f64,
    max_theta_iter: usize,
    offset: Option<PyReadonlyArray1<f64>>,
    weights: Option<PyReadonlyArray1<f64>>,
    max_iter: usize,
    tol: f64,
) -> PyResult<PyGLMResults> {
    let y_array: Array1<f64> = y.as_array().to_owned();
    let x_array: Array2<f64> = x.as_array().to_owned();

    let n_obs = y_array.len();
    let n_params = x_array.ncols();

    let offset_array: Option<Array1<f64>> = offset.map(|o| o.as_array().to_owned());
    let weights_array: Option<Array1<f64>> = weights.map(|w| w.as_array().to_owned());

    let irls_config = IRLSConfig {
        max_iterations: max_iter,
        tolerance: tol,
        min_weight: 1e-10,
        verbose: false,
    };

    // Get link (default to log for NegativeBinomial)
    let link_name = link.unwrap_or("log");
    let link_fn = link_from_name(link_name)?;
    if link_name != "log" && link_name != "identity" {
        return Err(PyValueError::new_err(format!(
            "Unknown link '{}' for NegativeBinomial. Use 'log' or 'identity'.", link_name
        )));
    }

    // Initialize theta using method-of-moments or provided value
    let mut theta = match init_theta {
        Some(t) => {
            if t <= 0.0 {
                return Err(PyValueError::new_err(
                    format!("init_theta must be > 0, got {}", t)
                ));
            }
            t
        }
        None => {
            // Use method-of-moments for initial estimate
            // First fit a Poisson to get initial mu
            let poisson = PoissonFamily;
            let init_result = fit_glm_full(
                &y_array, &x_array, &poisson, link_fn.as_ref(), &irls_config,
                offset_array.as_ref(), weights_array.as_ref(),
            ).map_err(|e| PyValueError::new_err(format!("Initial Poisson fit failed: {}", e)))?;
            
            estimate_theta_moments(&y_array, &init_result.fitted_values)
        }
    };

    let mut result: IRLSResult;

    // Profile likelihood iteration
    for _iter in 0..max_theta_iter {
        let family = NegativeBinomialFamily::new(theta);
        
        // Fit GLM with current theta
        result = fit_glm_full(
            &y_array, &x_array, &family, link_fn.as_ref(), &irls_config,
            offset_array.as_ref(), weights_array.as_ref(),
        ).map_err(|e| PyValueError::new_err(format!("GLM fitting failed: {}", e)))?;

        // Estimate optimal theta given fitted values
        let new_theta = estimate_theta_profile(
            &y_array,
            &result.fitted_values,
            weights_array.as_ref(),
            0.01,   // min_theta
            1000.0, // max_theta
            1e-6,   // optimization tolerance
        );

        // Check convergence
        if (new_theta - theta).abs() < theta_tol {
            theta = new_theta;
            break;
        }
        
        theta = new_theta;
    }

    // Final fit with converged theta
    let final_family = NegativeBinomialFamily::new(theta);
    result = fit_glm_full(
        &y_array, &x_array, &final_family, link_fn.as_ref(), &irls_config,
        offset_array.as_ref(), weights_array.as_ref(),
    ).map_err(|e| PyValueError::new_err(format!("Final GLM fit failed: {}", e)))?;

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
        y: y_array,
        family_name: format!("NegativeBinomial(theta={:.4})", theta),
        prior_weights: weights_array.unwrap_or_else(|| Array1::ones(n_obs)),
        penalty: result.penalty,
        design_matrix: x_array,
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
/// Use when indices are already computed (e.g., from factorization).
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
// Target Encoding (Ordered Target Statistics)
// =============================================================================

use rustystats_core::target_encoding;

/// Target encode categorical variables using ordered target statistics.
///
/// This encoding prevents target leakage during training by computing statistics
/// using only "past" observations in a random permutation order.
///
/// Parameters
/// ----------
/// categories : list[str]
///     Categorical values as strings
/// target : numpy.ndarray
///     Target variable (continuous or binary)
/// var_name : str
///     Variable name for output column
/// prior_weight : float, optional
///     Regularization strength toward global mean (default: 1.0)
/// n_permutations : int, optional
///     Number of random permutations to average (default: 4)
/// seed : int, optional
///     Random seed for reproducibility (default: None = random)
///
/// Returns
/// -------
/// tuple[numpy.ndarray, str, dict]
///     (encoded_values, column_name, level_stats)
///     level_stats is a dict mapping level -> (sum_target, count) for prediction
#[pyfunction]
#[pyo3(signature = (categories, target, var_name, prior_weight=1.0, n_permutations=4, seed=None))]
fn target_encode_py<'py>(
    py: Python<'py>,
    categories: Vec<String>,
    target: PyReadonlyArray1<f64>,
    var_name: &str,
    prior_weight: f64,
    n_permutations: usize,
    seed: Option<u64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, String, f64, std::collections::HashMap<String, (f64, usize)>)> {
    let target_vec: Vec<f64> = target.as_array().to_vec();
    
    let config = target_encoding::TargetEncodingConfig {
        prior_weight,
        n_permutations,
        seed,
    };
    
    let enc = target_encoding::target_encode(&categories, &target_vec, var_name, &config);
    
    // Convert level_stats to Python-friendly format
    let stats: std::collections::HashMap<String, (f64, usize)> = enc.level_stats
        .into_iter()
        .map(|(k, v)| (k, (v.sum_target, v.count)))
        .collect();
    
    Ok((
        enc.values.into_pyarray_bound(py),
        enc.name,
        enc.prior,
        stats,
    ))
}

/// Apply target encoding to new data using pre-computed statistics.
///
/// For prediction: uses full training statistics (no ordering needed).
///
/// Parameters
/// ----------
/// categories : list[str]
///     Categorical values for new data
/// level_stats : dict
///     Mapping of level -> (sum_target, count) from training
/// prior : float
///     Global prior (mean of training target)
/// prior_weight : float, optional
///     Prior weight (should match training, default: 1.0)
///
/// Returns
/// -------
/// numpy.ndarray
///     Encoded values for new data
#[pyfunction]
#[pyo3(signature = (categories, level_stats, prior, prior_weight=1.0))]
fn apply_target_encoding_py<'py>(
    py: Python<'py>,
    categories: Vec<String>,
    level_stats: std::collections::HashMap<String, (f64, usize)>,
    prior: f64,
    prior_weight: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let n = categories.len();
    let mut values = Vec::with_capacity(n);
    
    for cat in &categories {
        let encoded = if let Some(&(sum_target, count)) = level_stats.get(cat) {
            (sum_target + prior * prior_weight) / (count as f64 + prior_weight)
        } else {
            // Unseen category: use prior
            prior
        };
        values.push(encoded);
    }
    
    Ok(Array1::from_vec(values).into_pyarray_bound(py))
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
///     - target_encoding_terms: list[dict] with 'var_name', 'prior_weight', 'n_permutations'
///     - identity_terms: list[dict] with 'expression'
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
        
        // Convert target encoding terms
        let te_terms: Vec<_> = parsed.target_encoding_terms
            .into_iter()
            .map(|t| {
                let dict = PyDict::new_bound(py);
                dict.set_item("var_name", t.var_name).unwrap();
                dict.set_item("prior_weight", t.prior_weight).unwrap();
                dict.set_item("n_permutations", t.n_permutations).unwrap();
                dict.into_py(py)
            })
            .collect();
        result.insert("target_encoding_terms".to_string(), te_terms.into_py(py));
        
        // Convert identity terms (I() expressions)
        let identity_terms: Vec<_> = parsed.identity_terms
            .into_iter()
            .map(|i| {
                let dict = PyDict::new_bound(py);
                dict.set_item("expression", i.expression).unwrap();
                dict.into_py(py)
            })
            .collect();
        result.insert("identity_terms".to_string(), identity_terms.into_py(py));
        
        Ok(result)
    })
}

// =============================================================================
// Diagnostics Bindings
// =============================================================================

/// Compute calibration curve bins from Rust
#[pyfunction]
#[pyo3(signature = (y, mu, exposure=None, n_bins=10))]
fn compute_calibration_curve_py<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<f64>,
    mu: PyReadonlyArray1<f64>,
    exposure: Option<PyReadonlyArray1<f64>>,
    n_bins: usize,
) -> PyResult<Vec<PyObject>> {
    let y_arr = y.as_array().to_owned();
    let mu_arr = mu.as_array().to_owned();
    let exp_arr = exposure.map(|e| e.as_array().to_owned());
    
    let bins = compute_calibration_curve(
        &y_arr,
        &mu_arr,
        exp_arr.as_ref(),
        n_bins,
    );
    
    let result: Vec<PyObject> = bins.into_iter().map(|bin| {
        let dict = pyo3::types::PyDict::new_bound(py);
        dict.set_item("bin_index", bin.bin_index).unwrap();
        dict.set_item("predicted_lower", bin.predicted_lower).unwrap();
        dict.set_item("predicted_upper", bin.predicted_upper).unwrap();
        dict.set_item("predicted_mean", bin.predicted_mean).unwrap();
        dict.set_item("actual_mean", bin.actual_mean).unwrap();
        dict.set_item("actual_expected_ratio", bin.actual_expected_ratio).unwrap();
        dict.set_item("count", bin.count).unwrap();
        dict.set_item("exposure", bin.exposure).unwrap();
        dict.set_item("actual_sum", bin.actual_sum).unwrap();
        dict.set_item("predicted_sum", bin.predicted_sum).unwrap();
        dict.set_item("ae_ci_lower", bin.ae_ci_lower).unwrap();
        dict.set_item("ae_ci_upper", bin.ae_ci_upper).unwrap();
        dict.into_py(py)
    }).collect();
    
    Ok(result)
}

/// Compute discrimination stats (Gini, AUC, etc.) from Rust
#[pyfunction]
#[pyo3(signature = (y, mu, exposure=None))]
fn compute_discrimination_stats_py<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<f64>,
    mu: PyReadonlyArray1<f64>,
    exposure: Option<PyReadonlyArray1<f64>>,
) -> PyResult<PyObject> {
    let y_arr = y.as_array().to_owned();
    let mu_arr = mu.as_array().to_owned();
    let exp_arr = exposure.map(|e| e.as_array().to_owned());
    
    let stats = compute_discrimination_stats(&y_arr, &mu_arr, exp_arr.as_ref());
    
    let dict = pyo3::types::PyDict::new_bound(py);
    dict.set_item("gini", stats.gini_coefficient)?;
    dict.set_item("auc", stats.auc)?;
    dict.set_item("ks_statistic", stats.ks_statistic)?;
    dict.set_item("lift_at_10pct", stats.lift_at_10pct)?;
    dict.set_item("lift_at_20pct", stats.lift_at_20pct)?;
    
    Ok(dict.into_py(py))
}

/// Compute A/E bins for continuous factor from Rust
#[pyfunction]
#[pyo3(signature = (values, y, mu, exposure=None, n_bins=10, family="poisson"))]
fn compute_ae_continuous_py<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<f64>,
    y: PyReadonlyArray1<f64>,
    mu: PyReadonlyArray1<f64>,
    exposure: Option<PyReadonlyArray1<f64>>,
    n_bins: usize,
    family: &str,
) -> PyResult<Vec<PyObject>> {
    let values_arr = values.as_array().to_owned();
    let y_arr = y.as_array().to_owned();
    let mu_arr = mu.as_array().to_owned();
    let exp_arr = exposure.map(|e| e.as_array().to_owned());
    
    let bins = compute_ae_continuous(
        values_arr.as_slice().unwrap(),
        &y_arr,
        &mu_arr,
        exp_arr.as_ref(),
        family,
        n_bins,
        None,  // var_power
        None,  // theta
    );
    
    let result: Vec<PyObject> = bins.into_iter().map(|bin| {
        let dict = pyo3::types::PyDict::new_bound(py);
        dict.set_item("bin_index", bin.bin_index).unwrap();
        dict.set_item("bin_label", &bin.bin_label).unwrap();
        dict.set_item("bin_lower", bin.bin_lower).unwrap();
        dict.set_item("bin_upper", bin.bin_upper).unwrap();
        dict.set_item("count", bin.count).unwrap();
        dict.set_item("exposure", bin.exposure).unwrap();
        dict.set_item("actual_sum", bin.actual_sum).unwrap();
        dict.set_item("predicted_sum", bin.predicted_sum).unwrap();
        dict.set_item("actual_mean", bin.actual_mean).unwrap();
        dict.set_item("predicted_mean", bin.predicted_mean).unwrap();
        dict.set_item("actual_expected_ratio", bin.actual_expected_ratio).unwrap();
        dict.set_item("loss", bin.loss).unwrap();
        dict.set_item("ae_ci_lower", bin.ae_ci_lower).unwrap();
        dict.set_item("ae_ci_upper", bin.ae_ci_upper).unwrap();
        dict.into_py(py)
    }).collect();
    
    Ok(result)
}

/// Compute A/E bins for categorical factor from Rust
#[pyfunction]
#[pyo3(signature = (levels, y, mu, exposure=None, rare_threshold_pct=1.0, max_levels=20, family="poisson"))]
fn compute_ae_categorical_py<'py>(
    py: Python<'py>,
    levels: Vec<String>,
    y: PyReadonlyArray1<f64>,
    mu: PyReadonlyArray1<f64>,
    exposure: Option<PyReadonlyArray1<f64>>,
    rare_threshold_pct: f64,
    max_levels: usize,
    family: &str,
) -> PyResult<Vec<PyObject>> {
    let y_arr = y.as_array().to_owned();
    let mu_arr = mu.as_array().to_owned();
    let exp_arr = exposure.map(|e| e.as_array().to_owned());
    
    let bins = compute_ae_categorical(
        &levels,
        &y_arr,
        &mu_arr,
        exp_arr.as_ref(),
        family,
        None,  // var_power
        None,  // theta
        rare_threshold_pct,
        max_levels,
    );
    
    let result: Vec<PyObject> = bins.into_iter().map(|bin| {
        let dict = pyo3::types::PyDict::new_bound(py);
        dict.set_item("bin_index", bin.bin_index).unwrap();
        dict.set_item("bin_label", &bin.bin_label).unwrap();
        dict.set_item("count", bin.count).unwrap();
        dict.set_item("exposure", bin.exposure).unwrap();
        dict.set_item("actual_sum", bin.actual_sum).unwrap();
        dict.set_item("predicted_sum", bin.predicted_sum).unwrap();
        dict.set_item("actual_mean", bin.actual_mean).unwrap();
        dict.set_item("predicted_mean", bin.predicted_mean).unwrap();
        dict.set_item("actual_expected_ratio", bin.actual_expected_ratio).unwrap();
        dict.set_item("loss", bin.loss).unwrap();
        dict.set_item("ae_ci_lower", bin.ae_ci_lower).unwrap();
        dict.set_item("ae_ci_upper", bin.ae_ci_upper).unwrap();
        dict.into_py(py)
    }).collect();
    
    Ok(result)
}

/// Compute loss metrics from Rust
#[pyfunction]
fn compute_loss_metrics_py<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<f64>,
    mu: PyReadonlyArray1<f64>,
    family: &str,
) -> PyResult<PyObject> {
    let y_arr = y.as_array().to_owned();
    let mu_arr = mu.as_array().to_owned();
    
    let dict = pyo3::types::PyDict::new_bound(py);
    dict.set_item("mse", mse(&y_arr, &mu_arr, None))?;
    dict.set_item("rmse", rmse(&y_arr, &mu_arr, None))?;
    dict.set_item("mae", mae(&y_arr, &mu_arr, None))?;
    dict.set_item("family_loss", compute_family_loss(family, &y_arr, &mu_arr, None, None, None))?;
    
    Ok(dict.into_py(py))
}

/// Detect interactions from Rust
#[pyfunction]
#[pyo3(signature = (residuals, factor_names, factor_values, factor_is_categorical, max_factors=10, max_candidates=5))]
fn detect_interactions_py<'py>(
    py: Python<'py>,
    residuals: PyReadonlyArray1<f64>,
    factor_names: Vec<String>,
    factor_values: Vec<Vec<String>>,
    factor_is_categorical: Vec<bool>,
    max_factors: usize,
    max_candidates: usize,
) -> PyResult<Vec<PyObject>> {
    let resid_arr = residuals.as_array().to_owned();
    
    use std::collections::HashMap;
    let mut factors: HashMap<String, FactorData> = HashMap::new();
    for (i, name) in factor_names.iter().enumerate() {
        let is_cat = factor_is_categorical.get(i).copied().unwrap_or(false);
        let values = factor_values.get(i).cloned().unwrap_or_default();
        if is_cat {
            factors.insert(name.clone(), FactorData::Categorical(values));
        } else {
            // Parse as f64
            let floats: Vec<f64> = values.iter().map(|s| s.parse().unwrap_or(f64::NAN)).collect();
            factors.insert(name.clone(), FactorData::Continuous(floats));
        }
    }
    
    let config = InteractionConfig {
        max_factors_to_check: max_factors,
        min_residual_correlation: 0.01,
        max_candidates,
        min_cell_count: 30,
    };
    
    let interactions = detect_interactions(&factors, &resid_arr, &config);
    
    let result: Vec<PyObject> = interactions.into_iter().map(|int| {
        let dict = pyo3::types::PyDict::new_bound(py);
        dict.set_item("factor1", &int.factor1).unwrap();
        dict.set_item("factor2", &int.factor2).unwrap();
        dict.set_item("strength", int.interaction_strength).unwrap();
        dict.set_item("pvalue", int.pvalue).unwrap();
        dict.set_item("n_cells", int.n_cells).unwrap();
        dict.into_py(py)
    }).collect();
    
    Ok(result)
}

/// Compute Lorenz curve from Rust
#[pyfunction]
#[pyo3(signature = (y, mu, exposure=None, n_points=20))]
fn compute_lorenz_curve_py<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<f64>,
    mu: PyReadonlyArray1<f64>,
    exposure: Option<PyReadonlyArray1<f64>>,
    n_points: usize,
) -> PyResult<Vec<PyObject>> {
    let y_arr = y.as_array().to_owned();
    let mu_arr = mu.as_array().to_owned();
    let exp_arr = exposure.map(|e| e.as_array().to_owned());
    
    let points = compute_lorenz_curve(&y_arr, &mu_arr, exp_arr.as_ref(), n_points);
    
    let result: Vec<PyObject> = points.into_iter().map(|p| {
        let dict = pyo3::types::PyDict::new_bound(py);
        dict.set_item("cumulative_exposure_pct", p.cumulative_exposure_pct).unwrap();
        dict.set_item("cumulative_actual_pct", p.cumulative_actual_pct).unwrap();
        dict.set_item("cumulative_predicted_pct", p.cumulative_predicted_pct).unwrap();
        dict.into_py(py)
    }).collect();
    
    Ok(result)
}

/// Compute Hosmer-Lemeshow test from Rust
#[pyfunction]
fn hosmer_lemeshow_test_py<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<f64>,
    mu: PyReadonlyArray1<f64>,
    n_bins: usize,
) -> PyResult<PyObject> {
    let y_arr = y.as_array().to_owned();
    let mu_arr = mu.as_array().to_owned();
    
    let result = hosmer_lemeshow_test(&y_arr, &mu_arr, n_bins);
    
    let dict = pyo3::types::PyDict::new_bound(py);
    dict.set_item("chi2_statistic", result.statistic)?;
    dict.set_item("pvalue", result.pvalue)?;
    dict.set_item("degrees_of_freedom", result.degrees_of_freedom)?;
    
    Ok(dict.into_py(py))
}

/// Compute fit statistics from Rust
#[pyfunction]
fn compute_fit_statistics_py<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<f64>,
    mu: PyReadonlyArray1<f64>,
    deviance: f64,
    null_dev: f64,
    n_params: usize,
    family: &str,
) -> PyResult<PyObject> {
    let y_arr = y.as_array().to_owned();
    let mu_arr = mu.as_array().to_owned();
    let n_obs = y_arr.len();
    let df_resid = n_obs.saturating_sub(n_params);
    
    // Compute pearson chi2 based on family
    let fam = family_from_name(family)?;
    let pchi2 = pearson_chi2(&y_arr, &mu_arr, fam.as_ref(), None);
    
    // Compute log-likelihood based on family
    // Use estimated scale for gaussian
    let scale = if df_resid > 0 {
        y_arr.iter().zip(mu_arr.iter()).map(|(y, m)| (y - m).powi(2)).sum::<f64>() / df_resid as f64
    } else { 1.0 };
    
    let llf = match family.to_lowercase().as_str() {
        "gaussian" | "normal" => log_likelihood_gaussian(&y_arr, &mu_arr, scale, None),
        "poisson" | "quasipoisson" => log_likelihood_poisson(&y_arr, &mu_arr, None),
        "binomial" | "quasibinomial" => log_likelihood_binomial(&y_arr, &mu_arr, None),
        "gamma" => log_likelihood_gamma(&y_arr, &mu_arr, scale, None),
        _ => -deviance / 2.0,
    };
    
    let aic_val = aic(llf, n_params);
    let bic_val = bic(llf, n_params, n_obs);
    
    let deviance_explained = if null_dev > 0.0 { 1.0 - deviance / null_dev } else { 0.0 };
    let dispersion_deviance = if df_resid > 0 { deviance / df_resid as f64 } else { 1.0 };
    let dispersion_pearson = if df_resid > 0 { pchi2 / df_resid as f64 } else { 1.0 };
    
    let dict = pyo3::types::PyDict::new_bound(py);
    dict.set_item("deviance", deviance)?;
    dict.set_item("null_deviance", null_dev)?;
    dict.set_item("deviance_explained", deviance_explained)?;
    dict.set_item("log_likelihood", llf)?;
    dict.set_item("aic", aic_val)?;
    dict.set_item("bic", bic_val)?;
    dict.set_item("pearson_chi2", pchi2)?;
    dict.set_item("dispersion_deviance", dispersion_deviance)?;
    dict.set_item("dispersion_pearson", dispersion_pearson)?;
    
    Ok(dict.into_py(py))
}

/// Compute residual summary statistics from Rust
#[pyfunction]
fn compute_residual_summary_py<'py>(
    py: Python<'py>,
    residuals: PyReadonlyArray1<f64>,
) -> PyResult<PyObject> {
    let resid = residuals.as_array();
    let n = resid.len() as f64;
    
    if n == 0.0 {
        return Err(PyValueError::new_err("Empty residuals array"));
    }
    
    let mean = resid.iter().sum::<f64>() / n;
    let variance = resid.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let std = variance.sqrt();
    let min = resid.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = resid.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    
    // Skewness and kurtosis
    let skewness = if std > 0.0 {
        resid.iter().map(|x| ((x - mean) / std).powi(3)).sum::<f64>() / n
    } else { 0.0 };
    
    let kurtosis = if std > 0.0 {
        resid.iter().map(|x| ((x - mean) / std).powi(4)).sum::<f64>() / n - 3.0
    } else { 0.0 };
    
    // Percentiles
    let mut sorted: Vec<f64> = resid.iter().cloned().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    
    let percentile = |p: f64| -> f64 {
        let idx = (p / 100.0 * (sorted.len() - 1) as f64).round() as usize;
        sorted[idx.min(sorted.len() - 1)]
    };
    
    let dict = pyo3::types::PyDict::new_bound(py);
    dict.set_item("mean", mean)?;
    dict.set_item("std", std)?;
    dict.set_item("min", min)?;
    dict.set_item("max", max)?;
    dict.set_item("skewness", skewness)?;
    dict.set_item("kurtosis", kurtosis)?;
    dict.set_item("p1", percentile(1.0))?;
    dict.set_item("p5", percentile(5.0))?;
    dict.set_item("p10", percentile(10.0))?;
    dict.set_item("p25", percentile(25.0))?;
    dict.set_item("p50", percentile(50.0))?;
    dict.set_item("p75", percentile(75.0))?;
    dict.set_item("p90", percentile(90.0))?;
    dict.set_item("p95", percentile(95.0))?;
    dict.set_item("p99", percentile(99.0))?;
    
    Ok(dict.into_py(py))
}

/// Compute residual pattern for continuous factor from Rust
#[pyfunction]
#[pyo3(signature = (values, residuals, n_bins=10))]
fn compute_residual_pattern_py<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<f64>,
    residuals: PyReadonlyArray1<f64>,
    n_bins: usize,
) -> PyResult<PyObject> {
    let values_arr = values.as_array().to_owned();
    let resid_arr = residuals.as_array().to_owned();
    
    let pattern = compute_residual_pattern_continuous(
        values_arr.as_slice().unwrap(),
        &resid_arr,
        n_bins,
    );
    
    let dict = pyo3::types::PyDict::new_bound(py);
    dict.set_item("correlation_with_residuals", pattern.correlation_with_residuals)?;
    
    let means: Vec<PyObject> = pattern.mean_residual_by_bin.into_iter().enumerate().map(|(i, m)| {
        let d = pyo3::types::PyDict::new_bound(py);
        d.set_item("bin_index", i).unwrap();
        d.set_item("mean_residual", m).unwrap();
        d.into_py(py)
    }).collect();
    
    dict.set_item("mean_residual_by_bin", means)?;
    
    Ok(dict.into_py(py))
}

/// Compute Pearson residuals from Rust
#[pyfunction]
fn compute_pearson_residuals_py<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<f64>,
    mu: PyReadonlyArray1<f64>,
    family: &str,
) -> PyResult<Py<PyArray1<f64>>> {
    let y_arr = y.as_array().to_owned();
    let mu_arr = mu.as_array().to_owned();
    let fam = family_from_name(family)?;
    let resid = resid_pearson(&y_arr, &mu_arr, fam.as_ref());
    Ok(resid.into_pyarray_bound(py).unbind())
}

/// Compute deviance residuals from Rust
#[pyfunction]
fn compute_deviance_residuals_py<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<f64>,
    mu: PyReadonlyArray1<f64>,
    family: &str,
) -> PyResult<Py<PyArray1<f64>>> {
    let y_arr = y.as_array().to_owned();
    let mu_arr = mu.as_array().to_owned();
    let fam = family_from_name(family)?;
    let resid = resid_deviance(&y_arr, &mu_arr, fam.as_ref());
    Ok(resid.into_pyarray_bound(py).unbind())
}

/// Compute null deviance from Rust
#[pyfunction]
#[pyo3(signature = (y, family, exposure=None))]
fn compute_null_deviance_py(
    y: PyReadonlyArray1<f64>,
    family: &str,
    exposure: Option<PyReadonlyArray1<f64>>,
) -> PyResult<f64> {
    let y_arr = y.as_array().to_owned();
    let exp_arr = exposure.map(|e| e.as_array().to_owned());
    
    Ok(null_deviance(&y_arr, family, exp_arr.as_ref()))
}

/// Compute unit deviance from Rust
#[pyfunction]
fn compute_unit_deviance_py<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<f64>,
    mu: PyReadonlyArray1<f64>,
    family: &str,
) -> PyResult<Py<PyArray1<f64>>> {
    let y_arr = y.as_array().to_owned();
    let mu_arr = mu.as_array().to_owned();
    let fam = family_from_name(family)?;
    let unit_dev = fam.unit_deviance(&y_arr, &mu_arr);
    Ok(unit_dev.into_pyarray_bound(py).unbind())
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
    m.add_class::<PyQuasiPoissonFamily>()?;
    m.add_class::<PyQuasiBinomialFamily>()?;
    m.add_class::<PyNegativeBinomialFamily>()?;
    
    // Add GLM fitting
    m.add_class::<PyGLMResults>()?;
    m.add_function(wrap_pyfunction!(fit_glm_py, m)?)?;
    m.add_function(wrap_pyfunction!(fit_negbinomial_py, m)?)?;
    
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
    
    // Add target encoding
    m.add_function(wrap_pyfunction!(target_encode_py, m)?)?;
    m.add_function(wrap_pyfunction!(apply_target_encoding_py, m)?)?;
    
    // Add diagnostics functions
    m.add_function(wrap_pyfunction!(compute_calibration_curve_py, m)?)?;
    m.add_function(wrap_pyfunction!(compute_discrimination_stats_py, m)?)?;
    m.add_function(wrap_pyfunction!(compute_ae_continuous_py, m)?)?;
    m.add_function(wrap_pyfunction!(compute_ae_categorical_py, m)?)?;
    m.add_function(wrap_pyfunction!(compute_loss_metrics_py, m)?)?;
    m.add_function(wrap_pyfunction!(detect_interactions_py, m)?)?;
    m.add_function(wrap_pyfunction!(compute_lorenz_curve_py, m)?)?;
    m.add_function(wrap_pyfunction!(hosmer_lemeshow_test_py, m)?)?;
    m.add_function(wrap_pyfunction!(compute_fit_statistics_py, m)?)?;
    m.add_function(wrap_pyfunction!(compute_residual_summary_py, m)?)?;
    m.add_function(wrap_pyfunction!(compute_residual_pattern_py, m)?)?;
    m.add_function(wrap_pyfunction!(compute_pearson_residuals_py, m)?)?;
    m.add_function(wrap_pyfunction!(compute_deviance_residuals_py, m)?)?;
    m.add_function(wrap_pyfunction!(compute_null_deviance_py, m)?)?;
    m.add_function(wrap_pyfunction!(compute_unit_deviance_py, m)?)?;
    
    Ok(())
}
