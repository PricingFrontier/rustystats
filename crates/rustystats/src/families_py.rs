// =============================================================================
// Family and Link Function Python Wrappers
// =============================================================================
//
// These wrap the Rust distribution families and link functions for Python.
// Helper functions consolidate family/link dispatch logic used across the crate.
// =============================================================================

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};

use rustystats_core::families::{Family, GaussianFamily, PoissonFamily, BinomialFamily, GammaFamily, TweedieFamily, QuasiPoissonFamily, QuasiBinomialFamily, NegativeBinomialFamily};
use rustystats_core::links::{Link, IdentityLink, LogLink, LogitLink};

// =============================================================================
// Family and Link Helper Functions
// =============================================================================

/// Get a Family trait object from a family name string.
/// 
/// Handles case-insensitive matching and common aliases.
/// `var_power` is used only for Tweedie; `theta` only for NegativeBinomial.
/// Returns an error for unknown family names instead of silently defaulting.
pub(crate) fn family_from_name(name: &str, var_power: f64, theta: f64) -> PyResult<Box<dyn Family>> {
    let lower = name.to_lowercase();
    
    // Handle negativebinomial with optional theta parameter like "negativebinomial(theta=1.38)"
    if lower.starts_with("negativebinomial") || lower.starts_with("negbinomial") || lower.starts_with("negbin")
        || lower == "nb" || lower == "negative_binomial" || lower == "neg_binomial" || lower == "neg-binomial"
    {
        // Parse theta from name if present, otherwise use the `theta` arg
        let parsed_theta = if let Some(start) = lower.find("theta=") {
            let rest = &lower[start + 6..];
            let end = rest.find(')').unwrap_or(rest.len());
            let theta_str = &rest[..end];
            theta_str.parse::<f64>().map_err(|_| {
                PyValueError::new_err(format!(
                    "Failed to parse theta value '{}' in family '{}'. Expected a numeric value like 'negativebinomial(theta=1.5)'",
                    theta_str, name
                ))
            })?
        } else {
            theta
        };
        if parsed_theta <= 0.0 {
            return Err(PyValueError::new_err(format!("theta must be > 0 for Negative Binomial, got {}", parsed_theta)));
        }
        return Ok(Box::new(NegativeBinomialFamily::new(parsed_theta)));
    }
    
    match lower.as_str() {
        "gaussian" | "normal" => Ok(Box::new(GaussianFamily)),
        "poisson" => Ok(Box::new(PoissonFamily)),
        "binomial" => Ok(Box::new(BinomialFamily)),
        "gamma" => Ok(Box::new(GammaFamily)),
        "tweedie" => {
            if var_power > 0.0 && var_power < 1.0 {
                return Err(PyValueError::new_err(format!("var_power must be <= 0 or >= 1, got {}", var_power)));
            }
            Ok(Box::new(TweedieFamily::new(var_power)))
        }
        "quasipoisson" | "quasi-poisson" | "quasi_poisson" => Ok(Box::new(QuasiPoissonFamily)),
        "quasibinomial" | "quasi-binomial" | "quasi_binomial" => Ok(Box::new(QuasiBinomialFamily)),
        _ => Err(PyValueError::new_err(format!(
            "Unknown family '{}'. Use 'gaussian', 'poisson', 'binomial', 'gamma', 'tweedie', \
             'quasipoisson', 'quasibinomial', or 'negativebinomial'.", name
        ))),
    }
}

/// Get a Link trait object from a link name string.
/// Returns an error for unknown link names instead of silently defaulting.
pub(crate) fn link_from_name(name: &str) -> PyResult<Box<dyn Link>> {
    match name.to_lowercase().as_str() {
        "identity" => Ok(Box::new(IdentityLink)),
        "log" => Ok(Box::new(LogLink)),
        "logit" => Ok(Box::new(LogitLink)),
        _ => Err(PyValueError::new_err(format!(
            "Unknown link '{}'. Use 'identity', 'log', or 'logit'.", name
        ))),
    }
}

/// Return the default link name for a given family name.
pub(crate) fn default_link_name(family: &str) -> &'static str {
    match family.to_lowercase().as_str() {
        "gaussian" | "normal" => "identity",
        "binomial" | "quasibinomial" | "quasi-binomial" | "quasi_binomial" => "logit",
        _ => "log",  // poisson, gamma, tweedie, quasipoisson, negbinomial, etc.
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
            pub fn new() -> Self {
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
