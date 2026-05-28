// =============================================================================
// Family and Link Function Python Wrappers
// =============================================================================
//
// These wrap the Rust distribution families and link functions for Python.
// Helper functions consolidate family/link dispatch logic used across the crate.
// =============================================================================

use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use rustystats_core::families::{
    BinomialFamily, Family, GammaFamily, GaussianFamily, NegativeBinomialFamily, PoissonFamily,
    QuasiBinomialFamily, QuasiPoissonFamily, TweedieFamily,
};
use rustystats_core::links::{IdentityLink, Link, LogLink, LogitLink};

// =============================================================================
// Family and Link Helper Functions
// =============================================================================

/// Split a family name into its base name and optional parenthesized parameter.
///
/// Accepts exact forms like `tweedie` or `tweedie(p=1.5)`. Anything with a
/// dangling, nested, or trailing parenthesis is rejected instead of being
/// silently interpreted as a valid family.
fn split_family_name(name: &str) -> PyResult<(&str, Option<&str>)> {
    let trimmed = name.trim();
    let Some(open_idx) = trimmed.find('(') else {
        if trimmed.contains(')') {
            return Err(PyValueError::new_err(format!(
                "Malformed family '{}'. Unexpected ')' without matching '('.",
                name
            )));
        }
        return Ok((trimmed, None));
    };

    if !trimmed.ends_with(')') {
        return Err(PyValueError::new_err(format!(
            "Malformed family '{}'. Expected embedded parameters to end with ')'.",
            name
        )));
    }

    let base = trimmed[..open_idx].trim();
    let params = trimmed[open_idx + 1..trimmed.len() - 1].trim();
    if base.is_empty() || params.is_empty() || params.contains('(') || params.contains(')') {
        return Err(PyValueError::new_err(format!(
            "Malformed family '{}'. Expected a form like 'tweedie(p=1.5)' or \
             'negativebinomial(theta=1.5)'.",
            name
        )));
    }

    Ok((base, Some(params)))
}

/// Parse an embedded numeric parameter like `p=1.5` or `theta=2.0` from a
/// parenthesized family-name suffix. Returns `Ok(Some(value))` if the suffix is
/// present and uses the expected key, `Ok(None)` if no suffix is present, or a
/// `PyValueError` if the suffix is malformed. Case-insensitive on the key.
///
/// Used to support equivalent forms like:
///   `family="negativebinomial(theta=1.5)"` ≡ `family="negbinomial", theta=1.5`
///   `family="tweedie(p=1.5)"`              ≡ `family="tweedie",     var_power=1.5`
fn parse_embedded_param(name: &str, params: Option<&str>, key: &str) -> PyResult<Option<f64>> {
    let Some(params) = params else {
        return Ok(None);
    };

    let Some((param_key, value_str)) = params.split_once('=') else {
        return Err(PyValueError::new_err(format!(
            "Malformed parameter '{}' in family '{}'. Expected '{}=<number>'.",
            params, name, key
        )));
    };
    if !param_key.trim().eq_ignore_ascii_case(key) {
        return Err(PyValueError::new_err(format!(
            "Unexpected parameter '{}' in family '{}'. Expected '{}'.",
            param_key.trim(),
            name,
            key
        )));
    }

    let value_str = value_str.trim();
    if value_str.is_empty() {
        return Err(PyValueError::new_err(format!(
            "Missing {} value in family '{}'. Expected a numeric value.",
            key, name
        )));
    }

    value_str.parse::<f64>().map(Some).map_err(|_| {
        PyValueError::new_err(format!(
            "Failed to parse {} value '{}' in family '{}'. Expected a numeric value.",
            key, value_str, name
        ))
    })
}

fn tweedie_alternative_hint(var_power: f64) -> Option<&'static str> {
    if (var_power - 0.0).abs() < 1e-12 {
        Some("rs.GaussianFamily()")
    } else if (var_power - 1.0).abs() < 1e-12 {
        Some("rs.PoissonFamily()")
    } else if (var_power - 2.0).abs() < 1e-12 {
        Some("rs.GammaFamily()")
    } else {
        None
    }
}

pub(crate) fn validate_tweedie_power(var_power: f64, allow_extended_tweedie: bool) -> PyResult<()> {
    if !var_power.is_finite() {
        return Err(PyValueError::new_err(format!(
            "var_power must be finite, got {}",
            var_power
        )));
    }
    if var_power > 0.0 && var_power < 1.0 {
        return Err(PyValueError::new_err(format!(
            "Tweedie var_power={} is in the open interval (0, 1) — no \
             Tweedie distribution exists for these powers. Allowed: \
             p <= 0, p == 1 (Poisson), 1 < p < 2 (compound Poisson-Gamma), \
             p == 2 (Gamma), p > 2.",
            var_power
        )));
    }

    let in_interior = var_power > 1.0 && var_power < 2.0;
    if !in_interior && !allow_extended_tweedie {
        let hint = tweedie_alternative_hint(var_power)
            .map(|s| format!(" Use {} directly.", s))
            .unwrap_or_default();
        return Err(PyValueError::new_err(format!(
            "Tweedie var_power={} is outside the default compound \
             Poisson-Gamma interior (1 < p < 2). Pass \
             allow_extended_tweedie=True to opt in to the extended regime \
             (and its per-regime support rules).{}",
            var_power, hint
        )));
    }

    Ok(())
}

pub(crate) fn validate_tweedie_response(y: &Array1<f64>, var_power: f64) -> PyResult<()> {
    if var_power >= 2.0 {
        let n_invalid = y.iter().filter(|&&v| v <= 0.0).count();
        if n_invalid > 0 {
            return Err(PyValueError::new_err(format!(
                "Extended Tweedie with p={} requires strictly positive response \
                 values (y > 0). Found {} values <= 0. The Tweedie unit deviance \
                 at y == 0 diverges for p >= 2; filter or cap your zeros before fitting.",
                var_power, n_invalid
            )));
        }
    } else {
        let n_neg = y.iter().filter(|&&v| v < 0.0).count();
        if n_neg > 0 {
            return Err(PyValueError::new_err(format!(
                "Tweedie family requires non-negative response values (y >= 0). \
                 Found {} negative values.",
                n_neg
            )));
        }
    }
    Ok(())
}

pub(crate) fn resolve_tweedie_var_power(name: &str, var_power: f64) -> PyResult<Option<f64>> {
    let (base_name, params) = split_family_name(name)?;
    let lower = base_name.to_lowercase();
    if lower != "tweedie" {
        return Ok(None);
    }
    Ok(Some(
        parse_embedded_param(name, params, "p")?.unwrap_or(var_power),
    ))
}

pub(crate) fn validate_tweedie_fit_response(
    name: &str,
    y: &Array1<f64>,
    var_power: f64,
    allow_extended_tweedie: bool,
) -> PyResult<()> {
    if let Some(resolved_var_power) = resolve_tweedie_var_power(name, var_power)? {
        validate_tweedie_power(resolved_var_power, allow_extended_tweedie)?;
        validate_tweedie_response(y, resolved_var_power)?;
    }
    Ok(())
}

/// Get a Family trait object from a family name string.
///
/// Handles case-insensitive matching and common aliases.
/// `var_power` is used only for Tweedie; `theta` only for NegativeBinomial.
/// Parametric families accept an embedded form too — see
/// [`parse_embedded_param`].
///
/// Tweedie powers outside the default compound Poisson-Gamma interior
/// (`1 < p < 2`) require `allow_extended_tweedie=true` and are validated
/// against the per-regime support rules — see [`validate_tweedie_power`].
///
/// Returns an error for unknown family names instead of silently defaulting.
pub(crate) fn family_from_name_with_tweedie_support(
    name: &str,
    var_power: f64,
    theta: f64,
    allow_extended_tweedie: bool,
) -> PyResult<Box<dyn Family>> {
    let (base_name, params) = split_family_name(name)?;
    let lower = base_name.to_lowercase();

    // Handle negativebinomial with optional embedded theta like "negativebinomial(theta=1.38)"
    if matches!(
        lower.as_str(),
        "negativebinomial"
            | "negbinomial"
            | "negbin"
            | "nb"
            | "negative_binomial"
            | "negative-binomial"
            | "neg_binomial"
            | "neg-binomial"
    ) {
        let resolved_theta = parse_embedded_param(name, params, "theta")?.unwrap_or(theta);
        if !resolved_theta.is_finite() || resolved_theta <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "theta must be finite and > 0 for Negative Binomial, got {}",
                resolved_theta
            )));
        }
        return Ok(Box::new(
            NegativeBinomialFamily::new(resolved_theta).map_err(PyValueError::new_err)?,
        ));
    }

    // Handle tweedie with optional embedded variance power like "tweedie(p=1.5)".
    if lower == "tweedie" {
        let resolved_var_power = parse_embedded_param(name, params, "p")?.unwrap_or(var_power);
        validate_tweedie_power(resolved_var_power, allow_extended_tweedie)?;
        return Ok(Box::new(
            TweedieFamily::new(resolved_var_power).map_err(PyValueError::new_err)?,
        ));
    }

    if params.is_some() {
        return Err(PyValueError::new_err(format!(
            "Family '{}' does not accept embedded parameters.",
            base_name
        )));
    }

    match lower.as_str() {
        "gaussian" | "normal" => Ok(Box::new(GaussianFamily)),
        "poisson" => Ok(Box::new(PoissonFamily)),
        "binomial" => Ok(Box::new(BinomialFamily)),
        "gamma" => Ok(Box::new(GammaFamily)),
        "quasipoisson" | "quasi-poisson" | "quasi_poisson" => Ok(Box::new(QuasiPoissonFamily)),
        "quasibinomial" | "quasi-binomial" | "quasi_binomial" => Ok(Box::new(QuasiBinomialFamily)),
        _ => Err(PyValueError::new_err(format!(
            "Unknown family '{}'. Use 'gaussian', 'poisson', 'binomial', 'gamma', 'tweedie', \
             'quasipoisson', 'quasibinomial', or 'negativebinomial'.",
            name
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
            "Unknown link '{}'. Use 'identity', 'log', or 'logit'.",
            name
        ))),
    }
}

/// Return the default link name for a given family name.
pub(crate) fn default_link_name(family: &str) -> &'static str {
    match family.to_lowercase().as_str() {
        "gaussian" | "normal" => "identity",
        "binomial" | "quasibinomial" | "quasi-binomial" | "quasi_binomial" => "logit",
        _ => "log", // poisson, gamma, tweedie, quasipoisson, negbinomial, etc.
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
        #[pyclass(name = $py_str, skip_from_py_object)]
        #[derive(Clone)]
        pub struct $py_name {
            inner: $inner_type,
        }

        impl Default for $py_name {
            fn default() -> Self {
                Self::new()
            }
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

            fn link<'py>(
                &self,
                py: Python<'py>,
                mu: PyReadonlyArray1<f64>,
            ) -> Bound<'py, PyArray1<f64>> {
                self.inner.link(&mu.as_array().to_owned()).into_pyarray(py)
            }

            fn inverse<'py>(
                &self,
                py: Python<'py>,
                eta: PyReadonlyArray1<f64>,
            ) -> Bound<'py, PyArray1<f64>> {
                self.inner
                    .inverse(&eta.as_array().to_owned())
                    .into_pyarray(py)
            }

            fn derivative<'py>(
                &self,
                py: Python<'py>,
                mu: PyReadonlyArray1<f64>,
            ) -> Bound<'py, PyArray1<f64>> {
                self.inner
                    .derivative(&mu.as_array().to_owned())
                    .into_pyarray(py)
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
        #[pyclass(name = $py_str, skip_from_py_object)]
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

            fn variance<'py>(
                &self,
                py: Python<'py>,
                mu: PyReadonlyArray1<f64>,
            ) -> Bound<'py, PyArray1<f64>> {
                self.inner
                    .variance(&mu.as_array().to_owned())
                    .into_owned()
                    .into_pyarray(py)
            }

            fn unit_deviance<'py>(
                &self,
                py: Python<'py>,
                y: PyReadonlyArray1<f64>,
                mu: PyReadonlyArray1<f64>,
            ) -> Bound<'py, PyArray1<f64>> {
                self.inner
                    .unit_deviance(&y.as_array().to_owned(), &mu.as_array().to_owned())
                    .into_pyarray(py)
            }

            fn deviance(&self, y: PyReadonlyArray1<f64>, mu: PyReadonlyArray1<f64>) -> f64 {
                self.inner
                    .deviance(&y.as_array().to_owned(), &mu.as_array().to_owned(), None)
            }

            fn default_link(&self) -> $default_link {
                <$default_link>::new()
            }
        }
    };
}

// Generate simple family wrappers (6 types × ~50 lines = ~300 lines → ~6 lines each)
impl_py_family!(
    PyGaussianFamily,
    "GaussianFamily",
    GaussianFamily,
    GaussianFamily,
    PyIdentityLink
);
impl_py_family!(
    PyPoissonFamily,
    "PoissonFamily",
    PoissonFamily,
    PoissonFamily,
    PyLogLink
);
impl_py_family!(
    PyBinomialFamily,
    "BinomialFamily",
    BinomialFamily,
    BinomialFamily,
    PyLogitLink
);
impl_py_family!(
    PyGammaFamily,
    "GammaFamily",
    GammaFamily,
    GammaFamily,
    PyLogLink
);
impl_py_family!(
    PyQuasiPoissonFamily,
    "QuasiPoissonFamily",
    QuasiPoissonFamily,
    QuasiPoissonFamily,
    PyLogLink
);
impl_py_family!(
    PyQuasiBinomialFamily,
    "QuasiBinomialFamily",
    QuasiBinomialFamily,
    QuasiBinomialFamily,
    PyLogitLink
);

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
#[pyclass(name = "TweedieFamily", skip_from_py_object)]
#[derive(Clone)]
pub struct PyTweedieFamily {
    inner: TweedieFamily,
}

#[pymethods]
impl PyTweedieFamily {
    /// Construct a Tweedie family.
    ///
    /// Parameters
    /// ----------
    /// var_power : float, default 1.5
    ///     Variance power ``p``. The default ``1 < p < 2`` interior covers the
    ///     compound Poisson-Gamma actuarial pure-premium regime.
    /// allow_extended_tweedie : bool, default False
    ///     Set ``True`` to opt in to the extended regimes ``p <= 0``,
    ///     ``p == 1``, ``p == 2``, and ``p > 2``. The genuinely invalid band
    ///     ``0 < p < 1`` (no Tweedie distribution exists) is rejected always.
    ///
    /// Notes
    /// -----
    /// Per-regime support rules on the response (``y >= 0`` for the interior
    /// and the low-p extended regimes; ``y > 0`` for ``p >= 2`` because the
    /// Tweedie unit deviance at ``y == 0`` diverges) are enforced at fit time
    /// by ``rustystats.glm_dict``. Constructing a ``TweedieFamily`` here only
    /// gates the ``var_power`` itself.
    #[new]
    #[pyo3(signature = (var_power=1.5, allow_extended_tweedie=false))]
    fn new(var_power: f64, allow_extended_tweedie: bool) -> PyResult<Self> {
        validate_tweedie_power(var_power, allow_extended_tweedie)?;
        Ok(Self {
            inner: TweedieFamily::new(var_power).map_err(PyValueError::new_err)?,
        })
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    /// Get the variance power parameter
    #[getter]
    fn var_power(&self) -> f64 {
        self.inner.var_power
    }

    fn variance<'py>(
        &self,
        py: Python<'py>,
        mu: PyReadonlyArray1<f64>,
    ) -> Bound<'py, PyArray1<f64>> {
        let mu_array = mu.as_array().to_owned();
        let result = self.inner.variance(&mu_array);
        result.into_owned().into_pyarray(py)
    }

    fn unit_deviance<'py>(
        &self,
        py: Python<'py>,
        y: PyReadonlyArray1<f64>,
        mu: PyReadonlyArray1<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let y_array = y.as_array().to_owned();
        validate_tweedie_response(&y_array, self.inner.var_power)?;
        let mu_array = mu.as_array().to_owned();
        let result = self.inner.unit_deviance(&y_array, &mu_array);
        Ok(result.into_pyarray(py))
    }

    fn deviance(&self, y: PyReadonlyArray1<f64>, mu: PyReadonlyArray1<f64>) -> PyResult<f64> {
        let y_array = y.as_array().to_owned();
        validate_tweedie_response(&y_array, self.inner.var_power)?;
        let mu_array = mu.as_array().to_owned();
        Ok(self.inner.deviance(&y_array, &mu_array, None))
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
#[pyclass(name = "NegativeBinomialFamily", skip_from_py_object)]
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
            return Err(PyValueError::new_err(format!(
                "theta must be > 0, got {}",
                theta
            )));
        }
        Ok(Self {
            inner: NegativeBinomialFamily::new(theta).map_err(PyValueError::new_err)?,
        })
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

    fn variance<'py>(
        &self,
        py: Python<'py>,
        mu: PyReadonlyArray1<f64>,
    ) -> Bound<'py, PyArray1<f64>> {
        let mu_array = mu.as_array().to_owned();
        let result = self.inner.variance(&mu_array);
        result.into_owned().into_pyarray(py)
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
        result.into_pyarray(py)
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
