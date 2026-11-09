// =============================================================================
// GLM Results
// =============================================================================
//
// This class holds the results of fitting a GLM.
// It provides access to coefficients, fitted values, and diagnostic info.
// =============================================================================

use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use rustystats_core::diagnostics::{
    aic, bic, estimate_dispersion_pearson, null_deviance_for_family, pearson_chi2, resid_deviance,
    resid_pearson, resid_response, resid_working,
};
use rustystats_core::families::Family;
use rustystats_core::inference::{
    confidence_interval_z, pvalue_z, robust_covariance, robust_standard_errors, HCType,
};
use rustystats_core::regularization::Penalty;
use rustystats_core::solvers::passes::SolverStep;

use crate::families_py::family_from_name_with_tweedie_support;

/// Parse a fit's `robust=` list: HC type names, each once, in order. Robust
/// covariances need the bread, so they are refused without covariance.
pub(crate) fn parse_robust_types(
    robust: Option<Vec<String>>,
    skip_covariance: bool,
) -> PyResult<Vec<HCType>> {
    let mut types: Vec<HCType> = Vec::new();
    for name in robust.unwrap_or_default() {
        let hc_type = HCType::from_str(&name).ok_or_else(|| {
            PyValueError::new_err(format!(
                "Unknown robust covariance type {name:?}; use HC0, HC1, HC2 or HC3."
            ))
        })?;
        if !types.contains(&hc_type) {
            types.push(hc_type);
        }
    }
    if skip_covariance && !types.is_empty() {
        return Err(PyValueError::new_err(
            "robust= needs the covariance matrix, so it cannot be combined with \
             skipping covariance (compute_covariance=False).",
        ));
    }
    Ok(types)
}

/// The fit-time robust covariances of an in-memory fit: those its final pass
/// computed, or, for a route whose final pass did not, the same sandwich formed
/// from the rows it was given (`inference::robust_covariance`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn fit_time_robust(
    types: &[HCType],
    statistics: Option<&mut rustystats_core::solvers::FinalStatistics>,
    x: ndarray::ArrayView2<'_, f64>,
    y: &Array1<f64>,
    fitted_values: &Array1<f64>,
    irls_weights: &Array1<f64>,
    prior_weights: &Array1<f64>,
    bread: &Array2<f64>,
    family: &dyn Family,
) -> Vec<(HCType, Array2<f64>)> {
    if types.is_empty() {
        return Vec::new();
    }
    if let Some(statistics) = statistics {
        let computed: Vec<HCType> = statistics.robust.iter().map(|(t, _)| *t).collect();
        if computed == types {
            return std::mem::take(&mut statistics.robust);
        }
    }
    let pearson_resid = resid_pearson(y, fitted_values, family);
    types
        .iter()
        .map(|&hc_type| {
            (
                hc_type,
                rustystats_core::inference::robust_covariance_view(
                    x,
                    &pearson_resid,
                    irls_weights,
                    prior_weights,
                    bread,
                    hc_type,
                ),
            )
        })
        .collect()
}

/// Results from fitting a GLM.
///
/// Contains coefficients, fitted values, deviance, and diagnostic information.
/// Use this to make predictions and assess model fit.
#[pyclass(name = "GLMResults", skip_from_py_object)]
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
    /// Whether step-halving was triggered during fitting (RS-ACT-007).
    pub(crate) step_halving_used: bool,
    /// Terminal solver status: "converged", "max_iterations",
    /// "step_halving_no_improvement", or "stalled_nonstationary" (RS-ACT-007).
    pub(crate) solver_status: String,
    /// Exit stationarity (KKT) check result. `Some(...)` for smooth-path fits;
    /// `None` for paths that do not run the check.
    pub(crate) stationary: Option<bool>,
    /// Largest standardized score over unpenalized coordinates at exit.
    pub(crate) max_std_score: Option<f64>,
    /// User-facing warnings collected during fitting (e.g. non-convergence,
    /// fallback initialisation, final-extraction instability). Surfaced so the
    /// solver can communicate diagnostic context beyond the terminal status
    /// string (RS-ACT-007).
    pub(crate) warnings: Vec<String>,
    /// Statistics computed from the rows at fit time (`specs/out-of-core-fit.md`
    /// §8). When present, they replace recomputation from the per-row arrays; they
    /// equal it bit for bit because both use the same row functions and order.
    pub(crate) statistics: Option<rustystats_core::solvers::FinalStatistics>,
    /// Robust covariances computed at fit time (`robust=`), by type.
    pub(crate) robust_covariances: Vec<(HCType, Array2<f64>)>,
    /// The exact Negative Binomial θ the fit used. `family_name` shows it to four
    /// decimals, which must not be what later statistics use (§16, D12).
    pub(crate) exact_theta: Option<f64>,
    /// The exact Tweedie variance power the fit used (likewise).
    pub(crate) exact_var_power: Option<f64>,
    /// An out-of-core result: it holds no per-row arrays, and methods that need
    /// them refuse (`specs/out-of-core-fit.md` §4.5).
    pub(crate) rowless: bool,
    /// The drivers' semantic steps (`specs/out-of-core-fit.md` §4.5).
    pub(crate) solver_trace: Vec<SolverStep>,
}

/// A trace as Python tuples `(driver, kind, evaluations, accepted, status)`.
pub(crate) fn trace_tuples(trace: &[SolverStep]) -> Vec<(String, String, usize, bool, String)> {
    trace
        .iter()
        .map(|step| {
            (
                step.driver.to_string(),
                step.kind.to_string(),
                step.evaluations,
                step.accepted,
                step.status.clone(),
            )
        })
        .collect()
}

// =============================================================================
// Helper Methods (not exposed to Python)
// =============================================================================

impl PyGLMResults {
    /// Refuse a method that needs per-row training values on an out-of-core result.
    fn require_rows(&self, method: &str) -> PyResult<()> {
        if !self.rowless {
            return Ok(());
        }
        Python::attach(|py| {
            let message = format!(
                "{method} needs the per-row training values, which an out-of-core fit does \
                 not keep. Predict on the data instead."
            );
            let error_class = py
                .import("rustystats.exceptions")?
                .getattr("OutOfCoreResultError")?;
            Err(PyErr::from_value(error_class.call1((message,))?))
        })
    }

    /// Get the appropriate Family trait object based on family_name.
    /// Used internally by diagnostics and robust SE methods.
    /// Note: family_name is validated at model creation, so this should never fail.
    fn get_family(&self) -> Box<dyn Family> {
        if let Some(theta) = self.exact_theta {
            return Box::new(
                rustystats_core::families::NegativeBinomialFamily::new(theta)
                    .expect("the fitted theta is valid"),
            );
        }
        if let Some(var_power) = self.exact_var_power {
            return Box::new(
                rustystats_core::families::TweedieFamily::new(var_power)
                    .expect("the fitted variance power is valid"),
            );
        }
        family_from_name_with_tweedie_support(&self.family_name, 1.5, 1.0, true)
            .expect("Invalid family name stored in results - this is a bug")
    }

    fn is_quasi_likelihood(&self) -> bool {
        let base = self
            .family_name
            .to_lowercase()
            .split('(')
            .next()
            .unwrap_or("")
            .trim()
            .to_string();
        matches!(
            base.as_str(),
            "quasipoisson"
                | "quasi_poisson"
                | "quasi-poisson"
                | "quasibinomial"
                | "quasi_binomial"
                | "quasi-binomial"
        )
    }

    /// Get prior weights as Option, returning None if all weights are 1.0.
    /// Many functions accept Option<&Array1<f64>> for weights.
    fn maybe_weights(&self) -> Option<&Array1<f64>> {
        if let Some(statistics) = &self.statistics {
            return (!statistics.all_unit_weights).then_some(&self.prior_weights);
        }
        if self
            .prior_weights
            .iter()
            .all(|&w| (w - 1.0).abs() < rustystats_core::constants::ZERO_TOL)
        {
            None
        } else {
            Some(&self.prior_weights)
        }
    }

    /// Compute robust covariance matrix (internal helper).
    /// Factored out to avoid repeating the same logic in cov_robust, bse_robust, etc.
    /// Returns Err if design_matrix was not stored (lean mode).
    fn compute_robust_cov(&self, hc_type: HCType) -> PyResult<Array2<f64>> {
        if let Some((_, cov)) = self.robust_covariances.iter().find(|(t, _)| *t == hc_type) {
            return Ok(cov.clone());
        }
        self.require_rows(&format!(
            "robust standard errors of type {} (not listed in robust= at fit)",
            hc_type.name()
        ))?;
        let dm = self.design_matrix.as_ref().ok_or_else(|| {
            PyValueError::new_err(
                "Design matrix not stored (lean mode). Robust standard errors require \
                 store_design_matrix=True when fitting, or pass the design matrix via \
                 the Python diagnostics API.",
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
        self.coefficients.clone().into_pyarray(py)
    }

    /// Alias for params (statsmodels compatibility).
    #[getter]
    fn coefficients<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.coefficients.clone().into_pyarray(py)
    }

    /// Get the fitted values μ = g⁻¹(Xβ).
    ///
    /// These are the predicted means on the response scale.
    #[getter]
    fn fittedvalues<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.require_rows("fittedvalues")?;
        Ok(self.fitted_values.clone().into_pyarray(py))
    }

    /// Get the linear predictor η = Xβ.
    #[getter]
    fn linear_predictor<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.require_rows("linear_predictor")?;
        Ok(self.linear_predictor.clone().into_pyarray(py))
    }

    /// Get the original response variable the model was fit on.
    ///
    /// Mirrors `fittedvalues` / `linear_predictor`; exposed so callers can
    /// inspect the training response without recovering it from residuals.
    #[getter]
    fn response<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.require_rows("response")?;
        Ok(self.y.clone().into_pyarray(py))
    }

    /// Largest finite absolute training response — the robust reference scale for
    /// the unbounded-mean prediction ceiling applied by `GLMModel.predict`.
    ///
    /// Anchored to the observed response, not the fitted values: a fit that fails
    /// to converge can emit in-sample fitted means orders of magnitude beyond
    /// anything observed, so a fitted-value scale would be poisoned by the very
    /// pathology the ceiling must bound. The response cannot be inflated that way.
    /// Returns `None` when no finite positive response is available (degenerate fit).
    #[getter]
    fn response_scale(&self) -> Option<f64> {
        if let Some(statistics) = &self.statistics {
            return statistics.response_scale;
        }
        let scale = self
            .y
            .iter()
            .filter(|v| v.is_finite())
            .fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        if scale.is_finite() && scale > 0.0 {
            Some(scale)
        } else {
            None
        }
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

    /// Terminal solver status: "converged", "max_iterations",
    /// "step_halving_no_improvement", or "stalled_nonstationary" (RS-ACT-007).
    #[getter]
    fn solver_status(&self) -> String {
        self.solver_status.clone()
    }

    /// The drivers' semantic steps, as `(driver, kind, evaluations, accepted,
    /// status)` tuples in order (`specs/out-of-core-fit.md` §4.5).
    #[getter]
    fn solver_trace(&self) -> Vec<(String, String, usize, bool, String)> {
        trace_tuples(&self.solver_trace)
    }

    /// Exit stationarity (KKT) check: True when the score of every
    /// unpenalized, unconstrained coordinate vanished at exit. None for fit
    /// paths that do not run the check. `converged=True` implies this passed.
    #[getter]
    fn stationary(&self) -> Option<bool> {
        self.stationary
    }

    /// Largest standardized score |s_j| / sqrt(I_jj) over the coordinates
    /// covered by the exit stationarity check. None when the check did not run.
    #[getter]
    fn max_std_score(&self) -> Option<f64> {
        self.max_std_score
    }

    /// Whether step-halving was triggered during fitting (RS-ACT-007).
    #[getter]
    fn step_halving_used(&self) -> bool {
        self.step_halving_used
    }

    /// Warnings collected during fitting (RS-ACT-007).
    ///
    /// Empty when the fit completed without diagnostic concerns. Common
    /// messages include "IRLS did not converge ...", "Family ... initial μ
    /// values were invalid ...", and final-coefficient extraction fallbacks.
    /// The terminal `solver_status` is the primary signal — `warnings`
    /// carries the additional human-readable context.
    #[getter]
    fn warnings(&self) -> Vec<String> {
        self.warnings.clone()
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
        self.covariance_unscaled.clone().into_pyarray(py)
    }

    /// Get the design matrix X used in fitting.
    ///
    /// Returns None if store_design_matrix=False was used (lean mode).
    /// This is useful for computing score tests for unfitted factors.
    #[getter]
    fn get_design_matrix<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        self.design_matrix
            .as_ref()
            .map(|dm| dm.clone().into_pyarray(py))
    }

    /// Get the IRLS working weights from the final iteration.
    ///
    /// These are the diagonal elements of the weight matrix W in the
    /// weighted least squares problem: (X'WX)β = X'Wz.
    /// Useful for computing score tests for unfitted factors.
    #[getter]
    fn get_irls_weights<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.require_rows("get_irls_weights")?;
        Ok(self.irls_weights.clone().into_pyarray(py))
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
        if let Some(statistics) = &self.statistics {
            return statistics.scale;
        }
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
        se.into_pyarray(py)
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
        t.into_pyarray(py)
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
        pvals.into_pyarray(py)
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
        ci.into_pyarray(py)
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
    fn cov_robust<'py>(
        &self,
        py: Python<'py>,
        cov_type: &str,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let hc_type = HCType::from_str(cov_type).ok_or_else(|| {
            PyValueError::new_err(format!(
                "Unknown cov_type '{}'. Use 'HC0', 'HC1', 'HC2', or 'HC3'.",
                cov_type
            ))
        })?;

        let cov = self.compute_robust_cov(hc_type)?;
        Ok(cov.into_pyarray(py))
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
    fn bse_robust<'py>(
        &self,
        py: Python<'py>,
        cov_type: &str,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let hc_type = HCType::from_str(cov_type).ok_or_else(|| {
            PyValueError::new_err(format!(
                "Unknown cov_type '{}'. Use 'HC0', 'HC1', 'HC2', or 'HC3'.",
                cov_type
            ))
        })?;

        let cov = self.compute_robust_cov(hc_type)?;
        let se = robust_standard_errors(&cov);
        Ok(se.into_pyarray(py))
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
    fn tvalues_robust<'py>(
        &self,
        py: Python<'py>,
        cov_type: &str,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let hc_type = HCType::from_str(cov_type).ok_or_else(|| {
            PyValueError::new_err(format!(
                "Unknown cov_type '{}'. Use 'HC0', 'HC1', 'HC2', or 'HC3'.",
                cov_type
            ))
        })?;

        let cov = self.compute_robust_cov(hc_type)?;
        let se = robust_standard_errors(&cov);
        let t: Array1<f64> = self
            .coefficients
            .iter()
            .zip(se.iter())
            .map(|(&c, &s)| if s > 1e-10 { c / s } else { 0.0 })
            .collect();

        Ok(t.into_pyarray(py))
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
    fn pvalues_robust<'py>(
        &self,
        py: Python<'py>,
        cov_type: &str,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let hc_type = HCType::from_str(cov_type).ok_or_else(|| {
            PyValueError::new_err(format!(
                "Unknown cov_type '{}'. Use 'HC0', 'HC1', 'HC2', or 'HC3'.",
                cov_type
            ))
        })?;

        let cov = self.compute_robust_cov(hc_type)?;
        let se = robust_standard_errors(&cov);
        let pvals: Array1<f64> = self
            .coefficients
            .iter()
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

        Ok(pvals.into_pyarray(py))
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
    fn conf_int_robust<'py>(
        &self,
        py: Python<'py>,
        alpha: f64,
        cov_type: &str,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let hc_type = HCType::from_str(cov_type).ok_or_else(|| {
            PyValueError::new_err(format!(
                "Unknown cov_type '{}'. Use 'HC0', 'HC1', 'HC2', or 'HC3'.",
                cov_type
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

        Ok(ci.into_pyarray(py))
    }

    // =========================================================================
    // Residuals (statsmodels-compatible)
    // =========================================================================

    /// Get response residuals: y - μ
    ///
    /// Simple difference between observed and predicted values.
    /// Not standardized.
    fn resid_response<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.require_rows("resid_response")?;
        let resid = resid_response(&self.y, &self.fitted_values);
        Ok(resid.into_pyarray(py))
    }

    /// Get Pearson residuals: (y - μ) / √V(μ)
    ///
    /// Standardized residuals that account for the variance function.
    /// For a well-specified model, should have approximately constant variance.
    fn resid_pearson<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.require_rows("resid_pearson")?;
        let family = self.get_family();
        let resid = resid_pearson(&self.y, &self.fitted_values, family.as_ref());
        Ok(resid.into_pyarray(py))
    }

    /// Get deviance residuals: sign(y - μ) × √d_i
    ///
    /// Based on the unit deviance contributions. Often more normally
    /// distributed than Pearson residuals for non-Gaussian families.
    /// sum(resid_deviance²) = model deviance
    fn resid_deviance<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.require_rows("resid_deviance")?;
        let family = self.get_family();
        let resid = resid_deviance(&self.y, &self.fitted_values, family.as_ref());
        Ok(resid.into_pyarray(py))
    }

    /// Get working residuals: (y - μ) × g'(μ)
    ///
    /// Used internally by IRLS. On the scale of the linear predictor.
    /// Useful for understanding the fitting process.
    fn resid_working<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.require_rows("resid_working")?;
        let family = self.get_family();
        let link = family.default_link();
        let resid = resid_working(&self.y, &self.fitted_values, link.as_ref());
        Ok(resid.into_pyarray(py))
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
        if let Some(statistics) = &self.statistics {
            return statistics.pearson_chi2;
        }
        let family = self.get_family();
        pearson_chi2(
            &self.y,
            &self.fitted_values,
            family.as_ref(),
            self.maybe_weights(),
        )
    }

    /// Get dispersion estimated from Pearson residuals.
    ///
    /// φ_pearson = X² / (n - p)
    fn scale_pearson(&self) -> f64 {
        if let Some(statistics) = &self.statistics {
            let df_resid = self.df_resid();
            return if df_resid == 0 {
                1.0
            } else {
                statistics.pearson_chi2 / (df_resid as f64)
            };
        }
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
        if let Some(statistics) = &self.statistics {
            return statistics.log_likelihood;
        }
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
    fn aic(&self) -> Option<f64> {
        if self.is_quasi_likelihood() {
            None
        } else {
            Some(aic(self.llf(), self.n_params))
        }
    }

    /// Get the Bayesian Information Criterion.
    ///
    /// BIC = -2ℓ + p×log(n)
    ///
    /// Lower is better. Penalizes complexity more than AIC for large n.
    fn bic(&self) -> Option<f64> {
        if self.is_quasi_likelihood() {
            None
        } else {
            Some(bic(self.llf(), self.n_params, self.n_obs))
        }
    }

    /// Get the null deviance (deviance of intercept-only model).
    ///
    /// Measures total variation in y before accounting for predictors.
    /// Compare to residual deviance to assess explanatory power.
    /// Accounts for offset if present (e.g., exposure in count models).
    fn null_deviance(&self) -> f64 {
        if let Some(statistics) = &self.statistics {
            return statistics.null_deviance;
        }
        let family = self.get_family();
        null_deviance_for_family(
            &self.y,
            family.as_ref(),
            family.default_link().as_ref(),
            self.maybe_weights(),
            self.offset.as_ref(),
        )
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
            Penalty::Smooth(_) => None, // Smooth penalties don't have L1 ratio
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
        self.coefficients
            .iter()
            .skip(1)
            .filter(|&&c| c.abs() > 1e-10)
            .count()
    }

    /// Get indices of non-zero coefficients (selected variables).
    ///
    /// For Lasso/Elastic Net, this shows which variables were retained.
    fn selected_features(&self) -> Vec<usize> {
        self.coefficients
            .iter()
            .enumerate()
            .skip(1) // Skip intercept
            .filter(|(_, &c)| c.abs() > 1e-10)
            .map(|(i, _)| i)
            .collect()
    }
}
