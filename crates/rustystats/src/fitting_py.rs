// =============================================================================
// GLM Fitting Functions (Python Bindings)
// =============================================================================
//
// All GLM fitting entry points exposed to Python:
// - fit_glm_py: Standard GLM with optional regularization
// - fit_negbinomial_py: NegBin with automatic theta estimation
// - fit_smooth_glm_unified_py: Unified smooth GLM (single entry point)
// - fit_cv_path_py: Cross-validated regularization path
// =============================================================================

use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use rustystats_core::families::NegativeBinomialFamily;
use rustystats_core::regularization::{RegularizationConfig, Standardization};
use rustystats_core::solvers::{
    fit_glm_unified, fit_smooth_glm_full_matrix, FitConfig, IRLSConfig, IRLSResult, Monotonicity,
    SmoothGLMConfig, SmoothTermSpec,
};

use crate::families_py::{
    default_link_name, family_from_name_with_tweedie_support, link_from_name,
    resolve_negbinomial_theta, resolve_tweedie_var_power, validate_tweedie_fit_response,
};
use crate::out_of_core_py::parse_row_kernel;
use crate::results_py::{fit_time_robust, parse_robust_types, PyGLMResults};
use rustystats_core::solvers::passes::RowKernel;

// =============================================================================
// Helpers: reduce boilerplate for smooth result conversion
// =============================================================================

/// Convert SmoothGLMResult → (PyGLMResults, smooth_metadata_dict) Python tuple.
fn smooth_result_to_py<'py>(
    py: Python<'py>,
    mut result: rustystats_core::solvers::SmoothGLMResult,
    design_matrix: Option<Array2<f64>>,
    family_name: String,
    exact_theta: Option<f64>,
    exact_var_power: Option<f64>,
    robust_covariances: Vec<(rustystats_core::inference::HCType, ndarray::Array2<f64>)>,
) -> PyResult<Py<PyAny>> {
    let n_obs = result.y.len();
    let n_params = result.coefficients.len();

    result.family_name = family_name;
    let glm_result = PyGLMResults {
        link_name: None,
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
        design_matrix,
        irls_weights: result.irls_weights,
        offset: result.offset,
        step_halving_used: result.step_halving_used,
        solver_status: result.solver_status,
        stationary: Some(result.stationary),
        max_std_score: Some(result.max_std_score),
        warnings: result.warnings,
        statistics: result.statistics,
        robust_covariances,
        exact_theta,
        exact_var_power,
        rowless: false,
        solver_trace: result.solver_trace,
    };

    let smooth_dict = pyo3::types::PyDict::new(py);
    smooth_dict.set_item("lambdas", result.lambdas)?;
    smooth_dict.set_item("smooth_edfs", result.smooth_edfs)?;
    smooth_dict.set_item("total_edf", result.total_edf)?;
    smooth_dict.set_item("gcv", result.gcv)?;

    let tuple = pyo3::types::PyTuple::new(
        py,
        &[
            Bound::new(py, glm_result)?.into_any(),
            smooth_dict.into_any(),
        ],
    )?;
    Ok(tuple.unbind().into())
}

/// Build a SmoothGLMConfig from common parameters.
/// The smooth term specs of a smooth fit binding: column ranges, penalties and
/// optional monotonicity names (None, "increasing"/"inc", "decreasing"/"dec").
pub(crate) fn smooth_term_specs(
    smooth_col_ranges: &[(usize, usize)],
    smooth_penalties: &[PyReadonlyArray2<'_, f64>],
    smooth_monotonicity: Option<Vec<Option<String>>>,
) -> PyResult<Vec<SmoothTermSpec>> {
    let n_terms = smooth_col_ranges.len();
    if smooth_penalties.len() != n_terms {
        return Err(PyValueError::new_err(format!(
            "smooth_col_ranges has {} entries but smooth_penalties has {}",
            n_terms,
            smooth_penalties.len()
        )));
    }
    let mono_vec = smooth_monotonicity.unwrap_or_else(|| vec![None; n_terms]);
    if mono_vec.len() != n_terms {
        return Err(PyValueError::new_err(format!(
            "smooth_monotonicity has {} entries but expected {}",
            mono_vec.len(),
            n_terms
        )));
    }
    smooth_col_ranges
        .iter()
        .zip(smooth_penalties.iter())
        .zip(mono_vec.iter())
        .map(|((&(start, end), pen), mono)| {
            // Unrecognized monotonicity strings must error loudly rather than
            // silently disabling the constraint. `None` (Python) leaves the term
            // unconstrained by design.
            let monotonicity = match mono.as_deref() {
                None => Monotonicity::None,
                Some("increasing") | Some("inc") => Monotonicity::Increasing,
                Some("decreasing") | Some("dec") => Monotonicity::Decreasing,
                Some(other) => {
                    return Err(PyValueError::new_err(format!(
                        "Unrecognized smooth_monotonicity value {other:?}; \
                         accepted values are None, \"increasing\" (\"inc\"), or \
                         \"decreasing\" (\"dec\")."
                    )));
                }
            };
            Ok(SmoothTermSpec {
                col_start: start,
                col_end: end,
                penalty: pen.as_array().to_owned(),
                monotonicity,
                initial_lambda: 1.0,
            })
        })
        .collect()
}

pub(crate) fn build_smooth_config(
    max_iter: usize,
    tol: f64,
    lambda_min: f64,
    lambda_max: f64,
) -> SmoothGLMConfig {
    SmoothGLMConfig {
        irls_config: IRLSConfig {
            max_iterations: max_iter,
            tolerance: tol,
            min_weight: 1e-10,
            verbose: false,
            nonneg_indices: Vec::new(),
            nonpos_indices: Vec::new(),
            skip_covariance: false,
            row_kernel: RowKernel::default(),
        },
        n_lambda: 30,
        lambda_min,
        lambda_max,
        lambda_tol: 1e-4,
        max_lambda_iter: 10,
        lambda_method: "gcv".to_string(),
        robust: Vec::new(),
    }
}

pub(crate) fn build_standardization(
    center: Option<PyReadonlyArray1<f64>>,
    scale: Option<PyReadonlyArray1<f64>>,
    n_params: usize,
) -> PyResult<Option<Standardization>> {
    match (center, scale) {
        (None, None) => Ok(None),
        (Some(c), Some(s)) => {
            let center_vec = c.as_array().to_vec();
            let scale_vec = s.as_array().to_vec();
            if center_vec.len() != n_params {
                return Err(PyValueError::new_err(format!(
                    "center length {} does not match X columns {}",
                    center_vec.len(),
                    n_params
                )));
            }
            if scale_vec.len() != n_params {
                return Err(PyValueError::new_err(format!(
                    "scale length {} does not match X columns {}",
                    scale_vec.len(),
                    n_params
                )));
            }
            Standardization::new(center_vec, scale_vec)
                .map(Some)
                .map_err(|e| PyValueError::new_err(format!("Invalid standardization: {}", e)))
        }
        _ => Err(PyValueError::new_err(
            "center and scale must be provided together for standardization",
        )),
    }
}

/// Standardisation moments of `x` (§6.7). With `fold=(k, folds, seed)` only fold
/// k's training rows (DEC-2's hash) take part, reduced in blocks of global rows.
#[pyfunction]
#[pyo3(signature = (x, weights=None, pen_mask=None, fit_intercept=true, fold=None))]
pub fn compute_standardization_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    weights: Option<PyReadonlyArray1<f64>>,
    pen_mask: Option<PyReadonlyArray1<bool>>,
    fit_intercept: bool,
    fold: Option<(usize, usize, u64)>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let x_view = x.as_array();
    let (n, p) = x_view.dim();
    let weights_view = weights.as_ref().map(|w| w.as_array());
    if let Some(w) = weights_view.as_ref() {
        if w.ndim() != 1 || w.len() != n {
            return Err(PyValueError::new_err(format!(
                "weights must be length {} for standardization, got length {}.",
                n,
                w.len()
            )));
        }
    }

    let mut mask = vec![true; p];
    if let Some(pm) = pen_mask.as_ref() {
        let pm_view = pm.as_array();
        if pm_view.ndim() != 1 || pm_view.len() != p {
            return Err(PyValueError::new_err(format!(
                "pen_mask must be length {} for standardization, got length {}.",
                p,
                pm_view.len()
            )));
        }
        for (dst, src) in mask.iter_mut().zip(pm_view.iter()) {
            *dst = *src;
        }
    }
    if fit_intercept && p > 0 {
        mask[0] = false;
    }

    let active: Vec<usize> = mask
        .iter()
        .enumerate()
        .filter_map(|(idx, active)| active.then_some(idx))
        .collect();

    let training = fold.map(
        |(fold, folds, seed)| rustystats_core::solvers::passes::FoldFilter {
            fold,
            folds,
            seed,
            validation: false,
        },
    );
    let moments = py.detach(|| {
        rustystats_core::regularization::moments::standardization_moments_where(
            x_view,
            weights_view,
            &active,
            fit_intercept,
            |row| training.is_none_or(|filter| filter.includes(row)),
        )
    });
    let moments = moments.map_err(|error| PyValueError::new_err(error.to_string()))?;
    let (center, scale, ridge_xtx_diag) = (moments.center, moments.scale, moments.ridge_xtx_diag);

    Ok((
        center.into_pyarray(py),
        scale.into_pyarray(py),
        ridge_xtx_diag.into_pyarray(py),
    ))
}

// =============================================================================
// fit_glm_py — Standard GLM
// =============================================================================

#[pyfunction]
#[pyo3(signature = (y, x, family, link=None, var_power=1.5, theta=1.0, offset=None, weights=None, alpha=0.0, l1_ratio=0.0, max_iter=25, tol=1e-8, nonneg_indices=None, nonpos_indices=None, store_design_matrix=false, allow_extended_tweedie=false, fit_intercept=true, center=None, scale=None, skip_covariance=false, robust=None, max_row_nonzeros=None, row_kernel=None))]
#[allow(clippy::too_many_arguments)]
pub fn fit_glm_py(
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
    nonneg_indices: Option<Vec<usize>>,
    nonpos_indices: Option<Vec<usize>>,
    store_design_matrix: bool,
    allow_extended_tweedie: bool,
    fit_intercept: bool,
    center: Option<PyReadonlyArray1<f64>>,
    scale: Option<PyReadonlyArray1<f64>>,
    skip_covariance: bool,
    robust: Option<Vec<String>>,
    max_row_nonzeros: Option<usize>,
    row_kernel: Option<&str>,
) -> PyResult<PyGLMResults> {
    let robust_types = parse_robust_types(robust, skip_covariance)?;
    let y_array: Array1<f64> = y.as_array().to_owned();
    let x_view = x.as_array(); // Zero-copy view of numpy array
    let n_obs = y_array.len();
    let n_params = x_view.ncols();
    let offset_array: Option<Array1<f64>> = offset.map(|o| o.as_array().to_owned());
    let weights_array: Option<Array1<f64>> = weights.map(|w| w.as_array().to_owned());
    let standardization = build_standardization(center, scale, n_params)?;

    let reg_config = if alpha > 0.0 {
        if l1_ratio >= 1.0 {
            RegularizationConfig::lasso(alpha)
        } else if l1_ratio <= 0.0 {
            RegularizationConfig::ridge(alpha)
        } else {
            RegularizationConfig::elastic_net(alpha, l1_ratio)
        }
    } else {
        RegularizationConfig::none()
    }
    .with_intercept(fit_intercept);

    let config = FitConfig {
        max_iterations: max_iter,
        tolerance: tol,
        min_weight: 1e-10,
        verbose: false,
        nonneg_indices: nonneg_indices.unwrap_or_default(),
        nonpos_indices: nonpos_indices.unwrap_or_default(),
        regularization: reg_config,
        skip_covariance,
        standardization,
        robust: robust_types.clone(),
        row_kernel: parse_row_kernel(max_row_nonzeros, row_kernel)?,
    };

    // Gamma validation
    if family.to_lowercase() == "gamma" {
        let n_invalid = y_array.iter().filter(|&&v| v <= 0.0).count();
        if n_invalid > 0 {
            return Err(PyValueError::new_err(format!(
                "Gamma family requires strictly positive response values (y > 0). \
                 Found {} values <= 0 out of {} observations.",
                n_invalid,
                y_array.len()
            )));
        }
    }

    validate_tweedie_fit_response(family, &y_array, var_power, allow_extended_tweedie)?;

    let fam =
        family_from_name_with_tweedie_support(family, var_power, theta, allow_extended_tweedie)?;
    let lnk = link_from_name(link.unwrap_or(default_link_name(family)))?;

    let mut result: IRLSResult = fit_glm_unified(
        &y_array,
        x_view,
        fam.as_ref(),
        lnk.as_ref(),
        &config,
        offset_array.as_ref(),
        weights_array.as_ref(),
        None,
    )
    .map_err(|e| PyValueError::new_err(format!("GLM fitting failed: {}", e)))?;
    let robust_covariances = fit_time_robust(
        &robust_types,
        result.statistics.as_mut(),
        x_view,
        &result.y,
        &result.fitted_values,
        &result.irls_weights,
        &result.prior_weights,
        &result.covariance_unscaled,
        fam.as_ref(),
    );

    let exact_theta = resolve_negbinomial_theta(family, theta)?;
    let exact_var_power = if exact_theta.is_none() {
        resolve_tweedie_var_power(family, var_power)?
    } else {
        None
    };
    let family_name = if let Some(resolved_theta) = exact_theta {
        format!("NegativeBinomial(theta={:.4})", resolved_theta)
    } else if let Some(resolved_var_power) = exact_var_power {
        format!("Tweedie(p={:.4})", resolved_var_power)
    } else {
        result.family_name
    };

    Ok(PyGLMResults {
        link_name: Some(link.unwrap_or(default_link_name(family)).to_string()),
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
        family_name,
        prior_weights: result.prior_weights,
        penalty: result.penalty,
        design_matrix: if store_design_matrix {
            Some(x_view.to_owned())
        } else {
            None
        },
        irls_weights: result.irls_weights,
        offset: offset_array,
        step_halving_used: result.step_halving_used,
        solver_status: result.solver_status,
        stationary: None,
        max_std_score: None,
        warnings: result.warnings,
        statistics: result.statistics,
        robust_covariances,
        exact_theta,
        exact_var_power,
        rowless: false,
        solver_trace: result.solver_trace,
    })
}

// =============================================================================
// fit_negbinomial_py — NegBin with automatic theta
// =============================================================================

#[pyfunction]
#[pyo3(signature = (y, x, link=None, init_theta=None, theta_tol=1e-5, max_theta_iter=10, offset=None, weights=None, max_iter=25, tol=1e-8, alpha=0.0, l1_ratio=0.0, nonneg_indices=None, nonpos_indices=None, store_design_matrix=false, center=None, scale=None, robust=None, max_row_nonzeros=None, row_kernel=None))]
#[allow(clippy::too_many_arguments)]
pub fn fit_negbinomial_py<'py>(
    py: Python<'py>,
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
    alpha: f64,
    l1_ratio: f64,
    nonneg_indices: Option<Vec<usize>>,
    nonpos_indices: Option<Vec<usize>>,
    store_design_matrix: bool,
    center: Option<PyReadonlyArray1<f64>>,
    scale: Option<PyReadonlyArray1<f64>>,
    robust: Option<Vec<String>>,
    max_row_nonzeros: Option<usize>,
    row_kernel: Option<&str>,
) -> PyResult<Py<PyAny>> {
    let robust_types = parse_robust_types(robust, false)?;
    let y_array: Array1<f64> = y.as_array().to_owned();
    let x_view = x.as_array(); // Zero-copy view
    let n_obs = y_array.len();
    let n_params = x_view.ncols();
    let offset_array: Option<Array1<f64>> = offset.map(|o| o.as_array().to_owned());
    let weights_array: Option<Array1<f64>> = weights.map(|w| w.as_array().to_owned());
    // Standardization for penalized fits, mirroring `fit_glm_py`. Like this
    // binding's existing `alpha`/`l1_ratio`, it serves direct callers running a
    // penalized fit; the Python theta-estimation path always fits unpenalized
    // (`alpha=0`, where standardization is a no-op), so it passes `None`.
    let standardization = build_standardization(center, scale, n_params)?;

    let reg_config = if alpha > 0.0 {
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

    let config = FitConfig {
        max_iterations: max_iter,
        tolerance: tol,
        min_weight: 1e-10,
        verbose: false,
        nonneg_indices: nonneg_indices.unwrap_or_default(),
        nonpos_indices: nonpos_indices.unwrap_or_default(),
        regularization: reg_config,
        skip_covariance: false,
        standardization,
        robust: Vec::new(),
        row_kernel: parse_row_kernel(max_row_nonzeros, row_kernel)?,
    };

    let link_name = link.unwrap_or("log");
    let link_fn = link_from_name(link_name)?;
    if link_name != "log" && link_name != "identity" {
        return Err(PyValueError::new_err(format!(
            "Unknown link '{}' for NegativeBinomial. Use 'log' or 'identity'.",
            link_name
        )));
    }

    if let Some(t) = init_theta.filter(|&t| t <= 0.0) {
        return Err(PyValueError::new_err(format!(
            "init_theta must be > 0, got {t}"
        )));
    }
    // Only the final fit computes the robust covariances.
    let final_config = FitConfig {
        robust: robust_types.clone(),
        ..config.clone()
    };
    let (mut result, estimated) = rustystats_core::solvers::nb_theta::fit_negbinomial_in_memory(
        &y_array,
        x_view,
        link_fn.as_ref(),
        &config,
        &final_config,
        offset_array.as_ref(),
        weights_array.as_ref(),
        init_theta,
        theta_tol,
        max_theta_iter,
    )
    .map_err(|e| PyValueError::new_err(format!("GLM fitting failed: {}", e)))?;
    let theta = estimated.theta;
    let init_theta_used = estimated.init_theta;
    let theta_iterations = estimated.iterations;
    let theta_converged = estimated.converged;
    let final_family = NegativeBinomialFamily::new(theta)
        .map_err(|e| PyValueError::new_err(format!("Invalid NB theta: {}", e)))?;
    let robust_covariances = fit_time_robust(
        &robust_types,
        result.statistics.as_mut(),
        x_view,
        &result.y,
        &result.fitted_values,
        &result.irls_weights,
        &result.prior_weights,
        &result.covariance_unscaled,
        &final_family,
    );

    let glm_result = PyGLMResults {
        link_name: None,
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
        exact_theta: Some(theta),
        exact_var_power: None,
        rowless: false,
        solver_trace: result.solver_trace,
        prior_weights: weights_array.unwrap_or_else(|| Array1::ones(n_obs)),
        penalty: result.penalty,
        design_matrix: if store_design_matrix {
            Some(x_view.to_owned())
        } else {
            None
        },
        irls_weights: result.irls_weights,
        offset: offset_array,
        step_halving_used: result.step_halving_used,
        solver_status: result.solver_status,
        stationary: None,
        max_std_score: None,
        warnings: result.warnings,
        statistics: result.statistics,
        robust_covariances,
    };

    // Honest theta-estimation metadata (RS-ACT-010): the profile loop's init,
    // iteration count, convergence flag, tolerance, and final theta.
    let meta = pyo3::types::PyDict::new(py);
    meta.set_item("estimated", true)?;
    meta.set_item("theta", theta)?;
    meta.set_item("init_theta", init_theta_used)?;
    meta.set_item("theta_iterations", theta_iterations)?;
    meta.set_item("theta_converged", theta_converged)?;
    meta.set_item("theta_tol", theta_tol)?;
    meta.set_item("max_theta_iter", max_theta_iter)?;
    meta.set_item("glm_tol", tol)?;
    // Schema parity with the fixed-theta path (RS-ACT-010): estimation never
    // falls back, but the key is always present so consumers can rely on it.
    meta.set_item("fallback_reason", py.None())?;

    let tuple = pyo3::types::PyTuple::new(
        py,
        &[Bound::new(py, glm_result)?.into_any(), meta.into_any()],
    )?;
    Ok(tuple.unbind().into())
}

/// Unified smooth GLM fitting: takes full design matrix + smooth specs.
///
/// This replaces the 3 separate entry points (fast, multi, monotonic) with a single
/// function. The full design matrix is passed as-is — no column splitting/reordering
/// needed on the Python side.
///
/// # Arguments
/// * `smooth_col_ranges` - List of (start, end) column ranges for smooth terms
/// * `smooth_penalties` - List of penalty matrices (one per smooth term)
/// * `smooth_monotonicity` - List of monotonicity constraints: None, "increasing", "decreasing"
#[pyfunction]
#[pyo3(signature = (y, x_full, smooth_col_ranges, smooth_penalties, family, link=None, offset=None, weights=None, max_iter=25, tol=1e-8, lambda_min=0.001, lambda_max=1000.0, smooth_monotonicity=None, store_design_matrix=false, nonneg_indices=None, nonpos_indices=None, var_power=1.5, theta=1.0, allow_extended_tweedie=false, robust=None, max_row_nonzeros=None, row_kernel=None))]
#[allow(clippy::too_many_arguments)]
pub fn fit_smooth_glm_unified_py<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<f64>,
    x_full: PyReadonlyArray2<f64>,
    smooth_col_ranges: Vec<(usize, usize)>,
    smooth_penalties: Vec<PyReadonlyArray2<f64>>,
    family: &str,
    link: Option<&str>,
    offset: Option<PyReadonlyArray1<f64>>,
    weights: Option<PyReadonlyArray1<f64>>,
    max_iter: usize,
    tol: f64,
    lambda_min: f64,
    lambda_max: f64,
    smooth_monotonicity: Option<Vec<Option<String>>>,
    store_design_matrix: bool,
    nonneg_indices: Option<Vec<usize>>,
    nonpos_indices: Option<Vec<usize>>,
    var_power: f64,
    theta: f64,
    allow_extended_tweedie: bool,
    robust: Option<Vec<String>>,
    max_row_nonzeros: Option<usize>,
    row_kernel: Option<&str>,
) -> PyResult<Py<PyAny>> {
    let robust_types = parse_robust_types(robust, false)?;
    let y_arr = y.as_array().to_owned();
    let x_view = x_full.as_array(); // Zero-copy view
    let offset_arr = offset.map(|o| o.as_array().to_owned());
    let weights_arr = weights.map(|w| w.as_array().to_owned());

    validate_tweedie_fit_response(family, &y_arr, var_power, allow_extended_tweedie)?;
    let fam =
        family_from_name_with_tweedie_support(family, var_power, theta, allow_extended_tweedie)?;
    let lnk = match link {
        Some(l) => link_from_name(l)?,
        None => link_from_name(default_link_name(family))?,
    };

    let n_terms = smooth_col_ranges.len();
    if smooth_penalties.len() != n_terms {
        return Err(PyValueError::new_err(format!(
            "smooth_col_ranges has {} entries but smooth_penalties has {}",
            n_terms,
            smooth_penalties.len()
        )));
    }

    let specs = smooth_term_specs(&smooth_col_ranges, &smooth_penalties, smooth_monotonicity)?;

    let mut config = build_smooth_config(max_iter, tol, lambda_min, lambda_max);
    config.robust = robust_types.clone();
    config.irls_config.row_kernel = parse_row_kernel(max_row_nonzeros, row_kernel)?;

    let nn = nonneg_indices.unwrap_or_default();
    let np = nonpos_indices.unwrap_or_default();

    let mut result = fit_smooth_glm_full_matrix(
        &y_arr,
        x_view,
        &specs,
        fam.as_ref(),
        lnk.as_ref(),
        &config,
        offset_arr.as_ref(),
        weights_arr.as_ref(),
        if nn.is_empty() { None } else { Some(&nn) },
        if np.is_empty() { None } else { Some(&np) },
    )
    .map_err(|e| PyValueError::new_err(format!("Smooth GLM fitting failed: {}", e)))?;

    let exact_theta = resolve_negbinomial_theta(family, theta)?;
    let exact_var_power = if exact_theta.is_none() {
        resolve_tweedie_var_power(family, var_power)?
    } else {
        None
    };
    let family_name = if let Some(resolved_theta) = exact_theta {
        format!("NegativeBinomial(theta={:.4})", resolved_theta)
    } else if let Some(resolved_var_power) = exact_var_power {
        format!("Tweedie(p={:.4})", resolved_var_power)
    } else {
        fam.name().to_string()
    };

    // Smooth routes keep combined prior × IRLS weights as `irls_weights`, so prior
    // weights enter their robust covariance twice, as with a stored design (D9).
    // The pass driver forms them in its final pass; the monotone route from X.
    let robust_covariances = fit_time_robust(
        &robust_types,
        result.statistics.as_mut(),
        x_view,
        &result.y,
        &result.fitted_values,
        &result.irls_weights,
        &result.prior_weights,
        &result.covariance_unscaled,
        fam.as_ref(),
    );
    smooth_result_to_py(
        py,
        result,
        store_design_matrix.then(|| x_view.to_owned()),
        family_name,
        exact_theta,
        exact_var_power,
        robust_covariances,
    )
}
