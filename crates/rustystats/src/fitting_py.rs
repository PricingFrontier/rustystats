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

use ndarray::{Array1, Array2, ArrayView2};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::iter::IntoParallelIterator;
use rayon::prelude::*;
use std::time::Instant;

use rustystats_core::constants::MU_MIN_POSITIVE;
use rustystats_core::diagnostics::{estimate_theta_moments, estimate_theta_profile};
use rustystats_core::families::{Family, NegativeBinomialFamily, PoissonFamily};
use rustystats_core::links::Link;
use rustystats_core::regularization::{RegularizationConfig, Standardization};
use rustystats_core::solvers::{
    build_sparse_row_cache_if_beneficial, fit_glm_unified, fit_glm_unified_with_sparse_cache,
    fit_smooth_glm_full_matrix, FitConfig, IRLSConfig, IRLSResult, Monotonicity, SmoothGLMConfig,
    SmoothTermSpec,
};

use crate::families_py::{
    default_link_name, family_from_name_with_tweedie_support, link_from_name,
    resolve_negbinomial_theta, resolve_tweedie_var_power, validate_tweedie_fit_response,
};
use crate::results_py::PyGLMResults;

const MIN_WEIGHTED_STD: f64 = 1e-12;

// =============================================================================
// Helpers: reduce boilerplate for smooth result conversion
// =============================================================================

/// Convert SmoothGLMResult → (PyGLMResults, smooth_metadata_dict) Python tuple.
fn smooth_result_to_py<'py>(
    py: Python<'py>,
    mut result: rustystats_core::solvers::SmoothGLMResult,
    store_design_matrix: bool,
    family_name: String,
) -> PyResult<Py<PyAny>> {
    let n_obs = result.y.len();
    let n_params = result.coefficients.len();

    result.family_name = family_name;
    let glm_result = PyGLMResults {
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
        design_matrix: if store_design_matrix {
            Some(result.design_matrix)
        } else {
            None
        },
        irls_weights: result.irls_weights,
        offset: result.offset,
        step_halving_used: result.step_halving_used,
        solver_status: result.solver_status,
        warnings: result.warnings,
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
fn build_smooth_config(
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
        },
        n_lambda: 30,
        lambda_min,
        lambda_max,
        lambda_tol: 1e-4,
        max_lambda_iter: 10,
        lambda_method: "gcv".to_string(),
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

#[pyfunction]
#[pyo3(signature = (x, weights=None, pen_mask=None, fit_intercept=true))]
pub fn compute_standardization_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    weights: Option<PyReadonlyArray1<f64>>,
    pen_mask: Option<PyReadonlyArray1<bool>>,
    fit_intercept: bool,
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

    let (center, scale, ridge_xtx_diag) = py.detach(|| {
        let mut center = vec![0.0; p];
        let mut scale = vec![1.0; p];
        let mut ridge_xtx_diag = vec![0.0; p];
        if p == 0 || active.is_empty() {
            return Ok((center, scale, ridge_xtx_diag));
        }

        let weight_sum = match weights_view.as_ref() {
            Some(w) => w.iter().copied().sum::<f64>(),
            None => n as f64,
        };
        if !weight_sum.is_finite() || weight_sum <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "standardization requires positive finite sum(weights), got {}.",
                weight_sum
            )));
        }

        let active_len = active.len();
        let x_slice = if x_view.is_standard_layout() {
            x_view.as_slice()
        } else {
            None
        };
        let weights_slice = weights_view.as_ref().and_then(|w| w.as_slice());
        let target_chunks = rayon::current_num_threads().max(1);
        let chunk_size = n.div_ceil(target_chunks).max(1);
        let num_chunks = n.div_ceil(chunk_size);

        let sums = (0..num_chunks)
            .into_par_iter()
            .map(|chunk_idx| {
                let start = chunk_idx * chunk_size;
                let end = (start + chunk_size).min(n);
                let mut local = vec![0.0; active_len];
                match (x_slice, weights_slice) {
                    (Some(xs), Some(ws)) => {
                        for row in start..end {
                            let w = unsafe { *ws.get_unchecked(row) };
                            let row_start = row * p;
                            for (slot, &col) in active.iter().enumerate() {
                                local[slot] += w * unsafe { *xs.get_unchecked(row_start + col) };
                            }
                        }
                    }
                    (Some(xs), None) => {
                        for row in start..end {
                            let row_start = row * p;
                            for (slot, &col) in active.iter().enumerate() {
                                local[slot] += unsafe { *xs.get_unchecked(row_start + col) };
                            }
                        }
                    }
                    _ => {
                        for row in start..end {
                            let w = weights_view.as_ref().map_or(1.0, |wv| wv[row]);
                            for (slot, &col) in active.iter().enumerate() {
                                local[slot] += w * x_view[[row, col]];
                            }
                        }
                    }
                }
                local
            })
            .reduce(
                || vec![0.0; active_len],
                |mut a, b| {
                    for i in 0..active_len {
                        a[i] += b[i];
                    }
                    a
                },
            );

        let means: Vec<f64> = sums.iter().map(|sum| sum / weight_sum).collect();

        let variances = (0..num_chunks)
            .into_par_iter()
            .map(|chunk_idx| {
                let start = chunk_idx * chunk_size;
                let end = (start + chunk_size).min(n);
                let mut local = vec![0.0; active_len];
                match (x_slice, weights_slice) {
                    (Some(xs), Some(ws)) => {
                        for row in start..end {
                            let w = unsafe { *ws.get_unchecked(row) };
                            let row_start = row * p;
                            for (slot, &col) in active.iter().enumerate() {
                                let delta =
                                    unsafe { *xs.get_unchecked(row_start + col) } - means[slot];
                                local[slot] += w * delta * delta;
                            }
                        }
                    }
                    (Some(xs), None) => {
                        for row in start..end {
                            let row_start = row * p;
                            for (slot, &col) in active.iter().enumerate() {
                                let delta =
                                    unsafe { *xs.get_unchecked(row_start + col) } - means[slot];
                                local[slot] += delta * delta;
                            }
                        }
                    }
                    _ => {
                        for row in start..end {
                            let w = weights_view.as_ref().map_or(1.0, |wv| wv[row]);
                            for (slot, &col) in active.iter().enumerate() {
                                let delta = x_view[[row, col]] - means[slot];
                                local[slot] += w * delta * delta;
                            }
                        }
                    }
                }
                local
            })
            .reduce(
                || vec![0.0; active_len],
                |mut a, b| {
                    for i in 0..active_len {
                        a[i] += b[i];
                    }
                    a
                },
            );

        for (slot, &col) in active.iter().enumerate() {
            let mean = means[slot];
            let var = (variances[slot] / weight_sum).max(0.0);
            let sd = var.sqrt();
            let raw_diag = weight_sum * (var + mean * mean);
            let valid = mean.is_finite() && sd.is_finite() && sd > MIN_WEIGHTED_STD;

            ridge_xtx_diag[col] = if valid {
                if fit_intercept {
                    weight_sum
                } else {
                    raw_diag / (sd * sd)
                }
            } else {
                raw_diag
            };

            if valid {
                if fit_intercept {
                    center[col] = mean;
                }
                scale[col] = sd;
            }
        }

        Ok((center, scale, ridge_xtx_diag))
    })?;

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
#[pyo3(signature = (y, x, family, link=None, var_power=1.5, theta=1.0, offset=None, weights=None, alpha=0.0, l1_ratio=0.0, max_iter=25, tol=1e-8, nonneg_indices=None, nonpos_indices=None, store_design_matrix=false, allow_extended_tweedie=false, fit_intercept=true, center=None, scale=None, skip_covariance=false))]
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
) -> PyResult<PyGLMResults> {
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

    let result: IRLSResult = fit_glm_unified(
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

    let family_name = if let Some(resolved_theta) = resolve_negbinomial_theta(family, theta)? {
        format!("NegativeBinomial(theta={:.4})", resolved_theta)
    } else if let Some(resolved_var_power) = resolve_tweedie_var_power(family, var_power)? {
        format!("Tweedie(p={:.4})", resolved_var_power)
    } else {
        result.family_name
    };

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
        warnings: result.warnings,
    })
}

// =============================================================================
// fit_negbinomial_py — NegBin with automatic theta
// =============================================================================

#[pyfunction]
#[pyo3(signature = (y, x, link=None, init_theta=None, theta_tol=1e-5, max_theta_iter=10, offset=None, weights=None, max_iter=25, tol=1e-8, alpha=0.0, l1_ratio=0.0, nonneg_indices=None, nonpos_indices=None, store_design_matrix=false, center=None, scale=None))]
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
) -> PyResult<Py<PyAny>> {
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
    };

    let link_name = link.unwrap_or("log");
    let link_fn = link_from_name(link_name)?;
    if link_name != "log" && link_name != "identity" {
        return Err(PyValueError::new_err(format!(
            "Unknown link '{}' for NegativeBinomial. Use 'log' or 'identity'.",
            link_name
        )));
    }

    let mut theta = match init_theta {
        Some(t) if t <= 0.0 => {
            return Err(PyValueError::new_err(format!(
                "init_theta must be > 0, got {}",
                t
            )))
        }
        Some(t) => t,
        None => {
            let poisson = PoissonFamily;
            let init_config = FitConfig {
                regularization: RegularizationConfig::none(),
                ..config.clone()
            };
            let init_result = fit_glm_unified(
                &y_array,
                x_view,
                &poisson,
                link_fn.as_ref(),
                &init_config,
                offset_array.as_ref(),
                weights_array.as_ref(),
                None,
            )
            .map_err(|e| PyValueError::new_err(format!("Initial Poisson fit failed: {}", e)))?;
            estimate_theta_moments(&y_array, &init_result.fitted_values)
        }
    };
    let init_theta_used = theta;

    let mut result: IRLSResult;
    let mut coefficients: Option<Array1<f64>> = None;
    let mut theta_iterations: usize = 0;
    let mut theta_converged = false;

    for _iter in 0..max_theta_iter {
        theta_iterations += 1;
        let family = NegativeBinomialFamily::new(theta)
            .map_err(|e| PyValueError::new_err(format!("Invalid NB theta: {}", e)))?;
        result = fit_glm_unified(
            &y_array,
            x_view,
            &family,
            link_fn.as_ref(),
            &config,
            offset_array.as_ref(),
            weights_array.as_ref(),
            coefficients.as_ref(),
        )
        .map_err(|e| PyValueError::new_err(format!("GLM fitting failed: {}", e)))?;

        coefficients = Some(result.coefficients.clone());
        let new_theta = estimate_theta_profile(
            &y_array,
            &result.fitted_values,
            weights_array.as_ref(),
            0.01,
            1000.0,
            1e-6,
        );
        if (new_theta - theta).abs() < theta_tol {
            theta = new_theta;
            theta_converged = true;
            break;
        }
        theta = new_theta;
    }

    let final_family = NegativeBinomialFamily::new(theta)
        .map_err(|e| PyValueError::new_err(format!("Invalid NB theta: {}", e)))?;
    result = fit_glm_unified(
        &y_array,
        x_view,
        &final_family,
        link_fn.as_ref(),
        &config,
        offset_array.as_ref(),
        weights_array.as_ref(),
        coefficients.as_ref(),
    )
    .map_err(|e| PyValueError::new_err(format!("Final GLM fit failed: {}", e)))?;

    let glm_result = PyGLMResults {
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
        design_matrix: if store_design_matrix {
            Some(x_view.to_owned())
        } else {
            None
        },
        irls_weights: result.irls_weights,
        offset: offset_array,
        step_halving_used: result.step_halving_used,
        solver_status: result.solver_status,
        warnings: result.warnings,
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

// =============================================================================
// fit_cv_path_py — Cross-validated regularization path
// =============================================================================

#[derive(Clone)]
struct CVPathPoint {
    alpha: f64,
    cv_deviance_mean: f64,
    cv_deviance_se: f64,
}

#[derive(Clone, Default)]
struct CVFoldProfile {
    fold: usize,
    n_train: usize,
    n_val: usize,
    split_copy_seconds: f64,
    sparse_cache_seconds: f64,
    standardize_seconds: f64,
    setup_seconds: f64,
    fit_seconds: Vec<f64>,
    validation_dot_seconds: Vec<f64>,
    validation_score_seconds: Vec<f64>,
    iterations: Vec<usize>,
    statuses: Vec<String>,
}

fn validation_deviance_score(unit_deviance: &Array1<f64>, weights: Option<&Array1<f64>>) -> f64 {
    if let Some(w) = weights {
        let denom = w.sum();
        if denom <= 0.0 || !denom.is_finite() {
            return f64::INFINITY;
        }
        unit_deviance
            .iter()
            .zip(w.iter())
            .map(|(dev, weight)| dev * weight)
            .sum::<f64>()
            / denom
    } else {
        unit_deviance.mean().unwrap_or(f64::INFINITY)
    }
}

fn matrix_vector_dot(x: ArrayView2<'_, f64>, coefficients: &Array1<f64>) -> Array1<f64> {
    let n = x.nrows();
    let p = x.ncols();
    assert_eq!(
        coefficients.len(),
        p,
        "coefficient vector length must match X columns"
    );

    let x_slice = match x.as_slice() {
        Some(slice) => slice,
        None => return x.dot(coefficients),
    };
    let coef_slice = match coefficients.as_slice() {
        Some(slice) => slice,
        None => return x.dot(coefficients),
    };
    if n.saturating_mul(p) < 1_000_000 || rayon::current_num_threads() <= 1 {
        return x.dot(coefficients);
    }

    let target_chunks = rayon::current_num_threads().saturating_mul(4).max(1);
    let chunk_size = n.div_ceil(target_chunks).max(1);
    let mut output = vec![0.0; n];
    output
        .par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(chunk_idx, output_chunk)| {
            let row_start_idx = chunk_idx * chunk_size;
            for (local_row, out) in output_chunk.iter_mut().enumerate() {
                let row = row_start_idx + local_row;
                let offset = row * p;
                let mut sum = 0.0;
                for j in 0..p {
                    // SAFETY: row < n and j < p, so offset + j < n*p = x_slice.len();
                    // j < p = coef_slice.len().
                    sum += unsafe { *x_slice.get_unchecked(offset + j) }
                        * unsafe { *coef_slice.get_unchecked(j) };
                }
                *out = sum;
            }
        });

    Array1::from_vec(output)
}

fn poisson_log_validation_deviance_score(
    x: ArrayView2<'_, f64>,
    coefficients: &Array1<f64>,
    y: &Array1<f64>,
    offset: Option<&Array1<f64>>,
    weights: Option<&Array1<f64>>,
    family: &dyn Family,
    link: &dyn Link,
) -> Option<f64> {
    if !family.name().eq_ignore_ascii_case("poisson") || link.name() != "log" {
        return None;
    }
    let n = x.nrows();
    let p = x.ncols();
    if y.len() != n || coefficients.len() != p {
        return None;
    }
    let x_slice = x.as_slice()?;
    let coef_slice = coefficients.as_slice()?;
    let y_slice = y.as_slice()?;
    let offset_slice = match offset {
        Some(o) => Some(o.as_slice()?),
        None => None,
    };
    let weights_slice = match weights {
        Some(w) => Some(w.as_slice()?),
        None => None,
    };

    const EXP_MAX: f64 = 700.0;
    const EXP_MIN: f64 = -700.0;
    if n.saturating_mul(p) < 1_000_000 || rayon::current_num_threads() <= 1 {
        let mut dev_sum = 0.0;
        let mut denom = 0.0;
        for row in 0..n {
            let row_start = row * p;
            let mut eta = 0.0;
            for j in 0..p {
                // SAFETY: row < n and j < p, so row_start + j < n*p = x_slice.len();
                // j < p = coef_slice.len().
                eta += unsafe { *x_slice.get_unchecked(row_start + j) }
                    * unsafe { *coef_slice.get_unchecked(j) };
            }
            if let Some(o) = offset_slice {
                eta += o[row];
            }
            let mu = eta.clamp(EXP_MIN, EXP_MAX).exp().max(MU_MIN_POSITIVE);
            let yi = y_slice[row];
            let unit_dev = if yi == 0.0 {
                2.0 * mu
            } else {
                2.0 * (yi * (yi / mu).ln() - (yi - mu))
            };
            if let Some(w) = weights_slice {
                dev_sum += unit_dev * w[row];
                denom += w[row];
            } else {
                dev_sum += unit_dev;
                denom += 1.0;
            }
        }
        return if denom <= 0.0 || !denom.is_finite() {
            Some(f64::INFINITY)
        } else {
            Some(dev_sum / denom)
        };
    }

    let target_chunks = rayon::current_num_threads().saturating_mul(4).max(1);
    let chunk_size = n.div_ceil(target_chunks).max(1);
    let num_chunks = n.div_ceil(chunk_size);

    let (dev_sum, denom) = (0..num_chunks)
        .into_par_iter()
        .map(|chunk_idx| {
            let start = chunk_idx * chunk_size;
            let end = (start + chunk_size).min(n);
            let mut local_dev = 0.0;
            let mut local_denom = 0.0;
            for row in start..end {
                let row_start = row * p;
                let mut eta = 0.0;
                for j in 0..p {
                    // SAFETY: row < n and j < p, so row_start + j < n*p = x_slice.len();
                    // j < p = coef_slice.len().
                    eta += unsafe { *x_slice.get_unchecked(row_start + j) }
                        * unsafe { *coef_slice.get_unchecked(j) };
                }
                if let Some(o) = offset_slice {
                    eta += o[row];
                }
                let mu = eta.clamp(EXP_MIN, EXP_MAX).exp().max(MU_MIN_POSITIVE);
                let yi = y_slice[row];
                let unit_dev = if yi == 0.0 {
                    2.0 * mu
                } else {
                    2.0 * (yi * (yi / mu).ln() - (yi - mu))
                };
                if let Some(w) = weights_slice {
                    local_dev += unit_dev * w[row];
                    local_denom += w[row];
                } else {
                    local_dev += unit_dev;
                    local_denom += 1.0;
                }
            }
            (local_dev, local_denom)
        })
        .reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));

    if denom <= 0.0 || !denom.is_finite() {
        Some(f64::INFINITY)
    } else {
        Some(dev_sum / denom)
    }
}

fn poisson_log_validation_deviance_score_rows(
    x: ArrayView2<'_, f64>,
    coefficients: &Array1<f64>,
    y: &Array1<f64>,
    offset: Option<&Array1<f64>>,
    weights: Option<&Array1<f64>>,
    rows: &[usize],
) -> Option<f64> {
    let n = x.nrows();
    let p = x.ncols();
    if y.len() != n || coefficients.len() != p {
        return None;
    }
    let x_slice = x.as_slice()?;
    let coef_slice = coefficients.as_slice()?;
    let y_slice = y.as_slice()?;
    let offset_slice = match offset {
        Some(o) => Some(o.as_slice()?),
        None => None,
    };
    let weights_slice = match weights {
        Some(w) => Some(w.as_slice()?),
        None => None,
    };
    if rows.iter().any(|&row| row >= n) {
        return None;
    }

    const EXP_MAX: f64 = 700.0;
    const EXP_MIN: f64 = -700.0;
    if rows.len().saturating_mul(p) < 1_000_000 || rayon::current_num_threads() <= 1 {
        let mut dev_sum = 0.0;
        let mut denom = 0.0;
        for &row in rows {
            let row_start = row * p;
            let mut eta = 0.0;
            for j in 0..p {
                // SAFETY: row < n and j < p, so row_start + j < n*p = x_slice.len();
                // j < p = coef_slice.len().
                eta += unsafe { *x_slice.get_unchecked(row_start + j) }
                    * unsafe { *coef_slice.get_unchecked(j) };
            }
            if let Some(o) = offset_slice {
                eta += o[row];
            }
            let mu = eta.clamp(EXP_MIN, EXP_MAX).exp().max(MU_MIN_POSITIVE);
            let yi = y_slice[row];
            let unit_dev = if yi == 0.0 {
                2.0 * mu
            } else {
                2.0 * (yi * (yi / mu).ln() - (yi - mu))
            };
            if let Some(w) = weights_slice {
                dev_sum += unit_dev * w[row];
                denom += w[row];
            } else {
                dev_sum += unit_dev;
                denom += 1.0;
            }
        }
        return if denom <= 0.0 || !denom.is_finite() {
            Some(f64::INFINITY)
        } else {
            Some(dev_sum / denom)
        };
    }

    let target_chunks = rayon::current_num_threads().saturating_mul(4).max(1);
    let chunk_size = rows.len().div_ceil(target_chunks).max(1);

    let (dev_sum, denom) = rows
        .par_chunks(chunk_size)
        .map(|row_chunk| {
            let mut local_dev = 0.0;
            let mut local_denom = 0.0;
            for &row in row_chunk {
                let row_start = row * p;
                let mut eta = 0.0;
                for j in 0..p {
                    // SAFETY: row < n and j < p, so row_start + j < n*p = x_slice.len();
                    // j < p = coef_slice.len().
                    eta += unsafe { *x_slice.get_unchecked(row_start + j) }
                        * unsafe { *coef_slice.get_unchecked(j) };
                }
                if let Some(o) = offset_slice {
                    eta += o[row];
                }
                let mu = eta.clamp(EXP_MIN, EXP_MAX).exp().max(MU_MIN_POSITIVE);
                let yi = y_slice[row];
                let unit_dev = if yi == 0.0 {
                    2.0 * mu
                } else {
                    2.0 * (yi * (yi / mu).ln() - (yi - mu))
                };
                if let Some(w) = weights_slice {
                    local_dev += unit_dev * w[row];
                    local_denom += w[row];
                } else {
                    local_dev += unit_dev;
                    local_denom += 1.0;
                }
            }
            (local_dev, local_denom)
        })
        .reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));

    if denom <= 0.0 || !denom.is_finite() {
        Some(f64::INFINITY)
    } else {
        Some(dev_sum / denom)
    }
}

#[pyfunction]
#[pyo3(signature = (y_train, x_train, y_val, x_val, family, link=None, var_power=1.5, theta=1.0, offset_train=None, weights_train=None, offset_val=None, weights_val=None, alphas=None, l1_ratio=0.0, max_iter=25, tol=1e-8, nonneg_indices=None, nonpos_indices=None, allow_extended_tweedie=false, fit_intercept=true, center=None, scale=None))]
pub fn fit_fold_path_py<'py>(
    py: Python<'py>,
    y_train: PyReadonlyArray1<f64>,
    x_train: PyReadonlyArray2<f64>,
    y_val: PyReadonlyArray1<f64>,
    x_val: PyReadonlyArray2<f64>,
    family: &str,
    link: Option<&str>,
    var_power: f64,
    theta: f64,
    offset_train: Option<PyReadonlyArray1<f64>>,
    weights_train: Option<PyReadonlyArray1<f64>>,
    offset_val: Option<PyReadonlyArray1<f64>>,
    weights_val: Option<PyReadonlyArray1<f64>>,
    alphas: Option<Vec<f64>>,
    l1_ratio: f64,
    max_iter: usize,
    tol: f64,
    nonneg_indices: Option<Vec<usize>>,
    nonpos_indices: Option<Vec<usize>>,
    allow_extended_tweedie: bool,
    fit_intercept: bool,
    center: Option<PyReadonlyArray1<f64>>,
    scale: Option<PyReadonlyArray1<f64>>,
) -> PyResult<Py<PyAny>> {
    let total_start = Instant::now();
    let input_start = Instant::now();
    let y_train_array: Array1<f64> = y_train.as_array().to_owned();
    let y_val_array: Array1<f64> = y_val.as_array().to_owned();
    let x_train_view = x_train.as_array();
    let x_val_view = x_val.as_array();
    let n_train = y_train_array.len();
    let n_val = y_val_array.len();
    let p = x_train_view.ncols();
    let input_seconds = input_start.elapsed().as_secs_f64();

    let validation_start = Instant::now();
    if x_train_view.nrows() != n_train {
        return Err(PyValueError::new_err(format!(
            "x_train has {} rows but y_train has length {}; they must match",
            x_train_view.nrows(),
            n_train
        )));
    }
    if x_val_view.nrows() != n_val {
        return Err(PyValueError::new_err(format!(
            "x_val has {} rows but y_val has length {}; they must match",
            x_val_view.nrows(),
            n_val
        )));
    }
    if x_val_view.ncols() != p {
        return Err(PyValueError::new_err(format!(
            "x_val has {} columns but x_train has {}; they must match",
            x_val_view.ncols(),
            p
        )));
    }
    if let Some(ref o) = offset_train {
        if o.as_array().len() != n_train {
            return Err(PyValueError::new_err(format!(
                "offset_train has length {} but y_train has length {}; they must match",
                o.as_array().len(),
                n_train
            )));
        }
    }
    if let Some(ref w) = weights_train {
        if w.as_array().len() != n_train {
            return Err(PyValueError::new_err(format!(
                "weights_train has length {} but y_train has length {}; they must match",
                w.as_array().len(),
                n_train
            )));
        }
    }
    if let Some(ref o) = offset_val {
        if o.as_array().len() != n_val {
            return Err(PyValueError::new_err(format!(
                "offset_val has length {} but y_val has length {}; they must match",
                o.as_array().len(),
                n_val
            )));
        }
    }
    if let Some(ref w) = weights_val {
        if w.as_array().len() != n_val {
            return Err(PyValueError::new_err(format!(
                "weights_val has length {} but y_val has length {}; they must match",
                w.as_array().len(),
                n_val
            )));
        }
    }
    let validation_seconds = validation_start.elapsed().as_secs_f64();

    let setup_start = Instant::now();
    let alpha_vec = alphas.ok_or_else(|| {
        PyValueError::new_err("fit_fold_path_py requires an explicit `alphas=` grid")
    })?;
    let offset_train_array: Option<Array1<f64>> = offset_train.map(|o| o.as_array().to_owned());
    let weights_train_array: Option<Array1<f64>> = weights_train.map(|w| w.as_array().to_owned());
    let offset_val_array: Option<Array1<f64>> = offset_val.map(|o| o.as_array().to_owned());
    let weights_val_array: Option<Array1<f64>> = weights_val.map(|w| w.as_array().to_owned());
    let standardization = build_standardization(center, scale, p)?;

    validate_tweedie_fit_response(family, &y_train_array, var_power, allow_extended_tweedie)?;
    let fam =
        family_from_name_with_tweedie_support(family, var_power, theta, allow_extended_tweedie)?;
    let default_link = default_link_name(family);
    let link_fn = link_from_name(link.unwrap_or(default_link))?;
    let fit_scale_only_ridge_on_original = l1_ratio <= 0.0
        && standardization
            .as_ref()
            .is_some_and(|std| std.center.iter().all(|&center| center == 0.0));

    let standardize_start = Instant::now();
    let x_train_work = if let Some(std) = &standardization {
        if fit_scale_only_ridge_on_original {
            x_train_view.to_owned()
        } else {
            std.standardize_matrix(x_train_view)
                .map_err(|e| PyValueError::new_err(format!("Invalid standardization: {}", e)))?
        }
    } else {
        x_train_view.to_owned()
    };
    let standardize_seconds = standardize_start.elapsed().as_secs_f64();
    let sparse_cache_start = Instant::now();
    let sparse_cache = build_sparse_row_cache_if_beneficial(x_train_work.view());
    let sparse_cache_seconds = sparse_cache_start.elapsed().as_secs_f64();

    let nonneg = nonneg_indices.unwrap_or_default();
    let nonpos = nonpos_indices.unwrap_or_default();
    let setup_seconds = setup_start.elapsed().as_secs_f64();

    let mut warm_coefficients: Option<Array1<f64>> = None;
    let mut fold_deviances: Vec<f64> = Vec::with_capacity(alpha_vec.len());
    let mut fit_seconds: Vec<f64> = Vec::with_capacity(alpha_vec.len());
    let mut validation_dot_seconds: Vec<f64> = Vec::with_capacity(alpha_vec.len());
    let mut validation_score_seconds: Vec<f64> = Vec::with_capacity(alpha_vec.len());
    let mut iterations: Vec<usize> = Vec::with_capacity(alpha_vec.len());
    let mut statuses: Vec<String> = Vec::with_capacity(alpha_vec.len());

    for &alpha in &alpha_vec {
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

        let cv_config = FitConfig {
            max_iterations: max_iter,
            tolerance: tol,
            min_weight: 1e-10,
            verbose: false,
            nonneg_indices: nonneg.clone(),
            nonpos_indices: nonpos.clone(),
            regularization: reg_config,
            skip_covariance: true,
            standardization: if fit_scale_only_ridge_on_original {
                standardization.clone()
            } else {
                None
            },
        };

        let fit_start = Instant::now();
        let result = match fit_glm_unified_with_sparse_cache(
            &y_train_array,
            x_train_work.view(),
            fam.as_ref(),
            link_fn.as_ref(),
            &cv_config,
            offset_train_array.as_ref(),
            weights_train_array.as_ref(),
            warm_coefficients.as_ref(),
            sparse_cache.as_ref(),
        ) {
            Ok(r) => {
                fit_seconds.push(fit_start.elapsed().as_secs_f64());
                iterations.push(r.iterations);
                statuses.push(r.solver_status.clone());
                r
            }
            Err(_) => {
                fit_seconds.push(fit_start.elapsed().as_secs_f64());
                iterations.push(0);
                statuses.push("error".to_string());
                validation_dot_seconds.push(0.0);
                validation_score_seconds.push(0.0);
                fold_deviances.push(f64::INFINITY);
                continue;
            }
        };

        warm_coefficients = Some(result.coefficients.clone());
        let dot_start = Instant::now();
        let validation_coefficients = if let Some(std) = &standardization {
            if fit_scale_only_ridge_on_original {
                result.coefficients.clone()
            } else {
                match std.to_original_coefficients(&result.coefficients, fit_intercept) {
                    Ok(beta) => beta,
                    Err(_) => {
                        validation_dot_seconds.push(dot_start.elapsed().as_secs_f64());
                        validation_score_seconds.push(0.0);
                        fold_deviances.push(f64::INFINITY);
                        continue;
                    }
                }
            }
        } else {
            result.coefficients.clone()
        };

        if let Some(dev) = poisson_log_validation_deviance_score(
            x_val_view,
            &validation_coefficients,
            &y_val_array,
            offset_val_array.as_ref(),
            weights_val_array.as_ref(),
            fam.as_ref(),
            link_fn.as_ref(),
        ) {
            validation_dot_seconds.push(dot_start.elapsed().as_secs_f64());
            validation_score_seconds.push(0.0);
            fold_deviances.push(dev);
        } else {
            let lp = matrix_vector_dot(x_val_view, &validation_coefficients);
            validation_dot_seconds.push(dot_start.elapsed().as_secs_f64());
            let score_start = Instant::now();
            let lp_off = if let Some(ref o) = offset_val_array {
                &lp + o
            } else {
                lp
            };
            let mu_val = link_fn.inverse(&lp_off);
            let unit_dev = fam.unit_deviance(&y_val_array, &mu_val);
            fold_deviances.push(validation_deviance_score(
                &unit_dev,
                weights_val_array.as_ref(),
            ));
            validation_score_seconds.push(score_start.elapsed().as_secs_f64());
        }
    }

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("alphas", alpha_vec)?;
    dict.set_item("fold_deviances", fold_deviances)?;

    let profile = pyo3::types::PyDict::new(py);
    profile.set_item("n_train", n_train)?;
    profile.set_item("n_val", n_val)?;
    profile.set_item("p", p)?;
    profile.set_item("n_alphas", fit_seconds.len())?;
    profile.set_item("rayon_threads", rayon::current_num_threads())?;
    profile.set_item("input_seconds", input_seconds)?;
    profile.set_item("validation_seconds", validation_seconds)?;
    profile.set_item("setup_seconds", setup_seconds)?;
    profile.set_item("standardize_seconds", standardize_seconds)?;
    profile.set_item("sparse_cache_seconds", sparse_cache_seconds)?;
    profile.set_item("fit_seconds", fit_seconds)?;
    profile.set_item("validation_dot_seconds", validation_dot_seconds)?;
    profile.set_item("validation_score_seconds", validation_score_seconds)?;
    profile.set_item("iterations", iterations)?;
    profile.set_item("statuses", statuses)?;
    profile.set_item("total_wall_seconds", total_start.elapsed().as_secs_f64())?;
    dict.set_item("profile", profile)?;

    Ok(dict.into())
}

#[pyfunction]
#[pyo3(signature = (y, x, family, link=None, var_power=1.5, theta=1.0, offset=None, weights=None, alphas=None, l1_ratio=0.0, n_folds=5, max_iter=25, tol=1e-8, seed=None, nonneg_indices=None, nonpos_indices=None, allow_extended_tweedie=false, fit_intercept=true, center=None, scale=None))]
pub fn fit_cv_path_py<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<f64>,
    x: PyReadonlyArray2<f64>,
    family: &str,
    link: Option<&str>,
    var_power: f64,
    theta: f64,
    offset: Option<PyReadonlyArray1<f64>>,
    weights: Option<PyReadonlyArray1<f64>>,
    alphas: Option<Vec<f64>>,
    l1_ratio: f64,
    n_folds: usize,
    max_iter: usize,
    tol: f64,
    seed: Option<u64>,
    nonneg_indices: Option<Vec<usize>>,
    nonpos_indices: Option<Vec<usize>>,
    allow_extended_tweedie: bool,
    fit_intercept: bool,
    center: Option<PyReadonlyArray1<f64>>,
    scale: Option<PyReadonlyArray1<f64>>,
) -> PyResult<Py<PyAny>> {
    let total_start = Instant::now();
    let input_start = Instant::now();
    let y_array: Array1<f64> = y.as_array().to_owned();
    let x_view = x.as_array(); // Zero-copy view
    let n = y_array.len();
    let p = x_view.ncols();
    let input_seconds = input_start.elapsed().as_secs_f64();

    let validation_start = Instant::now();
    // Dimension validation BEFORE any slicing / modular arithmetic. A direct
    // caller passing mismatched arrays (or n_folds == 0) would otherwise hit an
    // out-of-bounds index or divide-by-zero panic inside the rayon closure,
    // crossing FFI as an opaque PanicException. Mirror the dimension contract
    // the core solver enforces for the other fit entry points.
    if x_view.nrows() != n {
        return Err(PyValueError::new_err(format!(
            "x has {} rows but y has length {}; they must match",
            x_view.nrows(),
            n
        )));
    }
    if let Some(ref o) = offset {
        if o.as_array().len() != n {
            return Err(PyValueError::new_err(format!(
                "offset has length {} but y has length {}; they must match",
                o.as_array().len(),
                n
            )));
        }
    }
    if let Some(ref w) = weights {
        if w.as_array().len() != n {
            return Err(PyValueError::new_err(format!(
                "weights has length {} but y has length {}; they must match",
                w.as_array().len(),
                n
            )));
        }
    }
    if n_folds < 1 {
        return Err(PyValueError::new_err(
            "n_folds must be at least 1".to_string(),
        ));
    }
    if n_folds > n {
        return Err(PyValueError::new_err(format!(
            "n_folds ({n_folds}) cannot exceed the number of observations ({n})"
        )));
    }
    let validation_seconds = validation_start.elapsed().as_secs_f64();

    let setup_start = Instant::now();
    let offset_array: Option<Array1<f64>> = offset.map(|o| o.as_array().to_owned());
    let weights_array: Option<Array1<f64>> = weights.map(|w| w.as_array().to_owned());
    let standardization = build_standardization(center, scale, p)?;

    // RS-ACT-005: the previous fallback here was a hard-coded geometric grid
    // (10 -> 1e-4, 20 points) that was completely blind to family, link,
    // offset, weights and n. It under-shot ``alpha_max`` by roughly a factor of
    // ``n`` (e.g. 0.799 vs the correct 479.6 on 600 rows), which left penalised
    // coefficients non-zero at the grid endpoint and biased CV alpha selection
    // toward over-shrinkage. The Python helper
    // ``rustystats.regularization_path.compute_alpha_max`` now derives the
    // grid from the GLM score at the offset/weight-aware null; route through
    // it instead of silently shipping a bad grid here.
    let alpha_vec = alphas.ok_or_else(|| {
        PyValueError::new_err(
            "fit_cv_path_py requires an explicit `alphas=` grid. Build one with \
             rustystats.regularization_path.compute_alpha_max + generate_alpha_path \
             (RS-ACT-005); the previous silent geometric fallback was score-blind and \
             under-sized the grid by roughly a factor of n.",
        )
    })?;

    let fold_assignments: Vec<usize> = {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        (0..n)
            .map(|i| {
                let mut h = DefaultHasher::new();
                (i, seed.unwrap_or(42)).hash(&mut h);
                (h.finish() as usize) % n_folds
            })
            .collect()
    };

    validate_tweedie_fit_response(family, &y_array, var_power, allow_extended_tweedie)?;
    let _fam =
        family_from_name_with_tweedie_support(family, var_power, theta, allow_extended_tweedie)?;
    let default_link = default_link_name(family);
    let _link_fn = link_from_name(link.unwrap_or(default_link))?;
    let score_validation_from_original_rows = _fam.name().eq_ignore_ascii_case("poisson")
        && _link_fn.name() == "log"
        && x_view.as_slice().is_some()
        && y_array.as_slice().is_some()
        && offset_array
            .as_ref()
            .is_none_or(|offset| offset.as_slice().is_some())
        && weights_array
            .as_ref()
            .is_none_or(|weights| weights.as_slice().is_some());
    let fit_scale_only_ridge_on_original = l1_ratio <= 0.0
        && standardization
            .as_ref()
            .is_some_and(|std| std.center.iter().all(|&center| center == 0.0));

    let nonneg = nonneg_indices.unwrap_or_default();
    let nonpos = nonpos_indices.unwrap_or_default();
    let setup_seconds = setup_start.elapsed().as_secs_f64();

    let cv_parallel_start = Instant::now();
    let fold_outputs: Vec<(Vec<f64>, CVFoldProfile)> = (0..n_folds)
        .into_par_iter()
        .map(|fold| {
            let split_start = Instant::now();
            let train_mask: Vec<bool> = fold_assignments.iter().map(|&f| f != fold).collect();
            let n_train = train_mask.iter().filter(|&&b| b).count();
            let n_val = n - n_train;
            let mut profile = CVFoldProfile {
                fold,
                n_train,
                n_val,
                ..CVFoldProfile::default()
            };

            let mut y_train = Array1::zeros(n_train);
            let mut x_train = Array2::zeros((n_train, p));
            let mut offset_train: Option<Array1<f64>> =
                offset_array.as_ref().map(|_| Array1::zeros(n_train));
            let mut weights_train: Option<Array1<f64>> =
                weights_array.as_ref().map(|_| Array1::zeros(n_train));
            let mut val_indices: Vec<usize> = if score_validation_from_original_rows {
                Vec::with_capacity(n_val)
            } else {
                Vec::new()
            };
            let mut y_val: Option<Array1<f64>> = if score_validation_from_original_rows {
                None
            } else {
                Some(Array1::zeros(n_val))
            };
            let mut x_val: Option<Array2<f64>> = if score_validation_from_original_rows {
                None
            } else {
                Some(Array2::zeros((n_val, p)))
            };
            let mut offset_val: Option<Array1<f64>> = if score_validation_from_original_rows {
                None
            } else {
                offset_array.as_ref().map(|_| Array1::zeros(n_val))
            };
            let mut weights_val: Option<Array1<f64>> = if score_validation_from_original_rows {
                None
            } else {
                weights_array.as_ref().map(|_| Array1::zeros(n_val))
            };

            let std_ref = standardization.as_ref();
            let (mut ti, mut vi) = (0, 0);
            for i in 0..n {
                if train_mask[i] {
                    y_train[ti] = y_array[i];
                    if let Some(std) = std_ref {
                        if fit_scale_only_ridge_on_original {
                            x_train.row_mut(ti).assign(&x_view.row(i));
                        } else {
                            for j in 0..p {
                                x_train[[ti, j]] = (x_view[[i, j]] - std.center[j]) / std.scale[j];
                            }
                        }
                    } else {
                        x_train.row_mut(ti).assign(&x_view.row(i));
                    }
                    if let (Some(ref o), Some(ref mut ot)) = (&offset_array, &mut offset_train) {
                        ot[ti] = o[i];
                    }
                    if let (Some(ref w), Some(ref mut wt)) = (&weights_array, &mut weights_train) {
                        wt[ti] = w[i];
                    }
                    ti += 1;
                } else {
                    if score_validation_from_original_rows {
                        val_indices.push(i);
                    } else {
                        if let Some(ref mut yv) = y_val {
                            yv[vi] = y_array[i];
                        }
                        if let Some(ref mut xv) = x_val {
                            xv.row_mut(vi).assign(&x_view.row(i));
                        }
                        if let (Some(ref o), Some(ref mut ov)) = (&offset_array, &mut offset_val) {
                            ov[vi] = o[i];
                        }
                        if let (Some(ref w), Some(ref mut wv)) = (&weights_array, &mut weights_val)
                        {
                            wv[vi] = w[i];
                        }
                    }
                    vi += 1;
                }
            }
            profile.split_copy_seconds = split_start.elapsed().as_secs_f64();

            let standardize_start = Instant::now();
            if let Some(std) = &standardization {
                if let Err(_err) = std.validate(p) {
                    profile.standardize_seconds = standardize_start.elapsed().as_secs_f64();
                    return (vec![f64::INFINITY; alpha_vec.len()], profile);
                }
            }
            profile.standardize_seconds = standardize_start.elapsed().as_secs_f64();
            let sparse_cache_start = Instant::now();
            let fold_sparse_cache = build_sparse_row_cache_if_beneficial(x_train.view());
            profile.sparse_cache_seconds = sparse_cache_start.elapsed().as_secs_f64();

            let fold_setup_start = Instant::now();
            let thread_fam = match family_from_name_with_tweedie_support(
                family,
                var_power,
                theta,
                allow_extended_tweedie,
            ) {
                Ok(fam) => fam,
                Err(_) => {
                    profile.setup_seconds = fold_setup_start.elapsed().as_secs_f64();
                    return (vec![f64::INFINITY; alpha_vec.len()], profile);
                }
            };
            let link_name = link.unwrap_or(default_link);
            let thread_link = match link_from_name(link_name) {
                Ok(link) => link,
                Err(_) => {
                    profile.setup_seconds = fold_setup_start.elapsed().as_secs_f64();
                    return (vec![f64::INFINITY; alpha_vec.len()], profile);
                }
            };

            let mut warm_coefficients: Option<Array1<f64>> = None;
            let mut fold_deviances: Vec<f64> = Vec::with_capacity(alpha_vec.len());
            profile.setup_seconds = fold_setup_start.elapsed().as_secs_f64();

            for &alpha in &alpha_vec {
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

                let cv_config = FitConfig {
                    max_iterations: max_iter,
                    tolerance: tol,
                    min_weight: 1e-10,
                    verbose: false,
                    nonneg_indices: nonneg.clone(),
                    nonpos_indices: nonpos.clone(),
                    regularization: reg_config,
                    skip_covariance: true,
                    standardization: if fit_scale_only_ridge_on_original {
                        standardization.clone()
                    } else {
                        None
                    },
                };

                let fit_start = Instant::now();
                let result = match fit_glm_unified_with_sparse_cache(
                    &y_train,
                    x_train.view(),
                    thread_fam.as_ref(),
                    thread_link.as_ref(),
                    &cv_config,
                    offset_train.as_ref(),
                    weights_train.as_ref(),
                    warm_coefficients.as_ref(),
                    fold_sparse_cache.as_ref(),
                ) {
                    Ok(r) => {
                        profile.fit_seconds.push(fit_start.elapsed().as_secs_f64());
                        profile.iterations.push(r.iterations);
                        profile.statuses.push(r.solver_status.clone());
                        r
                    }
                    Err(_) => {
                        profile.fit_seconds.push(fit_start.elapsed().as_secs_f64());
                        profile.iterations.push(0);
                        profile.statuses.push("error".to_string());
                        profile.validation_dot_seconds.push(0.0);
                        profile.validation_score_seconds.push(0.0);
                        fold_deviances.push(f64::INFINITY);
                        continue;
                    }
                };

                warm_coefficients = Some(result.coefficients.clone());
                let dot_start = Instant::now();
                let validation_coefficients = if let Some(std) = &standardization {
                    if fit_scale_only_ridge_on_original {
                        result.coefficients.clone()
                    } else {
                        match std.to_original_coefficients(&result.coefficients, fit_intercept) {
                            Ok(beta) => beta,
                            Err(_) => {
                                profile
                                    .validation_dot_seconds
                                    .push(dot_start.elapsed().as_secs_f64());
                                profile.validation_score_seconds.push(0.0);
                                fold_deviances.push(f64::INFINITY);
                                continue;
                            }
                        }
                    }
                } else {
                    result.coefficients.clone()
                };
                if score_validation_from_original_rows {
                    let dev = poisson_log_validation_deviance_score_rows(
                        x_view,
                        &validation_coefficients,
                        &y_array,
                        offset_array.as_ref(),
                        weights_array.as_ref(),
                        &val_indices,
                    )
                    .unwrap_or(f64::INFINITY);
                    profile
                        .validation_dot_seconds
                        .push(dot_start.elapsed().as_secs_f64());
                    profile.validation_score_seconds.push(0.0);
                    fold_deviances.push(dev);
                } else if let (Some(ref xv), Some(ref yv)) = (&x_val, &y_val) {
                    if let Some(dev) = poisson_log_validation_deviance_score(
                        xv.view(),
                        &validation_coefficients,
                        yv,
                        offset_val.as_ref(),
                        weights_val.as_ref(),
                        thread_fam.as_ref(),
                        thread_link.as_ref(),
                    ) {
                        profile
                            .validation_dot_seconds
                            .push(dot_start.elapsed().as_secs_f64());
                        profile.validation_score_seconds.push(0.0);
                        fold_deviances.push(dev);
                    } else {
                        let lp = matrix_vector_dot(xv.view(), &validation_coefficients);
                        profile
                            .validation_dot_seconds
                            .push(dot_start.elapsed().as_secs_f64());
                        let score_start = Instant::now();
                        let lp_off = if let Some(ref o) = offset_val {
                            &lp + o
                        } else {
                            lp
                        };
                        let mu_val = thread_link.inverse(&lp_off);
                        let unit_dev = thread_fam.unit_deviance(yv, &mu_val);
                        fold_deviances
                            .push(validation_deviance_score(&unit_dev, weights_val.as_ref()));
                        profile
                            .validation_score_seconds
                            .push(score_start.elapsed().as_secs_f64());
                    }
                } else {
                    profile
                        .validation_dot_seconds
                        .push(dot_start.elapsed().as_secs_f64());
                    profile.validation_score_seconds.push(0.0);
                    fold_deviances.push(f64::INFINITY);
                }
            }
            (fold_deviances, profile)
        })
        .collect();
    let cv_parallel_wall_seconds = cv_parallel_start.elapsed().as_secs_f64();

    let fold_all_results: Vec<Vec<f64>> = fold_outputs
        .iter()
        .map(|(scores, _profile)| scores.clone())
        .collect();
    let fold_profiles: Vec<CVFoldProfile> = fold_outputs
        .iter()
        .map(|(_scores, profile)| profile.clone())
        .collect();

    let aggregate_start = Instant::now();
    let mut path_results: Vec<CVPathPoint> = Vec::with_capacity(alpha_vec.len());
    for (ai, &alpha) in alpha_vec.iter().enumerate() {
        let fds: Vec<f64> = fold_all_results
            .iter()
            .map(|fr| fr.get(ai).copied().unwrap_or(f64::INFINITY))
            .collect();
        let all_folds_finite = fds.len() == n_folds && fds.iter().all(|x| x.is_finite());
        let mean = if all_folds_finite {
            fds.iter().sum::<f64>() / fds.len() as f64
        } else {
            f64::INFINITY
        };
        let se = if all_folds_finite && fds.len() > 1 {
            let var = fds.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (fds.len() - 1) as f64;
            (var / fds.len() as f64).sqrt()
        } else if all_folds_finite {
            0.0
        } else {
            f64::INFINITY
        };
        path_results.push(CVPathPoint {
            alpha,
            cv_deviance_mean: mean,
            cv_deviance_se: se,
        });
    }
    let aggregate_seconds = aggregate_start.elapsed().as_secs_f64();

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item(
        "alphas",
        path_results.iter().map(|r| r.alpha).collect::<Vec<_>>(),
    )?;
    dict.set_item(
        "cv_deviance_mean",
        path_results
            .iter()
            .map(|r| r.cv_deviance_mean)
            .collect::<Vec<_>>(),
    )?;
    dict.set_item(
        "cv_deviance_se",
        path_results
            .iter()
            .map(|r| r.cv_deviance_se)
            .collect::<Vec<_>>(),
    )?;
    let cv_fold_scores: Vec<Vec<f64>> = (0..alpha_vec.len())
        .map(|ai| {
            fold_all_results
                .iter()
                .map(|fr| fr.get(ai).copied().unwrap_or(f64::INFINITY))
                .collect::<Vec<_>>()
        })
        .collect();
    dict.set_item("cv_fold_scores", cv_fold_scores)?;
    // Surface the per-row fold assignment so callers can reproduce the split
    // (the seeded DefaultHasher partition is not reproducible outside Rust) and
    // hand-verify the per-fold weighted deviance (RS-ACT-001 backlog #5).
    dict.set_item("fold_assignments", fold_assignments)?;

    let profile = pyo3::types::PyDict::new(py);
    profile.set_item("n", n)?;
    profile.set_item("p", p)?;
    profile.set_item("n_folds", n_folds)?;
    profile.set_item("n_alphas", alpha_vec.len())?;
    profile.set_item("rayon_threads", rayon::current_num_threads())?;
    profile.set_item("input_seconds", input_seconds)?;
    profile.set_item("validation_seconds", validation_seconds)?;
    profile.set_item("setup_seconds", setup_seconds)?;
    profile.set_item("cv_parallel_wall_seconds", cv_parallel_wall_seconds)?;
    profile.set_item("aggregate_seconds", aggregate_seconds)?;
    profile.set_item("total_wall_seconds", total_start.elapsed().as_secs_f64())?;

    let total_split_copy: f64 = fold_profiles.iter().map(|f| f.split_copy_seconds).sum();
    let total_sparse_cache: f64 = fold_profiles.iter().map(|f| f.sparse_cache_seconds).sum();
    let total_standardize: f64 = fold_profiles.iter().map(|f| f.standardize_seconds).sum();
    let total_fold_setup: f64 = fold_profiles.iter().map(|f| f.setup_seconds).sum();
    let total_fit: f64 = fold_profiles
        .iter()
        .flat_map(|f| f.fit_seconds.iter())
        .sum();
    let total_validation_dot: f64 = fold_profiles
        .iter()
        .flat_map(|f| f.validation_dot_seconds.iter())
        .sum();
    let total_validation_score: f64 = fold_profiles
        .iter()
        .flat_map(|f| f.validation_score_seconds.iter())
        .sum();

    let totals = pyo3::types::PyDict::new(py);
    totals.set_item("fold_split_copy_seconds", total_split_copy)?;
    totals.set_item("fold_sparse_cache_seconds", total_sparse_cache)?;
    totals.set_item("fold_standardize_seconds", total_standardize)?;
    totals.set_item("fold_setup_seconds", total_fold_setup)?;
    totals.set_item("alpha_fit_seconds", total_fit)?;
    totals.set_item("alpha_validation_dot_seconds", total_validation_dot)?;
    totals.set_item("alpha_validation_score_seconds", total_validation_score)?;
    profile.set_item("summed_work_seconds", totals)?;

    let fold_profile_list = pyo3::types::PyList::empty(py);
    for fold_profile in &fold_profiles {
        let fold_dict = pyo3::types::PyDict::new(py);
        fold_dict.set_item("fold", fold_profile.fold)?;
        fold_dict.set_item("n_train", fold_profile.n_train)?;
        fold_dict.set_item("n_val", fold_profile.n_val)?;
        fold_dict.set_item("split_copy_seconds", fold_profile.split_copy_seconds)?;
        fold_dict.set_item("sparse_cache_seconds", fold_profile.sparse_cache_seconds)?;
        fold_dict.set_item("standardize_seconds", fold_profile.standardize_seconds)?;
        fold_dict.set_item("setup_seconds", fold_profile.setup_seconds)?;
        fold_dict.set_item("fit_seconds", fold_profile.fit_seconds.clone())?;
        fold_dict.set_item(
            "validation_dot_seconds",
            fold_profile.validation_dot_seconds.clone(),
        )?;
        fold_dict.set_item(
            "validation_score_seconds",
            fold_profile.validation_score_seconds.clone(),
        )?;
        fold_dict.set_item("iterations", fold_profile.iterations.clone())?;
        fold_dict.set_item("statuses", fold_profile.statuses.clone())?;
        fold_profile_list.append(fold_dict)?;
    }
    profile.set_item("folds", fold_profile_list)?;
    dict.set_item("profile", profile)?;

    let best_idx = path_results
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.cv_deviance_mean.total_cmp(&b.cv_deviance_mean))
        .map(|(i, _)| i)
        .unwrap_or(0);
    dict.set_item("best_alpha", path_results[best_idx].alpha)?;
    dict.set_item("best_cv_deviance", path_results[best_idx].cv_deviance_mean)?;
    Ok(dict.into())
}

#[cfg(test)]
mod tests {
    use super::{
        poisson_log_validation_deviance_score, poisson_log_validation_deviance_score_rows,
        validation_deviance_score,
    };
    use ndarray::{array, Array2};
    use rustystats_core::families::PoissonFamily;
    use rustystats_core::links::LogLink;

    #[test]
    fn validation_deviance_score_uses_validation_weights() {
        let unit_deviance = array![1.0, 9.0, 25.0];
        let weights = array![1.0, 3.0, 6.0];

        let score = validation_deviance_score(&unit_deviance, Some(&weights));

        assert!((score - 17.8).abs() < 1e-12);
    }

    #[test]
    fn validation_deviance_score_defaults_to_unweighted_mean() {
        let unit_deviance = array![1.0, 9.0, 26.0];

        let score = validation_deviance_score(&unit_deviance, None);

        assert!((score - 12.0).abs() < 1e-12);
    }

    #[test]
    fn poisson_log_validation_rows_match_dense_subset() {
        let x = Array2::from_shape_vec(
            (5, 3),
            vec![
                1.0, 0.2, 0.0, 1.0, -0.4, 1.0, 1.0, 0.8, 0.0, 1.0, 0.1, 1.0, 1.0, -0.2, 0.0,
            ],
        )
        .expect("test design should be valid");
        let rows = vec![1_usize, 3, 4];
        let x_subset = Array2::from_shape_fn((rows.len(), x.ncols()), |(i, j)| x[[rows[i], j]]);
        let y = array![0.2, 0.7, 0.4, 1.1, 0.3];
        let y_subset = array![y[1], y[3], y[4]];
        let offset = array![0.0, 0.1, -0.2, 0.05, 0.2];
        let offset_subset = array![offset[1], offset[3], offset[4]];
        let weights = array![1.0, 2.0, 1.5, 0.75, 3.0];
        let weights_subset = array![weights[1], weights[3], weights[4]];
        let coefficients = array![0.3, -0.15, 0.25];

        let dense = poisson_log_validation_deviance_score(
            x_subset.view(),
            &coefficients,
            &y_subset,
            Some(&offset_subset),
            Some(&weights_subset),
            &PoissonFamily,
            &LogLink,
        )
        .expect("dense scorer should handle poisson/log");
        let by_rows = poisson_log_validation_deviance_score_rows(
            x.view(),
            &coefficients,
            &y,
            Some(&offset),
            Some(&weights),
            &rows,
        )
        .expect("row scorer should handle original rows");

        assert!((dense - by_rows).abs() < 1e-12);
    }
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
#[pyo3(signature = (y, x_full, smooth_col_ranges, smooth_penalties, family, link=None, offset=None, weights=None, max_iter=25, tol=1e-8, lambda_min=0.001, lambda_max=1000.0, smooth_monotonicity=None, store_design_matrix=false, nonneg_indices=None, nonpos_indices=None, var_power=1.5, theta=1.0, allow_extended_tweedie=false))]
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
) -> PyResult<Py<PyAny>> {
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

    let mono_vec = smooth_monotonicity.unwrap_or_else(|| vec![None; n_terms]);
    if mono_vec.len() != n_terms {
        return Err(PyValueError::new_err(format!(
            "smooth_monotonicity has {} entries but expected {}",
            mono_vec.len(),
            n_terms
        )));
    }

    let specs: Vec<SmoothTermSpec> = smooth_col_ranges
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
        .collect::<PyResult<Vec<_>>>()?;

    let config = build_smooth_config(max_iter, tol, lambda_min, lambda_max);

    let nn = nonneg_indices.unwrap_or_default();
    let np = nonpos_indices.unwrap_or_default();

    let result = fit_smooth_glm_full_matrix(
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

    let family_name = if let Some(resolved_theta) = resolve_negbinomial_theta(family, theta)? {
        format!("NegativeBinomial(theta={:.4})", resolved_theta)
    } else if let Some(resolved_var_power) = resolve_tweedie_var_power(family, var_power)? {
        format!("Tweedie(p={:.4})", resolved_var_power)
    } else {
        fam.name().to_string()
    };

    smooth_result_to_py(py, result, store_design_matrix, family_name)
}
