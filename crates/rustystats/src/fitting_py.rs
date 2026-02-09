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

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use ndarray::{Array1, Array2};
use rayon::prelude::*;
use rayon::iter::IntoParallelIterator;

use rustystats_core::families::{PoissonFamily, NegativeBinomialFamily};
use rustystats_core::solvers::{FitConfig, fit_glm_unified, IRLSConfig, IRLSResult, fit_smooth_glm_full_matrix, SmoothGLMConfig, Monotonicity, SmoothTermSpec};
use rustystats_core::regularization::RegularizationConfig;
use rustystats_core::diagnostics::{estimate_theta_profile, estimate_theta_moments};

use crate::families_py::{family_from_name, link_from_name, default_link_name};
use crate::results_py::PyGLMResults;

// =============================================================================
// Helpers: reduce boilerplate for smooth result conversion
// =============================================================================

/// Convert SmoothGLMResult → (PyGLMResults, smooth_metadata_dict) Python tuple.
fn smooth_result_to_py<'py>(
    py: Python<'py>,
    result: rustystats_core::solvers::SmoothGLMResult,
    store_design_matrix: bool,
) -> PyResult<PyObject> {
    let n_obs = result.y.len();
    let n_params = result.coefficients.len();

    let glm_result = PyGLMResults {
        coefficients: result.coefficients,
        fitted_values: result.fitted_values,
        linear_predictor: result.linear_predictor,
        deviance: result.deviance,
        iterations: result.iterations,
        converged: result.converged,
        covariance_unscaled: result.covariance_unscaled,
        n_obs, n_params,
        y: result.y,
        family_name: result.family_name,
        prior_weights: result.prior_weights,
        penalty: result.penalty,
        design_matrix: if store_design_matrix { Some(result.design_matrix) } else { None },
        irls_weights: result.irls_weights,
        offset: result.offset,
    };

    let smooth_dict = pyo3::types::PyDict::new_bound(py);
    smooth_dict.set_item("lambdas", result.lambdas)?;
    smooth_dict.set_item("smooth_edfs", result.smooth_edfs)?;
    smooth_dict.set_item("total_edf", result.total_edf)?;
    smooth_dict.set_item("gcv", result.gcv)?;

    let tuple = pyo3::types::PyTuple::new_bound(py, &[
        glm_result.into_py(py).into_bound(py),
        smooth_dict.into_any(),
    ]);
    Ok(tuple.into())
}

/// Build a SmoothGLMConfig from common parameters.
fn build_smooth_config(max_iter: usize, tol: f64, lambda_min: f64, lambda_max: f64) -> SmoothGLMConfig {
    SmoothGLMConfig {
        irls_config: IRLSConfig {
            max_iterations: max_iter, tolerance: tol, min_weight: 1e-10,
            verbose: false, nonneg_indices: Vec::new(), nonpos_indices: Vec::new(),
        },
        n_lambda: 30, lambda_min, lambda_max, lambda_tol: 1e-4,
        max_lambda_iter: 10, lambda_method: "gcv".to_string(),
    }
}

// =============================================================================
// fit_glm_py — Standard GLM
// =============================================================================

#[pyfunction]
#[pyo3(signature = (y, x, family, link=None, var_power=1.5, theta=1.0, offset=None, weights=None, alpha=0.0, l1_ratio=0.0, max_iter=25, tol=1e-8, nonneg_indices=None, nonpos_indices=None, store_design_matrix=false))]
pub fn fit_glm_py(
    y: PyReadonlyArray1<f64>, x: PyReadonlyArray2<f64>,
    family: &str, link: Option<&str>,
    var_power: f64, theta: f64,
    offset: Option<PyReadonlyArray1<f64>>, weights: Option<PyReadonlyArray1<f64>>,
    alpha: f64, l1_ratio: f64, max_iter: usize, tol: f64,
    nonneg_indices: Option<Vec<usize>>, nonpos_indices: Option<Vec<usize>>,
    store_design_matrix: bool,
) -> PyResult<PyGLMResults> {
    let y_array: Array1<f64> = y.as_array().to_owned();
    let x_array: Array2<f64> = x.as_array().to_owned();
    let n_obs = y_array.len();
    let n_params = x_array.ncols();
    let offset_array: Option<Array1<f64>> = offset.map(|o| o.as_array().to_owned());
    let weights_array: Option<Array1<f64>> = weights.map(|w| w.as_array().to_owned());

    let reg_config = if alpha > 0.0 {
        if l1_ratio >= 1.0 { RegularizationConfig::lasso(alpha) }
        else if l1_ratio <= 0.0 { RegularizationConfig::ridge(alpha) }
        else { RegularizationConfig::elastic_net(alpha, l1_ratio) }
    } else { RegularizationConfig::none() };

    let config = FitConfig {
        max_iterations: max_iter, tolerance: tol, min_weight: 1e-10, verbose: false,
        nonneg_indices: nonneg_indices.unwrap_or_default(),
        nonpos_indices: nonpos_indices.unwrap_or_default(),
        regularization: reg_config,
        skip_covariance: false,
    };

    // Gamma validation
    if family.to_lowercase() == "gamma" {
        let n_invalid = y_array.iter().filter(|&&v| v <= 0.0).count();
        if n_invalid > 0 {
            return Err(PyValueError::new_err(format!(
                "Gamma family requires strictly positive response values (y > 0). \
                 Found {} values <= 0 out of {} observations.", n_invalid, y_array.len()
            )));
        }
    }

    let fam = family_from_name(family, var_power, theta)?;
    let lnk = link_from_name(link.unwrap_or(default_link_name(family)))?;

    let result: IRLSResult = fit_glm_unified(
        &y_array, &x_array, fam.as_ref(), lnk.as_ref(), &config,
        offset_array.as_ref(), weights_array.as_ref(), None,
    ).map_err(|e| PyValueError::new_err(format!("GLM fitting failed: {}", e)))?;

    let family_name = if result.family_name.to_lowercase().contains("negativebinomial")
        || result.family_name.to_lowercase().contains("negbinomial") {
        format!("NegativeBinomial(theta={:.4})", theta)
    } else { result.family_name };

    Ok(PyGLMResults {
        coefficients: result.coefficients, fitted_values: result.fitted_values,
        linear_predictor: result.linear_predictor, deviance: result.deviance,
        iterations: result.iterations, converged: result.converged,
        covariance_unscaled: result.covariance_unscaled, n_obs, n_params,
        y: result.y, family_name, prior_weights: result.prior_weights,
        penalty: result.penalty,
        design_matrix: if store_design_matrix { Some(x_array) } else { None },
        irls_weights: result.irls_weights, offset: offset_array,
    })
}

// =============================================================================
// fit_negbinomial_py — NegBin with automatic theta
// =============================================================================

#[pyfunction]
#[pyo3(signature = (y, x, link=None, init_theta=None, theta_tol=1e-5, max_theta_iter=10, offset=None, weights=None, max_iter=25, tol=1e-8, alpha=0.0, l1_ratio=0.0, store_design_matrix=false))]
pub fn fit_negbinomial_py(
    y: PyReadonlyArray1<f64>, x: PyReadonlyArray2<f64>,
    link: Option<&str>, init_theta: Option<f64>,
    theta_tol: f64, max_theta_iter: usize,
    offset: Option<PyReadonlyArray1<f64>>, weights: Option<PyReadonlyArray1<f64>>,
    max_iter: usize, tol: f64,
    alpha: f64, l1_ratio: f64,
    store_design_matrix: bool,
) -> PyResult<PyGLMResults> {
    let y_array: Array1<f64> = y.as_array().to_owned();
    let x_array: Array2<f64> = x.as_array().to_owned();
    let n_obs = y_array.len();
    let n_params = x_array.ncols();
    let offset_array: Option<Array1<f64>> = offset.map(|o| o.as_array().to_owned());
    let weights_array: Option<Array1<f64>> = weights.map(|w| w.as_array().to_owned());

    let reg_config = if alpha > 0.0 {
        if l1_ratio >= 1.0 { RegularizationConfig::lasso(alpha) }
        else if l1_ratio <= 0.0 { RegularizationConfig::ridge(alpha) }
        else { RegularizationConfig::elastic_net(alpha, l1_ratio) }
    } else { RegularizationConfig::none() };

    let config_loose = FitConfig {
        max_iterations: max_iter, tolerance: 1e-4, min_weight: 1e-10, verbose: false,
        nonneg_indices: Vec::new(), nonpos_indices: Vec::new(),
        regularization: reg_config.clone(), skip_covariance: false,
    };
    let config_final = FitConfig {
        tolerance: tol.max(1e-6), ..config_loose.clone()
    };

    let link_name = link.unwrap_or("log");
    let link_fn = link_from_name(link_name)?;
    if link_name != "log" && link_name != "identity" {
        return Err(PyValueError::new_err(format!(
            "Unknown link '{}' for NegativeBinomial. Use 'log' or 'identity'.", link_name)));
    }

    let mut theta = match init_theta {
        Some(t) if t <= 0.0 => return Err(PyValueError::new_err(format!("init_theta must be > 0, got {}", t))),
        Some(t) => t,
        None => {
            let poisson = PoissonFamily;
            let init_config = FitConfig { regularization: RegularizationConfig::none(), ..config_loose.clone() };
            let init_result = fit_glm_unified(&y_array, &x_array, &poisson, link_fn.as_ref(),
                &init_config, offset_array.as_ref(), weights_array.as_ref(), None,
            ).map_err(|e| PyValueError::new_err(format!("Initial Poisson fit failed: {}", e)))?;
            estimate_theta_moments(&y_array, &init_result.fitted_values)
        }
    };

    let mut result: IRLSResult;
    let mut coefficients: Option<Array1<f64>> = None;

    for _iter in 0..max_theta_iter {
        let family = NegativeBinomialFamily::new(theta);
        result = fit_glm_unified(&y_array, &x_array, &family, link_fn.as_ref(),
            &config_loose, offset_array.as_ref(), weights_array.as_ref(), coefficients.as_ref(),
        ).map_err(|e| PyValueError::new_err(format!("GLM fitting failed: {}", e)))?;

        coefficients = Some(result.coefficients.clone());
        let new_theta = estimate_theta_profile(&y_array, &result.fitted_values,
            weights_array.as_ref(), 0.01, 1000.0, 1e-6);
        if (new_theta - theta).abs() < theta_tol { theta = new_theta; break; }
        theta = new_theta;
    }

    let final_family = NegativeBinomialFamily::new(theta);
    result = fit_glm_unified(&y_array, &x_array, &final_family, link_fn.as_ref(),
        &config_final, offset_array.as_ref(), weights_array.as_ref(), coefficients.as_ref(),
    ).map_err(|e| PyValueError::new_err(format!("Final GLM fit failed: {}", e)))?;

    Ok(PyGLMResults {
        coefficients: result.coefficients, fitted_values: result.fitted_values,
        linear_predictor: result.linear_predictor, deviance: result.deviance,
        iterations: result.iterations, converged: result.converged,
        covariance_unscaled: result.covariance_unscaled, n_obs, n_params,
        y: y_array, family_name: format!("NegativeBinomial(theta={:.4})", theta),
        prior_weights: weights_array.unwrap_or_else(|| Array1::ones(n_obs)),
        penalty: result.penalty,
        design_matrix: if store_design_matrix { Some(x_array) } else { None },
        irls_weights: result.irls_weights, offset: offset_array,
    })
}

// =============================================================================
// fit_cv_path_py — Cross-validated regularization path
// =============================================================================

#[derive(Clone)]
struct CVPathPoint { alpha: f64, cv_deviance_mean: f64, cv_deviance_se: f64 }

#[pyfunction]
#[pyo3(signature = (y, x, family, link=None, var_power=1.5, theta=1.0, offset=None, weights=None, alphas=None, l1_ratio=0.0, n_folds=5, max_iter=25, tol=1e-8, seed=None))]
pub fn fit_cv_path_py<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<f64>, x: PyReadonlyArray2<f64>,
    family: &str, link: Option<&str>,
    var_power: f64, theta: f64,
    offset: Option<PyReadonlyArray1<f64>>, weights: Option<PyReadonlyArray1<f64>>,
    alphas: Option<Vec<f64>>, l1_ratio: f64, n_folds: usize,
    max_iter: usize, tol: f64, seed: Option<u64>,
) -> PyResult<PyObject> {
    let y_array: Array1<f64> = y.as_array().to_owned();
    let x_array: Array2<f64> = x.as_array().to_owned();
    let n = y_array.len();
    let p = x_array.ncols();
    let offset_array: Option<Array1<f64>> = offset.map(|o| o.as_array().to_owned());
    let weights_array: Option<Array1<f64>> = weights.map(|w| w.as_array().to_owned());

    let alpha_vec = alphas.unwrap_or_else(|| {
        (0..20).map(|i| { let t = i as f64 / 19.0; 10.0 * (0.0001f64 / 10.0).powf(t) }).collect()
    });

    let fold_assignments: Vec<usize> = {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        (0..n).map(|i| {
            let mut h = DefaultHasher::new();
            (i, seed.unwrap_or(42)).hash(&mut h);
            (h.finish() as usize) % n_folds
        }).collect()
    };

    let _fam = family_from_name(family, var_power, theta)?;
    let default_link = default_link_name(family);
    let _link_fn = link_from_name(link.unwrap_or(default_link))?;

    let fold_all_results: Vec<Vec<f64>> = (0..n_folds).into_par_iter().map(|fold| {
        let train_mask: Vec<bool> = fold_assignments.iter().map(|&f| f != fold).collect();
        let n_train = train_mask.iter().filter(|&&b| b).count();
        let n_val = n - n_train;

        let mut y_train = Array1::zeros(n_train);
        let mut x_train = Array2::zeros((n_train, p));
        let mut offset_train: Option<Array1<f64>> = offset_array.as_ref().map(|_| Array1::zeros(n_train));
        let mut weights_train: Option<Array1<f64>> = weights_array.as_ref().map(|_| Array1::zeros(n_train));
        let mut y_val = Array1::zeros(n_val);
        let mut x_val = Array2::zeros((n_val, p));
        let mut offset_val: Option<Array1<f64>> = offset_array.as_ref().map(|_| Array1::zeros(n_val));

        let (mut ti, mut vi) = (0, 0);
        for i in 0..n {
            if train_mask[i] {
                y_train[ti] = y_array[i]; x_train.row_mut(ti).assign(&x_array.row(i));
                if let (Some(ref o), Some(ref mut ot)) = (&offset_array, &mut offset_train) { ot[ti] = o[i]; }
                if let (Some(ref w), Some(ref mut wt)) = (&weights_array, &mut weights_train) { wt[ti] = w[i]; }
                ti += 1;
            } else {
                y_val[vi] = y_array[i]; x_val.row_mut(vi).assign(&x_array.row(i));
                if let (Some(ref o), Some(ref mut ov)) = (&offset_array, &mut offset_val) { ov[vi] = o[i]; }
                vi += 1;
            }
        }

        // Safe: family/link were validated before entering the parallel loop
        let thread_fam = family_from_name(family, var_power, theta)
            .expect("family pre-validated before parallel loop");
        let link_name = link.unwrap_or(default_link);
        let thread_link = link_from_name(link_name)
            .expect("link pre-validated before parallel loop");

        let mut warm_coefficients: Option<Array1<f64>> = None;
        let mut fold_deviances: Vec<f64> = Vec::with_capacity(alpha_vec.len());

        for &alpha in &alpha_vec {
            let reg_config = if alpha > 0.0 {
                if l1_ratio >= 1.0 { RegularizationConfig::lasso(alpha) }
                else if l1_ratio <= 0.0 { RegularizationConfig::ridge(alpha) }
                else { RegularizationConfig::elastic_net(alpha, l1_ratio) }
            } else { RegularizationConfig::none() };

            let cv_config = FitConfig {
                max_iterations: max_iter, tolerance: tol, min_weight: 1e-10, verbose: false,
                nonneg_indices: Vec::new(), nonpos_indices: Vec::new(),
                regularization: reg_config, skip_covariance: true,
            };

            let result = match fit_glm_unified(&y_train, &x_train, thread_fam.as_ref(),
                thread_link.as_ref(), &cv_config,
                offset_train.as_ref(), weights_train.as_ref(), warm_coefficients.as_ref())
            { Ok(r) => r, Err(_) => { fold_deviances.push(f64::INFINITY); continue; } };

            warm_coefficients = Some(result.coefficients.clone());
            let lp: Array1<f64> = x_val.dot(&result.coefficients);
            let lp_off = if let Some(ref o) = offset_val { &lp + o } else { lp };
            let mu_val = lp_off.mapv(|x| x.clamp(-700.0, 700.0).exp());
            let unit_dev = thread_fam.unit_deviance(&y_val, &mu_val);
            fold_deviances.push(unit_dev.mean().unwrap_or(f64::INFINITY));
        }
        fold_deviances
    }).collect();

    let mut path_results: Vec<CVPathPoint> = Vec::with_capacity(alpha_vec.len());
    for (ai, &alpha) in alpha_vec.iter().enumerate() {
        let fds: Vec<f64> = fold_all_results.iter()
            .map(|fr| fr.get(ai).copied().unwrap_or(f64::INFINITY)).filter(|&x| x.is_finite()).collect();
        let mean = if fds.is_empty() { f64::INFINITY } else { fds.iter().sum::<f64>() / fds.len() as f64 };
        let se = if fds.len() > 1 {
            let var = fds.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (fds.len() - 1) as f64;
            (var / fds.len() as f64).sqrt()
        } else { 0.0 };
        path_results.push(CVPathPoint { alpha, cv_deviance_mean: mean, cv_deviance_se: se });
    }

    let dict = pyo3::types::PyDict::new_bound(py);
    dict.set_item("alphas", path_results.iter().map(|r| r.alpha).collect::<Vec<_>>())?;
    dict.set_item("cv_deviance_mean", path_results.iter().map(|r| r.cv_deviance_mean).collect::<Vec<_>>())?;
    dict.set_item("cv_deviance_se", path_results.iter().map(|r| r.cv_deviance_se).collect::<Vec<_>>())?;

    let best_idx = path_results.iter().enumerate()
        .min_by(|(_, a), (_, b)| a.cv_deviance_mean.total_cmp(&b.cv_deviance_mean))
        .map(|(i, _)| i).unwrap_or(0);
    dict.set_item("best_alpha", path_results[best_idx].alpha)?;
    dict.set_item("best_cv_deviance", path_results[best_idx].cv_deviance_mean)?;
    Ok(dict.into())
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
#[pyo3(signature = (y, x_full, smooth_col_ranges, smooth_penalties, family, link=None, offset=None, weights=None, max_iter=25, tol=1e-8, lambda_min=0.001, lambda_max=1000.0, smooth_monotonicity=None, store_design_matrix=false))]
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
) -> PyResult<PyObject> {
    let y_arr = y.as_array().to_owned();
    let x_arr = x_full.as_array().to_owned();
    let offset_arr = offset.map(|o| o.as_array().to_owned());
    let weights_arr = weights.map(|w| w.as_array().to_owned());
    
    let fam = family_from_name(family, 1.5, 1.0)?;
    let lnk = match link {
        Some(l) => link_from_name(l)?,
        None => link_from_name(default_link_name(family))?,
    };
    
    let n_terms = smooth_col_ranges.len();
    if smooth_penalties.len() != n_terms {
        return Err(PyValueError::new_err(format!(
            "smooth_col_ranges has {} entries but smooth_penalties has {}",
            n_terms, smooth_penalties.len()
        )));
    }
    
    let mono_vec = smooth_monotonicity.unwrap_or_else(|| vec![None; n_terms]);
    if mono_vec.len() != n_terms {
        return Err(PyValueError::new_err(format!(
            "smooth_monotonicity has {} entries but expected {}",
            mono_vec.len(), n_terms
        )));
    }
    
    let specs: Vec<SmoothTermSpec> = smooth_col_ranges.iter()
        .zip(smooth_penalties.iter())
        .zip(mono_vec.iter())
        .map(|((&(start, end), pen), mono)| {
            let monotonicity = match mono.as_deref() {
                Some("increasing") | Some("inc") => Monotonicity::Increasing,
                Some("decreasing") | Some("dec") => Monotonicity::Decreasing,
                _ => Monotonicity::None,
            };
            SmoothTermSpec {
                col_start: start,
                col_end: end,
                penalty: pen.as_array().to_owned(),
                monotonicity,
                initial_lambda: 1.0,
            }
        })
        .collect();
    
    let config = build_smooth_config(max_iter, tol, lambda_min, lambda_max);
    
    let result = fit_smooth_glm_full_matrix(
        &y_arr, &x_arr, &specs, fam.as_ref(), lnk.as_ref(), &config,
        offset_arr.as_ref(), weights_arr.as_ref(),
    ).map_err(|e| PyValueError::new_err(format!("Smooth GLM fitting failed: {}", e)))?;
    
    smooth_result_to_py(py, result, store_design_matrix)
}
