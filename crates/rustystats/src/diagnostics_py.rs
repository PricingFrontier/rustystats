// =============================================================================
// Diagnostics Bindings
// =============================================================================
//
// PyO3 wrappers for diagnostic computations: calibration curves,
// discrimination stats, A/E analysis, residuals, loss metrics, etc.
// =============================================================================

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use rustystats_core::diagnostics::{
    resid_pearson, resid_deviance, pearson_chi2,
    aic, bic, null_deviance,
    compute_calibration_curve, compute_discrimination_stats, compute_lorenz_curve,
    hosmer_lemeshow_test,
    compute_ae_continuous, compute_ae_categorical,
    compute_residual_pattern_continuous,
    detect_interactions, InteractionConfig, FactorData,
    mse, rmse, mae, compute_family_loss,
};

use crate::families_py::family_from_name;

/// Compute calibration curve bins from Rust
#[pyfunction]
#[pyo3(signature = (y, mu, exposure=None, n_bins=10))]
pub fn compute_calibration_curve_py<'py>(
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
    
    let result: PyResult<Vec<PyObject>> = bins.into_iter().map(|bin| {
        let dict = pyo3::types::PyDict::new_bound(py);
        dict.set_item("bin_index", bin.bin_index)?;
        dict.set_item("predicted_lower", bin.predicted_lower)?;
        dict.set_item("predicted_upper", bin.predicted_upper)?;
        dict.set_item("predicted_mean", bin.predicted_mean)?;
        dict.set_item("actual_mean", bin.actual_mean)?;
        dict.set_item("actual_expected_ratio", bin.actual_expected_ratio)?;
        dict.set_item("count", bin.count)?;
        dict.set_item("exposure", bin.exposure)?;
        dict.set_item("actual_sum", bin.actual_sum)?;
        dict.set_item("predicted_sum", bin.predicted_sum)?;
        dict.set_item("ae_ci_lower", bin.ae_ci_lower)?;
        dict.set_item("ae_ci_upper", bin.ae_ci_upper)?;
        Ok(dict.into_py(py))
    }).collect();
    
    result
}

/// Compute discrimination stats (Gini, AUC, etc.) from Rust
#[pyfunction]
#[pyo3(signature = (y, mu, exposure=None))]
pub fn compute_discrimination_stats_py<'py>(
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
pub fn compute_ae_continuous_py<'py>(
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
    
    let values_slice = values_arr.as_slice()
        .ok_or_else(|| PyValueError::new_err("Values array is not contiguous in memory"))?;
    
    let bins = compute_ae_continuous(
        values_slice,
        &y_arr,
        &mu_arr,
        exp_arr.as_ref(),
        family,
        n_bins,
        None,  // var_power
        None,  // theta
    );
    
    let result: PyResult<Vec<PyObject>> = bins.into_iter().map(|bin| {
        let dict = pyo3::types::PyDict::new_bound(py);
        dict.set_item("bin_index", bin.bin_index)?;
        dict.set_item("bin_label", &bin.bin_label)?;
        dict.set_item("bin_lower", bin.bin_lower)?;
        dict.set_item("bin_upper", bin.bin_upper)?;
        dict.set_item("count", bin.count)?;
        dict.set_item("exposure", bin.exposure)?;
        dict.set_item("actual_sum", bin.actual_sum)?;
        dict.set_item("predicted_sum", bin.predicted_sum)?;
        dict.set_item("actual_mean", bin.actual_mean)?;
        dict.set_item("predicted_mean", bin.predicted_mean)?;
        dict.set_item("actual_expected_ratio", bin.actual_expected_ratio)?;
        dict.set_item("loss", bin.loss)?;
        dict.set_item("ae_ci_lower", bin.ae_ci_lower)?;
        dict.set_item("ae_ci_upper", bin.ae_ci_upper)?;
        Ok(dict.into_py(py))
    }).collect();
    
    result
}

/// Compute A/E bins for categorical factor from Rust
#[pyfunction]
#[pyo3(signature = (levels, y, mu, exposure=None, rare_threshold_pct=1.0, max_levels=20, family="poisson"))]
pub fn compute_ae_categorical_py<'py>(
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
    
    let result: PyResult<Vec<PyObject>> = bins.into_iter().map(|bin| {
        let dict = pyo3::types::PyDict::new_bound(py);
        dict.set_item("bin_index", bin.bin_index)?;
        dict.set_item("bin_label", &bin.bin_label)?;
        dict.set_item("count", bin.count)?;
        dict.set_item("exposure", bin.exposure)?;
        dict.set_item("actual_sum", bin.actual_sum)?;
        dict.set_item("predicted_sum", bin.predicted_sum)?;
        dict.set_item("actual_mean", bin.actual_mean)?;
        dict.set_item("predicted_mean", bin.predicted_mean)?;
        dict.set_item("actual_expected_ratio", bin.actual_expected_ratio)?;
        dict.set_item("loss", bin.loss)?;
        dict.set_item("ae_ci_lower", bin.ae_ci_lower)?;
        dict.set_item("ae_ci_upper", bin.ae_ci_upper)?;
        Ok(dict.into_py(py))
    }).collect();
    
    result
}

/// Compute factor deviance breakdown from Rust (fast groupby)
#[pyfunction]
#[pyo3(signature = (factor_name, factor_values, y, mu, family="poisson", var_power=1.5, theta=1.0))]
pub fn compute_factor_deviance_py<'py>(
    py: Python<'py>,
    factor_name: &str,
    factor_values: Vec<String>,
    y: PyReadonlyArray1<f64>,
    mu: PyReadonlyArray1<f64>,
    family: &str,
    var_power: f64,
    theta: f64,
) -> PyResult<PyObject> {
    use rustystats_core::diagnostics::compute_factor_deviance;
    
    let y_arr = y.as_array().to_owned();
    let mu_arr = mu.as_array().to_owned();
    
    let result = compute_factor_deviance(
        factor_name,
        &factor_values,
        &y_arr,
        &mu_arr,
        family,
        var_power,
        theta,
    );
    
    // Convert levels to list of dicts
    let levels_list: Vec<PyObject> = result.levels.into_iter().map(|level| -> PyResult<PyObject> {
        let dict = pyo3::types::PyDict::new_bound(py);
        dict.set_item("level", &level.level)?;
        dict.set_item("count", level.count)?;
        dict.set_item("deviance", level.deviance)?;
        dict.set_item("deviance_pct", level.deviance_pct)?;
        dict.set_item("mean_deviance", level.mean_deviance)?;
        dict.set_item("actual_sum", level.actual_sum)?;
        dict.set_item("predicted_sum", level.predicted_sum)?;
        dict.set_item("ae_ratio", level.ae_ratio)?;
        dict.set_item("is_problem", level.is_problem)?;
        Ok(dict.into_py(py))
    }).collect::<PyResult<Vec<_>>>()?;
    
    let dict = pyo3::types::PyDict::new_bound(py);
    dict.set_item("factor_name", result.factor_name)?;
    dict.set_item("total_deviance", result.total_deviance)?;
    dict.set_item("levels", levels_list)?;
    dict.set_item("problem_levels", result.problem_levels)?;
    
    Ok(dict.into_py(py))
}

/// Compute loss metrics from Rust
#[pyfunction]
pub fn compute_loss_metrics_py<'py>(
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
pub fn detect_interactions_py<'py>(
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
            // Parse as f64 - fail loudly if values can't be parsed
            let floats: Result<Vec<f64>, _> = values.iter()
                .enumerate()
                .map(|(j, s)| s.parse::<f64>().map_err(|_| (j, s.clone())))
                .collect();
            let floats = match floats {
                Ok(f) => f,
                Err((idx, val)) => return Err(PyValueError::new_err(format!(
                    "Failed to parse value '{}' at index {} for continuous factor '{}' as a number",
                    val, idx, name
                ))),
            };
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
    
    let result: PyResult<Vec<PyObject>> = interactions.into_iter().map(|int| {
        let dict = pyo3::types::PyDict::new_bound(py);
        dict.set_item("factor1", &int.factor1)?;
        dict.set_item("factor2", &int.factor2)?;
        dict.set_item("strength", int.interaction_strength)?;
        dict.set_item("pvalue", int.pvalue)?;
        dict.set_item("n_cells", int.n_cells)?;
        Ok(dict.into_py(py))
    }).collect();
    
    result
}

/// Compute Lorenz curve from Rust
#[pyfunction]
#[pyo3(signature = (y, mu, exposure=None, n_points=20))]
pub fn compute_lorenz_curve_py<'py>(
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
    
    let result: PyResult<Vec<PyObject>> = points.into_iter().map(|p| {
        let dict = pyo3::types::PyDict::new_bound(py);
        dict.set_item("cumulative_exposure_pct", p.cumulative_exposure_pct)?;
        dict.set_item("cumulative_actual_pct", p.cumulative_actual_pct)?;
        dict.set_item("cumulative_predicted_pct", p.cumulative_predicted_pct)?;
        Ok(dict.into_py(py))
    }).collect();
    
    result
}

/// Compute Hosmer-Lemeshow test from Rust
#[pyfunction]
pub fn hosmer_lemeshow_test_py<'py>(
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
pub fn compute_fit_statistics_py<'py>(
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
    let fam = family_from_name(family, 1.5, 1.0)?;
    let pchi2 = pearson_chi2(&y_arr, &mu_arr, fam.as_ref(), None);
    
    // Compute log-likelihood based on family
    // Use estimated scale for gaussian
    let scale = if df_resid > 0 {
        y_arr.iter().zip(mu_arr.iter()).map(|(y, m)| (y - m).powi(2)).sum::<f64>() / df_resid as f64
    } else { 1.0 };
    
    // Use trait dispatch for log-likelihood (no string matching needed)
    let llf = fam.log_likelihood(&y_arr, &mu_arr, scale, None);
    
    let aic_val = aic(llf, n_params);
    let bic_val = bic(llf, n_params, n_obs);
    
    let deviance_explained = if null_dev > 0.0 { 1.0 - deviance / null_dev } else { 0.0 };
    let _dispersion_deviance = if df_resid > 0 { deviance / df_resid as f64 } else { 1.0 };
    let dispersion_pearson = if df_resid > 0 { pchi2 / df_resid as f64 } else { 1.0 };
    
    let dict = pyo3::types::PyDict::new_bound(py);
    dict.set_item("deviance", deviance)?;
    dict.set_item("null_deviance", null_dev)?;
    dict.set_item("deviance_explained", deviance_explained)?;
    dict.set_item("log_likelihood", llf)?;
    dict.set_item("aic", aic_val)?;
    dict.set_item("bic", bic_val)?;
    dict.set_item("pearson_chi2", pchi2)?;
    dict.set_item("dispersion", dispersion_pearson)?;  // primary dispersion metric
    
    Ok(dict.into_py(py))
}

/// Compute dataset metrics (deviance, log-likelihood, AIC) for any dataset
/// 
/// This is the same loss function used by GBMs (XGBoost, LightGBM):
/// - Poisson: 2 * sum(y * log(y/μ) - (y - μ))
/// - Gamma: 2 * sum((y - μ)/μ - log(y/μ))
/// - Gaussian: sum((y - μ)²)
/// - Binomial: -sum(y * log(μ) + (1-y) * log(1-μ))
/// 
/// Returns deviance (sum), mean_deviance (per-obs), log_likelihood, and AIC.
///
/// # Arguments
/// * `scale` - Dispersion parameter for Gamma/Gaussian. If None, estimated from deviance.
///             For Poisson/Binomial, scale is always 1 regardless of this parameter.
#[pyfunction]
#[pyo3(signature = (y, mu, family, n_params, var_power=1.5, theta=1.0, scale=None))]
pub fn compute_dataset_metrics_py<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<f64>,
    mu: PyReadonlyArray1<f64>,
    family: &str,
    n_params: usize,
    var_power: f64,
    theta: f64,
    scale: Option<f64>,
) -> PyResult<PyObject> {
    use rustystats_core::diagnostics::loss::{
        poisson_deviance_loss, gamma_deviance_loss, mse, log_loss,
        tweedie_deviance_loss, negbinomial_deviance_loss,
    };
    
    let y_arr = y.as_array().to_owned();
    let mu_arr = mu.as_array().to_owned();
    let n_obs = y_arr.len();
    
    if n_obs == 0 {
        return Err(PyValueError::new_err("Empty arrays"));
    }
    
    let family_lower = family.to_lowercase();
    
    // Parse theta from family string if present (e.g., "negativebinomial(theta=1.38)")
    let parsed_theta = if family_lower.starts_with("negativebinomial") || family_lower.starts_with("negbinomial") {
        if let Some(start) = family_lower.find("theta=") {
            let rest = &family_lower[start + 6..];
            let end = rest.find(')').unwrap_or(rest.len());
            let theta_str = &rest[..end];
            theta_str.parse::<f64>().map_err(|_| {
                PyValueError::new_err(format!(
                    "Failed to parse theta value '{}' in family '{}'. Expected a numeric value.",
                    theta_str, family
                ))
            })?
        } else {
            theta
        }
    } else {
        theta
    };
    
    // Parse var_power from family string if present (e.g., "tweedie(p=1.5)")
    let parsed_var_power = if family_lower.starts_with("tweedie") {
        if let Some(start) = family_lower.find("p=") {
            let rest = &family_lower[start + 2..];
            let end = rest.find(')').unwrap_or(rest.len());
            let p_str = &rest[..end];
            p_str.parse::<f64>().map_err(|_| {
                PyValueError::new_err(format!(
                    "Failed to parse var_power value '{}' in family '{}'. Expected a numeric value.",
                    p_str, family
                ))
            })?
        } else {
            var_power
        }
    } else {
        var_power
    };
    
    // Compute mean deviance loss (this is the GBM loss function)
    let mean_deviance = if family_lower.starts_with("negativebinomial") || family_lower.starts_with("negbinomial") {
        negbinomial_deviance_loss(&y_arr, &mu_arr, parsed_theta, None)
    } else if family_lower.starts_with("tweedie") {
        tweedie_deviance_loss(&y_arr, &mu_arr, parsed_var_power, None)
    } else {
        match family_lower.as_str() {
            "gaussian" | "normal" => mse(&y_arr, &mu_arr, None),
            "poisson" | "quasipoisson" => poisson_deviance_loss(&y_arr, &mu_arr, None),
            "gamma" => gamma_deviance_loss(&y_arr, &mu_arr, None),
            "binomial" | "quasibinomial" => log_loss(&y_arr, &mu_arr, None),
            _ => return Err(PyValueError::new_err(format!("Unknown family: {}", family))),
        }
    };
    
    // Total deviance (sum, not mean)
    let deviance = mean_deviance * n_obs as f64;
    
    // Compute scale (dispersion) for log-likelihood calculation
    // For Gamma/Gaussian: use provided scale or estimate from deviance/(n-p)
    // For Poisson/Binomial: scale is always 1 by definition
    let df_resid = if n_obs > n_params { n_obs - n_params } else { 1 };
    let estimated_scale = deviance / df_resid as f64;
    
    // Use trait dispatch for scale and log-likelihood
    let fam = family_from_name(family, 1.5, 1.0)?;
    let effective_scale = if fam.fixed_dispersion() {
        1.0
    } else {
        scale.unwrap_or(estimated_scale)
    };
    
    let llf = fam.log_likelihood(&y_arr, &mu_arr, effective_scale, None);
    
    // AIC = -2 * LL + 2 * k
    let aic_val = aic(llf, n_params);
    
    let dict = pyo3::types::PyDict::new_bound(py);
    dict.set_item("deviance", deviance)?;
    dict.set_item("mean_deviance", mean_deviance)?;
    dict.set_item("log_likelihood", llf)?;
    dict.set_item("aic", aic_val)?;
    dict.set_item("n_obs", n_obs)?;
    dict.set_item("scale", effective_scale)?;
    
    Ok(dict.into_py(py))
}

/// Compute residual summary statistics from Rust
#[pyfunction]
pub fn compute_residual_summary_py<'py>(
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
    
    // Percentiles - use total_cmp to handle NaN values properly
    let mut sorted: Vec<f64> = resid.iter().cloned().collect();
    sorted.sort_by(|a, b| a.total_cmp(b));
    
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
pub fn compute_residual_pattern_py<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<f64>,
    residuals: PyReadonlyArray1<f64>,
    n_bins: usize,
) -> PyResult<PyObject> {
    let values_arr = values.as_array().to_owned();
    let resid_arr = residuals.as_array().to_owned();
    
    let values_slice = values_arr.as_slice()
        .ok_or_else(|| PyValueError::new_err("Values array is not contiguous in memory"))?;
    
    let pattern = compute_residual_pattern_continuous(
        values_slice,
        &resid_arr,
        n_bins,
    );
    
    let dict = pyo3::types::PyDict::new_bound(py);
    dict.set_item("correlation_with_residuals", pattern.correlation_with_residuals)?;
    
    let means: Vec<PyObject> = pattern.mean_residual_by_bin.into_iter().enumerate().map(|(i, m)| -> PyResult<PyObject> {
        let d = pyo3::types::PyDict::new_bound(py);
        d.set_item("bin_index", i)?;
        d.set_item("mean_residual", m)?;
        Ok(d.into_py(py))
    }).collect::<PyResult<Vec<_>>>()?;
    
    dict.set_item("mean_residual_by_bin", means)?;
    
    Ok(dict.into_py(py))
}

/// Compute Pearson residuals from Rust
#[pyfunction]
pub fn compute_pearson_residuals_py<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<f64>,
    mu: PyReadonlyArray1<f64>,
    family: &str,
) -> PyResult<Py<PyArray1<f64>>> {
    let y_arr = y.as_array().to_owned();
    let mu_arr = mu.as_array().to_owned();
    let fam = family_from_name(family, 1.5, 1.0)?;
    let resid = resid_pearson(&y_arr, &mu_arr, fam.as_ref());
    Ok(resid.into_pyarray_bound(py).unbind())
}

/// Compute deviance residuals from Rust
#[pyfunction]
pub fn compute_deviance_residuals_py<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<f64>,
    mu: PyReadonlyArray1<f64>,
    family: &str,
) -> PyResult<Py<PyArray1<f64>>> {
    let y_arr = y.as_array().to_owned();
    let mu_arr = mu.as_array().to_owned();
    let fam = family_from_name(family, 1.5, 1.0)?;
    let resid = resid_deviance(&y_arr, &mu_arr, fam.as_ref());
    Ok(resid.into_pyarray_bound(py).unbind())
}

/// Compute null deviance from Rust
#[pyfunction]
#[pyo3(signature = (y, family, exposure=None))]
pub fn compute_null_deviance_py(
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
pub fn compute_unit_deviance_py<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<f64>,
    mu: PyReadonlyArray1<f64>,
    family: &str,
) -> PyResult<Py<PyArray1<f64>>> {
    let y_arr = y.as_array().to_owned();
    let mu_arr = mu.as_array().to_owned();
    let fam = family_from_name(family, 1.5, 1.0)?;
    let unit_dev = fam.unit_deviance(&y_arr, &mu_arr);
    Ok(unit_dev.into_pyarray_bound(py).unbind())
}
