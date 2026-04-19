// =============================================================================
// Diagnostics Bindings
// =============================================================================
//
// PyO3 wrappers for diagnostic computations: calibration curves,
// discrimination stats, A/E analysis, residuals, loss metrics, etc.
// =============================================================================

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rustystats_core::diagnostics::{
    aic, compute_ae_by_decile, compute_ae_categorical, compute_ae_categorical_batch,
    compute_ae_continuous, compute_ae_continuous_batch, compute_calibration_curve,
    compute_discrimination_stats, compute_factor_significance_batch, compute_family_loss,
    compute_lorenz_curve, compute_residual_pattern_continuous,
    compute_residual_pattern_continuous_batch, detect_interactions, hosmer_lemeshow_test,
    inverse_diagonal_spd, mae, mse, null_deviance, partial_dependence_categorical_batch,
    resid_deviance, resid_pearson, rmse, ActualExpectedBin, DevianceByLevel, FactorData,
    FactorDevianceResult, InteractionConfig, ResidualPattern,
};

use crate::families_py::family_from_name;

// =============================================================================
// Dict construction helpers
// =============================================================================
//
// These helpers consolidate the repeated `let dict = PyDict::new(py); dict.set_item(...)? ...`
// patterns used to marshal Rust diagnostic structs into Python dicts. Each
// helper builds one dict shape; call sites then become a single function call
// per item.

/// Build a dict for an `ActualExpectedBin` produced by a continuous A/E call.
/// Includes the continuous-only `bin_lower` / `bin_upper` fields.
fn ae_continuous_bin_to_dict<'py>(py: Python<'py>, bin: &ActualExpectedBin) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
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
    Ok(dict.unbind().into())
}

/// Build a dict for an `ActualExpectedBin` produced by a categorical A/E call.
/// Drops the continuous-only `bin_lower` / `bin_upper` fields.
fn ae_categorical_bin_to_dict<'py>(
    py: Python<'py>,
    bin: &ActualExpectedBin,
) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
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
    Ok(dict.unbind().into())
}

/// Build a dict for one `DevianceByLevel` row in a factor-deviance breakdown.
fn deviance_level_to_dict<'py>(py: Python<'py>, level: &DevianceByLevel) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item("level", &level.level)?;
    dict.set_item("count", level.count)?;
    dict.set_item("deviance", level.deviance)?;
    dict.set_item("deviance_pct", level.deviance_pct)?;
    dict.set_item("mean_deviance", level.mean_deviance)?;
    dict.set_item("actual_sum", level.actual_sum)?;
    dict.set_item("predicted_sum", level.predicted_sum)?;
    dict.set_item("ae_ratio", level.ae_ratio)?;
    dict.set_item("is_problem", level.is_problem)?;
    Ok(dict.unbind().into())
}

/// Build a dict for a full `FactorDevianceResult` (factor name + level rows).
fn factor_deviance_result_to_dict<'py>(
    py: Python<'py>,
    result: FactorDevianceResult,
) -> PyResult<Py<PyAny>> {
    let levels_list: Vec<Py<PyAny>> = result
        .levels
        .iter()
        .map(|level| deviance_level_to_dict(py, level))
        .collect::<PyResult<Vec<_>>>()?;

    let dict = PyDict::new(py);
    dict.set_item("factor_name", result.factor_name)?;
    dict.set_item("total_deviance", result.total_deviance)?;
    dict.set_item("levels", levels_list)?;
    dict.set_item("problem_levels", result.problem_levels)?;
    Ok(dict.unbind().into())
}

/// Build a dict for a `ResidualPattern` (correlation + per-bin mean residuals).
fn residual_pattern_to_dict<'py>(py: Python<'py>, pattern: ResidualPattern) -> PyResult<Py<PyAny>> {
    let means: Vec<Py<PyAny>> = pattern
        .mean_residual_by_bin
        .into_iter()
        .enumerate()
        .map(|(i, m)| -> PyResult<Py<PyAny>> {
            let d = PyDict::new(py);
            d.set_item("bin_index", i)?;
            d.set_item("mean_residual", m)?;
            Ok(d.unbind().into())
        })
        .collect::<PyResult<Vec<_>>>()?;

    let dict = PyDict::new(py);
    dict.set_item(
        "correlation_with_residuals",
        pattern.correlation_with_residuals,
    )?;
    dict.set_item("mean_residual_by_bin", means)?;
    Ok(dict.unbind().into())
}

/// Compute calibration curve bins from Rust
#[pyfunction]
#[pyo3(signature = (y, mu, exposure=None, n_bins=10))]
pub fn compute_calibration_curve_py<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<f64>,
    mu: PyReadonlyArray1<f64>,
    exposure: Option<PyReadonlyArray1<f64>>,
    n_bins: usize,
) -> PyResult<Vec<Py<PyAny>>> {
    let y_arr = y.as_array().to_owned();
    let mu_arr = mu.as_array().to_owned();
    let exp_arr = exposure.map(|e| e.as_array().to_owned());

    let bins = compute_calibration_curve(&y_arr, &mu_arr, exp_arr.as_ref(), n_bins);

    let result: PyResult<Vec<Py<PyAny>>> = bins
        .into_iter()
        .map(|bin| {
            let dict = pyo3::types::PyDict::new(py);
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
            Ok(dict.unbind().into())
        })
        .collect();

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
) -> PyResult<Py<PyAny>> {
    let y_arr = y.as_array().to_owned();
    let mu_arr = mu.as_array().to_owned();
    let exp_arr = exposure.map(|e| e.as_array().to_owned());

    let stats = compute_discrimination_stats(&y_arr, &mu_arr, exp_arr.as_ref());

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("gini", stats.gini_coefficient)?;
    dict.set_item("auc", stats.auc)?;
    dict.set_item("ks_statistic", stats.ks_statistic)?;
    dict.set_item("lift_at_10pct", stats.lift_at_10pct)?;
    dict.set_item("lift_at_20pct", stats.lift_at_20pct)?;

    Ok(dict.unbind().into())
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
) -> PyResult<Vec<Py<PyAny>>> {
    let values_arr = values.as_array().to_owned();
    let y_arr = y.as_array().to_owned();
    let mu_arr = mu.as_array().to_owned();
    let exp_arr = exposure.map(|e| e.as_array().to_owned());

    let values_slice = values_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("Values array is not contiguous in memory"))?;

    let bins = compute_ae_continuous(
        values_slice,
        &y_arr,
        &mu_arr,
        exp_arr.as_ref(),
        family,
        n_bins,
        None, // var_power
        None, // theta
    );

    bins.iter()
        .map(|bin| ae_continuous_bin_to_dict(py, bin))
        .collect()
}

/// Compute A/E bins for many continuous factors at once, parallelized over factors.
///
/// `values_matrix` has shape (n, k) where each column is the per-row values for
/// one factor. Returns a list of length k, where each element is the same list
/// of bin dicts produced by `compute_ae_continuous_py`.
#[pyfunction]
#[pyo3(signature = (values_matrix, y, mu, exposure=None, n_bins=10, family="poisson"))]
pub fn compute_ae_continuous_batch_py<'py>(
    py: Python<'py>,
    values_matrix: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<f64>,
    mu: PyReadonlyArray1<f64>,
    exposure: Option<PyReadonlyArray1<f64>>,
    n_bins: usize,
    family: &str,
) -> PyResult<Vec<Vec<Py<PyAny>>>> {
    let values_view = values_matrix.as_array();
    let y_arr = y.as_array().to_owned();
    let mu_arr = mu.as_array().to_owned();
    let exp_arr = exposure.map(|e| e.as_array().to_owned());

    let n = values_view.nrows();
    let k = values_view.ncols();

    if y_arr.len() != n {
        return Err(PyValueError::new_err(format!(
            "values_matrix has {} rows but y has {} elements",
            n,
            y_arr.len()
        )));
    }

    // Materialize each column into an owned Vec<f64> so the slices we hand to
    // Rust are contiguous regardless of the input matrix's memory order.
    let columns: Vec<Vec<f64>> = (0..k).map(|j| values_view.column(j).to_vec()).collect();
    let column_slices: Vec<&[f64]> = columns.iter().map(|c| c.as_slice()).collect();

    // Release the GIL for the parallel Rust work, then re-enter to build the
    // Python dicts.
    let bin_results: Vec<Vec<rustystats_core::diagnostics::ActualExpectedBin>> = py.detach(|| {
        compute_ae_continuous_batch(
            &column_slices,
            &y_arr,
            &mu_arr,
            exp_arr.as_ref(),
            family,
            n_bins,
            None,
            None,
        )
    });

    let mut out: Vec<Vec<Py<PyAny>>> = Vec::with_capacity(bin_results.len());
    for bins in bin_results.iter() {
        let dicts: PyResult<Vec<Py<PyAny>>> = bins
            .iter()
            .map(|bin| ae_continuous_bin_to_dict(py, bin))
            .collect();
        out.push(dicts?);
    }
    Ok(out)
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
) -> PyResult<Vec<Py<PyAny>>> {
    let y_arr = y.as_array().to_owned();
    let mu_arr = mu.as_array().to_owned();
    let exp_arr = exposure.map(|e| e.as_array().to_owned());

    let bins = compute_ae_categorical(
        &levels,
        &y_arr,
        &mu_arr,
        exp_arr.as_ref(),
        family,
        None, // var_power
        None, // theta
        rare_threshold_pct,
        max_levels,
    );

    bins.iter()
        .map(|bin| ae_categorical_bin_to_dict(py, bin))
        .collect()
}

/// Compute A/E bins for many categorical factors at once, parallelized over factors.
///
/// `codes_matrix` has shape (n, k) where each column is a u32 code for one
/// factor (codes index into that factor's `levels_list[j]`). Returns a list
/// of length k, each element being the same list of bin dicts as
/// `compute_ae_categorical_py`.
///
/// This avoids the per-row string marshalling that dominates the singular
/// categorical path when n is large; all categorical A/E work is now a single
/// cross-FFI call into a rayon-parallel Rust loop.
#[pyfunction]
#[pyo3(signature = (codes_matrix, levels_list, y, mu, exposure=None, rare_threshold_pct=1.0, max_levels=20, family="poisson"))]
pub fn compute_ae_categorical_batch_py<'py>(
    py: Python<'py>,
    codes_matrix: PyReadonlyArray2<'py, u32>,
    levels_list: Vec<Vec<String>>,
    y: PyReadonlyArray1<f64>,
    mu: PyReadonlyArray1<f64>,
    exposure: Option<PyReadonlyArray1<f64>>,
    rare_threshold_pct: f64,
    max_levels: usize,
    family: &str,
) -> PyResult<Vec<Vec<Py<PyAny>>>> {
    let codes_view = codes_matrix.as_array();
    let y_arr = y.as_array().to_owned();
    let mu_arr = mu.as_array().to_owned();
    let exp_arr = exposure.map(|e| e.as_array().to_owned());

    let n = codes_view.nrows();
    let k = codes_view.ncols();

    if y_arr.len() != n {
        return Err(PyValueError::new_err(format!(
            "codes_matrix has {} rows but y has {} elements",
            n,
            y_arr.len()
        )));
    }

    if levels_list.len() != k {
        return Err(PyValueError::new_err(format!(
            "codes_matrix has {} columns but levels_list has {} entries",
            k,
            levels_list.len()
        )));
    }

    // Materialize each column as a contiguous Vec<u32> so we can hand slices
    // to Rust regardless of the input matrix's memory order.
    let code_columns: Vec<Vec<u32>> = (0..k).map(|j| codes_view.column(j).to_vec()).collect();
    let code_slices: Vec<&[u32]> = code_columns.iter().map(|c| c.as_slice()).collect();
    let level_slices: Vec<&[String]> = levels_list.iter().map(|l| l.as_slice()).collect();

    // Release the GIL for the parallel Rust work, then re-enter to build the
    // Python dicts.
    let bin_results: Vec<Vec<rustystats_core::diagnostics::ActualExpectedBin>> = py.detach(|| {
        compute_ae_categorical_batch(
            &code_slices,
            &level_slices,
            &y_arr,
            &mu_arr,
            exp_arr.as_ref(),
            family,
            None, // var_power
            None, // theta
            rare_threshold_pct,
            max_levels,
        )
    });

    let mut out: Vec<Vec<Py<PyAny>>> = Vec::with_capacity(bin_results.len());
    for bins in bin_results.iter() {
        let dicts: PyResult<Vec<Py<PyAny>>> = bins
            .iter()
            .map(|bin| ae_categorical_bin_to_dict(py, bin))
            .collect();
        out.push(dicts?);
    }
    Ok(out)
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
) -> PyResult<Py<PyAny>> {
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

    factor_deviance_result_to_dict(py, result)
}

/// Compute factor deviance breakdown for many categorical factors at once,
/// parallelized over factors.
///
/// `factor_names[i]` is paired with `factor_values_list[i]` (a `Vec<String>`
/// of length n with each row's level label for factor i). Shared `y`, `mu`,
/// `family`, `var_power`, and `theta` are reused. Returns a list of length k
/// where each element is the same dict produced by `compute_factor_deviance_py`.
///
/// This collapses k sequential cross-FFI calls into one, and runs the
/// per-factor HashMap groupby work in parallel via rayon. Unit deviances are
/// computed once and shared across all factors.
#[pyfunction]
#[pyo3(signature = (factor_names, factor_values_list, y, mu, family="poisson", var_power=1.5, theta=1.0))]
pub fn compute_factor_deviance_batch_py<'py>(
    py: Python<'py>,
    factor_names: Vec<String>,
    factor_values_list: Vec<Vec<String>>,
    y: PyReadonlyArray1<'py, f64>,
    mu: PyReadonlyArray1<'py, f64>,
    family: &str,
    var_power: f64,
    theta: f64,
) -> PyResult<Vec<Py<PyAny>>> {
    use rustystats_core::diagnostics::compute_factor_deviance_batch;

    if factor_names.len() != factor_values_list.len() {
        return Err(PyValueError::new_err(format!(
            "factor_names has {} entries but factor_values_list has {} entries",
            factor_names.len(),
            factor_values_list.len()
        )));
    }

    let y_arr = y.as_array().to_owned();
    let mu_arr = mu.as_array().to_owned();

    // Build the &[&[String]] view so the Rust batch entry can borrow each
    // factor's values without copying.
    let value_slices: Vec<&[String]> = factor_values_list.iter().map(|v| v.as_slice()).collect();

    // Release the GIL for the parallel Rust work, then re-enter to build the
    // Python dicts.
    let batch_results = py.detach(|| {
        compute_factor_deviance_batch(
            &factor_names,
            &value_slices,
            &y_arr,
            &mu_arr,
            family,
            var_power,
            theta,
        )
    });

    let mut out: Vec<Py<PyAny>> = Vec::with_capacity(batch_results.len());
    for result in batch_results.into_iter() {
        out.push(factor_deviance_result_to_dict(py, result)?);
    }

    Ok(out)
}

/// Code-based variant of `compute_factor_deviance_batch_py`.
///
/// `codes_matrix` has shape (n, k) where each column is a u32 code for one
/// factor (codes index into that factor's `levels_list[j]`). This avoids the
/// per-row string marshalling that dominates the wall-clock for large n. The
/// `cat_unique_cache` already holds these codes, so this is the natural
/// integration point on the Python side.
#[pyfunction]
#[pyo3(signature = (factor_names, codes_matrix, levels_list, y, mu, family="poisson", var_power=1.5, theta=1.0))]
pub fn compute_factor_deviance_batch_from_codes_py<'py>(
    py: Python<'py>,
    factor_names: Vec<String>,
    codes_matrix: PyReadonlyArray2<'py, u32>,
    levels_list: Vec<Vec<String>>,
    y: PyReadonlyArray1<'py, f64>,
    mu: PyReadonlyArray1<'py, f64>,
    family: &str,
    var_power: f64,
    theta: f64,
) -> PyResult<Vec<Py<PyAny>>> {
    use rustystats_core::diagnostics::compute_factor_deviance_batch_from_codes;

    let codes_view = codes_matrix.as_array();
    let y_arr = y.as_array().to_owned();
    let mu_arr = mu.as_array().to_owned();

    let n = codes_view.nrows();
    let k = codes_view.ncols();

    if y_arr.len() != n {
        return Err(PyValueError::new_err(format!(
            "codes_matrix has {} rows but y has {} elements",
            n,
            y_arr.len()
        )));
    }

    if factor_names.len() != k || levels_list.len() != k {
        return Err(PyValueError::new_err(format!(
            "codes_matrix has {} columns but factor_names has {} entries and \
             levels_list has {} entries",
            k,
            factor_names.len(),
            levels_list.len()
        )));
    }

    // Materialize each column as a contiguous Vec<u32> so we can hand slices
    // to Rust regardless of the input matrix's memory order.
    let code_columns: Vec<Vec<u32>> = (0..k).map(|j| codes_view.column(j).to_vec()).collect();
    let code_slices: Vec<&[u32]> = code_columns.iter().map(|c| c.as_slice()).collect();
    let level_slices: Vec<&[String]> = levels_list.iter().map(|l| l.as_slice()).collect();

    let batch_results = py.detach(|| {
        compute_factor_deviance_batch_from_codes(
            &factor_names,
            &code_slices,
            &level_slices,
            &y_arr,
            &mu_arr,
            family,
            var_power,
            theta,
        )
    });

    let mut out: Vec<Py<PyAny>> = Vec::with_capacity(batch_results.len());
    for result in batch_results.into_iter() {
        out.push(factor_deviance_result_to_dict(py, result)?);
    }

    Ok(out)
}

/// Compute loss metrics from Rust
#[pyfunction]
pub fn compute_loss_metrics_py<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<f64>,
    mu: PyReadonlyArray1<f64>,
    family: &str,
) -> PyResult<Py<PyAny>> {
    let y_arr = y.as_array().to_owned();
    let mu_arr = mu.as_array().to_owned();

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("mse", mse(&y_arr, &mu_arr, None))?;
    dict.set_item("rmse", rmse(&y_arr, &mu_arr, None))?;
    dict.set_item("mae", mae(&y_arr, &mu_arr, None))?;
    dict.set_item(
        "family_loss",
        compute_family_loss(family, &y_arr, &mu_arr, None, None, None)
            .map_err(pyo3::exceptions::PyValueError::new_err)?,
    )?;

    Ok(dict.unbind().into())
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
) -> PyResult<Vec<Py<PyAny>>> {
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
            let floats: Result<Vec<f64>, _> = values
                .iter()
                .enumerate()
                .map(|(j, s)| s.parse::<f64>().map_err(|_| (j, s.clone())))
                .collect();
            let floats = match floats {
                Ok(f) => f,
                Err((idx, val)) => {
                    return Err(PyValueError::new_err(format!(
                    "Failed to parse value '{}' at index {} for continuous factor '{}' as a number",
                    val, idx, name
                )))
                }
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

    let result: PyResult<Vec<Py<PyAny>>> = interactions
        .into_iter()
        .map(|int| {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("factor1", &int.factor1)?;
            dict.set_item("factor2", &int.factor2)?;
            dict.set_item("strength", int.interaction_strength)?;
            dict.set_item("pvalue", int.pvalue)?;
            dict.set_item("n_cells", int.n_cells)?;
            Ok(dict.unbind().into())
        })
        .collect();

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
) -> PyResult<Vec<Py<PyAny>>> {
    let y_arr = y.as_array().to_owned();
    let mu_arr = mu.as_array().to_owned();
    let exp_arr = exposure.map(|e| e.as_array().to_owned());

    let points = compute_lorenz_curve(&y_arr, &mu_arr, exp_arr.as_ref(), n_points);

    let result: PyResult<Vec<Py<PyAny>>> = points
        .into_iter()
        .map(|p| {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("cumulative_exposure_pct", p.cumulative_exposure_pct)?;
            dict.set_item("cumulative_actual_pct", p.cumulative_actual_pct)?;
            dict.set_item("cumulative_predicted_pct", p.cumulative_predicted_pct)?;
            Ok(dict.unbind().into())
        })
        .collect();

    result
}

/// Compute Hosmer-Lemeshow test from Rust
#[pyfunction]
pub fn hosmer_lemeshow_test_py<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<f64>,
    mu: PyReadonlyArray1<f64>,
    n_bins: usize,
) -> PyResult<Py<PyAny>> {
    let y_arr = y.as_array().to_owned();
    let mu_arr = mu.as_array().to_owned();

    let result = hosmer_lemeshow_test(&y_arr, &mu_arr, n_bins);

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("chi2_statistic", result.statistic)?;
    dict.set_item("pvalue", result.pvalue)?;
    dict.set_item("degrees_of_freedom", result.degrees_of_freedom)?;

    Ok(dict.unbind().into())
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
) -> PyResult<Py<PyAny>> {
    use rustystats_core::diagnostics::compute_fit_statistics;

    let y_arr = y.as_array().to_owned();
    let mu_arr = mu.as_array().to_owned();
    let fam = family_from_name(family, 1.5, 1.0)?;

    let stats = py.detach(|| {
        compute_fit_statistics(&y_arr, &mu_arr, fam.as_ref(), deviance, null_dev, n_params)
    });

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("deviance", stats.deviance)?;
    dict.set_item("null_deviance", stats.null_deviance)?;
    dict.set_item("deviance_explained", stats.deviance_explained)?;
    dict.set_item("log_likelihood", stats.log_likelihood)?;
    dict.set_item("aic", stats.aic)?;
    dict.set_item("bic", stats.bic)?;
    dict.set_item("pearson_chi2", stats.pearson_chi2)?;
    dict.set_item("dispersion", stats.dispersion)?; // primary dispersion metric

    Ok(dict.unbind().into())
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
) -> PyResult<Py<PyAny>> {
    use rustystats_core::diagnostics::compute_glm_deviance;
    use rustystats_core::diagnostics::loss::{
        gamma_deviance_loss, log_loss, mse, negbinomial_deviance_loss, poisson_deviance_loss,
        tweedie_deviance_loss,
    };
    use rustystats_core::diagnostics::parse_family_params;

    let y_arr = y.as_array().to_owned();
    let mu_arr = mu.as_array().to_owned();
    let n_obs = y_arr.len();

    if n_obs == 0 {
        return Err(PyValueError::new_err("Empty arrays"));
    }

    let family_lower = family.to_lowercase();

    // Extract embedded theta / var_power from family strings like
    // "negativebinomial(theta=1.38)" or "tweedie(p=1.5)" via the core helper.
    let parsed_params =
        parse_family_params(family, var_power, theta).map_err(PyValueError::new_err)?;
    let parsed_theta = parsed_params.theta;
    let parsed_var_power = parsed_params.var_power;

    // Compute mean deviance loss (this is the GBM loss function)
    let mean_deviance = if family_lower.starts_with("negativebinomial")
        || family_lower.starts_with("negbinomial")
    {
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

    // Textbook GLM deviance: sum_i unit_deviance(y_i, mu_i; family).
    //
    // For most families (Gaussian/Poisson/Gamma/Tweedie/NegBin), the family loss
    // already coincides with the unit deviance (or unit_deviance / n times n), so
    // `mean_loss * n_obs` would happen to be correct. But for Binomial the family
    // loss is NLL `mean(-(y log mu + (1-y) log(1-mu)))`, which differs from the
    // textbook unit deviance `2 (y log(y/mu) + (1-y) log((1-y)/(1-mu)))` by an
    // additive (saturated-model) constant. Compute the proper deviance directly
    // so `train_test.train.deviance` matches `result.deviance` and statsmodels.
    let deviance = compute_glm_deviance(&y_arr, &mu_arr, family, parsed_var_power, parsed_theta);

    // Compute scale (dispersion) for log-likelihood calculation
    // For Gamma/Gaussian: use provided scale or estimate from deviance/(n-p)
    // For Poisson/Binomial: scale is always 1 by definition
    let df_resid = if n_obs > n_params {
        n_obs - n_params
    } else {
        1
    };
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

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("deviance", deviance)?;
    dict.set_item("mean_deviance", mean_deviance)?;
    dict.set_item("log_likelihood", llf)?;
    dict.set_item("aic", aic_val)?;
    dict.set_item("n_obs", n_obs)?;
    dict.set_item("scale", effective_scale)?;

    Ok(dict.unbind().into())
}

/// Compute residual summary statistics from Rust
#[pyfunction]
pub fn compute_residual_summary_py<'py>(
    py: Python<'py>,
    residuals: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyAny>> {
    use rustystats_core::diagnostics::compute_residual_summary;

    let resid_arr = residuals.as_array().to_owned();

    let summary = py
        .detach(|| compute_residual_summary(&resid_arr))
        .ok_or_else(|| PyValueError::new_err("Empty residuals array"))?;

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("mean", summary.mean)?;
    dict.set_item("std", summary.std)?;
    dict.set_item("min", summary.min)?;
    dict.set_item("max", summary.max)?;
    dict.set_item("skewness", summary.skewness)?;
    dict.set_item("kurtosis", summary.kurtosis)?;
    dict.set_item("p1", summary.p1)?;
    dict.set_item("p5", summary.p5)?;
    dict.set_item("p10", summary.p10)?;
    dict.set_item("p25", summary.p25)?;
    dict.set_item("p50", summary.p50)?;
    dict.set_item("p75", summary.p75)?;
    dict.set_item("p90", summary.p90)?;
    dict.set_item("p95", summary.p95)?;
    dict.set_item("p99", summary.p99)?;

    Ok(dict.unbind().into())
}

/// Compute residual pattern for continuous factor from Rust
#[pyfunction]
#[pyo3(signature = (values, residuals, n_bins=10))]
pub fn compute_residual_pattern_py<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<f64>,
    residuals: PyReadonlyArray1<f64>,
    n_bins: usize,
) -> PyResult<Py<PyAny>> {
    let values_arr = values.as_array().to_owned();
    let resid_arr = residuals.as_array().to_owned();

    let values_slice = values_arr
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("Values array is not contiguous in memory"))?;

    let pattern = compute_residual_pattern_continuous(values_slice, &resid_arr, n_bins);

    residual_pattern_to_dict(py, pattern)
}

/// Compute residual patterns for many continuous factors at once, parallelized.
///
/// `values_matrix` has shape (n, k) where each column is the per-row values for
/// one factor. Returns a list of length k, each element being the same dict
/// produced by `compute_residual_pattern_py`.
#[pyfunction]
#[pyo3(signature = (values_matrix, residuals, n_bins=10))]
pub fn compute_residual_pattern_batch_py<'py>(
    py: Python<'py>,
    values_matrix: PyReadonlyArray2<'py, f64>,
    residuals: PyReadonlyArray1<f64>,
    n_bins: usize,
) -> PyResult<Vec<Py<PyAny>>> {
    let values_view = values_matrix.as_array();
    let resid_arr = residuals.as_array().to_owned();

    let n = values_view.nrows();
    let k = values_view.ncols();

    if resid_arr.len() != n {
        return Err(PyValueError::new_err(format!(
            "values_matrix has {} rows but residuals has {} elements",
            n,
            resid_arr.len()
        )));
    }

    let columns: Vec<Vec<f64>> = (0..k).map(|j| values_view.column(j).to_vec()).collect();
    let column_slices: Vec<&[f64]> = columns.iter().map(|c| c.as_slice()).collect();

    let patterns =
        py.detach(|| compute_residual_pattern_continuous_batch(&column_slices, &resid_arr, n_bins));

    let mut out: Vec<Py<PyAny>> = Vec::with_capacity(patterns.len());
    for pattern in patterns.into_iter() {
        out.push(residual_pattern_to_dict(py, pattern)?);
    }
    Ok(out)
}

/// Batch-compute factor significance for k factors in parallel.
///
/// `param_indices_per_factor[i]` is the list of parameter positions for factor
/// i (in the order they appear in `feature_names`). Each factor's joint Wald
/// chi-square is computed using the bread matrix sub-block, with the scale
/// inferred from the supplied `bse` vector.
///
/// Returns a list of length k. Each element is either `None` (no parameter
/// indices, or the covariance sub-block was not invertible) or a dict
/// `{chi2, pvalue, df}` matching the singular `compute_factor_significance`
/// numerics.
#[pyfunction]
pub fn compute_factor_significance_batch_py<'py>(
    py: Python<'py>,
    param_indices_per_factor: Vec<Vec<usize>>,
    params: PyReadonlyArray1<'py, f64>,
    bse: PyReadonlyArray1<'py, f64>,
    bread: PyReadonlyArray2<'py, f64>,
) -> PyResult<Vec<Option<Py<PyAny>>>> {
    // Materialize ndarray views into owned types so we can release the GIL for
    // the rayon-parallel work without borrowing from Python objects.
    let params_vec = params.as_array().to_vec();
    let bse_vec = bse.as_array().to_vec();
    let bread_arr = bread.as_array().to_owned();

    let raw_results = py.detach(|| {
        compute_factor_significance_batch(
            &param_indices_per_factor,
            &params_vec,
            &bse_vec,
            &bread_arr,
        )
    });

    let mut out: Vec<Option<Py<PyAny>>> = Vec::with_capacity(raw_results.len());
    for r in raw_results {
        match r {
            Some(raw) => {
                let dict = pyo3::types::PyDict::new(py);
                dict.set_item("chi2", raw.chi2)?;
                dict.set_item("pvalue", raw.pvalue)?;
                dict.set_item("df", raw.df)?;
                out.push(Some(dict.unbind().into()));
            }
            None => out.push(None),
        }
    }
    Ok(out)
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
    Ok(resid.into_pyarray(py).unbind())
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
    Ok(resid.into_pyarray(py).unbind())
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

    null_deviance(&y_arr, family, exp_arr.as_ref()).map_err(pyo3::exceptions::PyValueError::new_err)
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
    Ok(unit_dev.into_pyarray(py).unbind())
}

/// Compute A/E aggregates per decile (sorted by predicted value) from Rust.
///
/// `sort_idx` may be passed when the caller already holds `argsort(mu)` to
/// skip a redundant O(n log n) sort. It must be uintp-compatible (i.e.
/// `sort_idx.astype(np.uintp)` on the Python side); on 64-bit platforms this
/// matches the native `usize`.
#[pyfunction]
#[pyo3(signature = (y, mu, exposure=None, n_deciles=10, sort_idx=None))]
pub fn compute_ae_by_decile_py<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    mu: PyReadonlyArray1<'py, f64>,
    exposure: Option<PyReadonlyArray1<'py, f64>>,
    n_deciles: usize,
    sort_idx: Option<PyReadonlyArray1<'py, usize>>,
) -> PyResult<Vec<Py<PyAny>>> {
    let y_arr = y.as_array().to_owned();
    let mu_arr = mu.as_array().to_owned();
    let exp_arr = exposure.map(|e| e.as_array().to_owned());
    let sort_arr = sort_idx.map(|s| s.as_array().to_owned());

    let results = py.detach(|| {
        compute_ae_by_decile(
            &y_arr,
            &mu_arr,
            exp_arr.as_ref(),
            n_deciles,
            sort_arr.as_ref(),
        )
    });

    let mut out: Vec<Py<PyAny>> = Vec::with_capacity(results.len());
    for r in results {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("decile", r.decile)?;
        dict.set_item("n", r.n)?;
        dict.set_item("actual_sum", r.actual_sum)?;
        dict.set_item("predicted_sum", r.predicted_sum)?;
        dict.set_item("exposure_sum", r.exposure_sum)?;
        dict.set_item("mu_min", r.mu_min)?;
        dict.set_item("mu_max", r.mu_max)?;
        out.push(dict.unbind().into());
    }
    Ok(out)
}

/// Per-factor categorical partial-dependence aggregates, parallel over factors.
///
/// `codes_matrix` has shape `(n, k)` where each column is a u32 code per row
/// for one categorical factor. `mu` is the length-`n` prediction vector and
/// `n_levels_per_factor` gives the level count for each factor. Returns a
/// list of length `k`; each entry is `(counts, mu_sums)` of length
/// `n_levels_per_factor[j]`. The Python caller divides `mu_sums / counts` for
/// the per-level mean prediction. (OPT-20 — replaces 6 sequential
/// `np.bincount` × 2 calls per factor with a single rayon-parallel batch.)
#[pyfunction]
pub fn partial_dependence_categorical_batch_py<'py>(
    py: Python<'py>,
    codes_matrix: PyReadonlyArray2<'py, u32>,
    mu: PyReadonlyArray1<'py, f64>,
    n_levels_per_factor: Vec<usize>,
) -> PyResult<Vec<(Vec<f64>, Vec<f64>)>> {
    let codes_arr = codes_matrix.as_array().to_owned();
    let mu_arr = mu.as_array().to_owned();

    let n = mu_arr.len();
    let k = codes_arr.ncols();

    if codes_arr.nrows() != n {
        return Err(PyValueError::new_err(format!(
            "codes_matrix has {} rows but mu has {} elements",
            codes_arr.nrows(),
            n
        )));
    }

    if n_levels_per_factor.len() != k {
        return Err(PyValueError::new_err(format!(
            "codes_matrix has {} columns but n_levels_per_factor has {} entries",
            k,
            n_levels_per_factor.len()
        )));
    }

    let result = py
        .detach(|| partial_dependence_categorical_batch(&codes_arr, &mu_arr, &n_levels_per_factor));
    Ok(result)
}

/// Compute the diagonal of M^{-1} where M is symmetric positive-definite.
///
/// Used for VIF (Variance Inflation Factor) computation: VIF_j = diag(R^{-1})_j
/// where R is the correlation matrix of the design columns. Replaces a previous
/// scipy.linalg Cholesky/cho_solve path so that scipy is not a runtime
/// dependency. Returns NaN for every entry if the matrix is non-invertible
/// (caller should detect and raise).
#[pyfunction]
pub fn inverse_diagonal_spd_py<'py>(
    py: Python<'py>,
    matrix: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let arr = matrix.as_array().to_owned();
    if arr.nrows() != arr.ncols() {
        return Err(PyValueError::new_err(format!(
            "matrix must be square, got shape ({}, {})",
            arr.nrows(),
            arr.ncols()
        )));
    }
    let result = py.detach(|| inverse_diagonal_spd(&arr));
    Ok(result.into_pyarray(py))
}
