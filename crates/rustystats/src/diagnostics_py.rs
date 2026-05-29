// =============================================================================
// Diagnostics Bindings
// =============================================================================
//
// PyO3 wrappers for diagnostic computations: calibration curves,
// discrimination stats, A/E analysis, residuals, loss metrics, etc.
// =============================================================================

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rustystats_core::diagnostics::{
    aggregate_pair_cells, aic, compute_ae_by_decile, compute_ae_categorical,
    compute_ae_categorical_batch, compute_ae_continuous, compute_ae_continuous_batch,
    compute_calibration_curve, compute_discrimination_stats, compute_factor_significance_batch,
    compute_family_loss, compute_lorenz_curve, compute_residual_pattern_continuous,
    compute_residual_pattern_continuous_batch, correlation_and_vif, cramers_v_matrix_from_codes,
    detect_exploratory_interactions_from_codes, detect_interactions, hosmer_lemeshow_test,
    interaction_strength_from_codes, mae, mse, null_deviance, partial_dependence_categorical_batch,
    resid_deviance, resid_pearson, rmse, ActualExpectedBin, DevianceByLevel, FactorData,
    FactorDevianceResult, InteractionConfig, ResidualPattern,
};

use crate::families_py::{family_from_name_with_tweedie_support, validate_tweedie_fit_response};

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
/// `values_list` is a Python list of length k, where each element is a 1D
/// numpy array of length n holding per-row values for one factor. Returns a
/// list of length k, where each element is the same list of bin dicts
/// produced by `compute_ae_continuous_py`.
///
/// Memory note: the previous signature took a stacked (n, k) `values_matrix`
/// and re-copied each column into a `Vec<f64>` because columns of a row-major
/// matrix aren't contiguous. Both copies — the Python-side stacking AND the
/// Rust-side per-column copy — are eliminated by passing a list of
/// already-contiguous 1D arrays. At 1M rows × 24 factors this saves a 192 MB
/// transient peak (per copy, so 384 MB combined) on every factor diagnostics
/// call.
#[pyfunction]
#[pyo3(signature = (values_list, y, mu, exposure=None, n_bins=10, family="poisson"))]
pub fn compute_ae_continuous_batch_py<'py>(
    py: Python<'py>,
    values_list: Vec<PyReadonlyArray1<'py, f64>>,
    y: PyReadonlyArray1<f64>,
    mu: PyReadonlyArray1<f64>,
    exposure: Option<PyReadonlyArray1<f64>>,
    n_bins: usize,
    family: &str,
) -> PyResult<Vec<Vec<Py<PyAny>>>> {
    let y_arr = y.as_array().to_owned();
    let mu_arr = mu.as_array().to_owned();
    let exp_arr = exposure.map(|e| e.as_array().to_owned());

    let n = y_arr.len();

    // Borrow each input array as a contiguous slice. PyReadonlyArray1 holds a
    // reference to the underlying numpy buffer for the duration of the FFI
    // call, so no copy is needed. `as_slice()` requires the array to be
    // C-contiguous; numpy arrays from the Python caller already are.
    let column_slices: Vec<&[f64]> = values_list
        .iter()
        .enumerate()
        .map(|(j, arr)| {
            // `as_slice()` returns Option<&[T]> with lifetime tied to `arr`,
            // not to a local view, so it's safe to return from the closure.
            let slice = arr.as_slice().map_err(|_| {
                PyValueError::new_err(format!(
                    "values_list[{}] is not a contiguous numpy array; \
                     pass np.ascontiguousarray(arr) at the call site",
                    j
                ))
            })?;
            if slice.len() != n {
                return Err(PyValueError::new_err(format!(
                    "values_list[{}] has {} elements but y has {} (must match)",
                    j,
                    slice.len(),
                    n
                )));
            }
            Ok(slice)
        })
        .collect::<PyResult<Vec<&[f64]>>>()?;

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
/// `codes_list` is a Python list of length k where each element is a 1D
/// `np.uint32` array of length n holding per-row codes for one factor (codes
/// index into that factor's `levels_list[j]`). Returns a list of length k,
/// each element being the same list of bin dicts as `compute_ae_categorical_py`.
///
/// This avoids the per-row string marshalling that dominates the singular
/// categorical path when n is large; all categorical A/E work is now a single
/// cross-FFI call into a rayon-parallel Rust loop.
///
/// Memory note: previously took a stacked (n, k) `codes_matrix` plus a
/// per-column Rust copy. Switching to a list of contiguous u32 arrays
/// eliminates both, saving ~24 MB transient peak (per copy, 48 MB combined)
/// at 1M rows × 6 categorical factors.
#[pyfunction]
#[pyo3(signature = (codes_list, levels_list, y, mu, exposure=None, rare_threshold_pct=1.0, max_levels=20, family="poisson"))]
pub fn compute_ae_categorical_batch_py<'py>(
    py: Python<'py>,
    codes_list: Vec<PyReadonlyArray1<'py, u32>>,
    levels_list: Vec<Vec<String>>,
    y: PyReadonlyArray1<f64>,
    mu: PyReadonlyArray1<f64>,
    exposure: Option<PyReadonlyArray1<f64>>,
    rare_threshold_pct: f64,
    max_levels: usize,
    family: &str,
) -> PyResult<Vec<Vec<Py<PyAny>>>> {
    let y_arr = y.as_array().to_owned();
    let mu_arr = mu.as_array().to_owned();
    let exp_arr = exposure.map(|e| e.as_array().to_owned());

    let n = y_arr.len();
    let k = codes_list.len();

    if levels_list.len() != k {
        return Err(PyValueError::new_err(format!(
            "codes_list has {} entries but levels_list has {} entries",
            k,
            levels_list.len()
        )));
    }

    let code_slices: Vec<&[u32]> = codes_list
        .iter()
        .enumerate()
        .map(|(j, arr)| {
            let slice = arr.as_slice().map_err(|_| {
                PyValueError::new_err(format!(
                    "codes_list[{}] is not a contiguous numpy array; \
                     pass np.ascontiguousarray(arr, dtype=np.uint32) at the call site",
                    j
                ))
            })?;
            if slice.len() != n {
                return Err(PyValueError::new_err(format!(
                    "codes_list[{}] has {} elements but y has {} (must match)",
                    j,
                    slice.len(),
                    n
                )));
            }
            Ok(slice)
        })
        .collect::<PyResult<Vec<&[u32]>>>()?;
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

/// Compute a Cramer's V matrix from pre-factorized categorical codes.
///
/// `codes_list` is a list of contiguous uint32 arrays, one per categorical
/// factor. `n_levels_per_factor[j]` is the number of declared levels for
/// `codes_list[j]`. The row-heavy pairwise contingency aggregation runs in
/// Rust and returns only the small symmetric matrix to Python.
#[pyfunction]
pub fn compute_cramers_v_matrix_from_codes_py<'py>(
    py: Python<'py>,
    codes_list: Vec<PyReadonlyArray1<'py, u32>>,
    n_levels_per_factor: Vec<usize>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let k = codes_list.len();
    if n_levels_per_factor.len() != k {
        return Err(PyValueError::new_err(format!(
            "codes_list has {} entries but n_levels_per_factor has {} entries",
            k,
            n_levels_per_factor.len()
        )));
    }
    if k == 0 {
        return Ok(ndarray::Array2::<f64>::zeros((0, 0)).into_pyarray(py));
    }

    let n = codes_list[0].as_array().len();
    let slices: Vec<&[u32]> = codes_list
        .iter()
        .enumerate()
        .map(|(j, arr)| {
            let slice = arr.as_slice().map_err(|_| {
                PyValueError::new_err(format!(
                    "codes_list[{}] is not a contiguous numpy array; \
                     pass np.ascontiguousarray(arr) at the call site",
                    j
                ))
            })?;
            if slice.len() != n {
                return Err(PyValueError::new_err(format!(
                    "codes_list[{}] has {} rows but codes_list[0] has {} rows",
                    j,
                    slice.len(),
                    n
                )));
            }
            Ok(slice)
        })
        .collect::<PyResult<Vec<&[u32]>>>()?;

    let matrix = py
        .detach(|| cramers_v_matrix_from_codes(&slices, &n_levels_per_factor))
        .map_err(PyValueError::new_err)?;
    Ok(matrix.into_pyarray(py))
}

/// Detect pre-fit response-based interactions from integer-coded factor bins.
///
/// Categorical factors should pass their factorization codes. Continuous
/// factors should pass precomputed quantile-bin codes. This keeps public API
/// semantics in Python while moving the O(n * pair_count) aggregation into
/// Rust.
#[pyfunction]
#[pyo3(signature = (
    y,
    exposure,
    factor_names,
    codes_list,
    n_levels_per_factor,
    max_factors=10,
    min_effect_size=0.001,
    max_candidates=5,
    min_cell_count=30
))]
pub fn detect_exploratory_interactions_py<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    exposure: PyReadonlyArray1<'py, f64>,
    factor_names: Vec<String>,
    codes_list: Vec<PyReadonlyArray1<'py, u32>>,
    n_levels_per_factor: Vec<usize>,
    max_factors: usize,
    min_effect_size: f64,
    max_candidates: usize,
    min_cell_count: usize,
) -> PyResult<Vec<Py<PyAny>>> {
    let n = y.as_array().len();
    let y_slice = y.as_slice().map_err(|_| {
        PyValueError::new_err("y is not a contiguous numpy array; pass np.ascontiguousarray(y)")
    })?;
    let exposure_slice = exposure.as_slice().map_err(|_| {
        PyValueError::new_err(
            "exposure is not a contiguous numpy array; pass np.ascontiguousarray(exposure)",
        )
    })?;
    if exposure_slice.len() != n {
        return Err(PyValueError::new_err(format!(
            "y has {} rows but exposure has {} rows",
            n,
            exposure_slice.len()
        )));
    }

    let k = codes_list.len();
    if factor_names.len() != k || n_levels_per_factor.len() != k {
        return Err(PyValueError::new_err(format!(
            "factor_names, codes_list, and n_levels_per_factor must have the same length \
             (got {}, {}, {})",
            factor_names.len(),
            k,
            n_levels_per_factor.len()
        )));
    }

    let code_slices: Vec<&[u32]> = codes_list
        .iter()
        .enumerate()
        .map(|(j, arr)| {
            let slice = arr.as_slice().map_err(|_| {
                PyValueError::new_err(format!(
                    "codes_list[{}] is not a contiguous numpy array; \
                     pass np.ascontiguousarray(arr) at the call site",
                    j
                ))
            })?;
            if slice.len() != n {
                return Err(PyValueError::new_err(format!(
                    "codes_list[{}] has {} rows but y has {} rows",
                    j,
                    slice.len(),
                    n
                )));
            }
            Ok(slice)
        })
        .collect::<PyResult<Vec<&[u32]>>>()?;

    let candidates = py
        .detach(|| {
            detect_exploratory_interactions_from_codes(
                &factor_names,
                &code_slices,
                &n_levels_per_factor,
                y_slice,
                exposure_slice,
                max_factors,
                min_effect_size,
                max_candidates,
                min_cell_count,
            )
        })
        .map_err(PyValueError::new_err)?;

    candidates
        .into_iter()
        .map(|candidate| {
            let dict = PyDict::new(py);
            dict.set_item("factor1", candidate.factor1)?;
            dict.set_item("factor2", candidate.factor2)?;
            dict.set_item("interaction_strength", candidate.interaction_strength)?;
            dict.set_item("pvalue", candidate.pvalue)?;
            dict.set_item("n_cells", candidate.n_cells)?;
            Ok(dict.unbind().into())
        })
        .collect()
}

/// Aggregate ``(y, exposure, optional mu, count)`` by ``(code1, code2)`` cell
/// for an explicit pair of factor-code arrays. Returns only non-empty cells
/// as ``(r, c, count, exposure_sum, y_sum, mu_sum)`` tuples.
///
/// Used by the post-fit pair diagnostics pipeline so the O(n) bincount-style
/// aggregation happens in Rust (without the GIL held) rather than via three
/// or four sequential numpy bincount calls plus a Python dict assembly.
#[pyfunction]
#[pyo3(signature = (codes1, n_levels1, codes2, n_levels2, y, exposure, mu=None))]
pub fn aggregate_pair_cells_py<'py>(
    py: Python<'py>,
    codes1: PyReadonlyArray1<'py, u32>,
    n_levels1: usize,
    codes2: PyReadonlyArray1<'py, u32>,
    n_levels2: usize,
    y: PyReadonlyArray1<'py, f64>,
    exposure: PyReadonlyArray1<'py, f64>,
    mu: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Vec<(u32, u32, u64, f64, f64, f64)>> {
    let n = y.as_array().len();
    let codes1_slice = codes1.as_slice().map_err(|_| {
        PyValueError::new_err(
            "codes1 is not a contiguous numpy array; pass np.ascontiguousarray(codes1)",
        )
    })?;
    let codes2_slice = codes2.as_slice().map_err(|_| {
        PyValueError::new_err(
            "codes2 is not a contiguous numpy array; pass np.ascontiguousarray(codes2)",
        )
    })?;
    let y_slice = y.as_slice().map_err(|_| {
        PyValueError::new_err("y is not a contiguous numpy array; pass np.ascontiguousarray(y)")
    })?;
    let exposure_slice = exposure.as_slice().map_err(|_| {
        PyValueError::new_err(
            "exposure is not a contiguous numpy array; pass np.ascontiguousarray(exposure)",
        )
    })?;
    if codes1_slice.len() != n || codes2_slice.len() != n || exposure_slice.len() != n {
        return Err(PyValueError::new_err(format!(
            "length mismatch: y={}, codes1={}, codes2={}, exposure={}",
            n,
            codes1_slice.len(),
            codes2_slice.len(),
            exposure_slice.len()
        )));
    }
    let mu_owned = mu
        .as_ref()
        .map(|m| {
            m.as_slice().map_err(|_| {
                PyValueError::new_err(
                    "mu is not a contiguous numpy array; pass np.ascontiguousarray(mu)",
                )
            })
        })
        .transpose()?;
    if let Some(m) = mu_owned {
        if m.len() != n {
            return Err(PyValueError::new_err(format!(
                "mu has length {} but y has length {}",
                m.len(),
                n
            )));
        }
    }

    py.detach(|| {
        aggregate_pair_cells(
            codes1_slice,
            n_levels1,
            codes2_slice,
            n_levels2,
            y_slice,
            exposure_slice,
            mu_owned,
        )
    })
    .map_err(PyValueError::new_err)
}

/// Compute the cell-grouping R² ("interaction strength") for a single
/// pre-binned pair. Identical scalar to what
/// ``detect_exploratory_interactions_py`` returns on
/// ``InteractionCandidate.interaction_strength`` for the same pair —
/// reuses the same core ``interaction_strength_from_codes`` function so
/// no formula is duplicated on the Python side.
///
/// Raises ``ValueError`` when the strength cannot be computed: total
/// exposure is zero, total variance is zero, fewer than 4 non-empty
/// cells survive the ``min_cell_count`` filter, or ``n`` is smaller than
/// ``4 * min_cell_count``. Callers requesting an explicit pair should
/// see the error rather than a silently-defaulted scalar.
#[pyfunction]
#[pyo3(signature = (codes1, n_levels1, codes2, n_levels2, y, exposure, min_cell_count=0))]
pub fn interaction_strength_from_codes_py<'py>(
    py: Python<'py>,
    codes1: PyReadonlyArray1<'py, u32>,
    n_levels1: usize,
    codes2: PyReadonlyArray1<'py, u32>,
    n_levels2: usize,
    y: PyReadonlyArray1<'py, f64>,
    exposure: PyReadonlyArray1<'py, f64>,
    min_cell_count: usize,
) -> PyResult<f64> {
    let n = y.as_array().len();
    let codes1_slice = codes1.as_slice().map_err(|_| {
        PyValueError::new_err(
            "codes1 is not a contiguous numpy array; pass np.ascontiguousarray(codes1)",
        )
    })?;
    let codes2_slice = codes2.as_slice().map_err(|_| {
        PyValueError::new_err(
            "codes2 is not a contiguous numpy array; pass np.ascontiguousarray(codes2)",
        )
    })?;
    let y_slice = y.as_slice().map_err(|_| {
        PyValueError::new_err("y is not a contiguous numpy array; pass np.ascontiguousarray(y)")
    })?;
    let exposure_slice = exposure.as_slice().map_err(|_| {
        PyValueError::new_err(
            "exposure is not a contiguous numpy array; pass np.ascontiguousarray(exposure)",
        )
    })?;
    if codes1_slice.len() != n || codes2_slice.len() != n || exposure_slice.len() != n {
        return Err(PyValueError::new_err(format!(
            "length mismatch: y={}, codes1={}, codes2={}, exposure={}",
            n,
            codes1_slice.len(),
            codes2_slice.len(),
            exposure_slice.len()
        )));
    }

    let candidate = py.detach(|| {
        interaction_strength_from_codes(
            "factor1",
            codes1_slice,
            n_levels1,
            "factor2",
            codes2_slice,
            n_levels2,
            y_slice,
            exposure_slice,
            min_cell_count,
        )
    });

    candidate.map(|c| c.interaction_strength).ok_or_else(|| {
        PyValueError::new_err(format!(
            "interaction_strength_from_codes: cannot compute strength for this \
                 pair (n={n}, n_levels=({n_levels1}, {n_levels2}), \
                 min_cell_count={min_cell_count}). The pair has zero total exposure, \
                 zero total variance, or fewer than 4 non-empty cells after filtering."
        ))
    })
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
    validate_tweedie_fit_response(family, &y_arr, 1.5, true)?;
    let fam = family_from_name_with_tweedie_support(family, 1.5, 1.0, true)?;

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
    validate_tweedie_fit_response(family, &y_arr, parsed_var_power, true)?;
    let fam = family_from_name_with_tweedie_support(family, parsed_var_power, parsed_theta, true)?;
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
/// `values_list` is a Python list of length k where each element is a 1D
/// numpy array of length n holding per-row values for one factor. Returns a
/// list of length k, each element being the same dict produced by
/// `compute_residual_pattern_py`.
///
/// Memory note: like `compute_ae_continuous_batch_py`, this used to take a
/// stacked (n, k) `values_matrix` and re-copy each column into a `Vec<f64>`.
/// Both copies are eliminated by accepting a list of already-contiguous 1D
/// arrays, saving a 192 MB transient peak (per copy, 384 MB combined) at
/// 1M rows × 24 factors.
#[pyfunction]
#[pyo3(signature = (values_list, residuals, n_bins=10))]
pub fn compute_residual_pattern_batch_py<'py>(
    py: Python<'py>,
    values_list: Vec<PyReadonlyArray1<'py, f64>>,
    residuals: PyReadonlyArray1<f64>,
    n_bins: usize,
) -> PyResult<Vec<Py<PyAny>>> {
    let resid_arr = residuals.as_array().to_owned();
    let n = resid_arr.len();

    let column_slices: Vec<&[f64]> = values_list
        .iter()
        .enumerate()
        .map(|(j, arr)| {
            let slice = arr.as_slice().map_err(|_| {
                PyValueError::new_err(format!(
                    "values_list[{}] is not a contiguous numpy array; \
                     pass np.ascontiguousarray(arr) at the call site",
                    j
                ))
            })?;
            if slice.len() != n {
                return Err(PyValueError::new_err(format!(
                    "values_list[{}] has {} elements but residuals has {} (must match)",
                    j,
                    slice.len(),
                    n
                )));
            }
            Ok(slice)
        })
        .collect::<PyResult<Vec<&[f64]>>>()?;

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
    validate_tweedie_fit_response(family, &y_arr, 1.5, true)?;
    let fam = family_from_name_with_tweedie_support(family, 1.5, 1.0, true)?;
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
    validate_tweedie_fit_response(family, &y_arr, 1.5, true)?;
    let fam = family_from_name_with_tweedie_support(family, 1.5, 1.0, true)?;
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
    validate_tweedie_fit_response(family, &y_arr, 1.5, true)?;
    let fam = family_from_name_with_tweedie_support(family, 1.5, 1.0, true)?;
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
/// `codes_list` is a length-`k` list of 1D u32 numpy arrays; entry `j` has
/// shape `(n,)` holding the per-row code for factor `j`. `mu` is the length-
/// `n` prediction vector and `n_levels_per_factor` gives the level count for
/// each factor. Returns a list of length `k`; each entry is `(counts,
/// mu_sums)` of length `n_levels_per_factor[j]`. The Python caller divides
/// `mu_sums / counts` for the per-level mean prediction. (OPT-20 — replaces 6
/// sequential `np.bincount` × 2 calls per factor with a single rayon-parallel
/// batch.)
///
/// Memory note: this used to take a stacked `(n, k)` `codes_matrix` which
/// forced the Python caller to run `np.stack(cached_codes, axis=1)` — a
/// 400 MB transient at n=1M × k=100. Accepting a list of already-contiguous
/// 1D arrays lets the Rust side borrow the numpy buffers directly with no
/// copy or stacking.
#[pyfunction]
pub fn partial_dependence_categorical_batch_py<'py>(
    py: Python<'py>,
    codes_list: Vec<PyReadonlyArray1<'py, u32>>,
    mu: PyReadonlyArray1<'py, f64>,
    n_levels_per_factor: Vec<usize>,
) -> PyResult<Vec<(Vec<f64>, Vec<f64>)>> {
    let mu_arr = mu.as_array().to_owned();
    let n = mu_arr.len();
    let k = codes_list.len();

    if n_levels_per_factor.len() != k {
        return Err(PyValueError::new_err(format!(
            "codes_list has {} entries but n_levels_per_factor has {} entries",
            k,
            n_levels_per_factor.len()
        )));
    }

    // Borrow each input array as a contiguous slice. PyReadonlyArray1 holds a
    // reference to the underlying numpy buffer for the duration of the FFI
    // call, so no copy is needed. `as_slice()` requires the array to be
    // C-contiguous; numpy arrays from the Python caller already are.
    let column_slices: Vec<&[u32]> = codes_list
        .iter()
        .enumerate()
        .map(|(j, arr)| {
            let slice = arr.as_slice().map_err(|_| {
                PyValueError::new_err(format!(
                    "codes_list[{}] is not a contiguous numpy array; \
                     pass np.ascontiguousarray(arr) at the call site",
                    j
                ))
            })?;
            if slice.len() != n {
                return Err(PyValueError::new_err(format!(
                    "codes_list[{}] has {} elements but mu has {} (must match)",
                    j,
                    slice.len(),
                    n
                )));
            }
            Ok(slice)
        })
        .collect::<PyResult<Vec<&[u32]>>>()?;

    let result = py.detach(|| {
        partial_dependence_categorical_batch(&column_slices, &mu_arr, &n_levels_per_factor)
    });
    Ok(result)
}

/// Compute the column correlation matrix and the VIF (`diag((R + ε·I)^{-1})`)
/// of an (n x k) design matrix in a single Rust call.
///
/// This is the memory-efficient path for `Computer.compute_vif`. The Python
/// implementation that this replaces materializes a full (n, k) mean-centered
/// copy via `numpy.corrcoef`, peaking at O(n·k) bytes. The Rust path takes a
/// zero-copy `PyReadonlyArray2` view, computes the correlation matrix and VIF
/// in O(k²) extra memory, and returns only the small (k, k) and (k,) arrays
/// to Python.
///
/// `skip_cols` lets the caller drop leading columns (typically the intercept)
/// without first slicing in Python. Slicing in Python (`X[:, 1:]`) yields a
/// non-contiguous view that defeats the row-major fast path; skipping inside
/// Rust preserves the C-contiguous slice access for the inner loop.
///
/// Returns `(R, vif_diagonal)` where:
///   * `R` is the (k - skip_cols) x (k - skip_cols) correlation matrix (zero-
///     variance columns get an all-zero row/col with 0 on the diagonal —
///     caller may overwrite to 1).
///   * `vif_diagonal` is `diag((R + ε·I)^{-1})`. Returns NaN for every entry
///     if the regularized matrix is non-invertible (caller should detect and
///     raise).
#[pyfunction]
#[pyo3(signature = (x, epsilon, skip_cols=0))]
pub fn compute_correlation_and_vif_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    epsilon: f64,
    skip_cols: usize,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>)> {
    let full_view = x.as_array();
    let total_cols = full_view.ncols();
    if skip_cols >= total_cols {
        return Err(PyValueError::new_err(format!(
            "skip_cols ({}) must be less than the column count ({})",
            skip_cols, total_cols
        )));
    }
    // Slice off the leading `skip_cols`. For C-contiguous X this still leaves
    // a non-contiguous view, BUT the strided fallback is fast enough for
    // diagnostic use; the contiguous fast path triggers when skip_cols=0.
    let view = full_view.slice(ndarray::s![.., skip_cols..]);
    let (r, vif) = py.detach(|| correlation_and_vif(view, epsilon));
    Ok((r.into_pyarray(py), vif.into_pyarray(py)))
}
