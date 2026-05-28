// =============================================================================
// Inference and Statistical Distribution Python Bindings
// =============================================================================
//
// Rao's score tests for unfitted factors and statistical distribution CDFs.
// =============================================================================

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::families_py::family_from_name_with_tweedie_support;
use rustystats_core::diagnostics::{chi2_cdf, f_cdf, t_cdf};
use rustystats_core::inference::{
    score_test_categorical, score_test_continuous, score_test_continuous_batch, ScoreTestResult,
};

// =============================================================================
// Dict construction helpers
// =============================================================================

/// Build a `{statistic, df, pvalue, significant}` dict from a `ScoreTestResult`.
fn score_test_result_to_dict<'py>(
    py: Python<'py>,
    result: &ScoreTestResult,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("statistic", result.statistic)?;
    dict.set_item("df", result.df)?;
    dict.set_item("pvalue", result.pvalue)?;
    dict.set_item("significant", result.significant)?;
    Ok(dict)
}

// =============================================================================
// Rao's Score Test for Unfitted Factors
// =============================================================================

/// Compute Rao's score test for adding a continuous variable to a fitted model.
///
/// Tests whether adding this variable would significantly improve the model
/// without actually refitting.
///
/// # Arguments
/// * `z` - The new variable to test (n,)
/// * `x` - Design matrix of the fitted model (n, p)
/// * `y` - Response variable (n,)
/// * `mu` - Fitted values from the current model (n,)
/// * `weights` - Working weights from IRLS (n,)
/// * `bread` - (X'WX)^-1 matrix from the fitted model (p, p)
/// * `family` - Family name for variance function
///
/// # Returns
/// Dict with statistic, df, pvalue, significant
#[pyfunction]
pub fn score_test_continuous_py<'py>(
    py: Python<'py>,
    z: PyReadonlyArray1<'py, f64>,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    mu: PyReadonlyArray1<'py, f64>,
    weights: PyReadonlyArray1<'py, f64>,
    bread: PyReadonlyArray2<'py, f64>,
    family: &str,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let z_arr = z.as_array().to_owned();
    let x_arr = x.as_array().to_owned();
    let y_arr = y.as_array().to_owned();
    let mu_arr = mu.as_array().to_owned();
    let weights_arr = weights.as_array().to_owned();
    let bread_arr = bread.as_array().to_owned();

    let n = y_arr.len();
    let p_x = x_arr.ncols();

    if z_arr.len() != n {
        return Err(PyValueError::new_err(format!(
            "score_test_continuous: z has {} elements but y has {} (must match)",
            z_arr.len(),
            n
        )));
    }
    if x_arr.nrows() != n {
        return Err(PyValueError::new_err(format!(
            "score_test_continuous: x has {} rows but y has {} elements (must match)",
            x_arr.nrows(),
            n
        )));
    }
    if mu_arr.len() != n {
        return Err(PyValueError::new_err(format!(
            "score_test_continuous: mu has {} elements but y has {} (must match)",
            mu_arr.len(),
            n
        )));
    }
    if weights_arr.len() != n {
        return Err(PyValueError::new_err(format!(
            "score_test_continuous: weights has {} elements but y has {} (must match)",
            weights_arr.len(),
            n
        )));
    }
    if bread_arr.nrows() != p_x || bread_arr.ncols() != p_x {
        return Err(PyValueError::new_err(format!(
            "score_test_continuous: bread is ({}, {}) but x has {} columns (must be square p\u{00d7}p)",
            bread_arr.nrows(),
            bread_arr.ncols(),
            p_x
        )));
    }

    // Accept the extended Tweedie regime: a fitted model already opted in at
    // fit time, and the family string here may carry an embedded `p=` outside
    // the default (1, 2) interior.
    let family_obj = family_from_name_with_tweedie_support(family, 1.5, 1.0, true)?;

    let result = score_test_continuous(
        &z_arr,
        &x_arr,
        &y_arr,
        &mu_arr,
        &weights_arr,
        &bread_arr,
        &*family_obj,
    );

    score_test_result_to_dict(py, &result)
}

/// Batched Rao's score test for adding k continuous variables to a fitted model.
///
/// Same per-test result as `score_test_continuous_py`, but precomputes shared
/// quantities and parallelizes across the k columns of `zs`.
///
/// # Arguments
/// * `zs` - Candidate variables stacked as columns (n, k)
/// * `x` - Design matrix of the fitted model (n, p)
/// * `y` - Response variable (n,)
/// * `mu` - Fitted values from the current model (n,)
/// * `weights` - Working weights from IRLS (n,)
/// * `bread` - (X'WX)^-1 matrix from the fitted model (p, p)
/// * `family` - Family name for variance function
///
/// # Returns
/// List of dicts (one per column of zs), each with statistic, df, pvalue, significant.
#[pyfunction]
pub fn score_test_continuous_batch_py<'py>(
    py: Python<'py>,
    zs: PyReadonlyArray2<'py, f64>,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    mu: PyReadonlyArray1<'py, f64>,
    weights: PyReadonlyArray1<'py, f64>,
    bread: PyReadonlyArray2<'py, f64>,
    family: &str,
) -> PyResult<Bound<'py, pyo3::types::PyList>> {
    let zs_arr = zs.as_array().to_owned();
    let x_arr = x.as_array().to_owned();
    let y_arr = y.as_array().to_owned();
    let mu_arr = mu.as_array().to_owned();
    let weights_arr = weights.as_array().to_owned();
    let bread_arr = bread.as_array().to_owned();

    let n = y_arr.len();
    let p_x = x_arr.ncols();

    if zs_arr.nrows() != n {
        return Err(PyValueError::new_err(format!(
            "score_test_continuous_batch: zs has {} rows but y has {} elements (must match)",
            zs_arr.nrows(),
            n
        )));
    }
    if x_arr.nrows() != n {
        return Err(PyValueError::new_err(format!(
            "score_test_continuous_batch: x has {} rows but y has {} elements (must match)",
            x_arr.nrows(),
            n
        )));
    }
    if mu_arr.len() != n {
        return Err(PyValueError::new_err(format!(
            "score_test_continuous_batch: mu has {} elements but y has {} (must match)",
            mu_arr.len(),
            n
        )));
    }
    if weights_arr.len() != n {
        return Err(PyValueError::new_err(format!(
            "score_test_continuous_batch: weights has {} elements but y has {} (must match)",
            weights_arr.len(),
            n
        )));
    }
    if bread_arr.nrows() != p_x || bread_arr.ncols() != p_x {
        return Err(PyValueError::new_err(format!(
            "score_test_continuous_batch: bread is ({}, {}) but x has {} columns (must be square p\u{00d7}p)",
            bread_arr.nrows(),
            bread_arr.ncols(),
            p_x
        )));
    }

    // Accept extended Tweedie — see `score_test_continuous_py` for rationale.
    let family_obj = family_from_name_with_tweedie_support(family, 1.5, 1.0, true)?;

    let results = score_test_continuous_batch(
        &zs_arr,
        &x_arr,
        &y_arr,
        &mu_arr,
        &weights_arr,
        &bread_arr,
        &*family_obj,
    );

    let list = pyo3::types::PyList::empty(py);
    for result in results {
        list.append(score_test_result_to_dict(py, &result)?)?;
    }

    Ok(list)
}

/// Compute Rao's score test for adding a categorical variable to a fitted model.
///
/// Tests whether adding this variable would significantly improve the model
/// without actually refitting.
///
/// # Arguments
/// * `z_matrix` - Dummy-coded matrix for the categorical (n, k-1)
/// * `x` - Design matrix of the fitted model (n, p)
/// * `y` - Response variable (n,)
/// * `mu` - Fitted values from the current model (n,)
/// * `weights` - Working weights from IRLS (n,)
/// * `bread` - (X'WX)^-1 matrix from the fitted model (p, p)
/// * `family` - Family name for variance function
///
/// # Returns
/// Dict with statistic, df, pvalue, significant
#[pyfunction]
pub fn score_test_categorical_py<'py>(
    py: Python<'py>,
    z_matrix: PyReadonlyArray2<'py, f64>,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    mu: PyReadonlyArray1<'py, f64>,
    weights: PyReadonlyArray1<'py, f64>,
    bread: PyReadonlyArray2<'py, f64>,
    family: &str,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    // Accept extended Tweedie — see `score_test_continuous_py` for rationale.
    let family_obj = family_from_name_with_tweedie_support(family, 1.5, 1.0, true)?;

    let z_arr = z_matrix.as_array().to_owned();
    let x_arr = x.as_array().to_owned();
    let y_arr = y.as_array().to_owned();
    let mu_arr = mu.as_array().to_owned();
    let weights_arr = weights.as_array().to_owned();
    let bread_arr = bread.as_array().to_owned();

    let result = score_test_categorical(
        &z_arr,
        &x_arr,
        &y_arr,
        &mu_arr,
        &weights_arr,
        &bread_arr,
        &*family_obj,
    );

    score_test_result_to_dict(py, &result)
}

// =============================================================================
// Statistical Distribution CDFs (for p-value calculations)
// =============================================================================

/// Chi-squared distribution CDF: P(X <= x) where X ~ χ²(df)
#[pyfunction]
pub fn chi2_cdf_py(x: f64, df: f64) -> f64 {
    chi2_cdf(x, df)
}

/// Student's t-distribution CDF: P(X <= x) where X ~ t(df)
#[pyfunction]
pub fn t_cdf_py(x: f64, df: f64) -> f64 {
    t_cdf(x, df)
}

/// F-distribution CDF: P(X <= x) where X ~ F(df1, df2)
#[pyfunction]
pub fn f_cdf_py(x: f64, df1: f64, df2: f64) -> f64 {
    f_cdf(x, df1, df2)
}
