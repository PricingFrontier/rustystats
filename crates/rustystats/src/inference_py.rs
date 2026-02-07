// =============================================================================
// Inference and Statistical Distribution Python Bindings
// =============================================================================
//
// Rao's score tests for unfitted factors and statistical distribution CDFs.
// =============================================================================

use pyo3::prelude::*;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};

use rustystats_core::inference::{score_test_continuous, score_test_categorical};
use rustystats_core::diagnostics::{chi2_cdf, t_cdf, f_cdf};

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
    
    let result = score_test_continuous(&z_arr, &x_arr, &y_arr, &mu_arr, &weights_arr, &bread_arr, family);
    
    let dict = pyo3::types::PyDict::new_bound(py);
    dict.set_item("statistic", result.statistic)?;
    dict.set_item("df", result.df)?;
    dict.set_item("pvalue", result.pvalue)?;
    dict.set_item("significant", result.significant)?;
    
    Ok(dict)
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
    let z_arr = z_matrix.as_array().to_owned();
    let x_arr = x.as_array().to_owned();
    let y_arr = y.as_array().to_owned();
    let mu_arr = mu.as_array().to_owned();
    let weights_arr = weights.as_array().to_owned();
    let bread_arr = bread.as_array().to_owned();
    
    let result = score_test_categorical(&z_arr, &x_arr, &y_arr, &mu_arr, &weights_arr, &bread_arr, family);
    
    let dict = pyo3::types::PyDict::new_bound(py);
    dict.set_item("statistic", result.statistic)?;
    dict.set_item("df", result.df)?;
    dict.set_item("pvalue", result.pvalue)?;
    dict.set_item("significant", result.significant)?;
    
    Ok(dict)
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
