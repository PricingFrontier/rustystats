// =============================================================================
// Spline Basis Functions
// =============================================================================
//
// B-splines and natural splines for non-linear continuous effects in GLMs.
// These are computed in Rust for maximum performance.
// =============================================================================

use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1};

use rustystats_core::splines;

/// Compute B-spline basis matrix.
///
/// B-splines are flexible piecewise polynomial bases commonly used for
/// modeling non-linear continuous effects in regression models.
///
/// Parameters
/// ----------
/// x : numpy.ndarray
///     Data points (1D array of length n)
/// df : int
///     Degrees of freedom (number of basis functions)
/// degree : int, optional
///     Spline degree. Default 3 (cubic splines).
/// boundary_knots : tuple, optional
///     (min, max) boundary knots. If None, uses data range.
/// include_intercept : bool, optional
///     Whether to include an intercept column. Default False.
///
/// Returns
/// -------
/// numpy.ndarray
///     Basis matrix of shape (n, df) or (n, df-1) if include_intercept=False
///
/// Examples
/// --------
/// >>> import rustystats as rs
/// >>> import numpy as np
/// >>> x = np.linspace(0, 10, 100)
/// >>> basis = rs.bs(x, df=5)
/// >>> print(basis.shape)
/// (100, 4)
#[pyfunction]
#[pyo3(signature = (x, df, degree=3, boundary_knots=None, include_intercept=false))]
pub fn bs_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    df: usize,
    degree: usize,
    boundary_knots: Option<(f64, f64)>,
    include_intercept: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let x_array = x.as_array().to_owned();
    let result = splines::bs_basis(&x_array, df, degree, boundary_knots, include_intercept);
    Ok(result.into_pyarray_bound(py))
}

/// Compute natural cubic spline basis matrix.
///
/// Natural splines are cubic splines with the additional constraint that
/// the second derivative is zero at the boundaries. This makes extrapolation
/// linear beyond the data range, which is often more sensible for prediction.
///
/// Parameters
/// ----------
/// x : numpy.ndarray
///     Data points (1D array of length n)
/// df : int
///     Degrees of freedom (number of basis functions)
/// boundary_knots : tuple, optional
///     (min, max) boundary knots. If None, uses data range.
/// include_intercept : bool, optional
///     Whether to include an intercept column. Default False.
///
/// Returns
/// -------
/// numpy.ndarray
///     Basis matrix of shape (n, df) or (n, df-1)
///
/// Notes
/// -----
/// Natural splines are recommended when extrapolation beyond the data
/// range is needed, as they provide more sensible linear extrapolation
/// compared to B-splines which can have erratic behavior at boundaries.
///
/// Examples
/// --------
/// >>> import rustystats as rs
/// >>> import numpy as np
/// >>> x = np.linspace(0, 10, 100)
/// >>> basis = rs.ns(x, df=5)
/// >>> print(basis.shape)
/// (100, 4)
#[pyfunction]
#[pyo3(signature = (x, df, boundary_knots=None, include_intercept=false))]
pub fn ns_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    df: usize,
    boundary_knots: Option<(f64, f64)>,
    include_intercept: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let x_array = x.as_array().to_owned();
    let result = splines::ns_basis(&x_array, df, boundary_knots, include_intercept);
    Ok(result.into_pyarray_bound(py))
}

/// Compute natural spline basis with explicit interior knots.
///
/// This is essential for prediction on new data where the knots must
/// match those computed during training.
///
/// Parameters
/// ----------
/// x : numpy.ndarray
///     Data points (1D array)
/// interior_knots : list
///     Interior knot positions (computed from training data)
/// boundary_knots : tuple
///     (min, max) boundary knots
/// include_intercept : bool, optional
///     Whether to include intercept. Default False.
///
/// Returns
/// -------
/// numpy.ndarray
///     Natural spline basis matrix
#[pyfunction]
#[pyo3(signature = (x, interior_knots, boundary_knots, include_intercept=false))]
pub fn ns_with_knots_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    interior_knots: Vec<f64>,
    boundary_knots: (f64, f64),
    include_intercept: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let x_array = x.as_array().to_owned();
    let result = splines::ns_basis_with_knots(&x_array, &interior_knots, boundary_knots, include_intercept);
    Ok(result.into_pyarray_bound(py))
}

/// Compute B-spline basis with explicit knots.
///
/// For cases where you want to specify interior knots directly rather
/// than having them computed from the data.
///
/// Parameters
/// ----------
/// x : numpy.ndarray
///     Data points (1D array)
/// knots : list
///     Interior knot positions
/// degree : int, optional
///     Spline degree. Default 3.
/// boundary_knots : tuple, optional
///     (min, max) boundary knots.
///
/// Returns
/// -------
/// numpy.ndarray
///     Basis matrix
#[pyfunction]
#[pyo3(signature = (x, knots, degree=3, boundary_knots=None))]
pub fn bs_knots_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    knots: Vec<f64>,
    degree: usize,
    boundary_knots: Option<(f64, f64)>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let x_array = x.as_array().to_owned();
    let result = splines::bs_with_knots(&x_array, &knots, degree, boundary_knots);
    Ok(result.into_pyarray_bound(py))
}

/// Get column names for B-spline basis.
#[pyfunction]
#[pyo3(signature = (var_name, df, include_intercept=false))]
pub fn bs_names_py(var_name: &str, df: usize, include_intercept: bool) -> Vec<String> {
    splines::bs_names(var_name, df, include_intercept)
}

/// Get column names for natural spline basis.
#[pyfunction]
#[pyo3(signature = (var_name, df, include_intercept=false))]
pub fn ns_names_py(var_name: &str, df: usize, include_intercept: bool) -> Vec<String> {
    splines::ns_names(var_name, df, include_intercept)
}

/// Compute I-spline (monotonic spline) basis matrix.
///
/// I-splines are integrated M-splines that provide a basis for monotonic
/// regression. Each basis function is monotonically increasing from 0 to 1.
/// With non-negative coefficients, any linear combination produces a
/// monotonically increasing function.
///
/// This is the standard approach for fitting monotonic curves in GLMs,
/// commonly used in actuarial applications where effects should be
/// constrained to increase or decrease with the predictor.
///
/// Parameters
/// ----------
/// x : numpy.ndarray
///     Data points (1D array of length n)
/// df : int
///     Degrees of freedom (number of basis functions)
/// degree : int, optional
///     Spline degree. Default 3 (cubic).
/// boundary_knots : tuple, optional
///     (min, max) boundary knots. If None, uses data range.
/// increasing : bool, optional
///     If True (default), basis for monotonically increasing function.
///     If False, basis for monotonically decreasing function.
///
/// Returns
/// -------
/// numpy.ndarray
///     Basis matrix of shape (n, df). All values are in [0, 1].
///     Each column is monotonically increasing (or decreasing) in x.
///
/// Notes
/// -----
/// To fit a monotonic curve:
/// 1. Compute the I-spline basis: basis = ms(x, df=5)
/// 2. Fit model with non-negative coefficient constraint
/// 3. The fitted curve will be monotonic
///
/// For actuarial applications, this is useful for:
/// - Age effects that should increase with age
/// - Vehicle age effects that should decrease claim frequency
/// - Any relationship where business logic dictates monotonicity
///
/// Examples
/// --------
/// >>> import rustystats as rs
/// >>> import numpy as np
/// >>> x = np.linspace(0, 10, 100)
/// >>> basis = rs.ms(x, df=5)  # Monotonically increasing
/// >>> print(basis.shape)
/// (100, 5)
/// >>> print(f"All values in [0, 1]: {basis.min() >= 0 and basis.max() <= 1}")
/// True
#[pyfunction]
#[pyo3(signature = (x, df, degree=3, boundary_knots=None, increasing=true))]
pub fn ms_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    df: usize,
    degree: usize,
    boundary_knots: Option<(f64, f64)>,
    increasing: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let x_array = x.as_array().to_owned();
    let result = splines::is_basis(&x_array, df, degree, boundary_knots, increasing);
    Ok(result.into_pyarray_bound(py))
}

/// Compute I-spline (monotonic spline) basis with explicit interior knots.
///
/// Essential for prediction on new data where knots must match training.
#[pyfunction]
#[pyo3(signature = (x, interior_knots, degree, boundary_knots, df, increasing=true))]
pub fn ms_with_knots_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    interior_knots: Vec<f64>,
    degree: usize,
    boundary_knots: (f64, f64),
    df: usize,
    increasing: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let x_array = x.as_array().to_owned();
    let result = splines::is_basis_with_knots(&x_array, &interior_knots, degree, boundary_knots, df, increasing);
    Ok(result.into_pyarray_bound(py))
}

/// Get column names for I-spline (monotonic spline) basis.
#[pyfunction]
#[pyo3(signature = (var_name, df, increasing=true))]
pub fn ms_names_py(var_name: &str, df: usize, increasing: bool) -> Vec<String> {
    splines::is_names(var_name, df, increasing)
}

/// Compute knots for B-splines from data.
///
/// Returns (interior_knots, boundary_knots) where interior knots are placed
/// at quantiles of the data and boundary knots are (min, max).
///
/// Parameters
/// ----------
/// x : numpy.ndarray
///     Data points (1D array)
/// df : int
///     Degrees of freedom (number of basis functions)
/// degree : int, optional
///     Spline degree. Default 3 (cubic).
/// boundary_knots : tuple, optional
///     (min, max) boundary knots. If None, uses data range.
///
/// Returns
/// -------
/// tuple
///     (interior_knots: list[float], boundary_knots: tuple[float, float])
#[pyfunction]
#[pyo3(signature = (x, df, degree=3, boundary_knots=None))]
pub fn compute_knots_py(
    x: PyReadonlyArray1<f64>,
    df: usize,
    degree: usize,
    boundary_knots: Option<(f64, f64)>,
) -> PyResult<(Vec<f64>, (f64, f64))> {
    let x_array = x.as_array().to_owned();
    let full_knots = splines::compute_knots(&x_array, df, degree, boundary_knots);

    // Extract boundary knots (first and last values)
    let x_min = full_knots[0];
    let x_max = full_knots[full_knots.len() - 1];

    // Extract interior knots (everything between the repeated boundary knots)
    let interior = full_knots[degree + 1..full_knots.len() - degree - 1].to_vec();

    Ok((interior, (x_min, x_max)))
}

/// Compute knots for natural splines from data.
///
/// Returns (interior_knots, boundary_knots).
///
/// Parameters
/// ----------
/// x : numpy.ndarray
///     Data points (1D array)
/// df : int
///     Degrees of freedom (number of basis functions)
/// boundary_knots : tuple, optional
///     (min, max) boundary knots. If None, uses data range.
///
/// Returns
/// -------
/// tuple
///     (interior_knots: list[float], boundary_knots: tuple[float, float])
#[pyfunction]
#[pyo3(signature = (x, df, boundary_knots=None))]
pub fn compute_knots_natural_py(
    x: PyReadonlyArray1<f64>,
    df: usize,
    boundary_knots: Option<(f64, f64)>,
) -> PyResult<(Vec<f64>, (f64, f64))> {
    let x_array = x.as_array().to_owned();
    let (interior, x_min, x_max) = splines::compute_knots_natural(&x_array, df, boundary_knots);
    Ok((interior, (x_min, x_max)))
}
