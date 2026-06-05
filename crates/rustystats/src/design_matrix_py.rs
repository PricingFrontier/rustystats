// =============================================================================
// Design Matrix Functions
// =============================================================================
//
// Fast categorical encoding and interaction construction in Rust.
// =============================================================================

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use rustystats_core::design_matrix;

/// Encode categorical variable from string values.
///
/// Parameters
/// ----------
/// values : list[str]
///     String values for each observation
/// var_name : str
///     Variable name (for column naming)
/// drop_first : bool
///     Whether to drop the first level (reference category)
///
/// Returns
/// -------
/// tuple[numpy.ndarray, list[str], list[int], list[str]]
///     (dummy_matrix, column_names, indices, levels)
#[pyfunction]
#[pyo3(signature = (values, var_name, drop_first=true))]
pub fn encode_categorical_py<'py>(
    py: Python<'py>,
    values: Vec<String>,
    var_name: &str,
    drop_first: bool,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Vec<String>,
    Vec<i32>,
    Vec<String>,
)> {
    let enc = design_matrix::encode_categorical(&values, var_name, drop_first);
    Ok((
        enc.matrix.into_pyarray(py),
        enc.names,
        enc.indices,
        enc.levels,
    ))
}

/// Fast categorical factorization: string values → (sorted unique levels, integer codes).
///
/// O(n) HashMap-based encoding, replacing O(n log n) np.unique for diagnostics/exploration.
///
/// Parameters
/// ----------
/// values : list[str]
///     String values for each observation
///
/// Returns
/// -------
/// tuple[list[str], numpy.ndarray[uint32]]
///     (sorted_unique_levels, integer_codes) matching np.unique(return_inverse=True) output
#[pyfunction]
pub fn factorize_strings_py<'py>(
    py: Python<'py>,
    values: Vec<String>,
) -> PyResult<(Vec<String>, Bound<'py, PyArray1<u32>>)> {
    let (levels, codes) = design_matrix::factorize_strings(&values);
    Ok((levels, codes.into_pyarray(py)))
}

/// Encode categorical from pre-computed indices.
///
/// Use when indices are already computed (e.g., from factorization).
///
/// Parameters
/// ----------
/// indices : numpy.ndarray
///     Pre-computed level indices (0-indexed, int32)
/// n_levels : int
///     Total number of levels
/// level_names : list[str]
///     Names for each level
/// var_name : str
///     Variable name
/// drop_first : bool
///     Drop first level
///
/// Returns
/// -------
/// tuple[numpy.ndarray, list[str]]
///     (dummy_matrix, column_names)
#[pyfunction]
#[pyo3(signature = (indices, n_levels, level_names, var_name, drop_first=true))]
pub fn encode_categorical_indices_py<'py>(
    py: Python<'py>,
    indices: PyReadonlyArray1<i32>,
    n_levels: usize,
    level_names: Vec<String>,
    var_name: &str,
    drop_first: bool,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Vec<String>)> {
    let indices_vec: Vec<i32> = indices.as_array().to_vec();
    let enc = design_matrix::encode_categorical_from_indices(
        &indices_vec,
        n_levels,
        &level_names,
        var_name,
        drop_first,
    );
    Ok((enc.matrix.into_pyarray(py), enc.names))
}

/// Build categorical × categorical interaction matrix.
///
/// Parameters
/// ----------
/// idx1 : numpy.ndarray
///     Level indices for first categorical (0 = reference)
/// n_levels1 : int
///     Number of non-reference levels for first
/// idx2 : numpy.ndarray
///     Level indices for second categorical
/// n_levels2 : int
///     Number of non-reference levels for second
/// names1 : list[str]
///     Column names for first categorical dummies
/// names2 : list[str]
///     Column names for second categorical dummies
///
/// Returns
/// -------
/// tuple[numpy.ndarray, list[str]]
///     (interaction_matrix, column_names)
#[pyfunction]
pub fn build_cat_cat_interaction_py<'py>(
    py: Python<'py>,
    idx1: PyReadonlyArray1<i32>,
    n_levels1: usize,
    idx2: PyReadonlyArray1<i32>,
    n_levels2: usize,
    names1: Vec<String>,
    names2: Vec<String>,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Vec<String>)> {
    let idx1_vec: Vec<i32> = idx1.as_array().to_vec();
    let idx2_vec: Vec<i32> = idx2.as_array().to_vec();
    let (matrix, names) = design_matrix::build_categorical_categorical_interaction(
        &idx1_vec, n_levels1, &idx2_vec, n_levels2, &names1, &names2,
    );
    Ok((matrix.into_pyarray(py), names))
}

/// Build categorical × continuous interaction matrix.
///
/// Parameters
/// ----------
/// cat_indices : numpy.ndarray
///     Level indices for categorical (0 = reference)
/// n_levels : int
///     Number of non-reference levels
/// continuous : numpy.ndarray
///     Continuous variable values
/// cat_names : list[str]
///     Column names for categorical dummies
/// cont_name : str
///     Name of continuous variable
///
/// Returns
/// -------
/// tuple[numpy.ndarray, list[str]]
///     (interaction_matrix, column_names)
#[pyfunction]
pub fn build_cat_cont_interaction_py<'py>(
    py: Python<'py>,
    cat_indices: PyReadonlyArray1<i32>,
    n_levels: usize,
    continuous: PyReadonlyArray1<f64>,
    cat_names: Vec<String>,
    cont_name: &str,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Vec<String>)> {
    let idx_vec: Vec<i32> = cat_indices.as_array().to_vec();
    let cont_array = continuous.as_array().to_owned();
    let (matrix, names) = design_matrix::build_categorical_continuous_interaction(
        &idx_vec,
        n_levels,
        &cont_array,
        &cat_names,
        cont_name,
    );
    Ok((matrix.into_pyarray(py), names))
}

/// Build two-categorical × continuous interaction matrix directly.
///
/// This is equivalent to ``build_cat_cat_interaction_py(...); multiply_matrix_by_continuous_py(...)``
/// but avoids materialising the intermediate dense categorical interaction block.
#[pyfunction]
pub fn build_two_cat_cont_interaction_py<'py>(
    py: Python<'py>,
    idx1: PyReadonlyArray1<i32>,
    n_levels1: usize,
    idx2: PyReadonlyArray1<i32>,
    n_levels2: usize,
    continuous: PyReadonlyArray1<f64>,
    names1: Vec<String>,
    names2: Vec<String>,
    cont_name: &str,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Vec<String>)> {
    let idx1_vec: Vec<i32> = idx1.as_array().to_vec();
    let idx2_vec: Vec<i32> = idx2.as_array().to_vec();
    let cont_array = continuous.as_array().to_owned();

    if idx1_vec.len() != idx2_vec.len() || idx1_vec.len() != cont_array.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "idx1, idx2, and continuous must have matching lengths",
        ));
    }

    let (matrix, names) = design_matrix::build_two_categorical_continuous_interaction(
        &idx1_vec,
        n_levels1,
        &idx2_vec,
        n_levels2,
        &cont_array,
        &names1,
        &names2,
        cont_name,
    );
    Ok((matrix.into_pyarray(py), names))
}

/// Build continuous × continuous interaction.
///
/// Simple element-wise multiplication.
#[pyfunction]
pub fn build_cont_cont_interaction_py<'py>(
    py: Python<'py>,
    x1: PyReadonlyArray1<f64>,
    x2: PyReadonlyArray1<f64>,
    name1: &str,
    name2: &str,
) -> PyResult<(Bound<'py, PyArray1<f64>>, String)> {
    let x1_array = x1.as_array().to_owned();
    let x2_array = x2.as_array().to_owned();
    let (result, name) =
        design_matrix::build_continuous_continuous_interaction(&x1_array, &x2_array, name1, name2);
    Ok((result.into_pyarray(py), name))
}

/// Multiply each column of a matrix by a continuous vector.
///
/// Used for multi-categorical × continuous interactions.
#[pyfunction]
pub fn multiply_matrix_by_continuous_py<'py>(
    py: Python<'py>,
    matrix: PyReadonlyArray2<f64>,
    continuous: PyReadonlyArray1<f64>,
    matrix_names: Vec<String>,
    cont_name: &str,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Vec<String>)> {
    let matrix_array = matrix.as_array().to_owned();
    let cont_array = continuous.as_array().to_owned();
    let (result, names) = design_matrix::multiply_matrix_by_continuous(
        &matrix_array,
        &cont_array,
        &matrix_names,
        cont_name,
    );
    Ok((result.into_pyarray(py), names))
}

/// Stack a list of (n × c_i) f64 column blocks horizontally into a single
/// (n × Σc_i) matrix.
///
/// All blocks must have the same number of rows. Each block must be 2-D and
/// f64-typed; 1-D arrays should be reshaped to (n, 1) on the Python side
/// before calling.
///
/// Parameters
/// ----------
/// blocks : list[numpy.ndarray]
///     List of 2-D float64 arrays sharing the same first dimension.
///
/// Returns
/// -------
/// numpy.ndarray
///     The horizontally-stacked design matrix.
#[pyfunction]
pub fn stack_columns_horizontal_py<'py>(
    py: Python<'py>,
    blocks: Vec<PyReadonlyArray2<'py, f64>>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    if blocks.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "stack_columns_horizontal_py: blocks must be non-empty",
        ));
    }
    let views: Vec<_> = blocks.iter().map(|b| b.as_array()).collect();
    // Release the GIL while doing the (potentially large) memcpy work.
    let result = py.detach(|| design_matrix::stack_columns_horizontal(&views));
    Ok(result.into_pyarray(py))
}
