// =============================================================================
// ndarray ↔ nalgebra Conversion Utilities
// =============================================================================
//
// This module centralizes all conversions between ndarray (used for PyO3/numpy
// interop and array storage) and nalgebra (used for linear algebra operations).
//
// Previously, these conversions were duplicated ~15 times across solvers,
// splines, and GCV optimizer modules with verbose element-by-element loops.
//
// =============================================================================

use ndarray::{Array1, Array2};
use nalgebra::{DMatrix, DVector};

// =============================================================================
// ndarray → nalgebra
// =============================================================================

/// Convert an ndarray Array2 to a nalgebra DMatrix.
///
/// Handles non-contiguous arrays by making a contiguous copy first.
#[inline]
pub fn to_dmatrix(a: &Array2<f64>) -> DMatrix<f64> {
    let (nrows, ncols) = (a.nrows(), a.ncols());
    let contig = if a.is_standard_layout() {
        a.clone()
    } else {
        a.as_standard_layout().to_owned()
    };
    DMatrix::from_row_slice(nrows, ncols, contig.as_slice().unwrap())
}

/// Convert an ndarray Array1 to a nalgebra DVector.
#[inline]
pub fn to_dvector(v: &Array1<f64>) -> DVector<f64> {
    DVector::from_row_slice(v.as_slice().unwrap_or(&v.to_vec()))
}

// =============================================================================
// nalgebra → ndarray
// =============================================================================

/// Convert a nalgebra DMatrix to an ndarray Array2.
#[inline]
pub fn to_array2(m: &DMatrix<f64>) -> Array2<f64> {
    let (nrows, ncols) = m.shape();
    let mut result = Array2::zeros((nrows, ncols));
    for i in 0..nrows {
        for j in 0..ncols {
            result[[i, j]] = m[(i, j)];
        }
    }
    result
}

/// Convert a nalgebra DVector to an ndarray Array1.
#[inline]
pub fn to_array1(v: &DVector<f64>) -> Array1<f64> {
    Array1::from_vec(v.as_slice().to_vec())
}

// =============================================================================
// Common linear algebra operations (ndarray in, ndarray out)
// =============================================================================

/// Solve a symmetric positive-definite system Ax = b.
///
/// Tries Cholesky first, falls back to LU decomposition.
/// Operates on ndarray types, handling conversion internally.
pub fn solve_symmetric(a: &Array2<f64>, b: &Array1<f64>) -> Option<Array1<f64>> {
    let a_nalg = to_dmatrix(a);
    let b_nalg = to_dvector(b);

    if let Some(chol) = a_nalg.clone().cholesky() {
        Some(to_array1(&chol.solve(&b_nalg)))
    } else {
        a_nalg.lu().solve(&b_nalg).map(|x| to_array1(&x))
    }
}

/// Invert a square matrix.
///
/// Returns None if the matrix is singular.
pub fn invert_matrix(a: &Array2<f64>) -> Option<Array2<f64>> {
    let a_nalg = to_dmatrix(a);
    a_nalg.try_inverse().map(|inv| to_array2(&inv))
}

/// Solve Ax = b and also return A⁻¹, using Cholesky if possible.
///
/// This is the common pattern in WLS solvers where we need both
/// the solution and the inverse for covariance computation.
/// Returns (solution, inverse) or None if singular.
pub fn solve_and_invert(a: &DMatrix<f64>, b: &DVector<f64>, p: usize) -> Option<(Array1<f64>, Array2<f64>)> {
    if let Some(chol) = a.clone().cholesky() {
        let coefficients = chol.solve(b);
        let identity = DMatrix::identity(p, p);
        let inv = chol.solve(&identity);
        Some((to_array1(&coefficients), to_array2(&inv)))
    } else {
        // Fall back to LU
        let lu = a.clone().lu();
        let sol = lu.solve(b)?;
        let inv = a.clone().try_inverse()?;
        Some((to_array1(&sol), to_array2(&inv)))
    }
}

/// Solve AX = B where A is symmetric and B is a matrix.
///
/// Used for computing hat matrices: (X'WX + λS)⁻¹ X'WX.
/// Tries Cholesky, then LU, then SVD pseudo-inverse.
pub fn solve_symmetric_matrix(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let n = a.nrows();
    let a_nalg = to_dmatrix(a);
    let b_nalg = to_dmatrix(b);

    // Try Cholesky first
    if let Some(chol) = a_nalg.clone().cholesky() {
        let sol = chol.solve(&b_nalg);
        return to_array2(&sol);
    }

    // Fall back to LU via inverse
    if let Some(a_inv) = a_nalg.clone().try_inverse() {
        let sol = a_inv * b_nalg;
        return to_array2(&sol);
    }

    // Last resort: SVD pseudo-inverse
    let svd = a_nalg.svd(true, true);
    let u = svd.u.unwrap();
    let v_t = svd.v_t.unwrap();
    let s = svd.singular_values;
    let tol = 1e-10 * s[0];
    let mut s_inv = DMatrix::zeros(n, n);
    for i in 0..n {
        if s[i] > tol {
            s_inv[(i, i)] = 1.0 / s[i];
        }
    }
    let a_pinv = v_t.transpose() * s_inv * u.transpose();
    let sol = a_pinv * b_nalg;
    to_array2(&sol)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_roundtrip_matrix() {
        let a = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let m = to_dmatrix(&a);
        let back = to_array2(&m);
        assert_eq!(a, back);
    }

    #[test]
    fn test_roundtrip_vector() {
        let v = array![1.0, 2.0, 3.0];
        let dv = to_dvector(&v);
        let back = to_array1(&dv);
        assert_eq!(v, back);
    }

    #[test]
    fn test_solve_symmetric_identity() {
        let a = Array2::eye(3);
        let b = array![1.0, 2.0, 3.0];
        let x = solve_symmetric(&a, &b).unwrap();
        for i in 0..3 {
            assert!((x[i] - b[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn test_invert_identity() {
        let a = Array2::eye(3);
        let inv = invert_matrix(&a).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((inv[[i, j]] - expected).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_solve_and_invert() {
        let a = DMatrix::from_row_slice(2, 2, &[4.0, 1.0, 1.0, 3.0]);
        let b = DVector::from_row_slice(&[5.0, 4.0]);
        let (sol, inv) = solve_and_invert(&a, &b, 2).unwrap();
        // A * x should equal b
        let ax0 = 4.0 * sol[0] + 1.0 * sol[1];
        let ax1 = 1.0 * sol[0] + 3.0 * sol[1];
        assert!((ax0 - 5.0).abs() < 1e-10);
        assert!((ax1 - 4.0).abs() < 1e-10);
        // A * A^-1 should be identity
        assert!((inv[[0, 0]] * 4.0 + inv[[0, 1]] * 1.0 - 1.0).abs() < 1e-10);
    }
}
