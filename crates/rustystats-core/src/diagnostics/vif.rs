// =============================================================================
// VIF (Variance Inflation Factor) Helpers
// =============================================================================
//
// VIF for column j of a design matrix is the j-th diagonal of R^{-1}, where R
// is the correlation matrix of the non-intercept columns. We avoid forming the
// full inverse explicitly by using a Cholesky factorization of the (small,
// symmetric, positive-definite) k x k matrix.
//
// =============================================================================

use nalgebra::{Cholesky, DMatrix};
use ndarray::{Array1, Array2};

/// Compute the diagonal of M^{-1} where M is symmetric positive-definite (n x n).
///
/// Uses Cholesky decomposition: factors M = L L^T, then computes the inverse
/// from the factorization and extracts diag(M^{-1}). Falls back to LU inverse
/// if Cholesky fails (e.g., M is not strictly PD due to numerical edge cases).
///
/// If neither factorization succeeds, returns an Array1 filled with NaN so the
/// caller can detect the failure.
///
/// Returns the n diagonal values of M^{-1} as an Array1<f64>.
pub fn inverse_diagonal_spd(m: &Array2<f64>) -> Array1<f64> {
    let n = m.nrows();
    debug_assert_eq!(n, m.ncols(), "matrix must be square");

    if n == 0 {
        return Array1::zeros(0);
    }

    let mat = DMatrix::from_fn(n, n, |i, j| m[[i, j]]);

    let inv = match Cholesky::new(mat.clone()) {
        Some(chol) => chol.inverse(),
        None => match mat.try_inverse() {
            Some(m_inv) => m_inv,
            None => {
                return Array1::from_elem(n, f64::NAN);
            }
        },
    };

    Array1::from_iter((0..n).map(|i| inv[(i, i)]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_identity_matrix() {
        let i = Array2::eye(4);
        let diag = inverse_diagonal_spd(&i);
        for v in diag.iter() {
            assert_abs_diff_eq!(*v, 1.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_diagonal_matrix() {
        // Diagonal matrix [2, 4, 8] -> inverse diagonal [0.5, 0.25, 0.125]
        let mut m = Array2::zeros((3, 3));
        m[[0, 0]] = 2.0;
        m[[1, 1]] = 4.0;
        m[[2, 2]] = 8.0;

        let diag = inverse_diagonal_spd(&m);
        assert_abs_diff_eq!(diag[0], 0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(diag[1], 0.25, epsilon = 1e-12);
        assert_abs_diff_eq!(diag[2], 0.125, epsilon = 1e-12);
    }

    #[test]
    fn test_correlation_like_matrix() {
        // 2x2 correlation matrix with corr 0.5:
        // R = [[1, 0.5], [0.5, 1]] -> R^{-1} = (1/0.75) * [[1, -0.5], [-0.5, 1]]
        // diag = [1/0.75, 1/0.75] = [4/3, 4/3]
        let r = array![[1.0, 0.5], [0.5, 1.0]];
        let diag = inverse_diagonal_spd(&r);
        assert_abs_diff_eq!(diag[0], 4.0 / 3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(diag[1], 4.0 / 3.0, epsilon = 1e-12);
    }

    #[test]
    fn test_singular_matrix_returns_nan() {
        // All-zero matrix has no inverse; Cholesky and LU both fail.
        let m = Array2::<f64>::zeros((3, 3));
        let diag = inverse_diagonal_spd(&m);
        assert!(diag.iter().all(|v| v.is_nan()));
    }
}
