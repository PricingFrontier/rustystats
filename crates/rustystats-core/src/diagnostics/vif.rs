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
use ndarray::{Array1, Array2, ArrayView2};
use rayon::prelude::*;

/// Rows processed per rayon task when accumulating Gram/sum moments.
/// Sized so each chunk fits comfortably in L2 for typical `k ≤ 256`.
const GRAM_CHUNK_ROWS: usize = 8192;
const SPARSE_GRAM_DENSITY_THRESHOLD: f64 = 0.35;

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
fn inverse_diagonal_spd(m: &Array2<f64>) -> Array1<f64> {
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

/// Compute the Pearson correlation matrix of the columns of `x` (n x k).
///
/// Equivalent to `numpy.corrcoef(x, rowvar=False)` but without materializing a
/// mean-centered copy of `x`. We compute, in a single row-major pass over `x`:
///   * column sums (length k)
///   * the upper triangle of the Gram-style matrix `G_ij = Σ_r x_ri · x_rj`
///
/// then derive covariance and correlation from these moments. Memory cost is
/// O(k²) instead of numpy's O(n·k) mean-centered intermediate.
///
/// Numerical formula (population denominator cancels in the ratio):
///   cov_ij  = G_ij − n · mean_i · mean_j
///   var_i   = G_ii − n · mean_i²
///   corr_ij = cov_ij / sqrt(var_i · var_j)
///
/// For columns with zero variance, the entire row/column is set to 0 (matching
/// the calling Python convention that downstream code substitutes 0 for the
/// NaNs `numpy.corrcoef` would otherwise produce).
///
/// Performance: row-major iteration is cache-friendly because design matrices
/// arriving from Python are typically C-contiguous. We chunk the rows for
/// rayon parallelism and reduce per-chunk Gram matrices into the final result.
/// Mirrors the IRLS X'WX hot-loop in `solvers::irls::compute_xtwx_xtwz`.
fn correlation_matrix(x: ArrayView2<f64>) -> Array2<f64> {
    let n = x.nrows();
    let k = x.ncols();
    if k == 0 || n == 0 {
        return Array2::zeros((k, k));
    }

    // Use the contiguous fast path when possible: for a C-contiguous (n, k)
    // matrix, slice access lets the compiler generate tight inner loops with
    // no stride bookkeeping. Fall back to ndarray indexing otherwise (e.g.
    // when the caller passes a strided view such as `X[:, 1:]`).
    let (sums, gram_upper) = if let Some(slice) = x.as_slice() {
        compute_sums_and_gram_contiguous(slice, n, k)
    } else {
        compute_sums_and_gram_strided(x, n, k)
    };

    let n_f = n as f64;
    let means: Vec<f64> = sums.iter().map(|s| s / n_f).collect();

    // Variances along the diagonal (G_ii − n·mean²). Clamp tiny negatives
    // from cancellation to zero so the std-dev sqrt is well-defined.
    let variances: Vec<f64> = (0..k)
        .map(|i| {
            let v = gram_upper[i * k + i] - n_f * means[i] * means[i];
            if v > 0.0 {
                v
            } else {
                0.0
            }
        })
        .collect();
    let stds: Vec<f64> = variances.iter().map(|v| v.sqrt()).collect();

    let mut r = Array2::<f64>::zeros((k, k));
    for i in 0..k {
        for j in i..k {
            if stds[i] == 0.0 || stds[j] == 0.0 {
                // Leave row/col at 0; caller may overwrite the diagonal.
                continue;
            }
            let cov = gram_upper[i * k + j] - n_f * means[i] * means[j];
            let denom = stds[i] * stds[j];
            let corr = (cov / denom).clamp(-1.0, 1.0);
            r[[i, j]] = corr;
            if i != j {
                r[[j, i]] = corr;
            }
        }
    }
    r
}

/// Accumulate the column sums and upper-triangle Gram moments for `x`.
///
/// This exposes the same sparse-aware moment kernel used by
/// `correlation_matrix` so callers can stream row chunks, sum the returned
/// moments, and derive a single full-data correlation matrix without
/// materializing the full design matrix.
pub fn correlation_moments(x: ArrayView2<f64>) -> (Array1<f64>, Array2<f64>) {
    let n = x.nrows();
    let k = x.ncols();
    if k == 0 {
        return (Array1::zeros(0), Array2::zeros((0, 0)));
    }

    let (sums, gram_upper) = if n == 0 {
        (vec![0.0_f64; k], vec![0.0_f64; k * k])
    } else if let Some(slice) = x.as_slice() {
        compute_sums_and_gram_contiguous(slice, n, k)
    } else {
        compute_sums_and_gram_strided(x, n, k)
    };

    let gram = Array2::from_shape_vec((k, k), gram_upper)
        .expect("internal error: gram upper triangle shape must be k x k");
    (Array1::from_vec(sums), gram)
}

/// Fast path for C-contiguous design matrices (slice access).
/// Returns `(column_sums, gram_upper_flat)` where `gram_upper_flat[i*k + j]`
/// holds `Σ_r x_ri · x_rj` for `i <= j` (lower triangle is left at 0).
fn compute_sums_and_gram_contiguous(x_slice: &[f64], n: usize, k: usize) -> (Vec<f64>, Vec<f64>) {
    debug_assert_eq!(x_slice.len(), n * k);

    if should_use_sparse_gram_kernel_contiguous(x_slice, n, k) {
        return compute_sums_and_gram_contiguous_sparse(x_slice, n, k);
    }

    let num_chunks = n.div_ceil(GRAM_CHUNK_ROWS);

    (0..num_chunks)
        .into_par_iter()
        .map(|chunk_idx| {
            let chunk_start = chunk_idx * GRAM_CHUNK_ROWS;
            let chunk_end = (chunk_start + GRAM_CHUNK_ROWS).min(n);
            let mut sums_local = vec![0.0_f64; k];
            let mut gram_local = vec![0.0_f64; k * k];

            for r in chunk_start..chunk_end {
                let row_start = r * k;
                for i in 0..k {
                    // SAFETY: row_start + i < n*k = x_slice.len()
                    let xri = unsafe { *x_slice.get_unchecked(row_start + i) };
                    // SAFETY: i < k = sums_local.len()
                    unsafe { *sums_local.get_unchecked_mut(i) += xri };
                    for j in i..k {
                        // SAFETY: row_start + j < n*k = x_slice.len()
                        let xrj = unsafe { *x_slice.get_unchecked(row_start + j) };
                        // SAFETY: i*k + j < k*k = gram_local.len() (i, j < k)
                        unsafe { *gram_local.get_unchecked_mut(i * k + j) += xri * xrj };
                    }
                }
            }
            (sums_local, gram_local)
        })
        .reduce(
            || (vec![0.0_f64; k], vec![0.0_f64; k * k]),
            |(mut a_sums, mut a_gram), (b_sums, b_gram)| {
                for i in 0..a_sums.len() {
                    a_sums[i] += b_sums[i];
                }
                for i in 0..a_gram.len() {
                    a_gram[i] += b_gram[i];
                }
                (a_sums, a_gram)
            },
        )
}

fn should_use_sparse_gram_kernel_contiguous(x_slice: &[f64], n: usize, k: usize) -> bool {
    if n == 0 || k < 16 {
        return false;
    }

    let sample_rows = n.min(1024);
    let mut nonzero = 0usize;
    for sample_idx in 0..sample_rows {
        let row = sample_idx * n / sample_rows;
        let row_start = row * k;
        for j in 0..k {
            if x_slice[row_start + j] != 0.0 {
                nonzero += 1;
            }
        }
    }
    let density = nonzero as f64 / (sample_rows * k) as f64;
    density <= SPARSE_GRAM_DENSITY_THRESHOLD
}

fn compute_sums_and_gram_contiguous_sparse(
    x_slice: &[f64],
    n: usize,
    k: usize,
) -> (Vec<f64>, Vec<f64>) {
    let num_chunks = n.div_ceil(GRAM_CHUNK_ROWS);

    (0..num_chunks)
        .into_par_iter()
        .map(|chunk_idx| {
            let chunk_start = chunk_idx * GRAM_CHUNK_ROWS;
            let chunk_end = (chunk_start + GRAM_CHUNK_ROWS).min(n);
            let mut sums_local = vec![0.0_f64; k];
            let mut gram_local = vec![0.0_f64; k * k];
            let mut nz_idx: Vec<usize> = Vec::with_capacity(k);
            let mut nz_val: Vec<f64> = Vec::with_capacity(k);

            for r in chunk_start..chunk_end {
                nz_idx.clear();
                nz_val.clear();
                let row_start = r * k;
                for i in 0..k {
                    // SAFETY: row_start + i < n*k = x_slice.len()
                    let xri = unsafe { *x_slice.get_unchecked(row_start + i) };
                    if xri != 0.0 {
                        // SAFETY: i < k = sums_local.len()
                        unsafe { *sums_local.get_unchecked_mut(i) += xri };
                        nz_idx.push(i);
                        nz_val.push(xri);
                    }
                }

                for a in 0..nz_idx.len() {
                    // SAFETY: a is within nz vectors populated above.
                    let i = unsafe { *nz_idx.get_unchecked(a) };
                    let xri = unsafe { *nz_val.get_unchecked(a) };
                    for b in a..nz_idx.len() {
                        // SAFETY: b is within nz vectors populated above.
                        let j = unsafe { *nz_idx.get_unchecked(b) };
                        let xrj = unsafe { *nz_val.get_unchecked(b) };
                        // SAFETY: i, j < k, so i*k + j < k*k.
                        unsafe { *gram_local.get_unchecked_mut(i * k + j) += xri * xrj };
                    }
                }
            }
            (sums_local, gram_local)
        })
        .reduce(
            || (vec![0.0_f64; k], vec![0.0_f64; k * k]),
            |(mut a_sums, mut a_gram), (b_sums, b_gram)| {
                for i in 0..a_sums.len() {
                    a_sums[i] += b_sums[i];
                }
                for i in 0..a_gram.len() {
                    a_gram[i] += b_gram[i];
                }
                (a_sums, a_gram)
            },
        )
}

/// Strided fallback for non-contiguous matrices (e.g. `X[:, 1:]` slices that
/// hop a stride per row). Same algorithm but uses ndarray indexing instead of
/// raw slice access, so the compiler can't elide stride math.
fn compute_sums_and_gram_strided(x: ArrayView2<f64>, n: usize, k: usize) -> (Vec<f64>, Vec<f64>) {
    if should_use_sparse_gram_kernel_strided(x, n, k) {
        return compute_sums_and_gram_strided_sparse(x, n, k);
    }

    let num_chunks = n.div_ceil(GRAM_CHUNK_ROWS);

    (0..num_chunks)
        .into_par_iter()
        .map(|chunk_idx| {
            let chunk_start = chunk_idx * GRAM_CHUNK_ROWS;
            let chunk_end = (chunk_start + GRAM_CHUNK_ROWS).min(n);
            let mut sums_local = vec![0.0_f64; k];
            let mut gram_local = vec![0.0_f64; k * k];

            for r in chunk_start..chunk_end {
                for i in 0..k {
                    let xri = x[[r, i]];
                    sums_local[i] += xri;
                    for j in i..k {
                        gram_local[i * k + j] += xri * x[[r, j]];
                    }
                }
            }
            (sums_local, gram_local)
        })
        .reduce(
            || (vec![0.0_f64; k], vec![0.0_f64; k * k]),
            |(mut a_sums, mut a_gram), (b_sums, b_gram)| {
                for i in 0..a_sums.len() {
                    a_sums[i] += b_sums[i];
                }
                for i in 0..a_gram.len() {
                    a_gram[i] += b_gram[i];
                }
                (a_sums, a_gram)
            },
        )
}

fn should_use_sparse_gram_kernel_strided(x: ArrayView2<f64>, n: usize, k: usize) -> bool {
    if n == 0 || k < 16 {
        return false;
    }

    let sample_rows = n.min(1024);
    let mut nonzero = 0usize;
    for sample_idx in 0..sample_rows {
        let row = sample_idx * n / sample_rows;
        for j in 0..k {
            if x[[row, j]] != 0.0 {
                nonzero += 1;
            }
        }
    }
    let density = nonzero as f64 / (sample_rows * k) as f64;
    density <= SPARSE_GRAM_DENSITY_THRESHOLD
}

fn compute_sums_and_gram_strided_sparse(
    x: ArrayView2<f64>,
    n: usize,
    k: usize,
) -> (Vec<f64>, Vec<f64>) {
    let num_chunks = n.div_ceil(GRAM_CHUNK_ROWS);

    (0..num_chunks)
        .into_par_iter()
        .map(|chunk_idx| {
            let chunk_start = chunk_idx * GRAM_CHUNK_ROWS;
            let chunk_end = (chunk_start + GRAM_CHUNK_ROWS).min(n);
            let mut sums_local = vec![0.0_f64; k];
            let mut gram_local = vec![0.0_f64; k * k];
            let mut nz_idx: Vec<usize> = Vec::with_capacity(k);
            let mut nz_val: Vec<f64> = Vec::with_capacity(k);

            for r in chunk_start..chunk_end {
                nz_idx.clear();
                nz_val.clear();
                for i in 0..k {
                    let xri = x[[r, i]];
                    if xri != 0.0 {
                        sums_local[i] += xri;
                        nz_idx.push(i);
                        nz_val.push(xri);
                    }
                }

                for a in 0..nz_idx.len() {
                    let i = nz_idx[a];
                    let xri = nz_val[a];
                    for b in a..nz_idx.len() {
                        let j = nz_idx[b];
                        let xrj = nz_val[b];
                        gram_local[i * k + j] += xri * xrj;
                    }
                }
            }
            (sums_local, gram_local)
        })
        .reduce(
            || (vec![0.0_f64; k], vec![0.0_f64; k * k]),
            |(mut a_sums, mut a_gram), (b_sums, b_gram)| {
                for i in 0..a_sums.len() {
                    a_sums[i] += b_sums[i];
                }
                for i in 0..a_gram.len() {
                    a_gram[i] += b_gram[i];
                }
                (a_sums, a_gram)
            },
        )
}

/// Combined correlation-matrix + VIF computation used by the Python diagnostics.
///
/// Returns `(R, vif_diagonal)` where:
///   R is the k x k Pearson correlation matrix (zero-variance columns get
///     an all-zero row/col with 0 on the diagonal — caller may overwrite),
///   vif_diagonal is `diag((R + ε·I)^{-1})` of length k. Zero-variance columns
///     are skipped: the regularized inverse will yield ≈ 1/ε at that diagonal,
///     which the caller maps to a "severe" VIF.
///
/// Computing both inside one Rust call lets the Python side avoid materializing
/// any (n, k) intermediate (numpy.corrcoef centers in-place and copies X), so
/// transient memory drops from O(n·k) to O(k²).
pub fn correlation_and_vif(x: ArrayView2<f64>, epsilon: f64) -> (Array2<f64>, Array1<f64>) {
    let r = correlation_matrix(x);
    let k = r.nrows();
    if k == 0 {
        return (r, Array1::zeros(0));
    }
    let mut r_reg = r.clone();
    for i in 0..k {
        r_reg[[i, i]] += epsilon;
    }
    let vif = inverse_diagonal_spd(&r_reg);
    (r, vif)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{array, s};

    fn naive_correlation_matrix(x: ArrayView2<f64>) -> Array2<f64> {
        let n = x.nrows();
        let k = x.ncols();
        let mut r = Array2::<f64>::zeros((k, k));
        let mut means = vec![0.0; k];
        for j in 0..k {
            means[j] = (0..n).map(|i| x[[i, j]]).sum::<f64>() / n as f64;
        }

        for i in 0..k {
            for j in i..k {
                let mut cov = 0.0;
                let mut var_i = 0.0;
                let mut var_j = 0.0;
                for row in 0..n {
                    let di = x[[row, i]] - means[i];
                    let dj = x[[row, j]] - means[j];
                    cov += di * dj;
                    var_i += di * di;
                    var_j += dj * dj;
                }
                if var_i > 0.0 && var_j > 0.0 {
                    let corr = (cov / (var_i.sqrt() * var_j.sqrt())).clamp(-1.0, 1.0);
                    r[[i, j]] = corr;
                    r[[j, i]] = corr;
                }
            }
        }
        r
    }

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

    #[test]
    fn test_inverse_diagonal_lu_fallback_for_indefinite_invertible_matrix() {
        // Symmetric but indefinite: Cholesky fails, LU inverse succeeds.
        let m = array![[0.0, 2.0], [2.0, 0.0]];
        let diag = inverse_diagonal_spd(&m);
        assert_abs_diff_eq!(diag[0], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(diag[1], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_correlation_matrix_independent_columns() {
        // Independent uniform-ish columns — diag is 1, off-diag near 0.
        // Use a small deterministic dataset so we can compare to numpy by hand.
        // Columns: x1 = [1,2,3,4,5,6,7,8], x2 = [8,7,6,5,4,3,2,1]
        // -> perfect negative correlation (-1).
        let x = array![
            [1.0, 8.0],
            [2.0, 7.0],
            [3.0, 6.0],
            [4.0, 5.0],
            [5.0, 4.0],
            [6.0, 3.0],
            [7.0, 2.0],
            [8.0, 1.0]
        ];
        let r = correlation_matrix(x.view());
        assert_abs_diff_eq!(r[[0, 0]], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r[[1, 1]], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r[[0, 1]], -1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r[[1, 0]], -1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_correlation_matrix_matches_numpy_formula() {
        // Verify the standard sample correlation formula on a 3-column dataset.
        // x1 = [1,2,3,4,5], x2 = [2,4,5,4,5], x3 = [5,4,3,2,1]
        let x = array![
            [1.0, 2.0, 5.0],
            [2.0, 4.0, 4.0],
            [3.0, 5.0, 3.0],
            [4.0, 4.0, 2.0],
            [5.0, 5.0, 1.0]
        ];
        let r = correlation_matrix(x.view());
        // Hand-computed:
        // mean1=3, mean2=4, mean3=3
        // var1=10, var2=6, var3=10  (sum of squared deviations)
        // cov12 = 6, cov13 = -10
        // corr12 = 6 / sqrt(10*6) = 6/sqrt(60) = 0.7745966692414834
        // corr13 = -10 / sqrt(100) = -1
        assert_abs_diff_eq!(r[[0, 0]], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r[[0, 1]], 6.0 / (60.0_f64).sqrt(), epsilon = 1e-12);
        assert_abs_diff_eq!(r[[0, 2]], -1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r[[1, 0]], r[[0, 1]], epsilon = 1e-12);
        assert_abs_diff_eq!(r[[2, 0]], r[[0, 2]], epsilon = 1e-12);
    }

    #[test]
    fn test_correlation_matrix_zero_variance_column() {
        // Constant column → row/col should be all zero (no NaN propagation).
        let x = array![[1.0, 5.0], [2.0, 5.0], [3.0, 5.0], [4.0, 5.0]];
        let r = correlation_matrix(x.view());
        assert_abs_diff_eq!(r[[0, 0]], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r[[1, 1]], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r[[0, 1]], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r[[1, 0]], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_sparse_correlation_kernel_matches_dense_formula() {
        let n = 64;
        let k = 32;
        let mut x = Array2::<f64>::zeros((n, k));
        for row in 0..n {
            let j1 = row % k;
            let j2 = (row * 7 + 3) % k;
            x[[row, j1]] = 1.0 + (row % 5) as f64;
            x[[row, j2]] += (row as f64 / 3.0).sin();
        }

        assert!(should_use_sparse_gram_kernel_contiguous(
            x.as_slice().expect("test matrix should be contiguous"),
            n,
            k
        ));

        let actual = correlation_matrix(x.view());
        let expected = naive_correlation_matrix(x.view());
        for i in 0..k {
            for j in 0..k {
                assert_abs_diff_eq!(actual[[i, j]], expected[[i, j]], epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_sparse_strided_correlation_kernel_matches_dense_formula() {
        let n = 64;
        let k = 33;
        let mut x = Array2::<f64>::zeros((n, k));
        for row in 0..n {
            let j1 = 1 + row % (k - 1);
            let j2 = 1 + (row * 5 + 2) % (k - 1);
            x[[row, j1]] = 2.0 + (row % 3) as f64;
            x[[row, j2]] += (row as f64 / 5.0).cos();
        }

        let view = x.slice(s![.., 1..]);
        assert!(should_use_sparse_gram_kernel_strided(
            view,
            view.nrows(),
            view.ncols()
        ));

        let actual = correlation_matrix(view);
        let expected = naive_correlation_matrix(view);
        for i in 0..view.ncols() {
            for j in 0..view.ncols() {
                assert_abs_diff_eq!(actual[[i, j]], expected[[i, j]], epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_correlation_moments_empty_and_dense_strided_contracts() {
        let zero_cols = Array2::<f64>::zeros((5, 0));
        let (sums, gram) = correlation_moments(zero_cols.view());
        assert_eq!(sums.len(), 0);
        assert_eq!(gram.dim(), (0, 0));

        let zero_rows = Array2::<f64>::zeros((0, 3));
        let (sums, gram) = correlation_moments(zero_rows.view());
        assert_eq!(
            sums.as_slice().expect("contiguous result"),
            &[0.0, 0.0, 0.0]
        );
        assert_eq!(gram.dim(), (3, 3));
        assert!(gram.iter().all(|&v| v == 0.0));

        let x = array![
            [9.0, 1.0, 4.0, 2.0, -1.0, 0.0],
            [8.0, 2.0, 5.0, 3.0, -2.0, 1.0],
            [7.0, 3.0, 7.0, 5.0, -3.0, 2.0],
            [6.0, 4.0, 8.0, 7.0, -4.0, 3.0],
        ];
        let view = x.slice(s![.., 1..5]);
        assert!(!should_use_sparse_gram_kernel_strided(
            view,
            view.nrows(),
            view.ncols()
        ));

        let (sums, gram) = correlation_moments(view);
        for col in 0..view.ncols() {
            let expected_sum = (0..view.nrows()).map(|row| view[[row, col]]).sum::<f64>();
            assert_abs_diff_eq!(sums[col], expected_sum, epsilon = 1e-12);
            for other in col..view.ncols() {
                let expected_gram = (0..view.nrows())
                    .map(|row| view[[row, col]] * view[[row, other]])
                    .sum::<f64>();
                assert_abs_diff_eq!(gram[[col, other]], expected_gram, epsilon = 1e-12);
            }
        }

        let actual = correlation_matrix(view);
        let expected = naive_correlation_matrix(view);
        for i in 0..view.ncols() {
            for j in 0..view.ncols() {
                assert_abs_diff_eq!(actual[[i, j]], expected[[i, j]], epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_correlation_moments_sparse_contiguous_and_strided_match_naive_gram() {
        let n = 96;
        let k = 24;
        let mut x = Array2::<f64>::zeros((n, k + 1));
        for row in 0..n {
            let c1 = 1 + row % k;
            let c2 = 1 + (row * 11 + 7) % k;
            x[[row, c1]] = 1.0 + (row % 4) as f64;
            x[[row, c2]] += 0.5 + (row % 9) as f64 / 3.0;
        }

        let contiguous = x.slice(s![.., 1..]).to_owned();
        let (sums, gram) = correlation_moments(contiguous.view());
        for col in 0..contiguous.ncols() {
            let expected_sum = (0..n).map(|row| contiguous[[row, col]]).sum::<f64>();
            assert_abs_diff_eq!(sums[col], expected_sum, epsilon = 1e-12);
            for other in col..contiguous.ncols() {
                let expected_gram = (0..n)
                    .map(|row| contiguous[[row, col]] * contiguous[[row, other]])
                    .sum::<f64>();
                assert_abs_diff_eq!(gram[[col, other]], expected_gram, epsilon = 1e-12);
            }
        }

        let strided = x.slice(s![.., 1..]);
        let (sums, gram) = correlation_moments(strided);
        for col in 0..strided.ncols() {
            let expected_sum = (0..n).map(|row| strided[[row, col]]).sum::<f64>();
            assert_abs_diff_eq!(sums[col], expected_sum, epsilon = 1e-12);
            for other in col..strided.ncols() {
                let expected_gram = (0..n)
                    .map(|row| strided[[row, col]] * strided[[row, other]])
                    .sum::<f64>();
                assert_abs_diff_eq!(gram[[col, other]], expected_gram, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_correlation_and_vif_combined() {
        // 2-column case with corr 0.5 → VIF = 1/(1-0.25) = 4/3 (ε is tiny).
        // Choose data with corr exactly 0.5: the small dataset
        //   x1 = [-1, 0, 1, 2], x2 = [0, 1, 1, 2] gives roughly that.
        // For an exact answer, use synthetic columns: x1 = a, x2 = a+b
        // where corr(a, a+b) is whatever. We instead just verify the
        // produced matrix matches a computed-by-hand correlation and that
        // VIF = diag((R + εI)^{-1}).
        let x = array![[1.0, 1.0], [2.0, 3.0], [3.0, 5.0], [4.0, 7.0]];
        // x2 = 2*x1 - 1 -> perfect collinearity; VIF blows up but with ε it
        // resolves to ~ 1/(2ε) on each diagonal. Verify zero-variance is not
        // triggered (both columns vary), R is well-formed, and ε regularizes.
        let (r, vif) = correlation_and_vif(x.view(), 1e-10);
        assert_abs_diff_eq!(r[[0, 0]], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r[[1, 1]], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r[[0, 1]], 1.0, epsilon = 1e-12);
        assert!(vif[0].is_finite() && vif[1].is_finite());
        assert!(vif[0] > 1e8); // collinear → very large VIF after tiny ε
    }

    #[test]
    fn test_inverse_diagonal_spd_empty() {
        // 0×0 matrix — should return empty diagonal, not panic.
        let m = Array2::<f64>::zeros((0, 0));
        let diag = inverse_diagonal_spd(&m);
        assert_eq!(diag.len(), 0);
    }

    #[test]
    fn test_correlation_matrix_n_zero() {
        // 0 rows × k columns — degenerate input. The early-return preserves
        // shape `(k, k)` (filled with zeros) rather than allocating moments
        // from no data, so callers always get a square output of the
        // expected dimension.
        let x = Array2::<f64>::zeros((0, 3));
        let r = correlation_matrix(x.view());
        assert_eq!(r.dim(), (3, 3));
        assert!(r.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_correlation_and_vif_k_zero() {
        // n rows × 0 columns — no features. Should return empty diag/R
        // without dividing by zero.
        let x = Array2::<f64>::zeros((10, 0));
        let (r, vif) = correlation_and_vif(x.view(), 1e-8);
        assert_eq!(r.dim(), (0, 0));
        assert_eq!(vif.len(), 0);
    }
}
