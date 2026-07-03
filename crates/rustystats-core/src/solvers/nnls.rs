// =============================================================================
// Non-Negative Least Squares (NNLS) Solver
// =============================================================================
//
// Implements the Lawson-Hanson algorithm for solving:
//
//     min ||Ax - b||² subject to x >= 0
//
// This is used for monotonic spline fitting where we need non-negative
// coefficients to guarantee monotonicity when using I-spline basis functions.
//
// Reference:
//   Lawson, C.L. and Hanson, R.J. (1974). Solving Least Squares Problems.
//   Prentice-Hall. Chapter 23.
//
// =============================================================================

use nalgebra::{DMatrix, DVector};

/// Result of NNLS optimization
#[derive(Debug, Clone)]
pub struct NNLSResult {
    /// Solution vector (all components >= 0)
    pub x: DVector<f64>,
    /// Residual norm ||Ax - b||
    pub residual_norm: f64,
    /// Number of iterations
    pub iterations: usize,
    /// Whether the algorithm converged
    pub converged: bool,
}

/// Configuration for NNLS solver
#[derive(Debug, Clone)]
pub struct NNLSConfig {
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Tolerance for convergence
    pub tol: f64,
}

impl Default for NNLSConfig {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            tol: 1e-10,
        }
    }
}

/// Solve non-negative least squares: min ||Ax - b||² s.t. x >= 0
///
/// Uses the Lawson-Hanson active set algorithm.
///
/// # Arguments
/// * `a` - Design matrix (m x n)
/// * `b` - Response vector (m x 1)
/// * `config` - Solver configuration
///
/// # Returns
/// * `NNLSResult` containing the solution and diagnostics
pub fn nnls(a: &DMatrix<f64>, b: &DVector<f64>, config: &NNLSConfig) -> NNLSResult {
    let (m, n) = a.shape();
    assert_eq!(
        b.len(),
        m,
        "Dimension mismatch: A has {} rows but b has {} elements",
        m,
        b.len()
    );

    // Initialize
    let mut x = DVector::zeros(n);
    let mut w = nnls_gradient(a, b, &x);

    // P = indices in the positive set (active, can be non-zero)
    // Z = indices in the zero set (constrained to zero)
    let mut p_set: Vec<usize> = Vec::new();
    let mut z_set: Vec<usize> = (0..n).collect();

    let mut iter = 0;

    while !z_set.is_empty() && iter < config.max_iter {
        // Find index in Z with largest positive gradient
        let (max_w, max_idx) = largest_gradient_in_zero_set(&w, &z_set);

        // If no positive gradient, we're done
        if max_w <= config.tol {
            break;
        }

        let t = max_idx.expect("max_w > tol guarantees an index was found");

        // Move index t from Z to P
        if !move_zero_index_to_positive(t, &mut p_set, &mut z_set) {
            break;
        }

        // Inner loop: solve unconstrained problem on P, then fix negative components
        loop {
            iter += 1;
            if iter >= config.max_iter {
                break;
            }

            // Solve least squares on the positive set: A_P * z_P = b
            let z_p = solve_ls_subset(a, b, &p_set);

            // Check if all components in z_P are positive
            let all_positive = all_subset_components_positive(&z_p, &p_set, config.tol);

            if all_positive {
                // Accept the solution
                for &j in &p_set {
                    x[j] = z_p[j];
                }
                for &j in &z_set {
                    x[j] = 0.0;
                }
                break;
            } else {
                // Find the limiting alpha
                let (alpha, q_idx) = limiting_alpha(&x, &z_p, &p_set, config.tol);

                // Update x = x + alpha * (z - x)
                backtrack_active_solution(&mut x, &z_p, &p_set, alpha);

                // Move indices with x[j] = 0 from P to Z
                if let Some(q) = q_idx {
                    move_positive_index_to_zero(q, &mut x, &mut p_set, &mut z_set);
                }

                // Also move any other indices that became zero
                let mut to_move = Vec::new();
                for &j in &p_set {
                    if x[j].abs() <= config.tol {
                        to_move.push(j);
                    }
                }
                for j in to_move {
                    move_positive_index_to_zero(j, &mut x, &mut p_set, &mut z_set);
                }
            }
        }

        // Update gradient
        w = nnls_gradient(a, b, &x);
    }

    let residual = nnls_residual(a, b, &x);
    let residual_norm = residual.norm();

    NNLSResult {
        x,
        residual_norm,
        iterations: iter,
        converged: iter < config.max_iter,
    }
}

fn nnls_residual(a: &DMatrix<f64>, b: &DVector<f64>, x: &DVector<f64>) -> DVector<f64> {
    b - a * x
}

fn nnls_gradient(a: &DMatrix<f64>, b: &DVector<f64>, x: &DVector<f64>) -> DVector<f64> {
    a.transpose() * nnls_residual(a, b, x)
}

fn largest_gradient_in_zero_set(w: &DVector<f64>, z_set: &[usize]) -> (f64, Option<usize>) {
    let mut max_w = f64::NEG_INFINITY;
    let mut max_idx = None;

    for &j in z_set {
        if w[j] > max_w {
            max_w = w[j];
            max_idx = Some(j);
        }
    }

    (max_w, max_idx)
}

fn move_zero_index_to_positive(
    index: usize,
    p_set: &mut Vec<usize>,
    z_set: &mut Vec<usize>,
) -> bool {
    let Some(position) = z_set.iter().position(|&j| j == index) else {
        return false;
    };
    z_set.remove(position);
    if z_set.contains(&index) {
        return false;
    }
    if !p_set.contains(&index) {
        p_set.push(index);
    }
    p_set.sort();
    true
}

fn all_subset_components_positive(z_p: &DVector<f64>, p_set: &[usize], tol: f64) -> bool {
    p_set.iter().all(|&j| z_p[j] > tol)
}

fn limiting_alpha(
    x: &DVector<f64>,
    z_p: &DVector<f64>,
    p_set: &[usize],
    tol: f64,
) -> (f64, Option<usize>) {
    let mut alpha = 1.0;
    let mut q_idx = None;

    for &j in p_set {
        if z_p[j] <= tol {
            let ratio = x[j] / (x[j] - z_p[j]);
            if ratio < alpha {
                alpha = ratio;
                q_idx = Some(j);
            }
        }
    }

    (alpha, q_idx)
}

fn backtrack_active_solution(
    x: &mut DVector<f64>,
    z_p: &DVector<f64>,
    p_set: &[usize],
    alpha: f64,
) {
    for &j in p_set {
        x[j] = x[j] + alpha * (z_p[j] - x[j]);
    }
}

fn move_positive_index_to_zero(
    index: usize,
    x: &mut DVector<f64>,
    p_set: &mut Vec<usize>,
    z_set: &mut Vec<usize>,
) {
    x[index] = 0.0;
    p_set.retain(|&j| j != index);
    if !z_set.contains(&index) {
        z_set.push(index);
    }
    z_set.sort();
}

/// Solve unconstrained least squares on a subset of columns
fn solve_ls_subset(a: &DMatrix<f64>, b: &DVector<f64>, indices: &[usize]) -> DVector<f64> {
    let n = a.ncols();
    let mut result = DVector::zeros(n);

    if indices.is_empty() {
        return result;
    }

    // Extract submatrix A_P (columns in the positive set)
    let a_p = a.select_columns(indices);

    // Solve A_P * z_P = b using normal equations (A_P' A_P) z_P = A_P' b
    let ata = a_p.transpose() * &a_p;
    let atb = a_p.transpose() * b;

    // Use Cholesky if possible, otherwise SVD
    let z_p = if let Some(chol) = ata.clone().cholesky() {
        chol.solve(&atb)
    } else {
        // Fall back to SVD for ill-conditioned systems
        let svd = ata.svd(true, true);
        svd.solve(&atb, 1e-10).unwrap_or(atb)
    };

    // Place solution back into full vector
    for (i, &j) in indices.iter().enumerate() {
        result[j] = z_p[i];
    }

    result
}

/// Solve weighted NNLS: min ||W^{1/2}(Ax - b)||² s.t. x >= 0
///
/// This is equivalent to solving NNLS with A' = W^{1/2} A and b' = W^{1/2} b
pub fn nnls_weighted(
    a: &DMatrix<f64>,
    b: &DVector<f64>,
    weights: &DVector<f64>,
    config: &NNLSConfig,
) -> NNLSResult {
    let m = a.nrows();

    // Apply weights: A' = diag(sqrt(w)) * A, b' = diag(sqrt(w)) * b
    let sqrt_w = weights.map(|w| w.sqrt());

    let mut a_weighted = a.clone();
    let mut b_weighted = b.clone();

    for i in 0..m {
        let sw = sqrt_w[i];
        for j in 0..a.ncols() {
            a_weighted[(i, j)] *= sw;
        }
        b_weighted[i] *= sw;
    }

    nnls(&a_weighted, &b_weighted, config)
}

/// Solve penalized NNLS: min ||Ax - b||² + λ x'Sx s.t. x >= 0
///
/// This is used for penalized monotonic spline fitting.
/// The penalty term is incorporated by augmenting the system:
///
/// ```text
///     [    A   ]       [b]
///     [√λ L    ] x  ≈  [0]
/// ```
///
/// where S = L'L (Cholesky decomposition of penalty matrix)
pub fn nnls_penalized(
    a: &DMatrix<f64>,
    b: &DVector<f64>,
    penalty: &DMatrix<f64>,
    lambda: f64,
    config: &NNLSConfig,
) -> NNLSResult {
    let (m, n) = a.shape();
    let k = penalty.nrows();

    assert_eq!(
        penalty.ncols(),
        n,
        "Penalty matrix columns must match A columns"
    );
    assert_eq!(k, n, "Penalty matrix must be square");

    // Compute sqrt(lambda) * L where S = L'L
    // For difference penalty, we can use the penalty matrix directly
    // since it's already in the form D'D
    let sqrt_lambda = lambda.sqrt();

    // Augment the system
    let mut a_aug = DMatrix::zeros(m + k, n);
    let mut b_aug = DVector::zeros(m + k);

    // Copy A into top part
    for i in 0..m {
        for j in 0..n {
            a_aug[(i, j)] = a[(i, j)];
        }
        b_aug[i] = b[i];
    }

    // Add penalty: we need L such that S = L'L
    // For S = D'D where D is the difference matrix, L = D
    // We'll use SVD to get a valid L: S = U Σ U', so L = Σ^{1/2} U'
    let svd = penalty.clone().svd(true, true);
    let u = svd.u.as_ref().expect("SVD computed with u=true");
    let s = &svd.singular_values;

    for i in 0..k {
        let sqrt_s = s[i].sqrt();
        for j in 0..n {
            // L[i,j] = sqrt(s[i]) * U[j,i]
            a_aug[(augmented_penalty_row(m, i), j)] =
                penalty_augmented_entry(sqrt_lambda, sqrt_s, u[(j, i)]);
        }
        // b_aug[m + i] = 0 (already initialized)
    }

    nnls(&a_aug, &b_aug, config)
}

fn penalty_augmented_entry(sqrt_lambda: f64, sqrt_s: f64, u_ji: f64) -> f64 {
    sqrt_lambda * sqrt_s * u_ji
}

fn augmented_penalty_row(n_data_rows: usize, penalty_row: usize) -> usize {
    n_data_rows + penalty_row
}

/// Solve weighted penalized NNLS: min ||W^{1/2}(Ax - b)||² + λ x'Sx s.t. x >= 0
///
/// Combines weights and penalty for use in IRLS with monotonic constraints.
pub fn nnls_weighted_penalized(
    a: &DMatrix<f64>,
    b: &DVector<f64>,
    weights: &DVector<f64>,
    penalty: &DMatrix<f64>,
    lambda: f64,
    config: &NNLSConfig,
) -> NNLSResult {
    let (m, n) = a.shape();

    // Apply weights to A and b
    let sqrt_w = weights.map(|w| w.sqrt());

    let mut a_weighted = a.clone();
    let mut b_weighted = b.clone();

    for i in 0..m {
        let sw = sqrt_w[i];
        for j in 0..n {
            a_weighted[(i, j)] *= sw;
        }
        b_weighted[i] *= sw;
    }

    // Now solve penalized NNLS with the weighted system
    nnls_penalized(&a_weighted, &b_weighted, penalty, lambda, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_nnls_simple() {
        // Simple case: minimize ||Ax - b||² s.t. x >= 0
        // A = [[1, 0], [0, 1]], b = [1, -1]
        // Unconstrained solution: x = [1, -1]
        // NNLS solution: x = [1, 0]
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let b = DVector::from_row_slice(&[1.0, -1.0]);

        let config = NNLSConfig::default();
        let result = nnls(&a, &b, &config);

        assert!(result.converged);
        assert_relative_eq!(result.x[0], 1.0, epsilon = 1e-8);
        assert_relative_eq!(result.x[1], 0.0, epsilon = 1e-8);
    }

    #[test]
    fn test_nnls_all_positive() {
        // Case where unconstrained solution is already non-negative
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let b = DVector::from_row_slice(&[2.0, 3.0]);

        let config = NNLSConfig::default();
        let result = nnls(&a, &b, &config);

        assert!(result.converged);
        assert_relative_eq!(result.x[0], 2.0, epsilon = 1e-8);
        assert_relative_eq!(result.x[1], 3.0, epsilon = 1e-8);
    }

    #[test]
    fn test_nnls_overdetermined() {
        // Overdetermined system
        let a = DMatrix::from_row_slice(4, 2, &[1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0]);
        let b = DVector::from_row_slice(&[1.0, 2.0, 3.0, 4.0]);

        let config = NNLSConfig::default();
        let result = nnls(&a, &b, &config);

        assert!(result.converged);
        assert!(result.x[0] >= -1e-10);
        assert!(result.x[1] >= -1e-10);
    }

    #[test]
    fn test_nnls_iteration_cap_reports_nonconvergence() {
        let a = DMatrix::identity(3, 3);
        let b = DVector::from_row_slice(&[1.0, 2.0, 3.0]);

        let config = NNLSConfig {
            max_iter: 1,
            tol: 1e-12,
        };
        let result = nnls(&a, &b, &config);

        assert!(!result.converged);
        assert_eq!(result.iterations, 1);
        assert!(result.residual_norm.is_finite());
    }

    #[test]
    fn test_nnls_backtracks_when_active_solution_turns_negative() {
        // This fixture first admits column 1, then the unconstrained solve on
        // columns [0, 1] makes column 1 negative. Lawson-Hanson must backtrack,
        // remove it from the active set, and finish on the boundary x[1] = 0.
        let a = DMatrix::from_row_slice(
            3,
            2,
            &[
                0.78987976,
                1.24389962,
                0.68153217,
                0.88987654,
                -1.98035571,
                -2.10331921,
            ],
        );
        let b = DVector::from_row_slice(&[0.08505026, 2.25199636, -1.89627935]);
        let result = nnls(&a, &b, &NNLSConfig::default());

        assert!(result.converged);
        assert!(result.iterations >= 3);
        assert!(result.x[0] > 1.0);
        assert_relative_eq!(result.x[1], 0.0, epsilon = 1e-10);
        assert!(result.residual_norm.is_finite());
    }

    #[test]
    fn nnls_gradient_and_active_set_helpers_have_exact_contracts() {
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let b = DVector::from_row_slice(&[5.0, 6.0]);
        let x = DVector::from_row_slice(&[0.5, -1.0]);

        let residual = nnls_residual(&a, &b, &x);
        assert_relative_eq!(residual[0], 6.5, epsilon = 1e-12);
        assert_relative_eq!(residual[1], 8.5, epsilon = 1e-12);

        let gradient = nnls_gradient(&a, &b, &x);
        assert_relative_eq!(gradient[0], 32.0, epsilon = 1e-12);
        assert_relative_eq!(gradient[1], 47.0, epsilon = 1e-12);

        let w = DVector::from_row_slice(&[2.0, 5.0, 5.0, -1.0]);
        assert_eq!(
            largest_gradient_in_zero_set(&w, &[0, 1, 2, 3]),
            (5.0, Some(1))
        );
        assert_eq!(
            largest_gradient_in_zero_set(&w, &[]),
            (f64::NEG_INFINITY, None)
        );

        let mut p_set = vec![3];
        let mut z_set = vec![0, 2, 1];
        assert!(move_zero_index_to_positive(2, &mut p_set, &mut z_set));
        assert_eq!(p_set, vec![2, 3]);
        assert_eq!(z_set, vec![0, 1]);
        assert!(!move_zero_index_to_positive(9, &mut p_set, &mut z_set));
        assert_eq!(p_set, vec![2, 3]);
        assert_eq!(z_set, vec![0, 1]);

        let z_p = DVector::from_row_slice(&[0.0, 0.1, 0.2]);
        assert!(!all_subset_components_positive(&z_p, &[1, 2], 0.1));
        assert!(all_subset_components_positive(&z_p, &[2], 0.1));
    }

    #[test]
    fn nnls_backtracking_helpers_have_exact_contracts() {
        let x = DVector::from_row_slice(&[2.0, 3.0, 4.0]);
        let z_p = DVector::from_row_slice(&[-1.0, 1.0, -4.0]);
        let (alpha, q_idx) = limiting_alpha(&x, &z_p, &[0, 1, 2], 0.0);
        assert_relative_eq!(alpha, 0.5, epsilon = 1e-12);
        assert_eq!(q_idx, Some(2));

        let tie_x = DVector::from_row_slice(&[1.0, 2.0]);
        let tie_z_p = DVector::from_row_slice(&[-1.0, -2.0]);
        let (alpha, q_idx) = limiting_alpha(&tie_x, &tie_z_p, &[0, 1], 0.0);
        assert_relative_eq!(alpha, 0.5, epsilon = 1e-12);
        assert_eq!(q_idx, Some(0));

        let mut x_backtracked = x.clone();
        backtrack_active_solution(&mut x_backtracked, &z_p, &[0, 1, 2], alpha);
        assert_relative_eq!(x_backtracked[0], 0.5, epsilon = 1e-12);
        assert_relative_eq!(x_backtracked[1], 2.0, epsilon = 1e-12);
        assert_relative_eq!(x_backtracked[2], 0.0, epsilon = 1e-12);

        let mut p_set = vec![0, 1, 2];
        let mut z_set = vec![3];
        move_positive_index_to_zero(1, &mut x_backtracked, &mut p_set, &mut z_set);
        assert_relative_eq!(x_backtracked[1], 0.0, epsilon = 0.0);
        assert_eq!(p_set, vec![0, 2]);
        assert_eq!(z_set, vec![1, 3]);

        let mut duplicate_guard_x = DVector::from_row_slice(&[1.0, 2.0, 3.0]);
        let mut duplicate_guard_p = vec![0, 2];
        let mut duplicate_guard_z = vec![2, 4];
        move_positive_index_to_zero(
            2,
            &mut duplicate_guard_x,
            &mut duplicate_guard_p,
            &mut duplicate_guard_z,
        );
        assert_relative_eq!(duplicate_guard_x[2], 0.0, epsilon = 0.0);
        assert_eq!(duplicate_guard_p, vec![0]);
        assert_eq!(duplicate_guard_z, vec![2, 4]);
    }

    #[test]
    fn test_solve_ls_subset_empty_returns_full_zero_vector() {
        let a = DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = DVector::from_row_slice(&[1.0, 2.0]);
        let z = solve_ls_subset(&a, &b, &[]);

        assert_eq!(z.len(), 3);
        assert!(z.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn test_singular_subset_uses_svd_fallback() {
        let a = DMatrix::from_row_slice(3, 2, &[1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
        let b = DVector::from_row_slice(&[1.0, 2.0, 3.0]);

        let z = solve_ls_subset(&a, &b, &[0, 1]);

        assert!(z.iter().all(|v| v.is_finite()));
        assert!((a.clone() * z.clone() - b).norm() < 1e-8);
    }

    #[test]
    fn test_weighted_nnls_matches_explicit_row_scaling() {
        let a = DMatrix::from_row_slice(3, 2, &[1.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
        let b = DVector::from_row_slice(&[1.0, 3.0, 2.0]);
        let weights = DVector::from_row_slice(&[4.0, 1.0, 9.0]);
        let config = NNLSConfig::default();

        let weighted = nnls_weighted(&a, &b, &weights, &config);

        let mut a_explicit = a.clone();
        let mut b_explicit = b.clone();
        for i in 0..a.nrows() {
            let sw = weights[i].sqrt();
            for j in 0..a.ncols() {
                a_explicit[(i, j)] *= sw;
            }
            b_explicit[i] *= sw;
        }
        let explicit = nnls(&a_explicit, &b_explicit, &config);

        assert!(weighted.converged);
        for j in 0..a.ncols() {
            assert_relative_eq!(weighted.x[j], explicit.x[j], epsilon = 1e-10);
        }
        assert_relative_eq!(
            weighted.residual_norm,
            explicit.residual_norm,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_nnls_penalized() {
        // Test penalized NNLS
        let a = DMatrix::from_row_slice(
            4,
            3,
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        );
        let b = DVector::from_row_slice(&[1.0, 2.0, 3.0, 6.0]);

        // Simple identity penalty
        let penalty = DMatrix::identity(3, 3);

        let config = NNLSConfig::default();
        let result = nnls_penalized(&a, &b, &penalty, 0.1, &config);

        assert!(result.converged);
        assert!(result.x[0] >= -1e-10);
        assert!(result.x[1] >= -1e-10);
        assert!(result.x[2] >= -1e-10);
    }

    #[test]
    fn penalized_nnls_augmented_entry_is_exact_three_factor_product() {
        assert_eq!(augmented_penalty_row(3, 0), 3);
        assert_eq!(augmented_penalty_row(3, 2), 5);
        assert_relative_eq!(
            penalty_augmented_entry(2.0, 3.0, -0.25),
            -1.5,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            penalty_augmented_entry(0.5, 4.0, 1.25),
            2.5,
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_weighted_penalized_nnls_matches_two_step_construction() {
        let a = DMatrix::from_row_slice(3, 2, &[1.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
        let b = DVector::from_row_slice(&[1.0, 2.5, 2.0]);
        let weights = DVector::from_row_slice(&[0.25, 4.0, 1.0]);
        let penalty = DMatrix::from_row_slice(2, 2, &[1.0, -1.0, -1.0, 1.0]);
        let config = NNLSConfig::default();

        let combined = nnls_weighted_penalized(&a, &b, &weights, &penalty, 0.5, &config);

        let mut a_weighted = a.clone();
        let mut b_weighted = b.clone();
        for i in 0..a.nrows() {
            let sw = weights[i].sqrt();
            for j in 0..a.ncols() {
                a_weighted[(i, j)] *= sw;
            }
            b_weighted[i] *= sw;
        }
        let explicit = nnls_penalized(&a_weighted, &b_weighted, &penalty, 0.5, &config);

        assert!(combined.converged);
        for j in 0..a.ncols() {
            assert_relative_eq!(combined.x[j], explicit.x[j], epsilon = 1e-10);
        }
        assert_relative_eq!(
            combined.residual_norm,
            explicit.residual_norm,
            epsilon = 1e-10
        );
    }

    #[test]
    #[should_panic(expected = "Dimension mismatch")]
    fn test_nnls_dimension_mismatch_panics() {
        let a = DMatrix::identity(2, 2);
        let b = DVector::from_row_slice(&[1.0]);
        let _ = nnls(&a, &b, &NNLSConfig::default());
    }

    #[test]
    #[should_panic(expected = "Penalty matrix columns must match A columns")]
    fn test_penalized_nnls_rejects_wrong_penalty_columns() {
        let a = DMatrix::identity(2, 2);
        let b = DVector::from_row_slice(&[1.0, 1.0]);
        let penalty = DMatrix::identity(2, 3);
        let _ = nnls_penalized(&a, &b, &penalty, 0.1, &NNLSConfig::default());
    }

    #[test]
    #[should_panic(expected = "Penalty matrix must be square")]
    fn test_penalized_nnls_rejects_non_square_penalty() {
        let a = DMatrix::identity(3, 3);
        let b = DVector::from_row_slice(&[1.0, 1.0, 1.0]);
        let penalty = DMatrix::from_row_slice(2, 3, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let _ = nnls_penalized(&a, &b, &penalty, 0.1, &NNLSConfig::default());
    }
}
