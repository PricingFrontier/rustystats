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
    assert_eq!(b.len(), m, "Dimension mismatch: A has {} rows but b has {} elements", m, b.len());

    // Initialize
    let mut x = DVector::zeros(n);
    let mut w = a.transpose() * (b - a * &x); // Gradient of ||Ax - b||² w.r.t. x
    
    // P = indices in the positive set (active, can be non-zero)
    // Z = indices in the zero set (constrained to zero)
    let mut p_set: Vec<usize> = Vec::new();
    let mut z_set: Vec<usize> = (0..n).collect();
    
    let mut iter = 0;
    
    while !z_set.is_empty() && iter < config.max_iter {
        // Find index in Z with largest positive gradient
        let mut max_w = f64::NEG_INFINITY;
        let mut max_idx = None;
        
        for &j in &z_set {
            if w[j] > max_w {
                max_w = w[j];
                max_idx = Some(j);
            }
        }
        
        // If no positive gradient, we're done
        if max_w <= config.tol {
            break;
        }
        
        let t = max_idx.unwrap();
        
        // Move index t from Z to P
        z_set.retain(|&j| j != t);
        p_set.push(t);
        p_set.sort();
        
        // Inner loop: solve unconstrained problem on P, then fix negative components
        loop {
            iter += 1;
            if iter >= config.max_iter {
                break;
            }
            
            // Solve least squares on the positive set: A_P * z_P = b
            let z_p = solve_ls_subset(a, b, &p_set);
            
            // Check if all components in z_P are positive
            let all_positive = p_set.iter().all(|&j| z_p[j] > config.tol);
            
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
                let mut alpha = 1.0;
                let mut q_idx = None;
                
                for &j in &p_set {
                    if z_p[j] <= config.tol {
                        let ratio = x[j] / (x[j] - z_p[j]);
                        if ratio < alpha {
                            alpha = ratio;
                            q_idx = Some(j);
                        }
                    }
                }
                
                // Update x = x + alpha * (z - x)
                for &j in &p_set {
                    x[j] = x[j] + alpha * (z_p[j] - x[j]);
                }
                
                // Move indices with x[j] = 0 from P to Z
                if let Some(q) = q_idx {
                    x[q] = 0.0;
                    p_set.retain(|&j| j != q);
                    z_set.push(q);
                    z_set.sort();
                }
                
                // Also move any other indices that became zero
                let mut to_move = Vec::new();
                for &j in &p_set {
                    if x[j].abs() <= config.tol {
                        to_move.push(j);
                    }
                }
                for j in to_move {
                    x[j] = 0.0;
                    p_set.retain(|&k| k != j);
                    if !z_set.contains(&j) {
                        z_set.push(j);
                    }
                }
                z_set.sort();
            }
        }
        
        // Update gradient
        w = a.transpose() * (b - a * &x);
    }
    
    let residual = b - a * &x;
    let residual_norm = residual.norm();
    
    NNLSResult {
        x,
        residual_norm,
        iterations: iter,
        converged: iter < config.max_iter,
    }
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
    
    assert_eq!(penalty.ncols(), n, "Penalty matrix columns must match A columns");
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
    let u = svd.u.as_ref().unwrap();
    let s = &svd.singular_values;
    
    for i in 0..k {
        let sqrt_s = s[i].sqrt();
        for j in 0..n {
            // L[i,j] = sqrt(s[i]) * U[j,i]
            a_aug[(m + i, j)] = sqrt_lambda * sqrt_s * u[(j, i)];
        }
        // b_aug[m + i] = 0 (already initialized)
    }
    
    nnls(&a_aug, &b_aug, config)
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
        let a = DMatrix::from_row_slice(4, 2, &[
            1.0, 1.0,
            1.0, 2.0,
            1.0, 3.0,
            1.0, 4.0,
        ]);
        let b = DVector::from_row_slice(&[1.0, 2.0, 3.0, 4.0]);
        
        let config = NNLSConfig::default();
        let result = nnls(&a, &b, &config);
        
        assert!(result.converged);
        assert!(result.x[0] >= -1e-10);
        assert!(result.x[1] >= -1e-10);
    }

    #[test]
    fn test_nnls_penalized() {
        // Test penalized NNLS
        let a = DMatrix::from_row_slice(4, 3, &[
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
            1.0, 1.0, 1.0,
        ]);
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
}
