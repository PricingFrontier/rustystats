// =============================================================================
// PENALIZED SPLINES (P-SPLINES)
// =============================================================================
//
// This module provides the core infrastructure for penalized splines, which
// allow automatic smoothness selection via GCV or REML.
//
// THE KEY IDEA
// ------------
// Instead of fixing the number of basis functions (like ns(x, df=5)), we use
// a larger basis (e.g., k=10) and penalize wiggliness. The smoothing parameter
// λ controls the trade-off:
//   - λ = 0: No penalty, fits every wiggle (overfits)
//   - λ → ∞: Maximum penalty, fits a straight line (underfits)
//   - λ optimal: Data-driven balance via GCV
//
// PENALTY MATRIX
// --------------
// The penalty is based on differences of adjacent coefficients:
//   - Order 1: Penalize Σ(β_{j+1} - β_j)²  → piecewise constant
//   - Order 2: Penalize Σ(β_{j+2} - 2β_{j+1} + β_j)²  → piecewise linear
//
// This is encoded as S = D'D where D is a difference matrix.
//
// EFFECTIVE DEGREES OF FREEDOM
// ----------------------------
// The EDF measures how many parameters are "effectively" used:
//   EDF = trace((X'WX + λS)⁻¹ X'WX)
//
// - EDF ≈ k (basis size) when λ ≈ 0
// - EDF ≈ 1-2 when λ is large (nearly linear)
//
// GCV (GENERALIZED CROSS-VALIDATION)
// ----------------------------------
// GCV approximates leave-one-out CV without refitting:
//   GCV(λ) = n × Deviance / (n - EDF)²
//
// We search for λ that minimizes GCV.
//
// REFERENCES
// ----------
// - Eilers & Marx (1996): Flexible smoothing with B-splines and penalties
// - Wood (2017): Generalized Additive Models (2nd ed.)
//
// =============================================================================

use ndarray::Array2;

#[allow(unused_imports)]
use ndarray::Axis;

// =============================================================================
// DIFFERENCE MATRIX
// =============================================================================

/// Construct a difference matrix D of given order.
///
/// For a vector β of length k, the difference matrix D computes:
/// - Order 1: D @ β = [β₁-β₀, β₂-β₁, ..., β_{k-1}-β_{k-2}]  (k-1 rows)
/// - Order 2: D @ β = [β₂-2β₁+β₀, β₃-2β₂+β₁, ...]          (k-2 rows)
///
/// # Arguments
/// * `k` - Number of coefficients (columns of D)
/// * `order` - Difference order (1 or 2, typically 2 for smoothness)
///
/// # Returns
/// Difference matrix D of shape (k-order, k)
///
/// # Example
/// ```ignore
/// let d1 = difference_matrix(5, 1);
/// // D1 = [[-1, 1, 0, 0, 0],
/// //       [0, -1, 1, 0, 0],
/// //       [0, 0, -1, 1, 0],
/// //       [0, 0, 0, -1, 1]]
///
/// let d2 = difference_matrix(5, 2);
/// // D2 = [[1, -2, 1, 0, 0],
/// //       [0, 1, -2, 1, 0],
/// //       [0, 0, 1, -2, 1]]
/// ```
pub fn difference_matrix(k: usize, order: usize) -> Array2<f64> {
    if order == 0 {
        return Array2::eye(k);
    }
    
    if k <= order {
        // Not enough coefficients for this order
        return Array2::zeros((0, k));
    }
    
    // Build order-1 difference matrix recursively
    if order == 1 {
        let n_rows = k - 1;
        let mut d = Array2::zeros((n_rows, k));
        for i in 0..n_rows {
            d[[i, i]] = -1.0;
            d[[i, i + 1]] = 1.0;
        }
        return d;
    }
    
    // Higher orders: D_m = D_1 @ D_{m-1}
    let d1 = difference_matrix(k, 1);
    let d_prev = difference_matrix(k - 1, order - 1);
    
    // Matrix multiplication: d_prev @ d1
    d_prev.dot(&d1.slice(ndarray::s![.., ..k-1]).to_owned().insert_axis(Axis(1)).remove_axis(Axis(1)))
}

/// Optimized: Build order-2 difference matrix directly (most common case)
pub fn difference_matrix_order2(k: usize) -> Array2<f64> {
    if k < 3 {
        return Array2::zeros((0, k));
    }
    
    let n_rows = k - 2;
    let mut d = Array2::zeros((n_rows, k));
    
    for i in 0..n_rows {
        d[[i, i]] = 1.0;
        d[[i, i + 1]] = -2.0;
        d[[i, i + 2]] = 1.0;
    }
    
    d
}

// =============================================================================
// PENALTY MATRIX
// =============================================================================

/// Construct the penalty matrix S = D'D for a given order.
///
/// The penalty on coefficients β is: β' S β = ||D β||²
///
/// This penalizes the sum of squared differences of the given order.
///
/// # Arguments
/// * `k` - Number of basis functions (coefficients)
/// * `order` - Difference order (typically 2 for smoothness)
///
/// # Returns
/// Penalty matrix S of shape (k, k), positive semi-definite
///
/// # Properties
/// - S is symmetric
/// - S is positive semi-definite (eigenvalues ≥ 0)
/// - S has `order` zero eigenvalues (null space = polynomials of degree < order)
/// - For order=2: null space is {1, x} (constant and linear)
pub fn penalty_matrix(k: usize, order: usize) -> Array2<f64> {
    let d = if order == 2 {
        difference_matrix_order2(k)
    } else {
        difference_matrix(k, order)
    };
    
    // S = D' @ D
    d.t().dot(&d)
}

/// Construct a block-diagonal penalty matrix for multiple smooth terms.
///
/// Each smooth term gets its own penalty block on the diagonal.
///
/// # Arguments
/// * `term_sizes` - Number of basis functions for each smooth term
/// * `order` - Difference order (same for all terms)
///
/// # Returns
/// Block-diagonal penalty matrix
pub fn block_penalty_matrix(term_sizes: &[usize], order: usize) -> Array2<f64> {
    let total_size: usize = term_sizes.iter().sum();
    let mut s = Array2::zeros((total_size, total_size));
    
    let mut offset = 0;
    for &k in term_sizes {
        let s_block = penalty_matrix(k, order);
        for i in 0..k {
            for j in 0..k {
                s[[offset + i, offset + j]] = s_block[[i, j]];
            }
        }
        offset += k;
    }
    
    s
}

// =============================================================================
// EFFECTIVE DEGREES OF FREEDOM
// =============================================================================

/// Compute effective degrees of freedom for a penalized fit.
///
/// EDF = trace((X'WX + λS)⁻¹ X'WX)
///
/// This measures how many parameters are "effectively" used after penalization.
/// - EDF ≈ k when λ ≈ 0 (no penalty, all basis functions used)
/// - EDF ≈ order when λ → ∞ (maximum penalty, polynomial of degree order-1)
///
/// # Arguments
/// * `xtwx` - X'WX matrix (p × p)
/// * `penalty` - Penalty matrix S (p × p)
/// * `lambda` - Smoothing parameter
///
/// # Returns
/// Effective degrees of freedom (scalar)
pub fn compute_edf(xtwx: &Array2<f64>, penalty: &Array2<f64>, lambda: f64) -> f64 {
    let p = xtwx.nrows();
    
    if lambda <= 0.0 {
        // No penalty: EDF = rank(X) ≈ p
        return p as f64;
    }
    
    // Compute (X'WX + λS)
    let mut xtwx_pen = xtwx.clone();
    for i in 0..p {
        for j in 0..p {
            xtwx_pen[[i, j]] += lambda * penalty[[i, j]];
        }
    }
    
    // Compute (X'WX + λS)⁻¹ @ X'WX
    // Use Cholesky decomposition for stability
    let hat_matrix = solve_symmetric_system(&xtwx_pen, xtwx);
    
    // EDF = trace of hat matrix
    let mut edf = 0.0;
    for i in 0..p {
        edf += hat_matrix[[i, i]];
    }
    
    edf
}

/// Compute EDF for multiple smooth terms with separate lambdas.
///
/// Returns the EDF for each term individually.
///
/// # Arguments
/// * `xtwx` - Full X'WX matrix
/// * `penalties` - Penalty matrix for each smooth term
/// * `term_indices` - Column indices for each smooth term
/// * `lambdas` - Smoothing parameter for each term
///
/// # Returns
/// Vector of EDF values, one per smooth term
pub fn compute_edf_per_term(
    xtwx: &Array2<f64>,
    penalties: &[Array2<f64>],
    term_indices: &[std::ops::Range<usize>],
    lambdas: &[f64],
) -> Vec<f64> {
    let p = xtwx.nrows();
    
    // Build combined penalty matrix
    let mut combined_penalty: Array2<f64> = Array2::zeros((p, p));
    for (idx, (penalty, lambda)) in penalties.iter().zip(lambdas.iter()).enumerate() {
        let range = &term_indices[idx];
        for i in 0..penalty.nrows() {
            for j in 0..penalty.ncols() {
                combined_penalty[[range.start + i, range.start + j]] += lambda * penalty[[i, j]];
            }
        }
    }
    
    // Compute (X'WX + Σλ_jS_j)⁻¹
    let xtwx_pen = xtwx + &combined_penalty;
    let xtwx_pen_inv = invert_symmetric(&xtwx_pen);
    
    // Hat matrix H = (X'WX + λS)⁻¹ @ X'WX
    let hat_matrix = xtwx_pen_inv.dot(xtwx);
    
    // Extract EDF for each term: trace of the relevant block
    let mut edfs = Vec::with_capacity(term_indices.len());
    for range in term_indices {
        let mut term_edf = 0.0;
        for i in range.clone() {
            term_edf += hat_matrix[[i, i]];
        }
        edfs.push(term_edf);
    }
    
    edfs
}

// =============================================================================
// GCV (GENERALIZED CROSS-VALIDATION)
// =============================================================================

/// Compute the GCV score for a given smoothing parameter.
///
/// GCV(λ) = n × Deviance / (n - EDF)²
///
/// Lower GCV is better. This approximates leave-one-out cross-validation.
///
/// # Arguments
/// * `deviance` - Model deviance at this λ
/// * `n` - Number of observations
/// * `edf` - Effective degrees of freedom at this λ
///
/// # Returns
/// GCV score (lower is better)
pub fn gcv_score(deviance: f64, n: usize, edf: f64) -> f64 {
    let n_f = n as f64;
    let denominator = (n_f - edf).max(1.0);  // Avoid division by zero
    
    n_f * deviance / (denominator * denominator)
}

/// Alternative: AIC-like criterion for lambda selection
pub fn aic_score(deviance: f64, edf: f64) -> f64 {
    deviance + 2.0 * edf
}

/// Alternative: BIC-like criterion for lambda selection
pub fn bic_score(deviance: f64, n: usize, edf: f64) -> f64 {
    deviance + (n as f64).ln() * edf
}

// =============================================================================
// LAMBDA GRID SEARCH
// =============================================================================

/// Generate a logarithmic grid of lambda values for searching.
///
/// # Arguments
/// * `n_lambdas` - Number of lambda values in grid
/// * `lambda_min` - Minimum lambda (typically 1e-4)
/// * `lambda_max` - Maximum lambda (typically 1e4)
///
/// # Returns
/// Vector of lambda values on log scale
pub fn lambda_grid(n_lambdas: usize, lambda_min: f64, lambda_max: f64) -> Vec<f64> {
    if n_lambdas <= 1 {
        return vec![(lambda_min * lambda_max).sqrt()];
    }
    
    let log_min = lambda_min.ln();
    let log_max = lambda_max.ln();
    let step = (log_max - log_min) / (n_lambdas - 1) as f64;
    
    (0..n_lambdas)
        .map(|i| (log_min + i as f64 * step).exp())
        .collect()
}

/// Result from lambda grid search
#[derive(Debug, Clone)]
pub struct LambdaSearchResult {
    /// Optimal lambda value
    pub lambda: f64,
    /// GCV score at optimal lambda
    pub gcv: f64,
    /// EDF at optimal lambda
    pub edf: f64,
    /// All lambda values evaluated
    pub lambdas: Vec<f64>,
    /// GCV scores for all lambdas
    pub gcv_scores: Vec<f64>,
    /// EDF values for all lambdas
    pub edf_values: Vec<f64>,
}

// =============================================================================
// HELPER FUNCTIONS (Linear Algebra)
// =============================================================================

/// Solve A @ X = B where A is symmetric positive definite.
/// Returns X = A⁻¹ @ B
fn solve_symmetric_system(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    crate::convert::solve_symmetric_matrix(a, b)
}

/// Invert a symmetric positive (semi-)definite matrix.
fn invert_symmetric(a: &Array2<f64>) -> Array2<f64> {
    let n = a.nrows();
    let identity = Array2::eye(n);
    solve_symmetric_system(a, &identity)
}

// =============================================================================
// SMOOTH TERM STRUCT
// =============================================================================

/// Represents a single smooth term with its basis and penalty.
#[derive(Debug, Clone)]
pub struct SmoothTerm {
    /// Variable name
    pub name: String,
    /// Basis matrix (n × k)
    pub basis: Array2<f64>,
    /// Penalty matrix (k × k)
    pub penalty: Array2<f64>,
    /// Current smoothing parameter
    pub lambda: f64,
    /// Current effective degrees of freedom
    pub edf: f64,
    /// Penalty order (typically 2)
    pub penalty_order: usize,
}

impl SmoothTerm {
    /// Create a new smooth term from a basis matrix.
    pub fn new(name: String, basis: Array2<f64>, penalty_order: usize) -> Self {
        let k = basis.ncols();
        let penalty = penalty_matrix(k, penalty_order);
        
        Self {
            name,
            basis,
            penalty,
            lambda: 1.0,  // Initial guess
            edf: k as f64,
            penalty_order,
        }
    }
    
    /// Number of basis functions
    pub fn k(&self) -> usize {
        self.basis.ncols()
    }
    
    /// Number of observations
    pub fn n(&self) -> usize {
        self.basis.nrows()
    }
}

/// Collection of smooth terms for a model.
#[derive(Debug, Clone)]
pub struct SmoothTerms {
    pub terms: Vec<SmoothTerm>,
}

impl SmoothTerms {
    pub fn new() -> Self {
        Self { terms: Vec::new() }
    }
    
    pub fn add(&mut self, term: SmoothTerm) {
        self.terms.push(term);
    }
    
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }
    
    /// Total number of smooth basis columns
    pub fn total_basis_columns(&self) -> usize {
        self.terms.iter().map(|t| t.k()).sum()
    }
    
    /// Get column indices for each term in the combined design matrix.
    /// Assumes smooth terms come after parametric terms.
    pub fn column_indices(&self, parametric_cols: usize) -> Vec<std::ops::Range<usize>> {
        let mut ranges = Vec::with_capacity(self.terms.len());
        let mut offset = parametric_cols;
        
        for term in &self.terms {
            let start = offset;
            let end = offset + term.k();
            ranges.push(start..end);
            offset = end;
        }
        
        ranges
    }
    
    /// Build combined penalty matrix for all smooth terms.
    pub fn combined_penalty(&self, total_cols: usize, parametric_cols: usize) -> Array2<f64> {
        let mut combined: Array2<f64> = Array2::zeros((total_cols, total_cols));
        
        let indices = self.column_indices(parametric_cols);
        
        for (term, range) in self.terms.iter().zip(indices.iter()) {
            for i in 0..term.k() {
                for j in 0..term.k() {
                    combined[[range.start + i, range.start + j]] = 
                        term.lambda * term.penalty[[i, j]];
                }
            }
        }
        
        combined
    }
    
    /// Get all lambdas
    pub fn lambdas(&self) -> Vec<f64> {
        self.terms.iter().map(|t| t.lambda).collect()
    }
    
    /// Get all EDFs
    pub fn edfs(&self) -> Vec<f64> {
        self.terms.iter().map(|t| t.edf).collect()
    }
}

impl Default for SmoothTerms {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    use approx::assert_abs_diff_eq;
    
    #[test]
    fn test_difference_matrix_order1() {
        let d = difference_matrix(4, 1);
        assert_eq!(d.shape(), &[3, 4]);
        
        // Check structure: [-1, 1, 0, 0], [0, -1, 1, 0], [0, 0, -1, 1]
        assert_abs_diff_eq!(d[[0, 0]], -1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(d[[0, 1]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(d[[0, 2]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(d[[1, 1]], -1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(d[[1, 2]], 1.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_difference_matrix_order2() {
        let d = difference_matrix_order2(5);
        assert_eq!(d.shape(), &[3, 5]);
        
        // Check structure: [1, -2, 1, 0, 0], [0, 1, -2, 1, 0], [0, 0, 1, -2, 1]
        assert_abs_diff_eq!(d[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(d[[0, 1]], -2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(d[[0, 2]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(d[[1, 1]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(d[[1, 2]], -2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(d[[1, 3]], 1.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_penalty_matrix_symmetric() {
        let s = penalty_matrix(6, 2);
        
        // Check symmetry
        for i in 0..6 {
            for j in 0..6 {
                assert_abs_diff_eq!(s[[i, j]], s[[j, i]], epsilon = 1e-10);
            }
        }
    }
    
    #[test]
    fn test_penalty_matrix_positive_semidefinite() {
        let s = penalty_matrix(5, 2);
        
        // Check that all eigenvalues are non-negative
        use nalgebra::DMatrix;
        let s_nalg = DMatrix::from_row_slice(5, 5, s.as_slice().unwrap());
        let eigenvalues = s_nalg.symmetric_eigenvalues();
        
        for ev in eigenvalues.iter() {
            assert!(*ev >= -1e-10, "Eigenvalue {} should be non-negative", ev);
        }
    }
    
    #[test]
    fn test_penalty_matrix_null_space() {
        // For order=2 penalty, constant and linear functions should be in null space
        let s = penalty_matrix(5, 2);
        
        // Constant vector: β = [1, 1, 1, 1, 1]
        let constant = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0, 1.0]);
        let penalty_constant: f64 = constant.iter()
            .enumerate()
            .map(|(i, &bi)| {
                constant.iter().enumerate().map(|(j, &bj)| bi * s[[i, j]] * bj).sum::<f64>()
            })
            .sum();
        assert_abs_diff_eq!(penalty_constant, 0.0, epsilon = 1e-10);
        
        // Linear vector: β = [0, 1, 2, 3, 4]
        let linear = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        let penalty_linear: f64 = linear.iter()
            .enumerate()
            .map(|(i, &bi)| {
                linear.iter().enumerate().map(|(j, &bj)| bi * s[[i, j]] * bj).sum::<f64>()
            })
            .sum();
        assert_abs_diff_eq!(penalty_linear, 0.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_gcv_score() {
        // GCV = n * deviance / (n - edf)^2
        let gcv = gcv_score(100.0, 100, 10.0);
        let expected = 100.0 * 100.0 / (90.0 * 90.0);
        assert_abs_diff_eq!(gcv, expected, epsilon = 1e-10);
    }
    
    #[test]
    fn test_lambda_grid() {
        let grid = lambda_grid(5, 0.01, 100.0);
        
        assert_eq!(grid.len(), 5);
        assert_abs_diff_eq!(grid[0], 0.01, epsilon = 1e-10);
        assert_abs_diff_eq!(grid[4], 100.0, epsilon = 1e-6);
        
        // Check log-spacing
        let log_ratio1 = (grid[1] / grid[0]).ln();
        let log_ratio2 = (grid[2] / grid[1]).ln();
        assert_abs_diff_eq!(log_ratio1, log_ratio2, epsilon = 1e-10);
    }
    
    #[test]
    fn test_smooth_term_creation() {
        let basis = Array2::from_shape_vec((100, 10), vec![0.0; 1000]).unwrap();
        let term = SmoothTerm::new("age".to_string(), basis, 2);
        
        assert_eq!(term.name, "age");
        assert_eq!(term.k(), 10);
        assert_eq!(term.n(), 100);
        assert_eq!(term.penalty.shape(), &[10, 10]);
        assert_eq!(term.penalty_order, 2);
    }
    
    #[test]
    fn test_smooth_terms_collection() {
        let mut terms = SmoothTerms::new();
        
        let basis1 = Array2::from_shape_vec((100, 8), vec![0.0; 800]).unwrap();
        let basis2 = Array2::from_shape_vec((100, 10), vec![0.0; 1000]).unwrap();
        
        terms.add(SmoothTerm::new("age".to_string(), basis1, 2));
        terms.add(SmoothTerm::new("income".to_string(), basis2, 2));
        
        assert_eq!(terms.terms.len(), 2);
        assert_eq!(terms.total_basis_columns(), 18);
        
        // Check column indices (assuming 5 parametric columns)
        let indices = terms.column_indices(5);
        assert_eq!(indices[0], 5..13);  // age: cols 5-12
        assert_eq!(indices[1], 13..23); // income: cols 13-22
    }
}
