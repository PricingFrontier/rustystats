// =============================================================================
// GCV OPTIMIZER: Fast Lambda Selection via Brent's Method
// =============================================================================
//
// This module implements mgcv-style fast GCV optimization for smooth terms.
// Instead of refitting the model for each lambda, we optimize lambda within
// a single IRLS iteration using cheap matrix operations.
//
// THE KEY INSIGHT
// ---------------
// Once we have X'WX and X'Wz from IRLS weights, we can compute:
//   β(λ) = (X'WX + λS)⁻¹ X'Wz
//   EDF(λ) = trace((X'WX + λS)⁻¹ X'WX)
//   GCV(λ) = n × RSS(λ) / (n - EDF(λ))²
//
// All of these are cheap to evaluate once we have the matrices cached.
// We use Brent's method to find optimal λ in ~10-15 function evaluations.
//
// =============================================================================

use ndarray::{Array1, Array2};
use nalgebra::{DMatrix, DVector};

/// Result from Brent's optimization
#[derive(Debug, Clone)]
pub struct BrentResult {
    pub x_min: f64,
    pub f_min: f64,
    pub iterations: usize,
    pub converged: bool,
}

/// Brent's method for 1D minimization.
/// 
/// Finds the minimum of f(x) in the interval [a, b].
/// This is the gold standard for 1D optimization - guaranteed convergence,
/// superlinear in most cases.
/// 
/// # Arguments
/// * `f` - Function to minimize
/// * `a` - Lower bound of search interval
/// * `b` - Upper bound of search interval  
/// * `tol` - Convergence tolerance
/// * `max_iter` - Maximum iterations
pub fn brent_minimize<F>(f: F, a: f64, b: f64, tol: f64, max_iter: usize) -> BrentResult
where
    F: Fn(f64) -> f64,
{
    let golden = 0.381966011250105;  // (3 - sqrt(5)) / 2
    
    let mut a = a;
    let mut b = b;
    let mut x = a + golden * (b - a);
    let mut w = x;
    let mut v = x;
    let mut fx = f(x);
    let mut fw = fx;
    let mut fv = fx;
    
    let mut d: f64 = 0.0;
    let mut e: f64 = 0.0;
    
    for iter in 0..max_iter {
        let mid = 0.5 * (a + b);
        let tol1 = tol * x.abs() + 1e-10;
        let tol2 = 2.0 * tol1;
        
        // Check convergence
        if (x - mid).abs() <= tol2 - 0.5 * (b - a) {
            return BrentResult {
                x_min: x,
                f_min: fx,
                iterations: iter + 1,
                converged: true,
            };
        }
        
        // Try parabolic interpolation
        let mut use_golden = true;
        let mut u;
        
        if e.abs() > tol1 {
            // Fit parabola through x, w, v
            let r = (x - w) * (fx - fv);
            let q = (x - v) * (fx - fw);
            let p = (x - v) * q - (x - w) * r;
            let q = 2.0 * (q - r);
            
            let (p, q) = if q > 0.0 { (-p, q) } else { (p, -q) };
            
            let e_old = e;
            e = d;
            
            // Accept parabolic step if it's in bounds and small enough
            if p.abs() < (0.5 * q * e_old).abs() && p > q * (a - x) && p < q * (b - x) {
                d = p / q;
                u = x + d;
                
                // Don't evaluate too close to endpoints
                if u - a < tol2 || b - u < tol2 {
                    d = if x < mid { tol1 } else { -tol1 };
                }
                use_golden = false;
            }
        }
        
        if use_golden {
            // Golden section step
            e = if x < mid { b - x } else { a - x };
            d = golden * e;
        }
        
        // Evaluate at new point
        u = if d.abs() >= tol1 {
            x + d
        } else if d > 0.0 {
            x + tol1
        } else {
            x - tol1
        };
        
        let fu = f(u);
        
        // Update bracketing interval
        if fu <= fx {
            if u < x {
                b = x;
            } else {
                a = x;
            }
            v = w;
            fv = fw;
            w = x;
            fw = fx;
            x = u;
            fx = fu;
        } else {
            if u < x {
                a = u;
            } else {
                b = u;
            }
            if fu <= fw || w == x {
                v = w;
                fv = fw;
                w = u;
                fw = fu;
            } else if fu <= fv || v == x || v == w {
                v = u;
                fv = fu;
            }
        }
    }
    
    BrentResult {
        x_min: x,
        f_min: fx,
        iterations: max_iter,
        converged: false,
    }
}

// =============================================================================
// Shared helpers for GCV computation
// =============================================================================

/// Compute X'WX and X'Wz from nalgebra matrices.
fn compute_xtwx_xtwz_nalg(
    x: &DMatrix<f64>,
    z: &DVector<f64>,
    w: &DVector<f64>,
) -> (DMatrix<f64>, DVector<f64>) {
    let n = x.nrows();
    let p = x.ncols();
    
    let mut xtwx = DMatrix::zeros(p, p);
    for i in 0..n {
        let wi = w[i];
        for j in 0..p {
            let xij_w = x[(i, j)] * wi;
            for l in j..p {
                let val = xij_w * x[(i, l)];
                xtwx[(j, l)] += val;
                if l != j {
                    xtwx[(l, j)] += val;
                }
            }
        }
    }
    
    let mut xtwz = DVector::zeros(p);
    for i in 0..n {
        let wz = w[i] * z[i];
        for j in 0..p {
            xtwz[j] += x[(i, j)] * wz;
        }
    }
    
    (xtwx, xtwz)
}

/// Add penalty terms to X'WX: xtwx_pen = X'WX + Σ λᵢSᵢ
fn build_penalized_xtwx(
    xtwx: &DMatrix<f64>,
    penalties: &[DMatrix<f64>],
    col_ranges: &[(usize, usize)],
    lambdas: &[f64],
) -> DMatrix<f64> {
    let mut xtwx_pen = xtwx.clone();
    for (i, ((start, end), penalty)) in col_ranges.iter().zip(penalties).enumerate() {
        let lambda = lambdas[i];
        let k = end - start;
        for r in 0..k {
            for c in 0..k {
                xtwx_pen[(start + r, start + c)] += lambda * penalty[(r, c)];
            }
        }
    }
    xtwx_pen
}

/// Compute weighted RSS given coefficients.
fn compute_weighted_rss(
    x: &DMatrix<f64>,
    z: &DVector<f64>,
    w: &DVector<f64>,
    beta: &DVector<f64>,
) -> f64 {
    let n = x.nrows();
    let p = x.ncols();
    let mut rss = 0.0;
    for i in 0..n {
        let mut fitted = 0.0;
        for j in 0..p {
            fitted += x[(i, j)] * beta[j];
        }
        let resid = z[i] - fitted;
        rss += w[i] * resid * resid;
    }
    rss
}

/// Compute smooth EDF for given column ranges from (X'WX+λS)⁻¹ and X'WX.
fn compute_smooth_edfs_from_inv(
    xtwx_pen_inv: &DMatrix<f64>,
    xtwx: &DMatrix<f64>,
    col_ranges: &[(usize, usize)],
) -> Vec<f64> {
    let p = xtwx.nrows();
    let mut edfs = Vec::with_capacity(col_ranges.len());
    for (start, end) in col_ranges {
        let mut edf = 0.0;
        for i in *start..*end {
            for j in 0..p {
                edf += xtwx_pen_inv[(i, j)] * xtwx[(j, i)];
            }
        }
        edfs.push(edf);
    }
    edfs
}

/// Evaluate GCV score: n × RSS / (n - EDF)²
fn gcv_from_rss_edf(n: usize, rss: f64, total_edf: f64) -> f64 {
    let denom = (n as f64) - total_edf;
    if denom <= 1.0 {
        return f64::INFINITY;
    }
    (n as f64) * rss / (denom * denom)
}

/// Convert ndarray matrices to nalgebra for GCV computation.
fn to_nalgebra(
    x: &Array2<f64>,
    z: &Array1<f64>,
    w: &Array1<f64>,
) -> (DMatrix<f64>, DVector<f64>, DVector<f64>) {
    let n = x.nrows();
    let p = x.ncols();
    let x_contig = if x.is_standard_layout() { x.clone() } else { x.as_standard_layout().to_owned() };
    let x_nalg = DMatrix::from_row_slice(n, p, x_contig.as_slice().unwrap());
    let z_nalg = DVector::from_row_slice(z.to_owned().as_slice().unwrap());
    let w_nalg = DVector::from_row_slice(w.to_owned().as_slice().unwrap());
    (x_nalg, z_nalg, w_nalg)
}

// =============================================================================
// Single-term GCV cache
// =============================================================================

/// Cached matrices for fast GCV evaluation.
/// 
/// These are computed once per IRLS iteration and reused for all lambda evaluations.
#[derive(Debug, Clone)]
pub struct GCVCache {
    /// X'WX matrix (p × p)
    pub xtwx: DMatrix<f64>,
    /// X'Wz vector (p × 1)  
    pub xtwz: DVector<f64>,
    /// Penalty matrix S (k × k) for smooth term
    pub penalty: DMatrix<f64>,
    /// Column range for smooth term in full design matrix
    pub col_start: usize,
    pub col_end: usize,
    /// Number of observations
    pub n: usize,
    /// Number of parametric (unpenalized) columns
    pub n_parametric: usize,
    /// Working residual sum of squares at lambda=0 (for normalization)
    pub rss_base: f64,
    /// Working response z
    pub z: DVector<f64>,
    /// Design matrix X
    pub x: DMatrix<f64>,
    /// Weights W
    pub w: DVector<f64>,
}

impl GCVCache {
    /// Create a new GCV cache from IRLS iteration data.
    pub fn new(
        x: &Array2<f64>,
        z: &Array1<f64>,
        w: &Array1<f64>,
        penalty: &Array2<f64>,
        col_start: usize,
        col_end: usize,
        n_parametric: usize,
    ) -> Self {
        let n = x.nrows();
        let k = penalty.nrows();
        
        let (x_nalg, z_nalg, w_nalg) = to_nalgebra(x, z, w);
        let penalty_contig = if penalty.is_standard_layout() { penalty.clone() } else { penalty.as_standard_layout().to_owned() };
        let penalty_nalg = DMatrix::from_row_slice(k, k, penalty_contig.as_slice().unwrap());
        
        let (xtwx, xtwz) = compute_xtwx_xtwz_nalg(&x_nalg, &z_nalg, &w_nalg);
        
        Self {
            xtwx,
            xtwz,
            penalty: penalty_nalg,
            col_start,
            col_end,
            n,
            n_parametric,
            rss_base: 0.0,
            z: z_nalg,
            x: x_nalg,
            w: w_nalg,
        }
    }
    
    /// Evaluate GCV at a given lambda value.
    /// 
    /// This is the core function called by Brent's method.
    /// It computes coefficients, RSS, EDF, and GCV for the given lambda.
    pub fn evaluate_gcv(&self, log_lambda: f64) -> f64 {
        let lambda = log_lambda.exp();
        let p = self.xtwx.nrows();
        let col_range = (self.col_start, self.col_end);
        
        let xtwx_pen = build_penalized_xtwx(
            &self.xtwx, &[self.penalty.clone()], &[col_range], &[lambda],
        );
        
        let chol = match xtwx_pen.clone().cholesky() {
            Some(c) => c,
            None => return f64::INFINITY,
        };
        
        let beta = chol.solve(&self.xtwz);
        let rss = compute_weighted_rss(&self.x, &self.z, &self.w, &beta);
        
        let identity = DMatrix::identity(p, p);
        let xtwx_pen_inv = chol.solve(&identity);
        let edfs = compute_smooth_edfs_from_inv(&xtwx_pen_inv, &self.xtwx, &[col_range]);
        let total_edf = (self.n_parametric as f64) + edfs[0];
        
        gcv_from_rss_edf(self.n, rss, total_edf)
    }
    
    /// Find optimal lambda using Brent's method on log scale.
    pub fn optimize_lambda(
        &self,
        log_lambda_min: f64,
        log_lambda_max: f64,
        tol: f64,
    ) -> (f64, f64, f64) {
        // Use Brent's method on log scale
        let result = brent_minimize(
            |log_lam| self.evaluate_gcv(log_lam),
            log_lambda_min,
            log_lambda_max,
            tol,
            50,  // Max iterations
        );
        
        let optimal_lambda = result.x_min.exp();
        let optimal_gcv = result.f_min;
        
        // Compute EDF at optimal lambda
        let edf = self.compute_edf(optimal_lambda);
        
        (optimal_lambda, edf, optimal_gcv)
    }
    
    /// Compute EDF at a specific lambda.
    pub fn compute_edf(&self, lambda: f64) -> f64 {
        let p = self.xtwx.nrows();
        let col_range = (self.col_start, self.col_end);
        
        let xtwx_pen = build_penalized_xtwx(
            &self.xtwx, &[self.penalty.clone()], &[col_range], &[lambda],
        );
        
        let chol = match xtwx_pen.cholesky() {
            Some(c) => c,
            None => return (self.col_end - self.col_start) as f64,
        };
        
        let identity = DMatrix::identity(p, p);
        let xtwx_pen_inv = chol.solve(&identity);
        compute_smooth_edfs_from_inv(&xtwx_pen_inv, &self.xtwx, &[col_range])[0]
    }
    
    /// Solve for coefficients at a specific lambda.
    pub fn solve_coefficients(&self, lambda: f64) -> Option<DVector<f64>> {
        let col_range = (self.col_start, self.col_end);
        let xtwx_pen = build_penalized_xtwx(
            &self.xtwx, &[self.penalty.clone()], &[col_range], &[lambda],
        );
        xtwx_pen.cholesky().map(|chol| chol.solve(&self.xtwz))
    }
}

/// Fast GCV optimization for multiple smooth terms.
/// 
/// Uses coordinate descent: optimize each lambda while holding others fixed.
#[derive(Debug)]
pub struct MultiTermGCVOptimizer {
    pub xtwx: DMatrix<f64>,
    pub xtwz: DVector<f64>,
    pub penalties: Vec<DMatrix<f64>>,
    pub col_ranges: Vec<(usize, usize)>,
    pub n: usize,
    pub n_parametric: usize,
    pub z: DVector<f64>,
    pub x: DMatrix<f64>,
    pub w: DVector<f64>,
}

impl MultiTermGCVOptimizer {
    /// Create optimizer from matrices.
    pub fn new(
        x: &Array2<f64>,
        z: &Array1<f64>,
        w: &Array1<f64>,
        penalties: Vec<Array2<f64>>,
        col_ranges: Vec<(usize, usize)>,
        n_parametric: usize,
    ) -> Self {
        let n = x.nrows();
        
        let (x_nalg, z_nalg, w_nalg) = to_nalgebra(x, z, w);
        let (xtwx, xtwz) = compute_xtwx_xtwz_nalg(&x_nalg, &z_nalg, &w_nalg);
        
        let penalties_nalg: Vec<DMatrix<f64>> = penalties.iter()
            .map(|pen| DMatrix::from_row_slice(pen.nrows(), pen.ncols(), pen.as_slice().unwrap()))
            .collect();
        
        Self {
            xtwx,
            xtwz,
            penalties: penalties_nalg,
            col_ranges,
            n,
            n_parametric,
            z: z_nalg,
            x: x_nalg,
            w: w_nalg,
        }
    }
    
    /// Evaluate GCV for given lambdas.
    pub fn evaluate_gcv(&self, lambdas: &[f64]) -> f64 {
        let p = self.xtwx.nrows();
        
        let xtwx_pen = build_penalized_xtwx(
            &self.xtwx, &self.penalties, &self.col_ranges, lambdas,
        );
        
        let chol = match xtwx_pen.clone().cholesky() {
            Some(c) => c,
            None => return f64::INFINITY,
        };
        
        let beta = chol.solve(&self.xtwz);
        let rss = compute_weighted_rss(&self.x, &self.z, &self.w, &beta);
        
        let identity = DMatrix::identity(p, p);
        let xtwx_pen_inv = chol.solve(&identity);
        let edfs = compute_smooth_edfs_from_inv(&xtwx_pen_inv, &self.xtwx, &self.col_ranges);
        let total_edf = (self.n_parametric as f64) + edfs.iter().sum::<f64>();
        
        gcv_from_rss_edf(self.n, rss, total_edf)
    }
    
    /// Optimize all lambdas using coordinate descent.
    pub fn optimize_lambdas(
        &self,
        log_lambda_min: f64,
        log_lambda_max: f64,
        tol: f64,
        max_outer_iter: usize,
    ) -> Vec<f64> {
        let n_terms = self.penalties.len();
        let mut lambdas = vec![1.0; n_terms];
        
        for _ in 0..max_outer_iter {
            let old_lambdas = lambdas.clone();
            
            for term_idx in 0..n_terms {
                // Optimize this term's lambda while holding others fixed
                let result = brent_minimize(
                    |log_lam| {
                        let mut test_lambdas = lambdas.clone();
                        test_lambdas[term_idx] = log_lam.exp();
                        self.evaluate_gcv(&test_lambdas)
                    },
                    log_lambda_min,
                    log_lambda_max,
                    tol,
                    30,
                );
                
                lambdas[term_idx] = result.x_min.exp();
            }
            
            // Check convergence
            let max_change: f64 = lambdas.iter()
                .zip(&old_lambdas)
                .map(|(&new, &old)| ((new - old) / old.max(1e-10)).abs())
                .fold(0.0, f64::max);
            
            if max_change < 0.01 {
                break;
            }
        }
        
        lambdas
    }
    
    /// Compute EDFs for each term at given lambdas.
    pub fn compute_edfs(&self, lambdas: &[f64]) -> Vec<f64> {
        let p = self.xtwx.nrows();
        
        let xtwx_pen = build_penalized_xtwx(
            &self.xtwx, &self.penalties, &self.col_ranges, lambdas,
        );
        
        let chol = match xtwx_pen.cholesky() {
            Some(c) => c,
            None => return vec![0.0; lambdas.len()],
        };
        
        let identity = DMatrix::identity(p, p);
        let xtwx_pen_inv = chol.solve(&identity);
        compute_smooth_edfs_from_inv(&xtwx_pen_inv, &self.xtwx, &self.col_ranges)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};
    use crate::splines::{bs_basis};
    use crate::splines::penalized::penalty_matrix;

    // =========================================================================
    // Brent's method unit tests
    // =========================================================================
    
    #[test]
    fn test_brent_minimize_quadratic() {
        // Minimize (x - 2)^2
        let result = brent_minimize(|x| (x - 2.0).powi(2), 0.0, 5.0, 1e-6, 100);
        
        assert!(result.converged);
        assert!((result.x_min - 2.0).abs() < 1e-5);
        assert!(result.f_min < 1e-10);
    }
    
    #[test]
    fn test_brent_minimize_cosine() {
        // Minimize cos(x) in [2, 5] - minimum at π ≈ 3.14159
        let result = brent_minimize(|x| x.cos(), 2.0, 5.0, 1e-6, 100);
        
        assert!(result.converged);
        assert!((result.x_min - std::f64::consts::PI).abs() < 1e-5);
    }

    #[test]
    fn test_brent_minimize_at_boundary() {
        // Monotonically decreasing in [0, 5] => minimum at b=5
        let result = brent_minimize(|x| -x, 0.0, 5.0, 1e-6, 100);
        assert!((result.x_min - 5.0).abs() < 0.01);
    }

    // =========================================================================
    // GCVCache tests
    // =========================================================================

    /// Helper: build a simple smooth regression problem.
    /// Returns (x_combined, z, w, penalty, n_parametric, col_start, col_end)
    fn simple_smooth_problem(n: usize, k: usize) -> (Array2<f64>, Array1<f64>, Array1<f64>, Array2<f64>, usize, usize, usize) {
        let x_vals: Array1<f64> = (0..n).map(|i| i as f64 * 10.0 / n as f64).collect();
        let basis = bs_basis(&x_vals, k, 3, None, false);
        let k_actual = basis.ncols();

        // Build x_combined = [intercept | basis]
        let mut x_combined = Array2::zeros((n, 1 + k_actual));
        for i in 0..n {
            x_combined[[i, 0]] = 1.0;
            for j in 0..k_actual {
                x_combined[[i, 1 + j]] = basis[[i, j]];
            }
        }

        // Simulated working response and weights (from a hypothetical IRLS iteration)
        let z: Array1<f64> = x_vals.iter().map(|&xi| 2.0 + xi.sin()).collect();
        let w: Array1<f64> = Array1::ones(n);

        let penalty = penalty_matrix(k_actual, 2);
        let n_parametric = 1;
        let col_start = 1;
        let col_end = 1 + k_actual;

        (x_combined, z, w, penalty, n_parametric, col_start, col_end)
    }

    #[test]
    fn test_gcv_cache_creation() {
        let (x, z, w, penalty, n_param, col_start, col_end) = simple_smooth_problem(100, 8);
        let cache = GCVCache::new(&x, &z, &w, &penalty, col_start, col_end, n_param);

        assert_eq!(cache.n, 100);
        assert_eq!(cache.n_parametric, 1);
        assert_eq!(cache.col_start, 1);
    }

    #[test]
    fn test_gcv_cache_evaluate_returns_finite() {
        let (x, z, w, penalty, n_param, col_start, col_end) = simple_smooth_problem(100, 8);
        let cache = GCVCache::new(&x, &z, &w, &penalty, col_start, col_end, n_param);

        // Evaluate at several log-lambda values
        for log_lam in [-5.0, -2.0, 0.0, 2.0, 5.0] {
            let gcv = cache.evaluate_gcv(log_lam);
            assert!(gcv.is_finite(), "GCV should be finite at log_lambda={}", log_lam);
            assert!(gcv >= 0.0, "GCV should be non-negative");
        }
    }

    #[test]
    fn test_gcv_cache_optimize_lambda() {
        let (x, z, w, penalty, n_param, col_start, col_end) = simple_smooth_problem(200, 10);
        let cache = GCVCache::new(&x, &z, &w, &penalty, col_start, col_end, n_param);

        let (lambda, edf, gcv) = cache.optimize_lambda(-8.0, 12.0, 1e-4);

        assert!(lambda > 0.0, "Optimal lambda should be positive");
        assert!(edf > 1.0, "EDF should be > 1");
        assert!(edf < 10.0, "EDF should be < k");
        assert!(gcv > 0.0, "GCV should be positive");
    }

    #[test]
    fn test_gcv_cache_edf_monotone_in_lambda() {
        // EDF should decrease as lambda increases (more smoothing = fewer effective params)
        let (x, z, w, penalty, n_param, col_start, col_end) = simple_smooth_problem(200, 10);
        let cache = GCVCache::new(&x, &z, &w, &penalty, col_start, col_end, n_param);

        let edf_low_lambda = cache.compute_edf(0.001);
        let edf_high_lambda = cache.compute_edf(1000.0);

        assert!(edf_low_lambda > edf_high_lambda,
            "EDF at low lambda ({:.2}) should exceed EDF at high lambda ({:.2})",
            edf_low_lambda, edf_high_lambda);
    }

    #[test]
    fn test_gcv_cache_solve_coefficients() {
        let (x, z, w, penalty, n_param, col_start, col_end) = simple_smooth_problem(100, 8);
        let cache = GCVCache::new(&x, &z, &w, &penalty, col_start, col_end, n_param);

        let beta = cache.solve_coefficients(1.0);
        assert!(beta.is_some());
        let beta = beta.unwrap();
        assert_eq!(beta.len(), x.ncols());
    }

    // =========================================================================
    // MultiTermGCVOptimizer tests
    // =========================================================================

    #[test]
    fn test_multi_term_optimizer_single_term() {
        // Multi-term optimizer with 1 term should behave like GCVCache
        let (x, z, w, penalty, n_param, col_start, col_end) = simple_smooth_problem(200, 10);
        let optimizer = MultiTermGCVOptimizer::new(
            &x, &z, &w,
            vec![penalty],
            vec![(col_start, col_end)],
            n_param,
        );

        let lambdas = optimizer.optimize_lambdas(-8.0, 12.0, 1e-4, 10);
        assert_eq!(lambdas.len(), 1);
        assert!(lambdas[0] > 0.0);

        let edfs = optimizer.compute_edfs(&lambdas);
        assert_eq!(edfs.len(), 1);
        assert!(edfs[0] > 1.0);
        assert!(edfs[0] < 10.0);
    }

    #[test]
    fn test_multi_term_optimizer_two_terms() {
        let n = 200;
        let x_vals1: Array1<f64> = (0..n).map(|i| i as f64 * 10.0 / n as f64).collect();
        let x_vals2: Array1<f64> = (0..n).map(|i| i as f64 * 5.0 / n as f64).collect();
        let basis1 = bs_basis(&x_vals1, 8, 3, None, false);
        let basis2 = bs_basis(&x_vals2, 6, 3, None, false);
        let k1 = basis1.ncols();
        let k2 = basis2.ncols();

        // Build x_combined = [intercept | basis1 | basis2]
        let p_total = 1 + k1 + k2;
        let mut x = Array2::zeros((n, p_total));
        for i in 0..n {
            x[[i, 0]] = 1.0;
            for j in 0..k1 { x[[i, 1 + j]] = basis1[[i, j]]; }
            for j in 0..k2 { x[[i, 1 + k1 + j]] = basis2[[i, j]]; }
        }

        let z: Array1<f64> = x_vals1.iter().zip(x_vals2.iter())
            .map(|(&a, &b)| 2.0 + a.sin() + 0.5 * b.cos()).collect();
        let w = Array1::ones(n);
        let penalty1 = penalty_matrix(k1, 2);
        let penalty2 = penalty_matrix(k2, 2);

        let optimizer = MultiTermGCVOptimizer::new(
            &x, &z, &w,
            vec![penalty1, penalty2],
            vec![(1, 1 + k1), (1 + k1, 1 + k1 + k2)],
            1,
        );

        let lambdas = optimizer.optimize_lambdas(-8.0, 12.0, 1e-4, 10);
        assert_eq!(lambdas.len(), 2);
        assert!(lambdas.iter().all(|&l| l > 0.0));

        let edfs = optimizer.compute_edfs(&lambdas);
        assert_eq!(edfs.len(), 2);
        assert!(edfs.iter().all(|&e| e > 0.5));
    }

    #[test]
    fn test_multi_term_evaluate_gcv_finite() {
        let (x, z, w, penalty, n_param, col_start, col_end) = simple_smooth_problem(100, 8);
        let optimizer = MultiTermGCVOptimizer::new(
            &x, &z, &w,
            vec![penalty],
            vec![(col_start, col_end)],
            n_param,
        );

        let gcv = optimizer.evaluate_gcv(&[1.0]);
        assert!(gcv.is_finite());
        assert!(gcv >= 0.0);
    }
}
