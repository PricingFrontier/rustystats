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
        let mut u = 0.0;
        
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
        let p = x.ncols();
        let k = penalty.nrows();
        
        // Convert to nalgebra - ensure arrays are contiguous first
        let x_contig = if x.is_standard_layout() { x.clone() } else { x.as_standard_layout().to_owned() };
        let z_contig = z.to_owned();
        let w_contig = w.to_owned();
        let penalty_contig = if penalty.is_standard_layout() { penalty.clone() } else { penalty.as_standard_layout().to_owned() };
        
        let x_nalg = DMatrix::from_row_slice(n, p, x_contig.as_slice().unwrap());
        let z_nalg = DVector::from_row_slice(z_contig.as_slice().unwrap());
        let w_nalg = DVector::from_row_slice(w_contig.as_slice().unwrap());
        let penalty_nalg = DMatrix::from_row_slice(k, k, penalty_contig.as_slice().unwrap());
        
        // Compute X'WX
        let mut xtwx = DMatrix::zeros(p, p);
        for i in 0..n {
            let wi = w_nalg[i];
            for j in 0..p {
                let xij_w = x_nalg[(i, j)] * wi;
                for l in j..p {
                    let val = xij_w * x_nalg[(i, l)];
                    xtwx[(j, l)] += val;
                    if l != j {
                        xtwx[(l, j)] += val;
                    }
                }
            }
        }
        
        // Compute X'Wz
        let mut xtwz = DVector::zeros(p);
        for i in 0..n {
            let wz = w_nalg[i] * z_nalg[i];
            for j in 0..p {
                xtwz[j] += x_nalg[(i, j)] * wz;
            }
        }
        
        // Compute base RSS (at lambda=0) for reference
        let rss_base = 0.0;  // Will compute if needed
        
        Self {
            xtwx,
            xtwz,
            penalty: penalty_nalg,
            col_start,
            col_end,
            n,
            n_parametric,
            rss_base,
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
        let k = self.col_end - self.col_start;
        
        // Build penalized X'WX + λS
        let mut xtwx_pen = self.xtwx.clone();
        for i in 0..k {
            for j in 0..k {
                xtwx_pen[(self.col_start + i, self.col_start + j)] += 
                    lambda * self.penalty[(i, j)];
            }
        }
        
        // Solve for coefficients: β = (X'WX + λS)⁻¹ X'Wz
        let chol = match xtwx_pen.clone().cholesky() {
            Some(c) => c,
            None => return f64::INFINITY,  // Singular - skip this lambda
        };
        
        let beta = chol.solve(&self.xtwz);
        
        // Compute fitted values and weighted RSS
        let mut rss = 0.0;
        for i in 0..self.n {
            let mut fitted = 0.0;
            for j in 0..p {
                fitted += self.x[(i, j)] * beta[j];
            }
            let resid = self.z[i] - fitted;
            rss += self.w[i] * resid * resid;
        }
        
        // Compute EDF = trace((X'WX + λS)⁻¹ X'WX)
        // We compute the inverse and then trace of product
        let identity = DMatrix::identity(p, p);
        let xtwx_pen_inv = chol.solve(&identity);
        
        // EDF for the smooth term only (not parametric part)
        let mut edf_smooth = 0.0;
        for i in self.col_start..self.col_end {
            for j in 0..p {
                edf_smooth += xtwx_pen_inv[(i, j)] * self.xtwx[(j, i)];
            }
        }
        
        // Total EDF = parametric terms + smooth EDF
        let total_edf = (self.n_parametric as f64) + edf_smooth;
        
        // GCV = n * RSS / (n - EDF)²
        let denom = (self.n as f64) - total_edf;
        if denom <= 1.0 {
            return f64::INFINITY;  // Over-parameterized
        }
        
        let gcv = (self.n as f64) * rss / (denom * denom);
        
        gcv
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
        let k = self.col_end - self.col_start;
        
        // Build penalized X'WX + λS
        let mut xtwx_pen = self.xtwx.clone();
        for i in 0..k {
            for j in 0..k {
                xtwx_pen[(self.col_start + i, self.col_start + j)] += 
                    lambda * self.penalty[(i, j)];
            }
        }
        
        let chol = match xtwx_pen.clone().cholesky() {
            Some(c) => c,
            None => return self.col_end as f64 - self.col_start as f64,
        };
        
        let identity = DMatrix::identity(p, p);
        let xtwx_pen_inv = chol.solve(&identity);
        
        // EDF for smooth term
        let mut edf_smooth = 0.0;
        for i in self.col_start..self.col_end {
            for j in 0..p {
                edf_smooth += xtwx_pen_inv[(i, j)] * self.xtwx[(j, i)];
            }
        }
        
        edf_smooth
    }
    
    /// Solve for coefficients at a specific lambda.
    pub fn solve_coefficients(&self, lambda: f64) -> Option<DVector<f64>> {
        let p = self.xtwx.nrows();
        let k = self.col_end - self.col_start;
        
        let mut xtwx_pen = self.xtwx.clone();
        for i in 0..k {
            for j in 0..k {
                xtwx_pen[(self.col_start + i, self.col_start + j)] += 
                    lambda * self.penalty[(i, j)];
            }
        }
        
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
        let p = x.ncols();
        
        // Ensure arrays are contiguous
        let x_contig = if x.is_standard_layout() { x.clone() } else { x.as_standard_layout().to_owned() };
        let z_contig = z.to_owned();
        let w_contig = w.to_owned();
        
        let x_nalg = DMatrix::from_row_slice(n, p, x_contig.as_slice().unwrap());
        let z_nalg = DVector::from_row_slice(z_contig.as_slice().unwrap());
        let w_nalg = DVector::from_row_slice(w_contig.as_slice().unwrap());
        
        // Compute X'WX
        let mut xtwx = DMatrix::zeros(p, p);
        for i in 0..n {
            let wi = w_nalg[i];
            for j in 0..p {
                let xij_w = x_nalg[(i, j)] * wi;
                for l in j..p {
                    let val = xij_w * x_nalg[(i, l)];
                    xtwx[(j, l)] += val;
                    if l != j {
                        xtwx[(l, j)] += val;
                    }
                }
            }
        }
        
        // Compute X'Wz
        let mut xtwz = DVector::zeros(p);
        for i in 0..n {
            let wz = w_nalg[i] * z_nalg[i];
            for j in 0..p {
                xtwz[j] += x_nalg[(i, j)] * wz;
            }
        }
        
        // Convert penalties
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
        
        // Build penalized X'WX
        let mut xtwx_pen = self.xtwx.clone();
        for (i, ((start, end), penalty)) in self.col_ranges.iter().zip(&self.penalties).enumerate() {
            let lambda = lambdas[i];
            let k = end - start;
            for r in 0..k {
                for c in 0..k {
                    xtwx_pen[(start + r, start + c)] += lambda * penalty[(r, c)];
                }
            }
        }
        
        // Solve
        let chol = match xtwx_pen.clone().cholesky() {
            Some(c) => c,
            None => return f64::INFINITY,
        };
        
        let beta = chol.solve(&self.xtwz);
        
        // RSS
        let mut rss = 0.0;
        for i in 0..self.n {
            let mut fitted = 0.0;
            for j in 0..p {
                fitted += self.x[(i, j)] * beta[j];
            }
            let resid = self.z[i] - fitted;
            rss += self.w[i] * resid * resid;
        }
        
        // Total EDF
        let identity = DMatrix::identity(p, p);
        let xtwx_pen_inv = chol.solve(&identity);
        
        let mut total_edf = self.n_parametric as f64;
        for (start, end) in &self.col_ranges {
            for i in *start..*end {
                for j in 0..p {
                    total_edf += xtwx_pen_inv[(i, j)] * self.xtwx[(j, i)];
                }
            }
        }
        
        let denom = (self.n as f64) - total_edf;
        if denom <= 1.0 {
            return f64::INFINITY;
        }
        
        (self.n as f64) * rss / (denom * denom)
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
        
        let mut xtwx_pen = self.xtwx.clone();
        for (i, ((start, end), penalty)) in self.col_ranges.iter().zip(&self.penalties).enumerate() {
            let lambda = lambdas[i];
            let k = end - start;
            for r in 0..k {
                for c in 0..k {
                    xtwx_pen[(start + r, start + c)] += lambda * penalty[(r, c)];
                }
            }
        }
        
        let chol = match xtwx_pen.cholesky() {
            Some(c) => c,
            None => return vec![0.0; lambdas.len()],
        };
        
        let identity = DMatrix::identity(p, p);
        let xtwx_pen_inv = chol.solve(&identity);
        
        let mut edfs = Vec::with_capacity(lambdas.len());
        for (start, end) in &self.col_ranges {
            let mut edf = 0.0;
            for i in *start..*end {
                for j in 0..p {
                    edf += xtwx_pen_inv[(i, j)] * self.xtwx[(j, i)];
                }
            }
            edfs.push(edf);
        }
        
        edfs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
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
}
