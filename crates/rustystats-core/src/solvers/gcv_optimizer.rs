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

use nalgebra::{DMatrix, DVector};
use ndarray::{Array1, Array2};

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
    let golden = 0.381966011250105; // (3 - sqrt(5)) / 2

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

/// Compute smooth EDF for given column ranges via Cholesky forward/back-substitution.
///
/// Computes trace((X'WX+λS)⁻¹ · X'WX) restricted to each term's columns,
/// without forming the full p×p inverse. For each smooth column j, we solve
/// A·z = xtwx[:,j] (O(p²) per column via Cholesky) and accumulate z[j]
/// (the diagonal element of A⁻¹·X'WX). This replaces one O(p³) full inverse
/// with k_total O(p²) solves, where k_total = sum of smooth term dimensions.
fn compute_smooth_edfs_from_chol(
    chol: &nalgebra::linalg::Cholesky<f64, nalgebra::Dyn>,
    xtwx: &DMatrix<f64>,
    col_ranges: &[(usize, usize)],
) -> Vec<f64> {
    let mut edfs = Vec::with_capacity(col_ranges.len());
    for &(start, end) in col_ranges {
        let mut edf = 0.0;
        for col in start..end {
            // Solve (X'WX + λS) · z = xtwx[:, col]  -- O(p²) per column
            let rhs = xtwx.column(col).clone_owned();
            let z = chol.solve(&rhs);
            // Only the diagonal element z[col] contributes to the trace
            edf += z[col];
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
    let x_contig = if x.is_standard_layout() {
        x.clone()
    } else {
        x.as_standard_layout().to_owned()
    };
    let x_nalg = DMatrix::from_row_slice(n, p, x_contig.as_slice().expect("contiguous"));
    let z_nalg = DVector::from_row_slice(z.to_owned().as_slice().expect("contiguous"));
    let w_nalg = DVector::from_row_slice(w.to_owned().as_slice().expect("contiguous"));
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
        let penalty_contig = if penalty.is_standard_layout() {
            penalty.clone()
        } else {
            penalty.as_standard_layout().to_owned()
        };
        let penalty_nalg =
            DMatrix::from_row_slice(k, k, penalty_contig.as_slice().expect("contiguous"));

        let (xtwx, xtwz) = compute_xtwx_xtwz_nalg(&x_nalg, &z_nalg, &w_nalg);

        Self {
            xtwx,
            xtwz,
            penalty: penalty_nalg,
            col_start,
            col_end,
            n,
            n_parametric,
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
        let col_range = (self.col_start, self.col_end);

        let xtwx_pen = build_penalized_xtwx(
            &self.xtwx,
            std::slice::from_ref(&self.penalty),
            &[col_range],
            &[lambda],
        );

        let chol = match xtwx_pen.cholesky() {
            Some(c) => c,
            None => return f64::INFINITY,
        };

        let beta = chol.solve(&self.xtwz);
        let rss = compute_weighted_rss(&self.x, &self.z, &self.w, &beta);

        let edfs = compute_smooth_edfs_from_chol(&chol, &self.xtwx, &[col_range]);
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
            50, // Max iterations
        );

        let optimal_lambda = result.x_min.exp();
        let optimal_gcv = result.f_min;

        // Compute EDF at optimal lambda
        let edf = self.compute_edf(optimal_lambda);

        (optimal_lambda, edf, optimal_gcv)
    }

    /// Compute EDF at a specific lambda.
    pub fn compute_edf(&self, lambda: f64) -> f64 {
        let col_range = (self.col_start, self.col_end);

        let xtwx_pen = build_penalized_xtwx(
            &self.xtwx,
            std::slice::from_ref(&self.penalty),
            &[col_range],
            &[lambda],
        );

        let chol = match xtwx_pen.cholesky() {
            Some(c) => c,
            None => return (self.col_end - self.col_start) as f64,
        };

        compute_smooth_edfs_from_chol(&chol, &self.xtwx, &[col_range])[0]
    }

    /// Solve for coefficients at a specific lambda.
    pub fn solve_coefficients(&self, lambda: f64) -> Option<DVector<f64>> {
        let col_range = (self.col_start, self.col_end);
        let xtwx_pen = build_penalized_xtwx(
            &self.xtwx,
            std::slice::from_ref(&self.penalty),
            &[col_range],
            &[lambda],
        );
        xtwx_pen.cholesky().map(|chol| chol.solve(&self.xtwz))
    }
}

/// Compute weighted RSS from cached matrices without needing raw data.
///
/// RSS = z'Wz - 2·β'(X'Wz) + β'(X'WX)β
///
/// This is O(p²) instead of O(n·p), eliminating the need to store the
/// n-dimensional X, z, w arrays.
fn compute_rss_from_cached(
    xtwx: &DMatrix<f64>,
    xtwz: &DVector<f64>,
    ztwz: f64,
    beta: &DVector<f64>,
) -> f64 {
    let beta_xtwz = beta.dot(xtwz);
    let beta_xtwx_beta = beta.dot(&(xtwx * beta));
    ztwz - 2.0 * beta_xtwz + beta_xtwx_beta
}

fn max_relative_lambda_change(new_lambdas: &[f64], old_lambdas: &[f64]) -> f64 {
    new_lambdas
        .iter()
        .zip(old_lambdas)
        .map(|(&new, &old)| ((new - old) / old.max(1e-10)).abs())
        .fold(0.0, f64::max)
}

fn gradient_norm(grad: &[f64]) -> f64 {
    grad.iter().map(|&g| g * g).sum::<f64>().sqrt()
}

fn solve_reml_newton_delta(
    hess: &DMatrix<f64>,
    g_vec: &DVector<f64>,
    grad_norm: f64,
) -> DVector<f64> {
    match hess.clone().cholesky() {
        Some(h_chol) => h_chol.solve(g_vec) * -1.0,
        None => {
            let reg = hess.diagonal().amax() * 0.1 + 1e-6;
            let h_reg = hess + &DMatrix::from_diagonal(&DVector::from_element(hess.nrows(), reg));
            match h_reg.cholesky() {
                Some(h_chol) => h_chol.solve(g_vec) * -1.0,
                None => g_vec * (-1.0 / grad_norm),
            }
        }
    }
}

fn halve_step(step: f64) -> f64 {
    step * 0.5
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
    /// z'Wz scalar for cached RSS computation
    pub ztwz: f64,
}

impl MultiTermGCVOptimizer {
    /// Create optimizer from pre-computed X'WX, X'Wz, and z'Wz.
    ///
    /// This is the fast path — avoids the O(n·p²) X'WX recomputation.
    /// The caller (smooth_glm IRLS loop) computes X'WX once and passes it here.
    pub fn new_from_cached(
        xtwx: DMatrix<f64>,
        xtwz: DVector<f64>,
        ztwz: f64,
        penalties: Vec<Array2<f64>>,
        col_ranges: Vec<(usize, usize)>,
        n: usize,
        n_parametric: usize,
    ) -> Self {
        let penalties_nalg: Vec<DMatrix<f64>> = penalties
            .iter()
            .map(|pen| {
                let contig = if pen.is_standard_layout() {
                    pen.clone()
                } else {
                    pen.as_standard_layout().to_owned()
                };
                DMatrix::from_row_slice(
                    pen.nrows(),
                    pen.ncols(),
                    contig.as_slice().expect("contiguous"),
                )
            })
            .collect();

        Self {
            xtwx,
            xtwz,
            penalties: penalties_nalg,
            col_ranges,
            n,
            n_parametric,
            ztwz,
        }
    }

    /// Create optimizer from raw matrices (legacy path — recomputes X'WX).
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

        // Compute z'Wz
        let ztwz = z_nalg
            .iter()
            .zip(w_nalg.iter())
            .map(|(&zi, &wi)| wi * zi * zi)
            .sum::<f64>();

        let penalties_nalg: Vec<DMatrix<f64>> = penalties
            .iter()
            .map(|pen| {
                DMatrix::from_row_slice(
                    pen.nrows(),
                    pen.ncols(),
                    pen.as_slice().expect("contiguous"),
                )
            })
            .collect();

        Self {
            xtwx,
            xtwz,
            penalties: penalties_nalg,
            col_ranges,
            n,
            n_parametric,
            ztwz,
        }
    }

    /// Evaluate GCV for given lambdas.
    pub fn evaluate_gcv(&self, lambdas: &[f64]) -> f64 {
        let xtwx_pen = build_penalized_xtwx(&self.xtwx, &self.penalties, &self.col_ranges, lambdas);

        let chol = match xtwx_pen.cholesky() {
            Some(c) => c,
            None => return f64::INFINITY,
        };

        let beta = chol.solve(&self.xtwz);

        // Compute RSS from cached matrices: O(p²) instead of O(n·p)
        let rss = compute_rss_from_cached(&self.xtwx, &self.xtwz, self.ztwz, &beta);

        let edfs = compute_smooth_edfs_from_chol(&chol, &self.xtwx, &self.col_ranges);
        let total_edf = (self.n_parametric as f64) + edfs.iter().sum::<f64>();

        gcv_from_rss_edf(self.n, rss, total_edf)
    }

    /// Optimize all lambdas using coordinate descent.
    ///
    /// `initial_lambdas` provides a warm start from the previous iteration's
    /// optimal lambdas. This reduces the number of Brent evaluations needed
    /// since the starting point is already close to the optimum.
    pub fn optimize_lambdas(
        &self,
        initial_lambdas: &[f64],
        log_lambda_min: f64,
        log_lambda_max: f64,
        tol: f64,
        max_outer_iter: usize,
    ) -> Vec<f64> {
        let n_terms = self.penalties.len();
        let mut lambdas: Vec<f64> = initial_lambdas.to_vec();
        debug_assert_eq!(lambdas.len(), n_terms);

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
            let max_change = max_relative_lambda_change(&lambdas, &old_lambdas);

            if max_change < 0.01 {
                break;
            }
        }

        lambdas
    }

    /// Evaluate the REML criterion for given lambdas (Wood, 2011).
    ///
    /// R(rho) = RSS + penalty + log|H_p| - sum_j M_j * rho_j
    fn evaluate_reml_internal(&self, lambdas: &[f64], penalty_ranks: &[f64]) -> f64 {
        let xtwx_pen = build_penalized_xtwx(&self.xtwx, &self.penalties, &self.col_ranges, lambdas);
        let chol = match xtwx_pen.cholesky() {
            Some(c) => c,
            None => return f64::INFINITY,
        };
        let beta = chol.solve(&self.xtwz);
        let rss = compute_rss_from_cached(&self.xtwx, &self.xtwz, self.ztwz, &beta);

        let mut penalty_value = 0.0;
        for (j, (&(start, end), pen)) in self
            .col_ranges
            .iter()
            .zip(self.penalties.iter())
            .enumerate()
        {
            let beta_sub = beta.rows(start, end - start);
            penalty_value += lambdas[j] * beta_sub.dot(&(pen * beta_sub));
        }

        let l_mat = chol.l();
        let log_det: f64 = (0..l_mat.nrows())
            .map(|i| {
                let d = l_mat[(i, i)];
                if d > 0.0 {
                    d.ln()
                } else {
                    0.0
                }
            })
            .sum::<f64>()
            * 2.0;

        let log_lam_term: f64 = penalty_ranks
            .iter()
            .zip(lambdas.iter())
            .map(|(&m, &lam)| m * lam.max(1e-30).ln())
            .sum();

        rss + penalty_value + log_det - log_lam_term
    }

    /// Optimize all lambdas jointly using Newton's method on the REML
    /// criterion (Wood, 2011). This matches scam's approach.
    ///
    /// Gradient:
    ///   dR/d(rho_j) = lambda_j * beta' S_j beta + lambda_j * tr(H_p^{-1} S_j) - M_j
    ///
    /// Approximate Hessian:
    ///   d²R/d(rho_j)d(rho_l) = lambda_j * lambda_l * tr(H_p^{-1} S_j H_p^{-1} S_l)
    ///   diagonal += lambda_j * [beta' S_j beta + tr(H_p^{-1} S_j)]
    pub fn optimize_lambdas_reml(
        &self,
        initial_lambdas: &[f64],
        log_lambda_min: f64,
        log_lambda_max: f64,
        tol: f64,
        max_iter: usize,
    ) -> Vec<f64> {
        let m = self.penalties.len();
        let p = self.xtwx.nrows();
        let mut rho: Vec<f64> = initial_lambdas
            .iter()
            .map(|&lam| lam.max(1e-30).ln())
            .collect();

        // Penalty ranks (computed once)
        let penalty_ranks: Vec<f64> = self
            .penalties
            .iter()
            .map(|pen| {
                let eig = pen.clone().symmetric_eigen();
                eig.eigenvalues.iter().filter(|&&v| v.abs() > 1e-10).count() as f64
            })
            .collect();

        for _ in 0..max_iter {
            let lambdas: Vec<f64> = rho.iter().map(|&r| r.exp()).collect();

            // Build penalized system
            let xtwx_pen =
                build_penalized_xtwx(&self.xtwx, &self.penalties, &self.col_ranges, &lambdas);
            let chol = match xtwx_pen.cholesky() {
                Some(c) => c,
                None => break,
            };
            let beta = chol.solve(&self.xtwz);

            // Per-term: penalty value and tr(H_p^{-1} * lambda_j * S_j)
            let mut pen_vals = vec![0.0; m];
            let mut traces = vec![0.0; m];

            // Also store forward-solved penalty columns for Hessian
            let mut solved: Vec<DMatrix<f64>> = Vec::with_capacity(m);

            for (j, (&(start, end), pen)) in self
                .col_ranges
                .iter()
                .zip(self.penalties.iter())
                .enumerate()
            {
                let k = end - start;
                let beta_sub = beta.rows(start, k);
                pen_vals[j] = beta_sub.dot(&(pen * beta_sub));

                // Solve H_p * z = lambda_j * S_j[:, c] (embedded) for each column c
                let mut v = DMatrix::zeros(p, k);
                let mut tr = 0.0;
                for c in 0..k {
                    let mut rhs = DVector::zeros(p);
                    for (r, &val) in pen.column(c).iter().enumerate() {
                        rhs[start + r] = lambdas[j] * val;
                    }
                    let z = chol.solve(&rhs);
                    tr += z[start + c];
                    v.set_column(c, &z);
                }
                traces[j] = tr;
                solved.push(v);
            }

            // Gradient
            let mut grad = vec![0.0; m];
            for j in 0..m {
                grad[j] = lambdas[j] * pen_vals[j] + traces[j] - penalty_ranks[j];
            }

            // Check gradient convergence
            let grad_norm = gradient_norm(&grad);
            if grad_norm < tol {
                break;
            }

            // Hessian
            // Cross-term: tr(H^{-1} S_j_emb * H^{-1} S_l_emb)
            //   = sum_i (solved[j] row i) dot (solved[l] row i)
            //   restricted to the overlap of column ranges
            // But since solved[j] is p×k_j and represents columns start_j..end_j,
            // we need: sum_{i=0}^{p-1} sum_{k} [H^{-1}S_j]_{ik} [H^{-1}S_l]_{ki}
            // = sum_i sum_{a in 0..kj} solved[j][i,a] * solved[l][sj+a-sl, ?]
            // This only contributes when sj+a falls within [sl, el).
            //
            // Simpler: assemble as dense p×p products for small penalty blocks.
            let mut hess = DMatrix::zeros(m, m);
            for j in 0..m {
                for l in j..m {
                    let (sj, ej) = self.col_ranges[j];
                    let (sl, el) = self.col_ranges[l];
                    let kj = ej - sj;

                    // tr(A * B) where A = H_p^{-1}(lambda_j S_j)_emb, B = H_p^{-1}(lambda_l S_l)_emb
                    // = sum_k (row k of A) dot (col k of B)
                    // A has nonzero cols sj..ej, B has nonzero cols sl..el
                    let mut cross = 0.0;
                    for k in sl..el {
                        for a in 0..kj {
                            let i = sj + a;
                            cross += solved[j][(k, a)] * solved[l][(i, k - sl)];
                        }
                    }

                    hess[(j, l)] = cross;
                    if j != l {
                        hess[(l, j)] = cross;
                    }
                }
                // Diagonal extra term
                hess[(j, j)] += lambdas[j] * pen_vals[j] + traces[j];
            }

            // Newton step: delta = -H^{-1} g
            let g_vec = DVector::from_vec(grad);
            let delta = solve_reml_newton_delta(&hess, &g_vec, grad_norm);

            // Step halving on REML
            let current_reml = self.evaluate_reml_internal(&lambdas, &penalty_ranks);
            let mut step = 1.0;
            let mut accepted = false;
            for _ in 0..20 {
                let trial_rho: Vec<f64> = rho
                    .iter()
                    .enumerate()
                    .map(|(j, &r)| (r + step * delta[j]).clamp(log_lambda_min, log_lambda_max))
                    .collect();
                let trial_lam: Vec<f64> = trial_rho.iter().map(|&r| r.exp()).collect();
                let trial_reml = self.evaluate_reml_internal(&trial_lam, &penalty_ranks);
                if trial_reml < current_reml {
                    rho = trial_rho;
                    accepted = true;
                    break;
                }
                step = halve_step(step);
            }
            if !accepted {
                break;
            }
        }

        rho.iter().map(|&r| r.exp()).collect()
    }

    /// Compute EDFs for each term at given lambdas.
    pub fn compute_edfs(&self, lambdas: &[f64]) -> Vec<f64> {
        let xtwx_pen = build_penalized_xtwx(&self.xtwx, &self.penalties, &self.col_ranges, lambdas);

        let chol = match xtwx_pen.cholesky() {
            Some(c) => c,
            None => return vec![0.0; lambdas.len()],
        };

        compute_smooth_edfs_from_chol(&chol, &self.xtwx, &self.col_ranges)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::splines::bs_basis;
    use crate::splines::penalized::penalty_matrix;
    use ndarray::{Array1, Array2};
    use std::cell::RefCell;

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

    #[test]
    fn test_brent_minimize_reports_nonconvergence_when_budget_exhausted() {
        let result = brent_minimize(|x| (x - 1.0).powi(2), -5.0, 5.0, 1e-12, 0);
        assert!(!result.converged);
        assert_eq!(result.iterations, 0);
        assert!(result.x_min.is_finite());
        assert!(result.f_min.is_finite());
    }

    #[test]
    fn test_brent_initial_point_and_first_golden_step_are_exact() {
        let calls = RefCell::new(Vec::new());
        let result = brent_minimize(
            |x| {
                calls.borrow_mut().push(x);
                (x - 2.0).powi(2)
            },
            0.0,
            5.0,
            1e-6,
            1,
        );

        let calls = calls.into_inner();
        let golden = 0.381966011250105;
        let expected_x0 = golden * 5.0;
        let expected_x1 = expected_x0 + golden * (5.0 - expected_x0);

        assert_eq!(calls.len(), 2);
        assert!((calls[0] - expected_x0).abs() < 1e-15);
        assert!((calls[1] - expected_x1).abs() < 1e-15);
        assert!(!result.converged);
        assert_eq!(result.iterations, 1);
        assert!((result.x_min - expected_x0).abs() < 1e-15);
        assert!((result.f_min - (expected_x0 - 2.0).powi(2)).abs() < 1e-15);
    }

    #[test]
    fn test_brent_asymmetric_trace_matches_reference_sequence() {
        let calls = RefCell::new(Vec::new());
        let objective = |x: f64| {
            let c = x - 1.2345;
            c * c + 0.03 * c * c * c + 0.2 * (0.7 * x).sin()
        };

        let result = brent_minimize(
            |x| {
                calls.borrow_mut().push(x);
                objective(x)
            },
            -2.0,
            5.0,
            1e-8,
            12,
        );

        let expected_calls = [
            0.6737620787507348,
            2.3262379212492625,
            -0.3475241575014725,
            1.169029145879195,
            1.1800629128458504,
            1.1873616520504968,
            1.1872137261885387,
            1.1872122491447525,
            1.1872122616720797,
            1.1872122736442023,
        ];
        let calls = calls.into_inner();
        assert_eq!(calls.len(), expected_calls.len());
        for (actual, expected) in calls.iter().zip(expected_calls) {
            assert!((*actual - expected).abs() < 1e-10, "{actual} != {expected}");
        }

        assert!(result.converged);
        assert_eq!(result.iterations, 10);
        assert!((result.x_min - 1.1872122616720797).abs() < 1e-10);
        assert!((result.f_min - objective(result.x_min)).abs() < 1e-15);
    }

    fn assert_brent_trace<F>(
        objective: F,
        interval: (f64, f64),
        tol: f64,
        max_iter: usize,
        expected_calls: &[f64],
        expected_x: f64,
        expected_iterations: usize,
        expected_converged: bool,
    ) where
        F: Fn(f64) -> f64,
    {
        let calls = RefCell::new(Vec::new());
        let result = brent_minimize(
            |x| {
                calls.borrow_mut().push(x);
                objective(x)
            },
            interval.0,
            interval.1,
            tol,
            max_iter,
        );

        let calls = calls.into_inner();
        assert_eq!(calls.len(), expected_calls.len(), "actual calls: {calls:?}");
        for (actual, expected) in calls.iter().zip(expected_calls.iter()) {
            assert!(
                (*actual - *expected).abs() < 1e-12,
                "{actual} != {expected}"
            );
        }
        assert_eq!(result.iterations, expected_iterations);
        assert_eq!(result.converged, expected_converged);
        assert!(
            (result.x_min - expected_x).abs() < 1e-12,
            "x_min {} != {}",
            result.x_min,
            expected_x
        );
        assert!((result.f_min - objective(result.x_min)).abs() < 1e-15);
    }

    #[test]
    fn test_brent_endpoint_and_flat_traces_match_reference_sequences() {
        assert_brent_trace(
            |x| (x - 0.001).powi(2),
            (0.0, 5.0),
            1e-6,
            10,
            &[
                1.909830056250525,
                3.0901699437494732,
                1.1803398874989484,
                0.7294901687515774,
                0.4508497187473714,
                0.27864045000420623,
                0.17220926874316528,
                0.10643118126104105,
                0.06577808748212428,
                0.04065309377891681,
                0.025124993703207497,
            ],
            0.025124993703207497,
            10,
            false,
        );

        assert_brent_trace(
            |x| -x,
            (0.0, 5.0),
            1e-6,
            10,
            &[
                1.909830056250525,
                3.0901699437494732,
                3.8196601125010505,
                4.270509831248422,
                4.549150281252628,
                4.721359549995793,
                4.827790731256835,
                4.893568818738959,
                4.934221912517876,
                4.959346906221084,
                4.974875006296793,
            ],
            4.974875006296793,
            10,
            false,
        );

        assert_brent_trace(
            |x| (x - 1.0).powi(4) + 0.01 * (x + 0.3),
            (-2.0, 3.0),
            1e-7,
            12,
            &[
                -0.09016994374947496,
                1.0901699437494732,
                1.8196601125010505,
                1.1242094480968758,
                1.0986607675961519,
                0.6393202250021023,
                0.99255142537187,
                0.8312610217896113,
                0.7990171324069528,
                0.8883544418292069,
                0.9281541480172908,
                0.8684949291224098,
                0.8662946339689632,
            ],
            0.8662946339689632,
            12,
            false,
        );
    }

    fn poly_sin_objective(
        center: f64,
        quadratic: f64,
        cubic: f64,
        quartic: f64,
        sine: f64,
        frequency: f64,
    ) -> impl Fn(f64) -> f64 {
        move |x| {
            let c = x - center;
            quadratic * c * c
                + cubic * c * c * c
                + quartic * c * c * c * c
                + sine * (frequency * x).sin()
        }
    }

    #[test]
    fn test_brent_adversarial_parabolic_traces_match_reference_sequences() {
        assert_brent_trace(
            poly_sin_objective(
                -2.0079566942191485,
                0.9660555800788191,
                -0.1315883105115243,
                0.038732269809014855,
                0.06600816872886128,
                0.5718819950375469,
            ),
            (-2.500280764727953, -1.9036736548858035),
            4.181522340479938e-9,
            11,
            &[
                -2.272397126698094,
                -2.131557292915663,
                -2.0445134886682346,
                -2.0187565732807196,
                -2.0161337322278388,
                -2.015880233003032,
                -2.015874062923127,
                -2.0158740105756556,
                -2.0158740020462336,
                -2.0158740191050777,
            ],
            -2.0158740105756556,
            10,
            true,
        );

        assert_brent_trace(
            poly_sin_objective(
                -3.844917527402547,
                1.78231679587613,
                -0.21342502460256496,
                0.130508061371047,
                -0.3941068955843414,
                1.7672496924519376,
            ),
            (-4.959960165311165, 1.8470899940559056),
            8.095965392590611e-9,
            11,
            &[
                -2.359898367558334,
                -0.7529718036969284,
                -3.35303360144976,
                -3.3366966175625565,
                -3.5717042817484526,
                -4.101970844187392,
                -3.6336634647693913,
                -3.643762350273735,
                -3.6459955636414305,
                -3.646084569834434,
                -3.646086932257349,
                -3.646086961875943,
            ],
            -3.646086961875943,
            11,
            false,
        );

        assert_brent_trace(
            poly_sin_objective(
                0.2790335477615429,
                2.492018446003564,
                0.30712453305908294,
                0.04646162259549243,
                0.3731899671198792,
                0.6976749896671028,
            ),
            (-3.6042733806295537, 1.8699036309347008),
            1.0221065891057483e-5,
            13,
            &[
                -1.5133238226453356,
                -0.2210459270495193,
                0.5776257353388833,
                0.22448312947642018,
                0.2180985898492533,
                0.22672261760554405,
                0.22695238236139428,
                0.22695006256614003,
                0.22694774279459656,
            ],
            0.22695006256614003,
            9,
            true,
        );

        assert_brent_trace(
            poly_sin_objective(
                -2.9101750698225697,
                2.279222074664379,
                -0.03913473772467524,
                0.03638751543055143,
                0.41893464981372175,
                0.623938491662846,
            ),
            (-3.4143278969004527, 4.210893887201809),
            7.320960179661741e-8,
            7,
            &[
                -0.5017523471295022,
                1.2983183374308562,
                -1.6142572123400938,
                -2.3483098997061904,
                -2.801992649131509,
                -2.891360476322163,
                -2.896865744021123,
                -2.8967336001960233,
            ],
            -2.8967336001960233,
            7,
            false,
        );

        assert_brent_trace(
            poly_sin_objective(
                6.083464029445333,
                2.6776866501438326,
                0.20816993851109322,
                0.029303269507140507,
                -0.17408438668213622,
                1.9466131687817783,
            ),
            (-0.21485418970690606, 7.476125579341503),
            1.0411198330114626e-9,
            13,
            &[
                2.7228386752817686,
                4.538432714352826,
                5.660531540270444,
                6.072673456086898,
                6.164800561809594,
                6.135388435751589,
                6.134330352058865,
                6.134478252685441,
                6.13447645993344,
                6.1344764700510215,
                6.134476476537746,
            ],
            6.1344764700510215,
            11,
            true,
        );

        assert_brent_trace(
            poly_sin_objective(
                1.3260498661622095,
                0.5895225514974294,
                -0.2753445983417371,
                0.02506353671029482,
                0.19014477481947423,
                2.163490520714161,
            ),
            (-0.3282906239608314, 1.6942383466264188),
            1.2505142677956984e-8,
            9,
            &[
                0.44424669957216134,
                0.921701023093425,
                1.2167840231051545,
                1.5940544276511397,
                1.4499499560639964,
                1.5390114174356944,
                1.6323212795735307,
                1.6559714947040276,
                1.6705881314959219,
                1.6796217098345247,
            ],
            1.6796217098345247,
            9,
            false,
        );

        assert_brent_trace(
            poly_sin_objective(
                1.378041294354516,
                0.4823095462273046,
                0.03876200502708116,
                0.16558739428665317,
                0.4998867018614851,
                1.85078244187337,
            ),
            (0.6236148340993282, 6.8707564867798565),
            1.0110147779963037e-9,
            14,
            &[
                3.0098106128880984,
                4.4845607079910845,
                2.098364929202314,
                2.2663739913192473,
                2.0619109352185205,
                2.0066530014393553,
                1.9954810048604184,
                1.9929985544393423,
                1.992638375694634,
                1.9926237407022347,
                1.992623377428286,
                1.9926233753137144,
                1.9926233731991427,
            ],
            1.9926233753137144,
            13,
            true,
        );
    }

    // =========================================================================
    // GCVCache tests
    // =========================================================================

    type SmoothProblem = (
        Array2<f64>,
        Array1<f64>,
        Array1<f64>,
        Array2<f64>,
        usize,
        usize,
        usize,
    );

    /// Helper: build a simple smooth regression problem.
    /// Returns (x_combined, z, w, penalty, n_parametric, col_start, col_end)
    fn simple_smooth_problem(n: usize, k: usize) -> SmoothProblem {
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
    fn test_cached_matrix_helpers_match_direct_weighted_formula() {
        let x = DMatrix::from_row_slice(3, 2, &[1.0, 2.0, 1.0, -1.0, 0.5, 3.0]);
        let z = DVector::from_row_slice(&[2.0, -1.0, 4.0]);
        let w = DVector::from_row_slice(&[0.5, 2.0, 1.5]);

        let (xtwx, xtwz) = compute_xtwx_xtwz_nalg(&x, &z, &w);
        let w_diag = DMatrix::from_diagonal(&w);
        let direct_xtwx = x.transpose() * &w_diag * &x;
        let direct_xtwz = x.transpose() * &w_diag * &z;

        for i in 0..xtwx.nrows() {
            for j in 0..xtwx.ncols() {
                assert!((xtwx[(i, j)] - direct_xtwx[(i, j)]).abs() < 1e-12);
            }
        }
        for i in 0..xtwz.len() {
            assert!((xtwz[i] - direct_xtwz[i]).abs() < 1e-12);
        }

        let beta = DVector::from_row_slice(&[0.25, 1.5]);
        let raw_rss = compute_weighted_rss(&x, &z, &w, &beta);
        let ztwz = z.dot(&(w_diag * &z));
        let cached_rss = compute_rss_from_cached(&xtwx, &xtwz, ztwz, &beta);
        assert!((raw_rss - cached_rss).abs() < 1e-12);

        let penalty = DMatrix::from_row_slice(1, 1, &[3.0]);
        let penalized = build_penalized_xtwx(&xtwx, &[penalty], &[(1, 2)], &[2.0]);
        assert!((penalized[(1, 1)] - (xtwx[(1, 1)] + 6.0)).abs() < 1e-12);
        assert_eq!(penalized[(0, 0)], xtwx[(0, 0)]);

        assert!(gcv_from_rss_edf(5, 10.0, 4.5).is_infinite());
        assert!((gcv_from_rss_edf(10, 5.0, 2.0) - 50.0 / 64.0).abs() < 1e-12);
    }

    #[test]
    fn test_max_relative_lambda_change_uses_relative_scale_and_zero_guard() {
        let old = [1.0, 10.0, 0.0, 4.0];
        let new = [1.5, 5.0, 2.0e-10, 3.0];
        assert!((max_relative_lambda_change(&new, &old) - 2.0).abs() < 1e-12);

        let unchanged = [0.5, 2.0, 8.0];
        assert_eq!(max_relative_lambda_change(&unchanged, &unchanged), 0.0);
    }

    #[test]
    fn test_reml_newton_helper_math_is_exact_on_core_branches() {
        assert!((gradient_norm(&[3.0, 4.0]) - 5.0).abs() < 1e-12);
        assert_eq!(halve_step(1.0), 0.5);
        assert_eq!(halve_step(0.125), 0.0625);

        let positive_hess = DMatrix::from_diagonal(&DVector::from_vec(vec![2.0, 8.0]));
        let g_vec = DVector::from_vec(vec![4.0, -16.0]);
        let delta = solve_reml_newton_delta(&positive_hess, &g_vec, 5.0);
        assert!((delta[0] + 2.0).abs() < 1e-12);
        assert!((delta[1] - 2.0).abs() < 1e-12);

        let singular_hess = DMatrix::zeros(2, 2);
        let singular_delta = solve_reml_newton_delta(&singular_hess, &g_vec, 5.0);
        assert!((singular_delta[0] + 4.0e6).abs() < 1e-6);
        assert!((singular_delta[1] - 16.0e6).abs() < 1e-6);

        let indefinite_hess = DMatrix::from_diagonal(&DVector::from_vec(vec![-1.0, -2.0]));
        let fallback_g = DVector::from_vec(vec![3.0, 4.0]);
        let fallback_delta = solve_reml_newton_delta(&indefinite_hess, &fallback_g, 5.0);
        assert!((fallback_delta[0] + 0.6).abs() < 1e-12);
        assert!((fallback_delta[1] + 0.8).abs() < 1e-12);
    }

    #[test]
    fn test_to_nalgebra_accepts_nonstandard_layout() {
        let x = Array2::from_shape_vec((2, 3), vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0])
            .expect("shape")
            .reversed_axes();
        assert!(!x.is_standard_layout());
        let z = Array1::from_vec(vec![10.0, 20.0, 30.0]);
        let w = Array1::from_vec(vec![1.0, 0.5, 2.0]);

        let (x_nalg, z_nalg, w_nalg) = to_nalgebra(&x, &z, &w);

        assert_eq!(x_nalg.nrows(), 3);
        assert_eq!(x_nalg.ncols(), 2);
        assert_eq!(x_nalg[(0, 0)], 1.0);
        assert_eq!(x_nalg[(0, 1)], 2.0);
        assert_eq!(x_nalg[(2, 0)], 5.0);
        assert_eq!(x_nalg[(2, 1)], 6.0);
        assert_eq!(z_nalg[1], 20.0);
        assert_eq!(w_nalg[2], 2.0);
    }

    #[test]
    fn test_gcv_cache_accepts_nonstandard_penalty_layout() {
        let x = Array2::from_shape_vec(
            (4, 3),
            vec![
                1.0, 0.0, 1.0, 1.0, 1.0, 0.5, 1.0, 2.0, 0.25, 1.0, 3.0, 0.125,
            ],
        )
        .expect("shape");
        let z = Array1::from_vec(vec![1.0, 2.0, 2.5, 3.0]);
        let w = Array1::ones(4);
        let penalty = Array2::from_shape_vec((2, 2), vec![1.0, 0.25, 0.25, 2.0])
            .expect("shape")
            .reversed_axes();
        assert!(!penalty.is_standard_layout());

        let cache = GCVCache::new(&x, &z, &w, &penalty, 1, 3, 1);

        assert!(cache.evaluate_gcv(0.0).is_finite());
        assert!(cache.compute_edf(1.0).is_finite());
    }

    #[test]
    fn test_gcv_cache_matches_closed_form_orthogonal_problem() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, -1.0, 1.0, 0.0, 1.0, 1.0]).expect("shape");
        let z = Array1::from_vec(vec![1.0, 2.0, 5.0]);
        let w = Array1::ones(3);
        let penalty = Array2::from_shape_vec((1, 1), vec![1.0]).expect("shape");
        let cache = GCVCache::new(&x, &z, &w, &penalty, 1, 2, 1);

        let beta = cache
            .solve_coefficients(2.0)
            .expect("orthogonal penalized system is positive definite");
        assert!((beta[0] - 8.0 / 3.0).abs() < 1e-12);
        assert!((beta[1] - 1.0).abs() < 1e-12);
        assert!((cache.compute_edf(2.0) - 0.5).abs() < 1e-12);

        let expected_rss = 8.0 / 3.0;
        let expected_total_edf = 1.0 + 0.5;
        let expected_gcv = 3.0 * expected_rss / ((3.0_f64 - expected_total_edf).powi(2));
        assert!((cache.evaluate_gcv(2.0_f64.ln()) - expected_gcv).abs() < 1e-12);
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
            assert!(
                gcv.is_finite(),
                "GCV should be finite at log_lambda={}",
                log_lam
            );
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

        assert!(
            edf_low_lambda > edf_high_lambda,
            "EDF at low lambda ({:.2}) should exceed EDF at high lambda ({:.2})",
            edf_low_lambda,
            edf_high_lambda
        );
    }

    #[test]
    fn test_gcv_cache_solve_coefficients() {
        let (x, z, w, penalty, n_param, col_start, col_end) = simple_smooth_problem(100, 8);
        let cache = GCVCache::new(&x, &z, &w, &penalty, col_start, col_end, n_param);

        let beta = cache.solve_coefficients(1.0);
        assert!(beta.is_some());
        let beta = beta.expect("test setup should be valid");
        assert_eq!(beta.len(), x.ncols());
    }

    #[test]
    fn test_gcv_cache_singular_penalized_system_degrades() {
        let x = Array2::zeros((4, 2));
        let z = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let w = Array1::ones(4);
        let penalty = Array2::zeros((1, 1));
        let cache = GCVCache::new(&x, &z, &w, &penalty, 1, 2, 1);

        assert!(cache.evaluate_gcv(0.0).is_infinite());
        assert_eq!(cache.compute_edf(1.0), 1.0);
        assert!(cache.solve_coefficients(1.0).is_none());
    }

    // =========================================================================
    // MultiTermGCVOptimizer tests
    // =========================================================================

    #[test]
    fn test_multi_term_optimizer_single_term() {
        // Multi-term optimizer with 1 term should behave like GCVCache
        let (x, z, w, penalty, n_param, col_start, col_end) = simple_smooth_problem(200, 10);
        let optimizer = MultiTermGCVOptimizer::new(
            &x,
            &z,
            &w,
            vec![penalty],
            vec![(col_start, col_end)],
            n_param,
        );

        let lambdas = optimizer.optimize_lambdas(&[1.0], -8.0, 12.0, 1e-4, 10);
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
            for j in 0..k1 {
                x[[i, 1 + j]] = basis1[[i, j]];
            }
            for j in 0..k2 {
                x[[i, 1 + k1 + j]] = basis2[[i, j]];
            }
        }

        let z: Array1<f64> = x_vals1
            .iter()
            .zip(x_vals2.iter())
            .map(|(&a, &b)| 2.0 + a.sin() + 0.5 * b.cos())
            .collect();
        let w = Array1::ones(n);
        let penalty1 = penalty_matrix(k1, 2);
        let penalty2 = penalty_matrix(k2, 2);

        let optimizer = MultiTermGCVOptimizer::new(
            &x,
            &z,
            &w,
            vec![penalty1, penalty2],
            vec![(1, 1 + k1), (1 + k1, 1 + k1 + k2)],
            1,
        );

        let lambdas = optimizer.optimize_lambdas(&[1.0, 1.0], -8.0, 12.0, 1e-4, 10);
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
            &x,
            &z,
            &w,
            vec![penalty],
            vec![(col_start, col_end)],
            n_param,
        );

        let gcv = optimizer.evaluate_gcv(&[1.0]);
        assert!(gcv.is_finite());
        assert!(gcv >= 0.0);
    }

    #[test]
    fn test_multi_term_cached_constructor_matches_raw_constructor() {
        let (x, z, w, penalty, n_param, col_start, col_end) = simple_smooth_problem(80, 7);
        let raw = MultiTermGCVOptimizer::new(
            &x,
            &z,
            &w,
            vec![penalty.clone()],
            vec![(col_start, col_end)],
            n_param,
        );

        let cached = MultiTermGCVOptimizer::new_from_cached(
            raw.xtwx.clone(),
            raw.xtwz.clone(),
            raw.ztwz,
            vec![penalty.reversed_axes()],
            vec![(col_start, col_end)],
            raw.n,
            raw.n_parametric,
        );

        let raw_gcv = raw.evaluate_gcv(&[0.75]);
        let cached_gcv = cached.evaluate_gcv(&[0.75]);
        assert!((raw_gcv - cached_gcv).abs() < 1e-10);
        assert_eq!(cached.penalties[0].nrows(), col_end - col_start);
    }

    #[test]
    fn test_multi_term_gcv_matches_closed_form_orthogonal_problem() {
        let optimizer = MultiTermGCVOptimizer::new_from_cached(
            DMatrix::from_diagonal(&DVector::from_vec(vec![4.0, 4.0, 4.0])),
            DVector::from_vec(vec![14.0, 8.0, 6.0]),
            78.0,
            vec![
                Array2::from_shape_vec((1, 1), vec![1.0]).expect("shape"),
                Array2::from_shape_vec((1, 1), vec![1.0]).expect("shape"),
            ],
            vec![(1, 2), (2, 3)],
            4,
            1,
        );

        let lambdas = [1.0, 3.0];
        let edfs = optimizer.compute_edfs(&lambdas);
        assert!((edfs[0] - 4.0 / 5.0).abs() < 1e-12);
        assert!((edfs[1] - 4.0 / 7.0).abs() < 1e-12);

        let beta0 = 14.0 / 4.0;
        let beta1 = 8.0 / 5.0;
        let beta2 = 6.0 / 7.0;
        let rss = 78.0 - 2.0 * (beta0 * 14.0 + beta1 * 8.0 + beta2 * 6.0)
            + 4.0 * (beta0 * beta0 + beta1 * beta1 + beta2 * beta2);
        let total_edf = 1.0 + edfs[0] + edfs[1];
        let expected_gcv = 4.0 * rss / ((4.0_f64 - total_edf).powi(2));

        assert!((optimizer.evaluate_gcv(&lambdas) - expected_gcv).abs() < 1e-12);
    }

    #[test]
    fn test_reml_internal_matches_closed_form_single_penalty_problem() {
        let optimizer = MultiTermGCVOptimizer::new_from_cached(
            DMatrix::from_diagonal(&DVector::from_vec(vec![3.0, 2.0])),
            DVector::from_vec(vec![8.0, 4.0]),
            30.0,
            vec![Array2::from_shape_vec((1, 1), vec![1.0]).expect("shape")],
            vec![(1, 2)],
            3,
            1,
        );

        let expected = 14.0 / 3.0 + 6.0_f64.ln();
        assert!((optimizer.evaluate_reml_internal(&[2.0], &[1.0]) - expected).abs() < 1e-12);
    }

    #[test]
    fn test_reml_optimizer_deterministic_fixture_outputs() {
        let optimizer = MultiTermGCVOptimizer::new_from_cached(
            DMatrix::from_diagonal(&DVector::from_vec(vec![4.0, 5.0, 7.0])),
            DVector::from_vec(vec![9.0, 3.0, -2.0]),
            25.0,
            vec![
                Array2::from_shape_vec((1, 1), vec![2.0]).expect("shape"),
                Array2::from_shape_vec((1, 1), vec![3.0]).expect("shape"),
            ],
            vec![(1, 2), (2, 3)],
            6,
            1,
        );

        let coordinate = optimizer.optimize_lambdas(&[0.2, 8.0], -4.0, 4.0, 1e-5, 4);
        let reml = optimizer.optimize_lambdas_reml(&[0.2, 8.0], -4.0, 4.0, 1e-5, 4);
        let edfs = optimizer.compute_edfs(&[0.5, 2.0]);
        assert!((coordinate[0] - 1.7195923460542553).abs() < 1e-12);
        assert!((coordinate[1] - 54.59533366687368).abs() < 1e-10);
        assert!((reml[0] - 8.936684744519129).abs() < 1e-12);
        assert!((reml[1] - 10.76852497541976).abs() < 1e-12);
        assert!((edfs[0] - 0.8333333333333335).abs() < 1e-12);
        assert!((edfs[1] - 0.5384615384615385).abs() < 1e-12);
        assert!((optimizer.evaluate_gcv(&[0.5, 2.0]) - 1.16240682241007).abs() < 1e-12);
        assert!(
            (optimizer.evaluate_reml_internal(&[0.5, 2.0], &[1.0, 1.0]) - 8.685310880117177).abs()
                < 1e-12
        );
    }

    #[test]
    fn test_reml_optimizer_coupled_terms_fixture_outputs() {
        let optimizer = MultiTermGCVOptimizer::new_from_cached(
            DMatrix::from_row_slice(3, 3, &[4.0, 1.0, 0.5, 1.0, 5.0, 2.0, 0.5, 2.0, 7.0]),
            DVector::from_vec(vec![9.0, 3.0, -2.0]),
            25.0,
            vec![
                Array2::from_shape_vec((1, 1), vec![2.0]).expect("shape"),
                Array2::from_shape_vec((1, 1), vec![3.0]).expect("shape"),
            ],
            vec![(1, 2), (2, 3)],
            6,
            1,
        );

        let reml = optimizer.optimize_lambdas_reml(&[0.2, 8.0], -4.0, 4.0, 1e-5, 4);
        assert!((reml[0] - 54.598150033144236).abs() < 1e-10);
        assert!((reml[1] - 7.225896126585857).abs() < 1e-12);
    }

    #[test]
    fn test_multi_term_optimizer_zero_outer_iterations_preserves_warm_start() {
        let (x, z, w, penalty, n_param, col_start, col_end) = simple_smooth_problem(50, 6);
        let optimizer = MultiTermGCVOptimizer::new(
            &x,
            &z,
            &w,
            vec![penalty],
            vec![(col_start, col_end)],
            n_param,
        );

        let lambdas = optimizer.optimize_lambdas(&[2.5], -8.0, 8.0, 1e-4, 0);
        assert_eq!(lambdas, vec![2.5]);
    }

    #[test]
    fn test_multi_term_reml_and_singular_degradation_paths() {
        let (x, z, w, penalty, n_param, col_start, col_end) = simple_smooth_problem(80, 7);
        let optimizer = MultiTermGCVOptimizer::new(
            &x,
            &z,
            &w,
            vec![penalty],
            vec![(col_start, col_end)],
            n_param,
        );

        let lambdas = optimizer.optimize_lambdas_reml(&[1.0], -6.0, 6.0, 1e-4, 5);
        assert_eq!(lambdas.len(), 1);
        assert!(lambdas[0].is_finite());
        assert!(lambdas[0] > 0.0);

        let gradient_converged = optimizer.optimize_lambdas_reml(&[1.25], -6.0, 6.0, 1.0e12, 5);
        assert_eq!(gradient_converged.len(), 1);
        assert!((gradient_converged[0] - 1.25).abs() < 1e-12);
        assert!(optimizer.evaluate_reml_internal(&[1.0], &[1.0]).is_finite());

        let singular = MultiTermGCVOptimizer::new_from_cached(
            DMatrix::zeros(2, 2),
            DVector::zeros(2),
            1.0,
            vec![Array2::zeros((1, 1))],
            vec![(1, 2)],
            4,
            1,
        );
        assert!(singular.evaluate_gcv(&[1.0]).is_infinite());
        assert!(singular
            .evaluate_reml_internal(&[1.0], &[0.0])
            .is_infinite());
        assert_eq!(singular.compute_edfs(&[1.0]), vec![0.0]);
        let singular_lambdas = singular.optimize_lambdas_reml(&[3.0], -6.0, 6.0, 1e-4, 5);
        assert_eq!(singular_lambdas.len(), 1);
        assert!((singular_lambdas[0] - 3.0).abs() < 1e-12);
    }
}
