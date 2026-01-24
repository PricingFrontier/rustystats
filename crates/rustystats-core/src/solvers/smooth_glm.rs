// =============================================================================
// SMOOTH GLM: Generalized Additive Models with Penalized Splines
// =============================================================================
//
// This module implements GLM fitting with penalized smooth terms (P-splines).
// It extends standard IRLS to automatically select smoothing parameters via GCV.
//
// THE ALGORITHM
// -------------
// For a GAM with smooth terms s(x1), s(x2), ..., we:
//
// 1. Build design matrix X = [parametric | smooth basis columns]
// 2. Build penalty matrix S = block-diag(0, λ₁S₁, λ₂S₂, ...)
// 3. Run penalized IRLS: (X'WX + S)⁻¹ X'Wz at each iteration
// 4. Select λ by minimizing GCV(λ) = n × Deviance / (n - EDF)²
//
// LAMBDA SELECTION STRATEGIES
// ---------------------------
// - Grid search: Evaluate GCV on log-spaced grid
// - Performance iteration: Iterate between IRLS and lambda updates
// - REML: More stable but more complex (future work)
//
// =============================================================================

use ndarray::{Array1, Array2};

use crate::error::{RustyStatsError, Result};
use crate::families::Family;
use crate::links::Link;
use crate::regularization::{Penalty, SmoothPenalty};
use crate::splines::penalized::{gcv_score, lambda_grid, compute_edf, penalty_matrix};
use crate::solvers::irls::{IRLSConfig, solve_weighted_least_squares_with_penalty_matrix, compute_xtwx};
use crate::constants::{MU_MIN_POSITIVE, MU_MIN_PROBABILITY, MU_MAX_PROBABILITY};

/// Result from fitting a smooth GLM (GAM).
#[derive(Debug, Clone)]
pub struct SmoothGLMResult {
    /// Fitted coefficients (parametric + smooth basis)
    pub coefficients: Array1<f64>,
    
    /// Fitted values μ = g⁻¹(Xβ + offset)
    pub fitted_values: Array1<f64>,
    
    /// Linear predictor η = Xβ + offset
    pub linear_predictor: Array1<f64>,
    
    /// Final deviance
    pub deviance: f64,
    
    /// Number of IRLS iterations
    pub iterations: usize,
    
    /// Did the algorithm converge?
    pub converged: bool,
    
    /// Selected smoothing parameters (one per smooth term)
    pub lambdas: Vec<f64>,
    
    /// Effective degrees of freedom (one per smooth term)
    pub smooth_edfs: Vec<f64>,
    
    /// Total effective degrees of freedom (parametric + smooth)
    pub total_edf: f64,
    
    /// GCV score at selected lambdas
    pub gcv: f64,
    
    /// Unscaled covariance matrix (X'WX + S)⁻¹
    pub covariance_unscaled: Array2<f64>,
    
    /// Family name
    pub family_name: String,
    
    /// The smooth penalty configuration
    pub penalty: Penalty,
}

/// Configuration for smooth GLM fitting.
#[derive(Debug, Clone)]
pub struct SmoothGLMConfig {
    /// Base IRLS configuration
    pub irls_config: IRLSConfig,
    
    /// Number of lambda values to evaluate in grid search
    pub n_lambda: usize,
    
    /// Minimum lambda value (log scale)
    pub lambda_min: f64,
    
    /// Maximum lambda value (log scale)
    pub lambda_max: f64,
    
    /// Convergence tolerance for lambda optimization
    pub lambda_tol: f64,
    
    /// Maximum iterations for lambda optimization (outer loop)
    pub max_lambda_iter: usize,
    
    /// Method for lambda selection: "gcv" or "fixed"
    pub lambda_method: String,
}

impl Default for SmoothGLMConfig {
    fn default() -> Self {
        Self {
            irls_config: IRLSConfig::default(),
            n_lambda: 30,
            lambda_min: 1e-4,
            lambda_max: 1e6,
            lambda_tol: 1e-4,
            max_lambda_iter: 20,
            lambda_method: "gcv".to_string(),
        }
    }
}

/// Monotonicity constraint for smooth terms.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Monotonicity {
    /// No constraint
    None,
    /// Monotonically increasing (coefficients >= 0 with I-spline basis)
    Increasing,
    /// Monotonically decreasing (coefficients <= 0 with I-spline basis)
    Decreasing,
}

impl Default for Monotonicity {
    fn default() -> Self {
        Monotonicity::None
    }
}

/// Data for a single smooth term.
#[derive(Debug, Clone)]
pub struct SmoothTermData {
    /// Variable name
    pub name: String,
    /// Basis matrix for this term (n × k)
    pub basis: Array2<f64>,
    /// Penalty matrix S = D'D (k × k)
    pub penalty: Array2<f64>,
    /// Initial lambda (will be optimized if lambda_method = "gcv")
    pub initial_lambda: f64,
    /// Monotonicity constraint
    pub monotonicity: Monotonicity,
}

impl SmoothTermData {
    /// Create a new smooth term from a basis matrix.
    /// Automatically computes the second-order difference penalty.
    pub fn new(name: String, basis: Array2<f64>) -> Self {
        let k = basis.ncols();
        let penalty = penalty_matrix(k, 2);  // Second-order difference penalty
        Self {
            name,
            basis,
            penalty,
            initial_lambda: 1.0,
            monotonicity: Monotonicity::None,
        }
    }
    
    /// Create with a custom initial lambda.
    pub fn with_lambda(mut self, lambda: f64) -> Self {
        self.initial_lambda = lambda;
        self
    }
    
    /// Set monotonicity constraint.
    pub fn with_monotonicity(mut self, mono: Monotonicity) -> Self {
        self.monotonicity = mono;
        self
    }
    
    /// Check if this term has a monotonicity constraint.
    pub fn is_monotonic(&self) -> bool {
        self.monotonicity != Monotonicity::None
    }
    
    /// Number of basis functions.
    pub fn k(&self) -> usize {
        self.basis.ncols()
    }
}

/// Fit a GLM with smooth terms using penalized IRLS.
///
/// This is the main entry point for GAM fitting with automatic smoothness selection.
///
/// # Arguments
/// * `y` - Response variable (n × 1)
/// * `x_parametric` - Parametric part of design matrix (n × p), including intercept
/// * `smooth_terms` - Smooth term data (basis + penalty for each)
/// * `family` - Distribution family
/// * `link` - Link function
/// * `config` - Fitting configuration
/// * `offset` - Optional offset term
/// * `weights` - Optional prior weights
///
/// # Returns
/// * `Ok(SmoothGLMResult)` - Fitted model with selected lambdas and EDFs
/// * `Err(RustyStatsError)` - If fitting fails
pub fn fit_smooth_glm(
    y: &Array1<f64>,
    x_parametric: &Array2<f64>,
    smooth_terms: &[SmoothTermData],
    family: &dyn Family,
    link: &dyn Link,
    config: &SmoothGLMConfig,
    offset: Option<&Array1<f64>>,
    weights: Option<&Array1<f64>>,
) -> Result<SmoothGLMResult> {
    let n = y.len();
    let p_param = x_parametric.ncols();
    
    // Validate inputs
    if x_parametric.nrows() != n {
        return Err(RustyStatsError::DimensionMismatch(format!(
            "x_parametric has {} rows but y has {} elements", x_parametric.nrows(), n
        )));
    }
    
    for (i, term) in smooth_terms.iter().enumerate() {
        if term.basis.nrows() != n {
            return Err(RustyStatsError::DimensionMismatch(format!(
                "Smooth term {} has {} rows but y has {} elements", i, term.basis.nrows(), n
            )));
        }
    }
    
    // Build combined design matrix: [parametric | smooth1 | smooth2 | ...]
    let total_smooth_cols: usize = smooth_terms.iter().map(|t| t.k()).sum();
    let total_cols = p_param + total_smooth_cols;
    
    let mut x_combined = Array2::zeros((n, total_cols));
    
    // Copy parametric columns
    for i in 0..n {
        for j in 0..p_param {
            x_combined[[i, j]] = x_parametric[[i, j]];
        }
    }
    
    // Copy smooth basis columns
    let mut col_offset = p_param;
    let mut term_indices = Vec::with_capacity(smooth_terms.len());
    
    for term in smooth_terms {
        let start = col_offset;
        let end = col_offset + term.k();
        term_indices.push(start..end);
        
        for i in 0..n {
            for j in 0..term.k() {
                x_combined[[i, col_offset + j]] = term.basis[[i, j]];
            }
        }
        col_offset = end;
    }
    
    // Set up offset and weights
    let offset_vec = match offset {
        Some(o) => o.clone(),
        None => Array1::zeros(n),
    };
    
    let prior_weights = match weights {
        Some(w) => w.clone(),
        None => Array1::ones(n),
    };
    
    // Initialize lambdas
    let mut lambdas: Vec<f64> = smooth_terms.iter().map(|t| t.initial_lambda).collect();
    
    // Select lambdas via GCV if requested
    if config.lambda_method == "gcv" && !smooth_terms.is_empty() {
        lambdas = select_lambdas_gcv(
            y,
            &x_combined,
            smooth_terms,
            &term_indices,
            p_param,
            family,
            link,
            &config.irls_config,
            &offset_vec,
            &prior_weights,
            config,
        )?;
    }
    
    // Build final penalty matrix with selected lambdas
    let penalty_matrix = build_penalty_matrix(
        total_cols,
        smooth_terms,
        &term_indices,
        &lambdas,
    );
    
    // Fit model with selected lambdas
    let (coefficients, fitted_values, linear_predictor, deviance, iterations, converged, cov_unscaled, final_weights) = 
        fit_with_penalty(
            y,
            &x_combined,
            &penalty_matrix,
            family,
            link,
            &config.irls_config,
            &offset_vec,
            &prior_weights,
        )?;
    
    // Compute EDFs and GCV
    let xtwx = compute_xtwx(&x_combined, &final_weights);
    let smooth_edfs = compute_smooth_edfs(&xtwx, smooth_terms, &term_indices, &lambdas);
    let total_edf = (p_param as f64) + smooth_edfs.iter().sum::<f64>();
    let gcv = gcv_score(deviance, n, total_edf);
    
    // Build SmoothPenalty for result
    let mut smooth_penalty = SmoothPenalty::new();
    for (i, term) in smooth_terms.iter().enumerate() {
        smooth_penalty.add_term(term.penalty.clone(), lambdas[i], term_indices[i].clone());
    }
    
    Ok(SmoothGLMResult {
        coefficients,
        fitted_values,
        linear_predictor,
        deviance,
        iterations,
        converged,
        lambdas,
        smooth_edfs,
        total_edf,
        gcv,
        covariance_unscaled: cov_unscaled,
        family_name: family.name().to_string(),
        penalty: Penalty::Smooth(smooth_penalty),
    })
}

/// Select lambdas via GCV grid search.
fn select_lambdas_gcv(
    y: &Array1<f64>,
    x: &Array2<f64>,
    smooth_terms: &[SmoothTermData],
    term_indices: &[std::ops::Range<usize>],
    p_param: usize,
    family: &dyn Family,
    link: &dyn Link,
    irls_config: &IRLSConfig,
    offset: &Array1<f64>,
    weights: &Array1<f64>,
    config: &SmoothGLMConfig,
) -> Result<Vec<f64>> {
    let n = y.len();
    let n_terms = smooth_terms.len();
    let total_cols = x.ncols();
    
    if n_terms == 0 {
        return Ok(vec![]);
    }
    
    // Generate lambda grid
    let grid = lambda_grid(config.n_lambda, config.lambda_min, config.lambda_max);
    
    // For single smooth term, do simple grid search
    if n_terms == 1 {
        let mut best_lambda = smooth_terms[0].initial_lambda;
        let mut best_gcv = f64::INFINITY;
        
        for &lambda in &grid {
            let penalty_mat = build_penalty_matrix(
                total_cols,
                smooth_terms,
                term_indices,
                &[lambda],
            );
            
            match fit_with_penalty(y, x, &penalty_mat, family, link, irls_config, offset, weights) {
                Ok((_, _, _, deviance, _, _, _, final_weights)) => {
                    let xtwx = compute_xtwx(x, &final_weights);
                    let edfs = compute_smooth_edfs(&xtwx, smooth_terms, term_indices, &[lambda]);
                    let total_edf = (p_param as f64) + edfs.iter().sum::<f64>();
                    let gcv = gcv_score(deviance, n, total_edf);
                    
                    if gcv < best_gcv {
                        best_gcv = gcv;
                        best_lambda = lambda;
                    }
                }
                Err(_) => continue,  // Skip failed fits
            }
        }
        
        return Ok(vec![best_lambda]);
    }
    
    // For multiple smooth terms, use coordinate-wise optimization
    let mut lambdas: Vec<f64> = smooth_terms.iter().map(|t| t.initial_lambda).collect();
    
    for _outer_iter in 0..config.max_lambda_iter {
        let old_lambdas = lambdas.clone();
        
        // Optimize each lambda while holding others fixed
        for term_idx in 0..n_terms {
            let mut best_lambda = lambdas[term_idx];
            let mut best_gcv = f64::INFINITY;
            
            for &lambda in &grid {
                let mut test_lambdas = lambdas.clone();
                test_lambdas[term_idx] = lambda;
                
                let penalty_mat = build_penalty_matrix(
                    total_cols,
                    smooth_terms,
                    term_indices,
                    &test_lambdas,
                );
                
                match fit_with_penalty(y, x, &penalty_mat, family, link, irls_config, offset, weights) {
                    Ok((_, _, _, deviance, _, _, _, final_weights)) => {
                        let xtwx = compute_xtwx(x, &final_weights);
                        let edfs = compute_smooth_edfs(&xtwx, smooth_terms, term_indices, &test_lambdas);
                        let total_edf = (p_param as f64) + edfs.iter().sum::<f64>();
                        let gcv = gcv_score(deviance, n, total_edf);
                        
                        if gcv < best_gcv {
                            best_gcv = gcv;
                            best_lambda = lambda;
                        }
                    }
                    Err(_) => continue,
                }
            }
            
            lambdas[term_idx] = best_lambda;
        }
        
        // Check convergence
        let max_rel_change: f64 = lambdas.iter()
            .zip(old_lambdas.iter())
            .map(|(&new, &old)| ((new - old) / old.max(1e-10)).abs())
            .fold(0.0, f64::max);
        
        if max_rel_change < config.lambda_tol {
            break;
        }
    }
    
    Ok(lambdas)
}

/// Build combined penalty matrix from smooth terms.
fn build_penalty_matrix(
    total_cols: usize,
    smooth_terms: &[SmoothTermData],
    term_indices: &[std::ops::Range<usize>],
    lambdas: &[f64],
) -> Array2<f64> {
    let mut penalty = Array2::zeros((total_cols, total_cols));
    
    for (i, term) in smooth_terms.iter().enumerate() {
        let range = &term_indices[i];
        let lambda = lambdas[i];
        
        for r in 0..term.penalty.nrows() {
            for c in 0..term.penalty.ncols() {
                penalty[[range.start + r, range.start + c]] = lambda * term.penalty[[r, c]];
            }
        }
    }
    
    penalty
}

/// Compute EDF for each smooth term.
fn compute_smooth_edfs(
    xtwx: &Array2<f64>,
    smooth_terms: &[SmoothTermData],
    term_indices: &[std::ops::Range<usize>],
    lambdas: &[f64],
) -> Vec<f64> {
    let mut edfs = Vec::with_capacity(smooth_terms.len());
    
    for (i, term) in smooth_terms.iter().enumerate() {
        let range = &term_indices[i];
        let lambda = lambdas[i];
        
        // Extract the subblock of X'WX for this term
        let k = term.k();
        let mut xtwx_block = Array2::zeros((k, k));
        for r in 0..k {
            for c in 0..k {
                xtwx_block[[r, c]] = xtwx[[range.start + r, range.start + c]];
            }
        }
        
        // Compute EDF for this term
        let edf = compute_edf(&xtwx_block, &term.penalty, lambda);
        edfs.push(edf);
    }
    
    edfs
}

/// Fit model with a fixed penalty matrix.
/// Returns: (coefficients, fitted_values, linear_predictor, deviance, iterations, converged, cov_unscaled, final_weights)
fn fit_with_penalty(
    y: &Array1<f64>,
    x: &Array2<f64>,
    penalty_matrix: &Array2<f64>,
    family: &dyn Family,
    link: &dyn Link,
    config: &IRLSConfig,
    offset: &Array1<f64>,
    prior_weights: &Array1<f64>,
) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>, f64, usize, bool, Array2<f64>, Array1<f64>)> {
    let n = y.len();
    let p = x.ncols();
    
    // Initialize μ
    let mut mu = family.initialize_mu(y);
    let mut eta = link.link(&mu);
    let mut deviance = family.deviance(y, &mu, Some(prior_weights));
    
    let mut converged = false;
    let mut iteration = 0;
    let mut cov_unscaled = Array2::zeros((p, p));
    let mut final_weights = Array1::ones(n);
    let mut coefficients = Array1::zeros(p);
    
    while iteration < config.max_iterations {
        iteration += 1;
        let deviance_old = deviance;
        
        // Compute IRLS weights
        let link_deriv = link.derivative(&mu);
        let variance = family.variance(&mu);
        
        let irls_weights: Array1<f64> = (0..n)
            .map(|i| {
                let d = link_deriv[i];
                let v = variance[i];
                (1.0 / (v * d * d)).max(config.min_weight).min(1e10)
            })
            .collect();
        
        let combined_weights: Array1<f64> = prior_weights
            .iter()
            .zip(irls_weights.iter())
            .map(|(&pw, &iw)| pw * iw)
            .collect();
        
        // Working response
        let working_response: Array1<f64> = (0..n)
            .map(|i| {
                let e = eta[i] - offset[i];
                e + (y[i] - mu[i]) * link_deriv[i]
            })
            .collect();
        
        // Solve penalized WLS
        let (new_coef, xtwinv) = solve_weighted_least_squares_with_penalty_matrix(
            x,
            &working_response,
            &combined_weights,
            penalty_matrix,
        )?;
        
        // Update eta and mu
        let eta_base = x.dot(&new_coef);
        let eta_new: Array1<f64> = eta_base.iter().zip(offset.iter()).map(|(&e, &o)| e + o).collect();
        let mu_new = clamp_mu(&link.inverse(&eta_new), family);
        let deviance_new = family.deviance(y, &mu_new, Some(prior_weights));
        
        // Step halving if deviance increased
        if deviance_new > deviance_old * 1.0001 && iteration > 1 {
            let mut step = 0.5;
            let mut best_coef = new_coef.clone();
            let mut best_dev = deviance_new;
            
            for _ in 0..5 {
                let blended: Array1<f64> = coefficients.iter()
                    .zip(new_coef.iter())
                    .map(|(&old, &new)| (1.0 - step) * old + step * new)
                    .collect();
                
                let eta_blend = x.dot(&blended);
                let eta_full: Array1<f64> = eta_blend.iter().zip(offset.iter()).map(|(&e, &o)| e + o).collect();
                let mu_blend = clamp_mu(&link.inverse(&eta_full), family);
                let dev_blend = family.deviance(y, &mu_blend, Some(prior_weights));
                
                if dev_blend < best_dev {
                    best_dev = dev_blend;
                    best_coef = blended;
                }
                step *= 0.5;
            }
            
            coefficients = best_coef;
        } else {
            coefficients = new_coef;
        }
        
        // Update state
        let eta_base = x.dot(&coefficients);
        eta = eta_base.iter().zip(offset.iter()).map(|(&e, &o)| e + o).collect();
        mu = clamp_mu(&link.inverse(&eta), family);
        deviance = family.deviance(y, &mu, Some(prior_weights));
        cov_unscaled = xtwinv;
        final_weights = irls_weights;
        
        // Check convergence
        let rel_change = if deviance_old.abs() > 1e-10 {
            (deviance_old - deviance).abs() / deviance_old.abs()
        } else {
            (deviance_old - deviance).abs()
        };
        
        if rel_change < config.tolerance {
            converged = true;
            break;
        }
    }
    
    Ok((coefficients, mu, eta, deviance, iteration, converged, cov_unscaled, final_weights))
}

/// Clamp μ to valid range for the family.
fn clamp_mu(mu: &Array1<f64>, family: &dyn Family) -> Array1<f64> {
    let name = family.name();
    mu.mapv(|x| match name {
        "Poisson" | "Gamma" => x.max(MU_MIN_POSITIVE),
        "Binomial" => x.max(MU_MIN_PROBABILITY).min(MU_MAX_PROBABILITY),
        _ => x,
    })
}

// =============================================================================
// FAST SMOOTH GLM FITTING (mgcv-style)
// =============================================================================
//
// This approach optimizes lambda WITHIN a single IRLS fit using Brent's method.
// Instead of doing n_lambda separate fits, we:
// 1. Run IRLS normally
// 2. At each iteration (or every few), optimize lambda using cached X'WX
// 3. Update penalty and continue
//
// This is ~10-20x faster than grid search for large datasets.
// =============================================================================

use super::gcv_optimizer::{GCVCache, MultiTermGCVOptimizer};

/// Fit GLM with smooth terms using fast GCV optimization.
/// 
/// This is the fast version that optimizes lambda within IRLS iterations
/// instead of doing multiple separate fits.
pub fn fit_smooth_glm_fast(
    y: &Array1<f64>,
    x_parametric: &Array2<f64>,
    smooth_terms: &[SmoothTermData],
    family: &dyn Family,
    link: &dyn Link,
    config: &SmoothGLMConfig,
    offset: Option<&Array1<f64>>,
    weights: Option<&Array1<f64>>,
) -> Result<SmoothGLMResult> {
    let n = y.len();
    let p_param = x_parametric.ncols();
    
    // Validate inputs
    if x_parametric.nrows() != n {
        return Err(RustyStatsError::DimensionMismatch(format!(
            "x_parametric has {} rows but y has {} elements", x_parametric.nrows(), n
        )));
    }
    
    if smooth_terms.is_empty() {
        // No smooth terms - use standard IRLS
        return fit_smooth_glm(y, x_parametric, smooth_terms, family, link, config, offset, weights);
    }
    
    // Build combined design matrix
    let total_smooth_cols: usize = smooth_terms.iter().map(|t| t.k()).sum();
    let total_cols = p_param + total_smooth_cols;
    
    let mut x_combined = Array2::zeros((n, total_cols));
    for i in 0..n {
        for j in 0..p_param {
            x_combined[[i, j]] = x_parametric[[i, j]];
        }
    }
    
    let mut col_offset = p_param;
    let mut term_indices: Vec<(usize, usize)> = Vec::with_capacity(smooth_terms.len());
    
    for term in smooth_terms {
        let start = col_offset;
        let end = col_offset + term.k();
        term_indices.push((start, end));
        
        for i in 0..n {
            for j in 0..term.k() {
                x_combined[[i, col_offset + j]] = term.basis[[i, j]];
            }
        }
        col_offset = end;
    }
    
    // Set up offset and weights
    let offset_vec = offset.cloned().unwrap_or_else(|| Array1::zeros(n));
    let prior_weights = weights.cloned().unwrap_or_else(|| Array1::ones(n));
    
    // Initialize lambdas
    let mut lambdas: Vec<f64> = smooth_terms.iter().map(|t| t.initial_lambda).collect();
    
    // Initialize μ
    let mut mu = family.initialize_mu(y);
    let mut eta = link.link(&mu);
    let mut deviance = family.deviance(y, &mu, Some(&prior_weights));
    
    let mut converged = false;
    let mut iteration = 0;
    let mut coefficients = Array1::zeros(total_cols);
    let mut cov_unscaled = Array2::zeros((total_cols, total_cols));
    let mut final_weights = Array1::ones(n);
    
    // Log-scale bounds for lambda search
    let log_lambda_min = config.lambda_min.ln();
    let log_lambda_max = config.lambda_max.ln();
    
    while iteration < config.irls_config.max_iterations {
        iteration += 1;
        let deviance_old = deviance;
        
        // Compute IRLS weights
        let link_deriv = link.derivative(&mu);
        let variance = family.variance(&mu);
        
        let irls_weights: Array1<f64> = (0..n)
            .map(|i| {
                let d = link_deriv[i];
                let v = variance[i];
                (1.0 / (v * d * d)).max(config.irls_config.min_weight).min(1e10)
            })
            .collect();
        
        let combined_weights: Array1<f64> = prior_weights
            .iter()
            .zip(irls_weights.iter())
            .map(|(&pw, &iw)| pw * iw)
            .collect();
        
        // Working response
        let working_response: Array1<f64> = (0..n)
            .map(|i| {
                let e = eta[i] - offset_vec[i];
                e + (y[i] - mu[i]) * link_deriv[i]
            })
            .collect();
        
        // Optimize lambdas using fast GCV (every iteration for first few, then less often)
        if iteration <= 3 || iteration % 2 == 0 {
            let penalties: Vec<Array2<f64>> = smooth_terms.iter()
                .map(|t| t.penalty.clone())
                .collect();
            
            if smooth_terms.len() == 1 {
                // Single term - use simple Brent optimization
                let cache = GCVCache::new(
                    &x_combined,
                    &working_response,
                    &combined_weights,
                    &smooth_terms[0].penalty,
                    term_indices[0].0,
                    term_indices[0].1,
                    p_param,
                );
                
                let (opt_lambda, _, _) = cache.optimize_lambda(
                    log_lambda_min,
                    log_lambda_max,
                    config.lambda_tol,
                );
                lambdas[0] = opt_lambda;
            } else {
                // Multiple terms - use coordinate descent
                let optimizer = MultiTermGCVOptimizer::new(
                    &x_combined,
                    &working_response,
                    &combined_weights,
                    penalties,
                    term_indices.clone(),
                    p_param,
                );
                
                lambdas = optimizer.optimize_lambdas(
                    log_lambda_min,
                    log_lambda_max,
                    config.lambda_tol,
                    3,  // Just a few outer iterations per IRLS step
                );
            }
        }
        
        // Build penalty matrix with current lambdas
        let mut penalty_matrix = Array2::zeros((total_cols, total_cols));
        for (i, term) in smooth_terms.iter().enumerate() {
            let (start, _end) = term_indices[i];
            let lambda = lambdas[i];
            for r in 0..term.penalty.nrows() {
                for c in 0..term.penalty.ncols() {
                    penalty_matrix[[start + r, start + c]] = lambda * term.penalty[[r, c]];
                }
            }
        }
        
        // Solve penalized WLS
        let (new_coef, xtwinv) = solve_weighted_least_squares_with_penalty_matrix(
            &x_combined,
            &working_response,
            &combined_weights,
            &penalty_matrix,
        )?;
        
        // Update eta and mu
        let eta_base = x_combined.dot(&new_coef);
        let eta_new: Array1<f64> = eta_base.iter().zip(offset_vec.iter()).map(|(&e, &o)| e + o).collect();
        let mu_new = clamp_mu(&link.inverse(&eta_new), family);
        let deviance_new = family.deviance(y, &mu_new, Some(&prior_weights));
        
        // Step halving if deviance increased
        if deviance_new > deviance_old * 1.0001 && iteration > 1 {
            let mut step = 0.5;
            let mut best_coef = new_coef.clone();
            let mut best_dev = deviance_new;
            
            for _ in 0..5 {
                let blended: Array1<f64> = coefficients.iter()
                    .zip(new_coef.iter())
                    .map(|(&old, &new)| (1.0 - step) * old + step * new)
                    .collect();
                
                let eta_blend = x_combined.dot(&blended);
                let eta_full: Array1<f64> = eta_blend.iter().zip(offset_vec.iter()).map(|(&e, &o)| e + o).collect();
                let mu_blend = clamp_mu(&link.inverse(&eta_full), family);
                let dev_blend = family.deviance(y, &mu_blend, Some(&prior_weights));
                
                if dev_blend < best_dev {
                    best_dev = dev_blend;
                    best_coef = blended;
                }
                step *= 0.5;
            }
            
            coefficients = best_coef;
        } else {
            coefficients = new_coef;
        }
        
        // Update state
        let eta_base = x_combined.dot(&coefficients);
        eta = eta_base.iter().zip(offset_vec.iter()).map(|(&e, &o)| e + o).collect();
        mu = clamp_mu(&link.inverse(&eta), family);
        deviance = family.deviance(y, &mu, Some(&prior_weights));
        cov_unscaled = xtwinv;
        final_weights = irls_weights;
        
        // Check convergence
        let rel_change = if deviance_old.abs() > 1e-10 {
            (deviance_old - deviance).abs() / deviance_old.abs()
        } else {
            (deviance_old - deviance).abs()
        };
        
        if rel_change < config.irls_config.tolerance {
            converged = true;
            break;
        }
    }
    
    // Compute final EDFs
    let xtwx = compute_xtwx(&x_combined, &final_weights);
    let mut smooth_edfs = Vec::with_capacity(smooth_terms.len());
    
    for (i, term) in smooth_terms.iter().enumerate() {
        let (start, _end) = term_indices[i];
        let lambda = lambdas[i];
        
        // Extract subblock
        let k = term.k();
        let mut xtwx_block = Array2::zeros((k, k));
        for r in 0..k {
            for c in 0..k {
                xtwx_block[[r, c]] = xtwx[[start + r, start + c]];
            }
        }
        
        let edf = compute_edf(&xtwx_block, &term.penalty, lambda);
        smooth_edfs.push(edf);
    }
    
    let total_edf = (p_param as f64) + smooth_edfs.iter().sum::<f64>();
    let gcv = gcv_score(deviance, n, total_edf);
    
    // Build SmoothPenalty for result
    let mut smooth_penalty = SmoothPenalty::new();
    for (i, term) in smooth_terms.iter().enumerate() {
        let (start, end) = term_indices[i];
        smooth_penalty.add_term(term.penalty.clone(), lambdas[i], start..end);
    }
    
    Ok(SmoothGLMResult {
        coefficients,
        fitted_values: mu,
        linear_predictor: eta,
        deviance,
        iterations: iteration,
        converged,
        lambdas,
        smooth_edfs,
        total_edf,
        gcv,
        covariance_unscaled: cov_unscaled,
        family_name: family.name().to_string(),
        penalty: Penalty::Smooth(smooth_penalty),
    })
}

// =============================================================================
// CONSTRAINED SMOOTH GLM FITTING (Monotonic Splines)
// =============================================================================
//
// This extends the fast smooth GLM fitting to support monotonicity constraints.
// For monotonic terms, we use NNLS (Non-Negative Least Squares) to enforce
// that coefficients are non-negative (for I-spline basis).
//
// The approach:
// 1. Use I-spline basis for monotonic terms (provided externally)
// 2. Split coefficients into unconstrained (parametric) and constrained (smooth)
// 3. Use NNLS for the smooth term coefficients
// 4. Combine with penalty for smoothness control
//
// =============================================================================

use super::nnls::{nnls_weighted_penalized, NNLSConfig};
use nalgebra::{DMatrix, DVector};

/// Fit GLM with monotonic smooth terms using NNLS.
/// 
/// This version handles monotonicity constraints by using non-negative least squares
/// for smooth term coefficients. The basis should be I-splines for monotonic terms.
pub fn fit_smooth_glm_monotonic(
    y: &Array1<f64>,
    x_parametric: &Array2<f64>,
    smooth_terms: &[SmoothTermData],
    family: &dyn Family,
    link: &dyn Link,
    config: &SmoothGLMConfig,
    offset: Option<&Array1<f64>>,
    weights: Option<&Array1<f64>>,
) -> Result<SmoothGLMResult> {
    let n = y.len();
    let p_param = x_parametric.ncols();
    
    // Check if any term is monotonic
    let has_monotonic = smooth_terms.iter().any(|t| t.is_monotonic());
    
    if !has_monotonic {
        // No monotonic terms - use standard fast fitting
        return fit_smooth_glm_fast(y, x_parametric, smooth_terms, family, link, config, offset, weights);
    }
    
    // Validate inputs
    if x_parametric.nrows() != n {
        return Err(RustyStatsError::DimensionMismatch(format!(
            "x_parametric has {} rows but y has {} elements", x_parametric.nrows(), n
        )));
    }
    
    // Build combined design matrix
    let total_smooth_cols: usize = smooth_terms.iter().map(|t| t.k()).sum();
    let total_cols = p_param + total_smooth_cols;
    
    let mut x_combined = Array2::zeros((n, total_cols));
    for i in 0..n {
        for j in 0..p_param {
            x_combined[[i, j]] = x_parametric[[i, j]];
        }
    }
    
    let mut col_offset = p_param;
    let mut term_indices: Vec<(usize, usize)> = Vec::with_capacity(smooth_terms.len());
    
    for term in smooth_terms {
        let start = col_offset;
        let end = col_offset + term.k();
        term_indices.push((start, end));
        
        for i in 0..n {
            for j in 0..term.k() {
                x_combined[[i, col_offset + j]] = term.basis[[i, j]];
            }
        }
        col_offset = end;
    }
    
    // Set up offset and weights
    let offset_vec = offset.cloned().unwrap_or_else(|| Array1::zeros(n));
    let prior_weights = weights.cloned().unwrap_or_else(|| Array1::ones(n));
    
    // Initialize lambdas
    let mut lambdas: Vec<f64> = smooth_terms.iter().map(|t| t.initial_lambda).collect();
    
    // Initialize μ
    let mut mu = family.initialize_mu(y);
    let mut eta = link.link(&mu);
    let mut deviance = family.deviance(y, &mu, Some(&prior_weights));
    
    let mut converged = false;
    let mut iteration = 0;
    let mut coefficients = Array1::zeros(total_cols);
    let mut cov_unscaled = Array2::zeros((total_cols, total_cols));
    let mut final_weights = Array1::ones(n);
    
    // Log-scale bounds for lambda search
    let log_lambda_min = config.lambda_min.ln();
    let log_lambda_max = config.lambda_max.ln();
    
    // NNLS config
    let nnls_config = NNLSConfig::default();
    
    while iteration < config.irls_config.max_iterations {
        iteration += 1;
        let deviance_old = deviance;
        
        // Compute IRLS weights
        let link_deriv = link.derivative(&mu);
        let variance = family.variance(&mu);
        
        let irls_weights: Array1<f64> = (0..n)
            .map(|i| {
                let d = link_deriv[i];
                let v = variance[i];
                (1.0 / (v * d * d)).max(config.irls_config.min_weight).min(1e10)
            })
            .collect();
        
        let combined_weights: Array1<f64> = prior_weights
            .iter()
            .zip(irls_weights.iter())
            .map(|(&pw, &iw)| pw * iw)
            .collect();
        
        // Working response
        let working_response: Array1<f64> = (0..n)
            .map(|i| {
                let e = eta[i] - offset_vec[i];
                e + (y[i] - mu[i]) * link_deriv[i]
            })
            .collect();
        
        // Optimize lambdas (same as unconstrained version)
        if iteration <= 3 || iteration % 2 == 0 {
            if smooth_terms.len() == 1 {
                let cache = GCVCache::new(
                    &x_combined,
                    &working_response,
                    &combined_weights,
                    &smooth_terms[0].penalty,
                    term_indices[0].0,
                    term_indices[0].1,
                    p_param,
                );
                
                let (opt_lambda, _, _) = cache.optimize_lambda(
                    log_lambda_min,
                    log_lambda_max,
                    config.lambda_tol,
                );
                lambdas[0] = opt_lambda;
            } else {
                let penalties: Vec<Array2<f64>> = smooth_terms.iter()
                    .map(|t| t.penalty.clone())
                    .collect();
                
                let optimizer = MultiTermGCVOptimizer::new(
                    &x_combined,
                    &working_response,
                    &combined_weights,
                    penalties,
                    term_indices.clone(),
                    p_param,
                );
                
                lambdas = optimizer.optimize_lambdas(
                    log_lambda_min,
                    log_lambda_max,
                    config.lambda_tol,
                    3,
                );
            }
        }
        
        // Solve using constrained approach for monotonic terms
        let new_coef = solve_constrained_wls(
            &x_combined,
            &working_response,
            &combined_weights,
            smooth_terms,
            &term_indices,
            &lambdas,
            p_param,
            &nnls_config,
        )?;
        
        // Update eta and mu
        let eta_base = x_combined.dot(&new_coef);
        let eta_new: Array1<f64> = eta_base.iter().zip(offset_vec.iter()).map(|(&e, &o)| e + o).collect();
        let mu_new = clamp_mu(&link.inverse(&eta_new), family);
        let deviance_new = family.deviance(y, &mu_new, Some(&prior_weights));
        
        // Step halving if deviance increased
        if deviance_new > deviance_old * 1.0001 && iteration > 1 {
            let mut step = 0.5;
            let mut best_coef = new_coef.clone();
            let mut best_dev = deviance_new;
            
            for _ in 0..5 {
                let blended: Array1<f64> = coefficients.iter()
                    .zip(new_coef.iter())
                    .map(|(&old, &new)| (1.0 - step) * old + step * new)
                    .collect();
                
                let eta_blend = x_combined.dot(&blended);
                let eta_full: Array1<f64> = eta_blend.iter().zip(offset_vec.iter()).map(|(&e, &o)| e + o).collect();
                let mu_blend = clamp_mu(&link.inverse(&eta_full), family);
                let dev_blend = family.deviance(y, &mu_blend, Some(&prior_weights));
                
                if dev_blend < best_dev {
                    best_dev = dev_blend;
                    best_coef = blended;
                }
                step *= 0.5;
            }
            
            coefficients = best_coef;
        } else {
            coefficients = new_coef;
        }
        
        // Update state
        let eta_base = x_combined.dot(&coefficients);
        eta = eta_base.iter().zip(offset_vec.iter()).map(|(&e, &o)| e + o).collect();
        mu = clamp_mu(&link.inverse(&eta), family);
        deviance = family.deviance(y, &mu, Some(&prior_weights));
        final_weights = irls_weights;
        
        // Check convergence
        let rel_change = if deviance_old.abs() > 1e-10 {
            (deviance_old - deviance).abs() / deviance_old.abs()
        } else {
            (deviance_old - deviance).abs()
        };
        
        if rel_change < config.irls_config.tolerance {
            converged = true;
            break;
        }
    }
    
    // Compute final EDFs
    let xtwx = compute_xtwx(&x_combined, &final_weights);
    let mut smooth_edfs = Vec::with_capacity(smooth_terms.len());
    
    for (i, term) in smooth_terms.iter().enumerate() {
        let (start, _end) = term_indices[i];
        let lambda = lambdas[i];
        
        let k = term.k();
        let mut xtwx_block = Array2::zeros((k, k));
        for r in 0..k {
            for c in 0..k {
                xtwx_block[[r, c]] = xtwx[[start + r, start + c]];
            }
        }
        
        let edf = compute_edf(&xtwx_block, &term.penalty, lambda);
        smooth_edfs.push(edf);
    }
    
    let total_edf = (p_param as f64) + smooth_edfs.iter().sum::<f64>();
    let gcv = gcv_score(deviance, n, total_edf);
    
    // Build SmoothPenalty for result
    let mut smooth_penalty = SmoothPenalty::new();
    for (i, term) in smooth_terms.iter().enumerate() {
        let (start, end) = term_indices[i];
        smooth_penalty.add_term(term.penalty.clone(), lambdas[i], start..end);
    }
    
    // Compute approximate covariance (note: this is approximate for constrained problems)
    // For now, use the unconstrained covariance as an approximation
    let mut penalty_matrix = Array2::zeros((total_cols, total_cols));
    for (i, term) in smooth_terms.iter().enumerate() {
        let (start, _end) = term_indices[i];
        let lambda = lambdas[i];
        for r in 0..term.penalty.nrows() {
            for c in 0..term.penalty.ncols() {
                penalty_matrix[[start + r, start + c]] = lambda * term.penalty[[r, c]];
            }
        }
    }
    
    let xtwx_pen = &xtwx + &penalty_matrix;
    cov_unscaled = invert_matrix(&xtwx_pen).unwrap_or_else(|| Array2::eye(total_cols));
    
    Ok(SmoothGLMResult {
        coefficients,
        fitted_values: mu,
        linear_predictor: eta,
        deviance,
        iterations: iteration,
        converged,
        lambdas,
        smooth_edfs,
        total_edf,
        gcv,
        covariance_unscaled: cov_unscaled,
        family_name: family.name().to_string(),
        penalty: Penalty::Smooth(smooth_penalty),
    })
}

/// Solve constrained weighted least squares for monotonic smooth terms.
/// 
/// For each smooth term:
/// - If monotonic: use NNLS to enforce non-negative coefficients
/// - If unconstrained: use standard WLS
fn solve_constrained_wls(
    x: &Array2<f64>,
    z: &Array1<f64>,
    w: &Array1<f64>,
    smooth_terms: &[SmoothTermData],
    term_indices: &[(usize, usize)],
    lambdas: &[f64],
    p_param: usize,
    nnls_config: &NNLSConfig,
) -> Result<Array1<f64>> {
    let n = x.nrows();
    let p = x.ncols();
    
    // For simplicity, we solve each monotonic term separately using NNLS,
    // then combine. This is a block coordinate descent approach.
    
    // First, check if ALL smooth terms are monotonic
    let all_monotonic = smooth_terms.iter().all(|t| t.is_monotonic());
    let any_monotonic = smooth_terms.iter().any(|t| t.is_monotonic());
    
    if !any_monotonic {
        // No monotonic terms - use standard WLS
        let mut penalty_matrix = Array2::zeros((p, p));
        for (i, term) in smooth_terms.iter().enumerate() {
            let (start, _end) = term_indices[i];
            let lambda = lambdas[i];
            for r in 0..term.penalty.nrows() {
                for c in 0..term.penalty.ncols() {
                    penalty_matrix[[start + r, start + c]] = lambda * term.penalty[[r, c]];
                }
            }
        }
        let (coef, _) = solve_weighted_least_squares_with_penalty_matrix(x, z, w, &penalty_matrix)?;
        return Ok(coef);
    }
    
    // For monotonic terms, we use a hybrid approach:
    // 1. Solve for parametric coefficients using standard WLS with smooth terms fixed
    // 2. Solve for smooth coefficients using NNLS with parametric fixed
    // Iterate until convergence (usually 1-2 iterations)
    
    let mut coefficients = Array1::zeros(p);
    
    // Simple approach for single monotonic term (most common case)
    if smooth_terms.len() == 1 && all_monotonic {
        let term = &smooth_terms[0];
        let (start, end) = term_indices[0];
        let lambda = lambdas[0];
        let k = term.k();
        
        // Extract parametric part
        let x_param = x.slice(ndarray::s![.., 0..p_param]).to_owned();
        let x_smooth = x.slice(ndarray::s![.., start..end]).to_owned();
        
        // Solve jointly using augmented system with NNLS for smooth part
        // For now, use iterative approach: fix parametric, solve smooth; fix smooth, solve parametric
        
        let sqrt_w: Array1<f64> = w.iter().map(|&wi| wi.sqrt()).collect();
        
        // Apply weights
        let mut x_param_w = x_param.clone();
        let mut x_smooth_w = x_smooth.clone();
        let mut z_w = z.clone();
        
        for i in 0..n {
            let sw = sqrt_w[i];
            for j in 0..p_param {
                x_param_w[[i, j]] *= sw;
            }
            for j in 0..k {
                x_smooth_w[[i, j]] *= sw;
            }
            z_w[i] *= sw;
        }
        
        // Iterate between parametric and smooth (2 iterations is usually enough)
        let mut coef_param = Array1::zeros(p_param);
        let mut coef_smooth = Array1::zeros(k);
        
        // Pre-compute penalty matrix in nalgebra format once
        let penalty_contig = if term.penalty.is_standard_layout() {
            term.penalty.clone()
        } else {
            term.penalty.as_standard_layout().to_owned()
        };
        let penalty_nalg = DMatrix::from_row_slice(k, k, penalty_contig.as_slice().unwrap());
        
        // Pre-compute weighted smooth basis once
        let x_smooth_contig = if x_smooth_w.is_standard_layout() { 
            x_smooth_w.clone() 
        } else { 
            x_smooth_w.as_standard_layout().to_owned() 
        };
        let x_smooth_nalg = DMatrix::from_row_slice(n, k, x_smooth_contig.as_slice().unwrap());
        let w_ones = DVector::from_element(n, 1.0);  // Already weighted
        
        for _iter in 0..2 {
            // Fix smooth, solve for parametric
            let residual_param: Array1<f64> = z_w.iter()
                .zip(x_smooth_w.rows())
                .map(|(&zi, row)| {
                    let smooth_contrib: f64 = row.iter().zip(coef_smooth.iter()).map(|(&x, &c)| x * c).sum();
                    zi - smooth_contrib
                })
                .collect();
            
            // Standard least squares for parametric
            let xtx_param = x_param_w.t().dot(&x_param_w);
            let xtz_param = x_param_w.t().dot(&residual_param);
            coef_param = solve_symmetric(&xtx_param, &xtz_param)?;
            
            // Fix parametric, solve for smooth with NNLS
            let residual_smooth: Array1<f64> = z_w.iter()
                .zip(x_param_w.rows())
                .map(|(&zi, row)| {
                    let param_contrib: f64 = row.iter().zip(coef_param.iter()).map(|(&x, &c)| x * c).sum();
                    zi - param_contrib
                })
                .collect();
            
            // Convert residual to nalgebra (matrices already pre-computed above)
            let z_nalg = DVector::from_row_slice(residual_smooth.as_slice().unwrap());
            
            // Solve with NNLS (or negative NNLS for decreasing)
            let nnls_result = match term.monotonicity {
                Monotonicity::Increasing => {
                    nnls_weighted_penalized(&x_smooth_nalg, &z_nalg, &w_ones, &penalty_nalg, lambda, nnls_config)
                },
                Monotonicity::Decreasing => {
                    // For decreasing, negate the basis and result
                    let x_neg = -&x_smooth_nalg;
                    let result = nnls_weighted_penalized(&x_neg, &z_nalg, &w_ones, &penalty_nalg, lambda, nnls_config);
                    super::nnls::NNLSResult {
                        x: -result.x,
                        residual_norm: result.residual_norm,
                        iterations: result.iterations,
                        converged: result.converged,
                    }
                },
                Monotonicity::None => unreachable!(),
            };
            
            for j in 0..k {
                coef_smooth[j] = nnls_result.x[j];
            }
        }
        
        // Combine coefficients
        for j in 0..p_param {
            coefficients[j] = coef_param[j];
        }
        for j in 0..k {
            coefficients[start + j] = coef_smooth[j];
        }
        
        return Ok(coefficients);
    }
    
    // For multiple terms or mixed monotonic/unconstrained, use coordinate descent
    // Initialize with unconstrained solution
    let mut penalty_matrix = Array2::zeros((p, p));
    for (i, term) in smooth_terms.iter().enumerate() {
        let (start, _end) = term_indices[i];
        let lambda = lambdas[i];
        for r in 0..term.penalty.nrows() {
            for c in 0..term.penalty.ncols() {
                penalty_matrix[[start + r, start + c]] = lambda * term.penalty[[r, c]];
            }
        }
    }
    let (init_coef, _) = solve_weighted_least_squares_with_penalty_matrix(x, z, w, &penalty_matrix)?;
    coefficients = init_coef;
    
    // Project monotonic term coefficients to satisfy constraints
    for (i, term) in smooth_terms.iter().enumerate() {
        if term.is_monotonic() {
            let (start, end) = term_indices[i];
            for j in start..end {
                match term.monotonicity {
                    Monotonicity::Increasing => {
                        if coefficients[j] < 0.0 {
                            coefficients[j] = 0.0;
                        }
                    },
                    Monotonicity::Decreasing => {
                        if coefficients[j] > 0.0 {
                            coefficients[j] = 0.0;
                        }
                    },
                    Monotonicity::None => {},
                }
            }
        }
    }
    
    Ok(coefficients)
}

/// Simple symmetric system solver for small systems.
fn solve_symmetric(a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>> {
    let n = a.nrows();
    let a_contig = if a.is_standard_layout() { a.clone() } else { a.as_standard_layout().to_owned() };
    let a_nalg = DMatrix::from_row_slice(n, n, a_contig.as_slice().unwrap());
    let b_nalg = DVector::from_row_slice(b.as_slice().unwrap());
    
    if let Some(chol) = a_nalg.clone().cholesky() {
        let x = chol.solve(&b_nalg);
        Ok(Array1::from_vec(x.as_slice().to_vec()))
    } else {
        // Fall back to LU
        let lu = a_nalg.lu();
        let x = lu.solve(&b_nalg).ok_or_else(|| {
            RustyStatsError::LinearAlgebraError("Cannot solve linear system".to_string())
        })?;
        Ok(Array1::from_vec(x.as_slice().to_vec()))
    }
}

/// Simple matrix inversion helper.
fn invert_matrix(a: &Array2<f64>) -> Option<Array2<f64>> {
    let n = a.nrows();
    let a_contig = if a.is_standard_layout() { a.clone() } else { a.as_standard_layout().to_owned() };
    let a_nalg = DMatrix::from_row_slice(n, n, a_contig.as_slice().unwrap());
    
    a_nalg.clone().try_inverse().map(|inv| {
        let mut result = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                result[[i, j]] = inv[(i, j)];
            }
        }
        result
    })
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::families::GaussianFamily;
    use crate::links::IdentityLink;
    use crate::splines::bs_basis;
    
    #[test]
    fn test_smooth_term_creation() {
        let x = Array1::from_vec((0..100).map(|i| i as f64 / 10.0).collect());
        let basis = bs_basis(&x, 10, 3, None, false);
        
        let term = SmoothTermData::new("age".to_string(), basis.clone());
        
        assert_eq!(term.name, "age");
        assert_eq!(term.k(), 9);  // df=10, no intercept = 9 columns
        assert_eq!(term.penalty.shape(), &[9, 9]);
    }
    
    #[test]
    fn test_build_penalty_matrix() {
        let penalty1 = Array2::eye(5);
        let penalty2 = Array2::eye(3);
        
        let terms = vec![
            SmoothTermData {
                name: "x1".to_string(),
                basis: Array2::zeros((10, 5)),
                penalty: penalty1,
                initial_lambda: 1.0,
                monotonicity: Monotonicity::None,
            },
            SmoothTermData {
                name: "x2".to_string(),
                basis: Array2::zeros((10, 3)),
                penalty: penalty2,
                initial_lambda: 1.0,
                monotonicity: Monotonicity::None,
            },
        ];
        
        let term_indices = vec![2..7, 7..10];  // After 2 parametric columns
        let lambdas = vec![0.5, 2.0];
        
        let penalty = build_penalty_matrix(10, &terms, &term_indices, &lambdas);
        
        // Check shape
        assert_eq!(penalty.shape(), &[10, 10]);
        
        // Check that parametric columns have no penalty
        assert_eq!(penalty[[0, 0]], 0.0);
        assert_eq!(penalty[[1, 1]], 0.0);
        
        // Check that smooth columns have scaled penalty
        assert_eq!(penalty[[2, 2]], 0.5);  // lambda1 * I
        assert_eq!(penalty[[7, 7]], 2.0);  // lambda2 * I
    }
}
