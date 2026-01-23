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
        }
    }
    
    /// Create with a custom initial lambda.
    pub fn with_lambda(mut self, lambda: f64) -> Self {
        self.initial_lambda = lambda;
        self
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
            },
            SmoothTermData {
                name: "x2".to_string(),
                basis: Array2::zeros((10, 3)),
                penalty: penalty2,
                initial_lambda: 1.0,
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
