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

use ndarray::{Array1, Array2, ArrayView2, s};

use crate::error::{RustyStatsError, Result};
use crate::families::Family;
use crate::links::Link;
use crate::regularization::{Penalty, SmoothPenalty};
use crate::splines::penalized::{gcv_score, compute_edf, penalty_matrix};
use crate::solvers::irls::{IRLSConfig, FitConfig, fit_glm_unified, solve_weighted_least_squares_with_penalty_matrix, solve_wls_from_precomputed, compute_xtwx, compute_xtwx_xtwz};
// MU constants no longer needed here - clamp_mu delegates to Family::clamp_mu
use crate::convert;

/// Embed a scaled penalty sub-matrix into a larger penalty matrix.
/// `target[offset..offset+k, offset..offset+k] += scale * source`
fn embed_penalty(target: &mut Array2<f64>, source: &Array2<f64>, offset: usize, scale: f64) {
    let k = source.nrows();
    let mut slice = target.slice_mut(s![offset..offset+k, offset..offset+k]);
    slice.scaled_add(scale, source);
}

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
    
    /// IRLS weights from the final iteration (for robust SEs)
    pub irls_weights: Array1<f64>,
    
    /// Prior weights
    pub prior_weights: Array1<f64>,
    
    /// Combined design matrix [parametric | smooth]
    pub design_matrix: Array2<f64>,
    
    /// Original response variable (for residuals/diagnostics)
    pub y: Array1<f64>,
    
    /// Offset values (if any)
    pub offset: Option<Array1<f64>>,
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


/// Compute final EDFs, GCV, and assemble the SmoothGLMResult from SmoothTermSpec data.
fn assemble_smooth_result_from_specs(
    coefficients: Array1<f64>,
    mu: Array1<f64>,
    eta: Array1<f64>,
    deviance: f64,
    iterations: usize,
    converged: bool,
    final_weights: &Array1<f64>,
    x_combined: ArrayView2<'_, f64>,
    penalty_specs: &[(&Array2<f64>, usize, usize)],
    lambdas: &[f64],
    p_param: usize,
    family_name: &str,
    prior_weights: Array1<f64>,
    y: &Array1<f64>,
    offset: Option<&Array1<f64>>,
    cov_unscaled: Option<Array2<f64>>,
) -> SmoothGLMResult {
    let n = y.len();
    
    // Compute EDFs
    let xtwx = compute_xtwx(x_combined, final_weights);
    let mut smooth_edfs = Vec::with_capacity(penalty_specs.len());
    
    for (i, &(penalty, start, _end)) in penalty_specs.iter().enumerate() {
        let lambda = lambdas[i];
        let k = penalty.nrows();
        let xtwx_block = xtwx.slice(s![start..start+k, start..start+k]).to_owned();
        let edf = compute_edf(&xtwx_block, penalty, lambda);
        smooth_edfs.push(edf);
    }
    
    let total_edf = (p_param as f64) + smooth_edfs.iter().sum::<f64>();
    let gcv = gcv_score(deviance, n, total_edf);
    
    // Build SmoothPenalty for result
    let mut smooth_penalty = SmoothPenalty::new();
    for (i, &(penalty, start, end)) in penalty_specs.iter().enumerate() {
        smooth_penalty.add_term(penalty.clone(), lambdas[i], start..end);
    }
    
    // Use provided covariance or compute from X'WX + S
    let cov = cov_unscaled.unwrap_or_else(|| {
        let total_cols = x_combined.ncols();
        let mut penalty_matrix = Array2::zeros((total_cols, total_cols));
        for (i, &(penalty, start, _end)) in penalty_specs.iter().enumerate() {
            embed_penalty(&mut penalty_matrix, penalty, start, lambdas[i]);
        }
        let xtwx_pen = &xtwx + &penalty_matrix;
        invert_matrix(&xtwx_pen).unwrap_or_else(|| Array2::eye(total_cols))
    });
    
    SmoothGLMResult {
        coefficients,
        fitted_values: mu,
        linear_predictor: eta,
        deviance,
        iterations,
        converged,
        lambdas: lambdas.to_vec(),
        smooth_edfs,
        total_edf,
        gcv,
        covariance_unscaled: cov,
        family_name: family_name.to_string(),
        penalty: Penalty::Smooth(smooth_penalty),
        irls_weights: final_weights.clone(),
        prior_weights,
        design_matrix: x_combined.to_owned(),
        y: y.clone(),
        offset: offset.cloned(),
    }
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

use super::gcv_optimizer::MultiTermGCVOptimizer;
use super::nnls::{nnls_weighted_penalized, NNLSConfig};
use nalgebra::{DMatrix, DVector};

/// Solve constrained weighted least squares for monotonic smooth terms.
/// 
/// For each smooth term:
/// - If monotonic: use NNLS to enforce non-negative coefficients
/// - If unconstrained: use standard WLS
fn solve_constrained_wls(
    x: ArrayView2<'_, f64>,
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
            embed_penalty(&mut penalty_matrix, &term.penalty, term_indices[i].0, lambdas[i]);
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
        embed_penalty(&mut penalty_matrix, &term.penalty, term_indices[i].0, lambdas[i]);
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
    convert::solve_symmetric(a, b).ok_or_else(|| {
        RustyStatsError::LinearAlgebraError("Cannot solve linear system".to_string())
    })
}

/// Simple matrix inversion helper.
fn invert_matrix(a: &Array2<f64>) -> Option<Array2<f64>> {
    convert::invert_matrix(a)
}

// =============================================================================
// Unified entry point: takes full design matrix + smooth specs
// =============================================================================

/// Smooth term specification for the unified entry point.
///
/// Instead of passing separate basis matrices, callers provide the full design
/// matrix and indicate which column ranges are smooth terms via this struct.
#[derive(Debug, Clone)]
pub struct SmoothTermSpec {
    /// Start column index (inclusive) in the full design matrix
    pub col_start: usize,
    /// End column index (exclusive) in the full design matrix
    pub col_end: usize,
    /// Penalty matrix (k × k) for this smooth term
    pub penalty: Array2<f64>,
    /// Monotonicity constraint
    pub monotonicity: Monotonicity,
    /// Initial lambda value
    pub initial_lambda: f64,
}

/// Fit GLM with smooth terms from a full design matrix.
///
/// This is the unified entry point that eliminates the need for Python to split
/// the design matrix into parametric + smooth parts. The full design matrix is
/// passed with column ranges identifying smooth terms. Coefficients are returned
/// in the same column order as the input matrix — no reordering needed.
///
/// Handles both unconstrained and monotonic smooth terms in a single call.
pub fn fit_smooth_glm_full_matrix(
    y: &Array1<f64>,
    x_full: ArrayView2<'_, f64>,
    smooth_specs: &[SmoothTermSpec],
    family: &dyn Family,
    link: &dyn Link,
    config: &SmoothGLMConfig,
    offset: Option<&Array1<f64>>,
    weights: Option<&Array1<f64>>,
) -> Result<SmoothGLMResult> {
    let n = y.len();
    let p = x_full.ncols();
    
    if x_full.nrows() != n {
        return Err(RustyStatsError::DimensionMismatch(format!(
            "x_full has {} rows but y has {} elements", x_full.nrows(), n
        )));
    }
    
    if smooth_specs.is_empty() {
        // No smooth terms — delegate to standard GLM fit and wrap result
        let unified_config = FitConfig::from(&config.irls_config);
        let irls = fit_glm_unified(y, x_full, family, link, &unified_config, offset, weights, None)?;
        return Ok(SmoothGLMResult {
            coefficients: irls.coefficients,
            fitted_values: irls.fitted_values,
            linear_predictor: irls.linear_predictor,
            deviance: irls.deviance,
            iterations: irls.iterations,
            converged: irls.converged,
            lambdas: vec![],
            smooth_edfs: vec![],
            total_edf: p as f64,
            gcv: 0.0,
            covariance_unscaled: irls.covariance_unscaled,
            family_name: irls.family_name,
            penalty: irls.penalty,
            irls_weights: irls.irls_weights,
            prior_weights: irls.prior_weights,
            design_matrix: irls.design_matrix.unwrap_or_else(|| x_full.to_owned()),
            y: irls.y,
            offset: if irls.offset.iter().all(|&v| v == 0.0) { None } else { Some(irls.offset) },
        });
    }
    
    // Validate specs
    for (i, spec) in smooth_specs.iter().enumerate() {
        if spec.col_end > p || spec.col_start >= spec.col_end {
            return Err(RustyStatsError::InvalidValue(format!(
                "Smooth spec {} has invalid column range [{}, {}), matrix has {} columns",
                i, spec.col_start, spec.col_end, p
            )));
        }
        let k = spec.col_end - spec.col_start;
        if spec.penalty.nrows() != k || spec.penalty.ncols() != k {
            return Err(RustyStatsError::DimensionMismatch(format!(
                "Smooth spec {} penalty has shape ({}, {}) but expected ({}, {})",
                i, spec.penalty.nrows(), spec.penalty.ncols(), k, k
            )));
        }
    }
    
    // Determine parametric column count (everything NOT in a smooth term)
    let mut smooth_cols = std::collections::HashSet::new();
    for spec in smooth_specs {
        for c in spec.col_start..spec.col_end {
            smooth_cols.insert(c);
        }
    }
    let p_param = p - smooth_cols.len();
    
    // Build column ranges in x_full order (smooth specs are already indexed into x_full)
    let term_indices: Vec<(usize, usize)> = smooth_specs.iter()
        .map(|s| (s.col_start, s.col_end))
        .collect();
    
    let offset_vec = offset.cloned().unwrap_or_else(|| Array1::zeros(n));
    let prior_weights = weights.cloned().unwrap_or_else(|| Array1::ones(n));
    let mut lambdas: Vec<f64> = smooth_specs.iter().map(|s| s.initial_lambda).collect();
    
    let has_monotonic = smooth_specs.iter().any(|s| s.is_monotonic());
    
    // Use x_full directly as x_combined — no reassembly needed
    let x_combined = x_full;
    let total_cols = p;
    
    // Initialize μ
    let mut mu = family.initialize_mu(y);
    let mut eta = link.link(&mu);
    let mut deviance = family.deviance(y, &mu, Some(&prior_weights));
    
    let mut converged = false;
    let mut iteration = 0;
    let mut coefficients = Array1::zeros(total_cols);
    let mut cov_unscaled = Array2::zeros((total_cols, total_cols));
    let mut final_weights = Array1::ones(n);
    
    let log_lambda_min = config.lambda_min.ln();
    let log_lambda_max = config.lambda_max.ln();
    let mut penalty_matrix = Array2::zeros((total_cols, total_cols));
    let mut lambdas_stable_count = 0u32;  // Track consecutive iterations with stable lambdas
    
    while iteration < config.irls_config.max_iterations {
        iteration += 1;
        let deviance_old = deviance;
        
        // IRLS weights
        let link_deriv = link.derivative(&mu);
        let variance = family.variance(&mu);
        
        let irls_weights: Array1<f64> = (0..n)
            .map(|i| {
                let d = link_deriv[i];
                let v = variance[i];
                (1.0 / (v * d * d)).max(config.irls_config.min_weight).min(1e10)
            })
            .collect();
        
        let combined_weights = &prior_weights * &irls_weights;
        
        // Working response: (eta - offset) + (y - mu) * link_deriv
        let working_response = (&eta - &offset_vec) + &((y - &mu) * &link_deriv);
        
        // Compute X'WX and X'Wz ONCE per iteration — shared by GCV and WLS
        let (cached_xtwx, cached_xtwz) = compute_xtwx_xtwz(x_combined, &working_response, &combined_weights)?;
        
        // Compute z'Wz scalar for GCV RSS computation (O(n), trivial)
        let ztwz: f64 = working_response.iter().zip(combined_weights.iter())
            .map(|(&zi, &wi)| wi * zi * zi)
            .sum();
        
        // Optimize lambdas using cached matrices (no X'WX recomputation)
        // Skip GCV once lambdas have stabilized for 2 consecutive iterations
        let run_gcv = lambdas_stable_count < 2 && (iteration <= 3 || iteration % 2 == 0);
        if run_gcv {
            let old_lambdas = lambdas.clone();
            
            let penalties: Vec<Array2<f64>> = smooth_specs.iter()
                .map(|s| s.penalty.clone())
                .collect();
            
            let optimizer = MultiTermGCVOptimizer::new_from_cached(
                cached_xtwx.clone(),
                cached_xtwz.clone(),
                ztwz,
                penalties,
                term_indices.clone(),
                n,
                p_param,
            );
            
            lambdas = optimizer.optimize_lambdas(
                log_lambda_min,
                log_lambda_max,
                config.lambda_tol,
                3,
            );
            
            // Check if lambdas have stabilized (max relative change < 1%)
            let max_rel_change = old_lambdas.iter().zip(lambdas.iter())
                .map(|(&old, &new)| {
                    if old.abs() < 1e-12 { (new - old).abs() }
                    else { (new - old).abs() / old.abs() }
                })
                .fold(0.0f64, f64::max);
            
            if max_rel_change < 0.01 {
                lambdas_stable_count += 1;
            } else {
                lambdas_stable_count = 0;
            }
        }
        
        // Build penalty matrix (reuse allocation, zero and refill)
        penalty_matrix.fill(0.0);
        for (i, spec) in smooth_specs.iter().enumerate() {
            embed_penalty(&mut penalty_matrix, &spec.penalty, spec.col_start, lambdas[i]);
        }
        
        // Solve WLS from pre-computed X'WX — no redundant O(n·p²) computation
        let new_coef;
        if has_monotonic {
            // Build SmoothTermData for the constrained solver
            let term_data: Vec<SmoothTermData> = smooth_specs.iter()
                .map(|s| SmoothTermData {
                    name: String::new(),
                    basis: x_combined.slice(ndarray::s![.., s.col_start..s.col_end]).to_owned(),
                    penalty: s.penalty.clone(),
                    monotonicity: s.monotonicity,
                    initial_lambda: s.initial_lambda,
                })
                .collect();
            let nnls_config = super::nnls::NNLSConfig::default();
            new_coef = solve_constrained_wls(
                x_combined,
                &working_response,
                &combined_weights,
                &term_data,
                &term_indices,
                &lambdas,
                p_param,
                &nnls_config,
            )?;
        } else {
            let (coef, xtwinv) = solve_wls_from_precomputed(
                &cached_xtwx,
                &cached_xtwz,
                &penalty_matrix,
            )?;
            new_coef = coef;
            cov_unscaled = xtwinv;
        }
        
        // Update eta and mu with new coefficients
        let eta_new = &x_combined.dot(&new_coef) + &offset_vec;
        let mu_new = family.clamp_mu(&link.inverse(&eta_new));
        let deviance_new = family.deviance(y, &mu_new, Some(&prior_weights));
        
        // Step halving if deviance increased (blend old coefficients ↔ new)
        if deviance_new > deviance_old * 1.0001 && iteration > 1 {
            let mut step = 0.5;
            let mut best_coef = new_coef.clone();
            let mut best_dev = deviance_new;
            
            for _ in 0..5 {
                let blended = &coefficients * (1.0 - step) + &new_coef * step;
                
                let eta_full = &x_combined.dot(&blended) + &offset_vec;
                let mu_blend = family.clamp_mu(&link.inverse(&eta_full));
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
        eta = &x_combined.dot(&coefficients) + &offset_vec;
        mu = family.clamp_mu(&link.inverse(&eta));
        deviance = family.deviance(y, &mu, Some(&prior_weights));
        final_weights = irls_weights;
        
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
    
    // Assemble result directly from SmoothTermSpec (no SmoothTermData conversion)
    let penalty_specs: Vec<(&Array2<f64>, usize, usize)> = smooth_specs.iter()
        .map(|s| (&s.penalty, s.col_start, s.col_end))
        .collect();
    
    Ok(assemble_smooth_result_from_specs(
        coefficients, mu, eta, deviance, iteration, converged,
        &final_weights, x_combined, &penalty_specs, &lambdas,
        p_param, family.name(), prior_weights, y, offset,
        if has_monotonic { None } else { Some(cov_unscaled) },
    ))
}

impl SmoothTermSpec {
    /// Whether this term has a monotonicity constraint.
    pub fn is_monotonic(&self) -> bool {
        !matches!(self.monotonicity, Monotonicity::None)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::families::{GaussianFamily, PoissonFamily, GammaFamily};
    use crate::links::{IdentityLink, LogLink};
    use crate::splines::bs_basis;

    // =========================================================================
    // Unit tests for structs and helpers
    // =========================================================================

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
    fn test_embed_penalty() {
        let penalty1 = Array2::eye(5);
        let penalty2 = Array2::eye(3);
        
        let mut penalty = Array2::zeros((10, 10));
        embed_penalty(&mut penalty, &penalty1, 2, 0.5);
        embed_penalty(&mut penalty, &penalty2, 7, 2.0);
        
        // Check shape
        assert_eq!(penalty.shape(), &[10, 10]);
        
        // Check that parametric columns have no penalty
        assert_eq!(penalty[[0, 0]], 0.0);
        assert_eq!(penalty[[1, 1]], 0.0);
        
        // Check that smooth columns have scaled penalty
        assert_eq!(penalty[[2, 2]], 0.5);  // lambda1 * I
        assert_eq!(penalty[[7, 7]], 2.0);  // lambda2 * I
    }

    #[test]
    fn test_smooth_term_with_monotonicity() {
        let x = Array1::from_vec((0..50).map(|i| i as f64 / 5.0).collect());
        let basis = bs_basis(&x, 8, 3, None, false);

        let term = SmoothTermData::new("age".to_string(), basis)
            .with_monotonicity(Monotonicity::Increasing)
            .with_lambda(2.5);

        assert!(term.is_monotonic());
        assert_eq!(term.monotonicity, Monotonicity::Increasing);
        assert_eq!(term.initial_lambda, 2.5);
    }

    // =========================================================================
    // Integration tests: fit_smooth_glm_full_matrix (unified entry point)
    // =========================================================================

    /// Helper: generate Gaussian data with a smooth sin(x) effect.
    fn gaussian_smooth_data(n: usize) -> (Array1<f64>, Array2<f64>, Array1<f64>) {
        let x_vals: Array1<f64> = (0..n).map(|i| i as f64 * 10.0 / n as f64).collect();
        let y: Array1<f64> = x_vals.iter().map(|&xi| 2.0 + xi.sin() + 0.1 * (xi * 7.3).sin()).collect();
        // Parametric part: intercept column
        let x_param = Array2::from_shape_fn((n, 1), |(_, _)| 1.0);
        (y, x_param, x_vals)
    }

    /// Helper: generate Poisson data with a smooth effect.
    fn poisson_smooth_data(n: usize) -> (Array1<f64>, Array2<f64>, Array1<f64>) {
        let x_vals: Array1<f64> = (0..n).map(|i| i as f64 * 10.0 / n as f64).collect();
        let y: Array1<f64> = x_vals.iter().map(|&xi| {
            let mu = (0.5 + 0.3 * xi.sin()).exp();
            // Deterministic "Poisson-like" values (round to nearest int, min 0)
            (mu + 0.5).floor().max(0.0)
        }).collect();
        let x_param = Array2::from_shape_fn((n, 1), |(_, _)| 1.0);
        (y, x_param, x_vals)
    }

    /// Helper: concatenate parametric + basis into full design matrix and build SmoothTermSpec.
    fn make_full_matrix(x_param: &Array2<f64>, basis: &Array2<f64>) -> (Array2<f64>, Vec<SmoothTermSpec>) {
        let p_param = x_param.ncols();
        let k = basis.ncols();
        let x_full = ndarray::concatenate![ndarray::Axis(1), *x_param, *basis]
            .as_standard_layout().to_owned();
        let spec = SmoothTermSpec {
            col_start: p_param,
            col_end: p_param + k,
            penalty: crate::splines::penalized::penalty_matrix(k, 2),
            monotonicity: Monotonicity::None,
            initial_lambda: 1.0,
        };
        (x_full, vec![spec])
    }

    #[test]
    fn test_fit_smooth_glm_gaussian_converges() {
        let (y, x_param, x_vals) = gaussian_smooth_data(100);
        let basis = bs_basis(&x_vals, 10, 3, None, false);
        let (x_full, specs) = make_full_matrix(&x_param, &basis);
        let config = SmoothGLMConfig::default();

        let result = fit_smooth_glm_full_matrix(
            &y, x_full.view(), &specs,
            &GaussianFamily, &IdentityLink,
            &config, None, None,
        ).unwrap();

        assert!(result.converged, "Gaussian smooth GLM should converge");
        assert!(result.deviance > 0.0);
        assert!(result.iterations > 0);
    }

    #[test]
    fn test_fit_smooth_glm_gaussian_edf_less_than_k() {
        let (y, x_param, x_vals) = gaussian_smooth_data(200);
        let basis = bs_basis(&x_vals, 10, 3, None, false);
        let k = basis.ncols();
        let (x_full, specs) = make_full_matrix(&x_param, &basis);
        let config = SmoothGLMConfig::default();

        let result = fit_smooth_glm_full_matrix(
            &y, x_full.view(), &specs,
            &GaussianFamily, &IdentityLink,
            &config, None, None,
        ).unwrap();

        assert_eq!(result.smooth_edfs.len(), 1);
        assert!(result.smooth_edfs[0] > 1.0, "EDF should be > 1 for non-trivial smooth");
        assert!(result.smooth_edfs[0] < k as f64, "EDF {} should be < k {}", result.smooth_edfs[0], k);
        assert!(result.total_edf > 1.0);
    }

    #[test]
    fn test_fit_smooth_glm_gaussian_gcv_positive() {
        let (y, x_param, x_vals) = gaussian_smooth_data(100);
        let basis = bs_basis(&x_vals, 10, 3, None, false);
        let (x_full, specs) = make_full_matrix(&x_param, &basis);
        let config = SmoothGLMConfig::default();

        let result = fit_smooth_glm_full_matrix(
            &y, x_full.view(), &specs,
            &GaussianFamily, &IdentityLink,
            &config, None, None,
        ).unwrap();

        assert!(result.gcv > 0.0, "GCV should be positive");
        assert_eq!(result.lambdas.len(), 1);
        assert!(result.lambdas[0] > 0.0, "Selected lambda should be positive");
    }

    #[test]
    fn test_fit_smooth_glm_poisson_converges() {
        let (y, x_param, x_vals) = poisson_smooth_data(200);
        let basis = bs_basis(&x_vals, 8, 3, None, false);
        let (x_full, specs) = make_full_matrix(&x_param, &basis);
        let config = SmoothGLMConfig::default();

        let result = fit_smooth_glm_full_matrix(
            &y, x_full.view(), &specs,
            &PoissonFamily, &LogLink,
            &config, None, None,
        ).unwrap();

        assert!(result.converged, "Poisson smooth GLM should converge");
        assert!(result.fitted_values.iter().all(|&v| v > 0.0),
            "Poisson fitted values must be positive");
    }

    #[test]
    fn test_fit_smooth_glm_covariance_shape() {
        let (y, x_param, x_vals) = gaussian_smooth_data(100);
        let basis = bs_basis(&x_vals, 8, 3, None, false);
        let p_total = x_param.ncols() + basis.ncols();
        let (x_full, specs) = make_full_matrix(&x_param, &basis);
        let config = SmoothGLMConfig::default();

        let result = fit_smooth_glm_full_matrix(
            &y, x_full.view(), &specs,
            &GaussianFamily, &IdentityLink,
            &config, None, None,
        ).unwrap();

        assert_eq!(result.covariance_unscaled.shape(), &[p_total, p_total]);
        assert_eq!(result.coefficients.len(), p_total);
    }

    #[test]
    fn test_fit_smooth_glm_with_offset() {
        let n = 100;
        let (y, x_param, x_vals) = poisson_smooth_data(n);
        let basis = bs_basis(&x_vals, 8, 3, None, false);
        let (x_full, specs) = make_full_matrix(&x_param, &basis);
        let offset = Array1::from_vec(vec![0.5; n]);
        let mut config = SmoothGLMConfig::default();
        config.irls_config.max_iterations = 50;

        let result = fit_smooth_glm_full_matrix(
            &y, x_full.view(), &specs,
            &PoissonFamily, &LogLink,
            &config, Some(&offset), None,
        ).unwrap();

        assert!(result.converged);
    }

    #[test]
    fn test_fit_smooth_glm_with_weights() {
        let n = 100;
        let (y, x_param, x_vals) = gaussian_smooth_data(n);
        let basis = bs_basis(&x_vals, 8, 3, None, false);
        let (x_full, specs) = make_full_matrix(&x_param, &basis);
        let weights = Array1::from_vec(vec![2.0; n]);
        let config = SmoothGLMConfig::default();

        let result = fit_smooth_glm_full_matrix(
            &y, x_full.view(), &specs,
            &GaussianFamily, &IdentityLink,
            &config, None, Some(&weights),
        ).unwrap();

        assert!(result.converged);
    }

    #[test]
    fn test_fit_smooth_glm_dimension_mismatch() {
        let n = 100;
        let (y, x_param, _x_vals) = gaussian_smooth_data(n);
        let bad_x = Array1::from_vec((0..50).map(|i| i as f64).collect());
        let basis = bs_basis(&bad_x, 8, 3, None, false);
        // Build full matrix from mismatched basis (50 rows) and x_param (100 rows)
        // fit_smooth_glm_full_matrix validates x_full.nrows() == y.len()
        let k = basis.ncols();
        let x_full = ndarray::concatenate![ndarray::Axis(1), x_param.slice(ndarray::s![0..50, ..]), basis]
            .as_standard_layout().to_owned();
        let specs = vec![SmoothTermSpec {
            col_start: 1, col_end: 1 + k,
            penalty: crate::splines::penalized::penalty_matrix(k, 2),
            monotonicity: Monotonicity::None,
            initial_lambda: 1.0,
        }];
        let config = SmoothGLMConfig::default();

        let result = fit_smooth_glm_full_matrix(
            &y, x_full.view(), &specs,
            &GaussianFamily, &IdentityLink,
            &config, None, None,
        );

        assert!(result.is_err());
    }

    // =========================================================================
    // Multi-term tests
    // =========================================================================

    #[test]
    fn test_fit_smooth_glm_two_terms() {
        let n = 200;
        let x1: Array1<f64> = (0..n).map(|i| i as f64 * 10.0 / n as f64).collect();
        let x2: Array1<f64> = (0..n).map(|i| i as f64 * 5.0 / n as f64).collect();
        let y: Array1<f64> = x1.iter().zip(x2.iter())
            .map(|(&a, &b)| 2.0 + a.sin() + 0.5 * b.cos())
            .collect();

        let x_param = Array2::from_shape_fn((n, 1), |(_, _)| 1.0);
        let basis1 = bs_basis(&x1, 8, 3, None, false);
        let basis2 = bs_basis(&x2, 6, 3, None, false);
        let k1 = basis1.ncols();
        let k2 = basis2.ncols();
        let x_full = ndarray::concatenate![ndarray::Axis(1), x_param, basis1, basis2]
            .as_standard_layout().to_owned();
        let specs = vec![
            SmoothTermSpec {
                col_start: 1, col_end: 1 + k1,
                penalty: crate::splines::penalized::penalty_matrix(k1, 2),
                monotonicity: Monotonicity::None,
                initial_lambda: 1.0,
            },
            SmoothTermSpec {
                col_start: 1 + k1, col_end: 1 + k1 + k2,
                penalty: crate::splines::penalized::penalty_matrix(k2, 2),
                monotonicity: Monotonicity::None,
                initial_lambda: 1.0,
            },
        ];
        let config = SmoothGLMConfig::default();

        let result = fit_smooth_glm_full_matrix(
            &y, x_full.view(), &specs,
            &GaussianFamily, &IdentityLink,
            &config, None, None,
        ).unwrap();

        assert!(result.converged);
        assert_eq!(result.lambdas.len(), 2);
        assert_eq!(result.smooth_edfs.len(), 2);
        assert!(result.lambdas.iter().all(|&l| l > 0.0));
        assert!(result.smooth_edfs.iter().all(|&e| e > 1.0));
    }

    // =========================================================================
    // Result fields are populated correctly
    // =========================================================================

    #[test]
    fn test_smooth_result_fields_populated() {
        let n = 100;
        let (y, x_param, x_vals) = gaussian_smooth_data(n);
        let basis = bs_basis(&x_vals, 8, 3, None, false);
        let (x_full, specs) = make_full_matrix(&x_param, &basis);
        let config = SmoothGLMConfig::default();

        let result = fit_smooth_glm_full_matrix(
            &y, x_full.view(), &specs,
            &GaussianFamily, &IdentityLink,
            &config, None, None,
        ).unwrap();

        assert_eq!(result.fitted_values.len(), n);
        assert_eq!(result.linear_predictor.len(), n);
        assert_eq!(result.irls_weights.len(), n);
        assert_eq!(result.prior_weights.len(), n);
        assert_eq!(result.design_matrix.nrows(), n);
        assert_eq!(result.y.len(), n);
        assert!(result.family_name.contains("Gaussian") || result.family_name.contains("gaussian"));
    }

    // =========================================================================
    // Empty specs fallback (standard GLM through unified entry point)
    // =========================================================================

    #[test]
    fn test_fit_smooth_glm_no_smooth_terms() {
        let (y, x_param, _x_vals) = gaussian_smooth_data(100);
        let config = SmoothGLMConfig::default();

        let result = fit_smooth_glm_full_matrix(
            &y, x_param.view(), &[],
            &GaussianFamily, &IdentityLink,
            &config, None, None,
        ).unwrap();

        assert!(result.converged);
        assert!(result.lambdas.is_empty());
        assert!(result.smooth_edfs.is_empty());
        assert_eq!(result.coefficients.len(), 1); // intercept only
    }
}
