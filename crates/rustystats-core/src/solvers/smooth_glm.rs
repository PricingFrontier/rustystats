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

use ndarray::{s, Array1, Array2, ArrayView2};

use crate::constants::{MAX_IRLS_WEIGHT, SMOOTH_CONVERGENCE_TOL, ZERO_TOL};
use crate::convert;
use crate::error::{Result, RustyStatsError};
use crate::families::Family;
use crate::links::Link;
use crate::regularization::{Penalty, SmoothPenalty};
use crate::solvers::irls::{
    compute_xtwx, compute_xtwx_xtwz, fit_glm_unified,
    solve_weighted_least_squares_with_penalty_matrix, solve_wls_from_precomputed, FitConfig,
    IRLSConfig,
};
use crate::splines::penalized::{compute_edf, gcv_score, penalty_matrix};

/// Maximum absolute value for alpha parameters in exp reparameterization.
/// Prevents exp() overflow: exp(20) ~ 4.9e8 which is large but finite.
const MAX_EXP_ALPHA: f64 = 20.0;

/// Embed a scaled penalty sub-matrix into a larger penalty matrix.
/// `target[offset..offset+k, offset..offset+k] += scale * source`
fn embed_penalty(target: &mut Array2<f64>, source: &Array2<f64>, offset: usize, scale: f64) {
    let k = source.nrows();
    let mut slice = target.slice_mut(s![offset..offset + k, offset..offset + k]);
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
///
/// Enforced via exp reparameterization (Pya & Wood 2015, scam/mgcv approach):
/// - `beta_1 = alpha_1` (free intercept)
/// - `beta_j = beta_{j-1} + exp(alpha_j)` for `Increasing`
/// - `beta_j = beta_{j-1} - exp(alpha_j)` for `Decreasing`
///
/// This makes optimization unconstrained in alpha-space while guaranteeing monotonicity.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Monotonicity {
    /// No constraint
    #[default]
    None,
    /// Monotonically increasing
    Increasing,
    /// Monotonically decreasing
    Decreasing,
}

// =============================================================================
// Exp reparameterization for monotonic splines (Pya & Wood 2015)
// =============================================================================

/// Convert unconstrained alpha parameters to monotonic beta coefficients.
///
/// - `beta[0] = alpha[0]`
/// - `beta[j] = beta[j-1] + sign * exp(alpha[j])` for j >= 1
///
/// where `sign = +1` for increasing, `-1` for decreasing.
fn alpha_to_beta(alpha: &[f64], monotonicity: &Monotonicity) -> Array1<f64> {
    let k = alpha.len();
    let mut beta = Array1::zeros(k);
    if k == 0 {
        return beta;
    }
    let sign = match monotonicity {
        Monotonicity::Increasing => 1.0,
        Monotonicity::Decreasing => -1.0,
        Monotonicity::None => {
            // No transform needed
            return Array1::from_vec(alpha.to_vec());
        }
    };
    beta[0] = alpha[0];
    for j in 1..k {
        // Clamp alpha to prevent exp() overflow
        let clamped = alpha[j].clamp(-MAX_EXP_ALPHA, MAX_EXP_ALPHA);
        beta[j] = beta[j - 1] + sign * clamped.exp();
    }
    beta
}

/// Convert monotonic beta coefficients back to unconstrained alpha parameters.
///
/// This is used for initialization: given an unconstrained beta solution,
/// compute alpha such that `alpha_to_beta(alpha) ~= beta`.
///
/// - `alpha[0] = beta[0]`
/// - `alpha[j] = ln(|beta[j] - beta[j-1]|)` for j >= 1 (clamped to avoid ln(0))
fn beta_to_alpha(beta: &[f64], monotonicity: &Monotonicity) -> Array1<f64> {
    let k = beta.len();
    let mut alpha = Array1::zeros(k);
    if k == 0 {
        return alpha;
    }
    if *monotonicity == Monotonicity::None {
        return Array1::from_vec(beta.to_vec());
    }
    let sign = match monotonicity {
        Monotonicity::Increasing => 1.0,
        Monotonicity::Decreasing => -1.0,
        Monotonicity::None => unreachable!(),
    };
    alpha[0] = beta[0];
    for j in 1..k {
        let diff = sign * (beta[j] - beta[j - 1]);
        // Clamp to a small positive value to avoid ln(0) or ln(negative)
        alpha[j] = diff.max(ZERO_TOL).ln();
    }
    alpha
}

/// Build the lower-triangular Jacobian matrix J for the exp reparameterization.
///
/// J is k x k with:
/// - `J[i, 0] = 1` for all i
/// - `J[i, j] = sign * exp(alpha[j])` for i >= j >= 1
/// - `J[i, j] = 0` for i < j
///
/// This represents d(beta)/d(alpha).
fn compute_monotonic_jacobian(alpha: &[f64], monotonicity: &Monotonicity) -> Array2<f64> {
    let k = alpha.len();
    let mut j_mat = Array2::zeros((k, k));
    let sign = match monotonicity {
        Monotonicity::Increasing => 1.0,
        Monotonicity::Decreasing => -1.0,
        Monotonicity::None => {
            // Identity Jacobian
            return Array2::eye(k);
        }
    };
    // Column 0: all ones (d beta_i / d alpha_0 = 1 for all i)
    for i in 0..k {
        j_mat[[i, 0]] = 1.0;
    }
    // Columns j >= 1: J[i, j] = sign * exp(alpha[j]) for i >= j
    for j in 1..k {
        let clamped = alpha[j].clamp(-MAX_EXP_ALPHA, MAX_EXP_ALPHA);
        let val = sign * clamped.exp();
        for i in j..k {
            j_mat[[i, j]] = val;
        }
    }
    j_mat
}

/// Efficiently compute X_tilde = X_smooth * J for the monotonic reparameterization.
///
/// Since J is lower-triangular with cumulative structure, we use reverse cumulative
/// sums to compute this in O(n*k) instead of O(n*k^2).
///
/// Column j of X_tilde:
/// - Column 0: sum of all X columns (each multiplied by 1)
/// - Column j (j >= 1): sign * exp(alpha[j]) * sum of X columns j..k
///
/// Writes transformed smooth columns directly into `target` (in-place).
/// Uses reverse cumulative sums computed row-by-row (no intermediate allocation).
fn compute_x_tilde_inplace(
    x_smooth: &ArrayView2<'_, f64>,
    alpha: &[f64],
    monotonicity: &Monotonicity,
    target: &mut ndarray::ArrayViewMut2<'_, f64>,
) {
    let n = x_smooth.nrows();
    let k = x_smooth.ncols();
    let sign = match monotonicity {
        Monotonicity::Increasing => 1.0,
        Monotonicity::Decreasing => -1.0,
        Monotonicity::None => {
            target.assign(x_smooth);
            return;
        }
    };

    // Precompute exp(alpha[j]) * sign for each column
    let scales: Vec<f64> = (0..k)
        .map(|j| {
            if j == 0 {
                1.0 // Column 0: unscaled cumulative sum
            } else {
                sign * alpha[j].clamp(-MAX_EXP_ALPHA, MAX_EXP_ALPHA).exp()
            }
        })
        .collect();

    // Process row-by-row: compute reverse cumulative sum and scale in one pass.
    // No intermediate n×k allocation needed.
    for i in 0..n {
        // Reverse cumulative sum for this row
        let mut cumsum = 0.0;
        for j in (0..k).rev() {
            cumsum += x_smooth[[i, j]];
            target[[i, j]] = scales[j] * cumsum;
        }
    }
}

/// Compute the transformed penalty S_tilde = J' * S * J for monotonic terms.
fn compute_s_tilde(s_penalty: &Array2<f64>, j_mat: &Array2<f64>) -> Array2<f64> {
    // J' * S * J — straightforward matrix multiply for the small k x k matrices
    let sj = s_penalty.dot(j_mat);
    j_mat.t().dot(&sj)
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
        let penalty = penalty_matrix(k, 2); // Second-order difference penalty
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
        let xtwx_block = xtwx
            .slice(s![start..start + k, start..start + k])
            .to_owned();
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
    nonneg_indices: Option<&[usize]>,
    nonpos_indices: Option<&[usize]>,
) -> Result<SmoothGLMResult> {
    let n = y.len();
    let p = x_full.ncols();

    if x_full.nrows() != n {
        return Err(RustyStatsError::dim_mismatch(
            n,
            x_full.nrows(),
            "x_full rows vs y length",
        ));
    }

    if smooth_specs.is_empty() {
        // No smooth terms — delegate to standard GLM fit and wrap result
        let unified_config = FitConfig::from(&config.irls_config);
        let irls = fit_glm_unified(
            y,
            x_full,
            family,
            link,
            &unified_config,
            offset,
            weights,
            None,
        )?;
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
            offset: if irls.offset.iter().all(|&v| v == 0.0) {
                None
            } else {
                Some(irls.offset)
            },
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
            return Err(RustyStatsError::dim_mismatch(
                k,
                spec.penalty.nrows(),
                format!(
                    "smooth spec {} penalty shape ({}, {}) vs expected ({}, {})",
                    i,
                    spec.penalty.nrows(),
                    spec.penalty.ncols(),
                    k,
                    k
                ),
            ));
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
    let term_indices: Vec<(usize, usize)> = smooth_specs
        .iter()
        .map(|s| (s.col_start, s.col_end))
        .collect();

    let offset_vec = offset.cloned().unwrap_or_else(|| Array1::zeros(n));
    let prior_weights = weights.cloned().unwrap_or_else(|| Array1::ones(n));
    let mut lambdas: Vec<f64> = smooth_specs.iter().map(|s| s.initial_lambda).collect();

    let has_monotonic = smooth_specs.iter().any(|s| s.is_monotonic());

    // Use x_full directly as x_combined — no reassembly needed
    let x_combined = x_full;
    let total_cols = p;

    // Initialize mu
    let mut mu = family.initialize_mu(y);
    let mut eta = link.link(&mu);
    let mut deviance = family.deviance(y, &mu, Some(&prior_weights));

    let mut converged = false;
    let mut iteration = 0;
    let mut coefficients = Array1::zeros(total_cols);
    let mut final_weights = Array1::ones(n);

    let log_lambda_min = config.lambda_min.ln();
    let log_lambda_max = config.lambda_max.ln();
    let mut penalty_matrix = Array2::zeros((total_cols, total_cols));
    let mut lambdas_stable_count = 0u32;

    // Track previous lambdas to skip rebuilding penalty_matrix when unchanged.
    // Only used in the non-monotonic path (monotonic always rebuilds due to J changes).
    let mut prev_lambdas: Vec<f64> = vec![f64::NAN; smooth_specs.len()];
    let mut penalty_dirty = true;

    // Pre-allocate per-iteration buffers to avoid heap allocations inside the IRLS loop
    let mut irls_weights = Array1::zeros(n);
    let mut combined_weights = Array1::zeros(n);
    let mut working_response = Array1::zeros(n);

    // Use a looser tolerance for smooth models — GCV lambda perturbation
    // introduces noise larger than 1e-8, so tighter tolerance just wastes iterations
    let smooth_tolerance = config.irls_config.tolerance.max(SMOOTH_CONVERGENCE_TOL);

    // Pre-allocate adjusted_response for the monotonic path — avoids cloning
    // working_response each iteration (saves one n-sized allocation per iteration)
    let mut adjusted_response = if has_monotonic {
        Array1::zeros(n)
    } else {
        Array1::zeros(0) // Not used in non-monotonic path
    };

    // Pre-allocate x_tilde for the monotonic path — cloned once from x_combined,
    // then only the monotonic smooth columns are overwritten each iteration.
    // Parametric columns (intercept, etc.) never change and stay correct.
    let mut x_tilde = if has_monotonic {
        x_combined.to_owned()
    } else {
        Array2::zeros((0, 0)) // Not used in non-monotonic path
    };

    // =========================================================================
    // Exp reparameterization state for monotonic terms
    // =========================================================================
    // For each monotonic smooth term, we maintain alpha parameters in unconstrained
    // space. Non-monotonic terms and parametric columns are unaffected.
    //
    // alpha_params[i] holds the alpha vector for smooth_specs[i] (only used if monotonic).
    let mut alpha_params: Vec<Option<Array1<f64>>> = smooth_specs
        .iter()
        .map(|spec| {
            if spec.is_monotonic() {
                let k = spec.col_end - spec.col_start;
                // Initialize alpha to small values (beta starts near zero)
                Some(Array1::zeros(k))
            } else {
                None
            }
        })
        .collect();

    // Flag: has alpha been initialized from an unconstrained solve?
    let mut alpha_initialized = !has_monotonic; // skip init if no monotonic terms

    while iteration < config.irls_config.max_iterations {
        iteration += 1;
        let deviance_old = deviance;

        // IRLS weights, combined weights, and working response — computed in a
        // single fused loop using pre-allocated buffers (no per-iteration heap allocs)
        let link_deriv = link.derivative(&mu);
        let variance = family.variance(&mu);

        for i in 0..n {
            irls_weights[i] = (1.0 / (variance[i] * link_deriv[i] * link_deriv[i]))
                .max(config.irls_config.min_weight)
                .min(MAX_IRLS_WEIGHT);
            combined_weights[i] = prior_weights[i] * irls_weights[i];
            working_response[i] = (eta[i] - offset_vec[i]) + (y[i] - mu[i]) * link_deriv[i];
        }

        let mut new_coef;

        // Save alpha state before WLS update, for step-halving (monotonic path only)
        let pre_wls_alphas: Vec<Option<Array1<f64>>> = if has_monotonic {
            alpha_params.clone()
        } else {
            vec![]
        };

        if has_monotonic {
            // =================================================================
            // Exp reparameterization path for monotonic smooth terms
            // =================================================================
            //
            // On first iteration, solve unconstrained WLS to get initial beta,
            // then convert to alpha via beta_to_alpha.
            //
            // On each iteration:
            // 1. Build X_tilde by replacing monotonic columns with X_smooth * J
            // 2. Build penalty with S_tilde = J' * S * J for monotonic terms
            // 3. Solve standard (unconstrained) WLS in alpha-space
            // 4. Update alpha, recover beta = h(alpha)
            // =================================================================

            if !alpha_initialized {
                // Run GCV on the ORIGINAL (unconstrained) basis to select lambda.
                // Then freeze lambda for all subsequent exp-reparam PIRLS iterations.
                // This avoids GCV instability caused by X_tilde changing each iteration.
                let (init_xtwx, init_xtwz) =
                    compute_xtwx_xtwz(x_combined, &working_response, &combined_weights)?;
                let init_ztwz: f64 = working_response
                    .iter()
                    .zip(combined_weights.iter())
                    .map(|(&zi, &wi)| wi * zi * zi)
                    .sum();
                let init_penalties: Vec<Array2<f64>> =
                    smooth_specs.iter().map(|s| s.penalty.clone()).collect();
                let optimizer = MultiTermGCVOptimizer::new_from_cached(
                    init_xtwx,
                    init_xtwz,
                    init_ztwz,
                    init_penalties,
                    term_indices.clone(),
                    n,
                    p_param,
                );
                lambdas = optimizer.optimize_lambdas(
                    &lambdas,
                    log_lambda_min,
                    log_lambda_max,
                    config.lambda_tol,
                    5,
                );
                // Allow GCV to re-run on the transformed (X_tilde, S_tilde) system
                // during the first exp-reparam iterations. The init GCV on the
                // unconstrained basis provides a starting lambda, but the optimal
                // lambda for the transformed system may differ.
                lambdas_stable_count = 0;

                // Run full unconstrained IRLS to convergence before switching
                // to exp reparameterization. The exp reparam PIRLS needs a
                // well-converged starting point — a single WLS solve is not enough.
                let mut init_coef = coefficients.clone();
                let mut init_mu = mu.clone();
                let mut init_eta = eta.clone();
                let mut init_dev = deviance;
                for _init_iter in 0..config.irls_config.max_iterations {
                    let init_link_deriv = link.derivative(&init_mu);
                    let init_var = family.variance(&init_mu);
                    let init_iw: Array1<f64> = (0..n)
                        .map(|i| {
                            let d = init_link_deriv[i];
                            let v = init_var[i];
                            (1.0 / (v * d * d))
                                .max(config.irls_config.min_weight)
                                .min(MAX_IRLS_WEIGHT)
                        })
                        .collect();
                    let init_cw = &prior_weights * &init_iw;
                    let init_z = (&init_eta - &offset_vec) + &((y - &init_mu) * &init_link_deriv);

                    let mut init_penalty = Array2::zeros((total_cols, total_cols));
                    for (i, spec) in smooth_specs.iter().enumerate() {
                        embed_penalty(&mut init_penalty, &spec.penalty, spec.col_start, lambdas[i]);
                    }
                    let (new_init_coef, _) = solve_weighted_least_squares_with_penalty_matrix(
                        x_combined,
                        &init_z,
                        &init_cw,
                        &init_penalty,
                        true, // skip covariance — only coefficients needed for initialization
                    )?;
                    init_coef = new_init_coef;
                    init_eta = &x_combined.dot(&init_coef) + &offset_vec;
                    init_mu = family.clamp_mu(&link.inverse(&init_eta));
                    let new_dev = family.deviance(y, &init_mu, Some(&prior_weights));
                    let init_rel = if init_dev.abs() > ZERO_TOL {
                        (init_dev - new_dev).abs() / init_dev.abs()
                    } else {
                        (init_dev - new_dev).abs()
                    };
                    init_dev = new_dev;
                    if init_rel < config.irls_config.tolerance {
                        break;
                    }
                }
                // Convert unconstrained beta to alpha for each monotonic term
                for (i, spec) in smooth_specs.iter().enumerate() {
                    if let Some(ref mut alpha) = alpha_params[i] {
                        let beta_slice = &init_coef.as_slice().expect("contiguous array")
                            [spec.col_start..spec.col_end];
                        *alpha = beta_to_alpha(beta_slice, &spec.monotonicity);
                    }
                }

                // Set initial coefficients: parametric from init_coef, smooth from alpha->beta
                coefficients = init_coef;
                for (i, spec) in smooth_specs.iter().enumerate() {
                    if let Some(ref alpha) = alpha_params[i] {
                        let beta = alpha_to_beta(
                            alpha.as_slice().expect("contiguous array"),
                            &spec.monotonicity,
                        );
                        let coef_slice = coefficients.as_slice_mut().expect("contiguous array");
                        for (j, &b) in beta.iter().enumerate() {
                            coef_slice[spec.col_start + j] = b;
                        }
                    }
                }

                // Update predictions from the (now monotonic) coefficients
                eta = &x_combined.dot(&coefficients) + &offset_vec;
                mu = family.clamp_mu(&link.inverse(&eta));
                deviance = family.deviance(y, &mu, Some(&prior_weights));

                alpha_initialized = true;
                final_weights = combined_weights.clone();

                // Skip the rest of this iteration — we've just initialized
                continue;
            }

            // Update X_tilde: only overwrite monotonic smooth columns each iteration.
            // Parametric columns were set once during pre-allocation and never change.
            for (i, spec) in smooth_specs.iter().enumerate() {
                if let Some(ref alpha) = alpha_params[i] {
                    let x_smooth = x_combined.slice(s![.., spec.col_start..spec.col_end]);
                    let mut target = x_tilde.slice_mut(s![.., spec.col_start..spec.col_end]);
                    compute_x_tilde_inplace(
                        &x_smooth,
                        alpha.as_slice().expect("contiguous array"),
                        &spec.monotonicity,
                        &mut target,
                    );
                }
            }

            // Adjust working response for the linearization offset.
            // The model is nonlinear in alpha: eta = X * beta(alpha).
            // Taylor expansion: X*beta(alpha) ≈ X*beta(alpha_old) + X*J*(alpha - alpha_old)
            //                                  = X_tilde*alpha + X*(beta_old - J*alpha_old)
            // The offset c = X*(beta_old - J*alpha_old) must be subtracted from z
            // so that the WLS solve gives the full new alpha directly:
            //   z_adj = z - c, then alpha_new = (X_tilde'WX_tilde)^-1 X_tilde'W z_adj
            // Compute linearization offset c = X_smooth * (beta - J*alpha)
            // and adjust working response: z_adj = z - c
            adjusted_response.assign(&working_response);
            for (i, spec) in smooth_specs.iter().enumerate() {
                if let Some(ref alpha) = alpha_params[i] {
                    let alpha_slice = alpha.as_slice().expect("contiguous array");
                    let beta_mono = alpha_to_beta(alpha_slice, &spec.monotonicity);
                    let j_mat = compute_monotonic_jacobian(alpha_slice, &spec.monotonicity);
                    let j_alpha = j_mat.dot(alpha);
                    let offset_mono = &beta_mono - &j_alpha;
                    let x_smooth = x_combined.slice(s![.., spec.col_start..spec.col_end]);
                    // Subtract c = X_smooth @ offset_mono in-place (no extra allocation)
                    let k_sm = spec.col_end - spec.col_start;
                    let offset_slice = offset_mono.as_slice().expect("contiguous array");
                    let adj_slice = adjusted_response.as_slice_mut().expect("contiguous array");
                    for row in 0..n {
                        let mut dot = 0.0;
                        for col in 0..k_sm {
                            dot += x_smooth[[row, col]] * offset_slice[col];
                        }
                        adj_slice[row] -= dot;
                    }
                }
            }

            let x_tilde_view = x_tilde.view();
            let (cached_xtwx_t, cached_xtwz_t) =
                compute_xtwx_xtwz(x_tilde_view, &adjusted_response, &combined_weights)?;

            // Compute z'Wz for GCV (using adjusted response)
            let ztwz: f64 = adjusted_response
                .iter()
                .zip(combined_weights.iter())
                .map(|(&zi, &wi)| wi * zi * zi)
                .sum();

            // GCV optimization on transformed problem
            let run_gcv = lambdas_stable_count < 1 && (iteration <= 3 || iteration % 2 == 0);
            if run_gcv {
                let old_lambdas = lambdas.clone();

                // Build transformed penalties for GCV optimizer
                let transformed_penalties: Vec<Array2<f64>> = smooth_specs
                    .iter()
                    .enumerate()
                    .map(|(i, spec)| {
                        if let Some(ref alpha) = alpha_params[i] {
                            let j_mat = compute_monotonic_jacobian(
                                alpha.as_slice().expect("contiguous array"),
                                &spec.monotonicity,
                            );
                            compute_s_tilde(&spec.penalty, &j_mat)
                        } else {
                            spec.penalty.clone()
                        }
                    })
                    .collect();

                let optimizer = MultiTermGCVOptimizer::new_from_cached(
                    cached_xtwx_t.clone(),
                    cached_xtwz_t.clone(),
                    ztwz,
                    transformed_penalties,
                    term_indices.clone(),
                    n,
                    p_param,
                );

                lambdas = optimizer.optimize_lambdas(
                    &lambdas,
                    log_lambda_min,
                    log_lambda_max,
                    config.lambda_tol,
                    3,
                );

                let max_rel_change = old_lambdas
                    .iter()
                    .zip(lambdas.iter())
                    .map(|(&old, &new)| {
                        if old.abs() < 1e-12 {
                            (new - old).abs()
                        } else {
                            (new - old).abs() / old.abs()
                        }
                    })
                    .fold(0.0f64, f64::max);

                if max_rel_change < 0.01 {
                    lambdas_stable_count += 1;
                } else {
                    lambdas_stable_count = 0;
                }
            }

            // Build transformed penalty matrix for WLS solve
            penalty_matrix.fill(0.0);
            for (i, spec) in smooth_specs.iter().enumerate() {
                if let Some(ref alpha) = alpha_params[i] {
                    let j_mat = compute_monotonic_jacobian(
                        alpha.as_slice().expect("contiguous array"),
                        &spec.monotonicity,
                    );
                    let s_tilde = compute_s_tilde(&spec.penalty, &j_mat);
                    embed_penalty(&mut penalty_matrix, &s_tilde, spec.col_start, lambdas[i]);
                } else {
                    embed_penalty(
                        &mut penalty_matrix,
                        &spec.penalty,
                        spec.col_start,
                        lambdas[i],
                    );
                }
            }

            // Solve penalized WLS in alpha-space (X_tilde, S_tilde)
            let (alpha_coef, _) =
                solve_wls_from_precomputed(&cached_xtwx_t, &cached_xtwz_t, &penalty_matrix, true)?;

            new_coef = coefficients.clone();

            // Parametric columns: take the WLS solution directly (linear, no exp)
            if let Some(first_spec) = smooth_specs.first() {
                let coef_slice = new_coef.as_slice_mut().expect("contiguous array");
                for j in 0..first_spec.col_start {
                    coef_slice[j] = alpha_coef[j];
                }
            }

            // Non-monotonic smooth terms: accept WLS solution directly (linear)
            for (i, spec) in smooth_specs.iter().enumerate() {
                if alpha_params[i].is_none() {
                    let coef_slice = new_coef.as_slice_mut().expect("contiguous array");
                    for j in spec.col_start..spec.col_end {
                        coef_slice[j] = alpha_coef[j];
                    }
                }
            }

            // Monotonic smooth terms: accept WLS solution as the new alpha candidate,
            // with backtracking line search to prevent oscillation.
            //
            // The exp reparameterization makes the problem nonlinear in alpha, so
            // the linearized WLS step can overshoot. We try the full step first;
            // if deviance increases, we halve the step until deviance improves or
            // we've tried enough fractions.
            {
                // Compute WLS-proposed alpha (full step) for each monotonic term
                let wls_alphas: Vec<Option<Array1<f64>>> = smooth_specs
                    .iter()
                    .enumerate()
                    .map(|(i, spec)| {
                        if alpha_params[i].is_some() {
                            let alpha_wls = &alpha_coef.as_slice().expect("contiguous array")
                                [spec.col_start..spec.col_end];
                            let clamped: Vec<f64> = alpha_wls
                                .iter()
                                .enumerate()
                                .map(|(j, &a)| {
                                    if j == 0 {
                                        a
                                    } else {
                                        a.clamp(-MAX_EXP_ALPHA, MAX_EXP_ALPHA)
                                    }
                                })
                                .collect();
                            Some(Array1::from_vec(clamped))
                        } else {
                            None
                        }
                    })
                    .collect();

                // Parametric and non-monotonic smooth columns are already set
                // in new_coef above (lines before this block).

                // Try full step, then halve if deviance increases
                let mut best_step = 1.0_f64;
                let mut best_dev = f64::INFINITY;
                let mut best_alphas = wls_alphas.clone();

                for trial in 0..6 {
                    let step = if trial == 0 {
                        1.0
                    } else {
                        best_step * 0.5_f64.powi(trial)
                    };
                    let mut trial_coef = new_coef.clone();

                    // Blend alpha: old_alpha * (1 - step) + wls_alpha * step
                    let mut trial_alphas: Vec<Option<Array1<f64>>> =
                        Vec::with_capacity(smooth_specs.len());
                    for (i, spec) in smooth_specs.iter().enumerate() {
                        if let (Some(ref old_alpha), Some(ref new_alpha)) =
                            (&pre_wls_alphas[i], &wls_alphas[i])
                        {
                            let blended = if step >= 1.0 {
                                new_alpha.clone()
                            } else {
                                old_alpha * (1.0 - step) + new_alpha * step
                            };
                            let beta = alpha_to_beta(
                                blended.as_slice().expect("contiguous array"),
                                &spec.monotonicity,
                            );
                            let cs = trial_coef.as_slice_mut().expect("contiguous array");
                            for (j, &b) in beta.iter().enumerate() {
                                cs[spec.col_start + j] = b;
                            }
                            trial_alphas.push(Some(blended));
                        } else {
                            trial_alphas.push(None);
                        }
                    }

                    let trial_eta = &x_combined.dot(&trial_coef) + &offset_vec;
                    let trial_mu = family.clamp_mu(&link.inverse(&trial_eta));
                    let trial_dev = family.deviance(y, &trial_mu, Some(&prior_weights));

                    if trial == 0 || trial_dev < best_dev {
                        best_dev = trial_dev;
                        best_step = step;
                        best_alphas = trial_alphas;
                        new_coef = trial_coef;
                        // If full step didn't increase deviance, accept it
                        if trial == 0 && trial_dev <= deviance_old {
                            break;
                        }
                    }
                    // If we found a step that reduces deviance, stop
                    if trial > 0 && best_dev <= deviance_old {
                        break;
                    }
                }

                // If no step reduced deviance, keep old coefficients and alpha
                if best_dev > deviance_old {
                    new_coef = coefficients.clone();
                    // Restore alpha_params from pre-WLS state
                    for (i, _spec) in smooth_specs.iter().enumerate() {
                        if let Some(ref old_a) = pre_wls_alphas[i] {
                            if let Some(ref mut alpha) = alpha_params[i] {
                                *alpha = old_a.clone();
                            }
                        }
                    }
                } else {
                    // Update alpha_params with the best step
                    for (i, _spec) in smooth_specs.iter().enumerate() {
                        if let Some(ref best_a) = best_alphas[i] {
                            if let Some(ref mut alpha) = alpha_params[i] {
                                *alpha = best_a.clone();
                            }
                        }
                    }

                    // Recompute beta from accepted alpha for new_coef
                    for (i, spec) in smooth_specs.iter().enumerate() {
                        if let Some(ref alpha) = alpha_params[i] {
                            let beta = alpha_to_beta(
                                alpha.as_slice().expect("contiguous array"),
                                &spec.monotonicity,
                            );
                            let coef_slice = new_coef.as_slice_mut().expect("contiguous array");
                            for (j, &b) in beta.iter().enumerate() {
                                coef_slice[spec.col_start + j] = b;
                            }
                        }
                    }
                }
            }
        } else {
            // =================================================================
            // Standard (unconstrained) path — no monotonic terms
            // =================================================================

            // Compute X'WX and X'Wz ONCE per iteration — shared by GCV and WLS
            let (cached_xtwx, cached_xtwz) =
                compute_xtwx_xtwz(x_combined, &working_response, &combined_weights)?;

            // Compute z'Wz scalar for GCV RSS computation (O(n), trivial)
            let ztwz: f64 = working_response
                .iter()
                .zip(combined_weights.iter())
                .map(|(&zi, &wi)| wi * zi * zi)
                .sum();

            // Optimize lambdas using cached matrices (no X'WX recomputation)
            // Skip GCV once lambdas have stabilized for 2 consecutive iterations
            let run_gcv = lambdas_stable_count < 1 && (iteration <= 3 || iteration % 2 == 0);
            if run_gcv {
                let old_lambdas = lambdas.clone();

                let penalties: Vec<Array2<f64>> =
                    smooth_specs.iter().map(|s| s.penalty.clone()).collect();

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
                    &lambdas,
                    log_lambda_min,
                    log_lambda_max,
                    config.lambda_tol,
                    3,
                );

                // Check if lambdas have stabilized (max relative change < 1%)
                let max_rel_change = old_lambdas
                    .iter()
                    .zip(lambdas.iter())
                    .map(|(&old, &new)| {
                        if old.abs() < 1e-12 {
                            (new - old).abs()
                        } else {
                            (new - old).abs() / old.abs()
                        }
                    })
                    .fold(0.0f64, f64::max);

                if max_rel_change < 0.01 {
                    lambdas_stable_count += 1;
                } else {
                    lambdas_stable_count = 0;
                }

                // Mark penalty as dirty if any lambda changed
                penalty_dirty = lambdas != prev_lambdas;
            }

            // Build penalty matrix only when lambdas have changed
            if penalty_dirty {
                penalty_matrix.fill(0.0);
                for (i, spec) in smooth_specs.iter().enumerate() {
                    embed_penalty(
                        &mut penalty_matrix,
                        &spec.penalty,
                        spec.col_start,
                        lambdas[i],
                    );
                }
                prev_lambdas.clone_from_slice(&lambdas);
                penalty_dirty = false;
            }

            // Solve WLS from pre-computed X'WX — skip covariance on intermediate
            // iterations (O(p³) savings per iteration). The covariance is computed
            // once at the end by assemble_smooth_result_from_specs.
            let (coef, _) =
                solve_wls_from_precomputed(&cached_xtwx, &cached_xtwz, &penalty_matrix, true)?;
            new_coef = coef;
        }

        // Apply sign constraints on parametric (non-smooth) columns
        if let Some(nn) = nonneg_indices {
            for &idx in nn {
                if idx < new_coef.len() && new_coef[idx] < 0.0 {
                    new_coef[idx] = 0.0;
                }
            }
        }
        if let Some(np) = nonpos_indices {
            for &idx in np {
                if idx < new_coef.len() && new_coef[idx] > 0.0 {
                    new_coef[idx] = 0.0;
                }
            }
        }

        // Update eta and mu with new coefficients (always use original X, not X_tilde)
        let eta_new = &x_combined.dot(&new_coef) + &offset_vec;
        let mu_new = family.clamp_mu(&link.inverse(&eta_new));
        let deviance_new = family.deviance(y, &mu_new, Some(&prior_weights));

        // Step halving if deviance increased (blend old coefficients <-> new)
        // For monotonic terms, we blend in alpha-space to preserve the
        // exp reparameterization (blending in beta-space would break monotonicity).
        if deviance_new > deviance_old * 1.0001 && iteration > 1 {
            let mut step = 0.5;
            let mut best_coef = new_coef.clone();
            let mut best_dev = deviance_new;

            // For monotonic terms, use the saved pre-WLS alpha state as the "old"
            // alphas and the current alpha_params (post-WLS) as the "new" alphas.
            // This avoids the lossy beta_to_alpha recovery.
            let accepted_alphas: Vec<Option<Array1<f64>>> = alpha_params.to_vec();
            let old_alphas: Vec<Option<Array1<f64>>> = if has_monotonic {
                pre_wls_alphas.clone()
            } else {
                vec![]
            };
            let mut best_step = 1.0f64;

            for _ in 0..5 {
                let mut blended;
                if has_monotonic {
                    // Blend parametric columns linearly
                    blended = &coefficients * (1.0 - step) + &new_coef * step;

                    // For monotonic terms, blend in alpha-space using tracked alphas
                    for (i, spec) in smooth_specs.iter().enumerate() {
                        if let (Some(ref new_alpha), Some(ref old_alpha)) =
                            (&accepted_alphas[i], &old_alphas[i])
                        {
                            let blended_alpha = old_alpha * (1.0 - step) + new_alpha * step;
                            let blended_beta = alpha_to_beta(
                                blended_alpha.as_slice().expect("contiguous array"),
                                &spec.monotonicity,
                            );
                            let coef_slice = blended.as_slice_mut().expect("contiguous array");
                            for (j, &b) in blended_beta.iter().enumerate() {
                                coef_slice[spec.col_start + j] = b;
                            }
                        }
                    }
                } else {
                    blended = &coefficients * (1.0 - step) + &new_coef * step;
                }

                // Re-apply sign constraints after blending
                if let Some(nn) = nonneg_indices {
                    for &idx in nn {
                        if idx < blended.len() && blended[idx] < 0.0 {
                            blended[idx] = 0.0;
                        }
                    }
                }
                if let Some(np) = nonpos_indices {
                    for &idx in np {
                        if idx < blended.len() && blended[idx] > 0.0 {
                            blended[idx] = 0.0;
                        }
                    }
                }

                let eta_full = &x_combined.dot(&blended) + &offset_vec;
                let mu_blend = family.clamp_mu(&link.inverse(&eta_full));
                let dev_blend = family.deviance(y, &mu_blend, Some(&prior_weights));

                if dev_blend < best_dev {
                    best_dev = dev_blend;
                    best_coef = blended;
                    best_step = step;
                }
                step *= 0.5;
            }

            // Update alpha_params by blending at the winning step fraction,
            // instead of reverse-engineering via beta_to_alpha (which is lossy
            // when consecutive betas are nearly equal).
            if has_monotonic {
                for (i, _spec) in smooth_specs.iter().enumerate() {
                    if let (Some(ref mut alpha), Some(ref new_a), Some(ref old_a)) =
                        (&mut alpha_params[i], &accepted_alphas[i], &old_alphas[i])
                    {
                        *alpha = old_a * (1.0 - best_step) + new_a * best_step;
                    }
                }
            }

            coefficients = best_coef;
        } else {
            coefficients = new_coef;
        }

        // Update state
        eta = &x_combined.dot(&coefficients) + &offset_vec;
        mu = family.clamp_mu(&link.inverse(&eta));
        deviance = family.deviance(y, &mu, Some(&prior_weights));
        final_weights = combined_weights.clone();

        let rel_change = if deviance_old.abs() > ZERO_TOL {
            (deviance_old - deviance).abs() / deviance_old.abs()
        } else {
            (deviance_old - deviance).abs()
        };

        if rel_change < smooth_tolerance {
            converged = true;
            break;
        }
    }

    // Assemble result directly from SmoothTermSpec (no SmoothTermData conversion)
    let penalty_specs: Vec<(&Array2<f64>, usize, usize)> = smooth_specs
        .iter()
        .map(|s| (&s.penalty, s.col_start, s.col_end))
        .collect();

    Ok(assemble_smooth_result_from_specs(
        coefficients,
        mu,
        eta,
        deviance,
        iteration,
        converged,
        &final_weights,
        x_combined,
        &penalty_specs,
        &lambdas,
        p_param,
        family.name(),
        prior_weights,
        y,
        offset,
        // Covariance is computed once here by assemble_smooth_result_from_specs,
        // rather than redundantly on every intermediate IRLS iteration.
        None,
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
    use crate::families::{GammaFamily, GaussianFamily, PoissonFamily};
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
        assert_eq!(term.k(), 9); // df=10, no intercept = 9 columns
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
        assert_eq!(penalty[[2, 2]], 0.5); // lambda1 * I
        assert_eq!(penalty[[7, 7]], 2.0); // lambda2 * I
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
        let y: Array1<f64> = x_vals
            .iter()
            .map(|&xi| 2.0 + xi.sin() + 0.1 * (xi * 7.3).sin())
            .collect();
        // Parametric part: intercept column
        let x_param = Array2::from_shape_fn((n, 1), |(_, _)| 1.0);
        (y, x_param, x_vals)
    }

    /// Helper: generate Poisson data with a smooth effect.
    fn poisson_smooth_data(n: usize) -> (Array1<f64>, Array2<f64>, Array1<f64>) {
        let x_vals: Array1<f64> = (0..n).map(|i| i as f64 * 10.0 / n as f64).collect();
        let y: Array1<f64> = x_vals
            .iter()
            .map(|&xi| {
                let mu = (0.5 + 0.3 * xi.sin()).exp();
                // Deterministic "Poisson-like" values (round to nearest int, min 0)
                (mu + 0.5).floor().max(0.0)
            })
            .collect();
        let x_param = Array2::from_shape_fn((n, 1), |(_, _)| 1.0);
        (y, x_param, x_vals)
    }

    /// Helper: concatenate parametric + basis into full design matrix and build SmoothTermSpec.
    fn make_full_matrix(
        x_param: &Array2<f64>,
        basis: &Array2<f64>,
    ) -> (Array2<f64>, Vec<SmoothTermSpec>) {
        let p_param = x_param.ncols();
        let k = basis.ncols();
        let x_full = ndarray::concatenate![ndarray::Axis(1), *x_param, *basis]
            .as_standard_layout()
            .to_owned();
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
            &y,
            x_full.view(),
            &specs,
            &GaussianFamily,
            &IdentityLink,
            &config,
            None,
            None,
            None,
            None,
        )
        .unwrap();

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
            &y,
            x_full.view(),
            &specs,
            &GaussianFamily,
            &IdentityLink,
            &config,
            None,
            None,
            None,
            None,
        )
        .unwrap();

        assert_eq!(result.smooth_edfs.len(), 1);
        assert!(
            result.smooth_edfs[0] > 1.0,
            "EDF should be > 1 for non-trivial smooth"
        );
        assert!(
            result.smooth_edfs[0] < k as f64,
            "EDF {} should be < k {}",
            result.smooth_edfs[0],
            k
        );
        assert!(result.total_edf > 1.0);
    }

    #[test]
    fn test_fit_smooth_glm_gaussian_gcv_positive() {
        let (y, x_param, x_vals) = gaussian_smooth_data(100);
        let basis = bs_basis(&x_vals, 10, 3, None, false);
        let (x_full, specs) = make_full_matrix(&x_param, &basis);
        let config = SmoothGLMConfig::default();

        let result = fit_smooth_glm_full_matrix(
            &y,
            x_full.view(),
            &specs,
            &GaussianFamily,
            &IdentityLink,
            &config,
            None,
            None,
            None,
            None,
        )
        .unwrap();

        assert!(result.gcv > 0.0, "GCV should be positive");
        assert_eq!(result.lambdas.len(), 1);
        assert!(
            result.lambdas[0] > 0.0,
            "Selected lambda should be positive"
        );
    }

    #[test]
    fn test_fit_smooth_glm_poisson_converges() {
        let (y, x_param, x_vals) = poisson_smooth_data(200);
        let basis = bs_basis(&x_vals, 8, 3, None, false);
        let (x_full, specs) = make_full_matrix(&x_param, &basis);
        let config = SmoothGLMConfig::default();

        let result = fit_smooth_glm_full_matrix(
            &y,
            x_full.view(),
            &specs,
            &PoissonFamily,
            &LogLink,
            &config,
            None,
            None,
            None,
            None,
        )
        .unwrap();

        assert!(result.converged, "Poisson smooth GLM should converge");
        assert!(
            result.fitted_values.iter().all(|&v| v > 0.0),
            "Poisson fitted values must be positive"
        );
    }

    #[test]
    fn test_fit_smooth_glm_covariance_shape() {
        let (y, x_param, x_vals) = gaussian_smooth_data(100);
        let basis = bs_basis(&x_vals, 8, 3, None, false);
        let p_total = x_param.ncols() + basis.ncols();
        let (x_full, specs) = make_full_matrix(&x_param, &basis);
        let config = SmoothGLMConfig::default();

        let result = fit_smooth_glm_full_matrix(
            &y,
            x_full.view(),
            &specs,
            &GaussianFamily,
            &IdentityLink,
            &config,
            None,
            None,
            None,
            None,
        )
        .unwrap();

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
            &y,
            x_full.view(),
            &specs,
            &PoissonFamily,
            &LogLink,
            &config,
            Some(&offset),
            None,
            None,
            None,
        )
        .unwrap();

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
            &y,
            x_full.view(),
            &specs,
            &GaussianFamily,
            &IdentityLink,
            &config,
            None,
            Some(&weights),
            None,
            None,
        )
        .unwrap();

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
        let x_full = ndarray::concatenate![
            ndarray::Axis(1),
            x_param.slice(ndarray::s![0..50, ..]),
            basis
        ]
        .as_standard_layout()
        .to_owned();
        let specs = vec![SmoothTermSpec {
            col_start: 1,
            col_end: 1 + k,
            penalty: crate::splines::penalized::penalty_matrix(k, 2),
            monotonicity: Monotonicity::None,
            initial_lambda: 1.0,
        }];
        let config = SmoothGLMConfig::default();

        let result = fit_smooth_glm_full_matrix(
            &y,
            x_full.view(),
            &specs,
            &GaussianFamily,
            &IdentityLink,
            &config,
            None,
            None,
            None,
            None,
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
        let y: Array1<f64> = x1
            .iter()
            .zip(x2.iter())
            .map(|(&a, &b)| 2.0 + a.sin() + 0.5 * b.cos())
            .collect();

        let x_param = Array2::from_shape_fn((n, 1), |(_, _)| 1.0);
        let basis1 = bs_basis(&x1, 8, 3, None, false);
        let basis2 = bs_basis(&x2, 6, 3, None, false);
        let k1 = basis1.ncols();
        let k2 = basis2.ncols();
        let x_full = ndarray::concatenate![ndarray::Axis(1), x_param, basis1, basis2]
            .as_standard_layout()
            .to_owned();
        let specs = vec![
            SmoothTermSpec {
                col_start: 1,
                col_end: 1 + k1,
                penalty: crate::splines::penalized::penalty_matrix(k1, 2),
                monotonicity: Monotonicity::None,
                initial_lambda: 1.0,
            },
            SmoothTermSpec {
                col_start: 1 + k1,
                col_end: 1 + k1 + k2,
                penalty: crate::splines::penalized::penalty_matrix(k2, 2),
                monotonicity: Monotonicity::None,
                initial_lambda: 1.0,
            },
        ];
        let config = SmoothGLMConfig::default();

        let result = fit_smooth_glm_full_matrix(
            &y,
            x_full.view(),
            &specs,
            &GaussianFamily,
            &IdentityLink,
            &config,
            None,
            None,
            None,
            None,
        )
        .unwrap();

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
            &y,
            x_full.view(),
            &specs,
            &GaussianFamily,
            &IdentityLink,
            &config,
            None,
            None,
            None,
            None,
        )
        .unwrap();

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
            &y,
            x_param.view(),
            &[],
            &GaussianFamily,
            &IdentityLink,
            &config,
            None,
            None,
            None,
            None,
        )
        .unwrap();

        assert!(result.converged);
        assert!(result.lambdas.is_empty());
        assert!(result.smooth_edfs.is_empty());
        assert_eq!(result.coefficients.len(), 1); // intercept only
    }
}
