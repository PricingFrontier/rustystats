use super::{compute_irls_weights, initialize_mu_safe, validate_glm_inputs};

// =============================================================================
// IRLS: Iteratively Reweighted Least Squares
// =============================================================================
//
// This is THE algorithm for fitting GLMs. Understanding it will help you
// understand what the computer is actually doing when you call model.fit().
//
// THE BIG PICTURE
// ---------------
// We want to find β that maximizes the likelihood of seeing our data.
// For GLMs, we can't solve this directly, so we use an iterative approach:
//
//     Start with initial guess β⁰
//     Repeat:
//         1. Compute predicted values μ from current β
//         2. Compute "working weights" W based on variance and link
//         3. Compute "working response" z (linearized version of problem)
//         4. Solve weighted least squares: (X'WX)β = X'Wz
//         5. Check if converged; if not, go to step 1
//
// WHY "REWEIGHTED"?
// -----------------
// The weights W change at each iteration because:
//   - Variance depends on μ: Var(Y) = φ × V(μ)
//   - The link function derivative depends on μ
//
// Observations with higher variance get LESS weight (they're noisier).
// This is how GLMs handle heteroscedasticity automatically!
//
// THE WORKING RESPONSE (The Clever Trick)
// ---------------------------------------
// The "working response" z linearizes the problem:
//
//     z = η + (y - μ) × g'(μ)
//
// where η = g(μ) is the linear predictor and g'(μ) is the link derivative.
//
// This transforms our non-linear problem into a weighted linear regression!
//
// CONVERGENCE
// -----------
// We stop when the change in deviance (or coefficients) is small enough.
// If we don't converge, something might be wrong:
//   - Complete separation in logistic regression
//   - Outliers or data issues
//   - Need more iterations
//
// =============================================================================

const CONSTRAINED_BEST_EARLY_STOP_PATIENCE: usize = 2;
const SPARSE_XTWX_THREAD_CAP: usize = 2;
const SPARSE_XTWX_LOCAL_MATRIX_CAP_THRESHOLD_BYTES: usize = 1024 * 1024;
const SPARSE_XTWX_DENSITY_THRESHOLD: f64 = 0.35;
const SPARSE_ROW_CACHE_DENSITY_THRESHOLD: f64 = 0.35;

use nalgebra::{DMatrix, DVector};
use ndarray::{Array1, Array2, ArrayView2};
use rayon::prelude::*;
use std::time::Instant;

use crate::constants::{
    CONVERGENCE_TOL, DEFAULT_MAX_ITER, IRLS_ACCEPT_REL_SLACK, IRLS_MAX_HALF_STEPS, MAX_IRLS_WEIGHT,
    MIN_IRLS_WEIGHT, ZERO_TOL,
};
use crate::error::{Result, RustyStatsError};
use crate::families::Family;
use crate::links::Link;
use crate::regularization::{Penalty, RegularizationConfig, Standardization};

// =============================================================================
// Configuration
// =============================================================================

/// Configuration options for the IRLS algorithm.
///
/// These control how the fitting process works. The defaults are sensible
/// for most problems, but you may need to adjust them for difficult cases.
#[derive(Debug, Clone)]
pub struct IRLSConfig {
    /// Maximum number of iterations before giving up.
    /// Default: 25 (usually converges much faster)
    pub max_iterations: usize,

    /// Convergence tolerance for deviance change.
    /// We stop when: |deviance_new - deviance_old| / deviance_old < tolerance
    /// Default: 1e-8 (very tight convergence)
    pub tolerance: f64,

    /// Minimum value for weights to avoid numerical issues.
    /// Very small weights can cause instability.
    /// Default: 1e-10
    pub min_weight: f64,

    /// Whether to print iteration progress.
    /// Default: false
    pub verbose: bool,

    /// Coefficient indices that must be non-negative (β ≥ 0).
    /// After each WLS step, these coefficients are projected to max(0, β).
    /// Used for: monotonic splines (ms), pos() terms.
    /// Default: empty (no constraints)
    pub nonneg_indices: Vec<usize>,

    /// Coefficient indices that must be non-positive (β ≤ 0).
    /// After each WLS step, these coefficients are projected to min(0, β).
    /// Used for: neg() terms.
    /// Default: empty (no constraints)
    pub nonpos_indices: Vec<usize>,

    /// Skip the O(p³) covariance inverse in WLS solves (coefficients are still
    /// computed identically). Set by CV fold fits, which read only coefficients.
    /// Default: false
    pub skip_covariance: bool,
}

impl Default for IRLSConfig {
    fn default() -> Self {
        Self {
            max_iterations: DEFAULT_MAX_ITER,
            tolerance: CONVERGENCE_TOL,
            min_weight: MIN_IRLS_WEIGHT,
            verbose: false,
            nonneg_indices: Vec::new(),
            nonpos_indices: Vec::new(),
            skip_covariance: false,
        }
    }
}

// =============================================================================
// Unified Configuration
// =============================================================================

/// Unified configuration for GLM fitting.
///
/// Combines IRLS algorithm settings, regularization, and optional data inputs
/// into a single struct. This is the preferred way to configure GLM fitting —
/// it replaces the need to choose between `fit_glm`, `fit_glm_full`,
/// `fit_glm_warm_start`, `fit_glm_regularized`, and `fit_glm_regularized_warm`.
///
/// # Examples
/// ```ignore
/// // Simple unregularized fit:
/// let config = FitConfig::default();
///
/// // Ridge with warm start:
/// let config = FitConfig::default()
///     .with_regularization(RegularizationConfig::ridge(0.1))
///     .with_init_coefficients(prev_result.coefficients.clone());
///
/// // Lasso (automatically uses coordinate descent):
/// let config = FitConfig::default()
///     .with_regularization(RegularizationConfig::lasso(0.5));
/// ```
#[derive(Debug, Clone)]
pub struct FitConfig {
    /// Maximum number of IRLS iterations.
    /// Default: 25
    pub max_iterations: usize,

    /// Convergence tolerance for relative deviance change.
    /// Default: 1e-8
    pub tolerance: f64,

    /// Minimum IRLS weight to avoid numerical instability.
    /// Default: 1e-10
    pub min_weight: f64,

    /// Print iteration progress.
    /// Default: false
    pub verbose: bool,

    /// Coefficient indices constrained to be non-negative (β ≥ 0).
    /// Used for monotonic splines, pos() terms.
    pub nonneg_indices: Vec<usize>,

    /// Coefficient indices constrained to be non-positive (β ≤ 0).
    /// Used for neg() terms.
    pub nonpos_indices: Vec<usize>,

    /// Regularization configuration (penalty type + settings).
    /// Default: no regularization
    pub regularization: RegularizationConfig,

    /// Skip covariance matrix computation (for CV paths where only
    /// coefficients are needed). Saves O(n×p²) computation.
    /// Default: false
    pub skip_covariance: bool,

    /// Optional internal standardization for regularized parametric fits.
    /// Coefficients are accepted and returned on the caller's original scale.
    /// Default: None
    pub standardization: Option<Standardization>,
}

impl Default for FitConfig {
    fn default() -> Self {
        Self {
            max_iterations: DEFAULT_MAX_ITER,
            tolerance: CONVERGENCE_TOL,
            min_weight: MIN_IRLS_WEIGHT,
            verbose: false,
            nonneg_indices: Vec::new(),
            nonpos_indices: Vec::new(),
            regularization: RegularizationConfig::default(),
            skip_covariance: false,
            standardization: None,
        }
    }
}

impl FitConfig {
    /// Set regularization configuration.
    #[must_use]
    pub fn with_regularization(mut self, reg: RegularizationConfig) -> Self {
        self.regularization = reg;
        self
    }

    /// Set maximum iterations.
    #[must_use]
    pub fn with_max_iterations(mut self, max_iter: usize) -> Self {
        self.max_iterations = max_iter;
        self
    }

    /// Set convergence tolerance.
    #[must_use]
    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    /// Set verbose mode.
    #[must_use]
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Set non-negative coefficient constraints.
    #[must_use]
    pub fn with_nonneg_indices(mut self, indices: Vec<usize>) -> Self {
        self.nonneg_indices = indices;
        self
    }

    /// Set non-positive coefficient constraints.
    #[must_use]
    pub fn with_nonpos_indices(mut self, indices: Vec<usize>) -> Self {
        self.nonpos_indices = indices;
        self
    }

    /// Skip covariance computation (for CV paths).
    #[must_use]
    pub fn with_skip_covariance(mut self, skip: bool) -> Self {
        self.skip_covariance = skip;
        self
    }

    /// Extract an IRLSConfig from this FitConfig (for internal use / backward compat).
    pub fn to_irls_config(&self) -> IRLSConfig {
        IRLSConfig {
            max_iterations: self.max_iterations,
            tolerance: self.tolerance,
            min_weight: self.min_weight,
            verbose: self.verbose,
            nonneg_indices: self.nonneg_indices.clone(),
            nonpos_indices: self.nonpos_indices.clone(),
            skip_covariance: self.skip_covariance,
        }
    }
}

impl From<&IRLSConfig> for FitConfig {
    fn from(irls: &IRLSConfig) -> Self {
        Self {
            max_iterations: irls.max_iterations,
            tolerance: irls.tolerance,
            min_weight: irls.min_weight,
            verbose: irls.verbose,
            nonneg_indices: irls.nonneg_indices.clone(),
            nonpos_indices: irls.nonpos_indices.clone(),
            regularization: RegularizationConfig::default(),
            skip_covariance: false,
            standardization: None,
        }
    }
}

// =============================================================================
// Result Structure
// =============================================================================

/// Results from fitting a GLM using IRLS.
///
/// Contains everything you need for inference and diagnostics.
#[derive(Debug, Clone)]
pub struct IRLSResult {
    /// The fitted coefficients β
    /// These are what you use for predictions: η = Xβ + offset
    pub coefficients: Array1<f64>,

    /// Fitted values μ = g⁻¹(Xβ + offset)
    /// The predicted mean for each observation
    pub fitted_values: Array1<f64>,

    /// Linear predictor η = Xβ + offset
    pub linear_predictor: Array1<f64>,

    /// Final deviance (goodness-of-fit measure)
    /// Lower is better; used for model comparison
    pub deviance: f64,

    /// Number of iterations until convergence
    pub iterations: usize,

    /// Did the algorithm converge?
    pub converged: bool,

    /// The (X'WX)⁻¹ matrix - needed for standard errors
    /// Var(β̂) = φ × (X'WX)⁻¹
    pub covariance_unscaled: Array2<f64>,

    /// Final IRLS weights (useful for diagnostics)
    pub irls_weights: Array1<f64>,

    /// Observation weights (prior weights from user)
    pub prior_weights: Array1<f64>,

    /// Offset used in fitting (if any)
    pub offset: Array1<f64>,

    /// Original response variable (needed for residuals/diagnostics)
    pub y: Array1<f64>,

    /// Family name (needed for computing log-likelihood)
    pub family_name: String,

    /// Penalty applied during fitting (if any)
    pub penalty: Penalty,

    /// Design matrix X (needed for robust standard errors)
    /// Optional to avoid expensive copies for large datasets.
    /// Set to None by default; populated only when needed.
    pub design_matrix: Option<Array2<f64>>,

    /// Warnings collected during fitting (replaces stderr printing).
    pub warnings: Vec<String>,

    /// Whether step-halving produced the accepted step at any iteration.
    /// Set only when a halved step (not the full Newton step) was accepted.
    /// Stays false if every accepted step was the full step, even if the halving
    /// budget was exhausted on a rejected iteration (RS-ACT-007).
    pub step_halving_used: bool,

    /// Terminal solver status: "converged", "max_iterations", or
    /// "step_halving_no_improvement" (no step reduced the deviance, so the
    /// previous iterate was retained). Consumed by the Python inference layer.
    pub solver_status: String,

    /// Optional fine-grained timing profile for performance diagnostics.
    pub profile: Option<IRLSProfile>,
}

#[derive(Debug, Clone, Default)]
pub struct IRLSProfile {
    pub setup_seconds: f64,
    pub init_mu_seconds: f64,
    pub init_projection_seconds: f64,
    pub initial_deviance_seconds: f64,
    pub weight_seconds: f64,
    pub wls_seconds: f64,
    pub wls_gram_seconds: f64,
    pub wls_gram_local_init_seconds: f64,
    pub wls_gram_row_scan_seconds: f64,
    pub wls_gram_pairwise_accum_seconds: f64,
    pub wls_gram_reduce_seconds: f64,
    pub wls_gram_materialize_seconds: f64,
    pub wls_penalty_seconds: f64,
    pub wls_solve_seconds: f64,
    pub update_seconds: f64,
    pub bookkeeping_seconds: f64,
    pub final_extraction_seconds: f64,
    pub final_recompute_seconds: f64,
    pub total_seconds: f64,
}

#[derive(Debug, Clone, Default)]
struct WLSSolveProfile {
    gram_seconds: f64,
    gram_local_init_seconds: f64,
    gram_row_scan_seconds: f64,
    gram_pairwise_accum_seconds: f64,
    gram_reduce_seconds: f64,
    gram_materialize_seconds: f64,
    penalty_seconds: f64,
    solve_seconds: f64,
}

#[derive(Debug, Clone, Default)]
struct GramBuildProfile {
    local_init_seconds: f64,
    row_scan_seconds: f64,
    pairwise_accum_seconds: f64,
    reduce_seconds: f64,
    materialize_seconds: f64,
}

impl GramBuildProfile {
    fn add(&mut self, other: &Self) {
        add_profile_seconds(&mut self.local_init_seconds, other.local_init_seconds);
        add_profile_seconds(&mut self.row_scan_seconds, other.row_scan_seconds);
        add_profile_seconds(
            &mut self.pairwise_accum_seconds,
            other.pairwise_accum_seconds,
        );
        add_profile_seconds(&mut self.reduce_seconds, other.reduce_seconds);
        add_profile_seconds(&mut self.materialize_seconds, other.materialize_seconds);
    }
}

fn profile_gram_subtimers_enabled() -> bool {
    std::env::var_os("RUSTYSTATS_PROFILE_GRAM_SUBTIMERS").is_some()
}

fn add_profile_seconds(total: &mut f64, seconds: f64) {
    *total += seconds;
}

fn should_standardize_regularized_design(config: &FitConfig) -> bool {
    !config.regularization.penalty.is_none() && !config.regularization.penalty.is_smooth()
}

fn centers_are_all_zero(center: &[f64]) -> bool {
    center.iter().all(|&value| value == 0.0)
}

fn should_use_scale_only_standardized_ridge_path(
    l2_penalty: f64,
    requires_coordinate_descent: bool,
    center: &[f64],
) -> bool {
    l2_penalty > 0.0 && !requires_coordinate_descent && centers_are_all_zero(center)
}

fn linear_predictor_with_offset_cached(
    x: ArrayView2<'_, f64>,
    coefficients: &Array1<f64>,
    sparse_cache: Option<&SparseRowCache>,
    offset: &Array1<f64>,
) -> Array1<f64> {
    let eta_base = matrix_vector_dot_cached(x, coefficients, sparse_cache);
    add_offset_to_linear_predictor(&eta_base, offset)
}

fn add_offset_to_linear_predictor(eta_base: &Array1<f64>, offset: &Array1<f64>) -> Array1<f64> {
    eta_base + offset
}

fn eta_without_offset(eta: &Array1<f64>, offset: &Array1<f64>) -> Array1<f64> {
    eta - offset
}

fn coefficients_are_finite(coefficients: &Array1<f64>) -> bool {
    !coefficients_contain_nonfinite(coefficients)
}

fn coefficients_contain_nonfinite(coefficients: &Array1<f64>) -> bool {
    coefficients.iter().any(|&c| c.is_nan() || c.is_infinite())
}

fn finite_coefficients_or_none(coefficients: Array1<f64>) -> Option<Array1<f64>> {
    if coefficients_are_finite(&coefficients) {
        Some(coefficients)
    } else {
        None
    }
}

fn project_coefficients_to_sign_constraints(
    coefficients: &mut Array1<f64>,
    nonneg_indices: &[usize],
    nonpos_indices: &[usize],
) {
    for &idx in nonneg_indices {
        if idx < coefficients.len() && coefficients[idx] < 0.0 {
            coefficients[idx] = 0.0;
        }
    }
    for &idx in nonpos_indices {
        if idx < coefficients.len() && coefficients[idx] > 0.0 {
            coefficients[idx] = 0.0;
        }
    }
}

fn has_sign_constraints(nonneg_indices: &[usize], nonpos_indices: &[usize]) -> bool {
    !nonneg_indices.is_empty() || !nonpos_indices.is_empty()
}

fn should_use_poisson_log_weight_buffers(family_name: &str, link_name: &str) -> bool {
    family_name.eq_ignore_ascii_case("poisson") && link_name == "log"
}

fn trial_state_is_acceptable(
    eta: &Array1<f64>,
    mu: &Array1<f64>,
    objective: f64,
    accept_threshold: f64,
) -> bool {
    eta.iter().all(|v| v.is_finite())
        && mu.iter().all(|v| v.is_finite())
        && objective.is_finite()
        && objective <= accept_threshold
}

fn blend_coefficient(old: f64, new: f64, step_size: f64) -> f64 {
    (1.0 - step_size) * old + step_size * new
}

fn objective_relative_change(objective_old: f64, objective: f64) -> f64 {
    let abs_change = (objective_old - objective).abs();
    if objective_old.abs() > ZERO_TOL {
        abs_change / objective_old.abs()
    } else {
        abs_change
    }
}

fn best_objective_improved(objective: f64, best_objective: f64) -> bool {
    objective < best_objective
}

fn should_stop_after_stale_best(stale_iterations: usize) -> bool {
    stale_iterations >= CONSTRAINED_BEST_EARLY_STOP_PATIENCE
}

fn should_use_constrained_best(
    has_constraints: bool,
    best_objective: f64,
    objective: f64,
    constrained_best_plateau: bool,
) -> bool {
    has_constraints
        && (best_objective < objective || (constrained_best_plateau && best_objective.is_finite()))
}

fn combine_prior_and_irls_weights(
    prior_weights: &Array1<f64>,
    irls_weights: &Array1<f64>,
) -> Array1<f64> {
    prior_weights
        .iter()
        .zip(irls_weights.iter())
        .map(|(&pw, &iw)| pw * iw)
        .collect()
}

fn final_extraction_deviance_acceptable(dev_check: f64, final_deviance: f64) -> bool {
    dev_check.is_finite() && dev_check <= final_deviance
}

fn should_recompute_final_state(skip_covariance: bool) -> bool {
    !skip_covariance
}

fn should_warn_nonconverged(converged: bool) -> bool {
    !converged
}

// =============================================================================
// Unified Fitting Entry Point
// =============================================================================

/// Fit a GLM using the unified configuration.
///
/// This is the **single entry point** for all GLM fitting. It automatically
/// dispatches to the appropriate solver based on the regularization config:
/// - No penalty or Ridge (L2) → IRLS with optional penalty on diagonal
/// - Lasso (L1) or Elastic Net → coordinate descent
///
/// This replaces the need to choose between `fit_glm`, `fit_glm_full`,
/// `fit_glm_warm_start`, `fit_glm_regularized`, `fit_glm_regularized_warm`,
/// and `fit_glm_coordinate_descent`.
///
/// # Arguments
/// * `y` - Response variable (n × 1)
/// * `x` - Design matrix (n × p), should include intercept column if desired
/// * `family` - Distribution family (Gaussian, Poisson, Binomial, Gamma, etc.)
/// * `link` - Link function (Identity, Log, Logit, etc.)
/// * `config` - Unified fitting configuration (algorithm settings + regularization)
/// * `offset` - Optional offset term (e.g., log(exposure) for rate models)
/// * `weights` - Optional prior weights for each observation
/// * `init_coefficients` - Optional initial coefficients for warm starting
///
/// # Returns
/// * `Ok(IRLSResult)` - Fitted model results
/// * `Err(RustyStatsError)` - If fitting fails
pub fn fit_glm_unified(
    y: &Array1<f64>,
    x: ArrayView2<'_, f64>,
    family: &dyn Family,
    link: &dyn Link,
    config: &FitConfig,
    offset: Option<&Array1<f64>>,
    weights: Option<&Array1<f64>>,
    init_coefficients: Option<&Array1<f64>>,
) -> Result<IRLSResult> {
    fit_glm_unified_with_optional_sparse_cache(
        y,
        x,
        family,
        link,
        config,
        offset,
        weights,
        init_coefficients,
        None,
    )
}

/// Fit a GLM using a caller-supplied sparse row cache when it matches `x`.
///
/// This is used by regularization-path CV, where many alpha fits reuse the same
/// fold design matrix. Reusing the packed sparse rows avoids repeatedly scanning
/// and packing an unchanged wide design matrix.
pub fn fit_glm_unified_with_sparse_cache(
    y: &Array1<f64>,
    x: ArrayView2<'_, f64>,
    family: &dyn Family,
    link: &dyn Link,
    config: &FitConfig,
    offset: Option<&Array1<f64>>,
    weights: Option<&Array1<f64>>,
    init_coefficients: Option<&Array1<f64>>,
    sparse_cache: Option<&SparseRowCache>,
) -> Result<IRLSResult> {
    fit_glm_unified_with_optional_sparse_cache(
        y,
        x,
        family,
        link,
        config,
        offset,
        weights,
        init_coefficients,
        sparse_cache,
    )
}

fn fit_glm_unified_with_optional_sparse_cache(
    y: &Array1<f64>,
    x: ArrayView2<'_, f64>,
    family: &dyn Family,
    link: &dyn Link,
    config: &FitConfig,
    offset: Option<&Array1<f64>>,
    weights: Option<&Array1<f64>>,
    init_coefficients: Option<&Array1<f64>>,
    sparse_cache: Option<&SparseRowCache>,
) -> Result<IRLSResult> {
    if let Some(standardization) = &config.standardization {
        if should_standardize_regularized_design(config) {
            standardization.validate(x.ncols())?;
            let l2_penalty = config.regularization.penalty.l2_penalty();
            if should_use_scale_only_standardized_ridge_path(
                l2_penalty,
                config.regularization.penalty.requires_coordinate_descent(),
                &standardization.center,
            ) {
                let l2_penalty_factors: Vec<f64> = standardization
                    .scale
                    .iter()
                    .map(|scale| scale * scale)
                    .collect();
                let penalize_intercept = !config.regularization.fit_intercept;
                return fit_glm_core(
                    y,
                    x,
                    family,
                    link,
                    &config.to_irls_config(),
                    offset,
                    weights,
                    init_coefficients,
                    l2_penalty,
                    Some(&l2_penalty_factors),
                    penalize_intercept,
                    config.regularization.penalty.clone(),
                    sparse_cache,
                );
            }
            let x_work = standardization.standardize_matrix(x)?;
            let init_work = match init_coefficients {
                Some(init) => Some(
                    standardization
                        .to_standardized_coefficients(init, config.regularization.fit_intercept)?,
                ),
                None => None,
            };
            let mut work_config = config.clone();
            work_config.standardization = None;

            let mut result = fit_glm_unified_with_optional_sparse_cache(
                y,
                x_work.view(),
                family,
                link,
                &work_config,
                offset,
                weights,
                init_work.as_ref(),
                None,
            )?;
            result.coefficients = standardization.to_original_coefficients(
                &result.coefficients,
                config.regularization.fit_intercept,
            )?;
            if !config.skip_covariance {
                result.covariance_unscaled = standardization.to_original_covariance(
                    &result.covariance_unscaled,
                    config.regularization.fit_intercept,
                )?;
            }
            return Ok(result);
        }
    }

    if config.regularization.penalty.requires_coordinate_descent() {
        // L1 or Elastic Net → coordinate descent solver
        use super::coordinate_descent::fit_glm_coordinate_descent;
        fit_glm_coordinate_descent(
            y,
            x,
            family,
            link,
            &config.to_irls_config(),
            &config.regularization,
            offset,
            weights,
            init_coefficients,
            config.skip_covariance,
        )
    } else {
        // No penalty or pure L2 → standard IRLS
        let l2_penalty = config.regularization.penalty.l2_penalty();
        let penalize_intercept = !config.regularization.fit_intercept;
        fit_glm_core(
            y,
            x,
            family,
            link,
            &config.to_irls_config(),
            offset,
            weights,
            init_coefficients,
            l2_penalty,
            None,
            penalize_intercept,
            config.regularization.penalty.clone(),
            sparse_cache,
        )
    }
}

/// Core IRLS fitting function with optional warm start and optional L2 penalty.
///
/// This is the internal implementation called by `fit_glm_unified`.
/// When `init_coefficients` is provided, initialization starts from those
/// coefficients instead of the family's default. When `l2_penalty > 0`,
/// Ridge regularization is applied: (X'WX + λI)β = X'Wz.
fn fit_glm_core(
    y: &Array1<f64>,
    x: ArrayView2<'_, f64>,
    family: &dyn Family,
    link: &dyn Link,
    config: &IRLSConfig,
    offset: Option<&Array1<f64>>,
    weights: Option<&Array1<f64>>,
    init_coefficients: Option<&Array1<f64>>,
    l2_penalty: f64,
    l2_penalty_factors: Option<&[f64]>,
    penalize_intercept: bool,
    penalty: Penalty,
    provided_sparse_cache: Option<&SparseRowCache>,
) -> Result<IRLSResult> {
    let profile_total_start = Instant::now();
    let mut profile = IRLSProfile::default();
    // -------------------------------------------------------------------------
    // Step 0: Validate inputs and set up offset/weights
    // -------------------------------------------------------------------------
    let profile_setup_start = Instant::now();
    let n = y.len();
    let p = x.ncols();
    let validated = validate_glm_inputs(y, x, offset, weights)?;
    let offset_vec = validated.offset;
    let prior_weights_vec = validated.prior_weights;
    let mut warnings: Vec<String> = Vec::new();
    let provided_sparse_cache = provided_sparse_cache.filter(|cache| cache.is_compatible_with(x));
    let owned_sparse_cache = if provided_sparse_cache.is_none() {
        build_sparse_row_cache_if_beneficial(x)
    } else {
        None
    };
    let sparse_cache = provided_sparse_cache.or(owned_sparse_cache.as_ref());

    let mut iter_coefficients = Array1::zeros(p);
    profile.setup_seconds = profile_setup_start.elapsed().as_secs_f64();

    // -------------------------------------------------------------------------
    // Step 1: Initialize μ (from coefficients if warm-starting, else from family)
    // -------------------------------------------------------------------------
    let profile_init_mu_start = Instant::now();
    let mut mu = if let Some(init) = init_coefficients {
        if init.len() != p {
            return Err(RustyStatsError::dim_mismatch(
                p,
                init.len(),
                "init_coefficients length vs X columns",
            ));
        }
        iter_coefficients = init.clone();
        let eta_init = linear_predictor_with_offset_cached(x, init, sparse_cache, &offset_vec);
        let mu_init = link.inverse(&eta_init);
        family.clamp_mu(&mu_init)
    } else {
        let mu_init = family.initialize_mu(y);
        // Ensure μ is valid (e.g., positive for Poisson, in (0,1) for Binomial)
        if !family.is_valid_mu(&mu_init) {
            warnings.push(format!(
                "Family '{}' initial μ values were invalid. Using safe fallback initialization. \
                This may indicate unusual response values (e.g., all zeros, extreme values).",
                family.name()
            ));
            initialize_mu_safe(y, family)
        } else {
            mu_init
        }
    };
    profile.init_mu_seconds = profile_init_mu_start.elapsed().as_secs_f64();

    // -------------------------------------------------------------------------
    // Step 2: Initialize linear predictor η = Xβ + offset
    // -------------------------------------------------------------------------
    // Family initializers often produce a near-saturated μ that is not exactly
    // representable by the model matrix. Project it into coefficient space first
    // so the retained iterate, step-halving baseline, and returned coefficients
    // all describe the same fitted state (RS-ACT-007).
    let mut eta = link.link(&mu);
    if init_coefficients.is_none() {
        let profile_init_projection_start = Instant::now();
        let eta_no_offset = eta_without_offset(&eta, &offset_vec);
        match solve_weighted_least_squares_penalized(
            x,
            &eta_no_offset,
            &prior_weights_vec,
            l2_penalty,
            l2_penalty_factors,
            penalize_intercept,
            true, // init projection covariance is discarded
            sparse_cache,
            None,
        ) {
            Ok((coef, _, _)) => {
                if let Some(mut coef) = finite_coefficients_or_none(coef) {
                    project_coefficients_to_sign_constraints(
                        &mut coef,
                        &config.nonneg_indices,
                        &config.nonpos_indices,
                    );
                    iter_coefficients = coef;
                } else {
                    warnings.push(
                        "Initial coefficient projection failed. Starting IRLS from zero coefficients."
                            .to_string(),
                    );
                }
            }
            _ => {
                warnings.push(
                    "Initial coefficient projection failed. Starting IRLS from zero coefficients."
                        .to_string(),
                );
            }
        }
        eta = linear_predictor_with_offset_cached(x, &iter_coefficients, sparse_cache, &offset_vec);
        mu = family.clamp_mu(&link.inverse(&eta));
        profile.init_projection_seconds = profile_init_projection_start.elapsed().as_secs_f64();
    }

    // -------------------------------------------------------------------------
    // Step 3: Calculate initial deviance
    // -------------------------------------------------------------------------
    let profile_initial_deviance_start = Instant::now();
    let mut deviance = family.deviance(y, &mu, Some(&prior_weights_vec));
    let mut objective = penalized_irls_objective(
        deviance,
        &iter_coefficients,
        l2_penalty,
        l2_penalty_factors,
        penalize_intercept,
    );
    let mut objective_old: f64;
    profile.initial_deviance_seconds = profile_initial_deviance_start.elapsed().as_secs_f64();

    // -------------------------------------------------------------------------
    // Step 4: IRLS iteration loop
    // -------------------------------------------------------------------------
    let mut converged = false;
    let mut iteration = 0;
    let mut step_halving_used = false;
    // Set when no full or halved step reduces the deviance: the previous iterate
    // is retained rather than accepting a worse step (RS-ACT-007).
    let mut step_halving_failed = false;

    // Store the final accepted IRLS weights. Covariance is computed once after
    // the final accepted state is known, not on every iteration.
    let mut final_weights = Array1::zeros(n);

    // For constrained problems, track best solution seen (projection can make
    // either deviance or the penalized objective increase).
    let has_constraints = has_sign_constraints(&config.nonneg_indices, &config.nonpos_indices);
    let mut best_objective = f64::INFINITY;
    let mut best_deviance = f64::INFINITY;
    let mut best_coefficients = iter_coefficients.clone();
    let mut best_mu = mu.clone();
    let mut best_eta = eta.clone();
    let mut best_weights = final_weights.clone();
    let mut constrained_best_stale_iterations = 0usize;
    let mut constrained_best_plateau = false;
    let use_poisson_log_weight_buffers =
        should_use_poisson_log_weight_buffers(family.name(), link.name());
    let weight_buffer_len = if use_poisson_log_weight_buffers { n } else { 0 };
    let mut irls_weights_buf = Array1::zeros(weight_buffer_len);
    let mut combined_weights_buf = Array1::zeros(weight_buffer_len);
    let mut working_response_buf = Array1::zeros(weight_buffer_len);

    while iteration < config.max_iterations {
        iteration += 1;

        // ---------------------------------------------------------------------
        // Step 4a: Compute working weights W
        // ---------------------------------------------------------------------
        let profile_weight_start = Instant::now();
        let weight_result_storage = if use_poisson_log_weight_buffers {
            compute_poisson_log_irls_weights_in_place(
                y,
                &mu,
                &eta,
                &offset_vec,
                &prior_weights_vec,
                config.min_weight,
                &mut irls_weights_buf,
                &mut combined_weights_buf,
                &mut working_response_buf,
            )?;
            None
        } else {
            Some(compute_irls_weights(
                y,
                &mu,
                &eta,
                &offset_vec,
                &prior_weights_vec,
                family,
                link,
                config.min_weight,
            )?)
        };

        let (irls_weights, combined_weights, working_response) =
            if let Some(weight_result) = weight_result_storage.as_ref() {
                (
                    &weight_result.irls_weights,
                    &weight_result.combined_weights,
                    &weight_result.working_response,
                )
            } else {
                (
                    &irls_weights_buf,
                    &combined_weights_buf,
                    &working_response_buf,
                )
            };
        add_profile_seconds(
            &mut profile.weight_seconds,
            profile_weight_start.elapsed().as_secs_f64(),
        );

        // ---------------------------------------------------------------------
        // Step 4c: Solve weighted least squares: (X'WX)β = X'Wz
        // ---------------------------------------------------------------------
        // This is the core linear algebra step.
        // We're finding β that minimizes: Σ w_i (z_i - x_i'β)²
        // ---------------------------------------------------------------------
        let profile_wls_start = Instant::now();
        let (mut new_coefficients, _, wls_profile) = solve_weighted_least_squares_penalized(
            x,
            working_response,
            combined_weights,
            l2_penalty,
            l2_penalty_factors,
            penalize_intercept,
            true, // per-iteration inverse is unused; compute final covariance once
            sparse_cache,
            Some(&iter_coefficients),
        )?;
        add_profile_seconds(
            &mut profile.wls_seconds,
            profile_wls_start.elapsed().as_secs_f64(),
        );
        add_profile_seconds(&mut profile.wls_gram_seconds, wls_profile.gram_seconds);
        add_profile_seconds(
            &mut profile.wls_gram_local_init_seconds,
            wls_profile.gram_local_init_seconds,
        );
        add_profile_seconds(
            &mut profile.wls_gram_row_scan_seconds,
            wls_profile.gram_row_scan_seconds,
        );
        add_profile_seconds(
            &mut profile.wls_gram_pairwise_accum_seconds,
            wls_profile.gram_pairwise_accum_seconds,
        );
        add_profile_seconds(
            &mut profile.wls_gram_reduce_seconds,
            wls_profile.gram_reduce_seconds,
        );
        add_profile_seconds(
            &mut profile.wls_gram_materialize_seconds,
            wls_profile.gram_materialize_seconds,
        );
        add_profile_seconds(
            &mut profile.wls_penalty_seconds,
            wls_profile.penalty_seconds,
        );
        add_profile_seconds(&mut profile.wls_solve_seconds, wls_profile.solve_seconds);
        let profile_update_start = Instant::now();

        // Check for NaN in coefficients - indicates numerical instability
        if coefficients_contain_nonfinite(&new_coefficients) {
            return Err(RustyStatsError::NumericalError(
                "IRLS produced NaN or infinite coefficients. This usually indicates: \
                 (1) severe multicollinearity in predictors, \
                 (2) extreme scale differences between variables, or \
                 (3) separation in binary response data. \
                 Try standardizing continuous predictors or removing correlated terms."
                    .to_string(),
            ));
        }

        // ---------------------------------------------------------------------
        // Step 4c.1: Apply coefficient sign constraints
        // ---------------------------------------------------------------------
        project_coefficients_to_sign_constraints(
            &mut new_coefficients,
            &config.nonneg_indices,
            &config.nonpos_indices,
        );

        // ---------------------------------------------------------------------
        // Step 4d: Update η and μ with step-halving for stability
        // ---------------------------------------------------------------------
        // If the optimization objective increases, reduce step size to prevent
        // oscillation. For unregularized fits this is just deviance; for ridge
        // it is deviance + lambda * ||beta||^2 on the penalized subset.
        // For constrained problems, we blend coefficients (not eta) and re-apply
        // projection to ensure constraints are satisfied at each step.
        // ---------------------------------------------------------------------
        objective_old = objective;

        // Acceptance threshold: a step may not worsen the objective by more than a
        // tiny relative tolerance (RS-ACT-007). Coefficient blending is used for
        // both paths; for the unconstrained path it is algebraically identical to
        // blending eta, but keeps coefficients and (eta, mu) consistent.
        let accept_threshold = objective_old * IRLS_ACCEPT_REL_SLACK;

        // Try the full Newton step (with constraints projected).
        let mut trial_coefficients = new_coefficients.clone();
        project_coefficients_to_sign_constraints(
            &mut trial_coefficients,
            &config.nonneg_indices,
            &config.nonpos_indices,
        );
        let mut eta_new =
            linear_predictor_with_offset_cached(x, &trial_coefficients, sparse_cache, &offset_vec);
        let mut mu_new = family.clamp_mu(&link.inverse(&eta_new));
        let mut deviance_new = family.deviance(y, &mu_new, Some(&prior_weights_vec));
        let mut objective_new = penalized_irls_objective(
            deviance_new,
            &trial_coefficients,
            l2_penalty,
            l2_penalty_factors,
            penalize_intercept,
        );

        let mut step_accepted =
            trial_state_is_acceptable(&eta_new, &mu_new, objective_new, accept_threshold);

        // Step-halving: if the full step worsened the objective, try smaller
        // steps and accept the first one that meets the threshold. The
        // `step_halving_used` flag is set only when a halved step is the one
        // ultimately accepted (RS-ACT-007), not merely when halving was attempted.
        if !step_accepted {
            let mut step_size = 0.5;
            for _half_step in 0..IRLS_MAX_HALF_STEPS {
                let mut blended: Array1<f64> = iter_coefficients
                    .iter()
                    .zip(new_coefficients.iter())
                    .map(|(&old, &new)| blend_coefficient(old, new, step_size))
                    .collect();
                project_coefficients_to_sign_constraints(
                    &mut blended,
                    &config.nonneg_indices,
                    &config.nonpos_indices,
                );
                let e = linear_predictor_with_offset_cached(x, &blended, sparse_cache, &offset_vec);
                let m = family.clamp_mu(&link.inverse(&e));
                let d = family.deviance(y, &m, Some(&prior_weights_vec));
                let o = penalized_irls_objective(
                    d,
                    &blended,
                    l2_penalty,
                    l2_penalty_factors,
                    penalize_intercept,
                );
                if trial_state_is_acceptable(&e, &m, o, accept_threshold) {
                    trial_coefficients = blended;
                    eta_new = e;
                    mu_new = m;
                    deviance_new = d;
                    objective_new = o;
                    step_accepted = true;
                    step_halving_used = true;
                    break;
                }
                step_size *= 0.5;
            }
        }

        if !step_accepted {
            // No full or halved step reduced the objective: retain the previous
            // iterate (already held in iter_coefficients / eta / mu / deviance /
            // objective)
            // instead of accepting a worse one, and stop (RS-ACT-007).
            // `iteration` was incremented at the top before this (rejected) step
            // was attempted; report the index of the RETAINED iterate instead,
            // consistent with the converged / max_iterations count.
            step_halving_failed = true;
            iteration = iteration.saturating_sub(1);
            final_weights.assign(irls_weights);
            add_profile_seconds(
                &mut profile.update_seconds,
                profile_update_start.elapsed().as_secs_f64(),
            );
            break;
        }

        new_coefficients = trial_coefficients;
        eta = eta_new;
        mu = mu_new;
        deviance = deviance_new;
        objective = objective_new;

        // Relative change in the same objective used for acceptance.
        let rel_change = objective_relative_change(objective_old, objective);

        if config.verbose {
            eprintln!(
                "Iteration {}: deviance = {:.6}, objective = {:.6}, rel_change = {:.2e}",
                iteration, deviance, objective, rel_change
            );
        }

        // Store coefficients from this iteration
        iter_coefficients = new_coefficients;

        // For constrained problems, track the best solution seen. Projection can
        // leave subsequent accepted IRLS steps worse than the best projected
        // iterate; after a short stale window, adopt the best now rather than
        // spending the remaining iteration budget only to recover it at the end.
        if has_constraints {
            if best_objective_improved(objective, best_objective) {
                best_objective = objective;
                best_deviance = deviance;
                best_coefficients = iter_coefficients.clone();
                best_mu = mu.clone();
                best_eta = eta.clone();
                best_weights = irls_weights.clone();
                constrained_best_stale_iterations = 0;
            } else if best_objective.is_finite() {
                constrained_best_stale_iterations += 1;
                if should_stop_after_stale_best(constrained_best_stale_iterations) {
                    constrained_best_plateau = true;
                    final_weights.assign(irls_weights);
                    add_profile_seconds(
                        &mut profile.update_seconds,
                        profile_update_start.elapsed().as_secs_f64(),
                    );
                    break;
                }
            }
        }

        // Convergence requires the accepted step to be non-worsening AND small.
        // A clearly-worse step is not converged, but the non-worsening guard is
        // RELATIVE — mirroring the accept loop's `IRLS_ACCEPT_REL_SLACK`. A
        // terminal step that nudges the objective up by a tiny relative amount
        // (still below `config.tolerance` relative) is treated as converged,
        // matching the acceptance contract rather than an absolute 1e-10 floor.
        if irls_step_converged(objective_old, objective, rel_change, config.tolerance) {
            converged = true;
            final_weights.assign(irls_weights);
            add_profile_seconds(
                &mut profile.update_seconds,
                profile_update_start.elapsed().as_secs_f64(),
            );
            break;
        }

        // Store for final iteration
        final_weights.assign(irls_weights);
        add_profile_seconds(
            &mut profile.update_seconds,
            profile_update_start.elapsed().as_secs_f64(),
        );
    }

    // -------------------------------------------------------------------------
    // Step 5: Extract final coefficients
    // -------------------------------------------------------------------------
    // For constrained problems, use the best solution found during iteration
    // (objective can increase due to projection, so last iteration may not be best)
    let (mut final_mu, mut final_eta, mut final_deviance, use_coefficients) =
        if should_use_constrained_best(
            has_constraints,
            best_objective,
            objective,
            constrained_best_plateau,
        ) {
            // Best solution was found earlier — use it and treat as converged
            // (sign clamping can cause coefficient oscillation even when deviance
            // has stabilized, so the best-tracked solution is the correct answer).
            // Clear the step-halving-failure flag so the reported solver_status
            // is consistent with converged = true (the best iterate is adopted,
            // not the worse step that triggered the halving break).
            converged = true;
            step_halving_failed = false;
            final_weights = best_weights;
            (best_mu, best_eta, best_deviance, best_coefficients)
        } else {
            (mu, eta, deviance, iter_coefficients)
        };

    // Try final coefficient extraction, but fall back to iteration coefficients
    // if it produces NaN. Non-CV fits compute covariance here once, after the
    // final accepted weights are known, instead of in every IRLS iteration.
    let profile_final_extraction_start = Instant::now();
    let (final_coefficients, cov_unscaled) = if config.skip_covariance {
        (use_coefficients, Array2::zeros((p, p)))
    } else {
        // Compute working response accounting for offset
        let eta_no_offset = eta_without_offset(&final_eta, &offset_vec);

        // Combine prior weights with final IRLS weights
        let combined_final_weights =
            combine_prior_and_irls_weights(&prior_weights_vec, &final_weights);

        let final_working_response = compute_working_response(y, &final_mu, &eta_no_offset, link);

        match solve_weighted_least_squares_penalized(
            x,
            &final_working_response,
            &combined_final_weights,
            l2_penalty,
            l2_penalty_factors,
            penalize_intercept,
            false,
            sparse_cache,
            Some(&use_coefficients),
        ) {
            Ok((coef, cov, _)) => {
                let cov = if cov.iter().all(|v| v.is_finite()) {
                    cov
                } else {
                    return Err(RustyStatsError::NumericalError(
                        "Final covariance extraction produced NaN/Inf. \
                        This usually indicates numerical instability or a nearly singular design matrix."
                            .to_string(),
                    ));
                };
                if let Some(coef) = finite_coefficients_or_none(coef) {
                    // For constrained problems, apply projection and check if it's better than stored best
                    if has_constraints {
                        let mut proj_coef = coef;
                        project_coefficients_to_sign_constraints(
                            &mut proj_coef,
                            &config.nonneg_indices,
                            &config.nonpos_indices,
                        );
                        // Check if this extraction is better
                        let eta_check = matrix_vector_dot_cached(x, &proj_coef, sparse_cache);
                        let eta_full = add_offset_to_linear_predictor(&eta_check, &offset_vec);
                        let mu_check = family.clamp_mu(&link.inverse(&eta_full));
                        let dev_check = family.deviance(y, &mu_check, Some(&prior_weights_vec));
                        if final_extraction_deviance_acceptable(dev_check, final_deviance) {
                            (proj_coef, cov)
                        } else {
                            (use_coefficients, cov)
                        }
                    } else {
                        // Unconstrained: guard against a final extraction that is worse
                        // than the loop's retained iterate (RS-ACT-007).
                        let eta_full = linear_predictor_with_offset_cached(
                            x,
                            &coef,
                            sparse_cache,
                            &offset_vec,
                        );
                        let mu_check = family.clamp_mu(&link.inverse(&eta_full));
                        let dev_check = family.deviance(y, &mu_check, Some(&prior_weights_vec));
                        if final_extraction_deviance_acceptable(dev_check, final_deviance) {
                            (coef, cov)
                        } else {
                            (use_coefficients, cov)
                        }
                    }
                } else {
                    warnings.push(
                        "Final coefficient extraction produced NaN/Inf. \
                        Using coefficients from best iteration instead. This may indicate numerical instability."
                            .to_string(),
                    );
                    if coefficients_contain_nonfinite(&use_coefficients) {
                        return Err(RustyStatsError::NumericalError(
                            "IRLS produced NaN or infinite coefficients. This usually indicates: \
                             (1) severe multicollinearity in predictors, \
                             (2) extreme scale differences between variables, or \
                             (3) separation in binary response data. \
                             Try standardizing continuous predictors or removing correlated terms."
                                .to_string(),
                        ));
                    }
                    (use_coefficients, cov)
                }
            }
            Err(_) => {
                // Fail closed (pre-v0.8.14 contract): a failed final
                // covariance solve must not return a "successful" fit with an
                // all-zeros covariance — zero standard errors read as infinite
                // significance downstream. Callers that do not need inference
                // set skip_covariance and never reach this solve.
                return Err(RustyStatsError::LinearAlgebraError(
                    "Final coefficient/covariance extraction failed. \
                    This often indicates multicollinearity in predictors."
                        .to_string(),
                ));
            }
        }
    };
    profile.final_extraction_seconds = profile_final_extraction_start.elapsed().as_secs_f64();

    let final_coefficients = final_coefficients;

    // CV folds skip covariance and keep the accepted iterate coefficients, so
    // the retained final state above is already consistent. Avoid another
    // full training-set matrix-vector pass for every alpha/fold.
    if should_recompute_final_state(config.skip_covariance) {
        let profile_final_recompute_start = Instant::now();
        final_eta =
            linear_predictor_with_offset_cached(x, &final_coefficients, sparse_cache, &offset_vec);
        final_mu = family.clamp_mu(&link.inverse(&final_eta));
        final_deviance = family.deviance(y, &final_mu, Some(&prior_weights_vec));
        profile.final_recompute_seconds = profile_final_recompute_start.elapsed().as_secs_f64();
    }
    profile.total_seconds = profile_total_start.elapsed().as_secs_f64();

    let solver_status = if step_halving_failed {
        "step_halving_no_improvement"
    } else if converged {
        "converged"
    } else {
        "max_iterations"
    };
    if should_warn_nonconverged(converged) {
        warnings.push(format!(
            "IRLS did not converge (status: {solver_status}). Results may be \
             unreliable; consider increasing max_iter, loosening tol, or rescaling \
             predictors."
        ));
    }

    Ok(IRLSResult {
        coefficients: final_coefficients,
        fitted_values: final_mu,
        linear_predictor: final_eta,
        deviance: final_deviance,
        iterations: iteration,
        converged,
        covariance_unscaled: cov_unscaled,
        irls_weights: final_weights,
        prior_weights: prior_weights_vec,
        offset: offset_vec,
        y: y.to_owned(), // Only clone at the end, needed for diagnostics
        family_name: family.name().to_string(),
        penalty,
        design_matrix: None, // Computed lazily in Python layer to avoid expensive copy
        warnings,
        step_halving_used,
        solver_status: solver_status.to_string(),
        profile: Some(profile),
    })
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Decide whether an accepted IRLS step satisfies the convergence criterion.
///
/// Two conditions must both hold (RS-ACT-007, A2):
/// * the relative objective/metric change is small: `rel_change < tolerance`;
/// * the step is non-worsening up to a RELATIVE slack mirroring the accept loop:
///   `metric <= metric_old * (1 + tolerance)`, i.e.
///   `signed_change >= -(tolerance * |metric_old|)`, floored at `ZERO_TOL` so a
///   near-zero metric still has a tiny absolute allowance.
///
/// Using a relative (not absolute 1e-10) non-worsening guard means a genuinely
/// converged large-metric fit whose terminal step nudges the metric up by a tiny
/// relative amount is correctly flagged converged instead of running to
/// `max_iterations` with a spurious warning.
#[inline]
fn irls_step_converged(metric_old: f64, metric: f64, rel_change: f64, tolerance: f64) -> bool {
    let signed_change = metric_old - metric;
    let non_worsening_threshold = -(tolerance * metric_old.abs()).max(ZERO_TOL);
    signed_change >= non_worsening_threshold && rel_change < tolerance
}

#[inline]
fn penalized_irls_objective(
    deviance: f64,
    coefficients: &Array1<f64>,
    l2_penalty: f64,
    l2_penalty_factors: Option<&[f64]>,
    penalize_intercept: bool,
) -> f64 {
    if l2_penalty <= 0.0 {
        return deviance;
    }
    let start_idx = if penalize_intercept { 0 } else { 1 };
    deviance
        + l2_penalty
            * coefficients
                .iter()
                .enumerate()
                .skip(start_idx)
                .map(|(idx, coef)| {
                    let factor = l2_penalty_factors.map_or(1.0, |factors| factors[idx]);
                    factor * coef * coef
                })
                .sum::<f64>()
}

/// Compute X'WX and X'Wz using parallel chunked computation with raw slice access.
///
/// This is the hot-path inner loop shared by all WLS solvers.
/// Returns (X'WX as DMatrix, X'Wz as DVector) in nalgebra types.
///
/// OPTIMIZATION NOTES:
/// - Uses raw slice access with unsafe for maximum throughput
/// - Parallel chunked reduction via Rayon to utilize all cores
/// - Only computes upper triangle of X'WX (symmetric)
///
/// // SAFETY: All unsafe accesses are within bounds because:
/// - k ranges from 0 to n-1, and w_slice/z_slice have length n
/// - row_start + j = k*p + j where k < n and j < p, so max index is (n-1)*p + (p-1) < n*p = x_slice.len()
/// - i, j range over the upper triangle, and xtx_local has length p*(p+1)/2
#[inline]
pub fn compute_xtwx_xtwz(
    x: ArrayView2<'_, f64>,
    z: &Array1<f64>,
    w: &Array1<f64>,
) -> Result<(DMatrix<f64>, DVector<f64>)> {
    let (xtx, xtz, _) = compute_xtwx_xtwz_profiled(x, z, w)?;
    Ok((xtx, xtz))
}

fn compute_xtwx_xtwz_profiled(
    x: ArrayView2<'_, f64>,
    z: &Array1<f64>,
    w: &Array1<f64>,
) -> Result<(DMatrix<f64>, DVector<f64>, GramBuildProfile)> {
    let n = x.nrows();
    let p = x.ncols();

    let x_slice = match x.as_slice() {
        Some(s) => s,
        None => {
            return Err(RustyStatsError::LinearAlgebraError(
                "Design matrix X must be contiguous in memory (C-order)".to_string(),
            ));
        }
    };
    let w_slice = match w.as_slice() {
        Some(s) => s,
        None => {
            return Err(RustyStatsError::LinearAlgebraError(
                "Weight vector W must be contiguous in memory".to_string(),
            ));
        }
    };
    let z_slice = match z.as_slice() {
        Some(s) => s,
        None => {
            return Err(RustyStatsError::LinearAlgebraError(
                "Working response Z must be contiguous in memory".to_string(),
            ));
        }
    };

    // SAFETY: Bounds verification for all unsafe accesses below.
    // x_slice has length n*p, w_slice and z_slice have length n.
    // All accesses use indices < n*p, < n, and < p*(p+1)/2 respectively.
    assert_eq!(x_slice.len(), n * p, "x_slice length must be n*p");
    assert_eq!(w_slice.len(), n, "w_slice length must be n");
    assert_eq!(z_slice.len(), n, "z_slice length must be n");

    let collect_profile = profile_gram_subtimers_enabled();
    if should_use_sparse_xtwx_kernel(x_slice, n, p) {
        return compute_xtwx_xtwz_sparse(x_slice, z_slice, w_slice, n, p, collect_profile);
    }

    // Core-adaptive chunking: split rows across all available threads so this
    // kernel (the dominant per-iteration cost) saturates the CPU, instead of
    // the ~4 chunks the old fixed 8192 produced for a 25k-row CV fold.
    let chunk_size = n.div_ceil(rayon::current_num_threads()).max(1);
    let num_chunks = n.div_ceil(chunk_size);

    let upper_len = packed_upper_len(p);
    let (xtx_data, xtz_data, mut gram_profile): (Vec<f64>, Vec<f64>, GramBuildProfile) = (0
        ..num_chunks)
        .into_par_iter()
        .map(|chunk_idx| {
            let chunk_start = chunk_idx * chunk_size;
            let chunk_end = (chunk_start + chunk_size).min(n);
            let mut profile = GramBuildProfile::default();
            let local_init_start = collect_profile.then(Instant::now);
            let mut xtx_local = vec![0.0; upper_len];
            let mut xtz_local = vec![0.0; p];
            if let Some(start) = local_init_start {
                profile.local_init_seconds = start.elapsed().as_secs_f64();
            }

            for k in chunk_start..chunk_end {
                let pairwise_start = collect_profile.then(Instant::now);
                // SAFETY: k < n, so k < w_slice.len() and k < z_slice.len()
                let wk = unsafe { *w_slice.get_unchecked(k) };
                // SAFETY: k < n, so k < z_slice.len().
                let zk = unsafe { *z_slice.get_unchecked(k) };
                let wz = wk * zk;
                let row_start = k * p;
                let mut packed_idx = 0;

                for i in 0..p {
                    // SAFETY: row_start + i = k*p + i where k < n and i < p,
                    // so row_start + i < n*p = x_slice.len()
                    let xki = unsafe { *x_slice.get_unchecked(row_start + i) };
                    let xki_w = xki * wk;
                    // SAFETY: i < p = xtz_local.len()
                    unsafe { *xtz_local.get_unchecked_mut(i) += xki * wz };

                    for j in i..p {
                        // SAFETY: row_start + j < n*p = x_slice.len()
                        let xkj = unsafe { *x_slice.get_unchecked(row_start + j) };
                        // SAFETY: packed_idx advances exactly once for each
                        // upper-triangle entry, so it stays within upper_len.
                        unsafe { *xtx_local.get_unchecked_mut(packed_idx) += xki_w * xkj };
                        packed_idx += 1;
                    }
                }
                if let Some(start) = pairwise_start {
                    add_profile_seconds(
                        &mut profile.pairwise_accum_seconds,
                        start.elapsed().as_secs_f64(),
                    );
                }
            }
            (xtx_local, xtz_local, profile)
        })
        .reduce_with(
            |(mut a_xtx, mut a_xtz, mut a_profile), (b_xtx, b_xtz, b_profile)| {
                let reduce_start = collect_profile.then(Instant::now);
                a_profile.add(&b_profile);
                for i in 0..a_xtx.len() {
                    a_xtx[i] += b_xtx[i];
                }
                for i in 0..a_xtz.len() {
                    a_xtz[i] += b_xtz[i];
                }
                if let Some(start) = reduce_start {
                    add_profile_seconds(
                        &mut a_profile.reduce_seconds,
                        start.elapsed().as_secs_f64(),
                    );
                }
                (a_xtx, a_xtz, a_profile)
            },
        )
        .unwrap_or_else(|| {
            (
                vec![0.0; upper_len],
                vec![0.0; p],
                GramBuildProfile::default(),
            )
        });

    let materialize_start = collect_profile.then(Instant::now);
    // Convert packed upper triangle to nalgebra symmetric DMatrix.
    let mut xtx = DMatrix::zeros(p, p);
    let mut packed_idx = 0;
    for i in 0..p {
        for j in i..p {
            let val = xtx_data[packed_idx];
            xtx[(i, j)] = val;
            xtx[(j, i)] = val;
            packed_idx += 1;
        }
    }
    if let Some(start) = materialize_start {
        gram_profile.materialize_seconds = start.elapsed().as_secs_f64();
    }
    let xtz = DVector::from_vec(xtz_data);

    Ok((xtx, xtz, gram_profile))
}

#[inline]
fn packed_upper_len(p: usize) -> usize {
    p * (p + 1) / 2
}

#[inline]
fn sparse_xtwx_chunk_count(n: usize, p: usize) -> usize {
    let threads = rayon::current_num_threads().max(1);
    if n <= 1 {
        return 1;
    }
    let matrix_bytes = packed_upper_len(p).saturating_mul(std::mem::size_of::<f64>());
    // Wide sparse WLS is memory-bandwidth sensitive: each worker owns a large
    // thread-local Gram matrix and the reduction touches the whole matrix again.
    // Capping only those large-matrix cases avoids oversaturating memory while
    // leaving small problems free to use the whole pool.
    let cap = if matrix_bytes >= SPARSE_XTWX_LOCAL_MATRIX_CAP_THRESHOLD_BYTES {
        SPARSE_XTWX_THREAD_CAP
    } else {
        threads
    };
    threads.min(cap).min(n).max(1)
}

fn poisson_log_weight_chunk_size(n: usize) -> usize {
    let target_chunks = rayon::current_num_threads().saturating_mul(4).max(1);
    n.div_ceil(target_chunks).max(1)
}

fn compute_poisson_log_irls_weights_in_place(
    y: &Array1<f64>,
    mu: &Array1<f64>,
    eta: &Array1<f64>,
    offset: &Array1<f64>,
    prior_weights: &Array1<f64>,
    min_weight: f64,
    irls_weights: &mut Array1<f64>,
    combined_weights: &mut Array1<f64>,
    working_response: &mut Array1<f64>,
) -> Result<()> {
    let n = y.len();
    for (name, len) in [
        ("mu length vs y length", mu.len()),
        ("eta length vs y length", eta.len()),
        ("offset length vs y length", offset.len()),
        ("prior_weights length vs y length", prior_weights.len()),
        ("irls_weights length vs y length", irls_weights.len()),
        (
            "combined_weights length vs y length",
            combined_weights.len(),
        ),
        (
            "working_response length vs y length",
            working_response.len(),
        ),
    ] {
        if len != n {
            return Err(RustyStatsError::dim_mismatch(n, len, name));
        }
    }

    let y_slice = y.as_slice().ok_or_else(|| {
        RustyStatsError::LinearAlgebraError("Response vector y must be contiguous".to_string())
    })?;
    let mu_slice = mu.as_slice().ok_or_else(|| {
        RustyStatsError::LinearAlgebraError("Mean vector mu must be contiguous".to_string())
    })?;
    let eta_slice = eta.as_slice().ok_or_else(|| {
        RustyStatsError::LinearAlgebraError("Linear predictor eta must be contiguous".to_string())
    })?;
    let offset_slice = offset.as_slice().ok_or_else(|| {
        RustyStatsError::LinearAlgebraError("Offset vector must be contiguous".to_string())
    })?;
    let prior_slice = prior_weights.as_slice().ok_or_else(|| {
        RustyStatsError::LinearAlgebraError("Prior weights vector must be contiguous".to_string())
    })?;
    let irls_slice = irls_weights.as_slice_mut().ok_or_else(|| {
        RustyStatsError::LinearAlgebraError("IRLS weights buffer must be contiguous".to_string())
    })?;
    let combined_slice = combined_weights.as_slice_mut().ok_or_else(|| {
        RustyStatsError::LinearAlgebraError(
            "Combined weights buffer must be contiguous".to_string(),
        )
    })?;
    let response_slice = working_response.as_slice_mut().ok_or_else(|| {
        RustyStatsError::LinearAlgebraError(
            "Working response buffer must be contiguous".to_string(),
        )
    })?;

    let chunk_size = poisson_log_weight_chunk_size(n);
    irls_slice
        .par_chunks_mut(chunk_size)
        .zip(combined_slice.par_chunks_mut(chunk_size))
        .zip(response_slice.par_chunks_mut(chunk_size))
        .enumerate()
        .for_each(|(chunk_idx, ((iw_chunk, cw_chunk), wr_chunk))| {
            let start = chunk_idx * chunk_size;
            for local_idx in 0..iw_chunk.len() {
                let i = start + local_idx;
                let mui = unsafe { *mu_slice.get_unchecked(i) };
                let d = 1.0 / mui;
                let iw = (1.0 / (mui * d * d)).max(min_weight).min(MAX_IRLS_WEIGHT);
                unsafe {
                    *iw_chunk.get_unchecked_mut(local_idx) = iw;
                    *cw_chunk.get_unchecked_mut(local_idx) = *prior_slice.get_unchecked(i) * iw;
                    *wr_chunk.get_unchecked_mut(local_idx) = (*eta_slice.get_unchecked(i)
                        - *offset_slice.get_unchecked(i))
                        + (*y_slice.get_unchecked(i) - mui) * d;
                }
            }
        });

    Ok(())
}

pub struct SparseRowCache {
    n: usize,
    p: usize,
    data_ptr: usize,
    offsets: Vec<usize>,
    indices: Vec<u32>,
    values: Vec<f64>,
    packed_offsets: Vec<usize>,
}

impl SparseRowCache {
    fn is_compatible_with(&self, x: ArrayView2<'_, f64>) -> bool {
        x.as_slice().is_some_and(|slice| {
            self.n == x.nrows() && self.p == x.ncols() && self.data_ptr == slice.as_ptr() as usize
        })
    }
}

fn sampled_density(x_slice: &[f64], n: usize, p: usize) -> f64 {
    if n == 0 || p == 0 {
        return 0.0;
    }

    let sample_rows = n.min(1024);
    let mut nonzero = 0usize;
    for sample_idx in 0..sample_rows {
        let row = sample_idx * n / sample_rows;
        let row_start = row * p;
        for j in 0..p {
            if x_slice[row_start + j] != 0.0 {
                nonzero += 1;
            }
        }
    }
    nonzero as f64 / (sample_rows * p) as f64
}

fn sparse_row_cache_estimated_nnz(density: f64, n: usize, p: usize) -> f64 {
    density * (n as f64) * (p as f64)
}

fn should_build_sparse_row_cache(n: usize, p: usize, density: f64) -> bool {
    if n == 0 || p < 16 {
        return false;
    }
    let estimated_nnz = sparse_row_cache_estimated_nnz(density, n, p);
    density <= SPARSE_ROW_CACHE_DENSITY_THRESHOLD
        && estimated_nnz <= 60_000_000.0
        && n.saturating_mul(p) >= 10_000_000
}

pub fn build_sparse_row_cache_if_beneficial(x: ArrayView2<'_, f64>) -> Option<SparseRowCache> {
    let n = x.nrows();
    let p = x.ncols();
    let x_slice = x.as_slice()?;
    let density = sampled_density(x_slice, n, p);
    if !should_build_sparse_row_cache(n, p, density) {
        return None;
    }

    let estimated_nnz = sparse_row_cache_estimated_nnz(density, n, p);
    let reserve_nnz = estimated_nnz.ceil() as usize;
    let mut offsets = Vec::with_capacity(n + 1);
    let mut indices = Vec::with_capacity(reserve_nnz);
    let mut values = Vec::with_capacity(reserve_nnz);
    let packed_offsets = packed_upper_offsets(p);
    offsets.push(0);

    for row in 0..n {
        let row_start = row * p;
        for j in 0..p {
            let value = x_slice[row_start + j];
            if value != 0.0 {
                indices.push(j as u32);
                values.push(value);
            }
        }
        offsets.push(indices.len());
    }

    Some(SparseRowCache {
        n,
        p,
        data_ptr: x_slice.as_ptr() as usize,
        offsets,
        indices,
        values,
        packed_offsets,
    })
}

fn packed_upper_offsets(p: usize) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(p);
    let mut cursor = 0usize;
    for i in 0..p {
        offsets.push(cursor);
        cursor += p - i;
    }
    offsets
}

#[inline(always)]
fn accumulate_sparse_row_pairwise(
    xtx_local: &mut [f64],
    xtz_local: &mut [f64],
    packed_offsets: &[usize],
    nz_idx: &[usize],
    nz_val: &[f64],
    wk: f64,
    wz: f64,
) {
    debug_assert_eq!(nz_idx.len(), nz_val.len());
    let len = nz_idx.len();
    let xtx_ptr = xtx_local.as_mut_ptr();
    let xtz_ptr = xtz_local.as_mut_ptr();
    let idx_ptr = nz_idx.as_ptr();
    let val_ptr = nz_val.as_ptr();

    for a in 0..len {
        // SAFETY: callers populate nz_idx/nz_val in lockstep with column
        // indices < p. packed_offsets has length p, and base + j is the packed
        // upper-triangle offset for (i, j) where j >= i.
        unsafe {
            let i = *idx_ptr.add(a);
            let xki = *val_ptr.add(a);
            let xki_w = xki * wk;
            *xtz_ptr.add(i) += xki * wz;
            let base = *packed_offsets.get_unchecked(i) - i;

            let unrolled_end = sparse_pairwise_unrolled_end(a, len);
            for b in (a..unrolled_end).step_by(4) {
                let j0 = *idx_ptr.add(b);
                let j1 = *idx_ptr.add(b + 1);
                let j2 = *idx_ptr.add(b + 2);
                let j3 = *idx_ptr.add(b + 3);
                *xtx_ptr.add(base + j0) += xki_w * *val_ptr.add(b);
                *xtx_ptr.add(base + j1) += xki_w * *val_ptr.add(b + 1);
                *xtx_ptr.add(base + j2) += xki_w * *val_ptr.add(b + 2);
                *xtx_ptr.add(base + j3) += xki_w * *val_ptr.add(b + 3);
            }
            for b in unrolled_end..len {
                let j = *idx_ptr.add(b);
                *xtx_ptr.add(base + j) += xki_w * *val_ptr.add(b);
            }
        }
    }
}

#[inline(always)]
fn accumulate_cached_sparse_row_pairwise(
    cache: &SparseRowCache,
    start: usize,
    end: usize,
    xtx_local: &mut [f64],
    xtz_local: &mut [f64],
    wk: f64,
    wz: f64,
) {
    let xtx_ptr = xtx_local.as_mut_ptr();
    let xtz_ptr = xtz_local.as_mut_ptr();
    let idx_ptr = cache.indices.as_ptr();
    let val_ptr = cache.values.as_ptr();
    let packed_offsets_ptr = cache.packed_offsets.as_ptr();

    for a in start..end {
        // SAFETY: start/end are adjacent offsets from the cache, so every
        // position is within indices/values. Cached indices are sorted by row
        // scan order and are < p; base + j is therefore a valid packed offset.
        unsafe {
            let i = *idx_ptr.add(a) as usize;
            let xki = *val_ptr.add(a);
            let xki_w = xki * wk;
            *xtz_ptr.add(i) += xki * wz;
            let base = *packed_offsets_ptr.add(i) - i;

            let unrolled_end = sparse_pairwise_unrolled_end(a, end);
            for b in (a..unrolled_end).step_by(4) {
                let j0 = *idx_ptr.add(b) as usize;
                let j1 = *idx_ptr.add(b + 1) as usize;
                let j2 = *idx_ptr.add(b + 2) as usize;
                let j3 = *idx_ptr.add(b + 3) as usize;
                *xtx_ptr.add(base + j0) += xki_w * *val_ptr.add(b);
                *xtx_ptr.add(base + j1) += xki_w * *val_ptr.add(b + 1);
                *xtx_ptr.add(base + j2) += xki_w * *val_ptr.add(b + 2);
                *xtx_ptr.add(base + j3) += xki_w * *val_ptr.add(b + 3);
            }
            for b in unrolled_end..end {
                let j = *idx_ptr.add(b) as usize;
                *xtx_ptr.add(base + j) += xki_w * *val_ptr.add(b);
            }
        }
    }
}

#[inline(always)]
fn compute_xtwx_xtwz_sparse_cached(
    cache: &SparseRowCache,
    z: &Array1<f64>,
    w: &Array1<f64>,
    collect_profile: bool,
) -> Result<(DMatrix<f64>, DVector<f64>, GramBuildProfile)> {
    let n = cache.n;
    let p = cache.p;
    if z.len() != n || w.len() != n {
        return Err(RustyStatsError::dim_mismatch(
            n,
            z.len().min(w.len()),
            "sparse cache rows vs z/w length",
        ));
    }
    let z_slice = z.as_slice().ok_or_else(|| {
        RustyStatsError::LinearAlgebraError("Working response Z must be contiguous".to_string())
    })?;
    let w_slice = w.as_slice().ok_or_else(|| {
        RustyStatsError::LinearAlgebraError("Weight vector W must be contiguous".to_string())
    })?;

    let chunk_count = sparse_xtwx_chunk_count(n, p);
    let chunk_size = n.div_ceil(chunk_count).max(1);
    let num_chunks = n.div_ceil(chunk_size);
    let upper_len = packed_upper_len(p);

    let (xtx_data, xtz_data, mut gram_profile): (Vec<f64>, Vec<f64>, GramBuildProfile) = (0
        ..num_chunks)
        .into_par_iter()
        .map(|chunk_idx| {
            let chunk_start = chunk_idx * chunk_size;
            let chunk_end = (chunk_start + chunk_size).min(n);
            let mut profile = GramBuildProfile::default();
            let local_init_start = collect_profile.then(Instant::now);
            let mut xtx_local = vec![0.0; upper_len];
            let mut xtz_local = vec![0.0; p];
            if let Some(start) = local_init_start {
                profile.local_init_seconds = start.elapsed().as_secs_f64();
            }

            for row in chunk_start..chunk_end {
                let start = cache.offsets[row];
                let end = cache.offsets[row + 1];
                if start == end {
                    continue;
                }

                let wk = unsafe { *w_slice.get_unchecked(row) };
                let zk = unsafe { *z_slice.get_unchecked(row) };
                let wz = wk * zk;

                let pairwise_start = collect_profile.then(Instant::now);
                accumulate_cached_sparse_row_pairwise(
                    cache,
                    start,
                    end,
                    &mut xtx_local,
                    &mut xtz_local,
                    wk,
                    wz,
                );
                if let Some(start) = pairwise_start {
                    add_profile_seconds(
                        &mut profile.pairwise_accum_seconds,
                        start.elapsed().as_secs_f64(),
                    );
                }
            }
            (xtx_local, xtz_local, profile)
        })
        .reduce_with(
            |(mut a_xtx, mut a_xtz, mut a_profile), (b_xtx, b_xtz, b_profile)| {
                let reduce_start = collect_profile.then(Instant::now);
                a_profile.add(&b_profile);
                for i in 0..a_xtx.len() {
                    a_xtx[i] += b_xtx[i];
                }
                for i in 0..a_xtz.len() {
                    a_xtz[i] += b_xtz[i];
                }
                if let Some(start) = reduce_start {
                    add_profile_seconds(
                        &mut a_profile.reduce_seconds,
                        start.elapsed().as_secs_f64(),
                    );
                }
                (a_xtx, a_xtz, a_profile)
            },
        )
        .unwrap_or_else(|| {
            (
                vec![0.0; upper_len],
                vec![0.0; p],
                GramBuildProfile::default(),
            )
        });

    let materialize_start = collect_profile.then(Instant::now);
    let mut xtx = DMatrix::zeros(p, p);
    let mut packed_idx = 0usize;
    for i in 0..p {
        for j in i..p {
            let val = xtx_data[packed_idx];
            xtx[(i, j)] = val;
            xtx[(j, i)] = val;
            packed_idx += 1;
        }
    }
    if let Some(start) = materialize_start {
        gram_profile.materialize_seconds = start.elapsed().as_secs_f64();
    }
    Ok((xtx, DVector::from_vec(xtz_data), gram_profile))
}

#[inline]
fn matrix_vector_dot(x: ArrayView2<'_, f64>, coefficients: &Array1<f64>) -> Array1<f64> {
    let n = x.nrows();
    let p = x.ncols();
    assert_eq!(
        coefficients.len(),
        p,
        "coefficient vector length must match X columns"
    );

    let x_slice = match x.as_slice() {
        Some(slice) => slice,
        None => return x.dot(coefficients),
    };
    let coef_slice = match coefficients.as_slice() {
        Some(slice) => slice,
        None => return x.dot(coefficients),
    };
    if should_fallback_to_serial_matrix_vector_dot(n, p) {
        return x.dot(coefficients);
    }

    let target_chunks = rayon::current_num_threads().saturating_mul(4).max(1);
    let chunk_size = n.div_ceil(target_chunks).max(1);
    let mut output = vec![0.0; n];
    output
        .par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(chunk_idx, output_chunk)| {
            let row_start_idx = chunk_idx * chunk_size;
            for (local_row, out) in output_chunk.iter_mut().enumerate() {
                let row = row_start_idx + local_row;
                let offset = row * p;
                let mut sum = 0.0;
                for j in 0..p {
                    // SAFETY: row < n and j < p, so offset + j < n*p = x_slice.len();
                    // j < p = coef_slice.len().
                    sum += unsafe { *x_slice.get_unchecked(offset + j) }
                        * unsafe { *coef_slice.get_unchecked(j) };
                }
                *out = sum;
            }
        });

    Array1::from_vec(output)
}

#[inline]
pub fn matrix_vector_dot_cached(
    x: ArrayView2<'_, f64>,
    coefficients: &Array1<f64>,
    sparse_cache: Option<&SparseRowCache>,
) -> Array1<f64> {
    let Some(cache) = sparse_cache else {
        return matrix_vector_dot(x, coefficients);
    };
    let coef_slice = match coefficients.as_slice() {
        Some(slice) => slice,
        None => return matrix_vector_dot(x, coefficients),
    };
    let n = cache.n;
    let chunk_size = n
        .div_ceil(rayon::current_num_threads().saturating_mul(4).max(1))
        .max(1);
    let mut output = vec![0.0; n];
    output
        .par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(chunk_idx, output_chunk)| {
            let row_start_idx = chunk_idx * chunk_size;
            for (local_row, out) in output_chunk.iter_mut().enumerate() {
                let row = row_start_idx + local_row;
                let start = cache.offsets[row];
                let end = cache.offsets[row + 1];
                let mut sum = 0.0;
                for pos in start..end {
                    let col = unsafe { *cache.indices.get_unchecked(pos) as usize };
                    let value = unsafe { *cache.values.get_unchecked(pos) };
                    sum += value * unsafe { *coef_slice.get_unchecked(col) };
                }
                *out = sum;
            }
        });
    Array1::from_vec(output)
}

#[inline]
fn should_use_sparse_xtwx_kernel(x_slice: &[f64], n: usize, p: usize) -> bool {
    should_use_sparse_xtwx_kernel_at_density(n, p, sampled_density(x_slice, n, p))
}

fn should_use_sparse_xtwx_kernel_at_density(n: usize, p: usize, density: f64) -> bool {
    n != 0 && p >= 16 && density <= SPARSE_XTWX_DENSITY_THRESHOLD
}

fn should_use_parallel_matrix_vector_dot(n: usize, p: usize) -> bool {
    n.saturating_mul(p) >= 1_000_000 && rayon::current_num_threads() > 1
}

fn should_fallback_to_serial_matrix_vector_dot(n: usize, p: usize) -> bool {
    !should_use_parallel_matrix_vector_dot(n, p)
}

fn sparse_pairwise_unrolled_end(start: usize, end: usize) -> usize {
    start + ((end - start) / 4) * 4
}

#[inline]
fn compute_xtwx_xtwz_sparse(
    x_slice: &[f64],
    z_slice: &[f64],
    w_slice: &[f64],
    n: usize,
    p: usize,
    collect_profile: bool,
) -> Result<(DMatrix<f64>, DVector<f64>, GramBuildProfile)> {
    let chunk_count = sparse_xtwx_chunk_count(n, p);
    let chunk_size = n.div_ceil(chunk_count).max(1);
    let num_chunks = n.div_ceil(chunk_size);
    let upper_len = packed_upper_len(p);
    let packed_offsets = packed_upper_offsets(p);

    let (xtx_data, xtz_data, mut gram_profile): (Vec<f64>, Vec<f64>, GramBuildProfile) = (0
        ..num_chunks)
        .into_par_iter()
        .map(|chunk_idx| {
            let chunk_start = chunk_idx * chunk_size;
            let chunk_end = (chunk_start + chunk_size).min(n);
            let mut profile = GramBuildProfile::default();
            let local_init_start = collect_profile.then(Instant::now);
            let mut xtx_local = vec![0.0; upper_len];
            let mut xtz_local = vec![0.0; p];
            let mut nz_idx: Vec<usize> = Vec::with_capacity(p);
            let mut nz_val: Vec<f64> = Vec::with_capacity(p);
            if let Some(start) = local_init_start {
                profile.local_init_seconds = start.elapsed().as_secs_f64();
            }

            for k in chunk_start..chunk_end {
                let row_scan_start = collect_profile.then(Instant::now);
                nz_idx.clear();
                nz_val.clear();
                let row_start = k * p;
                for j in 0..p {
                    // SAFETY: row_start + j < n*p = x_slice.len().
                    let xkj = unsafe { *x_slice.get_unchecked(row_start + j) };
                    if xkj != 0.0 {
                        nz_idx.push(j);
                        nz_val.push(xkj);
                    }
                }
                if let Some(start) = row_scan_start {
                    add_profile_seconds(
                        &mut profile.row_scan_seconds,
                        start.elapsed().as_secs_f64(),
                    );
                }

                if nz_idx.is_empty() {
                    continue;
                }

                let pairwise_start = collect_profile.then(Instant::now);
                // SAFETY: k < n, so k is within both weight/response slices.
                let wk = unsafe { *w_slice.get_unchecked(k) };
                let zk = unsafe { *z_slice.get_unchecked(k) };
                let wz = wk * zk;
                accumulate_sparse_row_pairwise(
                    &mut xtx_local,
                    &mut xtz_local,
                    &packed_offsets,
                    &nz_idx,
                    &nz_val,
                    wk,
                    wz,
                );
                if let Some(start) = pairwise_start {
                    add_profile_seconds(
                        &mut profile.pairwise_accum_seconds,
                        start.elapsed().as_secs_f64(),
                    );
                }
            }
            (xtx_local, xtz_local, profile)
        })
        .reduce_with(
            |(mut a_xtx, mut a_xtz, mut a_profile), (b_xtx, b_xtz, b_profile)| {
                let reduce_start = collect_profile.then(Instant::now);
                a_profile.add(&b_profile);
                for i in 0..a_xtx.len() {
                    a_xtx[i] += b_xtx[i];
                }
                for i in 0..a_xtz.len() {
                    a_xtz[i] += b_xtz[i];
                }
                if let Some(start) = reduce_start {
                    add_profile_seconds(
                        &mut a_profile.reduce_seconds,
                        start.elapsed().as_secs_f64(),
                    );
                }
                (a_xtx, a_xtz, a_profile)
            },
        )
        .unwrap_or_else(|| {
            (
                vec![0.0; upper_len],
                vec![0.0; p],
                GramBuildProfile::default(),
            )
        });

    let materialize_start = collect_profile.then(Instant::now);
    let mut xtx = DMatrix::zeros(p, p);
    for i in 0..p {
        for j in i..p {
            let val = xtx_data[packed_offsets[i] + (j - i)];
            xtx[(i, j)] = val;
            xtx[(j, i)] = val;
        }
    }
    if let Some(start) = materialize_start {
        gram_profile.materialize_seconds = start.elapsed().as_secs_f64();
    }
    let xtz = DVector::from_vec(xtz_data);

    Ok((xtx, xtz, gram_profile))
}

/// Solve a symmetric positive-definite system Aβ = b using Cholesky decomposition.
///
/// Falls back to LU decomposition if Cholesky fails (near-singular systems).
/// Returns (coefficients, A⁻¹) as ndarray types.
///
/// When `skip_covariance` is true, the O(p³) matrix inverse is skipped and a
/// zero matrix is returned instead. This is useful for intermediate IRLS
/// iterations where only the coefficients are needed.
#[inline]
fn cholesky_solve(
    a: DMatrix<f64>,
    b: &DVector<f64>,
    skip_covariance: bool,
) -> Result<(Array1<f64>, Array2<f64>)> {
    let p = a.nrows();

    let chol = match a.clone().cholesky() {
        Some(c) => c,
        None => {
            // Fall back to LU decomposition
            match a.clone().lu().solve(b) {
                Some(sol) => {
                    let coef_array: Array1<f64> = sol.iter().copied().collect();
                    let cov_array = if skip_covariance {
                        Array2::zeros((p, p))
                    } else {
                        let a_inv = a.try_inverse().ok_or_else(|| {
                            RustyStatsError::LinearAlgebraError(
                                "Failed to compute covariance matrix - system is not invertible. \
                                 This often indicates multicollinearity in predictors."
                                    .to_string(),
                            )
                        })?;
                        let mut cov = Array2::zeros((p, p));
                        for i in 0..p {
                            for j in 0..p {
                                cov[[i, j]] = a_inv[(i, j)];
                            }
                        }
                        cov
                    };
                    return Ok((coef_array, cov_array));
                }
                None => {
                    return Err(RustyStatsError::LinearAlgebraError(
                        "Failed to solve linear system - matrix may be singular. \
                         This often indicates multicollinearity in predictors."
                            .to_string(),
                    ));
                }
            }
        }
    };

    let coefficients = chol.solve(b);

    let coef_array: Array1<f64> = coefficients.iter().copied().collect();
    let cov_array = if skip_covariance {
        Array2::zeros((p, p))
    } else {
        let identity = DMatrix::identity(p, p);
        let a_inv = chol.solve(&identity);
        let mut cov = Array2::zeros((p, p));
        for i in 0..p {
            for j in 0..p {
                cov[[i, j]] = a_inv[(i, j)];
            }
        }
        cov
    };

    Ok((coef_array, cov_array))
}

#[inline]
fn cholesky_solve_coefficients(a: DMatrix<f64>, b: &DVector<f64>) -> Result<Array1<f64>> {
    match a.clone().cholesky() {
        Some(chol) => Ok(chol.solve(b).iter().copied().collect()),
        None => match a.lu().solve(b) {
            Some(sol) => Ok(sol.iter().copied().collect()),
            None => Err(RustyStatsError::LinearAlgebraError(
                "Failed to solve linear system - matrix may be singular. \
                 This often indicates multicollinearity in predictors."
                    .to_string(),
            )),
        },
    }
}

#[inline]
fn cholesky_solve_spd_coefficients(a: DMatrix<f64>, b: &DVector<f64>) -> Result<Array1<f64>> {
    match a.cholesky() {
        Some(chol) => Ok(chol.solve(b).iter().copied().collect()),
        None => Err(RustyStatsError::LinearAlgebraError(
            "Failed to solve ridge linear system - matrix was not positive definite. \
             This usually indicates invalid weights or numerical instability."
                .to_string(),
        )),
    }
}

#[inline]
fn ridge_system_is_positive_definite_fast_path(
    w: &Array1<f64>,
    l2_penalty: f64,
    l2_penalty_factors: Option<&[f64]>,
    penalize_intercept: bool,
    p: usize,
) -> bool {
    if l2_penalty <= 0.0 || !w.iter().any(|&wi| wi > 0.0) {
        return false;
    }
    let start_idx = if penalize_intercept { 0 } else { 1 };
    (start_idx..p).all(|j| l2_penalty_factors.map_or(1.0, |factors| factors[j]) > 0.0)
}

fn pcg_max_iterations() -> usize {
    std::env::var("RUSTYSTATS_RIDGE_CV_PCG_MAX_ITER")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .unwrap_or(100)
}

fn pcg_tolerance() -> f64 {
    std::env::var("RUSTYSTATS_RIDGE_CV_PCG_TOL")
        .ok()
        .and_then(|raw| raw.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(1e-6)
}

fn pcg_input_slices_have_expected_lengths(
    x_len: usize,
    z_len: usize,
    w_len: usize,
    n: usize,
    p: usize,
) -> bool {
    x_len == n * p && z_len == n && w_len == n
}

fn pcg_positive_finite(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

fn pcg_residual_component(rhs: f64, applied: f64) -> f64 {
    rhs - applied
}

fn pcg_precondition_residual(residual: f64, diagonal: f64) -> f64 {
    if pcg_positive_finite(diagonal) {
        residual / diagonal
    } else {
        residual
    }
}

fn pcg_scaled_tolerance(tolerance: f64, rhs_norm: f64) -> f64 {
    tolerance * rhs_norm
}

fn pcg_step_size(rho: f64, denom: f64) -> f64 {
    rho / denom
}

fn pcg_beta(rho_next: f64, rho: f64) -> f64 {
    rho_next / rho
}

fn pcg_direction_component(z_precond: f64, beta_cg: f64, old_direction: f64) -> f64 {
    z_precond + beta_cg * old_direction
}

fn should_try_ridge_cv_pcg(skip_covariance: bool, l2_penalty: f64) -> bool {
    skip_covariance && l2_penalty > 0.0 && std::env::var_os("RUSTYSTATS_RIDGE_CV_PCG").is_some()
}

fn should_apply_l2_penalty(l2_penalty: f64) -> bool {
    l2_penalty > 0.0
}

fn penalty_matrix_shape_matches(p: usize, rows: usize, cols: usize) -> bool {
    rows == p && cols == p
}

#[allow(clippy::too_many_arguments)]
fn solve_weighted_least_squares_pcg(
    x: ArrayView2<'_, f64>,
    z: &Array1<f64>,
    w: &Array1<f64>,
    l2_penalty: f64,
    l2_penalty_factors: Option<&[f64]>,
    penalize_intercept: bool,
    initial_guess: Option<&Array1<f64>>,
    profile: &mut WLSSolveProfile,
) -> Result<Option<Array1<f64>>> {
    let n = x.nrows();
    let p = x.ncols();
    let x_slice = match x.as_slice() {
        Some(slice) => slice,
        None => return Ok(None),
    };
    let z_slice = match z.as_slice() {
        Some(slice) => slice,
        None => return Ok(None),
    };
    let w_slice = match w.as_slice() {
        Some(slice) => slice,
        None => return Ok(None),
    };
    if !pcg_input_slices_have_expected_lengths(x_slice.len(), z_slice.len(), w_slice.len(), n, p) {
        return Ok(None);
    }
    if let Some(init) = initial_guess {
        if init.len() != p {
            return Ok(None);
        }
    }

    let gram_start = Instant::now();
    let (rhs, mut diagonal) =
        compute_xtwz_and_weighted_diag_sparse_scan(x_slice, z_slice, w_slice, n, p);
    add_ridge_penalty_to_vector(
        &mut diagonal,
        l2_penalty,
        l2_penalty_factors,
        penalize_intercept,
    );
    profile.gram_seconds = gram_start.elapsed().as_secs_f64();

    if diagonal.iter().any(|value| !value.is_finite()) || rhs.iter().any(|value| !value.is_finite())
    {
        return Ok(None);
    }

    let solve_start = Instant::now();
    let mut beta = initial_guess
        .and_then(|init| init.as_slice().map(|slice| slice.to_vec()))
        .unwrap_or_else(|| vec![0.0; p]);
    let mut residual = {
        let applied = weighted_normal_matvec_sparse_scan(
            x_slice,
            w_slice,
            &beta,
            n,
            p,
            l2_penalty,
            l2_penalty_factors,
            penalize_intercept,
        );
        rhs.iter()
            .zip(applied.iter())
            .map(|(&b, &av)| pcg_residual_component(b, av))
            .collect::<Vec<_>>()
    };
    let mut z_precond = residual
        .iter()
        .zip(diagonal.iter())
        .map(|(&r, &d)| pcg_precondition_residual(r, d))
        .collect::<Vec<_>>();
    let mut direction = z_precond.clone();
    let mut rho = dot_slices(&residual, &z_precond);
    let rhs_norm = dot_slices(&rhs, &rhs).sqrt().max(1.0);
    let tolerance = pcg_scaled_tolerance(pcg_tolerance(), rhs_norm);
    if dot_slices(&residual, &residual).sqrt() <= tolerance {
        profile.solve_seconds = solve_start.elapsed().as_secs_f64();
        return Ok(Some(Array1::from_vec(beta)));
    }

    for _ in 0..pcg_max_iterations() {
        if !pcg_positive_finite(rho) {
            return Ok(None);
        }
        let mat_direction = weighted_normal_matvec_sparse_scan(
            x_slice,
            w_slice,
            &direction,
            n,
            p,
            l2_penalty,
            l2_penalty_factors,
            penalize_intercept,
        );
        let denom = dot_slices(&direction, &mat_direction);
        if !pcg_positive_finite(denom) {
            return Ok(None);
        }
        let step = pcg_step_size(rho, denom);
        for j in 0..p {
            beta[j] += step * direction[j];
            residual[j] -= step * mat_direction[j];
        }
        let residual_norm = dot_slices(&residual, &residual).sqrt();
        if residual_norm <= tolerance {
            profile.solve_seconds = solve_start.elapsed().as_secs_f64();
            return Ok(Some(Array1::from_vec(beta)));
        }
        for j in 0..p {
            let d = diagonal[j];
            z_precond[j] = pcg_precondition_residual(residual[j], d);
        }
        let rho_next = dot_slices(&residual, &z_precond);
        if !rho_next.is_finite() {
            return Ok(None);
        }
        let beta_cg = pcg_beta(rho_next, rho);
        for j in 0..p {
            direction[j] = pcg_direction_component(z_precond[j], beta_cg, direction[j]);
        }
        rho = rho_next;
    }

    Ok(None)
}

fn compute_xtwz_and_weighted_diag_sparse_scan(
    x_slice: &[f64],
    z_slice: &[f64],
    w_slice: &[f64],
    n: usize,
    p: usize,
) -> (Vec<f64>, Vec<f64>) {
    let chunk_count = sparse_xtwx_chunk_count(n, p);
    let chunk_size = n.div_ceil(chunk_count).max(1);
    let num_chunks = n.div_ceil(chunk_size);

    (0..num_chunks)
        .into_par_iter()
        .map(|chunk_idx| {
            let chunk_start = chunk_idx * chunk_size;
            let chunk_end = (chunk_start + chunk_size).min(n);
            let mut rhs_local = vec![0.0; p];
            let mut diag_local = vec![0.0; p];
            for row in chunk_start..chunk_end {
                let wk = unsafe { *w_slice.get_unchecked(row) };
                let zk = unsafe { *z_slice.get_unchecked(row) };
                let wz = wk * zk;
                let row_start = row * p;
                for j in 0..p {
                    let xij = unsafe { *x_slice.get_unchecked(row_start + j) };
                    if xij != 0.0 {
                        unsafe {
                            *rhs_local.get_unchecked_mut(j) += xij * wz;
                            *diag_local.get_unchecked_mut(j) += wk * xij * xij;
                        }
                    }
                }
            }
            (rhs_local, diag_local)
        })
        .reduce_with(|(mut a_rhs, mut a_diag), (b_rhs, b_diag)| {
            for j in 0..a_rhs.len() {
                a_rhs[j] += b_rhs[j];
                a_diag[j] += b_diag[j];
            }
            (a_rhs, a_diag)
        })
        .unwrap_or_else(|| (vec![0.0; p], vec![0.0; p]))
}

#[allow(clippy::too_many_arguments)]
fn weighted_normal_matvec_sparse_scan(
    x_slice: &[f64],
    w_slice: &[f64],
    vector: &[f64],
    n: usize,
    p: usize,
    l2_penalty: f64,
    l2_penalty_factors: Option<&[f64]>,
    penalize_intercept: bool,
) -> Vec<f64> {
    let chunk_count = sparse_xtwx_chunk_count(n, p);
    let chunk_size = n.div_ceil(chunk_count).max(1);
    let num_chunks = n.div_ceil(chunk_size);

    let mut out = (0..num_chunks)
        .into_par_iter()
        .map(|chunk_idx| {
            let chunk_start = chunk_idx * chunk_size;
            let chunk_end = (chunk_start + chunk_size).min(n);
            let mut local = vec![0.0; p];
            for row in chunk_start..chunk_end {
                let row_start = row * p;
                let mut dot = 0.0;
                for j in 0..p {
                    let xij = unsafe { *x_slice.get_unchecked(row_start + j) };
                    if xij != 0.0 {
                        dot += xij * unsafe { *vector.get_unchecked(j) };
                    }
                }
                let scaled = unsafe { *w_slice.get_unchecked(row) } * dot;
                if scaled == 0.0 {
                    continue;
                }
                for j in 0..p {
                    let xij = unsafe { *x_slice.get_unchecked(row_start + j) };
                    if xij != 0.0 {
                        unsafe { *local.get_unchecked_mut(j) += xij * scaled };
                    }
                }
            }
            local
        })
        .reduce_with(|mut a, b| {
            for j in 0..a.len() {
                a[j] += b[j];
            }
            a
        })
        .unwrap_or_else(|| vec![0.0; p]);

    add_ridge_penalty_to_matvec(
        &mut out,
        vector,
        l2_penalty,
        l2_penalty_factors,
        penalize_intercept,
    );
    out
}

fn add_ridge_penalty_to_vector(
    vector: &mut [f64],
    l2_penalty: f64,
    l2_penalty_factors: Option<&[f64]>,
    penalize_intercept: bool,
) {
    let start_idx = if penalize_intercept { 0 } else { 1 };
    for (j, value) in vector.iter_mut().enumerate().skip(start_idx) {
        let factor = l2_penalty_factors.map_or(1.0, |factors| factors[j]);
        *value += l2_penalty * factor;
    }
}

fn add_ridge_penalty_to_matvec(
    out: &mut [f64],
    vector: &[f64],
    l2_penalty: f64,
    l2_penalty_factors: Option<&[f64]>,
    penalize_intercept: bool,
) {
    let start_idx = if penalize_intercept { 0 } else { 1 };
    for j in start_idx..out.len() {
        let factor = l2_penalty_factors.map_or(1.0, |factors| factors[j]);
        out[j] += l2_penalty * factor * vector[j];
    }
}

fn dot_slices(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Solve penalized weighted least squares: minimize Σ w_i (z_i - x_i'β)² + λ Σ β_j²
///
/// Returns (coefficients, (X'WX + λI)⁻¹)
///
/// For Ridge (L2) regularization, we add λ to the diagonal of X'WX.
/// The intercept (first coefficient if `penalize_intercept` is false) is NOT penalized.
///
/// # Arguments
/// * `x` - Design matrix (n × p)
/// * `z` - Working response (n × 1)
/// * `w` - Observation weights (n × 1)
/// * `l2_penalty` - Ridge penalty λ (0.0 = no penalty)
/// * `penalize_intercept` - If false, first column is assumed to be intercept and not penalized
fn solve_weighted_least_squares_penalized(
    x: ArrayView2<'_, f64>,
    z: &Array1<f64>,
    w: &Array1<f64>,
    l2_penalty: f64,
    l2_penalty_factors: Option<&[f64]>,
    penalize_intercept: bool,
    skip_covariance: bool,
    sparse_cache: Option<&SparseRowCache>,
    initial_guess: Option<&Array1<f64>>,
) -> Result<(Array1<f64>, Array2<f64>, WLSSolveProfile)> {
    let p = x.ncols();
    if let Some(factors) = l2_penalty_factors {
        if factors.len() != p {
            return Err(RustyStatsError::dim_mismatch(
                p,
                factors.len(),
                "L2 penalty factors length vs X columns",
            ));
        }
    }
    let mut profile = WLSSolveProfile::default();

    if should_try_ridge_cv_pcg(skip_covariance, l2_penalty) {
        if let Some(coefficients) = solve_weighted_least_squares_pcg(
            x,
            z,
            w,
            l2_penalty,
            l2_penalty_factors,
            penalize_intercept,
            initial_guess,
            &mut profile,
        )? {
            return Ok((coefficients, Array2::zeros((0, 0)), profile));
        }
    }

    let gram_start = Instant::now();
    let collect_gram_subtimers = profile_gram_subtimers_enabled();
    let (mut xtx, xtz, gram_profile) = match sparse_cache {
        Some(cache) => compute_xtwx_xtwz_sparse_cached(cache, z, w, collect_gram_subtimers)?,
        None => compute_xtwx_xtwz_profiled(x, z, w)?,
    };
    profile.gram_seconds = gram_start.elapsed().as_secs_f64();
    profile.gram_local_init_seconds = gram_profile.local_init_seconds;
    profile.gram_row_scan_seconds = gram_profile.row_scan_seconds;
    profile.gram_pairwise_accum_seconds = gram_profile.pairwise_accum_seconds;
    profile.gram_reduce_seconds = gram_profile.reduce_seconds;
    profile.gram_materialize_seconds = gram_profile.materialize_seconds;

    // Add L2 (Ridge) penalty to diagonal: (X'WX + λI)
    // The intercept (first column) is typically NOT penalized.
    let penalty_start = Instant::now();
    if should_apply_l2_penalty(l2_penalty) {
        let start_idx = if penalize_intercept { 0 } else { 1 };
        for j in start_idx..p {
            let factor = l2_penalty_factors.map_or(1.0, |factors| factors[j]);
            xtx[(j, j)] += l2_penalty * factor;
        }
    }
    profile.penalty_seconds = penalty_start.elapsed().as_secs_f64();

    let solve_start = Instant::now();
    if skip_covariance {
        let coefficients = if ridge_system_is_positive_definite_fast_path(
            w,
            l2_penalty,
            l2_penalty_factors,
            penalize_intercept,
            p,
        ) {
            cholesky_solve_spd_coefficients(xtx, &xtz)?
        } else {
            cholesky_solve_coefficients(xtx, &xtz)?
        };
        profile.solve_seconds = solve_start.elapsed().as_secs_f64();
        Ok((coefficients, Array2::zeros((0, 0)), profile))
    } else {
        let (coefficients, covariance) = cholesky_solve(xtx, &xtz, false)?;
        profile.solve_seconds = solve_start.elapsed().as_secs_f64();
        Ok((coefficients, covariance, profile))
    }
}

/// Solve weighted least squares with a full penalty matrix.
///
/// This is used for penalized splines (P-splines, GAMs) where the penalty
/// is a structured matrix S = D'D rather than a scalar.
///
/// Solves: β = (X'WX + S)⁻¹ X'Wz
///
/// where S is the combined penalty matrix (already includes lambda scaling).
///
/// # Arguments
/// * `x` - Design matrix (n × p)
/// * `z` - Working response (n × 1)
/// * `w` - Observation weights (n × 1)
/// * `penalty_matrix` - Penalty matrix S (p × p), already scaled by lambdas
/// * `skip_covariance` - If true, skip the O(p³) matrix inverse and return zeros
///
/// # Returns
/// * Coefficients β (p × 1)
/// * Inverse of penalized normal equations (X'WX + S)⁻¹ (p × p), or zeros if skipped
pub fn solve_weighted_least_squares_with_penalty_matrix(
    x: ArrayView2<'_, f64>,
    z: &Array1<f64>,
    w: &Array1<f64>,
    penalty_matrix: &Array2<f64>,
    skip_covariance: bool,
) -> Result<(Array1<f64>, Array2<f64>)> {
    let p = x.ncols();

    // Validate penalty matrix dimensions
    if !penalty_matrix_shape_matches(p, penalty_matrix.nrows(), penalty_matrix.ncols()) {
        return Err(RustyStatsError::dim_mismatch(
            p,
            penalty_matrix.nrows(),
            format!(
                "penalty matrix shape ({}, {}) vs expected ({}, {})",
                penalty_matrix.nrows(),
                penalty_matrix.ncols(),
                p,
                p
            ),
        ));
    }

    let (mut xtx, xtz) = compute_xtwx_xtwz(x, z, w)?;

    // Add full penalty matrix S to X'WX
    for i in 0..p {
        for j in 0..p {
            xtx[(i, j)] += penalty_matrix[[i, j]];
        }
    }

    cholesky_solve(xtx, &xtz, skip_covariance)
}

/// Solve penalized WLS from pre-computed X'WX and X'Wz matrices.
///
/// This avoids recomputing the expensive O(n·p²) cross-product when it has
/// already been computed (e.g. for GCV optimization in the same IRLS iteration).
///
/// # Arguments
/// * `xtx` - Pre-computed X'WX (p × p) as nalgebra DMatrix
/// * `xtz` - Pre-computed X'Wz (p × 1) as nalgebra DVector
/// * `penalty_matrix` - Penalty matrix S (p × p) in ndarray format, already scaled
/// * `skip_covariance` - If true, skip the O(p³) matrix inverse and return zeros
///
/// # Returns
/// * Coefficients β (p × 1)
/// * Inverse of penalized normal equations (X'WX + S)⁻¹ (p × p), or zeros if skipped
pub fn solve_wls_from_precomputed(
    xtx: &DMatrix<f64>,
    xtz: &DVector<f64>,
    penalty_matrix: &Array2<f64>,
    skip_covariance: bool,
) -> Result<(Array1<f64>, Array2<f64>)> {
    let p = xtx.nrows();
    let mut xtx_pen = xtx.clone();

    // Add full penalty matrix S to X'WX
    for i in 0..p {
        for j in 0..p {
            xtx_pen[(i, j)] += penalty_matrix[[i, j]];
        }
    }

    cholesky_solve(xtx_pen, xtz, skip_covariance)
}

/// Compute X'WX matrix for EDF calculation.
///
/// This is needed for computing effective degrees of freedom in penalized regression.
pub fn compute_xtwx(x: ArrayView2<'_, f64>, w: &Array1<f64>) -> Array2<f64> {
    let p = x.ncols();

    let xtx_data = match xtwx_storage_path(x, w) {
        XtwxStoragePath::Contiguous { n, p } => {
            let x_slice = x.as_slice().expect("contiguous route requires X slice");
            let w_slice = w
                .as_slice()
                .expect("contiguous route requires weight slice");
            assert_eq!(x_slice.len(), n * p, "x_slice length must be n*p");
            assert_eq!(w_slice.len(), n, "w_slice length must be n");
            if should_use_sparse_xtwx_kernel(x_slice, n, p) {
                compute_xtwx_sparse_data(x_slice, w_slice, n, p)
            } else {
                compute_xtwx_dense_data(x_slice, w_slice, n, p)
            }
        }
        XtwxStoragePath::Strided => compute_xtwx_strided_data(x, w, x.nrows(), p),
    };

    xtx_data_to_array2(xtx_data, p)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum XtwxStoragePath {
    Contiguous { n: usize, p: usize },
    Strided,
}

fn xtwx_storage_path(x: ArrayView2<'_, f64>, w: &Array1<f64>) -> XtwxStoragePath {
    match (x.as_slice(), w.as_slice()) {
        (Some(_), Some(_)) => XtwxStoragePath::Contiguous {
            n: x.nrows(),
            p: x.ncols(),
        },
        _ => XtwxStoragePath::Strided,
    }
}

/// Compute `X.T @ diag(w) @ X`, reusing a pre-built sparse row cache when it
/// matches `x`.
///
/// The cache is independent of `w`, so callers that need many weighted Gram
/// matrices over the same design can build it once and avoid repeatedly
/// scanning sparse categorical rows. If the cache is absent or incompatible,
/// this falls back to [`compute_xtwx`].
pub fn compute_xtwx_with_sparse_cache(
    x: ArrayView2<'_, f64>,
    w: &Array1<f64>,
    sparse_cache: Option<&SparseRowCache>,
) -> Result<Array2<f64>> {
    let n = x.nrows();
    let p = x.ncols();
    if w.len() != n {
        return Err(RustyStatsError::dim_mismatch(
            n,
            w.len(),
            "X rows vs weight length",
        ));
    }

    let Some(cache) = sparse_cache.filter(|cache| cache.is_compatible_with(x)) else {
        return Ok(compute_xtwx(x, w));
    };

    let w_slice = w.as_slice().ok_or_else(|| {
        RustyStatsError::LinearAlgebraError("Weight vector W must be contiguous".to_string())
    })?;

    let chunk_count = sparse_xtwx_chunk_count(n, p);
    let chunk_size = n.div_ceil(chunk_count).max(1);
    let num_chunks = n.div_ceil(chunk_size);
    let upper_len = packed_upper_len(p);

    let xtx_data = (0..num_chunks)
        .into_par_iter()
        .map(|chunk_idx| {
            let chunk_start = chunk_idx * chunk_size;
            let chunk_end = (chunk_start + chunk_size).min(n);
            let mut xtx_local = vec![0.0; upper_len];

            for row in chunk_start..chunk_end {
                let start = cache.offsets[row];
                let end = cache.offsets[row + 1];
                if start == end {
                    continue;
                }

                let wk = unsafe { *w_slice.get_unchecked(row) };
                for a in start..end {
                    let i = unsafe { *cache.indices.get_unchecked(a) as usize };
                    let xki_w = unsafe { *cache.values.get_unchecked(a) } * wk;
                    let packed_row = unsafe { *cache.packed_offsets.get_unchecked(i) };

                    for b in a..end {
                        let j = unsafe { *cache.indices.get_unchecked(b) as usize };
                        let xkj = unsafe { *cache.values.get_unchecked(b) };
                        let packed_idx = packed_row + (j - i);
                        unsafe { *xtx_local.get_unchecked_mut(packed_idx) += xki_w * xkj };
                    }
                }
            }
            xtx_local
        })
        .reduce_with(|mut a, b| {
            for i in 0..a.len() {
                a[i] += b[i];
            }
            a
        })
        .unwrap_or_else(|| vec![0.0; upper_len]);

    let mut xtwx = Array2::zeros((p, p));
    let mut packed_idx = 0usize;
    for i in 0..p {
        for j in i..p {
            let val = xtx_data[packed_idx];
            xtwx[[i, j]] = val;
            xtwx[[j, i]] = val;
            packed_idx += 1;
        }
    }
    Ok(xtwx)
}

#[inline]
fn compute_xtwx_dense_data(x_slice: &[f64], w_slice: &[f64], n: usize, p: usize) -> Vec<f64> {
    let chunk_size = n.div_ceil(rayon::current_num_threads()).max(1);
    let num_chunks = n.div_ceil(chunk_size);

    (0..num_chunks)
        .into_par_iter()
        .map(|chunk_idx| {
            let chunk_start = chunk_idx * chunk_size;
            let chunk_end = (chunk_start + chunk_size).min(n);
            let mut xtx_local = vec![0.0; p * p];

            for k in chunk_start..chunk_end {
                // SAFETY: k < n, so k < w_slice.len().
                let wk = unsafe { *w_slice.get_unchecked(k) };
                let row_start = k * p;
                for i in 0..p {
                    // SAFETY: row_start + i < n*p = x_slice.len().
                    let xki = unsafe { *x_slice.get_unchecked(row_start + i) };
                    let xki_w = xki * wk;
                    for j in i..p {
                        // SAFETY: row_start + j < n*p = x_slice.len().
                        let xkj = unsafe { *x_slice.get_unchecked(row_start + j) };
                        // SAFETY: i, j < p, so i*p + j < xtx_local.len().
                        unsafe { *xtx_local.get_unchecked_mut(i * p + j) += xki_w * xkj };
                    }
                }
            }
            xtx_local
        })
        .reduce_with(|mut a, b| {
            for i in 0..a.len() {
                a[i] += b[i];
            }
            a
        })
        .unwrap_or_else(|| vec![0.0; p * p])
}

#[inline]
fn compute_xtwx_sparse_data(x_slice: &[f64], w_slice: &[f64], n: usize, p: usize) -> Vec<f64> {
    let chunk_size = n.div_ceil(rayon::current_num_threads()).max(1);
    let num_chunks = n.div_ceil(chunk_size);

    (0..num_chunks)
        .into_par_iter()
        .map(|chunk_idx| {
            let chunk_start = chunk_idx * chunk_size;
            let chunk_end = (chunk_start + chunk_size).min(n);
            let mut xtx_local = vec![0.0; p * p];
            let mut nz_idx: Vec<usize> = Vec::with_capacity(p);
            let mut nz_val: Vec<f64> = Vec::with_capacity(p);

            for k in chunk_start..chunk_end {
                nz_idx.clear();
                nz_val.clear();
                let row_start = k * p;
                for j in 0..p {
                    // SAFETY: row_start + j < n*p = x_slice.len().
                    let xkj = unsafe { *x_slice.get_unchecked(row_start + j) };
                    if xkj != 0.0 {
                        nz_idx.push(j);
                        nz_val.push(xkj);
                    }
                }

                if nz_idx.is_empty() {
                    continue;
                }

                // SAFETY: k < n, so k < w_slice.len().
                let wk = unsafe { *w_slice.get_unchecked(k) };
                for a in 0..nz_idx.len() {
                    // SAFETY: a is within both nz vectors populated in lockstep above.
                    let i = unsafe { *nz_idx.get_unchecked(a) };
                    let xki_w = unsafe { *nz_val.get_unchecked(a) } * wk;
                    for b in a..nz_idx.len() {
                        // SAFETY: b is within both nz vectors; i and j are
                        // column indices < p, so i*p + j < xtx_local.len().
                        let j = unsafe { *nz_idx.get_unchecked(b) };
                        let xkj = unsafe { *nz_val.get_unchecked(b) };
                        // SAFETY: i and j are column indices < p, so the
                        // flattened upper-triangle target is in bounds.
                        unsafe { *xtx_local.get_unchecked_mut(i * p + j) += xki_w * xkj };
                    }
                }
            }
            xtx_local
        })
        .reduce_with(|mut a, b| {
            for i in 0..a.len() {
                a[i] += b[i];
            }
            a
        })
        .unwrap_or_else(|| vec![0.0; p * p])
}

fn compute_xtwx_strided_data(
    x: ArrayView2<'_, f64>,
    w: &Array1<f64>,
    n: usize,
    p: usize,
) -> Vec<f64> {
    (0..n)
        .into_par_iter()
        .fold(
            || vec![0.0; p * p],
            |mut xtx_local, k| {
                let wk = w[k];
                for i in 0..p {
                    let xki_w = x[[k, i]] * wk;
                    for j in i..p {
                        xtx_local[i * p + j] += xki_w * x[[k, j]];
                    }
                }
                xtx_local
            },
        )
        .reduce_with(|mut a, b| {
            for i in 0..a.len() {
                a[i] += b[i];
            }
            a
        })
        .unwrap_or_else(|| vec![0.0; p * p])
}

fn xtx_data_to_array2(xtx_data: Vec<f64>, p: usize) -> Array2<f64> {
    let mut xtwx = Array2::zeros((p, p));
    for i in 0..p {
        for j in i..p {
            let val = xtx_data[i * p + j];
            xtwx[[i, j]] = val;
            xtwx[[j, i]] = val;
        }
    }

    xtwx
}

/// Compute working response: z = η + (y - μ) × g'(μ)
fn compute_working_response(
    y: &Array1<f64>,
    mu: &Array1<f64>,
    eta: &Array1<f64>,
    link: &dyn Link,
) -> Array1<f64> {
    let link_deriv = link.derivative(mu);

    eta.iter()
        .zip(y.iter())
        .zip(mu.iter())
        .zip(link_deriv.iter())
        .map(|(((&e, &yi), &mui), &d)| e + (yi - mui) * d)
        .collect()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::families::{BinomialFamily, GaussianFamily, PoissonFamily};
    use crate::links::{IdentityLink, LogLink, LogitLink};
    use ndarray::{array, s, Array2, ShapeBuilder};
    use std::{borrow::Cow, ffi::OsString, sync::Mutex};

    fn assert_array1_close(actual: &Array1<f64>, expected: &Array1<f64>, tol: f64) {
        assert_eq!(actual.len(), expected.len());
        for (idx, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < tol,
                "mismatch at {idx}: actual={a}, expected={e}, tol={tol}"
            );
        }
    }

    fn assert_array2_close(actual: &Array2<f64>, expected: &Array2<f64>, tol: f64) {
        assert_eq!(actual.dim(), expected.dim());
        for i in 0..actual.nrows() {
            for j in 0..actual.ncols() {
                let a = actual[[i, j]];
                let e = expected[[i, j]];
                assert!(
                    (a - e).abs() < tol,
                    "mismatch at ({i}, {j}): actual={a}, expected={e}, tol={tol}"
                );
            }
        }
    }

    fn assert_linear_algebra_contains(err: RustyStatsError, expected: &str) {
        match err {
            RustyStatsError::LinearAlgebraError(message) => assert!(
                message.contains(expected),
                "expected linear algebra message to contain '{expected}', got '{message}'"
            ),
            other => panic!("expected LinearAlgebraError, got {other:?}"),
        }
    }

    fn strided3(values: [f64; 3]) -> Array1<f64> {
        Array1::from_shape_vec(
            (3usize).strides(2),
            vec![values[0], 99.0, values[1], 99.0, values[2]],
        )
        .expect("valid strided vector")
    }

    fn strided3_buffer() -> Array1<f64> {
        strided3([0.0, 0.0, 0.0])
    }

    fn manual_sparse_cache(x: &Array2<f64>) -> SparseRowCache {
        let n = x.nrows();
        let p = x.ncols();
        let x_slice = x.as_slice().expect("test matrix should be contiguous");
        let mut offsets = Vec::with_capacity(n + 1);
        let mut indices = Vec::new();
        let mut values = Vec::new();
        offsets.push(0);
        for row in 0..n {
            let row_start = row * p;
            for col in 0..p {
                let value = x_slice[row_start + col];
                if value != 0.0 {
                    indices.push(col as u32);
                    values.push(value);
                }
            }
            offsets.push(indices.len());
        }
        SparseRowCache {
            n,
            p,
            data_ptr: x_slice.as_ptr() as usize,
            offsets,
            indices,
            values,
            packed_offsets: packed_upper_offsets(p),
        }
    }

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    struct EnvVarGuard {
        key: &'static str,
        previous: Option<OsString>,
    }

    impl EnvVarGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let previous = std::env::var_os(key);
            std::env::set_var(key, value);
            Self { key, previous }
        }

        fn unset(key: &'static str) -> Self {
            let previous = std::env::var_os(key);
            std::env::remove_var(key);
            Self { key, previous }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            if let Some(previous) = &self.previous {
                std::env::set_var(self.key, previous);
            } else {
                std::env::remove_var(self.key);
            }
        }
    }

    struct InvalidInitGaussian;

    impl Family for InvalidInitGaussian {
        fn name(&self) -> &str {
            "InvalidInitGaussian"
        }

        fn variance<'a>(&self, mu: &'a Array1<f64>) -> Cow<'a, Array1<f64>> {
            Cow::Owned(Array1::ones(mu.len()))
        }

        fn unit_deviance(&self, y: &Array1<f64>, mu: &Array1<f64>) -> Array1<f64> {
            (y - mu).mapv(|value| value * value)
        }

        fn unit_deviance_at(&self, yi: f64, mui: f64) -> f64 {
            let diff = yi - mui;
            diff * diff
        }

        fn default_link(&self) -> Box<dyn Link> {
            Box::new(IdentityLink)
        }

        fn initialize_mu(&self, y: &Array1<f64>) -> Array1<f64> {
            Array1::from_elem(y.len(), f64::NAN)
        }

        fn is_valid_mu(&self, mu: &Array1<f64>) -> bool {
            mu.iter().all(|value| value.is_finite())
        }

        fn clamp_mu(&self, mu: &Array1<f64>) -> Array1<f64> {
            mu.mapv(|value| if value.is_finite() { value } else { 0.0 })
        }
    }

    #[test]
    fn test_fit_config_builders_and_irls_conversion_contracts() {
        let reg = RegularizationConfig::ridge(0.25).with_intercept(false);
        let config = FitConfig::default()
            .with_regularization(reg.clone())
            .with_max_iterations(17)
            .with_tolerance(1e-5)
            .with_verbose(true)
            .with_nonneg_indices(vec![1, 3])
            .with_nonpos_indices(vec![2])
            .with_skip_covariance(true);

        assert_eq!(config.max_iterations, 17);
        assert_eq!(config.tolerance, 1e-5);
        assert!(config.verbose);
        assert_eq!(config.nonneg_indices, vec![1, 3]);
        assert_eq!(config.nonpos_indices, vec![2]);
        assert!(config.skip_covariance);
        assert_eq!(config.regularization.penalty, reg.penalty);
        assert!(!config.regularization.fit_intercept);

        let irls = config.to_irls_config();
        assert_eq!(irls.max_iterations, 17);
        assert_eq!(irls.tolerance, 1e-5);
        assert!(irls.verbose);
        assert_eq!(irls.nonneg_indices, vec![1, 3]);
        assert_eq!(irls.nonpos_indices, vec![2]);
        assert!(irls.skip_covariance);

        let legacy = IRLSConfig {
            max_iterations: 9,
            tolerance: 1e-4,
            min_weight: 1e-7,
            verbose: true,
            nonneg_indices: vec![4],
            nonpos_indices: vec![5],
            skip_covariance: true,
        };
        let from_legacy = FitConfig::from(&legacy);
        assert_eq!(from_legacy.max_iterations, legacy.max_iterations);
        assert_eq!(from_legacy.tolerance, legacy.tolerance);
        assert_eq!(from_legacy.min_weight, legacy.min_weight);
        assert!(from_legacy.verbose);
        assert_eq!(from_legacy.nonneg_indices, vec![4]);
        assert_eq!(from_legacy.nonpos_indices, vec![5]);
        assert!(from_legacy.regularization.penalty.is_none());
        assert!(
            !from_legacy.skip_covariance,
            "legacy conversion intentionally preserves the historical default"
        );
    }

    #[test]
    fn test_irls_scalar_helper_edge_contracts() {
        assert_eq!(packed_upper_len(0), 0);
        assert_eq!(packed_upper_len(1), 1);
        assert_eq!(packed_upper_len(4), 10);
        assert_eq!(
            SPARSE_XTWX_LOCAL_MATRIX_CAP_THRESHOLD_BYTES,
            1024 * 1024,
            "sparse XTWX local Gram cap is intentionally 1 MiB, not 1 KiB-scale"
        );

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("test thread pool should build");
        pool.install(|| {
            assert_eq!(sparse_xtwx_chunk_count(1, 2), 1);
            assert_eq!(sparse_xtwx_chunk_count(3, 2), 3);
            assert_eq!(
                sparse_xtwx_chunk_count(10, 23),
                4,
                "small local Gram matrices should use the full pool"
            );
            assert_eq!(sparse_xtwx_chunk_count(10, 600), SPARSE_XTWX_THREAD_CAP);
            assert_eq!(poisson_log_weight_chunk_size(33), 3);
        });

        assert!(!irls_step_converged(100.0, 99.0, 1e-4, 1e-4));
        assert!(irls_step_converged(100.0, 99.999, 1e-5, 1e-4));
    }

    #[test]
    fn test_profile_helpers_and_env_flag_have_exact_contracts() {
        let mut profile = GramBuildProfile {
            local_init_seconds: 1.0,
            row_scan_seconds: 2.0,
            pairwise_accum_seconds: 3.0,
            reduce_seconds: 4.0,
            materialize_seconds: 5.0,
        };
        let other = GramBuildProfile {
            local_init_seconds: 0.5,
            row_scan_seconds: 1.5,
            pairwise_accum_seconds: 2.5,
            reduce_seconds: 3.5,
            materialize_seconds: 4.5,
        };
        profile.add(&other);
        assert_eq!(profile.local_init_seconds, 1.5);
        assert_eq!(profile.row_scan_seconds, 3.5);
        assert_eq!(profile.pairwise_accum_seconds, 5.5);
        assert_eq!(profile.reduce_seconds, 7.5);
        assert_eq!(profile.materialize_seconds, 9.5);

        let mut total = 1.25;
        add_profile_seconds(&mut total, 2.75);
        assert_eq!(total, 4.0);

        let _env_lock = ENV_LOCK.lock().expect("env lock should not be poisoned");
        {
            let _unset = EnvVarGuard::unset("RUSTYSTATS_PROFILE_GRAM_SUBTIMERS");
            assert!(!profile_gram_subtimers_enabled());
        }
        {
            let _set = EnvVarGuard::set("RUSTYSTATS_PROFILE_GRAM_SUBTIMERS", "1");
            assert!(profile_gram_subtimers_enabled());
        }
    }

    #[test]
    fn test_sparse_routing_and_cache_threshold_helpers_have_exact_contracts() {
        let density_fixture = vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 0.0, 5.0, 0.0, 0.0];
        assert_eq!(sampled_density(&density_fixture, 4, 3), 5.0 / 12.0);

        assert_eq!(
            sparse_row_cache_estimated_nnz(0.3, 2_000_000, 100),
            60_000_000.0
        );
        assert!(!should_build_sparse_row_cache(0, 16, 0.0));
        assert!(!should_build_sparse_row_cache(625_000, 15, 0.0));
        assert!(!should_build_sparse_row_cache(1_000_000, 15, 0.0));
        assert!(!should_build_sparse_row_cache(624_999, 16, 0.0));
        assert!(should_build_sparse_row_cache(
            625_000,
            16,
            SPARSE_ROW_CACHE_DENSITY_THRESHOLD
        ));
        assert!(!should_build_sparse_row_cache(
            625_000,
            16,
            SPARSE_ROW_CACHE_DENSITY_THRESHOLD + 1e-12
        ));
        assert!(should_build_sparse_row_cache(2_000_000, 100, 0.3));
        assert!(!should_build_sparse_row_cache(2_000_000, 100, 0.30000001));

        assert!(!should_use_sparse_xtwx_kernel_at_density(0, 16, 0.0));
        assert!(!should_use_sparse_xtwx_kernel_at_density(10, 15, 0.0));
        assert!(should_use_sparse_xtwx_kernel_at_density(
            10,
            16,
            SPARSE_XTWX_DENSITY_THRESHOLD
        ));
        assert!(!should_use_sparse_xtwx_kernel_at_density(
            10,
            16,
            SPARSE_XTWX_DENSITY_THRESHOLD + 1e-12
        ));

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("test thread pool should build");
        pool.install(|| {
            assert!(!should_use_parallel_matrix_vector_dot(999, 1001));
            assert!(should_fallback_to_serial_matrix_vector_dot(999, 1001));
            assert!(should_use_parallel_matrix_vector_dot(1000, 1000));
            assert!(!should_fallback_to_serial_matrix_vector_dot(1000, 1000));
        });
        let single_thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("test thread pool should build");
        single_thread_pool.install(|| {
            assert!(!should_use_parallel_matrix_vector_dot(1000, 1000));
            assert!(should_fallback_to_serial_matrix_vector_dot(1000, 1000));
        });

        let x = Array2::from_shape_vec(
            (2, 16),
            vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0,
                0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0,
            ],
        )
        .expect("test matrix should be valid");
        let cache = manual_sparse_cache(&x);
        assert!(cache.is_compatible_with(x.view()));
        let same_shape_different_storage = x.clone();
        assert!(!cache.is_compatible_with(same_shape_different_storage.view()));
        let different_rows = x.slice(s![..1, ..]);
        assert!(!cache.is_compatible_with(different_rows));
        let different_cols = Array2::<f64>::zeros((2, 17));
        assert!(!cache.is_compatible_with(different_cols.view()));
    }

    #[test]
    fn test_sparse_pairwise_accumulators_match_manual_packed_formula() {
        let p = 6usize;
        let packed_offsets = packed_upper_offsets(p);
        let nz_idx = vec![0usize, 1, 2, 4, 5];
        let nz_val = vec![1.0, -2.0, 0.5, 3.0, -1.5];
        let wk = 2.0;
        let wz = -1.25;

        assert_eq!(sparse_pairwise_unrolled_end(0, 5), 4);
        assert_eq!(sparse_pairwise_unrolled_end(1, 5), 5);
        assert_eq!(sparse_pairwise_unrolled_end(2, 5), 2);

        let mut expected_xtx = vec![0.0; packed_upper_len(p)];
        let mut expected_xtz = vec![0.0; p];
        for a in 0..nz_idx.len() {
            let i = nz_idx[a];
            let xki = nz_val[a];
            let xki_w = xki * wk;
            expected_xtz[i] += xki * wz;
            let base = packed_offsets[i] - i;
            for b in a..nz_idx.len() {
                expected_xtx[base + nz_idx[b]] += xki_w * nz_val[b];
            }
        }

        let mut actual_xtx = vec![0.0; packed_upper_len(p)];
        let mut actual_xtz = vec![0.0; p];
        accumulate_sparse_row_pairwise(
            &mut actual_xtx,
            &mut actual_xtz,
            &packed_offsets,
            &nz_idx,
            &nz_val,
            wk,
            wz,
        );
        for (idx, (actual, expected)) in actual_xtx.iter().zip(expected_xtx.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-12,
                "uncached packed xtx mismatch at {idx}: actual={actual}, expected={expected}"
            );
        }
        assert_array1_close(
            &Array1::from_vec(actual_xtz.clone()),
            &Array1::from_vec(expected_xtz.clone()),
            1e-12,
        );

        let cache = SparseRowCache {
            n: 1,
            p,
            data_ptr: 0,
            offsets: vec![0, nz_idx.len()],
            indices: nz_idx.iter().map(|&idx| idx as u32).collect(),
            values: nz_val.clone(),
            packed_offsets,
        };
        let mut cached_xtx = vec![0.0; packed_upper_len(p)];
        let mut cached_xtz = vec![0.0; p];
        accumulate_cached_sparse_row_pairwise(
            &cache,
            0,
            nz_idx.len(),
            &mut cached_xtx,
            &mut cached_xtz,
            wk,
            wz,
        );
        for (idx, (actual, expected)) in cached_xtx.iter().zip(expected_xtx.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-12,
                "cached packed xtx mismatch at {idx}: actual={actual}, expected={expected}"
            );
        }
        assert_array1_close(
            &Array1::from_vec(cached_xtz),
            &Array1::from_vec(expected_xtz),
            1e-12,
        );
    }

    #[test]
    fn test_pcg_and_ridge_scalar_helpers_have_exact_contracts() {
        assert!(pcg_input_slices_have_expected_lengths(12, 3, 3, 3, 4));
        assert!(!pcg_input_slices_have_expected_lengths(11, 3, 3, 3, 4));
        assert!(!pcg_input_slices_have_expected_lengths(12, 2, 3, 3, 4));
        assert!(!pcg_input_slices_have_expected_lengths(12, 3, 2, 3, 4));

        assert!(pcg_positive_finite(1.0));
        assert!(!pcg_positive_finite(0.0));
        assert!(!pcg_positive_finite(-1.0));
        assert!(!pcg_positive_finite(f64::INFINITY));
        assert!(!pcg_positive_finite(f64::NAN));
        assert_eq!(pcg_residual_component(7.0, 2.5), 4.5);
        assert_eq!(pcg_precondition_residual(8.0, 4.0), 2.0);
        assert_eq!(pcg_precondition_residual(8.0, 0.0), 8.0);
        assert_eq!(pcg_precondition_residual(8.0, -1.0), 8.0);
        assert_eq!(pcg_precondition_residual(8.0, f64::INFINITY), 8.0);
        assert_eq!(pcg_scaled_tolerance(0.01, 5.0), 0.05);
        assert_eq!(pcg_step_size(6.0, 3.0), 2.0);
        assert_eq!(pcg_beta(2.0, 8.0), 0.25);
        assert_eq!(pcg_direction_component(3.0, 0.25, 8.0), 5.0);

        let weights = array![0.0, 2.0, 0.0];
        assert!(!ridge_system_is_positive_definite_fast_path(
            &weights, 0.0, None, false, 3
        ));
        assert!(!ridge_system_is_positive_definite_fast_path(
            &array![0.0, 0.0],
            0.5,
            None,
            false,
            3
        ));
        assert!(ridge_system_is_positive_definite_fast_path(
            &weights, 0.5, None, false, 3
        ));
        assert!(!ridge_system_is_positive_definite_fast_path(
            &weights,
            0.5,
            Some(&[1.0, 0.0, 2.0]),
            false,
            3
        ));
        assert!(ridge_system_is_positive_definite_fast_path(
            &weights,
            0.5,
            Some(&[0.0, 1.0, 2.0]),
            false,
            3
        ));

        let mut vector = vec![10.0, 20.0, 30.0];
        add_ridge_penalty_to_vector(&mut vector, 2.0, Some(&[100.0, 0.5, 3.0]), false);
        assert_eq!(vector, vec![10.0, 21.0, 36.0]);
        let mut vector = vec![10.0, 20.0];
        add_ridge_penalty_to_vector(&mut vector, 2.0, None, true);
        assert_eq!(vector, vec![12.0, 22.0]);

        assert!(!should_apply_l2_penalty(0.0));
        assert!(!should_apply_l2_penalty(-1.0));
        assert!(should_apply_l2_penalty(1e-12));
        assert!(penalty_matrix_shape_matches(2, 2, 2));
        assert!(!penalty_matrix_shape_matches(2, 1, 2));
        assert!(!penalty_matrix_shape_matches(2, 2, 1));
    }

    #[test]
    fn test_pcg_environment_and_xtwx_route_helpers_have_exact_contracts() {
        let _env_lock = ENV_LOCK.lock().expect("env lock should not be poisoned");
        {
            let _unset = EnvVarGuard::unset("RUSTYSTATS_RIDGE_CV_PCG_TOL");
            assert_eq!(pcg_tolerance(), 1e-6);
        }
        {
            let _set = EnvVarGuard::set("RUSTYSTATS_RIDGE_CV_PCG_TOL", "0.00025");
            assert_eq!(pcg_tolerance(), 0.00025);
        }
        {
            let _set = EnvVarGuard::set("RUSTYSTATS_RIDGE_CV_PCG_TOL", "0");
            assert_eq!(pcg_tolerance(), 1e-6);
        }
        {
            let _set = EnvVarGuard::set("RUSTYSTATS_RIDGE_CV_PCG_TOL", "inf");
            assert_eq!(pcg_tolerance(), 1e-6);
        }

        {
            let _unset = EnvVarGuard::unset("RUSTYSTATS_RIDGE_CV_PCG");
            assert!(!should_try_ridge_cv_pcg(true, 1.0));
        }
        {
            let _set = EnvVarGuard::set("RUSTYSTATS_RIDGE_CV_PCG", "1");
            assert!(should_try_ridge_cv_pcg(true, 1.0));
            assert!(!should_try_ridge_cv_pcg(false, 1.0));
            assert!(!should_try_ridge_cv_pcg(true, 0.0));
        }

        let x = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0])
            .expect("test matrix should be valid");
        let w = array![0.5, 1.5];
        assert!(matches!(
            xtwx_storage_path(x.view(), &w),
            XtwxStoragePath::Contiguous { n: 2, p: 2, .. }
        ));

        let strided_w = strided3([0.5, 1.5, 2.5]);
        assert!(matches!(
            xtwx_storage_path(x.view(), &strided_w),
            XtwxStoragePath::Strided
        ));

        let wide = Array2::from_shape_vec((2, 4), vec![1.0, 9.0, 2.0, 9.0, 3.0, 9.0, 4.0, 9.0])
            .expect("test matrix should be valid");
        let strided_x = wide.slice(s![.., ..;2]);
        assert!(matches!(
            xtwx_storage_path(strided_x, &w),
            XtwxStoragePath::Strided
        ));
    }

    #[test]
    fn test_fit_core_decision_and_arithmetic_helpers_have_exact_contracts() {
        let none_config = FitConfig::default();
        assert!(!should_standardize_regularized_design(&none_config));
        let ridge_config = FitConfig {
            regularization: RegularizationConfig::ridge(0.5),
            ..FitConfig::default()
        };
        assert!(should_standardize_regularized_design(&ridge_config));
        assert!(centers_are_all_zero(&[0.0, 0.0]));
        assert!(!centers_are_all_zero(&[0.0, 1e-12]));
        assert!(should_use_scale_only_standardized_ridge_path(
            0.5,
            false,
            &[0.0, 0.0]
        ));
        assert!(!should_use_scale_only_standardized_ridge_path(
            0.0,
            false,
            &[0.0]
        ));
        assert!(!should_use_scale_only_standardized_ridge_path(
            0.5,
            true,
            &[0.0]
        ));
        assert!(!should_use_scale_only_standardized_ridge_path(
            0.5,
            false,
            &[0.1]
        ));

        let eta_base = array![1.0, -2.0, 0.5];
        let offset = array![0.25, 3.0, -0.75];
        assert_array1_close(
            &add_offset_to_linear_predictor(&eta_base, &offset),
            &array![1.25, 1.0, -0.25],
            1e-12,
        );
        assert_array1_close(
            &eta_without_offset(&array![1.25, 1.0, -0.25], &offset),
            &eta_base,
            1e-12,
        );

        assert!(coefficients_are_finite(&array![0.0, 1.0]));
        assert!(!coefficients_are_finite(&array![0.0, f64::NAN]));
        assert!(!coefficients_are_finite(&array![0.0, f64::INFINITY]));
        assert!(!coefficients_contain_nonfinite(&array![0.0, 1.0]));
        assert!(coefficients_contain_nonfinite(&array![0.0, f64::NAN]));
        assert!(coefficients_contain_nonfinite(&array![
            0.0,
            f64::NEG_INFINITY
        ]));
        assert_array1_close(
            &finite_coefficients_or_none(array![2.0, -3.0]).unwrap(),
            &array![2.0, -3.0],
            1e-12,
        );
        assert!(finite_coefficients_or_none(array![f64::INFINITY]).is_none());

        let neg_zero = (-0.0_f64).to_bits();
        let mut coefficients = array![-2.0, -0.0, 3.0, -4.0, 5.0, -0.0];
        project_coefficients_to_sign_constraints(&mut coefficients, &[0, 1, 6], &[2, 5, 6]);
        assert_eq!(coefficients[0], 0.0);
        assert_eq!(coefficients[1].to_bits(), neg_zero);
        assert_eq!(coefficients[2], 0.0);
        assert_eq!(coefficients[3], -4.0);
        assert_eq!(coefficients[4], 5.0);
        assert_eq!(coefficients[5].to_bits(), neg_zero);

        assert!(!has_sign_constraints(&[], &[]));
        assert!(has_sign_constraints(&[1], &[]));
        assert!(has_sign_constraints(&[], &[2]));
        assert!(should_use_poisson_log_weight_buffers("PoIsSoN", "log"));
        assert!(!should_use_poisson_log_weight_buffers(
            "poisson", "identity"
        ));
        assert!(!should_use_poisson_log_weight_buffers("gaussian", "log"));

        assert!(trial_state_is_acceptable(
            &array![1.0, 2.0],
            &array![3.0, 4.0],
            10.0,
            10.0
        ));
        assert!(!trial_state_is_acceptable(
            &array![f64::NAN],
            &array![3.0],
            1.0,
            2.0
        ));
        assert!(!trial_state_is_acceptable(
            &array![1.0],
            &array![f64::INFINITY],
            1.0,
            2.0
        ));
        assert!(!trial_state_is_acceptable(
            &array![1.0],
            &array![3.0],
            2.1,
            2.0
        ));

        assert_eq!(blend_coefficient(10.0, 2.0, 0.25), 8.0);
        assert_eq!(objective_relative_change(8.0, 6.0), 0.25);
        assert_eq!(objective_relative_change(0.0, 2.0), 2.0);
        assert_eq!(objective_relative_change(ZERO_TOL, 0.0), ZERO_TOL);
        assert!(best_objective_improved(0.9, 1.0));
        assert!(!best_objective_improved(1.0, 1.0));
        assert!(!should_stop_after_stale_best(
            CONSTRAINED_BEST_EARLY_STOP_PATIENCE - 1
        ));
        assert!(should_stop_after_stale_best(
            CONSTRAINED_BEST_EARLY_STOP_PATIENCE
        ));

        assert!(should_use_constrained_best(true, 0.9, 1.0, false));
        assert!(!should_use_constrained_best(true, 1.0, 1.0, false));
        assert!(should_use_constrained_best(true, 1.0, 1.0, true));
        assert!(!should_use_constrained_best(false, 0.9, 1.0, true));

        assert_array1_close(
            &combine_prior_and_irls_weights(&array![2.0, 0.5], &array![3.0, 4.0]),
            &array![6.0, 2.0],
            1e-12,
        );
        assert!(final_extraction_deviance_acceptable(4.0, 4.0));
        assert!(!final_extraction_deviance_acceptable(4.1, 4.0));
        assert!(!final_extraction_deviance_acceptable(f64::NAN, 4.0));
        assert!(should_recompute_final_state(false));
        assert!(!should_recompute_final_state(true));
        assert!(should_warn_nonconverged(false));
        assert!(!should_warn_nonconverged(true));
    }

    #[test]
    fn test_dense_xtwx_data_and_materialization_are_exact_on_multiple_chunks() {
        let x = Array2::from_shape_vec(
            (5, 3),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -2.0, 1.0, 0.5, 0.0, -3.0, 2.0, 7.0, -1.0, 4.0,
            ],
        )
        .expect("test matrix should be valid");
        let weights = array![0.5, 1.5, 2.0, 0.75, 1.25];
        let n = x.nrows();
        let p = x.ncols();
        let x_slice = x.as_slice().expect("test matrix should be contiguous");
        let w_slice = weights
            .as_slice()
            .expect("test weights should be contiguous");

        let mut expected_data = vec![0.0; p * p];
        for row in 0..n {
            let row_start = row * p;
            for i in 0..p {
                let xki_w = x_slice[row_start + i] * w_slice[row];
                for j in i..p {
                    expected_data[i * p + j] += xki_w * x_slice[row_start + j];
                }
            }
        }

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("test thread pool should build");
        let actual_data = pool.install(|| compute_xtwx_dense_data(x_slice, w_slice, n, p));
        assert_eq!(actual_data.len(), p * p);
        for (idx, (actual, expected)) in actual_data.iter().zip(expected_data.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-12,
                "dense xtwx data mismatch at {idx}: actual={actual}, expected={expected}"
            );
        }

        let actual = xtx_data_to_array2(actual_data, p);
        let mut expected = Array2::<f64>::zeros((p, p));
        for i in 0..p {
            for j in i..p {
                expected[[i, j]] = expected_data[i * p + j];
                expected[[j, i]] = expected_data[i * p + j];
            }
        }
        assert_array2_close(&actual, &expected, 1e-12);
    }

    #[test]
    fn test_compute_working_response_uses_eta_plus_residual_times_link_derivative() {
        let y = array![8.0, 1.0, 10.0];
        let mu = array![2.0, 4.0, 5.0];
        let eta = array![3.0, 5.0, 7.0];

        assert_array1_close(
            &compute_working_response(&y, &mu, &eta, &LogLink),
            &array![6.0, 4.25, 8.0],
            1e-12,
        );
    }

    #[test]
    fn test_poisson_log_in_place_weights_match_public_weight_contract_and_validate_lengths() {
        let y = array![0.0, 1.0, 5.0];
        let mu = array![0.1, 1.5, 4.0];
        let eta = LogLink.link(&mu) + array![0.2, -0.1, 0.0];
        let offset = array![0.2, -0.1, 0.0];
        let prior = array![2.0, 0.5, 1.25];
        let mut irls = Array1::zeros(y.len());
        let mut combined = Array1::zeros(y.len());
        let mut response = Array1::zeros(y.len());

        compute_poisson_log_irls_weights_in_place(
            &y,
            &mu,
            &eta,
            &offset,
            &prior,
            0.25,
            &mut irls,
            &mut combined,
            &mut response,
        )
        .expect("valid buffers should be filled");

        assert_array1_close(&irls, &array![0.25, 1.5, 4.0], 1e-12);
        assert_array1_close(&combined, &array![0.5, 0.75, 5.0], 1e-12);
        let expected_response = (&eta - &offset) + (&y - &mu) / &mu;
        assert_array1_close(&response, &expected_response, 1e-12);

        let mut too_short = Array1::zeros(2);
        let err = compute_poisson_log_irls_weights_in_place(
            &y,
            &mu,
            &eta,
            &offset,
            &prior,
            0.25,
            &mut too_short,
            &mut combined,
            &mut response,
        )
        .expect_err("buffer length mismatch should be reported");
        assert!(matches!(err, RustyStatsError::DimensionMismatch { .. }));
    }

    #[test]
    fn test_poisson_log_in_place_parallel_chunks_index_global_rows_correctly() {
        let n = 33usize;
        let y: Array1<f64> = (0..n)
            .map(|i| 0.25 + (i as f64 * 0.37).sin().abs() * 8.0)
            .collect();
        let mu: Array1<f64> = (0..n)
            .map(|i| 0.5 + (i % 7) as f64 * 0.3 + i as f64 * 0.01)
            .collect();
        let offset: Array1<f64> = (0..n).map(|i| (i as f64 * 0.11).cos() * 0.2).collect();
        let eta = LogLink.link(&mu) + &offset;
        let prior: Array1<f64> = (0..n).map(|i| 0.75 + (i % 5) as f64 * 0.2).collect();
        let mut irls = Array1::zeros(n);
        let mut combined = Array1::zeros(n);
        let mut response = Array1::zeros(n);

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("test thread pool should build");
        pool.install(|| {
            compute_poisson_log_irls_weights_in_place(
                &y,
                &mu,
                &eta,
                &offset,
                &prior,
                1e-12,
                &mut irls,
                &mut combined,
                &mut response,
            )
            .expect("valid buffers should be filled");
        });

        for i in [0usize, 1, 2, 3, 4, 15, 16, 17, 31, 32] {
            assert!(
                (irls[i] - mu[i]).abs() < 1e-12,
                "IRLS weight mismatch at row {i}"
            );
            assert!(
                (combined[i] - prior[i] * mu[i]).abs() < 1e-12,
                "combined weight mismatch at row {i}"
            );
            let expected_response = (eta[i] - offset[i]) + (y[i] - mu[i]) / mu[i];
            assert!(
                (response[i] - expected_response).abs() < 1e-12,
                "working response mismatch at row {i}: actual={}, expected={expected_response}",
                response[i]
            );
        }
    }

    #[test]
    fn test_poisson_log_in_place_weights_reject_noncontiguous_inputs_and_buffers() {
        let y = array![0.0, 1.0, 5.0];
        let mu = array![0.1, 1.5, 4.0];
        let eta = LogLink.link(&mu);
        let offset = array![0.0, 0.0, 0.0];
        let prior = array![2.0, 0.5, 1.25];

        let mut irls = Array1::zeros(3);
        let mut combined = Array1::zeros(3);
        let mut response = Array1::zeros(3);
        let err = compute_poisson_log_irls_weights_in_place(
            &strided3([0.0, 1.0, 5.0]),
            &mu,
            &eta,
            &offset,
            &prior,
            0.25,
            &mut irls,
            &mut combined,
            &mut response,
        )
        .expect_err("non-contiguous y should be rejected");
        assert_linear_algebra_contains(err, "Response vector y");

        let mut irls = Array1::zeros(3);
        let mut combined = Array1::zeros(3);
        let mut response = Array1::zeros(3);
        let err = compute_poisson_log_irls_weights_in_place(
            &y,
            &strided3([0.1, 1.5, 4.0]),
            &eta,
            &offset,
            &prior,
            0.25,
            &mut irls,
            &mut combined,
            &mut response,
        )
        .expect_err("non-contiguous mu should be rejected");
        assert_linear_algebra_contains(err, "Mean vector mu");

        let mut irls = Array1::zeros(3);
        let mut combined = Array1::zeros(3);
        let mut response = Array1::zeros(3);
        let err = compute_poisson_log_irls_weights_in_place(
            &y,
            &mu,
            &strided3([eta[0], eta[1], eta[2]]),
            &offset,
            &prior,
            0.25,
            &mut irls,
            &mut combined,
            &mut response,
        )
        .expect_err("non-contiguous eta should be rejected");
        assert_linear_algebra_contains(err, "Linear predictor eta");

        let mut irls = Array1::zeros(3);
        let mut combined = Array1::zeros(3);
        let mut response = Array1::zeros(3);
        let err = compute_poisson_log_irls_weights_in_place(
            &y,
            &mu,
            &eta,
            &strided3([0.0, 0.0, 0.0]),
            &prior,
            0.25,
            &mut irls,
            &mut combined,
            &mut response,
        )
        .expect_err("non-contiguous offset should be rejected");
        assert_linear_algebra_contains(err, "Offset vector");

        let mut irls = Array1::zeros(3);
        let mut combined = Array1::zeros(3);
        let mut response = Array1::zeros(3);
        let err = compute_poisson_log_irls_weights_in_place(
            &y,
            &mu,
            &eta,
            &offset,
            &strided3([2.0, 0.5, 1.25]),
            0.25,
            &mut irls,
            &mut combined,
            &mut response,
        )
        .expect_err("non-contiguous prior weights should be rejected");
        assert_linear_algebra_contains(err, "Prior weights vector");

        let mut irls_bad = strided3_buffer();
        let mut combined = Array1::zeros(3);
        let mut response = Array1::zeros(3);
        let err = compute_poisson_log_irls_weights_in_place(
            &y,
            &mu,
            &eta,
            &offset,
            &prior,
            0.25,
            &mut irls_bad,
            &mut combined,
            &mut response,
        )
        .expect_err("non-contiguous IRLS buffer should be rejected");
        assert_linear_algebra_contains(err, "IRLS weights buffer");

        let mut irls = Array1::zeros(3);
        let mut combined_bad = strided3_buffer();
        let mut response = Array1::zeros(3);
        let err = compute_poisson_log_irls_weights_in_place(
            &y,
            &mu,
            &eta,
            &offset,
            &prior,
            0.25,
            &mut irls,
            &mut combined_bad,
            &mut response,
        )
        .expect_err("non-contiguous combined buffer should be rejected");
        assert_linear_algebra_contains(err, "Combined weights buffer");

        let mut irls = Array1::zeros(3);
        let mut combined = Array1::zeros(3);
        let mut response_bad = strided3_buffer();
        let err = compute_poisson_log_irls_weights_in_place(
            &y,
            &mu,
            &eta,
            &offset,
            &prior,
            0.25,
            &mut irls,
            &mut combined,
            &mut response_bad,
        )
        .expect_err("non-contiguous working-response buffer should be rejected");
        assert_linear_algebra_contains(err, "Working response buffer");
    }

    #[test]
    fn test_invalid_family_initial_mu_uses_safe_fallback_and_records_warning() {
        let x = Array2::from_shape_vec(
            (5, 2),
            vec![1.0, -2.0, 1.0, -1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 2.0],
        )
        .expect("test setup should be valid");
        let y = array![1.0, 2.0, 2.5, 4.0, 5.0];

        let result = fit_glm_unified(
            &y,
            x.view(),
            &InvalidInitGaussian,
            &IdentityLink,
            &FitConfig::default().with_max_iterations(25),
            None,
            None,
            None,
        )
        .expect("safe initialization should allow fitting to proceed");

        assert!(result.fitted_values.iter().all(|value| value.is_finite()));
        assert!(
            result
                .warnings
                .iter()
                .any(|warning| warning.contains("initial μ values were invalid")),
            "invalid family initializer must leave an audit warning: {:?}",
            result.warnings
        );
    }

    #[test]
    fn test_initial_projection_failure_warns_and_zero_iteration_fit_returns_safe_state() {
        let x = Array2::<f64>::zeros((4, 2));
        let y = array![1.0, 2.0, 3.0, 4.0];
        let config = FitConfig {
            max_iterations: 0,
            skip_covariance: true,
            ..FitConfig::default()
        };

        let result = fit_glm_unified(
            &y,
            x.view(),
            &GaussianFamily,
            &IdentityLink,
            &config,
            None,
            None,
            None,
        )
        .expect("singular initial projection should warn and retain zero coefficients");

        assert_eq!(result.coefficients, Array1::zeros(2));
        assert!(!result.converged);
        assert_eq!(result.solver_status, "max_iterations");
        assert!(
            result
                .warnings
                .iter()
                .any(|warning| warning.contains("Initial coefficient projection failed")),
            "expected initial projection warning, got {:?}",
            result.warnings
        );
    }

    #[test]
    fn test_initial_projection_applies_nonnegative_constraints_before_iteration() {
        let x = Array2::from_shape_vec(
            (5, 2),
            vec![1.0, -2.0, 1.0, -1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 2.0],
        )
        .expect("test setup should be valid");
        let y = array![5.0, 4.0, 3.0, 2.0, 1.0];
        let config = FitConfig {
            max_iterations: 0,
            skip_covariance: true,
            nonneg_indices: vec![1],
            ..FitConfig::default()
        };

        let result = fit_glm_unified(
            &y,
            x.view(),
            &GaussianFamily,
            &IdentityLink,
            &config,
            None,
            None,
            None,
        )
        .expect("zero-iteration constrained fit should return projected initialization");

        assert_eq!(result.coefficients[1], 0.0);
        assert!(result.coefficients[0].is_finite());
    }

    #[test]
    fn test_initial_projection_applies_nonpositive_constraints_before_iteration() {
        let x = Array2::from_shape_vec(
            (5, 2),
            vec![1.0, -2.0, 1.0, -1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 2.0],
        )
        .expect("test setup should be valid");
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let config = FitConfig {
            max_iterations: 0,
            skip_covariance: true,
            nonpos_indices: vec![1],
            ..FitConfig::default()
        };

        let result = fit_glm_unified(
            &y,
            x.view(),
            &GaussianFamily,
            &IdentityLink,
            &config,
            None,
            None,
            None,
        )
        .expect("zero-iteration constrained fit should return projected initialization");

        assert_eq!(result.coefficients[1], 0.0);
        assert!(result.coefficients[0].is_finite());
    }

    #[test]
    fn test_fit_rejects_bad_warm_start_length_before_iteration() {
        let x = Array2::from_shape_vec(
            (5, 2),
            vec![1.0, -2.0, 1.0, -1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 2.0],
        )
        .expect("test setup should be valid");
        let y = array![1.0, 2.0, 2.5, 4.0, 5.0];
        let bad_init = array![0.0];

        let err = fit_glm_unified(
            &y,
            x.view(),
            &GaussianFamily,
            &IdentityLink,
            &FitConfig::default(),
            None,
            None,
            Some(&bad_init),
        )
        .expect_err("warm-start coefficients must match design width");

        assert!(matches!(
            err,
            RustyStatsError::DimensionMismatch {
                expected: 2,
                got: 1,
                ref context
            } if context == "init_coefficients length vs X columns"
        ));
    }

    #[test]
    fn test_standardized_ridge_centered_path_matches_manual_standardized_reference() {
        let x = Array2::from_shape_vec(
            (8, 3),
            vec![
                1.0, 8.0, -3.0, 1.0, 9.5, -2.0, 1.0, 10.0, -1.0, 1.0, 11.0, -0.5, 1.0, 12.5, 0.0,
                1.0, 14.0, 1.0, 1.0, 15.0, 1.5, 1.0, 16.0, 2.0,
            ],
        )
        .expect("test setup should be valid");
        let y = array![1.2, 1.6, 1.9, 2.1, 2.9, 3.3, 3.8, 4.1];
        let init_original = array![0.25, 0.03, -0.02];
        let standardization = Standardization::new(vec![0.0, 12.0, -0.5], vec![1.0, 2.5, 0.75])
            .expect("standardization should validate");
        let regularization = RegularizationConfig::ridge(0.35);
        let config = FitConfig {
            max_iterations: 50,
            regularization: regularization.clone(),
            standardization: Some(standardization.clone()),
            ..FitConfig::default()
        };

        let actual = fit_glm_unified(
            &y,
            x.view(),
            &GaussianFamily,
            &IdentityLink,
            &config,
            None,
            None,
            Some(&init_original),
        )
        .expect("standardized fit should succeed");

        let x_work = standardization
            .standardize_matrix(x.view())
            .expect("manual standardization");
        let init_work = standardization
            .to_standardized_coefficients(&init_original, true)
            .expect("manual init conversion");
        let manual_config = FitConfig {
            max_iterations: 50,
            regularization,
            ..FitConfig::default()
        };
        let manual = fit_glm_unified(
            &y,
            x_work.view(),
            &GaussianFamily,
            &IdentityLink,
            &manual_config,
            None,
            None,
            Some(&init_work),
        )
        .expect("manual standardized fit should succeed");
        let expected_coefficients = standardization
            .to_original_coefficients(&manual.coefficients, true)
            .expect("manual coefficient conversion");
        let expected_covariance = standardization
            .to_original_covariance(&manual.covariance_unscaled, true)
            .expect("manual covariance conversion");

        assert_array1_close(&actual.coefficients, &expected_coefficients, 1e-8);
        assert_array2_close(&actual.covariance_unscaled, &expected_covariance, 1e-8);
        assert_array1_close(&actual.fitted_values, &manual.fitted_values, 1e-8);
        assert!((actual.deviance - manual.deviance).abs() < 1e-8);
    }

    #[test]
    fn test_standardized_ridge_centered_path_without_init_converts_covariance() {
        let x = Array2::from_shape_vec(
            (8, 3),
            vec![
                1.0, 8.0, -3.0, 1.0, 9.5, -2.0, 1.0, 10.0, -1.0, 1.0, 11.0, -0.5, 1.0, 12.5, 0.0,
                1.0, 14.0, 1.0, 1.0, 15.0, 1.5, 1.0, 16.0, 2.0,
            ],
        )
        .expect("test setup should be valid");
        let y = array![1.2, 1.6, 1.9, 2.1, 2.9, 3.3, 3.8, 4.1];
        let standardization = Standardization::new(vec![0.0, 12.0, -0.5], vec![1.0, 2.5, 0.75])
            .expect("standardization should validate");
        let regularization = RegularizationConfig::ridge(0.35);
        let config = FitConfig {
            max_iterations: 50,
            regularization: regularization.clone(),
            standardization: Some(standardization.clone()),
            ..FitConfig::default()
        };

        let actual = fit_glm_unified(
            &y,
            x.view(),
            &GaussianFamily,
            &IdentityLink,
            &config,
            None,
            None,
            None,
        )
        .expect("standardized fit should succeed without a warm start");

        let x_work = standardization
            .standardize_matrix(x.view())
            .expect("manual standardization");
        let manual = fit_glm_unified(
            &y,
            x_work.view(),
            &GaussianFamily,
            &IdentityLink,
            &FitConfig {
                max_iterations: 50,
                regularization,
                ..FitConfig::default()
            },
            None,
            None,
            None,
        )
        .expect("manual standardized fit should succeed without a warm start");
        let expected_coefficients = standardization
            .to_original_coefficients(&manual.coefficients, true)
            .expect("manual coefficient conversion");
        let expected_covariance = standardization
            .to_original_covariance(&manual.covariance_unscaled, true)
            .expect("manual covariance conversion");

        assert_array1_close(&actual.coefficients, &expected_coefficients, 1e-8);
        assert_array2_close(&actual.covariance_unscaled, &expected_covariance, 1e-8);
        assert_array1_close(&actual.fitted_values, &manual.fitted_values, 1e-8);
        assert!((actual.deviance - manual.deviance).abs() < 1e-8);
    }

    #[test]
    fn test_standardized_ridge_scale_only_fast_path_matches_manual_reference() {
        let x = Array2::from_shape_vec(
            (7, 3),
            vec![
                1.0, -3.0, 0.5, 1.0, -2.0, 1.5, 1.0, -1.0, 1.0, 1.0, 0.0, 2.0, 1.0, 1.0, 2.5, 1.0,
                2.0, 3.5, 1.0, 3.0, 4.0,
            ],
        )
        .expect("test setup should be valid");
        let y = array![-0.4, 0.2, 0.7, 1.0, 1.5, 1.9, 2.4];
        let standardization = Standardization::new(vec![0.0, 0.0, 0.0], vec![1.0, 2.0, 0.5])
            .expect("standardization should validate");
        let regularization = RegularizationConfig::ridge(0.7);
        let config = FitConfig {
            max_iterations: 50,
            regularization: regularization.clone(),
            standardization: Some(standardization.clone()),
            skip_covariance: true,
            ..FitConfig::default()
        };

        let actual = fit_glm_unified(
            &y,
            x.view(),
            &GaussianFamily,
            &IdentityLink,
            &config,
            None,
            None,
            None,
        )
        .expect("scale-only fast path should succeed");

        let x_work = standardization
            .standardize_matrix(x.view())
            .expect("manual standardization");
        let manual_config = FitConfig {
            max_iterations: 50,
            regularization,
            skip_covariance: true,
            ..FitConfig::default()
        };
        let manual = fit_glm_unified(
            &y,
            x_work.view(),
            &GaussianFamily,
            &IdentityLink,
            &manual_config,
            None,
            None,
            None,
        )
        .expect("manual standardized fit should succeed");
        let expected_coefficients = standardization
            .to_original_coefficients(&manual.coefficients, true)
            .expect("manual coefficient conversion");

        assert_array1_close(&actual.coefficients, &expected_coefficients, 1e-8);
        assert_eq!(actual.covariance_unscaled.dim(), (x.ncols(), x.ncols()));
        assert!(actual.covariance_unscaled.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn test_standardization_validation_and_lasso_dispatch_from_unified_api() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, -2.0, 1.0, -1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0],
        )
        .expect("test setup should be valid");
        let y = array![-2.0, -0.8, 0.1, 1.2, 2.1, 3.3];
        let bad_standardization =
            Standardization::new(vec![0.0], vec![1.0]).expect("one-column metadata is valid alone");
        let bad_config = FitConfig {
            regularization: RegularizationConfig::ridge(0.1),
            standardization: Some(bad_standardization),
            ..FitConfig::default()
        };
        let err = fit_glm_unified(
            &y,
            x.view(),
            &GaussianFamily,
            &IdentityLink,
            &bad_config,
            None,
            None,
            None,
        )
        .expect_err("standardization metadata must match design width");
        assert!(matches!(err, RustyStatsError::DimensionMismatch { .. }));

        let lasso_config = FitConfig {
            max_iterations: 50,
            regularization: RegularizationConfig::lasso(0.05),
            skip_covariance: true,
            ..FitConfig::default()
        };
        let lasso = fit_glm_unified(
            &y,
            x.view(),
            &GaussianFamily,
            &IdentityLink,
            &lasso_config,
            None,
            None,
            None,
        )
        .expect("unified API should dispatch L1 penalties to coordinate descent");
        assert_eq!(lasso.penalty.l1_penalty(), 0.05);
        assert!(lasso.coefficients.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn test_sparse_cache_public_fit_wrapper_matches_uncached_and_ignores_incompatible_cache() {
        let n = 48usize;
        let p = 20usize;
        let mut x = Array2::<f64>::zeros((n, p));
        for row in 0..n {
            x[[row, 0]] = 1.0;
            x[[row, 1 + row % (p - 1)]] = 0.4 + (row % 5) as f64 * 0.2;
            x[[row, 1 + (row * 7 + 3) % (p - 1)]] += (row as f64 * 0.11).sin();
        }
        let beta: Array1<f64> = (0..p).map(|j| (j as f64 * 0.13).cos() * 0.1).collect();
        let y = x.dot(&beta);
        let cache = manual_sparse_cache(&x);
        let config = FitConfig {
            max_iterations: 5,
            regularization: RegularizationConfig::ridge(0.5),
            skip_covariance: true,
            ..FitConfig::default()
        };

        let uncached = fit_glm_unified(
            &y,
            x.view(),
            &GaussianFamily,
            &IdentityLink,
            &config,
            None,
            None,
            None,
        )
        .expect("uncached fit should succeed");
        let cached = fit_glm_unified_with_sparse_cache(
            &y,
            x.view(),
            &GaussianFamily,
            &IdentityLink,
            &config,
            None,
            None,
            None,
            Some(&cache),
        )
        .expect("compatible cache should be used successfully");

        assert_array1_close(&cached.coefficients, &uncached.coefficients, 1e-10);
        assert_array1_close(&cached.fitted_values, &uncached.fitted_values, 1e-10);

        let x_other = x.clone();
        let incompatible = fit_glm_unified_with_sparse_cache(
            &y,
            x_other.view(),
            &GaussianFamily,
            &IdentityLink,
            &config,
            None,
            None,
            None,
            Some(&cache),
        )
        .expect("incompatible cache should be ignored, not trusted");
        assert_array1_close(&incompatible.coefficients, &uncached.coefficients, 1e-10);
    }

    #[test]
    fn test_noncontiguous_matrix_paths_and_sparse_cache_xtwx_contracts() {
        let base = Array2::from_shape_vec(
            (4, 4),
            vec![
                1.0, 99.0, 2.0, 88.0, 1.0, 77.0, 3.0, 66.0, 1.0, 55.0, 4.0, 44.0, 1.0, 33.0, 5.0,
                22.0,
            ],
        )
        .expect("test setup should be valid");
        let x = base.slice(s![.., ..;2]);
        let coefficients = array![0.5, -0.25];
        let dot = matrix_vector_dot(x, &coefficients);
        assert_array1_close(&dot, &x.dot(&coefficients), 1e-12);

        let w = array![1.0, 0.5, 2.0, 1.5];
        let actual_xtwx = compute_xtwx(x, &w);
        let mut expected_xtwx = Array2::<f64>::zeros((x.ncols(), x.ncols()));
        for row in 0..x.nrows() {
            for i in 0..x.ncols() {
                for j in 0..x.ncols() {
                    expected_xtwx[[i, j]] += w[row] * x[[row, i]] * x[[row, j]];
                }
            }
        }
        assert_array2_close(&actual_xtwx, &expected_xtwx, 1e-12);

        let z = array![1.0, 2.0, 3.0, 4.0];
        let err = compute_xtwx_xtwz(x, &z, &w)
            .expect_err("profiled WLS kernel requires contiguous design matrices");
        assert_linear_algebra_contains(err, "Design matrix X");

        let contiguous_x = x.as_standard_layout().to_owned();
        let strided_w = Array1::from_shape_vec(
            (4usize).strides(2),
            vec![1.0, 99.0, 0.5, 99.0, 2.0, 99.0, 1.5],
        )
        .expect("valid strided weight vector");
        assert!(strided_w.as_slice().is_none());
        let err = compute_xtwx_xtwz(contiguous_x.view(), &z, &strided_w)
            .expect_err("profiled WLS kernel requires contiguous weights");
        assert_linear_algebra_contains(err, "Weight vector W");

        let strided_z = Array1::from_shape_vec(
            (4usize).strides(2),
            vec![1.0, 99.0, 2.0, 99.0, 3.0, 99.0, 4.0],
        )
        .expect("valid strided working-response vector");
        assert!(strided_z.as_slice().is_none());
        let err = compute_xtwx_xtwz(contiguous_x.view(), &strided_z, &w)
            .expect_err("profiled WLS kernel requires contiguous working responses");
        assert_linear_algebra_contains(err, "Working response Z");

        let n = 32usize;
        let p = 18usize;
        let mut sparse = Array2::<f64>::zeros((n, p));
        for row in 0..n {
            sparse[[row, row % p]] = 1.0;
            sparse[[row, (row * 5 + 1) % p]] += 0.25;
        }
        let sparse_cache = manual_sparse_cache(&sparse);
        let sparse_w: Array1<f64> = (0..n).map(|idx| 0.5 + (idx % 4) as f64).collect();
        let cached_xtwx =
            compute_xtwx_with_sparse_cache(sparse.view(), &sparse_w, Some(&sparse_cache))
                .expect("compatible cache should compute X'WX");
        let expected_cached = compute_xtwx(sparse.view(), &sparse_w);
        assert_array2_close(&cached_xtwx, &expected_cached, 1e-12);

        let err = compute_xtwx_with_sparse_cache(
            sparse.view(),
            &Array1::ones(n - 1),
            Some(&sparse_cache),
        )
        .expect_err("weight length mismatch should be reported");
        assert!(matches!(err, RustyStatsError::DimensionMismatch { .. }));

        let mut sparse_w_storage = Vec::with_capacity(2 * n - 1);
        for (idx, value) in sparse_w.iter().copied().enumerate() {
            sparse_w_storage.push(value);
            if idx + 1 < n {
                sparse_w_storage.push(99.0);
            }
        }
        let sparse_w_strided = Array1::from_shape_vec((n).strides(2), sparse_w_storage)
            .expect("valid full-length strided sparse weights");
        let err =
            compute_xtwx_with_sparse_cache(sparse.view(), &sparse_w_strided, Some(&sparse_cache))
                .expect_err("compatible sparse cache requires contiguous weights");
        assert_linear_algebra_contains(err, "Weight vector W");

        let sparse_clone = sparse.clone();
        let fallback =
            compute_xtwx_with_sparse_cache(sparse_clone.view(), &sparse_w, Some(&sparse_cache))
                .expect("incompatible cache should fall back to uncached computation");
        assert_array2_close(
            &fallback,
            &compute_xtwx(sparse_clone.view(), &sparse_w),
            1e-12,
        );
    }

    #[test]
    fn test_weighted_least_squares_penalty_matrix_and_precomputed_contracts() {
        let x = Array2::from_shape_vec(
            (5, 2),
            vec![1.0, -2.0, 1.0, -1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 2.0],
        )
        .expect("test setup should be valid");
        let z = array![-1.0, 0.0, 0.5, 1.5, 2.0];
        let w = array![1.0, 0.7, 1.3, 0.8, 1.1];
        let penalty =
            Array2::from_shape_vec((2, 2), vec![0.25, 0.05, 0.05, 0.75]).expect("penalty");

        let (direct_coef, direct_cov) =
            solve_weighted_least_squares_with_penalty_matrix(x.view(), &z, &w, &penalty, false)
                .expect("penalized WLS should solve");
        let (xtx, xtz) = compute_xtwx_xtwz(x.view(), &z, &w).expect("normal equations");
        let (pre_coef, pre_cov) =
            solve_wls_from_precomputed(&xtx, &xtz, &penalty, false).expect("precomputed solve");
        assert_array1_close(&direct_coef, &pre_coef, 1e-12);
        assert_array2_close(&direct_cov, &pre_cov, 1e-12);

        let (_, skipped_cov) =
            solve_weighted_least_squares_with_penalty_matrix(x.view(), &z, &w, &penalty, true)
                .expect("skip covariance solve should still compute coefficients");
        assert_eq!(skipped_cov.dim(), (2, 2));
        assert!(skipped_cov.iter().all(|value| *value == 0.0));

        let bad_penalty = Array2::eye(3);
        let err =
            solve_weighted_least_squares_with_penalty_matrix(x.view(), &z, &w, &bad_penalty, true)
                .expect_err("penalty matrix dimensions must match X columns");
        assert!(matches!(err, RustyStatsError::DimensionMismatch { .. }));

        let factors_err = solve_weighted_least_squares_penalized(
            x.view(),
            &z,
            &w,
            1.0,
            Some(&[1.0]),
            false,
            true,
            None,
            None,
        )
        .expect_err("L2 penalty factors must match X columns");
        assert!(matches!(
            factors_err,
            RustyStatsError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn test_cholesky_lu_fallback_and_pcg_edge_contracts() {
        let indefinite = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 1.0, 0.0]);
        let rhs = DVector::from_vec(vec![2.0, 3.0]);
        let (coef, cov) =
            cholesky_solve(indefinite.clone(), &rhs, false).expect("LU fallback should solve");
        assert_array1_close(&coef, &array![3.0, 2.0], 1e-12);
        assert!((cov[[0, 1]] - 1.0).abs() < 1e-12);
        assert!((cov[[1, 0]] - 1.0).abs() < 1e-12);

        let (_, skipped_cov) =
            cholesky_solve(indefinite.clone(), &rhs, true).expect("LU fallback should solve");
        assert_eq!(skipped_cov.dim(), (2, 2));
        assert!(skipped_cov.iter().all(|value| *value == 0.0));
        let coef_only =
            cholesky_solve_coefficients(indefinite.clone(), &rhs).expect("LU coefficient solve");
        assert_array1_close(&coef_only, &array![3.0, 2.0], 1e-12);
        assert!(cholesky_solve_spd_coefficients(indefinite, &rhs).is_err());
        assert!(cholesky_solve(DMatrix::zeros(2, 2), &rhs, false).is_err());

        assert!(!ridge_system_is_positive_definite_fast_path(
            &array![0.0, 0.0],
            1.0,
            None,
            true,
            2
        ));
        assert!(!ridge_system_is_positive_definite_fast_path(
            &array![1.0, 1.0],
            0.0,
            None,
            true,
            2
        ));
        assert!(!ridge_system_is_positive_definite_fast_path(
            &array![1.0, 1.0],
            1.0,
            Some(&[1.0, 0.0]),
            true,
            2
        ));

        let x = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 1.0, 1.0, 1.0, 2.0])
            .expect("test setup should be valid");
        let z_zero = Array1::zeros(3);
        let w = Array1::ones(3);
        let mut profile = WLSSolveProfile::default();
        let pcg_zero_rhs = solve_weighted_least_squares_pcg(
            x.view(),
            &z_zero,
            &w,
            1.0,
            None,
            true,
            None,
            &mut profile,
        )
        .expect("PCG should not error on zero RHS")
        .expect("zero RHS should converge immediately");
        assert_array1_close(&pcg_zero_rhs, &array![0.0, 0.0], 1e-12);

        let mut profile = WLSSolveProfile::default();
        let bad_init = array![0.0];
        let no_pcg = solve_weighted_least_squares_pcg(
            x.view(),
            &array![1.0, 2.0, 3.0],
            &w,
            1.0,
            None,
            true,
            Some(&bad_init),
            &mut profile,
        )
        .expect("bad warm start length should degrade to direct solve");
        assert!(no_pcg.is_none());

        let mut profile = WLSSolveProfile::default();
        let noncontiguous = Array2::from_shape_vec(
            (3, 4),
            vec![
                1.0, 99.0, 0.0, 88.0, 1.0, 77.0, 1.0, 66.0, 1.0, 55.0, 2.0, 44.0,
            ],
        )
        .expect("test setup should be valid");
        let no_pcg = solve_weighted_least_squares_pcg(
            noncontiguous.slice(s![.., ..;2]),
            &array![1.0, 2.0, 3.0],
            &w,
            1.0,
            None,
            true,
            None,
            &mut profile,
        )
        .expect("non-contiguous design should degrade to direct solve");
        assert!(no_pcg.is_none());

        let mut profile = WLSSolveProfile::default();
        let no_pcg = solve_weighted_least_squares_pcg(
            x.view(),
            &strided3([1.0, 2.0, 3.0]),
            &w,
            1.0,
            None,
            true,
            None,
            &mut profile,
        )
        .expect("non-contiguous response should degrade to direct solve");
        assert!(no_pcg.is_none());

        let mut profile = WLSSolveProfile::default();
        let no_pcg = solve_weighted_least_squares_pcg(
            x.view(),
            &array![1.0, 2.0, 3.0],
            &strided3([1.0, 1.0, 1.0]),
            1.0,
            None,
            true,
            None,
            &mut profile,
        )
        .expect("non-contiguous weights should degrade to direct solve");
        assert!(no_pcg.is_none());

        let mut profile = WLSSolveProfile::default();
        let no_pcg = solve_weighted_least_squares_pcg(
            x.view(),
            &array![f64::INFINITY, 2.0, 3.0],
            &w,
            1.0,
            None,
            true,
            None,
            &mut profile,
        )
        .expect("non-finite PCG inputs should degrade to direct solve");
        assert!(no_pcg.is_none());

        let mut profile = WLSSolveProfile::default();
        let negative_weights = array![-1.0, -1.0];
        let identity_x = Array2::eye(2);
        let no_pcg = solve_weighted_least_squares_pcg(
            identity_x.view(),
            &array![1.0, 1.0],
            &negative_weights,
            0.0,
            None,
            true,
            None,
            &mut profile,
        )
        .expect("invalid curvature should degrade to direct solve");
        assert!(no_pcg.is_none());

        assert!(cholesky_solve_coefficients(DMatrix::zeros(2, 2), &rhs).is_err());
    }

    #[test]
    fn test_compute_xtwx_sparse_kernel_matches_dense_formula() {
        let n = 64;
        let p = 24;
        let mut x = Array2::<f64>::zeros((n, p));
        for row in 0..n {
            if row % 11 == 0 {
                continue;
            }
            for (slot, col) in [0usize, 3, 8, 13, 21].iter().copied().enumerate() {
                x[[row, col]] = (1.0 + (row % 4) as f64) * (slot as f64 + 0.5);
            }
        }
        let w: Array1<f64> = (0..n).map(|i| 0.75 + (i % 7) as f64 * 0.05).collect();

        assert!(should_use_sparse_xtwx_kernel(
            x.as_slice().expect("test matrix should be contiguous"),
            n,
            p
        ));

        let actual = compute_xtwx(x.view(), &w);
        let mut expected = Array2::<f64>::zeros((p, p));
        for row in 0..n {
            for i in 0..p {
                for j in 0..p {
                    expected[[i, j]] += w[row] * x[[row, i]] * x[[row, j]];
                }
            }
        }

        for i in 0..p {
            for j in 0..p {
                assert!((actual[[i, j]] - expected[[i, j]]).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn test_compute_xtwx_xtwz_dense_kernel_matches_formula() {
        let n = 48;
        let p = 20;
        let mut x = Array2::<f64>::zeros((n, p));
        for row in 0..n {
            for col in 0..p {
                x[[row, col]] = ((row + 1) as f64 * 0.03 + (col + 2) as f64 * 0.07).sin();
            }
        }
        let z: Array1<f64> = (0..n).map(|i| 0.5 + (i as f64 * 0.11).cos()).collect();
        let w: Array1<f64> = (0..n).map(|i| 0.75 + (i % 5) as f64 * 0.08).collect();

        assert!(!should_use_sparse_xtwx_kernel(
            x.as_slice().expect("test matrix should be contiguous"),
            n,
            p
        ));

        let (actual_xtx, actual_xtz) =
            compute_xtwx_xtwz(x.view(), &z, &w).expect("dense kernel should succeed");
        let mut expected_xtx = Array2::<f64>::zeros((p, p));
        let mut expected_xtz = Array1::<f64>::zeros(p);
        for row in 0..n {
            for i in 0..p {
                expected_xtz[i] += w[row] * x[[row, i]] * z[row];
                for j in 0..p {
                    expected_xtx[[i, j]] += w[row] * x[[row, i]] * x[[row, j]];
                }
            }
        }

        for i in 0..p {
            assert!((actual_xtz[i] - expected_xtz[i]).abs() < 1e-12);
            for j in 0..p {
                assert!((actual_xtx[(i, j)] - expected_xtx[[i, j]]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_profiled_dense_gram_subtimers_are_recorded_when_enabled() {
        let _env_lock = ENV_LOCK.lock().expect("env lock should not be poisoned");
        let _guard = EnvVarGuard::set("RUSTYSTATS_PROFILE_GRAM_SUBTIMERS", "1");

        let n = 40;
        let p = 18;
        let mut x = Array2::<f64>::zeros((n, p));
        for row in 0..n {
            for col in 0..p {
                x[[row, col]] = 0.2 + ((row + 3) as f64 * (col + 5) as f64 * 0.017).sin();
            }
        }
        let z: Array1<f64> = (0..n).map(|i| 0.75 + (i as f64 * 0.03).cos()).collect();
        let w: Array1<f64> = (0..n).map(|i| 0.8 + (i % 9) as f64 * 0.04).collect();

        let (xtx, xtz, profile) =
            compute_xtwx_xtwz_profiled(x.view(), &z, &w).expect("profiled dense kernel");

        assert_eq!(xtx.shape(), (p, p));
        assert_eq!(xtz.len(), p);
        assert!(profile.local_init_seconds >= 0.0);
        assert!(profile.pairwise_accum_seconds >= 0.0);
        assert!(profile.reduce_seconds >= 0.0);
        assert!(profile.materialize_seconds >= 0.0);
    }

    #[test]
    fn test_compute_xtwx_xtwz_sparse_kernel_matches_formula() {
        let n = 72;
        let p = 32;
        let mut x = Array2::<f64>::zeros((n, p));
        for row in 0..n {
            if row % 13 == 0 {
                continue;
            }
            for (slot, col) in [0usize, 4, 9, 17, 25].iter().copied().enumerate() {
                x[[row, col]] = (row as f64 * 0.13 + slot as f64).cos();
            }
        }
        let z: Array1<f64> = (0..n).map(|i| 0.25 + (i as f64 * 0.17).sin()).collect();
        let w: Array1<f64> = (0..n).map(|i| 0.5 + (i % 11) as f64 * 0.03).collect();

        assert!(should_use_sparse_xtwx_kernel(
            x.as_slice().expect("test matrix should be contiguous"),
            n,
            p
        ));

        let (actual_xtx, actual_xtz) =
            compute_xtwx_xtwz(x.view(), &z, &w).expect("sparse kernel should succeed");
        let mut expected_xtx = Array2::<f64>::zeros((p, p));
        let mut expected_xtz = Array1::<f64>::zeros(p);
        for row in 0..n {
            for i in 0..p {
                expected_xtz[i] += w[row] * x[[row, i]] * z[row];
                for j in 0..p {
                    expected_xtx[[i, j]] += w[row] * x[[row, i]] * x[[row, j]];
                }
            }
        }

        for i in 0..p {
            assert!((actual_xtz[i] - expected_xtz[i]).abs() < 1e-12);
            for j in 0..p {
                assert!((actual_xtx[(i, j)] - expected_xtx[[i, j]]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_sparse_row_cache_matches_uncached_sparse_kernel_and_dot() {
        let n = 72;
        let p = 32;
        let mut x = Array2::<f64>::zeros((n, p));
        for row in 0..n {
            if row % 13 == 0 {
                continue;
            }
            for (slot, col) in [0usize, 4, 9, 17, 25].iter().copied().enumerate() {
                x[[row, col]] = (row as f64 * 0.13 + slot as f64).cos();
            }
        }
        let z: Array1<f64> = (0..n).map(|i| 0.25 + (i as f64 * 0.17).sin()).collect();
        let w: Array1<f64> = (0..n).map(|i| 0.5 + (i % 11) as f64 * 0.03).collect();
        let coef: Array1<f64> = (0..p).map(|j| (j as f64 * 0.07).sin()).collect();

        let x_slice = x.as_slice().expect("test matrix should be contiguous");
        let mut offsets = Vec::with_capacity(n + 1);
        let mut indices = Vec::new();
        let mut values = Vec::new();
        offsets.push(0);
        for row in 0..n {
            let row_start = row * p;
            for col in 0..p {
                let value = x_slice[row_start + col];
                if value != 0.0 {
                    indices.push(col as u32);
                    values.push(value);
                }
            }
            offsets.push(indices.len());
        }
        let cache = SparseRowCache {
            n,
            p,
            data_ptr: x_slice.as_ptr() as usize,
            offsets,
            indices,
            values,
            packed_offsets: packed_upper_offsets(p),
        };

        let (actual_xtx, actual_xtz, _) = compute_xtwx_xtwz_sparse_cached(&cache, &z, &w, false)
            .expect("cached kernel should succeed");
        let (_, _, profiled) = compute_xtwx_xtwz_sparse_cached(&cache, &z, &w, true)
            .expect("profiled cached kernel should succeed");
        assert!(profiled.local_init_seconds >= 0.0);
        assert!(profiled.pairwise_accum_seconds >= 0.0);
        assert!(profiled.materialize_seconds >= 0.0);
        let (expected_xtx, expected_xtz) =
            compute_xtwx_xtwz(x.view(), &z, &w).expect("sparse kernel should succeed");
        for i in 0..p {
            assert!((actual_xtz[i] - expected_xtz[i]).abs() < 1e-12);
            for j in 0..p {
                assert!((actual_xtx[(i, j)] - expected_xtx[(i, j)]).abs() < 1e-12);
            }
        }

        let actual_dot = matrix_vector_dot_cached(x.view(), &coef, Some(&cache));
        let expected_dot = x.dot(&coef);
        for row in 0..n {
            assert!((actual_dot[row] - expected_dot[row]).abs() < 1e-12);
        }

        let cached_xtwx = compute_xtwx_with_sparse_cache(x.view(), &w, Some(&cache))
            .expect("compatible cache with empty rows should compute X'WX");
        let expected_xtwx = compute_xtwx(x.view(), &w);
        assert_array2_close(&cached_xtwx, &expected_xtwx, 1e-10);

        let err = compute_xtwx_xtwz_sparse_cached(&cache, &Array1::zeros(n - 1), &w, false)
            .expect_err("cached normal equations require matching rows");
        assert!(matches!(err, RustyStatsError::DimensionMismatch { .. }));

        let mut z_storage = Vec::with_capacity(2 * n - 1);
        for (idx, value) in z.iter().copied().enumerate() {
            z_storage.push(value);
            if idx + 1 < n {
                z_storage.push(99.0);
            }
        }
        let z_strided = Array1::from_shape_vec((n).strides(2), z_storage)
            .expect("valid full-length strided working-response vector");
        let err = compute_xtwx_xtwz_sparse_cached(&cache, &z_strided, &w, false)
            .expect_err("cached normal equations require contiguous working responses");
        assert_linear_algebra_contains(err, "Working response Z");

        let mut w_storage = Vec::with_capacity(2 * n - 1);
        for (idx, value) in w.iter().copied().enumerate() {
            w_storage.push(value);
            if idx + 1 < n {
                w_storage.push(99.0);
            }
        }
        let w_strided = Array1::from_shape_vec((n).strides(2), w_storage)
            .expect("valid full-length strided weight vector");
        let err = compute_xtwx_xtwz_sparse_cached(&cache, &z, &w_strided, false)
            .expect_err("cached normal equations require contiguous weights");
        assert_linear_algebra_contains(err, "Weight vector W");
    }

    #[test]
    fn test_sparse_cache_builder_thresholds_and_core_kernel_edge_paths() {
        assert_eq!(sampled_density(&[], 0, 16), 0.0);
        assert_eq!(sparse_xtwx_chunk_count(1, 512), 1);
        assert!(sparse_xtwx_chunk_count(64, 512) <= SPARSE_XTWX_THREAD_CAP);

        let x_empty = Array2::<f64>::zeros((0, 20));
        let z_empty = Array1::<f64>::zeros(0);
        let w_empty = Array1::<f64>::zeros(0);
        let (xtx, xtz) =
            compute_xtwx_xtwz(x_empty.view(), &z_empty, &w_empty).expect("empty dense kernel");
        assert_eq!(xtx.shape(), (20, 20));
        assert_eq!(xtz.len(), 20);

        let (sparse_xtx, sparse_xtz, sparse_profile) =
            compute_xtwx_xtwz_sparse(&[], &[], &[], 0, 20, true).expect("empty sparse kernel");
        assert_eq!(sparse_xtx.shape(), (20, 20));
        assert_eq!(sparse_xtz.len(), 20);
        assert!(sparse_profile.local_init_seconds >= 0.0);
        assert!(sparse_profile.materialize_seconds >= 0.0);

        let n = 625_000usize;
        let p = 16usize;
        let mut x = Array2::<f64>::zeros((n, p));
        for row in 0..n {
            x[[row, row % p]] = 1.0 + (row % 3) as f64;
        }
        let cache = build_sparse_row_cache_if_beneficial(x.view())
            .expect("large sparse design should build a row cache");
        assert!(cache.is_compatible_with(x.view()));
        assert_eq!(cache.offsets.len(), n + 1);
        assert_eq!(cache.indices.len(), n);
        assert_eq!(cache.values.len(), n);

        let coefficients: Array1<f64> = (0..p).map(|j| j as f64 * 0.25).collect();
        let cached_dot = matrix_vector_dot_cached(x.view(), &coefficients, Some(&cache));
        for row in [0usize, 1, 15, 16, n - 1] {
            assert!((cached_dot[row] - x.row(row).dot(&coefficients)).abs() < 1e-12);
        }
    }

    #[test]
    fn test_pcg_ridge_wls_matches_direct_solve() {
        let n = 80;
        let p = 24;
        let mut x = Array2::<f64>::zeros((n, p));
        for row in 0..n {
            x[[row, 0]] = 1.0;
            let j1 = 1 + row % (p - 1);
            let j2 = 1 + (row * 5 + 3) % (p - 1);
            x[[row, j1]] = 0.5 + (row % 7) as f64 * 0.1;
            x[[row, j2]] += (row as f64 * 0.09).sin();
        }
        let z: Array1<f64> = (0..n).map(|i| 0.25 + (i as f64 * 0.07).cos()).collect();
        let w: Array1<f64> = (0..n).map(|i| 1.0 + (i % 5) as f64 * 0.05).collect();
        let l2_penalty = 10.0;

        let (direct, _, _) = solve_weighted_least_squares_penalized(
            x.view(),
            &z,
            &w,
            l2_penalty,
            None,
            false,
            true,
            None,
            None,
        )
        .expect("direct ridge WLS should solve");
        let mut profile = WLSSolveProfile::default();
        let pcg = solve_weighted_least_squares_pcg(
            x.view(),
            &z,
            &w,
            l2_penalty,
            None,
            false,
            None,
            &mut profile,
        )
        .expect("PCG should not error")
        .expect("PCG should converge on a small ridge system");

        for (actual, expected) in pcg.iter().zip(direct.iter()) {
            assert!((actual - expected).abs() < 1e-5);
        }
        assert!(profile.gram_seconds >= 0.0);
        assert!(profile.solve_seconds >= 0.0);
    }

    #[test]
    fn test_sparse_scan_pcg_helpers_match_dense_references() {
        let n = 4usize;
        let p = 3usize;
        let x = Array2::from_shape_vec(
            (n, p),
            vec![
                1.0, 0.0, 2.0, //
                0.0, 3.0, 0.0, //
                1.0, -1.0, 0.5, //
                0.0, 0.0, 0.0,
            ],
        )
        .expect("test setup should be valid");
        let z = array![2.0, -1.0, 0.5, 99.0];
        let w = array![1.0, 0.5, 2.0, 3.0];
        let beta = vec![0.25, -0.5, 0.75];

        let x_slice = x.as_slice().expect("test matrix should be contiguous");
        let z_slice = z.as_slice().expect("test vector should be contiguous");
        let w_slice = w.as_slice().expect("test vector should be contiguous");
        let (rhs, diag) =
            compute_xtwz_and_weighted_diag_sparse_scan(x_slice, z_slice, w_slice, n, p);

        let mut expected_rhs = vec![0.0; p];
        let mut expected_diag = vec![0.0; p];
        let mut expected_matvec = vec![0.0; p];
        for row in 0..n {
            let dot = (0..p).map(|col| x[[row, col]] * beta[col]).sum::<f64>();
            for col in 0..p {
                expected_rhs[col] += x[[row, col]] * w[row] * z[row];
                expected_diag[col] += w[row] * x[[row, col]] * x[[row, col]];
                expected_matvec[col] += x[[row, col]] * w[row] * dot;
            }
        }
        expected_matvec[1] += 0.4 * 2.0 * beta[1];
        expected_matvec[2] += 0.4 * 3.0 * beta[2];

        for col in 0..p {
            assert!((rhs[col] - expected_rhs[col]).abs() < 1e-12);
            assert!((diag[col] - expected_diag[col]).abs() < 1e-12);
        }

        let actual_matvec = weighted_normal_matvec_sparse_scan(
            x_slice,
            w_slice,
            &beta,
            n,
            p,
            0.4,
            Some(&[1.0, 2.0, 3.0]),
            false,
        );
        for col in 0..p {
            assert!((actual_matvec[col] - expected_matvec[col]).abs() < 1e-12);
        }
    }

    #[test]
    fn test_matrix_vector_dot_small_strided_and_large_parallel_paths() {
        let x = Array2::from_shape_vec((3, 3), vec![1.0, 0.0, 2.0, 0.5, 1.0, 0.0, 2.0, 1.0, 1.0])
            .expect("test setup should be valid");
        let coefficients = array![0.5, -1.0, 2.0];
        assert_array1_close(
            &matrix_vector_dot(x.view(), &coefficients),
            &x.dot(&coefficients),
            1e-12,
        );

        let strided_coefficients = strided3([0.5, -1.0, 2.0]);
        assert_array1_close(
            &matrix_vector_dot(x.view(), &strided_coefficients),
            &x.dot(&strided_coefficients),
            1e-12,
        );

        let cache = manual_sparse_cache(&x);
        assert_array1_close(
            &matrix_vector_dot_cached(x.view(), &strided_coefficients, Some(&cache)),
            &x.dot(&strided_coefficients),
            1e-12,
        );

        let n = 1024usize;
        let p = 1024usize;
        let large_x = Array2::from_shape_fn((n, p), |(row, col)| {
            (((row * 31 + col * 17) % 23) as f64 - 11.0) * 0.001
        });
        let large_coef: Array1<f64> = (0..p).map(|j| ((j % 19) as f64 - 9.0) * 0.01).collect();
        let actual = matrix_vector_dot(large_x.view(), &large_coef);
        let expected = large_x.dot(&large_coef);
        for row in [0usize, 1, 127, 511, 1023] {
            assert!((actual[row] - expected[row]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_final_covariance_matches_gaussian_closed_form() {
        let x = Array2::from_shape_vec(
            (5, 2),
            vec![1.0, -2.0, 1.0, -1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 2.0],
        )
        .expect("test setup should be valid");
        let y = array![-3.0, -1.0, 1.0, 3.0, 5.0];

        let result = fit_glm_unified(
            &y,
            x.view(),
            &GaussianFamily,
            &IdentityLink,
            &FitConfig::default(),
            None,
            None,
            None,
        )
        .expect("fit should not error");

        // X'X is diagonal here: [[5, 0], [0, 10]], so the unscaled
        // covariance has a simple closed-form inverse.
        assert!((result.covariance_unscaled[[0, 0]] - 0.2).abs() < 1e-12);
        assert!(result.covariance_unscaled[[0, 1]].abs() < 1e-12);
        assert!(result.covariance_unscaled[[1, 0]].abs() < 1e-12);
        assert!((result.covariance_unscaled[[1, 1]] - 0.1).abs() < 1e-12);
    }

    #[test]
    fn test_singular_design_fails_closed() {
        // Fail-closed contract (pre-v0.8.14, restored): a rank-deficient
        // design must produce a loud error, never a "successful" fit whose
        // covariance is all zeros (zero SEs read as infinite significance).
        // The v0.8.14 refactor briefly converted the final-extraction failure
        // into a soft success carrying Array2::zeros — this pins the
        // fit-level contract for the singular class.
        let n = 30usize;
        let mut xv = Vec::with_capacity(n * 3);
        let mut yv = Vec::with_capacity(n);
        for i in 0..n {
            let xi = (i as f64) / (n as f64);
            xv.push(1.0);
            xv.push(xi);
            xv.push(xi); // exact duplicate column -> singular X'WX
            yv.push(0.5 + 0.3 * xi + if i % 2 == 0 { 0.05 } else { -0.05 });
        }
        let x = Array2::from_shape_vec((n, 3), xv).expect("test setup should be valid");
        let y = Array1::from_vec(yv);
        let result = fit_glm_unified(
            &y,
            x.view(),
            &GaussianFamily,
            &IdentityLink,
            &FitConfig::default(),
            None,
            None,
            None,
        );
        let err = result.expect_err("singular design must fail loudly, not fit");
        let msg = err.to_string();
        assert!(
            msg.contains("singular") || msg.contains("multicollinearity"),
            "error should point at the singular design, got: {msg}"
        );
    }

    #[test]
    fn test_computed_covariance_is_never_all_zeros() {
        // Companion canary to the fail-closed contract: whenever covariance
        // is computed (skip_covariance = false), a successful fit must carry a
        // usable covariance — an all-zeros matrix can only come from a
        // failure path being wired to a soft success.
        let n = 50usize;
        let mut xv = Vec::with_capacity(n * 2);
        let mut yv = Vec::with_capacity(n);
        for i in 0..n {
            let xi = (i as f64) / (n as f64);
            xv.push(1.0);
            xv.push(xi);
            yv.push(1.0 + 2.0 * xi + if i % 3 == 0 { 0.1 } else { -0.05 });
        }
        let x = Array2::from_shape_vec((n, 2), xv).expect("test setup should be valid");
        let y = Array1::from_vec(yv);
        let result = fit_glm_unified(
            &y,
            x.view(),
            &GaussianFamily,
            &IdentityLink,
            &FitConfig::default(),
            None,
            None,
            None,
        )
        .expect("well-conditioned fit should succeed");
        assert!(
            result
                .covariance_unscaled
                .diag()
                .iter()
                .all(|&v| v.is_finite() && v > 0.0),
            "computed covariance must have strictly positive finite diagonal"
        );
    }

    #[test]
    fn test_final_mu_clamped_for_separated_binomial() {
        // RS-ACT-007: perfect separation drives eta to ±inf, so the final fitted
        // mu must be clamped strictly inside (0, 1) rather than hitting 0 or 1.
        let n = 40usize;
        let mut xv = Vec::with_capacity(n * 2);
        let mut yv = Vec::with_capacity(n);
        for i in 0..n {
            let xi = (i as f64) - (n as f64) / 2.0;
            xv.push(1.0);
            xv.push(xi);
            yv.push(if xi > 0.0 { 1.0 } else { 0.0 }); // perfectly separated
        }
        let x = Array2::from_shape_vec((n, 2), xv).expect("test setup should be valid");
        let y = Array1::from_vec(yv);
        let config = FitConfig {
            max_iterations: 100,
            ..FitConfig::default()
        };
        let result = fit_glm_unified(
            &y,
            x.view(),
            &BinomialFamily,
            &LogitLink,
            &config,
            None,
            None,
            None,
        )
        .expect("fit should not error");
        assert!(
            result.fitted_values.iter().all(|&m| m > 0.0 && m < 1.0),
            "fitted mu must be strictly inside (0, 1), got {:?}",
            result.fitted_values
        );
    }

    #[test]
    fn test_solver_status_reports_max_iterations() {
        // RS-ACT-007: a budget-capped fit that has not converged reports the
        // honest terminal status rather than silently claiming success.
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0],
        )
        .expect("test setup should be valid");
        let y = array![2.0, 2.0, 3.0, 4.0, 5.0, 7.0];
        let config = FitConfig {
            max_iterations: 1,
            ..FitConfig::default()
        };
        let result = fit_glm_unified(
            &y,
            x.view(),
            &PoissonFamily,
            &LogLink,
            &config,
            None,
            None,
            None,
        )
        .expect("fit should not error");
        assert!(!result.converged);
        assert_eq!(result.solver_status, "max_iterations");
    }

    #[test]
    fn test_first_iteration_does_not_accept_worse_full_step() {
        // RS-ACT-007: the first IRLS update used to bypass step acceptance and
        // take a catastrophic full Newton step. The starting μ is projected into
        // coefficient space, and even iteration one must retain or improve that
        // model-space state.
        let slope = vec![
            1.50216891456,
            -1.5886661306,
            -10.9303180964,
            4.63779755206,
            -16.5846282977,
            -9.37066765182,
            -8.09338144726,
            -4.12131687693,
            8.41092596272,
            16.5668043175,
            17.2229532456,
            8.06259378466,
            4.3969386775,
            -23.3654886682,
            11.30119646,
            26.2894656688,
            4.52876432659,
            2.34039314763,
            6.98695826853,
            7.28628585473,
            1.34916334847,
            -3.73815866047,
            3.61759209669,
            -13.7178920955,
            -18.824265433,
            13.6093864072,
            -4.58061330336,
            7.15955476441,
            15.5459398945,
            12.2028034961,
            -5.53392708658,
            3.24754029511,
            -11.6608697799,
            -1.04573101417,
            10.4601695101,
            0.320262403627,
            -28.5136929773,
            -22.5126340704,
            -8.0158557057,
            13.332416593,
            1.99085261163,
            1.41273124012,
            -13.1596718357,
            1.46431651468,
            -22.7517231178,
            3.41832183287,
            11.7924037162,
            -8.17587327,
            10.843960777,
            -7.19895381705,
        ];
        let mut xv = Vec::with_capacity(slope.len() * 2);
        for value in slope {
            xv.push(1.0);
            xv.push(value);
        }
        let x = Array2::from_shape_vec((50, 2), xv).expect("test setup should be valid");
        let y = array![
            0.0, 0.0, 14.0, 2.0, 6.0, 5.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 14.0, 0.0, 10.0, 1.0,
            28.0, 0.0, 0.0, 2.0, 0.0, 6.0, 18.0, 11.0, 4.0, 1.0, 15.0, 4.0, 2.0, 2.0, 0.0, 0.0,
            13.0, 0.0, 1.0, 0.0, 6.0, 39.0, 8.0, 1.0, 5.0, 0.0, 21.0, 0.0, 1.0, 0.0, 3.0, 0.0,
            78.0, 0.0
        ];

        let initial_mu = PoissonFamily.initialize_mu(&y);
        let initial_eta = LogLink.link(&initial_mu);
        let weights = Array1::ones(y.len());
        let (initial_coef, _, _) = solve_weighted_least_squares_penalized(
            x.view(),
            &initial_eta,
            &weights,
            0.0,
            None,
            true,
            false,
            None,
            None,
        )
        .expect("initial projection should be valid");
        let initial_fit_eta = x.dot(&initial_coef);
        let initial_fit_mu = PoissonFamily.clamp_mu(&LogLink.inverse(&initial_fit_eta));
        let initial_model_deviance = PoissonFamily.deviance(&y, &initial_fit_mu, None);

        let config = FitConfig {
            max_iterations: 1,
            ..FitConfig::default()
        };
        let result = fit_glm_unified(
            &y,
            x.view(),
            &PoissonFamily,
            &LogLink,
            &config,
            None,
            None,
            None,
        )
        .expect("fit should not error");

        assert!(result.step_halving_used);
        assert!(
            result.deviance <= initial_model_deviance * 1.0001,
            "first iteration accepted a worse state: initial={initial_model_deviance}, result={}",
            result.deviance
        );
        assert!(
            result.deviance < 1392.5,
            "regression guard for the historical full-step deviance"
        );
    }

    #[test]
    fn test_step_halving_no_improvement_retains_previous_iterate() {
        // RS-ACT-007 (007.2): when every full and halved trial step produces an
        // infinite/worse deviance, IRLS must retain the previous accepted iterate
        // (NOT take a bad step) and report `step_halving_no_improvement`. We
        // construct a Poisson/log fixture where the WLS solver returns a
        // catastrophic slope and the blended step blows past the float ceiling
        // even at the smallest budgeted half-step, so the halving budget exhausts.
        //
        // The fixture: y is mostly small with a single extreme value, paired with
        // x values that span ~30 orders of magnitude. The WLS step jumps to a
        // slope whose product with the extreme x overflows the log link, leaving
        // the family deviance non-finite at every halved blend.
        let n = 6;
        let xv = vec![
            1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0e6, 1.0, 1.0e6, 1.0, 1.0e6,
        ];
        let x = Array2::from_shape_vec((n, 2), xv).expect("test setup should be valid");
        // y has a huge spike at the extreme-x rows, which makes the WLS solve
        // return a slope close to log(1e300)/1e6 — products with x[3..6] then
        // blow up to infinity in mu = exp(eta).
        let y = array![1.0, 1.0, 1.0, 1.0e300, 1.0e300, 1.0e300];

        let config = FitConfig {
            max_iterations: 5,
            ..FitConfig::default()
        };
        let result = fit_glm_unified(
            &y,
            x.view(),
            &PoissonFamily,
            &LogLink,
            &config,
            None,
            None,
            None,
        )
        .expect("fit should not error");

        assert_eq!(
            result.solver_status, "step_halving_no_improvement",
            "expected halving exhaustion; got {} (deviance={}, iters={})",
            result.solver_status, result.deviance, result.iterations
        );
        assert!(!result.converged, "exhausted halving must not be converged");
        // A1: status and converged must never contradict.
        assert!(
            !(result.converged && result.solver_status == "step_halving_no_improvement"),
            "converged must not coexist with step_halving_no_improvement"
        );
        // A3: the halving-exhaustion break reports the RETAINED iterate index,
        // not the rejected iteration. With this fixture the first step is
        // accepted and the SECOND step is rejected, so the loop counter reaches 2
        // at the break but the retained iterate is index 1. Before the fix the
        // reported count was the rejected iteration (2); it must now be 1.
        assert_eq!(
            result.iterations, 1,
            "halving-exhaustion break must report the retained iterate index, not the rejected one"
        );
        // Retained coefficients are the previous iterate's, not the last failed
        // trial. The previous iterate corresponds to the projected initial fit
        // (iteration 1's "previous" state), so the returned coefficients must
        // produce a finite deviance — the failed trial's coefs would not.
        assert!(
            result.coefficients.iter().all(|c| c.is_finite()),
            "retained coefs must be finite (previous iterate), got {:?}",
            result.coefficients
        );
        assert!(
            result.deviance.is_finite(),
            "retained deviance must be finite (previous iterate's deviance)"
        );
    }

    #[test]
    fn test_solver_status_consistent_with_converged_invariant() {
        // A1: the reported `solver_status` must never contradict `converged`.
        // In particular the constrained best-iterate recovery block sets
        // converged = true; the previously-sticky `step_halving_failed` flag
        // must not then make the status read "step_halving_no_improvement".
        // We assert the invariant across a representative sweep that includes
        // sign-constrained fits (which exercise the best-iterate recovery path).
        let check = |result: &IRLSResult| {
            if result.converged {
                assert_ne!(
                    result.solver_status, "step_halving_no_improvement",
                    "converged result must not report step_halving_no_improvement"
                );
                assert_eq!(
                    result.solver_status, "converged",
                    "converged result must report status 'converged'"
                );
            }
            if result.solver_status == "step_halving_no_improvement" {
                assert!(
                    !result.converged,
                    "step_halving_no_improvement must not be marked converged"
                );
            }
        };

        // Unconstrained Poisson/log.
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0],
        )
        .expect("test setup should be valid");
        let y = array![2.0, 2.0, 3.0, 4.0, 5.0, 7.0];
        let config = FitConfig::default().with_max_iterations(100);
        let r = fit_glm_unified(
            &y,
            x.view(),
            &PoissonFamily,
            &LogLink,
            &config,
            None,
            None,
            None,
        )
        .expect("fit should not error");
        check(&r);

        // Sign-constrained Poisson/log (slope forced non-negative) — exercises
        // the best-iterate recovery branch where converged is set late.
        let config_nn = FitConfig::default()
            .with_max_iterations(100)
            .with_nonneg_indices(vec![1]);
        let r_nn = fit_glm_unified(
            &y,
            x.view(),
            &PoissonFamily,
            &LogLink,
            &config_nn,
            None,
            None,
            None,
        )
        .expect("constrained fit should not error");
        check(&r_nn);

        // Sign-constrained where the constraint fights the data (slope forced
        // non-positive on increasing data) — the projection oscillation path.
        let config_np = FitConfig::default()
            .with_max_iterations(100)
            .with_nonpos_indices(vec![1]);
        let r_np = fit_glm_unified(
            &y,
            x.view(),
            &PoissonFamily,
            &LogLink,
            &config_np,
            None,
            None,
            None,
        )
        .expect("constrained fit should not error");
        check(&r_np);
        assert!(
            r_np.iterations < config_np.max_iterations,
            "constrained best-iterate plateau should stop before exhausting the iteration budget"
        );
    }

    #[test]
    fn test_convergence_guard_is_relative_not_absolute() {
        // A2: the non-worsening convergence guard must be RELATIVE (mirroring the
        // accept loop's slack), not an absolute 1e-10 floor. We exercise the
        // extracted `irls_step_converged` decision directly at the regression
        // boundary.
        let tol = 1e-8;

        // Large-deviance fit: terminal step worsens deviance by a tiny RELATIVE
        // amount that is well above 1e-10 absolute (5e-6) but below tol*deviance
        // (1e-5), and rel_change (5e-9) is below tol. The OLD absolute guard
        // (signed_change >= -1e-10) would WRONGLY reject this; the relative guard
        // accepts it.
        let deviance_old = 1000.0_f64;
        let deviance = 1000.0_f64 + 5e-6; // worsened slightly
        let rel_change = (deviance_old - deviance).abs() / deviance_old.abs(); // 5e-9 < tol
        assert!(rel_change < tol, "fixture rel_change must be below tol");
        // Sanity: this is the case the old absolute guard rejected.
        assert!(
            deviance_old - deviance < -ZERO_TOL,
            "absolute worsening must exceed the old 1e-10 floor"
        );
        assert!(
            irls_step_converged(deviance_old, deviance, rel_change, tol),
            "relative guard must treat a tiny-relative-worsening large-deviance step as converged"
        );

        // A clearly-worse step (worsened by 10% relative) is NOT converged, even
        // though its rel_change could be small for a different reason — here we
        // make rel_change small artificially to confirm the non-worsening guard
        // is what rejects it.
        let deviance_bad = 1100.0_f64; // +10%
        assert!(
            !irls_step_converged(deviance_old, deviance_bad, 1e-12, tol),
            "a clearly-worse step must not be reported converged"
        );

        // A genuine improvement within tolerance is converged.
        let deviance_better = 1000.0_f64 - 1e-6;
        let rel_better = (deviance_old - deviance_better).abs() / deviance_old.abs();
        assert!(
            irls_step_converged(deviance_old, deviance_better, rel_better, tol),
            "a small improving step must be converged"
        );

        // Near-zero deviance retains an absolute floor (ZERO_TOL) so float noise
        // does not deny convergence there either.
        assert!(
            irls_step_converged(1e-12, 1e-12 + 1e-11, 1e-30, tol),
            "near-zero deviance must keep the ZERO_TOL absolute allowance"
        );
    }

    #[test]
    fn test_penalized_irls_objective_skips_intercept_when_configured() {
        let coefficients = array![10.0, 2.0, -3.0];
        let deviance = 5.0;
        let lambda = 0.5;

        assert_eq!(
            penalized_irls_objective(deviance, &coefficients, 0.0, None, false),
            deviance
        );
        assert_eq!(
            penalized_irls_objective(deviance, &coefficients, lambda, None, false),
            deviance + lambda * (2.0_f64.powi(2) + (-3.0_f64).powi(2))
        );
        assert_eq!(
            penalized_irls_objective(deviance, &coefficients, lambda, None, true),
            deviance + lambda * (10.0_f64.powi(2) + 2.0_f64.powi(2) + (-3.0_f64).powi(2))
        );
        let factors = vec![100.0, 4.0, 9.0];
        assert_eq!(
            penalized_irls_objective(deviance, &coefficients, lambda, Some(&factors), false),
            deviance + lambda * (4.0 * 2.0_f64.powi(2) + 9.0 * (-3.0_f64).powi(2))
        );
    }

    #[test]
    fn test_plateau_full_step_inside_tol_is_converged() {
        // RS-ACT-007 (plateau guard): a fit that reaches its converged solution
        // via a clean full Newton step (no halving) whose relative deviance
        // change drops below `tol` is reported as converged, not failed. This
        // guards against a regression where the signed-change guard would
        // wrongly reject benign plateau steps.
        let x = Array2::from_shape_vec(
            (5, 2),
            vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0],
        )
        .expect("test setup should be valid");
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0]; // perfect linear, OLS hits a plateau

        let config = FitConfig::default().with_max_iterations(50);
        let result = fit_glm_unified(
            &y,
            x.view(),
            &GaussianFamily,
            &IdentityLink,
            &config,
            None,
            None,
            None,
        )
        .expect("fit should not error");

        assert!(result.converged, "plateau full-step fit must converge");
        assert_eq!(result.solver_status, "converged");
        assert!(
            !result.step_halving_used,
            "no halving expected on a clean Gaussian/identity fit; flag stayed: {}",
            result.step_halving_used
        );
    }

    #[test]
    fn test_gaussian_identity_is_ols() {
        let x = Array2::from_shape_vec(
            (5, 2),
            vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0],
        )
        .expect("test setup should be valid");
        let y = array![5.1, 7.9, 11.2, 13.8, 17.1];

        let config = FitConfig::default();
        let result = fit_glm_unified(
            &y,
            x.view(),
            &GaussianFamily,
            &IdentityLink,
            &config,
            None,
            None,
            None,
        )
        .expect("test setup should be valid");

        assert!(result.converged);
        assert!((result.coefficients[0] - 2.0).abs() < 0.5);
        assert!((result.coefficients[1] - 3.0).abs() < 0.2);
    }

    #[test]
    fn test_poisson_log_link() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0],
        )
        .expect("test setup should be valid");
        let y = array![2.0, 2.0, 3.0, 4.0, 5.0, 7.0];

        let config = FitConfig::default();
        let result = fit_glm_unified(
            &y,
            x.view(),
            &PoissonFamily,
            &LogLink,
            &config,
            None,
            None,
            None,
        )
        .expect("test setup should be valid");

        assert!(result.converged);
        assert!(result.fitted_values.iter().all(|&x| x > 0.0));
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0])
            .expect("test setup should be valid");
        let y = array![1.0, 2.0]; // Wrong length!

        let config = FitConfig::default();
        let result = fit_glm_unified(
            &y,
            x.view(),
            &GaussianFamily,
            &IdentityLink,
            &config,
            None,
            None,
            None,
        );

        assert!(result.is_err());
        assert!(matches!(
            result.expect_err("test should exercise the error path"),
            RustyStatsError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn test_convergence_with_max_iter() {
        let x = Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0])
            .expect("test setup should be valid");
        let y = array![2.0, 4.0, 6.0, 8.0];

        let config = FitConfig::default().with_max_iterations(50);
        let result = fit_glm_unified(
            &y,
            x.view(),
            &GaussianFamily,
            &IdentityLink,
            &config,
            None,
            None,
            None,
        )
        .expect("test setup should be valid");

        assert!(result.converged);
        assert!(result.iterations < 10);
    }

    // =========================================================================
    // Ridge (L2) Regularization Tests
    // =========================================================================

    fn make_5x2_data() -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_vec(
            (5, 2),
            vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0],
        )
        .expect("test setup should be valid");
        let y = array![5.0, 8.0, 11.0, 14.0, 17.0];
        (x, y)
    }

    #[test]
    fn test_ridge_shrinks_coefficients() {
        let (x, y) = make_5x2_data();

        let unreg = fit_glm_unified(
            &y,
            x.view(),
            &GaussianFamily,
            &IdentityLink,
            &FitConfig::default(),
            None,
            None,
            None,
        )
        .expect("test setup should be valid");

        let ridge = fit_glm_unified(
            &y,
            x.view(),
            &GaussianFamily,
            &IdentityLink,
            &FitConfig::default().with_regularization(RegularizationConfig::ridge(10.0)),
            None,
            None,
            None,
        )
        .expect("test setup should be valid");

        assert!(
            ridge.coefficients[1].abs() < unreg.coefficients[1].abs(),
            "Ridge should shrink slope: ridge={:.4}, unreg={:.4}",
            ridge.coefficients[1],
            unreg.coefficients[1]
        );
        assert!(unreg.converged);
        assert!(ridge.converged);
        assert!(!ridge.penalty.is_none());
        assert_eq!(ridge.penalty.l2_penalty(), 10.0);
    }

    #[test]
    fn test_ridge_no_penalty_equals_ols() {
        let (x, y) = make_5x2_data();

        let unreg = fit_glm_unified(
            &y,
            x.view(),
            &GaussianFamily,
            &IdentityLink,
            &FitConfig::default(),
            None,
            None,
            None,
        )
        .expect("test setup should be valid");

        let ridge_zero = fit_glm_unified(
            &y,
            x.view(),
            &GaussianFamily,
            &IdentityLink,
            &FitConfig::default().with_regularization(RegularizationConfig::ridge(0.0)),
            None,
            None,
            None,
        )
        .expect("test setup should be valid");

        for i in 0..2 {
            assert!(
                (unreg.coefficients[i] - ridge_zero.coefficients[i]).abs() < 1e-6,
                "Coefficient {} differs: unreg={:.6}, ridge={:.6}",
                i,
                unreg.coefficients[i],
                ridge_zero.coefficients[i]
            );
        }
    }

    #[test]
    fn test_ridge_intercept_not_penalized() {
        let (x, y) = make_5x2_data();

        let unreg = fit_glm_unified(
            &y,
            x.view(),
            &GaussianFamily,
            &IdentityLink,
            &FitConfig::default(),
            None,
            None,
            None,
        )
        .expect("test setup should be valid");

        let ridge = fit_glm_unified(
            &y,
            x.view(),
            &GaussianFamily,
            &IdentityLink,
            &FitConfig::default().with_regularization(RegularizationConfig::ridge(100.0)),
            None,
            None,
            None,
        )
        .expect("test setup should be valid");

        assert!(
            ridge.coefficients[1].abs() < unreg.coefficients[1].abs() * 0.5,
            "Slope should be heavily shrunk"
        );
        assert!(
            ridge.coefficients[0].abs() > 1.0,
            "Intercept should not be heavily shrunk: {:.4}",
            ridge.coefficients[0]
        );
    }

    #[test]
    fn test_ridge_poisson() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0],
        )
        .expect("test setup should be valid");
        let y = array![2.0, 3.0, 4.0, 6.0, 8.0, 12.0];

        let config = FitConfig::default().with_regularization(RegularizationConfig::ridge(1.0));
        let result = fit_glm_unified(
            &y,
            x.view(),
            &PoissonFamily,
            &LogLink,
            &config,
            None,
            None,
            None,
        )
        .expect("test setup should be valid");

        assert!(result.converged);
        assert!(result.fitted_values.iter().all(|&x| x > 0.0));
    }
}
