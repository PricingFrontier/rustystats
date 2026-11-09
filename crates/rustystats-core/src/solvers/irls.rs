use super::validate_glm_inputs;

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
const SPARSE_XTWX_DENSITY_THRESHOLD: f64 = 0.35;
const SPARSE_ROW_CACHE_DENSITY_THRESHOLD: f64 = 0.35;

use nalgebra::{DMatrix, DVector};
use ndarray::{Array1, Array2, ArrayView2};
use rayon::prelude::*;
use std::time::Instant;

use super::passes::{
    record_step, rows_stay_sparse, Eval, EvalSum, InMemoryExecutor, PassExecutor, PassRequest,
    Predictor, ResponseSpec, RowContext, RowKernel, SolverStep, WeightsSpec,
};
use crate::constants::{
    CONVERGENCE_TOL, DEFAULT_MAX_ITER, IRLS_ACCEPT_REL_SLACK, IRLS_MAX_HALF_STEPS, MIN_IRLS_WEIGHT,
    ZERO_TOL,
};
use crate::error::{Result, RustyStatsError};
use crate::families::Family;
use crate::inference::HCType;
use crate::links::Link;
use crate::regularization::{Penalty, RegularizationConfig, Standardization};
use crate::rows::{reduce_rows, GramPartial, GRAM_BLOCK_ROWS};

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

    /// The row kernel (§5.2): chosen from the design's structural bound on nonzeros
    /// per row (the frozen design's; without one, X's rows are scanned), or given.
    /// Kernel choice never changes results.
    /// Default: `RowKernel::Structural(None)`
    pub row_kernel: RowKernel,
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
            row_kernel: RowKernel::default(),
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

    /// Robust (HC) covariance types to compute in the final pass
    /// (`FinalStatistics::robust`). Needs covariance (`skip_covariance` false).
    /// Default: none
    pub robust: Vec<HCType>,

    /// The row kernel (§5.2): chosen from the design's structural bound on nonzeros
    /// per row (the frozen design's; without one, X's rows are scanned), or given.
    /// Kernel choice never changes results.
    /// Default: `RowKernel::Structural(None)`
    pub row_kernel: RowKernel,
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
            robust: Vec::new(),
            row_kernel: RowKernel::default(),
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
            row_kernel: self.row_kernel,
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
            robust: Vec::new(),
            row_kernel: irls.row_kernel,
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

    /// Statistics of the fit computed from the rows at fit time
    /// (`specs/out-of-core-fit.md` §8). Results read them instead of recomputing
    /// from per-row arrays; None for routes not yet on a pass driver.
    pub statistics: Option<FinalStatistics>,

    /// The drivers' semantic steps (`specs/out-of-core-fit.md` §4.5).
    pub solver_trace: Vec<SolverStep>,
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
struct GramBuildProfile {
    materialize_seconds: f64,
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
    fit_glm_routed(
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

/// The routing of [`fit_glm_unified`]; `row_standardization` standardizes every
/// row before the kernels (the centred-standardisation route, §6.7).
#[allow(clippy::too_many_arguments)]
fn fit_glm_routed(
    y: &Array1<f64>,
    x: ArrayView2<'_, f64>,
    family: &dyn Family,
    link: &dyn Link,
    config: &FitConfig,
    offset: Option<&Array1<f64>>,
    weights: Option<&Array1<f64>>,
    init_coefficients: Option<&Array1<f64>>,
    row_standardization: Option<&Standardization>,
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
                    &config.robust,
                    None,
                );
            }
            // Every other standardised fit runs on (x − c)/s, formed per row with
            // `standardize_matrix`'s expression instead of an n×p copy.
            let init_work = match init_coefficients {
                Some(init) => Some(
                    standardization
                        .to_standardized_coefficients(init, config.regularization.fit_intercept)?,
                ),
                None => None,
            };
            let mut work_config = config.clone();
            work_config.standardization = None;
            // The standardized rows would give a robust covariance on the wrong
            // scale; the caller computes these from the original rows.
            work_config.robust = Vec::new();

            let mut result = fit_glm_routed(
                y,
                x,
                family,
                link,
                &work_config,
                offset,
                weights,
                init_work.as_ref(),
                Some(standardization),
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
            &config.robust,
            row_standardization,
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
            &config.robust,
            row_standardization,
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
    robust: &[HCType],
    row_standardization: Option<&Standardization>,
) -> Result<IRLSResult> {
    let profile_total_start = Instant::now();
    let profile_setup_start = Instant::now();
    let validated = validate_glm_inputs(y, x, offset, weights)?;
    let offset_vec = validated.offset;
    let prior_weights_vec = validated.prior_weights;
    let sparse_rows =
        rows_stay_sparse(row_standardization) && config.row_kernel.sparse_in_memory(x);
    let y_slice = y.as_slice().ok_or_else(|| {
        RustyStatsError::LinearAlgebraError("Response y must be contiguous".to_string())
    })?;
    let ctx = RowContext {
        family,
        link,
        y: y_slice,
        offset: offset_vec
            .as_slice()
            .expect("validated offset is contiguous"),
        prior_weights: prior_weights_vec
            .as_slice()
            .expect("validated prior weights are contiguous"),
        y_mean: if family.initialize_needs_mean() || init_coefficients.is_none() {
            crate::families::response_mean(y)
        } else {
            f64::NAN
        },
        min_weight: config.min_weight,
        has_offset: offset.is_some(),
        standardization: None,
    };
    let mut executor = InMemoryExecutor::new(x, sparse_rows, ctx)?;
    executor.set_row_standardization(row_standardization.cloned());
    let setup_seconds = profile_setup_start.elapsed().as_secs_f64();

    let outcome = drive_irls(
        &mut executor,
        &IrlsDriveSettings {
            config,
            l2_penalty,
            l2_penalty_factors,
            penalize_intercept,
            init_coefficients,
            family_name: family.name(),
            family,
            null_offset: crate::diagnostics::NullOffset::for_link(offset.is_some(), link),
            robust,
            statistics_without_covariance: false,
        },
    )?;

    // In-memory results keep the per-row state of the returned fit.
    let (eta, mu) = executor.materialize(&outcome.fitted);
    let irls_weights = outcome.weights.materialize(&executor, x.nrows());
    let solver_trace = executor.take_trace();

    let mut profile = outcome.profile;
    profile.setup_seconds = setup_seconds;
    profile.total_seconds = profile_total_start.elapsed().as_secs_f64();

    Ok(IRLSResult {
        coefficients: outcome.coefficients,
        fitted_values: Array1::from_vec(mu),
        linear_predictor: Array1::from_vec(eta),
        deviance: outcome.deviance,
        iterations: outcome.iterations,
        converged: outcome.converged,
        covariance_unscaled: outcome.covariance_unscaled,
        irls_weights: Array1::from_vec(irls_weights),
        prior_weights: prior_weights_vec,
        offset: offset_vec,
        y: y.to_owned(),
        family_name: family.name().to_string(),
        penalty,
        design_matrix: None, // Computed lazily in Python layer to avoid expensive copy
        warnings: outcome.warnings,
        step_halving_used: outcome.step_halving_used,
        solver_status: outcome.solver_status.to_string(),
        profile: Some(profile),
        statistics: outcome.statistics,
        solver_trace,
    })
}

/// A fit served by any [`PassExecutor`]: everything but per-row arrays.
#[derive(Debug, Clone)]
pub struct ExecutorFit {
    pub coefficients: Array1<f64>,
    pub covariance_unscaled: Array2<f64>,
    pub deviance: f64,
    pub iterations: usize,
    pub converged: bool,
    pub step_halving_used: bool,
    pub solver_status: String,
    pub warnings: Vec<String>,
    pub statistics: Option<FinalStatistics>,
    pub penalty: Penalty,
}

/// Fit a GLM whose rows are served by `executor` (an out-of-core source, or any
/// executor that is not an in-memory design), with the routing of
/// [`fit_glm_unified`] (`specs/out-of-core-fit.md` §6.7, §7.4, §7.5).
///
/// - No penalty, ridge, and the scale-only standardised ridge run the IRLS driver
///   on the rows as served.
/// - Lasso and elastic net run the coordinate-descent driver.
/// - Every other standardised fit has the executor standardize each row, and its
///   coefficients and covariance go through the back-transforms. Its robust
///   covariances are then formed from the unstandardized rows in one more pass,
///   as a stored design would give them.
///
/// `link` is the executor's link and `y_mean` the block-ordered mean response
/// (`families::response_mean`) its rows were served with.
#[allow(clippy::too_many_arguments)]
pub fn fit_glm_with_executor(
    executor: &mut dyn PassExecutor,
    family: &dyn Family,
    link: &dyn Link,
    config: &FitConfig,
    init_coefficients: Option<&Array1<f64>>,
    has_offset: bool,
    y_mean: f64,
) -> Result<ExecutorFit> {
    let fitted = fit_on_rows(
        executor,
        family,
        link,
        config,
        init_coefficients,
        has_offset,
        y_mean,
    );
    executor.set_row_standardization(None);
    fitted.map(|driven| driven.fit)
}

/// [`fit_glm_with_executor`]'s fit, returning the retained provenances and leaving
/// the executor's row standardization in place (the fitted predictor is in its
/// coordinates); the caller resets it.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fit_on_rows(
    executor: &mut dyn PassExecutor,
    family: &dyn Family,
    link: &dyn Link,
    config: &FitConfig,
    init_coefficients: Option<&Array1<f64>>,
    has_offset: bool,
    y_mean: f64,
) -> Result<GlmDriven> {
    let p = executor.n_params();
    let row_standardization = row_standardization(config, p)?;
    let fit_intercept = config.regularization.fit_intercept;
    let init_work = match (row_standardization, init_coefficients) {
        (Some(standardization), Some(init)) => {
            Some(standardization.to_standardized_coefficients(init, fit_intercept)?)
        }
        _ => None,
    };
    let init = init_work.as_ref().or(init_coefficients);
    // The driver runs on the standardized rows with no standardization of its own.
    let work_config;
    let driver_config = if row_standardization.is_some() {
        let mut config = config.clone();
        config.standardization = None;
        // Standardized rows would give robust covariances on the wrong scale;
        // they are formed from the unstandardized rows below.
        config.robust = Vec::new();
        work_config = config;
        &work_config
    } else {
        config
    };
    executor.set_row_standardization(row_standardization.cloned());
    let GlmDriven {
        mut fit,
        fitted,
        weights,
    } = drive_glm(
        executor,
        family,
        link,
        driver_config,
        init,
        has_offset,
        y_mean,
        true,
    )?;

    if let Some(standardization) = row_standardization {
        fit.coefficients =
            standardization.to_original_coefficients(&fit.coefficients, fit_intercept)?;
        if !config.skip_covariance {
            fit.covariance_unscaled =
                standardization.to_original_covariance(&fit.covariance_unscaled, fit_intercept)?;
        }
        if !config.robust.is_empty() && !config.skip_covariance {
            // μ and the weights come from the standardized rows the fit ran on; the
            // meats and leverages from the rows as served, with the original-scale
            // bread (`inference::robust_covariance` on the unstandardized design).
            let mut request = PassRequest::default();
            let predictor = request.predictor(fitted.clone());
            let weights = weights.robust_spec(&mut request);
            let eval = request.eval(Eval::RobustMeats {
                predictor,
                weights,
                types: config.robust.clone(),
                bread: fit.covariance_unscaled.iter().copied().collect(),
                hc1: crate::inference::hc1_scale(executor.n_rows(), p),
                raw_rows: true,
            });
            let sums = executor.run(&request)?;
            let robust = config
                .robust
                .iter()
                .zip(sums[eval].meats())
                .map(|(&hc_type, meat)| {
                    (
                        hc_type,
                        crate::inference::sandwich_from_packed_meat(
                            &fit.covariance_unscaled,
                            &meat.xtx_packed,
                        ),
                    )
                })
                .collect();
            if let Some(statistics) = fit.statistics.as_mut() {
                statistics.robust = robust;
            }
        }
    }
    Ok(GlmDriven {
        fit,
        fitted,
        weights,
    })
}

/// The standardization a fit applies to its rows: `config.standardization` when
/// the fit standardizes and is not the scale-only ridge (which penalises with
/// s² on the raw rows instead), checked against p.
pub(crate) fn row_standardization(
    config: &FitConfig,
    p: usize,
) -> Result<Option<&Standardization>> {
    let Some(standardization) = &config.standardization else {
        return Ok(None);
    };
    if !should_standardize_regularized_design(config) {
        return Ok(None);
    }
    standardization.validate(p)?;
    let scale_only = should_use_scale_only_standardized_ridge_path(
        config.regularization.penalty.l2_penalty(),
        config.regularization.penalty.requires_coordinate_descent(),
        &standardization.center,
    );
    Ok((!scale_only).then_some(standardization))
}

/// A fit driven on the rows as served, with the retained provenances.
pub(crate) struct GlmDriven {
    pub fit: ExecutorFit,
    pub fitted: Predictor,
    pub weights: RetainedWeights,
}

/// The IRLS or coordinate-descent driver on the rows as the executor serves them:
/// no row standardization of its own (`config.standardization` may only be the
/// scale-only ridge, which penalises with s²).
#[allow(clippy::too_many_arguments)]
pub(crate) fn drive_glm(
    executor: &mut dyn PassExecutor,
    family: &dyn Family,
    link: &dyn Link,
    config: &FitConfig,
    init: Option<&Array1<f64>>,
    has_offset: bool,
    y_mean: f64,
    statistics_without_covariance: bool,
) -> Result<GlmDriven> {
    let p = executor.n_params();
    let l2_penalty = config.regularization.penalty.l2_penalty();
    if row_standardization(config, p)?.is_some() {
        return Err(RustyStatsError::InvalidValue(
            "A centred standardization must be applied to the rows before driving the fit."
                .to_string(),
        ));
    }
    let penalty_factors: Option<Vec<f64>> = config
        .standardization
        .as_ref()
        .filter(|_| should_standardize_regularized_design(config))
        .map(|standardization| standardization.scale.iter().map(|s| s * s).collect());
    let fit_intercept = config.regularization.fit_intercept;
    let irls_config = config.to_irls_config();
    let null_offset = crate::diagnostics::NullOffset::for_link(has_offset, link);
    if config.regularization.penalty.requires_coordinate_descent() {
        let outcome = super::coordinate_descent::drive_cd(
            executor,
            &super::coordinate_descent::CdDriveSettings {
                config: &irls_config,
                regularization: &config.regularization,
                init_coefficients: init,
                skip_covariance: config.skip_covariance,
                y_mean,
                family,
                link,
                null_offset,
                robust: &config.robust,
                statistics_without_covariance,
            },
        )?;
        Ok(GlmDriven {
            fit: ExecutorFit {
                coefficients: outcome.coefficients,
                covariance_unscaled: outcome.covariance_unscaled,
                deviance: outcome.deviance,
                iterations: outcome.iterations,
                converged: outcome.converged,
                step_halving_used: false,
                solver_status: outcome.solver_status.to_string(),
                warnings: outcome.warnings,
                statistics: outcome.statistics,
                penalty: config.regularization.penalty.clone(),
            },
            fitted: outcome.fitted,
            weights: outcome.weights,
        })
    } else {
        let outcome = drive_irls(
            executor,
            &IrlsDriveSettings {
                config: &irls_config,
                l2_penalty,
                l2_penalty_factors: penalty_factors.as_deref(),
                penalize_intercept: !fit_intercept,
                init_coefficients: init,
                family_name: family.name(),
                family,
                null_offset,
                robust: &config.robust,
                statistics_without_covariance,
            },
        )?;
        Ok(GlmDriven {
            fit: ExecutorFit {
                coefficients: outcome.coefficients,
                covariance_unscaled: outcome.covariance_unscaled,
                deviance: outcome.deviance,
                iterations: outcome.iterations,
                converged: outcome.converged,
                step_halving_used: outcome.step_halving_used,
                solver_status: outcome.solver_status.to_string(),
                warnings: outcome.warnings,
                statistics: outcome.statistics,
                penalty: config.regularization.penalty.clone(),
            },
            fitted: outcome.fitted,
            weights: outcome.weights,
        })
    }
}

/// Everything the IRLS driver needs besides the rows.
pub(crate) struct IrlsDriveSettings<'a> {
    pub config: &'a IRLSConfig,
    pub l2_penalty: f64,
    pub l2_penalty_factors: Option<&'a [f64]>,
    pub penalize_intercept: bool,
    pub init_coefficients: Option<&'a Array1<f64>>,
    /// For messages only.
    pub family_name: &'a str,
    /// The family, for the final statistics' combination rules.
    pub family: &'a dyn Family,
    /// How the offset enters the null model (`RowContext::null_offset`).
    pub null_offset: crate::diagnostics::NullOffset,
    /// Robust covariance types for the final pass (`FitConfig::robust`).
    pub robust: &'a [HCType],
    /// Compute the final statistics even when covariance is skipped (pass A at
    /// the retained predictor, §8). An out-of-core result keeps no rows to compute
    /// them from later; cross-validation folds need neither.
    pub statistics_without_covariance: bool,
}

/// The weights the covariance was built from (`specs/out-of-core-fit.md` §7.2),
/// which a result stores as its `irls_weights`.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum RetainedWeights {
    /// The ordinary raw IRLS weights at a predictor (stored without prior weights).
    Irls(Predictor),
    /// No iteration ran (ordinary IRLS and coordinate descent).
    Zeros,
    /// The smooth route's solve weights at a predictor (stored with prior weights).
    SmoothSolve(Predictor),
    /// No iteration ran (smooth fits).
    Ones,
}

impl RetainedWeights {
    /// The weights of the covariance's Gram.
    pub(crate) fn spec(&self, request: &mut PassRequest) -> WeightsSpec {
        match self {
            RetainedWeights::Irls(predictor) => {
                WeightsSpec::Irls(request.predictor(predictor.clone()))
            }
            RetainedWeights::Zeros => WeightsSpec::Zeros,
            RetainedWeights::SmoothSolve(predictor) => {
                WeightsSpec::SmoothIrls(request.predictor(predictor.clone()))
            }
            RetainedWeights::Ones => WeightsSpec::Ones,
        }
    }

    /// The robust covariance's combined weights: the stored `irls_weights` times
    /// prior weights (`inference::robust_covariance`), so smooth fits apply prior
    /// weights twice (D9).
    pub(crate) fn robust_spec(&self, request: &mut PassRequest) -> WeightsSpec {
        match self {
            RetainedWeights::Irls(_) | RetainedWeights::Zeros => self.spec(request),
            RetainedWeights::SmoothSolve(predictor) => {
                WeightsSpec::PriorTimesSmoothIrls(request.predictor(predictor.clone()))
            }
            RetainedWeights::Ones => WeightsSpec::Prior,
        }
    }

    /// The stored `irls_weights` of an in-memory result.
    pub(crate) fn materialize(&self, executor: &InMemoryExecutor<'_>, n: usize) -> Vec<f64> {
        match self {
            RetainedWeights::Irls(predictor) => executor.raw_weights(predictor),
            RetainedWeights::Zeros => vec![0.0; n],
            RetainedWeights::SmoothSolve(predictor) => {
                executor.solve_weights(WeightsSpec::SmoothIrls(0), std::slice::from_ref(predictor))
            }
            RetainedWeights::Ones => vec![1.0; n],
        }
    }
}

/// The result of driving IRLS: everything but per-row arrays.
pub(crate) struct IrlsDriveOutcome {
    pub coefficients: Array1<f64>,
    pub covariance_unscaled: Array2<f64>,
    pub deviance: f64,
    pub iterations: usize,
    pub converged: bool,
    pub step_halving_used: bool,
    pub solver_status: &'static str,
    pub warnings: Vec<String>,
    /// The predictor the returned fitted values and linear predictor come from.
    pub fitted: Predictor,
    /// The weights the returned `irls_weights` hold.
    pub weights: RetainedWeights,
    /// Statistics of the returned fit, computed in its final passes; None when
    /// covariance is skipped.
    pub statistics: Option<FinalStatistics>,
    pub profile: IRLSProfile,
}

/// Statistics of a fitted GLM computed from the rows at fit time
/// (`specs/out-of-core-fit.md` §8), with the conventions of the result object.
#[derive(Clone, Debug, PartialEq)]
pub struct FinalStatistics {
    /// Every prior weight is 1 within `ZERO_TOL` (`maybe_weights` is None).
    pub all_unit_weights: bool,
    /// max |y| over finite y, when positive and finite.
    pub response_scale: Option<f64>,
    pub null_deviance: f64,
    pub pearson_chi2: f64,
    /// 1 for fixed-dispersion families; else χ²/(n − p), or 1 when n − p is 0.
    pub scale: f64,
    pub log_likelihood: f64,
    /// The robust covariances `FitConfig::robust` asked for, in its order
    /// (`inference::robust_covariance` with the fit's bread and retained weights).
    pub robust: Vec<(HCType, Array2<f64>)>,
}

/// Facts from the first pass that the final statistics need.
#[derive(Clone, Debug)]
pub(crate) struct FirstPassFacts {
    all_unit_weights: bool,
    response_scale: Option<f64>,
    null_model: crate::diagnostics::NullModel,
}

impl FirstPassFacts {
    pub(crate) fn from_sums(
        facts: &super::passes::Facts,
        n: usize,
        null_offset: crate::diagnostics::NullOffset,
    ) -> Self {
        let all_unit_weights = facts.non_unit_weights == 0.0;
        let sums = if all_unit_weights {
            &facts.null_sums_unweighted
        } else {
            &facts.null_sums_weighted
        };
        let null_model =
            crate::diagnostics::null_model_from_sums(sums, n, !all_unit_weights, null_offset);
        let response_scale =
            (facts.max_abs_y.is_finite() && facts.max_abs_y > 0.0).then_some(facts.max_abs_y);
        Self {
            all_unit_weights,
            response_scale,
            null_model,
        }
    }
}

/// The robust covariances a final pass computes: the types, the retained weights
/// they combine with prior weights, and the bread (the unscaled covariance).
pub(crate) struct RobustRequest<'a> {
    pub types: &'a [HCType],
    pub weights: &'a RetainedWeights,
    pub bread: &'a Array2<f64>,
}

/// Where a final pass's statistics land in its sums.
pub(crate) struct StatisticsEvals {
    pearson: usize,
    log_likelihood: usize,
    null: usize,
    /// The meats' evaluation, the types and the bread.
    robust: Option<(usize, Vec<HCType>, Array2<f64>)>,
}

/// Add the final statistics' evaluations at `predictor` to `request`: Pearson,
/// the φ-free log-likelihood sums, the null deviance and any robust meats.
pub(crate) fn request_statistics(
    request: &mut PassRequest,
    predictor: usize,
    facts: &FirstPassFacts,
    robust: Option<RobustRequest<'_>>,
    (n, p): (usize, usize),
) -> StatisticsEvals {
    let weighted = !facts.all_unit_weights;
    let pearson = request.eval(Eval::Pearson {
        predictor,
        weighted,
    });
    let log_likelihood = request.eval(Eval::LogLikelihood {
        predictor,
        weighted,
        scale: f64::NAN,
    });
    let null = request.eval(Eval::NullDeviance {
        model: facts.null_model,
        weighted,
    });
    let robust = robust.filter(|r| !r.types.is_empty()).map(|r| {
        let weights = r.weights.robust_spec(request);
        let eval = request.eval(Eval::RobustMeats {
            predictor,
            weights,
            types: r.types.to_vec(),
            bread: r.bread.iter().copied().collect(),
            hc1: crate::inference::hc1_scale(n, p),
            raw_rows: false,
        });
        (eval, r.types.to_vec(), r.bread.clone())
    });
    StatisticsEvals {
        pearson,
        log_likelihood,
        null,
        robust,
    }
}

/// Turn the final pass's sums into statistics; Gamma-style families whose
/// per-row term reads φ take one more pass at `predictor`.
pub(crate) fn finish_statistics(
    executor: &mut dyn PassExecutor,
    family: &dyn Family,
    predictor: &Predictor,
    facts: &FirstPassFacts,
    sums: &[EvalSum],
    evals: StatisticsEvals,
) -> Result<FinalStatistics> {
    let StatisticsEvals {
        pearson,
        log_likelihood,
        null,
        robust,
    } = evals;
    use crate::families::LogLikelihoodForm;
    let n = executor.n_rows();
    let df_resid = n.saturating_sub(executor.n_params());
    let pearson_chi2 = sums[pearson].sum();
    let scale = if family.fixed_dispersion() || df_resid == 0 {
        1.0
    } else {
        pearson_chi2 / (df_resid as f64)
    };
    let weighted = !facts.all_unit_weights;
    let log_likelihood = match family.log_likelihood_form() {
        LogLikelihoodForm::Gaussian => {
            let [squares, total_weight] = sums[log_likelihood].sums2();
            let total_weight = if weighted { total_weight } else { n as f64 };
            crate::diagnostics::combine_gaussian_log_likelihood(squares, total_weight, scale)
        }
        LogLikelihoodForm::RowSum { needs_scale: false } => sums[log_likelihood].sums2()[0],
        LogLikelihoodForm::RowSum { needs_scale: true } => {
            let mut request = PassRequest::default();
            let index = request.predictor(predictor.clone());
            let eval = request.eval(Eval::LogLikelihood {
                predictor: index,
                weighted,
                scale,
            });
            executor.run(&request)?[eval].sums2()[0]
        }
        LogLikelihoodForm::DevianceFallback => -sums[log_likelihood].sums2()[0] / 2.0,
    };
    Ok(FinalStatistics {
        all_unit_weights: facts.all_unit_weights,
        response_scale: facts.response_scale,
        null_deviance: sums[null].sum(),
        pearson_chi2,
        scale,
        log_likelihood,
        robust: robust
            .map(|(eval, types, bread)| {
                types
                    .into_iter()
                    .zip(sums[eval].meats())
                    .map(|(hc_type, meat)| {
                        (
                            hc_type,
                            crate::inference::sandwich_from_packed_meat(&bread, &meat.xtx_packed),
                        )
                    })
                    .collect()
            })
            .unwrap_or_default(),
    })
}

/// A Gram accumulated in a pass, with the count of rows whose solve weight is
/// positive (the ridge fast path needs to know whether any is).
struct SolveSystem {
    gram: GramPartial,
    positive_weight_rows: f64,
}

/// Add a Gram (and its positive-weight count) under `weights`/`response` to `request`;
/// returns the evaluation indices.
fn request_gram(
    request: &mut PassRequest,
    weights: WeightsSpec,
    response: ResponseSpec,
) -> (usize, usize) {
    let gram = request.eval(Eval::Gram { weights, response });
    let positive = request.eval(Eval::PositiveWeights(weights));
    (gram, positive)
}

fn take_system(sums: &mut [EvalSum], (gram, positive): (usize, usize)) -> SolveSystem {
    let positive_weight_rows = sums[positive].count();
    let gram = std::mem::replace(&mut sums[gram], EvalSum::Count(0.0)).into_gram();
    SolveSystem {
        gram,
        positive_weight_rows,
    }
}

/// Solve the (ridge-penalised) normal equations of an accumulated Gram, exactly
/// as `solve_weighted_least_squares_penalized` did after forming X'WX.
fn solve_penalized_system(
    system: &SolveSystem,
    rhs: Option<&[f64]>,
    l2_penalty: f64,
    l2_penalty_factors: Option<&[f64]>,
    penalize_intercept: bool,
    skip_covariance: bool,
) -> Result<(Array1<f64>, Array2<f64>)> {
    let p = system.gram.p();
    if let Some(factors) = l2_penalty_factors {
        if factors.len() != p {
            return Err(RustyStatsError::dim_mismatch(
                p,
                factors.len(),
                "L2 penalty factors length vs X columns",
            ));
        }
    }
    let mut xtx = packed_to_dmatrix(&system.gram.xtx_packed, p);
    let xtz = DVector::from_vec(rhs.unwrap_or(&system.gram.xtz).to_vec());
    if should_apply_l2_penalty(l2_penalty) {
        let start_idx = if penalize_intercept { 0 } else { 1 };
        for j in start_idx..p {
            let factor = l2_penalty_factors.map_or(1.0, |factors| factors[j]);
            xtx[(j, j)] += l2_penalty * factor;
        }
    }
    if skip_covariance {
        let coefficients = if ridge_system_is_positive_definite_fast_path(
            system.positive_weight_rows > 0.0,
            l2_penalty,
            l2_penalty_factors,
            penalize_intercept,
            p,
        ) {
            cholesky_solve_spd_coefficients(xtx, &xtz)?
        } else {
            cholesky_solve_coefficients(xtx, &xtz)?
        };
        Ok((coefficients, Array2::zeros((0, 0))))
    } else {
        cholesky_solve(xtx, &xtz, false)
    }
}

/// Run one pass that evaluates the deviance at `predictor` and accumulates the
/// IRLS normal equations there (the next step's system if the state is kept).
fn evaluate_with_system(
    executor: &mut dyn PassExecutor,
    predictor: Predictor,
) -> Result<(f64, bool, SolveSystem)> {
    let mut request = PassRequest::default();
    let index = request.predictor(predictor);
    let deviance = request.eval(Eval::Deviance(index));
    let system = request_gram(
        &mut request,
        WeightsSpec::Irls(index),
        ResponseSpec::Working(index),
    );
    let mut sums = executor.run(&request)?;
    let (deviance, finite) = sums[deviance].deviance();
    Ok((deviance, finite, take_system(&mut sums, system)))
}

/// Deviance and finiteness at each predictor, in one pass.
fn evaluate_deviances(
    executor: &mut dyn PassExecutor,
    predictors: Vec<Predictor>,
) -> Result<Vec<(f64, bool)>> {
    let mut request = PassRequest::default();
    let evals: Vec<usize> = predictors
        .into_iter()
        .map(|predictor| {
            let index = request.predictor(predictor);
            request.eval(Eval::Deviance(index))
        })
        .collect();
    let sums = executor.run(&request)?;
    Ok(evals.into_iter().map(|e| sums[e].deviance()).collect())
}

pub(crate) fn coef_predictor(coefficients: &Array1<f64>) -> Predictor {
    Predictor::coef(
        coefficients
            .as_slice()
            .expect("coefficients are contiguous"),
    )
}

/// The IRLS step logic of `fit_glm_core`, over passes (`specs/out-of-core-fit.md`
/// §7.3–7.4). Every quantity it reads from the rows comes from `executor`, so an
/// in-memory and an out-of-core fit of the same data run exactly this code.
pub(crate) fn drive_irls(
    executor: &mut dyn PassExecutor,
    settings: &IrlsDriveSettings<'_>,
) -> Result<IrlsDriveOutcome> {
    let config = settings.config;
    let l2_penalty = settings.l2_penalty;
    let l2_penalty_factors = settings.l2_penalty_factors;
    let penalize_intercept = settings.penalize_intercept;
    let p = executor.n_params();
    let mut profile = IRLSProfile::default();
    let mut warnings: Vec<String> = Vec::new();

    // -------------------------------------------------------------------------
    // Start: a warm start keeps η = g(μ) at the given coefficients; a cold start
    // projects the family initializer into coefficient space (RS-ACT-007) so the
    // retained iterate, the step-halving baseline and the returned coefficients
    // all describe the same fitted state.
    // -------------------------------------------------------------------------
    let start_time = Instant::now();
    let mut first_facts: Option<super::passes::Facts> = None;
    let (mut iter_coefficients, mut state_predictor) = match settings.init_coefficients {
        Some(init) => {
            if init.len() != p {
                return Err(RustyStatsError::dim_mismatch(
                    p,
                    init.len(),
                    "init_coefficients length vs X columns",
                ));
            }
            let coefficients = init.clone();
            let predictor = Predictor::CoefRoundTrip(std::sync::Arc::from(
                coefficients
                    .as_slice()
                    .expect("coefficients are contiguous"),
            ));
            (coefficients, predictor)
        }
        None => {
            let mut request = PassRequest::default();
            let family_init = request.predictor(Predictor::Initializer { safe: false });
            let safe_init = request.predictor(Predictor::Initializer { safe: true });
            let invalid = request.eval(Eval::InitializerInvalid);
            let system = request_gram(
                &mut request,
                WeightsSpec::Prior,
                ResponseSpec::LinkMinusOffset(family_init),
            );
            let safe_rhs = request.eval(Eval::Rhs {
                weights: WeightsSpec::Prior,
                response: ResponseSpec::LinkMinusOffset(safe_init),
            });
            let facts_eval = request.eval(Eval::Facts);
            let mut sums = executor.run(&request)?;
            first_facts = Some(sums[facts_eval].facts().clone());
            let use_safe = sums[invalid].count() > 0.0;
            let safe_rhs = std::mem::replace(&mut sums[safe_rhs], EvalSum::Count(0.0)).into_rhs();
            let system = take_system(&mut sums, system);
            if use_safe {
                warnings.push(format!(
                    "Family '{}' initial μ values were invalid. Using safe fallback initialization. \
                    This may indicate unusual response values (e.g., all zeros, extreme values).",
                    settings.family_name
                ));
            }
            let projected = solve_penalized_system(
                &system,
                use_safe.then_some(safe_rhs.as_slice()),
                l2_penalty,
                l2_penalty_factors,
                penalize_intercept,
                true, // init projection covariance is discarded
            );
            let coefficients = match projected {
                Ok((coef, _)) => match finite_coefficients_or_none(coef) {
                    Some(mut coef) => {
                        project_coefficients_to_sign_constraints(
                            &mut coef,
                            &config.nonneg_indices,
                            &config.nonpos_indices,
                        );
                        coef
                    }
                    None => {
                        warnings.push(
                            "Initial coefficient projection failed. Starting IRLS from zero coefficients."
                                .to_string(),
                        );
                        Array1::zeros(p)
                    }
                },
                Err(_) => {
                    warnings.push(
                        "Initial coefficient projection failed. Starting IRLS from zero coefficients."
                            .to_string(),
                    );
                    Array1::zeros(p)
                }
            };
            let predictor = coef_predictor(&coefficients);
            (coefficients, predictor)
        }
    };
    profile.init_projection_seconds = start_time.elapsed().as_secs_f64();

    // The deviance at the start, and the first iteration's normal equations; a warm
    // start reads the y/weight/offset facts here (a cold start read them above).
    let initial_time = Instant::now();
    let (mut deviance, mut system) = {
        let mut request = PassRequest::default();
        let index = request.predictor(state_predictor.clone());
        let deviance = request.eval(Eval::Deviance(index));
        let system = request_gram(
            &mut request,
            WeightsSpec::Irls(index),
            ResponseSpec::Working(index),
        );
        let facts_eval = first_facts.is_none().then(|| request.eval(Eval::Facts));
        let mut sums = executor.run(&request)?;
        if let Some(eval) = facts_eval {
            first_facts = Some(sums[eval].facts().clone());
        }
        (sums[deviance].deviance().0, take_system(&mut sums, system))
    };
    let facts = FirstPassFacts::from_sums(
        first_facts
            .as_ref()
            .expect("the first pass reads the facts"),
        executor.n_rows(),
        settings.null_offset,
    );
    let mut objective = penalized_irls_objective(
        deviance,
        &iter_coefficients,
        l2_penalty,
        l2_penalty_factors,
        penalize_intercept,
    );
    profile.initial_deviance_seconds = initial_time.elapsed().as_secs_f64();

    let mut objective_old: f64;
    let mut converged = false;
    let mut iteration = 0;
    let mut step_halving_used = false;
    // Set when no full or halved step reduces the objective: the previous
    // iterate is retained rather than accepting a worse step (RS-ACT-007).
    let mut step_halving_failed = false;
    let mut final_weights = RetainedWeights::Zeros;

    // For constrained problems, track the best solution seen (projection can make
    // either deviance or the penalized objective increase).
    let has_constraints = has_sign_constraints(&config.nonneg_indices, &config.nonpos_indices);
    let mut best_objective = f64::INFINITY;
    let mut best_deviance = f64::INFINITY;
    let mut best_coefficients = iter_coefficients.clone();
    let mut best_predictor = state_predictor.clone();
    let mut best_weights = final_weights.clone();
    let mut constrained_best_stale_iterations = 0usize;
    let mut constrained_best_plateau = false;

    while iteration < config.max_iterations {
        iteration += 1;
        // The IRLS weights of this iteration are those at the current state.
        let weights_now = RetainedWeights::Irls(state_predictor.clone());

        let solve_time = Instant::now();
        let (mut new_coefficients, _) = solve_penalized_system(
            &system,
            None,
            l2_penalty,
            l2_penalty_factors,
            penalize_intercept,
            true, // per-iteration inverse is unused; compute final covariance once
        )?;
        add_profile_seconds(
            &mut profile.wls_solve_seconds,
            solve_time.elapsed().as_secs_f64(),
        );

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
        project_coefficients_to_sign_constraints(
            &mut new_coefficients,
            &config.nonneg_indices,
            &config.nonpos_indices,
        );

        objective_old = objective;
        // Acceptance threshold: a step may not worsen the objective by more than a
        // tiny relative tolerance (RS-ACT-007).
        let accept_threshold = objective_old * IRLS_ACCEPT_REL_SLACK;
        let acceptable = |deviance: f64, finite: bool, coefficients: &Array1<f64>| {
            let objective = penalized_irls_objective(
                deviance,
                coefficients,
                l2_penalty,
                l2_penalty_factors,
                penalize_intercept,
            );
            (
                finite && objective.is_finite() && objective <= accept_threshold,
                objective,
            )
        };

        // The full step: its deviance, and speculatively the next system.
        let update_time = Instant::now();
        let mut trial_coefficients = new_coefficients.clone();
        project_coefficients_to_sign_constraints(
            &mut trial_coefficients,
            &config.nonneg_indices,
            &config.nonpos_indices,
        );
        let (full_deviance, full_finite, full_system) =
            evaluate_with_system(executor, coef_predictor(&trial_coefficients))?;
        let (mut step_accepted, full_objective) =
            acceptable(full_deviance, full_finite, &trial_coefficients);
        let mut deviance_new = full_deviance;
        let mut objective_new = full_objective;
        let mut next_system = Some(full_system);
        // The trace counts the trials in the sequential rule's order: the full step,
        // then the halvings up to the first acceptable one (§4.5).
        let mut evaluations = 1usize;
        let mut step_status = "full_step";

        // Step halving: every halved trial is evaluated in one pass, then the
        // sequential rule picks the first acceptable one, as one trial at a time
        // would. Only the accepted trial's system is then accumulated.
        if !step_accepted {
            let mut trials = Vec::with_capacity(IRLS_MAX_HALF_STEPS);
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
                trials.push(blended);
                step_size *= 0.5;
            }
            let results =
                evaluate_deviances(executor, trials.iter().map(coef_predictor).collect())?;
            next_system = None;
            evaluations += IRLS_MAX_HALF_STEPS;
            step_status = "exhausted";
            for (trial, (blended, (d, finite))) in trials.into_iter().zip(results).enumerate() {
                let (ok, o) = acceptable(d, finite, &blended);
                if ok {
                    evaluations = trial + 2;
                    step_status = "halved";
                    let (_, _, accepted_system) =
                        evaluate_with_system(executor, coef_predictor(&blended))?;
                    trial_coefficients = blended;
                    deviance_new = d;
                    objective_new = o;
                    next_system = Some(accepted_system);
                    step_accepted = true;
                    step_halving_used = true;
                    break;
                }
            }
        }

        record_step(
            executor,
            SolverStep::new("irls", "iteration", evaluations, step_accepted, step_status),
        );
        if !step_accepted {
            // No full or halved step reduced the objective: retain the previous
            // iterate and stop (RS-ACT-007). Report the index of the retained
            // iterate, consistent with the converged / max_iterations count.
            step_halving_failed = true;
            iteration = iteration.saturating_sub(1);
            final_weights = weights_now;
            add_profile_seconds(
                &mut profile.update_seconds,
                update_time.elapsed().as_secs_f64(),
            );
            break;
        }

        new_coefficients = trial_coefficients;
        state_predictor = coef_predictor(&new_coefficients);
        system = next_system.expect("an accepted step carries its system");
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
        iter_coefficients = new_coefficients;

        // For constrained problems, track the best solution seen. Projection can
        // leave subsequent accepted IRLS steps worse than the best projected
        // iterate; after a short stale window, adopt the best now.
        if has_constraints {
            if best_objective_improved(objective, best_objective) {
                best_objective = objective;
                best_deviance = deviance;
                best_coefficients = iter_coefficients.clone();
                best_predictor = state_predictor.clone();
                best_weights = weights_now.clone();
                constrained_best_stale_iterations = 0;
            } else if best_objective.is_finite() {
                constrained_best_stale_iterations += 1;
                if should_stop_after_stale_best(constrained_best_stale_iterations) {
                    constrained_best_plateau = true;
                    final_weights = weights_now;
                    add_profile_seconds(
                        &mut profile.update_seconds,
                        update_time.elapsed().as_secs_f64(),
                    );
                    break;
                }
            }
        }

        // Convergence requires the accepted step to be non-worsening AND small.
        if irls_step_converged(objective_old, objective, rel_change, config.tolerance) {
            converged = true;
            final_weights = weights_now;
            add_profile_seconds(
                &mut profile.update_seconds,
                update_time.elapsed().as_secs_f64(),
            );
            break;
        }

        final_weights = weights_now;
        add_profile_seconds(
            &mut profile.update_seconds,
            update_time.elapsed().as_secs_f64(),
        );
    }

    // -------------------------------------------------------------------------
    // Final coefficients. For constrained problems the best solution found is
    // used when it beats the last iterate (sign clamping can make the objective
    // oscillate even when the deviance has stabilised).
    // -------------------------------------------------------------------------
    let (mut fitted, mut final_deviance, use_coefficients) = if should_use_constrained_best(
        has_constraints,
        best_objective,
        objective,
        constrained_best_plateau,
    ) {
        converged = true;
        step_halving_failed = false;
        final_weights = best_weights;
        record_step(
            executor,
            SolverStep::new("irls", "restore", 0, true, "constrained_best"),
        );
        (best_predictor, best_deviance, best_coefficients)
    } else {
        (state_predictor, deviance, iter_coefficients)
    };

    let extraction_time = Instant::now();
    let mut statistics: Option<FinalStatistics> = None;
    let (final_coefficients, cov_unscaled) = if config.skip_covariance {
        (use_coefficients, Array2::zeros((p, p)))
    } else {
        // Re-solve with the retained weights and the working response of the
        // retained state (one reconstruct pass, R), then check that the
        // extraction is no worse than the retained iterate (pass A).
        let mut request = PassRequest::reconstruct();
        let weights_spec = final_weights.spec(&mut request);
        let retained = request.predictor(fitted.clone());
        let system = request_gram(&mut request, weights_spec, ResponseSpec::Working(retained));
        let mut sums = executor.run(&request)?;
        let system = take_system(&mut sums, system);
        match solve_penalized_system(
            &system,
            None,
            l2_penalty,
            l2_penalty_factors,
            penalize_intercept,
            false,
        ) {
            Ok((coef, cov)) => {
                if !cov.iter().all(|v| v.is_finite()) {
                    return Err(RustyStatsError::NumericalError(
                        "Final covariance extraction produced NaN/Inf. \
                        This usually indicates numerical instability or a nearly singular design matrix."
                            .to_string(),
                    ));
                }
                if let Some(coef) = finite_coefficients_or_none(coef) {
                    let mut candidate = coef;
                    if has_constraints {
                        project_coefficients_to_sign_constraints(
                            &mut candidate,
                            &config.nonneg_indices,
                            &config.nonpos_indices,
                        );
                    }
                    let mut request = PassRequest::default();
                    let index = request.predictor(coef_predictor(&candidate));
                    let deviance_eval = request.eval(Eval::Deviance(index));
                    let stats = request_statistics(
                        &mut request,
                        index,
                        &facts,
                        Some(RobustRequest {
                            types: settings.robust,
                            weights: &final_weights,
                            bread: &cov,
                        }),
                        (executor.n_rows(), executor.n_params()),
                    );
                    let sums = executor.run(&request)?;
                    let check = sums[deviance_eval].deviance().0;
                    let extracted = final_extraction_deviance_acceptable(check, final_deviance);
                    record_step(
                        executor,
                        SolverStep::new(
                            "irls",
                            "extraction",
                            1,
                            extracted,
                            if extracted { "accepted" } else { "rejected" },
                        ),
                    );
                    if extracted {
                        // The recompute below would evaluate exactly this state.
                        fitted = coef_predictor(&candidate);
                        final_deviance = check;
                        statistics = Some(finish_statistics(
                            executor,
                            settings.family,
                            &fitted,
                            &facts,
                            &sums,
                            stats,
                        )?);
                        (candidate, cov)
                    } else {
                        (use_coefficients, cov)
                    }
                } else {
                    record_step(
                        executor,
                        SolverStep::new("irls", "extraction", 0, false, "nonfinite"),
                    );
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
                // Fail closed (pre-v0.8.14 contract): a failed final covariance
                // solve must not return a "successful" fit with an all-zeros
                // covariance. Callers that do not need inference set
                // skip_covariance and never reach this solve.
                return Err(RustyStatsError::LinearAlgebraError(
                    "Final coefficient/covariance extraction failed. \
                    This often indicates multicollinearity in predictors."
                        .to_string(),
                ));
            }
        }
    };
    profile.final_extraction_seconds = extraction_time.elapsed().as_secs_f64();

    // CV folds skip covariance and keep the accepted iterate; otherwise the final
    // state is recomputed at the final coefficients (unless pass A already did).
    if should_recompute_final_state(config.skip_covariance) {
        let final_predictor = coef_predictor(&final_coefficients);
        if final_predictor != fitted || statistics.is_none() {
            let recompute_time = Instant::now();
            let mut request = PassRequest::default();
            let index = request.predictor(final_predictor.clone());
            let deviance_eval = request.eval(Eval::Deviance(index));
            let stats = request_statistics(
                &mut request,
                index,
                &facts,
                Some(RobustRequest {
                    types: settings.robust,
                    weights: &final_weights,
                    bread: &cov_unscaled,
                }),
                (executor.n_rows(), executor.n_params()),
            );
            let sums = executor.run(&request)?;
            final_deviance = sums[deviance_eval].deviance().0;
            statistics = Some(finish_statistics(
                executor,
                settings.family,
                &final_predictor,
                &facts,
                &sums,
                stats,
            )?);
            profile.final_recompute_seconds = recompute_time.elapsed().as_secs_f64();
        }
        fitted = final_predictor;
    }
    if statistics.is_none() && settings.statistics_without_covariance {
        let mut request = PassRequest::default();
        let index = request.predictor(fitted.clone());
        let stats = request_statistics(
            &mut request,
            index,
            &facts,
            None,
            (executor.n_rows(), executor.n_params()),
        );
        let sums = executor.run(&request)?;
        statistics = Some(finish_statistics(
            executor,
            settings.family,
            &fitted,
            &facts,
            &sums,
            stats,
        )?);
    }

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
    record_step(
        executor,
        SolverStep::new("irls", "result", iteration, converged, solver_status),
    );

    Ok(IrlsDriveOutcome {
        coefficients: final_coefficients,
        covariance_unscaled: cov_unscaled,
        deviance: final_deviance,
        iterations: iteration,
        converged,
        step_halving_used,
        solver_status,
        warnings,
        fitted,
        weights: final_weights,
        statistics,
        profile,
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

    let x_slice = x.as_slice().ok_or_else(|| {
        RustyStatsError::LinearAlgebraError(
            "Design matrix X must be contiguous in memory (C-order)".to_string(),
        )
    })?;
    let w_slice = w.as_slice().ok_or_else(|| {
        RustyStatsError::LinearAlgebraError(
            "Weight vector W must be contiguous in memory".to_string(),
        )
    })?;
    let z_slice = z.as_slice().ok_or_else(|| {
        RustyStatsError::LinearAlgebraError(
            "Working response Z must be contiguous in memory".to_string(),
        )
    })?;
    assert_eq!(x_slice.len(), n * p, "x_slice length must be n*p");
    assert_eq!(w_slice.len(), n, "w_slice length must be n");
    assert_eq!(z_slice.len(), n, "z_slice length must be n");

    let collect_profile = profile_gram_subtimers_enabled();
    let use_sparse = should_use_sparse_xtwx_kernel(x_slice, n, p);
    let gram = reduce_rows(GramPartial::zeros(p), n, |partial, rows| {
        if use_sparse {
            accumulate_sparse_scan_rows(partial, x_slice, w_slice, Some(z_slice), p, rows);
        } else {
            accumulate_dense_rows(partial, x_slice, w_slice, Some(z_slice), p, rows);
        }
    });

    let mut gram_profile = GramBuildProfile::default();
    let materialize_start = collect_profile.then(Instant::now);
    let xtx = packed_to_dmatrix(&gram.xtx_packed, p);
    if let Some(start) = materialize_start {
        gram_profile.materialize_seconds = start.elapsed().as_secs_f64();
    }
    Ok((xtx, DVector::from_vec(gram.xtz), gram_profile))
}

// ---------------------------------------------------------------------------
// Block row kernels (`specs/out-of-core-fit.md` §5.1).
//
// Every weighted Gram in the crate is reduced with `rows::reduce_rows`, so the
// sums do not depend on thread count or on how rows arrive. Within a block a
// row adds `(x_ki * w_k) * x_kj` into packed entry (i, j) and `x_ki * (w_k *
// z_k)` into X'Wz, for i <= j in increasing order. The dense, sparse-scan and
// cached-sparse kernels evaluate that expression over the same nonzero
// products in the same order; a skipped product is an exact zero, so all three
// produce identical sums and kernel choice only affects speed.
// ---------------------------------------------------------------------------

/// Add the rows `rows` of a dense row-major X into `partial`.
#[inline]
fn accumulate_dense_rows(
    partial: &mut GramPartial,
    x_slice: &[f64],
    w_slice: &[f64],
    z_slice: Option<&[f64]>,
    p: usize,
    rows: std::ops::Range<usize>,
) {
    for k in rows {
        let row = &x_slice[k * p..(k + 1) * p];
        accumulate_dense_row(partial, row, w_slice[k], z_slice.map(|z| z[k]));
    }
}

/// Add one dense row into `partial`.
#[inline(always)]
pub(crate) fn accumulate_dense_row(
    partial: &mut GramPartial,
    row: &[f64],
    wk: f64,
    zk: Option<f64>,
) {
    let p = row.len();
    let mut packed_idx = 0usize;
    match zk {
        Some(zk) => {
            let wz = wk * zk;
            partial.ztwz += wz * zk;
            for i in 0..p {
                let xki = row[i];
                let xki_w = xki * wk;
                partial.xtz[i] += xki * wz;
                let width = p - i;
                let target = &mut partial.xtx_packed[packed_idx..packed_idx + width];
                for (entry, &xkj) in target.iter_mut().zip(&row[i..]) {
                    *entry += xki_w * xkj;
                }
                packed_idx += width;
            }
        }
        None => {
            for i in 0..p {
                let xki_w = row[i] * wk;
                let width = p - i;
                let target = &mut partial.xtx_packed[packed_idx..packed_idx + width];
                for (entry, &xkj) in target.iter_mut().zip(&row[i..]) {
                    *entry += xki_w * xkj;
                }
                packed_idx += width;
            }
        }
    }
}

/// Add the rows `rows` of a dense X, scanning each row for nonzeros first.
#[inline]
fn accumulate_sparse_scan_rows(
    partial: &mut GramPartial,
    x_slice: &[f64],
    w_slice: &[f64],
    z_slice: Option<&[f64]>,
    p: usize,
    rows: std::ops::Range<usize>,
) {
    let packed_offsets = packed_upper_offsets(p);
    let mut nz_idx: Vec<usize> = Vec::with_capacity(p);
    let mut nz_val: Vec<f64> = Vec::with_capacity(p);
    for k in rows {
        nz_idx.clear();
        nz_val.clear();
        let row = &x_slice[k * p..(k + 1) * p];
        for (j, &xkj) in row.iter().enumerate() {
            if xkj != 0.0 {
                nz_idx.push(j);
                nz_val.push(xkj);
            }
        }
        let wk = w_slice[k];
        let wz = match z_slice {
            Some(z) => {
                let zk = z[k];
                let wz = wk * zk;
                partial.ztwz += wz * zk;
                wz
            }
            None => 0.0,
        };
        if nz_idx.is_empty() {
            continue;
        }
        accumulate_sparse_row_pairwise(
            &mut partial.xtx_packed,
            &mut partial.xtz,
            &packed_offsets,
            &nz_idx,
            &nz_val,
            wk,
            wz,
        );
    }
}

/// Add the rows `rows` of a cached sparse X into `partial`.
#[inline]
fn accumulate_cached_rows(
    partial: &mut GramPartial,
    cache: &SparseRowCache,
    w_slice: &[f64],
    z_slice: Option<&[f64]>,
    rows: std::ops::Range<usize>,
) {
    for row in rows {
        let wk = w_slice[row];
        let wz = match z_slice {
            Some(z) => {
                let zk = z[row];
                let wz = wk * zk;
                partial.ztwz += wz * zk;
                wz
            }
            None => 0.0,
        };
        let start = cache.offsets[row];
        let end = cache.offsets[row + 1];
        if start == end {
            continue;
        }
        accumulate_cached_sparse_row_pairwise(
            cache,
            start,
            end,
            &mut partial.xtx_packed,
            &mut partial.xtz,
            wk,
            wz,
        );
    }
}

/// Add the rows `rows` of a non-contiguous X into `partial`, one row at a time.
#[inline]
fn accumulate_strided_rows(
    partial: &mut GramPartial,
    x: ArrayView2<'_, f64>,
    w: &Array1<f64>,
    z: Option<&Array1<f64>>,
    rows: std::ops::Range<usize>,
) {
    let mut row_buffer = vec![0.0; x.ncols()];
    for k in rows {
        for (slot, &value) in row_buffer.iter_mut().zip(x.row(k).iter()) {
            *slot = value;
        }
        accumulate_dense_row(partial, &row_buffer, w[k], z.map(|z| z[k]));
    }
}

/// Expand a packed upper triangle to a symmetric nalgebra matrix.
pub(crate) fn packed_to_dmatrix(packed: &[f64], p: usize) -> DMatrix<f64> {
    let mut matrix = DMatrix::zeros(p, p);
    let mut packed_idx = 0usize;
    for i in 0..p {
        for j in i..p {
            let value = packed[packed_idx];
            matrix[(i, j)] = value;
            matrix[(j, i)] = value;
            packed_idx += 1;
        }
    }
    matrix
}

/// Expand a packed upper triangle to a symmetric ndarray matrix.
pub(crate) fn packed_to_array2(packed: &[f64], p: usize) -> Array2<f64> {
    let mut matrix = Array2::zeros((p, p));
    let mut packed_idx = 0usize;
    for i in 0..p {
        for j in i..p {
            let value = packed[packed_idx];
            matrix[[i, j]] = value;
            matrix[[j, i]] = value;
            packed_idx += 1;
        }
    }
    matrix
}

pub struct SparseRowCache {
    n: usize,
    p: usize,
    data_ptr: usize,
    pub(crate) offsets: Vec<usize>,
    pub(crate) indices: Vec<u32>,
    pub(crate) values: Vec<f64>,
    pub(crate) packed_offsets: Vec<usize>,
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

fn should_build_sparse_row_cache_with_min_cells(
    n: usize,
    p: usize,
    density: f64,
    min_cells: usize,
) -> bool {
    if n == 0 || p < 16 {
        return false;
    }
    let estimated_nnz = sparse_row_cache_estimated_nnz(density, n, p);
    density <= SPARSE_ROW_CACHE_DENSITY_THRESHOLD
        && estimated_nnz <= 60_000_000.0
        && n.saturating_mul(p) >= min_cells
}

#[cfg(test)]
fn should_build_sparse_row_cache(n: usize, p: usize, density: f64) -> bool {
    should_build_sparse_row_cache_with_min_cells(n, p, density, 10_000_000)
}

pub fn build_sparse_row_cache_if_beneficial(x: ArrayView2<'_, f64>) -> Option<SparseRowCache> {
    build_sparse_row_cache_with_min_cells(x, 10_000_000)
}

/// Build a sparse row cache for callers that reuse the same design matrix across
/// many weighted Gram products, such as multinomial Hessian assembly.
pub fn build_sparse_row_cache_for_repeated_xtwx(x: ArrayView2<'_, f64>) -> Option<SparseRowCache> {
    build_sparse_row_cache_with_min_cells(x, 1_000_000)
}

pub(crate) fn build_sparse_row_cache_with_min_cells(
    x: ArrayView2<'_, f64>,
    min_cells: usize,
) -> Option<SparseRowCache> {
    let n = x.nrows();
    let p = x.ncols();
    let x_slice = x.as_slice()?;
    let density = sampled_density(x_slice, n, p);
    if !should_build_sparse_row_cache_with_min_cells(n, p, density, min_cells) {
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

pub(crate) fn packed_upper_offsets(p: usize) -> Vec<usize> {
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

/// The pairwise sparse Gram kernel over one row's nonzeros (ascending columns):
/// `accumulate_cached_sparse_row_pairwise`'s products in its order.
#[inline(always)]
pub(crate) fn accumulate_sparse_slices_pairwise(
    indices: &[u32],
    values: &[f64],
    packed_offsets: &[usize],
    xtx_local: &mut [f64],
    xtz_local: &mut [f64],
    wk: f64,
    wz: f64,
) {
    debug_assert_eq!(indices.len(), values.len());
    let len = indices.len();
    let xtx_ptr = xtx_local.as_mut_ptr();
    let xtz_ptr = xtz_local.as_mut_ptr();
    let idx_ptr = indices.as_ptr();
    let val_ptr = values.as_ptr();
    for a in 0..len {
        // SAFETY: indices and values have the same length; every index is a column
        // < p, packed_offsets has length p, and base + j (j >= i) is the packed
        // upper-triangle offset of (i, j).
        unsafe {
            let i = *idx_ptr.add(a) as usize;
            let xki = *val_ptr.add(a);
            let xki_w = xki * wk;
            *xtz_ptr.add(i) += xki * wz;
            let base = *packed_offsets.get_unchecked(i) - i;
            let unrolled_end = sparse_pairwise_unrolled_end(a, len);
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
            for b in unrolled_end..len {
                let j = *idx_ptr.add(b) as usize;
                *xtx_ptr.add(base + j) += xki_w * *val_ptr.add(b);
            }
        }
    }
}

#[inline(always)]
pub(crate) fn accumulate_cached_sparse_row_pairwise(
    cache: &SparseRowCache,
    start: usize,
    end: usize,
    xtx_local: &mut [f64],
    xtz_local: &mut [f64],
    wk: f64,
    wz: f64,
) {
    accumulate_sparse_slices_pairwise(
        &cache.indices[start..end],
        &cache.values[start..end],
        &cache.packed_offsets,
        xtx_local,
        xtz_local,
        wk,
        wz,
    );
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
    let coef: Vec<f64> = coefficients.iter().copied().collect();
    let mut output = vec![0.0; n];
    match x.as_slice() {
        Some(x_slice) => {
            output
                .par_chunks_mut(GRAM_BLOCK_ROWS)
                .enumerate()
                .for_each(|(chunk_idx, chunk)| {
                    let first = chunk_idx * GRAM_BLOCK_ROWS;
                    for (local, out) in chunk.iter_mut().enumerate() {
                        let row = first + local;
                        *out = row_dot(&x_slice[row * p..(row + 1) * p], &coef);
                    }
                });
        }
        None => {
            output
                .par_chunks_mut(GRAM_BLOCK_ROWS)
                .enumerate()
                .for_each(|(chunk_idx, chunk)| {
                    let first = chunk_idx * GRAM_BLOCK_ROWS;
                    for (local, out) in chunk.iter_mut().enumerate() {
                        let mut sum = 0.0;
                        for (&value, &beta) in x.row(first + local).iter().zip(coef.iter()) {
                            sum += value * beta;
                        }
                        *out = sum;
                    }
                });
        }
    }
    Array1::from_vec(output)
}

/// The linear predictor of one row: Σ_j x_j·β_j in column order (§5.1).
#[inline(always)]
pub(crate) fn row_dot(row: &[f64], coefficients: &[f64]) -> f64 {
    let mut sum = 0.0;
    for (&value, &beta) in row.iter().zip(coefficients) {
        sum += value * beta;
    }
    sum
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
    let coef: Vec<f64> = coefficients.iter().copied().collect();
    let n = cache.n;
    let mut output = vec![0.0; n];
    // Summing only the nonzero products in column order gives the dense
    // kernel's bits: every skipped product is an exact zero.
    output
        .par_chunks_mut(GRAM_BLOCK_ROWS)
        .enumerate()
        .for_each(|(chunk_idx, chunk)| {
            let first = chunk_idx * GRAM_BLOCK_ROWS;
            for (local, out) in chunk.iter_mut().enumerate() {
                let row = first + local;
                let mut sum = 0.0;
                for pos in cache.offsets[row]..cache.offsets[row + 1] {
                    sum += cache.values[pos] * coef[cache.indices[pos] as usize];
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

fn sparse_pairwise_unrolled_end(start: usize, end: usize) -> usize {
    start + ((end - start) / 4) * 4
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
    any_positive_weight: bool,
    l2_penalty: f64,
    l2_penalty_factors: Option<&[f64]>,
    penalize_intercept: bool,
    p: usize,
) -> bool {
    if l2_penalty <= 0.0 || !any_positive_weight {
        return false;
    }
    let start_idx = if penalize_intercept { 0 } else { 1 };
    (start_idx..p).all(|j| l2_penalty_factors.map_or(1.0, |factors| factors[j]) > 0.0)
}

fn should_apply_l2_penalty(l2_penalty: f64) -> bool {
    l2_penalty > 0.0
}

fn penalty_matrix_shape_matches(p: usize, rows: usize, cols: usize) -> bool {
    rows == p && cols == p
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
    let n = x.nrows();
    let p = x.ncols();
    assert_eq!(w.len(), n, "weight length must equal X rows");
    let gram = match (x.as_slice(), w.as_slice()) {
        (Some(x_slice), Some(w_slice)) => {
            let use_sparse = should_use_sparse_xtwx_kernel(x_slice, n, p);
            reduce_rows(GramPartial::zeros(p), n, |partial, rows| {
                if use_sparse {
                    accumulate_sparse_scan_rows(partial, x_slice, w_slice, None, p, rows);
                } else {
                    accumulate_dense_rows(partial, x_slice, w_slice, None, p, rows);
                }
            })
        }
        _ => reduce_rows(GramPartial::zeros(p), n, |partial, rows| {
            accumulate_strided_rows(partial, x, w, None, rows);
        }),
    };
    packed_to_array2(&gram.xtx_packed, p)
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

    let gram = reduce_rows(GramPartial::zeros(p), n, |partial, rows| {
        accumulate_cached_rows(partial, cache, w_slice, None, rows);
    });
    Ok(packed_to_array2(&gram.xtx_packed, p))
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
    use std::{ffi::OsString, sync::Mutex};

    #[test]
    fn test_final_statistics_match_array_definitions_bitwise() {
        use crate::diagnostics::{null_deviance_for_family, pearson_chi2};
        use crate::families::{
            BinomialFamily, GammaFamily, GaussianFamily, NegativeBinomialFamily, TweedieFamily,
        };
        let n = 2 * crate::rows::GRAM_BLOCK_ROWS + 211;
        let x = Array2::from_shape_fn((n, 3), |(i, j)| match j {
            0 => 1.0,
            1 => ((i * 37) % 101) as f64 / 101.0 - 0.5,
            _ => ((i * 13) % 7) as f64 / 7.0,
        });
        let positive = Array1::from_shape_fn(n, |i| 0.5 + ((i * 11) % 17) as f64 / 4.0);
        let counts = Array1::from_shape_fn(n, |i| ((i * 7) % 5) as f64);
        let binary = Array1::from_shape_fn(n, |i| ((i * 3) % 4 == 0) as u8 as f64);
        let weights = Array1::from_shape_fn(n, |i| 0.5 + (i % 3) as f64 * 0.75);
        let offset = Array1::from_shape_fn(n, |i| ((i % 5) as f64 - 2.0) * 0.05);
        let tweedie = TweedieFamily::new(1.5).expect("tweedie");
        let negbin = NegativeBinomialFamily::new(2.0).expect("negbin");
        let cases: Vec<(&dyn Family, &dyn Link, &Array1<f64>)> = vec![
            (&GaussianFamily, &IdentityLink, &positive),
            (&PoissonFamily, &LogLink, &counts),
            (&GammaFamily, &LogLink, &positive),
            (&tweedie, &LogLink, &positive),
            (&BinomialFamily, &LogitLink, &binary),
            (&negbin, &LogLink, &counts),
        ];
        for (family, link, y) in cases {
            for (w, o) in [
                (None, None),
                (Some(&weights), None),
                (Some(&weights), Some(&offset)),
            ] {
                let result =
                    fit_glm_unified(y, x.view(), family, link, &FitConfig::default(), o, w, None)
                        .expect("fit");
                let stats = result.statistics.clone().expect("statistics");
                let maybe_w = w.filter(|_| !stats.all_unit_weights);
                let mu = &result.fitted_values;
                let chi2 = pearson_chi2(y, mu, family, maybe_w);
                assert_eq!(
                    stats.pearson_chi2.to_bits(),
                    chi2.to_bits(),
                    "{}",
                    family.name()
                );
                let df = n - 3;
                let scale = if family.fixed_dispersion() {
                    1.0
                } else {
                    chi2 / df as f64
                };
                assert_eq!(stats.scale.to_bits(), scale.to_bits(), "{}", family.name());
                let llf = family.log_likelihood(y, mu, scale, maybe_w);
                assert_eq!(
                    stats.log_likelihood.to_bits(),
                    llf.to_bits(),
                    "{} llf",
                    family.name()
                );
                let null = null_deviance_for_family(y, family, link, maybe_w, o);
                assert_eq!(
                    stats.null_deviance.to_bits(),
                    null.to_bits(),
                    "{} null",
                    family.name()
                );
            }
        }
    }

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

        fn variance_at(&self, _mu: f64) -> f64 {
            1.0
        }

        fn initialize_mu_at(&self, _yi: f64, _y_mean: f64) -> f64 {
            f64::NAN
        }

        fn is_valid_mu_at(&self, mu: f64) -> bool {
            mu.is_finite()
        }

        fn clamp_mu_at(&self, mu: f64) -> f64 {
            if mu.is_finite() {
                mu
            } else {
                0.0
            }
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
            row_kernel: RowKernel::Structural(Some(3)),
        };
        let from_legacy = FitConfig::from(&legacy);
        assert_eq!(from_legacy.row_kernel, RowKernel::Structural(Some(3)));
        assert_eq!(from_legacy.to_irls_config().row_kernel, legacy.row_kernel);
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
        assert!(!irls_step_converged(100.0, 99.0, 1e-4, 1e-4));
        assert!(irls_step_converged(100.0, 99.999, 1e-5, 1e-4));
    }

    #[test]
    fn test_profile_helpers_and_env_flag_have_exact_contracts() {
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
        assert!(!should_build_sparse_row_cache_with_min_cells(
            62_499, 16, 0.0, 1_000_000
        ));
        assert!(should_build_sparse_row_cache_with_min_cells(
            62_500, 16, 0.0, 1_000_000
        ));
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

        let mut expected_xtx = vec![0.0; crate::rows::packed_len(p)];
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

        let mut actual_xtx = vec![0.0; crate::rows::packed_len(p)];
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
        let mut cached_xtx = vec![0.0; crate::rows::packed_len(p)];
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
    fn test_ridge_scalar_helpers_have_exact_contracts() {
        assert!(!ridge_system_is_positive_definite_fast_path(
            true, 0.0, None, false, 3
        ));
        assert!(!ridge_system_is_positive_definite_fast_path(
            false, 0.5, None, false, 3
        ));
        assert!(ridge_system_is_positive_definite_fast_path(
            true, 0.5, None, false, 3
        ));
        assert!(!ridge_system_is_positive_definite_fast_path(
            true,
            0.5,
            Some(&[1.0, 0.0, 2.0]),
            false,
            3
        ));
        assert!(ridge_system_is_positive_definite_fast_path(
            true,
            0.5,
            Some(&[0.0, 1.0, 2.0]),
            false,
            3
        ));

        assert!(!should_apply_l2_penalty(0.0));
        assert!(!should_apply_l2_penalty(-1.0));
        assert!(should_apply_l2_penalty(1e-12));
        assert!(penalty_matrix_shape_matches(2, 2, 2));
        assert!(!penalty_matrix_shape_matches(2, 1, 2));
        assert!(!penalty_matrix_shape_matches(2, 2, 1));
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
            &finite_coefficients_or_none(array![2.0, -3.0])
                .expect("finite coefficients should be retained"),
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

        assert!(final_extraction_deviance_acceptable(4.0, 4.0));
        assert!(!final_extraction_deviance_acceptable(4.1, 4.0));
        assert!(!final_extraction_deviance_acceptable(f64::NAN, 4.0));
        assert!(should_recompute_final_state(false));
        assert!(!should_recompute_final_state(true));
        assert!(should_warn_nonconverged(false));
        assert!(!should_warn_nonconverged(true));
    }

    #[test]
    fn test_weighted_gram_is_block_ordered_and_kernel_independent() {
        // Rows spanning several reduction blocks, values over many magnitudes and
        // a sparse pattern, so summation order and kernel choice would show.
        let n = 3 * GRAM_BLOCK_ROWS + 77;
        let p = 18;
        let mut x = Array2::<f64>::zeros((n, p));
        for row in 0..n {
            for col in 0..p {
                if (row + 3 * col) % 4 == 0 {
                    let magnitude = 10f64.powi(((row * 7 + col) % 13) as i32 - 6);
                    x[[row, col]] = ((row * 31 + col * 17) % 97) as f64 / 97.0 * magnitude;
                }
            }
        }
        let w: Array1<f64> = (0..n).map(|row| 0.25 + (row % 11) as f64 * 0.5).collect();
        let z: Array1<f64> = (0..n).map(|row| ((row % 29) as f64 - 14.0) * 0.3).collect();
        let x_slice = x.as_slice().expect("contiguous test matrix");
        let w_slice = w.as_slice().expect("contiguous test weights");
        let z_slice = z.as_slice().expect("contiguous test response");

        // Reference: each block summed row by row from zero, blocks added in order.
        let mut expected = GramPartial::zeros(p);
        for block_start in (0..n).step_by(GRAM_BLOCK_ROWS) {
            let mut partial = GramPartial::zeros(p);
            for row in block_start..(block_start + GRAM_BLOCK_ROWS).min(n) {
                let wk = w[row];
                let wz = wk * z[row];
                partial.ztwz += wz * z[row];
                let mut packed = 0;
                for i in 0..p {
                    let xki = x[[row, i]];
                    partial.xtz[i] += xki * wz;
                    for j in i..p {
                        partial.xtx_packed[packed] += (xki * wk) * x[[row, j]];
                        packed += 1;
                    }
                }
            }
            crate::rows::Partial::add_assign(&mut expected, &partial);
        }

        let cache = build_sparse_row_cache_with_min_cells(x.view(), 0)
            .expect("sparse test design builds a cache");
        let gram_with = |threads: usize, kernel: &str| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("test thread pool should build");
            pool.install(|| {
                reduce_rows(GramPartial::zeros(p), n, |partial, rows| match kernel {
                    "dense" => {
                        accumulate_dense_rows(partial, x_slice, w_slice, Some(z_slice), p, rows)
                    }
                    "sparse" => accumulate_sparse_scan_rows(
                        partial,
                        x_slice,
                        w_slice,
                        Some(z_slice),
                        p,
                        rows,
                    ),
                    _ => accumulate_cached_rows(partial, &cache, w_slice, Some(z_slice), rows),
                })
            })
        };

        let bits = |gram: &GramPartial| -> Vec<u64> {
            gram.xtx_packed
                .iter()
                .chain(gram.xtz.iter())
                .chain(std::iter::once(&gram.ztwz))
                .map(|v| v.to_bits())
                .collect()
        };
        let expected_bits = bits(&expected);
        for threads in [1, 3, 8] {
            for kernel in ["dense", "sparse", "cached"] {
                assert_eq!(
                    bits(&gram_with(threads, kernel)),
                    expected_bits,
                    "{kernel} kernel with {threads} threads"
                );
            }
        }

        let full = compute_xtwx(x.view(), &w);
        let materialized = packed_to_array2(&expected.xtx_packed, p);
        assert!(full
            .iter()
            .zip(materialized.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits()));
        assert_eq!(full[[2, 5]].to_bits(), full[[5, 2]].to_bits());
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

        let system = SolveSystem {
            gram: GramPartial::zeros(x.ncols()),
            positive_weight_rows: 1.0,
        };
        let factors_err = solve_penalized_system(&system, None, 1.0, Some(&[1.0]), false, true)
            .expect_err("L2 penalty factors must match X columns");
        assert!(matches!(
            factors_err,
            RustyStatsError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn test_cholesky_lu_fallback_and_ridge_fast_path_contracts() {
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
            false, 1.0, None, true, 2
        ));
        assert!(!ridge_system_is_positive_definite_fast_path(
            true, 0.0, None, true, 2
        ));
        assert!(!ridge_system_is_positive_definite_fast_path(
            true,
            1.0,
            Some(&[1.0, 0.0]),
            true,
            2
        ));

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

        let cached = reduce_rows(GramPartial::zeros(p), n, |partial, rows| {
            accumulate_cached_rows(
                partial,
                &cache,
                w.as_slice().expect("contiguous test array"),
                Some(z.as_slice().expect("contiguous test array")),
                rows,
            )
        });
        let actual_xtx = packed_to_dmatrix(&cached.xtx_packed, p);
        let actual_xtz = cached.xtz;
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
    }

    #[test]
    fn test_sparse_cache_builder_thresholds_and_core_kernel_edge_paths() {
        assert_eq!(sampled_density(&[], 0, 16), 0.0);

        let x_empty = Array2::<f64>::zeros((0, 20));
        let z_empty = Array1::<f64>::zeros(0);
        let w_empty = Array1::<f64>::zeros(0);
        let (xtx, xtz) =
            compute_xtwx_xtwz(x_empty.view(), &z_empty, &w_empty).expect("empty dense kernel");
        assert_eq!(xtx.shape(), (20, 20));
        assert_eq!(xtz.len(), 20);

        let empty_gram = reduce_rows(GramPartial::zeros(20), 0, |partial, rows| {
            accumulate_sparse_scan_rows(partial, &[], &[], Some(&[]), 20, rows);
        });
        assert_eq!(empty_gram, GramPartial::zeros(20));

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
        let (xtx, xtz) =
            compute_xtwx_xtwz(x.view(), &initial_eta, &weights).expect("projection system");
        let initial_coef =
            cholesky_solve_coefficients(xtx, &xtz).expect("initial projection should be valid");
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
