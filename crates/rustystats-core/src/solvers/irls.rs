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
const SPARSE_XTWX_THREAD_CAP: usize = 16;
const SPARSE_XTWX_LOCAL_MATRIX_CAP_THRESHOLD_BYTES: usize = 4 * 1024 * 1024;

use nalgebra::{DMatrix, DVector};
use ndarray::{Array1, Array2, ArrayView2};
use rayon::prelude::*;

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
        if !config.regularization.penalty.is_none() && !config.regularization.penalty.is_smooth() {
            standardization.validate(x.ncols())?;
            let l2_penalty = config.regularization.penalty.l2_penalty();
            if l2_penalty > 0.0
                && !config.regularization.penalty.requires_coordinate_descent()
                && standardization.center.iter().all(|&center| center == 0.0)
            {
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
    // -------------------------------------------------------------------------
    // Step 0: Validate inputs and set up offset/weights
    // -------------------------------------------------------------------------
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

    // -------------------------------------------------------------------------
    // Step 1: Initialize μ (from coefficients if warm-starting, else from family)
    // -------------------------------------------------------------------------
    let mut mu = if let Some(init) = init_coefficients {
        if init.len() != p {
            return Err(RustyStatsError::dim_mismatch(
                p,
                init.len(),
                "init_coefficients length vs X columns",
            ));
        }
        iter_coefficients = init.clone();
        let eta_init = matrix_vector_dot_cached(x, init, sparse_cache) + &offset_vec;
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

    // -------------------------------------------------------------------------
    // Step 2: Initialize linear predictor η = Xβ + offset
    // -------------------------------------------------------------------------
    // Family initializers often produce a near-saturated μ that is not exactly
    // representable by the model matrix. Project it into coefficient space first
    // so the retained iterate, step-halving baseline, and returned coefficients
    // all describe the same fitted state (RS-ACT-007).
    let mut eta = link.link(&mu);
    if init_coefficients.is_none() {
        let eta_no_offset = &eta - &offset_vec;
        match solve_weighted_least_squares_penalized(
            x,
            &eta_no_offset,
            &prior_weights_vec,
            l2_penalty,
            l2_penalty_factors,
            penalize_intercept,
            true, // init projection covariance is discarded
            sparse_cache,
        ) {
            Ok((mut coef, _)) if !coef.iter().any(|&c| c.is_nan() || c.is_infinite()) => {
                for &idx in &config.nonneg_indices {
                    if idx < coef.len() && coef[idx] < 0.0 {
                        coef[idx] = 0.0;
                    }
                }
                for &idx in &config.nonpos_indices {
                    if idx < coef.len() && coef[idx] > 0.0 {
                        coef[idx] = 0.0;
                    }
                }
                iter_coefficients = coef;
            }
            _ => {
                warnings.push(
                    "Initial coefficient projection failed. Starting IRLS from zero coefficients."
                        .to_string(),
                );
            }
        }
        eta = &matrix_vector_dot_cached(x, &iter_coefficients, sparse_cache) + &offset_vec;
        mu = family.clamp_mu(&link.inverse(&eta));
    }

    // -------------------------------------------------------------------------
    // Step 3: Calculate initial deviance
    // -------------------------------------------------------------------------
    let mut deviance = family.deviance(y, &mu, Some(&prior_weights_vec));
    let mut objective = penalized_irls_objective(
        deviance,
        &iter_coefficients,
        l2_penalty,
        l2_penalty_factors,
        penalize_intercept,
    );
    let mut objective_old: f64;

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
    let has_constraints = !config.nonneg_indices.is_empty() || !config.nonpos_indices.is_empty();
    let mut best_objective = f64::INFINITY;
    let mut best_deviance = f64::INFINITY;
    let mut best_coefficients = iter_coefficients.clone();
    let mut best_mu = mu.clone();
    let mut best_eta = eta.clone();
    let mut best_weights = final_weights.clone();
    let mut constrained_best_stale_iterations = 0usize;
    let mut constrained_best_plateau = false;
    let use_poisson_log_weight_buffers =
        family.name().eq_ignore_ascii_case("poisson") && link.name() == "log";
    let weight_buffer_len = if use_poisson_log_weight_buffers { n } else { 0 };
    let mut irls_weights_buf = Array1::zeros(weight_buffer_len);
    let mut combined_weights_buf = Array1::zeros(weight_buffer_len);
    let mut working_response_buf = Array1::zeros(weight_buffer_len);

    while iteration < config.max_iterations {
        iteration += 1;

        // ---------------------------------------------------------------------
        // Step 4a: Compute working weights W
        // ---------------------------------------------------------------------
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

        // ---------------------------------------------------------------------
        // Step 4c: Solve weighted least squares: (X'WX)β = X'Wz
        // ---------------------------------------------------------------------
        // This is the core linear algebra step.
        // We're finding β that minimizes: Σ w_i (z_i - x_i'β)²
        // ---------------------------------------------------------------------
        let (mut new_coefficients, _) = solve_weighted_least_squares_penalized(
            x,
            working_response,
            combined_weights,
            l2_penalty,
            l2_penalty_factors,
            penalize_intercept,
            true, // per-iteration inverse is unused; compute final covariance once
            sparse_cache,
        )?;

        // Check for NaN in coefficients - indicates numerical instability
        if new_coefficients
            .iter()
            .any(|&c| c.is_nan() || c.is_infinite())
        {
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
        // Project non-negative constrained coefficients to be >= 0 (for ms(), pos())
        for &idx in &config.nonneg_indices {
            if idx < new_coefficients.len() && new_coefficients[idx] < 0.0 {
                new_coefficients[idx] = 0.0;
            }
        }
        // Project non-positive constrained coefficients to be <= 0 (for neg())
        for &idx in &config.nonpos_indices {
            if idx < new_coefficients.len() && new_coefficients[idx] > 0.0 {
                new_coefficients[idx] = 0.0;
            }
        }

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

        let project = |coef: &mut Array1<f64>| {
            for &idx in &config.nonneg_indices {
                if idx < coef.len() && coef[idx] < 0.0 {
                    coef[idx] = 0.0;
                }
            }
            for &idx in &config.nonpos_indices {
                if idx < coef.len() && coef[idx] > 0.0 {
                    coef[idx] = 0.0;
                }
            }
        };

        // Try the full Newton step (with constraints projected).
        let mut trial_coefficients = new_coefficients.clone();
        project(&mut trial_coefficients);
        let mut eta_new =
            &matrix_vector_dot_cached(x, &trial_coefficients, sparse_cache) + &offset_vec;
        let mut mu_new = family.clamp_mu(&link.inverse(&eta_new));
        let mut deviance_new = family.deviance(y, &mu_new, Some(&prior_weights_vec));
        let mut objective_new = penalized_irls_objective(
            deviance_new,
            &trial_coefficients,
            l2_penalty,
            l2_penalty_factors,
            penalize_intercept,
        );

        let mut step_accepted = eta_new.iter().all(|v| v.is_finite())
            && mu_new.iter().all(|v| v.is_finite())
            && objective_new.is_finite()
            && objective_new <= accept_threshold;

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
                    .map(|(&old, &new)| (1.0 - step_size) * old + step_size * new)
                    .collect();
                project(&mut blended);
                let e = &matrix_vector_dot_cached(x, &blended, sparse_cache) + &offset_vec;
                let m = family.clamp_mu(&link.inverse(&e));
                let d = family.deviance(y, &m, Some(&prior_weights_vec));
                let o = penalized_irls_objective(
                    d,
                    &blended,
                    l2_penalty,
                    l2_penalty_factors,
                    penalize_intercept,
                );
                if e.iter().all(|v| v.is_finite())
                    && m.iter().all(|v| v.is_finite())
                    && o.is_finite()
                    && o <= accept_threshold
                {
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
            break;
        }

        new_coefficients = trial_coefficients;
        eta = eta_new;
        mu = mu_new;
        deviance = deviance_new;
        objective = objective_new;

        // Relative change in the same objective used for acceptance.
        let rel_change = if objective_old.abs() > ZERO_TOL {
            (objective_old - objective).abs() / objective_old.abs()
        } else {
            (objective_old - objective).abs()
        };

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
            if objective < best_objective {
                best_objective = objective;
                best_deviance = deviance;
                best_coefficients = iter_coefficients.clone();
                best_mu = mu.clone();
                best_eta = eta.clone();
                best_weights = irls_weights.clone();
                constrained_best_stale_iterations = 0;
            } else if best_objective.is_finite() {
                constrained_best_stale_iterations += 1;
                if constrained_best_stale_iterations >= CONSTRAINED_BEST_EARLY_STOP_PATIENCE {
                    constrained_best_plateau = true;
                    final_weights.assign(irls_weights);
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
            break;
        }

        // Store for final iteration
        final_weights.assign(irls_weights);
    }

    // -------------------------------------------------------------------------
    // Step 5: Extract final coefficients
    // -------------------------------------------------------------------------
    // For constrained problems, use the best solution found during iteration
    // (objective can increase due to projection, so last iteration may not be best)
    let (final_mu, final_eta, final_deviance, use_coefficients) = if has_constraints
        && (best_objective < objective || (constrained_best_plateau && best_objective.is_finite()))
    {
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
    let (final_coefficients, cov_unscaled) = if config.skip_covariance {
        (use_coefficients, Array2::zeros((p, p)))
    } else {
        // Compute working response accounting for offset
        let eta_no_offset: Array1<f64> = final_eta
            .iter()
            .zip(offset_vec.iter())
            .map(|(&e, &o)| e - o)
            .collect();

        // Combine prior weights with final IRLS weights
        let combined_final_weights: Array1<f64> = prior_weights_vec
            .iter()
            .zip(final_weights.iter())
            .map(|(&pw, &iw)| pw * iw)
            .collect();

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
        ) {
            Ok((coef, cov)) if !coef.iter().any(|&c| c.is_nan() || c.is_infinite()) => {
                let cov = if cov.iter().all(|v| v.is_finite()) {
                    cov
                } else {
                    return Err(RustyStatsError::NumericalError(
                        "Final covariance extraction produced NaN/Inf. \
                        This usually indicates numerical instability or a nearly singular design matrix."
                            .to_string(),
                    ));
                };
                // For constrained problems, apply projection and check if it's better than stored best
                if has_constraints {
                    let mut proj_coef = coef;
                    for &idx in &config.nonneg_indices {
                        if idx < proj_coef.len() && proj_coef[idx] < 0.0 {
                            proj_coef[idx] = 0.0;
                        }
                    }
                    for &idx in &config.nonpos_indices {
                        if idx < proj_coef.len() && proj_coef[idx] > 0.0 {
                            proj_coef[idx] = 0.0;
                        }
                    }
                    // Check if this extraction is better
                    let eta_check = matrix_vector_dot(x, &proj_coef);
                    let eta_full: Array1<f64> = eta_check
                        .iter()
                        .zip(offset_vec.iter())
                        .map(|(&e, &o)| e + o)
                        .collect();
                    let mu_check = family.clamp_mu(&link.inverse(&eta_full));
                    let dev_check = family.deviance(y, &mu_check, Some(&prior_weights_vec));
                    if dev_check <= final_deviance {
                        (proj_coef, cov)
                    } else {
                        (use_coefficients, cov)
                    }
                } else {
                    // Unconstrained: guard against a final extraction that is worse
                    // than the loop's retained iterate (RS-ACT-007).
                    let eta_full = &matrix_vector_dot(x, &coef) + &offset_vec;
                    let mu_check = family.clamp_mu(&link.inverse(&eta_full));
                    let dev_check = family.deviance(y, &mu_check, Some(&prior_weights_vec));
                    if dev_check.is_finite() && dev_check <= final_deviance {
                        (coef, cov)
                    } else {
                        (use_coefficients, cov)
                    }
                }
            }
            Ok((_coef, cov)) => {
                // Final extraction failed or produced NaN - use stored coefficients
                warnings.push(
                    "Final coefficient extraction produced NaN/Inf. \
                    Using coefficients from best iteration instead. This may indicate numerical instability."
                        .to_string(),
                );
                if use_coefficients
                    .iter()
                    .any(|&c| c.is_nan() || c.is_infinite())
                {
                    return Err(RustyStatsError::NumericalError(
                        "IRLS produced NaN or infinite coefficients. This usually indicates: \
                         (1) severe multicollinearity in predictors, \
                         (2) extreme scale differences between variables, or \
                         (3) separation in binary response data. \
                         Try standardizing continuous predictors or removing correlated terms."
                            .to_string(),
                    ));
                }
                let cov_unscaled = if cov.iter().all(|v| v.is_finite()) {
                    cov
                } else {
                    return Err(RustyStatsError::NumericalError(
                        "Final covariance extraction produced NaN/Inf. \
                        This usually indicates numerical instability or a nearly singular design matrix."
                            .to_string(),
                    ));
                };
                (use_coefficients, cov_unscaled)
            }
            Err(_) => {
                return Err(RustyStatsError::LinearAlgebraError(
                    "Final coefficient/covariance extraction failed. \
                    This often indicates multicollinearity in predictors."
                        .to_string(),
                ));
            }
        }
    };

    // Apply coefficient sign constraints to final coefficients (for unconstrained path)
    let mut final_coefficients = final_coefficients;
    if !has_constraints {
        for &idx in &config.nonneg_indices {
            if idx < final_coefficients.len() && final_coefficients[idx] < 0.0 {
                final_coefficients[idx] = 0.0;
            }
        }
        for &idx in &config.nonpos_indices {
            if idx < final_coefficients.len() && final_coefficients[idx] > 0.0 {
                final_coefficients[idx] = 0.0;
            }
        }
    }

    // Recompute final fitted values and deviance with the chosen coefficients
    let final_eta_base = matrix_vector_dot(x, &final_coefficients);
    let final_eta: Array1<f64> = final_eta_base
        .iter()
        .zip(offset_vec.iter())
        .map(|(&e, &o)| e + o)
        .collect();
    let final_mu = family.clamp_mu(&link.inverse(&final_eta));
    let final_deviance = family.deviance(y, &final_mu, Some(&prior_weights_vec));

    let solver_status = if step_halving_failed {
        "step_halving_no_improvement"
    } else if converged {
        "converged"
    } else {
        "max_iterations"
    };
    if !converged {
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

    if should_use_sparse_xtwx_kernel(x_slice, n, p) {
        return compute_xtwx_xtwz_sparse(x_slice, z_slice, w_slice, n, p);
    }

    // Core-adaptive chunking: split rows across all available threads so this
    // kernel (the dominant per-iteration cost) saturates the CPU, instead of
    // the ~4 chunks the old fixed 8192 produced for a 25k-row CV fold.
    let chunk_size = n.div_ceil(rayon::current_num_threads()).max(1);
    let num_chunks = n.div_ceil(chunk_size);

    let upper_len = packed_upper_len(p);
    let (xtx_data, xtz_data): (Vec<f64>, Vec<f64>) = (0..num_chunks)
        .into_par_iter()
        .map(|chunk_idx| {
            let chunk_start = chunk_idx * chunk_size;
            let chunk_end = (chunk_start + chunk_size).min(n);
            let mut xtx_local = vec![0.0; upper_len];
            let mut xtz_local = vec![0.0; p];

            for k in chunk_start..chunk_end {
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
            }
            (xtx_local, xtz_local)
        })
        .reduce_with(|(mut a_xtx, mut a_xtz), (b_xtx, b_xtz)| {
            for i in 0..a_xtx.len() {
                a_xtx[i] += b_xtx[i];
            }
            for i in 0..a_xtz.len() {
                a_xtz[i] += b_xtz[i];
            }
            (a_xtx, a_xtz)
        })
        .unwrap_or_else(|| (vec![0.0; upper_len], vec![0.0; p]));

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
    let xtz = DVector::from_vec(xtz_data);

    Ok((xtx, xtz))
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

pub fn build_sparse_row_cache_if_beneficial(x: ArrayView2<'_, f64>) -> Option<SparseRowCache> {
    let n = x.nrows();
    let p = x.ncols();
    if n == 0 || p < 16 {
        return None;
    }
    let x_slice = x.as_slice()?;
    let density = sampled_density(x_slice, n, p);
    let estimated_nnz = density * (n as f64) * (p as f64);
    if density > 0.15 || estimated_nnz > 60_000_000.0 || n.saturating_mul(p) < 10_000_000 {
        return None;
    }

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

fn compute_xtwx_xtwz_sparse_cached(
    cache: &SparseRowCache,
    z: &Array1<f64>,
    w: &Array1<f64>,
) -> Result<(DMatrix<f64>, DVector<f64>)> {
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

    let (xtx_data, xtz_data): (Vec<f64>, Vec<f64>) = (0..num_chunks)
        .into_par_iter()
        .map(|chunk_idx| {
            let chunk_start = chunk_idx * chunk_size;
            let chunk_end = (chunk_start + chunk_size).min(n);
            let mut xtx_local = vec![0.0; upper_len];
            let mut xtz_local = vec![0.0; p];

            for row in chunk_start..chunk_end {
                let start = cache.offsets[row];
                let end = cache.offsets[row + 1];
                if start == end {
                    continue;
                }

                let wk = unsafe { *w_slice.get_unchecked(row) };
                let zk = unsafe { *z_slice.get_unchecked(row) };
                let wz = wk * zk;

                for a in start..end {
                    let i = unsafe { *cache.indices.get_unchecked(a) as usize };
                    let xki = unsafe { *cache.values.get_unchecked(a) };
                    let xki_w = xki * wk;
                    unsafe { *xtz_local.get_unchecked_mut(i) += xki * wz };
                    let packed_row = unsafe { *cache.packed_offsets.get_unchecked(i) };

                    for b in a..end {
                        let j = unsafe { *cache.indices.get_unchecked(b) as usize };
                        let xkj = unsafe { *cache.values.get_unchecked(b) };
                        let packed_idx = packed_row + (j - i);
                        unsafe { *xtx_local.get_unchecked_mut(packed_idx) += xki_w * xkj };
                    }
                }
            }
            (xtx_local, xtz_local)
        })
        .reduce_with(|(mut a_xtx, mut a_xtz), (b_xtx, b_xtz)| {
            for i in 0..a_xtx.len() {
                a_xtx[i] += b_xtx[i];
            }
            for i in 0..a_xtz.len() {
                a_xtz[i] += b_xtz[i];
            }
            (a_xtx, a_xtz)
        })
        .unwrap_or_else(|| (vec![0.0; upper_len], vec![0.0; p]));

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
    Ok((xtx, DVector::from_vec(xtz_data)))
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
    if n.saturating_mul(p) < 1_000_000 || rayon::current_num_threads() <= 1 {
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
fn matrix_vector_dot_cached(
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
    if n == 0 || p < 16 {
        return false;
    }
    sampled_density(x_slice, n, p) <= 0.35
}

#[inline]
fn compute_xtwx_xtwz_sparse(
    x_slice: &[f64],
    z_slice: &[f64],
    w_slice: &[f64],
    n: usize,
    p: usize,
) -> Result<(DMatrix<f64>, DVector<f64>)> {
    let chunk_count = sparse_xtwx_chunk_count(n, p);
    let chunk_size = n.div_ceil(chunk_count).max(1);
    let num_chunks = n.div_ceil(chunk_size);

    let (xtx_data, xtz_data): (Vec<f64>, Vec<f64>) = (0..num_chunks)
        .into_par_iter()
        .map(|chunk_idx| {
            let chunk_start = chunk_idx * chunk_size;
            let chunk_end = (chunk_start + chunk_size).min(n);
            let mut xtx_local = vec![0.0; p * p];
            let mut xtz_local = vec![0.0; p];
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

                // SAFETY: k < n, so k is within both weight/response slices.
                let wk = unsafe { *w_slice.get_unchecked(k) };
                let zk = unsafe { *z_slice.get_unchecked(k) };
                let wz = wk * zk;

                for a in 0..nz_idx.len() {
                    // SAFETY: a is within both nz vectors populated in lockstep above.
                    let i = unsafe { *nz_idx.get_unchecked(a) };
                    let xki = unsafe { *nz_val.get_unchecked(a) };
                    let xki_w = xki * wk;
                    // SAFETY: i is a column index < p, so i < xtz_local.len().
                    unsafe { *xtz_local.get_unchecked_mut(i) += xki * wz };

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
            (xtx_local, xtz_local)
        })
        .reduce_with(|(mut a_xtx, mut a_xtz), (b_xtx, b_xtz)| {
            for i in 0..a_xtx.len() {
                a_xtx[i] += b_xtx[i];
            }
            for i in 0..a_xtz.len() {
                a_xtz[i] += b_xtz[i];
            }
            (a_xtx, a_xtz)
        })
        .unwrap_or_else(|| (vec![0.0; p * p], vec![0.0; p]));

    let mut xtx = DMatrix::zeros(p, p);
    for i in 0..p {
        for j in i..p {
            let val = xtx_data[i * p + j];
            xtx[(i, j)] = val;
            xtx[(j, i)] = val;
        }
    }
    let xtz = DVector::from_vec(xtz_data);

    Ok((xtx, xtz))
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
) -> Result<(Array1<f64>, Array2<f64>)> {
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
    let (mut xtx, xtz) = match sparse_cache {
        Some(cache) => compute_xtwx_xtwz_sparse_cached(cache, z, w)?,
        None => compute_xtwx_xtwz(x, z, w)?,
    };

    // Add L2 (Ridge) penalty to diagonal: (X'WX + λI)
    // The intercept (first column) is typically NOT penalized.
    if l2_penalty > 0.0 {
        let start_idx = if penalize_intercept { 0 } else { 1 };
        for j in start_idx..p {
            let factor = l2_penalty_factors.map_or(1.0, |factors| factors[j]);
            xtx[(j, j)] += l2_penalty * factor;
        }
    }

    if skip_covariance {
        let coefficients = cholesky_solve_coefficients(xtx, &xtz)?;
        Ok((coefficients, Array2::zeros((0, 0))))
    } else {
        cholesky_solve(xtx, &xtz, false)
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
    if penalty_matrix.nrows() != p || penalty_matrix.ncols() != p {
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

    let xtx_data = match (x.as_slice(), w.as_slice()) {
        (Some(x_slice), Some(w_slice)) => {
            assert_eq!(x_slice.len(), n * p, "x_slice length must be n*p");
            assert_eq!(w_slice.len(), n, "w_slice length must be n");
            if should_use_sparse_xtwx_kernel(x_slice, n, p) {
                compute_xtwx_sparse_data(x_slice, w_slice, n, p)
            } else {
                compute_xtwx_dense_data(x_slice, w_slice, n, p)
            }
        }
        _ => compute_xtwx_strided_data(x, w, n, p),
    };

    xtx_data_to_array2(xtx_data, p)
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
    use ndarray::{array, Array2};

    #[test]
    fn test_compute_xtwx_sparse_kernel_matches_dense_formula() {
        let n = 64;
        let p = 24;
        let mut x = Array2::<f64>::zeros((n, p));
        for row in 0..n {
            let j1 = row % p;
            let j2 = (row * 5 + 3) % p;
            x[[row, j1]] = 1.0 + (row % 4) as f64;
            x[[row, j2]] += (row as f64 / 4.0).sin();
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
                assert!((actual[[i, j]] - expected[[i, j]]).abs() < 1e-12);
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
    fn test_compute_xtwx_xtwz_sparse_kernel_matches_formula() {
        let n = 72;
        let p = 32;
        let mut x = Array2::<f64>::zeros((n, p));
        for row in 0..n {
            let j1 = row % p;
            let j2 = (row * 7 + 5) % p;
            x[[row, j1]] = 1.0 + (row % 3) as f64;
            x[[row, j2]] += (row as f64 * 0.13).cos();
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
            let j1 = row % p;
            let j2 = (row * 7 + 5) % p;
            x[[row, j1]] = 1.0 + (row % 3) as f64;
            x[[row, j2]] += (row as f64 * 0.13).cos();
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

        let (actual_xtx, actual_xtz) =
            compute_xtwx_xtwz_sparse_cached(&cache, &z, &w).expect("cached kernel should succeed");
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
        let (initial_coef, _) = solve_weighted_least_squares_penalized(
            x.view(),
            &initial_eta,
            &weights,
            0.0,
            None,
            true,
            false,
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
