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

use nalgebra::{DMatrix, DVector};
use ndarray::{Array1, Array2, ArrayView2};
use rayon::prelude::*;

use crate::constants::{
    CONVERGENCE_TOL, DEFAULT_MAX_ITER, IRLS_ACCEPT_REL_SLACK, IRLS_MAX_HALF_STEPS, MIN_IRLS_WEIGHT,
    ZERO_TOL,
};
use crate::error::{Result, RustyStatsError};
use crate::families::Family;
use crate::links::Link;
use crate::regularization::{Penalty, RegularizationConfig};

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
            penalize_intercept,
            config.regularization.penalty.clone(),
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
    penalize_intercept: bool,
    penalty: Penalty,
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
        let eta_init = x.dot(init) + &offset_vec;
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
            penalize_intercept,
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
        eta = &x.dot(&iter_coefficients) + &offset_vec;
        mu = family.clamp_mu(&link.inverse(&eta));
    }

    // -------------------------------------------------------------------------
    // Step 3: Calculate initial deviance
    // -------------------------------------------------------------------------
    let mut deviance = family.deviance(y, &mu, Some(&prior_weights_vec));
    let mut deviance_old: f64;

    // -------------------------------------------------------------------------
    // Step 4: IRLS iteration loop
    // -------------------------------------------------------------------------
    let mut converged = false;
    let mut iteration = 0;
    let mut step_halving_used = false;
    // Set when no full or halved step reduces the deviance: the previous iterate
    // is retained rather than accepting a worse step (RS-ACT-007).
    let mut step_halving_failed = false;

    // We'll store the final covariance matrix and coefficients from iteration
    let mut cov_unscaled = Array2::zeros((p, p));
    let mut final_weights = Array1::zeros(n);

    // For constrained problems, track best solution seen (deviance can increase due to projection)
    let has_constraints = !config.nonneg_indices.is_empty() || !config.nonpos_indices.is_empty();
    let mut best_deviance = f64::INFINITY; // Will be set after first iteration
    let mut best_coefficients = iter_coefficients.clone();
    let mut best_mu = mu.clone();
    let mut best_eta = eta.clone();
    let mut best_cov = cov_unscaled.clone();
    let mut best_weights = final_weights.clone();

    while iteration < config.max_iterations {
        iteration += 1;

        // ---------------------------------------------------------------------
        // Step 4a: Compute working weights W
        // ---------------------------------------------------------------------
        let weight_result = compute_irls_weights(
            y,
            &mu,
            &eta,
            &offset_vec,
            &prior_weights_vec,
            family,
            link,
            config.min_weight,
        )?;
        let irls_weights = weight_result.irls_weights;
        let combined_weights = weight_result.combined_weights;
        let working_response = weight_result.working_response;

        // ---------------------------------------------------------------------
        // Step 4c: Solve weighted least squares: (X'WX)β = X'Wz
        // ---------------------------------------------------------------------
        // This is the core linear algebra step.
        // We're finding β that minimizes: Σ w_i (z_i - x_i'β)²
        // ---------------------------------------------------------------------
        let (mut new_coefficients, xtwinv) = solve_weighted_least_squares_penalized(
            x,
            &working_response,
            &combined_weights,
            l2_penalty,
            penalize_intercept,
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
        // If deviance increases, reduce step size to prevent oscillation.
        // For constrained problems, we blend coefficients (not eta) and re-apply
        // projection to ensure constraints are satisfied at each step.
        // ---------------------------------------------------------------------
        deviance_old = deviance;

        // Acceptance threshold: a step may not worsen the deviance by more than a
        // tiny relative tolerance (RS-ACT-007). Coefficient blending is used for
        // both paths; for the unconstrained path it is algebraically identical to
        // blending eta, but keeps coefficients and (eta, mu) consistent.
        let accept_threshold = deviance_old * IRLS_ACCEPT_REL_SLACK;

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
        let mut eta_new = &x.dot(&trial_coefficients) + &offset_vec;
        let mut mu_new = family.clamp_mu(&link.inverse(&eta_new));
        let mut deviance_new = family.deviance(y, &mu_new, Some(&prior_weights_vec));

        let mut step_accepted = eta_new.iter().all(|v| v.is_finite())
            && mu_new.iter().all(|v| v.is_finite())
            && deviance_new.is_finite()
            && deviance_new <= accept_threshold;

        // Step-halving: if the full step worsened the deviance, try smaller steps
        // and accept the first one that meets the threshold. The
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
                let e = &x.dot(&blended) + &offset_vec;
                let m = family.clamp_mu(&link.inverse(&e));
                let d = family.deviance(y, &m, Some(&prior_weights_vec));
                if e.iter().all(|v| v.is_finite())
                    && m.iter().all(|v| v.is_finite())
                    && d.is_finite()
                    && d <= accept_threshold
                {
                    trial_coefficients = blended;
                    eta_new = e;
                    mu_new = m;
                    deviance_new = d;
                    step_accepted = true;
                    step_halving_used = true;
                    break;
                }
                step_size *= 0.5;
            }
        }

        if !step_accepted {
            // No full or halved step reduced the deviance: retain the previous
            // iterate (already held in iter_coefficients / eta / mu / deviance)
            // instead of accepting a worse one, and stop (RS-ACT-007).
            step_halving_failed = true;
            cov_unscaled = xtwinv.clone();
            final_weights = irls_weights.clone();
            break;
        }

        new_coefficients = trial_coefficients;
        eta = eta_new;
        mu = mu_new;
        deviance = deviance_new;

        // Relative change in deviance
        let rel_change = if deviance_old.abs() > ZERO_TOL {
            (deviance_old - deviance).abs() / deviance_old.abs()
        } else {
            (deviance_old - deviance).abs()
        };

        if config.verbose {
            eprintln!(
                "Iteration {}: deviance = {:.6}, rel_change = {:.2e}",
                iteration, deviance, rel_change
            );
        }

        // Store coefficients from this iteration
        iter_coefficients = new_coefficients;

        // For constrained problems, track the best solution seen
        if has_constraints && deviance < best_deviance {
            best_deviance = deviance;
            best_coefficients = iter_coefficients.clone();
            best_mu = mu.clone();
            best_eta = eta.clone();
            best_cov = xtwinv.clone();
            best_weights = irls_weights.clone();
        }

        // Convergence requires the accepted step to be non-worsening (signed
        // change ≤ 0) AND small. A worse-but-close step is not converged.
        // The accept loop above already enforces the slack on the signed
        // change; the final convergence flag must also.
        let signed_change = deviance_old - deviance;
        if signed_change >= -ZERO_TOL && rel_change < config.tolerance {
            converged = true;
            cov_unscaled = xtwinv;
            final_weights = irls_weights;
            break;
        }

        // Store for final iteration
        cov_unscaled = xtwinv;
        final_weights = irls_weights;
    }

    // -------------------------------------------------------------------------
    // Step 5: Extract final coefficients
    // -------------------------------------------------------------------------
    // For constrained problems, use the best solution found during iteration
    // (deviance can increase due to projection, so last iteration may not be best)
    let (final_mu, final_eta, final_deviance, use_coefficients) =
        if has_constraints && best_deviance < deviance {
            // Best solution was found earlier — use it and treat as converged
            // (sign clamping can cause coefficient oscillation even when deviance
            // has stabilized, so the best-tracked solution is the correct answer)
            converged = true;
            cov_unscaled = best_cov;
            final_weights = best_weights;
            (best_mu, best_eta, best_deviance, best_coefficients)
        } else {
            (mu, eta, deviance, iter_coefficients)
        };

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

    // Try final coefficient extraction, but fall back to iteration coefficients if it produces NaN
    let final_coefficients = match solve_weighted_least_squares_penalized(
        x,
        &compute_working_response(y, &final_mu, &eta_no_offset, link),
        &combined_final_weights,
        l2_penalty,
        penalize_intercept,
    ) {
        Ok((coef, _)) if !coef.iter().any(|&c| c.is_nan() || c.is_infinite()) => {
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
                let eta_check = x.dot(&proj_coef);
                let eta_full: Array1<f64> = eta_check
                    .iter()
                    .zip(offset_vec.iter())
                    .map(|(&e, &o)| e + o)
                    .collect();
                let mu_check = family.clamp_mu(&link.inverse(&eta_full));
                let dev_check = family.deviance(y, &mu_check, Some(&prior_weights_vec));
                if dev_check <= final_deviance {
                    proj_coef
                } else {
                    use_coefficients
                }
            } else {
                // Unconstrained: guard against a final extraction that is worse
                // than the loop's retained iterate (RS-ACT-007).
                let eta_full = &x.dot(&coef) + &offset_vec;
                let mu_check = family.clamp_mu(&link.inverse(&eta_full));
                let dev_check = family.deviance(y, &mu_check, Some(&prior_weights_vec));
                if dev_check.is_finite() && dev_check <= final_deviance {
                    coef
                } else {
                    use_coefficients
                }
            }
        }
        _ => {
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
            use_coefficients
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
    let final_eta_base = x.dot(&final_coefficients);
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
/// - i, j range from 0 to p-1, and xtx_local has length p*p, xtz_local has length p
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
    // All accesses use indices < n*p, < n, and < p*p respectively.
    assert_eq!(x_slice.len(), n * p, "x_slice length must be n*p");
    assert_eq!(w_slice.len(), n, "w_slice length must be n");
    assert_eq!(z_slice.len(), n, "z_slice length must be n");

    const CHUNK_SIZE: usize = 8192;
    let num_chunks = n.div_ceil(CHUNK_SIZE);

    let (xtx_data, xtz_data): (Vec<f64>, Vec<f64>) = (0..num_chunks)
        .into_par_iter()
        .map(|chunk_idx| {
            let chunk_start = chunk_idx * CHUNK_SIZE;
            let chunk_end = (chunk_start + CHUNK_SIZE).min(n);
            let mut xtx_local = vec![0.0; p * p];
            let mut xtz_local = vec![0.0; p];

            for k in chunk_start..chunk_end {
                // SAFETY: k < n, so k < w_slice.len() and k < z_slice.len()
                let wk = unsafe { *w_slice.get_unchecked(k) };
                let zk = unsafe { *z_slice.get_unchecked(k) };
                let wz = wk * zk;
                let row_start = k * p;

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
                        // SAFETY: i*p + j < p*p = xtx_local.len() (since i <= j < p)
                        unsafe { *xtx_local.get_unchecked_mut(i * p + j) += xki_w * xkj };
                    }
                }
            }
            (xtx_local, xtz_local)
        })
        .reduce(
            || (vec![0.0; p * p], vec![0.0; p]),
            |(mut a_xtx, mut a_xtz), (b_xtx, b_xtz)| {
                for i in 0..a_xtx.len() {
                    a_xtx[i] += b_xtx[i];
                }
                for i in 0..a_xtz.len() {
                    a_xtz[i] += b_xtz[i];
                }
                (a_xtx, a_xtz)
            },
        );

    // Convert to nalgebra symmetric DMatrix
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
    penalize_intercept: bool,
) -> Result<(Array1<f64>, Array2<f64>)> {
    let p = x.ncols();
    let (mut xtx, xtz) = compute_xtwx_xtwz(x, z, w)?;

    // Add L2 (Ridge) penalty to diagonal: (X'WX + λI)
    // The intercept (first column) is typically NOT penalized.
    if l2_penalty > 0.0 {
        let start_idx = if penalize_intercept { 0 } else { 1 };
        for j in start_idx..p {
            xtx[(j, j)] += l2_penalty;
        }
    }

    cholesky_solve(xtx, &xtz, false)
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

    let xtx_data: Vec<f64> = (0..n)
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
        .reduce(
            || vec![0.0; p * p],
            |mut a, b| {
                for i in 0..a.len() {
                    a[i] += b[i];
                }
                a
            },
        );

    // Convert to Array2, symmetrizing
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
    use ndarray::array;

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
        let (initial_coef, _) =
            solve_weighted_least_squares_penalized(x.view(), &initial_eta, &weights, 0.0, true)
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
