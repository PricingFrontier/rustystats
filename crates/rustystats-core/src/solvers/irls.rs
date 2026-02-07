use super::initialize_mu_safe;

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

use ndarray::{Array1, Array2};
use rayon::prelude::*;
use nalgebra::{DMatrix, DVector};

use crate::constants::{
    CONVERGENCE_TOL, MIN_IRLS_WEIGHT, DEFAULT_MAX_ITER, ZERO_TOL,
};
use crate::error::{RustyStatsError, Result};
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
}

// =============================================================================
// Main Fitting Function
// =============================================================================

/// Fit a GLM using Iteratively Reweighted Least Squares (simple version).
///
/// This is a convenience wrapper that calls `fit_glm_full` with no offset or weights.
///
/// # Arguments
/// * `y` - Response variable (n × 1)
/// * `x` - Design matrix (n × p), should include intercept column if desired
/// * `family` - Distribution family (Gaussian, Poisson, Binomial, Gamma)
/// * `link` - Link function (Identity, Log, Logit)
/// * `config` - Algorithm configuration options
///
/// # Returns
/// * `Ok(IRLSResult)` - Fitted model results
/// * `Err(RustyStatsError)` - If fitting fails
pub fn fit_glm(
    y: &Array1<f64>,
    x: &Array2<f64>,
    family: &dyn Family,
    link: &dyn Link,
    config: &IRLSConfig,
) -> Result<IRLSResult> {
    fit_glm_full(y, x, family, link, config, None, None)
}

/// Fit a GLM using Iteratively Reweighted Least Squares (full version).
///
/// This is the main entry point for GLM fitting with all options.
///
/// # Arguments
/// * `y` - Response variable (n × 1)
/// * `x` - Design matrix (n × p), should include intercept column if desired
/// * `family` - Distribution family (Gaussian, Poisson, Binomial, Gamma)
/// * `link` - Link function (Identity, Log, Logit)
/// * `config` - Algorithm configuration options
/// * `offset` - Optional offset term (e.g., log(exposure) for rate models)
/// * `weights` - Optional prior weights for each observation
///
/// # Returns
/// * `Ok(IRLSResult)` - Fitted model results
/// * `Err(RustyStatsError)` - If fitting fails
///
/// # Offset (for Actuaries)
/// The offset is added to the linear predictor: η = Xβ + offset
///
/// For claim frequency with varying exposure:
///   log(E[claims]) = Xβ + log(exposure)
///   
/// So you pass `log(exposure)` as the offset.
///
/// # Weights
/// Prior weights adjust the contribution of each observation.
/// Use for:
/// - Grouped data (weight = count of observations in group)
/// - Importance weighting
/// - Handling known variance differences
///
/// # Example
/// ```ignore
/// let offset = exposure.mapv(|e| e.ln());  // log(exposure)
/// let weights = Some(Array1::ones(n));     // Equal weights
/// let result = fit_glm_full(&y, &x, &PoissonFamily, &LogLink, &config, Some(&offset), weights.as_ref())?;
/// ```
pub fn fit_glm_full(
    y: &Array1<f64>,
    x: &Array2<f64>,
    family: &dyn Family,
    link: &dyn Link,
    config: &IRLSConfig,
    offset: Option<&Array1<f64>>,
    weights: Option<&Array1<f64>>,
) -> Result<IRLSResult> {
    fit_glm_core(y, x, family, link, config, offset, weights, None, 0.0, false, Penalty::None)
}

/// Core IRLS fitting function with optional warm start and optional L2 penalty.
///
/// This is the unified implementation used by `fit_glm_full`, `fit_glm_warm_start`,
/// and `fit_glm_regularized_warm`. When `init_coefficients` is provided, initialization
/// starts from those coefficients instead of the family's default. When `l2_penalty > 0`,
/// Ridge regularization is applied: (X'WX + λI)β = X'Wz.
fn fit_glm_core(
    y: &Array1<f64>,
    x: &Array2<f64>,
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
    // Step 0: Validate inputs
    // -------------------------------------------------------------------------
    let n = y.len();
    let p = x.ncols();

    if x.nrows() != n {
        return Err(RustyStatsError::DimensionMismatch(format!(
            "X has {} rows but y has {} elements",
            x.nrows(),
            n
        )));
    }

    if n == 0 {
        return Err(RustyStatsError::EmptyInput("y is empty".to_string()));
    }

    if p == 0 {
        return Err(RustyStatsError::EmptyInput("X has no columns".to_string()));
    }

    // -------------------------------------------------------------------------
    // Step 0b: Set up offset and prior weights
    // -------------------------------------------------------------------------
    // Offset: Added to linear predictor (e.g., log(exposure) for rate models)
    // Prior weights: User-specified observation weights
    let offset_vec = match offset {
        Some(o) => {
            if o.len() != n {
                return Err(RustyStatsError::DimensionMismatch(format!(
                    "offset has {} elements but y has {}",
                    o.len(),
                    n
                )));
            }
            o.clone()
        }
        None => Array1::zeros(n),
    };

    let prior_weights_vec = match weights {
        Some(w) => {
            if w.len() != n {
                return Err(RustyStatsError::DimensionMismatch(format!(
                    "weights has {} elements but y has {}",
                    w.len(),
                    n
                )));
            }
            // Ensure weights are positive
            if w.iter().any(|&x| x < 0.0) {
                return Err(RustyStatsError::InvalidValue(
                    "weights must be non-negative".to_string(),
                ));
            }
            w.clone()
        }
        None => Array1::ones(n),
    };

    // -------------------------------------------------------------------------
    // Step 1: Initialize μ (from coefficients if warm-starting, else from family)
    // -------------------------------------------------------------------------
    let mut mu = if let Some(init) = init_coefficients {
        if init.len() != p {
            return Err(RustyStatsError::DimensionMismatch(format!(
                "init_coefficients has {} elements but X has {} columns",
                init.len(), p
            )));
        }
        let eta_init = x.dot(init) + &offset_vec;
        let mu_init = link.inverse(&eta_init);
        family.clamp_mu(&mu_init)
    } else {
        let mu_init = family.initialize_mu(y);
        // Ensure μ is valid (e.g., positive for Poisson, in (0,1) for Binomial)
        if !family.is_valid_mu(&mu_init) {
            eprintln!(
                "Warning: Family '{}' initial μ values were invalid. Using safe fallback initialization. \
                This may indicate unusual response values (e.g., all zeros, extreme values).",
                family.name()
            );
            initialize_mu_safe(y, family)
        } else {
            mu_init
        }
    };

    // -------------------------------------------------------------------------
    // Step 2: Initialize linear predictor η = g(μ)
    // -------------------------------------------------------------------------
    // Note: We store η without offset for coefficient estimation
    // The full linear predictor is η + offset
    let mut eta = link.link(&mu);

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

    // We'll store the final covariance matrix and coefficients from iteration
    let mut cov_unscaled = Array2::zeros((p, p));
    let mut final_weights = Array1::zeros(n);
    let mut iter_coefficients = Array1::zeros(p);  // Store coefficients from iteration
    
    // For constrained problems, track best solution seen (deviance can increase due to projection)
    let has_constraints = !config.nonneg_indices.is_empty() || !config.nonpos_indices.is_empty();
    let mut best_deviance = f64::INFINITY;  // Will be set after first iteration
    let mut best_coefficients = iter_coefficients.clone();
    let mut best_mu = mu.clone();
    let mut best_eta = eta.clone();

    while iteration < config.max_iterations {
        iteration += 1;

        // ---------------------------------------------------------------------
        // Step 4a: Compute working weights W
        // ---------------------------------------------------------------------
        // The standard IRLS weight for observation i is:
        //     w_irls_i = 1 / (V(μ_i) × g'(μ_i)²)
        //
        // OPTIMIZATION: For certain family/link combinations (Gamma, Tweedie 1<p<2
        // with log link), using the true Hessian instead of Fisher information
        // can dramatically reduce iterations (50-100 → 5-10). This is because
        // the true Hessian provides better curvature information.
        //
        // Combined with prior weights:
        //     w_i = prior_weight_i × w_irls_i
        // ---------------------------------------------------------------------
        let link_deriv = link.derivative(&mu);
        
        // Check if family supports true Hessian weights (Gamma, Tweedie 1<p<2)
        let use_true_hessian = family.use_true_hessian_weights() && link.name() == "log";
        let hessian_weights = if use_true_hessian {
            Some(family.true_hessian_weights(&mu, y))
        } else {
            None
        };
        let variance = if use_true_hessian { None } else { Some(family.variance(&mu)) };

        // PARALLEL: Compute IRLS weights, combined weights, and working response
        // in a single parallel pass to minimize allocation overhead
        let n = y.len();
        let min_weight = config.min_weight;
        
        let results: Vec<(f64, f64, f64)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let d = link_deriv[i];
                
                // IRLS weight: use true Hessian if available, else Fisher info
                let iw = if let Some(ref hw) = hessian_weights {
                    // True Hessian weight - use directly without dividing by d²
                    // For Gamma+log link: w = μ (not μ/(1/μ)² = μ³ which was the bug)
                    hw[i].max(min_weight).min(1e10)
                } else {
                    // Standard Fisher information weight: w = 1/(V(μ) × (dη/dμ)²)
                    let v = variance.as_ref().unwrap()[i];
                    (1.0 / (v * d * d)).max(min_weight).min(1e10)
                };
                
                // Combined weight
                let cw = prior_weights_vec[i] * iw;
                
                // Working response: z = (η - offset) + (y - μ) × g'(μ)
                let e = eta[i] - offset_vec[i];
                let wr = e + (y[i] - mu[i]) * d;
                
                (iw, cw, wr)
            })
            .collect();
        
        let mut irls_weights_vec = Vec::with_capacity(n);
        let mut combined_weights_vec = Vec::with_capacity(n);
        let mut working_response_vec = Vec::with_capacity(n);
        for (iw, cw, wr) in results {
            irls_weights_vec.push(iw);
            combined_weights_vec.push(cw);
            working_response_vec.push(wr);
        }
        
        let irls_weights = Array1::from_vec(irls_weights_vec);
        let combined_weights = Array1::from_vec(combined_weights_vec);
        let working_response = Array1::from_vec(working_response_vec);

        // ---------------------------------------------------------------------
        // Step 4c: Solve weighted least squares: (X'WX)β = X'Wz
        // ---------------------------------------------------------------------
        // This is the core linear algebra step.
        // We're finding β that minimizes: Σ w_i (z_i - x_i'β)²
        // ---------------------------------------------------------------------
        let (mut new_coefficients, xtwinv) =
            solve_weighted_least_squares_penalized(x, &working_response, &combined_weights, l2_penalty, penalize_intercept)?;

        // Check for NaN in coefficients - indicates numerical instability
        if new_coefficients.iter().any(|&c| c.is_nan() || c.is_infinite()) {
            return Err(RustyStatsError::NumericalError(
                "IRLS produced NaN or infinite coefficients. This usually indicates: \
                 (1) severe multicollinearity in predictors, \
                 (2) extreme scale differences between variables, or \
                 (3) separation in binary response data. \
                 Try standardizing continuous predictors or removing correlated terms.".to_string()
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
        
        // Try full step first
        let eta_base = x.dot(&new_coefficients);
        let mut eta_new = &eta_base + &offset_vec;
        let mut mu_new = link.inverse(&eta_new);
        mu_new = family.clamp_mu(&mu_new);
        let mut deviance_new = family.deviance(y, &mu_new, Some(&prior_weights_vec));
        
        // Step-halving: if deviance increased, try smaller steps
        if deviance_new > deviance_old * 1.0001 && iteration > 1 {
            let mut step_size = 0.5;
            
            if has_constraints {
                // For constrained problems: blend coefficients and re-apply projection
                for _half_step in 0..8 {
                    // Blend coefficients: β_blend = (1-step)*β_old + step*β_new
                    let mut blended_coefficients: Array1<f64> = iter_coefficients.iter()
                        .zip(new_coefficients.iter())
                        .map(|(&old, &new)| (1.0 - step_size) * old + step_size * new)
                        .collect();
                    
                    // Re-apply coefficient constraints after blending
                    for &idx in &config.nonneg_indices {
                        if idx < blended_coefficients.len() && blended_coefficients[idx] < 0.0 {
                            blended_coefficients[idx] = 0.0;
                        }
                    }
                    for &idx in &config.nonpos_indices {
                        if idx < blended_coefficients.len() && blended_coefficients[idx] > 0.0 {
                            blended_coefficients[idx] = 0.0;
                        }
                    }
                    
                    let eta_blend = x.dot(&blended_coefficients);
                    eta_new = &eta_blend + &offset_vec;
                    mu_new = link.inverse(&eta_new);
                    mu_new = family.clamp_mu(&mu_new);
                    deviance_new = family.deviance(y, &mu_new, Some(&prior_weights_vec));
                    
                    if deviance_new <= deviance_old * 1.0001 {
                        // Accept this blended step - update new_coefficients
                        new_coefficients = blended_coefficients;
                        break;
                    }
                    step_size *= 0.5;
                }
            } else {
                // For unconstrained problems: blend eta directly (faster)
                let eta_old_base: Array1<f64> = eta.iter()
                    .zip(offset_vec.iter())
                    .map(|(&e, &o)| e - o)
                    .collect();
                
                for _half_step in 0..4 {
                    let eta_blend: Array1<f64> = eta_old_base.iter()
                        .zip(eta_base.iter())
                        .map(|(&old, &new)| (1.0 - step_size) * old + step_size * new)
                        .collect();
                    eta_new = &eta_blend + &offset_vec;
                    mu_new = link.inverse(&eta_new);
                    mu_new = family.clamp_mu(&mu_new);
                    deviance_new = family.deviance(y, &mu_new, Some(&prior_weights_vec));
                    
                    if deviance_new <= deviance_old * 1.0001 {
                        break;
                    }
                    step_size *= 0.5;
                }
            }
        }
        
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
        }

        if rel_change < config.tolerance {
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
    let (final_mu, final_eta, final_deviance, use_coefficients) = if has_constraints && best_deviance < deviance {
        // Best solution was found earlier - use it
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
    let final_coefficients = match solve_weighted_least_squares_penalized(x, &compute_working_response(y, &final_mu, &eta_no_offset, link), &combined_final_weights, l2_penalty, penalize_intercept) {
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
                let eta_full: Array1<f64> = eta_check.iter().zip(offset_vec.iter()).map(|(&e, &o)| e + o).collect();
                let mu_check = link.inverse(&eta_full);
                let dev_check = family.deviance(y, &mu_check, Some(&prior_weights_vec));
                if dev_check <= final_deviance {
                    proj_coef
                } else {
                    use_coefficients
                }
            } else {
                coef
            }
        },
        _ => {
            // Final extraction failed or produced NaN - use stored coefficients
            eprintln!(
                "Warning: Final coefficient extraction produced NaN/Inf. \
                Using coefficients from best iteration instead. This may indicate numerical instability."
            );
            if use_coefficients.iter().any(|&c| c.is_nan() || c.is_infinite()) {
                return Err(RustyStatsError::NumericalError(
                    "IRLS produced NaN or infinite coefficients. This usually indicates: \
                     (1) severe multicollinearity in predictors, \
                     (2) extreme scale differences between variables, or \
                     (3) separation in binary response data. \
                     Try standardizing continuous predictors or removing correlated terms.".to_string()
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
    let final_eta: Array1<f64> = final_eta_base.iter().zip(offset_vec.iter()).map(|(&e, &o)| e + o).collect();
    let final_mu = link.inverse(&final_eta);
    let final_deviance = family.deviance(y, &final_mu, Some(&prior_weights_vec));

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
        y: y.to_owned(),  // Only clone at the end, needed for diagnostics
        family_name: family.name().to_string(),
        penalty,
        design_matrix: None,  // Computed lazily in Python layer to avoid expensive copy
    })
}

/// Fit a GLM with warm start (initial coefficients) for faster convergence.
///
/// This version accepts initial coefficients from a previous fit, which is useful for:
/// - Iterative theta estimation in Negative Binomial
/// - Sequential model fitting with similar data
/// - Continuation from a partially converged model
///
/// Delegates to `fit_glm_core` with the full IRLS implementation including
/// true Hessian optimization and step-halving.
///
/// # Arguments
/// * `y` - Response variable (n × 1)
/// * `x` - Design matrix (n × p)
/// * `family` - Distribution family
/// * `link` - Link function
/// * `config` - Algorithm configuration
/// * `offset` - Optional offset term
/// * `weights` - Optional prior weights
/// * `init_coefficients` - Initial coefficient estimates (p × 1)
///
/// # Returns
/// * `Ok(IRLSResult)` - Fitted model results
/// * `Err(RustyStatsError)` - If fitting fails
pub fn fit_glm_warm_start(
    y: &Array1<f64>,
    x: &Array2<f64>,
    family: &dyn Family,
    link: &dyn Link,
    config: &IRLSConfig,
    offset: Option<&Array1<f64>>,
    weights: Option<&Array1<f64>>,
    init_coefficients: &Array1<f64>,
) -> Result<IRLSResult> {
    fit_glm_core(y, x, family, link, config, offset, weights, Some(init_coefficients), 0.0, false, Penalty::None)
}

/// Fit a regularized GLM using Iteratively Reweighted Least Squares.
///
/// This version supports Ridge (L2) regularization. For Lasso (L1) or Elastic Net,
/// use `fit_glm_coordinate_descent` (which handles the non-differentiable L1 term).
///
/// # Arguments
/// * `y` - Response variable (n × 1)
/// * `x` - Design matrix (n × p), should include intercept column if desired
/// * `family` - Distribution family (Gaussian, Poisson, Binomial, Gamma)
/// * `link` - Link function (Identity, Log, Logit)
/// * `irls_config` - IRLS algorithm configuration
/// * `reg_config` - Regularization configuration
/// * `offset` - Optional offset term
/// * `weights` - Optional prior weights
///
/// # Returns
/// * `Ok(IRLSResult)` - Fitted model results
/// * `Err(RustyStatsError)` - If fitting fails or L1 penalty requested
///
/// # Ridge Regularization
/// For Ridge (L2) regularization, we modify the normal equations:
///   (X'WX)β = X'Wz  →  (X'WX + λI)β = X'Wz
///
/// This shrinks coefficients toward zero, improving stability for
/// multicollinear data. The intercept is NOT penalized by default.
///
/// # Example
/// ```ignore
/// use rustystats_core::regularization::RegularizationConfig;
/// 
/// let reg_config = RegularizationConfig::ridge(0.1);
/// let result = fit_glm_regularized(&y, &x, &family, &link, &config, &reg_config, None, None)?;
/// ```
pub fn fit_glm_regularized(
    y: &Array1<f64>,
    x: &Array2<f64>,
    family: &dyn Family,
    link: &dyn Link,
    irls_config: &IRLSConfig,
    reg_config: &RegularizationConfig,
    offset: Option<&Array1<f64>>,
    weights: Option<&Array1<f64>>,
) -> Result<IRLSResult> {
    // Delegate to warm-start version with no initial coefficients
    fit_glm_regularized_warm(y, x, family, link, irls_config, reg_config, offset, weights, None)
}

/// Fit a regularized GLM with optional warm start from initial coefficients.
/// 
/// Delegates to `fit_glm_core` with Ridge penalty parameters.
/// This gives the regularized path the same benefits as the core solver
/// (true Hessian optimization, step-halving, constraint-aware best-solution tracking).
pub fn fit_glm_regularized_warm(
    y: &Array1<f64>,
    x: &Array2<f64>,
    family: &dyn Family,
    link: &dyn Link,
    irls_config: &IRLSConfig,
    reg_config: &RegularizationConfig,
    offset: Option<&Array1<f64>>,
    weights: Option<&Array1<f64>>,
    init_coefficients: Option<&Array1<f64>>,
) -> Result<IRLSResult> {
    // Check if L1 penalty is requested - this requires coordinate descent
    if reg_config.penalty.requires_coordinate_descent() {
        return Err(RustyStatsError::InvalidValue(
            "L1 (Lasso) and Elastic Net penalties require coordinate descent solver. \
             Use fit_glm_coordinate_descent instead, or use pure Ridge (L2) penalty."
                .to_string(),
        ));
    }

    let l2_penalty = reg_config.penalty.l2_penalty();
    let penalize_intercept = !reg_config.fit_intercept;

    fit_glm_core(
        y, x, family, link, irls_config, offset, weights,
        init_coefficients, l2_penalty, penalize_intercept, reg_config.penalty.clone(),
    )
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
/// Safety: All unsafe accesses are within bounds because:
/// - k ranges from 0 to n-1, and w_slice/z_slice have length n
/// - row_start + j = k*p + j where k < n and j < p, so max index is (n-1)*p + (p-1) < n*p = x_slice.len()
/// - i, j range from 0 to p-1, and xtx_local has length p*p, xtz_local has length p
#[inline]
fn compute_xtwx_xtwz(
    x: &Array2<f64>,
    z: &Array1<f64>,
    w: &Array1<f64>,
) -> Result<(DMatrix<f64>, DVector<f64>)> {
    let n = x.nrows();
    let p = x.ncols();

    let x_slice = match x.as_slice() {
        Some(s) => s,
        None => {
            return Err(RustyStatsError::LinearAlgebraError(
                "Design matrix X must be contiguous in memory (C-order)".to_string()
            ));
        }
    };
    let w_slice = match w.as_slice() {
        Some(s) => s,
        None => {
            return Err(RustyStatsError::LinearAlgebraError(
                "Weight vector W must be contiguous in memory".to_string()
            ));
        }
    };
    let z_slice = match z.as_slice() {
        Some(s) => s,
        None => {
            return Err(RustyStatsError::LinearAlgebraError(
                "Working response Z must be contiguous in memory".to_string()
            ));
        }
    };
    
    const CHUNK_SIZE: usize = 8192;
    let num_chunks = (n + CHUNK_SIZE - 1) / CHUNK_SIZE;
    
    let (xtx_data, xtz_data): (Vec<f64>, Vec<f64>) = (0..num_chunks)
        .into_par_iter()
        .map(|chunk_idx| {
            let chunk_start = chunk_idx * CHUNK_SIZE;
            let chunk_end = (chunk_start + CHUNK_SIZE).min(n);
            let mut xtx_local = vec![0.0; p * p];
            let mut xtz_local = vec![0.0; p];
            
            for k in chunk_start..chunk_end {
                let wk = unsafe { *w_slice.get_unchecked(k) };
                let zk = unsafe { *z_slice.get_unchecked(k) };
                let wz = wk * zk;
                let row_start = k * p;
                
                for i in 0..p {
                    let xki = unsafe { *x_slice.get_unchecked(row_start + i) };
                    let xki_w = xki * wk;
                    unsafe { *xtz_local.get_unchecked_mut(i) += xki * wz };
                    
                    for j in i..p {
                        let xkj = unsafe { *x_slice.get_unchecked(row_start + j) };
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
#[inline]
fn cholesky_solve(
    a: DMatrix<f64>,
    b: &DVector<f64>,
) -> Result<(Array1<f64>, Array2<f64>)> {
    let p = a.nrows();

    let chol = match a.clone().cholesky() {
        Some(c) => c,
        None => {
            // Fall back to LU decomposition
            match a.clone().lu().solve(b) {
                Some(sol) => {
                    let coef_array: Array1<f64> = sol.iter().copied().collect();
                    let a_inv = a.try_inverse().ok_or_else(|| {
                        RustyStatsError::LinearAlgebraError(
                            "Failed to compute covariance matrix - system is not invertible. \
                             This often indicates multicollinearity in predictors.".to_string()
                        )
                    })?;
                    let mut cov_array = Array2::zeros((p, p));
                    for i in 0..p {
                        for j in 0..p {
                            cov_array[[i, j]] = a_inv[(i, j)];
                        }
                    }
                    return Ok((coef_array, cov_array));
                }
                None => {
                    return Err(RustyStatsError::LinearAlgebraError(
                        "Failed to solve linear system - matrix may be singular. \
                         This often indicates multicollinearity in predictors.".to_string(),
                    ));
                }
            }
        }
    };

    let coefficients = chol.solve(b);
    let identity = DMatrix::identity(p, p);
    let a_inv = chol.solve(&identity);

    let coef_array: Array1<f64> = coefficients.iter().copied().collect();
    let mut cov_array = Array2::zeros((p, p));
    for i in 0..p {
        for j in 0..p {
            cov_array[[i, j]] = a_inv[(i, j)];
        }
    }

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
    x: &Array2<f64>,
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

    cholesky_solve(xtx, &xtz)
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
///
/// # Returns
/// * Coefficients β (p × 1)
/// * Inverse of penalized normal equations (X'WX + S)⁻¹ (p × p)
pub fn solve_weighted_least_squares_with_penalty_matrix(
    x: &Array2<f64>,
    z: &Array1<f64>,
    w: &Array1<f64>,
    penalty_matrix: &Array2<f64>,
) -> Result<(Array1<f64>, Array2<f64>)> {
    let p = x.ncols();

    // Validate penalty matrix dimensions
    if penalty_matrix.nrows() != p || penalty_matrix.ncols() != p {
        return Err(RustyStatsError::DimensionMismatch(format!(
            "Penalty matrix has shape ({}, {}) but expected ({}, {})",
            penalty_matrix.nrows(), penalty_matrix.ncols(), p, p
        )));
    }

    let (mut xtx, xtz) = compute_xtwx_xtwz(x, z, w)?;

    // Add full penalty matrix S to X'WX
    for i in 0..p {
        for j in i..p {
            xtx[(i, j)] += penalty_matrix[[i, j]];
            xtx[(j, i)] += penalty_matrix[[j, i]];
        }
    }

    cholesky_solve(xtx, &xtz)
}

/// Compute X'WX matrix for EDF calculation.
///
/// This is needed for computing effective degrees of freedom in penalized regression.
pub fn compute_xtwx(x: &Array2<f64>, w: &Array1<f64>) -> Array2<f64> {
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
    use crate::families::{GaussianFamily, PoissonFamily};
    use crate::links::{IdentityLink, LogLink};
    use ndarray::array;

    #[test]
    fn test_gaussian_identity_is_ols() {
        // For Gaussian with identity link, IRLS should give same result as OLS
        // y = 2 + 3*x + noise
        let x = Array2::from_shape_vec(
            (5, 2),
            vec![
                1.0, 1.0, // intercept, x
                1.0, 2.0,
                1.0, 3.0,
                1.0, 4.0,
                1.0, 5.0,
            ],
        )
        .unwrap();
        let y = array![5.1, 7.9, 11.2, 13.8, 17.1]; // approximately 2 + 3*x

        let family = GaussianFamily;
        let link = IdentityLink;
        let config = IRLSConfig::default();

        let result = fit_glm(&y, &x, &family, &link, &config).unwrap();

        assert!(result.converged);
        // Intercept should be close to 2, slope close to 3
        assert!((result.coefficients[0] - 2.0).abs() < 0.5);
        assert!((result.coefficients[1] - 3.0).abs() < 0.2);
    }

    #[test]
    fn test_poisson_log_link() {
        // Simple Poisson regression
        // True model: log(μ) = 0.5 + 0.3*x
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 0.0,
                1.0, 1.0,
                1.0, 2.0,
                1.0, 3.0,
                1.0, 4.0,
                1.0, 5.0,
            ],
        )
        .unwrap();
        // y values that roughly follow exp(0.5 + 0.3*x)
        let y = array![2.0, 2.0, 3.0, 4.0, 5.0, 7.0];

        let family = PoissonFamily;
        let link = LogLink;
        let config = IRLSConfig::default();

        let result = fit_glm(&y, &x, &family, &link, &config).unwrap();

        assert!(result.converged);
        // Fitted values should be positive
        assert!(result.fitted_values.iter().all(|&x| x > 0.0));
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0]).unwrap();
        let y = array![1.0, 2.0]; // Wrong length!

        let family = GaussianFamily;
        let link = IdentityLink;

        let result = fit_glm(&y, &x, &family, &link, &IRLSConfig::default());

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            RustyStatsError::DimensionMismatch(_)
        ));
    }

    #[test]
    fn test_convergence_with_verbose() {
        let x = Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0])
            .unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0];

        let mut config = IRLSConfig::default();
        config.verbose = false; // Set to true to see iteration output
        config.max_iterations = 50;

        let result = fit_glm(&y, &x, &GaussianFamily, &IdentityLink, &config).unwrap();

        assert!(result.converged);
        // Perfect linear relationship should converge quickly
        assert!(result.iterations < 10);
    }

    // =========================================================================
    // Ridge (L2) Regularization Tests
    // =========================================================================

    #[test]
    fn test_ridge_shrinks_coefficients() {
        // Ridge regression should shrink coefficients toward zero
        let x = Array2::from_shape_vec(
            (5, 2),
            vec![
                1.0, 1.0,
                1.0, 2.0,
                1.0, 3.0,
                1.0, 4.0,
                1.0, 5.0,
            ],
        )
        .unwrap();
        let y = array![5.0, 8.0, 11.0, 14.0, 17.0]; // y = 2 + 3*x

        let family = GaussianFamily;
        let link = IdentityLink;
        let irls_config = IRLSConfig::default();

        // Unregularized fit
        let unreg = fit_glm(&y, &x, &family, &link, &irls_config).unwrap();

        // Ridge with lambda = 10 (strong regularization)
        let reg_config = RegularizationConfig::ridge(10.0);
        let ridge = fit_glm_regularized(&y, &x, &family, &link, &irls_config, &reg_config, None, None).unwrap();

        // Ridge coefficients should be smaller in absolute value (except intercept)
        assert!(ridge.coefficients[1].abs() < unreg.coefficients[1].abs(),
            "Ridge should shrink slope: ridge={:.4}, unreg={:.4}", 
            ridge.coefficients[1], unreg.coefficients[1]);
        
        // Both should converge
        assert!(unreg.converged);
        assert!(ridge.converged);
        
        // Penalty should be recorded
        assert!(!ridge.penalty.is_none());
        assert_eq!(ridge.penalty.l2_penalty(), 10.0);
    }

    #[test]
    fn test_ridge_no_penalty_equals_ols() {
        // Ridge with lambda=0 should give same results as OLS
        let x = Array2::from_shape_vec(
            (5, 2),
            vec![
                1.0, 1.0,
                1.0, 2.0,
                1.0, 3.0,
                1.0, 4.0,
                1.0, 5.0,
            ],
        )
        .unwrap();
        let y = array![5.0, 8.0, 11.0, 14.0, 17.0];

        let family = GaussianFamily;
        let link = IdentityLink;
        let irls_config = IRLSConfig::default();

        let unreg = fit_glm(&y, &x, &family, &link, &irls_config).unwrap();
        
        let reg_config = RegularizationConfig::ridge(0.0);
        let ridge_zero = fit_glm_regularized(&y, &x, &family, &link, &irls_config, &reg_config, None, None).unwrap();

        // Coefficients should be essentially equal
        for i in 0..2 {
            assert!((unreg.coefficients[i] - ridge_zero.coefficients[i]).abs() < 1e-6,
                "Coefficient {} differs: unreg={:.6}, ridge={:.6}",
                i, unreg.coefficients[i], ridge_zero.coefficients[i]);
        }
    }

    #[test]
    fn test_ridge_intercept_not_penalized() {
        // The intercept should not be penalized by default
        let x = Array2::from_shape_vec(
            (5, 2),
            vec![
                1.0, 1.0,
                1.0, 2.0,
                1.0, 3.0,
                1.0, 4.0,
                1.0, 5.0,
            ],
        )
        .unwrap();
        let y = array![5.0, 8.0, 11.0, 14.0, 17.0];

        let family = GaussianFamily;
        let link = IdentityLink;
        let irls_config = IRLSConfig::default();

        // Compare very strong penalty with no penalty
        let unreg = fit_glm(&y, &x, &family, &link, &irls_config).unwrap();
        
        let reg_config = RegularizationConfig::ridge(100.0);
        let ridge = fit_glm_regularized(&y, &x, &family, &link, &irls_config, &reg_config, None, None).unwrap();

        // Slope should be heavily shrunk
        assert!(ridge.coefficients[1].abs() < unreg.coefficients[1].abs() * 0.5,
            "Slope should be heavily shrunk");
        
        // Intercept should still be reasonable (not shrunk to 0)
        // With y mean around 11, intercept shouldn't be close to 0
        assert!(ridge.coefficients[0].abs() > 1.0,
            "Intercept should not be heavily shrunk: {:.4}", ridge.coefficients[0]);
    }

    #[test]
    fn test_ridge_poisson() {
        // Ridge should work with non-Gaussian families
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 0.0,
                1.0, 1.0,
                1.0, 2.0,
                1.0, 3.0,
                1.0, 4.0,
                1.0, 5.0,
            ],
        )
        .unwrap();
        let y = array![2.0, 3.0, 4.0, 6.0, 8.0, 12.0];

        let family = PoissonFamily;
        let link = LogLink;
        let irls_config = IRLSConfig::default();

        let reg_config = RegularizationConfig::ridge(1.0);
        let result = fit_glm_regularized(&y, &x, &family, &link, &irls_config, &reg_config, None, None).unwrap();

        assert!(result.converged);
        assert!(result.fitted_values.iter().all(|&x| x > 0.0));
    }

    #[test]
    fn test_lasso_requires_coordinate_descent() {
        // Lasso should return an error (not yet implemented)
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0]).unwrap();
        let y = array![2.0, 4.0, 6.0];

        let family = GaussianFamily;
        let link = IdentityLink;
        let irls_config = IRLSConfig::default();
        let reg_config = RegularizationConfig::lasso(1.0);

        let result = fit_glm_regularized(&y, &x, &family, &link, &irls_config, &reg_config, None, None);
        
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, RustyStatsError::InvalidValue(_)));
    }
}
