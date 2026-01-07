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
    MU_MIN_POSITIVE, MU_MIN_PROBABILITY, MU_MAX_PROBABILITY,
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
}

impl Default for IRLSConfig {
    fn default() -> Self {
        Self {
            max_iterations: DEFAULT_MAX_ITER,
            tolerance: CONVERGENCE_TOL,
            min_weight: MIN_IRLS_WEIGHT,
            verbose: false,
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
    // Step 1: Initialize μ using the family's method
    // -------------------------------------------------------------------------
    let mut mu = family.initialize_mu(y);

    // Ensure μ is valid (e.g., positive for Poisson, in (0,1) for Binomial)
    if !family.is_valid_mu(&mu) {
        // Try a safer initialization
        mu = initialize_mu_safe(y, family);
    }

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

    while iteration < config.max_iterations {
        iteration += 1;

        // ---------------------------------------------------------------------
        // Step 4a: Compute working weights W
        // ---------------------------------------------------------------------
        // The IRLS weight for observation i is:
        //     w_irls_i = 1 / (V(μ_i) × g'(μ_i)²)
        //
        // Combined with prior weights:
        //     w_i = prior_weight_i × w_irls_i
        //
        // This accounts for:
        //   - Variance function V(μ): higher variance → lower weight
        //   - Link derivative g'(μ): transformation adjustment
        //   - Prior weights: user-specified importance
        // ---------------------------------------------------------------------
        let variance = family.variance(&mu);
        let link_deriv = link.derivative(&mu);

        // PARALLEL: Compute IRLS weights, combined weights, and working response
        // in a single parallel pass to minimize allocation overhead
        let n = y.len();
        let min_weight = config.min_weight;
        
        let results: Vec<(f64, f64, f64)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let v = variance[i];
                let d = link_deriv[i];
                
                // IRLS weight
                let iw = (1.0 / (v * d * d)).max(min_weight).min(1e10);
                
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
        let (new_coefficients, xtwinv) =
            solve_weighted_least_squares(x, &working_response, &combined_weights)?;

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
        // Step 4d: Update η and μ with step-halving for stability
        // ---------------------------------------------------------------------
        // If deviance increases, reduce step size to prevent oscillation
        // ---------------------------------------------------------------------
        deviance_old = deviance;
        
        // Try full step first
        let eta_base = x.dot(&new_coefficients);
        let mut eta_new = &eta_base + &offset_vec;
        let mut mu_new = link.inverse(&eta_new);
        mu_new = clamp_mu(&mu_new, family);
        let mut deviance_new = family.deviance(y, &mu_new, Some(&prior_weights_vec));
        
        // Step-halving: if deviance increased, try smaller steps
        if deviance_new > deviance_old * 1.0001 && iteration > 1 {
            // Get old eta without offset for blending
            let eta_old_base: Array1<f64> = eta.iter()
                .zip(offset_vec.iter())
                .map(|(&e, &o)| e - o)
                .collect();
            
            let mut step_size = 0.5;
            for _half_step in 0..4 {
                // Blend: eta_new = (1-step)*eta_old + step*eta_full
                let eta_blend: Array1<f64> = eta_old_base.iter()
                    .zip(eta_base.iter())
                    .map(|(&old, &new)| (1.0 - step_size) * old + step_size * new)
                    .collect();
                eta_new = &eta_blend + &offset_vec;
                mu_new = link.inverse(&eta_new);
                mu_new = clamp_mu(&mu_new, family);
                deviance_new = family.deviance(y, &mu_new, Some(&prior_weights_vec));
                
                if deviance_new <= deviance_old * 1.0001 {
                    break;
                }
                step_size *= 0.5;
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
    // Step 5: Extract final coefficients from last iteration
    // -------------------------------------------------------------------------
    // Compute working response accounting for offset
    let eta_no_offset: Array1<f64> = eta
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
    let final_coefficients = match solve_weighted_least_squares(x, &compute_working_response(y, &mu, &eta_no_offset, link), &combined_final_weights) {
        Ok((coef, _)) if !coef.iter().any(|&c| c.is_nan() || c.is_infinite()) => coef,
        _ => {
            // Final extraction failed or produced NaN - use coefficients from last iteration
            if iter_coefficients.iter().any(|&c| c.is_nan() || c.is_infinite()) {
                return Err(RustyStatsError::NumericalError(
                    "IRLS produced NaN or infinite coefficients. This usually indicates: \
                     (1) severe multicollinearity in predictors, \
                     (2) extreme scale differences between variables, or \
                     (3) separation in binary response data. \
                     Try standardizing continuous predictors or removing correlated terms.".to_string()
                ));
            }
            iter_coefficients
        }
    };

    Ok(IRLSResult {
        coefficients: final_coefficients,
        fitted_values: mu,
        linear_predictor: eta,
        deviance,
        iterations: iteration,
        converged,
        covariance_unscaled: cov_unscaled,
        irls_weights: final_weights,
        prior_weights: prior_weights_vec,
        offset: offset_vec,
        y: y.to_owned(),  // Only clone at the end, needed for diagnostics
        family_name: family.name().to_string(),
        penalty: Penalty::None,
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
    let n = y.len();
    let p = x.ncols();

    if x.nrows() != n {
        return Err(RustyStatsError::DimensionMismatch(format!(
            "X has {} rows but y has {} elements", x.nrows(), n
        )));
    }

    if init_coefficients.len() != p {
        return Err(RustyStatsError::DimensionMismatch(format!(
            "init_coefficients has {} elements but X has {} columns",
            init_coefficients.len(), p
        )));
    }

    let offset_vec = match offset {
        Some(o) => o.clone(),
        None => Array1::zeros(n),
    };

    let prior_weights_vec = match weights {
        Some(w) => w.clone(),
        None => Array1::ones(n),
    };

    // Initialize from provided coefficients instead of family method
    let mut eta: Array1<f64> = x.dot(init_coefficients) + &offset_vec;
    let mut mu = link.inverse(&eta);
    
    // Clamp mu to valid range for the family (using same function as fit_glm_full)
    mu = clamp_mu(&mu, family);

    let mut deviance = family.deviance(y, &mu, Some(&prior_weights_vec));
    let mut deviance_old: f64;
    let mut converged = false;
    let mut iteration = 0;
    let mut cov_unscaled = Array2::zeros((p, p));
    let mut final_weights = Array1::zeros(n);
    let mut iter_coefficients = init_coefficients.clone();  // Store coefficients from iteration

    while iteration < config.max_iterations {
        iteration += 1;

        let variance = family.variance(&mu);
        let link_deriv = link.derivative(&mu);

        let min_weight = config.min_weight;
        let results: Vec<(f64, f64, f64)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let v = variance[i];
                let d = link_deriv[i];
                let iw = (1.0 / (v * d * d)).max(min_weight).min(1e10);
                let cw = prior_weights_vec[i] * iw;
                let e = eta[i] - offset_vec[i];
                let wr = e + (y[i] - mu[i]) * d;
                (iw, cw, wr)
            })
            .collect();

        let mut irls_weights = Array1::zeros(n);
        let mut combined_weights = Array1::zeros(n);
        let mut working_response = Array1::zeros(n);
        for i in 0..n {
            irls_weights[i] = results[i].0;
            combined_weights[i] = results[i].1;
            working_response[i] = results[i].2;
        }

        let (beta, covariance) = solve_weighted_least_squares(x, &working_response, &combined_weights)?;
        
        // Check for NaN in coefficients - indicates numerical instability
        if beta.iter().any(|&c| c.is_nan() || c.is_infinite()) {
            return Err(RustyStatsError::NumericalError(
                "IRLS produced NaN or infinite coefficients. This usually indicates: \
                 (1) severe multicollinearity in predictors, \
                 (2) extreme scale differences between variables, or \
                 (3) separation in binary response data. \
                 Try standardizing continuous predictors or removing correlated terms.".to_string()
            ));
        }
        
        cov_unscaled = covariance;
        final_weights = irls_weights;
        iter_coefficients = beta.clone();  // Store coefficients from this iteration

        eta = x.dot(&beta) + &offset_vec;
        mu = link.inverse(&eta);
        
        // Clamp mu to valid range for the family
        mu = clamp_mu(&mu, family);

        deviance_old = deviance;
        deviance = family.deviance(y, &mu, Some(&prior_weights_vec));

        // Use same convergence criterion as fit_glm_full: relative change in deviance
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

    let eta_no_offset: Array1<f64> = eta.iter().zip(offset_vec.iter())
        .map(|(&e, &o)| e - o).collect();
    
    let combined_final_weights: Array1<f64> = prior_weights_vec.iter()
        .zip(final_weights.iter())
        .map(|(&pw, &iw)| pw * iw).collect();

    // Try final coefficient extraction, but fall back to iteration coefficients if it produces NaN
    let final_coefficients = match solve_weighted_least_squares(x, &compute_working_response(y, &mu, &eta_no_offset, link), &combined_final_weights) {
        Ok((coef, _)) if !coef.iter().any(|&c| c.is_nan() || c.is_infinite()) => coef,
        _ => {
            // Final extraction failed or produced NaN - use coefficients from last iteration
            if iter_coefficients.iter().any(|&c| c.is_nan() || c.is_infinite()) {
                return Err(RustyStatsError::NumericalError(
                    "IRLS produced NaN or infinite coefficients. This usually indicates: \
                     (1) severe multicollinearity in predictors, \
                     (2) extreme scale differences between variables, or \
                     (3) separation in binary response data. \
                     Try standardizing continuous predictors or removing correlated terms.".to_string()
                ));
            }
            iter_coefficients
        }
    };

    Ok(IRLSResult {
        coefficients: final_coefficients,
        fitted_values: mu,
        linear_predictor: eta,
        deviance,
        iterations: iteration,
        converged,
        covariance_unscaled: cov_unscaled,
        irls_weights: final_weights,
        prior_weights: prior_weights_vec,
        offset: offset_vec,
        y: y.to_owned(),
        family_name: family.name().to_string(),
        penalty: Penalty::None,
        design_matrix: None,
    })
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

    // -------------------------------------------------------------------------
    // Step 0: Validate inputs (same as fit_glm_full)
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
    // Step 1: Initialize μ using the family's method
    // -------------------------------------------------------------------------
    let mut mu = family.initialize_mu(y);

    if !family.is_valid_mu(&mu) {
        mu = initialize_mu_safe(y, family);
    }

    // -------------------------------------------------------------------------
    // Step 2: Initialize linear predictor η = g(μ)
    // -------------------------------------------------------------------------
    let mut eta = link.link(&mu);

    // -------------------------------------------------------------------------
    // Step 3: Calculate initial deviance
    // -------------------------------------------------------------------------
    let mut deviance = family.deviance(y, &mu, Some(&prior_weights_vec));
    let mut deviance_old: f64;

    // -------------------------------------------------------------------------
    // Step 4: IRLS iteration loop (with Ridge penalty)
    // -------------------------------------------------------------------------
    let mut converged = false;
    let mut iteration = 0;
    let mut cov_unscaled = Array2::zeros((p, p));
    let mut final_weights = Array1::zeros(n);

    while iteration < irls_config.max_iterations {
        iteration += 1;

        // Compute working weights W
        let variance = family.variance(&mu);
        let link_deriv = link.derivative(&mu);

        let min_weight = irls_config.min_weight;
        
        let results: Vec<(f64, f64, f64)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let v = variance[i];
                let d = link_deriv[i];
                let iw = (1.0 / (v * d * d)).max(min_weight).min(1e10);
                let cw = prior_weights_vec[i] * iw;
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

        // Solve PENALIZED weighted least squares: (X'WX + λI)β = X'Wz
        let (new_coefficients, xtwinv) =
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

        // Update η and μ
        let eta_base = x.dot(&new_coefficients);
        eta = &eta_base + &offset_vec;
        mu = link.inverse(&eta);
        mu = clamp_mu(&mu, family);

        // Check convergence
        deviance_old = deviance;
        deviance = family.deviance(y, &mu, Some(&prior_weights_vec));

        let rel_change = if deviance_old.abs() > ZERO_TOL {
            (deviance_old - deviance).abs() / deviance_old.abs()
        } else {
            (deviance_old - deviance).abs()
        };

        if irls_config.verbose {
            eprintln!(
                "Iteration {}: deviance = {:.6}, rel_change = {:.2e}, penalty = {:.4}",
                iteration, deviance, rel_change, l2_penalty
            );
        }

        if rel_change < irls_config.tolerance {
            converged = true;
            cov_unscaled = xtwinv;
            final_weights = irls_weights;
            break;
        }

        cov_unscaled = xtwinv;
        final_weights = irls_weights;
    }

    // -------------------------------------------------------------------------
    // Step 5: Extract final coefficients
    // -------------------------------------------------------------------------
    let eta_no_offset: Array1<f64> = eta
        .iter()
        .zip(offset_vec.iter())
        .map(|(&e, &o)| e - o)
        .collect();
    
    let combined_final_weights: Array1<f64> = prior_weights_vec
        .iter()
        .zip(final_weights.iter())
        .map(|(&pw, &iw)| pw * iw)
        .collect();

    let (final_coefficients, _) =
        solve_weighted_least_squares_penalized(
            x, 
            &compute_working_response(y, &mu, &eta_no_offset, link), 
            &combined_final_weights,
            l2_penalty,
            penalize_intercept,
        )?;

    // Check for NaN in final coefficients
    if final_coefficients.iter().any(|&c| c.is_nan() || c.is_infinite()) {
        return Err(RustyStatsError::NumericalError(
            "IRLS produced NaN or infinite coefficients in final extraction. This usually indicates: \
             (1) severe multicollinearity in predictors, \
             (2) extreme scale differences between variables, or \
             (3) separation in binary response data. \
             Try standardizing continuous predictors or removing correlated terms.".to_string()
        ));
    }

    Ok(IRLSResult {
        coefficients: final_coefficients,
        fitted_values: mu,
        linear_predictor: eta,
        deviance,
        iterations: iteration,
        converged,
        covariance_unscaled: cov_unscaled,
        irls_weights: final_weights,
        prior_weights: prior_weights_vec,
        offset: offset_vec,
        y: y.to_owned(),
        family_name: family.name().to_string(),
        penalty: reg_config.penalty.clone(),
        design_matrix: None,  // Computed lazily in Python layer to avoid expensive copy
    })
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Solve weighted least squares: minimize Σ w_i (z_i - x_i'β)²
///
/// Returns (coefficients, (X'WX)⁻¹)
/// 
/// OPTIMIZATION NOTES:
/// This is a hot path - called every IRLS iteration.
/// Key optimizations:
/// - PARALLEL: Split rows across threads for X'WX and X'Wz computation
/// - Compute X'WX directly without forming weighted X matrix
/// - Reuse Cholesky factorization for both solve and inverse
/// - Use chunk-based parallelism to reduce synchronization overhead
#[inline]
fn solve_weighted_least_squares(
    x: &Array2<f64>,
    z: &Array1<f64>,
    w: &Array1<f64>,
) -> Result<(Array1<f64>, Array2<f64>)> {
    let n = x.nrows();
    let p = x.ncols();

    // PARALLEL: Compute X'WX and X'Wz using parallel chunk processing
    let (xtx_data, xtz_data): (Vec<f64>, Vec<f64>) = (0..n)
        .into_par_iter()
        .fold(
            || (vec![0.0; p * p], vec![0.0; p]),
            |(mut xtx_local, mut xtz_local), k| {
                let wk = w[k];
                let wz = wk * z[k];
                
                for i in 0..p {
                    let xki = x[[k, i]];
                    let xki_w = xki * wk;
                    xtz_local[i] += xki * wz;
                    for j in i..p {
                        xtx_local[i * p + j] += xki_w * x[[k, j]];
                    }
                }
                (xtx_local, xtz_local)
            },
        )
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

    // Convert to nalgebra types
    let mut xtx = DMatrix::zeros(p, p);
    for i in 0..p {
        for j in i..p {
            let val = xtx_data[i * p + j];
            xtx[(i, j)] = val;
            xtx[(j, i)] = val;
        }
    }
    let xtz = DVector::from_vec(xtz_data);

    // Solve using Cholesky decomposition
    let chol = match xtx.clone().cholesky() {
        Some(c) => c,
        None => {
            match xtx.clone().lu().solve(&xtz) {
                Some(sol) => {
                    let coef_array: Array1<f64> = sol.iter().copied().collect();
                    let xtx_inv = xtx.try_inverse().ok_or_else(|| {
                        RustyStatsError::LinearAlgebraError(
                            "Failed to compute covariance matrix - X'WX is not invertible. \
                             This often indicates multicollinearity in predictors.".to_string()
                        )
                    })?;
                    let mut cov_array = Array2::zeros((p, p));
                    for i in 0..p {
                        for j in 0..p {
                            cov_array[[i, j]] = xtx_inv[(i, j)];
                        }
                    }
                    return Ok((coef_array, cov_array));
                }
                None => {
                    return Err(RustyStatsError::LinearAlgebraError(
                        "Failed to solve weighted least squares - matrix may be singular. \
                         This often indicates multicollinearity in predictors.".to_string(),
                    ));
                }
            }
        }
    };

    let coefficients = chol.solve(&xtz);
    let identity = DMatrix::identity(p, p);
    let xtx_inv = chol.solve(&identity);

    let coef_array: Array1<f64> = coefficients.iter().copied().collect();
    let mut cov_array = Array2::zeros((p, p));
    for i in 0..p {
        for j in 0..p {
            cov_array[[i, j]] = xtx_inv[(i, j)];
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
    let n = x.nrows();
    let p = x.ncols();

    // -------------------------------------------------------------------------
    // PARALLEL: Compute X'WX and X'Wz using parallel chunk processing
    // Each thread computes partial sums, then we reduce to final result
    // -------------------------------------------------------------------------
    
    // Parallel computation of X'WX and X'Wz
    // Each chunk returns (partial_xtx as Vec<f64>, partial_xtz as Vec<f64>)
    let (xtx_data, xtz_data): (Vec<f64>, Vec<f64>) = (0..n)
        .into_par_iter()
        .fold(
            || (vec![0.0; p * p], vec![0.0; p]),
            |(mut xtx_local, mut xtz_local), k| {
                let wk = w[k];
                let wz = wk * z[k];
                
                // Accumulate X'WX (upper triangle only)
                for i in 0..p {
                    let xki = x[[k, i]];
                    let xki_w = xki * wk;
                    
                    // X'Wz
                    xtz_local[i] += xki * wz;
                    
                    // X'WX upper triangle
                    for j in i..p {
                        xtx_local[i * p + j] += xki_w * x[[k, j]];
                    }
                }
                
                (xtx_local, xtz_local)
            },
        )
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

    // Convert to nalgebra types
    let mut xtx = DMatrix::zeros(p, p);
    for i in 0..p {
        for j in i..p {
            let val = xtx_data[i * p + j];
            xtx[(i, j)] = val;
            xtx[(j, i)] = val;  // Symmetrize
        }
    }
    
    let xtz = DVector::from_vec(xtz_data);

    // -------------------------------------------------------------------------
    // Add L2 (Ridge) penalty to diagonal: (X'WX + λI)
    // -------------------------------------------------------------------------
    // The intercept (first column) is typically NOT penalized.
    // This prevents the baseline from being shrunk toward zero.
    if l2_penalty > 0.0 {
        let start_idx = if penalize_intercept { 0 } else { 1 };
        for j in start_idx..p {
            xtx[(j, j)] += l2_penalty;
        }
    }

    // -------------------------------------------------------------------------
    // Solve using Cholesky decomposition (reuse for inverse)
    // -------------------------------------------------------------------------
    let chol = match xtx.clone().cholesky() {
        Some(c) => c,
        None => {
            // Fall back to LU decomposition
            match xtx.clone().lu().solve(&xtz) {
                Some(sol) => {
                    // LU worked for solve, try inverse
                    let coef_array: Array1<f64> = sol.iter().copied().collect();
                    let xtx_inv = xtx.try_inverse().ok_or_else(|| {
                        RustyStatsError::LinearAlgebraError(
                            "Failed to compute covariance matrix - X'WX is not invertible. \
                             This often indicates multicollinearity in predictors.".to_string()
                        )
                    })?;
                    let mut cov_array = Array2::zeros((p, p));
                    for i in 0..p {
                        for j in 0..p {
                            cov_array[[i, j]] = xtx_inv[(i, j)];
                        }
                    }
                    return Ok((coef_array, cov_array));
                }
                None => {
                    return Err(RustyStatsError::LinearAlgebraError(
                        "Failed to solve weighted least squares - matrix may be singular. \
                         This often indicates multicollinearity in predictors."
                            .to_string(),
                    ));
                }
            }
        }
    };

    // Solve for coefficients
    let coefficients = chol.solve(&xtz);

    // Compute inverse using same Cholesky factorization
    let identity = DMatrix::identity(p, p);
    let xtx_inv = chol.solve(&identity);

    // Convert back to ndarray
    let coef_array: Array1<f64> = coefficients.iter().copied().collect();
    let mut cov_array = Array2::zeros((p, p));
    for i in 0..p {
        for j in 0..p {
            cov_array[[i, j]] = xtx_inv[(i, j)];
        }
    }

    Ok((coef_array, cov_array))
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

/// Safe initialization of μ that works for any family
fn initialize_mu_safe(y: &Array1<f64>, family: &dyn Family) -> Array1<f64> {
    let y_mean = y.mean().unwrap_or(1.0).max(0.01);
    let name = family.name();

    // Initialize to a weighted average of y and the mean
    // This avoids extreme values while staying close to the data
    y.mapv(|yi| {
        let val = (yi + y_mean) / 2.0;
        // Clamp based on family
        match name {
            "Poisson" | "Gamma" => val.max(0.001),
            "Binomial" => val.max(0.001).min(0.999),
            _ => val,
        }
    })
}

/// Clamp μ to valid range for the family
fn clamp_mu(mu: &Array1<f64>, family: &dyn Family) -> Array1<f64> {
    let name = family.name();
    mu.mapv(|x| match name {
        "Poisson" | "Gamma" => x.max(MU_MIN_POSITIVE), // Must be positive
        "Binomial" => x.max(MU_MIN_PROBABILITY).min(MU_MAX_PROBABILITY), // Must be in (0, 1)
        _ => x, // Gaussian allows any value
    })
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
