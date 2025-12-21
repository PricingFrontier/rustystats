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
use nalgebra::{DMatrix, DVector};

use crate::error::{RustyStatsError, Result};
use crate::families::Family;
use crate::links::Link;

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
            max_iterations: 25,
            tolerance: 1e-8,
            min_weight: 1e-10,
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

    // We'll store the final covariance matrix
    let mut cov_unscaled = Array2::zeros((p, p));
    let mut final_weights = Array1::zeros(n);

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

        let irls_weights: Array1<f64> = variance
            .iter()
            .zip(link_deriv.iter())
            .map(|(&v, &d)| {
                let w = 1.0 / (v * d * d);
                // Clip weights to avoid numerical issues
                w.max(config.min_weight).min(1e10)
            })
            .collect();

        // Combined weights = prior_weights × irls_weights
        let combined_weights: Array1<f64> = prior_weights_vec
            .iter()
            .zip(irls_weights.iter())
            .map(|(&pw, &iw)| pw * iw)
            .collect();

        // ---------------------------------------------------------------------
        // Step 4b: Compute working response z
        // ---------------------------------------------------------------------
        // z_i = (η_i - offset_i) + (y_i - μ_i) × g'(μ_i)
        //
        // Note: We subtract offset because we're solving for Xβ, not Xβ + offset
        // This is the "linearized" response that we'll regress on.
        // ---------------------------------------------------------------------
        let eta_no_offset: Array1<f64> = eta
            .iter()
            .zip(offset_vec.iter())
            .map(|(&e, &o)| e - o)
            .collect();

        let working_response: Array1<f64> = eta_no_offset
            .iter()
            .zip(y.iter())
            .zip(mu.iter())
            .zip(link_deriv.iter())
            .map(|(((&e, &yi), &mui), &d)| e + (yi - mui) * d)
            .collect();

        // ---------------------------------------------------------------------
        // Step 4c: Solve weighted least squares: (X'WX)β = X'Wz
        // ---------------------------------------------------------------------
        // This is the core linear algebra step.
        // We're finding β that minimizes: Σ w_i (z_i - x_i'β)²
        // ---------------------------------------------------------------------
        let (new_coefficients, xtwinv) =
            solve_weighted_least_squares(x, &working_response, &combined_weights)?;

        // ---------------------------------------------------------------------
        // Step 4d: Update η and μ
        // ---------------------------------------------------------------------
        // η = Xβ + offset (full linear predictor)
        // μ = g⁻¹(η) (fitted values on response scale)
        // ---------------------------------------------------------------------
        let eta_base = x.dot(&new_coefficients);
        eta = &eta_base + &offset_vec;
        mu = link.inverse(&eta);

        // Ensure μ stays valid
        mu = clamp_mu(&mu, family);

        // ---------------------------------------------------------------------
        // Step 4e: Check convergence
        // ---------------------------------------------------------------------
        deviance_old = deviance;
        deviance = family.deviance(y, &mu, Some(&prior_weights_vec));

        // Relative change in deviance
        let rel_change = if deviance_old.abs() > 1e-10 {
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

    let (final_coefficients, _) =
        solve_weighted_least_squares(x, &compute_working_response(y, &mu, &eta_no_offset, link), &combined_final_weights)?;

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
    })
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Solve weighted least squares: minimize Σ w_i (z_i - x_i'β)²
///
/// Returns (coefficients, (X'WX)⁻¹)
fn solve_weighted_least_squares(
    x: &Array2<f64>,
    z: &Array1<f64>,
    w: &Array1<f64>,
) -> Result<(Array1<f64>, Array2<f64>)> {
    let n = x.nrows();
    let p = x.ncols();

    // Convert to nalgebra matrices for linear algebra
    // X'WX and X'Wz

    // Create diagonal weight matrix W^(1/2) and multiply into X
    // This is more efficient than forming the full diagonal matrix
    let sqrt_w: Vec<f64> = w.iter().map(|&wi| wi.sqrt()).collect();

    // Form X_w = W^(1/2) X  (each row scaled by sqrt(w_i))
    let mut x_weighted = DMatrix::zeros(n, p);
    for i in 0..n {
        for j in 0..p {
            x_weighted[(i, j)] = x[[i, j]] * sqrt_w[i];
        }
    }

    // Form z_w = W^(1/2) z
    let z_weighted: DVector<f64> = DVector::from_iterator(
        n,
        z.iter().zip(sqrt_w.iter()).map(|(&zi, &swi)| zi * swi),
    );

    // Now solve: (X_w'X_w)β = X_w'z_w
    // Which is equivalent to: (X'WX)β = X'Wz

    // Compute X'WX = X_w'X_w
    let xtx = x_weighted.transpose() * &x_weighted;

    // Compute X'Wz = X_w'z_w
    let xtz = x_weighted.transpose() * z_weighted;

    // Solve the system using Cholesky decomposition (for positive definite matrices)
    // If that fails, fall back to LU decomposition
    let coefficients = match xtx.clone().cholesky() {
        Some(chol) => chol.solve(&xtz),
        None => {
            // Try LU decomposition as fallback
            match xtx.clone().lu().solve(&xtz) {
                Some(sol) => sol,
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

    // Compute (X'WX)⁻¹ for standard errors
    let xtx_inv = match xtx.clone().cholesky() {
        Some(chol) => {
            let identity = DMatrix::identity(p, p);
            chol.solve(&identity)
        }
        None => match xtx.try_inverse() {
            Some(inv) => inv,
            None => {
                // Return zeros if we can't compute inverse
                // Standard errors will be unreliable
                DMatrix::zeros(p, p)
            }
        },
    };

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
        "Poisson" | "Gamma" => x.max(1e-10), // Must be positive
        "Binomial" => x.max(1e-10).min(1.0 - 1e-10), // Must be in (0, 1)
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
}
