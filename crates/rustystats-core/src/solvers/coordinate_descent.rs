use super::{compute_irls_weights, initialize_mu_safe, validate_glm_inputs};
use ndarray::ArrayView2;

// =============================================================================
// Coordinate Descent for Penalized GLMs
// =============================================================================
//
// This module implements coordinate descent for fitting GLMs with L1 (Lasso)
// and Elastic Net penalties. Unlike Ridge, these penalties have non-differentiable
// components that require special handling.
//
// ALGORITHM: Iteratively Reweighted Coordinate Descent (IRCD)
// -----------------------------------------------------------
// We combine two ideas:
//   1. IRLS: Handle the GLM part by iteratively computing working response/weights
//   2. Coordinate Descent: Handle the L1 penalty by updating one coefficient at a time
//
// The algorithm:
//   1. Initialize coefficients β
//   2. Outer loop (IRLS-like):
//      a. Compute working response z and weights W from current μ
//      b. Inner loop (coordinate descent):
//         For each j = 1, ..., p:
//           - Compute partial residual without β_j
//           - Update β_j using soft-thresholding
//         Until convergence
//      c. Update μ = g⁻¹(Xβ)
//      d. Check deviance convergence
//
// SOFT THRESHOLDING
// -----------------
// The key operation for L1 penalty. For weighted least squares:
//
//   β_j = S(r_j, λα) / (Σw_i x_ij² + λ(1-α))
//
// where:
//   - r_j = Σw_i x_ij (z_i - Σ_{k≠j} x_ik β_k) is the "partial residual"
//   - S(z, γ) = sign(z) × max(0, |z| - γ) is soft-thresholding
//   - α is the L1 ratio (1 for Lasso, 0 for Ridge)
//   - λ is the overall penalty strength
//
// INTERCEPT
// ---------
// The intercept is NOT penalized. It's updated as:
//   β_0 = Σw_i (z_i - Σ_{j>0} x_ij β_j) / Σw_i
//
// =============================================================================

use ndarray::{Array1, Array2};

use super::irls::{compute_xtwx, compute_xtwx_xtwz, IRLSConfig, IRLSResult};
use crate::constants::ZERO_TOL;
use crate::error::Result;
use crate::families::Family;
use crate::links::Link;
use crate::regularization::{soft_threshold, RegularizationConfig};

const ACTIVE_SET_REFRESH_INTERVAL: usize = 5;
const COEFFICIENT_UPDATE_TOL: f64 = 1e-15;

/// Fit a GLM using coordinate descent with L1/Elastic Net penalty.
///
/// This is the main entry point for Lasso and Elastic Net regularized GLMs.
///
/// # Arguments
/// * `y` - Response variable (n × 1)
/// * `x` - Design matrix (n × p), should include intercept column as first column
/// * `family` - Distribution family (Gaussian, Poisson, Binomial, Gamma)
/// * `link` - Link function (Identity, Log, Logit)
/// * `irls_config` - Outer loop (IRLS) configuration
/// * `reg_config` - Regularization configuration (must have L1 component)
/// * `offset` - Optional offset term
/// * `weights` - Optional prior weights
/// * `init_coefficients` - Optional initial coefficients for warm starting
/// * `skip_covariance` - If true, skip the expensive O(n×p²) covariance computation
///   (useful during CV path where only coefficients are needed)
///
/// # Returns
/// * `Ok(IRLSResult)` - Fitted model results
/// * `Err(RustyStatsError)` - If fitting fails
pub(crate) fn fit_glm_coordinate_descent(
    y: &Array1<f64>,
    x: ArrayView2<'_, f64>,
    family: &dyn Family,
    link: &dyn Link,
    irls_config: &IRLSConfig,
    reg_config: &RegularizationConfig,
    offset: Option<&Array1<f64>>,
    weights: Option<&Array1<f64>>,
    init_coefficients: Option<&Array1<f64>>,
    skip_covariance: bool,
) -> Result<IRLSResult> {
    // -------------------------------------------------------------------------
    // Step 0: Validate inputs and set up offset/weights
    // -------------------------------------------------------------------------
    let n = y.len();
    let p = x.ncols();
    let validated = validate_glm_inputs(y, x, offset, weights)?;
    let offset_vec = validated.offset;
    let prior_weights_vec = validated.prior_weights;

    // Get penalty parameters
    let l1_penalty = reg_config.penalty.l1_penalty();
    let l2_penalty = reg_config.penalty.l2_penalty();
    let has_intercept = reg_config.fit_intercept;

    // Starting index for penalized coefficients
    let pen_start = if has_intercept { 1 } else { 0 };

    // -------------------------------------------------------------------------
    // Step 1: Precompute X'X diagonal (sum of squared predictors, weighted)
    // These are recomputed each IRLS iteration with updated weights
    // -------------------------------------------------------------------------

    // -------------------------------------------------------------------------
    // Step 2: Initialize coefficients and μ (with warm start support)
    // -------------------------------------------------------------------------
    let mut warnings: Vec<String> = Vec::new();
    let mut coefficients = if let Some(init) = init_coefficients {
        if init.len() == p {
            init.clone()
        } else {
            // Dimension mismatch - fall back to cold start with warning
            warnings.push(format!(
                "Warm-start coefficient dimension mismatch (got {}, expected {}). \
                Falling back to cold start. This may indicate a bug in the caller.",
                init.len(),
                p
            ));
            Array1::zeros(p)
        }
    } else {
        Array1::zeros(p)
    };

    // Initialize intercept to link(mean(y)) only if not warm starting
    if should_initialize_intercept_from_response(init_coefficients.is_some(), has_intercept) {
        coefficients[0] = intercept_start_from_response(y, family, link);
    }

    // Initialize μ from coefficients if warm starting, otherwise from y
    let mut mu = if init_coefficients.is_some() {
        initialize_mu_from_coefficients(x, &coefficients, &offset_vec, family, link)
    } else {
        initialize_mu_from_response(y, family)
    };

    // -------------------------------------------------------------------------
    // Step 3: Initialize linear predictor
    // -------------------------------------------------------------------------
    let mut eta = link.link(&mu);

    // -------------------------------------------------------------------------
    // Step 4: Calculate initial deviance
    // -------------------------------------------------------------------------
    let mut deviance = family.deviance(y, &mu, Some(&prior_weights_vec));
    let mut deviance_old: f64;

    // -------------------------------------------------------------------------
    // Step 5: Outer IRLS loop
    // -------------------------------------------------------------------------
    let mut converged = false;
    let mut outer_iteration = 0;
    let mut irls_weights = Array1::zeros(n);

    while should_continue_iteration(outer_iteration, irls_config.max_iterations) {
        advance_iteration(&mut outer_iteration);

        // ---------------------------------------------------------------------
        // Step 5a: Compute working weights and working response
        // ---------------------------------------------------------------------
        let weight_result = compute_irls_weights(
            y,
            &mu,
            &eta,
            &offset_vec,
            &prior_weights_vec,
            family,
            link,
            irls_config.min_weight,
        )?;
        irls_weights = weight_result.irls_weights;
        let combined_weights = weight_result.combined_weights;
        let working_response = weight_result.working_response;

        // ---------------------------------------------------------------------
        // Step 5b: Coordinate descent using COVARIANCE UPDATES (glmnet-style)
        // ---------------------------------------------------------------------
        // Key optimization: Precompute X'Wz and X'WX, then use O(p) updates
        // instead of O(np) per coefficient.
        //
        // The normal equations: (X'WX + λI)β = X'Wz
        // For coordinate descent, we need:
        //   β_j = S(X_j'W(z - X_{-j}β_{-j}), λ₁) / (X_j'WX_j + λ₂)
        //
        // Using covariance trick:
        //   X_j'W(z - Xβ + X_jβ_j) = X_j'Wz - X_j'WXβ + X_j'WX_jβ_j
        //                          = grad_j + (X'WX)_{jj}β_j
        // where grad_j = X_j'Wz - Σ_k (X'WX)_{jk}β_k
        // ---------------------------------------------------------------------

        let mut cd_converged = false;
        let mut cd_iteration = 0;

        let (xwx_matrix, xwz_vector) = compute_xtwx_xtwz(x, &working_response, &combined_weights)?;

        // Keep the coordinate-update loop on a row-major flat matrix. nalgebra
        // stores DMatrix column-major, while each gradient update scans a full
        // row of X'WX.
        let xwz: Vec<f64> = xwz_vector.iter().copied().collect();
        let mut xwx = vec![0.0_f64; p * p];
        for j in 0..p {
            for k in 0..p {
                xwx[j * p + k] = xwx_matrix[(j, k)];
            }
        }

        // Active set: track which coefficients are non-zero for faster iterations
        let all_indices: Vec<usize> = (0..p).collect();
        let mut active_set = all_indices.clone();
        let mut use_active_set = false;

        while should_continue_iteration(cd_iteration, reg_config.max_cd_iterations) {
            advance_iteration(&mut cd_iteration);
            let mut max_change = 0.0_f64;

            // Decide which coefficients to update
            let indices_to_update: &[usize] = coordinate_indices_for_iteration(
                use_active_set,
                cd_iteration,
                &all_indices,
                &active_set,
            );

            // Update each coefficient using covariance updates
            for &j in indices_to_update {
                let old_coef = coefficients[j];

                // Compute gradient: grad_j = X_j'Wz - Σ_k (X'WX)_{jk}β_k
                let mut grad_j = xwz[j];
                for k in 0..p {
                    grad_j -= xwx[j * p + k] * coefficients[k];
                }

                // rho = grad_j + (X'WX)_{jj} * old_coef
                let xwx_jj = xwx[j * p + j];
                let rho = grad_j + xwx_jj * old_coef;

                // Update coefficient with soft-thresholding
                let new_coef =
                    coordinate_update_value(j, pen_start, rho, xwx_jj, l1_penalty, l2_penalty);

                let delta = apply_coordinate_update(&mut coefficients, j, old_coef, new_coef);
                max_change = max_change.max(delta);
            }

            // Update active set after first full pass
            if should_refresh_active_set(cd_iteration) {
                let refreshed = refresh_active_set(&coefficients, pen_start);
                active_set = refreshed.0;
                use_active_set = refreshed.1;
            }

            // Check convergence
            if coordinate_descent_converged(max_change, reg_config.cd_tolerance) {
                cd_converged = true;
                break;
            }
        }

        if !cd_converged {
            if irls_config.verbose {
                eprintln!(
                    "Warning: Coordinate descent did not converge in {} iterations",
                    reg_config.max_cd_iterations
                );
            }
            warnings.push(format!(
                "Coordinate descent did not converge in {} iterations",
                reg_config.max_cd_iterations
            ));
        }

        // ---------------------------------------------------------------------
        // Step 5c: Apply coefficient sign constraints
        // ---------------------------------------------------------------------
        project_sign_constraints(
            &mut coefficients,
            &irls_config.nonneg_indices,
            &irls_config.nonpos_indices,
        );

        // ---------------------------------------------------------------------
        // Step 5d: Update η and μ
        // ---------------------------------------------------------------------
        eta = linear_predictor_with_offset(x, &coefficients, &offset_vec);
        mu = link.inverse(&eta);
        mu = family.clamp_mu(&mu);

        // ---------------------------------------------------------------------
        // Step 5e: Check outer loop convergence
        // ---------------------------------------------------------------------
        deviance_old = deviance;
        deviance = family.deviance(y, &mu, Some(&prior_weights_vec));

        let (abs_change, rel_change) = deviance_changes(deviance_old, deviance);

        if irls_config.verbose {
            let n_nonzero = count_nonzero_penalized(&coefficients, pen_start);
            eprintln!(
                "IRLS iter {}: deviance = {:.6}, rel_change = {:.2e}, nonzero = {}",
                outer_iteration, deviance, rel_change, n_nonzero
            );
        }

        // Converge if relative change is small OR if deviance is very small (nearly perfect fit)
        if outer_deviance_converged(rel_change, deviance, abs_change, irls_config.tolerance) {
            converged = true;
            break;
        }
    }

    // -------------------------------------------------------------------------
    // Step 6: Compute covariance estimate
    // -------------------------------------------------------------------------
    // IMPORTANT LIMITATION FOR ACTUARIAL USERS:
    // For penalized models (Lasso/Elastic Net), standard errors are approximate.
    // The covariance is computed using only non-zero coefficients, which does not
    // account for the selection bias introduced by penalization.
    //
    // For rigorous inference on regularized models, consider:
    // 1. Bootstrap confidence intervals
    // 2. De-biased Lasso methods
    // 3. Post-selection inference techniques
    //
    // The standard errors returned here should be used with caution for
    // hypothesis testing or confidence interval construction.
    let cov_unscaled = if skip_covariance {
        Array2::zeros((p, p))
    } else {
        compute_penalized_covariance(
            x,
            &irls_weights,
            &prior_weights_vec,
            &coefficients,
            pen_start,
        )
    };

    Ok(IRLSResult {
        coefficients,
        fitted_values: mu,
        linear_predictor: eta,
        deviance,
        iterations: outer_iteration,
        converged,
        covariance_unscaled: cov_unscaled,
        irls_weights,
        prior_weights: prior_weights_vec,
        offset: offset_vec,
        y: y.to_owned(),
        family_name: family.name().to_string(),
        penalty: reg_config.penalty.clone(),
        design_matrix: None, // Computed lazily in Python layer to avoid expensive copy
        warnings,
        step_halving_used: false,
        solver_status: if converged {
            "converged".to_string()
        } else {
            "max_iterations".to_string()
        },
        profile: None,
    })
}

fn should_initialize_intercept_from_response(has_warm_start: bool, has_intercept: bool) -> bool {
    !has_warm_start && has_intercept
}

fn intercept_start_from_response(y: &Array1<f64>, family: &dyn Family, link: &dyn Link) -> f64 {
    let y_mean = y.mean().unwrap_or(1.0);
    let y_mean_clamped = family.clamp_mu(&Array1::from_elem(1, y_mean))[0];
    link.link(&Array1::from_elem(1, y_mean_clamped))[0]
}

fn initialize_mu_from_coefficients(
    x: ArrayView2<'_, f64>,
    coefficients: &Array1<f64>,
    offset: &Array1<f64>,
    family: &dyn Family,
    link: &dyn Link,
) -> Array1<f64> {
    let eta = linear_predictor_with_offset(x, coefficients, offset);
    let mu_init = link.inverse(&eta);
    family.clamp_mu(&mu_init)
}

fn initialize_mu_from_response(y: &Array1<f64>, family: &dyn Family) -> Array1<f64> {
    let mu_init = family.initialize_mu(y);
    if !family.is_valid_mu(&mu_init) {
        initialize_mu_safe(y, family)
    } else {
        mu_init
    }
}

fn should_continue_iteration(iteration: usize, max_iterations: usize) -> bool {
    iteration < max_iterations
}

fn advance_iteration(iteration: &mut usize) {
    *iteration += 1;
}

fn should_use_active_subset(use_active_set: bool, cd_iteration: usize) -> bool {
    use_active_set && !cd_iteration.is_multiple_of(ACTIVE_SET_REFRESH_INTERVAL)
}

fn coordinate_indices_for_iteration<'a>(
    use_active_set: bool,
    cd_iteration: usize,
    all_indices: &'a [usize],
    active_set: &'a [usize],
) -> &'a [usize] {
    if should_use_active_subset(use_active_set, cd_iteration) {
        // Use active set (non-zero coefficients + intercept)
        active_set
    } else {
        // Full pass every ACTIVE_SET_REFRESH_INTERVAL iterations or initially
        all_indices
    }
}

fn coordinate_update_value(
    j: usize,
    pen_start: usize,
    rho: f64,
    xwx_jj: f64,
    l1_penalty: f64,
    l2_penalty: f64,
) -> f64 {
    if j < pen_start {
        rho / xwx_jj
    } else {
        let denom = xwx_jj + l2_penalty;
        if denom.abs() < ZERO_TOL {
            0.0
        } else {
            soft_threshold(rho, l1_penalty) / denom
        }
    }
}

fn apply_coordinate_update(
    coefficients: &mut Array1<f64>,
    j: usize,
    old_coef: f64,
    new_coef: f64,
) -> f64 {
    let delta = (new_coef - old_coef).abs();
    if delta > COEFFICIENT_UPDATE_TOL {
        coefficients[j] = new_coef;
    }
    delta
}

fn should_refresh_active_set(cd_iteration: usize) -> bool {
    cd_iteration == 1 || cd_iteration.is_multiple_of(ACTIVE_SET_REFRESH_INTERVAL)
}

fn active_set_from_coefficients(coefficients: &Array1<f64>, pen_start: usize) -> Vec<usize> {
    let p = coefficients.len();
    let mut active_set: Vec<usize> = (0..pen_start.min(p)).collect();
    for j in pen_start..p {
        if coefficients[j].abs() > ZERO_TOL {
            active_set.push(j);
        }
    }
    active_set
}

fn refresh_active_set(coefficients: &Array1<f64>, pen_start: usize) -> (Vec<usize>, bool) {
    let active_set = active_set_from_coefficients(coefficients, pen_start);
    let use_active_set = active_set.len() < coefficients.len();
    (active_set, use_active_set)
}

fn coordinate_descent_converged(max_change: f64, tolerance: f64) -> bool {
    max_change < tolerance
}

fn project_sign_constraints(
    coefficients: &mut Array1<f64>,
    nonneg_indices: &[usize],
    nonpos_indices: &[usize],
) {
    // Project non-negative constrained coefficients to be >= 0 (for ms(), pos())
    for &idx in nonneg_indices {
        if idx < coefficients.len() && coefficients[idx] < 0.0 {
            coefficients[idx] = 0.0;
        }
    }
    // Project non-positive constrained coefficients to be <= 0 (for neg())
    for &idx in nonpos_indices {
        if idx < coefficients.len() && coefficients[idx] > 0.0 {
            coefficients[idx] = 0.0;
        }
    }
}

fn linear_predictor_with_offset(
    x: ArrayView2<'_, f64>,
    coefficients: &Array1<f64>,
    offset: &Array1<f64>,
) -> Array1<f64> {
    let eta_base = x.dot(coefficients);
    &eta_base + offset
}

fn deviance_changes(deviance_old: f64, deviance: f64) -> (f64, f64) {
    let abs_change = (deviance_old - deviance).abs();
    let rel_change = if deviance_old.abs() > ZERO_TOL {
        abs_change / deviance_old.abs()
    } else {
        abs_change
    };
    (abs_change, rel_change)
}

fn count_nonzero_penalized(coefficients: &Array1<f64>, pen_start: usize) -> usize {
    coefficients
        .iter()
        .skip(pen_start)
        .filter(|&&c| c.abs() > ZERO_TOL)
        .count()
}

fn outer_deviance_converged(
    rel_change: f64,
    deviance: f64,
    abs_change: f64,
    tolerance: f64,
) -> bool {
    rel_change < tolerance || (deviance < ZERO_TOL && abs_change < ZERO_TOL)
}

/// Compute an approximate covariance matrix for penalized estimates.
///
/// For Lasso/Elastic Net, standard errors are not well-defined in the classical sense.
/// This computes a naive estimate that can be used for rough inference.
fn compute_penalized_covariance(
    x: ArrayView2<'_, f64>,
    irls_weights: &Array1<f64>,
    prior_weights: &Array1<f64>,
    coefficients: &Array1<f64>,
    pen_start: usize,
) -> Array2<f64> {
    let p = x.ncols();

    let weights = combined_covariance_weights(irls_weights, prior_weights);

    let xtwx = compute_xtwx(x, &weights);

    let active_indices = active_set_from_coefficients(coefficients, pen_start);

    if active_indices.is_empty() {
        return Array2::zeros((p, p));
    }

    use nalgebra::DMatrix;
    let active_p = active_indices.len();
    let mut xtx_active = DMatrix::zeros(active_p, active_p);
    for (ai, &i) in active_indices.iter().enumerate() {
        for (aj, &j) in active_indices.iter().enumerate() {
            xtx_active[(ai, aj)] = xtwx[[i, j]];
        }
    }

    let mut cov = Array2::zeros((p, p));
    if let Some(inv) = xtx_active.try_inverse() {
        for (ai, &i) in active_indices.iter().enumerate() {
            for (aj, &j) in active_indices.iter().enumerate() {
                cov[[i, j]] = inv[(ai, aj)];
            }
        }
    } else {
        for &i in &active_indices {
            cov[[i, i]] = f64::NAN;
        }
    }

    cov
}

fn combined_covariance_weights(
    irls_weights: &Array1<f64>,
    prior_weights: &Array1<f64>,
) -> Array1<f64> {
    irls_weights
        .iter()
        .zip(prior_weights.iter())
        .map(|(&iw, &pw)| iw * pw)
        .collect()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::families::{GaussianFamily, PoissonFamily};
    use crate::links::{IdentityLink, Link, LogLink};
    use approx::assert_abs_diff_eq;
    use ndarray::array;
    use std::borrow::Cow;

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

    struct ShortVarianceGaussian;

    impl Family for ShortVarianceGaussian {
        fn name(&self) -> &str {
            "ShortVarianceGaussian"
        }

        fn variance<'a>(&self, mu: &'a Array1<f64>) -> Cow<'a, Array1<f64>> {
            Cow::Owned(Array1::ones(mu.len().saturating_sub(1)))
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
            y.clone()
        }

        fn is_valid_mu(&self, mu: &Array1<f64>) -> bool {
            mu.iter().all(|value| value.is_finite())
        }

        fn clamp_mu(&self, mu: &Array1<f64>) -> Array1<f64> {
            mu.clone()
        }
    }

    #[test]
    fn custom_family_fixtures_are_self_consistent() {
        let y = array![1.0, 2.0, 4.0];
        let mu = array![1.0, 1.5, 3.0];

        let invalid = InvalidInitGaussian;
        assert_eq!(invalid.name(), "InvalidInitGaussian");
        assert_abs_diff_eq!(
            invalid.unit_deviance(&y, &mu),
            array![0.0, 0.25, 1.0],
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(invalid.unit_deviance_at(4.0, 3.0), 1.0, epsilon = 1e-12);
        assert_eq!(invalid.default_link().name(), "identity");
        assert!(invalid.initialize_mu(&y).iter().all(|v| v.is_nan()));
        assert!(!invalid.is_valid_mu(&array![1.0, f64::NAN]));
        assert_abs_diff_eq!(
            invalid.clamp_mu(&array![f64::NAN, 2.0]),
            array![0.0, 2.0],
            epsilon = 1e-12
        );

        let short = ShortVarianceGaussian;
        assert_eq!(short.name(), "ShortVarianceGaussian");
        assert_eq!(short.variance(&mu).len(), 2);
        assert_abs_diff_eq!(
            short.unit_deviance(&y, &mu),
            array![0.0, 0.25, 1.0],
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(short.unit_deviance_at(4.0, 3.0), 1.0, epsilon = 1e-12);
        assert_eq!(short.default_link().name(), "identity");
        assert_abs_diff_eq!(short.initialize_mu(&y), y, epsilon = 1e-12);
        assert!(short.is_valid_mu(&mu));
        assert_abs_diff_eq!(short.clamp_mu(&mu), mu, epsilon = 1e-12);
    }

    #[test]
    fn coordinate_descent_initialization_helpers_have_exact_contracts() {
        assert!(should_initialize_intercept_from_response(false, true));
        assert!(!should_initialize_intercept_from_response(true, true));
        assert!(!should_initialize_intercept_from_response(false, false));
        assert!(!should_initialize_intercept_from_response(true, false));

        let y = array![1.0, 3.0, 5.0];
        assert_abs_diff_eq!(
            intercept_start_from_response(&y, &GaussianFamily, &IdentityLink),
            3.0,
            epsilon = 1e-12
        );

        let x = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 1.0, -1.0])
            .expect("test setup should be valid");
        let coefficients = array![3.0, 4.0];
        let offset = array![0.5, -0.25];
        let expected_eta = array![11.5, -1.25];

        assert_abs_diff_eq!(
            linear_predictor_with_offset(x.view(), &coefficients, &offset),
            expected_eta,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            initialize_mu_from_coefficients(
                x.view(),
                &coefficients,
                &offset,
                &GaussianFamily,
                &IdentityLink
            ),
            expected_eta,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            initialize_mu_from_response(&y, &GaussianFamily),
            y,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            initialize_mu_from_response(&y, &InvalidInitGaussian),
            array![2.0, 3.0, 4.0],
            epsilon = 1e-12
        );
    }

    #[test]
    fn coordinate_descent_iteration_and_active_set_helpers_have_exact_edges() {
        assert!(should_continue_iteration(4, 5));
        assert!(!should_continue_iteration(5, 5));
        assert!(!should_continue_iteration(6, 5));

        let mut iteration = 2;
        advance_iteration(&mut iteration);
        assert_eq!(iteration, 3);
        let mut zero_iteration = 0;
        advance_iteration(&mut zero_iteration);
        assert_eq!(zero_iteration, 1);

        assert!(!should_use_active_subset(false, 4));
        assert!(should_use_active_subset(true, 4));
        assert!(!should_use_active_subset(true, 5));
        assert!(should_use_active_subset(true, 6));

        assert!(should_refresh_active_set(1));
        assert!(!should_refresh_active_set(4));
        assert!(should_refresh_active_set(5));
        assert!(should_refresh_active_set(10));

        let all_indices = vec![0, 1, 2, 3];
        let active_set = vec![0, 3];
        assert_eq!(
            coordinate_indices_for_iteration(true, 4, &all_indices, &active_set),
            active_set.as_slice()
        );
        assert_eq!(
            coordinate_indices_for_iteration(true, 5, &all_indices, &active_set),
            all_indices.as_slice()
        );
        assert_eq!(
            coordinate_indices_for_iteration(false, 4, &all_indices, &active_set),
            all_indices.as_slice()
        );

        let coefficients = array![10.0, ZERO_TOL, -2.0 * ZERO_TOL, 0.5];
        assert_eq!(
            active_set_from_coefficients(&coefficients, 1),
            vec![0, 2, 3]
        );
        let (refreshed, use_active_set) = refresh_active_set(&coefficients, 1);
        assert_eq!(refreshed, vec![0, 2, 3]);
        assert!(use_active_set);

        let dense = array![1.0, 2.0, -3.0];
        let (refreshed, use_active_set) = refresh_active_set(&dense, 1);
        assert_eq!(refreshed, vec![0, 1, 2]);
        assert!(!use_active_set);
    }

    #[test]
    fn coordinate_descent_update_helpers_pin_thresholds_and_penalty_math() {
        assert_abs_diff_eq!(
            coordinate_update_value(0, 1, 8.0, 2.0, 100.0, 100.0),
            4.0,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            coordinate_update_value(1, 1, 5.0, 3.0, 2.0, 1.0),
            0.75,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            coordinate_update_value(1, 1, 3.0 * ZERO_TOL, ZERO_TOL, 1.0 * ZERO_TOL, 0.0),
            2.0,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            coordinate_update_value(1, 1, 3.0, 0.5 * ZERO_TOL, 1.0, 0.0),
            0.0,
            epsilon = 1e-12
        );

        let mut coefficients = array![0.0, -2.0];
        let delta = apply_coordinate_update(&mut coefficients, 0, 0.0, COEFFICIENT_UPDATE_TOL);
        assert_abs_diff_eq!(delta, COEFFICIENT_UPDATE_TOL, epsilon = 0.0);
        assert_abs_diff_eq!(coefficients[0], 0.0, epsilon = 0.0);

        let delta = apply_coordinate_update(
            &mut coefficients,
            1,
            -2.0,
            -2.0 - 2.0 * COEFFICIENT_UPDATE_TOL,
        );
        assert!(delta > COEFFICIENT_UPDATE_TOL);
        assert_abs_diff_eq!(
            coefficients[1],
            -2.0 - 2.0 * COEFFICIENT_UPDATE_TOL,
            epsilon = 0.0
        );

        assert!(coordinate_descent_converged(0.5, 1.0));
        assert!(!coordinate_descent_converged(1.0, 1.0));
        assert!(!coordinate_descent_converged(1.5, 1.0));
    }

    #[test]
    fn coordinate_descent_projection_and_deviance_helpers_pin_exact_edges() {
        let mut coefficients = array![-2.0, -0.0, 3.0, -4.0, 5.0, -0.0];
        let neg_zero_bits = (-0.0_f64).to_bits();
        project_sign_constraints(&mut coefficients, &[0, 1, 6], &[2, 5, 6]);

        assert_abs_diff_eq!(coefficients[0], 0.0, epsilon = 0.0);
        assert_eq!(coefficients[1].to_bits(), neg_zero_bits);
        assert_abs_diff_eq!(coefficients[2], 0.0, epsilon = 0.0);
        assert_abs_diff_eq!(coefficients[3], -4.0, epsilon = 0.0);
        assert_abs_diff_eq!(coefficients[4], 5.0, epsilon = 0.0);
        assert_eq!(coefficients[5].to_bits(), neg_zero_bits);

        let (abs_change, rel_change) = deviance_changes(8.0, 6.0);
        assert_abs_diff_eq!(abs_change, 2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(rel_change, 0.25, epsilon = 1e-12);
        let (abs_change, rel_change) = deviance_changes(0.0, 2.0);
        assert_abs_diff_eq!(abs_change, 2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(rel_change, 2.0, epsilon = 1e-12);
        let (abs_change, rel_change) = deviance_changes(ZERO_TOL, 0.0);
        assert_abs_diff_eq!(abs_change, ZERO_TOL, epsilon = 0.0);
        assert_abs_diff_eq!(rel_change, ZERO_TOL, epsilon = 0.0);

        assert_eq!(
            count_nonzero_penalized(&array![99.0, ZERO_TOL, -2.0 * ZERO_TOL, 0.0], 1),
            1
        );
        assert_eq!(
            count_nonzero_penalized(&array![99.0, 2.0 * ZERO_TOL, -3.0 * ZERO_TOL, 0.5], 1),
            3
        );
        assert!(outer_deviance_converged(0.009, 10.0, 1.0, 0.01));
        assert!(!outer_deviance_converged(
            0.01,
            ZERO_TOL,
            ZERO_TOL / 2.0,
            0.01
        ));
        assert!(!outer_deviance_converged(
            0.1,
            ZERO_TOL / 2.0,
            ZERO_TOL,
            0.01
        ));
        assert!(outer_deviance_converged(
            0.1,
            ZERO_TOL / 2.0,
            ZERO_TOL / 2.0,
            0.01
        ));
        assert!(!outer_deviance_converged(
            0.1,
            ZERO_TOL / 2.0,
            2.0 * ZERO_TOL,
            0.01
        ));
        assert!(!outer_deviance_converged(
            0.1,
            2.0 * ZERO_TOL,
            ZERO_TOL / 2.0,
            0.01
        ));
    }

    #[test]
    fn penalized_covariance_helpers_pin_weights_and_active_threshold() {
        let irls_weights = array![2.0, 4.0, 0.5];
        let prior_weights = array![0.5, 3.0, 8.0];
        assert_abs_diff_eq!(
            combined_covariance_weights(&irls_weights, &prior_weights),
            array![1.0, 12.0, 4.0],
            epsilon = 1e-12
        );

        let coefficients = array![0.0, ZERO_TOL, -2.0 * ZERO_TOL, 0.0];
        assert_eq!(active_set_from_coefficients(&coefficients, 0), vec![2]);
        assert_eq!(
            active_set_from_coefficients(&coefficients, 2),
            vec![0, 1, 2]
        );
    }

    #[test]
    fn test_lasso_produces_sparse_solution() {
        // Lasso should set some coefficients to exactly zero
        let x = Array2::from_shape_vec(
            (10, 4),
            vec![
                1.0, 1.0, 0.1, 0.2, 1.0, 2.0, 0.2, 0.1, 1.0, 3.0, 0.3, 0.3, 1.0, 4.0, 0.1, 0.2,
                1.0, 5.0, 0.2, 0.1, 1.0, 6.0, 0.3, 0.2, 1.0, 7.0, 0.1, 0.3, 1.0, 8.0, 0.2, 0.1,
                1.0, 9.0, 0.3, 0.2, 1.0, 10.0, 0.1, 0.1,
            ],
        )
        .expect("test setup should be valid");

        // y strongly related to x1, weakly to x2, x3
        let y = array![5.0, 8.0, 11.0, 14.0, 17.0, 20.0, 23.0, 26.0, 29.0, 32.0];

        let family = GaussianFamily;
        let link = IdentityLink;
        let irls_config = IRLSConfig::default();

        // Strong Lasso penalty
        let reg_config = RegularizationConfig::lasso(5.0);
        let result = fit_glm_coordinate_descent(
            &y,
            x.view(),
            &family,
            &link,
            &irls_config,
            &reg_config,
            None,
            None,
            None,
            false,
        )
        .expect("test setup should be valid");

        assert!(result.converged);

        // The weak predictors (columns 2, 3) should be shrunk toward or to zero
        // The strong predictor (column 1) should remain non-zero
        assert!(
            result.coefficients[1].abs() > 0.5,
            "Strong predictor should be non-zero"
        );

        // At least one of the weak predictors should be near zero
        let weak_coefs: Vec<f64> = vec![result.coefficients[2], result.coefficients[3]];
        let has_near_zero = weak_coefs.iter().any(|&c| c.abs() < 0.1);
        assert!(
            has_near_zero,
            "Lasso should shrink weak predictors toward zero"
        );
    }

    #[test]
    fn test_coordinate_descent_invalid_family_initializer_uses_safe_mu() {
        let x = Array2::from_shape_vec(
            (5, 2),
            vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0],
        )
        .expect("test setup should be valid");
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];
        let irls_config = IRLSConfig {
            max_iterations: 5,
            ..IRLSConfig::default()
        };
        let reg_config = RegularizationConfig::lasso(0.01);

        let result = fit_glm_coordinate_descent(
            &y,
            x.view(),
            &InvalidInitGaussian,
            &IdentityLink,
            &irls_config,
            &reg_config,
            None,
            None,
            None,
            false,
        )
        .expect("safe initializer fallback should allow coordinate descent to run");

        assert!(result.fitted_values.iter().all(|value| value.is_finite()));
        assert!(result
            .linear_predictor
            .iter()
            .all(|value| value.is_finite()));
        assert!(result.coefficients.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn test_coordinate_descent_propagates_malformed_weight_dimensions() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, -1.0, 1.0, 0.0, 1.0, 1.0])
            .expect("test setup should be valid");
        let y = array![0.5, 1.0, 1.5];
        let reg_config = RegularizationConfig::lasso(0.05);

        let err = fit_glm_coordinate_descent(
            &y,
            x.view(),
            &ShortVarianceGaussian,
            &IdentityLink,
            &IRLSConfig::default(),
            &reg_config,
            None,
            None,
            None,
            false,
        )
        .expect_err("malformed family variance must be reported, not panicked");

        assert!(matches!(
            err,
            crate::error::RustyStatsError::DimensionMismatch {
                expected: 3,
                got: 2,
                ref context
            } if context == "variance length vs y length"
        ));
    }

    #[test]
    fn test_lasso_vs_unpenalized() {
        // Lasso with small lambda should give similar results to unpenalized
        let x = Array2::from_shape_vec(
            (5, 2),
            vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0],
        )
        .expect("test setup should be valid");
        let y = array![5.0, 8.0, 11.0, 14.0, 17.0];

        let family = GaussianFamily;
        let link = IdentityLink;
        let irls_config = IRLSConfig {
            max_iterations: 50, // More iterations for small penalty
            ..IRLSConfig::default()
        };

        // Very small Lasso penalty
        let reg_config = RegularizationConfig::lasso(0.001);
        let result = fit_glm_coordinate_descent(
            &y,
            x.view(),
            &family,
            &link,
            &irls_config,
            &reg_config,
            None,
            None,
            None,
            false,
        )
        .expect("test setup should be valid");

        // With Gaussian + identity link, should converge quickly or reach good solution
        // Coefficients should be close to OLS (intercept ~2, slope ~3)
        assert!(
            (result.coefficients[0] - 2.0).abs() < 1.0,
            "Intercept: {}",
            result.coefficients[0]
        );
        assert!(
            (result.coefficients[1] - 3.0).abs() < 0.5,
            "Slope: {}",
            result.coefficients[1]
        );
    }

    #[test]
    fn test_elastic_net() {
        // Elastic Net should work (combination of L1 and L2)
        let x = Array2::from_shape_vec(
            (6, 3),
            vec![
                1.0, 1.0, 1.1, // x2 and x3 are correlated
                1.0, 2.0, 2.2, 1.0, 3.0, 3.1, 1.0, 4.0, 4.3, 1.0, 5.0, 5.2, 1.0, 6.0, 6.1,
            ],
        )
        .expect("test setup should be valid");
        let y = array![5.0, 8.0, 11.0, 14.0, 17.0, 20.0];

        let family = GaussianFamily;
        let link = IdentityLink;
        let irls_config = IRLSConfig {
            max_iterations: 50,
            ..IRLSConfig::default()
        };

        // Elastic Net: 50% L1, 50% L2
        let reg_config = RegularizationConfig::elastic_net(1.0, 0.5);
        let result = fit_glm_coordinate_descent(
            &y,
            x.view(),
            &family,
            &link,
            &irls_config,
            &reg_config,
            None,
            None,
            None,
            false,
        )
        .expect("test setup should be valid");

        // Should produce reasonable fitted values even if not converged
        assert!(result.fitted_values.iter().all(|&x| x.is_finite()));

        // Elastic net should spread weight across correlated predictors
        // (unlike pure Lasso which often picks just one)
    }

    #[test]
    fn test_lasso_poisson() {
        // Lasso should work with Poisson family
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0],
        )
        .expect("test setup should be valid");
        let y = array![2.0, 3.0, 4.0, 6.0, 8.0, 12.0];

        let family = PoissonFamily;
        let link = LogLink;
        let irls_config = IRLSConfig::default();

        let reg_config = RegularizationConfig::lasso(0.1);
        let result = fit_glm_coordinate_descent(
            &y,
            x.view(),
            &family,
            &link,
            &irls_config,
            &reg_config,
            None,
            None,
            None,
            false,
        )
        .expect("test setup should be valid");

        assert!(result.converged);
        assert!(result.fitted_values.iter().all(|&x| x > 0.0));
    }

    #[test]
    fn test_lasso_intercept_not_penalized() {
        // Intercept should never be zero even with strong penalty
        let x = Array2::from_shape_vec(
            (5, 2),
            vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0],
        )
        .expect("test setup should be valid");
        let y = array![10.0, 10.0, 10.0, 10.0, 10.0]; // Constant y

        let family = GaussianFamily;
        let link = IdentityLink;
        let irls_config = IRLSConfig::default();

        // Very strong Lasso penalty
        let reg_config = RegularizationConfig::lasso(100.0);
        let result = fit_glm_coordinate_descent(
            &y,
            x.view(),
            &family,
            &link,
            &irls_config,
            &reg_config,
            None,
            None,
            None,
            false,
        )
        .expect("test setup should be valid");

        assert!(result.converged);

        // Intercept should be around 10 (mean of y)
        assert!(
            (result.coefficients[0] - 10.0).abs() < 1.0,
            "Intercept should be ~10: {}",
            result.coefficients[0]
        );

        // Slope should be zero (no relationship + strong penalty)
        assert!(
            result.coefficients[1].abs() < 0.01,
            "Slope should be ~0: {}",
            result.coefficients[1]
        );
    }

    #[test]
    fn test_lasso_covariance_uses_active_set_when_full_design_singular() {
        let x = Array2::from_shape_vec(
            (8, 3),
            vec![
                1.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 3.0, 0.0, 1.0, 4.0, 0.0, 1.0, 5.0, 0.0, 1.0,
                6.0, 0.0, 1.0, 7.0, 0.0, 1.0, 8.0, 0.0,
            ],
        )
        .expect("test setup should be valid");
        let y = array![5.0, 8.0, 11.0, 14.0, 17.0, 20.0, 23.0, 26.0];

        let family = GaussianFamily;
        let link = IdentityLink;
        let irls_config = IRLSConfig {
            max_iterations: 50,
            ..IRLSConfig::default()
        };
        let reg_config = RegularizationConfig::lasso(0.01);
        let result = fit_glm_coordinate_descent(
            &y,
            x.view(),
            &family,
            &link,
            &irls_config,
            &reg_config,
            None,
            None,
            None,
            false,
        )
        .expect("test setup should be valid");

        assert!(result.coefficients[1].abs() > 0.5);
        assert!(result.covariance_unscaled[[0, 0]].is_finite());
        assert!(result.covariance_unscaled[[1, 1]].is_finite());
        assert!(result.covariance_unscaled[[0, 0]] > 0.0);
        assert!(result.covariance_unscaled[[1, 1]] > 0.0);
        assert_abs_diff_eq!(result.covariance_unscaled[[2, 2]], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_coordinate_descent_warm_start_offset_weights_and_skip_covariance_contracts() {
        let x = Array2::from_shape_vec(
            (6, 3),
            vec![
                1.0, 0.0, 1.0, 1.0, 1.0, 0.5, 1.0, 2.0, 1.5, 1.0, 3.0, 2.0, 1.0, 4.0, 2.5, 1.0,
                5.0, 3.0,
            ],
        )
        .expect("test setup should be valid");
        let y = array![3.0, 4.2, 5.9, 7.4, 9.1, 10.8];
        let offset = array![0.0, 0.1, -0.1, 0.2, -0.2, 0.0];
        let weights = array![1.0, 0.5, 1.5, 2.0, 0.75, 1.25];
        let init = array![1.0, 0.1, -0.1];

        let family = GaussianFamily;
        let link = IdentityLink;
        let irls_config = IRLSConfig {
            max_iterations: 25,
            ..IRLSConfig::default()
        };
        let reg_config = RegularizationConfig::elastic_net(0.05, 0.5);
        let result = fit_glm_coordinate_descent(
            &y,
            x.view(),
            &family,
            &link,
            &irls_config,
            &reg_config,
            Some(&offset),
            Some(&weights),
            Some(&init),
            true,
        )
        .expect("test setup should be valid");

        assert_eq!(result.offset, offset);
        assert_eq!(result.prior_weights, weights);
        assert!(result.coefficients.iter().all(|v| v.is_finite()));
        assert!(result.covariance_unscaled.iter().all(|&v| v == 0.0));
        assert!(result.warnings.is_empty());
    }

    #[test]
    fn test_coordinate_descent_warm_start_mismatch_and_cd_nonconvergence_warnings() {
        let x = Array2::from_shape_vec(
            (5, 2),
            vec![1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0],
        )
        .expect("test setup should be valid");
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let bad_init = array![99.0];
        let family = GaussianFamily;
        let link = IdentityLink;
        let irls_config = IRLSConfig {
            max_iterations: 1,
            verbose: true,
            ..IRLSConfig::default()
        };
        let reg_config = RegularizationConfig {
            max_cd_iterations: 0,
            ..RegularizationConfig::lasso(0.1)
        };

        let result = fit_glm_coordinate_descent(
            &y,
            x.view(),
            &family,
            &link,
            &irls_config,
            &reg_config,
            None,
            None,
            Some(&bad_init),
            true,
        )
        .expect("test setup should be valid");

        assert!(result
            .warnings
            .iter()
            .any(|w| w.contains("Warm-start coefficient dimension mismatch")));
        assert!(result
            .warnings
            .iter()
            .any(|w| w.contains("Coordinate descent did not converge")));
        assert_eq!(result.solver_status, "converged");
    }

    #[test]
    fn test_coordinate_descent_sign_constraints_project_coefficients() {
        let x = Array2::from_shape_vec(
            (8, 3),
            vec![
                1.0, -3.0, 2.0, 1.0, -2.0, -1.0, 1.0, -1.0, 1.0, 1.0, 0.0, -2.0, 1.0, 1.0, 3.0,
                1.0, 2.0, -3.0, 1.0, 3.0, 4.0, 1.0, 4.0, -4.0,
            ],
        )
        .expect("test setup should be valid");
        let y = array![11.0, 6.0, 6.0, 1.0, 11.0, -4.0, 17.0, -10.0];
        let family = GaussianFamily;
        let link = IdentityLink;
        let irls_config = IRLSConfig {
            max_iterations: 20,
            nonneg_indices: vec![1],
            nonpos_indices: vec![2],
            ..IRLSConfig::default()
        };
        let reg_config = RegularizationConfig::lasso(0.001);

        let result = fit_glm_coordinate_descent(
            &y,
            x.view(),
            &family,
            &link,
            &irls_config,
            &reg_config,
            None,
            None,
            None,
            true,
        )
        .expect("test setup should be valid");

        assert!(result.coefficients[1] >= -1e-12);
        assert!(result.coefficients[2] <= 1e-12);
    }

    #[test]
    fn test_penalized_covariance_empty_and_singular_active_set_contracts() {
        let x = Array2::from_shape_vec(
            (4, 3),
            vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0],
        )
        .expect("test setup should be valid");
        let irls_weights = array![1.0, 1.0, 1.0, 1.0];
        let prior_weights = array![1.0, 0.5, 2.0, 1.5];

        let inactive = array![0.0, 0.0, 0.0];
        let cov =
            compute_penalized_covariance(x.view(), &irls_weights, &prior_weights, &inactive, 0);
        assert_eq!(cov.dim(), (3, 3));
        assert!(cov.iter().all(|&v| v == 0.0));

        let active_singular = array![1.0, -2.0, 3.0];
        let cov = compute_penalized_covariance(
            x.view(),
            &irls_weights,
            &prior_weights,
            &active_singular,
            0,
        );
        assert!(cov[[0, 0]].is_nan());
        assert!(cov[[1, 1]].is_nan());
        assert!(cov[[2, 2]].is_nan());
    }
}
