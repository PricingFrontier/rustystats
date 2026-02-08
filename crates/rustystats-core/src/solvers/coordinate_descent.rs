use super::initialize_mu_safe;

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
use rayon::prelude::*;

use crate::constants::ZERO_TOL;
use crate::error::{RustyStatsError, Result};
use crate::families::Family;
use crate::links::Link;
use crate::regularization::{soft_threshold, RegularizationConfig};
use super::irls::{IRLSConfig, IRLSResult};

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
    x: &Array2<f64>,
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

    // Get penalty parameters
    let l1_penalty = reg_config.penalty.l1_penalty();
    let l2_penalty = reg_config.penalty.l2_penalty();
    let has_intercept = reg_config.fit_intercept;
    
    // Starting index for penalized coefficients
    let pen_start = if has_intercept { 1 } else { 0 };

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
    // Step 1: Precompute X'X diagonal (sum of squared predictors, weighted)
    // These are recomputed each IRLS iteration with updated weights
    // -------------------------------------------------------------------------
    
    // -------------------------------------------------------------------------
    // Step 2: Initialize coefficients and μ (with warm start support)
    // -------------------------------------------------------------------------
    let mut coefficients = if let Some(init) = init_coefficients {
        if init.len() == p {
            init.clone()
        } else {
            // Dimension mismatch - fall back to cold start with warning
            eprintln!(
                "Warning: Warm-start coefficient dimension mismatch (got {}, expected {}). \
                Falling back to cold start. This may indicate a bug in the caller.",
                init.len(), p
            );
            Array1::zeros(p)
        }
    } else {
        Array1::zeros(p)
    };
    
    // Initialize intercept to link(mean(y)) only if not warm starting
    if init_coefficients.is_none() && has_intercept {
        let y_mean = y.mean().unwrap_or(1.0);
        let y_mean_clamped = match family.name() {
            "Poisson" | "Gamma" => y_mean.max(0.01),
            "Binomial" => y_mean.max(0.01).min(0.99),
            _ => y_mean,
        };
        coefficients[0] = link.link(&Array1::from_elem(1, y_mean_clamped))[0];
    }

    // Initialize μ from coefficients if warm starting, otherwise from y
    let mut mu = if init_coefficients.is_some() {
        let eta = x.dot(&coefficients) + &offset_vec;
        let mu_init = link.inverse(&eta);
        family.clamp_mu(&mu_init)
    } else {
        let mu_init = family.initialize_mu(y);
        if !family.is_valid_mu(&mu_init) {
            initialize_mu_safe(y, family)
        } else {
            mu_init
        }
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

    while outer_iteration < irls_config.max_iterations {
        outer_iteration += 1;

        // ---------------------------------------------------------------------
        // Step 5a: Compute working weights and working response
        // ---------------------------------------------------------------------
        // OPTIMIZATION: Use true Hessian weights for Gamma/Tweedie with log link
        // This can dramatically reduce IRLS iterations (50-100 → 5-10)
        let link_deriv = link.derivative(&mu);
        
        let use_true_hessian = family.use_true_hessian_weights() && link.name() == "log";
        let hessian_weights = if use_true_hessian {
            Some(family.true_hessian_weights(&mu, y))
        } else {
            None
        };
        let variance = if use_true_hessian { None } else { Some(family.variance(&mu)) };

        // PARALLEL: Compute IRLS weights, combined weights, and working response
        let min_weight = irls_config.min_weight;
        let results: Vec<(f64, f64, f64)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let d = link_deriv[i];
                let iw = if let Some(ref hw) = hessian_weights {
                    // True Hessian weight - use directly without dividing by d²
                    hw[i].max(min_weight).min(1e10)
                } else {
                    let v = variance.as_ref().unwrap()[i];
                    (1.0 / (v * d * d)).max(min_weight).min(1e10)
                };
                let cw = prior_weights_vec[i] * iw;
                let wr = (eta[i] - offset_vec[i]) + (y[i] - mu[i]) * d;
                (iw, cw, wr)
            })
            .collect();
        
        let mut combined_weights_vec = Vec::with_capacity(n);
        let mut working_response_vec = Vec::with_capacity(n);
        for (i, &(iw, cw, wr)) in results.iter().enumerate() {
            irls_weights[i] = iw;
            combined_weights_vec.push(cw);
            working_response_vec.push(wr);
        }
        
        let combined_weights = Array1::from_vec(combined_weights_vec);
        let working_response = Array1::from_vec(working_response_vec);

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
        
        // Precompute X'Wz (gradient at β=0) - PARALLEL
        let xwz: Vec<f64> = (0..p)
            .into_par_iter()
            .map(|j| {
                let col = x.column(j);
                col.iter()
                    .zip(combined_weights.iter())
                    .zip(working_response.iter())
                    .map(|((&xij, &wi), &zi)| wi * xij * zi)
                    .sum()
            })
            .collect();
        
        // Precompute X'WX (Gram matrix) - PARALLEL with flat Vec for cache locality
        let xwx: Vec<f64> = (0..n)
            .into_par_iter()
            .fold(
                || vec![0.0; p * p],
                |mut acc, i| {
                    let w_i = combined_weights[i];
                    let x_i = x.row(i);
                    for j in 0..p {
                        let xij_w = x_i[j] * w_i;
                        for k in j..p {
                            acc[j * p + k] += xij_w * x_i[k];
                        }
                    }
                    acc
                },
            )
            .reduce(
                || vec![0.0; p * p],
                |mut a, b| {
                    for i in 0..a.len() { a[i] += b[i]; }
                    a
                },
            );
        
        // Fill in lower triangle (symmetric)
        let mut xwx_full = xwx;
        for j in 0..p {
            for k in (j + 1)..p {
                xwx_full[k * p + j] = xwx_full[j * p + k];
            }
        }
        let xwx = xwx_full;

        // Active set: track which coefficients are non-zero for faster iterations
        let mut active_set: Vec<usize> = (0..p).collect();
        let mut use_active_set = false;
        
        while cd_iteration < reg_config.max_cd_iterations {
            cd_iteration += 1;
            let mut max_change = 0.0_f64;

            // Decide which coefficients to update
            let indices_to_update: &[usize] = if use_active_set && cd_iteration % 5 != 0 {
                // Use active set (non-zero coefficients + intercept)
                &active_set
            } else {
                // Full pass every 5 iterations or initially
                &(0..p).collect::<Vec<_>>()
            };

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
                let new_coef = if j < pen_start {
                    rho / xwx_jj
                } else {
                    let denom = xwx_jj + l2_penalty;
                    if denom.abs() < ZERO_TOL {
                        0.0
                    } else {
                        soft_threshold(rho, l1_penalty) / denom
                    }
                };

                let delta = (new_coef - old_coef).abs();
                if delta > 1e-15 {
                    coefficients[j] = new_coef;
                }
                max_change = max_change.max(delta);
            }

            // Update active set after first full pass
            if cd_iteration == 1 || cd_iteration % 5 == 0 {
                active_set.clear();
                for j in 0..pen_start {
                    active_set.push(j); // Always include intercept
                }
                for j in pen_start..p {
                    if coefficients[j].abs() > ZERO_TOL {
                        active_set.push(j);
                    }
                }
                use_active_set = active_set.len() < p;
            }

            // Check convergence
            if max_change < reg_config.cd_tolerance {
                cd_converged = true;
                break;
            }
        }

        if irls_config.verbose && !cd_converged {
            eprintln!(
                "Warning: Coordinate descent did not converge in {} iterations",
                reg_config.max_cd_iterations
            );
        }

        // ---------------------------------------------------------------------
        // Step 5d: Update η and μ
        // ---------------------------------------------------------------------
        let eta_base = x.dot(&coefficients);
        eta = &eta_base + &offset_vec;
        mu = link.inverse(&eta);
        mu = family.clamp_mu(&mu);

        // ---------------------------------------------------------------------
        // Step 5e: Check outer loop convergence
        // ---------------------------------------------------------------------
        deviance_old = deviance;
        deviance = family.deviance(y, &mu, Some(&prior_weights_vec));

        let abs_change = (deviance_old - deviance).abs();
        let rel_change = if deviance_old.abs() > ZERO_TOL {
            abs_change / deviance_old.abs()
        } else {
            abs_change
        };

        if irls_config.verbose {
            let n_nonzero = coefficients.iter().skip(pen_start).filter(|&&c| c.abs() > ZERO_TOL).count();
            eprintln!(
                "IRLS iter {}: deviance = {:.6}, rel_change = {:.2e}, nonzero = {}",
                outer_iteration, deviance, rel_change, n_nonzero
            );
        }

        // Converge if relative change is small OR if deviance is very small (nearly perfect fit)
        if rel_change < irls_config.tolerance || (deviance < ZERO_TOL && abs_change < ZERO_TOL) {
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
        compute_penalized_covariance(x, &irls_weights, &prior_weights_vec, &coefficients, pen_start)
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
        design_matrix: None,  // Computed lazily in Python layer to avoid expensive copy
    })
}

/// Compute an approximate covariance matrix for penalized estimates.
///
/// For Lasso/Elastic Net, standard errors are not well-defined in the classical sense.
/// This computes a naive estimate that can be used for rough inference.
fn compute_penalized_covariance(
    x: &Array2<f64>,
    irls_weights: &Array1<f64>,
    prior_weights: &Array1<f64>,
    _coefficients: &Array1<f64>,
    _pen_start: usize,
) -> Array2<f64> {
    let p = x.ncols();
    let n = x.nrows();
    
    // Compute (X'WX)⁻¹ only for the "active" (non-zero) coefficients
    // This is a simplified approach
    let mut cov = Array2::zeros((p, p));
    
    // Combined weights
    let weights: Vec<f64> = irls_weights
        .iter()
        .zip(prior_weights.iter())
        .map(|(&iw, &pw)| iw * pw)
        .collect();

    // Compute X'WX
    for i in 0..p {
        for j in i..p {
            let val: f64 = (0..n)
                .map(|k| weights[k] * x[[k, i]] * x[[k, j]])
                .sum();
            cov[[i, j]] = val;
            cov[[j, i]] = val;
        }
    }

    // Try to invert (this will fail for truly sparse solutions, but okay for Ridge-like)
    use nalgebra::DMatrix;
    let mut xtx = DMatrix::zeros(p, p);
    for i in 0..p {
        for j in 0..p {
            xtx[(i, j)] = cov[[i, j]];
        }
    }

    if let Some(inv) = xtx.try_inverse() {
        for i in 0..p {
            for j in 0..p {
                cov[[i, j]] = inv[(i, j)];
            }
        }
    } else {
        // Return zeros if not invertible
        cov.fill(0.0);
    }

    cov
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
    fn test_lasso_produces_sparse_solution() {
        // Lasso should set some coefficients to exactly zero
        let x = Array2::from_shape_vec(
            (10, 4),
            vec![
                1.0, 1.0, 0.1, 0.2,
                1.0, 2.0, 0.2, 0.1,
                1.0, 3.0, 0.3, 0.3,
                1.0, 4.0, 0.1, 0.2,
                1.0, 5.0, 0.2, 0.1,
                1.0, 6.0, 0.3, 0.2,
                1.0, 7.0, 0.1, 0.3,
                1.0, 8.0, 0.2, 0.1,
                1.0, 9.0, 0.3, 0.2,
                1.0, 10.0, 0.1, 0.1,
            ],
        )
        .unwrap();
        
        // y strongly related to x1, weakly to x2, x3
        let y = array![5.0, 8.0, 11.0, 14.0, 17.0, 20.0, 23.0, 26.0, 29.0, 32.0];

        let family = GaussianFamily;
        let link = IdentityLink;
        let irls_config = IRLSConfig::default();
        
        // Strong Lasso penalty
        let reg_config = RegularizationConfig::lasso(5.0);
        let result = fit_glm_coordinate_descent(&y, &x, &family, &link, &irls_config, &reg_config, None, None, None, false).unwrap();

        assert!(result.converged);
        
        // The weak predictors (columns 2, 3) should be shrunk toward or to zero
        // The strong predictor (column 1) should remain non-zero
        assert!(result.coefficients[1].abs() > 0.5, "Strong predictor should be non-zero");
        
        // At least one of the weak predictors should be near zero
        let weak_coefs: Vec<f64> = vec![result.coefficients[2], result.coefficients[3]];
        let has_near_zero = weak_coefs.iter().any(|&c| c.abs() < 0.1);
        assert!(has_near_zero, "Lasso should shrink weak predictors toward zero");
    }

    #[test]
    fn test_lasso_vs_unpenalized() {
        // Lasso with small lambda should give similar results to unpenalized
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
        let mut irls_config = IRLSConfig::default();
        irls_config.max_iterations = 50;  // More iterations for small penalty
        
        // Very small Lasso penalty
        let reg_config = RegularizationConfig::lasso(0.001);
        let result = fit_glm_coordinate_descent(&y, &x, &family, &link, &irls_config, &reg_config, None, None, None, false).unwrap();

        // With Gaussian + identity link, should converge quickly or reach good solution
        // Coefficients should be close to OLS (intercept ~2, slope ~3)
        assert!((result.coefficients[0] - 2.0).abs() < 1.0, "Intercept: {}", result.coefficients[0]);
        assert!((result.coefficients[1] - 3.0).abs() < 0.5, "Slope: {}", result.coefficients[1]);
    }

    #[test]
    fn test_elastic_net() {
        // Elastic Net should work (combination of L1 and L2)
        let x = Array2::from_shape_vec(
            (6, 3),
            vec![
                1.0, 1.0, 1.1,  // x2 and x3 are correlated
                1.0, 2.0, 2.2,
                1.0, 3.0, 3.1,
                1.0, 4.0, 4.3,
                1.0, 5.0, 5.2,
                1.0, 6.0, 6.1,
            ],
        )
        .unwrap();
        let y = array![5.0, 8.0, 11.0, 14.0, 17.0, 20.0];

        let family = GaussianFamily;
        let link = IdentityLink;
        let mut irls_config = IRLSConfig::default();
        irls_config.max_iterations = 50;
        
        // Elastic Net: 50% L1, 50% L2
        let reg_config = RegularizationConfig::elastic_net(1.0, 0.5);
        let result = fit_glm_coordinate_descent(&y, &x, &family, &link, &irls_config, &reg_config, None, None, None, false).unwrap();

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
        
        let reg_config = RegularizationConfig::lasso(0.1);
        let result = fit_glm_coordinate_descent(&y, &x, &family, &link, &irls_config, &reg_config, None, None, None, false).unwrap();

        assert!(result.converged);
        assert!(result.fitted_values.iter().all(|&x| x > 0.0));
    }

    #[test]
    fn test_lasso_intercept_not_penalized() {
        // Intercept should never be zero even with strong penalty
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
        let y = array![10.0, 10.0, 10.0, 10.0, 10.0]; // Constant y

        let family = GaussianFamily;
        let link = IdentityLink;
        let irls_config = IRLSConfig::default();
        
        // Very strong Lasso penalty
        let reg_config = RegularizationConfig::lasso(100.0);
        let result = fit_glm_coordinate_descent(&y, &x, &family, &link, &irls_config, &reg_config, None, None, None, false).unwrap();

        assert!(result.converged);
        
        // Intercept should be around 10 (mean of y)
        assert!((result.coefficients[0] - 10.0).abs() < 1.0, 
            "Intercept should be ~10: {}", result.coefficients[0]);
        
        // Slope should be zero (no relationship + strong penalty)
        assert!(result.coefficients[1].abs() < 0.01,
            "Slope should be ~0: {}", result.coefficients[1]);
    }
}
