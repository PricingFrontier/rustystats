// =============================================================================
// Loss Functions for Model Evaluation
// =============================================================================
//
// This module provides loss functions for evaluating model performance.
// Each family has a default loss function that matches its deviance.
//
// Loss functions are used for:
// - Overall model quality assessment
// - Per-factor performance comparison
// - Cross-validation scoring
//
// =============================================================================

#[cfg(test)]
use crate::constants::MU_MIN_POSITIVE;
use crate::families::{Family, GammaFamily, NegativeBinomialFamily, PoissonFamily, TweedieFamily};
use ndarray::Array1;

/// Mean Squared Error (MSE)
///
/// MSE = mean((y - μ)²)
///
/// Default loss for Gaussian family.
pub fn mse(y: &Array1<f64>, mu: &Array1<f64>, weights: Option<&Array1<f64>>) -> f64 {
    let n = y.len();
    if n == 0 {
        return 0.0;
    }

    match weights {
        Some(w) => {
            let sum_w: f64 = w.sum();
            if sum_w == 0.0 {
                return 0.0;
            }
            let weighted_sum: f64 = y
                .iter()
                .zip(mu.iter())
                .zip(w.iter())
                .map(|((&yi, &mui), &wi)| wi * (yi - mui).powi(2))
                .sum();
            weighted_sum / sum_w
        }
        None => {
            let sum: f64 = y
                .iter()
                .zip(mu.iter())
                .map(|(&yi, &mui)| (yi - mui).powi(2))
                .sum();
            sum / n as f64
        }
    }
}

/// Root Mean Squared Error (RMSE)
pub fn rmse(y: &Array1<f64>, mu: &Array1<f64>, weights: Option<&Array1<f64>>) -> f64 {
    mse(y, mu, weights).sqrt()
}

/// Mean Absolute Error (MAE)
pub fn mae(y: &Array1<f64>, mu: &Array1<f64>, weights: Option<&Array1<f64>>) -> f64 {
    let n = y.len();
    if n == 0 {
        return 0.0;
    }

    match weights {
        Some(w) => {
            let sum_w: f64 = w.sum();
            if sum_w == 0.0 {
                return 0.0;
            }
            let weighted_sum: f64 = y
                .iter()
                .zip(mu.iter())
                .zip(w.iter())
                .map(|((&yi, &mui), &wi)| wi * (yi - mui).abs())
                .sum();
            weighted_sum / sum_w
        }
        None => {
            let sum: f64 = y
                .iter()
                .zip(mu.iter())
                .map(|(&yi, &mui)| (yi - mui).abs())
                .sum();
            sum / n as f64
        }
    }
}

/// Poisson Deviance Loss (mean unit deviance)
///
/// loss = 2 * mean(y * log(y/μ) - (y - μ))
///
/// Default loss for Poisson family.
///
/// # Implementation notes
/// Single-pass `fold` accumulates the (weighted) deviance numerator without
/// materialising a Vec of unit deviances (saves ~8 MB at n=1M). Per-row
/// arithmetic is delegated to `PoissonFamily::unit_deviance_at` so this
/// function cannot drift from the canonical `Family` trait implementation.
pub fn poisson_deviance_loss(
    y: &Array1<f64>,
    mu: &Array1<f64>,
    weights: Option<&Array1<f64>>,
) -> f64 {
    let n = y.len();
    if n == 0 {
        return 0.0;
    }

    let family = PoissonFamily;

    match weights {
        Some(w) => {
            let sum_w: f64 = w.sum();
            if sum_w == 0.0 {
                return 0.0;
            }
            let weighted_sum: f64 = y
                .iter()
                .zip(mu.iter())
                .zip(w.iter())
                .map(|((&yi, &mui), &wi)| wi * family.unit_deviance_at(yi, mui))
                .sum();
            weighted_sum / sum_w
        }
        None => {
            let sum: f64 = y
                .iter()
                .zip(mu.iter())
                .map(|(&yi, &mui)| family.unit_deviance_at(yi, mui))
                .sum();
            sum / n as f64
        }
    }
}

/// Gamma Deviance Loss (mean unit deviance)
///
/// loss = 2 * mean((y - μ)/μ - log(y/μ))
///
/// Default loss for Gamma family.
///
/// Single-pass `fold` delegating to `GammaFamily::unit_deviance_at`. Keeping
/// the arithmetic in the `Family` trait prevents drift between this path and
/// the vectorized `Family::unit_deviance` used by IRLS.
pub fn gamma_deviance_loss(
    y: &Array1<f64>,
    mu: &Array1<f64>,
    weights: Option<&Array1<f64>>,
) -> f64 {
    let n = y.len();
    if n == 0 {
        return 0.0;
    }

    let family = GammaFamily;

    match weights {
        Some(w) => {
            let sum_w: f64 = w.sum();
            if sum_w == 0.0 {
                return 0.0;
            }
            let weighted_sum: f64 = y
                .iter()
                .zip(mu.iter())
                .zip(w.iter())
                .map(|((&yi, &mui), &wi)| wi * family.unit_deviance_at(yi, mui))
                .sum();
            weighted_sum / sum_w
        }
        None => {
            let sum: f64 = y
                .iter()
                .zip(mu.iter())
                .map(|(&yi, &mui)| family.unit_deviance_at(yi, mui))
                .sum();
            sum / n as f64
        }
    }
}

/// Binomial Deviance Loss (Log Loss / Cross-Entropy)
///
/// loss = -mean(y * log(μ) + (1-y) * log(1-μ))
///
/// Default loss for Binomial family.
///
/// Single-pass `fold` (see `poisson_deviance_loss` for rationale).
pub fn log_loss(y: &Array1<f64>, mu: &Array1<f64>, weights: Option<&Array1<f64>>) -> f64 {
    let n = y.len();
    if n == 0 {
        return 0.0;
    }

    let unit_loss = |yi: f64, mui: f64| -> f64 {
        let mui_safe = mui.clamp(1e-15, 1.0 - 1e-15);
        -(yi * mui_safe.ln() + (1.0 - yi) * (1.0 - mui_safe).ln())
    };

    match weights {
        Some(w) => {
            let sum_w: f64 = w.sum();
            if sum_w == 0.0 {
                return 0.0;
            }
            let weighted_sum: f64 = y
                .iter()
                .zip(mu.iter())
                .zip(w.iter())
                .map(|((&yi, &mui), &wi)| wi * unit_loss(yi, mui))
                .sum();
            weighted_sum / sum_w
        }
        None => {
            let sum: f64 = y
                .iter()
                .zip(mu.iter())
                .map(|(&yi, &mui)| unit_loss(yi, mui))
                .sum();
            sum / n as f64
        }
    }
}

/// Tweedie Deviance Loss
///
/// For variance power p:
/// - p = 0: Gaussian
/// - p = 1: Poisson
/// - p = 2: Gamma
/// - 1 < p < 2: Compound Poisson-Gamma (insurance)
///
/// Single-pass `fold` delegating to `TweedieFamily::unit_deviance_at`. The
/// `Family` trait owns the per-row arithmetic for every branch (p=0, p=1,
/// p=2, general), so this function cannot drift from the canonical
/// implementation used by IRLS/null-deviance.
///
/// # Panics
/// Panics if `var_power` is in the open interval `(0, 1)` (the Tweedie
/// distribution is not defined there). All valid Tweedie powers
/// (p ≤ 0 or p ≥ 1) are supported.
pub fn tweedie_deviance_loss(
    y: &Array1<f64>,
    mu: &Array1<f64>,
    var_power: f64,
    weights: Option<&Array1<f64>>,
) -> f64 {
    let n = y.len();
    if n == 0 {
        return 0.0;
    }

    let family = TweedieFamily::new(var_power)
        .expect("tweedie_deviance_loss: var_power must be <= 0 or >= 1");

    match weights {
        Some(w) => {
            let sum_w: f64 = w.sum();
            if sum_w == 0.0 {
                return 0.0;
            }
            let weighted_sum: f64 = y
                .iter()
                .zip(mu.iter())
                .zip(w.iter())
                .map(|((&yi, &mui), &wi)| wi * family.unit_deviance_at(yi, mui))
                .sum();
            weighted_sum / sum_w
        }
        None => {
            let sum: f64 = y
                .iter()
                .zip(mu.iter())
                .map(|(&yi, &mui)| family.unit_deviance_at(yi, mui))
                .sum();
            sum / n as f64
        }
    }
}

/// Compute Tweedie unit deviance for a single observation.
///
/// Test-only reference retained for bit-exact equivalence checks in
/// `test_loss_fold_matches_naive_reference`. Production
/// `tweedie_deviance_loss` delegates to `TweedieFamily::unit_deviance_at`.
#[cfg(test)]
fn tweedie_unit_deviance(y: f64, mu: f64, p: f64) -> f64 {
    use crate::constants::ZERO_TOL;
    if (p - 0.0).abs() < ZERO_TOL {
        // Gaussian: (y - μ)²
        (y - mu).powi(2)
    } else if (p - 1.0).abs() < ZERO_TOL {
        // Poisson: 2 * (y * log(y/μ) - (y - μ))
        if y == 0.0 {
            2.0 * mu
        } else {
            2.0 * (y * (y / mu).ln() - (y - mu))
        }
    } else if (p - 2.0).abs() < ZERO_TOL {
        // Gamma: 2 * ((y - μ)/μ - log(y/μ))
        let y_safe = y.max(MU_MIN_POSITIVE);
        2.0 * ((y_safe - mu) / mu - (y_safe / mu).ln())
    } else {
        // General Tweedie
        let y_safe = y.max(MU_MIN_POSITIVE);
        2.0 * (y_safe.powf(2.0 - p) / ((1.0 - p) * (2.0 - p))
            - y_safe * mu.powf(1.0 - p) / (1.0 - p)
            + mu.powf(2.0 - p) / (2.0 - p))
    }
}

/// Negative Binomial Deviance Loss
///
/// Includes theta (dispersion) parameter.
///
/// Single-pass `fold` delegating to `NegativeBinomialFamily::unit_deviance_at`.
/// The y=0 branch (`2θ·log((μ+θ)/θ)`) is defined once in the trait
/// implementation, preventing the sign-error drift that used to exist
/// between this path and the canonical per-row arithmetic.
///
/// # Panics
/// Panics if `theta <= 0` (required by `NegativeBinomialFamily::new`).
pub fn negbinomial_deviance_loss(
    y: &Array1<f64>,
    mu: &Array1<f64>,
    theta: f64,
    weights: Option<&Array1<f64>>,
) -> f64 {
    let n = y.len();
    if n == 0 {
        return 0.0;
    }

    let family =
        NegativeBinomialFamily::new(theta).expect("negbinomial_deviance_loss: theta must be > 0");

    match weights {
        Some(w) => {
            let sum_w: f64 = w.sum();
            if sum_w == 0.0 {
                return 0.0;
            }
            let weighted_sum: f64 = y
                .iter()
                .zip(mu.iter())
                .zip(w.iter())
                .map(|((&yi, &mui), &wi)| wi * family.unit_deviance_at(yi, mui))
                .sum();
            weighted_sum / sum_w
        }
        None => {
            let sum: f64 = y
                .iter()
                .zip(mu.iter())
                .map(|(&yi, &mui)| family.unit_deviance_at(yi, mui))
                .sum();
            sum / n as f64
        }
    }
}

/// Get the default loss function name for a family.
/// Returns an error for unknown family names.
pub fn default_loss_name(family: &str) -> Result<&'static str, String> {
    match family.to_lowercase().as_str() {
        "gaussian" | "normal" => Ok("mse"),
        "poisson" | "quasipoisson" => Ok("poisson_deviance"),
        "gamma" => Ok("gamma_deviance"),
        "binomial" | "quasibinomial" => Ok("log_loss"),
        "tweedie" => Ok("tweedie_deviance"),
        "negativebinomial" | "negbinomial" | "nb" => Ok("negbinomial_deviance"),
        other => Err(format!("Unknown family '{}' in default_loss_name", other)),
    }
}

/// Compute the default loss for a given family.
/// Returns an error for unknown family names.
pub fn compute_family_loss(
    family: &str,
    y: &Array1<f64>,
    mu: &Array1<f64>,
    weights: Option<&Array1<f64>>,
    var_power: Option<f64>,
    theta: Option<f64>,
) -> Result<f64, String> {
    let lower = family.to_lowercase();

    // Handle negativebinomial with optional theta parameter like "negativebinomial(theta=1.38)"
    if lower.starts_with("negativebinomial")
        || lower.starts_with("negbinomial")
        || lower.starts_with("nb(")
        || lower == "nb"
    {
        // Parse theta from family string if present, otherwise use provided theta
        let parsed_theta = if let Some(start) = lower.find("theta=") {
            let rest = &lower[start + 6..];
            let end = rest.find(')').unwrap_or(rest.len());
            rest[..end].parse::<f64>().unwrap_or(1.0)
        } else {
            theta.unwrap_or(1.0)
        };
        return Ok(negbinomial_deviance_loss(y, mu, parsed_theta, weights));
    }

    match lower.as_str() {
        "gaussian" | "normal" => Ok(mse(y, mu, weights)),
        "poisson" | "quasipoisson" => Ok(poisson_deviance_loss(y, mu, weights)),
        "gamma" => Ok(gamma_deviance_loss(y, mu, weights)),
        "binomial" | "quasibinomial" => Ok(log_loss(y, mu, weights)),
        "tweedie" => Ok(tweedie_deviance_loss(
            y,
            mu,
            var_power.unwrap_or(1.5),
            weights,
        )),
        other => Err(format!("Unknown family '{}' in compute_family_loss", other)),
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_mse() {
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let mu = array![1.1, 2.0, 2.9, 4.2, 4.8];

        let result = mse(&y, &mu, None);
        // (0.01 + 0 + 0.01 + 0.04 + 0.04) / 5 = 0.02
        assert_abs_diff_eq!(result, 0.02, epsilon = 1e-10);
    }

    #[test]
    fn test_mse_weighted() {
        let y = array![1.0, 2.0];
        let mu = array![2.0, 2.0]; // errors: 1, 0
        let w = array![1.0, 3.0]; // weight more on second

        let result = mse(&y, &mu, Some(&w));
        // (1*1 + 3*0) / 4 = 0.25
        assert_abs_diff_eq!(result, 0.25, epsilon = 1e-10);
    }

    #[test]
    fn test_mse_empty() {
        let y: Array1<f64> = array![];
        let mu: Array1<f64> = array![];
        assert_abs_diff_eq!(mse(&y, &mu, None), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mse_zero_weights() {
        let y = array![1.0, 2.0];
        let mu = array![2.0, 3.0];
        let w = array![0.0, 0.0];
        assert_abs_diff_eq!(mse(&y, &mu, Some(&w)), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_rmse() {
        let y = array![1.0, 2.0, 3.0, 4.0];
        let mu = array![2.0, 3.0, 4.0, 5.0]; // All errors = 1

        let result = rmse(&y, &mu, None);
        // MSE = 1.0, RMSE = 1.0
        assert_abs_diff_eq!(result, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_rmse_weighted() {
        let y = array![0.0, 0.0];
        let mu = array![1.0, 2.0]; // errors: 1, 4
        let w = array![3.0, 1.0];

        // MSE = (3*1 + 1*4) / 4 = 7/4 = 1.75
        // RMSE = sqrt(1.75)
        let result = rmse(&y, &mu, Some(&w));
        assert_abs_diff_eq!(result, 1.75_f64.sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn test_mae() {
        let y = array![1.0, 2.0, 3.0];
        let mu = array![1.5, 2.0, 2.5];

        let result = mae(&y, &mu, None);
        // (0.5 + 0 + 0.5) / 3 = 0.333...
        assert_abs_diff_eq!(result, 1.0 / 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mae_weighted() {
        let y = array![0.0, 0.0];
        let mu = array![1.0, 2.0]; // abs errors: 1, 2
        let w = array![2.0, 1.0];

        // (2*1 + 1*2) / 3 = 4/3
        let result = mae(&y, &mu, Some(&w));
        assert_abs_diff_eq!(result, 4.0 / 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mae_empty() {
        let y: Array1<f64> = array![];
        let mu: Array1<f64> = array![];
        assert_abs_diff_eq!(mae(&y, &mu, None), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_poisson_deviance_loss() {
        let y = array![0.0, 1.0, 2.0, 5.0];
        let mu = array![0.5, 1.0, 2.5, 4.0];

        let result = poisson_deviance_loss(&y, &mu, None);
        // Should be positive and reasonable
        assert!(result > 0.0);
        assert!(result < 1.0);
    }

    #[test]
    fn test_poisson_deviance_loss_perfect() {
        let y = array![1.0, 2.0, 3.0];
        let mu = array![1.0, 2.0, 3.0];

        let result = poisson_deviance_loss(&y, &mu, None);
        assert_abs_diff_eq!(result, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_poisson_deviance_loss_weighted() {
        let y = array![0.0, 1.0];
        let mu = array![1.0, 1.0];
        let w = array![1.0, 2.0];

        let result = poisson_deviance_loss(&y, &mu, Some(&w));
        assert!(result > 0.0);
    }

    #[test]
    fn test_poisson_deviance_loss_empty() {
        let y: Array1<f64> = array![];
        let mu: Array1<f64> = array![];
        assert_abs_diff_eq!(poisson_deviance_loss(&y, &mu, None), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_log_loss_perfect() {
        let y = array![1.0, 0.0, 1.0, 0.0];
        let mu = array![0.99, 0.01, 0.99, 0.01];

        let result = log_loss(&y, &mu, None);
        // Near-perfect predictions should have very low loss
        assert!(result < 0.1);
    }

    #[test]
    fn test_log_loss_poor() {
        let y = array![1.0, 0.0, 1.0, 0.0];
        let mu = array![0.5, 0.5, 0.5, 0.5];

        let result = log_loss(&y, &mu, None);
        // Random predictions should have loss ~= log(2) ≈ 0.693
        assert_abs_diff_eq!(result, 0.693, epsilon = 0.01);
    }

    #[test]
    fn test_log_loss_weighted() {
        let y = array![1.0, 0.0];
        let mu = array![0.9, 0.1];
        let w = array![1.0, 1.0];

        let result = log_loss(&y, &mu, Some(&w));
        assert!(result > 0.0);
        assert!(result < 0.5);
    }

    #[test]
    fn test_log_loss_empty() {
        let y: Array1<f64> = array![];
        let mu: Array1<f64> = array![];
        assert_abs_diff_eq!(log_loss(&y, &mu, None), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_gamma_deviance_loss() {
        let y = array![1.0, 2.0, 3.0, 4.0];
        let mu = array![1.0, 2.0, 3.0, 4.0]; // Perfect fit

        let result = gamma_deviance_loss(&y, &mu, None);
        // Perfect fit should have zero loss
        assert_abs_diff_eq!(result, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_gamma_deviance_loss_imperfect() {
        let y = array![1.0, 2.0, 3.0];
        let mu = array![1.5, 2.5, 3.5];

        let result = gamma_deviance_loss(&y, &mu, None);
        assert!(result > 0.0);
    }

    #[test]
    fn test_gamma_deviance_loss_weighted() {
        let y = array![1.0, 2.0];
        let mu = array![1.0, 2.0];
        let w = array![2.0, 1.0];

        let result = gamma_deviance_loss(&y, &mu, Some(&w));
        assert_abs_diff_eq!(result, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_gamma_deviance_loss_empty() {
        let y: Array1<f64> = array![];
        let mu: Array1<f64> = array![];
        assert_abs_diff_eq!(gamma_deviance_loss(&y, &mu, None), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_tweedie_deviance_loss_gaussian() {
        let y = array![1.0, 2.0, 3.0];
        let mu = array![1.5, 2.5, 3.5];

        // var_power = 0 → Gaussian
        let result = tweedie_deviance_loss(&y, &mu, 0.0, None);
        // Should match MSE * 2 (unit deviance is (y-mu)^2 for Gaussian)
        let expected_mse = mse(&y, &mu, None);
        assert_abs_diff_eq!(result, expected_mse, epsilon = 1e-10);
    }

    #[test]
    fn test_tweedie_deviance_loss_poisson() {
        let y = array![1.0, 2.0, 3.0];
        let mu = array![1.0, 2.0, 3.0];

        // var_power = 1 → Poisson
        let result = tweedie_deviance_loss(&y, &mu, 1.0, None);
        assert_abs_diff_eq!(result, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_tweedie_deviance_loss_gamma() {
        let y = array![1.0, 2.0, 3.0];
        let mu = array![1.0, 2.0, 3.0];

        // var_power = 2 → Gamma
        let result = tweedie_deviance_loss(&y, &mu, 2.0, None);
        assert_abs_diff_eq!(result, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_tweedie_deviance_loss_compound_poisson() {
        let y = array![0.0, 1.0, 2.0, 5.0];
        let mu = array![0.5, 1.0, 2.0, 5.0];

        // var_power = 1.5 → Compound Poisson-Gamma
        let result = tweedie_deviance_loss(&y, &mu, 1.5, None);
        assert!(result >= 0.0);
    }

    #[test]
    fn test_tweedie_deviance_loss_weighted() {
        let y = array![1.0, 2.0];
        let mu = array![1.0, 2.0];
        let w = array![1.0, 2.0];

        let result = tweedie_deviance_loss(&y, &mu, 1.5, Some(&w));
        assert_abs_diff_eq!(result, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_tweedie_deviance_loss_empty() {
        let y: Array1<f64> = array![];
        let mu: Array1<f64> = array![];
        assert_abs_diff_eq!(
            tweedie_deviance_loss(&y, &mu, 1.5, None),
            0.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_negbinomial_deviance_loss() {
        let y = array![1.0, 2.0, 3.0];
        let mu = array![1.0, 2.0, 3.0];
        let theta = 1.0;

        let result = negbinomial_deviance_loss(&y, &mu, theta, None);
        // Perfect fit: deviance contributions should be ~0
        assert!(result.abs() < 0.1);
    }

    #[test]
    fn test_negbinomial_deviance_loss_imperfect() {
        let y = array![0.0, 1.0, 5.0];
        let mu = array![1.0, 2.0, 3.0];
        let theta = 2.0;

        let result = negbinomial_deviance_loss(&y, &mu, theta, None);
        assert!(result > 0.0);
    }

    #[test]
    fn test_negbinomial_deviance_loss_weighted() {
        let y = array![1.0, 2.0];
        let mu = array![1.0, 2.0];
        let w = array![1.0, 2.0];
        let theta = 1.5;

        let result = negbinomial_deviance_loss(&y, &mu, theta, Some(&w));
        assert!(result.abs() < 0.1);
    }

    #[test]
    fn test_negbinomial_deviance_loss_empty() {
        let y: Array1<f64> = array![];
        let mu: Array1<f64> = array![];
        assert_abs_diff_eq!(
            negbinomial_deviance_loss(&y, &mu, 1.0, None),
            0.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_default_loss_name() {
        assert_eq!(default_loss_name("gaussian").unwrap(), "mse");
        assert_eq!(default_loss_name("Gaussian").unwrap(), "mse");
        assert_eq!(default_loss_name("normal").unwrap(), "mse");
        assert_eq!(default_loss_name("poisson").unwrap(), "poisson_deviance");
        assert_eq!(default_loss_name("Poisson").unwrap(), "poisson_deviance");
        assert_eq!(
            default_loss_name("quasipoisson").unwrap(),
            "poisson_deviance"
        );
        assert_eq!(default_loss_name("gamma").unwrap(), "gamma_deviance");
        assert_eq!(default_loss_name("Gamma").unwrap(), "gamma_deviance");
        assert_eq!(default_loss_name("binomial").unwrap(), "log_loss");
        assert_eq!(default_loss_name("quasibinomial").unwrap(), "log_loss");
        assert_eq!(default_loss_name("tweedie").unwrap(), "tweedie_deviance");
        assert_eq!(
            default_loss_name("negativebinomial").unwrap(),
            "negbinomial_deviance"
        );
        assert_eq!(
            default_loss_name("negbinomial").unwrap(),
            "negbinomial_deviance"
        );
        assert_eq!(default_loss_name("nb").unwrap(), "negbinomial_deviance");
        assert!(default_loss_name("unknown").is_err());
    }

    #[test]
    fn test_compute_family_loss_gaussian() {
        let y = array![1.0, 2.0, 3.0];
        let mu = array![1.5, 2.5, 3.5];

        let result = compute_family_loss("gaussian", &y, &mu, None, None, None).unwrap();
        let expected = mse(&y, &mu, None);
        assert_abs_diff_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_compute_family_loss_poisson() {
        let y = array![1.0, 2.0, 3.0];
        let mu = array![1.0, 2.0, 3.0];

        let result = compute_family_loss("poisson", &y, &mu, None, None, None).unwrap();
        let expected = poisson_deviance_loss(&y, &mu, None);
        assert_abs_diff_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_compute_family_loss_gamma() {
        let y = array![1.0, 2.0, 3.0];
        let mu = array![1.0, 2.0, 3.0];

        let result = compute_family_loss("gamma", &y, &mu, None, None, None).unwrap();
        let expected = gamma_deviance_loss(&y, &mu, None);
        assert_abs_diff_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_compute_family_loss_binomial() {
        let y = array![0.0, 1.0, 0.0, 1.0];
        let mu = array![0.2, 0.8, 0.3, 0.7];

        let result = compute_family_loss("binomial", &y, &mu, None, None, None).unwrap();
        let expected = log_loss(&y, &mu, None);
        assert_abs_diff_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_compute_family_loss_tweedie() {
        let y = array![0.0, 1.0, 2.0];
        let mu = array![0.5, 1.0, 2.0];

        let result = compute_family_loss("tweedie", &y, &mu, None, Some(1.5), None).unwrap();
        let expected = tweedie_deviance_loss(&y, &mu, 1.5, None);
        assert_abs_diff_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_compute_family_loss_negbinomial() {
        let y = array![1.0, 2.0, 3.0];
        let mu = array![1.0, 2.0, 3.0];

        let result =
            compute_family_loss("negativebinomial", &y, &mu, None, None, Some(1.0)).unwrap();
        let expected = negbinomial_deviance_loss(&y, &mu, 1.0, None);
        assert_abs_diff_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_compute_family_loss_negbinomial_with_theta_in_name() {
        let y = array![1.0, 2.0, 3.0];
        let mu = array![1.0, 2.0, 3.0];

        let result =
            compute_family_loss("negativebinomial(theta=2.5)", &y, &mu, None, None, None).unwrap();
        let expected = negbinomial_deviance_loss(&y, &mu, 2.5, None);
        assert_abs_diff_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_compute_family_loss_quasipoisson() {
        let y = array![1.0, 2.0, 3.0];
        let mu = array![1.0, 2.0, 3.0];

        let result = compute_family_loss("quasipoisson", &y, &mu, None, None, None).unwrap();
        let expected = poisson_deviance_loss(&y, &mu, None);
        assert_abs_diff_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_compute_family_loss_quasibinomial() {
        let y = array![0.0, 1.0];
        let mu = array![0.3, 0.7];

        let result = compute_family_loss("quasibinomial", &y, &mu, None, None, None).unwrap();
        let expected = log_loss(&y, &mu, None);
        assert_abs_diff_eq!(result, expected, epsilon = 1e-10);
    }

    // -----------------------------------------------------------------
    // Bit-exact equivalence vs naive Vec-then-sum reference
    // -----------------------------------------------------------------
    //
    // The single-pass `fold` implementation must match the previous
    // `let v: Vec<f64> = ...; v.iter().sum()` pattern exactly because both
    // perform a sequential left-fold of f64 in the same order. These tests
    // pin the equivalence on a non-trivial input so any future refactor
    // (e.g. switching to Rayon or Kahan summation) is caught.

    fn lcg_pair(seed: u64, n: usize, lo: f64, hi: f64) -> Vec<f64> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u = ((s >> 11) as f64) / ((1u64 << 53) as f64);
                lo + (hi - lo) * u
            })
            .collect()
    }

    #[test]
    fn test_loss_fold_matches_naive_reference() {
        let n = 1_000;
        let y_vec = lcg_pair(0xCAFEBABE, n, 0.0, 5.0);
        let mu_vec = lcg_pair(0xDEADBEEF, n, 0.1, 5.0);
        let w_vec = lcg_pair(0x12345678, n, 0.5, 1.5);
        let y = Array1::from_vec(y_vec);
        let mu = Array1::from_vec(mu_vec);
        let w = Array1::from_vec(w_vec);

        // Helpers that recreate the OLD Vec-then-sum pattern explicitly.
        let reference = |unit: &dyn Fn(f64, f64) -> f64, w: Option<&Array1<f64>>| -> f64 {
            let v: Vec<f64> = y
                .iter()
                .zip(mu.iter())
                .map(|(&yi, &mui)| unit(yi, mui))
                .collect();
            match w {
                Some(w) => {
                    let sw: f64 = w.sum();
                    let ws: f64 = v.iter().zip(w.iter()).map(|(&d, &wi)| wi * d).sum();
                    ws / sw
                }
                None => v.iter().sum::<f64>() / n as f64,
            }
        };

        // poisson
        let unit_p = |yi: f64, mui: f64| {
            let mui_safe = mui.max(MU_MIN_POSITIVE);
            if yi == 0.0 {
                2.0 * mui_safe
            } else {
                2.0 * (yi * (yi / mui_safe).ln() - (yi - mui_safe))
            }
        };
        assert_eq!(
            poisson_deviance_loss(&y, &mu, None),
            reference(&unit_p, None)
        );
        assert_eq!(
            poisson_deviance_loss(&y, &mu, Some(&w)),
            reference(&unit_p, Some(&w))
        );

        // gamma
        let unit_g = |yi: f64, mui: f64| {
            let yi_safe = yi.max(MU_MIN_POSITIVE);
            let mui_safe = mui.max(MU_MIN_POSITIVE);
            let ratio = yi_safe / mui_safe;
            2.0 * ((yi_safe - mui_safe) / mui_safe - ratio.ln())
        };
        assert_eq!(gamma_deviance_loss(&y, &mu, None), reference(&unit_g, None));
        assert_eq!(
            gamma_deviance_loss(&y, &mu, Some(&w)),
            reference(&unit_g, Some(&w))
        );

        // log_loss (Bernoulli-ish y, mu in (0,1))
        let y_b: Array1<f64> = y.mapv(|v| if v > 2.5 { 1.0 } else { 0.0 });
        let mu_p: Array1<f64> = mu.mapv(|m| (m / (m + 1.0)).clamp(1e-15, 1.0 - 1e-15));
        let unit_l = |yi: f64, mui: f64| {
            let mui_safe = mui.clamp(1e-15, 1.0 - 1e-15);
            -(yi * mui_safe.ln() + (1.0 - yi) * (1.0 - mui_safe).ln())
        };
        let v: Vec<f64> = y_b
            .iter()
            .zip(mu_p.iter())
            .map(|(&yi, &mui)| unit_l(yi, mui))
            .collect();
        let ref_unweighted = v.iter().sum::<f64>() / n as f64;
        assert_eq!(log_loss(&y_b, &mu_p, None), ref_unweighted);
        let ref_weighted = {
            let sw: f64 = w.sum();
            let ws: f64 = v.iter().zip(w.iter()).map(|(&d, &wi)| wi * d).sum();
            ws / sw
        };
        assert_eq!(log_loss(&y_b, &mu_p, Some(&w)), ref_weighted);

        // tweedie p=1.5
        let unit_t = |yi: f64, mui: f64| {
            let mui_safe = mui.max(MU_MIN_POSITIVE);
            tweedie_unit_deviance(yi, mui_safe, 1.5)
        };
        assert_eq!(
            tweedie_deviance_loss(&y, &mu, 1.5, None),
            reference(&unit_t, None)
        );
        assert_eq!(
            tweedie_deviance_loss(&y, &mu, 1.5, Some(&w)),
            reference(&unit_t, Some(&w))
        );

        // negbinomial theta=2.0
        let theta = 2.0;
        let unit_nb = |yi: f64, mui: f64| {
            let mui_safe = mui.max(MU_MIN_POSITIVE);
            let yi_safe = yi.max(0.0);
            let term1 = if yi_safe > 0.0 {
                yi_safe * (yi_safe / mui_safe).ln()
            } else {
                0.0
            };
            let term2 = (yi_safe + theta) * ((yi_safe + theta) / (mui_safe + theta)).ln();
            2.0 * (term1 - term2)
        };
        assert_eq!(
            negbinomial_deviance_loss(&y, &mu, theta, None),
            reference(&unit_nb, None)
        );
        assert_eq!(
            negbinomial_deviance_loss(&y, &mu, theta, Some(&w)),
            reference(&unit_nb, Some(&w))
        );
    }

    // -----------------------------------------------------------------
    // Canonical-trait equivalence tests
    // -----------------------------------------------------------------
    //
    // These tests pin each `*_deviance_loss` function in this file against
    // the canonical `Family::unit_deviance_at` trait method. They exist so
    // that the planned refactor (delegating loss.rs formulas to the trait)
    // can be verified to be a behaviour-preserving change, AND so that any
    // future drift between the loss.rs inline arithmetic and the trait
    // implementations is caught at CI time.
    //
    // Every assertion computes the expected value by iterating over y/mu
    // (and weights, when present) and calling `family.unit_deviance_at`
    // explicitly — it is NEVER a hand-coded number. If someone flips a
    // sign, swaps (y, mu) argument order, or changes a clamp in either
    // path without updating the other, these tests fire.
    //
    // Tolerance policy:
    //   * Most tests use `assert_relative_eq!` at ~1e-13: this catches any
    //     algebraic drift (sign flips, dropped terms, swapped args) while
    //     allowing float-ordering differences between the two paths (e.g.
    //     log(a/b) vs -log(b/a)).
    //   * Tweedie general-p (p != 0, 1, 2) currently has a known y=0 floor
    //     discrepancy (loss.rs uses y.max(MU_MIN_POSITIVE)=1e-10, the trait
    //     guards with `if yi_safe > 0.0`). Tests that exercise y=0 at
    //     p=1.5 use a looser 1e-3 relative tolerance — still tight enough
    //     to catch sign flips, loose enough to let the current drift pass.
    //     After the refactor delegates to the trait, these tolerances
    //     should effectively drop to ULP-level.

    use crate::families::{
        BinomialFamily, Family, GammaFamily, GaussianFamily, NegativeBinomialFamily, PoissonFamily,
        TweedieFamily,
    };
    use approx::assert_relative_eq;

    /// Compute the mean (or weighted mean) of a unit-deviance kernel,
    /// mirroring the `(sum / n)` and `(weighted_sum / sum_w)` semantics of
    /// the loss functions. Used as the canonical reference for every
    /// trait-equivalence assertion below.
    fn trait_loss_reference(
        family: &dyn Family,
        y: &Array1<f64>,
        mu: &Array1<f64>,
        w: Option<&Array1<f64>>,
    ) -> f64 {
        let n = y.len();
        if n == 0 {
            return 0.0;
        }
        match w {
            Some(w) => {
                let sum_w: f64 = w.sum();
                if sum_w == 0.0 {
                    return 0.0;
                }
                let ws: f64 = y
                    .iter()
                    .zip(mu.iter())
                    .zip(w.iter())
                    .map(|((&yi, &mui), &wi)| wi * family.unit_deviance_at(yi, mui))
                    .sum();
                ws / sum_w
            }
            None => {
                let s: f64 = y
                    .iter()
                    .zip(mu.iter())
                    .map(|(&yi, &mui)| family.unit_deviance_at(yi, mui))
                    .sum();
                s / n as f64
            }
        }
    }

    /// Deterministic PRNG for synthetic data so tests are reproducible.
    fn synthetic_pair(seed: u64, n: usize, lo: f64, hi: f64) -> Array1<f64> {
        Array1::from_vec(lcg_pair(seed, n, lo, hi))
    }

    /// Inputs where `y` contains a mix of zeros and positive values so
    /// the y=0 branch of every family is exercised. `mu` is kept >= 0.1
    /// so loss.rs' `.max(MU_MIN_POSITIVE=1e-10)` clamp is a no-op and the
    /// comparison against the (unclamped in some cases) trait method is
    /// bit-exact.
    fn synthetic_with_zeros(seed_y: u64, seed_mu: u64, n: usize) -> (Array1<f64>, Array1<f64>) {
        // Start with uniform [0, 5) then zero out ~25% of rows to exercise
        // the y=0 code path (critical for catching the NegBin sign bug).
        let mut y = synthetic_pair(seed_y, n, 0.0, 5.0);
        for (i, v) in y.iter_mut().enumerate() {
            if i % 4 == 0 {
                *v = 0.0;
            }
        }
        let mu = synthetic_pair(seed_mu, n, 0.1, 5.0);
        (y, mu)
    }

    // --- Poisson --------------------------------------------------------

    /// Drift catcher: if `poisson_deviance_loss` ever flips the sign of
    /// the `(y - μ)` term or drops the y=0 special-case, the expected
    /// value (computed via `PoissonFamily::unit_deviance_at`) will no
    /// longer match. Tight 1e-13 relative tolerance: both paths use the
    /// same per-row arithmetic; residual drift is pure float ordering.
    #[test]
    fn test_poisson_deviance_loss_matches_family_trait() {
        let (y, mu) = synthetic_with_zeros(0x1111_2222, 0x3333_4444, 1_000);
        let expected = trait_loss_reference(&PoissonFamily, &y, &mu, None);
        assert_relative_eq!(
            poisson_deviance_loss(&y, &mu, None),
            expected,
            max_relative = 1e-13
        );
    }

    /// Same drift as the unweighted test, additionally catches any
    /// drift in how weights combine with per-row deviance (e.g. weight
    /// applied outside the sum, or `sum(w*d)/n` instead of
    /// `sum(w*d)/sum(w)`).
    #[test]
    fn test_poisson_deviance_loss_weighted_matches_family_trait() {
        let (y, mu) = synthetic_with_zeros(0x1111_2222, 0x3333_4444, 1_000);
        let w = synthetic_pair(0x5555_6666, 1_000, 0.5, 1.5);
        let expected = trait_loss_reference(&PoissonFamily, &y, &mu, Some(&w));
        assert_relative_eq!(
            poisson_deviance_loss(&y, &mu, Some(&w)),
            expected,
            max_relative = 1e-13
        );
    }

    // --- Gamma ----------------------------------------------------------

    /// Drift catcher: Gamma deviance is `2*((y-μ)/μ - log(y/μ))`; a
    /// common mis-edit is flipping numerator/denominator of the ratio or
    /// dropping the outer factor of 2. Compared to
    /// `GammaFamily::unit_deviance_at`.
    #[test]
    fn test_gamma_deviance_loss_matches_family_trait() {
        // Gamma requires y > 0; use strictly positive synthetic data.
        let y = synthetic_pair(0xA1A1_B2B2, 1_000, 0.1, 5.0);
        let mu = synthetic_pair(0xC3C3_D4D4, 1_000, 0.1, 5.0);
        let expected = trait_loss_reference(&GammaFamily, &y, &mu, None);
        assert_relative_eq!(
            gamma_deviance_loss(&y, &mu, None),
            expected,
            max_relative = 1e-13
        );
    }

    #[test]
    fn test_gamma_deviance_loss_weighted_matches_family_trait() {
        let y = synthetic_pair(0xA1A1_B2B2, 1_000, 0.1, 5.0);
        let mu = synthetic_pair(0xC3C3_D4D4, 1_000, 0.1, 5.0);
        let w = synthetic_pair(0xE5E5_F6F6, 1_000, 0.5, 1.5);
        let expected = trait_loss_reference(&GammaFamily, &y, &mu, Some(&w));
        assert_relative_eq!(
            gamma_deviance_loss(&y, &mu, Some(&w)),
            expected,
            max_relative = 1e-13
        );
    }

    // --- Tweedie (all 4 branches) ---------------------------------------

    /// Exercises the `p=0` Gaussian branch of `tweedie_unit_deviance`
    /// vs. `TweedieFamily { var_power: 0.0 }`. Drift catcher: any edit
    /// that changes `(y-μ)²` to `(μ-y)²` is fine (symmetric), but flipping
    /// the sign of an added term or dropping the squaring would be caught.
    /// `p=0` branch uses identical arithmetic on both sides so drift is
    /// pure float ordering; 1e-13 tolerance.
    #[test]
    fn test_tweedie_p0_matches_family_trait() {
        let (y, mu) = synthetic_with_zeros(0x7070_8080, 0x9090_A0A0, 500);
        let fam = TweedieFamily::new(0.0).expect("valid var_power");
        let expected = trait_loss_reference(&fam, &y, &mu, None);
        assert_relative_eq!(
            tweedie_deviance_loss(&y, &mu, 0.0, None),
            expected,
            max_relative = 1e-13
        );
    }

    /// Exercises the `p=1` Poisson branch. Drift catcher includes the
    /// y=0 special case (critical: dropping it gives NaN from `0*log(0/μ)`
    /// on some platforms). `p=1` branch uses identical arithmetic so
    /// 1e-13 tolerance is FP-ordering-only.
    #[test]
    fn test_tweedie_p1_matches_family_trait() {
        let (y, mu) = synthetic_with_zeros(0x7070_8080, 0x9090_A0A0, 500);
        let fam = TweedieFamily::new(1.0).expect("valid var_power");
        let expected = trait_loss_reference(&fam, &y, &mu, None);
        assert_relative_eq!(
            tweedie_deviance_loss(&y, &mu, 1.0, None),
            expected,
            max_relative = 1e-13
        );
    }

    /// Exercises the compound Poisson-Gamma (1 < p < 2) general branch.
    /// Drift catcher: this is the most algebraically fragile branch with
    /// three terms and signs `(+term1 - term2 + term3)` — any sign flip
    /// or dropped term fires the test.
    ///
    /// Tolerance note: after the refactor, `tweedie_deviance_loss`
    /// delegates to `TweedieFamily::unit_deviance_at`, so both sides
    /// evaluate the same arithmetic per row and agree to FP-ordering
    /// precision. Tolerance tightened to 1e-13.
    #[test]
    fn test_tweedie_p15_matches_family_trait() {
        let (y, mu) = synthetic_with_zeros(0x7070_8080, 0x9090_A0A0, 500);
        let fam = TweedieFamily::new(1.5).expect("valid var_power");
        let expected = trait_loss_reference(&fam, &y, &mu, None);
        assert_relative_eq!(
            tweedie_deviance_loss(&y, &mu, 1.5, None),
            expected,
            max_relative = 1e-13
        );
    }

    /// Exercises the `p=2` Gamma branch. Note we use y > 0 here since
    /// this branch uses `y/μ` inside a log.
    #[test]
    fn test_tweedie_p2_matches_family_trait() {
        let y = synthetic_pair(0x7070_8080, 500, 0.1, 5.0);
        let mu = synthetic_pair(0x9090_A0A0, 500, 0.1, 5.0);
        let fam = TweedieFamily::new(2.0).expect("valid var_power");
        let expected = trait_loss_reference(&fam, &y, &mu, None);
        assert_relative_eq!(
            tweedie_deviance_loss(&y, &mu, 2.0, None),
            expected,
            max_relative = 1e-13
        );
    }

    /// Weighted variant at p=1.5 (the insurance default). Pins
    /// `sum(w*d)/sum(w)` aggregation. After the refactor, both sides
    /// share the trait arithmetic — tolerance tightened to 1e-13.
    #[test]
    fn test_tweedie_weighted_matches_family_trait() {
        let (y, mu) = synthetic_with_zeros(0x7070_8080, 0x9090_A0A0, 500);
        let w = synthetic_pair(0xB0B0_C0C0, 500, 0.5, 1.5);
        let fam = TweedieFamily::new(1.5).expect("valid var_power");
        let expected = trait_loss_reference(&fam, &y, &mu, Some(&w));
        assert_relative_eq!(
            tweedie_deviance_loss(&y, &mu, 1.5, Some(&w)),
            expected,
            max_relative = 1e-13
        );
    }

    // --- Negative Binomial (must include y=0!) --------------------------

    /// **This is the test that catches the recently-found NegBin y=0
    /// sign bug.** If someone inverts the y=0 branch to
    /// `2θ*log(θ/(μ+θ))` (which gives a negative deviance), this test
    /// fails immediately because `NegativeBinomialFamily::unit_deviance_at`
    /// uses the correct sign `2θ*log((μ+θ)/θ)`. Synthetic inputs
    /// deliberately include ~25% zero y rows to ensure that branch is
    /// exercised (without zeros, the bug wouldn't show up).
    ///
    /// Tolerance: 1e-13 relative. The two paths compute the same value
    /// but via slightly different FP sequences (loss.rs computes
    /// `log((0+θ)/(μ+θ))` then negates, the trait computes
    /// `log((μ+θ)/θ)` directly). A sign flip gives O(1) drift and is
    /// caught; pure FP-ordering differences are not.
    #[test]
    fn test_negbinomial_deviance_loss_matches_family_trait_with_zeros() {
        let (y, mu) = synthetic_with_zeros(0xDE_AD_BE_EF, 0xCA_FE_BA_BE, 1_000);
        assert!(
            y.iter().any(|&yi| yi == 0.0),
            "test premise: synthetic y must contain zeros to exercise the NB y=0 branch"
        );
        let theta = 1.5;
        let fam = NegativeBinomialFamily::new(theta).unwrap();
        let expected = trait_loss_reference(&fam, &y, &mu, None);
        assert_relative_eq!(
            negbinomial_deviance_loss(&y, &mu, theta, None),
            expected,
            max_relative = 1e-13
        );
    }

    /// Weighted NegBin with zeros — same bug-catching purpose, extended
    /// to ensure the weight combination logic is also unified between
    /// the two code paths.
    #[test]
    fn test_negbinomial_deviance_loss_weighted_matches_family_trait() {
        let (y, mu) = synthetic_with_zeros(0xDE_AD_BE_EF, 0xCA_FE_BA_BE, 1_000);
        let w = synthetic_pair(0xFE_ED_FA_CE, 1_000, 0.5, 1.5);
        let theta = 1.5;
        let fam = NegativeBinomialFamily::new(theta).unwrap();
        let expected = trait_loss_reference(&fam, &y, &mu, Some(&w));
        assert_relative_eq!(
            negbinomial_deviance_loss(&y, &mu, theta, Some(&w)),
            expected,
            max_relative = 1e-13
        );
    }

    /// Different theta to ensure the test doesn't accidentally pass
    /// only at some canonical value (e.g. theta=1).
    #[test]
    fn test_negbinomial_deviance_loss_theta_sweep_matches_family_trait() {
        let (y, mu) = synthetic_with_zeros(0xDE_AD_BE_EF, 0xCA_FE_BA_BE, 500);
        for &theta in &[0.25, 1.0, 3.7, 10.0] {
            let fam = NegativeBinomialFamily::new(theta).unwrap();
            let expected = trait_loss_reference(&fam, &y, &mu, None);
            assert_relative_eq!(
                negbinomial_deviance_loss(&y, &mu, theta, None),
                expected,
                max_relative = 1e-13
            );
        }
    }

    // --- log_loss (NOT a deviance — compare to Binomial log-likelihood) -

    /// `log_loss` is negative-log-likelihood per obs, not a deviance:
    /// `loss = -mean(y*log(μ) + (1-y)*log(1-μ))`. Therefore the canonical
    /// reference is `-log_likelihood_binomial / sum_w` (or `/ n`). This
    /// test pins the relationship against the binomial NLL helper so
    /// any drift in clamping bounds (loss.rs uses `[1e-15, 1-1e-15]`,
    /// `log_likelihood_binomial` uses `MU_MIN_PROBABILITY=1e-10`) is
    /// caught — provided mu stays inside both clamps (which we enforce
    /// via the test's [1e-6, 1-1e-6] clamp). 1e-13 relative allows for
    /// residual float-ordering between per-row sign flips.
    #[test]
    fn test_log_loss_matches_binomial_nll() {
        // y must be in [0, 1] for log_loss — use 0/1 Bernoulli outcomes.
        let raw_y = synthetic_pair(0x1234_5678, 1_000, 0.0, 1.0);
        let y: Array1<f64> = raw_y.mapv(|v| if v > 0.5 { 1.0 } else { 0.0 });
        // mu in (0,1) — well inside both clamps (1e-10 and 1e-15).
        let raw_mu = synthetic_pair(0x8765_4321, 1_000, 0.0, 1.0);
        let mu: Array1<f64> = raw_mu.mapv(|v| v.clamp(1e-6, 1.0 - 1e-6));
        let n = y.len() as f64;
        let expected = -crate::diagnostics::log_likelihood_binomial(&y, &mu, None) / n;
        assert_relative_eq!(log_loss(&y, &mu, None), expected, max_relative = 1e-13);
    }

    /// Weighted log_loss: `loss = -sum(w * nll) / sum(w)`.
    #[test]
    fn test_log_loss_weighted_matches_binomial_nll() {
        let raw_y = synthetic_pair(0x1234_5678, 500, 0.0, 1.0);
        let y: Array1<f64> = raw_y.mapv(|v| if v > 0.5 { 1.0 } else { 0.0 });
        let raw_mu = synthetic_pair(0x8765_4321, 500, 0.0, 1.0);
        let mu: Array1<f64> = raw_mu.mapv(|v| v.clamp(1e-6, 1.0 - 1e-6));
        let w = synthetic_pair(0xABCD_EF01, 500, 0.5, 1.5);
        let sum_w: f64 = w.sum();
        let expected = -crate::diagnostics::log_likelihood_binomial(&y, &mu, Some(&w)) / sum_w;
        assert_relative_eq!(log_loss(&y, &mu, Some(&w)), expected, max_relative = 1e-13);
    }

    // --- Edge cases: n=0, n=1, all-zero y, all-identical y --------------

    /// n=0: every loss function should return 0.0 on empty input.
    #[test]
    fn test_deviance_losses_empty_input() {
        let y: Array1<f64> = Array1::from_vec(vec![]);
        let mu: Array1<f64> = Array1::from_vec(vec![]);
        assert_eq!(poisson_deviance_loss(&y, &mu, None), 0.0);
        assert_eq!(gamma_deviance_loss(&y, &mu, None), 0.0);
        assert_eq!(tweedie_deviance_loss(&y, &mu, 0.0, None), 0.0);
        assert_eq!(tweedie_deviance_loss(&y, &mu, 1.5, None), 0.0);
        assert_eq!(negbinomial_deviance_loss(&y, &mu, 1.0, None), 0.0);
        assert_eq!(log_loss(&y, &mu, None), 0.0);
    }

    /// n=1: the single-row result must equal `unit_deviance_at(y[0], mu[0])`.
    /// Drift catcher: a refactor that accidentally divides by `n-1` or
    /// drops a row is caught immediately.
    #[test]
    fn test_deviance_losses_n1_matches_trait() {
        // Poisson y=0 branch
        let y = ndarray::array![0.0];
        let mu = ndarray::array![1.5];
        assert_relative_eq!(
            poisson_deviance_loss(&y, &mu, None),
            PoissonFamily.unit_deviance_at(0.0, 1.5),
            max_relative = 1e-13
        );
        // Gamma y>0 branch
        let y = ndarray::array![2.3];
        let mu = ndarray::array![1.7];
        assert_relative_eq!(
            gamma_deviance_loss(&y, &mu, None),
            GammaFamily.unit_deviance_at(2.3, 1.7),
            max_relative = 1e-13
        );
        // NegBin y=0 branch (the bug-prone one) — mean over n=1 must
        // still equal the single unit deviance.
        let y = ndarray::array![0.0];
        let mu = ndarray::array![2.0];
        let theta = 1.5;
        assert_relative_eq!(
            negbinomial_deviance_loss(&y, &mu, theta, None),
            NegativeBinomialFamily::new(theta)
                .unwrap()
                .unit_deviance_at(0.0, 2.0),
            max_relative = 1e-13
        );
        // Tweedie p=1 (Poisson) with y=0 — same per-row arithmetic on
        // both sides, catches divide-by-(n-1) style regressions.
        let y = ndarray::array![0.0];
        let mu = ndarray::array![1.3];
        assert_relative_eq!(
            tweedie_deviance_loss(&y, &mu, 1.0, None),
            TweedieFamily::new(1.0).unwrap().unit_deviance_at(0.0, 1.3),
            max_relative = 1e-13
        );
        // Tweedie p=2 (Gamma) with y>0 — the aggregation check; we avoid
        // p=general with y=0 here because that branch has a known floor
        // discrepancy (covered by test_tweedie_p15_matches_family_trait).
        let y = ndarray::array![2.0];
        let mu = ndarray::array![1.3];
        assert_relative_eq!(
            tweedie_deviance_loss(&y, &mu, 2.0, None),
            TweedieFamily::new(2.0).unwrap().unit_deviance_at(2.0, 1.3),
            max_relative = 1e-13
        );
    }

    /// All-zero y for Poisson and NegBin: every unit deviance comes from
    /// the y=0 branch. Catches any regression that breaks the y=0 special
    /// case.
    #[test]
    fn test_all_zero_y_matches_trait() {
        let y: Array1<f64> = Array1::zeros(100);
        let mu = synthetic_pair(0xDEAD_0000, 100, 0.1, 5.0);

        // Poisson
        let expected_p = trait_loss_reference(&PoissonFamily, &y, &mu, None);
        assert_relative_eq!(
            poisson_deviance_loss(&y, &mu, None),
            expected_p,
            max_relative = 1e-13
        );

        // NegBin at a couple of thetas
        for &theta in &[0.5, 2.0] {
            let fam = NegativeBinomialFamily::new(theta).unwrap();
            let expected_nb = trait_loss_reference(&fam, &y, &mu, None);
            assert_relative_eq!(
                negbinomial_deviance_loss(&y, &mu, theta, None),
                expected_nb,
                max_relative = 1e-13
            );
        }
    }

    /// Perfect fit (y == mu) should give exactly zero mean deviance for
    /// every family. This is a sanity check on the canonical reference
    /// itself — if the trait returns nonzero at y==mu, the whole
    /// deviance definition is wrong.
    #[test]
    fn test_all_identical_perfect_fit_matches_trait() {
        // For Gaussian/Gamma/Tweedie we need y = mu exactly.
        let y = ndarray::array![1.0, 1.0, 1.0, 1.0, 1.0];
        let mu = ndarray::array![1.0, 1.0, 1.0, 1.0, 1.0];
        assert_eq!(
            poisson_deviance_loss(&y, &mu, None),
            trait_loss_reference(&PoissonFamily, &y, &mu, None)
        );
        assert_eq!(
            gamma_deviance_loss(&y, &mu, None),
            trait_loss_reference(&GammaFamily, &y, &mu, None)
        );
        // Also check Gaussian trait at p=0 (uses GaussianFamily variant).
        assert_eq!(
            tweedie_deviance_loss(&y, &mu, 0.0, None),
            trait_loss_reference(&TweedieFamily::new(0.0).unwrap(), &y, &mu, None)
        );
        // Sanity: GaussianFamily.unit_deviance_at(1.0, 1.0) == 0.
        assert_eq!(GaussianFamily.unit_deviance_at(1.0, 1.0), 0.0);
        // log_loss with y=mu=0.5 should equal -log(0.5)
        let y_bin = ndarray::array![0.5, 0.5, 0.5];
        let mu_bin = ndarray::array![0.5, 0.5, 0.5];
        let n = y_bin.len() as f64;
        let expected = -crate::diagnostics::log_likelihood_binomial(&y_bin, &mu_bin, None) / n;
        assert_relative_eq!(
            log_loss(&y_bin, &mu_bin, None),
            expected,
            max_relative = 1e-13
        );
        // Also sanity-check BinomialFamily::unit_deviance_at is finite.
        assert!(BinomialFamily.unit_deviance_at(0.5, 0.5).is_finite());
    }

    /// Zero weights: every weighted loss should short-circuit to 0.0.
    #[test]
    fn test_deviance_losses_zero_weights_short_circuit() {
        let y = ndarray::array![1.0, 2.0, 3.0];
        let mu = ndarray::array![1.5, 2.5, 3.5];
        let w = ndarray::array![0.0, 0.0, 0.0];
        assert_eq!(poisson_deviance_loss(&y, &mu, Some(&w)), 0.0);
        assert_eq!(gamma_deviance_loss(&y, &mu, Some(&w)), 0.0);
        assert_eq!(tweedie_deviance_loss(&y, &mu, 1.5, Some(&w)), 0.0);
        assert_eq!(negbinomial_deviance_loss(&y, &mu, 1.0, Some(&w)), 0.0);
        let y_bin = ndarray::array![0.0, 1.0, 1.0];
        let mu_bin = ndarray::array![0.3, 0.6, 0.9];
        assert_eq!(log_loss(&y_bin, &mu_bin, Some(&w)), 0.0);
    }
}
