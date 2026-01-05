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

use ndarray::Array1;
use crate::constants::{MU_MIN_POSITIVE, ZERO_TOL};

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
            let weighted_sum: f64 = y.iter()
                .zip(mu.iter())
                .zip(w.iter())
                .map(|((&yi, &mui), &wi)| wi * (yi - mui).powi(2))
                .sum();
            weighted_sum / sum_w
        }
        None => {
            let sum: f64 = y.iter()
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
            let weighted_sum: f64 = y.iter()
                .zip(mu.iter())
                .zip(w.iter())
                .map(|((&yi, &mui), &wi)| wi * (yi - mui).abs())
                .sum();
            weighted_sum / sum_w
        }
        None => {
            let sum: f64 = y.iter()
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
pub fn poisson_deviance_loss(y: &Array1<f64>, mu: &Array1<f64>, weights: Option<&Array1<f64>>) -> f64 {
    let n = y.len();
    if n == 0 {
        return 0.0;
    }
    
    let unit_deviances: Vec<f64> = y.iter()
        .zip(mu.iter())
        .map(|(&yi, &mui)| {
            let mui_safe = mui.max(MU_MIN_POSITIVE);
            if yi == 0.0 {
                2.0 * mui_safe
            } else {
                2.0 * (yi * (yi / mui_safe).ln() - (yi - mui_safe))
            }
        })
        .collect();
    
    match weights {
        Some(w) => {
            let sum_w: f64 = w.sum();
            if sum_w == 0.0 {
                return 0.0;
            }
            let weighted_sum: f64 = unit_deviances.iter()
                .zip(w.iter())
                .map(|(&d, &wi)| wi * d)
                .sum();
            weighted_sum / sum_w
        }
        None => {
            unit_deviances.iter().sum::<f64>() / n as f64
        }
    }
}

/// Gamma Deviance Loss (mean unit deviance)
/// 
/// loss = 2 * mean((y - μ)/μ - log(y/μ))
/// 
/// Default loss for Gamma family.
pub fn gamma_deviance_loss(y: &Array1<f64>, mu: &Array1<f64>, weights: Option<&Array1<f64>>) -> f64 {
    let n = y.len();
    if n == 0 {
        return 0.0;
    }
    
    let unit_deviances: Vec<f64> = y.iter()
        .zip(mu.iter())
        .map(|(&yi, &mui)| {
            let yi_safe = yi.max(MU_MIN_POSITIVE);
            let mui_safe = mui.max(MU_MIN_POSITIVE);
            let ratio = yi_safe / mui_safe;
            2.0 * ((yi_safe - mui_safe) / mui_safe - ratio.ln())
        })
        .collect();
    
    match weights {
        Some(w) => {
            let sum_w: f64 = w.sum();
            if sum_w == 0.0 {
                return 0.0;
            }
            let weighted_sum: f64 = unit_deviances.iter()
                .zip(w.iter())
                .map(|(&d, &wi)| wi * d)
                .sum();
            weighted_sum / sum_w
        }
        None => {
            unit_deviances.iter().sum::<f64>() / n as f64
        }
    }
}

/// Binomial Deviance Loss (Log Loss / Cross-Entropy)
/// 
/// loss = -mean(y * log(μ) + (1-y) * log(1-μ))
/// 
/// Default loss for Binomial family.
pub fn log_loss(y: &Array1<f64>, mu: &Array1<f64>, weights: Option<&Array1<f64>>) -> f64 {
    let n = y.len();
    if n == 0 {
        return 0.0;
    }
    
    let unit_losses: Vec<f64> = y.iter()
        .zip(mu.iter())
        .map(|(&yi, &mui)| {
            let mui_safe = mui.max(1e-15).min(1.0 - 1e-15);
            -(yi * mui_safe.ln() + (1.0 - yi) * (1.0 - mui_safe).ln())
        })
        .collect();
    
    match weights {
        Some(w) => {
            let sum_w: f64 = w.sum();
            if sum_w == 0.0 {
                return 0.0;
            }
            let weighted_sum: f64 = unit_losses.iter()
                .zip(w.iter())
                .map(|(&l, &wi)| wi * l)
                .sum();
            weighted_sum / sum_w
        }
        None => {
            unit_losses.iter().sum::<f64>() / n as f64
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
pub fn tweedie_deviance_loss(
    y: &Array1<f64>, 
    mu: &Array1<f64>, 
    var_power: f64,
    weights: Option<&Array1<f64>>
) -> f64 {
    let n = y.len();
    if n == 0 {
        return 0.0;
    }
    
    let unit_deviances: Vec<f64> = y.iter()
        .zip(mu.iter())
        .map(|(&yi, &mui)| {
            let mui_safe = mui.max(MU_MIN_POSITIVE);
            tweedie_unit_deviance(yi, mui_safe, var_power)
        })
        .collect();
    
    match weights {
        Some(w) => {
            let sum_w: f64 = w.sum();
            if sum_w == 0.0 {
                return 0.0;
            }
            let weighted_sum: f64 = unit_deviances.iter()
                .zip(w.iter())
                .map(|(&d, &wi)| wi * d)
                .sum();
            weighted_sum / sum_w
        }
        None => {
            unit_deviances.iter().sum::<f64>() / n as f64
        }
    }
}

/// Compute Tweedie unit deviance for a single observation
fn tweedie_unit_deviance(y: f64, mu: f64, p: f64) -> f64 {
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
        2.0 * (
            y_safe.powf(2.0 - p) / ((1.0 - p) * (2.0 - p))
            - y_safe * mu.powf(1.0 - p) / (1.0 - p)
            + mu.powf(2.0 - p) / (2.0 - p)
        )
    }
}

/// Negative Binomial Deviance Loss
/// 
/// Includes theta (dispersion) parameter.
pub fn negbinomial_deviance_loss(
    y: &Array1<f64>, 
    mu: &Array1<f64>, 
    theta: f64,
    weights: Option<&Array1<f64>>
) -> f64 {
    let n = y.len();
    if n == 0 {
        return 0.0;
    }
    
    let unit_deviances: Vec<f64> = y.iter()
        .zip(mu.iter())
        .map(|(&yi, &mui)| {
            let mui_safe = mui.max(MU_MIN_POSITIVE);
            let yi_safe = yi.max(0.0);
            
            // NB deviance: 2 * (y * log(y/μ) - (y + θ) * log((y + θ)/(μ + θ)))
            let term1 = if yi_safe > 0.0 {
                yi_safe * (yi_safe / mui_safe).ln()
            } else {
                0.0
            };
            let term2 = (yi_safe + theta) * ((yi_safe + theta) / (mui_safe + theta)).ln();
            2.0 * (term1 - term2)
        })
        .collect();
    
    match weights {
        Some(w) => {
            let sum_w: f64 = w.sum();
            if sum_w == 0.0 {
                return 0.0;
            }
            let weighted_sum: f64 = unit_deviances.iter()
                .zip(w.iter())
                .map(|(&d, &wi)| wi * d)
                .sum();
            weighted_sum / sum_w
        }
        None => {
            unit_deviances.iter().sum::<f64>() / n as f64
        }
    }
}

/// Get the default loss function name for a family.
/// Panics on unknown family - callers should validate family names first.
pub fn default_loss_name(family: &str) -> &'static str {
    match family.to_lowercase().as_str() {
        "gaussian" | "normal" => "mse",
        "poisson" | "quasipoisson" => "poisson_deviance",
        "gamma" => "gamma_deviance",
        "binomial" | "quasibinomial" => "log_loss",
        "tweedie" => "tweedie_deviance",
        "negativebinomial" | "negbinomial" | "nb" => "negbinomial_deviance",
        other => panic!("Unknown family '{}' in default_loss_name", other),
    }
}

/// Compute the default loss for a given family.
/// Panics on unknown family - callers should validate family names first.
pub fn compute_family_loss(
    family: &str,
    y: &Array1<f64>,
    mu: &Array1<f64>,
    weights: Option<&Array1<f64>>,
    var_power: Option<f64>,
    theta: Option<f64>,
) -> f64 {
    match family.to_lowercase().as_str() {
        "gaussian" | "normal" => mse(y, mu, weights),
        "poisson" | "quasipoisson" => poisson_deviance_loss(y, mu, weights),
        "gamma" => gamma_deviance_loss(y, mu, weights),
        "binomial" | "quasibinomial" => log_loss(y, mu, weights),
        "tweedie" => tweedie_deviance_loss(y, mu, var_power.unwrap_or(1.5), weights),
        "negativebinomial" | "negbinomial" | "nb" => {
            negbinomial_deviance_loss(y, mu, theta.unwrap_or(1.0), weights)
        }
        other => panic!("Unknown family '{}' in compute_family_loss", other),
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use approx::assert_abs_diff_eq;

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
        let mu = array![2.0, 2.0];  // errors: 1, 0
        let w = array![1.0, 3.0];   // weight more on second
        
        let result = mse(&y, &mu, Some(&w));
        // (1*1 + 3*0) / 4 = 0.25
        assert_abs_diff_eq!(result, 0.25, epsilon = 1e-10);
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
    fn test_poisson_deviance_loss() {
        let y = array![0.0, 1.0, 2.0, 5.0];
        let mu = array![0.5, 1.0, 2.5, 4.0];
        
        let result = poisson_deviance_loss(&y, &mu, None);
        // Should be positive and reasonable
        assert!(result > 0.0);
        assert!(result < 1.0);
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
    fn test_gamma_deviance_loss() {
        let y = array![1.0, 2.0, 3.0, 4.0];
        let mu = array![1.0, 2.0, 3.0, 4.0];  // Perfect fit
        
        let result = gamma_deviance_loss(&y, &mu, None);
        // Perfect fit should have zero loss
        assert_abs_diff_eq!(result, 0.0, epsilon = 1e-10);
    }
}
