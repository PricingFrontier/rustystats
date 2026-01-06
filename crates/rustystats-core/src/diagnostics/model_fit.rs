// =============================================================================
// Model Fit Statistics
// =============================================================================
//
// This module provides statistics for assessing and comparing GLM models:
//
// LOG-LIKELIHOOD:
// ---------------
// The log of the probability of observing the data given the model.
// Higher (less negative) is better. Used to compute AIC/BIC.
//
// AIC (Akaike Information Criterion):
// -----------------------------------
// AIC = -2ℓ + 2p
// Balances fit (likelihood) against complexity (number of parameters).
// Lower is better. Use for model comparison.
//
// BIC (Bayesian Information Criterion):
// -------------------------------------
// BIC = -2ℓ + p×log(n)
// Like AIC but penalizes complexity more strongly for large samples.
// Lower is better.
//
// NULL DEVIANCE:
// --------------
// Deviance of an intercept-only model. Measures total variation in y.
// Compare to residual deviance to assess how much variation is explained.
//
// PSEUDO R-SQUARED:
// -----------------
// Various measures that mimic R² for non-Gaussian models.
// Calculated from null and residual deviance.
//
// =============================================================================

use ndarray::Array1;
use std::f64::consts::PI;
use crate::constants::{MU_MIN_PROBABILITY, MU_MAX_PROBABILITY, MU_MIN_POSITIVE};

// =============================================================================
// Log-Likelihood Functions
// =============================================================================

/// Log-likelihood for Gaussian (Normal) family.
///
/// ℓ = -½ Σ[(y-μ)²/φ + log(2πφ)]
///
/// # Arguments
/// * `y` - Observed response values
/// * `mu` - Fitted mean values
/// * `scale` - Dispersion parameter (σ²)
/// * `weights` - Optional observation weights
///
/// # Returns
/// Total log-likelihood
pub fn log_likelihood_gaussian(
    y: &Array1<f64>,
    mu: &Array1<f64>,
    scale: f64,
    weights: Option<&Array1<f64>>,
) -> f64 {
    let n = y.len() as f64;
    let sum_wt = weights.map_or(n, |w| w.sum());
    
    // Sum of squared residuals
    let ss: f64 = ndarray::Zip::from(y)
        .and(mu)
        .fold(0.0, |acc, &yi, &mui| {
            let diff = yi - mui;
            acc + diff * diff
        });
    
    // If weighted, scale the sum
    let ss_weighted = match weights {
        Some(w) => {
            ndarray::Zip::from(y)
                .and(mu)
                .and(w)
                .fold(0.0, |acc, &yi, &mui, &wi| {
                    let diff = yi - mui;
                    acc + wi * diff * diff
                })
        }
        None => ss,
    };
    
    // Log-likelihood
    -0.5 * (ss_weighted / scale + sum_wt * (2.0 * PI * scale).ln())
}

/// Log-likelihood for Poisson family.
///
/// ℓ = Σ[y×log(μ) - μ - log(y!)]
///
/// # Arguments
/// * `y` - Observed response values (counts)
/// * `mu` - Fitted mean values
/// * `weights` - Optional observation weights
///
/// # Returns
/// Total log-likelihood
///
/// # Note
/// The log(y!) term is constant given the data and can be omitted
/// for model comparison, but we include it for completeness.
pub fn log_likelihood_poisson(
    y: &Array1<f64>,
    mu: &Array1<f64>,
    weights: Option<&Array1<f64>>,
) -> f64 {
    use statrs::function::gamma::ln_gamma;
    
    let contributions: Array1<f64> = ndarray::Zip::from(y)
        .and(mu)
        .map_collect(|&yi, &mui| {
            // y × log(μ) - μ - log(y!)
            // log(y!) = ln_gamma(y + 1)
            let log_factorial = ln_gamma(yi + 1.0);
            yi * mui.ln() - mui - log_factorial
        });
    
    match weights {
        Some(w) => (&contributions * w).sum(),
        None => contributions.sum(),
    }
}

/// Log-likelihood for Binomial family (binary case).
///
/// ℓ = Σ[y×log(μ) + (1-y)×log(1-μ)]
///
/// # Arguments
/// * `y` - Observed response values (0/1 or proportions)
/// * `mu` - Fitted probabilities
/// * `weights` - Optional observation weights
///
/// # Returns
/// Total log-likelihood
pub fn log_likelihood_binomial(
    y: &Array1<f64>,
    mu: &Array1<f64>,
    weights: Option<&Array1<f64>>,
) -> f64 {
    let contributions: Array1<f64> = ndarray::Zip::from(y)
        .and(mu)
        .map_collect(|&yi, &mui| {
            // Clamp μ to avoid log(0)
            let mui_safe = mui.max(MU_MIN_PROBABILITY).min(MU_MAX_PROBABILITY);
            
            // y × log(μ) + (1-y) × log(1-μ)
            let ll = if yi > 0.0 {
                yi * mui_safe.ln()
            } else {
                0.0
            };
            
            let ll = if yi < 1.0 {
                ll + (1.0 - yi) * (1.0 - mui_safe).ln()
            } else {
                ll
            };
            
            ll
        });
    
    match weights {
        Some(w) => (&contributions * w).sum(),
        None => contributions.sum(),
    }
}

/// Log-likelihood for Gamma family.
///
/// For Gamma with shape α and scale θ where μ = αθ and φ = 1/α:
/// ℓ_i = (α-1)·log(y) - α·y/μ + α·log(α/μ) - log(Γ(α))
///
/// This matches the statsmodels implementation exactly.
///
/// # Arguments
/// * `y` - Observed response values (positive)
/// * `mu` - Fitted mean values
/// * `scale` - Dispersion parameter φ = 1/α (inverse of shape)
/// * `weights` - Optional observation weights
///
/// # Returns
/// Total log-likelihood
pub fn log_likelihood_gamma(
    y: &Array1<f64>,
    mu: &Array1<f64>,
    scale: f64,
    weights: Option<&Array1<f64>>,
) -> f64 {
    use statrs::function::gamma::ln_gamma;
    use crate::constants::MU_MIN_POSITIVE;
    
    // shape α = 1/scale = 1/φ
    let alpha = 1.0 / scale;
    
    let contributions: Array1<f64> = ndarray::Zip::from(y)
        .and(mu)
        .map_collect(|&yi, &mui| {
            // Floor y and mu to prevent log(0) issues
            // Note: Gamma requires y > 0, but we handle zeros gracefully
            let yi_safe = yi.max(MU_MIN_POSITIVE);
            let mui_safe = mui.max(MU_MIN_POSITIVE);
            
            // Full Gamma log-likelihood (statsmodels formula):
            // ℓ_i = (α-1)·log(y) - α·y/μ + α·log(α/μ) - log(Γ(α))
            //     = (α-1)·log(y) - α·y/μ + α·log(α) - α·log(μ) - log(Γ(α))
            let ll = (alpha - 1.0) * yi_safe.ln()
                   - alpha * yi_safe / mui_safe
                   + alpha * (alpha / mui_safe).ln()
                   - ln_gamma(alpha);
            ll
        });
    
    match weights {
        Some(w) => (&contributions * w).sum(),
        None => contributions.sum(),
    }
}

// =============================================================================
// Information Criteria
// =============================================================================

/// Compute Akaike Information Criterion.
///
/// AIC = -2ℓ + 2p
///
/// # Arguments
/// * `llf` - Log-likelihood value
/// * `n_params` - Number of estimated parameters (including intercept)
///
/// # Returns
/// AIC value (lower is better)
pub fn aic(llf: f64, n_params: usize) -> f64 {
    -2.0 * llf + 2.0 * (n_params as f64)
}

/// Compute Bayesian Information Criterion.
///
/// BIC = -2ℓ + p×log(n)
///
/// # Arguments
/// * `llf` - Log-likelihood value
/// * `n_params` - Number of estimated parameters
/// * `n_obs` - Number of observations
///
/// # Returns
/// BIC value (lower is better)
pub fn bic(llf: f64, n_params: usize, n_obs: usize) -> f64 {
    -2.0 * llf + (n_params as f64) * (n_obs as f64).ln()
}

// =============================================================================
// Null Deviance
// =============================================================================

/// Compute the null deviance (deviance of intercept-only model).
///
/// The null deviance measures how much variation there is in y
/// before accounting for predictors. It's used to compute pseudo R².
///
/// For most families, the intercept-only model predicts the weighted
/// mean of y for all observations.
///
/// # Arguments
/// * `y` - Observed response values
/// * `family_name` - Name of the family ("Gaussian", "Poisson", etc.)
/// * `weights` - Optional observation weights
///
/// # Returns
/// Null deviance value
pub fn null_deviance(
    y: &Array1<f64>,
    family_name: &str,
    weights: Option<&Array1<f64>>,
) -> f64 {
    // Compute weighted mean
    let (sum_y, sum_w) = match weights {
        Some(w) => {
            let sy: f64 = ndarray::Zip::from(y).and(w).fold(0.0, |acc, &yi, &wi| acc + yi * wi);
            let sw: f64 = w.sum();
            (sy, sw)
        }
        None => (y.sum(), y.len() as f64),
    };
    
    let y_mean = sum_y / sum_w;
    
    // Create mu array with constant mean
    let mu_null = Array1::from_elem(y.len(), y_mean);
    
    // Compute unit deviances based on family (case-insensitive matching)
    let unit_dev: Array1<f64> = match family_name.to_lowercase().as_str() {
        "gaussian" | "normal" => {
            // (y - μ)²
            ndarray::Zip::from(y)
                .and(&mu_null)
                .map_collect(|&yi, &mui| {
                    let diff = yi - mui;
                    diff * diff
                })
        }
        "poisson" | "quasipoisson" => {
            // 2 × [y × log(y/μ) - (y - μ)]
            ndarray::Zip::from(y)
                .and(&mu_null)
                .map_collect(|&yi, &mui| {
                    if yi == 0.0 {
                        2.0 * mui
                    } else {
                        2.0 * (yi * (yi / mui).ln() - (yi - mui))
                    }
                })
        }
        "binomial" | "quasibinomial" => {
            // 2 × [y × log(y/μ) + (1-y) × log((1-y)/(1-μ))]
            // Clamp mean for numerical stability
            let mu_safe = y_mean.max(MU_MIN_PROBABILITY).min(MU_MAX_PROBABILITY);
            let mu_null_safe = Array1::from_elem(y.len(), mu_safe);
            
            ndarray::Zip::from(y)
                .and(&mu_null_safe)
                .map_collect(|&yi, &mui| {
                    let mut dev = 0.0;
                    if yi > 0.0 {
                        dev += yi * (yi / mui).ln();
                    }
                    if yi < 1.0 {
                        dev += (1.0 - yi) * ((1.0 - yi) / (1.0 - mui)).ln();
                    }
                    2.0 * dev
                })
        }
        "gamma" => {
            // 2 × [(y - μ)/μ - log(y/μ)]
            // Floor y to prevent log(0) issues
            ndarray::Zip::from(y)
                .and(&mu_null)
                .map_collect(|&yi, &mui| {
                    let yi_safe = yi.max(MU_MIN_POSITIVE);
                    let mui_safe = mui.max(MU_MIN_POSITIVE);
                    let ratio = yi_safe / mui_safe;
                    2.0 * ((yi_safe - mui_safe) / mui_safe - ratio.ln())
                })
        }
        other => {
            panic!("Unknown family '{}' in null_deviance computation. \
                   Supported families: gaussian, poisson, binomial, gamma, quasipoisson, quasibinomial.", other)
        }
    };
    
    // Sum up (weighted if applicable)
    match weights {
        Some(w) => (&unit_dev * w).sum(),
        None => unit_dev.sum(),
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
    fn test_log_likelihood_gaussian() {
        let y = array![1.0, 2.0, 3.0];
        let mu = array![1.0, 2.0, 3.0];  // Perfect fit
        let scale = 1.0;
        
        let llf = log_likelihood_gaussian(&y, &mu, scale, None);
        
        // With perfect fit, SS = 0
        // ℓ = -0.5 × n × log(2πσ²) = -0.5 × 3 × log(2π)
        let expected = -0.5 * 3.0 * (2.0 * PI).ln();
        assert_abs_diff_eq!(llf, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_log_likelihood_poisson() {
        // Simple test: y = μ = 1 for all observations
        let y = array![1.0, 1.0, 1.0];
        let mu = array![1.0, 1.0, 1.0];
        
        let llf = log_likelihood_poisson(&y, &mu, None);
        
        // For y=μ=1: 1×log(1) - 1 - log(1!) = 0 - 1 - 0 = -1 per obs
        // Total: -3
        assert_abs_diff_eq!(llf, -3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_log_likelihood_binomial() {
        let y = array![1.0, 0.0];
        let mu = array![0.8, 0.2];
        
        let llf = log_likelihood_binomial(&y, &mu, None);
        
        // 1×log(0.8) + 0×log(0.2) + 0×log(0.2) + 1×log(0.8)
        // = log(0.8) + log(0.8) = 2×log(0.8)
        let expected = 0.8_f64.ln() + 0.8_f64.ln();
        assert_abs_diff_eq!(llf, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_aic() {
        let llf = -100.0;
        let n_params = 5;
        
        let aic_val = aic(llf, n_params);
        
        // AIC = -2×(-100) + 2×5 = 200 + 10 = 210
        assert_abs_diff_eq!(aic_val, 210.0, epsilon = 1e-10);
    }

    #[test]
    fn test_bic() {
        let llf = -100.0;
        let n_params = 5;
        let n_obs = 100;
        
        let bic_val = bic(llf, n_params, n_obs);
        
        // BIC = -2×(-100) + 5×log(100) = 200 + 5×4.605... ≈ 223.03
        let expected = 200.0 + 5.0 * 100.0_f64.ln();
        assert_abs_diff_eq!(bic_val, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_null_deviance_gaussian() {
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];
        
        let null_dev = null_deviance(&y, "Gaussian", None);
        
        // Mean = 3.0
        // Null deviance = Σ(y - 3)² = 4 + 1 + 0 + 1 + 4 = 10
        assert_abs_diff_eq!(null_dev, 10.0, epsilon = 1e-10);
    }

    #[test]
    fn test_null_deviance_poisson() {
        let y = array![0.0, 1.0, 2.0, 3.0, 4.0];
        
        let null_dev = null_deviance(&y, "Poisson", None);
        
        // Mean = 2.0
        // This is more complex to compute manually, but should be positive
        assert!(null_dev > 0.0);
    }

    #[test]
    fn test_null_deviance_weighted() {
        let y = array![1.0, 5.0];
        let weights = array![3.0, 1.0];  // More weight on first obs
        
        let null_dev = null_deviance(&y, "Gaussian", Some(&weights));
        
        // Weighted mean = (3×1 + 1×5) / 4 = 8/4 = 2.0
        // Null deviance = 3×(1-2)² + 1×(5-2)² = 3×1 + 1×9 = 12
        assert_abs_diff_eq!(null_dev, 12.0, epsilon = 1e-10);
    }
}
