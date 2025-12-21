// =============================================================================
// Gamma Family
// =============================================================================
//
// The Gamma family is for POSITIVE CONTINUOUS data - values that must be > 0.
// This is the workhorse for claim SEVERITY (amount) modeling in insurance.
//
// PROPERTIES:
// -----------
// - Distribution: Y ~ Gamma(shape=α, scale=β) with mean μ = αβ
// - Variance function: V(μ) = μ² (variance proportional to mean squared)
// - Canonical link: Inverse (η = 1/μ), but LOG is more common in practice
// - Dispersion: φ = 1/α (the inverse of the shape parameter)
//
// KEY INSIGHT - CONSTANT COEFFICIENT OF VARIATION:
// ------------------------------------------------
// Since V(μ) = μ², the standard deviation is proportional to the mean:
//   SD(Y) = √(φ × μ²) = √φ × μ
//   
// The coefficient of variation (CV = SD/mean) is constant:
//   CV = √φ
//
// This is very realistic for claim amounts! A $1,000 claim might vary by
// $500, while a $100,000 claim might vary by $50,000 - same CV of 50%.
//
// WHY NOT GAUSSIAN FOR CLAIM AMOUNTS?
// -----------------------------------
// Gaussian assumes constant variance: a $1,000 claim and a $100,000 claim
// would have the same variance. This is unrealistic.
//
// Gamma's V(μ) = μ² assumption is much more appropriate for monetary amounts.
//
// LOG LINK VS INVERSE LINK:
// -------------------------
// - Inverse link (canonical): η = 1/μ, harder to interpret
// - Log link (common): η = log(μ), multiplicative interpretation
//
// In practice, the log link is almost always used because:
// 1. Coefficients have multiplicative interpretation (like frequency models)
// 2. Predictions are guaranteed positive
// 3. Easier to combine with Poisson frequency models
//
// =============================================================================

use ndarray::Array1;
use crate::links::{Link, LogLink};
use super::Family;

/// Gamma family for positive continuous data.
/// 
/// The standard choice for modeling claim severity in insurance.
/// 
/// # Example
/// ```
/// use rustystats_core::families::{Family, GammaFamily};
/// use ndarray::array;
/// 
/// let family = GammaFamily;
/// let mu = array![100.0, 1000.0, 10000.0];
/// let variance = family.variance(&mu);  // [10000, 1000000, 100000000]
/// ```
#[derive(Debug, Clone, Copy)]
pub struct GammaFamily;

impl Family for GammaFamily {
    fn name(&self) -> &str {
        "Gamma"
    }
    
    /// Variance function: V(μ) = μ²
    /// 
    /// This means larger claims have proportionally larger variance.
    /// The full variance is: Var(Y) = φ × μ² where φ is the dispersion.
    fn variance(&self, mu: &Array1<f64>) -> Array1<f64> {
        mu.mapv(|x| x * x)
    }
    
    /// Unit deviance: 2 × [-log(y/μ) + (y-μ)/μ]
    /// 
    /// Simplifies to: 2 × [(y-μ)/μ - log(y/μ)]
    fn unit_deviance(&self, y: &Array1<f64>, mu: &Array1<f64>) -> Array1<f64> {
        ndarray::Zip::from(y)
            .and(mu)
            .map_collect(|&yi, &mui| {
                // (y - μ) / μ - log(y/μ)
                let ratio = yi / mui;
                2.0 * ((yi - mui) / mui - ratio.ln())
            })
    }
    
    /// Default link for Gamma.
    /// 
    /// Note: The CANONICAL link is inverse (1/μ), but we use LOG because:
    /// - It's standard in actuarial practice
    /// - Better interpretability (multiplicative effects)
    /// - Consistent with Poisson frequency models
    fn default_link(&self) -> Box<dyn Link> {
        Box::new(LogLink)
    }
    
    /// Initialize μ to a smoothed version of y.
    /// 
    /// We need to ensure μ > 0. A simple approach is to use y directly
    /// (since y should be positive for Gamma) with a small floor.
    fn initialize_mu(&self, y: &Array1<f64>) -> Array1<f64> {
        // Use y, but ensure minimum value to avoid numerical issues
        let y_mean = y.mean().unwrap_or(1.0);
        let min_val = y_mean * 0.01;  // At least 1% of the mean
        
        y.mapv(|yi| yi.max(min_val))
    }
    
    /// μ must be strictly positive for Gamma.
    fn is_valid_mu(&self, mu: &Array1<f64>) -> bool {
        mu.iter().all(|&x| x > 0.0 && x.is_finite())
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
    fn test_gamma_variance() {
        let family = GammaFamily;
        let mu = array![10.0, 100.0, 1000.0];
        
        let var = family.variance(&mu);
        
        // V(μ) = μ²
        let expected = array![100.0, 10000.0, 1000000.0];
        assert_abs_diff_eq!(var, expected, epsilon = 1e-10);
    }
    
    #[test]
    fn test_gamma_constant_cv() {
        // Demonstrate that CV is constant
        let family = GammaFamily;
        let mu = array![10.0, 100.0, 1000.0];
        
        let var = family.variance(&mu);
        
        // CV² = Var / μ² = μ² / μ² = 1 (before dispersion scaling)
        // So the "unit CV²" is 1 for all μ
        for i in 0..mu.len() {
            let cv_squared = var[i] / (mu[i] * mu[i]);
            assert_abs_diff_eq!(cv_squared, 1.0, epsilon = 1e-10);
        }
    }
    
    #[test]
    fn test_gamma_unit_deviance_perfect_fit() {
        let family = GammaFamily;
        let y = array![100.0, 500.0, 1000.0];
        let mu = array![100.0, 500.0, 1000.0];  // Perfect predictions
        
        let dev = family.unit_deviance(&y, &mu);
        
        // Perfect fit: y/μ = 1, so deviance = 2×[0 - log(1)] = 0
        let expected = array![0.0, 0.0, 0.0];
        assert_abs_diff_eq!(dev, expected, epsilon = 1e-10);
    }
    
    #[test]
    fn test_gamma_unit_deviance() {
        let family = GammaFamily;
        let y = array![150.0];
        let mu = array![100.0];
        
        let dev = family.unit_deviance(&y, &mu);
        
        // 2 × [(150-100)/100 - log(150/100)]
        // = 2 × [0.5 - log(1.5)]
        // = 2 × [0.5 - 0.405...]
        // ≈ 0.189
        let expected = 2.0 * ((150.0 - 100.0) / 100.0 - (150.0_f64 / 100.0).ln());
        assert_abs_diff_eq!(dev[0], expected, epsilon = 1e-10);
    }
    
    #[test]
    fn test_gamma_default_link() {
        let family = GammaFamily;
        let link = family.default_link();
        
        // We use log link (not canonical inverse) for practical reasons
        assert_eq!(link.name(), "log");
    }
    
    #[test]
    fn test_gamma_initialize_positive() {
        let family = GammaFamily;
        let y = array![100.0, 500.0, 1000.0];
        
        let mu_init = family.initialize_mu(&y);
        
        // All values should be positive
        assert!(mu_init.iter().all(|&x| x > 0.0));
    }
    
    #[test]
    fn test_gamma_valid_mu() {
        let family = GammaFamily;
        
        // Positive values are valid
        assert!(family.is_valid_mu(&array![0.1, 1.0, 1000.0]));
        
        // Zero is NOT valid
        assert!(!family.is_valid_mu(&array![0.0, 1.0]));
        
        // Negative is NOT valid
        assert!(!family.is_valid_mu(&array![-1.0, 1.0]));
    }
    
    #[test]
    fn test_gamma_actuarial_interpretation() {
        // Demonstrate actuarial interpretation
        // If φ = 0.25 (estimated from data), then CV = √0.25 = 0.5 (50%)
        // This means claims typically vary by about 50% around their mean
        
        let family = GammaFamily;
        let dispersion = 0.25;
        
        // For a claim with expected value $1000
        let mu = array![1000.0];
        let unit_var = family.variance(&mu)[0];  // μ² = 1,000,000
        
        let actual_var = dispersion * unit_var;  // 250,000
        let actual_sd = actual_var.sqrt();       // 500
        let cv = actual_sd / mu[0];              // 0.5 = 50%
        
        assert_abs_diff_eq!(cv, 0.5, epsilon = 1e-10);
    }
}
