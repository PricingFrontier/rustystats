// =============================================================================
// Poisson Family
// =============================================================================
//
// The Poisson family is for COUNT DATA (non-negative integers: 0, 1, 2, ...).
// This is the workhorse of claim frequency modeling in insurance.
//
// PROPERTIES:
// -----------
// - Distribution: Y ~ Poisson(μ)
// - Variance function: V(μ) = μ (variance equals the mean!)
// - Canonical link: Log (η = log(μ))
// - Dispersion: φ = 1 (fixed, not estimated)
//
// KEY INSIGHT - VARIANCE = MEAN:
// ------------------------------
// For Poisson data, if the average claim count is 0.1 claims/year,
// the variance is also 0.1. If average is 2 claims/year, variance is 2.
// This is called "equidispersion".
//
// OVERDISPERSION:
// ---------------
// Real data often has MORE variance than Poisson predicts (overdispersion).
// Signs of overdispersion: Pearson chi-square / df >> 1
// Solutions: Use Negative Binomial family, or quasi-Poisson (adjust SE's)
//
// WHEN TO USE:
// ------------
// - Claim counts (number of claims per policy)
// - Event counts (number of accidents, hospitalizations)
// - Any count of independent events in a fixed period/area
//
// EXPOSURE:
// ---------
// Often used with an "exposure" term (e.g., policy years, miles driven).
// The model becomes: E(Y) = μ = exposure × exp(Xβ)
// Or equivalently: log(μ) = log(exposure) + Xβ
// where log(exposure) is an "offset" (coefficient fixed at 1).
//
// =============================================================================

use ndarray::Array1;
use crate::links::{Link, LogLink};
use super::Family;

/// Poisson family for count data.
/// 
/// The go-to family for modeling claim frequency in insurance.
/// 
/// # Example
/// ```
/// use rustystats_core::families::{Family, PoissonFamily};
/// use ndarray::array;
/// 
/// let family = PoissonFamily;
/// let mu = array![0.5, 1.0, 2.0];
/// let variance = family.variance(&mu);  // Same as mu: [0.5, 1.0, 2.0]
/// ```
#[derive(Debug, Clone, Copy)]
pub struct PoissonFamily;

impl Family for PoissonFamily {
    fn name(&self) -> &str {
        "Poisson"
    }
    
    /// Variance function: V(μ) = μ
    /// 
    /// This is the defining characteristic of Poisson: variance equals mean.
    /// Higher expected counts have higher variance.
    fn variance(&self, mu: &Array1<f64>) -> Array1<f64> {
        mu.clone()
    }
    
    /// Unit deviance: 2 × [y × log(y/μ) - (y - μ)]
    /// 
    /// This is derived from the Poisson log-likelihood.
    /// Note: When y = 0, the y × log(y/μ) term is defined as 0.
    fn unit_deviance(&self, y: &Array1<f64>, mu: &Array1<f64>) -> Array1<f64> {
        // For each observation, compute the unit deviance
        ndarray::Zip::from(y)
            .and(mu)
            .map_collect(|&yi, &mui| {
                if yi == 0.0 {
                    // When y=0: 2 × [0 - (0 - μ)] = 2μ
                    2.0 * mui
                } else {
                    // General case: 2 × [y × log(y/μ) - (y - μ)]
                    2.0 * (yi * (yi / mui).ln() - (yi - mui))
                }
            })
    }
    
    /// The canonical link for Poisson is the log link.
    /// 
    /// This ensures predictions are always positive and gives
    /// multiplicative interpretation to coefficients.
    fn default_link(&self) -> Box<dyn Link> {
        Box::new(LogLink)
    }
    
    /// Initialize μ to a smoothed version of y.
    /// 
    /// We can't use y directly because:
    /// - log(0) = -∞, which would break things
    /// - Starting too close to 0 can cause numerical issues
    /// 
    /// A common approach: μ_init = (y + ȳ) / 2, or μ_init = max(y, small_value)
    fn initialize_mu(&self, y: &Array1<f64>) -> Array1<f64> {
        // Use (y + 0.1) as a simple starting point
        // This ensures μ > 0 even when y = 0
        y.mapv(|yi| (yi + 0.1).max(0.1))
    }
    
    /// μ must be strictly positive for Poisson.
    fn is_valid_mu(&self, mu: &Array1<f64>) -> bool {
        mu.iter().all(|&x| x > 0.0 && x.is_finite())
    }
    
    fn clamp_mu(&self, mu: &Array1<f64>) -> Array1<f64> {
        use crate::constants::MU_MIN_POSITIVE;
        mu.mapv(|x| x.max(MU_MIN_POSITIVE))
    }
    
    fn fixed_dispersion(&self) -> bool { true }
    
    fn is_log_link_default(&self) -> bool { true }
    
    fn log_likelihood(&self, y: &Array1<f64>, mu: &Array1<f64>, _scale: f64, weights: Option<&Array1<f64>>) -> f64 {
        crate::diagnostics::log_likelihood_poisson(y, mu, weights)
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
    fn test_poisson_variance() {
        let family = PoissonFamily;
        let mu = array![0.5, 1.0, 2.0, 10.0];
        
        // Variance equals mean for Poisson
        let var = family.variance(&mu);
        assert_abs_diff_eq!(var, mu, epsilon = 1e-10);
    }
    
    #[test]
    fn test_poisson_unit_deviance_perfect_fit() {
        let family = PoissonFamily;
        let y = array![1.0, 2.0, 3.0];
        let mu = array![1.0, 2.0, 3.0];  // Perfect predictions
        
        let dev = family.unit_deviance(&y, &mu);
        
        // Perfect fit: y = μ, so deviance should be 0
        // 2 × [y × log(1) - 0] = 0
        let expected = array![0.0, 0.0, 0.0];
        assert_abs_diff_eq!(dev, expected, epsilon = 1e-10);
    }
    
    #[test]
    fn test_poisson_unit_deviance_zero_count() {
        let family = PoissonFamily;
        let y = array![0.0];
        let mu = array![1.0];
        
        let dev = family.unit_deviance(&y, &mu);
        
        // When y=0: deviance = 2 × μ = 2.0
        assert_abs_diff_eq!(dev[0], 2.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_poisson_unit_deviance_general() {
        let family = PoissonFamily;
        let y = array![5.0];
        let mu = array![3.0];
        
        let dev = family.unit_deviance(&y, &mu);
        
        // 2 × [5 × log(5/3) - (5 - 3)]
        // = 2 × [5 × 0.5108... - 2]
        // = 2 × [2.5541... - 2]
        // = 2 × 0.5541...
        // ≈ 1.108
        let expected = 2.0 * (5.0 * (5.0_f64 / 3.0).ln() - (5.0 - 3.0));
        assert_abs_diff_eq!(dev[0], expected, epsilon = 1e-10);
    }
    
    #[test]
    fn test_poisson_default_link() {
        let family = PoissonFamily;
        let link = family.default_link();
        
        assert_eq!(link.name(), "log");
    }
    
    #[test]
    fn test_poisson_initialize_handles_zeros() {
        let family = PoissonFamily;
        let y = array![0.0, 0.0, 1.0, 5.0];
        
        let mu_init = family.initialize_mu(&y);
        
        // All values should be positive
        assert!(mu_init.iter().all(|&x| x > 0.0));
    }
    
    #[test]
    fn test_poisson_valid_mu() {
        let family = PoissonFamily;
        
        // Positive values are valid
        assert!(family.is_valid_mu(&array![0.1, 1.0, 10.0]));
        
        // Zero is NOT valid (would cause log(0) = -∞)
        assert!(!family.is_valid_mu(&array![0.0, 1.0]));
        
        // Negative is NOT valid
        assert!(!family.is_valid_mu(&array![-1.0, 1.0]));
    }
}
