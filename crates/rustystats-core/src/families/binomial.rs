// =============================================================================
// Binomial Family
// =============================================================================
//
// The Binomial family is for BINARY or PROPORTION data.
// - Binary: 0 or 1 (did they claim? yes/no)
// - Proportion: k successes out of n trials (proportion who claimed)
//
// This is the foundation of LOGISTIC REGRESSION.
//
// PROPERTIES:
// -----------
// - Distribution: Y ~ Binomial(n, μ) where μ is the probability
// - Variance function: V(μ) = μ(1-μ)
// - Canonical link: Logit (η = log(μ/(1-μ)))
// - Dispersion: φ = 1 (fixed)
//
// UNDERSTANDING THE VARIANCE FUNCTION:
// ------------------------------------
// V(μ) = μ(1-μ) means:
// - At μ = 0.5: V = 0.25 (maximum variance - most uncertainty)
// - At μ = 0.1: V = 0.09 (less variance)
// - At μ = 0.01: V = 0.0099 (even less - rare events have less variability)
//
// This makes intuitive sense: when something almost always (or never)
// happens, there's not much variation in the outcomes.
//
// WHEN TO USE:
// ------------
// - Modeling claim probability (did they have a claim? 0/1)
// - Modeling lapse/retention (did they renew? 0/1)
// - Modeling conversion rates (did they buy? 0/1)
// - Any yes/no outcome
//
// ODDS AND ODDS RATIOS:
// ---------------------
// With logit link, coefficients have an odds ratio interpretation:
// - If β = 0.5, then exp(0.5) ≈ 1.65
// - This means "1.65 times the odds" for a 1-unit increase in X
//
// =============================================================================

use ndarray::Array1;
use crate::links::{Link, LogitLink};
use super::Family;

/// Binomial family for binary or proportion data.
/// 
/// Use this for logistic regression (modeling probabilities).
/// 
/// # Example
/// ```
/// use rustystats_core::families::{Family, BinomialFamily};
/// use ndarray::array;
/// 
/// let family = BinomialFamily;
/// let mu = array![0.2, 0.5, 0.8];
/// let variance = family.variance(&mu);  // [0.16, 0.25, 0.16]
/// ```
#[derive(Debug, Clone, Copy)]
pub struct BinomialFamily;

impl Family for BinomialFamily {
    fn name(&self) -> &str {
        "Binomial"
    }
    
    /// Variance function: V(μ) = μ(1-μ)
    /// 
    /// Maximum at μ = 0.5, approaches 0 as μ approaches 0 or 1.
    fn variance(&self, mu: &Array1<f64>) -> Array1<f64> {
        mu.mapv(|p| p * (1.0 - p))
    }
    
    /// Unit deviance for binomial: 2 × [y×log(y/μ) + (1-y)×log((1-y)/(1-μ))]
    /// 
    /// This handles the edge cases where y = 0 or y = 1.
    /// For binary data (y ∈ {0, 1}), this simplifies nicely.
    /// 
    /// OPTIMIZATION: Uses numerically stable formulations to avoid overflow/underflow
    /// for extreme probability values (μ very close to 0 or 1).
    fn unit_deviance(&self, y: &Array1<f64>, mu: &Array1<f64>) -> Array1<f64> {
        ndarray::Zip::from(y)
            .and(mu)
            .map_collect(|&yi, &mui| {
                // Clamp μ to avoid log(0) and division by zero
                let mui_safe = mui.max(1e-15).min(1.0 - 1e-15);
                
                let mut dev = 0.0;
                
                // First term: y × log(y/μ)
                // Only contributes when y > 0
                // Use log(y) - log(μ) for better numerical stability
                if yi > 0.0 {
                    // For y=1 (common case), this is -log(μ)
                    if yi >= 1.0 - 1e-15 {
                        dev += -mui_safe.ln();
                    } else {
                        dev += yi * (yi.ln() - mui_safe.ln());
                    }
                }
                
                // Second term: (1-y) × log((1-y)/(1-μ))
                // Only contributes when y < 1
                // Use log1p for better precision when μ or y is close to 1
                if yi < 1.0 - 1e-15 {
                    let one_minus_y = 1.0 - yi;
                    let one_minus_mu = 1.0 - mui_safe;
                    
                    // For y=0 (common case), this is -log(1-μ)
                    if yi <= 1e-15 {
                        // Use log1p(-μ) for better precision when μ is small
                        dev += -(-mui_safe).ln_1p().max(-1e10);
                    } else {
                        dev += one_minus_y * (one_minus_y.ln() - one_minus_mu.ln());
                    }
                }
                
                2.0 * dev.max(0.0)  // Ensure non-negative
            })
    }
    
    /// The canonical link for Binomial is the logit link.
    fn default_link(&self) -> Box<dyn Link> {
        Box::new(LogitLink)
    }
    
    /// Initialize μ to a smoothed proportion.
    /// 
    /// For binary data (0/1), we can't use y directly because:
    /// - logit(0) = -∞
    /// - logit(1) = +∞
    /// 
    /// We use a common trick: shrink y toward 0.5 slightly.
    fn initialize_mu(&self, y: &Array1<f64>) -> Array1<f64> {
        // Formula: (y + 0.5) / 2 = y/2 + 0.25
        // This maps: 0 -> 0.25, 0.5 -> 0.5, 1 -> 0.75
        // Keeps values safely away from 0 and 1
        y.mapv(|yi| (yi + 0.5) / 2.0)
    }
    
    /// μ must be strictly between 0 and 1 for Binomial.
    fn is_valid_mu(&self, mu: &Array1<f64>) -> bool {
        mu.iter().all(|&x| x > 0.0 && x < 1.0 && x.is_finite())
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
    fn test_binomial_variance() {
        let family = BinomialFamily;
        let mu = array![0.2, 0.5, 0.8];
        
        let var = family.variance(&mu);
        
        // V(0.2) = 0.2 × 0.8 = 0.16
        // V(0.5) = 0.5 × 0.5 = 0.25 (maximum)
        // V(0.8) = 0.8 × 0.2 = 0.16
        let expected = array![0.16, 0.25, 0.16];
        assert_abs_diff_eq!(var, expected, epsilon = 1e-10);
    }
    
    #[test]
    fn test_binomial_variance_symmetry() {
        let family = BinomialFamily;
        
        // V(p) should equal V(1-p)
        let mu1 = array![0.3];
        let mu2 = array![0.7];
        
        let var1 = family.variance(&mu1);
        let var2 = family.variance(&mu2);
        
        assert_abs_diff_eq!(var1, var2, epsilon = 1e-10);
    }
    
    #[test]
    fn test_binomial_unit_deviance_perfect_fit() {
        let family = BinomialFamily;
        let y = array![0.3, 0.5, 0.7];
        let mu = array![0.3, 0.5, 0.7];  // Perfect predictions
        
        let dev = family.unit_deviance(&y, &mu);
        
        // Perfect fit should have zero deviance
        let expected = array![0.0, 0.0, 0.0];
        assert_abs_diff_eq!(dev, expected, epsilon = 1e-10);
    }
    
    #[test]
    fn test_binomial_unit_deviance_binary() {
        let family = BinomialFamily;
        
        // Binary outcome y=1, predicted μ=0.8
        let y = array![1.0];
        let mu = array![0.8];
        
        let dev = family.unit_deviance(&y, &mu);
        
        // 2 × [1 × log(1/0.8) + 0 × log(...)] = 2 × log(1.25)
        let expected = 2.0 * (1.0 / 0.8_f64).ln();
        assert_abs_diff_eq!(dev[0], expected, epsilon = 1e-10);
    }
    
    #[test]
    fn test_binomial_unit_deviance_binary_zero() {
        let family = BinomialFamily;
        
        // Binary outcome y=0, predicted μ=0.2
        let y = array![0.0];
        let mu = array![0.2];
        
        let dev = family.unit_deviance(&y, &mu);
        
        // 2 × [0 × log(...) + 1 × log(1/0.8)] = 2 × log(1.25)
        let expected = 2.0 * (1.0 / 0.8_f64).ln();
        assert_abs_diff_eq!(dev[0], expected, epsilon = 1e-10);
    }
    
    #[test]
    fn test_binomial_default_link() {
        let family = BinomialFamily;
        let link = family.default_link();
        
        assert_eq!(link.name(), "logit");
    }
    
    #[test]
    fn test_binomial_initialize_binary() {
        let family = BinomialFamily;
        let y = array![0.0, 1.0, 0.0, 1.0];
        
        let mu_init = family.initialize_mu(&y);
        
        // All values should be in (0, 1)
        assert!(mu_init.iter().all(|&x| x > 0.0 && x < 1.0));
        
        // 0 -> 0.25, 1 -> 0.75
        assert_abs_diff_eq!(mu_init[0], 0.25, epsilon = 1e-10);
        assert_abs_diff_eq!(mu_init[1], 0.75, epsilon = 1e-10);
    }
    
    #[test]
    fn test_binomial_valid_mu() {
        let family = BinomialFamily;
        
        // Values strictly between 0 and 1 are valid
        assert!(family.is_valid_mu(&array![0.1, 0.5, 0.9]));
        
        // 0 and 1 are NOT valid (boundary issues)
        assert!(!family.is_valid_mu(&array![0.0, 0.5]));
        assert!(!family.is_valid_mu(&array![0.5, 1.0]));
        
        // Outside [0,1] is NOT valid
        assert!(!family.is_valid_mu(&array![-0.1, 0.5]));
        assert!(!family.is_valid_mu(&array![0.5, 1.1]));
    }
}
