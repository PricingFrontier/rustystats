// =============================================================================
// Gaussian (Normal) Family
// =============================================================================
//
// The Gaussian family is for continuous data that can take any real value.
// This is standard linear regression when used with the identity link.
//
// PROPERTIES:
// -----------
// - Distribution: Y ~ Normal(μ, σ²)
// - Variance function: V(μ) = 1 (constant variance)
// - Canonical link: Identity (η = μ)
// - Dispersion: σ² (estimated from residuals)
//
// WHEN TO USE:
// ------------
// - Continuous response with approximately constant variance
// - Residuals should be roughly normally distributed
// - Standard linear regression
//
// LIMITATIONS:
// ------------
// - Assumes constant variance (homoscedasticity)
// - Not ideal for strictly positive data (use Gamma instead)
// - Not ideal for bounded data
//
// ACTUARIAL EXAMPLE:
// ------------------
// Modeling log-transformed claim amounts (though Gamma on raw amounts
// is often preferred). Also useful for modeling standardized scores.
//
// =============================================================================

use ndarray::Array1;
use crate::links::{Link, IdentityLink};
use super::Family;

/// Gaussian (Normal) family for continuous response data.
/// 
/// This is the foundation of ordinary least squares (OLS) regression.
/// 
/// # Example
/// ```
/// use rustystats_core::families::{Family, GaussianFamily};
/// use ndarray::array;
/// 
/// let family = GaussianFamily;
/// let mu = array![1.0, 2.0, 3.0];
/// let variance = family.variance(&mu);  // All ones
/// ```
#[derive(Debug, Clone, Copy)]
pub struct GaussianFamily;

impl Family for GaussianFamily {
    fn name(&self) -> &str {
        "Gaussian"
    }
    
    /// Variance function: V(μ) = 1
    /// 
    /// The Gaussian family has constant variance - it doesn't depend on μ.
    /// This is the homoscedasticity assumption of linear regression.
    fn variance(&self, mu: &Array1<f64>) -> Array1<f64> {
        Array1::ones(mu.len())
    }
    
    /// Unit deviance: (y - μ)²
    /// 
    /// For Gaussian, the deviance is simply the sum of squared residuals.
    /// This connects GLM deviance to the familiar RSS in linear regression.
    fn unit_deviance(&self, y: &Array1<f64>, mu: &Array1<f64>) -> Array1<f64> {
        let residuals = y - mu;
        &residuals * &residuals  // Element-wise squaring
    }
    
    /// The canonical link for Gaussian is the identity link.
    fn default_link(&self) -> Box<dyn Link> {
        Box::new(IdentityLink)
    }
    
    /// Initialize μ to the observed values.
    /// 
    /// For Gaussian, starting at y is natural. We add a tiny bit of smoothing
    /// to avoid any numerical issues with exact zeros.
    fn initialize_mu(&self, y: &Array1<f64>) -> Array1<f64> {
        y.clone()
    }
    
    /// Any finite value is valid for Gaussian.
    fn is_valid_mu(&self, mu: &Array1<f64>) -> bool {
        mu.iter().all(|&x| x.is_finite())
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
    fn test_gaussian_variance() {
        let family = GaussianFamily;
        let mu = array![1.0, 2.0, 3.0, 100.0, -5.0];
        
        let var = family.variance(&mu);
        
        // Variance should be 1 everywhere
        let expected = array![1.0, 1.0, 1.0, 1.0, 1.0];
        assert_abs_diff_eq!(var, expected, epsilon = 1e-10);
    }
    
    #[test]
    fn test_gaussian_unit_deviance() {
        let family = GaussianFamily;
        let y = array![1.0, 2.0, 3.0];
        let mu = array![1.5, 2.0, 2.5];
        
        let dev = family.unit_deviance(&y, &mu);
        
        // (1-1.5)² = 0.25, (2-2)² = 0, (3-2.5)² = 0.25
        let expected = array![0.25, 0.0, 0.25];
        assert_abs_diff_eq!(dev, expected, epsilon = 1e-10);
    }
    
    #[test]
    fn test_gaussian_total_deviance() {
        let family = GaussianFamily;
        let y = array![1.0, 2.0, 3.0];
        let mu = array![1.0, 2.0, 3.0];  // Perfect fit
        
        let dev = family.deviance(&y, &mu, None);
        
        // Perfect fit should have zero deviance
        assert_abs_diff_eq!(dev, 0.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_gaussian_weighted_deviance() {
        let family = GaussianFamily;
        let y = array![1.0, 2.0];
        let mu = array![2.0, 2.0];  // First obs: residual=1, Second: residual=0
        let weights = array![2.0, 1.0];
        
        // Unit deviances: [1.0, 0.0]
        // Weighted: 2.0 * 1.0 + 1.0 * 0.0 = 2.0
        let dev = family.deviance(&y, &mu, Some(&weights));
        assert_abs_diff_eq!(dev, 2.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_gaussian_default_link() {
        let family = GaussianFamily;
        let link = family.default_link();
        
        assert_eq!(link.name(), "identity");
    }
    
    #[test]
    fn test_gaussian_valid_mu() {
        let family = GaussianFamily;
        
        // Regular values are valid
        assert!(family.is_valid_mu(&array![1.0, -1.0, 0.0, 100.0]));
        
        // Infinity is not valid
        assert!(!family.is_valid_mu(&array![1.0, f64::INFINITY]));
        
        // NaN is not valid
        assert!(!family.is_valid_mu(&array![1.0, f64::NAN]));
    }
}
