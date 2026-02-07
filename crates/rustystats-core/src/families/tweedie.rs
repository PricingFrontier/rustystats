// =============================================================================
// Tweedie Family
// =============================================================================
//
// The Tweedie distribution is a family of distributions characterized by a
// power parameter p that determines the variance-mean relationship:
//
//     Var(Y) = φ × μ^p
//
// Special cases:
//   p = 0: Normal (Gaussian)
//   p = 1: Poisson
//   1 < p < 2: Compound Poisson-Gamma (most useful for insurance)
//   p = 2: Gamma
//   p = 3: Inverse Gaussian
//
// INSURANCE APPLICATION:
// ----------------------
// For pure premium modeling (frequency × severity), p ∈ (1, 2) is ideal:
// - Allows exact zeros (no claims)
// - Continuous positive values (claim amounts)
// - Single model instead of separate frequency/severity models
//
// Typical values: p = 1.5 to 1.9 for insurance data
//
// VARIANCE POWER PARAMETER:
// -------------------------
// The power p controls the relationship between variance and mean:
// - Lower p (closer to 1): More Poisson-like, more zeros expected
// - Higher p (closer to 2): More Gamma-like, fewer zeros expected
//
// =============================================================================

use ndarray::Array1;
use crate::families::Family;
use crate::links::{Link, LogLink};
use crate::constants::{MU_MIN_POSITIVE, ZERO_TOL};

/// Tweedie distribution family.
///
/// The Tweedie family is parameterized by a variance power `p`:
/// - `Var(Y) = φ × μ^p`
///
/// # Parameters
/// - `var_power`: The variance power parameter p (typically 1 < p < 2 for insurance)
///
/// # Example
/// ```
/// use rustystats_core::families::TweedieFamily;
/// use rustystats_core::families::Family;
/// use ndarray::array;
///
/// // Create Tweedie with p=1.5 (common for insurance)
/// let family = TweedieFamily::new(1.5);
///
/// let mu = array![1.0, 2.0, 4.0];
/// let variance = family.variance(&mu);
///
/// // Variance = μ^1.5
/// assert!((variance[0] - 1.0).abs() < 1e-10);  // 1^1.5 = 1
/// assert!((variance[1] - 2.828).abs() < 0.001);  // 2^1.5 ≈ 2.828
/// ```
#[derive(Debug, Clone)]
pub struct TweedieFamily {
    /// Variance power parameter (typically 1 < p < 2 for insurance)
    pub var_power: f64,
}

impl TweedieFamily {
    /// Create a new Tweedie family with specified variance power.
    ///
    /// # Arguments
    /// * `var_power` - The power parameter p. Common values:
    ///   - p = 1.5: Balanced between Poisson and Gamma
    ///   - p = 1.6-1.9: More Gamma-like, fewer zeros
    ///
    /// # Panics
    /// Panics if var_power is in (0, 1) as this range is not supported.
    pub fn new(var_power: f64) -> Self {
        // Tweedie is not defined for 0 < p < 1
        if var_power > 0.0 && var_power < 1.0 {
            panic!("Tweedie var_power must be <= 0 or >= 1, got {}", var_power);
        }
        TweedieFamily { var_power }
    }

    /// Create Tweedie with default power p=1.5 (good starting point for insurance)
    pub fn default_insurance() -> Self {
        TweedieFamily { var_power: 1.5 }
    }
}

impl Family for TweedieFamily {
    fn name(&self) -> &str {
        "Tweedie"
    }

    /// Variance function: V(μ) = μ^p
    fn variance(&self, mu: &Array1<f64>) -> Array1<f64> {
        mu.mapv(|m| m.powf(self.var_power).max(MU_MIN_POSITIVE))
    }

    /// Unit deviance for Tweedie distribution.
    ///
    /// The formula depends on the value of p:
    /// - p = 0: (y - μ)²
    /// - p = 1: 2 × [y × log(y/μ) - (y - μ)]
    /// - p = 2: 2 × [(y - μ)/μ - log(y/μ)]
    /// - Otherwise: 2 × [y^(2-p)/((1-p)(2-p)) - y×μ^(1-p)/(1-p) + μ^(2-p)/(2-p)]
    fn unit_deviance(&self, y: &Array1<f64>, mu: &Array1<f64>) -> Array1<f64> {
        let p = self.var_power;

        ndarray::Zip::from(y)
            .and(mu)
            .map_collect(|&yi, &mui| {
                // Ensure positive values for numerical stability
                let yi_safe = yi.max(0.0);
                let mui_safe = mui.max(MU_MIN_POSITIVE);

                if (p - 0.0).abs() < ZERO_TOL {
                    // p = 0: Gaussian
                    let diff = yi_safe - mui_safe;
                    diff * diff
                } else if (p - 1.0).abs() < ZERO_TOL {
                    // p = 1: Poisson
                    if yi_safe == 0.0 {
                        2.0 * mui_safe
                    } else {
                        2.0 * (yi_safe * (yi_safe / mui_safe).ln() - (yi_safe - mui_safe))
                    }
                } else if (p - 2.0).abs() < ZERO_TOL {
                    // p = 2: Gamma
                    2.0 * ((yi_safe - mui_safe) / mui_safe - (yi_safe / mui_safe).ln())
                } else {
                    // General case: 1 < p < 2 (compound Poisson-Gamma) or p > 2
                    let term1 = if yi_safe > 0.0 {
                        yi_safe.powf(2.0 - p) / ((1.0 - p) * (2.0 - p))
                    } else {
                        0.0
                    };
                    let term2 = yi_safe * mui_safe.powf(1.0 - p) / (1.0 - p);
                    let term3 = mui_safe.powf(2.0 - p) / (2.0 - p);

                    2.0 * (term1 - term2 + term3)
                }
            })
    }

    /// Default link for Tweedie is log link.
    ///
    /// This ensures μ > 0, which is required for Tweedie.
    fn default_link(&self) -> Box<dyn Link> {
        Box::new(LogLink)
    }

    /// Initialize μ from observed y values.
    ///
    /// For Tweedie, we need μ > 0, but y can be exactly 0.
    /// We use a weighted average of y and the mean to avoid zeros.
    fn initialize_mu(&self, y: &Array1<f64>) -> Array1<f64> {
        let y_mean = y.mean().unwrap_or(1.0).max(0.1);

        y.mapv(|yi| {
            // Weighted average of y and mean, ensuring positive
            let val = (yi + y_mean) / 2.0;
            val.max(0.001)  // Ensure strictly positive
        })
    }

    /// Check if μ values are valid (must be positive for Tweedie).
    fn is_valid_mu(&self, mu: &Array1<f64>) -> bool {
        mu.iter().all(|&m| m > 0.0)
    }
    
    fn clamp_mu(&self, mu: &Array1<f64>) -> Array1<f64> {
        use crate::constants::MU_MIN_POSITIVE;
        mu.mapv(|x| x.max(MU_MIN_POSITIVE))
    }
    
    fn is_log_link_default(&self) -> bool { true }
    
    /// Tweedie (1 < p < 2) with log link benefits from true Hessian weights.
    /// 
    /// Using the observed Hessian can significantly reduce IRLS iterations.
    fn use_true_hessian_weights(&self) -> bool {
        // Only enable for 1 < p < 2 (compound Poisson-Gamma) where Hessian is PD
        self.var_power > 1.0 && self.var_power < 2.0
    }
    
    /// For Tweedie with log link, the true Hessian weight is μ^(2-p).
    /// 
    /// Derivation: For Tweedie with variance V(μ) = μ^p and log link:
    ///   - The Hessian of the log-likelihood w.r.t. η involves μ^(2-p)
    ///   - This provides better curvature information than Fisher info
    /// 
    /// For p in (1, 2), this gives weights between μ (p=1, Poisson) and 1 (p=2, Gamma).
    fn true_hessian_weights(&self, mu: &Array1<f64>, _y: &Array1<f64>) -> Array1<f64> {
        let exp = 2.0 - self.var_power;
        mu.mapv(|m| m.powf(exp).max(1e-10))
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
    fn test_tweedie_variance_power_1_5() {
        let family = TweedieFamily::new(1.5);
        let mu = array![1.0, 4.0, 9.0];

        let variance = family.variance(&mu);

        // V(μ) = μ^1.5
        assert_abs_diff_eq!(variance[0], 1.0, epsilon = 1e-10);  // 1^1.5 = 1
        assert_abs_diff_eq!(variance[1], 8.0, epsilon = 1e-10);  // 4^1.5 = 8
        assert_abs_diff_eq!(variance[2], 27.0, epsilon = 1e-10); // 9^1.5 = 27
    }

    #[test]
    fn test_tweedie_reduces_to_poisson() {
        // When p=1, Tweedie should behave like Poisson
        let tweedie = TweedieFamily::new(1.0);
        let mu = array![1.0, 2.0, 3.0];

        let variance = tweedie.variance(&mu);

        // For Poisson, V(μ) = μ
        assert_abs_diff_eq!(variance[0], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(variance[1], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(variance[2], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_tweedie_reduces_to_gamma() {
        // When p=2, Tweedie should behave like Gamma
        let tweedie = TweedieFamily::new(2.0);
        let mu = array![1.0, 2.0, 3.0];

        let variance = tweedie.variance(&mu);

        // For Gamma, V(μ) = μ²
        assert_abs_diff_eq!(variance[0], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(variance[1], 4.0, epsilon = 1e-10);
        assert_abs_diff_eq!(variance[2], 9.0, epsilon = 1e-10);
    }

    #[test]
    fn test_tweedie_deviance_with_zeros() {
        // Tweedie (1 < p < 2) should handle exact zeros
        let family = TweedieFamily::new(1.5);
        let y = array![0.0, 0.0, 1.0, 2.0];
        let mu = array![0.5, 1.0, 1.0, 2.0];

        let deviance = family.unit_deviance(&y, &mu);

        // Should not panic and produce finite values
        assert!(deviance.iter().all(|&d| d.is_finite()));
        assert!(deviance.iter().all(|&d| d >= 0.0));
    }

    #[test]
    fn test_tweedie_deviance_perfect_fit() {
        let family = TweedieFamily::new(1.5);
        let y = array![1.0, 2.0, 3.0];
        let mu = array![1.0, 2.0, 3.0];  // Perfect fit

        let deviance = family.unit_deviance(&y, &mu);

        // Perfect fit should have near-zero deviance
        for d in deviance.iter() {
            assert_abs_diff_eq!(*d, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_tweedie_default_link() {
        let family = TweedieFamily::new(1.5);
        let link = family.default_link();

        // Default is log link
        let mu = array![1.0, 2.0];
        let eta = link.link(&mu);

        assert_abs_diff_eq!(eta[0], 0.0, epsilon = 1e-10);  // log(1) = 0
        assert_abs_diff_eq!(eta[1], 2.0_f64.ln(), epsilon = 1e-10);
    }

    #[test]
    fn test_tweedie_initialize_handles_zeros() {
        let family = TweedieFamily::new(1.5);
        let y = array![0.0, 0.0, 5.0, 10.0];

        let mu = family.initialize_mu(&y);

        // All μ should be positive
        assert!(mu.iter().all(|&m| m > 0.0));
    }

    #[test]
    fn test_tweedie_valid_mu() {
        let family = TweedieFamily::new(1.5);

        assert!(family.is_valid_mu(&array![0.1, 1.0, 10.0]));
        assert!(!family.is_valid_mu(&array![0.0, 1.0, 10.0]));  // Zero invalid
        assert!(!family.is_valid_mu(&array![-1.0, 1.0, 10.0])); // Negative invalid
    }

    #[test]
    #[should_panic]
    fn test_tweedie_invalid_power() {
        // p in (0, 1) is not valid for Tweedie
        let _family = TweedieFamily::new(0.5);
    }

    #[test]
    fn test_tweedie_name() {
        let family = TweedieFamily::new(1.5);
        assert_eq!(family.name(), "Tweedie");
    }

    #[test]
    fn test_tweedie_insurance_default() {
        let family = TweedieFamily::default_insurance();
        assert_abs_diff_eq!(family.var_power, 1.5, epsilon = 1e-10);
    }
}
