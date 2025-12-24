// =============================================================================
// Negative Binomial Family
// =============================================================================
//
// The Negative Binomial (NB) family is for OVERDISPERSED COUNT DATA.
// It's an alternative to Poisson when variance exceeds the mean.
//
// PROPERTIES:
// -----------
// - Distribution: Y ~ NegBin(μ, θ) where μ is the mean, θ is dispersion
// - Variance function: V(μ) = μ + μ²/θ = μ(1 + μ/θ)
// - Canonical link: Log (η = log(μ))
// - Dispersion: θ (shape parameter, controls overdispersion)
//
// PARAMETERIZATION:
// -----------------
// We use the "NB2" parameterization (quadratic variance):
//   Var(Y) = μ + α×μ²  where α = 1/θ
//
// - When θ → ∞ (α → 0): Var(Y) → μ (Poisson)
// - Smaller θ (larger α): More overdispersion
//
// Common parameterizations:
// - NB1: Var(Y) = μ(1 + α) = μ + α×μ     (linear in μ)
// - NB2: Var(Y) = μ(1 + μ/θ) = μ + μ²/θ  (quadratic in μ) ← WE USE THIS
//
// COMPARISON TO QUASI-POISSON:
// ----------------------------
// | Aspect           | QuasiPoisson        | Negative Binomial    |
// |------------------|---------------------|----------------------|
// | Variance         | φ × μ               | μ + μ²/θ             |
// | True distribution| No                  | Yes                  |
// | Likelihood-based | No (quasi)          | Yes                  |
// | AIC/BIC valid    | Questionable        | Yes                  |
// | Prediction       | Less principled     | Proper intervals     |
//
// WHEN TO USE:
// ------------
// - Count data with overdispersion (variance > mean)
// - When you need valid likelihood-based inference (AIC, BIC)
// - When you want proper prediction intervals
// - Claim frequency with extra-Poisson variation
//
// EXPOSURE:
// ---------
// Like Poisson, often used with an offset for exposure:
//   log(E(Y)) = log(exposure) + Xβ
//
// =============================================================================

use ndarray::Array1;
use crate::links::{Link, LogLink};
use super::Family;

/// Negative Binomial family for overdispersed count data.
///
/// Uses the NB2 parameterization where variance is quadratic in the mean:
///   Var(Y) = μ + μ²/θ
///
/// # Parameters
/// - `theta`: Dispersion parameter (θ > 0). Larger θ = less overdispersion.
///   As θ → ∞, approaches Poisson. Typical values: 0.5 to 10.
///
/// # Example
/// ```
/// use rustystats_core::families::{Family, NegativeBinomialFamily};
/// use ndarray::array;
///
/// // Create NB with θ = 1.0 (moderate overdispersion)
/// let family = NegativeBinomialFamily::new(1.0);
///
/// let mu = array![1.0, 2.0, 4.0];
/// let variance = family.variance(&mu);
///
/// // Variance = μ + μ²/θ = μ + μ² (when θ=1)
/// // V(1) = 1 + 1 = 2
/// // V(2) = 2 + 4 = 6
/// // V(4) = 4 + 16 = 20
/// ```
#[derive(Debug, Clone)]
pub struct NegativeBinomialFamily {
    /// Dispersion parameter θ (theta). Larger = less overdispersion.
    /// Also called the "size" or "shape" parameter.
    pub theta: f64,
}

impl NegativeBinomialFamily {
    /// Create a new Negative Binomial family with specified dispersion.
    ///
    /// # Arguments
    /// * `theta` - Dispersion parameter (must be > 0).
    ///   - θ = 1: Moderate overdispersion (variance = μ + μ²)
    ///   - θ = 0.5: Strong overdispersion (variance = μ + 2μ²)
    ///   - θ = 10: Mild overdispersion (close to Poisson)
    ///
    /// # Panics
    /// Panics if theta ≤ 0.
    pub fn new(theta: f64) -> Self {
        if theta <= 0.0 {
            panic!("Negative Binomial theta must be > 0, got {}", theta);
        }
        NegativeBinomialFamily { theta }
    }

    /// Create with default θ = 1.0 (moderate overdispersion).
    pub fn default() -> Self {
        NegativeBinomialFamily { theta: 1.0 }
    }

    /// Get alpha = 1/theta (alternative parameterization).
    /// Var(Y) = μ + α×μ²
    #[inline]
    pub fn alpha(&self) -> f64 {
        1.0 / self.theta
    }
}

impl Family for NegativeBinomialFamily {
    fn name(&self) -> &str {
        "NegativeBinomial"
    }

    /// Variance function: V(μ) = μ + μ²/θ = μ(1 + μ/θ)
    ///
    /// This is the NB2 (quadratic) variance function.
    /// The full variance is Var(Y) = V(μ) (no additional dispersion parameter).
    #[inline]
    fn variance(&self, mu: &Array1<f64>) -> Array1<f64> {
        let alpha = self.alpha();
        mu.mapv(|m| m + alpha * m * m)
    }

    /// Unit deviance for Negative Binomial.
    ///
    /// d_i = 2 × [y × log(y/μ) - (y + θ) × log((y + θ)/(μ + θ))]
    ///
    /// When y = 0, this simplifies to:
    /// d_i = 2θ × log((μ + θ)/θ)
    fn unit_deviance(&self, y: &Array1<f64>, mu: &Array1<f64>) -> Array1<f64> {
        let theta = self.theta;

        ndarray::Zip::from(y)
            .and(mu)
            .map_collect(|&yi, &mui| {
                let mui_safe = mui.max(1e-10);
                
                if yi == 0.0 {
                    // When y = 0: 2θ × log((μ+θ)/θ)
                    2.0 * theta * ((mui_safe + theta) / theta).ln()
                } else {
                    // General case: 2 × [y×log(y/μ) - (y+θ)×log((y+θ)/(μ+θ))]
                    let term1 = yi * (yi / mui_safe).ln();
                    let term2 = (yi + theta) * ((yi + theta) / (mui_safe + theta)).ln();
                    2.0 * (term1 - term2)
                }
            })
    }

    /// Default link: Log (same as Poisson).
    fn default_link(&self) -> Box<dyn Link> {
        Box::new(LogLink)
    }

    /// Initialize μ from observed y values.
    ///
    /// Same strategy as Poisson: ensure positive starting values.
    fn initialize_mu(&self, y: &Array1<f64>) -> Array1<f64> {
        y.mapv(|yi| (yi + 0.1).max(0.1))
    }

    /// μ must be strictly positive.
    fn is_valid_mu(&self, mu: &Array1<f64>) -> bool {
        mu.iter().all(|&m| m > 0.0 && m.is_finite())
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
    fn test_negative_binomial_name() {
        let family = NegativeBinomialFamily::new(1.0);
        assert_eq!(family.name(), "NegativeBinomial");
    }

    #[test]
    fn test_negative_binomial_variance() {
        let family = NegativeBinomialFamily::new(1.0);  // θ = 1, α = 1
        let mu = array![1.0, 2.0, 4.0];

        let var = family.variance(&mu);

        // V(μ) = μ + μ²/θ = μ + μ² (when θ=1)
        let expected = array![
            1.0 + 1.0,   // 2
            2.0 + 4.0,   // 6
            4.0 + 16.0,  // 20
        ];
        assert_abs_diff_eq!(var, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_negative_binomial_variance_different_theta() {
        // θ = 2 means α = 0.5
        let family = NegativeBinomialFamily::new(2.0);
        let mu = array![2.0];

        let var = family.variance(&mu);

        // V(2) = 2 + 2²/2 = 2 + 2 = 4
        assert_abs_diff_eq!(var[0], 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_negative_binomial_approaches_poisson() {
        // Large θ should give variance ≈ μ (Poisson-like)
        let family = NegativeBinomialFamily::new(1000.0);
        let mu = array![1.0, 2.0, 5.0];

        let var = family.variance(&mu);

        // V(μ) = μ + μ²/1000 ≈ μ for large θ
        for (v, m) in var.iter().zip(mu.iter()) {
            assert!((v - m).abs() < 0.1);  // Close to Poisson variance
        }
    }

    #[test]
    fn test_negative_binomial_deviance_perfect_fit() {
        let family = NegativeBinomialFamily::new(1.0);
        let y = array![1.0, 2.0, 3.0];
        let mu = array![1.0, 2.0, 3.0];  // Perfect fit

        let dev = family.unit_deviance(&y, &mu);

        // Perfect fit should have zero deviance
        for d in dev.iter() {
            assert_abs_diff_eq!(*d, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_negative_binomial_deviance_with_zero() {
        let family = NegativeBinomialFamily::new(1.0);
        let y = array![0.0];
        let mu = array![1.0];

        let dev = family.unit_deviance(&y, &mu);

        // When y=0: 2θ × log(θ/(μ+θ)) = 2×1×log(1/2) = -2×log(2) ≈ -1.386
        // But deviance should be positive... let me recalculate
        // Actually: 2θ × log(θ/(μ+θ)) = 2×log(0.5) = 2×(-0.693) = -1.386
        // This is negative, which is wrong. Let me check the formula again.
        // 
        // The correct NB deviance for y=0 is:
        // d = 2 × [y×log(y/μ) - (y+θ)×log((y+θ)/(μ+θ))]
        //   = 2 × [0 - θ×log(θ/(μ+θ))]
        //   = -2θ × log(θ/(μ+θ))
        //   = 2θ × log((μ+θ)/θ)
        //   = 2×1×log(2/1) = 2×log(2) ≈ 1.386
        let expected = 2.0 * (2.0_f64).ln();
        assert_abs_diff_eq!(dev[0], expected, epsilon = 1e-10);
    }

    #[test]
    fn test_negative_binomial_default_link() {
        let family = NegativeBinomialFamily::new(1.0);
        let link = family.default_link();
        assert_eq!(link.name(), "log");
    }

    #[test]
    fn test_negative_binomial_initialize_handles_zeros() {
        let family = NegativeBinomialFamily::new(1.0);
        let y = array![0.0, 0.0, 1.0, 5.0];

        let mu_init = family.initialize_mu(&y);

        // All μ should be positive
        assert!(mu_init.iter().all(|&x| x > 0.0));
    }

    #[test]
    fn test_negative_binomial_valid_mu() {
        let family = NegativeBinomialFamily::new(1.0);

        assert!(family.is_valid_mu(&array![0.1, 1.0, 10.0]));
        assert!(!family.is_valid_mu(&array![0.0, 1.0]));  // Zero invalid
        assert!(!family.is_valid_mu(&array![-1.0, 1.0])); // Negative invalid
    }

    #[test]
    #[should_panic]
    fn test_negative_binomial_invalid_theta() {
        let _family = NegativeBinomialFamily::new(0.0);  // Should panic
    }

    #[test]
    #[should_panic]
    fn test_negative_binomial_negative_theta() {
        let _family = NegativeBinomialFamily::new(-1.0);  // Should panic
    }

    #[test]
    fn test_negative_binomial_alpha() {
        let family = NegativeBinomialFamily::new(2.0);
        assert_abs_diff_eq!(family.alpha(), 0.5, epsilon = 1e-10);

        let family2 = NegativeBinomialFamily::new(0.5);
        assert_abs_diff_eq!(family2.alpha(), 2.0, epsilon = 1e-10);
    }
}
