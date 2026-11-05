// =============================================================================
// Identity Link Function
// =============================================================================
//
// The simplest link function: η = μ (no transformation at all).
//
// WHEN TO USE:
// ------------
// - Gaussian (Normal) family - standard linear regression
// - When the response can take any real value (positive or negative)
// - When you want to model the mean directly
//
// PROPERTIES:
// -----------
// - Link:       η = μ
// - Inverse:    μ = η
// - Derivative: dη/dμ = 1
//
// EXAMPLE (Actuarial):
// --------------------
// Modeling the average claim amount where you expect a linear relationship
// with predictors. Note: For claim amounts, Gamma with log link is often
// preferred since amounts must be positive.
//
// =============================================================================

use super::Link;

/// The Identity link function: η = μ
///
/// This is the default link for the Gaussian (Normal) family.
/// It applies no transformation - the linear predictor equals the mean.
#[derive(Debug, Clone, Copy)]
pub struct IdentityLink;

impl Link for IdentityLink {
    fn name(&self) -> &str {
        "identity"
    }

    /// Link function: η = μ.
    #[inline]
    fn link_at(&self, mu: f64) -> f64 {
        mu
    }

    /// Inverse link: μ = η.
    #[inline]
    fn inverse_at(&self, eta: f64) -> f64 {
        eta
    }

    /// Derivative: dη/dμ = 1.
    #[inline]
    fn derivative_at(&self, _mu: f64) -> f64 {
        1.0
    }
}

// =============================================================================
// Tests
// =============================================================================
//
// These tests verify that our implementation is correct.
// Run them with: cargo test -p rustystats-core
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_identity_link() {
        let link = IdentityLink;
        let mu = array![1.0, 2.0, 3.0, -1.0, 0.0];

        // Link should return the same values
        let eta = link.link(&mu);
        assert_abs_diff_eq!(eta, mu, epsilon = 1e-10);
    }

    #[test]
    fn test_identity_inverse() {
        let link = IdentityLink;
        let eta = array![-2.0, 0.0, 1.5, 100.0];

        // Inverse should return the same values
        let mu = link.inverse(&eta);
        assert_abs_diff_eq!(mu, eta, epsilon = 1e-10);
    }

    #[test]
    fn test_identity_derivative() {
        let link = IdentityLink;
        let mu = array![1.0, 2.0, 3.0];

        // Derivative should be all ones
        let deriv = link.derivative(&mu);
        let expected = array![1.0, 1.0, 1.0];
        assert_abs_diff_eq!(deriv, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_identity_roundtrip() {
        // Applying link then inverse should give back the original
        let link = IdentityLink;
        let original = array![0.5, 1.0, 2.5, 10.0];

        let eta = link.link(&original);
        let recovered = link.inverse(&eta);

        assert_abs_diff_eq!(recovered, original, epsilon = 1e-10);
    }
}
