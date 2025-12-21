// =============================================================================
// Log Link Function
// =============================================================================
//
// The log link: η = log(μ), which means μ = exp(η).
//
// WHEN TO USE:
// ------------
// - Poisson family (count data like claim frequency)
// - Gamma family (positive continuous data like claim severity)
// - Negative Binomial (overdispersed counts)
// - Any time the response must be strictly positive
//
// PROPERTIES:
// -----------
// - Link:       η = log(μ)
// - Inverse:    μ = exp(η)
// - Derivative: dη/dμ = 1/μ
//
// WHY IT'S USEFUL:
// ----------------
// 1. Ensures predictions are always positive (exp(anything) > 0)
// 2. Multiplicative effects: a unit change in x multiplies μ by exp(β)
// 3. Natural for modeling rates and ratios
//
// EXAMPLE (Actuarial):
// --------------------
// Claim frequency modeling: If you have 1000 policyholders and want to
// predict the number of claims, a Poisson GLM with log link is standard.
// 
// If β₁ = 0.1 for "young driver", then young drivers have:
//   exp(0.1) ≈ 1.105 times the claim frequency (10.5% higher)
//
// =============================================================================

use ndarray::Array1;
use super::Link;

/// The Log link function: η = log(μ)
/// 
/// This is the canonical (default) link for Poisson family and a common
/// choice for Gamma family in actuarial applications.
#[derive(Debug, Clone, Copy)]
pub struct LogLink;

impl Link for LogLink {
    fn name(&self) -> &str {
        "log"
    }
    
    /// Link function: η = log(μ)
    /// 
    /// Takes the natural logarithm of each value.
    /// 
    /// # Warning
    /// μ must be positive! log(0) = -∞ and log(negative) is undefined.
    /// In practice, we should add bounds checking.
    fn link(&self, mu: &Array1<f64>) -> Array1<f64> {
        mu.mapv(|x| x.ln())
    }
    
    /// Inverse link: μ = exp(η)
    /// 
    /// Exponentiates each value. This always produces positive results,
    /// which is exactly what we want for count/positive data.
    fn inverse(&self, eta: &Array1<f64>) -> Array1<f64> {
        eta.mapv(|x| x.exp())
    }
    
    /// Derivative: dη/dμ = 1/μ
    /// 
    /// The derivative of log(μ) with respect to μ.
    fn derivative(&self, mu: &Array1<f64>) -> Array1<f64> {
        mu.mapv(|x| 1.0 / x)
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
    fn test_log_link() {
        let link = LogLink;
        let mu = array![1.0, 2.718281828, 10.0];  // e ≈ 2.718...
        
        let eta = link.link(&mu);
        
        // log(1) = 0, log(e) = 1, log(10) ≈ 2.303
        let expected = array![0.0, 1.0, 10.0_f64.ln()];
        assert_abs_diff_eq!(eta, expected, epsilon = 1e-6);
    }
    
    #[test]
    fn test_log_inverse() {
        let link = LogLink;
        let eta = array![0.0, 1.0, 2.0];
        
        let mu = link.inverse(&eta);
        
        // exp(0) = 1, exp(1) ≈ 2.718, exp(2) ≈ 7.389
        let expected = array![1.0, 1.0_f64.exp(), 2.0_f64.exp()];
        assert_abs_diff_eq!(mu, expected, epsilon = 1e-6);
    }
    
    #[test]
    fn test_log_derivative() {
        let link = LogLink;
        let mu = array![1.0, 2.0, 4.0];
        
        let deriv = link.derivative(&mu);
        
        // d/dμ log(μ) = 1/μ
        let expected = array![1.0, 0.5, 0.25];
        assert_abs_diff_eq!(deriv, expected, epsilon = 1e-10);
    }
    
    #[test]
    fn test_log_roundtrip() {
        // Applying link then inverse should give back the original
        let link = LogLink;
        let original = array![0.5, 1.0, 2.5, 10.0];
        
        let eta = link.link(&original);
        let recovered = link.inverse(&eta);
        
        assert_abs_diff_eq!(recovered, original, epsilon = 1e-10);
    }
    
    #[test]
    fn test_log_interpretation() {
        // Demonstrate the multiplicative interpretation
        let link = LogLink;
        
        // If η increases by 0.1, μ is multiplied by exp(0.1) ≈ 1.105
        let eta1 = array![1.0];
        let eta2 = array![1.1];
        
        let mu1 = link.inverse(&eta1);
        let mu2 = link.inverse(&eta2);
        
        let ratio = mu2[0] / mu1[0];
        let expected_ratio = 0.1_f64.exp();  // ≈ 1.105
        
        assert_abs_diff_eq!(ratio, expected_ratio, epsilon = 1e-10);
    }
}
