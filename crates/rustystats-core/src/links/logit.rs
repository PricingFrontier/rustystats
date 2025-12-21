// =============================================================================
// Logit Link Function
// =============================================================================
//
// The logit link: η = log(μ / (1-μ)), which means μ = 1 / (1 + exp(-η))
//
// WHEN TO USE:
// ------------
// - Binomial family (binary outcomes: yes/no, claim/no claim)
// - When modeling probabilities (values between 0 and 1)
// - Logistic regression
//
// PROPERTIES:
// -----------
// - Link:       η = log(μ / (1-μ))     [the "log-odds"]
// - Inverse:    μ = 1 / (1 + exp(-η))  [the "logistic function" or "sigmoid"]
// - Derivative: dη/dμ = 1 / (μ(1-μ))
//
// UNDERSTANDING LOG-ODDS:
// -----------------------
// If μ = 0.8 (80% probability), then:
//   - Odds = μ/(1-μ) = 0.8/0.2 = 4 (4-to-1 odds)
//   - Log-odds = log(4) ≈ 1.39
//
// The logit transforms probabilities (0 to 1) to the entire real line (-∞ to +∞).
// This is perfect for linear regression since predictions can be any value.
//
// EXAMPLE (Actuarial):
// --------------------
// Modeling whether a policyholder will make a claim (yes=1, no=0).
// 
// If β₁ = 0.5 for "previous claims", the odds ratio is:
//   exp(0.5) ≈ 1.65
// 
// Meaning: policyholders with previous claims have 1.65× the odds of claiming.
//
// =============================================================================

use ndarray::Array1;
use super::Link;

/// The Logit link function: η = log(μ / (1-μ))
/// 
/// This is the canonical (default) link for the Binomial family.
/// It's the foundation of logistic regression.
#[derive(Debug, Clone, Copy)]
pub struct LogitLink;

impl Link for LogitLink {
    fn name(&self) -> &str {
        "logit"
    }
    
    /// Link function: η = log(μ / (1-μ))
    /// 
    /// Transforms a probability (0 < μ < 1) to the log-odds scale (-∞ to +∞).
    /// 
    /// # Warning
    /// μ must be strictly between 0 and 1. At the boundaries:
    /// - logit(0) = -∞
    /// - logit(1) = +∞
    fn link(&self, mu: &Array1<f64>) -> Array1<f64> {
        mu.mapv(|p| (p / (1.0 - p)).ln())
    }
    
    /// Inverse link: μ = 1 / (1 + exp(-η))
    /// 
    /// This is the famous "sigmoid" or "logistic" function.
    /// It transforms any real number to a probability between 0 and 1.
    /// 
    /// Properties:
    /// - sigmoid(0) = 0.5
    /// - sigmoid(+large) → 1
    /// - sigmoid(-large) → 0
    fn inverse(&self, eta: &Array1<f64>) -> Array1<f64> {
        eta.mapv(|x| {
            // Use a numerically stable formulation
            // This avoids overflow for large negative values
            if x >= 0.0 {
                1.0 / (1.0 + (-x).exp())
            } else {
                let exp_x = x.exp();
                exp_x / (1.0 + exp_x)
            }
        })
    }
    
    /// Derivative: dη/dμ = 1 / (μ(1-μ))
    /// 
    /// Note: This is also 1 / Var(Y) for binomial, which connects
    /// to the variance function of the binomial family.
    fn derivative(&self, mu: &Array1<f64>) -> Array1<f64> {
        mu.mapv(|p| 1.0 / (p * (1.0 - p)))
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
    fn test_logit_link() {
        let link = LogitLink;
        let mu = array![0.5, 0.8, 0.2];
        
        let eta = link.link(&mu);
        
        // logit(0.5) = log(1) = 0
        // logit(0.8) = log(4) ≈ 1.386
        // logit(0.2) = log(0.25) ≈ -1.386
        let expected = array![0.0, (0.8 / 0.2_f64).ln(), (0.2 / 0.8_f64).ln()];
        assert_abs_diff_eq!(eta, expected, epsilon = 1e-10);
    }
    
    #[test]
    fn test_logit_inverse() {
        let link = LogitLink;
        let eta = array![0.0, 2.0, -2.0];
        
        let mu = link.inverse(&eta);
        
        // sigmoid(0) = 0.5
        // sigmoid(2) ≈ 0.881
        // sigmoid(-2) ≈ 0.119
        assert_abs_diff_eq!(mu[0], 0.5, epsilon = 1e-10);
        assert!(mu[1] > 0.5);  // Positive eta -> probability > 0.5
        assert!(mu[2] < 0.5);  // Negative eta -> probability < 0.5
        
        // sigmoid(-x) = 1 - sigmoid(x)  [symmetry property]
        assert_abs_diff_eq!(mu[1] + mu[2], 1.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_logit_derivative() {
        let link = LogitLink;
        let mu = array![0.5, 0.2, 0.8];
        
        let deriv = link.derivative(&mu);
        
        // At μ=0.5: 1/(0.5×0.5) = 4
        // At μ=0.2: 1/(0.2×0.8) = 6.25
        // At μ=0.8: 1/(0.8×0.2) = 6.25
        let expected = array![4.0, 6.25, 6.25];
        assert_abs_diff_eq!(deriv, expected, epsilon = 1e-10);
    }
    
    #[test]
    fn test_logit_roundtrip() {
        let link = LogitLink;
        let original = array![0.1, 0.3, 0.5, 0.7, 0.9];
        
        let eta = link.link(&original);
        let recovered = link.inverse(&eta);
        
        assert_abs_diff_eq!(recovered, original, epsilon = 1e-10);
    }
    
    #[test]
    fn test_odds_ratio_interpretation() {
        // Demonstrate odds ratio interpretation
        let link = LogitLink;
        
        // Person A has η = 0, Person B has η = 0.5
        let eta_a = array![0.0];
        let eta_b = array![0.5];
        
        let prob_a = link.inverse(&eta_a)[0];  // 0.5
        let prob_b = link.inverse(&eta_b)[0];  // ~0.622
        
        // Odds for A and B
        let odds_a = prob_a / (1.0 - prob_a);
        let odds_b = prob_b / (1.0 - prob_b);
        
        // Odds ratio should equal exp(0.5)
        let odds_ratio = odds_b / odds_a;
        assert_abs_diff_eq!(odds_ratio, 0.5_f64.exp(), epsilon = 1e-10);
    }
    
    #[test]
    fn test_numerical_stability_large_values() {
        // Test that we handle extreme values without overflow
        let link = LogitLink;
        let eta = array![-100.0, -50.0, 50.0, 100.0];
        
        let mu = link.inverse(&eta);
        
        // Should not produce NaN or infinity
        assert!(mu.iter().all(|&x| x.is_finite()));
        
        // Large negative -> very small probability (close to 0)
        assert!(mu[0] < 1e-10);
        assert!(mu[1] < 1e-10);
        
        // Large positive -> very high probability (close to 1)
        assert!(mu[2] > 1.0 - 1e-10);
        assert!(mu[3] > 1.0 - 1e-10);
    }
}
