// =============================================================================
// Quasi-Families for Overdispersed Data
// =============================================================================
//
// Quasi-families (QuasiPoisson, QuasiBinomial) handle OVERDISPERSION - when
// the actual variance in data exceeds what the base family assumes.
//
// THE PROBLEM:
// ------------
// Poisson assumes Var(Y) = μ (variance = mean)
// Binomial assumes Var(Y) = μ(1-μ)/n (for binary, Var = μ(1-μ))
//
// Real data often has MORE variance than these assumptions predict.
// This is called OVERDISPERSION.
//
// THE SOLUTION:
// -------------
// Quasi-families modify the variance assumption to:
//   Var(Y) = φ × V(μ)
//
// where φ is a dispersion parameter ESTIMATED from the data (not fixed at 1).
//
// WHAT CHANGES:
// -------------
// 1. Point estimates (coefficients) are UNCHANGED - same as Poisson/Binomial
// 2. Standard errors are INFLATED by √φ
// 3. P-values and confidence intervals become more conservative
//
// WHY THIS MATTERS:
// -----------------
// If you ignore overdispersion:
// - Standard errors will be too small
// - P-values will be too small
// - You'll find "significant" effects that aren't really there
//
// DETECTING OVERDISPERSION:
// -------------------------
// Pearson χ² / df >> 1 suggests overdispersion
// For quasi-families, this ratio estimates φ directly.
//
// WHEN TO USE:
// ------------
// - QuasiPoisson: Count data with extra variation (common in insurance)
// - QuasiBinomial: Binary/proportion data with extra variation
//
// ALTERNATIVES:
// -------------
// - Negative Binomial: Explicit distribution for overdispersed counts
// - Robust standard errors: HC0-HC3 sandwich estimators
//
// =============================================================================

use ndarray::Array1;
use crate::links::{Link, LogLink, LogitLink};
use super::{Family, PoissonFamily, BinomialFamily};

// =============================================================================
// QuasiPoisson Family
// =============================================================================

/// QuasiPoisson family for overdispersed count data.
///
/// Uses the same variance function as Poisson (V(μ) = μ) but estimates
/// the dispersion parameter φ from data instead of fixing it at 1.
///
/// This is the simplest approach to handling overdispersion in count data.
///
/// # Example
/// ```
/// use rustystats_core::families::{Family, QuasiPoissonFamily};
/// use ndarray::array;
///
/// let family = QuasiPoissonFamily;
/// let mu = array![0.5, 1.0, 2.0];
///
/// // Variance function is same as Poisson
/// let variance = family.variance(&mu);  // [0.5, 1.0, 2.0]
/// ```
///
/// # Note
/// The difference from Poisson is NOT in the family implementation itself,
/// but in how standard errors are computed. For QuasiPoisson, we estimate
/// φ = Pearson_χ² / df_resid instead of assuming φ = 1.
#[derive(Debug, Clone, Copy)]
pub struct QuasiPoissonFamily;

impl Family for QuasiPoissonFamily {
    fn name(&self) -> &str {
        "QuasiPoisson"
    }

    /// Variance function: V(μ) = μ (same as Poisson)
    ///
    /// The full variance is Var(Y) = φ × μ where φ is estimated.
    #[inline]
    fn variance(&self, mu: &Array1<f64>) -> Array1<f64> {
        // Delegate to Poisson - same variance function
        mu.clone()
    }

    /// Unit deviance (same as Poisson)
    ///
    /// Deviance is used for convergence checking and φ estimation,
    /// not for inference, so it's the same as Poisson.
    #[inline]
    fn unit_deviance(&self, y: &Array1<f64>, mu: &Array1<f64>) -> Array1<f64> {
        PoissonFamily.unit_deviance(y, mu)
    }

    /// Default link: Log (same as Poisson)
    fn default_link(&self) -> Box<dyn Link> {
        Box::new(LogLink)
    }

    /// Initialize μ (same as Poisson)
    fn initialize_mu(&self, y: &Array1<f64>) -> Array1<f64> {
        PoissonFamily.initialize_mu(y)
    }

    /// Valid μ check (same as Poisson: must be positive)
    fn is_valid_mu(&self, mu: &Array1<f64>) -> bool {
        PoissonFamily.is_valid_mu(mu)
    }
    
    fn clamp_mu(&self, mu: &Array1<f64>) -> Array1<f64> {
        PoissonFamily.clamp_mu(mu)
    }
    
    fn is_log_link_default(&self) -> bool { true }
    
    fn log_likelihood(&self, y: &Array1<f64>, mu: &Array1<f64>, _scale: f64, weights: Option<&Array1<f64>>) -> f64 {
        crate::diagnostics::log_likelihood_poisson(y, mu, weights)
    }
}

// =============================================================================
// QuasiBinomial Family
// =============================================================================

/// QuasiBinomial family for overdispersed binary/proportion data.
///
/// Uses the same variance function as Binomial (V(μ) = μ(1-μ)) but estimates
/// the dispersion parameter φ from data instead of fixing it at 1.
///
/// Useful when binary outcomes are more variable than Binomial predicts,
/// which can happen with clustered data or unobserved heterogeneity.
///
/// # Example
/// ```
/// use rustystats_core::families::{Family, QuasiBinomialFamily};
/// use ndarray::array;
///
/// let family = QuasiBinomialFamily;
/// let mu = array![0.2, 0.5, 0.8];
///
/// // Variance function is same as Binomial
/// let variance = family.variance(&mu);  // [0.16, 0.25, 0.16]
/// ```
///
/// # Note
/// The difference from Binomial is NOT in the family implementation itself,
/// but in how standard errors are computed. For QuasiBinomial, we estimate
/// φ = Pearson_χ² / df_resid instead of assuming φ = 1.
#[derive(Debug, Clone, Copy)]
pub struct QuasiBinomialFamily;

impl Family for QuasiBinomialFamily {
    fn name(&self) -> &str {
        "QuasiBinomial"
    }

    /// Variance function: V(μ) = μ(1-μ) (same as Binomial)
    ///
    /// The full variance is Var(Y) = φ × μ(1-μ) where φ is estimated.
    #[inline]
    fn variance(&self, mu: &Array1<f64>) -> Array1<f64> {
        // Delegate to Binomial - same variance function
        mu.mapv(|p| p * (1.0 - p))
    }

    /// Unit deviance (same as Binomial)
    ///
    /// Deviance is used for convergence checking and φ estimation,
    /// not for inference, so it's the same as Binomial.
    #[inline]
    fn unit_deviance(&self, y: &Array1<f64>, mu: &Array1<f64>) -> Array1<f64> {
        BinomialFamily.unit_deviance(y, mu)
    }

    /// Default link: Logit (same as Binomial)
    fn default_link(&self) -> Box<dyn Link> {
        Box::new(LogitLink)
    }

    /// Initialize μ (same as Binomial)
    fn initialize_mu(&self, y: &Array1<f64>) -> Array1<f64> {
        BinomialFamily.initialize_mu(y)
    }

    /// Valid μ check (same as Binomial: must be in (0, 1))
    fn is_valid_mu(&self, mu: &Array1<f64>) -> bool {
        BinomialFamily.is_valid_mu(mu)
    }
    
    fn clamp_mu(&self, mu: &Array1<f64>) -> Array1<f64> {
        BinomialFamily.clamp_mu(mu)
    }
    
    fn log_likelihood(&self, y: &Array1<f64>, mu: &Array1<f64>, _scale: f64, weights: Option<&Array1<f64>>) -> f64 {
        crate::diagnostics::log_likelihood_binomial(y, mu, weights)
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

    // =========================================================================
    // QuasiPoisson Tests
    // =========================================================================

    #[test]
    fn test_quasipoisson_name() {
        let family = QuasiPoissonFamily;
        assert_eq!(family.name(), "QuasiPoisson");
    }

    #[test]
    fn test_quasipoisson_variance_equals_poisson() {
        let quasi = QuasiPoissonFamily;
        let poisson = PoissonFamily;
        let mu = array![0.5, 1.0, 2.0, 10.0];

        let var_quasi = quasi.variance(&mu);
        let var_poisson = poisson.variance(&mu);

        // Variance functions should be identical
        assert_abs_diff_eq!(var_quasi, var_poisson, epsilon = 1e-10);
        // V(μ) = μ
        assert_abs_diff_eq!(var_quasi, mu, epsilon = 1e-10);
    }

    #[test]
    fn test_quasipoisson_deviance_equals_poisson() {
        let quasi = QuasiPoissonFamily;
        let poisson = PoissonFamily;
        let y = array![0.0, 1.0, 3.0, 5.0];
        let mu = array![1.0, 1.5, 2.5, 4.0];

        let dev_quasi = quasi.unit_deviance(&y, &mu);
        let dev_poisson = poisson.unit_deviance(&y, &mu);

        assert_abs_diff_eq!(dev_quasi, dev_poisson, epsilon = 1e-10);
    }

    #[test]
    fn test_quasipoisson_default_link() {
        let family = QuasiPoissonFamily;
        let link = family.default_link();
        assert_eq!(link.name(), "log");
    }

    #[test]
    fn test_quasipoisson_initialize_handles_zeros() {
        let family = QuasiPoissonFamily;
        let y = array![0.0, 0.0, 1.0, 5.0];

        let mu_init = family.initialize_mu(&y);

        // All μ should be positive (can't have log(0))
        assert!(mu_init.iter().all(|&x| x > 0.0));
    }

    #[test]
    fn test_quasipoisson_valid_mu() {
        let family = QuasiPoissonFamily;

        assert!(family.is_valid_mu(&array![0.1, 1.0, 10.0]));
        assert!(!family.is_valid_mu(&array![0.0, 1.0]));  // Zero invalid
        assert!(!family.is_valid_mu(&array![-1.0, 1.0])); // Negative invalid
    }

    // =========================================================================
    // QuasiBinomial Tests
    // =========================================================================

    #[test]
    fn test_quasibinomial_name() {
        let family = QuasiBinomialFamily;
        assert_eq!(family.name(), "QuasiBinomial");
    }

    #[test]
    fn test_quasibinomial_variance_equals_binomial() {
        let quasi = QuasiBinomialFamily;
        let binomial = BinomialFamily;
        let mu = array![0.1, 0.3, 0.5, 0.7, 0.9];

        let var_quasi = quasi.variance(&mu);
        let var_binomial = binomial.variance(&mu);

        // Variance functions should be identical
        assert_abs_diff_eq!(var_quasi, var_binomial, epsilon = 1e-10);
    }

    #[test]
    fn test_quasibinomial_variance_function() {
        let family = QuasiBinomialFamily;
        let mu = array![0.2, 0.5, 0.8];

        let var = family.variance(&mu);

        // V(μ) = μ(1-μ)
        let expected = array![0.16, 0.25, 0.16];
        assert_abs_diff_eq!(var, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_quasibinomial_deviance_equals_binomial() {
        let quasi = QuasiBinomialFamily;
        let binomial = BinomialFamily;
        let y = array![0.0, 0.3, 0.7, 1.0];
        let mu = array![0.2, 0.4, 0.6, 0.8];

        let dev_quasi = quasi.unit_deviance(&y, &mu);
        let dev_binomial = binomial.unit_deviance(&y, &mu);

        assert_abs_diff_eq!(dev_quasi, dev_binomial, epsilon = 1e-10);
    }

    #[test]
    fn test_quasibinomial_default_link() {
        let family = QuasiBinomialFamily;
        let link = family.default_link();
        assert_eq!(link.name(), "logit");
    }

    #[test]
    fn test_quasibinomial_initialize() {
        let family = QuasiBinomialFamily;
        let y = array![0.0, 1.0, 0.0, 1.0];

        let mu_init = family.initialize_mu(&y);

        // All μ should be in (0, 1)
        assert!(mu_init.iter().all(|&x| x > 0.0 && x < 1.0));
    }

    #[test]
    fn test_quasibinomial_valid_mu() {
        let family = QuasiBinomialFamily;

        assert!(family.is_valid_mu(&array![0.1, 0.5, 0.9]));
        assert!(!family.is_valid_mu(&array![0.0, 0.5]));  // Zero boundary invalid
        assert!(!family.is_valid_mu(&array![0.5, 1.0]));  // One boundary invalid
        assert!(!family.is_valid_mu(&array![-0.1, 0.5])); // Negative invalid
        assert!(!family.is_valid_mu(&array![0.5, 1.1]));  // > 1 invalid
    }
}
