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

use super::{BinomialFamily, Family, PoissonFamily};
use crate::links::{Link, LogLink, LogitLink};
use ndarray::Array1;

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

    fn log_likelihood_form(&self) -> crate::families::LogLikelihoodForm {
        PoissonFamily.log_likelihood_form()
    }

    #[inline]
    fn log_likelihood_row(&self, yi: f64, mui: f64, scale: f64) -> f64 {
        PoissonFamily.log_likelihood_row(yi, mui, scale)
    }

    /// Variance function V(μ) = μ (scaled by φ).
    #[inline]
    fn variance_at(&self, mu: f64) -> f64 {
        mu
    }

    #[inline]
    fn initialize_mu_at(&self, yi: f64, y_mean: f64) -> f64 {
        PoissonFamily.initialize_mu_at(yi, y_mean)
    }

    #[inline]
    fn is_valid_mu_at(&self, mu: f64) -> bool {
        PoissonFamily.is_valid_mu_at(mu)
    }

    #[inline]
    fn clamp_mu_at(&self, mu: f64) -> f64 {
        PoissonFamily.clamp_mu_at(mu)
    }

    /// Unit deviance (same as Poisson)
    ///
    /// Deviance is used for convergence checking and φ estimation,
    /// not for inference, so it's the same as Poisson.
    #[inline]
    fn unit_deviance(&self, y: &Array1<f64>, mu: &Array1<f64>) -> Array1<f64> {
        PoissonFamily.unit_deviance(y, mu)
    }

    #[inline]
    fn unit_deviance_at(&self, yi: f64, mui: f64) -> f64 {
        PoissonFamily.unit_deviance_at(yi, mui)
    }

    /// Default link: Log (same as Poisson)
    fn default_link(&self) -> Box<dyn Link> {
        Box::new(LogLink)
    }

    fn is_log_link_default(&self) -> bool {
        true
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

    fn log_likelihood_form(&self) -> crate::families::LogLikelihoodForm {
        BinomialFamily.log_likelihood_form()
    }

    #[inline]
    fn log_likelihood_row(&self, yi: f64, mui: f64, scale: f64) -> f64 {
        BinomialFamily.log_likelihood_row(yi, mui, scale)
    }

    /// Variance function V(μ) = μ(1 − μ) (scaled by φ).
    #[inline]
    fn variance_at(&self, p: f64) -> f64 {
        p * (1.0 - p)
    }

    #[inline]
    fn initialize_mu_at(&self, yi: f64, y_mean: f64) -> f64 {
        BinomialFamily.initialize_mu_at(yi, y_mean)
    }

    #[inline]
    fn is_valid_mu_at(&self, mu: f64) -> bool {
        BinomialFamily.is_valid_mu_at(mu)
    }

    #[inline]
    fn clamp_mu_at(&self, mu: f64) -> f64 {
        BinomialFamily.clamp_mu_at(mu)
    }

    /// Unit deviance (same as Binomial)
    ///
    /// Deviance is used for convergence checking and φ estimation,
    /// not for inference, so it's the same as Binomial.
    #[inline]
    fn unit_deviance(&self, y: &Array1<f64>, mu: &Array1<f64>) -> Array1<f64> {
        BinomialFamily.unit_deviance(y, mu)
    }

    #[inline]
    fn unit_deviance_at(&self, yi: f64, mui: f64) -> f64 {
        BinomialFamily.unit_deviance_at(yi, mui)
    }

    /// Default link: Logit (same as Binomial)
    fn default_link(&self) -> Box<dyn Link> {
        Box::new(LogitLink)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

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
        assert_abs_diff_eq!(*var_quasi, *var_poisson, epsilon = 1e-10);
        // V(μ) = μ
        assert_abs_diff_eq!(*var_quasi, mu, epsilon = 1e-10);
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
        assert_abs_diff_eq!(
            quasi.unit_deviance_at(0.0, 1.25),
            poisson.unit_deviance_at(0.0, 1.25),
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            quasi.unit_deviance_at(3.0, 2.5),
            poisson.unit_deviance_at(3.0, 2.5),
            epsilon = 1e-12
        );
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

        assert_eq!(mu_init.len(), y.len());
        assert_abs_diff_eq!(mu_init, array![0.1, 0.1, 1.1, 5.1], epsilon = 1e-12);
    }

    #[test]
    fn test_quasipoisson_valid_mu() {
        let family = QuasiPoissonFamily;

        assert!(family.is_valid_mu(&array![0.1, 1.0, 10.0]));
        assert!(!family.is_valid_mu(&array![0.0, 1.0])); // Zero invalid
        assert!(!family.is_valid_mu(&array![-1.0, 1.0])); // Negative invalid
    }

    #[test]
    fn test_quasipoisson_clamp_log_link_and_likelihood_delegate_to_poisson() {
        let quasi = QuasiPoissonFamily;
        let poisson = PoissonFamily;
        let mu = array![0.0, -1.0, 2.0];
        let clamped = quasi.clamp_mu(&mu);
        assert!(clamped.iter().all(|&value| value > 0.0));
        assert_eq!(clamped[2], 2.0);
        assert!(quasi.is_log_link_default());
        assert!(!quasi.fixed_dispersion());

        let y = array![0.0, 2.0, 5.0];
        let mu = array![0.4, 1.8, 4.5];
        let weights = array![1.0, 0.5, 2.0];
        assert_abs_diff_eq!(
            quasi.log_likelihood(&y, &mu, 99.0, Some(&weights)),
            poisson.log_likelihood(&y, &mu, 99.0, Some(&weights)),
            epsilon = 1e-12
        );
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
        assert_abs_diff_eq!(*var_quasi, *var_binomial, epsilon = 1e-10);
    }

    #[test]
    fn test_quasibinomial_variance_function() {
        let family = QuasiBinomialFamily;
        let mu = array![0.2, 0.5, 0.8];

        let var = family.variance(&mu);

        // V(μ) = μ(1-μ)
        let expected = array![0.16, 0.25, 0.16];
        assert_abs_diff_eq!(*var, expected, epsilon = 1e-10);
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
        assert_abs_diff_eq!(
            quasi.unit_deviance_at(0.0, 0.2),
            binomial.unit_deviance_at(0.0, 0.2),
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            quasi.unit_deviance_at(1.0, 0.8),
            binomial.unit_deviance_at(1.0, 0.8),
            epsilon = 1e-12
        );
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

        assert_eq!(mu_init.len(), y.len());
        assert_abs_diff_eq!(mu_init, array![0.25, 0.75, 0.25, 0.75], epsilon = 1e-12);
    }

    #[test]
    fn test_quasibinomial_valid_mu() {
        let family = QuasiBinomialFamily;

        assert!(family.is_valid_mu(&array![0.1, 0.5, 0.9]));
        assert!(!family.is_valid_mu(&array![0.0, 0.5])); // Zero boundary invalid
        assert!(!family.is_valid_mu(&array![0.5, 1.0])); // One boundary invalid
        assert!(!family.is_valid_mu(&array![-0.1, 0.5])); // Negative invalid
        assert!(!family.is_valid_mu(&array![0.5, 1.1])); // > 1 invalid
    }

    #[test]
    fn test_quasibinomial_clamp_and_likelihood_delegate_to_binomial() {
        let quasi = QuasiBinomialFamily;
        let binomial = BinomialFamily;
        let clamped = quasi.clamp_mu(&array![-1.0, 0.2, 2.0]);
        assert!(clamped[0] > 0.0);
        assert_eq!(clamped[1], 0.2);
        assert!(clamped[2] < 1.0);
        assert!(!quasi.fixed_dispersion());

        let y = array![0.0, 1.0, 1.0, 0.0];
        let mu = array![0.1, 0.8, 0.6, 0.2];
        let weights = array![1.0, 2.0, 0.5, 1.5];
        assert_abs_diff_eq!(
            quasi.log_likelihood(&y, &mu, 99.0, Some(&weights)),
            binomial.log_likelihood(&y, &mu, 99.0, Some(&weights)),
            epsilon = 1e-12
        );
    }
}
