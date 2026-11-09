// =============================================================================
// Gamma Family
// =============================================================================
//
// The Gamma family is for POSITIVE CONTINUOUS data - values that must be > 0.
// This is the workhorse for claim SEVERITY (amount) modeling in insurance.
//
// PROPERTIES:
// -----------
// - Distribution: Y ~ Gamma(shape=α, scale=β) with mean μ = αβ
// - Variance function: V(μ) = μ² (variance proportional to mean squared)
// - Canonical link: Inverse (η = 1/μ), but LOG is more common in practice
// - Dispersion: φ = 1/α (the inverse of the shape parameter)
//
// KEY INSIGHT - CONSTANT COEFFICIENT OF VARIATION:
// ------------------------------------------------
// Since V(μ) = μ², the standard deviation is proportional to the mean:
//   SD(Y) = √(φ × μ²) = √φ × μ
//
// The coefficient of variation (CV = SD/mean) is constant:
//   CV = √φ
//
// This is very realistic for claim amounts! A $1,000 claim might vary by
// $500, while a $100,000 claim might vary by $50,000 - same CV of 50%.
//
// WHY NOT GAUSSIAN FOR CLAIM AMOUNTS?
// -----------------------------------
// Gaussian assumes constant variance: a $1,000 claim and a $100,000 claim
// would have the same variance. This is unrealistic.
//
// Gamma's V(μ) = μ² assumption is much more appropriate for monetary amounts.
//
// LOG LINK VS INVERSE LINK:
// -------------------------
// - Inverse link (canonical): η = 1/μ, harder to interpret
// - Log link (common): η = log(μ), multiplicative interpretation
//
// In practice, the log link is almost always used because:
// 1. Coefficients have multiplicative interpretation (like frequency models)
// 2. Predictions are guaranteed positive
// 3. Easier to combine with Poisson frequency models
//
// =============================================================================

use super::Family;
use crate::links::{Link, LogLink};
use ndarray::Array1;

/// Gamma family for positive continuous data.
///
/// The standard choice for modeling claim severity in insurance.
///
/// # Example
/// ```
/// use rustystats_core::families::{Family, GammaFamily};
/// use ndarray::array;
///
/// let family = GammaFamily;
/// let mu = array![100.0, 1000.0, 10000.0];
/// let variance = family.variance(&mu);  // [10000, 1000000, 100000000]
/// ```
#[derive(Debug, Clone, Copy)]
pub struct GammaFamily;

impl Family for GammaFamily {
    fn name(&self) -> &str {
        "Gamma"
    }

    fn log_likelihood_form(&self) -> crate::families::LogLikelihoodForm {
        crate::families::LogLikelihoodForm::RowSum { needs_scale: true }
    }

    /// ℓ_i = (α−1)·log(y) − α·y/μ + α·log(α/μ) − log Γ(α) with α = 1/φ
    /// (statsmodels' formula); y and μ floored at `MU_MIN_POSITIVE`.
    #[inline]
    fn log_likelihood_row(&self, yi: f64, mui: f64, scale: f64) -> f64 {
        use crate::constants::MU_MIN_POSITIVE;
        use statrs::function::gamma::ln_gamma;
        let alpha = 1.0 / scale;
        let yi_safe = yi.max(MU_MIN_POSITIVE);
        let mui_safe = mui.max(MU_MIN_POSITIVE);
        (alpha - 1.0) * yi_safe.ln() - alpha * yi_safe / mui_safe + alpha * (alpha / mui_safe).ln()
            - ln_gamma(alpha)
    }

    /// Variance function V(μ) = μ².
    #[inline]
    fn variance_at(&self, mu: f64) -> f64 {
        mu * mu
    }

    /// Starting μ: y, but at least 1% of the mean response.
    #[inline]
    fn initialize_mu_at(&self, yi: f64, y_mean: f64) -> f64 {
        yi.max(y_mean * 0.01)
    }

    fn initialize_needs_mean(&self) -> bool {
        true
    }

    #[inline]
    fn is_valid_mu_at(&self, mu: f64) -> bool {
        mu > 0.0 && mu.is_finite()
    }

    #[inline]
    fn clamp_mu_at(&self, mu: f64) -> f64 {
        mu.max(crate::constants::MU_MIN_POSITIVE)
    }

    /// For Gamma with log link, the true Hessian weight is μ (unused while
    /// `use_true_hessian_weights` is false, which matches statsmodels' Fisher weights).
    #[inline]
    fn true_hessian_weight_at(&self, mu: f64, _y: f64) -> f64 {
        mu
    }

    /// Unit deviance: 2 × [-log(y/μ) + (y-μ)/μ]
    ///
    /// Simplifies to: 2 × [(y-μ)/μ - log(y/μ)]
    ///
    /// Note: y must be > 0 for Gamma. We use a small floor to prevent log(0) = -inf
    /// when data contains zeros (which is technically invalid for Gamma but can occur).
    fn unit_deviance(&self, y: &Array1<f64>, mu: &Array1<f64>) -> Array1<f64> {
        ndarray::Zip::from(y)
            .and(mu)
            .map_collect(|&yi, &mui| self.unit_deviance_at(yi, mui))
    }

    #[inline]
    fn unit_deviance_at(&self, yi: f64, mui: f64) -> f64 {
        use crate::constants::MU_MIN_POSITIVE;
        let yi_safe = yi.max(MU_MIN_POSITIVE);
        let mui_safe = mui.max(MU_MIN_POSITIVE);
        let ratio = yi_safe / mui_safe;
        2.0 * ((yi_safe - mui_safe) / mui_safe - ratio.ln())
    }

    /// Default link for Gamma.
    ///
    /// Note: The CANONICAL link is inverse (1/μ), but we use LOG because:
    /// - It's standard in actuarial practice
    /// - Better interpretability (multiplicative effects)
    /// - Consistent with Poisson frequency models
    fn default_link(&self) -> Box<dyn Link> {
        Box::new(LogLink)
    }

    fn is_log_link_default(&self) -> bool {
        true
    }

    fn null_variance_power(&self) -> f64 {
        2.0
    }

    /// Whether to use true Hessian weights for IRLS.
    ///
    /// Note: While true Hessian can reduce IRLS iterations, it produces a different
    /// covariance matrix than Fisher information. Statsmodels uses Fisher weights
    /// (W=1 for Gamma+log link), so we disable true Hessian to match standard inference.
    fn use_true_hessian_weights(&self) -> bool {
        false
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

    #[test]
    fn test_gamma_name() {
        let family = GammaFamily;
        assert_eq!(family.name(), "Gamma");
    }

    #[test]
    fn test_gamma_variance() {
        let family = GammaFamily;
        let mu = array![10.0, 100.0, 1000.0];

        let var = family.variance(&mu);

        // V(μ) = μ²
        let expected = array![100.0, 10000.0, 1000000.0];
        assert_abs_diff_eq!(*var, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_gamma_constant_cv() {
        // Demonstrate that CV is constant
        let family = GammaFamily;
        let mu = array![10.0, 100.0, 1000.0];

        let var = family.variance(&mu);

        // CV² = Var / μ² = μ² / μ² = 1 (before dispersion scaling)
        // So the "unit CV²" is 1 for all μ
        for i in 0..mu.len() {
            let cv_squared = var[i] / (mu[i] * mu[i]);
            assert_abs_diff_eq!(cv_squared, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_gamma_unit_deviance_perfect_fit() {
        let family = GammaFamily;
        let y = array![100.0, 500.0, 1000.0];
        let mu = array![100.0, 500.0, 1000.0]; // Perfect predictions

        let dev = family.unit_deviance(&y, &mu);

        // Perfect fit: y/μ = 1, so deviance = 2×[0 - log(1)] = 0
        let expected = array![0.0, 0.0, 0.0];
        assert_abs_diff_eq!(dev, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_gamma_unit_deviance() {
        let family = GammaFamily;
        let y = array![150.0];
        let mu = array![100.0];

        let dev = family.unit_deviance(&y, &mu);

        // 2 × [(150-100)/100 - log(150/100)]
        // = 2 × [0.5 - log(1.5)]
        // = 2 × [0.5 - 0.405...]
        // ≈ 0.189
        let expected = 2.0 * ((150.0 - 100.0) / 100.0 - (150.0_f64 / 100.0).ln());
        assert_abs_diff_eq!(dev[0], expected, epsilon = 1e-10);
    }

    #[test]
    fn test_gamma_default_link() {
        let family = GammaFamily;
        let link = family.default_link();

        // We use log link (not canonical inverse) for practical reasons
        assert_eq!(link.name(), "log");
        assert!(family.is_log_link_default());
        assert!(!family.fixed_dispersion());
        assert!(!family.use_true_hessian_weights());
    }

    #[test]
    fn test_gamma_initialize_positive() {
        let family = GammaFamily;
        let y = array![0.0, 100.0, 200.0];

        let mu_init = family.initialize_mu(&y);

        assert_eq!(mu_init.len(), y.len());
        assert_abs_diff_eq!(mu_init, array![1.0, 100.0, 200.0], epsilon = 1e-12);
    }

    #[test]
    fn test_gamma_valid_mu() {
        let family = GammaFamily;

        // Positive values are valid
        assert!(family.is_valid_mu(&array![0.1, 1.0, 1000.0]));

        // Zero is NOT valid
        assert!(!family.is_valid_mu(&array![0.0, 1.0]));

        // Negative is NOT valid
        assert!(!family.is_valid_mu(&array![-1.0, 1.0]));
    }

    #[test]
    fn test_gamma_clamp_likelihood_and_true_hessian_contracts() {
        let family = GammaFamily;
        let clamped = family.clamp_mu(&array![0.0, -1.0, 2.5]);
        assert!(clamped[0] > 0.0);
        assert!(clamped[1] > 0.0);
        assert_eq!(clamped[2], 2.5);

        let mu = array![100.0, 250.0, 500.0];
        assert_abs_diff_eq!(family.true_hessian_weights(&mu, &mu), mu, epsilon = 1e-12);

        let y = array![120.0, 210.0, 650.0];
        let weights = array![1.0, 0.5, 2.0];
        let ll = family.log_likelihood(&y, &mu, 0.75, Some(&weights));
        let expected = crate::diagnostics::log_likelihood_gamma(&y, &mu, 0.75, Some(&weights));
        assert_abs_diff_eq!(ll, expected, epsilon = 1e-12);
    }

    #[test]
    fn test_gamma_actuarial_interpretation() {
        // Demonstrate actuarial interpretation
        // If φ = 0.25 (estimated from data), then CV = √0.25 = 0.5 (50%)
        // This means claims typically vary by about 50% around their mean

        let family = GammaFamily;
        let dispersion = 0.25;

        // For a claim with expected value $1000
        let mu = array![1000.0];
        let unit_var = family.variance(&mu)[0]; // μ² = 1,000,000

        let actual_var = dispersion * unit_var; // 250,000
        let actual_sd = actual_var.sqrt(); // 500
        let cv = actual_sd / mu[0]; // 0.5 = 50%

        assert_abs_diff_eq!(cv, 0.5, epsilon = 1e-10);
    }
}
