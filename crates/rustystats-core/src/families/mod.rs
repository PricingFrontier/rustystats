// =============================================================================
// Distribution Families for GLMs
// =============================================================================
//
// WHAT IS A FAMILY?
// -----------------
// A "family" (or "distribution family") specifies what type of data you're
// modeling. Different types of data follow different probability distributions:
//
//   - Continuous data (any value):        Gaussian (Normal) family
//   - Count data (0, 1, 2, ...):          Poisson family
//   - Binary data (0 or 1):               Binomial family
//   - Positive continuous (> 0):          Gamma family
//   - Proportions with known n:           Binomial family
//   - Mixed zeros and positives:          Tweedie family (1 < p < 2)
//
// WHAT DOES A FAMILY PROVIDE?
// ---------------------------
// Each family defines:
//
//   1. Variance function V(μ) - How variance relates to the mean
//      This is key! It captures the heteroscedasticity structure.
//
//   2. Deviance - A measure of model fit (like sum of squared residuals)
//
//   3. Default link function - The "canonical" link for this family
//
// THE VARIANCE FUNCTION IS CRUCIAL:
// ---------------------------------
// The variance function V(μ) determines how spread changes with the mean:
//
//   Gaussian:   V(μ) = 1         (constant variance - homoscedastic)
//   Poisson:    V(μ) = μ         (variance equals mean)
//   Binomial:   V(μ) = μ(1-μ)    (highest at μ=0.5, zero at 0 and 1)
//   Gamma:      V(μ) = μ²        (variance proportional to mean squared)
//
// ACTUARIAL CONTEXT:
// ------------------
//   - Claim frequency: Poisson (counts, variance ≈ mean)
//   - Claim severity:  Gamma (positive, right-skewed, CV often constant)
//   - Claim occurrence: Binomial (yes/no)
//
// =============================================================================

use std::borrow::Cow;

use crate::links::Link;
use ndarray::Array1;

// Sub-modules for each family
mod binomial;
mod gamma;
mod gaussian;
mod negative_binomial;
mod poisson;
mod quasi;
mod tweedie;

// Re-export concrete implementations
pub use binomial::BinomialFamily;
pub use gamma::GammaFamily;
pub use gaussian::GaussianFamily;
pub use negative_binomial::NegativeBinomialFamily;
pub use poisson::PoissonFamily;
pub use quasi::{QuasiBinomialFamily, QuasiPoissonFamily};
pub use tweedie::TweedieFamily;

// =============================================================================
// The Family Trait
// =============================================================================
//
// Every distribution family must implement these methods.
// The IRLS fitting algorithm uses these to:
//   - Compute working weights
//   - Calculate deviance for convergence checking
//   - Initialize starting values
//
// =============================================================================

/// How a family's log-likelihood combines its rows (`specs/out-of-core-fit.md` §8).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogLikelihoodForm {
    /// −½·(Σ w(y−μ)²/φ + Σw·ln(2πφ)), from two sums that do not read φ.
    Gaussian,
    /// Σ w·ℓ_i(y, μ, φ). `needs_scale` when ℓ_i reads φ (Gamma).
    RowSum { needs_scale: bool },
    /// −½ × the unweighted deviance.
    DevianceFallback,
}

/// The unweighted mean response, block-ordered (`rows::reduce_rows`) so it does
/// not depend on thread count or batching; 1.0 for an empty response.
pub fn response_mean(y: &Array1<f64>) -> f64 {
    let n = y.len();
    if n == 0 {
        return 1.0;
    }
    let total = crate::rows::reduce_rows(0.0f64, n, |acc, rows| {
        for i in rows {
            *acc += y[i];
        }
    });
    total / n as f64
}

/// The Family trait defines the interface for all distribution families.
///
/// # Key Concepts
///
/// - **Variance function**: Defines how Var(Y) relates to E(Y) = μ
/// - **Deviance**: Measures goodness-of-fit (lower is better)
/// - **Link function**: Connects linear predictor η to mean μ
pub trait Family: Send + Sync {
    /// Returns the name of this family (for display purposes).
    fn name(&self) -> &str;

    /// The variance function at one μ: V(μ).
    ///
    /// The scalar methods are the one definition of each family's arithmetic;
    /// the array methods map them, so per-row kernels and whole-array code
    /// produce identical bits.
    fn variance_at(&self, mu: f64) -> f64;

    /// The variance function V(μ) for every element.
    fn variance<'a>(&self, mu: &'a Array1<f64>) -> Cow<'a, Array1<f64>> {
        Cow::Owned(mu.mapv(|value| self.variance_at(value)))
    }

    /// Compute the unit deviance for each observation.
    ///
    /// The deviance measures how well the model fits. It's defined as:
    ///   D = 2 × [log L(saturated model) - log L(fitted model)]
    ///
    /// Lower deviance = better fit. The unit deviance is the contribution
    /// from each observation.
    ///
    /// # Arguments
    /// * `y` - Array of observed response values
    /// * `mu` - Array of fitted mean values
    ///
    /// # Returns
    /// Array of unit deviance values (one per observation)
    fn unit_deviance(&self, y: &Array1<f64>, mu: &Array1<f64>) -> Array1<f64>;

    /// Per-row scalar unit deviance.
    ///
    /// This is the streaming counterpart of [`unit_deviance`]: identical
    /// arithmetic, but evaluated for a single (y, μ) pair without allocating.
    /// Streaming callers (e.g. null-deviance computation) fold over this in a
    /// single pass instead of materialising an `Array1<f64>` of unit deviances.
    ///
    /// Implementations MUST mirror the per-row arithmetic of `unit_deviance`
    /// exactly so the two paths agree to bit-equality.
    fn unit_deviance_at(&self, yi: f64, mui: f64) -> f64;

    /// Compute the total deviance (sum of unit deviances).
    ///
    /// This can be weighted if weights are provided.
    fn deviance(&self, y: &Array1<f64>, mu: &Array1<f64>, weights: Option<&Array1<f64>>) -> f64 {
        // Block-ordered (`rows::reduce_rows`): independent of thread count and of
        // how rows arrive, with today's per-row expression `unit_deviance * w`.
        let n = y.len();
        crate::rows::reduce_rows(0.0f64, n, |total, rows| match weights {
            Some(w) => {
                for i in rows {
                    *total += self.unit_deviance_at(y[i], mu[i]) * w[i];
                }
            }
            None => {
                for i in rows {
                    *total += self.unit_deviance_at(y[i], mu[i]);
                }
            }
        })
    }

    /// Return the default (canonical) link function for this family.
    ///
    /// Each family has a "natural" link that simplifies the math.
    /// However, you can use other links if they make more sense for your data.
    fn default_link(&self) -> Box<dyn Link>;

    /// The starting μ for one response value.
    ///
    /// `y_mean` is the unweighted mean response (block-ordered, `rows::reduce_rows`)
    /// when [`Family::initialize_needs_mean`] is true, and NaN otherwise.
    fn initialize_mu_at(&self, yi: f64, y_mean: f64) -> f64;

    /// Whether [`Family::initialize_mu_at`] reads the mean response.
    fn initialize_needs_mean(&self) -> bool {
        false
    }

    /// Starting values for μ: the IRLS algorithm needs a sensible start from the
    /// observed data.
    fn initialize_mu(&self, y: &Array1<f64>) -> Array1<f64> {
        let y_mean = if self.initialize_needs_mean() {
            response_mean(y)
        } else {
            f64::NAN
        };
        y.mapv(|yi| self.initialize_mu_at(yi, y_mean))
    }

    /// Whether one μ is in the valid range for this family (for example,
    /// positive for Poisson, inside (0, 1) for Binomial).
    fn is_valid_mu_at(&self, mu: f64) -> bool;

    /// Whether every μ is in the valid range for this family.
    fn is_valid_mu(&self, mu: &Array1<f64>) -> bool {
        mu.iter().all(|&value| self.is_valid_mu_at(value))
    }

    /// Clamp one μ into the family's valid range. Default: unchanged (Gaussian).
    fn clamp_mu_at(&self, mu: f64) -> f64 {
        mu
    }

    /// Clamp μ to the valid range for this family, after each IRLS update.
    fn clamp_mu(&self, mu: &Array1<f64>) -> Array1<f64> {
        mu.mapv(|value| self.clamp_mu_at(value))
    }

    /// Whether this family has a fixed dispersion parameter (φ = 1).
    ///
    /// True for Poisson and Binomial (exact distributions with known variance).
    /// False for Gaussian, Gamma, Tweedie, Quasi-families (dispersion estimated from data).
    fn fixed_dispersion(&self) -> bool {
        false
    }

    /// How this family's log-likelihood combines its rows. Default: −½ × the
    /// unweighted deviance, for families without a closed form.
    fn log_likelihood_form(&self) -> LogLikelihoodForm {
        LogLikelihoodForm::DevianceFallback
    }

    /// One row's log-likelihood term ℓ_i(y, μ, φ), for
    /// [`LogLikelihoodForm::RowSum`] families. The total is Σ w·ℓ_i (or Σ ℓ_i
    /// unweighted), reduced block by block.
    fn log_likelihood_row(&self, _y: f64, _mu: f64, _scale: f64) -> f64 {
        f64::NAN
    }

    /// The log-likelihood at fitted means `mu` with dispersion `scale`.
    ///
    /// One implementation for every family (`diagnostics::log_likelihood_for_family`),
    /// shared with the fit-time passes so both give identical bits.
    fn log_likelihood(
        &self,
        y: &Array1<f64>,
        mu: &Array1<f64>,
        scale: f64,
        weights: Option<&Array1<f64>>,
    ) -> f64 {
        crate::diagnostics::log_likelihood_for_family(self, y, mu, scale, weights)
    }

    /// Whether this family uses a log link by default.
    /// Used by null deviance computation to handle offsets correctly.
    fn is_log_link_default(&self) -> bool {
        false
    }

    /// The power p used for the null model's constant under a log link and an offset:
    /// c = Σw·y·a^(1−p) / Σw·a^(2−p) with a = exp(offset), which is exact when V(μ) = μ^p.
    /// A family whose variance is not a power (negative binomial, binomial) has no closed
    /// form, and a single-pass fit cannot iterate for one, so it keeps the Poisson-form constant.
    fn null_variance_power(&self) -> f64 {
        1.0
    }

    /// Whether this family supports true Hessian-based IRLS weights for the log link.
    ///
    /// For certain family/link combinations (Gamma, Tweedie with log link),
    /// using the true Hessian instead of the Fisher information approximation
    /// can dramatically reduce the number of IRLS iterations (e.g., 50-100 → 5-10).
    ///
    /// Default: false (use standard Fisher-based weights)
    fn use_true_hessian_weights(&self) -> bool {
        false
    }

    /// The true-Hessian IRLS weight at one row, used only when
    /// [`Family::use_true_hessian_weights`] is true. Default: the variance, which
    /// gives standard IRLS behaviour.
    fn true_hessian_weight_at(&self, mu: f64, _y: f64) -> f64 {
        self.variance_at(mu)
    }

    /// True-Hessian IRLS weights for every row (Gamma with log link: μ;
    /// Tweedie, 1 < p < 2, with log link: μ^(2−p)).
    fn true_hessian_weights(&self, mu: &Array1<f64>, y: &Array1<f64>) -> Array1<f64> {
        ndarray::Zip::from(mu)
            .and(y)
            .map_collect(|&mui, &yi| self.true_hessian_weight_at(mui, yi))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::links::IdentityLink;

    struct FallbackFamily;

    impl Family for FallbackFamily {
        fn name(&self) -> &str {
            "fallback"
        }

        fn variance_at(&self, mu: f64) -> f64 {
            mu + 1.0
        }

        fn initialize_mu_at(&self, yi: f64, _y_mean: f64) -> f64 {
            yi
        }

        fn is_valid_mu_at(&self, mu: f64) -> bool {
            mu.is_finite()
        }

        fn unit_deviance(&self, y: &Array1<f64>, mu: &Array1<f64>) -> Array1<f64> {
            (y - mu).mapv(|v| v * v)
        }

        fn unit_deviance_at(&self, yi: f64, mui: f64) -> f64 {
            let residual = yi - mui;
            residual * residual
        }

        fn default_link(&self) -> Box<dyn Link> {
            Box::new(IdentityLink)
        }
    }

    #[test]
    fn default_family_trait_methods_are_consistent() {
        let family = FallbackFamily;
        let y = Array1::from_vec(vec![1.0, 3.0, 6.0]);
        let mu = Array1::from_vec(vec![0.5, 2.0, 4.0]);
        let weights = Array1::from_vec(vec![2.0, 0.5, 4.0]);

        assert_eq!(family.name(), "fallback");
        assert_eq!(family.default_link().name(), "identity");
        assert!(family.is_valid_mu(&mu));
        assert_eq!(family.initialize_mu(&y), y);
        assert_eq!(family.clamp_mu(&mu), mu);
        assert!(!family.fixed_dispersion());
        assert!(!family.is_log_link_default());
        assert!(!family.use_true_hessian_weights());

        let unit = family.unit_deviance(&y, &mu);
        let unweighted = unit.sum();
        let weighted = (&unit * &weights).sum();
        assert_eq!(family.deviance(&y, &mu, None), unweighted);
        assert_eq!(family.deviance(&y, &mu, Some(&weights)), weighted);
        assert_eq!(
            family.log_likelihood(&y, &mu, 2.0, Some(&weights)),
            -unweighted / 2.0
        );
        assert_eq!(
            family.true_hessian_weights(&mu, &y),
            family.variance(&mu).into_owned()
        );

        for i in 0..y.len() {
            assert_eq!(unit[i], family.unit_deviance_at(y[i], mu[i]));
        }
    }
}
