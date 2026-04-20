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

    /// Compute the variance function V(μ).
    ///
    /// The variance function relates the variance of Y to its mean:
    ///   Var(Y) = φ × V(μ)
    ///
    /// where φ is the dispersion parameter (φ=1 for Poisson and Binomial).
    ///
    /// Returns `Cow::Borrowed` when variance equals mu (e.g. Poisson),
    /// avoiding allocation on the hot path.
    ///
    /// # Arguments
    /// * `mu` - Array of mean values
    ///
    /// # Returns
    /// Array of variance function values (borrowed or owned)
    fn variance<'a>(&self, mu: &'a Array1<f64>) -> Cow<'a, Array1<f64>>;

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
        let unit_dev = self.unit_deviance(y, mu);
        match weights {
            Some(w) => (&unit_dev * w).sum(),
            None => unit_dev.sum(),
        }
    }

    /// Return the default (canonical) link function for this family.
    ///
    /// Each family has a "natural" link that simplifies the math.
    /// However, you can use other links if they make more sense for your data.
    fn default_link(&self) -> Box<dyn Link>;

    /// Initialize starting values for μ.
    ///
    /// The IRLS algorithm needs starting values. This provides sensible
    /// defaults based on the observed data.
    ///
    /// # Arguments
    /// * `y` - Array of observed response values
    ///
    /// # Returns
    /// Array of initial μ values
    fn initialize_mu(&self, y: &Array1<f64>) -> Array1<f64>;

    /// Check if μ values are in the valid range for this family.
    ///
    /// For example:
    /// - Poisson: μ must be positive
    /// - Binomial: μ must be in (0, 1)
    fn is_valid_mu(&self, mu: &Array1<f64>) -> bool;

    /// Clamp μ to the valid range for this family.
    ///
    /// This is called after each IRLS iteration to ensure μ stays in bounds.
    /// Default: no clamping (Gaussian).
    fn clamp_mu(&self, mu: &Array1<f64>) -> Array1<f64> {
        mu.clone()
    }

    /// Whether this family has a fixed dispersion parameter (φ = 1).
    ///
    /// True for Poisson and Binomial (exact distributions with known variance).
    /// False for Gaussian, Gamma, Tweedie, Quasi-families (dispersion estimated from data).
    fn fixed_dispersion(&self) -> bool {
        false
    }

    /// Compute the log-likelihood for this family.
    ///
    /// Default implementation returns a deviance-based approximation: -deviance/2.
    /// Families with closed-form log-likelihoods override this.
    ///
    /// # Arguments
    /// * `y` - Observed response
    /// * `mu` - Fitted values
    /// * `scale` - Dispersion parameter φ
    /// * `weights` - Optional prior weights
    fn log_likelihood(
        &self,
        y: &Array1<f64>,
        mu: &Array1<f64>,
        _scale: f64,
        _weights: Option<&Array1<f64>>,
    ) -> f64 {
        // Fallback: deviance-based approximation
        -self.deviance(y, mu, None) / 2.0
    }

    /// Whether this family uses a log link by default.
    /// Used by null deviance computation to handle offsets correctly.
    fn is_log_link_default(&self) -> bool {
        false
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

    /// Compute optimized IRLS weights using true Hessian (when applicable).
    ///
    /// For Gamma with log link: w = μ (instead of 1 from Fisher info)
    /// For Tweedie (1 < p < 2) with log link: w = μ^(2-p)
    ///
    /// This method is only called when `use_true_hessian_weights()` returns true.
    ///
    /// # Arguments
    /// * `mu` - Array of fitted mean values
    /// * `y` - Array of observed response values (needed for some Hessians)
    ///
    /// # Returns
    /// Array of optimized IRLS weights
    fn true_hessian_weights(&self, mu: &Array1<f64>, _y: &Array1<f64>) -> Array1<f64> {
        // Default: same as variance (which gives standard IRLS behavior)
        self.variance(mu).into_owned()
    }
}
