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

use ndarray::Array1;
use crate::links::Link;

// Sub-modules for each family
mod gaussian;
mod poisson;
mod binomial;
mod gamma;

// Re-export concrete implementations
pub use gaussian::GaussianFamily;
pub use poisson::PoissonFamily;
pub use binomial::BinomialFamily;
pub use gamma::GammaFamily;

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
    /// # Arguments
    /// * `mu` - Array of mean values
    /// 
    /// # Returns
    /// Array of variance function values
    fn variance(&self, mu: &Array1<f64>) -> Array1<f64>;
    
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
}
