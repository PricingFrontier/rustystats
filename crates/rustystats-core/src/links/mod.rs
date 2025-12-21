// =============================================================================
// Link Functions for GLMs
// =============================================================================
//
// WHAT IS A LINK FUNCTION?
// ------------------------
// In a GLM, the link function connects the linear predictor (η = Xβ) to the
// mean of the response variable (μ). It's written as:
//
//     η = g(μ)    or equivalently    μ = g⁻¹(η)
//
// where g is the link function and g⁻¹ is its inverse.
//
// WHY DO WE NEED LINK FUNCTIONS?
// ------------------------------
// Different types of response variables need different transformations:
//
//   - Continuous data (e.g., claim amounts): Identity link, η = μ
//   - Count data (e.g., claim counts): Log link, η = log(μ)
//   - Binary data (e.g., claim/no claim): Logit link, η = log(μ/(1-μ))
//
// The link function ensures that predictions stay in a valid range.
// For example, with a log link, μ = exp(η) is always positive.
//
// STRUCTURE:
// ----------
// Each link function must provide three things:
//   1. link(μ)     - Transform mean to linear predictor
//   2. inverse(η)  - Transform linear predictor back to mean
//   3. derivative(μ) - The derivative dη/dμ (needed for IRLS fitting)
//
// =============================================================================

// Import ndarray for array operations
use ndarray::Array1;

// Sub-modules for each link function
mod identity;
mod log;
mod logit;

// Re-export the concrete implementations
pub use identity::IdentityLink;
pub use log::LogLink;
pub use logit::LogitLink;

// =============================================================================
// The Link Trait
// =============================================================================
// 
// This defines what every link function must be able to do.
// In Rust, a "trait" is like an interface - it specifies a contract that
// types must fulfill.
//
// Any new link function you create must implement all these methods.
// =============================================================================

/// The Link trait defines the interface for all link functions.
/// 
/// # Example
/// ```
/// use rustystats_core::links::{Link, IdentityLink};
/// use ndarray::array;
/// 
/// let link = IdentityLink;
/// let mu = array![1.0, 2.0, 3.0];
/// let eta = link.link(&mu);  // For identity: eta = mu
/// ```
pub trait Link: Send + Sync {
    /// Returns the name of this link function (for display purposes).
    fn name(&self) -> &str;
    
    /// Apply the link function: η = g(μ)
    /// 
    /// Transforms the mean (μ) to the linear predictor scale (η).
    /// 
    /// # Arguments
    /// * `mu` - Array of mean values
    /// 
    /// # Returns
    /// Array of linear predictor values
    fn link(&self, mu: &Array1<f64>) -> Array1<f64>;
    
    /// Apply the inverse link function: μ = g⁻¹(η)
    /// 
    /// Transforms the linear predictor (η) back to the mean scale (μ).
    /// This is used to get predicted values from the model.
    /// 
    /// # Arguments
    /// * `eta` - Array of linear predictor values
    /// 
    /// # Returns
    /// Array of mean values
    fn inverse(&self, eta: &Array1<f64>) -> Array1<f64>;
    
    /// Compute the derivative of the link function: dη/dμ
    /// 
    /// This is needed by the IRLS algorithm to compute working weights.
    /// 
    /// # Arguments
    /// * `mu` - Array of mean values
    /// 
    /// # Returns
    /// Array of derivative values at each point
    fn derivative(&self, mu: &Array1<f64>) -> Array1<f64>;
}
