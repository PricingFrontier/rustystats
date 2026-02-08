// =============================================================================
// GLM Solvers
// =============================================================================
//
// This module contains algorithms for fitting Generalized Linear Models.
// The main algorithm is IRLS (Iteratively Reweighted Least Squares).
//
// HOW GLM FITTING WORKS (High-Level Overview)
// -------------------------------------------
//
// We want to find coefficients β that best explain the relationship:
//
//     g(E[Y]) = Xβ
//
// where:
//   - Y is the response variable (what we're predicting)
//   - X is the design matrix (predictors/features)
//   - β is the coefficient vector (what we're solving for)
//   - g is the link function
//   - E[Y] = μ is the expected value of Y
//
// Unlike ordinary least squares, we can't solve this directly because:
//   1. The link function g() makes it non-linear
//   2. The variance depends on μ (heteroscedasticity)
//
// IRLS solves this by iteratively:
//   1. Linearizing the problem around current estimates
//   2. Solving a weighted least squares problem
//   3. Updating estimates and repeating until convergence
//
// =============================================================================

mod irls;
mod coordinate_descent;
pub mod smooth_glm;
pub mod gcv_optimizer;
pub mod nnls;

pub use irls::{FitConfig, fit_glm_unified, IRLSConfig, IRLSResult};
pub use irls::{solve_weighted_least_squares_with_penalty_matrix, compute_xtwx};
pub use smooth_glm::{SmoothGLMResult, SmoothGLMConfig, SmoothTermData, Monotonicity, SmoothTermSpec, fit_smooth_glm_full_matrix};
pub use gcv_optimizer::{MultiTermGCVOptimizer, brent_minimize};
pub use nnls::{NNLSResult, NNLSConfig, nnls, nnls_weighted, nnls_penalized, nnls_weighted_penalized};

use ndarray::Array1;
use crate::families::Family;

/// Safe initialization of μ that works for any family.
///
/// Used as fallback when `family.initialize_mu(y)` produces invalid values
/// (e.g., all zeros for Poisson). Computes a weighted average of each y_i
/// with the global mean, then clamps to the family's valid range.
pub(crate) fn initialize_mu_safe(y: &Array1<f64>, family: &dyn Family) -> Array1<f64> {
    let y_mean = y.mean().unwrap_or(1.0).max(0.01);
    let raw: Array1<f64> = y.mapv(|yi| (yi + y_mean) / 2.0);
    family.clamp_mu(&raw)
}
