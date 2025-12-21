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

pub use irls::{IRLSConfig, IRLSResult, fit_glm, fit_glm_full};
