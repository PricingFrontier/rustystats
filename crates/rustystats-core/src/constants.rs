// =============================================================================
// Numerical Constants for RustyStats
// =============================================================================
//
// This module defines all numerical constants used throughout the library.
// Centralizing these makes the code more maintainable and self-documenting.
//
// GUIDELINES:
// -----------
// - Use descriptive names that explain the PURPOSE, not just the value
// - Group related constants together
// - Document why each constant exists and when to use it
//
// =============================================================================

// =============================================================================
// Convergence and Tolerance Constants
// =============================================================================

/// Default convergence tolerance for IRLS and other iterative algorithms.
/// We stop when: |deviance_new - deviance_old| / deviance_old < CONVERGENCE_TOL
pub const CONVERGENCE_TOL: f64 = 1e-8;

/// Tolerance for checking if a value is effectively zero.
/// Used for comparing floating point numbers to zero.
pub const ZERO_TOL: f64 = 1e-10;

/// Tolerance for standard error comparisons.
/// Standard errors below this are considered numerically zero.
pub const SE_TOL: f64 = 1e-10;

/// Tolerance for coefficient sparsity checks (Lasso/Elastic Net).
/// Coefficients with absolute value below this are considered zero.
pub const COEF_ZERO_TOL: f64 = 1e-10;

// =============================================================================
// Numerical Stability Constants  
// =============================================================================

/// Minimum value for predicted means (μ) in count/positive families.
/// Prevents log(0) and division by zero in Poisson, Gamma, etc.
pub const MU_MIN_POSITIVE: f64 = 1e-10;

/// Maximum value for predicted probabilities in Binomial family.
/// Keeps μ in (ε, 1-ε) to prevent log(0) issues.
pub const MU_MAX_PROBABILITY: f64 = 1.0 - 1e-10;

/// Minimum value for predicted probabilities in Binomial family.
pub const MU_MIN_PROBABILITY: f64 = 1e-10;

/// Minimum weight to prevent numerical instability in IRLS.
/// Very small weights can cause condition number issues.
pub const MIN_IRLS_WEIGHT: f64 = 1e-10;

// =============================================================================
// Algorithm Defaults
// =============================================================================

/// Default maximum iterations for IRLS algorithm.
pub const DEFAULT_MAX_ITER: usize = 25;

/// Default maximum iterations for theta estimation in Negative Binomial.
pub const DEFAULT_MAX_THETA_ITER: usize = 10;

/// Default tolerance for theta convergence in Negative Binomial.
pub const THETA_CONVERGENCE_TOL: f64 = 1e-5;

// =============================================================================
// Spline Constants
// =============================================================================

/// Default degree for B-splines (cubic).
pub const DEFAULT_SPLINE_DEGREE: usize = 3;

/// Tolerance for knot spacing comparisons in spline basis computation.
pub const KNOT_TOL: f64 = 1e-10;
