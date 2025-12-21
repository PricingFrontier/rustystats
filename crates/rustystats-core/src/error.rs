// =============================================================================
// Error Types for RustyStats
// =============================================================================
//
// This module defines all the errors that can occur in the library.
// Using proper error types (instead of just panicking) means:
//   - Users get helpful error messages
//   - Errors can be handled gracefully in Python
//   - The code is more robust
//
// We use the `thiserror` crate which makes defining errors simple.
// Each error variant has a human-readable message that explains what went wrong.
//
// =============================================================================

use thiserror::Error;

/// All possible errors that can occur in RustyStats.
/// 
/// When something goes wrong, you'll get one of these variants with a 
/// descriptive message explaining the problem.
#[derive(Error, Debug)]
pub enum RustyStatsError {
    // -------------------------------------------------------------------------
    // Input Validation Errors
    // -------------------------------------------------------------------------
    
    /// The input arrays have incompatible shapes.
    /// For example: trying to fit a model where X has 100 rows but y has 50.
    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),
    
    /// A value is outside its valid range.
    /// For example: a probability must be between 0 and 1.
    #[error("Invalid value: {0}")]
    InvalidValue(String),
    
    /// An input array is empty when it shouldn't be.
    #[error("Empty input: {0}")]
    EmptyInput(String),

    // -------------------------------------------------------------------------
    // Numerical Errors
    // -------------------------------------------------------------------------
    
    /// The fitting algorithm didn't converge within the maximum iterations.
    /// This might mean: data issues, poor starting values, or need more iterations.
    #[error("Convergence failed: {0}")]
    ConvergenceFailure(String),
    
    /// A matrix operation failed (e.g., trying to invert a singular matrix).
    /// This often indicates multicollinearity in the predictor variables.
    #[error("Linear algebra error: {0}")]
    LinearAlgebraError(String),
    
    /// A numerical computation produced NaN or infinity.
    /// This can happen with extreme values or poor model specification.
    #[error("Numerical error: {0}")]
    NumericalError(String),

    // -------------------------------------------------------------------------
    // Configuration Errors
    // -------------------------------------------------------------------------
    
    /// An invalid combination of family and link function was specified.
    #[error("Invalid family/link combination: {0}")]
    InvalidFamilyLink(String),
    
    /// A required parameter is missing.
    #[error("Missing parameter: {0}")]
    MissingParameter(String),
}

/// A convenient Result type that uses our error type.
/// 
/// Instead of writing `Result<T, RustyStatsError>` everywhere,
/// we can just write `Result<T>`.
pub type Result<T> = std::result::Result<T, RustyStatsError>;
