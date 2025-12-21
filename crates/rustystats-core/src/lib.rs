// =============================================================================
// RustyStats Core Library
// =============================================================================
//
// This is the entry point for the pure Rust statistics library.
// All the mathematical heavy-lifting happens here - no Python code involved.
//
// STRUCTURE:
// ----------
// The library is organized into modules, each handling a specific concern:
//
//   - families:    Distribution families (Gaussian, Poisson, Binomial, etc.)
//   - links:       Link functions (Identity, Log, Logit, etc.)
//   - solvers:     Fitting algorithms (IRLS - Iteratively Reweighted Least Squares)
//   - inference:   Statistical inference (standard errors, p-values, etc.)
//   - diagnostics: Model diagnostics (residuals, influence measures)
//   - error:       Error types used throughout the library
//
// FOR MAINTAINERS:
// ----------------
// When adding new functionality:
//   1. Add it to the appropriate module (or create a new one)
//   2. Write tests in that module (see existing tests for examples)
//   3. Re-export public items here so users can access them easily
//   4. Update the Python bindings in the `rustystats` crate
//
// =============================================================================

// Declare our modules - each is in its own file or folder
pub mod error;
pub mod families;
pub mod inference;
pub mod links;
pub mod solvers;

// Re-export commonly used items at the top level for convenience
// Users can write `use rustystats_core::GaussianFamily` instead of
// `use rustystats_core::families::gaussian::GaussianFamily`
pub use error::{RustyStatsError, Result};
pub use families::Family;
pub use inference::{pvalue_z, pvalue_t, confidence_interval_z, confidence_interval_t};
pub use links::Link;
pub use solvers::{IRLSConfig, IRLSResult, fit_glm, fit_glm_full};
