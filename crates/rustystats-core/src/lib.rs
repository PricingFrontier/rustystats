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
pub mod constants;
pub mod convert;
pub mod design_matrix;
pub mod diagnostics;
pub mod error;
pub mod families;
pub mod inference;
pub mod interactions;
pub mod links;
pub mod regularization;
pub mod solvers;
pub mod splines;
pub mod target_encoding;

// Re-export commonly used items at the top level for convenience
// Users can write `use rustystats_core::GaussianFamily` instead of
// `use rustystats_core::families::gaussian::GaussianFamily`
pub use design_matrix::{
    build_categorical_basis_interaction, build_categorical_categorical_interaction,
    build_categorical_continuous_interaction, build_continuous_continuous_interaction,
    build_design_matrix, encode_categorical, encode_categorical_from_indices,
    multiply_matrix_by_continuous, predict_categorical_basis_interaction, stack_columns_horizontal,
    CategoricalEncoding, DesignColumn,
};
pub use error::{Result, RustyStatsError};
pub use families::Family;
pub use inference::{
    confidence_interval_t, confidence_interval_z, pvalue_t, pvalue_z, robust_covariance,
    robust_standard_errors, score_test_categorical, score_test_continuous, HCType, ScoreTestResult,
};
pub use links::Link;
pub use regularization::{soft_threshold, Penalty, RegularizationConfig, Standardization};
pub use solvers::{fit_glm_unified, FitConfig, IRLSConfig, IRLSResult};
pub use splines::{
    bs, bs_basis, bs_names, bs_with_knots, compute_knots, ns, ns_basis, ns_names, DEFAULT_DEGREE,
};
pub use target_encoding::{
    apply_target_encoding, target_encode, LevelStatistics, TargetEncoding, TargetEncodingConfig,
};
