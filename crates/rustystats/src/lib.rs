// =============================================================================
// RustyStats Python Bindings — Module Registration
// =============================================================================
//
// This is the entry point for the Python module. All implementation lives in
// focused submodules; this file only declares modules and registers them.
//
// Submodules:
// - families_py:       Family/link dispatch helpers + PyO3 wrapper types
// - fitting_py:        GLM fitting functions (standard, smooth, CV path)
// - inference_py:      Score tests + statistical CDFs
// - results_py:        PyGLMResults class
// - diagnostics_py:    Model diagnostics
// - splines_py:        B-spline / natural spline basis functions
// - design_matrix_py:  Categorical encoding + interaction building
// - target_encoding_py: Target / frequency encoding
// =============================================================================

mod families_py;
mod fitting_py;
mod inference_py;
mod diagnostics_py;
mod splines_py;
mod design_matrix_py;
mod target_encoding_py;
mod results_py;
mod export_onnx_py;

pub use results_py::PyGLMResults;

use pyo3::prelude::*;

// =============================================================================
// Module Registration
// =============================================================================

/// RustyStats: Fast GLM fitting with a Rust backend
/// 
/// This is the internal Rust module. Users should import from the
/// Python wrapper: `import rustystats`
#[pymodule]
fn _rustystats(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Link functions
    m.add_class::<families_py::PyIdentityLink>()?;
    m.add_class::<families_py::PyLogLink>()?;
    m.add_class::<families_py::PyLogitLink>()?;
    
    // Families
    m.add_class::<families_py::PyGaussianFamily>()?;
    m.add_class::<families_py::PyPoissonFamily>()?;
    m.add_class::<families_py::PyBinomialFamily>()?;
    m.add_class::<families_py::PyGammaFamily>()?;
    m.add_class::<families_py::PyTweedieFamily>()?;
    m.add_class::<families_py::PyQuasiPoissonFamily>()?;
    m.add_class::<families_py::PyQuasiBinomialFamily>()?;
    m.add_class::<families_py::PyNegativeBinomialFamily>()?;
    
    // GLM results
    m.add_class::<PyGLMResults>()?;
    
    // GLM fitting
    m.add_function(wrap_pyfunction!(fitting_py::fit_glm_py, m)?)?;
    m.add_function(wrap_pyfunction!(fitting_py::fit_negbinomial_py, m)?)?;
    m.add_function(wrap_pyfunction!(fitting_py::fit_smooth_glm_unified_py, m)?)?;
    m.add_function(wrap_pyfunction!(fitting_py::fit_cv_path_py, m)?)?;
    
    // Inference (score tests + CDFs)
    m.add_function(wrap_pyfunction!(inference_py::score_test_continuous_py, m)?)?;
    m.add_function(wrap_pyfunction!(inference_py::score_test_categorical_py, m)?)?;
    m.add_function(wrap_pyfunction!(inference_py::chi2_cdf_py, m)?)?;
    m.add_function(wrap_pyfunction!(inference_py::t_cdf_py, m)?)?;
    m.add_function(wrap_pyfunction!(inference_py::f_cdf_py, m)?)?;
    
    // Spline functions
    m.add_function(wrap_pyfunction!(splines_py::bs_py, m)?)?;
    m.add_function(wrap_pyfunction!(splines_py::ns_py, m)?)?;
    m.add_function(wrap_pyfunction!(splines_py::ns_with_knots_py, m)?)?;
    m.add_function(wrap_pyfunction!(splines_py::bs_knots_py, m)?)?;
    m.add_function(wrap_pyfunction!(splines_py::bs_names_py, m)?)?;
    m.add_function(wrap_pyfunction!(splines_py::ns_names_py, m)?)?;
    m.add_function(wrap_pyfunction!(splines_py::ms_py, m)?)?;
    m.add_function(wrap_pyfunction!(splines_py::ms_with_knots_py, m)?)?;
    m.add_function(wrap_pyfunction!(splines_py::ms_names_py, m)?)?;
    m.add_function(wrap_pyfunction!(splines_py::compute_knots_py, m)?)?;
    m.add_function(wrap_pyfunction!(splines_py::compute_knots_natural_py, m)?)?;
    
    // Design matrix functions
    m.add_function(wrap_pyfunction!(design_matrix_py::encode_categorical_py, m)?)?;
    m.add_function(wrap_pyfunction!(design_matrix_py::factorize_strings_py, m)?)?;
    m.add_function(wrap_pyfunction!(design_matrix_py::encode_categorical_indices_py, m)?)?;
    m.add_function(wrap_pyfunction!(design_matrix_py::build_cat_cat_interaction_py, m)?)?;
    m.add_function(wrap_pyfunction!(design_matrix_py::build_cat_cont_interaction_py, m)?)?;
    m.add_function(wrap_pyfunction!(design_matrix_py::build_cont_cont_interaction_py, m)?)?;
    m.add_function(wrap_pyfunction!(design_matrix_py::multiply_matrix_by_continuous_py, m)?)?;
    
    // Target encoding
    m.add_function(wrap_pyfunction!(target_encoding_py::target_encode_py, m)?)?;
    m.add_function(wrap_pyfunction!(target_encoding_py::apply_target_encoding_py, m)?)?;
    m.add_function(wrap_pyfunction!(target_encoding_py::target_encode_with_exposure_py, m)?)?;
    m.add_function(wrap_pyfunction!(target_encoding_py::apply_exposure_weighted_target_encoding_py, m)?)?;
    m.add_function(wrap_pyfunction!(target_encoding_py::frequency_encode_py, m)?)?;
    m.add_function(wrap_pyfunction!(target_encoding_py::apply_frequency_encoding_py, m)?)?;
    m.add_function(wrap_pyfunction!(target_encoding_py::target_encode_interaction_py, m)?)?;
    m.add_function(wrap_pyfunction!(target_encoding_py::target_encode_interaction_with_exposure_py, m)?)?;
    
    // Diagnostics
    m.add_function(wrap_pyfunction!(diagnostics_py::compute_calibration_curve_py, m)?)?;
    m.add_function(wrap_pyfunction!(diagnostics_py::compute_discrimination_stats_py, m)?)?;
    m.add_function(wrap_pyfunction!(diagnostics_py::compute_ae_continuous_py, m)?)?;
    m.add_function(wrap_pyfunction!(diagnostics_py::compute_ae_categorical_py, m)?)?;
    m.add_function(wrap_pyfunction!(diagnostics_py::compute_factor_deviance_py, m)?)?;
    m.add_function(wrap_pyfunction!(diagnostics_py::compute_loss_metrics_py, m)?)?;
    m.add_function(wrap_pyfunction!(diagnostics_py::detect_interactions_py, m)?)?;
    m.add_function(wrap_pyfunction!(diagnostics_py::compute_lorenz_curve_py, m)?)?;
    m.add_function(wrap_pyfunction!(diagnostics_py::hosmer_lemeshow_test_py, m)?)?;
    m.add_function(wrap_pyfunction!(diagnostics_py::compute_fit_statistics_py, m)?)?;
    m.add_function(wrap_pyfunction!(diagnostics_py::compute_dataset_metrics_py, m)?)?;
    m.add_function(wrap_pyfunction!(diagnostics_py::compute_residual_summary_py, m)?)?;
    m.add_function(wrap_pyfunction!(diagnostics_py::compute_residual_pattern_py, m)?)?;
    m.add_function(wrap_pyfunction!(diagnostics_py::compute_pearson_residuals_py, m)?)?;
    m.add_function(wrap_pyfunction!(diagnostics_py::compute_deviance_residuals_py, m)?)?;
    m.add_function(wrap_pyfunction!(diagnostics_py::compute_null_deviance_py, m)?)?;
    m.add_function(wrap_pyfunction!(diagnostics_py::compute_unit_deviance_py, m)?)?;
    
    // ONNX export
    m.add_function(wrap_pyfunction!(export_onnx_py::build_onnx_glm_scoring_py, m)?)?;
    m.add_function(wrap_pyfunction!(export_onnx_py::serialize_onnx_graph_py, m)?)?;
    
    Ok(())
}
