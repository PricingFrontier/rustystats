// =============================================================================
// IRLS Residual / Working-Response Helpers (Python Bindings)
// =============================================================================
//
// Exposes the IRLS working response z and combined working weight w for an
// arbitrary linear predictor η — without requiring a fitted GLM. Useful for
// link-scale boosting loops (e.g. destyler) that iterate on residuals between
// layers and need to match what the IRLS solver would compute at each step.
//
// This binding is a thin bridge: it dispatches the family/link strings via
// the shared helpers in families_py and delegates all validation and math to
// rustystats-core::solvers. No algorithm logic lives here.
// =============================================================================

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use rustystats_core::constants::MIN_IRLS_WEIGHT;
use rustystats_core::solvers::{
    compute_irls_weights, validate_residual_inputs, IRLSWeightResult, ValidatedInputs,
};

use crate::families_py::{
    default_link_name, family_from_name_with_tweedie_support, link_from_name,
};

/// Compute IRLS working response z and combined working weight w.
///
/// Parameters mirror `fit_glm_py` for consistency: `family`/`link` are strings,
/// `var_power` is used only for Tweedie, `theta` only for Negative Binomial.
/// `link=None` resolves to the family's canonical link.
///
/// `eta` is the linear predictor *excluding* offset; the offset is added
/// internally to form μ = g⁻¹(η + offset). This matches the boosting use case
/// where η is the running model prediction on the link scale and the offset
/// (e.g. log-exposure) is held fixed across iterations.
///
/// Returns a tuple `(z, w)` of f64 numpy arrays of length n.
#[pyfunction]
#[pyo3(signature = (y, eta, family, link=None, var_power=1.5, theta=1.0, offset=None, weights=None, allow_extended_tweedie=false))]
pub fn working_response_weights_py<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<f64>,
    eta: PyReadonlyArray1<f64>,
    family: &str,
    link: Option<&str>,
    var_power: f64,
    theta: f64,
    offset: Option<PyReadonlyArray1<f64>>,
    weights: Option<PyReadonlyArray1<f64>>,
    allow_extended_tweedie: bool,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let fam =
        family_from_name_with_tweedie_support(family, var_power, theta, allow_extended_tweedie)?;
    let lnk = link_from_name(link.unwrap_or(default_link_name(family)))?;

    let y_arr = y.as_array().to_owned();
    let eta_arr = eta.as_array().to_owned();
    let offset_owned = offset.map(|o| o.as_array().to_owned());
    let weights_owned = weights.map(|w| w.as_array().to_owned());

    let (working_response, combined_weights) = py
        .detach(|| -> Result<_, rustystats_core::error::RustyStatsError> {
            let ValidatedInputs {
                offset: offset_full,
                prior_weights,
            } = validate_residual_inputs(
                &y_arr,
                &eta_arr,
                offset_owned.as_ref(),
                weights_owned.as_ref(),
            )?;

            // Build the full linear predictor (η + offset) and μ. Passing
            // η_full + the same offset into compute_irls_weights makes its
            // internal `working_response = (η_full − offset) + (y − μ)·g'(μ)`
            // equal `eta + (y − μ)·g'(μ)`, which is the public formula with
            // η = user's η-without-offset.
            let eta_full = &eta_arr + &offset_full;
            let mu = fam.clamp_mu(&lnk.inverse(&eta_full));
            let IRLSWeightResult {
                combined_weights,
                working_response,
                ..
            } = compute_irls_weights(
                &y_arr,
                &mu,
                &eta_full,
                &offset_full,
                &prior_weights,
                fam.as_ref(),
                lnk.as_ref(),
                MIN_IRLS_WEIGHT,
            )?;
            Ok((working_response, combined_weights))
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok((
        working_response.into_pyarray(py),
        combined_weights.into_pyarray(py),
    ))
}
