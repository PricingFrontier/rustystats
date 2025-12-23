// =============================================================================
// Model Diagnostics
// =============================================================================
//
// This module provides diagnostic tools for assessing GLM model quality:
//
// - RESIDUALS: Different ways to measure prediction errors
// - DISPERSION: Estimating the scale parameter φ
// - MODEL FIT: AIC, BIC, log-likelihood, and goodness-of-fit measures
//
// These diagnostics help answer:
// - Is my model a good fit for the data?
// - Are there patterns in the residuals suggesting model misspecification?
// - How does this model compare to alternatives?
//
// STATSMODELS COMPATIBILITY:
// --------------------------
// Method names and calculations follow statsmodels conventions:
// - resid_response: Raw residuals (y - μ)
// - resid_pearson: Standardized by variance
// - resid_deviance: Based on deviance contributions
// - resid_working: Used internally in IRLS
//
// =============================================================================

mod residuals;
mod dispersion;
mod model_fit;

pub use residuals::{
    resid_response,
    resid_pearson,
    resid_deviance,
    resid_working,
};

pub use dispersion::{
    estimate_dispersion_pearson,
    estimate_dispersion_deviance,
    pearson_chi2,
};

pub use model_fit::{
    log_likelihood_gaussian,
    log_likelihood_poisson,
    log_likelihood_binomial,
    log_likelihood_gamma,
    aic,
    bic,
    null_deviance,
};
