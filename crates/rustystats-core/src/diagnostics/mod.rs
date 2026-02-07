// =============================================================================
// Model Diagnostics
// =============================================================================
//
// This module provides diagnostic tools for assessing GLM model quality:
//
// - RESIDUALS: Different ways to measure prediction errors
// - DISPERSION: Estimating the scale parameter φ
// - MODEL FIT: AIC, BIC, log-likelihood, and goodness-of-fit measures
// - LOSS: Family-specific loss functions for evaluation
// - CALIBRATION: A/E ratios, calibration curves, discrimination metrics
// - FACTOR DIAGNOSTICS: Per-factor analysis for fitted and unfitted variables
// - INTERACTION DETECTION: Greedy residual-based interaction discovery
//
// These diagnostics help answer:
// - Is my model a good fit for the data?
// - Are there patterns in the residuals suggesting model misspecification?
// - How does this model compare to alternatives?
// - Which factors need improvement?
// - Are there missing interactions?
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
mod negbinomial;
mod distributions;
pub mod loss;
pub mod calibration;
pub mod factor_diagnostics;
pub mod interactions;

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
    null_deviance_with_offset,
    null_deviance_for_family,
};

pub use negbinomial::{
    nb_loglikelihood,
    estimate_theta_profile,
    estimate_theta_moments,
};

pub use loss::{
    mse, rmse, mae,
    poisson_deviance_loss, gamma_deviance_loss, log_loss,
    tweedie_deviance_loss, negbinomial_deviance_loss,
    compute_family_loss, default_loss_name,
};

pub use calibration::{
    CalibrationStats, CalibrationBin, compute_calibration_stats, compute_calibration_curve,
    HosmerLemeshowResult, hosmer_lemeshow_test,
    DiscriminationStats, compute_discrimination_stats,
    LorenzPoint, compute_lorenz_curve,
};

pub use factor_diagnostics::{
    FactorType, FactorConfig,
    ContinuousStats, Percentiles, compute_continuous_stats,
    CategoricalDistribution, LevelStats, compute_categorical_distribution,
    ActualExpectedBin, compute_ae_continuous, compute_ae_categorical,
    ResidualPattern, compute_residual_pattern_continuous, compute_residual_pattern_categorical,
    DevianceByLevel, FactorDevianceResult, compute_factor_deviance,
};

pub use interactions::{
    InteractionCandidate, InteractionConfig, FactorData, detect_interactions,
};

pub use distributions::{
    chi2_cdf, t_cdf, f_cdf,
};
