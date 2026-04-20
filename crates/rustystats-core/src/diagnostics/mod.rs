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

pub mod calibration;
pub mod ci;
pub mod decile;
mod dispersion;
mod distributions;
pub mod factor_diagnostics;
mod family_params;
mod fit_statistics;
pub mod interactions;
pub mod loss;
mod model_fit;
mod negbinomial;
mod partial_dependence;
mod residuals;
mod vif;

pub use ci::wilson_poisson_rate_ci;

pub use residuals::{
    compute_residual_summary, resid_deviance, resid_pearson, resid_response, resid_working,
    ResidualSummary,
};

pub use dispersion::{estimate_dispersion_deviance, estimate_dispersion_pearson, pearson_chi2};

pub use family_params::{parse_family_params, FamilyParams};

pub use fit_statistics::{compute_fit_statistics, FitStatistics};

pub use model_fit::{
    aic, bic, log_likelihood_binomial, log_likelihood_gamma, log_likelihood_gaussian,
    log_likelihood_poisson, null_deviance, null_deviance_for_family, null_deviance_with_offset,
};

pub use negbinomial::{estimate_theta_moments, estimate_theta_profile, nb_loglikelihood};

pub use loss::{
    compute_family_loss, default_loss_name, gamma_deviance_loss, log_loss, mae, mse,
    negbinomial_deviance_loss, poisson_deviance_loss, rmse, tweedie_deviance_loss,
};

pub use calibration::{
    compute_calibration_curve, compute_calibration_stats, compute_discrimination_stats,
    compute_lorenz_curve, hosmer_lemeshow_test, CalibrationBin, CalibrationStats,
    DiscriminationStats, HosmerLemeshowResult, LorenzPoint,
};

pub use decile::{compute_ae_by_decile, DecileMetricsRaw};

pub use partial_dependence::partial_dependence_categorical_batch;

pub use factor_diagnostics::{
    compute_ae_categorical, compute_ae_categorical_batch, compute_ae_categorical_from_codes,
    compute_ae_continuous, compute_ae_continuous_batch, compute_categorical_distribution,
    compute_continuous_stats, compute_factor_deviance, compute_factor_deviance_batch,
    compute_factor_deviance_batch_from_codes, compute_factor_significance_batch,
    compute_glm_deviance, compute_residual_pattern_categorical,
    compute_residual_pattern_continuous, compute_residual_pattern_continuous_batch,
    ActualExpectedBin, CategoricalDistribution, ContinuousStats, DevianceByLevel, FactorConfig,
    FactorDevianceResult, FactorSignificanceRaw, FactorType, LevelStats, Percentiles,
    ResidualPattern,
};

pub use interactions::{detect_interactions, FactorData, InteractionCandidate, InteractionConfig};

pub use distributions::{chi2_cdf, f_cdf, t_cdf};

pub use vif::correlation_and_vif;
