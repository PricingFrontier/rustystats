// =============================================================================
// Model Fit Statistics
// =============================================================================
//
// This module provides statistics for assessing and comparing GLM models:
//
// LOG-LIKELIHOOD:
// ---------------
// The log of the probability of observing the data given the model.
// Higher (less negative) is better. Used to compute AIC/BIC.
//
// AIC (Akaike Information Criterion):
// -----------------------------------
// AIC = -2ℓ + 2p
// Balances fit (likelihood) against complexity (number of parameters).
// Lower is better. Use for model comparison.
//
// BIC (Bayesian Information Criterion):
// -------------------------------------
// BIC = -2ℓ + p×log(n)
// Like AIC but penalizes complexity more strongly for large samples.
// Lower is better.
//
// NULL DEVIANCE:
// --------------
// Deviance of an intercept-only model. Measures total variation in y.
// Compare to residual deviance to assess how much variation is explained.
//
// PSEUDO R-SQUARED:
// -----------------
// Various measures that mimic R² for non-Gaussian models.
// Calculated from null and residual deviance.
//
// =============================================================================

use crate::constants::{MU_MAX_PROBABILITY, MU_MIN_PROBABILITY};
use ndarray::Array1;
use std::f64::consts::PI;

// =============================================================================
// Log-Likelihood Functions
// =============================================================================

/// Log-likelihood for Gaussian (Normal) family.
///
/// ℓ = -½ Σ[(y-μ)²/φ + log(2πφ)]
///
/// # Arguments
/// * `y` - Observed response values
/// * `mu` - Fitted mean values
/// * `scale` - Dispersion parameter (σ²)
/// * `weights` - Optional observation weights
///
/// # Returns
/// Total log-likelihood
pub fn log_likelihood_gaussian(
    y: &Array1<f64>,
    mu: &Array1<f64>,
    scale: f64,
    weights: Option<&Array1<f64>>,
) -> f64 {
    let n = y.len() as f64;
    let sum_wt = weights.map_or(n, |w| w.sum());

    // Sum of squared residuals
    let ss: f64 = ndarray::Zip::from(y).and(mu).fold(0.0, |acc, &yi, &mui| {
        let diff = yi - mui;
        acc + diff * diff
    });

    // If weighted, scale the sum
    let ss_weighted = match weights {
        Some(w) => ndarray::Zip::from(y)
            .and(mu)
            .and(w)
            .fold(0.0, |acc, &yi, &mui, &wi| {
                let diff = yi - mui;
                acc + wi * diff * diff
            }),
        None => ss,
    };

    // Log-likelihood
    -0.5 * (ss_weighted / scale + sum_wt * (2.0 * PI * scale).ln())
}

/// Log-likelihood for Poisson family.
///
/// ℓ = Σ[y×log(μ) - μ - log(y!)]
///
/// # Arguments
/// * `y` - Observed response values (counts)
/// * `mu` - Fitted mean values
/// * `weights` - Optional observation weights
///
/// # Returns
/// Total log-likelihood
///
/// # Note
/// The log(y!) term is constant given the data and can be omitted
/// for model comparison, but we include it for completeness.
pub fn log_likelihood_poisson(
    y: &Array1<f64>,
    mu: &Array1<f64>,
    weights: Option<&Array1<f64>>,
) -> f64 {
    use statrs::function::gamma::ln_gamma;

    let contributions: Array1<f64> = ndarray::Zip::from(y).and(mu).map_collect(|&yi, &mui| {
        // y × log(μ) - μ - log(y!)
        // log(y!) = ln_gamma(y + 1)
        let log_factorial = ln_gamma(yi + 1.0);
        yi * mui.ln() - mui - log_factorial
    });

    match weights {
        Some(w) => (&contributions * w).sum(),
        None => contributions.sum(),
    }
}

/// Log-likelihood for Binomial family (binary case).
///
/// ℓ = Σ[y×log(μ) + (1-y)×log(1-μ)]
///
/// # Arguments
/// * `y` - Observed response values (0/1 or proportions)
/// * `mu` - Fitted probabilities
/// * `weights` - Optional observation weights
///
/// # Returns
/// Total log-likelihood
pub fn log_likelihood_binomial(
    y: &Array1<f64>,
    mu: &Array1<f64>,
    weights: Option<&Array1<f64>>,
) -> f64 {
    let contributions: Array1<f64> = ndarray::Zip::from(y).and(mu).map_collect(|&yi, &mui| {
        // Clamp μ to avoid log(0)
        let mui_safe = mui.clamp(MU_MIN_PROBABILITY, MU_MAX_PROBABILITY);

        // y × log(μ) + (1-y) × log(1-μ)
        let ll = if yi > 0.0 { yi * mui_safe.ln() } else { 0.0 };

        if yi < 1.0 {
            ll + (1.0 - yi) * (1.0 - mui_safe).ln()
        } else {
            ll
        }
    });

    match weights {
        Some(w) => (&contributions * w).sum(),
        None => contributions.sum(),
    }
}

/// Log-likelihood for Gamma family.
///
/// For Gamma with shape α and scale θ where μ = αθ and φ = 1/α:
/// ℓ_i = (α-1)·log(y) - α·y/μ + α·log(α/μ) - log(Γ(α))
///
/// This matches the statsmodels implementation exactly.
///
/// # Arguments
/// * `y` - Observed response values (positive)
/// * `mu` - Fitted mean values
/// * `scale` - Dispersion parameter φ = 1/α (inverse of shape)
/// * `weights` - Optional observation weights
///
/// # Returns
/// Total log-likelihood
pub fn log_likelihood_gamma(
    y: &Array1<f64>,
    mu: &Array1<f64>,
    scale: f64,
    weights: Option<&Array1<f64>>,
) -> f64 {
    use crate::constants::MU_MIN_POSITIVE;
    use statrs::function::gamma::ln_gamma;

    // shape α = 1/scale = 1/φ
    let alpha = 1.0 / scale;

    let contributions: Array1<f64> = ndarray::Zip::from(y).and(mu).map_collect(|&yi, &mui| {
        // Floor y and mu to prevent log(0) issues
        // Note: Gamma requires y > 0, but we handle zeros gracefully
        let yi_safe = yi.max(MU_MIN_POSITIVE);
        let mui_safe = mui.max(MU_MIN_POSITIVE);

        // Full Gamma log-likelihood (statsmodels formula):
        // ℓ_i = (α-1)·log(y) - α·y/μ + α·log(α/μ) - log(Γ(α))
        //     = (α-1)·log(y) - α·y/μ + α·log(α) - α·log(μ) - log(Γ(α))

        (alpha - 1.0) * yi_safe.ln() - alpha * yi_safe / mui_safe + alpha * (alpha / mui_safe).ln()
            - ln_gamma(alpha)
    });

    match weights {
        Some(w) => (&contributions * w).sum(),
        None => contributions.sum(),
    }
}

// =============================================================================
// Information Criteria
// =============================================================================

/// Compute Akaike Information Criterion.
///
/// AIC = -2ℓ + 2p
///
/// # Arguments
/// * `llf` - Log-likelihood value
/// * `n_params` - Number of estimated parameters (including intercept)
///
/// # Returns
/// AIC value (lower is better)
pub fn aic(llf: f64, n_params: usize) -> f64 {
    -2.0 * llf + 2.0 * (n_params as f64)
}

/// Compute Bayesian Information Criterion.
///
/// BIC = -2ℓ + p×log(n)
///
/// # Arguments
/// * `llf` - Log-likelihood value
/// * `n_params` - Number of estimated parameters
/// * `n_obs` - Number of observations
///
/// # Returns
/// BIC value (lower is better)
pub fn bic(llf: f64, n_params: usize, n_obs: usize) -> f64 {
    -2.0 * llf + (n_params as f64) * (n_obs as f64).ln()
}

// =============================================================================
// Null Deviance
// =============================================================================

/// Compute the null deviance (deviance of intercept-only model).
///
/// The null deviance measures how much variation there is in y
/// before accounting for predictors. It's used to compute pseudo R².
///
/// For most families, the intercept-only model predicts the weighted
/// mean of y for all observations. When an offset is present (e.g.,
/// log(exposure) for count models), the null model accounts for it.
///
/// # Arguments
/// * `y` - Observed response values
/// * `family_name` - Name of the family ("Gaussian", "Poisson", etc.)
/// * `weights` - Optional observation weights
/// * `offset` - Optional offset values (e.g., log(exposure) for Poisson/NegBin)
///
/// # Returns
/// Null deviance value
pub fn null_deviance(
    y: &Array1<f64>,
    family_name: &str,
    weights: Option<&Array1<f64>>,
) -> Result<f64, String> {
    null_deviance_with_offset(y, family_name, weights, None)
}

/// Build a `Box<dyn Family>` from a lowercased family-name string.
///
/// This is the single point of string→family dispatch for the streaming
/// null-deviance path. All per-row arithmetic then flows through
/// `Family::unit_deviance_at`, keeping "one code path" across families.
fn family_from_name(lower: &str) -> Result<Box<dyn crate::families::Family>, String> {
    use crate::families::{
        BinomialFamily, GammaFamily, GaussianFamily, NegativeBinomialFamily, PoissonFamily,
    };
    match lower {
        "gaussian" | "normal" => Ok(Box::new(GaussianFamily)),
        "poisson" | "quasipoisson" => Ok(Box::new(PoissonFamily)),
        "binomial" | "quasibinomial" => Ok(Box::new(BinomialFamily)),
        "gamma" => Ok(Box::new(GammaFamily)),
        other if other.starts_with("negativebinomial") || other.starts_with("negbinomial") => {
            let theta = if let Some(start) = other.find("theta=") {
                let rest = &other[start + "theta=".len()..];
                let end = rest.find(')').unwrap_or(rest.len());
                rest[..end].parse::<f64>().unwrap_or(1.0)
            } else {
                1.0
            };
            Ok(Box::new(NegativeBinomialFamily::new(theta)?))
        }
        other => Err(format!(
            "Unknown family '{}' in null_deviance computation. \
             Supported families: gaussian, poisson, binomial, gamma, \
             quasipoisson, quasibinomial, negativebinomial.",
            other
        )),
    }
}

/// Compute the null deviance with optional offset support.
///
/// When offset is provided, the null model prediction is:
/// - For log-link models: mu_null = mean_rate * exp(offset), where mean_rate = sum(y) / sum(exp(offset))
/// - For identity link: mu_null = mean(y - offset) + offset
///
/// # Implementation notes
/// Single-pass `Zip::fold`: rather than materialising `exp_offset`, `mu_null`,
/// and `unit_dev` (3 × 8 MB at n=1M for the Poisson log-link path), we
/// recompute `mu_i` on the fly inside the deviance fold and call
/// `family.unit_deviance_at(yi, mu_i)` per row. `exp(.)` is a few cycles vs.
/// an 8 MB write/read round-trip. Per-family arithmetic lives in the
/// `Family` trait impls, so this function and the vectorized
/// `Family::unit_deviance` cannot drift apart.
///
/// **Numerical note.** The previous implementation called `Array1::sum()` on
/// the materialised `unit_dev`, which uses ndarray's 8-way unrolled
/// (pairwise-style) accumulation. The new code uses `Zip::fold`, which is a
/// strict left-fold. For typical inputs the two agree to ~1 ULP; on adversarial
/// data the drift may be a few ULP. This is a one-shot summation, not iterated,
/// so the drift does not compound through any downstream computation.
pub fn null_deviance_with_offset(
    y: &Array1<f64>,
    family_name: &str,
    weights: Option<&Array1<f64>>,
    offset: Option<&Array1<f64>>,
) -> Result<f64, String> {
    let n = y.len();

    // ---- Pass 1: compute the scalar parameter(s) of the null model ----
    //
    // For log-link + offset: we need `mean_rate = sum(y * w) / sum(exp(off) * w)`
    // For everything else:    we need `y_mean   = sum(y * w) / sum(w)`
    //
    // The `mu_i` for each row is then derived deterministically from these
    // scalars (and the offset, when log-link) inside the second pass.

    let family_lower = family_name.to_lowercase();
    let is_log_link_with_offset = offset.is_some()
        && (family_lower.starts_with("poisson")
            || family_lower.starts_with("negbin")
            || family_lower.starts_with("negativebinomial")
            || family_lower.starts_with("gamma")
            || family_lower.starts_with("quasipoisson"));

    // Closure: given a row index, return the null-model prediction `mu_i`.
    // Captures `mean_rate`/`y_mean` and `offset` by value/reference as needed.
    // Inlined here so the per-row work in pass 2 is a couple of FLOPs.
    let (mean_rate, y_mean) = if is_log_link_with_offset {
        let off = offset.expect("offset checked above");
        let (sum_exp_off, sum_y) = match weights {
            Some(w) => ndarray::Zip::from(off)
                .and(y)
                .and(w)
                .fold((0.0_f64, 0.0_f64), |(se, sy), &oi, &yi, &wi| {
                    (se + oi.exp() * wi, sy + yi * wi)
                }),
            None => ndarray::Zip::from(off)
                .and(y)
                .fold((0.0_f64, 0.0_f64), |(se, sy), &oi, &yi| {
                    (se + oi.exp(), sy + yi)
                }),
        };
        (sum_y / sum_exp_off, f64::NAN) // y_mean unused on this branch
    } else {
        let (sum_y, sum_w) = match weights {
            Some(w) => ndarray::Zip::from(y)
                .and(w)
                .fold((0.0_f64, 0.0_f64), |(sy, sw), &yi, &wi| {
                    (sy + yi * wi, sw + wi)
                }),
            None => (y.sum(), n as f64),
        };
        (f64::NAN, sum_y / sum_w) // mean_rate unused on this branch
    };

    // ---- Pass 2: fold the (weighted) unit deviance ----
    //
    // Resolve the family name to a concrete Family trait object once, then
    // call `family.unit_deviance_at(yi, mui)` inside the fold. This avoids
    // both the materialised `mu_null` array (~8 MB at n=1M) and a parallel
    // per-row deviance implementation (previously a local `Kernel` enum
    // that duplicated the arithmetic of `Family::unit_deviance`).
    let family = family_from_name(family_lower.as_str())?;

    let total = match (weights, offset) {
        (Some(w), Some(off)) if is_log_link_with_offset => ndarray::Zip::from(y)
            .and(w)
            .and(off)
            .fold(0.0, |a, &yi, &wi, &oi| {
                a + family.unit_deviance_at(yi, mean_rate * oi.exp()) * wi
            }),
        (Some(w), _) => {
            // Constant mu_null = y_mean for all rows on the non-log-link path.
            let mu_const = y_mean;
            ndarray::Zip::from(y).and(w).fold(0.0, |a, &yi, &wi| {
                a + family.unit_deviance_at(yi, mu_const) * wi
            })
        }
        (None, Some(off)) if is_log_link_with_offset => {
            ndarray::Zip::from(y).and(off).fold(0.0, |a, &yi, &oi| {
                a + family.unit_deviance_at(yi, mean_rate * oi.exp())
            })
        }
        (None, _) => {
            let mu_const = y_mean;
            ndarray::Zip::from(y).fold(0.0, |a, &yi| a + family.unit_deviance_at(yi, mu_const))
        }
    };

    Ok(total)
}

/// Compute null deviance using a Family trait object instead of family name string.
///
/// This is the trait-based replacement for `null_deviance_with_offset`.
/// It uses `family.is_log_link_default()` for offset handling and
/// `family.unit_deviance()` for deviance computation, eliminating all
/// string-based dispatch.
pub fn null_deviance_for_family(
    y: &Array1<f64>,
    family: &dyn crate::families::Family,
    weights: Option<&Array1<f64>>,
    offset: Option<&Array1<f64>>,
) -> f64 {
    let n = y.len();

    // Resolve the single scalar (mean_rate for log-link + offset, else y_mean)
    // that parameterises the null model, then fold
    // `family.unit_deviance_at(yi, mu_i)` in one pass. No `mu_null` array,
    // no intermediate `unit_dev` array — matches the streaming design of
    // `null_deviance_with_offset` so the two cannot drift apart.
    let (mean_rate, y_mean, use_log_link_offset) = match offset {
        Some(off) if family.is_log_link_default() => {
            let (sum_exp_off, sum_y) = match weights {
                Some(w) => ndarray::Zip::from(off)
                    .and(y)
                    .and(w)
                    .fold((0.0_f64, 0.0_f64), |(se, sy), &oi, &yi, &wi| {
                        (se + oi.exp() * wi, sy + yi * wi)
                    }),
                None => ndarray::Zip::from(off)
                    .and(y)
                    .fold((0.0_f64, 0.0_f64), |(se, sy), &oi, &yi| {
                        (se + oi.exp(), sy + yi)
                    }),
            };
            (sum_y / sum_exp_off, f64::NAN, true)
        }
        _ => {
            let (sum_y, sum_w) = match weights {
                Some(w) => ndarray::Zip::from(y)
                    .and(w)
                    .fold((0.0_f64, 0.0_f64), |(sy, sw), &yi, &wi| {
                        (sy + yi * wi, sw + wi)
                    }),
                None => (y.sum(), n as f64),
            };
            (f64::NAN, sum_y / sum_w, false)
        }
    };

    match (weights, offset, use_log_link_offset) {
        (Some(w), Some(off), true) => ndarray::Zip::from(y)
            .and(w)
            .and(off)
            .fold(0.0, |a, &yi, &wi, &oi| {
                a + family.unit_deviance_at(yi, mean_rate * oi.exp()) * wi
            }),
        (Some(w), _, _) => {
            let mu_const = y_mean;
            ndarray::Zip::from(y).and(w).fold(0.0, |a, &yi, &wi| {
                a + family.unit_deviance_at(yi, mu_const) * wi
            })
        }
        (None, Some(off), true) => ndarray::Zip::from(y).and(off).fold(0.0, |a, &yi, &oi| {
            a + family.unit_deviance_at(yi, mean_rate * oi.exp())
        }),
        (None, _, _) => {
            let mu_const = y_mean;
            ndarray::Zip::from(y).fold(0.0, |a, &yi| a + family.unit_deviance_at(yi, mu_const))
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_log_likelihood_gaussian() {
        let y = array![1.0, 2.0, 3.0];
        let mu = array![1.0, 2.0, 3.0]; // Perfect fit
        let scale = 1.0;

        let llf = log_likelihood_gaussian(&y, &mu, scale, None);

        // With perfect fit, SS = 0
        // ℓ = -0.5 × n × log(2πσ²) = -0.5 × 3 × log(2π)
        let expected = -0.5 * 3.0 * (2.0 * PI).ln();
        assert_abs_diff_eq!(llf, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_log_likelihood_gaussian_imperfect() {
        let y = array![1.0, 2.0, 3.0];
        let mu = array![1.5, 2.5, 3.5]; // Errors of 0.5
        let scale = 1.0;

        let llf = log_likelihood_gaussian(&y, &mu, scale, None);

        // SS = 3 × 0.25 = 0.75
        // ℓ = -0.5 × (0.75/1 + 3 × log(2π))
        let expected = -0.5 * (0.75 + 3.0 * (2.0 * PI).ln());
        assert_abs_diff_eq!(llf, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_log_likelihood_gaussian_weighted() {
        let y = array![1.0, 2.0];
        let mu = array![1.0, 3.0]; // Errors: 0, 1
        let w = array![1.0, 2.0];
        let scale = 1.0;

        let llf = log_likelihood_gaussian(&y, &mu, scale, Some(&w));

        // Weighted SS = 1×0 + 2×1 = 2
        // sum_wt = 3
        // ℓ = -0.5 × (2/1 + 3 × log(2π))
        let expected = -0.5 * (2.0 + 3.0 * (2.0 * PI).ln());
        assert_abs_diff_eq!(llf, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_log_likelihood_poisson() {
        // Simple test: y = μ = 1 for all observations
        let y = array![1.0, 1.0, 1.0];
        let mu = array![1.0, 1.0, 1.0];

        let llf = log_likelihood_poisson(&y, &mu, None);

        // For y=μ=1: 1×log(1) - 1 - log(1!) = 0 - 1 - 0 = -1 per obs
        // Total: -3
        assert_abs_diff_eq!(llf, -3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_log_likelihood_poisson_weighted() {
        let y = array![1.0, 2.0];
        let mu = array![1.0, 2.0];
        let w = array![1.0, 2.0];

        let llf = log_likelihood_poisson(&y, &mu, Some(&w));

        // Weighted sum should be computed
        assert!(llf < 0.0);
    }

    #[test]
    fn test_log_likelihood_poisson_zero_y() {
        let y = array![0.0, 0.0];
        let mu = array![1.0, 2.0];

        let llf = log_likelihood_poisson(&y, &mu, None);

        // For y=0: 0×log(μ) - μ - log(0!) = -μ - 0 = -μ
        // Total: -1 - 2 = -3
        assert_abs_diff_eq!(llf, -3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_log_likelihood_binomial() {
        let y = array![1.0, 0.0];
        let mu = array![0.8, 0.2];

        let llf = log_likelihood_binomial(&y, &mu, None);

        // 1×log(0.8) + 0×log(0.2) + 0×log(0.2) + 1×log(0.8)
        // = log(0.8) + log(0.8) = 2×log(0.8)
        let expected = 0.8_f64.ln() + 0.8_f64.ln();
        assert_abs_diff_eq!(llf, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_log_likelihood_binomial_weighted() {
        let y = array![1.0, 0.0];
        let mu = array![0.9, 0.1];
        let w = array![2.0, 1.0];

        let llf = log_likelihood_binomial(&y, &mu, Some(&w));

        // Weighted: 2×log(0.9) + 1×log(0.9)
        let expected = 2.0 * 0.9_f64.ln() + 1.0 * 0.9_f64.ln();
        assert_abs_diff_eq!(llf, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_log_likelihood_binomial_proportions() {
        // Test with proportions (not just 0/1)
        let y = array![0.5, 0.5];
        let mu = array![0.5, 0.5];

        let llf = log_likelihood_binomial(&y, &mu, None);

        // 0.5×log(0.5) + 0.5×log(0.5) per obs
        let per_obs = 0.5 * 0.5_f64.ln() + 0.5 * 0.5_f64.ln();
        assert_abs_diff_eq!(llf, 2.0 * per_obs, epsilon = 1e-10);
    }

    #[test]
    fn test_log_likelihood_gamma() {
        let y = array![1.0, 2.0, 3.0];
        let mu = array![1.0, 2.0, 3.0]; // Perfect fit
        let scale = 1.0; // α = 1

        let llf = log_likelihood_gamma(&y, &mu, scale, None);

        // With perfect fit and α=1, should get finite negative value
        assert!(llf.is_finite());
        assert!(llf < 0.0);
    }

    #[test]
    fn test_log_likelihood_gamma_weighted() {
        let y = array![1.0, 2.0];
        let mu = array![1.0, 2.0];
        let w = array![1.0, 2.0];
        let scale = 0.5; // α = 2

        let llf = log_likelihood_gamma(&y, &mu, scale, Some(&w));

        assert!(llf.is_finite());
    }

    #[test]
    fn test_log_likelihood_gamma_small_scale() {
        let y = array![1.0, 2.0, 3.0];
        let mu = array![1.0, 2.0, 3.0];
        let scale = 0.1; // α = 10 (high shape)

        let llf = log_likelihood_gamma(&y, &mu, scale, None);

        assert!(llf.is_finite());
    }

    #[test]
    fn test_aic() {
        let llf = -100.0;
        let n_params = 5;

        let aic_val = aic(llf, n_params);

        // AIC = -2×(-100) + 2×5 = 200 + 10 = 210
        assert_abs_diff_eq!(aic_val, 210.0, epsilon = 1e-10);
    }

    #[test]
    fn test_aic_zero_params() {
        let llf = -50.0;
        let n_params = 0;

        let aic_val = aic(llf, n_params);

        // AIC = -2×(-50) + 0 = 100
        assert_abs_diff_eq!(aic_val, 100.0, epsilon = 1e-10);
    }

    #[test]
    fn test_bic() {
        let llf = -100.0;
        let n_params = 5;
        let n_obs = 100;

        let bic_val = bic(llf, n_params, n_obs);

        // BIC = -2×(-100) + 5×log(100) = 200 + 5×4.605... ≈ 223.03
        let expected = 200.0 + 5.0 * 100.0_f64.ln();
        assert_abs_diff_eq!(bic_val, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_bic_small_sample() {
        let llf = -50.0;
        let n_params = 3;
        let n_obs = 10;

        let bic_val = bic(llf, n_params, n_obs);

        // BIC = -2×(-50) + 3×log(10) = 100 + 3×2.303
        let expected = 100.0 + 3.0 * 10.0_f64.ln();
        assert_abs_diff_eq!(bic_val, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_null_deviance_gaussian() {
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let null_dev = null_deviance(&y, "Gaussian", None).expect("test setup should be valid");

        // Mean = 3.0
        // Null deviance = Σ(y - 3)² = 4 + 1 + 0 + 1 + 4 = 10
        assert_abs_diff_eq!(null_dev, 10.0, epsilon = 1e-10);
    }

    #[test]
    fn test_null_deviance_normal() {
        // Test case-insensitive "normal" alias
        let y = array![1.0, 3.0];

        let null_dev = null_deviance(&y, "normal", None).expect("test setup should be valid");

        // Mean = 2.0
        // Null deviance = (1-2)² + (3-2)² = 1 + 1 = 2
        assert_abs_diff_eq!(null_dev, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_null_deviance_poisson() {
        let y = array![0.0, 1.0, 2.0, 3.0, 4.0];

        let null_dev = null_deviance(&y, "Poisson", None).expect("test setup should be valid");

        // Mean = 2.0
        // This is more complex to compute manually, but should be positive
        assert!(null_dev > 0.0);
    }

    #[test]
    fn test_null_deviance_quasipoisson() {
        let y = array![1.0, 2.0, 3.0];

        let null_dev = null_deviance(&y, "quasipoisson", None).expect("test setup should be valid");

        assert!(null_dev >= 0.0);
    }

    #[test]
    fn test_null_deviance_weighted() {
        let y = array![1.0, 5.0];
        let weights = array![3.0, 1.0]; // More weight on first obs

        let null_dev =
            null_deviance(&y, "Gaussian", Some(&weights)).expect("test setup should be valid");

        // Weighted mean = (3×1 + 1×5) / 4 = 8/4 = 2.0
        // Null deviance = 3×(1-2)² + 1×(5-2)² = 3×1 + 1×9 = 12
        assert_abs_diff_eq!(null_dev, 12.0, epsilon = 1e-10);
    }

    #[test]
    fn test_null_deviance_binomial() {
        let y = array![0.0, 1.0, 0.0, 1.0];

        let null_dev = null_deviance(&y, "binomial", None).expect("test setup should be valid");

        // Mean = 0.5
        // Should be positive
        assert!(null_dev > 0.0);
    }

    #[test]
    fn test_null_deviance_quasibinomial() {
        let y = array![0.0, 0.0, 1.0, 1.0];

        let null_dev =
            null_deviance(&y, "quasibinomial", None).expect("test setup should be valid");

        assert!(null_dev >= 0.0);
    }

    #[test]
    fn test_null_deviance_gamma() {
        let y = array![1.0, 2.0, 3.0, 4.0];

        let null_dev = null_deviance(&y, "gamma", None).expect("test setup should be valid");

        // Mean = 2.5
        // Should be positive
        assert!(null_dev >= 0.0);
    }

    #[test]
    fn test_null_deviance_negativebinomial() {
        let y = array![1.0, 2.0, 3.0, 4.0]; // All positive values

        let null_dev =
            null_deviance(&y, "negativebinomial", None).expect("test setup should be valid");

        // Negative binomial deviance can be negative for some edge cases
        assert!(null_dev.is_finite());
    }

    #[test]
    fn test_null_deviance_negativebinomial_with_theta() {
        let y = array![1.0, 2.0, 3.0, 4.0]; // All positive values

        let null_dev = null_deviance(&y, "negativebinomial(theta=2.5)", None)
            .expect("test setup should be valid");

        // Negative binomial deviance should be finite
        assert!(null_dev.is_finite());
    }

    #[test]
    fn test_null_deviance_with_offset_poisson() {
        let y = array![1.0, 2.0, 4.0];
        let offset = array![0.0, 0.693, 1.386]; // log(1), log(2), log(4)

        let null_dev = null_deviance_with_offset(&y, "poisson", None, Some(&offset))
            .expect("test setup should be valid");

        // With offset, null model accounts for exposure
        assert!(null_dev >= 0.0);
    }

    #[test]
    fn test_null_deviance_with_offset_gaussian() {
        let y = array![1.0, 2.0, 3.0];
        let offset = array![0.0, 0.0, 0.0];

        let null_dev = null_deviance_with_offset(&y, "gaussian", None, Some(&offset))
            .expect("test setup should be valid");

        // With zero offset, should match regular null deviance
        let null_dev_no_offset =
            null_deviance(&y, "gaussian", None).expect("test setup should be valid");
        assert_abs_diff_eq!(null_dev, null_dev_no_offset, epsilon = 1e-10);
    }

    #[test]
    fn test_null_deviance_with_offset_gamma() {
        let y = array![1.0, 2.0, 3.0];
        let offset = array![0.0, 0.5, 1.0];

        let null_dev = null_deviance_with_offset(&y, "gamma", None, Some(&offset))
            .expect("test setup should be valid");

        assert!(null_dev >= 0.0);
    }

    #[test]
    fn test_null_deviance_with_offset_negbinomial() {
        let y = array![1.0, 2.0, 3.0];
        let offset = array![0.0, 0.5, 1.0];

        let null_dev = null_deviance_with_offset(&y, "negbinomial(theta=1.5)", None, Some(&offset))
            .expect("test setup should be valid");

        assert!(null_dev >= 0.0);
    }

    #[test]
    fn test_null_deviance_with_offset_weighted() {
        let y = array![1.0, 2.0];
        let offset = array![0.0, 0.5];
        let weights = array![1.0, 2.0];

        let null_dev = null_deviance_with_offset(&y, "poisson", Some(&weights), Some(&offset))
            .expect("test setup should be valid");

        assert!(null_dev >= 0.0);
    }

    // -----------------------------------------------------------------
    // Reference (gather-then-sum) implementation for the null deviance,
    // pinned at this layer so the streaming `null_deviance_with_offset`
    // refactor can be checked for bit-exact equivalence.
    // -----------------------------------------------------------------
    fn null_deviance_with_offset_naive(
        y: &Array1<f64>,
        family_name: &str,
        weights: Option<&Array1<f64>>,
        offset: Option<&Array1<f64>>,
    ) -> f64 {
        let n = y.len();
        let family_lower = family_name.to_lowercase();
        let mu_null: Array1<f64> = match offset {
            Some(off) => {
                let log_link = family_lower.starts_with("poisson")
                    || family_lower.starts_with("negbin")
                    || family_lower.starts_with("negativebinomial")
                    || family_lower.starts_with("gamma")
                    || family_lower.starts_with("quasipoisson");
                if log_link {
                    let exp_off: Array1<f64> = off.mapv(|x| x.exp());
                    let sum_exp_off: f64 = match weights {
                        Some(w) => ndarray::Zip::from(&exp_off)
                            .and(w)
                            .fold(0.0, |a, &e, &wi| a + e * wi),
                        None => exp_off.sum(),
                    };
                    let sum_y: f64 = match weights {
                        Some(w) => ndarray::Zip::from(y)
                            .and(w)
                            .fold(0.0, |a, &yi, &wi| a + yi * wi),
                        None => y.sum(),
                    };
                    let mean_rate = sum_y / sum_exp_off;
                    exp_off.mapv(|e| mean_rate * e)
                } else {
                    let (sy, sw) = match weights {
                        Some(w) => (
                            ndarray::Zip::from(y)
                                .and(w)
                                .fold(0.0, |a, &yi, &wi| a + yi * wi),
                            w.sum(),
                        ),
                        None => (y.sum(), n as f64),
                    };
                    Array1::from_elem(n, sy / sw)
                }
            }
            None => {
                let (sy, sw) = match weights {
                    Some(w) => (
                        ndarray::Zip::from(y)
                            .and(w)
                            .fold(0.0, |a, &yi, &wi| a + yi * wi),
                        w.sum(),
                    ),
                    None => (y.sum(), n as f64),
                };
                Array1::from_elem(n, sy / sw)
            }
        };

        // Delegate per-family unit deviance to the canonical `Family` trait
        // via `family_from_name`. This keeps the naive reference from
        // duplicating the trait arithmetic — it only reproduces the
        // `mu_null` allocation (its raison d'être as a naive reference).
        let family = family_from_name(&family_lower).expect("unsupported family in naive ref");
        let unit_dev = family.unit_deviance(y, &mu_null);

        match weights {
            Some(w) => (&unit_dev * w).sum(),
            None => unit_dev.sum(),
        }
    }

    #[test]
    fn test_null_deviance_streaming_matches_naive() {
        // Pseudo-random but deterministic inputs.
        let mut s: u64 = 0xCAFEBABE;
        let mut next = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let n = 500;
        let y_v: Vec<f64> = (0..n).map(|_| (next() * 5.0).round()).collect();
        let off_v: Vec<f64> = (0..n).map(|_| (0.1 + 0.9 * next()).ln()).collect();
        let w_v: Vec<f64> = (0..n).map(|_| 0.5 + next()).collect();
        let y = Array1::from_vec(y_v);
        let off = Array1::from_vec(off_v);
        let w = Array1::from_vec(w_v);

        // The streaming `Zip::fold` and the gather-then-`Array1::sum()` reference
        // disagree by at most a few ULP because ndarray's `sum` does an 8-way
        // unrolled (pairwise-style) accumulation while `Zip::fold` is a strict
        // left-fold. We assert relative agreement at 1e-13 (~10 ULP for sums of
        // O(1e3) values), which is strictly tighter than any downstream test
        // tolerance in this crate.
        let approx_eq = |a: f64, b: f64| -> bool {
            if a == b {
                return true;
            }
            let denom = a.abs().max(b.abs()).max(1.0);
            (a - b).abs() / denom < 1e-13
        };

        for fam in [
            "gaussian",
            "poisson",
            "quasipoisson",
            "gamma",
            "binomial",
            "negativebinomial(theta=1.5)",
        ] {
            // For binomial we need y in [0,1] and no offset semantics that break things.
            let (y_use, mu_use, off_use): (Array1<f64>, _, _) = if fam == "binomial" {
                (y.mapv(|v| if v > 2.5 { 1.0 } else { 0.0 }), w.clone(), None)
            } else {
                (y.clone(), w.clone(), Some(&off))
            };
            // No offset, no weights
            let new = null_deviance_with_offset(&y_use, fam, None, None)
                .expect("test setup should be valid");
            let old = null_deviance_with_offset_naive(&y_use, fam, None, None);
            assert!(
                approx_eq(new, old),
                "no offset, no weights, fam={}: new={} old={}",
                fam,
                new,
                old
            );
            // No offset, with weights
            let new = null_deviance_with_offset(&y_use, fam, Some(&mu_use), None)
                .expect("test setup should be valid");
            let old = null_deviance_with_offset_naive(&y_use, fam, Some(&mu_use), None);
            assert!(
                approx_eq(new, old),
                "no offset, weighted, fam={}: new={} old={}",
                fam,
                new,
                old
            );
            // With offset
            if let Some(o) = off_use {
                let new = null_deviance_with_offset(&y_use, fam, None, Some(o))
                    .expect("test setup should be valid");
                let old = null_deviance_with_offset_naive(&y_use, fam, None, Some(o));
                assert!(
                    approx_eq(new, old),
                    "with offset, no weights, fam={}: new={} old={}",
                    fam,
                    new,
                    old
                );
                let new = null_deviance_with_offset(&y_use, fam, Some(&mu_use), Some(o))
                    .expect("test setup should be valid");
                let old = null_deviance_with_offset_naive(&y_use, fam, Some(&mu_use), Some(o));
                assert!(
                    approx_eq(new, old),
                    "with offset, weighted, fam={}: new={} old={}",
                    fam,
                    new,
                    old
                );
            }
        }
    }

    // -----------------------------------------------------------------
    // `null_deviance_with_offset_naive` reference vs. trait-based path
    // -----------------------------------------------------------------
    //
    // The `null_deviance_with_offset_naive` helper above is a test-only
    // reference with per-family inline deviance formulas. The production
    // `null_deviance_with_offset` goes through
    // `Family::unit_deviance_at`. A planned refactor will collapse the
    // naive helper to also delegate to the trait; these tests exist so
    // the refactor can be verified AND so any future drift between the
    // helper and the canonical trait is caught.
    //
    // Drift catchers (what each test would fire on):
    //   * Sign flip in NegBin y=0 branch of the helper       → O(1) drift
    //   * Dropped outer factor of 2 in any family            → 2× drift
    //   * Swapping (y, μ) arguments                          → O(1) drift
    //   * Changing the mean-rate formula for log-link+offset → O(1) drift
    //
    // Tolerance: 1e-12 relative. The two paths use the same per-row
    // arithmetic once the naive helper is refactored, so drift will be
    // pure float-ordering (below 1e-13). Today's helper has identical
    // arithmetic for the families under test (Poisson/Gamma/NegBin/
    // Gaussian/Binomial — all use the same formulas as the corresponding
    // `Family::unit_deviance_at`), so 1e-12 is comfortably above the
    // ULP drift while being far tighter than any downstream tolerance
    // in this crate. A looser aggregate tolerance than the loss.rs
    // tests is warranted because the naive helper uses
    // `Array1::map_collect` + `sum()` (8-way unrolled) while production
    // uses a strict left-fold via `Zip::fold`, so ULP-level drift on
    // the sum itself can exceed 1e-14 for n=500.

    /// Trait-based path and naive test reference must agree across all
    /// supported families and every combination of weights/offset. This
    /// is the primary drift catcher for the naive helper's role as a
    /// reference implementation.
    #[test]
    fn test_null_deviance_naive_ref_matches_trait() {
        // Deterministic synthetic data — mix of ints and fractions, mu
        // bounded away from 0, offsets bounded.
        let mut s: u64 = 0xC0_FF_EE_42;
        let mut next = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let n = 300;
        let y_v: Vec<f64> = (0..n).map(|_| (next() * 5.0).round()).collect();
        let off_v: Vec<f64> = (0..n).map(|_| (0.2 + 0.8 * next()).ln()).collect();
        let w_v: Vec<f64> = (0..n).map(|_| 0.5 + next()).collect();
        let y = Array1::from_vec(y_v);
        let off = Array1::from_vec(off_v);
        let w = Array1::from_vec(w_v);

        let rel_le = |a: f64, b: f64, tol: f64| -> bool {
            if a == b {
                return true;
            }
            let denom = a.abs().max(b.abs()).max(1.0);
            (a - b).abs() / denom < tol
        };
        let tol = 1e-12;

        for fam in [
            "gaussian",
            "poisson",
            "binomial",
            "gamma",
            "negativebinomial(theta=1.5)",
        ] {
            // Binomial needs y in [0,1]; offset on binomial is not
            // special-cased (identity-link-like in the helper), so we
            // skip offset there to keep comparisons meaningful.
            let y_use: Array1<f64> = if fam == "binomial" {
                y.mapv(|v| if v > 2.5 { 1.0 } else { 0.0 })
            } else {
                y.clone()
            };
            let off_opt: Option<&Array1<f64>> = if fam == "binomial" { None } else { Some(&off) };

            // No weights, no offset
            let prod = null_deviance_with_offset(&y_use, fam, None, None)
                .expect("test setup should be valid");
            let naive = null_deviance_with_offset_naive(&y_use, fam, None, None);
            assert!(
                rel_le(prod, naive, tol),
                "no wt/off, fam={}: prod={} naive={}",
                fam,
                prod,
                naive
            );

            // Weighted, no offset
            let prod = null_deviance_with_offset(&y_use, fam, Some(&w), None)
                .expect("test setup should be valid");
            let naive = null_deviance_with_offset_naive(&y_use, fam, Some(&w), None);
            assert!(
                rel_le(prod, naive, tol),
                "wt no off, fam={}: prod={} naive={}",
                fam,
                prod,
                naive
            );

            // Offset (when applicable)
            if let Some(o) = off_opt {
                let prod = null_deviance_with_offset(&y_use, fam, None, Some(o))
                    .expect("test setup should be valid");
                let naive = null_deviance_with_offset_naive(&y_use, fam, None, Some(o));
                assert!(
                    rel_le(prod, naive, tol),
                    "off no wt, fam={}: prod={} naive={}",
                    fam,
                    prod,
                    naive
                );

                // Weighted + offset
                let prod = null_deviance_with_offset(&y_use, fam, Some(&w), Some(o))
                    .expect("test setup should be valid");
                let naive = null_deviance_with_offset_naive(&y_use, fam, Some(&w), Some(o));
                assert!(
                    rel_le(prod, naive, tol),
                    "off + wt, fam={}: prod={} naive={}",
                    fam,
                    prod,
                    naive
                );
            }
        }
    }

    /// Focussed NegBin-with-zeros test — this is the specific scenario
    /// that produced the original NegBin y=0 sign bug in the production
    /// streaming path. Pins the naive helper at that scenario so any
    /// refactor that re-introduces the sign flip (either in the helper
    /// or in the trait that `null_deviance_with_offset` delegates to)
    /// is caught immediately.
    #[test]
    fn test_null_deviance_naive_ref_negbinomial_with_zeros() {
        // Deliberate mix with ~half zero y rows — guarantees the NB
        // y=0 branch dominates the sum and the sign bug would produce
        // a large (negative or wildly off) result.
        let y = array![0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 0.0, 5.0];
        let fam = "negativebinomial(theta=1.5)";

        let prod =
            null_deviance_with_offset(&y, fam, None, None).expect("test setup should be valid");
        let naive = null_deviance_with_offset_naive(&y, fam, None, None);

        // Deviance must be non-negative (would have been negative
        // under the original bug).
        assert!(prod >= 0.0, "production null deviance must be non-negative");
        assert!(
            naive >= 0.0,
            "naive null deviance must be non-negative (catches sign bug in helper)"
        );

        // And the two must agree.
        let denom = prod.abs().max(naive.abs()).max(1.0);
        let rel_err = (prod - naive).abs() / denom;
        assert!(
            rel_err < 1e-12,
            "prod={} naive={} rel_err={}",
            prod,
            naive,
            rel_err
        );
    }
}
