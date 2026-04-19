// =============================================================================
// Fit Statistics
// =============================================================================
//
// Aggregated GLM fit summary returned by the public `compute_fit_statistics`
// helper. Combines deviance-based, likelihood-based, and dispersion-based
// goodness-of-fit measures into a single struct so callers (Python wrappers
// in particular) don't have to re-derive any of them.
// =============================================================================

use crate::diagnostics::dispersion::pearson_chi2;
use crate::diagnostics::model_fit::{aic, bic};
use crate::families::Family;
use ndarray::Array1;

/// Combined GLM fit statistics returned by `compute_fit_statistics`.
///
/// `dispersion` is the Pearson chi-square divided by residual degrees of
/// freedom — the primary dispersion estimator returned to Python.
#[derive(Debug, Clone, PartialEq)]
pub struct FitStatistics {
    pub deviance: f64,
    pub null_deviance: f64,
    pub deviance_explained: f64,
    pub log_likelihood: f64,
    pub aic: f64,
    pub bic: f64,
    pub pearson_chi2: f64,
    pub dispersion: f64,
}

/// Compute the standard set of GLM fit statistics from a fitted model.
///
/// Pulls together what the Python summary needs in a single call:
/// - Pearson chi-square via `pearson_chi2`
/// - Pearson dispersion = chi2 / df_resid
/// - Gaussian-style scale = SSR / df_resid (used for log-likelihood when
///   the family doesn't have a fixed dispersion; this matches the previous
///   wrapper behaviour exactly)
/// - Log-likelihood via the family's trait method
/// - AIC / BIC via the standard `-2 ll + 2k` / `-2 ll + k log(n)` formulas
/// - Deviance explained = 1 − D / D_null when D_null > 0, else 0
///
/// `df_resid = max(0, n_obs − n_params)`. When df_resid == 0 the scale and
/// dispersion fall back to 1.0 so the formulas don't divide by zero.
///
/// # Arguments
/// * `y` - Observed response values.
/// * `mu` - Fitted mean values.
/// * `family` - Distribution family (provides variance, log-likelihood).
/// * `deviance` - Pre-computed model deviance.
/// * `null_dev` - Pre-computed null deviance (intercept-only model).
/// * `n_params` - Number of estimated parameters (including intercept).
pub fn compute_fit_statistics(
    y: &Array1<f64>,
    mu: &Array1<f64>,
    family: &dyn Family,
    deviance: f64,
    null_dev: f64,
    n_params: usize,
) -> FitStatistics {
    let n_obs = y.len();
    let df_resid = n_obs.saturating_sub(n_params);

    // Pearson chi-square based on the family's variance function.
    let pchi2 = pearson_chi2(y, mu, family, None);

    // Gaussian-style scale (SSR / df_resid). This mirrors the prior wrapper
    // logic exactly: it is fed into log_likelihood for families that don't
    // have a fixed dispersion. Falls back to 1.0 if df_resid == 0.
    let scale = if df_resid > 0 {
        let ssr: f64 = ndarray::Zip::from(y)
            .and(mu)
            .fold(0.0, |acc, &yi, &mui| acc + (yi - mui).powi(2));
        ssr / df_resid as f64
    } else {
        1.0
    };

    let llf = family.log_likelihood(y, mu, scale, None);

    let aic_val = aic(llf, n_params);
    let bic_val = bic(llf, n_params, n_obs);

    let deviance_explained = if null_dev > 0.0 {
        1.0 - deviance / null_dev
    } else {
        0.0
    };

    let dispersion = if df_resid > 0 {
        pchi2 / df_resid as f64
    } else {
        1.0
    };

    FitStatistics {
        deviance,
        null_deviance: null_dev,
        deviance_explained,
        log_likelihood: llf,
        aic: aic_val,
        bic: bic_val,
        pearson_chi2: pchi2,
        dispersion,
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diagnostics::dispersion::pearson_chi2;
    use crate::diagnostics::model_fit::{aic, bic};
    use crate::families::{GaussianFamily, PoissonFamily};
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_fit_statistics_gaussian_basic() {
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let mu = array![1.1, 2.1, 3.0, 3.9, 4.9];
        let family = GaussianFamily;

        let deviance = family.deviance(&y, &mu, None);
        let null_dev = 10.0; // arbitrary positive
        let n_params = 2;

        let stats = compute_fit_statistics(&y, &mu, &family, deviance, null_dev, n_params);

        // Sanity-check key fields against direct calls.
        let pchi2_direct = pearson_chi2(&y, &mu, &family, None);
        assert_abs_diff_eq!(stats.pearson_chi2, pchi2_direct, epsilon = 1e-12);

        let df_resid = y.len() - n_params;
        let ssr: f64 = ndarray::Zip::from(&y)
            .and(&mu)
            .fold(0.0, |acc, &yi, &mui| acc + (yi - mui).powi(2));
        let scale = ssr / df_resid as f64;
        let llf_direct = family.log_likelihood(&y, &mu, scale, None);
        assert_abs_diff_eq!(stats.log_likelihood, llf_direct, epsilon = 1e-12);

        assert_abs_diff_eq!(stats.aic, aic(llf_direct, n_params), epsilon = 1e-12);
        assert_abs_diff_eq!(
            stats.bic,
            bic(llf_direct, n_params, y.len()),
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            stats.dispersion,
            pchi2_direct / df_resid as f64,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            stats.deviance_explained,
            1.0 - deviance / null_dev,
            epsilon = 1e-12
        );
        assert_eq!(stats.deviance, deviance);
        assert_eq!(stats.null_deviance, null_dev);
    }

    #[test]
    fn test_fit_statistics_zero_null_deviance() {
        let y = array![1.0, 2.0, 3.0];
        let mu = array![1.0, 2.0, 3.0];
        let family = GaussianFamily;

        let stats = compute_fit_statistics(&y, &mu, &family, 0.0, 0.0, 1);
        // null_dev == 0 → deviance_explained == 0 (not divide-by-zero).
        assert_eq!(stats.deviance_explained, 0.0);
    }

    #[test]
    fn test_fit_statistics_df_resid_zero_falls_back() {
        // n_obs == n_params → df_resid == 0 → scale and dispersion default to 1.
        let y = array![1.0, 2.0];
        let mu = array![1.0, 2.0];
        let family = GaussianFamily;

        let stats = compute_fit_statistics(&y, &mu, &family, 0.0, 5.0, 2);
        assert_eq!(stats.dispersion, 1.0);
        // log_likelihood was evaluated with scale = 1.0; sanity check it's finite.
        assert!(stats.log_likelihood.is_finite());
    }

    #[test]
    fn test_fit_statistics_poisson() {
        let y = array![1.0, 2.0, 3.0, 4.0];
        let mu = array![1.2, 1.8, 3.1, 3.9];
        let family = PoissonFamily;

        let deviance = family.deviance(&y, &mu, None);
        let stats = compute_fit_statistics(&y, &mu, &family, deviance, 5.0, 2);

        assert_eq!(stats.deviance, deviance);
        assert!(stats.log_likelihood.is_finite());
        assert!(stats.dispersion > 0.0);
        assert_abs_diff_eq!(
            stats.pearson_chi2,
            pearson_chi2(&y, &mu, &family, None),
            epsilon = 1e-12
        );
    }
}
