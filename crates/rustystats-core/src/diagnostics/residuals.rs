// =============================================================================
// Residuals for GLM Diagnostics
// =============================================================================
//
// Residuals measure how far predictions are from observed values.
// Different types of residuals are useful for different purposes.
//
// TYPES OF RESIDUALS:
// -------------------
// 1. Response (raw): y - μ
//    Simple difference. Not standardized, so hard to compare across obs.
//
// 2. Pearson: (y - μ) / √V(μ)
//    Standardized by the standard deviation. Should have roughly constant
//    variance if model is correct.
//
// 3. Deviance: sign(y - μ) × √d_i
//    Based on deviance contributions. Often more normally distributed
//    than Pearson residuals for non-Gaussian families.
//
// 4. Working: (y - μ) × g'(μ)
//    Used internally in IRLS. On the scale of the linear predictor.
//
// INTERPRETATION:
// ---------------
// - Large residuals suggest outliers or poor fit
// - Patterns in residuals vs fitted values suggest model misspecification
// - Non-constant variance suggests wrong family or link
//
// =============================================================================

use crate::constants::ZERO_TOL;
use crate::families::Family;
use crate::links::Link;
use ndarray::Array1;

/// Compute response (raw) residuals: y - μ
///
/// These are the simplest residuals - just the difference between
/// observed and predicted values on the response scale.
///
/// # Arguments
/// * `y` - Observed response values
/// * `mu` - Fitted mean values
///
/// # Returns
/// Array of residuals (same length as y)
///
/// # Note
/// Not standardized, so magnitude depends on the scale of y.
/// Use Pearson residuals for standardized residuals.
pub fn resid_response(y: &Array1<f64>, mu: &Array1<f64>) -> Array1<f64> {
    y - mu
}

/// Compute Pearson residuals: (y - μ) / √V(μ)
///
/// Standardized residuals that account for the variance function.
/// For a well-specified model, these should have approximately:
/// - Mean 0
/// - Variance φ (the dispersion parameter)
///
/// # Arguments
/// * `y` - Observed response values
/// * `mu` - Fitted mean values
/// * `family` - Distribution family (provides variance function)
///
/// # Returns
/// Array of Pearson residuals
///
/// # Interpretation
/// Large |residuals| (e.g., > 2-3) may indicate outliers.
/// Pattern in residuals vs fitted may indicate model problems.
pub fn resid_pearson(y: &Array1<f64>, mu: &Array1<f64>, family: &dyn Family) -> Array1<f64> {
    let variance = family.variance(mu);

    ndarray::Zip::from(y)
        .and(mu)
        .and(&*variance)
        .map_collect(|&yi, &mui, &vi| {
            let std_dev = vi.sqrt().max(ZERO_TOL);
            (yi - mui) / std_dev
        })
}

/// Compute deviance residuals: sign(y - μ) × √d_i
///
/// Based on the unit deviance contributions. Often preferred because:
/// - More normally distributed than Pearson for non-Gaussian families
/// - Sum of squares equals the model deviance
///
/// # Arguments
/// * `y` - Observed response values
/// * `mu` - Fitted mean values
/// * `family` - Distribution family (provides unit deviance)
///
/// # Returns
/// Array of deviance residuals
///
/// # Property
/// sum(resid_deviance²) = model deviance
pub fn resid_deviance(y: &Array1<f64>, mu: &Array1<f64>, family: &dyn Family) -> Array1<f64> {
    let unit_dev = family.unit_deviance(y, mu);

    ndarray::Zip::from(y)
        .and(mu)
        .and(&unit_dev)
        .map_collect(|&yi, &mui, &di| {
            let sign = if yi > mui { 1.0 } else { -1.0 };
            sign * di.sqrt()
        })
}

/// Compute working residuals: (y - μ) × g'(μ)
///
/// These are used internally by IRLS. They're on the scale of the
/// linear predictor and are useful for understanding the fitting process.
///
/// # Arguments
/// * `y` - Observed response values
/// * `mu` - Fitted mean values
/// * `link` - Link function (provides derivative)
///
/// # Returns
/// Array of working residuals
pub fn resid_working(y: &Array1<f64>, mu: &Array1<f64>, link: &dyn Link) -> Array1<f64> {
    let link_deriv = link.derivative(mu);

    ndarray::Zip::from(y)
        .and(mu)
        .and(&link_deriv)
        .map_collect(|&yi, &mui, &di| (yi - mui) * di)
}

// =============================================================================
// Residual Summary Statistics
// =============================================================================

/// Summary statistics for a vector of residuals.
///
/// Includes central moments (mean, std, skewness, kurtosis), extremes
/// (min/max), and a fixed grid of percentiles. Used by diagnostic output
/// to characterise the residual distribution at a glance.
///
/// `kurtosis` is the excess kurtosis (population kurtosis − 3).
/// `skewness` is the population (biased) skewness.
/// Percentiles use the nearest-rank rule on `total_cmp`-sorted data so
/// NaN handling is deterministic.
#[derive(Debug, Clone, PartialEq)]
pub struct ResidualSummary {
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
    pub skewness: f64,
    pub kurtosis: f64,
    pub p1: f64,
    pub p5: f64,
    pub p10: f64,
    pub p25: f64,
    pub p50: f64,
    pub p75: f64,
    pub p90: f64,
    pub p95: f64,
    pub p99: f64,
}

/// Compute summary statistics for a residual vector.
///
/// Returns `None` if the input is empty (callers should reject empty
/// arrays before computing diagnostics).
///
/// Skewness and kurtosis are zero when std is zero (degenerate case).
/// Percentiles use the nearest-rank rule and `total_cmp` for NaN-safe
/// ordering.
///
/// # Arguments
/// * `residuals` - Array of residual values.
///
/// # Returns
/// `Some(ResidualSummary)` with all fields populated, or `None` for
/// empty input.
pub fn compute_residual_summary(residuals: &Array1<f64>) -> Option<ResidualSummary> {
    let n = residuals.len();
    if n == 0 {
        return None;
    }
    let n_f = n as f64;

    let mean = residuals.iter().sum::<f64>() / n_f;
    let variance = residuals.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n_f;
    let std = variance.sqrt();
    let min = residuals.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = residuals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // Skewness and excess kurtosis (population formulas, biased).
    // Match the original Python-side formulas exactly.
    let (skewness, kurtosis) = if std > 0.0 {
        let skew = residuals
            .iter()
            .map(|x| ((x - mean) / std).powi(3))
            .sum::<f64>()
            / n_f;
        let kurt = residuals
            .iter()
            .map(|x| ((x - mean) / std).powi(4))
            .sum::<f64>()
            / n_f
            - 3.0;
        (skew, kurt)
    } else {
        (0.0, 0.0)
    };

    // Percentiles via nearest-rank.
    //
    // Strategy: clone the residuals once (we cannot mutate the input), then
    // run `select_nth_unstable_by` for each requested percentile in
    // ascending order. Each call partitions the array around the target
    // index in O(n) average time, in-place, with no extra allocation. This
    // replaces a full `sort_by` (Timsort: O(n log n) AND an O(n) scratch
    // buffer ~= 8 MB at n=1M).
    //
    // We process the percentiles in ascending index order so that earlier
    // selects' partitions stay intact for later selects, which we run on
    // the right-hand sub-slice. The result is bit-identical to the previous
    // sort-and-index path because nearest-rank percentiles only depend on
    // the value at a specific sorted position, not on the relative ordering
    // of the rest of the array.
    let mut buf: Vec<f64> = residuals.iter().cloned().collect();
    let last = buf.len() - 1;
    let percentile_idx = |p: f64| -> usize {
        let idx = (p / 100.0 * last as f64).round() as usize;
        idx.min(last)
    };

    // Pre-compute the (sorted, deduplicated) target indices.
    let percent_specs = [1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0, 99.0];
    let mut targets: Vec<usize> = percent_specs.iter().map(|&p| percentile_idx(p)).collect();
    targets.sort_unstable();
    targets.dedup();

    // Walk the targets left-to-right, each time selecting on the slice of
    // values that are still unordered relative to the previous select.
    let mut start = 0usize;
    for &t in &targets {
        // After this call, buf[t] holds the value that would be at sorted
        // position `t`, and elements in buf[start..t] are all <= buf[t].
        let local = t - start;
        // `select_nth_unstable_by` returns (lesser, pivot, greater) but we
        // only need its side-effect: after the call, buf[t] holds the sorted
        // pivot value. Discard the tuple directly.
        let _ = buf[start..].select_nth_unstable_by(local, |a, b| a.total_cmp(b));
        start = t + 1;
    }

    let percentile = |p: f64| -> f64 { buf[percentile_idx(p)] };

    Some(ResidualSummary {
        mean,
        std,
        min,
        max,
        skewness,
        kurtosis,
        p1: percentile(1.0),
        p5: percentile(5.0),
        p10: percentile(10.0),
        p25: percentile(25.0),
        p50: percentile(50.0),
        p75: percentile(75.0),
        p90: percentile(90.0),
        p95: percentile(95.0),
        p99: percentile(99.0),
    })
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::families::{GaussianFamily, PoissonFamily};
    use crate::links::{IdentityLink, LogLink};
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_response_residuals() {
        let y = array![1.0, 2.0, 3.0];
        let mu = array![1.1, 2.0, 2.8];

        let resid = resid_response(&y, &mu);

        let expected = array![-0.1, 0.0, 0.2];
        assert_abs_diff_eq!(resid, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_pearson_residuals_gaussian() {
        let y = array![1.0, 2.0, 3.0];
        let mu = array![1.5, 2.0, 2.5];
        let family = GaussianFamily;

        let resid = resid_pearson(&y, &mu, &family);

        // For Gaussian, V(μ) = 1, so Pearson = response
        let expected = array![-0.5, 0.0, 0.5];
        assert_abs_diff_eq!(resid, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_pearson_residuals_poisson() {
        let y = array![4.0];
        let mu = array![2.0];
        let family = PoissonFamily;

        let resid = resid_pearson(&y, &mu, &family);

        // (4 - 2) / √2 = 2 / 1.414... ≈ 1.414
        let expected = (4.0 - 2.0) / 2.0_f64.sqrt();
        assert_abs_diff_eq!(resid[0], expected, epsilon = 1e-10);
    }

    #[test]
    fn test_deviance_residuals_gaussian() {
        let y = array![1.0, 2.0, 3.0];
        let mu = array![1.5, 2.0, 2.5];
        let family = GaussianFamily;

        let resid = resid_deviance(&y, &mu, &family);

        // For Gaussian, unit deviance = (y-μ)²
        // Deviance residual = sign(y-μ) × |y-μ| = y-μ
        let expected = array![-0.5, 0.0, 0.5];
        assert_abs_diff_eq!(resid, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_deviance_residuals_sum_equals_deviance() {
        let y = array![1.0, 2.0, 5.0, 3.0];
        let mu = array![1.2, 1.8, 4.5, 3.2];
        let family = PoissonFamily;

        let resid = resid_deviance(&y, &mu, &family);
        let sum_sq: f64 = resid.iter().map(|&r| r * r).sum();

        let deviance = family.deviance(&y, &mu, None);

        assert_abs_diff_eq!(sum_sq, deviance, epsilon = 1e-10);
    }

    #[test]
    fn test_working_residuals_identity() {
        let y = array![1.0, 2.0, 3.0];
        let mu = array![1.5, 2.0, 2.5];
        let link = IdentityLink;

        let resid = resid_working(&y, &mu, &link);

        // For identity link, g'(μ) = 1, so working = response
        let expected = array![-0.5, 0.0, 0.5];
        assert_abs_diff_eq!(resid, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_working_residuals_log() {
        let y = array![3.0];
        let mu = array![2.0];
        let link = LogLink;

        let resid = resid_working(&y, &mu, &link);

        // For log link, g'(μ) = 1/μ
        // Working residual = (y - μ) / μ = (3 - 2) / 2 = 0.5
        let expected = (3.0 - 2.0) / 2.0;
        assert_abs_diff_eq!(resid[0], expected, epsilon = 1e-10);
    }

    // -----------------------------------------------------------------
    // Residual summary tests
    // -----------------------------------------------------------------

    #[test]
    fn test_residual_summary_empty_returns_none() {
        let r: Array1<f64> = Array1::zeros(0);
        assert!(compute_residual_summary(&r).is_none());
    }

    #[test]
    fn test_residual_summary_constant_zero_std() {
        let r = array![2.0, 2.0, 2.0, 2.0];
        let s = compute_residual_summary(&r).unwrap();
        assert_abs_diff_eq!(s.mean, 2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(s.std, 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(s.skewness, 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(s.kurtosis, 0.0, epsilon = 1e-12);
        assert_eq!(s.min, 2.0);
        assert_eq!(s.max, 2.0);
        assert_eq!(s.p50, 2.0);
    }

    #[test]
    fn test_residual_summary_basic_moments() {
        let r = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let s = compute_residual_summary(&r).unwrap();
        // Mean = 3, var = (4+1+0+1+4)/5 = 2 (population), std = sqrt(2)
        assert_abs_diff_eq!(s.mean, 3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(s.std, 2.0_f64.sqrt(), epsilon = 1e-12);
        assert_eq!(s.min, 1.0);
        assert_eq!(s.max, 5.0);
        // Symmetric around mean → skew ≈ 0
        assert_abs_diff_eq!(s.skewness, 0.0, epsilon = 1e-12);
        // Median of 1..5 is 3
        assert_eq!(s.p50, 3.0);
    }

    #[test]
    fn test_residual_summary_percentile_nearest_rank() {
        // With n=11, last=10, idx = round(p/100 * 10).
        // Sorted ascending, value at idx i is i (0-indexed).
        let r: Array1<f64> = (0..11).map(|i| i as f64).collect();
        let s = compute_residual_summary(&r).unwrap();
        assert_eq!(s.p1, 0.0); // round(0.1) = 0
        assert_eq!(s.p10, 1.0); // round(1.0) = 1
        assert_eq!(s.p50, 5.0); // round(5.0) = 5
        assert_eq!(s.p99, 10.0); // round(9.9) = 10
    }

    #[test]
    fn test_residual_summary_skewness_known() {
        // Right-skewed sample: large positive tail
        let r = array![0.0, 0.0, 0.0, 0.0, 10.0];
        let s = compute_residual_summary(&r).unwrap();
        // Mean = 2, var = (4+4+4+4+64)/5 = 16, std = 4
        // skewness = mean of ((x-2)/4)^3 = ((-0.5)^3*4 + (2.0)^3) / 5
        //          = ((-0.125)*4 + 8) / 5 = (-0.5 + 8)/5 = 1.5
        assert_abs_diff_eq!(s.mean, 2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(s.std, 4.0, epsilon = 1e-12);
        assert_abs_diff_eq!(s.skewness, 1.5, epsilon = 1e-10);
    }

    /// Bit-exact equivalence vs the previous full-sort percentile path.
    ///
    /// Nearest-rank percentiles only depend on the value at a specific sorted
    /// position — not on the relative ordering of the rest of the array — so
    /// repeated `select_nth_unstable_by` calls (in ascending target-index
    /// order) must return the *same* float at each percentile index as a
    /// full sort would. This test pins that invariant on a 1k-row LCG-random
    /// vector with both unique and tied values.
    #[test]
    fn test_residual_summary_select_nth_matches_full_sort() {
        // Naive reference: full sort + index lookup.
        fn naive(residuals: &Array1<f64>) -> [f64; 9] {
            let mut sorted: Vec<f64> = residuals.iter().cloned().collect();
            sorted.sort_by(|a, b| a.total_cmp(b));
            let last = sorted.len() - 1;
            let p = |q: f64| {
                let idx = (q / 100.0 * last as f64).round() as usize;
                sorted[idx.min(last)]
            };
            [
                p(1.0),
                p(5.0),
                p(10.0),
                p(25.0),
                p(50.0),
                p(75.0),
                p(90.0),
                p(95.0),
                p(99.0),
            ]
        }
        // Mixed unique & tied values to exercise total_cmp ordering.
        let mut s: u64 = 0xC0FFEE;
        let n = 1000;
        let v: Vec<f64> = (0..n)
            .map(|i| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u = ((s >> 11) as f64) / ((1u64 << 53) as f64);
                if i % 7 == 0 {
                    // Inject ties at every 7th element to stress tie-breaking.
                    0.5
                } else {
                    -1.0 + 2.0 * u
                }
            })
            .collect();
        let r = Array1::from_vec(v);
        let summary = compute_residual_summary(&r).unwrap();
        let ref_pcts = naive(&r);
        let new_pcts = [
            summary.p1,
            summary.p5,
            summary.p10,
            summary.p25,
            summary.p50,
            summary.p75,
            summary.p90,
            summary.p95,
            summary.p99,
        ];
        // total_cmp is total ordering on f64, so equal sorted positions
        // contain identical values — bit-exact equality is required.
        assert_eq!(new_pcts, ref_pcts, "percentiles must match full-sort path");
    }
}
