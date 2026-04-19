//! Confidence interval helpers for diagnostics.

use crate::constants::{DEFAULT_CI_ALPHA, DEFAULT_CI_Z};

/// Wilson score interval for the rate λ = k / predicted, where k is an
/// observed Poisson count and predicted is the expected count.
///
/// More accurate than the Wald approximation for small k or rates near 0.
/// Formula: center = (k + z²/2) / predicted; half_width = z·√(k + z²/4) / predicted.
///
/// Returns (lower, upper) clamped so lower >= 0.
/// If predicted <= 0, returns (0.0, NaN).
pub fn wilson_poisson_rate_ci(actual_sum: f64, predicted_sum: f64, alpha: f64) -> (f64, f64) {
    if predicted_sum <= 0.0 {
        return (0.0, f64::NAN);
    }
    debug_assert!(
        (alpha - DEFAULT_CI_ALPHA).abs() < 1e-12,
        "Currently only DEFAULT_CI_ALPHA (0.05) is supported"
    );
    let z = DEFAULT_CI_Z;
    let k = actual_sum.max(0.0);
    let z2 = z * z;
    let center = (k + z2 / 2.0) / predicted_sum;
    let half_width = z * (k + z2 / 4.0).sqrt() / predicted_sum;
    ((center - half_width).max(0.0), center + half_width)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wilson_moderate() {
        let (lo, hi) = wilson_poisson_rate_ci(10.0, 10.0, 0.05);
        assert!((lo - 0.5432).abs() < 0.01);
        assert!((hi - 1.8409).abs() < 0.01);
    }

    #[test]
    fn test_wilson_zero_count() {
        let (lo, hi) = wilson_poisson_rate_ci(0.0, 10.0, 0.05);
        assert_eq!(lo, 0.0);
        assert!((hi - 0.3841).abs() < 0.01);
    }

    #[test]
    fn test_wilson_zero_predicted() {
        let (lo, hi) = wilson_poisson_rate_ci(5.0, 0.0, 0.05);
        assert_eq!(lo, 0.0);
        assert!(hi.is_nan());
    }
}
