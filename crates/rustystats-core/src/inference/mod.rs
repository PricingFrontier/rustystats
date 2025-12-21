// =============================================================================
// Statistical Inference
// =============================================================================
//
// This module provides tools for statistical inference on GLM results:
//   - P-values: Test if coefficients are significantly different from zero
//   - Confidence intervals: Range estimates for true parameter values
//   - Hypothesis testing utilities
//
// FOR ACTUARIES:
// --------------
// Statistical inference tells us how confident we can be in our estimates.
//
// Example: You fit a model and get β_age = 0.05 for age effect.
// But how reliable is this estimate?
//   - p-value < 0.05 → The effect is statistically significant
//   - 95% CI = [0.02, 0.08] → We're 95% confident the true effect is in this range
//
// IMPORTANT CAVEATS:
// - Statistical significance ≠ practical significance
// - With large samples, tiny effects become "significant"
// - Always consider the magnitude of effects, not just p-values
//
// =============================================================================

use statrs::distribution::{ContinuousCDF, Normal, StudentsT};

// =============================================================================
// P-Value Calculation
// =============================================================================

/// Calculate two-tailed p-value from a z-statistic.
///
/// Uses the standard normal distribution.
/// Appropriate for large samples or when variance is known.
///
/// # Arguments
/// * `z` - The z-statistic (coefficient / standard_error)
///
/// # Returns
/// P-value: probability of seeing a test statistic this extreme or more,
/// assuming the null hypothesis (β = 0) is true.
///
/// # Interpretation
/// - p < 0.05: Traditionally "significant" at 5% level
/// - p < 0.01: "Highly significant" at 1% level
/// - p < 0.001: "Very highly significant"
///
/// But remember: p-values are just one piece of evidence!
pub fn pvalue_z(z: f64) -> f64 {
    if !z.is_finite() {
        return f64::NAN;
    }
    
    let normal = Normal::new(0.0, 1.0).unwrap();
    
    // Two-tailed test: probability in both tails
    // P(|Z| > |z|) = 2 * P(Z > |z|) = 2 * (1 - Φ(|z|))
    2.0 * (1.0 - normal.cdf(z.abs()))
}

/// Calculate two-tailed p-value from a t-statistic.
///
/// Uses Student's t-distribution with specified degrees of freedom.
/// More appropriate for small samples when variance is estimated.
///
/// # Arguments
/// * `t` - The t-statistic (coefficient / standard_error)
/// * `df` - Degrees of freedom (typically n - p for GLMs)
///
/// # Returns
/// P-value from the t-distribution
pub fn pvalue_t(t: f64, df: f64) -> f64 {
    if !t.is_finite() || df <= 0.0 {
        return f64::NAN;
    }
    
    // For very large df, use normal approximation for efficiency
    if df > 1000.0 {
        return pvalue_z(t);
    }
    
    let t_dist = match StudentsT::new(0.0, 1.0, df) {
        Ok(d) => d,
        Err(_) => return f64::NAN,
    };
    
    // Two-tailed test
    2.0 * (1.0 - t_dist.cdf(t.abs()))
}

// =============================================================================
// Confidence Intervals
// =============================================================================

/// Calculate confidence interval using z-distribution.
///
/// # Arguments
/// * `estimate` - Point estimate (coefficient value)
/// * `std_error` - Standard error of the estimate
/// * `confidence` - Confidence level (e.g., 0.95 for 95% CI)
///
/// # Returns
/// (lower_bound, upper_bound)
///
/// # Interpretation
/// A 95% CI means: If we repeated this analysis many times,
/// 95% of the intervals would contain the true parameter value.
///
/// For a log link: exp(CI) gives you the relativity confidence interval.
pub fn confidence_interval_z(estimate: f64, std_error: f64, confidence: f64) -> (f64, f64) {
    if !estimate.is_finite() || !std_error.is_finite() || std_error <= 0.0 {
        return (f64::NAN, f64::NAN);
    }
    
    let normal = Normal::new(0.0, 1.0).unwrap();
    
    // For 95% CI, alpha = 0.05, so we need z_{0.975}
    let alpha = 1.0 - confidence;
    let z_critical = normal.inverse_cdf(1.0 - alpha / 2.0);
    
    let margin = z_critical * std_error;
    (estimate - margin, estimate + margin)
}

/// Calculate confidence interval using t-distribution.
///
/// # Arguments
/// * `estimate` - Point estimate (coefficient value)
/// * `std_error` - Standard error of the estimate
/// * `df` - Degrees of freedom
/// * `confidence` - Confidence level (e.g., 0.95 for 95% CI)
///
/// # Returns
/// (lower_bound, upper_bound)
pub fn confidence_interval_t(
    estimate: f64,
    std_error: f64,
    df: f64,
    confidence: f64,
) -> (f64, f64) {
    if !estimate.is_finite() || !std_error.is_finite() || std_error <= 0.0 || df <= 0.0 {
        return (f64::NAN, f64::NAN);
    }
    
    // For very large df, use z approximation
    if df > 1000.0 {
        return confidence_interval_z(estimate, std_error, confidence);
    }
    
    let t_dist = match StudentsT::new(0.0, 1.0, df) {
        Ok(d) => d,
        Err(_) => return (f64::NAN, f64::NAN),
    };
    
    let alpha = 1.0 - confidence;
    let t_critical = t_dist.inverse_cdf(1.0 - alpha / 2.0);
    
    let margin = t_critical * std_error;
    (estimate - margin, estimate + margin)
}

// =============================================================================
// Significance Stars (for summary tables)
// =============================================================================

/// Get significance stars for a p-value.
///
/// Returns a string of stars indicating significance level:
/// - "***" : p < 0.001
/// - "**"  : p < 0.01
/// - "*"   : p < 0.05
/// - "."   : p < 0.1
/// - ""    : p >= 0.1
pub fn significance_stars(pvalue: f64) -> &'static str {
    if pvalue < 0.001 {
        "***"
    } else if pvalue < 0.01 {
        "**"
    } else if pvalue < 0.05 {
        "*"
    } else if pvalue < 0.1 {
        "."
    } else {
        ""
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_pvalue_z_zero() {
        // z = 0 should give p = 1 (no evidence against null)
        let p = pvalue_z(0.0);
        assert_abs_diff_eq!(p, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_pvalue_z_large() {
        // Large z should give small p
        let p = pvalue_z(3.0);
        assert!(p < 0.01);
        
        let p = pvalue_z(5.0);
        assert!(p < 0.0001);
    }

    #[test]
    fn test_pvalue_z_symmetric() {
        // P-value should be same for positive and negative z
        let p_pos = pvalue_z(2.0);
        let p_neg = pvalue_z(-2.0);
        assert_abs_diff_eq!(p_pos, p_neg, epsilon = 1e-10);
    }

    #[test]
    fn test_pvalue_z_known_value() {
        // z = 1.96 should give p ≈ 0.05 (two-tailed)
        let p = pvalue_z(1.96);
        assert_abs_diff_eq!(p, 0.05, epsilon = 0.001);
    }

    #[test]
    fn test_pvalue_t_large_df() {
        // With large df, t-distribution ≈ normal
        let p_t = pvalue_t(2.0, 1000.0);
        let p_z = pvalue_z(2.0);
        assert_abs_diff_eq!(p_t, p_z, epsilon = 0.001);
    }

    #[test]
    fn test_confidence_interval_95() {
        // 95% CI with z-distribution
        let (lower, upper) = confidence_interval_z(1.0, 0.5, 0.95);
        
        // Should be approximately 1.0 ± 1.96 * 0.5
        assert_abs_diff_eq!(lower, 1.0 - 1.96 * 0.5, epsilon = 0.01);
        assert_abs_diff_eq!(upper, 1.0 + 1.96 * 0.5, epsilon = 0.01);
    }

    #[test]
    fn test_confidence_interval_symmetric() {
        let (lower, upper) = confidence_interval_z(0.0, 1.0, 0.95);
        
        // CI around 0 should be symmetric
        assert_abs_diff_eq!(-lower, upper, epsilon = 1e-10);
    }

    #[test]
    fn test_significance_stars() {
        assert_eq!(significance_stars(0.0001), "***");
        assert_eq!(significance_stars(0.005), "**");
        assert_eq!(significance_stars(0.03), "*");
        assert_eq!(significance_stars(0.08), ".");
        assert_eq!(significance_stars(0.5), "");
    }
}
