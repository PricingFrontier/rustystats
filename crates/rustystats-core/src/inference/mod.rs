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
// Robust Covariance Estimation (Sandwich Estimators)
// =============================================================================
//
// The sandwich estimator provides heteroscedasticity-consistent (HC) standard
// errors. Unlike model-based standard errors that assume the variance function
// is correctly specified, robust standard errors are valid even when the
// variance is misspecified.
//
// The sandwich formula is:
//   Var_robust(β̂) = (X'WX)⁻¹ B (X'WX)⁻¹
//
// Where B (the "meat") is computed from weighted squared residuals.
// The "bread" is (X'WX)⁻¹ which we already have.
//
// HC VARIANTS (following White, MacKinnon & White):
// - HC0: No correction (may be biased in small samples)
// - HC1: Degrees of freedom correction: n/(n-p)
// - HC2: Leverage correction: divide by (1 - h_ii)
// - HC3: Stronger leverage correction: divide by (1 - h_ii)²
//
// FOR ACTUARIES:
// Use robust standard errors when you suspect:
// - Misspecified variance function
// - Heteroscedasticity not captured by the GLM family
// - Clustering effects (although cluster-robust is even better for that)
//
// =============================================================================

use ndarray::{Array1, Array2};
use rayon::prelude::*;

/// Type of heteroscedasticity-consistent (HC) standard errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HCType {
    /// HC0: No small-sample correction. B = X'ΩX where Ω = diag(ε²)
    HC0,
    /// HC1: Degrees of freedom correction. Multiplies by n/(n-p)
    HC1,
    /// HC2: Leverage-adjusted. Ω = diag(ε² / (1 - h_ii))
    HC2,
    /// HC3: Jackknife-like. Ω = diag(ε² / (1 - h_ii)²)
    HC3,
}

impl HCType {
    /// Parse from string (case-insensitive)
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "hc0" => Some(HCType::HC0),
            "hc1" => Some(HCType::HC1),
            "hc2" => Some(HCType::HC2),
            "hc3" => Some(HCType::HC3),
            _ => None,
        }
    }
}

/// Compute robust (sandwich) covariance matrix for GLM coefficients.
///
/// # Arguments
/// * `x` - Design matrix (n × p)
/// * `pearson_resid` - Pearson residuals (y - μ) / sqrt(V(μ))
/// * `irls_weights` - IRLS working weights (from final iteration)
/// * `prior_weights` - User-supplied prior weights (or all 1s)
/// * `bread` - The (X'WX)⁻¹ matrix (unscaled covariance)
/// * `hc_type` - Which HC variant to use
///
/// # Returns
/// Robust covariance matrix (p × p)
///
/// # Details
/// For GLMs, we use a modified sandwich where:
/// - Working weights W = prior_weights × irls_weights
/// - Residuals are Pearson residuals scaled by sqrt(W)
///
/// The meat B = X' Ω X where Ω depends on the HC type.
pub fn robust_covariance(
    x: &Array2<f64>,
    pearson_resid: &Array1<f64>,
    irls_weights: &Array1<f64>,
    prior_weights: &Array1<f64>,
    bread: &Array2<f64>,
    hc_type: HCType,
) -> Array2<f64> {
    let n = x.nrows();
    let p = x.ncols();
    
    // Combined weights
    let combined_weights: Array1<f64> = prior_weights
        .iter()
        .zip(irls_weights.iter())
        .map(|(&pw, &iw)| pw * iw)
        .collect();
    
    // Compute leverage values for HC2/HC3 if needed
    let leverage = if matches!(hc_type, HCType::HC2 | HCType::HC3) {
        compute_leverage(x, &combined_weights, bread)
    } else {
        Array1::zeros(n)
    };
    
    // Compute the "meat" matrix: X' Ω X
    // Ω is diagonal with entries that depend on HC type
    let meat = compute_meat(x, pearson_resid, &combined_weights, &leverage, hc_type, n, p);
    
    // Sandwich: bread × meat × bread
    bread.dot(&meat).dot(bread)
}

/// Compute leverage (hat matrix diagonal) values.
///
/// h_ii = x_i' (X'WX)⁻¹ x_i × w_i
///
/// These measure how much each observation influences its own fitted value.
/// PARALLEL: Uses Rayon for large datasets.
fn compute_leverage(
    x: &Array2<f64>,
    weights: &Array1<f64>,
    cov_unscaled: &Array2<f64>,
) -> Array1<f64> {
    let n = x.nrows();
    let p = x.ncols();
    
    // Convert cov_unscaled to a flat vec for thread-safe access
    let cov_flat: Vec<f64> = cov_unscaled.iter().copied().collect();
    
    // PARALLEL: Compute leverage for each observation
    let leverage_vec: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|i| {
            let x_i = x.row(i);
            let w_i = weights[i];
            
            // Compute x_i' × (X'WX)⁻¹ × x_i manually for thread safety
            let mut h_ii = 0.0;
            for j in 0..p {
                let mut temp_j = 0.0;
                for k in 0..p {
                    temp_j += cov_flat[j * p + k] * x_i[k];
                }
                h_ii += x_i[j] * temp_j;
            }
            h_ii *= w_i;
            
            // Clamp to avoid numerical issues (h should be in [0, 1])
            h_ii.clamp(0.0, 0.9999)
        })
        .collect();
    
    Array1::from_vec(leverage_vec)
}

/// Compute the "meat" matrix for the sandwich estimator.
fn compute_meat(
    x: &Array2<f64>,
    pearson_resid: &Array1<f64>,
    weights: &Array1<f64>,
    leverage: &Array1<f64>,
    hc_type: HCType,
    n: usize,
    p: usize,
) -> Array2<f64> {
    // Compute adjusted squared residuals based on HC type
    let omega: Array1<f64> = match hc_type {
        HCType::HC0 => {
            // ω_i = w_i × ε_i²
            pearson_resid
                .iter()
                .zip(weights.iter())
                .map(|(&r, &w)| w * r * r)
                .collect()
        }
        HCType::HC1 => {
            // ω_i = w_i × ε_i² × n/(n-p)
            let scale = n as f64 / (n.saturating_sub(p)) as f64;
            pearson_resid
                .iter()
                .zip(weights.iter())
                .map(|(&r, &w)| scale * w * r * r)
                .collect()
        }
        HCType::HC2 => {
            // ω_i = w_i × ε_i² / (1 - h_ii)
            pearson_resid
                .iter()
                .zip(weights.iter())
                .zip(leverage.iter())
                .map(|((&r, &w), &h)| {
                    let denom = (1.0 - h).max(0.01); // Avoid division by zero
                    w * r * r / denom
                })
                .collect()
        }
        HCType::HC3 => {
            // ω_i = w_i × ε_i² / (1 - h_ii)²
            pearson_resid
                .iter()
                .zip(weights.iter())
                .zip(leverage.iter())
                .map(|((&r, &w), &h)| {
                    let denom = (1.0 - h).max(0.01);
                    w * r * r / (denom * denom)
                })
                .collect()
        }
    };
    
    // Compute X' Ω X where Ω = diag(omega)
    // This is equivalent to: sum over i of omega[i] * x_i * x_i'
    // PARALLEL: Use fold-reduce pattern for thread-safe accumulation
    let p = x.ncols();
    let n = x.nrows();
    
    let meat_flat: Vec<f64> = (0..n)
        .into_par_iter()
        .fold(
            || vec![0.0; p * p],
            |mut acc, i| {
                let omega_i = omega[i];
                let x_i = x.row(i);
                // Only compute upper triangle (symmetric matrix)
                for j in 0..p {
                    let xij_omega = x_i[j] * omega_i;
                    for k in j..p {
                        acc[j * p + k] += xij_omega * x_i[k];
                    }
                }
                acc
            },
        )
        .reduce(
            || vec![0.0; p * p],
            |mut a, b| {
                for i in 0..a.len() {
                    a[i] += b[i];
                }
                a
            },
        );
    
    // Convert to Array2 and fill symmetric entries
    let mut meat = Array2::zeros((p, p));
    for j in 0..p {
        for k in j..p {
            let val = meat_flat[j * p + k];
            meat[[j, k]] = val;
            meat[[k, j]] = val;
        }
    }
    
    meat
}

/// Compute robust standard errors from robust covariance matrix.
pub fn robust_standard_errors(robust_cov: &Array2<f64>) -> Array1<f64> {
    let p = robust_cov.nrows();
    (0..p)
        .map(|i| robust_cov[[i, i]].max(0.0).sqrt())
        .collect()
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

    #[test]
    fn test_hc_type_from_str() {
        assert_eq!(HCType::from_str("hc0"), Some(HCType::HC0));
        assert_eq!(HCType::from_str("HC1"), Some(HCType::HC1));
        assert_eq!(HCType::from_str("hC2"), Some(HCType::HC2));
        assert_eq!(HCType::from_str("HC3"), Some(HCType::HC3));
        assert_eq!(HCType::from_str("invalid"), None);
    }

    #[test]
    fn test_robust_covariance_basic() {
        use ndarray::{arr1, arr2};
        
        // Simple 3-observation, 2-parameter case
        let x = arr2(&[
            [1.0, 1.0],
            [1.0, 2.0],
            [1.0, 3.0],
        ]);
        let pearson_resid = arr1(&[0.1, -0.2, 0.15]);
        let irls_weights = arr1(&[1.0, 1.0, 1.0]);
        let prior_weights = arr1(&[1.0, 1.0, 1.0]);
        
        // Create a simple bread matrix (identity for testing)
        let bread = arr2(&[
            [0.5, 0.0],
            [0.0, 0.5],
        ]);
        
        // HC0 should produce a valid covariance matrix
        let cov = robust_covariance(&x, &pearson_resid, &irls_weights, &prior_weights, &bread, HCType::HC0);
        
        // Should be symmetric
        assert_abs_diff_eq!(cov[[0, 1]], cov[[1, 0]], epsilon = 1e-10);
        
        // Diagonal should be non-negative
        assert!(cov[[0, 0]] >= 0.0);
        assert!(cov[[1, 1]] >= 0.0);
    }

    #[test]
    fn test_robust_standard_errors() {
        use ndarray::arr2;
        
        // Positive definite covariance matrix
        let cov = arr2(&[
            [0.04, 0.01],
            [0.01, 0.09],
        ]);
        
        let se = robust_standard_errors(&cov);
        
        assert_abs_diff_eq!(se[0], 0.2, epsilon = 1e-10);
        assert_abs_diff_eq!(se[1], 0.3, epsilon = 1e-10);
    }

    #[test]
    fn test_hc1_larger_than_hc0() {
        use ndarray::{arr1, arr2};
        
        // HC1 should give larger standard errors than HC0 due to n/(n-p) correction
        let x = arr2(&[
            [1.0, 1.0],
            [1.0, 2.0],
            [1.0, 3.0],
            [1.0, 4.0],
        ]);
        let pearson_resid = arr1(&[0.1, -0.2, 0.15, -0.1]);
        let irls_weights = arr1(&[1.0, 1.0, 1.0, 1.0]);
        let prior_weights = arr1(&[1.0, 1.0, 1.0, 1.0]);
        let bread = arr2(&[
            [0.5, 0.0],
            [0.0, 0.5],
        ]);
        
        let cov_hc0 = robust_covariance(&x, &pearson_resid, &irls_weights, &prior_weights, &bread, HCType::HC0);
        let cov_hc1 = robust_covariance(&x, &pearson_resid, &irls_weights, &prior_weights, &bread, HCType::HC1);
        
        // HC1 should be larger by factor of n/(n-p) = 4/2 = 2
        let expected_ratio = 4.0 / 2.0;
        assert_abs_diff_eq!(cov_hc1[[0, 0]] / cov_hc0[[0, 0]], expected_ratio, epsilon = 1e-10);
    }
}
