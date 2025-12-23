// =============================================================================
// Dispersion Parameter Estimation
// =============================================================================
//
// The dispersion parameter φ (phi) scales the variance in GLMs:
//   Var(Y) = φ × V(μ)
//
// KNOWN VS ESTIMATED DISPERSION:
// ------------------------------
// - Poisson: φ = 1 (by assumption, variance = mean)
// - Binomial: φ = 1 (by assumption)
// - Gaussian: φ = σ² (estimated from residuals)
// - Gamma: φ = 1/shape (estimated from residuals)
//
// WHY IT MATTERS:
// ---------------
// Dispersion affects standard errors! If φ > 1 (overdispersion),
// the true standard errors are larger than what a φ=1 assumption gives.
//
// ESTIMATION METHODS:
// -------------------
// 1. Pearson-based: φ = X²/(n-p) where X² = Σ(y-μ)²/V(μ)
// 2. Deviance-based: φ = D/(n-p) where D = model deviance
//
// Both are consistent estimators. Pearson is more robust to outliers
// in some cases, deviance is more commonly used.
//
// OVERDISPERSION:
// ---------------
// If φ >> 1 for Poisson or Binomial, you have overdispersion.
// Solutions:
// - Use Quasi-Poisson/Quasi-Binomial (scale SE's by √φ)
// - Use Negative Binomial (for count data)
// - Add random effects
//
// =============================================================================

use ndarray::Array1;
use crate::families::Family;

/// Compute Pearson chi-squared statistic: X² = Σ(y-μ)²/V(μ)
///
/// This measures overall goodness of fit. For a well-specified model
/// with known dispersion φ=1, X² should be approximately chi-squared
/// with (n-p) degrees of freedom.
///
/// # Arguments
/// * `y` - Observed response values
/// * `mu` - Fitted mean values
/// * `family` - Distribution family (provides variance function)
/// * `weights` - Optional observation weights
///
/// # Returns
/// The Pearson chi-squared statistic
///
/// # Interpretation
/// X²/(n-p) estimates the dispersion parameter φ.
/// If >> 1 for Poisson/Binomial, indicates overdispersion.
pub fn pearson_chi2(
    y: &Array1<f64>,
    mu: &Array1<f64>,
    family: &dyn Family,
    weights: Option<&Array1<f64>>,
) -> f64 {
    let variance = family.variance(mu);
    
    let contributions: Array1<f64> = ndarray::Zip::from(y)
        .and(mu)
        .and(&variance)
        .map_collect(|&yi, &mui, &vi| {
            let diff = yi - mui;
            (diff * diff) / vi.max(1e-10)
        });
    
    match weights {
        Some(w) => (&contributions * w).sum(),
        None => contributions.sum(),
    }
}

/// Estimate dispersion using Pearson method: φ = X²/(n-p)
///
/// This is a moment-based estimator of the dispersion parameter.
///
/// # Arguments
/// * `y` - Observed response values
/// * `mu` - Fitted mean values
/// * `family` - Distribution family
/// * `df_resid` - Residual degrees of freedom (n - p)
/// * `weights` - Optional observation weights
///
/// # Returns
/// Estimated dispersion parameter
///
/// # Note
/// For Poisson and Binomial, this estimates what the dispersion
/// would be if you allowed it to vary (quasi-likelihood approach).
pub fn estimate_dispersion_pearson(
    y: &Array1<f64>,
    mu: &Array1<f64>,
    family: &dyn Family,
    df_resid: usize,
    weights: Option<&Array1<f64>>,
) -> f64 {
    if df_resid == 0 {
        return 1.0;
    }
    
    let chi2 = pearson_chi2(y, mu, family, weights);
    chi2 / (df_resid as f64)
}

/// Estimate dispersion using deviance method: φ = D/(n-p)
///
/// The deviance-based estimator is commonly used and is the default
/// in many GLM implementations.
///
/// # Arguments
/// * `deviance` - Model deviance
/// * `df_resid` - Residual degrees of freedom (n - p)
///
/// # Returns
/// Estimated dispersion parameter
///
/// # Note
/// This is equivalent to what statsmodels reports as `scale`.
pub fn estimate_dispersion_deviance(deviance: f64, df_resid: usize) -> f64 {
    if df_resid == 0 {
        return 1.0;
    }
    deviance / (df_resid as f64)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::families::{GaussianFamily, PoissonFamily};
    use ndarray::array;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_pearson_chi2_gaussian() {
        let y = array![1.0, 2.0, 3.0];
        let mu = array![1.0, 2.0, 3.0];  // Perfect fit
        let family = GaussianFamily;
        
        let chi2 = pearson_chi2(&y, &mu, &family, None);
        
        // Perfect fit should have X² = 0
        assert_abs_diff_eq!(chi2, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_pearson_chi2_gaussian_residuals() {
        let y = array![1.0, 2.0, 3.0];
        let mu = array![1.5, 2.0, 2.5];
        let family = GaussianFamily;
        
        let chi2 = pearson_chi2(&y, &mu, &family, None);
        
        // For Gaussian, V(μ) = 1, so X² = Σ(y-μ)²
        // = 0.25 + 0 + 0.25 = 0.5
        assert_abs_diff_eq!(chi2, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_pearson_chi2_poisson() {
        let y = array![2.0, 4.0];
        let mu = array![2.0, 2.0];
        let family = PoissonFamily;
        
        let chi2 = pearson_chi2(&y, &mu, &family, None);
        
        // (2-2)²/2 + (4-2)²/2 = 0 + 4/2 = 2.0
        assert_abs_diff_eq!(chi2, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_pearson_chi2_weighted() {
        let y = array![1.0, 2.0];
        let mu = array![2.0, 2.0];
        let family = GaussianFamily;
        let weights = array![2.0, 1.0];
        
        let chi2 = pearson_chi2(&y, &mu, &family, Some(&weights));
        
        // (1-2)² × 2 + (2-2)² × 1 = 1 × 2 + 0 = 2.0
        assert_abs_diff_eq!(chi2, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_dispersion_pearson() {
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let mu = array![1.5, 2.0, 3.0, 4.0, 4.5];
        let family = GaussianFamily;
        let df_resid = 3;  // 5 obs, 2 parameters
        
        let phi = estimate_dispersion_pearson(&y, &mu, &family, df_resid, None);
        
        // X² = 0.25 + 0 + 0 + 0 + 0.25 = 0.5
        // φ = 0.5 / 3 ≈ 0.167
        assert_abs_diff_eq!(phi, 0.5 / 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_dispersion_deviance() {
        let deviance = 10.0;
        let df_resid = 5;
        
        let phi = estimate_dispersion_deviance(deviance, df_resid);
        
        assert_abs_diff_eq!(phi, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_dispersion_zero_df() {
        // Edge case: saturated model with df_resid = 0
        let phi_dev = estimate_dispersion_deviance(10.0, 0);
        assert_abs_diff_eq!(phi_dev, 1.0, epsilon = 1e-10);
        
        let y = array![1.0];
        let mu = array![1.0];
        let phi_pear = estimate_dispersion_pearson(&y, &mu, &GaussianFamily, 0, None);
        assert_abs_diff_eq!(phi_pear, 1.0, epsilon = 1e-10);
    }
}
