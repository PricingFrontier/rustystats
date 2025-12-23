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

use ndarray::Array1;
use crate::families::Family;
use crate::links::Link;

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
        .and(&variance)
        .map_collect(|&yi, &mui, &vi| {
            let std_dev = vi.sqrt().max(1e-10);
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
        .map_collect(|&yi, &mui, &di| {
            (yi - mui) * di
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
    use ndarray::array;
    use approx::assert_abs_diff_eq;

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
}
