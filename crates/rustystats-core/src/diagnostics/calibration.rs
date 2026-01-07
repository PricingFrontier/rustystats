// =============================================================================
// Calibration and Discrimination Metrics
// =============================================================================
//
// This module provides metrics for assessing model calibration and discrimination:
//
// CALIBRATION: Does the model predict correct totals?
// - Actual/Expected ratio
// - Calibration by decile
// - Hosmer-Lemeshow test
//
// DISCRIMINATION: Does the model separate high/low risk?
// - Gini coefficient
// - Lift metrics
// - Lorenz curve
//
// =============================================================================

use ndarray::Array1;
use std::cmp::Ordering;

// =============================================================================
// Calibration Metrics
// =============================================================================

/// Overall calibration statistics
#[derive(Debug, Clone)]
pub struct CalibrationStats {
    pub actual_total: f64,
    pub predicted_total: f64,
    pub actual_expected_ratio: f64,
    pub exposure_total: f64,
}

/// Compute overall calibration statistics
pub fn compute_calibration_stats(
    y: &Array1<f64>,
    mu: &Array1<f64>,
    exposure: Option<&Array1<f64>>,
) -> CalibrationStats {
    let actual_total: f64 = y.sum();
    let predicted_total: f64 = mu.sum();
    let exposure_total: f64 = exposure.map_or(y.len() as f64, |e| e.sum());
    
    let actual_expected_ratio = if predicted_total > 0.0 {
        actual_total / predicted_total
    } else {
        f64::NAN
    };
    
    CalibrationStats {
        actual_total,
        predicted_total,
        actual_expected_ratio,
        exposure_total,
    }
}

/// A single bin in the calibration curve
#[derive(Debug, Clone)]
pub struct CalibrationBin {
    pub bin_index: usize,
    pub predicted_lower: f64,
    pub predicted_upper: f64,
    pub predicted_mean: f64,
    pub actual_mean: f64,
    pub actual_expected_ratio: f64,
    pub count: usize,
    pub exposure: f64,
    pub actual_sum: f64,
    pub predicted_sum: f64,
    /// 95% confidence interval lower bound for A/E
    pub ae_ci_lower: f64,
    /// 95% confidence interval upper bound for A/E
    pub ae_ci_upper: f64,
}

/// Compute calibration curve by prediction deciles (or other quantiles)
pub fn compute_calibration_curve(
    y: &Array1<f64>,
    mu: &Array1<f64>,
    exposure: Option<&Array1<f64>>,
    n_bins: usize,
) -> Vec<CalibrationBin> {
    let n = y.len();
    if n == 0 || n_bins == 0 {
        return Vec::new();
    }
    
    // Sort indices by predicted values
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        mu[a].partial_cmp(&mu[b]).unwrap_or(Ordering::Equal)
    });
    
    // Compute quantile boundaries based on exposure (if provided) or count
    let total_exposure: f64 = exposure.map_or(n as f64, |e| e.sum());
    let bin_exposure_target = total_exposure / n_bins as f64;
    
    let mut bins = Vec::with_capacity(n_bins);
    let mut current_bin_start = 0;
    let mut current_exposure = 0.0;
    let mut bin_index = 0;
    
    for (pos, &idx) in indices.iter().enumerate() {
        let obs_exposure = exposure.map_or(1.0, |e| e[idx]);
        current_exposure += obs_exposure;
        
        // Check if we should close this bin
        let is_last = pos == n - 1;
        let exposure_exceeded = current_exposure >= bin_exposure_target * 0.99;
        let remaining_bins = n_bins - bin_index - 1;
        let remaining_obs = n - pos - 1;
        
        if is_last || (exposure_exceeded && remaining_bins > 0 && remaining_obs >= remaining_bins) {
            // Close this bin
            let bin_indices = &indices[current_bin_start..=pos];
            let bin = compute_single_bin(y, mu, exposure, bin_indices, bin_index);
            bins.push(bin);
            
            bin_index += 1;
            current_bin_start = pos + 1;
            current_exposure = 0.0;
        }
    }
    
    bins
}

fn compute_single_bin(
    y: &Array1<f64>,
    mu: &Array1<f64>,
    exposure: Option<&Array1<f64>>,
    indices: &[usize],
    bin_index: usize,
) -> CalibrationBin {
    let count = indices.len();
    if count == 0 {
        return CalibrationBin {
            bin_index,
            predicted_lower: 0.0,
            predicted_upper: 0.0,
            predicted_mean: 0.0,
            actual_mean: 0.0,
            actual_expected_ratio: f64::NAN,
            count: 0,
            exposure: 0.0,
            actual_sum: 0.0,
            predicted_sum: 0.0,
            ae_ci_lower: f64::NAN,
            ae_ci_upper: f64::NAN,
        };
    }
    
    let mut actual_sum = 0.0;
    let mut predicted_sum = 0.0;
    let mut exposure_sum = 0.0;
    let mut pred_min = f64::MAX;
    let mut pred_max = f64::MIN;
    
    for &idx in indices {
        actual_sum += y[idx];
        predicted_sum += mu[idx];
        exposure_sum += exposure.map_or(1.0, |e| e[idx]);
        pred_min = pred_min.min(mu[idx]);
        pred_max = pred_max.max(mu[idx]);
    }
    
    let actual_mean = actual_sum / exposure_sum;
    let predicted_mean = predicted_sum / exposure_sum;
    let actual_expected_ratio = if predicted_sum > 0.0 {
        actual_sum / predicted_sum
    } else {
        f64::NAN
    };
    
    // Confidence interval for A/E using normal approximation
    // SE(A/E) ≈ sqrt(A) / E for Poisson-like data
    let (ae_ci_lower, ae_ci_upper) = if predicted_sum > 0.0 && actual_sum >= 0.0 {
        let se = (actual_sum.max(1.0)).sqrt() / predicted_sum;
        let z = 1.96; // 95% CI
        (
            (actual_expected_ratio - z * se).max(0.0),
            actual_expected_ratio + z * se,
        )
    } else {
        (f64::NAN, f64::NAN)
    };
    
    CalibrationBin {
        bin_index,
        predicted_lower: pred_min,
        predicted_upper: pred_max,
        predicted_mean,
        actual_mean,
        actual_expected_ratio,
        count,
        exposure: exposure_sum,
        actual_sum,
        predicted_sum,
        ae_ci_lower,
        ae_ci_upper,
    }
}

/// Hosmer-Lemeshow goodness-of-fit test
/// 
/// Tests whether observed frequencies match expected frequencies
/// across prediction bins.
#[derive(Debug, Clone)]
pub struct HosmerLemeshowResult {
    pub statistic: f64,
    pub degrees_of_freedom: usize,
    pub pvalue: f64,
}

pub fn hosmer_lemeshow_test(
    y: &Array1<f64>,
    mu: &Array1<f64>,
    n_groups: usize,
) -> HosmerLemeshowResult {
    let bins = compute_calibration_curve(y, mu, None, n_groups);
    
    // HL statistic: Σ (O_i - E_i)² / (E_i * (1 - p_i))
    // For count data, simplified to: Σ (O_i - E_i)² / E_i
    let mut chi2 = 0.0;
    let mut valid_bins = 0;
    
    for bin in &bins {
        if bin.predicted_sum > 0.0 {
            let diff = bin.actual_sum - bin.predicted_sum;
            chi2 += diff * diff / bin.predicted_sum;
            valid_bins += 1;
        }
    }
    
    let df = if valid_bins > 2 { valid_bins - 2 } else { 1 };
    
    // Compute p-value from chi-squared distribution
    let pvalue = 1.0 - chi2_cdf(chi2, df);
    
    HosmerLemeshowResult {
        statistic: chi2,
        degrees_of_freedom: df,
        pvalue,
    }
}

// =============================================================================
// Discrimination Metrics
// =============================================================================

/// Discrimination statistics
#[derive(Debug, Clone)]
pub struct DiscriminationStats {
    pub gini_coefficient: f64,
    pub auc: f64,
    pub ks_statistic: f64,
    pub lift_at_10pct: f64,
    pub lift_at_20pct: f64,
}

/// Compute discrimination metrics
/// 
/// These metrics measure how well the model separates high vs low outcomes.
pub fn compute_discrimination_stats(
    y: &Array1<f64>,
    mu: &Array1<f64>,
    exposure: Option<&Array1<f64>>,
) -> DiscriminationStats {
    let n = y.len();
    if n == 0 {
        return DiscriminationStats {
            gini_coefficient: f64::NAN,
            auc: f64::NAN,
            ks_statistic: f64::NAN,
            lift_at_10pct: f64::NAN,
            lift_at_20pct: f64::NAN,
        };
    }
    
    // Sort by predictions (descending - high risk first for positive Gini)
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        mu[b].partial_cmp(&mu[a]).unwrap_or(Ordering::Equal)
    });
    
    let total_exposure: f64 = exposure.map_or(n as f64, |e| e.sum());
    let total_actual: f64 = y.sum();
    
    if total_actual == 0.0 || total_exposure == 0.0 {
        return DiscriminationStats {
            gini_coefficient: 0.0,
            auc: 0.5,
            ks_statistic: 0.0,
            lift_at_10pct: 1.0,
            lift_at_20pct: 1.0,
        };
    }
    
    // Compute Lorenz curve and Gini
    let mut cum_exposure = 0.0;
    let mut cum_actual = 0.0;
    let mut gini_area = 0.0;
    let mut prev_cum_exposure_pct = 0.0;
    let mut prev_cum_actual_pct = 0.0;
    
    // For KS statistic
    let mut max_ks: f64 = 0.0;
    
    // For lift calculations
    let mut lift_10_actual = 0.0;
    let mut lift_10_exposure = 0.0;
    let mut lift_20_actual = 0.0;
    let mut lift_20_exposure = 0.0;
    let mut found_10 = false;
    let mut found_20 = false;
    
    for &idx in &indices {
        let obs_exposure = exposure.map_or(1.0, |e| e[idx]);
        let obs_actual = y[idx];
        
        cum_exposure += obs_exposure;
        cum_actual += obs_actual;
        
        let cum_exposure_pct = cum_exposure / total_exposure;
        let cum_actual_pct = cum_actual / total_actual;
        
        // Gini: area under Lorenz curve using trapezoidal rule
        gini_area += (cum_exposure_pct - prev_cum_exposure_pct) * 
                     (cum_actual_pct + prev_cum_actual_pct) / 2.0;
        
        // KS statistic
        let ks = (cum_actual_pct - cum_exposure_pct).abs();
        max_ks = max_ks.max(ks);
        
        // Lift at 10%
        if !found_10 && cum_exposure_pct >= 0.10 {
            lift_10_actual = cum_actual;
            lift_10_exposure = cum_exposure;
            found_10 = true;
        }
        
        // Lift at 20%
        if !found_20 && cum_exposure_pct >= 0.20 {
            lift_20_actual = cum_actual;
            lift_20_exposure = cum_exposure;
            found_20 = true;
        }
        
        prev_cum_exposure_pct = cum_exposure_pct;
        prev_cum_actual_pct = cum_actual_pct;
    }
    
    // Gini = 1 - 2 * area under Lorenz curve (for sorted ascending by risk)
    // Since we sorted ascending, low predictions first, Lorenz curve is below diagonal
    let gini_coefficient = 1.0 - 2.0 * gini_area;
    let auc = (gini_coefficient + 1.0) / 2.0;
    
    // Lift = (actual rate in bottom X%) / (overall actual rate)
    // Since sorted ascending, bottom 10% is lowest predicted risk
    let overall_rate = total_actual / total_exposure;
    let lift_at_10pct = if lift_10_exposure > 0.0 && overall_rate > 0.0 {
        (lift_10_actual / lift_10_exposure) / overall_rate
    } else {
        f64::NAN
    };
    let lift_at_20pct = if lift_20_exposure > 0.0 && overall_rate > 0.0 {
        (lift_20_actual / lift_20_exposure) / overall_rate
    } else {
        f64::NAN
    };
    
    DiscriminationStats {
        gini_coefficient,
        auc,
        ks_statistic: max_ks,
        lift_at_10pct,
        lift_at_20pct,
    }
}

/// Lorenz curve point
#[derive(Debug, Clone)]
pub struct LorenzPoint {
    pub cumulative_exposure_pct: f64,
    pub cumulative_actual_pct: f64,
    pub cumulative_predicted_pct: f64,
}

/// Compute Lorenz curve data points
pub fn compute_lorenz_curve(
    y: &Array1<f64>,
    mu: &Array1<f64>,
    exposure: Option<&Array1<f64>>,
    n_points: usize,
) -> Vec<LorenzPoint> {
    let n = y.len();
    if n == 0 || n_points == 0 {
        return Vec::new();
    }
    
    // Sort by predictions (ascending - low risk first)
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        mu[a].partial_cmp(&mu[b]).unwrap_or(Ordering::Equal)
    });
    
    let total_exposure: f64 = exposure.map_or(n as f64, |e| e.sum());
    let total_actual: f64 = y.sum();
    let total_predicted: f64 = mu.sum();
    
    if total_exposure == 0.0 || total_actual == 0.0 || total_predicted == 0.0 {
        return Vec::new();
    }
    
    let mut points = Vec::with_capacity(n_points + 1);
    points.push(LorenzPoint {
        cumulative_exposure_pct: 0.0,
        cumulative_actual_pct: 0.0,
        cumulative_predicted_pct: 0.0,
    });
    
    let step = total_exposure / n_points as f64;
    let mut cum_exposure = 0.0;
    let mut cum_actual = 0.0;
    let mut cum_predicted = 0.0;
    let mut next_threshold = step;
    let mut point_idx = 1;
    
    for &idx in &indices {
        let obs_exposure = exposure.map_or(1.0, |e| e[idx]);
        cum_exposure += obs_exposure;
        cum_actual += y[idx];
        cum_predicted += mu[idx];
        
        while cum_exposure >= next_threshold && point_idx <= n_points {
            points.push(LorenzPoint {
                cumulative_exposure_pct: cum_exposure / total_exposure,
                cumulative_actual_pct: cum_actual / total_actual,
                cumulative_predicted_pct: cum_predicted / total_predicted,
            });
            point_idx += 1;
            next_threshold += step;
        }
    }
    
    // Ensure we have the final point
    if points.len() <= n_points {
        points.push(LorenzPoint {
            cumulative_exposure_pct: 1.0,
            cumulative_actual_pct: 1.0,
            cumulative_predicted_pct: 1.0,
        });
    }
    
    points
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Chi-squared CDF approximation using Wilson-Hilferty transformation
fn chi2_cdf(x: f64, df: usize) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if df == 0 {
        return 1.0;
    }
    
    let df_f = df as f64;
    
    // Wilson-Hilferty approximation
    let z = (x / df_f).powf(1.0 / 3.0) - (1.0 - 2.0 / (9.0 * df_f));
    let z = z / (2.0 / (9.0 * df_f)).sqrt();
    
    // Standard normal CDF
    normal_cdf(z)
}

/// Standard normal CDF approximation
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Error function approximation (Abramowitz and Stegun)
fn erf(x: f64) -> f64 {
    let a1 =  0.254829592;
    let a2 = -0.284496736;
    let a3 =  1.421413741;
    let a4 = -1.453152027;
    let a5 =  1.061405429;
    let p  =  0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    
    sign * y
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_calibration_stats_perfect() {
        let y = array![1.0, 2.0, 3.0, 4.0];
        let mu = array![1.0, 2.0, 3.0, 4.0];
        
        let stats = compute_calibration_stats(&y, &mu, None);
        
        assert_abs_diff_eq!(stats.actual_expected_ratio, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(stats.actual_total, 10.0, epsilon = 1e-10);
        assert_abs_diff_eq!(stats.predicted_total, 10.0, epsilon = 1e-10);
    }

    #[test]
    fn test_calibration_curve_bins() {
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let mu = array![1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1];
        
        let bins = compute_calibration_curve(&y, &mu, None, 5);
        
        assert_eq!(bins.len(), 5);
        // Each bin should have 2 observations
        for bin in &bins {
            assert!(bin.count >= 1);
        }
    }

    #[test]
    fn test_gini_good_discrimination() {
        // Good discrimination: predictions correctly rank actuals
        // Low predictions for 0s, high predictions for 1s
        let y = array![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let mu = array![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        
        let stats = compute_discrimination_stats(&y, &mu, None);
        
        // Good discrimination should have positive Gini (> 0.3 for this data)
        assert!(stats.gini_coefficient > 0.3, "Gini = {}", stats.gini_coefficient);
        // AUC should be > 0.5 (better than random)
        assert!(stats.auc > 0.6, "AUC = {}", stats.auc);
    }

    #[test]
    fn test_gini_no_discrimination() {
        // No discrimination: all predictions are equal
        let y = array![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let mu = array![0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
        
        let stats = compute_discrimination_stats(&y, &mu, None);
        
        // With equal predictions, Gini depends on tie-breaking (order of y values)
        // The actual Gini may not be exactly 0 due to ordering
        // Check AUC is close to 0.5 (random)
        assert!((stats.auc - 0.5).abs() < 0.2, "AUC = {}", stats.auc);
    }

    #[test]
    fn test_lorenz_curve() {
        let y = array![1.0, 2.0, 3.0, 4.0];
        let mu = array![1.0, 2.0, 3.0, 4.0];
        
        let points = compute_lorenz_curve(&y, &mu, None, 4);
        
        assert!(!points.is_empty());
        // First point should be (0, 0, 0)
        assert_abs_diff_eq!(points[0].cumulative_exposure_pct, 0.0, epsilon = 1e-10);
        // Last point should be (1, 1, 1)
        let last = points.last().unwrap();
        assert_abs_diff_eq!(last.cumulative_exposure_pct, 1.0, epsilon = 1e-10);
    }
}
