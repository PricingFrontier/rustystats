// =============================================================================
// Factor-Level Diagnostics
// =============================================================================
//
// This module provides diagnostics for individual factors (variables),
// both those included in the model and those not yet fitted.
//
// For each factor, we compute:
// - Actual vs Expected by level/bin
// - Loss metrics by level/bin
// - Residual patterns (correlation with residuals)
// - Improvement potential (for unfitted factors)
//
// =============================================================================

use ndarray::Array1;
use std::collections::HashMap;
use std::cmp::Ordering;

use super::loss::compute_family_loss;

// =============================================================================
// Factor Types
// =============================================================================

/// Type of factor
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FactorType {
    Continuous,
    Categorical,
}

/// Configuration for a factor to analyze
#[derive(Debug, Clone)]
pub struct FactorConfig {
    pub name: String,
    pub factor_type: FactorType,
    pub in_model: bool,
    pub transformation: Option<String>, // e.g., "bs(age, df=5)"
}

// =============================================================================
// Univariate Statistics
// =============================================================================

/// Basic statistics for a continuous factor
#[derive(Debug, Clone)]
pub struct ContinuousStats {
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
    pub missing_count: usize,
    pub percentiles: Percentiles,
}

#[derive(Debug, Clone)]
pub struct Percentiles {
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

/// Compute univariate statistics for a continuous variable
pub fn compute_continuous_stats(values: &[f64]) -> ContinuousStats {
    let mut valid_values: Vec<f64> = values.iter()
        .filter(|&&v| !v.is_nan() && !v.is_infinite())
        .cloned()
        .collect();
    
    let missing_count = values.len() - valid_values.len();
    
    if valid_values.is_empty() {
        return ContinuousStats {
            mean: f64::NAN,
            std: f64::NAN,
            min: f64::NAN,
            max: f64::NAN,
            missing_count,
            percentiles: Percentiles {
                p1: f64::NAN, p5: f64::NAN, p10: f64::NAN, p25: f64::NAN,
                p50: f64::NAN, p75: f64::NAN, p90: f64::NAN, p95: f64::NAN, p99: f64::NAN,
            },
        };
    }
    
    valid_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    
    let n = valid_values.len();
    let mean: f64 = valid_values.iter().sum::<f64>() / n as f64;
    let variance: f64 = valid_values.iter()
        .map(|&v| (v - mean).powi(2))
        .sum::<f64>() / n as f64;
    let std = variance.sqrt();
    
    let percentile = |p: f64| -> f64 {
        let idx = (p * (n - 1) as f64).round() as usize;
        valid_values[idx.min(n - 1)]
    };
    
    ContinuousStats {
        mean,
        std,
        min: valid_values[0],
        max: valid_values[n - 1],
        missing_count,
        percentiles: Percentiles {
            p1: percentile(0.01),
            p5: percentile(0.05),
            p10: percentile(0.10),
            p25: percentile(0.25),
            p50: percentile(0.50),
            p75: percentile(0.75),
            p90: percentile(0.90),
            p95: percentile(0.95),
            p99: percentile(0.99),
        },
    }
}

/// Statistics for a categorical factor level
#[derive(Debug, Clone)]
pub struct LevelStats {
    pub level: String,
    pub count: usize,
    pub percentage: f64,
}

/// Distribution of categorical levels
#[derive(Debug, Clone)]
pub struct CategoricalDistribution {
    pub n_levels: usize,
    pub levels: Vec<LevelStats>,
    pub n_rare_levels: usize,
    pub rare_level_total_pct: f64,
}

/// Compute distribution for categorical variable
pub fn compute_categorical_distribution(
    values: &[String],
    rare_threshold_pct: f64,
) -> CategoricalDistribution {
    let n = values.len();
    if n == 0 {
        return CategoricalDistribution {
            n_levels: 0,
            levels: Vec::new(),
            n_rare_levels: 0,
            rare_level_total_pct: 0.0,
        };
    }
    
    // Count occurrences
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for v in values {
        *counts.entry(v.as_str()).or_insert(0) += 1;
    }
    
    // Convert to sorted vector (by count, descending)
    let mut levels: Vec<LevelStats> = counts.iter()
        .map(|(&level, &count)| LevelStats {
            level: level.to_string(),
            count,
            percentage: 100.0 * count as f64 / n as f64,
        })
        .collect();
    
    levels.sort_by(|a, b| b.count.cmp(&a.count));
    
    // Count rare levels
    let n_rare_levels = levels.iter()
        .filter(|l| l.percentage < rare_threshold_pct)
        .count();
    let rare_level_total_pct: f64 = levels.iter()
        .filter(|l| l.percentage < rare_threshold_pct)
        .map(|l| l.percentage)
        .sum();
    
    CategoricalDistribution {
        n_levels: levels.len(),
        levels,
        n_rare_levels,
        rare_level_total_pct,
    }
}

// =============================================================================
// Actual vs Expected Analysis
// =============================================================================

/// A/E statistics for a single bin or level
#[derive(Debug, Clone)]
pub struct ActualExpectedBin {
    pub bin_index: usize,
    pub bin_label: String,
    pub bin_lower: Option<f64>,  // For continuous
    pub bin_upper: Option<f64>,  // For continuous
    pub count: usize,
    pub exposure: f64,
    pub actual_sum: f64,
    pub predicted_sum: f64,
    pub actual_mean: f64,
    pub predicted_mean: f64,
    pub actual_expected_ratio: f64,
    pub loss: f64,
    pub ae_ci_lower: f64,
    pub ae_ci_upper: f64,
}

/// Compute A/E analysis for a continuous factor using quantile bins
pub fn compute_ae_continuous(
    factor_values: &[f64],
    y: &Array1<f64>,
    mu: &Array1<f64>,
    exposure: Option<&Array1<f64>>,
    family: &str,
    n_bins: usize,
    var_power: Option<f64>,
    theta: Option<f64>,
) -> Vec<ActualExpectedBin> {
    let n = factor_values.len();
    if n == 0 || n != y.len() {
        return Vec::new();
    }
    
    // Get quantile boundaries
    let mut sorted_vals: Vec<(usize, f64)> = factor_values.iter()
        .enumerate()
        .filter(|(_, &v)| !v.is_nan() && !v.is_infinite())
        .map(|(i, &v)| (i, v))
        .collect();
    sorted_vals.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    
    if sorted_vals.is_empty() {
        return Vec::new();
    }
    
    // Compute quantile boundaries
    let quantiles: Vec<f64> = (0..=n_bins)
        .map(|i| {
            let p = i as f64 / n_bins as f64;
            let idx = ((sorted_vals.len() - 1) as f64 * p).round() as usize;
            sorted_vals[idx].1
        })
        .collect();
    
    // Assign each observation to a bin
    let mut bin_data: Vec<Vec<usize>> = vec![Vec::new(); n_bins];
    for (orig_idx, val) in factor_values.iter().enumerate() {
        if val.is_nan() || val.is_infinite() {
            continue;
        }
        for bin_idx in 0..n_bins {
            let lower = quantiles[bin_idx];
            let upper = quantiles[bin_idx + 1];
            if *val >= lower && (*val < upper || bin_idx == n_bins - 1) {
                bin_data[bin_idx].push(orig_idx);
                break;
            }
        }
    }
    
    // Compute statistics for each bin
    bin_data.iter()
        .enumerate()
        .map(|(bin_idx, indices)| {
            compute_ae_bin(
                indices,
                bin_idx,
                format!("{:.2}-{:.2}", quantiles[bin_idx], quantiles[bin_idx + 1]),
                Some(quantiles[bin_idx]),
                Some(quantiles[bin_idx + 1]),
                y,
                mu,
                exposure,
                family,
                var_power,
                theta,
            )
        })
        .collect()
}

/// Compute A/E analysis for a categorical factor
pub fn compute_ae_categorical(
    factor_values: &[String],
    y: &Array1<f64>,
    mu: &Array1<f64>,
    exposure: Option<&Array1<f64>>,
    family: &str,
    var_power: Option<f64>,
    theta: Option<f64>,
    rare_threshold_pct: f64,
    max_levels: usize,
) -> Vec<ActualExpectedBin> {
    let n = factor_values.len();
    if n == 0 || n != y.len() {
        return Vec::new();
    }
    
    // Group by level
    let mut level_indices: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, level) in factor_values.iter().enumerate() {
        level_indices.entry(level.as_str()).or_insert_with(Vec::new).push(i);
    }
    
    // Sort levels by exposure (descending)
    let total_exposure: f64 = exposure.map_or(n as f64, |e| e.sum());
    let mut level_exposures: Vec<(&str, f64)> = level_indices.iter()
        .map(|(&level, indices)| {
            let exp: f64 = indices.iter()
                .map(|&i| exposure.map_or(1.0, |e| e[i]))
                .sum();
            (level, exp)
        })
        .collect();
    level_exposures.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    
    // Compute bins, grouping rare levels into "Other"
    let mut bins = Vec::new();
    let mut other_indices = Vec::new();
    
    for (bin_idx, &(level, exp)) in level_exposures.iter().enumerate() {
        let pct = 100.0 * exp / total_exposure;
        let indices = &level_indices[level];
        
        if pct < rare_threshold_pct || bin_idx >= max_levels - 1 {
            // Add to "Other" category
            other_indices.extend(indices.iter().cloned());
        } else {
            bins.push(compute_ae_bin(
                indices,
                bin_idx,
                level.to_string(),
                None,
                None,
                y,
                mu,
                exposure,
                family,
                var_power,
                theta,
            ));
        }
    }
    
    // Add "Other" bin if non-empty
    if !other_indices.is_empty() {
        bins.push(compute_ae_bin(
            &other_indices,
            bins.len(),
            "_Other".to_string(),
            None,
            None,
            y,
            mu,
            exposure,
            family,
            var_power,
            theta,
        ));
    }
    
    bins
}

fn compute_ae_bin(
    indices: &[usize],
    bin_idx: usize,
    label: String,
    lower: Option<f64>,
    upper: Option<f64>,
    y: &Array1<f64>,
    mu: &Array1<f64>,
    exposure: Option<&Array1<f64>>,
    family: &str,
    var_power: Option<f64>,
    theta: Option<f64>,
) -> ActualExpectedBin {
    let count = indices.len();
    if count == 0 {
        return ActualExpectedBin {
            bin_index: bin_idx,
            bin_label: label,
            bin_lower: lower,
            bin_upper: upper,
            count: 0,
            exposure: 0.0,
            actual_sum: 0.0,
            predicted_sum: 0.0,
            actual_mean: f64::NAN,
            predicted_mean: f64::NAN,
            actual_expected_ratio: f64::NAN,
            loss: f64::NAN,
            ae_ci_lower: f64::NAN,
            ae_ci_upper: f64::NAN,
        };
    }
    
    let mut actual_sum = 0.0;
    let mut predicted_sum = 0.0;
    let mut exposure_sum = 0.0;
    let mut y_bin = Vec::with_capacity(count);
    let mut mu_bin = Vec::with_capacity(count);
    let mut w_bin = Vec::with_capacity(count);
    
    for &i in indices {
        let yi = y[i];
        let mui = mu[i];
        let wi = exposure.map_or(1.0, |e| e[i]);
        
        actual_sum += yi;
        predicted_sum += mui;
        exposure_sum += wi;
        y_bin.push(yi);
        mu_bin.push(mui);
        w_bin.push(wi);
    }
    
    let actual_mean = actual_sum / exposure_sum;
    let predicted_mean = predicted_sum / exposure_sum;
    let actual_expected_ratio = if predicted_sum > 0.0 {
        actual_sum / predicted_sum
    } else {
        f64::NAN
    };
    
    // Compute loss for this bin
    let y_arr = Array1::from_vec(y_bin);
    let mu_arr = Array1::from_vec(mu_bin);
    let w_arr = Array1::from_vec(w_bin);
    let loss = compute_family_loss(family, &y_arr, &mu_arr, Some(&w_arr), var_power, theta);
    
    // Confidence interval for A/E
    let (ae_ci_lower, ae_ci_upper) = if predicted_sum > 0.0 && actual_sum >= 0.0 {
        let se = (actual_sum.max(1.0)).sqrt() / predicted_sum;
        let z = 1.96;
        (
            (actual_expected_ratio - z * se).max(0.0),
            actual_expected_ratio + z * se,
        )
    } else {
        (f64::NAN, f64::NAN)
    };
    
    ActualExpectedBin {
        bin_index: bin_idx,
        bin_label: label,
        bin_lower: lower,
        bin_upper: upper,
        count,
        exposure: exposure_sum,
        actual_sum,
        predicted_sum,
        actual_mean,
        predicted_mean,
        actual_expected_ratio,
        loss,
        ae_ci_lower,
        ae_ci_upper,
    }
}

// =============================================================================
// Residual Pattern Analysis
// =============================================================================

/// Residual pattern statistics for a factor
#[derive(Debug, Clone)]
pub struct ResidualPattern {
    pub correlation_with_residuals: f64,
    pub mean_residual_by_bin: Vec<f64>,
    pub trend_slope: f64,
    pub trend_pvalue: f64,
    pub residual_variance_explained: f64,
}

/// Compute residual patterns for a continuous factor
pub fn compute_residual_pattern_continuous(
    factor_values: &[f64],
    residuals: &Array1<f64>,
    n_bins: usize,
) -> ResidualPattern {
    let n = factor_values.len();
    if n == 0 || n != residuals.len() {
        return ResidualPattern {
            correlation_with_residuals: f64::NAN,
            mean_residual_by_bin: Vec::new(),
            trend_slope: f64::NAN,
            trend_pvalue: f64::NAN,
            residual_variance_explained: f64::NAN,
        };
    }
    
    // Compute correlation
    let valid_pairs: Vec<(f64, f64)> = factor_values.iter()
        .zip(residuals.iter())
        .filter(|(&f, _)| !f.is_nan() && !f.is_infinite())
        .map(|(&f, &r)| (f, r))
        .collect();
    
    let correlation = compute_correlation(&valid_pairs);
    
    // Compute mean residual by bin
    let mut sorted_pairs = valid_pairs.clone();
    sorted_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    
    let bin_size = (sorted_pairs.len() + n_bins - 1) / n_bins;
    let mean_residual_by_bin: Vec<f64> = sorted_pairs
        .chunks(bin_size.max(1))
        .map(|chunk| {
            let sum: f64 = chunk.iter().map(|&(_, r)| r).sum();
            sum / chunk.len() as f64
        })
        .collect();
    
    // Compute linear trend
    let (slope, pvalue) = compute_linear_trend(&valid_pairs);
    
    // R² of residuals ~ factor (how much variance could this factor explain)
    let r_squared = correlation * correlation;
    
    ResidualPattern {
        correlation_with_residuals: correlation,
        mean_residual_by_bin,
        trend_slope: slope,
        trend_pvalue: pvalue,
        residual_variance_explained: r_squared,
    }
}

/// Compute residual patterns for a categorical factor
pub fn compute_residual_pattern_categorical(
    factor_values: &[String],
    residuals: &Array1<f64>,
) -> ResidualPattern {
    let n = factor_values.len();
    if n == 0 || n != residuals.len() {
        return ResidualPattern {
            correlation_with_residuals: f64::NAN,
            mean_residual_by_bin: Vec::new(),
            trend_slope: f64::NAN,
            trend_pvalue: f64::NAN,
            residual_variance_explained: f64::NAN,
        };
    }
    
    // Group residuals by level
    let mut level_residuals: HashMap<&str, Vec<f64>> = HashMap::new();
    for (i, level) in factor_values.iter().enumerate() {
        level_residuals.entry(level.as_str())
            .or_insert_with(Vec::new)
            .push(residuals[i]);
    }
    
    // Compute mean residual by level
    let mut level_means: Vec<(&str, f64, usize)> = level_residuals.iter()
        .map(|(&level, resids)| {
            let mean = resids.iter().sum::<f64>() / resids.len() as f64;
            (level, mean, resids.len())
        })
        .collect();
    level_means.sort_by(|a, b| b.2.cmp(&a.2)); // Sort by count
    
    let mean_residual_by_bin: Vec<f64> = level_means.iter()
        .map(|&(_, mean, _)| mean)
        .collect();
    
    // Compute variance explained (eta-squared)
    let overall_mean: f64 = residuals.sum() / n as f64;
    let ss_total: f64 = residuals.iter()
        .map(|&r| (r - overall_mean).powi(2))
        .sum();
    
    let ss_between: f64 = level_means.iter()
        .map(|&(_, level_mean, count)| {
            count as f64 * (level_mean - overall_mean).powi(2)
        })
        .sum();
    
    let eta_squared = if ss_total > 0.0 { ss_between / ss_total } else { 0.0 };
    
    // Mean absolute residual correlation (approximation)
    let mean_abs_resid: f64 = mean_residual_by_bin.iter()
        .map(|&m| m.abs())
        .sum::<f64>() / mean_residual_by_bin.len().max(1) as f64;
    
    ResidualPattern {
        correlation_with_residuals: mean_abs_resid, // For categorical, use mean abs residual
        mean_residual_by_bin,
        trend_slope: f64::NAN, // Not applicable for categorical
        trend_pvalue: f64::NAN,
        residual_variance_explained: eta_squared,
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

fn compute_correlation(pairs: &[(f64, f64)]) -> f64 {
    let n = pairs.len();
    if n < 2 {
        return f64::NAN;
    }
    
    let sum_x: f64 = pairs.iter().map(|&(x, _)| x).sum();
    let sum_y: f64 = pairs.iter().map(|&(_, y)| y).sum();
    let mean_x = sum_x / n as f64;
    let mean_y = sum_y / n as f64;
    
    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    
    for &(x, y) in pairs {
        let dx = x - mean_x;
        let dy = y - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    
    if var_x == 0.0 || var_y == 0.0 {
        return 0.0;
    }
    
    cov / (var_x * var_y).sqrt()
}

fn compute_linear_trend(pairs: &[(f64, f64)]) -> (f64, f64) {
    let n = pairs.len();
    if n < 3 {
        return (f64::NAN, f64::NAN);
    }
    
    let sum_x: f64 = pairs.iter().map(|&(x, _)| x).sum();
    let sum_y: f64 = pairs.iter().map(|&(_, y)| y).sum();
    let mean_x = sum_x / n as f64;
    let mean_y = sum_y / n as f64;
    
    let mut ss_xx = 0.0;
    let mut ss_xy = 0.0;
    
    for &(x, y) in pairs {
        let dx = x - mean_x;
        let dy = y - mean_y;
        ss_xx += dx * dx;
        ss_xy += dx * dy;
    }
    
    if ss_xx == 0.0 {
        return (0.0, 1.0);
    }
    
    let slope = ss_xy / ss_xx;
    
    // Compute t-statistic for slope
    let ss_res: f64 = pairs.iter()
        .map(|&(x, y)| {
            let pred = mean_y + slope * (x - mean_x);
            (y - pred).powi(2)
        })
        .sum();
    
    let df = n - 2;
    let mse = ss_res / df as f64;
    let se_slope = (mse / ss_xx).sqrt();
    
    let t_stat = if se_slope > 0.0 { slope / se_slope } else { 0.0 };
    
    // Approximate p-value from t-distribution
    let pvalue = 2.0 * (1.0 - t_cdf(t_stat.abs(), df));
    
    (slope, pvalue)
}

/// Approximation of t-distribution CDF
fn t_cdf(t: f64, df: usize) -> f64 {
    // Use normal approximation for large df
    if df > 30 {
        return normal_cdf_approx(t);
    }
    
    // Simple approximation for small df
    let x = df as f64 / (df as f64 + t * t);
    let a = df as f64 / 2.0;
    let b = 0.5;
    
    // Incomplete beta function approximation
    0.5 + 0.5 * t.signum() * (1.0 - incomplete_beta_approx(x, a, b))
}

fn normal_cdf_approx(x: f64) -> f64 {
    0.5 * (1.0 + erf_approx(x / std::f64::consts::SQRT_2))
}

fn erf_approx(x: f64) -> f64 {
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

fn incomplete_beta_approx(x: f64, a: f64, b: f64) -> f64 {
    // Simple approximation - for accurate values, use a proper library
    if x <= 0.0 { return 0.0; }
    if x >= 1.0 { return 1.0; }
    
    // Continued fraction approximation (first few terms)
    let mut result = x.powf(a) * (1.0 - x).powf(b) / a;
    result *= (a + b) / (a + 1.0);
    result.min(1.0).max(0.0)
}

// =============================================================================
// Factor Deviance Computation
// =============================================================================

/// Result for deviance by factor level
#[derive(Debug, Clone)]
pub struct DevianceByLevel {
    pub level: String,
    pub count: usize,
    pub deviance: f64,
    pub deviance_pct: f64,
    pub mean_deviance: f64,
    pub actual_sum: f64,
    pub predicted_sum: f64,
    pub ae_ratio: f64,
    pub is_problem: bool,
}

/// Result for factor deviance computation
#[derive(Debug, Clone)]
pub struct FactorDevianceResult {
    pub factor_name: String,
    pub total_deviance: f64,
    pub levels: Vec<DevianceByLevel>,
    pub problem_levels: Vec<String>,
}

/// Compute deviance breakdown by categorical factor level
/// 
/// This is much faster than Python loops for large datasets
pub fn compute_factor_deviance(
    factor_name: &str,
    factor_values: &[String],
    y: &Array1<f64>,
    mu: &Array1<f64>,
    family: &str,
    var_power: f64,
    theta: f64,
) -> FactorDevianceResult {
    let n = factor_values.len();
    if n == 0 || n != y.len() || n != mu.len() {
        return FactorDevianceResult {
            factor_name: factor_name.to_string(),
            total_deviance: 0.0,
            levels: Vec::new(),
            problem_levels: Vec::new(),
        };
    }
    
    // Compute unit deviances using family-specific formula
    let unit_deviances: Vec<f64> = y.iter().zip(mu.iter())
        .map(|(&yi, &mui)| unit_deviance_for_family(yi, mui, family, var_power, theta))
        .collect();
    
    let total_deviance: f64 = unit_deviances.iter().sum();
    let mean_unit_deviance = total_deviance / n as f64;
    
    // Group by level using HashMap for O(n) complexity
    let mut level_data: HashMap<&str, (usize, f64, f64, f64)> = HashMap::new();
    
    for (i, level) in factor_values.iter().enumerate() {
        let entry = level_data.entry(level.as_str()).or_insert((0, 0.0, 0.0, 0.0));
        entry.0 += 1;                    // count
        entry.1 += unit_deviances[i];    // deviance sum
        entry.2 += y[i];                 // actual sum
        entry.3 += mu[i];                // predicted sum
    }
    
    // Build results
    let mut levels: Vec<DevianceByLevel> = Vec::with_capacity(level_data.len());
    let mut problem_levels: Vec<String> = Vec::new();
    
    for (level, (count, deviance, actual, predicted)) in level_data {
        let deviance_pct = if total_deviance > 0.0 { 100.0 * deviance / total_deviance } else { 0.0 };
        let mean_deviance = if count > 0 { deviance / count as f64 } else { 0.0 };
        let ae_ratio = if predicted > 0.0 { actual / predicted } else { f64::NAN };
        
        // Problem detection
        let expected_pct = 100.0 * count as f64 / n as f64;
        let is_problem = mean_deviance > mean_unit_deviance * 1.5 
            || (ae_ratio - 1.0).abs() > 0.15
            || deviance_pct > expected_pct * 2.0;
        
        if is_problem {
            problem_levels.push(level.to_string());
        }
        
        levels.push(DevianceByLevel {
            level: level.to_string(),
            count,
            deviance,
            deviance_pct,
            mean_deviance,
            actual_sum: actual,
            predicted_sum: predicted,
            ae_ratio,
            is_problem,
        });
    }
    
    // Sort by deviance (highest first)
    levels.sort_by(|a, b| b.deviance.partial_cmp(&a.deviance).unwrap_or(Ordering::Equal));
    
    FactorDevianceResult {
        factor_name: factor_name.to_string(),
        total_deviance,
        levels,
        problem_levels,
    }
}

/// Compute unit deviance for a single observation
fn unit_deviance_for_family(y: f64, mu: f64, family: &str, var_power: f64, theta: f64) -> f64 {
    let lower = family.to_lowercase();
    let mu_safe = mu.max(1e-10);
    let y_safe = y.max(0.0);
    
    match lower.as_str() {
        "gaussian" | "normal" => (y - mu).powi(2),
        "poisson" => {
            if y_safe > 0.0 {
                2.0 * (y_safe * (y_safe / mu_safe).ln() - (y_safe - mu_safe))
            } else {
                2.0 * mu_safe
            }
        }
        "binomial" => {
            let y_clamp = y.max(1e-10).min(1.0 - 1e-10);
            let mu_clamp = mu.max(1e-10).min(1.0 - 1e-10);
            2.0 * (y_clamp * (y_clamp / mu_clamp).ln() + (1.0 - y_clamp) * ((1.0 - y_clamp) / (1.0 - mu_clamp)).ln())
        }
        "gamma" => {
            2.0 * ((y_safe - mu_safe) / mu_safe - (y_safe / mu_safe).ln())
        }
        "tweedie" => {
            // Tweedie deviance depends on var_power
            if (var_power - 1.0).abs() < 1e-6 {
                // Quasi-Poisson
                2.0 * (y_safe * (y_safe / mu_safe).ln() - (y_safe - mu_safe))
            } else if (var_power - 2.0).abs() < 1e-6 {
                // Gamma
                2.0 * ((y_safe - mu_safe) / mu_safe - (y_safe / mu_safe).ln())
            } else {
                // General Tweedie
                let p = var_power;
                if y_safe > 0.0 {
                    2.0 * (y_safe.powf(2.0 - p) / ((1.0 - p) * (2.0 - p)) 
                           - y_safe * mu_safe.powf(1.0 - p) / (1.0 - p)
                           + mu_safe.powf(2.0 - p) / (2.0 - p))
                } else {
                    2.0 * mu_safe.powf(2.0 - p) / (2.0 - p)
                }
            }
        }
        _ if lower.starts_with("negbin") || lower.starts_with("negativebinomial") => {
            // Negative binomial deviance
            if y_safe > 0.0 {
                2.0 * (y_safe * (y_safe / mu_safe).ln() - (y_safe + theta) * ((y_safe + theta) / (mu_safe + theta)).ln())
            } else {
                2.0 * theta * ((theta) / (mu_safe + theta)).ln()
            }
        }
        _ => (y - mu).powi(2), // Default to Gaussian
    }
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
    fn test_continuous_stats() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let stats = compute_continuous_stats(&values);
        
        assert_abs_diff_eq!(stats.mean, 5.5, epsilon = 1e-10);
        assert_abs_diff_eq!(stats.min, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(stats.max, 10.0, epsilon = 1e-10);
        assert_eq!(stats.missing_count, 0);
    }

    #[test]
    fn test_categorical_distribution() {
        let values = vec![
            "A".to_string(), "A".to_string(), "A".to_string(),
            "B".to_string(), "B".to_string(),
            "C".to_string(),
        ];
        
        let dist = compute_categorical_distribution(&values, 10.0);
        
        assert_eq!(dist.n_levels, 3);
        assert_eq!(dist.levels[0].level, "A");
        assert_eq!(dist.levels[0].count, 3);
    }

    #[test]
    fn test_ae_continuous() {
        let factor = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y = array![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let mu = array![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        
        let bins = compute_ae_continuous(&factor, &y, &mu, None, "gaussian", 5, None, None);
        
        assert_eq!(bins.len(), 5);
        // Perfect predictions should have A/E ≈ 1
        for bin in &bins {
            assert!((bin.actual_expected_ratio - 1.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_residual_correlation() {
        // Perfect positive correlation
        let pairs = vec![(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0)];
        let corr = compute_correlation(&pairs);
        assert_abs_diff_eq!(corr, 1.0, epsilon = 1e-10);
        
        // No correlation
        let pairs = vec![(1.0, 2.0), (2.0, 1.0), (3.0, 2.0), (4.0, 1.0)];
        let corr = compute_correlation(&pairs);
        assert!(corr.abs() < 0.5);
    }
}
