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

use ndarray::{Array1, Array2};
use rayon::prelude::*;
use std::collections::HashMap;

use super::ci::wilson_poisson_rate_ci;
use super::distributions::chi2_cdf;
use super::loss::compute_family_loss;
use crate::constants::{
    DEFAULT_CI_ALPHA, MU_MAX_PROBABILITY, MU_MIN_POSITIVE, MU_MIN_PROBABILITY,
    TWEEDIE_VAR_POWER_TOL,
};

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
    let mut valid_values: Vec<f64> = values
        .iter()
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
                p1: f64::NAN,
                p5: f64::NAN,
                p10: f64::NAN,
                p25: f64::NAN,
                p50: f64::NAN,
                p75: f64::NAN,
                p90: f64::NAN,
                p95: f64::NAN,
                p99: f64::NAN,
            },
        };
    }

    valid_values.sort_by(|a, b| a.total_cmp(b));

    let n = valid_values.len();
    let mean: f64 = valid_values.iter().sum::<f64>() / n as f64;
    let variance: f64 = valid_values
        .iter()
        .map(|&v| (v - mean).powi(2))
        .sum::<f64>()
        / n as f64;
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
    let mut levels: Vec<LevelStats> = counts
        .iter()
        .map(|(&level, &count)| LevelStats {
            level: level.to_string(),
            count,
            percentage: 100.0 * count as f64 / n as f64,
        })
        .collect();

    levels.sort_by(|a, b| b.count.cmp(&a.count));

    // Count rare levels
    let n_rare_levels = levels
        .iter()
        .filter(|l| l.percentage < rare_threshold_pct)
        .count();
    let rare_level_total_pct: f64 = levels
        .iter()
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
    pub bin_lower: Option<f64>, // For continuous
    pub bin_upper: Option<f64>, // For continuous
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
    let mut sorted_vals: Vec<(usize, f64)> = factor_values
        .iter()
        .enumerate()
        .filter(|(_, &v)| !v.is_nan() && !v.is_infinite())
        .map(|(i, &v)| (i, v))
        .collect();
    sorted_vals.sort_by(|a, b| a.1.total_cmp(&b.1));

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
    bin_data
        .iter()
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

/// Compute A/E analysis for many continuous factors in parallel.
///
/// Each entry in `factor_values_list` is one factor's per-row values; the
/// shared `y`, `mu`, `exposure`, `family`, `n_bins`, `var_power`, and `theta`
/// are reused across all factors. The expensive per-factor work (sort for
/// quantile binning, bin assignment, per-bin loss) is independent across
/// factors, so we use rayon to parallelize across the k factors.
pub fn compute_ae_continuous_batch(
    factor_values_list: &[&[f64]],
    y: &Array1<f64>,
    mu: &Array1<f64>,
    exposure: Option<&Array1<f64>>,
    family: &str,
    n_bins: usize,
    var_power: Option<f64>,
    theta: Option<f64>,
) -> Vec<Vec<ActualExpectedBin>> {
    factor_values_list
        .par_iter()
        .map(|values| {
            compute_ae_continuous(values, y, mu, exposure, family, n_bins, var_power, theta)
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
        level_indices.entry(level.as_str()).or_default().push(i);
    }

    // Sort levels by exposure (descending)
    let total_exposure: f64 = exposure.map_or(n as f64, |e| e.sum());
    let mut level_exposures: Vec<(&str, f64)> = level_indices
        .iter()
        .map(|(&level, indices)| {
            let exp: f64 = indices
                .iter()
                .map(|&i| exposure.map_or(1.0, |e| e[i]))
                .sum();
            (level, exp)
        })
        .collect();
    level_exposures.sort_by(|a, b| b.1.total_cmp(&a.1));

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

/// Code-based A/E computation for a categorical factor.
///
/// Accepts u32 codes (one per row, indexing into `levels`) plus the sorted
/// level labels. This avoids per-row string hashing that dominates the
/// `compute_ae_categorical` wall-clock for large n, because we can group with
/// a simple `Vec<Vec<usize>>` indexed by code instead of a
/// `HashMap<&str, Vec<usize>>` built via string hashing.
pub fn compute_ae_categorical_from_codes(
    codes: &[u32],
    levels: &[String],
    y: &Array1<f64>,
    mu: &Array1<f64>,
    exposure: Option<&Array1<f64>>,
    family: &str,
    var_power: Option<f64>,
    theta: Option<f64>,
    rare_threshold_pct: f64,
    max_levels: usize,
) -> Vec<ActualExpectedBin> {
    let n = codes.len();
    if n == 0 || n != y.len() {
        return Vec::new();
    }

    let k = levels.len();
    if k == 0 {
        return Vec::new();
    }

    // Group row indices by code. A plain Vec<Vec<usize>> indexed by code is
    // O(n) with a single allocation per level; no string hashing is needed.
    let mut level_indices: Vec<Vec<usize>> = vec![Vec::new(); k];
    for (i, &c) in codes.iter().enumerate() {
        let c = c as usize;
        if c < k {
            level_indices[c].push(i);
        }
    }

    // Sort levels by exposure (descending). Preserve the same contract as
    // `compute_ae_categorical`: levels below `rare_threshold_pct` or beyond
    // `max_levels - 1` get merged into an "_Other" bucket.
    let total_exposure: f64 = exposure.map_or(n as f64, |e| e.sum());
    let mut level_exposures: Vec<(usize, f64)> = (0..k)
        .map(|code| {
            let indices = &level_indices[code];
            let exp: f64 = indices
                .iter()
                .map(|&i| exposure.map_or(1.0, |e| e[i]))
                .sum();
            (code, exp)
        })
        // Skip empty levels so they don't consume a "max_levels" slot.
        .filter(|(code, _)| !level_indices[*code].is_empty())
        .collect();
    level_exposures.sort_by(|a, b| b.1.total_cmp(&a.1));

    let mut bins = Vec::new();
    let mut other_indices = Vec::new();

    for (bin_idx, &(code, exp)) in level_exposures.iter().enumerate() {
        let pct = 100.0 * exp / total_exposure;
        let indices = &level_indices[code];

        if pct < rare_threshold_pct || bin_idx >= max_levels - 1 {
            other_indices.extend(indices.iter().cloned());
        } else {
            bins.push(compute_ae_bin(
                indices,
                bin_idx,
                levels[code].clone(),
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

/// Compute A/E for many categorical factors in parallel.
///
/// Each factor supplies its own `codes` (length n) and `levels` labels.
/// Shared `y`, `mu`, `exposure`, `family`, `rare_threshold_pct`, `max_levels`
/// are reused. Per-factor work (grouping by code, per-bin stats) is
/// independent across factors, so we use rayon to parallelize across the
/// k factors. Mirrors the design of `compute_ae_continuous_batch`.
pub fn compute_ae_categorical_batch(
    codes_list: &[&[u32]],
    levels_list: &[&[String]],
    y: &Array1<f64>,
    mu: &Array1<f64>,
    exposure: Option<&Array1<f64>>,
    family: &str,
    var_power: Option<f64>,
    theta: Option<f64>,
    rare_threshold_pct: f64,
    max_levels: usize,
) -> Vec<Vec<ActualExpectedBin>> {
    codes_list
        .par_iter()
        .zip(levels_list.par_iter())
        .map(|(codes, levels)| {
            compute_ae_categorical_from_codes(
                codes,
                levels,
                y,
                mu,
                exposure,
                family,
                var_power,
                theta,
                rare_threshold_pct,
                max_levels,
            )
        })
        .collect()
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
    let loss = compute_family_loss(family, &y_arr, &mu_arr, Some(&w_arr), var_power, theta)
        .unwrap_or(f64::NAN);

    // Confidence interval for A/E using the Wilson score interval for a
    // Poisson rate. This is preferred over a Wald approximation in
    // actuarial / insurance settings: it stays inside [0, ∞), behaves well
    // for small counts, and does not need the historical `actual_sum.max(1.0)`
    // fudge that masked the true k=0 behavior.
    let (ae_ci_lower, ae_ci_upper) = if predicted_sum > 0.0 && actual_sum >= 0.0 {
        wilson_poisson_rate_ci(actual_sum, predicted_sum, DEFAULT_CI_ALPHA)
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
    let valid_pairs: Vec<(f64, f64)> = factor_values
        .iter()
        .zip(residuals.iter())
        .filter(|(&f, _)| !f.is_nan() && !f.is_infinite())
        .map(|(&f, &r)| (f, r))
        .collect();

    let correlation = compute_correlation(&valid_pairs);

    // Compute mean residual by bin
    let mut sorted_pairs = valid_pairs.clone();
    sorted_pairs.sort_by(|a, b| a.0.total_cmp(&b.0));

    let bin_size = sorted_pairs.len().div_ceil(n_bins);
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

/// Compute residual patterns for many continuous factors in parallel.
///
/// Each entry in `factor_values_list` is one factor's per-row values; the
/// shared `residuals` array is reused across all factors. Per-factor work
/// (filtering, sorting for binning, correlation, linear trend) is independent
/// across factors, so we parallelize across the k factors with rayon.
pub fn compute_residual_pattern_continuous_batch(
    factor_values_list: &[&[f64]],
    residuals: &Array1<f64>,
    n_bins: usize,
) -> Vec<ResidualPattern> {
    factor_values_list
        .par_iter()
        .map(|values| compute_residual_pattern_continuous(values, residuals, n_bins))
        .collect()
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
        level_residuals
            .entry(level.as_str())
            .or_default()
            .push(residuals[i]);
    }

    // Compute mean residual by level
    let mut level_means: Vec<(&str, f64, usize)> = level_residuals
        .iter()
        .map(|(&level, resids)| {
            let mean = resids.iter().sum::<f64>() / resids.len() as f64;
            (level, mean, resids.len())
        })
        .collect();
    level_means.sort_by(|a, b| b.2.cmp(&a.2)); // Sort by count

    let mean_residual_by_bin: Vec<f64> = level_means.iter().map(|&(_, mean, _)| mean).collect();

    // Compute variance explained (eta-squared)
    let overall_mean: f64 = residuals.sum() / n as f64;
    let ss_total: f64 = residuals.iter().map(|&r| (r - overall_mean).powi(2)).sum();

    let ss_between: f64 = level_means
        .iter()
        .map(|&(_, level_mean, count)| count as f64 * (level_mean - overall_mean).powi(2))
        .sum();

    let eta_squared = if ss_total > 0.0 {
        ss_between / ss_total
    } else {
        0.0
    };

    // Mean absolute residual correlation (approximation)
    let mean_abs_resid: f64 = mean_residual_by_bin.iter().map(|&m| m.abs()).sum::<f64>()
        / mean_residual_by_bin.len().max(1) as f64;

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
    let ss_res: f64 = pairs
        .iter()
        .map(|&(x, y)| {
            let pred = mean_y + slope * (x - mean_x);
            (y - pred).powi(2)
        })
        .sum();

    let df = n - 2;
    let mse = ss_res / df as f64;
    let se_slope = (mse / ss_xx).sqrt();

    let t_stat = if se_slope > 0.0 {
        slope / se_slope
    } else {
        0.0
    };

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
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

fn incomplete_beta_approx(x: f64, a: f64, b: f64) -> f64 {
    // Simple approximation - for accurate values, use a proper library
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }

    // Continued fraction approximation (first few terms)
    let mut result = x.powf(a) * (1.0 - x).powf(b) / a;
    result *= (a + b) / (a + 1.0);
    result.clamp(0.0, 1.0)
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
    let unit_deviances: Vec<f64> = y
        .iter()
        .zip(mu.iter())
        .map(|(&yi, &mui)| unit_deviance_for_family(yi, mui, family, var_power, theta))
        .collect();

    let total_deviance: f64 = unit_deviances.iter().sum();
    let mean_unit_deviance = total_deviance / n as f64;

    // Group by level using HashMap for O(n) complexity
    let mut level_data: HashMap<&str, (usize, f64, f64, f64)> = HashMap::new();

    for (i, level) in factor_values.iter().enumerate() {
        let entry = level_data
            .entry(level.as_str())
            .or_insert((0, 0.0, 0.0, 0.0));
        entry.0 += 1; // count
        entry.1 += unit_deviances[i]; // deviance sum
        entry.2 += y[i]; // actual sum
        entry.3 += mu[i]; // predicted sum
    }

    // Build results
    let mut levels: Vec<DevianceByLevel> = Vec::with_capacity(level_data.len());
    let mut problem_levels: Vec<String> = Vec::new();

    for (level, (count, deviance, actual, predicted)) in level_data {
        let deviance_pct = if total_deviance > 0.0 {
            100.0 * deviance / total_deviance
        } else {
            0.0
        };
        let mean_deviance = if count > 0 {
            deviance / count as f64
        } else {
            0.0
        };
        let ae_ratio = if predicted > 0.0 {
            actual / predicted
        } else {
            f64::NAN
        };

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
    levels.sort_by(|a, b| b.deviance.total_cmp(&a.deviance));

    FactorDevianceResult {
        factor_name: factor_name.to_string(),
        total_deviance,
        levels,
        problem_levels,
    }
}

/// Batch: compute factor deviance for multiple categorical factors in parallel.
///
/// Each entry in `factor_values_list` is one factor's per-row string values
/// (length n). Shared `y`, `mu`, `family`, `var_power`, and `theta` are reused
/// across all factors.
///
/// Optimization: unit deviances are computed ONCE for the shared (y, mu, family)
/// and then reused across all factors, instead of being recomputed inside each
/// per-factor task. The remaining per-factor work (HashMap groupby + result
/// assembly) is independent across factors and runs in parallel via rayon.
/// Mirrors the design of `compute_ae_continuous_batch`.
pub fn compute_factor_deviance_batch(
    factor_names: &[String],
    factor_values_list: &[&[String]],
    y: &Array1<f64>,
    mu: &Array1<f64>,
    family: &str,
    var_power: f64,
    theta: f64,
) -> Vec<FactorDevianceResult> {
    let n = y.len();
    if n == 0 || n != mu.len() {
        // Fall back to per-factor handling so each result mirrors the
        // singular function's empty-input contract.
        return factor_names
            .iter()
            .zip(factor_values_list.iter())
            .map(|(name, values)| {
                compute_factor_deviance(name, values, y, mu, family, var_power, theta)
            })
            .collect();
    }

    // Compute unit deviances ONCE — shared across all factors.
    let unit_deviances: Vec<f64> = y
        .iter()
        .zip(mu.iter())
        .map(|(&yi, &mui)| unit_deviance_for_family(yi, mui, family, var_power, theta))
        .collect();
    let total_deviance: f64 = unit_deviances.iter().sum();
    let mean_unit_deviance = total_deviance / n as f64;

    factor_names
        .par_iter()
        .zip(factor_values_list.par_iter())
        .map(|(name, values)| {
            compute_factor_deviance_with_precomputed(
                name,
                values,
                y,
                mu,
                &unit_deviances,
                total_deviance,
                mean_unit_deviance,
            )
        })
        .collect()
}

/// Per-factor groupby + result assembly given pre-computed unit deviances.
///
/// Mirrors the post-unit-deviance section of `compute_factor_deviance` exactly:
/// same HashMap groupby, same problem-detection rule, same descending-deviance
/// sort order. Used by `compute_factor_deviance_batch` to share unit-deviance
/// work across factors.
fn compute_factor_deviance_with_precomputed(
    factor_name: &str,
    factor_values: &[String],
    y: &Array1<f64>,
    mu: &Array1<f64>,
    unit_deviances: &[f64],
    total_deviance: f64,
    mean_unit_deviance: f64,
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

    // Group by level using HashMap for O(n) complexity.
    let mut level_data: HashMap<&str, (usize, f64, f64, f64)> = HashMap::new();
    for (i, level) in factor_values.iter().enumerate() {
        let entry = level_data
            .entry(level.as_str())
            .or_insert((0, 0.0, 0.0, 0.0));
        entry.0 += 1;
        entry.1 += unit_deviances[i];
        entry.2 += y[i];
        entry.3 += mu[i];
    }

    let mut levels: Vec<DevianceByLevel> = Vec::with_capacity(level_data.len());
    let mut problem_levels: Vec<String> = Vec::new();

    for (level, (count, deviance, actual, predicted)) in level_data {
        let deviance_pct = if total_deviance > 0.0 {
            100.0 * deviance / total_deviance
        } else {
            0.0
        };
        let mean_deviance = if count > 0 {
            deviance / count as f64
        } else {
            0.0
        };
        let ae_ratio = if predicted > 0.0 {
            actual / predicted
        } else {
            f64::NAN
        };

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

    levels.sort_by(|a, b| b.deviance.total_cmp(&a.deviance));

    FactorDevianceResult {
        factor_name: factor_name.to_string(),
        total_deviance,
        levels,
        problem_levels,
    }
}

/// Code-based per-factor groupby + result assembly with pre-computed unit deviances.
///
/// Equivalent to `compute_factor_deviance_with_precomputed` but groups by u32
/// `codes` (indexing into `levels`) instead of by string. This avoids the
/// per-row string hashing that dominates the wall-clock for large n; instead
/// we use a `Vec<(usize, f64, f64, f64)>` indexed by code (one allocation per
/// level), an O(n) pass with no hashing.
fn compute_factor_deviance_from_codes_with_precomputed(
    factor_name: &str,
    codes: &[u32],
    levels: &[String],
    y: &Array1<f64>,
    mu: &Array1<f64>,
    unit_deviances: &[f64],
    total_deviance: f64,
    mean_unit_deviance: f64,
) -> FactorDevianceResult {
    let n = codes.len();
    if n == 0 || n != y.len() || n != mu.len() {
        return FactorDevianceResult {
            factor_name: factor_name.to_string(),
            total_deviance: 0.0,
            levels: Vec::new(),
            problem_levels: Vec::new(),
        };
    }

    let k = levels.len();
    if k == 0 {
        return FactorDevianceResult {
            factor_name: factor_name.to_string(),
            total_deviance,
            levels: Vec::new(),
            problem_levels: Vec::new(),
        };
    }

    // (count, deviance_sum, actual_sum, predicted_sum) per level code.
    let mut acc: Vec<(usize, f64, f64, f64)> = vec![(0, 0.0, 0.0, 0.0); k];
    for (i, &c) in codes.iter().enumerate() {
        let c = c as usize;
        if c < k {
            let entry = &mut acc[c];
            entry.0 += 1;
            entry.1 += unit_deviances[i];
            entry.2 += y[i];
            entry.3 += mu[i];
        }
    }

    let mut out_levels: Vec<DevianceByLevel> = Vec::with_capacity(k);
    let mut problem_levels: Vec<String> = Vec::new();

    for (code, &(count, deviance, actual, predicted)) in acc.iter().enumerate() {
        if count == 0 {
            // Skip levels that have no rows so the result mirrors the
            // string-keyed HashMap path (which only emits seen levels).
            continue;
        }

        let deviance_pct = if total_deviance > 0.0 {
            100.0 * deviance / total_deviance
        } else {
            0.0
        };
        let mean_deviance = deviance / count as f64;
        let ae_ratio = if predicted > 0.0 {
            actual / predicted
        } else {
            f64::NAN
        };

        let expected_pct = 100.0 * count as f64 / n as f64;
        let is_problem = mean_deviance > mean_unit_deviance * 1.5
            || (ae_ratio - 1.0).abs() > 0.15
            || deviance_pct > expected_pct * 2.0;

        if is_problem {
            problem_levels.push(levels[code].clone());
        }

        out_levels.push(DevianceByLevel {
            level: levels[code].clone(),
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

    out_levels.sort_by(|a, b| b.deviance.total_cmp(&a.deviance));

    FactorDevianceResult {
        factor_name: factor_name.to_string(),
        total_deviance,
        levels: out_levels,
        problem_levels,
    }
}

/// Code-based batch: compute factor deviance for multiple categorical factors
/// in parallel, given each factor's u32 codes + level labels.
///
/// Same shape as `compute_factor_deviance_batch` but consumes pre-factorized
/// codes (matching the contract used by `compute_ae_categorical_batch` /
/// OPT-8). Unit deviances are computed once for shared (y, mu, family).
pub fn compute_factor_deviance_batch_from_codes(
    factor_names: &[String],
    codes_list: &[&[u32]],
    levels_list: &[&[String]],
    y: &Array1<f64>,
    mu: &Array1<f64>,
    family: &str,
    var_power: f64,
    theta: f64,
) -> Vec<FactorDevianceResult> {
    let n = y.len();
    if n == 0 || n != mu.len() {
        return factor_names
            .iter()
            .map(|name| FactorDevianceResult {
                factor_name: name.to_string(),
                total_deviance: 0.0,
                levels: Vec::new(),
                problem_levels: Vec::new(),
            })
            .collect();
    }

    let unit_deviances: Vec<f64> = y
        .iter()
        .zip(mu.iter())
        .map(|(&yi, &mui)| unit_deviance_for_family(yi, mui, family, var_power, theta))
        .collect();
    let total_deviance: f64 = unit_deviances.iter().sum();
    let mean_unit_deviance = total_deviance / n as f64;

    factor_names
        .par_iter()
        .zip(codes_list.par_iter())
        .zip(levels_list.par_iter())
        .map(|((name, codes), levels)| {
            compute_factor_deviance_from_codes_with_precomputed(
                name,
                codes,
                levels,
                y,
                mu,
                &unit_deviances,
                total_deviance,
                mean_unit_deviance,
            )
        })
        .collect()
}

/// Compute the textbook GLM total deviance over all observations.
///
/// This is the canonical `D = sum_i d_i` where `d_i = unit_deviance(y_i, mu_i; family)`,
/// matching `families::Family::deviance` and statsmodels' `GLMResults.deviance`.
///
/// Use this (rather than `mean(family_loss) * n`) whenever you need the textbook
/// GLM deviance — Binomial in particular: `family_loss` for Binomial is NLL, not
/// unit deviance, so the two differ by an additive constant per observation.
///
/// # Arguments
/// * `y` - Observed responses.
/// * `mu` - Fitted means.
/// * `family` - Family name ("gaussian", "poisson", "binomial", "gamma", "tweedie",
///   "negativebinomial", or quasi-variants).
/// * `var_power` - Tweedie variance power (ignored for non-Tweedie families).
/// * `theta` - Negative binomial dispersion (ignored for non-NB families).
pub fn compute_glm_deviance(
    y: &Array1<f64>,
    mu: &Array1<f64>,
    family: &str,
    var_power: f64,
    theta: f64,
) -> f64 {
    y.iter()
        .zip(mu.iter())
        .map(|(&yi, &mui)| unit_deviance_for_family(yi, mui, family, var_power, theta))
        .sum()
}

/// Compute unit deviance for a single observation
pub(crate) fn unit_deviance_for_family(
    y: f64,
    mu: f64,
    family: &str,
    var_power: f64,
    theta: f64,
) -> f64 {
    let lower = family.to_lowercase();
    let mu_safe = mu.max(MU_MIN_POSITIVE);
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
            let y_clamp = y.clamp(MU_MIN_PROBABILITY, MU_MAX_PROBABILITY);
            let mu_clamp = mu.clamp(MU_MIN_PROBABILITY, MU_MAX_PROBABILITY);
            2.0 * (y_clamp * (y_clamp / mu_clamp).ln()
                + (1.0 - y_clamp) * ((1.0 - y_clamp) / (1.0 - mu_clamp)).ln())
        }
        "gamma" => 2.0 * ((y_safe - mu_safe) / mu_safe - (y_safe / mu_safe).ln()),
        "tweedie" => {
            // Tweedie deviance depends on var_power
            if (var_power - 1.0).abs() < TWEEDIE_VAR_POWER_TOL {
                // Quasi-Poisson
                2.0 * (y_safe * (y_safe / mu_safe).ln() - (y_safe - mu_safe))
            } else if (var_power - 2.0).abs() < TWEEDIE_VAR_POWER_TOL {
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
                2.0 * (y_safe * (y_safe / mu_safe).ln()
                    - (y_safe + theta) * ((y_safe + theta) / (mu_safe + theta)).ln())
            } else {
                2.0 * theta * ((theta) / (mu_safe + theta)).ln()
            }
        }
        _ => (y - mu).powi(2), // Default to Gaussian
    }
}

// =============================================================================
// Factor Significance (joint Wald test)
// =============================================================================

/// Raw result for one factor's significance test.
///
/// This is intentionally minimal so the Python wrapper can apply its own
/// rounding/formatting (matching the legacy `compute_factor_significance` shape).
#[derive(Clone, Debug)]
pub struct FactorSignificanceRaw {
    /// Wald chi-square statistic (β_S' @ Cov_SS⁻¹ @ β_S, or sum of z² when df=1).
    pub chi2: f64,
    /// Upper-tail p-value: 1 - F_χ²(chi2, df).
    pub pvalue: f64,
    /// Degrees of freedom = number of parameter indices for this factor.
    pub df: usize,
}

/// Batch-compute factor significance for k factors in parallel.
///
/// For each factor, identifies its parameter indices (provided as input via
/// `param_indices_per_factor[i]`), computes the joint Wald chi-square via the
/// bread matrix (or sum of z² for single-parameter factors), and returns the
/// chi-square + p-value.
///
/// Returns `None` for any factor whose parameter index list is empty (matches
/// the singular path's contract: such factors get no significance entry) or
/// whose covariance sub-block is not invertible.
///
/// `params` and `bse` are the full-length parameter and standard-error vectors
/// for the fitted model; `bread` is the full p×p bread matrix from which the
/// scale is inferred via `bse[i]² / bread[i,i]`.
pub fn compute_factor_significance_batch(
    param_indices_per_factor: &[Vec<usize>],
    params: &[f64],
    bse: &[f64],
    bread: &Array2<f64>,
) -> Vec<Option<FactorSignificanceRaw>> {
    param_indices_per_factor
        .par_iter()
        .map(|idx| compute_one_factor_significance(idx, params, bse, bread))
        .collect()
}

/// Compute significance for a single factor (used inside the rayon batch loop).
///
/// Pulled out for clarity and easier testing. Mirrors the singular Python
/// `compute_factor_significance` numerics exactly.
fn compute_one_factor_significance(
    idx: &[usize],
    params: &[f64],
    bse: &[f64],
    bread: &Array2<f64>,
) -> Option<FactorSignificanceRaw> {
    if idx.is_empty() {
        return None;
    }
    let df = idx.len();

    let chi2 = if df > 1 {
        // Joint Wald: β_S' @ Cov_SS⁻¹ @ β_S
        // Cov = scale * bread; infer scale from bse for first valid index
        // (matches the singular path: walk `idx` looking for the first entry
        // where both bread[i,i] > 0 and bse[i] > 0).
        let mut scale = 1.0;
        for &i in idx {
            if bread[[i, i]] > 0.0 && bse[i] > 0.0 {
                scale = (bse[i] * bse[i]) / bread[[i, i]];
                break;
            }
        }

        // Build the k×k covariance sub-block (scale * bread[idx, idx]).
        let k = df;
        let mut cov_sub = nalgebra::DMatrix::<f64>::zeros(k, k);
        for ii in 0..k {
            for jj in 0..k {
                cov_sub[(ii, jj)] = scale * bread[[idx[ii], idx[jj]]];
            }
        }

        // Try Cholesky first (covariance should be SPD); fall back to LU.
        let inv = match cov_sub.clone().cholesky() {
            Some(chol) => chol.inverse(),
            None => cov_sub.try_inverse()?,
        };

        // wald = beta_s.T @ inv @ beta_s
        let beta_s: Vec<f64> = idx.iter().map(|&i| params[i]).collect();
        let mut tmp = vec![0.0_f64; k];
        for i in 0..k {
            let mut s = 0.0;
            for j in 0..k {
                s += inv[(i, j)] * beta_s[j];
            }
            tmp[i] = s;
        }
        beta_s.iter().zip(tmp.iter()).map(|(a, b)| a * b).sum()
    } else {
        // Single param: chi2 = (beta/se)^2
        let i = idx[0];
        if bse[i] > 0.0 {
            let z = params[i] / bse[i];
            z * z
        } else {
            0.0
        }
    };

    let pvalue = 1.0 - chi2_cdf(chi2, df as f64);

    Some(FactorSignificanceRaw { chi2, pvalue, df })
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
    fn test_continuous_stats() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let stats = compute_continuous_stats(&values);

        assert_abs_diff_eq!(stats.mean, 5.5, epsilon = 1e-10);
        assert_abs_diff_eq!(stats.min, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(stats.max, 10.0, epsilon = 1e-10);
        assert_eq!(stats.missing_count, 0);
    }

    #[test]
    fn test_continuous_stats_with_nan() {
        let values = vec![1.0, f64::NAN, 3.0, f64::INFINITY, 5.0];
        let stats = compute_continuous_stats(&values);

        assert_abs_diff_eq!(stats.mean, 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(stats.min, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(stats.max, 5.0, epsilon = 1e-10);
        assert_eq!(stats.missing_count, 2); // NaN and Infinity
    }

    #[test]
    fn test_continuous_stats_empty() {
        let values: Vec<f64> = vec![];
        let stats = compute_continuous_stats(&values);

        assert!(stats.mean.is_nan());
        assert!(stats.min.is_nan());
        assert!(stats.max.is_nan());
        assert_eq!(stats.missing_count, 0);
    }

    #[test]
    fn test_continuous_stats_all_nan() {
        let values = vec![f64::NAN, f64::NAN];
        let stats = compute_continuous_stats(&values);

        assert!(stats.mean.is_nan());
        assert_eq!(stats.missing_count, 2);
    }

    #[test]
    fn test_continuous_stats_percentiles() {
        let values: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let stats = compute_continuous_stats(&values);

        assert_abs_diff_eq!(stats.percentiles.p50, 50.0, epsilon = 1.0);
        assert!(stats.percentiles.p1 <= stats.percentiles.p5);
        assert!(stats.percentiles.p5 <= stats.percentiles.p25);
        assert!(stats.percentiles.p25 <= stats.percentiles.p50);
        assert!(stats.percentiles.p50 <= stats.percentiles.p75);
        assert!(stats.percentiles.p75 <= stats.percentiles.p95);
        assert!(stats.percentiles.p95 <= stats.percentiles.p99);
    }

    #[test]
    fn test_categorical_distribution() {
        let values = vec![
            "A".to_string(),
            "A".to_string(),
            "A".to_string(),
            "B".to_string(),
            "B".to_string(),
            "C".to_string(),
        ];

        let dist = compute_categorical_distribution(&values, 10.0);

        assert_eq!(dist.n_levels, 3);
        assert_eq!(dist.levels[0].level, "A");
        assert_eq!(dist.levels[0].count, 3);
    }

    #[test]
    fn test_categorical_distribution_empty() {
        let values: Vec<String> = vec![];

        let dist = compute_categorical_distribution(&values, 10.0);

        assert_eq!(dist.n_levels, 0);
        assert_eq!(dist.levels.len(), 0);
        assert_eq!(dist.n_rare_levels, 0);
    }

    #[test]
    fn test_categorical_distribution_rare_levels() {
        let values = vec![
            "A".to_string(),
            "A".to_string(),
            "A".to_string(),
            "A".to_string(),
            "A".to_string(),
            "A".to_string(),
            "A".to_string(),
            "A".to_string(),
            "A".to_string(),
            "A".to_string(),
            "B".to_string(), // 10% - rare
            "C".to_string(), // 10% - rare
        ];

        // Levels with < 15% are rare
        let dist = compute_categorical_distribution(&values, 15.0);

        assert_eq!(dist.n_rare_levels, 2); // B and C
        assert!(dist.rare_level_total_pct > 15.0);
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
    fn test_ae_continuous_empty() {
        let factor: Vec<f64> = vec![];
        let y = array![];
        let mu = array![];

        let bins = compute_ae_continuous(&factor, &y, &mu, None, "gaussian", 5, None, None);

        assert_eq!(bins.len(), 0);
    }

    #[test]
    fn test_ae_continuous_with_exposure() {
        let factor = vec![1.0, 2.0, 3.0, 4.0];
        let y = array![1.0, 2.0, 3.0, 4.0];
        let mu = array![1.0, 2.0, 3.0, 4.0];
        let exposure = array![1.0, 2.0, 1.0, 2.0];

        let bins =
            compute_ae_continuous(&factor, &y, &mu, Some(&exposure), "poisson", 2, None, None);

        assert_eq!(bins.len(), 2);
        for bin in &bins {
            assert!(bin.exposure > 0.0);
        }
    }

    #[test]
    fn test_ae_continuous_with_nan() {
        let factor = vec![1.0, f64::NAN, 3.0, 4.0];
        let y = array![1.0, 2.0, 3.0, 4.0];
        let mu = array![1.0, 2.0, 3.0, 4.0];

        let bins = compute_ae_continuous(&factor, &y, &mu, None, "gaussian", 2, None, None);

        // Should handle NaN gracefully
        assert!(bins.len() <= 2);
    }

    #[test]
    fn test_ae_categorical() {
        let factor = vec![
            "A".to_string(),
            "A".to_string(),
            "B".to_string(),
            "B".to_string(),
        ];
        let y = array![1.0, 2.0, 3.0, 4.0];
        let mu = array![1.0, 2.0, 3.0, 4.0];

        let bins = compute_ae_categorical(&factor, &y, &mu, None, "gaussian", None, None, 5.0, 10);

        assert_eq!(bins.len(), 2); // A and B
        for bin in &bins {
            assert!((bin.actual_expected_ratio - 1.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_ae_categorical_empty() {
        let factor: Vec<String> = vec![];
        let y = array![];
        let mu = array![];

        let bins = compute_ae_categorical(&factor, &y, &mu, None, "gaussian", None, None, 5.0, 10);

        assert_eq!(bins.len(), 0);
    }

    #[test]
    fn test_ae_categorical_with_other() {
        // Create data where some levels are rare
        let mut factor = Vec::new();
        for _ in 0..90 {
            factor.push("A".to_string());
        }
        for _ in 0..5 {
            factor.push("B".to_string());
        }
        for _ in 0..3 {
            factor.push("C".to_string());
        }
        for _ in 0..2 {
            factor.push("D".to_string());
        }

        let n = factor.len();
        let y = Array1::from_vec(vec![1.0; n]);
        let mu = Array1::from_vec(vec![1.0; n]);

        // Rare threshold 5%, max 3 levels
        let bins = compute_ae_categorical(&factor, &y, &mu, None, "gaussian", None, None, 5.0, 3);

        // Should have A, B, and "_Other" (C+D grouped)
        assert!(bins.len() <= 3);
        let has_other = bins.iter().any(|b| b.bin_label == "_Other");
        assert!(has_other);
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

    #[test]
    fn test_residual_correlation_negative() {
        // Perfect negative correlation
        let pairs = vec![(1.0, 4.0), (2.0, 3.0), (3.0, 2.0), (4.0, 1.0)];
        let corr = compute_correlation(&pairs);
        assert_abs_diff_eq!(corr, -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_residual_correlation_insufficient_data() {
        let pairs = vec![(1.0, 1.0)];
        let corr = compute_correlation(&pairs);
        assert!(corr.is_nan());

        let empty: Vec<(f64, f64)> = vec![];
        let corr = compute_correlation(&empty);
        assert!(corr.is_nan());
    }

    #[test]
    fn test_residual_correlation_zero_variance() {
        let pairs = vec![(1.0, 1.0), (1.0, 2.0), (1.0, 3.0)];
        let corr = compute_correlation(&pairs);
        assert_abs_diff_eq!(corr, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_residual_pattern_continuous() {
        let factor = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let residuals = array![0.1, 0.2, 0.3, 0.4, 0.5];

        let pattern = compute_residual_pattern_continuous(&factor, &residuals, 3);

        assert!((pattern.correlation_with_residuals - 1.0).abs() < 0.01);
        assert_eq!(pattern.mean_residual_by_bin.len(), 3);
        assert!(pattern.trend_slope > 0.0);
    }

    #[test]
    fn test_residual_pattern_continuous_empty() {
        let factor: Vec<f64> = vec![];
        let residuals = array![];

        let pattern = compute_residual_pattern_continuous(&factor, &residuals, 3);

        assert!(pattern.correlation_with_residuals.is_nan());
        assert_eq!(pattern.mean_residual_by_bin.len(), 0);
    }

    #[test]
    fn test_residual_pattern_categorical() {
        let factor = vec![
            "A".to_string(),
            "A".to_string(),
            "B".to_string(),
            "B".to_string(),
        ];
        let residuals = array![0.1, 0.2, 0.3, 0.4];

        let pattern = compute_residual_pattern_categorical(&factor, &residuals);

        assert_eq!(pattern.mean_residual_by_bin.len(), 2);
        assert!(pattern.residual_variance_explained >= 0.0);
        assert!(pattern.residual_variance_explained <= 1.0);
    }

    #[test]
    fn test_residual_pattern_categorical_empty() {
        let factor: Vec<String> = vec![];
        let residuals = array![];

        let pattern = compute_residual_pattern_categorical(&factor, &residuals);

        assert!(pattern.correlation_with_residuals.is_nan());
    }

    #[test]
    fn test_linear_trend() {
        // Strong but not perfect linear trend
        let pairs = vec![
            (1.0, 1.1),
            (2.0, 1.9),
            (3.0, 3.1),
            (4.0, 3.9),
            (5.0, 5.1),
            (6.0, 5.9),
        ];
        let (slope, pvalue) = compute_linear_trend(&pairs);

        // Slope should be close to 1
        assert!((slope - 1.0).abs() < 0.1);
        // P-value should be finite and indicate significance
        assert!(pvalue.is_finite());
    }

    #[test]
    fn test_linear_trend_insufficient_data() {
        let pairs = vec![(1.0, 1.0), (2.0, 2.0)];
        let (slope, pvalue) = compute_linear_trend(&pairs);

        assert!(slope.is_nan());
        assert!(pvalue.is_nan());
    }

    #[test]
    fn test_compute_factor_deviance() {
        let factor = vec![
            "A".to_string(),
            "A".to_string(),
            "B".to_string(),
            "B".to_string(),
        ];
        let y = array![1.0, 2.0, 3.0, 4.0];
        let mu = array![1.0, 2.0, 3.0, 4.0];

        let result = compute_factor_deviance("test", &factor, &y, &mu, "gaussian", 1.5, 1.0);

        assert_eq!(result.factor_name, "test");
        assert_eq!(result.levels.len(), 2);
        assert_abs_diff_eq!(result.total_deviance, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_compute_factor_deviance_poisson() {
        let factor = vec!["A".to_string(), "B".to_string()];
        let y = array![1.0, 5.0];
        let mu = array![2.0, 3.0];

        let result = compute_factor_deviance("count", &factor, &y, &mu, "poisson", 1.5, 1.0);

        assert!(result.total_deviance > 0.0);
        assert_eq!(result.levels.len(), 2);
    }

    #[test]
    fn test_compute_factor_deviance_binomial() {
        let factor = vec!["A".to_string(), "B".to_string()];
        let y = array![0.0, 1.0];
        let mu = array![0.3, 0.7];

        let result = compute_factor_deviance("binary", &factor, &y, &mu, "binomial", 1.5, 1.0);

        assert!(result.total_deviance > 0.0);
    }

    #[test]
    fn test_compute_factor_deviance_gamma() {
        let factor = vec!["A".to_string(), "B".to_string()];
        let y = array![1.0, 2.0];
        let mu = array![1.5, 2.5];

        let result = compute_factor_deviance("amount", &factor, &y, &mu, "gamma", 1.5, 1.0);

        assert!(result.total_deviance > 0.0);
    }

    #[test]
    fn test_compute_factor_deviance_tweedie() {
        let factor = vec!["A".to_string(), "B".to_string()];
        let y = array![0.0, 5.0];
        let mu = array![1.0, 4.0];

        let result = compute_factor_deviance("claim", &factor, &y, &mu, "tweedie", 1.5, 1.0);

        assert!(result.total_deviance >= 0.0);
    }

    #[test]
    fn test_compute_factor_deviance_negbinomial() {
        let factor = vec!["A".to_string(), "B".to_string()];
        let y = array![0.0, 5.0];
        let mu = array![1.0, 4.0];

        let result =
            compute_factor_deviance("count", &factor, &y, &mu, "negativebinomial", 1.5, 2.0);

        assert!(result.total_deviance.is_finite());
    }

    #[test]
    fn test_compute_factor_deviance_empty() {
        let factor: Vec<String> = vec![];
        let y = array![];
        let mu = array![];

        let result = compute_factor_deviance("empty", &factor, &y, &mu, "gaussian", 1.5, 1.0);

        assert_eq!(result.levels.len(), 0);
        assert_eq!(result.total_deviance, 0.0);
    }

    #[test]
    fn test_compute_factor_deviance_problem_detection() {
        // Create data with a problematic level
        let factor = vec![
            "Good".to_string(),
            "Good".to_string(),
            "Good".to_string(),
            "Good".to_string(),
            "Bad".to_string(),
        ];
        let y = array![1.0, 1.0, 1.0, 1.0, 10.0]; // Bad level has outlier
        let mu = array![1.0, 1.0, 1.0, 1.0, 1.0];

        let result = compute_factor_deviance("problem", &factor, &y, &mu, "gaussian", 1.5, 1.0);

        // Bad level should be detected as problematic
        assert!(!result.problem_levels.is_empty() || result.levels.iter().any(|l| l.is_problem));
    }

    #[test]
    fn test_factor_type_enum() {
        let cont = FactorType::Continuous;
        let cat = FactorType::Categorical;

        assert_eq!(cont, FactorType::Continuous);
        assert_eq!(cat, FactorType::Categorical);
        assert_ne!(cont, cat);
    }

    #[test]
    fn test_factor_config() {
        let config = FactorConfig {
            name: "age".to_string(),
            factor_type: FactorType::Continuous,
            in_model: true,
            transformation: Some("bs(age, df=5)".to_string()),
        };

        assert_eq!(config.name, "age");
        assert_eq!(config.factor_type, FactorType::Continuous);
        assert!(config.in_model);
        assert!(config.transformation.is_some());
    }

    #[test]
    fn test_t_cdf_large_df() {
        // Large df should approximate normal
        let result = t_cdf(1.96, 100);
        assert!((result - 0.975).abs() < 0.01);
    }

    #[test]
    fn test_t_cdf_small_df() {
        let result = t_cdf(2.0, 5);
        assert!(result > 0.9);
        assert!(result < 1.0);
    }

    #[test]
    fn test_normal_cdf_approx() {
        assert!((normal_cdf_approx(0.0) - 0.5).abs() < 0.01);
        assert!((normal_cdf_approx(1.96) - 0.975).abs() < 0.01);
        assert!(normal_cdf_approx(-3.0) < 0.01);
        assert!(normal_cdf_approx(3.0) > 0.99);
    }

    #[test]
    fn test_erf_approx() {
        assert!((erf_approx(0.0) - 0.0).abs() < 0.001);
        assert!(erf_approx(1.0) > 0.8);
        assert!(erf_approx(-1.0) < -0.8);
    }

    #[test]
    fn test_factor_significance_batch_single_param() {
        // Single parameter case: chi2 should be (beta/se)^2.
        // params=[0.5, 1.0], bse=[0.1, 0.2], bread=identity, indices=[[1]].
        // Expected: chi2 = (1.0/0.2)^2 = 25.0, df=1, p ≈ very small.
        let params = vec![0.5, 1.0];
        let bse = vec![0.1, 0.2];
        let bread = ndarray::Array2::<f64>::eye(2);
        let indices = vec![vec![1_usize]];

        let results = compute_factor_significance_batch(&indices, &params, &bse, &bread);

        assert_eq!(results.len(), 1);
        let res = results[0]
            .as_ref()
            .expect("single-param case should yield Some");
        assert_eq!(res.df, 1);
        assert!(
            (res.chi2 - 25.0).abs() < 1e-6,
            "expected chi2 ≈ 25.0, got {}",
            res.chi2
        );
        // chi2_cdf(25, 1) is essentially 1.0, so survival p-value should be tiny.
        assert!(res.pvalue >= 0.0 && res.pvalue < 1e-3);
    }

    #[test]
    fn test_factor_significance_batch_multi_param() {
        // Multi-parameter case: chi2 = beta_S' * Cov_SS^-1 * beta_S.
        // bread = identity (2x2), bse = [1.0, 1.0] so scale = bse[0]^2 / bread[0,0] = 1.0.
        // Cov_sub = scale * I = I, so inv = I, and chi2 = 0.5^2 + 0.5^2 = 0.5 with df=2.
        let params = vec![0.5, 0.5];
        let bse = vec![1.0, 1.0];
        let bread = ndarray::Array2::<f64>::eye(2);
        let indices = vec![vec![0_usize, 1_usize]];

        let results = compute_factor_significance_batch(&indices, &params, &bse, &bread);

        assert_eq!(results.len(), 1);
        let res = results[0]
            .as_ref()
            .expect("multi-param case should yield Some");
        assert_eq!(res.df, 2);
        assert!(
            (res.chi2 - 0.5).abs() < 1e-6,
            "expected chi2 ≈ 0.5, got {}",
            res.chi2
        );
        // p-value should be valid in [0, 1].
        assert!(res.pvalue >= 0.0 && res.pvalue <= 1.0);
    }

    #[test]
    fn test_factor_significance_batch_empty_indices() {
        // Empty index vector for a factor → returns None.
        let params = vec![0.5, 1.0];
        let bse = vec![0.1, 0.2];
        let bread = ndarray::Array2::<f64>::eye(2);
        let indices: Vec<Vec<usize>> = vec![vec![]];

        let results = compute_factor_significance_batch(&indices, &params, &bse, &bread);

        assert_eq!(results.len(), 1);
        assert!(results[0].is_none(), "empty index list should yield None");
    }

    #[test]
    fn test_factor_significance_batch_singular_information() {
        // Pathological case: bread sub-block is singular (two identical rows).
        // bread = [[1, 1], [1, 1]] is rank-1; Cholesky fails AND try_inverse fails.
        // bse=[1, 1] gives scale = 1/1 = 1, so cov_sub = bread = singular.
        let params = vec![0.5, 0.5];
        let bse = vec![1.0, 1.0];
        let bread = ndarray::array![[1.0, 1.0], [1.0, 1.0]];
        let indices = vec![vec![0_usize, 1_usize]];

        let results = compute_factor_significance_batch(&indices, &params, &bse, &bread);

        assert_eq!(results.len(), 1);
        assert!(
            results[0].is_none(),
            "singular covariance sub-block should yield None"
        );
    }

    // ===== Wilson Poisson rate CI =====

    #[test]
    fn test_wilson_poisson_rate_ci_moderate() {
        // k=10, predicted=10 → point estimate 1.0.
        // center = (10 + z²/2)/10 = 1.19207, half_width = z·sqrt(10+z²/4)/10 ≈ 0.64887.
        // → CI ≈ (0.54320, 1.84095). Symmetric on the *shifted* center, so the
        // CI sits slightly above 1.0, matching the actuarial spec "roughly [0.5, 1.85]".
        let (lo, hi) = wilson_poisson_rate_ci(10.0, 10.0, DEFAULT_CI_ALPHA);
        assert!((lo - 0.54320).abs() < 1e-3, "lower {} not ≈ 0.543", lo);
        assert!((hi - 1.84095).abs() < 1e-3, "upper {} not ≈ 1.841", hi);
        assert!(lo < 1.0 && hi > 1.0);
    }

    #[test]
    fn test_wilson_poisson_rate_ci_zero_count() {
        // k=0, predicted=10 → lower clamped to 0, upper ≈ 0.384.
        // center = (0 + z²/2) / 10 = z²/20; half_width = z * (z²/4).sqrt() / 10 = z²/20.
        let (lo, hi) = wilson_poisson_rate_ci(0.0, 10.0, DEFAULT_CI_ALPHA);
        assert_eq!(lo, 0.0);
        assert!((hi - 0.38415).abs() < 1e-3, "upper {} not ≈ 0.384", hi);
    }

    #[test]
    fn test_wilson_poisson_rate_ci_large_n() {
        // k=100, predicted=100 → CI narrows around 1.0, ≈ (0.822, 1.216).
        // (Spec says "roughly (0.81, 1.21)" — the Wilson interval is shifted
        // upward relative to a symmetric Wald, so the lower end sits near 0.82.)
        let (lo, hi) = wilson_poisson_rate_ci(100.0, 100.0, DEFAULT_CI_ALPHA);
        assert!((lo - 0.82227).abs() < 1e-3, "lower {} not ≈ 0.822", lo);
        assert!((hi - 1.21614).abs() < 1e-3, "upper {} not ≈ 1.216", hi);
    }

    #[test]
    fn test_wilson_poisson_rate_ci_no_predicted() {
        // predicted=0 → degenerate: (0.0, NaN).
        let (lo, hi) = wilson_poisson_rate_ci(5.0, 0.0, DEFAULT_CI_ALPHA);
        assert_eq!(lo, 0.0);
        assert!(hi.is_nan());
    }

    #[test]
    fn test_wilson_poisson_rate_ci_lower_bound_nonnegative() {
        // Even for moderate k the lower bound must remain in [0, ∞).
        for k in [0.0, 0.5, 1.0, 3.0, 7.0, 50.0] {
            for predicted in [0.5, 1.0, 5.0, 50.0] {
                let (lo, hi) = wilson_poisson_rate_ci(k, predicted, DEFAULT_CI_ALPHA);
                assert!(
                    lo >= 0.0,
                    "lower {} negative for k={}, p={}",
                    lo,
                    k,
                    predicted
                );
                assert!(
                    hi >= lo,
                    "upper {} < lower {} for k={}, p={}",
                    hi,
                    lo,
                    k,
                    predicted
                );
            }
        }
    }

    #[test]
    fn test_ae_continuous_ci_uses_wilson() {
        // Ensure compute_ae_continuous now reports Wilson-style CIs (in
        // particular: lower bound ≥ 0 even when actual_sum is small, and the
        // CI for a perfect-fit moderately sized bin straddles 1.0).
        let factor: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        let y = Array1::from_vec(vec![1.0; 20]);
        let mu = Array1::from_vec(vec![1.0; 20]);

        let bins = compute_ae_continuous(&factor, &y, &mu, None, "poisson", 4, None, None);

        for bin in &bins {
            // 5 obs each => actual_sum=5, predicted_sum=5, A/E=1.0.
            // Wilson CI: lower ≈ 0.41, upper ≈ 2.41 (much wider than Wald with
            // se=sqrt(5)/5 ≈ 0.447 → Wald [0.124, 1.876]). Importantly, both
            // bounds are non-negative and contain the point estimate.
            assert!(bin.ae_ci_lower >= 0.0);
            assert!(bin.ae_ci_lower < bin.actual_expected_ratio);
            assert!(bin.ae_ci_upper > bin.actual_expected_ratio);
        }
    }
}
