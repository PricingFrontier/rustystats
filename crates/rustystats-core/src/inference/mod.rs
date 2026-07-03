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

use crate::constants::ZERO_TOL;

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

    let normal = Normal::new(0.0, 1.0).expect("standard normal");

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

    let normal = Normal::new(0.0, 1.0).expect("standard normal");

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

use crate::families::Family;

const SPARSE_SANDWICH_DENSITY_THRESHOLD: f64 = 0.35;

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
    let meat = compute_meat(
        x,
        pearson_resid,
        &combined_weights,
        &leverage,
        hc_type,
        n,
        p,
    );

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

    if let Some(x_slice) = x.as_slice() {
        if should_use_sparse_sandwich_kernel(x_slice, n, p) {
            return compute_leverage_sparse(x_slice, weights, &cov_flat, n, p);
        }
    }

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

fn compute_leverage_sparse(
    x_slice: &[f64],
    weights: &Array1<f64>,
    cov_flat: &[f64],
    n: usize,
    p: usize,
) -> Array1<f64> {
    let leverage_vec: Vec<f64> = (0..n)
        .into_par_iter()
        .map_init(
            || (Vec::with_capacity(p), Vec::with_capacity(p)),
            |(nz_idx, nz_val), i| {
                nz_idx.clear();
                nz_val.clear();
                let row_start = i * p;
                for j in 0..p {
                    // SAFETY: row_start + j < n*p = x_slice.len()
                    let xij = unsafe { *x_slice.get_unchecked(row_start + j) };
                    if xij != 0.0 {
                        nz_idx.push(j);
                        nz_val.push(xij);
                    }
                }

                let mut h_ii = 0.0;
                for a in 0..nz_idx.len() {
                    // SAFETY: a is within both nz vectors populated in lockstep above.
                    let j = unsafe { *nz_idx.get_unchecked(a) };
                    let xij = unsafe { *nz_val.get_unchecked(a) };
                    let mut temp_j = 0.0;
                    for b in 0..nz_idx.len() {
                        // SAFETY: b is within both nz vectors; j and k are column
                        // indices < p, so j*p + k < cov_flat.len().
                        let k = unsafe { *nz_idx.get_unchecked(b) };
                        let xik = unsafe { *nz_val.get_unchecked(b) };
                        // SAFETY: j and k are column indices < p, so this
                        // flattened covariance index is within cov_flat.
                        temp_j += unsafe { *cov_flat.get_unchecked(j * p + k) } * xik;
                    }
                    h_ii += xij * temp_j;
                }
                h_ii *= weights[i];
                h_ii.clamp(0.0, 0.9999)
            },
        )
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

    if let Some(x_slice) = x.as_slice() {
        if should_use_sparse_sandwich_kernel(x_slice, n, p) {
            return compute_meat_sparse(x_slice, &omega, n, p);
        }
    }

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

fn should_use_sparse_sandwich_kernel(x_slice: &[f64], n: usize, p: usize) -> bool {
    if n == 0 || p < 16 {
        return false;
    }

    let sample_rows = n.min(1024);
    let mut nonzero = 0usize;
    for sample_idx in 0..sample_rows {
        let row = sample_idx * n / sample_rows;
        let row_start = row * p;
        for j in 0..p {
            if x_slice[row_start + j] != 0.0 {
                nonzero += 1;
            }
        }
    }
    let density = nonzero as f64 / (sample_rows * p) as f64;
    density <= SPARSE_SANDWICH_DENSITY_THRESHOLD
}

fn compute_meat_sparse(x_slice: &[f64], omega: &Array1<f64>, n: usize, p: usize) -> Array2<f64> {
    let (meat_flat, _, _): (Vec<f64>, Vec<usize>, Vec<f64>) = (0..n)
        .into_par_iter()
        .fold(
            || {
                (
                    vec![0.0; p * p],
                    Vec::with_capacity(p),
                    Vec::with_capacity(p),
                )
            },
            |(mut acc, mut nz_idx, mut nz_val), i| {
                let omega_i = omega[i];
                if omega_i == 0.0 {
                    return (acc, nz_idx, nz_val);
                }

                nz_idx.clear();
                nz_val.clear();
                let row_start = i * p;
                for j in 0..p {
                    // SAFETY: row_start + j < n*p = x_slice.len()
                    let xij = unsafe { *x_slice.get_unchecked(row_start + j) };
                    if xij != 0.0 {
                        nz_idx.push(j);
                        nz_val.push(xij);
                    }
                }

                for a in 0..nz_idx.len() {
                    // SAFETY: a is within both nz vectors populated in lockstep above.
                    let j = unsafe { *nz_idx.get_unchecked(a) };
                    let xij_omega = unsafe { *nz_val.get_unchecked(a) } * omega_i;
                    for b in a..nz_idx.len() {
                        // SAFETY: b is within both nz vectors populated in lockstep above.
                        let k = unsafe { *nz_idx.get_unchecked(b) };
                        let xik = unsafe { *nz_val.get_unchecked(b) };
                        // SAFETY: j, k < p, so j*p + k < p*p.
                        unsafe { *acc.get_unchecked_mut(j * p + k) += xij_omega * xik };
                    }
                }
                (acc, nz_idx, nz_val)
            },
        )
        .reduce(
            || (vec![0.0; p * p], Vec::new(), Vec::new()),
            |(mut a, a_idx, a_val), (b, _b_idx, _b_val)| {
                for i in 0..a.len() {
                    a[i] += b[i];
                }
                (a, a_idx, a_val)
            },
        );

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
    (0..p).map(|i| robust_cov[[i, i]].max(0.0).sqrt()).collect()
}

// =============================================================================
// Rao's Score Test for Unfitted Factors
// =============================================================================
//
// The score test (Lagrange Multiplier test) evaluates whether adding a new
// variable to a model would significantly improve the fit, WITHOUT actually
// refitting the model.
//
// This is useful for:
// - Quickly screening candidate variables
// - Model selection and diagnostics
// - Testing if unfitted factors should be added
//
// FORMULA:
// --------
// For GLMs, the score statistic for adding variable Z to a model with X:
//
// 1. Compute residuals from the restricted model: r = (y - μ) / sqrt(V(μ))
// 2. Score contribution: U = Z' W r (where W = working weights)
// 3. Score information: I = Z' W Z - Z' W X (X'WX)^-1 X' W Z
// 4. Score statistic: S = U' I^-1 U ~ χ²(df)
//
// For a single continuous variable, df = 1.
// For a categorical with k levels, df = k - 1 (after excluding base).
//
// =============================================================================

/// Result of Rao's score test for an unfitted factor.
#[derive(Debug, Clone)]
pub struct ScoreTestResult {
    /// Score test statistic (chi-squared distributed)
    pub statistic: f64,
    /// Degrees of freedom
    pub df: usize,
    /// P-value from chi-squared distribution
    pub pvalue: f64,
    /// Whether the factor is significant at 0.05 level
    pub significant: bool,
}

/// Compute Rao's score test for adding a single continuous variable.
///
/// # Arguments
/// * `z` - The new variable to test (n × 1)
/// * `x` - Design matrix of the fitted model (n × p)
/// * `y` - Response variable (n)
/// * `mu` - Fitted values from the current model (n)
/// * `weights` - Working weights from IRLS (n)
/// * `bread` - (X'WX)^-1 matrix from the fitted model (p × p)
/// * `family` - Family name for variance function
///
/// # Returns
/// Score test result with statistic, df, and p-value
pub fn score_test_continuous(
    z: &Array1<f64>,
    x: &Array2<f64>,
    y: &Array1<f64>,
    mu: &Array1<f64>,
    weights: &Array1<f64>,
    bread: &Array2<f64>,
    _family: &dyn Family,
) -> ScoreTestResult {
    // The score test checks if adding variable z would improve the model.
    //
    // For GLMs, the IRLS weights passed in are: w_i = 1 / (V(μ_i) * g'(μ_i)²)
    // For Poisson with log link: w_i = μ_i
    // For Gaussian with identity: w_i = 1
    //
    // The score vector for the new variable is: U = Z'(y - μ)
    // The information matrix uses the same weights as (X'WX).
    //
    // Score statistic: S = U' I_zz^{-1} U ~ χ²(df)
    // where I_zz = Z'WZ - Z'WX (X'WX)^{-1} X'WZ

    // Score: U = Σ z_i (y_i - μ_i)
    let u: f64 = z
        .iter()
        .zip(y.iter())
        .zip(mu.iter())
        .map(|((&zi, &yi), &mui)| zi * (yi - mui))
        .sum();

    // Information: I_zz = Z'WZ - Z'WX (X'WX)^-1 X'WZ
    // where W = diag(weights) is the IRLS weight matrix

    // Z'WZ (scalar)
    let zwz: f64 = z
        .iter()
        .zip(weights.iter())
        .map(|(&zi, &wi)| zi * zi * wi)
        .sum();

    // Z'WX (1 × p vector)
    let p = x.ncols();
    let zwx: Array1<f64> = (0..p)
        .map(|j| {
            z.iter()
                .zip(weights.iter())
                .zip(x.column(j).iter())
                .map(|((&zi, &wi), &xij)| zi * wi * xij)
                .sum()
        })
        .collect();

    // (X'WX)^-1 X'WZ = bread × (Z'WX)^T = bread × zwx
    let bread_zwx: Array1<f64> = (0..p)
        .map(|i| (0..p).map(|j| bread[[i, j]] * zwx[j]).sum::<f64>())
        .collect();

    // Z'WX (X'WX)^-1 X'WZ = zwx · bread_zwx (dot product)
    let correction: f64 = zwx.iter().zip(bread_zwx.iter()).map(|(&a, &b)| a * b).sum();

    // Information for Z after adjusting for X
    let info = zwz - correction;

    // Score statistic: S = U² / I
    let statistic = if info > 1e-10 { u * u / info } else { 0.0 };

    // P-value from chi-squared with df=1
    let pvalue = 1.0 - chi2_cdf_internal(statistic, 1.0);

    ScoreTestResult {
        statistic,
        df: 1,
        pvalue,
        significant: pvalue < 0.05,
    }
}

/// Batched Rao score test for k continuous candidate variables.
///
/// Computes the same score statistic as `score_test_continuous` for each
/// column of `zs`, but reuses precomputed quantities across all k tests
/// and parallelizes across k via rayon.
///
/// # Arguments
/// * `zs` - Candidate variables stacked as columns (n × k)
/// * `x` - Design matrix of the fitted model (n × p)
/// * `y` - Response variable (n)
/// * `mu` - Fitted values (n)
/// * `weights` - IRLS working weights (n)
/// * `bread` - (X'WX)^{-1} (p × p)
/// * `_family` - kept for symmetry with the singular API
///
/// # Returns
/// `Vec<ScoreTestResult>` of length k, in column order of `zs`.
pub fn score_test_continuous_batch(
    zs: &Array2<f64>,
    x: &Array2<f64>,
    y: &Array1<f64>,
    mu: &Array1<f64>,
    weights: &Array1<f64>,
    bread: &Array2<f64>,
    _family: &dyn Family,
) -> Vec<ScoreTestResult> {
    let n = x.nrows();
    let p = x.ncols();
    let k = zs.ncols();
    debug_assert_eq!(zs.nrows(), n);
    debug_assert_eq!(y.len(), n);
    debug_assert_eq!(mu.len(), n);
    debug_assert_eq!(weights.len(), n);

    // Precompute residuals (y - mu) once — independent of z.
    let resid: Array1<f64> = y
        .iter()
        .zip(mu.iter())
        .map(|(&yi, &mui)| yi - mui)
        .collect();

    let x_slice = x.as_slice();

    // Process the k columns in parallel.
    (0..k)
        .into_par_iter()
        .map(|col| {
            let z = zs.column(col);

            // U, Z'WZ and Z'WX all share the same row scan. Fuse them to avoid
            // allocating WX (n x p) and touching X once before the batched tests.
            let mut u = 0.0_f64;
            let mut zwz = 0.0_f64;
            let mut zwx = vec![0.0_f64; p];
            if let Some(xs) = x_slice {
                for i in 0..n {
                    let zi = z[i];
                    let wi = weights[i];
                    u += zi * resid[i];
                    zwz += zi * zi * wi;
                    let zi_wi = zi * wi;
                    let row_start = i * p;
                    for j in 0..p {
                        // SAFETY: i < n and j < p, so row_start + j < n*p = xs.len().
                        zwx[j] += zi_wi * unsafe { *xs.get_unchecked(row_start + j) };
                    }
                }
            } else {
                for i in 0..n {
                    let zi = z[i];
                    let wi = weights[i];
                    u += zi * resid[i];
                    zwz += zi * zi * wi;
                    let zi_wi = zi * wi;
                    let row = x.row(i);
                    for j in 0..p {
                        zwx[j] += zi_wi * row[j];
                    }
                }
            }

            // bread @ zwx
            let mut bread_zwx = vec![0.0_f64; p];
            for i in 0..p {
                let row = bread.row(i);
                let mut s = 0.0;
                for j in 0..p {
                    s += row[j] * zwx[j];
                }
                bread_zwx[i] = s;
            }

            // correction = zwx · bread_zwx
            let correction: f64 = zwx.iter().zip(bread_zwx.iter()).map(|(&a, &b)| a * b).sum();
            let info = zwz - correction;
            let statistic = if info > ZERO_TOL { u * u / info } else { 0.0 };
            let pvalue = 1.0 - chi2_cdf_internal(statistic, 1.0);
            ScoreTestResult {
                statistic,
                df: 1,
                pvalue,
                significant: pvalue < 0.05,
            }
        })
        .collect()
}

/// Compute Rao's score test for adding a categorical variable.
///
/// # Arguments
/// * `z_matrix` - Dummy-coded matrix for the categorical (n × (k-1))
/// * `x` - Design matrix of the fitted model (n × p)
/// * `y` - Response variable (n)
/// * `mu` - Fitted values from the current model (n)
/// * `weights` - Working weights from IRLS (n)
/// * `bread` - (X'WX)^-1 matrix from the fitted model (p × p)
/// * `family` - Family name for variance function
///
/// # Returns
/// Score test result with statistic, df (= k-1), and p-value
pub fn score_test_categorical(
    z_matrix: &Array2<f64>,
    x: &Array2<f64>,
    y: &Array1<f64>,
    mu: &Array1<f64>,
    weights: &Array1<f64>,
    bread: &Array2<f64>,
    family: &dyn Family,
) -> ScoreTestResult {
    let _n = z_matrix.nrows();
    let k = z_matrix.ncols(); // df = k (number of dummy columns)
    let p = x.ncols();

    if k == 0 {
        return ScoreTestResult {
            statistic: 0.0,
            df: 0,
            pvalue: 1.0,
            significant: false,
        };
    }

    // Compute variance function values using the Family trait
    let variance = family.variance(mu);

    // Working weights: w_i * V(μ_i)
    let w: Array1<f64> = weights
        .iter()
        .zip(variance.iter())
        .map(|(&wi, &vi)| wi * vi)
        .collect();

    // Pearson residuals scaled by sqrt(weight)
    let weighted_resid: Array1<f64> = y
        .iter()
        .zip(mu.iter())
        .zip(weights.iter())
        .zip(variance.iter())
        .map(|(((&yi, &mui), &wi), &vi)| {
            if vi > 1e-10 {
                wi * (yi - mui) / vi.sqrt()
            } else {
                0.0
            }
        })
        .collect();

    // Score vector: U = Z' W (y - μ) / sqrt(V) = Z' weighted_resid (k × 1)
    let u: Array1<f64> = (0..k)
        .map(|j| {
            z_matrix
                .column(j)
                .iter()
                .zip(weighted_resid.iter())
                .map(|(&zj, &r)| zj * r)
                .sum()
        })
        .collect();

    // Z'WZ (k × k matrix)
    let mut zwz = Array2::<f64>::zeros((k, k));
    for i in 0..k {
        for j in i..k {
            let val: f64 = z_matrix
                .column(i)
                .iter()
                .zip(z_matrix.column(j).iter())
                .zip(w.iter())
                .map(|((&zi, &zj), &wi)| zi * zj * wi)
                .sum();
            zwz[[i, j]] = val;
            zwz[[j, i]] = val;
        }
    }

    // Z'WX (k × p matrix)
    let mut zwx = Array2::<f64>::zeros((k, p));
    for i in 0..k {
        for j in 0..p {
            zwx[[i, j]] = z_matrix
                .column(i)
                .iter()
                .zip(x.column(j).iter())
                .zip(w.iter())
                .map(|((&zi, &xj), &wi)| zi * xj * wi)
                .sum();
        }
    }

    // X'WZ = (Z'WX)' (p × k matrix) - we'll compute (X'WX)^-1 X'WZ = bread × X'WZ
    // Result is p × k
    let mut bread_xwz = Array2::<f64>::zeros((p, k));
    for i in 0..p {
        for j in 0..k {
            let mut val = 0.0;
            for l in 0..p {
                val += bread[[i, l]] * zwx[[j, l]]; // zwx[[j, l]] = (X'WZ)[[l, j]]
            }
            bread_xwz[[i, j]] = val;
        }
    }

    // Z'WX (X'WX)^-1 X'WZ = ZWX × bread_xwz (k × k matrix)
    let mut correction = Array2::<f64>::zeros((k, k));
    for i in 0..k {
        for j in 0..k {
            let mut val = 0.0;
            for l in 0..p {
                val += zwx[[i, l]] * bread_xwz[[l, j]];
            }
            correction[[i, j]] = val;
        }
    }

    // Information matrix: I = Z'WZ - correction (k × k)
    let mut info = Array2::<f64>::zeros((k, k));
    for i in 0..k {
        for j in 0..k {
            info[[i, j]] = zwz[[i, j]] - correction[[i, j]];
        }
    }

    // Score statistic: S = U' I^-1 U
    // Need to invert the k × k information matrix
    let statistic = invert_and_quadratic(&info, &u).unwrap_or(0.0);

    // P-value from chi-squared with df = k
    let pvalue = 1.0 - chi2_cdf_internal(statistic, k as f64);

    ScoreTestResult {
        statistic,
        df: k,
        pvalue,
        significant: pvalue < 0.05,
    }
}

/// Invert a small matrix and compute quadratic form u' A^-1 u
fn invert_and_quadratic(a: &Array2<f64>, u: &Array1<f64>) -> Option<f64> {
    let k = a.nrows();

    if k == 1 {
        // Simple case
        if a[[0, 0]].abs() < 1e-10 {
            return None;
        }
        return Some(u[0] * u[0] / a[[0, 0]]);
    }

    // Use Cholesky decomposition for small symmetric positive definite matrix
    // For simplicity, use LU decomposition via Gaussian elimination
    let mut work = a.clone();
    let mut pivot = vec![0usize; k];

    // LU decomposition with partial pivoting
    for i in 0..k {
        // Find pivot
        let mut max_val = work[[i, i]].abs();
        let mut max_row = i;
        for r in (i + 1)..k {
            if work[[r, i]].abs() > max_val {
                max_val = work[[r, i]].abs();
                max_row = r;
            }
        }

        if max_val < 1e-12 {
            return None; // Singular
        }

        pivot[i] = max_row;

        // Swap rows
        if max_row != i {
            for j in 0..k {
                let tmp = work[[i, j]];
                work[[i, j]] = work[[max_row, j]];
                work[[max_row, j]] = tmp;
            }
        }

        // Eliminate
        for r in (i + 1)..k {
            let factor = work[[r, i]] / work[[i, i]];
            work[[r, i]] = factor;
            for c in (i + 1)..k {
                work[[r, c]] -= factor * work[[i, c]];
            }
        }
    }

    // Solve L*y = P*u
    let mut y = u.clone();
    for i in 0..k {
        let pi = pivot[i];
        if pi != i {
            let tmp = y[i];
            y[i] = y[pi];
            y[pi] = tmp;
        }
        for j in 0..i {
            y[i] -= work[[i, j]] * y[j];
        }
    }

    // Solve U*x = y
    let mut x = y;
    for i in (0..k).rev() {
        for j in (i + 1)..k {
            x[i] -= work[[i, j]] * x[j];
        }
        if work[[i, i]].abs() < 1e-12 {
            return None;
        }
        x[i] /= work[[i, i]];
    }

    // Quadratic form: u' * x = u' * A^-1 * u
    let result: f64 = u.iter().zip(x.iter()).map(|(&ui, &xi)| ui * xi).sum();
    Some(result.max(0.0))
}

/// Internal chi-squared CDF (avoids circular dependency)
fn chi2_cdf_internal(x: f64, df: f64) -> f64 {
    use statrs::distribution::{ChiSquared, ContinuousCDF};
    if x < 0.0 || df <= 0.0 {
        return 0.0;
    }
    match ChiSquared::new(df) {
        Ok(dist) => dist.cdf(x),
        Err(_) => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::families::{GammaFamily, GaussianFamily};
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
    fn test_pvalue_and_confidence_interval_invalid_and_large_df_contracts() {
        assert!(pvalue_z(f64::NAN).is_nan());
        assert!(pvalue_t(f64::INFINITY, 10.0).is_nan());
        assert!(pvalue_t(1.0, 0.0).is_nan());

        let p_t = pvalue_t(2.5, 1001.0);
        let p_z = pvalue_z(2.5);
        assert_abs_diff_eq!(p_t, p_z, epsilon = 1e-12);

        let (lo, hi) = confidence_interval_z(f64::NAN, 1.0, 0.95);
        assert!(lo.is_nan() && hi.is_nan());
        let (lo, hi) = confidence_interval_z(1.0, 0.0, 0.95);
        assert!(lo.is_nan() && hi.is_nan());
        let (lo, hi) = confidence_interval_t(1.0, 0.5, 0.0, 0.95);
        assert!(lo.is_nan() && hi.is_nan());

        let z_ci = confidence_interval_z(1.25, 0.4, 0.90);
        let t_ci = confidence_interval_t(1.25, 0.4, 1001.0, 0.90);
        assert_abs_diff_eq!(t_ci.0, z_ci.0, epsilon = 1e-12);
        assert_abs_diff_eq!(t_ci.1, z_ci.1, epsilon = 1e-12);
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
        let x = arr2(&[[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]);
        let pearson_resid = arr1(&[0.1, -0.2, 0.15]);
        let irls_weights = arr1(&[1.0, 1.0, 1.0]);
        let prior_weights = arr1(&[1.0, 1.0, 1.0]);

        // Create a simple bread matrix (identity for testing)
        let bread = arr2(&[[0.5, 0.0], [0.0, 0.5]]);

        // HC0 should produce a valid covariance matrix
        let cov = robust_covariance(
            &x,
            &pearson_resid,
            &irls_weights,
            &prior_weights,
            &bread,
            HCType::HC0,
        );

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
        let cov = arr2(&[[0.04, 0.01], [0.01, 0.09]]);

        let se = robust_standard_errors(&cov);

        assert_abs_diff_eq!(se[0], 0.2, epsilon = 1e-10);
        assert_abs_diff_eq!(se[1], 0.3, epsilon = 1e-10);
    }

    #[test]
    fn test_hc1_larger_than_hc0() {
        use ndarray::{arr1, arr2};

        // HC1 should give larger standard errors than HC0 due to n/(n-p) correction
        let x = arr2(&[[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]]);
        let pearson_resid = arr1(&[0.1, -0.2, 0.15, -0.1]);
        let irls_weights = arr1(&[1.0, 1.0, 1.0, 1.0]);
        let prior_weights = arr1(&[1.0, 1.0, 1.0, 1.0]);
        let bread = arr2(&[[0.5, 0.0], [0.0, 0.5]]);

        let cov_hc0 = robust_covariance(
            &x,
            &pearson_resid,
            &irls_weights,
            &prior_weights,
            &bread,
            HCType::HC0,
        );
        let cov_hc1 = robust_covariance(
            &x,
            &pearson_resid,
            &irls_weights,
            &prior_weights,
            &bread,
            HCType::HC1,
        );

        // HC1 should be larger by factor of n/(n-p) = 4/2 = 2
        let expected_ratio = 4.0 / 2.0;
        assert_abs_diff_eq!(
            cov_hc1[[0, 0]] / cov_hc0[[0, 0]],
            expected_ratio,
            epsilon = 1e-10
        );
    }

    fn naive_robust_covariance(
        x: &Array2<f64>,
        pearson_resid: &Array1<f64>,
        irls_weights: &Array1<f64>,
        prior_weights: &Array1<f64>,
        bread: &Array2<f64>,
        hc_type: HCType,
    ) -> Array2<f64> {
        let n = x.nrows();
        let p = x.ncols();
        let combined_weights: Array1<f64> = prior_weights
            .iter()
            .zip(irls_weights.iter())
            .map(|(&pw, &iw)| pw * iw)
            .collect();

        let mut leverage = Array1::<f64>::zeros(n);
        if matches!(hc_type, HCType::HC2 | HCType::HC3) {
            for i in 0..n {
                let mut h_ii = 0.0;
                for j in 0..p {
                    let mut temp_j = 0.0;
                    for k in 0..p {
                        temp_j += bread[[j, k]] * x[[i, k]];
                    }
                    h_ii += x[[i, j]] * temp_j;
                }
                leverage[i] = (h_ii * combined_weights[i]).clamp(0.0, 0.9999);
            }
        }

        let scale = if matches!(hc_type, HCType::HC1) {
            n as f64 / (n.saturating_sub(p)) as f64
        } else {
            1.0
        };
        let mut meat = Array2::<f64>::zeros((p, p));
        for i in 0..n {
            let mut omega = scale * combined_weights[i] * pearson_resid[i] * pearson_resid[i];
            if matches!(hc_type, HCType::HC2) {
                omega /= (1.0 - leverage[i]).max(0.01);
            } else if matches!(hc_type, HCType::HC3) {
                let denom = (1.0 - leverage[i]).max(0.01);
                omega /= denom * denom;
            }
            for j in 0..p {
                for k in 0..p {
                    meat[[j, k]] += omega * x[[i, j]] * x[[i, k]];
                }
            }
        }
        bread.dot(&meat).dot(bread)
    }

    #[test]
    fn test_sparse_robust_covariance_matches_dense_formula() {
        let n = 48;
        let p = 24;
        let mut x = Array2::<f64>::zeros((n, p));
        for row in 0..n {
            x[[row, 0]] = 1.0;
            let j1 = 1 + row % (p - 1);
            let j2 = 1 + (row * 5 + 3) % (p - 1);
            x[[row, j1]] = 1.0 + (row % 4) as f64;
            x[[row, j2]] += (row as f64 / 4.0).sin();
        }
        assert!(should_use_sparse_sandwich_kernel(
            x.as_slice().expect("test matrix should be contiguous"),
            n,
            p
        ));

        let pearson_resid: Array1<f64> = (0..n).map(|i| ((i as f64) * 0.37).sin() * 0.2).collect();
        let irls_weights: Array1<f64> = (0..n).map(|i| 0.8 + (i % 5) as f64 * 0.03).collect();
        let prior_weights: Array1<f64> = (0..n).map(|i| 0.9 + (i % 7) as f64 * 0.02).collect();
        let mut bread = Array2::<f64>::zeros((p, p));
        for j in 0..p {
            bread[[j, j]] = 0.01 + j as f64 * 0.0001;
        }

        for hc_type in [HCType::HC0, HCType::HC1, HCType::HC2, HCType::HC3] {
            let actual = robust_covariance(
                &x,
                &pearson_resid,
                &irls_weights,
                &prior_weights,
                &bread,
                hc_type,
            );
            let expected = naive_robust_covariance(
                &x,
                &pearson_resid,
                &irls_weights,
                &prior_weights,
                &bread,
                hc_type,
            );
            for j in 0..p {
                for k in 0..p {
                    assert_abs_diff_eq!(actual[[j, k]], expected[[j, k]], epsilon = 1e-12);
                }
            }
        }
    }

    #[test]
    fn test_dense_robust_covariance_all_hc_types_match_naive_reference() {
        use ndarray::{arr1, arr2};

        let x = arr2(&[
            [1.0, -2.0, 0.5],
            [1.0, -1.0, 1.5],
            [1.0, 0.0, -0.5],
            [1.0, 1.0, 2.5],
            [1.0, 2.0, -1.5],
            [1.0, 3.0, 0.0],
        ]);
        let pearson_resid = arr1(&[0.0, -0.3, 0.2, 0.4, -0.1, 0.15]);
        let irls_weights = arr1(&[1.0, 0.8, 1.2, 0.9, 1.1, 1.4]);
        let prior_weights = arr1(&[0.7, 1.3, 1.0, 0.6, 1.5, 0.9]);
        let bread = arr2(&[[0.20, -0.01, 0.02], [-0.01, 0.08, 0.01], [0.02, 0.01, 0.05]]);

        for hc_type in [HCType::HC0, HCType::HC1, HCType::HC2, HCType::HC3] {
            let actual = robust_covariance(
                &x,
                &pearson_resid,
                &irls_weights,
                &prior_weights,
                &bread,
                hc_type,
            );
            let expected = naive_robust_covariance(
                &x,
                &pearson_resid,
                &irls_weights,
                &prior_weights,
                &bread,
                hc_type,
            );
            for j in 0..x.ncols() {
                for k in 0..x.ncols() {
                    assert_abs_diff_eq!(actual[[j, k]], expected[[j, k]], epsilon = 1e-12);
                }
            }
        }
    }

    #[test]
    fn test_score_test_continuous_basic() {
        use ndarray::arr2;

        // Simple case: test if adding a variable correlated with residuals is significant
        let n = 100;
        let x = Array2::from_shape_fn((n, 2), |(i, j)| if j == 0 { 1.0 } else { i as f64 / 10.0 });
        let y: Array1<f64> = (0..n).map(|i| (i as f64 / 10.0) + 0.5).collect();
        let mu: Array1<f64> = (0..n).map(|i| (i as f64 / 10.0) + 0.3).collect(); // Slightly off
        let weights = Array1::ones(n);
        let bread = arr2(&[[0.1, 0.0], [0.0, 0.1]]);

        // New variable that explains residuals
        let z: Array1<f64> = (0..n).map(|i| (i as f64).sin()).collect();

        let result = score_test_continuous(&z, &x, &y, &mu, &weights, &bread, &GaussianFamily);

        // Should produce a valid result
        assert!(result.statistic >= 0.0);
        assert_eq!(result.df, 1);
        assert!(result.pvalue >= 0.0 && result.pvalue <= 1.0);
    }

    #[test]
    fn test_score_test_continuous_null_variable() {
        use ndarray::arr2;

        // Test with a variable that has no relationship to residuals
        let n = 50;
        let x = Array2::from_shape_fn((n, 2), |(i, j)| if j == 0 { 1.0 } else { i as f64 });
        let y: Array1<f64> = (0..n).map(|i| i as f64 + 1.0).collect();
        let mu = y.clone(); // Perfect fit - no residuals
        let weights = Array1::ones(n);
        let bread = arr2(&[[0.5, 0.0], [0.0, 0.01]]);

        // Random variable unrelated to (zero) residuals
        let z = Array1::ones(n);

        let result = score_test_continuous(&z, &x, &y, &mu, &weights, &bread, &GaussianFamily);

        // With zero residuals, score should be 0
        assert_abs_diff_eq!(result.statistic, 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result.pvalue, 1.0, epsilon = 0.01);
        assert!(!result.significant);
    }

    #[test]
    fn test_score_test_continuous_batch_matches_single_path_and_strided_x() {
        let n = 40;
        let x = Array2::from_shape_fn((n, 2), |(i, j)| {
            if j == 0 {
                1.0
            } else {
                (i as f64 - 20.0) / 10.0
            }
        });
        let x_nonstandard =
            Array2::from_shape_fn((2, n), |(j, i)| if j == 0 { 1.0 } else { i as f64 / 10.0 })
                .reversed_axes();
        assert!(x_nonstandard.as_slice().is_none());

        let y = Array1::from_iter((0..n).map(|i| 2.0 + 0.1 * i as f64 + (i as f64).sin() * 0.2));
        let mu = Array1::from_iter((0..n).map(|i| 2.0 + 0.1 * i as f64));
        let weights = Array1::from_iter((0..n).map(|i| 0.8 + (i % 5) as f64 * 0.1));
        let bread = ndarray::arr2(&[[0.05, -0.002], [-0.002, 0.015]]);
        let zs = Array2::from_shape_fn((n, 3), |(i, j)| match j {
            0 => (i as f64 / 3.0).sin(),
            1 => (i as f64 / 5.0).cos(),
            _ => {
                if i % 2 == 0 {
                    1.0
                } else {
                    -1.0
                }
            }
        });

        let batch =
            score_test_continuous_batch(&zs, &x, &y, &mu, &weights, &bread, &GaussianFamily);
        assert_eq!(batch.len(), zs.ncols());
        for j in 0..zs.ncols() {
            let single = score_test_continuous(
                &zs.column(j).to_owned(),
                &x,
                &y,
                &mu,
                &weights,
                &bread,
                &GaussianFamily,
            );
            assert_abs_diff_eq!(batch[j].statistic, single.statistic, epsilon = 1e-12);
            assert_abs_diff_eq!(batch[j].pvalue, single.pvalue, epsilon = 1e-12);
            assert_eq!(batch[j].significant, single.significant);
        }

        let strided_batch = score_test_continuous_batch(
            &zs,
            &x_nonstandard,
            &y,
            &mu,
            &weights,
            &bread,
            &GaussianFamily,
        );
        assert_eq!(strided_batch.len(), zs.ncols());
        assert!(strided_batch.iter().all(|r| r.statistic >= 0.0));
    }

    #[test]
    fn test_score_test_categorical_basic() {
        use ndarray::arr2;

        // Test categorical score test
        let n = 60;
        let x = Array2::from_shape_fn((n, 1), |_| 1.0); // Intercept only
        let y: Array1<f64> = (0..n)
            .map(|i| {
                if i < 20 {
                    1.0
                } else if i < 40 {
                    2.0
                } else {
                    3.0
                }
            })
            .collect();
        let mu = Array1::from_elem(n, 2.0); // Mean prediction
        let weights = Array1::ones(n);
        let bread = arr2(&[[1.0 / n as f64]]);

        // Dummy matrix for 3-level categorical (2 columns after base exclusion)
        let mut z_matrix = Array2::zeros((n, 2));
        for i in 20..40 {
            z_matrix[[i, 0]] = 1.0; // Level 2
        }
        for i in 40..n {
            z_matrix[[i, 1]] = 1.0; // Level 3
        }

        let result =
            score_test_categorical(&z_matrix, &x, &y, &mu, &weights, &bread, &GaussianFamily);

        assert!(result.statistic >= 0.0);
        assert_eq!(result.df, 2);
        assert!(result.pvalue >= 0.0 && result.pvalue <= 1.0);
        // The categorical should be significant since y varies by level
        assert!(result.significant);
    }

    #[test]
    fn test_score_test_empty_categorical() {
        use ndarray::arr2;

        // Test with empty categorical (0 columns)
        let n = 10;
        let x = Array2::from_shape_fn((n, 1), |_| 1.0);
        let y = Array1::ones(n);
        let mu = Array1::ones(n);
        let weights = Array1::ones(n);
        let bread = arr2(&[[0.1]]);
        let z_matrix = Array2::zeros((n, 0));

        let result =
            score_test_categorical(&z_matrix, &x, &y, &mu, &weights, &bread, &GaussianFamily);

        assert_eq!(result.df, 0);
        assert_abs_diff_eq!(result.pvalue, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_score_test_categorical_one_column_singular_and_zero_variance_paths() {
        let n = 12;
        let x = Array2::from_elem((n, 1), 1.0);
        let y = Array1::from_iter((0..n).map(|i| if i < 6 { 1.0 } else { 2.0 }));
        let mu = Array1::from_elem(n, 1.5);
        let weights = Array1::ones(n);
        let bread = ndarray::arr2(&[[1.0 / n as f64]]);

        let mut one_col = Array2::zeros((n, 1));
        for i in 6..n {
            one_col[[i, 0]] = 1.0;
        }
        let result =
            score_test_categorical(&one_col, &x, &y, &mu, &weights, &bread, &GaussianFamily);
        assert_eq!(result.df, 1);
        assert!(result.statistic >= 0.0);

        let singular = Array2::<f64>::zeros((n, 1));
        let result =
            score_test_categorical(&singular, &x, &y, &mu, &weights, &bread, &GaussianFamily);
        assert_eq!(result.df, 1);
        assert_abs_diff_eq!(result.statistic, 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(result.pvalue, 1.0, epsilon = 1e-12);

        let zero_mu = Array1::zeros(n);
        let result =
            score_test_categorical(&one_col, &x, &y, &zero_mu, &weights, &bread, &GammaFamily);
        assert_eq!(result.df, 1);
        assert_abs_diff_eq!(result.statistic, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_invert_quadratic_and_chi_square_edge_contracts() {
        let one = ndarray::arr2(&[[4.0]]);
        let u = ndarray::arr1(&[2.0]);
        assert_abs_diff_eq!(
            invert_and_quadratic(&one, &u).expect("invertible 1x1"),
            1.0,
            epsilon = 1e-12
        );

        let singular_one = ndarray::arr2(&[[0.0]]);
        assert!(invert_and_quadratic(&singular_one, &u).is_none());

        let pivot = ndarray::arr2(&[[0.0, 1.0], [1.0, 2.0]]);
        let u = ndarray::arr1(&[1.0, 2.0]);
        assert_abs_diff_eq!(
            invert_and_quadratic(&pivot, &u).expect("pivoted matrix is invertible"),
            2.0,
            epsilon = 1e-12
        );

        let singular = ndarray::arr2(&[[1.0, 2.0], [2.0, 4.0]]);
        let u = ndarray::arr1(&[1.0, 1.0]);
        assert!(invert_and_quadratic(&singular, &u).is_none());

        assert_abs_diff_eq!(chi2_cdf_internal(-1.0, 1.0), 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(chi2_cdf_internal(1.0, 0.0), 0.0, epsilon = 1e-12);
    }
}
