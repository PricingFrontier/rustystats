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
    _family: &str,
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
    let u: f64 = z.iter()
        .zip(y.iter())
        .zip(mu.iter())
        .map(|((&zi, &yi), &mui)| zi * (yi - mui))
        .sum();
    
    // Information: I_zz = Z'WZ - Z'WX (X'WX)^-1 X'WZ
    // where W = diag(weights) is the IRLS weight matrix
    
    // Z'WZ (scalar)
    let zwz: f64 = z.iter()
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
        .map(|i| {
            (0..p).map(|j| bread[[i, j]] * zwx[j]).sum::<f64>()
        })
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
    family: &str,
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
    
    // Compute variance function values
    let variance = compute_variance(mu, family);
    
    // Working weights: w_i * V(μ_i)
    let w: Array1<f64> = weights.iter()
        .zip(variance.iter())
        .map(|(&wi, &vi)| wi * vi)
        .collect();
    
    // Pearson residuals scaled by sqrt(weight)
    let weighted_resid: Array1<f64> = y.iter()
        .zip(mu.iter())
        .zip(weights.iter())
        .zip(variance.iter())
        .map(|(((&yi, &mui), &wi), &vi)| {
            if vi > 1e-10 { wi * (yi - mui) / vi.sqrt() } else { 0.0 }
        })
        .collect();
    
    // Score vector: U = Z' W (y - μ) / sqrt(V) = Z' weighted_resid (k × 1)
    let u: Array1<f64> = (0..k)
        .map(|j| {
            z_matrix.column(j).iter()
                .zip(weighted_resid.iter())
                .map(|(&zj, &r)| zj * r)
                .sum()
        })
        .collect();
    
    // Z'WZ (k × k matrix)
    let mut zwz = Array2::<f64>::zeros((k, k));
    for i in 0..k {
        for j in i..k {
            let val: f64 = z_matrix.column(i).iter()
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
            zwx[[i, j]] = z_matrix.column(i).iter()
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
    let statistic = match invert_and_quadratic(&info, &u) {
        Some(s) => s,
        None => 0.0, // Singular matrix
    };
    
    // P-value from chi-squared with df = k
    let pvalue = 1.0 - chi2_cdf_internal(statistic, k as f64);
    
    ScoreTestResult {
        statistic,
        df: k,
        pvalue,
        significant: pvalue < 0.05,
    }
}

/// Compute variance function for a family
fn compute_variance(mu: &Array1<f64>, family: &str) -> Array1<f64> {
    let lower = family.to_lowercase();
    
    // Handle NegBin with theta parameter
    if lower.starts_with("negativebinomial") || lower.starts_with("negbinomial") || lower.starts_with("negbin") {
        let theta = if let Some(start) = lower.find("theta=") {
            let rest = &lower[start + 6..];
            let end = rest.find(')').unwrap_or(rest.len());
            rest[..end].parse::<f64>().unwrap_or(1.0)
        } else {
            1.0
        };
        return mu.iter().map(|&m| m + m * m / theta).collect();
    }
    
    match lower.as_str() {
        "gaussian" | "normal" => Array1::ones(mu.len()),
        "poisson" | "quasipoisson" => mu.clone(),
        "binomial" | "quasibinomial" => mu.iter().map(|&m| m * (1.0 - m).max(1e-10)).collect(),
        "gamma" => mu.iter().map(|&m| m * m).collect(),
        _ => mu.clone(), // Default to Poisson-like
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

    #[test]
    fn test_score_test_continuous_basic() {
        use ndarray::{arr1, arr2};
        
        // Simple case: test if adding a variable correlated with residuals is significant
        let n = 100;
        let x = Array2::from_shape_fn((n, 2), |(i, j)| if j == 0 { 1.0 } else { i as f64 / 10.0 });
        let y: Array1<f64> = (0..n).map(|i| (i as f64 / 10.0) + 0.5).collect();
        let mu: Array1<f64> = (0..n).map(|i| (i as f64 / 10.0) + 0.3).collect(); // Slightly off
        let weights = Array1::ones(n);
        let bread = arr2(&[[0.1, 0.0], [0.0, 0.1]]);
        
        // New variable that explains residuals
        let z: Array1<f64> = (0..n).map(|i| (i as f64).sin()).collect();
        
        let result = score_test_continuous(&z, &x, &y, &mu, &weights, &bread, "gaussian");
        
        // Should produce a valid result
        assert!(result.statistic >= 0.0);
        assert_eq!(result.df, 1);
        assert!(result.pvalue >= 0.0 && result.pvalue <= 1.0);
    }

    #[test]
    fn test_score_test_continuous_null_variable() {
        use ndarray::{arr1, arr2};
        
        // Test with a variable that has no relationship to residuals
        let n = 50;
        let x = Array2::from_shape_fn((n, 2), |(i, j)| if j == 0 { 1.0 } else { i as f64 });
        let y: Array1<f64> = (0..n).map(|i| i as f64 + 1.0).collect();
        let mu = y.clone(); // Perfect fit - no residuals
        let weights = Array1::ones(n);
        let bread = arr2(&[[0.5, 0.0], [0.0, 0.01]]);
        
        // Random variable unrelated to (zero) residuals
        let z = Array1::ones(n);
        
        let result = score_test_continuous(&z, &x, &y, &mu, &weights, &bread, "gaussian");
        
        // With zero residuals, score should be 0
        assert_abs_diff_eq!(result.statistic, 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result.pvalue, 1.0, epsilon = 0.01);
        assert!(!result.significant);
    }

    #[test]
    fn test_score_test_categorical_basic() {
        use ndarray::arr2;
        
        // Test categorical score test
        let n = 60;
        let x = Array2::from_shape_fn((n, 1), |_| 1.0); // Intercept only
        let y: Array1<f64> = (0..n).map(|i| if i < 20 { 1.0 } else if i < 40 { 2.0 } else { 3.0 }).collect();
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
        
        let result = score_test_categorical(&z_matrix, &x, &y, &mu, &weights, &bread, "gaussian");
        
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
        
        let result = score_test_categorical(&z_matrix, &x, &y, &mu, &weights, &bread, "gaussian");
        
        assert_eq!(result.df, 0);
        assert_abs_diff_eq!(result.pvalue, 1.0, epsilon = 1e-10);
    }
}
