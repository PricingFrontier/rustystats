//! Statistical distribution functions for p-value calculations.
//!
//! Implements CDFs for chi-squared, t, and F distributions without external dependencies.
//! Uses the incomplete beta and gamma functions.

use std::f64::consts::PI;

// =============================================================================
// Constants
// =============================================================================

const EPSILON: f64 = 1e-15;
const MAX_ITER: usize = 1000;

// Lanczos approximation coefficients for gamma function
const LANCZOS_G: f64 = 7.0;
const LANCZOS_COEFFS: [f64; 9] = [
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
];

// =============================================================================
// Gamma Function
// =============================================================================

/// Compute ln(Gamma(x)) using Lanczos approximation.
fn ln_gamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
    }
    
    if x < 0.5 {
        // Reflection formula: Gamma(x) * Gamma(1-x) = pi / sin(pi*x)
        let sin_pi_x = (PI * x).sin();
        if sin_pi_x.abs() < EPSILON {
            return f64::INFINITY;
        }
        return PI.ln() - sin_pi_x.abs().ln() - ln_gamma(1.0 - x);
    }
    
    let x = x - 1.0;
    let mut sum = LANCZOS_COEFFS[0];
    for i in 1..9 {
        sum += LANCZOS_COEFFS[i] / (x + i as f64);
    }
    
    let t = x + LANCZOS_G + 0.5;
    0.5 * (2.0 * PI).ln() + (x + 0.5) * t.ln() - t + sum.ln()
}

// =============================================================================
// Incomplete Gamma Function
// =============================================================================

/// Lower incomplete gamma function P(a, x) = gamma(a, x) / Gamma(a)
/// Using series expansion for x < a + 1, continued fraction otherwise.
fn lower_incomplete_gamma(a: f64, x: f64) -> f64 {
    if x < 0.0 || a <= 0.0 {
        return 0.0;
    }
    if x == 0.0 {
        return 0.0;
    }
    
    if x < a + 1.0 {
        // Series expansion
        gamma_series(a, x)
    } else {
        // Continued fraction
        1.0 - gamma_continued_fraction(a, x)
    }
}

/// Series expansion for lower incomplete gamma
fn gamma_series(a: f64, x: f64) -> f64 {
    let ln_gamma_a = ln_gamma(a);
    
    let mut sum = 1.0 / a;
    let mut term = 1.0 / a;
    
    for n in 1..MAX_ITER {
        term *= x / (a + n as f64);
        sum += term;
        if term.abs() < sum.abs() * EPSILON {
            break;
        }
    }
    
    sum * (-x + a * x.ln() - ln_gamma_a).exp()
}

/// Continued fraction for upper incomplete gamma
fn gamma_continued_fraction(a: f64, x: f64) -> f64 {
    let ln_gamma_a = ln_gamma(a);
    
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / EPSILON;
    let mut d = 1.0 / b;
    let mut h = d;
    
    for i in 1..MAX_ITER {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < EPSILON {
            d = EPSILON;
        }
        c = b + an / c;
        if c.abs() < EPSILON {
            c = EPSILON;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPSILON {
            break;
        }
    }
    
    (-x + a * x.ln() - ln_gamma_a).exp() * h
}

// =============================================================================
// Incomplete Beta Function
// =============================================================================

/// Regularized incomplete beta function I_x(a, b)
fn incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    
    // Use symmetry relation for better convergence
    if x > (a + 1.0) / (a + b + 2.0) {
        return 1.0 - incomplete_beta(1.0 - x, b, a);
    }
    
    // Continued fraction expansion
    let ln_beta = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b);
    let front = (a * x.ln() + b * (1.0 - x).ln() - ln_beta).exp() / a;
    
    beta_continued_fraction(x, a, b) * front
}

/// Continued fraction for incomplete beta
fn beta_continued_fraction(x: f64, a: f64, b: f64) -> f64 {
    let mut c = 1.0;
    let mut d = 1.0 / (1.0 - (a + b) * x / (a + 1.0)).max(EPSILON);
    let mut h = d;
    
    for m in 1..MAX_ITER {
        let m = m as f64;
        
        // Even step
        let num = m * (b - m) * x / ((a + 2.0 * m - 1.0) * (a + 2.0 * m));
        d = 1.0 + num * d;
        if d.abs() < EPSILON {
            d = EPSILON;
        }
        c = 1.0 + num / c;
        if c.abs() < EPSILON {
            c = EPSILON;
        }
        d = 1.0 / d;
        h *= d * c;
        
        // Odd step
        let num = -(a + m) * (a + b + m) * x / ((a + 2.0 * m) * (a + 2.0 * m + 1.0));
        d = 1.0 + num * d;
        if d.abs() < EPSILON {
            d = EPSILON;
        }
        c = 1.0 + num / c;
        if c.abs() < EPSILON {
            c = EPSILON;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        
        if (del - 1.0).abs() < EPSILON {
            break;
        }
    }
    
    h
}

// =============================================================================
// Distribution CDFs
// =============================================================================

/// Chi-squared distribution CDF.
///
/// P(X <= x) where X ~ χ²(df)
///
/// # Arguments
/// * `x` - Value to evaluate CDF at
/// * `df` - Degrees of freedom (must be positive)
///
/// # Returns
/// Probability P(X <= x)
pub fn chi2_cdf(x: f64, df: f64) -> f64 {
    if x <= 0.0 || df <= 0.0 {
        return 0.0;
    }
    lower_incomplete_gamma(df / 2.0, x / 2.0)
}

/// Student's t-distribution CDF.
///
/// P(X <= x) where X ~ t(df)
///
/// # Arguments
/// * `x` - Value to evaluate CDF at
/// * `df` - Degrees of freedom (must be positive)
///
/// # Returns
/// Probability P(X <= x)
pub fn t_cdf(x: f64, df: f64) -> f64 {
    if df <= 0.0 {
        return f64::NAN;
    }
    
    let t2 = x * x;
    let p = incomplete_beta(df / (df + t2), df / 2.0, 0.5);
    
    if x >= 0.0 {
        1.0 - 0.5 * p
    } else {
        0.5 * p
    }
}

/// F-distribution CDF.
///
/// P(X <= x) where X ~ F(df1, df2)
///
/// # Arguments
/// * `x` - Value to evaluate CDF at
/// * `df1` - Numerator degrees of freedom (must be positive)
/// * `df2` - Denominator degrees of freedom (must be positive)
///
/// # Returns
/// Probability P(X <= x)
pub fn f_cdf(x: f64, df1: f64, df2: f64) -> f64 {
    if x <= 0.0 || df1 <= 0.0 || df2 <= 0.0 {
        return 0.0;
    }
    
    let z = df1 * x / (df1 * x + df2);
    incomplete_beta(z, df1 / 2.0, df2 / 2.0)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    
    #[test]
    fn test_chi2_cdf() {
        // Test against known values (from scipy)
        // chi2.cdf(3.84, 1) ≈ 0.95
        assert_abs_diff_eq!(chi2_cdf(3.841, 1.0), 0.95, epsilon = 0.001);
        
        // chi2.cdf(5.99, 2) ≈ 0.95
        assert_abs_diff_eq!(chi2_cdf(5.991, 2.0), 0.95, epsilon = 0.001);
        
        // chi2.cdf(9.49, 4) ≈ 0.95
        assert_abs_diff_eq!(chi2_cdf(9.488, 4.0), 0.95, epsilon = 0.001);
        
        // Edge cases
        assert_eq!(chi2_cdf(0.0, 1.0), 0.0);
        assert_eq!(chi2_cdf(-1.0, 1.0), 0.0);
    }
    
    #[test]
    fn test_t_cdf() {
        // t.cdf(1.96, inf) ≈ 0.975 (approaches normal)
        // t.cdf(1.96, 1000) ≈ 0.975
        assert_abs_diff_eq!(t_cdf(1.96, 1000.0), 0.975, epsilon = 0.001);
        
        // t.cdf(0, any) = 0.5 (symmetric)
        assert_abs_diff_eq!(t_cdf(0.0, 10.0), 0.5, epsilon = 0.001);
        
        // t.cdf(-2.228, 10) ≈ 0.025
        assert_abs_diff_eq!(t_cdf(-2.228, 10.0), 0.025, epsilon = 0.002);
    }
    
    #[test]
    fn test_f_cdf() {
        // f.cdf(3.84, 1, 100) ≈ 0.947 
        assert_abs_diff_eq!(f_cdf(3.84, 1.0, 100.0), 0.947, epsilon = 0.01);
        
        // f.cdf(2.70, 2, 100) ≈ 0.928
        assert_abs_diff_eq!(f_cdf(2.70, 2.0, 100.0), 0.928, epsilon = 0.01);
        
        // Edge cases
        assert_eq!(f_cdf(0.0, 1.0, 1.0), 0.0);
        assert_eq!(f_cdf(-1.0, 1.0, 1.0), 0.0);
    }
}
