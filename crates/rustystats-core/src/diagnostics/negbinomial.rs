// =============================================================================
// Negative Binomial Theta Estimation
// =============================================================================
//
// This module provides functions for estimating the dispersion parameter θ
// in Negative Binomial regression using profile likelihood.
//
// ALGORITHM:
// ----------
// Given fitted values μ (from GLM with current θ), we maximize the profile
// log-likelihood with respect to θ:
//
// l(θ; y, μ) = Σ[lgamma(y+θ) - lgamma(θ) - lgamma(y+1) 
//              + θ*log(θ/(μ+θ)) + y*log(μ/(μ+θ))]
//
// We use Brent's method (golden section + parabolic interpolation) for
// robust optimization within bounds [min_theta, max_theta].
//
// PROFILE LIKELIHOOD ITERATION:
// -----------------------------
// 1. Start with initial θ (e.g., 1.0)
// 2. Fit NB GLM with current θ → get μ
// 3. Optimize θ given μ using profile likelihood
// 4. If |θ_new - θ_old| < tol, stop; else goto 2
//
// REFERENCES:
// -----------
// - Venables & Ripley (2002), "Modern Applied Statistics with S", Ch. 7.4
// - R's MASS::glm.nb implementation
// - statsmodels NegativeBinomial implementation
//
// =============================================================================

use ndarray::Array1;
use std::f64::consts::PI;

/// Lanczos approximation for log-gamma function.
#[inline]
fn lgamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
    }
    
    const G: f64 = 7.0;
    const C: [f64; 9] = [
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
    
    if x < 0.5 {
        PI.ln() - (PI * x).sin().ln() - lgamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let mut sum = C[0];
        for i in 1..9 {
            sum += C[i] / (x + i as f64);
        }
        let t = x + G + 0.5;
        0.5 * (2.0 * PI).ln() + (t.ln() * (x + 0.5)) - t + sum.ln()
    }
}

/// Negative Binomial log-likelihood for a single observation.
///
/// l_i = lgamma(y+θ) - lgamma(θ) - lgamma(y+1) + θ*log(θ/(μ+θ)) + y*log(μ/(μ+θ))
#[inline]
fn nb_loglik_single(y: f64, mu: f64, theta: f64) -> f64 {
    let mu_safe = mu.max(1e-10);
    let theta_safe = theta.max(1e-10);
    
    let term1 = lgamma(y + theta_safe) - lgamma(theta_safe) - lgamma(y + 1.0);
    let term2 = theta_safe * (theta_safe / (mu_safe + theta_safe)).ln();
    let term3 = y * (mu_safe / (mu_safe + theta_safe)).ln();
    
    term1 + term2 + term3
}

/// Compute the Negative Binomial log-likelihood for all observations.
///
/// # Arguments
/// * `y` - Response values
/// * `mu` - Fitted values (means)
/// * `theta` - Dispersion parameter
/// * `weights` - Optional prior weights
///
/// # Returns
/// Total log-likelihood
pub fn nb_loglikelihood(
    y: &Array1<f64>,
    mu: &Array1<f64>,
    theta: f64,
    weights: Option<&Array1<f64>>,
) -> f64 {
    // For very large theta, NegBin converges to Poisson
    // Use Poisson log-likelihood to avoid numerical instability from lgamma of large values
    if theta > 100.0 {
        return poisson_loglikelihood(y, mu, weights);
    }
    
    match weights {
        Some(w) => {
            y.iter()
                .zip(mu.iter())
                .zip(w.iter())
                .map(|((&yi, &mui), &wi)| wi * nb_loglik_single(yi, mui, theta))
                .sum()
        }
        None => {
            y.iter()
                .zip(mu.iter())
                .map(|(&yi, &mui)| nb_loglik_single(yi, mui, theta))
                .sum()
        }
    }
}

/// Poisson log-likelihood (used when NegBin theta is very large).
/// 
/// When theta → ∞, NegBin → Poisson. Using Poisson LL avoids numerical
/// instability from lgamma of very large arguments.
fn poisson_loglikelihood(
    y: &Array1<f64>,
    mu: &Array1<f64>,
    weights: Option<&Array1<f64>>,
) -> f64 {
    // Poisson LL: y*log(mu) - mu - lgamma(y+1)
    match weights {
        Some(w) => {
            y.iter().zip(mu.iter()).zip(w.iter())
                .map(|((&yi, &mui), &wi)| {
                    let mui_safe = mui.max(1e-10);
                    let y_term = if yi > 0.0 { yi * mui_safe.ln() } else { 0.0 };
                    wi * (y_term - mui_safe - lgamma(yi + 1.0))
                })
                .sum()
        }
        None => {
            y.iter().zip(mu.iter())
                .map(|(&yi, &mui)| {
                    let mui_safe = mui.max(1e-10);
                    let y_term = if yi > 0.0 { yi * mui_safe.ln() } else { 0.0 };
                    y_term - mui_safe - lgamma(yi + 1.0)
                })
                .sum()
        }
    }
}

/// Estimate optimal θ using profile likelihood given fitted values μ.
///
/// Uses Brent's method for robust optimization within bounds.
///
/// # Arguments
/// * `y` - Response values
/// * `mu` - Fitted values from current GLM fit
/// * `weights` - Optional prior weights
/// * `min_theta` - Lower bound for θ search (default: 0.01)
/// * `max_theta` - Upper bound for θ search (default: 1000.0)
/// * `tol` - Convergence tolerance (default: 1e-5)
///
/// # Returns
/// Estimated optimal θ
pub fn estimate_theta_profile(
    y: &Array1<f64>,
    mu: &Array1<f64>,
    weights: Option<&Array1<f64>>,
    min_theta: f64,
    max_theta: f64,
    tol: f64,
) -> f64 {
    // Objective: maximize log-likelihood = minimize negative log-likelihood
    let objective = |theta: f64| -> f64 {
        -nb_loglikelihood(y, mu, theta, weights)
    };
    
    // Brent's method for minimization
    brent_minimize(objective, min_theta, max_theta, tol)
}

/// Brent's method for finding minimum of a univariate function.
///
/// Combines golden section search with parabolic interpolation for
/// robust and efficient minimization.
fn brent_minimize<F>(f: F, a: f64, b: f64, tol: f64) -> f64
where
    F: Fn(f64) -> f64,
{
    const GOLDEN: f64 = 0.381966011250105; // (3 - sqrt(5)) / 2
    const MAX_ITER: usize = 100;
    
    let mut a = a;
    let mut b = b;
    let mut x = a + GOLDEN * (b - a);
    let mut w = x;
    let mut v = x;
    let mut fx = f(x);
    let mut fw = fx;
    let mut fv = fx;
    let mut e: f64 = 0.0;
    let mut d: f64 = 0.0;
    
    for _ in 0..MAX_ITER {
        let midpoint = 0.5 * (a + b);
        let tol1 = tol * x.abs() + 1e-10;
        let tol2 = 2.0 * tol1;
        
        // Check convergence
        if (x - midpoint).abs() <= tol2 - 0.5 * (b - a) {
            return x;
        }
        
        // Try parabolic fit
        let mut use_golden = true;
        
        if e.abs() > tol1 {
            // Parabolic interpolation
            let r = (x - w) * (fx - fv);
            let mut q = (x - v) * (fx - fw);
            let mut p = (x - v) * q - (x - w) * r;
            q = 2.0 * (q - r);
            
            if q > 0.0 {
                p = -p;
            } else {
                q = -q;
            }
            
            let r = e;
            e = d;
            
            // Check if parabolic step is acceptable
            if p.abs() < (0.5 * q * r).abs() && p > q * (a - x) && p < q * (b - x) {
                d = p / q;
                let u = x + d;
                
                // f must not be evaluated too close to a or b
                if (u - a) < tol2 || (b - u) < tol2 {
                    d = if x < midpoint { tol1 } else { -tol1 };
                }
                use_golden = false;
            }
        }
        
        if use_golden {
            // Golden section step
            e = if x < midpoint { b - x } else { a - x };
            d = GOLDEN * e;
        }
        
        // f must not be evaluated too close to x
        let u = if d.abs() >= tol1 {
            x + d
        } else if d > 0.0 {
            x + tol1
        } else {
            x - tol1
        };
        
        let fu = f(u);
        
        // Update interval
        if fu <= fx {
            if u < x {
                b = x;
            } else {
                a = x;
            }
            v = w;
            fv = fw;
            w = x;
            fw = fx;
            x = u;
            fx = fu;
        } else {
            if u < x {
                a = u;
            } else {
                b = u;
            }
            if fu <= fw || w == x {
                v = w;
                fv = fw;
                w = u;
                fw = fu;
            } else if fu <= fv || v == x || v == w {
                v = u;
                fv = fu;
            }
        }
    }
    
    x
}

/// Method-of-moments estimator for initial θ.
///
/// Uses the relationship: Var(Y) = μ + μ²/θ
/// Solving for θ: θ = μ² / (Var(Y) - μ)
///
/// This provides a quick initial estimate before profile likelihood refinement.
pub fn estimate_theta_moments(y: &Array1<f64>, mu: &Array1<f64>) -> f64 {
    let n = y.len() as f64;
    
    // Compute mean of mu
    let mu_mean: f64 = mu.iter().sum::<f64>() / n;
    
    // Compute sample variance of y
    let y_mean: f64 = y.iter().sum::<f64>() / n;
    let y_var: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum::<f64>() / (n - 1.0);
    
    // theta = mu^2 / (var - mu)
    // If var <= mu (no overdispersion), return large theta (close to Poisson)
    if y_var <= mu_mean {
        return 100.0;  // Near-Poisson
    }
    
    let theta = mu_mean.powi(2) / (y_var - mu_mean);
    
    // Clamp to reasonable range
    theta.max(0.01).min(1000.0)
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
    fn test_nb_loglik_basic() {
        // Basic sanity check: log-likelihood should be negative (log of probability)
        let ll = nb_loglik_single(1.0, 1.0, 1.0);
        assert!(ll < 0.0);
        assert!(ll.is_finite());
    }

    #[test]
    fn test_nb_loglikelihood_array() {
        let y = array![0.0, 1.0, 2.0, 3.0];
        let mu = array![1.0, 1.5, 2.0, 2.5];
        
        let ll = nb_loglikelihood(&y, &mu, 1.0, None);
        
        assert!(ll.is_finite());
        assert!(ll < 0.0);  // Log-likelihood should be negative
    }

    #[test]
    fn test_estimate_theta_profile() {
        // Generate data that looks like NB with known theta
        let y = array![0.0, 1.0, 0.0, 2.0, 1.0, 3.0, 0.0, 1.0, 4.0, 2.0];
        let mu = array![1.0, 1.0, 1.0, 2.0, 1.0, 2.0, 1.0, 1.0, 3.0, 2.0];
        
        let theta = estimate_theta_profile(&y, &mu, None, 0.01, 100.0, 1e-5);
        
        // Should return a reasonable positive value
        assert!(theta > 0.0);
        assert!(theta < 100.0);
    }

    #[test]
    fn test_estimate_theta_moments() {
        // Create overdispersed data
        let y = array![0.0, 5.0, 1.0, 8.0, 0.0, 3.0, 0.0, 10.0];
        let mu = array![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0];
        
        let theta = estimate_theta_moments(&y, &mu);
        
        // With high variance relative to mean, theta should be small
        assert!(theta > 0.0);
        assert!(theta < 10.0);
    }

    #[test]
    fn test_brent_minimize_quadratic() {
        // Test with a simple quadratic: (x-3)^2, minimum at x=3
        let f = |x: f64| (x - 3.0).powi(2);
        
        let x_min = brent_minimize(f, 0.0, 10.0, 1e-8);
        
        assert_abs_diff_eq!(x_min, 3.0, epsilon = 1e-6);
    }

    #[test]
    fn test_theta_higher_for_less_overdispersion() {
        // Less overdispersed data should give higher theta estimate
        let y_low_var = array![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0];
        let y_high_var = array![0.0, 5.0, 0.0, 6.0, 0.0, 4.0, 0.0, 7.0];
        let mu = array![1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5];
        
        let theta_low = estimate_theta_moments(&y_low_var, &mu);
        let theta_high = estimate_theta_moments(&y_high_var, &mu);
        
        // Lower variance -> higher theta (less overdispersion)
        assert!(theta_low > theta_high);
    }
}
