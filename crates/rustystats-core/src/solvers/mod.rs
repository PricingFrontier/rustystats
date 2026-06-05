// =============================================================================
// GLM Solvers
// =============================================================================
//
// This module contains algorithms for fitting Generalized Linear Models.
// The main algorithm is IRLS (Iteratively Reweighted Least Squares).
//
// HOW GLM FITTING WORKS (High-Level Overview)
// -------------------------------------------
//
// We want to find coefficients β that best explain the relationship:
//
//     g(E[Y]) = Xβ
//
// where:
//   - Y is the response variable (what we're predicting)
//   - X is the design matrix (predictors/features)
//   - β is the coefficient vector (what we're solving for)
//   - g is the link function
//   - E[Y] = μ is the expected value of Y
//
// Unlike ordinary least squares, we can't solve this directly because:
//   1. The link function g() makes it non-linear
//   2. The variance depends on μ (heteroscedasticity)
//
// IRLS solves this by iteratively:
//   1. Linearizing the problem around current estimates
//   2. Solving a weighted least squares problem
//   3. Updating estimates and repeating until convergence
//
// =============================================================================

mod coordinate_descent;
pub mod gcv_optimizer;
mod irls;
pub mod nnls;
pub mod smooth_glm;

pub use gcv_optimizer::{brent_minimize, MultiTermGCVOptimizer};
pub use irls::{
    compute_xtwx, compute_xtwx_xtwz, solve_weighted_least_squares_with_penalty_matrix,
    solve_wls_from_precomputed,
};
pub use irls::{fit_glm_unified, FitConfig, IRLSConfig, IRLSResult};
pub use nnls::{
    nnls, nnls_penalized, nnls_weighted, nnls_weighted_penalized, NNLSConfig, NNLSResult,
};
pub use smooth_glm::{
    fit_smooth_glm_full_matrix, Monotonicity, SmoothGLMConfig, SmoothGLMResult, SmoothTermData,
    SmoothTermSpec,
};

use ndarray::{Array1, ArrayView2};
use rayon::prelude::*;

use crate::constants::MAX_IRLS_WEIGHT;
use crate::error::{Result, RustyStatsError};
use crate::families::Family;
use crate::links::Link;

/// Safe initialization of μ that works for any family.
///
/// Used as fallback when `family.initialize_mu(y)` produces invalid values
/// (e.g., all zeros for Poisson). Computes a weighted average of each y_i
/// with the global mean, then clamps to the family's valid range.
pub(crate) fn initialize_mu_safe(y: &Array1<f64>, family: &dyn Family) -> Array1<f64> {
    let y_mean = y.mean().unwrap_or(1.0).max(0.01);
    let raw: Array1<f64> = y.mapv(|yi| (yi + y_mean) / 2.0);
    family.clamp_mu(&raw)
}

// =============================================================================
// Shared GLM Input Validation
// =============================================================================

/// Validated and prepared GLM inputs (offset and weights ready to use).
pub struct ValidatedInputs {
    /// Offset on the linear-predictor scale, defaulted to zeros when absent.
    pub offset: Array1<f64>,
    /// Prior observation weights, defaulted to ones when absent.
    pub prior_weights: Array1<f64>,
}

/// Validate `offset` and `weights` against an expected length and default
/// them when absent. Shared by [`validate_glm_inputs`] and
/// [`validate_residual_inputs`] — keep them in one place so the defaulting
/// rules cannot drift.
fn prepare_offset_weights(
    n: usize,
    offset: Option<&Array1<f64>>,
    weights: Option<&Array1<f64>>,
) -> Result<ValidatedInputs> {
    let offset_vec = match offset {
        Some(o) => {
            if o.len() != n {
                return Err(RustyStatsError::dim_mismatch(
                    n,
                    o.len(),
                    "offset vs y length",
                ));
            }
            o.clone()
        }
        None => Array1::zeros(n),
    };

    let prior_weights_vec = match weights {
        Some(w) => {
            if w.len() != n {
                return Err(RustyStatsError::dim_mismatch(
                    n,
                    w.len(),
                    "weights vs y length",
                ));
            }
            if w.iter().any(|&x| x < 0.0) {
                return Err(RustyStatsError::InvalidValue(
                    "weights must be non-negative".to_string(),
                ));
            }
            w.clone()
        }
        None => Array1::ones(n),
    };

    Ok(ValidatedInputs {
        offset: offset_vec,
        prior_weights: prior_weights_vec,
    })
}

/// Validate GLM inputs and prepare offset/weights.
///
/// Checks dimension compatibility of X, y, offset, and weights.
/// Returns owned, ready-to-use offset and prior_weights arrays.
pub(crate) fn validate_glm_inputs(
    y: &Array1<f64>,
    x: ArrayView2<'_, f64>,
    offset: Option<&Array1<f64>>,
    weights: Option<&Array1<f64>>,
) -> Result<ValidatedInputs> {
    let n = y.len();
    let p = x.ncols();

    if x.nrows() != n {
        return Err(RustyStatsError::dim_mismatch(
            n,
            x.nrows(),
            "X rows vs y length",
        ));
    }

    if n == 0 {
        return Err(RustyStatsError::EmptyInput("y is empty".to_string()));
    }

    if p == 0 {
        return Err(RustyStatsError::EmptyInput("X has no columns".to_string()));
    }

    prepare_offset_weights(n, offset, weights)
}

/// Validate IRLS-residual inputs and prepare offset/weights.
///
/// Sibling of [`validate_glm_inputs`] for callers that operate on a linear
/// predictor `eta` rather than a design matrix `X`. Used by the public
/// `working_response_weights` helper.
pub fn validate_residual_inputs(
    y: &Array1<f64>,
    eta: &Array1<f64>,
    offset: Option<&Array1<f64>>,
    weights: Option<&Array1<f64>>,
) -> Result<ValidatedInputs> {
    let n = y.len();

    if eta.len() != n {
        return Err(RustyStatsError::dim_mismatch(
            n,
            eta.len(),
            "eta length vs y length",
        ));
    }

    if n == 0 {
        return Err(RustyStatsError::EmptyInput("y is empty".to_string()));
    }

    prepare_offset_weights(n, offset, weights)
}

// =============================================================================
// Shared IRLS Weight Computation
// =============================================================================

/// Result of IRLS weight computation for a single iteration.
pub struct IRLSWeightResult {
    /// Per-row IRLS curvature weights before prior weights are applied.
    pub irls_weights: Array1<f64>,
    /// Per-row weights used by WLS: `prior_weight * irls_weight`.
    pub combined_weights: Array1<f64>,
    /// Per-row working response used as the WLS target.
    pub working_response: Array1<f64>,
}

/// Compute IRLS weights, combined weights, and working response in a single parallel pass.
///
/// Supports both Fisher information and true Hessian weighting for Gamma/Tweedie with log link.
///
/// # Errors
///
/// Returns a dimension-mismatch error instead of panicking if any row-oriented
/// input, link derivative, variance, or true-Hessian weight has a length that
/// differs from `y`.
pub fn compute_irls_weights(
    y: &Array1<f64>,
    mu: &Array1<f64>,
    eta: &Array1<f64>,
    offset: &Array1<f64>,
    prior_weights: &Array1<f64>,
    family: &dyn Family,
    link: &dyn Link,
    min_weight: f64,
) -> Result<IRLSWeightResult> {
    let n = y.len();
    if mu.len() != n {
        return Err(RustyStatsError::dim_mismatch(
            n,
            mu.len(),
            "mu length vs y length",
        ));
    }
    if eta.len() != n {
        return Err(RustyStatsError::dim_mismatch(
            n,
            eta.len(),
            "eta length vs y length",
        ));
    }
    if offset.len() != n {
        return Err(RustyStatsError::dim_mismatch(
            n,
            offset.len(),
            "offset length vs y length",
        ));
    }
    if prior_weights.len() != n {
        return Err(RustyStatsError::dim_mismatch(
            n,
            prior_weights.len(),
            "prior_weights length vs y length",
        ));
    }

    if family.name().eq_ignore_ascii_case("poisson") && link.name() == "log" {
        return Ok(compute_poisson_log_irls_weights(
            y,
            mu,
            eta,
            offset,
            prior_weights,
            min_weight,
        ));
    }

    let link_deriv = link.derivative(mu);
    if link_deriv.len() != n {
        return Err(RustyStatsError::dim_mismatch(
            n,
            link_deriv.len(),
            "link derivative length vs y length",
        ));
    }

    let use_true_hessian = family.use_true_hessian_weights() && link.name() == "log";
    let hessian_weights = if use_true_hessian {
        Some(family.true_hessian_weights(mu, y))
    } else {
        None
    };
    let variance = if use_true_hessian {
        None
    } else {
        Some(family.variance(mu))
    };
    if let Some(ref hw) = hessian_weights {
        if hw.len() != n {
            return Err(RustyStatsError::dim_mismatch(
                n,
                hw.len(),
                "true Hessian weights length vs y length",
            ));
        }
    }
    if let Some(ref v) = variance {
        if v.len() != n {
            return Err(RustyStatsError::dim_mismatch(
                n,
                v.len(),
                "variance length vs y length",
            ));
        }
    }

    let mut irls_weights_vec = vec![0.0; n];
    let mut combined_weights_vec = vec![0.0; n];
    let mut working_response_vec = vec![0.0; n];
    let chunk_size = irls_chunk_size(n);
    irls_weights_vec
        .par_chunks_mut(chunk_size)
        .zip(combined_weights_vec.par_chunks_mut(chunk_size))
        .zip(working_response_vec.par_chunks_mut(chunk_size))
        .enumerate()
        .for_each(|(chunk_idx, ((iw_chunk, cw_chunk), wr_chunk))| {
            let start = chunk_idx * chunk_size;
            for local_idx in 0..iw_chunk.len() {
                let i = start + local_idx;
                let d = link_deriv[i];

                let iw = if let Some(ref hw) = hessian_weights {
                    hw[i].max(min_weight).min(MAX_IRLS_WEIGHT)
                } else {
                    let v = variance
                        .as_ref()
                        .expect("variance present in Fisher branch")[i];
                    (1.0 / (v * d * d)).max(min_weight).min(MAX_IRLS_WEIGHT)
                };

                iw_chunk[local_idx] = iw;
                cw_chunk[local_idx] = prior_weights[i] * iw;
                wr_chunk[local_idx] = (eta[i] - offset[i]) + (y[i] - mu[i]) * d;
            }
        });

    Ok(IRLSWeightResult {
        irls_weights: Array1::from_vec(irls_weights_vec),
        combined_weights: Array1::from_vec(combined_weights_vec),
        working_response: Array1::from_vec(working_response_vec),
    })
}

fn compute_poisson_log_irls_weights(
    y: &Array1<f64>,
    mu: &Array1<f64>,
    eta: &Array1<f64>,
    offset: &Array1<f64>,
    prior_weights: &Array1<f64>,
    min_weight: f64,
) -> IRLSWeightResult {
    let n = y.len();
    let mut irls_weights_vec = vec![0.0; n];
    let mut combined_weights_vec = vec![0.0; n];
    let mut working_response_vec = vec![0.0; n];
    let chunk_size = irls_chunk_size(n);
    irls_weights_vec
        .par_chunks_mut(chunk_size)
        .zip(combined_weights_vec.par_chunks_mut(chunk_size))
        .zip(working_response_vec.par_chunks_mut(chunk_size))
        .enumerate()
        .for_each(|(chunk_idx, ((iw_chunk, cw_chunk), wr_chunk))| {
            let start = chunk_idx * chunk_size;
            for local_idx in 0..iw_chunk.len() {
                let i = start + local_idx;
                let mui = mu[i];
                let d = 1.0 / mui;
                let iw = (1.0 / (mui * d * d)).max(min_weight).min(MAX_IRLS_WEIGHT);
                iw_chunk[local_idx] = iw;
                cw_chunk[local_idx] = prior_weights[i] * iw;
                wr_chunk[local_idx] = (eta[i] - offset[i]) + (y[i] - mui) * d;
            }
        });

    IRLSWeightResult {
        irls_weights: Array1::from_vec(irls_weights_vec),
        combined_weights: Array1::from_vec(combined_weights_vec),
        working_response: Array1::from_vec(working_response_vec),
    }
}

fn irls_chunk_size(n: usize) -> usize {
    let target_chunks = rayon::current_num_threads().saturating_mul(4).max(1);
    n.div_ceil(target_chunks).max(1)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    //! Tests for `compute_irls_weights`.
    //!
    //! These pin down the analytic forms of the working response z and the
    //! combined working weight w for each family×link combination supported by
    //! rustystats. They are the contract for the public
    //! `working_response_weights` PyO3 binding.
    //!
    //! Naming convention:
    //!   z = (eta − offset) + (y − μ) · g'(μ)        [working response]
    //!   w = prior_weight × IRLS_weight              [combined weight]
    //!
    //! where IRLS_weight is either the Fisher information weight
    //! `1 / (V(μ) · g'(μ)²)` or, for log-link Tweedie with 1<p<2, the true
    //! Hessian weight `μ^(2−p)`. Gamma+log uses Fisher (w=1) per the
    //! conscious decision in families/gamma.rs to match statsmodels.

    use super::*;
    use crate::constants::MIN_IRLS_WEIGHT;
    use crate::families::{
        BinomialFamily, GammaFamily, GaussianFamily, NegativeBinomialFamily, PoissonFamily,
        QuasiBinomialFamily, QuasiPoissonFamily, TweedieFamily,
    };
    use crate::links::{IdentityLink, LogLink, LogitLink};
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    // ---------------------------------------------------------------------
    // Per-family analytic checks at the canonical link
    // ---------------------------------------------------------------------

    #[test]
    fn gaussian_identity_z_equals_eta_plus_residual() {
        // Gaussian + identity: g'(μ)=1, V(μ)=1
        //   z = eta + (y − μ)·1 = eta + (y − eta) = y          (μ = eta when identity, no offset)
        //   w = 1 / (1 · 1²) = 1
        let y = array![1.0, 2.0, 3.0];
        let eta = array![1.5, 2.0, 2.5];
        let offset = Array1::zeros(3);
        let prior = Array1::ones(3);
        let fam = GaussianFamily;
        let link = IdentityLink;
        let mu = link.inverse(&eta);
        let r = compute_irls_weights(&y, &mu, &eta, &offset, &prior, &fam, &link, MIN_IRLS_WEIGHT)
            .expect("valid IRLS inputs");
        assert_abs_diff_eq!(r.working_response, y, epsilon = 1e-12);
        assert_abs_diff_eq!(r.irls_weights, array![1.0, 1.0, 1.0], epsilon = 1e-12);
        assert_abs_diff_eq!(r.combined_weights, r.irls_weights, epsilon = 1e-12);
    }

    #[test]
    fn poisson_log_analytic_w_equals_mu() {
        // Poisson + log: g'(μ) = 1/μ, V(μ) = μ
        //   z = eta + (y − μ)/μ
        //   w = 1 / (μ · 1/μ²) = μ
        let y = array![1.0, 2.0, 3.0];
        let eta = array![0.0, 1.0_f64.ln(), 3.0_f64.ln()]; // log(1), log(1), log(3) — mixed
        let offset = Array1::zeros(3);
        let prior = Array1::ones(3);
        let fam = PoissonFamily;
        let link = LogLink;
        let mu = link.inverse(&eta);
        let r = compute_irls_weights(&y, &mu, &eta, &offset, &prior, &fam, &link, MIN_IRLS_WEIGHT)
            .expect("valid IRLS inputs");

        let expected_z = &eta + &(&(&y - &mu) / &mu);
        let expected_w = mu.clone();
        assert_abs_diff_eq!(r.working_response, expected_z, epsilon = 1e-12);
        assert_abs_diff_eq!(r.irls_weights, expected_w, epsilon = 1e-12);
    }

    #[test]
    fn gamma_log_fisher_weight_is_one() {
        // Gamma + log: g'(μ) = 1/μ, V(μ) = μ². Fisher branch (use_true_hessian_weights=false):
        //   w = 1 / (μ² · 1/μ²) = 1
        //   z = eta + (y − μ)/μ
        let y = array![1.0, 2.0, 5.0];
        let eta = array![0.5_f64.ln(), 2.0_f64.ln(), 4.0_f64.ln()];
        let offset = Array1::zeros(3);
        let prior = Array1::ones(3);
        let fam = GammaFamily;
        let link = LogLink;
        let mu = link.inverse(&eta);
        let r = compute_irls_weights(&y, &mu, &eta, &offset, &prior, &fam, &link, MIN_IRLS_WEIGHT)
            .expect("valid IRLS inputs");

        assert_abs_diff_eq!(r.irls_weights, array![1.0, 1.0, 1.0], epsilon = 1e-12);
        let expected_z = &eta + &(&(&y - &mu) / &mu);
        assert_abs_diff_eq!(r.working_response, expected_z, epsilon = 1e-12);
    }

    #[test]
    fn tweedie_log_p15_true_hessian_weight_is_sqrt_mu() {
        // Tweedie + log with 1<p<2 uses true Hessian:
        //   w = μ^(2−p)        — for p=1.5, that's μ^0.5
        //   z = eta + (y − μ)/μ
        let y = array![0.5, 2.0, 4.0];
        let eta = array![1.0_f64.ln(), 2.0_f64.ln(), 5.0_f64.ln()];
        let offset = Array1::zeros(3);
        let prior = Array1::ones(3);
        let fam = TweedieFamily::new(1.5).expect("valid var_power");
        let link = LogLink;
        let mu = link.inverse(&eta);
        let r = compute_irls_weights(&y, &mu, &eta, &offset, &prior, &fam, &link, MIN_IRLS_WEIGHT)
            .expect("valid IRLS inputs");

        let expected_w = mu.mapv(|m| m.powf(0.5));
        assert_abs_diff_eq!(r.irls_weights, expected_w, epsilon = 1e-12);
        let expected_z = &eta + &(&(&y - &mu) / &mu);
        assert_abs_diff_eq!(r.working_response, expected_z, epsilon = 1e-12);
    }

    #[test]
    fn binomial_logit_fisher_weight_is_mu_one_minus_mu() {
        // Binomial + logit: g'(μ) = 1/(μ(1−μ)), V(μ) = μ(1−μ)
        //   w = 1 / (μ(1−μ) · 1/(μ(1−μ))²) = μ(1−μ)
        //   z = logit(μ) + (y − μ) · 1/(μ(1−μ)) = eta + (y − μ)/(μ(1−μ))
        let y = array![0.0, 1.0, 1.0];
        let mu = array![0.3, 0.7, 0.5];
        let link = LogitLink;
        let eta = link.link(&mu);
        let offset = Array1::zeros(3);
        let prior = Array1::ones(3);
        let fam = BinomialFamily;
        let r = compute_irls_weights(&y, &mu, &eta, &offset, &prior, &fam, &link, MIN_IRLS_WEIGHT)
            .expect("valid IRLS inputs");

        let expected_w = mu.mapv(|p| p * (1.0 - p));
        assert_abs_diff_eq!(r.irls_weights, expected_w, epsilon = 1e-12);
        let expected_z = &eta + &(&(&y - &mu) / &expected_w);
        assert_abs_diff_eq!(r.working_response, expected_z, epsilon = 1e-12);
    }

    #[test]
    fn negbinomial_log_w_matches_theta_formula() {
        // NB + log with V(μ) = μ + μ²/θ, g'(μ) = 1/μ:
        //   w = 1 / ((μ + μ²/θ) · 1/μ²) = μ² / (μ + μ²/θ) = μθ / (θ + μ)
        let theta = 2.0;
        let y = array![0.0, 1.0, 3.0];
        let eta = array![0.5_f64.ln(), 2.0_f64.ln(), 4.0_f64.ln()];
        let offset = Array1::zeros(3);
        let prior = Array1::ones(3);
        let fam = NegativeBinomialFamily::new(theta).expect("valid theta");
        let link = LogLink;
        let mu = link.inverse(&eta);
        let r = compute_irls_weights(&y, &mu, &eta, &offset, &prior, &fam, &link, MIN_IRLS_WEIGHT)
            .expect("valid IRLS inputs");

        let expected_w = mu.mapv(|m| m * theta / (theta + m));
        assert_abs_diff_eq!(r.irls_weights, expected_w, epsilon = 1e-12);
    }

    #[test]
    fn quasi_poisson_matches_poisson() {
        // QuasiPoisson has the same V(μ) = μ as Poisson → identical irls weights.
        let y = array![0.0, 1.0, 3.0];
        let eta = array![0.5_f64.ln(), 2.0_f64.ln(), 4.0_f64.ln()];
        let offset = Array1::zeros(3);
        let prior = Array1::ones(3);
        let link = LogLink;
        let mu = link.inverse(&eta);

        let poisson = compute_irls_weights(
            &y,
            &mu,
            &eta,
            &offset,
            &prior,
            &PoissonFamily,
            &link,
            MIN_IRLS_WEIGHT,
        )
        .expect("valid IRLS inputs");
        let quasi = compute_irls_weights(
            &y,
            &mu,
            &eta,
            &offset,
            &prior,
            &QuasiPoissonFamily,
            &link,
            MIN_IRLS_WEIGHT,
        )
        .expect("valid IRLS inputs");
        assert_abs_diff_eq!(poisson.irls_weights, quasi.irls_weights, epsilon = 1e-12);
        assert_abs_diff_eq!(
            poisson.working_response,
            quasi.working_response,
            epsilon = 1e-12
        );
    }

    #[test]
    fn quasi_binomial_matches_binomial() {
        // QuasiBinomial has the same V(μ) = μ(1−μ) → identical irls weights to Binomial.
        let y = array![0.0, 1.0, 1.0];
        let mu = array![0.2, 0.6, 0.8];
        let link = LogitLink;
        let eta = link.link(&mu);
        let offset = Array1::zeros(3);
        let prior = Array1::ones(3);

        let binom = compute_irls_weights(
            &y,
            &mu,
            &eta,
            &offset,
            &prior,
            &BinomialFamily,
            &link,
            MIN_IRLS_WEIGHT,
        )
        .expect("valid IRLS inputs");
        let quasi = compute_irls_weights(
            &y,
            &mu,
            &eta,
            &offset,
            &prior,
            &QuasiBinomialFamily,
            &link,
            MIN_IRLS_WEIGHT,
        )
        .expect("valid IRLS inputs");
        assert_abs_diff_eq!(binom.irls_weights, quasi.irls_weights, epsilon = 1e-12);
        assert_abs_diff_eq!(
            binom.working_response,
            quasi.working_response,
            epsilon = 1e-12
        );
    }

    // ---------------------------------------------------------------------
    // Offset / prior weights
    // ---------------------------------------------------------------------

    #[test]
    fn working_response_subtracts_offset_from_eta() {
        // The working response stores (eta − offset) + (y − μ)·g'(μ), so that
        // the offset is *added back* implicitly during the next WLS step.
        // This test confirms that contract — destyler's public helper relies
        // on it (it passes `eta_full = eta + offset` so the public formula
        // z = η + (y − μ)·g'(μ) holds with the user's η-without-offset).
        let y = array![1.0, 2.0];
        let eta = array![1.5, 1.7]; // pretend this is eta_full (incl. offset)
        let offset = array![0.5, 0.7];
        let prior = Array1::ones(2);
        let fam = GaussianFamily;
        let link = IdentityLink;
        let mu = link.inverse(&eta);
        let r = compute_irls_weights(&y, &mu, &eta, &offset, &prior, &fam, &link, MIN_IRLS_WEIGHT)
            .expect("valid IRLS inputs");

        // For Gaussian/identity with eta=mu:
        //   wr = (eta − offset) + (y − mu)·1 = (eta − offset) + (y − eta)
        let expected_z = array![(1.5 - 0.5) + (1.0 - 1.5), (1.7 - 0.7) + (2.0 - 1.7)];
        assert_abs_diff_eq!(r.working_response, expected_z, epsilon = 1e-12);
    }

    #[test]
    fn combined_weights_are_prior_times_irls() {
        // Combined weight is the elementwise product prior_weight × irls_weight.
        let y = array![1.0, 2.0, 3.0];
        let eta = array![0.0, 1.0_f64.ln(), 3.0_f64.ln()];
        let offset = Array1::zeros(3);
        let prior = array![2.0, 0.5, 3.0];
        let fam = PoissonFamily;
        let link = LogLink;
        let mu = link.inverse(&eta);
        let r = compute_irls_weights(&y, &mu, &eta, &offset, &prior, &fam, &link, MIN_IRLS_WEIGHT)
            .expect("valid IRLS inputs");

        let expected_combined = &prior * &r.irls_weights;
        assert_abs_diff_eq!(r.combined_weights, expected_combined, epsilon = 1e-12);
    }

    #[test]
    fn compute_irls_weights_rejects_mismatched_lengths() {
        let y = array![1.0, 2.0, 3.0];
        let mu = array![1.0, 2.0];
        let eta = array![0.0, 1.0_f64.ln(), 3.0_f64.ln()];
        let offset = Array1::zeros(3);
        let prior = Array1::ones(3);
        let fam = PoissonFamily;
        let link = LogLink;

        let err = match compute_irls_weights(
            &y,
            &mu,
            &eta,
            &offset,
            &prior,
            &fam,
            &link,
            MIN_IRLS_WEIGHT,
        ) {
            Ok(_) => panic!("mismatched mu length should be rejected"),
            Err(err) => err,
        };

        match err {
            RustyStatsError::DimensionMismatch {
                expected,
                got,
                context,
            } => {
                assert_eq!(expected, 3);
                assert_eq!(got, 2);
                assert_eq!(context, "mu length vs y length");
            }
            other => panic!("expected DimensionMismatch, got {other:?}"),
        }
    }

    // ---------------------------------------------------------------------
    // Default link helper
    // ---------------------------------------------------------------------

    #[test]
    fn initialize_mu_safe_is_in_valid_range() {
        // Smoke test for the other public helper in this module.
        // For Poisson, clamp_mu enforces μ ≥ MU_MIN_POSITIVE.
        let y = array![0.0, 0.0, 0.0]; // pathological — all zeros
        let mu = initialize_mu_safe(&y, &PoissonFamily);
        assert!(
            mu.iter().all(|&v| v > 0.0),
            "initialize_mu_safe must return strictly positive μ for Poisson, got {:?}",
            mu
        );
    }
}
