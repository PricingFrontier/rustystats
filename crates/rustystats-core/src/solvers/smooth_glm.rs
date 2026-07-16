// =============================================================================
// SMOOTH GLM: Generalized Additive Models with Penalized Splines
// =============================================================================
//
// This module implements GLM fitting with penalized smooth terms (P-splines).
// It extends standard IRLS to automatically select smoothing parameters via GCV.
//
// THE ALGORITHM
// -------------
// For a GAM with smooth terms s(x1), s(x2), ..., we:
//
// 1. Build design matrix X = [parametric | smooth basis columns]
// 2. Build penalty matrix S = block-diag(0, λ₁S₁, λ₂S₂, ...)
// 3. Run penalized IRLS: (X'WX + S)⁻¹ X'Wz at each iteration
// 4. Select λ by minimizing GCV(λ) = n × Deviance / (n - EDF)²
//
// LAMBDA SELECTION STRATEGIES
// ---------------------------
// - Grid search: Evaluate GCV on log-spaced grid
// - Performance iteration: Iterate between IRLS and lambda updates
// - REML: Used for monotonic models (Wood, 2011)
//
// =============================================================================

use ndarray::{s, Array1, Array2, ArrayView2};

use crate::constants::{
    MAX_IRLS_WEIGHT, SMOOTH_CONVERGENCE_TOL, SMOOTH_INNER_MAX_PER_CYCLE, SMOOTH_KKT_SCORE_TOL,
    SMOOTH_WARM_START_MAX_ITER, ZERO_TOL,
};
use crate::convert;
use crate::error::{Result, RustyStatsError};
use crate::families::Family;
use crate::links::Link;
use crate::regularization::{Penalty, SmoothPenalty};
use crate::solvers::irls::{
    compute_xtwx, compute_xtwx_xtwz, fit_glm_unified,
    solve_weighted_least_squares_with_penalty_matrix, solve_wls_from_precomputed, FitConfig,
    IRLSConfig,
};
use crate::splines::penalized::{compute_edf, gcv_score, penalty_matrix};

/// Maximum absolute value for alpha parameters in exp reparameterization.
/// Prevents exp() overflow: exp(20) ~ 4.9e8 which is large but finite.
const MAX_EXP_ALPHA: f64 = 20.0;

/// Embed a scaled penalty sub-matrix into a larger penalty matrix.
/// `target[offset..offset+k, offset..offset+k] += scale * source`
fn embed_penalty(target: &mut Array2<f64>, source: &Array2<f64>, offset: usize, scale: f64) {
    let k = source.nrows();
    let mut slice = target.slice_mut(s![offset..offset + k, offset..offset + k]);
    slice.scaled_add(scale, source);
}

/// Result from fitting a smooth GLM (GAM).
#[derive(Debug, Clone)]
pub struct SmoothGLMResult {
    /// Fitted coefficients (parametric + smooth basis)
    pub coefficients: Array1<f64>,

    /// Fitted values μ = g⁻¹(Xβ + offset)
    pub fitted_values: Array1<f64>,

    /// Linear predictor η = Xβ + offset
    pub linear_predictor: Array1<f64>,

    /// Final deviance
    pub deviance: f64,

    /// Number of IRLS iterations
    pub iterations: usize,

    /// Did the algorithm converge?
    pub converged: bool,

    /// Selected smoothing parameters (one per smooth term)
    pub lambdas: Vec<f64>,

    /// Effective degrees of freedom (one per smooth term)
    pub smooth_edfs: Vec<f64>,

    /// Total effective degrees of freedom (parametric + smooth)
    pub total_edf: f64,

    /// GCV score at selected lambdas
    pub gcv: f64,

    /// Unscaled covariance matrix (X'WX + S)⁻¹
    pub covariance_unscaled: Array2<f64>,

    /// Family name
    pub family_name: String,

    /// The smooth penalty configuration
    pub penalty: Penalty,

    /// IRLS weights from the final iteration (for robust SEs)
    pub irls_weights: Array1<f64>,

    /// Prior weights
    pub prior_weights: Array1<f64>,

    /// Combined design matrix [parametric | smooth]
    pub design_matrix: Array2<f64>,

    /// Original response variable (for residuals/diagnostics)
    pub y: Array1<f64>,

    /// Offset values (if any)
    pub offset: Option<Array1<f64>>,

    /// Warnings collected during fitting.
    pub warnings: Vec<String>,

    /// Whether step-halving produced an accepted step at any iteration.
    pub step_halving_used: bool,

    /// Terminal solver status.
    pub solver_status: String,

    /// Whether the returned fit satisfies first-order (KKT) stationarity for
    /// every unpenalized, unconstrained coordinate — the intercept above all.
    /// `converged` is gated on this: a progress-based inner convergence test
    /// alone cannot claim convergence at a non-stationary point.
    pub stationary: bool,

    /// Largest standardized score |s_j| / sqrt(I_jj) over the coordinates
    /// covered by the stationarity check (0.0 when nothing is checked).
    pub max_std_score: f64,
}

/// Configuration for smooth GLM fitting.
#[derive(Debug, Clone)]
pub struct SmoothGLMConfig {
    /// Base IRLS configuration
    pub irls_config: IRLSConfig,

    /// Number of lambda values to evaluate in grid search
    pub n_lambda: usize,

    /// Minimum lambda value (log scale)
    pub lambda_min: f64,

    /// Maximum lambda value (log scale)
    pub lambda_max: f64,

    /// Convergence tolerance for lambda optimization
    pub lambda_tol: f64,

    /// Maximum iterations for lambda optimization (outer loop)
    pub max_lambda_iter: usize,

    /// Method for lambda selection: "gcv" or "fixed"
    pub lambda_method: String,
}

impl Default for SmoothGLMConfig {
    fn default() -> Self {
        Self {
            irls_config: IRLSConfig::default(),
            n_lambda: 30,
            lambda_min: 1e-4,
            lambda_max: 1e6,
            lambda_tol: 1e-4,
            max_lambda_iter: 20,
            lambda_method: "gcv".to_string(),
        }
    }
}

/// Monotonicity constraint for smooth terms.
///
/// Enforced via exp reparameterization (Pya & Wood 2015, scam/mgcv approach):
/// - `beta_1 = alpha_1` (free intercept)
/// - `beta_j = beta_{j-1} + exp(alpha_j)` for `Increasing`
/// - `beta_j = beta_{j-1} - exp(alpha_j)` for `Decreasing`
///
/// This makes optimization unconstrained in alpha-space while guaranteeing monotonicity.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Monotonicity {
    /// No constraint
    #[default]
    None,
    /// Monotonically increasing
    Increasing,
    /// Monotonically decreasing
    Decreasing,
}

// =============================================================================
// Exp reparameterization for monotonic splines (Pya & Wood 2015)
// =============================================================================

/// Convert unconstrained alpha parameters to monotonic beta coefficients.
///
/// - `beta[0] = alpha[0]`
/// - `beta[j] = beta[j-1] + sign * exp(alpha[j])` for j >= 1
///
/// where `sign = +1` for increasing, `-1` for decreasing.
fn alpha_to_beta(alpha: &[f64], monotonicity: &Monotonicity) -> Array1<f64> {
    let k = alpha.len();
    let mut beta = Array1::zeros(k);
    if k == 0 {
        return beta;
    }
    let sign = match monotonicity {
        Monotonicity::Increasing => 1.0,
        Monotonicity::Decreasing => -1.0,
        Monotonicity::None => {
            // No transform needed
            return Array1::from_vec(alpha.to_vec());
        }
    };
    beta[0] = alpha[0];
    for j in 1..k {
        // Clamp alpha to prevent exp() overflow
        let clamped = alpha[j].clamp(-MAX_EXP_ALPHA, MAX_EXP_ALPHA);
        beta[j] = beta[j - 1] + sign * clamped.exp();
    }
    beta
}

/// Project a sequence onto the monotone cone using the Pool Adjacent
/// Violators (PAV) algorithm. Returns the closest (L2) non-decreasing
/// sequence if `increasing` is true, or non-increasing if false.
fn isotonic_projection(x: &[f64], increasing: bool) -> Vec<f64> {
    let n = x.len();
    if n == 0 {
        return vec![];
    }
    let mut result = x.to_vec();
    if !increasing {
        for v in result.iter_mut() {
            *v = -*v;
        }
    }
    // PAV: merge adjacent violators into their weighted average
    let mut blocks: Vec<(f64, usize)> = Vec::with_capacity(n); // (sum, count)
    for &val in result.iter() {
        blocks.push((val, 1));
        while blocks.len() >= 2 {
            let len = blocks.len();
            let avg_last = block_average(blocks[len - 1].0, blocks[len - 1].1);
            let avg_prev = block_average(blocks[len - 2].0, blocks[len - 2].1);
            if pav_blocks_should_merge(avg_last, avg_prev) {
                let last = blocks.pop().expect("PAV: blocks non-empty after push");
                let prev = blocks.last_mut().expect("PAV: blocks non-empty after pop");
                prev.0 += last.0;
                prev.1 += last.1;
            } else {
                break;
            }
        }
    }
    let mut idx = 0;
    for (sum, count) in &blocks {
        let avg = block_average(*sum, *count);
        for _ in 0..*count {
            result[idx] = avg;
            idx += 1;
        }
    }
    if !increasing {
        for v in result.iter_mut() {
            *v = -*v;
        }
    }
    result
}

fn block_average(sum: f64, count: usize) -> f64 {
    sum / count as f64
}

fn pav_blocks_should_merge(avg_last: f64, avg_prev: f64) -> bool {
    avg_last < avg_prev
}

fn project_coefficients_to_bounds(
    coefficients: &mut Array1<f64>,
    nonneg_indices: Option<&[usize]>,
    nonpos_indices: Option<&[usize]>,
) {
    if let Some(indices) = nonneg_indices {
        for &idx in indices {
            if idx < coefficients.len() && coefficients[idx] < 0.0 {
                coefficients[idx] = 0.0;
            }
        }
    }
    if let Some(indices) = nonpos_indices {
        for &idx in indices {
            if idx < coefficients.len() && coefficients[idx] > 0.0 {
                coefficients[idx] = 0.0;
            }
        }
    }
}

fn irls_work_weight(variance: f64, link_derivative: f64, min_weight: f64) -> f64 {
    (1.0 / (variance * link_derivative * link_derivative))
        .max(min_weight)
        .min(MAX_IRLS_WEIGHT)
}

fn working_response_value(eta: f64, offset: f64, y: f64, mu: f64, link_derivative: f64) -> f64 {
    (eta - offset) + (y - mu) * link_derivative
}

fn update_irls_work_arrays(
    eta: &Array1<f64>,
    offset: &Array1<f64>,
    y: &Array1<f64>,
    mu: &Array1<f64>,
    prior_weights: &Array1<f64>,
    variance: &Array1<f64>,
    link_derivative: &Array1<f64>,
    min_weight: f64,
    irls_weights: &mut Array1<f64>,
    combined_weights: &mut Array1<f64>,
    working_response: &mut Array1<f64>,
) {
    for i in 0..eta.len() {
        irls_weights[i] = irls_work_weight(variance[i], link_derivative[i], min_weight);
        combined_weights[i] = prior_weights[i] * irls_weights[i];
        working_response[i] =
            working_response_value(eta[i], offset[i], y[i], mu[i], link_derivative[i]);
    }
}

fn weighted_square(value: f64, weight: f64) -> f64 {
    weight * value * value
}

fn weighted_response_sum_squares(response: &Array1<f64>, weights: &Array1<f64>) -> f64 {
    response
        .iter()
        .zip(weights.iter())
        .map(|(&value, &weight)| weighted_square(value, weight))
        .sum()
}

fn blend_arrays(old: &Array1<f64>, new: &Array1<f64>, fraction: f64) -> Array1<f64> {
    old * (1.0 - fraction) + new * fraction
}

fn blend_step_arrays(old: &Array1<f64>, new: &Array1<f64>, step: f64) -> Array1<f64> {
    if step >= 1.0 {
        new.clone()
    } else {
        blend_arrays(old, new, step)
    }
}

fn smooth_step_fraction_for_trial(trial: i32, final_trial: i32) -> f64 {
    if trial == final_trial {
        0.0
    } else {
        0.5_f64.powi(trial)
    }
}

fn next_smooth_half_step(step: f64) -> f64 {
    step * 0.5
}

fn smooth_accept_threshold(old_penalized_deviance: f64, tolerance: f64) -> f64 {
    old_penalized_deviance + tolerance * (1.0 + old_penalized_deviance.abs())
}

fn finite_array(values: &Array1<f64>) -> bool {
    values.iter().all(|value| value.is_finite())
}

fn initial_projection_if_finite(projected: Array1<f64>) -> Option<Array1<f64>> {
    if finite_array(&projected) {
        Some(projected)
    } else {
        None
    }
}

fn smooth_trial_values_are_finite(
    eta: &Array1<f64>,
    mu: &Array1<f64>,
    deviance: f64,
    penalized_deviance: f64,
) -> bool {
    finite_array(eta) && finite_array(mu) && deviance.is_finite() && penalized_deviance.is_finite()
}

fn smooth_step_candidate_accepted(
    eta: &Array1<f64>,
    mu: &Array1<f64>,
    deviance: f64,
    penalized_deviance: f64,
    accept_threshold: f64,
) -> bool {
    smooth_trial_values_are_finite(eta, mu, deviance, penalized_deviance)
        && penalized_deviance <= accept_threshold
}

fn should_skip_smooth_trial(
    eta: &Array1<f64>,
    mu: &Array1<f64>,
    deviance: f64,
    penalized_deviance: f64,
) -> bool {
    !smooth_trial_values_are_finite(eta, mu, deviance, penalized_deviance)
}

fn smooth_trial_is_better(best_trial: Option<i32>, trial_pen_dev: f64, best_pen_dev: f64) -> bool {
    best_trial.is_none() || trial_pen_dev < best_pen_dev
}

fn smooth_trial_is_nonworsening(penalized_deviance: f64, old_penalized_deviance: f64) -> bool {
    penalized_deviance <= old_penalized_deviance
}

fn immediate_trial_accepts(trial: i32, trial_pen_dev: f64, old_penalized_deviance: f64) -> bool {
    trial == 0 && smooth_trial_is_nonworsening(trial_pen_dev, old_penalized_deviance)
}

fn halved_trial_accepts(trial: i32, best_pen_dev: f64, old_penalized_deviance: f64) -> bool {
    trial > 0 && smooth_trial_is_nonworsening(best_pen_dev, old_penalized_deviance)
}

fn smooth_step_accepted(best_trial: Option<i32>, best_pen_dev: f64, accept_threshold: f64) -> bool {
    best_trial.is_some() && best_pen_dev <= accept_threshold
}

fn smooth_step_halving_failed(
    best_trial: Option<i32>,
    best_pen_dev: f64,
    accept_threshold: f64,
) -> bool {
    !smooth_step_accepted(best_trial, best_pen_dev, accept_threshold)
}

fn genuine_halved_step(best_trial: Option<i32>, final_trial: i32) -> bool {
    matches!(best_trial, Some(trial) if trial > 0 && trial < final_trial)
}

fn relative_change_with_unit_offset(old: f64, new: f64) -> f64 {
    (old - new).abs() / (1.0 + old.abs())
}

fn relative_change_with_floor(old: f64, new: f64, floor: f64) -> f64 {
    if old.abs() > floor {
        (old - new).abs() / old.abs()
    } else {
        (old - new).abs()
    }
}

fn change_below_tolerance(change: f64, tolerance: f64) -> bool {
    change < tolerance
}

fn lambda_relative_change(old: f64, new: f64) -> f64 {
    relative_change_with_floor(old, new, 1e-12)
}

fn max_lambda_relative_change(old_lambdas: &[f64], new_lambdas: &[f64]) -> f64 {
    old_lambdas
        .iter()
        .zip(new_lambdas.iter())
        .map(|(&old, &new)| lambda_relative_change(old, new))
        .fold(0.0f64, f64::max)
}

fn lambdas_stable(max_lambda_change: f64) -> bool {
    change_below_tolerance(max_lambda_change, 0.01)
}

fn should_stop_monotonic_outer(max_lambda_change: f64, converged: bool) -> bool {
    lambdas_stable(max_lambda_change) && converged
}

fn should_run_smooth_gcv(lambdas_stable_count: u32, iteration: usize) -> bool {
    lambdas_stable_count < 1 && (iteration <= 3 || iteration.is_multiple_of(2))
}

fn next_lambdas_stable_count(max_rel_change: f64, current: u32) -> u32 {
    if lambdas_stable(max_rel_change) {
        current + 1
    } else {
        0
    }
}

fn lambdas_changed(lambdas: &[f64], previous_lambdas: &[f64]) -> bool {
    lambdas != previous_lambdas
}

fn parametric_column_count(total_cols: usize, smooth_col_count: usize) -> usize {
    total_cols - smooth_col_count
}

fn smooth_term_width(spec: &SmoothTermSpec) -> usize {
    spec.col_end - spec.col_start
}

fn centered_eta(eta: &Array1<f64>, offset: &Array1<f64>) -> Array1<f64> {
    eta - offset
}

fn linear_predictor_from_coefficients(
    x: ArrayView2<'_, f64>,
    coefficients: &Array1<f64>,
    offset: &Array1<f64>,
) -> Array1<f64> {
    &x.dot(coefficients) + offset
}

fn smooth_coefficient_index(col_start: usize, offset: usize) -> usize {
    col_start + offset
}

fn fraction_is_genuine_halving(fraction: f64) -> bool {
    fraction > 0.0
}

fn clamp_monotonic_alpha_value(position: usize, alpha: f64) -> f64 {
    if position == 0 {
        alpha
    } else {
        alpha.clamp(-MAX_EXP_ALPHA, MAX_EXP_ALPHA)
    }
}

fn smooth_solver_status(
    step_halving_failed: bool,
    converged: bool,
    stationary: bool,
) -> &'static str {
    if step_halving_failed {
        "step_halving_no_improvement"
    } else if converged && stationary {
        "converged"
    } else if converged {
        // The progress test passed but the exit KKT check did not: the solver
        // stalled at a non-stationary point and must not report convergence.
        "stalled_nonstationary"
    } else {
        "max_iterations"
    }
}

fn should_warn_smooth_nonconvergence(converged: bool) -> bool {
    !converged
}

fn smooth_nonconvergence_warning(solver_status: &str) -> String {
    format!("Smooth GLM did not converge (status: {solver_status}). Results may be approximate.")
}

// =============================================================================
// Exit stationarity (KKT) check and intercept refresh (termination contract)
// =============================================================================

/// Locate the intercept: the first column outside all smooth ranges and all
/// sign-constraint sets whose entries are exactly 1.0.
fn find_intercept_column(
    x: ArrayView2<'_, f64>,
    smooth_cols: &std::collections::HashSet<usize>,
    nonneg_indices: Option<&[usize]>,
    nonpos_indices: Option<&[usize]>,
) -> Option<usize> {
    let constrained: std::collections::HashSet<usize> = nonneg_indices
        .unwrap_or(&[])
        .iter()
        .chain(nonpos_indices.unwrap_or(&[]).iter())
        .copied()
        .collect();
    (0..x.ncols())
        .filter(|c| !smooth_cols.contains(c) && !constrained.contains(c))
        .find(|&c| x.column(c).iter().all(|&v| v == 1.0))
}

/// Guard a quasi-score denominator away from zero while preserving its sign.
fn guarded_denominator(denom: f64) -> f64 {
    if denom.abs() < ZERO_TOL {
        ZERO_TOL.copysign(denom)
    } else {
        denom
    }
}

/// Per-observation quasi-score residuals r_i = w_i (y_i - μ_i) / (g'(μ_i) V(μ_i))
/// and expected-information weights f_i = w_i / (g'(μ_i)² V(μ_i)).
///
/// The score for coefficient j is Σ_i r_i x_ij and its information diagonal is
/// Σ_i f_i x_ij². For canonical links g'(μ) V(μ) = 1, so the intercept score
/// reduces to Σ w (y - μ): the mean-prediction identity.
fn quasi_score_arrays(
    y: &Array1<f64>,
    mu: &Array1<f64>,
    prior_weights: &Array1<f64>,
    family: &dyn Family,
    link: &dyn Link,
) -> (Array1<f64>, Array1<f64>) {
    let link_deriv = link.derivative(mu);
    let variance = family.variance(mu);
    let n = y.len();
    let mut score_resid = Array1::zeros(n);
    let mut info_weight = Array1::zeros(n);
    for i in 0..n {
        let denom = guarded_denominator(link_deriv[i] * variance[i]);
        score_resid[i] = prior_weights[i] * (y[i] - mu[i]) / denom;
        let info_denom = (link_deriv[i] * link_deriv[i] * variance[i]).max(ZERO_TOL);
        info_weight[i] = prior_weights[i] / info_denom;
    }
    (score_resid, info_weight)
}

/// Score and standardized score of a single design column.
fn column_std_score(
    x: ArrayView2<'_, f64>,
    col: usize,
    score_resid: &Array1<f64>,
    info_weight: &Array1<f64>,
) -> (f64, f64) {
    let mut score = 0.0;
    let mut info = 0.0;
    for i in 0..x.nrows() {
        let v = x[[i, col]];
        score += score_resid[i] * v;
        info += info_weight[i] * v * v;
    }
    (score, score.abs() / info.max(ZERO_TOL).sqrt())
}

/// KKT stationarity report over the unpenalized coordinates.
///
/// At any (constrained) local optimum, the score of every unpenalized,
/// unconstrained coordinate must vanish; sign-constrained coordinates at an
/// active bound may only carry a score pushing INTO the constraint
/// (complementary slackness). Penalized (smooth-basis) coordinates balance
/// the penalty gradient instead and are not checked here.
///
/// Returns `(stationary, max_std_score)`.
#[allow(clippy::too_many_arguments)]
fn stationarity_report(
    x: ArrayView2<'_, f64>,
    y: &Array1<f64>,
    mu: &Array1<f64>,
    prior_weights: &Array1<f64>,
    family: &dyn Family,
    link: &dyn Link,
    smooth_cols: &std::collections::HashSet<usize>,
    coefficients: &Array1<f64>,
    nonneg_indices: Option<&[usize]>,
    nonpos_indices: Option<&[usize]>,
) -> (bool, f64) {
    let (score_resid, info_weight) = quasi_score_arrays(y, mu, prior_weights, family, link);
    let nonneg: std::collections::HashSet<usize> =
        nonneg_indices.unwrap_or(&[]).iter().copied().collect();
    let nonpos: std::collections::HashSet<usize> =
        nonpos_indices.unwrap_or(&[]).iter().copied().collect();

    let mut max_std_score = 0.0f64;
    for col in 0..x.ncols() {
        if smooth_cols.contains(&col) {
            continue;
        }
        let (score, std_score) = column_std_score(x, col, &score_resid, &info_weight);
        let at_lower_bound = nonneg.contains(&col) && coefficients[col] <= ZERO_TOL;
        let at_upper_bound = nonpos.contains(&col) && coefficients[col] >= -ZERO_TOL;
        if (at_lower_bound && score <= 0.0) || (at_upper_bound && score >= 0.0) {
            continue;
        }
        max_std_score = max_std_score.max(std_score);
    }
    (max_std_score < SMOOTH_KKT_SCORE_TOL, max_std_score)
}

/// Final unconstrained-block re-solve (termination contract, bug.md Fix 1).
///
/// Holding the monotone (exp-reparameterised) blocks and sign-constrained
/// columns fixed as an offset, the remaining coordinates — intercept,
/// parametric columns, and unconstrained penalized smooths — form a penalized
/// GLM sub-problem that is convex for canonical links. The stalled main loop
/// can leave these coordinates non-stationary (its inner test measures
/// progress, not stationarity); solving the sub-problem to convergence
/// removes that residual bias. This is a block coordinate-descent step, so it
/// can only improve the full penalized objective; it is accepted only when it
/// does not worsen it.
///
/// Returns `Ok(true)` when a refinement was applied.
#[allow(clippy::too_many_arguments)]
fn refine_unconstrained_block(
    y: &Array1<f64>,
    x_combined: ArrayView2<'_, f64>,
    smooth_specs: &[SmoothTermSpec],
    lambdas: &[f64],
    family: &dyn Family,
    link: &dyn Link,
    offset_vec: &Array1<f64>,
    prior_weights: &Array1<f64>,
    min_weight: f64,
    smooth_tolerance: f64,
    nonneg_indices: Option<&[usize]>,
    nonpos_indices: Option<&[usize]>,
    coefficients: &mut Array1<f64>,
    eta: &mut Array1<f64>,
    mu: &mut Array1<f64>,
    deviance: &mut f64,
    iteration: &mut usize,
    budget: usize,
) -> Result<bool> {
    let mut fixed_cols = std::collections::HashSet::new();
    for spec in smooth_specs {
        if spec.is_monotonic() {
            for col in spec.col_start..spec.col_end {
                fixed_cols.insert(col);
            }
        }
    }
    for &col in nonneg_indices.unwrap_or(&[]) {
        fixed_cols.insert(col);
    }
    for &col in nonpos_indices.unwrap_or(&[]) {
        fixed_cols.insert(col);
    }

    let total_cols = x_combined.ncols();
    let refit_cols: Vec<usize> = (0..total_cols)
        .filter(|col| !fixed_cols.contains(col))
        .collect();
    if refit_cols.is_empty() || fixed_cols.is_empty() || *iteration >= budget {
        // Nothing to refine, or the whole problem was already the
        // unconstrained block (handled by the main loop), or no budget left.
        return Ok(false);
    }

    let n = y.len();
    let n_refit = refit_cols.len();

    // Sub-design of the refit columns; fixed columns become a link-scale offset.
    let mut x_sub = Array2::zeros((n, n_refit));
    for (j_new, &j_old) in refit_cols.iter().enumerate() {
        x_sub.column_mut(j_new).assign(&x_combined.column(j_old));
    }
    let mut offset_fixed = offset_vec.clone();
    for &col in &fixed_cols {
        let coef = coefficients[col];
        if coef != 0.0 {
            offset_fixed.scaled_add(coef, &x_combined.column(col));
        }
    }

    // Penalties of the unconstrained smooth terms, re-indexed into sub-columns.
    // Monotone ranges and sign-constrained columns are whole features, so a
    // non-monotone smooth range stays contiguous after removal.
    let mut penalty_sub = Array2::zeros((n_refit, n_refit));
    for (i, spec) in smooth_specs.iter().enumerate() {
        if spec.is_monotonic() {
            continue;
        }
        let removed_before = (0..spec.col_start)
            .filter(|col| fixed_cols.contains(col))
            .count();
        embed_penalty(
            &mut penalty_sub,
            &spec.penalty,
            spec.col_start - removed_before,
            lambdas[i],
        );
    }

    let write_back = |beta_sub: &Array1<f64>, base: &Array1<f64>| -> Array1<f64> {
        let mut full = base.clone();
        for (j_new, &j_old) in refit_cols.iter().enumerate() {
            full[j_old] = beta_sub[j_new];
        }
        full
    };

    let pen_dev_start = smooth_penalized_deviance(*deviance, coefficients, smooth_specs, lambdas);

    let mut beta_sub: Array1<f64> = refit_cols.iter().map(|&c| coefficients[c]).collect();
    let mut eta_cur = eta.clone();
    let mut mu_cur = mu.clone();
    let mut dev_cur = *deviance;
    let mut pen_dev_cur = pen_dev_start;

    let mut irls_weights = Array1::zeros(n);
    let mut combined_weights = Array1::zeros(n);
    let mut working_response = Array1::zeros(n);

    while *iteration < budget {
        *iteration += 1;

        let link_deriv = link.derivative(&mu_cur);
        let variance = family.variance(&mu_cur);
        update_irls_work_arrays(
            &eta_cur,
            &offset_fixed,
            y,
            &mu_cur,
            prior_weights,
            &variance,
            &link_deriv,
            min_weight,
            &mut irls_weights,
            &mut combined_weights,
            &mut working_response,
        );
        let (xtwx, xtwz) = compute_xtwx_xtwz(x_sub.view(), &working_response, &combined_weights)?;
        let (beta_new, _) = solve_wls_from_precomputed(&xtwx, &xtwz, &penalty_sub, true)?;

        // Step halving on the full penalized objective (convex sub-problem:
        // the full step almost always accepts; the 0-fraction fallback exactly
        // retains the previous iterate).
        let mut accepted = false;
        let mut pen_dev_next = pen_dev_cur;
        for trial in 0..=10 {
            let fraction = smooth_step_fraction_for_trial(trial, 10);
            let beta_trial = blend_step_arrays(&beta_sub, &beta_new, fraction);
            let full_trial = write_back(&beta_trial, coefficients);
            let eta_trial = &x_sub.dot(&beta_trial) + &offset_fixed;
            let mu_trial = family.clamp_mu(&link.inverse(&eta_trial));
            let dev_trial = family.deviance(y, &mu_trial, Some(prior_weights));
            let pen_dev_trial =
                smooth_penalized_deviance(dev_trial, &full_trial, smooth_specs, lambdas);
            if should_skip_smooth_trial(&eta_trial, &mu_trial, dev_trial, pen_dev_trial) {
                continue;
            }
            if pen_dev_trial <= smooth_accept_threshold(pen_dev_cur, smooth_tolerance) {
                beta_sub = beta_trial;
                eta_cur = eta_trial;
                mu_cur = mu_trial;
                dev_cur = dev_trial;
                pen_dev_next = pen_dev_trial;
                accepted = true;
                break;
            }
        }
        if !accepted {
            break;
        }

        let rel_change = relative_change_with_unit_offset(pen_dev_cur, pen_dev_next);
        pen_dev_cur = pen_dev_next;
        if change_below_tolerance(rel_change, smooth_tolerance) {
            break;
        }
    }

    if pen_dev_cur <= pen_dev_start + smooth_tolerance * (1.0 + pen_dev_start.abs()) {
        *coefficients = write_back(&beta_sub, coefficients);
        *eta = eta_cur;
        *mu = mu_cur;
        *deviance = dev_cur;
        Ok(true)
    } else {
        Ok(false)
    }
}

/// Intercept-direction quasi-score at shift `d`: Σ w (y - μ(η + d)) / (g' V).
fn intercept_shift_score(
    shift: f64,
    y: &Array1<f64>,
    eta: &Array1<f64>,
    prior_weights: &Array1<f64>,
    family: &dyn Family,
    link: &dyn Link,
) -> f64 {
    let eta_shifted = eta.mapv(|e| e + shift);
    let mu_shifted = family.clamp_mu(&link.inverse(&eta_shifted));
    let link_deriv = link.derivative(&mu_shifted);
    let variance = family.variance(&mu_shifted);
    let mut score = 0.0;
    for i in 0..y.len() {
        let denom = guarded_denominator(link_deriv[i] * variance[i]);
        score += prior_weights[i] * (y[i] - mu_shifted[i]) / denom;
    }
    score
}

/// Solve `intercept_shift_score(d) = 0` by bracketing bisection.
///
/// This is the exact one-dimensional MLE along the intercept direction, so
/// applying the root cannot materially worsen the deviance. Returns `None`
/// when no sign change exists within |d| <= 64 (e.g. quasi-separation) or the
/// score is non-finite.
fn solve_intercept_shift(
    y: &Array1<f64>,
    eta: &Array1<f64>,
    prior_weights: &Array1<f64>,
    family: &dyn Family,
    link: &dyn Link,
) -> Option<f64> {
    let score_at = |d: f64| intercept_shift_score(d, y, eta, prior_weights, family, link);

    let score_zero = score_at(0.0);
    if !score_zero.is_finite() {
        return None;
    }
    if score_zero == 0.0 {
        return Some(0.0);
    }

    let mut span = 1.0f64;
    let (mut lo, mut hi) = (-span, span);
    let (mut score_lo, mut score_hi) = (score_at(lo), score_at(hi));
    while score_lo.is_finite()
        && score_hi.is_finite()
        && score_lo.signum() == score_hi.signum()
        && span < 64.0
    {
        span *= 2.0;
        lo = -span;
        hi = span;
        score_lo = score_at(lo);
        score_hi = score_at(hi);
    }
    if !score_lo.is_finite() || !score_hi.is_finite() || score_lo.signum() == score_hi.signum() {
        return None;
    }

    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        let score_mid = score_at(mid);
        if !score_mid.is_finite() {
            return None;
        }
        if score_mid == 0.0 {
            return Some(mid);
        }
        if score_mid.signum() == score_lo.signum() {
            lo = mid;
            score_lo = score_mid;
        } else {
            hi = mid;
        }
        if (hi - lo).abs() < 1e-13 * (1.0 + hi.abs().max(lo.abs())) {
            break;
        }
    }
    Some(0.5 * (lo + hi))
}

fn smooth_penalty_contribution(
    lambda: f64,
    coeffs: ndarray::ArrayView1<'_, f64>,
    penalty: &Array2<f64>,
) -> f64 {
    lambda * coeffs.dot(&penalty.dot(&coeffs))
}

fn smooth_penalized_deviance(
    raw_deviance: f64,
    coefficients: &Array1<f64>,
    smooth_specs: &[SmoothTermSpec],
    lambdas: &[f64],
) -> f64 {
    let mut penalized = raw_deviance;
    let cs = coefficients.as_slice().expect("contiguous");
    for (idx, spec) in smooth_specs.iter().enumerate() {
        let coeffs = ndarray::ArrayView1::from(&cs[spec.col_start..spec.col_end]);
        penalized += smooth_penalty_contribution(lambdas[idx], coeffs, &spec.penalty);
    }
    penalized
}

fn add_penalty_to_xtwx(xtwx: &Array2<f64>, penalty_matrix: &Array2<f64>) -> Array2<f64> {
    xtwx + penalty_matrix
}

fn monotonic_offset(beta: &Array1<f64>, j_alpha: &Array1<f64>) -> Array1<f64> {
    beta - j_alpha
}

fn subtract_smooth_offset(
    adjusted_response: &mut Array1<f64>,
    x_smooth: ArrayView2<'_, f64>,
    offset_mono: &Array1<f64>,
) {
    let offset_slice = offset_mono.as_slice().expect("contiguous array");
    let adj_slice = adjusted_response.as_slice_mut().expect("contiguous array");
    for row in 0..x_smooth.nrows() {
        let mut dot = 0.0;
        for col in 0..x_smooth.ncols() {
            dot += x_smooth[[row, col]] * offset_slice[col];
        }
        adj_slice[row] -= dot;
    }
}

/// Convert monotonic beta coefficients back to unconstrained alpha parameters.
///
/// Before conversion, betas are projected onto the monotone cone via isotonic
/// regression (PAV). This gives the closest L2 monotonic approximation,
/// avoiding extreme alpha values that would freeze parameters.
///
/// - `alpha[0] = beta_proj[0]`
/// - `alpha[j] = ln(beta_proj[j] - beta_proj[j-1])` for increasing, or
///   `ln(beta_proj[j-1] - beta_proj[j])` for decreasing
fn beta_to_alpha(beta: &[f64], monotonicity: &Monotonicity) -> Array1<f64> {
    let k = beta.len();
    let mut alpha = Array1::zeros(k);
    if k == 0 {
        return alpha;
    }
    if *monotonicity == Monotonicity::None {
        return Array1::from_vec(beta.to_vec());
    }
    let increasing = *monotonicity == Monotonicity::Increasing;
    let sign = if increasing { 1.0 } else { -1.0 };

    // Project onto monotone cone for a clean initialization
    let mono_beta = isotonic_projection(beta, increasing);

    alpha[0] = mono_beta[0];
    for j in 1..k {
        let diff = sign * (mono_beta[j] - mono_beta[j - 1]);
        alpha[j] = diff.max(ZERO_TOL).ln();
    }
    alpha
}

/// Build the lower-triangular Jacobian matrix J for the exp reparameterization.
///
/// J is k x k with:
/// - `J[i, 0] = 1` for all i
/// - `J[i, j] = sign * exp(alpha[j])` for i >= j >= 1
/// - `J[i, j] = 0` for i < j
///
/// This represents d(beta)/d(alpha).
fn compute_monotonic_jacobian(alpha: &[f64], monotonicity: &Monotonicity) -> Array2<f64> {
    let k = alpha.len();
    let mut j_mat = Array2::zeros((k, k));
    let sign = match monotonicity {
        Monotonicity::Increasing => 1.0,
        Monotonicity::Decreasing => -1.0,
        Monotonicity::None => {
            // Identity Jacobian
            return Array2::eye(k);
        }
    };
    // Column 0: all ones (d beta_i / d alpha_0 = 1 for all i)
    for i in 0..k {
        j_mat[[i, 0]] = 1.0;
    }
    // Columns j >= 1: J[i, j] = sign * exp(alpha[j]) for i >= j
    for j in 1..k {
        let clamped = alpha[j].clamp(-MAX_EXP_ALPHA, MAX_EXP_ALPHA);
        let val = sign * clamped.exp();
        for i in j..k {
            j_mat[[i, j]] = val;
        }
    }
    j_mat
}

/// Efficiently compute X_tilde = X_smooth * J for the monotonic reparameterization.
///
/// Since J is lower-triangular with cumulative structure, we use reverse cumulative
/// sums to compute this in O(n*k) instead of O(n*k^2).
///
/// Column j of X_tilde:
/// - Column 0: sum of all X columns (each multiplied by 1)
/// - Column j (j >= 1): sign * exp(alpha[j]) * sum of X columns j..k
///
/// Writes transformed smooth columns directly into `target` (in-place).
/// Uses reverse cumulative sums computed row-by-row (no intermediate allocation).
fn compute_x_tilde_inplace(
    x_smooth: &ArrayView2<'_, f64>,
    alpha: &[f64],
    monotonicity: &Monotonicity,
    target: &mut ndarray::ArrayViewMut2<'_, f64>,
) {
    let n = x_smooth.nrows();
    let k = x_smooth.ncols();
    let sign = match monotonicity {
        Monotonicity::Increasing => 1.0,
        Monotonicity::Decreasing => -1.0,
        Monotonicity::None => {
            target.assign(x_smooth);
            return;
        }
    };

    // Precompute exp(alpha[j]) * sign for each column
    let scales: Vec<f64> = (0..k)
        .map(|j| {
            if j == 0 {
                1.0 // Column 0: unscaled cumulative sum
            } else {
                sign * alpha[j].clamp(-MAX_EXP_ALPHA, MAX_EXP_ALPHA).exp()
            }
        })
        .collect();

    // Process row-by-row: compute reverse cumulative sum and scale in one pass.
    // No intermediate n×k allocation needed.
    for i in 0..n {
        // Reverse cumulative sum for this row
        let mut cumsum = 0.0;
        for j in (0..k).rev() {
            cumsum += x_smooth[[i, j]];
            target[[i, j]] = scales[j] * cumsum;
        }
    }
}

/// Compute the transformed penalty S_tilde = J' * S * J for monotonic terms.
fn compute_s_tilde(s_penalty: &Array2<f64>, j_mat: &Array2<f64>) -> Array2<f64> {
    // J' * S * J — straightforward matrix multiply for the small k x k matrices
    let sj = s_penalty.dot(j_mat);
    j_mat.t().dot(&sj)
}

/// Data for a single smooth term.
#[derive(Debug, Clone)]
pub struct SmoothTermData {
    /// Variable name
    pub name: String,
    /// Basis matrix for this term (n × k)
    pub basis: Array2<f64>,
    /// Penalty matrix S = D'D (k × k)
    pub penalty: Array2<f64>,
    /// Initial lambda (will be optimized if lambda_method = "gcv")
    pub initial_lambda: f64,
    /// Monotonicity constraint
    pub monotonicity: Monotonicity,
}

impl SmoothTermData {
    /// Create a new smooth term from a basis matrix.
    /// Automatically computes the second-order difference penalty.
    pub fn new(name: String, basis: Array2<f64>) -> Self {
        let k = basis.ncols();
        let penalty = penalty_matrix(k, 2); // Second-order difference penalty
        Self {
            name,
            basis,
            penalty,
            initial_lambda: 1.0,
            monotonicity: Monotonicity::None,
        }
    }

    /// Create with a custom initial lambda.
    pub fn with_lambda(mut self, lambda: f64) -> Self {
        self.initial_lambda = lambda;
        self
    }

    /// Set monotonicity constraint.
    pub fn with_monotonicity(mut self, mono: Monotonicity) -> Self {
        self.monotonicity = mono;
        self
    }

    /// Check if this term has a monotonicity constraint.
    pub fn is_monotonic(&self) -> bool {
        self.monotonicity != Monotonicity::None
    }

    /// Number of basis functions.
    pub fn k(&self) -> usize {
        self.basis.ncols()
    }
}

/// Compute final EDFs, GCV, and assemble the SmoothGLMResult from SmoothTermSpec data.
fn assemble_smooth_result_from_specs(
    coefficients: Array1<f64>,
    mu: Array1<f64>,
    eta: Array1<f64>,
    deviance: f64,
    iterations: usize,
    converged: bool,
    final_weights: &Array1<f64>,
    x_combined: ArrayView2<'_, f64>,
    penalty_specs: &[(&Array2<f64>, usize, usize)],
    lambdas: &[f64],
    p_param: usize,
    family_name: &str,
    prior_weights: Array1<f64>,
    y: &Array1<f64>,
    offset: Option<&Array1<f64>>,
    cov_unscaled: Option<Array2<f64>>,
    warnings: Vec<String>,
    step_halving_used: bool,
    solver_status: String,
    stationary: bool,
    max_std_score: f64,
) -> SmoothGLMResult {
    let n = y.len();

    // Compute EDFs
    let xtwx = compute_xtwx(x_combined, final_weights);
    let mut smooth_edfs = Vec::with_capacity(penalty_specs.len());

    for (i, &(penalty, start, _end)) in penalty_specs.iter().enumerate() {
        let lambda = lambdas[i];
        let k = penalty.nrows();
        let xtwx_block = xtwx
            .slice(s![start..start + k, start..start + k])
            .to_owned();
        let edf = compute_edf(&xtwx_block, penalty, lambda);
        smooth_edfs.push(edf);
    }

    let total_edf = (p_param as f64) + smooth_edfs.iter().sum::<f64>();
    let gcv = gcv_score(deviance, n, total_edf);

    // Build SmoothPenalty for result
    let mut smooth_penalty = SmoothPenalty::new();
    for (i, &(penalty, start, end)) in penalty_specs.iter().enumerate() {
        smooth_penalty.add_term(penalty.clone(), lambdas[i], start..end);
    }

    // Use provided covariance or compute from X'WX + S
    let cov = cov_unscaled.unwrap_or_else(|| {
        let total_cols = x_combined.ncols();
        let mut penalty_matrix = Array2::zeros((total_cols, total_cols));
        for (i, &(penalty, start, _end)) in penalty_specs.iter().enumerate() {
            embed_penalty(&mut penalty_matrix, penalty, start, lambdas[i]);
        }
        let xtwx_pen = add_penalty_to_xtwx(&xtwx, &penalty_matrix);
        invert_matrix(&xtwx_pen).unwrap_or_else(|| Array2::eye(total_cols))
    });

    SmoothGLMResult {
        coefficients,
        fitted_values: mu,
        linear_predictor: eta,
        deviance,
        iterations,
        converged,
        lambdas: lambdas.to_vec(),
        smooth_edfs,
        total_edf,
        gcv,
        covariance_unscaled: cov,
        family_name: family_name.to_string(),
        penalty: Penalty::Smooth(smooth_penalty),
        irls_weights: final_weights.clone(),
        prior_weights,
        design_matrix: x_combined.to_owned(),
        y: y.clone(),
        offset: offset.cloned(),
        warnings,
        step_halving_used,
        solver_status,
        stationary,
        max_std_score,
    }
}

// =============================================================================
// FAST SMOOTH GLM FITTING (mgcv-style)
// =============================================================================
//
// This approach optimizes lambda WITHIN a single IRLS fit using Brent's method.
// Instead of doing n_lambda separate fits, we:
// 1. Run IRLS normally
// 2. At each iteration (or every few), optimize lambda using cached X'WX
// 3. Update penalty and continue
//
// This is ~10-20x faster than grid search for large datasets.
// =============================================================================

use super::gcv_optimizer::MultiTermGCVOptimizer;

/// Simple matrix inversion helper.
fn invert_matrix(a: &Array2<f64>) -> Option<Array2<f64>> {
    convert::invert_matrix(a)
}

// =============================================================================
// Unified entry point: takes full design matrix + smooth specs
// =============================================================================

/// Smooth term specification for the unified entry point.
///
/// Instead of passing separate basis matrices, callers provide the full design
/// matrix and indicate which column ranges are smooth terms via this struct.
#[derive(Debug, Clone)]
pub struct SmoothTermSpec {
    /// Start column index (inclusive) in the full design matrix
    pub col_start: usize,
    /// End column index (exclusive) in the full design matrix
    pub col_end: usize,
    /// Penalty matrix (k × k) for this smooth term
    pub penalty: Array2<f64>,
    /// Monotonicity constraint
    pub monotonicity: Monotonicity,
    /// Initial lambda value
    pub initial_lambda: f64,
}

/// Fit GLM with smooth terms from a full design matrix.
///
/// This is the unified entry point that eliminates the need for Python to split
/// the design matrix into parametric + smooth parts. The full design matrix is
/// passed with column ranges identifying smooth terms. Coefficients are returned
/// in the same column order as the input matrix — no reordering needed.
///
/// Handles both unconstrained and monotonic smooth terms in a single call.
pub fn fit_smooth_glm_full_matrix(
    y: &Array1<f64>,
    x_full: ArrayView2<'_, f64>,
    smooth_specs: &[SmoothTermSpec],
    family: &dyn Family,
    link: &dyn Link,
    config: &SmoothGLMConfig,
    offset: Option<&Array1<f64>>,
    weights: Option<&Array1<f64>>,
    nonneg_indices: Option<&[usize]>,
    nonpos_indices: Option<&[usize]>,
) -> Result<SmoothGLMResult> {
    let n = y.len();
    let p = x_full.ncols();

    if x_full.nrows() != n {
        return Err(RustyStatsError::dim_mismatch(
            n,
            x_full.nrows(),
            "x_full rows vs y length",
        ));
    }

    if smooth_specs.is_empty() {
        // No smooth terms — delegate to standard GLM fit and wrap result
        let unified_config = FitConfig::from(&config.irls_config);
        let irls = fit_glm_unified(
            y,
            x_full,
            family,
            link,
            &unified_config,
            offset,
            weights,
            None,
        )?;
        let no_smooth_cols = std::collections::HashSet::new();
        let (stationary, max_std_score) = stationarity_report(
            x_full,
            y,
            &irls.fitted_values,
            &irls.prior_weights,
            family,
            link,
            &no_smooth_cols,
            &irls.coefficients,
            nonneg_indices,
            nonpos_indices,
        );
        return Ok(SmoothGLMResult {
            coefficients: irls.coefficients,
            fitted_values: irls.fitted_values,
            linear_predictor: irls.linear_predictor,
            deviance: irls.deviance,
            iterations: irls.iterations,
            converged: irls.converged && stationary,
            lambdas: vec![],
            smooth_edfs: vec![],
            total_edf: p as f64,
            gcv: 0.0,
            covariance_unscaled: irls.covariance_unscaled,
            family_name: irls.family_name,
            penalty: irls.penalty,
            irls_weights: irls.irls_weights,
            prior_weights: irls.prior_weights,
            design_matrix: irls.design_matrix.unwrap_or_else(|| x_full.to_owned()),
            y: irls.y,
            offset: if irls.offset.iter().all(|&v| v == 0.0) {
                None
            } else {
                Some(irls.offset)
            },
            warnings: irls.warnings,
            step_halving_used: irls.step_halving_used,
            solver_status: irls.solver_status,
            stationary,
            max_std_score,
        });
    }

    // Validate specs
    for (i, spec) in smooth_specs.iter().enumerate() {
        if spec.col_end > p || spec.col_start >= spec.col_end {
            return Err(RustyStatsError::InvalidValue(format!(
                "Smooth spec {} has invalid column range [{}, {}), matrix has {} columns",
                i, spec.col_start, spec.col_end, p
            )));
        }
        let k = spec.col_end - spec.col_start;
        if spec.penalty.nrows() != k || spec.penalty.ncols() != k {
            return Err(RustyStatsError::dim_mismatch(
                k,
                spec.penalty.nrows(),
                format!(
                    "smooth spec {} penalty shape ({}, {}) vs expected ({}, {})",
                    i,
                    spec.penalty.nrows(),
                    spec.penalty.ncols(),
                    k,
                    k
                ),
            ));
        }
    }

    // Determine parametric column count (everything NOT in a smooth term)
    let mut smooth_cols = std::collections::HashSet::new();
    for spec in smooth_specs {
        for c in spec.col_start..spec.col_end {
            smooth_cols.insert(c);
        }
    }
    let p_param = parametric_column_count(p, smooth_cols.len());

    // Build column ranges in x_full order (smooth specs are already indexed into x_full)
    let term_indices: Vec<(usize, usize)> = smooth_specs
        .iter()
        .map(|s| (s.col_start, s.col_end))
        .collect();

    let offset_vec = offset.cloned().unwrap_or_else(|| Array1::zeros(n));
    let prior_weights = weights.cloned().unwrap_or_else(|| Array1::ones(n));
    // Clamp initial lambdas into the configured bounds. Callers pin a fixed
    // lambda by passing lambda_min == lambda_max; without the clamp the
    // monotone path's REML update can keep an out-of-bounds initial value.
    let mut lambdas: Vec<f64> = smooth_specs
        .iter()
        .map(|s| s.initial_lambda.clamp(config.lambda_min, config.lambda_max))
        .collect();

    let has_monotonic = smooth_specs.iter().any(|s| s.is_monotonic());

    // Use x_full directly as x_combined — no reassembly needed
    let x_combined = x_full;
    let total_cols = p;

    let mut warnings: Vec<String> = Vec::new();
    let mut coefficients = Array1::zeros(total_cols);

    // Initialize mu
    let mut mu = family.initialize_mu(y);
    let mut eta = link.link(&mu);
    let mut deviance = family.deviance(y, &mu, Some(&prior_weights));

    // Family initializers can be near-saturated and not representable by the
    // smooth design matrix (Gaussian identity starts at mu=y). Project that
    // initializer into coefficient space so the step-halving baseline is a real
    // fitted iterate instead of the saturated response.
    {
        let eta_no_offset = centered_eta(&eta, &offset_vec);
        let zero_penalty = Array2::zeros((total_cols, total_cols));
        match solve_weighted_least_squares_with_penalty_matrix(
            x_combined,
            &eta_no_offset,
            &prior_weights,
            &zero_penalty,
            true,
        ) {
            Ok((projected, _)) => {
                if let Some(mut projected) = initial_projection_if_finite(projected) {
                    project_coefficients_to_bounds(&mut projected, nonneg_indices, nonpos_indices);
                    coefficients = projected;
                    eta =
                        linear_predictor_from_coefficients(x_combined, &coefficients, &offset_vec);
                    mu = family.clamp_mu(&link.inverse(&eta));
                    deviance = family.deviance(y, &mu, Some(&prior_weights));
                } else {
                    warnings.push(
                        "Initial smooth coefficient projection failed. Starting from zero coefficients."
                            .to_string(),
                    );
                }
            }
            _ => warnings.push(
                "Initial smooth coefficient projection failed. Starting from zero coefficients."
                    .to_string(),
            ),
        }
    }

    let mut converged = false;
    let mut iteration = 0;
    let mut final_weights = Array1::ones(n);
    let mut step_halving_used = false;
    let mut step_halving_failed = false;

    let log_lambda_min = config.lambda_min.ln();
    let log_lambda_max = config.lambda_max.ln();
    let mut penalty_matrix = Array2::zeros((total_cols, total_cols));
    let mut lambdas_stable_count = 0u32;

    // Track previous lambdas to skip rebuilding penalty_matrix when unchanged.
    // Only used in the non-monotonic path (monotonic always rebuilds due to J changes).
    let mut prev_lambdas: Vec<f64> = vec![f64::NAN; smooth_specs.len()];
    let mut penalty_dirty = true;

    // Pre-allocate per-iteration buffers to avoid heap allocations inside the IRLS loop
    let mut irls_weights = Array1::zeros(n);
    let mut combined_weights = Array1::zeros(n);
    let mut working_response = Array1::zeros(n);

    // Use a looser tolerance for smooth models — GCV lambda perturbation
    // introduces noise larger than 1e-8, so tighter tolerance just wastes iterations
    let smooth_tolerance = config.irls_config.tolerance.max(SMOOTH_CONVERGENCE_TOL);

    // Pre-allocate adjusted_response for the monotonic path — avoids cloning
    // working_response each iteration (saves one n-sized allocation per iteration)
    let mut adjusted_response = if has_monotonic {
        Array1::zeros(n)
    } else {
        Array1::zeros(0) // Not used in non-monotonic path
    };

    // Pre-allocate x_tilde for the monotonic path — cloned once from x_combined,
    // then only the monotonic smooth columns are overwritten each iteration.
    // Parametric columns (intercept, etc.) never change and stay correct.
    let mut x_tilde = if has_monotonic {
        x_combined.to_owned()
    } else {
        Array2::zeros((0, 0)) // Not used in non-monotonic path
    };

    // =========================================================================
    // Exp reparameterization state for monotonic terms
    // =========================================================================
    // For each monotonic smooth term, we maintain alpha parameters in unconstrained
    // space. Non-monotonic terms and parametric columns are unaffected.
    //
    // alpha_params[i] holds the alpha vector for smooth_specs[i] (only used if monotonic).
    let mut alpha_params: Vec<Option<Array1<f64>>> = smooth_specs
        .iter()
        .map(|spec| {
            if spec.is_monotonic() {
                let k = smooth_term_width(spec);
                // Initialize alpha to small values (beta starts near zero)
                Some(Array1::zeros(k))
            } else {
                None
            }
        })
        .collect();

    if has_monotonic {
        // =================================================================
        // MONOTONIC PATH: Nested iteration following scam (Pya & Wood 2015)
        // =================================================================
        //
        // Architecture: outer loop updates lambda via GCV, inner loop runs
        // PIRLS to convergence with fixed lambda. This is more stable than
        // interleaving GCV within PIRLS because X_tilde changes every
        // iteration (as alpha updates the Jacobian).
        //
        // Initialization: zero alpha (flat monotonic start), matching scam.
        // REML selects initial lambda, then nested iteration refines.
        // =================================================================

        // Phase 1: GCV on unconstrained basis for starting lambda
        {
            let link_deriv = link.derivative(&mu);
            let variance = family.variance(&mu);
            update_irls_work_arrays(
                &eta,
                &offset_vec,
                y,
                &mu,
                &prior_weights,
                &variance,
                &link_deriv,
                config.irls_config.min_weight,
                &mut irls_weights,
                &mut combined_weights,
                &mut working_response,
            );
            let (init_xtwx, init_xtwz) =
                compute_xtwx_xtwz(x_combined, &working_response, &combined_weights)?;
            let init_ztwz = weighted_response_sum_squares(&working_response, &combined_weights);
            let init_penalties: Vec<Array2<f64>> =
                smooth_specs.iter().map(|s| s.penalty.clone()).collect();
            let optimizer = MultiTermGCVOptimizer::new_from_cached(
                init_xtwx,
                init_xtwz,
                init_ztwz,
                init_penalties,
                term_indices.clone(),
                n,
                p_param,
            );
            lambdas = optimizer.optimize_lambdas_reml(
                &lambdas,
                log_lambda_min,
                log_lambda_max,
                config.lambda_tol,
                5,
            );
        }

        // Warm-start: run unconstrained IRLS to convergence, then convert
        // betas to alpha via beta_to_alpha (with PAV projection). This seeds
        // the PIRLS with good parametric estimates and a reasonable monotonic
        // starting shape, matching scam's approach of initializing from an
        // unconstrained preliminary fit.
        {
            let mut init_coef = coefficients.clone();
            let mut init_mu = mu.clone();
            let mut init_eta = eta.clone();
            let mut init_dev = deviance;
            // The warm start is counted against the caller's total iteration
            // budget so fit(max_iter=N) bounds ALL smooth-path work.
            let warm_cap = SMOOTH_WARM_START_MAX_ITER.min(config.irls_config.max_iterations);
            for _init_iter in 0..warm_cap {
                iteration += 1;
                let init_link_deriv = link.derivative(&init_mu);
                let init_var = family.variance(&init_mu);
                let mut init_iw = Array1::zeros(n);
                let mut init_cw = Array1::zeros(n);
                let mut init_z = Array1::zeros(n);
                update_irls_work_arrays(
                    &init_eta,
                    &offset_vec,
                    y,
                    &init_mu,
                    &prior_weights,
                    &init_var,
                    &init_link_deriv,
                    config.irls_config.min_weight,
                    &mut init_iw,
                    &mut init_cw,
                    &mut init_z,
                );
                let mut init_penalty = Array2::zeros((total_cols, total_cols));
                for (i, spec) in smooth_specs.iter().enumerate() {
                    embed_penalty(&mut init_penalty, &spec.penalty, spec.col_start, lambdas[i]);
                }
                let (new_init_coef, _) = solve_weighted_least_squares_with_penalty_matrix(
                    x_combined,
                    &init_z,
                    &init_cw,
                    &init_penalty,
                    true,
                )?;
                init_coef = new_init_coef;
                init_eta = linear_predictor_from_coefficients(x_combined, &init_coef, &offset_vec);
                init_mu = family.clamp_mu(&link.inverse(&init_eta));
                let new_dev = family.deviance(y, &init_mu, Some(&prior_weights));
                let init_rel = relative_change_with_unit_offset(init_dev, new_dev);
                init_dev = new_dev;
                if change_below_tolerance(init_rel, config.irls_config.tolerance) {
                    break;
                }
            }

            // Convert unconstrained betas to alpha (PAV projection ensures monotonicity)
            for (i, spec) in smooth_specs.iter().enumerate() {
                if let Some(ref mut alpha) = alpha_params[i] {
                    let beta_slice =
                        &init_coef.as_slice().expect("contiguous")[spec.col_start..spec.col_end];
                    *alpha = beta_to_alpha(beta_slice, &spec.monotonicity);
                }
            }

            // Set coefficients: parametric from init, monotonic from alpha→beta
            coefficients = init_coef;
            for (i, spec) in smooth_specs.iter().enumerate() {
                if let Some(ref alpha) = alpha_params[i] {
                    let beta = alpha_to_beta(
                        alpha.as_slice().expect("contiguous array"),
                        &spec.monotonicity,
                    );
                    let cs = coefficients.as_slice_mut().expect("contiguous array");
                    for (j, &b) in beta.iter().enumerate() {
                        cs[smooth_coefficient_index(spec.col_start, j)] = b;
                    }
                }
            }

            project_coefficients_to_bounds(&mut coefficients, nonneg_indices, nonpos_indices);

            eta = linear_predictor_from_coefficients(x_combined, &coefficients, &offset_vec);
            mu = family.clamp_mu(&link.inverse(&eta));
            deviance = family.deviance(y, &mu, Some(&prior_weights));
        }

        // Outer loop: update lambda, then converge inner PIRLS.
        //
        // The loop is bounded by the caller's TOTAL iteration budget
        // (config.irls_config.max_iterations) instead of a hard-coded outer
        // cap, so fit(max_iter=N) genuinely bounds the work done here:
        // N=5 visibly truncates, N=5000 is not silently stopped at an
        // internal cap. Each outer cycle with remaining budget performs at
        // least one inner iteration, so the outer loop always terminates.
        let budget = config.irls_config.max_iterations;
        while iteration < budget {
            let lambdas_at_start = lambdas.clone();

            // Inner PIRLS loop: converge coefficients for fixed lambda.
            // scam uses up to 200 inner iterations per lambda update.
            for _inner in 0..SMOOTH_INNER_MAX_PER_CYCLE {
                if iteration >= budget {
                    break;
                }
                iteration += 1;
                let deviance_old = deviance;

                // Compute IRLS weights and working response
                let link_deriv = link.derivative(&mu);
                let variance = family.variance(&mu);
                update_irls_work_arrays(
                    &eta,
                    &offset_vec,
                    y,
                    &mu,
                    &prior_weights,
                    &variance,
                    &link_deriv,
                    config.irls_config.min_weight,
                    &mut irls_weights,
                    &mut combined_weights,
                    &mut working_response,
                );
                // Reflect the current iterate's weights immediately so that a
                // step-halving failure on the FIRST iteration still yields the
                // retained-iterate weights for covariance/EDF/GCV — not all-ones.
                final_weights = combined_weights.clone();

                // Save alpha state before WLS for step halving
                let pre_wls_alphas: Vec<Option<Array1<f64>>> = alpha_params.clone();

                // Build X_tilde: replace monotonic smooth columns with X*J
                for (i, spec) in smooth_specs.iter().enumerate() {
                    if let Some(ref alpha) = alpha_params[i] {
                        let x_smooth = x_combined.slice(s![.., spec.col_start..spec.col_end]);
                        let mut target = x_tilde.slice_mut(s![.., spec.col_start..spec.col_end]);
                        compute_x_tilde_inplace(
                            &x_smooth,
                            alpha.as_slice().expect("contiguous array"),
                            &spec.monotonicity,
                            &mut target,
                        );
                    }
                }

                // Pre-compute Jacobians for monotonic terms (used for both linearization and penalty)
                let cached_jacobians: Vec<Option<Array2<f64>>> = smooth_specs
                    .iter()
                    .enumerate()
                    .map(|(i, spec)| {
                        alpha_params[i].as_ref().map(|alpha| {
                            compute_monotonic_jacobian(
                                alpha.as_slice().expect("contiguous array"),
                                &spec.monotonicity,
                            )
                        })
                    })
                    .collect();

                // Linearization offset: z_adj = z - X*(beta - J*alpha)
                adjusted_response.assign(&working_response);
                for (i, spec) in smooth_specs.iter().enumerate() {
                    if let Some(ref alpha) = alpha_params[i] {
                        let alpha_slice = alpha.as_slice().expect("contiguous array");
                        let beta_mono = alpha_to_beta(alpha_slice, &spec.monotonicity);
                        let j_mat = cached_jacobians[i]
                            .as_ref()
                            .expect("cached Jacobian exists for monotonic term");
                        let j_alpha = j_mat.dot(alpha);
                        let offset_mono = monotonic_offset(&beta_mono, &j_alpha);
                        let x_smooth = x_combined.slice(s![.., spec.col_start..spec.col_end]);
                        subtract_smooth_offset(&mut adjusted_response, x_smooth, &offset_mono);
                    }
                }

                // Build transformed penalty and solve WLS
                let x_tilde_view = x_tilde.view();
                let (cached_xtwx_t, cached_xtwz_t) =
                    compute_xtwx_xtwz(x_tilde_view, &adjusted_response, &combined_weights)?;

                penalty_matrix.fill(0.0);
                for (i, spec) in smooth_specs.iter().enumerate() {
                    if let Some(ref _alpha) = alpha_params[i] {
                        let j_mat = cached_jacobians[i]
                            .as_ref()
                            .expect("cached Jacobian exists for monotonic term");
                        let s_tilde = compute_s_tilde(&spec.penalty, j_mat);
                        embed_penalty(&mut penalty_matrix, &s_tilde, spec.col_start, lambdas[i]);
                    } else {
                        embed_penalty(
                            &mut penalty_matrix,
                            &spec.penalty,
                            spec.col_start,
                            lambdas[i],
                        );
                    }
                }

                let (alpha_coef, _) = solve_wls_from_precomputed(
                    &cached_xtwx_t,
                    &cached_xtwz_t,
                    &penalty_matrix,
                    true,
                )?;

                // Build candidate coefficients: all non-monotonic from WLS
                let new_coef = alpha_coef.clone();

                // WLS-proposed alpha for monotonic terms (clamped)
                let wls_alphas: Vec<Option<Array1<f64>>> = smooth_specs
                    .iter()
                    .enumerate()
                    .map(|(i, spec)| {
                        if alpha_params[i].is_some() {
                            let alpha_wls = &alpha_coef.as_slice().expect("contiguous")
                                [spec.col_start..spec.col_end];
                            let clamped: Vec<f64> = alpha_wls
                                .iter()
                                .enumerate()
                                .map(|(j, &a)| clamp_monotonic_alpha_value(j, a))
                                .collect();
                            Some(Array1::from_vec(clamped))
                        } else {
                            None
                        }
                    })
                    .collect();

                // Penalized deviance at current point. With REML-selected lambdas,
                // penalized deviance is well-calibrated for step acceptance.
                let pen_dev_old =
                    smooth_penalized_deviance(deviance_old, &coefficients, smooth_specs, &lambdas);

                // Step halving on penalized deviance with unified blending
                // of all coefficients. Accept only finite non-worsening steps
                // so a failed PIRLS proposal cannot silently replace the last
                // valid iterate.
                let mut best_pen_dev = f64::INFINITY;
                let mut best_coef = new_coef.clone();
                let mut best_alphas = wls_alphas.clone();
                let mut best_trial: Option<i32> = None;

                // 20 halving trials plus a final fraction = 0 fallback (trial 20),
                // which exactly retains the previous accepted iterate (alphas =
                // pre_wls_alphas, coef = coefficients) and is known finite and
                // non-worsening. This guarantees a finite non-worsening step
                // always exists when the previous iterate was valid.
                for trial in 0..=20 {
                    let step = smooth_step_fraction_for_trial(trial, 20);

                    // Blend all coefficients: parametric + non-monotonic smooth
                    let mut trial_coef = blend_step_arrays(&coefficients, &new_coef, step);

                    // Monotonic terms: blend in alpha-space, recover beta
                    let mut trial_alphas: Vec<Option<Array1<f64>>> =
                        Vec::with_capacity(smooth_specs.len());
                    for (i, spec) in smooth_specs.iter().enumerate() {
                        if let (Some(ref old_alpha), Some(ref new_alpha)) =
                            (&pre_wls_alphas[i], &wls_alphas[i])
                        {
                            let blended = blend_step_arrays(old_alpha, new_alpha, step);
                            let beta = alpha_to_beta(
                                blended.as_slice().expect("contiguous array"),
                                &spec.monotonicity,
                            );
                            let cs = trial_coef.as_slice_mut().expect("contiguous array");
                            for (j, &b) in beta.iter().enumerate() {
                                cs[smooth_coefficient_index(spec.col_start, j)] = b;
                            }
                            trial_alphas.push(Some(blended));
                        } else {
                            trial_alphas.push(None);
                        }
                    }

                    // Re-apply sign constraints after blending
                    project_coefficients_to_bounds(&mut trial_coef, nonneg_indices, nonpos_indices);

                    let trial_eta =
                        linear_predictor_from_coefficients(x_combined, &trial_coef, &offset_vec);
                    let trial_mu = family.clamp_mu(&link.inverse(&trial_eta));
                    let trial_dev = family.deviance(y, &trial_mu, Some(&prior_weights));
                    // Penalized deviance for step acceptance
                    let trial_pen_dev =
                        smooth_penalized_deviance(trial_dev, &trial_coef, smooth_specs, &lambdas);
                    if should_skip_smooth_trial(&trial_eta, &trial_mu, trial_dev, trial_pen_dev) {
                        continue;
                    }

                    if smooth_trial_is_better(best_trial, trial_pen_dev, best_pen_dev) {
                        best_pen_dev = trial_pen_dev;
                        best_coef = trial_coef;
                        best_alphas = trial_alphas;
                        best_trial = Some(trial);
                        if immediate_trial_accepts(trial, trial_pen_dev, pen_dev_old) {
                            break;
                        }
                    }
                    if halved_trial_accepts(trial, best_pen_dev, pen_dev_old) {
                        break;
                    }
                }

                let accept_threshold = smooth_accept_threshold(pen_dev_old, smooth_tolerance);
                if smooth_step_halving_failed(best_trial, best_pen_dev, accept_threshold) {
                    step_halving_failed = true;
                    warnings.push(
                        "Smooth PIRLS step halving found no finite non-worsening step; \
                         retained the previous iterate."
                            .to_string(),
                    );
                    break;
                }
                // trial 20 is the fraction-0 fallback (retained previous iterate),
                // not a genuine halved step.
                if genuine_halved_step(best_trial, 20) {
                    step_halving_used = true;
                }

                for (i, _spec) in smooth_specs.iter().enumerate() {
                    if let Some(ref best_a) = best_alphas[i] {
                        if let Some(ref mut alpha) = alpha_params[i] {
                            *alpha = best_a.clone();
                        }
                    }
                }
                coefficients = best_coef;

                // Update state
                eta = linear_predictor_from_coefficients(x_combined, &coefficients, &offset_vec);
                mu = family.clamp_mu(&link.inverse(&eta));
                deviance = family.deviance(y, &mu, Some(&prior_weights));
                final_weights = combined_weights.clone();

                // Inner convergence: penalized deviance change
                let rel_change = relative_change_with_unit_offset(pen_dev_old, best_pen_dev);
                if change_below_tolerance(rel_change, smooth_tolerance) {
                    converged = true;
                    break;
                }
            }
            if step_halving_failed {
                break;
            }
            if iteration >= budget {
                // Budget exhausted mid-cycle: the inner state may claim
                // convergence, but lambda stability was never verified.
                converged = false;
                break;
            }

            // Outer loop: update lambdas via GCV on the converged X_tilde
            {
                // Recompute IRLS quantities at convergence point
                let link_deriv = link.derivative(&mu);
                let variance = family.variance(&mu);
                update_irls_work_arrays(
                    &eta,
                    &offset_vec,
                    y,
                    &mu,
                    &prior_weights,
                    &variance,
                    &link_deriv,
                    config.irls_config.min_weight,
                    &mut irls_weights,
                    &mut combined_weights,
                    &mut working_response,
                );

                // Rebuild X_tilde at converged alpha
                for (i, spec) in smooth_specs.iter().enumerate() {
                    if let Some(ref alpha) = alpha_params[i] {
                        let x_smooth = x_combined.slice(s![.., spec.col_start..spec.col_end]);
                        let mut target = x_tilde.slice_mut(s![.., spec.col_start..spec.col_end]);
                        compute_x_tilde_inplace(
                            &x_smooth,
                            alpha.as_slice().expect("contiguous array"),
                            &spec.monotonicity,
                            &mut target,
                        );
                    }
                }

                // Pre-compute Jacobians for monotonic terms (used for both linearization and penalty)
                let cached_jacobians_outer: Vec<Option<Array2<f64>>> = smooth_specs
                    .iter()
                    .enumerate()
                    .map(|(i, spec)| {
                        alpha_params[i].as_ref().map(|alpha| {
                            compute_monotonic_jacobian(
                                alpha.as_slice().expect("contiguous array"),
                                &spec.monotonicity,
                            )
                        })
                    })
                    .collect();

                // Rebuild adjusted response at converged alpha
                adjusted_response.assign(&working_response);
                for (i, spec) in smooth_specs.iter().enumerate() {
                    if let Some(ref alpha) = alpha_params[i] {
                        let alpha_slice = alpha.as_slice().expect("contiguous array");
                        let beta_mono = alpha_to_beta(alpha_slice, &spec.monotonicity);
                        let j_mat = cached_jacobians_outer[i]
                            .as_ref()
                            .expect("cached Jacobian exists for monotonic term");
                        let j_alpha = j_mat.dot(alpha);
                        let offset_mono = monotonic_offset(&beta_mono, &j_alpha);
                        let x_smooth = x_combined.slice(s![.., spec.col_start..spec.col_end]);
                        subtract_smooth_offset(&mut adjusted_response, x_smooth, &offset_mono);
                    }
                }

                let x_tilde_view = x_tilde.view();
                let (cached_xtwx_t, cached_xtwz_t) =
                    compute_xtwx_xtwz(x_tilde_view, &adjusted_response, &combined_weights)?;
                let ztwz = weighted_response_sum_squares(&adjusted_response, &combined_weights);

                let transformed_penalties: Vec<Array2<f64>> = smooth_specs
                    .iter()
                    .enumerate()
                    .map(|(i, spec)| {
                        if let Some(ref _alpha) = alpha_params[i] {
                            let j_mat = cached_jacobians_outer[i]
                                .as_ref()
                                .expect("cached Jacobian exists for monotonic term");
                            compute_s_tilde(&spec.penalty, j_mat)
                        } else {
                            spec.penalty.clone()
                        }
                    })
                    .collect();

                let optimizer = MultiTermGCVOptimizer::new_from_cached(
                    cached_xtwx_t,
                    cached_xtwz_t,
                    ztwz,
                    transformed_penalties,
                    term_indices.clone(),
                    n,
                    p_param,
                );

                // Use REML for monotonic models — more stable than GCV because
                // it accounts for posterior uncertainty via the log-determinant
                // term, rather than relying on trace-based EDF which doesn't
                // capture the effect of monotonicity constraints.
                lambdas = optimizer.optimize_lambdas_reml(
                    &lambdas,
                    log_lambda_min,
                    log_lambda_max,
                    config.lambda_tol,
                    5,
                );
            }

            // Check outer convergence: lambdas stabilized?
            let max_lambda_change = max_lambda_relative_change(&lambdas_at_start, &lambdas);

            if should_stop_monotonic_outer(max_lambda_change, converged) {
                break;
            }
            // Reset inner convergence flag for next outer iteration
            converged = false;
        }
        // converged remains true only if the last inner loop converged
        // AND lambdas were stable. If the outer loop exhausted the iteration
        // budget without meeting the break condition, converged stays false.
    }

    // =========================================================================
    // NON-MONOTONIC PATH: Standard penalized IRLS with interleaved GCV
    // =========================================================================
    while !has_monotonic && iteration < config.irls_config.max_iterations {
        iteration += 1;
        let deviance_old = deviance;

        let link_deriv = link.derivative(&mu);
        let variance = family.variance(&mu);

        update_irls_work_arrays(
            &eta,
            &offset_vec,
            y,
            &mu,
            &prior_weights,
            &variance,
            &link_deriv,
            config.irls_config.min_weight,
            &mut irls_weights,
            &mut combined_weights,
            &mut working_response,
        );
        // Reflect the current iterate's weights immediately so that a
        // step-halving failure on the FIRST iteration still yields the
        // retained-iterate weights for covariance/EDF/GCV — not all-ones.
        final_weights = combined_weights.clone();

        let mut new_coef;

        // Compute X'WX and X'Wz ONCE per iteration — shared by GCV and WLS
        let (cached_xtwx, cached_xtwz) =
            compute_xtwx_xtwz(x_combined, &working_response, &combined_weights)?;

        let ztwz = weighted_response_sum_squares(&working_response, &combined_weights);

        let run_gcv = should_run_smooth_gcv(lambdas_stable_count, iteration);
        if run_gcv {
            let old_lambdas = lambdas.clone();
            let penalties: Vec<Array2<f64>> =
                smooth_specs.iter().map(|s| s.penalty.clone()).collect();
            let optimizer = MultiTermGCVOptimizer::new_from_cached(
                cached_xtwx.clone(),
                cached_xtwz.clone(),
                ztwz,
                penalties,
                term_indices.clone(),
                n,
                p_param,
            );
            lambdas = optimizer.optimize_lambdas(
                &lambdas,
                log_lambda_min,
                log_lambda_max,
                config.lambda_tol,
                3,
            );
            let max_rel_change = max_lambda_relative_change(&old_lambdas, &lambdas);
            lambdas_stable_count = next_lambdas_stable_count(max_rel_change, lambdas_stable_count);
            penalty_dirty = lambdas_changed(&lambdas, &prev_lambdas);
        }

        if penalty_dirty {
            penalty_matrix.fill(0.0);
            for (i, spec) in smooth_specs.iter().enumerate() {
                embed_penalty(
                    &mut penalty_matrix,
                    &spec.penalty,
                    spec.col_start,
                    lambdas[i],
                );
            }
            prev_lambdas.clone_from_slice(&lambdas);
            penalty_dirty = false;
        }

        let (coef, _) =
            solve_wls_from_precomputed(&cached_xtwx, &cached_xtwz, &penalty_matrix, true)?;
        new_coef = coef;

        // Enforce sign constraints before evaluating deviance
        project_coefficients_to_bounds(&mut new_coef, nonneg_indices, nonpos_indices);

        // Step halving if the full step is non-finite or worsens the fitted
        // smooth objective. Raw deviance can rise slightly as smoothing
        // increases; the penalized deviance is the objective this solver
        // actually optimizes.
        let pen_dev_old =
            smooth_penalized_deviance(deviance_old, &coefficients, smooth_specs, &lambdas);
        let accept_threshold = smooth_accept_threshold(pen_dev_old, smooth_tolerance);
        let eta_new = linear_predictor_from_coefficients(x_combined, &new_coef, &offset_vec);
        let mu_new = family.clamp_mu(&link.inverse(&eta_new));
        let deviance_new = family.deviance(y, &mu_new, Some(&prior_weights));
        let pen_dev_new =
            smooth_penalized_deviance(deviance_new, &new_coef, smooth_specs, &lambdas);
        let mut step_accepted = smooth_step_candidate_accepted(
            &eta_new,
            &mu_new,
            deviance_new,
            pen_dev_new,
            accept_threshold,
        );
        let mut accepted_coef = if step_accepted {
            new_coef.clone()
        } else {
            coefficients.clone()
        };

        if !step_accepted {
            let mut step = 0.5;
            // 10 halving steps plus a final fraction = 0 fallback (the previous
            // accepted coefficients), which is known finite and non-worsening.
            // This guarantees a finite non-worsening step always exists when the
            // previous iterate was valid, rather than blending toward but never
            // reaching it.
            for half_step in 0..=10 {
                let fraction = if half_step == 10 { 0.0 } else { step };
                let mut blended = blend_arrays(&coefficients, &new_coef, fraction);

                project_coefficients_to_bounds(&mut blended, nonneg_indices, nonpos_indices);

                let eta_full =
                    linear_predictor_from_coefficients(x_combined, &blended, &offset_vec);
                let mu_blend = family.clamp_mu(&link.inverse(&eta_full));
                let dev_blend = family.deviance(y, &mu_blend, Some(&prior_weights));
                let pen_dev_blend =
                    smooth_penalized_deviance(dev_blend, &blended, smooth_specs, &lambdas);

                if smooth_step_candidate_accepted(
                    &eta_full,
                    &mu_blend,
                    dev_blend,
                    pen_dev_blend,
                    accept_threshold,
                ) {
                    accepted_coef = blended;
                    step_accepted = true;
                    // A fraction-0 fallback is not a genuine halving "step",
                    // it simply retains the previous iterate.
                    if fraction_is_genuine_halving(fraction) {
                        step_halving_used = true;
                    }
                    break;
                }
                step = next_smooth_half_step(step);
            }
        }

        if !step_accepted {
            step_halving_failed = true;
            warnings.push(
                "Smooth IRLS step halving found no finite non-worsening step; \
                 retained the previous iterate."
                    .to_string(),
            );
            break;
        }
        coefficients = accepted_coef;

        // Update state
        eta = linear_predictor_from_coefficients(x_combined, &coefficients, &offset_vec);
        mu = family.clamp_mu(&link.inverse(&eta));
        deviance = family.deviance(y, &mu, Some(&prior_weights));
        final_weights = combined_weights.clone();
        let pen_dev = smooth_penalized_deviance(deviance, &coefficients, smooth_specs, &lambdas);

        let rel_change = relative_change_with_floor(pen_dev_old, pen_dev, ZERO_TOL);

        if change_below_tolerance(rel_change, smooth_tolerance) {
            converged = true;
            break;
        }
    }

    // =========================================================================
    // Termination contract: intercept refresh + exit stationarity check
    // =========================================================================
    // The inner convergence test measures PROGRESS (penalized-deviance change),
    // which a stalled solver passes. At any true (constrained) optimum the
    // score of every unpenalized, unconstrained coordinate — the intercept
    // above all — must vanish. Two safeguards enforce that contract:
    //
    // 1. Intercept refresh: one exact 1-D score solve along the intercept
    //    direction. Guarantees mean(prediction) == mean(response) for
    //    canonical links on every returned fit, whatever happened upstream.
    //    Skipped when the intercept is already comfortably stationary, so
    //    healthy fits are untouched.
    // 2. Stationarity gate: `converged=True` additionally requires the KKT
    //    check to pass; a stalled fit reports `stalled_nonstationary`.
    //
    // Before both, a final unconstrained-block re-solve: holding the monotone
    // blocks fixed, the remaining coordinates form a convex penalized GLM that
    // the stalled main loop may have left slightly off-optimum. Solving it is
    // a block coordinate-descent step (never worsens the penalized objective).
    if has_monotonic && iteration < config.irls_config.max_iterations {
        let refined = refine_unconstrained_block(
            y,
            x_combined,
            smooth_specs,
            &lambdas,
            family,
            link,
            &offset_vec,
            &prior_weights,
            config.irls_config.min_weight,
            smooth_tolerance,
            nonneg_indices,
            nonpos_indices,
            &mut coefficients,
            &mut eta,
            &mut mu,
            &mut deviance,
            &mut iteration,
            config.irls_config.max_iterations,
        )?;
        if refined {
            warnings.push(
                "Applied final unconstrained-block re-solve (parametric and \
                 unconstrained smooth coordinates) to restore exit stationarity."
                    .to_string(),
            );
        }
    }

    if let Some(intercept_col) =
        find_intercept_column(x_combined, &smooth_cols, nonneg_indices, nonpos_indices)
    {
        let (score_resid, info_weight) = quasi_score_arrays(y, &mu, &prior_weights, family, link);
        let (_, intercept_std_score) =
            column_std_score(x_combined, intercept_col, &score_resid, &info_weight);
        if intercept_std_score >= 0.1 * SMOOTH_KKT_SCORE_TOL {
            if let Some(shift) = solve_intercept_shift(y, &eta, &prior_weights, family, link) {
                if shift != 0.0 {
                    let eta_shifted = eta.mapv(|e| e + shift);
                    let mu_shifted = family.clamp_mu(&link.inverse(&eta_shifted));
                    let dev_shifted = family.deviance(y, &mu_shifted, Some(&prior_weights));
                    // The exact 1-D score solve minimizes deviance along the
                    // intercept direction; reject on the (non-canonical-link)
                    // off-chance it does not improve within tolerance.
                    if dev_shifted <= deviance + smooth_tolerance * (1.0 + deviance.abs()) {
                        coefficients[intercept_col] += shift;
                        eta = eta_shifted;
                        mu = mu_shifted;
                        deviance = dev_shifted;
                        warnings.push(format!(
                            "Applied final intercept refresh of {shift:+.6e} to restore \
                             the intercept score equation (mean prediction = mean response)."
                        ));
                    }
                }
            }
        }
    }

    let (stationary, max_std_score) = stationarity_report(
        x_combined,
        y,
        &mu,
        &prior_weights,
        family,
        link,
        &smooth_cols,
        &coefficients,
        nonneg_indices,
        nonpos_indices,
    );
    let converged_raw = converged;
    let converged = converged_raw && stationary;
    if !stationary {
        warnings.push(format!(
            "Fit is non-stationary at exit: max standardized score {max_std_score:.3e} \
             over unpenalized coordinates (tolerance {SMOOTH_KKT_SCORE_TOL:.0e})."
        ));
    }

    // Assemble result directly from SmoothTermSpec (no SmoothTermData conversion)
    let penalty_specs: Vec<(&Array2<f64>, usize, usize)> = smooth_specs
        .iter()
        .map(|s| (&s.penalty, s.col_start, s.col_end))
        .collect();
    let solver_status = smooth_solver_status(step_halving_failed, converged_raw, stationary);
    if should_warn_smooth_nonconvergence(converged) {
        warnings.push(smooth_nonconvergence_warning(solver_status));
    }

    Ok(assemble_smooth_result_from_specs(
        coefficients,
        mu,
        eta,
        deviance,
        iteration,
        converged,
        &final_weights,
        x_combined,
        &penalty_specs,
        &lambdas,
        p_param,
        family.name(),
        prior_weights,
        y,
        offset,
        // Covariance is computed once here by assemble_smooth_result_from_specs,
        // rather than redundantly on every intermediate IRLS iteration.
        None,
        warnings,
        step_halving_used,
        solver_status.to_string(),
        stationary,
        max_std_score,
    ))
}

impl SmoothTermSpec {
    /// Whether this term has a monotonicity constraint.
    pub fn is_monotonic(&self) -> bool {
        !matches!(self.monotonicity, Monotonicity::None)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::families::{BinomialFamily, GaussianFamily, PoissonFamily};
    use crate::links::{IdentityLink, LogLink, LogitLink};
    use crate::splines::{bs_basis, is_basis};
    use approx::{assert_abs_diff_eq, assert_relative_eq};
    use ndarray::array;

    fn assert_array1_close(actual: &Array1<f64>, expected: &[f64], epsilon: f64) {
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(*a, *e, epsilon = epsilon);
        }
    }

    fn assert_array2_close(actual: &Array2<f64>, expected: &Array2<f64>, epsilon: f64) {
        assert_eq!(actual.shape(), expected.shape());
        for ((i, j), a) in actual.indexed_iter() {
            assert_abs_diff_eq!(*a, expected[[i, j]], epsilon = epsilon);
        }
    }

    // =========================================================================
    // Unit tests for structs and helpers
    // =========================================================================

    #[test]
    fn test_smooth_term_creation() {
        let x = Array1::from_vec((0..100).map(|i| i as f64 / 10.0).collect());
        let basis = bs_basis(&x, 10, 3, None, false);

        let term = SmoothTermData::new("age".to_string(), basis.clone());

        assert_eq!(term.name, "age");
        assert_eq!(term.k(), 9); // df=10, no intercept = 9 columns
        assert_eq!(term.penalty.shape(), &[9, 9]);
    }

    #[test]
    fn test_embed_penalty() {
        let penalty1 = Array2::eye(5);
        let penalty2 = Array2::eye(3);

        let mut penalty = Array2::zeros((10, 10));
        embed_penalty(&mut penalty, &penalty1, 2, 0.5);
        embed_penalty(&mut penalty, &penalty2, 7, 2.0);

        // Check shape
        assert_eq!(penalty.shape(), &[10, 10]);

        // Check that parametric columns have no penalty
        assert_eq!(penalty[[0, 0]], 0.0);
        assert_eq!(penalty[[1, 1]], 0.0);

        // Check that smooth columns have scaled penalty
        assert_eq!(penalty[[2, 2]], 0.5); // lambda1 * I
        assert_eq!(penalty[[7, 7]], 2.0); // lambda2 * I
    }

    #[test]
    fn smooth_glm_numeric_helper_contracts_are_exact() {
        assert_abs_diff_eq!(block_average(6.0, 3), 2.0, epsilon = 1e-12);
        assert!(pav_blocks_should_merge(1.0, 2.0));
        assert!(!pav_blocks_should_merge(2.0, 2.0));
        assert_array1_close(
            &Array1::from_vec(isotonic_projection(&[3.0, 1.0, 2.0, 5.0], true)),
            &[2.0, 2.0, 2.0, 5.0],
            1e-12,
        );
        assert_array1_close(
            &Array1::from_vec(isotonic_projection(&[1.0, 3.0, 2.0], false)),
            &[2.0, 2.0, 2.0],
            1e-12,
        );

        let mut bounded = array![-1.0, 2.0, -3.0, -0.0, 5.0];
        project_coefficients_to_bounds(&mut bounded, Some(&[0, 3, 5]), Some(&[1, 3, 5]));
        assert_eq!(bounded, array![0.0, 0.0, -3.0, -0.0, 5.0]);
        assert_eq!(bounded[3].to_bits(), (-0.0_f64).to_bits());
        project_coefficients_to_bounds(&mut bounded, None, Some(&[4]));
        assert_eq!(bounded, array![0.0, 0.0, -3.0, -0.0, 0.0]);
        assert_eq!(bounded[3].to_bits(), (-0.0_f64).to_bits());

        assert_abs_diff_eq!(irls_work_weight(2.0, 0.5, 0.1), 2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(
            working_response_value(2.0, 0.5, 4.0, 3.0, 2.0),
            3.5,
            epsilon = 1e-12
        );
        let eta = array![2.0, 3.0];
        let offset = array![0.5, 1.0];
        let y = array![4.0, 1.0];
        let mu = array![3.0, 2.0];
        let prior = array![1.5, 2.0];
        let variance = array![2.0, 4.0];
        let deriv = array![2.0, -1.0];
        let mut irls = Array1::zeros(2);
        let mut combined = Array1::zeros(2);
        let mut z = Array1::zeros(2);
        update_irls_work_arrays(
            &eta,
            &offset,
            &y,
            &mu,
            &prior,
            &variance,
            &deriv,
            0.1,
            &mut irls,
            &mut combined,
            &mut z,
        );
        assert_array1_close(&irls, &[0.125, 0.25], 1e-12);
        assert_array1_close(&combined, &[0.1875, 0.5], 1e-12);
        assert_array1_close(&z, &[3.5, 3.0], 1e-12);
        assert_abs_diff_eq!(weighted_square(3.0, 2.0), 18.0, epsilon = 1e-12);
        assert_abs_diff_eq!(
            weighted_response_sum_squares(&array![2.0, 3.0], &array![0.5, 2.0]),
            20.0,
            epsilon = 1e-12
        );

        assert_eq!(
            blend_arrays(&array![1.0, 5.0], &array![5.0, 1.0], 0.25),
            array![2.0, 4.0]
        );
        assert_eq!(
            blend_step_arrays(&array![f64::NAN, 5.0], &array![5.0, 1.0], 1.0),
            array![5.0, 1.0],
            "exact full steps should take the candidate without mixing stale values"
        );
        assert_eq!(
            blend_step_arrays(&array![1.0, 5.0], &array![5.0, 1.0], 0.5),
            array![3.0, 3.0]
        );
        assert_abs_diff_eq!(smooth_step_fraction_for_trial(0, 20), 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(smooth_step_fraction_for_trial(2, 20), 0.25, epsilon = 1e-12);
        assert_abs_diff_eq!(smooth_step_fraction_for_trial(20, 20), 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(next_smooth_half_step(0.5), 0.25, epsilon = 1e-12);
        assert_abs_diff_eq!(smooth_accept_threshold(10.0, 0.01), 10.11, epsilon = 1e-12);
        assert!(finite_array(&array![1.0, 2.0]));
        assert!(!finite_array(&array![1.0, f64::NAN]));
        assert_eq!(
            initial_projection_if_finite(array![1.0, 2.0]).expect("finite projection"),
            array![1.0, 2.0]
        );
        assert!(initial_projection_if_finite(array![1.0, f64::NAN]).is_none());
        assert!(smooth_trial_values_are_finite(
            &array![1.0],
            &array![2.0],
            3.0,
            4.0
        ));
        assert!(!smooth_trial_values_are_finite(
            &array![1.0],
            &array![2.0],
            3.0,
            f64::INFINITY
        ));
        assert!(smooth_step_candidate_accepted(
            &array![1.0],
            &array![2.0],
            3.0,
            4.0,
            4.0
        ));
        assert!(!smooth_step_candidate_accepted(
            &array![1.0],
            &array![2.0],
            3.0,
            4.1,
            4.0
        ));
        assert!(should_skip_smooth_trial(
            &array![1.0],
            &array![f64::NAN],
            3.0,
            4.0
        ));
        assert!(!should_skip_smooth_trial(
            &array![1.0],
            &array![2.0],
            3.0,
            4.0
        ));

        assert!(smooth_trial_is_better(None, 9.0, 10.0));
        assert!(smooth_trial_is_better(Some(0), 9.0, 10.0));
        assert!(!smooth_trial_is_better(Some(0), 10.0, 10.0));
        assert!(!smooth_trial_is_better(Some(0), 11.0, 10.0));
        assert!(smooth_trial_is_nonworsening(10.0, 10.0));
        assert!(!smooth_trial_is_nonworsening(11.0, 10.0));
        assert!(immediate_trial_accepts(0, 10.0, 10.0));
        assert!(!immediate_trial_accepts(1, 10.0, 10.0));
        assert!(halved_trial_accepts(1, 9.0, 10.0));
        assert!(!halved_trial_accepts(0, 9.0, 10.0));
        assert!(smooth_step_accepted(Some(1), 5.0, 6.0));
        assert!(!smooth_step_accepted(None, 5.0, 6.0));
        assert!(!smooth_step_accepted(Some(1), 7.0, 6.0));
        assert!(!smooth_step_halving_failed(Some(1), 5.0, 6.0));
        assert!(smooth_step_halving_failed(None, 5.0, 6.0));
        assert!(smooth_step_halving_failed(Some(1), 7.0, 6.0));
        assert!(genuine_halved_step(Some(1), 20));
        assert!(!genuine_halved_step(Some(0), 20));
        assert!(!genuine_halved_step(Some(20), 20));

        assert_abs_diff_eq!(
            relative_change_with_unit_offset(10.0, 7.0),
            3.0 / 11.0,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            relative_change_with_floor(10.0, 7.0, 0.1),
            0.3,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            relative_change_with_floor(0.01, 0.04, 0.1),
            0.03,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            relative_change_with_floor(0.1, 0.05, 0.1),
            0.05,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(lambda_relative_change(0.0, 0.02), 0.02, epsilon = 1e-12);
        assert!(change_below_tolerance(0.009, 0.01));
        assert!(!change_below_tolerance(0.01, 0.01));
        assert_abs_diff_eq!(
            max_lambda_relative_change(&[0.0, 10.0], &[0.02, 15.0]),
            0.5,
            epsilon = 1e-12
        );
        assert!(lambdas_stable(0.009));
        assert!(!lambdas_stable(0.01));
        assert!(should_stop_monotonic_outer(0.009, true));
        assert!(!should_stop_monotonic_outer(0.009, false));
        assert!(should_run_smooth_gcv(0, 1));
        assert!(should_run_smooth_gcv(0, 4));
        assert!(!should_run_smooth_gcv(0, 5));
        assert!(!should_run_smooth_gcv(1, 2));
        assert_eq!(next_lambdas_stable_count(0.009, 2), 3);
        assert_eq!(next_lambdas_stable_count(0.01, 2), 0);
        assert!(lambdas_changed(&[1.0, 2.0], &[1.0, 3.0]));
        assert!(!lambdas_changed(&[1.0, 2.0], &[1.0, 2.0]));
        assert_eq!(parametric_column_count(5, 2), 3);
        let width_spec = SmoothTermSpec {
            col_start: 2,
            col_end: 5,
            penalty: Array2::eye(3),
            monotonicity: Monotonicity::Increasing,
            initial_lambda: 1.0,
        };
        assert_eq!(smooth_term_width(&width_spec), 3);
        assert_eq!(
            centered_eta(&array![2.0, 5.0], &array![0.5, 1.5]),
            array![1.5, 3.5]
        );
        assert_eq!(
            linear_predictor_from_coefficients(
                array![[1.0, 2.0], [3.0, 4.0]].view(),
                &array![0.5, -1.0],
                &array![10.0, 20.0],
            ),
            array![8.5, 17.5]
        );
        assert_eq!(smooth_coefficient_index(4, 3), 7);
        assert!(fraction_is_genuine_halving(0.5));
        assert!(!fraction_is_genuine_halving(0.0));
        assert_eq!(clamp_monotonic_alpha_value(0, 25.0), 25.0);
        assert_eq!(clamp_monotonic_alpha_value(1, 25.0), MAX_EXP_ALPHA);
        assert_eq!(clamp_monotonic_alpha_value(1, -25.0), -MAX_EXP_ALPHA);
        assert_eq!(
            smooth_solver_status(true, true, true),
            "step_halving_no_improvement",
            "step-halving failure takes precedence over the convergence flag"
        );
        assert_eq!(smooth_solver_status(false, true, true), "converged");
        assert_eq!(
            smooth_solver_status(false, true, false),
            "stalled_nonstationary",
            "a progress-converged but non-stationary exit must not report converged"
        );
        assert_eq!(smooth_solver_status(false, false, true), "max_iterations");
        assert_eq!(smooth_solver_status(false, false, false), "max_iterations");
        assert!(should_warn_smooth_nonconvergence(false));
        assert!(!should_warn_smooth_nonconvergence(true));
        assert_eq!(
            smooth_nonconvergence_warning("max_iterations"),
            "Smooth GLM did not converge (status: max_iterations). Results may be approximate."
        );

        let penalty = array![[2.0, 0.5], [0.5, 1.0]];
        let coeffs = array![2.0, 3.0];
        assert_abs_diff_eq!(
            smooth_penalty_contribution(0.5, coeffs.view(), &penalty),
            11.5,
            epsilon = 1e-12
        );
        let spec = SmoothTermSpec {
            col_start: 1,
            col_end: 3,
            penalty,
            monotonicity: Monotonicity::None,
            initial_lambda: 0.5,
        };
        assert_abs_diff_eq!(
            smooth_penalized_deviance(7.0, &array![99.0, 2.0, 3.0], &[spec], &[0.5]),
            18.5,
            epsilon = 1e-12
        );
        assert_eq!(
            add_penalty_to_xtwx(
                &array![[1.0, 2.0], [3.0, 4.0]],
                &array![[0.5, 1.5], [2.5, 3.5]]
            ),
            array![[1.5, 3.5], [5.5, 7.5]]
        );

        let offset_mono = monotonic_offset(&array![3.0, 5.0], &array![1.0, 2.0]);
        assert_eq!(offset_mono, array![2.0, 3.0]);
        let mut adjusted = array![10.0, 20.0];
        subtract_smooth_offset(
            &mut adjusted,
            array![[1.0, 2.0], [3.0, 4.0]].view(),
            &offset_mono,
        );
        assert_eq!(adjusted, array![2.0, 2.0]);
    }

    #[test]
    fn test_smooth_term_with_monotonicity() {
        let x = Array1::from_vec((0..50).map(|i| i as f64 / 5.0).collect());
        let basis = bs_basis(&x, 8, 3, None, false);

        let term = SmoothTermData::new("age".to_string(), basis)
            .with_monotonicity(Monotonicity::Increasing)
            .with_lambda(2.5);

        assert!(term.is_monotonic());
        assert_eq!(term.monotonicity, Monotonicity::Increasing);
        assert_eq!(term.initial_lambda, 2.5);
    }

    #[test]
    fn test_alpha_to_beta_none_empty_and_clamped_monotone_paths() {
        assert_eq!(
            alpha_to_beta(&[], &Monotonicity::Increasing).len(),
            0,
            "empty alpha should return an empty beta"
        );

        let none = alpha_to_beta(&[1.0, 2.0, 3.0], &Monotonicity::None);
        assert_array1_close(&none, &[1.0, 2.0, 3.0], 1e-12);

        let increasing = alpha_to_beta(&[1.0, 0.0, 30.0], &Monotonicity::Increasing);
        assert_array1_close(&increasing, &[1.0, 2.0, 2.0 + 20.0_f64.exp()], 1e-6);

        let decreasing = alpha_to_beta(&[1.0, 0.0, -30.0], &Monotonicity::Decreasing);
        assert_array1_close(&decreasing, &[1.0, 0.0, -(-20.0_f64).exp()], 1e-12);
    }

    #[test]
    fn test_isotonic_projection_handles_empty_identity_and_pooling() {
        assert!(isotonic_projection(&[], true).is_empty());
        assert_eq!(
            isotonic_projection(&[1.0, 2.0, 3.0], true),
            vec![1.0, 2.0, 3.0]
        );
        assert_eq!(
            isotonic_projection(&[3.0, 1.0, 2.0], true),
            vec![2.0, 2.0, 2.0],
            "PAV should pool adjacent increasing violations"
        );
        assert_eq!(
            isotonic_projection(&[3.0, 2.0, 1.0], false),
            vec![3.0, 2.0, 1.0]
        );
        assert_eq!(
            isotonic_projection(&[1.0, 3.0, 2.0], false),
            vec![2.0, 2.0, 2.0],
            "decreasing projection should be symmetric to increasing projection"
        );
    }

    #[test]
    fn test_beta_to_alpha_projects_and_recovers_monotone_sequences() {
        assert_eq!(
            beta_to_alpha(&[], &Monotonicity::Decreasing).len(),
            0,
            "empty beta should return an empty alpha"
        );

        let none = beta_to_alpha(&[1.0, 2.0, 3.0], &Monotonicity::None);
        assert_array1_close(&none, &[1.0, 2.0, 3.0], 1e-12);

        let increasing = beta_to_alpha(&[1.0, 2.0, 4.0], &Monotonicity::Increasing);
        assert_array1_close(&increasing, &[1.0, 0.0, 2.0_f64.ln()], 1e-12);
        let beta_roundtrip = alpha_to_beta(
            increasing.as_slice().expect("contiguous array"),
            &Monotonicity::Increasing,
        );
        assert_array1_close(&beta_roundtrip, &[1.0, 2.0, 4.0], 1e-12);

        let decreasing = beta_to_alpha(&[4.0, 2.0, 1.0], &Monotonicity::Decreasing);
        assert_array1_close(&decreasing, &[4.0, 2.0_f64.ln(), 0.0], 1e-12);
        let beta_roundtrip = alpha_to_beta(
            decreasing.as_slice().expect("contiguous array"),
            &Monotonicity::Decreasing,
        );
        assert_array1_close(&beta_roundtrip, &[4.0, 2.0, 1.0], 1e-12);

        let projected = beta_to_alpha(&[3.0, 1.0, 2.0], &Monotonicity::Increasing);
        assert_array1_close(&projected, &[2.0, ZERO_TOL.ln(), ZERO_TOL.ln()], 1e-12);
    }

    #[test]
    fn test_compute_monotonic_jacobian_contracts() {
        let identity = compute_monotonic_jacobian(&[5.0, 7.0, 9.0], &Monotonicity::None);
        assert_array2_close(&identity, &Array2::eye(3), 1e-12);

        let alpha = [5.0, 2.0_f64.ln(), 3.0_f64.ln()];
        let increasing = compute_monotonic_jacobian(&alpha, &Monotonicity::Increasing);
        let expected_increasing = array![[1.0, 0.0, 0.0], [1.0, 2.0, 0.0], [1.0, 2.0, 3.0]];
        assert_array2_close(&increasing, &expected_increasing, 1e-12);

        let decreasing = compute_monotonic_jacobian(&alpha, &Monotonicity::Decreasing);
        let expected_decreasing = array![[1.0, 0.0, 0.0], [1.0, -2.0, 0.0], [1.0, -2.0, -3.0]];
        assert_array2_close(&decreasing, &expected_decreasing, 1e-12);

        let clamped = compute_monotonic_jacobian(&[0.0, 30.0], &Monotonicity::Increasing);
        assert_relative_eq!(clamped[[1, 1]], 20.0_f64.exp(), max_relative = 1e-12);
    }

    #[test]
    fn test_compute_x_tilde_inplace_matches_cumulative_jacobian_product() {
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let alpha = [0.0, 2.0_f64.ln(), 3.0_f64.ln()];

        let mut copied = Array2::zeros((2, 3));
        compute_x_tilde_inplace(
            &x.view(),
            &alpha,
            &Monotonicity::None,
            &mut copied.view_mut(),
        );
        assert_array2_close(&copied, &x, 1e-12);

        let mut increasing = Array2::zeros((2, 3));
        compute_x_tilde_inplace(
            &x.view(),
            &alpha,
            &Monotonicity::Increasing,
            &mut increasing.view_mut(),
        );
        let expected_increasing = array![[6.0, 10.0, 9.0], [15.0, 22.0, 18.0]];
        assert_array2_close(&increasing, &expected_increasing, 1e-12);

        let mut decreasing = Array2::zeros((2, 3));
        compute_x_tilde_inplace(
            &x.view(),
            &alpha,
            &Monotonicity::Decreasing,
            &mut decreasing.view_mut(),
        );
        let expected_decreasing = array![[6.0, -10.0, -9.0], [15.0, -22.0, -18.0]];
        assert_array2_close(&decreasing, &expected_decreasing, 1e-12);
    }

    #[test]
    fn test_compute_s_tilde_matches_j_transpose_s_j() {
        let penalty = array![[2.0, 1.0], [1.0, 3.0]];
        let jacobian = array![[1.0, 0.0], [1.0, 2.0]];
        let transformed = compute_s_tilde(&penalty, &jacobian);
        let expected = array![[7.0, 8.0], [8.0, 12.0]];
        assert_array2_close(&transformed, &expected, 1e-12);
    }

    #[test]
    fn test_smooth_term_data_and_spec_monotonicity_contracts() {
        let plain = SmoothTermData::new("plain".to_string(), Array2::ones((4, 3)));
        assert!(!plain.is_monotonic());
        assert_eq!(plain.k(), 3);
        assert_eq!(plain.initial_lambda, 1.0);
        assert_eq!(plain.penalty.shape(), &[3, 3]);

        let decreasing = plain
            .clone()
            .with_lambda(0.25)
            .with_monotonicity(Monotonicity::Decreasing);
        assert!(decreasing.is_monotonic());
        assert_eq!(decreasing.initial_lambda, 0.25);
        assert_eq!(decreasing.monotonicity, Monotonicity::Decreasing);

        let penalty = Array2::eye(2);
        let unconstrained = SmoothTermSpec {
            col_start: 1,
            col_end: 3,
            penalty: penalty.clone(),
            monotonicity: Monotonicity::None,
            initial_lambda: 1.0,
        };
        assert!(!unconstrained.is_monotonic());

        let constrained = SmoothTermSpec {
            monotonicity: Monotonicity::Increasing,
            ..unconstrained
        };
        assert!(constrained.is_monotonic());
    }

    #[test]
    fn test_invert_matrix_returns_inverse_or_none_for_singular_input() {
        let invertible = array![[4.0, 7.0], [2.0, 6.0]];
        let inverse = invert_matrix(&invertible).expect("matrix should be invertible");
        let expected = array![[0.6, -0.7], [-0.2, 0.4]];
        assert_array2_close(&inverse, &expected, 1e-12);

        let singular = array![[1.0, 2.0], [2.0, 4.0]];
        assert!(invert_matrix(&singular).is_none());
    }

    #[test]
    fn test_assemble_smooth_result_from_specs_preserves_fields_and_penalty_metadata() {
        let coefficients = array![0.5, 0.25, -0.1];
        let mu = array![1.1, 1.9, 3.0];
        let eta = array![1.0, 2.0, 3.1];
        let y = array![1.0, 2.0, 3.0];
        let prior_weights = array![1.0, 2.0, 1.0];
        let final_weights = array![1.5, 2.5, 3.5];
        let x = array![[1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]];
        let penalty = Array2::eye(2);
        let lambdas = vec![0.5];
        let offset = array![0.1, 0.2, 0.3];
        let cov = array![[2.0, 0.1, 0.2], [0.1, 3.0, 0.3], [0.2, 0.3, 4.0],];

        let result = assemble_smooth_result_from_specs(
            coefficients.clone(),
            mu.clone(),
            eta.clone(),
            4.0,
            7,
            true,
            &final_weights,
            x.view(),
            &[(&penalty, 1, 3)],
            &lambdas,
            1,
            "gaussian",
            prior_weights.clone(),
            &y,
            Some(&offset),
            Some(cov.clone()),
            vec!["kept warning".to_string()],
            true,
            "converged".to_string(),
            true,
            0.0,
        );

        assert_array1_close(
            &result.coefficients,
            coefficients
                .as_slice()
                .expect("test coefficients should be contiguous"),
            1e-12,
        );
        assert_array1_close(
            &result.fitted_values,
            mu.as_slice().expect("test mu should be contiguous"),
            1e-12,
        );
        assert_array1_close(
            &result.linear_predictor,
            eta.as_slice().expect("test eta should be contiguous"),
            1e-12,
        );
        assert_abs_diff_eq!(result.deviance, 4.0, epsilon = 1e-12);
        assert_eq!(result.iterations, 7);
        assert!(result.converged);
        assert_eq!(result.lambdas, lambdas);
        assert_eq!(result.smooth_edfs.len(), 1);
        assert_abs_diff_eq!(
            result.total_edf,
            1.0 + result.smooth_edfs[0],
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            result.gcv,
            gcv_score(result.deviance, y.len(), result.total_edf),
            epsilon = 1e-12
        );
        assert_array2_close(&result.covariance_unscaled, &cov, 1e-12);
        assert_eq!(result.family_name, "gaussian");
        assert_array1_close(
            &result.irls_weights,
            final_weights
                .as_slice()
                .expect("test final weights should be contiguous"),
            1e-12,
        );
        assert_array1_close(
            &result.prior_weights,
            prior_weights
                .as_slice()
                .expect("test prior weights should be contiguous"),
            1e-12,
        );
        assert_array2_close(&result.design_matrix, &x, 1e-12);
        assert_array1_close(
            &result.y,
            y.as_slice().expect("test y should be contiguous"),
            1e-12,
        );
        assert_array1_close(
            result.offset.as_ref().expect("offset should be cloned"),
            offset.as_slice().expect("test offset should be contiguous"),
            1e-12,
        );
        assert_eq!(result.warnings, vec!["kept warning"]);
        assert!(result.step_halving_used);
        assert_eq!(result.solver_status, "converged");

        let smooth_penalty = result.penalty.as_smooth().expect("smooth penalty result");
        assert_eq!(smooth_penalty.n_terms(), 1);
        assert_eq!(smooth_penalty.lambdas, vec![0.5]);
        assert_eq!(smooth_penalty.term_indices[0], 1..3);
        assert_array2_close(&smooth_penalty.penalty_matrices[0], &penalty, 1e-12);
    }

    #[test]
    fn test_assemble_smooth_result_falls_back_to_identity_covariance_for_singular_system() {
        let coefficients = array![0.0, 0.0];
        let mu = array![0.0, 0.0];
        let eta = array![0.0, 0.0];
        let y = array![0.0, 0.0];
        let weights = array![1.0, 1.0];
        let x = Array2::zeros((2, 2));
        let penalty = Array2::zeros((2, 2));

        let result = assemble_smooth_result_from_specs(
            coefficients,
            mu,
            eta,
            0.0,
            0,
            false,
            &weights,
            x.view(),
            &[(&penalty, 0, 2)],
            &[0.0],
            0,
            "gaussian",
            weights.clone(),
            &y,
            None,
            None,
            Vec::new(),
            false,
            "max_iterations".to_string(),
            false,
            f64::NAN,
        );

        assert_array2_close(&result.covariance_unscaled, &Array2::eye(2), 1e-12);
        assert_eq!(result.offset, None);
        assert_eq!(result.total_edf, 2.0);
    }

    // =========================================================================
    // Unit tests: termination-contract helpers (stationarity check, intercept
    // refresh, unconstrained-block re-solve)
    // =========================================================================

    #[test]
    fn test_guarded_denominator_preserves_sign_and_floors_magnitude() {
        assert_eq!(guarded_denominator(0.5), 0.5);
        assert_eq!(guarded_denominator(-0.5), -0.5);
        assert_eq!(guarded_denominator(1e-14), ZERO_TOL);
        assert_eq!(guarded_denominator(-1e-14), -ZERO_TOL);
    }

    #[test]
    fn test_find_intercept_column_respects_smooth_and_constraint_exclusions() {
        // col 0: all-ones but sign-constrained; col 1: all-ones but inside a
        // smooth range; col 2: the true intercept; col 3: data.
        let x = array![
            [1.0, 1.0, 1.0, 0.2],
            [1.0, 1.0, 1.0, 0.7],
            [1.0, 1.0, 1.0, 0.4]
        ];
        let mut smooth_cols = std::collections::HashSet::new();
        smooth_cols.insert(1usize);
        assert_eq!(
            find_intercept_column(x.view(), &smooth_cols, Some(&[0]), None),
            Some(2)
        );
        assert_eq!(
            find_intercept_column(x.view(), &smooth_cols, Some(&[0]), Some(&[2])),
            None,
            "all candidate ones-columns excluded"
        );
        let no_ones = array![[0.5, 2.0], [1.0, 3.0]];
        let empty = std::collections::HashSet::new();
        assert_eq!(
            find_intercept_column(no_ones.view(), &empty, None, None),
            None
        );
    }

    #[test]
    fn test_quasi_score_arrays_reduce_to_weighted_residuals_for_canonical_logit() {
        let y = array![1.0, 0.0, 1.0];
        let mu = array![0.8, 0.3, 0.5];
        let prior_weights = array![1.0, 2.0, 1.0];
        let (score_resid, info_weight) =
            quasi_score_arrays(&y, &mu, &prior_weights, &BinomialFamily, &LogitLink);
        // Canonical link: g'(mu) V(mu) = 1, so r_i = w_i (y_i - mu_i).
        assert_abs_diff_eq!(score_resid[0], 0.2, epsilon = 1e-12);
        assert_abs_diff_eq!(score_resid[1], -0.6, epsilon = 1e-12);
        assert_abs_diff_eq!(score_resid[2], 0.5, epsilon = 1e-12);
        // Expected-information weight f_i = w_i mu (1 - mu).
        assert_abs_diff_eq!(info_weight[0], 0.16, epsilon = 1e-12);
        assert_abs_diff_eq!(info_weight[1], 0.42, epsilon = 1e-12);
        assert_abs_diff_eq!(info_weight[2], 0.25, epsilon = 1e-12);
        // Column score of an intercept column: s = sum(r), I = sum(f).
        let x = array![[1.0], [1.0], [1.0]];
        let (score, std_score) = column_std_score(x.view(), 0, &score_resid, &info_weight);
        assert_abs_diff_eq!(score, 0.1, epsilon = 1e-12);
        assert_abs_diff_eq!(
            std_score,
            0.1 / (0.16f64 + 0.42 + 0.25).sqrt(),
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_stationarity_report_passes_at_score_zero_and_flags_biased_mean() {
        let x = array![[1.0], [1.0], [1.0], [1.0]];
        let y = array![1.0, 0.0, 1.0, 0.0];
        let prior_weights = Array1::ones(4);
        let empty = std::collections::HashSet::new();
        let coefficients = array![0.0];

        // mu == mean(y): the intercept score vanishes exactly.
        let mu_exact = Array1::from_elem(4, 0.5);
        let (stationary, max_std_score) = stationarity_report(
            x.view(),
            &y,
            &mu_exact,
            &prior_weights,
            &BinomialFamily,
            &LogitLink,
            &empty,
            &coefficients,
            None,
            None,
        );
        assert!(stationary);
        assert_abs_diff_eq!(max_std_score, 0.0, epsilon = 1e-12);

        // Biased mean (the historical failure signature): flagged, and the
        // standardized score matches the hand computation.
        let mu_biased = Array1::from_elem(4, 0.35);
        let (stationary, max_std_score) = stationarity_report(
            x.view(),
            &y,
            &mu_biased,
            &prior_weights,
            &BinomialFamily,
            &LogitLink,
            &empty,
            &coefficients,
            None,
            None,
        );
        assert!(!stationary);
        // score = 4 * 0.15 = 0.6; info = 4 * 0.35 * 0.65 = 0.91.
        assert_abs_diff_eq!(max_std_score, 0.6 / 0.91f64.sqrt(), epsilon = 1e-10);

        // The same violating column inside a smooth range is penalized and
        // therefore not part of the check.
        let mut smooth_cols = std::collections::HashSet::new();
        smooth_cols.insert(0usize);
        let (stationary, max_std_score) = stationarity_report(
            x.view(),
            &y,
            &mu_biased,
            &prior_weights,
            &BinomialFamily,
            &LogitLink,
            &smooth_cols,
            &coefficients,
            None,
            None,
        );
        assert!(stationary);
        assert_abs_diff_eq!(max_std_score, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_stationarity_report_complementary_slackness_for_sign_constraints() {
        let x = array![[1.0], [1.0], [1.0], [1.0]];
        let prior_weights = Array1::ones(4);
        let empty = std::collections::HashSet::new();
        let coefficients = array![0.0]; // constrained coordinate at its bound
        let mu = Array1::from_elem(4, 0.5);

        // Score pushes INTO a beta >= 0 bound (negative score at beta = 0):
        // stationary by complementary slackness.
        let y_low = array![0.0, 0.0, 1.0, 0.0];
        let (stationary, _) = stationarity_report(
            x.view(),
            &y_low,
            &mu,
            &prior_weights,
            &BinomialFamily,
            &LogitLink,
            &empty,
            &coefficients,
            Some(&[0]),
            None,
        );
        assert!(stationary);

        // Score pushes OUT of the bound (positive): a genuine KKT violation.
        let y_high = array![1.0, 1.0, 1.0, 0.0];
        let (stationary, max_std_score) = stationarity_report(
            x.view(),
            &y_high,
            &mu,
            &prior_weights,
            &BinomialFamily,
            &LogitLink,
            &empty,
            &coefficients,
            Some(&[0]),
            None,
        );
        assert!(!stationary);
        assert!(max_std_score > SMOOTH_KKT_SCORE_TOL);

        // Mirrored for beta <= 0: the same positive score pushes INTO that
        // bound and is admissible.
        let (stationary, _) = stationarity_report(
            x.view(),
            &y_high,
            &mu,
            &prior_weights,
            &BinomialFamily,
            &LogitLink,
            &empty,
            &coefficients,
            None,
            Some(&[0]),
        );
        assert!(stationary);
    }

    #[test]
    fn test_solve_intercept_shift_matches_closed_forms() {
        // Binomial/logit at eta = 0: the root of sum(y - sigmoid(d)) = 0 is
        // logit(mean(y)).
        let mut yv = vec![1.0; 7];
        yv.extend(vec![0.0; 3]);
        let y = Array1::from_vec(yv);
        let eta = Array1::zeros(10);
        let prior_weights = Array1::ones(10);
        let shift = solve_intercept_shift(&y, &eta, &prior_weights, &BinomialFamily, &LogitLink)
            .expect("root is bracketed");
        assert_abs_diff_eq!(shift, (0.7f64 / 0.3).ln(), epsilon = 1e-9);

        // Gaussian/identity: the root is mean(y) - mean(eta).
        let y2 = array![1.0, 2.0, 6.0];
        let eta2 = Array1::zeros(3);
        let pw2 = Array1::ones(3);
        let shift2 = solve_intercept_shift(&y2, &eta2, &pw2, &GaussianFamily, &IdentityLink)
            .expect("root is bracketed");
        assert_abs_diff_eq!(shift2, 3.0, epsilon = 1e-9);
    }

    #[test]
    fn test_refine_unconstrained_block_solves_conditional_optimum() {
        // Gaussian identity, x = [intercept | monotone column held fixed].
        // y is exactly linear, so refining the free block (the intercept)
        // must land on the conditional — here global — optimum.
        let n = 12;
        let mut xv = Vec::with_capacity(n * 2);
        let mut yv = Vec::with_capacity(n);
        for i in 0..n {
            let c = i as f64 / n as f64;
            xv.push(1.0);
            xv.push(c);
            yv.push(2.0 + 3.0 * c);
        }
        let x = Array2::from_shape_vec((n, 2), xv).expect("test setup should be valid");
        let y = Array1::from_vec(yv);
        let specs = vec![SmoothTermSpec {
            col_start: 1,
            col_end: 2,
            penalty: Array2::zeros((1, 1)),
            monotonicity: Monotonicity::Increasing,
            initial_lambda: 0.0,
        }];
        let lambdas = vec![0.0];
        let offset = Array1::zeros(n);
        let prior_weights = Array1::ones(n);

        let mut coefficients = array![0.0, 3.0]; // slope at truth, intercept off
        let mut eta = x.dot(&coefficients);
        let mut mu = eta.clone();
        let mut deviance = GaussianFamily.deviance(&y, &mu, Some(&prior_weights));
        let mut iteration = 0usize;
        let refined = refine_unconstrained_block(
            &y,
            x.view(),
            &specs,
            &lambdas,
            &GaussianFamily,
            &IdentityLink,
            &offset,
            &prior_weights,
            1e-10,
            1e-8,
            None,
            None,
            &mut coefficients,
            &mut eta,
            &mut mu,
            &mut deviance,
            &mut iteration,
            50,
        )
        .expect("refine should not error");
        assert!(refined);
        assert_abs_diff_eq!(coefficients[0], 2.0, epsilon = 1e-8);
        assert_abs_diff_eq!(coefficients[1], 3.0, epsilon = 1e-12); // held fixed
        assert!(deviance < 1e-12);
        assert!(iteration >= 1);

        // Exhausted budget: no-op, state untouched.
        let mut coef2 = array![0.0, 3.0];
        let mut eta2 = x.dot(&coef2);
        let mut mu2 = eta2.clone();
        let mut dev2 = GaussianFamily.deviance(&y, &mu2, Some(&prior_weights));
        let mut iter_full = 50usize;
        let refined2 = refine_unconstrained_block(
            &y,
            x.view(),
            &specs,
            &lambdas,
            &GaussianFamily,
            &IdentityLink,
            &offset,
            &prior_weights,
            1e-10,
            1e-8,
            None,
            None,
            &mut coef2,
            &mut eta2,
            &mut mu2,
            &mut dev2,
            &mut iter_full,
            50,
        )
        .expect("refine should not error");
        assert!(!refined2);
        assert_abs_diff_eq!(coef2[0], 0.0, epsilon = 0.0);

        // No fixed block (nothing monotone): the main loop owns the whole
        // problem and the refine step declines.
        let free_specs = vec![SmoothTermSpec {
            col_start: 1,
            col_end: 2,
            penalty: Array2::zeros((1, 1)),
            monotonicity: Monotonicity::None,
            initial_lambda: 0.0,
        }];
        let mut coef3 = array![0.0, 3.0];
        let mut eta3 = x.dot(&coef3);
        let mut mu3 = eta3.clone();
        let mut dev3 = GaussianFamily.deviance(&y, &mu3, Some(&prior_weights));
        let mut iter3 = 0usize;
        let refined3 = refine_unconstrained_block(
            &y,
            x.view(),
            &free_specs,
            &lambdas,
            &GaussianFamily,
            &IdentityLink,
            &offset,
            &prior_weights,
            1e-10,
            1e-8,
            None,
            None,
            &mut coef3,
            &mut eta3,
            &mut mu3,
            &mut dev3,
            &mut iter3,
            50,
        )
        .expect("refine should not error");
        assert!(!refined3);
    }

    // =========================================================================
    // Integration tests: fit_smooth_glm_full_matrix (unified entry point)
    // =========================================================================

    /// Helper: generate Gaussian data with a smooth sin(x) effect.
    fn gaussian_smooth_data(n: usize) -> (Array1<f64>, Array2<f64>, Array1<f64>) {
        let x_vals: Array1<f64> = (0..n).map(|i| i as f64 * 10.0 / n as f64).collect();
        let y: Array1<f64> = x_vals
            .iter()
            .map(|&xi| 2.0 + xi.sin() + 0.1 * (xi * 7.3).sin())
            .collect();
        // Parametric part: intercept column
        let x_param = Array2::from_shape_fn((n, 1), |(_, _)| 1.0);
        (y, x_param, x_vals)
    }

    /// Helper: generate Poisson data with a smooth effect.
    fn poisson_smooth_data(n: usize) -> (Array1<f64>, Array2<f64>, Array1<f64>) {
        let x_vals: Array1<f64> = (0..n).map(|i| i as f64 * 10.0 / n as f64).collect();
        let y: Array1<f64> = x_vals
            .iter()
            .map(|&xi| {
                let mu = (0.5 + 0.3 * xi.sin()).exp();
                // Deterministic "Poisson-like" values (round to nearest int, min 0)
                (mu + 0.5).floor().max(0.0)
            })
            .collect();
        let x_param = Array2::from_shape_fn((n, 1), |(_, _)| 1.0);
        (y, x_param, x_vals)
    }

    /// Helper: concatenate parametric + basis into full design matrix and build SmoothTermSpec.
    fn make_full_matrix(
        x_param: &Array2<f64>,
        basis: &Array2<f64>,
    ) -> (Array2<f64>, Vec<SmoothTermSpec>) {
        let p_param = x_param.ncols();
        let k = basis.ncols();
        let x_full = ndarray::concatenate![ndarray::Axis(1), *x_param, *basis]
            .as_standard_layout()
            .to_owned();
        let spec = SmoothTermSpec {
            col_start: p_param,
            col_end: p_param + k,
            penalty: crate::splines::penalized::penalty_matrix(k, 2),
            monotonicity: Monotonicity::None,
            initial_lambda: 1.0,
        };
        (x_full, vec![spec])
    }

    fn make_full_matrix_with_monotonicity(
        x_param: &Array2<f64>,
        basis: &Array2<f64>,
        monotonicity: Monotonicity,
    ) -> (Array2<f64>, Vec<SmoothTermSpec>) {
        let p_param = x_param.ncols();
        let k = basis.ncols();
        let x_full = ndarray::concatenate![ndarray::Axis(1), *x_param, *basis]
            .as_standard_layout()
            .to_owned();
        let spec = SmoothTermSpec {
            col_start: p_param,
            col_end: p_param + k,
            penalty: crate::splines::penalized::penalty_matrix(k, 2),
            monotonicity,
            initial_lambda: 1.0,
        };
        (x_full, vec![spec])
    }

    fn smooth_coefficients_are_monotone(result: &SmoothGLMResult, spec: &SmoothTermSpec) -> bool {
        let coef = result
            .coefficients
            .as_slice()
            .expect("contiguous coefficients");
        let smooth = &coef[spec.col_start..spec.col_end];
        match spec.monotonicity {
            Monotonicity::Increasing => smooth.windows(2).all(|w| w[1] >= w[0] - 1e-10),
            Monotonicity::Decreasing => smooth.windows(2).all(|w| w[1] <= w[0] + 1e-10),
            Monotonicity::None => true,
        }
    }

    #[test]
    fn test_fit_smooth_glm_gaussian_converges() {
        let (y, x_param, x_vals) = gaussian_smooth_data(100);
        let basis = bs_basis(&x_vals, 10, 3, None, false);
        let (x_full, specs) = make_full_matrix(&x_param, &basis);
        let config = SmoothGLMConfig::default();

        let result = fit_smooth_glm_full_matrix(
            &y,
            x_full.view(),
            &specs,
            &GaussianFamily,
            &IdentityLink,
            &config,
            None,
            None,
            None,
            None,
        )
        .expect("test setup should be valid");

        assert!(result.converged, "Gaussian smooth GLM should converge");
        assert!(result.deviance > 0.0);
        assert!(result.iterations > 0);
    }

    #[test]
    fn test_fit_smooth_glm_monotonic_increasing_runs_nested_pirls_path() {
        let n = 60;
        let x_vals: Array1<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        let y: Array1<f64> = x_vals
            .iter()
            .map(|&x| 1.0 + 1.5 * x + 0.05 * (12.0 * x).sin())
            .collect();
        let x_param = Array2::from_shape_fn((n, 1), |(_, _)| 1.0);
        let basis = is_basis(&x_vals, 6, 3, Some((0.0, 1.0)), true);
        let (x_full, specs) =
            make_full_matrix_with_monotonicity(&x_param, &basis, Monotonicity::Increasing);
        let mut config = SmoothGLMConfig::default();
        config.irls_config.max_iterations = 25;
        config.lambda_tol = 1e-3;

        let result = fit_smooth_glm_full_matrix(
            &y,
            x_full.view(),
            &specs,
            &GaussianFamily,
            &IdentityLink,
            &config,
            None,
            None,
            None,
            None,
        )
        .expect("monotonic increasing fit should produce a finite result");

        assert!(result.iterations > 0);
        assert!(result.deviance.is_finite());
        assert!(result.coefficients.iter().all(|v| v.is_finite()));
        assert_eq!(result.lambdas.len(), 1);
        assert_eq!(result.smooth_edfs.len(), 1);
        assert!(result.lambdas[0].is_finite() && result.lambdas[0] > 0.0);
        assert!(
            smooth_coefficients_are_monotone(&result, &specs[0]),
            "increasing monotonic reparameterization must return ordered smooth coefficients"
        );
    }

    #[test]
    fn test_fit_smooth_glm_monotonic_decreasing_runs_nested_pirls_path() {
        let n = 60;
        let x_vals: Array1<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        let y: Array1<f64> = x_vals
            .iter()
            .map(|&x| 3.0 - 1.25 * x + 0.03 * (10.0 * x).cos())
            .collect();
        let x_param = Array2::from_shape_fn((n, 1), |(_, _)| 1.0);
        let basis = is_basis(&x_vals, 6, 3, Some((0.0, 1.0)), false);
        let (x_full, specs) =
            make_full_matrix_with_monotonicity(&x_param, &basis, Monotonicity::Decreasing);
        let mut config = SmoothGLMConfig::default();
        config.irls_config.max_iterations = 25;
        config.lambda_tol = 1e-3;

        let result = fit_smooth_glm_full_matrix(
            &y,
            x_full.view(),
            &specs,
            &GaussianFamily,
            &IdentityLink,
            &config,
            None,
            None,
            None,
            None,
        )
        .expect("monotonic decreasing fit should produce a finite result");

        assert!(result.iterations > 0);
        assert!(result.deviance.is_finite());
        assert!(result.coefficients.iter().all(|v| v.is_finite()));
        assert_eq!(result.lambdas.len(), 1);
        assert_eq!(result.smooth_edfs.len(), 1);
        assert!(result.lambdas[0].is_finite() && result.lambdas[0] > 0.0);
        assert!(
            smooth_coefficients_are_monotone(&result, &specs[0]),
            "decreasing monotonic reparameterization must return ordered smooth coefficients"
        );
    }

    #[test]
    fn test_fit_smooth_glm_gaussian_edf_less_than_k() {
        let (y, x_param, x_vals) = gaussian_smooth_data(200);
        let basis = bs_basis(&x_vals, 10, 3, None, false);
        let k = basis.ncols();
        let (x_full, specs) = make_full_matrix(&x_param, &basis);
        let config = SmoothGLMConfig::default();

        let result = fit_smooth_glm_full_matrix(
            &y,
            x_full.view(),
            &specs,
            &GaussianFamily,
            &IdentityLink,
            &config,
            None,
            None,
            None,
            None,
        )
        .expect("test setup should be valid");

        assert_eq!(result.smooth_edfs.len(), 1);
        assert!(
            result.smooth_edfs[0] > 1.0,
            "EDF should be > 1 for non-trivial smooth"
        );
        assert!(
            result.smooth_edfs[0] < k as f64,
            "EDF {} should be < k {}",
            result.smooth_edfs[0],
            k
        );
        assert!(result.total_edf > 1.0);
    }

    #[test]
    fn test_fit_smooth_glm_gaussian_gcv_positive() {
        let (y, x_param, x_vals) = gaussian_smooth_data(100);
        let basis = bs_basis(&x_vals, 10, 3, None, false);
        let (x_full, specs) = make_full_matrix(&x_param, &basis);
        let config = SmoothGLMConfig::default();

        let result = fit_smooth_glm_full_matrix(
            &y,
            x_full.view(),
            &specs,
            &GaussianFamily,
            &IdentityLink,
            &config,
            None,
            None,
            None,
            None,
        )
        .expect("test setup should be valid");

        assert!(result.gcv > 0.0, "GCV should be positive");
        assert_eq!(result.lambdas.len(), 1);
        assert!(
            result.lambdas[0] > 0.0,
            "Selected lambda should be positive"
        );
    }

    #[test]
    fn test_fit_smooth_glm_poisson_converges() {
        let (y, x_param, x_vals) = poisson_smooth_data(200);
        let basis = bs_basis(&x_vals, 8, 3, None, false);
        let (x_full, specs) = make_full_matrix(&x_param, &basis);
        let config = SmoothGLMConfig::default();

        let result = fit_smooth_glm_full_matrix(
            &y,
            x_full.view(),
            &specs,
            &PoissonFamily,
            &LogLink,
            &config,
            None,
            None,
            None,
            None,
        )
        .expect("test setup should be valid");

        assert!(result.converged, "Poisson smooth GLM should converge");
        assert!(
            result.fitted_values.iter().all(|&v| v > 0.0),
            "Poisson fitted values must be positive"
        );
    }

    #[test]
    fn test_fit_smooth_glm_covariance_shape() {
        let (y, x_param, x_vals) = gaussian_smooth_data(100);
        let basis = bs_basis(&x_vals, 8, 3, None, false);
        let p_total = x_param.ncols() + basis.ncols();
        let (x_full, specs) = make_full_matrix(&x_param, &basis);
        let config = SmoothGLMConfig::default();

        let result = fit_smooth_glm_full_matrix(
            &y,
            x_full.view(),
            &specs,
            &GaussianFamily,
            &IdentityLink,
            &config,
            None,
            None,
            None,
            None,
        )
        .expect("test setup should be valid");

        assert_eq!(result.covariance_unscaled.shape(), &[p_total, p_total]);
        assert_eq!(result.coefficients.len(), p_total);
    }

    #[test]
    fn test_fit_smooth_glm_with_offset() {
        let n = 100;
        let (y, x_param, x_vals) = poisson_smooth_data(n);
        let basis = bs_basis(&x_vals, 8, 3, None, false);
        let (x_full, specs) = make_full_matrix(&x_param, &basis);
        let offset = Array1::from_vec(vec![0.5; n]);
        let mut config = SmoothGLMConfig::default();
        config.irls_config.max_iterations = 50;

        let result = fit_smooth_glm_full_matrix(
            &y,
            x_full.view(),
            &specs,
            &PoissonFamily,
            &LogLink,
            &config,
            Some(&offset),
            None,
            None,
            None,
        )
        .expect("test setup should be valid");

        assert!(result.converged);
    }

    #[test]
    fn test_fit_smooth_glm_with_weights() {
        let n = 100;
        let (y, x_param, x_vals) = gaussian_smooth_data(n);
        let basis = bs_basis(&x_vals, 8, 3, None, false);
        let (x_full, specs) = make_full_matrix(&x_param, &basis);
        let weights = Array1::from_vec(vec![2.0; n]);
        let config = SmoothGLMConfig::default();

        let result = fit_smooth_glm_full_matrix(
            &y,
            x_full.view(),
            &specs,
            &GaussianFamily,
            &IdentityLink,
            &config,
            None,
            Some(&weights),
            None,
            None,
        )
        .expect("test setup should be valid");

        assert!(result.converged);
    }

    #[test]
    fn test_fit_smooth_glm_dimension_mismatch() {
        let n = 100;
        let (y, x_param, _x_vals) = gaussian_smooth_data(n);
        let bad_x = Array1::from_vec((0..50).map(|i| i as f64).collect());
        let basis = bs_basis(&bad_x, 8, 3, None, false);
        // Build full matrix from mismatched basis (50 rows) and x_param (100 rows)
        // fit_smooth_glm_full_matrix validates x_full.nrows() == y.len()
        let k = basis.ncols();
        let x_full = ndarray::concatenate![
            ndarray::Axis(1),
            x_param.slice(ndarray::s![0..50, ..]),
            basis
        ]
        .as_standard_layout()
        .to_owned();
        let specs = vec![SmoothTermSpec {
            col_start: 1,
            col_end: 1 + k,
            penalty: crate::splines::penalized::penalty_matrix(k, 2),
            monotonicity: Monotonicity::None,
            initial_lambda: 1.0,
        }];
        let config = SmoothGLMConfig::default();

        let result = fit_smooth_glm_full_matrix(
            &y,
            x_full.view(),
            &specs,
            &GaussianFamily,
            &IdentityLink,
            &config,
            None,
            None,
            None,
            None,
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_fit_smooth_glm_rejects_invalid_smooth_specs() {
        let y = array![1.0, 2.0, 3.0];
        let x_full = array![[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]];
        let config = SmoothGLMConfig::default();

        let invalid_range = vec![SmoothTermSpec {
            col_start: 2,
            col_end: 3,
            penalty: Array2::eye(1),
            monotonicity: Monotonicity::None,
            initial_lambda: 1.0,
        }];
        let err = fit_smooth_glm_full_matrix(
            &y,
            x_full.view(),
            &invalid_range,
            &GaussianFamily,
            &IdentityLink,
            &config,
            None,
            None,
            None,
            None,
        )
        .expect_err("column ranges outside the design matrix must be rejected");
        assert!(err.to_string().contains("invalid column range"));

        let bad_penalty_shape = vec![SmoothTermSpec {
            col_start: 1,
            col_end: 2,
            penalty: Array2::eye(2),
            monotonicity: Monotonicity::None,
            initial_lambda: 1.0,
        }];
        let err = fit_smooth_glm_full_matrix(
            &y,
            x_full.view(),
            &bad_penalty_shape,
            &GaussianFamily,
            &IdentityLink,
            &config,
            None,
            None,
            None,
            None,
        )
        .expect_err("penalty shape must match the smooth column range");
        assert!(err.to_string().contains("penalty shape"));

        let bad_penalty_cols_only = vec![SmoothTermSpec {
            col_start: 1,
            col_end: 2,
            penalty: Array2::zeros((1, 2)),
            monotonicity: Monotonicity::None,
            initial_lambda: 1.0,
        }];
        let err = fit_smooth_glm_full_matrix(
            &y,
            x_full.view(),
            &bad_penalty_cols_only,
            &GaussianFamily,
            &IdentityLink,
            &config,
            None,
            None,
            None,
            None,
        )
        .expect_err("one-axis penalty shape mismatch must be rejected");
        assert!(err.to_string().contains("penalty shape"));
    }

    #[test]
    fn test_fit_smooth_glm_applies_nonnegative_and_nonpositive_constraints() {
        let n = 30;
        let x_vals: Array1<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        let y: Array1<f64> = x_vals.iter().map(|&x| 5.0 + 2.0 * x).collect();
        let x_full = ndarray::concatenate![
            ndarray::Axis(1),
            Array2::from_shape_fn((n, 1), |(_, _)| 1.0),
            x_vals.clone().insert_axis(ndarray::Axis(1))
        ]
        .as_standard_layout()
        .to_owned();
        let specs = vec![SmoothTermSpec {
            col_start: 1,
            col_end: 2,
            penalty: Array2::zeros((1, 1)),
            monotonicity: Monotonicity::None,
            initial_lambda: 1.0,
        }];
        let mut config = SmoothGLMConfig::default();
        config.irls_config.max_iterations = 20;
        let nonneg = vec![1usize];
        let nonpos = vec![0usize];

        let result = fit_smooth_glm_full_matrix(
            &y,
            x_full.view(),
            &specs,
            &GaussianFamily,
            &IdentityLink,
            &config,
            None,
            None,
            Some(&nonneg),
            Some(&nonpos),
        )
        .expect("sign-constrained smooth fit should remain finite");

        assert!(result.coefficients[1] >= -1e-10);
        assert!(
            result.coefficients[0] <= 1e-10,
            "nonpositive intercept constraint leaked: coefficients={:?}",
            result.coefficients
        );
        assert!(result.coefficients.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_fit_smooth_glm_sign_constraints_clip_negative_smooth_projection() {
        let n = 30;
        let x_vals: Array1<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        let y: Array1<f64> = x_vals.iter().map(|&x| 5.0 - 2.0 * x).collect();
        let x_full = ndarray::concatenate![
            ndarray::Axis(1),
            Array2::from_shape_fn((n, 1), |(_, _)| 1.0),
            x_vals.clone().insert_axis(ndarray::Axis(1))
        ]
        .as_standard_layout()
        .to_owned();
        let specs = vec![SmoothTermSpec {
            col_start: 1,
            col_end: 2,
            penalty: Array2::zeros((1, 1)),
            monotonicity: Monotonicity::None,
            initial_lambda: 1.0,
        }];
        let mut config = SmoothGLMConfig::default();
        config.irls_config.max_iterations = 3;
        let nonneg = vec![1usize];
        let nonpos = vec![0usize];

        let result = fit_smooth_glm_full_matrix(
            &y,
            x_full.view(),
            &specs,
            &GaussianFamily,
            &IdentityLink,
            &config,
            None,
            None,
            Some(&nonneg),
            Some(&nonpos),
        )
        .expect("sign-constrained decreasing fit should remain finite");

        assert!(result.coefficients[1] >= -1e-10);
        assert!(
            result.coefficients[0] <= 1e-10,
            "nonpositive intercept constraint leaked: coefficients={:?}",
            result.coefficients
        );
        assert!(result.coefficients.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_fit_smooth_glm_mixed_monotonic_and_unconstrained_terms() {
        let n = 48;
        let x1: Array1<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        let x2: Array1<f64> = (0..n)
            .map(|i| 2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64)
            .collect();
        let y: Array1<f64> = x1
            .iter()
            .zip(x2.iter())
            .map(|(&a, &b)| 1.0 + 1.2 * a + 0.25 * b.sin())
            .collect();
        let x_param = Array2::from_shape_fn((n, 1), |(_, _)| 1.0);
        let monotone_basis = is_basis(&x1, 6, 3, Some((0.0, 1.0)), true);
        let free_basis = bs_basis(&x2, 6, 3, None, false);
        let k_mono = monotone_basis.ncols();
        let k_free = free_basis.ncols();
        let x_full = ndarray::concatenate![ndarray::Axis(1), x_param, monotone_basis, free_basis]
            .as_standard_layout()
            .to_owned();
        let specs = vec![
            SmoothTermSpec {
                col_start: 1,
                col_end: 1 + k_mono,
                penalty: crate::splines::penalized::penalty_matrix(k_mono, 2),
                monotonicity: Monotonicity::Increasing,
                initial_lambda: 1.0,
            },
            SmoothTermSpec {
                col_start: 1 + k_mono,
                col_end: 1 + k_mono + k_free,
                penalty: crate::splines::penalized::penalty_matrix(k_free, 2),
                monotonicity: Monotonicity::None,
                initial_lambda: 0.5,
            },
        ];
        let mut config = SmoothGLMConfig::default();
        config.irls_config.max_iterations = 12;
        config.lambda_tol = 1e-3;

        let result = fit_smooth_glm_full_matrix(
            &y,
            x_full.view(),
            &specs,
            &GaussianFamily,
            &IdentityLink,
            &config,
            None,
            None,
            None,
            None,
        )
        .expect("mixed monotonic/unconstrained smooth fit should remain finite");

        assert!(result.iterations > 0);
        assert!(result.deviance.is_finite());
        assert!(result.coefficients.iter().all(|v| v.is_finite()));
        assert_eq!(result.lambdas.len(), 2);
        assert_eq!(result.smooth_edfs.len(), 2);
        assert!(
            smooth_coefficients_are_monotone(&result, &specs[0]),
            "only the monotonic term should be projected into ordered coefficients"
        );
    }

    #[test]
    fn test_fit_smooth_glm_monotonic_alpha_blending_reapplies_sign_constraints() {
        let n = 28;
        let x_vals: Array1<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        let y: Array1<f64> = x_vals.iter().map(|&x| 1.0 + x).collect();
        let x_param = Array2::from_shape_fn((n, 1), |(_, _)| 1.0);
        let basis = is_basis(&x_vals, 6, 3, Some((0.0, 1.0)), true);
        let (x_full, specs) =
            make_full_matrix_with_monotonicity(&x_param, &basis, Monotonicity::Increasing);
        let nonneg: Vec<usize> = (specs[0].col_start..specs[0].col_end).collect();
        let nonpos = vec![0usize];
        let mut config = SmoothGLMConfig::default();
        config.irls_config.max_iterations = 3;
        config.lambda_tol = 1e-3;

        let result = fit_smooth_glm_full_matrix(
            &y,
            x_full.view(),
            &specs,
            &GaussianFamily,
            &IdentityLink,
            &config,
            None,
            None,
            Some(&nonneg),
            Some(&nonpos),
        )
        .expect("monotonic sign constraints should retain a finite blended iterate");

        assert!(result.deviance.is_finite());
        assert!(
            result.coefficients[0] <= 1e-10,
            "nonpositive intercept constraint leaked in monotonic path: coefficients={:?}",
            result.coefficients
        );
        for idx in nonneg {
            assert!(result.coefficients[idx] >= -1e-10);
        }
        assert!(smooth_coefficients_are_monotone(&result, &specs[0]));
    }

    // =========================================================================
    // Multi-term tests
    // =========================================================================

    #[test]
    fn test_fit_smooth_glm_two_terms() {
        let n = 200;
        let x1: Array1<f64> = (0..n).map(|i| i as f64 * 10.0 / n as f64).collect();
        let x2: Array1<f64> = (0..n).map(|i| i as f64 * 5.0 / n as f64).collect();
        let y: Array1<f64> = x1
            .iter()
            .zip(x2.iter())
            .map(|(&a, &b)| 2.0 + a.sin() + 0.5 * b.cos())
            .collect();

        let x_param = Array2::from_shape_fn((n, 1), |(_, _)| 1.0);
        let basis1 = bs_basis(&x1, 8, 3, None, false);
        let basis2 = bs_basis(&x2, 6, 3, None, false);
        let k1 = basis1.ncols();
        let k2 = basis2.ncols();
        let x_full = ndarray::concatenate![ndarray::Axis(1), x_param, basis1, basis2]
            .as_standard_layout()
            .to_owned();
        let specs = vec![
            SmoothTermSpec {
                col_start: 1,
                col_end: 1 + k1,
                penalty: crate::splines::penalized::penalty_matrix(k1, 2),
                monotonicity: Monotonicity::None,
                initial_lambda: 1.0,
            },
            SmoothTermSpec {
                col_start: 1 + k1,
                col_end: 1 + k1 + k2,
                penalty: crate::splines::penalized::penalty_matrix(k2, 2),
                monotonicity: Monotonicity::None,
                initial_lambda: 1.0,
            },
        ];
        let config = SmoothGLMConfig::default();

        let result = fit_smooth_glm_full_matrix(
            &y,
            x_full.view(),
            &specs,
            &GaussianFamily,
            &IdentityLink,
            &config,
            None,
            None,
            None,
            None,
        )
        .expect("test setup should be valid");

        assert!(result.converged);
        assert_eq!(result.lambdas.len(), 2);
        assert_eq!(result.smooth_edfs.len(), 2);
        assert!(result.lambdas.iter().all(|&l| l > 0.0));
        assert!(result.smooth_edfs.iter().all(|&e| e > 1.0));
    }

    // =========================================================================
    // Result fields are populated correctly
    // =========================================================================

    #[test]
    fn test_smooth_result_fields_populated() {
        let n = 100;
        let (y, x_param, x_vals) = gaussian_smooth_data(n);
        let basis = bs_basis(&x_vals, 8, 3, None, false);
        let (x_full, specs) = make_full_matrix(&x_param, &basis);
        let config = SmoothGLMConfig::default();

        let result = fit_smooth_glm_full_matrix(
            &y,
            x_full.view(),
            &specs,
            &GaussianFamily,
            &IdentityLink,
            &config,
            None,
            None,
            None,
            None,
        )
        .expect("test setup should be valid");

        assert_eq!(result.fitted_values.len(), n);
        assert_eq!(result.linear_predictor.len(), n);
        assert_eq!(result.irls_weights.len(), n);
        assert_eq!(result.prior_weights.len(), n);
        assert_eq!(result.design_matrix.nrows(), n);
        assert_eq!(result.y.len(), n);
        assert!(result.family_name.contains("Gaussian") || result.family_name.contains("gaussian"));
    }

    // =========================================================================
    // Empty specs fallback (standard GLM through unified entry point)
    // =========================================================================

    #[test]
    fn test_fit_smooth_glm_no_smooth_terms() {
        let (y, x_param, _x_vals) = gaussian_smooth_data(100);
        let config = SmoothGLMConfig::default();

        let result = fit_smooth_glm_full_matrix(
            &y,
            x_param.view(),
            &[],
            &GaussianFamily,
            &IdentityLink,
            &config,
            None,
            None,
            None,
            None,
        )
        .expect("test setup should be valid");

        assert!(result.converged);
        assert!(result.lambdas.is_empty());
        assert!(result.smooth_edfs.is_empty());
        assert_eq!(result.coefficients.len(), 1); // intercept only
        assert!(result.offset.is_none());
    }

    #[test]
    fn test_fit_smooth_glm_no_smooth_terms_preserves_nonzero_offset() {
        let n = 24;
        let offset = Array1::from_vec((0..n).map(|i| 0.05 * i as f64).collect());
        let y = offset.mapv(|o| 2.0 + o);
        let x_param = Array2::from_shape_fn((n, 1), |(_, _)| 1.0);
        let config = SmoothGLMConfig::default();

        let result = fit_smooth_glm_full_matrix(
            &y,
            x_param.view(),
            &[],
            &GaussianFamily,
            &IdentityLink,
            &config,
            Some(&offset),
            None,
            None,
            None,
        )
        .expect("standard GLM fallback should preserve a supplied offset");

        assert!(result.converged);
        assert!(result.lambdas.is_empty());
        assert!(result.smooth_edfs.is_empty());
        assert_array1_close(
            result
                .offset
                .as_ref()
                .expect("nonzero offset should be retained"),
            offset.as_slice().expect("contiguous offset"),
            1e-12,
        );
    }

    #[test]
    fn test_fit_smooth_glm_no_smooth_terms_propagates_glm_validation_errors() {
        let y = array![1.0, 2.0, 3.0];
        let x_param = Array2::from_shape_fn((3, 1), |(_, _)| 1.0);
        let short_weights = array![1.0, 1.0];
        let config = SmoothGLMConfig::default();

        let err = fit_smooth_glm_full_matrix(
            &y,
            x_param.view(),
            &[],
            &GaussianFamily,
            &IdentityLink,
            &config,
            None,
            Some(&short_weights),
            None,
            None,
        )
        .expect_err("empty smooth-spec path must propagate standard GLM validation failures");

        assert!(
            err.to_string().contains("weights"),
            "error should name the delegated GLM validation failure: {err}"
        );
    }

    #[test]
    fn test_fit_smooth_glm_zero_iterations_reports_max_iterations() {
        let (y, x_param, x_vals) = gaussian_smooth_data(30);
        let basis = bs_basis(&x_vals, 6, 3, None, false);
        let (x_full, specs) = make_full_matrix(&x_param, &basis);
        let mut config = SmoothGLMConfig::default();
        config.irls_config.max_iterations = 0;

        let result = fit_smooth_glm_full_matrix(
            &y,
            x_full.view(),
            &specs,
            &GaussianFamily,
            &IdentityLink,
            &config,
            None,
            None,
            None,
            None,
        )
        .expect("zero-iteration smooth fit should return the initialized state");

        assert_eq!(result.iterations, 0);
        assert!(!result.converged);
        assert_eq!(result.solver_status, "max_iterations");
        assert!(result
            .warnings
            .iter()
            .any(|w| w.contains("did not converge")));
    }

    #[test]
    fn test_fit_smooth_glm_zero_deviance_uses_absolute_convergence_change() {
        let n = 12;
        let y = Array1::zeros(n);
        let x_vals: Array1<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        let x_full = ndarray::concatenate![
            ndarray::Axis(1),
            Array2::from_shape_fn((n, 1), |(_, _)| 1.0),
            x_vals.insert_axis(ndarray::Axis(1))
        ]
        .as_standard_layout()
        .to_owned();
        let specs = vec![SmoothTermSpec {
            col_start: 1,
            col_end: 2,
            penalty: Array2::eye(1),
            monotonicity: Monotonicity::None,
            initial_lambda: 1.0,
        }];
        let mut config = SmoothGLMConfig::default();
        config.irls_config.max_iterations = 1;

        let result = fit_smooth_glm_full_matrix(
            &y,
            x_full.view(),
            &specs,
            &GaussianFamily,
            &IdentityLink,
            &config,
            None,
            None,
            None,
            None,
        )
        .expect("zero-deviance Gaussian fit should be well-defined");

        assert!(result.converged);
        assert_eq!(result.solver_status, "converged");
        assert_abs_diff_eq!(result.deviance, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_fit_smooth_glm_failed_initial_projection_warns_and_returns_initialized_state() {
        let y = array![1.0, 2.0, 3.0, 4.0];
        let x_full = Array2::zeros((4, 2));
        let specs = vec![SmoothTermSpec {
            col_start: 1,
            col_end: 2,
            penalty: Array2::eye(1),
            monotonicity: Monotonicity::None,
            initial_lambda: 1.0,
        }];
        let mut config = SmoothGLMConfig::default();
        config.irls_config.max_iterations = 0;

        let result = fit_smooth_glm_full_matrix(
            &y,
            x_full.view(),
            &specs,
            &GaussianFamily,
            &IdentityLink,
            &config,
            None,
            None,
            None,
            None,
        )
        .expect("projection failure should degrade to the zero-coefficient initialization");

        assert_eq!(result.iterations, 0);
        assert_eq!(result.solver_status, "max_iterations");
        assert!(result.coefficients.iter().all(|&c| c == 0.0));
        assert!(result
            .warnings
            .iter()
            .any(|w| w.contains("Initial smooth coefficient projection failed")));
    }

    #[test]
    fn test_fit_smooth_glm_zero_initial_lambda_enters_absolute_lambda_change_path() {
        let (y, x_param, x_vals) = gaussian_smooth_data(40);
        let basis = bs_basis(&x_vals, 6, 3, None, false);
        let (x_full, mut specs) = make_full_matrix(&x_param, &basis);
        specs[0].initial_lambda = 0.0;
        let mut config = SmoothGLMConfig::default();
        config.irls_config.max_iterations = 3;
        config.lambda_tol = 1e-3;

        let result = fit_smooth_glm_full_matrix(
            &y,
            x_full.view(),
            &specs,
            &GaussianFamily,
            &IdentityLink,
            &config,
            None,
            None,
            None,
            None,
        )
        .expect("zero initial lambda should be optimized onto a finite path");

        assert_eq!(result.lambdas.len(), 1);
        assert!(result.lambdas[0].is_finite());
        assert!(result.lambdas[0] >= 0.0);
        assert!(result.coefficients.iter().all(|c| c.is_finite()));
        assert!(result.deviance.is_finite());
    }

    // =========================================================================
    // B1 / B2: step-halving robustness in the smooth solver
    // =========================================================================

    /// Build a Poisson/log smooth fixture with a single large y spike that makes
    /// the full Newton step overshoot, forcing the step-halving loop to engage
    /// (and exercising the fraction-0 fallback added in B2).
    fn poisson_overshoot_fixture() -> (Array1<f64>, Array2<f64>, Vec<SmoothTermSpec>) {
        let n = 30usize;
        let x_vals: Array1<f64> = (0..n).map(|i| i as f64).collect();
        let mut yv = vec![1.0; n];
        yv[n - 1] = 1.0e3; // large spike → first full step overshoots
        let y = Array1::from_vec(yv);
        let x_param = Array2::from_shape_fn((n, 1), |(_, _)| 1.0);
        let basis = bs_basis(&x_vals, 8, 3, None, false);
        let (x_full, specs) = make_full_matrix(&x_param, &basis);
        (y, x_full, specs)
    }

    #[test]
    fn test_smooth_step_halving_engages_and_weights_are_real() {
        // B1 + B2: a fit that genuinely exercises the step-halving loop must (a)
        // succeed with a finite result (the fraction-0 fallback guarantees a
        // finite non-worsening step always exists) and (b) carry the retained
        // iterate's real IRLS weights into the result — never the all-ones init
        // (which would silently corrupt covariance/EDF/GCV).
        let (y, x_full, specs) = poisson_overshoot_fixture();
        let config = SmoothGLMConfig::default();

        let result = fit_smooth_glm_full_matrix(
            &y,
            x_full.view(),
            &specs,
            &PoissonFamily,
            &LogLink,
            &config,
            None,
            None,
            None,
            None,
        )
        .expect("overshoot fixture must still produce a finite fit (B2 fallback)");

        // Step-halving must have engaged for this fixture (regression guard: if
        // it stops engaging, the fixture no longer tests the halving path).
        assert!(
            result.step_halving_used,
            "fixture should exercise step-halving (status={}, iters={})",
            result.solver_status, result.iterations
        );
        assert!(result.deviance.is_finite(), "deviance must be finite (B2)");
        assert!(
            result.coefficients.iter().all(|c| c.is_finite()),
            "coefficients must be finite (B2)"
        );
        // B1: weights reflect the real iterate, not the all-ones init.
        let all_unit = result.irls_weights.iter().all(|&w| (w - 1.0).abs() < 1e-12);
        assert!(
            !all_unit,
            "irls_weights must reflect the retained iterate, not the all-ones init"
        );
        assert!(
            result
                .irls_weights
                .iter()
                .all(|w| w.is_finite() && *w > 0.0),
            "retained irls_weights must be finite and positive"
        );
    }

    #[test]
    fn test_smooth_step_halving_reapplies_nonnegative_constraints_to_blended_steps() {
        let (y, x_full, specs) = poisson_overshoot_fixture();
        let config = SmoothGLMConfig::default();
        let all_indices: Vec<usize> = (0..x_full.ncols()).collect();

        let result = fit_smooth_glm_full_matrix(
            &y,
            x_full.view(),
            &specs,
            &PoissonFamily,
            &LogLink,
            &config,
            None,
            None,
            Some(&all_indices),
            None,
        )
        .expect("sign-constrained step-halving fixture should retain a finite iterate");

        assert!(result.step_halving_used);
        assert!(result.coefficients.iter().all(|c| *c >= -1e-10));
        assert!(result
            .fitted_values
            .iter()
            .all(|mu| mu.is_finite() && *mu > 0.0));
    }

    #[test]
    fn test_smooth_normal_fit_weights_populated() {
        // B1 (positive path): a normal converging Poisson smooth fit must carry
        // genuine (non-unit) IRLS weights through to the result.
        let (y, x_param, x_vals) = poisson_smooth_data(120);
        let basis = bs_basis(&x_vals, 10, 3, None, false);
        let (x_full, specs) = make_full_matrix(&x_param, &basis);
        let config = SmoothGLMConfig::default();

        let result = fit_smooth_glm_full_matrix(
            &y,
            x_full.view(),
            &specs,
            &PoissonFamily,
            &LogLink,
            &config,
            None,
            None,
            None,
            None,
        )
        .expect("normal Poisson smooth fit should succeed");

        assert!(
            result
                .irls_weights
                .iter()
                .all(|w| w.is_finite() && *w > 0.0),
            "weights must be finite and positive"
        );
        let all_unit = result.irls_weights.iter().all(|&w| (w - 1.0).abs() < 1e-12);
        assert!(
            !all_unit,
            "Poisson IRLS weights should not be all-ones for a real fit"
        );
    }
}
