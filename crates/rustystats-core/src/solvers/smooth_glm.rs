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
    SMOOTH_CONVERGENCE_TOL, SMOOTH_INNER_MAX_PER_CYCLE, SMOOTH_KKT_SCORE_TOL,
    SMOOTH_WARM_START_MAX_ITER, ZERO_TOL,
};
use crate::convert;
use crate::error::{Result, RustyStatsError};
use crate::families::Family;
use crate::inference::HCType;
use crate::links::Link;
use crate::regularization::{Penalty, SmoothPenalty};
use crate::rows::GramPartial;
use crate::solvers::irls::{
    coef_predictor, finish_statistics, fit_glm_unified, packed_to_array2, packed_to_dmatrix,
    request_statistics, solve_wls_from_precomputed, FinalStatistics, FirstPassFacts, FitConfig,
    IRLSConfig, RetainedWeights, RobustRequest, StatisticsEvals,
};
use crate::solvers::passes::{
    record_step, Eval, EvalSum, InMemoryExecutor, MonotoneTermTransform, MonotoneTransform,
    PassExecutor, PassRequest, Predictor, ResponseSpec, RowContext, SolverStep, WeightsSpec,
};
use crate::splines::penalized::{compute_edf, gcv_score, penalty_matrix};
use nalgebra::DVector;

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
    /// Statistics of the fit computed in its final passes (penalised smooth fits
    /// without monotonicity); None for the routes that do not run on passes yet.
    pub statistics: Option<FinalStatistics>,

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

    /// The drivers' semantic steps (`specs/out-of-core-fit.md` §4.5).
    pub solver_trace: Vec<SolverStep>,
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

    /// Robust covariance types for the final pass (`FinalStatistics::robust`).
    pub robust: Vec<HCType>,
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
            robust: Vec::new(),
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
            statistics: irls.statistics,
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
            solver_trace: irls.solver_trace,
        });
    }

    validate_smooth_specs(smooth_specs, p)?;
    fit_smooth_in_memory(
        y,
        x_full,
        smooth_specs,
        family,
        link,
        config,
        offset,
        weights,
        nonneg_indices,
        nonpos_indices,
    )
}

// =============================================================================
// Smooth driver over passes (`specs/out-of-core-fit.md` §7.6)
// =============================================================================

/// A penalised smooth fit served by any [`PassExecutor`]: everything but
/// per-row arrays.
#[derive(Debug, Clone)]
pub struct SmoothExecutorFit {
    pub coefficients: Array1<f64>,
    pub deviance: f64,
    pub iterations: usize,
    pub converged: bool,
    pub lambdas: Vec<f64>,
    pub smooth_edfs: Vec<f64>,
    pub total_edf: f64,
    pub gcv: f64,
    pub covariance_unscaled: Array2<f64>,
    pub penalty: Penalty,
    pub warnings: Vec<String>,
    pub step_halving_used: bool,
    pub solver_status: String,
    pub stationary: bool,
    pub max_std_score: f64,
    pub statistics: FinalStatistics,
}

/// Fit a GLM with penalised smooth terms, monotone or not, whose rows `executor`
/// serves (`specs/out-of-core-fit.md` §7.6, §7.7): the driver an in-memory fit runs.
#[allow(clippy::too_many_arguments)]
pub fn fit_smooth_glm_with_executor(
    executor: &mut dyn PassExecutor,
    smooth_specs: &[SmoothTermSpec],
    family: &dyn Family,
    link: &dyn Link,
    config: &SmoothGLMConfig,
    nonneg_indices: Option<&[usize]>,
    nonpos_indices: Option<&[usize]>,
    has_offset: bool,
) -> Result<SmoothExecutorFit> {
    if smooth_specs.is_empty() {
        return Err(RustyStatsError::InvalidValue(
            "A smooth fit needs at least one smooth term.".to_string(),
        ));
    }
    let outcome = drive_smooth(
        executor,
        &SmoothDriveSettings {
            specs: smooth_specs,
            config,
            family,
            nonneg_indices,
            nonpos_indices,
            null_offset: crate::diagnostics::NullOffset::for_fit(has_offset, link, family),
            robust: &config.robust,
        },
    )?;
    Ok(SmoothExecutorFit {
        coefficients: outcome.coefficients,
        deviance: outcome.deviance,
        iterations: outcome.iterations,
        converged: outcome.converged,
        lambdas: outcome.lambdas,
        smooth_edfs: outcome.smooth_edfs,
        total_edf: outcome.total_edf,
        gcv: outcome.gcv,
        covariance_unscaled: outcome.covariance_unscaled,
        penalty: outcome.penalty,
        warnings: outcome.warnings,
        step_halving_used: outcome.step_halving_used,
        solver_status: outcome.solver_status.to_string(),
        stationary: outcome.stationary,
        max_std_score: outcome.max_std_score,
        statistics: outcome.statistics,
    })
}

/// A penalised smooth fit on an in-memory design: the pass driver
/// [`drive_smooth`] over an [`InMemoryExecutor`].
#[allow(clippy::too_many_arguments)]
fn fit_smooth_in_memory(
    y: &Array1<f64>,
    x: ArrayView2<'_, f64>,
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
    let offset_vec = offset.cloned().unwrap_or_else(|| Array1::zeros(n));
    let prior_weights = weights.cloned().unwrap_or_else(|| Array1::ones(n));
    if offset_vec.len() != n || prior_weights.len() != n {
        return Err(RustyStatsError::dim_mismatch(
            n,
            if offset_vec.len() != n {
                offset_vec.len()
            } else {
                prior_weights.len()
            },
            "smooth offset/weights length vs y length",
        ));
    }
    let y_slice = y.as_slice().ok_or_else(|| {
        RustyStatsError::LinearAlgebraError("Response y must be contiguous".to_string())
    })?;
    let sparse_rows = config.irls_config.row_kernel.sparse_in_memory(x);
    let ctx = RowContext {
        family,
        link,
        y: y_slice,
        offset: offset_vec
            .as_slice()
            .expect("an owned offset is contiguous"),
        prior_weights: prior_weights
            .as_slice()
            .expect("owned prior weights are contiguous"),
        y_mean: crate::families::response_mean(y),
        min_weight: config.irls_config.min_weight,
        has_offset: offset.is_some(),
        standardization: None,
    };
    let mut executor = InMemoryExecutor::new(x, sparse_rows, ctx)?;
    let outcome = drive_smooth(
        &mut executor,
        &SmoothDriveSettings {
            specs: smooth_specs,
            config,
            family,
            nonneg_indices,
            nonpos_indices,
            null_offset: crate::diagnostics::NullOffset::for_fit(offset.is_some(), link, family),
            robust: &config.robust,
        },
    )?;
    let (eta, mu) = executor.materialize(&outcome.fitted);
    let irls_weights = outcome.weights.materialize(&executor, n);
    let solver_trace = executor.take_trace();
    Ok(SmoothGLMResult {
        coefficients: outcome.coefficients,
        fitted_values: Array1::from_vec(mu),
        linear_predictor: Array1::from_vec(eta),
        deviance: outcome.deviance,
        iterations: outcome.iterations,
        converged: outcome.converged,
        lambdas: outcome.lambdas,
        smooth_edfs: outcome.smooth_edfs,
        total_edf: outcome.total_edf,
        gcv: outcome.gcv,
        covariance_unscaled: outcome.covariance_unscaled,
        family_name: family.name().to_string(),
        penalty: outcome.penalty,
        irls_weights: Array1::from_vec(irls_weights),
        prior_weights,
        y: y.clone(),
        offset: offset.cloned(),
        warnings: outcome.warnings,
        step_halving_used: outcome.step_halving_used,
        solver_status: outcome.solver_status.to_string(),
        stationary: outcome.stationary,
        max_std_score: outcome.max_std_score,
        statistics: Some(outcome.statistics),
        solver_trace,
    })
}

/// Bisection levels one speculative pass evaluates when the executor prefers few
/// passes: a tree of 2^8 − 1 = 255 midpoints per row.
const SPECULATIVE_BISECTION_LEVELS: usize = 8;

/// The bracket spans the intercept refresh tries: 1, 2, …, 64.
const INTERCEPT_BRACKET_SPANS: [f64; 7] = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0];

/// Check the smooth specs against a design of `p` columns.
fn validate_smooth_specs(smooth_specs: &[SmoothTermSpec], p: usize) -> Result<()> {
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
    Ok(())
}

/// What [`drive_smooth`] needs besides the rows.
pub(crate) struct SmoothDriveSettings<'a> {
    pub specs: &'a [SmoothTermSpec],
    pub config: &'a SmoothGLMConfig,
    pub family: &'a dyn Family,
    pub nonneg_indices: Option<&'a [usize]>,
    pub nonpos_indices: Option<&'a [usize]>,
    /// Whether the null model scales by exp(offset) (`RowContext::null_log_offset`).
    pub null_offset: crate::diagnostics::NullOffset,
    pub robust: &'a [HCType],
}

/// The result of driving a penalised smooth fit: everything but per-row arrays.
pub(crate) struct SmoothDriveOutcome {
    pub coefficients: Array1<f64>,
    pub deviance: f64,
    pub iterations: usize,
    pub converged: bool,
    pub lambdas: Vec<f64>,
    pub smooth_edfs: Vec<f64>,
    pub total_edf: f64,
    pub gcv: f64,
    pub covariance_unscaled: Array2<f64>,
    pub penalty: Penalty,
    pub warnings: Vec<String>,
    pub step_halving_used: bool,
    pub solver_status: &'static str,
    pub stationary: bool,
    pub max_std_score: f64,
    /// The predictor the fitted values come from (`Shifted` after a refresh).
    pub fitted: Predictor,
    /// The weights of the covariance, stored as the result's `irls_weights`.
    pub weights: RetainedWeights,
    pub statistics: FinalStatistics,
}

/// The ordinary smooth Gram at a predictor: X'WX, X'Wz and z'Wz with the smooth
/// route's working weights and response (`update_irls_work_arrays`).
fn smooth_gram(predictor: usize) -> Eval {
    Eval::Gram {
        weights: WeightsSpec::SmoothIrls(predictor),
        response: ResponseSpec::Working(predictor),
    }
}

fn take_gram_sum(sums: &mut [EvalSum], eval: usize) -> GramPartial {
    std::mem::replace(&mut sums[eval], EvalSum::Count(0.0)).into_gram()
}

/// The sums of one exit pass (A or C) at a predictor.
struct ExitPass {
    deviance: f64,
    kkt: Vec<f64>,
    shift_scores: Vec<f64>,
    sums: Vec<EvalSum>,
    stats: StatisticsEvals,
}

/// One exit pass: the deviance, the KKT sums of the unpenalized columns, the
/// intercept shift scores asked for, and the final statistics (robust included).
#[allow(clippy::too_many_arguments)]
fn exit_pass(
    executor: &mut dyn PassExecutor,
    predictor: &Predictor,
    kkt_columns: &std::sync::Arc<[usize]>,
    shifts: &[f64],
    facts: &FirstPassFacts,
    robust: RobustRequest<'_>,
) -> Result<ExitPass> {
    let (n, p) = (executor.n_rows(), executor.n_params());
    let mut request = PassRequest::default();
    let index = request.predictor(predictor.clone());
    let deviance = request.eval(Eval::Deviance(index));
    let kkt = request.eval(Eval::Kkt {
        predictor: index,
        columns: kkt_columns.clone(),
    });
    let scores = (!shifts.is_empty()).then(|| {
        request.eval(Eval::ShiftScores {
            predictor: index,
            shifts: shifts.into(),
        })
    });
    let stats = request_statistics(&mut request, index, facts, Some(robust), (n, p));
    let sums = executor.run(&request)?;
    Ok(ExitPass {
        deviance: sums[deviance].deviance().0,
        kkt: sums[kkt].sums().to_vec(),
        shift_scores: scores
            .map(|eval| sums[eval].sums().to_vec())
            .unwrap_or_default(),
        sums,
        stats,
    })
}

/// Intercept-direction scores at `shifts` from `base` in one pass.
fn shift_scores(
    executor: &mut dyn PassExecutor,
    base: &Predictor,
    shifts: &[f64],
) -> Result<Vec<f64>> {
    let mut request = PassRequest::default();
    let index = request.predictor(base.clone());
    let eval = request.eval(Eval::ShiftScores {
        predictor: index,
        shifts: shifts.into(),
    });
    Ok(executor.run(&request)?[eval].sums().to_vec())
}

/// The midpoints of `depth` bisection levels from (lo, hi), in level order: node i
/// bisects its interval at 0.5·(lo + hi); its first child keeps the lower half
/// (hi = mid), its second the upper half (lo = mid).
fn bisection_tree(lo: f64, hi: f64, depth: usize) -> Vec<f64> {
    let nodes = (1usize << depth) - 1;
    let mut intervals = Vec::with_capacity(nodes);
    intervals.push((lo, hi));
    let mut mids = Vec::with_capacity(nodes);
    for node in 0..nodes {
        let (lo, hi) = intervals[node];
        let mid = 0.5 * (lo + hi);
        mids.push(mid);
        if 2 * node + 2 < nodes {
            intervals.push((lo, mid));
            intervals.push((mid, hi));
        }
    }
    mids
}

/// `solve_intercept_shift` over passes: the same bracketing and bisection, its
/// scores evaluated a pair or a speculative tree of levels per pass. `scores` are
/// pass A's: the score at 0 and, when the executor prefers few passes, at every
/// bracket end (−1, 1, −2, 2, …, −64, 64). Also returns the scores the sequential
/// search read (at 0, each bracket pair, each midpoint), whatever was speculated.
fn solve_intercept_shift_over_passes(
    executor: &mut dyn PassExecutor,
    base: &Predictor,
    scores: &[f64],
) -> Result<(Option<f64>, usize)> {
    let few = executor.prefers_few_passes();
    let score_zero = scores[0];
    let mut evaluations = 1usize;
    if !score_zero.is_finite() {
        return Ok((None, evaluations));
    }
    if score_zero == 0.0 {
        return Ok((Some(0.0), evaluations));
    }

    let bracket = |k: usize, executor: &mut dyn PassExecutor| -> Result<(f64, f64)> {
        if few {
            Ok((scores[1 + 2 * k], scores[2 + 2 * k]))
        } else {
            let span = INTERCEPT_BRACKET_SPANS[k];
            let pair = shift_scores(executor, base, &[-span, span])?;
            Ok((pair[0], pair[1]))
        }
    };
    let mut k = 0;
    let mut span = INTERCEPT_BRACKET_SPANS[k];
    let (mut lo, mut hi) = (-span, span);
    let (mut score_lo, mut score_hi) = bracket(k, executor)?;
    evaluations += 2;
    while score_lo.is_finite()
        && score_hi.is_finite()
        && score_lo.signum() == score_hi.signum()
        && span < 64.0
    {
        k += 1;
        span = INTERCEPT_BRACKET_SPANS[k];
        lo = -span;
        hi = span;
        (score_lo, score_hi) = bracket(k, executor)?;
        evaluations += 2;
    }
    if !score_lo.is_finite() || !score_hi.is_finite() || score_lo.signum() == score_hi.signum() {
        return Ok((None, evaluations));
    }

    // Moving lo keeps its score's sign, so every comparison is against this sign.
    let sign_lo = score_lo.signum();
    let levels = if few { SPECULATIVE_BISECTION_LEVELS } else { 1 };
    let mut steps = 0usize;
    while steps < 200 {
        let depth = levels.min(200 - steps);
        let mids = bisection_tree(lo, hi, depth);
        let mid_scores = shift_scores(executor, base, &mids)?;
        let mut node = 0usize;
        for _ in 0..depth {
            let mid = mids[node];
            let score_mid = mid_scores[node];
            steps += 1;
            evaluations += 1;
            if !score_mid.is_finite() {
                return Ok((None, evaluations));
            }
            if score_mid == 0.0 {
                return Ok((Some(mid), evaluations));
            }
            if score_mid.signum() == sign_lo {
                lo = mid;
                node = 2 * node + 2;
            } else {
                hi = mid;
                node = 2 * node + 1;
            }
            if (hi - lo).abs() < 1e-13 * (1.0 + hi.abs().max(lo.abs())) {
                return Ok((Some(0.5 * (lo + hi)), evaluations));
            }
        }
    }
    Ok((Some(0.5 * (lo + hi)), evaluations))
}

/// The iterate a smooth driver carries between its phases.
struct SmoothLoop {
    coefficients: Array1<f64>,
    /// The predictor the coefficients' fitted state comes from.
    state: Predictor,
    deviance: f64,
    lambdas: Vec<f64>,
    iteration: usize,
    converged: bool,
    final_weights: RetainedWeights,
    step_halving_used: bool,
    step_halving_failed: bool,
    warnings: Vec<String>,
}

/// The setting-derived constants of a smooth fit.
struct SmoothShape {
    n: usize,
    p: usize,
    p_param: usize,
    term_indices: Vec<(usize, usize)>,
    smooth_tolerance: f64,
    log_lambda_min: f64,
    log_lambda_max: f64,
}

/// Penalised IRLS for smooth terms over passes (§7.6, §7.7): the arithmetic of
/// `fit_smooth_glm_full_matrix`, for fits with and without monotone terms.
///
/// - Initialisation: one pass for the family initializer's projection (with the
///   y/weight/offset facts and the intercept candidates' exact-one counts), one
///   for the projected state's deviance and first Gram.
/// - Without monotone terms, each iteration runs GCV λ updates and the solve on
///   the Gram held from the last pass, then one candidate pass (deviance, and the
///   next Gram speculatively); a rejected candidate costs a trials pass and a
///   Gram pass.
/// - With monotone terms: REML λ on the start Gram, the warm start (one fused
///   pass per iteration), the α reset, then nested PIRLS whose Gram is formed on
///   the monotone-transformed rows per block, and the refinement of the
///   unconstrained block (`Partitioned` state, fixed columns in ascending order).
/// - Exit: the covariance and EDFs from the covariance weights' Gram on X (the
///   last iteration's when it was on X, else pass R); pass A at the final state
///   for the KKT sums, the intercept shift scores and the statistics; the
///   intercept refresh's bisection passes; pass C at the refreshed state when a
///   refresh is applied.
///
/// D1 is fixed: a penalised Gram that does not invert is an error.
pub(crate) fn drive_smooth(
    executor: &mut dyn PassExecutor,
    settings: &SmoothDriveSettings<'_>,
) -> Result<SmoothDriveOutcome> {
    let n = executor.n_rows();
    let p = executor.n_params();
    let specs = settings.specs;
    let config = settings.config;
    let family = settings.family;
    let (nonneg_indices, nonpos_indices) = (settings.nonneg_indices, settings.nonpos_indices);
    validate_smooth_specs(specs, p)?;

    let mut is_smooth = vec![false; p];
    for spec in specs {
        for flag in &mut is_smooth[spec.col_start..spec.col_end] {
            *flag = true;
        }
    }
    let shape = SmoothShape {
        n,
        p,
        p_param: parametric_column_count(p, is_smooth.iter().filter(|&&flag| flag).count()),
        term_indices: specs.iter().map(|s| (s.col_start, s.col_end)).collect(),
        // Use a looser tolerance for smooth models — GCV lambda perturbation
        // introduces noise larger than 1e-8, so tighter tolerance just wastes iterations
        smooth_tolerance: config.irls_config.tolerance.max(SMOOTH_CONVERGENCE_TOL),
        log_lambda_min: config.lambda_min.ln(),
        log_lambda_max: config.lambda_max.ln(),
    };
    let constrained: std::collections::BTreeSet<usize> = nonneg_indices
        .unwrap_or(&[])
        .iter()
        .chain(nonpos_indices.unwrap_or(&[]).iter())
        .copied()
        .collect();
    let intercept_candidates: std::sync::Arc<[usize]> = (0..p)
        .filter(|&c| !is_smooth[c] && !constrained.contains(&c))
        .collect();
    let kkt_columns: std::sync::Arc<[usize]> = (0..p).filter(|&c| !is_smooth[c]).collect();
    let max_iterations = config.irls_config.max_iterations;
    let has_monotonic = specs.iter().any(|s| s.is_monotonic());
    let driver = if has_monotonic { "monotone" } else { "smooth" };

    // Pass 1: the family initializer (η = g(μ₀)) projected into coefficient space
    // with prior weights; the facts; which candidate intercept columns are all ones.
    let start = Predictor::Initializer { safe: false };
    let (facts, non_one, start_deviance, projection) = {
        let mut request = PassRequest::default();
        let index = request.predictor(start.clone());
        let deviance = request.eval(Eval::Deviance(index));
        let gram = request.eval(Eval::Gram {
            weights: WeightsSpec::Prior,
            response: ResponseSpec::LinkMinusOffset(index),
        });
        let facts = request.eval(Eval::Facts);
        let non_one = request.eval(Eval::NonOneCounts(intercept_candidates.clone()));
        let mut sums = executor.run(&request)?;
        (
            FirstPassFacts::from_sums(sums[facts].facts(), n, settings.null_offset),
            sums[non_one].sums().to_vec(),
            sums[deviance].deviance().0,
            take_gram_sum(&mut sums, gram),
        )
    };
    let mut ls = SmoothLoop {
        coefficients: Array1::zeros(p),
        state: start.clone(),
        deviance: start_deviance,
        // Clamp initial lambdas into the configured bounds. Callers pin a fixed
        // lambda by passing lambda_min == lambda_max.
        lambdas: specs
            .iter()
            .map(|s| s.initial_lambda.clamp(config.lambda_min, config.lambda_max))
            .collect(),
        iteration: 0,
        converged: false,
        final_weights: RetainedWeights::Ones,
        step_halving_used: false,
        step_halving_failed: false,
        warnings: Vec::new(),
    };
    {
        let xtx = packed_to_dmatrix(&projection.xtx_packed, p);
        let xtz = DVector::from_column_slice(&projection.xtz);
        let projected = solve_wls_from_precomputed(&xtx, &xtz, &Array2::zeros((p, p)), true);
        match projected
            .ok()
            .and_then(|(beta, _)| initial_projection_if_finite(beta))
        {
            Some(mut projected) => {
                project_coefficients_to_bounds(&mut projected, nonneg_indices, nonpos_indices);
                ls.coefficients = projected;
                ls.state = coef_predictor(&ls.coefficients);
            }
            None => ls.warnings.push(
                "Initial smooth coefficient projection failed. Starting from zero coefficients."
                    .to_string(),
            ),
        }
    }

    // Pass 2: the start's deviance, and its Gram (the first iteration's, or the
    // monotone path's REML and warm start); with no iteration at all, the
    // all-ones Gram the covariance then uses.
    let mut retained_gram: Option<GramPartial> = None;
    let mut start_gram: Option<GramPartial> = None;
    {
        let mut request = PassRequest::default();
        let index = request.predictor(ls.state.clone());
        let deviance_eval = (ls.state != start).then(|| request.eval(Eval::Deviance(index)));
        let gram_eval = request.eval(if max_iterations > 0 {
            smooth_gram(index)
        } else {
            Eval::Gram {
                weights: WeightsSpec::Ones,
                response: ResponseSpec::Working(index),
            }
        });
        let mut sums = executor.run(&request)?;
        if let Some(eval) = deviance_eval {
            ls.deviance = sums[eval].deviance().0;
        }
        let accumulated = take_gram_sum(&mut sums, gram_eval);
        if max_iterations > 0 {
            start_gram = Some(accumulated);
        } else {
            retained_gram = Some(accumulated);
        }
    }

    if has_monotonic {
        if let Some(start_gram) = start_gram {
            monotone_pirls(executor, settings, &shape, &mut ls, start_gram)?;
        }
        // Before the exit checks, a final unconstrained-block re-solve: holding
        // the monotone blocks fixed, the remaining coordinates form a convex
        // penalized GLM the stalled main loop may have left slightly off-optimum.
        if ls.iteration < max_iterations && refine_over_passes(executor, settings, &shape, &mut ls)?
        {
            ls.warnings.push(
                "Applied final unconstrained-block re-solve (parametric and \
                 unconstrained smooth coordinates) to restore exit stationarity."
                    .to_string(),
            );
        }
    } else if let Some(start_gram) = start_gram {
        retained_gram = Some(gcv_pirls(executor, settings, &shape, &mut ls, start_gram)?);
    }

    // Result assembly from the covariance weights' Gram on X (`compute_xtwx` form):
    // held from the last iteration when it was on X, else pass R.
    let retained_gram = match retained_gram {
        Some(gram) => gram,
        None => {
            let mut request = PassRequest::reconstruct();
            let weights = ls.final_weights.spec(&mut request);
            let index = request.predictor(ls.state.clone());
            let eval = request.eval(Eval::Gram {
                weights,
                response: ResponseSpec::Working(index),
            });
            let mut sums = executor.run(&request)?;
            take_gram_sum(&mut sums, eval)
        }
    };
    let xtwx = packed_to_array2(&retained_gram.xtx_packed, p);
    drop(retained_gram);
    let lambdas = ls.lambdas.clone();
    let mut smooth_edfs = Vec::with_capacity(specs.len());
    for (i, spec) in specs.iter().enumerate() {
        let (start_col, k) = (spec.col_start, spec.penalty.nrows());
        let xtwx_block = xtwx
            .slice(s![start_col..start_col + k, start_col..start_col + k])
            .to_owned();
        smooth_edfs.push(compute_edf(&xtwx_block, &spec.penalty, lambdas[i]));
    }
    let total_edf = (shape.p_param as f64) + smooth_edfs.iter().sum::<f64>();
    let mut full_penalty = Array2::zeros((p, p));
    for (i, spec) in specs.iter().enumerate() {
        embed_penalty(&mut full_penalty, &spec.penalty, spec.col_start, lambdas[i]);
    }
    let covariance =
        invert_matrix(&add_penalty_to_xtwx(&xtwx, &full_penalty)).ok_or_else(|| {
            RustyStatsError::LinearAlgebraError(
                "The smooth fit's penalised normal equations X'WX + S are singular, so its \
             covariance is undefined. This usually means collinear or empty basis \
             columns; remove them or raise the smoothing penalty."
                    .to_string(),
            )
        })?;
    drop(xtwx);

    // Exit: KKT sums, intercept scores and statistics at the final state (pass A).
    let few_passes = executor.prefers_few_passes();
    let intercept_col = intercept_candidates
        .iter()
        .zip(non_one.iter())
        .find(|(_, &count)| count == 0.0)
        .map(|(&column, _)| column);
    let mut shifts = Vec::new();
    if intercept_col.is_some() {
        shifts.push(0.0);
        if few_passes {
            for span in INTERCEPT_BRACKET_SPANS {
                shifts.push(-span);
                shifts.push(span);
            }
        }
    }
    let state = ls.state.clone();
    let mut fitted = state.clone();
    let mut exit = exit_pass(
        executor,
        &state,
        &kkt_columns,
        &shifts,
        &facts,
        RobustRequest {
            types: settings.robust,
            weights: &ls.final_weights,
            bread: &covariance,
        },
    )?;
    let kkt_position = |column: usize| {
        kkt_columns
            .binary_search(&column)
            .expect("an unpenalized column has KKT sums")
    };
    if let Some(intercept_col) = intercept_col {
        let position = kkt_position(intercept_col);
        let (score, info) = (exit.kkt[2 * position], exit.kkt[2 * position + 1]);
        let intercept_std_score = score.abs() / info.max(ZERO_TOL).sqrt();
        if intercept_std_score >= 0.1 * SMOOTH_KKT_SCORE_TOL {
            let (shift, mut evaluations) =
                solve_intercept_shift_over_passes(executor, &state, &exit.shift_scores)?;
            let mut status = match shift {
                None => "unbracketed",
                Some(_) => "zero",
            };
            let mut applied = false;
            if let Some(shift) = shift.filter(|&shift| shift != 0.0) {
                // The deviance at the shifted state decides the refresh.
                evaluations += 1;
                status = "rejected";
                let shifted = Predictor::Shifted(Box::new(state.clone()), shift);
                let refreshed = exit_pass(
                    executor,
                    &shifted,
                    &kkt_columns,
                    &[],
                    &facts,
                    RobustRequest {
                        types: settings.robust,
                        weights: &ls.final_weights,
                        bread: &covariance,
                    },
                )?;
                // The exact 1-D score solve minimizes deviance along the
                // intercept direction; reject on the (non-canonical-link)
                // off-chance it does not improve within tolerance.
                let deviance = ls.deviance;
                if refreshed.deviance <= deviance + shape.smooth_tolerance * (1.0 + deviance.abs())
                {
                    status = "applied";
                    applied = true;
                    ls.coefficients[intercept_col] += shift;
                    ls.deviance = refreshed.deviance;
                    fitted = shifted;
                    exit = refreshed;
                    ls.warnings.push(format!(
                        "Applied final intercept refresh of {shift:+.6e} to restore \
                         the intercept score equation (mean prediction = mean response)."
                    ));
                }
            }
            record_step(
                executor,
                SolverStep::new(driver, "refresh", evaluations, applied, status),
            );
        }
    }

    // KKT stationarity over the unpenalized coordinates.
    let nonneg: std::collections::BTreeSet<usize> =
        nonneg_indices.unwrap_or(&[]).iter().copied().collect();
    let nonpos: std::collections::BTreeSet<usize> =
        nonpos_indices.unwrap_or(&[]).iter().copied().collect();
    let mut max_std_score = 0.0f64;
    for (position, &col) in kkt_columns.iter().enumerate() {
        let (score, info) = (exit.kkt[2 * position], exit.kkt[2 * position + 1]);
        let std_score = score.abs() / info.max(ZERO_TOL).sqrt();
        let at_lower_bound = nonneg.contains(&col) && ls.coefficients[col] <= ZERO_TOL;
        let at_upper_bound = nonpos.contains(&col) && ls.coefficients[col] >= -ZERO_TOL;
        if (at_lower_bound && score <= 0.0) || (at_upper_bound && score >= 0.0) {
            continue;
        }
        max_std_score = max_std_score.max(std_score);
    }
    let stationary = max_std_score < SMOOTH_KKT_SCORE_TOL;
    let converged_raw = ls.converged;
    let converged = converged_raw && stationary;
    if !stationary {
        ls.warnings.push(format!(
            "Fit is non-stationary at exit: max standardized score {max_std_score:.3e} \
             over unpenalized coordinates (tolerance {SMOOTH_KKT_SCORE_TOL:.0e})."
        ));
    }
    let solver_status = smooth_solver_status(ls.step_halving_failed, converged_raw, stationary);
    if should_warn_smooth_nonconvergence(converged) {
        ls.warnings
            .push(smooth_nonconvergence_warning(solver_status));
    }
    record_step(
        executor,
        SolverStep::new(driver, "result", ls.iteration, converged, solver_status),
    );

    let statistics = finish_statistics(executor, family, &fitted, &facts, &exit.sums, exit.stats)?;
    let mut smooth_penalty = SmoothPenalty::new();
    for (i, spec) in specs.iter().enumerate() {
        smooth_penalty.add_term(
            spec.penalty.clone(),
            lambdas[i],
            spec.col_start..spec.col_end,
        );
    }
    Ok(SmoothDriveOutcome {
        coefficients: ls.coefficients,
        deviance: ls.deviance,
        iterations: ls.iteration,
        converged,
        lambdas,
        smooth_edfs,
        total_edf,
        gcv: gcv_score(ls.deviance, n, total_edf),
        covariance_unscaled: covariance,
        penalty: Penalty::Smooth(smooth_penalty),
        warnings: ls.warnings,
        step_halving_used: ls.step_halving_used,
        solver_status,
        stationary,
        max_std_score,
        fitted,
        weights: ls.final_weights,
        statistics,
    })
}

/// A pass's deviance (total, all rows finite) and Gram, each when asked for.
type DevianceAndGram = (Option<(f64, bool)>, Option<GramPartial>);

/// A monotone PIRLS trial: its coefficients and its per-term α.
type MonotoneTrial = (Array1<f64>, Vec<Option<Array1<f64>>>);

/// One pass at `state`: its deviance (when asked) and the Gram `gram` (when asked).
fn deviance_and_gram(
    executor: &mut dyn PassExecutor,
    state: &Predictor,
    with_deviance: bool,
    gram: Option<&dyn Fn(usize) -> Eval>,
) -> Result<DevianceAndGram> {
    let mut request = PassRequest::default();
    let index = request.predictor(state.clone());
    let deviance = with_deviance.then(|| request.eval(Eval::Deviance(index)));
    let gram_eval = gram.map(|gram| request.eval(gram(index)));
    let mut sums = executor.run(&request)?;
    Ok((
        deviance.map(|eval| sums[eval].deviance()),
        gram_eval.map(|eval| take_gram_sum(&mut sums, eval)),
    ))
}

/// Penalised IRLS with interleaved GCV λ updates (fits without monotone terms).
/// Returns the Gram of the covariance weights (the last iteration's).
fn gcv_pirls(
    executor: &mut dyn PassExecutor,
    settings: &SmoothDriveSettings<'_>,
    shape: &SmoothShape,
    ls: &mut SmoothLoop,
    start_gram: GramPartial,
) -> Result<GramPartial> {
    let (specs, config) = (settings.specs, settings.config);
    let (nonneg_indices, nonpos_indices) = (settings.nonneg_indices, settings.nonpos_indices);
    let (n, p) = (shape.n, shape.p);
    let max_iterations = config.irls_config.max_iterations;
    let few_passes = executor.prefers_few_passes();
    let mut gram = Some(start_gram);
    let mut retained_gram = None;
    let mut penalty_matrix = Array2::zeros((p, p));
    let mut lambdas_stable_count = 0u32;
    let mut prev_lambdas: Vec<f64> = vec![f64::NAN; specs.len()];
    let mut penalty_dirty = true;

    while ls.iteration < max_iterations {
        ls.iteration += 1;
        let deviance_old = ls.deviance;
        let system = gram
            .take()
            .expect("the pass at this iteration's state accumulated its Gram");
        // These weights are the covariance's if the loop ends here.
        ls.final_weights = RetainedWeights::SmoothSolve(ls.state.clone());
        let cached_xtwx = packed_to_dmatrix(&system.xtx_packed, p);
        let cached_xtwz = DVector::from_column_slice(&system.xtz);
        let ztwz = system.ztwz;
        retained_gram = Some(system);

        if should_run_smooth_gcv(lambdas_stable_count, ls.iteration) {
            let old_lambdas = ls.lambdas.clone();
            let penalties: Vec<Array2<f64>> = specs.iter().map(|s| s.penalty.clone()).collect();
            let optimizer = MultiTermGCVOptimizer::new_from_cached(
                cached_xtwx.clone(),
                cached_xtwz.clone(),
                ztwz,
                penalties,
                shape.term_indices.clone(),
                n,
                shape.p_param,
            );
            ls.lambdas = optimizer.optimize_lambdas(
                &ls.lambdas,
                shape.log_lambda_min,
                shape.log_lambda_max,
                config.lambda_tol,
                3,
            );
            let max_rel_change = max_lambda_relative_change(&old_lambdas, &ls.lambdas);
            lambdas_stable_count = next_lambdas_stable_count(max_rel_change, lambdas_stable_count);
            penalty_dirty = lambdas_changed(&ls.lambdas, &prev_lambdas);
            record_step(
                executor,
                SolverStep::new(
                    "smooth",
                    "lambda_update",
                    0,
                    lambdas_changed(&ls.lambdas, &old_lambdas),
                    "gcv",
                ),
            );
        }

        if penalty_dirty {
            penalty_matrix.fill(0.0);
            for (i, spec) in specs.iter().enumerate() {
                embed_penalty(
                    &mut penalty_matrix,
                    &spec.penalty,
                    spec.col_start,
                    ls.lambdas[i],
                );
            }
            prev_lambdas.clone_from_slice(&ls.lambdas);
            penalty_dirty = false;
        }

        let (mut new_coef, _) =
            solve_wls_from_precomputed(&cached_xtwx, &cached_xtwz, &penalty_matrix, true)?;
        // Enforce sign constraints before evaluating deviance
        project_coefficients_to_bounds(&mut new_coef, nonneg_indices, nonpos_indices);

        // The penalized deviance is the objective this solver optimizes.
        let lambdas = ls.lambdas.clone();
        let pen_dev_old =
            smooth_penalized_deviance(deviance_old, &ls.coefficients, specs, &lambdas);
        let accept_threshold = smooth_accept_threshold(pen_dev_old, shape.smooth_tolerance);
        let more = ls.iteration < max_iterations;
        let acceptable = |dev: (f64, bool), coef: &Array1<f64>| {
            let pen_dev = smooth_penalized_deviance(dev.0, coef, specs, &lambdas);
            dev.1 && dev.0.is_finite() && pen_dev.is_finite() && pen_dev <= accept_threshold
        };

        // Candidate pass: its deviance, and the next iteration's Gram speculatively.
        let candidate = coef_predictor(&new_coef);
        let (candidate_deviance, candidate_gram) =
            deviance_and_gram(executor, &candidate, true, more.then_some(&smooth_gram))?;
        let candidate_deviance = candidate_deviance.expect("asked for");

        let accepted_gram = if acceptable(candidate_deviance, &new_coef) {
            record_step(
                executor,
                SolverStep::new("smooth", "iteration", 1, true, "full_step"),
            );
            ls.coefficients = new_coef;
            ls.state = candidate;
            ls.deviance = candidate_deviance.0;
            candidate_gram
        } else {
            // 10 halving steps plus a final fraction = 0 fallback (the previous
            // accepted coefficients), which is known finite and non-worsening.
            let mut fractions = Vec::with_capacity(11);
            let mut step = 0.5;
            for half_step in 0..=10 {
                fractions.push(if half_step == 10 { 0.0 } else { step });
                step = next_smooth_half_step(step);
            }
            let blends: Vec<Array1<f64>> = fractions
                .iter()
                .map(|&fraction| {
                    let mut blended = blend_arrays(&ls.coefficients, &new_coef, fraction);
                    project_coefficients_to_bounds(&mut blended, nonneg_indices, nonpos_indices);
                    blended
                })
                .collect();
            let predictors: Vec<Predictor> = blends.iter().map(coef_predictor).collect();
            let mut deviances = LazyDeviances::new(predictors, few_passes);
            let mut accepted = None;
            for trial in 0..blends.len() {
                let trial_deviance = deviances.get(executor, trial)?;
                if acceptable(trial_deviance, &blends[trial]) {
                    accepted = Some((trial, trial_deviance));
                    break;
                }
            }
            let Some((trial, trial_deviance)) = accepted else {
                record_step(
                    executor,
                    SolverStep::new("smooth", "iteration", 1 + blends.len(), false, "exhausted"),
                );
                ls.step_halving_failed = true;
                ls.warnings.push(
                    "Smooth IRLS step halving found no finite non-worsening step; \
                     retained the previous iterate."
                        .to_string(),
                );
                break;
            };
            // A fraction-0 fallback is not a genuine halving "step", it simply
            // retains the previous iterate.
            let genuine = fraction_is_genuine_halving(fractions[trial]);
            record_step(
                executor,
                SolverStep::new(
                    "smooth",
                    "iteration",
                    trial + 2,
                    true,
                    if genuine { "halved" } else { "retained" },
                ),
            );
            if genuine {
                ls.step_halving_used = true;
            }
            ls.coefficients = blends[trial].clone();
            ls.state = coef_predictor(&ls.coefficients);
            ls.deviance = trial_deviance.0;
            None
        };

        let pen_dev = smooth_penalized_deviance(ls.deviance, &ls.coefficients, specs, &lambdas);
        let rel_change = relative_change_with_floor(pen_dev_old, pen_dev, ZERO_TOL);
        if change_below_tolerance(rel_change, shape.smooth_tolerance) {
            ls.converged = true;
            break;
        }
        gram = match accepted_gram {
            Some(accepted_gram) => Some(accepted_gram),
            None if more => deviance_and_gram(executor, &ls.state, false, Some(&smooth_gram))?.1,
            None => None,
        };
    }
    Ok(retained_gram.expect("at least one iteration ran"))
}

/// Deviances of a list of trial predictors, evaluated on demand: one per pass, or
/// every remaining one in a single pass when the executor prefers few passes.
struct LazyDeviances {
    predictors: Vec<Predictor>,
    values: Vec<Option<(f64, bool)>>,
    few_passes: bool,
}

impl LazyDeviances {
    fn new(predictors: Vec<Predictor>, few_passes: bool) -> Self {
        let values = vec![None; predictors.len()];
        Self {
            predictors,
            values,
            few_passes,
        }
    }

    /// Record a deviance already evaluated (a candidate pass's).
    fn set(&mut self, trial: usize, value: (f64, bool)) {
        self.values[trial] = Some(value);
    }

    fn get(&mut self, executor: &mut dyn PassExecutor, trial: usize) -> Result<(f64, bool)> {
        if let Some(value) = self.values[trial] {
            return Ok(value);
        }
        let last = if self.few_passes {
            self.predictors.len()
        } else {
            trial + 1
        };
        let mut request = PassRequest::default();
        let mut evals = Vec::new();
        for t in trial..last {
            if self.values[t].is_none() {
                let index = request.predictor(self.predictors[t].clone());
                evals.push((t, request.eval(Eval::Deviance(index))));
            }
        }
        let sums = executor.run(&request)?;
        for (t, eval) in evals {
            self.values[t] = Some(sums[eval].deviance());
        }
        Ok(self.values[trial].expect("just evaluated"))
    }
}

/// The monotone transform of the current α: per monotone term, the column scales
/// (1, then ±exp(α_j) clamped) and the linearisation offset β(α) − J(α)α.
fn monotone_transform(
    specs: &[SmoothTermSpec],
    alpha_params: &[Option<Array1<f64>>],
) -> MonotoneTransform {
    let terms = specs
        .iter()
        .zip(alpha_params.iter())
        .filter_map(|(spec, alpha)| {
            let alpha = alpha.as_ref()?;
            let alpha_slice = alpha.as_slice().expect("contiguous array");
            let sign = match spec.monotonicity {
                Monotonicity::Increasing => 1.0,
                Monotonicity::Decreasing => -1.0,
                Monotonicity::None => return None,
            };
            let scales = (0..alpha_slice.len())
                .map(|j| {
                    if j == 0 {
                        1.0 // Column 0: unscaled cumulative sum
                    } else {
                        sign * alpha_slice[j].clamp(-MAX_EXP_ALPHA, MAX_EXP_ALPHA).exp()
                    }
                })
                .collect();
            let beta_mono = alpha_to_beta(alpha_slice, &spec.monotonicity);
            let j_mat = compute_monotonic_jacobian(alpha_slice, &spec.monotonicity);
            let j_alpha = j_mat.dot(alpha);
            let offset = monotonic_offset(&beta_mono, &j_alpha).to_vec();
            Some(MonotoneTermTransform {
                start: spec.col_start,
                end: spec.col_end,
                scales,
                offset,
            })
        })
        .collect();
    MonotoneTransform { terms }
}

/// The penalties of the reparameterised problem: S̃ = J'SJ for monotone terms.
fn transformed_penalties(
    specs: &[SmoothTermSpec],
    alpha_params: &[Option<Array1<f64>>],
) -> Vec<Array2<f64>> {
    specs
        .iter()
        .zip(alpha_params.iter())
        .map(|(spec, alpha)| match alpha {
            Some(alpha) => {
                let j_mat = compute_monotonic_jacobian(
                    alpha.as_slice().expect("contiguous array"),
                    &spec.monotonicity,
                );
                compute_s_tilde(&spec.penalty, &j_mat)
            }
            None => spec.penalty.clone(),
        })
        .collect()
}

/// The monotone path's nested iteration (scam, Pya & Wood 2015): REML λ on the
/// start Gram, an unconstrained warm start, the α reset, then PIRLS with fixed λ
/// inside an outer REML λ update, on the monotone-transformed rows.
fn monotone_pirls(
    executor: &mut dyn PassExecutor,
    settings: &SmoothDriveSettings<'_>,
    shape: &SmoothShape,
    ls: &mut SmoothLoop,
    start_gram: GramPartial,
) -> Result<()> {
    let (specs, config) = (settings.specs, settings.config);
    let (nonneg_indices, nonpos_indices) = (settings.nonneg_indices, settings.nonpos_indices);
    let (n, p) = (shape.n, shape.p);
    let few_passes = executor.prefers_few_passes();
    let budget = config.irls_config.max_iterations;

    // Phase 1: REML on the unconstrained basis for starting lambda.
    {
        let optimizer = MultiTermGCVOptimizer::new_from_cached(
            packed_to_dmatrix(&start_gram.xtx_packed, p),
            DVector::from_column_slice(&start_gram.xtz),
            start_gram.ztwz,
            specs.iter().map(|s| s.penalty.clone()).collect(),
            shape.term_indices.clone(),
            n,
            shape.p_param,
        );
        let old_lambdas = ls.lambdas.clone();
        ls.lambdas = optimizer.optimize_lambdas_reml(
            &ls.lambdas,
            shape.log_lambda_min,
            shape.log_lambda_max,
            config.lambda_tol,
            5,
        );
        record_step(
            executor,
            SolverStep::new(
                "monotone",
                "lambda_update",
                0,
                lambdas_changed(&ls.lambdas, &old_lambdas),
                "reml",
            ),
        );
    }

    // Warm start: unconstrained penalised IRLS, counted against the caller's
    // total iteration budget, then betas → alpha (with PAV projection).
    let mut init_coef = ls.coefficients.clone();
    let mut init_dev = ls.deviance;
    let warm_cap = SMOOTH_WARM_START_MAX_ITER.min(budget);
    let mut warm_gram = Some(start_gram);
    for warm in 0..warm_cap {
        ls.iteration += 1;
        let gram = warm_gram
            .take()
            .expect("the pass at the warm start's state accumulated its Gram");
        let mut init_penalty = Array2::zeros((p, p));
        for (i, spec) in specs.iter().enumerate() {
            embed_penalty(
                &mut init_penalty,
                &spec.penalty,
                spec.col_start,
                ls.lambdas[i],
            );
        }
        let (new_init_coef, _) = solve_wls_from_precomputed(
            &packed_to_dmatrix(&gram.xtx_packed, p),
            &DVector::from_column_slice(&gram.xtz),
            &init_penalty,
            true,
        )?;
        init_coef = new_init_coef;
        let more = warm + 1 < warm_cap;
        let (deviance, next_gram) = deviance_and_gram(
            executor,
            &coef_predictor(&init_coef),
            true,
            more.then_some(&smooth_gram),
        )?;
        let new_dev = deviance.expect("asked for").0;
        let init_rel = relative_change_with_unit_offset(init_dev, new_dev);
        init_dev = new_dev;
        let settled = change_below_tolerance(init_rel, config.irls_config.tolerance);
        record_step(
            executor,
            SolverStep::new(
                "monotone",
                "warm_start",
                1,
                true,
                if settled { "converged" } else { "step" },
            ),
        );
        if settled {
            break;
        }
        warm_gram = next_gram;
    }

    // Convert unconstrained betas to alpha; set the coefficients from alpha.
    let mut alpha_params: Vec<Option<Array1<f64>>> = specs
        .iter()
        .map(|spec| {
            spec.is_monotonic().then(|| {
                let beta_slice =
                    &init_coef.as_slice().expect("contiguous")[spec.col_start..spec.col_end];
                beta_to_alpha(beta_slice, &spec.monotonicity)
            })
        })
        .collect();
    ls.coefficients = init_coef;
    for (spec, alpha) in specs.iter().zip(alpha_params.iter()) {
        if let Some(alpha) = alpha {
            let beta = alpha_to_beta(
                alpha.as_slice().expect("contiguous array"),
                &spec.monotonicity,
            );
            let cs = ls.coefficients.as_slice_mut().expect("contiguous array");
            for (j, &b) in beta.iter().enumerate() {
                cs[smooth_coefficient_index(spec.col_start, j)] = b;
            }
        }
    }
    project_coefficients_to_bounds(&mut ls.coefficients, nonneg_indices, nonpos_indices);
    ls.state = coef_predictor(&ls.coefficients);
    let tilde = |transform: MonotoneTransform| {
        let transform = std::sync::Arc::new(transform);
        move |predictor: usize| Eval::MonotoneGram {
            predictor,
            transform: transform.clone(),
        }
    };
    // The state after the reset, and the first inner iteration's transformed Gram.
    let mut tilde_gram = {
        let gram = (ls.iteration < budget).then(|| tilde(monotone_transform(specs, &alpha_params)));
        let (deviance, gram) = deviance_and_gram(
            executor,
            &ls.state,
            true,
            gram.as_ref().map(|g| g as &dyn Fn(usize) -> Eval),
        )?;
        ls.deviance = deviance.expect("asked for").0;
        record_step(
            executor,
            SolverStep::new("monotone", "alpha_reset", 1, true, "reset"),
        );
        gram
    };

    // Outer loop: update lambda, then converge inner PIRLS, within the caller's
    // total iteration budget.
    while ls.iteration < budget {
        let lambdas_at_start = ls.lambdas.clone();

        for _inner in 0..SMOOTH_INNER_MAX_PER_CYCLE {
            if ls.iteration >= budget {
                break;
            }
            ls.iteration += 1;
            let deviance_old = ls.deviance;
            ls.final_weights = RetainedWeights::SmoothSolve(ls.state.clone());
            let pre_wls_alphas = alpha_params.clone();
            let gram = tilde_gram
                .take()
                .expect("the pass at this iteration's state accumulated its transformed Gram");

            let mut penalty_matrix = Array2::zeros((p, p));
            for (i, (spec, penalty)) in specs
                .iter()
                .zip(transformed_penalties(specs, &alpha_params))
                .enumerate()
            {
                embed_penalty(&mut penalty_matrix, &penalty, spec.col_start, ls.lambdas[i]);
            }
            let (alpha_coef, _) = solve_wls_from_precomputed(
                &packed_to_dmatrix(&gram.xtx_packed, p),
                &DVector::from_column_slice(&gram.xtz),
                &penalty_matrix,
                true,
            )?;
            let new_coef = alpha_coef.clone();
            // WLS-proposed alpha for monotonic terms (clamped)
            let wls_alphas: Vec<Option<Array1<f64>>> = specs
                .iter()
                .enumerate()
                .map(|(i, spec)| {
                    alpha_params[i].as_ref().map(|_| {
                        let alpha_wls = &alpha_coef.as_slice().expect("contiguous")
                            [spec.col_start..spec.col_end];
                        alpha_wls
                            .iter()
                            .enumerate()
                            .map(|(j, &a)| clamp_monotonic_alpha_value(j, a))
                            .collect()
                    })
                })
                .collect();

            let lambdas = ls.lambdas.clone();
            let pen_dev_old =
                smooth_penalized_deviance(deviance_old, &ls.coefficients, specs, &lambdas);

            // 20 halving trials plus a final fraction = 0 fallback (trial 20),
            // which exactly retains the previous accepted iterate.
            let trials: Vec<MonotoneTrial> = (0..=20)
                .map(|trial| {
                    let step = smooth_step_fraction_for_trial(trial, 20);
                    let mut trial_coef = blend_step_arrays(&ls.coefficients, &new_coef, step);
                    let mut trial_alphas = Vec::with_capacity(specs.len());
                    for (i, spec) in specs.iter().enumerate() {
                        if let (Some(old_alpha), Some(new_alpha)) =
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
                    project_coefficients_to_bounds(&mut trial_coef, nonneg_indices, nonpos_indices);
                    (trial_coef, trial_alphas)
                })
                .collect();

            // Candidate pass (trial 0): its deviance and, speculatively, the next
            // transformed Gram at its state and α.
            let more = ls.iteration < budget;
            let candidate = coef_predictor(&trials[0].0);
            let candidate_gram_eval = more.then(|| tilde(monotone_transform(specs, &trials[0].1)));
            let (candidate_deviance, candidate_gram) = deviance_and_gram(
                executor,
                &candidate,
                true,
                candidate_gram_eval
                    .as_ref()
                    .map(|g| g as &dyn Fn(usize) -> Eval),
            )?;
            let mut deviances = LazyDeviances::new(
                trials
                    .iter()
                    .map(|(coef, _)| coef_predictor(coef))
                    .collect(),
                few_passes,
            );
            deviances.set(0, candidate_deviance.expect("asked for"));

            let mut best_pen_dev = f64::INFINITY;
            let mut best_trial: Option<i32> = None;
            // Trials the sequential rule read, up to the one it stopped at.
            let mut examined = 0usize;
            for trial in 0..=20i32 {
                let t = trial as usize;
                examined = t + 1;
                let (trial_dev, finite) = deviances.get(executor, t)?;
                let trial_pen_dev =
                    smooth_penalized_deviance(trial_dev, &trials[t].0, specs, &lambdas);
                if !(finite && trial_dev.is_finite() && trial_pen_dev.is_finite()) {
                    continue;
                }
                if smooth_trial_is_better(best_trial, trial_pen_dev, best_pen_dev) {
                    best_pen_dev = trial_pen_dev;
                    best_trial = Some(trial);
                    if immediate_trial_accepts(trial, trial_pen_dev, pen_dev_old) {
                        break;
                    }
                }
                if halved_trial_accepts(trial, best_pen_dev, pen_dev_old) {
                    break;
                }
            }

            let accept_threshold = smooth_accept_threshold(pen_dev_old, shape.smooth_tolerance);
            let failed = smooth_step_halving_failed(best_trial, best_pen_dev, accept_threshold);
            record_step(
                executor,
                SolverStep::new(
                    "monotone",
                    "iteration",
                    examined,
                    !failed,
                    match best_trial {
                        _ if failed => "exhausted",
                        Some(0) => "full_step",
                        _ if genuine_halved_step(best_trial, 20) => "halved",
                        _ => "retained",
                    },
                ),
            );
            if failed {
                ls.step_halving_failed = true;
                ls.warnings.push(
                    "Smooth PIRLS step halving found no finite non-worsening step; \
                     retained the previous iterate."
                        .to_string(),
                );
                break;
            }
            // trial 20 is the fraction-0 fallback (retained previous iterate),
            // not a genuine halved step.
            if genuine_halved_step(best_trial, 20) {
                ls.step_halving_used = true;
            }
            let best = best_trial.expect("an accepted step has a trial") as usize;
            let (best_coef, best_alphas) = trials[best].clone();
            for (alpha, best_alpha) in alpha_params.iter_mut().zip(best_alphas.iter()) {
                if let (Some(alpha), Some(best_alpha)) = (alpha.as_mut(), best_alpha) {
                    *alpha = best_alpha.clone();
                }
            }
            ls.coefficients = best_coef;
            ls.state = coef_predictor(&ls.coefficients);
            ls.deviance = deviances.get(executor, best)?.0;

            // The transformed Gram at the new state and α, for the next inner
            // iteration or the outer REML update.
            tilde_gram = if best == 0 {
                candidate_gram
            } else if more {
                let eval = tilde(monotone_transform(specs, &alpha_params));
                deviance_and_gram(executor, &ls.state, false, Some(&eval))?.1
            } else {
                None
            };

            // Inner convergence: penalized deviance change
            let rel_change = relative_change_with_unit_offset(pen_dev_old, best_pen_dev);
            if change_below_tolerance(rel_change, shape.smooth_tolerance) {
                ls.converged = true;
                break;
            }
        }
        if ls.step_halving_failed {
            break;
        }
        if ls.iteration >= budget {
            // Budget exhausted mid-cycle: the inner state may claim
            // convergence, but lambda stability was never verified.
            ls.converged = false;
            break;
        }

        // Outer loop: REML λ update on the transformed sums at the current state
        // and α (the Gram the next inner iteration also uses).
        {
            let gram = tilde_gram
                .as_ref()
                .expect("the transformed Gram at the current state was accumulated");
            let optimizer = MultiTermGCVOptimizer::new_from_cached(
                packed_to_dmatrix(&gram.xtx_packed, p),
                DVector::from_column_slice(&gram.xtz),
                gram.ztwz,
                transformed_penalties(specs, &alpha_params),
                shape.term_indices.clone(),
                n,
                shape.p_param,
            );
            ls.lambdas = optimizer.optimize_lambdas_reml(
                &ls.lambdas,
                shape.log_lambda_min,
                shape.log_lambda_max,
                config.lambda_tol,
                5,
            );
        }
        record_step(
            executor,
            SolverStep::new(
                "monotone",
                "lambda_update",
                0,
                lambdas_changed(&ls.lambdas, &lambdas_at_start),
                "reml",
            ),
        );

        // Check outer convergence: lambdas stabilized?
        let max_lambda_change = max_lambda_relative_change(&lambdas_at_start, &ls.lambdas);
        if should_stop_monotonic_outer(max_lambda_change, ls.converged) {
            break;
        }
        // Reset inner convergence flag for next outer iteration
        ls.converged = false;
    }
    Ok(())
}

/// `refine_unconstrained_block` over passes: penalised IRLS on the refit columns
/// R with the monotone and sign-constrained columns F held as the offset
/// o_F = o + Σ_F β_c·x_c (ascending columns, D11). Returns whether it was applied.
fn refine_over_passes(
    executor: &mut dyn PassExecutor,
    settings: &SmoothDriveSettings<'_>,
    shape: &SmoothShape,
    ls: &mut SmoothLoop,
) -> Result<bool> {
    let specs = settings.specs;
    let budget = settings.config.irls_config.max_iterations;
    let mut fixed_cols = std::collections::BTreeSet::new();
    for spec in specs {
        if spec.is_monotonic() {
            fixed_cols.extend(spec.col_start..spec.col_end);
        }
    }
    fixed_cols.extend(settings.nonneg_indices.unwrap_or(&[]).iter().copied());
    fixed_cols.extend(settings.nonpos_indices.unwrap_or(&[]).iter().copied());
    let refit_cols: Vec<usize> = (0..shape.p)
        .filter(|col| !fixed_cols.contains(col))
        .collect();
    if refit_cols.is_empty() || fixed_cols.is_empty() || ls.iteration >= budget {
        // Nothing to refine, or the whole problem was already the
        // unconstrained block (handled by the main loop), or no budget left.
        return Ok(false);
    }
    let n_refit = refit_cols.len();
    let refit: std::sync::Arc<[usize]> = refit_cols.iter().copied().collect();
    let fixed: std::sync::Arc<[(usize, f64)]> = fixed_cols
        .iter()
        .map(|&col| (col, ls.coefficients[col]))
        .filter(|&(_, coef)| coef != 0.0)
        .collect();

    // Penalties of the unconstrained smooth terms, re-indexed into sub-columns.
    let mut penalty_sub = Array2::zeros((n_refit, n_refit));
    for (i, spec) in specs.iter().enumerate() {
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
            ls.lambdas[i],
        );
    }
    let write_back = |beta_sub: &Array1<f64>, base: &Array1<f64>| -> Array1<f64> {
        let mut full = base.clone();
        for (j_new, &j_old) in refit_cols.iter().enumerate() {
            full[j_old] = beta_sub[j_new];
        }
        full
    };
    let partitioned = |beta: &Array1<f64>| Predictor::Partitioned {
        refit: refit.clone(),
        beta: beta.iter().copied().collect(),
        fixed: fixed.clone(),
    };
    let sub_gram = |predictor: usize| Eval::SubGram {
        columns: refit.clone(),
        weights: WeightsSpec::SmoothIrls(predictor),
        response: ResponseSpec::PartitionedWorking {
            predictor,
            fixed: fixed.clone(),
        },
    };

    let lambdas = ls.lambdas.clone();
    let tolerance = shape.smooth_tolerance;
    let pen_dev_start = smooth_penalized_deviance(ls.deviance, &ls.coefficients, specs, &lambdas);
    let mut beta_sub: Array1<f64> = refit_cols.iter().map(|&c| ls.coefficients[c]).collect();
    let mut state = ls.state.clone();
    let mut dev_cur = ls.deviance;
    let mut pen_dev_cur = pen_dev_start;
    let mut gram: Option<GramPartial> = None;
    let few_passes = executor.prefers_few_passes();

    while ls.iteration < budget {
        ls.iteration += 1;
        let system = match gram.take() {
            Some(system) => system,
            None => deviance_and_gram(executor, &state, false, Some(&sub_gram))?
                .1
                .expect("asked for"),
        };
        let (beta_new, _) = solve_wls_from_precomputed(
            &packed_to_dmatrix(&system.xtx_packed, n_refit),
            &DVector::from_column_slice(&system.xtz),
            &penalty_sub,
            true,
        )?;

        // Step halving on the full penalized objective; the 0-fraction fallback
        // exactly retains the previous iterate.
        let threshold = smooth_accept_threshold(pen_dev_cur, tolerance);
        let betas: Vec<Array1<f64>> = (0..=10)
            .map(|trial| {
                blend_step_arrays(
                    &beta_sub,
                    &beta_new,
                    smooth_step_fraction_for_trial(trial, 10),
                )
            })
            .collect();
        let more = ls.iteration < budget;
        let (candidate_deviance, candidate_gram) = deviance_and_gram(
            executor,
            &partitioned(&betas[0]),
            true,
            more.then_some(&sub_gram),
        )?;
        let mut deviances =
            LazyDeviances::new(betas.iter().map(&partitioned).collect(), few_passes);
        deviances.set(0, candidate_deviance.expect("asked for"));
        let mut accepted = None;
        let mut examined = 0usize;
        for (trial, beta_trial) in betas.iter().enumerate() {
            examined = trial + 1;
            let (dev_trial, finite) = deviances.get(executor, trial)?;
            let pen_dev_trial = smooth_penalized_deviance(
                dev_trial,
                &write_back(beta_trial, &ls.coefficients),
                specs,
                &lambdas,
            );
            if !(finite && dev_trial.is_finite() && pen_dev_trial.is_finite()) {
                continue;
            }
            if pen_dev_trial <= threshold {
                accepted = Some((trial, dev_trial, pen_dev_trial));
                break;
            }
        }
        let last_trial = betas.len() - 1;
        record_step(
            executor,
            SolverStep::new(
                "monotone",
                "refine_iteration",
                examined,
                accepted.is_some(),
                match accepted {
                    None => "exhausted",
                    Some((0, _, _)) => "full_step",
                    Some((trial, _, _)) if trial < last_trial => "halved",
                    Some(_) => "retained",
                },
            ),
        );
        let Some((trial, dev_trial, pen_dev_next)) = accepted else {
            break;
        };
        beta_sub = betas[trial].clone();
        state = partitioned(&beta_sub);
        dev_cur = dev_trial;
        gram = if trial == 0 { candidate_gram } else { None };

        let rel_change = relative_change_with_unit_offset(pen_dev_cur, pen_dev_next);
        pen_dev_cur = pen_dev_next;
        if change_below_tolerance(rel_change, tolerance) {
            break;
        }
    }

    let applied = pen_dev_cur <= pen_dev_start + tolerance * (1.0 + pen_dev_start.abs());
    record_step(
        executor,
        SolverStep::new(
            "monotone",
            "refine",
            0,
            applied,
            if applied { "applied" } else { "rejected" },
        ),
    );
    if applied {
        ls.coefficients = write_back(&beta_sub, &ls.coefficients);
        ls.state = state;
        ls.deviance = dev_cur;
        Ok(true)
    } else {
        Ok(false)
    }
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
        assert!(lambdas_changed(&[1.0, 2.0], &[1.0, 3.0]));
        assert!(!lambdas_changed(&[1.0, 2.0], &[1.0, 2.0]));
        assert_eq!(parametric_column_count(5, 2), 3);
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

    /// An in-memory executor that asks drivers to speculate, as an out-of-core
    /// executor does.
    struct Speculative<'a>(InMemoryExecutor<'a>);

    impl PassExecutor for Speculative<'_> {
        fn n_rows(&self) -> usize {
            self.0.n_rows()
        }
        fn n_params(&self) -> usize {
            self.0.n_params()
        }
        fn run(&mut self, request: &PassRequest) -> Result<Vec<EvalSum>> {
            self.0.run(request)
        }
        fn set_row_standardization(&mut self, s: Option<crate::regularization::Standardization>) {
            self.0.set_row_standardization(s)
        }
        fn set_intercept_only(&mut self, on: bool) {
            self.0.set_intercept_only(on)
        }
        fn prefers_few_passes(&self) -> bool {
            true
        }
        fn trace(&mut self) -> Option<&mut Vec<SolverStep>> {
            self.0.trace()
        }
    }

    fn in_memory<'a>(
        x: ArrayView2<'a, f64>,
        y: &'a Array1<f64>,
        offset: &'a Array1<f64>,
        prior: &'a Array1<f64>,
        family: &'a dyn Family,
        link: &'a dyn Link,
    ) -> InMemoryExecutor<'a> {
        let ctx = RowContext {
            family,
            link,
            y: y.as_slice().expect("contiguous"),
            offset: offset.as_slice().expect("contiguous"),
            prior_weights: prior.as_slice().expect("contiguous"),
            y_mean: crate::families::response_mean(y),
            min_weight: 1e-10,
            has_offset: false,
            standardization: None,
        };
        InMemoryExecutor::new(x, false, ctx).expect("executor")
    }

    #[test]
    fn test_intercept_shift_over_passes_matches_closed_forms_in_both_modes() {
        // Binomial/logit at eta = 0: the root of sum(y - sigmoid(d)) = 0 is
        // logit(mean(y)); Gaussian/identity: the root is mean(y) - mean(eta).
        let mut yv = vec![1.0; 7];
        yv.extend(vec![0.0; 3]);
        type Case<'a> = (Array1<f64>, &'a dyn Family, &'a dyn Link, f64);
        let cases: [Case<'_>; 2] = [
            (
                Array1::from_vec(yv),
                &BinomialFamily,
                &LogitLink,
                (0.7f64 / 0.3).ln(),
            ),
            (array![1.0, 2.0, 6.0], &GaussianFamily, &IdentityLink, 3.0),
        ];
        for (y, family, link, expected) in cases {
            let n = y.len();
            let x = Array2::zeros((n, 1));
            let (offset, prior) = (Array1::zeros(n), Array1::ones(n));
            let base = Predictor::coef(&[0.0]);
            let mut shifts = vec![0.0];
            for span in INTERCEPT_BRACKET_SPANS {
                shifts.extend([-span, span]);
            }
            let mut sequential = in_memory(x.view(), &y, &offset, &prior, family, link);
            let scores = shift_scores(&mut sequential, &base, &[0.0]).expect("pass");
            let (one, read) =
                solve_intercept_shift_over_passes(&mut sequential, &base, &scores).expect("passes");
            let one = one.expect("root is bracketed");
            assert_abs_diff_eq!(one, expected, epsilon = 1e-9);

            let mut speculative =
                Speculative(in_memory(x.view(), &y, &offset, &prior, family, link));
            let scores = shift_scores(&mut speculative, &base, &shifts).expect("pass");
            let (tree, tree_read) =
                solve_intercept_shift_over_passes(&mut speculative, &base, &scores)
                    .expect("passes");
            let tree = tree.expect("root is bracketed");
            assert_eq!(one.to_bits(), tree.to_bits(), "speculation changes nothing");
            assert_eq!(read, tree_read, "the trace counts the same scores");
        }
    }

    #[test]
    fn test_refine_over_passes_solves_conditional_optimum() {
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
        let (offset, prior) = (Array1::zeros(n), Array1::ones(n));
        let mut config = SmoothGLMConfig::default();
        config.irls_config.max_iterations = 50;
        let run = |monotonicity: Monotonicity, iteration: usize| {
            let specs = vec![SmoothTermSpec {
                col_start: 1,
                col_end: 2,
                penalty: Array2::zeros((1, 1)),
                monotonicity,
                initial_lambda: 0.0,
            }];
            let settings = SmoothDriveSettings {
                specs: &specs,
                config: &config,
                family: &GaussianFamily,
                nonneg_indices: None,
                nonpos_indices: None,
                null_offset: crate::diagnostics::NullOffset::Absent,
                robust: &[],
            };
            let shape = SmoothShape {
                n,
                p: 2,
                p_param: 1,
                term_indices: vec![(1, 2)],
                smooth_tolerance: 1e-8,
                log_lambda_min: 0.0,
                log_lambda_max: 0.0,
            };
            let coefficients = array![0.0, 3.0]; // slope at truth, intercept off
            let state = coef_predictor(&coefficients);
            let mut executor = in_memory(
                x.view(),
                &y,
                &offset,
                &prior,
                &GaussianFamily,
                &IdentityLink,
            );
            let (deviance, _) = deviance_and_gram(&mut executor, &state, true, None).expect("pass");
            let mut ls = SmoothLoop {
                coefficients,
                state,
                deviance: deviance.expect("asked for").0,
                lambdas: vec![0.0],
                iteration,
                converged: false,
                final_weights: RetainedWeights::Ones,
                step_halving_used: false,
                step_halving_failed: false,
                warnings: Vec::new(),
            };
            let refined =
                refine_over_passes(&mut executor, &settings, &shape, &mut ls).expect("refine");
            (refined, ls)
        };

        let (refined, ls) = run(Monotonicity::Increasing, 0);
        assert!(refined);
        assert_abs_diff_eq!(ls.coefficients[0], 2.0, epsilon = 1e-8);
        assert_abs_diff_eq!(ls.coefficients[1], 3.0, epsilon = 1e-12); // held fixed
        assert!(ls.deviance < 1e-12);
        assert!(ls.iteration >= 1);
        assert!(matches!(ls.state, Predictor::Partitioned { .. }));

        // Exhausted budget: no-op, state untouched.
        let (refined, ls) = run(Monotonicity::Increasing, 50);
        assert!(!refined);
        assert_eq!(ls.coefficients[0], 0.0);

        // No fixed block (nothing monotone): the main loop owns the whole
        // problem and the refine step declines.
        let (refined, _) = run(Monotonicity::None, 0);
        assert!(!refined);
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
    fn test_smooth_trace_and_fit_do_not_depend_on_speculation() {
        // A GCV fit and a monotone fit, driven one pass per trial (in memory) and
        // with every trial and bisection level speculated (out of core): the same
        // bits and the same solver trace.
        let n = 80;
        let x_vals: Array1<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        let y: Array1<f64> = x_vals
            .iter()
            .map(|&x| (0.4 + 1.2 * x + 0.3 * (9.0 * x).sin()).exp().round())
            .collect();
        let x_param = Array2::from_shape_fn((n, 1), |(_, _)| 1.0);
        let cases = [
            make_full_matrix(&x_param, &bs_basis(&x_vals, 8, 3, None, false)),
            make_full_matrix_with_monotonicity(
                &x_param,
                &is_basis(&x_vals, 6, 3, Some((0.0, 1.0)), true),
                Monotonicity::Increasing,
            ),
        ];
        for (x_full, specs) in cases {
            let (offset, prior) = (Array1::zeros(n), Array1::ones(n));
            let mut config = SmoothGLMConfig::default();
            config.irls_config.max_iterations = 30;
            let fit = |executor: &mut dyn PassExecutor| {
                fit_smooth_glm_with_executor(
                    executor,
                    &specs,
                    &PoissonFamily,
                    &LogLink,
                    &config,
                    None,
                    None,
                    false,
                )
                .expect("fit")
            };
            let mut sequential =
                in_memory(x_full.view(), &y, &offset, &prior, &PoissonFamily, &LogLink);
            let one = fit(&mut sequential);
            let one_trace = sequential.take_trace();
            let mut speculative = Speculative(in_memory(
                x_full.view(),
                &y,
                &offset,
                &prior,
                &PoissonFamily,
                &LogLink,
            ));
            let tree = fit(&mut speculative);
            let tree_trace = speculative.0.take_trace();
            let bits = |c: &Array1<f64>| c.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&one.coefficients), bits(&tree.coefficients));
            assert_eq!(one_trace, tree_trace);
            let last = one_trace.last().expect("a result step");
            assert_eq!(last.kind, "result");
            assert_eq!(last.status, one.solver_status);
            assert!(one_trace.iter().any(|step| step.kind == "iteration"));
        }
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
        // An intercept and an empty smooth column: X'X is singular, so the
        // unpenalised projection fails, while X'X + S inverts.
        let y = array![1.0, 2.0, 3.0, 4.0];
        let mut x_full = Array2::zeros((4, 2));
        x_full.column_mut(0).fill(1.0);
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
    fn test_fit_smooth_glm_singular_penalised_gram_is_an_error_not_identity() {
        // D1: an unpenalised empty column makes X'WX + S singular; the covariance
        // used to fall back to the identity matrix.
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
        let error = fit_smooth_glm_full_matrix(
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
        .expect_err("a singular penalised Gram is refused");
        assert!(error.to_string().contains("singular"), "{error}");
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
