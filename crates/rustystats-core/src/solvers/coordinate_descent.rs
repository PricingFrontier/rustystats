use super::validate_glm_inputs;
use ndarray::ArrayView2;

// =============================================================================
// Coordinate Descent for Penalized GLMs
// =============================================================================
//
// This module implements coordinate descent for fitting GLMs with L1 (Lasso)
// and Elastic Net penalties. Unlike Ridge, these penalties have non-differentiable
// components that require special handling.
//
// ALGORITHM: Iteratively Reweighted Coordinate Descent (IRCD)
// -----------------------------------------------------------
// We combine two ideas:
//   1. IRLS: Handle the GLM part by iteratively computing working response/weights
//   2. Coordinate Descent: Handle the L1 penalty by updating one coefficient at a time
//
// The algorithm:
//   1. Initialize coefficients β
//   2. Outer loop (IRLS-like):
//      a. Compute working response z and weights W from current μ
//      b. Inner loop (coordinate descent):
//         For each j = 1, ..., p:
//           - Compute partial residual without β_j
//           - Update β_j using soft-thresholding
//         Until convergence
//      c. Update μ = g⁻¹(Xβ)
//      d. Check deviance convergence
//
// SOFT THRESHOLDING
// -----------------
// The key operation for L1 penalty. For weighted least squares:
//
//   β_j = S(r_j, λα) / (Σw_i x_ij² + λ(1-α))
//
// where:
//   - r_j = Σw_i x_ij (z_i - Σ_{k≠j} x_ik β_k) is the "partial residual"
//   - S(z, γ) = sign(z) × max(0, |z| - γ) is soft-thresholding
//   - α is the L1 ratio (1 for Lasso, 0 for Ridge)
//   - λ is the overall penalty strength
//
// INTERCEPT
// ---------
// The intercept is NOT penalized. It's updated as:
//   β_0 = Σw_i (z_i - Σ_{j>0} x_ij β_j) / Σw_i
//
// =============================================================================

use ndarray::{Array1, Array2};

use super::irls::{
    coef_predictor, finish_statistics, request_statistics, FinalStatistics, FirstPassFacts,
    IRLSConfig, IRLSResult, RetainedWeights, RobustRequest, StatisticsEvals,
};
use super::passes::{
    record_step, rows_stay_sparse, Eval, EvalSum, Facts, InMemoryExecutor, PassExecutor,
    PassRequest, Predictor, ResponseSpec, RowContext, SolverStep, WeightsSpec,
};
use crate::constants::ZERO_TOL;
use crate::error::{Result, RustyStatsError};
use crate::families::Family;
use crate::inference::HCType;
use crate::links::Link;
use crate::regularization::{soft_threshold, RegularizationConfig, Standardization};
use crate::rows::GramPartial;

const ACTIVE_SET_REFRESH_INTERVAL: usize = 5;
const COEFFICIENT_UPDATE_TOL: f64 = 1e-15;

/// Fit a GLM using coordinate descent with L1/Elastic Net penalty.
///
/// This is the main entry point for Lasso and Elastic Net regularized GLMs. The
/// rows are served by an in-memory [`InMemoryExecutor`] to [`drive_cd`], the same
/// driver an out-of-core fit runs (`specs/out-of-core-fit.md` §7.5).
///
/// # Arguments
/// * `y` - Response variable (n × 1)
/// * `x` - Design matrix (n × p), should include intercept column as first column
/// * `family` - Distribution family (Gaussian, Poisson, Binomial, Gamma)
/// * `link` - Link function (Identity, Log, Logit)
/// * `irls_config` - Outer loop (IRLS) configuration
/// * `reg_config` - Regularization configuration (must have L1 component)
/// * `offset` - Optional offset term
/// * `weights` - Optional prior weights
/// * `init_coefficients` - Optional initial coefficients for warm starting
/// * `skip_covariance` - If true, skip the covariance and the final statistics
///   (useful during CV path where only coefficients are needed)
/// * `robust` - Robust covariance types for the final pass
/// * `row_standardization` - Standardize every row before the kernels (§6.7)
///
/// # Returns
/// * `Ok(IRLSResult)` - Fitted model results
/// * `Err(RustyStatsError)` - If fitting fails
#[allow(clippy::too_many_arguments)]
pub(crate) fn fit_glm_coordinate_descent(
    y: &Array1<f64>,
    x: ArrayView2<'_, f64>,
    family: &dyn Family,
    link: &dyn Link,
    irls_config: &IRLSConfig,
    reg_config: &RegularizationConfig,
    offset: Option<&Array1<f64>>,
    weights: Option<&Array1<f64>>,
    init_coefficients: Option<&Array1<f64>>,
    skip_covariance: bool,
    robust: &[HCType],
    row_standardization: Option<&Standardization>,
) -> Result<IRLSResult> {
    let validated = validate_glm_inputs(y, x, offset, weights)?;
    let offset_vec = validated.offset;
    let prior_weights_vec = validated.prior_weights;
    let y_slice = y.as_slice().ok_or_else(|| {
        RustyStatsError::LinearAlgebraError("Response y must be contiguous".to_string())
    })?;
    let y_mean = crate::families::response_mean(y);
    let sparse_rows =
        rows_stay_sparse(row_standardization) && irls_config.row_kernel.sparse_in_memory(x);
    let ctx = RowContext {
        family,
        link,
        y: y_slice,
        offset: offset_vec
            .as_slice()
            .expect("validated offset is contiguous"),
        prior_weights: prior_weights_vec
            .as_slice()
            .expect("validated prior weights are contiguous"),
        y_mean,
        min_weight: irls_config.min_weight,
        has_offset: offset.is_some(),
        standardization: None,
    };
    let mut executor = InMemoryExecutor::new(x, sparse_rows, ctx)?;
    executor.set_row_standardization(row_standardization.cloned());
    let outcome = drive_cd(
        &mut executor,
        &CdDriveSettings {
            config: irls_config,
            regularization: reg_config,
            init_coefficients,
            skip_covariance,
            y_mean,
            family,
            link,
            null_offset: crate::diagnostics::NullOffset::for_link(offset.is_some(), link),
            robust,
            statistics_without_covariance: false,
        },
    )?;

    // In-memory results keep the per-row state of the returned fit.
    let (eta, mu) = executor.materialize(&outcome.fitted);
    let irls_weights = outcome.weights.materialize(&executor, x.nrows());
    let solver_trace = executor.take_trace();
    drop(executor);

    Ok(IRLSResult {
        coefficients: outcome.coefficients,
        fitted_values: Array1::from_vec(mu),
        linear_predictor: Array1::from_vec(eta),
        deviance: outcome.deviance,
        iterations: outcome.iterations,
        converged: outcome.converged,
        covariance_unscaled: outcome.covariance_unscaled,
        irls_weights: Array1::from_vec(irls_weights),
        prior_weights: prior_weights_vec,
        offset: offset_vec,
        y: y.to_owned(),
        family_name: family.name().to_string(),
        penalty: reg_config.penalty.clone(),
        design_matrix: None, // Computed lazily in Python layer to avoid expensive copy
        warnings: outcome.warnings,
        step_halving_used: false,
        solver_status: outcome.solver_status.to_string(),
        profile: None,
        statistics: outcome.statistics,
        solver_trace,
    })
}

/// What [`drive_cd`] needs besides the rows.
pub(crate) struct CdDriveSettings<'a> {
    /// Outer-loop settings: max_iterations, tolerance, verbose and sign constraints.
    pub config: &'a IRLSConfig,
    pub regularization: &'a RegularizationConfig,
    pub init_coefficients: Option<&'a Array1<f64>>,
    /// Skip the covariance and the final statistics (cross-validation folds).
    pub skip_covariance: bool,
    /// The block-ordered mean response (`families::response_mean`): a cold start's
    /// intercept is g(clamp(mean y)).
    pub y_mean: f64,
    pub family: &'a dyn Family,
    pub link: &'a dyn Link,
    /// How the offset enters the null model (`RowContext::null_offset`).
    pub null_offset: crate::diagnostics::NullOffset,
    pub robust: &'a [HCType],
    /// Compute the final statistics even when covariance is skipped (§8).
    pub statistics_without_covariance: bool,
}

/// The result of driving coordinate descent: everything but per-row arrays.
pub(crate) struct CdDriveOutcome {
    pub coefficients: Array1<f64>,
    pub covariance_unscaled: Array2<f64>,
    pub deviance: f64,
    pub iterations: usize,
    pub converged: bool,
    pub solver_status: &'static str,
    pub warnings: Vec<String>,
    /// The predictor the returned fitted values and linear predictor come from.
    pub fitted: Predictor,
    /// The weights of the last iteration's normal equations (the covariance's).
    pub weights: RetainedWeights,
    pub statistics: Option<FinalStatistics>,
}

/// Iteratively reweighted coordinate descent over passes (§7.5).
///
/// A cold start reads the initializer's validity and the y/weight/offset facts in
/// one pass. Then each pass evaluates the deviance at the current predictor and
/// accumulates the next iteration's normal equations there, so an iteration costs
/// one pass. The sweeps run on the p×p Gram with their existing caps. The
/// covariance reuses the last iteration's Gram, and the final statistics take one
/// more pass (fused into the last one when `max_iterations` ends the loop).
pub(crate) fn drive_cd(
    executor: &mut dyn PassExecutor,
    settings: &CdDriveSettings<'_>,
) -> Result<CdDriveOutcome> {
    let config = settings.config;
    let reg_config = settings.regularization;
    let (family, link) = (settings.family, settings.link);
    let n = executor.n_rows();
    let p = executor.n_params();

    // Get penalty parameters
    let l1_penalty = reg_config.penalty.l1_penalty();
    let l2_penalty = reg_config.penalty.l2_penalty();
    let has_intercept = reg_config.fit_intercept;

    // Starting index for penalized coefficients
    let pen_start = if has_intercept { 1 } else { 0 };

    // -------------------------------------------------------------------------
    // Initialize coefficients (with warm start support)
    // -------------------------------------------------------------------------
    let mut warnings: Vec<String> = Vec::new();
    let warm_start = settings.init_coefficients.is_some();
    let mut coefficients = match settings.init_coefficients {
        Some(init) if init.len() == p => init.clone(),
        Some(init) => {
            // Dimension mismatch - fall back to cold start with warning
            warnings.push(format!(
                "Warm-start coefficient dimension mismatch (got {}, expected {}). \
                Falling back to cold start. This may indicate a bug in the caller.",
                init.len(),
                p
            ));
            Array1::zeros(p)
        }
        None => Array1::zeros(p),
    };

    // Initialize intercept to link(mean(y)) only if not warm starting
    if should_initialize_intercept_from_response(warm_start, has_intercept) {
        coefficients[0] = intercept_start_from_mean(settings.y_mean, family, link);
    }

    // μ from the coefficients on a warm start (η = g(μ)), otherwise from y with the
    // family initializer, or the safe one when any initial μ is invalid.
    let mut first_facts: Option<Facts> = None;
    let mut current = if warm_start {
        Predictor::CoefRoundTrip(std::sync::Arc::from(
            coefficients
                .as_slice()
                .expect("coefficients are contiguous"),
        ))
    } else {
        let mut request = PassRequest::default();
        let invalid = request.eval(Eval::InitializerInvalid);
        let facts = request.eval(Eval::Facts);
        let sums = executor.run(&request)?;
        first_facts = Some(sums[facts].facts().clone());
        Predictor::Initializer {
            safe: sums[invalid].count() > 0.0,
        }
    };

    // The deviance at the start, and the first iteration's normal equations.
    let max_iterations = config.max_iterations;
    let (mut deviance, mut gram) = {
        let mut request = PassRequest::default();
        let index = request.predictor(current.clone());
        let deviance = request.eval(Eval::Deviance(index));
        let gram = (max_iterations > 0).then(|| request.eval(irls_gram(index)));
        let facts = first_facts.is_none().then(|| request.eval(Eval::Facts));
        let mut sums = executor.run(&request)?;
        if let Some(eval) = facts {
            first_facts = Some(sums[eval].facts().clone());
        }
        (
            sums[deviance].deviance().0,
            gram.map(|eval| take_gram(&mut sums, eval)),
        )
    };
    let facts = FirstPassFacts::from_sums(
        first_facts
            .as_ref()
            .expect("the first pass reads the facts"),
        n,
        settings.null_offset,
    );

    // -------------------------------------------------------------------------
    // Outer IRLS loop
    // -------------------------------------------------------------------------
    let mut converged = false;
    let mut outer_iteration = 0;
    let mut retained = RetainedWeights::Zeros;
    let mut retained_xwx: Option<Vec<f64>> = None;
    let mut covariance: Option<Array2<f64>> = None;
    let mut final_sums: Option<(Vec<EvalSum>, StatisticsEvals)> = None;

    while should_continue_iteration(outer_iteration, max_iterations) {
        advance_iteration(&mut outer_iteration);

        // ---------------------------------------------------------------------
        // Coordinate descent using COVARIANCE UPDATES (glmnet-style) on the Gram
        // accumulated at the current predictor:
        //   β_j = S(X_j'W(z - X_{-j}β_{-j}), λ₁) / (X_j'WX_j + λ₂)
        // with X_j'W(z - Xβ + X_jβ_j) = grad_j + (X'WX)_{jj}β_j,
        // where grad_j = X_j'Wz - Σ_k (X'WX)_{jk}β_k.
        // ---------------------------------------------------------------------
        let system = gram
            .take()
            .expect("the pass at this iteration's predictor accumulated its Gram");
        let xwz = system.xtz;
        // Row-major X'WX: each gradient update scans a full row.
        let xwx = packed_to_row_major(&system.xtx_packed, p);
        drop(system.xtx_packed);

        let mut cd_converged = false;
        let mut cd_iteration = 0;

        // Active set: track which coefficients are non-zero for faster iterations
        let all_indices: Vec<usize> = (0..p).collect();
        let mut active_set = all_indices.clone();
        let mut use_active_set = false;

        while should_continue_iteration(cd_iteration, reg_config.max_cd_iterations) {
            advance_iteration(&mut cd_iteration);
            let mut max_change = 0.0_f64;

            // Decide which coefficients to update
            let indices_to_update: &[usize] = coordinate_indices_for_iteration(
                use_active_set,
                cd_iteration,
                &all_indices,
                &active_set,
            );

            // Update each coefficient using covariance updates
            for &j in indices_to_update {
                let old_coef = coefficients[j];

                // Compute gradient: grad_j = X_j'Wz - Σ_k (X'WX)_{jk}β_k
                let mut grad_j = xwz[j];
                for k in 0..p {
                    grad_j -= xwx[j * p + k] * coefficients[k];
                }

                // rho = grad_j + (X'WX)_{jj} * old_coef
                let xwx_jj = xwx[j * p + j];
                let rho = grad_j + xwx_jj * old_coef;

                // Update coefficient with soft-thresholding
                let new_coef =
                    coordinate_update_value(j, pen_start, rho, xwx_jj, l1_penalty, l2_penalty);

                let delta = apply_coordinate_update(&mut coefficients, j, old_coef, new_coef);
                max_change = max_change.max(delta);
            }

            // Update active set after first full pass
            if should_refresh_active_set(cd_iteration) {
                let refreshed = refresh_active_set(&coefficients, pen_start);
                active_set = refreshed.0;
                use_active_set = refreshed.1;
            }

            // Check convergence
            if coordinate_descent_converged(max_change, reg_config.cd_tolerance) {
                cd_converged = true;
                break;
            }
        }

        if !cd_converged {
            if config.verbose {
                eprintln!(
                    "Warning: Coordinate descent did not converge in {} iterations",
                    reg_config.max_cd_iterations
                );
            }
            warnings.push(format!(
                "Coordinate descent did not converge in {} iterations",
                reg_config.max_cd_iterations
            ));
        }

        // Apply coefficient sign constraints
        project_sign_constraints(
            &mut coefficients,
            &config.nonneg_indices,
            &config.nonpos_indices,
        );

        // This iteration's weights are the covariance's if it is the last.
        retained = RetainedWeights::Irls(current.clone());
        let next = coef_predictor(&coefficients);

        // The next pass: the deviance at the new coefficients, and the next
        // iteration's normal equations unless max_iterations ends the loop, in which
        // case it is the final pass and carries the final statistics too.
        let last = outer_iteration == max_iterations;
        let cov_now = (last && !settings.skip_covariance)
            .then(|| penalized_covariance(&xwx, &coefficients, pen_start));
        let mut request = PassRequest::default();
        let index = request.predictor(next.clone());
        let deviance_eval = request.eval(Eval::Deviance(index));
        let gram_eval = (!last).then(|| request.eval(irls_gram(index)));
        let stats = cov_now.as_ref().map(|bread| {
            request_statistics(
                &mut request,
                index,
                &facts,
                Some(RobustRequest {
                    types: settings.robust,
                    weights: &retained,
                    bread,
                }),
                (n, p),
            )
        });
        let mut sums = executor.run(&request)?;
        record_step(
            executor,
            SolverStep::new(
                "cd",
                "iteration",
                1,
                true,
                if cd_converged {
                    "sweeps_converged"
                } else {
                    "sweeps_exhausted"
                },
            ),
        );
        let deviance_old = deviance;
        deviance = sums[deviance_eval].deviance().0;
        gram = gram_eval.map(|eval| take_gram(&mut sums, eval));
        current = next;
        retained_xwx = Some(xwx);
        if let (Some(cov), Some(stats)) = (cov_now, stats) {
            covariance = Some(cov);
            final_sums = Some((sums, stats));
        }

        // Check outer loop convergence
        let (abs_change, rel_change) = deviance_changes(deviance_old, deviance);

        if config.verbose {
            let n_nonzero = count_nonzero_penalized(&coefficients, pen_start);
            eprintln!(
                "IRLS iter {}: deviance = {:.6}, rel_change = {:.2e}, nonzero = {}",
                outer_iteration, deviance, rel_change, n_nonzero
            );
        }

        // Converge if relative change is small OR if deviance is very small (nearly perfect fit)
        if outer_deviance_converged(rel_change, deviance, abs_change, config.tolerance) {
            converged = true;
            break;
        }
    }

    // -------------------------------------------------------------------------
    // Covariance estimate
    // -------------------------------------------------------------------------
    // IMPORTANT LIMITATION FOR ACTUARIAL USERS:
    // For penalized models (Lasso/Elastic Net), standard errors are approximate.
    // The covariance is computed using only non-zero coefficients, which does not
    // account for the selection bias introduced by penalization.
    //
    // For rigorous inference on regularized models, consider:
    // 1. Bootstrap confidence intervals
    // 2. De-biased Lasso methods
    // 3. Post-selection inference techniques
    //
    // The standard errors returned here should be used with caution for
    // hypothesis testing or confidence interval construction.
    let (covariance_unscaled, statistics) = if settings.skip_covariance {
        let statistics = if settings.statistics_without_covariance {
            let mut request = PassRequest::default();
            let index = request.predictor(current.clone());
            let stats = request_statistics(&mut request, index, &facts, None, (n, p));
            let sums = executor.run(&request)?;
            Some(finish_statistics(
                executor, family, &current, &facts, &sums, stats,
            )?)
        } else {
            None
        };
        (Array2::zeros((p, p)), statistics)
    } else {
        // X'WX of the last iteration's weights; no iteration means zero weights,
        // whose X'WX is exactly zero.
        let covariance = covariance.unwrap_or_else(|| match &retained_xwx {
            Some(xwx) => penalized_covariance(xwx, &coefficients, pen_start),
            None => penalized_covariance(&vec![0.0; p * p], &coefficients, pen_start),
        });
        let (sums, stats) = match final_sums {
            Some(final_sums) => final_sums,
            None => {
                let mut request = PassRequest::default();
                let index = request.predictor(current.clone());
                let stats = request_statistics(
                    &mut request,
                    index,
                    &facts,
                    Some(RobustRequest {
                        types: settings.robust,
                        weights: &retained,
                        bread: &covariance,
                    }),
                    (n, p),
                );
                (executor.run(&request)?, stats)
            }
        };
        let statistics = finish_statistics(executor, family, &current, &facts, &sums, stats)?;
        (covariance, Some(statistics))
    };

    let solver_status = if converged {
        "converged"
    } else {
        "max_iterations"
    };
    record_step(
        executor,
        SolverStep::new("cd", "result", outer_iteration, converged, solver_status),
    );
    Ok(CdDriveOutcome {
        coefficients,
        covariance_unscaled,
        deviance,
        iterations: outer_iteration,
        converged,
        solver_status,
        warnings,
        fitted: current,
        weights: retained,
        statistics,
    })
}

/// The ordinary IRLS Gram at a predictor: X'WX and X'Wz with W the prior times
/// IRLS weights and z the working response (`compute_xtwx_xtwz` of
/// `compute_irls_weights`).
fn irls_gram(predictor: usize) -> Eval {
    Eval::Gram {
        weights: WeightsSpec::Irls(predictor),
        response: ResponseSpec::Working(predictor),
    }
}

fn take_gram(sums: &mut [EvalSum], eval: usize) -> GramPartial {
    std::mem::replace(&mut sums[eval], EvalSum::Count(0.0)).into_gram()
}

/// The full row-major matrix of a packed upper triangle.
fn packed_to_row_major(packed: &[f64], p: usize) -> Vec<f64> {
    let mut matrix = vec![0.0_f64; p * p];
    let mut idx = 0usize;
    for j in 0..p {
        for k in j..p {
            matrix[j * p + k] = packed[idx];
            matrix[k * p + j] = packed[idx];
            idx += 1;
        }
    }
    matrix
}

fn should_initialize_intercept_from_response(has_warm_start: bool, has_intercept: bool) -> bool {
    !has_warm_start && has_intercept
}

/// A cold start's intercept: g(clamp(mean y)).
fn intercept_start_from_mean(y_mean: f64, family: &dyn Family, link: &dyn Link) -> f64 {
    link.link_at(family.clamp_mu_at(y_mean))
}

fn should_continue_iteration(iteration: usize, max_iterations: usize) -> bool {
    iteration < max_iterations
}

fn advance_iteration(iteration: &mut usize) {
    *iteration += 1;
}

fn should_use_active_subset(use_active_set: bool, cd_iteration: usize) -> bool {
    use_active_set && !cd_iteration.is_multiple_of(ACTIVE_SET_REFRESH_INTERVAL)
}

fn coordinate_indices_for_iteration<'a>(
    use_active_set: bool,
    cd_iteration: usize,
    all_indices: &'a [usize],
    active_set: &'a [usize],
) -> &'a [usize] {
    if should_use_active_subset(use_active_set, cd_iteration) {
        // Use active set (non-zero coefficients + intercept)
        active_set
    } else {
        // Full pass every ACTIVE_SET_REFRESH_INTERVAL iterations or initially
        all_indices
    }
}

fn coordinate_update_value(
    j: usize,
    pen_start: usize,
    rho: f64,
    xwx_jj: f64,
    l1_penalty: f64,
    l2_penalty: f64,
) -> f64 {
    if j < pen_start {
        rho / xwx_jj
    } else {
        let denom = xwx_jj + l2_penalty;
        if denom.abs() < ZERO_TOL {
            0.0
        } else {
            soft_threshold(rho, l1_penalty) / denom
        }
    }
}

fn apply_coordinate_update(
    coefficients: &mut Array1<f64>,
    j: usize,
    old_coef: f64,
    new_coef: f64,
) -> f64 {
    let delta = (new_coef - old_coef).abs();
    if delta > COEFFICIENT_UPDATE_TOL {
        coefficients[j] = new_coef;
    }
    delta
}

fn should_refresh_active_set(cd_iteration: usize) -> bool {
    cd_iteration == 1 || cd_iteration.is_multiple_of(ACTIVE_SET_REFRESH_INTERVAL)
}

fn active_set_from_coefficients(coefficients: &Array1<f64>, pen_start: usize) -> Vec<usize> {
    let p = coefficients.len();
    let mut active_set: Vec<usize> = (0..pen_start.min(p)).collect();
    for j in pen_start..p {
        if coefficients[j].abs() > ZERO_TOL {
            active_set.push(j);
        }
    }
    active_set
}

fn refresh_active_set(coefficients: &Array1<f64>, pen_start: usize) -> (Vec<usize>, bool) {
    let active_set = active_set_from_coefficients(coefficients, pen_start);
    let use_active_set = active_set.len() < coefficients.len();
    (active_set, use_active_set)
}

fn coordinate_descent_converged(max_change: f64, tolerance: f64) -> bool {
    max_change < tolerance
}

fn project_sign_constraints(
    coefficients: &mut Array1<f64>,
    nonneg_indices: &[usize],
    nonpos_indices: &[usize],
) {
    // Project non-negative constrained coefficients to be >= 0 (for ms(), pos())
    for &idx in nonneg_indices {
        if idx < coefficients.len() && coefficients[idx] < 0.0 {
            coefficients[idx] = 0.0;
        }
    }
    // Project non-positive constrained coefficients to be <= 0 (for neg())
    for &idx in nonpos_indices {
        if idx < coefficients.len() && coefficients[idx] > 0.0 {
            coefficients[idx] = 0.0;
        }
    }
}

fn deviance_changes(deviance_old: f64, deviance: f64) -> (f64, f64) {
    let abs_change = (deviance_old - deviance).abs();
    let rel_change = if deviance_old.abs() > ZERO_TOL {
        abs_change / deviance_old.abs()
    } else {
        abs_change
    };
    (abs_change, rel_change)
}

fn count_nonzero_penalized(coefficients: &Array1<f64>, pen_start: usize) -> usize {
    coefficients
        .iter()
        .skip(pen_start)
        .filter(|&&c| c.abs() > ZERO_TOL)
        .count()
}

fn outer_deviance_converged(
    rel_change: f64,
    deviance: f64,
    abs_change: f64,
    tolerance: f64,
) -> bool {
    rel_change < tolerance || (deviance < ZERO_TOL && abs_change < ZERO_TOL)
}

/// Compute an approximate covariance matrix for penalized estimates.
///
/// For Lasso/Elastic Net, standard errors are not well-defined in the classical sense.
/// This computes a naive estimate that can be used for rough inference.
fn penalized_covariance(xtwx: &[f64], coefficients: &Array1<f64>, pen_start: usize) -> Array2<f64> {
    let p = coefficients.len();

    let active_indices = active_set_from_coefficients(coefficients, pen_start);

    if active_indices.is_empty() {
        return Array2::zeros((p, p));
    }

    use nalgebra::DMatrix;
    let active_p = active_indices.len();
    let mut xtx_active = DMatrix::zeros(active_p, active_p);
    for (ai, &i) in active_indices.iter().enumerate() {
        for (aj, &j) in active_indices.iter().enumerate() {
            xtx_active[(ai, aj)] = xtwx[i * p + j];
        }
    }

    let mut cov = Array2::zeros((p, p));
    if let Some(inv) = xtx_active.try_inverse() {
        for (ai, &i) in active_indices.iter().enumerate() {
            for (aj, &j) in active_indices.iter().enumerate() {
                cov[[i, j]] = inv[(ai, aj)];
            }
        }
    } else {
        for &i in &active_indices {
            cov[[i, i]] = f64::NAN;
        }
    }

    cov
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::families::{GaussianFamily, PoissonFamily};
    use crate::links::{IdentityLink, Link, LogLink};
    use approx::assert_abs_diff_eq;
    use ndarray::array;
    use std::borrow::Cow;

    struct InvalidInitGaussian;

    impl Family for InvalidInitGaussian {
        fn name(&self) -> &str {
            "InvalidInitGaussian"
        }

        fn variance_at(&self, _mu: f64) -> f64 {
            1.0
        }

        fn initialize_mu_at(&self, _yi: f64, _y_mean: f64) -> f64 {
            f64::NAN
        }

        fn is_valid_mu_at(&self, mu: f64) -> bool {
            mu.is_finite()
        }

        fn clamp_mu_at(&self, mu: f64) -> f64 {
            if mu.is_finite() {
                mu
            } else {
                0.0
            }
        }

        fn unit_deviance(&self, y: &Array1<f64>, mu: &Array1<f64>) -> Array1<f64> {
            (y - mu).mapv(|value| value * value)
        }

        fn unit_deviance_at(&self, yi: f64, mui: f64) -> f64 {
            let diff = yi - mui;
            diff * diff
        }

        fn default_link(&self) -> Box<dyn Link> {
            Box::new(IdentityLink)
        }
    }

    struct ShortVarianceGaussian;

    impl Family for ShortVarianceGaussian {
        fn name(&self) -> &str {
            "ShortVarianceGaussian"
        }

        fn variance_at(&self, _mu: f64) -> f64 {
            1.0
        }

        fn initialize_mu_at(&self, yi: f64, _y_mean: f64) -> f64 {
            yi
        }

        fn is_valid_mu_at(&self, mu: f64) -> bool {
            mu.is_finite()
        }

        fn variance<'a>(&self, mu: &'a Array1<f64>) -> Cow<'a, Array1<f64>> {
            Cow::Owned(Array1::ones(mu.len().saturating_sub(1)))
        }

        fn unit_deviance(&self, y: &Array1<f64>, mu: &Array1<f64>) -> Array1<f64> {
            (y - mu).mapv(|value| value * value)
        }

        fn unit_deviance_at(&self, yi: f64, mui: f64) -> f64 {
            let diff = yi - mui;
            diff * diff
        }

        fn default_link(&self) -> Box<dyn Link> {
            Box::new(IdentityLink)
        }
    }

    #[test]
    fn custom_family_fixtures_are_self_consistent() {
        let y = array![1.0, 2.0, 4.0];
        let mu = array![1.0, 1.5, 3.0];

        let invalid = InvalidInitGaussian;
        assert_eq!(invalid.name(), "InvalidInitGaussian");
        assert_abs_diff_eq!(
            invalid.unit_deviance(&y, &mu),
            array![0.0, 0.25, 1.0],
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(invalid.unit_deviance_at(4.0, 3.0), 1.0, epsilon = 1e-12);
        assert_eq!(invalid.default_link().name(), "identity");
        assert!(invalid.initialize_mu(&y).iter().all(|v| v.is_nan()));
        assert!(!invalid.is_valid_mu(&array![1.0, f64::NAN]));
        assert_abs_diff_eq!(
            invalid.clamp_mu(&array![f64::NAN, 2.0]),
            array![0.0, 2.0],
            epsilon = 1e-12
        );

        let short = ShortVarianceGaussian;
        assert_eq!(short.name(), "ShortVarianceGaussian");
        assert_eq!(short.variance(&mu).len(), 2);
        assert_abs_diff_eq!(
            short.unit_deviance(&y, &mu),
            array![0.0, 0.25, 1.0],
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(short.unit_deviance_at(4.0, 3.0), 1.0, epsilon = 1e-12);
        assert_eq!(short.default_link().name(), "identity");
        assert_abs_diff_eq!(short.initialize_mu(&y), y, epsilon = 1e-12);
        assert!(short.is_valid_mu(&mu));
        assert_abs_diff_eq!(short.clamp_mu(&mu), mu, epsilon = 1e-12);
    }

    #[test]
    fn coordinate_descent_initialization_helpers_have_exact_contracts() {
        assert!(should_initialize_intercept_from_response(false, true));
        assert!(!should_initialize_intercept_from_response(true, true));
        assert!(!should_initialize_intercept_from_response(false, false));
        assert!(!should_initialize_intercept_from_response(true, false));

        assert_abs_diff_eq!(
            intercept_start_from_mean(3.0, &GaussianFamily, &IdentityLink),
            3.0,
            epsilon = 1e-12
        );

        // With no iteration the fit keeps its start: μ = Xβ + o on a warm start,
        // the family initializer on a cold start, or the safe initializer when any
        // initial μ is invalid.
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 1.0, -1.0, 1.0, 0.0])
            .expect("test setup should be valid");
        let y = array![1.0, 3.0, 5.0];
        let offset = array![0.5, -0.25, 0.0];
        let no_iterations = IRLSConfig {
            max_iterations: 0,
            ..IRLSConfig::default()
        };
        let lasso = RegularizationConfig::lasso(0.1);
        let start = |family: &dyn Family, init: Option<&Array1<f64>>| {
            fit_glm_coordinate_descent(
                &y,
                x.view(),
                family,
                &IdentityLink,
                &no_iterations,
                &lasso,
                Some(&offset),
                None,
                init,
                true,
                &[],
                None,
            )
            .expect("fit")
            .fitted_values
        };
        assert_abs_diff_eq!(
            start(&GaussianFamily, Some(&array![3.0, 4.0])),
            array![11.5, -1.25, 3.0],
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(start(&GaussianFamily, None), y, epsilon = 1e-12);
        assert_abs_diff_eq!(
            start(&InvalidInitGaussian, None),
            array![2.0, 3.0, 4.0],
            epsilon = 1e-12
        );
    }

    #[test]
    fn coordinate_descent_iteration_and_active_set_helpers_have_exact_edges() {
        assert!(should_continue_iteration(4, 5));
        assert!(!should_continue_iteration(5, 5));
        assert!(!should_continue_iteration(6, 5));

        let mut iteration = 2;
        advance_iteration(&mut iteration);
        assert_eq!(iteration, 3);
        let mut zero_iteration = 0;
        advance_iteration(&mut zero_iteration);
        assert_eq!(zero_iteration, 1);

        assert!(!should_use_active_subset(false, 4));
        assert!(should_use_active_subset(true, 4));
        assert!(!should_use_active_subset(true, 5));
        assert!(should_use_active_subset(true, 6));

        assert!(should_refresh_active_set(1));
        assert!(!should_refresh_active_set(4));
        assert!(should_refresh_active_set(5));
        assert!(should_refresh_active_set(10));

        let all_indices = vec![0, 1, 2, 3];
        let active_set = vec![0, 3];
        assert_eq!(
            coordinate_indices_for_iteration(true, 4, &all_indices, &active_set),
            active_set.as_slice()
        );
        assert_eq!(
            coordinate_indices_for_iteration(true, 5, &all_indices, &active_set),
            all_indices.as_slice()
        );
        assert_eq!(
            coordinate_indices_for_iteration(false, 4, &all_indices, &active_set),
            all_indices.as_slice()
        );

        let coefficients = array![10.0, ZERO_TOL, -2.0 * ZERO_TOL, 0.5];
        assert_eq!(
            active_set_from_coefficients(&coefficients, 1),
            vec![0, 2, 3]
        );
        let (refreshed, use_active_set) = refresh_active_set(&coefficients, 1);
        assert_eq!(refreshed, vec![0, 2, 3]);
        assert!(use_active_set);

        let dense = array![1.0, 2.0, -3.0];
        let (refreshed, use_active_set) = refresh_active_set(&dense, 1);
        assert_eq!(refreshed, vec![0, 1, 2]);
        assert!(!use_active_set);
    }

    #[test]
    fn coordinate_descent_update_helpers_pin_thresholds_and_penalty_math() {
        assert_abs_diff_eq!(
            coordinate_update_value(0, 1, 8.0, 2.0, 100.0, 100.0),
            4.0,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            coordinate_update_value(1, 1, 5.0, 3.0, 2.0, 1.0),
            0.75,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            coordinate_update_value(1, 1, 3.0 * ZERO_TOL, ZERO_TOL, 1.0 * ZERO_TOL, 0.0),
            2.0,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            coordinate_update_value(1, 1, 3.0, 0.5 * ZERO_TOL, 1.0, 0.0),
            0.0,
            epsilon = 1e-12
        );

        let mut coefficients = array![0.0, -2.0];
        let delta = apply_coordinate_update(&mut coefficients, 0, 0.0, COEFFICIENT_UPDATE_TOL);
        assert_abs_diff_eq!(delta, COEFFICIENT_UPDATE_TOL, epsilon = 0.0);
        assert_abs_diff_eq!(coefficients[0], 0.0, epsilon = 0.0);

        let delta = apply_coordinate_update(
            &mut coefficients,
            1,
            -2.0,
            -2.0 - 2.0 * COEFFICIENT_UPDATE_TOL,
        );
        assert!(delta > COEFFICIENT_UPDATE_TOL);
        assert_abs_diff_eq!(
            coefficients[1],
            -2.0 - 2.0 * COEFFICIENT_UPDATE_TOL,
            epsilon = 0.0
        );

        assert!(coordinate_descent_converged(0.5, 1.0));
        assert!(!coordinate_descent_converged(1.0, 1.0));
        assert!(!coordinate_descent_converged(1.5, 1.0));
    }

    #[test]
    fn coordinate_descent_projection_and_deviance_helpers_pin_exact_edges() {
        let mut coefficients = array![-2.0, -0.0, 3.0, -4.0, 5.0, -0.0];
        let neg_zero_bits = (-0.0_f64).to_bits();
        project_sign_constraints(&mut coefficients, &[0, 1, 6], &[2, 5, 6]);

        assert_abs_diff_eq!(coefficients[0], 0.0, epsilon = 0.0);
        assert_eq!(coefficients[1].to_bits(), neg_zero_bits);
        assert_abs_diff_eq!(coefficients[2], 0.0, epsilon = 0.0);
        assert_abs_diff_eq!(coefficients[3], -4.0, epsilon = 0.0);
        assert_abs_diff_eq!(coefficients[4], 5.0, epsilon = 0.0);
        assert_eq!(coefficients[5].to_bits(), neg_zero_bits);

        let (abs_change, rel_change) = deviance_changes(8.0, 6.0);
        assert_abs_diff_eq!(abs_change, 2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(rel_change, 0.25, epsilon = 1e-12);
        let (abs_change, rel_change) = deviance_changes(0.0, 2.0);
        assert_abs_diff_eq!(abs_change, 2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(rel_change, 2.0, epsilon = 1e-12);
        let (abs_change, rel_change) = deviance_changes(ZERO_TOL, 0.0);
        assert_abs_diff_eq!(abs_change, ZERO_TOL, epsilon = 0.0);
        assert_abs_diff_eq!(rel_change, ZERO_TOL, epsilon = 0.0);

        assert_eq!(
            count_nonzero_penalized(&array![99.0, ZERO_TOL, -2.0 * ZERO_TOL, 0.0], 1),
            1
        );
        assert_eq!(
            count_nonzero_penalized(&array![99.0, 2.0 * ZERO_TOL, -3.0 * ZERO_TOL, 0.5], 1),
            3
        );
        assert!(outer_deviance_converged(0.009, 10.0, 1.0, 0.01));
        assert!(!outer_deviance_converged(
            0.01,
            ZERO_TOL,
            ZERO_TOL / 2.0,
            0.01
        ));
        assert!(!outer_deviance_converged(
            0.1,
            ZERO_TOL / 2.0,
            ZERO_TOL,
            0.01
        ));
        assert!(outer_deviance_converged(
            0.1,
            ZERO_TOL / 2.0,
            ZERO_TOL / 2.0,
            0.01
        ));
        assert!(!outer_deviance_converged(
            0.1,
            ZERO_TOL / 2.0,
            2.0 * ZERO_TOL,
            0.01
        ));
        assert!(!outer_deviance_converged(
            0.1,
            2.0 * ZERO_TOL,
            ZERO_TOL / 2.0,
            0.01
        ));
    }

    #[test]
    fn penalized_covariance_helpers_pin_weights_and_active_threshold() {
        let coefficients = array![0.0, ZERO_TOL, -2.0 * ZERO_TOL, 0.0];
        assert_eq!(active_set_from_coefficients(&coefficients, 0), vec![2]);
        assert_eq!(
            active_set_from_coefficients(&coefficients, 2),
            vec![0, 1, 2]
        );
    }

    #[test]
    fn test_lasso_produces_sparse_solution() {
        // Lasso should set some coefficients to exactly zero
        let x = Array2::from_shape_vec(
            (10, 4),
            vec![
                1.0, 1.0, 0.1, 0.2, 1.0, 2.0, 0.2, 0.1, 1.0, 3.0, 0.3, 0.3, 1.0, 4.0, 0.1, 0.2,
                1.0, 5.0, 0.2, 0.1, 1.0, 6.0, 0.3, 0.2, 1.0, 7.0, 0.1, 0.3, 1.0, 8.0, 0.2, 0.1,
                1.0, 9.0, 0.3, 0.2, 1.0, 10.0, 0.1, 0.1,
            ],
        )
        .expect("test setup should be valid");

        // y strongly related to x1, weakly to x2, x3
        let y = array![5.0, 8.0, 11.0, 14.0, 17.0, 20.0, 23.0, 26.0, 29.0, 32.0];

        let family = GaussianFamily;
        let link = IdentityLink;
        let irls_config = IRLSConfig::default();

        // Strong Lasso penalty
        let reg_config = RegularizationConfig::lasso(5.0);
        let result = fit_glm_coordinate_descent(
            &y,
            x.view(),
            &family,
            &link,
            &irls_config,
            &reg_config,
            None,
            None,
            None,
            false,
            &[],
            None,
        )
        .expect("test setup should be valid");

        assert!(result.converged);

        // The weak predictors (columns 2, 3) should be shrunk toward or to zero
        // The strong predictor (column 1) should remain non-zero
        assert!(
            result.coefficients[1].abs() > 0.5,
            "Strong predictor should be non-zero"
        );

        // At least one of the weak predictors should be near zero
        let weak_coefs: Vec<f64> = vec![result.coefficients[2], result.coefficients[3]];
        let has_near_zero = weak_coefs.iter().any(|&c| c.abs() < 0.1);
        assert!(
            has_near_zero,
            "Lasso should shrink weak predictors toward zero"
        );
    }

    #[test]
    fn test_coordinate_descent_invalid_family_initializer_uses_safe_mu() {
        let x = Array2::from_shape_vec(
            (5, 2),
            vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0],
        )
        .expect("test setup should be valid");
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];
        let irls_config = IRLSConfig {
            max_iterations: 5,
            ..IRLSConfig::default()
        };
        let reg_config = RegularizationConfig::lasso(0.01);

        let result = fit_glm_coordinate_descent(
            &y,
            x.view(),
            &InvalidInitGaussian,
            &IdentityLink,
            &irls_config,
            &reg_config,
            None,
            None,
            None,
            false,
            &[],
            None,
        )
        .expect("safe initializer fallback should allow coordinate descent to run");

        assert!(result.fitted_values.iter().all(|value| value.is_finite()));
        assert!(result
            .linear_predictor
            .iter()
            .all(|value| value.is_finite()));
        assert!(result.coefficients.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn test_coordinate_descent_reads_only_per_row_family_arithmetic() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, -1.0, 1.0, 0.0, 1.0, 1.0])
            .expect("test setup should be valid");
        let y = array![0.5, 1.0, 1.5];
        let reg_config = RegularizationConfig::lasso(0.05);

        // The array helper still reports a malformed array variance...
        let err = crate::solvers::compute_irls_weights(
            &y,
            &y,
            &y,
            &Array1::zeros(3),
            &Array1::ones(3),
            &ShortVarianceGaussian,
            &IdentityLink,
            crate::constants::MIN_IRLS_WEIGHT,
        )
        .err()
        .expect("malformed family variance must be reported, not panicked");
        assert!(matches!(
            err,
            crate::error::RustyStatsError::DimensionMismatch {
                expected: 3,
                got: 2,
                ref context
            } if context == "variance length vs y length"
        ));

        // ...but coordinate descent works per row (`variance_at`), so it fits.
        let result = fit_glm_coordinate_descent(
            &y,
            x.view(),
            &ShortVarianceGaussian,
            &IdentityLink,
            &IRLSConfig::default(),
            &reg_config,
            None,
            None,
            None,
            false,
            &[],
            None,
        )
        .expect("coordinate descent reads per-row variance");
        assert!(result.coefficients.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn test_lasso_vs_unpenalized() {
        // Lasso with small lambda should give similar results to unpenalized
        let x = Array2::from_shape_vec(
            (5, 2),
            vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0],
        )
        .expect("test setup should be valid");
        let y = array![5.0, 8.0, 11.0, 14.0, 17.0];

        let family = GaussianFamily;
        let link = IdentityLink;
        let irls_config = IRLSConfig {
            max_iterations: 50, // More iterations for small penalty
            ..IRLSConfig::default()
        };

        // Very small Lasso penalty
        let reg_config = RegularizationConfig::lasso(0.001);
        let result = fit_glm_coordinate_descent(
            &y,
            x.view(),
            &family,
            &link,
            &irls_config,
            &reg_config,
            None,
            None,
            None,
            false,
            &[],
            None,
        )
        .expect("test setup should be valid");

        // With Gaussian + identity link, should converge quickly or reach good solution
        // Coefficients should be close to OLS (intercept ~2, slope ~3)
        assert!(
            (result.coefficients[0] - 2.0).abs() < 1.0,
            "Intercept: {}",
            result.coefficients[0]
        );
        assert!(
            (result.coefficients[1] - 3.0).abs() < 0.5,
            "Slope: {}",
            result.coefficients[1]
        );
    }

    #[test]
    fn test_elastic_net() {
        // Elastic Net should work (combination of L1 and L2)
        let x = Array2::from_shape_vec(
            (6, 3),
            vec![
                1.0, 1.0, 1.1, // x2 and x3 are correlated
                1.0, 2.0, 2.2, 1.0, 3.0, 3.1, 1.0, 4.0, 4.3, 1.0, 5.0, 5.2, 1.0, 6.0, 6.1,
            ],
        )
        .expect("test setup should be valid");
        let y = array![5.0, 8.0, 11.0, 14.0, 17.0, 20.0];

        let family = GaussianFamily;
        let link = IdentityLink;
        let irls_config = IRLSConfig {
            max_iterations: 50,
            ..IRLSConfig::default()
        };

        // Elastic Net: 50% L1, 50% L2
        let reg_config = RegularizationConfig::elastic_net(1.0, 0.5);
        let result = fit_glm_coordinate_descent(
            &y,
            x.view(),
            &family,
            &link,
            &irls_config,
            &reg_config,
            None,
            None,
            None,
            false,
            &[],
            None,
        )
        .expect("test setup should be valid");

        // Should produce reasonable fitted values even if not converged
        assert!(result.fitted_values.iter().all(|&x| x.is_finite()));

        // Elastic net should spread weight across correlated predictors
        // (unlike pure Lasso which often picks just one)
    }

    #[test]
    fn test_lasso_poisson() {
        // Lasso should work with Poisson family
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0],
        )
        .expect("test setup should be valid");
        let y = array![2.0, 3.0, 4.0, 6.0, 8.0, 12.0];

        let family = PoissonFamily;
        let link = LogLink;
        let irls_config = IRLSConfig::default();

        let reg_config = RegularizationConfig::lasso(0.1);
        let result = fit_glm_coordinate_descent(
            &y,
            x.view(),
            &family,
            &link,
            &irls_config,
            &reg_config,
            None,
            None,
            None,
            false,
            &[],
            None,
        )
        .expect("test setup should be valid");

        assert!(result.converged);
        assert!(result.fitted_values.iter().all(|&x| x > 0.0));
    }

    #[test]
    fn test_lasso_intercept_not_penalized() {
        // Intercept should never be zero even with strong penalty
        let x = Array2::from_shape_vec(
            (5, 2),
            vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0],
        )
        .expect("test setup should be valid");
        let y = array![10.0, 10.0, 10.0, 10.0, 10.0]; // Constant y

        let family = GaussianFamily;
        let link = IdentityLink;
        let irls_config = IRLSConfig::default();

        // Very strong Lasso penalty
        let reg_config = RegularizationConfig::lasso(100.0);
        let result = fit_glm_coordinate_descent(
            &y,
            x.view(),
            &family,
            &link,
            &irls_config,
            &reg_config,
            None,
            None,
            None,
            false,
            &[],
            None,
        )
        .expect("test setup should be valid");

        assert!(result.converged);

        // Intercept should be around 10 (mean of y)
        assert!(
            (result.coefficients[0] - 10.0).abs() < 1.0,
            "Intercept should be ~10: {}",
            result.coefficients[0]
        );

        // Slope should be zero (no relationship + strong penalty)
        assert!(
            result.coefficients[1].abs() < 0.01,
            "Slope should be ~0: {}",
            result.coefficients[1]
        );
    }

    #[test]
    fn test_lasso_covariance_uses_active_set_when_full_design_singular() {
        let x = Array2::from_shape_vec(
            (8, 3),
            vec![
                1.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 3.0, 0.0, 1.0, 4.0, 0.0, 1.0, 5.0, 0.0, 1.0,
                6.0, 0.0, 1.0, 7.0, 0.0, 1.0, 8.0, 0.0,
            ],
        )
        .expect("test setup should be valid");
        let y = array![5.0, 8.0, 11.0, 14.0, 17.0, 20.0, 23.0, 26.0];

        let family = GaussianFamily;
        let link = IdentityLink;
        let irls_config = IRLSConfig {
            max_iterations: 50,
            ..IRLSConfig::default()
        };
        let reg_config = RegularizationConfig::lasso(0.01);
        let result = fit_glm_coordinate_descent(
            &y,
            x.view(),
            &family,
            &link,
            &irls_config,
            &reg_config,
            None,
            None,
            None,
            false,
            &[],
            None,
        )
        .expect("test setup should be valid");

        assert!(result.coefficients[1].abs() > 0.5);
        assert!(result.covariance_unscaled[[0, 0]].is_finite());
        assert!(result.covariance_unscaled[[1, 1]].is_finite());
        assert!(result.covariance_unscaled[[0, 0]] > 0.0);
        assert!(result.covariance_unscaled[[1, 1]] > 0.0);
        assert_abs_diff_eq!(result.covariance_unscaled[[2, 2]], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_coordinate_descent_warm_start_offset_weights_and_skip_covariance_contracts() {
        let x = Array2::from_shape_vec(
            (6, 3),
            vec![
                1.0, 0.0, 1.0, 1.0, 1.0, 0.5, 1.0, 2.0, 1.5, 1.0, 3.0, 2.0, 1.0, 4.0, 2.5, 1.0,
                5.0, 3.0,
            ],
        )
        .expect("test setup should be valid");
        let y = array![3.0, 4.2, 5.9, 7.4, 9.1, 10.8];
        let offset = array![0.0, 0.1, -0.1, 0.2, -0.2, 0.0];
        let weights = array![1.0, 0.5, 1.5, 2.0, 0.75, 1.25];
        let init = array![1.0, 0.1, -0.1];

        let family = GaussianFamily;
        let link = IdentityLink;
        let irls_config = IRLSConfig {
            max_iterations: 25,
            ..IRLSConfig::default()
        };
        let reg_config = RegularizationConfig::elastic_net(0.05, 0.5);
        let result = fit_glm_coordinate_descent(
            &y,
            x.view(),
            &family,
            &link,
            &irls_config,
            &reg_config,
            Some(&offset),
            Some(&weights),
            Some(&init),
            true,
            &[],
            None,
        )
        .expect("test setup should be valid");

        assert_eq!(result.offset, offset);
        assert_eq!(result.prior_weights, weights);
        assert!(result.coefficients.iter().all(|v| v.is_finite()));
        assert!(result.covariance_unscaled.iter().all(|&v| v == 0.0));
        assert!(result.warnings.is_empty());
    }

    #[test]
    fn test_coordinate_descent_warm_start_mismatch_and_cd_nonconvergence_warnings() {
        let x = Array2::from_shape_vec(
            (5, 2),
            vec![1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0],
        )
        .expect("test setup should be valid");
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let bad_init = array![99.0];
        let family = GaussianFamily;
        let link = IdentityLink;
        let irls_config = IRLSConfig {
            max_iterations: 1,
            verbose: true,
            ..IRLSConfig::default()
        };
        let reg_config = RegularizationConfig {
            max_cd_iterations: 0,
            ..RegularizationConfig::lasso(0.1)
        };

        let result = fit_glm_coordinate_descent(
            &y,
            x.view(),
            &family,
            &link,
            &irls_config,
            &reg_config,
            None,
            None,
            Some(&bad_init),
            true,
            &[],
            None,
        )
        .expect("test setup should be valid");

        assert!(result
            .warnings
            .iter()
            .any(|w| w.contains("Warm-start coefficient dimension mismatch")));
        assert!(result
            .warnings
            .iter()
            .any(|w| w.contains("Coordinate descent did not converge")));
        assert_eq!(result.solver_status, "converged");
    }

    #[test]
    fn test_coordinate_descent_sign_constraints_project_coefficients() {
        let x = Array2::from_shape_vec(
            (8, 3),
            vec![
                1.0, -3.0, 2.0, 1.0, -2.0, -1.0, 1.0, -1.0, 1.0, 1.0, 0.0, -2.0, 1.0, 1.0, 3.0,
                1.0, 2.0, -3.0, 1.0, 3.0, 4.0, 1.0, 4.0, -4.0,
            ],
        )
        .expect("test setup should be valid");
        let y = array![11.0, 6.0, 6.0, 1.0, 11.0, -4.0, 17.0, -10.0];
        let family = GaussianFamily;
        let link = IdentityLink;
        let irls_config = IRLSConfig {
            max_iterations: 20,
            nonneg_indices: vec![1],
            nonpos_indices: vec![2],
            ..IRLSConfig::default()
        };
        let reg_config = RegularizationConfig::lasso(0.001);

        let result = fit_glm_coordinate_descent(
            &y,
            x.view(),
            &family,
            &link,
            &irls_config,
            &reg_config,
            None,
            None,
            None,
            true,
            &[],
            None,
        )
        .expect("test setup should be valid");

        assert!(result.coefficients[1] >= -1e-12);
        assert!(result.coefficients[2] <= 1e-12);
    }

    #[test]
    fn test_penalized_covariance_empty_and_singular_active_set_contracts() {
        // X'WX of rows (k, k, k), k = 1..4, with weights 1, 0.5, 2, 1.5: every entry 45.
        let xtwx = vec![45.0; 9];

        let inactive = array![0.0, 0.0, 0.0];
        let cov = penalized_covariance(&xtwx, &inactive, 0);
        assert_eq!(cov.dim(), (3, 3));
        assert!(cov.iter().all(|&v| v == 0.0));

        let active_singular = array![1.0, -2.0, 3.0];
        let cov = penalized_covariance(&xtwx, &active_singular, 0);
        assert!(cov[[0, 0]].is_nan());
        assert!(cov[[1, 1]].is_nan());
        assert!(cov[[2, 2]].is_nan());
    }
}
