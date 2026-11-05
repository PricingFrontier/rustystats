//! Cross-validated regularisation paths over passes (`specs/out-of-core-fit.md` §7.8).
//!
//! Both steps run on any [`PassExecutor`], so an in-memory and an out-of-core fit
//! compute the same sums in the same order:
//!
//! - [`alpha_max`]: the intercept-only fit (closed form or a width-1 IRLS), then one
//!   pass for the GLM score at it (or, for ridge, the weighted column squares).
//! - [`cv_path`]: every fold's path over the alpha grid. Fold k's driver fits on the
//!   rows outside fold k and scores the rows inside it; rows are assigned to folds
//!   by DEC-2's hash. The fold drivers run in lockstep on their own threads: each
//!   posts its pass to a rendezvous, and the calling thread merges the posted
//!   passes into one and runs it on the real executor. No fold copies the design.

use std::sync::{Condvar, Mutex};

use ndarray::Array1;

use super::irls::{drive_glm, FitConfig};
use super::passes::{
    record_step, trace_len, truncate_trace, Eval, EvalSum, FoldFilter, PassExecutor, PassRequest,
    Predictor, ScoreForm, SolverStep, ValidationForm,
};
use crate::error::{Result, RustyStatsError};
use crate::families::Family;
use crate::links::Link;
use crate::regularization::{RegularizationConfig, Standardization};

/// The smallest alpha_max a grid starts from (`constants.ALPHA_MAX_FLOOR`).
pub const ALPHA_MAX_FLOOR: f64 = 1e-4;

/// Stack size of a fold driver's thread.
const FOLD_STACK_BYTES: usize = 2 * 1024 * 1024;

/// How the intercept-only fit is solved (`regularization_path.py`
/// `_fit_null_intercept`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NullFit {
    /// β₀ = ln(Σw·y / Σw·exp(o)) when both are usable, else the IRLS fit.
    PoissonLog,
    /// β₀ = Σw·(y − o) / Σw.
    GaussianIdentity,
    /// A width-1 IRLS fit (200 iterations, tolerance 1e-12).
    Irls,
}

/// What [`alpha_max`] needs besides the rows.
pub struct AlphaMaxSettings<'a> {
    pub family: &'a dyn Family,
    pub link: &'a dyn Link,
    pub null_fit: NullFit,
    pub score_form: ScoreForm,
    pub l1_ratio: f64,
    /// Penalised columns (the intercept excluded).
    pub penalty_mask: &'a [bool],
    pub standardization: Option<&'a Standardization>,
    /// Σw·((x − c)/s)² per column, when the moments already formed it.
    pub ridge_diag: Option<&'a [f64]>,
    /// The block-ordered mean response the executor serves.
    pub y_mean: f64,
    pub has_offset: bool,
}

/// The largest alpha that zeroes every penalised coefficient (RS-ACT-005), or for
/// ridge the median weighted column norm × 10, never below [`ALPHA_MAX_FLOOR`].
pub fn alpha_max(executor: &mut dyn PassExecutor, settings: &AlphaMaxSettings<'_>) -> Result<f64> {
    if !settings.penalty_mask.iter().any(|&m| m) {
        record_step(
            executor,
            SolverStep::new("alpha_max", "alpha_max", 0, true, "floor"),
        );
        return Ok(ALPHA_MAX_FLOOR);
    }
    let p = executor.n_params();
    if settings.l1_ratio > 0.0 {
        let intercept = null_intercept(executor, settings)?;
        let mut request = PassRequest::default();
        let eval = request.eval(Eval::ScoreFactor {
            intercept,
            form: settings.score_form,
        });
        let sums = executor.run(&request)?;
        let sums = sums[eval].sums();
        let (scores, total) = (&sums[..p], sums[p]);
        let mut max_score = 0.0f64;
        for j in 0..p {
            if !settings.penalty_mask[j] {
                continue;
            }
            // Centering term included explicitly (not assuming a zero intercept score).
            let score = match settings.standardization {
                Some(s) => (scores[j] - s.center[j] * total) / s.scale[j],
                None => scores[j],
            };
            max_score = max_score.max(score.abs());
        }
        record_step(
            executor,
            SolverStep::new("alpha_max", "alpha_max", 1, true, "score"),
        );
        return Ok((max_score / settings.l1_ratio).max(ALPHA_MAX_FLOOR));
    }
    // Ridge: the median weighted column norm in the grid's coordinates.
    record_step(
        executor,
        SolverStep::new(
            "alpha_max",
            "alpha_max",
            usize::from(settings.ridge_diag.is_none()),
            true,
            if settings.ridge_diag.is_some() {
                "ridge_diag"
            } else {
                "ridge_norms"
            },
        ),
    );
    let diag: Vec<f64> = match settings.ridge_diag {
        Some(diag) => diag.to_vec(),
        None => {
            executor.set_row_standardization(settings.standardization.cloned());
            let mut request = PassRequest::default();
            let eval = request.eval(Eval::WeightedSquares);
            let sums = executor.run(&request);
            executor.set_row_standardization(None);
            sums?[eval].sums().to_vec()
        }
    };
    let mut penalised: Vec<f64> = diag
        .iter()
        .zip(settings.penalty_mask.iter())
        .filter(|(_, &m)| m)
        .map(|(&d, _)| d)
        .collect();
    if penalised.is_empty() {
        return Ok(ALPHA_MAX_FLOOR);
    }
    penalised.sort_by(f64::total_cmp);
    let m = penalised.len() / 2;
    let median = if penalised.len() % 2 == 1 {
        penalised[m]
    } else {
        (penalised[m - 1] + penalised[m]) / 2.0
    };
    Ok((median * 10.0).max(ALPHA_MAX_FLOOR))
}

/// β₀ of the intercept-only GLM η = β₀ + offset.
fn null_intercept(executor: &mut dyn PassExecutor, settings: &AlphaMaxSettings<'_>) -> Result<f64> {
    if settings.null_fit != NullFit::Irls {
        let mut request = PassRequest::default();
        let eval = request.eval(Eval::NullInterceptSums);
        let sums = executor.run(&request)?;
        let sums = sums[eval].sums();
        match settings.null_fit {
            NullFit::PoissonLog => {
                // Σw·exp(o) can overflow; then the IRLS fit is used.
                let (denom, num) = (sums[0], sums[1]);
                if denom.is_finite() && denom > 0.0 && num > 0.0 {
                    record_step(
                        executor,
                        SolverStep::new("alpha_max", "null_fit", 1, true, "closed_form"),
                    );
                    return Ok((num / denom).ln());
                }
            }
            NullFit::GaussianIdentity => {
                let sum_w = sums[2];
                if !sum_w.is_finite() || sum_w <= 0.0 {
                    return Err(RustyStatsError::InvalidValue(format!(
                        "Gaussian-identity intercept solve requires sum(weights) > 0 and \
                         finite, got {sum_w}."
                    )));
                }
                record_step(
                    executor,
                    SolverStep::new("alpha_max", "null_fit", 1, true, "closed_form"),
                );
                return Ok(sums[3] / sum_w);
            }
            NullFit::Irls => unreachable!(),
        }
    }
    let config = FitConfig {
        max_iterations: 200,
        tolerance: 1e-12,
        regularization: RegularizationConfig::none().with_intercept(true),
        ..FitConfig::default()
    };
    executor.set_intercept_only(true);
    // The width-1 IRLS is one step of this driver, not a fit of its own.
    let mark = trace_len(executor);
    let driven = drive_glm(
        executor,
        settings.family,
        settings.link,
        &config,
        None,
        settings.has_offset,
        settings.y_mean,
        false,
    );
    executor.set_intercept_only(false);
    truncate_trace(executor, mark);
    let fit = driven?.fit;
    record_step(
        executor,
        SolverStep::new(
            "alpha_max",
            "null_fit",
            fit.iterations,
            fit.converged,
            fit.solver_status.clone(),
        ),
    );
    Ok(fit.coefficients[0])
}

/// What [`cv_path`] needs besides the rows.
pub struct CvSettings<'a> {
    pub family: &'a dyn Family,
    pub link: &'a dyn Link,
    pub alphas: &'a [f64],
    pub l1_ratio: f64,
    pub folds: usize,
    pub seed: u64,
    pub max_iter: usize,
    pub tol: f64,
    pub nonneg: &'a [usize],
    pub nonpos: &'a [usize],
    pub fit_intercept: bool,
    /// The solver's standardization: penalty factors s² for a scale-only ridge,
    /// else applied to every row.
    pub standardization: Option<&'a Standardization>,
    pub has_offset: bool,
    /// Whether validation deviances are prior-weighted means.
    pub weighted: bool,
    pub validation_form: ValidationForm,
    /// How many folds may solve between passes at once (each holds 2·8p²).
    pub solve_workspaces: usize,
}

/// The CV path: per alpha, every fold's validation deviance and their mean and
/// standard error (INF when any fold failed).
#[derive(Clone, Debug)]
pub struct CvPath {
    pub alphas: Vec<f64>,
    pub mean: Vec<f64>,
    pub se: Vec<f64>,
    /// Per alpha, each fold's score.
    pub fold_scores: Vec<Vec<f64>>,
    /// Per fold, per alpha: IRLS/CD iterations and solver status ("error" on failure).
    pub iterations: Vec<Vec<usize>>,
    pub statuses: Vec<Vec<String>>,
    /// Passes the coordinator ran.
    pub passes: usize,
}

/// Every fold's path over the alpha grid, in lockstep (§7.8).
pub fn cv_path(executor: &mut dyn PassExecutor, settings: &CvSettings<'_>) -> Result<CvPath> {
    let n = executor.n_rows();
    let folds = settings.folds;
    if folds < 1 {
        return Err(RustyStatsError::InvalidValue(
            "n_folds must be at least 1".to_string(),
        ));
    }
    if folds > n {
        return Err(RustyStatsError::InvalidValue(format!(
            "n_folds ({folds}) cannot exceed the number of observations ({n})"
        )));
    }
    let filter = |fold: usize, validation: bool| FoldFilter {
        fold,
        folds,
        seed: settings.seed,
        validation,
    };
    // Each fold's training mean response (its drivers' initializers read it).
    let fold_means: Vec<f64> = {
        let mut request = PassRequest::default();
        let evals: Vec<usize> = (0..folds)
            .map(|k| request.eval_filtered(Eval::ResponseSum, Some(filter(k, false))))
            .collect();
        let sums = executor.run(&request)?;
        evals
            .iter()
            .map(|&e| {
                let s = sums[e].sums();
                if s[1] == 0.0 {
                    1.0
                } else {
                    s[0] / s[1]
                }
            })
            .collect()
    };
    let row_standardization = settings
        .standardization
        .filter(|_| !scale_only_ridge(settings));
    executor.set_row_standardization(row_standardization.cloned());

    let p = executor.n_params();
    let rendezvous = Rendezvous::new(folds, settings.solve_workspaces.max(1));
    let mut passes = 0usize;
    let mut abort: Option<RustyStatsError> = None;
    let outcomes: Vec<FoldOutcome> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..folds)
            .map(|k| {
                let rendezvous = &rendezvous;
                let fold_mean = fold_means[k];
                std::thread::Builder::new()
                    .stack_size(FOLD_STACK_BYTES)
                    .spawn_scoped(scope, move || {
                        let mut executor = FoldExecutor {
                            fold: k,
                            channel: rendezvous,
                            y_mean: fold_mean,
                            training: filter(k, false),
                            validation: filter(k, true),
                            pending: None,
                            n_rows: n,
                            n_params: p,
                        };
                        rendezvous.acquire_permit();
                        // Marks the fold finished and returns its permit even if it
                        // panics, so the coordinator never waits for it.
                        let _guard = FoldGuard {
                            rendezvous,
                            fold: k,
                        };
                        run_fold(&mut executor, settings, fold_mean)
                    })
                    .expect("a fold thread starts")
            })
            .collect();
        // The coordinator: merge every active fold's posted pass and run it.
        while let Some(posted) = rendezvous.collect() {
            let (merged, ranges) = merge(&posted);
            let result = if abort.is_some() {
                Err(RustyStatsError::PassAborted(
                    "an earlier pass failed".to_string(),
                ))
            } else {
                passes += 1;
                executor.run(&merged)
            };
            match result {
                Ok(mut sums) => {
                    let mut parts: Vec<(usize, Vec<EvalSum>)> = Vec::with_capacity(ranges.len());
                    for (fold, range) in ranges.into_iter().rev() {
                        parts.push((fold, sums.split_off(range.start)));
                    }
                    rendezvous.deliver(parts.into_iter().map(|(k, s)| (k, Ok(s))).collect());
                }
                Err(error) => {
                    let message = error.to_string();
                    if abort.is_none() {
                        abort = Some(error);
                    }
                    rendezvous.deliver(
                        posted
                            .iter()
                            .map(|(k, _)| (*k, Err(RustyStatsError::PassAborted(message.clone()))))
                            .collect(),
                    );
                }
            }
        }
        handles
            .into_iter()
            .map(|handle| handle.join().expect("a fold thread does not panic"))
            .collect()
    });
    executor.set_row_standardization(None);
    if let Some(error) = abort {
        return Err(error);
    }

    let n_alphas = settings.alphas.len();
    record_fold_steps(
        executor,
        &outcomes,
        &rendezvous_scores(&rendezvous, folds, n_alphas),
    );
    let mut path = CvPath {
        alphas: settings.alphas.to_vec(),
        mean: Vec::with_capacity(n_alphas),
        se: Vec::with_capacity(n_alphas),
        fold_scores: Vec::with_capacity(n_alphas),
        iterations: outcomes.iter().map(|o| o.iterations.clone()).collect(),
        statuses: outcomes.iter().map(|o| o.statuses.clone()).collect(),
        passes,
    };
    let fold_scores = rendezvous_scores(&rendezvous, folds, n_alphas);
    for a in 0..n_alphas {
        let scores: Vec<f64> = fold_scores.iter().map(|fold| fold[a]).collect();
        let finite = scores.len() == folds && scores.iter().all(|s| s.is_finite());
        let mean = if finite {
            scores.iter().sum::<f64>() / scores.len() as f64
        } else {
            f64::INFINITY
        };
        let se = if finite && scores.len() > 1 {
            let var =
                scores.iter().map(|&s| (s - mean).powi(2)).sum::<f64>() / (scores.len() - 1) as f64;
            (var / scores.len() as f64).sqrt()
        } else if finite {
            0.0
        } else {
            f64::INFINITY
        };
        path.mean.push(mean);
        path.se.push(se);
        path.fold_scores.push(scores);
    }
    Ok(path)
}

/// Each fold's validation score per alpha (INF where none was recorded).
fn rendezvous_scores(rendezvous: &Rendezvous, folds: usize, n_alphas: usize) -> Vec<Vec<f64>> {
    let mut fold_scores = vec![vec![f64::INFINITY; n_alphas]; folds];
    let state = rendezvous.state.lock().expect("rendezvous");
    for (fold, scores) in state.scores.iter().enumerate() {
        for &(a, score) in scores {
            fold_scores[fold][a] = score;
        }
    }
    fold_scores
}

/// The CV driver's steps (§4.5): per fold, then per alpha, the fold's fit (its
/// iterations, and its status or "error") and whether it scored finite. The folds'
/// own IRLS and CD steps are not recorded.
fn record_fold_steps(
    executor: &mut dyn PassExecutor,
    outcomes: &[FoldOutcome],
    fold_scores: &[Vec<f64>],
) {
    for (outcome, scores) in outcomes.iter().zip(fold_scores) {
        for ((&iterations, status), score) in
            outcome.iterations.iter().zip(&outcome.statuses).zip(scores)
        {
            record_step(
                executor,
                SolverStep::new("cv", "fold_alpha", iterations, score.is_finite(), status),
            );
        }
    }
}

/// A fold's mean response over its training rows (its drivers' initializers read it).
fn fold_mean(executor: &mut dyn PassExecutor, training: FoldFilter) -> Result<f64> {
    let mut request = PassRequest::default();
    let eval = request.eval_filtered(Eval::ResponseSum, Some(training));
    let sums = executor.run(&request)?;
    let s = sums[eval].sums();
    Ok(if s[1] == 0.0 { 1.0 } else { s[0] / s[1] })
}

/// One fold's path over `settings.alphas` (§7.9).
#[derive(Clone, Debug)]
pub struct FoldPath {
    /// Per alpha, the fold's validation score (INF for a failed fit).
    pub scores: Vec<f64>,
    pub iterations: Vec<usize>,
    pub statuses: Vec<String>,
    /// Passes run on the executor.
    pub passes: usize,
}

/// A fold driver's passes, run straight on the executor.
struct Direct<'e> {
    executor: &'e mut dyn PassExecutor,
    scores: Vec<(usize, f64)>,
    passes: usize,
    /// The first aborted pass: the fold stops being served and the error returns.
    aborted: Option<String>,
}

impl FoldChannel for Direct<'_> {
    fn post(&mut self, _fold: usize, request: PassRequest) -> Result<Vec<EvalSum>> {
        if let Some(message) = &self.aborted {
            return Err(RustyStatsError::PassAborted(message.clone()));
        }
        self.passes += 1;
        self.executor.run(&request).inspect_err(|error| {
            if let RustyStatsError::PassAborted(message) = error {
                self.aborted = Some(message.clone());
            }
        })
    }

    fn record_score(&mut self, _fold: usize, alpha: usize, score: f64) {
        self.scores.push((alpha, score));
    }
}

/// Fold `fold`'s path alone, its passes straight on `executor` (§7.9): the fold's
/// warm-started fits over `settings.alphas` on its training rows, each scored on
/// its validation rows, with [`cv_path`]'s per-fold semantics (a failed fit scores
/// INF and leaves the warm start unchanged). `executor` serves the fold's own
/// design for every row. An aborted pass is returned as the error, never scored.
pub fn fold_path(
    executor: &mut dyn PassExecutor,
    settings: &CvSettings<'_>,
    fold: usize,
) -> Result<FoldPath> {
    let filter = |validation: bool| FoldFilter {
        fold,
        folds: settings.folds,
        seed: settings.seed,
        validation,
    };
    let y_mean = fold_mean(executor, filter(false))?;
    let row_standardization = settings
        .standardization
        .filter(|_| !scale_only_ridge(settings));
    executor.set_row_standardization(row_standardization.cloned());
    let (n_rows, n_params) = (executor.n_rows(), executor.n_params());
    let mut fold_executor = FoldExecutor {
        fold,
        channel: Direct {
            executor: &mut *executor,
            scores: Vec::new(),
            passes: 0,
            aborted: None,
        },
        y_mean,
        training: filter(false),
        validation: filter(true),
        pending: None,
        n_rows,
        n_params,
    };
    let outcome = run_fold(&mut fold_executor, settings, y_mean);
    let channel = fold_executor.channel;
    let (scores, passes, aborted) = (channel.scores, channel.passes, channel.aborted);
    executor.set_row_standardization(None);
    if let Some(message) = aborted {
        return Err(RustyStatsError::PassAborted(message));
    }
    let mut fold_scores = vec![f64::INFINITY; settings.alphas.len()];
    for (alpha, score) in scores {
        fold_scores[alpha] = score;
    }
    record_fold_steps(
        executor,
        std::slice::from_ref(&outcome),
        std::slice::from_ref(&fold_scores),
    );
    Ok(FoldPath {
        scores: fold_scores,
        iterations: outcome.iterations,
        statuses: outcome.statuses,
        passes: passes + 1,
    })
}

/// An executor restricted to a fold's training rows: every evaluation filtered to
/// them and the initializers reading the fold's mean response.
struct TrainingRows<'e> {
    executor: &'e mut dyn PassExecutor,
    training: FoldFilter,
    y_mean: f64,
}

impl PassExecutor for TrainingRows<'_> {
    fn n_rows(&self) -> usize {
        self.executor.n_rows()
    }

    fn n_params(&self) -> usize {
        self.executor.n_params()
    }

    fn run(&mut self, request: &PassRequest) -> Result<Vec<EvalSum>> {
        self.executor
            .run(&localize(request, self.y_mean, self.training))
    }

    fn set_row_standardization(&mut self, standardization: Option<Standardization>) {
        self.executor.set_row_standardization(standardization);
    }

    fn set_intercept_only(&mut self, intercept_only: bool) {
        self.executor.set_intercept_only(intercept_only);
    }

    fn trace(&mut self) -> Option<&mut Vec<SolverStep>> {
        self.executor.trace()
    }
}

/// [`alpha_max`] over a fold's training rows (§7.9), `executor` serving the fold's
/// design. `settings.y_mean` is replaced by the fold's training mean response.
pub fn fold_alpha_max(
    executor: &mut dyn PassExecutor,
    settings: &AlphaMaxSettings<'_>,
    training: FoldFilter,
) -> Result<f64> {
    let y_mean = fold_mean(executor, training)?;
    let mut rows = TrainingRows {
        executor,
        training,
        y_mean,
    };
    alpha_max(
        &mut rows,
        &AlphaMaxSettings {
            family: settings.family,
            link: settings.link,
            null_fit: settings.null_fit,
            score_form: settings.score_form,
            l1_ratio: settings.l1_ratio,
            penalty_mask: settings.penalty_mask,
            standardization: settings.standardization,
            ridge_diag: settings.ridge_diag,
            y_mean,
            has_offset: settings.has_offset,
        },
    )
}

/// Marks a fold finished and returns its solve permit when its thread ends.
struct FoldGuard<'r> {
    rendezvous: &'r Rendezvous,
    fold: usize,
}

impl Drop for FoldGuard<'_> {
    fn drop(&mut self) {
        self.rendezvous.release_permit();
        self.rendezvous.finish(self.fold);
    }
}

/// The scale-only ridge keeps the raw rows and penalises with s².
fn scale_only_ridge(settings: &CvSettings<'_>) -> bool {
    settings.l1_ratio <= 0.0
        && settings
            .standardization
            .is_some_and(|std| std.center.iter().all(|&c| c == 0.0))
}

/// One fold's fits (its scores are recorded in the rendezvous).
struct FoldOutcome {
    iterations: Vec<usize>,
    statuses: Vec<String>,
}

/// Fold k's path: a warm-started fit per alpha on its training rows, each scored on
/// its validation rows (`fit_cv_path_py`'s semantics: a failed fit scores INF and
/// leaves the warm start unchanged).
fn run_fold<C: FoldChannel>(
    executor: &mut FoldExecutor<C>,
    settings: &CvSettings<'_>,
    y_mean: f64,
) -> FoldOutcome {
    let n_alphas = settings.alphas.len();
    let mut outcome = FoldOutcome {
        iterations: Vec::with_capacity(n_alphas),
        statuses: Vec::with_capacity(n_alphas),
    };
    let scale_only = scale_only_ridge(settings);
    let mut warm: Option<Array1<f64>> = None;
    for (a, &alpha) in settings.alphas.iter().enumerate() {
        let regularization = if alpha > 0.0 {
            if settings.l1_ratio >= 1.0 {
                RegularizationConfig::lasso(alpha)
            } else if settings.l1_ratio <= 0.0 {
                RegularizationConfig::ridge(alpha)
            } else {
                RegularizationConfig::elastic_net(alpha, settings.l1_ratio)
            }
        } else {
            RegularizationConfig::none()
        }
        .with_intercept(settings.fit_intercept);
        let config = FitConfig {
            max_iterations: settings.max_iter,
            tolerance: settings.tol,
            min_weight: 1e-10,
            verbose: false,
            nonneg_indices: settings.nonneg.to_vec(),
            nonpos_indices: settings.nonpos.to_vec(),
            regularization,
            skip_covariance: true,
            standardization: if scale_only {
                settings.standardization.cloned()
            } else {
                None
            },
            robust: Vec::new(),
            // The fold executor serves the rows; its kernel is already chosen.
            row_kernel: Default::default(),
        };
        let driven = drive_glm(
            executor,
            settings.family,
            settings.link,
            &config,
            warm.as_ref(),
            settings.has_offset,
            y_mean,
            false,
        );
        let fit = match driven {
            Ok(driven) => driven.fit,
            Err(_) => {
                outcome.iterations.push(0);
                outcome.statuses.push("error".to_string());
                continue;
            }
        };
        outcome.iterations.push(fit.iterations);
        outcome.statuses.push(fit.solver_status.clone());
        warm = Some(fit.coefficients.clone());
        let validation_beta = match settings.standardization {
            Some(std) if !scale_only => {
                match std.to_original_coefficients(&fit.coefficients, settings.fit_intercept) {
                    Ok(beta) => beta,
                    Err(_) => continue,
                }
            }
            _ => fit.coefficients,
        };
        // A validation still queued (a later fit failed before any pass) goes first.
        if executor.pending.is_some() {
            let _ = executor.run(&PassRequest::default());
        }
        // The validation rides on this fold's next pass (the next alpha's start).
        executor.pending = Some((
            a,
            Eval::ValidationDeviance {
                beta: validation_beta.iter().copied().collect(),
                form: settings.validation_form,
                weighted: settings.weighted,
            },
        ));
    }
    if executor.pending.is_some() {
        let _ = executor.run(&PassRequest::default());
    }
    outcome
}

/// `request` for a fold's training rows: every evaluation filtered to them, and the
/// initializers reading the fold's mean response.
fn localize(request: &PassRequest, y_mean: f64, training: FoldFilter) -> PassRequest {
    let mut local = PassRequest::default();
    let map: Vec<usize> = request
        .predictors
        .iter()
        .map(|predictor| {
            local.predictor(match predictor {
                Predictor::Initializer { safe } => Predictor::InitializerAt {
                    safe: *safe,
                    y_mean,
                },
                other => other.clone(),
            })
        })
        .collect();
    for eval in &request.evals {
        let eval = match eval {
            Eval::InitializerInvalid => Eval::InitializerInvalidAt(y_mean),
            other => other.remap(&map),
        };
        local.eval_filtered(eval, Some(training));
    }
    local
}

/// Where a fold driver's passes go: the lockstep rendezvous, or straight to an
/// executor when the fold runs alone (§7.9).
trait FoldChannel {
    fn post(&mut self, fold: usize, request: PassRequest) -> Result<Vec<EvalSum>>;
    fn record_score(&mut self, fold: usize, alpha: usize, score: f64);
}

impl FoldChannel for &Rendezvous {
    fn post(&mut self, fold: usize, request: PassRequest) -> Result<Vec<EvalSum>> {
        Rendezvous::post(self, fold, request)
    }

    fn record_score(&mut self, fold: usize, alpha: usize, score: f64) {
        Rendezvous::record_score(self, fold, alpha, score);
    }
}

/// A fold's view of the executor: its passes go through its channel, every
/// evaluation filtered to the fold's training rows, its initializers reading the
/// fold's mean response, and its pending validation evaluation riding along.
struct FoldExecutor<C: FoldChannel> {
    fold: usize,
    channel: C,
    y_mean: f64,
    training: FoldFilter,
    validation: FoldFilter,
    /// (alpha index, validation evaluation) to add to the next pass.
    pending: Option<(usize, Eval)>,
    n_rows: usize,
    n_params: usize,
}

impl<C: FoldChannel> PassExecutor for FoldExecutor<C> {
    fn n_rows(&self) -> usize {
        self.n_rows
    }

    fn n_params(&self) -> usize {
        self.n_params
    }

    fn run(&mut self, request: &PassRequest) -> Result<Vec<EvalSum>> {
        let mut local = localize(request, self.y_mean, self.training);
        let validation = self.pending.take().map(|(a, eval)| {
            let index = local.eval_filtered(eval, Some(self.validation));
            (a, index)
        });
        let mut sums = self.channel.post(self.fold, local)?;
        if let Some((a, index)) = validation {
            let s = sums[index].sums();
            let score = if s[1] <= 0.0 || !s[1].is_finite() {
                f64::INFINITY
            } else {
                s[0] / s[1]
            };
            self.channel.record_score(self.fold, a, score);
            sums.truncate(index);
        }
        Ok(sums)
    }

    fn set_row_standardization(&mut self, _standardization: Option<Standardization>) {
        // Every fold shares the coordinator's row standardization.
    }

    fn set_intercept_only(&mut self, _intercept_only: bool) {}
}

/// The coordination point of the fold drivers and the calling thread.
struct Rendezvous {
    state: Mutex<RendezvousState>,
    posted_cv: Condvar,
    delivered_cv: Condvar,
    permits: Mutex<usize>,
    permit_cv: Condvar,
}

struct RendezvousState {
    posted: Vec<Option<PassRequest>>,
    results: Vec<Option<Result<Vec<EvalSum>>>>,
    finished: Vec<bool>,
    scores: Vec<Vec<(usize, f64)>>,
}

impl Rendezvous {
    fn new(folds: usize, permits: usize) -> Self {
        Self {
            state: Mutex::new(RendezvousState {
                posted: (0..folds).map(|_| None).collect(),
                results: (0..folds).map(|_| None).collect(),
                finished: vec![false; folds],
                scores: vec![Vec::new(); folds],
            }),
            posted_cv: Condvar::new(),
            delivered_cv: Condvar::new(),
            permits: Mutex::new(permits),
            permit_cv: Condvar::new(),
        }
    }

    fn acquire_permit(&self) {
        let mut permits = self.permits.lock().expect("permits");
        while *permits == 0 {
            permits = self.permit_cv.wait(permits).expect("permits");
        }
        *permits -= 1;
    }

    fn release_permit(&self) {
        *self.permits.lock().expect("permits") += 1;
        self.permit_cv.notify_one();
    }

    /// Post fold k's pass and wait for its sums. The fold's solve permit is given
    /// up while it waits.
    fn post(&self, fold: usize, request: PassRequest) -> Result<Vec<EvalSum>> {
        self.release_permit();
        let result = {
            let mut state = self.state.lock().expect("rendezvous");
            state.posted[fold] = Some(request);
            self.posted_cv.notify_all();
            loop {
                if let Some(result) = state.results[fold].take() {
                    break result;
                }
                state = self.delivered_cv.wait(state).expect("rendezvous");
            }
        };
        self.acquire_permit();
        result
    }

    fn record_score(&self, fold: usize, alpha: usize, score: f64) {
        self.state.lock().expect("rendezvous").scores[fold].push((alpha, score));
    }

    fn finish(&self, fold: usize) {
        self.state.lock().expect("rendezvous").finished[fold] = true;
        self.posted_cv.notify_all();
    }

    /// Wait until every active fold has posted or finished; take the posted passes
    /// (None once every fold has finished).
    fn collect(&self) -> Option<Vec<(usize, PassRequest)>> {
        let mut state = self.state.lock().expect("rendezvous");
        loop {
            let ready =
                (0..state.posted.len()).all(|k| state.finished[k] || state.posted[k].is_some());
            if ready {
                break;
            }
            state = self.posted_cv.wait(state).expect("rendezvous");
        }
        let posted: Vec<(usize, PassRequest)> = state
            .posted
            .iter_mut()
            .enumerate()
            .filter_map(|(k, request)| request.take().map(|r| (k, r)))
            .collect();
        (!posted.is_empty()).then_some(posted)
    }

    fn deliver(&self, results: Vec<(usize, Result<Vec<EvalSum>>)>) {
        let mut state = self.state.lock().expect("rendezvous");
        for (fold, result) in results {
            state.results[fold] = Some(result);
        }
        self.delivered_cv.notify_all();
    }
}

/// One request holding every fold's evaluations; each fold's evaluations occupy a
/// contiguous range, in fold order.
fn merge(posted: &[(usize, PassRequest)]) -> (PassRequest, Vec<(usize, std::ops::Range<usize>)>) {
    let mut merged = PassRequest::default();
    let mut ranges = Vec::with_capacity(posted.len());
    for (fold, request) in posted {
        let map: Vec<usize> = request
            .predictors
            .iter()
            .map(|predictor| merged.predictor(predictor.clone()))
            .collect();
        let start = merged.evals.len();
        for (eval, filter) in request.evals.iter().zip(request.filters.iter()) {
            merged.eval_filtered(eval.remap(&map), *filter);
        }
        ranges.push((*fold, start..merged.evals.len()));
    }
    (merged, ranges)
}
