//! Negative Binomial θ estimation over passes (`specs/out-of-core-fit.md` §7.10).
//!
//! The steps of `fit_negbinomial_py`, on any [`PassExecutor`]:
//!
//! 1. Without an initial θ, a cold Poisson fit, then the moment θ₀ from one pass at
//!    its fitted state.
//! 2. Rounds (at most `max_theta_iter`): an NB fit at the current θ (the first a
//!    cold start, later ones warm-started), μ kept by the executor (in RAM, or a
//!    spill file out of core), then Brent's profile search over θ in [0.01, 1000]
//!    with tolerance 1e-6, one θ pass per evaluation. Rounds stop when θ moves by
//!    less than `theta_tol`.
//! 3. The final NB fit at θ, warm-started.

use ndarray::Array1;

use super::irls::{fit_on_rows, ExecutorFit, FitConfig, IRLSResult, RetainedWeights};
use super::passes::{
    record_step, Eval, InMemoryExecutor, PassExecutor, PassRequest, Predictor, RowContext,
    SolverStep,
};
use super::validate_glm_inputs;
use crate::error::{Result, RustyStatsError};
use crate::families::{NegativeBinomialFamily, PoissonFamily};
use crate::links::Link;
use crate::regularization::RegularizationConfig;

/// What [`fit_negbinomial_with_executor`] needs besides the rows.
pub struct NbThetaSettings<'a> {
    pub link: &'a dyn Link,
    /// The rounds' fits (no robust covariances).
    pub config: &'a FitConfig,
    /// The final fit's configuration.
    pub final_config: &'a FitConfig,
    pub init_theta: Option<f64>,
    pub theta_tol: f64,
    pub max_theta_iter: usize,
    pub has_offset: bool,
    /// Whether the θ profile weights each row by its prior weight.
    pub weighted: bool,
    /// The block-ordered mean response.
    pub y_mean: f64,
}

/// The final fit and how θ was found.
pub struct NbThetaFit {
    pub fit: ExecutorFit,
    pub theta: f64,
    pub init_theta: f64,
    pub iterations: usize,
    pub converged: bool,
    pub(crate) fitted: Predictor,
    pub(crate) weights: RetainedWeights,
}

fn nb_family(theta: f64) -> Result<NegativeBinomialFamily> {
    NegativeBinomialFamily::new(theta)
        .map_err(|error| RustyStatsError::InvalidValue(format!("Invalid NB theta: {error}")))
}

/// Estimate θ by profile likelihood and fit at it (§7.10).
pub fn fit_negbinomial_with_executor(
    executor: &mut dyn PassExecutor,
    settings: &NbThetaSettings<'_>,
) -> Result<NbThetaFit> {
    let link = settings.link;
    let mut theta = match settings.init_theta {
        Some(theta) if theta <= 0.0 => {
            return Err(RustyStatsError::InvalidValue(format!(
                "init_theta must be > 0, got {theta}"
            )))
        }
        Some(theta) => {
            record_step(
                executor,
                SolverStep::new("nb_theta", "initial_theta", 0, true, "given"),
            );
            theta
        }
        None => {
            let init_config = FitConfig {
                regularization: RegularizationConfig::none(),
                ..settings.config.clone()
            };
            executor.set_family(Box::new(PoissonFamily));
            let poisson = fit_on_rows(
                executor,
                &PoissonFamily,
                link,
                &init_config,
                None,
                settings.has_offset,
                settings.y_mean,
            );
            let moments = poisson.and_then(|driven| {
                let mut request = PassRequest::default();
                let predictor = request.predictor(driven.fitted);
                let eval = request.eval(Eval::NbMoments {
                    predictor,
                    y_mean: settings.y_mean,
                });
                Ok(executor.run(&request)?[eval].sums().to_vec())
            });
            executor.set_row_standardization(None);
            // An aborted pass is the caller's exception, never a solver failure: it
            // crosses this boundary unchanged (§7.1).
            let sums = moments.map_err(|error| match error {
                RustyStatsError::PassAborted(_) => error,
                error => {
                    RustyStatsError::InvalidValue(format!("Initial Poisson fit failed: {error}"))
                }
            })?;
            record_step(
                executor,
                SolverStep::new("nb_theta", "initial_theta", 1, true, "moments"),
            );
            crate::diagnostics::theta_moments_from_sums(sums[0], sums[1], sums[2])
        }
    };
    let init_theta = theta;

    let mut coefficients: Option<Array1<f64>> = None;
    let mut iterations = 0;
    let mut converged = false;
    for _ in 0..settings.max_theta_iter {
        iterations += 1;
        let family = nb_family(theta)?;
        executor.set_family(Box::new(nb_family(theta)?));
        let round = fit_on_rows(
            executor,
            &family,
            link,
            settings.config,
            coefficients.as_ref(),
            settings.has_offset,
            settings.y_mean,
        )
        .and_then(|driven| {
            executor.spill_mu(&driven.fitted)?;
            Ok(driven)
        });
        executor.set_row_standardization(None);
        let round = round?;
        coefficients = Some(round.fit.coefficients);
        let weighted = settings.weighted;
        let mut evaluations = 0usize;
        let mut objective = |t: f64| {
            evaluations += 1;
            executor.spilled_nb_loglik(t, weighted).map(|ll| -ll)
        };
        let new_theta =
            crate::diagnostics::estimate_theta_with(&mut objective, 0.01, 1000.0, 1e-6)?;
        let settled = (new_theta - theta).abs() < settings.theta_tol;
        record_step(
            executor,
            SolverStep::new(
                "nb_theta",
                "round",
                evaluations,
                settled,
                if settled { "converged" } else { "continue" },
            ),
        );
        if settled {
            theta = new_theta;
            converged = true;
            break;
        }
        theta = new_theta;
    }

    let final_family = nb_family(theta)?;
    executor.set_family(Box::new(nb_family(theta)?));
    let driven = fit_on_rows(
        executor,
        &final_family,
        link,
        settings.final_config,
        coefficients.as_ref(),
        settings.has_offset,
        settings.y_mean,
    );
    executor.set_row_standardization(None);
    let driven = driven?;
    Ok(NbThetaFit {
        fit: driven.fit,
        theta,
        init_theta,
        iterations,
        converged,
        fitted: driven.fitted,
        weights: driven.weights,
    })
}

/// [`fit_negbinomial_with_executor`] on an in-memory design, with the per-row
/// arrays an in-memory result keeps (μ is held in RAM between rounds).
#[allow(clippy::too_many_arguments)]
pub fn fit_negbinomial_in_memory(
    y: &Array1<f64>,
    x: ndarray::ArrayView2<'_, f64>,
    link: &dyn Link,
    config: &FitConfig,
    final_config: &FitConfig,
    offset: Option<&Array1<f64>>,
    weights: Option<&Array1<f64>>,
    init_theta: Option<f64>,
    theta_tol: f64,
    max_theta_iter: usize,
) -> Result<(IRLSResult, NbThetaFit)> {
    let validated = validate_glm_inputs(y, x, offset, weights)?;
    let y_slice = y.as_slice().ok_or_else(|| {
        RustyStatsError::LinearAlgebraError("Response y must be contiguous".to_string())
    })?;
    let y_mean = crate::families::response_mean(y);
    let sparse_rows = config.row_kernel.sparse_in_memory(x);
    let ctx = RowContext {
        family: &PoissonFamily,
        link,
        y: y_slice,
        offset: validated
            .offset
            .as_slice()
            .expect("validated offset is contiguous"),
        prior_weights: validated
            .prior_weights
            .as_slice()
            .expect("validated prior weights are contiguous"),
        y_mean,
        min_weight: config.min_weight,
        has_offset: offset.is_some(),
        standardization: None,
    };
    let mut executor = InMemoryExecutor::new(x, sparse_rows, ctx)?;
    let fitted = fit_negbinomial_with_executor(
        &mut executor,
        &NbThetaSettings {
            link,
            config,
            final_config,
            init_theta,
            theta_tol,
            max_theta_iter,
            has_offset: offset.is_some(),
            weighted: weights.is_some(),
            y_mean,
        },
    )?;
    // The final fit's rows: its standardization (when centred) and family.
    executor.set_family(Box::new(nb_family(fitted.theta)?));
    executor.set_row_standardization(
        super::irls::row_standardization(final_config, x.ncols())?.cloned(),
    );
    let (eta, mu) = executor.materialize(&fitted.fitted);
    let irls_weights = fitted.weights.materialize(&executor, x.nrows());
    executor.set_row_standardization(None);
    let solver_trace = executor.take_trace();
    let fit = &fitted.fit;
    let result = IRLSResult {
        coefficients: fit.coefficients.clone(),
        fitted_values: Array1::from_vec(mu),
        linear_predictor: Array1::from_vec(eta),
        deviance: fit.deviance,
        iterations: fit.iterations,
        converged: fit.converged,
        covariance_unscaled: fit.covariance_unscaled.clone(),
        irls_weights: Array1::from_vec(irls_weights),
        prior_weights: validated.prior_weights,
        offset: validated.offset,
        y: y.to_owned(),
        family_name: format!("NegativeBinomial(theta={:.4})", fitted.theta),
        penalty: fit.penalty.clone(),
        design_matrix: None,
        warnings: fit.warnings.clone(),
        step_halving_used: fit.step_halving_used,
        solver_status: fit.solver_status.clone(),
        profile: None,
        statistics: fit.statistics.clone(),
        solver_trace,
    };
    Ok((result, fitted))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::families::Family;
    use crate::links::LogLink;
    use crate::regularization::Standardization;
    use crate::solvers::passes::EvalSum;
    use ndarray::Array2;

    /// An in-memory executor whose `abort_at`-th pass is aborted, as a batched one's is
    /// when the caller's callback raises.
    struct Aborting<'a> {
        inner: InMemoryExecutor<'a>,
        runs: usize,
        abort_at: usize,
    }

    impl PassExecutor for Aborting<'_> {
        fn n_rows(&self) -> usize {
            self.inner.n_rows()
        }
        fn n_params(&self) -> usize {
            self.inner.n_params()
        }
        fn run(&mut self, request: &PassRequest) -> Result<Vec<EvalSum>> {
            self.runs += 1;
            if self.runs == self.abort_at {
                return Err(RustyStatsError::PassAborted("sentinel".to_string()));
            }
            self.inner.run(request)
        }
        fn set_row_standardization(&mut self, standardization: Option<Standardization>) {
            self.inner.set_row_standardization(standardization);
        }
        fn set_intercept_only(&mut self, intercept_only: bool) {
            self.inner.set_intercept_only(intercept_only);
        }
        fn set_family(&mut self, family: Box<dyn Family>) {
            self.inner.set_family(family);
        }
        fn spill_mu(&mut self, predictor: &Predictor) -> Result<()> {
            self.inner.spill_mu(predictor)
        }
        fn spilled_nb_loglik(&mut self, theta: f64, weighted: bool) -> Result<f64> {
            self.inner.spilled_nb_loglik(theta, weighted)
        }
    }

    /// §7.1: an aborted pass in the initial Poisson fit or its moments pass reaches
    /// the caller as the abort, never as "Initial Poisson fit failed".
    #[test]
    fn an_aborted_initial_poisson_pass_stays_an_abort() {
        let n = 300;
        let x = Array2::from_shape_fn((n, 2), |(i, j)| {
            if j == 0 {
                1.0
            } else {
                ((i * 37) % 101) as f64 / 101.0 - 0.5
            }
        });
        let y = Array1::from_shape_fn(n, |i| ((i * 13) % 7) as f64);
        let offset = Array1::zeros(n);
        let weights = Array1::ones(n);
        let config = FitConfig::default();
        let settings = NbThetaSettings {
            link: &LogLink,
            config: &config,
            final_config: &config,
            init_theta: None,
            theta_tol: 1e-5,
            max_theta_iter: 3,
            has_offset: false,
            weighted: false,
            y_mean: crate::families::response_mean(&y),
        };
        let executor = |abort_at| Aborting {
            inner: InMemoryExecutor::new(
                x.view(),
                false,
                RowContext {
                    family: &PoissonFamily,
                    link: &LogLink,
                    y: y.as_slice().expect("contiguous"),
                    offset: offset.as_slice().expect("contiguous"),
                    prior_weights: weights.as_slice().expect("contiguous"),
                    y_mean: settings.y_mean,
                    min_weight: config.min_weight,
                    has_offset: false,
                    standardization: None,
                },
            )
            .expect("executor"),
            runs: 0,
            abort_at,
        };
        let mut counting = executor(usize::MAX);
        fit_negbinomial_with_executor(&mut counting, &settings).expect("the fit runs");
        // The initial Poisson fit's passes, then the moments pass.
        let mut probe = executor(usize::MAX);
        fit_on_rows(
            &mut probe,
            &PoissonFamily,
            &LogLink,
            &config,
            None,
            false,
            settings.y_mean,
        )
        .expect("the initial fit runs");
        let through_moments = probe.runs + 1;
        assert!(through_moments >= 3 && through_moments < counting.runs);
        for abort_at in 1..=through_moments {
            let mut aborting = executor(abort_at);
            match fit_negbinomial_with_executor(&mut aborting, &settings) {
                Err(RustyStatsError::PassAborted(message)) => {
                    assert_eq!(message, "sentinel", "pass {abort_at}");
                }
                Err(other) => panic!("pass {abort_at}: {other}"),
                Ok(_) => panic!("pass {abort_at}: the abort was lost"),
            }
            assert_eq!(aborting.runs, abort_at, "no pass after the abort");
        }
    }
}
