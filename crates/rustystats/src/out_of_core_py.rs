//! Out-of-core pass executor (`specs/out-of-core-fit.md` §7.1).
//!
//! The driver asks for passes; this executor serves each one by calling back into
//! Python, which opens the source, encodes each slice into a block and yields it.
//! Rows are reduced with the GIL released, through the same `process_rows` and
//! block reducer as an in-memory fit, so the two give identical bits.
//!
//! A Python exception raised while a pass is being served (by the source, the
//! encoder or `on_batch`) is kept, cancels every later pass (latched), and is
//! re-raised unchanged once the fit returns to Python.

use ndarray::Array1;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use rustystats_core::error::{Result as CoreResult, RustyStatsError};
use rustystats_core::families::Family;
use rustystats_core::links::Link;
use rustystats_core::regularization::{RegularizationConfig, Standardization};
use rustystats_core::rows::BlockReducer;
use rustystats_core::solvers::passes::{
    process_block, rows_stay_sparse, sparse_row_offsets, EvalSum, PassExecutor, PassPartial,
    PassRequest, Predictor, RowContext, RowKernel, RowRef, SolverStep,
};
use rustystats_core::solvers::{fit_glm_with_executor, FitConfig};

use crate::families_py::{
    default_link_name, family_from_name_with_tweedie_support, link_from_name,
    resolve_negbinomial_theta, resolve_tweedie_var_power,
};
use crate::fitting_py::build_standardization;
use crate::results_py::{parse_robust_types, PyGLMResults};

/// Serves passes from encoded blocks that a Python callable yields.
struct PyBatchExecutor<'py> {
    py: Python<'py>,
    /// `open_pass(kind)` returns an iterable of `(x, y, weights, offset)` blocks
    /// covering every row in order; `x` is C-ordered float64 of shape (rows, p).
    /// `kind` names the pass (`PassRequest::kind`, §4.4).
    open_pass: Bound<'py, PyAny>,
    n_rows: usize,
    n_params: usize,
    family: Box<dyn Family>,
    link: Box<dyn Link>,
    y_mean: f64,
    min_weight: f64,
    has_offset: bool,
    /// The first Python exception raised while serving a pass (latched).
    aborted: Option<PyErr>,
    /// Rows are standardized before the kernels when set (§6.7).
    standardization: Option<Standardization>,
    /// Serve every row as the single column [1.0] (`set_intercept_only`).
    intercept_only: bool,
    /// `spill(block)` appends a block of μ to the θ spill, `spill(None)` starts it
    /// over (§10).
    spill_sink: Option<Bound<'py, PyAny>>,
    /// `open_theta()` yields `(y, weights, mu)` blocks of every row in order, μ read
    /// back from the spill.
    open_theta: Option<Bound<'py, PyAny>>,
    /// The drivers' steps (§4.5).
    trace: Vec<SolverStep>,
    /// The packed upper-triangle offsets when rows are served as sparse rows, formed
    /// block by block (§5.2; `serving`); None serves dense rows.
    sparse_offsets: Option<Vec<usize>>,
    /// The pool the row reductions run in (`threads=`); None is the global pool.
    pool: Option<rayon::ThreadPool>,
}

/// The row kernel a binding's `max_row_nonzeros=` and `row_kernel=` ask for (§5.2):
/// chosen from the structural bound when `row_kernel` is None; `"dense"` or
/// `"sparse"` give it explicitly (tests and benchmarks).
pub(crate) fn parse_row_kernel(
    max_row_nonzeros: Option<usize>,
    row_kernel: Option<&str>,
) -> PyResult<RowKernel> {
    match row_kernel {
        None => Ok(RowKernel::Structural(max_row_nonzeros)),
        Some("dense") => Ok(RowKernel::Dense),
        Some("sparse") => Ok(RowKernel::Sparse),
        Some(other) => Err(PyValueError::new_err(format!(
            "row_kernel must be None, 'dense' or 'sparse', got {other:?}"
        ))),
    }
}

/// A pool of `threads` threads for a fit's row reductions, or None (the global
/// pool) when no count is given.
pub(crate) fn fit_thread_pool(threads: Option<usize>) -> PyResult<Option<rayon::ThreadPool>> {
    match threads {
        None => Ok(None),
        Some(0) => Err(PyValueError::new_err("threads must be positive, got 0")),
        Some(threads) => rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|index| format!("rs-fit-{index}"))
            .build()
            .map(Some)
            .map_err(|error| {
                PyValueError::new_err(format!("cannot start {threads} threads: {error}"))
            }),
    }
}

/// Run `op` in `pool`, or in the current pool without one.
fn in_pool<R: Send>(pool: Option<&rayon::ThreadPool>, op: impl FnOnce() -> R + Send) -> R {
    match pool {
        Some(pool) => pool.install(op),
        None => op(),
    }
}

/// A block reducer with one worker partial per thread of `pool` (or the global pool).
fn pool_reducer<P: rustystats_core::rows::Partial + Clone>(
    pool: Option<&rayon::ThreadPool>,
    zero: P,
) -> BlockReducer<P> {
    let threads = pool.map_or_else(rayon::current_num_threads, |pool| {
        pool.current_num_threads()
    });
    BlockReducer::with_threads(zero, threads)
}

impl PyBatchExecutor<'_> {
    fn abort(&mut self, error: PyErr) -> RustyStatsError {
        let message = error.to_string();
        if self.aborted.is_none() {
            self.aborted = Some(error);
        }
        RustyStatsError::PassAborted(message)
    }

    fn serve(&mut self, request: &PassRequest) -> PyResult<Vec<EvalSum>> {
        let p = self.n_params;
        let py = self.py;
        let width = if self.intercept_only { 1 } else { p };
        let pool = self.pool.as_ref();
        let mut reducer = pool_reducer(pool, PassPartial::zeros(request, width));
        // A centred column makes every row dense in it (§6.7).
        let packed = self
            .sparse_offsets
            .as_deref()
            .filter(|_| rows_stay_sparse(self.standardization.as_ref()));
        let blocks = self.open_pass.call1((request.kind(),))?;
        for block in blocks.try_iter()? {
            let block = block?;
            let (x, y, weights, offset): (
                PyReadonlyArray2<'_, f64>,
                PyReadonlyArray1<'_, f64>,
                PyReadonlyArray1<'_, f64>,
                PyReadonlyArray1<'_, f64>,
            ) = block.extract()?;
            let x_slice = x.as_slice().map_err(|_| {
                PyValueError::new_err("an encoded block must be C-contiguous float64")
            })?;
            let y_slice = y.as_slice()?;
            let rows = y_slice.len();
            let w_slice = weights.as_slice()?;
            let o_slice = offset.as_slice()?;
            if x_slice.len() != rows * p || w_slice.len() != rows || o_slice.len() != rows {
                return Err(PyValueError::new_err(format!(
                    "an encoded block must hold {rows} rows of {p} columns with matching \
                     y, weights and offset"
                )));
            }
            let ctx = RowContext {
                family: self.family.as_ref(),
                link: self.link.as_ref(),
                y: y_slice,
                offset: o_slice,
                prior_weights: w_slice,
                y_mean: self.y_mean,
                min_weight: self.min_weight,
                has_offset: self.has_offset,
                standardization: self.standardization.as_ref(),
            };
            let first_row = reducer.rows();
            let intercept_only = self.intercept_only;
            py.detach(|| {
                in_pool(pool, || {
                    reducer.feed(rows, |partial, range| {
                        process_block(
                            request,
                            &ctx,
                            x_slice,
                            p,
                            range,
                            first_row,
                            partial,
                            packed,
                            intercept_only,
                        )
                    })
                })
            });
        }
        if reducer.rows() != self.n_rows {
            return Err(PyValueError::new_err(format!(
                "the source yielded {} rows in this pass, but the fit has {} rows: the \
                 source changed between passes",
                reducer.rows(),
                self.n_rows
            )));
        }
        Ok(reducer.finish().0)
    }
}

impl PyBatchExecutor<'_> {
    /// One encoded pass writing μ under `predictor` to the spill, block by block.
    fn spill_pass(&mut self, predictor: &Predictor) -> PyResult<()> {
        let sink = self
            .spill_sink
            .clone()
            .ok_or_else(|| PyValueError::new_err("θ estimation needs a spill sink"))?;
        sink.call1((self.py.None(),))?;
        let p = self.n_params;
        let mut rows_seen = 0usize;
        for block in self.open_pass.call1(("theta",))?.try_iter()? {
            let (x, y, weights, offset): (
                PyReadonlyArray2<'_, f64>,
                PyReadonlyArray1<'_, f64>,
                PyReadonlyArray1<'_, f64>,
                PyReadonlyArray1<'_, f64>,
            ) = block?.extract()?;
            let x_slice = x.as_slice().map_err(|_| {
                PyValueError::new_err("an encoded block must be C-contiguous float64")
            })?;
            let (y_slice, w_slice, o_slice) =
                (y.as_slice()?, weights.as_slice()?, offset.as_slice()?);
            let rows = y_slice.len();
            let ctx = RowContext {
                family: self.family.as_ref(),
                link: self.link.as_ref(),
                y: y_slice,
                offset: o_slice,
                prior_weights: w_slice,
                y_mean: self.y_mean,
                min_weight: self.min_weight,
                has_offset: self.has_offset,
                standardization: self.standardization.as_ref(),
            };
            let mut eta = vec![0.0; rows];
            let mut mu = vec![0.0; rows];
            rustystats_core::solvers::passes::row_states(
                predictor,
                &ctx,
                0..rows,
                &mut eta,
                &mut mu,
                |i| RowRef::Dense(&x_slice[i * p..(i + 1) * p]),
            );
            sink.call1((numpy::PyArray1::from_vec(self.py, mu),))?;
            rows_seen += rows;
        }
        if rows_seen != self.n_rows {
            return Err(PyValueError::new_err(format!(
                "the source yielded {rows_seen} rows in this pass, but the fit has {} rows: \
                 the source changed between passes",
                self.n_rows
            )));
        }
        Ok(())
    }

    /// One θ pass: Σ w·ℓ(y, μ; θ) over `open_theta()`'s blocks, block-ordered.
    fn theta_pass(&mut self, theta: f64, weighted: bool) -> PyResult<f64> {
        let open = self
            .open_theta
            .clone()
            .ok_or_else(|| PyValueError::new_err("θ estimation needs open_theta"))?;
        let py = self.py;
        let pool = self.pool.as_ref();
        let mut reducer = pool_reducer(pool, 0.0f64);
        for block in open.call0()?.try_iter()? {
            let (y, weights, mu): (
                PyReadonlyArray1<'_, f64>,
                PyReadonlyArray1<'_, f64>,
                PyReadonlyArray1<'_, f64>,
            ) = block?.extract()?;
            let (y, weights, mu) = (y.as_slice()?, weights.as_slice()?, mu.as_slice()?);
            if weights.len() != y.len() || mu.len() != y.len() {
                return Err(PyValueError::new_err(
                    "a θ block must hold y, weights and μ of the same rows",
                ));
            }
            py.detach(|| {
                in_pool(pool, || {
                    reducer.feed(y.len(), |total, range| {
                        rustystats_core::diagnostics::fold_nb_loglik(
                            total,
                            &y[range.clone()],
                            &mu[range.clone()],
                            weighted.then(|| &weights[range]),
                            theta,
                        )
                    })
                })
            });
        }
        if reducer.rows() != self.n_rows {
            return Err(PyValueError::new_err(format!(
                "the θ pass read {} rows, but the fit has {} rows",
                reducer.rows(),
                self.n_rows
            )));
        }
        Ok(reducer.finish())
    }
}

impl PassExecutor for PyBatchExecutor<'_> {
    fn n_rows(&self) -> usize {
        self.n_rows
    }

    fn n_params(&self) -> usize {
        if self.intercept_only {
            1
        } else {
            self.n_params
        }
    }

    fn run(&mut self, request: &PassRequest) -> CoreResult<Vec<EvalSum>> {
        if self.aborted.is_some() {
            return Err(RustyStatsError::PassAborted(
                "an earlier pass of this fit was aborted".to_string(),
            ));
        }
        self.serve(request).map_err(|error| self.abort(error))
    }

    fn set_row_standardization(&mut self, standardization: Option<Standardization>) {
        self.standardization = standardization;
    }

    fn set_intercept_only(&mut self, intercept_only: bool) {
        self.intercept_only = intercept_only;
    }

    fn set_family(&mut self, family: Box<dyn Family>) {
        self.family = family;
    }

    fn spill_mu(&mut self, predictor: &Predictor) -> CoreResult<()> {
        if self.aborted.is_some() {
            return Err(RustyStatsError::PassAborted(
                "an earlier pass of this fit was aborted".to_string(),
            ));
        }
        self.spill_pass(predictor)
            .map_err(|error| self.abort(error))
    }

    fn spilled_nb_loglik(&mut self, theta: f64, weighted: bool) -> CoreResult<f64> {
        if self.aborted.is_some() {
            return Err(RustyStatsError::PassAborted(
                "an earlier pass of this fit was aborted".to_string(),
            ));
        }
        self.theta_pass(theta, weighted)
            .map_err(|error| self.abort(error))
    }

    fn prefers_few_passes(&self) -> bool {
        true
    }

    fn trace(&mut self) -> Option<&mut Vec<SolverStep>> {
        Some(&mut self.trace)
    }
}

/// Fit a GLM whose encoded rows a Python callable serves, pass by pass.
///
/// `open_pass(kind)` is called once per pass and must yield `(x, y, weights, offset)`
/// blocks covering all `n_rows` rows in the same order every time. `y_mean` is
/// the block-ordered unweighted mean response (`families::response_mean`). The
/// result holds no per-row arrays.
#[pyfunction]
#[pyo3(signature = (open_pass, n_rows, n_params, y_mean, family, link=None, var_power=1.5, theta=1.0, has_offset=false, alpha=0.0, l1_ratio=0.0, max_iter=25, tol=1e-8, nonneg_indices=None, nonpos_indices=None, allow_extended_tweedie=false, fit_intercept=true, center=None, scale=None, skip_covariance=false, robust=None, max_row_nonzeros=None, row_kernel=None, threads=None))]
#[allow(clippy::too_many_arguments)]
pub fn fit_glm_batched_py(
    py: Python<'_>,
    open_pass: Bound<'_, PyAny>,
    n_rows: usize,
    n_params: usize,
    y_mean: f64,
    family: &str,
    link: Option<&str>,
    var_power: f64,
    theta: f64,
    has_offset: bool,
    alpha: f64,
    l1_ratio: f64,
    max_iter: usize,
    tol: f64,
    nonneg_indices: Option<Vec<usize>>,
    nonpos_indices: Option<Vec<usize>>,
    allow_extended_tweedie: bool,
    fit_intercept: bool,
    center: Option<PyReadonlyArray1<f64>>,
    scale: Option<PyReadonlyArray1<f64>>,
    skip_covariance: bool,
    robust: Option<Vec<String>>,
    max_row_nonzeros: Option<usize>,
    row_kernel: Option<&str>,
    threads: Option<usize>,
) -> PyResult<PyGLMResults> {
    let robust_types = parse_robust_types(robust, skip_covariance)?;
    let standardization = build_standardization(center, scale, n_params)?;
    let reg_config = if alpha > 0.0 {
        if l1_ratio >= 1.0 {
            RegularizationConfig::lasso(alpha)
        } else if l1_ratio <= 0.0 {
            RegularizationConfig::ridge(alpha)
        } else {
            RegularizationConfig::elastic_net(alpha, l1_ratio)
        }
    } else {
        RegularizationConfig::none()
    }
    .with_intercept(fit_intercept);
    let config = FitConfig {
        max_iterations: max_iter,
        tolerance: tol,
        min_weight: 1e-10,
        verbose: false,
        nonneg_indices: nonneg_indices.unwrap_or_default(),
        nonpos_indices: nonpos_indices.unwrap_or_default(),
        regularization: reg_config,
        skip_covariance,
        standardization,
        robust: robust_types,
        row_kernel: parse_row_kernel(max_row_nonzeros, row_kernel)?,
    };
    let fam =
        family_from_name_with_tweedie_support(family, var_power, theta, allow_extended_tweedie)?;
    let link_name = link.unwrap_or(default_link_name(family));
    let lnk = link_from_name(link_name)?;
    let link_for_fit = link_from_name(link_name)?;
    let family_for_fit =
        family_from_name_with_tweedie_support(family, var_power, theta, allow_extended_tweedie)?;

    let mut executor = PyBatchExecutor::new(
        py, open_pass, n_rows, n_params, fam, lnk, y_mean, has_offset,
    )
    .serving(parse_row_kernel(max_row_nonzeros, row_kernel)?, threads)?;
    executor.min_weight = config.min_weight;
    let fitted = fit_glm_with_executor(
        &mut executor,
        family_for_fit.as_ref(),
        link_for_fit.as_ref(),
        &config,
        None,
        has_offset,
        y_mean,
    );
    let mut fitted = match fitted {
        Ok(fitted) => fitted,
        Err(RustyStatsError::PassAborted(message)) => {
            return Err(executor
                .aborted
                .take()
                .unwrap_or_else(|| PyValueError::new_err(message)));
        }
        Err(error) => {
            return Err(PyValueError::new_err(format!(
                "GLM fitting failed: {error}"
            )));
        }
    };

    let exact_theta = resolve_negbinomial_theta(family, theta)?;
    let exact_var_power = if exact_theta.is_none() {
        resolve_tweedie_var_power(family, var_power)?
    } else {
        None
    };
    let family_name = if let Some(resolved_theta) = exact_theta {
        format!("NegativeBinomial(theta={resolved_theta:.4})")
    } else if let Some(resolved_var_power) = exact_var_power {
        format!("Tweedie(p={resolved_var_power:.4})")
    } else {
        family_for_fit.name().to_string()
    };

    let robust_covariances = fitted
        .statistics
        .as_mut()
        .map(|statistics| std::mem::take(&mut statistics.robust))
        .unwrap_or_default();
    let empty = || Array1::<f64>::zeros(0);
    Ok(PyGLMResults {
        coefficients: fitted.coefficients,
        fitted_values: empty(),
        linear_predictor: empty(),
        deviance: fitted.deviance,
        iterations: fitted.iterations,
        converged: fitted.converged,
        covariance_unscaled: fitted.covariance_unscaled,
        n_obs: n_rows,
        n_params,
        y: empty(),
        family_name,
        prior_weights: empty(),
        penalty: fitted.penalty,
        design_matrix: None,
        irls_weights: empty(),
        offset: None,
        step_halving_used: fitted.step_halving_used,
        solver_status: fitted.solver_status,
        stationary: None,
        max_std_score: None,
        warnings: fitted.warnings,
        statistics: fitted.statistics,
        robust_covariances,
        exact_theta,
        exact_var_power,
        rowless: true,
        solver_trace: std::mem::take(&mut executor.trace),
    })
}

/// Fit a GLM with penalised smooth terms whose encoded rows a Python callable
/// serves, pass by pass (§7.6). Arguments as `fit_smooth_glm_unified_py`, with the
/// rows from `open_pass(kind)` as for `fit_glm_batched_py`. Returns the rowless result
/// and the smooth metadata dict.
#[pyfunction]
#[pyo3(signature = (open_pass, n_rows, n_params, y_mean, smooth_col_ranges, smooth_penalties, family, link=None, has_offset=false, max_iter=25, tol=1e-8, lambda_min=0.001, lambda_max=1000.0, smooth_monotonicity=None, nonneg_indices=None, nonpos_indices=None, var_power=1.5, theta=1.0, allow_extended_tweedie=false, robust=None, max_row_nonzeros=None, row_kernel=None, threads=None))]
#[allow(clippy::too_many_arguments)]
pub fn fit_smooth_glm_batched_py<'py>(
    py: Python<'py>,
    open_pass: Bound<'py, PyAny>,
    n_rows: usize,
    n_params: usize,
    y_mean: f64,
    smooth_col_ranges: Vec<(usize, usize)>,
    smooth_penalties: Vec<PyReadonlyArray2<'py, f64>>,
    family: &str,
    link: Option<&str>,
    has_offset: bool,
    max_iter: usize,
    tol: f64,
    lambda_min: f64,
    lambda_max: f64,
    smooth_monotonicity: Option<Vec<Option<String>>>,
    nonneg_indices: Option<Vec<usize>>,
    nonpos_indices: Option<Vec<usize>>,
    var_power: f64,
    theta: f64,
    allow_extended_tweedie: bool,
    robust: Option<Vec<String>>,
    max_row_nonzeros: Option<usize>,
    row_kernel: Option<&str>,
    threads: Option<usize>,
) -> PyResult<Py<PyAny>> {
    use rustystats_core::solvers::fit_smooth_glm_with_executor;
    let robust_types = parse_robust_types(robust, false)?;
    let specs = crate::fitting_py::smooth_term_specs(
        &smooth_col_ranges,
        &smooth_penalties,
        smooth_monotonicity,
    )?;
    let mut config = crate::fitting_py::build_smooth_config(max_iter, tol, lambda_min, lambda_max);
    config.robust = robust_types;
    let fam =
        family_from_name_with_tweedie_support(family, var_power, theta, allow_extended_tweedie)?;
    let family_for_fit =
        family_from_name_with_tweedie_support(family, var_power, theta, allow_extended_tweedie)?;
    let link_name = link.unwrap_or(default_link_name(family));
    let lnk = link_from_name(link_name)?;
    let link_for_fit = link_from_name(link_name)?;
    let mut executor = PyBatchExecutor::new(
        py, open_pass, n_rows, n_params, fam, lnk, y_mean, has_offset,
    )
    .serving(parse_row_kernel(max_row_nonzeros, row_kernel)?, threads)?;
    executor.min_weight = config.irls_config.min_weight;
    let nonneg = nonneg_indices.unwrap_or_default();
    let nonpos = nonpos_indices.unwrap_or_default();
    let fitted = fit_smooth_glm_with_executor(
        &mut executor,
        &specs,
        family_for_fit.as_ref(),
        link_for_fit.as_ref(),
        &config,
        (!nonneg.is_empty()).then_some(nonneg.as_slice()),
        (!nonpos.is_empty()).then_some(nonpos.as_slice()),
        has_offset,
    );
    let mut fitted = match fitted {
        Ok(fitted) => fitted,
        Err(RustyStatsError::PassAborted(message)) => {
            return Err(executor
                .aborted
                .take()
                .unwrap_or_else(|| PyValueError::new_err(message)));
        }
        Err(error) => {
            return Err(PyValueError::new_err(format!(
                "Smooth GLM fitting failed: {error}"
            )));
        }
    };

    let exact_theta = resolve_negbinomial_theta(family, theta)?;
    let exact_var_power = if exact_theta.is_none() {
        resolve_tweedie_var_power(family, var_power)?
    } else {
        None
    };
    let family_name = if let Some(resolved_theta) = exact_theta {
        format!("NegativeBinomial(theta={resolved_theta:.4})")
    } else if let Some(resolved_var_power) = exact_var_power {
        format!("Tweedie(p={resolved_var_power:.4})")
    } else {
        family_for_fit.name().to_string()
    };
    let robust_covariances = std::mem::take(&mut fitted.statistics.robust);
    let empty = || Array1::<f64>::zeros(0);
    let result = PyGLMResults {
        coefficients: fitted.coefficients,
        fitted_values: empty(),
        linear_predictor: empty(),
        deviance: fitted.deviance,
        iterations: fitted.iterations,
        converged: fitted.converged,
        covariance_unscaled: fitted.covariance_unscaled,
        n_obs: n_rows,
        n_params,
        y: empty(),
        family_name,
        prior_weights: empty(),
        penalty: fitted.penalty,
        design_matrix: None,
        irls_weights: empty(),
        offset: None,
        step_halving_used: fitted.step_halving_used,
        solver_status: fitted.solver_status,
        stationary: Some(fitted.stationary),
        max_std_score: Some(fitted.max_std_score),
        warnings: fitted.warnings,
        statistics: Some(fitted.statistics),
        robust_covariances,
        exact_theta,
        exact_var_power,
        rowless: true,
        solver_trace: std::mem::take(&mut executor.trace),
    };
    let smooth = pyo3::types::PyDict::new(py);
    smooth.set_item("lambdas", fitted.lambdas)?;
    smooth.set_item("smooth_edfs", fitted.smooth_edfs)?;
    smooth.set_item("total_edf", fitted.total_edf)?;
    smooth.set_item("gcv", fitted.gcv)?;
    let tuple =
        pyo3::types::PyTuple::new(py, &[Bound::new(py, result)?.into_any(), smooth.into_any()])?;
    Ok(tuple.unbind().into())
}

impl<'py> PyBatchExecutor<'py> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'py>,
        open_pass: Bound<'py, PyAny>,
        n_rows: usize,
        n_params: usize,
        family: Box<dyn Family>,
        link: Box<dyn Link>,
        y_mean: f64,
        has_offset: bool,
    ) -> Self {
        Self {
            py,
            open_pass,
            n_rows,
            n_params,
            family,
            link,
            y_mean,
            min_weight: 1e-10,
            has_offset,
            aborted: None,
            standardization: None,
            intercept_only: false,
            spill_sink: None,
            open_theta: None,
            trace: Vec::new(),
            sparse_offsets: None,
            pool: None,
        }
    }

    /// Serve the rows `kernel` picks (§5.2), and run the row reductions in a pool of
    /// `threads` threads (None: the global pool), the count the memory plan
    /// budgets. Neither changes results.
    fn serving(mut self, kernel: RowKernel, threads: Option<usize>) -> PyResult<Self> {
        self.sparse_offsets = sparse_row_offsets(self.n_rows, self.n_params, kernel);
        self.pool = fit_thread_pool(threads)?;
        Ok(self)
    }

    /// The threads the row reductions run on: the requested pool's, else the global
    /// pool's.
    fn threads(&self) -> usize {
        self.pool
            .as_ref()
            .map_or_else(rayon::current_num_threads, |pool| {
                pool.current_num_threads()
            })
    }

    /// A core error as the Python exception to raise: an aborted pass re-raises the
    /// exception that aborted it.
    fn error(&mut self, error: RustyStatsError, context: &str) -> PyErr {
        match error {
            RustyStatsError::PassAborted(message) => self
                .aborted
                .take()
                .unwrap_or_else(|| PyValueError::new_err(message)),
            error => PyValueError::new_err(format!("{context}: {error}")),
        }
    }
}

/// alpha_max (`regularization_path.compute_alpha_max`) over batches: the
/// intercept-only fit and the score pass of `rustystats_core::solvers::cv`.
#[pyfunction]
#[pyo3(signature = (open_pass, n_rows, n_params, y_mean, family, link, l1_ratio, has_offset=false, var_power=1.5, theta=1.0, pen_mask=None, center=None, scale=None, ridge_xtx_diag=None, allow_extended_tweedie=false, trace=None, max_row_nonzeros=None, row_kernel=None, threads=None))]
#[allow(clippy::too_many_arguments)]
pub fn alpha_max_batched_py(
    py: Python<'_>,
    open_pass: Bound<'_, PyAny>,
    n_rows: usize,
    n_params: usize,
    y_mean: f64,
    family: &str,
    link: &str,
    l1_ratio: f64,
    has_offset: bool,
    var_power: f64,
    theta: f64,
    pen_mask: Option<Vec<bool>>,
    center: Option<PyReadonlyArray1<f64>>,
    scale: Option<PyReadonlyArray1<f64>>,
    ridge_xtx_diag: Option<Vec<f64>>,
    allow_extended_tweedie: bool,
    trace: Option<Bound<'_, pyo3::types::PyList>>,
    max_row_nonzeros: Option<usize>,
    row_kernel: Option<&str>,
    threads: Option<usize>,
) -> PyResult<f64> {
    use rustystats_core::solvers::cv::{alpha_max, AlphaMaxSettings};
    let standardization = build_standardization(center, scale, n_params)?;
    let (null_fit, score_form) = crate::cv_py::alpha_forms(family, link, var_power, theta)?;
    let fam =
        family_from_name_with_tweedie_support(family, var_power, theta, allow_extended_tweedie)?;
    let lnk = link_from_name(link)?;
    let mask = crate::cv_py::penalty_mask(pen_mask, n_params, None)?;
    let mut executor = PyBatchExecutor::new(
        py,
        open_pass,
        n_rows,
        n_params,
        family_from_name_with_tweedie_support(family, var_power, theta, allow_extended_tweedie)?,
        link_from_name(link)?,
        y_mean,
        has_offset,
    )
    .serving(parse_row_kernel(max_row_nonzeros, row_kernel)?, threads)?;
    let result = alpha_max(
        &mut executor,
        &AlphaMaxSettings {
            family: fam.as_ref(),
            link: lnk.as_ref(),
            null_fit,
            score_form,
            l1_ratio,
            penalty_mask: &mask,
            standardization: standardization.as_ref(),
            ridge_diag: ridge_xtx_diag.as_deref(),
            y_mean,
            has_offset,
        },
    );
    crate::cv_py::append_trace(trace.as_ref(), &std::mem::take(&mut executor.trace))?;
    result.map_err(|error| executor.error(error, "alpha_max"))
}

/// The CV path over an alpha grid from batches (§7.8): `fit_cv_path_py`'s folds in
/// lockstep, every merged pass served by `open_pass(kind)`.
#[pyfunction]
#[pyo3(signature = (open_pass, n_rows, n_params, y_mean, family, link=None, var_power=1.5, theta=1.0, has_offset=false, weighted=false, alphas=None, l1_ratio=0.0, n_folds=5, max_iter=25, tol=1e-8, seed=None, nonneg_indices=None, nonpos_indices=None, allow_extended_tweedie=false, fit_intercept=true, center=None, scale=None, solve_workspaces=1, max_row_nonzeros=None, row_kernel=None, threads=None))]
#[allow(clippy::too_many_arguments)]
pub fn fit_cv_path_batched_py<'py>(
    py: Python<'py>,
    open_pass: Bound<'py, PyAny>,
    n_rows: usize,
    n_params: usize,
    y_mean: f64,
    family: &str,
    link: Option<&str>,
    var_power: f64,
    theta: f64,
    has_offset: bool,
    weighted: bool,
    alphas: Option<Vec<f64>>,
    l1_ratio: f64,
    n_folds: usize,
    max_iter: usize,
    tol: f64,
    seed: Option<u64>,
    nonneg_indices: Option<Vec<usize>>,
    nonpos_indices: Option<Vec<usize>>,
    allow_extended_tweedie: bool,
    fit_intercept: bool,
    center: Option<PyReadonlyArray1<f64>>,
    scale: Option<PyReadonlyArray1<f64>>,
    solve_workspaces: usize,
    max_row_nonzeros: Option<usize>,
    row_kernel: Option<&str>,
    threads: Option<usize>,
) -> PyResult<Py<PyAny>> {
    use rustystats_core::solvers::cv::{cv_path, CvSettings};
    let start = std::time::Instant::now();
    let alphas = alphas.ok_or_else(|| {
        PyValueError::new_err("fit_cv_path_batched_py requires an explicit `alphas=` grid.")
    })?;
    let standardization = build_standardization(center, scale, n_params)?;
    let link_name = link.unwrap_or(default_link_name(family));
    let fam =
        family_from_name_with_tweedie_support(family, var_power, theta, allow_extended_tweedie)?;
    let lnk = link_from_name(link_name)?;
    let mut executor = PyBatchExecutor::new(
        py,
        open_pass,
        n_rows,
        n_params,
        family_from_name_with_tweedie_support(family, var_power, theta, allow_extended_tweedie)?,
        link_from_name(link_name)?,
        y_mean,
        has_offset,
    )
    .serving(parse_row_kernel(max_row_nonzeros, row_kernel)?, threads)?;
    let nonneg = nonneg_indices.unwrap_or_default();
    let nonpos = nonpos_indices.unwrap_or_default();
    let path = cv_path(
        &mut executor,
        &CvSettings {
            family: fam.as_ref(),
            link: lnk.as_ref(),
            alphas: &alphas,
            l1_ratio,
            folds: n_folds,
            seed: seed.unwrap_or(42),
            max_iter,
            tol,
            nonneg: &nonneg,
            nonpos: &nonpos,
            fit_intercept,
            standardization: standardization.as_ref(),
            has_offset,
            weighted,
            validation_form: crate::cv_py::validation_form(fam.as_ref(), lnk.as_ref()),
            solve_workspaces,
        },
    );
    let path = path.map_err(|error| executor.error(error, "CV path failed"))?;
    let dict = crate::cv_py::cv_result_dict(
        py,
        &path,
        n_rows,
        n_params,
        n_folds,
        start.elapsed().as_secs_f64(),
        None,
        executor.threads(),
    )?;
    dict.set_item(
        "solver_trace",
        crate::results_py::trace_tuples(&executor.trace),
    )?;
    Ok(dict.unbind().into())
}

/// Negative Binomial θ estimation from batches (§7.10): `fit_negbinomial_py`'s
/// steps, μ written to a spill through `spill(block)` (`spill(None)` starts it over)
/// and read back by `open_theta()` as `(y, weights, mu)` blocks for each θ pass.
/// Returns the rowless result and the θ metadata dict.
#[pyfunction]
#[pyo3(signature = (open_pass, spill, open_theta, n_rows, n_params, y_mean, link=None, init_theta=None, theta_tol=1e-5, max_theta_iter=10, has_offset=false, weighted=false, max_iter=25, tol=1e-8, robust=None, max_row_nonzeros=None, row_kernel=None, threads=None))]
#[allow(clippy::too_many_arguments)]
pub fn fit_negbinomial_batched_py<'py>(
    py: Python<'py>,
    open_pass: Bound<'py, PyAny>,
    spill: Bound<'py, PyAny>,
    open_theta: Bound<'py, PyAny>,
    n_rows: usize,
    n_params: usize,
    y_mean: f64,
    link: Option<&str>,
    init_theta: Option<f64>,
    theta_tol: f64,
    max_theta_iter: usize,
    has_offset: bool,
    weighted: bool,
    max_iter: usize,
    tol: f64,
    robust: Option<Vec<String>>,
    max_row_nonzeros: Option<usize>,
    row_kernel: Option<&str>,
    threads: Option<usize>,
) -> PyResult<Py<PyAny>> {
    use rustystats_core::solvers::nb_theta::{fit_negbinomial_with_executor, NbThetaSettings};
    let robust_types = parse_robust_types(robust, false)?;
    let link_name = link.unwrap_or("log");
    if link_name != "log" && link_name != "identity" {
        return Err(PyValueError::new_err(format!(
            "Unknown link '{link_name}' for NegativeBinomial. Use 'log' or 'identity'."
        )));
    }
    let config = FitConfig {
        max_iterations: max_iter,
        tolerance: tol,
        min_weight: 1e-10,
        ..FitConfig::default()
    };
    let final_config = FitConfig {
        robust: robust_types,
        ..config.clone()
    };
    let lnk = link_from_name(link_name)?;
    let mut executor = PyBatchExecutor::new(
        py,
        open_pass,
        n_rows,
        n_params,
        Box::new(rustystats_core::families::PoissonFamily),
        link_from_name(link_name)?,
        y_mean,
        has_offset,
    )
    .serving(parse_row_kernel(max_row_nonzeros, row_kernel)?, threads)?;
    executor.spill_sink = Some(spill);
    executor.open_theta = Some(open_theta);
    let fitted = fit_negbinomial_with_executor(
        &mut executor,
        &NbThetaSettings {
            link: lnk.as_ref(),
            config: &config,
            final_config: &final_config,
            init_theta,
            theta_tol,
            max_theta_iter,
            has_offset,
            weighted,
            y_mean,
        },
    );
    let mut fitted = fitted.map_err(|error| executor.error(error, "GLM fitting failed"))?;
    let theta = fitted.theta;
    let robust_covariances = fitted
        .fit
        .statistics
        .as_mut()
        .map(|statistics| std::mem::take(&mut statistics.robust))
        .unwrap_or_default();
    let fit = fitted.fit;
    let empty = || Array1::<f64>::zeros(0);
    let result = PyGLMResults {
        coefficients: fit.coefficients,
        fitted_values: empty(),
        linear_predictor: empty(),
        deviance: fit.deviance,
        iterations: fit.iterations,
        converged: fit.converged,
        covariance_unscaled: fit.covariance_unscaled,
        n_obs: n_rows,
        n_params,
        y: empty(),
        family_name: format!("NegativeBinomial(theta={theta:.4})"),
        prior_weights: empty(),
        penalty: fit.penalty,
        design_matrix: None,
        irls_weights: empty(),
        offset: None,
        step_halving_used: fit.step_halving_used,
        solver_status: fit.solver_status,
        stationary: None,
        max_std_score: None,
        warnings: fit.warnings,
        statistics: fit.statistics,
        robust_covariances,
        exact_theta: Some(theta),
        exact_var_power: None,
        rowless: true,
        solver_trace: std::mem::take(&mut executor.trace),
    };
    let meta = pyo3::types::PyDict::new(py);
    meta.set_item("estimated", true)?;
    meta.set_item("theta", theta)?;
    meta.set_item("init_theta", fitted.init_theta)?;
    meta.set_item("theta_iterations", fitted.iterations)?;
    meta.set_item("theta_converged", fitted.converged)?;
    meta.set_item("theta_tol", theta_tol)?;
    meta.set_item("max_theta_iter", max_theta_iter)?;
    meta.set_item("glm_tol", tol)?;
    meta.set_item("fallback_reason", py.None())?;
    let tuple =
        pyo3::types::PyTuple::new(py, &[Bound::new(py, result)?.into_any(), meta.into_any()])?;
    Ok(tuple.unbind().into())
}

/// One pass over `open_pass("moments")`'s blocks, folding each block's rows into `reducer`
/// (`fold(partial, x, weights, rows)`), checking the row count.
fn moments_pass<P, F>(
    py: Python<'_>,
    pool: Option<&rayon::ThreadPool>,
    open_pass: &Bound<'_, PyAny>,
    n_rows: usize,
    n_params: usize,
    zero: P,
    fold: F,
) -> PyResult<P>
where
    P: rustystats_core::rows::Partial + Clone + Send,
    F: Fn(
            &mut P,
            ndarray::ArrayView2<'_, f64>,
            ndarray::ArrayView1<'_, f64>,
            std::ops::Range<usize>,
            usize,
        ) + Sync,
{
    let mut reducer = pool_reducer(pool, zero);
    for block in open_pass.call1(("moments",))?.try_iter()? {
        let (x, _y, weights, _offset): (
            PyReadonlyArray2<'_, f64>,
            PyReadonlyArray1<'_, f64>,
            PyReadonlyArray1<'_, f64>,
            PyReadonlyArray1<'_, f64>,
        ) = block?.extract()?;
        let x = x.as_array();
        let weights = weights.as_array();
        let rows = weights.len();
        if x.dim() != (rows, n_params) {
            return Err(PyValueError::new_err(format!(
                "an encoded block must hold {rows} rows of {n_params} columns"
            )));
        }
        let first_row = reducer.rows();
        py.detach(|| {
            in_pool(pool, || {
                reducer.feed(rows, |partial, range| {
                    fold(partial, x, weights, range, first_row)
                })
            })
        });
    }
    if reducer.rows() != n_rows {
        return Err(PyValueError::new_err(format!(
            "the source yielded {} rows in this pass, but the fit has {n_rows} rows: the \
             source changed between passes",
            reducer.rows()
        )));
    }
    Ok(reducer.finish())
}

/// Standardisation moments over batches (§6.7): passes M1 and M2 over
/// `open_pass("moments")`'s `(x, y, weights, offset)` blocks, with the folds and finishing
/// of `compute_standardization_py`, so the two agree bit for bit. With
/// `fold=(k, folds, seed)` only fold k's training rows take part (§7.9).
#[pyfunction]
#[pyo3(signature = (open_pass, n_rows, n_params, weighted=false, pen_mask=None, fit_intercept=true, fold=None, threads=None))]
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn compute_standardization_batched_py<'py>(
    py: Python<'py>,
    open_pass: Bound<'py, PyAny>,
    n_rows: usize,
    n_params: usize,
    weighted: bool,
    pen_mask: Option<Vec<bool>>,
    fit_intercept: bool,
    fold: Option<(usize, usize, u64)>,
    threads: Option<usize>,
) -> PyResult<(
    Bound<'py, numpy::PyArray1<f64>>,
    Bound<'py, numpy::PyArray1<f64>>,
    Bound<'py, numpy::PyArray1<f64>>,
)> {
    use numpy::IntoPyArray;
    use rustystats_core::regularization::moments::{
        finalize_moments, fold_first_moments_where, fold_second_moments_where, means_from_sums,
    };
    let training = fold.map(
        |(fold, folds, seed)| rustystats_core::solvers::passes::FoldFilter {
            fold,
            folds,
            seed,
            validation: false,
        },
    );
    let include = |first_row: usize| {
        move |row: usize| training.is_none_or(|filter| filter.includes(first_row + row))
    };
    let mut mask = pen_mask.unwrap_or_else(|| vec![true; n_params]);
    if mask.len() != n_params {
        return Err(PyValueError::new_err(format!(
            "pen_mask must be length {n_params} for standardization, got length {}.",
            mask.len()
        )));
    }
    if fit_intercept && n_params > 0 {
        mask[0] = false;
    }
    let active: Vec<usize> = mask
        .iter()
        .enumerate()
        .filter_map(|(idx, active)| active.then_some(idx))
        .collect();
    if n_params == 0 || active.is_empty() {
        return Ok((
            vec![0.0; n_params].into_pyarray(py),
            vec![1.0; n_params].into_pyarray(py),
            vec![0.0; n_params].into_pyarray(py),
        ));
    }
    let active_len = active.len();
    let pool = fit_thread_pool(threads)?;
    let sums = moments_pass(
        py,
        pool.as_ref(),
        &open_pass,
        n_rows,
        n_params,
        vec![0.0; active_len + 1],
        |sums, x, weights, rows, first_row| {
            fold_first_moments_where(
                sums,
                x,
                weighted.then_some(weights),
                &active,
                rows,
                include(first_row),
            )
        },
    )?;
    let weight_sum = sums[active_len];
    if !weight_sum.is_finite() || weight_sum <= 0.0 {
        return Err(PyValueError::new_err(format!(
            "standardization requires positive finite sum(weights), got {weight_sum}."
        )));
    }
    let means = means_from_sums(&sums, active_len);
    let squares = moments_pass(
        py,
        pool.as_ref(),
        &open_pass,
        n_rows,
        n_params,
        vec![0.0; active_len],
        |squares, x, weights, rows, first_row| {
            fold_second_moments_where(
                squares,
                x,
                weighted.then_some(weights),
                &active,
                &means,
                rows,
                include(first_row),
            )
        },
    )?;
    let moments = finalize_moments(n_params, &active, &sums, &squares, fit_intercept)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok((
        moments.center.into_pyarray(py),
        moments.scale.into_pyarray(py),
        moments.ridge_xtx_diag.into_pyarray(py),
    ))
}

/// The fixed solver constants a result records in `solver_settings` (§4.5): the
/// step-halving candidates per IRLS iteration and the rows per reduction block.
#[pyfunction]
pub fn solver_constants_py() -> (usize, usize) {
    (
        rustystats_core::constants::IRLS_MAX_HALF_STEPS,
        rustystats_core::rows::GRAM_BLOCK_ROWS,
    )
}

/// Whether a batched fit of `n_rows` rows and `n_params` columns, whose design has at
/// most `max_row_nonzeros` nonzeros per row, serves its rows as sparse rows (§5.2):
/// each reduction thread then holds one block's CSR besides its partial (§11).
#[pyfunction]
#[pyo3(signature = (n_rows, n_params, max_row_nonzeros=None))]
pub fn sparse_rows_py(n_rows: usize, n_params: usize, max_row_nonzeros: Option<usize>) -> bool {
    sparse_row_offsets(n_rows, n_params, RowKernel::Structural(max_row_nonzeros)).is_some()
}

/// Threads of Rayon's global pool: the pool a fit without `threads=` runs its row
/// reductions in, so the count a memory plan budgets for it.
#[pyfunction]
pub fn rayon_threads_py() -> usize {
    rayon::current_num_threads()
}

/// Xβ for every row of `x`, each row summed in column order (`row_dot`, the
/// fit's row kernel), so a row's value depends on nothing but the row (§6.8 predict).
#[pyfunction]
pub fn row_dots_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    beta: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    use numpy::IntoPyArray;
    use rayon::prelude::*;
    let x = x.as_array();
    let beta = beta.as_slice()?;
    if x.ncols() != beta.len() {
        return Err(PyValueError::new_err(format!(
            "x has {} columns but beta has length {}",
            x.ncols(),
            beta.len()
        )));
    }
    let out: Vec<f64> = py.detach(|| {
        (0..x.nrows())
            .into_par_iter()
            .map(|i| {
                let mut sum = 0.0;
                for (&value, &b) in x.row(i).iter().zip(beta.iter()) {
                    sum += value * b;
                }
                sum
            })
            .collect()
    });
    Ok(out.into_pyarray(py))
}

/// The block-ordered unweighted mean response (`families::response_mean`), the
/// `y_mean` a batched fit takes.
#[pyfunction]
pub fn response_mean_py(y: PyReadonlyArray1<f64>) -> f64 {
    rustystats_core::families::response_mean(&y.as_array().to_owned())
}

/// A streaming block-ordered sum of a column (`rows::BlockReducer`): fed slice by
/// slice in row order, it gives the bits `families::response_mean`'s reduction
/// gives over the whole column, whatever the slicing.
#[pyclass(name = "RowSum")]
pub struct RowSumPy {
    inner: BlockReducer<f64>,
}

#[pymethods]
impl RowSumPy {
    #[new]
    fn new() -> Self {
        Self {
            inner: BlockReducer::new(0.0),
        }
    }

    /// Add the next rows' values, in row order.
    fn update(&mut self, values: PyReadonlyArray1<f64>) -> PyResult<()> {
        let values = values.as_slice()?;
        self.inner.feed(values.len(), |sum, range| {
            for value in &values[range] {
                *sum += value;
            }
        });
        Ok(())
    }

    /// Rows seen so far.
    #[getter]
    fn rows(&self) -> usize {
        self.inner.rows()
    }

    /// The sum of everything seen so far.
    fn total(&self) -> f64 {
        self.inner.clone().finish()
    }

    /// The mean as `families::response_mean` forms it (1.0 with no rows).
    fn mean(&self) -> f64 {
        let n = self.inner.rows();
        if n == 0 {
            return 1.0;
        }
        self.inner.clone().finish() / n as f64
    }
}

/// A streaming, batching-independent fingerprint of one column
/// (`rustystats_core::fingerprint`, §9).
#[pyclass(name = "ColumnFingerprint")]
pub struct ColumnFingerprintPy {
    inner: rustystats_core::fingerprint::ColumnFingerprint,
}

#[pymethods]
impl ColumnFingerprintPy {
    #[new]
    fn new() -> Self {
        Self {
            inner: rustystats_core::fingerprint::ColumnFingerprint::new(),
        }
    }

    /// Add float64 values (exact bits); `valid` marks non-null rows.
    #[pyo3(signature = (values, valid=None))]
    fn update_f64(
        &mut self,
        values: PyReadonlyArray1<f64>,
        valid: Option<PyReadonlyArray1<bool>>,
    ) -> PyResult<()> {
        let values = values.as_slice()?;
        let valid = valid.as_ref().map(|v| v.as_slice()).transpose()?;
        check_validity_length(values.len(), valid)?;
        self.inner.update_f64(values, valid);
        Ok(())
    }

    /// Add integers by their 64-bit patterns; `valid` marks non-null rows.
    #[pyo3(signature = (values, valid=None))]
    fn update_bits(
        &mut self,
        values: PyReadonlyArray1<u64>,
        valid: Option<PyReadonlyArray1<bool>>,
    ) -> PyResult<()> {
        let values = values.as_slice()?;
        let valid = valid.as_ref().map(|v| v.as_slice()).transpose()?;
        check_validity_length(values.len(), valid)?;
        self.inner.update_bits(values, valid);
        Ok(())
    }

    /// Add values of `kind` (`rustystats_core::fingerprint::kind`: 2 integer,
    /// 3 boolean, 4 temporal) by their 64-bit patterns; `valid` marks non-null rows.
    #[pyo3(signature = (kind, values, valid=None))]
    fn update_words(
        &mut self,
        kind: u8,
        values: PyReadonlyArray1<u64>,
        valid: Option<PyReadonlyArray1<bool>>,
    ) -> PyResult<()> {
        let values = values.as_slice()?;
        let valid = valid.as_ref().map(|v| v.as_slice()).transpose()?;
        check_validity_length(values.len(), valid)?;
        self.inner.update_words(kind, values, valid);
        Ok(())
    }

    /// Add byte-string values of `kind` (5 string or categorical label, 6 binary,
    /// 7 other text), one per row in order: each item `bytes` or None (a null). The
    /// bytes are read in place, without a copy.
    fn update_bytes(&mut self, kind: u8, values: &Bound<'_, PyAny>) -> PyResult<()> {
        for item in values.try_iter()? {
            let item = item?;
            if item.is_none() {
                self.inner.update_bytes(kind, None);
            } else {
                let bytes = item.cast::<pyo3::types::PyBytes>().map_err(|_| {
                    pyo3::exceptions::PyTypeError::new_err(format!(
                        "a key value must be bytes or None, got {}",
                        item.get_type()
                    ))
                })?;
                self.inner.update_bytes(kind, Some(bytes.as_bytes()));
            }
        }
        Ok(())
    }

    /// Rows seen so far.
    #[getter]
    fn rows(&self) -> u64 {
        self.inner.rows()
    }

    /// The 128-bit digest of everything seen so far.
    fn digest(&self) -> u128 {
        self.inner.digest()
    }
}

fn check_validity_length(rows: usize, valid: Option<&[bool]>) -> PyResult<()> {
    match valid {
        Some(valid) if valid.len() != rows => Err(PyValueError::new_err(format!(
            "the validity mask has {} rows, the values {rows}",
            valid.len()
        ))),
        _ => Ok(()),
    }
}

/// alpha_max over fold `fold`'s training rows from batches (§7.9): `open_pass`
/// serves the fold's design for every row; `fold_alpha_max_py`'s twin.
#[pyfunction]
#[pyo3(signature = (open_pass, n_rows, n_params, y_mean, family, link, l1_ratio, fold, n_folds, seed, has_offset=false, var_power=1.5, theta=1.0, pen_mask=None, center=None, scale=None, ridge_xtx_diag=None, allow_extended_tweedie=false, trace=None, max_row_nonzeros=None, row_kernel=None, threads=None))]
#[allow(clippy::too_many_arguments)]
pub fn fold_alpha_max_batched_py(
    py: Python<'_>,
    open_pass: Bound<'_, PyAny>,
    n_rows: usize,
    n_params: usize,
    y_mean: f64,
    family: &str,
    link: &str,
    l1_ratio: f64,
    fold: usize,
    n_folds: usize,
    seed: u64,
    has_offset: bool,
    var_power: f64,
    theta: f64,
    pen_mask: Option<Vec<bool>>,
    center: Option<PyReadonlyArray1<f64>>,
    scale: Option<PyReadonlyArray1<f64>>,
    ridge_xtx_diag: Option<Vec<f64>>,
    allow_extended_tweedie: bool,
    trace: Option<Bound<'_, pyo3::types::PyList>>,
    max_row_nonzeros: Option<usize>,
    row_kernel: Option<&str>,
    threads: Option<usize>,
) -> PyResult<f64> {
    use rustystats_core::solvers::cv::{fold_alpha_max, AlphaMaxSettings};
    use rustystats_core::solvers::passes::FoldFilter;
    let standardization = build_standardization(center, scale, n_params)?;
    let (null_fit, score_form) = crate::cv_py::alpha_forms(family, link, var_power, theta)?;
    let fam =
        family_from_name_with_tweedie_support(family, var_power, theta, allow_extended_tweedie)?;
    let lnk = link_from_name(link)?;
    let mask = crate::cv_py::penalty_mask(pen_mask, n_params, None)?;
    let mut executor = PyBatchExecutor::new(
        py,
        open_pass,
        n_rows,
        n_params,
        family_from_name_with_tweedie_support(family, var_power, theta, allow_extended_tweedie)?,
        link_from_name(link)?,
        y_mean,
        has_offset,
    )
    .serving(parse_row_kernel(max_row_nonzeros, row_kernel)?, threads)?;
    let result = fold_alpha_max(
        &mut executor,
        &AlphaMaxSettings {
            family: fam.as_ref(),
            link: lnk.as_ref(),
            null_fit,
            score_form,
            l1_ratio,
            penalty_mask: &mask,
            standardization: standardization.as_ref(),
            ridge_diag: ridge_xtx_diag.as_deref(),
            y_mean,
            has_offset,
        },
        FoldFilter {
            fold,
            folds: n_folds,
            seed,
            validation: false,
        },
    );
    crate::cv_py::append_trace(trace.as_ref(), &std::mem::take(&mut executor.trace))?;
    result.map_err(|error| executor.error(error, "alpha_max"))
}

/// Fold `fold`'s path over an alpha grid from batches (§7.9): `open_pass` serves
/// the fold's design for every row; `fold_path_py`'s twin.
#[pyfunction]
#[pyo3(signature = (open_pass, n_rows, n_params, y_mean, family, fold, n_folds, seed, link=None, var_power=1.5, theta=1.0, has_offset=false, weighted=false, alphas=None, l1_ratio=0.0, max_iter=25, tol=1e-8, nonneg_indices=None, nonpos_indices=None, allow_extended_tweedie=false, fit_intercept=true, center=None, scale=None, max_row_nonzeros=None, row_kernel=None, threads=None))]
#[allow(clippy::too_many_arguments)]
pub fn fold_path_batched_py<'py>(
    py: Python<'py>,
    open_pass: Bound<'py, PyAny>,
    n_rows: usize,
    n_params: usize,
    y_mean: f64,
    family: &str,
    fold: usize,
    n_folds: usize,
    seed: u64,
    link: Option<&str>,
    var_power: f64,
    theta: f64,
    has_offset: bool,
    weighted: bool,
    alphas: Option<Vec<f64>>,
    l1_ratio: f64,
    max_iter: usize,
    tol: f64,
    nonneg_indices: Option<Vec<usize>>,
    nonpos_indices: Option<Vec<usize>>,
    allow_extended_tweedie: bool,
    fit_intercept: bool,
    center: Option<PyReadonlyArray1<f64>>,
    scale: Option<PyReadonlyArray1<f64>>,
    max_row_nonzeros: Option<usize>,
    row_kernel: Option<&str>,
    threads: Option<usize>,
) -> PyResult<Py<PyAny>> {
    use rustystats_core::solvers::cv::{fold_path, CvSettings};
    if n_folds < 1 || fold >= n_folds {
        return Err(PyValueError::new_err(format!(
            "fold {fold} is not one of {n_folds} folds"
        )));
    }
    let alphas = alphas.ok_or_else(|| {
        PyValueError::new_err("fold_path_batched_py requires an explicit `alphas=` grid.")
    })?;
    let standardization = build_standardization(center, scale, n_params)?;
    let link_name = link.unwrap_or(default_link_name(family));
    let fam =
        family_from_name_with_tweedie_support(family, var_power, theta, allow_extended_tweedie)?;
    let lnk = link_from_name(link_name)?;
    let mut executor = PyBatchExecutor::new(
        py,
        open_pass,
        n_rows,
        n_params,
        family_from_name_with_tweedie_support(family, var_power, theta, allow_extended_tweedie)?,
        link_from_name(link_name)?,
        y_mean,
        has_offset,
    )
    .serving(parse_row_kernel(max_row_nonzeros, row_kernel)?, threads)?;
    let nonneg = nonneg_indices.unwrap_or_default();
    let nonpos = nonpos_indices.unwrap_or_default();
    let path = fold_path(
        &mut executor,
        &CvSettings {
            family: fam.as_ref(),
            link: lnk.as_ref(),
            alphas: &alphas,
            l1_ratio,
            folds: n_folds,
            seed,
            max_iter,
            tol,
            nonneg: &nonneg,
            nonpos: &nonpos,
            fit_intercept,
            standardization: standardization.as_ref(),
            has_offset,
            weighted,
            validation_form: crate::cv_py::validation_form(fam.as_ref(), lnk.as_ref()),
            solve_workspaces: 1,
        },
        fold,
    );
    let path = path.map_err(|error| executor.error(error, "CV fold path failed"))?;
    let dict = crate::cv_py::fold_path_dict(py, &alphas, &path)?;
    dict.set_item(
        "solver_trace",
        crate::results_py::trace_tuples(&executor.trace),
    )?;
    Ok(dict.unbind().into())
}
