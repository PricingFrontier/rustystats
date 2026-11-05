//! Cross-validated regularisation paths and alpha_max (`specs/out-of-core-fit.md`
//! §7.8), on an in-memory design. The batched twins live in `out_of_core_py`; both
//! run `rustystats_core::solvers::cv` over a pass executor, so they agree bit for bit.

use ndarray::Array1;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use rustystats_core::families::Family;
use rustystats_core::links::Link;
use rustystats_core::solvers::cv::{
    alpha_max, cv_path, AlphaMaxSettings, CvPath, CvSettings, NullFit,
};
use rustystats_core::solvers::passes::{
    FoldFilter, InMemoryExecutor, RowContext, ScoreForm, ScoreLink, ScoreVariance, SolverStep,
    ValidationForm,
};

use crate::families_py::{
    default_link_name, family_from_name_with_tweedie_support, link_from_name,
    validate_tweedie_fit_response,
};
use crate::fitting_py::build_standardization;
use crate::out_of_core_py::parse_row_kernel;

/// `regularization_path._alpha_family_base`.
fn alpha_family_base(family: &str) -> String {
    let base = family
        .to_lowercase()
        .split('(')
        .next()
        .unwrap_or("")
        .trim()
        .to_string();
    match base.as_str() {
        "normal" => "gaussian".to_string(),
        "negativebinomial" | "negative_binomial" | "nb" => "negbinomial".to_string(),
        _ => base,
    }
}

/// How alpha_max's null fit and score factor read the family and link.
pub(crate) fn alpha_forms(
    family: &str,
    link: &str,
    var_power: f64,
    theta: f64,
) -> PyResult<(NullFit, ScoreForm)> {
    let base = alpha_family_base(family);
    let score_link = match link {
        "identity" => ScoreLink::Identity,
        "log" => ScoreLink::Log,
        "logit" => ScoreLink::Logit,
        other => {
            return Err(PyValueError::new_err(format!(
                "alpha_max: unsupported link {other:?}"
            )))
        }
    };
    let variance = match base.as_str() {
        "gaussian" => ScoreVariance::Unit,
        "poisson" | "quasipoisson" => ScoreVariance::Mu,
        "gamma" => ScoreVariance::MuSquared,
        "binomial" | "quasibinomial" => ScoreVariance::Binomial,
        "tweedie" => ScoreVariance::Power(var_power),
        "negbinomial" => ScoreVariance::NegativeBinomial(theta),
        other => {
            return Err(PyValueError::new_err(format!(
                "alpha_max: unsupported family {other:?}"
            )))
        }
    };
    let null_fit = match (base.as_str(), link) {
        ("poisson" | "quasipoisson", "log") => NullFit::PoissonLog,
        ("gaussian", "identity") => NullFit::GaussianIdentity,
        _ => NullFit::Irls,
    };
    Ok((
        null_fit,
        ScoreForm {
            link: score_link,
            variance,
        },
    ))
}

/// How a validation fold's deviance is formed for a family and link.
pub(crate) fn validation_form(family: &dyn Family, link: &dyn Link) -> ValidationForm {
    if family.name().eq_ignore_ascii_case("poisson") && link.name() == "log" {
        ValidationForm::PoissonLog
    } else if family.name().eq_ignore_ascii_case("binomial") && link.name() == "logit" {
        ValidationForm::BinomialLogit
    } else {
        ValidationForm::Generic
    }
}

/// A penalty mask from Python's booleans, the intercept column excluded.
pub(crate) fn penalty_mask(
    pen_mask: Option<Vec<bool>>,
    p: usize,
    intercept_col: Option<usize>,
) -> PyResult<Vec<bool>> {
    let mut mask = pen_mask.unwrap_or_else(|| vec![true; p]);
    if mask.len() != p {
        return Err(PyValueError::new_err(format!(
            "pen_mask must be length {p} for alpha_max, got length {}.",
            mask.len()
        )));
    }
    if let Some(col) = intercept_col.filter(|&c| c < p) {
        mask[col] = false;
    }
    Ok(mask)
}

/// The profile dict a CV path reports (`RegularizationPathInfo.cv_profile`).
pub(crate) fn cv_result_dict<'py>(
    py: Python<'py>,
    path: &CvPath,
    n: usize,
    p: usize,
    n_folds: usize,
    seconds: f64,
    fold_assignments: Option<Vec<usize>>,
    threads: usize,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("alphas", path.alphas.clone())?;
    dict.set_item("cv_deviance_mean", path.mean.clone())?;
    dict.set_item("cv_deviance_se", path.se.clone())?;
    dict.set_item("cv_fold_scores", path.fold_scores.clone())?;
    if let Some(folds) = fold_assignments {
        dict.set_item("fold_assignments", folds)?;
    }
    let profile = pyo3::types::PyDict::new(py);
    profile.set_item("n", n)?;
    profile.set_item("p", p)?;
    profile.set_item("n_folds", n_folds)?;
    profile.set_item("n_alphas", path.alphas.len())?;
    profile.set_item("passes", path.passes)?;
    // The threads the row reductions ran on (the requested pool, or the global one).
    profile.set_item("rayon_threads", threads)?;
    profile.set_item("total_wall_seconds", seconds)?;
    profile.set_item("summed_work_seconds", seconds)?;
    let folds = pyo3::types::PyList::empty(py);
    for (fold, (iterations, statuses)) in path.iterations.iter().zip(&path.statuses).enumerate() {
        let entry = pyo3::types::PyDict::new(py);
        entry.set_item("fold", fold)?;
        entry.set_item("iterations", iterations.clone())?;
        entry.set_item("statuses", statuses.clone())?;
        folds.append(entry)?;
    }
    profile.set_item("folds", folds)?;
    dict.set_item("profile", profile)?;
    let best = path
        .mean
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i)
        .unwrap_or(0);
    if let (Some(&alpha), Some(&deviance)) = (path.alphas.get(best), path.mean.get(best)) {
        dict.set_item("best_alpha", alpha)?;
        dict.set_item("best_cv_deviance", deviance)?;
    }
    Ok(dict)
}

fn in_memory_executor<'a>(
    x: ndarray::ArrayView2<'a, f64>,
    sparse_rows: bool,
    family: &'a dyn Family,
    link: &'a dyn Link,
    y: &'a Array1<f64>,
    offset: &'a Array1<f64>,
    weights: &'a Array1<f64>,
    min_weight: f64,
    has_offset: bool,
) -> PyResult<InMemoryExecutor<'a>> {
    let contiguous = |name: &str| PyValueError::new_err(format!("{name} must be contiguous"));
    let ctx = RowContext {
        family,
        link,
        y: y.as_slice().ok_or_else(|| contiguous("y"))?,
        offset: offset.as_slice().ok_or_else(|| contiguous("offset"))?,
        prior_weights: weights.as_slice().ok_or_else(|| contiguous("weights"))?,
        y_mean: rustystats_core::families::response_mean(y),
        min_weight,
        has_offset,
        standardization: None,
    };
    InMemoryExecutor::new(x, sparse_rows, ctx).map_err(|e| PyValueError::new_err(e.to_string()))
}

fn check_lengths(
    n: usize,
    x_rows: usize,
    offset: Option<&Array1<f64>>,
    weights: Option<&Array1<f64>>,
) -> PyResult<()> {
    if x_rows != n {
        return Err(PyValueError::new_err(format!(
            "x has {x_rows} rows but y has length {n}; they must match"
        )));
    }
    for (name, array) in [("offset", offset), ("weights", weights)] {
        if let Some(array) = array {
            if array.len() != n {
                return Err(PyValueError::new_err(format!(
                    "{name} has length {} but y has length {n}; they must match",
                    array.len()
                )));
            }
        }
    }
    Ok(())
}

/// alpha_max (`regularization_path.compute_alpha_max`) on an in-memory design.
#[pyfunction]
#[pyo3(signature = (y, x, family, link, l1_ratio, offset=None, weights=None, var_power=1.5, theta=1.0, pen_mask=None, intercept_col=Some(0), center=None, scale=None, ridge_xtx_diag=None, allow_extended_tweedie=false, trace=None))]
#[allow(clippy::too_many_arguments)]
pub fn alpha_max_py(
    y: PyReadonlyArray1<f64>,
    x: PyReadonlyArray2<f64>,
    family: &str,
    link: &str,
    l1_ratio: f64,
    offset: Option<PyReadonlyArray1<f64>>,
    weights: Option<PyReadonlyArray1<f64>>,
    var_power: f64,
    theta: f64,
    pen_mask: Option<Vec<bool>>,
    intercept_col: Option<usize>,
    center: Option<PyReadonlyArray1<f64>>,
    scale: Option<PyReadonlyArray1<f64>>,
    ridge_xtx_diag: Option<Vec<f64>>,
    allow_extended_tweedie: bool,
    trace: Option<Bound<'_, pyo3::types::PyList>>,
) -> PyResult<f64> {
    let y = y.as_array().to_owned();
    let x = x.as_array();
    let (n, p) = x.dim();
    let offset_array = offset.map(|o| o.as_array().to_owned());
    let weights_array = weights.map(|w| w.as_array().to_owned());
    check_lengths(y.len(), n, offset_array.as_ref(), weights_array.as_ref())?;
    let has_offset = offset_array.is_some();
    let offset = offset_array.unwrap_or_else(|| Array1::zeros(n));
    let weights = weights_array.unwrap_or_else(|| Array1::ones(n));
    let standardization = build_standardization(center, scale, p)?;
    let (null_fit, score_form) = alpha_forms(family, link, var_power, theta)?;
    let fam =
        family_from_name_with_tweedie_support(family, var_power, theta, allow_extended_tweedie)?;
    let lnk = link_from_name(link)?;
    let mask = penalty_mask(pen_mask, p, intercept_col)?;
    let mut executor = in_memory_executor(
        x,
        false,
        fam.as_ref(),
        lnk.as_ref(),
        &y,
        &offset,
        &weights,
        1e-10,
        has_offset,
    )?;
    let y_mean = executor.ctx.y_mean;
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
    append_trace(trace.as_ref(), &executor.take_trace())?;
    result.map_err(|e| PyValueError::new_err(e.to_string()))
}

/// The CV path over an alpha grid on an in-memory design (§7.8): every fold's
/// warm-started fits on its training rows, scored on its validation rows, without
/// copying the design. Fold assignment is DEC-2's hash.
#[pyfunction]
#[pyo3(signature = (y, x, family, link=None, var_power=1.5, theta=1.0, offset=None, weights=None, alphas=None, l1_ratio=0.0, n_folds=5, max_iter=25, tol=1e-8, seed=None, nonneg_indices=None, nonpos_indices=None, allow_extended_tweedie=false, fit_intercept=true, center=None, scale=None, max_row_nonzeros=None, row_kernel=None))]
#[allow(clippy::too_many_arguments)]
pub fn fit_cv_path_py<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<f64>,
    x: PyReadonlyArray2<f64>,
    family: &str,
    link: Option<&str>,
    var_power: f64,
    theta: f64,
    offset: Option<PyReadonlyArray1<f64>>,
    weights: Option<PyReadonlyArray1<f64>>,
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
    max_row_nonzeros: Option<usize>,
    row_kernel: Option<&str>,
) -> PyResult<Py<PyAny>> {
    let start = std::time::Instant::now();
    let y = y.as_array().to_owned();
    let x = x.as_array();
    let (n, p) = x.dim();
    let offset_array = offset.map(|o| o.as_array().to_owned());
    let weights_array = weights.map(|w| w.as_array().to_owned());
    check_lengths(y.len(), n, offset_array.as_ref(), weights_array.as_ref())?;
    if n_folds < 1 {
        return Err(PyValueError::new_err(
            "n_folds must be at least 1".to_string(),
        ));
    }
    if n_folds > n {
        return Err(PyValueError::new_err(format!(
            "n_folds ({n_folds}) cannot exceed the number of observations ({n})"
        )));
    }
    let alphas = alphas.ok_or_else(|| {
        PyValueError::new_err(
            "fit_cv_path_py requires an explicit `alphas=` grid. Build one with \
             rustystats.regularization_path.compute_alpha_max + generate_alpha_path \
             (RS-ACT-005).",
        )
    })?;
    validate_tweedie_fit_response(family, &y, var_power, allow_extended_tweedie)?;
    let standardization = build_standardization(center, scale, p)?;
    let fam =
        family_from_name_with_tweedie_support(family, var_power, theta, allow_extended_tweedie)?;
    let lnk = link_from_name(link.unwrap_or(default_link_name(family)))?;
    let has_offset = offset_array.is_some();
    let weighted = weights_array.is_some();
    let offset = offset_array.unwrap_or_else(|| Array1::zeros(n));
    let weights = weights_array.unwrap_or_else(|| Array1::ones(n));
    let mut executor = in_memory_executor(
        x,
        parse_row_kernel(max_row_nonzeros, row_kernel)?.sparse_in_memory(x),
        fam.as_ref(),
        lnk.as_ref(),
        &y,
        &offset,
        &weights,
        1e-10,
        has_offset,
    )?;
    let seed = seed.unwrap_or(42);
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
            seed,
            max_iter,
            tol,
            nonneg: &nonneg,
            nonpos: &nonpos,
            fit_intercept,
            standardization: standardization.as_ref(),
            has_offset,
            weighted,
            validation_form: validation_form(fam.as_ref(), lnk.as_ref()),
            // In memory every fold may solve at once.
            solve_workspaces: n_folds,
        },
    )
    .map_err(|e| PyValueError::new_err(format!("CV path failed: {e}")))?;
    let fold_assignments: Vec<usize> = (0..n)
        .map(|r| FoldFilter::fold_of(r, n_folds, seed))
        .collect();
    let dict = cv_result_dict(
        py,
        &path,
        n,
        p,
        n_folds,
        start.elapsed().as_secs_f64(),
        Some(fold_assignments),
        rayon::current_num_threads(),
    )?;
    dict.set_item(
        "solver_trace",
        crate::results_py::trace_tuples(&executor.take_trace()),
    )?;
    Ok(dict.unbind().into())
}

/// Append a driver's steps to Python's `trace` list, when one was given (§4.5).
pub(crate) fn append_trace(
    trace: Option<&Bound<'_, pyo3::types::PyList>>,
    steps: &[SolverStep],
) -> PyResult<()> {
    if let Some(list) = trace {
        for step in crate::results_py::trace_tuples(steps) {
            list.append(step)?;
        }
    }
    Ok(())
}

/// A fold path's dict: per alpha, the fold's validation deviance (INF for a failed
/// fit), iterations and solver status, and the passes it ran.
pub(crate) fn fold_path_dict<'py>(
    py: Python<'py>,
    alphas: &[f64],
    path: &rustystats_core::solvers::cv::FoldPath,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("alphas", alphas.to_vec())?;
    dict.set_item("fold_deviances", path.scores.clone())?;
    dict.set_item("iterations", path.iterations.clone())?;
    dict.set_item("statuses", path.statuses.clone())?;
    dict.set_item("passes", path.passes)?;
    Ok(dict)
}

/// alpha_max over fold `fold`'s training rows of an in-memory design (§7.9): `x`
/// is the fold's design for every row, rows assigned to folds by DEC-2's hash.
#[pyfunction]
#[pyo3(signature = (y, x, family, link, l1_ratio, fold, n_folds, seed, offset=None, weights=None, var_power=1.5, theta=1.0, pen_mask=None, center=None, scale=None, ridge_xtx_diag=None, allow_extended_tweedie=false, trace=None))]
#[allow(clippy::too_many_arguments)]
pub fn fold_alpha_max_py(
    y: PyReadonlyArray1<f64>,
    x: PyReadonlyArray2<f64>,
    family: &str,
    link: &str,
    l1_ratio: f64,
    fold: usize,
    n_folds: usize,
    seed: u64,
    offset: Option<PyReadonlyArray1<f64>>,
    weights: Option<PyReadonlyArray1<f64>>,
    var_power: f64,
    theta: f64,
    pen_mask: Option<Vec<bool>>,
    center: Option<PyReadonlyArray1<f64>>,
    scale: Option<PyReadonlyArray1<f64>>,
    ridge_xtx_diag: Option<Vec<f64>>,
    allow_extended_tweedie: bool,
    trace: Option<Bound<'_, pyo3::types::PyList>>,
) -> PyResult<f64> {
    use rustystats_core::solvers::cv::fold_alpha_max;
    let y = y.as_array().to_owned();
    let x = x.as_array();
    let (n, p) = x.dim();
    let offset_array = offset.map(|o| o.as_array().to_owned());
    let weights_array = weights.map(|w| w.as_array().to_owned());
    check_lengths(y.len(), n, offset_array.as_ref(), weights_array.as_ref())?;
    let has_offset = offset_array.is_some();
    let offset = offset_array.unwrap_or_else(|| Array1::zeros(n));
    let weights = weights_array.unwrap_or_else(|| Array1::ones(n));
    let standardization = build_standardization(center, scale, p)?;
    let (null_fit, score_form) = alpha_forms(family, link, var_power, theta)?;
    let fam =
        family_from_name_with_tweedie_support(family, var_power, theta, allow_extended_tweedie)?;
    let lnk = link_from_name(link)?;
    let mask = penalty_mask(pen_mask, p, None)?;
    let mut executor = in_memory_executor(
        x,
        false,
        fam.as_ref(),
        lnk.as_ref(),
        &y,
        &offset,
        &weights,
        1e-10,
        has_offset,
    )?;
    let y_mean = executor.ctx.y_mean;
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
    append_trace(trace.as_ref(), &executor.take_trace())?;
    result.map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Fold `fold`'s path over an alpha grid on an in-memory design (§7.9): `x` is
/// the fold's design for every row; the fold fits its training rows and scores
/// its validation rows (DEC-2's hash), as one fold of `fit_cv_path_py`.
#[pyfunction]
#[pyo3(signature = (y, x, family, fold, n_folds, seed, link=None, var_power=1.5, theta=1.0, offset=None, weights=None, alphas=None, l1_ratio=0.0, max_iter=25, tol=1e-8, nonneg_indices=None, nonpos_indices=None, allow_extended_tweedie=false, fit_intercept=true, center=None, scale=None, max_row_nonzeros=None, row_kernel=None))]
#[allow(clippy::too_many_arguments)]
pub fn fold_path_py<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<f64>,
    x: PyReadonlyArray2<f64>,
    family: &str,
    fold: usize,
    n_folds: usize,
    seed: u64,
    link: Option<&str>,
    var_power: f64,
    theta: f64,
    offset: Option<PyReadonlyArray1<f64>>,
    weights: Option<PyReadonlyArray1<f64>>,
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
) -> PyResult<Py<PyAny>> {
    use rustystats_core::solvers::cv::fold_path;
    let y = y.as_array().to_owned();
    let x = x.as_array();
    let (n, p) = x.dim();
    let offset_array = offset.map(|o| o.as_array().to_owned());
    let weights_array = weights.map(|w| w.as_array().to_owned());
    check_lengths(y.len(), n, offset_array.as_ref(), weights_array.as_ref())?;
    if n_folds < 1 || fold >= n_folds {
        return Err(PyValueError::new_err(format!(
            "fold {fold} is not one of {n_folds} folds"
        )));
    }
    let alphas = alphas
        .ok_or_else(|| PyValueError::new_err("fold_path_py requires an explicit `alphas=` grid"))?;
    let standardization = build_standardization(center, scale, p)?;
    let fam =
        family_from_name_with_tweedie_support(family, var_power, theta, allow_extended_tweedie)?;
    let lnk = link_from_name(link.unwrap_or(default_link_name(family)))?;
    let has_offset = offset_array.is_some();
    let weighted = weights_array.is_some();
    let offset = offset_array.unwrap_or_else(|| Array1::zeros(n));
    let weights = weights_array.unwrap_or_else(|| Array1::ones(n));
    let mut executor = in_memory_executor(
        x,
        parse_row_kernel(max_row_nonzeros, row_kernel)?.sparse_in_memory(x),
        fam.as_ref(),
        lnk.as_ref(),
        &y,
        &offset,
        &weights,
        1e-10,
        has_offset,
    )?;
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
            validation_form: validation_form(fam.as_ref(), lnk.as_ref()),
            solve_workspaces: 1,
        },
        fold,
    )
    .map_err(|e| PyValueError::new_err(format!("CV fold path failed: {e}")))?;
    let dict = fold_path_dict(py, &alphas, &path)?;
    dict.set_item(
        "solver_trace",
        crate::results_py::trace_tuples(&executor.take_trace()),
    )?;
    Ok(dict.unbind().into())
}
