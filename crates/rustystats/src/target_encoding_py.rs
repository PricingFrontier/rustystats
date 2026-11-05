// =============================================================================
// Target Encoding (Ordered Target Statistics)
// =============================================================================
//
// PyO3 wrappers for target encoding, frequency encoding, and interaction
// encoding functions.
// =============================================================================

use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

use rustystats_core::target_encoding;

/// Target encode categorical variables using ordered target statistics.
///
/// This encoding prevents target leakage during training by computing statistics
/// using only "past" observations in a random permutation order.
///
/// Parameters
/// ----------
/// categories : list[str]
///     Categorical values as strings
/// target : numpy.ndarray
///     Target variable (continuous or binary)
/// var_name : str
///     Variable name for output column
/// prior_weight : float, optional
///     Regularization strength toward global mean (default: 1.0)
/// n_permutations : int, optional
///     Number of random permutations to average (default: 4)
/// seed : int, optional
///     Random seed for reproducibility (default: None = random)
///
/// Returns
/// -------
/// tuple[numpy.ndarray, str, dict]
///     (encoded_values, column_name, level_stats)
///     level_stats is a dict mapping level -> (sum_target, count) for prediction
#[pyfunction]
#[pyo3(signature = (categories, target, var_name, prior_weight=1.0, n_permutations=4, seed=None))]
pub fn target_encode_py<'py>(
    py: Python<'py>,
    categories: Vec<String>,
    target: PyReadonlyArray1<f64>,
    var_name: &str,
    prior_weight: f64,
    n_permutations: usize,
    seed: Option<u64>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    String,
    f64,
    std::collections::HashMap<String, (f64, usize)>,
)> {
    let target_vec: Vec<f64> = target.as_array().to_vec();

    let config = target_encoding::TargetEncodingConfig {
        prior_weight,
        n_permutations,
        seed,
    };

    let enc = target_encoding::target_encode(&categories, &target_vec, var_name, &config);

    // Convert level_stats to Python-friendly format
    let stats: std::collections::HashMap<String, (f64, usize)> = enc
        .level_stats
        .into_iter()
        .map(|(k, v)| (k, (v.sum_target, v.count)))
        .collect();

    Ok((enc.values.into_pyarray(py), enc.name, enc.prior, stats))
}

/// Apply target encoding to new data using pre-computed statistics.
///
/// For prediction: uses full training statistics (no ordering needed).
///
/// Parameters
/// ----------
/// categories : list[str]
///     Categorical values for new data
/// level_stats : dict
///     Mapping of level -> (sum_target, count) from training
/// prior : float
///     Global prior (mean of training target)
/// prior_weight : float, optional
///     Prior weight (should match training, default: 1.0)
///
/// Returns
/// -------
/// numpy.ndarray
///     Encoded values for new data
#[pyfunction]
#[pyo3(signature = (categories, level_stats, prior, prior_weight=1.0))]
pub fn apply_target_encoding_py<'py>(
    py: Python<'py>,
    categories: Vec<String>,
    level_stats: std::collections::HashMap<String, (f64, usize)>,
    prior: f64,
    prior_weight: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    // PHASE 1: precompute the encoded value for each unique level once
    // (k entries vs n rows -> avoids n arithmetic ops in the hot loop)
    let encoded_per_level: std::collections::HashMap<&str, f64> = level_stats
        .iter()
        .map(|(level, &(sum_target, count))| {
            let val = (sum_target + prior * prior_weight) / (count as f64 + prior_weight);
            (level.as_str(), val)
        })
        .collect();

    // PHASE 2: per-row lookup (no arithmetic) — parallelized with rayon.
    // Releases the GIL while the parallel work runs.
    let values: Vec<f64> = py.detach(|| {
        categories
            .par_iter()
            .map(|cat| {
                encoded_per_level
                    .get(cat.as_str())
                    .copied()
                    .unwrap_or(prior)
            })
            .collect()
    });

    Ok(Array1::from_vec(values).into_pyarray(py))
}

/// Exposure-weighted target encode categorical variables.
///
/// For frequency models with varying exposure per observation.
/// Uses cumulative claims / cumulative exposure for each category,
/// aligned with actuarial credibility theory.
///
/// Parameters
/// ----------
/// categories : list[str]
///     Categorical values as strings
/// claims : numpy.ndarray
///     Claim counts (target variable)
/// exposure : numpy.ndarray
///     Exposure values (e.g., policy years)
/// var_name : str
///     Variable name for output column
/// prior_weight : float, optional
///     Regularization strength (interpreted as equivalent exposure, default: 1.0)
/// n_permutations : int, optional
///     Number of random permutations to average (default: 4)
/// seed : int, optional
///     Random seed for reproducibility
///
/// Returns
/// -------
/// tuple[numpy.ndarray, str, float, dict]
///     (encoded_values, column_name, prior, level_stats)
///     level_stats is a dict mapping level -> (sum_claims, sum_exposure) for prediction
#[pyfunction]
#[pyo3(signature = (categories, claims, exposure, var_name, prior_weight=1.0, n_permutations=4, seed=None))]
pub fn target_encode_with_exposure_py<'py>(
    py: Python<'py>,
    categories: Vec<String>,
    claims: PyReadonlyArray1<f64>,
    exposure: PyReadonlyArray1<f64>,
    var_name: &str,
    prior_weight: f64,
    n_permutations: usize,
    seed: Option<u64>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    String,
    f64,
    std::collections::HashMap<String, (f64, f64)>,
)> {
    let claims_vec: Vec<f64> = claims.as_array().to_vec();
    let exposure_vec: Vec<f64> = exposure.as_array().to_vec();

    let config = target_encoding::TargetEncodingConfig {
        prior_weight,
        n_permutations,
        seed,
    };

    let enc = target_encoding::target_encode_with_exposure(
        &categories,
        &claims_vec,
        &exposure_vec,
        var_name,
        &config,
    );

    // Convert level_stats to Python-friendly format
    let stats: std::collections::HashMap<String, (f64, f64)> = enc
        .level_stats
        .into_iter()
        .map(|(k, v)| (k, (v.sum_claims, v.sum_exposure)))
        .collect();

    Ok((enc.values.into_pyarray(py), enc.name, enc.prior, stats))
}

/// Apply exposure-weighted target encoding to new data using pre-computed statistics.
///
/// For prediction: uses full training statistics (no ordering needed).
///
/// Parameters
/// ----------
/// categories : list[str]
///     Categorical values for new data
/// level_stats : dict
///     Mapping of level -> (sum_claims, sum_exposure) from training
/// prior : float
///     Global prior (total claims / total exposure from training)
/// prior_weight : float, optional
///     Prior weight (should match training, default: 1.0)
///
/// Returns
/// -------
/// numpy.ndarray
///     Encoded values for new data
#[pyfunction]
#[pyo3(signature = (categories, level_stats, prior, prior_weight=1.0))]
pub fn apply_exposure_weighted_target_encoding_py<'py>(
    py: Python<'py>,
    categories: Vec<String>,
    level_stats: std::collections::HashMap<String, (f64, f64)>,
    prior: f64,
    prior_weight: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    // PHASE 1: precompute the encoded value for each unique level once
    // (k entries vs n rows -> avoids n arithmetic ops in the hot loop)
    let encoded_per_level: std::collections::HashMap<&str, f64> = level_stats
        .iter()
        .map(|(level, &(sum_claims, sum_exposure))| {
            let val = (sum_claims + prior * prior_weight) / (sum_exposure + prior_weight);
            (level.as_str(), val)
        })
        .collect();

    // PHASE 2: per-row lookup (no arithmetic) — parallelized with rayon.
    let values: Vec<f64> = py.detach(|| {
        categories
            .par_iter()
            .map(|cat| {
                encoded_per_level
                    .get(cat.as_str())
                    .copied()
                    .unwrap_or(prior)
            })
            .collect()
    });

    Ok(Array1::from_vec(values).into_pyarray(py))
}

/// Frequency encode categorical variables.
///
/// Encodes categories by their frequency (count / max_count).
/// No target variable involved - purely based on category prevalence.
/// Useful when category frequency itself is predictive.
///
/// Parameters
/// ----------
/// categories : list[str]
///     Categorical values as strings
/// var_name : str
///     Variable name for output column
///
/// Returns
/// -------
/// tuple[numpy.ndarray, str, dict, int, int]
///     (encoded_values, column_name, level_counts, max_count, n_obs)
///     level_counts is a dict mapping level -> count for prediction
#[pyfunction]
pub fn frequency_encode_py<'py>(
    py: Python<'py>,
    categories: Vec<String>,
    var_name: &str,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    String,
    std::collections::HashMap<String, usize>,
    usize,
    usize,
)> {
    let enc = target_encoding::frequency_encode(&categories, var_name);

    Ok((
        enc.values.into_pyarray(py),
        enc.name,
        enc.level_counts,
        enc.max_count,
        enc.n_obs,
    ))
}

/// Apply frequency encoding to new data using pre-computed statistics.
///
/// Parameters
/// ----------
/// categories : list[str]
///     Categorical values for new data
/// level_counts : dict
///     Mapping of level -> count from training
/// max_count : int
///     Maximum count from training (for normalization)
///
/// Returns
/// -------
/// numpy.ndarray
///     Encoded values for new data (unseen categories get 0.0)
#[pyfunction]
pub fn apply_frequency_encoding_py<'py>(
    py: Python<'py>,
    categories: Vec<String>,
    level_counts: std::collections::HashMap<String, usize>,
    max_count: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let values: Vec<f64> = categories
        .iter()
        .map(|cat| {
            let count = level_counts.get(cat).copied().unwrap_or(0);
            count as f64 / max_count as f64
        })
        .collect();

    Ok(Array1::from_vec(values).into_pyarray(py))
}

/// Target encode a categorical interaction (two variables combined).
///
/// Creates combined categories like "brand:region" and applies
/// ordered target statistics encoding.
///
/// Parameters
/// ----------
/// cat1 : list[str]
///     First categorical variable values
/// cat2 : list[str]
///     Second categorical variable values
/// target : numpy.ndarray
///     Target variable
/// var_name1 : str
///     Name of first variable
/// var_name2 : str
///     Name of second variable
/// prior_weight : float, optional
///     Regularization strength (default: 1.0)
/// n_permutations : int, optional
///     Number of permutations (default: 4)
/// seed : int, optional
///     Random seed
///
/// Returns
/// -------
/// tuple[numpy.ndarray, str, float, dict]
///     (encoded_values, column_name, prior, level_stats)
#[pyfunction]
#[pyo3(signature = (cat1, cat2, target, var_name1, var_name2, prior_weight=1.0, n_permutations=4, seed=None))]
pub fn target_encode_interaction_py<'py>(
    py: Python<'py>,
    cat1: Vec<String>,
    cat2: Vec<String>,
    target: PyReadonlyArray1<f64>,
    var_name1: &str,
    var_name2: &str,
    prior_weight: f64,
    n_permutations: usize,
    seed: Option<u64>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    String,
    f64,
    std::collections::HashMap<String, (f64, usize)>,
)> {
    let target_vec: Vec<f64> = target.as_array().to_vec();

    let config = target_encoding::TargetEncodingConfig {
        prior_weight,
        n_permutations,
        seed,
    };

    let enc = target_encoding::target_encode_interaction(
        &cat1,
        &cat2,
        &target_vec,
        var_name1,
        var_name2,
        &config,
    );

    // Convert level_stats to Python-friendly format
    let stats: std::collections::HashMap<String, (f64, usize)> = enc
        .level_stats
        .into_iter()
        .map(|(k, v)| (k, (v.sum_target, v.count)))
        .collect();

    Ok((enc.values.into_pyarray(py), enc.name, enc.prior, stats))
}

/// Target encode a categorical interaction with exposure weighting.
///
/// Creates combined categories like "brand_Nike:region_North" and applies
/// exposure-weighted ordered target statistics encoding using
/// sum(claims) / sum(exposure) per combined level.
///
/// Parameters
/// ----------
/// cat1 : list[str]
///     First categorical variable values
/// cat2 : list[str]
///     Second categorical variable values
/// claims : numpy.ndarray
///     Claim counts (target variable)
/// exposure : numpy.ndarray
///     Exposure values (e.g., policy years)
/// var_name1 : str
///     Name of first variable
/// var_name2 : str
///     Name of second variable
/// prior_weight : float, optional
///     Prior weight for regularization (default: 1.0)
/// n_permutations : int, optional
///     Number of random permutations (default: 4)
/// seed : int, optional
///     Random seed for reproducibility
///
/// Returns
/// -------
/// tuple[numpy.ndarray, str, float, dict]
///     (encoded_values, column_name, prior, level_stats)
///     level_stats is a dict mapping level -> (sum_claims, sum_exposure)
#[pyfunction]
#[pyo3(signature = (cat1, cat2, claims, exposure, var_name1, var_name2, prior_weight=1.0, n_permutations=4, seed=None))]
pub fn target_encode_interaction_with_exposure_py<'py>(
    py: Python<'py>,
    cat1: Vec<String>,
    cat2: Vec<String>,
    claims: PyReadonlyArray1<f64>,
    exposure: PyReadonlyArray1<f64>,
    var_name1: &str,
    var_name2: &str,
    prior_weight: f64,
    n_permutations: usize,
    seed: Option<u64>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    String,
    f64,
    std::collections::HashMap<String, (f64, f64)>,
)> {
    let claims_vec: Vec<f64> = claims.as_array().to_vec();
    let exposure_vec: Vec<f64> = exposure.as_array().to_vec();

    let config = target_encoding::TargetEncodingConfig {
        prior_weight,
        n_permutations,
        seed,
    };

    let enc = target_encoding::target_encode_interaction_with_exposure(
        &cat1,
        &cat2,
        &claims_vec,
        &exposure_vec,
        var_name1,
        var_name2,
        &config,
    );

    // Convert level_stats to Python-friendly format
    let stats: std::collections::HashMap<String, (f64, f64)> = enc
        .level_stats
        .into_iter()
        .map(|(k, v)| (k, (v.sum_claims, v.sum_exposure)))
        .collect();

    Ok((enc.values.into_pyarray(py), enc.name, enc.prior, stats))
}

// =============================================================================
// Keyed-order target encoding for GLM fits (specs/out-of-core-fit.md §6.6)
// =============================================================================

use pyo3::exceptions::PyValueError;
use rustystats_core::error::RustyStatsError;
use rustystats_core::target_encoding::ordered::{
    LevelTotals, PriorSums, SpillStage, TermEncoder, TermSettings, TermValues,
};

/// A core error as a Python exception: spill failures raise `SpillError`.
pub(crate) fn core_error(py: Python<'_>, error: RustyStatsError) -> PyErr {
    if let RustyStatsError::Spill(message) = &error {
        if let Ok(class) = py
            .import("rustystats.exceptions")
            .and_then(|module| module.getattr("SpillError"))
        {
            return PyErr::from_value(
                class
                    .call1((format!("Spill file failed: {message}"),))
                    .unwrap_or_else(|e| e.into_value(py).into_bound(py).into_any()),
            );
        }
    }
    PyValueError::new_err(error.to_string())
}

fn optional_slice<'a, T: numpy::Element>(
    array: &'a Option<PyReadonlyArray1<'_, T>>,
) -> PyResult<Option<&'a [T]>> {
    array
        .as_ref()
        .map(|a| a.as_slice())
        .transpose()
        .map_err(Into::into)
}

/// The prior and "auto" prior weight of a target-encoded term, reduced in blocks
/// of global rows over its participating rows (§5.3).
#[pyclass(name = "TargetEncodingPrior")]
pub struct TargetEncodingPriorPy {
    inner: PriorSums,
}

#[pymethods]
impl TargetEncodingPriorPy {
    #[new]
    fn new(exposure_weighted: bool) -> Self {
        Self {
            inner: PriorSums::new(exposure_weighted),
        }
    }

    /// Add the next rows in global order; rows `include` marks False take no part.
    #[pyo3(signature = (target, exposure=None, include=None))]
    fn update(
        &mut self,
        py: Python<'_>,
        target: PyReadonlyArray1<f64>,
        exposure: Option<PyReadonlyArray1<f64>>,
        include: Option<PyReadonlyArray1<bool>>,
    ) -> PyResult<()> {
        let target = target.as_slice()?;
        let exposure_slice = optional_slice(&exposure)?;
        let include_slice = optional_slice(&include)?;
        self.inner
            .update(target, exposure_slice, include_slice)
            .map_err(|e| core_error(py, e))
    }

    /// The mean target, or total claims over total exposure.
    fn prior(&self) -> f64 {
        self.inner.prior()
    }

    /// The `prior_weight="auto"` value.
    fn auto_prior_weight(&self) -> f64 {
        self.inner.auto_prior_weight()
    }

    /// Participating rows so far.
    #[getter]
    fn rows(&self) -> u64 {
        self.inner.rows()
    }
}

/// Each label's totals over a term's participating rows, added in row order.
#[pyclass(name = "TargetEncodingLevelTotals")]
pub struct TargetEncodingLevelTotalsPy {
    inner: LevelTotals,
}

#[pymethods]
impl TargetEncodingLevelTotalsPy {
    #[new]
    fn new(n_levels: usize, exposure_weighted: bool) -> Self {
        Self {
            inner: LevelTotals::new(n_levels, exposure_weighted),
        }
    }

    /// Add the next rows in row order; rows `include` marks False take no part.
    #[pyo3(signature = (codes, target, exposure=None, include=None))]
    fn update(
        &mut self,
        py: Python<'_>,
        codes: PyReadonlyArray1<i32>,
        target: PyReadonlyArray1<f64>,
        exposure: Option<PyReadonlyArray1<f64>>,
        include: Option<PyReadonlyArray1<bool>>,
    ) -> PyResult<()> {
        let exposure_slice = optional_slice(&exposure)?;
        let include_slice = optional_slice(&include)?;
        self.inner
            .update(
                codes.as_slice()?,
                target.as_slice()?,
                exposure_slice,
                include_slice,
            )
            .map_err(|e| core_error(py, e))
    }

    /// Per label: the target (or claims) sums, the counts and the exposure sums.
    #[allow(clippy::type_complexity)]
    fn totals<'py>(
        &self,
        py: Python<'py>,
    ) -> (
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<u64>>,
        Bound<'py, PyArray1<f64>>,
    ) {
        (
            self.inner.sums.clone().into_pyarray(py),
            self.inner.counts.clone().into_pyarray(py),
            self.inner.exposures.clone().into_pyarray(py),
        )
    }
}

/// A target-encoded term's training values in DEC-1's keyed order, one
/// permutation at a time (§6.6). With `scratch_dir`, every `run_records` records
/// are sorted and spilled as a run, and the values end in a column file; without
/// it every record stays in RAM. Both walk the same order, so their values are
/// identical. Files still on disk are removed by `close()` or when the object goes.
#[pyclass(name = "TargetEncodingTerm", unsendable)]
pub struct TargetEncodingTermPy {
    inner: Option<TermEncoder>,
}

impl TargetEncodingTermPy {
    fn encoder(&mut self) -> PyResult<&mut TermEncoder> {
        self.inner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("the target-encoding term is closed"))
    }
}

/// Run `body` with a progress callback that calls Python's `progress(kind, done,
/// total, start)`; an exception it raises cancels the stage and is returned unchanged.
fn with_progress<T>(
    py: Python<'_>,
    progress: Option<Bound<'_, PyAny>>,
    body: impl FnOnce(
        &mut rustystats_core::target_encoding::ordered::Progress<'_>,
    ) -> Result<T, RustyStatsError>,
) -> PyResult<T> {
    let mut raised: Option<PyErr> = None;
    let result = {
        let mut callback = |stage: SpillStage, done: u64, total: u64, start: bool| {
            if let Some(progress) = &progress {
                if let Err(error) = progress.call1((stage.kind(), done, total, start)) {
                    raised = Some(error);
                    return Err(RustyStatsError::PassAborted(
                        "a progress callback raised".to_string(),
                    ));
                }
            }
            Ok(())
        };
        body(&mut callback)
    };
    match result {
        Ok(value) => Ok(value),
        Err(error) => Err(raised.take().unwrap_or_else(|| core_error(py, error))),
    }
}

#[pymethods]
impl TargetEncodingTermPy {
    #[new]
    #[pyo3(signature = (seed, n_permutations, n_levels, prior, prior_weight, exposure_weighted, scratch_dir=None, run_records=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        seed: u64,
        n_permutations: usize,
        n_levels: usize,
        prior: f64,
        prior_weight: f64,
        exposure_weighted: bool,
        scratch_dir: Option<std::path::PathBuf>,
        run_records: Option<usize>,
    ) -> PyResult<Self> {
        let settings = TermSettings {
            prior,
            prior_weight,
            exposure_weighted,
            n_levels,
        };
        let spill = match (&scratch_dir, run_records) {
            (Some(dir), Some(records)) => Some((dir.as_path(), records)),
            (None, None) => None,
            _ => {
                return Err(PyValueError::new_err(
                    "scratch_dir and run_records are given together",
                ))
            }
        };
        let inner = TermEncoder::new(settings, seed, n_permutations, spill)
            .map_err(|e| core_error(py, e))?;
        Ok(Self { inner: Some(inner) })
    }

    /// Start permutation `permutation` (0, 1, ... in order).
    fn begin(&mut self, py: Python<'_>, permutation: usize) -> PyResult<()> {
        self.encoder()?
            .begin(permutation)
            .map_err(|e| core_error(py, e))
    }

    /// Add global rows `first_row..first_row + len(codes)`: each row's label code,
    /// target and exposure; rows `include` marks False take no part.
    #[pyo3(signature = (first_row, codes, target, exposure=None, include=None))]
    fn push(
        &mut self,
        py: Python<'_>,
        first_row: u64,
        codes: PyReadonlyArray1<i32>,
        target: PyReadonlyArray1<f64>,
        exposure: Option<PyReadonlyArray1<f64>>,
        include: Option<PyReadonlyArray1<bool>>,
    ) -> PyResult<()> {
        let codes = codes.as_slice()?;
        let target = target.as_slice()?;
        let exposure_slice = optional_slice(&exposure)?;
        let include_slice = optional_slice(&include)?;
        let encoder = self.encoder()?;
        py.detach(|| encoder.push(first_row, codes, target, exposure_slice, include_slice))
            .map_err(|e| core_error(py, e))
    }

    /// Finish the open permutation (te_merge and te_walk when spilled);
    /// `progress(kind, records_done, records_total, start)` is called as the stages
    /// run, with `start` True once before each stage.
    #[pyo3(signature = (progress=None))]
    fn end(&mut self, py: Python<'_>, progress: Option<Bound<'_, PyAny>>) -> PyResult<()> {
        let encoder = self.encoder()?;
        with_progress(py, progress, |callback| encoder.end(callback))
    }

    /// Average the permutations (te_average): the values as a float64 array in
    /// memory, or the path of the spilled column file, which the caller now owns.
    #[pyo3(signature = (progress=None))]
    fn finish(
        &mut self,
        py: Python<'_>,
        progress: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let encoder = self
            .inner
            .take()
            .ok_or_else(|| PyValueError::new_err("the target-encoding term is closed"))?;
        let values = with_progress(py, progress, |callback| encoder.finish(callback))?;
        Ok(match values {
            TermValues::Memory(values) => values.into_pyarray(py).into_any().unbind(),
            TermValues::File(path) => path
                .to_string_lossy()
                .into_owned()
                .into_pyobject(py)?
                .into_any()
                .unbind(),
        })
    }

    /// Remove every file the term still owns.
    fn close(&mut self) {
        self.inner = None;
    }
}

/// The DEC-2 fold of each of the global rows `start..start + count`
/// (`mix64(seed ^ mix64(r)) mod folds`), as `fit_cv_path_py` assigns them.
#[pyfunction]
pub fn cv_fold_ids_py(
    py: Python<'_>,
    start: usize,
    count: usize,
    folds: usize,
    seed: u64,
) -> PyResult<Bound<'_, PyArray1<i64>>> {
    if folds == 0 {
        return Err(PyValueError::new_err("folds must be at least 1"));
    }
    let ids: Vec<i64> = (start..start + count)
        .map(|row| rustystats_core::solvers::passes::FoldFilter::fold_of(row, folds, seed) as i64)
        .collect();
    Ok(ids.into_pyarray(py))
}
