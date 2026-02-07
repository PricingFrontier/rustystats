// =============================================================================
// Target Encoding (Ordered Target Statistics)
// =============================================================================
//
// PyO3 wrappers for target encoding, frequency encoding, and interaction
// encoding functions.
// =============================================================================

use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use ndarray::Array1;

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
) -> PyResult<(Bound<'py, PyArray1<f64>>, String, f64, std::collections::HashMap<String, (f64, usize)>)> {
    let target_vec: Vec<f64> = target.as_array().to_vec();
    
    let config = target_encoding::TargetEncodingConfig {
        prior_weight,
        n_permutations,
        seed,
    };
    
    let enc = target_encoding::target_encode(&categories, &target_vec, var_name, &config);
    
    // Convert level_stats to Python-friendly format
    let stats: std::collections::HashMap<String, (f64, usize)> = enc.level_stats
        .into_iter()
        .map(|(k, v)| (k, (v.sum_target, v.count)))
        .collect();
    
    Ok((
        enc.values.into_pyarray_bound(py),
        enc.name,
        enc.prior,
        stats,
    ))
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
    let n = categories.len();
    let mut values = Vec::with_capacity(n);
    
    for cat in &categories {
        let encoded = if let Some(&(sum_target, count)) = level_stats.get(cat) {
            (sum_target + prior * prior_weight) / (count as f64 + prior_weight)
        } else {
            // Unseen category: use prior
            prior
        };
        values.push(encoded);
    }
    
    Ok(Array1::from_vec(values).into_pyarray_bound(py))
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
) -> PyResult<(Bound<'py, PyArray1<f64>>, String, f64, std::collections::HashMap<String, (f64, f64)>)> {
    let claims_vec: Vec<f64> = claims.as_array().to_vec();
    let exposure_vec: Vec<f64> = exposure.as_array().to_vec();
    
    let config = target_encoding::TargetEncodingConfig {
        prior_weight,
        n_permutations,
        seed,
    };
    
    let enc = target_encoding::target_encode_with_exposure(&categories, &claims_vec, &exposure_vec, var_name, &config);
    
    // Convert level_stats to Python-friendly format
    let stats: std::collections::HashMap<String, (f64, f64)> = enc.level_stats
        .into_iter()
        .map(|(k, v)| (k, (v.sum_claims, v.sum_exposure)))
        .collect();
    
    Ok((
        enc.values.into_pyarray_bound(py),
        enc.name,
        enc.prior,
        stats,
    ))
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
    let n = categories.len();
    let mut values = Vec::with_capacity(n);
    
    for cat in &categories {
        let encoded = if let Some(&(sum_claims, sum_exposure)) = level_stats.get(cat) {
            (sum_claims + prior * prior_weight) / (sum_exposure + prior_weight)
        } else {
            // Unseen category: use prior
            prior
        };
        values.push(encoded);
    }
    
    Ok(Array1::from_vec(values).into_pyarray_bound(py))
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
) -> PyResult<(Bound<'py, PyArray1<f64>>, String, std::collections::HashMap<String, usize>, usize, usize)> {
    let enc = target_encoding::frequency_encode(&categories, var_name);
    
    Ok((
        enc.values.into_pyarray_bound(py),
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
    
    Ok(Array1::from_vec(values).into_pyarray_bound(py))
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
) -> PyResult<(Bound<'py, PyArray1<f64>>, String, f64, std::collections::HashMap<String, (f64, usize)>)> {
    let target_vec: Vec<f64> = target.as_array().to_vec();
    
    let config = target_encoding::TargetEncodingConfig {
        prior_weight,
        n_permutations,
        seed,
    };
    
    let enc = target_encoding::target_encode_interaction(&cat1, &cat2, &target_vec, var_name1, var_name2, &config);
    
    // Convert level_stats to Python-friendly format
    let stats: std::collections::HashMap<String, (f64, usize)> = enc.level_stats
        .into_iter()
        .map(|(k, v)| (k, (v.sum_target, v.count)))
        .collect();
    
    Ok((
        enc.values.into_pyarray_bound(py),
        enc.name,
        enc.prior,
        stats,
    ))
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
) -> PyResult<(Bound<'py, PyArray1<f64>>, String, f64, std::collections::HashMap<String, (f64, f64)>)> {
    let claims_vec: Vec<f64> = claims.as_array().to_vec();
    let exposure_vec: Vec<f64> = exposure.as_array().to_vec();
    
    let config = target_encoding::TargetEncodingConfig {
        prior_weight,
        n_permutations,
        seed,
    };
    
    let enc = target_encoding::target_encode_interaction_with_exposure(
        &cat1, &cat2, &claims_vec, &exposure_vec, var_name1, var_name2, &config
    );
    
    // Convert level_stats to Python-friendly format
    let stats: std::collections::HashMap<String, (f64, f64)> = enc.level_stats
        .into_iter()
        .map(|(k, v)| (k, (v.sum_claims, v.sum_exposure)))
        .collect();
    
    Ok((
        enc.values.into_pyarray_bound(py),
        enc.name,
        enc.prior,
        stats,
    ))
}
