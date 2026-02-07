// =============================================================================
// TARGET ENCODING (ORDERED TARGET STATISTICS)
// =============================================================================
//
// Implements CatBoost's ordered target statistics for categorical encoding.
// This prevents target leakage during training by using only "past" observations
// in the permutation order to compute statistics.
//
// Reference: https://arxiv.org/abs/1706.09516 (CatBoost paper)
//
// =============================================================================

use ndarray::Array1;
use rayon::prelude::*;
use std::collections::HashMap;

/// Result of target encoding
#[derive(Debug, Clone)]
pub struct TargetEncoding {
    /// Encoded values (one per observation)
    pub values: Array1<f64>,
    /// Column name
    pub name: String,
    /// Unique levels (sorted)
    pub levels: Vec<String>,
    /// Statistics for each level (for prediction on new data)
    pub level_stats: HashMap<String, LevelStatistics>,
    /// Global prior (mean of target)
    pub prior: f64,
}

/// Statistics for a single categorical level
#[derive(Debug, Clone)]
pub struct LevelStatistics {
    /// Sum of target values for this level
    pub sum_target: f64,
    /// Count of observations with this level
    pub count: usize,
}

impl LevelStatistics {
    /// Compute encoded value using these statistics
    pub fn encode(&self, prior: f64, prior_weight: f64) -> f64 {
        (self.sum_target + prior * prior_weight) / (self.count as f64 + prior_weight)
    }
}

/// Statistics for a single categorical level (exposure-weighted version)
#[derive(Debug, Clone)]
pub struct ExposureWeightedLevelStatistics {
    /// Sum of claims (target) for this level
    pub sum_claims: f64,
    /// Sum of exposure for this level
    pub sum_exposure: f64,
}

impl ExposureWeightedLevelStatistics {
    /// Compute encoded value using these statistics
    /// prior_weight is interpreted as "equivalent exposure"
    pub fn encode(&self, prior: f64, prior_weight: f64) -> f64 {
        (self.sum_claims + prior * prior_weight) / (self.sum_exposure + prior_weight)
    }
}

/// Result of exposure-weighted target encoding
#[derive(Debug, Clone)]
pub struct ExposureWeightedTargetEncoding {
    /// Encoded values (one per observation)
    pub values: Array1<f64>,
    /// Column name
    pub name: String,
    /// Unique levels (sorted)
    pub levels: Vec<String>,
    /// Statistics for each level (for prediction on new data)
    pub level_stats: HashMap<String, ExposureWeightedLevelStatistics>,
    /// Global prior (total claims / total exposure)
    pub prior: f64,
}

/// Configuration for target encoding
#[derive(Debug, Clone)]
pub struct TargetEncodingConfig {
    /// Weight for the prior (regularization strength)
    /// Higher values = more regularization toward global mean
    pub prior_weight: f64,
    /// Number of random permutations to average (reduces variance)
    pub n_permutations: usize,
    /// Random seed for reproducibility (None = random)
    pub seed: Option<u64>,
}

impl Default for TargetEncodingConfig {
    fn default() -> Self {
        Self {
            prior_weight: 1.0,
            n_permutations: 4,
            seed: None,
        }
    }
}

/// Compute ordered target statistics encoding.
///
/// For training data: uses ordered statistics to prevent target leakage.
/// Multiple random permutations are averaged to reduce variance.
///
/// # Algorithm
/// For each observation i in permutation order:
/// ```text
/// encoded[i] = (sum_target_before[category] + prior * prior_weight) / (count_before[category] + prior_weight)
/// ```
///
/// # Arguments
/// * `categories` - Categorical values as strings
/// * `target` - Target variable (continuous or binary)
/// * `var_name` - Variable name for output column
/// * `config` - Encoding configuration
///
/// # Returns
/// TargetEncoding with encoded values and statistics for prediction
pub fn target_encode(
    categories: &[String],
    target: &[f64],
    var_name: &str,
    config: &TargetEncodingConfig,
) -> TargetEncoding {
    let n = categories.len();
    assert_eq!(n, target.len(), "categories and target must have same length");
    
    // Compute global prior (mean of target)
    let prior: f64 = target.iter().sum::<f64>() / n as f64;
    
    // Get unique levels
    let mut levels: Vec<String> = categories.iter().cloned().collect();
    levels.sort();
    levels.dedup();
    
    // Create level-to-index mapping
    let level_map: HashMap<&str, usize> = levels
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i))
        .collect();
    
    // Convert categories to indices
    let cat_indices: Vec<usize> = categories
        .iter()
        .map(|c| *level_map.get(c.as_str()).unwrap_or(&0))
        .collect();
    
    // Compute full statistics for each level (for prediction)
    let mut level_stats = HashMap::new();
    let mut sum_by_level = vec![0.0; levels.len()];
    let mut count_by_level = vec![0usize; levels.len()];
    
    for i in 0..n {
        let idx = cat_indices[i];
        sum_by_level[idx] += target[i];
        count_by_level[idx] += 1;
    }
    
    for (i, level) in levels.iter().enumerate() {
        level_stats.insert(level.clone(), LevelStatistics {
            sum_target: sum_by_level[i],
            count: count_by_level[i],
        });
    }
    
    // Compute ordered target statistics with multiple permutations
    let encoded_values = if config.n_permutations > 1 {
        // Average over multiple permutations (parallel)
        let permutation_results: Vec<Vec<f64>> = (0..config.n_permutations)
            .into_par_iter()
            .map(|perm_idx| {
                let seed = config.seed.map(|s| s + perm_idx as u64);
                compute_ordered_target_stats(
                    &cat_indices,
                    target,
                    levels.len(),
                    prior,
                    config.prior_weight,
                    seed,
                )
            })
            .collect();
        
        // Average the results
        let mut averaged = vec![0.0; n];
        for i in 0..n {
            let sum: f64 = permutation_results.iter().map(|r| r[i]).sum();
            averaged[i] = sum / config.n_permutations as f64;
        }
        averaged
    } else {
        compute_ordered_target_stats(
            &cat_indices,
            target,
            levels.len(),
            prior,
            config.prior_weight,
            config.seed,
        )
    };
    
    TargetEncoding {
        values: Array1::from_vec(encoded_values),
        name: format!("TE({})", var_name),
        levels,
        level_stats,
        prior,
    }
}

/// Compute ordered target statistics for a single permutation.
fn compute_ordered_target_stats(
    cat_indices: &[usize],
    target: &[f64],
    n_levels: usize,
    prior: f64,
    prior_weight: f64,
    seed: Option<u64>,
) -> Vec<f64> {
    let n = cat_indices.len();
    
    // Generate random permutation
    let permutation = generate_permutation(n, seed);
    
    // Track running statistics for each level
    let mut sum_by_level = vec![0.0; n_levels];
    let mut count_by_level = vec![0usize; n_levels];
    
    // Compute encoded values in permutation order
    let mut encoded = vec![0.0; n];
    
    for &perm_idx in &permutation {
        let cat_idx = cat_indices[perm_idx];
        
        // Encode using ONLY observations seen so far (before current in permutation)
        let sum_before = sum_by_level[cat_idx];
        let count_before = count_by_level[cat_idx];
        
        encoded[perm_idx] = (sum_before + prior * prior_weight) / (count_before as f64 + prior_weight);
        
        // Update running statistics with current observation
        sum_by_level[cat_idx] += target[perm_idx];
        count_by_level[cat_idx] += 1;
    }
    
    encoded
}

/// Generate a random permutation of indices [0, n).
fn generate_permutation(n: usize, seed: Option<u64>) -> Vec<usize> {
    
    let mut indices: Vec<usize> = (0..n).collect();
    
    // Simple LCG-based shuffle (deterministic if seed provided)
    let mut state: u64 = seed.unwrap_or_else(|| {
        // Use system time for random seed
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(42)
    });
    
    // Fisher-Yates shuffle
    for i in (1..n).rev() {
        // LCG: state = (a * state + c) mod m
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let j = (state as usize) % (i + 1);
        indices.swap(i, j);
    }
    
    indices
}

// =============================================================================
// EXPOSURE-WEIGHTED TARGET ENCODING
// =============================================================================
//
// For frequency/severity modeling where exposure varies by observation.
// Uses cumulative claims / cumulative exposure instead of mean of individual rates.
// This gives exposure-weighted estimates aligned with actuarial credibility theory.
//
// =============================================================================

/// Compute exposure-weighted ordered target statistics encoding.
///
/// For frequency models with varying exposure per observation.
/// Uses cumulative claims / cumulative exposure for each category.
///
/// # Algorithm
/// For each observation i in permutation order:
/// ```text
/// encoded[i] = (sum_claims_before[category] + prior * prior_weight) / (sum_exposure_before[category] + prior_weight)
/// ```
///
/// # Arguments
/// * `categories` - Categorical values as strings
/// * `claims` - Claim counts (target variable)
/// * `exposure` - Exposure values (e.g., policy years)
/// * `var_name` - Variable name for output column
/// * `config` - Encoding configuration (prior_weight is interpreted as "equivalent exposure")
///
/// # Returns
/// ExposureWeightedTargetEncoding with encoded values and statistics for prediction
pub fn target_encode_with_exposure(
    categories: &[String],
    claims: &[f64],
    exposure: &[f64],
    var_name: &str,
    config: &TargetEncodingConfig,
) -> ExposureWeightedTargetEncoding {
    let n = categories.len();
    assert_eq!(n, claims.len(), "categories and claims must have same length");
    assert_eq!(n, exposure.len(), "categories and exposure must have same length");
    
    // Compute global prior (total claims / total exposure)
    let total_claims: f64 = claims.iter().sum();
    let total_exposure: f64 = exposure.iter().sum();
    let prior = total_claims / total_exposure.max(1e-10);
    
    // Get unique levels
    let mut levels: Vec<String> = categories.iter().cloned().collect();
    levels.sort();
    levels.dedup();
    
    // Create level-to-index mapping
    let level_map: HashMap<&str, usize> = levels
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i))
        .collect();
    
    // Convert categories to indices
    let cat_indices: Vec<usize> = categories
        .iter()
        .map(|c| *level_map.get(c.as_str()).unwrap_or(&0))
        .collect();
    
    // Compute full statistics for each level (for prediction)
    let mut level_stats = HashMap::new();
    let mut claims_by_level = vec![0.0; levels.len()];
    let mut exposure_by_level = vec![0.0; levels.len()];
    
    for i in 0..n {
        let idx = cat_indices[i];
        claims_by_level[idx] += claims[i];
        exposure_by_level[idx] += exposure[i];
    }
    
    for (i, level) in levels.iter().enumerate() {
        level_stats.insert(level.clone(), ExposureWeightedLevelStatistics {
            sum_claims: claims_by_level[i],
            sum_exposure: exposure_by_level[i],
        });
    }
    
    // Compute ordered target statistics with multiple permutations
    let encoded_values = if config.n_permutations > 1 {
        // Average over multiple permutations (parallel)
        let permutation_results: Vec<Vec<f64>> = (0..config.n_permutations)
            .into_par_iter()
            .map(|perm_idx| {
                let seed = config.seed.map(|s| s + perm_idx as u64);
                compute_ordered_exposure_weighted_stats(
                    &cat_indices,
                    claims,
                    exposure,
                    levels.len(),
                    prior,
                    config.prior_weight,
                    seed,
                )
            })
            .collect();
        
        // Average the results
        let mut averaged = vec![0.0; n];
        for i in 0..n {
            let sum: f64 = permutation_results.iter().map(|r| r[i]).sum();
            averaged[i] = sum / config.n_permutations as f64;
        }
        averaged
    } else {
        compute_ordered_exposure_weighted_stats(
            &cat_indices,
            claims,
            exposure,
            levels.len(),
            prior,
            config.prior_weight,
            config.seed,
        )
    };
    
    ExposureWeightedTargetEncoding {
        values: Array1::from_vec(encoded_values),
        name: format!("TE({})", var_name),
        levels,
        level_stats,
        prior,
    }
}

/// Compute exposure-weighted ordered target statistics for a single permutation.
fn compute_ordered_exposure_weighted_stats(
    cat_indices: &[usize],
    claims: &[f64],
    exposure: &[f64],
    n_levels: usize,
    prior: f64,
    prior_weight: f64,
    seed: Option<u64>,
) -> Vec<f64> {
    let n = cat_indices.len();
    
    // Generate random permutation
    let permutation = generate_permutation(n, seed);
    
    // Track running statistics for each level (claims and exposure)
    let mut claims_by_level = vec![0.0; n_levels];
    let mut exposure_by_level = vec![0.0; n_levels];
    
    // Compute encoded values in permutation order
    let mut encoded = vec![0.0; n];
    
    for &perm_idx in &permutation {
        let cat_idx = cat_indices[perm_idx];
        
        // Encode using ONLY observations seen so far (before current in permutation)
        let claims_before = claims_by_level[cat_idx];
        let exposure_before = exposure_by_level[cat_idx];
        
        // Exposure-weighted encoding: (sum_claims + prior * prior_weight) / (sum_exposure + prior_weight)
        encoded[perm_idx] = (claims_before + prior * prior_weight) / (exposure_before + prior_weight);
        
        // Update running statistics with current observation
        claims_by_level[cat_idx] += claims[perm_idx];
        exposure_by_level[cat_idx] += exposure[perm_idx];
    }
    
    encoded
}

/// Apply exposure-weighted target encoding to new data using pre-computed statistics.
///
/// For prediction: uses full training statistics (no ordering needed).
///
/// # Arguments
/// * `categories` - Categorical values for new data
/// * `encoding` - ExposureWeightedTargetEncoding from training data
/// * `prior_weight` - Prior weight (should match training, interpreted as equivalent exposure)
///
/// # Returns
/// Encoded values for new data
pub fn apply_exposure_weighted_target_encoding(
    categories: &[String],
    encoding: &ExposureWeightedTargetEncoding,
    prior_weight: f64,
) -> Array1<f64> {
    let n = categories.len();
    let mut values = Vec::with_capacity(n);
    
    for cat in categories {
        let encoded = if let Some(stats) = encoding.level_stats.get(cat) {
            stats.encode(encoding.prior, prior_weight)
        } else {
            // Unseen category: use prior
            encoding.prior
        };
        values.push(encoded);
    }
    
    Array1::from_vec(values)
}

/// Apply target encoding to new data using pre-computed statistics.
///
/// For prediction: uses full training statistics (no ordering needed).
///
/// # Arguments
/// * `categories` - Categorical values for new data
/// * `encoding` - TargetEncoding from training data
/// * `prior_weight` - Prior weight (should match training)
///
/// # Returns
/// Encoded values for new data
pub fn apply_target_encoding(
    categories: &[String],
    encoding: &TargetEncoding,
    prior_weight: f64,
) -> Array1<f64> {
    let n = categories.len();
    let mut values = Vec::with_capacity(n);
    
    for cat in categories {
        let encoded = if let Some(stats) = encoding.level_stats.get(cat) {
            stats.encode(encoding.prior, prior_weight)
        } else {
            // Unseen category: use prior
            encoding.prior
        };
        values.push(encoded);
    }
    
    Array1::from_vec(values)
}

// =============================================================================
// FREQUENCY ENCODING
// =============================================================================
//
// Encodes categorical variables by their frequency (count/max_count).
// No target variable involved - purely based on category prevalence.
// Useful when category frequency itself is predictive.
//
// =============================================================================

/// Result of frequency encoding
#[derive(Debug, Clone)]
pub struct FrequencyEncoding {
    /// Encoded values (one per observation)
    pub values: Array1<f64>,
    /// Column name
    pub name: String,
    /// Unique levels (sorted)
    pub levels: Vec<String>,
    /// Count for each level (for prediction on new data)
    pub level_counts: HashMap<String, usize>,
    /// Maximum count (for normalization)
    pub max_count: usize,
    /// Total number of observations
    pub n_obs: usize,
}

/// Frequency encode categorical variables.
///
/// Each category is encoded as: count(category) / max_count
/// This gives values in (0, 1] where 1.0 is the most frequent category.
///
/// # Arguments
/// * `categories` - Categorical values as strings
/// * `var_name` - Variable name for output column
///
/// # Returns
/// FrequencyEncoding with encoded values and statistics for prediction
pub fn frequency_encode(
    categories: &[String],
    var_name: &str,
) -> FrequencyEncoding {
    let n = categories.len();
    
    // Count occurrences of each level
    let mut level_counts: HashMap<String, usize> = HashMap::new();
    for cat in categories {
        *level_counts.entry(cat.clone()).or_insert(0) += 1;
    }
    
    // Find maximum count for normalization
    let max_count = level_counts.values().copied().max().unwrap_or(1);
    
    // Get sorted unique levels
    let mut levels: Vec<String> = level_counts.keys().cloned().collect();
    levels.sort();
    
    // Encode values (parallel for large data)
    let values: Vec<f64> = if n > 10000 {
        categories
            .par_iter()
            .map(|cat| {
                let count = level_counts.get(cat).copied().unwrap_or(0);
                count as f64 / max_count as f64
            })
            .collect()
    } else {
        categories
            .iter()
            .map(|cat| {
                let count = level_counts.get(cat).copied().unwrap_or(0);
                count as f64 / max_count as f64
            })
            .collect()
    };
    
    FrequencyEncoding {
        values: Array1::from_vec(values),
        name: format!("FE({})", var_name),
        levels,
        level_counts,
        max_count,
        n_obs: n,
    }
}

/// Apply frequency encoding to new data using pre-computed statistics.
///
/// # Arguments
/// * `categories` - Categorical values for new data
/// * `encoding` - FrequencyEncoding from training data
///
/// # Returns
/// Encoded values for new data (unseen categories get 0.0)
pub fn apply_frequency_encoding(
    categories: &[String],
    encoding: &FrequencyEncoding,
) -> Array1<f64> {
    let max_count = encoding.max_count;
    
    let values: Vec<f64> = categories
        .iter()
        .map(|cat| {
            let count = encoding.level_counts.get(cat).copied().unwrap_or(0);
            count as f64 / max_count as f64
        })
        .collect();
    
    Array1::from_vec(values)
}

// =============================================================================
// TARGET ENCODING FOR INTERACTIONS
// =============================================================================
//
// Encodes categorical interactions (e.g., brand:region) as combined categories.
// Uses the same ordered target statistics as single-variable encoding.
//
// =============================================================================

/// Target encode a categorical interaction (two variables combined).
///
/// Creates combined categories like "brand_Nike:region_North" and applies
/// ordered target statistics encoding.
///
/// # Arguments
/// * `cat1` - First categorical variable values
/// * `cat2` - Second categorical variable values
/// * `target` - Target variable
/// * `var_name1` - Name of first variable
/// * `var_name2` - Name of second variable
/// * `config` - Encoding configuration
///
/// # Returns
/// TargetEncoding with encoded values for the interaction
pub fn target_encode_interaction(
    cat1: &[String],
    cat2: &[String],
    target: &[f64],
    var_name1: &str,
    var_name2: &str,
    config: &TargetEncodingConfig,
) -> TargetEncoding {
    let n = cat1.len();
    assert_eq!(n, cat2.len(), "cat1 and cat2 must have same length");
    assert_eq!(n, target.len(), "categories and target must have same length");
    
    // Create combined categories
    let combined: Vec<String> = cat1
        .iter()
        .zip(cat2.iter())
        .map(|(a, b)| format!("{}:{}", a, b))
        .collect();
    
    // Apply standard target encoding to combined categories
    let var_name = format!("{}:{}", var_name1, var_name2);
    target_encode(&combined, target, &var_name, config)
}

/// Target encode a categorical interaction with exposure weighting.
///
/// Creates combined categories like "brand_Nike:region_North" and applies
/// exposure-weighted ordered target statistics encoding.
///
/// # Arguments
/// * `cat1` - First categorical variable values
/// * `cat2` - Second categorical variable values
/// * `claims` - Claim counts (target variable)
/// * `exposure` - Exposure values (e.g., policy years)
/// * `var_name1` - Name of first variable
/// * `var_name2` - Name of second variable
/// * `config` - Encoding configuration
///
/// # Returns
/// ExposureWeightedTargetEncoding with encoded values for the interaction
pub fn target_encode_interaction_with_exposure(
    cat1: &[String],
    cat2: &[String],
    claims: &[f64],
    exposure: &[f64],
    var_name1: &str,
    var_name2: &str,
    config: &TargetEncodingConfig,
) -> ExposureWeightedTargetEncoding {
    let n = cat1.len();
    assert_eq!(n, cat2.len(), "cat1 and cat2 must have same length");
    assert_eq!(n, claims.len(), "categories and claims must have same length");
    assert_eq!(n, exposure.len(), "categories and exposure must have same length");
    
    // Create combined categories
    let combined: Vec<String> = cat1
        .iter()
        .zip(cat2.iter())
        .map(|(a, b)| format!("{}:{}", a, b))
        .collect();
    
    // Apply exposure-weighted target encoding to combined categories
    let var_name = format!("{}:{}", var_name1, var_name2);
    target_encode_with_exposure(&combined, claims, exposure, &var_name, config)
}

/// Target encode a multi-way categorical interaction.
///
/// Creates combined categories from multiple variables.
///
/// # Arguments
/// * `categories` - Vector of categorical variable values (each inner vec is one variable)
/// * `target` - Target variable
/// * `var_names` - Names of variables
/// * `config` - Encoding configuration
///
/// # Returns
/// TargetEncoding with encoded values for the interaction
pub fn target_encode_multi_interaction(
    categories: &[Vec<String>],
    target: &[f64],
    var_names: &[&str],
    config: &TargetEncodingConfig,
) -> TargetEncoding {
    assert!(!categories.is_empty(), "categories must not be empty");
    let n = categories[0].len();
    
    for (i, cat) in categories.iter().enumerate() {
        assert_eq!(cat.len(), n, "All category vectors must have same length (vector {} has {} vs {})", i, cat.len(), n);
    }
    assert_eq!(n, target.len(), "categories and target must have same length");
    
    // Create combined categories by joining with ":"
    let combined: Vec<String> = (0..n)
        .map(|i| {
            categories
                .iter()
                .map(|cat| cat[i].as_str())
                .collect::<Vec<_>>()
                .join(":")
        })
        .collect();
    
    // Apply standard target encoding
    let var_name = var_names.join(":");
    target_encode(&combined, target, &var_name, config)
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_target_encode_basic() {
        let categories: Vec<String> = vec!["A", "B", "A", "B", "A", "B"]
            .into_iter()
            .map(String::from)
            .collect();
        let target = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        
        let config = TargetEncodingConfig {
            prior_weight: 1.0,
            n_permutations: 1,
            seed: Some(42),
        };
        
        let enc = target_encode(&categories, &target, "cat", &config);
        
        // Check that we got values
        assert_eq!(enc.values.len(), 6);
        assert_eq!(enc.levels.len(), 2);
        assert_eq!(enc.name, "TE(cat)");
        
        // Prior should be mean of target
        assert!((enc.prior - 0.5).abs() < 1e-10);
        
        // Level statistics should be correct
        assert_eq!(enc.level_stats["A"].count, 3);
        assert_eq!(enc.level_stats["B"].count, 3);
        assert!((enc.level_stats["A"].sum_target - 3.0).abs() < 1e-10);
        assert!((enc.level_stats["B"].sum_target - 0.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_target_encode_prevents_leakage() {
        // Create perfect predictor scenario (each category is unique)
        let categories: Vec<String> = (0..10)
            .map(|i| format!("cat_{}", i))
            .collect();
        let target: Vec<f64> = (0..10).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect();
        
        let config = TargetEncodingConfig {
            prior_weight: 1.0,
            n_permutations: 1,
            seed: Some(42),
        };
        
        let enc = target_encode(&categories, &target, "cat", &config);
        
        // With ordered statistics, first observation of each unique category
        // should get only the prior (no data seen yet for that category)
        // So encoded values should NOT perfectly predict target
        
        // All first observations should get the prior
        let prior = enc.prior;
        for i in 0..10 {
            // Each category is unique, so each gets (0 + prior*1) / (0 + 1) = prior
            assert!((enc.values[i] - prior).abs() < 1e-10, 
                "Unique category should get prior, got {} vs {}", enc.values[i], prior);
        }
    }
    
    #[test]
    fn test_target_encode_multiple_permutations() {
        let categories: Vec<String> = vec!["A", "B", "A", "B", "A", "B", "A", "B"]
            .into_iter()
            .map(String::from)
            .collect();
        let target = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        
        let config = TargetEncodingConfig {
            prior_weight: 1.0,
            n_permutations: 10,
            seed: Some(42),
        };
        
        let enc = target_encode(&categories, &target, "cat", &config);
        
        // With multiple permutations, variance should be reduced
        // Values for same category should be more similar
        assert_eq!(enc.values.len(), 8);
    }
    
    #[test]
    fn test_apply_target_encoding() {
        let categories: Vec<String> = vec!["A", "B", "A", "B"]
            .into_iter()
            .map(String::from)
            .collect();
        let target = vec![1.0, 0.0, 1.0, 0.0];
        
        let config = TargetEncodingConfig::default();
        let enc = target_encode(&categories, &target, "cat", &config);
        
        // Apply to new data
        let new_categories: Vec<String> = vec!["A", "B", "C"]
            .into_iter()
            .map(String::from)
            .collect();
        
        let new_encoded = apply_target_encoding(&new_categories, &enc, 1.0);
        
        assert_eq!(new_encoded.len(), 3);
        
        // A: (2.0 + 0.5*1) / (2 + 1) = 2.5/3 ≈ 0.833
        assert!((new_encoded[0] - 2.5/3.0).abs() < 1e-10);
        
        // B: (0.0 + 0.5*1) / (2 + 1) = 0.5/3 ≈ 0.167
        assert!((new_encoded[1] - 0.5/3.0).abs() < 1e-10);
        
        // C: unseen category, gets prior = 0.5
        assert!((new_encoded[2] - 0.5).abs() < 1e-10);
    }
    
    #[test]
    fn test_deterministic_with_seed() {
        let categories: Vec<String> = vec!["A", "B", "C", "A", "B", "C"]
            .into_iter()
            .map(String::from)
            .collect();
        let target = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        
        let config = TargetEncodingConfig {
            prior_weight: 1.0,
            n_permutations: 4,
            seed: Some(12345),
        };
        
        let enc1 = target_encode(&categories, &target, "cat", &config);
        let enc2 = target_encode(&categories, &target, "cat", &config);
        
        // Should be identical with same seed
        for i in 0..6 {
            assert!((enc1.values[i] - enc2.values[i]).abs() < 1e-10);
        }
    }
    
    // =========================================================================
    // Frequency Encoding Tests
    // =========================================================================
    
    #[test]
    fn test_frequency_encode_basic() {
        let categories: Vec<String> = vec!["A", "B", "A", "A", "B", "C"]
            .into_iter()
            .map(String::from)
            .collect();
        
        let enc = frequency_encode(&categories, "cat");
        
        // Check structure
        assert_eq!(enc.values.len(), 6);
        assert_eq!(enc.name, "FE(cat)");
        assert_eq!(enc.levels.len(), 3);
        assert_eq!(enc.max_count, 3);  // A appears 3 times
        
        // Check counts
        assert_eq!(enc.level_counts["A"], 3);
        assert_eq!(enc.level_counts["B"], 2);
        assert_eq!(enc.level_counts["C"], 1);
        
        // Check encoded values: count / max_count
        // A=3, B=2, C=1, max=3
        // [A, B, A, A, B, C] -> [3/3, 2/3, 3/3, 3/3, 2/3, 1/3]
        assert!((enc.values[0] - 1.0).abs() < 1e-10);      // A: 3/3
        assert!((enc.values[1] - 2.0/3.0).abs() < 1e-10);  // B: 2/3
        assert!((enc.values[2] - 1.0).abs() < 1e-10);      // A: 3/3
        assert!((enc.values[3] - 1.0).abs() < 1e-10);      // A: 3/3
        assert!((enc.values[4] - 2.0/3.0).abs() < 1e-10);  // B: 2/3
        assert!((enc.values[5] - 1.0/3.0).abs() < 1e-10);  // C: 1/3
    }
    
    #[test]
    fn test_apply_frequency_encoding() {
        let categories: Vec<String> = vec!["A", "B", "A", "A"]
            .into_iter()
            .map(String::from)
            .collect();
        
        let enc = frequency_encode(&categories, "cat");
        
        // Apply to new data including unseen category
        let new_categories: Vec<String> = vec!["A", "B", "C", "D"]
            .into_iter()
            .map(String::from)
            .collect();
        
        let new_encoded = apply_frequency_encoding(&new_categories, &enc);
        
        assert_eq!(new_encoded.len(), 4);
        assert!((new_encoded[0] - 1.0).abs() < 1e-10);      // A: 3/3
        assert!((new_encoded[1] - 1.0/3.0).abs() < 1e-10);  // B: 1/3
        assert!((new_encoded[2] - 0.0).abs() < 1e-10);      // C: unseen -> 0
        assert!((new_encoded[3] - 0.0).abs() < 1e-10);      // D: unseen -> 0
    }
    
    #[test]
    fn test_frequency_encode_single_category() {
        let categories: Vec<String> = vec!["X", "X", "X"]
            .into_iter()
            .map(String::from)
            .collect();
        
        let enc = frequency_encode(&categories, "cat");
        
        // All should be 1.0 (max count = only count)
        assert_eq!(enc.max_count, 3);
        for i in 0..3 {
            assert!((enc.values[i] - 1.0).abs() < 1e-10);
        }
    }
    
    // =========================================================================
    // Target Encoding Interaction Tests
    // =========================================================================
    
    #[test]
    fn test_target_encode_interaction_basic() {
        let cat1: Vec<String> = vec!["A", "A", "B", "B"]
            .into_iter()
            .map(String::from)
            .collect();
        let cat2: Vec<String> = vec!["X", "Y", "X", "Y"]
            .into_iter()
            .map(String::from)
            .collect();
        let target = vec![1.0, 2.0, 3.0, 4.0];
        
        let config = TargetEncodingConfig {
            prior_weight: 1.0,
            n_permutations: 1,
            seed: Some(42),
        };
        
        let enc = target_encode_interaction(&cat1, &cat2, &target, "c1", "c2", &config);
        
        // Check structure
        assert_eq!(enc.values.len(), 4);
        assert_eq!(enc.name, "TE(c1:c2)");
        
        // Should have 4 unique combinations: A:X, A:Y, B:X, B:Y
        assert_eq!(enc.levels.len(), 4);
        
        // Check that prior is correct (mean of target)
        assert!((enc.prior - 2.5).abs() < 1e-10);
    }
    
    #[test]
    fn test_target_encode_interaction_repeated_combinations() {
        let cat1: Vec<String> = vec!["A", "A", "A", "B", "B", "B"]
            .into_iter()
            .map(String::from)
            .collect();
        let cat2: Vec<String> = vec!["X", "X", "Y", "X", "Y", "Y"]
            .into_iter()
            .map(String::from)
            .collect();
        let target = vec![1.0, 1.0, 0.0, 1.0, 0.0, 0.0];
        
        let config = TargetEncodingConfig {
            prior_weight: 1.0,
            n_permutations: 4,
            seed: Some(42),
        };
        
        let enc = target_encode_interaction(&cat1, &cat2, &target, "c1", "c2", &config);
        
        // Check that level stats reflect combined categories
        assert!(enc.level_stats.contains_key("A:X"));
        assert!(enc.level_stats.contains_key("A:Y"));
        assert!(enc.level_stats.contains_key("B:X"));
        assert!(enc.level_stats.contains_key("B:Y"));
        
        // A:X appears twice with sum=2.0
        assert_eq!(enc.level_stats["A:X"].count, 2);
        assert!((enc.level_stats["A:X"].sum_target - 2.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_target_encode_multi_interaction() {
        let cat1: Vec<String> = vec!["A", "A", "B", "B"]
            .into_iter()
            .map(String::from)
            .collect();
        let cat2: Vec<String> = vec!["X", "Y", "X", "Y"]
            .into_iter()
            .map(String::from)
            .collect();
        let cat3: Vec<String> = vec!["1", "1", "2", "2"]
            .into_iter()
            .map(String::from)
            .collect();
        let target = vec![1.0, 2.0, 3.0, 4.0];
        
        let config = TargetEncodingConfig::default();
        
        let enc = target_encode_multi_interaction(
            &[cat1, cat2, cat3],
            &target,
            &["c1", "c2", "c3"],
            &config,
        );
        
        // Check structure
        assert_eq!(enc.values.len(), 4);
        assert_eq!(enc.name, "TE(c1:c2:c3)");
        
        // Should have 4 unique three-way combinations
        assert_eq!(enc.levels.len(), 4);
        assert!(enc.levels.contains(&"A:X:1".to_string()));
        assert!(enc.levels.contains(&"B:Y:2".to_string()));
    }
}
