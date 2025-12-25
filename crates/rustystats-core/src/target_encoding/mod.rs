// =============================================================================
// TARGET ENCODING (CATBOOST-STYLE ORDERED TARGET STATISTICS)
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

/// Compute ordered target statistics encoding (CatBoost-style).
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
}
