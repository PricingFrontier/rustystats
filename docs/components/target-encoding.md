# Target Encoding Component

Target encoding converts high-cardinality categorical variables into numerical features using the target variable. RustyStats implements CatBoost-style ordered target statistics to prevent target leakage.

## Code Location

```
crates/rustystats-core/src/target_encoding/
└── mod.rs  # All target encoding functionality

python/rustystats/target_encoding.py  # Python API
```

## Why Target Encoding?

### The Problem with One-Hot Encoding

For a categorical with k levels, one-hot creates k-1 columns:
- 10 levels → 9 columns
- 1000 levels → 999 columns
- 100,000 levels (e.g., ZIP codes) → 99,999 columns

This causes:
- Memory explosion
- Overfitting
- Slow training

### Target Encoding Solution

Replace each category with a single number based on the target:

```
Category    Target    Encoded
--------    ------    -------
Toyota      1.0       0.65
Ford        0.0       0.40
BMW         1.0       0.75
Toyota      0.5       0.65
...
```

## The Target Leakage Problem

Naive target encoding causes **target leakage**:

```python
# WRONG: Uses observation's own target in encoding
for category in categories:
    encoded[category] = mean(target[category == categories])
```

The model sees information about the target it's trying to predict → overfitting.

## CatBoost Solution: Ordered Statistics

CatBoost's approach computes encodings using only "past" observations:

```
For observation i in random order:
    encoded[i] = (sum of target for this category BEFORE i) / (count BEFORE i)
```

### Algorithm

```rust
pub fn target_encode(
    categories: &[String],
    target: &Array1<f64>,
    prior_weight: f64,
    n_permutations: usize,
    seed: u64,
) -> (Array1<f64>, f64, HashMap<String, LevelStatistics>) {
    let n = categories.len();
    let prior = target.mean().unwrap();  // Global mean
    
    // Average across multiple permutations for stability
    let mut encoded_sum = Array1::zeros(n);
    
    for perm_idx in 0..n_permutations {
        // Random permutation of indices
        let permutation = generate_permutation(n, seed + perm_idx as u64);
        
        // Running statistics per category
        let mut category_sum: HashMap<String, f64> = HashMap::new();
        let mut category_count: HashMap<String, usize> = HashMap::new();
        
        let mut encoded_perm = Array1::zeros(n);
        
        for &i in &permutation {
            let cat = &categories[i];
            
            // Get statistics from observations BEFORE this one
            let sum_before = *category_sum.get(cat).unwrap_or(&0.0);
            let count_before = *category_count.get(cat).unwrap_or(&0);
            
            // Compute encoding with regularization toward prior
            encoded_perm[i] = (sum_before + prior * prior_weight) 
                            / (count_before as f64 + prior_weight);
            
            // Update running statistics
            *category_sum.entry(cat.clone()).or_insert(0.0) += target[i];
            *category_count.entry(cat.clone()).or_insert(0) += 1;
        }
        
        encoded_sum = encoded_sum + encoded_perm;
    }
    
    // Average across permutations
    let encoded = encoded_sum / n_permutations as f64;
    
    // Compute full statistics for prediction
    let stats = compute_full_statistics(categories, target, prior, prior_weight);
    
    (encoded, prior, stats)
}
```

### Key Properties

1. **No leakage**: Each observation's encoding uses only prior observations
2. **Regularization**: `prior_weight` shrinks rare categories toward global mean
3. **Stability**: Multiple permutations reduce variance

## Regularization with Prior Weight

The encoding formula:

\[
\text{encoded}_i = \frac{\sum_{j < i} y_j + \mu \cdot w}{\text{count}_{j < i} + w}
\]

where:
- \(\mu\) = global mean (prior)
- \(w\) = prior weight

| prior_weight | Effect |
|--------------|--------|
| 0 | No regularization (pure empirical mean) |
| 1 | Mild regularization |
| 10 | Strong regularization (rare categories → global mean) |

### Example

For a rare category with 2 observations (target = 1, 1):

| prior_weight | Encoded value (prior=0.5) |
|--------------|---------------------------|
| 0 | 1.0 |
| 1 | (2.0 + 0.5×1) / (2 + 1) = 0.83 |
| 10 | (2.0 + 0.5×10) / (2 + 10) = 0.58 |

## Applying to New Data

For prediction, use full training statistics:

```rust
pub fn apply_target_encoding(
    categories: &[String],
    stats: &HashMap<String, LevelStatistics>,
    prior: f64,
) -> Array1<f64> {
    categories.iter()
        .map(|cat| {
            stats.get(cat)
                .map(|s| s.mean)
                .unwrap_or(prior)  // Unseen categories get prior
        })
        .collect()
}
```

## Python API

### Direct API

```python
import rustystats as rs
import numpy as np

categories = ["A", "B", "A", "C", "B", "A"]
target = np.array([1.0, 0.0, 0.5, 1.0, 0.2, 0.8])

# Encode
encoded, column_name, prior, stats = rs.target_encode(
    categories, 
    target,
    column_name="category",
    prior_weight=1.0,
    n_permutations=4,
    seed=42
)

# Apply to new data
new_cats = ["A", "B", "D"]  # D is unseen
new_encoded = rs.apply_target_encoding(new_cats, stats, prior)
# D gets the prior (global mean)
```

### Sklearn-Style API

```python
encoder = rs.TargetEncoder(prior_weight=1.0, n_permutations=4)

# Fit and transform training data
train_encoded = encoder.fit_transform(train_categories, train_target)

# Transform test data (uses full training statistics)
test_encoded = encoder.transform(test_categories)
```

### Formula API

```python
result = rs.glm(
    "ClaimNb ~ TE(VehicleBrand) + TE(ZipCode, prior_weight=2.0) + Age",
    data=data,
    family="poisson",
    offset="Exposure"
).fit()
```

## Data Structures

```rust
pub struct LevelStatistics {
    pub level: String,
    pub count: usize,
    pub sum: f64,
    pub mean: f64,
}

pub struct TargetEncoding {
    pub encoded_values: Array1<f64>,
    pub prior: f64,
    pub level_stats: HashMap<String, LevelStatistics>,
    pub column_name: String,
}

pub struct TargetEncodingConfig {
    pub prior_weight: f64,
    pub n_permutations: usize,
    pub seed: u64,
}
```

## Testing

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_no_leakage() {
        // With 1 observation per category, encoding should equal prior
        // (no past observations to use)
        let categories = vec!["A".into(), "B".into(), "C".into()];
        let target = array![1.0, 0.0, 0.5];
        
        let (encoded, prior, _) = target_encode(
            &categories, &target, 1.0, 1, 42
        );
        
        // First observation of each category should be regularized toward prior
        let expected_prior = target.mean().unwrap();
        for &e in encoded.iter() {
            // With prior_weight=1, first obs: (0 + prior*1) / (0 + 1) = prior
            assert!((e - expected_prior).abs() < 0.01);
        }
    }
    
    #[test]
    fn test_unseen_category() {
        let categories = vec!["A".into(), "A".into()];
        let target = array![1.0, 1.0];
        
        let (_, prior, stats) = target_encode(&categories, &target, 1.0, 1, 42);
        
        // Apply to unseen category
        let new_cats = vec!["B".into()];
        let new_encoded = apply_target_encoding(&new_cats, &stats, prior);
        
        // Unseen category gets prior
        assert!((new_encoded[0] - prior).abs() < 1e-10);
    }
    
    #[test]
    fn test_prior_weight_effect() {
        let categories = vec!["A".into(), "A".into()];
        let target = array![1.0, 1.0];
        
        // Low prior weight
        let (_, _, stats_low) = target_encode(&categories, &target, 0.1, 10, 42);
        
        // High prior weight
        let (_, _, stats_high) = target_encode(&categories, &target, 10.0, 10, 42);
        
        // High prior weight should pull mean toward prior
        let prior = 1.0;  // mean of [1.0, 1.0]
        assert!(stats_high.get("A").unwrap().mean < stats_low.get("A").unwrap().mean);
    }
}
```

## When to Use Target Encoding

### Good Use Cases

- High-cardinality categoricals (100s or 1000s of levels)
- ZIP codes, product IDs, user IDs
- When feature count would explode with one-hot

### Avoid When

- Low-cardinality (< 10 levels): one-hot is fine
- Need interpretable per-level effects
- Target is unavailable at prediction time

### Comparison

| Encoding | Columns | Leakage Risk | Interpretability |
|----------|---------|--------------|------------------|
| One-Hot | k-1 | None | High |
| Target (naive) | 1 | High | Medium |
| Target (CatBoost) | 1 | Low | Medium |
| Ordinal | 1 | None | Low |
