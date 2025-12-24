// =============================================================================
// LAZY INTERACTION TERMS
// =============================================================================
//
// This module provides lazy computation of interaction terms for GLM fitting.
// Instead of materializing interaction columns in the design matrix, we compute
// their contributions to X'WX and X'Wz on-the-fly.
//
// PERFORMANCE BENEFITS
// -------------------
// 1. Memory: O(n) instead of O(n × k1 × k2) for categorical interactions
// 2. Speed: Avoid allocating/filling large matrices
// 3. Cache: Better locality by not polluting cache with sparse interaction columns
//
// SUPPORTED INTERACTION TYPES
// ---------------------------
// - Continuous × Continuous: x1 * x2
// - Categorical × Continuous: cat:x (each level multiplied by x)
// - Categorical × Categorical: cat1:cat2 (sparse, most entries are 0)
//
// =============================================================================

use ndarray::{Array1, Array2};
use rayon::prelude::*;

/// Specification for a lazy interaction term.
#[derive(Debug, Clone)]
pub enum InteractionSpec {
    /// Continuous × Continuous: multiply two column indices
    ContinuousContinuous {
        col1: usize,
        col2: usize,
    },
    
    /// Categorical × Continuous: category levels × continuous value
    /// The categorical is represented by level indices (0..n_levels)
    CategoricalContinuous {
        /// Index in the design matrix where this categorical's dummies start
        cat_col_start: usize,
        /// Number of levels (excluding reference)
        n_levels: usize,
        /// Level index for each observation (0 = reference, 1..n_levels = dummies)
        level_indices: Vec<u32>,
        /// Column index for the continuous variable
        cont_col: usize,
    },
    
    /// Categorical × Categorical: sparse interaction
    CategoricalCategorical {
        /// Level indices for first categorical
        level_indices1: Vec<u32>,
        /// Number of levels for first (excluding reference)
        n_levels1: usize,
        /// Level indices for second categorical
        level_indices2: Vec<u32>,
        /// Number of levels for second (excluding reference)
        n_levels2: usize,
    },
}

impl InteractionSpec {
    /// Number of columns this interaction contributes
    pub fn n_columns(&self) -> usize {
        match self {
            InteractionSpec::ContinuousContinuous { .. } => 1,
            InteractionSpec::CategoricalContinuous { n_levels, .. } => *n_levels,
            InteractionSpec::CategoricalCategorical { n_levels1, n_levels2, .. } => {
                n_levels1 * n_levels2
            }
        }
    }
}

/// Compute X'WX contribution from a continuous × continuous interaction.
///
/// For interaction x1:x2, the contribution to X'WX is:
///   xtx_interaction = Σ w_i × (x1_i × x2_i)²
///
/// Cross-terms with other columns require:
///   xtx_cross[j] = Σ w_i × (x1_i × x2_i) × X[i, j]
#[inline]
pub fn xtx_continuous_continuous(
    x: &Array2<f64>,
    col1: usize,
    col2: usize,
    weights: &Array1<f64>,
) -> f64 {
    let n = x.nrows();
    
    (0..n)
        .into_par_iter()
        .map(|i| {
            let interaction = x[[i, col1]] * x[[i, col2]];
            weights[i] * interaction * interaction
        })
        .sum()
}

/// Compute X'Wz contribution from a continuous × continuous interaction.
#[inline]
pub fn xtz_continuous_continuous(
    x: &Array2<f64>,
    col1: usize,
    col2: usize,
    weights: &Array1<f64>,
    z: &Array1<f64>,
) -> f64 {
    let n = x.nrows();
    
    (0..n)
        .into_par_iter()
        .map(|i| {
            let interaction = x[[i, col1]] * x[[i, col2]];
            weights[i] * z[i] * interaction
        })
        .sum()
}

/// Compute X'WX diagonal entries for categorical × categorical interaction.
///
/// For categorical × categorical, each interaction column (i, j) is 1 only for
/// observations where cat1 = level_i AND cat2 = level_j.
///
/// The diagonal entry is simply the sum of weights for those observations.
///
/// Returns: Vec of length n_levels1 × n_levels2 with diagonal entries
pub fn xtx_categorical_categorical_diagonal(
    level_indices1: &[u32],
    n_levels1: usize,
    level_indices2: &[u32],
    n_levels2: usize,
    weights: &Array1<f64>,
) -> Vec<f64> {
    let n = weights.len();
    let n_cols = n_levels1 * n_levels2;
    
    // Accumulate weights for each (level1, level2) combination
    // Use parallel reduction for large data
    if n > 10000 {
        let chunk_size = (n + rayon::current_num_threads() - 1) / rayon::current_num_threads();
        
        let partial_sums: Vec<Vec<f64>> = (0..n)
            .into_par_iter()
            .chunks(chunk_size)
            .map(|chunk| {
                let mut local = vec![0.0; n_cols];
                for i in chunk {
                    let l1 = level_indices1[i] as usize;
                    let l2 = level_indices2[i] as usize;
                    // Only include if both are non-reference levels
                    if l1 > 0 && l2 > 0 {
                        let col = (l1 - 1) * n_levels2 + (l2 - 1);
                        local[col] += weights[i];
                    }
                }
                local
            })
            .collect();
        
        // Reduce partial sums
        let mut result = vec![0.0; n_cols];
        for partial in partial_sums {
            for (r, p) in result.iter_mut().zip(partial.iter()) {
                *r += *p;
            }
        }
        result
    } else {
        let mut result = vec![0.0; n_cols];
        for i in 0..n {
            let l1 = level_indices1[i] as usize;
            let l2 = level_indices2[i] as usize;
            if l1 > 0 && l2 > 0 {
                let col = (l1 - 1) * n_levels2 + (l2 - 1);
                result[col] += weights[i];
            }
        }
        result
    }
}

/// Compute X'Wz entries for categorical × categorical interaction.
pub fn xtz_categorical_categorical(
    level_indices1: &[u32],
    n_levels1: usize,
    level_indices2: &[u32],
    n_levels2: usize,
    weights: &Array1<f64>,
    z: &Array1<f64>,
) -> Vec<f64> {
    let n = weights.len();
    let n_cols = n_levels1 * n_levels2;
    
    if n > 10000 {
        let chunk_size = (n + rayon::current_num_threads() - 1) / rayon::current_num_threads();
        
        let partial_sums: Vec<Vec<f64>> = (0..n)
            .into_par_iter()
            .chunks(chunk_size)
            .map(|chunk| {
                let mut local = vec![0.0; n_cols];
                for i in chunk {
                    let l1 = level_indices1[i] as usize;
                    let l2 = level_indices2[i] as usize;
                    if l1 > 0 && l2 > 0 {
                        let col = (l1 - 1) * n_levels2 + (l2 - 1);
                        local[col] += weights[i] * z[i];
                    }
                }
                local
            })
            .collect();
        
        let mut result = vec![0.0; n_cols];
        for partial in partial_sums {
            for (r, p) in result.iter_mut().zip(partial.iter()) {
                *r += *p;
            }
        }
        result
    } else {
        let mut result = vec![0.0; n_cols];
        for i in 0..n {
            let l1 = level_indices1[i] as usize;
            let l2 = level_indices2[i] as usize;
            if l1 > 0 && l2 > 0 {
                let col = (l1 - 1) * n_levels2 + (l2 - 1);
                result[col] += weights[i] * z[i];
            }
        }
        result
    }
}

/// Compute linear predictor contribution from categorical × categorical interaction.
///
/// For coefficients β and level indices, computes:
///   η_i += β[col] where col = (l1-1) * n2 + (l2-1) if l1, l2 > 0
pub fn linear_predictor_categorical_categorical(
    level_indices1: &[u32],
    _n_levels1: usize,
    level_indices2: &[u32],
    n_levels2: usize,
    coefficients: &[f64],
    coef_start: usize,
) -> Vec<f64> {
    let n = level_indices1.len();
    
    (0..n)
        .into_par_iter()
        .map(|i| {
            let l1 = level_indices1[i] as usize;
            let l2 = level_indices2[i] as usize;
            if l1 > 0 && l2 > 0 {
                let col = (l1 - 1) * n_levels2 + (l2 - 1);
                coefficients[coef_start + col]
            } else {
                0.0
            }
        })
        .collect()
}

/// Materialize interaction columns for small matrices (fallback).
///
/// For small data or when we need the full matrix, this materializes
/// the interaction columns.
pub fn materialize_categorical_categorical(
    level_indices1: &[u32],
    n_levels1: usize,
    level_indices2: &[u32],
    n_levels2: usize,
) -> Array2<f64> {
    let n = level_indices1.len();
    let n_cols = n_levels1 * n_levels2;
    
    let mut result = Array2::zeros((n, n_cols));
    
    for i in 0..n {
        let l1 = level_indices1[i] as usize;
        let l2 = level_indices2[i] as usize;
        if l1 > 0 && l2 > 0 {
            let col = (l1 - 1) * n_levels2 + (l2 - 1);
            result[[i, col]] = 1.0;
        }
    }
    
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_xtx_continuous() {
        let x = Array2::from_shape_vec((4, 2), vec![
            1.0, 2.0,
            2.0, 3.0,
            3.0, 4.0,
            4.0, 5.0,
        ]).unwrap();
        let weights = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0]);
        
        // x1:x2 values are [2, 6, 12, 20]
        // (x1:x2)^2 = [4, 36, 144, 400]
        // sum = 584
        let xtx = xtx_continuous_continuous(&x, 0, 1, &weights);
        assert!((xtx - 584.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_xtx_categorical_categorical() {
        // 4 observations:
        // level1: 0=ref(A), 1=B, 2=C
        // level2: 0=ref(X), 1=Y
        let level1 = vec![0, 1, 2, 1];  // obs: A, B, C, B
        let level2 = vec![1, 0, 1, 1];  // obs: Y, X, Y, Y
        let weights = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0]);
        
        // Interaction columns (excluding reference levels):
        // i=0: A:Y → both l1=0 (ref), skip
        // i=1: B:X → l2=0 (ref), skip
        // i=2: C:Y → l1=2, l2=1 → col = (2-1)*1 + (1-1) = 1
        // i=3: B:Y → l1=1, l2=1 → col = (1-1)*1 + (1-1) = 0
        //
        // Total: 2 columns (2 × 1)
        // col[0] = B:Y = weight at i=3 = 1.0
        // col[1] = C:Y = weight at i=2 = 1.0
        let diag = xtx_categorical_categorical_diagonal(&level1, 2, &level2, 1, &weights);
        assert_eq!(diag.len(), 2);
        assert!((diag[0] - 1.0).abs() < 1e-10, "B:Y should have weight 1.0, got {}", diag[0]);
        assert!((diag[1] - 1.0).abs() < 1e-10, "C:Y should have weight 1.0, got {}", diag[1]);
    }
}
