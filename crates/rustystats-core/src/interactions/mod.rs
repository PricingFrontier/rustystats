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
    fn test_xtx_continuous_with_weights() {
        let x = Array2::from_shape_vec((4, 2), vec![
            1.0, 2.0,
            2.0, 3.0,
            3.0, 4.0,
            4.0, 5.0,
        ]).unwrap();
        let weights = Array1::from_vec(vec![2.0, 0.5, 1.0, 0.5]);
        
        // x1:x2 values are [2, 6, 12, 20]
        // (x1:x2)^2 = [4, 36, 144, 400]
        // weighted sum = 2*4 + 0.5*36 + 1*144 + 0.5*400 = 8 + 18 + 144 + 200 = 370
        let xtx = xtx_continuous_continuous(&x, 0, 1, &weights);
        assert!((xtx - 370.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_xtz_continuous_continuous() {
        let x = Array2::from_shape_vec((4, 2), vec![
            1.0, 2.0,
            2.0, 3.0,
            3.0, 4.0,
            4.0, 5.0,
        ]).unwrap();
        let weights = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0]);
        let z = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        
        // x1:x2 values are [2, 6, 12, 20]
        // weighted: w * z * (x1:x2) = [1*1*2, 1*2*6, 1*3*12, 1*4*20]
        //                           = [2, 12, 36, 80] = 130
        let xtz = xtz_continuous_continuous(&x, 0, 1, &weights, &z);
        assert!((xtz - 130.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_xtz_continuous_continuous_with_weights() {
        let x = Array2::from_shape_vec((3, 2), vec![
            1.0, 1.0,
            2.0, 2.0,
            3.0, 3.0,
        ]).unwrap();
        let weights = Array1::from_vec(vec![2.0, 1.0, 0.5]);
        let z = Array1::from_vec(vec![1.0, 1.0, 1.0]);
        
        // x1:x2 = [1, 4, 9]
        // w * z * (x1:x2) = [2*1*1, 1*1*4, 0.5*1*9] = [2, 4, 4.5] = 10.5
        let xtz = xtz_continuous_continuous(&x, 0, 1, &weights, &z);
        assert!((xtz - 10.5).abs() < 1e-10);
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
    
    #[test]
    fn test_xtx_categorical_categorical_with_weights() {
        let level1 = vec![1, 1, 2, 2];
        let level2 = vec![1, 1, 1, 2];
        let weights = Array1::from_vec(vec![2.0, 3.0, 1.0, 4.0]);
        
        // Columns: B:Y (col 0), B:Z (col 1), C:Y (col 2), C:Z (col 3)
        // i=0: B:Y → col 0, weight 2.0
        // i=1: B:Y → col 0, weight 3.0
        // i=2: C:Y → col 2, weight 1.0
        // i=3: C:Z → col 3, weight 4.0
        let diag = xtx_categorical_categorical_diagonal(&level1, 2, &level2, 2, &weights);
        assert_eq!(diag.len(), 4);
        assert!((diag[0] - 5.0).abs() < 1e-10); // B:Y = 2+3
        assert!((diag[1] - 0.0).abs() < 1e-10); // B:Z = 0
        assert!((diag[2] - 1.0).abs() < 1e-10); // C:Y = 1
        assert!((diag[3] - 4.0).abs() < 1e-10); // C:Z = 4
    }
    
    #[test]
    fn test_xtz_categorical_categorical() {
        let level1 = vec![1, 2, 1];
        let level2 = vec![1, 1, 2];
        let weights = Array1::from_vec(vec![1.0, 1.0, 1.0]);
        let z = Array1::from_vec(vec![2.0, 3.0, 4.0]);
        
        // Columns: B:Y (0), B:Z (1), C:Y (2), C:Z (3)
        // i=0: B:Y → w*z = 1*2 = 2
        // i=1: C:Y → w*z = 1*3 = 3
        // i=2: B:Z → w*z = 1*4 = 4
        let xtz = xtz_categorical_categorical(&level1, 2, &level2, 2, &weights, &z);
        assert_eq!(xtz.len(), 4);
        assert!((xtz[0] - 2.0).abs() < 1e-10); // B:Y
        assert!((xtz[1] - 4.0).abs() < 1e-10); // B:Z
        assert!((xtz[2] - 3.0).abs() < 1e-10); // C:Y
        assert!((xtz[3] - 0.0).abs() < 1e-10); // C:Z
    }
    
    #[test]
    fn test_xtz_categorical_categorical_reference_levels() {
        // Test that reference levels (0) contribute nothing
        let level1 = vec![0, 0, 1];
        let level2 = vec![0, 1, 0];
        let weights = Array1::from_vec(vec![1.0, 1.0, 1.0]);
        let z = Array1::from_vec(vec![10.0, 10.0, 10.0]);
        
        // Only non-reference × non-reference contributes
        // All observations have at least one reference level → all zeros
        let xtz = xtz_categorical_categorical(&level1, 1, &level2, 1, &weights, &z);
        assert_eq!(xtz.len(), 1);
        assert!((xtz[0] - 0.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_linear_predictor_categorical_categorical() {
        let level1 = vec![0, 1, 2, 1];
        let level2 = vec![1, 1, 1, 0];
        let coefficients = vec![100.0, 1.0, 2.0, 3.0, 4.0]; // intercept + 4 interaction coefs
        let coef_start = 1;
        
        // Columns: B:Y (0), B:Z (1), C:Y (2), C:Z (3)
        // i=0: A:Y → reference, 0
        // i=1: B:Y → coef[1] = 1.0
        // i=2: C:Y → coef[3] = 3.0
        // i=3: B:X → reference, 0
        let eta = linear_predictor_categorical_categorical(
            &level1, 2, &level2, 2, &coefficients, coef_start
        );
        assert_eq!(eta.len(), 4);
        assert!((eta[0] - 0.0).abs() < 1e-10);
        assert!((eta[1] - 1.0).abs() < 1e-10);
        assert!((eta[2] - 3.0).abs() < 1e-10);
        assert!((eta[3] - 0.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_materialize_categorical_categorical() {
        let level1 = vec![0, 1, 2, 1];
        let level2 = vec![1, 1, 1, 0];
        
        let matrix = materialize_categorical_categorical(&level1, 2, &level2, 1);
        
        assert_eq!(matrix.shape(), &[4, 2]); // 2 × 1 = 2 columns
        
        // Row 0: A:Y → reference, all zeros
        assert_eq!(matrix[[0, 0]], 0.0);
        assert_eq!(matrix[[0, 1]], 0.0);
        
        // Row 1: B:Y → col 0
        assert_eq!(matrix[[1, 0]], 1.0);
        assert_eq!(matrix[[1, 1]], 0.0);
        
        // Row 2: C:Y → col 1
        assert_eq!(matrix[[2, 0]], 0.0);
        assert_eq!(matrix[[2, 1]], 1.0);
        
        // Row 3: B:X → reference, all zeros
        assert_eq!(matrix[[3, 0]], 0.0);
        assert_eq!(matrix[[3, 1]], 0.0);
    }
    
    #[test]
    fn test_materialize_categorical_categorical_multiple_levels() {
        let level1 = vec![1, 2];
        let level2 = vec![1, 2];
        
        let matrix = materialize_categorical_categorical(&level1, 2, &level2, 2);
        
        // 2 × 2 = 4 columns
        assert_eq!(matrix.shape(), &[2, 4]);
        
        // Row 0: B:Y → col 0
        assert_eq!(matrix[[0, 0]], 1.0);
        assert_eq!(matrix[[0, 1]], 0.0);
        assert_eq!(matrix[[0, 2]], 0.0);
        assert_eq!(matrix[[0, 3]], 0.0);
        
        // Row 1: C:Z → col 3
        assert_eq!(matrix[[1, 0]], 0.0);
        assert_eq!(matrix[[1, 1]], 0.0);
        assert_eq!(matrix[[1, 2]], 0.0);
        assert_eq!(matrix[[1, 3]], 1.0);
    }
    
    #[test]
    fn test_interaction_spec_n_columns() {
        let cont_cont = InteractionSpec::ContinuousContinuous { col1: 0, col2: 1 };
        assert_eq!(cont_cont.n_columns(), 1);
        
        let cat_cont = InteractionSpec::CategoricalContinuous {
            cat_col_start: 0,
            n_levels: 5,
            level_indices: vec![0, 1, 2],
            cont_col: 3,
        };
        assert_eq!(cat_cont.n_columns(), 5);
        
        let cat_cat = InteractionSpec::CategoricalCategorical {
            level_indices1: vec![0, 1],
            n_levels1: 3,
            level_indices2: vec![0, 1],
            n_levels2: 4,
        };
        assert_eq!(cat_cat.n_columns(), 12); // 3 × 4
    }
    
    #[test]
    fn test_empty_interactions() {
        let level1: Vec<u32> = vec![];
        let level2: Vec<u32> = vec![];
        let weights = Array1::from_vec(vec![]);
        let z = Array1::from_vec(vec![]);
        
        let diag = xtx_categorical_categorical_diagonal(&level1, 2, &level2, 2, &weights);
        assert_eq!(diag.len(), 4);
        assert!(diag.iter().all(|&x| x == 0.0));
        
        let xtz = xtz_categorical_categorical(&level1, 2, &level2, 2, &weights, &z);
        assert_eq!(xtz.len(), 4);
        assert!(xtz.iter().all(|&x| x == 0.0));
    }
    
    #[test]
    fn test_single_observation() {
        let level1 = vec![1];
        let level2 = vec![1];
        let weights = Array1::from_vec(vec![2.5]);
        let z = Array1::from_vec(vec![3.0]);
        
        let diag = xtx_categorical_categorical_diagonal(&level1, 1, &level2, 1, &weights);
        assert_eq!(diag.len(), 1);
        assert!((diag[0] - 2.5).abs() < 1e-10);
        
        let xtz = xtz_categorical_categorical(&level1, 1, &level2, 1, &weights, &z);
        assert_eq!(xtz.len(), 1);
        assert!((xtz[0] - 7.5).abs() < 1e-10); // 2.5 * 3.0
    }
}
