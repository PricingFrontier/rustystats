// =============================================================================
// DESIGN MATRIX CONSTRUCTION
// =============================================================================
//
// High-performance design matrix construction for GLMs.
// All heavy computation is done in Rust for maximum speed.
//
// FEATURES:
// ---------
// - Categorical encoding (dummy variables)
// - Interaction terms (cat×cat, cat×cont, cont×cont)
// - Parallel construction using Rayon
//
// =============================================================================

use ndarray::{Array1, Array2};
use rayon::prelude::*;
use std::collections::HashMap;

// =============================================================================
// CATEGORICAL ENCODING
// =============================================================================

/// Result of categorical encoding
#[derive(Debug, Clone)]
pub struct CategoricalEncoding {
    /// Dummy-encoded matrix (n_obs × n_levels-1 if drop_first, else n_obs × n_levels)
    pub matrix: Array2<f64>,
    /// Level names for each column
    pub names: Vec<String>,
    /// Original level indices (0-indexed, sorted)
    pub indices: Vec<i32>,
    /// All unique levels (sorted)
    pub levels: Vec<String>,
}

/// Encode a categorical variable as dummy variables.
///
/// Takes string values and returns a dummy-encoded matrix.
/// Uses parallel sorting and HashMap for fast factorization.
///
/// # Arguments
/// * `values` - String values for each observation
/// * `var_name` - Variable name (for column naming)
/// * `drop_first` - Whether to drop the first level (reference category)
///
/// # Returns
/// CategoricalEncoding with dummy matrix and metadata
pub fn encode_categorical(
    values: &[String],
    var_name: &str,
    drop_first: bool,
) -> CategoricalEncoding {
    let n = values.len();
    
    // Get unique levels and sort them
    let mut levels: Vec<String> = values.iter().cloned().collect();
    levels.sort();
    levels.dedup();
    
    // Create level-to-index mapping
    let level_map: HashMap<&str, i32> = levels
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i as i32))
        .collect();
    
    // Convert values to indices (parallel for large data)
    let indices: Vec<i32> = if n > 10000 {
        values
            .par_iter()
            .map(|v| *level_map.get(v.as_str()).unwrap_or(&0))
            .collect()
    } else {
        values
            .iter()
            .map(|v| *level_map.get(v.as_str()).unwrap_or(&0))
            .collect()
    };
    
    // Build dummy matrix
    let n_levels = levels.len();
    let start_idx: i32 = if drop_first { 1 } else { 0 };
    let n_cols = if drop_first { n_levels.saturating_sub(1) } else { n_levels };
    
    if n_cols == 0 {
        return CategoricalEncoding {
            matrix: Array2::zeros((n, 0)),
            names: vec![],
            indices,
            levels,
        };
    }
    
    // Pre-allocate and fill (parallel for large data)
    let mut matrix = Array2::zeros((n, n_cols));
    
    if n > 50000 {
        // Parallel construction for large data
        let rows: Vec<Vec<f64>> = indices
            .par_iter()
            .map(|&idx| {
                let mut row = vec![0.0; n_cols];
                let col = idx - start_idx;
                if col >= 0 && (col as usize) < n_cols {
                    row[col as usize] = 1.0;
                }
                row
            })
            .collect();
        
        for (i, row) in rows.into_iter().enumerate() {
            for (j, val) in row.into_iter().enumerate() {
                matrix[[i, j]] = val;
            }
        }
    } else {
        // Sequential for smaller data (less overhead)
        for (i, &idx) in indices.iter().enumerate() {
            let col = idx - start_idx;
            if col >= 0 && (col as usize) < n_cols {
                matrix[[i, col as usize]] = 1.0;
            }
        }
    }
    
    // Generate column names
    let names: Vec<String> = (0..n_cols)
        .map(|i| format!("{}[T.{}]", var_name, levels[i + start_idx as usize]))
        .collect();
    
    CategoricalEncoding {
        matrix,
        names,
        indices,
        levels,
    }
}

/// Encode categorical from pre-computed indices.
///
/// Use this when indices are already computed (e.g., from factorization).
///
/// # Arguments
/// * `indices` - Pre-computed level indices (0-indexed)
/// * `n_levels` - Total number of levels
/// * `level_names` - Names for each level
/// * `var_name` - Variable name
/// * `drop_first` - Drop first level
pub fn encode_categorical_from_indices(
    indices: &[i32],
    n_levels: usize,
    level_names: &[String],
    var_name: &str,
    drop_first: bool,
) -> CategoricalEncoding {
    let n = indices.len();
    let start_idx: i32 = if drop_first { 1 } else { 0 };
    let n_cols = if drop_first { n_levels.saturating_sub(1) } else { n_levels };
    
    if n_cols == 0 {
        return CategoricalEncoding {
            matrix: Array2::zeros((n, 0)),
            names: vec![],
            indices: indices.to_vec(),
            levels: level_names.to_vec(),
        };
    }
    
    let mut matrix = Array2::zeros((n, n_cols));
    
    // Use parallel for large data
    if n > 50000 {
        let rows: Vec<(usize, usize)> = indices
            .par_iter()
            .enumerate()
            .filter_map(|(i, &idx)| {
                let col = idx - start_idx;
                if col >= 0 && (col as usize) < n_cols {
                    Some((i, col as usize))
                } else {
                    None
                }
            })
            .collect();
        
        for (row, col) in rows {
            matrix[[row, col]] = 1.0;
        }
    } else {
        for (i, &idx) in indices.iter().enumerate() {
            let col = idx - start_idx;
            if col >= 0 && (col as usize) < n_cols {
                matrix[[i, col as usize]] = 1.0;
            }
        }
    }
    
    let names: Vec<String> = (0..n_cols)
        .map(|i| {
            let level_idx = i + start_idx as usize;
            if level_idx < level_names.len() {
                format!("{}[T.{}]", var_name, level_names[level_idx])
            } else {
                format!("{}[T.{}]", var_name, level_idx)
            }
        })
        .collect();
    
    CategoricalEncoding {
        matrix,
        names,
        indices: indices.to_vec(),
        levels: level_names.to_vec(),
    }
}

// =============================================================================
// INTERACTION TERMS
// =============================================================================

/// Build categorical × categorical interaction matrix.
///
/// For two categorical variables with levels (excluding reference):
/// - Cat1 has n1 levels (after dropping first)
/// - Cat2 has n2 levels (after dropping first)
/// - Result has n1 × n2 interaction columns
///
/// # Arguments
/// * `idx1` - Level indices for first categorical (0 = reference)
/// * `n_levels1` - Number of levels for first (excluding reference)
/// * `idx2` - Level indices for second categorical
/// * `n_levels2` - Number of levels for second (excluding reference)
/// * `names1` - Column names for first categorical dummies
/// * `names2` - Column names for second categorical dummies
pub fn build_categorical_categorical_interaction(
    idx1: &[i32],
    n_levels1: usize,
    idx2: &[i32],
    n_levels2: usize,
    names1: &[String],
    names2: &[String],
) -> (Array2<f64>, Vec<String>) {
    let n = idx1.len();
    let n_cols = n_levels1 * n_levels2;
    
    if n_cols == 0 {
        return (Array2::zeros((n, 0)), vec![]);
    }
    
    let mut result = Array2::zeros((n, n_cols));
    
    // Parallel construction for large data
    if n > 50000 {
        let entries: Vec<(usize, usize)> = (0..n)
            .into_par_iter()
            .filter_map(|i| {
                let i1 = idx1[i];
                let i2 = idx2[i];
                // Only non-reference levels (idx >= 1 means level index >= 1)
                if i1 >= 1 && i2 >= 1 {
                    let col = ((i1 - 1) as usize) * n_levels2 + ((i2 - 1) as usize);
                    if col < n_cols {
                        return Some((i, col));
                    }
                }
                None
            })
            .collect();
        
        for (row, col) in entries {
            result[[row, col]] = 1.0;
        }
    } else {
        for i in 0..n {
            let i1 = idx1[i];
            let i2 = idx2[i];
            if i1 >= 1 && i2 >= 1 {
                let col = ((i1 - 1) as usize) * n_levels2 + ((i2 - 1) as usize);
                if col < n_cols {
                    result[[i, col]] = 1.0;
                }
            }
        }
    }
    
    // Generate column names
    let mut col_names = Vec::with_capacity(n_cols);
    for i in 0..n_levels1 {
        for j in 0..n_levels2 {
            let name1 = names1.get(i).map(|s| s.as_str()).unwrap_or("?");
            let name2 = names2.get(j).map(|s| s.as_str()).unwrap_or("?");
            col_names.push(format!("{}:{}", name1, name2));
        }
    }
    
    (result, col_names)
}

/// Build categorical × continuous interaction matrix.
///
/// Each level of the categorical gets multiplied by the continuous variable.
///
/// # Arguments
/// * `cat_indices` - Level indices for categorical (0 = reference)
/// * `n_levels` - Number of non-reference levels
/// * `continuous` - Continuous variable values
/// * `cat_names` - Column names for categorical dummies
/// * `cont_name` - Name of continuous variable
pub fn build_categorical_continuous_interaction(
    cat_indices: &[i32],
    n_levels: usize,
    continuous: &Array1<f64>,
    cat_names: &[String],
    cont_name: &str,
) -> (Array2<f64>, Vec<String>) {
    let n = cat_indices.len();
    
    if n_levels == 0 {
        return (Array2::zeros((n, 0)), vec![]);
    }
    
    let mut result = Array2::zeros((n, n_levels));
    
    // Parallel for large data
    if n > 50000 {
        let rows: Vec<Vec<f64>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut row = vec![0.0; n_levels];
                let idx = cat_indices[i];
                if idx >= 1 {
                    let col = (idx - 1) as usize;
                    if col < n_levels {
                        row[col] = continuous[i];
                    }
                }
                row
            })
            .collect();
        
        for (i, row) in rows.into_iter().enumerate() {
            for (j, val) in row.into_iter().enumerate() {
                result[[i, j]] = val;
            }
        }
    } else {
        for i in 0..n {
            let idx = cat_indices[i];
            if idx >= 1 {
                let col = (idx - 1) as usize;
                if col < n_levels {
                    result[[i, col]] = continuous[i];
                }
            }
        }
    }
    
    // Generate column names
    let col_names: Vec<String> = cat_names
        .iter()
        .map(|name| format!("{}:{}", name, cont_name))
        .collect();
    
    (result, col_names)
}

/// Build continuous × continuous interaction.
///
/// Simple element-wise multiplication.
pub fn build_continuous_continuous_interaction(
    x1: &Array1<f64>,
    x2: &Array1<f64>,
    name1: &str,
    name2: &str,
) -> (Array1<f64>, String) {
    let result = x1 * x2;
    let name = format!("{}:{}", name1, name2);
    (result, name)
}

/// Multiply each column of a matrix by a continuous vector.
///
/// Used for multi-categorical × continuous interactions where we have
/// already built the categorical interaction matrix and need to multiply
/// each column by the continuous values.
///
/// # Arguments
/// * `matrix` - Categorical interaction matrix (n_obs × n_cols)
/// * `continuous` - Continuous values (n_obs,)
/// * `matrix_names` - Names for each column of the matrix
/// * `cont_name` - Name of the continuous variable
pub fn multiply_matrix_by_continuous(
    matrix: &Array2<f64>,
    continuous: &Array1<f64>,
    matrix_names: &[String],
    cont_name: &str,
) -> (Array2<f64>, Vec<String>) {
    let n = matrix.nrows();
    let n_cols = matrix.ncols();
    
    let mut result = Array2::zeros((n, n_cols));
    
    // Parallel for large data
    if n > 50000 {
        let rows: Vec<Vec<f64>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let cont_val = continuous[i];
                (0..n_cols).map(|j| matrix[[i, j]] * cont_val).collect()
            })
            .collect();
        
        for (i, row) in rows.into_iter().enumerate() {
            for (j, val) in row.into_iter().enumerate() {
                result[[i, j]] = val;
            }
        }
    } else {
        for i in 0..n {
            let cont_val = continuous[i];
            for j in 0..n_cols {
                result[[i, j]] = matrix[[i, j]] * cont_val;
            }
        }
    }
    
    let names: Vec<String> = matrix_names
        .iter()
        .map(|name| format!("{}:{}", name, cont_name))
        .collect();
    
    (result, names)
}

// =============================================================================
// DESIGN MATRIX BUILDER
// =============================================================================

/// Column type for design matrix construction
#[derive(Debug, Clone)]
pub enum DesignColumn {
    /// Intercept column (all 1s)
    Intercept,
    /// Continuous variable
    Continuous { values: Array1<f64>, name: String },
    /// Categorical variable (pre-encoded)
    Categorical { encoding: CategoricalEncoding },
    /// Interaction term
    Interaction { matrix: Array2<f64>, names: Vec<String> },
    /// Spline basis
    Spline { matrix: Array2<f64>, names: Vec<String> },
}

/// Build complete design matrix from column specifications.
///
/// Efficiently stacks all columns into a single contiguous matrix.
pub fn build_design_matrix(columns: Vec<DesignColumn>, n_obs: usize) -> (Array2<f64>, Vec<String>) {
    // Calculate total columns
    let total_cols: usize = columns.iter().map(|c| match c {
        DesignColumn::Intercept => 1,
        DesignColumn::Continuous { .. } => 1,
        DesignColumn::Categorical { encoding } => encoding.matrix.ncols(),
        DesignColumn::Interaction { matrix, .. } => matrix.ncols(),
        DesignColumn::Spline { matrix, .. } => matrix.ncols(),
    }).sum();
    
    let mut result = Array2::zeros((n_obs, total_cols));
    let mut names = Vec::with_capacity(total_cols);
    let mut col_offset = 0;
    
    for column in columns {
        match column {
            DesignColumn::Intercept => {
                for i in 0..n_obs {
                    result[[i, col_offset]] = 1.0;
                }
                names.push("Intercept".to_string());
                col_offset += 1;
            }
            DesignColumn::Continuous { values, name } => {
                for i in 0..n_obs {
                    result[[i, col_offset]] = values[i];
                }
                names.push(name);
                col_offset += 1;
            }
            DesignColumn::Categorical { encoding } => {
                let n_cols = encoding.matrix.ncols();
                for i in 0..n_obs {
                    for j in 0..n_cols {
                        result[[i, col_offset + j]] = encoding.matrix[[i, j]];
                    }
                }
                names.extend(encoding.names);
                col_offset += n_cols;
            }
            DesignColumn::Interaction { matrix, names: int_names } => {
                let n_cols = matrix.ncols();
                for i in 0..n_obs {
                    for j in 0..n_cols {
                        result[[i, col_offset + j]] = matrix[[i, j]];
                    }
                }
                names.extend(int_names);
                col_offset += n_cols;
            }
            DesignColumn::Spline { matrix, names: spline_names } => {
                let n_cols = matrix.ncols();
                for i in 0..n_obs {
                    for j in 0..n_cols {
                        result[[i, col_offset + j]] = matrix[[i, j]];
                    }
                }
                names.extend(spline_names);
                col_offset += n_cols;
            }
        }
    }
    
    (result, names)
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_encode_categorical() {
        let values: Vec<String> = vec!["A", "B", "C", "A", "B", "C"]
            .into_iter()
            .map(String::from)
            .collect();
        
        let enc = encode_categorical(&values, "cat", true);
        
        // Should have 2 columns (B, C) after dropping A
        assert_eq!(enc.matrix.ncols(), 2);
        assert_eq!(enc.matrix.nrows(), 6);
        assert_eq!(enc.names.len(), 2);
        
        // Check encoding
        // Row 0: A -> [0, 0]
        assert_eq!(enc.matrix[[0, 0]], 0.0);
        assert_eq!(enc.matrix[[0, 1]], 0.0);
        // Row 1: B -> [1, 0]
        assert_eq!(enc.matrix[[1, 0]], 1.0);
        assert_eq!(enc.matrix[[1, 1]], 0.0);
        // Row 2: C -> [0, 1]
        assert_eq!(enc.matrix[[2, 0]], 0.0);
        assert_eq!(enc.matrix[[2, 1]], 1.0);
    }
    
    #[test]
    fn test_encode_categorical_no_drop() {
        let values: Vec<String> = vec!["X", "Y", "X"]
            .into_iter()
            .map(String::from)
            .collect();
        
        let enc = encode_categorical(&values, "var", false);
        
        // Should have 2 columns (X, Y)
        assert_eq!(enc.matrix.ncols(), 2);
        // Row 0: X -> [1, 0]
        assert_eq!(enc.matrix[[0, 0]], 1.0);
        assert_eq!(enc.matrix[[0, 1]], 0.0);
    }
    
    #[test]
    fn test_categorical_categorical_interaction() {
        // Cat1: A(ref), B, C -> indices 0, 1, 2
        // Cat2: X(ref), Y -> indices 0, 1
        let idx1 = vec![0i32, 1, 2, 1, 0];  // A, B, C, B, A
        let idx2 = vec![0i32, 1, 1, 0, 1];  // X, Y, Y, X, Y
        
        let names1 = vec!["cat1[T.B]".to_string(), "cat1[T.C]".to_string()];
        let names2 = vec!["cat2[T.Y]".to_string()];
        
        let (matrix, names) = build_categorical_categorical_interaction(
            &idx1, 2, &idx2, 1, &names1, &names2
        );
        
        // 2 × 1 = 2 interaction columns
        assert_eq!(matrix.ncols(), 2);
        assert_eq!(names.len(), 2);
        
        // Row 0: A:X -> both reference, no 1s
        assert_eq!(matrix[[0, 0]], 0.0);
        assert_eq!(matrix[[0, 1]], 0.0);
        
        // Row 1: B:Y -> col 0 (B×Y)
        assert_eq!(matrix[[1, 0]], 1.0);
        assert_eq!(matrix[[1, 1]], 0.0);
        
        // Row 2: C:Y -> col 1 (C×Y)
        assert_eq!(matrix[[2, 0]], 0.0);
        assert_eq!(matrix[[2, 1]], 1.0);
    }
    
    #[test]
    fn test_categorical_continuous_interaction() {
        let cat_idx = vec![0i32, 1, 2, 1];  // Ref, Level1, Level2, Level1
        let cont = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let cat_names = vec!["cat[T.B]".to_string(), "cat[T.C]".to_string()];
        
        let (matrix, names) = build_categorical_continuous_interaction(
            &cat_idx, 2, &cont, &cat_names, "x"
        );
        
        assert_eq!(matrix.ncols(), 2);
        
        // Row 0: ref level -> [0, 0]
        assert_eq!(matrix[[0, 0]], 0.0);
        assert_eq!(matrix[[0, 1]], 0.0);
        
        // Row 1: Level1 × 2.0 -> [2.0, 0]
        assert_eq!(matrix[[1, 0]], 2.0);
        assert_eq!(matrix[[1, 1]], 0.0);
        
        // Row 2: Level2 × 3.0 -> [0, 3.0]
        assert_eq!(matrix[[2, 0]], 0.0);
        assert_eq!(matrix[[2, 1]], 3.0);
    }
    
    #[test]
    fn test_continuous_continuous_interaction() {
        let x1 = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let x2 = Array1::from_vec(vec![4.0, 5.0, 6.0]);
        
        let (result, name) = build_continuous_continuous_interaction(&x1, &x2, "a", "b");
        
        assert_eq!(name, "a:b");
        assert_eq!(result[0], 4.0);
        assert_eq!(result[1], 10.0);
        assert_eq!(result[2], 18.0);
    }
    
    #[test]
    fn test_build_design_matrix() {
        let n = 3;
        
        let columns = vec![
            DesignColumn::Intercept,
            DesignColumn::Continuous {
                values: Array1::from_vec(vec![1.0, 2.0, 3.0]),
                name: "x".to_string(),
            },
        ];
        
        let (matrix, names) = build_design_matrix(columns, n);
        
        assert_eq!(matrix.shape(), &[3, 2]);
        assert_eq!(names, vec!["Intercept", "x"]);
        
        // Check values
        assert_eq!(matrix[[0, 0]], 1.0);  // Intercept
        assert_eq!(matrix[[0, 1]], 1.0);  // x[0]
        assert_eq!(matrix[[2, 1]], 3.0);  // x[2]
    }
}
