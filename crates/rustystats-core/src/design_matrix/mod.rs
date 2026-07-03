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

use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};
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
    let mut levels: Vec<String> = values.to_vec();
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
    let n_cols = if drop_first {
        n_levels.saturating_sub(1)
    } else {
        n_levels
    };

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

/// Fast categorical factorization: string values → (sorted_unique_levels, integer_codes).
///
/// Uses HashMap for O(n) encoding instead of O(n log n) sort-based np.unique.
/// Returns sorted unique levels and 0-indexed integer codes.
///
/// # Arguments
/// * `values` - String values for each observation
///
/// # Returns
/// (sorted_unique_levels, integer_codes) where codes[i] is the index of values[i] in levels
pub fn factorize_strings(values: &[String]) -> (Vec<String>, Vec<u32>) {
    let n = values.len();

    // First pass: build HashMap to assign temporary codes
    let mut level_map: HashMap<&str, u32> = HashMap::new();
    let mut levels_order: Vec<&str> = Vec::new();
    let mut temp_codes: Vec<u32> = Vec::with_capacity(n);

    for v in values.iter() {
        let s = v.as_str();
        let code = match level_map.get(s) {
            Some(&c) => c,
            None => {
                let c = levels_order.len() as u32;
                level_map.insert(s, c);
                levels_order.push(s);
                c
            }
        };
        temp_codes.push(code);
    }

    // Sort levels alphabetically (to match np.unique behavior)
    let mut sorted_levels: Vec<String> = levels_order.iter().map(|s| s.to_string()).collect();
    let mut sort_indices: Vec<usize> = (0..sorted_levels.len()).collect();
    sort_indices.sort_by(|&a, &b| sorted_levels[a].cmp(&sorted_levels[b]));
    sorted_levels.sort();

    // Build old→new code mapping
    let k = sort_indices.len();
    let mut remap = vec![0u32; k];
    for (new_idx, &old_idx) in sort_indices.iter().enumerate() {
        remap[old_idx] = new_idx as u32;
    }

    // Remap codes (parallel for large data)
    let codes: Vec<u32> = if n > 50000 {
        temp_codes.par_iter().map(|&c| remap[c as usize]).collect()
    } else {
        temp_codes.iter().map(|&c| remap[c as usize]).collect()
    };

    (sorted_levels, codes)
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
    let n_cols = if drop_first {
        n_levels.saturating_sub(1)
    } else {
        n_levels
    };

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

    for i in 0..n {
        let idx = cat_indices[i];
        if idx >= 1 {
            let col = (idx - 1) as usize;
            if col < n_levels {
                result[[i, col]] = continuous[i];
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

/// Build categorical × basis interaction directly from level indices.
///
/// The output column order is basis-major, then categorical level:
/// ``cat_1:basis_1, cat_2:basis_1, ..., cat_1:basis_2, ...``.
/// This matches the Python builder's historical nested loop while avoiding
/// dense dummy-matrix multiplication for mostly-zero categorical columns.
pub fn build_categorical_basis_interaction(
    cat_indices: &[i32],
    n_levels: usize,
    basis: &Array2<f64>,
    cat_names: &[String],
    basis_names: &[String],
) -> (Array2<f64>, Vec<String>) {
    let n = cat_indices.len();
    let n_basis = basis.ncols();
    let n_cols = n_levels * n_basis;

    if n_cols == 0 {
        return (Array2::zeros((n, 0)), vec![]);
    }

    let mut result = Array2::zeros((n, n_cols));

    for i in 0..n {
        let idx = cat_indices[i];
        if idx >= 1 {
            let cat_col = (idx - 1) as usize;
            if cat_col < n_levels {
                for basis_col in 0..n_basis {
                    let out_col = basis_col * n_levels + cat_col;
                    result[[i, out_col]] = basis[[i, basis_col]];
                }
            }
        }
    }

    let mut col_names = Vec::with_capacity(n_cols);
    for basis_name in basis_names.iter() {
        for cat_name in cat_names.iter() {
            col_names.push(format!("{}:{}", cat_name, basis_name));
        }
    }

    (result, col_names)
}

/// Score a categorical × basis interaction without materialising its design block.
///
/// ``params`` must be ordered basis-major, then categorical level, matching
/// ``build_categorical_basis_interaction``.
pub fn predict_categorical_basis_interaction(
    cat_indices: &[i32],
    n_levels: usize,
    basis: &Array2<f64>,
    params: &[f64],
) -> Array1<f64> {
    predict_categorical_basis_interaction_view(
        ArrayView1::from(cat_indices),
        n_levels,
        basis.view(),
        ArrayView1::from(params),
    )
}

/// View-based variant used by the Python bridge to avoid copying NumPy inputs.
pub fn predict_categorical_basis_interaction_view(
    cat_indices: ArrayView1<'_, i32>,
    n_levels: usize,
    basis: ArrayView2<'_, f64>,
    params: ArrayView1<'_, f64>,
) -> Array1<f64> {
    let n = cat_indices.len();
    let n_basis = basis.ncols();

    if n_levels == 0 || n_basis == 0 {
        return Array1::zeros(n);
    }

    if let (Some(idx_slice), Some(basis_slice), Some(params_slice)) =
        (cat_indices.as_slice(), basis.as_slice(), params.as_slice())
    {
        let required_params = n_levels.saturating_mul(n_basis);
        let values: Vec<f64> = if n > 50_000 {
            (0..n)
                .into_par_iter()
                .map(|i| {
                    score_categorical_basis_row(
                        idx_slice[i],
                        n_levels,
                        &basis_slice[i * n_basis..(i + 1) * n_basis],
                        params_slice,
                        required_params,
                    )
                })
                .collect()
        } else {
            (0..n)
                .map(|i| {
                    score_categorical_basis_row(
                        idx_slice[i],
                        n_levels,
                        &basis_slice[i * n_basis..(i + 1) * n_basis],
                        params_slice,
                        required_params,
                    )
                })
                .collect()
        };
        return Array1::from_vec(values);
    }

    let required_params = n_levels.saturating_mul(n_basis);
    let values: Vec<f64> = if n > 50_000 {
        (0..n)
            .into_par_iter()
            .map(|i| {
                let idx = cat_indices[i];
                if idx < 1 {
                    return 0.0;
                }
                let cat_col = (idx - 1) as usize;
                if cat_col >= n_levels {
                    return 0.0;
                }
                let mut total = 0.0;
                for basis_col in 0..n_basis {
                    let param_idx = basis_col * n_levels + cat_col;
                    if required_params <= params.len() || param_idx < params.len() {
                        total += basis[[i, basis_col]] * params[param_idx];
                    }
                }
                total
            })
            .collect()
    } else {
        (0..n)
            .map(|i| {
                let idx = cat_indices[i];
                if idx < 1 {
                    return 0.0;
                }
                let cat_col = (idx - 1) as usize;
                if cat_col >= n_levels {
                    return 0.0;
                }
                let mut total = 0.0;
                for basis_col in 0..n_basis {
                    let param_idx = basis_col * n_levels + cat_col;
                    if required_params <= params.len() || param_idx < params.len() {
                        total += basis[[i, basis_col]] * params[param_idx];
                    }
                }
                total
            })
            .collect()
    };

    Array1::from_vec(values)
}

#[inline]
fn score_categorical_basis_row(
    idx: i32,
    n_levels: usize,
    basis_row: &[f64],
    params: &[f64],
    required_params: usize,
) -> f64 {
    if idx < 1 {
        return 0.0;
    }
    let cat_col = (idx - 1) as usize;
    if cat_col >= n_levels {
        return 0.0;
    }

    let mut total = 0.0;
    if required_params <= params.len() {
        for (basis_col, value) in basis_row.iter().enumerate() {
            total += value * params[basis_col * n_levels + cat_col];
        }
    } else {
        for (basis_col, value) in basis_row.iter().enumerate() {
            let param_idx = basis_col * n_levels + cat_col;
            if param_idx < params.len() {
                total += value * params[param_idx];
            }
        }
    }
    total
}

/// Build a two-categorical × continuous interaction matrix directly from level
/// indices.
///
/// This is equivalent to first building ``cat1:cat2`` dummy columns and then
/// multiplying each column by ``continuous``. It avoids materialising that
/// intermediate dense categorical interaction block.
pub fn build_two_categorical_continuous_interaction(
    idx1: &[i32],
    n_levels1: usize,
    idx2: &[i32],
    n_levels2: usize,
    continuous: &Array1<f64>,
    names1: &[String],
    names2: &[String],
    cont_name: &str,
) -> (Array2<f64>, Vec<String>) {
    let n = idx1.len();
    let n_cols = n_levels1 * n_levels2;

    if n_cols == 0 {
        return (Array2::zeros((n, 0)), vec![]);
    }

    let mut result = Array2::zeros((n, n_cols));

    for i in 0..n {
        let i1 = idx1[i];
        let i2 = idx2[i];
        if i1 >= 1 && i2 >= 1 {
            let col = ((i1 - 1) as usize) * n_levels2 + ((i2 - 1) as usize);
            if col < n_cols {
                result[[i, col]] = continuous[i];
            }
        }
    }

    let mut col_names = Vec::with_capacity(n_cols);
    for i in 0..n_levels1 {
        for j in 0..n_levels2 {
            let name1 = names1.get(i).map(|s| s.as_str()).unwrap_or("?");
            let name2 = names2.get(j).map(|s| s.as_str()).unwrap_or("?");
            col_names.push(format!("{}:{}:{}", name1, name2, cont_name));
        }
    }

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

    for i in 0..n {
        let cont_val = continuous[i];
        for j in 0..n_cols {
            result[[i, j]] = matrix[[i, j]] * cont_val;
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
    Interaction {
        matrix: Array2<f64>,
        names: Vec<String>,
    },
    /// Spline basis
    Spline {
        matrix: Array2<f64>,
        names: Vec<String>,
    },
}

/// Build complete design matrix from column specifications.
///
/// Efficiently stacks all columns into a single contiguous matrix.
pub fn build_design_matrix(columns: Vec<DesignColumn>, n_obs: usize) -> (Array2<f64>, Vec<String>) {
    // Calculate total columns
    let total_cols: usize = columns
        .iter()
        .map(|c| match c {
            DesignColumn::Intercept => 1,
            DesignColumn::Continuous { .. } => 1,
            DesignColumn::Categorical { encoding } => encoding.matrix.ncols(),
            DesignColumn::Interaction { matrix, .. } => matrix.ncols(),
            DesignColumn::Spline { matrix, .. } => matrix.ncols(),
        })
        .sum();

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
            DesignColumn::Interaction {
                matrix,
                names: int_names,
            } => {
                let n_cols = matrix.ncols();
                for i in 0..n_obs {
                    for j in 0..n_cols {
                        result[[i, col_offset + j]] = matrix[[i, j]];
                    }
                }
                names.extend(int_names);
                col_offset += n_cols;
            }
            DesignColumn::Spline {
                matrix,
                names: spline_names,
            } => {
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
// HORIZONTAL COLUMN-BLOCK STACKING
// =============================================================================

/// Stack a list of (n × c_i) column blocks horizontally into a single
/// (n × Σc_i) row-major matrix.
///
/// Each block is copied into its own slice of the output via `assign` (memcpy
/// under the hood for f64). Blocks with disjoint column ranges in the output
/// are copied in parallel via rayon: each worker writes to a unique column
/// range, so the writes never overlap.
///
/// All blocks must share the same number of rows. The function panics if a
/// block's row count differs from the first block (or if `blocks` is empty
/// and the caller relies on a row count).
///
/// # Arguments
/// * `blocks` - List of 2-D column blocks (each shape `(n, c_i)`).
///
/// # Returns
/// An owned `Array2<f64>` of shape `(n, Σc_i)`.
pub fn stack_columns_horizontal(blocks: &[ArrayView2<f64>]) -> Array2<f64> {
    if blocks.is_empty() {
        return Array2::<f64>::zeros((0, 0));
    }

    let n_rows = blocks[0].nrows();
    let total_cols: usize = blocks.iter().map(|b| b.ncols()).sum();

    // Validate row counts (catches caller bugs early; cheap).
    for (i, b) in blocks.iter().enumerate() {
        assert_eq!(
            b.nrows(),
            n_rows,
            "stack_columns_horizontal: block {} has {} rows, expected {}",
            i,
            b.nrows(),
            n_rows
        );
    }

    if total_cols == 0 {
        return Array2::<f64>::zeros((n_rows, 0));
    }

    // Compute starting column offset for each block.
    let mut offsets = Vec::with_capacity(blocks.len());
    let mut cur = 0usize;
    for b in blocks {
        offsets.push(cur);
        cur += b.ncols();
    }

    // Allocate output. `Array2::zeros` writes the buffer once; the subsequent
    // assigns overwrite it with the block contents.
    let mut out = Array2::<f64>::zeros((n_rows, total_cols));

    // Parallel copy each block into its disjoint column range of `out`.
    //
    // SAFETY: Each task writes to a disjoint column range
    // `[offsets[i], offsets[i] + blocks[i].ncols())` of `out`. Because the
    // ranges are disjoint and we never read from `out` during these writes,
    // no two tasks ever touch the same memory location. We materialize a
    // `*mut Array2<f64>` and reconstruct the disjoint slice views inside
    // each task; rayon ensures the closure outlives the work.
    //
    // We use `addr_of_mut!` (rather than an `&mut -> *mut -> usize` round-trip)
    // to preserve pointer provenance under Rust's strict provenance model
    // (so this code is sound under Miri's strict checks). `*mut T` is `!Send`
    // by default, so we wrap it in a small `SyncPtr` newtype with manual
    // `Send`/`Sync` impls justified by the disjoint-write argument above.
    struct SyncPtr<T>(*mut T);
    // SAFETY: see comment above; tasks only touch disjoint column ranges.
    unsafe impl<T> Send for SyncPtr<T> {}
    unsafe impl<T> Sync for SyncPtr<T> {}

    let out_ptr = SyncPtr(std::ptr::addr_of_mut!(out));
    let out_ptr_ref = &out_ptr;
    blocks
        .par_iter()
        .zip(offsets.par_iter())
        .for_each(|(block, &off)| {
            let w = block.ncols();
            // SAFETY: see comment above; column ranges across tasks are disjoint.
            let out_ref = unsafe { &mut *out_ptr_ref.0 };
            let mut dst = out_ref.slice_mut(s![.., off..off + w]);
            dst.assign(block);
        });

    out
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::s;

    fn assert_row_eq(matrix: &Array2<f64>, row: usize, expected: &[f64]) {
        assert_eq!(matrix.ncols(), expected.len());
        for (col, &value) in expected.iter().enumerate() {
            assert_eq!(
                matrix[[row, col]],
                value,
                "row {row} col {col}: expected {value}, got {}",
                matrix[[row, col]]
            );
        }
    }

    #[test]
    fn test_factorize_and_large_categorical_encoding_parallel_paths() {
        let values: Vec<String> = vec!["b", "a", "c", "b", "a"]
            .into_iter()
            .map(String::from)
            .collect();
        let (levels, codes) = factorize_strings(&values);
        assert_eq!(levels, vec!["a", "b", "c"]);
        assert_eq!(codes, vec![1, 0, 2, 1, 0]);

        let n = 60_001usize;
        let large_values: Vec<String> = (0..n)
            .map(|idx| match idx % 3 {
                0 => "b".to_string(),
                1 => "a".to_string(),
                _ => "c".to_string(),
            })
            .collect();
        let (large_levels, large_codes) = factorize_strings(&large_values);
        assert_eq!(large_levels, vec!["a", "b", "c"]);
        assert_eq!(large_codes[0], 1);
        assert_eq!(large_codes[1], 0);
        assert_eq!(large_codes[2], 2);
        assert_eq!(
            large_codes[n - 1],
            ((n - 1) % 3 != 1) as u32 + ((n - 1) % 3 == 2) as u32
        );

        let enc = encode_categorical(&large_values, "rating", true);
        assert_eq!(enc.matrix.dim(), (n, 2));
        assert_eq!(enc.names, vec!["rating[T.b]", "rating[T.c]"]);
        assert_row_eq(&enc.matrix, 0, &[1.0, 0.0]);
        assert_row_eq(&enc.matrix, 1, &[0.0, 0.0]);
        assert_row_eq(&enc.matrix, 2, &[0.0, 1.0]);

        let indices: Vec<i32> = large_codes.iter().map(|&code| code as i32).collect();
        let from_indices =
            encode_categorical_from_indices(&indices, 3, &large_levels, "rating", true);
        assert_eq!(from_indices.matrix, enc.matrix);
        assert_eq!(from_indices.names, enc.names);
    }

    #[test]
    fn test_encode_categorical_from_indices_name_fallback_and_invalid_codes() {
        let indices = vec![-1, 0, 1, 2, 3, 99];
        let level_names = vec!["A".to_string(), "B".to_string()];
        let enc = encode_categorical_from_indices(&indices, 4, &level_names, "zone", false);

        assert_eq!(
            enc.names,
            vec!["zone[T.A]", "zone[T.B]", "zone[T.2]", "zone[T.3]"]
        );
        assert_row_eq(&enc.matrix, 0, &[0.0, 0.0, 0.0, 0.0]);
        assert_row_eq(&enc.matrix, 1, &[1.0, 0.0, 0.0, 0.0]);
        assert_row_eq(&enc.matrix, 2, &[0.0, 1.0, 0.0, 0.0]);
        assert_row_eq(&enc.matrix, 3, &[0.0, 0.0, 1.0, 0.0]);
        assert_row_eq(&enc.matrix, 4, &[0.0, 0.0, 0.0, 1.0]);
        assert_row_eq(&enc.matrix, 5, &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_large_categorical_categorical_interaction_parallel_path() {
        let n = 60_001usize;
        let idx1: Vec<i32> = (0..n).map(|idx| (idx % 4) as i32).collect();
        let idx2: Vec<i32> = (0..n).map(|idx| ((idx * 2 + 1) % 4) as i32).collect();
        let names1 = vec![
            "driver[T.B]".to_string(),
            "driver[T.C]".to_string(),
            "driver[T.D]".to_string(),
        ];
        let names2 = vec![
            "territory[T.Y]".to_string(),
            "territory[T.Z]".to_string(),
            "territory[T.W]".to_string(),
        ];

        let (matrix, names) =
            build_categorical_categorical_interaction(&idx1, 3, &idx2, 3, &names1, &names2);

        assert_eq!(matrix.dim(), (n, 9));
        assert_eq!(names.len(), 9);
        assert_eq!(names[0], "driver[T.B]:territory[T.Y]");
        assert_eq!(names[8], "driver[T.D]:territory[T.W]");
        assert_eq!(matrix.row(0).sum(), 0.0);
        assert_eq!(matrix[[1, 2]], 1.0);
        assert_eq!(matrix[[2, 3]], 1.0);
        assert_eq!(matrix[[3, 8]], 1.0);
        assert_eq!(matrix.row(4).sum(), 0.0);
        assert_eq!(matrix.row(n - 1).sum(), 0.0);
    }

    #[test]
    fn test_categorical_basis_interaction_edge_and_prediction_contracts() {
        let basis =
            Array2::from_shape_vec((4, 2), vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0])
                .expect("valid basis");
        let cat_idx = vec![0i32, 1, 3, 2];
        let cat_names = vec!["cat[T.B]".to_string(), "cat[T.C]".to_string()];
        let basis_names = vec!["b0".to_string(), "b1".to_string()];

        let (zero, zero_names) =
            build_categorical_basis_interaction(&cat_idx, 0, &basis, &[], &basis_names);
        assert_eq!(zero.dim(), (4, 0));
        assert!(zero_names.is_empty());

        let (matrix, names) =
            build_categorical_basis_interaction(&cat_idx, 2, &basis, &cat_names, &basis_names);
        assert_eq!(
            names,
            vec!["cat[T.B]:b0", "cat[T.C]:b0", "cat[T.B]:b1", "cat[T.C]:b1"]
        );
        assert_row_eq(&matrix, 0, &[0.0, 0.0, 0.0, 0.0]);
        assert_row_eq(&matrix, 1, &[2.0, 0.0, 20.0, 0.0]);
        assert_row_eq(&matrix, 2, &[0.0, 0.0, 0.0, 0.0]);
        assert_row_eq(&matrix, 3, &[0.0, 4.0, 0.0, 40.0]);

        let zero_pred =
            predict_categorical_basis_interaction(&cat_idx, 0, &basis, &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(zero_pred.to_vec(), vec![0.0; 4]);
        let zero_basis = Array2::<f64>::zeros((4, 0));
        let zero_basis_pred = predict_categorical_basis_interaction(&cat_idx, 2, &zero_basis, &[]);
        assert_eq!(zero_basis_pred.to_vec(), vec![0.0; 4]);

        let short_params = Array1::from_vec(vec![0.5, 1.5, 2.0]);
        let mut wide_basis = Array2::<f64>::zeros((4, 4));
        wide_basis
            .slice_mut(s![.., 0])
            .assign(&basis.slice(s![.., 0]));
        wide_basis
            .slice_mut(s![.., 2])
            .assign(&basis.slice(s![.., 1]));
        let noncontiguous_basis = wide_basis.slice(s![.., ..;2]);
        let fallback_values = predict_categorical_basis_interaction_view(
            ArrayView1::from(&cat_idx),
            2,
            noncontiguous_basis,
            short_params.view(),
        );
        assert_eq!(fallback_values.to_vec(), vec![0.0, 41.0, 0.0, 6.0]);
    }

    #[test]
    fn test_predict_categorical_basis_large_parallel_paths() {
        let n = 60_001usize;
        let cat_idx: Vec<i32> = (0..n).map(|idx| (idx % 4) as i32).collect();
        let mut basis = Array2::<f64>::zeros((n, 2));
        for row in 0..n {
            basis[[row, 0]] = row as f64 * 0.01;
            basis[[row, 1]] = 1.0 + (row % 7) as f64;
        }
        let params = vec![0.5, 1.5, 2.0, 3.0];

        let values = predict_categorical_basis_interaction(&cat_idx, 2, &basis, &params);
        assert_eq!(values[0], 0.0);
        assert!((values[1] - (0.01 * 0.5 + 2.0 * 2.0)).abs() < 1e-12);
        assert!((values[2] - (0.02 * 1.5 + 3.0 * 3.0)).abs() < 1e-12);
        assert_eq!(values[3], 0.0);

        let mut wide_basis = Array2::<f64>::zeros((n, 4));
        for row in 0..n {
            wide_basis[[row, 0]] = basis[[row, 0]];
            wide_basis[[row, 2]] = basis[[row, 1]];
        }
        let noncontiguous_values = predict_categorical_basis_interaction_view(
            ArrayView1::from(&cat_idx),
            2,
            wide_basis.slice(s![.., ..;2]),
            ArrayView1::from(&params),
        );
        assert_eq!(noncontiguous_values[0], values[0]);
        assert_eq!(noncontiguous_values[1], values[1]);
        assert_eq!(noncontiguous_values[2], values[2]);
        assert_eq!(noncontiguous_values[n - 1], values[n - 1]);
    }

    #[test]
    fn test_two_categorical_continuous_zero_levels_and_out_of_range_codes() {
        let idx1 = vec![0i32, 1, 3, 2];
        let idx2 = vec![0i32, 1, 2, 9];
        let continuous = Array1::from_vec(vec![10.0, 20.0, 30.0, 40.0]);
        let names1 = vec!["a[T.B]".to_string(), "a[T.C]".to_string()];
        let names2 = vec!["b[T.Y]".to_string(), "b[T.Z]".to_string()];

        let (zero, zero_names) = build_two_categorical_continuous_interaction(
            &idx1,
            0,
            &idx2,
            2,
            &continuous,
            &[],
            &names2,
            "x",
        );
        assert_eq!(zero.dim(), (4, 0));
        assert!(zero_names.is_empty());

        let (matrix, names) = build_two_categorical_continuous_interaction(
            &idx1,
            2,
            &idx2,
            2,
            &continuous,
            &names1,
            &names2,
            "x",
        );
        assert_eq!(
            names,
            vec![
                "a[T.B]:b[T.Y]:x",
                "a[T.B]:b[T.Z]:x",
                "a[T.C]:b[T.Y]:x",
                "a[T.C]:b[T.Z]:x"
            ]
        );
        assert_row_eq(&matrix, 0, &[0.0, 0.0, 0.0, 0.0]);
        assert_row_eq(&matrix, 1, &[20.0, 0.0, 0.0, 0.0]);
        assert_row_eq(&matrix, 2, &[0.0, 0.0, 0.0, 0.0]);
        assert_row_eq(&matrix, 3, &[0.0, 0.0, 0.0, 0.0]);
    }

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
        let values: Vec<String> = vec!["X", "Y", "X"].into_iter().map(String::from).collect();

        let enc = encode_categorical(&values, "var", false);

        // Should have 2 columns (X, Y)
        assert_eq!(enc.matrix.ncols(), 2);
        // Row 0: X -> [1, 0]
        assert_eq!(enc.matrix[[0, 0]], 1.0);
        assert_eq!(enc.matrix[[0, 1]], 0.0);
    }

    #[test]
    fn test_encode_categorical_single_level() {
        let values: Vec<String> = vec!["A", "A", "A"].into_iter().map(String::from).collect();

        let enc = encode_categorical(&values, "cat", true);

        // Single level with drop_first → 0 columns
        assert_eq!(enc.matrix.ncols(), 0);
        assert_eq!(enc.names.len(), 0);
        assert_eq!(enc.levels.len(), 1);
    }

    #[test]
    fn test_encode_categorical_preserves_indices() {
        let values: Vec<String> = vec!["B", "A", "C", "B"]
            .into_iter()
            .map(String::from)
            .collect();

        let enc = encode_categorical(&values, "cat", true);

        // Levels are sorted: A=0, B=1, C=2
        assert_eq!(enc.indices, vec![1, 0, 2, 1]);
        assert_eq!(enc.levels, vec!["A", "B", "C"]);
    }

    #[test]
    fn test_encode_categorical_from_indices() {
        let indices = vec![0, 1, 2, 1, 0];
        let level_names = vec!["A".to_string(), "B".to_string(), "C".to_string()];

        let enc = encode_categorical_from_indices(&indices, 3, &level_names, "cat", true);

        // 3 levels - 1 = 2 columns
        assert_eq!(enc.matrix.ncols(), 2);
        assert_eq!(enc.matrix.nrows(), 5);

        // Row 0: A (idx 0) → reference, [0, 0]
        assert_eq!(enc.matrix[[0, 0]], 0.0);
        assert_eq!(enc.matrix[[0, 1]], 0.0);

        // Row 1: B (idx 1) → [1, 0]
        assert_eq!(enc.matrix[[1, 0]], 1.0);
        assert_eq!(enc.matrix[[1, 1]], 0.0);

        // Row 2: C (idx 2) → [0, 1]
        assert_eq!(enc.matrix[[2, 0]], 0.0);
        assert_eq!(enc.matrix[[2, 1]], 1.0);
    }

    #[test]
    fn test_encode_categorical_from_indices_no_drop() {
        let indices = vec![0, 1, 0];
        let level_names = vec!["X".to_string(), "Y".to_string()];

        let enc = encode_categorical_from_indices(&indices, 2, &level_names, "cat", false);

        // 2 columns without dropping
        assert_eq!(enc.matrix.ncols(), 2);

        // Row 0: X → [1, 0]
        assert_eq!(enc.matrix[[0, 0]], 1.0);
        assert_eq!(enc.matrix[[0, 1]], 0.0);

        // Row 1: Y → [0, 1]
        assert_eq!(enc.matrix[[1, 0]], 0.0);
        assert_eq!(enc.matrix[[1, 1]], 1.0);
    }

    #[test]
    fn test_encode_categorical_from_indices_single_level() {
        let indices = vec![0, 0, 0];
        let level_names = vec!["A".to_string()];

        let enc = encode_categorical_from_indices(&indices, 1, &level_names, "cat", true);

        // Single level with drop → 0 columns
        assert_eq!(enc.matrix.ncols(), 0);
    }

    #[test]
    fn test_categorical_categorical_interaction() {
        // Cat1: A(ref), B, C -> indices 0, 1, 2
        // Cat2: X(ref), Y -> indices 0, 1
        let idx1 = vec![0i32, 1, 2, 1, 0]; // A, B, C, B, A
        let idx2 = vec![0i32, 1, 1, 0, 1]; // X, Y, Y, X, Y

        let names1 = vec!["cat1[T.B]".to_string(), "cat1[T.C]".to_string()];
        let names2 = vec!["cat2[T.Y]".to_string()];

        let (matrix, names) =
            build_categorical_categorical_interaction(&idx1, 2, &idx2, 1, &names1, &names2);

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
    fn test_categorical_categorical_interaction_empty() {
        let idx1: Vec<i32> = vec![];
        let idx2: Vec<i32> = vec![];
        let names1 = vec!["a".to_string()];
        let names2 = vec!["b".to_string()];

        let (matrix, names) =
            build_categorical_categorical_interaction(&idx1, 1, &idx2, 1, &names1, &names2);

        assert_eq!(matrix.shape(), &[0, 1]);
        assert_eq!(names.len(), 1);
    }

    #[test]
    fn test_categorical_categorical_interaction_zero_levels() {
        let idx1 = vec![0i32, 1];
        let idx2 = vec![0i32, 1];
        let names1: Vec<String> = vec![];
        let names2: Vec<String> = vec![];

        let (matrix, names) =
            build_categorical_categorical_interaction(&idx1, 0, &idx2, 0, &names1, &names2);

        assert_eq!(matrix.ncols(), 0);
        assert_eq!(names.len(), 0);
    }

    #[test]
    fn test_categorical_continuous_interaction() {
        let cat_idx = vec![0i32, 1, 2, 1]; // Ref, Level1, Level2, Level1
        let cont = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let cat_names = vec!["cat[T.B]".to_string(), "cat[T.C]".to_string()];

        let (matrix, _names) =
            build_categorical_continuous_interaction(&cat_idx, 2, &cont, &cat_names, "x");

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
    fn test_categorical_continuous_interaction_zero_levels() {
        let cat_idx = vec![0i32, 0];
        let cont = Array1::from_vec(vec![1.0, 2.0]);
        let cat_names: Vec<String> = vec![];

        let (matrix, names) =
            build_categorical_continuous_interaction(&cat_idx, 0, &cont, &cat_names, "x");

        assert_eq!(matrix.ncols(), 0);
        assert_eq!(names.len(), 0);
    }

    #[test]
    fn test_categorical_basis_interaction() {
        let cat_idx = vec![0i32, 1, 2, 1];
        let basis =
            Array2::from_shape_vec((4, 2), vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0])
                .expect("valid test basis shape");
        let cat_names = vec!["cat[T.B]".to_string(), "cat[T.C]".to_string()];
        let basis_names = vec!["bs(x, 1/2)".to_string(), "bs(x, 2/2)".to_string()];

        let (matrix, names) =
            build_categorical_basis_interaction(&cat_idx, 2, &basis, &cat_names, &basis_names);

        assert_eq!(
            names,
            vec![
                "cat[T.B]:bs(x, 1/2)",
                "cat[T.C]:bs(x, 1/2)",
                "cat[T.B]:bs(x, 2/2)",
                "cat[T.C]:bs(x, 2/2)",
            ]
        );
        assert_eq!(matrix.shape(), &[4, 4]);
        assert_eq!(matrix.row(0).to_vec(), vec![0.0, 0.0, 0.0, 0.0]);
        assert_eq!(matrix.row(1).to_vec(), vec![2.0, 0.0, 20.0, 0.0]);
        assert_eq!(matrix.row(2).to_vec(), vec![0.0, 3.0, 0.0, 30.0]);
        assert_eq!(matrix.row(3).to_vec(), vec![4.0, 0.0, 40.0, 0.0]);
    }

    #[test]
    fn test_predict_categorical_basis_interaction() {
        let cat_idx = vec![0i32, 1, 2, 1];
        let basis =
            Array2::from_shape_vec((4, 2), vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0])
                .expect("valid test basis shape");
        let params = vec![0.5, 1.5, 2.0, 3.0];

        let values = predict_categorical_basis_interaction(&cat_idx, 2, &basis, &params);

        assert_eq!(values.to_vec(), vec![0.0, 41.0, 94.5, 82.0]);
    }

    #[test]
    fn test_two_categorical_continuous_interaction_matches_composed_path() {
        let idx1 = vec![0i32, 1, 2, 1, 2];
        let idx2 = vec![0i32, 1, 1, 0, 2];
        let continuous = Array1::from_vec(vec![10.0, 20.0, 30.0, 40.0, 50.0]);
        let names1 = vec!["cat1[T.B]".to_string(), "cat1[T.C]".to_string()];
        let names2 = vec!["cat2[T.Y]".to_string(), "cat2[T.Z]".to_string()];

        let (cat_cat, cat_cat_names) =
            build_categorical_categorical_interaction(&idx1, 2, &idx2, 2, &names1, &names2);
        let (composed, composed_names) =
            multiply_matrix_by_continuous(&cat_cat, &continuous, &cat_cat_names, "x");
        let (direct, direct_names) = build_two_categorical_continuous_interaction(
            &idx1,
            2,
            &idx2,
            2,
            &continuous,
            &names1,
            &names2,
            "x",
        );

        assert_eq!(direct_names, composed_names);
        assert_eq!(direct.shape(), composed.shape());
        assert_eq!(direct, composed);
        assert_eq!(direct[[1, 0]], 20.0);
        assert_eq!(direct[[2, 2]], 30.0);
        assert_eq!(direct[[4, 3]], 50.0);
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
    fn test_multiply_matrix_by_continuous() {
        let matrix = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
            .expect("test setup should be valid");
        let continuous = Array1::from_vec(vec![2.0, 3.0, 4.0]);
        let names = vec!["a".to_string(), "b".to_string()];

        let (result, result_names) =
            multiply_matrix_by_continuous(&matrix, &continuous, &names, "x");

        assert_eq!(result.shape(), &[3, 2]);
        assert_eq!(result_names, vec!["a:x", "b:x"]);

        // Row 0: [1, 0] * 2 = [2, 0]
        assert_eq!(result[[0, 0]], 2.0);
        assert_eq!(result[[0, 1]], 0.0);

        // Row 1: [0, 1] * 3 = [0, 3]
        assert_eq!(result[[1, 0]], 0.0);
        assert_eq!(result[[1, 1]], 3.0);

        // Row 2: [1, 1] * 4 = [4, 4]
        assert_eq!(result[[2, 0]], 4.0);
        assert_eq!(result[[2, 1]], 4.0);
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
        assert_eq!(matrix[[0, 0]], 1.0); // Intercept
        assert_eq!(matrix[[0, 1]], 1.0); // x[0]
        assert_eq!(matrix[[2, 1]], 3.0); // x[2]
    }

    #[test]
    fn test_build_design_matrix_with_categorical() {
        let n = 4;

        let values: Vec<String> = vec!["A", "B", "A", "B"]
            .into_iter()
            .map(String::from)
            .collect();
        let enc = encode_categorical(&values, "cat", true);

        let columns = vec![
            DesignColumn::Intercept,
            DesignColumn::Categorical { encoding: enc },
        ];

        let (matrix, names) = build_design_matrix(columns, n);

        assert_eq!(matrix.shape(), &[4, 2]); // Intercept + 1 dummy
        assert_eq!(names.len(), 2);

        // All intercepts = 1
        for i in 0..4 {
            assert_eq!(matrix[[i, 0]], 1.0);
        }

        // Dummies: A=0, B=1
        assert_eq!(matrix[[0, 1]], 0.0); // A
        assert_eq!(matrix[[1, 1]], 1.0); // B
        assert_eq!(matrix[[2, 1]], 0.0); // A
        assert_eq!(matrix[[3, 1]], 1.0); // B
    }

    #[test]
    fn test_build_design_matrix_with_interaction() {
        let n = 2;

        let int_matrix = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0])
            .expect("test setup should be valid");

        let columns = vec![
            DesignColumn::Intercept,
            DesignColumn::Interaction {
                matrix: int_matrix,
                names: vec!["a:b".to_string(), "a:c".to_string()],
            },
        ];

        let (matrix, names) = build_design_matrix(columns, n);

        assert_eq!(matrix.shape(), &[2, 3]);
        assert_eq!(names, vec!["Intercept", "a:b", "a:c"]);
    }

    #[test]
    fn test_build_design_matrix_with_spline() {
        let n = 3;

        let spline_matrix = Array2::from_shape_vec((3, 2), vec![0.5, 0.5, 0.3, 0.7, 0.1, 0.9])
            .expect("test setup should be valid");

        let columns = vec![DesignColumn::Spline {
            matrix: spline_matrix,
            names: vec!["bs(x, 1)".to_string(), "bs(x, 2)".to_string()],
        }];

        let (matrix, names) = build_design_matrix(columns, n);

        assert_eq!(matrix.shape(), &[3, 2]);
        assert_eq!(names, vec!["bs(x, 1)", "bs(x, 2)"]);
        assert_eq!(matrix[[0, 0]], 0.5);
        assert_eq!(matrix[[2, 1]], 0.9);
    }

    #[test]
    fn test_build_design_matrix_empty() {
        let columns: Vec<DesignColumn> = vec![];
        let (matrix, names) = build_design_matrix(columns, 5);

        assert_eq!(matrix.shape(), &[5, 0]);
        assert_eq!(names.len(), 0);
    }

    #[test]
    fn test_stack_columns_horizontal_basic() {
        let a = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("test setup should be valid");
        let b = Array2::from_shape_vec((3, 1), vec![7.0, 8.0, 9.0])
            .expect("test setup should be valid");
        let c = Array2::from_shape_vec(
            (3, 3),
            vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
        )
        .expect("test setup should be valid");

        let blocks = vec![a.view(), b.view(), c.view()];
        let out = stack_columns_horizontal(&blocks);

        assert_eq!(out.shape(), &[3, 6]);
        // Row 0: [1, 2, 7, 10, 11, 12]
        assert_eq!(out[[0, 0]], 1.0);
        assert_eq!(out[[0, 1]], 2.0);
        assert_eq!(out[[0, 2]], 7.0);
        assert_eq!(out[[0, 3]], 10.0);
        assert_eq!(out[[0, 4]], 11.0);
        assert_eq!(out[[0, 5]], 12.0);
        // Row 2: [5, 6, 9, 16, 17, 18]
        assert_eq!(out[[2, 0]], 5.0);
        assert_eq!(out[[2, 1]], 6.0);
        assert_eq!(out[[2, 2]], 9.0);
        assert_eq!(out[[2, 3]], 16.0);
        assert_eq!(out[[2, 5]], 18.0);
    }

    #[test]
    fn test_stack_columns_horizontal_single_block() {
        let a = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0])
            .expect("test setup should be valid");
        let blocks = vec![a.view()];
        let out = stack_columns_horizontal(&blocks);
        assert_eq!(out.shape(), &[2, 2]);
        assert_eq!(out, a);
    }

    #[test]
    fn test_stack_columns_horizontal_empty_blocks() {
        let blocks: Vec<ArrayView2<f64>> = vec![];
        let out = stack_columns_horizontal(&blocks);
        assert_eq!(out.shape(), &[0, 0]);
    }

    #[test]
    fn test_stack_columns_horizontal_zero_cols() {
        let a = Array2::<f64>::zeros((4, 0));
        let blocks = vec![a.view()];
        let out = stack_columns_horizontal(&blocks);
        assert_eq!(out.shape(), &[4, 0]);
    }

    #[test]
    #[should_panic(expected = "stack_columns_horizontal: block 1 has 3 rows, expected 2")]
    fn test_stack_columns_horizontal_rejects_mismatched_row_counts() {
        let a = Array2::<f64>::zeros((2, 1));
        let b = Array2::<f64>::zeros((3, 1));
        let blocks = vec![a.view(), b.view()];
        let _ = stack_columns_horizontal(&blocks);
    }

    #[test]
    fn test_stack_columns_horizontal_many_blocks() {
        // Stress-test parallel copy: 30 blocks of (1000, 1)
        let block = Array2::<f64>::from_elem((1000, 1), 7.5);
        let owned: Vec<Array2<f64>> = (0..30).map(|_| block.clone()).collect();
        let views: Vec<_> = owned.iter().map(|b| b.view()).collect();
        let out = stack_columns_horizontal(&views);
        assert_eq!(out.shape(), &[1000, 30]);
        for r in [0usize, 500, 999] {
            for c in 0..30 {
                assert_eq!(out[[r, c]], 7.5);
            }
        }
    }
}
