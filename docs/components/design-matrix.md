# Design Matrix Component

The design matrix module handles construction of the feature matrix X from raw data, including categorical encoding, interactions, and splines.

## Code Location

```
crates/rustystats-core/src/design_matrix/
└── mod.rs  # All design matrix functionality
```

## Overview

The design matrix X is the n × p matrix where:
- n = number of observations
- p = number of features (including intercept)

Each row represents an observation, each column a feature.

## Categorical Encoding

### Dummy Coding (Reference Level)

Default encoding drops the first level to avoid collinearity:

```rust
pub fn encode_categorical(
    values: &[String],
    reference_level: Option<&str>,
) -> (Array2<f64>, Vec<String>) {
    // Get unique levels
    let mut levels: Vec<String> = values.iter()
        .cloned()
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    levels.sort();  // Consistent ordering
    
    // Determine reference level
    let ref_level = reference_level
        .unwrap_or(&levels[0]);
    
    // Create column for each non-reference level
    let non_ref_levels: Vec<_> = levels.iter()
        .filter(|l| l.as_str() != ref_level)
        .collect();
    
    let n = values.len();
    let k = non_ref_levels.len();
    let mut encoded = Array2::zeros((n, k));
    
    for (i, val) in values.iter().enumerate() {
        if let Some(j) = non_ref_levels.iter().position(|l| *l == val) {
            encoded[[i, j]] = 1.0;
        }
    }
    
    // Column names: "Category_Level"
    let names = non_ref_levels.iter()
        .map(|l| format!("{}_{}", "Category", l))
        .collect();
    
    (encoded, names)
}
```

### Example

For `Region = ["A", "B", "C", "A", "B"]`:

| Row | Region | Region_B | Region_C |
|-----|--------|----------|----------|
| 0 | A | 0 | 0 |
| 1 | B | 1 | 0 |
| 2 | C | 0 | 1 |
| 3 | A | 0 | 0 |
| 4 | B | 1 | 0 |

Level "A" is the reference (coefficient absorbed into intercept).

## Index-Based Encoding

For efficiency with large datasets, encode from pre-computed indices:

```rust
pub fn encode_categorical_from_indices(
    indices: &Array1<usize>,
    n_levels: usize,
    include_reference: bool,
) -> Array2<f64> {
    let n = indices.len();
    let k = if include_reference { n_levels } else { n_levels - 1 };
    let offset = if include_reference { 0 } else { 1 };
    
    let mut encoded = Array2::zeros((n, k));
    
    for (i, &idx) in indices.iter().enumerate() {
        if idx >= offset {
            encoded[[i, idx - offset]] = 1.0;
        }
    }
    
    encoded
}
```

## Interaction Terms

### Categorical × Categorical

Creates all pairwise combinations:

```rust
pub fn build_categorical_categorical_interaction(
    cat1: &Array2<f64>,
    cat2: &Array2<f64>,
    names1: &[String],
    names2: &[String],
) -> (Array2<f64>, Vec<String>) {
    let n = cat1.nrows();
    let k1 = cat1.ncols();
    let k2 = cat2.ncols();
    let k = k1 * k2;
    
    let mut interaction = Array2::zeros((n, k));
    let mut names = Vec::with_capacity(k);
    
    for (j1, name1) in names1.iter().enumerate() {
        for (j2, name2) in names2.iter().enumerate() {
            let col_idx = j1 * k2 + j2;
            
            // Element-wise product of columns
            for i in 0..n {
                interaction[[i, col_idx]] = cat1[[i, j1]] * cat2[[i, j2]];
            }
            
            names.push(format!("{}:{}", name1, name2));
        }
    }
    
    (interaction, names)
}
```

### Categorical × Continuous

Each categorical level gets its own slope:

```rust
pub fn build_categorical_continuous_interaction(
    categorical: &Array2<f64>,
    continuous: &Array1<f64>,
    cat_names: &[String],
    cont_name: &str,
) -> (Array2<f64>, Vec<String>) {
    let n = categorical.nrows();
    let k = categorical.ncols();
    
    let mut interaction = Array2::zeros((n, k));
    let mut names = Vec::with_capacity(k);
    
    for j in 0..k {
        // Multiply categorical indicator by continuous variable
        for i in 0..n {
            interaction[[i, j]] = categorical[[i, j]] * continuous[i];
        }
        
        names.push(format!("{}:{}", cat_names[j], cont_name));
    }
    
    (interaction, names)
}
```

### Continuous × Continuous

Simple product:

```rust
pub fn build_continuous_continuous_interaction(
    x1: &Array1<f64>,
    x2: &Array1<f64>,
    name1: &str,
    name2: &str,
) -> (Array1<f64>, String) {
    let interaction = x1 * x2;
    let name = format!("{}:{}", name1, name2);
    (interaction, name)
}
```

## The Full Design Matrix Builder

Combines all components:

```rust
pub fn build_design_matrix(
    data: &DataSource,
    terms: &[Term],
    include_intercept: bool,
) -> Result<(Array2<f64>, Vec<String>)> {
    let n = data.n_rows();
    let mut columns: Vec<Array1<f64>> = Vec::new();
    let mut names: Vec<String> = Vec::new();
    
    // 1. Add intercept
    if include_intercept {
        columns.push(Array1::ones(n));
        names.push("Intercept".to_string());
    }
    
    // 2. Process each term
    for term in terms {
        match term {
            Term::Continuous(name) => {
                columns.push(data.get_column(name)?);
                names.push(name.clone());
            }
            
            Term::Categorical(name) => {
                let values = data.get_categorical(name)?;
                let (encoded, term_names) = encode_categorical(&values, None);
                for col in encoded.axis_iter(Axis(1)) {
                    columns.push(col.to_owned());
                }
                names.extend(term_names);
            }
            
            Term::Spline { name, kind, df } => {
                let x = data.get_column(name)?;
                let basis = match kind {
                    SplineKind::BSpline => bs(&x, *df, 3, None),
                    SplineKind::NaturalSpline => ns(&x, *df, None),
                };
                for (j, col) in basis.axis_iter(Axis(1)).enumerate() {
                    columns.push(col.to_owned());
                    names.push(format!("{}({}, df={})_{}", kind, name, df, j));
                }
            }
            
            Term::Interaction(t1, t2) => {
                // Build interaction from sub-terms
                // ...
            }
        }
    }
    
    // 3. Stack columns
    let p = columns.len();
    let mut x = Array2::zeros((n, p));
    for (j, col) in columns.into_iter().enumerate() {
        x.column_mut(j).assign(&col);
    }
    
    Ok((x, names))
}
```

## Data Source Abstraction

Supports both NumPy arrays and DataFrames:

```rust
pub enum DataSource<'a> {
    Arrays {
        y: &'a Array1<f64>,
        x: &'a Array2<f64>,
        column_names: Option<&'a [String]>,
    },
    DataFrame {
        columns: HashMap<String, ColumnData>,
    },
}

pub enum ColumnData {
    Float64(Array1<f64>),
    String(Vec<String>),
    Int64(Array1<i64>),
}
```

## Performance Considerations

### 1. Pre-allocation

Avoid repeated allocations:

```rust
// Pre-calculate total columns
let total_cols = terms.iter().map(|t| t.n_columns()).sum();
let mut x = Array2::zeros((n, total_cols));
```

### 2. Parallel Construction

For large datasets:

```rust
use rayon::prelude::*;

let columns: Vec<_> = terms.par_iter()
    .map(|term| build_term_columns(term, data))
    .collect();
```

### 3. Sparse Consideration

Categorical encoding produces sparse columns (mostly zeros). For very high cardinality, consider sparse matrix representations.

## Testing

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_categorical_encoding() {
        let values = vec!["A".into(), "B".into(), "A".into(), "C".into()];
        let (encoded, names) = encode_categorical(&values, None);
        
        assert_eq!(encoded.shape(), &[4, 2]);  // 3 levels - 1
        assert_eq!(names, vec!["Category_B", "Category_C"]);
        
        // Check encoding
        assert_eq!(encoded[[0, 0]], 0.0);  // A → [0, 0]
        assert_eq!(encoded[[1, 0]], 1.0);  // B → [1, 0]
        assert_eq!(encoded[[3, 1]], 1.0);  // C → [0, 1]
    }
    
    #[test]
    fn test_interaction_dimensions() {
        let cat1 = Array2::zeros((10, 3));  // 4 levels
        let cat2 = Array2::zeros((10, 2));  // 3 levels
        
        let (interaction, names) = build_categorical_categorical_interaction(
            &cat1, &cat2, &["A", "B", "C"], &["X", "Y"]
        );
        
        assert_eq!(interaction.shape(), &[10, 6]);  // 3 × 2
        assert_eq!(names.len(), 6);
    }
}
```
