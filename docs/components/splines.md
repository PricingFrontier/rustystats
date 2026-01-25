# Splines Component

Spline basis functions allow modeling non-linear relationships while staying within the GLM framework. RustyStats implements B-splines and natural splines with **automatic smoothness selection by default**.

## Simplified API

The spline API uses sensible defaults:

```python
# Default: penalized smooth with automatic tuning via GCV
rs.glm("y ~ bs(age) + ns(income)", data, family="poisson").fit()

# Fixed degrees of freedom (no penalty)
rs.glm("y ~ bs(age, df=5) + ns(income, df=4)", data, family="poisson").fit()

# Monotonic effects (increasing/decreasing)
rs.glm("y ~ bs(age, monotonicity='increasing')", data, family="poisson").fit()
```

| Call | Behavior |
|------|----------|
| `bs(x)` | Penalized smooth, k=10, auto-tuned via GCV |
| `bs(x, df=5)` | Fixed 5 degrees of freedom, no penalty |
| `bs(x, k=15)` | Penalized smooth with 15 basis functions |
| `bs(x, monotonicity='increasing')` | Monotonic constraint (I-spline basis) |

## Code Location

```
crates/rustystats-core/src/splines/
└── mod.rs  # All spline functionality

python/rustystats/splines.py  # Python API
```

## Why Splines?

Instead of assuming a linear effect:
\[
\eta = \beta_0 + \beta_1 x
\]

Splines allow flexible shapes:
\[
\eta = \beta_0 + \sum_{j=1}^{k} \beta_j B_j(x)
\]

where \(B_j(x)\) are basis functions.

## B-Splines

### Mathematical Background

B-splines are piecewise polynomials defined by:
- **Degree**: Polynomial degree (default 3 = cubic)
- **Knots**: Points where pieces join

The Cox-de Boor recursion formula:

\[
B_{i,0}(x) = \begin{cases} 1 & \text{if } t_i \leq x < t_{i+1} \\ 0 & \text{otherwise} \end{cases}
\]

\[
B_{i,k}(x) = \frac{x - t_i}{t_{i+k} - t_i} B_{i,k-1}(x) + \frac{t_{i+k+1} - x}{t_{i+k+1} - t_{i+1}} B_{i+1,k-1}(x)
\]

### Implementation

```rust
pub const DEFAULT_DEGREE: usize = 3;

/// Compute B-spline basis matrix
pub fn bs_basis(
    x: &Array1<f64>,
    knots: &[f64],
    degree: usize,
) -> Array2<f64> {
    let n = x.len();
    let n_basis = knots.len() - degree - 1;
    let mut basis = Array2::zeros((n, n_basis));
    
    for i in 0..n {
        for j in 0..n_basis {
            basis[[i, j]] = b_spline_value(x[i], j, degree, knots);
        }
    }
    
    basis
}

/// Single B-spline basis function via Cox-de Boor
fn b_spline_value(x: f64, i: usize, k: usize, knots: &[f64]) -> f64 {
    if k == 0 {
        // Base case: indicator function
        if x >= knots[i] && x < knots[i + 1] {
            1.0
        } else if i + 1 == knots.len() - 1 && x == knots[i + 1] {
            // Include right endpoint
            1.0
        } else {
            0.0
        }
    } else {
        // Recursive case
        let left_denom = knots[i + k] - knots[i];
        let left = if left_denom.abs() > 1e-10 {
            (x - knots[i]) / left_denom * b_spline_value(x, i, k - 1, knots)
        } else {
            0.0
        };
        
        let right_denom = knots[i + k + 1] - knots[i + 1];
        let right = if right_denom.abs() > 1e-10 {
            (knots[i + k + 1] - x) / right_denom * b_spline_value(x, i + 1, k - 1, knots)
        } else {
            0.0
        };
        
        left + right
    }
}
```

### Knot Placement

Knots are placed at quantiles of the data:

```rust
pub fn compute_knots(
    x: &Array1<f64>,
    n_interior_knots: usize,
    degree: usize,
    boundary_knots: Option<(f64, f64)>,
) -> Vec<f64> {
    let (x_min, x_max) = boundary_knots.unwrap_or_else(|| {
        let mut sorted = x.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (sorted[0], sorted[sorted.len() - 1])
    });
    
    let mut knots = Vec::with_capacity(n_interior_knots + 2 * (degree + 1));
    
    // Boundary knots (repeated degree+1 times)
    for _ in 0..=degree {
        knots.push(x_min);
    }
    
    // Interior knots at quantiles
    for i in 1..=n_interior_knots {
        let q = i as f64 / (n_interior_knots + 1) as f64;
        knots.push(quantile(x, q));
    }
    
    // Right boundary knots
    for _ in 0..=degree {
        knots.push(x_max);
    }
    
    knots
}
```

### Degrees of Freedom

The `df` parameter controls flexibility:
- `df` = number of basis functions
- Interior knots = `df - degree - 1` (for B-splines without intercept)

```rust
pub fn bs(
    x: &Array1<f64>,
    df: usize,
    degree: usize,
    boundary_knots: Option<(f64, f64)>,
) -> Array2<f64> {
    // df includes intercept basis
    let n_interior = df - degree - 1;
    let knots = compute_knots(x, n_interior, degree, boundary_knots);
    bs_basis(x, &knots, degree)
}
```

## Natural Splines

Natural splines add constraints for better extrapolation:
- Linear beyond the boundary knots
- Reduces df by 2 compared to B-splines

### Implementation

```rust
pub fn ns_basis(
    x: &Array1<f64>,
    df: usize,
    boundary_knots: Option<(f64, f64)>,
) -> Array2<f64> {
    let degree = 3;  // Natural splines are cubic
    
    // Compute B-spline basis
    let n_interior = df - 1;  // Natural splines use more interior knots
    let knots = compute_knots(x, n_interior, degree, boundary_knots);
    let bs = bs_basis(x, &knots, degree);
    
    // Apply natural spline constraints
    // The constraint matrix ensures:
    // f''(x_min) = 0 and f''(x_max) = 0
    let constraint_matrix = compute_natural_spline_constraint(&knots, degree);
    
    // Project B-spline basis onto constrained space
    bs.dot(&constraint_matrix)
}
```

### Properties

| Property | B-Spline | Natural Spline |
|----------|----------|----------------|
| Behavior at boundaries | Polynomial | Linear |
| Extrapolation | Unstable | Stable |
| df for same flexibility | Higher | Lower |
| Recommended for | Interpolation | Prediction |

## Python API

```python
# python/rustystats/splines.py

def bs(x, df=None, k=None, degree=3, boundary_knots=None, 
       include_intercept=False, monotonicity=None):
    """
    B-spline basis matrix.
    
    Parameters
    ----------
    x : array-like
        Values at which to evaluate the spline basis
    df : int, optional
        Degrees of freedom (fixed). Use for fixed-complexity splines.
    k : int, optional
        Maximum basis size for penalized smooth terms with automatic
        smoothness selection via GCV. Default: k=10 if neither df nor k provided.
    degree : int, default 3
        Polynomial degree (3 = cubic)
    boundary_knots : tuple, optional
        (min, max) boundary knots
    include_intercept : bool, default False
        Whether to include constant basis function
    monotonicity : str, optional
        'increasing' or 'decreasing' for monotonic constraints (uses I-splines)
    
    Returns
    -------
    ndarray
        Basis matrix of shape (len(x), basis_size)
    """
```

```python
def ns(x, df=None, k=None, boundary_knots=None, include_intercept=False):
    """
    Natural cubic spline basis matrix.
    
    Parameters
    ----------
    x : array-like
        Values at which to evaluate the spline basis
    df : int, optional
        Degrees of freedom (fixed). Use for fixed-complexity splines.
    k : int, optional
        Maximum basis size for penalized smooth terms with automatic
        smoothness selection via GCV. Default: k=10 if neither df nor k provided.
    boundary_knots : tuple, optional
        (min, max) boundary knots - spline is linear beyond these
    include_intercept : bool, default False
        Whether to include constant basis function
    
    Returns
    -------
    ndarray
        Basis matrix of shape (len(x), basis_size)
    """
```

## Formula Integration

Splines can be used directly in formulas:

```python
# Default: penalized smooth with automatic tuning
result = rs.glm(
    "y ~ bs(age) + ns(income) + C(region)",
    data=data,
    family="poisson"
).fit()

# Fixed df (no penalty)
result = rs.glm(
    "y ~ bs(age, df=5) + ns(income, df=4) + C(region)",
    data=data,
    family="poisson"
).fit()

# Monotonic constraint
result = rs.glm(
    "y ~ bs(age, monotonicity='increasing') + C(region)",
    data=data,
    family="poisson"
).fit()
```

The formula parser recognizes `bs()` and `ns()` terms with their parameters:

```python
# In splines.py
class SplineTerm:
    def __init__(self, var_name, spline_type='bs', df=10, degree=3,
                 boundary_knots=None, monotonicity=None):
        self.var_name = var_name
        self.spline_type = spline_type  # 'bs' or 'ns'
        self.df = df
        self.degree = degree
        self.monotonicity = monotonicity
        self._is_smooth = False  # Set True for penalized terms
```

## Parallel Computation

For large datasets, parallelize over observations:

```rust
pub fn bs_basis_parallel(
    x: &Array1<f64>,
    knots: &[f64],
    degree: usize,
) -> Array2<f64> {
    let n = x.len();
    let n_basis = knots.len() - degree - 1;
    
    // Parallel over rows
    let rows: Vec<Vec<f64>> = (0..n).into_par_iter()
        .map(|i| {
            (0..n_basis)
                .map(|j| b_spline_value(x[i], j, degree, knots))
                .collect()
        })
        .collect();
    
    // Assemble matrix
    let mut basis = Array2::zeros((n, n_basis));
    for (i, row) in rows.into_iter().enumerate() {
        for (j, val) in row.into_iter().enumerate() {
            basis[[i, j]] = val;
        }
    }
    
    basis
}
```

## Testing

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_bs_partition_of_unity() {
        // B-splines should sum to 1 at any point
        let x = array![0.0, 0.25, 0.5, 0.75, 1.0];
        let knots = vec![0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0];
        let basis = bs_basis(&x, &knots, 3);
        
        for i in 0..x.len() {
            let sum: f64 = basis.row(i).sum();
            assert!((sum - 1.0).abs() < 1e-10);
        }
    }
    
    #[test]
    fn test_ns_linear_extrapolation() {
        // Natural splines should extrapolate linearly
        let x_train = Array1::linspace(0.0, 1.0, 100);
        let x_test = array![-0.5, 1.5];
        
        // Fit basis on training data
        let basis_train = ns_basis(&x_train, 4, Some((0.0, 1.0)));
        let basis_test = ns_basis(&x_test, 4, Some((0.0, 1.0)));
        
        // Linear extrapolation means second derivative = 0
        // (Would need derivative implementation to test properly)
    }
    
    #[test]
    fn test_df_vs_columns() {
        let x = Array1::linspace(0.0, 1.0, 50);
        
        for df in 3..10 {
            let basis = bs(&x, df, 3, None);
            assert_eq!(basis.ncols(), df);
        }
    }
}
```

## Monotonic Splines

Monotonic splines constrain the fitted curve to be monotonically increasing or decreasing. Use `bs()` with the `monotonicity` parameter.

### Mathematical Background

Internally uses I-splines (integrated M-splines):
- Each I-spline basis function \(I_j(x)\) increases from 0 to 1
- With non-negative coefficients \(\beta_j \geq 0\), any linear combination is monotonically increasing
- For decreasing functions, the basis is flipped

Reference: Ramsay, J.O. (1988). *Monotone Regression Splines in Action*. Statistical Science.

### Properties

| Property | Monotonic Spline |
|----------|------------------|
| Range | [0, 1] |
| At x_min | 0 (increasing) or 1 (decreasing) |
| At x_max | 1 (increasing) or 0 (decreasing) |
| Shape constraint | Monotonic with β ≥ 0 |
| Use case | Age effects, risk scores |

### Usage

```python
# Via bs() with monotonicity parameter
basis = rs.bs(x, monotonicity='increasing')   # Monotonically increasing
basis = rs.bs(x, df=5, monotonicity='decreasing')  # Fixed df, decreasing

# In formulas
result = rs.glm(
    "ClaimNb ~ bs(Age, monotonicity='increasing') + C(Region)",
    data=data,
    family="poisson",
    offset="Exposure"
).fit()

# Combine with other spline types
result = rs.glm(
    "y ~ bs(age, monotonicity='increasing') + bs(income, df=4) + ns(experience)",
    data=data,
    family="gaussian"
).fit()
```

### When to Use Monotonic Splines

| Use Case | Formula |
|----------|---------|
| Age → claim frequency | `bs(age, monotonicity='increasing')` |
| Vehicle age → value | `bs(veh_age, monotonicity='decreasing')` |
| Credit score → risk | `bs(score, df=5, monotonicity='decreasing')` |
| Income → spending | Domain-dependent |

**Key advantage**: Monotonic splines prevent implausible "wiggles" in the fitted curve that can occur with unconstrained splines.

### Deprecated: `ms()` Function

The standalone `ms()` function is deprecated. Use `bs(monotonicity=...)` instead:

```python
# Old (deprecated)
basis = rs.ms(x, df=5, increasing=True)

# New
basis = rs.bs(x, df=5, monotonicity='increasing')
```

## Performance

| Dataset Size | df | Time (release) |
|-------------|-----|----------------|
| 10K | 5 | ~5ms |
| 100K | 10 | ~50ms |
| 1M | 10 | ~500ms |

Performance scales as O(n × df × degree).
