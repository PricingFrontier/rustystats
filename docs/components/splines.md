# Splines Component

Spline basis functions allow modeling non-linear relationships while staying within the GLM framework. RustyStats implements B-splines and natural splines.

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

def bs(x, df=None, knots=None, degree=3, boundary_knots=None, 
       include_intercept=False):
    """
    B-spline basis matrix.
    
    Parameters
    ----------
    x : array-like
        Values at which to evaluate the spline basis
    df : int, optional
        Degrees of freedom (number of columns)
    degree : int, default 3
        Polynomial degree (3 = cubic)
    boundary_knots : tuple, optional
        (min, max) boundary knots
    include_intercept : bool, default False
        Whether to include constant basis function
    
    Returns
    -------
    ndarray
        Basis matrix of shape (len(x), df)
    """
    x = np.asarray(x, dtype=np.float64)
    
    if df is None:
        df = 4  # Default
    
    # Call Rust implementation
    from rustystats._rustystats import bs_basis
    basis = bs_basis(x, df, degree, boundary_knots)
    
    if not include_intercept:
        # Drop first column (intercept absorbed)
        basis = basis[:, 1:]
    
    return basis
```

## Formula Integration

Splines can be used directly in formulas:

```python
result = rs.glm(
    "y ~ bs(age, df=5) + ns(income, df=4) + C(region)",
    data=data,
    family="poisson"
).fit()
```

The formula parser recognizes `bs()` and `ns()` terms:

```python
# In formula.py
class SplineTerm:
    def __init__(self, variable, df, kind='bs', degree=3):
        self.variable = variable
        self.df = df
        self.kind = kind  # 'bs' or 'ns'
        self.degree = degree
    
    def build(self, data):
        x = data[self.variable].to_numpy()
        if self.kind == 'bs':
            return bs(x, df=self.df, degree=self.degree)
        else:
            return ns(x, df=self.df)
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

## Monotonic Splines (I-Splines)

Monotonic splines constrain the fitted curve to be monotonically increasing or decreasing. This is essential for actuarial applications where business logic dictates monotonic relationships.

### Mathematical Background

I-splines are integrated M-splines (normalized B-splines):
- Each I-spline basis function \(I_j(x)\) increases from 0 to 1
- With non-negative coefficients \(\beta_j \geq 0\), any linear combination is monotonically increasing
- For decreasing functions, use \(1 - I_j(x)\) or set `increasing=False`

Reference: Ramsay, J.O. (1988). *Monotone Regression Splines in Action*. Statistical Science.

### Implementation

```rust
/// Compute I-spline (monotonic spline) basis matrix
pub fn is_basis(
    x: &Array1<f64>,
    df: usize,
    degree: usize,
    boundary_knots: Option<(f64, f64)>,
    increasing: bool,
) -> Array2<f64> {
    // Compute B-spline basis
    let knots = compute_knots(x, df, degree, boundary_knots);
    
    // I-splines are cumulative sums of B-splines
    // I_j(x) = sum_{i >= j} B_i(x)
    let bs_values = bspline_all_basis_at_point(x, degree, &knots, n_basis);
    
    // Cumulative sum from right to left
    let mut cumsum = 0.0;
    for j in (0..n_basis).rev() {
        cumsum += bs_values[j];
        result[j] = cumsum.clamp(0.0, 1.0);
    }
    
    // For decreasing: flip values
    if !increasing {
        result[j] = 1.0 - result[j];
    }
}
```

### Properties

| Property | I-Spline |
|----------|----------|
| Range | [0, 1] |
| At x_min | 0 (increasing) or 1 (decreasing) |
| At x_max | 1 (increasing) or 0 (decreasing) |
| Shape constraint | Monotonic with β ≥ 0 |
| Use case | Age effects, risk scores |

### Python API

```python
def ms(x, df=5, degree=3, boundary_knots=None, increasing=True):
    """
    Monotonic spline (I-spline) basis matrix.
    
    Parameters
    ----------
    x : array-like
        Values at which to evaluate the spline basis
    df : int, default 5
        Degrees of freedom (number of basis functions)
    degree : int, default 3
        Polynomial degree (3 = cubic)
    boundary_knots : tuple, optional
        (min, max) boundary knots
    increasing : bool, default True
        If True, basis for increasing function; if False, decreasing
    
    Returns
    -------
    ndarray
        Basis matrix of shape (len(x), df). All values in [0, 1].
    """
```

### Formula Integration

```python
# Monotonically increasing effect (e.g., age → risk)
result = rs.glm(
    "ClaimNb ~ ms(Age, df=5) + C(Region)",
    data=data,
    family="poisson",
    offset="Exposure"
).fit()

# Monotonically decreasing effect (e.g., vehicle value with age)
result = rs.glm(
    "ClaimAmt ~ ms(VehAge, df=4, increasing=false)",
    data=data,
    family="gamma"
).fit()

# Combine with other spline types
result = rs.glm(
    "y ~ ms(age, df=5) + bs(income, df=4) + ns(experience, df=3)",
    data=data,
    family="gaussian"
).fit()
```

### When to Use Monotonic Splines

| Use Case | Recommendation |
|----------|---------------|
| Age → claim frequency | `ms(age, increasing=True)` |
| Vehicle age → value | `ms(veh_age, increasing=False)` |
| Credit score → risk | `ms(score, increasing=False)` |
| Income → spending | Domain-dependent |

**Key advantage**: Monotonic splines prevent implausible "wiggles" in the fitted curve that can occur with unconstrained splines.

## Performance

| Dataset Size | df | Time (release) |
|-------------|-----|----------------|
| 10K | 5 | ~5ms |
| 100K | 10 | ~50ms |
| 1M | 10 | ~500ms |

Performance scales as O(n × df × degree).
