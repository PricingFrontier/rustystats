# Rust Core Library

The `rustystats-core` crate contains all mathematical computation. This chapter provides a deep dive into its structure and implementation.

## Crate Overview

```
crates/rustystats-core/
├── Cargo.toml          # Dependencies and metadata
└── src/
    ├── lib.rs          # Entry point, re-exports
    ├── error.rs        # Error types
    ├── families/       # Distribution families
    ├── links/          # Link functions
    ├── solvers/        # IRLS, coordinate descent
    ├── inference/      # Standard errors, p-values
    ├── diagnostics/    # Residuals, calibration
    ├── splines/        # B-splines, natural splines
    ├── formula/        # Formula parsing
    ├── design_matrix/  # Design matrix construction
    ├── target_encoding/# CatBoost-style encoding
    ├── regularization/ # Penalty configuration
    └── interactions/   # Interaction term handling
```

## Dependencies

From `Cargo.toml`:

```toml
[dependencies]
ndarray = "0.15"           # N-dimensional arrays
nalgebra = "0.32"          # Linear algebra (Cholesky, etc.)
rayon = "1.8"              # Parallel iterators
statrs = "0.16"            # Statistical distributions
thiserror = "1.0"          # Error handling
```

### Why These Crates?

| Crate | Purpose | Why Chosen |
|-------|---------|------------|
| `ndarray` | Array operations | NumPy-like API, good performance |
| `nalgebra` | Linear algebra | Robust decompositions |
| `rayon` | Parallelism | Simple, efficient data parallelism |
| `statrs` | Statistics | p-values from distributions |
| `thiserror` | Errors | Clean error type derivation |

## Module: lib.rs

The entry point declares modules and re-exports public items:

```rust
// Module declarations
pub mod error;
pub mod families;
pub mod links;
pub mod solvers;
pub mod inference;
pub mod diagnostics;
pub mod splines;
pub mod formula;
pub mod design_matrix;
pub mod target_encoding;
pub mod regularization;
pub mod interactions;

// Re-exports for convenience
pub use error::{RustyStatsError, Result};
pub use families::Family;
pub use links::Link;
pub use solvers::{IRLSConfig, IRLSResult, fit_glm, fit_glm_full};
// ... more re-exports
```

This allows users to write `use rustystats_core::Family` instead of `use rustystats_core::families::Family`.

## Module: error.rs

Custom error types using `thiserror`:

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum RustyStatsError {
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    
    #[error("Convergence failed after {iterations} iterations (tolerance: {tolerance})")]
    ConvergenceFailure { iterations: usize, tolerance: f64 },
    
    #[error("Numerical error: {0}")]
    NumericalError(String),
    
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
}

pub type Result<T> = std::result::Result<T, RustyStatsError>;
```

## Module: families/

### The Family Trait

```rust
pub trait Family: Send + Sync {
    fn name(&self) -> &str;
    fn variance(&self, mu: &Array1<f64>) -> Array1<f64>;
    fn unit_deviance(&self, y: &Array1<f64>, mu: &Array1<f64>) -> Array1<f64>;
    fn deviance(&self, y: &Array1<f64>, mu: &Array1<f64>, 
                weights: Option<&Array1<f64>>) -> f64;
    fn default_link(&self) -> Box<dyn Link>;
    fn initialize_mu(&self, y: &Array1<f64>) -> Array1<f64>;
    fn is_valid_mu(&self, mu: &Array1<f64>) -> bool;
}
```

### Example Implementation: Poisson

```rust
// families/poisson.rs
pub struct PoissonFamily;

impl Family for PoissonFamily {
    fn name(&self) -> &str { "Poisson" }
    
    fn variance(&self, mu: &Array1<f64>) -> Array1<f64> {
        mu.clone()  // V(μ) = μ
    }
    
    fn unit_deviance(&self, y: &Array1<f64>, mu: &Array1<f64>) -> Array1<f64> {
        Zip::from(y).and(mu)
            .map_collect(|&yi, &mui| {
                if yi > 0.0 {
                    2.0 * (yi * (yi / mui).ln() - (yi - mui))
                } else {
                    2.0 * mui  // Limit as y → 0
                }
            })
    }
    
    fn default_link(&self) -> Box<dyn Link> {
        Box::new(LogLink)
    }
    
    fn initialize_mu(&self, y: &Array1<f64>) -> Array1<f64> {
        y.mapv(|yi| (yi + 0.1).max(0.1))  // Avoid log(0)
    }
    
    fn is_valid_mu(&self, mu: &Array1<f64>) -> bool {
        mu.iter().all(|&m| m > 0.0)
    }
}
```

## Module: links/

### The Link Trait

```rust
pub trait Link: Send + Sync {
    fn name(&self) -> &str;
    fn link(&self, mu: &Array1<f64>) -> Array1<f64>;
    fn inverse(&self, eta: &Array1<f64>) -> Array1<f64>;
    fn derivative(&self, mu: &Array1<f64>) -> Array1<f64>;
}
```

### Example Implementation: Log

```rust
// links/log.rs
pub struct LogLink;

impl Link for LogLink {
    fn name(&self) -> &str { "Log" }
    
    fn link(&self, mu: &Array1<f64>) -> Array1<f64> {
        mu.mapv(|m| m.max(1e-10).ln())
    }
    
    fn inverse(&self, eta: &Array1<f64>) -> Array1<f64> {
        eta.mapv(|e| e.exp())
    }
    
    fn derivative(&self, mu: &Array1<f64>) -> Array1<f64> {
        mu.mapv(|m| 1.0 / m.max(1e-10))
    }
}
```

## Module: solvers/

### IRLS Implementation

The IRLS solver is the heart of GLM fitting:

```rust
pub fn fit_glm_full(
    y: &Array1<f64>,
    x: &Array2<f64>,
    family: &dyn Family,
    link: &dyn Link,
    config: &IRLSConfig,
    offset: Option<&Array1<f64>>,
    weights: Option<&Array1<f64>>,
) -> Result<IRLSResult> {
    let n = y.len();
    let p = x.ncols();
    
    // Initialize
    let mut mu = family.initialize_mu(y);
    let mut eta = link.link(&mu);
    
    // Add offset if present
    if let Some(off) = offset {
        eta = &eta + off;
        mu = link.inverse(&eta);
    }
    
    let mut prev_deviance = family.deviance(y, &mu, weights);
    
    for iter in 0..config.max_iterations {
        // Step 1: Compute working weights
        let var = family.variance(&mu);
        let link_deriv = link.derivative(&mu);
        let w = compute_weights(&var, &link_deriv, weights, config.min_weight);
        
        // Step 2: Compute working response
        let z = compute_working_response(y, &mu, &eta, &link_deriv);
        
        // Step 3: Solve weighted least squares (X'WX)β = X'Wz
        let (beta, cov_unscaled) = solve_wls(x, &w, &z)?;
        
        // Step 4: Update predictions
        eta = x.dot(&beta);
        if let Some(off) = offset {
            eta = &eta + off;
        }
        mu = link.inverse(&eta);
        
        // Step 5: Check convergence
        let deviance = family.deviance(y, &mu, weights);
        let rel_change = (deviance - prev_deviance).abs() / prev_deviance.max(1e-10);
        
        if rel_change < config.tolerance {
            return Ok(IRLSResult { 
                coefficients: beta,
                fitted_values: mu,
                deviance,
                iterations: iter + 1,
                converged: true,
                covariance_unscaled: cov_unscaled,
                // ...
            });
        }
        
        prev_deviance = deviance;
    }
    
    // Did not converge
    Ok(IRLSResult { converged: false, ... })
}
```

### Parallel WLS

The weighted least squares step is parallelized:

```rust
fn solve_wls(
    x: &Array2<f64>,
    w: &Array1<f64>,
    z: &Array1<f64>,
) -> Result<(Array1<f64>, Array2<f64>)> {
    let n = x.nrows();
    let p = x.ncols();
    
    // Parallel computation of X'WX and X'Wz
    let (xtwx, xtwz) = (0..n).into_par_iter()
        .fold(
            || (Array2::<f64>::zeros((p, p)), Array1::<f64>::zeros(p)),
            |(mut acc_xtwx, mut acc_xtwz), i| {
                let xi = x.row(i);
                let wi = w[i];
                let wz_i = wi * z[i];
                
                // X'Wz contribution
                for j in 0..p {
                    acc_xtwz[j] += xi[j] * wz_i;
                }
                
                // X'WX contribution (only upper triangle for efficiency)
                for j in 0..p {
                    for k in j..p {
                        acc_xtwx[[j, k]] += wi * xi[j] * xi[k];
                    }
                }
                
                (acc_xtwx, acc_xtwz)
            }
        )
        .reduce(
            || (Array2::zeros((p, p)), Array1::zeros(p)),
            |(a1, b1), (a2, b2)| (a1 + a2, b1 + b2)
        );
    
    // Fill lower triangle
    for j in 0..p {
        for k in (j+1)..p {
            xtwx[[k, j]] = xtwx[[j, k]];
        }
    }
    
    // Solve using Cholesky decomposition
    let beta = cholesky_solve(&xtwx, &xtwz)?;
    let cov = cholesky_inverse(&xtwx)?;
    
    Ok((beta, cov))
}
```

### Coordinate Descent

For regularized models:

```rust
pub fn fit_glm_coordinate_descent(
    y: &Array1<f64>,
    x: &Array2<f64>,
    family: &dyn Family,
    link: &dyn Link,
    config: &IRLSConfig,
    reg_config: &RegularizationConfig,
    offset: Option<&Array1<f64>>,
    weights: Option<&Array1<f64>>,
) -> Result<IRLSResult> {
    // Outer loop: IRLS for working response/weights
    // Inner loop: Coordinate descent for penalized WLS
    
    for outer_iter in 0..config.max_iterations {
        // Compute working response and weights (same as IRLS)
        let (z, w) = compute_working_response_and_weights(...);
        
        // Precompute Gram matrix X'WX (done once per outer iteration)
        let gram = compute_gram_matrix(x, &w);
        
        // Coordinate descent inner loop
        for inner_iter in 0..max_inner_iterations {
            for j in 0..p {
                // Compute partial residual
                let r_j = compute_partial_residual(&z, x, &beta, j, &w);
                
                // Update β_j with soft thresholding
                let xtwx_jj = gram[[j, j]];
                let xtwz_j = weighted_dot(&x.column(j), &w, &r_j);
                
                beta[j] = soft_threshold(
                    xtwz_j,
                    reg_config.alpha * reg_config.l1_ratio
                ) / (xtwx_jj + reg_config.alpha * (1.0 - reg_config.l1_ratio));
            }
        }
        
        // Update predictions and check convergence
        // ...
    }
}
```

## Module: inference/

### Standard Errors and P-values

```rust
pub fn pvalue_z(z: f64) -> f64 {
    let normal = Normal::new(0.0, 1.0).unwrap();
    2.0 * (1.0 - normal.cdf(z.abs()))
}

pub fn confidence_interval_z(
    estimate: f64,
    se: f64,
    alpha: f64,
) -> (f64, f64) {
    let normal = Normal::new(0.0, 1.0).unwrap();
    let z_crit = normal.inverse_cdf(1.0 - alpha / 2.0);
    (estimate - z_crit * se, estimate + z_crit * se)
}
```

### Robust Standard Errors

HC0, HC1, HC2, HC3 sandwich estimators:

```rust
pub fn robust_covariance(
    x: &Array2<f64>,
    resid: &Array1<f64>,
    irls_weights: &Array1<f64>,
    prior_weights: &Array1<f64>,
    cov_unscaled: &Array2<f64>,
    hc_type: HCType,
) -> Array2<f64> {
    let n = x.nrows();
    let p = x.ncols();
    let df = n - p;
    
    // Compute "meat" matrix: X' diag(u²) X
    // where u depends on HC type
    let meat = compute_meat_matrix(x, resid, irls_weights, prior_weights, hc_type);
    
    // Sandwich: (X'WX)⁻¹ Meat (X'WX)⁻¹
    cov_unscaled.dot(&meat).dot(cov_unscaled)
}
```

## Module: splines/

B-spline and natural spline basis functions:

```rust
pub fn bs_basis(
    x: &Array1<f64>,
    knots: &[f64],
    degree: usize,
) -> Array2<f64> {
    // Cox-de Boor recursive algorithm
    let n = x.len();
    let n_basis = knots.len() - degree - 1;
    let mut basis = Array2::zeros((n, n_basis));
    
    for i in 0..n {
        for j in 0..n_basis {
            basis[[i, j]] = b_spline_basis(x[i], j, degree, knots);
        }
    }
    
    basis
}

fn b_spline_basis(x: f64, i: usize, k: usize, knots: &[f64]) -> f64 {
    if k == 0 {
        // Base case: indicator function
        if knots[i] <= x && x < knots[i + 1] {
            1.0
        } else {
            0.0
        }
    } else {
        // Recursive case: Cox-de Boor
        let left = if (knots[i + k] - knots[i]).abs() > 1e-10 {
            (x - knots[i]) / (knots[i + k] - knots[i]) 
                * b_spline_basis(x, i, k - 1, knots)
        } else {
            0.0
        };
        
        let right = if (knots[i + k + 1] - knots[i + 1]).abs() > 1e-10 {
            (knots[i + k + 1] - x) / (knots[i + k + 1] - knots[i + 1])
                * b_spline_basis(x, i + 1, k - 1, knots)
        } else {
            0.0
        };
        
        left + right
    }
}
```

## Testing

Each module includes tests:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_poisson_variance() {
        let family = PoissonFamily;
        let mu = array![1.0, 2.0, 5.0];
        let var = family.variance(&mu);
        assert_eq!(var, mu);
    }
    
    #[test]
    fn test_log_link_inverse() {
        let link = LogLink;
        let eta = array![0.0, 1.0, 2.0];
        let mu = link.inverse(&eta);
        assert_relative_eq!(mu[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(mu[1], E, epsilon = 1e-10);
    }
}
```

Run tests: `cargo test -p rustystats-core`
