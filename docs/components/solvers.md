# Solvers

RustyStats includes two main solvers: IRLS for standard GLMs and Coordinate Descent for regularized models. This chapter covers implementation details and optimization strategies.

## Code Location

```
crates/rustystats-core/src/solvers/
├── mod.rs                  # Re-exports
├── irls.rs                 # IRLS implementation (~1300 lines)
└── coordinate_descent.rs   # Regularized solver (~900 lines)
```

## IRLS Solver

### Configuration

```rust
pub struct IRLSConfig {
    pub max_iterations: usize,  // Default: 25
    pub tolerance: f64,         // Default: 1e-8
    pub min_weight: f64,        // Default: 1e-10
    pub verbose: bool,          // Default: false
}
```

### Main Function Signature

```rust
pub fn fit_glm_full(
    y: &Array1<f64>,
    x: &Array2<f64>,
    family: &dyn Family,
    link: &dyn Link,
    config: &IRLSConfig,
    offset: Option<&Array1<f64>>,
    weights: Option<&Array1<f64>>,
) -> Result<IRLSResult>
```

### Algorithm Steps

```rust
// 1. Initialize
let mut mu = family.initialize_mu(y);
let mut eta = link.link(&mu);

// 2. Main loop
for iter in 0..config.max_iterations {
    // 2a. Compute variance and link derivative
    let variance = family.variance(&mu);
    let link_deriv = link.derivative(&mu);
    
    // 2b. Compute working weights
    //     W = 1 / [V(μ) × g'(μ)²]
    let w = compute_irls_weights(&variance, &link_deriv, weights);
    
    // 2c. Compute working response
    //     z = η + (y - μ) × g'(μ)
    let z = compute_working_response(y, &mu, &eta, &link_deriv);
    
    // 2d. Solve weighted least squares
    //     (X'WX)β = X'Wz
    let (beta, cov) = solve_wls_parallel(x, &w, &z)?;
    
    // 2e. Update predictions
    eta = x.dot(&beta);
    if let Some(off) = offset {
        eta = &eta + off;
    }
    mu = link.inverse(&eta);
    
    // 2f. Check convergence
    let deviance = family.deviance(y, &mu, weights);
    if converged(deviance, prev_deviance, config.tolerance) {
        return Ok(build_result(..., converged: true));
    }
    prev_deviance = deviance;
}
```

### Parallel WLS Solver

The weighted least squares step dominates computation time. We parallelize it:

```rust
fn solve_wls_parallel(
    x: &Array2<f64>,
    w: &Array1<f64>,
    z: &Array1<f64>,
) -> Result<(Array1<f64>, Array2<f64>)> {
    let n = x.nrows();
    let p = x.ncols();
    
    // Parallel fold-reduce for X'WX and X'Wz
    let (xtwx, xtwz) = (0..n).into_par_iter()
        .fold(
            || (vec![0.0; p * p], vec![0.0; p]),
            |(mut gram, mut moment), i| {
                let wi = w[i];
                let wzi = wi * z[i];
                
                for j in 0..p {
                    let xij = x[[i, j]];
                    moment[j] += xij * wzi;
                    
                    // Upper triangle only (symmetric)
                    for k in j..p {
                        gram[j * p + k] += wi * xij * x[[i, k]];
                    }
                }
                (gram, moment)
            }
        )
        .reduce(
            || (vec![0.0; p * p], vec![0.0; p]),
            |(mut g1, mut m1), (g2, m2)| {
                for i in 0..g1.len() { g1[i] += g2[i]; }
                for i in 0..m1.len() { m1[i] += m2[i]; }
                (g1, m1)
            }
        );
    
    // Convert to matrices and solve
    let gram_matrix = reshape_to_symmetric(xtwx, p);
    let moment_vector = Array1::from_vec(xtwz);
    
    cholesky_solve(&gram_matrix, &moment_vector)
}
```

### Numerical Stability

```rust
// Clamp weights to avoid numerical issues
fn compute_irls_weights(
    variance: &Array1<f64>,
    link_deriv: &Array1<f64>,
    prior_weights: Option<&Array1<f64>>,
    min_weight: f64,
) -> Array1<f64> {
    let n = variance.len();
    let mut w = Array1::zeros(n);
    
    for i in 0..n {
        let v = variance[i].max(min_weight);
        let d = link_deriv[i];
        let pw = prior_weights.map_or(1.0, |pw| pw[i]);
        
        w[i] = (pw / (v * d * d)).max(min_weight);
    }
    
    w
}
```

## Coordinate Descent Solver

For regularized GLMs (Lasso, Ridge, Elastic Net).

### Configuration

```rust
pub struct RegularizationConfig {
    pub alpha: f64,        // Overall penalty strength
    pub l1_ratio: f64,     // Mix: 1.0 = Lasso, 0.0 = Ridge
    pub standardize: bool, // Standardize features
}
```

### Algorithm Overview

Coordinate descent optimizes one coefficient at a time while holding others fixed:

```rust
pub fn fit_glm_coordinate_descent(
    y: &Array1<f64>,
    x: &Array2<f64>,
    family: &dyn Family,
    link: &dyn Link,
    irls_config: &IRLSConfig,
    reg_config: &RegularizationConfig,
    offset: Option<&Array1<f64>>,
    weights: Option<&Array1<f64>>,
) -> Result<IRLSResult> {
    // Outer loop: IRLS for working response
    for outer in 0..irls_config.max_iterations {
        // Compute working response and weights
        let (z, w) = compute_working_response_weights(...);
        
        // Precompute Gram matrix (X'WX) once per outer iteration
        let gram = compute_gram_matrix(x, &w);
        
        // Inner loop: Coordinate descent
        for inner in 0..max_inner_iterations {
            let max_change = 0.0;
            
            for j in 0..p {
                // Skip intercept from penalization
                if j == 0 && has_intercept {
                    // Update without penalty
                    beta[j] = update_intercept(...);
                    continue;
                }
                
                // Compute partial residual
                let r_j = z - x_without_j.dot(&beta_without_j);
                
                // Coordinate update with soft thresholding
                let xwz_j = weighted_dot(&x.column(j), &w, &r_j);
                let xwx_jj = gram[[j, j]];
                
                let beta_old = beta[j];
                beta[j] = soft_threshold(
                    xwz_j,
                    reg_config.alpha * reg_config.l1_ratio
                ) / (xwx_jj + reg_config.alpha * (1.0 - reg_config.l1_ratio));
                
                max_change = max_change.max((beta[j] - beta_old).abs());
            }
            
            if max_change < inner_tolerance {
                break;
            }
        }
        
        // Update predictions
        // Check outer convergence
    }
}
```

### Soft Thresholding

The key operation for L1 penalty:

```rust
pub fn soft_threshold(z: f64, gamma: f64) -> f64 {
    if z > gamma {
        z - gamma
    } else if z < -gamma {
        z + gamma
    } else {
        0.0  // Sets coefficient to exactly zero
    }
}
```

Visualization:

```
    Output
      ↑
      │      /
      │     /
      │    /
──────┼────────→ Input
     /│    γ
    / │
   /  │
      │
```

### Active Set Strategy

For efficiency with many zero coefficients:

```rust
// Track which coefficients are non-zero
let mut active_set: Vec<usize> = (0..p).collect();

for cycle in 0..max_cycles {
    // First pass: update only active set
    for &j in &active_set {
        update_coefficient(j, ...);
    }
    
    // Periodically: check all coefficients
    if cycle % 10 == 0 {
        active_set.clear();
        for j in 0..p {
            if beta[j].abs() > 1e-10 || should_enter(j, ...) {
                active_set.push(j);
            }
        }
    }
}
```

### Warm Starts

When computing regularization paths, use previous solution as starting point:

```rust
pub fn lasso_path(
    y: &Array1<f64>,
    x: &Array2<f64>,
    alphas: &[f64],  // Decreasing sequence
    ...
) -> Vec<IRLSResult> {
    let mut results = Vec::with_capacity(alphas.len());
    let mut beta = Array1::zeros(p);  // Start from zero
    
    for &alpha in alphas {
        // Warm start from previous solution
        let config = RegularizationConfig { alpha, ... };
        let result = fit_glm_coordinate_descent(
            ...,
            initial_beta: Some(&beta),
        )?;
        
        beta = result.coefficients.clone();
        results.push(result);
    }
    
    results
}
```

## Performance Optimizations

### 1. Cache-Friendly Access

Store Gram matrix in flat Vec for better cache performance:

```rust
// Instead of Array2<f64>
let gram_flat: Vec<f64> = vec![0.0; p * p];

// Access with manual indexing
let value = gram_flat[i * p + j];
```

### 2. SIMD-Friendly Operations

Structure loops for auto-vectorization:

```rust
// Good: contiguous memory access
for i in 0..n {
    result += x[i] * w[i];
}

// Bad: strided access
for i in 0..n {
    result += x[[i, j]] * w[i];  // Column access is strided
}
```

### 3. Parallel Reduction

Use Rayon's parallel fold-reduce pattern:

```rust
let sum = (0..n).into_par_iter()
    .map(|i| compute(i))
    .reduce(|| 0.0, |a, b| a + b);
```

### 4. Avoid Allocations in Hot Loops

Pre-allocate and reuse buffers:

```rust
// Pre-allocate outside loop
let mut working_buffer = Array1::zeros(n);

for iter in 0..max_iterations {
    // Reuse buffer
    working_buffer.fill(0.0);
    compute_into(&mut working_buffer);
}
```

## Benchmarking

### Simple Benchmark

```rust
#[cfg(test)]
mod benchmarks {
    use super::*;
    use std::time::Instant;
    
    #[test]
    #[ignore]  // Run with: cargo test --release -- --ignored
    fn bench_irls() {
        let n = 100_000;
        let p = 50;
        
        let y = generate_poisson_data(n);
        let x = generate_design_matrix(n, p);
        
        let start = Instant::now();
        let result = fit_glm(&y, &x, &PoissonFamily, &LogLink, &IRLSConfig::default());
        let elapsed = start.elapsed();
        
        println!("IRLS: {:?} ({} iterations)", elapsed, result.unwrap().iterations);
    }
}
```

### Expected Performance

| Dataset Size | Features | Family | Time (release) |
|-------------|----------|--------|----------------|
| 100K | 20 | Poisson | ~100ms |
| 500K | 50 | Poisson | ~800ms |
| 1M | 100 | Gaussian | ~2s |

Performance scales roughly as O(n × p²) for IRLS.
