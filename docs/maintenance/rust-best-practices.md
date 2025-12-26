# Rust Best Practices

This guide covers Rust coding conventions and best practices for maintaining the RustyStats codebase.

## Code Organization

### Module Structure

Each major component follows this pattern:

```
component/
├── mod.rs      # Public interface, re-exports
├── types.rs    # Data structures (optional)
├── impl.rs     # Main implementation (optional)
└── tests.rs    # Tests (or inline with #[cfg(test)])
```

For simpler components, everything goes in `mod.rs`:

```rust
// component/mod.rs

// Types
pub struct MyType { ... }

// Implementation
impl MyType { ... }

// Tests
#[cfg(test)]
mod tests { ... }
```

### Public API

Export public items at the crate root for convenient access:

```rust
// lib.rs
pub use families::Family;
pub use families::poisson::PoissonFamily;
```

Users can then write:
```rust
use rustystats_core::PoissonFamily;
```

---

## Error Handling

### Use thiserror for Error Types

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum RustyStatsError {
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    
    #[error("Convergence failed after {iterations} iterations")]
    ConvergenceFailure { iterations: usize, tolerance: f64 },
    
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
}

pub type Result<T> = std::result::Result<T, RustyStatsError>;
```

### Use ? for Propagation

```rust
pub fn compute(x: &Array1<f64>) -> Result<f64> {
    let validated = validate_input(x)?;  // Propagates error
    let result = process(validated)?;
    Ok(result)
}
```

### Provide Context

```rust
// Bad: loses context
let result = operation().map_err(|_| RustyStatsError::InvalidInput("failed".into()))?;

// Good: preserves context
let result = operation().map_err(|e| 
    RustyStatsError::InvalidInput(format!("operation failed: {}", e))
)?;
```

---

## Traits

### Use Trait Objects for Polymorphism

```rust
pub trait Family: Send + Sync {
    fn variance(&self, mu: &Array1<f64>) -> Array1<f64>;
    // ...
}

pub fn fit_glm(
    y: &Array1<f64>,
    x: &Array2<f64>,
    family: &dyn Family,  // Trait object
    link: &dyn Link,
) -> Result<IRLSResult> {
    // Works with any Family implementation
}
```

### Require Send + Sync for Thread Safety

```rust
// Required for parallel processing with Rayon
pub trait Family: Send + Sync { ... }
pub trait Link: Send + Sync { ... }
```

### Default Implementations Where Sensible

```rust
pub trait Family {
    fn unit_deviance(&self, y: &Array1<f64>, mu: &Array1<f64>) -> Array1<f64>;
    
    // Default implementation using unit_deviance
    fn deviance(&self, y: &Array1<f64>, mu: &Array1<f64>, 
                weights: Option<&Array1<f64>>) -> f64 {
        let unit_dev = self.unit_deviance(y, mu);
        match weights {
            Some(w) => (&unit_dev * w).sum(),
            None => unit_dev.sum(),
        }
    }
}
```

---

## Memory and Performance

### Prefer References Over Clones

```rust
// Bad: unnecessary clone
fn process(data: Array1<f64>) { ... }

// Good: borrow when possible
fn process(data: &Array1<f64>) { ... }
```

### Use Views for Read-Only Access

```rust
// ndarray view (no copy)
let column = matrix.column(j);

// Iterate without owning
for row in matrix.rows() {
    // row is a view, not a copy
}
```

### Clone Explicitly When Needed

```rust
// When you need ownership
let owned_data = data.to_owned();

// When returning modified data
fn double(x: &Array1<f64>) -> Array1<f64> {
    x * 2.0  // Creates new array
}
```

### Pre-allocate Containers

```rust
// Bad: grows dynamically
let mut results = Vec::new();
for i in 0..n {
    results.push(compute(i));
}

// Good: pre-allocate
let mut results = Vec::with_capacity(n);
for i in 0..n {
    results.push(compute(i));
}
```

---

## Parallelism with Rayon

### Use Parallel Iterators

```rust
use rayon::prelude::*;

// Sequential
let sum: f64 = (0..n).map(|i| compute(i)).sum();

// Parallel
let sum: f64 = (0..n).into_par_iter().map(|i| compute(i)).sum();
```

### Parallel Fold-Reduce Pattern

For accumulating results:

```rust
let result = (0..n).into_par_iter()
    .fold(
        || initial_value(),  // Per-thread initializer
        |acc, i| accumulate(acc, i),  // Fold within thread
    )
    .reduce(
        || initial_value(),  // Identity for reduce
        |a, b| combine(a, b),  // Combine thread results
    );
```

### Example: Parallel Matrix Computation

```rust
let (xtwx, xtwz) = (0..n).into_par_iter()
    .fold(
        || (Array2::zeros((p, p)), Array1::zeros(p)),
        |(mut gram, mut moment), i| {
            let xi = x.row(i);
            let wi = w[i];
            
            // Accumulate
            for j in 0..p {
                moment[j] += xi[j] * wi * z[i];
                for k in j..p {
                    gram[[j, k]] += wi * xi[j] * xi[k];
                }
            }
            (gram, moment)
        }
    )
    .reduce(
        || (Array2::zeros((p, p)), Array1::zeros(p)),
        |(g1, m1), (g2, m2)| (g1 + g2, m1 + m2)
    );
```

---

## Numerical Stability

### Avoid Division by Zero

```rust
// Clamp denominators
let safe_denom = denom.max(1e-10);
let result = numerator / safe_denom;
```

### Avoid Log of Zero

```rust
// Clamp before log
let safe_x = x.max(1e-10);
let log_x = safe_x.ln();
```

### Use Stable Algorithms

```rust
// Naive log-sum-exp (can overflow)
let result = values.iter().map(|x| x.exp()).sum::<f64>().ln();

// Stable log-sum-exp
fn log_sum_exp(values: &[f64]) -> f64 {
    let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max_val.is_infinite() {
        return max_val;
    }
    max_val + values.iter().map(|x| (x - max_val).exp()).sum::<f64>().ln()
}
```

### Check for NaN/Inf

```rust
fn validate_result(x: f64) -> Result<f64> {
    if x.is_nan() {
        return Err(RustyStatsError::NumericalError("Result is NaN".into()));
    }
    if x.is_infinite() {
        return Err(RustyStatsError::NumericalError("Result is infinite".into()));
    }
    Ok(x)
}
```

---

## Testing

### Unit Tests in Same Module

```rust
// In families/poisson.rs
pub struct PoissonFamily;

impl Family for PoissonFamily { ... }

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_variance() {
        let family = PoissonFamily;
        let mu = array![1.0, 2.0, 5.0];
        let var = family.variance(&mu);
        assert_eq!(var, mu);
    }
}
```

### Use approx for Float Comparisons

```rust
use approx::assert_relative_eq;

#[test]
fn test_computation() {
    let result = compute();
    assert_relative_eq!(result, expected, epsilon = 1e-10);
}
```

### Test Edge Cases

```rust
#[test]
fn test_empty_input() {
    let empty = Array1::<f64>::zeros(0);
    let result = process(&empty);
    assert!(result.is_err());
}

#[test]
fn test_single_observation() {
    let single = array![1.0];
    let result = process(&single);
    assert!(result.is_ok());
}

#[test]
fn test_extreme_values() {
    let extreme = array![1e-300, 1e300];
    let result = process(&extreme);
    // Check numerical stability
    assert!(!result.unwrap().iter().any(|x| x.is_nan()));
}
```

---

## Documentation

### Document Public Items

```rust
/// Compute the variance function V(μ) for the Poisson family.
/// 
/// For Poisson, V(μ) = μ (variance equals mean).
/// 
/// # Arguments
/// * `mu` - Array of mean values, must be positive
/// 
/// # Returns
/// Array of variance values
/// 
/// # Example
/// ```
/// use rustystats_core::families::PoissonFamily;
/// use ndarray::array;
/// 
/// let family = PoissonFamily;
/// let mu = array![1.0, 2.0, 3.0];
/// let var = family.variance(&mu);
/// assert_eq!(var, mu);
/// ```
pub fn variance(&self, mu: &Array1<f64>) -> Array1<f64> {
    mu.clone()
}
```

### Use Module-Level Documentation

```rust
//! # Distribution Families
//!
//! This module implements distribution families for GLMs.
//!
//! ## Available Families
//! - [`GaussianFamily`] - For continuous data
//! - [`PoissonFamily`] - For count data
//! - [`BinomialFamily`] - For binary data
//!
//! ## Example
//! ```
//! use rustystats_core::families::{Family, PoissonFamily};
//! ```

pub struct PoissonFamily;
```

---

## Common Patterns

### Builder Pattern for Complex Types

```rust
pub struct IRLSConfig {
    pub max_iterations: usize,
    pub tolerance: f64,
    pub verbose: bool,
}

impl IRLSConfig {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }
    
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }
}

// Usage
let config = IRLSConfig::new()
    .max_iterations(50)
    .tolerance(1e-6);
```

### Newtype Pattern for Type Safety

```rust
pub struct Deviance(f64);
pub struct LogLikelihood(f64);

impl Deviance {
    pub fn value(&self) -> f64 { self.0 }
}

// Can't accidentally confuse deviance with log-likelihood
fn compute_aic(ll: LogLikelihood, p: usize) -> f64 {
    -2.0 * ll.0 + 2.0 * p as f64
}
```

### Iterator Chains

```rust
// Process with iterator chain
let result: Vec<_> = data.iter()
    .filter(|&x| *x > 0.0)
    .map(|x| x.ln())
    .collect();

// ndarray version
let result = data.mapv(|x| if x > 0.0 { x.ln() } else { 0.0 });
```
