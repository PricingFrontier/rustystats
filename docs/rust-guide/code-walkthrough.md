# RustyStats Code Walkthrough

This guide walks through the actual RustyStats source code, explaining every Rust syntax element. Each section takes a real function from the codebase and annotates it line-by-line.

**Goal**: After reading this, you should be able to read and modify any function in the codebase.

---

## Part 1: Understanding Rust Syntax Basics

Before diving into the code, here's a quick reference for Rust syntax you'll encounter:

| Syntax | Meaning | Example |
|--------|---------|---------|
| `fn name() -> T` | Function returning type `T` | `fn variance() -> Array1<f64>` |
| `&` | Immutable reference (borrow) | `&self`, `&Array1<f64>` |
| `&mut` | Mutable reference | `&mut self` |
| `pub` | Public (visible outside module) | `pub fn fit_glm()` |
| `let` | Variable binding (immutable) | `let x = 5;` |
| `let mut` | Mutable variable | `let mut count = 0;` |
| `::` | Path separator | `Array1::zeros(n)` |
| `<T>` | Generic type parameter | `Vec<f64>` |
| `impl` | Implementation block | `impl Family for PoissonFamily` |
| `self` | Current instance | `self.name()` |
| `?` | Propagate error if Result is Err | `result?` |
| `\|x\| expr` | Closure (anonymous function) | `\|x\| x * 2` |
| `.iter()` | Create iterator | `vec.iter()` |
| `.map()` | Transform each element | `.map(\|x\| x + 1)` |
| `.collect()` | Gather iterator into collection | `.collect::<Vec<_>>()` |

---

## Part 2: The Family Trait

**File**: `crates/rustystats-core/src/families/mod.rs`

A **trait** defines shared behavior. Every distribution family implements this trait.

```rust
/// The Family trait defines the interface for all distribution families.
pub trait Family: Send + Sync {
```

**Line-by-line breakdown**:

| Code | Explanation |
|------|-------------|
| `///` | Documentation comment (appears in generated docs) |
| `pub` | This trait is public (can be used outside this module) |
| `trait Family` | Declares a trait named `Family` |
| `: Send + Sync` | **Trait bounds**: Types implementing `Family` must also implement `Send` and `Sync` (required for thread safety with Rayon) |
| `{` | Start of trait definition |

### Trait Methods

```rust
    /// Returns the name of this family.
    fn name(&self) -> &str;
```

| Code | Explanation |
|------|-------------|
| `fn name` | Method named `name` |
| `(&self)` | Takes an immutable reference to self (the instance) |
| `-> &str` | Returns a string slice (borrowed string) |
| `;` | No body—this is a **required method** that implementors must define |

```rust
    /// Compute the variance function V(μ).
    fn variance(&self, mu: &Array1<f64>) -> Array1<f64>;
```

| Code | Explanation |
|------|-------------|
| `mu: &Array1<f64>` | Parameter `mu` is a reference to a 1D array of `f64` (64-bit floats) |
| `-> Array1<f64>` | Returns an owned array (not a reference) |

```rust
    /// Compute the total deviance with optional weights.
    fn deviance(&self, y: &Array1<f64>, mu: &Array1<f64>, weights: Option<&Array1<f64>>) -> f64 {
        let unit_dev = self.unit_deviance(y, mu);
        match weights {
            Some(w) => (&unit_dev * w).sum(),
            None => unit_dev.sum(),
        }
    }
```

| Code | Explanation |
|------|-------------|
| `weights: Option<&Array1<f64>>` | `Option` type: either `Some(value)` or `None`. This makes the parameter optional. |
| `-> f64` | Returns a single float |
| `{ ... }` | Has a body—this is a **default implementation** (can be overridden) |
| `let unit_dev = ...` | Bind result to immutable variable |
| `self.unit_deviance(y, mu)` | Call another method on self |
| `match weights { ... }` | Pattern matching on the Option |
| `Some(w) =>` | If weights is Some, bind the inner value to `w` |
| `(&unit_dev * w).sum()` | Element-wise multiply arrays, then sum |
| `None =>` | If weights is None |
| `unit_dev.sum()` | Just sum without weighting |

---

## Part 3: Implementing a Family (PoissonFamily)

**File**: `crates/rustystats-core/src/families/poisson.rs`

### The Struct Definition

```rust
use ndarray::Array1;
use crate::links::{Link, LogLink};
use super::Family;

#[derive(Debug, Clone, Copy)]
pub struct PoissonFamily;
```

| Code | Explanation |
|------|-------------|
| `use ndarray::Array1` | Import `Array1` from the `ndarray` crate |
| `use crate::links::{Link, LogLink}` | Import from our own crate's `links` module |
| `use super::Family` | Import `Family` from parent module (`super` = parent) |
| `#[derive(...)]` | **Derive macro**: auto-generate trait implementations |
| `Debug` | Enables `{:?}` formatting for debugging |
| `Clone` | Enables `.clone()` method |
| `Copy` | Type can be copied implicitly (no move semantics) |
| `pub struct PoissonFamily;` | Empty struct (no fields)—a "unit struct" |

### The Implementation Block

```rust
impl Family for PoissonFamily {
    fn name(&self) -> &str {
        "Poisson"
    }
```

| Code | Explanation |
|------|-------------|
| `impl Family for PoissonFamily` | "Implement the Family trait for PoissonFamily" |
| `fn name(&self) -> &str` | Must match trait signature exactly |
| `"Poisson"` | String literal; last expression without `;` is the return value |

### The Variance Function

```rust
    fn variance(&self, mu: &Array1<f64>) -> Array1<f64> {
        mu.clone()
    }
```

| Code | Explanation |
|------|-------------|
| `mu.clone()` | Create a copy of the array. For Poisson, V(μ) = μ, so we return a copy of mu. |

### The Unit Deviance Function

```rust
    fn unit_deviance(&self, y: &Array1<f64>, mu: &Array1<f64>) -> Array1<f64> {
        ndarray::Zip::from(y)
            .and(mu)
            .map_collect(|&yi, &mui| {
                if yi == 0.0 {
                    2.0 * mui
                } else {
                    2.0 * (yi * (yi / mui).ln() - (yi - mui))
                }
            })
    }
```

| Code | Explanation |
|------|-------------|
| `ndarray::Zip::from(y)` | Create a parallel zipper starting with array `y` |
| `.and(mu)` | Add array `mu` to the zip |
| `.map_collect(\|&yi, &mui\| { ... })` | Apply closure to each pair, collect into new array |
| `\|&yi, &mui\|` | Closure parameters. `&` pattern-matches the reference to get the value. |
| `if yi == 0.0 { ... } else { ... }` | Conditional expression (returns a value) |
| `(yi / mui).ln()` | Divide, then natural log. Method chaining. |

### Returning a Boxed Trait Object

```rust
    fn default_link(&self) -> Box<dyn Link> {
        Box::new(LogLink)
    }
```

| Code | Explanation |
|------|-------------|
| `Box<dyn Link>` | A heap-allocated trait object. `dyn Link` means "some type implementing Link" |
| `Box::new(LogLink)` | Allocate `LogLink` on the heap and return a pointer |

**Why Box?** The trait method must return a consistent type, but different families return different link types. `Box<dyn Link>` erases the concrete type, allowing runtime polymorphism.

### Array Mapping

```rust
    fn initialize_mu(&self, y: &Array1<f64>) -> Array1<f64> {
        y.mapv(|yi| (yi + 0.1).max(0.1))
    }
```

| Code | Explanation |
|------|-------------|
| `y.mapv(\|yi\| ...)` | Map over values. `mapv` = map values (as opposed to `map` which maps references) |
| `(yi + 0.1).max(0.1)` | Add 0.1, then take max with 0.1. Ensures result ≥ 0.1 |

### Iterator with All

```rust
    fn is_valid_mu(&self, mu: &Array1<f64>) -> bool {
        mu.iter().all(|&x| x > 0.0 && x.is_finite())
    }
```

| Code | Explanation |
|------|-------------|
| `mu.iter()` | Create an iterator over the array |
| `.all(\|&x\| ...)` | Returns `true` if predicate is true for ALL elements |
| `x > 0.0 && x.is_finite()` | Check positive AND finite (not NaN or infinity) |

---

## Part 4: The Link Trait and LogLink

**File**: `crates/rustystats-core/src/links/log.rs`

```rust
use ndarray::Array1;
use super::Link;

#[derive(Debug, Clone, Copy)]
pub struct LogLink;

impl Link for LogLink {
    fn name(&self) -> &str {
        "log"
    }
    
    fn link(&self, mu: &Array1<f64>) -> Array1<f64> {
        mu.mapv(|x| x.ln())
    }
    
    fn inverse(&self, eta: &Array1<f64>) -> Array1<f64> {
        eta.mapv(|x| x.exp())
    }
    
    fn derivative(&self, mu: &Array1<f64>) -> Array1<f64> {
        mu.mapv(|x| 1.0 / x)
    }
}
```

| Code | Explanation |
|------|-------------|
| `x.ln()` | Natural logarithm (method on `f64`) |
| `x.exp()` | Exponential function e^x |
| `1.0 / x` | Division. `1.0` is a float literal. |

---

## Part 5: Error Handling

**File**: `crates/rustystats-core/src/error.rs`

### Defining Error Types

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum RustyStatsError {
    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),
    
    #[error("Invalid value: {0}")]
    InvalidValue(String),
    
    #[error("Convergence failed: {0}")]
    ConvergenceFailure(String),
}
```

| Code | Explanation |
|------|-------------|
| `use thiserror::Error` | Import the `Error` derive macro from `thiserror` crate |
| `#[derive(Error, Debug)]` | Auto-implement `std::error::Error` and `Debug` traits |
| `pub enum RustyStatsError` | Define an enum (sum type) for all error variants |
| `#[error("...")]` | Attribute macro: defines the error message |
| `{0}` | Placeholder for first field in the variant |
| `DimensionMismatch(String)` | Variant with one `String` field |

### The Result Type Alias

```rust
pub type Result<T> = std::result::Result<T, RustyStatsError>;
```

| Code | Explanation |
|------|-------------|
| `pub type Result<T>` | Define a public type alias with generic parameter `T` |
| `= std::result::Result<T, RustyStatsError>` | Alias for `Result` with our error type |

Now instead of writing `Result<IRLSResult, RustyStatsError>`, we write `Result<IRLSResult>`.

---

## Part 6: The IRLS Solver

**File**: `crates/rustystats-core/src/solvers/irls.rs`

### Configuration Struct

```rust
#[derive(Debug, Clone)]
pub struct IRLSConfig {
    pub max_iterations: usize,
    pub tolerance: f64,
    pub min_weight: f64,
    pub verbose: bool,
}
```

| Code | Explanation |
|------|-------------|
| `#[derive(Debug, Clone)]` | Auto-implement Debug and Clone |
| `pub struct IRLSConfig` | Public struct |
| `pub max_iterations: usize` | Public field of type `usize` (unsigned integer, pointer-sized) |
| `pub tolerance: f64` | 64-bit floating point |
| `pub verbose: bool` | Boolean |

### Default Implementation

```rust
impl Default for IRLSConfig {
    fn default() -> Self {
        Self {
            max_iterations: 25,
            tolerance: 1e-8,
            min_weight: 1e-10,
            verbose: false,
        }
    }
}
```

| Code | Explanation |
|------|-------------|
| `impl Default for IRLSConfig` | Implement the `Default` trait |
| `fn default() -> Self` | Returns an instance of Self (IRLSConfig) |
| `Self { ... }` | Struct literal. `Self` is an alias for the implementing type. |
| `1e-8` | Scientific notation: 1 × 10⁻⁸ |

### The Main Fitting Function

```rust
pub fn fit_glm(
    y: &Array1<f64>,
    x: &Array2<f64>,
    family: &dyn Family,
    link: &dyn Link,
    config: &IRLSConfig,
) -> Result<IRLSResult> {
    fit_glm_full(y, x, family, link, config, None, None)
}
```

| Code | Explanation |
|------|-------------|
| `pub fn fit_glm(...)` | Public function |
| `y: &Array1<f64>` | Borrow a 1D array |
| `x: &Array2<f64>` | Borrow a 2D array (matrix) |
| `family: &dyn Family` | Borrow a trait object (any type implementing Family) |
| `-> Result<IRLSResult>` | Returns `Ok(IRLSResult)` on success, `Err(...)` on failure |
| `None, None` | Pass `None` for optional parameters |

### Input Validation

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

    if x.nrows() != n {
        return Err(RustyStatsError::DimensionMismatch(format!(
            "X has {} rows but y has {} elements",
            x.nrows(),
            n
        )));
    }
```

| Code | Explanation |
|------|-------------|
| `offset: Option<&Array1<f64>>` | Optional borrowed array |
| `let n = y.len()` | Get array length |
| `let p = x.ncols()` | Get number of columns |
| `x.nrows()` | Get number of rows |
| `return Err(...)` | Early return with error |
| `format!("...", x.nrows(), n)` | String formatting macro (like Python f-strings) |

### Handling Optional Parameters

```rust
    let offset_vec = match offset {
        Some(o) => {
            if o.len() != n {
                return Err(RustyStatsError::DimensionMismatch(...));
            }
            o.clone()
        }
        None => Array1::zeros(n),
    };
```

| Code | Explanation |
|------|-------------|
| `match offset { ... }` | Pattern match on the Option |
| `Some(o) => { ... }` | If Some, bind inner value to `o`, execute block |
| `o.clone()` | Clone the borrowed array to get an owned copy |
| `None => Array1::zeros(n)` | If None, create zero array of length n |

### The IRLS Loop

```rust
    let mut converged = false;
    let mut iteration = 0;

    while iteration < config.max_iterations {
        iteration += 1;

        // Compute variance and link derivative
        let variance = family.variance(&mu);
        let link_deriv = link.derivative(&mu);
```

| Code | Explanation |
|------|-------------|
| `let mut converged = false` | Mutable boolean, initially false |
| `while condition { ... }` | While loop |
| `iteration += 1` | Increment (shorthand for `iteration = iteration + 1`) |
| `family.variance(&mu)` | Call trait method, passing borrow of mu |

### Parallel Iteration with Rayon

```rust
        let results: Vec<(f64, f64, f64)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let v = variance[i];
                let d = link_deriv[i];
                let iw = (1.0 / (v * d * d)).max(min_weight).min(1e10);
                let cw = prior_weights_vec[i] * iw;
                let wr = (eta[i] - offset_vec[i]) + (y[i] - mu[i]) * d;
                (iw, cw, wr)
            })
            .collect();
```

| Code | Explanation |
|------|-------------|
| `Vec<(f64, f64, f64)>` | Vector of tuples, each containing 3 floats |
| `(0..n)` | Range from 0 to n-1 |
| `.into_par_iter()` | Convert to **parallel iterator** (Rayon) |
| `.map(\|i\| { ... })` | Transform each index in parallel |
| `variance[i]` | Index into array |
| `(1.0 / (v * d * d))` | Arithmetic expression |
| `.max(min_weight)` | Take maximum of value and min_weight |
| `.min(1e10)` | Take minimum (clamp to upper bound) |
| `(iw, cw, wr)` | Return a tuple |
| `.collect()` | Gather parallel results into Vec |

### Building Arrays from Vectors

```rust
        let mut irls_weights_vec = Vec::with_capacity(n);
        let mut combined_weights_vec = Vec::with_capacity(n);
        let mut working_response_vec = Vec::with_capacity(n);
        
        for (iw, cw, wr) in results {
            irls_weights_vec.push(iw);
            combined_weights_vec.push(cw);
            working_response_vec.push(wr);
        }
        
        let irls_weights = Array1::from_vec(irls_weights_vec);
```

| Code | Explanation |
|------|-------------|
| `Vec::with_capacity(n)` | Create Vec with pre-allocated capacity (performance) |
| `for (iw, cw, wr) in results` | Destructure tuple in for loop |
| `.push(iw)` | Append to vector |
| `Array1::from_vec(...)` | Convert Vec to ndarray Array1 |

### Matrix Operations

```rust
        let eta_base = x.dot(&new_coefficients);
        eta = &eta_base + &offset_vec;
        mu = link.inverse(&eta);
```

| Code | Explanation |
|------|-------------|
| `x.dot(&new_coefficients)` | Matrix-vector multiplication |
| `&eta_base + &offset_vec` | Element-wise addition of array references |
| `link.inverse(&eta)` | Apply inverse link function |

### Convergence Check

```rust
        let rel_change = if deviance_old.abs() > 1e-10 {
            (deviance_old - deviance).abs() / deviance_old.abs()
        } else {
            (deviance_old - deviance).abs()
        };

        if rel_change < config.tolerance {
            converged = true;
            break;
        }
```

| Code | Explanation |
|------|-------------|
| `if ... { ... } else { ... }` | If-else expression (returns value) |
| `.abs()` | Absolute value |
| `break` | Exit the loop |

### Returning the Result

```rust
    Ok(IRLSResult {
        coefficients: final_coefficients,
        fitted_values: mu,
        linear_predictor: eta,
        deviance,
        iterations: iteration,
        converged,
        covariance_unscaled: cov_unscaled,
        irls_weights: final_weights,
        prior_weights: prior_weights_vec,
        offset: offset_vec,
        y: y.to_owned(),
        family_name: family.name().to_string(),
        penalty: Penalty::None,
        design_matrix: None,
    })
}
```

| Code | Explanation |
|------|-------------|
| `Ok(IRLSResult { ... })` | Return success with struct |
| `coefficients: final_coefficients` | Field name: value |
| `deviance,` | Shorthand when field name equals variable name |
| `y.to_owned()` | Convert borrowed slice to owned data |
| `.to_string()` | Convert &str to owned String |
| `Penalty::None` | Enum variant |
| `None` | Option::None |

---

## Part 7: Tests

**File**: `crates/rustystats-core/src/families/poisson.rs`

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use approx::assert_abs_diff_eq;
    
    #[test]
    fn test_poisson_variance() {
        let family = PoissonFamily;
        let mu = array![0.5, 1.0, 2.0, 10.0];
        
        let var = family.variance(&mu);
        assert_abs_diff_eq!(var, mu, epsilon = 1e-10);
    }
```

| Code | Explanation |
|------|-------------|
| `#[cfg(test)]` | Conditional compilation: only compile when testing |
| `mod tests` | Nested module for tests |
| `use super::*` | Import everything from parent module |
| `use ndarray::array` | Import the `array!` macro |
| `#[test]` | Mark function as a test |
| `fn test_poisson_variance()` | Test function (no parameters, no return) |
| `array![0.5, 1.0, 2.0, 10.0]` | Macro to create array literal |
| `assert_abs_diff_eq!(var, mu, epsilon = 1e-10)` | Assert arrays are approximately equal |

---

## Part 8: Coordinate Descent Specifics

**File**: `crates/rustystats-core/src/solvers/coordinate_descent.rs`

### Soft Thresholding

```rust
pub fn soft_threshold(z: f64, gamma: f64) -> f64 {
    if z > gamma {
        z - gamma
    } else if z < -gamma {
        z + gamma
    } else {
        0.0
    }
}
```

| Code | Explanation |
|------|-------------|
| `pub fn soft_threshold(z: f64, gamma: f64) -> f64` | Public function taking two floats, returning float |
| `if ... else if ... else` | Chained conditionals |
| `0.0` | Float literal (no semicolon = return value) |

### Parallel Fold-Reduce for Gram Matrix

```rust
let xwx: Vec<f64> = (0..n)
    .into_par_iter()
    .fold(
        || vec![0.0; p * p],
        |mut acc, i| {
            let w_i = combined_weights[i];
            let x_i = x.row(i);
            for j in 0..p {
                let xij_w = x_i[j] * w_i;
                for k in j..p {
                    acc[j * p + k] += xij_w * x_i[k];
                }
            }
            acc
        },
    )
    .reduce(
        || vec![0.0; p * p],
        |mut a, b| {
            for i in 0..a.len() {
                a[i] += b[i];
            }
            a
        },
    );
```

| Code | Explanation |
|------|-------------|
| `.fold(init, f)` | Parallel fold with initial value factory and accumulator function |
| `\|\| vec![0.0; p * p]` | Closure returning new zero vector (called per thread) |
| `vec![0.0; p * p]` | Vec macro: create vector of `p*p` zeros |
| `\|mut acc, i\|` | Closure taking mutable accumulator and index |
| `x.row(i)` | Get row i of matrix as array view |
| `acc[j * p + k] += ...` | Accumulate into flattened 2D index |
| `.reduce(init, combine)` | Combine thread-local results |
| `\|mut a, b\|` | Combine two accumulators |

---

## Part 9: Common Patterns Summary

### Pattern: Method Chaining

```rust
mu.iter().all(|&x| x > 0.0 && x.is_finite())
```

Read left-to-right: take `mu`, create iterator, check if all elements satisfy condition.

### Pattern: Error Propagation with `?`

```rust
let result = some_fallible_operation()?;
```

If `some_fallible_operation()` returns `Err`, immediately return that error. Otherwise, unwrap the `Ok` value.

### Pattern: Builder-style APIs

```rust
let config = IRLSConfig {
    max_iterations: 50,
    ..Default::default()  // Fill remaining fields with defaults
};
```

### Pattern: Iterating and Collecting

```rust
let squares: Vec<f64> = (0..10).map(|x| (x * x) as f64).collect();
```

### Pattern: Parallel Processing with Rayon

```rust
use rayon::prelude::*;

let results: Vec<_> = data.par_iter().map(|x| expensive_operation(x)).collect();
```

Just change `.iter()` to `.par_iter()` for parallelism.

---

## Quick Reference: Common Methods

| Method | On Type | Returns | Description |
|--------|---------|---------|-------------|
| `.len()` | Array/Vec | `usize` | Number of elements |
| `.nrows()` | Array2 | `usize` | Number of rows |
| `.ncols()` | Array2 | `usize` | Number of columns |
| `.clone()` | Any Clone | Owned copy | Deep copy |
| `.iter()` | Collection | Iterator | Iterate by reference |
| `.into_iter()` | Collection | Iterator | Iterate by value (consumes) |
| `.map(f)` | Iterator | Iterator | Transform elements |
| `.filter(f)` | Iterator | Iterator | Keep matching elements |
| `.collect()` | Iterator | Collection | Gather into collection |
| `.sum()` | Iterator | Number | Sum all elements |
| `.all(f)` | Iterator | bool | True if all match |
| `.any(f)` | Iterator | bool | True if any match |
| `.zip(other)` | Iterator | Iterator | Pair with another iterator |
| `.mapv(f)` | ndarray | Array | Map values |
| `.dot(&other)` | Array | Array/scalar | Matrix multiplication |
| `.abs()` | Number | Number | Absolute value |
| `.ln()` | f64 | f64 | Natural log |
| `.exp()` | f64 | f64 | Exponential |
| `.max(other)` | f64 | f64 | Maximum |
| `.min(other)` | f64 | f64 | Minimum |
| `.is_finite()` | f64 | bool | Not NaN or infinity |

---

## Next Steps

- [Rust Fundamentals](fundamentals.md) — Ownership, borrowing, and core concepts
- [Adding a New Family](../maintenance/adding-family.md) — Practice implementing a trait
- [Adding a New Link](../maintenance/adding-link.md) — Another implementation exercise
