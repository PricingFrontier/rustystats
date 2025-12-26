# Adding a New Family

This guide walks through adding a new distribution family to RustyStats.

## Overview

Adding a new family requires:
1. Implement the `Family` trait in Rust
2. Add Python bindings via PyO3
3. Register in the Python package
4. Add tests
5. Update documentation

We'll use **Inverse Gaussian** as an example.

---

## Step 1: Create the Rust Implementation

### Create the File

```bash
touch crates/rustystats-core/src/families/inverse_gaussian.rs
```

### Implement the Family Trait

```rust
// crates/rustystats-core/src/families/inverse_gaussian.rs

use ndarray::Array1;
use crate::links::{Link, LogLink};

/// Inverse Gaussian family for positive continuous data.
///
/// The Inverse Gaussian distribution is useful for modeling
/// positive continuous data with positive skew.
///
/// Variance function: V(μ) = μ³
pub struct InverseGaussianFamily;

impl super::Family for InverseGaussianFamily {
    fn name(&self) -> &str {
        "InverseGaussian"
    }
    
    fn variance(&self, mu: &Array1<f64>) -> Array1<f64> {
        // V(μ) = μ³
        mu.mapv(|m| m.powi(3))
    }
    
    fn unit_deviance(&self, y: &Array1<f64>, mu: &Array1<f64>) -> Array1<f64> {
        // d(y, μ) = (y - μ)² / (y × μ²)
        use ndarray::Zip;
        
        Zip::from(y).and(mu).map_collect(|&yi, &mui| {
            let yi_safe = yi.max(1e-10);
            let mui_safe = mui.max(1e-10);
            (yi_safe - mui_safe).powi(2) / (yi_safe * mui_safe.powi(2))
        })
    }
    
    fn default_link(&self) -> Box<dyn Link> {
        // Log link is common for positive data
        Box::new(LogLink)
    }
    
    fn initialize_mu(&self, y: &Array1<f64>) -> Array1<f64> {
        // Start at observed values, clamped to positive
        y.mapv(|yi| yi.max(0.1))
    }
    
    fn is_valid_mu(&self, mu: &Array1<f64>) -> bool {
        mu.iter().all(|&m| m > 0.0)
    }
}
```

### Add Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_name() {
        let family = InverseGaussianFamily;
        assert_eq!(family.name(), "InverseGaussian");
    }
    
    #[test]
    fn test_variance() {
        let family = InverseGaussianFamily;
        let mu = array![1.0, 2.0, 3.0];
        let var = family.variance(&mu);
        
        assert_relative_eq!(var[0], 1.0, epsilon = 1e-10);  // 1³
        assert_relative_eq!(var[1], 8.0, epsilon = 1e-10);  // 2³
        assert_relative_eq!(var[2], 27.0, epsilon = 1e-10); // 3³
    }
    
    #[test]
    fn test_deviance_perfect_fit() {
        let family = InverseGaussianFamily;
        let y = array![1.0, 2.0, 3.0];
        let mu = array![1.0, 2.0, 3.0];
        let dev = family.unit_deviance(&y, &mu);
        
        // Perfect fit: deviance should be 0
        for d in dev.iter() {
            assert_relative_eq!(*d, 0.0, epsilon = 1e-10);
        }
    }
    
    #[test]
    fn test_valid_mu() {
        let family = InverseGaussianFamily;
        
        assert!(family.is_valid_mu(&array![0.1, 1.0, 10.0]));
        assert!(!family.is_valid_mu(&array![0.0, 1.0, 2.0]));
        assert!(!family.is_valid_mu(&array![-1.0, 1.0, 2.0]));
    }
}
```

### Export from Module

```rust
// crates/rustystats-core/src/families/mod.rs

mod inverse_gaussian;
pub use inverse_gaussian::InverseGaussianFamily;
```

### Run Rust Tests

```bash
cargo test -p rustystats-core inverse_gaussian
```

---

## Step 2: Add Python Bindings

### Add the Wrapper Class

```rust
// crates/rustystats/src/lib.rs

use rustystats_core::families::InverseGaussianFamily;

/// Inverse Gaussian family for positive continuous data.
///
/// Variance function: V(μ) = μ³
/// Use for positive, right-skewed continuous data.
#[pyclass(name = "InverseGaussianFamily")]
#[derive(Clone)]
pub struct PyInverseGaussianFamily {
    inner: InverseGaussianFamily,
}

#[pymethods]
impl PyInverseGaussianFamily {
    #[new]
    fn new() -> Self {
        Self { inner: InverseGaussianFamily }
    }
    
    fn name(&self) -> &str {
        self.inner.name()
    }
    
    fn variance<'py>(&self, py: Python<'py>, mu: PyReadonlyArray1<f64>) -> Bound<'py, PyArray1<f64>> {
        let mu_array = mu.as_array().to_owned();
        let result = self.inner.variance(&mu_array);
        result.into_pyarray_bound(py)
    }
    
    fn unit_deviance<'py>(
        &self,
        py: Python<'py>,
        y: PyReadonlyArray1<f64>,
        mu: PyReadonlyArray1<f64>,
    ) -> Bound<'py, PyArray1<f64>> {
        let y_array = y.as_array().to_owned();
        let mu_array = mu.as_array().to_owned();
        let result = self.inner.unit_deviance(&y_array, &mu_array);
        result.into_pyarray_bound(py)
    }
    
    fn deviance(&self, y: PyReadonlyArray1<f64>, mu: PyReadonlyArray1<f64>) -> f64 {
        let y_array = y.as_array().to_owned();
        let mu_array = mu.as_array().to_owned();
        self.inner.deviance(&y_array, &mu_array, None)
    }
    
    fn default_link(&self) -> PyLogLink {
        PyLogLink::new()
    }
}
```

### Register in Module

```rust
// In the #[pymodule] function
#[pymodule]
fn _rustystats(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // ... existing registrations
    m.add_class::<PyInverseGaussianFamily>()?;
    Ok(())
}
```

### Add to fit_glm Family Matching

```rust
// In the fit_glm function
let (family_obj, link_obj): (Box<dyn Family>, Box<dyn Link>) = match family {
    "gaussian" => (Box::new(GaussianFamily), Box::new(IdentityLink)),
    "poisson" => (Box::new(PoissonFamily), Box::new(LogLink)),
    // ... existing families
    "inversegaussian" | "inverse_gaussian" => {
        (Box::new(InverseGaussianFamily), Box::new(LogLink))
    }
    _ => return Err(PyValueError::new_err(format!("Unknown family: {}", family))),
};
```

---

## Step 3: Export from Python Package

```python
# python/rustystats/__init__.py

from rustystats._rustystats import (
    # ... existing imports
    InverseGaussianFamily,
)

# Add to families module
from rustystats import families
families.InverseGaussian = InverseGaussianFamily
```

```python
# python/rustystats/families.py

from rustystats._rustystats import (
    # ... existing imports
    InverseGaussianFamily as InverseGaussian,
)
```

---

## Step 4: Add Python Tests

```python
# tests/python/test_inverse_gaussian.py

import pytest
import numpy as np
import rustystats as rs

class TestInverseGaussianFamily:
    """Tests for Inverse Gaussian family."""
    
    def test_family_exists(self):
        """Test that family can be instantiated."""
        family = rs.families.InverseGaussian()
        assert family.name() == "InverseGaussian"
    
    def test_variance_function(self):
        """Test V(μ) = μ³."""
        family = rs.families.InverseGaussian()
        mu = np.array([1.0, 2.0, 3.0])
        var = family.variance(mu)
        
        expected = np.array([1.0, 8.0, 27.0])
        np.testing.assert_allclose(var, expected)
    
    def test_fit_glm(self):
        """Test fitting with inverse gaussian family."""
        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        mu = np.exp(1.0 + 0.5 * x)
        y = np.random.wald(mu, 1.0)  # Inverse Gaussian samples
        
        X = np.column_stack([np.ones(n), x])
        result = rs.fit_glm(y, X, family="inversegaussian")
        
        assert result.converged
        assert len(result.params) == 2
    
    def test_predictions_positive(self):
        """Test that predictions are always positive."""
        np.random.seed(42)
        n = 50
        y = np.abs(np.random.randn(n)) + 0.1
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        
        result = rs.fit_glm(y, X, family="inversegaussian")
        
        assert np.all(result.fittedvalues > 0)
```

### Run Python Tests

```bash
uv run pytest tests/python/test_inverse_gaussian.py -v
```

---

## Step 5: Update Documentation

### Add to Families Theory Page

```markdown
<!-- docs/theory/families.md -->

## Inverse Gaussian Family

**Use for**: Positive continuous data with right skew.

### Properties

| Property | Value |
|----------|-------|
| Variance | V(μ) = μ³ |
| Canonical link | 1/μ² |
| Common link | Log |
| Valid μ range | (0, +∞) |

### Unit Deviance

$$
d(y, \mu) = \frac{(y - \mu)^2}{y \mu^2}
$$
```

### Add to Quick Start

```markdown
<!-- docs/getting-started/quickstart.md -->

| Data Type | Family | Example |
|-----------|--------|---------|
| ... | ... | ... |
| Positive right-skewed | `"inversegaussian"` | Waiting times, durations |
```

---

## Step 6: Rebuild and Test

```bash
# Rebuild
maturin develop

# Run all tests
cargo test -p rustystats-core
uv run pytest tests/python/ -v

# Verify docs build
mkdocs build
```

---

## Checklist

- [ ] Rust implementation in `families/`
- [ ] Unit tests for variance, deviance, initialization
- [ ] Export from `families/mod.rs`
- [ ] PyO3 wrapper class
- [ ] Register in `#[pymodule]`
- [ ] Add to `fit_glm` family matching
- [ ] Export from Python package
- [ ] Python tests
- [ ] Documentation updates
- [ ] Full test suite passes
