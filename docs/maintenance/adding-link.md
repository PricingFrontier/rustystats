# Adding a New Link Function

This guide walks through adding a new link function to RustyStats.

## Overview

Adding a new link requires:
1. Implement the `Link` trait in Rust
2. Add Python bindings via PyO3
3. Register in the Python package
4. Add tests
5. Update documentation

We'll use **Probit** (inverse normal CDF) as an example.

---

## Step 1: Create the Rust Implementation

### Create the File

```bash
touch crates/rustystats-core/src/links/probit.rs
```

### Implement the Link Trait

```rust
// crates/rustystats-core/src/links/probit.rs

use ndarray::Array1;
use statrs::distribution::{ContinuousCDF, Normal};

/// Probit link function: η = Φ⁻¹(μ)
///
/// Uses the inverse of the standard normal CDF.
/// Alternative to logit for binary data.
pub struct ProbitLink {
    normal: Normal,
}

impl ProbitLink {
    pub fn new() -> Self {
        Self {
            normal: Normal::new(0.0, 1.0).unwrap(),
        }
    }
}

impl Default for ProbitLink {
    fn default() -> Self {
        Self::new()
    }
}

impl super::Link for ProbitLink {
    fn name(&self) -> &str {
        "Probit"
    }
    
    fn link(&self, mu: &Array1<f64>) -> Array1<f64> {
        // η = Φ⁻¹(μ)
        mu.mapv(|m| {
            // Clamp to avoid infinite values
            let m_safe = m.clamp(1e-10, 1.0 - 1e-10);
            self.normal.inverse_cdf(m_safe)
        })
    }
    
    fn inverse(&self, eta: &Array1<f64>) -> Array1<f64> {
        // μ = Φ(η)
        eta.mapv(|e| self.normal.cdf(e))
    }
    
    fn derivative(&self, mu: &Array1<f64>) -> Array1<f64> {
        // dη/dμ = 1/φ(Φ⁻¹(μ))
        // where φ is the standard normal PDF
        use statrs::distribution::Continuous;
        
        mu.mapv(|m| {
            let m_safe = m.clamp(1e-10, 1.0 - 1e-10);
            let z = self.normal.inverse_cdf(m_safe);
            let pdf = self.normal.pdf(z);
            
            if pdf > 1e-10 {
                1.0 / pdf
            } else {
                1e10  // Cap at large value
            }
        })
    }
}

// Need Clone for PyO3
impl Clone for ProbitLink {
    fn clone(&self) -> Self {
        Self::new()
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
        let link = ProbitLink::new();
        assert_eq!(link.name(), "Probit");
    }
    
    #[test]
    fn test_link_at_half() {
        let link = ProbitLink::new();
        let mu = array![0.5];
        let eta = link.link(&mu);
        
        // Φ⁻¹(0.5) = 0
        assert_relative_eq!(eta[0], 0.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_inverse_at_zero() {
        let link = ProbitLink::new();
        let eta = array![0.0];
        let mu = link.inverse(&eta);
        
        // Φ(0) = 0.5
        assert_relative_eq!(mu[0], 0.5, epsilon = 1e-10);
    }
    
    #[test]
    fn test_roundtrip() {
        let link = ProbitLink::new();
        let mu_original = array![0.1, 0.3, 0.5, 0.7, 0.9];
        
        let eta = link.link(&mu_original);
        let mu_recovered = link.inverse(&eta);
        
        for i in 0..mu_original.len() {
            assert_relative_eq!(mu_original[i], mu_recovered[i], epsilon = 1e-8);
        }
    }
    
    #[test]
    fn test_bounds() {
        let link = ProbitLink::new();
        
        // Very negative eta should give mu near 0
        let eta_neg = array![-5.0];
        let mu = link.inverse(&eta_neg);
        assert!(mu[0] < 0.001);
        assert!(mu[0] > 0.0);
        
        // Very positive eta should give mu near 1
        let eta_pos = array![5.0];
        let mu = link.inverse(&eta_pos);
        assert!(mu[0] > 0.999);
        assert!(mu[0] < 1.0);
    }
    
    #[test]
    fn test_derivative_positive() {
        let link = ProbitLink::new();
        let mu = array![0.1, 0.5, 0.9];
        let deriv = link.derivative(&mu);
        
        // Derivative should always be positive
        for d in deriv.iter() {
            assert!(*d > 0.0);
        }
    }
}
```

### Export from Module

```rust
// crates/rustystats-core/src/links/mod.rs

mod probit;
pub use probit::ProbitLink;
```

### Run Rust Tests

```bash
cargo test -p rustystats-core probit
```

---

## Step 2: Add Python Bindings

### Add the Wrapper Class

```rust
// crates/rustystats/src/lib.rs

use rustystats_core::links::ProbitLink;

/// Probit link function: η = Φ⁻¹(μ)
///
/// Uses the inverse standard normal CDF.
/// Alternative to logit for Binomial family.
#[pyclass(name = "ProbitLink")]
#[derive(Clone)]
pub struct PyProbitLink {
    inner: ProbitLink,
}

#[pymethods]
impl PyProbitLink {
    #[new]
    fn new() -> Self {
        Self { inner: ProbitLink::new() }
    }
    
    fn name(&self) -> &str {
        self.inner.name()
    }
    
    fn link<'py>(&self, py: Python<'py>, mu: PyReadonlyArray1<f64>) -> Bound<'py, PyArray1<f64>> {
        let mu_array = mu.as_array().to_owned();
        let result = self.inner.link(&mu_array);
        result.into_pyarray_bound(py)
    }
    
    fn inverse<'py>(&self, py: Python<'py>, eta: PyReadonlyArray1<f64>) -> Bound<'py, PyArray1<f64>> {
        let eta_array = eta.as_array().to_owned();
        let result = self.inner.inverse(&eta_array);
        result.into_pyarray_bound(py)
    }
    
    fn derivative<'py>(&self, py: Python<'py>, mu: PyReadonlyArray1<f64>) -> Bound<'py, PyArray1<f64>> {
        let mu_array = mu.as_array().to_owned();
        let result = self.inner.derivative(&mu_array);
        result.into_pyarray_bound(py)
    }
}
```

### Register in Module

```rust
// In the #[pymodule] function
#[pymodule]
fn _rustystats(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // ... existing registrations
    m.add_class::<PyProbitLink>()?;
    Ok(())
}
```

### Add to fit_glm Link Resolution

```rust
// Helper function to get link from string
fn get_link(link: Option<&str>, family: &str) -> Box<dyn Link> {
    match link {
        Some("identity") => Box::new(IdentityLink),
        Some("log") => Box::new(LogLink),
        Some("logit") => Box::new(LogitLink),
        Some("probit") => Box::new(ProbitLink::new()),
        None => {
            // Use family default
            match family {
                "gaussian" => Box::new(IdentityLink),
                "poisson" | "gamma" => Box::new(LogLink),
                "binomial" => Box::new(LogitLink),
                _ => Box::new(IdentityLink),
            }
        }
        Some(other) => panic!("Unknown link: {}", other),
    }
}
```

---

## Step 3: Export from Python Package

```python
# python/rustystats/__init__.py

from rustystats._rustystats import (
    # ... existing imports
    ProbitLink,
)
```

```python
# python/rustystats/links.py

from rustystats._rustystats import (
    IdentityLink as Identity,
    LogLink as Log,
    LogitLink as Logit,
    ProbitLink as Probit,
)
```

---

## Step 4: Add Python Tests

```python
# tests/python/test_probit.py

import pytest
import numpy as np
import rustystats as rs
from scipy import stats

class TestProbitLink:
    """Tests for Probit link function."""
    
    def test_link_exists(self):
        """Test that link can be instantiated."""
        link = rs.ProbitLink()
        assert link.name() == "Probit"
    
    def test_link_at_half(self):
        """Test Φ⁻¹(0.5) = 0."""
        link = rs.ProbitLink()
        mu = np.array([0.5])
        eta = link.link(mu)
        np.testing.assert_allclose(eta, [0.0], atol=1e-10)
    
    def test_inverse_at_zero(self):
        """Test Φ(0) = 0.5."""
        link = rs.ProbitLink()
        eta = np.array([0.0])
        mu = link.inverse(eta)
        np.testing.assert_allclose(mu, [0.5], atol=1e-10)
    
    def test_vs_scipy(self):
        """Compare to scipy implementation."""
        link = rs.ProbitLink()
        mu = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        
        eta_rustystats = link.link(mu)
        eta_scipy = stats.norm.ppf(mu)
        
        np.testing.assert_allclose(eta_rustystats, eta_scipy, rtol=1e-6)
    
    def test_roundtrip(self):
        """Test link → inverse roundtrip."""
        link = rs.ProbitLink()
        mu_original = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        
        eta = link.link(mu_original)
        mu_recovered = link.inverse(eta)
        
        np.testing.assert_allclose(mu_original, mu_recovered, rtol=1e-8)
    
    def test_fit_binomial_with_probit(self):
        """Test fitting binomial with probit link."""
        np.random.seed(42)
        n = 200
        x = np.random.randn(n)
        
        # Generate probit model data
        eta = 0.5 + 0.3 * x
        p = stats.norm.cdf(eta)
        y = np.random.binomial(1, p)
        
        X = np.column_stack([np.ones(n), x])
        result = rs.fit_glm(y, X, family="binomial", link="probit")
        
        assert result.converged
        # Coefficients should be close to true values
        np.testing.assert_allclose(result.params, [0.5, 0.3], atol=0.2)
```

### Run Python Tests

```bash
uv run pytest tests/python/test_probit.py -v
```

---

## Step 5: Update Documentation

### Add to Links Theory Page

```markdown
<!-- docs/theory/links.md -->

## Probit Link

**Formula**: η = Φ⁻¹(μ)

Uses the inverse of the standard normal CDF.

| Property | Formula |
|----------|---------|
| Link | g(μ) = Φ⁻¹(μ) |
| Inverse | g⁻¹(η) = Φ(η) |
| Derivative | g'(μ) = 1/φ(Φ⁻¹(μ)) |

### When to Use

- Alternative to logit for binomial data
- When underlying process involves normal distribution
- Dose-response models

### Comparison to Logit

Both map (0,1) to (-∞, +∞), but:
- Logit: heavier tails
- Probit: lighter tails, closer to 0/1 more quickly
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

- [ ] Rust implementation in `links/`
- [ ] Unit tests for link, inverse, derivative, roundtrip
- [ ] Export from `links/mod.rs`
- [ ] PyO3 wrapper class
- [ ] Register in `#[pymodule]`
- [ ] Add to link resolution in `fit_glm`
- [ ] Export from Python package
- [ ] Python tests (including comparison to scipy)
- [ ] Documentation updates
- [ ] Full test suite passes
