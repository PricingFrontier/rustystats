# Python Bindings

The `rustystats` crate provides the bridge between Python and the Rust core library using PyO3. This chapter explains how the bindings work and how to extend them.

## PyO3 Overview

[PyO3](https://pyo3.rs/) is a Rust crate that enables:
- Calling Rust code from Python
- Calling Python code from Rust
- Converting between Python and Rust types

### Key Concepts

| Concept | Description |
|---------|-------------|
| `#[pymodule]` | Defines a Python module |
| `#[pyclass]` | Makes a Rust struct a Python class |
| `#[pymethods]` | Exposes methods to Python |
| `#[pyfunction]` | Exposes a standalone function |
| `#[getter]` | Makes a method a property getter |

## Module Structure

All bindings are in a single file:

```
crates/rustystats/src/lib.rs
```

The file is organized into sections:
1. Link function wrappers
2. Family wrappers
3. GLMResults class
4. Standalone functions
5. Module registration

## The Module Entry Point

```rust
#[pymodule]
fn _rustystats(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register classes
    m.add_class::<PyIdentityLink>()?;
    m.add_class::<PyLogLink>()?;
    m.add_class::<PyLogitLink>()?;
    
    m.add_class::<PyGaussianFamily>()?;
    m.add_class::<PyPoissonFamily>()?;
    m.add_class::<PyBinomialFamily>()?;
    m.add_class::<PyGammaFamily>()?;
    m.add_class::<PyTweedieFamily>()?;
    m.add_class::<PyQuasiPoissonFamily>()?;
    m.add_class::<PyQuasiBinomialFamily>()?;
    m.add_class::<PyNegativeBinomialFamily>()?;
    
    m.add_class::<PyGLMResults>()?;
    
    // Register functions
    m.add_function(wrap_pyfunction!(fit_glm, m)?)?;
    m.add_function(wrap_pyfunction!(fit_glm_regularized, m)?)?;
    // ... more functions
    
    Ok(())
}
```

The module is named `_rustystats` (with underscore) and exposed as `rustystats._rustystats` in Python. The Python package imports from this and re-exports.

## Wrapping Rust Types

### Pattern: Inner Type Wrapper

Each Rust type is wrapped in a Python-facing struct:

```rust
// The Rust type (from rustystats-core)
use rustystats_core::families::PoissonFamily;

// The Python wrapper
#[pyclass(name = "PoissonFamily")]
#[derive(Clone)]
pub struct PyPoissonFamily {
    inner: PoissonFamily,  // Holds the actual Rust type
}

#[pymethods]
impl PyPoissonFamily {
    #[new]
    fn new() -> Self {
        Self { inner: PoissonFamily }
    }
    
    fn name(&self) -> &str {
        self.inner.name()
    }
    
    fn variance<'py>(
        &self,
        py: Python<'py>,
        mu: PyReadonlyArray1<f64>
    ) -> Bound<'py, PyArray1<f64>> {
        let mu_array = mu.as_array().to_owned();
        let result = self.inner.variance(&mu_array);
        result.into_pyarray_bound(py)
    }
    // ... more methods
}
```

### Why This Pattern?

1. **Separation**: Python concerns stay in the wrapper
2. **Type conversion**: Handle NumPy ↔ ndarray conversion
3. **Safety**: PyO3 requirements (Clone, etc.) don't affect core types

## NumPy Integration

### Reading Arrays from Python

```rust
use numpy::{PyReadonlyArray1, PyReadonlyArray2};

fn process(
    y: PyReadonlyArray1<f64>,  // Read-only 1D array
    x: PyReadonlyArray2<f64>,  // Read-only 2D array
) {
    // Get a view (no copy)
    let y_view = y.as_array();
    
    // Get an owned copy (when needed)
    let y_owned = y.as_array().to_owned();
}
```

### Returning Arrays to Python

```rust
use numpy::{IntoPyArray, PyArray1, PyArray2};

fn compute<'py>(py: Python<'py>, n: usize) -> Bound<'py, PyArray1<f64>> {
    let result: Array1<f64> = Array1::zeros(n);
    result.into_pyarray_bound(py)  // Transfer ownership to Python
}
```

### The Lifetime Parameter

The `'py` lifetime ties returned objects to the Python interpreter:

```rust
fn method<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
    // The returned PyArray1 lives as long as 'py (the Python session)
}
```

## The GLMResults Class

This is the most complex wrapper, providing all inference methods:

```rust
#[pyclass(name = "GLMResults")]
#[derive(Clone)]
pub struct PyGLMResults {
    coefficients: Array1<f64>,
    fitted_values: Array1<f64>,
    linear_predictor: Array1<f64>,
    deviance: f64,
    iterations: usize,
    converged: bool,
    covariance_unscaled: Array2<f64>,
    n_obs: usize,
    n_params: usize,
    y: Array1<f64>,
    family_name: String,
    prior_weights: Array1<f64>,
    penalty: Penalty,
    design_matrix: Array2<f64>,
    irls_weights: Array1<f64>,
}

#[pymethods]
impl PyGLMResults {
    // Properties (getters)
    #[getter]
    fn params<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.coefficients.clone().into_pyarray_bound(py)
    }
    
    #[getter]
    fn fittedvalues<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.fitted_values.clone().into_pyarray_bound(py)
    }
    
    #[getter]
    fn deviance(&self) -> f64 {
        self.deviance
    }
    
    // Methods
    fn bse<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let scale = self.scale();
        let se: Array1<f64> = (0..self.n_params)
            .map(|i| (scale * self.covariance_unscaled[[i, i]]).sqrt())
            .collect();
        se.into_pyarray_bound(py)
    }
    
    fn pvalues<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        // Compute z-statistics and convert to p-values
        // ...
    }
    
    // Residual methods call back to rustystats-core
    fn resid_response<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        use rustystats_core::diagnostics::resid_response;
        let resid = resid_response(&self.y, &self.fitted_values);
        resid.into_pyarray_bound(py)
    }
}
```

## Handling Optional Parameters

Use `#[pyo3(signature = ...)]` for optional arguments:

```rust
#[pyfunction]
#[pyo3(signature = (y, x, family="gaussian", alpha=0.0, l1_ratio=1.0, offset=None, weights=None))]
fn fit_glm<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<f64>,
    x: PyReadonlyArray2<f64>,
    family: &str,
    alpha: f64,
    l1_ratio: f64,
    offset: Option<PyReadonlyArray1<f64>>,
    weights: Option<PyReadonlyArray1<f64>>,
) -> PyResult<PyGLMResults> {
    // ...
}
```

In Python:
```python
result = rs._rustystats.fit_glm(y, X)  # Uses defaults
result = rs._rustystats.fit_glm(y, X, family="poisson", alpha=0.1)
```

## Error Handling

### Rust → Python Errors

Convert Rust errors to Python exceptions:

```rust
use pyo3::exceptions::PyValueError;

impl From<RustyStatsError> for PyErr {
    fn from(err: RustyStatsError) -> PyErr {
        match err {
            RustyStatsError::InvalidInput(msg) => 
                PyValueError::new_err(msg),
            RustyStatsError::DimensionMismatch { expected, got } =>
                PyValueError::new_err(
                    format!("Dimension mismatch: expected {}, got {}", expected, got)
                ),
            _ => PyValueError::new_err(format!("{}", err)),
        }
    }
}
```

### Using PyResult

Functions that can fail return `PyResult<T>`:

```rust
#[pyfunction]
fn fit_glm(...) -> PyResult<PyGLMResults> {
    // The ? operator converts RustyStatsError to PyErr
    let result = rustystats_core::fit_glm_full(...)?;
    Ok(wrap_result(result))
}
```

## Releasing the GIL

For long computations, release the GIL to allow other Python threads:

```rust
#[pyfunction]
fn fit_glm<'py>(py: Python<'py>, ...) -> PyResult<PyGLMResults> {
    // Convert inputs first (needs GIL)
    let y_owned = y.as_array().to_owned();
    let x_owned = x.as_array().to_owned();
    
    // Release GIL for computation
    let result = py.allow_threads(|| {
        rustystats_core::fit_glm_full(&y_owned, &x_owned, ...)
    })?;
    
    // Re-acquire GIL for output conversion
    Ok(wrap_result(result))
}
```

## The Python Layer

The compiled Rust module is imported by the Python package:

```python
# python/rustystats/__init__.py
from rustystats._rustystats import (
    IdentityLink, LogLink, LogitLink,
    GaussianFamily, PoissonFamily, BinomialFamily,
    # ...
)

from rustystats.glm import fit_glm, GLM, GLMResults
from rustystats.formula import glm
```

The Python layer adds:
- Higher-level APIs (formula parsing, DataFrame handling)
- Documentation and type hints
- Convenience functions

## Adding a New Binding

### Step 1: Implement in rustystats-core

```rust
// In rustystats-core/src/mymodule.rs
pub fn my_new_function(x: &Array1<f64>) -> Array1<f64> {
    // Implementation
}
```

### Step 2: Add PyO3 Wrapper

```rust
// In rustystats/src/lib.rs

#[pyfunction]
fn my_new_function<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray1<f64>> {
    let x_owned = x.as_array().to_owned();
    let result = rustystats_core::my_new_function(&x_owned);
    result.into_pyarray_bound(py)
}
```

### Step 3: Register in Module

```rust
#[pymodule]
fn _rustystats(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // ... existing registrations
    m.add_function(wrap_pyfunction!(my_new_function, m)?)?;
    Ok(())
}
```

### Step 4: Export from Python

```python
# python/rustystats/__init__.py
from rustystats._rustystats import my_new_function
```

### Step 5: Rebuild

```bash
maturin develop
```

## Testing Bindings

```python
# tests/python/test_bindings.py
def test_my_new_function():
    import rustystats as rs
    import numpy as np
    
    x = np.array([1.0, 2.0, 3.0])
    result = rs.my_new_function(x)
    
    assert result.shape == x.shape
    # More assertions...
```

## Common Patterns

### Constructor with Validation

```rust
#[pymethods]
impl PyTweedieFamily {
    #[new]
    #[pyo3(signature = (var_power=1.5))]
    fn new(var_power: f64) -> PyResult<Self> {
        if var_power > 0.0 && var_power < 1.0 {
            return Err(PyValueError::new_err(
                format!("var_power must be <= 0 or >= 1, got {}", var_power)
            ));
        }
        Ok(Self { inner: TweedieFamily::new(var_power) })
    }
}
```

### Method Returning Multiple Values

```rust
fn conf_int<'py>(
    &self,
    py: Python<'py>,
    alpha: f64,
) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
    let (lower, upper) = compute_ci(&self.coefficients, alpha);
    (lower.into_pyarray_bound(py), upper.into_pyarray_bound(py))
}
```

In Python: `lower, upper = result.conf_int(0.05)`

### Property with Computed Value

```rust
#[getter]
fn df_resid(&self) -> usize {
    self.n_obs.saturating_sub(self.n_params)
}
```
