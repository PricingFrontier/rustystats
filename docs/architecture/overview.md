# Architecture Overview

RustyStats is a hybrid Rust/Python library. This chapter explains how the components fit together and the design principles behind the architecture.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Python User Code                          │
│       import rustystats as rs; rs.glm_dict(...).fit()           │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Python API Layer                              │
│              python/rustystats/*.py                              │
│   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌──────────┐             │
│   │ glm.py  │ │formula.py│ │splines.py│ │diagnostics│             │
│   └────┬────┘ └────┬────┘ └────┬────┘ └─────┬────┘             │
└────────┼───────────┼───────────┼────────────┼───────────────────┘
         │           │           │            │
         └───────────┴─────┬─────┴────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PyO3 Bindings Layer                           │
│              crates/rustystats/src/lib.rs                        │
│   Converts Python objects ↔ Rust types using NumPy/PyO3         │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Rust Core Library                             │
│              crates/rustystats-core/src/                         │
│   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌──────────┐             │
│   │families │ │ links   │ │ solvers │ │inference │             │
│   └─────────┘ └─────────┘ └─────────┘ └──────────┘             │
│   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌──────────┐             │
│   │ splines │ │ formula │ │design_mx│ │diagnostics│             │
│   └─────────┘ └─────────┘ └─────────┘ └──────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

## Design Principles

### 1. Separation of Concerns

The codebase is split into three layers:

| Layer | Location | Responsibility |
|-------|----------|----------------|
| **Python API** | `python/rustystats/` | User-facing interface, DataFrame handling |
| **PyO3 Bindings** | `crates/rustystats/` | Type conversion, Python ↔ Rust bridge |
| **Rust Core** | `crates/rustystats-core/` | All mathematical computation |

### 2. Pure Rust Core

`rustystats-core` has **no Python dependencies**. It's a pure Rust library that could be used independently. Benefits:

- Testable without Python
- Could support other language bindings (R, Julia)
- Clear API boundary

### 3. Minimal Python Dependencies

The Python layer requires only `numpy`. Optional dependencies (`polars`) are imported lazily.

### 4. Trait-Based Extensibility

Core abstractions use Rust traits:

```rust
pub trait Family: Send + Sync { ... }
pub trait Link: Send + Sync { ... }
```

New families/links can be added by implementing these traits.

### 5. Parallel by Default

Computationally intensive operations use Rayon for automatic parallelization:

```rust
use rayon::prelude::*;

// Parallel matrix multiplication
let result = (0..n).into_par_iter()
    .fold(|| init, |acc, i| compute(acc, i))
    .reduce(|| init, |a, b| combine(a, b));
```

## Crate Structure

### rustystats-core

The pure Rust computation library:

```
crates/rustystats-core/
├── Cargo.toml
└── src/
    ├── lib.rs              # Re-exports, module declarations
    ├── error.rs            # Error types
    ├── families/           # Distribution families
    │   ├── mod.rs          # Family trait
    │   ├── gaussian.rs
    │   ├── poisson.rs
    │   └── ...
    ├── links/              # Link functions
    │   ├── mod.rs          # Link trait
    │   ├── identity.rs
    │   ├── log.rs
    │   └── logit.rs
    ├── solvers/            # Fitting algorithms
    │   ├── mod.rs
    │   ├── irls.rs         # Main IRLS solver
    │   └── coordinate_descent.rs
    ├── inference/          # Statistical inference
    │   └── mod.rs          # SEs, p-values, robust SEs
    ├── diagnostics/        # Model diagnostics
    │   ├── mod.rs
    │   ├── residuals.rs
    │   ├── calibration.rs
    │   ├── negbinomial.rs  # NegBin theta estimation, log-likelihood
    │   └── ...
    ├── splines/            # Spline basis functions
    │   └── mod.rs
    ├── formula/            # Formula parsing
    │   └── mod.rs
    ├── design_matrix/      # Design matrix construction
    │   └── mod.rs
    ├── target_encoding/    # Target encoding
    │   └── mod.rs
    ├── regularization/     # Penalty configuration
    │   └── mod.rs
    └── interactions/       # Interaction terms
        └── mod.rs
```

### rustystats (Python bindings)

The PyO3 bridge:

```
crates/rustystats/
├── Cargo.toml
└── src/
    └── lib.rs              # All Python-facing code
```

This single file:
- Wraps Rust types as Python classes (`#[pyclass]`)
- Exposes functions to Python (`#[pyfunction]`)
- Handles NumPy array conversion

### Python Package

High-level Python API:

```
python/rustystats/
├── __init__.py             # Public exports
├── glm.py                  # Summary formatting functions
├── formula.py              # Formula API, glm()
├── families.py             # Python family wrappers
├── links.py                # Python link wrappers
├── splines.py              # bs(), ns() functions
├── target_encoding.py      # target_encode(), TargetEncoder
├── interactions.py         # Interaction term utilities
└── diagnostics.py          # ModelDiagnostics, explore_data()
```

## Data Flow

### Fitting a Model

```
User calls rs.glm_dict(response="y", terms={...}, data=data).fit()
           │
           ▼
┌──────────────────────────────────────────┐
│ python/rustystats/formula.py             │
│ - Convert dict spec to ParsedFormula     │
│ - Extract columns from DataFrame         │
│ - Build design matrix                    │
│ - Handle categoricals, splines, etc.     │
│ - Call Rust via _rustystats              │
└──────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────┐
│ crates/rustystats/src/lib.rs             │
│ - Convert PyArray → ndarray::Array       │
│ - Create Rust Family/Link objects        │
│ - Call fit_glm_full()                    │
└──────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────┐
│ crates/rustystats-core/src/solvers/      │
│ - Run IRLS iterations                    │
│ - Compute X'WX, X'Wz (parallel)          │
│ - Solve linear system                    │
│ - Return IRLSResult                      │
└──────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────┐
│ Back to Python                           │
│ - Wrap IRLSResult as PyGLMResults        │
│ - Convert arrays back to NumPy           │
│ - Return GLMModel to user       │
└──────────────────────────────────────────┘
```

## Error Handling

### Rust Errors

Custom error type with context:

```rust
#[derive(Debug)]
pub enum RustyStatsError {
    InvalidInput(String),
    ConvergenceFailure { iterations: usize, tolerance: f64 },
    NumericalError(String),
    DimensionMismatch { expected: usize, got: usize },
}
```

### Python Errors

Rust errors are converted to Python exceptions:

```rust
impl From<RustyStatsError> for PyErr {
    fn from(err: RustyStatsError) -> PyErr {
        PyValueError::new_err(format!("{}", err))
    }
}
```

## Memory Management

### Zero-Copy When Possible

NumPy arrays can be viewed without copying:

```rust
fn process(arr: PyReadonlyArray1<f64>) -> ... {
    let view = arr.as_array();  // No copy, just a view
    // ... work with view
}
```

### Copies When Necessary

When Rust needs ownership or the array will be modified:

```rust
let owned = arr.as_array().to_owned();  // Copy to owned Array
```

### Returning to Python

Arrays are converted back to NumPy:

```rust
result.into_pyarray_bound(py)  // Moves ownership to Python
```

## Thread Safety

### Rust Side

All traits require `Send + Sync`:

```rust
pub trait Family: Send + Sync { ... }
```

This allows parallel iteration with Rayon.

### Python GIL

PyO3 handles the GIL automatically. Rust code releases the GIL during computation:

```rust
py.allow_threads(|| {
    // This code runs without the GIL
    // Python can run other threads
    expensive_computation()
})
```

## Testing Strategy

### Rust Unit Tests

Each module has tests:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_variance_function() {
        let family = PoissonFamily;
        let mu = array![1.0, 2.0, 3.0];
        let var = family.variance(&mu);
        assert_eq!(var, mu);  // Poisson: V(μ) = μ
    }
}
```

Run with: `cargo test -p rustystats-core`

### Python Integration Tests

Located in `tests/python/`:

```python
def test_poisson_fit():
    data = pl.DataFrame({
        "y": np.random.poisson(5, 100),
        "x": np.random.randn(100),
    })
    result = rs.glm_dict(response="y", terms={"x": {"type": "linear"}}, data=data, family="poisson").fit()
    assert result.converged
```

Run with: `uv run pytest tests/python/ -v`

### Comparison Tests

Compare against statsmodels:

```python
def test_vs_statsmodels():
    data = pl.DataFrame({"y": y, "x": x})
    
    # Fit with RustyStats
    rs_result = rs.glm_dict(response="y", terms={"x": {"type": "linear"}}, data=data, family="gaussian").fit()
    
    # Fit with statsmodels
    sm_result = sm.GLM(y, sm.add_constant(x), family=sm.families.Gaussian()).fit()
    
    # Compare coefficients
    np.testing.assert_allclose(rs_result.params, sm_result.params, rtol=1e-5)
```

## Build System

### Maturin

The Python package is built using maturin:

```toml
# pyproject.toml
[build-system]
requires = ["maturin>=1.4"]
build-backend = "maturin"

[tool.maturin]
manifest-path = "crates/rustystats/Cargo.toml"
python-source = "python"
module-name = "rustystats._rustystats"
```

### Development Workflow

```bash
# Compile Rust and install Python package
maturin develop

# Release build (optimized)
maturin develop --release

# Run tests
cargo test                          # Rust tests
uv run pytest tests/python/ -v      # Python tests
```
