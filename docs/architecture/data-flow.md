# Data Flow

This chapter traces data through the system from user input to final results, helping you understand where transformations happen and how to debug issues.

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│ User Input                                                       │
│   • NumPy arrays (y, X)                                         │
│   • Or: DataFrame + formula string                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Python API Layer                                                 │
│   • Validate inputs                                              │
│   • Parse formula (if used)                                      │
│   • Build design matrix                                          │
│   • Convert to NumPy arrays                                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ PyO3 Binding Layer                                               │
│   • Convert NumPy → ndarray                                      │
│   • Create Rust Family/Link objects                              │
│   • Call core library                                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Rust Core (rustystats-core)                                      │
│   • Run IRLS/Coordinate Descent                                  │
│   • Return IRLSResult                                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ PyO3 Binding Layer                                               │
│   • Wrap IRLSResult as PyGLMResults                             │
│   • Convert ndarray → NumPy (when accessed)                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ User Output                                                      │
│   • GLMResults object with all methods                          │
└─────────────────────────────────────────────────────────────────┘
```

## Path 1: Array API

### User Code

```python
import rustystats as rs
import numpy as np

y = np.array([1, 2, 3, 4, 5], dtype=np.float64)
X = np.column_stack([np.ones(5), [1, 2, 3, 4, 5]])

result = rs.fit_glm(y, X, family="poisson")
```

### Step 1: Python API (`python/rustystats/glm.py`)

```python
def fit_glm(y, X, family="gaussian", link=None, offset=None, 
            weights=None, alpha=0.0, l1_ratio=1.0, ...):
    
    # 1. Convert to numpy arrays
    y = np.asarray(y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    
    # 2. Validate dimensions
    if y.ndim != 1:
        raise ValueError("y must be 1-dimensional")
    if X.ndim != 2:
        raise ValueError("X must be 2-dimensional")
    if y.shape[0] != X.shape[0]:
        raise ValueError("y and X must have same number of rows")
    
    # 3. Call Rust binding
    from rustystats._rustystats import fit_glm as _fit_glm
    rust_result = _fit_glm(y, X, family, alpha, l1_ratio, offset, weights)
    
    # 4. Return wrapped result
    return GLMResults(rust_result)
```

### Step 2: PyO3 Binding (`crates/rustystats/src/lib.rs`)

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
    
    // 1. Convert NumPy to ndarray (owned copies)
    let y_arr = y.as_array().to_owned();
    let x_arr = x.as_array().to_owned();
    let offset_arr = offset.map(|o| o.as_array().to_owned());
    let weights_arr = weights.map(|w| w.as_array().to_owned());
    
    // 2. Create Family and Link objects
    let (family_obj, link_obj): (Box<dyn Family>, Box<dyn Link>) = 
        match family {
            "gaussian" => (Box::new(GaussianFamily), Box::new(IdentityLink)),
            "poisson" => (Box::new(PoissonFamily), Box::new(LogLink)),
            // ... other families
        };
    
    // 3. Release GIL and call core library
    let result = py.allow_threads(|| {
        if alpha > 0.0 {
            fit_glm_regularized(...)
        } else {
            fit_glm_full(...)
        }
    })?;
    
    // 4. Wrap result
    Ok(PyGLMResults::from_irls_result(result, family))
}
```

### Step 3: Rust Core (`crates/rustystats-core/src/solvers/irls.rs`)

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
    
    // 1. Initialize
    let n = y.len();
    let p = x.ncols();
    let mut mu = family.initialize_mu(y);
    
    // 2. IRLS iterations
    for iter in 0..config.max_iterations {
        // Compute weights, working response
        // Solve WLS
        // Update predictions
        // Check convergence
    }
    
    // 3. Return result
    Ok(IRLSResult {
        coefficients,
        fitted_values: mu,
        deviance,
        converged,
        // ...
    })
}
```

### Step 4: Return Path

The `IRLSResult` is wrapped as `PyGLMResults` and returned to Python. Arrays are converted to NumPy only when accessed:

```rust
// In PyGLMResults
#[getter]
fn params<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
    // Conversion happens here, when user accesses .params
    self.coefficients.clone().into_pyarray_bound(py)
}
```

## Path 2: Formula API

### User Code

```python
import rustystats as rs
import polars as pl

data = pl.read_parquet("insurance.parquet")

result = rs.glm(
    formula="ClaimNb ~ Age + C(Region) + bs(VehPower, df=4)",
    data=data,
    family="poisson",
    offset="Exposure"
).fit()
```

### Step 1: Formula Parsing (`python/rustystats/formula.py`)

```python
class FormulaGLM:
    def __init__(self, formula, data, family, offset=None, ...):
        self.formula = formula
        self.data = data
        
        # Parse formula
        self.parsed = parse_formula(formula)
        # parsed = {
        #     'response': 'ClaimNb',
        #     'terms': ['Age', 'C(Region)', 'bs(VehPower, df=4)'],
        #     'interactions': []
        # }
    
    def fit(self):
        # 1. Extract response
        y = self.data[self.parsed['response']].to_numpy()
        
        # 2. Build design matrix
        X = self._build_design_matrix()
        
        # 3. Handle offset
        offset = None
        if self.offset_col:
            offset = np.log(self.data[self.offset_col].to_numpy())
        
        # 4. Call array API
        result = fit_glm(y, X, family=self.family, offset=offset, ...)
        
        # 5. Attach metadata
        result._feature_names = self._feature_names
        return result
```

### Step 2: Design Matrix Construction

```python
def _build_design_matrix(self):
    columns = [np.ones(len(self.data))]  # Intercept
    names = ['Intercept']
    
    for term in self.parsed['terms']:
        if term.startswith('C('):
            # Categorical encoding
            col_name = extract_column_name(term)
            encoded, term_names = encode_categorical(
                self.data[col_name].to_numpy()
            )
            columns.append(encoded)
            names.extend(term_names)
            
        elif term.startswith('bs(') or term.startswith('ns('):
            # Spline basis
            col_name, df = parse_spline_term(term)
            x = self.data[col_name].to_numpy()
            basis = bs(x, df=df) if term.startswith('bs') else ns(x, df=df)
            columns.append(basis)
            names.extend([f"{term}_{i}" for i in range(basis.shape[1])])
            
        else:
            # Continuous variable
            columns.append(self.data[term].to_numpy().reshape(-1, 1))
            names.append(term)
    
    self._feature_names = names
    return np.column_stack(columns)
```

### Subsequent Steps

After design matrix construction, the flow follows the Array API path.

## Data Type Conversions

### Python → Rust

| Python Type | Rust Type | Notes |
|-------------|-----------|-------|
| `np.ndarray` (1D, float64) | `Array1<f64>` | Via `PyReadonlyArray1` |
| `np.ndarray` (2D, float64) | `Array2<f64>` | Via `PyReadonlyArray2` |
| `str` | `&str` | Borrowed reference |
| `float` | `f64` | Direct conversion |
| `int` | `i64` or `usize` | Direct conversion |
| `None` | `None` (Option) | For optional params |

### Rust → Python

| Rust Type | Python Type | Notes |
|-----------|-------------|-------|
| `Array1<f64>` | `np.ndarray` (1D) | Via `into_pyarray_bound` |
| `Array2<f64>` | `np.ndarray` (2D) | Via `into_pyarray_bound` |
| `String` or `&str` | `str` | Automatic |
| `f64` | `float` | Automatic |
| `usize` | `int` | Automatic |
| `bool` | `bool` | Automatic |

## Memory Considerations

### Copies vs Views

**View (no copy):**
```rust
let y_view = y.as_array();  // Borrows NumPy memory
```

**Copy (owns data):**
```rust
let y_owned = y.as_array().to_owned();  // Copies to Rust memory
```

### When Copies Happen

1. **Python → Rust**: Always copied before IRLS (Rust needs ownership for parallel processing)
2. **Inside Rust**: Minimal copies; use views and in-place operations
3. **Rust → Python**: Copied when returning arrays (NumPy takes ownership)

### Large Dataset Considerations

For large datasets:
- Design matrix is the largest object
- IRLS stores: X, y, μ, η, weights, covariance matrix
- Regularization additionally stores the Gram matrix (p × p)

Memory estimate: ~5× the size of the design matrix during fitting.

## Tracing Issues

### Debug Points

1. **Python input validation**: Add prints in `fit_glm()`
2. **Design matrix**: Check `X.shape`, `X.dtype`, look for NaN/Inf
3. **Rust entry**: Add `println!` in the binding function
4. **IRLS iterations**: Enable `verbose=True` in config

### Common Issues

| Symptom | Likely Cause | Debug Step |
|---------|--------------|------------|
| Type error | Wrong dtype | Check `y.dtype`, `X.dtype` |
| Dimension error | Shape mismatch | Print shapes at each step |
| Convergence failure | Data issues | Check for outliers, collinearity |
| NaN in results | Numerical overflow | Check input ranges, add clamping |

### Example Debug Session

```python
# Enable verbose IRLS
result = rs.fit_glm(y, X, family="poisson", verbose=True)

# Check intermediate values
print(f"y range: [{y.min()}, {y.max()}]")
print(f"X shape: {X.shape}")
print(f"Any NaN in X: {np.isnan(X).any()}")
print(f"X condition number: {np.linalg.cond(X)}")
```

## Performance Profiling

### Python Level

```python
import time

start = time.time()
result = rs.fit_glm(y, X, family="poisson")
print(f"Total time: {time.time() - start:.3f}s")
print(f"IRLS iterations: {result.iterations}")
```

### Rust Level

```bash
# Build with profiling
cargo build --release -p rustystats-core

# Use perf or flamegraph
perf record -g ./target/release/benchmark
perf report
```

### Key Bottlenecks

1. **X'WX computation**: O(np²) - parallelized
2. **Matrix solve**: O(p³) - uses optimized BLAS via nalgebra
3. **Prediction updates**: O(np) - parallelized

For large n, the bottleneck is usually X'WX. For large p, matrix inversion dominates.
