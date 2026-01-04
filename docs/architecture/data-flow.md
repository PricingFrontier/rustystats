# Data Flow

This chapter traces data through the system from user input to final results, helping you understand where transformations happen and how to debug issues.

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│ User Input                                                       │
│   • DataFrame + formula string                                   │
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

## Formula API Data Flow

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

### Step 3: Rust Core Fitting

After design matrix construction, the formula API calls the Rust core library via PyO3 bindings to run the IRLS solver.

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

1. **Python input validation**: Add prints in formula.py
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
import polars as pl

# Check your data
data = pl.read_parquet("insurance.parquet")
print(f"Data shape: {data.shape}")
print(f"Columns: {data.columns}")

# Fit with verbose output
result = rs.glm("y ~ x1 + C(cat)", data, family="poisson").fit()
print(f"Converged: {result.converged}")
print(f"Iterations: {result.iterations}")
```

## Performance Profiling

### Python Level

```python
import time

start = time.time()
result = rs.glm("y ~ x1 + x2 + C(cat)", data, family="poisson").fit()
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
