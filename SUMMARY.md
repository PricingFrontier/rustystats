# RustyStats: Project Summary

A high-performance Generalized Linear Models (GLM) library with a Rust backend and Python API, designed for actuarial applications.

## What's Been Built

### Core Rust Library (`crates/rustystats-core`)

| Component | Status | Description |
|-----------|--------|-------------|
| **Link Functions** | ✅ Complete | Identity, Log, Logit |
| **Distribution Families** | ✅ Complete | Gaussian, Poisson, Binomial, Gamma, Tweedie |
| **IRLS Solver** | ✅ Complete | Iteratively Reweighted Least Squares fitting algorithm |
| **Offset Support** | ✅ Complete | For exposure-based rate models |
| **Prior Weights** | ✅ Complete | For grouped/aggregated data |
| **Statistical Inference** | ✅ Complete | Standard errors, z-values, p-values, confidence intervals |
| **Model Diagnostics** | ✅ Complete | Residuals, dispersion, AIC, BIC, log-likelihood |

### Python Bindings (`crates/rustystats`)

| Component | Status | Description |
|-----------|--------|-------------|
| **PyO3 Wrappers** | ✅ Complete | All Rust types exposed to Python |
| **NumPy Integration** | ✅ Complete | Seamless array conversion |
| **GLMResults Class** | ✅ Complete | Full results object with all inference methods |

### Python API (`python/rustystats`)

| Component | Status | Description |
|-----------|--------|-------------|
| **`fit_glm()`** | ✅ Complete | Array-based fitting function |
| **`GLM` Class** | ✅ Complete | OOP interface (statsmodels-style) |
| **`glm()` Formula API** | ✅ Complete | R-style formula interface with DataFrame support |
| **Polars Support** | ✅ Complete | Native Polars DataFrame integration |
| **Pandas Support** | ✅ Complete | Works with Pandas DataFrames too |
| **`summary()`** | ✅ Complete | Formatted regression tables |
| **`summary_relativities()`** | ✅ Complete | Factor tables for pricing |
| **`predict()`** | ✅ Complete | Predictions on new data |

### Testing

| Component | Count | Description |
|-----------|-------|-------------|
| **Rust Unit Tests** | 89 | Core library tests (families, diagnostics, solvers) |
| **Python Tests** | 88 | API and integration tests |

### Examples

| Notebook | Description |
|----------|-------------|
| `examples/getting_started.ipynb` | Comprehensive tutorial with all families |
| `examples/frequency.ipynb` | Claim frequency model with real insurance data |

---

## API Overview

### Array-Based API
```python
import rustystats as rs
import numpy as np

# Fit a GLM with numpy arrays
result = rs.fit_glm(
    y, X,
    family="poisson",
    offset=np.log(exposure),
    weights=weights
)
```

### Formula-Based API (NEW)
```python
import rustystats as rs
import polars as pl

data = pl.read_parquet("insurance.parquet")

result = rs.glm(
    formula="ClaimCount ~ VehPower + VehAge + C(Area) + C(Region)",
    data=data,
    family="poisson",
    offset="Exposure"
).fit()

# Rich results
print(result.summary())
print(result.coef_table())      # As pandas DataFrame
print(result.relativities())    # exp(coef) for log-link models
```

### Results Methods
```python
# Coefficients & Inference
result.params              # Coefficients
result.fittedvalues        # Predicted means
result.deviance            # Model deviance
result.bse()               # Standard errors
result.tvalues()           # z-statistics
result.pvalues()           # P-values
result.conf_int(alpha)     # Confidence intervals
result.significance_codes()# *, **, *** markers

# Diagnostics (statsmodels-compatible)
result.resid_response()    # Raw residuals (y - μ)
result.resid_pearson()     # Pearson residuals
result.resid_deviance()    # Deviance residuals
result.resid_working()     # Working residuals
result.llf()               # Log-likelihood
result.aic()               # Akaike Information Criterion
result.bic()               # Bayesian Information Criterion
result.null_deviance()     # Null model deviance
result.pearson_chi2()      # Pearson chi-squared
result.scale()             # Dispersion (deviance-based)
result.scale_pearson()     # Dispersion (Pearson-based)
result.family              # Family name
```

---

## Features To Be Added

### Medium Priority

| Feature | Description | Use Case |
|---------|-------------|----------|
| **Regularization** | Ridge (L2), Lasso (L1), Elastic Net | High-dimensional data, variable selection |
| **Quasi-Families** | Quasi-Poisson, Quasi-Binomial | Overdispersion handling |
| **Robust Standard Errors** | Sandwich estimator (HC0-HC3) | Heteroscedasticity |
| **Interaction Terms** | `x1 * x2` in formulas | Complex relationships |
| **Splines** | B-splines, natural splines | Non-linear continuous effects |

### Lower Priority

| Feature | Description | Use Case |
|---------|-------------|----------|
| **Negative Binomial** | Alternative to Poisson for overdispersion | Count data with extra variance |
| **Zero-Inflated Models** | ZIP, ZINB | Excess zeros in count data |
| **Mixed Effects / GLMM** | Random effects | Hierarchical/panel data |
| **Bootstrap CI** | Non-parametric confidence intervals | Small samples |
| **Cross-Validation** | K-fold CV utilities | Model selection |
| **Prediction Intervals** | Not just point predictions | Uncertainty quantification |

### Infrastructure

| Feature | Description |
|---------|-------------|
| **GPU Acceleration** | CUDA/Metal for large datasets |
| **Parallel Fitting** | Multi-threaded IRLS |
| **Streaming/Chunked Data** | Handle datasets larger than memory |
| **Model Serialization** | Save/load fitted models |
| **ONNX Export** | Deploy models in production |

---

## Project Structure

```
rustystats/
├── Cargo.toml                    # Workspace config
├── pyproject.toml                # Python package config
├── SUMMARY.md                    # This file
│
├── crates/
│   ├── rustystats-core/          # Pure Rust GLM library
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── error.rs          # Error types
│   │       ├── families/         # Gaussian, Poisson, Binomial, Gamma
│   │       ├── links/            # Identity, Log, Logit
│   │       ├── solvers/          # IRLS algorithm
│   │       └── inference/        # P-values, confidence intervals
│   │
│   └── rustystats/               # Python bindings (PyO3)
│       └── src/lib.rs
│
├── python/rustystats/            # Python package
│   ├── __init__.py               # Main exports
│   ├── glm.py                    # GLM class, fit_glm, summary
│   ├── formula.py                # Formula API with DataFrame support
│   ├── families.py               # Family wrappers
│   └── links.py                  # Link wrappers
│
├── examples/
│   ├── getting_started.ipynb     # Tutorial notebook
│   └── frequency.ipynb           # Claim frequency example
│
└── tests/
    └── python/
        ├── test_glm.py           # GLM tests (77 tests)
        ├── test_families.py      # Family tests
        └── test_links.py         # Link tests
```

---

## Performance

- **678,000 rows** fitted in ~2 seconds (Poisson with 9 parameters)
- Pure Rust IRLS solver with NumPy array zero-copy where possible
- Typically converges in 4-8 iterations

---

## Dependencies

### Rust
- `ndarray` - N-dimensional arrays
- `nalgebra` - Linear algebra
- `statrs` - Statistical distributions
- `pyo3` - Python bindings
- `numpy` - NumPy interop

### Python
- `numpy` - Array operations
- `polars` - DataFrame support
- `formulaic` - Formula parsing
- `pyarrow` - Polars ↔ Pandas conversion

---

## Getting Started

```bash
# Install in development mode
cd rustystats
uv run maturin develop

# Run tests
uv run pytest tests/python/ -v

# Start Jupyter
uv run jupyter notebook examples/frequency.ipynb
```
