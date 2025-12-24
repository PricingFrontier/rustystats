# RustyStats: Project Summary

A high-performance Generalized Linear Models (GLM) library with a Rust backend and Python API, designed for actuarial applications.

## What's Been Built

### Core Rust Library (`crates/rustystats-core`)

| Component | Status | Description |
|-----------|--------|-------------|
| **Link Functions** | ✅ Complete | Identity, Log, Logit |
| **Distribution Families** | ✅ Complete | Gaussian, Poisson, Binomial, Gamma, Tweedie |
| **IRLS Solver** | ✅ Complete | Multi-threaded Iteratively Reweighted Least Squares (parallel) |
| **Coordinate Descent** | ✅ Complete | For Lasso/Elastic Net with L1 penalty |
| **Formula Parsing** | ✅ Complete | R-style formula parsing (y ~ x1*x2 + C(cat) + bs(x, df=5)) |
| **Design Matrix Builder** | ✅ Complete | Categorical encoding, interactions, splines (parallel) |
| **Regularization** | ✅ Complete | Ridge (L2), Lasso (L1), Elastic Net |
| **Offset Support** | ✅ Complete | For exposure-based rate models |
| **Prior Weights** | ✅ Complete | For grouped/aggregated data |
| **Statistical Inference** | ✅ Complete | Standard errors, z-values, p-values, confidence intervals |
| **Robust Standard Errors** | ✅ Complete | Sandwich estimators (HC0, HC1, HC2, HC3) |
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
| **`fit_glm()`** | ✅ Complete | Array-based fitting function with regularization |
| **`GLM` Class** | ✅ Complete | OOP interface (statsmodels-style) |
| **`glm()` Formula API** | ✅ Complete | R-style formula interface with DataFrame support |
| **Polars Support** | ✅ Complete | Native Polars DataFrame integration |
| **Pandas Support** | ✅ Complete | Works with Pandas DataFrames too |
| **`summary()`** | ✅ Complete | Formatted regression tables |
| **`summary_relativities()`** | ✅ Complete | Factor tables for pricing |
| **`predict()`** | ✅ Complete | Predictions on new data |
| **`lasso_path()`** | ✅ Complete | Coefficient paths over alpha grid |
| **`cv_glm()`** | ✅ Complete | Cross-validation for optimal regularization |
| **Interaction Terms** | ✅ Complete | `x1*x2`, `C(cat):x`, `C(cat1)*C(cat2)` in formulas |
| **Spline Basis Functions** | ✅ Complete | `bs(x, df)`, `ns(x, df)` for non-linear effects |
| **Minimal Dependencies** | ✅ Complete | Core requires only numpy; pandas/polars optional |

### Testing

| Component | Count | Description |
|-----------|-------|-------------|
| **Rust Unit Tests** | 130+ | Core library tests (families, diagnostics, solvers, regularization, inference, interactions, splines, formula, design_matrix) |
| **Python Tests** | 175 | API, integration, regularization, robust SE, interaction, and spline tests |

### Examples

| Notebook | Description |
|----------|-------------|
| `examples/getting_started.ipynb` | Comprehensive tutorial with all families |
| `examples/frequency.ipynb` | Claim frequency model with real insurance data |
| `examples/regularization.ipynb` | Ridge, Lasso, Elastic Net with cross-validation |

---

## RustyStats vs Statsmodels

### What RustyStats Has That Statsmodels Doesn't

| Feature | RustyStats | Statsmodels |
|---------|------------|-------------|
| **Parallel IRLS Solver** | ✅ Multi-threaded via Rayon | ❌ Single-threaded only |
| **Native Polars Support** | ✅ Formula API works with Polars DataFrames | ❌ Pandas only |
| **Built-in Lasso/Elastic Net for GLMs** | ✅ Fast coordinate descent with all families | ⚠️ Limited (regularized linear only) |
| **Variable Selection Utilities** | ✅ `lasso_path()`, `cv_glm()` | ❌ Not built-in |
| **Relativities Table** | ✅ `summary_relativities()` for pricing | ❌ Must compute manually |
| **Robust Standard Errors** | ✅ HC0, HC1, HC2, HC3 sandwich estimators | ✅ HC0-HC3 |
| **Performance on Large Data** | ✅ 678K rows in ~1s | ⚠️ Significantly slower |

### Performance Comparison (678,012 rows × 28 features)

| Operation | RustyStats | Statsmodels |
|-----------|------------|-------------|
| Poisson GLM | ~1.0s | ~5-10s |
| Ridge GLM | ~1.0s | ~5-10s |
| Lasso GLM | ~2.8s | Not available for GLMs |
| Elastic Net GLM | ~2.6s | Not available for GLMs |

### When to Use RustyStats

- **Large datasets** - Parallel solver scales better
- **Regularized GLMs** - Built-in Lasso/Ridge/Elastic Net for any family
- **Actuarial/Insurance** - Relativities tables, Tweedie, exposure offsets
- **Polars workflows** - Native DataFrame support without pandas conversion
- **Variable selection** - Automatic feature selection with cross-validation

### When to Use Statsmodels

- **Broader model coverage** - OLS, WLS, GLS, mixed effects, time series
- **Established ecosystem** - More documentation, Stack Overflow answers
- **Advanced diagnostics** - Influence plots, leverage, Cook's distance

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

# Robust Standard Errors (sandwich estimators)
result.bse_robust("HC1")   # Robust SE (HC0, HC1, HC2, HC3)
result.tvalues_robust()    # z-stats with robust SE
result.pvalues_robust()    # P-values with robust SE
result.conf_int_robust()   # Confidence intervals with robust SE
result.cov_robust()        # Full robust covariance matrix

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

### Interaction Terms
```python
import rustystats as rs
import polars as pl

data = pl.read_parquet("insurance.parquet")

# Continuous × Continuous interaction (main effects + interaction)
result = rs.glm(
    "ClaimNb ~ Age*VehPower",  # Equivalent to Age + VehPower + Age:VehPower
    data, family="poisson", offset="Exposure"
).fit()

# Categorical × Continuous interaction
result = rs.glm(
    "ClaimNb ~ C(Area)*Age",  # Each area level has different age effect
    data, family="poisson", offset="Exposure"
).fit()

# Categorical × Categorical interaction
result = rs.glm(
    "ClaimNb ~ C(Area)*C(VehBrand)",
    data, family="poisson", offset="Exposure"
).fit()

# Pure interaction (no main effects added)
result = rs.glm(
    "ClaimNb ~ Age + C(Area):VehPower",  # Area-specific VehPower slopes
    data, family="poisson", offset="Exposure"
).fit()

# View coefficients
print(result.summary())
print(result.coef_table())
```

### Regularization API (NEW)
```python
import rustystats as rs

# Ridge (L2) - shrinks coefficients, keeps all variables
result = rs.fit_glm(y, X, family="gaussian", alpha=0.1, l1_ratio=0.0)

# Lasso (L1) - variable selection, zeros out weak predictors
result = rs.fit_glm(y, X, family="poisson", alpha=0.1, l1_ratio=1.0)
print(f"Selected {result.n_nonzero()} variables")
print(f"Features: {result.selected_features()}")

# Elastic Net - mix of L1 and L2
result = rs.fit_glm(y, X, family="gaussian", alpha=0.1, l1_ratio=0.5)

# Coefficient path - see how coefficients shrink with alpha
path = rs.lasso_path(y, X, family="gaussian", n_alphas=50)
path.plot()  # Visualize the path

# Cross-validation - find optimal alpha
cv_result = rs.cv_glm(y, X, family="poisson", l1_ratio=1.0, cv=5)
print(f"Best alpha: {cv_result.alpha_best}")
print(f"1-SE alpha: {cv_result.alpha_1se}")  # More parsimonious model
cv_result.plot()  # Visualize CV curve
```

### Spline Basis Functions (NEW)
```python
import rustystats as rs
import numpy as np

# B-spline basis - flexible piecewise polynomials
x = np.linspace(0, 10, 100)
basis = rs.bs(x, df=5)  # 5 degrees of freedom (4 basis columns)

# Natural splines - linear extrapolation at boundaries
basis_ns = rs.ns(x, df=5)  # Better for prediction outside training range

# Use splines in formulas - automatic parsing
result = rs.glm(
    "ClaimNb ~ bs(Age, df=5) + ns(VehPower, df=4) + C(Region)",
    data=data,
    family="poisson",
    offset="Exposure"
).fit()

# Combine splines with interactions
result = rs.glm(
    "y ~ bs(age, df=4)*C(gender) + ns(income, df=3)",
    data=data,
    family="gaussian"
).fit()

# Direct basis computation for custom use
basis = rs.bs(age_values, df=6, degree=3, boundary_knots=(18, 80))
```

**When to use each spline type:**
- **B-splines (`bs`)**: Standard choice, more flexible at boundaries
- **Natural splines (`ns`)**: Better extrapolation, linear beyond boundaries (recommended for actuarial work)

---

## Features To Be Added

### Medium Priority

| Feature | Description | Use Case |
|---------|-------------|----------|
| **Quasi-Families** | Quasi-Poisson, Quasi-Binomial | Overdispersion handling |

### Lower Priority

| Feature | Description | Use Case |
|---------|-------------|----------|
| **Negative Binomial** | Alternative to Poisson for overdispersion | Count data with extra variance |
| **Zero-Inflated Models** | ZIP, ZINB | Excess zeros in count data |
| **Mixed Effects / GLMM** | Random effects | Hierarchical/panel data |
| **Bootstrap CI** | Non-parametric confidence intervals | Small samples |
| **Prediction Intervals** | Not just point predictions | Uncertainty quantification |

### Infrastructure

| Feature | Description |
|---------|-------------|
| **GPU Acceleration** | CUDA/Metal for large datasets |
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
│   │       ├── solvers/          # IRLS, coordinate descent
│   │       ├── inference/        # P-values, CIs, robust SE (HC0-HC3)
│   │       ├── interactions/     # Lazy interaction term computation
│   │       ├── splines/          # B-spline and natural spline basis functions
│   │       ├── design_matrix/    # Categorical encoding, interaction matrices
│   │       ├── formula/          # R-style formula parsing
│   │       └── diagnostics/      # Residuals, dispersion, AIC/BIC
│   │
│   └── rustystats/               # Python bindings (PyO3)
│       └── src/lib.rs
│
├── python/rustystats/            # Python package
│   ├── __init__.py               # Main exports
│   ├── glm.py                    # GLM class, fit_glm, summary
│   ├── formula.py                # Formula API with DataFrame support
│   ├── interactions.py           # Optimized interaction term handling
│   ├── splines.py                # bs() and ns() spline basis functions
│   ├── families.py               # Family wrappers
│   └── links.py                  # Link wrappers
│
├── examples/
│   ├── getting_started.ipynb     # Tutorial notebook
│   └── frequency.ipynb           # Claim frequency example
│
└── tests/
    └── python/
        ├── test_glm.py           # GLM tests
        ├── test_families.py      # Family tests
        ├── test_links.py         # Link tests
        ├── test_interactions.py  # Interaction term tests
        ├── test_regularization.py # Lasso/Ridge/Elastic Net tests
        ├── test_robust_se.py     # Robust standard error tests
        └── test_splines.py       # Spline basis function tests
```

---

## Performance

- **678,000 rows × 28 features** fitted in ~1 second (Poisson GLM)
- **Lasso/Elastic Net** use glmnet-style covariance updates for O(p²) coordinate descent
- Pure Rust IRLS solver with multi-threaded parallelism (Rayon)
- NumPy array zero-copy where possible
- Typically converges in 4-8 IRLS iterations

### Parallelized Operations

| Operation | Parallelization |
|-----------|-----------------|
| **X'WX computation** (IRLS) | Parallel fold-reduce over observations |
| **X'WX Gram matrix** (Lasso/CD) | Parallel fold-reduce with flat Vec for cache locality |
| **Robust covariance meat** | Parallel fold-reduce with upper-triangle optimization |
| **Leverage computation** (HC2/HC3) | Parallel over observations |

---

## Dependencies

### Rust
- `ndarray` - N-dimensional arrays
- `nalgebra` - Linear algebra
- `rayon` - Parallel iterators (multi-threading)
- `statrs` - Statistical distributions
- `pyo3` - Python bindings
- `numpy` - NumPy interop

### Python (Core)
- `numpy` - Array operations (required)

### Python (Optional)
- `polars` - DataFrame support (for formula API with DataFrames)
- `pandas` - Only needed for `summary_df()`, `relativities()` output as DataFrames
- `formulaic` - Fallback formula parsing (rustystats has its own Rust parser)

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
