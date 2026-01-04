# RustyStats: Project Summary

A high-performance Generalized Linear Models (GLM) library with a Rust backend and Python API, designed for actuarial applications.

## What's Been Built

### Core Rust Library (`crates/rustystats-core`)

| Component | Status | Description |
|-----------|--------|-------------|
| **Link Functions** | ✅ Complete | Identity, Log, Logit |
| **Distribution Families** | ✅ Complete | Gaussian, Poisson, Binomial, Gamma, Tweedie, QuasiPoisson, QuasiBinomial, NegativeBinomial |
| **IRLS Solver** | ✅ Complete | Multi-threaded Iteratively Reweighted Least Squares (parallel) |
| **Coordinate Descent** | ✅ Complete | For Lasso/Elastic Net with L1 penalty |
| **Formula Parsing** | ✅ Complete | R-style formula parsing (y ~ x1*x2 + C(cat) + bs(x, df=5) + TE(brand)) |
| **Design Matrix Builder** | ✅ Complete | Categorical encoding, interactions, splines (parallel) |
| **Target Encoding** | ✅ Complete | CatBoost-style ordered target statistics (prevents target leakage) |
| **Regularization** | ✅ Complete | Ridge (L2), Lasso (L1), Elastic Net |
| **Offset Support** | ✅ Complete | For exposure-based rate models |
| **Prior Weights** | ✅ Complete | For grouped/aggregated data |
| **Statistical Inference** | ✅ Complete | Standard errors, z-values, p-values, confidence intervals |
| **Robust Standard Errors** | ✅ Complete | Sandwich estimators (HC0, HC1, HC2, HC3) |
| **Model Diagnostics** | ✅ Complete | Residuals, dispersion, AIC, BIC, log-likelihood |
| **Advanced Diagnostics** | ✅ Complete | Calibration (A/E, Gini, Lorenz), discrimination, per-factor diagnostics, interaction detection |

### Python Bindings (`crates/rustystats`)

| Component | Status | Description |
|-----------|--------|-------------|
| **PyO3 Wrappers** | ✅ Complete | All Rust types exposed to Python |
| **NumPy Integration** | ✅ Complete | Seamless array conversion |
| **GLMResults Class** | ✅ Complete | Full results object with all inference methods |

### Python API (`python/rustystats`)

| Component | Status | Description |
|-----------|--------|-------------|
| **`glm()` Formula API** | ✅ Complete | R-style formula interface with DataFrame support |
| **Polars Support** | ✅ Complete | Native Polars DataFrame integration |
| **`summary()`** | ✅ Complete | Formatted regression tables |
| **`relativities()`** | ✅ Complete | Factor tables for pricing (exp(coef) for log-link) |
| **Regularization** | ✅ Complete | Ridge, Lasso, Elastic Net via `fit(alpha=..., l1_ratio=...)` |
| **Interaction Terms** | ✅ Complete | `x1*x2`, `C(cat):x`, `C(cat1)*C(cat2)` in formulas |
| **Spline Basis Functions** | ✅ Complete | `bs(x, df)`, `ns(x, df)` for non-linear effects |
| **Target Encoding** | ✅ Complete | `TE()` in formulas, `TargetEncoder` class |
| **Quasi-Families** | ✅ Complete | `quasipoisson`, `quasibinomial` for overdispersion |
| **Negative Binomial** | ✅ Complete | `negbinomial` with auto θ estimation |
| **Model Diagnostics** | ✅ Complete | `result.diagnostics()`, `explore_data()`, JSON export |
| **Minimal Dependencies** | ✅ Complete | Core requires only numpy; polars required for formula API |

### Testing

| Component | Count | Description |
|-----------|-------|-------------|
| **Rust Unit Tests** | 190+ | Core library tests (families, diagnostics, solvers, regularization, inference, interactions, splines, formula, design_matrix, target_encoding, calibration, loss) |
| **Python Tests** | 100+ | Formula API, interaction, spline, target encoding, and diagnostics tests |

### Examples

| Notebook | Description |
|----------|-------------|
| `examples/getting_started.ipynb` | Comprehensive tutorial with all families |
| `examples/frequency.ipynb` | Claim frequency model with real insurance data |

---

## RustyStats vs Statsmodels

### What RustyStats Has That Statsmodels Doesn't

| Feature | RustyStats | Statsmodels |
|---------|------------|-------------|
| **Parallel IRLS Solver** | ✅ Multi-threaded via Rayon | ❌ Single-threaded only |
| **Native Polars Support** | ✅ Formula API works with Polars DataFrames | ❌ Pandas only |
| **Built-in Lasso/Elastic Net for GLMs** | ✅ Fast coordinate descent with all families | ⚠️ Limited (regularized linear only) |
| **Relativities Table** | ✅ `result.relativities()` for pricing | ❌ Must compute manually |
| **Robust Standard Errors** | ✅ HC0, HC1, HC2, HC3 sandwich estimators | ✅ HC0-HC3 |
| **Performance on Large Data** | ✅ 678K rows in ~1s | ⚠️ Significantly slower |

### Performance Comparison (678,012 rows × 28 features)

| Operation | RustyStats | Statsmodels |
|-----------|------------|-------------|
| Poisson GLM | ~1.0s | ~5-10s |
| Ridge GLM | ~1.0s | ~5-10s |
| Lasso GLM | ~2.8s | Not available for GLMs |

### When to Use RustyStats

- **Large datasets** - Parallel solver scales better
- **Regularized GLMs** - Built-in Lasso/Ridge/Elastic Net for any family
- **Actuarial/Insurance** - Relativities tables, Tweedie, exposure offsets
- **Polars workflows** - Native Polars DataFrame support

### When to Use Statsmodels

- **Broader model coverage** - OLS, WLS, GLS, mixed effects, time series
- **Established ecosystem** - More documentation, Stack Overflow answers

---

## API Overview

### Formula-Based API
```python
import rustystats as rs
import polars as pl

data = pl.read_parquet("insurance.parquet")

result = rs.glm(
    formula="ClaimCount ~ VehPower + VehAge + C(Area) + C(Region)",
    data=data,
    family="poisson",      # or "gaussian", "binomial", "gamma", "tweedie", "quasipoisson", "negbinomial"
    offset="Exposure"
).fit()

# Rich results
print(result.summary())
print(result.coef_table())      # As Polars DataFrame
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

### Regularization
```python
import rustystats as rs
import polars as pl

data = pl.read_parquet("insurance.parquet")

# Ridge (L2) - shrinks coefficients, keeps all variables
result = rs.glm("y ~ x1 + x2 + C(cat)", data, family="gaussian").fit(
    alpha=0.1, l1_ratio=0.0
)

# Lasso (L1) - variable selection, zeros out weak predictors
result = rs.glm("y ~ x1 + x2 + C(cat)", data, family="poisson").fit(
    alpha=0.1, l1_ratio=1.0
)
print(f"Selected {result.n_nonzero()} variables")
print(f"Features: {result.selected_features()}")

# Elastic Net - mix of L1 and L2
result = rs.glm("y ~ x1 + x2 + C(cat)", data, family="gaussian").fit(
    alpha=0.1, l1_ratio=0.5
)
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

### Spline Basis Functions
```python
import rustystats as rs
import polars as pl

data = pl.read_parquet("insurance.parquet")

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
import numpy as np
x = np.linspace(0, 10, 100)
basis = rs.bs(x, df=5)  # 5 degrees of freedom (4 basis columns)
basis_ns = rs.ns(x, df=5)  # Natural splines - linear extrapolation at boundaries
```

**When to use each spline type:**
- **B-splines (`bs`)**: Standard choice, more flexible at boundaries
- **Natural splines (`ns`)**: Better extrapolation, linear beyond boundaries (recommended for actuarial work)

### Quasi-Families for Overdispersion
```python
import rustystats as rs
import polars as pl

data = pl.read_parquet("insurance.parquet")

# Fit a standard Poisson model first
result_poisson = rs.glm("ClaimNb ~ Age + C(Region)", data, family="poisson", offset="Exposure").fit()

# Check for overdispersion: Pearson χ² / df >> 1 indicates overdispersion
dispersion_ratio = result_poisson.pearson_chi2() / result_poisson.df_resid
print(f"Dispersion ratio: {dispersion_ratio:.2f}")  # If >> 1, use quasi-family

# Fit QuasiPoisson if overdispersed
result_quasi = rs.glm("ClaimNb ~ Age + C(Region)", data, family="quasipoisson", offset="Exposure").fit()

# Coefficients are IDENTICAL to Poisson
# But standard errors are inflated by √φ
print(f"Estimated dispersion (φ): {result_quasi.scale():.3f}")

# For binary data with overdispersion
result_qb = rs.glm("Binary ~ x1 + x2", data, family="quasibinomial").fit()
```

**Key properties of quasi-families:**
- **Point estimates**: Identical to base family (Poisson/Binomial)
- **Standard errors**: Inflated by √φ where φ = Pearson χ²/(n-p)
- **P-values**: More conservative (larger), accounting for extra variance
- **Confidence intervals**: Wider, correctly reflecting uncertainty

**When to use:**
- **QuasiPoisson**: Count data where Pearson χ²/df >> 1
- **QuasiBinomial**: Binary data with clusters or unobserved heterogeneity

### Negative Binomial for Overdispersed Counts
```python
import rustystats as rs
import polars as pl

data = pl.read_parquet("insurance.parquet")

# Automatic θ estimation (default when theta not supplied)
result = rs.glm("ClaimNb ~ Age + C(Region)", data, family="negbinomial", offset="Exposure").fit()
print(result.family)  # "NegativeBinomial(theta=2.1234)"

# Fixed θ value
result = rs.glm("ClaimNb ~ Age + C(Region)", data, family="negbinomial", theta=1.0, offset="Exposure").fit()

# θ controls overdispersion: Var(Y) = μ + μ²/θ
# - θ=0.5: Strong overdispersion (variance = μ + 2μ²)
# - θ=1.0: Moderate overdispersion (variance = μ + μ²)
# - θ=10: Mild overdispersion (close to Poisson)
# - θ→∞: Approaches Poisson (variance = μ)
```

**NegativeBinomial vs QuasiPoisson:**
| Aspect | QuasiPoisson | NegativeBinomial |
|--------|--------------|------------------|
| **Variance** | φ × μ | μ + μ²/θ |
| **True distribution** | No (quasi) | Yes |
| **Likelihood-based** | No | Yes |
| **AIC/BIC valid** | Questionable | Yes |
| **Prediction intervals** | Not principled | Proper |

**When to use:**
- **QuasiPoisson**: Quick fix, no θ to specify
- **NegativeBinomial**: Proper inference, prediction intervals

### Target Encoding for High-Cardinality Categoricals
```python
import rustystats as rs
import polars as pl

data = pl.read_parquet("insurance.parquet")

# Formula API - TE() in formulas
result = rs.glm(
    "ClaimNb ~ TE(Brand) + TE(Model) + Age + C(Region)",
    data=data,
    family="poisson",
    offset="Exposure"
).fit()

# With options
result = rs.glm(
    "y ~ TE(brand, prior_weight=2.0, n_permutations=8) + age",
    data=data,
    family="gaussian"
).fit()

# Direct API for custom use
import numpy as np
categories = ["Toyota", "Ford", "BMW", "Toyota", "Ford", "BMW"]
target = np.array([1.0, 0.0, 1.0, 0.5, 0.2, 0.8])

encoded, name, prior, stats = rs.target_encode(
    categories, target, "brand",
    prior_weight=1.0,      # Regularization toward global mean
    n_permutations=4,      # Average across permutations for stability
    seed=42                # For reproducibility
)

# Sklearn-style API
encoder = rs.TargetEncoder(prior_weight=1.0, n_permutations=4)
train_encoded = encoder.fit_transform(train_categories, train_target)
test_encoded = encoder.transform(test_categories)
```

**How it works (CatBoost algorithm):**
1. Shuffle data with random permutation
2. For each observation i in permutation order:
   `encoded[i] = (sum_target_before + prior × prior_weight) / (count_before + prior_weight)`
3. Average across multiple permutations to reduce variance
4. For prediction: use full training statistics

**Key benefits:**
- **No target leakage**: Each observation's encoding uses only "past" data
- **Regularization**: Prior weight controls shrinkage toward global mean
- **High-cardinality**: Single column instead of thousands of dummies
- **Rare categories**: Automatically regularized toward global mean

### Model Diagnostics with JSON Export
```python
import rustystats as rs
import polars as pl

data = pl.read_parquet("insurance.parquet")

# Fit a model
result = rs.glm(
    "ClaimNb ~ Age + C(Region) + C(VehBrand)",
    data=data,
    family="poisson",
    offset="Exposure"
).fit()

# Compute all diagnostics at once
diagnostics = result.diagnostics(
    data=data,
    categorical_factors=["Region", "VehBrand", "Area"],  # Including non-fitted
    continuous_factors=["Age", "Income", "VehPower"],    # Including non-fitted
)

# Export as compact JSON (optimized for LLM consumption)
json_str = diagnostics.to_json()
print(json_str)

# Pre-fit data exploration (no model needed)
exploration = rs.explore_data(
    data=data,
    response="ClaimNb",
    categorical_factors=["Region", "VehBrand", "Area"],
    continuous_factors=["Age", "VehPower", "Income"],
    exposure="Exposure",
    family="poisson",
    detect_interactions=True,
)
print(exploration.to_json())
```

**Key Features:**
| Feature | Description |
|---------|-------------|
| **Calibration** | Overall A/E ratio, calibration by decile with CIs, Hosmer-Lemeshow test |
| **Discrimination** | Gini coefficient, AUC, KS statistic, lift metrics, Lorenz curve |
| **Factor Diagnostics** | A/E by level/bin for ALL factors (fitted and non-fitted) |
| **Residual Patterns** | Correlation of residuals with each factor (identifies missing effects) |
| **Interaction Detection** | Greedy residual-based detection of potential interactions |
| **Warnings** | Auto-generated alerts for high dispersion, poor calibration, missing factors |
| **JSON Export** | Token-efficient serialization for LLM consumption |

---

## Features To Be Added

### Lower Priority

| Feature | Description | Use Case |
|---------|-------------|----------|
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
│   │       ├── families/         # Gaussian, Poisson, Binomial, Gamma, Tweedie, Quasi, NegativeBinomial
│   │       ├── links/            # Identity, Log, Logit
│   │       ├── solvers/          # IRLS, coordinate descent
│   │       ├── inference/        # P-values, CIs, robust SE (HC0-HC3)
│   │       ├── interactions/     # Lazy interaction term computation
│   │       ├── splines/          # B-spline and natural spline basis functions
│   │       ├── design_matrix/    # Categorical encoding, interaction matrices
│   │       ├── formula/          # R-style formula parsing
│   │       ├── target_encoding/  # CatBoost-style ordered target statistics
│   │       └── diagnostics/      # Residuals, dispersion, AIC/BIC, calibration, loss, factor diagnostics
│   │
│   └── rustystats/               # Python bindings (PyO3)
│       └── src/lib.rs
│
├── python/rustystats/            # Python package
│   ├── __init__.py               # Main exports
│   ├── glm.py                    # Summary formatting functions
│   ├── formula.py                # Formula API with DataFrame support
│   ├── interactions.py           # Optimized interaction term handling
│   ├── splines.py                # bs() and ns() spline basis functions
│   ├── target_encoding.py        # CatBoost-style target encoding
│   ├── diagnostics.py            # Model diagnostics with JSON export
│   ├── families.py               # Family wrappers
│   └── links.py                  # Link wrappers
│
├── examples/
│   ├── getting_started.ipynb     # Tutorial notebook
│   └── frequency.ipynb           # Claim frequency example
│
└── tests/
    └── python/
        ├── test_families.py      # Family tests
        ├── test_links.py         # Link tests
        ├── test_interactions.py  # Interaction term tests
        ├── test_splines.py       # Spline basis function tests
        ├── test_target_encoding.py # Target encoding tests
        └── test_diagnostics.py   # Model diagnostics tests
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
- `polars` - DataFrame support (required for formula API)

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
