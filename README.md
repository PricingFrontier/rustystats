# RustyStats 🦀📊

**High-performance Generalized Linear Models with a Rust backend and Python API**

Built for actuarial applications. Fits 678K rows in ~1 second.

## Features

- **Fast** — Parallel IRLS solver in Rust (Rayon)
- **Complete** — Families, regularization, inference, diagnostics
- **Flexible** — R-style formulas with interactions and splines
- **Minimal** — Core requires only `numpy` and `polars`

## Installation

```bash
# Development install
git clone https://github.com/PricingFrontier/rustystats.git
cd rustystats
uv run maturin develop --release

# Run tests
uv run pytest tests/python/
```

## Quick Start

```python
import rustystats as rs
import polars as pl

# Load data
data = pl.read_parquet("insurance.parquet")

# Fit a Poisson GLM for claim frequency
result = rs.glm(
    "ClaimCount ~ VehAge + VehPower + C(Area) + C(Region)",
    data=data,
    family="poisson",
    offset="Exposure"
).fit()

# View results
print(result.summary())
print(result.relativities())  # exp(coef) for pricing
```

---

## Families & Links

| Family | Default Link | Use Case |
|--------|--------------|----------|
| `gaussian` | identity | Linear regression |
| `poisson` | log | Claim frequency |
| `binomial` | logit | Binary outcomes |
| `gamma` | log | Claim severity |
| `tweedie` | log | Pure premium (var_power=1.5) |
| `quasipoisson` | log | Overdispersed counts |
| `quasibinomial` | logit | Overdispersed binary |
| `negbinomial` | log | Overdispersed counts (proper distribution) |

---

## Formula Syntax

```python
# Main effects
"y ~ x1 + x2 + C(category)"

# Interactions
"y ~ x1*x2"              # x1 + x2 + x1:x2
"y ~ C(area):age"        # Area-specific age effects
"y ~ C(area)*C(brand)"   # Categorical × categorical

# Splines (non-linear effects)
"y ~ bs(age, df=5)"      # B-spline basis
"y ~ ns(income, df=4)"   # Natural spline (better extrapolation)

# Target encoding (high-cardinality categoricals)
"y ~ TE(brand) + TE(model)"

# Combined
"y ~ bs(age, df=5) + C(region)*income + ns(vehicle_age, df=3) + TE(brand)"
```

---

## Results Methods

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

---

## Regularization

```python
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

---

## Interaction Terms

```python
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
```

---

## Spline Basis Functions

```python
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

---

## Quasi-Families for Overdispersion

```python
# Fit a standard Poisson model first
result_poisson = rs.glm("ClaimNb ~ Age + C(Region)", data, family="poisson", offset="Exposure").fit()

# Check for overdispersion: Pearson χ² / df >> 1 indicates overdispersion
dispersion_ratio = result_poisson.pearson_chi2() / result_poisson.df_resid
print(f"Dispersion ratio: {dispersion_ratio:.2f}")  # If >> 1, use quasi-family

# Fit QuasiPoisson if overdispersed
result_quasi = rs.glm("ClaimNb ~ Age + C(Region)", data, family="quasipoisson", offset="Exposure").fit()

# Coefficients are IDENTICAL to Poisson, but standard errors are inflated by √φ
print(f"Estimated dispersion (φ): {result_quasi.scale():.3f}")

# For binary data with overdispersion
result_qb = rs.glm("Binary ~ x1 + x2", data, family="quasibinomial").fit()
```

**Key properties of quasi-families:**
- **Point estimates**: Identical to base family (Poisson/Binomial)
- **Standard errors**: Inflated by √φ where φ = Pearson χ²/(n-p)
- **P-values**: More conservative (larger), accounting for extra variance

---

## Negative Binomial for Overdispersed Counts

```python
# Automatic θ estimation (default when theta not supplied)
result = rs.glm("ClaimNb ~ Age + C(Region)", data, family="negbinomial", offset="Exposure").fit()
print(result.family)  # "NegativeBinomial(theta=2.1234)"

# Fixed θ value
result = rs.glm("ClaimNb ~ Age + C(Region)", data, family="negbinomial", theta=1.0, offset="Exposure").fit()

# θ controls overdispersion: Var(Y) = μ + μ²/θ
# - θ=0.5: Strong overdispersion (variance = μ + 2μ²)
# - θ=1.0: Moderate overdispersion (variance = μ + μ²)
# - θ→∞: Approaches Poisson (variance = μ)
```

**NegativeBinomial vs QuasiPoisson:**
| Aspect | QuasiPoisson | NegativeBinomial |
|--------|--------------|------------------|
| **Variance** | φ × μ | μ + μ²/θ |
| **True distribution** | No (quasi) | Yes |
| **AIC/BIC valid** | Questionable | Yes |
| **Prediction intervals** | Not principled | Proper |

---

## Target Encoding for High-Cardinality Categoricals

```python
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

# Sklearn-style API
encoder = rs.TargetEncoder(prior_weight=1.0, n_permutations=4)
train_encoded = encoder.fit_transform(train_categories, train_target)
test_encoded = encoder.transform(test_categories)
```

**Key benefits:**
- **No target leakage**: CatBoost-style ordered target statistics
- **Regularization**: Prior weight controls shrinkage toward global mean
- **High-cardinality**: Single column instead of thousands of dummies

---

## Model Diagnostics

```python
# Compute all diagnostics at once
diagnostics = result.diagnostics(
    data=data,
    categorical_factors=["Region", "VehBrand", "Area"],  # Including non-fitted
    continuous_factors=["Age", "Income", "VehPower"],    # Including non-fitted
)

# Export as compact JSON (optimized for LLM consumption)
json_str = diagnostics.to_json()

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
```

**Diagnostic Features:**
- **Calibration**: Overall A/E ratio, calibration by decile with CIs, Hosmer-Lemeshow test
- **Discrimination**: Gini coefficient, AUC, KS statistic, lift metrics
- **Factor Diagnostics**: A/E by level/bin for ALL factors (fitted and non-fitted)
- **Interaction Detection**: Greedy residual-based detection of potential interactions
- **Warnings**: Auto-generated alerts for high dispersion, poor calibration, missing factors

---

## RustyStats vs Statsmodels

| Feature | RustyStats | Statsmodels |
|---------|------------|-------------|
| **Parallel IRLS Solver** | ✅ Multi-threaded via Rayon | ❌ Single-threaded only |
| **Native Polars Support** | ✅ Formula API works with Polars DataFrames | ❌ Pandas only |
| **Built-in Lasso/Elastic Net for GLMs** | ✅ Fast coordinate descent with all families | ⚠️ Limited |
| **Relativities Table** | ✅ `result.relativities()` for pricing | ❌ Must compute manually |
| **Robust Standard Errors** | ✅ HC0, HC1, HC2, HC3 sandwich estimators | ✅ HC0-HC3 |

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

## Project Structure

```
rustystats/
├── Cargo.toml                    # Workspace config
├── pyproject.toml                # Python package config
│
├── crates/
│   ├── rustystats-core/          # Pure Rust GLM library
│   │   └── src/
│   │       ├── families/         # Gaussian, Poisson, Binomial, Gamma, Tweedie, Quasi, NegativeBinomial
│   │       ├── links/            # Identity, Log, Logit
│   │       ├── solvers/          # IRLS, coordinate descent
│   │       ├── inference/        # P-values, CIs, robust SE (HC0-HC3)
│   │       ├── interactions/     # Lazy interaction term computation
│   │       ├── splines/          # B-spline and natural spline basis functions
│   │       ├── design_matrix/    # Categorical encoding, interaction matrices
│   │       ├── formula/          # R-style formula parsing
│   │       ├── target_encoding/  # CatBoost-style ordered target statistics
│   │       └── diagnostics/      # Residuals, dispersion, AIC/BIC, calibration, loss
│   │
│   └── rustystats/               # Python bindings (PyO3)
│       └── src/lib.rs
│
├── python/rustystats/            # Python package
│   ├── __init__.py               # Main exports
│   ├── formula.py                # Formula API with DataFrame support
│   ├── splines.py                # bs() and ns() spline basis functions
│   ├── target_encoding.py        # CatBoost-style target encoding
│   ├── diagnostics.py            # Model diagnostics with JSON export
│   └── families.py               # Family wrappers
│
├── examples/
│   └── frequency.ipynb           # Claim frequency example
│
└── tests/python/                 # Python test suite
```

---

## Dependencies

### Rust
- `ndarray`, `nalgebra` - Linear algebra
- `rayon` - Parallel iterators (multi-threading)
- `statrs` - Statistical distributions
- `pyo3` - Python bindings

### Python
- `numpy` - Array operations (required)
- `polars` - DataFrame support (required)

---

## License

MIT
