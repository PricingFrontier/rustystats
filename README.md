# RustyStats 🦀📊

**High-performance Generalized Linear Models with a Rust backend and Python API**

**Codebase Documentation**: [pricingfrontier.github.io/rustystats/](https://pricingfrontier.github.io/rustystats/)

## Performance Benchmarks

**RustyStats vs Statsmodels** — Synthetic data, 101 features (10 continuous + 10 categorical with 10 levels each).

| Family | 10K rows | 250K rows | 500K rows |
|--------|----------|-----------|-----------|
| Gaussian | **15.6x** | **5.7x** | **4.3x** |
| Poisson | **16.3x** | **6.2x** | **4.2x** |
| Binomial | **19.5x** | **6.8x** | **4.4x** |
| Gamma | **33.7x** | **13.4x** | **8.4x** |
| NegBinomial | **26.7x** | **6.7x** | **5.0x** |

**Average speedup: 10.5x** (range: 4.2x – 33.7x)

### Memory Usage

RustyStats uses significantly less RAM by reusing buffers and avoiding Python object overhead:

| Rows | RustyStats | Statsmodels | Reduction |
|------|------------|-------------|-----------|
| 10K | 38 MB | 72 MB | **1.9x** |
| 250K | 460 MB | 1,796 MB | **3.9x** |
| 500K | 836 MB | 3,590 MB | **4.3x** |

*Memory advantage grows with data size — at 500K rows, RustyStats uses ~4x less RAM.*

<details>
<summary>Full benchmark details</summary>

| Family | Rows | RustyStats | Statsmodels | Speedup |
|--------|------|------------|-------------|---------|
| Gaussian | 10,000 | 0.100s | 1.559s | **15.6x** |
| Gaussian | 250,000 | 1.991s | 11.363s | **5.7x** |
| Gaussian | 500,000 | 4.023s | 17.386s | **4.3x** |
| Poisson | 10,000 | 0.165s | 2.692s | **16.3x** |
| Poisson | 250,000 | 2.429s | 15.072s | **6.2x** |
| Poisson | 500,000 | 5.668s | 23.693s | **4.2x** |
| Binomial | 10,000 | 0.112s | 2.189s | **19.5x** |
| Binomial | 250,000 | 1.946s | 13.155s | **6.8x** |
| Binomial | 500,000 | 4.708s | 20.862s | **4.4x** |
| Gamma | 10,000 | 0.129s | 4.353s | **33.7x** |
| Gamma | 250,000 | 2.385s | 31.885s | **13.4x** |
| Gamma | 500,000 | 5.499s | 46.167s | **8.4x** |
| NegBinomial | 10,000 | 0.119s | 3.177s | **26.7x** |
| NegBinomial | 250,000 | 2.281s | 15.278s | **6.7x** |
| NegBinomial | 500,000 | 4.821s | 24.331s | **5.0x** |

*Times are median of 3 runs. Benchmark scripts in `benchmarks/`.*

</details>

---

## Features

- **Fast** - Parallel Rust backend, 4-30x faster than statsmodels
- **Memory Efficient** - 4x less RAM than statsmodels at scale
- **Stable** - Step-halving IRLS, warm starts for robust convergence
- **Splines** - B-splines `bs()` and natural splines `ns()` with auto-tuned smoothing by default
- **Polynomials** - Identity terms `I(x ** 2)` for polynomial and arithmetic expressions
- **Target Encoding** - CatBoost-style `TE()` for high-cardinality categoricals (exposure-aware)
- **Regularisation** - Ridge, Lasso, and Elastic Net via coordinate descent
- **Validation** - Design matrix checks with fix suggestions before fitting
- **Complete** - 8 families, robust SEs, full diagnostics, VIF, partial dependence
- **Minimal** - Only `numpy` and `polars` required

## Installation

```bash
uv add rustystats
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

# Single-level categorical indicators
"y ~ C(Region, level='Paris')"              # 0/1 indicator for Paris only
"y ~ C(Region, levels=['Paris', 'Lyon'])"   # Indicators for specific levels

# Interactions
"y ~ x1*x2"              # x1 + x2 + x1:x2
"y ~ C(area):age"        # Area-specific age effects
"y ~ C(area)*C(brand)"   # Categorical × categorical

# Splines (non-linear effects)
"y ~ bs(age)"            # Penalized smooth (auto-tuned, default)
"y ~ bs(age, df=5)"      # Fixed 5 degrees of freedom
"y ~ ns(income)"         # Natural spline (auto-tuned)
"y ~ ns(income, df=4)"   # Natural spline (fixed df)
"y ~ bs(age, monotonicity='increasing')"   # Monotonic increasing
"y ~ bs(veh_age, df=4, monotonicity='decreasing')"  # Monotonic decreasing

# Identity terms (polynomial/arithmetic expressions)
"y ~ I(age ** 2)"        # Polynomial terms
"y ~ I(x1 * x2)"         # Explicit products
"y ~ I(income / 1000)"   # Scaled variables

# Coefficient constraints
"y ~ pos(age)"           # Coefficient ≥ 0
"y ~ neg(risk)"          # Coefficient ≤ 0
"y ~ neg(I(age ** 2))"   # Force downward curvature

# Target encoding (high-cardinality categoricals)
"y ~ TE(brand) + TE(model)"

# Combined
"y ~ bs(age, df=5) + C(region)*income + ns(vehicle_age, df=3) + TE(brand) + I(age ** 2)"
```

---

## Dict-Based API

Alternative to formula strings for programmatic model building. Useful for automated workflows and agentic systems.

```python
result = rs.glm_dict(
    response="ClaimCount",
    terms={
        "VehAge": {"type": "bs", "monotonicity": "increasing"},  # Monotonic (auto-tuned)
        "DrivAge": {"type": "bs"},                               # Penalized smooth (default)
        "Income": {"type": "bs", "df": 5},                       # Fixed 5 df
        "BonusMalus": {"type": "linear", "monotonicity": "increasing"},  # Constrained coefficient
        "Region": {"type": "categorical"},
        "Brand": {"type": "target_encoding"},
        "Age2": {"type": "expression", "expr": "DrivAge**2"},
    },
    interactions=[
        {
            "VehAge": {"type": "linear"}, 
            "Region": {"type": "categorical"}, 
            "include_main": True
        },
    ],
    data=data,
    family="poisson",
    offset="Exposure",
    seed=42,
).fit(regularization="elastic_net")
```

### Term Types

| Type | Parameters | Description |
|------|------------|-------------|
| `linear` | - | Raw continuous variable |
| `categorical` | `levels` (optional) | Dummy encoding |
| `bs` | `df` or `k`, `degree=3`, `monotonicity` | B-spline (default: penalized smooth, k=10) |
| `ns` | `df` or `k` | Natural spline (default: penalized smooth, k=10) |
| `target_encoding` | `prior_weight=1` | Regularized target encoding |
| `expression` | `expr` | Arbitrary expression (like `I()`) |

**Spline parameters:**
- No parameters → penalized smooth with automatic tuning (k=10)
- `df=5` → fixed 5 degrees of freedom
- `k=15` → penalized smooth with 15 basis functions
- `monotonicity="increasing"` or `"decreasing"` → constrained effect (bs only)

Add `"monotonicity": "increasing"` or `"decreasing"` to `linear` or `expression` terms to constrain coefficient sign.

### Interactions

Each interaction is a dict with variable specs and `include_main`:

```python
interactions=[
    # Main effects + interaction (like x*y)
    {
        "DrivAge": {"type": "bs", "df": 5}, 
        "Brand": {"type": "target_encoding"},
        "include_main": True
    },
    # Interaction only (like x:y)
    {
        "VehAge": {"type": "linear"}, 
        "Region": {"type": "categorical"}, 
        "include_main": False
    },
]
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

### CV-Based Regularization (Recommended)

```python
# Just specify regularization type - cv=5 is automatic
result = rs.glm("y ~ x1 + x2 + C(cat)", data, family="poisson").fit(
    regularization="ridge"  # "ridge", "lasso", or "elastic_net"
)

print(f"Selected alpha: {result.alpha}")
print(f"CV deviance: {result.cv_deviance}")
```

**Options:**
- `regularization`: `"ridge"` (L2), `"lasso"` (L1), or `"elastic_net"` (mix)
- `selection`: `"min"` (best fit) or `"1se"` (more conservative, default: `"min"`)
- `cv`: Number of folds (default: 5)

### Explicit Alpha

```python
# Skip CV, use specific alpha
result = rs.glm("y ~ x1 + x2", data).fit(alpha=0.1, l1_ratio=0.0)  # Ridge
result = rs.glm("y ~ x1 + x2", data).fit(alpha=0.1, l1_ratio=1.0)  # Lasso
result = rs.glm("y ~ x1 + x2", data).fit(alpha=0.1, l1_ratio=0.5)  # Elastic Net
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
# Default: penalized smooth with automatic tuning via GCV
result = rs.glm(
    "ClaimNb ~ bs(Age) + ns(VehPower) + C(Region)",
    data=data,
    family="poisson",
    offset="Exposure"
).fit()

# Fixed degrees of freedom (no penalty)
result = rs.glm(
    "ClaimNb ~ bs(Age, df=5) + ns(VehPower, df=4) + C(Region)",
    data=data,
    family="poisson",
    offset="Exposure"
).fit()

# Combine splines with interactions
result = rs.glm(
    "y ~ bs(age, df=4)*C(gender) + ns(income)",
    data=data,
    family="gaussian"
).fit()

# Direct basis computation
import numpy as np
x = np.linspace(0, 10, 100)
basis = rs.bs(x)        # Penalized smooth (default k=10)
basis = rs.bs(x, df=5)  # Fixed 5 df (4 basis columns)
basis = rs.ns(x, df=5)  # Natural spline, fixed 5 df
```

**When to use each spline type:**
- **B-splines (`bs`)**: Standard choice, more flexible at boundaries, supports monotonicity
- **Natural splines (`ns`)**: Better extrapolation, linear beyond boundaries (recommended for actuarial work)

**When to use `df` vs default:**
- **Default (no params)**: Auto-tuned smoothing via GCV - best for exploratory analysis
- **Explicit `df`**: Fixed complexity - use when you know the exact flexibility needed

---

## Monotonic Splines

Monotonic splines constrain the fitted curve to be monotonically increasing or decreasing. Essential when business logic dictates a monotonic relationship. Use `bs()` with the `monotonicity` parameter.

```python
# Monotonically increasing effect (e.g., age → risk)
result = rs.glm(
    "ClaimNb ~ bs(Age, monotonicity='increasing') + C(Region)",
    data=data,
    family="poisson",
    offset="Exposure"
).fit()

# Monotonically decreasing effect (e.g., vehicle value with age)
result = rs.glm(
    "ClaimAmt ~ bs(VehAge, df=4, monotonicity='decreasing')",
    data=data,
    family="gamma"
).fit()

# Combine with other spline types
result = rs.glm(
    "y ~ bs(age, monotonicity='increasing') + bs(income, df=4) + ns(experience)",
    data=data,
    family="gaussian"
).fit()

# Direct basis computation
basis = rs.bs(x, monotonicity='increasing')   # Monotonically increasing
basis = rs.bs(x, df=5, monotonicity='decreasing')  # Fixed df, decreasing
```

**Key properties:**
- Uses I-spline (integrated spline) basis internally
- All basis values in [0, 1]
- With non-negative coefficients, fitted curve is guaranteed monotonic
- Prevents implausible "wiggles" that can occur with unconstrained splines

**When to use:**
| Use Case | Formula |
|----------|---------|
| Age → claim frequency | `bs(age, monotonicity='increasing')` |
| Vehicle age → value | `bs(veh_age, monotonicity='decreasing')` |
| Credit score → risk | `bs(score, df=5, monotonicity='decreasing')` |

---

## Coefficient Constraints

Constrain coefficient signs using `pos()` (β ≥ 0) and `neg()` (β ≤ 0). Useful for enforcing business logic on linear and polynomial terms.

```python
# Constrain age coefficient to be positive
result = rs.glm(
    "y ~ pos(age) + income",
    data=data,
    family="poisson"
).fit()

# Force quadratic to bend downward (diminishing returns)
result = rs.glm(
    "y ~ age + neg(I(age ** 2))",
    data=data,
    family="gaussian"
).fit()

# Combine with monotonic splines
result = rs.glm(
    "ClaimNb ~ bs(VehAge, monotonicity='increasing') + pos(BonusMalus) + neg(I(DrivAge ** 2))",
    data=data,
    family="poisson",
    offset="Exposure"
).fit()
```

**Supported patterns:**
| Constraint | Effect | Example |
|------------|--------|---------|
| `pos(x)` | β ≥ 0 | `pos(age)` - positive effect |
| `neg(x)` | β ≤ 0 | `neg(risk)` - negative effect |
| `pos(I(x ** 2))` | β ≥ 0 | Upward curvature |
| `neg(I(x ** 2))` | β ≤ 0 | Downward curvature |

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
- **No target leakage**: Ordered target statistics
- **Regularization**: Prior weight controls shrinkage toward global mean
- **High-cardinality**: Single column instead of thousands of dummies
- **Exposure-aware**: For frequency models with `offset="Exposure"`, TE() automatically uses claim rate (ClaimCount/Exposure) instead of raw counts, preventing near-constant encoded values

---

## Identity Terms for Polynomials

```python
# Polynomial terms
result = rs.glm(
    "y ~ age + I(age ** 2) + I(age ** 3)",
    data=data,
    family="gaussian"
).fit()

# Arithmetic expressions
result = rs.glm(
    "y ~ I(income / 1000) + I(weight * height)",
    data=data,
    family="gaussian"
).fit()
```

**Supported operations:** `+`, `-`, `*`, `/`, `**` (power)

---

## Design Matrix Validation

```python
# Check for issues before fitting
model = rs.glm("y ~ ns(x, df=4) + C(cat)", data, family="poisson")
results = model.validate()  # Prints diagnostics

if not results['valid']:
    print("Issues:", results['suggestions'])

# Validation runs automatically on fit failure with helpful suggestions
```

**Checks performed:**
- Rank deficiency (linearly dependent columns)
- High multicollinearity (condition number)
- Zero variance columns
- NaN/Inf values
- Highly correlated column pairs (>0.999)

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
- **VIF/Multicollinearity**: Variance inflation factors for design matrix columns
- **Partial Dependence**: Effect plots with shape detection and recommendations
- **Overfitting Detection**: Compare train vs test metrics when test data provided
- **Interaction Detection**: Greedy residual-based detection of potential interactions
- **Warnings**: Auto-generated alerts for high dispersion, poor calibration, missing factors

---

## RustyStats vs Statsmodels

| Feature | RustyStats | Statsmodels |
|---------|------------|-------------|
| **Parallel IRLS Solver** | ✅ Multi-threaded | ❌ Single-threaded only |
| **Native Polars Support** | ✅ Polars only | ❌ Pandas only |
| **Built-in Lasso/Elastic Net for GLMs** | ✅ Fast coordinate descent with all families | ⚠️ Limited |
| **Relativities Table** | ✅ `result.relativities()` for pricing | ❌ Must compute manually |
| **Robust Standard Errors** | ✅ HC0, HC1, HC2, HC3 sandwich estimators | ✅ HC0-HC3 |

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
│   │       ├── target_encoding/  # Ordered target statistics
│   │       └── diagnostics/      # Residuals, dispersion, AIC/BIC, calibration, loss
│   │
│   └── rustystats/               # Python bindings (PyO3)
│       └── src/lib.rs
│
├── python/rustystats/            # Python package
│   ├── __init__.py               # Main exports
│   ├── formula.py                # Formula API with DataFrame support
│   ├── interactions.py           # Interaction terms, I() expressions, design matrix
│   ├── splines.py                # bs() and ns() spline basis functions
│   ├── target_encoding.py        # Target encoding (exposure-aware)
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

[Elastic License 2.0 (ELv2)](LICENSE) — Free to use, modify, and distribute. Cannot be offered as a hosted/managed service.
