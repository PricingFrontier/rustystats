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
| **`fit_glm()`** | ✅ Complete | Array-based fitting function with regularization |
| **`GLM` Class** | ✅ Complete | OOP interface (statsmodels-style) |
| **`glm()` Formula API** | ✅ Complete | R-style formula interface with DataFrame support |
| **Polars Support** | ✅ Complete | Native Polars DataFrame integration |
| **`summary()`** | ✅ Complete | Formatted regression tables |
| **`summary_relativities()`** | ✅ Complete | Factor tables for pricing |
| **`predict()`** | ✅ Complete | Predictions on new data |
| **`lasso_path()`** | ✅ Complete | Coefficient paths over alpha grid |
| **`cv_glm()`** | ✅ Complete | Cross-validation for optimal regularization |
| **Interaction Terms** | ✅ Complete | `x1*x2`, `C(cat):x`, `C(cat1)*C(cat2)` in formulas |
| **Spline Basis Functions** | ✅ Complete | `bs(x, df)`, `ns(x, df)` for non-linear effects |
| **Target Encoding** | ✅ Complete | `target_encode()`, `TE()` in formulas, `TargetEncoder` class |
| **Quasi-Families** | ✅ Complete | `quasipoisson`, `quasibinomial` for overdispersion |
| **Negative Binomial** | ✅ Complete | `negbinomial` with auto θ estimation |
| **Model Diagnostics** | ✅ Complete | `compute_diagnostics()`, `explore_data()`, JSON export for LLM consumption |
| **Minimal Dependencies** | ✅ Complete | Core requires only numpy; polars optional |

### Testing

| Component | Count | Description |
|-----------|-------|-------------|
| **Rust Unit Tests** | 190+ | Core library tests (families, diagnostics, solvers, regularization, inference, interactions, splines, formula, design_matrix, target_encoding, calibration, loss) |
| **Python Tests** | 280 | API, integration, regularization, robust SE, interaction, spline, quasi-family, negative binomial, target encoding, and diagnostics tests |

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
- **Polars workflows** - Native Polars DataFrame support
- **Variable selection** - Automatic feature selection with cross-validation

### When to Use Statsmodels

- **Broader model coverage** - OLS, WLS, GLS, mixed effects, time series
- **Established ecosystem** - More documentation, Stack Overflow answers

---

## API Overview

### Array-Based API
```python
import rustystats as rs
import numpy as np

# Fit a GLM with numpy arrays
result = rs.fit_glm(
    y, X,
    family="poisson",           # or "gaussian", "binomial", "gamma", "tweedie", "quasipoisson", "negbinomial"
    offset=np.log(exposure),
    weights=weights
)

# Negative Binomial for overdispersed count data
result = rs.fit_glm(y, X, family="negbinomial", theta=1.0)  # θ controls overdispersion
```

### Formula-Based API (NEW)
```python
import rustystats as rs
import polars as pl

data = pl.read_parquet("insurance.parquet")

result = rs.glm(
    formula="ClaimCount ~ VehPower + VehAge + C(Area) + C(Region)",
    data=data,
    family="poisson",      # or "negbinomial" (auto-estimates θ)
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

### Quasi-Families for Overdispersion (NEW)
```python
import rustystats as rs
import numpy as np

# Fit a standard Poisson model first
result_poisson = rs.fit_glm(y, X, family="poisson")

# Check for overdispersion: Pearson χ² / df >> 1 indicates overdispersion
dispersion_ratio = result_poisson.pearson_chi2() / result_poisson.df_resid
print(f"Dispersion ratio: {dispersion_ratio:.2f}")  # If >> 1, use quasi-family

# Fit QuasiPoisson if overdispersed
result_quasi = rs.fit_glm(y, X, family="quasipoisson")

# Coefficients are IDENTICAL to Poisson
np.allclose(result_quasi.params, result_poisson.params)  # True

# But standard errors are inflated by √φ
print(f"Estimated dispersion (φ): {result_quasi.scale():.3f}")
print(f"Poisson SE: {result_poisson.bse()}")
print(f"QuasiPoisson SE: {result_quasi.bse()}")  # Larger by √φ

# For binary data with overdispersion
result_qb = rs.fit_glm(y_binary, X, family="quasibinomial")
```

**Key properties of quasi-families:**
- **Point estimates**: Identical to base family (Poisson/Binomial)
- **Standard errors**: Inflated by √φ where φ = Pearson χ²/(n-p)
- **P-values**: More conservative (larger), accounting for extra variance
- **Confidence intervals**: Wider, correctly reflecting uncertainty

**When to use:**
- **QuasiPoisson**: Count data where Pearson χ²/df >> 1
- **QuasiBinomial**: Binary data with clusters or unobserved heterogeneity

### Negative Binomial for Overdispersed Counts (NEW)
```python
import rustystats as rs
import numpy as np

# Negative Binomial with θ=1.0 (moderate overdispersion)
result = rs.fit_glm(y, X, family="negbinomial", theta=1.0)

# θ controls overdispersion: Var(Y) = μ + μ²/θ
# - θ=0.5: Strong overdispersion (variance = μ + 2μ²)
# - θ=1.0: Moderate overdispersion (variance = μ + μ²)
# - θ=10: Mild overdispersion (close to Poisson)
# - θ→∞: Approaches Poisson (variance = μ)

# Compare to QuasiPoisson
result_quasi = rs.fit_glm(y, X, family="quasipoisson")
result_nb = rs.fit_glm(y, X, family="negbinomial", theta=2.0)

# NB is a proper probability distribution - AIC/BIC are valid
print(f"NB AIC: {result_nb.aic():.1f}")

# AUTOMATIC THETA ESTIMATION
# Option 1: Array API
result_auto = rs.fit_negbinomial(y, X)
print(result_auto.family)  # "NegativeBinomial(theta=2.1234)"

# Option 2: Formula API (auto when theta not supplied)
result = rs.glm("y ~ x1 + x2", data, family="negbinomial").fit()
print(result.family)  # "NegativeBinomial(theta=2.1234)"

# Option 3: Fixed theta
result = rs.glm("y ~ x1 + x2", data, family="negbinomial", theta=1.0).fit()
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

### Target Encoding for High-Cardinality Categoricals (NEW)
```python
import rustystats as rs
import numpy as np

# CatBoost-style ordered target statistics
# Prevents target leakage during training

# Direct API
categories = ["Toyota", "Ford", "BMW", "Toyota", "Ford", "BMW"]
target = np.array([1.0, 0.0, 1.0, 0.5, 0.2, 0.8])

encoded, name, prior, stats = rs.target_encode(
    categories, target, "brand",
    prior_weight=1.0,      # Regularization toward global mean
    n_permutations=4,      # Average across permutations for stability
    seed=42                # For reproducibility
)

# For prediction on new data (uses full training statistics)
new_cats = ["Toyota", "Honda"]  # Honda is unseen
new_encoded = rs.apply_target_encoding(new_cats, stats, prior)
# Unseen categories get the prior (global mean)

# Sklearn-style API
encoder = rs.TargetEncoder(prior_weight=1.0, n_permutations=4)
train_encoded = encoder.fit_transform(train_categories, train_target)
test_encoded = encoder.transform(test_categories)

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

**When to use:**
- High-cardinality categorical features (100s or 1000s of levels)
- When one-hot encoding would create too many columns
- When you want the model to learn category-target relationships

**Target Encoding vs One-Hot Encoding:**
| Aspect | Target Encoding | One-Hot Encoding |
|--------|-----------------|------------------|
| **Columns** | 1 per feature | k-1 per feature |
| **High cardinality** | Efficient | Explosive |
| **Target info** | Embedded | None |
| **Overfitting risk** | Controlled by prior | Lower |
| **Interpretability** | Single effect | Per-level effects |

---

### Model Diagnostics with JSON Export (NEW)
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

# =============================
# COMPREHENSIVE DIAGNOSTICS
# =============================

# Compute all diagnostics at once (returns ModelDiagnostics object)
diagnostics = result.diagnostics(
    data=data,
    categorical_factors=["Region", "VehBrand", "Area"],  # Including non-fitted
    continuous_factors=["Age", "Income", "VehPower"],    # Including non-fitted
)

# Export as compact JSON (optimized for LLM consumption)
json_str = diagnostics.to_json()
print(json_str)

# Or use the convenience method
json_str = result.diagnostics_json(
    data=data,
    categorical_factors=["Region", "VehBrand"],
    continuous_factors=["Age", "Income"],
)

# =============================
# WHAT'S IN THE JSON OUTPUT
# =============================

# Model summary
diagnostics.model_summary
# {"family": "poisson", "n_observations": 678012, "n_parameters": 15, ...}

# Fit statistics
diagnostics.fit_statistics
# {"deviance": 123456.7, "aic": 123500.1, "bic": 123600.2, "dispersion_pearson": 1.05, ...}

# Calibration metrics
diagnostics.calibration
# {"actual_expected_ratio": 0.998, "gini_coefficient": 0.42, "by_decile": [...], ...}

# Per-factor diagnostics (both fitted AND non-fitted factors)
for factor in diagnostics.factors:
    print(f"{factor.name}: in_model={factor.in_model}")
    print(f"  A/E by level: {len(factor.actual_vs_expected)} bins")
    print(f"  Residual correlation: {factor.residual_pattern.correlation_with_residuals:.3f}")

# Interaction candidates (greedy residual-based detection)
for ic in diagnostics.interaction_candidates:
    print(f"{ic.factor1} x {ic.factor2}: strength={ic.interaction_strength:.3f}")

# Warnings (auto-generated from diagnostics)
for warning in diagnostics.warnings:
    print(f"[{warning['type']}] {warning['message']}")

# =============================
# PRE-FIT DATA EXPLORATION
# =============================

# Explore data BEFORE fitting (no model needed)
exploration = rs.explore_data(
    data=data,
    response="ClaimNb",
    categorical_factors=["Region", "VehBrand", "Area"],
    continuous_factors=["Age", "VehPower", "Income"],
    exposure="Exposure",
    family="poisson",
    detect_interactions=True,
)

# View response distribution
print(exploration.response_stats)
# {"n_observations": 678012, "mean_rate": 0.045, "zeros_pct": 95.2, ...}

# Factor statistics (before fitting)
for factor in exploration.factor_stats:
    print(f"{factor['name']}: {factor['type']}")
    
# Interaction candidates (based on response variance)
for ic in exploration.interaction_candidates:
    print(f"Consider: {ic.factor1} * {ic.factor2}")

# Export as JSON
print(exploration.to_json())
```

**JSON Output Structure:**
```json
{
  "model_summary": {"family": "poisson", "n_observations": 678012, ...},
  "fit_statistics": {"deviance": 123456.7, "aic": 123500.1, ...},
  "calibration": {"actual_expected_ratio": 0.998, "by_decile": [...], ...},
  "discrimination": {"gini_coefficient": 0.42, "auc": 0.71, ...},
  "factors": [
    {
      "name": "Region",
      "factor_type": "categorical",
      "in_model": true,
      "actual_vs_expected": [{"level": "A", "ae_ratio": 1.02, ...}, ...],
      "residual_pattern": {"correlation": 0.01, ...}
    },
    {
      "name": "Income",
      "factor_type": "continuous", 
      "in_model": false,
      "actual_vs_expected": [{"bin": 1, "ae_ratio": 1.15, ...}, ...],
      "residual_pattern": {"correlation": 0.08, ...}
    }
  ],
  "interaction_candidates": [{"factor1": "Age", "factor2": "Region", "strength": 0.03}],
  "warnings": [{"type": "missing_factor", "message": "Income explains 2% of residual variance"}]
}
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

**Use Cases:**
- **Model Validation**: Check calibration across risk segments
- **Variable Selection**: Identify non-fitted factors with residual signal
- **Interaction Discovery**: Find missing interactions automatically
- **LLM Integration**: Feed compact JSON to AI for model analysis
- **Reporting**: Generate model assessment reports

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
│   ├── glm.py                    # GLM class, fit_glm, summary
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
        ├── test_glm.py           # GLM tests
        ├── test_families.py      # Family tests
        ├── test_links.py         # Link tests
        ├── test_interactions.py  # Interaction term tests
        ├── test_regularization.py # Lasso/Ridge/Elastic Net tests
        ├── test_robust_se.py     # Robust standard error tests
        ├── test_splines.py       # Spline basis function tests
        ├── test_quasi_families.py # QuasiPoisson/QuasiBinomial tests
        ├── test_negative_binomial.py # Negative Binomial tests
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

### Python (Optional)
- `polars` - DataFrame support (for formula API with DataFrames)

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
