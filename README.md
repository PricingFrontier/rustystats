# RustyStats 🦀📊

**High-performance Generalized Linear Models with a Rust backend and Python API**

**Codebase Documentation**: [pricingfrontier.github.io/rustystats/](https://pricingfrontier.github.io/rustystats/)

## Performance Benchmarks

**RustyStats vs Statsmodels** — Synthetic data, 101 features (10 continuous + 10 categorical with 10 levels each).

| Family | 10K rows | 250K rows | 500K rows |
|--------|----------|-----------|-----------|
| Gaussian | **18.3x** | **6.4x** | **5.1x** |
| Poisson | **19.6x** | **7.1x** | **5.2x** |
| Binomial | **23.5x** | **7.1x** | **5.4x** |
| Gamma | **9.0x** | **13.4x** | **8.9x** |
| NegBinomial | **22.5x** | **7.2x** | **5.0x** |

**Average speedup: 10.9x** (range: 5.0x – 23.5x)

### Memory Usage

| Rows | RustyStats | Statsmodels | Reduction |
|------|------------|-------------|-----------|
| 10K | 4 MB | 72 MB | **18x** |
| 250K | 253 MB | 1,796 MB | **7.1x** |
| 500K | 780 MB | 3,590 MB | **4.6x** |

*Memory advantage grows with data size — at 500K rows, RustyStats uses ~4.6x less RAM.*

<details>
<summary>Full benchmark details</summary>

| Family | Rows | RustyStats | Statsmodels | Speedup |
|--------|------|------------|-------------|--------|
| Gaussian | 10,000 | 0.085s | 1.559s | **18.3x** |
| Gaussian | 250,000 | 1.769s | 11.363s | **6.4x** |
| Gaussian | 500,000 | 3.399s | 17.386s | **5.1x** |
| Poisson | 10,000 | 0.137s | 2.692s | **19.6x** |
| Poisson | 250,000 | 2.128s | 15.072s | **7.1x** |
| Poisson | 500,000 | 4.581s | 23.693s | **5.2x** |
| Binomial | 10,000 | 0.093s | 2.189s | **23.5x** |
| Binomial | 250,000 | 1.851s | 13.155s | **7.1x** |
| Binomial | 500,000 | 3.842s | 20.862s | **5.4x** |
| Gamma | 10,000 | 0.486s | 4.353s | **9.0x** |
| Gamma | 250,000 | 2.377s | 31.885s | **13.4x** |
| Gamma | 500,000 | 5.202s | 46.167s | **8.9x** |
| NegBinomial | 10,000 | 0.141s | 3.177s | **22.5x** |
| NegBinomial | 250,000 | 2.128s | 15.278s | **7.2x** |
| NegBinomial | 500,000 | 4.900s | 24.331s | **5.0x** |

*Times are median of 3 runs. Benchmark scripts in `benchmarks/`.*

</details>

---

## Features

- **Dict-First API** - Programmatic model building ideal for automated workflows and agents
- **Fast** - Parallel Rust backend, 4-30x faster than statsmodels
- **Memory Efficient** - 4x less RAM than statsmodels at scale
- **Stable** - Step-halving IRLS, warm starts for robust convergence
- **Splines** - B-splines and natural splines with auto-tuned smoothing and monotonicity
- **Target Encoding** - Ordered target encoding for high-cardinality categoricals
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
result = rs.glm_dict(
    response="ClaimCount",
    terms={
        "VehAge": {"type": "linear"},
        "VehPower": {"type": "linear"},
        "Area": {"type": "categorical"},
        "Region": {"type": "categorical"},
    },
    data=data,
    family="poisson",
    offset="Exposure",
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

## Dict-Based API

API built for programmatic model building.

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
| `linear` | `monotonicity` (optional) | Raw continuous variable |
| `categorical` | `levels` (optional) | Dummy encoding |
| `bs` | `df` or `k`, `degree=3`, `monotonicity` | B-spline (default: penalized smooth, k=10) |
| `ns` | `df` or `k` | Natural spline (default: penalized smooth, k=10) |
| `target_encoding` | `prior_weight=1` | Regularized target encoding |
| `expression` | `expr`, `monotonicity` (optional) | Arbitrary expression (like `I()`) |

**Spline parameters:**
- No parameters → penalized smooth with automatic tuning (k=10)
- `df=5` → fixed 5 degrees of freedom
- `k=15` → penalized smooth with 15 basis functions
- `monotonicity="increasing"` or `"decreasing"` → constrained effect (bs only)

Add `"monotonicity": "increasing"` or `"decreasing"` to `linear` or `expression` terms to constrain coefficient sign.

### Interactions

Each interaction is a dict with variable specs. Use `include_main` to also add main effects.

```python
interactions=[
    # Standard interaction: product terms (main effects + interaction)
    {
        "DrivAge": {"type": "bs", "df": 5}, 
        "Brand": {"type": "target_encoding"},
        "include_main": True
    },
    # Categorical × continuous (interaction only)
    {
        "VehAge": {"type": "linear"}, 
        "Region": {"type": "categorical"}, 
        "include_main": False
    },
    # TE interaction: combined target encoding TE(Brand:Region)
    {
        "Brand": {"type": "categorical"},
        "Region": {"type": "categorical"},
        "target_encoding": True,
        "prior_weight": 1.0,  # optional
    },
    # FE interaction: combined frequency encoding FE(Brand:Region)
    {
        "Brand": {"type": "categorical"},
        "Region": {"type": "categorical"},
        "frequency_encoding": True,
    },
]
```

**Interaction types:**
| Flag | Effect |
|------|--------|
| (none) | Standard product terms (cat×cat, cat×cont, etc.) |
| `target_encoding: True` | Combined TE encoding: `TE(var1:var2)` |
| `frequency_encoding: True` | Combined FE encoding: `FE(var1:var2)` |

---

## Formula Syntax (Alternative)

For those who prefer R-style formula strings:

```python
result = rs.glm("ClaimCount ~ VehAge + C(Region) + TE(Brand)", data, family="poisson").fit()
```

<details>
<summary>Formula syntax reference</summary>

```python
# Main effects
"y ~ x1 + x2 + C(category)"

# Single-level categorical indicators
"y ~ C(Region, level='Paris')"              # 0/1 indicator for Paris only

# Interactions
"y ~ x1*x2"              # x1 + x2 + x1:x2
"y ~ C(area):age"        # Area-specific age effects

# Splines (non-linear effects)
"y ~ bs(age)"            # Penalized smooth (auto-tuned)
"y ~ bs(age, df=5)"      # Fixed 5 degrees of freedom
"y ~ ns(income)"         # Natural spline (auto-tuned)
"y ~ bs(age, monotonicity='increasing')"   # Monotonic

# Identity terms (polynomial/arithmetic expressions)
"y ~ I(age ** 2)"        # Polynomial terms

# Coefficient constraints
"y ~ pos(age)"           # Coefficient ≥ 0
"y ~ neg(risk)"          # Coefficient ≤ 0

# Target encoding (high-cardinality categoricals)
"y ~ TE(brand) + TE(model)"
```

</details>

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

## Model Serialization

Save and load fitted models for later use:

```python
# Fit and save
result = rs.glm_dict(
    response="ClaimNb",
    terms={
        "Age": {"type": "bs"},
        "Region": {"type": "categorical"},
        "Brand": {"type": "target_encoding"},
    },
    data=data,
    family="poisson",
    offset="Exposure",
).fit()
model_bytes = result.to_bytes()

with open("model.bin", "wb") as f:
    f.write(model_bytes)

# Load later
with open("model.bin", "rb") as f:
    loaded = rs.GLMModel.from_bytes(f.read())

# Predict with loaded model
predictions = loaded.predict(new_data)
```

**What's preserved:**
- Coefficients and feature names
- Categorical encoding levels
- Spline knot positions
- Target encoding statistics
- Formula, family, link function

**Compact storage:** Only prediction-essential state is stored (~KB, not MB).

---

## Regularization

### CV-Based Regularization

```python
# Just specify regularization type - cv=5 is automatic
result = rs.glm_dict(
    response="y",
    terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}, "cat": {"type": "categorical"}},
    data=data,
    family="poisson",
).fit(regularization="ridge")  # "ridge", "lasso", or "elastic_net"

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
result = rs.glm_dict(response="y", terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}}, data=data).fit(alpha=0.1, l1_ratio=0.0)  # Ridge
result = rs.glm_dict(response="y", terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}}, data=data).fit(alpha=0.1, l1_ratio=1.0)  # Lasso
result = rs.glm_dict(response="y", terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}}, data=data).fit(alpha=0.1, l1_ratio=0.5)  # Elastic Net
```

---

## Interaction Terms

```python
# Continuous × Continuous interaction (main effects + interaction)
result = rs.glm_dict(
    response="ClaimNb",
    terms={},
    interactions=[{
        "Age": {"type": "linear"},
        "VehPower": {"type": "linear"},
        "include_main": True,  # Includes Age + VehPower + Age:VehPower
    }],
    data=data, family="poisson", offset="Exposure",
).fit()

# Categorical × Continuous interaction
result = rs.glm_dict(
    response="ClaimNb",
    terms={},
    interactions=[{
        "Area": {"type": "categorical"},
        "Age": {"type": "linear"},
        "include_main": True,  # Each area level has different age effect
    }],
    data=data, family="poisson", offset="Exposure",
).fit()

# Pure interaction (no main effects added)
result = rs.glm_dict(
    response="ClaimNb",
    terms={"Age": {"type": "linear"}},
    interactions=[{
        "Area": {"type": "categorical"},
        "VehPower": {"type": "linear"},
        "include_main": False,  # Area-specific VehPower slopes only
    }],
    data=data, family="poisson", offset="Exposure",
).fit()

# Target encoding interaction: TE(Brand:Region)
# Creates single encoded column for brand×region combinations
result = rs.glm_dict(
    response="ClaimNb",
    terms={"Age": {"type": "linear"}},
    interactions=[{
        "Brand": {"type": "categorical"},
        "Region": {"type": "categorical"},
        "target_encoding": True,
        "prior_weight": 1.0,
    }],
    data=data, family="poisson", offset="Exposure",
).fit()

# Frequency encoding interaction: FE(Brand:Region)
result = rs.glm_dict(
    response="ClaimNb",
    terms={"Age": {"type": "linear"}},
    interactions=[{
        "Brand": {"type": "categorical"},
        "Region": {"type": "categorical"},
        "frequency_encoding": True,
    }],
    data=data, family="poisson", offset="Exposure",
).fit()
```

---

## Spline Basis Functions

```python
# Default: penalized smooth with automatic tuning via GCV
result = rs.glm_dict(
    response="ClaimNb",
    terms={
        "Age": {"type": "bs"},           # B-spline (auto-tuned)
        "VehPower": {"type": "ns"},      # Natural spline (auto-tuned)
        "Region": {"type": "categorical"},
    },
    data=data, family="poisson", offset="Exposure",
).fit()

# Fixed degrees of freedom (no penalty)
result = rs.glm_dict(
    response="ClaimNb",
    terms={
        "Age": {"type": "bs", "df": 5},       # Fixed 5 df
        "VehPower": {"type": "ns", "df": 4},  # Fixed 4 df
        "Region": {"type": "categorical"},
    },
    data=data, family="poisson", offset="Exposure",
).fit()

# Splines with interactions
result = rs.glm_dict(
    response="y",
    terms={"income": {"type": "ns"}},
    interactions=[{
        "age": {"type": "bs", "df": 4},
        "gender": {"type": "categorical"},
        "include_main": True,
    }],
    data=data, family="gaussian",
).fit()

```

**When to use each spline type:**
- **B-splines (`bs`)**: Standard choice, more flexible at boundaries, supports monotonicity
- **Natural splines (`ns`)**: Better extrapolation, linear beyond boundaries

---

## Monotonic Splines

Monotonic splines constrain the fitted curve to be monotonically increasing or decreasing. Essential when business logic dictates a monotonic relationship.

```python
# Monotonically increasing effect (e.g., age → risk)
result = rs.glm_dict(
    response="ClaimNb",
    terms={
        "Age": {"type": "bs", "monotonicity": "increasing"},
        "Region": {"type": "categorical"},
    },
    data=data, family="poisson", offset="Exposure",
).fit()

# Monotonically decreasing effect (e.g., vehicle value with age)
result = rs.glm_dict(
    response="ClaimAmt",
    terms={"VehAge": {"type": "bs", "df": 4, "monotonicity": "decreasing"}},
    data=data, family="gamma",
).fit()

# Combine monotonic and unconstrained splines
result = rs.glm_dict(
    response="y",
    terms={
        "age": {"type": "bs", "monotonicity": "increasing"},
        "income": {"type": "bs", "df": 4},
        "experience": {"type": "ns"},
    },
    data=data, family="gaussian",
).fit()

```
---

## Coefficient Constraints

Constrain coefficient signs using `monotonicity` on linear and expression terms.

```python
# Constrain age coefficient to be positive
result = rs.glm_dict(
    response="y",
    terms={
        "age": {"type": "linear", "monotonicity": "increasing"},  # β ≥ 0
        "income": {"type": "linear"},
    },
    data=data, family="poisson",
).fit()

# Force quadratic to bend downward (diminishing returns)
result = rs.glm_dict(
    response="y",
    terms={
        "age": {"type": "linear"},
        "age2": {"type": "expression", "expr": "age ** 2", "monotonicity": "decreasing"},  # β ≤ 0
    },
    data=data, family="gaussian",
).fit()

# Combine with monotonic splines
result = rs.glm_dict(
    response="ClaimNb",
    terms={
        "VehAge": {"type": "bs", "monotonicity": "increasing"},
        "BonusMalus": {"type": "linear", "monotonicity": "increasing"},
        "DrivAge2": {"type": "expression", "expr": "DrivAge ** 2", "monotonicity": "decreasing"},
    },
    data=data, family="poisson", offset="Exposure",
).fit()
```

**Supported patterns:**
| Constraint | Term Spec | Effect |
|------------|-----------|--------|
| β ≥ 0 | `"monotonicity": "increasing"` | Positive effect |
| β ≤ 0 | `"monotonicity": "decreasing"` | Negative effect |

---

## Quasi-Families for Overdispersion

```python
# Fit a standard Poisson model first
result_poisson = rs.glm_dict(
    response="ClaimNb",
    terms={"Age": {"type": "linear"}, "Region": {"type": "categorical"}},
    data=data, family="poisson", offset="Exposure",
).fit()

# Check for overdispersion: Pearson χ² / df >> 1 indicates overdispersion
dispersion_ratio = result_poisson.pearson_chi2() / result_poisson.df_resid
print(f"Dispersion ratio: {dispersion_ratio:.2f}")  # If >> 1, use quasi-family

# Fit QuasiPoisson if overdispersed
result_quasi = rs.glm_dict(
    response="ClaimNb",
    terms={"Age": {"type": "linear"}, "Region": {"type": "categorical"}},
    data=data, family="quasipoisson", offset="Exposure",
).fit()

# Coefficients are IDENTICAL to Poisson, but standard errors are inflated by √φ
print(f"Estimated dispersion (φ): {result_quasi.scale():.3f}")

# For binary data with overdispersion
result_qb = rs.glm_dict(
    response="Binary",
    terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
    data=data, family="quasibinomial",
).fit()
```

---

## Target Encoding for High-Cardinality Categoricals

```python
# Dict API - target_encoding type
result = rs.glm_dict(
    response="ClaimNb",
    terms={
        "Brand": {"type": "target_encoding"},
        "Model": {"type": "target_encoding"},
        "Age": {"type": "linear"},
        "Region": {"type": "categorical"},
    },
    data=data, family="poisson", offset="Exposure",
).fit()

# With options
result = rs.glm_dict(
    response="y",
    terms={
        "brand": {"type": "target_encoding", "prior_weight": 2.0},
        "age": {"type": "linear"},
    },
    data=data, family="gaussian",
).fit()

# Sklearn-style API
encoder = rs.TargetEncoder(prior_weight=1.0, n_permutations=4)
train_encoded = encoder.fit_transform(train_categories, train_target)
test_encoded = encoder.transform(test_categories)

# TE interaction: encode combinations of multiple categoricals
result = rs.glm_dict(
    response="ClaimNb",
    terms={"Age": {"type": "linear"}},
    interactions=[{
        "Brand": {"type": "categorical"},
        "Region": {"type": "categorical"},
        "target_encoding": True,  # Creates TE(Brand:Region)
    }],
    data=data, family="poisson", offset="Exposure",
).fit()
```

**Key benefits:**
- **No target leakage**: Ordered target statistics
- **Regularization**: Prior weight controls shrinkage toward global mean
- **High-cardinality**: Single column instead of thousands of dummies
- **Exposure-aware**: For frequency models with `offset="Exposure"`, target encoding automatically uses claim rate (ClaimCount/Exposure) instead of raw counts
- **Interactions**: Use `target_encoding: True` in interactions to encode variable combinations

---

## Expression Terms for Polynomials

```python
# Polynomial terms
result = rs.glm_dict(
    response="y",
    terms={
        "age": {"type": "linear"},
        "age2": {"type": "expression", "expr": "age ** 2"},
        "age3": {"type": "expression", "expr": "age ** 3"},
    },
    data=data, family="gaussian",
).fit()

# Arithmetic expressions
result = rs.glm_dict(
    response="y",
    terms={
        "income_k": {"type": "expression", "expr": "income / 1000"},
        "bmi": {"type": "expression", "expr": "weight / (height ** 2)"},
    },
    data=data, family="gaussian",
).fit()
```

**Supported operations:** `+`, `-`, `*`, `/`, `**` (power)

---

## Design Matrix Validation

```python
# Check for issues before fitting
model = rs.glm_dict(
    response="y",
    terms={"x": {"type": "ns", "df": 4}, "cat": {"type": "categorical"}},
    data=data, family="poisson",
)
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
- **Base Model Comparison**: Compare new model against existing/benchmark predictions

### Comparing Against a Base Model

Compare your new model against predictions from an existing model (e.g., current production model):

```python
# Add base model predictions to your data
data = data.with_columns(pl.lit(old_model_predictions).alias("base_pred"))

# Run diagnostics with base_predictions
diagnostics = result.diagnostics(
    train_data=data,
    categorical_factors=["Region", "VehBrand"],
    continuous_factors=["Age", "VehPower"],
    base_predictions="base_pred",  # Column name with base model predictions
)

# Access comparison results
bc = diagnostics.base_predictions_comparison

# Side-by-side metrics
print(f"Model loss: {bc.model_metrics.loss}, Base loss: {bc.base_metrics.loss}")
print(f"Model Gini: {bc.model_metrics.gini}, Base Gini: {bc.base_metrics.gini}")

# Improvement metrics (positive = new model is better)
print(f"Loss improvement: {bc.loss_improvement_pct}%")
print(f"Gini improvement: {bc.gini_improvement}")
print(f"AUC improvement: {bc.auc_improvement}")

# Decile analysis sorted by model/base prediction ratio
for d in bc.model_vs_base_deciles:
    print(f"Decile {d.decile}: actual={d.actual:.4f}, "
          f"model={d.model_predicted:.4f}, base={d.base_predicted:.4f}")
```

The comparison includes:
- **Side-by-side metrics**: Loss (mean deviance), Gini, AUC, A/E ratio for both models
- **Improvement metrics**: `loss_improvement_pct`, `gini_improvement`, `auc_improvement`
- **Decile analysis**: Data sorted by model/base ratio, showing where the new model diverges
- **Calibration comparison**: Count of deciles where each model has better A/E

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
