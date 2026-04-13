# RustyStats 🦀📊

[![CI](https://github.com/PricingFrontier/rustystats/actions/workflows/ci.yml/badge.svg)](https://github.com/PricingFrontier/rustystats/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/rustystats)](https://pypi.org/project/rustystats/)
[![Rust](https://img.shields.io/badge/rust-%E2%9C%94-orange?logo=rust)](https://www.rust-lang.org/)
[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue)](LICENSE)

**High-performance Generalized Linear Models with a Rust backend and Python API**

**Codebase Documentation**: [pricingfrontier.github.io/rustystats/](https://pricingfrontier.github.io/rustystats/)

## Features

- **Dict-First API** - Programmatic model building ideal for automated workflows and agents
- **Fast** - Parallel Rust backend for high-throughput fitting
- **Memory Efficient** - Low memory footprint at scale
- **Stable** - Step-halving IRLS, warm starts for robust convergence
- **Splines** - B-splines and natural splines with auto-tuned smoothing and monotonicity
- **Target Encoding** - Ordered target encoding for high-cardinality categoricals
- **Regularisation** - Ridge, Lasso, and Elastic Net via coordinate descent
- **Lasso Credibility** - Shrink toward a prior model instead of zero (CAS Monograph 13)
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
| `bs` | `df` or `k`, `knots`, `boundary_knots`, `degree=3`, `monotonicity` | B-spline (default: penalized smooth, k=10) |
| `ns` | `df` or `k`, `knots`, `boundary_knots` | Natural spline (default: penalized smooth, k=10) |
| `target_encoding` | `prior_weight=1` | Regularized target encoding |
| `expression` | `expr`, `monotonicity` (optional) | Arbitrary expression (like `I()`) |

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

| Flag | Effect |
|------|--------|
| (none) | Standard product terms (cat×cat, cat×cont, etc.) |
| `target_encoding: True` | Combined TE encoding: `TE(var1:var2)` |
| `frequency_encoding: True` | Combined FE encoding: `FE(var1:var2)` |

---

## Splines

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
```

**Spline parameters:**
- No parameters → penalized smooth with automatic tuning (k=10)
- `df=5` → fixed 5 degrees of freedom
- `k=15` → penalized smooth with 15 basis functions
- `knots=[2.0, 5.0, 8.0]` → explicit interior knot positions (mutually exclusive with `df`/`k`)
- `boundary_knots=(0.0, 10.0)` → custom boundary knots (optional, defaults to data range)
- `monotonicity="increasing"` or `"decreasing"` → constrained effect (bs only)

**When to use each type:**
- **B-splines (`bs`)**: Standard choice, more flexible at boundaries, supports monotonicity
- **Natural splines (`ns`)**: Better extrapolation, linear beyond boundaries

### Monotonic Splines

Constrain the fitted curve to be monotonically increasing or decreasing. Essential when business logic dictates a monotonic relationship.

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
```

---

## Coefficient Constraints

Constrain coefficient signs using `monotonicity` on linear and expression terms.

```python
result = rs.glm_dict(
    response="y",
    terms={
        "age": {"type": "linear", "monotonicity": "increasing"},  # β ≥ 0
        "age2": {"type": "expression", "expr": "age ** 2", "monotonicity": "decreasing"},  # β ≤ 0
        "income": {"type": "linear"},
    },
    data=data, family="poisson",
).fit()
```

| Constraint | Term Spec | Effect |
|------------|-----------|--------|
| β ≥ 0 | `"monotonicity": "increasing"` | Positive effect |
| β ≤ 0 | `"monotonicity": "decreasing"` | Negative effect |

---

## Target Encoding

Ordered target encoding for high-cardinality categoricals.

```python
# Dict API
result = rs.glm_dict(
    response="ClaimNb",
    terms={
        "Brand": {"type": "target_encoding"},
        "Model": {"type": "target_encoding", "prior_weight": 2.0},
        "Age": {"type": "linear"},
        "Region": {"type": "categorical"},
    },
    data=data, family="poisson", offset="Exposure",
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
- **Exposure-aware**: For frequency models with `offset="Exposure"`, automatically uses claim rate (ClaimCount/Exposure) instead of raw counts
- **Interactions**: Use `target_encoding: True` in interactions to encode variable combinations

---

## Expression Terms

```python
result = rs.glm_dict(
    response="y",
    terms={
        "age": {"type": "linear"},
        "age2": {"type": "expression", "expr": "age ** 2"},
        "age3": {"type": "expression", "expr": "age ** 3"},
        "income_k": {"type": "expression", "expr": "income / 1000"},
        "bmi": {"type": "expression", "expr": "weight / (height ** 2)"},
    },
    data=data, family="gaussian",
).fit()
```

**Supported operations:** `+`, `-`, `*`, `/`, `**` (power)

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

### Lasso Credibility

Shrink model coefficients toward a prior model (complement of credibility) instead of toward zero. Based on the methodology in CAS Monograph 13 (Holmes & Casotto, 2025).

When lasso zeroes a coefficient, the prediction for that term falls back to the complement rather than vanishing — making regularized models directly usable as rating plans.

```python
# 1. Fit a countrywide (prior) model
cw_result = rs.glm_dict(
    response="ClaimCount",
    terms={"VehAge": {"type": "bs"}, "DrivAge": {"type": "bs"}},
    data=countrywide_data,
    family="poisson",
    offset="Exposure",
).fit()

# 2. Fit a state model with lasso, shrinking toward countrywide rates
state_result = rs.glm_dict(
    response="ClaimCount",
    terms={
        "VehAge": {"type": "bs"},
        "DrivAge": {"type": "bs"},
        "Region": {"type": "categorical"},
    },
    data=state_data,
    family="poisson",
    offset="Exposure",
    complement="countrywide_rate",  # Column with prior rates (response scale)
).fit(regularization="lasso")

# 3. Inspect which terms the data supports vs. trusts the complement
print(state_result.summary())            # Shows "Lasso Credibility Results"
print(state_result.credibility_summary())  # Deviation from complement per term
```

**Complement sources:**
- `str` — column name in the DataFrame (rates for log-link, probabilities for logit)
- `np.ndarray` — array of prior values on the response scale
- `GLMModel` — fitted model; predictions are computed automatically

Works with all families/links, splines, categoricals, interactions, and target encoding.

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

### Diagnostics JSON Shape

The `diagnostics.to_json()` output includes:

```json
{
  "model_summary": {
    "formula": "...", "family": "poisson", "link": "log",
    "n_obs": 2000, "n_params": 6, "df_resid": 1994,
    "converged": true, "iterations": 5,
    "scale": 1.0,
    "scale_pearson": 1.0148,
    "null_deviance": 1408.77,
    "robust_se_type": "HC1"
  },
  "train_test": {
    "train": {
      "n_obs": 2000, "deviance": 2118.42,
      "log_likelihood": -1059.21,
      "aic": 2130.42, "bic": 2162.70,
      "gini": 0.3241, "auc": 0.6621, "ae_ratio": 1.0
    }
  },
  "coefficient_summary": [
    {
      "feature": "Age", "estimate": 0.00996,
      "std_error": 0.00339, "z_value": 2.941,
      "p_value": 0.0033, "significant": true,
      "conf_int": [0.003318, 0.01659],
      "relativity": 1.01, "relativity_ci": [1.0033, 1.0167],
      "robust_std_error": 0.003378, "robust_z_value": 2.947,
      "robust_p_value": 0.0032, "robust_significant": true
    }
  ]
}
```

- **BIC** (`train_test.train.bic`): Bayesian Information Criterion alongside AIC
- **Scale** (`model_summary.scale`): Deviance-based dispersion parameter
- **Scale Pearson** (`model_summary.scale_pearson`): Pearson-based dispersion estimate
- **Null Deviance** (`model_summary.null_deviance`): Intercept-only model deviance
- **Confidence Intervals** (`coefficient_summary[].conf_int`): 95% CI `[lower, upper]`
- **Robust SEs** (`coefficient_summary[].robust_*`): HC1 sandwich estimators for each coefficient
- **Robust SE Type** (`model_summary.robust_se_type`): Present only when robust SEs were computed

Robust SE fields are `null` when `store_design_matrix=False` (lean mode) or for deserialized models.

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

## Model Serialization

Save and load fitted models for later use:

```python
# Fit and save
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
- Complement of credibility specification

**Compact storage:** Only prediction-essential state is stored (~KB, not MB).

---

## Model Export (PMML & ONNX)

Export fitted models to standard formats for deployment — **no extra dependencies required**. PMML uses stdlib XML; ONNX protobuf serialization is implemented from scratch in Rust.

### PMML

```python
# Export to PMML 4.4 XML
pmml_xml = result.to_pmml()
result.to_pmml(path="model.pmml")

# Load & predict (consumer side)
# uv add pypmml
from pypmml import Model
pmml_model = Model.fromFile("model.pmml")

new_data = pl.DataFrame({"VehAge": [3, 5, 1], "Area": ["C", "A", "B"]})
preds = pmml_model.predict(new_data.to_dict(as_series=False))
```

### ONNX

```python
# Export — "scoring" requires pre-built design matrix, "full" embeds preprocessing
result.to_onnx(path="model.onnx", mode="scoring")
result.to_onnx(path="model_full.onnx", mode="full")

# Predict (consumer side)
# uv add onnxruntime
import onnxruntime as ort
session = ort.InferenceSession("model_full.onnx")
preds = session.run(None, {"input": raw_features})[0]
```
| Size | Smaller | Larger |

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

[AGPL-3.0](LICENSE)
