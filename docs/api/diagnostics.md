# Diagnostics API Reference

This page documents the model diagnostics functionality.

## result.diagnostics()

Compute comprehensive model diagnostics.

```python
diagnostics = result.diagnostics(
    data,
    categorical_factors=None,
    continuous_factors=None,
    exposure=None,
    n_bins=10,
    detect_interactions=True,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | DataFrame | required | Original data used for fitting |
| `categorical_factors` | list | `None` | Categorical columns to analyze |
| `continuous_factors` | list | `None` | Continuous columns to analyze |
| `exposure` | str | `None` | Exposure column name |
| `n_bins` | int | `10` | Number of bins for calibration |
| `detect_interactions` | bool | `True` | Whether to detect interactions |

### Returns

`ModelDiagnostics` object.

---

## ModelDiagnostics

### Attributes

#### model_summary

Basic model information.

```python
diagnostics.model_summary
# {
#     'family': 'poisson',
#     'link': 'log',
#     'n_observations': 10000,
#     'n_parameters': 15,
#     'converged': True,
#     'iterations': 5
# }
```

#### fit_statistics

Goodness-of-fit metrics.

```python
diagnostics.fit_statistics
# {
#     'deviance': 12345.67,
#     'null_deviance': 15000.00,
#     'aic': 12375.67,
#     'bic': 12450.00,
#     'log_likelihood': -6172.84,
#     'dispersion_pearson': 1.05,
#     'dispersion_deviance': 1.03,
#     'pseudo_r2': 0.177
# }
```

#### calibration

Calibration metrics.

```python
diagnostics.calibration
# {
#     'overall_ae': 0.998,
#     'by_decile': [
#         {'decile': 1, 'actual': 100, 'expected': 95, 'ae_ratio': 1.053, ...},
#         {'decile': 2, 'actual': 150, 'expected': 148, 'ae_ratio': 1.014, ...},
#         ...
#     ],
#     'hosmer_lemeshow': {'statistic': 8.5, 'df': 8, 'pvalue': 0.38}
# }
```

#### discrimination

Discrimination metrics.

```python
diagnostics.discrimination
# {
#     'gini_coefficient': 0.42,
#     'auc': 0.71,
#     'ks_statistic': 0.35,
#     'lorenz_curve': [(0.0, 0.0), (0.1, 0.05), ..., (1.0, 1.0)],
#     'lift_top_decile': 2.5
# }
```

#### factors

Per-factor diagnostics (list of `FactorDiagnostic`).

```python
for factor in diagnostics.factors:
    print(f"{factor.name}:")
    print(f"  Type: {factor.factor_type}")
    print(f"  In model: {factor.in_model}")
    print(f"  A/E range: {factor.ae_range}")
    print(f"  Residual correlation: {factor.residual_correlation}")
```

#### interaction_candidates

Detected potential interactions.

```python
for ic in diagnostics.interaction_candidates:
    print(f"{ic['factor1']} × {ic['factor2']}: strength={ic['strength']:.3f}")
```

#### warnings

Auto-generated warnings.

```python
for warning in diagnostics.warnings:
    print(f"[{warning['type']}] {warning['message']}")
```

### Methods

#### to_json()

Export diagnostics as JSON string.

```python
json_str = diagnostics.to_json()
```

#### to_dict()

Export as Python dictionary.

```python
data = diagnostics.to_dict()
```

---

## result.diagnostics_json()

Convenience method to get JSON directly.

```python
json_str = result.diagnostics_json(
    data=data,
    categorical_factors=["region"],
    continuous_factors=["age"],
)
```

---

## explore_data()

Pre-fit data exploration (no model required).

```python
exploration = rs.explore_data(
    data,
    response,
    categorical_factors=None,
    continuous_factors=None,
    exposure=None,
    family="poisson",
    detect_interactions=True,
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | DataFrame | Data to explore |
| `response` | str | Response column name |
| `categorical_factors` | list | Categorical columns |
| `continuous_factors` | list | Continuous columns |
| `exposure` | str | Exposure column |
| `family` | str | Expected family (for rate calculation) |
| `detect_interactions` | bool | Whether to detect interactions |

### Returns

`DataExploration` object.

### DataExploration

```python
exploration.response_stats
# {
#     'n_observations': 10000,
#     'mean': 0.05,
#     'std': 0.22,
#     'min': 0,
#     'max': 5,
#     'zeros_pct': 95.2,
#     'total_exposure': 9500.0
# }

exploration.factor_stats
# [
#     {'name': 'region', 'type': 'categorical', 'n_levels': 5, ...},
#     {'name': 'age', 'type': 'continuous', 'mean': 42.3, 'std': 15.2, ...},
# ]

exploration.interaction_candidates
# [{'factor1': 'age', 'factor2': 'region', 'strength': 0.08}, ...]

exploration.to_json()  # Export as JSON
```

### Example

```python
import rustystats as rs
import polars as pl

data = pl.read_parquet("insurance.parquet")

# Explore before fitting
exploration = rs.explore_data(
    data=data,
    response="ClaimNb",
    categorical_factors=["Region", "VehBrand", "Area"],
    continuous_factors=["DrivAge", "VehPower", "Density"],
    exposure="Exposure",
    family="poisson",
    detect_interactions=True,
)

print("Response distribution:")
print(exploration.response_stats)

print("\nFactor summary:")
for f in exploration.factor_stats:
    print(f"  {f['name']}: {f['type']}")

print("\nSuggested interactions:")
for ic in exploration.interaction_candidates[:5]:
    print(f"  {ic['factor1']} × {ic['factor2']} (strength: {ic['strength']:.3f})")
```

---

## FactorDiagnostic

Per-factor diagnostics object.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | str | Factor name |
| `factor_type` | str | `"categorical"` or `"continuous"` |
| `in_model` | bool | Whether factor is in the model |
| `actual_vs_expected` | list | A/E by level/bin |
| `residual_pattern` | dict | Residual analysis |
| `ae_range` | tuple | (min A/E, max A/E) |
| `residual_correlation` | float | Correlation with residuals |

### actual_vs_expected

For categorical factors:
```python
[
    {'level': 'A', 'exposure': 1000, 'actual': 50, 'expected': 48, 'ae_ratio': 1.04},
    {'level': 'B', 'exposure': 1500, 'actual': 70, 'expected': 75, 'ae_ratio': 0.93},
    ...
]
```

For continuous factors:
```python
[
    {'bin': 1, 'range': (18, 25), 'exposure': 500, 'actual': 30, 'expected': 25, 'ae_ratio': 1.20},
    {'bin': 2, 'range': (25, 35), 'exposure': 800, 'actual': 35, 'expected': 38, 'ae_ratio': 0.92},
    ...
]
```

---

## JSON Structure

The JSON export is optimized for LLM consumption:

```json
{
  "model_summary": {
    "family": "poisson",
    "link": "log",
    "n_observations": 10000,
    "n_parameters": 15
  },
  "fit_statistics": {
    "deviance": 12345.67,
    "aic": 12375.67,
    "dispersion_pearson": 1.05
  },
  "calibration": {
    "overall_ae": 0.998,
    "by_decile": [...]
  },
  "discrimination": {
    "gini_coefficient": 0.42,
    "auc": 0.71
  },
  "factors": [
    {
      "name": "Region",
      "factor_type": "categorical",
      "in_model": true,
      "actual_vs_expected": [...],
      "residual_pattern": {"correlation": 0.01}
    }
  ],
  "interaction_candidates": [
    {"factor1": "Age", "factor2": "Region", "strength": 0.03}
  ],
  "warnings": [
    {"type": "overdispersion", "message": "Dispersion ratio is 1.5..."}
  ]
}
```

---

## Complete Example

```python
import rustystats as rs
import polars as pl

# Load and fit
data = pl.read_parquet("insurance.parquet")
result = rs.glm(
    "ClaimNb ~ C(Area) + C(VehBrand) + bs(DrivAge, df=4)",
    data=data,
    family="poisson",
    offset="Exposure"
).fit()

# Compute diagnostics (including non-fitted factors)
diagnostics = result.diagnostics(
    data=data,
    categorical_factors=["Area", "VehBrand", "Region"],  # Region not in model
    continuous_factors=["DrivAge", "VehPower", "Density"],  # VehPower, Density not in model
)

# Check overall calibration
ae = diagnostics.calibration['overall_ae']
print(f"Overall A/E: {ae:.3f}")
if abs(ae - 1.0) > 0.02:
    print("  ⚠️ Model may be miscalibrated")

# Check discrimination
gini = diagnostics.discrimination['gini_coefficient']
print(f"Gini: {gini:.3f}")

# Check for missing factors
for factor in diagnostics.factors:
    if not factor.in_model and abs(factor.residual_correlation) > 0.03:
        print(f"  ⚠️ Consider adding {factor.name} (residual corr: {factor.residual_correlation:.3f})")

# Check for interactions
for ic in diagnostics.interaction_candidates[:3]:
    print(f"  Consider: {ic['factor1']} × {ic['factor2']}")

# View warnings
for w in diagnostics.warnings:
    print(f"  ⚠️ [{w['type']}] {w['message']}")

# Export for LLM analysis
json_output = diagnostics.to_json()
```

---

## Base Model Comparison

Compare your new model against predictions from an existing model (e.g., current production model).

### Usage

```python
# Add base model predictions to your data
data = data.with_columns(pl.lit(old_model_predictions).alias("base_pred"))

# Run diagnostics with base_predictions
diagnostics = result.diagnostics(
    data=data,
    categorical_factors=["Region", "VehBrand"],
    continuous_factors=["Age", "VehPower"],
    base_predictions="base_pred",  # Column name with base model predictions
)
```

### BaseModelComparison

Access via `diagnostics.base_predictions_comparison`:

```python
bc = diagnostics.base_predictions_comparison

# Side-by-side metrics
print(f"Model loss: {bc.model_metrics.loss}")
print(f"Base loss: {bc.base_metrics.loss}")
print(f"Model Gini: {bc.model_metrics.gini}")
print(f"Base Gini: {bc.base_metrics.gini}")

# Improvement metrics (positive = new model is better)
print(f"Loss improvement: {bc.loss_improvement_pct}%")
print(f"Gini improvement: {bc.gini_improvement}")
print(f"AUC improvement: {bc.auc_improvement}")
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `model_metrics` | `ModelMetrics` | Metrics for new model |
| `base_metrics` | `ModelMetrics` | Metrics for base model |
| `loss_improvement_pct` | float | % improvement in loss (positive = better) |
| `gini_improvement` | float | Absolute Gini improvement |
| `auc_improvement` | float | Absolute AUC improvement |
| `model_vs_base_deciles` | list | Decile analysis by model/base ratio |

### ModelMetrics

| Attribute | Type | Description |
|-----------|------|-------------|
| `loss` | float | Mean deviance loss |
| `gini` | float | Gini coefficient |
| `auc` | float | Area under ROC curve |
| `ae_ratio` | float | Actual/Expected ratio |

### Decile Analysis

Data sorted by model/base prediction ratio, showing where the new model diverges:

```python
for d in bc.model_vs_base_deciles:
    print(f"Decile {d.decile}: actual={d.actual:.4f}, "
          f"model={d.model_predicted:.4f}, base={d.base_predicted:.4f}")
```

| Attribute | Type | Description |
|-----------|------|-------------|
| `decile` | int | Decile number (1-10) |
| `actual` | float | Actual response rate |
| `model_predicted` | float | New model prediction |
| `base_predicted` | float | Base model prediction |
| `exposure` | float | Total exposure in decile |
| `model_ae` | float | New model A/E |
| `base_ae` | float | Base model A/E |

### Complete Example

```python
import rustystats as rs
import polars as pl

# Load data with production model predictions
data = pl.read_parquet("insurance.parquet")
data = data.with_columns(
    pl.col("production_model_pred").alias("base_pred")
)

# Fit new model
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

# Compare against production model
diagnostics = result.diagnostics(
    data=data,
    categorical_factors=["Region"],
    continuous_factors=["Age"],
    base_predictions="base_pred",
)

bc = diagnostics.base_predictions_comparison

print("=== Model Comparison ===")
print(f"Loss:  New={bc.model_metrics.loss:.4f}, Base={bc.base_metrics.loss:.4f}")
print(f"       Improvement: {bc.loss_improvement_pct:+.2f}%")
print(f"Gini:  New={bc.model_metrics.gini:.3f}, Base={bc.base_metrics.gini:.3f}")
print(f"       Improvement: {bc.gini_improvement:+.3f}")
print(f"A/E:   New={bc.model_metrics.ae_ratio:.3f}, Base={bc.base_metrics.ae_ratio:.3f}")

# Where does new model differ most?
print("\n=== Decile Analysis (sorted by model/base ratio) ===")
for d in bc.model_vs_base_deciles:
    ratio = d.model_predicted / d.base_predicted if d.base_predicted > 0 else float('inf')
    better = "✓" if abs(d.model_ae - 1) < abs(d.base_ae - 1) else ""
    print(f"D{d.decile:2d}: ratio={ratio:.2f}, model_ae={d.model_ae:.2f}, base_ae={d.base_ae:.2f} {better}")
```
