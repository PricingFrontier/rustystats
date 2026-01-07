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
