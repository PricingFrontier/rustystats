# Diagnostics API Reference

This page documents the model diagnostics functionality.

## result.diagnostics()

Compute comprehensive model diagnostics with optional train/test comparison.

```python
diagnostics = result.diagnostics(
    train_data,
    test_data=None,
    categorical_factors=None,
    continuous_factors=None,
    n_calibration_bins=10,
    detect_interactions=True,
    compute_vif=True,
    compute_coefficients=True,
    compute_deviance_by_level=True,
    compute_lift=True,
    compute_partial_dep=True,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `train_data` | DataFrame | required | Training data used for fitting |
| `test_data` | DataFrame | `None` | Test data for overfitting detection |
| `categorical_factors` | list | `None` | Categorical columns to analyze |
| `continuous_factors` | list | `None` | Continuous columns to analyze |
| `n_calibration_bins` | int | `10` | Number of bins for calibration |
| `detect_interactions` | bool | `True` | Whether to detect interactions |
| `compute_vif` | bool | `True` | Compute VIF/multicollinearity |
| `compute_coefficients` | bool | `True` | Compute coefficient summary |
| `compute_deviance_by_level` | bool | `True` | Compute deviance breakdown |
| `compute_lift` | bool | `True` | Compute lift chart |
| `compute_partial_dep` | bool | `True` | Compute partial dependence |

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

#### vif

VIF/multicollinearity scores (list of `VIFResult`).

```python
for v in diagnostics.vif:
    print(f"{v.feature}: VIF={v.vif:.1f} ({v.severity})")
    if v.collinear_with:
        print(f"  Collinear with: {v.collinear_with}")

# Example output:
# bs(VehPower, df=4)[0]: VIF=12.3 (severe)
#   Collinear with: ['VehPower', 'bs(VehPower, df=4)[1]']
```

#### coefficient_summary

Coefficient interpretations (list of `CoefficientSummary`).

```python
for c in diagnostics.coefficient_summary:
    print(f"{c.feature}: relativity={c.relativity:.3f}")
    print(f"  Magnitude: {c.magnitude}, Direction: {c.direction}")
    print(f"  Recommendation: {c.recommendation}")
```

#### factor_deviance

Deviance breakdown by factor level (list of `FactorDeviance`).

```python
for fd in diagnostics.factor_deviance:
    print(f"{fd.factor}: total_deviance={fd.total_deviance:.0f}")
    if fd.problem_levels:
        print(f"  Problem levels: {fd.problem_levels}")
    for level in fd.levels[:5]:  # Top 5 by deviance
        print(f"    {level.level}: {level.deviance_pct:.1f}% (A/E={level.ae_ratio})")
```

#### lift_chart

Full lift chart with decile analysis.

```python
lc = diagnostics.lift_chart
print(f"Top decile lift: {lc.top_decile_lift:.2f}x")
print(f"Weak deciles: {lc.weak_deciles}")  # Deciles with lift < 1.1
for d in lc.deciles:
    print(f"  Decile {d.decile}: lift={d.lift:.2f}, A/E={d.ae_ratio:.3f}")
```

#### partial_dependence

Partial dependence plots (list of `PartialDependence`).

```python
for pd in diagnostics.partial_dependence:
    print(f"{pd.variable} ({pd.variable_type}):")
    print(f"  Shape: {pd.shape}")
    print(f"  Recommendation: {pd.recommendation}")
    if pd.relativities:
        print(f"  Relativities: {pd.relativities[:5]}...")
```

#### train_test

Train/test comparison with overfitting flags.

```python
if diagnostics.train_test:
    tt = diagnostics.train_test
    print(f"Train Gini: {tt.train.gini:.3f}, Test Gini: {tt.test.gini:.3f}")
    print(f"Gini gap: {tt.gini_gap:.3f}")
    
    # Overfitting flags
    if tt.overfitting_risk:
        print("⚠️ Overfitting detected (Gini gap > 0.03)")
    if tt.calibration_drift:
        print(f"⚠️ Calibration drift (Test A/E={tt.test.ae_ratio:.3f})")
    if tt.unstable_factors:
        print(f"⚠️ Unstable factors: {tt.unstable_factors}")
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

The JSON export is optimized for LLM consumption (~70KB for typical models):

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
  "factors": [...],
  "interaction_candidates": [...],
  "warnings": [...],
  
  "vif": [
    {"feature": "VehPower", "vif": 2.3, "severity": "none", "collinear_with": null},
    {"feature": "bs(Age)[0]", "vif": 15.2, "severity": "severe", "collinear_with": ["Age"]}
  ],
  "coefficient_summary": [
    {"feature": "C(Region)[A]", "coefficient": 0.15, "relativity": 1.16, 
     "magnitude": "moderate", "direction": "positive", "recommendation": "Keep"}
  ],
  "factor_deviance": [
    {"factor": "Region", "total_deviance": 12345.0, 
     "levels": [...], "problem_levels": ["Region_X"]}
  ],
  "lift_chart": {
    "deciles": [...], "top_decile_lift": 2.5, "weak_deciles": [5, 6]
  },
  "partial_dependence": [
    {"variable": "Age", "variable_type": "continuous", "shape": "increasing",
     "grid_values": [...], "predictions": [...], "recommendation": "Linear OK"}
  ],
  "train_test": {
    "train": {"dataset": "train", "gini": 0.42, "ae_ratio": 1.00, ...},
    "test": {"dataset": "test", "gini": 0.38, "ae_ratio": 1.03, ...},
    "gini_gap": 0.04,
    "overfitting_risk": true,
    "calibration_drift": false,
    "unstable_factors": ["Region_X"]
  }
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
