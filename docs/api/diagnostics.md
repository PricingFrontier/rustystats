# Diagnostics API Reference

This page documents the post-fit model diagnostics functionality.

## result.diagnostics()

Compute comprehensive model diagnostics.

```python
diagnostics = result.diagnostics(
    train_data,
    categorical_factors=None,
    continuous_factors=None,
    n_calibration_bins=10,
    n_factor_bins=10,
    rare_threshold_pct=1.0,
    max_categorical_levels=20,
    detect_interactions=False,
    max_interaction_factors=10,
    test_data=None,
    compute_vif=True,
    compute_coefficients=True,
    compute_deviance_by_level=True,
    compute_lift=True,
    compute_partial_dep=True,
    compute_robust_se=True,
    compute_score_tests=True,
    base_predictions=None,
)
```

The response and exposure (offset) columns are inferred from the model's
formula, so you do not pass them again. Results are auto-saved to
`analysis/diagnostics.json` as a side effect of the call.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `train_data` | `pl.DataFrame` | required | Training data used for fitting |
| `categorical_factors` | `list[str]` | `None` | Categorical columns to analyze (fitted or not) |
| `continuous_factors` | `list[str]` | `None` | Continuous columns to analyze (fitted or not) |
| `n_calibration_bins` | int | `10` | Bins for calibration curve |
| `n_factor_bins` | int | `10` | Quantile bins for continuous factors |
| `rare_threshold_pct` | float | `1.0` | Pct threshold for grouping levels into "Other" |
| `max_categorical_levels` | int | `20` | Maximum categorical levels to show |
| `detect_interactions` | bool | `False` | Run residual-based interaction detection |
| `max_interaction_factors` | int | `10` | Max factors for interaction search |
| `test_data` | `pl.DataFrame` | `None` | Holdout data for overfitting checks |
| `compute_vif` | bool | `True` | Compute VIF / multicollinearity scores |
| `compute_coefficients` | bool | `True` | Compute coefficient summary |
| `compute_deviance_by_level` | bool | `True` | Deviance breakdown by categorical level |
| `compute_lift` | bool | `True` | Full lift chart with all deciles |
| `compute_partial_dep` | bool | `True` | Partial dependence per variable |
| `compute_robust_se` | bool | `True` | Enrich coefficient summary with HC1 robust SEs |
| `compute_score_tests` | bool | `True` | Rao score tests for unfitted factors |
| `base_predictions` | str | `None` | Column in `train_data` with predictions from another model |

### Returns

A `ModelDiagnostics` object.

---

## ModelDiagnostics

Top-level diagnostics container. Always-present fields are populated from the
fitted model and `train_data`; optional fields are populated according to the
`compute_*` flags and the presence of `test_data` / `base_predictions`.

### Always-present fields

| Field | Type | Description |
|-------|------|-------------|
| `model_summary` | dict | Family, link, formula, n_obs, n_params, df_resid, scale, null deviance, etc. |
| `train_test` | `TrainTestComparison` | Train metrics and (optionally) test comparison |
| `calibration` | dict | Calibration bins / Hosmer-Lemeshow style metrics |
| `residual_summary` | `dict[str, ResidualSummary]` | Mean / std / skew per residual type |
| `factors` | `list[FactorDiagnostics]` | Per-factor A/E, residual pattern, significance, score tests |
| `interaction_candidates` | `list[InteractionCandidate]` | Detected interactions (empty unless `detect_interactions=True`) |
| `model_comparison` | `dict[str, float]` | Aggregate comparison metrics (e.g. AIC, BIC) |
| `warnings` | `list[dict[str, str]]` | Auto-generated alerts (overfitting, drift, overdispersion, ...) |

### Optional fields

| Field | Type | Populated when |
|-------|------|----------------|
| `vif` | `list[VIFResult]` \| `None` | `compute_vif=True` and a design matrix is available |
| `smooth_terms` | `list[SmoothTermDiagnostics]` \| `None` | Model has smooth (penalized spline) terms |
| `coefficient_summary` | `list[CoefficientSummary]` \| `None` | `compute_coefficients=True` |
| `factor_deviance` | `list[FactorDeviance]` \| `None` | `compute_deviance_by_level=True` and at least one categorical |
| `lift_chart` | `LiftChart` \| `None` | `compute_lift=True` |
| `partial_dependence` | `list[PartialDependence]` \| `None` | `compute_partial_dep=True` and factors provided |
| `overdispersion` | `dict` \| `None` | Family is Poisson / Binomial / NegativeBinomial |
| `spline_info` | `dict` \| `None` | Model has spline terms with knot info |
| `base_predictions_comparison` | `BasePredictionsComparison` \| `None` | `base_predictions` column provided |

### model_summary

```python
diagnostics.model_summary
# {
#     'formula': 'ClaimNb ~ Area + VehBrand + bs(DrivAge, df=4)',
#     'family': 'Poisson',
#     'link': 'log',
#     'n_obs': 10000,
#     'n_params': 15,
#     'df_resid': 9985,
#     'converged': True,
#     'iterations': 5,
#     'scale': 1.0,
#     'scale_pearson': 1.0148,
#     'null_deviance': 1408.77,
#     # 'regularization': {...}     # only if alpha > 0
#     # 'robust_se_type': 'HC1'     # only if robust SEs were enriched
# }
```

### train_test

`TrainTestComparison` always contains a `train` `DatasetDiagnostics`. When
`test_data` is supplied, `test` and the comparison fields are filled.

```python
tt = diagnostics.train_test

tt.train.gini       # Gini on the training set
tt.train.auc        # AUC
tt.train.ae_ratio   # Total actual / predicted
tt.train.deviance
tt.train.aic
tt.train.bic
tt.train.ae_by_decile          # list[DecileMetrics]
tt.train.factor_diagnostics    # dict[str, list[FactorLevelMetrics]]
tt.train.continuous_diagnostics  # dict[str, list[ContinuousBandMetrics]]

# Only populated when test_data is provided:
tt.test                # DatasetDiagnostics on holdout
tt.gini_gap            # train.gini - test.gini
tt.ae_ratio_diff       # |train.ae_ratio - test.ae_ratio|
tt.decile_comparison   # list[dict] per decile
tt.factor_divergence   # dict[factor, list[divergent levels]]
tt.overfitting_risk    # True if gini_gap > 0.03
tt.calibration_drift   # True if test ae_ratio outside [0.95, 1.05]
tt.unstable_factors    # list[str] like 'Region[A]'
```

### calibration

```python
diagnostics.calibration
# {
#     'bins': [
#         CalibrationBin(bin_index=0, predicted_lower=..., predicted_upper=...,
#                        predicted_mean=..., actual_mean=..., actual_expected_ratio=...,
#                        count=..., exposure=..., actual_sum=..., predicted_sum=...,
#                        ae_confidence_interval_lower=..., ae_confidence_interval_upper=...),
#         ...
#     ],
#     'overall_ae': 0.998,
#     'hosmer_lemeshow': {'statistic': 8.5, 'df': 8, 'pvalue': 0.38},
#     ...
# }
```

### residual_summary

Maps residual type (e.g. `"pearson"`, `"deviance"`, `"response"`) to a
`ResidualSummary` with `mean`, `std`, `skewness`.

### factors

A list of `FactorDiagnostics`, one per factor name passed in.

```python
for f in diagnostics.factors:
    print(f.name, f.factor_type, f.in_model)
    print(f.residual_pattern.resid_corr)        # correlation with residuals
    print(f.residual_pattern.var_explained)
    if f.significance:                           # Type-III tests, only for fitted
        print(f.significance.chi2, f.significance.p, f.significance.dev_pct)
    if f.score_test:                             # Rao score test, only for unfitted
        print(f.score_test.statistic, f.score_test.pvalue, f.score_test.significant)
    if f.relative_importance is not None:
        print(f.relative_importance, "% of fitted dev contribution")
    for bin_ in f.actual_vs_expected:            # ActualExpectedBin
        print(bin_.bin, bin_.actual, bin_.expected, bin_.ae_ratio, bin_.ae_ci)
```

### lift_chart

```python
lc = diagnostics.lift_chart
lc.gini
lc.ks_statistic
lc.ks_decile
lc.weak_deciles               # deciles where discrimination is poor
for d in lc.deciles:          # LiftDecile
    print(d.decile, d.actual, d.predicted, d.ae_ratio,
          d.lift, d.cumulative_lift)
```

### partial_dependence

```python
for pd in diagnostics.partial_dependence:    # PartialDependence
    print(pd.variable, pd.variable_type, pd.shape)
    print(pd.recommendation)
    # pd.grid_values, pd.predictions, pd.relativities
```

`shape` is one of `"flat"`, `"monotonic"`, `"u_shaped"`, `"inverted_u"`,
`"complex"`. Recommendations such as "consider a spline" trigger a warning
in `diagnostics.warnings`.

### vif

```python
for v in diagnostics.vif:        # VIFResult
    print(v.feature, v.vif, v.severity)   # severity in {none, moderate, severe, expected}
```

### coefficient_summary

```python
for c in diagnostics.coefficient_summary:    # CoefficientSummary
    print(c.feature, c.estimate, c.std_error, c.z_value, c.p_value, c.significant)
    print(c.conf_int, c.relativity, c.relativity_ci)
    # Robust SE fields are populated when compute_robust_se=True
    print(c.robust_std_error, c.robust_z_value, c.robust_p_value, c.robust_significant)
```

Robust SE fields are `None` when `store_design_matrix=False` (lean mode) or
for deserialized models.

### factor_deviance

```python
for fd in diagnostics.factor_deviance:        # FactorDeviance
    print(fd.factor, fd.total_deviance, fd.problem_levels)
    for lvl in fd.levels:                      # DevianceByLevel
        print(lvl.level, lvl.n, lvl.deviance, lvl.deviance_pct,
              lvl.mean_deviance, lvl.ae_ratio, lvl.problem)
```

### overdispersion

Populated for Poisson / Binomial / NegativeBinomial families:

```python
diagnostics.overdispersion
# {
#     'pearson_dispersion': 1.05,
#     'pearson_chi2': 10500.0,
#     'df_resid': 9985,
#     'raw_dispersion': 1.07,
#     'mean_count': 0.05,
#     'var_count': 0.054,
#     'severity': 'none',          # one of: none, mild, moderate, severe
#     'recommendation': 'Poisson assumption appears reasonable',
# }
```

### smooth_terms

Populated when the model contains penalized smooths (`s(...)`):

```python
for st in diagnostics.smooth_terms:           # SmoothTermDiagnostics
    print(st.variable, st.k, st.edf, st.lambda_, st.gcv,
          st.ref_df, st.chi2, st.p_value)
```

### interaction_candidates

Empty unless `detect_interactions=True`:

```python
for ic in diagnostics.interaction_candidates:    # InteractionCandidate
    print(ic.factor1, ic.factor2,
          ic.interaction_strength, ic.pvalue, ic.n_cells,
          ic.recommendation)
```

### warnings

```python
for w in diagnostics.warnings:
    print(f"[{w['type']}] {w['message']}")
```

Common types: `overdispersion`, `overfitting`, `calibration_drift`,
`unstable_factors`, `weak_discrimination`, `nonlinear_effect`,
`problem_factor_levels`, `insignificant_smooth`, `undersmoothed`,
`model_improvement`, `model_regression`.

### Methods

#### to_dict()

Recursively convert to a JSON-friendly dict (rounds floats, drops NaN/Inf).

```python
data = diagnostics.to_dict()
```

#### to_json(indent=None)

Serialize as a JSON string.

```python
json_str = diagnostics.to_json(indent=2)
```

---

## result.diagnostics_json()

Convenience method that calls `diagnostics()` and returns the JSON string
directly.

```python
json_str = result.diagnostics_json(
    train_data=train_data,
    categorical_factors=["Region"],
    continuous_factors=["Age"],
    test_data=test_data,         # optional
    indent=2,                    # optional
)
```

---

## explore_data()

Pre-fit data exploration. No fitted model required.

```python
exploration = rs.explore_data(
    data,
    response,
    categorical_factors=None,
    continuous_factors=None,
    exposure=None,
    family="poisson",
    n_bins=10,
    rare_threshold_pct=1.0,
    max_categorical_levels=20,
    detect_interactions=True,
    max_interaction_factors=10,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | DataFrame | required | Data to explore |
| `response` | str | required | Response column name |
| `categorical_factors` | list | `None` | Categorical columns |
| `continuous_factors` | list | `None` | Continuous columns |
| `exposure` | str | `None` | Exposure column |
| `family` | str | `"poisson"` | Expected family |
| `n_bins` | int | `10` | Bins for continuous factors |
| `rare_threshold_pct` | float | `1.0` | Threshold for rare-level grouping |
| `max_categorical_levels` | int | `20` | Maximum levels to retain |
| `detect_interactions` | bool | `True` | Detect interaction candidates |
| `max_interaction_factors` | int | `10` | Max factors for interaction search |

### Returns

A `DataExploration` object. Results are auto-saved to
`analysis/exploration.json`.

### DataExploration

| Field | Type | Description |
|-------|------|-------------|
| `data_summary` | dict | n_obs and per-column dtype/missing counts |
| `factor_stats` | list[dict] | Univariate stats per factor |
| `missing_values` | dict | Missing-value summary |
| `univariate_tests` | list[dict] | Per-factor univariate significance tests |
| `correlations` | dict | Pairwise correlations among continuous factors |
| `cramers_v` | dict | Pairwise association among categoricals |
| `vif` | list[dict] | Pre-fit VIF estimates |
| `zero_inflation` | dict | Zero-rate analysis of the response |
| `overdispersion` | dict | Variance/mean ratio of the response |
| `interaction_candidates` | `list[InteractionCandidate]` | Detected interactions |
| `response_stats` | dict | Mean/var/zero-pct of the response |

`exploration.to_dict()` and `exploration.to_json(indent=...)` mirror
`ModelDiagnostics`.

---

## Complete Example

```python
import rustystats as rs
import polars as pl

# Load and split
data = pl.read_parquet("insurance.parquet")
train_data, test_data = data.head(8000), data.tail(2000)

# Fit
result = rs.glm_dict(
    response="ClaimNb",
    terms={
        "Area": {"type": "categorical"},
        "VehBrand": {"type": "categorical"},
        "DrivAge": {"type": "bs", "df": 4},
    },
    data=train_data,
    family="poisson",
    offset="Exposure",
).fit()

# Compute diagnostics, including factors not in the model
diagnostics = result.diagnostics(
    train_data=train_data,
    test_data=test_data,
    categorical_factors=["Area", "VehBrand", "Region"],   # Region not in model
    continuous_factors=["DrivAge", "VehPower", "Density"],
)

# Train metrics live on train_test.train
tt = diagnostics.train_test
print(f"Train Gini: {tt.train.gini:.3f}, AUC: {tt.train.auc:.3f}")
print(f"Train A/E:  {tt.train.ae_ratio:.3f}")

if tt.test is not None:
    print(f"Test  Gini: {tt.test.gini:.3f}, A/E: {tt.test.ae_ratio:.3f}")
    if tt.overfitting_risk:
        print(f"Overfitting risk (gini_gap={tt.gini_gap:.3f})")
    if tt.calibration_drift:
        print(f"Calibration drift on test set")

# Score tests for unfitted factors
for f in diagnostics.factors:
    if not f.in_model and f.score_test and f.score_test.significant:
        print(f"Consider adding {f.name} (score p={f.score_test.pvalue:.4f})")

# Multicollinearity
if diagnostics.vif:
    for v in diagnostics.vif:
        if v.severity in ("severe", "moderate"):
            print(f"{v.feature}: VIF={v.vif:.1f} ({v.severity})")

# Auto-generated warnings
for w in diagnostics.warnings:
    print(f"[{w['type']}] {w['message']}")

# Export for downstream consumers
json_str = diagnostics.to_json(indent=2)
```

---

## Comparing Against a Base Model

Compare your new model against predictions from another model (for example,
a current production model) by passing the column name via `base_predictions`.

```python
# Add base model predictions to your data
data = data.with_columns(pl.lit(old_model_predictions).alias("base_pred"))

diagnostics = result.diagnostics(
    train_data=data,
    categorical_factors=["Region", "VehBrand"],
    continuous_factors=["Age", "VehPower"],
    base_predictions="base_pred",
)
```

### BasePredictionsComparison

Access via `diagnostics.base_predictions_comparison`.

| Field | Type | Description |
|-------|------|-------------|
| `model_metrics` | `BasePredictionsMetrics` | Metrics for the new model |
| `base_metrics` | `BasePredictionsMetrics` | Metrics for the base predictions |
| `model_vs_base_deciles` | `list[ModelVsBaseDecile]` | Decile analysis sorted by model/base ratio |
| `model_better_deciles` | int | Count of deciles where the new model has better A/E |
| `base_better_deciles` | int | Count of deciles where the base does |
| `loss_improvement_pct` | float | Pct improvement in mean deviance loss (positive = new better) |
| `gini_improvement` | float | Absolute Gini improvement |
| `auc_improvement` | float | Absolute AUC improvement |

### BasePredictionsMetrics

| Field | Type | Description |
|-------|------|-------------|
| `total_predicted` | float | Sum of predictions |
| `ae_ratio` | float | Total actual / total predicted |
| `loss` | float | Mean deviance loss |
| `gini` | float | Gini coefficient |
| `auc` | float | Area under ROC curve |

### ModelVsBaseDecile

| Field | Type | Description |
|-------|------|-------------|
| `decile` | int | Decile number (1-10) |
| `n` | int | Observation count in decile |
| `exposure` | float | Total exposure in decile |
| `actual` | float | Actual response (rate or count) |
| `model_predicted` | float | New model prediction |
| `base_predicted` | float | Base model prediction |
| `model_ae_ratio` | float | New model A/E in decile |
| `base_ae_ratio` | float | Base model A/E in decile |
| `model_base_ratio_mean` | float | Mean of model/base prediction ratio |

### Example

```python
import rustystats as rs
import polars as pl

data = pl.read_parquet("insurance.parquet")
data = data.with_columns(pl.col("production_model_pred").alias("base_pred"))

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

diagnostics = result.diagnostics(
    train_data=data,
    categorical_factors=["Region"],
    continuous_factors=["Age"],
    base_predictions="base_pred",
)

bc = diagnostics.base_predictions_comparison

print("=== Side-by-side ===")
print(f"Loss:  new={bc.model_metrics.loss:.4f}  base={bc.base_metrics.loss:.4f}")
print(f"Gini:  new={bc.model_metrics.gini:.3f}  base={bc.base_metrics.gini:.3f}")
print(f"A/E:   new={bc.model_metrics.ae_ratio:.3f}  base={bc.base_metrics.ae_ratio:.3f}")

print("=== Improvement ===")
print(f"Loss improvement: {bc.loss_improvement_pct:+.2f}%")
print(f"Gini improvement: {bc.gini_improvement:+.3f}")
print(f"AUC improvement:  {bc.auc_improvement:+.3f}")
print(f"New model wins {bc.model_better_deciles}/10 deciles by A/E")

print("=== Decile analysis (sorted by model/base ratio) ===")
for d in bc.model_vs_base_deciles:
    print(
        f"D{d.decile:2d}: model/base={d.model_base_ratio_mean:.2f}  "
        f"model_ae={d.model_ae_ratio:.2f}  base_ae={d.base_ae_ratio:.2f}"
    )
```

---

## Calibration Primitives (RS-ACT-009)

Stand-alone calibration primitives sit alongside the model — they never silently
fold into the GLM coefficients. Use them to *measure* calibration (`A/E` overall,
by bin, by factor) and to *fit* an explicit calibration object (global or
isotonic) that callers apply on top of `result.predict()`.

!!! warning "In-sample optimism"
    Fitting calibration on the same rows used to fit the model overstates
    calibration quality. Prefer a held-out calibration fold, out-of-fold
    predictions, or untouched holdout data.

### rs.calibration_summary

Standalone array-level summary. Returns overall, per-bin and (optional)
per-factor aggregates.

```python
summary = rs.calibration_summary(
    y,
    pred,
    exposure=None,     # bins are rate-ranked when exposure is present
    weights=None,      # Σ(w·y) / Σ(w·μ) when supplied
    by=None,           # mapping {factor_name: array} or None
    n_bins=10,
    ranking="auto",    # "auto" | "mean" | "rate"
    min_exposure=0.0,  # per-factor cells below this are flagged suppressed
)
```

`summary["overall"]` carries `actual`, `expected`, `ae_ratio`, `n_obs`,
`total_weight`. `summary["bins"]` is a list of per-bin dicts with
`bin_index`, `count`, `actual`, `expected`, `ae_ratio`, `exposure`,
`predicted_rate_min/max/mean`, `actual_rate`, `expected_rate`.
`summary["by_factor"]` (when `by=` is given) groups the same per-bin shape by
factor level with a boolean `suppressed` flag.

### Result-bound wrappers

`GLMModel.calibration_summary(data, ...)` resolves response / exposure / weights
through the fitted model and delegates to `rs.calibration_summary`.
`GLMModel.fit_calibration(data, method=...)` returns a `GlobalCalibration` (for
`method="global"`) or an `IsotonicCalibration` (for `method="isotonic"`):

```python
result.calibration_summary(holdout, by="Region", min_exposure=10.0)

global_cal = result.fit_calibration(holdout, method="global")
iso_cal    = result.fit_calibration(holdout, method="isotonic")

calibrated = global_cal.predict(result.predict(new_data))
```

### rs.GlobalCalibration

Scalar multiplicative map `y_hat = factor * pred`, with
`factor = Σ(w·y) / Σ(w·μ)`.

```python
cal = rs.fit_global_calibration(y, pred, weights=None)
cal.factor                       # the multiplicative factor
cal.predict(np.array([...]))     # = factor * pred
state = cal.to_dict()
cal2  = rs.GlobalCalibration.from_dict(state)
```

### rs.IsotonicCalibration

Monotone Pool-Adjacent-Violators (PAV) map with optional per-observation
weights. Predictions outside the fitted threshold range are clamped to the
nearest knot (`numpy.interp` semantics).

```python
iso = rs.fit_isotonic_calibration(y, pred, weights=None, increasing=True)
iso.thresholds_   # ascending pred-space knot positions
iso.values_       # monotone calibrated values at those knots
iso.predict(np.linspace(...))
iso.to_dict()
```

Isotonic calibration is **opt-in only** and **serialised separately** from the
GLM. Use it explicitly on top of `result.predict()`; it is never applied
implicitly inside `predict()`.

### result.relevel — log-link intercept shift

For log-link GLMs, a global calibration can be applied as an intercept shift —
preserving every relativity exactly:

```python
releveled = result.relevel(holdout, weights=None, inplace=False)
```

Computes `c = Σ(w·y) / Σ(w·μ)` on the calibration data and updates the
intercept by `+log(c)`. The non-intercept coefficients `β[1:]` are
**bit-identical**, so every `exp(β_j)` relativity is unchanged — only the
model's overall level shifts. Returns a new `GLMModel` by default; pass
`inplace=True` to mutate the current object.

`releveled.intercept_delta` and `releveled.relevel_history` expose the
accumulated shift and per-call provenance (original/new intercept, factor,
log shift, n_obs, total_weight). Relevel state round-trips through
`to_bytes`/`from_bytes`.

For non-log links `relevel()` raises `ValidationError`; attach a
`GlobalCalibration` via `fit_calibration(method="global")` instead.
