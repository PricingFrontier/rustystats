# Diagnostics: Percentage Deviance Fields

Two new percentage fields for the `factors[]` array in diagnostics output. Both express deviance contributions as a percentage of the **current model's train deviance**, putting fitted and unfitted factors on the same interpretable scale.

---

## 1. `dev_pct` — Actual % Deviance Reduction (fitted factors)

### Location
`diagnostics.factors[].significance.dev_pct`

Sits alongside the existing `dev_contrib` field on each fitted factor's `significance` object.

### Calculation

```
dev_pct = (dev_contrib / model_deviance) * 100
```

Where:
- **`dev_contrib`** is the Type III deviance contribution already computed for the factor (i.e., the increase in deviance when this factor is dropped from the full model)
- **`model_deviance`** is the current (full) model's train deviance

### Example

Given a model with train deviance = 131,691:
- Factor A has `dev_contrib = 4,520` → `dev_pct = 3.43%`
- Factor B has `dev_contrib = 860` → `dev_pct = 0.65%`

### Interpretation

"Removing this factor would increase the model deviance by X%." A factor with `dev_pct = 3.43%` accounts for 3.43% of the current model's total deviance — directly comparable to an unfitted factor's `expected_dev_pct`.

### Notes
- Only present for fitted factors (where `significance` exists)
- `dev_contrib` is already available; this just normalises it by model deviance
- The model deviance used should be the **train** deviance (same split used for fitting)

---

## 2. `expected_dev_pct` — Expected % Deviance Improvement (unfitted factors)

### Location
`diagnostics.factors[].score_test.expected_dev_pct`

Sits alongside the existing `statistic`, `df`, and `pvalue` fields on each unfitted factor's `score_test` object.

### Calculation

```
expected_dev_pct = (statistic / model_deviance) * 100
```

Where:
- **`statistic`** is the Rao score test χ² statistic for the factor (already computed)
- **`model_deviance`** is the current model's train deviance

### Rationale

The score test χ² statistic approximates the likelihood ratio test statistic, which in turn approximates the deviance reduction that would result from adding the factor. Dividing by the current model deviance converts this to a percentage on the same scale as `dev_pct` for fitted factors.

### Example

Given a model with train deviance = 131,691:
- Unfitted factor X has `statistic = 520.3` → `expected_dev_pct = 0.40%`
- Unfitted factor Y has `statistic = 12.1` → `expected_dev_pct = 0.01%`

### Interpretation

"Adding this factor is expected to reduce model deviance by ~X%." This is an approximation (the score test is evaluated at the current model's parameters without actually fitting the new factor), so the frontend displays it with a `~` prefix to indicate it's an estimate.

### Notes
- Only present for unfitted factors (where `score_test` exists)
- The approximation is best when the true deviance reduction is small relative to total deviance, which is typically the case for individual factors
- Uses the same `model_deviance` denominator as `dev_pct`, ensuring the two metrics are directly comparable

---

## Summary

| Field | Object | Formula | Present when |
|---|---|---|---|
| `dev_pct` | `factors[].significance` | `dev_contrib / model_deviance * 100` | Factor is fitted |
| `expected_dev_pct` | `factors[].score_test` | `statistic / model_deviance * 100` | Factor is unfitted |

Both use the **current model's train deviance** as the denominator, so a fitted factor showing `3.4%` and an unfitted factor showing `~0.4%` are directly comparable — the user can see at a glance that BonusMalus contributes 3.4% and adding VehBrand would contribute roughly another 0.4%.
