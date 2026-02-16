# Data Quality Validation in RustyStats

This document describes the data validation checks performed by rustystats before GLM fitting. Use this to inform UI validation, error handling, and user messaging.

---

## Overview

RustyStats validates all inputs before passing data to the Rust fitting engine. Validation fails fast with actionable error messages, preventing cryptic numerical errors downstream.

**Entry point:** `rustystats.validate_glm_inputs(y, X, family, weights, offset, feature_names, is_exposure_offset)`

---

## 1. Array Coercion

All arrays are coerced to `float64` before fitting. This handles:

| Input Type | Behavior |
|------------|----------|
| `int`, `float` | Direct conversion |
| `Decimal` (from Polars/databases) | Converted to float64 |
| `None` / `null` | Converted to `NaN`, then rejected (see below) |
| String-like | Raises `ValidationError` |

### Validation Checks

| Check | Error Message |
|-------|---------------|
| Cannot convert to numeric | `"{name} cannot be converted to numeric values. Ensure all values are numeric (int, float, Decimal)."` |
| Contains NaN | `"{name} contains {count} NaN values ({pct}%). Either remove rows with missing values or impute them before fitting."` |
| Contains Inf | `"{name} contains {count} infinite values. Replace Inf/-Inf with finite values or remove those rows."` |

---

## 2. Response Validation by Family

Each GLM family has constraints on valid response values:

### Gaussian (default)
- **Constraint:** None (any real values)
- **Use case:** Continuous outcomes

### Binomial
- **Constraint:** `y ∈ [0, 1]`
- **Error:** `"Binomial family requires response in [0, 1] (proportions or 0/1 binary). Found {n} values outside this range (min={min}, max={max}). For count data (successes/trials), divide by trials to get proportions."`
- **Warning:** If exactly 2 unique values but not [0, 1], suggests recoding

### Poisson
- **Constraint:** `y ≥ 0`
- **Error:** `"Poisson family requires non-negative response (counts). Found {n} negative values. Poisson models count data; negative counts are impossible."`
- **Warning:** If non-integer values, suggests quasipoisson for overdispersed data

### Gamma
- **Constraint:** `y > 0` (strictly positive)
- **Error:** `"Gamma family requires strictly positive response (y > 0). Found {n} values <= 0. Gamma models positive continuous data like claim amounts or durations."`

### Negative Binomial
- **Constraint:** `y ≥ 0`
- **Error:** `"Negative Binomial family requires non-negative response. Found {n} negative values."`

### Tweedie
- **Constraint:** `y ≥ 0`
- **Error:** `"Tweedie family requires non-negative response. Found {n} negative values."`

### Inverse Gaussian
- **Constraint:** `y > 0`
- **Error:** `"Inverse Gaussian family requires strictly positive response (y > 0). Found {n} values <= 0."`

### Universal Checks (all families)

| Check | Error Message |
|-------|---------------|
| Empty response | `"response is empty. Cannot fit model with no observations."` |
| Constant response | `"response is constant (all values = {value}). A GLM requires variation in the response variable."` |

---

## 3. Design Matrix (X) Validation

| Check | Error Message |
|-------|---------------|
| Not 2D | `"design matrix must be 2-dimensional (n_obs x n_features). Got shape {shape}."` |
| No observations | `"design matrix has no observations."` |
| No features | `"design matrix has no features."` |
| Contains NaN | `"design matrix contains NaN values in columns: {columns}. Remove or impute missing values before fitting."` |
| Contains Inf | `"design matrix contains infinite values in columns: {columns}. Replace Inf/-Inf with finite values."` |
| n_obs < n_features | **Warning:** `"design matrix has fewer observations ({n_obs}) than features ({n_features}). Model will be underdetermined. Consider regularization (alpha > 0)."` |
| Dimension mismatch | `"Response has {n_y} observations but design matrix has {n_X} rows. They must match."` |

---

## 4. Weights Validation

| Check | Error Message |
|-------|---------------|
| Length mismatch | `"weights length ({len}) does not match number of observations ({n_obs})."` |
| Negative weights | `"weights contains {n} negative values. Weights must be non-negative."` |
| All zero | `"weights sum to zero. At least some observations must have positive weight."` |
| Many zeros (>50%) | **Warning:** `"{pct}% of weights are zero. These observations will not contribute to the fit."` |

---

## 5. Offset / Exposure Validation

| Check | Applies To | Error Message |
|-------|------------|---------------|
| Length mismatch | All | `"offset length ({len}) does not match number of observations ({n_obs})."` |
| Non-positive exposure | Poisson, Gamma, NegBinomial with log link | `"Exposure must be strictly positive for {family} family with log link. Found {n} values <= 0. Exposure represents the denominator (e.g., time, population) and cannot be zero or negative."` |

---

## 6. UI Recommendations

### Pre-submission Validation
Consider validating these client-side before API calls:

1. **No empty columns** in selected factors
2. **Response column exists** and is numeric
3. **Exposure > 0** for rate models (Poisson, Gamma)
4. **Response bounds** based on family selection:
   - Binomial: warn if values outside [0, 1]
   - Poisson/NegBin: warn if negative values
   - Gamma: warn if zero or negative values

### Error Display
Validation errors include:
- **What went wrong** (e.g., "contains 15 NaN values")
- **Where** (column name if applicable)
- **How to fix** (actionable guidance)

Example error structure:
```json
{
  "error_type": "ValidationError",
  "field": "response",
  "message": "Gamma family requires strictly positive response (y > 0). Found 3 values <= 0.",
  "suggestion": "Gamma models positive continuous data like claim amounts. Remove or filter rows with zero values."
}
```

### Warning Display
Some issues produce warnings (model still fits but results may be suspect):
- Non-integer Poisson response
- Binomial response with unusual coding
- Many zero weights
- Underdetermined model (n < p)

---

## 7. Quick Reference Table

| Family | Response Constraint | Typical Use Case |
|--------|---------------------|------------------|
| `gaussian` | Any real | Continuous outcomes |
| `binomial` | [0, 1] | Binary/proportion outcomes |
| `poisson` | ≥ 0 (integers) | Claim counts, event frequencies |
| `gamma` | > 0 | Claim severity, durations |
| `tweedie` | ≥ 0 | Pure premium (zeros + positives) |
| `negbinomial` | ≥ 0 | Overdispersed counts |
| `inverse_gaussian` | > 0 | Positive continuous with high variance |

---

## 8. Code Reference

Validation is implemented in:
- `python/rustystats/validation.py` — Main validation module
- `python/rustystats/formula.py` — Integration in `_fit_glm_core()`

To manually validate before fitting:
```python
import rustystats as rs

y, X, weights, offset = rs.validate_glm_inputs(
    y=response_array,
    X=design_matrix,
    family="poisson",
    weights=weight_array,  # optional
    offset=exposure_array,  # optional
    feature_names=["Intercept", "Age", "Region_A", ...],
    is_exposure_offset=True,  # True for Poisson/Gamma with log link
)
```
