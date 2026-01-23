# RustyStats Diagnostics Enhancement Request

## Context

RustyStats is a Rust-based GLM fitting library with a Python API used by the GLAM actuarial modeling agent. The agent makes automated decisions about model structure (adding variables, converting to splines, tuning spline degrees of freedom, removing variables, etc.) based on statistical diagnostics.

The current diagnostics output is comprehensive but missing several key features required for **evidence-based automated decision making**. This document specifies the required enhancements.

---

## Feature 1: GCV Grid for Spline Degrees of Freedom Selection

### Purpose
When the agent considers converting a linear term to a spline or tuning an existing spline's degrees of freedom, it needs an objective criterion to select the optimal df. Generalized Cross-Validation (GCV) provides this.

### Requirements

1. **For each continuous variable currently in the model as a spline (`ns()` term):**
   - Compute GCV scores for df values from 3 to `max_df` (configurable, default 7)
   - Identify the optimal df (minimum GCV)
   - Report the current df and whether a change is warranted

2. **For each continuous variable currently in the model as linear:**
   - Compute GCV scores for the linear term (df=1) and spline terms (df 3-7)
   - Report whether conversion to spline is justified by GCV improvement

3. **GCV Formula:**
   ```
   GCV(df) = n * Deviance(df) / (n - df_effective)^2
   ```
   Where `df_effective` accounts for the smoothing parameter.

4. **Implementation approach:**
   - For each candidate df, refit the model with that spline specification
   - Extract deviance and effective degrees of freedom
   - Compute GCV score

### Output Schema

```json
{
  "spline_tuning": [
    {
      "variable": "VehAge",
      "current_df": 3,
      "current_gcv": 0.3142,
      "optimal_df": 4,
      "optimal_gcv": 0.3089,
      "gcv_reduction_pct": 1.69,
      "gcv_by_df": {
        "1": 0.3298,
        "3": 0.3142,
        "4": 0.3089,
        "5": 0.3095,
        "6": 0.3102,
        "7": 0.3118
      },
      "recommendation_strength": "moderate"
    },
    {
      "variable": "DrivAge",
      "current_df": 4,
      "current_gcv": 0.3089,
      "optimal_df": 4,
      "optimal_gcv": 0.3089,
      "gcv_reduction_pct": 0.0,
      "gcv_by_df": {
        "1": 0.3245,
        "3": 0.3112,
        "4": 0.3089,
        "5": 0.3091,
        "6": 0.3098,
        "7": 0.3107
      },
      "recommendation_strength": "none"
    }
  ]
}
```

### Field Definitions

| Field | Type | Description |
|-------|------|-------------|
| `variable` | string | Variable name |
| `current_df` | int | Current degrees of freedom (1 for linear) |
| `current_gcv` | float | GCV score at current df |
| `optimal_df` | int | df with minimum GCV |
| `optimal_gcv` | float | GCV score at optimal df |
| `gcv_reduction_pct` | float | `(current_gcv - optimal_gcv) / current_gcv * 100` |
| `gcv_by_df` | object | Map of df → GCV score for all evaluated values |
| `recommendation_strength` | string | One of: `"none"` (<0.5%), `"weak"` (0.5-1%), `"moderate"` (1-2%), `"strong"` (>2%) based on `gcv_reduction_pct` |

### Configuration Parameters

```python
result.diagnostics(
    ...,
    spline_tuning_config={
        "min_df": 3,
        "max_df": 7,
        "include_linear": True,  # Include df=1 in comparison
        "variables": None,  # None = all continuous in model, or list of specific variables
    }
)
```

---

## Feature 2: Joint Significance Test for Spline Terms

### Purpose
A spline term like `ns(VehAge, df=4)` generates multiple basis coefficients. The individual coefficient p-values test whether each basis function contributes, but the agent needs to know whether the **entire spline term** is jointly significant (i.e., should the variable remain in the model at all).

### Requirements

1. **For each spline term in the model:**
   - Perform a joint Wald test (χ²) on all coefficients belonging to that term
   - Compute degrees of freedom (number of basis functions)
   - Compute p-value

2. **Test specification:**
   - H₀: All coefficients for the spline term = 0
   - H₁: At least one coefficient ≠ 0
   - Test statistic: `χ² = β'(Var(β))⁻¹β` for the subvector of spline coefficients

3. **Also compute for categorical factors** (already partially present as `significance.chi2`, but ensure consistency)

### Output Schema

Add to the existing `factors[]` array:

```json
{
  "factors": [
    {
      "name": "VehAge",
      "factor_type": "continuous",
      "in_model": true,
      "transform": "ns(VehAge, df=4)",
      "coefficients": [...],
      "joint_test": {
        "test_type": "wald",
        "chi2": 892.45,
        "df": 4,
        "p_value": 0.0,
        "significant_01": true,
        "significant_05": true,
        "deviance_contribution": 876.32,
        "deviance_contribution_pct": 0.68
      }
    },
    {
      "name": "DrivAge",
      "factor_type": "continuous",
      "in_model": true,
      "transform": "ns(DrivAge, df=3)",
      "coefficients": [...],
      "joint_test": {
        "test_type": "wald",
        "chi2": 45.67,
        "df": 3,
        "p_value": 0.0,
        "significant_01": true,
        "significant_05": true,
        "deviance_contribution": 44.89,
        "deviance_contribution_pct": 0.035
      }
    },
    {
      "name": "BonusMalus",
      "factor_type": "continuous",
      "in_model": true,
      "transform": null,
      "coefficients": [...],
      "joint_test": {
        "test_type": "wald",
        "chi2": 3009.89,
        "df": 1,
        "p_value": 0.0,
        "significant_01": true,
        "significant_05": true,
        "deviance_contribution": 2987.45,
        "deviance_contribution_pct": 2.33
      }
    }
  ]
}
```

### Field Definitions

| Field | Type | Description |
|-------|------|-------------|
| `test_type` | string | `"wald"` - type of joint test performed |
| `chi2` | float | Chi-squared test statistic |
| `df` | int | Degrees of freedom (number of coefficients for this term) |
| `p_value` | float | P-value from χ² distribution |
| `significant_01` | bool | `p_value < 0.01` |
| `significant_05` | bool | `p_value < 0.05` |
| `deviance_contribution` | float | Increase in deviance if this term is dropped |
| `deviance_contribution_pct` | float | As percentage of null deviance |

### Notes
- For linear terms (df=1), this is equivalent to the squared z-test
- For categorical factors with target encoding (single coefficient), same applies
- For categorical factors with dummy encoding, test all level coefficients jointly

---

## Feature 3: Bootstrap Standard Errors and Confidence Intervals

### Purpose
Asymptotic standard errors assume the model is correctly specified and may underestimate uncertainty. Bootstrap SEs provide a more robust measure of coefficient stability, which is critical for the stability gates in the agent specification.

### Requirements

1. **For each coefficient in the model:**
   - Perform B bootstrap resamples (default B=200)
   - Refit the model on each resample
   - Compute bootstrap SE as standard deviation of coefficient estimates
   - Compute bootstrap percentile CI (2.5%, 97.5%)

2. **Resampling strategy:**
   - Case resampling (resample rows with replacement)
   - Stratified by response if highly imbalanced (optional)

3. **Also compute:**
   - Bootstrap bias: `mean(β_boot) - β_original`
   - Bootstrap coefficient of variation: `bootstrap_se / |β|`

### Output Schema

Extend `coefficient_summary[]`:

```json
{
  "coefficient_summary": [
    {
      "feature": "BonusMalus",
      "estimate": 0.023348,
      "std_error": 0.000426,
      "z_value": 54.862,
      "p_value": 0.0,
      "significant": true,
      "relativity": 1.0236,
      "relativity_ci": [1.0228, 1.0245],
      "bootstrap": {
        "n_resamples": 200,
        "se": 0.000445,
        "bias": 0.000012,
        "cv": 0.019,
        "ci_lower": 0.022478,
        "ci_upper": 0.024198,
        "ci_method": "percentile",
        "relativity_ci_lower": 1.0227,
        "relativity_ci_upper": 1.0245
      }
    },
    {
      "feature": "ns(VehAge, 1/3)",
      "estimate": -0.045044,
      "std_error": 0.001904,
      "z_value": -23.654,
      "p_value": 0.0,
      "significant": true,
      "relativity": 0.956,
      "relativity_ci": [0.9524, 0.9595],
      "bootstrap": {
        "n_resamples": 200,
        "se": 0.002145,
        "bias": -0.000089,
        "cv": 0.048,
        "ci_lower": -0.049234,
        "ci_upper": -0.040912,
        "ci_method": "percentile",
        "relativity_ci_lower": 0.9519,
        "relativity_ci_upper": 0.9599
      }
    }
  ]
}
```

### Field Definitions

| Field | Type | Description |
|-------|------|-------------|
| `bootstrap.n_resamples` | int | Number of bootstrap samples used |
| `bootstrap.se` | float | Bootstrap standard error |
| `bootstrap.bias` | float | `mean(β_boot) - β_original` |
| `bootstrap.cv` | float | Coefficient of variation: `se / abs(estimate)` |
| `bootstrap.ci_lower` | float | 2.5th percentile of bootstrap distribution |
| `bootstrap.ci_upper` | float | 97.5th percentile of bootstrap distribution |
| `bootstrap.ci_method` | string | `"percentile"` or `"bca"` (bias-corrected accelerated) |
| `bootstrap.relativity_ci_lower` | float | `exp(ci_lower)` for log-link |
| `bootstrap.relativity_ci_upper` | float | `exp(ci_upper)` for log-link |

### Configuration Parameters

```python
result.diagnostics(
    ...,
    bootstrap_config={
        "enabled": True,
        "n_resamples": 200,
        "ci_method": "percentile",  # or "bca"
        "seed": 42,
        "parallel": True,
    }
)
```

### Performance Note
Bootstrap is computationally expensive. Consider:
- Parallel execution across resamples
- Option to run on subset of coefficients
- Caching mechanism for repeated calls

---

## Feature 4: Deviance Confidence Intervals

### Purpose
When comparing models (e.g., before/after adding a variable), the agent needs to know if the deviance improvement is statistically significant or could be due to sampling variation.

### Requirements

1. **For train and test datasets:**
   - Compute bootstrap CI for total deviance
   - Compute bootstrap CI for mean deviance (deviance / n)

2. **For model comparison:**
   - CI for deviance difference between current and null model
   - CI for deviance difference between current and saturated model

### Output Schema

Extend `train_test`:

```json
{
  "train_test": {
    "train": {
      "dataset": "train",
      "n_obs": 406439,
      "deviance": 128249.87,
      "deviance_ci": {
        "n_resamples": 200,
        "ci_lower": 127845.23,
        "ci_upper": 128654.51,
        "ci_method": "percentile"
      },
      "mean_deviance": 0.3155,
      "mean_deviance_ci": {
        "ci_lower": 0.3145,
        "ci_upper": 0.3165
      },
      "log_likelihood": -84898.76,
      "aic": 169817.52,
      "gini": 0.242,
      "auc": 0.621,
      "ae_ratio": 1.0001
    },
    "test": {
      "dataset": "test",
      "n_obs": 135957,
      "deviance": 43099.07,
      "deviance_ci": {
        "n_resamples": 200,
        "ci_lower": 42756.89,
        "ci_upper": 43441.25,
        "ci_method": "percentile"
      },
      "mean_deviance": 0.317,
      "mean_deviance_ci": {
        "ci_lower": 0.3145,
        "ci_upper": 0.3196
      },
      "log_likelihood": -28465.67,
      "aic": 56951.35,
      "gini": 0.2465,
      "auc": 0.6232,
      "ae_ratio": 0.9954
    }
  }
}
```

### Also Add Model Comparison CIs

Extend `model_comparison`:

```json
{
  "model_comparison": {
    "null_deviance": 132315.35,
    "model_deviance": 128249.87,
    "deviance_reduction": 4065.48,
    "deviance_reduction_ci": {
      "ci_lower": 3892.15,
      "ci_upper": 4238.81,
      "significant": true
    },
    "likelihood_ratio_chi2": 4065.48,
    "likelihood_ratio_df": 9,
    "likelihood_ratio_pvalue": 0.0,
    "deviance_reduction_pct": 3.0723,
    "aic_improvement": 4047.48
  }
}
```

---

## Summary: Complete Enhanced Diagnostics Schema

Below is the full schema showing where new fields integrate with existing structure:

```json
{
  "model_summary": { ... },
  
  "train_test": {
    "train": {
      "deviance": 128249.87,
      "deviance_ci": { "ci_lower": ..., "ci_upper": ..., "n_resamples": 200 },
      "mean_deviance_ci": { "ci_lower": ..., "ci_upper": ... },
      ...
    },
    "test": { ... }
  },
  
  "model_comparison": {
    "deviance_reduction": 4065.48,
    "deviance_reduction_ci": { "ci_lower": ..., "ci_upper": ..., "significant": true },
    ...
  },
  
  "factors": [
    {
      "name": "VehAge",
      "joint_test": {
        "test_type": "wald",
        "chi2": 892.45,
        "df": 4,
        "p_value": 0.0,
        "significant_01": true,
        "significant_05": true,
        "deviance_contribution": 876.32,
        "deviance_contribution_pct": 0.68
      },
      ...
    }
  ],
  
  "coefficient_summary": [
    {
      "feature": "BonusMalus",
      "estimate": 0.023348,
      "std_error": 0.000426,
      "bootstrap": {
        "n_resamples": 200,
        "se": 0.000445,
        "bias": 0.000012,
        "cv": 0.019,
        "ci_lower": 0.022478,
        "ci_upper": 0.024198,
        "ci_method": "percentile",
        "relativity_ci_lower": 1.0227,
        "relativity_ci_upper": 1.0245
      },
      ...
    }
  ],
  
  "spline_tuning": [
    {
      "variable": "VehAge",
      "current_df": 3,
      "current_gcv": 0.3142,
      "optimal_df": 4,
      "optimal_gcv": 0.3089,
      "gcv_reduction_pct": 1.69,
      "gcv_by_df": { "1": 0.3298, "3": 0.3142, "4": 0.3089, ... },
      "recommendation_strength": "moderate"
    }
  ],
  
  "vif": [ ... ],
  "warnings": [ ... ]
}
```

---

## Implementation Priority

| Feature | Priority | Complexity | Notes |
|---------|----------|------------|-------|
| **GCV for spline df** | 🔴 P0 | Medium | Requires refitting model for each df candidate |
| **Joint significance test** | 🔴 P0 | Low | Matrix algebra on existing covariance matrix |
| **Bootstrap SE** | 🟡 P1 | High | Computationally expensive, needs parallelization |
| **Deviance CI** | 🟡 P1 | Medium | Shares bootstrap infrastructure with Feature 3 |

---

## Testing Requirements

1. **GCV Tuning:**
   - Verify optimal_df matches manual grid search
   - Test with linear term (should recommend spline when appropriate)
   - Test edge cases (all GCV equal, monotonic GCV)

2. **Joint Test:**
   - Verify chi2 matches sum of squared z-values for single-coef terms
   - Verify against known statistical software output (R's `car::Anova`)
   - Test with both spline and categorical factors

3. **Bootstrap:**
   - Verify bootstrap SE ≈ asymptotic SE for large n, simple model
   - Verify CI coverage on simulated data
   - Test reproducibility with seed

4. **Deviance CI:**
   - Verify CI contains true deviance on simulated data
   - Test significance flag matches LRT p-value

---

## API Backwards Compatibility

All new fields should be **additive**. Existing code consuming the diagnostics JSON should continue to work. New fields are optional in the output unless explicitly requested via config.

```python
# Existing call continues to work
result.diagnostics(train_data=df, test_data=test_df)

# New features enabled via config
result.diagnostics(
    train_data=df,
    test_data=test_df,
    spline_tuning_config={"max_df": 7},
    bootstrap_config={"enabled": True, "n_resamples": 200},
)
```
