# GLM Modeling Workflow

This guide walks through a complete GLM modeling workflow using RustyStats, showing how to use diagnostics at each step to make informed decisions about model development.

**Target audience**: Data scientists and actuaries building production GLM models.

---

## Overview: The Iterative Modeling Process

Building a good GLM is iterative. At each stage, diagnostics guide your next decision:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        GLM MODELING WORKFLOW                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. DATA EXPLORATION                                                 │
│     └─► explore_data() → Understand distributions, spot issues       │
│                                                                      │
│  2. FAMILY & LINK SELECTION                                          │
│     └─► Response distribution → Choose appropriate family/link       │
│                                                                      │
│  3. INITIAL MODEL FIT                                                │
│     └─► rs.glm().fit() → Baseline model with key predictors         │
│                                                                      │
│  4. DIAGNOSTICS CHECK                                                │
│     └─► result.diagnostics() → Assess fit quality                   │
│         ├─► Calibration OK? → Continue                              │
│         ├─► Overdispersion? → Consider QuasiPoisson/NegBin          │
│         └─► Poor discrimination? → Add predictors/interactions      │
│                                                                      │
│  5. FACTOR ANALYSIS                                                  │
│     └─► Per-factor A/E → Identify non-linear effects, missing vars  │
│                                                                      │
│  6. MODEL REFINEMENT                                                 │
│     └─► Add splines, interactions, transformations                  │
│                                                                      │
│  7. REGULARIZATION (optional)                                        │
│     └─► fit(alpha=...) → Variable selection for many predictors     │
│                                                                      │
│  8. FINAL VALIDATION                                                 │
│     └─► Out-of-sample testing, stability checks                     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Data Exploration

Before fitting any model, understand your data.

### 1.1 Load and Inspect Data

```python
import rustystats as rs
import polars as pl
import numpy as np

# Load your data
data = pl.read_csv("insurance_claims.csv")

# Basic summary
print(f"Rows: {len(data)}")
print(f"Columns: {data.columns}")
print(data.describe())
```

### 1.2 Use `explore_data()` for Pre-Fit Analysis

```python
exploration = rs.explore_data(
    data=data,
    response="claim_amount",
    exposure="exposure",
    categorical_factors=["region", "vehicle_type", "driver_age_band"],
    continuous_factors=["vehicle_age", "driver_experience", "credit_score"],
)

# View as JSON (useful for LLM analysis or dashboards)
print(exploration.to_json(indent=2))
```

### 1.3 Key Diagnostics to Check

| Diagnostic | What to Look For | Action if Problem |
|------------|------------------|-------------------|
| **Response distribution** | Zeros, skewness, outliers | Choose appropriate family |
| **Missing values** | High % missing in key factors | Impute or exclude |
| **Rare categories** | Levels with < 1% of data | Combine into "Other" |
| **Correlation with response** | Factors with no signal | Consider excluding |

**Decision Point**: Based on response distribution:

```python
response_stats = exploration.response_stats

# Check for zeros
if response_stats["pct_zero"] > 10:
    print("⚠️ Many zeros → Consider Tweedie or zero-inflated model")
    
# Check for overdispersion (counts)
if response_stats["variance"] > response_stats["mean"] * 1.5:
    print("⚠️ Overdispersion → Consider QuasiPoisson or NegBin")
```

---

## Step 2: Choose Family and Link

The response distribution determines your family choice.

### 2.1 Decision Tree

```python
def suggest_family(y: np.ndarray) -> str:
    """Suggest a GLM family based on response characteristics."""
    
    # Check data type and range
    is_binary = set(np.unique(y)) <= {0, 1}
    is_count = np.all(y == np.floor(y)) and np.all(y >= 0)
    is_positive = np.all(y > 0)
    has_zeros = np.any(y == 0)
    
    if is_binary:
        return "binomial"
    
    if is_count:
        # Check dispersion
        mean_y = np.mean(y)
        var_y = np.var(y)
        dispersion_ratio = var_y / mean_y
        
        if 0.8 <= dispersion_ratio <= 1.2:
            return "poisson"
        elif dispersion_ratio > 1.2:
            return "quasipoisson"  # or "negbinomial"
        else:
            return "poisson"  # underdispersion is rare
    
    # Continuous
    if has_zeros and is_positive or np.any(y == 0):
        return "tweedie"  # Can handle zeros
    
    if is_positive:
        return "gamma"
    
    return "gaussian"  # Default for continuous data

# Usage
y = data["claim_amount"].to_numpy()
suggested = suggest_family(y)
print(f"Suggested family: {suggested}")
```

### 2.2 Family-Link Combinations

| Family | Default Link | When to Use |
|--------|--------------|-------------|
| `gaussian` | identity | Continuous, can be negative |
| `poisson` | log | Counts with mean ≈ variance |
| `binomial` | logit | Binary outcomes |
| `gamma` | log | Positive continuous, CV constant |
| `tweedie` | log | Positive with exact zeros |
| `quasipoisson` | log | Overdispersed counts |
| `negbinomial` | log | Overdispersed counts (proper likelihood) |

---

## Step 3: Fit Initial Model

Start with a baseline model using your most important predictors.

### 3.1 Fit Initial Model

```python
result = rs.glm(
    "claim_amount ~ vehicle_age + driver_experience + C(region)",
    data=data,
    family="gamma",
    offset="exposure",
).fit()
```

### 3.2 Quick Convergence Check

```python
print(f"Converged: {result.converged}")
print(f"Iterations: {result.iterations}")
print(f"Deviance: {result.deviance:.2f}")

if not result.converged:
    print("⚠️ Model did not converge!")
    print("   → Try: increase max_iter, check for separation, add regularization")
```

---

## Step 4: Comprehensive Diagnostics

This is the critical step. Use `result.diagnostics()` to assess model quality.

### 4.1 Compute Full Diagnostics

```python
diagnostics = result.diagnostics(
    data=data,
    categorical_factors=["region", "vehicle_type", "driver_age_band"],
    continuous_factors=["vehicle_age", "driver_experience", "credit_score"],
    n_bins=10,
)
```

### 4.2 Overall Fit Statistics

```python
fit_stats = diagnostics.fit_statistics

print("=== FIT STATISTICS ===")
print(f"Deviance:       {fit_stats['deviance']:.2f}")
print(f"Null Deviance:  {fit_stats['null_deviance']:.2f}")
print(f"Pseudo R²:      {fit_stats['pseudo_r2']:.4f}")
print(f"AIC:            {fit_stats['aic']:.2f}")
print(f"BIC:            {fit_stats['bic']:.2f}")
```

**Decision Points**:

| Metric | Good Range | Action if Bad |
|--------|------------|---------------|
| Pseudo R² | > 0.1 for GLMs | Add predictors, check family |
| AIC/BIC | Lower is better | Use for model comparison |
| Deviance/df | ≈ 1 for Poisson/Binomial | Check overdispersion |

### 4.3 Calibration Diagnostics

Calibration measures whether predictions match actuals on average.

```python
calibration = diagnostics.calibration

print("=== CALIBRATION ===")
print(f"Overall A/E:     {calibration['actual_expected_ratio']:.4f}")
print(f"Calibration Err: {calibration['calibration_error']:.4f}")
print(f"H-L p-value:     {calibration['hosmer_lemeshow_pvalue']:.4f}")

# Check calibration by decile
print("\nCalibration by Decile:")
print("Decile | Predicted | Actual | A/E Ratio | Count")
print("-" * 55)
for bin_data in calibration['by_decile']:
    print(f"  {bin_data['bin_index']:2d}   | {bin_data['predicted_mean']:9.2f} | "
          f"{bin_data['actual_mean']:6.2f} | {bin_data['actual_expected_ratio']:9.3f} | "
          f"{bin_data['count']:5d}")
```

**Decision Points**:

| Diagnostic | Threshold | Interpretation | Action |
|------------|-----------|----------------|--------|
| Overall A/E | 0.95 - 1.05 | Good calibration | Continue |
| Overall A/E | < 0.95 or > 1.05 | Systematic bias | Check model specification |
| H-L p-value | > 0.05 | No evidence of poor fit | Continue |
| H-L p-value | < 0.05 | Significant miscalibration | Add predictors or interactions |
| Decile A/E | All near 1.0 | Uniform calibration | Good |
| Decile A/E | Trend (e.g., low→high) | Non-linearity | Add splines or transforms |

```python
# Automated calibration check
def check_calibration(diagnostics) -> dict:
    """Check calibration and return recommendations."""
    cal = diagnostics.calibration
    issues = []
    
    ae = cal['actual_expected_ratio']
    if ae < 0.95 or ae > 1.05:
        issues.append(f"Overall A/E = {ae:.3f}, outside [0.95, 1.05]")
    
    hl_p = cal['hosmer_lemeshow_pvalue']
    if hl_p < 0.05:
        issues.append(f"Hosmer-Lemeshow p = {hl_p:.4f} < 0.05")
    
    # Check for trend in deciles
    decile_ae = [b['actual_expected_ratio'] for b in cal['by_decile']]
    if len(decile_ae) >= 5:
        correlation = np.corrcoef(range(len(decile_ae)), decile_ae)[0, 1]
        if abs(correlation) > 0.7:
            issues.append(f"Strong trend in decile A/E (r = {correlation:.2f})")
    
    return {
        "passed": len(issues) == 0,
        "issues": issues,
        "recommendation": "Add interactions or non-linear terms" if issues else "Calibration OK"
    }

cal_check = check_calibration(diagnostics)
print(f"\nCalibration Check: {'✓ PASS' if cal_check['passed'] else '✗ FAIL'}")
for issue in cal_check['issues']:
    print(f"  → {issue}")
```

### 4.4 Discrimination Diagnostics

Discrimination measures how well the model separates high from low risk.

```python
discrimination = diagnostics.discrimination

print("=== DISCRIMINATION ===")
print(f"Gini Coefficient: {discrimination['gini_coefficient']:.4f}")
print(f"AUC:              {discrimination['auc']:.4f}")
print(f"KS Statistic:     {discrimination['ks_statistic']:.4f}")
print(f"Lift @ 10%:       {discrimination['lift_at_10pct']:.2f}x")
print(f"Lift @ 20%:       {discrimination['lift_at_20pct']:.2f}x")
```

**Interpretation Guide**:

| Gini | Model Quality |
|------|---------------|
| < 0.20 | Poor discrimination |
| 0.20 - 0.40 | Fair |
| 0.40 - 0.60 | Good |
| > 0.60 | Excellent |

**Decision Points**:

```python
def check_discrimination(diagnostics) -> dict:
    """Check discrimination and return recommendations."""
    disc = diagnostics.discrimination
    if disc is None:
        return {"passed": True, "issues": [], "recommendation": "N/A for this model type"}
    
    issues = []
    gini = disc['gini_coefficient']
    
    if gini < 0.20:
        issues.append(f"Poor Gini = {gini:.3f}, need better predictors")
    elif gini < 0.30:
        issues.append(f"Fair Gini = {gini:.3f}, consider adding features")
    
    lift_10 = disc['lift_at_10pct']
    if lift_10 < 1.5:
        issues.append(f"Low lift@10% = {lift_10:.2f}x, model not separating well")
    
    return {
        "passed": len(issues) == 0,
        "gini": gini,
        "issues": issues,
        "recommendation": "Add more predictive features or interactions" if issues else "Discrimination OK"
    }
```

### 4.5 Residual Analysis

Residuals reveal systematic patterns the model is missing.

```python
resid_summary = diagnostics.residual_summary

print("=== RESIDUAL SUMMARY ===")
for resid_type in ['pearson', 'deviance']:
    r = resid_summary[resid_type]
    print(f"\n{resid_type.upper()} Residuals:")
    print(f"  Mean:     {r.mean:8.4f}  (should be ≈ 0)")
    print(f"  Std:      {r.std:8.4f}  (should be ≈ 1 for Pearson)")
    print(f"  Skewness: {r.skewness:8.4f}  (should be ≈ 0)")
    print(f"  Kurtosis: {r.kurtosis:8.4f}  (should be ≈ 3)")
    print(f"  Range:    [{r.min:.2f}, {r.max:.2f}]")
```

**Decision Points**:

| Symptom | Possible Cause | Action |
|---------|----------------|--------|
| Mean ≠ 0 | Bias in predictions | Check calibration, add intercept |
| Std >> 1 | Overdispersion | Use quasi-family or negative binomial |
| High skewness | Non-normality, outliers | Check for outliers, consider robust fitting |
| Large outliers | Extreme observations | Investigate, possibly winsorize |

### 4.6 Overdispersion Check

Critical for Poisson and Binomial models.

```python
def check_overdispersion(result, diagnostics) -> dict:
    """Check for overdispersion in count/binary models."""
    
    family = result.family.lower()
    if family not in ['poisson', 'binomial']:
        return {"applicable": False}
    
    # Dispersion ratio = Pearson χ² / df_resid
    pearson_chi2 = result.pearson_chi2()
    df_resid = result.df_resid
    dispersion = pearson_chi2 / df_resid
    
    print(f"Dispersion ratio: {dispersion:.3f}")
    
    if dispersion > 1.5:
        return {
            "applicable": True,
            "overdispersed": True,
            "dispersion": dispersion,
            "recommendation": f"Significant overdispersion ({dispersion:.2f}). "
                            f"Switch to {'quasipoisson' if family == 'poisson' else 'quasibinomial'} "
                            f"or negative binomial."
        }
    elif dispersion < 0.7:
        return {
            "applicable": True,
            "overdispersed": False,
            "underdispersed": True,
            "dispersion": dispersion,
            "recommendation": f"Possible underdispersion ({dispersion:.2f}). Check model specification."
        }
    else:
        return {
            "applicable": True,
            "overdispersed": False,
            "dispersion": dispersion,
            "recommendation": f"Dispersion OK ({dispersion:.2f})"
        }

dispersion_check = check_overdispersion(result, diagnostics)
print(dispersion_check['recommendation'])
```

---

## Step 5: Per-Factor Analysis

Examine each predictor's relationship with residuals.

### 5.1 Review Factor Diagnostics

```python
print("=== FACTOR DIAGNOSTICS ===")
for factor in diagnostics.factors:
    print(f"\n{factor.name} ({factor.factor_type})")
    print(f"  In model: {factor.in_model}")
    
    # Residual pattern
    rp = factor.residual_pattern
    print(f"  Residual correlation: {rp.correlation_with_residuals:.4f}")
    print(f"  Variance explained:   {rp.residual_variance_explained:.4f}")
    
    # A/E summary
    ae_ratios = [b.actual_expected_ratio for b in factor.actual_vs_expected]
    ae_range = max(ae_ratios) - min(ae_ratios)
    print(f"  A/E range: {min(ae_ratios):.3f} - {max(ae_ratios):.3f} (spread: {ae_range:.3f})")
```

### 5.2 Identify Problem Factors

```python
def identify_problem_factors(diagnostics, 
                              ae_threshold: float = 0.15,
                              resid_corr_threshold: float = 0.05) -> list:
    """Identify factors that need attention."""
    problems = []
    
    for factor in diagnostics.factors:
        issues = []
        
        # Check A/E spread
        ae_ratios = [b.actual_expected_ratio for b in factor.actual_vs_expected]
        ae_range = max(ae_ratios) - min(ae_ratios)
        
        if ae_range > ae_threshold:
            issues.append(f"Large A/E spread ({ae_range:.3f})")
        
        # Check residual correlation
        resid_corr = abs(factor.residual_pattern.correlation_with_residuals)
        if resid_corr > resid_corr_threshold:
            issues.append(f"Residual correlation ({resid_corr:.3f})")
        
        # Check if factor is NOT in model but shows signal
        if not factor.in_model and (ae_range > ae_threshold / 2 or resid_corr > resid_corr_threshold / 2):
            issues.append("NOT in model but shows predictive signal")
        
        if issues:
            problems.append({
                "factor": factor.name,
                "type": factor.factor_type,
                "in_model": factor.in_model,
                "issues": issues,
                "recommendation": get_factor_recommendation(factor, issues)
            })
    
    return problems


def get_factor_recommendation(factor, issues) -> str:
    """Generate recommendation for a problem factor."""
    if not factor.in_model:
        return "Consider adding to model"
    
    if factor.factor_type == "continuous":
        ae_ratios = [b.actual_expected_ratio for b in factor.actual_vs_expected]
        # Check for non-monotonic pattern
        diffs = np.diff(ae_ratios)
        sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
        if sign_changes > 2:
            return "Add spline: ns(factor, df=4) or bs(factor, df=5)"
        else:
            return "Consider polynomial or log transformation"
    else:
        return "Check category groupings, consider interactions"


# Run analysis
problems = identify_problem_factors(diagnostics)

print("\n=== FACTORS NEEDING ATTENTION ===")
for p in problems:
    print(f"\n{p['factor']} ({p['type']}, in_model={p['in_model']})")
    for issue in p['issues']:
        print(f"  ⚠️ {issue}")
    print(f"  → {p['recommendation']}")
```

### 5.3 Visualize A/E by Factor

```python
import matplotlib.pyplot as plt

def plot_ae_by_factor(factor_diagnostics, figsize=(10, 6)):
    """Plot A/E ratios for a factor."""
    fig, ax = plt.subplots(figsize=figsize)
    
    bins = factor_diagnostics.actual_vs_expected
    labels = [b.bin_label for b in bins]
    ae_ratios = [b.actual_expected_ratio for b in bins]
    ci_lower = [b.ae_confidence_interval_lower for b in bins]
    ci_upper = [b.ae_confidence_interval_upper for b in bins]
    
    x = range(len(bins))
    
    # Plot bars
    bars = ax.bar(x, ae_ratios, color='steelblue', alpha=0.7)
    
    # Add error bars for confidence intervals
    ax.errorbar(x, ae_ratios, 
                yerr=[np.array(ae_ratios) - np.array(ci_lower), 
                      np.array(ci_upper) - np.array(ae_ratios)],
                fmt='none', color='black', capsize=3)
    
    # Reference line at 1.0
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, label='Perfect calibration')
    ax.axhline(y=1.05, color='orange', linestyle=':', alpha=0.7)
    ax.axhline(y=0.95, color='orange', linestyle=':', alpha=0.7)
    
    ax.set_xlabel(factor_diagnostics.name)
    ax.set_ylabel('Actual / Expected Ratio')
    ax.set_title(f'Calibration by {factor_diagnostics.name}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    return fig, ax

# Plot for each problem factor
for p in problems[:3]:  # Top 3 problems
    factor = next(f for f in diagnostics.factors if f.name == p['factor'])
    plot_ae_by_factor(factor)
    plt.show()
```

---

## Step 6: Model Refinement

Based on diagnostics, refine your model.

### 6.1 Adding Non-Linear Effects (Splines)

If continuous factors show non-linear A/E patterns:

```python
# Before: linear effect
result_v1 = rs.glm(
    "claim_amount ~ vehicle_age + driver_experience + C(region)",
    data=data,
    family="gamma",
    exposure="exposure",
).fit()

# After: non-linear effects with natural splines
result_v2 = rs.glm(
    "claim_amount ~ ns(vehicle_age, df=4) + ns(driver_experience, df=3) + C(region)",
    data=data,
    family="gamma", 
    exposure="exposure",
).fit()

# Compare
print(f"Model v1 AIC: {result_v1.aic():.2f}")
print(f"Model v2 AIC: {result_v2.aic():.2f}")
print(f"Improvement: {result_v1.aic() - result_v2.aic():.2f} (lower is better)")
```

### 6.2 Adding Interactions

If `detect_interactions()` finds significant pairs:

```python
# Check diagnostics for interaction candidates
interactions = diagnostics.interaction_candidates

print("=== INTERACTION CANDIDATES ===")
for ic in interactions:
    sig = "***" if ic.pvalue < 0.001 else "**" if ic.pvalue < 0.01 else "*" if ic.pvalue < 0.05 else ""
    print(f"{ic.factor1} × {ic.factor2}: strength={ic.interaction_strength:.4f}, p={ic.pvalue:.4f}{sig}")

# Add top interaction to model
if interactions and interactions[0].pvalue < 0.05:
    f1, f2 = interactions[0].factor1, interactions[0].factor2
    formula = f"claim_amount ~ vehicle_age + driver_experience + C(region) + {f1}:{f2}"
    result_v3 = rs.glm(formula, data=data, family="gamma", exposure="exposure").fit()
    print(f"Model with interaction AIC: {result_v3.aic():.2f}")
```

### 6.3 Handling Overdispersion

If dispersion check showed overdispersion:

```python
# Original Poisson
result_pois = rs.glm(
    "claim_count ~ vehicle_age + C(region)",
    data=data,
    family="poisson",
    exposure="exposure",
).fit()

# QuasiPoisson (adjusts standard errors)
result_quasi = rs.glm(
    "claim_count ~ vehicle_age + C(region)",
    data=data,
    family="quasipoisson",
    exposure="exposure",
).fit()

# Negative Binomial (estimates dispersion parameter)
result_nb = rs.glm(
    "claim_count ~ vehicle_age + C(region)",
    data=data,
    family="negbinomial",
    exposure="exposure",
).fit()

# Compare standard errors
print("\nCoefficient Standard Errors:")
print(f"Poisson:      {result_pois.bse()[1]:.4f}")
print(f"QuasiPoisson: {result_quasi.bse()[1]:.4f}")
print(f"NegBinomial:  {result_nb.bse()[1]:.4f}")
```

### 6.4 Re-Run Diagnostics After Refinement

Always re-check diagnostics after changes:

```python
# Fit refined model
result_refined = rs.glm(
    "claim_amount ~ ns(vehicle_age, df=4) + ns(driver_experience, df=3) + "
    "C(region) + vehicle_age:C(region)",
    data=data,
    family="gamma",
    exposure="exposure",
).fit()

# Re-compute diagnostics
diagnostics_refined = result_refined.diagnostics(
    data=data,
    categorical_factors=["region", "vehicle_type"],
    continuous_factors=["vehicle_age", "driver_experience"],
)

# Compare calibration
cal_before = diagnostics.calibration['actual_expected_ratio']
cal_after = diagnostics_refined.calibration['actual_expected_ratio']
print(f"Overall A/E: {cal_before:.4f} → {cal_after:.4f}")

# Compare discrimination
gini_before = diagnostics.discrimination['gini_coefficient']
gini_after = diagnostics_refined.discrimination['gini_coefficient']
print(f"Gini:        {gini_before:.4f} → {gini_after:.4f}")
```

---

## Step 7: Regularization for Variable Selection

When you have many predictors, use regularization to select the important ones.

### 7.1 Ridge Regularization (L2)

Ridge shrinks coefficients toward zero but keeps all variables:

```python
# Fit with Ridge regularization
result_ridge = rs.glm(
    "claim_amount ~ vehicle_age + driver_experience + C(region) + C(vehicle_type)",
    data=data,
    family="gamma",
    offset="exposure",
).fit(alpha=0.1, l1_ratio=0.0)

print(f"Ridge coefficients: {result_ridge.params}")
```

### 7.2 Lasso Regularization (L1)

Lasso performs variable selection by zeroing out weak predictors:

```python
# Fit with Lasso regularization
result_lasso = rs.glm(
    "claim_amount ~ vehicle_age + driver_experience + C(region) + C(vehicle_type)",
    data=data,
    family="gamma",
    offset="exposure",
).fit(alpha=0.1, l1_ratio=1.0)

print(f"Non-zero coefficients: {result_lasso.n_nonzero()}")
print(f"Selected features: {result_lasso.selected_features()}")
```

### 7.3 Elastic Net (L1 + L2)

Elastic Net combines Ridge and Lasso:

```python
# Fit with Elastic Net (50% L1, 50% L2)
result_enet = rs.glm(
    "claim_amount ~ vehicle_age + driver_experience + C(region) + C(vehicle_type)",
    data=data,
    family="gamma",
    offset="exposure",
).fit(alpha=0.1, l1_ratio=0.5)

print(f"Non-zero coefficients: {result_enet.n_nonzero()}")
```

### 7.4 Tuning Alpha

Try different alpha values to find the right sparsity level:

```python
# Test different regularization strengths
for alpha in [0.001, 0.01, 0.1, 1.0]:
    result = rs.glm(
        "claim_amount ~ vehicle_age + driver_experience + C(region) + C(vehicle_type)",
        data=data,
        family="gamma",
        offset="exposure",
    ).fit(alpha=alpha, l1_ratio=1.0)
    
    print(f"α={alpha}: {result.n_nonzero()} features, deviance={result.deviance:.2f}")
```

---

## Step 8: Final Validation

Before deploying, validate on held-out data.

### 8.1 Train/Test Split

```python
from sklearn.model_selection import train_test_split

# Split data
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Fit on training data
result_train = rs.glm(
    "claim_amount ~ ns(vehicle_age, df=4) + C(region)",
    data=train_data,
    family="gamma",
    exposure="exposure",
).fit()

# Predict on test data
test_predictions = result_train.predict(test_data)

# Compute test diagnostics
test_y = test_data["claim_amount"].to_numpy()
test_exposure = test_data["exposure"].to_numpy()

# Manual A/E calculation
test_ae = np.sum(test_y) / np.sum(test_predictions)
print(f"Test set A/E: {test_ae:.4f}")
```

### 8.2 Stability Check

```python
def stability_check(data, formula, family, exposure_col, n_bootstrap=10, sample_frac=0.8):
    """Check coefficient stability via bootstrap."""
    
    coef_samples = []
    
    for i in range(n_bootstrap):
        # Bootstrap sample
        sample = data.sample(fraction=sample_frac, shuffle=True, seed=i)
        
        # Fit model
        result = rs.glm(formula, data=sample, family=family, exposure=exposure_col).fit()
        coef_samples.append(result.params)
    
    coef_array = np.vstack(coef_samples)
    
    # Compute coefficient variation
    coef_mean = np.mean(coef_array, axis=0)
    coef_std = np.std(coef_array, axis=0)
    coef_cv = np.abs(coef_std / (coef_mean + 1e-10))
    
    return {
        "mean": coef_mean,
        "std": coef_std,
        "cv": coef_cv,
        "stable": np.all(coef_cv < 0.5),  # CV < 50% considered stable
    }

stability = stability_check(
    data, 
    "claim_amount ~ vehicle_age + C(region)",
    "gamma",
    "exposure",
    n_bootstrap=10
)

print(f"Coefficients stable: {stability['stable']}")
print(f"Coefficient CVs: {stability['cv']}")
```

---

## Complete Workflow Summary

```python
def full_glm_workflow(data, response, exposure, categorical_vars, continuous_vars):
    """Complete GLM workflow with decision points."""
    
    results = {"steps": []}
    
    # Step 1: Explore
    exploration = rs.explore_data(
        data=data,
        response=response,
        exposure=exposure,
        categorical_factors=categorical_vars,
        continuous_factors=continuous_vars,
    )
    results["exploration"] = exploration
    results["steps"].append("1. Data exploration complete")
    
    # Step 2: Choose family
    y = data[response].to_numpy()
    family = suggest_family(y)
    results["suggested_family"] = family
    results["steps"].append(f"2. Suggested family: {family}")
    
    # Step 3: Initial fit
    formula = f"{response} ~ " + " + ".join(
        [f"C({v})" for v in categorical_vars] + continuous_vars
    )
    result = rs.glm(formula, data=data, family=family, exposure=exposure).fit()
    results["initial_result"] = result
    results["steps"].append(f"3. Initial model fit (converged={result.converged})")
    
    # Step 4: Diagnostics
    diagnostics = result.diagnostics(
        data=data,
        categorical_factors=categorical_vars,
        continuous_factors=continuous_vars,
    )
    results["diagnostics"] = diagnostics
    
    # Calibration check
    cal_check = check_calibration(diagnostics)
    results["calibration_check"] = cal_check
    results["steps"].append(f"4. Calibration: {'PASS' if cal_check['passed'] else 'FAIL'}")
    
    # Discrimination check  
    disc_check = check_discrimination(diagnostics)
    results["discrimination_check"] = disc_check
    results["steps"].append(f"5. Discrimination (Gini={disc_check.get('gini', 'N/A'):.3f})")
    
    # Step 5: Factor analysis
    problem_factors = identify_problem_factors(diagnostics)
    results["problem_factors"] = problem_factors
    results["steps"].append(f"6. Identified {len(problem_factors)} factors needing attention")
    
    # Step 6: Recommendations
    recommendations = []
    
    if not cal_check["passed"]:
        recommendations.append("Improve calibration: add interactions or non-linear terms")
    
    if disc_check.get("gini", 1.0) < 0.3:
        recommendations.append("Improve discrimination: add more predictive features")
    
    for pf in problem_factors:
        recommendations.append(f"Fix {pf['factor']}: {pf['recommendation']}")
    
    results["recommendations"] = recommendations
    
    return results


# Run workflow
workflow_results = full_glm_workflow(
    data=data,
    response="claim_amount",
    exposure="exposure",
    categorical_vars=["region", "vehicle_type"],
    continuous_vars=["vehicle_age", "driver_experience"],
)

print("\n=== WORKFLOW SUMMARY ===")
for step in workflow_results["steps"]:
    print(f"  {step}")

print("\n=== RECOMMENDATIONS ===")
for rec in workflow_results["recommendations"]:
    print(f"  → {rec}")
```

---

## Quick Reference: Diagnostic Decision Table

| Diagnostic | Good | Warning | Action |
|------------|------|---------|--------|
| **Convergence** | converged=True | converged=False | ↑ max_iter, add regularization |
| **Overall A/E** | 0.95-1.05 | <0.95 or >1.05 | Check model, add predictors |
| **H-L p-value** | >0.05 | <0.05 | Add interactions/splines |
| **Gini** | >0.30 | <0.20 | Add predictive features |
| **Dispersion** | 0.8-1.2 | >1.5 | Use quasi-family or NegBin |
| **Factor A/E spread** | <0.10 | >0.15 | Add splines/interactions |
| **Residual correlation** | <0.03 | >0.05 | Factor not captured well |

---

## Next Steps

- [Regularization Theory](../theory/regularization.md) — Mathematical background
- [Diagnostics API Reference](../api/diagnostics.md) — Full API documentation
- [Adding Interactions](../components/design-matrix.md) — How interactions work
- [Spline Basis Functions](../components/splines.md) — Non-linear effects
