# Model Diagnostics

After fitting a GLM, you need to assess whether the model fits well and identify areas for improvement. This chapter covers the diagnostic tools available in RustyStats.

## Residuals

Residuals measure the discrepancy between observed and fitted values. GLMs have several types of residuals, each with different properties.

### Response Residuals

The simplest residual - just the difference:

\[
r_i^{\text{response}} = y_i - \hat{\mu}_i
\]

```python
resid = result.resid_response()
```

**Properties:**
- Easy to interpret
- Not standardized (scale depends on \(\mu\))
- Variance is not constant

### Pearson Residuals

Standardized by the variance function:

\[
r_i^{\text{Pearson}} = \frac{y_i - \hat{\mu}_i}{\sqrt{V(\hat{\mu}_i)}}
\]

```python
resid = result.resid_pearson()
```

**Properties:**
- Approximately standardized (unit variance under the model)
- Useful for detecting outliers
- Sum of squares = Pearson \(\chi^2\) statistic

### Deviance Residuals

Based on the contribution to deviance:

\[
r_i^{\text{deviance}} = \text{sign}(y_i - \hat{\mu}_i) \sqrt{d_i}
\]

where \(d_i\) is the unit deviance.

```python
resid = result.resid_deviance()
```

**Properties:**
- Sum of squares = Model deviance
- More normally distributed than Pearson residuals
- Preferred for most diagnostic plots

### Working Residuals

Used internally by IRLS:

\[
r_i^{\text{working}} = (y_i - \hat{\mu}_i) \cdot g'(\hat{\mu}_i)
\]

```python
resid = result.resid_working()
```

**Properties:**
- On the linear predictor scale
- Useful for partial residual plots

## Goodness-of-Fit Statistics

### Deviance

The deviance measures overall model fit:

\[
D = 2[\ell(\text{saturated}) - \ell(\text{fitted})]
\]

```python
deviance = result.deviance
```

Lower deviance = better fit. For comparing nested models:

\[
D_{\text{reduced}} - D_{\text{full}} \sim \chi^2_{p_{\text{full}} - p_{\text{reduced}}}
\]

### Null Deviance

Deviance of the intercept-only model:

```python
null_dev = result.null_deviance()
```

**Pseudo-R²** (one of many definitions):
\[
R^2_{\text{pseudo}} = 1 - \frac{D}{D_{\text{null}}}
\]

### Pearson Chi-Squared

\[
\chi^2_{\text{Pearson}} = \sum_i \frac{(y_i - \hat{\mu}_i)^2}{V(\hat{\mu}_i)}
\]

```python
chi2 = result.pearson_chi2()
```

### Dispersion Estimation

For families with unknown dispersion (Gaussian, Gamma, Quasi-families):

\[
\hat{\phi}_{\text{Pearson}} = \frac{\chi^2_{\text{Pearson}}}{n - p}
\]

\[
\hat{\phi}_{\text{deviance}} = \frac{D}{n - p}
\]

```python
phi_deviance = result.scale()
phi_pearson = result.scale_pearson()
```

### Checking for Overdispersion

For Poisson/Binomial, dispersion should be ≈ 1:

```python
dispersion_ratio = result.pearson_chi2() / result.df_resid
print(f"Dispersion ratio: {dispersion_ratio:.2f}")

# > 1.5 suggests overdispersion
# < 0.7 suggests underdispersion
```

## Information Criteria

### Log-Likelihood

```python
ll = result.llf()
```

### AIC (Akaike Information Criterion)

\[
\text{AIC} = -2\ell + 2p
\]

Balances fit (likelihood) with complexity (number of parameters).

```python
aic = result.aic()
```

**Lower is better.** AIC favors parsimony.

### BIC (Bayesian Information Criterion)

\[
\text{BIC} = -2\ell + p \log(n)
\]

```python
bic = result.bic()
```

**More conservative** than AIC for large samples (penalizes complexity more).

### When to Use Each

| Criterion | Penalty | Best For |
|-----------|---------|----------|
| AIC | 2p | Prediction, model averaging |
| BIC | p log(n) | Model selection, large samples |

## Calibration Diagnostics

Calibration measures how well predicted probabilities/means match observed values.

### Actual vs. Expected Ratio

\[
\text{A/E} = \frac{\sum y_i}{\sum \hat{\mu}_i}
\]

```python
diagnostics = result.diagnostics(data=data, categorical_factors=["Region"])
print(f"Overall A/E: {diagnostics.calibration['actual_expected_ratio']:.3f}")
```

**Interpretation:**
- A/E = 1.0: Perfect overall calibration
- A/E > 1.0: Model underpredicts
- A/E < 1.0: Model overpredicts

### Calibration by Decile

Split predictions into 10 groups and compare A/E in each:

```python
for decile in diagnostics.calibration['by_decile']:
    print(f"Decile {decile['decile']}: A/E = {decile['ae_ratio']:.3f}")
```

Good calibration: A/E ≈ 1 in all deciles.

### Hosmer-Lemeshow Test

Formal test for calibration (Binomial models):

\[
H = \sum_{g=1}^{G} \frac{(O_g - E_g)^2}{E_g(1 - E_g/n_g)}
\]

Under \(H_0\) (good calibration): \(H \sim \chi^2_{G-2}\)

## Discrimination Diagnostics

Discrimination measures how well the model separates high and low values.

### Gini Coefficient

Measures the area between the Lorenz curve and the line of equality:

\[
\text{Gini} = 2 \times \text{AUC} - 1
\]

```python
gini = diagnostics.discrimination['gini_coefficient']
```

**Interpretation:**
- Gini = 0: No discrimination (random predictions)
- Gini = 1: Perfect discrimination
- Typical values: 0.3-0.5 for insurance models

### Lorenz Curve

Plots cumulative % of response vs. cumulative % of exposure, ordered by predicted risk:

```python
lorenz = diagnostics.discrimination['lorenz_curve']
# Plot: x = cumulative exposure %, y = cumulative claims %
```

### Lift

How much better is the model than random in the top decile?

\[
\text{Lift} = \frac{\text{Response rate in top decile}}{\text{Overall response rate}}
\]

## Factor Diagnostics

Analyze model performance by individual factors.

### A/E by Factor Level

For categorical factors:

```python
for factor in diagnostics.factors:
    if factor.factor_type == "categorical":
        for level in factor.actual_vs_expected:
            print(f"{factor.name}={level['level']}: A/E={level['ae_ratio']:.3f}")
```

### Residual Patterns

Check if residuals are correlated with factors:

```python
for factor in diagnostics.factors:
    corr = factor.residual_pattern['correlation_with_residuals']
    if abs(corr) > 0.05:
        print(f"Warning: {factor.name} has residual correlation {corr:.3f}")
```

High correlation suggests the factor effect is misspecified.

## Interaction Detection

RustyStats can automatically detect potential interactions:

```python
diagnostics = result.diagnostics(
    data=data,
    categorical_factors=["Region", "VehBrand"],
    continuous_factors=["Age"],
)

for ic in diagnostics.interaction_candidates:
    print(f"Consider: {ic['factor1']} × {ic['factor2']} (strength={ic['strength']:.3f})")
```

The algorithm uses greedy residual-based detection to find factor pairs that explain residual variance.

## Pre-Fit Data Exploration

Explore data before fitting any model:

```python
exploration = rs.explore_data(
    data=data,
    response="ClaimNb",
    categorical_factors=["Region", "VehBrand"],
    continuous_factors=["Age", "VehPower"],
    exposure="Exposure",
    family="poisson",
    detect_interactions=True,
)

print(exploration.response_stats)
print(exploration.to_json())
```

## JSON Export for LLM Integration

Export diagnostics in compact JSON format:

```python
json_str = result.diagnostics_json(
    data=data,
    categorical_factors=["Region"],
    continuous_factors=["Age"],
)

# Feed to LLM for analysis
response = llm.analyze(f"Analyze this model: {json_str}")
```

## Diagnostic Plots (External)

RustyStats focuses on computation. For visualization, use matplotlib/plotly:

```python
import matplotlib.pyplot as plt

# Residual vs. Fitted
plt.scatter(result.fittedvalues, result.resid_deviance())
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Deviance Residuals')

# Q-Q plot
from scipy import stats
stats.probplot(result.resid_deviance(), plot=plt)

# Calibration plot
# ... using diagnostics.calibration data
```

## Summary

Key diagnostics workflow:

1. **Check convergence**: `result.converged`, `result.iterations`
2. **Assess overall fit**: Deviance, AIC, BIC
3. **Check dispersion**: Pearson χ²/df ≈ 1
4. **Examine residuals**: Patterns suggest model misspecification
5. **Verify calibration**: A/E by decile
6. **Assess discrimination**: Gini coefficient
7. **Factor analysis**: A/E by level, residual correlations
8. **Detect interactions**: Automated candidates
