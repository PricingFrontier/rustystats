# Quick Start

This guide walks through the essential RustyStats functionality in 10 minutes.

## Your First GLM

### Using NumPy Arrays

```python
import rustystats as rs
import numpy as np

# Generate sample data
np.random.seed(42)
n = 1000
x1 = np.random.normal(0, 1, n)
x2 = np.random.normal(0, 1, n)

# True model: y ~ Poisson(exp(0.5 + 0.3*x1 - 0.2*x2))
eta = 0.5 + 0.3 * x1 - 0.2 * x2
y = np.random.poisson(np.exp(eta))

# Build design matrix (include intercept!)
X = np.column_stack([np.ones(n), x1, x2])

# Fit the model
result = rs.fit_glm(y, X, family="poisson")

# View results
print(f"Coefficients: {result.params}")      # [0.50, 0.30, -0.20] approximately
print(f"Standard Errors: {result.bse()}")
print(f"P-values: {result.pvalues()}")
print(f"Deviance: {result.deviance:.2f}")
```

### Using the Formula API

The formula API is more intuitive for real-world data:

```python
import rustystats as rs
import polars as pl

# Load data
data = pl.DataFrame({
    "claims": [0, 1, 0, 2, 0, 1, 3, 0, 1, 0],
    "age": [25, 35, 45, 55, 28, 38, 48, 58, 32, 42],
    "region": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
    "exposure": [1.0, 1.0, 0.5, 1.0, 0.8, 1.0, 1.2, 0.9, 1.0, 0.7]
})

# Fit model with formula
result = rs.glm(
    formula="claims ~ age + C(region)",
    data=data,
    family="poisson",
    offset="exposure"  # log(exposure) applied automatically
).fit()

# Rich output
print(result.summary())
```

## Understanding the Output

```python
# Coefficients and inference
result.params              # Coefficient estimates
result.bse()               # Standard errors
result.tvalues()           # z-statistics (coef / SE)
result.pvalues()           # Two-sided p-values
result.conf_int(0.05)      # 95% confidence intervals

# Model fit
result.deviance            # Model deviance
result.aic()               # Akaike Information Criterion
result.bic()               # Bayesian Information Criterion
result.llf()               # Log-likelihood

# Predictions
result.fittedvalues        # Fitted values (μ)
result.linear_predictor    # Linear predictor (η = Xβ)
```

## Choosing a Family

| Data Type | Family | Example |
|-----------|--------|---------|
| Continuous (any value) | `"gaussian"` | Claim amounts, temperatures |
| Counts (0, 1, 2, ...) | `"poisson"` | Claim frequency, event counts |
| Binary (0 or 1) | `"binomial"` | Claim occurrence, churn |
| Positive continuous | `"gamma"` | Claim severity, durations |
| Mixed zeros + positive | `"tweedie"` | Pure premium (frequency × severity) |

```python
# Examples
result = rs.fit_glm(y, X, family="gaussian")     # Linear regression
result = rs.fit_glm(y, X, family="poisson")      # Count data
result = rs.fit_glm(y, X, family="binomial")     # Binary outcomes
result = rs.fit_glm(y, X, family="gamma")        # Positive continuous
result = rs.fit_glm(y, X, family="tweedie", var_power=1.5)  # Pure premium
```

## Working with Categorical Variables

Use `C()` to mark categorical variables in formulas:

```python
result = rs.glm(
    formula="claims ~ age + C(region) + C(vehicle_type)",
    data=data,
    family="poisson"
).fit()

# View relativities (exp(coef) for multiplicative interpretation)
print(result.relativities())
```

## Adding Regularization

```python
# Lasso (L1) - variable selection
result = rs.fit_glm(y, X, family="poisson", alpha=0.1, l1_ratio=1.0)
print(f"Non-zero coefficients: {result.n_nonzero()}")

# Ridge (L2) - shrinkage without selection
result = rs.fit_glm(y, X, family="gaussian", alpha=0.1, l1_ratio=0.0)

# Elastic Net - mix of both
result = rs.fit_glm(y, X, family="gaussian", alpha=0.1, l1_ratio=0.5)

# Cross-validation for optimal alpha
cv_result = rs.cv_glm(y, X, family="poisson", l1_ratio=1.0, cv=5)
print(f"Best alpha: {cv_result.alpha_best}")
```

## Non-linear Effects with Splines

```python
# Add smooth age effect
result = rs.glm(
    formula="claims ~ bs(age, df=5) + C(region)",
    data=data,
    family="poisson"
).fit()

# Natural splines (linear at boundaries - better extrapolation)
result = rs.glm(
    formula="claims ~ ns(age, df=4) + C(region)",
    data=data,
    family="poisson"
).fit()
```

## Handling Overdispersion

When variance exceeds what the model predicts:

```python
# Check for overdispersion
result = rs.fit_glm(y, X, family="poisson")
dispersion = result.pearson_chi2() / result.df_resid
print(f"Dispersion ratio: {dispersion:.2f}")  # >> 1 indicates overdispersion

# Use QuasiPoisson for inflated standard errors
result_quasi = rs.fit_glm(y, X, family="quasipoisson")

# Or Negative Binomial for proper likelihood
result_nb = rs.fit_glm(y, X, family="negbinomial", theta=1.0)
```

## Next Steps

- [GLM Theory](../theory/glm-intro.md) - Understand the mathematics
- [Architecture](../architecture/overview.md) - How the code is organized
- [Python API](../api/array-api.md) - Complete API reference
