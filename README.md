# RustyStats 🦀📊

**High-performance Generalized Linear Models with a Rust backend and Python API**

Built for actuarial applications. Fits 678K rows in ~1 second.

## Features

- **Fast** — Parallel IRLS solver in Rust (Rayon)
- **Complete** — Families, regularization, inference, diagnostics
- **Flexible** — R-style formulas with interactions and splines
- **Minimal** — Core requires only `numpy`

## Installation

```bash
# Development install
git clone https://github.com/PricingFrontier/rustystats.git
cd rustystats
uv run maturin develop --release

# Run tests
uv run pytest tests/python/
```

## Quick Start

### Formula API (recommended)

```python
import rustystats as rs
import polars as pl

# Load data
data = pl.read_parquet("insurance.parquet")

# Fit a Poisson GLM for claim frequency
result = rs.glm(
    "ClaimCount ~ VehAge + VehPower + C(Area) + C(Region)",
    data=data,
    family="poisson",
    offset="Exposure"
).fit()

# View results
print(result.summary())
print(result.relativities())  # exp(coef) for pricing
```

### Array API

```python
import rustystats as rs
import numpy as np

# Fit with numpy arrays
result = rs.fit_glm(
    y, X,
    family="poisson",
    offset=np.log(exposure),
    alpha=0.1,        # Regularization strength
    l1_ratio=0.5      # Elastic net mixing (0=Ridge, 1=Lasso)
)

print(f"Coefficients: {result.params}")
print(f"Deviance: {result.deviance}")
```

## Families & Links

| Family | Default Link | Use Case |
|--------|--------------|----------|
| `gaussian` | identity | Linear regression |
| `poisson` | log | Claim frequency |
| `binomial` | logit | Binary outcomes |
| `gamma` | log | Claim severity |
| `tweedie` | log | Pure premium (var_power=1.5) |

## Formula Syntax

```python
# Main effects
"y ~ x1 + x2 + C(category)"

# Interactions
"y ~ x1*x2"              # x1 + x2 + x1:x2
"y ~ C(area):age"        # Area-specific age effects
"y ~ C(area)*C(brand)"   # Categorical × categorical

# Splines (non-linear effects)
"y ~ bs(age, df=5)"      # B-spline basis
"y ~ ns(income, df=4)"   # Natural spline (better extrapolation)

# Combined
"y ~ bs(age, df=5) + C(region)*income + ns(vehicle_age, df=3)"
```

## Regularization

```python
# Ridge (L2) - shrinks coefficients
result = rs.fit_glm(y, X, family="poisson", alpha=0.1, l1_ratio=0.0)

# Lasso (L1) - variable selection
result = rs.fit_glm(y, X, family="poisson", alpha=0.1, l1_ratio=1.0)

# Elastic Net - mix of both
result = rs.fit_glm(y, X, family="poisson", alpha=0.1, l1_ratio=0.5)

# Cross-validation for optimal alpha
cv = rs.cv_glm(y, X, family="poisson", l1_ratio=1.0, cv=5)
print(f"Best alpha: {cv.alpha_best}")

# Coefficient path
path = rs.lasso_path(y, X, family="poisson", n_alphas=50)
```

## Results Methods

```python
result.params              # Coefficients
result.bse()               # Standard errors
result.pvalues()           # P-values
result.conf_int()          # Confidence intervals
result.fittedvalues        # Predicted values

# Robust standard errors (sandwich estimators)
result.bse_robust("HC1")   # HC0, HC1, HC2, HC3

# Diagnostics
result.deviance            # Model deviance
result.aic()               # Akaike Information Criterion
result.bic()               # Bayesian Information Criterion
result.resid_pearson()     # Pearson residuals
result.resid_deviance()    # Deviance residuals
```

## Performance

| Dataset | RustyStats | Statsmodels |
|---------|------------|-------------|
| 678K rows, Poisson | ~1.0s | ~5-10s |
| 678K rows, Lasso | ~2.8s | N/A for GLMs |

## Dependencies

**Required:** `numpy`

**Optional:**
- `polars` — DataFrame support for formula API
- `pandas` — Only for `summary_df()` output

## Documentation

See [SUMMARY.md](SUMMARY.md) for detailed API documentation and architecture.

## License

MIT
