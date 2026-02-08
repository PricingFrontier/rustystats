# Regularization

Automatic regularization strength selection via cross-validation.

## Quick Start

```python
import rustystats as rs

# Just specify regularization type - cv=5 is automatic
result = rs.glm_dict(response="ClaimCount",
    terms={"VehAge": {"type": "linear"}, "BonusMalus": {"type": "linear"}, "Region": {"type": "target_encoding"}},
    data=data, family="poisson", offset="Exposure").fit(
    regularization="ridge"  # "ridge", "lasso", or "elastic_net"
)

print(f"Selected alpha: {result.alpha}")
print(f"CV deviance: {result.cv_deviance}")
```

## When to Use

- **Ridge**: Multicollinearity, want to keep all predictors
- **Lasso**: Variable selection, sparse models
- **Elastic Net**: Groups of correlated predictors

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `regularization` | - | `"ridge"`, `"lasso"`, or `"elastic_net"` |
| `selection` | `"min"` | `"min"` (best fit) or `"1se"` (more conservative) |
| `cv` | `5` | Number of CV folds |
| `n_alphas` | `20` | Alpha grid size |
| `cv_seed` | `None` | Seed for reproducible folds |

## Selection Methods

- **"min"**: Alpha with minimum CV deviance (best fit)
- **"1se"**: Largest alpha within 1 SE of minimum (recommended for production)

## Explicit Alpha (Skip CV)

```python
result = model.fit(alpha=0.1, l1_ratio=0.0)  # Ridge
result = model.fit(alpha=0.1, l1_ratio=1.0)  # Lasso
result = model.fit(alpha=0.1, l1_ratio=0.5)  # Elastic Net
```

## Result Attributes

```python
result.alpha               # Selected regularization strength
result.cv_deviance         # CV deviance at selected alpha
result.cv_deviance_se      # Standard error
result.regularization_type # "ridge", "lasso", "elastic_net", or "none"
result.n_cv_folds          # Number of folds used
result.regularization_path # Full path results (list of dicts)
```

## Diagnostics Output

Regularization info is included in diagnostics when alpha > 0:

```json
{
  "model_summary": {
    "formula": "y ~ x1 + x2",
    "regularization": {
      "type": "ridge",
      "alpha": 0.1,
      "l1_ratio": 0.0,
      "cv_deviance": 0.315,
      "cv_folds": 5,
      "selection": "min"
    }
  }
}
```

## Best Practices

```python
# Production: use "1se" for more conservative selection
result = model.fit(regularization="ridge", selection="1se")

# Reproducibility: set seed
result = model.fit(regularization="ridge", cv_seed=42)
```

## Performance

- **Parallel CV**: Folds fitted in parallel via Rust/Rayon
- **Warm-start**: Coefficients from α[i] initialize α[i+1] for faster convergence
- **~2s for 30k rows × 50 predictors, 5 folds, 20 alphas** (Ridge)
- **~5s for 30k rows × 50 predictors, 5 folds, 20 alphas** (Lasso/Elastic Net)


## Complete Example

```python
import rustystats as rs
import polars as pl

# Load data
df = pl.read_csv("insurance.csv")

# Ridge with automatic CV
ridge_result = rs.glm_dict(
    response="ClaimAmount",
    terms={"Age": {"type": "linear"}, "VehAge": {"type": "linear"}, "BonusMalus": {"type": "linear"}, "Density": {"type": "linear"}},
    data=df,
    family="gamma"
).fit(
    cv=5,
    regularization="ridge",
    n_alphas=50,
    selection="1se",
    cv_seed=42
)

print(f"Selected alpha: {ridge_result.alpha:.4f}")
print(f"CV deviance: {ridge_result.cv_deviance:.4f}")
print(ridge_result.summary())

# Lasso for variable selection
lasso_result = rs.glm_dict(
    response="ClaimCount",
    terms={f"x{i}": {"type": "linear"} for i in range(100)},
    data=df,
    family="poisson"
).fit(
    cv=5,
    regularization="lasso",
    n_alphas=30
)

# See which coefficients are non-zero
nonzero = [(name, coef) for name, coef in 
           zip(lasso_result.feature_names, lasso_result.params) 
           if abs(coef) > 1e-6]
print(f"Selected {len(nonzero)} features")

# Elastic Net for correlated groups
enet_result = rs.glm_dict(response="y", terms={f"x{i}": {"type": "linear"} for i in range(1, 6)}, data=df).fit(
    cv=5,
    regularization="elastic_net",
    l1_ratio=0.5,  # 50% L1, 50% L2
    n_alphas=20
)
```

## Supported Families

CV regularization works with all GLM families:

| Family | Ridge | Lasso | Elastic Net |
|--------|-------|-------|-------------|
| Gaussian | ✓ | ✓ | ✓ |
| Poisson | ✓ | ✓ | ✓ |
| Binomial | ✓ | ✓ | ✓ |
| Gamma | ✓ | ✓ | ✓ |
| Tweedie | ✓ | ✓ | ✓ |
| NegBinomial | ✓ | ✓ | ✓ |

## Important: Standard Errors for Penalized Models

**For actuarial and statistical inference applications:**

Standard errors returned by Lasso and Elastic Net models are **approximate**. The covariance
matrix is computed using only non-zero coefficients, which does not account for the selection
bias introduced by penalization.

For rigorous inference on regularized models, consider:

1. **Bootstrap confidence intervals** — Resample and refit to get empirical distributions
2. **De-biased Lasso methods** — Available in specialized packages
3. **Post-selection inference** — Techniques that account for variable selection

Ridge regression standard errors are more reliable since no variable selection occurs.

```python
# For rigorous inference, use bootstrap
from sklearn.utils import resample

bootstrap_coefs = []
for _ in range(1000):
    idx = resample(range(len(df)))
    boot_df = df[idx]
    result = rs.glm_dict(response="y", terms=terms, data=boot_df, family="poisson").fit(
        regularization="lasso", cv=5
    )
    bootstrap_coefs.append(result.params)

# Compute percentile confidence intervals from bootstrap_coefs
```
