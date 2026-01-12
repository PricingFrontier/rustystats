# Regularization

Automatic regularization strength selection via cross-validation.

## Quick Start

```python
import rustystats as rs

# Just specify regularization type - cv=5 is automatic
result = rs.glm("ClaimCount ~ VehAge + BonusMalus + TE(Region)", data,
                family="poisson", offset="Exposure").fit(
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
- **~8s for 500k rows, 5 folds, 20 alphas**
