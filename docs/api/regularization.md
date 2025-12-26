# Regularization API Reference

This page documents the regularization and variable selection functionality.

## lasso_path

Compute coefficients along a regularization path.

```python
rustystats.lasso_path(
    y,
    X,
    family="gaussian",
    n_alphas=100,
    alpha_min_ratio=0.001,
    l1_ratio=1.0,
    max_iterations=25,
    tolerance=1e-8,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `y` | array-like | required | Response variable |
| `X` | array-like | required | Design matrix |
| `family` | str | `"gaussian"` | Distribution family |
| `n_alphas` | int | `100` | Number of alpha values |
| `alpha_min_ratio` | float | `0.001` | Ratio of min to max alpha |
| `l1_ratio` | float | `1.0` | Elastic Net mixing |
| `max_iterations` | int | `25` | Max IRLS iterations per alpha |
| `tolerance` | float | `1e-8` | Convergence tolerance |

### Returns

`LassoPathResult` object.

### LassoPathResult

| Attribute | Type | Description |
|-----------|------|-------------|
| `alphas` | ndarray | Alpha values (descending) |
| `coefs` | ndarray | Coefficients (p × n_alphas) |
| `deviances` | ndarray | Deviance at each alpha |
| `n_nonzero` | ndarray | Non-zero count at each alpha |

### Methods

```python
path.plot()           # Plot coefficient paths
path.coef_at(alpha)   # Interpolate coefficients at specific alpha
```

### Example

```python
import rustystats as rs
import numpy as np

y = np.random.randn(100)
X = np.column_stack([np.ones(100), np.random.randn(100, 10)])

# Compute path
path = rs.lasso_path(y, X, family="gaussian", n_alphas=50)

# View results
print(f"Alphas: {path.alphas[:5]}...")
print(f"Coefficients shape: {path.coefs.shape}")

# Find alpha where 5 features are selected
for i, n in enumerate(path.n_nonzero):
    if n <= 5:
        print(f"Alpha={path.alphas[i]:.4f} selects {n} features")
        break

# Plot
path.plot()
```

---

## cv_glm

Cross-validation for optimal regularization parameter.

```python
rustystats.cv_glm(
    y,
    X,
    family="gaussian",
    cv=5,
    n_alphas=100,
    alpha_min_ratio=0.001,
    l1_ratio=1.0,
    scoring="deviance",
    max_iterations=25,
    tolerance=1e-8,
    seed=None,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `y` | array-like | required | Response variable |
| `X` | array-like | required | Design matrix |
| `family` | str | `"gaussian"` | Distribution family |
| `cv` | int | `5` | Number of CV folds |
| `n_alphas` | int | `100` | Number of alpha values |
| `alpha_min_ratio` | float | `0.001` | Ratio of min to max alpha |
| `l1_ratio` | float | `1.0` | Elastic Net mixing |
| `scoring` | str | `"deviance"` | CV scoring metric |
| `seed` | int | `None` | Random seed for fold assignment |

### Returns

`CVGLMResult` object.

### CVGLMResult

| Attribute | Type | Description |
|-----------|------|-------------|
| `alphas` | ndarray | Alpha values tested |
| `cv_scores` | ndarray | Mean CV score at each alpha |
| `cv_scores_std` | ndarray | Std of CV score at each alpha |
| `alpha_best` | float | Alpha with minimum CV score |
| `alpha_1se` | float | Largest alpha within 1 SE of minimum |
| `coef_best` | ndarray | Coefficients at alpha_best |
| `coef_1se` | ndarray | Coefficients at alpha_1se |
| `n_nonzero_best` | int | Non-zero count at alpha_best |
| `n_nonzero_1se` | int | Non-zero count at alpha_1se |

### Methods

```python
cv_result.plot()              # Plot CV curve with error bars
cv_result.refit(y, X)         # Refit model at alpha_best
cv_result.refit_1se(y, X)     # Refit model at alpha_1se
```

### Example

```python
import rustystats as rs
import numpy as np

np.random.seed(42)
n, p = 500, 20
X = np.column_stack([np.ones(n), np.random.randn(n, p)])
beta_true = np.zeros(p + 1)
beta_true[:6] = [1.0, 0.5, -0.5, 0.3, -0.3, 0.2]
y = np.random.poisson(np.exp(X @ beta_true))

# Cross-validation
cv_result = rs.cv_glm(y, X, family="poisson", cv=5, l1_ratio=1.0)

print(f"Best alpha: {cv_result.alpha_best:.4f}")
print(f"Features at best: {cv_result.n_nonzero_best}")
print(f"1-SE alpha: {cv_result.alpha_1se:.4f}")
print(f"Features at 1-SE: {cv_result.n_nonzero_1se}")

# Plot CV curve
cv_result.plot()

# Refit final model
final_result = cv_result.refit_1se(y, X)
print(f"Final model has {final_result.n_nonzero()} features")
```

---

## cv_lasso

Convenience function for Lasso cross-validation (l1_ratio=1.0).

```python
rustystats.cv_lasso(y, X, family="gaussian", cv=5, **kwargs)
```

Equivalent to `cv_glm(..., l1_ratio=1.0)`.

---

## cv_ridge

Convenience function for Ridge cross-validation (l1_ratio=0.0).

```python
rustystats.cv_ridge(y, X, family="gaussian", cv=5, **kwargs)
```

Equivalent to `cv_glm(..., l1_ratio=0.0)`.

---

## Regularized fit_glm

The main `fit_glm` function supports regularization via `alpha` and `l1_ratio` parameters.

### Lasso (L1)

```python
result = rs.fit_glm(y, X, family="poisson", alpha=0.1, l1_ratio=1.0)
```

### Ridge (L2)

```python
result = rs.fit_glm(y, X, family="gaussian", alpha=0.1, l1_ratio=0.0)
```

### Elastic Net

```python
result = rs.fit_glm(y, X, family="gaussian", alpha=0.1, l1_ratio=0.5)
```

### Regularized Results

Regularized models have additional methods:

```python
result.n_nonzero()         # Number of non-zero coefficients
result.selected_features() # Indices of selected features
```

---

## Choosing Alpha

### Using CV

```python
cv_result = rs.cv_glm(y, X, family="poisson", l1_ratio=1.0, cv=5)

# Option 1: Minimum CV error
result = rs.fit_glm(y, X, family="poisson", 
                     alpha=cv_result.alpha_best, l1_ratio=1.0)

# Option 2: 1-SE rule (more parsimonious)
result = rs.fit_glm(y, X, family="poisson",
                     alpha=cv_result.alpha_1se, l1_ratio=1.0)
```

### Manual Selection

```python
# Trace the path
path = rs.lasso_path(y, X, family="poisson")

# Find alpha for desired sparsity
target_features = 10
for i, n in enumerate(path.n_nonzero):
    if n <= target_features:
        alpha = path.alphas[i]
        break

result = rs.fit_glm(y, X, family="poisson", alpha=alpha, l1_ratio=1.0)
```

---

## Complete Workflow Example

```python
import rustystats as rs
import numpy as np

# Generate data
np.random.seed(42)
n, p = 1000, 50
X = np.column_stack([np.ones(n), np.random.randn(n, p)])

# True sparse model: only 5 features matter
beta_true = np.zeros(p + 1)
beta_true[:6] = [2.0, 0.5, -0.3, 0.2, -0.1, 0.1]
y = np.random.poisson(np.exp(X @ beta_true))

# Step 1: Cross-validation
print("Running cross-validation...")
cv_result = rs.cv_glm(y, X, family="poisson", l1_ratio=1.0, cv=5)
print(f"Best α: {cv_result.alpha_best:.4f} ({cv_result.n_nonzero_best} features)")
print(f"1-SE α: {cv_result.alpha_1se:.4f} ({cv_result.n_nonzero_1se} features)")

# Step 2: Visualize path
print("\nCoefficient path:")
path = rs.lasso_path(y, X, family="poisson", n_alphas=50)
# path.plot()

# Step 3: Fit final model with 1-SE rule
print("\nFitting final model...")
result = rs.fit_glm(y, X, family="poisson",
                     alpha=cv_result.alpha_1se, l1_ratio=1.0)

# Step 4: Evaluate
print(f"\nFinal model:")
print(f"  Selected features: {result.n_nonzero()} of {p + 1}")
print(f"  Deviance: {result.deviance:.2f}")
print(f"  Non-zero indices: {result.selected_features()}")

# Step 5: Compare to true
selected = set(result.selected_features())
true_nonzero = set(np.where(beta_true != 0)[0])
print(f"\nTrue positives: {len(selected & true_nonzero)}")
print(f"False positives: {len(selected - true_nonzero)}")
print(f"False negatives: {len(true_nonzero - selected)}")
```
