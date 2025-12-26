# Array API Reference

The array API provides low-level access to GLM fitting using NumPy arrays directly.

## fit_glm

Fit a Generalized Linear Model.

```python
rustystats.fit_glm(
    y,
    X,
    family="gaussian",
    link=None,
    offset=None,
    weights=None,
    alpha=0.0,
    l1_ratio=1.0,
    max_iterations=25,
    tolerance=1e-8,
    theta=None,
    var_power=1.5,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `y` | array-like | required | Response variable (n,) |
| `X` | array-like | required | Design matrix (n, p), include intercept column |
| `family` | str | `"gaussian"` | Distribution family |
| `link` | str | `None` | Link function (uses family default if None) |
| `offset` | array-like | `None` | Offset term (n,) |
| `weights` | array-like | `None` | Prior weights (n,) |
| `alpha` | float | `0.0` | Regularization strength |
| `l1_ratio` | float | `1.0` | Elastic Net mixing (1=Lasso, 0=Ridge) |
| `max_iterations` | int | `25` | Maximum IRLS iterations |
| `tolerance` | float | `1e-8` | Convergence tolerance |
| `theta` | float | `None` | Negative Binomial dispersion |
| `var_power` | float | `1.5` | Tweedie variance power |

### Family Options

| Value | Description |
|-------|-------------|
| `"gaussian"` | Normal/Gaussian - continuous data |
| `"poisson"` | Poisson - count data |
| `"binomial"` | Binomial - binary/proportion data |
| `"gamma"` | Gamma - positive continuous |
| `"tweedie"` | Tweedie - mixed zeros and positives |
| `"quasipoisson"` | QuasiPoisson - overdispersed counts |
| `"quasibinomial"` | QuasiBinomial - overdispersed binary |
| `"negbinomial"` | Negative Binomial - overdispersed counts |

### Returns

`GLMResults` object with fitted model.

### Examples

```python
import rustystats as rs
import numpy as np

# Basic Gaussian model
y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
X = np.column_stack([np.ones(5), [1, 2, 3, 4, 5]])
result = rs.fit_glm(y, X, family="gaussian")

# Poisson with offset
claims = np.array([0, 1, 2, 0, 1])
exposure = np.array([1.0, 1.0, 2.0, 0.5, 1.5])
X = np.column_stack([np.ones(5), np.random.randn(5)])
result = rs.fit_glm(claims, X, family="poisson", offset=np.log(exposure))

# Regularized model
result = rs.fit_glm(y, X, family="gaussian", alpha=0.1, l1_ratio=0.5)

# Negative Binomial with theta
result = rs.fit_glm(counts, X, family="negbinomial", theta=2.0)
```

---

## fit_negbinomial

Fit Negative Binomial GLM with automatic θ estimation.

```python
rustystats.fit_negbinomial(
    y,
    X,
    offset=None,
    weights=None,
    max_iterations=25,
    tolerance=1e-8,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `y` | array-like | required | Count response (n,) |
| `X` | array-like | required | Design matrix (n, p) |
| `offset` | array-like | `None` | Offset term |
| `weights` | array-like | `None` | Prior weights |

### Returns

`GLMResults` with estimated θ in `result.family`.

### Example

```python
result = rs.fit_negbinomial(counts, X)
print(result.family)  # "NegativeBinomial(theta=2.34)"
```

---

## GLMResults

Results object returned by `fit_glm`.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `params` | ndarray | Fitted coefficients (p,) |
| `fittedvalues` | ndarray | Predicted means (n,) |
| `linear_predictor` | ndarray | Linear predictor η = Xβ (n,) |
| `deviance` | float | Model deviance |
| `iterations` | int | Number of IRLS iterations |
| `converged` | bool | Whether fitting converged |
| `nobs` | int | Number of observations |
| `df_resid` | int | Residual degrees of freedom (n - p) |
| `df_model` | int | Model degrees of freedom (p - 1) |
| `family` | str | Family name |

### Inference Methods

```python
# Standard errors
result.bse()           # Standard errors of coefficients

# Test statistics
result.tvalues()       # z-statistics (coef / SE)
result.pvalues()       # Two-sided p-values

# Confidence intervals
result.conf_int(alpha=0.05)  # Returns (lower, upper) arrays
```

### Robust Inference Methods

```python
# Robust standard errors (sandwich estimators)
result.bse_robust(hc_type="HC1")     # HC0, HC1, HC2, HC3
result.tvalues_robust(hc_type="HC1")
result.pvalues_robust(hc_type="HC1")
result.conf_int_robust(alpha=0.05, hc_type="HC1")

# Full covariance matrix
result.cov_params()        # Model-based covariance
result.cov_robust("HC1")   # Robust covariance
```

### Residual Methods

```python
result.resid_response()   # y - μ
result.resid_pearson()    # (y - μ) / √V(μ)
result.resid_deviance()   # sign(y-μ) × √d
result.resid_working()    # (y - μ) × g'(μ)
```

### Fit Statistics

```python
result.llf()              # Log-likelihood
result.aic()              # Akaike Information Criterion
result.bic()              # Bayesian Information Criterion
result.null_deviance()    # Deviance of intercept-only model
result.pearson_chi2()     # Pearson chi-squared statistic
result.scale()            # Dispersion (deviance-based)
result.scale_pearson()    # Dispersion (Pearson-based)
```

### Regularization Methods

```python
result.n_nonzero()         # Number of non-zero coefficients
result.selected_features() # Indices of non-zero coefficients
```

### Example

```python
result = rs.fit_glm(y, X, family="poisson")

print(f"Coefficients: {result.params}")
print(f"Standard Errors: {result.bse()}")
print(f"P-values: {result.pvalues()}")
print(f"AIC: {result.aic():.2f}")
print(f"Converged: {result.converged} in {result.iterations} iterations")

# Check significance
for i, (coef, pval) in enumerate(zip(result.params, result.pvalues())):
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
    print(f"β{i}: {coef:.4f} (p={pval:.4f}) {sig}")
```

---

## Spline Functions

### bs

B-spline basis matrix.

```python
rustystats.bs(
    x,
    df=None,
    knots=None,
    degree=3,
    boundary_knots=None,
    include_intercept=False,
)
```

### ns

Natural spline basis matrix.

```python
rustystats.ns(
    x,
    df=None,
    knots=None,
    boundary_knots=None,
    include_intercept=False,
)
```

### Example

```python
import rustystats as rs
import numpy as np

x = np.linspace(0, 10, 100)

# B-spline basis
basis_bs = rs.bs(x, df=5)
print(basis_bs.shape)  # (100, 4) - df-1 columns without intercept

# Natural spline basis
basis_ns = rs.ns(x, df=5)
print(basis_ns.shape)  # (100, 4)

# Custom boundary knots
basis = rs.bs(x, df=6, boundary_knots=(0, 10))
```

---

## Target Encoding

### target_encode

Apply CatBoost-style target encoding.

```python
rustystats.target_encode(
    categories,
    target,
    column_name="category",
    prior_weight=1.0,
    n_permutations=4,
    seed=None,
)
```

### Returns

Tuple: `(encoded_values, column_name, prior, level_stats)`

### apply_target_encoding

Apply learned encoding to new data.

```python
rustystats.apply_target_encoding(
    categories,
    level_stats,
    prior,
)
```

### TargetEncoder

Sklearn-style interface.

```python
encoder = rs.TargetEncoder(prior_weight=1.0, n_permutations=4)
train_encoded = encoder.fit_transform(train_categories, train_target)
test_encoded = encoder.transform(test_categories)
```

### Example

```python
import rustystats as rs
import numpy as np

categories = ["A", "B", "A", "C", "B"]
target = np.array([1.0, 0.0, 0.5, 1.0, 0.2])

# Encode
encoded, name, prior, stats = rs.target_encode(
    categories, target, "category",
    prior_weight=1.0, n_permutations=4, seed=42
)

# Apply to new data
new_cats = ["A", "D"]  # D is unseen
new_encoded = rs.apply_target_encoding(new_cats, stats, prior)
```
