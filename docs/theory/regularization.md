# Regularization

Regularization adds a penalty to the objective function to prevent overfitting and enable variable selection. This chapter covers Ridge, Lasso, and Elastic Net regularization for GLMs.

## Why Regularize?

Standard GLM fitting minimizes deviance:

\[
\hat{\boldsymbol{\beta}} = \arg\min_{\boldsymbol{\beta}} D(\boldsymbol{\beta})
\]

Problems can arise with:
- **Many predictors**: Overfitting, poor generalization
- **Correlated predictors**: Unstable coefficients
- **Variable selection**: Which predictors matter?

Regularization adds a penalty term:

\[
\hat{\boldsymbol{\beta}} = \arg\min_{\boldsymbol{\beta}} \left[ D(\boldsymbol{\beta}) + \lambda P(\boldsymbol{\beta}) \right]
\]

where \(\lambda\) controls the penalty strength and \(P(\boldsymbol{\beta})\) is the penalty function.

## Ridge Regression (L2)

### The Penalty

\[
P_{\text{Ridge}}(\boldsymbol{\beta}) = \sum_{j=1}^{p} \beta_j^2 = \|\boldsymbol{\beta}\|_2^2
\]

### Full Objective

\[
\min_{\boldsymbol{\beta}} \left[ D(\boldsymbol{\beta}) + \lambda \sum_{j=1}^{p} \beta_j^2 \right]
\]

### Properties

- **Shrinks** all coefficients toward zero
- **Never sets coefficients exactly to zero**
- Handles multicollinearity well
- Smooth, differentiable penalty

### When to Use

- Many correlated predictors
- Want to keep all variables
- Stabilize coefficient estimates

### Example

```python
import rustystats as rs

# Ridge regression (l1_ratio = 0)
result = rs.fit_glm(
    y, X,
    family="gaussian",
    alpha=0.1,      # Penalty strength
    l1_ratio=0.0    # Pure L2 (Ridge)
)
```

## Lasso Regression (L1)

### The Penalty

\[
P_{\text{Lasso}}(\boldsymbol{\beta}) = \sum_{j=1}^{p} |\beta_j| = \|\boldsymbol{\beta}\|_1
\]

### Full Objective

\[
\min_{\boldsymbol{\beta}} \left[ D(\boldsymbol{\beta}) + \lambda \sum_{j=1}^{p} |\beta_j| \right]
\]

### Properties

- **Sets some coefficients exactly to zero**
- Performs **variable selection**
- Selects at most \(n\) variables (in high-dimensional settings)
- Non-smooth penalty (requires special optimization)

### The Soft Thresholding Operator

Lasso uses soft thresholding:

\[
S(z, \gamma) = \text{sign}(z) \max(|z| - \gamma, 0)
\]

```rust
pub fn soft_threshold(z: f64, gamma: f64) -> f64 {
    if z > gamma {
        z - gamma
    } else if z < -gamma {
        z + gamma
    } else {
        0.0
    }
}
```

### When to Use

- Feature selection is desired
- Many predictors, few are relevant
- Interpretable sparse models

### Example

```python
# Lasso (l1_ratio = 1)
result = rs.fit_glm(
    y, X,
    family="poisson",
    alpha=0.1,      # Penalty strength
    l1_ratio=1.0    # Pure L1 (Lasso)
)

print(f"Non-zero coefficients: {result.n_nonzero()}")
print(f"Selected features: {result.selected_features()}")
```

## Elastic Net

### The Penalty

A weighted combination of L1 and L2:

\[
P_{\text{EN}}(\boldsymbol{\beta}) = \rho \|\boldsymbol{\beta}\|_1 + \frac{1-\rho}{2} \|\boldsymbol{\beta}\|_2^2
\]

where \(\rho \in [0, 1]\) is the L1 ratio.

### Full Objective

\[
\min_{\boldsymbol{\beta}} \left[ D(\boldsymbol{\beta}) + \lambda \left( \rho \sum_j |\beta_j| + \frac{1-\rho}{2} \sum_j \beta_j^2 \right) \right]
\]

### Properties

- Combines variable selection (L1) with stability (L2)
- Can select more than \(n\) variables
- Handles groups of correlated predictors better than pure Lasso

### When to Use

- Correlated predictors that should be selected together
- Want some variable selection but also stability
- General default when unsure

### Example

```python
# Elastic Net (0 < l1_ratio < 1)
result = rs.fit_glm(
    y, X,
    family="gaussian",
    alpha=0.1,
    l1_ratio=0.5    # 50% L1, 50% L2
)
```

## Coordinate Descent Algorithm

RustyStats uses **coordinate descent** for regularized GLMs, following the glmnet approach.

### Algorithm Overview

```
Initialize β = 0 (or from IRLS)
Repeat until converged:
    For j = 1, ..., p:
        Compute partial residual: r_j = z - X_{-j} β_{-j}
        Update β_j by soft thresholding:
            β_j = S(⟨X_j, W r_j⟩, λρ) / (⟨X_j, W X_j⟩ + λ(1-ρ))
```

### Why Coordinate Descent?

- **Efficient**: \(O(np)\) per cycle
- **Warm starts**: Path from large λ to small λ
- **Sparse updates**: Skip zero coefficients

### Implementation

```
crates/rustystats-core/src/solvers/coordinate_descent.rs
```

Key optimizations:
- Covariance updates (compute X'WX once)
- Active set strategy (focus on non-zero coefficients)
- Parallelization via Rayon

## Choosing λ (Alpha)

The penalty strength λ (called `alpha` in RustyStats API) is crucial.

### Lasso Path

Trace coefficients as λ varies:

```python
path = rs.lasso_path(y, X, family="gaussian", n_alphas=50)

# View path
print(path.alphas)        # λ values
print(path.coefs)         # Coefficients at each λ

# Plot
path.plot()
```

### Cross-Validation

Find optimal λ by cross-validation:

```python
cv_result = rs.cv_glm(
    y, X,
    family="poisson",
    l1_ratio=1.0,    # Lasso
    cv=5,            # 5-fold CV
    n_alphas=100
)

print(f"Best α (min CV error): {cv_result.alpha_best}")
print(f"1-SE α (more parsimonious): {cv_result.alpha_1se}")

# Refit with optimal α
result = rs.fit_glm(y, X, family="poisson", 
                     alpha=cv_result.alpha_best, l1_ratio=1.0)
```

### The 1-SE Rule

Choose the largest λ within one standard error of the minimum CV error:

```
λ_1se: largest λ such that CV_error(λ) ≤ CV_error(λ_min) + SE(λ_min)
```

This gives a simpler model with comparable performance.

## Standardization

### Why Standardize?

The penalty is applied uniformly to all coefficients. Without standardization, variables on different scales are penalized differently.

### Internal Standardization

RustyStats internally standardizes features before fitting:

```python
# Features are standardized internally
# Coefficients are returned on original scale
result = rs.fit_glm(y, X, family="gaussian", alpha=0.1, l1_ratio=1.0)
```

### Intercept

The intercept is **never penalized**:

\[
\min_{\beta_0, \boldsymbol{\beta}} \left[ D(\beta_0, \boldsymbol{\beta}) + \lambda P(\boldsymbol{\beta}) \right]
\]

## Regularization with Different Families

Regularization works with all GLM families:

```python
# Regularized Poisson
result = rs.fit_glm(y, X, family="poisson", alpha=0.1, l1_ratio=1.0)

# Regularized Binomial (Logistic)
result = rs.fit_glm(y, X, family="binomial", alpha=0.1, l1_ratio=0.5)

# Regularized Gamma
result = rs.fit_glm(y, X, family="gamma", alpha=0.1, l1_ratio=0.0)
```

The coordinate descent algorithm adapts to each family's variance function and link.

## Degrees of Freedom

For regularized models, the effective degrees of freedom is less than \(p\):

- **Ridge**: \(\text{df} = \text{tr}(\mathbf{X}(\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T)\)
- **Lasso**: Approximately the number of non-zero coefficients

## Regularization and Inference

!!! warning "Standard Errors with Regularization"
    Standard errors from regularized models are **biased** and should be interpreted with caution. The coefficients are shrunk toward zero, which affects the sampling distribution.

For valid inference with selected variables:
1. Use cross-validation to select variables
2. Refit an unregularized model with selected variables
3. Use standard errors from the unregularized model

## Summary

| Method | Penalty | Variable Selection | Best For |
|--------|---------|-------------------|----------|
| Ridge | L2: \(\|\boldsymbol{\beta}\|_2^2\) | No | Multicollinearity |
| Lasso | L1: \(\|\boldsymbol{\beta}\|_1\) | Yes | Sparse models |
| Elastic Net | Mix | Yes (grouped) | Correlated predictors |

Key parameters:
- `alpha`: Penalty strength (larger = more shrinkage)
- `l1_ratio`: Mix of L1/L2 (1 = Lasso, 0 = Ridge)

Best practices:
1. Use cross-validation to choose `alpha`
2. Consider the 1-SE rule for parsimony
3. Start with `l1_ratio=0.5` (Elastic Net) unless you have a specific reason
