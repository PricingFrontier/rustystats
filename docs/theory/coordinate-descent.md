# Coordinate Descent for Penalized GLMs

This chapter provides a complete derivation of the coordinate descent algorithm used for fitting regularized GLMs. While IRLS handles unpenalized GLMs elegantly, adding L1 (Lasso) penalties requires a different approach because the L1 norm is non-differentiable at zero.

**Prerequisites**: Understanding of [GLMs](glm-intro.md), [IRLS](irls.md), and the [regularization framework](regularization.md).

---

## Part 1: Why Not Just Use IRLS?

### 1.1 The Problem with L1 Penalties

Recall that IRLS solves the GLM by iteratively solving weighted least squares problems. At each iteration, we solve:

$$
(\mathbf{X}^T \mathbf{W} \mathbf{X}) \boldsymbol{\beta} = \mathbf{X}^T \mathbf{W} \mathbf{z}
$$

For **Ridge regression** (L2 penalty), we can simply add a penalty to the diagonal:

$$
(\mathbf{X}^T \mathbf{W} \mathbf{X} + \lambda \mathbf{I}) \boldsymbol{\beta} = \mathbf{X}^T \mathbf{W} \mathbf{z}
$$

This works because the L2 penalty $\|\boldsymbol{\beta}\|_2^2$ is differentiable everywhere.

**However**, the L1 penalty $\|\boldsymbol{\beta}\|_1 = \sum_j |\beta_j|$ is **not differentiable at zero**:

$$
\frac{\partial}{\partial \beta_j} |\beta_j| = \begin{cases}
+1 & \text{if } \beta_j > 0 \\
-1 & \text{if } \beta_j < 0 \\
\text{undefined} & \text{if } \beta_j = 0
\end{cases}
$$

This non-differentiability is precisely what enables Lasso to set coefficients **exactly to zero**—but it means we can't use standard gradient-based methods.

### 1.2 The Solution: Coordinate Descent

**Coordinate descent** solves this by updating one coefficient at a time while holding others fixed. For each single-variable subproblem, we can derive a closed-form solution using the **soft-thresholding operator**.

The key insight: even though the overall objective is non-smooth, each single-variable optimization has an explicit solution.

---

## Part 2: The Soft-Thresholding Operator

### 2.1 Derivation for Simple Lasso

Consider the simplest case: minimizing a univariate Lasso objective:

$$
\min_\beta \frac{1}{2}(\beta - z)^2 + \lambda |\beta|
$$

where $z$ is some target value.

**Case 1: $\beta > 0$**

The objective becomes $\frac{1}{2}(\beta - z)^2 + \lambda \beta$.

Taking the derivative and setting to zero:
$$
\beta - z + \lambda = 0 \implies \beta = z - \lambda
$$

This is valid only if $\beta > 0$, which requires $z > \lambda$.

**Case 2: $\beta < 0$**

The objective becomes $\frac{1}{2}(\beta - z)^2 - \lambda \beta$.

Taking the derivative:
$$
\beta - z - \lambda = 0 \implies \beta = z + \lambda
$$

This is valid only if $\beta < 0$, which requires $z < -\lambda$.

**Case 3: $\beta = 0$**

If $|z| \leq \lambda$, neither Case 1 nor Case 2 applies. We check that $\beta = 0$ is optimal by verifying the **subgradient condition**: at $\beta = 0$, the subgradient of $|\beta|$ is the interval $[-1, 1]$, so stationarity requires $-z + \lambda \cdot s = 0$ for some $s \in [-1, 1]$, i.e., $|z| \leq \lambda$.

### 2.2 The Soft-Thresholding Function

Combining all cases, the solution is the **soft-thresholding operator**:

$$
S(z, \lambda) = \text{sign}(z) \cdot \max(|z| - \lambda, 0)
$$

Equivalently:

$$
S(z, \lambda) = \begin{cases}
z - \lambda & \text{if } z > \lambda \\
0 & \text{if } |z| \leq \lambda \\
z + \lambda & \text{if } z < -\lambda
\end{cases}
$$

!!! info "Geometric Interpretation"
    Soft-thresholding "shrinks" the value $z$ toward zero by amount $\lambda$. If $|z| \leq \lambda$, it shrinks all the way to zero. This is what produces **exact zeros** in Lasso solutions.

### 2.3 Implementation

```rust
/// Soft-thresholding operator S(z, γ) = sign(z) × max(|z| - γ, 0)
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

---

## Part 3: Coordinate Descent for Weighted Lasso

### 3.1 The Weighted Lasso Problem

In each IRLS iteration, we need to solve a **weighted** Lasso problem:

$$
\min_{\boldsymbol{\beta}} \frac{1}{2} \sum_{i=1}^n w_i (z_i - \mathbf{x}_i^T \boldsymbol{\beta})^2 + \lambda \sum_{j=1}^p |\beta_j|
$$

where:
- $z_i$ is the working response for observation $i$
- $w_i$ is the combined weight (IRLS weight × prior weight)
- $\lambda$ is the L1 penalty strength

### 3.2 Single-Coordinate Update

To update $\beta_j$ while holding all other coefficients fixed, define the **partial residual**:

$$
r_i^{(j)} = z_i - \sum_{k \neq j} x_{ik} \beta_k
$$

This is the "residual" if we remove the contribution of all predictors except $j$.

The objective for $\beta_j$ alone becomes:

$$
\min_{\beta_j} \frac{1}{2} \sum_{i=1}^n w_i (r_i^{(j)} - x_{ij} \beta_j)^2 + \lambda |\beta_j|
$$

Expanding the quadratic:

$$
= \frac{1}{2} \sum_i w_i (r_i^{(j)})^2 - \beta_j \sum_i w_i x_{ij} r_i^{(j)} + \frac{\beta_j^2}{2} \sum_i w_i x_{ij}^2 + \lambda |\beta_j|
$$

Let:
- $\rho_j = \sum_i w_i x_{ij} r_i^{(j)}$ (the weighted correlation of predictor $j$ with partial residuals)
- $\nu_j = \sum_i w_i x_{ij}^2$ (the weighted sum of squares for predictor $j$)

The objective becomes:

$$
\min_{\beta_j} \frac{1}{2} \nu_j \beta_j^2 - \rho_j \beta_j + \lambda |\beta_j| + \text{const}
$$

Dividing by $\nu_j$ (assuming $\nu_j > 0$):

$$
\min_{\beta_j} \frac{1}{2} \beta_j^2 - \frac{\rho_j}{\nu_j} \beta_j + \frac{\lambda}{\nu_j} |\beta_j|
$$

This is equivalent to:

$$
\min_{\beta_j} \frac{1}{2} \left(\beta_j - \frac{\rho_j}{\nu_j}\right)^2 + \frac{\lambda}{\nu_j} |\beta_j|
$$

### 3.3 The Update Formula

Applying soft-thresholding:

$$
\boxed{\beta_j^{\text{new}} = \frac{S(\rho_j, \lambda)}{\nu_j} = \frac{\text{sign}(\rho_j) \max(|\rho_j| - \lambda, 0)}{\nu_j}}
$$

where:
- $\rho_j = \sum_i w_i x_{ij} r_i^{(j)} = \sum_i w_i x_{ij} (z_i - \sum_{k \neq j} x_{ik} \beta_k)$
- $\nu_j = \sum_i w_i x_{ij}^2$

### 3.4 Elastic Net Extension

For Elastic Net with L1 ratio $\alpha$ (called `l1_ratio` in RustyStats), the penalty is:

$$
\lambda \left[ \alpha \sum_j |\beta_j| + \frac{1-\alpha}{2} \sum_j \beta_j^2 \right]
$$

The L2 part modifies the denominator:

$$
\boxed{\beta_j^{\text{new}} = \frac{S(\rho_j, \lambda \alpha)}{\nu_j + \lambda(1-\alpha)}}
$$

When $\alpha = 1$ (pure Lasso), this reduces to the previous formula.
When $\alpha = 0$ (pure Ridge), there's no soft-thresholding, just shrinkage.

---

## Part 4: The Full Algorithm (IRCD)

RustyStats uses **Iteratively Reweighted Coordinate Descent (IRCD)**, which combines:
1. **Outer loop**: IRLS-style updates of working response and weights
2. **Inner loop**: Coordinate descent for the penalized weighted least squares

### 4.1 Algorithm Pseudocode

```
Input: y, X, family, link, λ, α
Output: β̂

1. Initialize:
   β ← 0  (or warm start)
   μ ← family.initialize_mu(y)
   η ← g(μ)

2. Outer loop (IRLS): repeat until deviance converges
   
   a. Compute working response and weights:
      For i = 1, ..., n:
        V_i ← family.variance(μ_i)
        d_i ← link.derivative(μ_i)
        w_i ← 1 / (V_i × d_i²)                    # IRLS weight
        z_i ← η_i + (y_i - μ_i) × d_i            # Working response
   
   b. Inner loop (coordinate descent): repeat until β converges
      
      For j = 1, ..., p:
        # Compute partial residual correlation
        ρ_j ← Σᵢ wᵢ xᵢⱼ (zᵢ - Σₖ≠ⱼ xᵢₖ βₖ)
        
        # Compute weighted sum of squares
        ν_j ← Σᵢ wᵢ xᵢⱼ²
        
        # Update with soft-thresholding (skip intercept)
        if j == 0 (intercept):
          β_j ← ρ_j / ν_j
        else:
          β_j ← S(ρ_j, λα) / (ν_j + λ(1-α))
   
   c. Update predictions:
      η ← Xβ + offset
      μ ← g⁻¹(η)
   
   d. Check convergence:
      D_new ← family.deviance(y, μ)
      if |D_new - D_old| / D_old < tolerance:
        STOP
      D_old ← D_new

3. Return β
```

### 4.2 The Intercept is Never Penalized

Notice that the intercept ($j = 0$) uses a simple update without soft-thresholding:

$$
\beta_0 = \frac{\sum_i w_i (z_i - \sum_{k > 0} x_{ik} \beta_k)}{\sum_i w_i}
$$

This is critical because:
1. Penalizing the intercept would bias the model's overall level
2. The intercept controls the baseline prediction, which should be determined by the data

---

## Part 5: Computational Optimizations

### 5.1 Covariance Updates (glmnet-style)

Computing $\rho_j$ naively requires $O(n)$ operations per coefficient, giving $O(np)$ per coordinate descent cycle.

**Key insight**: We can use precomputed quantities to reduce this to $O(p)$ per coefficient.

Define:
- $\mathbf{G} = \mathbf{X}^T \mathbf{W} \mathbf{X}$ (the weighted Gram matrix)
- $\mathbf{c} = \mathbf{X}^T \mathbf{W} \mathbf{z}$ (weighted correlation with working response)

Then:

$$
\rho_j = c_j - \sum_{k \neq j} G_{jk} \beta_k = c_j - \sum_k G_{jk} \beta_k + G_{jj} \beta_j
$$

Since we track the gradient $\nabla_j = c_j - \sum_k G_{jk} \beta_k$:

$$
\rho_j = \nabla_j + G_{jj} \beta_j^{\text{old}}
$$

After updating $\beta_j$, we update the gradient for all $k$:

$$
\nabla_k \leftarrow \nabla_k - G_{kj} (\beta_j^{\text{new}} - \beta_j^{\text{old}})
$$

This gives $O(p)$ per coefficient update, and the Gram matrix $\mathbf{G}$ is computed once per outer iteration at cost $O(np^2)$.

### 5.2 RustyStats Implementation

```rust
// Precompute X'Wz (gradient at β=0) - PARALLEL
let xwz: Vec<f64> = (0..p)
    .into_par_iter()
    .map(|j| {
        x.column(j).iter()
            .zip(weights.iter())
            .zip(working_response.iter())
            .map(|((&xij, &wi), &zi)| wi * xij * zi)
            .sum()
    })
    .collect();

// Precompute X'WX (Gram matrix) - PARALLEL fold-reduce
let xwx: Vec<f64> = (0..n)
    .into_par_iter()
    .fold(
        || vec![0.0; p * p],
        |mut acc, i| {
            let w_i = weights[i];
            for j in 0..p {
                for k in j..p {  // Upper triangle only
                    acc[j * p + k] += w_i * x[[i, j]] * x[[i, k]];
                }
            }
            acc
        },
    )
    .reduce(|| vec![0.0; p * p], |a, b| /* element-wise sum */);
```

### 5.3 Active Set Strategy

When $\lambda$ is large, many coefficients are zero. We can skip updating them:

1. **Strong rules**: Predict which coefficients will be zero before fitting
2. **Active set cycling**: Only iterate over non-zero coefficients after initial pass
3. **KKT checking**: Verify zero coefficients satisfy optimality conditions

RustyStats uses implicit active set management: coefficients that become zero stay zero unless $\rho_j$ exceeds the threshold.

### 5.4 Warm Starts

When computing a regularization path (fitting at many $\lambda$ values), we use **warm starts**:

1. Start with large $\lambda$ where all coefficients (except intercept) are zero
2. Use the solution at $\lambda_k$ as the starting point for $\lambda_{k+1}$

This dramatically speeds up path computation because solutions at adjacent $\lambda$ values are similar.

---

## Part 6: Convergence

### 6.1 Inner Loop Convergence

The inner coordinate descent loop converges when the maximum coefficient change is small:

$$
\max_j |\beta_j^{\text{new}} - \beta_j^{\text{old}}| < \epsilon_{\text{cd}}
$$

**Default**: $\epsilon_{\text{cd}} = 10^{-6}$

### 6.2 Outer Loop Convergence

The outer IRLS loop converges when the deviance stabilizes:

$$
\frac{|D^{(t)} - D^{(t-1)}|}{|D^{(t-1)}|} < \epsilon_{\text{irls}}
$$

**Default**: $\epsilon_{\text{irls}} = 10^{-8}$

### 6.3 Convergence Theory

Coordinate descent for Lasso is guaranteed to converge because:
1. The objective is **convex** (sum of convex loss and convex penalty)
2. Each coordinate update is the **exact minimizer** for that coordinate
3. The objective decreases monotonically

For non-Gaussian families, the IRLS outer loop adds another layer, but convergence is typically fast because:
- Working weights stabilize quickly
- Each inner optimization starts from a good solution

---

## Part 7: Worked Example

### 7.1 Setup

Consider a simple problem:
- $n = 4$ observations
- $p = 3$ predictors (including intercept)
- Gaussian family (so $w_i = 1$ for all $i$)
- $\lambda = 1.0$, $\alpha = 1.0$ (pure Lasso)

Data:
$$
\mathbf{X} = \begin{pmatrix} 1 & 2 & 1 \\ 1 & 4 & 2 \\ 1 & 6 & 3 \\ 1 & 8 & 4 \end{pmatrix}, \quad
\mathbf{y} = \begin{pmatrix} 5 \\ 9 \\ 13 \\ 17 \end{pmatrix}
$$

Note: $x_1$ (column 2) and $x_2$ (column 3) are perfectly correlated ($x_2 = x_1/2$). Lasso will select only one.

### 7.2 Iteration 1

**Initialize**: $\boldsymbol{\beta} = (0, 0, 0)^T$

**Compute $\nu_j$** (sum of squared predictor values):
- $\nu_0 = 1^2 + 1^2 + 1^2 + 1^2 = 4$
- $\nu_1 = 2^2 + 4^2 + 6^2 + 8^2 = 120$
- $\nu_2 = 1^2 + 2^2 + 3^2 + 4^2 = 30$

**Update $\beta_0$ (intercept, no penalty)**:
$$
\rho_0 = \sum_i x_{i0}(y_i - 0) = 5 + 9 + 13 + 17 = 44
$$
$$
\beta_0 = 44 / 4 = 11
$$

**Update $\beta_1$**:

Partial residual (removing intercept contribution):
$$
r_i^{(1)} = y_i - 11 \cdot 1 = y_i - 11
$$
So $r^{(1)} = (-6, -2, 2, 6)^T$

$$
\rho_1 = 2(-6) + 4(-2) + 6(2) + 8(6) = -12 - 8 + 12 + 48 = 40
$$

Apply soft-thresholding:
$$
\beta_1 = S(40, 1.0) / 120 = 39 / 120 = 0.325
$$

**Update $\beta_2$**:

Now the residual accounts for both $\beta_0$ and $\beta_1$:
$$
r_i^{(2)} = y_i - 11 - 0.325 \cdot x_{i1}
$$
$$
r^{(2)} = (5 - 11 - 0.65, 9 - 11 - 1.3, 13 - 11 - 1.95, 17 - 11 - 2.6) = (-6.65, -3.3, 0.05, 3.4)
$$

$$
\rho_2 = 1(-6.65) + 2(-3.3) + 3(0.05) + 4(3.4) = -6.65 - 6.6 + 0.15 + 13.6 = 0.5
$$

Apply soft-thresholding:
$$
\beta_2 = S(0.5, 1.0) / 30 = 0 / 30 = 0
$$

Since $|\rho_2| = 0.5 < \lambda = 1$, the coefficient is set to **exactly zero**.

### 7.3 Subsequent Iterations

The algorithm continues cycling through coefficients. After a few iterations:
- $\beta_0 \approx 1.0$ (intercept)
- $\beta_1 \approx 2.0$ (selected predictor)
- $\beta_2 = 0$ (excluded due to collinearity)

Lasso chose $x_1$ over $x_2$ because it had a larger initial correlation with $y$.

---

## Part 8: Comparison with Other Methods

### 8.1 Coordinate Descent vs Gradient Descent

| Aspect | Coordinate Descent | Gradient Descent |
|--------|-------------------|------------------|
| Update | One variable at a time | All variables simultaneously |
| L1 handling | Exact via soft-threshold | Requires subgradients or proximal |
| Convergence | Often faster for sparse | Can be slow near optimum |
| Parallelism | Sequential by nature | Easily parallelized |

RustyStats uses coordinate descent because it handles L1 penalties naturally and converges quickly for sparse solutions.

### 8.2 IRCD vs ADMM

ADMM (Alternating Direction Method of Multipliers) is another approach for L1 problems. Coordinate descent is typically faster for Lasso/Elastic Net, while ADMM is more general and handles complex constraints better.

---

## Part 9: RustyStats Implementation Details

### 9.1 File Location

```
crates/rustystats-core/src/solvers/coordinate_descent.rs
```

### 9.2 Key Functions

```rust
/// Main entry point for regularized GLM fitting
pub fn fit_glm_coordinate_descent(
    y: &Array1<f64>,
    x: &Array2<f64>,
    family: &dyn Family,
    link: &dyn Link,
    irls_config: &IRLSConfig,
    reg_config: &RegularizationConfig,
    offset: Option<&Array1<f64>>,
    weights: Option<&Array1<f64>>,
) -> Result<IRLSResult>
```

### 9.3 Parallelization

The Gram matrix computation uses Rayon's parallel fold-reduce:

```rust
let xwx: Vec<f64> = (0..n)
    .into_par_iter()
    .fold(
        || vec![0.0; p * p],  // Per-thread accumulator
        |mut acc, i| {
            // Accumulate contribution from observation i
            for j in 0..p {
                for k in j..p {
                    acc[j * p + k] += w_i * x[[i, j]] * x[[i, k]];
                }
            }
            acc
        },
    )
    .reduce(
        || vec![0.0; p * p],  // Identity
        |a, b| /* element-wise sum */
    );
```

This parallelizes across observations, giving near-linear speedup on multi-core systems.

### 9.4 Numerical Stability

- **Minimum weight floor**: Prevents division by near-zero weights
- **μ clamping**: Keeps predicted values in valid range for family
- **Denominator check**: Avoids division by zero in coefficient updates

```rust
let new_coef = if j < pen_start {
    rho / xwx_jj  // Intercept: no penalty
} else {
    let denom = xwx_jj + l2_penalty;
    if denom.abs() < 1e-10 {
        0.0  // Avoid division by zero
    } else {
        soft_threshold(rho, l1_penalty) / denom
    }
};
```

---

## Summary

| Concept | Formula/Description |
|---------|---------------------|
| **Soft-thresholding** | $S(z, \lambda) = \text{sign}(z) \max(\|z\| - \lambda, 0)$ |
| **Lasso update** | $\beta_j = S(\rho_j, \lambda) / \nu_j$ |
| **Elastic Net update** | $\beta_j = S(\rho_j, \lambda\alpha) / (\nu_j + \lambda(1-\alpha))$ |
| **Partial residual** | $\rho_j = \sum_i w_i x_{ij} (z_i - \sum_{k \neq j} x_{ik} \beta_k)$ |
| **Intercept** | Never penalized: $\beta_0 = \rho_0 / \nu_0$ |
| **Convergence** | Inner: $\max \|\Delta\beta\| < \epsilon$; Outer: $\|{\Delta D}/{D}\| < \epsilon$ |

**Key advantages of coordinate descent**:
1. Handles non-differentiable L1 penalty exactly
2. Produces **exact zeros** (true sparsity)
3. Efficient with covariance updates: $O(p)$ per coefficient
4. Natural warm starts for regularization paths

---

## Next Steps

- [Regularization Theory](regularization.md) — Ridge, Lasso, Elastic Net penalties
- [Cross-Validation](../guides/glm-workflow.md#step-7-regularization-for-variable-selection) — Choosing optimal λ
- [IRLS Algorithm](irls.md) — The unpenalized solver
