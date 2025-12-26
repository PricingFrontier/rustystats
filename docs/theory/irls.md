# The IRLS Algorithm

**Iteratively Reweighted Least Squares (IRLS)** is the workhorse algorithm for fitting GLMs. This chapter provides a complete derivation from first principles, showing exactly why IRLS works and how it's implemented in RustyStats.

**Prerequisites**: You should understand the GLM framework from the [previous chapter](glm-intro.md), including the score equations and Fisher information.

---

## Part 1: The Problem We're Solving

### 1.1 Recap: The Score Equations

From maximum likelihood theory, we want to find $\boldsymbol{\beta}$ that satisfies the score equations:

$$
\frac{\partial \ell}{\partial \beta_j} = \sum_{i=1}^n \frac{(y_i - \mu_i) x_{ij}}{\phi \cdot V(\mu_i) \cdot g'(\mu_i)} = 0 \quad \text{for } j = 0, 1, \ldots, p-1
$$

In matrix notation, define the diagonal matrices:
- $\mathbf{V} = \text{diag}(V(\mu_1), \ldots, V(\mu_n))$ — variance function values
- $\mathbf{G} = \text{diag}(g'(\mu_1), \ldots, g'(\mu_n))$ — link derivatives

Then the score equations become:

$$
\mathbf{U} = \frac{1}{\phi} \mathbf{X}^T \mathbf{V}^{-1} \mathbf{G}^{-1} (\mathbf{y} - \boldsymbol{\mu}) = \mathbf{0}
$$

### 1.2 Why We Can't Solve Directly

For linear regression (Gaussian family, identity link), $\mu_i = \mathbf{x}_i^T\boldsymbol{\beta}$ is linear in $\boldsymbol{\beta}$, and we get the normal equations with a closed-form solution.

For general GLMs, $\mu_i = g^{-1}(\mathbf{x}_i^T\boldsymbol{\beta})$ is **nonlinear** in $\boldsymbol{\beta}$. For example:

- **Poisson with log link**: $\mu_i = e^{\mathbf{x}_i^T\boldsymbol{\beta}}$
- **Binomial with logit link**: $\mu_i = \frac{e^{\mathbf{x}_i^T\boldsymbol{\beta}}}{1 + e^{\mathbf{x}_i^T\boldsymbol{\beta}}}$

The score equations involve these nonlinear functions, so there's no closed-form solution. We need an iterative method.

---

## Part 2: Newton-Raphson and Fisher Scoring

### 2.1 Newton-Raphson Method

Newton-Raphson is a general method for solving equations $f(\mathbf{x}) = \mathbf{0}$. The idea is to linearize $f$ around the current estimate and solve the linear approximation.

**One-dimensional case**: To solve $f(x) = 0$:

1. Start with initial guess $x^{(0)}$
2. Taylor expand: $f(x) \approx f(x^{(t)}) + f'(x^{(t)})(x - x^{(t)})$
3. Set approximation to zero and solve: $x^{(t+1)} = x^{(t)} - \frac{f(x^{(t)})}{f'(x^{(t)})}$

**Multi-dimensional case**: To solve $\mathbf{f}(\mathbf{x}) = \mathbf{0}$:

$$
\mathbf{x}^{(t+1)} = \mathbf{x}^{(t)} - \mathbf{J}^{-1}(\mathbf{x}^{(t)}) \mathbf{f}(\mathbf{x}^{(t)})
$$

where $\mathbf{J}$ is the Jacobian matrix with $J_{jk} = \frac{\partial f_j}{\partial x_k}$.

### 2.2 Newton-Raphson for Maximum Likelihood

For maximum likelihood, we want to solve $\mathbf{U}(\boldsymbol{\beta}) = \mathbf{0}$ where $\mathbf{U} = \frac{\partial \ell}{\partial \boldsymbol{\beta}}$ is the score.

The Jacobian of $\mathbf{U}$ is the **Hessian** of $\ell$:

$$
\mathbf{H} = \frac{\partial \mathbf{U}}{\partial \boldsymbol{\beta}^T} = \frac{\partial^2 \ell}{\partial \boldsymbol{\beta} \partial \boldsymbol{\beta}^T}
$$

So Newton-Raphson gives:

$$
\boldsymbol{\beta}^{(t+1)} = \boldsymbol{\beta}^{(t)} - \mathbf{H}^{-1}(\boldsymbol{\beta}^{(t)}) \mathbf{U}(\boldsymbol{\beta}^{(t)})
$$

### 2.3 The Problem with the Hessian

Computing the Hessian requires second derivatives, which can be:
- Complicated to derive
- Computationally expensive
- Potentially non-positive-definite (causing instability)

**Fisher's insight**: Replace the Hessian with its **expected value**.

### 2.4 Fisher Information

The **Fisher information matrix** is defined as:

$$
\mathcal{I}(\boldsymbol{\beta}) = E\left[-\mathbf{H}\right] = E\left[-\frac{\partial^2 \ell}{\partial \boldsymbol{\beta} \partial \boldsymbol{\beta}^T}\right]
$$

Equivalently (under regularity conditions):

$$
\mathcal{I}(\boldsymbol{\beta}) = E\left[\mathbf{U}\mathbf{U}^T\right]
$$

The Fisher information has a beautiful property: it tells us the **maximum precision** we can achieve when estimating $\boldsymbol{\beta}$.

### 2.5 Deriving the Fisher Information for GLMs

Let's compute $\mathcal{I}$ for a GLM. We need the second derivative of the log-likelihood.

Starting from:
$$
\frac{\partial \ell}{\partial \beta_j} = \frac{1}{\phi} \sum_i \frac{(y_i - \mu_i) x_{ij}}{V(\mu_i) g'(\mu_i)}
$$

Taking the derivative with respect to $\beta_k$:

$$
\frac{\partial^2 \ell}{\partial \beta_j \partial \beta_k} = \frac{1}{\phi} \sum_i x_{ij} \frac{\partial}{\partial \beta_k}\left[\frac{y_i - \mu_i}{V(\mu_i) g'(\mu_i)}\right]
$$

This involves derivatives of $\mu_i$, $V(\mu_i)$, and $g'(\mu_i)$ with respect to $\beta_k$. After careful calculation (which is tedious but straightforward), we get:

$$
\frac{\partial^2 \ell}{\partial \beta_j \partial \beta_k} = -\frac{1}{\phi} \sum_i \frac{x_{ij} x_{ik}}{V(\mu_i)[g'(\mu_i)]^2} + \text{(terms involving } y_i - \mu_i \text{)}
$$

Taking expectations (using $E[y_i - \mu_i] = 0$):

$$
\mathcal{I}_{jk} = E\left[-\frac{\partial^2 \ell}{\partial \beta_j \partial \beta_k}\right] = \frac{1}{\phi} \sum_i \frac{x_{ij} x_{ik}}{V(\mu_i)[g'(\mu_i)]^2}
$$

In matrix form:

$$
\mathcal{I} = \frac{1}{\phi} \mathbf{X}^T \mathbf{W} \mathbf{X}
$$

where $\mathbf{W}$ is a diagonal matrix with:

$$
W_{ii} = \frac{1}{V(\mu_i)[g'(\mu_i)]^2}
$$

This is the **weight matrix** for IRLS!

### 2.6 Fisher Scoring Algorithm

**Fisher scoring** replaces Newton-Raphson's Hessian with the Fisher information:

$$
\boldsymbol{\beta}^{(t+1)} = \boldsymbol{\beta}^{(t)} + \mathcal{I}^{-1}(\boldsymbol{\beta}^{(t)}) \mathbf{U}(\boldsymbol{\beta}^{(t)})
$$

Substituting our expressions:

$$
\boldsymbol{\beta}^{(t+1)} = \boldsymbol{\beta}^{(t)} + \phi(\mathbf{X}^T \mathbf{W}^{(t)} \mathbf{X})^{-1} \cdot \frac{1}{\phi} \mathbf{X}^T \mathbf{V}^{-1} \mathbf{G}^{-1} (\mathbf{y} - \boldsymbol{\mu}^{(t)})
$$

$$
= \boldsymbol{\beta}^{(t)} + (\mathbf{X}^T \mathbf{W}^{(t)} \mathbf{X})^{-1} \mathbf{X}^T \mathbf{W}^{(t)} \mathbf{G}^{(t)} (\mathbf{y} - \boldsymbol{\mu}^{(t)})
$$

(The second line uses $\mathbf{V}^{-1}\mathbf{G}^{-1} = \mathbf{W}\mathbf{G}$ since $W_{ii} = 1/(V_i G_i^2)$.)

---

## Part 3: Transforming to Weighted Least Squares

### 3.1 The Working Response

The key insight is to define the **working response** (or adjusted dependent variable):

$$
z_i = \eta_i + (y_i - \mu_i) g'(\mu_i)
$$

where $\eta_i = g(\mu_i) = \mathbf{x}_i^T\boldsymbol{\beta}^{(t)}$ is the current linear predictor.

In matrix form:
$$
\mathbf{z} = \boldsymbol{\eta} + \mathbf{G}(\mathbf{y} - \boldsymbol{\mu})
$$

### 3.2 The Magic: IRLS Update Equals Weighted Least Squares

Let's show that the Fisher scoring update is equivalent to weighted least squares of $\mathbf{z}$ on $\mathbf{X}$ with weights $\mathbf{W}$.

**Weighted least squares solution**:

$$
\hat{\boldsymbol{\beta}}_{\text{WLS}} = (\mathbf{X}^T \mathbf{W} \mathbf{X})^{-1} \mathbf{X}^T \mathbf{W} \mathbf{z}
$$

**Substituting $\mathbf{z}$**:

$$
\hat{\boldsymbol{\beta}}_{\text{WLS}} = (\mathbf{X}^T \mathbf{W} \mathbf{X})^{-1} \mathbf{X}^T \mathbf{W} [\boldsymbol{\eta} + \mathbf{G}(\mathbf{y} - \boldsymbol{\mu})]
$$

$$
= (\mathbf{X}^T \mathbf{W} \mathbf{X})^{-1} \mathbf{X}^T \mathbf{W} \boldsymbol{\eta} + (\mathbf{X}^T \mathbf{W} \mathbf{X})^{-1} \mathbf{X}^T \mathbf{W} \mathbf{G}(\mathbf{y} - \boldsymbol{\mu})
$$

Now, $\boldsymbol{\eta} = \mathbf{X}\boldsymbol{\beta}^{(t)}$, so:

$$
(\mathbf{X}^T \mathbf{W} \mathbf{X})^{-1} \mathbf{X}^T \mathbf{W} \mathbf{X} \boldsymbol{\beta}^{(t)} = \boldsymbol{\beta}^{(t)}
$$

Therefore:

$$
\hat{\boldsymbol{\beta}}_{\text{WLS}} = \boldsymbol{\beta}^{(t)} + (\mathbf{X}^T \mathbf{W} \mathbf{X})^{-1} \mathbf{X}^T \mathbf{W} \mathbf{G}(\mathbf{y} - \boldsymbol{\mu})
$$

This is exactly the Fisher scoring update! $\square$

### 3.3 Why This Matters

This equivalence is profound:

1. **Conceptual**: GLM fitting is "just" repeated weighted linear regression
2. **Computational**: We can use highly optimized linear algebra routines
3. **Software**: The same code structure handles all GLM families

---

## Part 4: Understanding the Components

### 4.1 The Working Response in Detail

$$
z_i = \eta_i + (y_i - \mu_i) g'(\mu_i)
$$

**Intuition**: The working response is a first-order Taylor approximation of $g(y_i)$ around $\mu_i$:

$$
g(y_i) \approx g(\mu_i) + g'(\mu_i)(y_i - \mu_i) = \eta_i + g'(\mu_i)(y_i - \mu_i)
$$

So $z_i$ is approximately what the linear predictor "should be" to match observation $y_i$.

**Examples**:

| Family + Link | $g'(\mu)$ | Working Response $z_i$ |
|---------------|-----------|------------------------|
| Gaussian + Identity | $1$ | $\eta_i + (y_i - \mu_i) = y_i$ |
| Poisson + Log | $1/\mu$ | $\eta_i + (y_i - \mu_i)/\mu_i$ |
| Binomial + Logit | $1/[\mu(1-\mu)]$ | $\eta_i + (y_i - \mu_i)/[\mu_i(1-\mu_i)]$ |

For Gaussian, the working response is just $y_i$—no adjustment needed!

### 4.2 The Working Weights in Detail

$$
w_i = \frac{1}{V(\mu_i)[g'(\mu_i)]^2}
$$

**Interpretation**:
- **Variance factor**: $1/V(\mu_i)$ downweights observations with high variance (more noise → less information)
- **Link factor**: $1/[g'(\mu_i)]^2$ adjusts for the curvature of the link function

**Examples**:

| Family + Link | $V(\mu)$ | $g'(\mu)$ | Weight $w_i$ |
|---------------|----------|-----------|--------------|
| Gaussian + Identity | $1$ | $1$ | $1$ |
| Poisson + Log | $\mu$ | $1/\mu$ | $\mu$ |
| Binomial + Logit | $\mu(1-\mu)$ | $1/[\mu(1-\mu)]$ | $\mu(1-\mu)$ |
| Gamma + Log | $\mu^2$ | $1/\mu$ | $1$ |

!!! note "Constant Weights"
    For Gamma + Log, the weights are constant! This is because $V(\mu) = \mu^2$ and $g'(\mu) = 1/\mu$, so:
    $$w = \frac{1}{\mu^2 \cdot (1/\mu)^2} = 1$$

### 4.3 Why "Reweighted"?

The weights $w_i$ depend on $\mu_i$, which depends on the current $\boldsymbol{\beta}^{(t)}$. Each iteration:

1. Updates $\boldsymbol{\beta}$
2. Updates $\boldsymbol{\mu} = g^{-1}(\mathbf{X}\boldsymbol{\beta})$
3. Updates weights $\mathbf{W}$ based on new $\boldsymbol{\mu}$

Hence "**iteratively reweighted**" least squares.

---

## Part 5: A Complete Worked Example

Let's trace through IRLS by hand for a tiny Poisson regression problem.

### 5.1 Setup

**Data**:
| $i$ | $y_i$ | $x_i$ |
|-----|-------|-------|
| 1 | 1 | 0 |
| 2 | 4 | 1 |
| 3 | 7 | 2 |

**Model**: $\log(\mu_i) = \beta_0 + \beta_1 x_i$

**Design matrix**: 
$$\mathbf{X} = \begin{pmatrix} 1 & 0 \\ 1 & 1 \\ 1 & 2 \end{pmatrix}$$

**Link function**: $g(\mu) = \log(\mu)$, so $g'(\mu) = 1/\mu$ and $g^{-1}(\eta) = e^\eta$

**Variance function**: $V(\mu) = \mu$

### 5.2 Iteration 0: Initialization

Initialize $\mu_i$ from the data (common choice: $\mu_i^{(0)} = y_i + 0.1$ to avoid $\log(0)$):

$$\boldsymbol{\mu}^{(0)} = \begin{pmatrix} 1.1 \\ 4.1 \\ 7.1 \end{pmatrix}$$

Compute initial linear predictor:
$$\boldsymbol{\eta}^{(0)} = \log(\boldsymbol{\mu}^{(0)}) = \begin{pmatrix} 0.095 \\ 1.411 \\ 1.960 \end{pmatrix}$$

We could solve for initial $\boldsymbol{\beta}^{(0)}$ by WLS, but let's just proceed to iteration 1.

### 5.3 Iteration 1

**Step 1: Compute weights**

$$w_i = \frac{1}{V(\mu_i)[g'(\mu_i)]^2} = \frac{1}{\mu_i \cdot (1/\mu_i)^2} = \mu_i$$

$$\mathbf{W}^{(0)} = \text{diag}(1.1, 4.1, 7.1)$$

**Step 2: Compute working response**

$$z_i = \eta_i + (y_i - \mu_i) g'(\mu_i) = \eta_i + \frac{y_i - \mu_i}{\mu_i}$$

$$z_1 = 0.095 + \frac{1 - 1.1}{1.1} = 0.095 - 0.091 = 0.004$$
$$z_2 = 1.411 + \frac{4 - 4.1}{4.1} = 1.411 - 0.024 = 1.387$$
$$z_3 = 1.960 + \frac{7 - 7.1}{7.1} = 1.960 - 0.014 = 1.946$$

$$\mathbf{z}^{(0)} = \begin{pmatrix} 0.004 \\ 1.387 \\ 1.946 \end{pmatrix}$$

**Step 3: Compute $\mathbf{X}^T\mathbf{W}\mathbf{X}$**

$$\mathbf{X}^T\mathbf{W}\mathbf{X} = \begin{pmatrix} 1 & 1 & 1 \\ 0 & 1 & 2 \end{pmatrix} \begin{pmatrix} 1.1 & 0 & 0 \\ 0 & 4.1 & 0 \\ 0 & 0 & 7.1 \end{pmatrix} \begin{pmatrix} 1 & 0 \\ 1 & 1 \\ 1 & 2 \end{pmatrix}$$

$$= \begin{pmatrix} 1.1 & 4.1 & 7.1 \\ 0 & 4.1 & 14.2 \end{pmatrix} \begin{pmatrix} 1 & 0 \\ 1 & 1 \\ 1 & 2 \end{pmatrix} = \begin{pmatrix} 12.3 & 18.3 \\ 18.3 & 32.5 \end{pmatrix}$$

**Step 4: Compute $\mathbf{X}^T\mathbf{W}\mathbf{z}$**

$$\mathbf{X}^T\mathbf{W}\mathbf{z} = \begin{pmatrix} 1.1 & 4.1 & 7.1 \\ 0 & 4.1 & 14.2 \end{pmatrix} \begin{pmatrix} 0.004 \\ 1.387 \\ 1.946 \end{pmatrix} = \begin{pmatrix} 19.51 \\ 33.32 \end{pmatrix}$$

**Step 5: Solve for $\boldsymbol{\beta}^{(1)}$**

$$\boldsymbol{\beta}^{(1)} = (\mathbf{X}^T\mathbf{W}\mathbf{X})^{-1} \mathbf{X}^T\mathbf{W}\mathbf{z}$$

First, compute the inverse:
$$(\mathbf{X}^T\mathbf{W}\mathbf{X})^{-1} = \frac{1}{12.3 \times 32.5 - 18.3^2} \begin{pmatrix} 32.5 & -18.3 \\ -18.3 & 12.3 \end{pmatrix}$$

$$= \frac{1}{64.86} \begin{pmatrix} 32.5 & -18.3 \\ -18.3 & 12.3 \end{pmatrix} = \begin{pmatrix} 0.501 & -0.282 \\ -0.282 & 0.190 \end{pmatrix}$$

Then:
$$\boldsymbol{\beta}^{(1)} = \begin{pmatrix} 0.501 & -0.282 \\ -0.282 & 0.190 \end{pmatrix} \begin{pmatrix} 19.51 \\ 33.32 \end{pmatrix} = \begin{pmatrix} 0.38 \\ 0.83 \end{pmatrix}$$

**Step 6: Update linear predictor and means**

$$\boldsymbol{\eta}^{(1)} = \mathbf{X}\boldsymbol{\beta}^{(1)} = \begin{pmatrix} 0.38 \\ 1.21 \\ 2.04 \end{pmatrix}$$

$$\boldsymbol{\mu}^{(1)} = e^{\boldsymbol{\eta}^{(1)}} = \begin{pmatrix} 1.46 \\ 3.35 \\ 7.69 \end{pmatrix}$$

**Step 7: Compute deviance**

$$D = 2\sum_i [y_i \log(y_i/\mu_i) - (y_i - \mu_i)]$$

$$= 2[(1)\log(1/1.46) - (1-1.46) + (4)\log(4/3.35) - (4-3.35) + (7)\log(7/7.69) - (7-7.69)]$$

$$= 2[(-0.378 + 0.46) + (0.716 - 0.65) + (-0.661 + 0.69)]$$

$$= 2[0.082 + 0.066 + 0.029] = 0.354$$

### 5.4 Iteration 2 (Summary)

Repeating with $\boldsymbol{\mu}^{(1)}$:

- New weights: $\mathbf{W}^{(1)} = \text{diag}(1.46, 3.35, 7.69)$
- New working response computed
- Solve WLS again
- Get $\boldsymbol{\beta}^{(2)} \approx (0.38, 0.83)$ (nearly unchanged!)
- Deviance $\approx 0.354$ (unchanged)

**Converged!** The change in deviance is below tolerance.

### 5.5 Final Results

$$\hat{\beta}_0 = 0.38, \quad \hat{\beta}_1 = 0.83$$

Interpretation: Each unit increase in $x$ multiplies the expected count by $e^{0.83} \approx 2.29$.

---

## Part 6: The Complete IRLS Algorithm

```
Algorithm: IRLS for GLM Fitting
════════════════════════════════════════════════════════════════

Input:
  - y: response vector (n × 1)
  - X: design matrix (n × p)
  - family: distribution family (defines V(μ))
  - link: link function (defines g, g⁻¹, g')
  - config: {max_iter, tolerance, min_weight}

Output:
  - β: coefficient estimates (p × 1)
  - μ: fitted means (n × 1)
  - D: final deviance
  - Σ: covariance matrix (p × p)

════════════════════════════════════════════════════════════════

1. INITIALIZE
   μ⁽⁰⁾ ← family.initialize(y)     // e.g., μ = y + 0.1 for Poisson
   η⁽⁰⁾ ← g(μ⁽⁰⁾)                   // initial linear predictor
   D⁽⁰⁾ ← compute_deviance(y, μ⁽⁰⁾)

2. ITERATE for t = 1, 2, ..., max_iter:

   // 2a. Compute working weights
   For i = 1 to n:
       V_i ← V(μᵢ⁽ᵗ⁻¹⁾)
       g'_i ← g'(μᵢ⁽ᵗ⁻¹⁾)
       w_i ← 1 / (V_i × g'_i²)
       w_i ← max(w_i, min_weight)    // numerical stability
   
   W ← diag(w₁, ..., wₙ)

   // 2b. Compute working response
   For i = 1 to n:
       z_i ← ηᵢ⁽ᵗ⁻¹⁾ + (yᵢ - μᵢ⁽ᵗ⁻¹⁾) × g'(μᵢ⁽ᵗ⁻¹⁾)
   
   // 2c. Solve weighted least squares
   β⁽ᵗ⁾ ← solve(X'WX, X'Wz)         // Using Cholesky decomposition

   // 2d. Update linear predictor and means
   η⁽ᵗ⁾ ← Xβ⁽ᵗ⁾ + offset             // offset is 0 if not specified
   μ⁽ᵗ⁾ ← g⁻¹(η⁽ᵗ⁾)
   
   // Clamp means to valid range
   μ⁽ᵗ⁾ ← family.clamp(μ⁽ᵗ⁾)        // e.g., μ > 0 for Poisson

   // 2e. Compute deviance
   D⁽ᵗ⁾ ← compute_deviance(y, μ⁽ᵗ⁾)

   // 2f. Check convergence
   If |D⁽ᵗ⁾ - D⁽ᵗ⁻¹⁾| / |D⁽ᵗ⁻¹⁾| < tolerance:
       converged ← true
       BREAK

3. COMPUTE COVARIANCE
   Σ_unscaled ← (X'WX)⁻¹
   φ̂ ← estimate_dispersion(y, μ, family)
   Σ ← φ̂ × Σ_unscaled

4. RETURN β, μ, D, Σ, converged

════════════════════════════════════════════════════════════════
```

---

## Part 7: Convergence Properties

### 7.1 When Does IRLS Converge?

IRLS inherits the convergence properties of Fisher scoring:

- **Local convergence**: If started close enough to the MLE, IRLS converges
- **Quadratic rate**: Near the solution, the error roughly squares each iteration
- **Not guaranteed globally**: Can fail with poor initialization or problematic data

### 7.2 Typical Convergence Behavior

| Family | Link | Typical Iterations |
|--------|------|-------------------|
| Gaussian | Identity | **1** (exact in one step!) |
| Poisson | Log | 4-8 |
| Binomial | Logit | 4-10 |
| Gamma | Log | 4-8 |
| Tweedie | Log | 5-15 |

### 7.3 Why Gaussian Converges in One Iteration

For Gaussian with identity link:
- $V(\mu) = 1$, $g'(\mu) = 1$
- Weights $w_i = 1$ (constant)
- Working response $z_i = \mu_i + (y_i - \mu_i) = y_i$

So IRLS becomes ordinary least squares of $\mathbf{y}$ on $\mathbf{X}$—the exact solution in one step!

### 7.4 When Convergence Fails

**Problem 1: Complete Separation (Logistic Regression)**

If one predictor perfectly separates the 0s from the 1s, the MLE doesn't exist—coefficients should be $\pm\infty$.

*Symptom*: Coefficients grow larger each iteration; deviance decreases slowly.

*Solution*: Add regularization (ridge/lasso), remove the separating variable, or use Firth's bias-reduced logistic regression.

**Problem 2: Near-Zero Counts (Poisson)**

If many $y_i = 0$, the means $\mu_i$ can become very small, causing numerical issues.

*Symptom*: Warnings about small weights or invalid means.

*Solution*: Check data quality; consider zero-inflated models.

**Problem 3: Multicollinearity**

If columns of $\mathbf{X}$ are nearly linearly dependent, $\mathbf{X}^T\mathbf{W}\mathbf{X}$ is nearly singular.

*Symptom*: Large standard errors, coefficients jump between iterations.

*Solution*: Remove correlated predictors, add regularization.

---

## Part 8: Implementation in RustyStats

### 8.1 Code Structure

```
crates/rustystats-core/src/solvers/
├── mod.rs              // Module exports
├── irls.rs             // Main IRLS implementation
└── coordinate_descent.rs  // For regularized GLMs
```

### 8.2 Key Data Structures

```rust
/// Configuration for IRLS
pub struct IRLSConfig {
    /// Maximum number of iterations (default: 25)
    pub max_iterations: usize,
    
    /// Convergence tolerance for deviance change (default: 1e-8)
    pub tolerance: f64,
    
    /// Minimum weight to prevent numerical issues (default: 1e-10)
    pub min_weight: f64,
    
    /// Whether to print iteration info (default: false)
    pub verbose: bool,
}

/// Results from IRLS fitting
pub struct IRLSResult {
    /// Estimated coefficients β
    pub coefficients: Array1<f64>,
    
    /// Fitted means μ = g⁻¹(Xβ)
    pub fitted_values: Array1<f64>,
    
    /// Linear predictor η = Xβ + offset
    pub linear_predictor: Array1<f64>,
    
    /// Final deviance
    pub deviance: f64,
    
    /// Number of iterations used
    pub iterations: usize,
    
    /// Whether algorithm converged
    pub converged: bool,
    
    /// Unscaled covariance matrix (X'WX)⁻¹
    pub covariance_unscaled: Array2<f64>,
    
    /// Final IRLS weights
    pub irls_weights: Array1<f64>,
    
    // ... additional fields
}
```

### 8.3 The Core Loop (Simplified)

```rust
pub fn fit_glm(
    y: &Array1<f64>,
    x: &Array2<f64>,
    family: &dyn Family,
    link: &dyn Link,
    config: &IRLSConfig,
) -> Result<IRLSResult> {
    let (n, p) = (y.len(), x.ncols());
    
    // Step 1: Initialize
    let mut mu = family.initialize_mu(y);
    let mut eta = link.link(&mu);
    let mut deviance = family.deviance(y, &mu, None);
    
    // Step 2: Iterate
    for iteration in 1..=config.max_iterations {
        // 2a: Compute weights
        let variance = family.variance(&mu);
        let link_deriv = link.derivative(&mu);
        let weights = compute_weights(&variance, &link_deriv, config.min_weight);
        
        // 2b: Compute working response
        let z = compute_working_response(&eta, y, &mu, &link_deriv);
        
        // 2c: Solve weighted least squares
        let xtwx = compute_xtwx(x, &weights);
        let xtwz = compute_xtwz(x, &weights, &z);
        let beta = cholesky_solve(&xtwx, &xtwz)?;
        
        // 2d: Update
        eta = x.dot(&beta);
        mu = link.inverse(&eta);
        mu = family.clamp_mu(&mu);
        
        // 2e: Check convergence
        let new_deviance = family.deviance(y, &mu, None);
        let relative_change = (deviance - new_deviance).abs() / deviance.max(1e-10);
        
        if relative_change < config.tolerance {
            return Ok(IRLSResult { 
                coefficients: beta, 
                converged: true,
                iterations: iteration,
                // ...
            });
        }
        
        deviance = new_deviance;
    }
    
    // Did not converge
    Ok(IRLSResult { converged: false, ... })
}
```

### 8.4 Parallel Computation

The most expensive operation is computing $\mathbf{X}^T\mathbf{W}\mathbf{X}$, which is $O(np^2)$. RustyStats parallelizes this using Rayon:

```rust
fn compute_xtwx_parallel(x: &Array2<f64>, w: &Array1<f64>) -> Array2<f64> {
    let (n, p) = x.dim();
    
    // Parallel fold-reduce pattern
    (0..n).into_par_iter()
        .fold(
            // Each thread starts with a zero matrix
            || Array2::zeros((p, p)),
            // Accumulate: add w_i * x_i * x_i^T
            |mut acc, i| {
                let xi = x.row(i);
                let wi = w[i];
                for j in 0..p {
                    for k in j..p {  // Only upper triangle (symmetric)
                        acc[[j, k]] += wi * xi[j] * xi[k];
                    }
                }
                acc
            }
        )
        .reduce(
            // Combine thread results
            || Array2::zeros((p, p)),
            |a, b| a + b
        )
        // Fill lower triangle from upper
        .symmetrize()
}
```

This achieves near-linear speedup with the number of cores for large datasets.

---

## Part 9: Numerical Stability

### 9.1 Weight Clamping

Very small weights cause numerical problems:

```rust
fn compute_weights(variance: &Array1<f64>, link_deriv: &Array1<f64>, min_weight: f64) -> Array1<f64> {
    Zip::from(variance).and(link_deriv).map_collect(|&v, &g| {
        let w = 1.0 / (v * g * g);
        w.max(min_weight)  // Prevent division by near-zero
    })
}
```

### 9.2 Mean Clamping

Means must stay in valid ranges:

```rust
// Poisson: μ must be positive
impl Family for PoissonFamily {
    fn clamp_mu(&self, mu: &Array1<f64>) -> Array1<f64> {
        mu.mapv(|m| m.max(1e-10))
    }
}

// Binomial: μ must be in (0, 1)
impl Family for BinomialFamily {
    fn clamp_mu(&self, mu: &Array1<f64>) -> Array1<f64> {
        mu.mapv(|m| m.clamp(1e-10, 1.0 - 1e-10))
    }
}
```

### 9.3 Cholesky Decomposition

Instead of computing $(\mathbf{X}^T\mathbf{W}\mathbf{X})^{-1}$ directly, we use Cholesky decomposition:

1. $\mathbf{X}^T\mathbf{W}\mathbf{X}$ is symmetric positive definite
2. Factor: $\mathbf{X}^T\mathbf{W}\mathbf{X} = \mathbf{L}\mathbf{L}^T$ where $\mathbf{L}$ is lower triangular
3. Solve $\mathbf{L}\mathbf{L}^T\boldsymbol{\beta} = \mathbf{X}^T\mathbf{W}\mathbf{z}$ by forward/backward substitution

This is faster and more numerically stable than computing the inverse.

---

## Part 10: Summary

IRLS converts the nonlinear GLM optimization problem into a sequence of weighted linear regressions:

1. **Linearize** the problem using the working response $\mathbf{z}$
2. **Reweight** based on variance and link curvature using weights $\mathbf{W}$
3. **Solve** weighted least squares to update $\boldsymbol{\beta}$
4. **Repeat** until convergence

Key insights:
- Derived from Fisher scoring (Newton-Raphson with expected Hessian)
- Converges in 1 step for Gaussian/identity (reduces to OLS)
- Typically 4-10 iterations for other families
- Parallelizable for large datasets
- Provides covariance matrix for inference as a byproduct

---

## Exercises

!!! question "Exercise 1: Weights for Binomial/Logit"
    Verify that for Binomial with logit link:
    
    a) $g'(\mu) = \frac{1}{\mu(1-\mu)}$
    
    b) The weight simplifies to $w_i = \mu_i(1-\mu_i)$
    
    c) Why are weights smallest when $\mu \approx 0$ or $\mu \approx 1$?

!!! question "Exercise 2: Working Response"
    For Poisson with log link, show that the working response can be written as:
    
    $$z_i = \log(\mu_i) + \frac{y_i - \mu_i}{\mu_i} = \log(\mu_i) + \frac{y_i}{\mu_i} - 1$$
    
    Interpret this: what happens when $y_i = \mu_i$ (perfect prediction)?

!!! question "Exercise 3: Hand Calculation"
    For the data: $y = (2, 5)$, $x = (0, 1)$, fit a Poisson model $\log(\mu) = \beta_0 + \beta_1 x$.
    
    a) Initialize with $\mu^{(0)} = y$
    
    b) Compute $\mathbf{W}^{(0)}$, $\mathbf{z}^{(0)}$
    
    c) Solve for $\boldsymbol{\beta}^{(1)}$
    
    d) How does this compare to the true MLE?

!!! question "Exercise 4: Gaussian Convergence"
    Prove algebraically that for Gaussian/identity:
    
    a) The weights are $w_i = 1$
    
    b) The working response is $z_i = y_i$
    
    c) One IRLS iteration gives $\boldsymbol{\beta} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$
    
    d) This is the OLS solution, so no further iterations change $\boldsymbol{\beta}$

---

## Further Reading

- McCullagh, P. and Nelder, J.A. (1989). *Generalized Linear Models*, Chapter 2. — Original derivation
- Green, P.J. (1984). Iteratively Reweighted Least Squares for Maximum Likelihood Estimation. *JRSS B*. — Theoretical foundations
- Hastie, T. and Tibshirani, R. (1990). *Generalized Additive Models*. — Extensions to GAMs
