# Introduction to Generalized Linear Models

This chapter provides a comprehensive mathematical foundation for understanding Generalized Linear Models (GLMs). We build from first principles, starting with ordinary linear regression and systematically extending it to the full GLM framework. By the end of this chapter, you will understand not just *what* GLMs are, but *why* they work the way they do.

**Prerequisites**: Basic calculus (derivatives, chain rule), linear algebra (matrix multiplication, inverses), and probability (expected value, variance). No prior statistics knowledge is assumed.

---

## Part 1: From Linear Regression to GLMs

### 1.1 Ordinary Linear Regression: A Review

Before diving into GLMs, let's establish a solid foundation with ordinary linear regression (OLS). Even if you've seen this before, this section introduces the notation and concepts we'll generalize later.

#### The Model

In linear regression, we model a continuous response variable $Y$ as a linear function of predictors plus random noise:

$$
Y_i = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \cdots + \beta_{p-1} x_{i,p-1} + \varepsilon_i
$$

where:

- $Y_i$ is the response for observation $i$ (what we're trying to predict)
- $x_{i1}, x_{i2}, \ldots$ are the predictor values for observation $i$
- $\beta_0, \beta_1, \ldots$ are the coefficients (what we estimate)
- $\varepsilon_i$ is the random error term

We can write this more compactly using matrix notation. Define:

$$
\mathbf{y} = \begin{pmatrix} y_1 \\ y_2 \\ \vdots \\ y_n \end{pmatrix}, \quad
\mathbf{X} = \begin{pmatrix} 
1 & x_{11} & x_{12} & \cdots & x_{1,p-1} \\
1 & x_{21} & x_{22} & \cdots & x_{2,p-1} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x_{n1} & x_{n2} & \cdots & x_{n,p-1}
\end{pmatrix}, \quad
\boldsymbol{\beta} = \begin{pmatrix} \beta_0 \\ \beta_1 \\ \vdots \\ \beta_{p-1} \end{pmatrix}
$$

The column of 1s in $\mathbf{X}$ corresponds to the intercept $\beta_0$. Now we can write:

$$
\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}
$$

!!! info "Matrix Multiplication Refresher"
    The product $\mathbf{X}\boldsymbol{\beta}$ gives an $n \times 1$ vector where each element is:
    
    $$
    (\mathbf{X}\boldsymbol{\beta})_i = \sum_{j=0}^{p-1} X_{ij} \beta_j = \beta_0 + \beta_1 x_{i1} + \cdots
    $$
    
    This is exactly the linear combination we want.

#### The Assumptions

Linear regression makes several key assumptions:

1. **Linearity**: $E(Y_i | \mathbf{x}_i) = \mathbf{x}_i^T \boldsymbol{\beta}$ (the mean is a linear function of predictors)
2. **Normality**: $\varepsilon_i \sim N(0, \sigma^2)$ (errors are normally distributed)
3. **Homoscedasticity**: $\text{Var}(\varepsilon_i) = \sigma^2$ (constant variance)
4. **Independence**: The $\varepsilon_i$ are independent of each other

Under these assumptions, we can write:

$$
Y_i \sim N(\mu_i, \sigma^2) \quad \text{where} \quad \mu_i = \mathbf{x}_i^T \boldsymbol{\beta}
$$

This says: "The response $Y_i$ follows a normal distribution with mean $\mu_i$ (which depends on the predictors) and constant variance $\sigma^2$."

#### Estimation: Least Squares

The ordinary least squares (OLS) estimator minimizes the sum of squared residuals:

$$
\hat{\boldsymbol{\beta}}_{\text{OLS}} = \arg\min_{\boldsymbol{\beta}} \sum_{i=1}^n (y_i - \mathbf{x}_i^T \boldsymbol{\beta})^2 = \arg\min_{\boldsymbol{\beta}} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2
$$

Let's derive the solution step by step. Define the sum of squares:

$$
S(\boldsymbol{\beta}) = (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})^T(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})
$$

Expanding this (using $(A-B)^T(A-B) = A^TA - 2A^TB + B^TB$):

$$
S(\boldsymbol{\beta}) = \mathbf{y}^T\mathbf{y} - 2\boldsymbol{\beta}^T\mathbf{X}^T\mathbf{y} + \boldsymbol{\beta}^T\mathbf{X}^T\mathbf{X}\boldsymbol{\beta}
$$

Taking the derivative with respect to $\boldsymbol{\beta}$:

$$
\frac{\partial S}{\partial \boldsymbol{\beta}} = -2\mathbf{X}^T\mathbf{y} + 2\mathbf{X}^T\mathbf{X}\boldsymbol{\beta}
$$

!!! note "Matrix Calculus Rule"
    For a vector $\mathbf{x}$ and matrix $\mathbf{A}$:
    
    - $\frac{\partial}{\partial \mathbf{x}} (\mathbf{a}^T\mathbf{x}) = \mathbf{a}$
    - $\frac{\partial}{\partial \mathbf{x}} (\mathbf{x}^T\mathbf{A}\mathbf{x}) = 2\mathbf{A}\mathbf{x}$ (if $\mathbf{A}$ is symmetric)

Setting the derivative to zero and solving:

$$
-2\mathbf{X}^T\mathbf{y} + 2\mathbf{X}^T\mathbf{X}\boldsymbol{\beta} = \mathbf{0}
$$

$$
\mathbf{X}^T\mathbf{X}\boldsymbol{\beta} = \mathbf{X}^T\mathbf{y}
$$

$$
\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}
$$

This is the famous **normal equations** solution.

!!! info "Why 'Normal Equations'?"
    The name comes from the fact that the residual vector $\mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}}$ is *orthogonal* (perpendicular, or "normal" in geometric terms) to the column space of $\mathbf{X}$. Geometrically, we're projecting $\mathbf{y}$ onto the space spanned by the columns of $\mathbf{X}$.

#### A Worked Example

Let's do a tiny example by hand. Suppose we have 3 observations:

| $i$ | $y_i$ | $x_{i1}$ |
|-----|-------|----------|
| 1 | 2 | 1 |
| 2 | 4 | 2 |
| 3 | 5 | 3 |

Our model is $Y = \beta_0 + \beta_1 x_1 + \varepsilon$.

The matrices are:

$$
\mathbf{y} = \begin{pmatrix} 2 \\ 4 \\ 5 \end{pmatrix}, \quad
\mathbf{X} = \begin{pmatrix} 1 & 1 \\ 1 & 2 \\ 1 & 3 \end{pmatrix}, \quad
\boldsymbol{\beta} = \begin{pmatrix} \beta_0 \\ \beta_1 \end{pmatrix}
$$

Computing $\mathbf{X}^T\mathbf{X}$:

$$
\mathbf{X}^T\mathbf{X} = \begin{pmatrix} 1 & 1 & 1 \\ 1 & 2 & 3 \end{pmatrix} \begin{pmatrix} 1 & 1 \\ 1 & 2 \\ 1 & 3 \end{pmatrix} = \begin{pmatrix} 3 & 6 \\ 6 & 14 \end{pmatrix}
$$

Computing $\mathbf{X}^T\mathbf{y}$:

$$
\mathbf{X}^T\mathbf{y} = \begin{pmatrix} 1 & 1 & 1 \\ 1 & 2 & 3 \end{pmatrix} \begin{pmatrix} 2 \\ 4 \\ 5 \end{pmatrix} = \begin{pmatrix} 11 \\ 25 \end{pmatrix}
$$

Computing $(\mathbf{X}^T\mathbf{X})^{-1}$:

For a 2×2 matrix $\begin{pmatrix} a & b \\ c & d \end{pmatrix}$, the inverse is $\frac{1}{ad-bc}\begin{pmatrix} d & -b \\ -c & a \end{pmatrix}$.

$$
(\mathbf{X}^T\mathbf{X})^{-1} = \frac{1}{3 \cdot 14 - 6 \cdot 6}\begin{pmatrix} 14 & -6 \\ -6 & 3 \end{pmatrix} = \frac{1}{6}\begin{pmatrix} 14 & -6 \\ -6 & 3 \end{pmatrix}
$$

Finally:

$$
\hat{\boldsymbol{\beta}} = \frac{1}{6}\begin{pmatrix} 14 & -6 \\ -6 & 3 \end{pmatrix} \begin{pmatrix} 11 \\ 25 \end{pmatrix} = \frac{1}{6}\begin{pmatrix} 154 - 150 \\ -66 + 75 \end{pmatrix} = \frac{1}{6}\begin{pmatrix} 4 \\ 9 \end{pmatrix} = \begin{pmatrix} 2/3 \\ 3/2 \end{pmatrix}
$$

So $\hat{\beta}_0 = 0.667$ and $\hat{\beta}_1 = 1.5$. The fitted line is $\hat{y} = 0.667 + 1.5x$.

---

### 1.2 The Limitations of Linear Regression

Linear regression works beautifully for continuous data that can reasonably be assumed normal. But many real-world problems involve data that violate these assumptions:

#### Problem 1: Count Data

**Example**: Number of insurance claims per policy, number of website visits, number of defects in manufacturing.

Counts are:

- Non-negative integers (0, 1, 2, 3, ...)
- Often right-skewed (many zeros, few large values)
- Variance typically increases with the mean

Linear regression can predict negative counts (nonsense!) and assumes constant variance (wrong!).

**Concrete example**: If we model claim counts with linear regression and get $\hat{y} = -0.5$, what does that mean? Negative half a claim? This is meaningless.

#### Problem 2: Binary Data

**Example**: Did a customer churn? (yes/no), Did a patient survive? (yes/no), Is this email spam? (yes/no)

Binary outcomes are:

- Either 0 or 1
- What we really want to model is $P(Y=1)$, a probability between 0 and 1

Linear regression can predict probabilities outside [0, 1] (impossible!) and the normality assumption makes no sense for binary data.

**Concrete example**: If we model churn with linear regression and get $\hat{y} = 1.3$ for a customer, what's their churn probability? 130%? Impossible.

#### Problem 3: Positive Continuous Data

**Example**: Insurance claim amounts, time until failure, income.

Positive continuous data is:

- Strictly positive
- Often right-skewed
- Variance often proportional to mean (or mean squared)

Linear regression can predict negative values and assumes the wrong variance structure.

**Concrete example**: Claim amounts can't be negative, but a linear model might predict $\hat{y} = -\$500$. Also, larger claims are more variable than smaller ones—the variance isn't constant.

---

### 1.3 The GLM Solution: Three Components

GLMs solve these problems elegantly by generalizing linear regression in three ways. A GLM has three components:

#### Component 1: Random Component (Distribution Family)

Instead of assuming $Y \sim N(\mu, \sigma^2)$, we allow $Y$ to follow any distribution from the **exponential family**, which includes:

| Distribution | Typical Use Case | Support |
|--------------|------------------|---------|
| Normal (Gaussian) | Continuous data | $(-\infty, +\infty)$ |
| Poisson | Count data | $\{0, 1, 2, \ldots\}$ |
| Binomial | Binary/proportion data | $\{0, 1\}$ or $[0, 1]$ |
| Gamma | Positive continuous data | $(0, +\infty)$ |
| Inverse Gaussian | Positive continuous data | $(0, +\infty)$ |
| Negative Binomial | Overdispersed counts | $\{0, 1, 2, \ldots\}$ |

Each distribution has a **variance function** $V(\mu)$ that describes how variance relates to the mean:

| Distribution | Variance Function $V(\mu)$ | Meaning |
|--------------|---------------------------|---------|
| Normal | $1$ (constant) | Variance doesn't depend on mean |
| Poisson | $\mu$ | Higher mean → higher variance |
| Binomial | $\mu(1-\mu)$ | Max variance at $\mu=0.5$ |
| Gamma | $\mu^2$ | Variance grows quadratically |

#### Component 2: Systematic Component (Linear Predictor)

We keep the linear structure, defining the **linear predictor**:

$$
\eta_i = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \cdots = \mathbf{x}_i^T \boldsymbol{\beta}
$$

This is exactly like linear regression—a linear combination of predictors. The $\eta_i$ can be any real number.

#### Component 3: Link Function

The innovation of GLMs is the **link function** $g(\cdot)$ that connects the mean $\mu$ to the linear predictor $\eta$:

$$
g(\mu_i) = \eta_i = \mathbf{x}_i^T \boldsymbol{\beta}
$$

Or equivalently, the **inverse link** (also called the **mean function**):

$$
\mu_i = g^{-1}(\eta_i) = g^{-1}(\mathbf{x}_i^T \boldsymbol{\beta})
$$

The link function serves two critical purposes:

**Purpose 1: Maps the mean to the correct range**

For Poisson, $\mu > 0$ (means must be positive). Using $g(\mu) = \log(\mu)$ (log link):

- The linear predictor $\eta = \mathbf{x}^T\boldsymbol{\beta}$ can be any real number
- But $\mu = e^\eta > 0$ is always positive

For Binomial, $\mu \in (0, 1)$ (probabilities). Using $g(\mu) = \log\frac{\mu}{1-\mu}$ (logit link):

- The linear predictor $\eta$ can be any real number
- But $\mu = \frac{e^\eta}{1+e^\eta} \in (0, 1)$ is always a valid probability

**Purpose 2: Enables meaningful interpretation**

With log link, $\eta = \log(\mu)$, so:

$$
\log(\mu) = \beta_0 + \beta_1 x_1 + \cdots
$$

A one-unit increase in $x_1$ increases $\log(\mu)$ by $\beta_1$, which means it *multiplies* $\mu$ by $e^{\beta_1}$.

Example: If $\beta_1 = 0.2$, then each unit increase in $x_1$ multiplies the expected count by $e^{0.2} \approx 1.22$, a 22% increase.

---

### 1.4 Putting It Together: The Full GLM

A GLM specifies:

1. **Distribution**: $Y_i \sim \text{SomeDistribution}(\mu_i)$ with variance $\text{Var}(Y_i) = \phi \cdot V(\mu_i)$
2. **Linear predictor**: $\eta_i = \mathbf{x}_i^T \boldsymbol{\beta}$
3. **Link**: $g(\mu_i) = \eta_i$

The parameter $\phi$ is called the **dispersion parameter**. For some families (Poisson, Binomial) it equals 1 by definition. For others (Gaussian, Gamma) it must be estimated.

**Example: Poisson Regression for Count Data**

For modeling counts (like insurance claims):

- Distribution: $Y_i \sim \text{Poisson}(\mu_i)$
- Variance function: $V(\mu) = \mu$, and $\phi = 1$
- Link function: $g(\mu) = \log(\mu)$ (the "log link")

This means:

$$
\log(\mu_i) = \beta_0 + \beta_1 x_{i1} + \cdots
$$

$$
\mu_i = \exp(\beta_0 + \beta_1 x_{i1} + \cdots)
$$

The mean is always positive (good for counts!), and the variance equals the mean (a property of the Poisson distribution).

**Example: Logistic Regression for Binary Data**

For modeling binary outcomes (like customer churn):

- Distribution: $Y_i \sim \text{Bernoulli}(\mu_i)$ where $\mu_i = P(Y_i = 1)$
- Variance function: $V(\mu) = \mu(1-\mu)$, and $\phi = 1$
- Link function: $g(\mu) = \log\frac{\mu}{1-\mu}$ (the "logit link")

This means:

$$
\log\frac{\mu_i}{1-\mu_i} = \beta_0 + \beta_1 x_{i1} + \cdots
$$

The quantity $\frac{\mu}{1-\mu}$ is called the **odds**. If $\mu = 0.75$, the odds are $\frac{0.75}{0.25} = 3$ ("3 to 1 in favor"). 

The $\log\frac{\mu}{1-\mu}$ is the **log-odds**. Solving for $\mu$:

$$
\mu_i = \frac{e^{\eta_i}}{1 + e^{\eta_i}} = \frac{1}{1 + e^{-\eta_i}}
$$

This is the famous **logistic function** (also called sigmoid), which maps any real number to the interval (0, 1)—perfect for probabilities!

```
μ
1 |                 ___________
  |              __/
  |           __/
  |        __/
  |     __/
0 |____/
  +---------------------------- η
      -4  -2   0   2   4
```

The logistic function looks like an S-curve: very small $\eta$ gives $\mu \approx 0$, very large $\eta$ gives $\mu \approx 1$, and $\eta = 0$ gives $\mu = 0.5$.

---

## Part 2: The Exponential Family Foundation

Understanding *why* certain distributions work well in GLMs requires understanding the **exponential family**. This section develops the mathematical theory that underlies everything. It's more technical, but understanding it will give you deep insight into how GLMs work.

### 2.1 What is the Exponential Family?

A probability distribution belongs to the exponential family if its probability density (or mass) function can be written as:

$$
f(y; \theta, \phi) = \exp\left(\frac{y\theta - b(\theta)}{a(\phi)} + c(y, \phi)\right)
$$

where:

- $\theta$ is the **canonical parameter** (related to the mean)
- $\phi$ is the **dispersion parameter** (scale parameter)
- $a(\phi)$ is typically just $\phi$ (we'll use this simplification)
- $b(\theta)$ is the **cumulant function** (determines the distribution)
- $c(y, \phi)$ is the **normalizing term** (ensures probabilities sum/integrate to 1)

!!! note "Why This Form Matters"
    This might look like arbitrary notation, but this specific form has powerful mathematical properties:
    
    1. Derivatives of $b(\theta)$ give us moments of the distribution
    2. The likelihood equations have a nice form
    3. The canonical link simplifies everything further

### 2.2 Key Properties of the Exponential Family

Here are the key results (we'll prove them):

**Property 1: Mean equals derivative of b**

$$
E(Y) = \mu = b'(\theta) = \frac{db(\theta)}{d\theta}
$$

**Property 2: Variance from second derivative**

$$
\text{Var}(Y) = a(\phi) \cdot b''(\theta) = \phi \cdot V(\mu)
$$

where $V(\mu) = b''(\theta)$ is the variance function, expressed in terms of $\mu$.

**Property 3: Canonical link**

The **canonical link** is defined by $g(\mu) = \theta$. This directly links the mean to the canonical parameter.

Let's prove Property 1 and 2.

#### Proof of Properties 1 and 2

The density must integrate to 1:

$$
\int f(y; \theta, \phi) dy = 1
$$

Substituting the exponential family form:

$$
\int \exp\left(\frac{y\theta - b(\theta)}{\phi} + c(y, \phi)\right) dy = 1
$$

This can be rewritten as:

$$
\exp\left(-\frac{b(\theta)}{\phi}\right) \int \exp\left(\frac{y\theta}{\phi} + c(y, \phi)\right) dy = 1
$$

Taking the derivative with respect to $\theta$ of both sides:

$$
\frac{d}{d\theta}\left[\int f(y; \theta, \phi) dy\right] = \frac{d}{d\theta}[1] = 0
$$

By Leibniz's rule (swapping derivative and integral):

$$
\int \frac{\partial f(y; \theta, \phi)}{\partial \theta} dy = 0
$$

Now, $\frac{\partial}{\partial \theta}\left[\frac{y\theta - b(\theta)}{\phi}\right] = \frac{y - b'(\theta)}{\phi}$, so:

$$
\frac{\partial f}{\partial \theta} = f(y; \theta, \phi) \cdot \frac{y - b'(\theta)}{\phi}
$$

Therefore:

$$
\int f(y) \cdot \frac{y - b'(\theta)}{\phi} dy = 0
$$

$$
\frac{1}{\phi}\left[\int y \cdot f(y) dy - b'(\theta) \int f(y) dy\right] = 0
$$

$$
E(Y) - b'(\theta) \cdot 1 = 0
$$

$$
E(Y) = b'(\theta) = \mu \quad \checkmark
$$

For the variance, differentiate again with respect to $\theta$. After similar calculations:

$$
\text{Var}(Y) = \phi \cdot b''(\theta) \quad \checkmark
$$

### 2.3 Examples: Deriving Variance Functions

Let's verify these properties for specific distributions by putting them in exponential family form.

#### Poisson Distribution

The Poisson probability mass function is:

$$
P(Y = y) = \frac{\mu^y e^{-\mu}}{y!} \quad \text{for } y = 0, 1, 2, \ldots
$$

**Step 1: Rewrite using exponential**

$$
P(Y = y) = \exp\left(\log\left(\frac{\mu^y e^{-\mu}}{y!}\right)\right) = \exp\left(y\log\mu - \mu - \log(y!)\right)
$$

**Step 2: Match to exponential family form**

Comparing with $\exp\left(\frac{y\theta - b(\theta)}{\phi} + c(y, \phi)\right)$:

- $\theta = \log\mu$ (canonical parameter)
- $b(\theta) = e^\theta = \mu$ (since $\theta = \log\mu$ means $\mu = e^\theta$)
- $\phi = 1$ (no separate dispersion for Poisson)
- $c(y, \phi) = -\log(y!)$

**Step 3: Verify properties**

- $b'(\theta) = \frac{d}{d\theta}e^\theta = e^\theta = \mu$ ✓ (confirms $E(Y) = \mu$)
- $b''(\theta) = \frac{d^2}{d\theta^2}e^\theta = e^\theta = \mu$, so $V(\mu) = \mu$ ✓ (variance equals mean)
- Canonical link: $g(\mu) = \theta = \log\mu$ (the log link) ✓

#### Bernoulli (Binomial with n=1) Distribution

The Bernoulli PMF is:

$$
P(Y = y) = \mu^y (1-\mu)^{1-y} \quad \text{for } y \in \{0, 1\}
$$

**Step 1: Rewrite**

$$
P(Y = y) = \exp\left(y\log\mu + (1-y)\log(1-\mu)\right)
$$

$$
= \exp\left(y\log\mu - y\log(1-\mu) + \log(1-\mu)\right)
$$

$$
= \exp\left(y\log\frac{\mu}{1-\mu} + \log(1-\mu)\right)
$$

**Step 2: Match to exponential family form**

- $\theta = \log\frac{\mu}{1-\mu}$ (log-odds)
- $b(\theta) = \log(1 + e^\theta)$ (since $\log(1-\mu) = -\log(1+e^\theta)$ when $\theta = \log\frac{\mu}{1-\mu}$)
- $\phi = 1$
- $c(y, \phi) = 0$

**Step 3: Verify properties**

To check $b'(\theta) = \mu$:

$$
b'(\theta) = \frac{e^\theta}{1+e^\theta}
$$

And indeed, if $\theta = \log\frac{\mu}{1-\mu}$, then $e^\theta = \frac{\mu}{1-\mu}$, so:

$$
b'(\theta) = \frac{\mu/(1-\mu)}{1 + \mu/(1-\mu)} = \frac{\mu/(1-\mu)}{(1-\mu+\mu)/(1-\mu)} = \frac{\mu/(1-\mu)}{1/(1-\mu)} = \mu \quad \checkmark
$$

For the variance:

$$
b''(\theta) = \frac{e^\theta}{(1+e^\theta)^2} = \frac{\mu/(1-\mu)}{(1/(1-\mu))^2} = \mu(1-\mu) \quad \checkmark
$$

So $V(\mu) = \mu(1-\mu)$, which is the variance of a Bernoulli distribution.

Canonical link: $g(\mu) = \theta = \log\frac{\mu}{1-\mu}$ (logit) ✓

#### Normal (Gaussian) Distribution

The Normal PDF is:

$$
f(y) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y-\mu)^2}{2\sigma^2}\right)
$$

**Step 1: Expand the square**

$$
-\frac{(y-\mu)^2}{2\sigma^2} = -\frac{y^2 - 2y\mu + \mu^2}{2\sigma^2} = \frac{y\mu}{\sigma^2} - \frac{\mu^2}{2\sigma^2} - \frac{y^2}{2\sigma^2}
$$

So:

$$
f(y) = \exp\left(\frac{y\mu - \mu^2/2}{\sigma^2} - \frac{y^2}{2\sigma^2} - \frac{1}{2}\log(2\pi\sigma^2)\right)
$$

**Step 2: Match to exponential family form**

- $\theta = \mu$ (canonical parameter equals mean!)
- $b(\theta) = \theta^2/2 = \mu^2/2$
- $\phi = \sigma^2$ (dispersion is variance)
- $c(y, \phi) = -\frac{y^2}{2\phi} - \frac{1}{2}\log(2\pi\phi)$

**Step 3: Verify**

- $b'(\theta) = \theta = \mu$ ✓
- $b''(\theta) = 1$, so $V(\mu) = 1$ ✓ (constant variance)
- Canonical link: $g(\mu) = \theta = \mu$ (identity link) ✓

---

## Part 3: Maximum Likelihood Estimation

Now that we understand the GLM structure, how do we estimate the parameters $\boldsymbol{\beta}$? The answer is **maximum likelihood estimation (MLE)**.

### 3.1 The Likelihood Principle

The **likelihood function** answers: "Given the data we observed, how likely is it that the true parameters are $\boldsymbol{\beta}$?"

Mathematically, if we observe data $y_1, \ldots, y_n$ (assumed independent), the likelihood is:

$$
L(\boldsymbol{\beta}) = \prod_{i=1}^n f(y_i; \mu_i(\boldsymbol{\beta}), \phi)
$$

where $\mu_i(\boldsymbol{\beta}) = g^{-1}(\mathbf{x}_i^T\boldsymbol{\beta})$ is the predicted mean for observation $i$.

The **maximum likelihood estimate** $\hat{\boldsymbol{\beta}}$ is the value that maximizes $L(\boldsymbol{\beta})$.

### 3.2 The Log-Likelihood

Working with products is cumbersome, so we take the logarithm:

$$
\ell(\boldsymbol{\beta}) = \log L(\boldsymbol{\beta}) = \sum_{i=1}^n \log f(y_i; \mu_i, \phi)
$$

Since $\log$ is monotonically increasing, maximizing $\ell$ is equivalent to maximizing $L$.

For exponential family distributions:

$$
\ell(\boldsymbol{\beta}) = \sum_{i=1}^n \left[\frac{y_i\theta_i - b(\theta_i)}{\phi} + c(y_i, \phi)\right]
$$

The $c(y_i, \phi)$ term doesn't depend on $\boldsymbol{\beta}$, so for optimization we can ignore it:

$$
\ell(\boldsymbol{\beta}) \propto \sum_{i=1}^n \frac{y_i\theta_i - b(\theta_i)}{\phi}
$$

### 3.3 The Score Function

To find the maximum, we take the derivative of $\ell$ with respect to $\boldsymbol{\beta}$ and set it to zero.

The **score function** is:

$$
\mathbf{U}(\boldsymbol{\beta}) = \frac{\partial \ell}{\partial \boldsymbol{\beta}} = \begin{pmatrix} \frac{\partial \ell}{\partial \beta_0} \\ \frac{\partial \ell}{\partial \beta_1} \\ \vdots \end{pmatrix}
$$

Let's compute $\frac{\partial \ell}{\partial \beta_j}$ using the chain rule:

$$
\frac{\partial \ell}{\partial \beta_j} = \sum_{i=1}^n \frac{\partial \ell_i}{\partial \theta_i} \cdot \frac{\partial \theta_i}{\partial \mu_i} \cdot \frac{\partial \mu_i}{\partial \eta_i} \cdot \frac{\partial \eta_i}{\partial \beta_j}
$$

Let's compute each piece:

**Piece 1**: $\frac{\partial \ell_i}{\partial \theta_i}$

$$
\ell_i = \frac{y_i\theta_i - b(\theta_i)}{\phi}
$$

$$
\frac{\partial \ell_i}{\partial \theta_i} = \frac{y_i - b'(\theta_i)}{\phi} = \frac{y_i - \mu_i}{\phi}
$$

**Piece 2**: $\frac{\partial \theta_i}{\partial \mu_i}$

Since $\mu = b'(\theta)$, we have $\frac{d\mu}{d\theta} = b''(\theta) = V(\mu)$, so:

$$
\frac{d\theta}{d\mu} = \frac{1}{V(\mu)}
$$

**Piece 3**: $\frac{\partial \mu_i}{\partial \eta_i}$

Since $\mu = g^{-1}(\eta)$, we have:

$$
\frac{d\mu}{d\eta} = \frac{d g^{-1}(\eta)}{d\eta} = \frac{1}{g'(\mu)}
$$

(This uses the inverse function derivative rule: if $\eta = g(\mu)$, then $\frac{d\mu}{d\eta} = 1/\frac{d\eta}{d\mu} = 1/g'(\mu)$.)

**Piece 4**: $\frac{\partial \eta_i}{\partial \beta_j}$

Since $\eta_i = \mathbf{x}_i^T\boldsymbol{\beta} = \sum_k x_{ik}\beta_k$:

$$
\frac{\partial \eta_i}{\partial \beta_j} = x_{ij}
$$

**Putting it all together**:

$$
\frac{\partial \ell}{\partial \beta_j} = \sum_{i=1}^n \frac{y_i - \mu_i}{\phi} \cdot \frac{1}{V(\mu_i)} \cdot \frac{1}{g'(\mu_i)} \cdot x_{ij}
$$

$$
= \sum_{i=1}^n \frac{(y_i - \mu_i) x_{ij}}{\phi \cdot V(\mu_i) \cdot g'(\mu_i)}
$$

Setting this to zero for each $j$ gives the **score equations**:

$$
\sum_{i=1}^n \frac{(y_i - \mu_i) x_{ij}}{V(\mu_i) \cdot g'(\mu_i)} = 0 \quad \text{for } j = 0, 1, \ldots, p-1
$$

### 3.4 Why We Need Iteration

Unlike linear regression, these equations are **nonlinear** in $\boldsymbol{\beta}$.

For linear regression:
- $\mu_i = \eta_i = \mathbf{x}_i^T\boldsymbol{\beta}$ (linear in $\boldsymbol{\beta}$)
- Score equations are linear → closed-form solution

For GLMs:
- $\mu_i = g^{-1}(\mathbf{x}_i^T\boldsymbol{\beta})$ (nonlinear in $\boldsymbol{\beta}$ through $g^{-1}$)
- $V(\mu_i)$ and $g'(\mu_i)$ also depend on $\boldsymbol{\beta}$
- Score equations are nonlinear → need iterative solution

Example for Poisson:
$$
\mu_i = e^{\mathbf{x}_i^T\boldsymbol{\beta}}
$$

The score equation involves $e^{\mathbf{x}_i^T\boldsymbol{\beta}}$ terms—clearly nonlinear!

---

## Part 4: Deviance and Model Assessment

Before moving to the estimation algorithm, let's understand how we measure model fit.

### 4.1 The Concept of Deviance

The **deviance** measures how much our model deviates from a perfect fit. It's defined as:

$$
D = 2[\ell(\text{saturated}) - \ell(\text{fitted})]
$$

where:
- **Saturated model**: A model with one parameter per observation, achieving $\hat{\mu}_i = y_i$ (perfect fit to data)
- **Fitted model**: Our actual model with $p$ parameters

The deviance is always non-negative (the saturated model has the highest possible likelihood).

### 4.2 Unit Deviance

Each observation contributes to the total deviance:

$$
D = \sum_{i=1}^n d(y_i, \mu_i)
$$

where $d(y, \mu)$ is the **unit deviance**.

| Family | Unit Deviance $d(y, \mu)$ |
|--------|---------------------------|
| Gaussian | $(y - \mu)^2$ |
| Poisson | $2[y\log(y/\mu) - (y - \mu)]$ |
| Binomial | $2[y\log(y/\mu) + (1-y)\log((1-y)/(1-\mu))]$ |
| Gamma | $2[-\log(y/\mu) + (y - \mu)/\mu]$ |

!!! note "Deviance for Gaussian"
    For Gaussian, the deviance is $D = \sum_i (y_i - \mu_i)^2$, the residual sum of squares! 
    
    This is why linear regression minimizes sum of squares—it's equivalent to maximizing likelihood for normal data.

### 4.3 Using Deviance

**Model comparison**: For nested models (Model 1 is a special case of Model 2):

$$
D_1 - D_2 \sim \chi^2_{p_2 - p_1} \quad \text{(approximately)}
$$

This is the **likelihood ratio test**.

**Assessing fit**: The residual deviance should be roughly equal to its degrees of freedom ($n - p$). If deviance >> $n - p$, the model fits poorly.

---

## Part 5: Summary and Next Steps

We've built up GLMs from first principles:

1. **Linear regression** works for normal data but fails for counts, binary data, etc.
2. **GLMs** generalize by allowing different distributions (families) and using link functions
3. **The exponential family** provides the mathematical foundation with nice properties
4. **Maximum likelihood** gives us the estimation framework
5. **Deviance** measures model fit

**What's next**: The [IRLS Algorithm](irls.md) chapter shows how we actually solve the nonlinear score equations efficiently by converting them to iterated weighted least squares problems.

---

## Exercises

!!! question "Exercise 1: Link Function Practice"
    For each scenario, which link function is most appropriate and why?
    
    a) Modeling the probability a customer will click an ad  
    b) Modeling the number of cars passing a sensor per hour  
    c) Modeling a student's test score (0-100 continuous)  
    d) Modeling insurance claim severity (positive dollar amounts)

!!! question "Exercise 2: Exponential Family - Gamma Distribution"
    The Gamma PDF is:
    
    $$
    f(y; \mu, \nu) = \frac{1}{\Gamma(\nu)}\left(\frac{\nu}{\mu}\right)^\nu y^{\nu-1} e^{-\nu y/\mu}
    $$
    
    where $\nu$ is the shape parameter.
    
    a) Put this in exponential family form. What is $\theta$?  
    b) Find $b(\theta)$ and compute $b'(\theta)$ to verify $E(Y) = \mu$  
    c) Find $V(\mu)$. What is the canonical link?

!!! question "Exercise 3: Score Equations for Poisson"
    For Poisson regression with log link:
    
    a) Write out the log-likelihood explicitly  
    b) Compute $\frac{\partial \ell}{\partial \beta_j}$ directly (not using the chain rule result)  
    c) Verify it matches the general formula with $V(\mu) = \mu$ and $g'(\mu) = 1/\mu$

!!! question "Exercise 4: Hand Calculation"
    Given the tiny dataset:
    
    | $y$ | $x$ |
    |-----|-----|
    | 1 | 0 |
    | 3 | 1 |
    | 7 | 2 |
    
    For a Poisson model $\log(\mu) = \beta_0 + \beta_1 x$:
    
    a) If $\beta_0 = 0.5$ and $\beta_1 = 0.8$, what are the fitted means $\mu_i$?  
    b) Compute the deviance  
    c) Compute the Pearson residuals $(y_i - \mu_i)/\sqrt{\mu_i}$

---

## Solutions

??? success "Solution to Exercise 1"
    a) **Logit link** - probability must be in (0,1)  
    b) **Log link** - counts must be positive  
    c) **Identity link** - continuous on full range (or bounded, could consider)  
    d) **Log link** - amounts must be positive

??? success "Solution to Exercise 4"
    a) $\mu_1 = e^{0.5} \approx 1.65$, $\mu_2 = e^{0.5+0.8} \approx 3.67$, $\mu_3 = e^{0.5+1.6} \approx 8.17$
    
    b) Deviance = $2\sum[y_i\log(y_i/\mu_i) - (y_i - \mu_i)]$
       
       $= 2[(1)\log(1/1.65) - (1-1.65) + (3)\log(3/3.67) - (3-3.67) + (7)\log(7/8.17) - (7-8.17)]$
       
       $= 2[(-0.50 + 0.65) + (-0.60 + 0.67) + (-1.10 + 1.17)]$
       
       $= 2[0.15 + 0.07 + 0.07] = 0.58$
    
    c) Pearson residuals: $(1-1.65)/\sqrt{1.65} \approx -0.51$, $(3-3.67)/\sqrt{3.67} \approx -0.35$, $(7-8.17)/\sqrt{8.17} \approx -0.41$

---

## Further Reading

- McCullagh, P. and Nelder, J.A. (1989). *Generalized Linear Models*, 2nd ed. Chapman & Hall. — The classic reference.
- Dobson, A.J. and Barnett, A.G. (2018). *An Introduction to Generalized Linear Models*, 4th ed. CRC Press. — More accessible introduction.
- Wood, S.N. (2017). *Generalized Additive Models: An Introduction with R*, 2nd ed. CRC Press. — Extends to nonlinear effects.
