# Distribution Families

This chapter provides a comprehensive treatment of the distribution families available in RustyStats. For each family, we derive the key quantities from first principles and show how they connect to the exponential family framework.

**Prerequisites**: Familiarity with the [exponential family and GLM framework](glm-intro.md).

---

## Part 1: The Exponential Family Foundation

### 1.1 Recap: Exponential Family Form

A distribution belongs to the exponential family if its density can be written as:

$$
f(y; \theta, \phi) = \exp\left\{\frac{y\theta - b(\theta)}{\phi} + c(y, \phi)\right\}
$$

where:
- $\theta$ = canonical parameter (related to the mean)
- $\phi$ = dispersion parameter
- $b(\theta)$ = cumulant function
- $c(y, \phi)$ = normalization term

### 1.2 Key Derived Quantities

From $b(\theta)$, we derive:

| Quantity | Formula | Meaning |
|----------|---------|---------|
| Mean | $\mu = b'(\theta)$ | Expected value of $Y$ |
| Variance | $\text{Var}(Y) = \phi \cdot b''(\theta)$ | Spread of $Y$ |
| Variance function | $V(\mu) = b''(\theta)$ | Variance as function of mean |

### 1.3 The Family Trait in RustyStats

Every family in RustyStats implements:

```rust
pub trait Family: Send + Sync {
    /// Distribution name
    fn name(&self) -> &str;
    
    /// Variance function V(μ): how variance depends on mean
    fn variance(&self, mu: &Array1<f64>) -> Array1<f64>;
    
    /// Unit deviance d(y, μ): contribution of each observation
    fn unit_deviance(&self, y: &Array1<f64>, mu: &Array1<f64>) -> Array1<f64>;
    
    /// Total deviance (sum of weighted unit deviances)
    fn deviance(&self, y: &Array1<f64>, mu: &Array1<f64>, 
                weights: Option<&Array1<f64>>) -> f64;
    
    /// Canonical link function for this family
    fn default_link(&self) -> Box<dyn Link>;
    
    /// Initialize μ from y for IRLS
    fn initialize_mu(&self, y: &Array1<f64>) -> Array1<f64>;
    
    /// Check if μ values are in valid range
    fn is_valid_mu(&self, mu: &Array1<f64>) -> bool;
}
```

---

## Part 2: The Gaussian Family

### 2.1 The Distribution

The Gaussian (Normal) distribution has density:

$$
f(y; \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left\{-\frac{(y-\mu)^2}{2\sigma^2}\right\}
$$

**Support**: $y \in (-\infty, +\infty)$

**Parameters**: 
- $\mu$ = mean (location)
- $\sigma^2$ = variance (spread)

### 2.2 Exponential Family Form

Rewrite the density:

$$
f(y; \mu, \sigma^2) = \exp\left\{-\frac{(y-\mu)^2}{2\sigma^2} - \frac{1}{2}\log(2\pi\sigma^2)\right\}
$$

Expand the quadratic:

$$
= \exp\left\{-\frac{y^2 - 2y\mu + \mu^2}{2\sigma^2} - \frac{1}{2}\log(2\pi\sigma^2)\right\}
$$

$$
= \exp\left\{\frac{y\mu - \mu^2/2}{\sigma^2} - \frac{y^2}{2\sigma^2} - \frac{1}{2}\log(2\pi\sigma^2)\right\}
$$

Matching to exponential family form:

| Component | Value |
|-----------|-------|
| $\theta$ | $\mu$ |
| $\phi$ | $\sigma^2$ |
| $b(\theta)$ | $\theta^2/2 = \mu^2/2$ |
| $c(y, \phi)$ | $-\frac{y^2}{2\phi} - \frac{1}{2}\log(2\pi\phi)$ |

### 2.3 Derived Quantities

**Mean**: $\mu = b'(\theta) = \theta$ ✓

**Variance function**: $V(\mu) = b''(\theta) = 1$

**Variance**: $\text{Var}(Y) = \phi \cdot V(\mu) = \sigma^2 \cdot 1 = \sigma^2$ ✓

### 2.4 Unit Deviance

The deviance compares the saturated model (one parameter per observation) to the fitted model. For Gaussian:

$$
D = 2[\ell_{\text{saturated}} - \ell_{\text{fitted}}]
$$

The log-likelihood is:
$$
\ell = -\frac{1}{2\sigma^2}\sum_i (y_i - \mu_i)^2 - \frac{n}{2}\log(2\pi\sigma^2)
$$

For the saturated model, $\hat{\mu}_i = y_i$, so $\ell_{\text{sat}} = -\frac{n}{2}\log(2\pi\sigma^2)$.

For the fitted model with predicted means $\hat{\mu}_i$:
$$
\ell_{\text{fit}} = -\frac{1}{2\sigma^2}\sum_i (y_i - \hat{\mu}_i)^2 - \frac{n}{2}\log(2\pi\sigma^2)
$$

Thus:
$$
D = 2[\ell_{\text{sat}} - \ell_{\text{fit}}] = \frac{1}{\sigma^2}\sum_i (y_i - \hat{\mu}_i)^2
$$

The **scaled deviance** is $D^* = D/\phi = \sum_i (y_i - \hat{\mu}_i)^2$.

The **unit deviance** is:
$$
\boxed{d(y, \mu) = (y - \mu)^2}
$$

This is just the squared residual!

### 2.5 Canonical Link

For Gaussian, $\theta = \mu$, so the canonical link is the **identity**:
$$
\eta = g(\mu) = \mu
$$

### 2.6 IRLS Properties

With identity link:
- Weights: $w_i = 1/[V(\mu_i) \cdot g'(\mu_i)^2] = 1/[1 \cdot 1] = 1$
- Working response: $z_i = \eta_i + (y_i - \mu_i) \cdot 1 = y_i$

IRLS reduces to **ordinary least squares** and converges in **one iteration**.

### 2.7 Implementation

```rust
pub struct GaussianFamily;

impl Family for GaussianFamily {
    fn name(&self) -> &str { "Gaussian" }
    
    fn variance(&self, mu: &Array1<f64>) -> Array1<f64> {
        Array1::ones(mu.len())  // V(μ) = 1
    }
    
    fn unit_deviance(&self, y: &Array1<f64>, mu: &Array1<f64>) -> Array1<f64> {
        (y - mu).mapv(|r| r * r)  // d(y,μ) = (y-μ)²
    }
    
    fn default_link(&self) -> Box<dyn Link> {
        Box::new(IdentityLink)
    }
    
    fn initialize_mu(&self, y: &Array1<f64>) -> Array1<f64> {
        y.clone()  // Start with μ = y
    }
    
    fn is_valid_mu(&self, _mu: &Array1<f64>) -> bool {
        true  // Any real μ is valid
    }
}
```

### 2.8 When to Use

- Continuous responses that can be positive or negative
- When variance doesn't depend on the mean
- Traditional linear regression problems

---

## Part 3: The Poisson Family

### 3.1 The Distribution

The Poisson distribution models counts:

$$
P(Y = y) = \frac{\mu^y e^{-\mu}}{y!}, \quad y = 0, 1, 2, \ldots
$$

**Support**: $y \in \{0, 1, 2, \ldots\}$

**Parameter**: $\mu > 0$ = mean (and variance)

### 3.2 Exponential Family Form

Take the log:
$$
\log P(Y = y) = y \log\mu - \mu - \log(y!)
$$

Exponentiating:
$$
P(Y = y) = \exp\{y \log\mu - \mu - \log(y!)\}
$$

Matching to exponential family:

| Component | Value |
|-----------|-------|
| $\theta$ | $\log\mu$ |
| $\phi$ | $1$ (fixed) |
| $b(\theta)$ | $e^\theta = \mu$ |
| $c(y, \phi)$ | $-\log(y!)$ |

### 3.3 Derived Quantities

**Mean**: $\mu = b'(\theta) = e^\theta$ ✓ (since $\theta = \log\mu$)

**Variance function**: $V(\mu) = b''(\theta) = e^\theta = \mu$

**Variance**: $\text{Var}(Y) = \phi \cdot V(\mu) = 1 \cdot \mu = \mu$

!!! info "Equidispersion"
    For Poisson, mean = variance. This is called **equidispersion** and is a key assumption to check.

### 3.4 Unit Deviance Derivation

The log-likelihood for one observation:
$$
\ell(y; \mu) = y\log\mu - \mu - \log(y!)
$$

Saturated model ($\hat{\mu} = y$):
$$
\ell_{\text{sat}} = y\log y - y - \log(y!) \quad \text{(for } y > 0\text{)}
$$

Fitted model:
$$
\ell_{\text{fit}} = y\log\mu - \mu - \log(y!)
$$

Unit deviance:
$$
d(y, \mu) = 2[\ell_{\text{sat}} - \ell_{\text{fit}}] = 2[y\log y - y - y\log\mu + \mu]
$$

$$
\boxed{d(y, \mu) = 2\left[y\log\frac{y}{\mu} - (y - \mu)\right]}
$$

**Special case**: When $y = 0$:
$$
d(0, \mu) = 2[0 - 0 - 0 + \mu] = 2\mu
$$

### 3.5 Canonical Link

Since $\theta = \log\mu$, the canonical link is the **log**:
$$
\eta = g(\mu) = \log\mu
$$

### 3.6 IRLS Properties

With log link:
- $g'(\mu) = 1/\mu$
- Weights: $w_i = 1/[V(\mu_i) \cdot g'(\mu_i)^2] = 1/[\mu_i \cdot (1/\mu_i)^2] = \mu_i$
- Working response: $z_i = \log\mu_i + (y_i - \mu_i)/\mu_i$

Observations with larger predicted means get higher weight.

### 3.7 Worked Example: Claim Frequency

**Data**: 5 policyholders with claim counts

| Policyholder | Exposure | Claims | Rate |
|--------------|----------|--------|------|
| 1 | 1.0 | 0 | 0.0 |
| 2 | 0.5 | 1 | 2.0 |
| 3 | 1.0 | 2 | 2.0 |
| 4 | 0.75 | 1 | 1.33 |
| 5 | 1.0 | 3 | 3.0 |

**Model**: $\log(\mu_i) = \beta_0 + \log(\text{exposure}_i)$

The offset $\log(\text{exposure})$ makes this a **rate model**:
$$
\log(\mu_i/\text{exposure}_i) = \beta_0
$$

**MLE for constant rate model**:
$$
\hat{\beta}_0 = \log\left(\frac{\sum y_i}{\sum \text{exposure}_i}\right) = \log\left(\frac{7}{4.25}\right) = 0.499
$$

**Predicted rate**: $e^{0.499} = 1.65$ claims per exposure unit.

**Deviance**:
$$
D = 2\sum_i \left[y_i\log\frac{y_i}{\hat{\mu}_i} - (y_i - \hat{\mu}_i)\right]
$$

where $\hat{\mu}_i = 1.65 \times \text{exposure}_i$.

### 3.8 Checking Overdispersion

If Var$(Y) > \mu$, the data is **overdispersed**. Test using:

$$
\text{Dispersion ratio} = \frac{\text{Pearson } \chi^2}{n - p} = \frac{\sum (y_i - \hat{\mu}_i)^2/\hat{\mu}_i}{n - p}
$$

- Ratio $\approx 1$: Poisson is appropriate
- Ratio $> 1$: Consider QuasiPoisson or Negative Binomial

### 3.9 Implementation

```rust
pub struct PoissonFamily;

impl Family for PoissonFamily {
    fn name(&self) -> &str { "Poisson" }
    
    fn variance(&self, mu: &Array1<f64>) -> Array1<f64> {
        mu.clone()  // V(μ) = μ
    }
    
    fn unit_deviance(&self, y: &Array1<f64>, mu: &Array1<f64>) -> Array1<f64> {
        Zip::from(y).and(mu).map_collect(|&yi, &mui| {
            if yi > 0.0 {
                2.0 * (yi * (yi / mui).ln() - (yi - mui))
            } else {
                2.0 * mui  // Special case for y = 0
            }
        })
    }
    
    fn default_link(&self) -> Box<dyn Link> {
        Box::new(LogLink)
    }
    
    fn initialize_mu(&self, y: &Array1<f64>) -> Array1<f64> {
        y.mapv(|yi| (yi + 0.1).max(0.1))  // Avoid log(0)
    }
    
    fn is_valid_mu(&self, mu: &Array1<f64>) -> bool {
        mu.iter().all(|&m| m > 0.0)
    }
}
```

---

## Part 4: The Binomial Family

### 4.1 The Distribution

For $n$ trials with success probability $p$:

$$
P(Y = y) = \binom{n}{y} p^y (1-p)^{n-y}, \quad y = 0, 1, \ldots, n
$$

For binary data ($n = 1$):
$$
P(Y = 1) = \mu, \quad P(Y = 0) = 1 - \mu
$$

**Support**: $y \in \{0, 1\}$ (binary) or $y \in [0, 1]$ (proportion)

**Parameter**: $\mu = p \in (0, 1)$

### 4.2 Exponential Family Form

For binary $Y \in \{0, 1\}$:
$$
P(Y = y) = \mu^y (1-\mu)^{1-y}
$$

Take the log:
$$
\log P(Y = y) = y\log\mu + (1-y)\log(1-\mu) = y\log\frac{\mu}{1-\mu} + \log(1-\mu)
$$

Exponentiating:
$$
P(Y = y) = \exp\left\{y\log\frac{\mu}{1-\mu} + \log(1-\mu)\right\}
$$

Matching to exponential family:

| Component | Value |
|-----------|-------|
| $\theta$ | $\log\frac{\mu}{1-\mu}$ (log-odds) |
| $\phi$ | $1$ (fixed) |
| $b(\theta)$ | $\log(1 + e^\theta)$ |
| $c(y, \phi)$ | $0$ |

### 4.3 Derived Quantities

**Mean**: We have $\mu = \frac{e^\theta}{1 + e^\theta}$. Check:
$$
b'(\theta) = \frac{e^\theta}{1 + e^\theta} = \mu \quad ✓
$$

**Variance function**:
$$
b''(\theta) = \frac{e^\theta}{(1 + e^\theta)^2} = \mu(1-\mu)
$$

So $V(\mu) = \mu(1-\mu)$.

**Variance**: $\text{Var}(Y) = \phi \cdot V(\mu) = \mu(1-\mu)$

The variance is maximized at $\mu = 0.5$ and goes to zero at $\mu \to 0$ or $\mu \to 1$.

### 4.4 Unit Deviance Derivation

Log-likelihood for binary $y$:
$$
\ell(y; \mu) = y\log\mu + (1-y)\log(1-\mu)
$$

Saturated model: $\hat{\mu} = y$, but this gives $\log 0$ when $y \in \{0, 1\}$. By convention, $\ell_{\text{sat}} = 0$.

Fitted model:
$$
\ell_{\text{fit}} = y\log\mu + (1-y)\log(1-\mu)
$$

Unit deviance:
$$
d(y, \mu) = -2\ell_{\text{fit}} = -2[y\log\mu + (1-y)\log(1-\mu)]
$$

Rewriting:
$$
\boxed{d(y, \mu) = 2\left[y\log\frac{y}{\mu} + (1-y)\log\frac{1-y}{1-\mu}\right]}
$$

**Special cases**:
- $y = 1$: $d(1, \mu) = -2\log\mu$
- $y = 0$: $d(0, \mu) = -2\log(1-\mu)$

### 4.5 Canonical Link: The Logit

The canonical link is:
$$
\eta = g(\mu) = \log\frac{\mu}{1-\mu} \quad \text{(log-odds or logit)}
$$

The inverse (logistic function):
$$
\mu = g^{-1}(\eta) = \frac{e^\eta}{1 + e^\eta} = \frac{1}{1 + e^{-\eta}}
$$

The derivative:
$$
g'(\mu) = \frac{d}{d\mu}\log\frac{\mu}{1-\mu} = \frac{1}{\mu} + \frac{1}{1-\mu} = \frac{1}{\mu(1-\mu)}
$$

### 4.6 IRLS Properties

With logit link:
- Weights: $w_i = 1/[V(\mu_i) \cdot g'(\mu_i)^2] = 1/[\mu_i(1-\mu_i) \cdot 1/(\mu_i(1-\mu_i))^2] = \mu_i(1-\mu_i)$
- Working response: $z_i = \log\frac{\mu_i}{1-\mu_i} + \frac{y_i - \mu_i}{\mu_i(1-\mu_i)}$

Weights are largest when $\mu \approx 0.5$ (most uncertainty) and smallest near 0 or 1.

### 4.7 Interpretation: Odds Ratios

The logit model:
$$
\log\frac{\mu}{1-\mu} = \beta_0 + \beta_1 x_1 + \cdots + \beta_p x_p
$$

For a one-unit increase in $x_j$:
$$
\log\frac{\mu_{\text{new}}}{1-\mu_{\text{new}}} - \log\frac{\mu_{\text{old}}}{1-\mu_{\text{old}}} = \beta_j
$$

Thus:
$$
\frac{\text{Odds}_{\text{new}}}{\text{Odds}_{\text{old}}} = e^{\beta_j}
$$

**$e^{\beta_j}$ is the odds ratio** for a one-unit increase in $x_j$.

### 4.8 Worked Example: Claims Occurrence

**Data**: Whether policyholder had a claim

| Age | Has Claim |
|-----|-----------|
| 25 | 1 |
| 35 | 0 |
| 45 | 1 |
| 55 | 1 |
| 65 | 0 |

**Model**: $\text{logit}(\mu_i) = \beta_0 + \beta_1 \cdot \text{age}_i$

Let $x = (\text{age} - 45)/10$ for numerical stability.

**Iteration 1**:
- Initialize: $\mu^{(0)} = (0.55, 0.45, 0.55, 0.55, 0.45)$ (slightly toward observed)
- Compute weights: $w_i = \mu_i(1-\mu_i)$
- Compute working response
- Solve WLS...

After convergence (typically 5-8 iterations):
- $\hat{\beta}_0 \approx 0.4$ (log-odds at age 45)
- $\hat{\beta}_1 \approx 0.05$ (effect of 10-year age increase)
- Odds ratio for 10-year increase: $e^{0.05} \approx 1.05$

### 4.9 Complete Separation

A special problem in logistic regression: if a predictor perfectly separates 0s from 1s, the MLE doesn't exist.

**Example**: All young people have claims, all old people don't.

The algorithm tries $\beta_1 \to \infty$, which never converges.

**Solutions**:
1. Remove the separating variable
2. Add regularization (Ridge/Lasso)
3. Use Firth's penalized likelihood

### 4.10 Implementation

```rust
pub struct BinomialFamily;

impl Family for BinomialFamily {
    fn name(&self) -> &str { "Binomial" }
    
    fn variance(&self, mu: &Array1<f64>) -> Array1<f64> {
        mu.mapv(|m| m * (1.0 - m))  // V(μ) = μ(1-μ)
    }
    
    fn unit_deviance(&self, y: &Array1<f64>, mu: &Array1<f64>) -> Array1<f64> {
        Zip::from(y).and(mu).map_collect(|&yi, &mui| {
            let term1 = if yi > 0.0 { yi * (yi / mui).ln() } else { 0.0 };
            let term2 = if yi < 1.0 { (1.0 - yi) * ((1.0 - yi) / (1.0 - mui)).ln() } else { 0.0 };
            2.0 * (term1 + term2)
        })
    }
    
    fn default_link(&self) -> Box<dyn Link> {
        Box::new(LogitLink)
    }
    
    fn initialize_mu(&self, y: &Array1<f64>) -> Array1<f64> {
        y.mapv(|yi| (yi + 0.5) / 2.0)  // Shrink toward 0.5
    }
    
    fn is_valid_mu(&self, mu: &Array1<f64>) -> bool {
        mu.iter().all(|&m| m > 0.0 && m < 1.0)
    }
}
```

---

## Part 5: The Gamma Family

### 5.1 The Distribution

The Gamma distribution for positive continuous data:

$$
f(y; \alpha, \beta) = \frac{\beta^\alpha}{\Gamma(\alpha)} y^{\alpha-1} e^{-\beta y}, \quad y > 0
$$

Reparameterized in terms of mean $\mu$ and shape $\nu$:
- $\mu = \alpha/\beta$ (mean)
- $\nu = \alpha$ (shape)

$$
E(Y) = \mu, \quad \text{Var}(Y) = \mu^2/\nu
$$

### 5.2 Exponential Family Form

With $\alpha = \nu$ and $\beta = \nu/\mu$:

$$
f(y; \mu, \nu) = \frac{(\nu/\mu)^\nu}{\Gamma(\nu)} y^{\nu-1} \exp\{-\nu y/\mu\}
$$

Rewriting:
$$
\log f = \nu[\log(\nu/\mu) + (\nu-1)/\nu \cdot \log y - y/\mu] - \log\Gamma(\nu)
$$

This is messy, but the key matching gives:

| Component | Value |
|-----------|-------|
| $\theta$ | $-1/\mu$ |
| $\phi$ | $1/\nu$ |
| $b(\theta)$ | $-\log(-\theta)$ |

### 5.3 Derived Quantities

**Mean**: $\mu = b'(\theta) = -1/\theta = \mu$ ✓

**Variance function**: 
$$
b''(\theta) = 1/\theta^2 = \mu^2
$$

So $V(\mu) = \mu^2$.

**Variance**: $\text{Var}(Y) = \phi \cdot V(\mu) = \mu^2/\nu$

### 5.4 Key Property: Constant CV

The coefficient of variation (CV) is:
$$
\text{CV} = \frac{\text{SD}}{\text{Mean}} = \frac{\sqrt{\mu^2/\nu}}{\mu} = \frac{1}{\sqrt{\nu}}
$$

This is **constant** regardless of the mean! This makes Gamma ideal for:
- Claim severity (larger claims are more variable, proportionally)
- Financial data with percentage-based volatility

### 5.5 Unit Deviance

$$
\boxed{d(y, \mu) = 2\left[-\log\frac{y}{\mu} + \frac{y - \mu}{\mu}\right]}
$$

Or equivalently:
$$
d(y, \mu) = 2\left[\frac{y - \mu}{\mu} - \log\frac{y}{\mu}\right]
$$

### 5.6 Links

**Canonical link** (inverse): $\eta = -1/\mu$

This is rarely used because:
- $\mu > 0$ but $\eta$ can be any real number
- The relationship is awkward to interpret

**Common link** (log): $\eta = \log\mu$
- $\mu = e^\eta$ is always positive
- Coefficients have multiplicative interpretation
- Recommended in most applications

### 5.7 IRLS with Log Link

With log link:
- $g'(\mu) = 1/\mu$
- Weights: $w_i = 1/[\mu_i^2 \cdot (1/\mu_i)^2] = 1$

**Constant weights!** This is a happy coincidence when $V(\mu) = \mu^2$ and $g'(\mu) = 1/\mu$.

### 5.8 Worked Example: Claim Severity

**Data**: Claim amounts for 4 claims

| Claim | Amount | log(Amount) |
|-------|--------|-------------|
| 1 | 500 | 6.21 |
| 2 | 1200 | 7.09 |
| 3 | 800 | 6.68 |
| 4 | 2500 | 7.82 |

**Model**: $\log(\mu_i) = \beta_0$ (constant model)

**MLE**: $\hat{\beta}_0 = \frac{1}{4}\sum \log(y_i) = 6.95$

(Note: For Gamma, the MLE of log-mean is the mean of logs, not log of means!)

**Predicted mean**: $\hat{\mu} = e^{6.95} = 1047$

**Dispersion**: Estimated from residuals, $\hat{\phi} \approx 0.15$, so $\hat{\nu} \approx 6.7$.

### 5.9 Implementation

```rust
pub struct GammaFamily;

impl Family for GammaFamily {
    fn name(&self) -> &str { "Gamma" }
    
    fn variance(&self, mu: &Array1<f64>) -> Array1<f64> {
        mu.mapv(|m| m * m)  // V(μ) = μ²
    }
    
    fn unit_deviance(&self, y: &Array1<f64>, mu: &Array1<f64>) -> Array1<f64> {
        Zip::from(y).and(mu).map_collect(|&yi, &mui| {
            2.0 * ((yi - mui) / mui - (yi / mui).ln())
        })
    }
    
    fn default_link(&self) -> Box<dyn Link> {
        Box::new(LogLink)  // Log is more practical than canonical inverse
    }
    
    fn initialize_mu(&self, y: &Array1<f64>) -> Array1<f64> {
        y.mapv(|yi| yi.max(1e-10))  // Ensure positive
    }
    
    fn is_valid_mu(&self, mu: &Array1<f64>) -> bool {
        mu.iter().all(|&m| m > 0.0)
    }
}
```

---

## Part 6: The Tweedie Family

### 6.1 The Power Variance Family

Tweedie distributions have variance function:
$$
V(\mu) = \mu^p
$$

where $p$ is the **variance power**.

| Power $p$ | Distribution | Support |
|-----------|--------------|---------|
| $p = 0$ | Gaussian | $(-\infty, \infty)$ |
| $p = 1$ | Poisson | $\{0, 1, 2, \ldots\}$ |
| $1 < p < 2$ | Compound Poisson-Gamma | $\{0\} \cup (0, \infty)$ |
| $p = 2$ | Gamma | $(0, \infty)$ |
| $p = 3$ | Inverse Gaussian | $(0, \infty)$ |

### 6.2 The Sweet Spot: $1 < p < 2$

For $1 < p < 2$, Tweedie has a remarkable property: it supports **exact zeros AND positive continuous values**.

This is modeled as a compound Poisson process:
$$
Y = \sum_{j=1}^{N} Z_j
$$

where:
- $N \sim \text{Poisson}(\lambda)$ — number of events
- $Z_j \sim \text{Gamma}(\alpha, \beta)$ — individual event sizes (i.i.d.)

**When $N = 0$**: $Y = 0$ (exact zero)
**When $N > 0$**: $Y > 0$ (sum of positive Gammas)

### 6.3 Connection to Frequency-Severity

In insurance, this naturally models **pure premium**:
- $N$ = number of claims (Poisson)
- $Z_j$ = individual claim amount (Gamma)
- $Y$ = total claims = pure premium

The parameters connect:
- $E(N) = \lambda$
- $E(Z) = \alpha/\beta$
- $E(Y) = \lambda \cdot \alpha/\beta = \mu$

### 6.4 Unit Deviance

For $1 < p < 2$:

$$
\boxed{d(y, \mu) = 2\left[\frac{y^{2-p}}{(1-p)(2-p)} - \frac{y\mu^{1-p}}{1-p} + \frac{\mu^{2-p}}{2-p}\right]}
$$

When $y = 0$:
$$
d(0, \mu) = \frac{2\mu^{2-p}}{2-p}
$$

### 6.5 Choosing the Variance Power $p$

- $p \to 1$: More Poisson-like (frequency dominates)
- $p = 1.5$: Balanced (common default)
- $p \to 2$: More Gamma-like (severity dominates)

**Estimating $p$**: Profile the likelihood over a grid of $p$ values.

### 6.6 Implementation

```rust
pub struct TweedieFamily {
    pub var_power: f64,  // p in V(μ) = μ^p
}

impl Family for TweedieFamily {
    fn variance(&self, mu: &Array1<f64>) -> Array1<f64> {
        mu.mapv(|m| m.powf(self.var_power))
    }
    
    fn unit_deviance(&self, y: &Array1<f64>, mu: &Array1<f64>) -> Array1<f64> {
        let p = self.var_power;
        Zip::from(y).and(mu).map_collect(|&yi, &mui| {
            if yi == 0.0 {
                2.0 * mui.powf(2.0 - p) / (2.0 - p)
            } else {
                let t1 = yi.powf(2.0 - p) / ((1.0 - p) * (2.0 - p));
                let t2 = yi * mui.powf(1.0 - p) / (1.0 - p);
                let t3 = mui.powf(2.0 - p) / (2.0 - p);
                2.0 * (t1 - t2 + t3)
            }
        })
    }
    
    fn default_link(&self) -> Box<dyn Link> {
        Box::new(LogLink)
    }
}
```

---

## Part 7: Quasi-Families

### 7.1 The Quasi-Likelihood Approach

Sometimes we don't have a full probability distribution—just:
- A mean model: $E(Y) = \mu$
- A variance model: $\text{Var}(Y) = \phi \cdot V(\mu)$

**Quasi-likelihood** uses just these two relationships without specifying the full distribution.

### 7.2 QuasiPoisson

Same variance function as Poisson, but $\phi$ is estimated:

$$
V(\mu) = \mu, \quad \phi \text{ estimated from data}
$$

**Effect**:
- Point estimates: Identical to Poisson
- Standard errors: Multiplied by $\sqrt{\hat{\phi}}$
- Confidence intervals: Wider if $\hat{\phi} > 1$

### 7.3 QuasiBinomial

Same idea for binomial:

$$
V(\mu) = \mu(1-\mu), \quad \phi \text{ estimated}
$$

### 7.4 Dispersion Estimation

$$
\hat{\phi} = \frac{1}{n-p} \sum_{i=1}^n \frac{(y_i - \hat{\mu}_i)^2}{V(\hat{\mu}_i)}
$$

This is the Pearson chi-square divided by residual degrees of freedom.

### 7.5 When to Use Quasi-Families

- **Quick fix** for overdispersion
- When you don't trust the full distributional assumption
- When AIC/BIC aren't the primary concern

**Caution**: Quasi-likelihood inference is valid for large samples. AIC/BIC interpretation is murky.

---

## Part 8: The Negative Binomial Family

### 8.1 The Distribution

The Negative Binomial with mean $\mu$ and shape $\theta$:

$$
P(Y = y) = \frac{\Gamma(y + \theta)}{\Gamma(\theta) \cdot y!} \left(\frac{\theta}{\theta + \mu}\right)^\theta \left(\frac{\mu}{\theta + \mu}\right)^y
$$

**Variance**:
$$
\text{Var}(Y) = \mu + \frac{\mu^2}{\theta}
$$

### 8.2 Variance Function

$$
V(\mu) = \mu + \frac{\mu^2}{\theta}
$$

This is **quadratic** in $\mu$, unlike QuasiPoisson's linear variance.

### 8.3 The $\theta$ Parameter

- Large $\theta$ ($\to \infty$): Approaches Poisson ($V \to \mu$)
- Small $\theta$ (near 0): Strong overdispersion ($V \to \mu^2/\theta$)

### 8.4 Comparison: QuasiPoisson vs. Negative Binomial

| Aspect | QuasiPoisson | Negative Binomial |
|--------|--------------|-------------------|
| Variance | $\phi \cdot \mu$ (linear) | $\mu + \mu^2/\theta$ (quadratic) |
| True distribution | No | Yes |
| AIC/BIC valid | No | Yes |
| Estimation | $\phi$ from residuals | $\theta$ from likelihood |

**Rule of thumb**: Use NegBin when overdispersion increases with the mean.

### 8.5 Estimating $\theta$

RustyStats estimates $\theta$ using an iterative moment-based approach:

1. **Initial fit**: Fit a Poisson GLM to get starting $\hat{\mu}$
2. **Moment estimation**: Estimate $\theta$ from residuals:
   $$\hat{\theta} = \frac{\bar{\mu}^2}{\text{Var}(Y - \mu) - \bar{\mu}}$$
3. **Iterate**: Fit NegBin GLM with current $\theta$, re-estimate $\theta$, repeat until convergence

**Bounds**: $\theta$ is constrained to $[0.01, 1000]$ to prevent numerical issues.

### 8.6 Numerical Stability

Negative Binomial fitting is numerically challenging due to:

1. **Larger variance function**: $V(\mu) = \mu + \mu^2/\theta$ produces smaller IRLS weights than Poisson
2. **Ill-conditioned design matrices**: Common with splines + target encoding
3. **Large $\theta$ instability**: When $\theta \to \infty$, the log-likelihood involves $\log\Gamma(y + \theta) - \log\Gamma(\theta)$ which loses precision

**RustyStats Solution**:

RustyStats applies **minimum ridge regularization** (α ≥ 1e-6) automatically for NegBin models:

```python
# The formula API automatically applies minimum regularization
result = rs.glm_dict("claims ~ ns(age, df=5) + TE(region)", 
                data, family="negbinomial").fit()

# You'll see "Method: IRLS + Ridge" in the summary
```

!!! warning "Coefficient Interpretation"
    The ridge penalty introduces negligible bias (~1e-6 × coefficient magnitude) but makes inference approximate. For actuarial applications, this bias is typically irrelevant.

### 8.7 Log-Likelihood for Large $\theta$

When $\theta > 100$, the NegBin distribution is essentially Poisson. RustyStats uses the Poisson log-likelihood in this case to avoid numerical overflow:

$$
\ell(\mu; y) = \sum_i \left[ y_i \log\mu_i - \mu_i - \log\Gamma(y_i + 1) \right]
$$

### 8.8 Diagnostics Warnings

The diagnostics system generates warnings for NegBin models:

| Warning | Condition | Recommendation |
|---------|-----------|----------------|
| `negbinomial_regularization` | Always | Informational: ridge penalty applied |
| `negbinomial_large_theta` | $\theta \geq 100$ | Consider Poisson instead |
| `negbinomial_small_theta` | $\theta \leq 0.1$ | Check for missing covariates |

```python
# Access warnings via diagnostics
diag = result.diagnostics(data, ...)
for w in diag.warnings:
    print(f"[{w['type']}]: {w['message']}")
```

---

## Part 9: Summary and Selection Guide

### 9.1 Decision Tree

```
Response type?
│
├── Continuous
│   ├── Can be negative → Gaussian
│   └── Always positive
│       ├── Has exact zeros → Tweedie (p ∈ (1,2))
│       └── No zeros → Gamma
│
└── Discrete
    ├── Binary (0/1) → Binomial
    │   └── Overdispersed? → QuasiBinomial
    │
    └── Counts (0, 1, 2, ...)
        ├── Var ≈ Mean → Poisson
        └── Var > Mean (overdispersion)
            ├── Quick fix → QuasiPoisson
            └── Better model → Negative Binomial
```

### 9.2 Key Formulas Summary

| Family | $V(\mu)$ | $\phi$ | Canonical Link | Support |
|--------|----------|--------|----------------|---------|
| Gaussian | $1$ | estimated | Identity | $\mathbb{R}$ |
| Poisson | $\mu$ | $1$ | Log | $\mathbb{N}_0$ |
| Binomial | $\mu(1-\mu)$ | $1$ | Logit | $\{0,1\}$ |
| Gamma | $\mu^2$ | estimated | Inverse (Log used) | $\mathbb{R}^+$ |
| Tweedie | $\mu^p$ | estimated | Log | $\{0\} \cup \mathbb{R}^+$ |
| NegBin | $\mu + \mu^2/\theta$ | $1$ | Log | $\mathbb{N}_0$ |

---

## Exercises

!!! question "Exercise 1: Exponential Family"
    Show that the Exponential distribution ($f(y) = \lambda e^{-\lambda y}$) is a special case of the Gamma family with $\nu = 1$.

!!! question "Exercise 2: Deviance Calculation"
    For data $y = (3, 0, 2, 5)$ with fitted means $\mu = (2.5, 0.5, 2.5, 4.5)$:
    
    a) Calculate the unit deviances for Poisson
    
    b) Calculate the total deviance
    
    c) What does a high deviance indicate?

!!! question "Exercise 3: Overdispersion"
    You fit a Poisson model and get Pearson $\chi^2 = 180$ with 90 degrees of freedom.
    
    a) Calculate the dispersion ratio
    
    b) Is this evidence of overdispersion?
    
    c) How would QuasiPoisson adjust the standard errors?

!!! question "Exercise 4: Tweedie"
    A Tweedie model with $p = 1.6$ and $\mu = 100$ has $\phi = 2$.
    
    a) Calculate $V(\mu)$
    
    b) Calculate $\text{Var}(Y)$
    
    c) What's the CV?

---

## Further Reading

- McCullagh, P. and Nelder, J.A. (1989). *Generalized Linear Models*, 2nd ed. Chapter 2.
- Jørgensen, B. (1997). *The Theory of Dispersion Models*. — Comprehensive treatment of Tweedie
- Hilbe, J.M. (2014). *Modeling Count Data*. — Poisson, NegBin, and overdispersion
