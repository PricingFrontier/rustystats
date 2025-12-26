# Link Functions

The link function is the bridge between the linear predictor and the mean response. This chapter explains each link function, its mathematical properties, and when to use it.

## What is a Link Function?

In a GLM, we model:

\[
\eta = g(\mu) \quad \Leftrightarrow \quad \mu = g^{-1}(\eta)
\]

where:
- \(\eta = X\beta\) is the **linear predictor** (can be any real number)
- \(\mu = E(Y)\) is the **mean response** (must be in valid range)
- \(g(\cdot)\) is the **link function**
- \(g^{-1}(\cdot)\) is the **inverse link** (also called the mean function)

## The Link Trait

In Rust, every link function implements:

```rust
pub trait Link: Send + Sync {
    fn name(&self) -> &str;
    fn link(&self, mu: &Array1<f64>) -> Array1<f64>;      // η = g(μ)
    fn inverse(&self, eta: &Array1<f64>) -> Array1<f64>;  // μ = g⁻¹(η)
    fn derivative(&self, mu: &Array1<f64>) -> Array1<f64>; // dη/dμ
}
```

The **derivative** is crucial for the IRLS algorithm - it determines how changes in \(\mu\) affect changes in \(\eta\).

---

## Identity Link

**Formula**: \(\eta = \mu\)

The simplest link - no transformation at all.

| Property | Value |
|----------|-------|
| Link | \(g(\mu) = \mu\) |
| Inverse | \(g^{-1}(\eta) = \eta\) |
| Derivative | \(g'(\mu) = 1\) |
| Valid \(\mu\) range | \((-\infty, +\infty)\) |

### Mathematical Properties

- Linear relationship between predictor and mean
- No constraints on predictions
- Canonical link for Gaussian family

### When to Use

- **Gaussian family** (linear regression)
- When the response can be any real number
- When you want coefficients to represent additive effects

### Interpretation

With identity link:
\[
\mu = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots
\]

**A one-unit increase in \(x_1\) adds \(\beta_1\) to the expected response.**

### Code Location

```
crates/rustystats-core/src/links/identity.rs
```

### Implementation

```rust
pub struct IdentityLink;

impl Link for IdentityLink {
    fn name(&self) -> &str { "Identity" }
    
    fn link(&self, mu: &Array1<f64>) -> Array1<f64> {
        mu.clone()
    }
    
    fn inverse(&self, eta: &Array1<f64>) -> Array1<f64> {
        eta.clone()
    }
    
    fn derivative(&self, mu: &Array1<f64>) -> Array1<f64> {
        Array1::ones(mu.len())
    }
}
```

---

## Log Link

**Formula**: \(\eta = \log(\mu)\)

Ensures predictions are always positive.

| Property | Value |
|----------|-------|
| Link | \(g(\mu) = \log(\mu)\) |
| Inverse | \(g^{-1}(\eta) = e^\eta\) |
| Derivative | \(g'(\mu) = 1/\mu\) |
| Valid \(\mu\) range | \((0, +\infty)\) |

### Mathematical Properties

- Maps positive means to real numbers
- Inverse always produces positive values
- Canonical link for Poisson family

### When to Use

- **Poisson family** (counts)
- **Gamma family** (positive continuous)
- Any response that must be positive
- When multiplicative effects make sense

### Interpretation

With log link:
\[
\log(\mu) = \beta_0 + \beta_1 x_1 + \cdots
\]

Exponentiating:
\[
\mu = e^{\beta_0} \cdot e^{\beta_1 x_1} \cdot e^{\beta_2 x_2} \cdots
\]

**A one-unit increase in \(x_1\) multiplies the expected response by \(e^{\beta_1}\).**

This multiplicative factor is called a **relativity** in actuarial pricing.

### Relativities

```python
result = rs.glm("claims ~ age + C(region)", data, family="poisson").fit()

# Relativities = exp(coefficients)
relativities = np.exp(result.params)
# or
print(result.relativities())
```

### Code Location

```
crates/rustystats-core/src/links/log.rs
```

### Implementation

```rust
pub struct LogLink;

impl Link for LogLink {
    fn name(&self) -> &str { "Log" }
    
    fn link(&self, mu: &Array1<f64>) -> Array1<f64> {
        mu.mapv(|m| m.max(1e-10).ln())  // Clamp to avoid log(0)
    }
    
    fn inverse(&self, eta: &Array1<f64>) -> Array1<f64> {
        eta.mapv(|e| e.exp())
    }
    
    fn derivative(&self, mu: &Array1<f64>) -> Array1<f64> {
        mu.mapv(|m| 1.0 / m.max(1e-10))
    }
}
```

---

## Logit Link

**Formula**: \(\eta = \log\frac{\mu}{1-\mu}\)

Maps probabilities to the real line (log-odds transformation).

| Property | Value |
|----------|-------|
| Link | \(g(\mu) = \log\frac{\mu}{1-\mu}\) |
| Inverse | \(g^{-1}(\eta) = \frac{e^\eta}{1+e^\eta} = \frac{1}{1+e^{-\eta}}\) |
| Derivative | \(g'(\mu) = \frac{1}{\mu(1-\mu)}\) |
| Valid \(\mu\) range | \((0, 1)\) |

### Mathematical Properties

- Maps probabilities (0,1) to real numbers
- Symmetric: \(\text{logit}(p) = -\text{logit}(1-p)\)
- Canonical link for Binomial family

### When to Use

- **Binomial family** (binary outcomes)
- Logistic regression
- When the response is a probability

### Interpretation: Odds Ratios

The logit link gives coefficients as **log odds ratios**.

**Odds** of an event:
\[
\text{Odds} = \frac{P(Y=1)}{P(Y=0)} = \frac{\mu}{1-\mu}
\]

With logit link:
\[
\log\left(\frac{\mu}{1-\mu}\right) = \beta_0 + \beta_1 x_1 + \cdots
\]

**A one-unit increase in \(x_1\) multiplies the odds by \(e^{\beta_1}\).**

| \(e^{\beta}\) | Interpretation |
|---------------|----------------|
| 1.0 | No effect on odds |
| 2.0 | Doubles the odds |
| 0.5 | Halves the odds |
| 1.5 | Increases odds by 50% |

### Example

```python
result = rs.fit_glm(binary_outcome, X, family="binomial")
odds_ratios = np.exp(result.params)
print(f"Odds ratios: {odds_ratios}")
```

### Code Location

```
crates/rustystats-core/src/links/logit.rs
```

### Implementation

```rust
pub struct LogitLink;

impl Link for LogitLink {
    fn name(&self) -> &str { "Logit" }
    
    fn link(&self, mu: &Array1<f64>) -> Array1<f64> {
        mu.mapv(|m| {
            let m_clamped = m.clamp(1e-10, 1.0 - 1e-10);
            (m_clamped / (1.0 - m_clamped)).ln()
        })
    }
    
    fn inverse(&self, eta: &Array1<f64>) -> Array1<f64> {
        eta.mapv(|e| 1.0 / (1.0 + (-e).exp()))
    }
    
    fn derivative(&self, mu: &Array1<f64>) -> Array1<f64> {
        mu.mapv(|m| {
            let m_clamped = m.clamp(1e-10, 1.0 - 1e-10);
            1.0 / (m_clamped * (1.0 - m_clamped))
        })
    }
}
```

---

## Canonical Links

The **canonical link** is the link function that arises naturally from the exponential family form. Using the canonical link simplifies the math (the sufficient statistic equals the linear predictor).

| Family | Canonical Link |
|--------|----------------|
| Gaussian | Identity |
| Poisson | Log |
| Binomial | Logit |
| Gamma | Inverse (\(\eta = -1/\mu\)) |

!!! note "Non-Canonical Links"
    You can use non-canonical links. For example, Gamma with log link is common because:
    
    - Log link ensures positive predictions
    - Coefficients have multiplicative interpretation
    - Inverse link can produce negative predictions
    
    RustyStats uses log link as default for Gamma.

---

## Link Function Derivatives in IRLS

The link derivative \(g'(\mu)\) appears in the IRLS working weights:

\[
W = \frac{1}{V(\mu) \cdot [g'(\mu)]^2}
\]

And in the working response:

\[
z = \eta + (y - \mu) \cdot g'(\mu)
\]

| Link | \(g'(\mu)\) | Effect on Weights |
|------|------------|-------------------|
| Identity | 1 | Weights depend only on \(V(\mu)\) |
| Log | \(1/\mu\) | Small \(\mu\) gets higher derivative |
| Logit | \(1/[\mu(1-\mu)]\) | Extreme probabilities get higher derivative |

---

## Choosing a Link Function

### Default Choices

| Family | Default Link | Reason |
|--------|--------------|--------|
| Gaussian | Identity | Mean can be any real number |
| Poisson | Log | Counts must be positive |
| Binomial | Logit | Probabilities in (0,1) |
| Gamma | Log | Amounts must be positive |
| Tweedie | Log | Ensures positive predictions |

### When to Override Defaults

1. **Interpretability**: Log link gives multiplicative effects
2. **Prediction range**: Ensure predictions stay valid
3. **Domain knowledge**: Some links may be more natural for your problem

---

## Adding a New Link Function

See [Adding a New Link](../maintenance/adding-link.md) for instructions on implementing additional link functions like:

- **Probit**: \(\Phi^{-1}(\mu)\) for binomial (normal CDF inverse)
- **Complementary log-log**: \(\log(-\log(1-\mu))\) for rare events
- **Inverse**: \(-1/\mu\) for Gamma (canonical)
- **Power**: \(\mu^\lambda\) family
