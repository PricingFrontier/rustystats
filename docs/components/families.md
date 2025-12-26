# Families Component

This chapter provides implementation details for each distribution family in RustyStats.

## Code Location

```
crates/rustystats-core/src/families/
├── mod.rs              # Family trait definition
├── gaussian.rs         # Gaussian (Normal)
├── poisson.rs          # Poisson
├── binomial.rs         # Binomial
├── gamma.rs            # Gamma
├── tweedie.rs          # Tweedie
├── quasi.rs            # QuasiPoisson, QuasiBinomial
└── negative_binomial.rs # Negative Binomial
```

## The Family Trait

Every family must implement:

```rust
pub trait Family: Send + Sync {
    /// Display name
    fn name(&self) -> &str;
    
    /// Variance function V(μ)
    fn variance(&self, mu: &Array1<f64>) -> Array1<f64>;
    
    /// Unit deviance d(y, μ) per observation
    fn unit_deviance(&self, y: &Array1<f64>, mu: &Array1<f64>) -> Array1<f64>;
    
    /// Total deviance (sum of weighted unit deviances)
    fn deviance(&self, y: &Array1<f64>, mu: &Array1<f64>, 
                weights: Option<&Array1<f64>>) -> f64;
    
    /// Canonical link function
    fn default_link(&self) -> Box<dyn Link>;
    
    /// Starting values for μ
    fn initialize_mu(&self, y: &Array1<f64>) -> Array1<f64>;
    
    /// Check if μ values are valid
    fn is_valid_mu(&self, mu: &Array1<f64>) -> bool;
}
```

## Gaussian Family

**File**: `gaussian.rs`

### Properties

| Property | Value |
|----------|-------|
| Variance | \(V(\mu) = 1\) |
| Deviance | \(d(y, \mu) = (y - \mu)^2\) |
| Canonical link | Identity |
| Valid μ range | \((-\infty, +\infty)\) |

### Implementation

```rust
pub struct GaussianFamily;

impl Family for GaussianFamily {
    fn name(&self) -> &str { "Gaussian" }
    
    fn variance(&self, mu: &Array1<f64>) -> Array1<f64> {
        Array1::ones(mu.len())  // Constant variance
    }
    
    fn unit_deviance(&self, y: &Array1<f64>, mu: &Array1<f64>) -> Array1<f64> {
        (y - mu).mapv(|r| r * r)  // (y - μ)²
    }
    
    fn default_link(&self) -> Box<dyn Link> {
        Box::new(IdentityLink)
    }
    
    fn initialize_mu(&self, y: &Array1<f64>) -> Array1<f64> {
        y.clone()  // Start at observed values
    }
    
    fn is_valid_mu(&self, _mu: &Array1<f64>) -> bool {
        true  // Any real number is valid
    }
}
```

## Poisson Family

**File**: `poisson.rs`

### Properties

| Property | Value |
|----------|-------|
| Variance | \(V(\mu) = \mu\) |
| Deviance | \(d(y, \mu) = 2[y \log(y/\mu) - (y - \mu)]\) |
| Canonical link | Log |
| Valid μ range | \((0, +\infty)\) |

### Implementation

```rust
pub struct PoissonFamily;

impl Family for PoissonFamily {
    fn name(&self) -> &str { "Poisson" }
    
    fn variance(&self, mu: &Array1<f64>) -> Array1<f64> {
        mu.clone()  // V(μ) = μ
    }
    
    fn unit_deviance(&self, y: &Array1<f64>, mu: &Array1<f64>) -> Array1<f64> {
        Zip::from(y).and(mu).map_collect(|&yi, &mui| {
            let mui_safe = mui.max(1e-10);
            if yi > 0.0 {
                2.0 * (yi * (yi / mui_safe).ln() - (yi - mui_safe))
            } else {
                2.0 * mui_safe  // Limit as y → 0
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

### Note: Handling y = 0

When \(y = 0\), the term \(y \log(y/\mu)\) requires special handling:

\[
\lim_{y \to 0} y \log(y/\mu) = 0
\]

So the unit deviance simplifies to \(2\mu\).

## Binomial Family

**File**: `binomial.rs`

### Properties

| Property | Value |
|----------|-------|
| Variance | \(V(\mu) = \mu(1-\mu)\) |
| Deviance | \(d(y, \mu) = 2[y \log(y/\mu) + (1-y)\log((1-y)/(1-\mu))]\) |
| Canonical link | Logit |
| Valid μ range | \((0, 1)\) |

### Implementation

```rust
pub struct BinomialFamily;

impl Family for BinomialFamily {
    fn name(&self) -> &str { "Binomial" }
    
    fn variance(&self, mu: &Array1<f64>) -> Array1<f64> {
        mu.mapv(|m| {
            let m_safe = m.clamp(1e-10, 1.0 - 1e-10);
            m_safe * (1.0 - m_safe)
        })
    }
    
    fn unit_deviance(&self, y: &Array1<f64>, mu: &Array1<f64>) -> Array1<f64> {
        Zip::from(y).and(mu).map_collect(|&yi, &mui| {
            let mui_safe = mui.clamp(1e-10, 1.0 - 1e-10);
            let yi_safe = yi.clamp(1e-10, 1.0 - 1e-10);
            
            2.0 * (yi_safe * (yi_safe / mui_safe).ln() 
                   + (1.0 - yi_safe) * ((1.0 - yi_safe) / (1.0 - mui_safe)).ln())
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

## Gamma Family

**File**: `gamma.rs`

### Properties

| Property | Value |
|----------|-------|
| Variance | \(V(\mu) = \mu^2\) |
| Deviance | \(d(y, \mu) = 2[-\log(y/\mu) + (y-\mu)/\mu]\) |
| Canonical link | Inverse (\(-1/\mu\)) |
| Common link | Log (used in RustyStats) |
| Valid μ range | \((0, +\infty)\) |

### Implementation

```rust
pub struct GammaFamily;

impl Family for GammaFamily {
    fn name(&self) -> &str { "Gamma" }
    
    fn variance(&self, mu: &Array1<f64>) -> Array1<f64> {
        mu.mapv(|m| m * m)  // V(μ) = μ²
    }
    
    fn unit_deviance(&self, y: &Array1<f64>, mu: &Array1<f64>) -> Array1<f64> {
        Zip::from(y).and(mu).map_collect(|&yi, &mui| {
            let mui_safe = mui.max(1e-10);
            let yi_safe = yi.max(1e-10);
            2.0 * (-(yi_safe / mui_safe).ln() + (yi_safe - mui_safe) / mui_safe)
        })
    }
    
    fn default_link(&self) -> Box<dyn Link> {
        Box::new(LogLink)  // Log link, not canonical inverse
    }
    
    fn initialize_mu(&self, y: &Array1<f64>) -> Array1<f64> {
        y.mapv(|yi| yi.max(0.1))
    }
    
    fn is_valid_mu(&self, mu: &Array1<f64>) -> bool {
        mu.iter().all(|&m| m > 0.0)
    }
}
```

## Tweedie Family

**File**: `tweedie.rs`

### Properties

| Property | Value |
|----------|-------|
| Variance | \(V(\mu) = \mu^p\) where \(p\) is variance power |
| Valid \(p\) | \(p \leq 0\) or \(p \geq 1\) |
| Valid μ range | \((0, +\infty)\) for \(p > 0\) |

### Implementation

```rust
pub struct TweedieFamily {
    pub var_power: f64,
}

impl TweedieFamily {
    pub fn new(var_power: f64) -> Self {
        assert!(var_power <= 0.0 || var_power >= 1.0,
                "var_power must be <= 0 or >= 1");
        Self { var_power }
    }
}

impl Family for TweedieFamily {
    fn name(&self) -> &str { "Tweedie" }
    
    fn variance(&self, mu: &Array1<f64>) -> Array1<f64> {
        let p = self.var_power;
        mu.mapv(|m| m.powf(p))
    }
    
    fn unit_deviance(&self, y: &Array1<f64>, mu: &Array1<f64>) -> Array1<f64> {
        let p = self.var_power;
        
        Zip::from(y).and(mu).map_collect(|&yi, &mui| {
            let mui_safe = mui.max(1e-10);
            
            if (p - 1.0).abs() < 1e-10 {
                // p = 1: Poisson limit
                2.0 * (yi * (yi / mui_safe).ln() - (yi - mui_safe))
            } else if (p - 2.0).abs() < 1e-10 {
                // p = 2: Gamma
                2.0 * (-(yi / mui_safe).ln() + (yi - mui_safe) / mui_safe)
            } else {
                // General case
                let term1 = yi.powf(2.0 - p) / ((1.0 - p) * (2.0 - p));
                let term2 = yi * mui_safe.powf(1.0 - p) / (1.0 - p);
                let term3 = mui_safe.powf(2.0 - p) / (2.0 - p);
                2.0 * (term1 - term2 + term3)
            }
        })
    }
    
    fn default_link(&self) -> Box<dyn Link> {
        Box::new(LogLink)
    }
    
    // ...
}
```

## Quasi-Families

**File**: `quasi.rs`

Quasi-families estimate dispersion from data instead of fixing it at 1.

### QuasiPoisson

```rust
pub struct QuasiPoissonFamily;

impl Family for QuasiPoissonFamily {
    fn name(&self) -> &str { "QuasiPoisson" }
    
    fn variance(&self, mu: &Array1<f64>) -> Array1<f64> {
        mu.clone()  // Same as Poisson
    }
    
    fn unit_deviance(&self, y: &Array1<f64>, mu: &Array1<f64>) -> Array1<f64> {
        // Same deviance formula as Poisson
        // Dispersion is estimated separately
        // ...
    }
    
    // Same initialization and validation as Poisson
}
```

The key difference is in how dispersion is handled in `PyGLMResults`:

```rust
fn scale(&self) -> f64 {
    match self.family_name.as_str() {
        "Poisson" | "Binomial" => 1.0,  // Fixed
        "QuasiPoisson" | "QuasiBinomial" => {
            // Estimate from Pearson residuals
            estimate_dispersion_pearson(...)
        }
        _ => self.deviance / self.df_resid() as f64,
    }
}
```

## Negative Binomial Family

**File**: `negative_binomial.rs`

### Properties

| Property | Value |
|----------|-------|
| Variance | \(V(\mu) = \mu + \mu^2/\theta\) |
| Parameter | \(\theta > 0\) controls overdispersion |
| Valid μ range | \((0, +\infty)\) |

### Implementation

```rust
pub struct NegativeBinomialFamily {
    pub theta: f64,
}

impl NegativeBinomialFamily {
    pub fn new(theta: f64) -> Self {
        assert!(theta > 0.0, "theta must be positive");
        Self { theta }
    }
    
    pub fn alpha(&self) -> f64 {
        1.0 / self.theta  // Alternative parameterization
    }
}

impl Family for NegativeBinomialFamily {
    fn name(&self) -> &str { "NegativeBinomial" }
    
    fn variance(&self, mu: &Array1<f64>) -> Array1<f64> {
        let theta = self.theta;
        mu.mapv(|m| m + m * m / theta)
    }
    
    fn unit_deviance(&self, y: &Array1<f64>, mu: &Array1<f64>) -> Array1<f64> {
        let theta = self.theta;
        
        Zip::from(y).and(mu).map_collect(|&yi, &mui| {
            let mui_safe = mui.max(1e-10);
            
            let term1 = if yi > 0.0 {
                2.0 * yi * (yi / mui_safe).ln()
            } else {
                0.0
            };
            
            let term2 = 2.0 * (yi + theta) * 
                ((yi + theta) / (mui_safe + theta)).ln();
            
            term1 - term2
        })
    }
    
    fn default_link(&self) -> Box<dyn Link> {
        Box::new(LogLink)
    }
    
    // ...
}
```

### Theta Estimation

RustyStats can automatically estimate θ:

```rust
// In diagnostics module
pub fn estimate_theta_moments(
    y: &Array1<f64>,
    mu: &Array1<f64>,
) -> f64 {
    // Method of moments estimator
    let n = y.len() as f64;
    let mean = mu.mean().unwrap();
    
    let var = y.iter().zip(mu.iter())
        .map(|(yi, mui)| (yi - mui).powi(2) / mui)
        .sum::<f64>() / n;
    
    // Var = μ + μ²/θ  →  θ = μ² / (Var - μ)
    let excess_var = (var - 1.0).max(0.01);
    mean / excess_var
}
```

## Testing Families

Each family has unit tests:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_poisson_variance() {
        let family = PoissonFamily;
        let mu = array![1.0, 2.0, 5.0];
        let var = family.variance(&mu);
        
        assert_eq!(var, mu);  // V(μ) = μ
    }
    
    #[test]
    fn test_poisson_deviance_zero() {
        let family = PoissonFamily;
        let y = array![0.0];
        let mu = array![1.0];
        let dev = family.unit_deviance(&y, &mu);
        
        assert_relative_eq!(dev[0], 2.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_binomial_variance_bounds() {
        let family = BinomialFamily;
        let mu = array![0.0001, 0.5, 0.9999];
        let var = family.variance(&mu);
        
        // All variances should be positive and bounded
        for v in var.iter() {
            assert!(*v > 0.0 && *v <= 0.25);
        }
    }
}
```
