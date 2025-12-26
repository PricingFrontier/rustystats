# Link Functions Component

This chapter covers the implementation of link functions in RustyStats.

## Code Location

```
crates/rustystats-core/src/links/
├── mod.rs       # Link trait definition
├── identity.rs  # Identity link
├── log.rs       # Log link
└── logit.rs     # Logit link
```

## The Link Trait

```rust
pub trait Link: Send + Sync {
    /// Display name
    fn name(&self) -> &str;
    
    /// Forward transformation: η = g(μ)
    fn link(&self, mu: &Array1<f64>) -> Array1<f64>;
    
    /// Inverse transformation: μ = g⁻¹(η)
    fn inverse(&self, eta: &Array1<f64>) -> Array1<f64>;
    
    /// Derivative: dη/dμ
    fn derivative(&self, mu: &Array1<f64>) -> Array1<f64>;
}
```

The `Send + Sync` bounds allow link functions to be used across threads (required for parallel IRLS).

## Identity Link

**File**: `identity.rs`

| Property | Formula |
|----------|---------|
| Link | \(g(\mu) = \mu\) |
| Inverse | \(g^{-1}(\eta) = \eta\) |
| Derivative | \(g'(\mu) = 1\) |

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

### Notes

- Simplest link - no transformation
- Used with Gaussian family (linear regression)
- Predictions can be any real number (no bounds)

## Log Link

**File**: `log.rs`

| Property | Formula |
|----------|---------|
| Link | \(g(\mu) = \log(\mu)\) |
| Inverse | \(g^{-1}(\eta) = e^\eta\) |
| Derivative | \(g'(\mu) = 1/\mu\) |

### Implementation

```rust
pub struct LogLink;

impl Link for LogLink {
    fn name(&self) -> &str { "Log" }
    
    fn link(&self, mu: &Array1<f64>) -> Array1<f64> {
        mu.mapv(|m| {
            // Clamp to avoid log(0) = -∞
            m.max(1e-10).ln()
        })
    }
    
    fn inverse(&self, eta: &Array1<f64>) -> Array1<f64> {
        eta.mapv(|e| e.exp())
    }
    
    fn derivative(&self, mu: &Array1<f64>) -> Array1<f64> {
        mu.mapv(|m| {
            // Clamp to avoid division by zero
            1.0 / m.max(1e-10)
        })
    }
}
```

### Numerical Considerations

1. **Log of small values**: Clamp μ to avoid `log(0) = -∞`
2. **Derivative at small μ**: Large derivative can cause numerical issues
3. **Exp overflow**: Very large η can cause `exp(η) = ∞`

### Overflow Protection

For very large η, exp can overflow. Consider adding bounds:

```rust
fn inverse(&self, eta: &Array1<f64>) -> Array1<f64> {
    eta.mapv(|e| {
        // Prevent overflow: exp(700) ≈ 1e304
        e.min(700.0).exp()
    })
}
```

## Logit Link

**File**: `logit.rs`

| Property | Formula |
|----------|---------|
| Link | \(g(\mu) = \log\frac{\mu}{1-\mu}\) |
| Inverse | \(g^{-1}(\eta) = \frac{1}{1+e^{-\eta}}\) |
| Derivative | \(g'(\mu) = \frac{1}{\mu(1-\mu)}\) |

### Implementation

```rust
pub struct LogitLink;

impl Link for LogitLink {
    fn name(&self) -> &str { "Logit" }
    
    fn link(&self, mu: &Array1<f64>) -> Array1<f64> {
        mu.mapv(|m| {
            // Clamp to (ε, 1-ε) to avoid log(0) and log(-x)
            let m_safe = m.clamp(1e-10, 1.0 - 1e-10);
            (m_safe / (1.0 - m_safe)).ln()
        })
    }
    
    fn inverse(&self, eta: &Array1<f64>) -> Array1<f64> {
        eta.mapv(|e| {
            // Sigmoid function
            // Use stable formulation to avoid overflow
            if e >= 0.0 {
                1.0 / (1.0 + (-e).exp())
            } else {
                let exp_e = e.exp();
                exp_e / (1.0 + exp_e)
            }
        })
    }
    
    fn derivative(&self, mu: &Array1<f64>) -> Array1<f64> {
        mu.mapv(|m| {
            let m_safe = m.clamp(1e-10, 1.0 - 1e-10);
            1.0 / (m_safe * (1.0 - m_safe))
        })
    }
}
```

### Numerical Stability

The sigmoid function can be computed stably:

```rust
// Naive (can overflow):
let mu = 1.0 / (1.0 + (-eta).exp());  // exp(-eta) overflows if eta << 0

// Stable:
let mu = if eta >= 0.0 {
    1.0 / (1.0 + (-eta).exp())
} else {
    let exp_eta = eta.exp();
    exp_eta / (1.0 + exp_eta)
};
```

## Role in IRLS

Link functions appear in two key places in IRLS:

### 1. Working Weights

\[
W = \frac{1}{V(\mu) \cdot [g'(\mu)]^2}
\]

```rust
let link_deriv = link.derivative(&mu);
let weights = variance.iter().zip(link_deriv.iter())
    .map(|(v, d)| 1.0 / (v * d * d))
    .collect();
```

### 2. Working Response

\[
z = \eta + (y - \mu) \cdot g'(\mu)
\]

```rust
let z = Zip::from(&eta).and(y).and(&mu).and(&link_deriv)
    .map_collect(|&e, &yi, &mui, &d| {
        e + (yi - mui) * d
    });
```

## Testing Link Functions

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_log_link_roundtrip() {
        let link = LogLink;
        let mu = array![1.0, 2.0, 5.0];
        
        let eta = link.link(&mu);
        let mu_back = link.inverse(&eta);
        
        for i in 0..mu.len() {
            assert_relative_eq!(mu[i], mu_back[i], epsilon = 1e-10);
        }
    }
    
    #[test]
    fn test_logit_bounds() {
        let link = LogitLink;
        
        // Very negative eta → mu near 0
        let eta_neg = array![-100.0];
        let mu = link.inverse(&eta_neg);
        assert!(mu[0] < 1e-40);
        assert!(mu[0] > 0.0);
        
        // Very positive eta → mu near 1
        let eta_pos = array![100.0];
        let mu = link.inverse(&eta_pos);
        assert!(mu[0] > 1.0 - 1e-40);
        assert!(mu[0] < 1.0);
    }
    
    #[test]
    fn test_derivative_numerical() {
        let link = LogLink;
        let mu = array![1.0, 2.0, 5.0];
        let deriv = link.derivative(&mu);
        
        // Compare to numerical derivative
        let eps = 1e-7;
        for i in 0..mu.len() {
            let mu_plus = mu[i] + eps;
            let mu_minus = mu[i] - eps;
            let numerical = (mu_plus.ln() - mu_minus.ln()) / (2.0 * eps);
            assert_relative_eq!(deriv[i], numerical, epsilon = 1e-5);
        }
    }
}
```

## Adding a New Link Function

To add a new link (e.g., Probit):

### 1. Create the File

```rust
// links/probit.rs
use ndarray::Array1;
use statrs::distribution::{Normal, ContinuousCDF};

pub struct ProbitLink;

impl Link for ProbitLink {
    fn name(&self) -> &str { "Probit" }
    
    fn link(&self, mu: &Array1<f64>) -> Array1<f64> {
        let normal = Normal::new(0.0, 1.0).unwrap();
        mu.mapv(|m| {
            let m_safe = m.clamp(1e-10, 1.0 - 1e-10);
            normal.inverse_cdf(m_safe)
        })
    }
    
    fn inverse(&self, eta: &Array1<f64>) -> Array1<f64> {
        let normal = Normal::new(0.0, 1.0).unwrap();
        eta.mapv(|e| normal.cdf(e))
    }
    
    fn derivative(&self, mu: &Array1<f64>) -> Array1<f64> {
        let normal = Normal::new(0.0, 1.0).unwrap();
        mu.mapv(|m| {
            let m_safe = m.clamp(1e-10, 1.0 - 1e-10);
            let z = normal.inverse_cdf(m_safe);
            1.0 / normal.pdf(z)
        })
    }
}
```

### 2. Export from mod.rs

```rust
mod probit;
pub use probit::ProbitLink;
```

### 3. Add Python Binding

See [Adding a New Link](../maintenance/adding-link.md) for complete instructions.
