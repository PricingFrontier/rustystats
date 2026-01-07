# Diagnostics Component

The diagnostics module provides comprehensive model assessment tools including residuals, calibration metrics, discrimination measures, and interaction detection.

## Code Location

```
crates/rustystats-core/src/diagnostics/
├── mod.rs              # Re-exports
├── residuals.rs        # Residual computations
├── dispersion.rs       # Dispersion estimation
├── likelihood.rs       # Log-likelihood, AIC, BIC
├── calibration.rs      # A/E ratios, calibration curves
├── discrimination.rs   # Gini, AUC, lift
├── loss.rs             # Loss functions (MSE, MAE)
├── interactions.rs     # Interaction detection
└── factor_diagnostics.rs # Per-factor diagnostics, VIF, partial dependence

python/rustystats/diagnostics.py  # Python API
```

## Performance

Diagnostics computation is optimized with Rust backends achieving **6x speedup** over pure Python:

| Component | Method | Complexity |
|-----------|--------|------------|
| VIF | Correlation matrix inverse | O(k³) vs O(k×n×k²) |
| Factor deviance | Rust HashMap groupby | O(n) |
| Categorical PD | Rust HashMap groupby | O(n) |
| A/E by level | Rust aggregation | O(n) |
| Residual patterns | Pandas vectorized groupby | O(n) |

**Typical timings** (500K rows, 8 factors):
- Full diagnostics: ~4.5s
- With train/test comparison: ~5.5s

## Residuals

### Types of Residuals

```rust
/// Response residuals: y - μ
pub fn resid_response(
    y: &Array1<f64>,
    mu: &Array1<f64>,
) -> Array1<f64> {
    y - mu
}

/// Pearson residuals: (y - μ) / √V(μ)
pub fn resid_pearson(
    y: &Array1<f64>,
    mu: &Array1<f64>,
    family: &dyn Family,
) -> Array1<f64> {
    let var = family.variance(mu);
    (y - mu) / var.mapv(|v| v.sqrt())
}

/// Deviance residuals: sign(y - μ) × √d
pub fn resid_deviance(
    y: &Array1<f64>,
    mu: &Array1<f64>,
    family: &dyn Family,
) -> Array1<f64> {
    let unit_dev = family.unit_deviance(y, mu);
    Zip::from(y).and(mu).and(&unit_dev)
        .map_collect(|&yi, &mui, &di| {
            let sign = if yi > mui { 1.0 } else { -1.0 };
            sign * di.sqrt()
        })
}

/// Working residuals: (y - μ) × g'(μ)
pub fn resid_working(
    y: &Array1<f64>,
    mu: &Array1<f64>,
    link: &dyn Link,
) -> Array1<f64> {
    let deriv = link.derivative(mu);
    (y - mu) * deriv
}
```

### When to Use Each

| Residual | Use Case |
|----------|----------|
| Response | Simple interpretation |
| Pearson | Detecting outliers, checking overdispersion |
| Deviance | Most diagnostic plots, Q-Q plots |
| Working | Partial residual plots |

## Dispersion Estimation

```rust
/// Pearson-based dispersion estimate
pub fn estimate_dispersion_pearson(
    y: &Array1<f64>,
    mu: &Array1<f64>,
    family: &dyn Family,
    df_resid: usize,
    weights: Option<&Array1<f64>>,
) -> f64 {
    let chi2 = pearson_chi2(y, mu, family, weights);
    chi2 / df_resid as f64
}

/// Pearson chi-squared statistic
pub fn pearson_chi2(
    y: &Array1<f64>,
    mu: &Array1<f64>,
    family: &dyn Family,
    weights: Option<&Array1<f64>>,
) -> f64 {
    let var = family.variance(mu);
    
    let mut chi2 = 0.0;
    for i in 0..y.len() {
        let r = (y[i] - mu[i]).powi(2) / var[i];
        let w = weights.map_or(1.0, |w| w[i]);
        chi2 += w * r;
    }
    
    chi2
}
```

## Likelihood and Information Criteria

```rust
/// Log-likelihood for Poisson
pub fn log_likelihood_poisson(
    y: &Array1<f64>,
    mu: &Array1<f64>,
    weights: Option<&Array1<f64>>,
) -> f64 {
    let mut ll = 0.0;
    for i in 0..y.len() {
        let w = weights.map_or(1.0, |w| w[i]);
        // log P(Y=y) = y*log(μ) - μ - log(y!)
        let log_factorial = (1..=(y[i] as usize))
            .map(|k| (k as f64).ln())
            .sum::<f64>();
        ll += w * (y[i] * mu[i].ln() - mu[i] - log_factorial);
    }
    ll
}

/// AIC = -2*loglik + 2*p
pub fn aic(log_likelihood: f64, n_params: usize) -> f64 {
    -2.0 * log_likelihood + 2.0 * n_params as f64
}

/// BIC = -2*loglik + p*log(n)
pub fn bic(log_likelihood: f64, n_params: usize, n_obs: usize) -> f64 {
    -2.0 * log_likelihood + (n_params as f64) * (n_obs as f64).ln()
}

/// Null deviance (intercept-only model)
pub fn null_deviance(
    y: &Array1<f64>,
    family: &dyn Family,
    weights: Option<&Array1<f64>>,
) -> f64 {
    // Fit intercept-only model: μ = weighted mean of y
    let mu_null = match weights {
        Some(w) => {
            let sum_wy: f64 = y.iter().zip(w.iter()).map(|(y, w)| y * w).sum();
            let sum_w: f64 = w.sum();
            sum_wy / sum_w
        }
        None => y.mean().unwrap(),
    };
    
    let mu_null_vec = Array1::from_elem(y.len(), mu_null);
    family.deviance(y, &mu_null_vec, weights)
}
```

## Calibration Metrics

### Actual vs Expected

```rust
pub struct CalibrationResult {
    pub overall_ae: f64,
    pub by_decile: Vec<DecileStats>,
    pub hosmer_lemeshow: Option<HLTest>,
}

pub fn compute_calibration_curve(
    y: &Array1<f64>,
    mu: &Array1<f64>,
    weights: Option<&Array1<f64>>,
    n_bins: usize,
) -> CalibrationResult {
    let n = y.len();
    
    // Overall A/E
    let total_actual: f64 = match weights {
        Some(w) => y.iter().zip(w.iter()).map(|(y, w)| y * w).sum(),
        None => y.sum(),
    };
    let total_expected: f64 = match weights {
        Some(w) => mu.iter().zip(w.iter()).map(|(m, w)| m * w).sum(),
        None => mu.sum(),
    };
    let overall_ae = total_actual / total_expected;
    
    // Sort by predicted risk
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| mu[a].partial_cmp(&mu[b]).unwrap());
    
    // Compute A/E by decile
    let bin_size = n / n_bins;
    let mut by_decile = Vec::with_capacity(n_bins);
    
    for bin in 0..n_bins {
        let start = bin * bin_size;
        let end = if bin == n_bins - 1 { n } else { (bin + 1) * bin_size };
        
        let mut actual = 0.0;
        let mut expected = 0.0;
        let mut exposure = 0.0;
        
        for &i in &indices[start..end] {
            let w = weights.map_or(1.0, |w| w[i]);
            actual += y[i] * w;
            expected += mu[i] * w;
            exposure += w;
        }
        
        by_decile.push(DecileStats {
            decile: bin + 1,
            actual,
            expected,
            ae_ratio: actual / expected,
            exposure,
        });
    }
    
    CalibrationResult { overall_ae, by_decile, hosmer_lemeshow: None }
}
```

## Discrimination Metrics

### Gini Coefficient

```rust
pub fn compute_gini(
    y: &Array1<f64>,
    mu: &Array1<f64>,
    weights: Option<&Array1<f64>>,
) -> f64 {
    // Gini = 2 * AUC - 1
    let auc = compute_auc(y, mu, weights);
    2.0 * auc - 1.0
}

pub fn compute_auc(
    y: &Array1<f64>,
    mu: &Array1<f64>,
    weights: Option<&Array1<f64>>,
) -> f64 {
    let n = y.len();
    
    // Sort by predicted risk (descending)
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| mu[b].partial_cmp(&mu[a]).unwrap());
    
    // Compute AUC using trapezoidal rule
    let mut cum_actual = 0.0;
    let mut cum_exposure = 0.0;
    let total_actual: f64 = y.sum();
    let total_exposure: f64 = weights.map_or(n as f64, |w| w.sum());
    
    let mut auc = 0.0;
    let mut prev_x = 0.0;
    let mut prev_y = 0.0;
    
    for &i in &indices {
        let w = weights.map_or(1.0, |w| w[i]);
        cum_actual += y[i];
        cum_exposure += w;
        
        let x = cum_exposure / total_exposure;
        let y_val = cum_actual / total_actual;
        
        // Trapezoidal area
        auc += (x - prev_x) * (y_val + prev_y) / 2.0;
        
        prev_x = x;
        prev_y = y_val;
    }
    
    auc
}
```

### Lorenz Curve

```rust
pub fn compute_lorenz_curve(
    y: &Array1<f64>,
    mu: &Array1<f64>,
    weights: Option<&Array1<f64>>,
    n_points: usize,
) -> Vec<(f64, f64)> {
    let n = y.len();
    
    // Sort by predicted risk
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| mu[a].partial_cmp(&mu[b]).unwrap());
    
    let total_actual: f64 = y.sum();
    let total_exposure: f64 = weights.map_or(n as f64, |w| w.sum());
    
    let mut curve = Vec::with_capacity(n_points + 1);
    curve.push((0.0, 0.0));
    
    let step = n / n_points;
    let mut cum_actual = 0.0;
    let mut cum_exposure = 0.0;
    
    for (idx, &i) in indices.iter().enumerate() {
        let w = weights.map_or(1.0, |w| w[i]);
        cum_actual += y[i];
        cum_exposure += w;
        
        if (idx + 1) % step == 0 || idx == n - 1 {
            curve.push((
                cum_exposure / total_exposure,
                cum_actual / total_actual,
            ));
        }
    }
    
    curve
}
```

## Interaction Detection

Greedy residual-based detection of potential interactions:

```rust
pub struct InteractionConfig {
    pub max_candidates: usize,
    pub min_strength: f64,
}

pub fn detect_interactions(
    residuals: &Array1<f64>,
    factors: &[FactorData],
    config: &InteractionConfig,
) -> Vec<InteractionCandidate> {
    let n_factors = factors.len();
    let mut candidates = Vec::new();
    
    // Test all pairs
    for i in 0..n_factors {
        for j in (i + 1)..n_factors {
            let strength = compute_interaction_strength(
                residuals,
                &factors[i],
                &factors[j],
            );
            
            if strength > config.min_strength {
                candidates.push(InteractionCandidate {
                    factor1: factors[i].name.clone(),
                    factor2: factors[j].name.clone(),
                    strength,
                });
            }
        }
    }
    
    // Sort by strength
    candidates.sort_by(|a, b| b.strength.partial_cmp(&a.strength).unwrap());
    candidates.truncate(config.max_candidates);
    
    candidates
}

fn compute_interaction_strength(
    residuals: &Array1<f64>,
    factor1: &FactorData,
    factor2: &FactorData,
) -> f64 {
    // Compute variance of residuals explained by interaction
    // Uses grouping by factor combinations
    
    // ... implementation details
    
    explained_variance / total_variance
}
```

## Python API: ModelDiagnostics

```python
class ModelDiagnostics:
    """Comprehensive model diagnostics with JSON export."""
    
    def __init__(self, result, data, categorical_factors, continuous_factors):
        self.model_summary = self._compute_model_summary(result)
        self.fit_statistics = self._compute_fit_stats(result)
        self.calibration = self._compute_calibration(result, data)
        self.discrimination = self._compute_discrimination(result, data)
        self.factors = self._compute_factor_diagnostics(
            result, data, categorical_factors, continuous_factors
        )
        self.interaction_candidates = self._detect_interactions(result, data)
        self.warnings = self._generate_warnings()
    
    def to_json(self) -> str:
        """Export as compact JSON for LLM consumption."""
        return json.dumps({
            'model_summary': self.model_summary,
            'fit_statistics': self.fit_statistics,
            'calibration': self.calibration,
            'discrimination': self.discrimination,
            'factors': [f.to_dict() for f in self.factors],
            'interaction_candidates': self.interaction_candidates,
            'warnings': self.warnings,
        }, indent=2)
```

### Usage

```python
result = rs.glm("y ~ x1 + C(region)", data, family="poisson").fit()

diagnostics = result.diagnostics(
    data=data,
    categorical_factors=["region", "brand"],
    continuous_factors=["age", "income"],
)

# View summary
print(diagnostics.to_json())

# Check specific metrics
print(f"Gini: {diagnostics.discrimination['gini_coefficient']:.3f}")
print(f"A/E: {diagnostics.calibration['overall_ae']:.3f}")

# Check for issues
for warning in diagnostics.warnings:
    print(f"⚠️ {warning['message']}")
```

## VIF / Multicollinearity

Fast VIF computation using correlation matrix inverse (O(k³) vs O(k×n×k²)):

```rust
/// Compute VIF via correlation matrix inverse
/// VIF_j = diag(R^{-1})_j where R is correlation matrix
pub fn compute_vif(X: &Array2<f64>) -> Vec<f64> {
    // Standardize columns
    let X_std = standardize_columns(X);
    
    // Correlation matrix R = X'X / n
    let R = X_std.t().dot(&X_std) / n as f64;
    
    // Add small regularization for numerical stability
    R += Array2::eye(k) * 1e-10;
    
    // VIF = diagonal of R^{-1}
    let R_inv = R.inv().unwrap();
    R_inv.diag().to_vec()
}
```

**Severity Classification:**
- `none`: VIF < 5
- `moderate`: 5 ≤ VIF < 10
- `severe`: VIF ≥ 10

## Factor Deviance

Rust implementation for fast deviance breakdown by categorical level:

```rust
/// Compute deviance breakdown by categorical factor level
pub fn compute_factor_deviance(
    factor_name: &str,
    factor_values: &[String],
    y: &Array1<f64>,
    mu: &Array1<f64>,
    family: &str,
) -> FactorDevianceResult {
    // Single pass groupby using HashMap
    let mut level_data: HashMap<&str, (usize, f64, f64, f64)> = HashMap::new();
    
    for (i, level) in factor_values.iter().enumerate() {
        let entry = level_data.entry(level.as_str()).or_insert((0, 0.0, 0.0, 0.0));
        entry.0 += 1;                          // count
        entry.1 += unit_deviance(y[i], mu[i]); // deviance
        entry.2 += y[i];                       // actual
        entry.3 += mu[i];                      // predicted
    }
    
    // Build results sorted by deviance
    // ...
}
```

## Partial Dependence

Fast categorical partial dependence via Rust HashMap:

```rust
/// Compute mean prediction for each categorical level
pub fn compute_categorical_pd(
    factor_values: &[String],
    predictions: &[f64],
) -> CategoricalPDResult {
    let mut level_sums: HashMap<&str, (f64, usize)> = HashMap::new();
    
    for (i, level) in factor_values.iter().enumerate() {
        let entry = level_sums.entry(level.as_str()).or_insert((0.0, 0));
        entry.0 += predictions[i];
        entry.1 += 1;
    }
    
    // Return means, counts, relativities
}
```

## Train/Test Comparison

Comprehensive train vs test diagnostics with overfitting detection:

```python
@dataclass
class TrainTestComparison:
    train: DatasetDiagnostics    # Per-set metrics
    test: DatasetDiagnostics
    gini_gap: float              # train.gini - test.gini
    overfitting_risk: bool       # gini_gap > 0.03
    calibration_drift: bool      # test A/E outside [0.95, 1.05]
    unstable_factors: List[str]  # Factors with A/E diff > 0.1
```

**Overfitting Flags:**
- `overfitting_risk`: Train Gini significantly better than Test Gini (gap > 0.03)
- `calibration_drift`: Test A/E ratio outside acceptable range [0.95, 1.05]
- `unstable_factors`: Factors where train/test A/E differ by more than 0.1

## Warning Generation

```rust
pub fn generate_warnings(diagnostics: &Diagnostics) -> Vec<Warning> {
    let mut warnings = Vec::new();
    
    // Check dispersion
    if diagnostics.dispersion_pearson > 2.0 {
        warnings.push(Warning {
            warning_type: "overdispersion".into(),
            message: format!(
                "High dispersion ({:.2}). Consider QuasiPoisson or NegBinomial.",
                diagnostics.dispersion_pearson
            ),
        });
    }
    
    // Check calibration
    if (diagnostics.overall_ae - 1.0).abs() > 0.05 {
        warnings.push(Warning {
            warning_type: "calibration".into(),
            message: format!(
                "A/E ratio is {:.3}. Model {} overall.",
                diagnostics.overall_ae,
                if diagnostics.overall_ae > 1.0 { "underpredicts" } else { "overpredicts" }
            ),
        });
    }
    
    // Check multicollinearity
    for vif in &diagnostics.vif {
        if vif.severity == "severe" {
            warnings.push(Warning {
                warning_type: "multicollinearity".into(),
                message: format!(
                    "Feature '{}' has VIF={:.1f} (severe). Consider removing.",
                    vif.feature, vif.vif
                ),
            });
        }
    }
    
    // Check overfitting
    if let Some(tt) = &diagnostics.train_test {
        if tt.overfitting_risk {
            warnings.push(Warning {
                warning_type: "overfitting".into(),
                message: format!(
                    "Overfitting detected: Gini gap={:.3f}. Consider regularization.",
                    tt.gini_gap
                ),
            });
        }
    }
    
    // Check for non-fitted factors with signal
    for factor in &diagnostics.factors {
        if !factor.in_model && factor.residual_correlation.abs() > 0.05 {
            warnings.push(Warning {
                warning_type: "missing_factor".into(),
                message: format!(
                    "Factor '{}' not in model but explains {:.1}% of residual variance.",
                    factor.name,
                    factor.residual_correlation.abs() * 100.0
                ),
            });
        }
    }
    
    warnings
}
```

## Testing

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_gini_perfect_model() {
        // Perfect predictions should give Gini = 1
        let y = array![0.0, 0.0, 1.0, 1.0];
        let mu = array![0.1, 0.2, 0.8, 0.9];  // Perfect ranking
        
        let gini = compute_gini(&y, &mu, None);
        assert!((gini - 1.0).abs() < 0.01);
    }
    
    #[test]
    fn test_gini_random_model() {
        // Random predictions should give Gini ≈ 0
        let y = array![0.0, 1.0, 0.0, 1.0];
        let mu = array![0.5, 0.5, 0.5, 0.5];  // No discrimination
        
        let gini = compute_gini(&y, &mu, None);
        assert!(gini.abs() < 0.01);
    }
    
    #[test]
    fn test_ae_ratio() {
        let y = array![1.0, 2.0, 3.0];
        let mu = array![1.0, 2.0, 3.0];  // Perfect predictions
        
        let cal = compute_calibration_curve(&y, &mu, None, 3);
        assert!((cal.overall_ae - 1.0).abs() < 1e-10);
    }
}
```
