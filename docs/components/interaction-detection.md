# Interaction Detection

This document explains how RustyStats detects potential interactions between factors, helping you decide which interaction terms to include in your GLM.

---

## High-Level Explanation (Non-Technical)

### What is an Interaction?

An **interaction** occurs when the effect of one factor on the outcome depends on the value of another factor.

**Example without interaction:**
- Older drivers have fewer claims
- Sports cars have more claims
- These effects are independent—the age effect is the same whether driving a sedan or sports car

**Example with interaction:**
- Young drivers in sports cars have *disproportionately* more claims than you'd expect from adding the "young driver" effect and "sports car" effect together
- The combination is worse than the sum of its parts

### Why Detect Interactions?

Including important interactions in your model:

1. **Improves accuracy** — Captures real patterns the main effects miss
2. **Better pricing** — Avoids under/over-charging specific segments
3. **Regulatory compliance** — Demonstrates you've properly modeled risk

Missing important interactions can lead to:

- Systematic under-pricing for high-risk segments
- Cross-subsidization between customer groups
- Poor model performance on specific populations

### How RustyStats Detects Interactions

RustyStats uses two approaches depending on whether you have a fitted model:

#### Pre-Fit Detection (Data Exploration)

Before fitting a model, we look at how the **response rate varies** across combinations of factors:

1. **Divide each factor into groups** (bins for continuous, levels for categorical)
2. **Create cells** from all combinations (e.g., Age bin × Region)
3. **Calculate average response** in each cell
4. **Compare to what we'd expect** if effects were independent
5. **Flag pairs** where the combined effect differs significantly

This answers: *"Which factor pairs show non-additive effects on the response?"*

#### Post-Fit Detection (Model Diagnostics)

After fitting a model, we look at **residual patterns**:

1. **Compute residuals** (actual - predicted)
2. **Group by factor combinations**
3. **Check if residuals are systematically high/low** for certain combinations
4. **Flag pairs** where the model misses a pattern

This answers: *"Which factor pairs does my current model fail to capture?"*

### Interpreting the Results

The output includes:

| Field | Meaning |
|-------|---------|
| `factor1`, `factor2` | The two factors with potential interaction |
| `interaction_strength` | How much variance is explained (0-1 scale) |
| `pvalue` | Statistical significance (lower = more confident) |
| `n_cells` | Number of valid combinations tested |

**Rules of thumb:**

- **strength > 0.01**: Worth investigating
- **strength > 0.05**: Likely important
- **pvalue < 0.01**: Statistically significant

### What to Do With the Results

1. **Review top candidates** — Look at the factor pairs with highest strength
2. **Visualize the pattern** — Plot response rates by combination
3. **Business sense check** — Does the interaction make intuitive sense?
4. **Test in model** — Add the interaction term and compare fit metrics

---

## Technical Explanation

### Mathematical Framework

#### Interaction Effect Definition

For two factors X₁ and X₂, an interaction exists when:

```
E[Y | X₁, X₂] ≠ g⁻¹(β₀ + β₁X₁ + β₂X₂)
```

where g is the link function. The model requires an additional term:

```
E[Y | X₁, X₂] = g⁻¹(β₀ + β₁X₁ + β₂X₂ + β₃X₁X₂)
```

### Pre-Fit Detection Algorithm

#### Step 1: Factor Ranking by Marginal Effect

First, we rank factors by their univariate association with the response using eta-squared (η²):

```python
def compute_eta_squared(y_rate, exposure, factor_bins):
    """Compute variance explained by factor grouping."""
    overall_mean = weighted_average(y_rate, exposure)
    
    ss_total = sum(exposure * (y_rate - overall_mean)²)
    
    ss_between = 0
    for level in unique(factor_bins):
        mask = factor_bins == level
        level_mean = weighted_average(y_rate[mask], exposure[mask])
        ss_between += sum(exposure[mask]) * (level_mean - overall_mean)²
    
    return ss_between / ss_total
```

Interpretation:
- η² = 0: Factor has no marginal effect
- η² = 0.01: 1% of response variance explained
- η² = 0.10: Strong effect

#### Step 2: Pairwise Interaction Testing

For top-ranked factors, we test all pairs:

```python
def compute_interaction_strength(y_rate, exposure, bins1, bins2, min_cell_count):
    """Compute R² from interaction cell grouping."""
    
    # Create interaction cells
    cell_ids = bins1 * 1000 + bins2
    
    # Filter cells with sufficient data
    valid_cells = [c for c in unique(cell_ids) 
                   if count(cell_ids == c) >= min_cell_count]
    
    if len(valid_cells) < 4:
        return None  # Insufficient data
    
    # Compute variance explained by cell grouping
    overall_mean = weighted_average(y_rate, exposure)
    ss_total = sum(exposure * (y_rate - overall_mean)²)
    
    ss_model = 0
    for cell in valid_cells:
        mask = cell_ids == cell
        cell_mean = weighted_average(y_rate[mask], exposure[mask])
        ss_model += sum(exposure[mask]) * (cell_mean - overall_mean)²
    
    r_squared = ss_model / ss_total
    return r_squared
```

#### Step 3: Statistical Significance

We use an F-test to assess significance:

```
F = (SS_model / df_model) / (SS_residual / df_residual)

where:
  df_model = n_cells - 1
  df_residual = n_observations - n_cells
```

The p-value comes from the F-distribution with (df_model, df_residual) degrees of freedom.

### Post-Fit Detection Algorithm

After fitting a model, we detect interactions the model is missing:

#### Step 1: Compute Residuals

```python
pearson_residuals = (y - mu) / sqrt(variance(mu))
```

#### Step 2: Residual Association

For each factor, compute correlation with residuals:

**Continuous factors:**
```python
correlation = weighted_corr(factor_values, pearson_residuals, weights=exposure)
```

**Categorical factors:**
```python
eta_squared = variance_between_levels / total_variance
```

#### Step 3: Interaction Residual Patterns

For factor pairs, check if residual means vary across combinations:

```python
def interaction_residual_strength(residuals, bins1, bins2):
    """How much residual variance is explained by the interaction?"""
    
    cell_ids = bins1 * 1000 + bins2
    overall_mean = mean(residuals)
    
    ss_total = sum((residuals - overall_mean)²)
    
    ss_between = 0
    for cell in unique(cell_ids):
        mask = cell_ids == cell
        cell_mean = mean(residuals[mask])
        ss_between += count(mask) * (cell_mean - overall_mean)²
    
    return ss_between / ss_total
```

### Discretization Strategy

Continuous factors are discretized into 5 quantile bins:

```python
def discretize(values, n_bins=5):
    quantiles = percentile(values, linspace(0, 100, n_bins + 1))
    return digitize(values, quantiles[1:-1])
```

Using 5 bins provides:
- Sufficient granularity to detect non-linear interactions
- Enough observations per cell for stable estimates
- Manageable number of combinations (5 × 5 = 25 cells maximum)

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_factors` | 10 | Maximum factors to consider (top by marginal effect) |
| `min_effect_size` | 0.001 | Minimum η² to include factor in pairwise testing |
| `max_candidates` | 5 | Maximum interaction candidates to return |
| `min_cell_count` | 30 | Minimum observations per cell for valid estimate |

### Computational Complexity

- **Factor ranking**: O(n × p) where n = observations, p = factors
- **Pairwise testing**: O(n × p²) worst case
- **Total**: O(n × p²)

For 678,000 observations and 10 factors, this takes < 1 second.

### Limitations

1. **Only pairwise**: Does not detect 3-way or higher interactions
2. **Discretization**: May miss interactions that depend on exact continuous values
3. **Correlation ≠ Causation**: Statistical significance doesn't imply the interaction should be modeled
4. **Multiple testing**: With many factor pairs, some will appear significant by chance

### Best Practices

1. **Use domain knowledge** — Prioritize interactions that make business sense
2. **Validate on holdout** — Test if interaction improves out-of-sample performance
3. **Check stability** — Ensure interaction effect is consistent across time periods
4. **Consider parsimony** — Only add interactions that meaningfully improve the model

---

## Code Examples

### Pre-Fit Exploration

```python
import rustystats as rs

# Explore data for potential interactions
exploration = rs.explore_data(
    data=data,
    response="ClaimNb",
    categorical_factors=["Region", "VehBrand", "Area"],
    continuous_factors=["DrivAge", "VehAge", "VehPower"],
    exposure="Exposure",
    family="poisson",
)

# Check interaction candidates
for ic in exploration.interaction_candidates:
    print(f"{ic.factor1} × {ic.factor2}: strength={ic.interaction_strength:.4f}, p={ic.pvalue:.4f}")
```

### Post-Fit Diagnostics

```python
# Fit base model
result = rs.glm_dict(
    response="ClaimNb",
    terms={"DrivAge": {"type": "linear"}, "VehAge": {"type": "linear"}, "Region": {"type": "categorical"}},
    data=data,
    family="poisson",
    offset="Exposure"
).fit()

# Check for missing interactions
diagnostics = result.diagnostics(
    data=data,
    categorical_factors=["Region", "VehBrand"],
    continuous_factors=["DrivAge", "VehAge"],
)

# If interactions detected, add to model
for ic in diagnostics.interaction_candidates:
    if ic.interaction_strength > 0.01:
        print(f"Consider adding: {ic.factor1}:{ic.factor2}")
```

### Adding an Interaction

```python
# Add interaction term
result_with_interaction = rs.glm_dict(
    response="ClaimNb",
    terms={"DrivAge": {"type": "linear"}, "VehAge": {"type": "linear"}, "Region": {"type": "categorical"}},
    interactions=[{"DrivAge": {"type": "linear"}, "VehAge": {"type": "linear"}}],
    data=data,
    family="poisson",
    offset="Exposure",
).fit()

# Compare AIC
print(f"Base AIC: {result.aic():.1f}")
print(f"With interaction AIC: {result_with_interaction.aic():.1f}")
```
