# Coefficient Constraints

Constrain coefficient signs to enforce monotonic relationships when business logic dictates.

## Quick Start

```python
import rustystats as rs

# Dict API - constrain coefficient sign
result = rs.glm_dict(
    response="ClaimNb",
    terms={
        "Age": {"type": "linear", "monotonicity": "increasing"},  # β ≥ 0
        "SafetyScore": {"type": "linear", "monotonicity": "decreasing"},  # β ≤ 0
        "Region": {"type": "categorical"},
    },
    data=data,
    family="poisson",
    offset="Exposure",
).fit()
```

---

## Types of Constraints

### Linear Term Constraints

Constrain the sign of a single coefficient:

```python
terms = {
    "Age": {"type": "linear", "monotonicity": "increasing"},  # β ≥ 0
    "Discount": {"type": "linear", "monotonicity": "decreasing"},  # β ≤ 0
}
```

| Constraint | Spec | Effect |
|------------|------|--------|
| Positive coefficient | `"monotonicity": "increasing"` | β ≥ 0 |
| Negative coefficient | `"monotonicity": "decreasing"` | β ≤ 0 |

### Expression Term Constraints

Constrain coefficients on computed expressions:

```python
terms = {
    "Age": {"type": "linear"},
    "Age2": {
        "type": "expression",
        "expr": "Age ** 2",
        "monotonicity": "decreasing",  # Force diminishing returns
    },
}
```

### Monotonic Spline Constraints

Constrain the entire spline curve to be monotonic:

```python
terms = {
    "Age": {"type": "bs", "monotonicity": "increasing"},  # Monotonically increasing curve
    "VehAge": {"type": "bs", "df": 5, "monotonicity": "decreasing"},  # Decreasing with fixed df
}
```

See [Splines](splines.md#monotonic-splines) for details on I-spline basis.

---

## Formula Syntax

For formula-based models, use `pos()` and `neg()`:

```python
# Positive coefficient (β ≥ 0)
result = rs.glm_dict(response="y", terms={"age": {"type": "linear", "monotonicity": "increasing"}, "x2": {"type": "linear"}}, data=data, family="poisson").fit()

# Negative coefficient (β ≤ 0)
result = rs.glm_dict(response="y", terms={"risk_score": {"type": "linear", "monotonicity": "decreasing"}, "x2": {"type": "linear"}}, data=data, family="poisson").fit()

# Multiple constraints
result = rs.glm_dict(response="y", terms={"age": {"type": "linear", "monotonicity": "increasing"}, "discount": {"type": "linear", "monotonicity": "decreasing"}, "region": {"type": "categorical"}}, data=data, family="poisson").fit()
```

---

## Implementation

Constraints are enforced via **projected gradient descent** during IRLS:

1. Compute unconstrained update: β_new = β + Δβ
2. Project onto constraint set: β_new = max(0, β_new) for increasing
3. Continue IRLS until convergence

For monotonic splines, the I-spline basis ensures monotonicity when all coefficients are non-negative.

---

## Use Cases

### Insurance Pricing

```python
result = rs.glm_dict(
    response="ClaimNb",
    terms={
        # Age should increase risk (more claims with age)
        "DrivAge": {"type": "bs", "monotonicity": "increasing"},
        
        # Bonus-malus should increase risk (higher BM = more risk)
        "BonusMalus": {"type": "linear", "monotonicity": "increasing"},
        
        # Vehicle age might decrease value, but increase frequency
        "VehAge": {"type": "bs", "monotonicity": "increasing"},
        
        # Safety features should decrease risk
        "SafetyRating": {"type": "linear", "monotonicity": "decreasing"},
        
        "Region": {"type": "categorical"},
    },
    data=data,
    family="poisson",
    offset="Exposure",
).fit()
```

### Credit Risk

```python
result = rs.glm_dict(
    response="Default",
    terms={
        # Higher credit score = lower default (decreasing)
        "CreditScore": {"type": "bs", "monotonicity": "decreasing"},
        
        # Higher debt ratio = higher default (increasing)
        "DebtRatio": {"type": "linear", "monotonicity": "increasing"},
        
        # Longer employment = lower default (decreasing)
        "YearsEmployed": {"type": "bs", "monotonicity": "decreasing"},
    },
    data=data,
    family="binomial",
).fit()
```

### Polynomial with Diminishing Returns

```python
result = rs.glm_dict(
    response="y",
    terms={
        "x": {"type": "linear"},  # Positive linear effect
        "x2": {
            "type": "expression",
            "expr": "x ** 2",
            "monotonicity": "decreasing",  # Negative quadratic = diminishing returns
        },
    },
    data=data,
    family="gaussian",
).fit()
```

---

## Checking Constraints

After fitting, verify constraints are satisfied:

```python
result = rs.glm_dict(
    response="y",
    terms={
        "age": {"type": "linear", "monotonicity": "increasing"},
        "risk": {"type": "linear", "monotonicity": "decreasing"},
    },
    data=data,
    family="poisson",
).fit()

# Check coefficients
for name, coef in zip(result.feature_names, result.params):
    print(f"{name}: {coef:.4f}")

# Verify: age coefficient should be ≥ 0, risk coefficient should be ≤ 0
```

---

## When to Use Constraints

### Good Use Cases

- **Business logic requires monotonicity** (e.g., "more experience = less risk")
- **Prevent implausible fitted effects** (e.g., risk decreasing with age)
- **Regulatory requirements** for model interpretability
- **Stabilize spline curves** that would otherwise wiggle

### Avoid When

- **Relationship is genuinely non-monotonic** (U-shaped effects)
- **Exploratory analysis** where you want to discover patterns
- **Constraint would conflict with data** (causes poor fit)

---

## Troubleshooting

### Model won't converge with constraints

The constrained optimization may need more iterations:

```python
result = model.fit(max_iter=100)  # Increase from default
```

### Coefficient stuck at boundary

If a coefficient is exactly 0 (at boundary), the constraint is binding. This may indicate:
- The unconstrained coefficient has opposite sign
- The variable has weak signal in the constrained direction

```python
# Check if constraint is binding
for name, coef in zip(result.feature_names, result.params):
    if abs(coef) < 1e-6:
        print(f"{name}: coefficient at boundary (constraint binding)")
```
