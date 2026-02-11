# Relative Factor Importance — Calculation Spec

## Problem

After fitting a GLM with multiple factors, actuaries need to see at a glance which factors
contribute most to the model. The existing `significance.dev_contrib` (Type III deviance
contribution) is useful for hypothesis testing but its absolute value is hard to interpret
— e.g., "Δdev = 199.8" doesn't tell you whether that's big or small.

## Solution: Normalised Relative Importance

Normalise each factor's Type III deviance contribution so that the values across all
**fitted** factors sum to 100%.

### Formula

For a model with fitted factors $f_1, f_2, \ldots, f_k$, each with Type III deviance
contribution $\Delta D_i$ (i.e., the increase in deviance when factor $f_i$ is dropped
from the full model while keeping all other factors):

$$
\text{importance}_i = \frac{\Delta D_i}{\sum_{j=1}^{k} \Delta D_j} \times 100
$$

### Example (freMTPL2freq, Poisson, 2 factors)

| Factor     | Type | dev_contrib (Δdev) | Relative Importance |
|------------|------|--------------------|---------------------|
| BonusMalus | ns   | 199.80             | 83.9%               |
| VehAge     | bs   | 38.28              | 16.1%               |
| **Total**  |      | **238.08**         | **100.0%**          |

### Why Type III (not Type I)

- **Type I** (sequential) deviance depends on the order factors are added.
  Different orderings give different attributions.
- **Type III** (marginal) measures each factor's unique contribution given all
  other factors are in the model. Order-independent, which is what we want.

### Where to add it in the diagnostics JSON

Inside each entry in the `factors[]` array, alongside the existing `significance` object:

```json
{
  "name": "BonusMalus",
  "in_model": true,
  "significance": {
    "chi2": 199.80,
    "p": 0.0,
    "dev_contrib": 199.80
  },
  "relative_importance": 83.9
}
```

And/or as a top-level summary:

```json
{
  "factor_importance": [
    { "factor": "BonusMalus", "dev_contrib": 199.80, "importance_pct": 83.9 },
    { "factor": "VehAge",     "dev_contrib":  38.28, "importance_pct": 16.1 }
  ]
}
```

### Implementation (Python pseudocode)

```python
def compute_relative_importance(factors: list[dict]) -> list[dict]:
    """
    Given a list of per-factor diagnostics (each with significance.dev_contrib),
    compute normalised relative importance for fitted factors.
    """
    fitted = [f for f in factors if f.get("significance")]
    total = sum(f["significance"]["dev_contrib"] for f in fitted)

    result = []
    for f in fitted:
        pct = (f["significance"]["dev_contrib"] / total * 100) if total > 0 else 0
        result.append({
            "factor": f["name"],
            "dev_contrib": f["significance"]["dev_contrib"],
            "importance_pct": round(pct, 2),
        })

    # Sort descending by importance
    result.sort(key=lambda x: -x["importance_pct"])
    return result
```

### Notes

- Only computed for **fitted** (in-model) factors. Unfitted factors use score test rank instead.
- If a model has a single factor, that factor gets 100%.
- Correlated factors will share importance (their marginal contributions overlap), so
  individual percentages are conservative. This is acceptable and expected for Type III.
- The raw `dev_contrib` and `chi2` remain available for formal hypothesis testing;
  `relative_importance` is purely for visual/UX purposes.
