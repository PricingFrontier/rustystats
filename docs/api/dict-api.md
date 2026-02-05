# Dict API Reference

The Dict API (`glm_dict`) is RustyStats' primary interface, designed for programmatic model building ideal for automated workflows and agents.

## glm_dict

Create a GLM specification with dict-based term definitions.

```python
rustystats.glm_dict(
    response,
    terms,
    data,
    family="gaussian",
    link=None,
    offset=None,
    weights=None,
    interactions=None,
    theta=None,
    var_power=1.5,
    seed=None,
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `response` | str | Column name for response variable |
| `terms` | dict | Term specifications (see below) |
| `data` | DataFrame | Polars or Pandas DataFrame |
| `family` | str | Distribution family |
| `link` | str | Link function (optional) |
| `offset` | str | Column name for offset |
| `weights` | str | Column name for weights |
| `interactions` | list | Interaction specifications (see below) |
| `theta` | float | Negative Binomial dispersion |
| `var_power` | float | Tweedie variance power |
| `seed` | int | Random seed for reproducibility |

### Returns

`FormulaGLMDict` object - call `.fit()` to fit the model.

---

## Term Types

Each term in the `terms` dict maps a variable name to a specification dict.

### linear

Raw continuous variable.

```python
terms = {
    "Age": {"type": "linear"},
    "VehPower": {"type": "linear", "monotonicity": "increasing"},  # β ≥ 0
}
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `monotonicity` | str | `"increasing"` (β ≥ 0) or `"decreasing"` (β ≤ 0) |

### categorical

Dummy encoding for categorical variables.

```python
terms = {
    "Region": {"type": "categorical"},
    "Area": {"type": "categorical", "levels": ["A", "B", "C"]},  # Explicit levels
    "IsParis": {"type": "categorical", "level": "Paris"},  # Single level indicator
}
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `levels` | list | Explicit level ordering (optional) |
| `level` | str | Single level to create 0/1 indicator for |

#### Single-Level Indicators

Create a binary indicator for a specific category level:

```python
terms = {
    # 0/1 indicator: 1 if Region == "Paris", else 0
    "IsParis": {"type": "categorical", "level": "Paris", "source": "Region"},
}
```

Useful for:
- Testing specific level effects
- Creating custom groupings
- Simplifying high-cardinality factors to key levels

### bs (B-spline)

B-spline basis for non-linear effects.

```python
terms = {
    "Age": {"type": "bs"},                              # Penalized smooth (default k=10)
    "VehAge": {"type": "bs", "df": 5},                  # Fixed 5 df
    "Income": {"type": "bs", "k": 15},                  # Penalized with 15 basis functions
    "Risk": {"type": "bs", "monotonicity": "increasing"},  # Monotonic
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | int | - | Fixed degrees of freedom (no penalty) |
| `k` | int | 10 | Basis size for penalized smooth |
| `degree` | int | 3 | Polynomial degree |
| `monotonicity` | str | - | `"increasing"` or `"decreasing"` |

**Behavior:**
- No `df` or `k` → penalized smooth with k=10, auto-tuned via GCV
- `df=5` → fixed 5 degrees of freedom, no penalty
- `k=15` → penalized smooth with 15 basis functions
- `monotonicity` → I-spline basis with coefficient constraints

### ns (Natural spline)

Natural cubic spline with linear extrapolation beyond boundaries.

```python
terms = {
    "Age": {"type": "ns"},           # Penalized smooth (default k=10)
    "Income": {"type": "ns", "df": 4},  # Fixed 4 df
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | int | - | Fixed degrees of freedom |
| `k` | int | 10 | Basis size for penalized smooth |

### target_encoding

Regularized target encoding for high-cardinality categoricals.

```python
terms = {
    "Brand": {"type": "target_encoding"},
    "Model": {"type": "target_encoding", "prior_weight": 2.0},
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prior_weight` | float | 1.0 | Regularization toward global mean |

### expression

Arbitrary arithmetic expressions (like R's `I()`).

```python
terms = {
    "Age2": {"type": "expression", "expr": "Age ** 2"},
    "BMI": {"type": "expression", "expr": "Weight / (Height ** 2)"},
    "LogDensity": {"type": "expression", "expr": "log(Density)"},
}
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `expr` | str | Python expression using column names |
| `monotonicity` | str | `"increasing"` or `"decreasing"` (optional) |

**Supported operations:** `+`, `-`, `*`, `/`, `**`, `log`, `exp`, `sqrt`

---

## Interactions

Interactions are specified as a list of dicts. Each interaction dict contains variable specifications plus control flags.

### Standard Interactions

Product terms between variables.

```python
interactions = [
    # Continuous × Continuous
    {
        "Age": {"type": "linear"},
        "VehPower": {"type": "linear"},
        "include_main": True,  # Adds Age + VehPower + Age:VehPower
    },
    # Categorical × Continuous
    {
        "Region": {"type": "categorical"},
        "Age": {"type": "bs", "df": 4},
        "include_main": True,  # Region-specific age curves
    },
    # Categorical × Categorical
    {
        "Region": {"type": "categorical"},
        "Area": {"type": "categorical"},
        "include_main": False,  # Interaction only
    },
]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `include_main` | bool | True | Include main effects alongside interaction |

### Target Encoding Interactions

Combined target encoding for variable combinations: `TE(Brand:Region)`.

```python
interactions = [
    {
        "Brand": {"type": "categorical"},
        "Region": {"type": "categorical"},
        "target_encoding": True,
        "prior_weight": 1.0,  # Optional
    },
]
```

Creates a single encoded column for the brand×region combination, useful for high-cardinality interaction effects.

### Frequency Encoding Interactions

Combined frequency encoding for variable combinations: `FE(Brand:Region)`.

```python
interactions = [
    {
        "Brand": {"type": "categorical"},
        "Region": {"type": "categorical"},
        "frequency_encoding": True,
    },
]
```

Encodes combinations by their frequency in the training data.

---

## Fitting

### fit()

Fit the model with optional regularization.

```python
result = model.fit()  # Standard IRLS

# With CV-based regularization
result = model.fit(regularization="ridge")  # "ridge", "lasso", "elastic_net"
result = model.fit(regularization="lasso", selection="1se", cv=5)

# With explicit alpha
result = model.fit(alpha=0.1, l1_ratio=0.0)  # Ridge
result = model.fit(alpha=0.1, l1_ratio=1.0)  # Lasso
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `regularization` | str | None | `"ridge"`, `"lasso"`, or `"elastic_net"` |
| `selection` | str | `"min"` | `"min"` or `"1se"` for CV selection |
| `cv` | int | 5 | Number of CV folds |
| `alpha` | float | 0.0 | Explicit regularization strength |
| `l1_ratio` | float | 1.0 | Elastic Net mixing (0=Ridge, 1=Lasso) |
| `cv_seed` | int | None | Seed for reproducible CV folds |

---

## Complete Examples

### Insurance Frequency Model

```python
import rustystats as rs
import polars as pl

data = pl.read_parquet("insurance.parquet")

result = rs.glm_dict(
    response="ClaimCount",
    terms={
        "VehAge": {"type": "bs", "monotonicity": "increasing"},
        "DrivAge": {"type": "bs"},
        "BonusMalus": {"type": "linear", "monotonicity": "increasing"},
        "VehPower": {"type": "linear"},
        "Region": {"type": "categorical"},
        "Brand": {"type": "target_encoding"},
    },
    interactions=[
        {
            "VehAge": {"type": "linear"},
            "Region": {"type": "categorical"},
            "include_main": True,
        },
    ],
    data=data,
    family="poisson",
    offset="Exposure",
    seed=42,
).fit()

print(result.summary())
```

### Regularized Model

```python
result = rs.glm_dict(
    response="ClaimCount",
    terms={
        "Age": {"type": "linear"},
        "Income": {"type": "linear"},
        "Region": {"type": "categorical"},
    },
    data=data,
    family="poisson",
).fit(regularization="elastic_net", selection="1se")

print(f"Selected alpha: {result.alpha}")
print(f"Non-zero features: {result.n_nonzero()}")
```

### High-Cardinality Features

```python
result = rs.glm_dict(
    response="ClaimCount",
    terms={
        "Age": {"type": "bs"},
        "Brand": {"type": "target_encoding"},
        "Model": {"type": "target_encoding"},
        "ZipCode": {"type": "target_encoding", "prior_weight": 2.0},
    },
    interactions=[
        {
            "Brand": {"type": "categorical"},
            "Region": {"type": "categorical"},
            "target_encoding": True,  # TE(Brand:Region)
        },
    ],
    data=data,
    family="poisson",
    offset="Exposure",
).fit()
```

---

## Validation

### validate()

Check design matrix for issues before fitting.

```python
model = rs.glm_dict(
    response="y",
    terms={"x": {"type": "ns", "df": 4}, "cat": {"type": "categorical"}},
    data=data,
    family="poisson",
)
results = model.validate()

if not results['valid']:
    print("Issues:", results['suggestions'])
```

**Checks performed:**
- Rank deficiency (linearly dependent columns)
- High multicollinearity (condition number)
- Zero variance columns
- NaN/Inf values
- Highly correlated column pairs (>0.999)

---

## Comparison: Dict API vs Formula API

| Feature | Dict API | Formula API |
|---------|----------|-------------|
| Programmatic building | ✓ Native | Requires string construction |
| Agent/automation friendly | ✓ Yes | String parsing |
| Complex interactions | ✓ Explicit | Limited syntax |
| TE interactions | ✓ Yes | Limited |
| FE interactions | ✓ Yes | No |
| Monotonicity constraints | ✓ All term types | Limited |

The Dict API is recommended for production systems and automated workflows.
