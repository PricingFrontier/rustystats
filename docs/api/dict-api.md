# Dict API Reference

The Dict API (`glm_dict`) is RustyStats' primary interface, designed for programmatic model building ideal for automated workflows and agents.

## glm_dict

Create a GLM specification with dict-based term definitions.

```python
rustystats.glm_dict(
    response,
    terms,
    data,
    interactions=None,
    intercept=True,
    family="gaussian",
    link=None,
    var_power=1.5,
    theta=None,
    exposure=None,
    offset=None,
    weights=None,
    seed=None,
    complement=None,
    input_transforms=None,
    allow_extended_tweedie=False,
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `response` | str | Column name for response variable |
| `terms` | dict | Term specifications (see below) |
| `data` | DataFrame | Polars DataFrame or LazyFrame |
| `interactions` | list | Interaction specifications (see below) |
| `intercept` | bool | Include an intercept term. Default `True`. |
| `family` | str | Distribution family |
| `link` | str | Link function (optional) |
| `var_power` | float | Tweedie variance power, default 1.5 (compound Poisson-Gamma interior) |
| `theta` | float or `"estimate"` | Negative Binomial dispersion. Pass a positive number for a fixed-theta fit, or `"estimate"` for profile estimation on the plain unpenalized path. For Negative Binomial, leaving `theta` unspecified raises. Embedded forms such as `family="negbinomial(theta=2.0)"` are also accepted. |
| `exposure` | str or array | **Preferred** raw positive denominator for rate models. Added as `log(exposure)` to the linear predictor under log link, and used as the rate denominator for exposure-weighted target encoding. |
| `offset` | str or array | Link-scale additive offset, used as-is. A string names a column added verbatim on the link scale; it is **never** treated as raw exposure. Use `exposure=` for the rate denominator. |
| `weights` | str or array | Prior weights |
| `seed` | int | Random seed for reproducibility |
| `complement` | str, array, or `GLMModel` | Complement-of-credibility prior (response scale). Used by lasso shrinkage. |
| `input_transforms` | list[dict] | Deterministic raw-input transforms applied before design-matrix construction. See [Input Transforms](#input-transforms). |
| `allow_extended_tweedie` | bool | Opt-in for Tweedie powers outside the default `1 < p < 2` interior. Default `False`. See [Distribution Families: Tweedie support contract](../theory/families.md#66-support-contract-rs-act-006). |

### exposure= vs offset= (RS-ACT-002)

These are *separate* concepts in RustyStats:

* **`exposure=`** — a raw positive denominator (units, person-years, vehicle-years).
  Validates as finite and strictly positive. Under log link, contributes
  `log(exposure)` to the linear predictor. Also flows into rate-ranked
  diagnostics and exposure-weighted target encoding denominators.
* **`offset=`** — an arbitrary link-scale adjustment (e.g. a fixed coefficient
  estimated elsewhere). Treated verbatim on the link scale; never used as a
  rate denominator for target encoding.

If both are supplied, both contribute additively:
`η_total = η_terms + log(exposure) + offset`.

#### Three migration spellings

```python
# Preferred — explicit raw exposure.
rs.glm_dict(..., family="poisson", exposure="Exposure")

# Already-log-transformed exposure + a separate exposure column for
# diagnostics / target encoding.
rs.glm_dict(..., family="poisson", offset=np.log(exposure), exposure=exposure)

# A string `offset=` is a link-scale column added verbatim (NOT raw exposure).
# Use `exposure=` for the rate denominator.
rs.glm_dict(..., family="poisson", offset="link_scale_adjustment_column")
```

A subtle but **deliberate behaviour change**: an array `offset=np.log(...)`
no longer feeds exposure-weighted target encoding as if it were raw exposure.
Set `exposure=` explicitly to recover that behaviour.

### Returns

`FormulaGLMDict` object - call `.fit()` to fit the model.

---

## multinomial_dict

Create a native baseline-category multinomial logit specification for mutually
exclusive class outcomes, such as insurance product-tier conversion.

```python
rustystats.multinomial_dict(
    response,
    data,
    terms=None,
    shared_terms=None,
    alternative_terms=None,
    interactions=None,
    intercept=True,
    classes=None,
    reference=None,
    availability=None,
    weights=None,
    class_weights=None,
    offset=None,
    seed=None,
    input_transforms=None,
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `response` | str | Categorical response column |
| `data` | DataFrame | Polars DataFrame or LazyFrame |
| `terms` / `shared_terms` | dict | Shared-covariate term specifications. Pass only one. |
| `alternative_terms` | dict | Wide-format class-specific covariates such as tier price, deductible, limit, or richness |
| `interactions` | list | Standard interaction specifications |
| `intercept` | bool | Include a shared-design intercept. Default `True`. |
| `classes` | list | Explicit output class order. Recommended for pricing workflows. |
| `reference` | str | Baseline class. Defaults to the first class. |
| `availability` | dict or array | Optional class availability mask. Dict values may be booleans, column names, or arrays. |
| `weights` | str or array | Row weights |
| `class_weights` | dict | Multipliers applied by observed response class |
| `offset` | dict or array | Class-specific utility/logit offsets. This is not exposure. |
| `seed` | int | Random seed for deterministic design components |
| `input_transforms` | list[dict] | Deterministic raw-input transforms applied before design construction |

Fit with:

```python
result = model.fit(
    alpha=0.0,
    l1_ratio=0.0,
    regularization=None,
    max_iter=100,
    tol=1e-8,
    standardize=True,
    compute_covariance=True,
)
```

The multinomial path supports unpenalized and ridge dense Newton fits for shared
covariates and `alternative_terms`. Alternative-term ridge uses the same
standardization/back-transform policy as shared covariates, with inference
labelled as naive after regularization. Lasso, elastic net, CV, automatic smooth
penalties, target encoding, monotonic constraints, exposure, symmetric
reference-invariant ridge, and PMML/ONNX export are rejected with explicit
validation errors or reserved for later native support.

Alternative terms use wide-format columns:

```python
result = rustystats.multinomial_dict(
    response="PurchasedTier",
    shared_terms={"DriverAge": {"type": "bs", "df": 6}},
    alternative_terms={
        "price": {
            "columns": {
                "basic": "price_basic",
                "standard": "price_standard",
                "premium": "price_premium",
            },
            "coefficient": "generic",
            "transform": "log",
        },
        "richness": {
            "columns": {
                "basic": "richness_basic",
                "standard": "richness_standard",
                "premium": "richness_premium",
            },
            "coefficient": "class_specific",
        },
    },
    data=quotes,
    classes=["none", "basic", "standard", "premium"],
    reference="none",
).fit()
```

`coefficient="generic"` estimates one shared coefficient across alternatives.
`coefficient="class_specific"` estimates one coefficient per non-reference
class. Missing classes in an alternative term default to zero, which is useful
for a `"none"` reference with no offered price.

Prediction methods:

```python
result.predict_proba(new_data)
result.predict_log_proba(new_data)
result.decision_function(new_data)
result.predict(new_data)
result.predict_top_k(new_data, k=2)
result.tier_mix(new_data)
result.scenario(new_data, changes={"price_premium": 1.03})
result.diagnostics(
    train_data=quotes,
    test_data=holdout,
    categorical_factors=["Region", "Channel"],
    continuous_factors=["DriverAge", "VehicleValue"],
)
result.diagnostics_json(train_data=quotes, test_data=holdout)
```

Unpenalized fits are invariant to the chosen reference class up to coefficient
reparametrization. Baseline ridge is reference-dependent because it shrinks
non-reference utilities toward the selected reference.

`result.diagnostics()` returns a `MultinomialDiagnostics` object with weighted
log loss, deviance/null deviance, McFadden pseudo R2, AIC/BIC when likelihood
inference is well-defined, a `K x K` confusion matrix, accuracy, balanced
accuracy, macro/per-class precision/recall/F1, actual versus predicted class
mix, class-wise calibration curves and expected calibration error, reliability
by predicted winning class, factor-level class-mix diagnostics, and optional
train/test comparison. Class-weighted and regularized fits keep diagnostics
available but label coefficient/AIC-style inference as naive or not applicable.

`result.scenario()` returns a `MultinomialScenario` with base/scenario class
mix, class-mix deltas, optional expected value comparison via `value_columns=`,
and optional segment-level mix deltas via `categorical_factors=` /
`continuous_factors=`. If a scenario change updates a column named in
`value_columns`, the scenario expected value uses the changed column values
while the base expected value uses the original values.

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

## Input Transforms

`input_transforms` let a fitted `GLMModel` score raw production data directly
when some model terms are deterministic derived columns. The first supported
transform type is `lookup`, which maps one or more raw source columns to a
numeric or string output column before the design matrix is built.

```python
input_transforms = [
    {
        "type": "lookup",
        "name": "brand_region_effect",
        "sources": ["Brand", "Region"],
        "output": "brand_region_fts",
        "output_dtype": "float64",
        "keys": [["Ford", "North"], ["BMW", "South"]],
        "values": [0.07, -0.03],
        "default": 0.0,
        "on_unseen": "default",
        "on_null": "default",
    }
]

result = rs.glm_dict(
    response="ClaimNb",
    terms={"brand_region_fts": {"type": "linear"}},
    input_transforms=input_transforms,
    data=train_raw,
    family="poisson",
    exposure="Exposure",
).fit()

# score_raw only needs Brand, Region, and Exposure.
pred = result.predict(score_raw)
```

Lookup keys are structured rows, not delimiter-joined strings, so multi-column
keys are safe when raw levels contain characters such as `"|"`. Transforms are
stored with the fitted model and are applied consistently by `predict`,
`predict_contributions`, diagnostics, calibration helpers, and serialization.

`result.prepare_input(raw_df)` is available for debugging and parity tests. It
returns a new Polars DataFrame with derived columns added and does not mutate
the caller's input.

### Lookup Transform Fields

| Field | Type | Description |
|-------|------|-------------|
| `type` | `"lookup"` | Transform type |
| `name` | str | Stable transform identifier |
| `sources` | list[str] | Raw source columns, in key order |
| `output` | str | Derived column created before modelling |
| `output_dtype` | `"float64"` or `"string"` | Output dtype |
| `keys` | list[list[str \| null]] | Structured lookup keys |
| `values` | list[float \| str] | Values aligned to `keys` |
| `default` | float or str | Fallback when default policy is used |
| `on_unseen` | `"default"` or `"raise"` | Unseen-key policy |
| `on_null` | `"default"`, `"raise"`, or `"match"` | Null-key policy |

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
| `standardize` | bool | True | Standardize penalized columns internally, then report original-scale coefficients |
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
    exposure="Exposure",
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
    exposure="Exposure",
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
