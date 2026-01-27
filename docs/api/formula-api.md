# Formula API Reference

The formula API provides a high-level interface using R-style formulas and DataFrames.

## glm

Create a GLM specification with formula.

```python
rustystats.glm(
    formula,
    data,
    family="gaussian",
    link=None,
    offset=None,
    weights=None,
    alpha=0.0,
    l1_ratio=1.0,
    theta=None,
    var_power=1.5,
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `formula` | str | R-style formula (e.g., `"y ~ x1 + x2"`) |
| `data` | DataFrame | Polars or Pandas DataFrame |
| `family` | str | Distribution family |
| `link` | str | Link function (optional) |
| `offset` | str | Column name for offset |
| `weights` | str | Column name for weights |
| `alpha` | float | Regularization strength |
| `l1_ratio` | float | Elastic Net mixing |
| `theta` | float | Negative Binomial dispersion |
| `var_power` | float | Tweedie variance power |

### Returns

`FormulaGLM` object - call `.fit()` to fit the model.

### Example

```python
import rustystats as rs
import polars as pl

data = pl.DataFrame({
    "y": [1, 2, 3, 4, 5],
    "x1": [1.0, 2.0, 3.0, 4.0, 5.0],
    "x2": [5.0, 4.0, 3.0, 2.0, 1.0],
    "region": ["A", "B", "A", "B", "A"],
})

result = rs.glm(
    formula="y ~ x1 + x2 + C(region)",
    data=data,
    family="gaussian"
).fit()
```

---

## Formula Syntax

### Basic Terms

```python
"y ~ x1 + x2"           # Continuous variables
"y ~ x1 + C(region)"    # Categorical with C()
"y ~ x1 - 1"            # Remove intercept
```

### Categorical Variables

Use `C()` to mark categorical variables:

```python
"y ~ C(region)"                    # Single categorical
"y ~ C(region) + C(vehicle_type)"  # Multiple categoricals
"y ~ x1 + C(region, ref='A')"      # Custom reference level
```

### Interactions

```python
"y ~ x1 * x2"           # Main effects + interaction: x1 + x2 + x1:x2
"y ~ x1 : x2"           # Pure interaction only
"y ~ C(region) * age"   # Categorical × continuous
"y ~ C(a) * C(b)"       # Categorical × categorical
```

### Splines

```python
"y ~ bs(age, df=5)"                    # B-spline with 5 df
"y ~ ns(income, df=4)"                 # Natural spline
"y ~ bs(age, df=5, degree=3)"          # Specify degree
"y ~ bs(age, df=4) * C(gender)"        # Spline × categorical
```

### Target Encoding

```python
"y ~ TE(brand)"                        # Target encoding
"y ~ TE(zipcode, prior_weight=2.0)"    # With options
"y ~ TE(brand) + age + C(region)"      # Mixed
```

### Complex Formulas

```python
# Insurance frequency model
"ClaimNb ~ C(Area) + C(VehBrand) + bs(DrivAge, df=4) + VehPower + log(Density)"

# With interactions
"ClaimNb ~ C(Area) * VehPower + C(VehBrand) + ns(DrivAge, df=5)"

# With target encoding for high-cardinality
"ClaimNb ~ TE(VehicleModel) + C(Area) + bs(Age, df=4)"
```

---

## FormulaGLM

Object returned by `glm()`.

### Methods

| Method | Description |
|--------|-------------|
| `.fit()` | Fit the model, returns GLMModel |

### Example

```python
model = rs.glm("y ~ x1 + C(region)", data, family="poisson")
result = model.fit()
```

---

## GLMModel

Results from fitting a formula-based GLM. Inherits all methods from `GLMResults` plus formula-specific methods.

### Additional Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `feature_names` | list | Names of all features including encoded categoricals |
| `formula` | str | Original formula string |

### Additional Methods

```python
# Formatted summary table
result.summary()

# Coefficient table as DataFrame
result.coef_table()

# Relativities (exp(coef) for log-link models)
result.relativities()  # Returns DataFrame
```

### summary()

Print formatted regression table:

```python
result = rs.glm("y ~ x1 + C(region)", data, family="poisson").fit()
print(result.summary())
```

Output:
```
                 GLM Results                  
==============================================
Family:        Poisson
Link:          Log
Observations:  1000
Df Residual:   997
Df Model:      2
Deviance:      1234.56
AIC:           1240.56

                  coef    std err      z      P>|z|
--------------------------------------------------
Intercept       0.5234     0.0512   10.22    0.000 ***
x1              0.1234     0.0234    5.27    0.000 ***
C(region)_B    -0.2341     0.0456   -5.13    0.000 ***
C(region)_C     0.1567     0.0423    3.70    0.000 ***
--------------------------------------------------
Signif. codes: *** p<0.001, ** p<0.01, * p<0.05
```

### coef_table()

Get coefficients as a DataFrame:

```python
table = result.coef_table()
print(table)
```

| feature | coef | std_err | z | pvalue | ci_lower | ci_upper |
|---------|------|---------|---|--------|----------|----------|
| Intercept | 0.523 | 0.051 | 10.22 | 0.000 | 0.423 | 0.624 |
| x1 | 0.123 | 0.023 | 5.27 | 0.000 | 0.078 | 0.169 |
| ... | ... | ... | ... | ... | ... | ... |

### relativities()

Get multiplicative effects (exp(coef)) for log-link models as a DataFrame:

```python
rel = result.relativities()
```

| Feature | Relativity | CI_Lower | CI_Upper |
|---------|------------|----------|----------|
| Intercept | 1.687 | 1.526 | 1.868 |
| C(region)_B | 0.791 | 0.722 | 0.868 |
| ... | ... | ... | ... |

---

## Prediction

### predict()

Make predictions on new data:

```python
new_data = pl.DataFrame({
    "x1": [1.0, 2.0],
    "region": ["A", "B"],
})

predictions = result.predict(new_data)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `new_data` | DataFrame | New data with same columns as training |
| `offset` | str or array | Offset for predictions (optional) |

### Example

```python
# Predict on response scale (μ = E[Y])
predictions = result.predict(new_data)

# With custom offset
predictions = result.predict(new_data, offset="Exposure")
```

---

## Diagnostics

### diagnostics()

Compute comprehensive model diagnostics:

```python
diagnostics = result.diagnostics(
    data=data,
    categorical_factors=["region", "brand"],
    continuous_factors=["age", "income"],
)
```

### diagnostics_json()

Get diagnostics as JSON string:

```python
json_str = result.diagnostics_json(
    data=data,
    categorical_factors=["region"],
    continuous_factors=["age"],
)
```

See [Diagnostics API](diagnostics.md) for details.

---

## Complete Example

```python
import rustystats as rs
import polars as pl

# Load data
data = pl.read_parquet("insurance_claims.parquet")

# Fit model
result = rs.glm(
    formula="ClaimNb ~ bs(DrivAge, df=5) + C(Area) + C(VehBrand) + VehPower",
    data=data,
    family="poisson",
    offset="Exposure"  # log(Exposure) applied automatically
).fit()

# View results
print(result.summary())

# Relativities for pricing
print("\nRelativities:")
print(result.relativities_table())

# Model diagnostics
diagnostics = result.diagnostics(
    data=data,
    categorical_factors=["Area", "VehBrand"],
    continuous_factors=["DrivAge", "VehPower"],
)
print(f"\nGini: {diagnostics.discrimination['gini_coefficient']:.3f}")
print(f"A/E: {diagnostics.calibration['overall_ae']:.3f}")

# Predict on new data
new_policies = pl.DataFrame({
    "DrivAge": [25, 45],
    "Area": ["A", "B"],
    "VehBrand": ["Toyota", "BMW"],
    "VehPower": [5, 7],
    "Exposure": [1.0, 1.0],
})
predictions = result.predict(new_policies)
print(f"\nPredicted frequencies: {predictions}")
```
