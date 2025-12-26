# Results Object Reference

This page provides complete documentation for the `GLMResults` and `FormulaGLMResults` objects.

## GLMResults

Returned by `fit_glm()`. Contains all fitted model information.

### Construction

```python
result = rs.fit_glm(y, X, family="poisson")
```

---

## Coefficient Access

### params

Fitted coefficients as NumPy array.

```python
coefficients = result.params
print(coefficients)  # array([0.5, 0.3, -0.2])
```

### coefficients

Alias for `params` (statsmodels compatibility).

---

## Fitted Values

### fittedvalues

Predicted means μ = g⁻¹(Xβ).

```python
predicted = result.fittedvalues
print(f"Mean prediction: {predicted.mean():.4f}")
```

### linear_predictor

Linear predictor η = Xβ + offset.

```python
eta = result.linear_predictor
```

---

## Model Information

### deviance

Total model deviance.

```python
print(f"Deviance: {result.deviance:.2f}")
```

### iterations

Number of IRLS iterations until convergence.

```python
print(f"Converged in {result.iterations} iterations")
```

### converged

Whether the algorithm converged.

```python
if not result.converged:
    print("Warning: Model did not converge!")
```

### nobs

Number of observations.

```python
n = result.nobs
```

### df_resid

Residual degrees of freedom (n - p).

```python
df = result.df_resid
```

### df_model

Model degrees of freedom (p - 1, excluding intercept).

```python
df = result.df_model
```

### family

Family name as string.

```python
print(result.family)  # "Poisson" or "NegativeBinomial(theta=2.34)"
```

---

## Standard Errors and Inference

### bse()

Standard errors of coefficients.

```python
se = result.bse()
```

**Formula**: SE(β̂) = √(φ × diag((X'WX)⁻¹))

### tvalues()

z-statistics (or t-statistics).

```python
z = result.tvalues()
```

**Formula**: z = β̂ / SE(β̂)

### pvalues()

Two-sided p-values from z-distribution.

```python
p = result.pvalues()
for i, pval in enumerate(p):
    if pval < 0.05:
        print(f"Coefficient {i} is significant (p={pval:.4f})")
```

### conf_int()

Confidence intervals for coefficients.

```python
lower, upper = result.conf_int(alpha=0.05)  # 95% CI
```

**Parameters**:
- `alpha`: Significance level (default 0.05)

### significance_codes()

Get significance markers for each coefficient.

```python
codes = result.significance_codes()
# ['***', '**', '*', ''] for p < 0.001, 0.01, 0.05, otherwise
```

---

## Robust Standard Errors

Sandwich estimators that are robust to heteroscedasticity.

### bse_robust()

Robust standard errors.

```python
se_robust = result.bse_robust(hc_type="HC1")
```

**HC Types**:
- `"HC0"`: White's original estimator
- `"HC1"`: With (n/(n-p)) adjustment (default for most software)
- `"HC2"`: With leverage adjustment
- `"HC3"`: Jackknife-like (most conservative)

### tvalues_robust()

z-statistics using robust SEs.

```python
z_robust = result.tvalues_robust(hc_type="HC1")
```

### pvalues_robust()

p-values using robust SEs.

```python
p_robust = result.pvalues_robust(hc_type="HC1")
```

### conf_int_robust()

Confidence intervals using robust SEs.

```python
lower, upper = result.conf_int_robust(alpha=0.05, hc_type="HC1")
```

### cov_robust()

Full robust covariance matrix.

```python
cov = result.cov_robust(hc_type="HC1")
```

---

## Covariance Matrices

### cov_params_unscaled

Unscaled covariance matrix (X'WX)⁻¹.

```python
cov_unscaled = result.cov_params_unscaled
```

### cov_params()

Scaled covariance matrix φ(X'WX)⁻¹.

```python
cov = result.cov_params()
```

---

## Residuals

### resid_response()

Response residuals: y - μ.

```python
r = result.resid_response()
```

### resid_pearson()

Pearson residuals: (y - μ) / √V(μ).

```python
r = result.resid_pearson()
# Should be approximately N(0,1) if model is correct
```

### resid_deviance()

Deviance residuals: sign(y - μ) × √d.

```python
r = result.resid_deviance()
# Sum of squares equals deviance
print(f"Check: {(r**2).sum():.2f} ≈ {result.deviance:.2f}")
```

### resid_working()

Working residuals: (y - μ) × g'(μ).

```python
r = result.resid_working()
```

---

## Fit Statistics

### llf()

Log-likelihood of the fitted model.

```python
ll = result.llf()
```

### aic()

Akaike Information Criterion.

```python
aic = result.aic()
```

**Formula**: AIC = -2 × loglik + 2p

### bic()

Bayesian Information Criterion.

```python
bic = result.bic()
```

**Formula**: BIC = -2 × loglik + p × log(n)

### null_deviance()

Deviance of intercept-only model.

```python
null_dev = result.null_deviance()
pseudo_r2 = 1 - result.deviance / null_dev
```

### pearson_chi2()

Pearson chi-squared statistic.

```python
chi2 = result.pearson_chi2()
```

**Formula**: Σ (y - μ)² / V(μ)

### scale()

Dispersion parameter (deviance-based).

```python
phi = result.scale()
```

For Poisson/Binomial: Always 1
For QuasiPoisson/QuasiBinomial: Estimated from Pearson residuals
For Gaussian/Gamma: Deviance / df_resid

### scale_pearson()

Dispersion parameter (Pearson-based).

```python
phi = result.scale_pearson()
```

**Formula**: Pearson χ² / df_resid

---

## Regularization Methods

### n_nonzero()

Number of non-zero coefficients (for regularized models).

```python
result = rs.fit_glm(y, X, alpha=0.1, l1_ratio=1.0)
print(f"Selected {result.n_nonzero()} of {len(result.params)} features")
```

### selected_features()

Indices of non-zero coefficients.

```python
indices = result.selected_features()
print(f"Selected features: {indices}")
```

---

## Diagnostics Integration

### diagnostics()

Compute comprehensive diagnostics.

```python
diag = result.diagnostics(
    data=data,
    categorical_factors=["region"],
    continuous_factors=["age"],
)
```

### diagnostics_json()

Get diagnostics as JSON string.

```python
json_str = result.diagnostics_json(
    data=data,
    categorical_factors=["region"],
)
```

---

## FormulaGLMResults

Extends `GLMResults` with formula-specific functionality.

### feature_names

List of feature names.

```python
names = result.feature_names
# ['Intercept', 'x1', 'C(region)_B', 'C(region)_C']
```

### summary()

Print formatted summary table.

```python
print(result.summary())
```

### coef_table()

Coefficients as Polars DataFrame.

```python
df = result.coef_table()
```

### relativities()

Multiplicative effects (exp(coef)).

```python
rel = result.relativities()
```

### relativities_table()

Relativities as Polars DataFrame.

```python
df = result.relativities_table()
```

### predict()

Predict on new data.

```python
predictions = result.predict(new_data)
predictions_link = result.predict(new_data, type="link")
```
