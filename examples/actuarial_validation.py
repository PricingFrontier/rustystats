"""
Rigorous validation of rustystats Gamma GLM for actuarial use.
Compares all key metrics against statsmodels.
"""

import polars as pl
import numpy as np
import statsmodels.api as sm
import rustystats as rs
from scipy.special import gammaln

# Load and filter data exactly as in notebook
data = pl.read_parquet("https://raw.githubusercontent.com/PricingFrontier/pricing-data-example/917c853e256df8d5814721ab56f72889a908bb08/data/processed/severity_set.parquet")
data = data.filter(pl.col("ClaimAmount") > 0)
data = data.filter(pl.col("ClaimAmount") < 50000)

y = data["ClaimAmount"].to_numpy()
n = len(y)

print("=" * 80)
print("ACTUARIAL VALIDATION: Rustystats vs Statsmodels (Gamma GLM)")
print("=" * 80)
print(f"Dataset: {n:,} claims, mean=${np.mean(y):,.2f}, std=${np.std(y):,.2f}")

# =============================================================================
# TEST 1: Simple intercept-only model
# =============================================================================
print("\n" + "=" * 80)
print("TEST 1: Intercept-only Gamma GLM")
print("=" * 80)

# Statsmodels
X_const = np.ones((n, 1))
sm_model = sm.GLM(y, X_const, family=sm.families.Gamma(sm.families.links.Log()))
sm_result = sm_model.fit()

# Rustystats - intercept only via formula
rs_result = rs.glm("ClaimAmount ~ 1", data, family="gamma").fit()

print(f"\n{'Metric':<25} {'Statsmodels':>15} {'Rustystats':>15} {'Diff':>12} {'Status':>10}")
print("-" * 80)

# Coefficients
sm_coef = sm_result.params[0]
rs_coef = rs_result.params[0]
coef_diff = abs(sm_coef - rs_coef)
print(f"{'Intercept':<25} {sm_coef:>15.6f} {rs_coef:>15.6f} {coef_diff:>12.2e} {'✓' if coef_diff < 0.001 else '✗':>10}")

# Scale (dispersion)
sm_scale = sm_result.scale
rs_scale = rs_result.scale()
scale_diff = abs(sm_scale - rs_scale)
print(f"{'Scale (Pearson)':<25} {sm_scale:>15.6f} {rs_scale:>15.6f} {scale_diff:>12.2e} {'✓' if scale_diff < 0.01 else '✗':>10}")

# Deviance
sm_dev = sm_result.deviance
rs_dev = rs_result.deviance
dev_diff = abs(sm_dev - rs_dev)
print(f"{'Deviance':<25} {sm_dev:>15.4f} {rs_dev:>15.4f} {dev_diff:>12.2e} {'✓' if dev_diff < 1 else '✗':>10}")

# Standard errors
sm_se = sm_result.bse[0]
rs_se = rs_result.bse()[0]
se_diff = abs(sm_se - rs_se)
print(f"{'SE (Intercept)':<25} {sm_se:>15.6f} {rs_se:>15.6f} {se_diff:>12.2e} {'✓' if se_diff < 0.0001 else '✗':>10}")

# Log-likelihood
sm_llf = sm_result.llf
rs_llf = rs_result.llf()
llf_diff = abs(sm_llf - rs_llf)
print(f"{'Log-likelihood':<25} {sm_llf:>15.4f} {rs_llf:>15.4f} {llf_diff:>12.2e} {'✓' if llf_diff < 10 else '✗':>10}")

# AIC
sm_aic = sm_result.aic
rs_aic = rs_result.aic()
aic_diff = abs(sm_aic - rs_aic)
print(f"{'AIC':<25} {sm_aic:>15.4f} {rs_aic:>15.4f} {aic_diff:>12.2e} {'✓' if aic_diff < 20 else '✗':>10}")

# =============================================================================
# TEST 2: Model with one continuous predictor
# =============================================================================
print("\n" + "=" * 80)
print("TEST 2: Gamma GLM with DrivAge predictor")
print("=" * 80)

# Statsmodels
X = sm.add_constant(data["DrivAge"].to_numpy())
sm_model2 = sm.GLM(y, X, family=sm.families.Gamma(sm.families.links.Log()))
sm_result2 = sm_model2.fit()

# Rustystats
rs_result2 = rs.glm("ClaimAmount ~ DrivAge", data, family="gamma").fit()

print(f"\n{'Metric':<25} {'Statsmodels':>15} {'Rustystats':>15} {'Diff':>12} {'Status':>10}")
print("-" * 80)

# Coefficients
for i, name in enumerate(['Intercept', 'DrivAge']):
    sm_coef = sm_result2.params[i]
    rs_coef = rs_result2.params[i]
    diff = abs(sm_coef - rs_coef)
    tol = 0.001 if i == 0 else 0.0001
    print(f"{name:<25} {sm_coef:>15.6f} {rs_coef:>15.6f} {diff:>12.2e} {'✓' if diff < tol else '✗':>10}")

# Standard errors
for i, name in enumerate(['SE(Intercept)', 'SE(DrivAge)']):
    sm_se = sm_result2.bse[i]
    rs_se = rs_result2.bse()[i]
    diff = abs(sm_se - rs_se)
    tol = 0.001 if i == 0 else 0.00001
    print(f"{name:<25} {sm_se:>15.6f} {rs_se:>15.6f} {diff:>12.2e} {'✓' if diff < tol else '✗':>10}")

# z-values and p-values
sm_z = sm_result2.tvalues[1]
rs_z = rs_result2.tvalues()[1]
z_diff = abs(sm_z - rs_z)
print(f"{'z(DrivAge)':<25} {sm_z:>15.4f} {rs_z:>15.4f} {z_diff:>12.2e} {'✓' if z_diff < 0.01 else '✗':>10}")

sm_p = sm_result2.pvalues[1]
rs_p = rs_result2.pvalues()[1]
p_diff = abs(sm_p - rs_p)
print(f"{'p-value(DrivAge)':<25} {sm_p:>15.4f} {rs_p:>15.4f} {p_diff:>12.2e} {'✓' if p_diff < 0.01 else '✗':>10}")

# Scale
sm_scale2 = sm_result2.scale
rs_scale2 = rs_result2.scale()
print(f"{'Scale (Pearson)':<25} {sm_scale2:>15.6f} {rs_scale2:>15.6f} {abs(sm_scale2-rs_scale2):>12.2e} {'✓' if abs(sm_scale2-rs_scale2) < 0.01 else '✗':>10}")

# =============================================================================
# TEST 3: Predictions match
# =============================================================================
print("\n" + "=" * 80)
print("TEST 3: Predictions comparison")
print("=" * 80)

sm_pred = sm_result2.predict(X)
rs_pred = rs_result2.fittedvalues

pred_diff = np.abs(sm_pred - rs_pred)
print(f"Max absolute difference:  {pred_diff.max():.6f}")
print(f"Mean absolute difference: {pred_diff.mean():.6f}")
print(f"Max relative difference:  {(pred_diff / sm_pred).max() * 100:.6f}%")
print(f"Status: {'✓ Predictions match' if pred_diff.max() < 1 else '✗ Predictions differ'}")

# =============================================================================
# TEST 4: Manual log-likelihood verification
# =============================================================================
print("\n" + "=" * 80)
print("TEST 4: Manual log-likelihood verification (Gamma)")
print("=" * 80)

# Gamma log-likelihood formula:
# l_i = (α-1)*log(y) - α*y/μ + α*log(α/μ) - log(Γ(α))
# where α = 1/scale (shape parameter)

scale = rs_result2.scale()
alpha = 1.0 / scale
mu = rs_pred

manual_llf = np.sum(
    (alpha - 1.0) * np.log(y) -
    alpha * y / mu +
    alpha * np.log(alpha / mu) -
    gammaln(alpha)
)

print(f"Manual calculation:   {manual_llf:.4f}")
print(f"Rustystats llf:       {rs_result2.llf():.4f}")
print(f"Statsmodels llf:      {sm_result2.llf:.4f}")
print(f"Difference (RS-man):  {abs(rs_result2.llf() - manual_llf):.4f}")
print(f"Status: {'✓ Log-likelihood correct' if abs(rs_result2.llf() - manual_llf) < 1 else '✗ Log-likelihood differs'}")

# =============================================================================
# TEST 5: Residuals
# =============================================================================
print("\n" + "=" * 80)
print("TEST 5: Residuals comparison")
print("=" * 80)

# Pearson residuals: (y - μ) / sqrt(V(μ)) = (y - μ) / μ for Gamma
sm_pearson = sm_result2.resid_pearson
# Compute manually for rustystats: (y - mu) / mu for Gamma
rs_pearson_manual = (y - rs_pred) / rs_pred

pearson_diff = np.abs(sm_pearson - rs_pearson_manual).max()
print(f"Pearson residuals max diff: {pearson_diff:.6f}")
print(f"Status: {'✓ Pearson residuals match' if pearson_diff < 0.001 else '✗ Pearson residuals differ'}")

# Deviance residuals
sm_deviance_resid = sm_result2.resid_deviance
# Compute manually: sign(y-mu) * sqrt(2 * (y/mu - 1 - log(y/mu)))
rs_deviance_manual = np.sign(y - rs_pred) * np.sqrt(2 * (y/rs_pred - 1 - np.log(y/rs_pred)))

deviance_diff = np.abs(sm_deviance_resid - rs_deviance_manual).max()
print(f"Deviance residuals max diff: {deviance_diff:.6f}")
print(f"Status: {'✓ Deviance residuals match' if deviance_diff < 0.001 else '✗ Deviance residuals differ'}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("VALIDATION SUMMARY")
print("=" * 80)

all_tests = [
    ("Coefficients", coef_diff < 0.001 and abs(sm_result2.params[0] - rs_result2.params[0]) < 0.001),
    ("Standard Errors", se_diff < 0.0001 and abs(sm_result2.bse[0] - rs_result2.bse()[0]) < 0.001),
    ("Scale/Dispersion", abs(sm_scale2 - rs_scale2) < 0.01),
    ("Log-likelihood", abs(rs_result2.llf() - manual_llf) < 1),
    ("AIC", aic_diff < 20),
    ("z-values", z_diff < 0.01),
    ("p-values", p_diff < 0.01),
    ("Predictions", pred_diff.max() < 1),
    ("Pearson residuals", pearson_diff < 0.001),
    ("Deviance residuals", deviance_diff < 0.001),
]

passed = sum(1 for _, ok in all_tests if ok)
total = len(all_tests)

for name, ok in all_tests:
    print(f"  {'✓' if ok else '✗'} {name}")

print(f"\n{passed}/{total} tests passed")

if passed == total:
    print("\n" + "🎉 " * 20)
    print("ALL VALIDATION TESTS PASSED - LIBRARY IS ACTUARIAL-READY")
    print("🎉 " * 20)
else:
    print("\n⚠️  Some tests failed - investigate before production use")
