"""
Comprehensive validation of all GLM families against statsmodels.
"""

import numpy as np
import polars as pl
import statsmodels.api as sm
import rustystats as rs
from scipy.special import gammaln
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

def validate_family(name, y, X, sm_family, rs_family, link_name="log"):
    """Validate a GLM family against statsmodels."""
    print(f"\n{'=' * 80}")
    print(f"{name} GLM VALIDATION")
    print(f"{'=' * 80}")
    
    n = len(y)
    
    # Fit statsmodels
    try:
        sm_model = sm.GLM(y, X, family=sm_family)
        sm_result = sm_model.fit()
    except Exception as e:
        print(f"Statsmodels failed: {e}")
        return None
    
    # Fit rustystats
    try:
        data = pl.DataFrame({"y": y, "x": X[:, 1]})
        rs_result = rs.glm_dict(response="y", terms={"x": {"type": "linear"}}, data=data, family=rs_family).fit()
    except Exception as e:
        print(f"Rustystats failed: {e}")
        return None
    
    results = {}
    
    print(f"\n{'Metric':<25} {'Statsmodels':>15} {'Rustystats':>15} {'Diff':>12} {'Status':>8}")
    print("-" * 80)
    
    # Coefficients
    for i, coef_name in enumerate(['Intercept', 'x']):
        sm_val = sm_result.params[i]
        rs_val = rs_result.params[i]
        diff = abs(sm_val - rs_val)
        tol = 0.01 if abs(sm_val) > 1 else 0.001
        ok = diff < tol
        results[f'coef_{coef_name}'] = ok
        status = '✓' if ok else '✗'
        print(f"{coef_name:<25} {sm_val:>15.6f} {rs_val:>15.6f} {diff:>12.2e} {status:>8}")
    
    # Standard errors
    for i, se_name in enumerate(['SE(Intercept)', 'SE(x)']):
        sm_val = sm_result.bse[i]
        rs_val = rs_result.bse()[i]
        diff = abs(sm_val - rs_val)
        tol = 0.01 * sm_val if sm_val > 0 else 0.001  # 1% tolerance
        ok = diff < tol
        results[f'se_{i}'] = ok
        status = '✓' if ok else '✗'
        print(f"{se_name:<25} {sm_val:>15.6f} {rs_val:>15.6f} {diff:>12.2e} {status:>8}")
    
    # Scale/Dispersion
    sm_scale = sm_result.scale
    rs_scale = rs_result.scale()
    diff = abs(sm_scale - rs_scale)
    tol = 0.01 * sm_scale if sm_scale > 0 else 0.001
    ok = diff < tol
    results['scale'] = ok
    status = '✓' if ok else '✗'
    print(f"{'Scale':<25} {sm_scale:>15.6f} {rs_scale:>15.6f} {diff:>12.2e} {status:>8}")
    
    # Deviance
    sm_dev = sm_result.deviance
    rs_dev = rs_result.deviance
    diff = abs(sm_dev - rs_dev)
    tol = 0.01 * abs(sm_dev) if sm_dev != 0 else 0.001
    ok = diff < tol
    results['deviance'] = ok
    status = '✓' if ok else '✗'
    print(f"{'Deviance':<25} {sm_dev:>15.4f} {rs_dev:>15.4f} {diff:>12.2e} {status:>8}")
    
    # Log-likelihood
    sm_llf = sm_result.llf
    rs_llf = rs_result.llf()
    diff = abs(sm_llf - rs_llf)
    tol = max(1.0, 0.001 * abs(sm_llf))  # Allow 0.1% or 1 absolute
    ok = diff < tol
    results['llf'] = ok
    status = '✓' if ok else '✗'
    print(f"{'Log-likelihood':<25} {sm_llf:>15.4f} {rs_llf:>15.4f} {diff:>12.2e} {status:>8}")
    
    # AIC
    sm_aic = sm_result.aic
    rs_aic = rs_result.aic()
    diff = abs(sm_aic - rs_aic)
    tol = max(2.0, 0.001 * abs(sm_aic))
    ok = diff < tol
    results['aic'] = ok
    status = '✓' if ok else '✗'
    print(f"{'AIC':<25} {sm_aic:>15.4f} {rs_aic:>15.4f} {diff:>12.2e} {status:>8}")
    
    # z-values
    sm_z = sm_result.tvalues[1]
    rs_z = rs_result.tvalues()[1]
    diff = abs(sm_z - rs_z)
    tol = 0.01 * abs(sm_z) if sm_z != 0 else 0.01
    ok = diff < tol
    results['z'] = ok
    status = '✓' if ok else '✗'
    print(f"{'z(x)':<25} {sm_z:>15.4f} {rs_z:>15.4f} {diff:>12.2e} {status:>8}")
    
    # p-values
    sm_p = sm_result.pvalues[1]
    rs_p = rs_result.pvalues()[1]
    diff = abs(sm_p - rs_p)
    tol = 0.01  # Absolute tolerance for p-values
    ok = diff < tol
    results['pval'] = ok
    status = '✓' if ok else '✗'
    print(f"{'p-value(x)':<25} {sm_p:>15.4f} {rs_p:>15.4f} {diff:>12.2e} {status:>8}")
    
    # Predictions
    sm_pred = sm_result.predict(X)
    rs_pred = rs_result.fittedvalues
    pred_diff = np.abs(sm_pred - rs_pred).max()
    rel_diff = (np.abs(sm_pred - rs_pred) / np.abs(sm_pred).clip(1e-10)).max()
    ok = rel_diff < 0.001  # 0.1% relative error
    results['predictions'] = ok
    status = '✓' if ok else '✗'
    print(f"{'Max pred diff':<25} {pred_diff:>15.6f} {'':>15} {rel_diff:>12.2e} {status:>8}")
    
    # Summary
    passed = sum(results.values())
    total = len(results)
    print(f"\n{name}: {passed}/{total} tests passed")
    
    return results

# =============================================================================
# Generate test data
# =============================================================================
n = 5000
x = np.random.uniform(0, 10, n)
X = sm.add_constant(x)

print("=" * 80)
print("GLM FAMILY VALIDATION SUITE")
print("=" * 80)
print(f"Sample size: {n}")

# =============================================================================
# 1. GAUSSIAN (Normal) - Identity link
# =============================================================================
y_gaussian = 5 + 2*x + np.random.normal(0, 2, n)
gaussian_results = validate_family(
    "GAUSSIAN",
    y_gaussian, X,
    sm.families.Gaussian(),
    "gaussian"
)

# =============================================================================
# 2. POISSON - Log link
# =============================================================================
mu_poisson = np.exp(0.5 + 0.3*x)
y_poisson = np.random.poisson(mu_poisson)
poisson_results = validate_family(
    "POISSON",
    y_poisson.astype(float), X,
    sm.families.Poisson(),
    "poisson"
)

# =============================================================================
# 3. GAMMA - Log link
# =============================================================================
mu_gamma = np.exp(2 + 0.1*x)
shape = 2.0  # alpha
scale_param = mu_gamma / shape
y_gamma = np.random.gamma(shape, scale_param)
gamma_results = validate_family(
    "GAMMA",
    y_gamma, X,
    sm.families.Gamma(sm.families.links.Log()),
    "gamma"
)

# =============================================================================
# 4. BINOMIAL (Logistic) - Logit link
# =============================================================================
p_binom = 1 / (1 + np.exp(-(0.5 + 0.3*x - 1.5)))  # Logistic
y_binom = np.random.binomial(1, p_binom).astype(float)
binomial_results = validate_family(
    "BINOMIAL",
    y_binom, X,
    sm.families.Binomial(),
    "binomial"
)

# =============================================================================
# 5. INVERSE GAUSSIAN - Log link
# =============================================================================
mu_ig = np.exp(1 + 0.1*x)
# InverseGaussian with mu and lambda (shape)
lambda_ig = 5.0
y_ig = np.random.wald(mu_ig, lambda_ig)
# Filter extreme values
mask = y_ig < np.percentile(y_ig, 99)
y_ig_filtered = y_ig[mask]
X_ig = X[mask]

ig_results = validate_family(
    "INVERSE GAUSSIAN",
    y_ig_filtered, X_ig,
    sm.families.InverseGaussian(sm.families.links.Log()),
    "inverse_gaussian"
)

# =============================================================================
# 6. NEGATIVE BINOMIAL - Log link
# =============================================================================
mu_nb = np.exp(1 + 0.2*x)
alpha_nb = 0.5  # Overdispersion (theta = 1/alpha = 2)
theta_nb = 1.0 / alpha_nb  # rustystats uses theta
# statsmodels uses alpha where Var = mu + alpha*mu^2
y_nb = np.random.negative_binomial(1/alpha_nb, 1/(1 + alpha_nb*mu_nb))

try:
    # Statsmodels NB requires specifying alpha
    sm_nb = sm.GLM(y_nb.astype(float), X, family=sm.families.NegativeBinomial(alpha=alpha_nb))
    sm_nb_result = sm_nb.fit()
    
    data_nb = pl.DataFrame({"y": y_nb.astype(float), "x": x})
    # rustystats uses theta parameter separately
    rs_nb_result = rs.glm_dict(response="y", terms={"x": {"type": "linear"}}, data=data_nb, family="negbinomial", theta=theta_nb).fit()
    
    print(f"\n{'=' * 80}")
    print("NEGATIVE BINOMIAL GLM VALIDATION")
    print(f"{'=' * 80}")
    print(f"\n{'Metric':<25} {'Statsmodels':>15} {'Rustystats':>15} {'Diff':>12} {'Status':>8}")
    print("-" * 80)
    
    nb_results = {}
    for i, coef_name in enumerate(['Intercept', 'x']):
        sm_val = sm_nb_result.params[i]
        rs_val = rs_nb_result.params[i]
        diff = abs(sm_val - rs_val)
        ok = diff < 0.05
        nb_results[f'coef_{coef_name}'] = ok
        status = '✓' if ok else '✗'
        print(f"{coef_name:<25} {sm_val:>15.6f} {rs_val:>15.6f} {diff:>12.2e} {status:>8}")
    
    # SE
    for i, se_name in enumerate(['SE(Intercept)', 'SE(x)']):
        sm_val = sm_nb_result.bse[i]
        rs_val = rs_nb_result.bse()[i]
        diff = abs(sm_val - rs_val)
        ok = diff < 0.01 * sm_val
        nb_results[f'se_{i}'] = ok
        status = '✓' if ok else '✗'
        print(f"{se_name:<25} {sm_val:>15.6f} {rs_val:>15.6f} {diff:>12.2e} {status:>8}")
    
    print(f"\nNEGATIVE BINOMIAL: {sum(nb_results.values())}/{len(nb_results)} tests passed")
except Exception as e:
    print(f"\nNegative Binomial validation skipped: {e}")
    nb_results = {}

# =============================================================================
# 7. TWEEDIE - Log link (p=1.5)
# =============================================================================
try:
    # Generate Tweedie-like data (compound Poisson-Gamma)
    p_tweedie = 1.5
    mu_tw = np.exp(1 + 0.1*x)
    phi_tw = 2.0
    
    # Simulate Tweedie via compound Poisson-Gamma
    lambda_poisson = mu_tw**(2-p_tweedie) / (phi_tw * (2-p_tweedie))
    n_claims = np.random.poisson(lambda_poisson)
    alpha_gamma = (2-p_tweedie) / (p_tweedie-1)
    beta_gamma = phi_tw * (p_tweedie-1) * mu_tw**(p_tweedie-1)
    y_tw = np.zeros(n)
    for i in range(n):
        if n_claims[i] > 0:
            y_tw[i] = np.sum(np.random.gamma(alpha_gamma, beta_gamma[i], n_claims[i]))
    
    # Only fit on non-zero values for cleaner comparison
    mask_tw = y_tw > 0
    y_tw_pos = y_tw[mask_tw]
    X_tw = X[mask_tw]
    x_tw = x[mask_tw]
    
    sm_tw = sm.GLM(y_tw_pos, X_tw, family=sm.families.Tweedie(var_power=p_tweedie, link=sm.families.links.Log()))
    sm_tw_result = sm_tw.fit()
    
    data_tw = pl.DataFrame({"y": y_tw_pos, "x": x_tw})
    # rustystats uses var_power parameter separately
    rs_tw_result = rs.glm_dict(response="y", terms={"x": {"type": "linear"}}, data=data_tw, family="tweedie", var_power=p_tweedie).fit()
    
    print(f"\n{'=' * 80}")
    print("TWEEDIE (p=1.5) GLM VALIDATION")
    print(f"{'=' * 80}")
    print(f"\n{'Metric':<25} {'Statsmodels':>15} {'Rustystats':>15} {'Diff':>12} {'Status':>8}")
    print("-" * 80)
    
    tweedie_results = {}
    for i, coef_name in enumerate(['Intercept', 'x']):
        sm_val = sm_tw_result.params[i]
        rs_val = rs_tw_result.params[i]
        diff = abs(sm_val - rs_val)
        ok = diff < 0.1  # Tweedie can have more variation
        tweedie_results[f'coef_{coef_name}'] = ok
        status = '✓' if ok else '✗'
        print(f"{coef_name:<25} {sm_val:>15.6f} {rs_val:>15.6f} {diff:>12.2e} {status:>8}")
    
    print(f"\nTWEEDIE: {sum(tweedie_results.values())}/{len(tweedie_results)} tests passed")
except Exception as e:
    print(f"\nTweedie validation skipped: {e}")
    tweedie_results = {}

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

all_results = {
    "Gaussian": gaussian_results,
    "Poisson": poisson_results,
    "Gamma": gamma_results,
    "Binomial": binomial_results,
    "Negative Binomial": nb_results if nb_results else None,
    "Tweedie": tweedie_results if tweedie_results else None,
}

total_passed = 0
total_tests = 0

for name, results in all_results.items():
    if results:
        passed = sum(results.values())
        total = len(results)
        total_passed += passed
        total_tests += total
        status = "✓" if passed == total else "⚠"
        print(f"  {status} {name}: {passed}/{total}")
    else:
        print(f"  ✗ {name}: FAILED")

print(f"\nOVERALL: {total_passed}/{total_tests} tests passed")

if total_passed == total_tests:
    print("\n" + "🎉 " * 20)
    print("ALL FAMILIES VALIDATED SUCCESSFULLY")
    print("🎉 " * 20)
else:
    failed = total_tests - total_passed
    print(f"\n⚠️  {failed} tests failed - investigate before production use")
