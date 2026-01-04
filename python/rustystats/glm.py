"""
GLM Summary Functions
=====================

This module provides summary formatting functions for GLM results.
These are used internally by the formula API.

Note: The array-based API (fit_glm, GLM class) has been removed.
Use the formula-based API instead:

>>> import rustystats as rs
>>> result = rs.glm("y ~ x1 + x2 + C(cat)", data, family="poisson").fit()
>>> print(result.summary())
"""

import numpy as np
from typing import Optional, List

from rustystats._rustystats import GLMResults


def summary(
    result: GLMResults,
    feature_names: Optional[List[str]] = None,
    title: str = "GLM Results",
    alpha: float = 0.05,
) -> str:
    """
    Generate a summary table for GLM results (statsmodels-style).
    
    Parameters
    ----------
    result : GLMResults
        Fitted GLM results object.
        
    feature_names : list of str, optional
        Names for each coefficient. If None, uses x0, x1, x2, ...
        
    title : str, optional
        Title for the summary table.
        
    alpha : float, optional
        Significance level for confidence intervals. Default 0.05 (95% CI).
    
    Returns
    -------
    str
        Formatted summary table.
    """
    n_params = len(result.params)
    
    # Generate feature names if not provided
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(n_params)]
    elif len(feature_names) != n_params:
        raise ValueError(
            f"feature_names has {len(feature_names)} elements but model has {n_params} parameters"
        )
    
    # Get statistics
    coefs = result.params
    std_errs = result.bse()
    z_vals = result.tvalues()
    p_vals = result.pvalues()
    conf_ints = result.conf_int(alpha)
    sig_codes = result.significance_codes()
    
    # Get diagnostics
    try:
        llf = result.llf()
        aic_val = result.aic()
        bic_val = result.bic()
        pearson_chi2 = result.pearson_chi2()
        null_dev = result.null_deviance()
        family_name = result.family
        scale = result.scale()
    except Exception:
        # Fallback if diagnostics not available
        llf = aic_val = bic_val = pearson_chi2 = null_dev = float('nan')
        family_name = "Unknown"
        scale = 1.0
    
    # Build the table
    lines = []
    lines.append("=" * 78)
    lines.append(title.center(78))
    lines.append("=" * 78)
    lines.append("")
    
    # Model info - statsmodels style
    lines.append(f"{'Family:':<20} {family_name:<15} {'No. Observations:':<20} {result.nobs:>10}")
    lines.append(f"{'Link Function:':<20} {'(default)':<15} {'Df Residuals:':<20} {result.df_resid:>10}")
    
    # Show regularization info if applicable
    try:
        is_reg = result.is_regularized
        penalty_type = result.penalty_type if is_reg else "none"
    except Exception:
        is_reg = False
        penalty_type = "none"
    
    if is_reg:
        method = f"IRLS + {penalty_type.title()}"
        lines.append(f"{'Method:':<20} {method:<15} {'Df Model:':<20} {result.df_model:>10}")
        lines.append(f"{'Scale:':<20} {scale:<15.4f} {'Alpha (λ):':<20} {result.alpha:>10.4f}")
        l1_val = result.l1_ratio if result.l1_ratio is not None else 0.0
        lines.append(f"{'L1 Ratio:':<20} {l1_val:<15.2f} {'Iterations:':<20} {result.iterations:>10}")
        try:
            n_nonzero = result.n_nonzero()
            lines.append(f"{'Non-zero coefs:':<20} {n_nonzero:<15}")
        except Exception:
            pass
    else:
        lines.append(f"{'Method:':<20} {'IRLS':<15} {'Df Model:':<20} {result.df_model:>10}")
        lines.append(f"{'Scale:':<20} {scale:<15.4f} {'Iterations:':<20} {result.iterations:>10}")
    lines.append("")
    
    # Goodness of fit
    lines.append(f"{'Log-Likelihood:':<20} {llf:>15.4f} {'Deviance:':<20} {result.deviance:>15.4f}")
    lines.append(f"{'AIC:':<20} {aic_val:>15.4f} {'Null Deviance:':<20} {null_dev:>15.4f}")
    lines.append(f"{'BIC:':<20} {bic_val:>15.4f} {'Pearson chi2:':<20} {pearson_chi2:>15.2f}")
    lines.append(f"{'Converged:':<20} {str(result.converged):<15}")
    lines.append("")
    lines.append("-" * 78)
    
    # Calculate dynamic column width for variable names
    # Use max of 16 chars or longest name (capped at 30)
    max_name_len = max(len(name) for name in feature_names)
    name_width = min(max(16, max_name_len), 30)
    
    # Coefficient table header
    ci_label = f"{int((1-alpha)*100)}% CI"
    header = f"{'Variable':<{name_width}} {'Coef':>10} {'Std.Err':>10} {'z':>8} {'P>|z|':>8} {ci_label:>22} {'':>4}"
    lines.append(header)
    lines.append("-" * 78)
    
    # Coefficient rows
    for i in range(n_params):
        name = feature_names[i][:name_width]  # Truncate only if exceeds max
        coef = coefs[i]
        se = std_errs[i]
        z = z_vals[i]
        p = p_vals[i]
        ci_low, ci_high = conf_ints[i]
        sig = sig_codes[i]
        
        # Format p-value
        if p < 0.0001:
            p_str = "<0.0001"
        else:
            p_str = f"{p:.4f}"
        
        ci_str = f"[{ci_low:>8.4f}, {ci_high:>8.4f}]"
        row = f"{name:<{name_width}} {coef:>10.4f} {se:>10.4f} {z:>8.3f} {p_str:>8} {ci_str:>22} {sig:>4}"
        lines.append(row)
    
    lines.append("-" * 78)
    lines.append("Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
    lines.append("=" * 78)
    
    return "\n".join(lines)


def summary_relativities(
    result: GLMResults,
    feature_names: Optional[List[str]] = None,
    title: str = "GLM Relativities (Log Link)",
    alpha: float = 0.05,
) -> str:
    """
    Generate a summary table showing relativities (exp of coefficients).
    
    This is appropriate for models with a log link (Poisson, Gamma).
    Relativities show the multiplicative effect of each variable.
    
    Parameters
    ----------
    result : GLMResults
        Fitted GLM results object (should use log link).
        
    feature_names : list of str, optional
        Names for each coefficient.
        
    title : str, optional
        Title for the summary table.
        
    alpha : float, optional
        Significance level for confidence intervals.
    
    Returns
    -------
    str
        Formatted summary table with relativities.
    
    Interpretation
    --------------
    A relativity of 1.15 for "Age 25-35" means that group has 15% higher
    claim frequency than the base level, all else being equal.
    """
    n_params = len(result.params)
    
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(n_params)]
    elif len(feature_names) != n_params:
        raise ValueError(
            f"feature_names has {len(feature_names)} elements but model has {n_params} parameters"
        )
    
    coefs = result.params
    conf_ints = result.conf_int(alpha)
    p_vals = result.pvalues()
    sig_codes = result.significance_codes()
    
    # Build the table
    lines = []
    lines.append("=" * 70)
    lines.append(title.center(70))
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"No. Observations: {result.nobs:>10}     Deviance: {result.deviance:>10.4f}")
    lines.append("")
    lines.append("-" * 70)
    
    ci_label = f"{int((1-alpha)*100)}% CI"
    header = f"{'Variable':<15} {'Coef':>10} {'Relativity':>12} {ci_label + ' (Rel)':>24} {'P>|z|':>8}"
    lines.append(header)
    lines.append("-" * 70)
    
    for i in range(n_params):
        name = feature_names[i][:15]
        coef = coefs[i]
        rel = np.exp(coef)
        ci_low_rel = np.exp(conf_ints[i, 0])
        ci_high_rel = np.exp(conf_ints[i, 1])
        p = p_vals[i]
        sig = sig_codes[i]
        
        if p < 0.0001:
            p_str = "<0.0001"
        else:
            p_str = f"{p:.4f}"
        
        ci_str = f"[{ci_low_rel:>8.4f}, {ci_high_rel:>8.4f}]"
        row = f"{name:<15} {coef:>10.4f} {rel:>12.4f} {ci_str:>24} {p_str:>8} {sig}"
        lines.append(row)
    
    lines.append("-" * 70)
    lines.append("Relativity = exp(Coef). Values > 1 increase the response.")
    lines.append("=" * 70)
    
    return "\n".join(lines)
