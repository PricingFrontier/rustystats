"""
GLM Summary Functions
=====================

This module provides summary formatting functions for GLM results.
These are used internally by the formula API.

Note: The array-based API (fit_glm, GLM class) has been removed.
Use the formula-based API instead:

>>> import rustystats as rs
>>> result = rs.glm_dict(response="y", terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}, "cat": {"type": "categorical"}}, data=data, family="poisson").fit()
>>> print(result.summary())
"""

from __future__ import annotations

import numpy as np

from rustystats._rustystats import GLMResults
from rustystats.exceptions import FittingError, ValidationError

_VALID_INFERENCE = frozenset({"valid_standard", "valid_robust"})

_INFERENCE_CAVEATS = {
    "naive_after_regularization": "ridge penalty shrinks coefficients",
    "naive_after_selection": "lasso/elastic-net performs variable selection",
    "naive_after_cv_selection": "alpha was chosen by cross-validation",
    "constrained_boundary": "sign/monotonicity constraints are active",
    "unavailable": "penalized smooth terms (use effective df)",
}


def summary(
    result: GLMResults,
    feature_names: list[str] | None = None,
    title: str = "GLM Results",
    alpha: float = 0.05,
    inference_status: str | None = None,
    solver_status: str | None = None,
    optimizer_route: str | None = None,
    effective_df: float | None = None,
    is_quasi_likelihood: bool = False,
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
        raise ValidationError(
            f"feature_names has {len(feature_names)} elements but model has {n_params} parameters"
        )

    # RS-ACT-011: standard inference is only trustworthy for an unpenalized,
    # unselected, unconstrained, non-smooth fit. Otherwise suppress the
    # significance machinery rather than present it as valid.
    suppress_inference = inference_status is not None and inference_status not in _VALID_INFERENCE

    # Get statistics
    coefs = result.params
    if not suppress_inference:
        std_errs = result.bse()
        z_vals = result.tvalues()
        p_vals = result.pvalues()
        conf_ints = result.conf_int(alpha)
        sig_codes = result.significance_codes()

    # Get diagnostics
    try:
        llf = result.llf()
        pearson_chi2 = result.pearson_chi2()
        null_dev = result.null_deviance()
        family_name = result.family
        scale = result.scale()
    except Exception as e:
        # Re-raise - summary diagnostics shouldn't fail silently
        raise FittingError(f"Failed to compute model summary diagnostics: {e}") from e
    if not is_quasi_likelihood:
        family_base = str(family_name).lower().split("(", 1)[0].strip()
        is_quasi_likelihood = family_base in {
            "quasipoisson",
            "quasi-poisson",
            "quasi_poisson",
            "quasibinomial",
            "quasi-binomial",
            "quasi_binomial",
        }

    aic_label = "AIC:"
    bic_label = "BIC:"
    aic_val: float | None = None
    bic_val: float | None = None

    # RS-ACT-008: quasi-likelihood families (quasi-Poisson, quasi-Binomial) do
    # not have a proper likelihood — print AIC/BIC as NA and never compute the
    # ordinary-likelihood value that would be silently misleading.
    if is_quasi_likelihood:
        aic_label = "AIC:"
        bic_label = "BIC:"
        # aic_val / bic_val stay None → printed as "NA" below.
    # RS-ACT-011: penalized smooth fits must be scored with their effective df,
    # not the basis-column count, so AIC/BIC reflect the realized complexity.
    elif effective_df is not None:
        aic_label = "AIC (edf):"
        bic_label = "BIC (edf):"
        aic_val = -2.0 * llf + 2.0 * effective_df
        bic_val = -2.0 * llf + effective_df * np.log(result.nobs)
    elif not suppress_inference:
        aic_val = result.aic()
        bic_val = result.bic()

    aic_text = "NA" if aic_val is None else f"{aic_val:.4f}"
    bic_text = "NA" if bic_val is None else f"{bic_val:.4f}"

    # Build the table
    lines = []
    lines.append("=" * 78)
    lines.append(title.center(78))
    lines.append("=" * 78)
    lines.append("")

    # Model info - statsmodels style
    lines.append(f"{'Family:':<20} {family_name:<15} {'No. Observations:':<20} {result.nobs:>10}")
    lines.append(
        f"{'Link Function:':<20} {'(default)':<15} {'Df Residuals:':<20} {result.df_resid:>10}"
    )

    # Show regularization info if applicable
    is_reg = result.is_regularized
    penalty_type = result.penalty_type if is_reg else "none"

    if is_reg:
        method = f"IRLS + {penalty_type.title()}"
        lines.append(f"{'Method:':<20} {method:<15} {'Df Model:':<20} {result.df_model:>10}")
        lines.append(f"{'Scale:':<20} {scale:<15.4f} {'Alpha (λ):':<20} {result.alpha:>10.4f}")
        l1_val = result.l1_ratio if result.l1_ratio is not None else 0.0
        lines.append(
            f"{'L1 Ratio:':<20} {l1_val:<15.2f} {'Iterations:':<20} {result.iterations:>10}"
        )
        # n_nonzero should always be available for regularized models
        n_nonzero = result.n_nonzero()
        lines.append(f"{'Non-zero coefs:':<20} {n_nonzero:<15}")
    else:
        lines.append(f"{'Method:':<20} {'IRLS':<15} {'Df Model:':<20} {result.df_model:>10}")
        lines.append(f"{'Scale:':<20} {scale:<15.4f} {'Iterations:':<20} {result.iterations:>10}")
    lines.append("")

    # Goodness of fit — relabel for quasi (RS-ACT-008): the value returned by
    # the family's loglik helper is the underlying Poisson / Binomial loglik,
    # but for quasi families it is *not* a true log-likelihood for the model
    # being fitted (dispersion φ is estimated separately). Showing the same
    # "Log-Likelihood:" label as for proper-likelihood families invites users
    # to compare it across non-nested models, which is invalid here.
    loglik_label = "Quasi-Log-Likelihood:" if is_quasi_likelihood else "Log-Likelihood:"
    lines.append(f"{loglik_label:<20} {llf:>15.4f} {'Deviance:':<20} {result.deviance:>15.4f}")
    lines.append(f"{aic_label:<20} {aic_text:>15} {'Null Deviance:':<20} {null_dev:>15.4f}")
    lines.append(f"{bic_label:<20} {bic_text:>15} {'Pearson chi2:':<20} {pearson_chi2:>15.2f}")
    lines.append(f"{'Converged:':<20} {result.converged!s:<15}")
    if inference_status is not None:
        lines.append(f"{'Inference:':<20} {inference_status:<15}")
    if solver_status is not None:
        route = f" / {optimizer_route}" if optimizer_route else ""
        lines.append(f"{'Solver status:':<20} {solver_status + route:<15}")
    if effective_df is not None:
        lines.append(f"{'Effective df:':<20} {effective_df:<15.3f}")
    lines.append("")
    lines.append("-" * 78)

    # Calculate dynamic column width for variable names
    # Use max of 16 chars or longest name (capped at 30)
    max_name_len = max(len(name) for name in feature_names)
    name_width = min(max(16, max_name_len), 30)

    # Coefficient table header
    if suppress_inference:
        header = f"{'Variable':<{name_width}} {'Coef':>10}"
    else:
        ci_label = f"{int((1 - alpha) * 100)}% CI"
        header = f"{'Variable':<{name_width}} {'Coef':>10} {'Std.Err':>10} {'z':>8} {'P>|z|':>8} {ci_label:>22} {'':>4}"
    lines.append(header)
    lines.append("-" * 78)

    # Coefficient rows
    for i in range(n_params):
        name = feature_names[i][:name_width]  # Truncate only if exceeds max
        coef = coefs[i]
        if suppress_inference:
            row = f"{name:<{name_width}} {coef:>10.4f}"
        else:
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
    if suppress_inference:
        reason = _INFERENCE_CAVEATS.get(inference_status, "the fitting procedure")
        if effective_df is None:
            lines.append(
                f"WARNING: standard errors, p-values, and AIC/BIC are NOT valid here ({reason});"
            )
        else:
            lines.append(f"WARNING: standard errors and p-values are NOT valid here ({reason});")
            lines.append("         AIC/BIC are shown with effective df, not raw parameter count.")
        lines.append(
            f"         inference_status={inference_status}. Treat coefficients descriptively."
        )
    else:
        lines.append("Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
    lines.append("=" * 78)

    return "\n".join(lines)


def summary_relativities(
    result: GLMResults,
    feature_names: list[str] | None = None,
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
        raise ValidationError(
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

    ci_label = f"{int((1 - alpha) * 100)}% CI"
    header = (
        f"{'Variable':<15} {'Coef':>10} {'Relativity':>12} {ci_label + ' (Rel)':>24} {'P>|z|':>8}"
    )
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
