"""
GLM: Generalized Linear Models
==============================

This module provides a user-friendly interface for fitting GLMs,
designed to be familiar to statsmodels users.

Basic Usage
-----------
>>> import rustystats as rs
>>> import numpy as np
>>>
>>> # Prepare data (X should include intercept column)
>>> X = np.column_stack([np.ones(n), x1, x2])
>>> y = ...
>>>
>>> # Method 1: Direct function call
>>> result = rs.fit_glm(y, X, family="poisson")
>>>
>>> # Method 2: Object-oriented (statsmodels-style)
>>> model = rs.GLM(y, X, family="poisson")
>>> result = model.fit()
>>>
>>> # Access results
>>> print(result.params)        # Coefficients
>>> print(result.bse())         # Standard errors
>>> print(result.pvalues())     # P-values
>>> print(result.conf_int())    # Confidence intervals
>>> print(result.deviance)      # Model deviance
>>>
>>> # Nice summary table
>>> print(summary(result))
"""

import numpy as np
from typing import Optional, Union, List

# Import the Rust fitting function
from rustystats._rustystats import fit_glm_py as _fit_glm_rust, GLMResults


def fit_glm(
    y: np.ndarray,
    X: np.ndarray,
    family: str = "gaussian",
    link: Optional[str] = None,
    offset: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    max_iter: int = 25,
    tol: float = 1e-8,
) -> GLMResults:
    """
    Fit a Generalized Linear Model.
    
    This is the main function for fitting GLMs. It uses IRLS
    (Iteratively Reweighted Least Squares) internally.
    
    Parameters
    ----------
    y : array-like, shape (n,)
        Response variable. Should be:
        - Any real values for Gaussian
        - Non-negative integers for Poisson
        - 0/1 or proportions in [0,1] for Binomial
        - Positive values for Gamma
        
    X : array-like, shape (n, p)
        Design matrix (predictors/features).
        **Important**: Include a column of 1s if you want an intercept!
        
    family : str, default="gaussian"
        Distribution family. Options:
        - "gaussian" or "normal": For continuous data
        - "poisson": For count data (claim frequency)
        - "binomial": For binary/proportion data
        - "gamma": For positive continuous data (claim severity)
        
    link : str, optional
        Link function. If None, uses the canonical link:
        - Gaussian: "identity"
        - Poisson: "log"
        - Binomial: "logit"
        - Gamma: "log"
        
        Other options depend on family:
        - "identity": η = μ
        - "log": η = log(μ)
        - "logit": η = log(μ/(1-μ))
    
    offset : array-like, shape (n,), optional
        Offset term added to the linear predictor.
        For rate models with exposure: offset = log(exposure)
        
        Example: Claim frequency per policy-year
        >>> offset = np.log(exposure_years)
        >>> result = rs.fit_glm(claims, X, family="poisson", offset=offset)
        
    weights : array-like, shape (n,), optional
        Prior weights for each observation.
        Use for:
        - Grouped/aggregated data (weight = group size)
        - Known variance differences
        - Importance weighting
        
    max_iter : int, default=25
        Maximum number of IRLS iterations.
        
    tol : float, default=1e-8
        Convergence tolerance. Iteration stops when relative
        change in deviance is less than this.
    
    Returns
    -------
    GLMResults
        Object containing:
        - params: Fitted coefficients
        - fittedvalues: Predicted means
        - deviance: Model deviance
        - converged: Whether fitting converged
        - bse(): Standard errors
        - tvalues(): t/z statistics
        
    Examples
    --------
    >>> import rustystats as rs
    >>> import numpy as np
    >>>
    >>> # Linear regression (Gaussian with identity link)
    >>> n = 100
    >>> X = np.column_stack([np.ones(n), np.random.randn(n)])
    >>> y = 2 + 3 * X[:, 1] + np.random.randn(n) * 0.5
    >>> result = rs.fit_glm(y, X, family="gaussian")
    >>> print(f"Intercept: {result.params[0]:.2f}")  # ≈ 2
    >>> print(f"Slope: {result.params[1]:.2f}")      # ≈ 3
    >>>
    >>> # Poisson regression (claim frequency)
    >>> claims = np.random.poisson(np.exp(0.5 + 0.2 * X[:, 1]))
    >>> result = rs.fit_glm(claims, X, family="poisson")
    >>> print(f"Rate relativity: {np.exp(result.params[1]):.2f}")
    >>>
    >>> # Logistic regression (claim probability)
    >>> prob = 1 / (1 + np.exp(-(0.5 + 0.3 * X[:, 1])))
    >>> claim_flag = np.random.binomial(1, prob)
    >>> result = rs.fit_glm(claim_flag, X, family="binomial")
    >>> print(f"Odds ratio: {np.exp(result.params[1]):.2f}")
    
    Notes
    -----
    **Interpreting Coefficients**
    
    The interpretation depends on the link function:
    
    - **Identity link** (Gaussian default): Coefficients are additive.
      A 1-unit increase in X changes Y by β.
      
    - **Log link** (Poisson/Gamma default): Coefficients are multiplicative.
      A 1-unit increase in X multiplies E[Y] by exp(β).
      This is the "rate relativity" in insurance pricing.
      
    - **Logit link** (Binomial default): Coefficients are log-odds ratios.
      A 1-unit increase in X multiplies the odds by exp(β).
    
    **Don't Forget the Intercept!**
    
    Unlike some libraries, you must explicitly add an intercept column:
    
    >>> X = np.column_stack([np.ones(n), x1, x2, x3])
    
    This gives you more control but is a common source of errors.
    """
    # Ensure arrays are the right type
    y = np.asarray(y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    
    # Validate inputs
    if y.ndim != 1:
        raise ValueError(f"y must be 1-dimensional, got shape {y.shape}")
    if X.ndim != 2:
        raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")
    if len(y) != X.shape[0]:
        raise ValueError(
            f"y has {len(y)} observations but X has {X.shape[0]} rows"
        )
    
    # Convert optional arrays
    if offset is not None:
        offset = np.asarray(offset, dtype=np.float64)
        if offset.shape != y.shape:
            raise ValueError(
                f"offset has shape {offset.shape} but y has shape {y.shape}"
            )
    
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != y.shape:
            raise ValueError(
                f"weights has shape {weights.shape} but y has shape {y.shape}"
            )
    
    # Call the Rust implementation
    return _fit_glm_rust(y, X, family, link, offset, weights, max_iter, tol)


class GLM:
    """
    Generalized Linear Model class (statsmodels-style API).
    
    This provides an object-oriented interface similar to statsmodels.
    
    Parameters
    ----------
    endog : array-like, shape (n,)
        Response variable (y).
        
    exog : array-like, shape (n, p)
        Design matrix (X). Include intercept column if desired.
        
    family : str, default="gaussian"
        Distribution family: "gaussian", "poisson", "binomial", "gamma"
        
    link : str, optional
        Link function. If None, uses canonical link.
        
    offset : array-like, shape (n,), optional
        Offset term (e.g., log(exposure) for rate models).
        
    weights : array-like, shape (n,), optional
        Prior weights for each observation.
    
    Examples
    --------
    >>> import rustystats as rs
    >>> import numpy as np
    >>>
    >>> # Create model
    >>> model = rs.GLM(y, X, family="poisson")
    >>>
    >>> # Fit with default options
    >>> result = model.fit()
    >>>
    >>> # Fit with custom options
    >>> result = model.fit(max_iter=50, tol=1e-10)
    >>>
    >>> # Access results
    >>> print(result.params)
    >>> print(result.deviance)
    >>>
    >>> # With exposure (rate model)
    >>> model = rs.GLM(claims, X, family="poisson", offset=np.log(exposure))
    >>> result = model.fit()
    """
    
    def __init__(
        self,
        endog: np.ndarray,
        exog: np.ndarray,
        family: str = "gaussian",
        link: Optional[str] = None,
        offset: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
    ):
        """Initialize a GLM model."""
        self.endog = np.asarray(endog, dtype=np.float64)
        self.exog = np.asarray(exog, dtype=np.float64)
        self.family = family.lower()
        self.link = link
        self.offset = None if offset is None else np.asarray(offset, dtype=np.float64)
        self.weights = None if weights is None else np.asarray(weights, dtype=np.float64)
        
        # Validate
        if self.endog.ndim != 1:
            raise ValueError(f"endog must be 1D, got shape {self.endog.shape}")
        if self.exog.ndim != 2:
            raise ValueError(f"exog must be 2D, got shape {self.exog.shape}")
        if len(self.endog) != self.exog.shape[0]:
            raise ValueError(
                f"endog has {len(self.endog)} obs but exog has {self.exog.shape[0]} rows"
            )
        if self.offset is not None and self.offset.shape != self.endog.shape:
            raise ValueError(
                f"offset has shape {self.offset.shape} but endog has shape {self.endog.shape}"
            )
        if self.weights is not None and self.weights.shape != self.endog.shape:
            raise ValueError(
                f"weights has shape {self.weights.shape} but endog has shape {self.endog.shape}"
            )
    
    @property
    def nobs(self) -> int:
        """Number of observations."""
        return len(self.endog)
    
    @property
    def df_model(self) -> int:
        """Degrees of freedom for model (number of parameters - 1)."""
        return self.exog.shape[1] - 1
    
    @property
    def df_resid(self) -> int:
        """Degrees of freedom for residuals (n - p)."""
        return self.nobs - self.exog.shape[1]
    
    def fit(
        self,
        max_iter: int = 25,
        tol: float = 1e-8,
    ) -> GLMResults:
        """
        Fit the GLM model.
        
        Parameters
        ----------
        max_iter : int, default=25
            Maximum IRLS iterations.
            
        tol : float, default=1e-8
            Convergence tolerance.
        
        Returns
        -------
        GLMResults
            Fitted model results.
        """
        return fit_glm(
            self.endog,
            self.exog,
            family=self.family,
            link=self.link,
            offset=self.offset,
            weights=self.weights,
            max_iter=max_iter,
            tol=tol,
        )


# =============================================================================
# Prediction Function
# =============================================================================

def predict(
    result: GLMResults,
    X_new: np.ndarray,
    link: str = "log",
    offset: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Make predictions using fitted GLM coefficients.
    
    Parameters
    ----------
    result : GLMResults
        Fitted model results containing coefficients.
        
    X_new : array-like, shape (m, p)
        New design matrix for prediction.
        Must have same number of columns as training X.
        
    link : str, default="log"
        Link function used in the model. Options:
        - "identity": μ = Xβ
        - "log": μ = exp(Xβ)
        - "logit": μ = 1 / (1 + exp(-Xβ))
        
    offset : array-like, shape (m,), optional
        Offset for new predictions (e.g., log(new_exposure)).
        
    Returns
    -------
    np.ndarray
        Predicted values on the response scale (μ).
    
    Examples
    --------
    >>> # Fit model
    >>> result = rs.fit_glm(y_train, X_train, family="poisson")
    >>>
    >>> # Predict on new data
    >>> y_pred = rs.glm.predict(result, X_test, link="log")
    >>>
    >>> # With exposure offset
    >>> y_pred = rs.glm.predict(result, X_test, link="log", 
    ...                         offset=np.log(exposure_test))
    """
    X_new = np.asarray(X_new, dtype=np.float64)
    
    if X_new.ndim == 1:
        X_new = X_new.reshape(1, -1)
    
    n_params = len(result.params)
    if X_new.shape[1] != n_params:
        raise ValueError(
            f"X_new has {X_new.shape[1]} columns but model has {n_params} parameters"
        )
    
    # Compute linear predictor
    eta = X_new @ result.params
    
    # Add offset if provided
    if offset is not None:
        offset = np.asarray(offset, dtype=np.float64)
        if offset.shape[0] != eta.shape[0]:
            raise ValueError(
                f"offset has {offset.shape[0]} elements but X_new has {X_new.shape[0]} rows"
            )
        eta = eta + offset
    
    # Apply inverse link
    if link == "identity":
        return eta
    elif link == "log":
        return np.exp(eta)
    elif link == "logit":
        return 1.0 / (1.0 + np.exp(-eta))
    else:
        raise ValueError(f"Unknown link '{link}'. Use 'identity', 'log', or 'logit'.")


# Convenience aliases for common models
def ols(y: np.ndarray, X: np.ndarray, **kwargs) -> GLMResults:
    """
    Ordinary Least Squares (Gaussian GLM with identity link).
    
    Shortcut for: fit_glm(y, X, family="gaussian", link="identity")
    """
    return fit_glm(y, X, family="gaussian", link="identity", **kwargs)


def poisson_regression(y: np.ndarray, X: np.ndarray, **kwargs) -> GLMResults:
    """
    Poisson Regression (Poisson GLM with log link).
    
    Shortcut for: fit_glm(y, X, family="poisson", link="log")
    
    Common use: Claim frequency modeling
    """
    return fit_glm(y, X, family="poisson", link="log", **kwargs)


def logistic_regression(y: np.ndarray, X: np.ndarray, **kwargs) -> GLMResults:
    """
    Logistic Regression (Binomial GLM with logit link).
    
    Shortcut for: fit_glm(y, X, family="binomial", link="logit")
    
    Common use: Binary classification, claim occurrence
    """
    return fit_glm(y, X, family="binomial", link="logit", **kwargs)


def gamma_regression(y: np.ndarray, X: np.ndarray, **kwargs) -> GLMResults:
    """
    Gamma Regression (Gamma GLM with log link).
    
    Shortcut for: fit_glm(y, X, family="gamma", link="log")
    
    Common use: Claim severity modeling
    """
    return fit_glm(y, X, family="gamma", link="log", **kwargs)


# =============================================================================
# Summary Functions
# =============================================================================

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
    
    Examples
    --------
    >>> result = rs.fit_glm(y, X, family="poisson")
    >>> print(rs.glm.summary(result, feature_names=["intercept", "age", "gender"]))
    
    >>> # For log-link models, show relativities
    >>> print(rs.glm.summary(result))
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
    
    # Coefficient table header
    ci_label = f"{int((1-alpha)*100)}% CI"
    header = f"{'Variable':<12} {'Coef':>10} {'Std.Err':>10} {'z':>8} {'P>|z|':>8} {ci_label:>22} {'':>4}"
    lines.append(header)
    lines.append("-" * 78)
    
    # Coefficient rows
    for i in range(n_params):
        name = feature_names[i][:12]  # Truncate long names
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
        row = f"{name:<12} {coef:>10.4f} {se:>10.4f} {z:>8.3f} {p_str:>8} {ci_str:>22} {sig:>4}"
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
    
    Examples
    --------
    >>> result = rs.fit_glm(claims, X, family="poisson")
    >>> print(rs.glm.summary_relativities(result, 
    ...     feature_names=["Base", "Age 25-35", "Age 35-50", "Age 50+"]))
    
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
