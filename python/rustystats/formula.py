"""
Formula-based API for RustyStats GLM.

This module provides R-style formula support for fitting GLMs with DataFrames.
It uses the `formulaic` library for formula parsing and supports Polars DataFrames.

Example
-------
>>> import rustystats as rs
>>> import polars as pl
>>> 
>>> data = pl.read_parquet("insurance_data.parquet")
>>> model = rs.glm(
...     formula="ClaimNb ~ VehPower + VehAge + C(VehBrand)",
...     data=data,
...     family="poisson",
...     offset="Exposure"
... )
>>> result = model.fit()
>>> print(rs.summary(result))
"""

from __future__ import annotations

from typing import Optional, Union, List, TYPE_CHECKING
import numpy as np

# Lazy imports for optional dependencies
if TYPE_CHECKING:
    import polars as pl


def _get_column(data: "pl.DataFrame", column: str) -> np.ndarray:
    """Extract a column as numpy array from Polars DataFrame."""
    return data[column].to_numpy()


# Import build_design_matrix from interactions module (the canonical implementation)
from rustystats.interactions import build_design_matrix


class FormulaGLM:
    """
    GLM model with formula-based specification.
    
    This class provides an R-like interface for fitting GLMs using
    formulas and DataFrames.
    
    Parameters
    ----------
    formula : str
        R-style formula specifying the model.
        Examples:
        - "y ~ x1 + x2": Linear model with intercept
        - "y ~ x1 + C(cat)": Include categorical variable
        - "y ~ 0 + x1 + x2": No intercept
        
    data : pl.DataFrame
        Polars DataFrame containing the data.
        
    family : str, default="gaussian"
        Distribution family: "gaussian", "poisson", "binomial", "gamma"
        
    link : str, optional
        Link function. If None, uses canonical link for family.
        
    offset : str or array-like, optional
        Offset term. Can be:
        - Column name (str): Will extract from data
        - Array: Use directly
        For Poisson family, typically log(exposure).
        
    weights : str or array-like, optional
        Prior weights. Can be column name or array.
        
    Attributes
    ----------
    formula : str
        The formula used
    data : DataFrame
        Original data
    family : str
        Distribution family
    feature_names : list[str]
        Names of features in the design matrix
        
    Examples
    --------
    >>> import rustystats as rs
    >>> import polars as pl
    >>> 
    >>> data = pl.DataFrame({
    ...     "claims": [0, 1, 2, 0, 1],
    ...     "age": [25, 35, 45, 55, 65],
    ...     "exposure": [1.0, 0.5, 1.0, 0.8, 1.0]
    ... })
    >>> 
    >>> model = rs.glm(
    ...     formula="claims ~ age",
    ...     data=data,
    ...     family="poisson",
    ...     offset="exposure"  # Will auto-apply log()
    ... )
    >>> result = model.fit()
    """
    
    def __init__(
        self,
        formula: str,
        data: "pl.DataFrame",
        family: str = "gaussian",
        link: Optional[str] = None,
        var_power: float = 1.5,
        theta: Optional[float] = None,
        offset: Optional[Union[str, np.ndarray]] = None,
        weights: Optional[Union[str, np.ndarray]] = None,
    ):
        self.formula = formula
        self.data = data
        self.family = family.lower()
        self.link = link
        self.var_power = var_power
        self.theta = theta  # None means auto-estimate for negbinomial
        self._offset_spec = offset
        self._weights_spec = weights
        
        # Build design matrix (uses optimized backend for interactions)
        self.y, self.X, self.feature_names = build_design_matrix(formula, data)
        self.n_obs = len(self.y)
        self.n_params = self.X.shape[1]
        
        # Process offset
        self.offset = self._process_offset(offset)
        
        # Process weights
        self.weights = self._process_weights(weights)
    
    def _process_offset(
        self, 
        offset: Optional[Union[str, np.ndarray]]
    ) -> Optional[np.ndarray]:
        """Process offset specification."""
        if offset is None:
            return None
            
        if isinstance(offset, str):
            # It's a column name
            offset_values = _get_column(self.data, offset)
            
            # For Poisson/Gamma with log link, auto-apply log to exposure
            if self.family in ("poisson", "gamma") and self.link in (None, "log"):
                # Check if values look like exposure (positive, not already logged)
                if np.all(offset_values > 0) and np.mean(offset_values) > 0.01:
                    offset_values = np.log(offset_values)
            
            return offset_values.astype(np.float64)
        else:
            return np.asarray(offset, dtype=np.float64)
    
    def _process_weights(
        self, 
        weights: Optional[Union[str, np.ndarray]]
    ) -> Optional[np.ndarray]:
        """Process weights specification."""
        if weights is None:
            return None
            
        if isinstance(weights, str):
            return _get_column(self.data, weights).astype(np.float64)
        else:
            return np.asarray(weights, dtype=np.float64)
    
    @property
    def df_model(self) -> int:
        """Degrees of freedom for model (number of parameters - 1)."""
        return self.n_params - 1
    
    @property
    def df_resid(self) -> int:
        """Degrees of freedom for residuals (n - p)."""
        return self.n_obs - self.n_params
    
    def fit(
        self,
        alpha: float = 0.0,
        l1_ratio: float = 0.0,
        max_iter: int = 25,
        tol: float = 1e-8,
    ):
        """
        Fit the GLM model, optionally with regularization.
        
        Parameters
        ----------
        alpha : float, default=0.0
            Regularization strength. Higher values = more shrinkage.
            - alpha=0: No regularization (standard GLM)
            - alpha>0: Regularized GLM
            
        l1_ratio : float, default=0.0
            Elastic Net mixing parameter:
            - l1_ratio=0.0: Ridge (L2) penalty
            - l1_ratio=1.0: Lasso (L1) penalty - performs variable selection
            - 0 < l1_ratio < 1: Elastic Net
            
        max_iter : int, default=25
            Maximum IRLS iterations.
        tol : float, default=1e-8
            Convergence tolerance.
            
        Returns
        -------
        FormulaGLMResults
            Fitted model results with feature names attached.
            
        Examples
        --------
        >>> # Standard GLM
        >>> result = model.fit()
        
        >>> # Ridge regularization
        >>> result = model.fit(alpha=0.1, l1_ratio=0.0)
        
        >>> # Lasso for variable selection
        >>> result = model.fit(alpha=0.1, l1_ratio=1.0)
        """
        from rustystats._rustystats import fit_glm_py as _fit_glm_rust, fit_negbinomial_py as _fit_negbinomial_rust
        
        # Check if we need auto theta estimation for negbinomial
        is_negbinomial = self.family in ("negbinomial", "negativebinomial", "negative_binomial", "neg-binomial", "nb")
        auto_theta = is_negbinomial and self.theta is None
        
        if auto_theta:
            # Use profile likelihood to auto-estimate theta
            result = _fit_negbinomial_rust(
                self.y,
                self.X,
                self.link,
                None,  # init_theta (use method-of-moments)
                1e-5,  # theta_tol
                10,    # max_theta_iter
                self.offset,
                self.weights,
                max_iter,
                tol,
            )
            result_family = result.family  # Contains estimated theta
        else:
            # Use fixed theta (default 1.0 for negbinomial if not auto)
            theta = self.theta if self.theta is not None else 1.0
            result = _fit_glm_rust(
                self.y,
                self.X,
                self.family,
                self.link,
                self.var_power,
                theta,
                self.offset,
                self.weights,
                alpha,
                l1_ratio,
                max_iter,
                tol,
            )
            result_family = self.family
        
        # Wrap result with formula metadata
        return FormulaGLMResults(
            result=result,
            feature_names=self.feature_names,
            formula=self.formula,
            family=result_family,
            link=self.link,
        )


class FormulaGLMResults:
    """
    Results from a formula-based GLM fit.
    
    This wraps the base GLMResults and adds formula-specific functionality
    like named coefficients and automatic summary formatting.
    
    Attributes
    ----------
    params : np.ndarray
        Fitted coefficients
    feature_names : list[str]
        Names corresponding to each coefficient
    formula : str
        The formula used to fit the model
    """
    
    def __init__(
        self,
        result,
        feature_names: List[str],
        formula: str,
        family: str,
        link: Optional[str],
    ):
        self._result = result
        self.feature_names = feature_names
        self.formula = formula
        self.family = family
        self.link = link or self._default_link(family)
    
    @staticmethod
    def _default_link(family: str) -> str:
        """Get default link for family."""
        return {
            "gaussian": "identity",
            "poisson": "log",
            "binomial": "logit",
            "gamma": "log",
        }.get(family, "identity")
    
    # Delegate to underlying result
    @property
    def params(self) -> np.ndarray:
        """Fitted coefficients."""
        return self._result.params
    
    @property
    def fittedvalues(self) -> np.ndarray:
        """Fitted values (predicted means)."""
        return self._result.fittedvalues
    
    @property
    def linear_predictor(self) -> np.ndarray:
        """Linear predictor (eta = X @ beta)."""
        return self._result.linear_predictor
    
    @property
    def deviance(self) -> float:
        """Model deviance."""
        return self._result.deviance
    
    @property
    def converged(self) -> bool:
        """Whether IRLS converged."""
        return self._result.converged
    
    @property
    def iterations(self) -> int:
        """Number of IRLS iterations."""
        return self._result.iterations
    
    def bse(self) -> np.ndarray:
        """Standard errors of coefficients."""
        return self._result.bse()
    
    def tvalues(self) -> np.ndarray:
        """z/t statistics."""
        return self._result.tvalues()
    
    def pvalues(self) -> np.ndarray:
        """P-values for coefficients."""
        return self._result.pvalues()
    
    def conf_int(self, alpha: float = 0.05) -> np.ndarray:
        """Confidence intervals."""
        return self._result.conf_int(alpha)
    
    def significance_codes(self) -> List[str]:
        """Significance codes."""
        return self._result.significance_codes()
    
    # Robust standard errors (sandwich estimators)
    def bse_robust(self, cov_type: str = "HC1") -> np.ndarray:
        """Robust standard errors of coefficients (HC/sandwich estimator).
        
        Unlike model-based standard errors that assume correct variance
        specification, robust standard errors are valid under heteroscedasticity.
        
        Parameters
        ----------
        cov_type : str, optional
            Type of robust covariance. Options:
            - "HC0": No small-sample correction
            - "HC1": Degrees of freedom correction (default, recommended)
            - "HC2": Leverage-adjusted
            - "HC3": Jackknife-like (most conservative)
        
        Returns
        -------
        numpy.ndarray
            Array of robust standard errors, one for each coefficient.
        """
        return self._result.bse_robust(cov_type)
    
    def tvalues_robust(self, cov_type: str = "HC1") -> np.ndarray:
        """z/t statistics using robust standard errors.
        
        Parameters
        ----------
        cov_type : str, optional
            Type of robust covariance. Default "HC1".
        
        Returns
        -------
        numpy.ndarray
            Array of t/z statistics (coefficient / robust SE).
        """
        return self._result.tvalues_robust(cov_type)
    
    def pvalues_robust(self, cov_type: str = "HC1") -> np.ndarray:
        """P-values using robust standard errors.
        
        Parameters
        ----------
        cov_type : str, optional
            Type of robust covariance. Default "HC1".
        
        Returns
        -------
        numpy.ndarray
            Array of p-values.
        """
        return self._result.pvalues_robust(cov_type)
    
    def conf_int_robust(self, alpha: float = 0.05, cov_type: str = "HC1") -> np.ndarray:
        """Confidence intervals using robust standard errors.
        
        Parameters
        ----------
        alpha : float, optional
            Significance level. Default 0.05 gives 95% CI.
        cov_type : str, optional
            Type of robust covariance. Default "HC1".
        
        Returns
        -------
        numpy.ndarray
            2D array of shape (n_params, 2) with [lower, upper] bounds.
        """
        return self._result.conf_int_robust(alpha, cov_type)
    
    def cov_robust(self, cov_type: str = "HC1") -> np.ndarray:
        """Robust covariance matrix (HC/sandwich estimator).
        
        Parameters
        ----------
        cov_type : str, optional
            Type of robust covariance. Default "HC1".
        
        Returns
        -------
        numpy.ndarray
            Robust covariance matrix (p × p).
        """
        return self._result.cov_robust(cov_type)
    
    # Diagnostic methods (statsmodels-compatible)
    def resid_response(self) -> np.ndarray:
        """Response residuals: y - μ."""
        return self._result.resid_response()
    
    def resid_pearson(self) -> np.ndarray:
        """Pearson residuals: (y - μ) / √V(μ)."""
        return self._result.resid_pearson()
    
    def resid_deviance(self) -> np.ndarray:
        """Deviance residuals: sign(y - μ) × √d_i."""
        return self._result.resid_deviance()
    
    def resid_working(self) -> np.ndarray:
        """Working residuals: (y - μ) × g'(μ)."""
        return self._result.resid_working()
    
    def llf(self) -> float:
        """Log-likelihood of the fitted model."""
        return self._result.llf()
    
    def aic(self) -> float:
        """Akaike Information Criterion."""
        return self._result.aic()
    
    def bic(self) -> float:
        """Bayesian Information Criterion."""
        return self._result.bic()
    
    def null_deviance(self) -> float:
        """Deviance of intercept-only model."""
        return self._result.null_deviance()
    
    def pearson_chi2(self) -> float:
        """Pearson chi-squared statistic."""
        return self._result.pearson_chi2()
    
    def scale(self) -> float:
        """Estimated dispersion parameter (deviance-based)."""
        return self._result.scale()
    
    def scale_pearson(self) -> float:
        """Estimated dispersion parameter (Pearson-based)."""
        return self._result.scale_pearson()
    
    # Regularization properties
    @property
    def alpha(self) -> float:
        """Regularization strength (lambda)."""
        return self._result.alpha
    
    @property
    def l1_ratio(self):
        """L1 ratio for Elastic Net (1.0=Lasso, 0.0=Ridge)."""
        return self._result.l1_ratio
    
    @property
    def is_regularized(self) -> bool:
        """Whether this is a regularized model."""
        return self._result.is_regularized
    
    @property
    def penalty_type(self) -> str:
        """Type of penalty: 'none', 'ridge', 'lasso', or 'elasticnet'."""
        return self._result.penalty_type
    
    def n_nonzero(self) -> int:
        """Number of non-zero coefficients (excluding intercept)."""
        return self._result.n_nonzero()
    
    def selected_features(self) -> List[str]:
        """
        Get names of features with non-zero coefficients.
        
        Useful for Lasso/Elastic Net to see which variables were selected.
        """
        indices = self._result.selected_features()
        return [self.feature_names[i] for i in indices]
    
    @property
    def nobs(self) -> int:
        """Number of observations."""
        return self._result.nobs
    
    @property
    def df_resid(self) -> int:
        """Residual degrees of freedom."""
        return self._result.df_resid
    
    @property
    def df_model(self) -> int:
        """Model degrees of freedom."""
        return self._result.df_model
    
    def coef_table(self) -> "pl.DataFrame":
        """
        Return coefficients as a DataFrame with names.
        
        Returns
        -------
        pl.DataFrame
            DataFrame with columns: Feature, Estimate, Std.Error, z, Pr(>|z|), Signif
        """
        import polars as pl
        
        return pl.DataFrame({
            "Feature": self.feature_names,
            "Estimate": self.params,
            "Std.Error": self.bse(),
            "z": self.tvalues(),
            "Pr(>|z|)": self.pvalues(),
            "Signif": self.significance_codes(),
        })
    
    def relativities(self) -> "pl.DataFrame":
        """
        Return relativities (exp(coef)) for log-link models.
        
        Returns
        -------
        pl.DataFrame
            DataFrame with Feature, Relativity and confidence interval columns
        """
        import polars as pl
        
        if self.link not in ("log",):
            raise ValueError(
                f"Relativities only meaningful for log link, not '{self.link}'"
            )
        
        ci = self.conf_int()
        
        return pl.DataFrame({
            "Feature": self.feature_names,
            "Relativity": np.exp(self.params),
            "CI_Lower": np.exp(ci[:, 0]),
            "CI_Upper": np.exp(ci[:, 1]),
        })
    
    def summary(self) -> str:
        """
        Generate a formatted summary string.
        
        Returns
        -------
        str
            Formatted summary table
        """
        from rustystats.glm import summary
        return summary(self._result, feature_names=self.feature_names)
    
    def __repr__(self) -> str:
        return (
            f"<FormulaGLMResults: {self.family} family, "
            f"{len(self.params)} parameters, "
            f"deviance={self.deviance:.2f}>"
        )


def glm(
    formula: str,
    data: "pl.DataFrame",
    family: str = "gaussian",
    link: Optional[str] = None,
    var_power: float = 1.5,
    theta: Optional[float] = None,
    offset: Optional[Union[str, np.ndarray]] = None,
    weights: Optional[Union[str, np.ndarray]] = None,
) -> FormulaGLM:
    """
    Create a GLM model from a formula and DataFrame.
    
    This is the main entry point for the formula-based API.
    
    Parameters
    ----------
    formula : str
        R-style formula specifying the model.
        
        Supported syntax:
        - Main effects: ``x1``, ``x2``, ``C(cat)`` (categorical)
        - Two-way interactions: ``x1:x2`` (interaction only), ``x1*x2`` (main effects + interaction)
        - Categorical interactions: ``C(cat1)*C(cat2)``, ``C(cat):x``
        - Higher-order: ``x1:x2:x3``
        - Splines: ``bs(x, df=5)``, ``ns(x, df=4)``
        - Intercept: included by default, use ``0 +`` or ``- 1`` to remove
        
    data : pl.DataFrame
        Polars DataFrame containing the variables.
        
    family : str, default="gaussian"
        Distribution family: "gaussian", "poisson", "binomial", "gamma", "tweedie",
        "quasipoisson", "quasibinomial", or "negbinomial"
        
    link : str, optional
        Link function. If None, uses canonical link.
        
    var_power : float, default=1.5
        Variance power for Tweedie family (ignored for others).
        
    theta : float, optional
        Dispersion parameter for Negative Binomial family (ignored for others).
        If None (default), theta is automatically estimated using profile likelihood.
        
    offset : str or array-like, optional
        Offset term. If string, treated as column name.
        For Poisson, log() is auto-applied to exposure columns.
        
    weights : str or array-like, optional
        Prior weights. If string, treated as column name.
        
    Returns
    -------
    FormulaGLM
        Model object. Call .fit() to fit the model.
        
    Examples
    --------
    >>> import rustystats as rs
    >>> import polars as pl
    >>> 
    >>> # Load data
    >>> data = pl.read_parquet("insurance.parquet")
    >>> 
    >>> # Fit Poisson model for claim frequency
    >>> model = rs.glm(
    ...     formula="ClaimNb ~ VehPower + VehAge + C(VehBrand) + C(Area)",
    ...     data=data,
    ...     family="poisson",
    ...     offset="Exposure"
    ... )
    >>> result = model.fit()
    >>> 
    >>> # Model with interactions
    >>> model = rs.glm(
    ...     formula="ClaimNb ~ VehPower*VehAge + C(Area):DrivAge",
    ...     data=data,
    ...     family="poisson",
    ...     offset="Exposure"
    ... )
    >>> result = model.fit()
    >>> print(result.summary())
    """
    return FormulaGLM(
        formula=formula,
        data=data,
        family=family,
        link=link,
        var_power=var_power,
        theta=theta,
        offset=offset,
        weights=weights,
    )
