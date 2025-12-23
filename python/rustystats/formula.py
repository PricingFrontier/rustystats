"""
Formula-based API for RustyStats GLM.

This module provides R-style formula support for fitting GLMs with DataFrames.
It uses the `formulaic` library for formula parsing and supports both
Polars and Pandas DataFrames.

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
    import pandas as pd


def _to_pandas(data: Union["pl.DataFrame", "pd.DataFrame"]) -> "pd.DataFrame":
    """Convert Polars DataFrame to Pandas if needed."""
    # Check if it's a Polars DataFrame
    if hasattr(data, "to_pandas"):
        return data.to_pandas()
    return data


def _get_column(
    data: Union["pl.DataFrame", "pd.DataFrame"], 
    column: str
) -> np.ndarray:
    """Extract a column as numpy array from either Polars or Pandas."""
    if hasattr(data, "to_pandas"):
        # Polars
        return data[column].to_numpy()
    else:
        # Pandas
        return data[column].values


def build_design_matrix(
    formula: str,
    data: Union["pl.DataFrame", "pd.DataFrame"],
) -> tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build design matrix and response vector from a formula and DataFrame.
    
    Parameters
    ----------
    formula : str
        R-style formula, e.g., "y ~ x1 + x2 + C(cat_var)"
        - Use `C(var)` to treat a variable as categorical
        - Intercept is included by default
        - Use `0 +` or `-1` to remove intercept
        
    data : DataFrame
        Polars or Pandas DataFrame containing the variables.
        
    Returns
    -------
    y : np.ndarray
        Response variable (n,)
    X : np.ndarray
        Design matrix (n, p) including intercept if specified
    feature_names : list[str]
        Names of the columns in X
        
    Examples
    --------
    >>> y, X, names = build_design_matrix("ClaimNb ~ VehPower + C(Area)", data)
    >>> print(names)
    ['Intercept', 'VehPower', 'Area[T.B]', 'Area[T.C]', ...]
    """
    try:
        from formulaic import model_matrix
    except ImportError:
        raise ImportError(
            "formulaic is required for formula-based API. "
            "Install it with: pip install formulaic"
        )
    
    # Convert to pandas for formulaic compatibility
    pdf = _to_pandas(data)
    
    # Build model matrices
    y_matrix, X_matrix = model_matrix(formula, pdf)
    
    # Convert to numpy
    y = np.asarray(y_matrix).ravel().astype(np.float64)
    X = np.asarray(X_matrix).astype(np.float64)
    
    # Get feature names
    feature_names = list(X_matrix.columns)
    
    return y, X, feature_names


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
        
    data : DataFrame
        Polars or Pandas DataFrame containing the data.
        
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
        data: Union["pl.DataFrame", "pd.DataFrame"],
        family: str = "gaussian",
        link: Optional[str] = None,
        offset: Optional[Union[str, np.ndarray]] = None,
        weights: Optional[Union[str, np.ndarray]] = None,
    ):
        self.formula = formula
        self.data = data
        self.family = family.lower()
        self.link = link
        self._offset_spec = offset
        self._weights_spec = weights
        
        # Build design matrix
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
        max_iter: int = 25,
        tol: float = 1e-8,
    ):
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
        FormulaGLMResults
            Fitted model results with feature names attached.
        """
        from rustystats._rustystats import fit_glm_py as _fit_glm_rust
        
        # Call Rust backend
        result = _fit_glm_rust(
            self.y,
            self.X,
            self.family,
            self.link,
            self.offset,
            self.weights,
            max_iter,
            tol,
        )
        
        # Wrap result with formula metadata
        return FormulaGLMResults(
            result=result,
            feature_names=self.feature_names,
            formula=self.formula,
            family=self.family,
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
    
    def coef_table(self) -> "pd.DataFrame":
        """
        Return coefficients as a DataFrame with names.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: Estimate, Std.Error, z, Pr(>|z|), Signif
        """
        import pandas as pd
        
        return pd.DataFrame({
            "Estimate": self.params,
            "Std.Error": self.bse(),
            "z": self.tvalues(),
            "Pr(>|z|)": self.pvalues(),
            "": self.significance_codes(),
        }, index=self.feature_names)
    
    def relativities(self) -> "pd.DataFrame":
        """
        Return relativities (exp(coef)) for log-link models.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with Relativity and confidence interval columns
        """
        import pandas as pd
        
        if self.link not in ("log",):
            raise ValueError(
                f"Relativities only meaningful for log link, not '{self.link}'"
            )
        
        ci = self.conf_int()
        
        return pd.DataFrame({
            "Relativity": np.exp(self.params),
            "CI_Lower": np.exp(ci[:, 0]),
            "CI_Upper": np.exp(ci[:, 1]),
        }, index=self.feature_names)
    
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
    data: Union["pl.DataFrame", "pd.DataFrame"],
    family: str = "gaussian",
    link: Optional[str] = None,
    offset: Optional[Union[str, np.ndarray]] = None,
    weights: Optional[Union[str, np.ndarray]] = None,
) -> FormulaGLM:
    """
    Create a GLM model from a formula and DataFrame.
    
    This is the main entry point for the formula-based API.
    
    Parameters
    ----------
    formula : str
        R-style formula, e.g., "y ~ x1 + x2 + C(cat_var)"
        
    data : DataFrame
        Polars or Pandas DataFrame containing the variables.
        
    family : str, default="gaussian"
        Distribution family: "gaussian", "poisson", "binomial", "gamma"
        
    link : str, optional
        Link function. If None, uses canonical link.
        
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
    >>> # View results
    >>> print(result.summary())
    >>> print(result.relativities())
    """
    return FormulaGLM(
        formula=formula,
        data=data,
        family=family,
        link=link,
        offset=offset,
        weights=weights,
    )
