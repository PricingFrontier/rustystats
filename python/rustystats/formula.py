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
    from rustystats.regularization_path import RegularizationPathInfo


def _get_column(data: "pl.DataFrame", column: str) -> np.ndarray:
    """Extract a column as numpy array from Polars DataFrame."""
    return data[column].to_numpy()


# Import from interactions module (the canonical implementation)
from rustystats.interactions import build_design_matrix, InteractionBuilder


def _get_constraint_indices(feature_names: List[str]) -> tuple:
    """
    Compute coefficient constraint indices from feature names.
    
    Returns
    -------
    nonneg_indices : list[int]
        Indices of coefficients that must be non-negative (β ≥ 0)
    nonpos_indices : list[int]
        Indices of coefficients that must be non-positive (β ≤ 0)
    """
    # ms()/ns() with + and pos() terms require non-negative coefficients
    nonneg_indices = [
        i for i, name in enumerate(feature_names)
        if name.startswith("pos(") or 
        (name.startswith("ms(") and ", +)" in name) or
        (name.startswith("ns(") and ", +)" in name)
    ]
    # ms()/ns() with - and neg() terms require non-positive coefficients
    nonpos_indices = [
        i for i, name in enumerate(feature_names)
        if name.startswith("neg(") or
        (name.startswith("ms(") and ", -)" in name) or
        (name.startswith("ns(") and ", -)" in name)
    ]
    return nonneg_indices, nonpos_indices


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
        seed: Optional[int] = None,
    ):
        self.formula = formula
        self.data = data
        self.family = family.lower()
        self.link = link
        self.var_power = var_power
        self.theta = theta  # None means auto-estimate for negbinomial
        self._offset_spec = offset
        self._weights_spec = weights
        self._seed = seed
        
        # Extract raw exposure for target encoding BEFORE building design matrix
        # For frequency models with log link, offset is typically log(exposure)
        # but target encoding needs raw exposure to compute claim rates
        raw_exposure = self._get_raw_exposure(offset)
        
        # Build design matrix (uses optimized backend for interactions)
        # Pass raw exposure so target encoding can use rate (y/exposure) instead of raw y
        # Pass seed for deterministic target encoding
        self._builder = InteractionBuilder(data)
        self.y, self.X, self.feature_names = self._builder.build_design_matrix(
            formula, exposure=raw_exposure, seed=seed
        )
        self.n_obs = len(self.y)
        self.n_params = self.X.shape[1]
        
        # Store validation results (computed lazily)
        self._validation_results = None
        
        # Process offset (applies log for Poisson/Gamma families)
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
            
            # For Poisson/Gamma/QuasiPoisson/NegBinomial with log link, auto-apply log to exposure
            if self.family in ("poisson", "quasipoisson", "negbinomial", "gamma") and self.link in (None, "log"):
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
    
    def _get_raw_exposure(
        self,
        offset: Optional[Union[str, np.ndarray]]
    ) -> Optional[np.ndarray]:
        """
        Get raw exposure values for target encoding.
        
        For frequency models (Poisson, NegBinomial, etc.), the offset is typically
        log(exposure). However, target encoding needs the raw exposure values
        to compute claim rates (claims/exposure) instead of raw claim counts.
        
        This method extracts the raw exposure BEFORE log transformation.
        """
        if offset is None:
            return None
        
        if isinstance(offset, str):
            # It's a column name - extract raw values
            return _get_column(self.data, offset).astype(np.float64)
        else:
            # It's an array - assume it's raw exposure values
            # (if user passed log(exposure), they'll get log-rate encoding which is also valid)
            return np.asarray(offset, dtype=np.float64)
    
    @property
    def df_model(self) -> int:
        """Degrees of freedom for model (number of parameters - 1)."""
        return self.n_params - 1
    
    @property
    def df_resid(self) -> int:
        """Degrees of freedom for residuals (n - p)."""
        return self.n_obs - self.n_params
    
    def validate(self, verbose: bool = True) -> dict:
        """
        Validate the design matrix before fitting.
        
        Checks for common issues that cause fitting failures:
        - Rank deficiency (linearly dependent columns)
        - High multicollinearity
        - Zero variance columns
        - NaN/Inf values
        
        Parameters
        ----------
        verbose : bool, default=True
            Print diagnostic messages with fix suggestions.
            
        Returns
        -------
        dict
            Validation results including 'valid' (bool) and 'suggestions' (list).
            
        Examples
        --------
        >>> model = rs.glm("y ~ ns(x, df=4) + C(cat)", data, family="poisson")
        >>> results = model.validate()
        >>> if not results['valid']:
        ...     print("Issues found:", results['suggestions'])
        """
        self._validation_results = self._builder.validate_design_matrix(
            self.X, self.feature_names, verbose=verbose
        )
        return self._validation_results
    
    def explore(
        self,
        categorical_factors: Optional[List[str]] = None,
        continuous_factors: Optional[List[str]] = None,
        n_bins: int = 10,
        rare_threshold_pct: float = 1.0,
        max_categorical_levels: int = 20,
        detect_interactions: bool = True,
        max_interaction_factors: int = 10,
    ):
        """
        Explore data before fitting the model.
        
        This provides pre-fit analysis including factor statistics and
        interaction detection based on the response variable.
        
        Parameters
        ----------
        categorical_factors : list of str, optional
            Names of categorical factors to analyze.
        continuous_factors : list of str, optional
            Names of continuous factors to analyze.
        n_bins : int, default=10
            Number of bins for continuous factors.
        rare_threshold_pct : float, default=1.0
            Threshold (%) below which categorical levels are grouped.
        max_categorical_levels : int, default=20
            Maximum categorical levels to show.
        detect_interactions : bool, default=True
            Whether to detect potential interactions.
        max_interaction_factors : int, default=10
            Maximum factors for interaction detection.
        
        Returns
        -------
        DataExploration
            Pre-fit exploration results with to_json() method.
        
        Examples
        --------
        >>> model = rs.glm("ClaimNb ~ Age + C(Region)", data, family="poisson")
        >>> 
        >>> # Explore before fitting
        >>> exploration = model.explore(
        ...     categorical_factors=["Region", "VehBrand"],
        ...     continuous_factors=["Age", "VehPower"],
        ... )
        >>> print(exploration.to_json())
        >>> 
        >>> # Then fit
        >>> result = model.fit()
        """
        from rustystats.diagnostics import explore_data
        
        # Parse formula to get response column name
        response = self.formula.split("~")[0].strip()
        
        # Get exposure column if set
        exposure_col = None
        if isinstance(self._offset_spec, str):
            exposure_col = self._offset_spec
        
        return explore_data(
            data=self.data,
            response=response,
            categorical_factors=categorical_factors,
            continuous_factors=continuous_factors,
            exposure=exposure_col,
            family=self.family,
            n_bins=n_bins,
            rare_threshold_pct=rare_threshold_pct,
            max_categorical_levels=max_categorical_levels,
            detect_interactions=detect_interactions,
            max_interaction_factors=max_interaction_factors,
        )
    
    def _fit_negbinomial_profile(
        self,
        X: np.ndarray,
        alpha: float,
        l1_ratio: float,
        max_iter: int,
        tol: float,
        theta_tol: float = 1e-5,
        max_theta_iter: int = 10,
    ) -> tuple:
        """
        Fit negative binomial GLM with moment-based theta estimation.
        
        Applies minimum ridge regularization (alpha >= 1e-6) for numerical
        stability when fitting negative binomial models.
        
        Parameters
        ----------
        X : np.ndarray
            Design matrix
        alpha : float
            User-specified regularization (will be at least 1e-6)
        l1_ratio : float
            Elastic net mixing parameter
        max_iter : int
            Maximum IRLS iterations per fit
        tol : float
            IRLS convergence tolerance
        theta_tol : float
            Convergence tolerance for theta estimation
        max_theta_iter : int
            Maximum iterations for theta estimation
            
        Returns
        -------
        tuple
            (GLMResults, family_string) where family_string includes theta
        """
        from rustystats._rustystats import fit_glm_py as _fit_glm_rust
        
        # Initial Poisson fit to get starting mu values
        poisson_result = _fit_glm_rust(
            self.y, X, "poisson", self.link, 1.5, 1.0,
            self.offset, self.weights, alpha, l1_ratio, max_iter, tol
        )
        mu = poisson_result.fittedvalues
        
        # Estimate initial theta from method of moments
        # Var(Y) = mu + mu^2/theta => theta = mu^2 / (Var(Y) - mu)
        y_arr = np.asarray(self.y)
        residuals = y_arr - mu
        var_estimate = np.mean(residuals**2)
        mean_mu = np.mean(mu)
        theta = max(0.01, min(1000.0, mean_mu**2 / max(var_estimate - mean_mu, 0.01)))
        
        # Profile likelihood iteration with minimum ridge for stability
        effective_alpha = max(alpha, 1e-6)
        
        coefficients = poisson_result.params
        result = poisson_result  # Fallback if all iterations fail
        
        for _ in range(max_theta_iter):
            result = _fit_glm_rust(
                self.y, X, "negbinomial", self.link, 1.5, theta,
                self.offset, self.weights, effective_alpha, l1_ratio, max_iter, tol
            )
            
            # If NaN, increase regularization and retry
            if np.any(np.isnan(result.params)):
                effective_alpha *= 10
                if effective_alpha > 1.0:
                    raise ValueError(
                        "Negative binomial fitting failed due to numerical instability. "
                        "Try simplifying the model or using Poisson instead."
                    )
                continue
            
            coefficients = result.params
            mu = result.fittedvalues
            
            # Moment-based theta update
            residuals = y_arr - mu
            excess_var = np.mean(residuals**2) - np.mean(mu)
            if excess_var > 0:
                new_theta = np.mean(mu)**2 / excess_var
                new_theta = max(0.01, min(1000.0, new_theta))
            else:
                new_theta = 1000.0  # No overdispersion
            
            if abs(new_theta - theta) < theta_tol:
                theta = new_theta
                break
            theta = new_theta
        
        # Final fit with converged theta
        final_result = _fit_glm_rust(
            self.y, X, "negbinomial", self.link, 1.5, theta,
            self.offset, self.weights, max(alpha, 1e-6), l1_ratio, max_iter, tol
        )
        
        # Fall back to iteration result if final has NaN
        if np.any(np.isnan(final_result.params)) and not np.any(np.isnan(coefficients)):
            final_result = result
        
        return final_result, f"NegativeBinomial(theta={theta:.4f})"
    
    def fit(
        self,
        alpha: float = 0.0,
        l1_ratio: float = 0.0,
        max_iter: int = 25,
        tol: float = 1e-8,
        # Cross-validation based regularization path parameters
        cv: Optional[int] = None,
        selection: str = "min",
        regularization: Optional[str] = None,
        n_alphas: int = 20,
        alpha_min_ratio: float = 0.0001,
        cv_seed: Optional[int] = None,
        include_unregularized: bool = True,
        verbose: bool = False,
    ):
        """
        Fit the GLM model, optionally with regularization.
        
        Parameters
        ----------
        alpha : float, default=0.0
            Regularization strength. Higher values = more shrinkage.
            - alpha=0: No regularization (standard GLM)
            - alpha>0: Regularized GLM
            Ignored if cv is specified.
            
        l1_ratio : float, default=0.0
            Elastic Net mixing parameter:
            - l1_ratio=0.0: Ridge (L2) penalty
            - l1_ratio=1.0: Lasso (L1) penalty - performs variable selection
            - 0 < l1_ratio < 1: Elastic Net
            Ignored if cv is specified with regularization type.
            
        max_iter : int, default=25
            Maximum IRLS iterations.
        tol : float, default=1e-8
            Convergence tolerance.
            
        cv : int, optional
            Number of cross-validation folds for regularization path.
            If specified, fits a path of alpha values and selects optimal via CV.
            Requires `regularization` to be set.
            
        selection : str, default="min"
            Selection method for CV-based regularization:
            - "min": Select alpha with minimum CV deviance
            - "1se": Select largest alpha within 1 SE of minimum (more conservative)
            
        regularization : str, optional
            Type of regularization for CV path: "ridge", "lasso", or "elastic_net".
            Required when cv is specified.
            
        n_alphas : int, default=20
            Number of alpha values to try in regularization path.
            
        alpha_min_ratio : float, default=0.001
            Smallest alpha as ratio of alpha_max.
            
        cv_seed : int, optional
            Random seed for CV fold creation.
            
        include_unregularized : bool, default=True
            Include unregularized model (alpha=0) in CV comparison.
            
        verbose : bool, default=False
            Print progress during CV fitting.
            
        Returns
        -------
        FormulaGLMResults
            Fitted model results with feature names attached.
            When cv is used, includes additional attributes:
            - cv_deviance: CV deviance at selected alpha
            - cv_deviance_se: Standard error of CV deviance
            - regularization_path: Full path results
            
        Examples
        --------
        >>> # Standard GLM
        >>> result = model.fit()
        
        >>> # Ridge regularization with explicit alpha
        >>> result = model.fit(alpha=0.1, l1_ratio=0.0)
        
        >>> # Lasso for variable selection
        >>> result = model.fit(alpha=0.1, l1_ratio=1.0)
        
        >>> # CV-based regularization selection (recommended)
        >>> result = model.fit(cv=5, regularization="ridge", selection="1se")
        >>> print(f"Selected alpha: {result.alpha}")
        >>> print(f"CV deviance: {result.cv_deviance}")
        """
        from rustystats._rustystats import fit_glm_py as _fit_glm_rust, fit_negbinomial_py as _fit_negbinomial_rust
        
        # Check if we need auto theta estimation for negbinomial
        is_negbinomial = self.family in ("negbinomial", "negativebinomial", "negative_binomial", "neg-binomial", "nb")
        auto_theta = is_negbinomial and self.theta is None
        
        # Handle CV-based regularization path
        # If regularization is specified without cv, default to cv=5
        if regularization is not None and cv is None:
            cv = 5
        
        if cv is not None:
            if regularization is None:
                raise ValueError(
                    "When cv is specified, 'regularization' must be set to 'ridge', 'lasso', or 'elastic_net'"
                )
            
            from rustystats.regularization_path import fit_cv_regularization_path
            
            # For negbinomial with auto theta, first estimate theta then do CV
            if auto_theta:
                # Quick fit to estimate theta
                _, result_family = self._fit_negbinomial_profile(
                    self.X, 0.0, 0.0, max_iter, tol
                )
                # Extract theta from family string
                theta_start = result_family.find("theta=") + 6
                theta_end = result_family.find(")", theta_start)
                estimated_theta = float(result_family[theta_start:theta_end])
                self.theta = estimated_theta  # Store for CV fits
            
            # Determine l1_ratio from regularization type
            if regularization == "ridge":
                cv_l1_ratio = 0.0
            elif regularization == "lasso":
                cv_l1_ratio = 1.0
            elif regularization == "elastic_net":
                cv_l1_ratio = l1_ratio if l1_ratio > 0 else 0.5
            else:
                raise ValueError(f"Unknown regularization type: {regularization}")
            
            # Fit regularization path with CV
            path_info = fit_cv_regularization_path(
                glm_instance=self,
                cv=cv,
                selection=selection,
                regularization=regularization,
                n_alphas=n_alphas,
                alpha_min_ratio=alpha_min_ratio,
                l1_ratio=cv_l1_ratio,
                max_iter=max_iter,
                tol=tol,
                seed=cv_seed,
                include_unregularized=include_unregularized,
                verbose=verbose,
            )
            
            # Use selected alpha for final fit
            alpha = path_info.selected_alpha
            l1_ratio = path_info.selected_l1_ratio
            
            if verbose:
                print(f"\nRefitting on full data with alpha={alpha:.6f}")
        else:
            path_info = None
        
        # For negbinomial with auto theta, use Python-side profile likelihood
        # This allows regularization to be applied for numerical stability
        try:
            if auto_theta:
                result, result_family = self._fit_negbinomial_profile(
                    self.X, alpha, l1_ratio, max_iter, tol
                )
            else:
                # Use fixed theta (default 1.0 for negbinomial if not auto)
                theta = self.theta if self.theta is not None else 1.0
                
                # Compute coefficient constraint indices
                nonneg_indices, nonpos_indices = _get_constraint_indices(self.feature_names)
                
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
                    nonneg_indices if nonneg_indices else None,
                    nonpos_indices if nonpos_indices else None,
                )
                # Include theta in family string for negbinomial so deviance calculations use correct theta
                if is_negbinomial:
                    result_family = f"NegativeBinomial(theta={theta:.4f})"
                else:
                    result_family = self.family
        except ValueError as e:
            if "singular" in str(e).lower() or "multicollinearity" in str(e).lower() or "nan" in str(e).lower():
                # Run validation to provide helpful diagnostics
                print("\n" + "=" * 60)
                print("MODEL FITTING FAILED - Running diagnostics...")
                print("=" * 60)
                validation = self.validate(verbose=True)
                raise ValueError(
                    f"GLM fitting failed due to design matrix issues. "
                    f"See diagnostics above for specific problems and fixes.\n"
                    f"You can also run model.validate() before fit() to check for issues.\n"
                    f"Original error: {e}"
                ) from None
            else:
                raise
        
        # Wrap result with formula metadata
        return FormulaGLMResults(
            result=result,
            feature_names=self.feature_names,
            formula=self.formula,
            family=result_family,
            link=self.link,
            builder=self._builder,
            design_matrix=self.X,  # Pass design matrix for VIF calculation
            offset_spec=self._offset_spec,
            offset_is_exposure=(self.family in ("poisson", "quasipoisson", "negbinomial", "gamma") and self.link in (None, "log")),
            regularization_path_info=path_info,
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
        builder: Optional["InteractionBuilder"] = None,
        design_matrix: Optional[np.ndarray] = None,
        offset_spec: Optional[Union[str, np.ndarray]] = None,
        offset_is_exposure: bool = False,
        regularization_path_info: Optional["RegularizationPathInfo"] = None,
    ):
        self._result = result
        self.feature_names = feature_names
        self.formula = formula
        self.family = family
        self._regularization_path_info = regularization_path_info
        self.link = link or self._default_link(family)
        self._builder = builder
        self._design_matrix = design_matrix  # Store for VIF calculation
        self._offset_spec = offset_spec
        self._offset_is_exposure = offset_is_exposure
    
    @staticmethod
    def _default_link(family: str) -> str:
        """Get default link for family."""
        # Handle NegativeBinomial(theta=...) format
        family_lower = family.lower()
        if family_lower.startswith("negativebinomial"):
            return "log"
        return {
            "gaussian": "identity",
            "poisson": "log",
            "quasipoisson": "log",
            "negbinomial": "log",
            "binomial": "logit",
            "gamma": "log",
            "inversegaussian": "inverse",
            "tweedie": "log",
        }.get(family_lower, "identity")
    
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
    
    # CV-based regularization path properties
    @property
    def cv_deviance(self) -> Optional[float]:
        """CV deviance at selected alpha (only available when fit with cv=)."""
        if self._regularization_path_info is None:
            return None
        return self._regularization_path_info.cv_deviance
    
    @property
    def cv_deviance_se(self) -> Optional[float]:
        """Standard error of CV deviance (only available when fit with cv=)."""
        if self._regularization_path_info is None:
            return None
        return self._regularization_path_info.cv_deviance_se
    
    @property
    def regularization_type(self) -> Optional[str]:
        """Type of regularization: 'ridge', 'lasso', 'elastic_net', or 'none'."""
        if self._regularization_path_info is None:
            # Fall back to penalty_type from underlying result
            return self.penalty_type
        return self._regularization_path_info.regularization_type
    
    @property
    def regularization_path(self) -> Optional[List[dict]]:
        """
        Full regularization path results (only available when fit with cv=).
        
        Returns list of dicts with keys: alpha, l1_ratio, cv_deviance_mean, 
        cv_deviance_se, n_nonzero, max_coef.
        """
        if self._regularization_path_info is None:
            return None
        return [
            {
                "alpha": r.alpha,
                "l1_ratio": r.l1_ratio,
                "cv_deviance_mean": r.cv_deviance_mean,
                "cv_deviance_se": r.cv_deviance_se,
                "n_nonzero": r.n_nonzero,
                "max_coef": r.max_coef,
            }
            for r in self._regularization_path_info.path
        ]
    
    @property
    def cv_selection_method(self) -> Optional[str]:
        """Selection method used: 'min' or '1se' (only available when fit with cv=)."""
        if self._regularization_path_info is None:
            return None
        return self._regularization_path_info.selection_method
    
    @property
    def n_cv_folds(self) -> Optional[int]:
        """Number of CV folds used (only available when fit with cv=)."""
        if self._regularization_path_info is None:
            return None
        return self._regularization_path_info.n_folds
    
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
    
    def compute_loss(
        self, 
        data: "pl.DataFrame",
        response: Optional[str] = None,
        exposure: Optional[str] = None,
    ) -> float:
        """
        Compute family-appropriate loss (mean deviance) on given data.
        
        This method re-predicts on the data to ensure consistent encoding,
        which is critical for TE() terms that use leave-one-out during fit
        but full encoding for prediction.
        
        Parameters
        ----------
        data : pl.DataFrame
            Data to compute loss on (can be train, test, or holdout).
        response : str, optional
            Response column name. Auto-detected from formula if not provided.
        exposure : str, optional
            Exposure column name for rate models.
            
        Returns
        -------
        float
            Mean deviance (family-appropriate loss metric).
            
        Examples
        --------
        >>> train_loss = result.compute_loss(train_data)
        >>> test_loss = result.compute_loss(test_data)
        >>> assert train_loss < test_loss  # Expected for non-overfitting models
        """
        from rustystats._rustystats import compute_loss_metrics_py as _rust_loss_metrics
        
        # Get response column from formula
        if response is None:
            formula_parts = self.formula.split('~')
            response = formula_parts[0].strip() if formula_parts else None
        
        if response is None or response not in data.columns:
            raise ValueError(f"Response column '{response}' not found in data")
        
        y = data[response].to_numpy().astype(np.float64)
        
        # Re-predict to get consistent encoding (critical for TE terms)
        mu = np.asarray(self.predict(data), dtype=np.float64)
        
        # Compute family-appropriate loss
        loss_metrics = _rust_loss_metrics(y, mu, self.family)
        return loss_metrics["family_loss"]
    
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
    
    def diagnostics(
        self,
        train_data: "pl.DataFrame",
        categorical_factors: Optional[List[str]] = None,
        continuous_factors: Optional[List[str]] = None,
        n_calibration_bins: int = 10,
        n_factor_bins: int = 10,
        rare_threshold_pct: float = 1.0,
        max_categorical_levels: int = 20,
        detect_interactions: bool = True,
        max_interaction_factors: int = 10,
        # Test data for overfitting detection (response/exposure auto-inferred)
        test_data: Optional["pl.DataFrame"] = None,
        # Control enhanced diagnostics
        compute_vif: bool = True,
        compute_coefficients: bool = True,
        compute_deviance_by_level: bool = True,
        compute_lift: bool = True,
        compute_partial_dep: bool = True,
        # Legacy parameter (deprecated)
        data: Optional["pl.DataFrame"] = None,
    ):
        """
        Compute comprehensive model diagnostics.
        
        Parameters
        ----------
        train_data : pl.DataFrame
            Training data used for fitting.
        categorical_factors : list of str, optional
            Names of categorical factors to analyze (both fitted and unfitted).
        continuous_factors : list of str, optional
            Names of continuous factors to analyze (both fitted and unfitted).
        n_calibration_bins : int, default=10
            Number of bins for calibration curve.
        n_factor_bins : int, default=10
            Number of quantile bins for continuous factors.
        rare_threshold_pct : float, default=1.0
            Threshold (%) below which categorical levels are grouped into "Other".
        max_categorical_levels : int, default=20
            Maximum number of categorical levels to show.
        detect_interactions : bool, default=True
            Whether to detect potential interactions.
        max_interaction_factors : int, default=10
            Maximum factors to consider for interaction detection.
        test_data : pl.DataFrame, optional
            Test/holdout data for overfitting detection. Response and exposure
            columns are automatically inferred from the model's formula.
        compute_vif : bool, default=True
            Compute VIF/multicollinearity scores for design matrix (train-only).
        compute_coefficients : bool, default=True
            Compute coefficient summary with interpretations (train-only).
        compute_deviance_by_level : bool, default=True
            Compute deviance breakdown by categorical factor levels.
        compute_lift : bool, default=True
            Compute full lift chart with all deciles.
        compute_partial_dep : bool, default=True
            Compute partial dependence plots for each variable.
        
        Returns
        -------
        ModelDiagnostics
            Complete diagnostics object with to_json() method.
            
            Fields for agentic workflows:
            - vif: VIF scores detecting multicollinearity (train-only)
            - coefficient_summary: Coefficient magnitudes and recommendations (train-only)
            - factor_deviance: Deviance by categorical level
            - lift_chart: Full lift chart showing discrimination by decile
            - partial_dependence: Marginal effect shapes for linear vs spline decisions
            - train_test: Comprehensive train vs test comparison with flags:
                - overfitting_risk: True if gini_gap > 0.03
                - calibration_drift: True if test A/E outside [0.95, 1.05]
                - unstable_factors: Factors where train/test A/E differ by > 0.1
        
        Examples
        --------
        >>> result = rs.glm("ClaimNb ~ Age + C(Region)", data, family="poisson", offset="Exposure").fit()
        >>> 
        >>> # Basic diagnostics
        >>> diagnostics = result.diagnostics(
        ...     train_data=train_data,
        ...     categorical_factors=["Region", "VehBrand"],
        ...     continuous_factors=["Age", "VehPower"]
        ... )
        >>> 
        >>> # With test data for overfitting detection
        >>> diagnostics = result.diagnostics(
        ...     train_data=train_data,
        ...     test_data=test_data,
        ...     categorical_factors=["Region"],
        ...     continuous_factors=["Age"],
        ... )
        >>> 
        >>> # Check overfitting flags
        >>> if diagnostics.train_test and diagnostics.train_test.overfitting_risk:
        ...     print("Warning: Overfitting detected!")
        >>> 
        >>> print(diagnostics.to_json())
        """
        from rustystats.diagnostics import compute_diagnostics
        
        # Support legacy 'data' parameter
        if train_data is None and data is not None:
            train_data = data
        
        # Get design matrix for VIF calculation
        design_matrix = None
        if compute_vif and self._design_matrix is not None:
            design_matrix = self._design_matrix
        
        return compute_diagnostics(
            result=self,
            train_data=train_data,
            categorical_factors=categorical_factors,
            continuous_factors=continuous_factors,
            n_calibration_bins=n_calibration_bins,
            n_factor_bins=n_factor_bins,
            rare_threshold_pct=rare_threshold_pct,
            max_categorical_levels=max_categorical_levels,
            detect_interactions=detect_interactions,
            max_interaction_factors=max_interaction_factors,
            test_data=test_data,
            design_matrix=design_matrix,
            compute_vif=compute_vif,
            compute_coefficients=compute_coefficients,
            compute_deviance_by_level=compute_deviance_by_level,
            compute_lift=compute_lift,
            compute_partial_dep=compute_partial_dep,
        )
    
    def diagnostics_json(
        self,
        train_data: "pl.DataFrame",
        categorical_factors: Optional[List[str]] = None,
        continuous_factors: Optional[List[str]] = None,
        n_calibration_bins: int = 10,
        n_factor_bins: int = 10,
        rare_threshold_pct: float = 1.0,
        max_categorical_levels: int = 20,
        detect_interactions: bool = True,
        max_interaction_factors: int = 10,
        test_data: Optional["pl.DataFrame"] = None,
        indent: Optional[int] = None,
        # Legacy parameter
        data: Optional["pl.DataFrame"] = None,
    ) -> str:
        """
        Compute diagnostics and return as JSON string.
        
        This is a convenience method that calls diagnostics() and converts
        the result to JSON. The output is optimized for LLM consumption.
        
        Parameters
        ----------
        train_data : pl.DataFrame
            Training data used for fitting.
        categorical_factors : list of str, optional
            Names of categorical factors to analyze.
        continuous_factors : list of str, optional
            Names of continuous factors to analyze.
        test_data : pl.DataFrame, optional
            Test data for overfitting detection.
        indent : int, optional
            JSON indentation. None for compact output.
        
        Returns
        -------
        str
            JSON string containing all diagnostics.
        """
        # Support legacy 'data' parameter
        if train_data is None and data is not None:
            train_data = data
        
        diag = self.diagnostics(
            train_data=train_data,
            categorical_factors=categorical_factors,
            continuous_factors=continuous_factors,
            n_calibration_bins=n_calibration_bins,
            n_factor_bins=n_factor_bins,
            rare_threshold_pct=rare_threshold_pct,
            max_categorical_levels=max_categorical_levels,
            detect_interactions=detect_interactions,
            max_interaction_factors=max_interaction_factors,
            test_data=test_data,
        )
        return diag.to_json(indent=indent)
    
    def predict(
        self,
        new_data: "pl.DataFrame",
        offset: Optional[Union[str, np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Predict on new data using the fitted model.
        
        Parameters
        ----------
        new_data : pl.DataFrame
            New data to predict on. Must have the same columns as training data.
        offset : str or array-like, optional
            Offset for new data. If None and the model was fit with an offset
            column name, that column will be extracted from new_data.
            For Poisson/Gamma with log link, log() is auto-applied to exposure.
            
        Returns
        -------
        np.ndarray
            Predicted values (on the response scale, i.e., μ = E[Y]).
            
        Examples
        --------
        >>> model = rs.glm("ClaimNb ~ Age + C(Region)", data, family="poisson", offset="Exposure")
        >>> result = model.fit()
        >>> 
        >>> # Predict on new data
        >>> predictions = result.predict(new_data)
        >>> 
        >>> # Predict with custom offset
        >>> predictions = result.predict(new_data, offset=np.log(new_exposures))
        """
        if self._builder is None:
            raise ValueError(
                "Cannot predict: model was not fitted with formula API. "
                "Use fittedvalues for training data predictions."
            )
        
        # Build design matrix for new data using stored encoding state
        X_new = self._builder.transform_new_data(new_data)
        
        # Compute linear predictor: η = X @ β
        linear_pred = X_new @ self.params
        
        # Handle offset
        # If offset is provided as a string, extract column and apply log() for log-link models
        # If offset is provided as array, use directly (user handles transformation)
        # If offset is None but model was fit with offset, use the stored offset column
        offset_to_use = offset
        if offset_to_use is None and hasattr(self, '_offset_spec') and self._offset_spec is not None:
            # Auto-use the offset column from fitting
            offset_to_use = self._offset_spec
        
        if offset_to_use is not None:
            if isinstance(offset_to_use, str):
                offset_values = new_data[offset_to_use].to_numpy().astype(np.float64)
                # Apply log() for log-link models (same as fitting)
                if self.link == "log":
                    offset_values = np.log(offset_values)
            else:
                offset_values = np.asarray(offset_to_use, dtype=np.float64)
            linear_pred = linear_pred + offset_values
        
        # Apply inverse link function to get predictions on response scale
        return self._apply_inverse_link(linear_pred)
    
    def _apply_inverse_link(self, eta: np.ndarray) -> np.ndarray:
        """Apply inverse link function to linear predictor."""
        link = self.link
        if link == "identity":
            return eta
        elif link == "log":
            return np.exp(eta)
        elif link == "logit":
            return 1.0 / (1.0 + np.exp(-eta))
        elif link == "inverse":
            return 1.0 / eta
        else:
            # Default to identity
            return eta
    
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
    seed: Optional[int] = None,
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
        
    seed : int, optional
        Random seed for deterministic target encoding (TE). If None,
        TE uses random permutations which may produce different results
        on each run. Set to a fixed value for reproducibility.
        
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
        seed=seed,
    )


# =============================================================================
# Dict-based API
# =============================================================================

from typing import Dict, Any, Set
from rustystats.interactions import (
    ParsedFormula, InteractionTerm, TargetEncodingTermSpec, 
    IdentityTermSpec, CategoricalTermSpec, ConstraintTermSpec
)
from rustystats.splines import SplineTerm


def _parse_term_spec(
    var_name: str,
    spec: Dict[str, Any],
    categorical_vars: Set[str],
    main_effects: List[str],
    spline_terms: List[SplineTerm],
    target_encoding_terms: List[TargetEncodingTermSpec],
    identity_terms: List[IdentityTermSpec],
    categorical_terms: List[CategoricalTermSpec],
    constraint_terms: List[ConstraintTermSpec],
) -> None:
    """Parse a single term specification and add to appropriate lists."""
    term_type = spec.get("type", "linear")
    monotonicity = spec.get("monotonicity")  # "increasing" or "decreasing"
    
    if term_type == "linear":
        if monotonicity:
            # Constrained linear term
            constraint = "pos" if monotonicity == "increasing" else "neg"
            constraint_terms.append(ConstraintTermSpec(
                var_name=var_name,
                constraint=constraint,
            ))
        else:
            main_effects.append(var_name)
    
    elif term_type == "categorical":
        categorical_vars.add(var_name)
        levels = spec.get("levels")
        if levels:
            # Specific levels only
            categorical_terms.append(CategoricalTermSpec(
                var_name=var_name,
                levels=levels,
            ))
        else:
            main_effects.append(var_name)
    
    elif term_type == "bs":
        df = spec.get("df", 5)
        degree = spec.get("degree", 3)
        if monotonicity:
            # Monotonic B-spline (I-spline)
            increasing = monotonicity == "increasing"
            term = SplineTerm(
                var_name=var_name,
                spline_type="ms",
                df=df,
                degree=degree,
                increasing=increasing,
            )
            term._monotonic = True
            spline_terms.append(term)
        else:
            spline_terms.append(SplineTerm(
                var_name=var_name,
                spline_type="bs",
                df=df,
                degree=degree,
            ))
    
    elif term_type == "ns":
        df = spec.get("df", 4)
        if monotonicity:
            raise ValueError(
                f"Monotonicity constraints are not supported for natural splines (ns). "
                f"Use type='ms' instead for monotonic effects."
            )
        spline_terms.append(SplineTerm(
            var_name=var_name,
            spline_type="ns",
            df=df,
        ))
    
    elif term_type == "ms":
        # Monotonic spline (I-spline)
        df = spec.get("df", 4)
        degree = spec.get("degree", 3)
        # Support both 'monotonicity' (consistent with other terms) and 'increasing' (legacy)
        if monotonicity:
            increasing = monotonicity == "increasing"
        else:
            increasing = spec.get("increasing", True)
        term = SplineTerm(
            var_name=var_name,
            spline_type="ms",
            df=df,
            degree=degree,
            increasing=increasing,
        )
        term._monotonic = True
        spline_terms.append(term)
    
    elif term_type == "target_encoding":
        prior_weight = spec.get("prior_weight", 1.0)
        n_permutations = spec.get("n_permutations", 4)
        target_encoding_terms.append(TargetEncodingTermSpec(
            var_name=var_name,
            prior_weight=prior_weight,
            n_permutations=n_permutations,
        ))
    
    elif term_type == "expression":
        expr = spec.get("expr", var_name)
        if monotonicity:
            constraint = "pos" if monotonicity == "increasing" else "neg"
            constraint_terms.append(ConstraintTermSpec(
                var_name=f"I({expr})",
                constraint=constraint,
            ))
        else:
            identity_terms.append(IdentityTermSpec(expression=expr))
    
    else:
        raise ValueError(f"Unknown term type: {term_type}")


def _parse_interaction_spec(
    interaction: Dict[str, Any],
    interactions: List[InteractionTerm],
    categorical_vars: Set[str],
    main_effects: List[str],
    spline_terms: List[SplineTerm],
    target_encoding_terms: List[TargetEncodingTermSpec],
    identity_terms: List[IdentityTermSpec],
    categorical_terms: List[CategoricalTermSpec],
    constraint_terms: List[ConstraintTermSpec],
) -> None:
    """Parse an interaction specification."""
    include_main = interaction.get("include_main", False)
    
    # Extract variable specs (everything except include_main)
    var_specs = {k: v for k, v in interaction.items() if k != "include_main"}
    
    if len(var_specs) < 2:
        raise ValueError("Interaction must have at least 2 variables")
    
    factors = list(var_specs.keys())
    
    # Determine which factors are categorical or splines
    cat_factors = set()
    spline_factors = []
    
    for var_name, spec in var_specs.items():
        term_type = spec.get("type", "linear")
        
        if term_type == "categorical":
            cat_factors.add(var_name)
            categorical_vars.add(var_name)
        elif term_type in ("bs", "ns"):
            df = spec.get("df", 5 if term_type == "bs" else 4)
            degree = spec.get("degree", 3)
            monotonicity = spec.get("monotonicity")
            if monotonicity:
                spline = SplineTerm(
                    var_name=var_name,
                    spline_type="ms",
                    df=df,
                    degree=degree,
                    increasing=monotonicity == "increasing",
                )
            else:
                spline = SplineTerm(
                    var_name=var_name,
                    spline_type=term_type,
                    df=df,
                    degree=degree,
                )
            spline_factors.append((var_name, spline))
        elif term_type == "target_encoding":
            prior_weight = spec.get("prior_weight", 1.0)
            # TE in interaction - add to TE terms if include_main
            if include_main:
                target_encoding_terms.append(TargetEncodingTermSpec(
                    var_name=var_name,
                    prior_weight=prior_weight,
                ))
    
    # Build interaction term
    is_pure_cat = all(var_specs[f].get("type") == "categorical" for f in factors)
    is_pure_cont = all(var_specs[f].get("type") in ("linear", None) for f in factors)
    
    interaction_term = InteractionTerm(
        factors=factors,
        categorical_factors=list(cat_factors),
        is_pure_categorical=is_pure_cat,
        is_pure_continuous=is_pure_cont,
        spline_factors=spline_factors if spline_factors else None,
    )
    interactions.append(interaction_term)
    
    # Add main effects if requested
    if include_main:
        for var_name, spec in var_specs.items():
            _parse_term_spec(
                var_name, spec, categorical_vars, main_effects,
                spline_terms, target_encoding_terms, identity_terms,
                categorical_terms, constraint_terms,
            )


def dict_to_parsed_formula(
    response: str,
    terms: Dict[str, Dict[str, Any]],
    interactions: Optional[List[Dict[str, Any]]] = None,
    intercept: bool = True,
) -> ParsedFormula:
    """
    Convert dict specification to ParsedFormula.
    
    Parameters
    ----------
    response : str
        Name of the response variable
    terms : dict
        Dictionary mapping variable names to term specifications
    interactions : list of dict, optional
        List of interaction specifications
    intercept : bool, default=True
        Whether to include an intercept
        
    Returns
    -------
    ParsedFormula
        Parsed formula object compatible with build_design_matrix
    """
    categorical_vars: Set[str] = set()
    main_effects: List[str] = []
    spline_terms_list: List[SplineTerm] = []
    target_encoding_terms_list: List[TargetEncodingTermSpec] = []
    identity_terms_list: List[IdentityTermSpec] = []
    categorical_terms_list: List[CategoricalTermSpec] = []
    constraint_terms_list: List[ConstraintTermSpec] = []
    interaction_terms_list: List[InteractionTerm] = []
    
    # Parse main terms
    for var_name, spec in terms.items():
        _parse_term_spec(
            var_name, spec, categorical_vars, main_effects,
            spline_terms_list, target_encoding_terms_list, identity_terms_list,
            categorical_terms_list, constraint_terms_list,
        )
    
    # Parse interactions
    if interactions:
        for interaction in interactions:
            _parse_interaction_spec(
                interaction, interaction_terms_list, categorical_vars,
                main_effects, spline_terms_list, target_encoding_terms_list,
                identity_terms_list, categorical_terms_list, constraint_terms_list,
            )
    
    return ParsedFormula(
        response=response,
        main_effects=main_effects,
        interactions=interaction_terms_list,
        categorical_vars=categorical_vars,
        spline_terms=spline_terms_list,
        target_encoding_terms=target_encoding_terms_list,
        identity_terms=identity_terms_list,
        categorical_terms=categorical_terms_list,
        constraint_terms=constraint_terms_list,
        has_intercept=intercept,
    )


class FormulaGLMDict:
    """
    GLM model with dict-based specification.
    
    Alternative to formula strings for programmatic model building.
    """
    
    def __init__(
        self,
        response: str,
        terms: Dict[str, Dict[str, Any]],
        data: "pl.DataFrame",
        interactions: Optional[List[Dict[str, Any]]] = None,
        intercept: bool = True,
        family: str = "gaussian",
        link: Optional[str] = None,
        var_power: float = 1.5,
        theta: Optional[float] = None,
        offset: Optional[Union[str, np.ndarray]] = None,
        weights: Optional[Union[str, np.ndarray]] = None,
        seed: Optional[int] = None,
    ):
        self.response = response
        self.terms = terms
        self.interactions_spec = interactions
        self.intercept = intercept
        self.data = data
        self.family = family.lower()
        self.link = link
        self.var_power = var_power
        self.theta = theta
        self._offset_spec = offset
        self._weights_spec = weights
        self._seed = seed
        
        # Build formula string for compatibility (used in results/diagnostics)
        self.formula = self._build_formula_string()
        
        # Convert dict to ParsedFormula
        parsed = dict_to_parsed_formula(
            response=response,
            terms=terms,
            interactions=interactions,
            intercept=intercept,
        )
        
        # Extract raw exposure for target encoding
        raw_exposure = self._get_raw_exposure(offset)
        
        # Build design matrix using existing pipeline
        self._builder = InteractionBuilder(data)
        self.y, self.X, self.feature_names = self._builder.build_design_matrix_from_parsed(
            parsed, exposure=raw_exposure, seed=seed
        )
        self.n_obs = len(self.y)
        self.n_params = self.X.shape[1]
        
        # Process offset and weights
        self.offset = self._process_offset(offset)
        self.weights = self._process_weights(weights)
    
    def _build_formula_string(self) -> str:
        """Build a formula string representation for display purposes."""
        parts = [self.response, "~"]
        term_strs = []
        
        for var_name, spec in self.terms.items():
            term_type = spec.get("type", "linear")
            if term_type == "linear":
                term_strs.append(var_name)
            elif term_type == "categorical":
                term_strs.append(f"C({var_name})")
            elif term_type == "bs":
                df = spec.get("df", 5)
                term_strs.append(f"bs({var_name}, df={df})")
            elif term_type == "ns":
                df = spec.get("df", 4)
                term_strs.append(f"ns({var_name}, df={df})")
            elif term_type == "target_encoding":
                term_strs.append(f"TE({var_name})")
            elif term_type == "expression":
                expr = spec.get("expr", var_name)
                term_strs.append(f"I({expr})")
        
        if not self.intercept:
            term_strs.insert(0, "0")
        
        parts.append(" + ".join(term_strs) if term_strs else "1")
        return " ".join(parts)
    
    def _get_raw_exposure(self, offset) -> Optional[np.ndarray]:
        """Extract raw exposure values for target encoding."""
        if offset is None:
            return None
        if isinstance(offset, str):
            return self.data[offset].to_numpy().astype(np.float64)
        return np.asarray(offset, dtype=np.float64)
    
    def _process_offset(self, offset) -> Optional[np.ndarray]:
        """Process offset, applying log for log-link families."""
        if offset is None:
            return None
        
        if isinstance(offset, str):
            offset_values = self.data[offset].to_numpy().astype(np.float64)
        else:
            offset_values = np.asarray(offset, dtype=np.float64)
        
        # Apply log for Poisson/Gamma families
        if self.family in ("poisson", "gamma", "quasipoisson", "tweedie", "negbinomial"):
            return np.log(offset_values)
        return offset_values
    
    def _process_weights(self, weights) -> Optional[np.ndarray]:
        """Process weights."""
        if weights is None:
            return None
        if isinstance(weights, str):
            return self.data[weights].to_numpy().astype(np.float64)
        return np.asarray(weights, dtype=np.float64)
    
    def fit(
        self,
        max_iter: int = 25,
        tol: float = 1e-8,
        alpha: float = 0.0,
        l1_ratio: float = 0.0,
    ) -> FormulaGLMResults:
        """Fit the GLM model."""
        from rustystats._rustystats import fit_glm_py as _fit_glm_rust
        
        is_negbinomial = self.family in ("negbinomial", "negativebinomial", "negative_binomial", "neg-binomial", "nb")
        theta = self.theta if self.theta is not None else 1.0
        
        # Compute coefficient constraint indices
        nonneg_indices, nonpos_indices = _get_constraint_indices(self.feature_names)
        
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
            nonneg_indices if nonneg_indices else None,
            nonpos_indices if nonpos_indices else None,
        )
        
        result_family = f"NegativeBinomial(theta={theta:.4f})" if is_negbinomial else self.family
        
        return FormulaGLMResults(
            result=result,
            feature_names=self.feature_names,
            formula=self.formula,
            family=result_family,
            link=self.link,
            builder=self._builder,
            design_matrix=self.X,
            offset_spec=self._offset_spec,
            offset_is_exposure=(self.family in ("poisson", "quasipoisson", "negbinomial", "gamma") and self.link in (None, "log")),
        )


def glm_dict(
    response: str,
    terms: Dict[str, Dict[str, Any]],
    data: "pl.DataFrame",
    interactions: Optional[List[Dict[str, Any]]] = None,
    intercept: bool = True,
    family: str = "gaussian",
    link: Optional[str] = None,
    var_power: float = 1.5,
    theta: Optional[float] = None,
    offset: Optional[Union[str, np.ndarray]] = None,
    weights: Optional[Union[str, np.ndarray]] = None,
    seed: Optional[int] = None,
) -> FormulaGLMDict:
    """
    Create a GLM model from a dict specification.
    
    This is an alternative to the formula-based API for programmatic model building.
    
    Parameters
    ----------
    response : str
        Name of the response variable column.
    terms : dict
        Dictionary mapping variable names to term specifications.
        Each specification is a dict with 'type' and optional parameters:
        
        - ``{"type": "linear"}`` - continuous variable
        - ``{"type": "categorical"}`` - dummy encoding
        - ``{"type": "categorical", "levels": ["A", "B"]}`` - specific levels
        - ``{"type": "bs", "df": 5}`` - B-spline
        - ``{"type": "bs", "df": 5, "degree": 2}`` - quadratic B-spline
        - ``{"type": "ns", "df": 4}`` - natural spline
        - ``{"type": "bs", "df": 4, "monotonicity": "increasing"}`` - monotonic
        - ``{"type": "target_encoding"}`` - target encoding
        - ``{"type": "expression", "expr": "x**2"}`` - expression
        - ``{"type": "linear", "monotonicity": "increasing"}`` - constrained
        
    data : pl.DataFrame
        Polars DataFrame containing the data.
    interactions : list of dict, optional
        List of interaction specifications. Each is a dict with variable
        names as keys and their specs as values, plus 'include_main'.
    intercept : bool, default=True
        Whether to include an intercept.
    family : str, default="gaussian"
        Distribution family.
    link : str, optional
        Link function. If None, uses canonical link.
    var_power : float, default=1.5
        Variance power for Tweedie family.
    theta : float, optional
        Dispersion for Negative Binomial.
    offset : str or array-like, optional
        Offset term.
    weights : str or array-like, optional
        Prior weights.
    seed : int, optional
        Random seed for deterministic target encoding.
        
    Returns
    -------
    FormulaGLMDict
        Model object. Call .fit() to fit the model.
        
    Examples
    --------
    >>> result = rs.glm_dict(
    ...     response="ClaimCount",
    ...     terms={
    ...         "VehAge": {"type": "linear"},
    ...         "DrivAge": {"type": "bs", "df": 5},
    ...         "Region": {"type": "categorical"},
    ...         "Brand": {"type": "target_encoding"},
    ...     },
    ...     interactions=[
    ...         {"VehAge": {"type": "linear"}, "Region": {"type": "categorical"}, "include_main": True},
    ...     ],
    ...     data=data,
    ...     family="poisson",
    ...     offset="Exposure",
    ... ).fit()
    """
    return FormulaGLMDict(
        response=response,
        terms=terms,
        data=data,
        interactions=interactions,
        intercept=intercept,
        family=family,
        link=link,
        var_power=var_power,
        theta=theta,
        offset=offset,
        weights=weights,
        seed=seed,
    )
