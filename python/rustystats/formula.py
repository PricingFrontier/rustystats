"""
Dict-based API for RustyStats GLM.

This module provides the dict-based API for fitting GLMs with DataFrames.

Example
-------
>>> import rustystats as rs
>>> import polars as pl
>>> 
>>> data = pl.read_parquet("insurance_data.parquet")
>>> result = rs.glm_dict(
...     response="ClaimNb",
...     terms={
...         "VehPower": {"type": "linear"},
...         "VehAge": {"type": "linear"},
...         "VehBrand": {"type": "categorical"},
...     },
...     data=data,
...     family="poisson",
...     offset="Exposure",
... ).fit()
>>> print(rs.summary(result))
"""

from __future__ import annotations

import weakref
from typing import Optional, Union, List, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass
import warnings
import numpy as np

from rustystats.exceptions import (
    ValidationError,
    FittingError,
    PredictionError,
    SerializationError,
)

from rustystats.constants import (
    DEFAULT_NEGBINOMIAL_THETA,
    DEFAULT_MAX_ITER,
    DEFAULT_TOLERANCE,
    DEFAULT_N_ALPHAS,
    DEFAULT_ALPHA_MIN_RATIO,
    DEFAULT_SPLINE_DF,
    DEFAULT_LAMBDA_MIN,
    DEFAULT_LAMBDA_MAX,
    DEFAULT_N_LAMBDA,
    DEFAULT_THETA_TOL,
    DEFAULT_MAX_THETA_ITER,
    DEFAULT_N_CALIBRATION_BINS,
    DEFAULT_N_FACTOR_BINS,
    DEFAULT_RARE_THRESHOLD_PCT,
    DEFAULT_MAX_CATEGORICAL_LEVELS,
    DEFAULT_MAX_INTERACTION_FACTORS,
    DEFAULT_LINKS,
    NEGBINOMIAL_ALIASES,
)


def is_negbinomial_family(family: str) -> bool:
    """Check if the family string refers to a Negative Binomial distribution."""
    return family.lower() in NEGBINOMIAL_ALIASES


def get_default_link(family: str) -> str:
    """
    Get the canonical default link function for a GLM family.
    
    Parameters
    ----------
    family : str
        Family name (e.g., "gaussian", "poisson", "binomial")
        
    Returns
    -------
    str
        Default link function name (e.g., "identity", "log", "logit")
        
    Raises
    ------
    ValueError
        If family is not recognized.
    """
    family_lower = family.lower()
    # Handle NegativeBinomial(theta=...) format from result strings
    if family_lower.startswith("negativebinomial"):
        return "log"
    link = DEFAULT_LINKS.get(family_lower)
    if link is None:
        raise ValidationError(
            f"Unknown family '{family}'. "
            f"Supported families: {sorted(DEFAULT_LINKS.keys())}"
        )
    return link

# Lazy imports for optional dependencies
if TYPE_CHECKING:
    import polars as pl
    from rustystats.regularization_path import RegularizationPathInfo
    from rustystats.diagnostics.types import DataExploration, ModelDiagnostics


def _get_column(data: "pl.DataFrame", column: str) -> np.ndarray:
    """Extract a column as numpy array from Polars DataFrame."""
    return data[column].to_numpy()


# Import from interactions module (the canonical implementation)
from rustystats.interactions import InteractionBuilder


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


@dataclass
class SmoothTermResult:
    """Result for a single smooth term after fitting."""
    variable: str
    k: int
    edf: float
    lambda_: float
    gcv: float
    col_start: int
    col_end: int


def _fit_with_smooth_penalties(
    y: np.ndarray,
    X: np.ndarray,
    smooth_terms: List[Any],
    smooth_col_indices: List[tuple],
    family: str,
    link: str,
    var_power: float,
    theta: float,
    offset: Optional[np.ndarray],
    weights: Optional[np.ndarray],
    max_iter: int = DEFAULT_MAX_ITER,
    tol: float = DEFAULT_TOLERANCE,
    n_lambda: int = DEFAULT_N_LAMBDA,
    lambda_min: float = DEFAULT_LAMBDA_MIN,
    lambda_max: float = DEFAULT_LAMBDA_MAX,
) -> tuple:
    """
    Fit GLM with penalized smooth terms using fast GCV optimization.
    
    Uses a unified Rust entry point that takes the full design matrix and
    smooth term specs (column ranges + penalties + monotonicity). No column
    splitting or coefficient reordering needed.
    
    Parameters
    ----------
    y : array
        Response variable
    X : array
        Full design matrix
    smooth_terms : list
        List of SplineTerm objects marked as smooth
    smooth_col_indices : list
        List of (start, end) column indices for each smooth term
    family, link, var_power, theta : model parameters
    offset, weights : optional arrays
    max_iter, tol : IRLS parameters
    n_lambda, lambda_min, lambda_max : GCV grid search parameters
    
    Returns
    -------
    result : GLMResult from Rust
    smooth_results : list of SmoothTermResult
    total_edf : float
    gcv : float
    """
    n, p = X.shape
    n_terms = len(smooth_terms)
    
    from rustystats._rustystats import fit_smooth_glm_unified_py as _fit_smooth_unified
    
    # Build penalty matrices and monotonicity specs for each smooth term
    penalties = []
    monotonicity_specs = []
    for i, term in enumerate(smooth_terms):
        start, end = smooth_col_indices[i]
        k = end - start
        penalties.append(term.compute_penalty_matrix(k)[:k, :k])
        
        mono = getattr(term, '_smooth_monotonicity', None) or \
               getattr(term, 'monotonicity', None)
        monotonicity_specs.append(mono)
    
    # Call unified Rust solver — full design matrix, no splitting needed
    rust_result, smooth_meta = _fit_smooth_unified(
        y, X, smooth_col_indices, penalties, family,
        link, offset, weights, max_iter, tol, lambda_min, lambda_max,
        monotonicity_specs if any(m is not None for m in monotonicity_specs) else None,
    )
    
    # Build smooth term results — coefficients are already in original column order
    smooth_results = []
    for i, term in enumerate(smooth_terms):
        start, end = smooth_col_indices[i]
        smooth_results.append(SmoothTermResult(
            variable=term.var_name,
            k=term.df,
            edf=smooth_meta['smooth_edfs'][i],
            lambda_=smooth_meta['lambdas'][i],
            gcv=smooth_meta['gcv'],
            col_start=start,
            col_end=end,
        ))
        term._lambda = smooth_meta['lambdas'][i]
        term._edf = smooth_meta['smooth_edfs'][i]
    
    return rust_result, smooth_results, smooth_meta['total_edf'], smooth_meta['gcv']


def _fit_glm_core(
    y: np.ndarray,
    X: np.ndarray,
    family: str,
    link: str,
    var_power: float,
    theta: float,
    offset: Optional[np.ndarray],
    weights: Optional[np.ndarray],
    alpha: float,
    l1_ratio: float,
    max_iter: int,
    tol: float,
    feature_names: List[str],
    builder: "InteractionBuilder",
) -> tuple:
    """
    Core GLM fitting logic shared by FormulaGLM and FormulaGLMDict.
    
    Handles smooth term fitting with GCV-based lambda selection and
    standard fitting with coefficient constraints.
    
    Returns
    -------
    result : GLMResult
        Fitted model result from Rust
    smooth_results : list or None
        Smooth term results if applicable
    total_edf : float or None
        Total effective degrees of freedom
    gcv : float or None
        GCV score for smooth models
    """
    from rustystats._rustystats import fit_glm_py as _fit_glm_rust
    
    # Check for smooth terms (s() terms with automatic lambda selection)
    smooth_terms, smooth_col_indices = builder.get_smooth_terms()
    
    if smooth_terms and alpha == 0.0:
        # Use penalized fitting with GCV-based lambda selection
        result, smooth_results, total_edf, gcv = _fit_with_smooth_penalties(
            y, X, smooth_terms, smooth_col_indices,
            family, link, var_power, theta,
            offset, weights, max_iter, tol,
        )
        return result, smooth_results, total_edf, gcv
    else:
        # Standard fitting (no smooth terms or regularization already applied)
        # Compute coefficient constraint indices
        nonneg_indices, nonpos_indices = _get_constraint_indices(feature_names)
        
        result = _fit_glm_rust(
            y, X, family, link, var_power, theta,
            offset, weights, alpha, l1_ratio, max_iter, tol,
            nonneg_indices if nonneg_indices else None,
            nonpos_indices if nonpos_indices else None,
        )
        return result, None, None, None


def _build_results(
    result: Any,
    feature_names: List[str],
    formula: str,
    family: str,
    link: Optional[str],
    builder: "InteractionBuilder",
    X: np.ndarray,
    offset_spec: Optional[Union[str, np.ndarray]],
    is_exposure_offset: bool,
    path_info: Optional["RegularizationPathInfo"],
    smooth_results: Optional[List["SmoothTermResult"]],
    total_edf: Optional[float],
    gcv: Optional[float],
    store_design_matrix: bool = True,
    terms_dict: Optional[Dict[str, Dict[str, Any]]] = None,
    interactions_spec: Optional[List[Dict[str, Any]]] = None,
) -> "GLMModel":
    """Build GLMModel with all metadata."""
    # Clear builder caches to free memory (keep TE stats for prediction)
    if builder is not None:
        builder.clear_caches()
    
    return GLMModel(
        result=result,
        feature_names=feature_names,
        formula=formula,
        family=family,
        link=link,
        builder=builder,
        design_matrix=X if store_design_matrix else None,
        offset_spec=offset_spec,
        offset_is_exposure=is_exposure_offset,
        regularization_path_info=path_info,
        smooth_results=smooth_results,
        total_edf=total_edf,
        gcv=gcv,
        terms_dict=terms_dict,
        interactions_spec=interactions_spec,
    )


class _GLMBase:
    """
    Shared base for FormulaGLM and FormulaGLMDict.
    
    Provides common data access, offset/weights processing, and CV path handling.
    Subclasses must set: _data_ref, family, link, _offset_spec, _seed.
    """
    
    @property
    def data(self) -> "pl.DataFrame":
        """Access the original DataFrame (may raise if garbage collected)."""
        d = self._data_ref()
        if d is None:
            raise ValidationError(
                "Original DataFrame has been garbage collected. "
                "Keep a reference to the DataFrame if you need to access it after fitting."
            )
        return d
    
    def _uses_log_link(self) -> bool:
        """Check if model uses log link (explicit or canonical)."""
        if self.link == "log":
            return True
        if self.link is None and self.family in ("poisson", "quasipoisson", "negbinomial", "gamma"):
            return True
        return False
    
    def _process_offset(
        self,
        offset: Optional[Union[str, np.ndarray]],
    ) -> Optional[np.ndarray]:
        """Process offset specification, applying log for log-link families."""
        if offset is None:
            return None
        
        if isinstance(offset, str):
            offset_values = _get_column(self.data, offset)
            if self._uses_log_link():
                if np.all(offset_values > 0) and np.mean(offset_values) > 0.01:
                    offset_values = np.log(offset_values)
            return offset_values.astype(np.float64)
        else:
            return np.asarray(offset, dtype=np.float64)
    
    def _process_weights(
        self,
        weights: Optional[Union[str, np.ndarray]],
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
        offset: Optional[Union[str, np.ndarray]],
    ) -> Optional[np.ndarray]:
        """Get raw exposure values for target encoding (before log transform)."""
        if offset is None:
            return None
        if isinstance(offset, str):
            return _get_column(self.data, offset).astype(np.float64)
        else:
            return np.asarray(offset, dtype=np.float64)
    
    def _resolve_cv_path(
        self,
        alpha: float,
        l1_ratio: float,
        max_iter: int,
        tol: float,
        cv: Optional[int],
        selection: str,
        regularization: Optional[str],
        n_alphas: int,
        alpha_min_ratio: float,
        cv_seed: Optional[int],
        include_unregularized: bool,
        verbose: bool,
    ) -> tuple:
        """
        Handle CV-based regularization path if requested.
        
        Returns (alpha, l1_ratio, path_info) with updated alpha/l1_ratio
        from CV selection, or original values if no CV.
        """
        if regularization is not None and cv is None:
            cv = 5
        
        if cv is None:
            return alpha, l1_ratio, None
        
        if regularization is None:
            raise ValidationError(
                "When cv is specified, 'regularization' must be set to 'ridge', 'lasso', or 'elastic_net'"
            )
        
        from rustystats.regularization_path import fit_cv_regularization_path
        
        if regularization == "ridge":
            cv_l1_ratio = 0.0
        elif regularization == "lasso":
            cv_l1_ratio = 1.0
        elif regularization == "elastic_net":
            cv_l1_ratio = l1_ratio if l1_ratio > 0 else 0.5
        else:
            raise ValidationError(f"Unknown regularization type: {regularization}")
        
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
            seed=cv_seed if cv_seed is not None else self._seed,
            include_unregularized=include_unregularized,
            verbose=verbose,
        )
        
        if verbose:
            print(f"\nRefitting on full data with alpha={path_info.selected_alpha:.6f}")
        
        return path_info.selected_alpha, path_info.selected_l1_ratio, path_info


@dataclass
class _DeserializedResult:
    """
    Minimal result object for deserialized models.
    
    This provides the interface needed by GLMModel for prediction
    without requiring the full Rust GLMResults object.
    
    Note: fittedvalues and linear_predictor are not stored as they're
    large arrays not needed for prediction on new data.
    """
    params: np.ndarray
    deviance: float
    iterations: int
    converged: bool
    nobs: int
    df_resid: int
    df_model: int
    alpha: float
    l1_ratio: float
    is_regularized: bool
    penalty_type: str
    
    @property
    def fittedvalues(self) -> np.ndarray:
        raise AttributeError(
            "fittedvalues not available on deserialized models. "
            "Only coefficients are stored for prediction."
        )
    
    @property
    def linear_predictor(self) -> np.ndarray:
        raise AttributeError(
            "linear_predictor not available on deserialized models. "
            "Only coefficients are stored for prediction."
        )


class _DeserializedBuilder(InteractionBuilder):
    """
    Minimal builder for deserialized models.
    
    Inherits all transform_new_data / prediction logic from InteractionBuilder.
    Only overrides __init__ to restore state from a serialized dict instead
    of building from a live DataFrame.
    """
    
    def __init__(self, state: dict):
        # Bypass InteractionBuilder.__init__ — set state directly from dict
        self._parsed_formula = state["parsed_formula"]
        self._cat_encoding_cache = state["cat_encoding_cache"]
        self._fitted_splines = state["fitted_splines"]
        self._te_stats = state["te_stats"]
        self._fe_stats: Dict[str, dict] = {}
        self.dtype = state["dtype"]
        self.data = None
        self._n = 0


class GLMModel:
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
        smooth_results: Optional[List[SmoothTermResult]] = None,
        total_edf: Optional[float] = None,
        gcv: Optional[float] = None,
        terms_dict: Optional[Dict[str, Dict[str, Any]]] = None,
        interactions_spec: Optional[List[Dict[str, Any]]] = None,
    ):
        self._result = result
        self._is_deserialized = isinstance(result, _DeserializedResult)
        self._smooth_results = smooth_results
        self._total_edf = total_edf
        self._gcv = gcv
        self.feature_names = feature_names
        self.formula = formula
        self.family = family
        self._regularization_path_info = regularization_path_info
        self.link = link or get_default_link(family)
        self._builder = builder
        self._design_matrix = design_matrix  # Store for VIF calculation
        self._offset_spec = offset_spec
        self._offset_is_exposure = offset_is_exposure
        self._terms_dict = terms_dict
        self._interactions_spec = interactions_spec
    
    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying result object.
        
        This handles all properties and methods from PyGLMResults that are
        not explicitly defined on GLMModel (params, fittedvalues, deviance,
        bse, tvalues, pvalues, conf_int, resid_*, llf, aic, bic, scale,
        robust SEs, regularization properties, etc.).
        """
        if name.startswith('_'):
            raise AttributeError(name)
        return getattr(self._result, name)
    
    @property
    def smooth_terms(self) -> Optional[List[SmoothTermResult]]:
        """Smooth term results with EDF, lambda, and GCV for each s() term."""
        return self._smooth_results
    
    @property
    def total_edf(self) -> Optional[float]:
        """Total effective degrees of freedom (parametric + smooth terms)."""
        return self._total_edf
    
    @property
    def gcv(self) -> Optional[float]:
        """Generalized Cross-Validation score for smoothness selection."""
        return self._gcv
    
    def has_smooth_terms(self) -> bool:
        """Check if model contains smooth terms with automatic smoothing."""
        return self._smooth_results is not None and len(self._smooth_results) > 0
    
    @property
    def terms_dict(self) -> Optional[Dict[str, Dict[str, Any]]]:
        """Original terms dictionary used to specify the model (dict API only)."""
        return self._terms_dict
    
    @property
    def interactions_spec(self) -> Optional[List[Dict[str, Any]]]:
        """Original interactions specification used to specify the model (dict API only)."""
        return self._interactions_spec
    
    def get_design_matrix(self) -> Optional[np.ndarray]:
        """Get the design matrix X used in fitting."""
        try:
            return np.asarray(self._result.design_matrix)
        except AttributeError:
            return None
    
    def get_irls_weights(self) -> Optional[np.ndarray]:
        """Get the IRLS working weights from final iteration."""
        try:
            return np.asarray(self._result.irls_weights)
        except AttributeError:
            return None
    
    def get_bread_matrix(self) -> Optional[np.ndarray]:
        """Get the (X'WX)^-1 matrix (unscaled covariance)."""
        try:
            return np.asarray(self._result.cov_params_unscaled)
        except AttributeError:
            return None
    
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
    def df_resid(self) -> float:
        """Residual degrees of freedom.
        
        For smooth models, uses n - total_edf instead of n - p,
        where total_edf accounts for the effective complexity of
        penalized smooth terms.
        """
        if self._total_edf is not None:
            return self._result.nobs - self._total_edf
        return self._result.df_resid
    
    @property
    def df_model(self) -> float:
        """Model degrees of freedom.
        
        For smooth models, uses total_edf - 1 (excluding intercept)
        instead of raw p - 1.
        """
        if self._total_edf is not None:
            return self._total_edf - 1
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
            raise ValidationError(f"Response column '{response}' not found in data")
        
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
            raise ValidationError(
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
        # Base predictions comparison
        base_predictions: Optional[str] = None,
    ) -> "ModelDiagnostics":
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
        base_predictions : str, optional
            Column name in train_data containing predictions from another model
            (e.g., a base/benchmark model). When provided, computes:
            - A/E ratio, loss, Gini for base predictions
            - Model vs base decile analysis sorted by model/base ratio
            - Summary of which model performs better in each decile
        
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
        >>> result = rs.glm_dict(response="ClaimNb", terms={"Age": {"type": "linear"}, "Region": {"type": "categorical"}}, data=data, family="poisson", offset="Exposure").fit()
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
        
        # Deserialized models lack covariance / design matrix — disable
        # features that depend on them to avoid AttributeErrors.
        if self._is_deserialized:
            compute_vif = False
            compute_coefficients = False
        
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
            base_predictions=base_predictions,
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
        >>> model = rs.glm_dict(response="ClaimNb", terms={"Age": {"type": "linear"}, "Region": {"type": "categorical"}}, data=data, family="poisson", offset="Exposure")
        >>> result = model.fit()
        >>> 
        >>> # Predict on new data
        >>> predictions = result.predict(new_data)
        >>> 
        >>> # Predict with custom offset
        >>> predictions = result.predict(new_data, offset=np.log(new_exposures))
        """
        if self._builder is None:
            raise PredictionError(
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
                if self._offset_is_exposure:
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
    
    def to_bytes(self) -> bytes:
        """
        Serialize the fitted model to bytes for storage or transfer.
        
        The serialized model can be loaded with `GLMModel.from_bytes()`.
        All state needed for prediction is preserved, including:
        - Coefficients and feature names
        - Categorical encoding levels
        - Spline knot positions
        - Target encoding statistics
        
        Returns
        -------
        bytes
            Serialized model as bytes.
            
        Examples
        --------
        >>> result = rs.glm_dict(response="y", terms={"x1": {"type": "linear"}, "cat": {"type": "categorical"}}, data=data, family="poisson").fit()
        >>> model_bytes = result.to_bytes()
        >>> 
        >>> # Save to file
        >>> with open("model.bin", "wb") as f:
        ...     f.write(model_bytes)
        >>> 
        >>> # Load later
        >>> with open("model.bin", "rb") as f:
        ...     loaded = rs.GLMModel.from_bytes(f.read())
        >>> predictions = loaded.predict(new_data)
        """
        import pickle
        
        # Extract state from the Rust result object
        # NOTE: We intentionally exclude fittedvalues and linear_predictor
        # as they are large arrays not needed for prediction (can be ~5MB each)
        result_state = {
            "params": np.asarray(self._result.params),
            "deviance": self._result.deviance,
            "iterations": self._result.iterations,
            "converged": self._result.converged,
            "nobs": self._result.nobs,
            "df_resid": self._result.df_resid,
            "df_model": self._result.df_model,
            "alpha": self._result.alpha,
            "l1_ratio": self._result.l1_ratio,
            "is_regularized": self._result.is_regularized,
            "penalty_type": self._result.penalty_type,
        }
        
        # Extract builder state for prediction
        builder_state = None
        if self._builder is not None:
            builder_state = {
                "parsed_formula": self._builder._parsed_formula,
                "cat_encoding_cache": self._builder._cat_encoding_cache,
                "fitted_splines": self._builder._fitted_splines,
                "te_stats": getattr(self._builder, "_te_stats", {}),
                "dtype": self._builder.dtype,
            }
        
        state = {
            "version": 1,
            "result_state": result_state,
            "feature_names": self.feature_names,
            "formula": self.formula,
            "family": self.family,
            "link": self.link,
            "builder_state": builder_state,
            "offset_spec": self._offset_spec,
            "offset_is_exposure": self._offset_is_exposure,
            "smooth_results": self._smooth_results,
            "total_edf": self._total_edf,
            "gcv": self._gcv,
            "terms_dict": self._terms_dict,
            "interactions_spec": self._interactions_spec,
        }
        
        return pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
    
    @classmethod
    def from_bytes(cls, data: bytes) -> "GLMModel":
        """
        Load a fitted model from bytes.
        
        Parameters
        ----------
        data : bytes
            Serialized model bytes from `to_bytes()`.
            
        Returns
        -------
        GLMModel
            Reconstructed fitted model ready for prediction.
            
        Examples
        --------
        >>> # Load from file
        >>> with open("model.bin", "rb") as f:
        ...     result = rs.GLMModel.from_bytes(f.read())
        >>> 
        >>> # Make predictions
        >>> predictions = result.predict(new_data)
        """
        import pickle
        
        state = pickle.loads(data)
        
        if state.get("version", 0) != 1:
            raise SerializationError(
                f"Unsupported serialization version: {state.get('version')}. "
                "Model was saved with a different version of rustystats."
            )
        
        result_state = state["result_state"]
        
        # Create a minimal result object that supports prediction
        result = _DeserializedResult(
            params=result_state["params"],
            deviance=result_state["deviance"],
            iterations=result_state["iterations"],
            converged=result_state["converged"],
            nobs=result_state["nobs"],
            df_resid=result_state["df_resid"],
            df_model=result_state["df_model"],
            alpha=result_state["alpha"],
            l1_ratio=result_state["l1_ratio"],
            is_regularized=result_state["is_regularized"],
            penalty_type=result_state["penalty_type"],
        )
        
        # Reconstruct builder if it was saved
        builder = None
        if state["builder_state"] is not None:
            builder = _DeserializedBuilder(state["builder_state"])
        
        return cls(
            result=result,
            feature_names=state["feature_names"],
            formula=state["formula"],
            family=state["family"],
            link=state["link"],
            builder=builder,
            design_matrix=None,
            offset_spec=state["offset_spec"],
            offset_is_exposure=state["offset_is_exposure"],
            regularization_path_info=None,
            smooth_results=state["smooth_results"],
            total_edf=state["total_edf"],
            gcv=state["gcv"],
            terms_dict=state.get("terms_dict"),
            interactions_spec=state.get("interactions_spec"),
        )
    
    def __repr__(self) -> str:
        return (
            f"<GLMModel: {self.family} family, "
            f"{len(self.params)} parameters, "
            f"deviance={self.deviance:.2f}>"
        )


# =============================================================================
# Dict-based API
# =============================================================================

from typing import Dict, Any, Set
from rustystats.interactions import (
    ParsedFormula, InteractionTerm, TargetEncodingTermSpec, 
    IdentityTermSpec, CategoricalTermSpec, ConstraintTermSpec,
    FrequencyEncodingTermSpec,
)
from rustystats.splines import SplineTerm
from rustystats.constants import (
    DEFAULT_SPLINE_DF,
    DEFAULT_SPLINE_DEGREE,
    DEFAULT_PRIOR_WEIGHT,
    DEFAULT_N_PERMUTATIONS,
)


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
    frequency_encoding_terms: Optional[List] = None,
) -> None:
    """Parse a single term specification and add to appropriate lists."""
    # Valid keys for each term type
    VALID_KEYS = {
        "linear": {"type", "monotonicity"},
        "categorical": {"type", "levels"},
        "bs": {"type", "df", "k", "degree", "monotonicity"},
        "ns": {"type", "df", "k"},
        "target_encoding": {"type", "prior_weight", "n_permutations", "variable"},
        "frequency_encoding": {"type", "variable"},
        "expression": {"type", "expr", "monotonicity"},
    }
    
    term_type = spec.get("type", "linear")
    
    # Validate keys
    valid_keys = VALID_KEYS.get(term_type, set())
    unknown_keys = set(spec.keys()) - valid_keys
    if unknown_keys:
        # Check for common typos
        typo_suggestions = {
            "monoticity": "monotonicity",
            "montonicity": "monotonicity",
            "increaing": "increasing",
            "decreaing": "decreasing",
        }
        suggestions = []
        for key in unknown_keys:
            if key in typo_suggestions:
                suggestions.append(f"'{key}' (did you mean '{typo_suggestions[key]}'?)")
            else:
                suggestions.append(f"'{key}'")
        raise ValidationError(
            f"Unknown key(s) in term spec for '{var_name}': {', '.join(suggestions)}. "
            f"Valid keys for type='{term_type}' are: {sorted(valid_keys)}"
        )
    
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
        # Default to penalized smooth (k=DEFAULT_SPLINE_DF) if neither df nor k specified
        k = spec.get("k")
        df = spec.get("df")
        if df is None and k is None:
            df = DEFAULT_SPLINE_DF  # Default: penalized smooth
            is_penalized = True
        elif k is not None:
            df = k
            is_penalized = True
        else:
            is_penalized = False
        degree = spec.get("degree", DEFAULT_SPLINE_DEGREE)
        term = SplineTerm(
            var_name=var_name,
            spline_type="bs",
            df=df,
            degree=degree,
            monotonicity=monotonicity,
        )
        if is_penalized:
            term._is_smooth = True
        if monotonicity:
            term._monotonic = True
        spline_terms.append(term)
    
    elif term_type == "ns":
        # Default to penalized smooth (k=DEFAULT_SPLINE_DF) if neither df nor k specified
        k = spec.get("k")
        df = spec.get("df")
        if df is None and k is None:
            df = DEFAULT_SPLINE_DF  # Default: penalized smooth
            is_penalized = True
        elif k is not None:
            df = k
            is_penalized = True
        else:
            is_penalized = False
        if monotonicity:
            raise ValidationError(
                f"Monotonicity constraints are not supported for natural splines (ns). "
                f"Use type='bs' with monotonicity parameter instead for monotonic effects."
            )
        term = SplineTerm(
            var_name=var_name,
            spline_type="ns",
            df=df,
        )
        if is_penalized:
            term._is_smooth = True
        spline_terms.append(term)
    
    elif term_type == "target_encoding":
        prior_weight = spec.get("prior_weight", DEFAULT_PRIOR_WEIGHT)
        n_permutations = spec.get("n_permutations", DEFAULT_N_PERMUTATIONS)
        # Single variable TE - use 'variable' key if provided
        # For TE interactions, use the interactions list with target_encoding: True
        actual_var = spec.get("variable", var_name)
        existing_te_vars = {te.var_name for te in target_encoding_terms}
        if actual_var not in existing_te_vars:
            target_encoding_terms.append(TargetEncodingTermSpec(
                var_name=actual_var,
                prior_weight=prior_weight,
                n_permutations=n_permutations,
            ))
    
    elif term_type == "frequency_encoding":
        from rustystats.interactions import FrequencyEncodingTermSpec as FETermSpec
        if frequency_encoding_terms is None:
            raise ValidationError(
                f"frequency_encoding type not supported in this context. "
                f"Use formula string 'FE({var_name})' instead."
            )
        # Use 'variable' key if provided, otherwise use the dict key
        actual_var = spec.get("variable", var_name)
        frequency_encoding_terms.append(FETermSpec(var_name=actual_var))
    
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
        raise ValidationError(f"Unknown term type: {term_type}")


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
    frequency_encoding_terms: Optional[List] = None,
) -> None:
    """Parse an interaction specification.
    
    Supports two modes:
    1. Standard interaction: creates product terms (cat×cat, cat×cont, etc.)
    2. Encoding interaction: combines variables into single encoded value
       - target_encoding: True → TE(var1:var2:...)
       - frequency_encoding: True → FE(var1:var2:...)
    """
    # Reserved keys (not variable specs)
    RESERVED_KEYS = {"include_main", "target_encoding", "frequency_encoding", 
                     "prior_weight", "n_permutations"}
    
    include_main = interaction.get("include_main", False)
    is_te_interaction = interaction.get("target_encoding", False)
    is_fe_interaction = interaction.get("frequency_encoding", False)
    
    if is_te_interaction and is_fe_interaction:
        raise ValidationError(
            "Cannot specify both target_encoding and frequency_encoding for same interaction"
        )
    
    # Extract variable specs (everything except reserved keys)
    var_specs = {k: v for k, v in interaction.items() if k not in RESERVED_KEYS}
    
    if len(var_specs) < 2:
        raise ValidationError("Interaction must have at least 2 variables")
    
    # Helper: track categorical vars and optionally add main effects
    def _process_encoding_interaction() -> None:
        for var_name, spec in var_specs.items():
            if spec.get("type", "categorical") == "categorical":
                categorical_vars.add(var_name)
        
        if include_main:
            for var_name, spec in var_specs.items():
                _parse_term_spec(
                    var_name, spec, categorical_vars, main_effects,
                    spline_terms, target_encoding_terms, identity_terms,
                    categorical_terms, constraint_terms, frequency_encoding_terms,
                )
    
    # Handle TE interaction: TE(var1:var2:...)
    if is_te_interaction:
        interaction_vars = list(var_specs.keys())
        target_encoding_terms.append(TargetEncodingTermSpec(
            var_name=":".join(interaction_vars),
            prior_weight=interaction.get("prior_weight", DEFAULT_PRIOR_WEIGHT),
            n_permutations=interaction.get("n_permutations", DEFAULT_N_PERMUTATIONS),
            interaction_vars=interaction_vars,
        ))
        _process_encoding_interaction()
        return
    
    # Handle FE interaction: FE(var1:var2:...)
    if is_fe_interaction:
        if frequency_encoding_terms is None:
            raise ValidationError(
                "frequency_encoding interaction not supported in this context"
            )
        interaction_vars = list(var_specs.keys())
        frequency_encoding_terms.append(FrequencyEncodingTermSpec(
            var_name=":".join(interaction_vars),
            interaction_vars=interaction_vars,
        ))
        _process_encoding_interaction()
        return
    
    # Standard interaction: product terms
    # Determine which factors are categorical, splines, or TE
    cat_factors = set()
    linear_factors = set()  # Factors explicitly typed as linear (no spline expansion)
    spline_factors = []
    te_factor_names = {}  # Maps original name -> TE(name) format
    
    for var_name, spec in var_specs.items():
        term_type = spec.get("type", "linear")
        
        if term_type == "linear":
            linear_factors.add(var_name)
        elif term_type == "categorical":
            cat_factors.add(var_name)
            categorical_vars.add(var_name)
        elif term_type in ("bs", "ns", "s"):
            # For s() smooth terms, use k parameter; for bs/ns use df
            if term_type == "s":
                df = spec.get("k", DEFAULT_SPLINE_DF)
            else:
                df = spec.get("df", 5 if term_type == "bs" else 4)
            degree = spec.get("degree", DEFAULT_SPLINE_DEGREE)
            monotonicity = spec.get("monotonicity")
            # Use unified bs with monotonicity parameter
            spline_type_out = "bs" if term_type == "s" else term_type
            spline = SplineTerm(
                var_name=var_name,
                spline_type=spline_type_out,
                df=df,
                degree=degree,
                monotonicity=monotonicity,
            )
            # Mark s() terms as smooth for penalized fitting
            if term_type == "s":
                spline._is_smooth = True
                if monotonicity:
                    spline._smooth_monotonicity = monotonicity
            spline_factors.append((var_name, spline))
        elif term_type == "target_encoding":
            prior_weight = spec.get("prior_weight", DEFAULT_PRIOR_WEIGHT)
            te_factor_names[var_name] = f"TE({var_name})"
            # TE in interaction - add to TE terms so encoding is available (if not already present)
            existing_te_vars = {te.var_name for te in target_encoding_terms}
            if var_name not in existing_te_vars:
                target_encoding_terms.append(TargetEncodingTermSpec(
                    var_name=var_name,
                    prior_weight=prior_weight,
                ))
    
    # Build factors list, renaming TE factors to TE(name) format
    factors = [te_factor_names.get(k, k) for k in var_specs.keys()]
    
    # Build interaction term - categorical_flags is a bool for each factor
    categorical_flags = [f in cat_factors for f in factors]
    
    interaction_term = InteractionTerm(
        factors=factors,
        categorical_flags=categorical_flags,
        force_linear=linear_factors if linear_factors else None,
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
    from rustystats.interactions import FrequencyEncodingTermSpec
    
    categorical_vars: Set[str] = set()
    main_effects: List[str] = []
    spline_terms_list: List[SplineTerm] = []
    target_encoding_terms_list: List[TargetEncodingTermSpec] = []
    frequency_encoding_terms_list: List[FrequencyEncodingTermSpec] = []
    identity_terms_list: List[IdentityTermSpec] = []
    categorical_terms_list: List[CategoricalTermSpec] = []
    constraint_terms_list: List[ConstraintTermSpec] = []
    interaction_terms_list: List[InteractionTerm] = []
    
    # Parse main terms
    for var_name, spec in terms.items():
        _parse_term_spec(
            var_name, spec, categorical_vars, main_effects,
            spline_terms_list, target_encoding_terms_list, identity_terms_list,
            categorical_terms_list, constraint_terms_list, frequency_encoding_terms_list,
        )
    
    # Parse interactions
    if interactions:
        for interaction in interactions:
            _parse_interaction_spec(
                interaction, interaction_terms_list, categorical_vars,
                main_effects, spline_terms_list, target_encoding_terms_list,
                identity_terms_list, categorical_terms_list, constraint_terms_list,
                frequency_encoding_terms_list,
            )
    
    return ParsedFormula(
        response=response,
        main_effects=main_effects,
        interactions=interaction_terms_list,
        categorical_vars=categorical_vars,
        spline_terms=spline_terms_list,
        target_encoding_terms=target_encoding_terms_list,
        frequency_encoding_terms=frequency_encoding_terms_list,
        identity_terms=identity_terms_list,
        categorical_terms=categorical_terms_list,
        constraint_terms=constraint_terms_list,
        has_intercept=intercept,
    )


class FormulaGLMDict(_GLMBase):
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
        # Store weak reference to data to allow garbage collection
        self._data_ref = weakref.ref(data)
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
                interaction = spec.get("interaction")
                if interaction:
                    term_strs.append(f"TE({':'.join(interaction)})")
                else:
                    term_strs.append(f"TE({var_name})")
            elif term_type == "frequency_encoding":
                term_strs.append(f"FE({var_name})")
            elif term_type == "expression":
                expr = spec.get("expr", var_name)
                term_strs.append(f"I({expr})")
        
        if not self.intercept:
            term_strs.insert(0, "0")
        
        parts.append(" + ".join(term_strs) if term_strs else "1")
        return " ".join(parts)
    
    def explore(
        self,
        categorical_factors: Optional[List[str]] = None,
        continuous_factors: Optional[List[str]] = None,
        n_bins: int = 10,
        rare_threshold_pct: float = 1.0,
        max_categorical_levels: int = 20,
        detect_interactions: bool = True,
        max_interaction_factors: int = 10,
    ) -> "DataExploration":
        """
        Explore data before fitting the model.
        
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
        """
        from rustystats.diagnostics import explore_data
        
        exposure_col = None
        if isinstance(self._offset_spec, str):
            exposure_col = self._offset_spec
        
        return explore_data(
            data=self.data,
            response=self.response,
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
    
    def fit(
        self,
        alpha: float = 0.0,
        l1_ratio: float = 0.0,
        max_iter: int = DEFAULT_MAX_ITER,
        tol: float = DEFAULT_TOLERANCE,
        # Cross-validation based regularization path parameters
        cv: Optional[int] = None,
        selection: str = "min",
        regularization: Optional[str] = None,
        n_alphas: int = DEFAULT_N_ALPHAS,
        alpha_min_ratio: float = DEFAULT_ALPHA_MIN_RATIO,
        cv_seed: Optional[int] = None,
        include_unregularized: bool = True,
        verbose: bool = False,
        # Memory optimization
        store_design_matrix: bool = True,
    ) -> GLMModel:
        """
        Fit the GLM model, optionally with regularization.
        
        Parameters
        ----------
        alpha : float, default=0.0
            Regularization strength. Higher values = more shrinkage.
            Ignored if regularization is specified (uses CV to find optimal).
            
        l1_ratio : float, default=0.0
            Elastic Net mixing parameter (0=Ridge, 1=Lasso).
            Ignored if regularization is specified with type.
            
        max_iter : int, default=25
            Maximum IRLS iterations.
        tol : float, default=1e-8
            Convergence tolerance.
            
        cv : int, optional
            Number of cross-validation folds. Defaults to 5 if regularization is set.
            
        selection : str, default="min"
            CV selection method: "min" or "1se".
            
        regularization : str, optional
            Type: "ridge", "lasso", or "elastic_net". Triggers CV-based alpha selection.
            
        n_alphas : int, default=20
            Number of alpha values in CV path.
            
        alpha_min_ratio : float, default=0.0001
            Smallest alpha as ratio of alpha_max.
            
        cv_seed : int, optional
            Random seed for CV folds.
            
        include_unregularized : bool, default=True
            Include alpha=0 in CV comparison.
            
        verbose : bool, default=False
            Print progress.
            
        Returns
        -------
        GLMModel
            Fitted model results.
        """
        is_negbinomial = is_negbinomial_family(self.family)
        
        # Handle CV-based regularization path (shared logic in _GLMBase)
        alpha, l1_ratio, path_info = self._resolve_cv_path(
            alpha, l1_ratio, max_iter, tol, cv, selection, regularization,
            n_alphas, alpha_min_ratio, cv_seed, include_unregularized, verbose,
        )
        
        theta = self.theta if self.theta is not None else DEFAULT_NEGBINOMIAL_THETA
        
        # Use shared core fitting logic
        result, smooth_results, total_edf, gcv = _fit_glm_core(
            self.y, self.X, self.family, self.link, self.var_power, theta,
            self.offset, self.weights, alpha, l1_ratio, max_iter, tol,
            self.feature_names, self._builder,
        )
        self._smooth_results = smooth_results
        self._total_edf = total_edf
        self._gcv = gcv
        
        result_family = f"NegativeBinomial(theta={theta:.4f})" if is_negbinomial else self.family
        
        # Wrap result with formula metadata
        is_exposure_offset = self.family in ("poisson", "quasipoisson", "negbinomial", "gamma") and self.link in (None, "log")
        return _build_results(
            result, self.feature_names, self.formula, result_family, self.link,
            self._builder, self.X, self._offset_spec, is_exposure_offset, path_info,
            self._smooth_results, self._total_edf, self._gcv,
            store_design_matrix=store_design_matrix,
            terms_dict=self.terms,
            interactions_spec=self.interactions_spec,
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
