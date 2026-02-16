"""
Input validation for GLM fitting.

Provides comprehensive validation of response, predictors, weights, and offsets
before fitting to catch common data issues early with actionable error messages.
Follows patterns from statsmodels and scikit-learn.
"""

from __future__ import annotations

import warnings

import numpy as np

from rustystats.exceptions import ValidationError


__all__ = [
    "validate_glm_inputs",
    "validate_response",
    "validate_design_matrix",
    "validate_weights",
    "validate_offset",
    "coerce_to_float64",
]


# =============================================================================
# Array Coercion
# =============================================================================

def coerce_to_float64(
    arr: np.ndarray,
    name: str = "array",
    allow_nan: bool = False,
    allow_inf: bool = False,
) -> np.ndarray:
    """
    Coerce array to float64, handling Decimal and other numeric types.
    
    Parameters
    ----------
    arr : np.ndarray
        Input array (may be object dtype with Decimal, mixed types, etc.)
    name : str
        Name for error messages.
    allow_nan : bool
        If False, raises on NaN values.
    allow_inf : bool
        If False, raises on Inf values.
        
    Returns
    -------
    np.ndarray
        Array coerced to float64.
        
    Raises
    ------
    ValidationError
        If array cannot be coerced or contains invalid values.
    """
    # Try to coerce to float64
    try:
        result = np.asarray(arr, dtype=np.float64)
    except (ValueError, TypeError) as e:
        raise ValidationError(
            f"{name} cannot be converted to numeric values. "
            f"Ensure all values are numeric (int, float, Decimal). Error: {e}"
        )
    
    # Check for NaN
    nan_count = np.isnan(result).sum()
    if nan_count > 0 and not allow_nan:
        nan_pct = 100 * nan_count / len(result)
        raise ValidationError(
            f"{name} contains {nan_count} NaN values ({nan_pct:.1f}%). "
            "Either remove rows with missing values or impute them before fitting."
        )
    
    # Check for Inf
    inf_count = np.isinf(result).sum()
    if inf_count > 0 and not allow_inf:
        raise ValidationError(
            f"{name} contains {inf_count} infinite values. "
            "Replace Inf/-Inf with finite values or remove those rows."
        )
    
    return result


# =============================================================================
# Response Validation by Family
# =============================================================================

def validate_response(
    y: np.ndarray,
    family: str,
    name: str = "response (y)",
) -> np.ndarray:
    """
    Validate response variable for the specified GLM family.
    
    Parameters
    ----------
    y : np.ndarray
        Response variable.
    family : str
        GLM family name (gaussian, binomial, poisson, gamma, etc.)
    name : str
        Name for error messages.
        
    Returns
    -------
    np.ndarray
        Validated response as float64.
        
    Raises
    ------
    ValidationError
        If response values are invalid for the family.
    """
    y = coerce_to_float64(y, name)
    n = len(y)
    family_lower = family.lower()
    
    # Check for empty response
    if n == 0:
        raise ValidationError(f"{name} is empty. Cannot fit model with no observations.")
    
    # Check for constant response
    if np.all(y == y[0]):
        raise ValidationError(
            f"{name} is constant (all values = {y[0]}). "
            "A GLM requires variation in the response variable."
        )
    
    # Family-specific validation
    if family_lower in ("binomial", "quasibinomial"):
        _validate_binomial_response(y, name)
    
    elif family_lower in ("poisson", "quasipoisson"):
        _validate_poisson_response(y, name)
    
    elif family_lower == "gamma":
        _validate_gamma_response(y, name)
    
    elif family_lower in ("negbinomial", "negativebinomial", "nb", "negative_binomial"):
        _validate_negbinomial_response(y, name)
    
    elif family_lower == "tweedie":
        _validate_tweedie_response(y, name)
    
    elif family_lower == "inverse_gaussian":
        _validate_inverse_gaussian_response(y, name)
    
    # gaussian and others: no special constraints
    
    return y


def _validate_binomial_response(y: np.ndarray, name: str) -> None:
    """Validate response for binomial family: must be in [0, 1]."""
    if np.any(y < 0) or np.any(y > 1):
        n_invalid = np.sum((y < 0) | (y > 1))
        y_min, y_max = y.min(), y.max()
        raise ValidationError(
            f"Binomial family requires {name} in [0, 1] (proportions or 0/1 binary). "
            f"Found {n_invalid} values outside this range (min={y_min:.4g}, max={y_max:.4g}). "
            "For count data (successes/trials), divide by trials to get proportions."
        )
    
    # Warn if not truly binary
    unique_vals = np.unique(y)
    if len(unique_vals) == 2 and not np.allclose(unique_vals, [0, 1]):
        warnings.warn(
            f"Binomial response has 2 unique values {unique_vals} but not exactly [0, 1]. "
            "Consider recoding to 0/1 for clarity.",
            UserWarning
        )


def _validate_poisson_response(y: np.ndarray, name: str) -> None:
    """Validate response for Poisson family: must be non-negative integers."""
    if np.any(y < 0):
        n_neg = np.sum(y < 0)
        raise ValidationError(
            f"Poisson family requires non-negative {name} (counts). "
            f"Found {n_neg} negative values. "
            "Poisson models count data; negative counts are impossible."
        )
    
    # Warn if not integer-valued (common with aggregated data)
    if not np.allclose(y, np.round(y)):
        warnings.warn(
            f"Poisson {name} contains non-integer values. "
            "Poisson is designed for count data. Consider quasipoisson for overdispersed "
            "or continuous-valued count-like data.",
            UserWarning
        )


def _validate_gamma_response(y: np.ndarray, name: str) -> None:
    """Validate response for Gamma family: must be strictly positive."""
    if np.any(y <= 0):
        n_invalid = np.sum(y <= 0)
        raise ValidationError(
            f"Gamma family requires strictly positive {name} (y > 0). "
            f"Found {n_invalid} values <= 0. "
            "Gamma models positive continuous data like claim amounts or durations."
        )


def _validate_negbinomial_response(y: np.ndarray, name: str) -> None:
    """Validate response for Negative Binomial: non-negative, typically counts."""
    if np.any(y < 0):
        n_neg = np.sum(y < 0)
        raise ValidationError(
            f"Negative Binomial family requires non-negative {name}. "
            f"Found {n_neg} negative values."
        )


def _validate_tweedie_response(y: np.ndarray, name: str) -> None:
    """Validate response for Tweedie: must be non-negative."""
    if np.any(y < 0):
        n_neg = np.sum(y < 0)
        raise ValidationError(
            f"Tweedie family requires non-negative {name}. "
            f"Found {n_neg} negative values."
        )


def _validate_inverse_gaussian_response(y: np.ndarray, name: str) -> None:
    """Validate response for Inverse Gaussian: strictly positive."""
    if np.any(y <= 0):
        n_invalid = np.sum(y <= 0)
        raise ValidationError(
            f"Inverse Gaussian family requires strictly positive {name} (y > 0). "
            f"Found {n_invalid} values <= 0."
        )


# =============================================================================
# Design Matrix Validation
# =============================================================================

def validate_design_matrix(
    X: np.ndarray,
    feature_names: Optional[List[str]] = None,
    name: str = "design matrix (X)",
) -> np.ndarray:
    """
    Validate design matrix for common issues.
    
    Parameters
    ----------
    X : np.ndarray
        Design matrix (n_obs x n_features).
    feature_names : list of str, optional
        Feature names for better error messages.
    name : str
        Name for error messages.
        
    Returns
    -------
    np.ndarray
        Validated design matrix as float64.
    """
    X = coerce_to_float64(X, name)
    
    if X.ndim != 2:
        raise ValidationError(
            f"{name} must be 2-dimensional (n_obs x n_features). Got shape {X.shape}."
        )
    
    n_obs, n_features = X.shape
    
    if n_obs == 0:
        raise ValidationError(f"{name} has no observations.")
    
    if n_features == 0:
        raise ValidationError(f"{name} has no features.")
    
    if n_obs < n_features:
        warnings.warn(
            f"{name} has fewer observations ({n_obs}) than features ({n_features}). "
            "Model will be underdetermined. Consider regularization (alpha > 0).",
            UserWarning
        )
    
    # Check for NaN/Inf (should be caught by coerce_to_float64, but double-check)
    nan_mask = np.isnan(X)
    if nan_mask.any():
        nan_cols = np.where(nan_mask.any(axis=0))[0]
        col_names = [feature_names[i] if feature_names else f"column {i}" for i in nan_cols[:5]]
        raise ValidationError(
            f"{name} contains NaN values in columns: {col_names}. "
            "Remove or impute missing values before fitting."
        )
    
    inf_mask = np.isinf(X)
    if inf_mask.any():
        inf_cols = np.where(inf_mask.any(axis=0))[0]
        col_names = [feature_names[i] if feature_names else f"column {i}" for i in inf_cols[:5]]
        raise ValidationError(
            f"{name} contains infinite values in columns: {col_names}. "
            "Replace Inf/-Inf with finite values."
        )
    
    return X


# =============================================================================
# Weights Validation
# =============================================================================

def validate_weights(
    weights: Optional[np.ndarray],
    n_obs: int,
    name: str = "weights",
) -> Optional[np.ndarray]:
    """
    Validate observation weights.
    
    Parameters
    ----------
    weights : np.ndarray or None
        Observation weights.
    n_obs : int
        Expected number of observations.
    name : str
        Name for error messages.
        
    Returns
    -------
    np.ndarray or None
        Validated weights as float64, or None if no weights.
    """
    if weights is None:
        return None
    
    weights = coerce_to_float64(weights, name)
    
    if len(weights) != n_obs:
        raise ValidationError(
            f"{name} length ({len(weights)}) does not match number of observations ({n_obs})."
        )
    
    # Check for negative weights
    if np.any(weights < 0):
        n_neg = np.sum(weights < 0)
        raise ValidationError(
            f"{name} contains {n_neg} negative values. "
            "Weights must be non-negative."
        )
    
    # Check for all-zero weights
    if np.sum(weights) == 0:
        raise ValidationError(
            f"{name} sum to zero. At least some observations must have positive weight."
        )
    
    # Warn if many zero weights
    n_zero = np.sum(weights == 0)
    if n_zero > 0:
        pct_zero = 100 * n_zero / n_obs
        if pct_zero > 50:
            warnings.warn(
                f"{pct_zero:.1f}% of {name} are zero. "
                "These observations will not contribute to the fit.",
                UserWarning
            )
    
    return weights


# =============================================================================
# Offset/Exposure Validation
# =============================================================================

def validate_offset(
    offset: Optional[np.ndarray],
    n_obs: int,
    family: str,
    is_exposure: bool = False,
    name: str = "offset",
) -> Optional[np.ndarray]:
    """
    Validate offset or exposure.
    
    Parameters
    ----------
    offset : np.ndarray or None
        Offset values (or exposure for Poisson/Gamma).
    n_obs : int
        Expected number of observations.
    family : str
        GLM family name.
    is_exposure : bool
        If True, validates as exposure (must be positive for log-link families).
    name : str
        Name for error messages.
        
    Returns
    -------
    np.ndarray or None
        Validated offset as float64, or None.
    """
    if offset is None:
        return None
    
    offset = coerce_to_float64(offset, name)
    
    if len(offset) != n_obs:
        raise ValidationError(
            f"{name} length ({len(offset)}) does not match number of observations ({n_obs})."
        )
    
    # For exposure with log-link families, must be strictly positive
    family_lower = family.lower()
    if is_exposure and family_lower in ("poisson", "quasipoisson", "gamma", "negbinomial", 
                                         "negativebinomial", "nb", "negative_binomial"):
        if np.any(offset <= 0):
            n_invalid = np.sum(offset <= 0)
            raise ValidationError(
                f"Exposure must be strictly positive for {family} family with log link. "
                f"Found {n_invalid} values <= 0. "
                "Exposure represents the denominator (e.g., time, population) and cannot be zero or negative."
            )
    
    return offset


# =============================================================================
# Combined Validation
# =============================================================================

def validate_glm_inputs(
    y: np.ndarray,
    X: np.ndarray,
    family: str,
    weights: Optional[np.ndarray] = None,
    offset: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    is_exposure_offset: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Validate all GLM inputs before fitting.
    
    This is the main entry point for validation. Call this before passing
    data to the Rust fitting engine.
    
    Parameters
    ----------
    y : np.ndarray
        Response variable.
    X : np.ndarray
        Design matrix.
    family : str
        GLM family name.
    weights : np.ndarray, optional
        Observation weights.
    offset : np.ndarray, optional  
        Offset or exposure.
    feature_names : list of str, optional
        Feature names for error messages.
    is_exposure_offset : bool
        If True, offset is treated as exposure (validated as positive).
        
    Returns
    -------
    tuple
        (y, X, weights, offset) all validated and coerced to float64.
        
    Raises
    ------
    ValidationError
        If any validation check fails.
    """
    # Validate response
    y = validate_response(y, family)
    n_obs = len(y)
    
    # Validate design matrix
    X = validate_design_matrix(X, feature_names)
    
    # Check dimension match
    if X.shape[0] != n_obs:
        raise ValidationError(
            f"Response has {n_obs} observations but design matrix has {X.shape[0]} rows. "
            "They must match."
        )
    
    # Validate weights
    weights = validate_weights(weights, n_obs)
    
    # Validate offset/exposure
    offset = validate_offset(
        offset, n_obs, family, 
        is_exposure=is_exposure_offset,
        name="exposure" if is_exposure_offset else "offset"
    )
    
    return y, X, weights, offset
