"""
Shared utility functions for diagnostics modules.

Contains common computation helpers used by both DiagnosticsComputer
(post-fit) and DataExplorer (pre-fit) to avoid duplication.
"""

from __future__ import annotations

import numpy as np

from rustystats.exceptions import ValidationError


def discretize(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Discretize values into bins.
    
    For categorical arrays, maps unique values to integer indices.
    For continuous arrays, uses quantile-based binning.
    
    Parameters
    ----------
    values : np.ndarray
        Values to discretize.
    n_bins : int
        Number of bins for continuous values.
        
    Returns
    -------
    np.ndarray
        Integer bin assignments.
    """
    # Check if array is non-numeric (object or string dtype)
    is_string_like = values.dtype.kind in ('O', 'U', 'S')  # object, unicode, bytes
    
    if is_string_like:
        # Try to coerce to float64 - handles Decimal, mixed numeric, and None/null
        try:
            values = np.asarray(values, dtype=np.float64)
        except (ValueError, TypeError):
            # Truly categorical - map to integers
            # Filter None/null for sorting, assign them to a separate bin
            non_null_mask = np.array([v is not None for v in values])
            if not np.any(non_null_mask):
                return np.zeros(len(values), dtype=int)
            
            unique_vals = np.unique(values[non_null_mask])
            mapping = {v: i for i, v in enumerate(unique_vals)}
            result = np.array([mapping.get(v, len(unique_vals)) for v in values])
            return result
    
    # Continuous - quantile bins
    values = values.astype(np.float64)
    valid_mask = ~np.isnan(values) & ~np.isinf(values)
    
    if not np.any(valid_mask):
        return np.zeros(len(values), dtype=int)
    
    quantiles = np.percentile(values[valid_mask], np.linspace(0, 100, n_bins + 1))
    bins = np.digitize(values, quantiles[1:-1])
    bins[~valid_mask] = n_bins  # Invalid values in separate bin
    return bins


def validate_factor_in_data(name: str, data, factor_type: str = "Factor") -> None:
    """Validate that a factor column exists in the data.
    
    Parameters
    ----------
    name : str
        Column name to check.
    data : pl.DataFrame
        DataFrame to check against.
    factor_type : str
        Label for error message (e.g., "Categorical factor", "Continuous factor").
        
    Raises
    ------
    ValueError
        If the column is not found in the data.
    """
    if name not in data.columns:
        raise ValidationError(
            f"{factor_type} '{name}' not found in data columns: {list(data.columns)}"
        )
