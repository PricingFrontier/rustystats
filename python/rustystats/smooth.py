"""Penalty matrices and utility functions for penalized spline fitting.

This module provides utilities for penalized spline regression (P-splines),
including penalty matrix construction, GCV scoring, and EDF computation.

Key Functions
-------------
- `penalty_matrix()` - Compute the difference penalty matrix S = D'D
- `difference_matrix()` - Compute the difference operator D
- `gcv_score()` - Generalized cross-validation score
- `compute_edf()` - Effective degrees of freedom

Example
-------
>>> import rustystats as rs
>>> import numpy as np
>>> 
>>> # Use penalized splines in formula API with k parameter
>>> result = rs.glm(
...     "y ~ bs(age, k=10) + bs(income, k=15) + C(region)",
...     data=data,
...     family="poisson"
... ).fit()
>>> 
>>> # Check effective degrees of freedom
>>> if result.has_smooth_terms():
...     for st in result.smooth_terms:
...         print(f"{st.variable}: EDF={st.edf:.2f}")

Mathematical Background
-----------------------
Penalized splines control smoothness by adding a penalty on coefficients:

    Penalty = λ × β' S β

where S = D'D and D is a difference matrix. The smoothing parameter λ
is selected to minimize GCV:

    GCV(λ) = n × Deviance / (n - EDF)²

where EDF = trace((X'WX + λS)⁻¹ X'WX) measures effective model complexity.
"""

from __future__ import annotations

from typing import Optional
import numpy as np


def difference_matrix(k: int, order: int = 2) -> np.ndarray:
    """
    Compute a difference matrix of given order.
    
    For a coefficient vector β of length k, the difference matrix D computes:
    - Order 1: Dβ = [β₁-β₀, β₂-β₁, ..., β_{k-1}-β_{k-2}]
    - Order 2: Dβ = [β₂-2β₁+β₀, β₃-2β₂+β₁, ...]
    
    Parameters
    ----------
    k : int
        Number of coefficients (columns in D)
    order : int, default=2
        Difference order. Order 2 is standard for smoothness penalties.
    
    Returns
    -------
    np.ndarray
        Difference matrix of shape (k-order, k)
    
    Examples
    --------
    >>> D1 = difference_matrix(5, order=1)
    >>> print(D1)
    [[-1  1  0  0  0]
     [ 0 -1  1  0  0]
     [ 0  0 -1  1  0]
     [ 0  0  0 -1  1]]
    
    >>> D2 = difference_matrix(5, order=2)
    >>> print(D2)
    [[ 1 -2  1  0  0]
     [ 0  1 -2  1  0]
     [ 0  0  1 -2  1]]
    """
    if order == 0:
        return np.eye(k)
    
    if k <= order:
        return np.zeros((0, k))
    
    if order == 1:
        n_rows = k - 1
        D = np.zeros((n_rows, k))
        for i in range(n_rows):
            D[i, i] = -1
            D[i, i + 1] = 1
        return D
    
    # Higher orders: D_m = D_1 @ D_{m-1}
    D1 = difference_matrix(k, 1)
    D_prev = difference_matrix(k - 1, order - 1)
    return D_prev @ D1[:k-1, :]


def penalty_matrix(k: int, order: int = 2) -> np.ndarray:
    """
    Compute the penalty matrix S = D'D for smoothness regularization.
    
    The penalty on coefficients β is: β' S β = ||D β||²
    
    This penalizes the sum of squared differences of the given order,
    encouraging smooth functions.
    
    Parameters
    ----------
    k : int
        Number of basis functions
    order : int, default=2
        Difference order. Order 2 penalizes second differences,
        which corresponds to penalizing curvature.
    
    Returns
    -------
    np.ndarray
        Penalty matrix of shape (k, k), symmetric positive semi-definite.
    
    Properties
    ----------
    - S is symmetric
    - S is positive semi-definite (eigenvalues ≥ 0)
    - S has `order` zero eigenvalues (null space = polynomials of degree < order)
    - For order=2: null space is {1, x} (constant and linear)
    
    Examples
    --------
    >>> S = penalty_matrix(5, order=2)
    >>> print(S.shape)
    (5, 5)
    >>> 
    >>> # S is symmetric
    >>> np.allclose(S, S.T)
    True
    >>> 
    >>> # Constant vector is in null space
    >>> beta = np.ones(5)
    >>> print(beta @ S @ beta)  # Close to 0
    """
    D = difference_matrix(k, order)
    return D.T @ D


def gcv_score(deviance: float, n: int, edf: float) -> float:
    """
    Compute the Generalized Cross-Validation score.
    
    GCV(λ) = n × Deviance / (n - EDF)²
    
    Lower GCV is better. This approximates leave-one-out cross-validation
    without requiring refitting.
    
    Parameters
    ----------
    deviance : float
        Model deviance at current λ
    n : int
        Number of observations
    edf : float
        Effective degrees of freedom
    
    Returns
    -------
    float
        GCV score (lower is better)
    
    Examples
    --------
    >>> gcv = gcv_score(deviance=100.0, n=1000, edf=10.0)
    >>> print(f"GCV: {gcv:.4f}")
    """
    denominator = max(n - edf, 1.0)
    return n * deviance / (denominator ** 2)


def compute_edf(xtwx: np.ndarray, penalty: np.ndarray, lambda_: float) -> float:
    """
    Compute effective degrees of freedom for a penalized fit.
    
    EDF = trace((X'WX + λS)⁻¹ X'WX)
    
    Parameters
    ----------
    xtwx : np.ndarray
        X'WX matrix (p × p)
    penalty : np.ndarray
        Penalty matrix S (p × p)
    lambda_ : float
        Smoothing parameter
    
    Returns
    -------
    float
        Effective degrees of freedom
    
    Notes
    -----
    - EDF ≈ k (basis size) when λ ≈ 0 (no penalty)
    - EDF ≈ order when λ → ∞ (maximum penalty, polynomial)
    """
    if lambda_ <= 0:
        return float(xtwx.shape[0])
    
    xtwx_pen = xtwx + lambda_ * penalty
    try:
        xtwx_pen_inv = np.linalg.inv(xtwx_pen)
        hat_matrix = xtwx_pen_inv @ xtwx
        return np.trace(hat_matrix)
    except np.linalg.LinAlgError:
        import warnings
        warnings.warn(
            f"EDF computation failed due to singular matrix (lambda={lambda_:.4f}). "
            "Returning NaN. Consider using a larger lambda or reducing basis size.",
            RuntimeWarning,
            stacklevel=2
        )
        return float('nan')
