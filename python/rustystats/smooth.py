"""
Penalized smooth terms for Generalized Additive Models (GAMs).

This module provides the `s()` function for creating smooth terms with automatic
smoothness selection via GCV. Unlike fixed-df splines (bs, ns), smooth terms
use a larger basis with a penalty on wiggliness, allowing the data to determine
the effective complexity.

Key Functions
-------------
- `s()` - Create a smooth term with automatic lambda selection
- `penalty_matrix()` - Compute the difference penalty matrix S = D'D
- `difference_matrix()` - Compute the difference operator D
- `gcv_score()` - Generalized cross-validation score
- `compute_edf()` - Effective degrees of freedom

Example
-------
>>> import rustystats as rs
>>> import numpy as np
>>> 
>>> # Create a smooth term for age with 10 basis functions
>>> age = np.array([25, 35, 45, 55, 65])
>>> basis, penalty = rs.s(age, k=10)
>>> print(f"Basis shape: {basis.shape}")
>>> print(f"Penalty shape: {penalty.shape}")

>>> # Use in formula API
>>> result = rs.glm(
...     "y ~ s(age) + s(income, k=15) + C(region)",
...     data=data,
...     family="poisson"
... ).fit()

Comparison with Fixed Splines
-----------------------------
**Fixed splines (`bs`, `ns`):**
- You choose df (e.g., df=5)
- Exactly df degrees of freedom used
- Risk of under/over-fitting if df is wrong

**Smooth terms (`s`):**
- You choose k (e.g., k=10) - max basis size
- Effective df chosen automatically via GCV
- EDF typically 2-8, data-driven
- More robust to df specification

Mathematical Background
-----------------------
A smooth term s(x) is represented as:

    f(x) = Σ βⱼ Bⱼ(x)

where Bⱼ are B-spline basis functions. To control smoothness, we add a
penalty on the second differences of coefficients:

    Penalty = λ × β' S β

where S = D'D and D is a difference matrix. The smoothing parameter λ
is selected to minimize GCV:

    GCV(λ) = n × Deviance / (n - EDF)²

where EDF = trace((X'WX + λS)⁻¹ X'WX) measures effective model complexity.
"""

from __future__ import annotations

from typing import Optional, List, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    pass

from rustystats.splines import bs


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
        return float(xtwx.shape[0])


def s(
    x: np.ndarray,
    k: int = 10,
    penalty_order: int = 2,
    boundary_knots: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a smooth term basis and penalty matrix.
    
    This is the main function for creating penalized smooth terms. It returns
    both the B-spline basis matrix and the penalty matrix for use in GAM fitting.
    
    Parameters
    ----------
    x : array-like
        Data values to create the basis for.
    k : int, default=10
        Number of basis functions. This is the maximum complexity;
        the effective complexity (EDF) will be determined by GCV.
        Typical range: 5-20. More complex effects may need k=15-30.
    penalty_order : int, default=2
        Order of the difference penalty. Order 2 (default) penalizes
        second differences, encouraging smooth curves. Order 1 penalizes
        first differences, encouraging piecewise constant functions.
    boundary_knots : tuple, optional
        (min, max) boundary for the spline. If not provided, uses the
        range of x.
    
    Returns
    -------
    basis : np.ndarray
        B-spline basis matrix of shape (n, k-1)
    penalty : np.ndarray
        Penalty matrix of shape (k-1, k-1)
    
    Notes
    -----
    The returned basis has k-1 columns (not k) because the intercept column
    is dropped for use in models that already have an intercept.
    
    Examples
    --------
    >>> import numpy as np
    >>> import rustystats as rs
    >>> 
    >>> x = np.linspace(0, 10, 100)
    >>> basis, penalty = rs.s(x, k=10)
    >>> 
    >>> print(f"Basis shape: {basis.shape}")  # (100, 9)
    >>> print(f"Penalty shape: {penalty.shape}")  # (9, 9)
    >>> 
    >>> # Penalty matrix is symmetric positive semi-definite
    >>> eigenvalues = np.linalg.eigvalsh(penalty)
    >>> print(f"Min eigenvalue: {eigenvalues.min():.6f}")  # ≥ 0
    
    See Also
    --------
    bs : B-spline basis (fixed df, no penalty)
    ns : Natural spline basis (fixed df, linear at boundaries)
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    
    # Create B-spline basis with k degrees of freedom
    # include_intercept=False drops the first column
    basis = bs(x, df=k, degree=3, boundary_knots=boundary_knots, include_intercept=False)
    
    # Number of columns in basis (k-1 because no intercept)
    n_cols = basis.shape[1]
    
    # Create penalty matrix for the actual basis size
    penalty = penalty_matrix(n_cols, order=penalty_order)
    
    return basis, penalty


def smooth_names(var_name: str, k: int) -> List[str]:
    """
    Generate column names for smooth term basis functions.
    
    Parameters
    ----------
    var_name : str
        Name of the original variable (e.g., "age")
    k : int
        Number of basis functions
    
    Returns
    -------
    list of str
        Names like ['s(age, 1/10)', 's(age, 2/10)', ...]
    
    Example
    -------
    >>> smooth_names("age", k=10)
    ['s(age, 1/10)', 's(age, 2/10)', ..., 's(age, 9/10)']
    """
    # k-1 columns because intercept is dropped
    return [f"s({var_name}, {i+1}/{k})" for i in range(k - 1)]


class SmoothTerm:
    """
    Represents a smooth term for use in formula parsing.
    
    This class stores the specification for a smooth transformation
    and can compute the basis matrix and penalty matrix when given data.
    
    Attributes
    ----------
    var_name : str
        Name of the variable to transform
    k : int
        Number of basis functions
    penalty_order : int
        Order of the difference penalty
    boundary_knots : tuple or None
        Boundary knot positions
    
    Examples
    --------
    >>> term = SmoothTerm("age", k=10)
    >>> basis, penalty, names = term.transform(age_data)
    """
    
    def __init__(
        self,
        var_name: str,
        k: int = 10,
        penalty_order: int = 2,
        boundary_knots: Optional[Tuple[float, float]] = None,
    ):
        self.var_name = var_name
        self.k = k
        self.penalty_order = penalty_order
        self.boundary_knots = boundary_knots
        
        # Computed during transform - stores knot information
        self._computed_boundary_knots: Optional[Tuple[float, float]] = None
        
        # Store lambda and EDF after fitting
        self.lambda_: Optional[float] = None
        self.edf: Optional[float] = None
    
    def transform(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Compute the smooth term basis and penalty for the given data.
        
        Parameters
        ----------
        x : np.ndarray
            Data values to transform
        
        Returns
        -------
        basis : np.ndarray
            Basis matrix
        penalty : np.ndarray
            Penalty matrix
        names : list of str
            Column names for the basis
        """
        x_arr = np.asarray(x).ravel()
        
        # Compute boundary knots if not already set
        if self._computed_boundary_knots is None:
            if self.boundary_knots is not None:
                self._computed_boundary_knots = self.boundary_knots
            else:
                self._computed_boundary_knots = (float(np.min(x_arr)), float(np.max(x_arr)))
        
        # Create basis and penalty
        basis, penalty = s(
            x_arr,
            k=self.k,
            penalty_order=self.penalty_order,
            boundary_knots=self._computed_boundary_knots,
        )
        
        names = smooth_names(self.var_name, self.k)
        
        # Ensure names match columns
        if len(names) != basis.shape[1]:
            names = [f"s({self.var_name}, {i+1}/{basis.shape[1]+1})" 
                    for i in range(basis.shape[1])]
        
        return basis, penalty, names
    
    def get_info(self) -> dict:
        """
        Get information about this smooth term.
        
        Returns
        -------
        dict
            Dictionary with term configuration and fitted values
        """
        info = {
            "type": "smooth",
            "variable": self.var_name,
            "k": self.k,
            "penalty_order": self.penalty_order,
        }
        if self._computed_boundary_knots is not None:
            info["boundary_knots"] = list(self._computed_boundary_knots)
        if self.lambda_ is not None:
            info["lambda"] = self.lambda_
        if self.edf is not None:
            info["edf"] = self.edf
        return info
    
    def __repr__(self) -> str:
        return f"s({self.var_name}, k={self.k})"
