"""
Spline basis functions for non-linear continuous effects in GLMs.

This module provides B-splines and natural splines for modeling non-linear
relationships between continuous predictors and the response variable.

Key Functions
-------------
- `bs()` - B-spline basis (flexible piecewise polynomials)
- `ns()` - Natural spline basis (linear extrapolation at boundaries)

Parameters
----------
Both `bs()` and `ns()` support two modes:

**Default (penalized smooth):** No parameters = auto-tuned via GCV
    >>> basis = rs.bs(x)  # Defaults to k=10, penalized smooth

**Fixed df mode:** Use `df` parameter for fixed degrees of freedom
    >>> basis = rs.bs(x, df=5)  # Exactly 5 df, no penalty

**Explicit penalized smooth:** Use `k` parameter 
    >>> basis = rs.bs(x, k=15)  # 15 basis functions, penalized

**Monotonicity (bs only):** Use `monotonicity` parameter for constrained effects
    >>> basis = rs.bs(x, df=5, monotonicity="increasing")

Example
-------
>>> import rustystats as rs
>>> import numpy as np
>>> 
>>> # Fixed df B-spline
>>> age = np.array([25, 35, 45, 55, 65])
>>> age_basis = rs.bs(age, df=5)
>>> print(age_basis.shape)
(5, 4)

>>> # Penalized smooth (auto-tuned via GCV during fitting)
>>> result = rs.glm("y ~ bs(age, k=10)", data=data, family="poisson").fit()

>>> # Monotonically increasing effect
>>> result = rs.glm("y ~ bs(age, df=5, monotonicity='increasing')", data=data).fit()

When to Use Each Type
---------------------
**B-splines (`bs`):**
- More flexible at boundaries
- Good when you don't need to extrapolate
- Standard choice for most applications
- Supports monotonicity constraints via I-spline basis

**Natural splines (`ns`):**
- Linear extrapolation beyond boundaries
- Better for prediction on new data outside training range
- More stable parameter estimates at boundaries
- Recommended for actuarial applications

Performance Note
----------------
Spline basis computation is implemented in Rust with parallel
evaluation over observations, making it very fast even for
large datasets.

"""

from __future__ import annotations

from typing import Optional, Union, List, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    import polars as pl

# Import Rust implementations
from rustystats._rustystats import (
    bs_py as _bs_rust,
    ns_py as _ns_rust,
    ns_with_knots_py as _ns_with_knots_rust,
    bs_knots_py as _bs_knots_rust,
    bs_names_py as _bs_names_rust,
    ns_names_py as _ns_names_rust,
    ms_py as _ms_rust,  # Used internally by bs() with monotonicity parameter
    ms_with_knots_py as _ms_with_knots_rust,  # Monotonic splines with explicit knots
)


def bs(
    x: np.ndarray,
    df: Optional[int] = None,
    k: Optional[int] = None,
    degree: int = 3,
    knots: Optional[List[float]] = None,
    boundary_knots: Optional[Tuple[float, float]] = None,
    include_intercept: bool = False,
    monotonicity: Optional[str] = None,
) -> np.ndarray:
    """
    Compute B-spline basis matrix.
    
    B-splines (basis splines) are piecewise polynomial functions that provide
    a flexible way to model non-linear relationships. They are the foundation
    for many modern smoothing techniques.
    
    Parameters
    ----------
    x : array-like
        Data points to evaluate the basis at. Will be converted to 1D numpy array.
    df : int, optional
        Degrees of freedom (fixed). Use this for fixed-complexity splines.
    k : int, optional
        Maximum basis size for penalized smooth terms. When k is provided,
        the spline becomes a penalized smooth term with automatic smoothness
        selection via GCV during model fitting. Typical range: 5-20.
        **Default**: If neither df nor k is provided, defaults to k=10
        (penalized smooth with automatic tuning).
    degree : int, default=3
        Polynomial degree of the splines:
        - 0: Step functions (not smooth)
        - 1: Linear splines (continuous but not smooth)
        - 2: Quadratic splines (smooth first derivative)
        - 3: Cubic splines (smooth first and second derivatives, most common)
    knots : list, optional
        Interior knot positions. If not provided, knots are placed at
        quantiles of x based on the df/k parameter.
    boundary_knots : tuple, optional
        (min, max) defining the boundary of the spline basis.
        If not provided, uses the range of x.
    include_intercept : bool, default=False
        Whether to include the intercept (constant) basis function.
        Usually False when used in regression models that already have
        an intercept term.
    monotonicity : str, optional
        Constrain the spline to be monotonic. Options:
        - "increasing": Effect must increase with x
        - "decreasing": Effect must decrease with x
        - None (default): No monotonicity constraint
        Uses I-spline (integrated M-spline) basis internally.
    
    Returns
    -------
    numpy.ndarray
        Basis matrix of shape (n, basis_size) where n is the length of x.
    
    Notes
    -----
    **Fixed df vs penalized smooth (k):**
    
    - Use `df` when you know the exact flexibility needed
    - Use `k` when you want automatic smoothness selection via GCV
    
    **Monotonicity:**
    
    When monotonicity is specified, I-splines are used. Each basis function
    increases from 0 to 1, and with non-negative coefficients (enforced during
    fitting), the resulting curve is monotonic.
    
    Examples
    --------
    >>> import rustystats as rs
    >>> import numpy as np
    >>> 
    >>> # Fixed df B-spline
    >>> x = np.linspace(0, 10, 100)
    >>> basis = rs.bs(x, df=5)
    >>> print(basis.shape)
    (100, 4)
    
    >>> # Penalized smooth (auto-smoothing via GCV)
    >>> basis = rs.bs(x, k=10)  # Will be penalized during fitting
    
    >>> # Monotonically increasing spline
    >>> basis = rs.bs(x, df=5, monotonicity="increasing")
    
    See Also
    --------
    ns : Natural spline basis (linear at boundaries)
    """
    # Convert to numpy array
    x = np.asarray(x, dtype=np.float64).ravel()
    
    # Determine effective df
    if df is not None and k is not None:
        raise ValueError("Specify either 'df' (fixed) or 'k' (penalized), not both.")
    
    # Default to penalized smooth (k=10) if neither df nor k specified
    effective_df = df if df is not None else (k if k is not None else 10)
    
    # Handle monotonicity - use I-spline basis
    if monotonicity is not None:
        if monotonicity not in ("increasing", "decreasing"):
            raise ValueError(
                f"monotonicity must be 'increasing' or 'decreasing', got '{monotonicity}'"
            )
        increasing = monotonicity == "increasing"
        # Use explicit knots if provided for consistent prediction on new data
        if knots is not None and boundary_knots is not None:
            return _ms_with_knots_rust(x, knots, degree, boundary_knots, effective_df, increasing)
        return _ms_rust(x, effective_df, degree, boundary_knots, increasing)
    
    if knots is not None:
        # Use explicit knots
        return _bs_knots_rust(x, knots, degree, boundary_knots)
    else:
        # Compute knots automatically based on df
        return _bs_rust(x, effective_df, degree, boundary_knots, include_intercept)


def ns(
    x: np.ndarray,
    df: Optional[int] = None,
    k: Optional[int] = None,
    knots: Optional[List[float]] = None,
    boundary_knots: Optional[Tuple[float, float]] = None,
    include_intercept: bool = False,
) -> np.ndarray:
    """
    Compute natural cubic spline basis matrix.
    
    Natural splines are cubic splines with the additional constraint that
    the function is linear beyond the boundary knots. This constraint:
    - Reduces the effective degrees of freedom by 2
    - Provides more sensible extrapolation behavior
    - Often gives more stable parameter estimates
    
    Parameters
    ----------
    x : array-like
        Data points to evaluate the basis at.
    df : int, optional
        Degrees of freedom (fixed). Use this for fixed-complexity splines.
    k : int, optional
        Maximum basis size for penalized smooth terms. When k is provided,
        the spline becomes a penalized smooth term with automatic smoothness
        selection via GCV during model fitting. Typical range: 5-20.
        **Default**: If neither df nor k is provided, defaults to k=10
        (penalized smooth with automatic tuning).
    knots : list, optional
        Interior knot positions. If not provided, knots are placed at
        quantiles of x.
    boundary_knots : tuple, optional
        (min, max) defining the boundary. Beyond these points, the
        spline is constrained to be linear.
    include_intercept : bool, default=False
        Whether to include an intercept basis function.
    
    Returns
    -------
    numpy.ndarray
        Basis matrix of shape (n, basis_size).
    
    Notes
    -----
    **Fixed df vs penalized smooth (k):**
    
    - Use `df` when you know the exact flexibility needed
    - Use `k` when you want automatic smoothness selection via GCV
    
    Natural splines impose the constraint that the second derivative
    is zero at the boundaries. This means:
    
    1. The spline is linear (not curved) outside the boundary knots
    2. Extrapolation beyond the data range is more sensible
    3. The fit is often more stable near the boundaries
    
    For these reasons, natural splines are often preferred for:
    - Prediction on new data that may be outside the training range
    - Actuarial applications where extrapolation is common
    - When boundary behavior needs to be controlled
    
    Examples
    --------
    >>> import rustystats as rs
    >>> import numpy as np
    >>> 
    >>> # Fixed df natural spline
    >>> age = np.array([20, 30, 40, 50, 60, 70])
    >>> basis = rs.ns(age, df=4)
    >>> print(basis.shape)
    (6, 3)
    
    >>> # Penalized smooth (auto-smoothing via GCV)
    >>> basis = rs.ns(age, k=10)  # Will be penalized during fitting
    
    >>> # For an age effect in a GLM with linear extrapolation
    >>> basis = rs.ns(age, df=4, boundary_knots=(20, 70))
    
    See Also
    --------
    bs : B-spline basis (more flexible at boundaries)
    """
    # Convert to numpy array
    x = np.asarray(x, dtype=np.float64).ravel()
    
    # Determine effective df
    if df is not None and k is not None:
        raise ValueError("Specify either 'df' (fixed) or 'k' (penalized), not both.")
    
    # Default to penalized smooth (k=10) if neither df nor k specified
    effective_df = df if df is not None else (k if k is not None else 10)
    
    # If explicit interior knots are provided, use them for consistent prediction
    if knots is not None and boundary_knots is not None:
        return _ns_with_knots_rust(x, knots, boundary_knots, include_intercept)
    
    # Otherwise compute knots from data (training mode)
    return _ns_rust(x, effective_df, boundary_knots, include_intercept)


def bs_names(
    var_name: str,
    df: int,
    include_intercept: bool = False,
) -> List[str]:
    """
    Generate column names for B-spline basis functions.
    
    Parameters
    ----------
    var_name : str
        Name of the original variable (e.g., "age")
    df : int
        Degrees of freedom used
    include_intercept : bool, default=False
        Whether intercept was included
    
    Returns
    -------
    list of str
        Names like ['bs(age, 1/5)', 'bs(age, 2/5)', ...]
    
    Example
    -------
    >>> rs.bs_names("age", df=5)
    ['bs(age, 2/5)', 'bs(age, 3/5)', 'bs(age, 4/5)', 'bs(age, 5/5)']
    """
    return _bs_names_rust(var_name, df, include_intercept)


def ns_names(
    var_name: str,
    df: int,
    include_intercept: bool = False,
) -> List[str]:
    """
    Generate column names for natural spline basis functions.
    
    Parameters
    ----------
    var_name : str
        Name of the original variable
    df : int
        Degrees of freedom used
    include_intercept : bool, default=False
        Whether intercept was included
    
    Returns
    -------
    list of str
        Names like ['ns(age, 1/5)', 'ns(age, 2/5)', ...]
    """
    return _ns_names_rust(var_name, df, include_intercept)


class SplineTerm:
    """
    Represents a spline term for use in formula parsing.
    
    This class stores the specification for a spline transformation
    and can compute the basis matrix when given data.
    
    Attributes
    ----------
    var_name : str
        Name of the variable to transform
    spline_type : str
        'bs' (B-spline) or 'ns' (natural)
    df : int
        Degrees of freedom (fixed) or basis size (if penalized)
    degree : int
        Polynomial degree (for B-splines)
    boundary_knots : tuple or None
        Boundary knot positions
    monotonicity : str or None
        'increasing', 'decreasing', or None (no constraint)
    """
    
    def __init__(
        self,
        var_name: str,
        spline_type: str = "bs",
        df: int = 5,
        degree: int = 3,
        boundary_knots: Optional[Tuple[float, float]] = None,
        monotonicity: Optional[str] = None,
    ):
        self.var_name = var_name
        self.spline_type = spline_type.lower()
        self.df = df
        self.degree = degree
        self.boundary_knots = boundary_knots
        self.monotonicity = monotonicity
        # Computed during transform - stores knot information
        self._computed_boundary_knots: Optional[Tuple[float, float]] = None
        self._computed_internal_knots: Optional[List[float]] = None
        # Track if this is a smooth term with automatic lambda selection
        self._is_smooth = False
        # Penalty matrix for smooth terms (computed during transform)
        self._penalty_matrix: Optional[np.ndarray] = None
        # Lambda and EDF after fitting (set by fitting code)
        self._lambda: Optional[float] = None
        self._edf: Optional[float] = None
        
        if self.spline_type not in ("bs", "ns", "ms"):
            raise ValueError(f"spline_type must be 'bs', 'ns', or 'ms', got '{spline_type}'")
    
    def transform(self, x: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """
        Compute the spline basis for the given data.
        
        Parameters
        ----------
        x : np.ndarray
            Data values to transform
        
        Returns
        -------
        basis : np.ndarray
            Basis matrix
        names : list of str
            Column names for the basis
        """
        # Compute boundary knots if not already computed (first call = training)
        # On subsequent calls (prediction), reuse the stored boundary knots
        x_arr = np.asarray(x).ravel()
        if self._computed_boundary_knots is None:
            # First call - compute and store knots using Rust for consistency
            from rustystats._rustystats import compute_knots_py, compute_knots_natural_py
            
            bk = self.boundary_knots  # user-specified or None
            
            if self.spline_type == "bs":
                interior, (x_min, x_max) = compute_knots_py(
                    x_arr, self.df, self.degree, bk,
                )
            else:  # ns, ms
                interior, (x_min, x_max) = compute_knots_natural_py(
                    x_arr, self.df, bk,
                )
            
            self._computed_boundary_knots = (x_min, x_max)
            self._computed_internal_knots = list(interior)
        
        # Use stored boundary knots for basis computation
        boundary_knots_to_use = self._computed_boundary_knots
        
        effective_monotonicity = self.monotonicity
        
        if self.spline_type == "bs":
            # Use bs() with monotonicity parameter for unified API
            # Pass stored internal knots to ensure consistent basis on new data
            basis = bs(x, df=self.df, degree=self.degree,
                      knots=self._computed_internal_knots,
                      boundary_knots=boundary_knots_to_use, include_intercept=False,
                      monotonicity=effective_monotonicity)
            
            # Generate appropriate names
            if self._is_smooth:
                if effective_monotonicity:
                    sign = "+" if effective_monotonicity == "increasing" else "-"
                    names = [f"bs({self.var_name}, {i+1}/{self.df}, k, {sign})" for i in range(self.df)]
                else:
                    names = [f"bs({self.var_name}, {i+1}/{self.df}, k)" for i in range(self.df)]
            elif effective_monotonicity:
                sign = "+" if effective_monotonicity == "increasing" else "-"
                names = [f"bs({self.var_name}, {i+1}/{self.df}, {sign})" for i in range(self.df)]
            else:
                names = bs_names(self.var_name, self.df, include_intercept=False)
        elif self.spline_type == "ns":
            # Check if monotonicity was requested on natural splines
            if effective_monotonicity:
                raise ValueError(
                    f"Monotonicity constraints are not supported for natural splines (ns). "
                    f"Use bs({self.var_name}, df={self.df}, monotonicity='increasing') "
                    f"instead, which uses I-splines designed for monotonic effects."
                )
            # Pass stored internal knots to ensure consistent basis on new data
            basis = ns(x, df=self.df, knots=self._computed_internal_knots,
                      boundary_knots=boundary_knots_to_use,
                      include_intercept=False)
            if self._is_smooth:
                names = [f"ns({self.var_name}, {i+1}/{self.df}, k)" for i in range(self.df)]
            else:
                names = ns_names(self.var_name, self.df, include_intercept=False)
        elif self.spline_type == "ms":
            # Monotonic splines use I-spline basis via bs() with monotonicity
            mono = self.monotonicity or "increasing"
            basis = bs(x, df=self.df, degree=self.degree,
                      knots=self._computed_internal_knots,
                      boundary_knots=boundary_knots_to_use, include_intercept=False,
                      monotonicity=mono)
            sign = "+" if mono == "increasing" else "-"
            if self._is_smooth:
                names = [f"ms({self.var_name}, {i+1}/{self.df}, k, {sign})" for i in range(self.df)]
            else:
                names = [f"ms({self.var_name}, {i+1}/{self.df}, {sign})" for i in range(self.df)]
        else:
            raise ValueError(f"Unknown spline_type: {self.spline_type}")
        
        # Ensure names match columns
        if len(names) != basis.shape[1]:
            names = [f"{self.spline_type}({self.var_name}, {i+1}/{basis.shape[1]})" 
                    for i in range(basis.shape[1])]
        
        return basis, names
    
    def compute_penalty_matrix(self, n_cols: int, penalty_order: int = 2) -> np.ndarray:
        """
        Compute the penalty matrix for this smooth term.
        
        Parameters
        ----------
        n_cols : int
            Number of columns in the basis (after transform)
        penalty_order : int
            Order of the difference penalty (default: 2 for smoothness)
        
        Returns
        -------
        np.ndarray
            Penalty matrix of shape (n_cols, n_cols)
        """
        from rustystats.smooth import penalty_matrix
        self._penalty_matrix = penalty_matrix(n_cols, order=penalty_order)
        return self._penalty_matrix
    
    def get_knot_info(self) -> dict:
        """
        Get knot information after transform has been called.
        
        Returns
        -------
        dict
            Dictionary with spline type, df, internal knots, and boundary knots
        """
        info = {
            "type": self.spline_type,
            "df": self.df,
        }
        if self.spline_type == "bs":
            info["degree"] = self.degree
        if self.monotonicity:
            info["monotonicity"] = self.monotonicity
        if self._computed_internal_knots is not None:
            info["knots"] = self._computed_internal_knots
        if self._computed_boundary_knots is not None:
            info["boundary_knots"] = list(self._computed_boundary_knots)
        # Include smooth term info if this is a penalized smooth
        if self._is_smooth:
            info["is_smooth"] = True
            if self._lambda is not None:
                info["lambda"] = self._lambda
            if self._edf is not None:
                info["edf"] = self._edf
        return info
    
    def __repr__(self) -> str:
        if self.spline_type == "bs":
            if self.monotonicity:
                return f"bs({self.var_name}, df={self.df}, monotonicity='{self.monotonicity}')"
            return f"bs({self.var_name}, df={self.df}, degree={self.degree})"
        else:  # ns
            return f"ns({self.var_name}, df={self.df})"
