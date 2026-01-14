"""
Spline basis functions for non-linear continuous effects in GLMs.

This module provides B-splines, natural splines, and monotonic splines (I-splines),
which are essential for modeling non-linear relationships between continuous 
predictors and the response variable.

Key Functions
-------------
- `bs()` - B-spline basis (flexible piecewise polynomials)
- `ns()` - Natural spline basis (linear extrapolation at boundaries)
- `ms()` - Monotonic spline basis (I-splines, constrained to be monotonic)

Example
-------
>>> import rustystats as rs
>>> import numpy as np
>>> 
>>> # Create spline basis for age with 5 degrees of freedom
>>> age = np.array([25, 35, 45, 55, 65])
>>> age_basis = rs.bs(age, df=5)
>>> print(age_basis.shape)
(5, 4)

>>> # Use in formula API
>>> result = rs.glm(
...     "y ~ bs(age, df=5) + C(region)",
...     data=data,
...     family="poisson"
... ).fit()

When to Use Each Type
---------------------
**B-splines (`bs`):**
- More flexible at boundaries
- Good when you don't need to extrapolate
- Standard choice for most applications

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
    bs_knots_py as _bs_knots_rust,
    bs_names_py as _bs_names_rust,
    ns_names_py as _ns_names_rust,
    ms_py as _ms_rust,
    ms_names_py as _ms_names_rust,
)


def bs(
    x: np.ndarray,
    df: int = 5,
    degree: int = 3,
    knots: Optional[List[float]] = None,
    boundary_knots: Optional[Tuple[float, float]] = None,
    include_intercept: bool = False,
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
    df : int, default=5
        Degrees of freedom, i.e., number of basis functions to generate.
        Higher df = more flexible fit but risk of overfitting.
        Typical range: 3-10 for most applications.
    degree : int, default=3
        Polynomial degree of the splines:
        - 0: Step functions (not smooth)
        - 1: Linear splines (continuous but not smooth)
        - 2: Quadratic splines (smooth first derivative)
        - 3: Cubic splines (smooth first and second derivatives, most common)
    knots : list, optional
        Interior knot positions. If not provided, knots are placed at
        quantiles of x based on the df parameter.
    boundary_knots : tuple, optional
        (min, max) defining the boundary of the spline basis.
        If not provided, uses the range of x.
    include_intercept : bool, default=False
        Whether to include the intercept (constant) basis function.
        Usually False when used in regression models that already have
        an intercept term.
    
    Returns
    -------
    numpy.ndarray
        Basis matrix of shape (n, k) where n is the length of x and
        k is the number of basis functions (df or df-1 depending on
        include_intercept).
    
    Notes
    -----
    The number of basis functions is:
    - df if include_intercept=True
    - df-1 if include_intercept=False (default)
    
    B-splines have the "partition of unity" property: the basis functions
    sum to 1 at any point. They are also non-negative and have local support
    (each basis function is non-zero only over a limited range).
    
    Examples
    --------
    >>> import rustystats as rs
    >>> import numpy as np
    >>> 
    >>> # Basic usage
    >>> x = np.linspace(0, 10, 100)
    >>> basis = rs.bs(x, df=5)
    >>> print(basis.shape)
    (100, 4)
    
    >>> # With explicit knots
    >>> basis = rs.bs(x, knots=[2.5, 5.0, 7.5], degree=3)
    >>> print(basis.shape)
    (100, 7)
    
    >>> # For use in regression with intercept already present
    >>> X = np.column_stack([np.ones(100), rs.bs(x, df=4)])
    
    See Also
    --------
    ns : Natural spline basis (linear at boundaries)
    """
    # Convert to numpy array
    x = np.asarray(x, dtype=np.float64).ravel()
    
    if knots is not None:
        # Use explicit knots
        return _bs_knots_rust(x, knots, degree, boundary_knots)
    else:
        # Compute knots automatically based on df
        return _bs_rust(x, df, degree, boundary_knots, include_intercept)


def ns(
    x: np.ndarray,
    df: int = 5,
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
    df : int, default=5
        Degrees of freedom. The number of basis functions generated
        will be df (or df-1 if include_intercept=False).
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
        Basis matrix of shape (n, k).
    
    Notes
    -----
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
    >>> # Basic usage
    >>> age = np.array([20, 30, 40, 50, 60, 70])
    >>> basis = rs.ns(age, df=4)
    >>> print(basis.shape)
    (6, 3)
    
    >>> # For an age effect in a GLM
    >>> # The spline will be linear for ages below 20 and above 70
    >>> basis = rs.ns(age, df=4, boundary_knots=(20, 70))
    
    See Also
    --------
    bs : B-spline basis (more flexible at boundaries)
    """
    # Convert to numpy array
    x = np.asarray(x, dtype=np.float64).ravel()
    
    # Natural splines don't support explicit interior knots in our implementation
    # (knots are computed from df)
    return _ns_rust(x, df, boundary_knots, include_intercept)


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


def ms(
    x: np.ndarray,
    df: int = 5,
    degree: int = 3,
    boundary_knots: Optional[Tuple[float, float]] = None,
    increasing: bool = True,
) -> np.ndarray:
    """
    Compute monotonic spline (I-spline) basis matrix.
    
    I-splines are integrated M-splines that provide a basis for monotonic
    regression. Each basis function is monotonically increasing from 0 to 1.
    With non-negative coefficients, any linear combination produces a
    monotonically increasing function.
    
    This is the standard approach for fitting monotonic curves in GLMs,
    commonly used in actuarial applications where effects should be
    constrained to increase or decrease with the predictor.
    
    Parameters
    ----------
    x : array-like
        Data points to evaluate the basis at. Will be converted to 1D numpy array.
    df : int, default=5
        Degrees of freedom, i.e., number of basis functions to generate.
        Higher df = more flexible fit within the monotonicity constraint.
    degree : int, default=3
        Polynomial degree of the underlying splines (3 = cubic).
    boundary_knots : tuple, optional
        (min, max) defining the boundary of the spline basis.
        If not provided, uses the range of x.
    increasing : bool, default=True
        If True, basis for monotonically increasing function.
        If False, basis for monotonically decreasing function.
    
    Returns
    -------
    numpy.ndarray
        Basis matrix of shape (n, df). All values are in [0, 1].
        Each column is monotonically increasing (or decreasing) in x.
    
    Notes
    -----
    **How monotonic splines work:**
    
    I-splines have the property that:
    - Each basis function I_j(x) increases from 0 to 1 as x increases
    - If all coefficients β_j >= 0, then f(x) = Σ β_j I_j(x) is monotonically increasing
    - The constraint β_j >= 0 is enforced during model fitting
    
    **When to use monotonic splines:**
    
    - When business logic dictates a monotonic relationship
    - Age effects in insurance (older → higher risk)
    - Vehicle age effects (older → different risk profile)
    - Any domain where non-monotonic fits would be implausible
    
    **Comparison with other splines:**
    
    - B-splines (bs): Flexible, no shape constraints
    - Natural splines (ns): Linear extrapolation, no shape constraints  
    - Monotonic splines (ms): Constrained to be monotonic
    
    Examples
    --------
    >>> import rustystats as rs
    >>> import numpy as np
    >>> 
    >>> # Create monotonically increasing basis
    >>> x = np.linspace(0, 10, 100)
    >>> basis = rs.ms(x, df=5)
    >>> print(basis.shape)
    (100, 5)
    >>> 
    >>> # All values should be in [0, 1]
    >>> print(f"Range: [{basis.min():.3f}, {basis.max():.3f}]")
    Range: [0.000, 1.000]
    >>> 
    >>> # Each column is monotonically increasing
    >>> for j in range(basis.shape[1]):
    ...     assert np.all(np.diff(basis[:, j]) >= -1e-10)
    >>> 
    >>> # For decreasing relationship (e.g., vehicle value with age)
    >>> basis_dec = rs.ms(x, df=5, increasing=False)
    
    See Also
    --------
    bs : B-spline basis (unconstrained)
    ns : Natural spline basis (linear at boundaries)
    """
    # Convert to numpy array
    x = np.asarray(x, dtype=np.float64).ravel()
    
    return _ms_rust(x, df, degree, boundary_knots, increasing)


def ms_names(
    var_name: str,
    df: int,
    increasing: bool = True,
) -> List[str]:
    """
    Generate column names for monotonic spline basis functions.
    
    Parameters
    ----------
    var_name : str
        Name of the original variable (e.g., "age")
    df : int
        Degrees of freedom used
    increasing : bool, default=True
        Whether the spline is increasing or decreasing
    
    Returns
    -------
    list of str
        Names like ['ms(age, 1/5, +)', 'ms(age, 2/5, +)', ...]
        The '+' or '-' indicates direction.
    
    Example
    -------
    >>> rs.ms_names("age", df=5)
    ['ms(age, 1/5, +)', 'ms(age, 2/5, +)', 'ms(age, 3/5, +)', 'ms(age, 4/5, +)', 'ms(age, 5/5, +)']
    """
    return _ms_names_rust(var_name, df, increasing)


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
        'bs' (B-spline), 'ns' (natural), or 'ms' (monotonic)
    df : int
        Degrees of freedom
    degree : int
        Polynomial degree (for B-splines and monotonic splines)
    boundary_knots : tuple or None
        Boundary knot positions
    increasing : bool
        For monotonic splines, whether increasing (True) or decreasing (False)
    """
    
    def __init__(
        self,
        var_name: str,
        spline_type: str = "bs",
        df: int = 5,
        degree: int = 3,
        boundary_knots: Optional[Tuple[float, float]] = None,
        increasing: bool = True,
    ):
        self.var_name = var_name
        self.spline_type = spline_type.lower()
        self.df = df
        self.degree = degree
        self.boundary_knots = boundary_knots
        self.increasing = increasing
        # Computed during transform - stores knot information
        self._computed_boundary_knots: Optional[Tuple[float, float]] = None
        self._computed_internal_knots: Optional[List[float]] = None
        # Track if monotonicity constraint is explicitly set (for ns with constraints)
        self._monotonic = False
        
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
            # First call - compute and store boundary knots
            if self.boundary_knots is not None:
                self._computed_boundary_knots = self.boundary_knots
            else:
                self._computed_boundary_knots = (float(np.min(x_arr)), float(np.max(x_arr)))
            
            # Compute internal knots based on quantiles (only on first call)
            # Number of internal knots = df - degree - 1 (for bs) or df - 1 (for ns/ms)
            if self.spline_type == "bs":
                n_internal = max(0, self.df - self.degree - 1)
            else:  # ns, ms
                n_internal = max(0, self.df - 1)
            
            if n_internal > 0:
                quantiles = np.linspace(0, 1, n_internal + 2)[1:-1]
                self._computed_internal_knots = [float(np.quantile(x_arr, q)) for q in quantiles]
            else:
                self._computed_internal_knots = []
        
        # Use stored boundary knots for basis computation
        boundary_knots_to_use = self._computed_boundary_knots
        
        if self.spline_type == "bs":
            basis = bs(x, df=self.df, degree=self.degree, 
                      boundary_knots=boundary_knots_to_use, include_intercept=False)
            names = bs_names(self.var_name, self.df, include_intercept=False)
        elif self.spline_type == "ns":
            # Check if monotonicity was requested on natural splines
            if self._monotonic:
                raise ValueError(
                    f"Monotonicity constraints are not supported for natural splines (ns). "
                    f"Use ms({self.var_name}, df={self.df}, increasing={'true' if self.increasing else 'false'}) "
                    f"instead, which uses I-splines designed for monotonic effects."
                )
            basis = ns(x, df=self.df, boundary_knots=boundary_knots_to_use,
                      include_intercept=False)
            names = ns_names(self.var_name, self.df, include_intercept=False)
        else:  # ms (monotonic spline)
            basis = ms(x, df=self.df, degree=self.degree,
                      boundary_knots=boundary_knots_to_use, increasing=self.increasing)
            names = ms_names(self.var_name, self.df, increasing=self.increasing)
        
        # Ensure names match columns
        if len(names) != basis.shape[1]:
            names = [f"{self.spline_type}({self.var_name}, {i+1}/{basis.shape[1]})" 
                    for i in range(basis.shape[1])]
        
        return basis, names
    
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
        if self.spline_type in ("bs", "ms"):
            info["degree"] = self.degree
        if self.spline_type == "ms":
            info["increasing"] = self.increasing
        if self._computed_internal_knots is not None:
            info["knots"] = self._computed_internal_knots
        if self._computed_boundary_knots is not None:
            info["boundary_knots"] = list(self._computed_boundary_knots)
        return info
    
    def __repr__(self) -> str:
        if self.spline_type == "bs":
            return f"bs({self.var_name}, df={self.df}, degree={self.degree})"
        elif self.spline_type == "ns":
            return f"ns({self.var_name}, df={self.df})"
        else:  # ms
            direction = "increasing" if self.increasing else "decreasing"
            return f"ms({self.var_name}, df={self.df}, {direction})"
