"""
Regularization Path Fitting for GLMs.

This module provides K-fold cross-validation based regularization path fitting
for automatic selection of the optimal penalty parameter (alpha/lambda).

Key features:
- Fit a path of models across alpha values
- Select optimal alpha via K-fold CV on training data
- Support for "min" (minimum CV deviance) and "1se" (1 standard error rule) selection
- Warm starting for performance
- Support for Ridge, Lasso, and Elastic Net

Example
-------
>>> import rustystats as rs
>>> 
>>> model = rs.glm(
...     formula="ClaimCount ~ VehAge + BonusMalus + TE(Region)",
...     data=train_df,
...     family="negbinomial",
...     offset="Exposure",
... )
>>> 
>>> # Fit with CV-based regularization selection
>>> result = model.fit(cv=5, selection="1se", regularization="ridge")
>>> 
>>> print(f"Selected alpha: {result.alpha}")
>>> print(f"CV deviance: {result.cv_deviance}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Literal, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    import polars as pl


@dataclass
class RegularizationPathResult:
    """
    Results from a single point on the regularization path.
    
    Attributes
    ----------
    alpha : float
        Regularization strength
    l1_ratio : float
        L1/L2 mix (0=Ridge, 1=Lasso)
    cv_deviance_mean : float
        Mean deviance across CV folds
    cv_deviance_se : float
        Standard error of CV deviance
    n_nonzero : int
        Number of non-zero coefficients
    max_coef : float
        Maximum absolute coefficient value
    """
    alpha: float
    l1_ratio: float
    cv_deviance_mean: float
    cv_deviance_se: float
    n_nonzero: int
    max_coef: float


@dataclass
class RegularizationPathInfo:
    """
    Complete regularization path information.
    
    Attributes
    ----------
    selected_alpha : float
        The alpha value selected by CV
    selected_l1_ratio : float
        The l1_ratio for the selected model
    cv_deviance : float
        CV deviance at selected alpha
    cv_deviance_se : float
        Standard error of CV deviance at selected alpha
    selection_method : str
        "min" or "1se"
    regularization_type : str
        "ridge", "lasso", "elastic_net", or "none"
    path : List[RegularizationPathResult]
        Full path results for all alpha values tried
    n_folds : int
        Number of CV folds used
    """
    selected_alpha: float
    selected_l1_ratio: float
    cv_deviance: float
    cv_deviance_se: float
    selection_method: str
    regularization_type: str
    path: List[RegularizationPathResult]
    n_folds: int


def _apply_inverse_link(eta: np.ndarray, link: str) -> np.ndarray:
    """
    Apply inverse link function to linear predictor.
    
    Parameters
    ----------
    eta : np.ndarray
        Linear predictor values
    link : str or None
        Link function name
        
    Returns
    -------
    np.ndarray
        Predicted means (mu)
    """
    if link in (None, "log"):
        return np.exp(eta)
    elif link == "identity":
        return eta
    elif link == "logit":
        return 1.0 / (1.0 + np.exp(-eta))
    elif link == "inverse":
        return 1.0 / eta
    # Default to log link for unknown
    return np.exp(eta)


def compute_alpha_max(
    X: np.ndarray,
    y: np.ndarray,
    l1_ratio: float,
    weights: Optional[np.ndarray] = None,
) -> float:
    """
    Compute the maximum alpha that would zero out all coefficients.
    
    For Lasso/Elastic Net, this is based on the maximum gradient at beta=0.
    For Ridge, we use a heuristic based on the data scale.
    
    Parameters
    ----------
    X : np.ndarray
        Design matrix (n, p)
    y : np.ndarray
        Response vector (n,)
    l1_ratio : float
        L1 ratio (0=Ridge, 1=Lasso)
    weights : np.ndarray, optional
        Observation weights
        
    Returns
    -------
    float
        Maximum alpha value
    """
    n, p = X.shape
    
    if weights is not None:
        w = weights / weights.sum() * n
    else:
        w = np.ones(n)
    
    # Center y for gradient calculation
    y_centered = y - np.average(y, weights=w)
    
    if l1_ratio > 0:
        # For Lasso/Elastic Net: max |X'Wy| / (n * l1_ratio)
        # Skip intercept column (usually first)
        gradients = np.abs(X[:, 1:].T @ (w * y_centered)) / n
        alpha_max = np.max(gradients) / max(l1_ratio, 1e-3)
    else:
        # For pure Ridge: use heuristic based on coefficient scale
        # Start with alpha that gives ~50% shrinkage on largest coefficient
        XtX_diag = np.sum(X[:, 1:] ** 2, axis=0) / n
        alpha_max = np.median(XtX_diag) * 10
    
    return max(alpha_max, 1e-4)


def generate_alpha_path(
    alpha_max: float,
    n_alphas: int = 100,
    alpha_min_ratio: float = 0.001,
) -> np.ndarray:
    """
    Generate a logarithmically-spaced path of alpha values.
    
    Parameters
    ----------
    alpha_max : float
        Maximum alpha value
    n_alphas : int
        Number of alpha values to generate
    alpha_min_ratio : float
        Ratio of alpha_min to alpha_max
        
    Returns
    -------
    np.ndarray
        Array of alpha values from alpha_max to alpha_min
    """
    alpha_min = alpha_max * alpha_min_ratio
    return np.logspace(np.log10(alpha_max), np.log10(alpha_min), n_alphas)


def create_cv_folds(
    n: int,
    n_folds: int,
    seed: Optional[int] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create K-fold cross-validation indices.
    
    Parameters
    ----------
    n : int
        Number of observations
    n_folds : int
        Number of folds
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    List[Tuple[np.ndarray, np.ndarray]]
        List of (train_indices, val_indices) tuples
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    fold_sizes = np.full(n_folds, n // n_folds, dtype=int)
    fold_sizes[:n % n_folds] += 1
    
    folds = []
    current = 0
    for fold_size in fold_sizes:
        val_idx = indices[current:current + fold_size]
        train_idx = np.concatenate([indices[:current], indices[current + fold_size:]])
        folds.append((train_idx, val_idx))
        current += fold_size
    
    return folds


def compute_deviance(
    y: np.ndarray,
    mu: np.ndarray,
    family: str,
    theta: float = 1.0,
    weights: Optional[np.ndarray] = None,
) -> float:
    """
    Compute mean deviance for a GLM family.
    
    Parameters
    ----------
    y : np.ndarray
        Observed values
    mu : np.ndarray
        Predicted means
    family : str
        Family name (may include theta, e.g., "NegativeBinomial(theta=1.89)")
    theta : float
        Dispersion parameter for negative binomial
    weights : np.ndarray, optional
        Observation weights
        
    Returns
    -------
    float
        Mean deviance
    """
    from rustystats._rustystats import compute_dataset_metrics_py as _rust_dataset_metrics
    
    n_params = 1  # Placeholder, not used for deviance calculation
    metrics = _rust_dataset_metrics(y, mu, family, n_params)
    return metrics["mean_deviance"]


def select_optimal_alpha(
    path_results: List[RegularizationPathResult],
    selection: Literal["min", "1se"] = "min",
) -> RegularizationPathResult:
    """
    Select optimal alpha from path results.
    
    Parameters
    ----------
    path_results : List[RegularizationPathResult]
        Results from regularization path
    selection : str
        Selection method:
        - "min": Select alpha with minimum CV deviance
        - "1se": Select largest alpha within 1 SE of minimum (more conservative)
        
    Returns
    -------
    RegularizationPathResult
        The selected result
    """
    # Filter out infinite deviances
    valid_results = [r for r in path_results if np.isfinite(r.cv_deviance_mean)]
    
    if not valid_results:
        raise ValueError("All regularization path fits failed")
    
    if selection == "min":
        # Select minimum CV deviance
        return min(valid_results, key=lambda r: r.cv_deviance_mean)
    
    elif selection == "1se":
        # Find minimum and its SE
        min_result = min(valid_results, key=lambda r: r.cv_deviance_mean)
        threshold = min_result.cv_deviance_mean + min_result.cv_deviance_se
        
        # Find largest alpha (most regularized) below threshold
        # Path is ordered from large alpha to small alpha
        for r in valid_results:
            if r.cv_deviance_mean <= threshold:
                return r
        
        # Fallback to minimum
        return min_result
    
    else:
        raise ValueError(f"Unknown selection method: {selection}")


def fit_cv_regularization_path(
    glm_instance,
    cv: int = 5,
    selection: Literal["min", "1se"] = "min",
    regularization: Literal["ridge", "lasso", "elastic_net"] = "ridge",
    n_alphas: int = 20,
    alpha_min_ratio: float = 0.0001,
    l1_ratio: Optional[float] = None,
    max_iter: int = 25,
    tol: float = 1e-8,
    seed: Optional[int] = None,
    include_unregularized: bool = True,
    verbose: bool = False,
) -> RegularizationPathInfo:
    """
    Fit regularization path with CV and return best model.
    
    This is the main entry point for CV-based regularization tuning.
    
    Parameters
    ----------
    glm_instance : FormulaGLM
        The GLM model instance
    cv : int
        Number of CV folds
    selection : str
        "min" or "1se"
    regularization : str
        Type of regularization: "ridge", "lasso", or "elastic_net"
    n_alphas : int
        Number of alpha values to try
    alpha_min_ratio : float
        Smallest alpha as ratio of alpha_max
    l1_ratio : float, optional
        L1 ratio for elastic_net (default 0.5)
    max_iter : int
        Maximum IRLS iterations
    tol : float
        Convergence tolerance
    seed : int, optional
        Random seed
    include_unregularized : bool
        Include alpha=0 (unregularized) in comparison
    verbose : bool
        Print progress
        
    Returns
    -------
    Tuple[result, RegularizationPathInfo]
        The fitted result at optimal alpha and the path info
    """
    from rustystats._rustystats import fit_glm_py as _fit_glm_rust
    
    # Determine l1_ratio based on regularization type
    if regularization == "ridge":
        effective_l1_ratio = 0.0
    elif regularization == "lasso":
        effective_l1_ratio = 1.0
    elif regularization == "elastic_net":
        effective_l1_ratio = l1_ratio if l1_ratio is not None else 0.5
    else:
        raise ValueError(f"Unknown regularization type: {regularization}")
    
    X = glm_instance.X
    y = glm_instance.y
    family = glm_instance.family
    link = glm_instance.link
    var_power = glm_instance.var_power
    from rustystats.formula import DEFAULT_NEGBINOMIAL_THETA
    theta = glm_instance.theta if glm_instance.theta is not None else DEFAULT_NEGBINOMIAL_THETA
    offset = glm_instance.offset
    weights = glm_instance.weights
    
    # Compute alpha path
    alpha_max = compute_alpha_max(X, y, effective_l1_ratio, weights)
    alphas = generate_alpha_path(alpha_max, n_alphas, alpha_min_ratio)
    
    if verbose:
        print(f"Fitting regularization path: {regularization}")
        print(f"  Alpha range: {alphas[-1]:.6f} to {alphas[0]:.6f}")
        print(f"  L1 ratio: {effective_l1_ratio}")
        print(f"  CV folds: {cv}")
    
    # Use Rust parallel implementation (no fallback)
    from rustystats._rustystats import fit_cv_path_py as _fit_cv_path_rust
    
    if verbose:
        print("  Using Rust parallel CV")
    
    rust_result = _fit_cv_path_rust(
        y, X, family, link, var_power, theta,
        offset, weights,
        list(alphas), effective_l1_ratio,
        cv, max_iter, tol,
        seed if seed is not None else 42,
    )
    
    # Convert Rust result to path_results format
    path_results = [
        RegularizationPathResult(
            alpha=rust_result["alphas"][i],
            l1_ratio=effective_l1_ratio,
            cv_deviance_mean=rust_result["cv_deviance_mean"][i],
            cv_deviance_se=rust_result["cv_deviance_se"][i],
            n_nonzero=X.shape[1] - 1,
            max_coef=0.0,
        )
        for i in range(len(rust_result["alphas"]))
    ]
    
    # Optionally include unregularized fit
    if include_unregularized:
        if verbose:
            print("  Fitting unregularized model for comparison...")
        
        folds = create_cv_folds(len(y), cv, seed)
        fold_deviances = []
        
        for train_idx, val_idx in folds:
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            offset_train = offset[train_idx] if offset is not None else None
            offset_val = offset[val_idx] if offset is not None else None
            weights_train = weights[train_idx] if weights is not None else None
            
            try:
                result = _fit_glm_rust(
                    y_train, X_train, family, link, var_power, theta,
                    offset_train, weights_train, 0.0, 0.0, max_iter, tol
                )
            except ValueError:
                continue
            linear_pred = X_val @ result.params
            if offset_val is not None:
                linear_pred = linear_pred + offset_val
            mu_val = _apply_inverse_link(linear_pred, link)
            dev = compute_deviance(y_val, mu_val, family)
            fold_deviances.append(dev)
        
        valid_deviances = [d for d in fold_deviances if np.isfinite(d)]
        if valid_deviances:
            unreg_result = RegularizationPathResult(
                alpha=0.0,
                l1_ratio=0.0,
                cv_deviance_mean=np.mean(valid_deviances),
                cv_deviance_se=np.std(valid_deviances) / np.sqrt(len(valid_deviances)),
                n_nonzero=X.shape[1] - 1,
                max_coef=0.0,  # Will be updated after final fit
            )
            path_results.append(unreg_result)
    
    # Select optimal alpha
    best = select_optimal_alpha(path_results, selection)
    
    if verbose:
        print(f"\nSelected: alpha={best.alpha:.6f}, CV deviance={best.cv_deviance_mean:.6f}")
    
    # Determine regularization type for the selected model
    if best.alpha == 0.0:
        reg_type = "none"
    elif effective_l1_ratio >= 1.0:
        reg_type = "lasso"
    elif effective_l1_ratio <= 0.0:
        reg_type = "ridge"
    else:
        reg_type = "elastic_net"
    
    # Create path info
    path_info = RegularizationPathInfo(
        selected_alpha=best.alpha,
        selected_l1_ratio=best.l1_ratio,
        cv_deviance=best.cv_deviance_mean,
        cv_deviance_se=best.cv_deviance_se,
        selection_method=selection,
        regularization_type=reg_type,
        path=path_results,
        n_folds=cv,
    )
    
    return path_info
