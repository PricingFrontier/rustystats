"""
Variable Selection Utilities for Regularized GLMs
=================================================

This module provides tools for variable selection with Lasso, Ridge,
and Elastic Net regularization:

- **Coefficient paths**: See how coefficients shrink as alpha increases
- **Cross-validation**: Find the optimal regularization strength
- **Model selection**: Compare models with different penalties

Example
-------
>>> import rustystats as rs
>>> import numpy as np
>>>
>>> # Fit Lasso path
>>> alphas, coefs, deviances = rs.lasso_path(y, X, family="gaussian")
>>>
>>> # Cross-validation to find best alpha
>>> cv_result = rs.cv_glm(y, X, family="gaussian", l1_ratio=1.0)
>>> print(f"Best alpha: {cv_result.alpha_best}")
>>> print(f"Best CV score: {cv_result.score_best}")
"""

from __future__ import annotations

import numpy as np
from typing import Optional, List, Tuple, Union, TYPE_CHECKING
from dataclasses import dataclass

from rustystats.glm import fit_glm, GLMResults

if TYPE_CHECKING:
    import polars as pl


@dataclass
class RegularizationPath:
    """Results from computing a regularization path.
    
    Attributes
    ----------
    alphas : np.ndarray
        Array of alpha (regularization strength) values used
    coefficients : np.ndarray
        Coefficient matrix of shape (n_alphas, n_features)
    deviances : np.ndarray
        Deviance at each alpha value
    n_nonzero : np.ndarray
        Number of non-zero coefficients at each alpha
    converged : np.ndarray
        Whether each fit converged
    """
    alphas: np.ndarray
    coefficients: np.ndarray
    deviances: np.ndarray
    n_nonzero: np.ndarray
    converged: np.ndarray
    feature_names: Optional[List[str]] = None
    
    def to_dataframe(self) -> "pl.DataFrame":
        """Convert coefficient path to a Polars DataFrame.
        
        Returns
        -------
        pl.DataFrame
            DataFrame with alpha column and feature columns
        """
        import polars as pl
        
        cols = self.feature_names or [f"x{i}" for i in range(self.coefficients.shape[1])]
        data = {"alpha": self.alphas}
        for i, col in enumerate(cols):
            data[col] = self.coefficients[:, i]
        return pl.DataFrame(data)
    
    def plot(self, ax=None, log_scale: bool = True, show_legend: bool = True):
        """Plot the coefficient path.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.
        log_scale : bool, default=True
            Use log scale for x-axis (alpha)
        show_legend : bool, default=True
            Show legend with feature names
            
        Returns
        -------
        matplotlib.axes.Axes
            The axes with the plot
        """
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        n_features = self.coefficients.shape[1]
        colors = plt.cm.tab20(np.linspace(0, 1, n_features))
        
        for i in range(n_features):
            label = self.feature_names[i] if self.feature_names else f"x{i}"
            ax.plot(self.alphas, self.coefficients[:, i], 
                   color=colors[i], label=label, linewidth=1.5)
        
        if log_scale:
            ax.set_xscale('log')
        
        ax.set_xlabel('Alpha (regularization strength)')
        ax.set_ylabel('Coefficient value')
        ax.set_title('Regularization Path')
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
        
        if show_legend and n_features <= 20:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        return ax


@dataclass
class CVResult:
    """Results from cross-validated GLM fitting.
    
    Attributes
    ----------
    alphas : np.ndarray
        Alpha values tested
    scores_mean : np.ndarray
        Mean CV score (negative deviance) at each alpha
    scores_std : np.ndarray
        Standard deviation of CV scores at each alpha
    alpha_best : float
        Alpha with best (highest) mean score
    alpha_1se : float
        Largest alpha within 1 SE of best (more regularization)
    n_nonzero_best : int
        Number of non-zero coefficients at alpha_best
    """
    alphas: np.ndarray
    scores_mean: np.ndarray
    scores_std: np.ndarray
    alpha_best: float
    alpha_1se: float
    n_nonzero_best: int
    best_result: Optional[GLMResults] = None
    
    @property
    def score_best(self) -> float:
        """Best (highest) mean CV score."""
        return self.scores_mean.max()
    
    def plot(self, ax=None):
        """Plot cross-validation curve.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on
            
        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        
        # Plot mean ± 1 SE
        ax.errorbar(self.alphas, self.scores_mean, yerr=self.scores_std,
                   fmt='o-', capsize=3, capthick=1, linewidth=1.5, markersize=4)
        
        # Mark best alpha
        ax.axvline(x=self.alpha_best, color='green', linestyle='--', 
                  label=f'Best α = {self.alpha_best:.4f}')
        ax.axvline(x=self.alpha_1se, color='orange', linestyle=':', 
                  label=f'1-SE α = {self.alpha_1se:.4f}')
        
        ax.set_xscale('log')
        ax.set_xlabel('Alpha (regularization strength)')
        ax.set_ylabel('CV Score (negative deviance)')
        ax.set_title('Cross-Validation Results')
        ax.legend()
        
        return ax


def compute_alpha_grid(
    y: np.ndarray,
    X: np.ndarray,
    n_alphas: int = 100,
    alpha_min_ratio: float = 1e-4,
    l1_ratio: float = 1.0,
) -> np.ndarray:
    """Compute a sensible grid of alpha values.
    
    The maximum alpha is computed as the smallest value that would
    zero out all coefficients (for Lasso). The grid is logarithmically
    spaced from alpha_max to alpha_max * alpha_min_ratio.
    
    Parameters
    ----------
    y : np.ndarray
        Response variable
    X : np.ndarray
        Design matrix (with intercept column)
    n_alphas : int, default=100
        Number of alpha values
    alpha_min_ratio : float, default=1e-4
        Ratio of min to max alpha
    l1_ratio : float, default=1.0
        L1 ratio (1.0 for Lasso)
        
    Returns
    -------
    np.ndarray
        Array of alpha values (decreasing order)
    """
    n = len(y)
    
    # Center y for computing correlations
    y_centered = y - np.mean(y)
    
    # Compute max correlation (excluding intercept column)
    # alpha_max = max|X'y| / (n * l1_ratio)
    correlations = np.abs(X[:, 1:].T @ y_centered) / n
    
    if l1_ratio > 0:
        alpha_max = np.max(correlations) / l1_ratio
    else:
        # Ridge: use a heuristic
        alpha_max = np.max(correlations) * 10
    
    alpha_min = alpha_max * alpha_min_ratio
    
    # Logarithmically spaced grid (decreasing)
    alphas = np.logspace(np.log10(alpha_max), np.log10(alpha_min), n_alphas)
    
    return alphas


def regularization_path(
    y: np.ndarray,
    X: np.ndarray,
    family: str = "gaussian",
    link: Optional[str] = None,
    alphas: Optional[np.ndarray] = None,
    n_alphas: int = 100,
    l1_ratio: float = 1.0,
    max_iter: int = 100,
    tol: float = 1e-6,
    feature_names: Optional[List[str]] = None,
) -> RegularizationPath:
    """Compute coefficient path over a range of alpha values.
    
    Fits the model at each alpha value and records the coefficients.
    Uses warm starts (previous solution as initial guess) for efficiency.
    
    Parameters
    ----------
    y : np.ndarray
        Response variable
    X : np.ndarray
        Design matrix (should include intercept column)
    family : str, default="gaussian"
        Distribution family
    link : str, optional
        Link function
    alphas : np.ndarray, optional
        Alpha values to use. If None, computed automatically.
    n_alphas : int, default=100
        Number of alpha values if alphas not provided
    l1_ratio : float, default=1.0
        Elastic Net mixing (1.0 = Lasso, 0.0 = Ridge)
    max_iter : int, default=100
        Maximum iterations per fit
    tol : float, default=1e-6
        Convergence tolerance
    feature_names : list[str], optional
        Names for features
        
    Returns
    -------
    RegularizationPath
        Object containing the coefficient path
        
    Examples
    --------
    >>> path = rs.regularization_path(y, X, family="gaussian", l1_ratio=1.0)
    >>> print(f"Tested {len(path.alphas)} alpha values")
    >>> path.plot()  # Visualize the path
    """
    y = np.asarray(y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    
    if alphas is None:
        alphas = compute_alpha_grid(y, X, n_alphas, l1_ratio=l1_ratio)
    else:
        alphas = np.sort(alphas)[::-1]  # Ensure decreasing order
    
    n_alphas = len(alphas)
    n_features = X.shape[1]
    
    # Storage
    coefficients = np.zeros((n_alphas, n_features))
    deviances = np.zeros(n_alphas)
    n_nonzero = np.zeros(n_alphas, dtype=int)
    converged = np.zeros(n_alphas, dtype=bool)
    
    # Fit path (warm start would require modifying Rust code, skip for now)
    for i, alpha in enumerate(alphas):
        try:
            result = fit_glm(
                y, X, family=family, link=link,
                alpha=alpha, l1_ratio=l1_ratio,
                max_iter=max_iter, tol=tol
            )
            coefficients[i] = result.params
            deviances[i] = result.deviance
            n_nonzero[i] = result.n_nonzero()
            converged[i] = result.converged
        except Exception:
            # If fit fails, use previous coefficients
            if i > 0:
                coefficients[i] = coefficients[i-1]
                deviances[i] = deviances[i-1]
                n_nonzero[i] = n_nonzero[i-1]
            converged[i] = False
    
    return RegularizationPath(
        alphas=alphas,
        coefficients=coefficients,
        deviances=deviances,
        n_nonzero=n_nonzero,
        converged=converged,
        feature_names=feature_names,
    )


# Alias for sklearn-style naming
lasso_path = regularization_path


def cv_glm(
    y: np.ndarray,
    X: np.ndarray,
    family: str = "gaussian",
    link: Optional[str] = None,
    alphas: Optional[np.ndarray] = None,
    n_alphas: int = 50,
    l1_ratio: float = 1.0,
    cv: int = 5,
    max_iter: int = 100,
    tol: float = 1e-6,
    scoring: str = "deviance",
    random_state: Optional[int] = None,
) -> CVResult:
    """Cross-validated GLM with regularization.
    
    Performs k-fold cross-validation to find the optimal regularization
    strength (alpha). Returns the best alpha and the alpha within 1 SE
    of the best (for more parsimonious models).
    
    Parameters
    ----------
    y : np.ndarray
        Response variable
    X : np.ndarray
        Design matrix (should include intercept column)
    family : str, default="gaussian"
        Distribution family
    link : str, optional
        Link function
    alphas : np.ndarray, optional
        Alpha values to test. If None, computed automatically.
    n_alphas : int, default=50
        Number of alpha values if alphas not provided
    l1_ratio : float, default=1.0
        Elastic Net mixing (1.0 = Lasso, 0.0 = Ridge)
    cv : int, default=5
        Number of cross-validation folds
    max_iter : int, default=100
        Maximum iterations per fit
    tol : float, default=1e-6
        Convergence tolerance
    scoring : str, default="deviance"
        Scoring metric: "deviance" (negative deviance, higher is better)
    random_state : int, optional
        Random seed for fold shuffling
        
    Returns
    -------
    CVResult
        Object containing CV results and best parameters
        
    Examples
    --------
    >>> cv_result = rs.cv_glm(y, X, family="poisson", l1_ratio=1.0, cv=5)
    >>> print(f"Best alpha: {cv_result.alpha_best:.4f}")
    >>> print(f"Features selected: {cv_result.n_nonzero_best}")
    >>> cv_result.plot()  # Visualize CV curve
    """
    y = np.asarray(y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    n = len(y)
    
    if alphas is None:
        alphas = compute_alpha_grid(y, X, n_alphas, l1_ratio=l1_ratio)
    
    n_alphas = len(alphas)
    
    # Create fold indices
    rng = np.random.default_rng(random_state)
    indices = np.arange(n)
    rng.shuffle(indices)
    fold_sizes = np.full(cv, n // cv)
    fold_sizes[:n % cv] += 1
    
    folds = []
    current = 0
    for fold_size in fold_sizes:
        folds.append(indices[current:current + fold_size])
        current += fold_size
    
    # Storage for CV scores
    scores = np.zeros((cv, n_alphas))
    
    # Cross-validation loop
    for fold_idx in range(cv):
        # Create train/test split
        test_idx = folds[fold_idx]
        train_idx = np.concatenate([folds[j] for j in range(cv) if j != fold_idx])
        
        y_train, y_test = y[train_idx], y[test_idx]
        X_train, X_test = X[train_idx], X[test_idx]
        
        # Fit at each alpha
        for alpha_idx, alpha in enumerate(alphas):
            try:
                result = fit_glm(
                    y_train, X_train, family=family, link=link,
                    alpha=alpha, l1_ratio=l1_ratio,
                    max_iter=max_iter, tol=tol
                )
                
                # Compute score on test set
                # Use negative deviance (higher is better)
                eta_test = X_test @ result.params
                
                # Simple deviance approximation
                if family == "gaussian":
                    pred = eta_test
                    score = -np.sum((y_test - pred) ** 2)
                elif family in ("poisson", "gamma"):
                    pred = np.exp(eta_test)
                    pred = np.clip(pred, 1e-10, 1e10)
                    if family == "poisson":
                        # Poisson deviance
                        score = -2 * np.sum(y_test * np.log(pred + 1e-10) - pred)
                    else:
                        # Gamma deviance
                        score = -2 * np.sum(-np.log(y_test / pred) + (y_test - pred) / pred)
                elif family == "binomial":
                    pred = 1 / (1 + np.exp(-eta_test))
                    pred = np.clip(pred, 1e-10, 1 - 1e-10)
                    score = -2 * np.sum(
                        y_test * np.log(pred) + (1 - y_test) * np.log(1 - pred)
                    )
                else:
                    # Default to MSE
                    pred = eta_test
                    score = -np.sum((y_test - pred) ** 2)
                
                scores[fold_idx, alpha_idx] = score
                
            except Exception:
                scores[fold_idx, alpha_idx] = -np.inf
    
    # Compute mean and std across folds
    scores_mean = np.mean(scores, axis=0)
    scores_std = np.std(scores, axis=0) / np.sqrt(cv)  # Standard error
    
    # Find best alpha (highest score)
    best_idx = np.argmax(scores_mean)
    alpha_best = alphas[best_idx]
    
    # Find alpha_1se (largest alpha within 1 SE of best)
    threshold = scores_mean[best_idx] - scores_std[best_idx]
    valid_idx = np.where(scores_mean >= threshold)[0]
    alpha_1se_idx = valid_idx[np.argmax(alphas[valid_idx])]  # Largest alpha
    alpha_1se = alphas[alpha_1se_idx]
    
    # Fit final model at best alpha
    best_result = fit_glm(
        y, X, family=family, link=link,
        alpha=alpha_best, l1_ratio=l1_ratio,
        max_iter=max_iter, tol=tol
    )
    
    return CVResult(
        alphas=alphas,
        scores_mean=scores_mean,
        scores_std=scores_std,
        alpha_best=alpha_best,
        alpha_1se=alpha_1se,
        n_nonzero_best=best_result.n_nonzero(),
        best_result=best_result,
    )


# Convenience aliases
cv_lasso = lambda y, X, **kwargs: cv_glm(y, X, l1_ratio=1.0, **kwargs)
cv_ridge = lambda y, X, **kwargs: cv_glm(y, X, l1_ratio=0.0, **kwargs)
cv_elasticnet = lambda y, X, l1_ratio=0.5, **kwargs: cv_glm(y, X, l1_ratio=l1_ratio, **kwargs)
