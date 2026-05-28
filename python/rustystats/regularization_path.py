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
>>> model = rs.glm_dict(
...     response="ClaimCount",
...     terms={"VehAge": {"type": "linear"}, "BonusMalus": {"type": "linear"}, "Region": {"type": "target_encoding"}},
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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from rustystats.constants import (
    ALPHA_MAX_FLOOR,
    DEFAULT_ALPHA_MIN_RATIO,
    DEFAULT_CV_FOLDS,
    DEFAULT_CV_SEED,
    DEFAULT_ELASTIC_NET_L1_RATIO,
    DEFAULT_MAX_ITER,
    DEFAULT_N_ALPHAS,
    DEFAULT_NEGBINOMIAL_THETA,
    DEFAULT_TOLERANCE,
    L1_RATIO_MIN_CLAMP,
)
from rustystats.exceptions import FittingError, ValidationError

if TYPE_CHECKING:
    pass


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
    path : list[RegularizationPathResult]
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
    path: list[RegularizationPathResult]
    n_folds: int
    cv_max_iter: int = DEFAULT_MAX_ITER
    cv_tol: float = DEFAULT_TOLERANCE
    fold_safe_target_encoding: bool = False


def _apply_inverse_link(eta: np.ndarray, link: str) -> np.ndarray:
    """Apply inverse link function to linear predictor.

    Delegates to the shared implementation in formula.py which raises
    on unknown links instead of silently defaulting.
    """
    from rustystats.formula import apply_inverse_link

    return apply_inverse_link(eta, link)


def _glm_score_gradient_factor(
    y: np.ndarray,
    mu: np.ndarray,
    family: str,
    link: str,
    var_power: float,
    theta: float,
    weights: np.ndarray,
) -> np.ndarray:
    """Per-observation factor for the GLM log-likelihood score at the null model.

    Returns ``f`` such that ``score_j = sum_i x_ij * f_i``. For canonical link
    families (Gaussian/identity, Poisson/log, Binomial/logit) this collapses
    to ``w_i * (y_i - μ_i)`` because ``(∂μ/∂η) / V(μ) ≡ 1`` along the
    canonical link. For non-canonical links (e.g. Gamma + log) the chain factor
    ``(∂μ/∂η) / V(μ)`` is included explicitly.

    Notes
    -----
    The IRLS solver minimises the deviance/-loglik using the same gradient,
    so matching this expression here is what makes ``alpha_max`` *tight* (i.e.
    soft-thresholds every penalised coefficient to zero) on the solver's own
    objective rather than on a centred-``y`` proxy that drifts by a factor of
    ``n`` (the bug RS-ACT-005 closes).
    """
    if link == "identity":
        dmu_deta = np.ones_like(mu)
    elif link == "log":
        dmu_deta = mu
    elif link == "logit":
        dmu_deta = mu * (1.0 - mu)
    else:
        raise ValidationError(f"alpha_max: unsupported link {link!r}")

    if family in ("gaussian",):
        variance = np.ones_like(mu)
    elif family in ("poisson", "quasipoisson"):
        variance = mu
    elif family in ("gamma",):
        variance = mu * mu
    elif family in ("binomial", "quasibinomial"):
        variance = mu * (1.0 - mu)
    elif family in ("tweedie",):
        variance = np.power(mu, var_power)
    elif family in ("negbinomial",):
        variance = mu + mu * mu / theta
    else:
        raise ValidationError(f"alpha_max: unsupported family {family!r}")

    # Guard against zero variance (e.g. Poisson mu=0); the score contribution
    # vanishes there because y must also be 0 with non-zero probability.
    safe_variance = np.where(variance > 0.0, variance, 1.0)
    return weights * (y - mu) * dmu_deta / safe_variance


def _fit_null_intercept(
    y: np.ndarray,
    family: str,
    link: str,
    offset: np.ndarray,
    weights: np.ndarray,
    var_power: float,
    theta: float,
) -> float:
    """Fit the intercept-only GLM (``η = β_0 + offset``).

    Closed-form for the two most common offset+canonical-link cases (the only
    ones where alpha_max would otherwise be needed on the very first IRLS
    step, where a Rust round-trip would be the bottleneck). All other families
    or non-zero offsets fall back to a single Rust ``fit_glm_py`` call on a
    width-1 design — same code path as a normal fit, so anything that
    converges in production converges here.
    """
    # Closed-form: log-link with linear offset → β_0 = log(Σwy / Σw·exp(offset)).
    # Works uniformly for Poisson, Gamma, Tweedie, NegBin (all log-link) and
    # avoids a Rust call inside the alpha-path inner loop.
    if link == "log":
        denom = float(np.sum(weights * np.exp(offset)))
        if denom > 0.0:
            num = float(np.sum(weights * y))
            if num > 0.0:
                return float(np.log(num / denom))
    # Identity link with offset: β_0 minimises Σ w (y - β_0 - offset)² →
    # β_0 = Σw(y - offset) / Σw.
    if link == "identity":
        return float(np.sum(weights * (y - offset)) / np.sum(weights))

    # Fall back to a one-feature Rust fit for non-closed-form combinations.
    from rustystats._rustystats import fit_glm_py

    x_intercept = np.ones((y.shape[0], 1), dtype=np.float64)
    null_res = fit_glm_py(
        y,
        x_intercept,
        family,
        link,
        var_power,
        theta,
        offset if not np.all(offset == 0.0) else None,
        weights if not np.allclose(weights, 1.0) else None,
        0.0,
        0.0,
        200,
        1e-12,
        None,
        None,
        False,
    )
    return float(np.asarray(null_res.params)[0])


def compute_alpha_max(
    X: np.ndarray,
    y: np.ndarray,
    l1_ratio: float,
    *,
    family: str,
    link: str,
    offset: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    var_power: float = 1.5,
    theta: float = 1.0,
    intercept_col: int | None = 0,
) -> float:
    """Maximum alpha that zeroes every penalised coefficient (RS-ACT-005).

    For an elastic-net objective with ``l1_ratio > 0`` this is

    .. math::

        \\alpha_{\\max} = \\frac{\\max_j |\\,X_j' \\, f(y, \\mu_0)\\,|}{l_1}

    where ``μ_0`` is the offset/weight-aware intercept-only fit and
    ``f(y, μ_0)`` is the GLM-score residual produced by
    :func:`_glm_score_gradient_factor`. The intercept column and any other
    unpenalised columns are excluded from the max.

    The scaling intentionally matches the solver's (raw weighted sums, no
    implicit ``1/n``) — the legacy formula divided by ``n``, which under-sized
    the grid by roughly that factor and left penalised coefficients non-zero
    at ``alpha_max``.

    For pure ridge (``l1_ratio == 0``) the KKT condition does not produce a
    finite ``alpha_max``; we return a documented heuristic anchored on the
    median diagonal of the weighted Gram matrix (``Σ w X_j²``), preserved
    from the pre-RS-ACT-005 implementation but now correctly weight-aware.

    Parameters
    ----------
    X : ndarray, shape (n, p)
        Design matrix. Column ``intercept_col`` (default 0) is treated as the
        intercept and excluded from the score maximisation.
    y : ndarray, shape (n,)
        Response, on the data scale (counts for Poisson, etc.).
    l1_ratio : float in [0, 1]
        Elastic-net mix. ``0`` → pure ridge (heuristic grid); ``> 0`` → KKT
        formula above.
    family, link : str
        GLM family and link names; must match the fit you will run at the
        returned alpha for the result to be a tight upper bound.
    offset : ndarray, optional
        Link-scale offset that the fit will also use.
    weights : ndarray, optional
        Prior weights; defaults to ones. Treated as a strict weighting (no
        renormalisation by ``n``), matching the solver.
    var_power, theta : float
        Tweedie ``var_power`` and Negative-Binomial ``theta`` — required for
        the variance function on those families; ignored otherwise.
    intercept_col : int or None
        Column index of the intercept (excluded from the score). Pass ``None``
        if the design has no intercept (rare for GLMs).
    """
    # Resolve link to the family default when the caller passed None / "" —
    # ``FormulaGLMDict.link`` is allowed to stay unset until fit time, and we
    # used to break with ``unsupported link None`` for CV-regularised fits on
    # those models. Mirrors ``formula.py``'s ``link or get_default_link()`` idiom.
    if not link:
        from rustystats.formula import get_default_link

        link = get_default_link(family)

    n, p = X.shape
    if weights is None:
        w = np.ones(n)
    else:
        w = np.asarray(weights, dtype=np.float64)
    off = np.zeros(n) if offset is None else np.asarray(offset, dtype=np.float64)

    pen_mask = np.ones(p, dtype=bool)
    if intercept_col is not None and 0 <= intercept_col < p:
        pen_mask[intercept_col] = False

    if l1_ratio > 0:
        beta_0 = _fit_null_intercept(y, family, link, off, w, var_power, theta)
        eta_0 = beta_0 + off
        mu_0 = _apply_inverse_link(eta_0, link)
        score_factor = _glm_score_gradient_factor(y, mu_0, family, link, var_power, theta, w)
        # Per-feature unpenalised gradient magnitudes.
        scores = X.T @ score_factor
        if not np.any(pen_mask):
            return ALPHA_MAX_FLOOR
        max_score = float(np.max(np.abs(scores[pen_mask])))
        alpha_max = max_score / max(l1_ratio, L1_RATIO_MIN_CLAMP)
    else:
        # Ridge has no all-zero KKT; use the median weighted column norm as a
        # documented heuristic. Same shape as the pre-RS-ACT-005 fallback but
        # weight-aware and intercept-excluding, with no division by ``n`` so
        # the magnitude matches the solver's loss scale.
        if not np.any(pen_mask):
            return ALPHA_MAX_FLOOR
        XtX_diag = np.sum((X[:, pen_mask] ** 2) * w[:, None], axis=0)
        alpha_max = float(np.median(XtX_diag)) * 10.0 if XtX_diag.size > 0 else ALPHA_MAX_FLOOR

    return max(alpha_max, ALPHA_MAX_FLOOR)


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
    seed: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
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
    list[tuple[np.ndarray, np.ndarray]]
        List of (train_indices, val_indices) tuples
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    fold_sizes = np.full(n_folds, n // n_folds, dtype=int)
    fold_sizes[: n % n_folds] += 1

    folds = []
    current = 0
    for fold_size in fold_sizes:
        val_idx = indices[current : current + fold_size]
        train_idx = np.concatenate([indices[:current], indices[current + fold_size :]])
        folds.append((train_idx, val_idx))
        current += fold_size

    return folds


def compute_deviance(
    y: np.ndarray,
    mu: np.ndarray,
    family: str,
    theta: float = 1.0,
    weights: np.ndarray | None = None,
    var_power: float = 1.5,
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
    from rustystats._rustystats import (
        BinomialFamily,
        GammaFamily,
        GaussianFamily,
        NegativeBinomialFamily,
        PoissonFamily,
        QuasiBinomialFamily,
        QuasiPoissonFamily,
        TweedieFamily,
    )

    family_lower = family.lower()
    if family_lower in ("gaussian", "normal"):
        fam = GaussianFamily()
    elif family_lower == "poisson":
        fam = PoissonFamily()
    elif family_lower == "quasipoisson":
        fam = QuasiPoissonFamily()
    elif family_lower == "gamma":
        fam = GammaFamily()
    elif family_lower == "binomial":
        fam = BinomialFamily()
    elif family_lower == "quasibinomial":
        fam = QuasiBinomialFamily()
    elif family_lower.startswith(("negativebinomial", "negbinomial")) or family_lower in (
        "negative_binomial",
        "nb",
    ):
        fam = NegativeBinomialFamily(theta)
    elif family_lower.startswith("tweedie"):
        fam = TweedieFamily(var_power)
    else:
        raise ValidationError(f"Unknown family: {family}")

    unit_dev = np.asarray(fam.unit_deviance(y, mu), dtype=np.float64)
    if weights is None:
        return float(np.mean(unit_dev))

    w = np.asarray(weights, dtype=np.float64)
    denom = float(np.sum(w))
    if denom <= 0 or not np.isfinite(denom):
        return float("inf")
    return float(np.sum(w * unit_dev) / denom)


def select_optimal_alpha(
    path_results: list[RegularizationPathResult],
    selection: Literal["min", "1se"] = "min",
) -> RegularizationPathResult:
    """
    Select optimal alpha from path results.

    Parameters
    ----------
    path_results : list[RegularizationPathResult]
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
        raise FittingError("All regularization path fits failed")

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
        raise ValidationError(f"Unknown selection method: {selection}")


def fit_cv_regularization_path(
    glm_instance,
    cv: int = DEFAULT_CV_FOLDS,
    selection: Literal["min", "1se"] = "min",
    regularization: Literal["ridge", "lasso", "elastic_net"] = "ridge",
    n_alphas: int = DEFAULT_N_ALPHAS,
    alpha_min_ratio: float = DEFAULT_ALPHA_MIN_RATIO,
    l1_ratio: float | None = None,
    max_iter: int = DEFAULT_MAX_ITER,
    tol: float = DEFAULT_TOLERANCE,
    seed: int | None = None,
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
    tuple[result, RegularizationPathInfo]
        The fitted result at optimal alpha and the path info
    """
    from rustystats._rustystats import fit_glm_py as _fit_glm_rust

    # Determine l1_ratio based on regularization type
    if regularization == "ridge":
        effective_l1_ratio = 0.0
    elif regularization == "lasso":
        effective_l1_ratio = 1.0
    elif regularization == "elastic_net":
        effective_l1_ratio = l1_ratio if l1_ratio is not None else DEFAULT_ELASTIC_NET_L1_RATIO
    else:
        raise ValidationError(f"Unknown regularization type: {regularization}")

    X = glm_instance.X
    y = glm_instance.y
    family = glm_instance.family
    link = glm_instance.link
    var_power = glm_instance.var_power
    theta = glm_instance.theta if glm_instance.theta is not None else DEFAULT_NEGBINOMIAL_THETA
    offset = glm_instance.offset
    weights = glm_instance.weights

    # Compute alpha path: GLM-score-aware, offset+weight+family+link aware.
    alpha_max = compute_alpha_max(
        X,
        y,
        effective_l1_ratio,
        family=family,
        link=link,
        offset=offset,
        weights=weights,
        var_power=var_power,
        theta=theta,
    )
    alphas = generate_alpha_path(alpha_max, n_alphas, alpha_min_ratio)

    if verbose:
        print(f"Fitting regularization path: {regularization}")
        print(f"  Alpha range: {alphas[-1]:.6f} to {alphas[0]:.6f}")
        print(f"  L1 ratio: {effective_l1_ratio}")
        print(f"  CV folds: {cv}")

    # Use Rust parallel implementation (no fallback)
    from rustystats._rustystats import fit_cv_path_py as _fit_cv_path_rust
    from rustystats.formula import _get_constraint_indices

    if verbose:
        print("  Using Rust parallel CV")

    # CV fold fits use the requested convergence settings (RS-ACT-001). Silently
    # relaxing them can change which alpha is selected; a faster approximate mode,
    # if ever added, must be explicit and recorded in the path metadata below.
    cv_max_iter = max_iter
    cv_tol = tol

    # Pass sign constraints so CV folds respect monotonicity/pos/neg
    nonneg_indices, nonpos_indices = _get_constraint_indices(glm_instance.feature_names)

    rust_result = _fit_cv_path_rust(
        y,
        X,
        family,
        link,
        var_power,
        theta,
        offset,
        weights,
        list(alphas),
        effective_l1_ratio,
        cv,
        cv_max_iter,
        cv_tol,
        seed if seed is not None else DEFAULT_CV_SEED,
        nonneg_indices=nonneg_indices if nonneg_indices else None,
        nonpos_indices=nonpos_indices if nonpos_indices else None,
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
            weights_val = weights[val_idx] if weights is not None else None

            try:
                result = _fit_glm_rust(
                    y_train,
                    X_train,
                    family,
                    link,
                    var_power,
                    theta,
                    offset_train,
                    weights_train,
                    0.0,
                    0.0,
                    max_iter,
                    tol,
                )
            except ValueError:
                continue
            linear_pred = X_val @ result.params
            if offset_val is not None:
                linear_pred = linear_pred + offset_val
            mu_val = _apply_inverse_link(linear_pred, link)
            dev = compute_deviance(
                y_val,
                mu_val,
                family,
                theta=theta,
                weights=weights_val,
                var_power=var_power,
            )
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
        cv_max_iter=cv_max_iter,
        cv_tol=cv_tol,
    )

    return path_info


def build_fold_design_matrices(
    data,
    parsed,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    raw_exposure: np.ndarray | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build ``(X_train, X_val, names)`` with target encoding fit on training rows only.

    A fresh :class:`InteractionBuilder` is fit on the held-in rows, so every
    response-dependent encoding (target encoding, including interactions and the
    exposure-weighted variant) sees only training targets. The held-out rows are
    then transformed through that fold-training encoder, which maps any level
    absent from training to the training prior. This is what makes cross-validated
    alpha selection fold-safe for target-encoded terms (RS-ACT-001b): no
    validation target ever enters the encoding used to score a fold.

    Per-fold column counts may differ when a categorical level is absent from a
    fold's training rows; that is harmless because each fold fits and scores in
    its own column space and only the scalar validation deviance is aggregated.
    """
    from rustystats.interactions import InteractionBuilder

    data_train = data[train_idx]
    data_val = data[val_idx]
    exposure_train = raw_exposure[train_idx] if raw_exposure is not None else None

    builder = InteractionBuilder(data_train)
    _y_train, x_train, names = builder.build_design_matrix_from_parsed(
        parsed, exposure=exposure_train, seed=seed
    )
    x_val = builder.transform_new_data(data_val)
    return x_train, x_val, names


def fit_cv_te_regularization_path(
    glm_instance,
    cv: int = DEFAULT_CV_FOLDS,
    selection: Literal["min", "1se"] = "min",
    regularization: Literal["ridge", "lasso", "elastic_net"] = "ridge",
    n_alphas: int = DEFAULT_N_ALPHAS,
    alpha_min_ratio: float = DEFAULT_ALPHA_MIN_RATIO,
    l1_ratio: float | None = None,
    max_iter: int = DEFAULT_MAX_ITER,
    tol: float = DEFAULT_TOLERANCE,
    seed: int | None = None,
    include_unregularized: bool = True,
    verbose: bool = False,
) -> RegularizationPathInfo:
    """Fold-safe CV regularization path for models with target-encoded terms.

    Unlike the fast Rust array path (:func:`fit_cv_regularization_path`, which
    slices a design matrix built once on the full data), this rebuilds the design
    matrix per fold via :func:`build_fold_design_matrices` so each fold's target
    encoding is fit on its training rows only (RS-ACT-001b). Each candidate alpha
    is fit on the fold-training design and scored on the fold-validation design,
    mirroring the production fit -> predict pipeline. Non-target-encoded models
    should use the fast path; this one is reserved for the target-encoding case.
    """
    from rustystats._rustystats import fit_glm_py as _fit_glm_rust

    if regularization == "ridge":
        effective_l1_ratio = 0.0
    elif regularization == "lasso":
        effective_l1_ratio = 1.0
    elif regularization == "elastic_net":
        effective_l1_ratio = l1_ratio if l1_ratio is not None else DEFAULT_ELASTIC_NET_L1_RATIO
    else:
        raise ValidationError(f"Unknown regularization type: {regularization}")

    X = glm_instance.X
    y = glm_instance.y
    family = glm_instance.family
    link = glm_instance.link
    var_power = glm_instance.var_power
    theta = glm_instance.theta if glm_instance.theta is not None else DEFAULT_NEGBINOMIAL_THETA
    offset = glm_instance.offset
    weights = glm_instance.weights

    parsed = glm_instance._builder._parsed_formula
    data = glm_instance.data
    raw_exposure = getattr(glm_instance, "_raw_exposure", None)

    # CV fold fits use the requested convergence settings (RS-ACT-001), never a
    # silently relaxed mode that could change which alpha is selected.
    cv_max_iter = max_iter
    cv_tol = tol

    if verbose:
        print(f"Fold-safe target-encoding CV: {regularization}, {cv} folds, {n_alphas} alphas")

    folds = create_cv_folds(len(y), cv, seed)
    from rustystats.formula import _get_constraint_indices

    fold_alpha_maxes = []
    for train_idx, val_idx in folds:
        x_train, _x_val, _names = build_fold_design_matrices(
            data, parsed, train_idx, val_idx, raw_exposure=raw_exposure, seed=seed
        )
        y_train = y[train_idx]
        weights_train = weights[train_idx] if weights is not None else None
        offset_train = offset[train_idx] if offset is not None else None
        fold_alpha_maxes.append(
            compute_alpha_max(
                x_train,
                y_train,
                effective_l1_ratio,
                family=family,
                link=link,
                offset=offset_train,
                weights=weights_train,
                var_power=var_power,
                theta=theta,
            )
        )

    # Build the alpha search range only from fold-training designs. Taking the
    # maximum keeps the grid regularized enough for every fold without letting
    # validation targets enter a full-data target-encoded design.
    alpha_max = max(fold_alpha_maxes)
    alphas = list(generate_alpha_path(alpha_max, n_alphas, alpha_min_ratio))

    # alpha -> per-fold validation deviances; alpha == 0.0 is the unregularized fit.
    candidate_alphas = list(alphas)
    if include_unregularized:
        candidate_alphas.append(0.0)
    deviances: dict[float, list[float]] = {a: [] for a in candidate_alphas}
    failed_alphas: set[float] = set()

    for train_idx, val_idx in folds:
        x_train, x_val, names = build_fold_design_matrices(
            data, parsed, train_idx, val_idx, raw_exposure=raw_exposure, seed=seed
        )
        y_train = y[train_idx]
        y_val = y[val_idx]
        offset_train = offset[train_idx] if offset is not None else None
        offset_val = offset[val_idx] if offset is not None else None
        weights_train = weights[train_idx] if weights is not None else None
        weights_val = weights[val_idx] if weights is not None else None
        nonneg_indices, nonpos_indices = _get_constraint_indices(names)

        for alpha in candidate_alphas:
            if alpha in failed_alphas:
                continue
            try:
                result = _fit_glm_rust(
                    y_train,
                    x_train,
                    family,
                    link,
                    var_power,
                    theta,
                    offset_train,
                    weights_train,
                    alpha,
                    effective_l1_ratio,
                    cv_max_iter,
                    cv_tol,
                    nonneg_indices if nonneg_indices else None,
                    nonpos_indices if nonpos_indices else None,
                    False,
                )
            except ValueError:
                failed_alphas.add(alpha)
                continue
            linear_pred = x_val @ result.params
            if offset_val is not None:
                linear_pred = linear_pred + offset_val
            mu_val = _apply_inverse_link(linear_pred, link)
            dev = compute_deviance(
                y_val,
                mu_val,
                family,
                theta=theta,
                weights=weights_val,
                var_power=var_power,
            )
            if np.isfinite(dev):
                deviances[alpha].append(dev)
            else:
                failed_alphas.add(alpha)

    n_nonzero = X.shape[1] - 1
    path_results = []
    for alpha in candidate_alphas:
        if alpha in failed_alphas:
            continue
        fold_devs = deviances[alpha]
        if len(fold_devs) != len(folds):
            continue
        path_results.append(
            RegularizationPathResult(
                alpha=alpha,
                l1_ratio=0.0 if alpha == 0.0 else effective_l1_ratio,
                cv_deviance_mean=float(np.mean(fold_devs)),
                cv_deviance_se=float(np.std(fold_devs) / np.sqrt(len(fold_devs))),
                n_nonzero=n_nonzero,
                max_coef=0.0,
            )
        )

    if not path_results:
        raise ValidationError(
            "Fold-safe target-encoding CV produced no finite validation deviances; "
            "check the data, family, and fold count."
        )

    best = select_optimal_alpha(path_results, selection)

    if verbose:
        print(f"\nSelected: alpha={best.alpha:.6f}, CV deviance={best.cv_deviance_mean:.6f}")

    if best.alpha == 0.0:
        reg_type = "none"
    elif effective_l1_ratio >= 1.0:
        reg_type = "lasso"
    elif effective_l1_ratio <= 0.0:
        reg_type = "ridge"
    else:
        reg_type = "elastic_net"

    return RegularizationPathInfo(
        selected_alpha=best.alpha,
        selected_l1_ratio=best.l1_ratio,
        cv_deviance=best.cv_deviance_mean,
        cv_deviance_se=best.cv_deviance_se,
        selection_method=selection,
        regularization_type=reg_type,
        path=path_results,
        n_folds=cv,
        cv_max_iter=cv_max_iter,
        cv_tol=cv_tol,
        fold_safe_target_encoding=True,
    )
