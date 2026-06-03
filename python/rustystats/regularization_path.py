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

NB theta policy (RS-ACT-010): the CV path is fixed-theta only — every fit on
the path uses the ``theta`` resolved by
:func:`rustystats.formula.FormulaGLMDict._resolve_negbinomial_theta`, which
fails closed and refuses to estimate theta on regularized fits. Callers must
pass an explicit numeric ``theta=`` for regularized Negative Binomial models.

Example
-------
>>> import rustystats as rs
>>>
>>> model = rs.glm_dict(
...     response="ClaimCount",
...     terms={"VehAge": {"type": "linear"}, "BonusMalus": {"type": "linear"}, "Region": {"type": "target_encoding"}},
...     data=train_df,
...     family="negbinomial",
...     exposure="Exposure",
... )
>>>
>>> # Fit with CV-based regularization selection
>>> result = model.fit(cv=5, selection="1se", regularization="ridge")
>>>
>>> print(f"Selected alpha: {result.alpha}")
>>> print(f"CV deviance: {result.cv_deviance}")
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Literal

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
)
from rustystats.exceptions import FittingError, ValidationError


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
        Number of non-zero coefficients at this alpha. NOTE: the fold-safe
        target-encoding CV path selects alpha by validation deviance without
        refitting coefficients per alpha, so on that route this reports the
        full-data design column count (the design width), not a post-fit count.
    max_coef : float
        Maximum absolute coefficient value at this alpha, or ``0.0`` on the
        fold-safe target-encoding CV path, which does not refit coefficients
        during alpha selection.
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
    cv_max_iter : int
        Maximum IRLS iterations actually used for every per-fold fit on the
        regularization path (RS-ACT-001: pinned to the caller's ``max_iter``,
        never silently relaxed).
    cv_tol : float
        Convergence tolerance actually used for every per-fold fit on the
        regularization path (RS-ACT-001: pinned to the caller's ``tol``,
        never silently relaxed).
    fold_safe_target_encoding : bool
        Whether the fold-safe target-encoding CV path was used (RS-ACT-001b).
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
    cv_fold_scores: dict[float, list[float]] | None = None
    cv_scoring_objective: str = "weighted_mean_unit_deviance"


def _apply_inverse_link(eta: np.ndarray, link: str) -> np.ndarray:
    """Apply inverse link function to linear predictor.

    Delegates to the shared implementation in formula.py which raises
    on unknown links instead of silently defaulting.
    """
    from rustystats.formula import apply_inverse_link

    return apply_inverse_link(eta, link)


def _alpha_family_base(family: str) -> str:
    """Normalize family strings used by alpha_max helpers."""
    family_base = family.lower().split("(", 1)[0].strip()
    if family_base == "normal":
        return "gaussian"
    if family_base in ("negativebinomial", "negative_binomial", "nb"):
        return "negbinomial"
    return family_base


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

    family = _alpha_family_base(family)

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
    allow_extended_tweedie: bool = False,
) -> float:
    """Fit the intercept-only GLM (``η = β_0 + offset``).

    Closed-form for score equations where the intercept solve is exact; all
    other family/link combinations fall back to a single Rust ``fit_glm_py``
    call on a width-1 design — same code path as a normal fit, so anything
    that converges in production converges here.
    """
    family = _alpha_family_base(family)

    # Closed-form for Poisson-family log links:
    # score = Σw(y - exp(β0 + offset)) = 0.
    if family in ("poisson", "quasipoisson") and link == "log":
        # Guard against inf from large offsets — Σ w exp(offset) overflows
        # silently to inf and the log(num/inf) = -inf branch would poison the
        # alpha grid. When the closed form is unusable, fall through to the
        # Rust IRLS path below intentionally.
        denom = float(np.sum(weights * np.exp(offset)))
        if np.isfinite(denom) and denom > 0.0:
            num = float(np.sum(weights * y))
            if num > 0.0:
                return float(np.log(num / denom))
    # Gaussian identity with offset: β_0 minimises Σ w (y - β_0 - offset)² →
    # β_0 = Σw(y - offset) / Σw.
    if family == "gaussian" and link == "identity":
        sum_w = float(np.sum(weights))
        if not np.isfinite(sum_w) or sum_w <= 0.0:
            raise ValidationError(
                "Gaussian-identity intercept solve requires sum(weights) > 0 and finite, "
                f"got {sum_w}."
            )
        return float(np.sum(weights * (y - offset)) / sum_w)

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
        allow_extended_tweedie,
        True,
    )
    return float(np.asarray(null_res.params)[0])


# Columns whose weighted standard deviation falls below this are treated as
# constant: left unstandardized (center=0, scale=1) so the penalty path never
# divides by ~0. Sits well above f64 round-off yet below any real predictor sd.
MIN_WEIGHTED_STD = 1e-12


def penalized_column_mask(n_cols: int, fit_intercept: bool) -> np.ndarray:
    """Boolean mask of the columns a global penalty applies to.

    Every column is penalized except the intercept (assumed at index 0 when
    ``fit_intercept``). Shared by the standardization and alpha-grid paths so
    they cannot disagree about which columns the penalty — and therefore the
    standardization — acts on.
    """
    mask = np.ones(n_cols, dtype=bool)
    if fit_intercept and n_cols > 0:
        mask[0] = False
    return mask


def compute_standardization(
    X: np.ndarray,
    weights: np.ndarray | None = None,
    pen_mask: np.ndarray | None = None,
    *,
    fit_intercept: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted center/scale vectors for internally standardized penalties.

    The returned vectors are length ``p`` and safe to pass straight to the Rust
    fit boundary: unpenalized columns, intercept columns, constant columns and
    no-intercept fits receive ``center=0``; non-constant penalized columns get a
    weighted mean and population standard deviation.
    """
    x = np.asarray(X, dtype=np.float64)
    if x.ndim != 2:
        raise ValidationError(f"X must be a 2D design matrix, got shape {x.shape}.")
    n, p = x.shape
    if p == 0:
        return np.zeros(0, dtype=np.float64), np.ones(0, dtype=np.float64)

    if weights is None:
        w = np.ones(n, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.ndim != 1 or w.shape[0] != n:
            raise ValidationError(
                f"weights must be length {n} for standardization, got shape {w.shape}."
            )
    weight_sum = float(np.sum(w))
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        raise ValidationError(
            f"standardization requires positive finite sum(weights), got {weight_sum}."
        )

    if pen_mask is None:
        mask = penalized_column_mask(p, fit_intercept)
    else:
        mask = np.asarray(pen_mask, dtype=bool)
        if mask.ndim != 1 or mask.shape[0] != p:
            raise ValidationError(
                f"pen_mask must be length {p} for standardization, got shape {mask.shape}."
            )
        mask = mask.copy()
    if fit_intercept and p > 0:
        mask[0] = False

    center = np.zeros(p, dtype=np.float64)
    scale = np.ones(p, dtype=np.float64)
    active = np.flatnonzero(mask)
    if active.size == 0:
        return center, scale

    cols = x[:, active]
    mean = (w @ cols) / weight_sum
    # Two-pass weighted variance: centre before squaring. The one-pass
    # ``E[x²] - E[x]²`` loses precision to catastrophic cancellation on the
    # very high-magnitude columns this standardization exists to tame — a
    # mean ~1e10 column can lose its variance entirely, which would clamp a
    # genuinely varying column to "constant" below and let it escape the
    # penalty (reintroducing the RS-ACT-012 defect for that column).
    centered = cols - mean
    var = np.maximum((w @ (centered * centered)) / weight_sum, 0.0)
    sd = np.sqrt(var)
    valid = np.isfinite(mean) & np.isfinite(sd) & (sd > MIN_WEIGHTED_STD)
    valid_cols = active[valid]
    if valid_cols.size == 0:
        return center, scale

    if fit_intercept:
        center[valid_cols] = mean[valid]
    scale[valid_cols] = sd[valid]
    return center, scale


def solver_standardization(
    center: np.ndarray | None,
    scale: np.ndarray | None,
    *,
    fit_intercept: bool = True,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return the affine transform to pass to the Rust solver.

    The alpha-grid calculation still uses the weighted centers from
    :func:`compute_standardization`. For the actual penalized fit, an
    unpenalized intercept can absorb column centering exactly, while keeping a
    scale-only design preserves sparse zero structure in one-hot-heavy models.
    """
    if center is None or scale is None or not fit_intercept:
        return center, scale
    return np.zeros_like(center), scale


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
    center: np.ndarray | None = None,
    scale: np.ndarray | None = None,
    pen_mask: np.ndarray | None = None,
    allow_extended_tweedie: bool = False,
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
    family = _alpha_family_base(family)

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

    if pen_mask is None:
        penalty_mask = np.ones(p, dtype=bool)
        if intercept_col is not None and 0 <= intercept_col < p:
            penalty_mask[intercept_col] = False
    else:
        penalty_mask = np.asarray(pen_mask, dtype=bool)
        if penalty_mask.ndim != 1 or penalty_mask.shape[0] != p:
            raise ValidationError(
                f"pen_mask must be length {p} for alpha_max, got shape {penalty_mask.shape}."
            )
        penalty_mask = penalty_mask.copy()
        if intercept_col is not None and 0 <= intercept_col < p:
            penalty_mask[intercept_col] = False

    use_standardization = center is not None or scale is not None
    if use_standardization:
        if center is None or scale is None:
            raise ValidationError("center and scale must be provided together for alpha_max.")
        c = np.asarray(center, dtype=np.float64)
        s = np.asarray(scale, dtype=np.float64)
        if c.shape != (p,) or s.shape != (p,):
            raise ValidationError(
                f"center/scale must be length {p} for alpha_max, "
                f"got center={c.shape}, scale={s.shape}."
            )
        if np.any(~np.isfinite(c)) or np.any(~np.isfinite(s)) or np.any(s <= 0.0):
            raise ValidationError("center/scale for alpha_max must be finite with scale > 0.")
    else:
        c = None
        s = None

    if l1_ratio > 0:
        beta_0 = _fit_null_intercept(
            y,
            family,
            link,
            off,
            w,
            var_power,
            theta,
            allow_extended_tweedie=allow_extended_tweedie,
        )
        eta_0 = beta_0 + off
        mu_0 = _apply_inverse_link(eta_0, link)
        score_factor = _glm_score_gradient_factor(y, mu_0, family, link, var_power, theta, w)
        # Per-feature unpenalised gradient magnitudes.
        scores = X.T @ score_factor
        if use_standardization:
            # Centering term is included explicitly instead of assuming the
            # intercept score is exactly zero; this keeps the endpoint stable
            # across non-canonical links and fallback null fits.
            scores = (scores - c * float(np.sum(score_factor))) / s
        if not np.any(penalty_mask):
            return ALPHA_MAX_FLOOR
        max_score = float(np.max(np.abs(scores[penalty_mask])))
        alpha_max = max_score / l1_ratio
    else:
        # Ridge has no all-zero KKT; use the median weighted column norm as a
        # documented heuristic. Same shape as the pre-RS-ACT-005 fallback but
        # weight-aware and intercept-excluding, with no division by ``n`` so
        # the magnitude matches the solver's loss scale.
        if not np.any(penalty_mask):
            return ALPHA_MAX_FLOOR
        if use_standardization:
            x_pen = (X[:, penalty_mask] - c[penalty_mask][None, :]) / s[penalty_mask][None, :]
            XtX_diag = np.sum((x_pen**2) * w[:, None], axis=0)
        else:
            XtX_diag = np.sum((X[:, penalty_mask] ** 2) * w[:, None], axis=0)
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
    allow_extended_tweedie: bool = False,
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
        fam = TweedieFamily(var_power, allow_extended_tweedie=allow_extended_tweedie)
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
    standardize: bool = True,
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
    allow_extended_tweedie = bool(getattr(glm_instance, "allow_extended_tweedie", False))
    fit_intercept = bool(getattr(glm_instance, "intercept", True))

    pen_mask = penalized_column_mask(X.shape[1], fit_intercept)
    center = scale = None
    if standardize:
        center, scale = compute_standardization(
            X,
            weights,
            pen_mask,
            fit_intercept=fit_intercept,
        )

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
        intercept_col=0 if fit_intercept else None,
        center=center,
        scale=scale,
        pen_mask=pen_mask,
        allow_extended_tweedie=allow_extended_tweedie,
    )
    alphas = generate_alpha_path(alpha_max, n_alphas, alpha_min_ratio)
    if include_unregularized and (alphas.size == 0 or not np.any(alphas == 0.0)):
        # Score alpha=0 inside the Rust CV path so it uses the same fold
        # assignment as every regularized candidate.
        alphas = np.concatenate([alphas, np.array([0.0], dtype=np.float64)])

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
    solver_center, solver_scale = solver_standardization(
        center,
        scale,
        fit_intercept=fit_intercept,
    )

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
        allow_extended_tweedie=allow_extended_tweedie,
        fit_intercept=fit_intercept,
        center=solver_center,
        scale=solver_scale,
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
        cv_fold_scores={
            float(rust_result["alphas"][i]): list(map(float, scores))
            for i, scores in enumerate(rust_result.get("cv_fold_scores", []))
        }
        or None,
        cv_scoring_objective="weighted_mean_unit_deviance",
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

    parsed_fold = copy.deepcopy(parsed)
    # ``ParsedFormula`` carries mutable ``SplineTerm`` objects. The full-data
    # model has already transformed them once before CV, so their computed
    # knots may contain validation-only values. Reset fold copies so spline
    # state is fit from fold-training rows only, just like target encoders.
    for spline in getattr(parsed_fold, "spline_terms", []):
        spline._computed_boundary_knots = None
        spline._computed_internal_knots = None
        spline._penalty_matrix = None
        spline._lambda = None
        spline._edf = None
    for attr in ("_spline_by_var", "_te_by_var"):
        if hasattr(parsed_fold, attr):
            setattr(parsed_fold, attr, None)

    builder = InteractionBuilder(data_train)
    _y_train, x_train, names = builder.build_design_matrix_from_parsed(
        parsed_fold, exposure=exposure_train, seed=seed
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
    standardize: bool = True,
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
    allow_extended_tweedie = bool(getattr(glm_instance, "allow_extended_tweedie", False))
    fit_intercept = bool(getattr(glm_instance, "intercept", True))

    parsed = glm_instance._builder._parsed_formula
    data = glm_instance.data
    raw_exposure = getattr(glm_instance, "_raw_exposure", None)

    # CV fold fits use the requested convergence settings (RS-ACT-001), never a
    # silently relaxed mode that could change which alpha is selected.
    cv_max_iter = max_iter
    cv_tol = tol

    if verbose:
        print(f"Fold-safe target-encoding CV: {regularization}, {cv} folds, {n_alphas} alphas")

    # Default the CV seed to DEFAULT_CV_SEED for determinism. NOTE: only the seed
    # value is shared with the fast Rust array path — NOT the resulting fold
    # partition. This path splits a seeded ``permutation(n)`` into contiguous
    # blocks (``create_cv_folds``); the Rust path assigns each row to a fold via
    # ``(row, seed)`` hashing. For the same seed the two produce different
    # partitions, so CV deviances and the selected alpha are not comparable
    # across the two routes (each is internally deterministic). A model silently
    # switches routes based on whether it has a target-encoded/spline term.
    cv_seed = seed if seed is not None else DEFAULT_CV_SEED
    folds = create_cv_folds(len(y), cv, cv_seed)
    from rustystats.formula import _get_constraint_indices

    # Cache the per-fold designs from the alpha-grid pass so the fit pass below
    # can reuse them — building TE designs is the expensive step (sort + per-
    # level shrinkage), and recomputing them once per fold roughly halves the
    # builder workload on TE CV fits. ``build_fold_design_matrices`` is
    # deterministic in ``(data, parsed, train_idx, val_idx, raw_exposure, seed)``,
    # so caching is behaviour-preserving.
    fold_designs: list[
        tuple[np.ndarray, np.ndarray, list[str], np.ndarray | None, np.ndarray | None]
    ] = []
    fold_alpha_maxes = []
    for train_idx, val_idx in folds:
        x_train, x_val, names = build_fold_design_matrices(
            data, parsed, train_idx, val_idx, raw_exposure=raw_exposure, seed=cv_seed
        )
        y_train = y[train_idx]
        weights_train = weights[train_idx] if weights is not None else None
        offset_train = offset[train_idx] if offset is not None else None
        fold_center = fold_scale = None
        fold_pen_mask = penalized_column_mask(x_train.shape[1], fit_intercept)
        if standardize:
            fold_center, fold_scale = compute_standardization(
                x_train,
                weights_train,
                fold_pen_mask,
                fit_intercept=fit_intercept,
            )
        solver_center, solver_scale = solver_standardization(
            fold_center,
            fold_scale,
            fit_intercept=fit_intercept,
        )
        fold_designs.append((x_train, x_val, names, solver_center, solver_scale))
        try:
            fold_alpha_max = compute_alpha_max(
                x_train,
                y_train,
                effective_l1_ratio,
                family=family,
                link=link,
                offset=offset_train,
                weights=weights_train,
                var_power=var_power,
                theta=theta,
                intercept_col=0 if fit_intercept else None,
                center=fold_center,
                scale=fold_scale,
                pen_mask=fold_pen_mask,
                allow_extended_tweedie=allow_extended_tweedie,
            )
        except (ValidationError, FittingError, ValueError, RuntimeError) as exc:
            # A degenerate fold (all-zero counts, perfect separation, a singular
            # null-intercept solve) must not abort the entire CV: the per-alpha
            # fits below already fail closed, so the alpha_max that *gates* the
            # grid must too. Floor this fold's contribution instead of letting a
            # solver error escape.
            if verbose:
                print(f"  fold alpha_max computation failed ({exc!r}); flooring this fold.")
            fold_alpha_max = ALPHA_MAX_FLOOR
        fold_alpha_maxes.append(fold_alpha_max)

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

    for (train_idx, val_idx), (x_train, x_val, names, fold_center, fold_scale) in zip(
        folds, fold_designs, strict=True
    ):
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
                    allow_extended_tweedie,
                    fit_intercept,
                    fold_center,
                    fold_scale,
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
                allow_extended_tweedie=allow_extended_tweedie,
            )
            if np.isfinite(dev):
                deviances[alpha].append(dev)
            else:
                failed_alphas.add(alpha)

    # ``n_nonzero`` is approximated from the full-data design column count
    # (minus intercept). Per-fold target-encoded designs can have a slightly
    # different column count when a categorical level is absent from a fold's
    # training rows; we report the full-data column count for stability rather
    # than averaging across folds. The penalised refit on the full data after
    # alpha selection always has the full-data column count.
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
                cv_deviance_se=float(
                    np.std(fold_devs, ddof=1) / np.sqrt(len(fold_devs))
                    if len(fold_devs) > 1
                    else 0.0
                ),
                # See n_nonzero comment above — approximate from full-data design.
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
        cv_fold_scores={float(a): list(map(float, ds)) for a, ds in deviances.items()},
        cv_scoring_objective="weighted_mean_unit_deviance",
    )
