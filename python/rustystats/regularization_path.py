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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

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
from rustystats.solver_trace import SolverStep, solver_steps

if TYPE_CHECKING:
    from rustystats.formula import ParsedFormula


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
        Number of non-zero coefficients at this alpha. NOTE: the scalar GLM
        fold-safe target-encoding CV route selects alpha by validation deviance
        without refitting coefficients per alpha, so on that route this reports
        the full-data design column count (the design width), not a post-fit
        count. The multinomial fold-safe-TE CV path refits per (fold, alpha) and
        reports the genuine averaged post-fit non-zero count.
    max_coef : float
        Maximum absolute coefficient value at this alpha, or ``0.0`` on the
        scalar GLM fold-safe target-encoding CV route, which does not refit
        coefficients during alpha selection. The multinomial fold-safe-TE CV
        path reports the genuine maximum absolute coefficient from its per-fold
        refits.
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
    cv_profile: dict[str, object] | None = None
    final_fit_center: np.ndarray | None = None
    final_fit_scale: np.ndarray | None = None
    #: The alpha_max and CV drivers' steps (§4.5), which the result's trace starts with.
    solver_trace: tuple = ()


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


def compute_standardization_with_ridge_diag(
    X: np.ndarray,
    weights: np.ndarray | None = None,
    pen_mask: np.ndarray | None = None,
    *,
    fit_intercept: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weighted center/scale vectors plus ridge-grid column norms.

    The returned center/scale vectors are length ``p`` and safe to pass straight
    to the Rust fit boundary: unpenalized columns, intercept columns, constant
    columns and no-intercept fits receive ``center=0``; non-constant penalized
    columns get a weighted mean and population standard deviation.

    The third return value is also length ``p``. For each penalized column, it
    contains ``Σ w * ((x - center) / scale)^2`` in the alpha-grid coordinate
    system. Ridge alpha-max uses this exact diagonal, avoiding a second scan
    over the design matrix after standardization.
    """
    x = np.asarray(X, dtype=np.float64)
    if x.ndim != 2:
        raise ValidationError(f"X must be a 2D design matrix, got shape {x.shape}.")
    n, p = x.shape
    if p == 0:
        return (
            np.zeros(0, dtype=np.float64),
            np.ones(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
        )

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

    # The moments are computed in Rust with the block-ordered reduction every fit
    # shares (specs/out-of-core-fit.md §6.7), so they do not depend on thread count.
    from rustystats._rustystats import compute_standardization_py as _rust_standardization

    return _rust_standardization(x, w, mask, fit_intercept)


def compute_standardization(
    X: np.ndarray,
    weights: np.ndarray | None = None,
    pen_mask: np.ndarray | None = None,
    *,
    fit_intercept: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted center/scale vectors for internally standardized penalties."""
    center, scale, _ridge_xtx_diag = compute_standardization_with_ridge_diag(
        X,
        weights,
        pen_mask,
        fit_intercept=fit_intercept,
    )
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
    ridge_xtx_diag: np.ndarray | None = None,
    pen_mask: np.ndarray | None = None,
    allow_extended_tweedie: bool = False,
    trace: list | None = None,
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
    ridge_xtx_diag : ndarray, optional
        Precomputed ``Σw * X_j²`` diagonal in the coordinate system used for the
        ridge alpha grid. When supplied for ``l1_ratio == 0``, this avoids a
        second full scan of the design matrix after standardization.
    """
    family = _alpha_family_base(family)

    # Resolve link to the family default when the caller passed None / "" —
    # ``FormulaGLMDict.link`` is allowed to stay unset until fit time, and we
    # used to break with ``unsupported link None`` for CV-regularised fits on
    # those models. Mirrors ``formula.py``'s ``link or get_default_link()`` idiom.
    if not link:
        from rustystats.formula import get_default_link

        link = get_default_link(family)

    _n, p = X.shape

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

    ridge_diag = None
    if ridge_xtx_diag is not None:
        ridge_diag = np.asarray(ridge_xtx_diag, dtype=np.float64)
        if ridge_diag.shape != (p,):
            raise ValidationError(
                f"ridge_xtx_diag must be length {p} for alpha_max, got shape {ridge_diag.shape}."
            )
        if np.any(~np.isfinite(ridge_diag)) or np.any(ridge_diag < 0.0):
            raise ValidationError("ridge_xtx_diag for alpha_max must be finite and non-negative.")

    # The null fit, the score and the column norms are block-ordered reductions in
    # Rust (specs/out-of-core-fit.md §5.3, §7.8), the same an out-of-core fit runs.
    from rustystats._rustystats import alpha_max_py

    try:
        value = alpha_max_py(
            np.ascontiguousarray(y, dtype=np.float64),
            np.ascontiguousarray(X, dtype=np.float64),
            family,
            link,
            float(l1_ratio),
            offset=None if offset is None else np.ascontiguousarray(offset, dtype=np.float64),
            weights=None if weights is None else np.ascontiguousarray(weights, dtype=np.float64),
            var_power=float(var_power),
            theta=float(theta),
            pen_mask=[bool(v) for v in penalty_mask],
            intercept_col=None,
            center=c,
            scale=s,
            ridge_xtx_diag=None if ridge_diag is None else [float(v) for v in ridge_diag],
            allow_extended_tweedie=allow_extended_tweedie,
            trace=trace,
        )
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    return float(value)


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
    ridge_xtx_diag = None
    if standardize:
        center, scale, ridge_xtx_diag = compute_standardization_with_ridge_diag(
            X,
            weights,
            pen_mask,
            fit_intercept=fit_intercept,
        )

    # Compute alpha path: GLM-score-aware, offset+weight+family+link aware.
    trace: list = []
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
        ridge_xtx_diag=ridge_xtx_diag,
        pen_mask=pen_mask,
        allow_extended_tweedie=allow_extended_tweedie,
        trace=trace,
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
    from rustystats.frozen_design import builder_max_row_nonzeros

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
        # The frozen design's structural nonzeros per row (§5.2).
        max_row_nonzeros=builder_max_row_nonzeros(getattr(glm_instance, "_builder", None)),
    )

    return cv_path_info(
        rust_result,
        l1_ratio=effective_l1_ratio,
        selection=selection,
        n_folds=cv,
        n_params=X.shape[1],
        cv_max_iter=cv_max_iter,
        cv_tol=cv_tol,
        final_fit_center=solver_center,
        final_fit_scale=solver_scale,
        verbose=verbose,
        solver_trace=trace,
    )


def cv_path_info(
    rust_result: dict,
    *,
    l1_ratio: float,
    selection: str,
    n_folds: int,
    n_params: int,
    cv_max_iter: int,
    cv_tol: float,
    final_fit_center: np.ndarray | None,
    final_fit_scale: np.ndarray | None,
    verbose: bool = False,
    solver_trace: list | tuple = (),
) -> RegularizationPathInfo:
    """The selected alpha and path metadata of a Rust CV path result.

    ``solver_trace`` holds the steps before the path (alpha_max's); the path's own
    follow from ``rust_result["solver_trace"]``.
    """
    effective_l1_ratio = l1_ratio
    path_results = [
        RegularizationPathResult(
            alpha=rust_result["alphas"][i],
            l1_ratio=effective_l1_ratio,
            cv_deviance_mean=rust_result["cv_deviance_mean"][i],
            cv_deviance_se=rust_result["cv_deviance_se"][i],
            n_nonzero=n_params - 1,
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

    return RegularizationPathInfo(
        selected_alpha=best.alpha,
        selected_l1_ratio=best.l1_ratio,
        cv_deviance=best.cv_deviance_mean,
        cv_deviance_se=best.cv_deviance_se,
        selection_method=selection,
        regularization_type=reg_type,
        path=path_results,
        n_folds=n_folds,
        cv_max_iter=cv_max_iter,
        cv_tol=cv_tol,
        cv_fold_scores={
            float(rust_result["alphas"][i]): list(map(float, scores))
            for i, scores in enumerate(rust_result.get("cv_fold_scores", []))
        }
        or None,
        cv_scoring_objective="weighted_mean_unit_deviance",
        cv_profile=dict(rust_result["profile"]) if "profile" in rust_result else None,
        final_fit_center=final_fit_center,
        final_fit_scale=final_fit_scale,
        solver_trace=solver_steps(solver_trace, rust_result.get("solver_trace")),
    )


def reset_fold_local_spline_state(parsed: ParsedFormula) -> None:
    """Reset mutable per-fit ``SplineTerm`` state on a (deep-copied) ParsedFormula.

    The full-data model transforms splines once before CV, caching computed knots
    that may contain validation-only values. Fold copies must refit spline state
    from fold-training rows only, just like target encoders. Shared by the scalar
    fold-safe CV path and the multinomial fold builder so a new mutable SplineTerm
    field only needs updating in one place.
    """
    for spline in getattr(parsed, "spline_terms", []):
        spline._computed_boundary_knots = None
        spline._computed_internal_knots = None
        spline._penalty_matrix = None
        spline._lambda = None
        spline._edf = None
    for attr in ("_spline_by_var", "_te_by_var"):
        if hasattr(parsed, attr):
            setattr(parsed, attr, None)


# ── Cross-validation with target encoding (specs/out-of-core-fit.md §7.9) ──────


def _effective_l1_ratio(regularization: str, l1_ratio: float | None) -> float:
    if regularization == "ridge":
        return 0.0
    if regularization == "lasso":
        return 1.0
    if regularization == "elastic_net":
        return l1_ratio if l1_ratio is not None else DEFAULT_ELASTIC_NET_L1_RATIO
    raise ValidationError(f"Unknown regularization type: {regularization}")


class TargetEncodedFolds(Protocol):
    """One fit mode's view of the K fold designs of a target-encoded model (§7.9).

    Fold k's design was learned from its training rows and encodes every row:
    training rows with its training values, validation rows from its full
    statistics. Each method reduces over the rows of that design, in blocks of
    global rows, so the in-memory and out-of-core modes give identical bits.
    """

    def names(self, fold: int) -> list[str]: ...

    def moments(
        self, fold: int, pen_mask: np.ndarray, fit_intercept: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Standardisation (centre, scale, ridge diagonal) over the training rows."""
        ...

    def alpha_max(
        self,
        fold: int,
        *,
        family: str,
        link: str,
        l1_ratio: float,
        pen_mask: np.ndarray,
        center: np.ndarray | None,
        scale: np.ndarray | None,
        ridge_xtx_diag: np.ndarray | None,
        trace: list,
    ) -> float:
        """alpha_max over the training rows; its steps are appended to ``trace``."""
        ...

    def path(
        self,
        fold: int,
        alphas: list[float],
        *,
        l1_ratio: float,
        max_iter: int,
        tol: float,
        nonneg: list[int],
        nonpos: list[int],
        center: np.ndarray | None,
        scale: np.ndarray | None,
    ) -> dict:
        """The fold's warm-started path, scored on its validation rows (its steps in
        ``"solver_trace"``)."""
        ...

    def release(self, fold: int) -> None:
        """Fold ``fold``'s path has ended: free what its design holds."""
        ...


def target_encoded_cv_path(
    folds: TargetEncodedFolds,
    *,
    n_folds: int,
    n_params: int,
    family: str,
    link: str | None,
    regularization: str,
    l1_ratio: float | None,
    n_alphas: int,
    alpha_min_ratio: float,
    max_iter: int,
    tol: float,
    include_unregularized: bool,
    standardize: bool,
    fit_intercept: bool,
    selection: str,
    verbose: bool,
) -> RegularizationPathInfo:
    """The fold-safe CV path of a target-encoded model, in either fit mode (§7.9).

    Every fold's design is learned from its training rows, so no validation target
    enters the encoding that scores it (RS-ACT-001b). alpha_max is the maximum of the
    folds' training-row alpha_max (a fold whose computation fails contributes the
    floor). The folds then run in fold order; an alpha that failed in an earlier fold
    is dropped from later folds' grids, as their warm-start chains always were.
    """
    from rustystats.formula import _get_constraint_indices, get_default_link

    effective_l1_ratio = _effective_l1_ratio(regularization, l1_ratio)
    if verbose:
        print(f"Fold-safe target-encoding CV: {regularization}, {n_folds} folds, {n_alphas} alphas")
    alpha_family = _alpha_family_base(family)
    alpha_link = link or get_default_link(alpha_family)

    standardizations: list[tuple[np.ndarray | None, np.ndarray | None]] = []
    fold_alpha_maxes: list[float] = []
    trace: list = []
    for fold in range(n_folds):
        p = len(folds.names(fold))
        pen_mask = penalized_column_mask(p, fit_intercept)
        center = scale = ridge = None
        if standardize:
            center, scale, ridge = folds.moments(fold, pen_mask, fit_intercept)
        standardizations.append(solver_standardization(center, scale, fit_intercept=fit_intercept))
        try:
            fold_alpha_max = folds.alpha_max(
                fold,
                family=alpha_family,
                link=alpha_link,
                l1_ratio=effective_l1_ratio,
                pen_mask=pen_mask,
                center=center,
                scale=scale,
                ridge_xtx_diag=ridge,
                trace=trace,
            )
        except (ValidationError, FittingError, ValueError, RuntimeError) as exc:
            # A degenerate fold (all-zero counts, separation, a singular null fit)
            # must not abort the CV: floor its contribution to the grid.
            if verbose:
                print(f"  fold alpha_max computation failed ({exc!r}); flooring this fold.")
            fold_alpha_max = ALPHA_MAX_FLOOR
            trace.append(SolverStep("alpha_max", "alpha_max", 0, False, "failed"))
        fold_alpha_maxes.append(fold_alpha_max)

    alphas = list(generate_alpha_path(max(fold_alpha_maxes), n_alphas, alpha_min_ratio))
    # alpha -> per-fold validation deviances; alpha == 0.0 is the unregularized fit.
    candidate_alphas = list(alphas)
    if include_unregularized:
        candidate_alphas.append(0.0)
    deviances: dict[float, list[float]] = {a: [] for a in candidate_alphas}
    failed_alphas: set[float] = set()

    for fold in range(n_folds):
        active_alphas = [alpha for alpha in candidate_alphas if alpha not in failed_alphas]
        if not active_alphas:
            folds.release(fold)
            continue
        nonneg, nonpos = _get_constraint_indices(folds.names(fold))
        center, scale = standardizations[fold]
        try:
            fold_result = folds.path(
                fold,
                active_alphas,
                l1_ratio=effective_l1_ratio,
                max_iter=max_iter,
                tol=tol,
                nonneg=nonneg,
                nonpos=nonpos,
                center=center,
                scale=scale,
            )
        except ValueError:
            failed_alphas.update(active_alphas)
            trace.extend(SolverStep("cv", "fold_alpha", 0, False, "error") for _ in active_alphas)
            continue
        finally:
            folds.release(fold)
        trace.extend(fold_result.get("solver_trace", ()))
        for alpha, dev in zip(active_alphas, fold_result["fold_deviances"], strict=True):
            if np.isfinite(dev):
                deviances[alpha].append(dev)
            else:
                failed_alphas.add(alpha)

    # ``n_nonzero`` is the full-data design's column count (minus the intercept): a
    # fold's design can be narrower when a level is absent from its training rows,
    # and the refit at the selected alpha always has the full-data width.
    n_nonzero = n_params - 1
    path_results = []
    for alpha in candidate_alphas:
        if alpha in failed_alphas:
            continue
        fold_devs = deviances[alpha]
        if len(fold_devs) != n_folds:
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
        n_folds=n_folds,
        cv_max_iter=max_iter,
        cv_tol=tol,
        fold_safe_target_encoding=True,
        cv_fold_scores={float(a): list(map(float, ds)) for a, ds in deviances.items()},
        cv_scoring_objective="weighted_mean_unit_deviance",
        solver_trace=solver_steps(trace),
    )


class _InMemoryTargetFolds:
    """K fold designs over an in-memory frame: fold k's X (n × p_k) is encoded for
    every row when the fold is used and freed when it is released."""

    def __init__(
        self,
        data,
        designs: list,
        *,
        y: np.ndarray,
        offset: np.ndarray | None,
        weights: np.ndarray | None,
        family: str,
        link: str | None,
        var_power: float,
        theta: float,
        fit_intercept: bool,
        allow_extended_tweedie: bool,
        cv_seed: int,
    ) -> None:
        self.data = data
        self.designs = designs
        self.y = np.ascontiguousarray(y, dtype=np.float64)
        self.offset = None if offset is None else np.ascontiguousarray(offset, dtype=np.float64)
        self.weights = None if weights is None else np.ascontiguousarray(weights, dtype=np.float64)
        self.family = family
        self.link = link
        self.var_power = var_power
        self.theta = theta
        self.fit_intercept = fit_intercept
        self.allow_extended_tweedie = allow_extended_tweedie
        self.cv_seed = cv_seed
        self._fold: int | None = None
        self._x: np.ndarray | None = None

    def _design(self, fold: int) -> np.ndarray:
        if self._fold != fold:
            from rustystats.frozen_design import E_LEARN

            self._x = None
            design = self.designs[fold]
            n = self.data.height
            x = np.empty((n, design.p), dtype=np.float64)
            for start in range(0, n, E_LEARN):
                rows = self.data.slice(start, E_LEARN)
                design.encode_into(rows, x[start : start + rows.height], first_row=start)
            self._x, self._fold = x, fold
        assert self._x is not None
        return self._x

    def _fold_spec(self, fold: int) -> tuple[int, int, int]:
        return (fold, len(self.designs), self.cv_seed)

    def names(self, fold: int) -> list[str]:
        return list(self.designs[fold].names)

    def moments(self, fold, pen_mask, fit_intercept):
        from rustystats._rustystats import compute_standardization_py

        x = self._design(fold)
        weights = self.weights if self.weights is not None else np.ones(x.shape[0])
        center, scale, ridge = compute_standardization_py(
            x,
            weights,
            np.asarray(pen_mask, dtype=bool),
            fit_intercept,
            fold=self._fold_spec(fold),
        )
        return np.asarray(center), np.asarray(scale), np.asarray(ridge)

    def alpha_max(
        self, fold, *, family, link, l1_ratio, pen_mask, center, scale, ridge_xtx_diag, trace
    ):
        from rustystats._rustystats import fold_alpha_max_py

        fold_index, n_folds, seed = self._fold_spec(fold)
        return float(
            fold_alpha_max_py(
                self.y,
                self._design(fold),
                family,
                link,
                float(l1_ratio),
                fold_index,
                n_folds,
                seed,
                offset=self.offset,
                weights=self.weights,
                var_power=float(self.var_power),
                theta=float(self.theta),
                pen_mask=[bool(v) for v in pen_mask],
                center=center,
                scale=scale,
                ridge_xtx_diag=None
                if ridge_xtx_diag is None
                else [float(v) for v in ridge_xtx_diag],
                allow_extended_tweedie=self.allow_extended_tweedie,
                trace=trace,
            )
        )

    def path(self, fold, alphas, *, l1_ratio, max_iter, tol, nonneg, nonpos, center, scale):
        from rustystats._rustystats import fold_path_py

        fold_index, n_folds, seed = self._fold_spec(fold)
        return fold_path_py(
            self.y,
            self._design(fold),
            self.family,
            fold_index,
            n_folds,
            seed,
            link=self.link,
            var_power=self.var_power,
            theta=self.theta,
            offset=self.offset,
            weights=self.weights,
            alphas=list(alphas),
            l1_ratio=l1_ratio,
            max_iter=max_iter,
            tol=tol,
            nonneg_indices=nonneg or None,
            nonpos_indices=nonpos or None,
            allow_extended_tweedie=self.allow_extended_tweedie,
            fit_intercept=self.fit_intercept,
            center=center,
            scale=scale,
            max_row_nonzeros=self.designs[fold].max_row_nonzeros,
        )

    def release(self, fold: int) -> None:
        if self._fold == fold:
            self._x, self._fold = None, None
        self.designs[fold].close()


def fold_formulas(glm_instance, n_folds: int) -> list[ParsedFormula]:
    """A fresh parse of the model's terms per fold: no knots or other state learned
    from the full data reach a fold's design."""
    from rustystats.formula import dict_to_parsed_formula

    return [
        dict_to_parsed_formula(
            response=glm_instance.response,
            terms=glm_instance.terms,
            interactions=glm_instance.interactions_spec,
            intercept=glm_instance.intercept,
        )
        for _ in range(n_folds)
    ]


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
    """Fold-safe CV regularization path for an in-memory target-encoded model (§7.9).

    Rows are assigned to folds by DEC-2's hash of the row and the CV seed. Each
    fold's design is learned from its training rows (target encoding in the model's
    keyed order, over those rows only), and every row is encoded through it: the
    fold fits its training rows and scores its validation rows, which take their
    target encoding from the fold's full statistics. An out-of-core fit runs the same
    steps over batches (``out_of_core_fit``) and gives the same bits.
    """
    from rustystats.frozen_design import (
        E_LEARN,
        FoldRows,
        FrozenDesignUnsupported,
        in_memory_target_inputs,
        learn_frozen_designs,
    )

    _effective_l1_ratio(regularization, l1_ratio)
    data = glm_instance.data
    builder = glm_instance._builder
    cv_seed = seed if seed is not None else DEFAULT_CV_SEED
    n = data.height
    slices = [data.slice(start, E_LEARN) for start in range(0, n, E_LEARN)]
    try:
        designs = learn_frozen_designs(
            fold_formulas(glm_instance, cv),
            slices,
            [FoldRows(fold, cv, cv_seed) for fold in range(cv)],
            target=in_memory_target_inputs(
                builder._parsed_formula.response,
                getattr(glm_instance, "_raw_exposure", None),
                builder._te_seed,
            ),
        )
    except FrozenDesignUnsupported as exc:
        raise ValidationError(
            f"Cross-validation with target encoding cannot learn this model's fold designs: {exc}."
        ) from exc
    theta = glm_instance.theta if glm_instance.theta is not None else DEFAULT_NEGBINOMIAL_THETA
    folds = _InMemoryTargetFolds(
        data,
        designs,
        y=glm_instance.y,
        offset=glm_instance.offset,
        weights=glm_instance.weights,
        family=glm_instance.family,
        link=glm_instance.link,
        var_power=glm_instance.var_power,
        theta=theta,
        fit_intercept=bool(getattr(glm_instance, "intercept", True)),
        allow_extended_tweedie=bool(getattr(glm_instance, "allow_extended_tweedie", False)),
        cv_seed=cv_seed,
    )
    try:
        return target_encoded_cv_path(
            folds,
            n_folds=cv,
            n_params=glm_instance.X.shape[1],
            family=glm_instance.family,
            link=glm_instance.link,
            regularization=regularization,
            l1_ratio=l1_ratio,
            n_alphas=n_alphas,
            alpha_min_ratio=alpha_min_ratio,
            max_iter=max_iter,
            tol=tol,
            include_unregularized=include_unregularized,
            standardize=standardize,
            fit_intercept=bool(getattr(glm_instance, "intercept", True)),
            selection=selection,
            verbose=verbose,
        )
    finally:
        for design in designs:
            design.close()
