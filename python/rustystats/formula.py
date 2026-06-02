"""
Dict-based API for RustyStats GLM.

This module provides the dict-based API for fitting GLMs with DataFrames.

Example
-------
>>> import rustystats as rs
>>> import polars as pl
>>>
>>> data = pl.read_parquet("insurance_data.parquet")
>>> result = rs.glm_dict(
...     response="ClaimNb",
...     terms={
...         "VehPower": {"type": "linear"},
...         "VehAge": {"type": "linear"},
...         "VehBrand": {"type": "categorical"},
...     },
...     data=data,
...     family="poisson",
...     offset="Exposure",
... ).fit()
>>> print(rs.summary(result))
"""

from __future__ import annotations

import copy
import warnings
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from rustystats.constants import (
    DEFAULT_ALPHA_MIN_RATIO,
    DEFAULT_LAMBDA_MAX,
    DEFAULT_LAMBDA_MIN,
    DEFAULT_LINKS,
    DEFAULT_MAX_ITER,
    DEFAULT_N_ALPHAS,
    DEFAULT_N_LAMBDA,
    DEFAULT_NEGBINOMIAL_THETA,
    DEFAULT_SPLINE_DF,
    DEFAULT_TOLERANCE,
    NEGBINOMIAL_ALIASES,
)
from rustystats.exceptions import (
    PredictionError,
    ValidationError,
)
from rustystats.input_transforms import (
    CompiledInputTransform,
    apply_input_transforms,
    compile_input_transforms,
    validate_input_transforms,
)


def is_negbinomial_family(family: str) -> bool:
    """Check if the family string refers to a Negative Binomial distribution."""
    return family.lower().split("(", 1)[0].strip() in NEGBINOMIAL_ALIASES


def is_tweedie_family(family: str) -> bool:
    """Check if the family string refers to a Tweedie distribution."""
    return family.lower().split("(", 1)[0].strip() == "tweedie"


def _split_embedded_family_param(family: str) -> tuple[str, str | None]:
    """Return ``(base_family, parameter_text)`` for strings like ``tweedie(p=1.5)``."""
    raw = family.strip()
    if "(" not in raw:
        return raw, None
    if not raw.endswith(")"):
        raise ValidationError(f"Malformed family parameter string {family!r}.")
    base, params = raw.split("(", 1)
    return base.strip(), params[:-1].strip()


def _parse_embedded_numeric_param(family: str, expected_key: str) -> tuple[str, float | None]:
    """Parse a single embedded numeric family parameter.

    Examples
    --------
    ``tweedie(p=1.7)`` -> ``("tweedie", 1.7)``.
    """
    base, params = _split_embedded_family_param(family)
    if params is None:
        return base, None
    if "=" not in params:
        raise ValidationError(
            f"Malformed family parameter string {family!r}; expected {expected_key}=<number>."
        )
    key, value = [part.strip() for part in params.split("=", 1)]
    if key != expected_key:
        raise ValidationError(
            f"Family {family!r} uses unsupported parameter {key!r}; expected {expected_key!r}."
        )
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValidationError(
            f"Family {family!r} has non-numeric {expected_key}={value!r}."
        ) from exc
    if not np.isfinite(parsed):
        raise ValidationError(f"Family {family!r} has non-finite {expected_key}.")
    return base, parsed


def _format_result_family(family: str, var_power: float, theta: float) -> str:
    """Format family metadata with the fitted parameters embedded."""
    if is_negbinomial_family(family):
        return f"NegativeBinomial(theta={theta:.4f})"
    if is_tweedie_family(family):
        return f"Tweedie(p={var_power:.4f})"
    return family


def get_default_link(family: str) -> str:
    """
    Get the canonical default link function for a GLM family.

    Parameters
    ----------
    family : str
        Family name (e.g., "gaussian", "poisson", "binomial")

    Returns
    -------
    str
        Default link function name (e.g., "identity", "log", "logit")

    Raises
    ------
    ValueError
        If family is not recognized.
    """
    family_lower = family.lower()
    family_base = family_lower.split("(", 1)[0].strip()
    # Handle embedded-parameter result strings such as NegativeBinomial(theta=...)
    if family_base in NEGBINOMIAL_ALIASES:
        return "log"
    link = DEFAULT_LINKS.get(family_base)
    if link is None:
        raise ValidationError(
            f"Unknown family '{family}'. Supported families: {sorted(DEFAULT_LINKS.keys())}"
        )
    return link


# Default row-chunk size for predict(). Caps transient memory of the
# design-matrix builder so a 1M-row × 100-param prediction allocates ~200k×100×8B
# (~160 MB) per chunk instead of materializing the full ~800 MB X.
# Only kicks in when n_rows exceeds the threshold; smaller inputs keep the
# fast single-shot path. The 200_000 sweet spot was picked from the per-chunk
# RSS-vs-throughput sweep in `benchmarks/bench_diagnostics_memory.py` — not arbitrary.
#
# For wide models (many features — e.g. high-cardinality categoricals), the
# row-count cap alone is insufficient: a 200k × 10k float64 chunk is ~16 GB.
# `_compute_predict_chunk_size` combines the row cap with a per-chunk byte
# budget so the design-matrix allocation stays bounded regardless of width.
_PREDICT_ROW_CHUNK_DEFAULT = 200_000
_PREDICT_CHUNK_BYTES_BUDGET = 200_000_000  # ~200 MB per-chunk design matrix cap


def _compute_predict_chunk_size(n_features: int) -> int:
    """Rows per chunk in predict(), adaptive to model width.

    Caps the per-chunk design matrix at ~_PREDICT_CHUNK_BYTES_BUDGET bytes.
    For narrow models (p <= ~125) the default 200k row cap dominates; for
    wider models we shrink so (chunk_size * p * 8) <= budget. Always
    returns at least 1000 rows so very wide models still make progress.
    """
    if n_features <= 0:
        return _PREDICT_ROW_CHUNK_DEFAULT
    budget_rows = _PREDICT_CHUNK_BYTES_BUDGET // (n_features * 8)
    return max(1000, min(_PREDICT_ROW_CHUNK_DEFAULT, budget_rows))


def apply_link(mu: np.ndarray, link: str) -> np.ndarray:
    """
    Apply forward link function to transform response-scale values to linear predictor scale.

    Parameters
    ----------
    mu : np.ndarray
        Values on response scale (means)
    link : str
        Link function name ("identity", "log", "logit", "inverse")

    Returns
    -------
    np.ndarray
        Values on linear predictor scale (eta)
    """
    if link == "identity":
        return mu
    elif link in (None, "log"):
        return np.log(mu)
    elif link == "logit":
        return np.log(mu / (1.0 - mu))
    elif link == "inverse":
        return 1.0 / mu
    else:
        raise ValidationError(
            f"Unknown link function '{link}'. "
            f"Supported links: 'identity', 'log', 'logit', 'inverse'."
        )


def apply_inverse_link(eta: np.ndarray, link: str) -> np.ndarray:
    """
    Apply inverse link function to linear predictor.

    Parameters
    ----------
    eta : np.ndarray
        Linear predictor values
    link : str
        Link function name ("identity", "log", "logit", "inverse")

    Returns
    -------
    np.ndarray
        Predicted means (mu)

    Raises
    ------
    ValidationError
        If link function is not recognized.
    """
    if link == "identity":
        return eta
    elif link in (None, "log"):
        # Mirror LogLink::inverse in crates/rustystats-core/src/links/log.rs:
        # clamp to [-700, 700] before exp() so extreme eta saturates at
        # ~1e-304 / 1e304 instead of underflowing/overflowing to NaN.
        return np.exp(np.clip(eta, -700.0, 700.0))
    elif link == "logit":
        # Mirror LogitLink::inverse in crates/rustystats-core/src/links/logit.rs:
        # branch on sign so the exponent passed to exp() is always <= 0.
        # For eta >= 0:  1 / (1 + exp(-eta))
        # For eta <  0:  exp(eta) / (1 + exp(eta))
        eta_arr = np.asarray(eta, dtype=np.float64)
        out = np.empty_like(eta_arr)
        pos = eta_arr >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-eta_arr[pos]))
        exp_neg = np.exp(eta_arr[~pos])
        out[~pos] = exp_neg / (1.0 + exp_neg)
        return out
    elif link == "inverse":
        return 1.0 / eta
    else:
        raise ValidationError(
            f"Unknown link function '{link}'. "
            f"Supported links: 'identity', 'log', 'logit', 'inverse'."
        )


# Lazy imports for optional dependencies
if TYPE_CHECKING:
    import polars as pl

    from rustystats.diagnostics.types import DataExploration, ModelDiagnostics
    from rustystats.regularization_path import RegularizationPathInfo


def _get_column(data: pl.DataFrame, column: str) -> np.ndarray:
    """Extract a column as numpy array from Polars DataFrame."""
    return data[column].to_numpy()


def _extract_needed_columns(
    terms: dict[str, dict[str, Any]],
    response: str | None = None,
    interactions: list[dict[str, Any]] | None = None,
    input_transforms: list[dict[str, Any]] | None = None,
    offset: str | np.ndarray | None = None,
    weights: str | np.ndarray | None = None,
    exposure: str | np.ndarray | None = None,
    complement: Any = None,
    _seen_models: set[int] | None = None,
) -> set[str]:
    """Extract all DataFrame column names needed to build this model.

    Parameters
    ----------
    terms : dict
        Term specifications (same format as glm_dict).
    response : str, optional
        Response column name. Omit for prediction (no response needed).
    interactions, offset, weights, complement
        Same as glm_dict parameters.
    """
    import re

    cols: set[str] = set()
    transform_specs = input_transforms if input_transforms is not None else []
    transform_outputs = {
        spec.get("output")
        for spec in transform_specs
        if isinstance(spec, dict) and isinstance(spec.get("output"), str)
    }
    produced: set[str] = set()
    for spec in transform_specs:
        if not isinstance(spec, dict):
            continue
        for source in spec.get("sources", []):
            if isinstance(source, str) and source not in produced:
                cols.add(source)
        output = spec.get("output")
        if isinstance(output, str):
            produced.add(output)
    if response is not None:
        cols.add(response)

    for var_name, spec in terms.items():
        term_type = spec.get("type", "linear")
        if term_type == "expression":
            expr = spec["expr"]
            for token in re.findall(r"\b([A-Za-z_]\w*)\b", expr):
                if token not in transform_outputs:
                    cols.add(token)
        else:
            if var_name not in transform_outputs:
                cols.add(var_name)

    if interactions:
        for ix in interactions:
            for key in ix:
                if key in (
                    "include_main",
                    "target_encoding",
                    "frequency_encoding",
                    "prior_weight",
                    "n_permutations",
                ):
                    continue
                if key not in transform_outputs:
                    cols.add(key)

    if isinstance(offset, str) and offset not in transform_outputs:
        cols.add(offset)
    if isinstance(exposure, str) and exposure not in transform_outputs:
        cols.add(exposure)
    if isinstance(weights, str) and weights not in transform_outputs:
        cols.add(weights)
    if isinstance(complement, str) and complement not in transform_outputs:
        cols.add(complement)
    else:
        cols.update(_extract_model_needed_columns(complement, _seen_models))

    return cols


def _extract_model_needed_columns(model: Any, seen_models: set[int] | None = None) -> set[str]:
    """Return prediction columns needed by a GLMModel-like complement."""
    if model is None:
        return set()

    terms = getattr(model, "_terms_dict", None)
    if terms is None:
        return set()

    model_id = id(model)
    seen = set() if seen_models is None else set(seen_models)
    if model_id in seen:
        return set()
    seen.add(model_id)

    return _extract_needed_columns(
        terms=terms,
        interactions=getattr(model, "_interactions_spec", None),
        input_transforms=getattr(model, "_input_transforms", None),
        offset=getattr(model, "_offset_spec", None),
        exposure=getattr(model, "_exposure_spec", None),
        complement=getattr(model, "_complement_spec", None),
        _seen_models=seen,
    )


def _collect_lazyframe(
    data: pl.DataFrame | pl.LazyFrame,
    needed_columns: set[str],
) -> pl.DataFrame:
    """If data is a LazyFrame, select only needed columns and collect. Otherwise return as-is."""
    import polars as pl

    if not isinstance(data, pl.LazyFrame):
        return data

    if needed_columns:
        return data.select(sorted(needed_columns)).collect()

    return data.collect()


# Import from interactions module (the canonical implementation)
from rustystats.interactions import InteractionBuilder


def _get_constraint_indices(feature_names: list[str]) -> tuple:
    """
    Compute coefficient constraint indices from feature names.

    For smooth (penalized) monotonic bs() terms, the solver handles monotonicity
    internally via exp reparameterization on B-spline coefficients (Pya & Wood, 2015).
    The bs() sign constraints here serve as a fallback for fixed-df monotonic terms
    that go through the IRLS path (which doesn't yet have exp reparameterization).

    Returns
    -------
    nonneg_indices : list[int]
        Indices of coefficients that must be non-negative (β ≥ 0)
    nonpos_indices : list[int]
        Indices of coefficients that must be non-positive (β ≤ 0)
    """
    # ms()/ns()/bs() with + and pos() terms require non-negative coefficients.
    # Smooth (penalized) terms with ", k," in the name use exp reparameterization
    # for monotonicity and must NOT have sign clamping (which would corrupt beta[0]).
    nonneg_indices = [
        i
        for i, name in enumerate(feature_names)
        if name.startswith("pos(")
        or (name.startswith("ms(") and ", +)" in name and ", k," not in name)
        or (name.startswith("ns(") and ", +)" in name)
        or (name.startswith("bs(") and ", +)" in name and ", k," not in name)
    ]
    # ms()/ns()/bs() with - and neg() terms require non-positive coefficients.
    nonpos_indices = [
        i
        for i, name in enumerate(feature_names)
        if name.startswith("neg(")
        or (name.startswith("ms(") and ", -)" in name and ", k," not in name)
        or (name.startswith("ns(") and ", -)" in name)
        or (name.startswith("bs(") and ", -)" in name and ", k," not in name)
    ]
    return nonneg_indices, nonpos_indices


@dataclass
class SmoothTermResult:
    """Result for a single smooth term after fitting."""

    variable: str
    k: int
    edf: float
    lambda_: float
    gcv: float
    col_start: int
    col_end: int


def _fit_with_fixed_spline_penalties(
    y: np.ndarray,
    X: np.ndarray,
    spline_terms: list[Any],
    spline_col_indices: list[tuple],
    family: str,
    link: str,
    var_power: float,
    theta: float,
    offset: np.ndarray | None,
    weights: np.ndarray | None,
    alpha: float,
    max_iter: int = DEFAULT_MAX_ITER,
    tol: float = DEFAULT_TOLERANCE,
    store_design_matrix: bool = False,
    allow_extended_tweedie: bool = False,
) -> tuple:
    """
    Fit GLM with fixed-df splines using D'D penalty scaled by alpha.

    Instead of scalar ridge (α·I), this applies α·D'D difference penalties
    to spline basis columns. This gives better convergence because the
    penalty structure matches the spline basis.

    Uses the same Rust smooth solver as GCV, but with fixed lambdas.
    """
    from rustystats._rustystats import fit_smooth_glm_unified_py as _fit_smooth_unified

    penalties = []
    monotonicity_specs = []
    for i, term in enumerate(spline_terms):
        start, end = spline_col_indices[i]
        k = end - start
        penalties.append(term.compute_penalty_matrix(k)[:k, :k])
        mono = getattr(term, "_smooth_monotonicity", None) or getattr(term, "monotonicity", None)
        monotonicity_specs.append(mono)

    # Fixed lambda = alpha for all terms (no GCV search)
    # Pass lambda_min = lambda_max = alpha so optimizer returns alpha immediately
    rust_result, smooth_meta = _fit_smooth_unified(
        y,
        X,
        spline_col_indices,
        penalties,
        family,
        link,
        offset,
        weights,
        max_iter,
        tol,
        alpha,
        alpha,  # lambda_min = lambda_max = alpha → fixed lambda
        monotonicity_specs if any(m is not None for m in monotonicity_specs) else None,
        store_design_matrix,
        var_power=var_power,
        theta=theta,
        allow_extended_tweedie=allow_extended_tweedie,
    )

    smooth_results = []
    for i, term in enumerate(spline_terms):
        start, end = spline_col_indices[i]
        smooth_results.append(
            SmoothTermResult(
                variable=term.var_name,
                k=term.df,
                edf=smooth_meta["smooth_edfs"][i],
                lambda_=smooth_meta["lambdas"][i],
                gcv=smooth_meta["gcv"],
                col_start=start,
                col_end=end,
            )
        )
        term._lambda = smooth_meta["lambdas"][i]
        term._edf = smooth_meta["smooth_edfs"][i]

    return rust_result, smooth_results, smooth_meta["total_edf"], smooth_meta["gcv"]


def _fit_with_smooth_penalties(
    y: np.ndarray,
    X: np.ndarray,
    smooth_terms: list[Any],
    smooth_col_indices: list[tuple],
    family: str,
    link: str,
    var_power: float,
    theta: float,
    offset: np.ndarray | None,
    weights: np.ndarray | None,
    max_iter: int = DEFAULT_MAX_ITER,
    tol: float = DEFAULT_TOLERANCE,
    n_lambda: int = DEFAULT_N_LAMBDA,
    lambda_min: float = DEFAULT_LAMBDA_MIN,
    lambda_max: float = DEFAULT_LAMBDA_MAX,
    store_design_matrix: bool = False,
    nonneg_indices: list[int] | None = None,
    nonpos_indices: list[int] | None = None,
    allow_extended_tweedie: bool = False,
) -> tuple:
    """
    Fit GLM with penalized smooth terms using fast GCV optimization.

    Uses a unified Rust entry point that takes the full design matrix and
    smooth term specs (column ranges + penalties + monotonicity). No column
    splitting or coefficient reordering needed.

    Parameters
    ----------
    y : array
        Response variable
    X : array
        Full design matrix
    smooth_terms : list
        List of SplineTerm objects marked as smooth
    smooth_col_indices : list
        List of (start, end) column indices for each smooth term
    family, link, var_power, theta : model parameters
    offset, weights : optional arrays
    max_iter, tol : IRLS parameters
    n_lambda, lambda_min, lambda_max : GCV grid search parameters

    Returns
    -------
    result : GLMResult from Rust
    smooth_results : list of SmoothTermResult
    total_edf : float
    gcv : float
    """
    _n, _p = X.shape
    len(smooth_terms)

    from rustystats._rustystats import fit_smooth_glm_unified_py as _fit_smooth_unified

    # Build penalty matrices and monotonicity specs for each smooth term
    penalties = []
    monotonicity_specs = []
    for i, term in enumerate(smooth_terms):
        start, end = smooth_col_indices[i]
        k = end - start
        penalties.append(term.compute_penalty_matrix(k)[:k, :k])

        mono = getattr(term, "_smooth_monotonicity", None) or getattr(term, "monotonicity", None)
        monotonicity_specs.append(mono)

    # Call unified Rust solver — full design matrix, no splitting needed
    has_monotonic = any(m is not None for m in monotonicity_specs)
    rust_result, smooth_meta = _fit_smooth_unified(
        y,
        X,
        smooth_col_indices,
        penalties,
        family,
        link,
        offset,
        weights,
        max_iter,
        tol,
        lambda_min,
        lambda_max,
        monotonicity_specs if has_monotonic else None,
        store_design_matrix,
        nonneg_indices if nonneg_indices else None,
        nonpos_indices if nonpos_indices else None,
        var_power=var_power,
        theta=theta,
        allow_extended_tweedie=allow_extended_tweedie,
    )

    # Build smooth term results — coefficients are already in original column order
    smooth_results = []
    for i, term in enumerate(smooth_terms):
        start, end = smooth_col_indices[i]
        smooth_results.append(
            SmoothTermResult(
                variable=term.var_name,
                k=term.df,
                edf=smooth_meta["smooth_edfs"][i],
                lambda_=smooth_meta["lambdas"][i],
                gcv=smooth_meta["gcv"],
                col_start=start,
                col_end=end,
            )
        )
        term._lambda = smooth_meta["lambdas"][i]
        term._edf = smooth_meta["smooth_edfs"][i]

    return rust_result, smooth_results, smooth_meta["total_edf"], smooth_meta["gcv"]


def _fit_glm_core(
    y: np.ndarray,
    X: np.ndarray,
    family: str,
    link: str,
    var_power: float,
    theta: float,
    offset: np.ndarray | None,
    weights: np.ndarray | None,
    alpha: float,
    l1_ratio: float,
    max_iter: int,
    tol: float,
    feature_names: list[str],
    builder: InteractionBuilder,
    fit_intercept: bool = True,
    store_design_matrix: bool = False,
    allow_extended_tweedie: bool = False,
    standardize: bool = True,
) -> tuple:
    """
    Core GLM fitting logic for FormulaGLMDict.

    Handles smooth term fitting with GCV-based lambda selection and
    standard fitting with coefficient constraints.

    Returns
    -------
    result : GLMResult
        Fitted model result from Rust
    smooth_results : list or None
        Smooth term results if applicable
    total_edf : float or None
        Total effective degrees of freedom
    gcv : float or None
        GCV score for smooth models
    """
    from rustystats._rustystats import fit_glm_py as _fit_glm_rust
    from rustystats.validation import validate_glm_inputs

    # Validate inputs before fitting - catches NaN, Inf, invalid response values, etc.
    # Note: is_exposure_offset=False because offset is already log-transformed by _process_offset
    # (raw exposure validation happens there before log-transform).
    # RS-ACT-006: thread var_power + allow_extended_tweedie so the Tweedie
    # regime table is enforced before any deviance is evaluated.
    y, X, weights, offset = validate_glm_inputs(
        y,
        X,
        family,
        weights,
        offset,
        feature_names,
        is_exposure_offset=False,
        var_power=var_power,
        allow_extended_tweedie=allow_extended_tweedie,
    )

    # Check for smooth terms (s() terms with automatic lambda selection)
    smooth_terms, smooth_col_indices = builder.get_smooth_terms()

    # Compute sign constraints from feature names (pos()/neg() and monotonic splines)
    nonneg_indices, nonpos_indices = _get_constraint_indices(feature_names)

    if smooth_terms and alpha == 0.0:
        # Use penalized fitting with GCV-based lambda selection
        result, smooth_results, total_edf, gcv = _fit_with_smooth_penalties(
            y,
            X,
            smooth_terms,
            smooth_col_indices,
            family,
            link,
            var_power,
            theta,
            offset,
            weights,
            max_iter,
            tol,
            store_design_matrix=store_design_matrix,
            nonneg_indices=nonneg_indices if nonneg_indices else None,
            nonpos_indices=nonpos_indices if nonpos_indices else None,
            allow_extended_tweedie=allow_extended_tweedie,
        )
        return result, smooth_results, total_edf, gcv

    center = scale = None
    if standardize and alpha > 0.0:
        from rustystats.regularization_path import compute_standardization

        # pen_mask defaults to "all columns except the intercept"; this path
        # does not build an alpha grid, so it need not share an explicit mask.
        center, scale = compute_standardization(X, weights, fit_intercept=fit_intercept)

    result = _fit_glm_rust(
        y,
        X,
        family,
        link,
        var_power,
        theta,
        offset,
        weights,
        alpha,
        l1_ratio,
        max_iter,
        tol,
        nonneg_indices if nonneg_indices else None,
        nonpos_indices if nonpos_indices else None,
        store_design_matrix,
        allow_extended_tweedie,
        fit_intercept,
        center,
        scale,
    )
    return result, None, None, None


def _estimate_negbinomial(
    y: np.ndarray,
    X: np.ndarray,
    link: str | None,
    offset: np.ndarray | None,
    weights: np.ndarray | None,
    feature_names: list[str],
    max_iter: int = DEFAULT_MAX_ITER,
    tol: float = DEFAULT_TOLERANCE,
    init_theta: float | None = None,
    store_design_matrix: bool = False,
) -> tuple[Any, dict]:
    """Estimate Negative Binomial theta by profile likelihood (RS-ACT-010).

    Wires the offset/weights-aware Rust estimator ``fit_negbinomial_py`` and
    returns ``(rust_result, theta_metadata)``. Only valid for the plain GLM path
    (no smooth terms, no regularization, no sign constraints); callers enforce
    that before invoking. The returned result's family string carries the
    estimated theta; the metadata records the estimation provenance.
    """
    from rustystats._rustystats import fit_negbinomial_py as _fit_nb_rust
    from rustystats.validation import validate_glm_inputs

    y, X, weights, offset = validate_glm_inputs(
        y, X, "negbinomial", weights, offset, feature_names, is_exposure_offset=False
    )
    result, theta_meta = _fit_nb_rust(
        y,
        X,
        link or "log",
        init_theta,  # init_theta (None -> moment estimate)
        1e-5,  # theta_tol
        25,  # max_theta_iter
        offset,
        weights,
        max_iter,
        tol,
        0.0,  # alpha (plain path only)
        0.0,  # l1_ratio
        None,  # nonneg_indices
        None,  # nonpos_indices
        store_design_matrix,
    )
    return result, dict(theta_meta)


def _build_results(
    result: Any,
    feature_names: list[str],
    formula: str,
    family: str,
    link: str | None,
    builder: InteractionBuilder,
    offset_spec: str | np.ndarray | None,
    exposure_spec: str | np.ndarray | None,
    path_info: RegularizationPathInfo | None,
    smooth_results: list[SmoothTermResult] | None,
    total_edf: float | None,
    gcv: float | None,
    terms_dict: dict[str, dict[str, Any]] | None = None,
    interactions_spec: list[dict[str, Any]] | None = None,
    input_transforms: list[dict[str, Any]] | None = None,
    compiled_input_transforms: list[CompiledInputTransform] | None = None,
    complement_spec: str | GLMModel | None = None,
    complement_values: np.ndarray | None = None,
    array_exposure_requires_prediction_override: bool = False,
    regularization_standardized: bool = False,
) -> GLMModel:
    """Build GLMModel with all metadata."""
    # Clear builder caches to free memory (keep TE stats for prediction)
    if builder is not None:
        builder.clear_caches()

    return GLMModel(
        result=result,
        feature_names=feature_names,
        formula=formula,
        family=family,
        link=link,
        builder=builder,
        offset_spec=offset_spec,
        exposure_spec=exposure_spec,
        regularization_path_info=path_info,
        smooth_results=smooth_results,
        total_edf=total_edf,
        gcv=gcv,
        terms_dict=terms_dict,
        interactions_spec=interactions_spec,
        input_transforms=input_transforms,
        compiled_input_transforms=compiled_input_transforms,
        complement_spec=complement_spec,
        complement_values=complement_values,
        array_exposure_requires_prediction_override=array_exposure_requires_prediction_override,
        regularization_standardized=regularization_standardized,
    )


class _GLMBase:
    """
    Base class for FormulaGLMDict.

    Provides data access, offset/weights processing, and CV path handling.
    Subclasses must set: _data_ref, family, link, _offset_spec, _seed.
    """

    @property
    def data(self) -> pl.DataFrame:
        """Access the original DataFrame (may raise if garbage collected)."""
        d = self._data_ref()
        if d is None:
            raise ValidationError(
                "Original DataFrame has been garbage collected. "
                "Keep a reference to the DataFrame if you need to access it after fitting."
            )
        return d

    def _uses_log_link(self) -> bool:
        """Check if model uses log link (explicit or canonical).

        For ``link=None``, falls back to ``get_default_link`` so family aliases
        whose default is "log" (Tweedie's ``tweedie``; the
        ``negativebinomial``/``nb`` aliases that resolve through
        ``NEGBINOMIAL_ALIASES``) are recognized without having to enumerate
        each spelling here.
        """
        if self.link == "log":
            return True
        if self.link is None:
            try:
                return get_default_link(self.family) == "log"
            except ValidationError:
                return False
        return False

    def _process_offset(
        self,
        offset: str | np.ndarray | None,
    ) -> np.ndarray | None:
        """Resolve a link-scale offset (column name or array), used verbatim.

        An ``offset`` is added to the linear predictor as-is. The rate
        denominator that gets log-transformed is passed via ``exposure=``
        instead (RS-ACT-002).
        """
        if offset is None:
            return None
        if isinstance(offset, str):
            return _get_column(self.data, offset).astype(np.float64)
        return np.asarray(offset, dtype=np.float64)

    def _process_weights(
        self,
        weights: str | np.ndarray | None,
    ) -> np.ndarray | None:
        """Process weights specification."""
        if weights is None:
            return None
        if isinstance(weights, str):
            return _get_column(self.data, weights).astype(np.float64)
        else:
            return np.asarray(weights, dtype=np.float64)

    def _process_complement(
        self,
        complement: str | np.ndarray | GLMModel | None,
        raw_exposure: np.ndarray | None,
    ) -> np.ndarray | None:
        """Process complement of credibility and merge into offset.

        The complement represents prior predictions on the response scale
        (rates for log-link models, probabilities for logit-link).
        It is transformed to the link scale and added to the existing offset,
        so that regularization shrinks coefficients toward the complement
        rather than toward zero.

        Parameters
        ----------
        complement : str, array-like, GLMModel, or None
            Complement of credibility. If str, column name in data with values
            on response scale. If GLMModel, predictions are computed on this
            data (divided by exposure if applicable). If array, used directly.
        raw_exposure : np.ndarray or None
            Raw exposure values (needed to convert a GLMModel complement's
            predicted counts back to a rate when the model uses exposure).

        Returns
        -------
        np.ndarray or None
            Complement values on link scale, ready to add to offset.
        """
        if complement is None:
            return None

        # Extract response-scale complement values
        if isinstance(complement, GLMModel):
            comp_values = complement.predict(self.data)
            # If the model has raw exposure, divide by it to recover the rate.
            if raw_exposure is not None:
                comp_values = comp_values / raw_exposure
        elif isinstance(complement, str):
            comp_values = _get_column(self.data, complement)
        else:
            comp_values = np.asarray(complement, dtype=np.float64)

        comp_values = np.asarray(comp_values, dtype=np.float64)

        # Validate
        link = self.link or get_default_link(self.family)
        if link in (None, "log"):
            n_invalid = np.sum(comp_values <= 0)
            if n_invalid > 0:
                raise ValidationError(
                    f"Complement values must be strictly positive for {self.family} family with log link. "
                    f"Found {n_invalid} values <= 0."
                )
        elif link == "logit":
            n_invalid = np.sum((comp_values <= 0) | (comp_values >= 1))
            if n_invalid > 0:
                raise ValidationError(
                    f"Complement values must be in (0, 1) for {self.family} family with logit link. "
                    f"Found {n_invalid} values outside range."
                )

        # Store raw complement for reporting (before link transform)
        self._complement_values = comp_values

        # Transform to link scale
        return apply_link(comp_values, link)

    def _resolve_exposure_values(self, exposure: str | np.ndarray) -> np.ndarray:
        """Resolve and validate a raw exposure spec to a positive float array."""
        if isinstance(exposure, str):
            vals = _get_column(self.data, exposure).astype(np.float64)
        else:
            vals = np.asarray(exposure, dtype=np.float64)
            if vals.ndim != 1:
                raise ValidationError(
                    f"exposure must be a one-dimensional array of length "
                    f"{self.data.height}; got shape {vals.shape}."
                )
            if vals.shape[0] != self.data.height:
                raise ValidationError(
                    f"exposure array length {vals.shape[0]} does not match data "
                    f"length {self.data.height}."
                )
        if not np.all(np.isfinite(vals)):
            raise ValidationError("exposure must be finite.")
        if np.any(vals <= 0):
            raise ValidationError("exposure must be strictly positive.")
        return vals

    def _has_target_encoding(self) -> bool:
        """True if any term or interaction requests target encoding.

        Called twice in the fit lifecycle: once during ``__init__`` (before the
        ``InteractionBuilder`` has been constructed, so ``self._builder`` is
        ``None`` and we must inspect the raw ``self.terms`` / ``self.interactions_spec``
        the caller passed in) and again after the builder has produced its
        ``_parsed_formula`` (where the parsed view is canonical). Both code
        paths must agree; this method keeps them in sync.
        """
        builder = getattr(self, "_builder", None)
        parsed = getattr(builder, "_parsed_formula", None)
        if parsed is not None and parsed.target_encoding_terms:
            return True
        if any(spec.get("type") == "target_encoding" for spec in self.terms.values()):
            return True
        reserved_keys = {
            "include_main",
            "target_encoding",
            "frequency_encoding",
            "prior_weight",
            "n_permutations",
        }
        for interaction in self.interactions_spec or []:
            if interaction.get("target_encoding"):
                return True
            for key, spec in interaction.items():
                if key in reserved_keys:
                    continue
                if isinstance(spec, dict) and spec.get("type") == "target_encoding":
                    return True
        return False

    def _get_raw_exposure(
        self,
        exposure: str | np.ndarray | None,
    ) -> np.ndarray | None:
        """Raw positive exposure for target encoding / rate diagnostics.

        Sourced only from an explicit ``exposure=`` -- never from a link-scale
        ``offset`` (RS-ACT-002).
        """
        if exposure is not None:
            return self._resolve_exposure_values(exposure)
        return None

    def _resolve_cv_path(
        self,
        alpha: float,
        l1_ratio: float,
        max_iter: int,
        tol: float,
        cv: int | None,
        selection: str,
        regularization: str | None,
        n_alphas: int,
        alpha_min_ratio: float,
        cv_seed: int | None,
        include_unregularized: bool,
        verbose: bool,
        standardize: bool,
    ) -> tuple:
        """
        Handle CV-based regularization path if requested.

        Returns (alpha, l1_ratio, path_info) with updated alpha/l1_ratio
        from CV selection, or original values if no CV.
        """
        if regularization is not None and cv is None:
            cv = 5

        if cv is None:
            return alpha, l1_ratio, None

        if regularization is None:
            raise ValidationError(
                "When cv is specified, 'regularization' must be set to 'ridge', 'lasso', or 'elastic_net'"
            )

        # RS-ACT-001b: CV with target encoding is fold-safe. Models with a
        # target-encoded term route through the per-fold fit/transform path
        # (encoding fit on each fold's training rows only); everything else keeps
        # using the fast Rust array path, which slices a single full-data design.
        from rustystats.regularization_path import (
            fit_cv_regularization_path,
            fit_cv_te_regularization_path,
        )

        if regularization == "ridge":
            cv_l1_ratio = 0.0
        elif regularization == "lasso":
            cv_l1_ratio = 1.0
        elif regularization == "elastic_net":
            cv_l1_ratio = l1_ratio if l1_ratio > 0 else 0.5
        else:
            raise ValidationError(f"Unknown regularization type: {regularization}")

        cv_path_fn = (
            fit_cv_te_regularization_path
            if self._has_target_encoding()
            else fit_cv_regularization_path
        )

        path_info = cv_path_fn(
            glm_instance=self,
            cv=cv,
            selection=selection,
            regularization=regularization,
            n_alphas=n_alphas,
            alpha_min_ratio=alpha_min_ratio,
            l1_ratio=cv_l1_ratio,
            max_iter=max_iter,
            tol=tol,
            seed=cv_seed if cv_seed is not None else self._seed,
            include_unregularized=include_unregularized,
            verbose=verbose,
            standardize=standardize,
        )

        if verbose:
            print(f"\nRefitting on full data with alpha={path_info.selected_alpha:.6f}")

        return path_info.selected_alpha, path_info.selected_l1_ratio, path_info


@dataclass
class _DeserializedResult:
    """
    Minimal result object for deserialized models.

    This provides the interface needed by GLMModel for prediction
    without requiring the full Rust GLMResults object.

    Note: fittedvalues and linear_predictor are not stored as they're
    large arrays not needed for prediction on new data.
    """

    params: np.ndarray
    deviance: float
    iterations: int
    converged: bool
    nobs: int
    df_resid: int
    df_model: int
    alpha: float
    l1_ratio: float
    is_regularized: bool
    penalty_type: str

    @property
    def fittedvalues(self) -> np.ndarray:
        raise AttributeError(
            "fittedvalues not available on deserialized models. "
            "Only coefficients are stored for prediction."
        )

    @property
    def linear_predictor(self) -> np.ndarray:
        raise AttributeError(
            "linear_predictor not available on deserialized models. "
            "Only coefficients are stored for prediction."
        )


class _DeserializedBuilder(InteractionBuilder):
    """
    Minimal builder for deserialized models.

    Inherits all transform_new_data / prediction logic from InteractionBuilder.
    Only overrides __init__ to restore state from a serialized dict instead
    of building from a live DataFrame.
    """

    def __init__(self, state: dict):
        # Bypass InteractionBuilder.__init__ — set state directly from dict
        self._parsed_formula = state["parsed_formula"]
        self._cat_encoding_cache = state["cat_encoding_cache"]
        self._fitted_splines = state["fitted_splines"]
        self._te_stats = state["te_stats"]
        self._fe_stats: dict[str, dict] = state.get("fe_stats", {})
        self.dtype = state["dtype"]
        self.data = None
        self._n = 0
        self._term_slots = state["term_slots"]


def _resolve_predict_offset(
    new_data: pl.DataFrame,
    offset_override: str | np.ndarray | None,
    stored_offset_spec: str | np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    """Resolve a prediction-time offset to its raw and link-scale forms.

    Returns ``(raw, link_scale, column_name)`` or all-``None`` when no
    offset applies. After RS-ACT-002b normalization (legacy exposure aliases
    are stored on ``_exposure_spec``, never on ``_offset_spec``), the offset
    column is always already on the link scale; ``raw`` and ``link_scale``
    share the same array.
    """
    offset_to_use = offset_override if offset_override is not None else stored_offset_spec
    if offset_to_use is None:
        return None, None, None
    if isinstance(offset_to_use, str):
        if offset_to_use not in new_data.columns:
            raise PredictionError(
                f"Model was fit with offset='{offset_to_use}', but column "
                f"'{offset_to_use}' is not present in the prediction data. "
                "Pass offset= explicitly to predict()."
            )
        raw = new_data[offset_to_use].to_numpy().astype(np.float64)
        return raw, raw, offset_to_use
    arr = np.asarray(offset_to_use, dtype=np.float64)
    if arr.ndim != 1 or arr.shape[0] != len(new_data):
        raise PredictionError(
            f"offset array length {arr.shape[0]} does not match prediction data "
            f"length {len(new_data)}. Pass an offset= array for the new data."
        )
    return arr, arr, None


def _resolve_predict_exposure(
    new_data: pl.DataFrame,
    exposure_spec: str | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str | None]:
    """Resolve prediction-time raw exposure to raw and log-scale forms.

    Raises ``PredictionError`` when a stored exposure column is absent from the
    prediction data and no explicit exposure array is supplied.
    """
    if isinstance(exposure_spec, str):
        if exposure_spec not in new_data.columns:
            raise PredictionError(
                f"Model was fit with exposure='{exposure_spec}', but column "
                f"'{exposure_spec}' is not present in the prediction data. "
                "Pass exposure= explicitly to predict()."
            )
        raw = new_data[exposure_spec].to_numpy().astype(np.float64)
    else:
        raw = np.asarray(exposure_spec, dtype=np.float64)
    if raw.ndim != 1:
        raise PredictionError(
            f"exposure must be a one-dimensional array of length {len(new_data)}; "
            f"got shape {raw.shape}."
        )
    if raw.shape[0] != len(new_data):
        raise PredictionError(
            f"exposure array length {raw.shape[0]} does not match prediction data "
            f"length {len(new_data)}."
        )
    if not np.all(np.isfinite(raw)) or np.any(raw <= 0):
        raise PredictionError("exposure must be finite and strictly positive.")
    return raw, np.log(raw), exposure_spec if isinstance(exposure_spec, str) else None


def _resolve_predict_exposure_link(
    new_data: pl.DataFrame,
    exposure_spec: str | np.ndarray,
) -> np.ndarray:
    """Resolve a prediction-time raw exposure to its log-scale contribution."""
    _, link, _ = _resolve_predict_exposure(new_data, exposure_spec)
    return link


def _resolve_predict_complement(
    new_data: pl.DataFrame,
    complement_override: Any,
    stored_complement_spec: Any,
    exposure_spec_for_complement: str | np.ndarray | None,
    link: str,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Resolve a prediction-time complement to its response and link scales.

    Returns ``(response_scale, link_scale)`` or both ``None`` when no
    complement applies. When the complement is a fitted ``GLMModel`` and the
    current model carries an exposure spec, the prior's response is divided by
    exposure to recover the rate before being passed through the link.
    """
    comp_to_use = complement_override if complement_override is not None else stored_complement_spec
    if comp_to_use is None:
        return None, None
    if isinstance(comp_to_use, GLMModel):
        comp_response = comp_to_use.predict(new_data)
        if exposure_spec_for_complement is not None:
            exposure, _, _ = _resolve_predict_exposure(new_data, exposure_spec_for_complement)
            comp_response = comp_response / exposure
    elif isinstance(comp_to_use, str):
        comp_response = new_data[comp_to_use].to_numpy().astype(np.float64)
    else:
        comp_response = np.asarray(comp_to_use, dtype=np.float64)
    comp_response = comp_response.astype(np.float64, copy=False)
    return comp_response, apply_link(comp_response, link)


class GLMModel:
    """
    Results from a formula-based GLM fit.

    This wraps the base GLMResults and adds formula-specific functionality
    like named coefficients and automatic summary formatting.

    Attributes
    ----------
    params : np.ndarray
        Fitted coefficients
    feature_names : list[str]
        Names corresponding to each coefficient
    formula : str
        The formula used to fit the model
    """

    # Serialized-state schema version written by ``to_bytes`` and required by
    # ``from_bytes``. Bumped whenever the persisted state shape changes.
    _SCHEMA_VERSION = 4

    def __init__(
        self,
        result,
        feature_names: list[str],
        formula: str,
        family: str,
        link: str | None,
        builder: InteractionBuilder | None = None,
        offset_spec: str | np.ndarray | None = None,
        exposure_spec: str | np.ndarray | None = None,
        regularization_path_info: RegularizationPathInfo | None = None,
        smooth_results: list[SmoothTermResult] | None = None,
        total_edf: float | None = None,
        gcv: float | None = None,
        terms_dict: dict[str, dict[str, Any]] | None = None,
        interactions_spec: list[dict[str, Any]] | None = None,
        input_transforms: list[dict[str, Any]] | None = None,
        compiled_input_transforms: list[CompiledInputTransform] | None = None,
        complement_spec: str | GLMModel | None = None,
        complement_values: np.ndarray | None = None,
        array_exposure_requires_prediction_override: bool = False,
        regularization_standardized: bool = False,
    ):
        self._result = result
        self._is_deserialized = isinstance(result, _DeserializedResult)
        self._smooth_results = smooth_results
        self._total_edf = total_edf
        self._gcv = gcv
        self.feature_names = feature_names
        self.formula = formula
        self.family = family
        # Tweedie var_power is encoded in the (serialized, round-tripped) family
        # string, e.g. "Tweedie(p=2.5000)". Derive it so releveled deviance/llf
        # use the fitted power instead of the 1.5 default — this fixes both
        # in-memory and deserialized models (RS-ACT-006).
        from rustystats.diagnostics.api import _parse_family_params

        self.var_power, _ = _parse_family_params(family)
        self.allow_extended_tweedie = True
        self._regularization_path_info = regularization_path_info
        self.link = link or get_default_link(family)
        self._builder = builder
        self._offset_spec = offset_spec
        self._exposure_spec = exposure_spec
        self._array_exposure_requires_prediction_override = bool(
            array_exposure_requires_prediction_override
        )
        self._terms_dict = terms_dict
        self._interactions_spec = interactions_spec
        self._input_transforms = validate_input_transforms(input_transforms)
        self._compiled_input_transforms = (
            list(compiled_input_transforms)
            if compiled_input_transforms is not None
            else compile_input_transforms(self._input_transforms, assume_validated=True)
        )
        self._complement_spec = complement_spec
        self._complement_values = complement_values
        self._regularization_standardized = bool(regularization_standardized)
        # Post-fit intercept shift applied by ``relevel()``; zero for ordinary
        # fits. Stored Python-side rather than mutating the Rust result so the
        # underlying ``self._result`` stays the original immutable fit.
        self._intercept_delta: float = 0.0
        self._intercept_delta_var: float = 0.0
        self._relevel_history: list[dict[str, Any]] = []

    @property
    def input_transforms(self) -> list[dict[str, Any]]:
        """Canonical deterministic input transforms stored on this model."""
        return copy.deepcopy(self._input_transforms)

    @property
    def warnings(self) -> list[str]:
        """Fit-time warnings collected by the solver (non-convergence, clamps, etc.).

        Empty for deserialized models, which do not carry the solver's warning
        buffer; ``solver_status`` remains the durable signal there.
        """
        warns = getattr(self._result, "warnings", None)
        if warns is None:
            return []
        resolved = warns() if callable(warns) else warns
        return list(resolved)

    def prepare_input(self, raw_df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
        """Return a DataFrame with deterministic input transforms applied.

        This is primarily useful for debugging and parity tests. Prediction
        methods call the same transform layer internally.
        """
        data = _collect_lazyframe(raw_df, set())
        return self._apply_model_input_transforms(data)

    def _apply_model_input_transforms(self, data: pl.DataFrame) -> pl.DataFrame:
        """Apply this model's transforms, recomputing any existing outputs."""
        if not self._compiled_input_transforms:
            return data
        drop_outputs = [
            spec["output"] for spec in self._input_transforms if spec["output"] in data.columns
        ]
        if drop_outputs:
            data = data.drop(drop_outputs)
        return apply_input_transforms(data, self._compiled_input_transforms)

    def _uses_transform_output(self, spec: Any) -> bool:
        """Return True when a string prediction spec is produced by a transform."""
        if not isinstance(spec, str):
            return False
        return any(transform["output"] == spec for transform in self._input_transforms)

    def _prediction_aux_needs_prepared_data(
        self,
        *,
        offset: str | np.ndarray | None,
        exposure: str | np.ndarray | None,
        complement: str | np.ndarray | GLMModel | None,
    ) -> bool:
        """Whether prediction auxiliaries must be read after transforms."""
        exposure_to_use = exposure if exposure is not None else self._exposure_spec
        offset_to_use = offset if offset is not None else self._offset_spec
        complement_to_use = complement if complement is not None else self._complement_spec
        return (
            self._uses_transform_output(exposure_to_use)
            or self._uses_transform_output(offset_to_use)
            or self._uses_transform_output(complement_to_use)
        )

    @property
    def params(self) -> np.ndarray:
        """Fitted coefficients with any post-fit intercept relevel applied.

        For un-releveled models this is the underlying Rust coefficient array;
        after :meth:`relevel`, the intercept (column 0, named ``"Intercept"``)
        carries the accumulated ``log(c)`` shift while every other entry is
        bit-identical to the original fit.
        """
        raw = self._result.params
        if (
            not self._intercept_delta
            or not self.feature_names
            or self.feature_names[0] != "Intercept"
        ):
            return raw
        shifted = np.array(raw, dtype=np.float64, copy=True)
        shifted[0] = shifted[0] + self._intercept_delta
        return shifted

    @property
    def intercept_delta(self) -> float:
        """Accumulated ``log(c)`` intercept shift applied by ``relevel()``."""
        return self._intercept_delta

    @property
    def intercept_delta_var(self) -> float:
        """Accumulated approximate ``Var(log c)`` from ``relevel()``.

        Folded into the intercept's standard error so its CI/z/p reflect the
        calibration step's own uncertainty (delta method; ``0.0`` for
        un-releveled models). See :meth:`relevel`.
        """
        return self._intercept_delta_var

    @property
    def relevel_history(self) -> list[dict[str, Any]]:
        """Per-call metadata for every ``relevel()`` applied to this model.

        Each entry records the calibration factor, intercept before/after,
        ``log_shift``, ``n_obs`` and ``total_weight``; the list is empty for
        un-releveled models.
        """
        return [dict(entry) for entry in self._relevel_history]

    @property
    def linear_predictor(self) -> np.ndarray:
        raw = np.asarray(self._result.linear_predictor, dtype=np.float64)
        if self._intercept_delta and self.link == "log":
            return raw + self._intercept_delta
        return raw

    @property
    def fittedvalues(self) -> np.ndarray:
        raw = np.asarray(self._result.fittedvalues, dtype=np.float64)
        if self._intercept_delta and self.link == "log":
            return raw * float(np.exp(self._intercept_delta))
        return raw

    @property
    def deviance(self) -> float:
        if not self._intercept_delta:
            return self._result.deviance
        try:
            from rustystats.regularization_path import compute_deviance

            y = np.asarray(self._result.fittedvalues, dtype=np.float64) + np.asarray(
                self._result.resid_response(), dtype=np.float64
            )
            return float(
                compute_deviance(
                    y,
                    self.fittedvalues,
                    self.family,
                    var_power=getattr(self, "var_power", 1.5),
                    theta=getattr(self, "theta", 1.0) or 1.0,
                    allow_extended_tweedie=getattr(self, "allow_extended_tweedie", True),
                )
            )
        except Exception:
            return self._result.deviance

    def llf(self) -> float:
        if not self._intercept_delta:
            return self._result.llf()
        try:
            from rustystats._rustystats import compute_dataset_metrics_py

            y = np.asarray(self._result.fittedvalues, dtype=np.float64) + np.asarray(
                self._result.resid_response(), dtype=np.float64
            )
            metrics = compute_dataset_metrics_py(
                y,
                self.fittedvalues,
                self.family,
                len(self.params),
                getattr(self, "var_power", 1.5),
                getattr(self, "theta", 1.0) or 1.0,
                self._result.scale(),
            )
            return float(metrics["log_likelihood"])
        except Exception:
            return self._result.llf()

    def _intercept_releveled(self) -> bool:
        """Whether :meth:`relevel` has shifted the intercept.

        When true, the intercept's Wald inference is recentred on the shifted
        estimate and its standard error is inflated by the relevel calibration
        variance (``sqrt(se² + Var(log c))``), so the CI / z / p reflect both
        the shift and the calibration step's own uncertainty. Other coefficients
        are untouched. See :meth:`_releveled_intercept_inference`.
        """
        return (
            bool(self._intercept_delta)
            and bool(self.feature_names)
            and self.feature_names[0] == "Intercept"
        )

    def _releveled_intercept_inference(
        self, raw_se0: float, raw_ci0: tuple[float, float] | None = None
    ) -> dict[str, Any] | None:
        """Corrected intercept Wald row for a releveled model (else ``None``).

        Given the *raw* (model-based or robust) intercept SE — and optionally its
        raw CI bounds on the link scale — recentre on the shifted estimate
        ``params[0]``, inflate the SE by the relevel calibration variance, and
        rebuild ``z``/``p``/``signif`` (normal Wald) and the CI (preserving the
        raw critical multiplier). Pass robust ``raw_se0``/``raw_ci0`` to get the
        robust-flavoured correction.
        """
        if not self._intercept_releveled():
            return None
        from rustystats.glm import _normal_two_sided_p, _significance_code

        est = float(self.params[0])
        raw_se0 = float(raw_se0)
        se = float(np.sqrt(raw_se0**2 + self._intercept_delta_var)) if raw_se0 > 0 else raw_se0
        if se > 1e-10:
            z = est / se
            p = _normal_two_sided_p(z)
            signif = _significance_code(p)
        else:
            z, p, signif = float("nan"), float("nan"), ""
        out: dict[str, Any] = {"se": se, "z": z, "p": p, "signif": signif}
        if raw_ci0 is not None and raw_se0 > 1e-10:
            z_crit = 0.5 * (float(raw_ci0[1]) - float(raw_ci0[0])) / raw_se0
            out["ci_lo"] = est - z_crit * se
            out["ci_hi"] = est + z_crit * se
        return out

    def _with_releveled_intercept_stat(
        self, values: np.ndarray, raw_se: np.ndarray, key: str
    ) -> np.ndarray:
        """Replace the intercept statistic with its relevel-aware value."""
        corr = self._releveled_intercept_inference(raw_se[0]) if len(raw_se) else None
        if corr is None:
            return values
        out = values.copy()
        out[0] = corr[key]
        return out

    def _with_releveled_intercept_ci(self, ci: np.ndarray, raw_se: np.ndarray) -> np.ndarray:
        """Replace the intercept CI with its relevel-aware interval."""
        if len(raw_se) == 0 or ci.shape[0] == 0:
            return ci
        corr = self._releveled_intercept_inference(raw_se[0], (ci[0, 0], ci[0, 1]))
        if corr is None:
            return ci
        out = ci.copy()
        out[0, 0] = corr["ci_lo"]
        out[0, 1] = corr["ci_hi"]
        return out

    def bse(self) -> np.ndarray:
        """Model-based standard errors with relevel calibration variance applied."""
        se = np.asarray(self._result.bse(), dtype=np.float64)
        return self._with_releveled_intercept_stat(se, se, "se")

    def tvalues(self) -> np.ndarray:
        """Wald z/t values with a releveled intercept recentred when applicable."""
        z = np.asarray(self._result.tvalues(), dtype=np.float64)
        raw_se = np.asarray(self._result.bse(), dtype=np.float64)
        return self._with_releveled_intercept_stat(z, raw_se, "z")

    def pvalues(self) -> np.ndarray:
        """Two-sided Wald p-values with a releveled intercept recentred when applicable."""
        p = np.asarray(self._result.pvalues(), dtype=np.float64)
        raw_se = np.asarray(self._result.bse(), dtype=np.float64)
        return self._with_releveled_intercept_stat(p, raw_se, "p")

    def significance_codes(self) -> list[str]:
        """Significance codes aligned to :meth:`pvalues`."""
        codes = list(self._result.significance_codes())
        raw_se = np.asarray(self._result.bse(), dtype=np.float64)
        corr = self._releveled_intercept_inference(raw_se[0]) if len(raw_se) else None
        if corr is not None and codes:
            codes[0] = corr["signif"]
        return codes

    def conf_int(self, alpha: float = 0.05) -> np.ndarray:
        """Model-based confidence intervals with a releveled intercept recentred."""
        ci = np.asarray(self._result.conf_int(alpha), dtype=np.float64)
        raw_se = np.asarray(self._result.bse(), dtype=np.float64)
        return self._with_releveled_intercept_ci(ci, raw_se)

    def bse_robust(self, hc_type: str = "HC1") -> np.ndarray:
        """Robust standard errors with relevel calibration variance applied."""
        se = np.asarray(self._result.bse_robust(hc_type), dtype=np.float64)
        return self._with_releveled_intercept_stat(se, se, "se")

    def tvalues_robust(self, hc_type: str = "HC1") -> np.ndarray:
        """Robust Wald z/t values with a releveled intercept recentred."""
        z = np.asarray(self._result.tvalues_robust(hc_type), dtype=np.float64)
        raw_se = np.asarray(self._result.bse_robust(hc_type), dtype=np.float64)
        return self._with_releveled_intercept_stat(z, raw_se, "z")

    def pvalues_robust(self, hc_type: str = "HC1") -> np.ndarray:
        """Robust two-sided Wald p-values with a releveled intercept recentred."""
        p = np.asarray(self._result.pvalues_robust(hc_type), dtype=np.float64)
        raw_se = np.asarray(self._result.bse_robust(hc_type), dtype=np.float64)
        return self._with_releveled_intercept_stat(p, raw_se, "p")

    def conf_int_robust(self, alpha: float = 0.05, cov_type: str = "HC1") -> np.ndarray:
        """Robust confidence intervals with a releveled intercept recentred."""
        ci = np.asarray(
            self._result.conf_int_robust(alpha=alpha, cov_type=cov_type),
            dtype=np.float64,
        )
        raw_se = np.asarray(self._result.bse_robust(cov_type), dtype=np.float64)
        return self._with_releveled_intercept_ci(ci, raw_se)

    def _family_unit_variance(self, mu: np.ndarray) -> np.ndarray:
        """Family variance function ``V(mu)`` for the log-link families relevel
        supports — used only for the relevel calibration-variance estimate."""
        base = self.family.lower().split("(", 1)[0].strip()
        mu = np.asarray(mu, dtype=np.float64)
        if base in ("poisson", "quasipoisson"):
            return mu
        if base == "gamma":
            return mu * mu
        from rustystats.diagnostics.api import _parse_family_params

        var_power, theta = _parse_family_params(self.family)
        if base == "tweedie":
            return np.power(mu, var_power)
        if base in ("negativebinomial", "negbinomial"):
            t = theta if theta and theta > 0 else 1.0
            return mu + mu * mu / t
        return np.ones_like(mu)  # gaussian (log link) and any other

    def _relevel_log_factor_variance(
        self, y: np.ndarray, mu: np.ndarray, weights_arr: np.ndarray | None
    ) -> float:
        """Delta-method ``Var(log c)`` for one relevel call (0.0 if not finite).

        Treats the expected total ``Σ(w·mu)`` as known and the actual total
        ``A = Σ(w·y)`` as random with ``Var(y_i) = phi·V(mu_i)/w_i``, giving
        ``Var(log c) ≈ phi·Σ(w_i·V(mu_i)) / A²``.
        """
        w = np.ones_like(mu) if weights_arr is None else np.asarray(weights_arr, dtype=np.float64)
        a = float(np.sum(w * y))
        if a <= 0.0:
            return 0.0
        try:
            phi = float(self._result.scale())
        except Exception:
            phi = 1.0
        if not np.isfinite(phi) or phi <= 0.0:
            phi = 1.0
        var = phi * float(np.sum(w * self._family_unit_variance(mu))) / (a * a)
        return var if np.isfinite(var) and var >= 0.0 else 0.0

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying result object.

        This handles all properties and methods from PyGLMResults that are
        not explicitly defined on GLMModel (resid_*, llf, aic, bic, scale,
        regularization properties, etc.). Public inference accessors are
        defined above so relevel() can recentre the intercept consistently.
        """
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._result, name)

    @property
    def smooth_terms(self) -> list[SmoothTermResult] | None:
        """Smooth term results with EDF, lambda, and GCV for each s() term."""
        return self._smooth_results

    @property
    def total_edf(self) -> float | None:
        """Total effective degrees of freedom (parametric + smooth terms)."""
        return self._total_edf

    @property
    def gcv(self) -> float | None:
        """Generalized Cross-Validation score for smoothness selection."""
        return self._gcv

    def has_smooth_terms(self) -> bool:
        """Check if model contains smooth terms with automatic smoothing."""
        return self._smooth_results is not None and len(self._smooth_results) > 0

    @property
    def terms_dict(self) -> dict[str, dict[str, Any]] | None:
        """Original terms dictionary used to specify the model (dict API only)."""
        return self._terms_dict

    @property
    def interactions_spec(self) -> list[dict[str, Any]] | None:
        """Original interactions specification used to specify the model (dict API only)."""
        return self._interactions_spec

    def get_design_matrix(self) -> np.ndarray | None:
        """Get the design matrix X used in fitting.

        Returns None if store_design_matrix=False was used (lean mode).
        """
        try:
            dm = self._result.design_matrix
            if dm is None:
                return None
            return np.asarray(dm)
        except AttributeError:
            return None

    def get_irls_weights(self) -> np.ndarray | None:
        """Get the IRLS working weights from final iteration."""
        try:
            return np.asarray(self._result.irls_weights)
        except AttributeError:
            return None

    def get_bread_matrix(self) -> np.ndarray | None:
        """Get the (X'WX)^-1 matrix (unscaled covariance)."""
        try:
            return np.asarray(self._result.cov_params_unscaled)
        except AttributeError:
            return None

    def selected_features(self) -> list[str]:
        """
        Get names of features with non-zero coefficients.

        Useful for Lasso/Elastic Net to see which variables were selected.
        """
        indices = self._result.selected_features()
        return [self.feature_names[i] for i in indices]

    @property
    def required_columns(self) -> list[str]:
        """Raw DataFrame columns required to predict with this model.

        Unlike ``feature_names`` (which lists the post-encoding design-matrix
        columns, e.g. ``Region[B]``, ``VehAge_bs1``), this returns the raw
        input columns — including any source columns referenced by
        ``expression`` terms, plus offset and complement columns. Use it to
        project a LazyFrame before collecting:

            df.lazy().select(model.required_columns).collect()
        """
        if self._terms_dict is None:
            raise RuntimeError(
                "required_columns is only available for models fitted via glm_dict()"
            )
        return sorted(
            _extract_needed_columns(
                terms=self._terms_dict,
                interactions=self._interactions_spec,
                input_transforms=self._input_transforms,
                offset=self._offset_spec,
                exposure=self._exposure_spec,
                complement=self._complement_spec,
            )
        )

    # CV-based regularization path properties
    @property
    def cv_deviance(self) -> float | None:
        """CV deviance at selected alpha (only available when fit with cv=)."""
        if self._regularization_path_info is None:
            return None
        return self._regularization_path_info.cv_deviance

    @property
    def cv_deviance_se(self) -> float | None:
        """Standard error of CV deviance (only available when fit with cv=)."""
        if self._regularization_path_info is None:
            return None
        return self._regularization_path_info.cv_deviance_se

    @property
    def regularization_type(self) -> str | None:
        """Type of regularization: 'ridge', 'lasso', 'elastic_net', or 'none'."""
        if self._regularization_path_info is None:
            # Fall back to penalty_type from underlying result
            return self.penalty_type
        return self._regularization_path_info.regularization_type

    @property
    def regularization_path(self) -> list[dict] | None:
        """
        Full regularization path results (only available when fit with cv=).

        Returns list of dicts with keys: alpha, l1_ratio, cv_deviance_mean,
        cv_deviance_se, n_nonzero, max_coef.
        """
        if self._regularization_path_info is None:
            return None
        return [
            {
                "alpha": r.alpha,
                "l1_ratio": r.l1_ratio,
                "cv_deviance_mean": r.cv_deviance_mean,
                "cv_deviance_se": r.cv_deviance_se,
                "n_nonzero": r.n_nonzero,
                "max_coef": r.max_coef,
            }
            for r in self._regularization_path_info.path
        ]

    @property
    def cv_selection_method(self) -> str | None:
        """Selection method used: 'min' or '1se' (only available when fit with cv=)."""
        if self._regularization_path_info is None:
            return None
        return self._regularization_path_info.selection_method

    @property
    def n_cv_folds(self) -> int | None:
        """Number of CV folds used (only available when fit with cv=)."""
        if self._regularization_path_info is None:
            return None
        return self._regularization_path_info.n_folds

    @property
    def cv_convergence(self) -> dict | None:
        """Convergence settings used for CV fold fits (RS-ACT-001).

        ``{"max_iter": int, "tol": float}`` reflecting the settings actually used
        for the cross-validation fold fits, or ``None`` when not fit with ``cv=``.
        """
        if self._regularization_path_info is None:
            return None
        return {
            "max_iter": self._regularization_path_info.cv_max_iter,
            "tol": self._regularization_path_info.cv_tol,
        }

    @property
    def cv_fold_scores(self) -> dict[float, list[float]] | None:
        """Per-alpha validation fold scores when retained by the CV path."""
        if self._regularization_path_info is None:
            return None
        return self._regularization_path_info.cv_fold_scores

    @property
    def cv_scoring_objective(self) -> str | None:
        """Name of the CV scoring objective."""
        if self._regularization_path_info is None:
            return None
        return self._regularization_path_info.cv_scoring_objective

    @property
    def fold_safe_target_encoding(self) -> bool | None:
        """Whether CV used fold-specific stateful transforms for target encoding."""
        if self._regularization_path_info is None:
            return None
        return self._regularization_path_info.fold_safe_target_encoding

    @property
    def boundary_active_coefficients(self) -> list[dict[str, Any]]:
        """Constrained coefficients that are active at their boundary."""
        nonneg, nonpos = _get_constraint_indices(self.feature_names)
        rows: list[dict[str, Any]] = []
        params = np.asarray(self.params, dtype=np.float64)
        for idx in nonneg:
            if idx < len(params) and params[idx] <= 1e-10:
                rows.append(
                    {
                        "feature": self.feature_names[idx],
                        "constraint": "nonnegative",
                        "coefficient": float(params[idx]),
                    }
                )
        for idx in nonpos:
            if idx < len(params) and params[idx] >= -1e-10:
                rows.append(
                    {
                        "feature": self.feature_names[idx],
                        "constraint": "nonpositive",
                        "coefficient": float(params[idx]),
                    }
                )
        return rows

    @property
    def nobs(self) -> int:
        """Number of observations."""
        return self._result.nobs

    @property
    def df_resid(self) -> float:
        """Residual degrees of freedom.

        For smooth models, uses n - total_edf instead of n - p,
        where total_edf accounts for the effective complexity of
        penalized smooth terms.
        """
        if self._total_edf is not None:
            return self._result.nobs - self._total_edf
        return self._result.df_resid

    @property
    def df_model(self) -> float:
        """Model degrees of freedom.

        For smooth models, uses total_edf - 1 (excluding intercept)
        instead of raw p - 1.
        """
        if self._total_edf is not None:
            return self._total_edf - 1
        return self._result.df_model

    @property
    def is_quasi_likelihood(self) -> bool:
        """Whether the fit is a quasi-likelihood family (RS-ACT-008).

        ``True`` for quasi-Poisson and quasi-Binomial — neither has a proper
        log-likelihood, so AIC/BIC are not meaningful in the ordinary sense
        and ``aic()`` / ``bic()`` return ``None`` for these fits. The summary
        labels the loglik-like value as "Quasi-Log-Likelihood".

        Derived from ``self.family`` so it survives a ``to_bytes`` /
        ``from_bytes`` round-trip without any extra serialisation state.
        """
        family_lower = (self.family or "").lower()
        family_base = family_lower.split("(", 1)[0].strip()
        # Match the same alias forms as families_py.rs / default_link_name.
        return family_base in {
            "quasipoisson",
            "quasi-poisson",
            "quasi_poisson",
            "quasibinomial",
            "quasi-binomial",
            "quasi_binomial",
        }

    def _raw_information_criteria_are_valid(self) -> bool:
        """Whether the raw (parameter-count) AIC/BIC are methodologically valid.

        ``False`` after regularization, post-selection / post-CV fits, or
        active sign constraints — the effective degrees of freedom are not
        the raw parameter count, and the summary already hides Std.Err/
        p-value columns for these statuses. The penalized-smooth path is
        unaffected because it uses effective df directly (computed in
        :meth:`aic` / :meth:`bic`), not ``self._result.aic()`` /
        ``self._result.bic()``. Mirrors the gate used by ``summary()`` so
        the public accessors do not emit numbers the summary would refuse
        to display (RS-ACT-011).
        """
        status = getattr(self, "inference_status", None)
        # An unknown status (None) preserves the historical default of
        # returning a number, so unpenalized GLMs that pre-date inference
        # gating still behave as before. The summary also defaults to
        # showing standard inference when status is None.
        return status is None or status in {"valid_standard", "valid_robust"}

    def aic(self) -> float | None:
        """Akaike Information Criterion.

        Returns ``None`` for quasi-likelihood families (RS-ACT-008) and for
        regularized / post-selection / constrained fits where the raw
        parameter count is not the effective df (RS-ACT-011). For penalized
        smooth models, uses effective degrees of freedom (``total_edf``) in
        place of the raw basis-column count.
        """
        if self.is_quasi_likelihood:
            return None
        if self._total_edf is not None:
            # Penalized smooth: EDF-based AIC is the correct measure here,
            # regardless of inference_status (which is "unavailable" for
            # smooth fits but for an orthogonal reason — model-based SEs
            # need the smoothed bread, not because the parameter count is
            # wrong).
            return -2.0 * self.llf() + 2.0 * self._total_edf
        if not self._raw_information_criteria_are_valid():
            return None
        raw_aic = getattr(self._result, "aic", None)
        return raw_aic() if callable(raw_aic) else None

    def bic(self) -> float | None:
        """Bayesian Information Criterion.

        Returns ``None`` for quasi-likelihood families (RS-ACT-008) and for
        regularized / post-selection / constrained fits where the raw
        parameter count is not the effective df (RS-ACT-011); see
        :meth:`aic` for the EDF rationale.
        """
        if self.is_quasi_likelihood:
            return None
        if self._total_edf is not None:
            return -2.0 * self.llf() + self._total_edf * np.log(self._result.nobs)
        if not self._raw_information_criteria_are_valid():
            return None
        raw_bic = getattr(self._result, "bic", None)
        return raw_bic() if callable(raw_bic) else None

    def compute_loss(
        self,
        data: pl.DataFrame,
        response: str | None = None,
        exposure: str | None = None,
    ) -> float:
        """
        Compute family-appropriate loss (mean deviance) on given data.

        This method re-predicts on the data to ensure consistent encoding,
        which is critical for TE() terms that use leave-one-out during fit
        but full encoding for prediction.

        Parameters
        ----------
        data : pl.DataFrame
            Data to compute loss on (can be train, test, or holdout).
        response : str, optional
            Response column name. Auto-detected from formula if not provided.
        exposure : str, optional
            Exposure column name for rate models.

        Returns
        -------
        float
            Mean deviance (family-appropriate loss metric).

        Examples
        --------
        >>> train_loss = result.compute_loss(train_data)
        >>> test_loss = result.compute_loss(test_data)
        >>> assert train_loss < test_loss  # Expected for non-overfitting models
        """
        from rustystats._rustystats import compute_loss_metrics_py as _rust_loss_metrics

        # Get response column from formula
        if response is None:
            formula_parts = self.formula.split("~")
            response = formula_parts[0].strip() if formula_parts else None

        if response is None or response not in data.columns:
            raise ValidationError(f"Response column '{response}' not found in data")

        y = data[response].to_numpy().astype(np.float64)

        # Re-predict to get consistent encoding (critical for TE terms)
        mu = np.asarray(self.predict(data), dtype=np.float64)

        # Compute family-appropriate loss
        loss_metrics = _rust_loss_metrics(y, mu, self.family)
        return loss_metrics["family_loss"]

    def coef_table(self) -> pl.DataFrame:
        """
        Return coefficients as a DataFrame with names.

        Returns
        -------
        pl.DataFrame
            DataFrame with columns: Feature, Estimate, Std.Error, z, Pr(>|z|), Signif
        """
        import polars as pl

        status = getattr(self, "inference_status", None)
        show_standard_inference = status is None or status in {"valid_standard", "valid_robust"}
        n = len(self.feature_names)
        if show_standard_inference:
            se = np.asarray(self.bse(), dtype=np.float64)
            z = np.asarray(self.tvalues(), dtype=np.float64)
            p = np.asarray(self.pvalues(), dtype=np.float64)
            signif = list(self.significance_codes())
        else:
            se = np.full(n, np.nan)
            z = np.full(n, np.nan)
            p = np.full(n, np.nan)
            signif = [""] * n

        return pl.DataFrame(
            {
                "Feature": self.feature_names,
                "Estimate": self.params,
                "Std.Error": se,
                "z": z,
                "Pr(>|z|)": p,
                "Signif": signif,
            }
        )

    def relativities(self) -> pl.DataFrame:
        """
        Return relativities (exp(coef)) for log-link models.

        Returns
        -------
        pl.DataFrame
            DataFrame with Feature, Relativity and confidence interval columns
        """
        import polars as pl

        if self.link not in ("log",):
            raise ValidationError(f"Relativities only meaningful for log link, not '{self.link}'")

        # RS-ACT-011: ordinary confidence intervals are not valid after
        # regularization / CV selection / constraints. Keep the point relativity
        # (always meaningful) but null the CI, mirroring coef_table().
        status = getattr(self, "inference_status", None)
        show_standard_inference = status is None or status in {"valid_standard", "valid_robust"}
        if show_standard_inference:
            ci = self.conf_int()
            ci_lower = np.exp(ci[:, 0])
            ci_upper = np.exp(ci[:, 1])
        else:
            n = len(self.feature_names)
            ci_lower = np.full(n, np.nan)
            ci_upper = np.full(n, np.nan)

        return pl.DataFrame(
            {
                "Feature": self.feature_names,
                "Relativity": np.exp(self.params),
                "CI_Lower": ci_lower,
                "CI_Upper": ci_upper,
            }
        )

    @property
    def has_complement(self) -> bool:
        """Whether the model was fitted with a complement of credibility."""
        return self._complement_values is not None

    def credibility_summary(self) -> pl.DataFrame:
        """
        Credibility summary for models fitted with a complement.

        Shows each coefficient's deviation from the complement. Coefficients
        shrunk to zero indicate full credibility in the complement for that
        term. Non-zero deviations indicate the data supports a different value.

        For log-link models, the Deviation_Factor column shows exp(beta) — the
        multiplicative adjustment applied to the complement. A value of 1.0
        means the complement is fully trusted.

        Returns
        -------
        pl.DataFrame
            DataFrame with Feature, Coef (deviation from complement),
            Zeroed (whether lasso shrunk to zero), and Deviation_Factor
            (for log-link models).

        Raises
        ------
        ValidationError
            If the model was not fitted with a complement.
        """
        import polars as pl

        if not self.has_complement:
            raise ValidationError(
                "credibility_summary() requires a model fitted with complement=. "
                "Use summary() for standard models."
            )

        coefs = self.params
        zeroed = np.abs(coefs) < 1e-10
        n_zeroed = int(np.sum(zeroed))
        n_total = len(coefs) - 1  # Exclude intercept

        result = {
            "Feature": self.feature_names,
            "Deviation": coefs,
            "Zeroed": zeroed.tolist(),
        }

        if self.link == "log":
            result["Deviation_Factor"] = np.exp(coefs)

        df = pl.DataFrame(result)
        # Add summary note as metadata attribute
        df.__dict__["_credibility_note"] = (
            f"{n_zeroed}/{n_total} non-intercept terms zeroed (complement fully trusted)"
        )
        return df

    def summary(self) -> str:
        """
        Generate a formatted summary string.

        When the model was fitted with a complement of credibility,
        the summary includes a note that coefficients represent
        deviations from the complement.

        Returns
        -------
        str
            Formatted summary table
        """
        from rustystats.glm import summary

        title = "GLM Results"
        if self.has_complement:
            title = "Lasso Credibility Results"
            if self._complement_spec is not None:
                if isinstance(self._complement_spec, str):
                    title += f" (complement: {self._complement_spec})"
                elif isinstance(self._complement_spec, GLMModel):
                    title += " (complement: GLMModel)"

        result = summary(
            self,
            feature_names=self.feature_names,
            title=title,
            inference_status=getattr(self, "inference_status", None),
            solver_status=getattr(self, "solver_status", None),
            optimizer_route=getattr(self, "optimizer_route", None),
            effective_df=self._total_edf,
            is_quasi_likelihood=self.is_quasi_likelihood,
        )

        if self.has_complement:
            n_zeroed = int(np.sum(np.abs(self.params[1:]) < 1e-10))
            n_total = len(self.params) - 1
            result += (
                f"\nNote: Coefficients are deviations from the complement of credibility.\n"
                f"      {n_zeroed}/{n_total} non-intercept terms zeroed "
                f"(complement fully trusted).\n"
            )

        if self._intercept_releveled():
            result += (
                "\nNote: the intercept reflects relevel(); its CI/z/p are recentred on the\n"
                "      shifted estimate and its SE is inflated by the calibration variance\n"
                "      Var(log c). Other coefficients and relativities are unchanged.\n"
            )

        if self._regularization_standardized:
            result += (
                "\nNote: regularized coefficients were fit with internal column standardization\n"
                "      and reported on the original design scale.\n"
            )

        return result

    def diagnostics(
        self,
        train_data: pl.DataFrame,
        categorical_factors: list[str] | None = None,
        continuous_factors: list[str] | None = None,
        n_calibration_bins: int = 10,
        n_factor_bins: int = 10,
        rare_threshold_pct: float = 1.0,
        max_categorical_levels: int = 20,
        detect_interactions: bool = False,
        max_interaction_factors: int = 10,
        # User-specified interaction pairs for per-pair surface diagnostics.
        interactions: list[Any] | None = None,
        # Test data for overfitting detection (response/exposure auto-inferred)
        test_data: pl.DataFrame | None = None,
        # Control enhanced diagnostics
        compute_vif: bool = True,
        compute_coefficients: bool = True,
        compute_deviance_by_level: bool = True,
        compute_lift: bool = True,
        compute_partial_dep: bool = True,
        compute_robust_se: bool = True,
        compute_score_tests: bool = True,
        # Base predictions comparison
        base_predictions: str | None = None,
        ranking: str = "auto",
        exposure: str | np.ndarray | None = None,
        weights: str | np.ndarray | None = None,
    ) -> ModelDiagnostics:
        """
        Compute comprehensive model diagnostics.

        Parameters
        ----------
        train_data : pl.DataFrame
            Training data used for fitting.
        categorical_factors : list of str, optional
            Names of categorical factors to analyze (both fitted and unfitted).
        continuous_factors : list of str, optional
            Names of continuous factors to analyze (both fitted and unfitted).
        n_calibration_bins : int, default=10
            Number of bins for calibration curve.
        n_factor_bins : int, default=10
            Number of quantile bins for continuous factors.
        rare_threshold_pct : float, default=1.0
            Threshold (%) below which categorical levels are grouped into "Other".
        max_categorical_levels : int, default=20
            Maximum number of categorical levels to show.
        detect_interactions : bool, default=True
            Whether to detect potential interactions.
        max_interaction_factors : int, default=10
            Maximum factors to consider for interaction detection.
        interactions : list, optional
            Explicit list of variable pairs for per-pair surface diagnostics.
            Each entry: ``{"factor1": ..., "factor2": ...}``, ``(a, b)``, or
            ``[a, b]``. Pairs do not need to appear in the fitted model;
            ``InteractionDiagnostics.in_model`` is set from TermSlot membership.
            Independent of ``detect_interactions=`` (which fills
            ``interaction_candidates``); both can be used simultaneously.
        test_data : pl.DataFrame, optional
            Test/holdout data for overfitting detection. Response and exposure
            columns are automatically inferred from the model's formula. When
            supplied alongside ``interactions=``, each pair also receives a
            ``test_surface_grid`` cell-aligned with the train surface.
        compute_vif : bool, default=True
            Compute VIF/multicollinearity scores for design matrix (train-only).
        compute_coefficients : bool, default=True
            Compute coefficient summary with interpretations (train-only).
        compute_deviance_by_level : bool, default=True
            Compute deviance breakdown by categorical factor levels.
        compute_lift : bool, default=True
            Compute full lift chart with all deciles.
        compute_partial_dep : bool, default=True
            Compute partial dependence plots for each variable.
        base_predictions : str, optional
            Column name in train_data containing predictions from another model
            (e.g., a base/benchmark model). When provided, computes:
            - A/E ratio, loss, Gini for base predictions
            - Model vs base decile analysis sorted by model/base ratio
            - Summary of which model performs better in each decile
        ranking : {"auto", "mean", "rate"}, default="auto"
            Decile/lift ranking mode. ``"auto"`` ranks by predicted rate when
            exposure is present and by raw predicted mean otherwise.
        weights : str or array-like, optional
            Prior weights for the decile/lift aggregates, which then report
            Σw·y / Σw·μ / Σw·exposure (RS-ACT-004). When omitted, the model's
            fitted prior weights are auto-propagated; pass an explicit array (or
            ``np.ones(n)`` to force unweighted) to override. Ranking is
            unaffected — weights scale the aggregates, not the per-row rate.

        Returns
        -------
        ModelDiagnostics
            Complete diagnostics object with to_json() method.

            Fields for agentic workflows:
            - vif: VIF scores detecting multicollinearity (train-only)
            - coefficient_summary: Coefficient magnitudes and recommendations (train-only)
            - factor_deviance: Deviance by categorical level
            - lift_chart: Full lift chart showing discrimination by decile
            - partial_dependence: Marginal effect shapes for linear vs spline decisions
            - train_test: Comprehensive train vs test comparison with flags:
                - overfitting_risk: True if gini_gap > 0.03
                - calibration_drift: True if test A/E outside [0.95, 1.05]
                - unstable_factors: Factors where train/test A/E differ by > 0.1

        Examples
        --------
        >>> result = rs.glm_dict(response="ClaimNb", terms={"Age": {"type": "linear"}, "Region": {"type": "categorical"}}, data=data, family="poisson", exposure="Exposure").fit()
        >>>
        >>> # Basic diagnostics
        >>> diagnostics = result.diagnostics(
        ...     train_data=train_data,
        ...     categorical_factors=["Region", "VehBrand"],
        ...     continuous_factors=["Age", "VehPower"]
        ... )
        >>>
        >>> # With test data for overfitting detection
        >>> diagnostics = result.diagnostics(
        ...     train_data=train_data,
        ...     test_data=test_data,
        ...     categorical_factors=["Region"],
        ...     continuous_factors=["Age"],
        ... )
        >>>
        >>> # Check overfitting flags
        >>> if diagnostics.train_test and diagnostics.train_test.overfitting_risk:
        ...     print("Warning: Overfitting detected!")
        >>>
        >>> print(diagnostics.to_json())
        """
        from rustystats.diagnostics import compute_diagnostics

        # Deserialized models lack covariance / design matrix — disable
        # features that depend on them to avoid AttributeErrors.
        if self._is_deserialized:
            compute_vif = False
            compute_coefficients = False
            compute_robust_se = False

        return compute_diagnostics(
            result=self,
            train_data=train_data,
            categorical_factors=categorical_factors,
            continuous_factors=continuous_factors,
            n_calibration_bins=n_calibration_bins,
            n_factor_bins=n_factor_bins,
            rare_threshold_pct=rare_threshold_pct,
            max_categorical_levels=max_categorical_levels,
            detect_interactions=detect_interactions,
            max_interaction_factors=max_interaction_factors,
            interactions=interactions,
            test_data=test_data,
            compute_vif=compute_vif,
            compute_coefficients=compute_coefficients,
            compute_deviance_by_level=compute_deviance_by_level,
            compute_lift=compute_lift,
            compute_partial_dep=compute_partial_dep,
            compute_robust_se=compute_robust_se,
            compute_score_tests=compute_score_tests,
            base_predictions=base_predictions,
            ranking=ranking,
            exposure=exposure,
            weights=weights,
        )

    def diagnostics_json(
        self,
        train_data: pl.DataFrame,
        categorical_factors: list[str] | None = None,
        continuous_factors: list[str] | None = None,
        n_calibration_bins: int = 10,
        n_factor_bins: int = 10,
        rare_threshold_pct: float = 1.0,
        max_categorical_levels: int = 20,
        detect_interactions: bool = False,
        max_interaction_factors: int = 10,
        interactions: list[Any] | None = None,
        test_data: pl.DataFrame | None = None,
        compute_score_tests: bool = True,
        ranking: str = "auto",
        exposure: str | np.ndarray | None = None,
        indent: int | None = None,
    ) -> str:
        """
        Compute diagnostics and return as JSON string.

        This is a convenience method that calls diagnostics() and converts
        the result to JSON. The output is optimized for LLM consumption.

        Parameters
        ----------
        train_data : pl.DataFrame
            Training data used for fitting.
        categorical_factors : list of str, optional
            Names of categorical factors to analyze.
        continuous_factors : list of str, optional
            Names of continuous factors to analyze.
        test_data : pl.DataFrame, optional
            Test data for overfitting detection.
        compute_score_tests : bool, default=True
            Whether to compute Rao score tests for unfitted factors. Default True.
        ranking : {"auto", "mean", "rate"}, default="auto"
            Decile/lift ranking mode.
        indent : int, optional
            JSON indentation. None for compact output.

        Returns
        -------
        str
            JSON string containing all diagnostics.
        """
        diag = self.diagnostics(
            train_data=train_data,
            categorical_factors=categorical_factors,
            continuous_factors=continuous_factors,
            n_calibration_bins=n_calibration_bins,
            n_factor_bins=n_factor_bins,
            rare_threshold_pct=rare_threshold_pct,
            max_categorical_levels=max_categorical_levels,
            detect_interactions=detect_interactions,
            max_interaction_factors=max_interaction_factors,
            interactions=interactions,
            test_data=test_data,
            compute_score_tests=compute_score_tests,
            ranking=ranking,
            exposure=exposure,
        )
        return diag.to_json(indent=indent)

    def predict(
        self,
        new_data: pl.DataFrame | pl.LazyFrame,
        offset: str | np.ndarray | None = None,
        complement: str | np.ndarray | GLMModel | None = None,
        exposure: str | np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Predict on new data using the fitted model.

        Parameters
        ----------
        new_data : pl.DataFrame or pl.LazyFrame
            New data to predict on. Must have the same columns as training data.
            If a LazyFrame, only needed columns are collected.
        offset : str or array-like, optional
            Link-scale offset for new data. If ``None`` and the model was fit
            with an offset column name, that column is extracted from
            ``new_data``. The values are added to the linear predictor as-is;
            no ``log()`` is applied. For raw positive exposure, pass
            ``exposure=`` instead.
        exposure : str or array-like, optional
            Raw positive exposure for new data (log-link rate models). Added to
            the linear predictor as ``log(exposure)``. If None, exposure is taken
            from the column the model was fit with (when present in ``new_data``).
        complement : str, array-like, or GLMModel, optional
            Complement of credibility for new data (response scale).
            If None and the model was fit with a complement column name,
            that column will be extracted from new_data.

        Returns
        -------
        np.ndarray
            Predicted values (on the response scale, i.e., μ = E[Y]).

        Examples
        --------
        >>> model = rs.glm_dict(response="ClaimNb", terms={"Age": {"type": "linear"}, "Region": {"type": "categorical"}}, data=data, family="poisson", exposure="Exposure")
        >>> result = model.fit()
        >>>
        >>> # Predict on new data
        >>> predictions = result.predict(new_data)
        >>>
        >>> # Predict with custom exposure
        >>> predictions = result.predict(new_data, exposure=new_exposures)
        """
        if self._builder is None:
            raise PredictionError(
                "Cannot predict: model was not fitted with formula API. "
                "Use fittedvalues for training data predictions."
            )

        # Resolve LazyFrame: select only columns needed for prediction
        if self._terms_dict is not None:
            exposure_needed = exposure if exposure is not None else self._exposure_spec
            if exposure_needed is not None and not isinstance(exposure_needed, str):
                exposure_needed = None
            needed = _extract_needed_columns(
                terms=self._terms_dict,
                interactions=self._interactions_spec,
                input_transforms=self._input_transforms,
                offset=offset if offset is not None else self._offset_spec,
                exposure=exposure_needed,
                complement=complement if complement is not None else self._complement_spec,
            )
            new_data = _collect_lazyframe(new_data, needed)
        else:
            new_data = _collect_lazyframe(new_data, set())

        # Compute linear predictor: η = X @ β
        # For large inputs we build the design matrix in row-chunks so that the
        # full (n × p) materialization never coexists in memory; each chunk's
        # X is dropped after its slice of η is written. The chunk size is
        # adaptive to model width via `_compute_predict_chunk_size` — narrow
        # models use the full `_PREDICT_ROW_CHUNK_DEFAULT` row cap, while wide
        # models (many features) shrink per-chunk rows so the design matrix
        # stays within `_PREDICT_CHUNK_BYTES_BUDGET` bytes. The result is
        # FP-equivalent — but not bit-identical — to the single-shot `X @ β`:
        # BLAS gemv reduces row contributions in a different order between one
        # large call and several smaller calls, so per-element diffs are bounded
        # at ~1-2 ULP. See
        # `tests/python/test_dict_api.py::test_predict_chunked_matches_singleshot`.
        n_rows = len(new_data)
        n_features = len(self.params)
        chunk_size = _compute_predict_chunk_size(n_features)
        params = np.asarray(self.params, dtype=np.float64)
        prepared_all: pl.DataFrame | None = None
        if n_rows <= chunk_size:
            # Small input: skip slicing overhead, keep behavior identical to
            # the pre-chunking implementation.
            prepared_all = self._apply_model_input_transforms(new_data)
            X_new = self._builder.transform_new_data(prepared_all)
            linear_pred = X_new @ params
            del X_new
        else:
            linear_pred = np.empty(n_rows, dtype=np.float64)
            for start in range(0, n_rows, chunk_size):
                stop = min(start + chunk_size, n_rows)
                chunk = self._apply_model_input_transforms(new_data.slice(start, stop - start))
                X_chunk = self._builder.transform_new_data(chunk)
                # Write directly into the pre-allocated output slice; the
                # X_chunk reference is rebound on the next iteration so the
                # ~chunk_size × p matrix is freed before the next one is built.
                linear_pred[start:stop] = X_chunk @ params
                del X_chunk, chunk

        aux_data = new_data
        if self._prediction_aux_needs_prepared_data(
            offset=offset,
            exposure=exposure,
            complement=complement,
        ):
            if prepared_all is None:
                prepared_all = self._apply_model_input_transforms(new_data)
            aux_data = prepared_all

        exposure_to_use = exposure if exposure is not None else self._exposure_spec
        if exposure is None and getattr(
            self, "_array_exposure_requires_prediction_override", False
        ):
            raise PredictionError(
                "This model was fit with an array exposure, which is fit-time data and "
                "cannot be reused as a prediction default. Pass exposure= for the "
                "prediction data, or fit with exposure='<column>' so the column can be "
                "resolved from new_data."
            )
        exposure_link = None
        if exposure_to_use is not None:
            if self.link != "log":
                raise ValidationError("exposure= is only meaningful for log-link rate models.")
            exposure_link = _resolve_predict_exposure_link(aux_data, exposure_to_use)
            linear_pred = linear_pred + exposure_link

        _, offset_link, _ = _resolve_predict_offset(aux_data, offset, self._offset_spec)
        if offset_link is not None:
            linear_pred = linear_pred + offset_link

        _, complement_link = _resolve_predict_complement(
            aux_data,
            complement,
            self._complement_spec,
            exposure_to_use,
            self.link,
        )
        if complement_link is not None:
            linear_pred = linear_pred + complement_link

        return self._apply_inverse_link(linear_pred)

    def _apply_inverse_link(self, eta: np.ndarray) -> np.ndarray:
        """Apply inverse link function to linear predictor."""
        return apply_inverse_link(eta, self.link)

    def predict_contributions(
        self,
        new_data: pl.DataFrame | pl.LazyFrame,
        *,
        offset: str | np.ndarray | None = None,
        exposure: str | np.ndarray | None = None,
        complement: str | np.ndarray | GLMModel | None = None,
        group_terms: bool = True,
        include_design_columns: bool = False,
        return_format: str = "records",
        validate: bool = True,
        atol: float = 1e-9,
        rtol: float = 1e-7,
    ) -> list[dict] | pl.DataFrame:
        """Decompose each prediction into per-term contributions.

        For every row in ``new_data`` returns the additive decomposition::

            base_value + sum(contributions)  ==  linear predictor
            inverse_link(linear predictor) ==  prediction

        Spline bases, categorical dummies, target/frequency encoding columns,
        and interaction tensor products are grouped back to their source term
        so the output renders a factor-level contribution ladder.

        Parameters
        ----------
        new_data : pl.DataFrame or pl.LazyFrame
            Data to decompose. Must contain every column the model needs.
        offset : str or array-like, optional
            Override the link-scale offset used during fitting. ``str`` resolves
            a column in ``new_data``; arrays are used directly. The values are
            added to the contribution ladder verbatim; no ``log()`` is applied.
            For raw positive exposure, pass ``exposure=`` instead.
        exposure : str or array-like, optional
            Override raw positive exposure for log-link rate models. Added to
            the contribution ladder as ``log(exposure)``.
        complement : str, array-like, or GLMModel, optional
            Override the complement of credibility. When set (here or at fit
            time), ``base_value`` becomes per-row equal to
            ``link(complement_value_for_row)`` and the intercept appears as a
            regular contribution row representing the deviation.
        group_terms : bool, default True
            Group design columns back to source terms. ``False`` expands
            multi-column terms (splines, categoricals, TE/FE) into one
            contribution row per design column. Interactions remain grouped
            in either mode (per-column rows of an interaction are not
            individually meaningful).
        include_design_columns : bool, default False
            When ``group_terms=True``, attach a ``design_columns`` list to
            each grouped term containing per-column ``basis_value``,
            ``coefficient``, and ``contribution``.
        return_format : {"records", "dataframe"}, default "records"
            ``"records"``: one ``dict`` per row of ``new_data`` with a nested
            ``contributions`` list (matches the trace contract).
            ``"dataframe"``: long-format ``pl.DataFrame`` with one row per
            ``(input_row, term)`` pair. Faster for large ``N``.
        validate : bool, default True
            Verify ``base + sum(contribs) == linear predictor`` and
            ``inverse_link(linear predictor) == predict()``. Raises
            ``PredictionError`` on tolerance breach.
        atol, rtol : float
            Additivity tolerance: ``|delta| > atol + rtol * |actual|``.

        Returns
        -------
        list[dict] or pl.DataFrame
            Per-row decomposition. See module docstring for the full record
            shape.

        Raises
        ------
        PredictionError
            If the model lacks term-slot metadata (e.g. deserialized from a
            pre-feature payload), or if the additivity check fails.

        Examples
        --------
        >>> result = rs.glm_dict(
        ...     response="sale_flag",
        ...     terms={"diff_to_market": {"type": "ns", "df": 10}},
        ...     data=train,
        ...     family="binomial",
        ... ).fit()
        >>> rows = result.predict_contributions(new_data)
        >>> rows[0]["contributions"][0]["term"]
        'diff_to_market'
        """
        from rustystats.contributions import compute_contributions

        return compute_contributions(
            self,
            new_data,
            offset=offset,
            exposure=exposure,
            complement=complement,
            group_terms=group_terms,
            include_design_columns=include_design_columns,
            return_format=return_format,
            validate=validate,
            atol=atol,
            rtol=rtol,
        )

    # ------------------------------------------------------------------
    # Calibration primitives (RS-ACT-009 / PR11)
    # ------------------------------------------------------------------

    def _calibration_response_column(self) -> str:
        """Return the response column name, parsed from ``self.formula``."""
        response = self.formula.split("~", 1)[0].strip()
        if not response:
            raise ValidationError(
                "Cannot infer response column from formula; "
                "calibration / relevel requires a fitted model with a known response."
            )
        return response

    def _calibration_extract_arrays(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        exposure: str | np.ndarray | None,
        weights: str | np.ndarray | None,
    ) -> tuple[pl.DataFrame, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Materialize ``y``, ``μ = predict(data)``, exposure and weights.

        ``exposure`` defaults to the spec the model was fit with (so log-link
        rate models pass the exposure through to ``predict``). ``weights``
        defaults to ``None`` — the spec is deliberate that ``relevel``'s
        denominator is ``Σ(w·μ)``, with ``exposure`` already inside ``μ``.
        """
        response = self._calibration_response_column()
        # Collect LazyFrame; we keep all columns because we don't yet know which
        # ``by=``/``weights=`` columns the caller will reference.
        data = _collect_lazyframe(data, set())
        if response not in data.columns:
            raise ValidationError(f"response column '{response}' not found in calibration data.")
        y = data[response].to_numpy().astype(np.float64)

        exposure_to_use: str | np.ndarray | None = exposure
        if exposure_to_use is None and self._exposure_spec is not None:
            exposure_to_use = self._exposure_spec

        mu = np.asarray(self.predict(data, exposure=exposure), dtype=np.float64)

        exposure_arr: np.ndarray | None
        if exposure_to_use is None:
            exposure_arr = None
        elif isinstance(exposure_to_use, str):
            if exposure_to_use in data.columns:
                exposure_arr = data[exposure_to_use].to_numpy().astype(np.float64)
            else:
                exposure_arr = None
        else:
            exposure_arr = np.asarray(exposure_to_use, dtype=np.float64)

        if weights is None:
            weights_arr: np.ndarray | None = None
        elif isinstance(weights, str):
            if weights not in data.columns:
                raise ValidationError(f"weights column '{weights}' not found in calibration data.")
            weights_arr = data[weights].to_numpy().astype(np.float64)
        else:
            weights_arr = np.asarray(weights, dtype=np.float64)

        return data, y, mu, exposure_arr, weights_arr

    def calibration_summary(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        *,
        exposure: str | np.ndarray | None = None,
        weights: str | np.ndarray | None = None,
        by: str | list[str] | None = None,
        n_bins: int = 10,
        ranking: str = "auto",
        min_exposure: float = 0.0,
    ) -> dict[str, Any]:
        """Compute calibration diagnostics for ``self.predict(data)`` (RS-ACT-009).

        Returns the same structure as the standalone
        :func:`rustystats.calibration_summary`, but resolves response, exposure
        and weights through the fitted model. ``by=`` may be a single column
        name or a list of names; each named factor produces an aggregated
        per-level table with ``suppressed=True`` for cells below
        ``min_exposure``.

        See :mod:`rustystats.calibration` for the in-sample-optimism caveat.
        """
        from rustystats.calibration import calibration_summary as _cs

        resolved, y, mu, exposure_arr, weights_arr = self._calibration_extract_arrays(
            data, exposure, weights
        )
        by_dict: dict[str, np.ndarray] | None = None
        if by is not None:
            names = [by] if isinstance(by, str) else list(by)
            missing = [n for n in names if n not in resolved.columns]
            if missing:
                raise ValidationError(f"by= columns not present in calibration data: {missing}")
            by_dict = {n: resolved[n].to_numpy() for n in names}
        return _cs(
            y,
            mu,
            exposure=exposure_arr,
            weights=weights_arr,
            by=by_dict,
            n_bins=n_bins,
            ranking=ranking,
            min_exposure=min_exposure,
        )

    def fit_calibration(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        *,
        method: str = "global",
        exposure: str | np.ndarray | None = None,
        weights: str | np.ndarray | None = None,
        increasing: bool = True,
    ):
        """Fit a separate calibration object on this model's predictions.

        ``method="global"`` returns a :class:`~rustystats.GlobalCalibration`;
        ``method="isotonic"`` returns an :class:`~rustystats.IsotonicCalibration`.
        The returned object is *not* attached to the model and must be applied
        explicitly by the caller — raw and calibrated predictions remain
        separately accessible.
        """
        from rustystats.calibration import (
            fit_global_calibration,
            fit_isotonic_calibration,
        )

        _, y, mu, _exposure_arr, weights_arr = self._calibration_extract_arrays(
            data, exposure, weights
        )
        if method == "global":
            return fit_global_calibration(y, mu, weights=weights_arr)
        if method == "isotonic":
            return fit_isotonic_calibration(y, mu, weights=weights_arr, increasing=increasing)
        raise ValidationError(
            f"unknown calibration method {method!r}; expected 'global' or 'isotonic'."
        )

    def relevel(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        *,
        exposure: str | np.ndarray | None = None,
        weights: str | np.ndarray | None = None,
        inplace: bool = False,
    ) -> GLMModel:
        """Apply a global multiplicative calibration as a log-link intercept shift.

        For a log-link GLM with predictions ``μ`` on ``data``, computes the
        calibration factor

        .. math:: c = \\frac{\\sum_i w_i\\, y_i}{\\sum_i w_i\\, \\mu_i}

        and updates the intercept by ``+log(c)``. Every other coefficient is
        bit-identical, so multiplicative relativities (``exp(β_j)``) are
        preserved — the rate table keeps its shape and only the model level
        changes.

        Notes
        -----
        ``exposure=`` is only used by the prediction that builds ``μ``; it is
        *never* a denominator in ``c``. Under a log link, exposure already
        lives inside ``μ`` via the ``log(exposure)`` offset.

        Calibrating on the same rows used to fit the model overstates the
        calibration quality — see :mod:`rustystats.calibration`.

        The intercept's inference is recentred on the shifted estimate and its
        standard error is inflated by an approximate ``Var(log c)`` (delta
        method, ``φ·Σ(w·V(μ)) / (Σ w·y)²``), accumulated in
        :attr:`intercept_delta_var`, so its CI/z/p are not falsely tight. Every
        other coefficient and all relativities are unchanged. The variance
        treats the calibration actuals as independent of the fit — exact for an
        out-of-sample calibration fold (the recommended workflow above) and
        optimistic if you calibrate in-sample.

        Returns a new :class:`GLMModel` by default; pass ``inplace=True`` to
        mutate this object instead.
        """
        from rustystats.calibration import fit_global_calibration

        if self.link != "log":
            raise ValidationError(
                f"relevel(method='global') is only supported for log-link models, "
                f"got link='{self.link}'. Use fit_calibration(method='global') to "
                f"attach a multiplicative calibration object instead."
            )
        if not self.feature_names or self.feature_names[0] != "Intercept":
            raise ValidationError(
                "relevel() requires a model fitted with an intercept "
                "(first feature must be 'Intercept')."
            )

        _, y, mu, _exposure_arr, weights_arr = self._calibration_extract_arrays(
            data, exposure, weights
        )
        cal = fit_global_calibration(y, mu, weights=weights_arr)
        c = cal.factor
        if not np.isfinite(c) or c <= 0.0:
            raise ValidationError(
                f"relevel factor c={c!r} is not finite/positive; cannot apply log-shift."
            )
        log_shift = float(np.log(c))
        # RS-ACT-009 backlog #4: fold the calibration factor's own (delta-method)
        # variance into the intercept SE so its CI/z/p are not falsely tight.
        log_factor_var = self._relevel_log_factor_variance(y, mu, weights_arr)

        target = self if inplace else copy.copy(self)
        intercept_before = float(target.params[0])
        intercept_after = float(intercept_before + log_shift)
        # Lists are shared by ``copy.copy``; rebuild on the target so the
        # original's history is not mutated.
        history = list(target._relevel_history)
        history.append(
            {
                "factor": float(c),
                "original_intercept": intercept_before,
                "new_intercept": intercept_after,
                "log_shift": log_shift,
                "log_factor_var": log_factor_var,
                "n_obs": int(y.shape[0]),
                "total_weight": float(
                    np.sum(weights_arr) if weights_arr is not None else y.shape[0]
                ),
            }
        )
        target._relevel_history = history
        target._intercept_delta = target._intercept_delta + log_shift
        target._intercept_delta_var = target._intercept_delta_var + log_factor_var
        return target

    def to_pmml(
        self,
        path: str | None = None,
        n_grid_points: int = 200,
    ) -> str:
        """
        Export the fitted model to PMML 4.4 XML.

        Spline basis functions are collapsed to piecewise-linear
        ``NormContinuous`` derived fields.  Categorical, target-encoding,
        frequency-encoding, expression, and interaction terms are all
        supported.

        Parameters
        ----------
        path : str, optional
            If given, write the PMML XML to this file path.
        n_grid_points : int, default 200
            Grid resolution for piecewise-linear spline approximation.

        Returns
        -------
        str
            The PMML XML document as a string.
        """
        if self._input_transforms:
            names = [spec["name"] for spec in self._input_transforms]
            raise ValidationError(
                "PMML raw-data export does not yet support input_transforms; "
                f"unsupported transform(s): {names}."
            )
        from rustystats.export_pmml import to_pmml

        return to_pmml(self, path=path, n_grid_points=n_grid_points)

    def to_onnx(
        self,
        path: str | None = None,
        n_grid_points: int = 200,
        mode: str = "scoring",
    ) -> bytes:
        """
        Export the fitted model to ONNX format.

        Protobuf serialization is implemented from scratch in Rust —
        no external dependencies are required.

        Parameters
        ----------
        path : str, optional
            If given, write the ONNX model to this file path.
        n_grid_points : int, default 200
            Grid resolution for piecewise-linear spline approximation
            (only used in ``"full"`` mode).
        mode : {"scoring", "full"}, default "scoring"
            * ``"scoring"`` — input is a pre-built design matrix
              ``X (batch, n_features)`` without intercept column.
            * ``"full"`` — input is raw feature values; preprocessing
              (one-hot, splines, TE/FE) is embedded in the graph.
              Categorical variables are passed as integer codes.

        Returns
        -------
        bytes
            Raw ONNX protobuf bytes.  Load with
            ``onnxruntime.InferenceSession(onnx_bytes)`` or write to disk.
        """
        if mode == "full" and self._input_transforms:
            names = [spec["name"] for spec in self._input_transforms]
            raise ValidationError(
                "ONNX full raw-data export does not yet support input_transforms; "
                f"unsupported transform(s): {names}."
            )
        from rustystats.export_onnx import to_onnx

        return to_onnx(self, path=path, n_grid_points=n_grid_points, mode=mode)

    def to_rate_tables(
        self,
        path: str | None = None,
        *,
        format: str = "dict",
        style: str = "resolved",
        deployment: bool = True,
        spline_strategy: str = "unsupported",
        spline_grids: dict[str, list[float]] | None = None,
        spline_interpolation: str = "linear",
        spline_extrapolation: str = "clip",
        include_components: bool = False,
    ) -> dict[str, Any]:
        """Export the fitted model as concise resolved rate tables."""
        from rustystats.rate_tables import to_rate_tables

        return to_rate_tables(
            self,
            path=path,
            format=format,
            style=style,
            deployment=deployment,
            spline_strategy=spline_strategy,
            spline_grids=spline_grids,
            spline_interpolation=spline_interpolation,
            spline_extrapolation=spline_extrapolation,
            include_components=include_components,
        )

    def to_bytes(self) -> bytes:
        """
        Serialize the fitted model to bytes for storage or transfer.

        The serialized model can be loaded with `GLMModel.from_bytes()`.
        All state needed for prediction is preserved, including:
        - Coefficients and feature names
        - Categorical encoding levels
        - Spline knot positions
        - Target encoding statistics
        - Deterministic input transforms
        - Family parameter metadata such as Negative Binomial theta
        - Relevel intercept shifts and metadata

        Returns
        -------
        bytes
            Serialized model as bytes.

        Examples
        --------
        >>> result = rs.glm_dict(response="y", terms={"x1": {"type": "linear"}, "cat": {"type": "categorical"}}, data=data, family="poisson").fit()
        >>> model_bytes = result.to_bytes()
        >>>
        >>> # Save to file
        >>> with open("model.bin", "wb") as f:
        ...     f.write(model_bytes)
        >>>
        >>> # Load later
        >>> with open("model.bin", "rb") as f:
        ...     loaded = rs.GLMModel.from_bytes(f.read())
        >>> predictions = loaded.predict(new_data)
        """
        import pickle

        # Extract state from the Rust result object
        # NOTE: We intentionally exclude fittedvalues and linear_predictor
        # as they are large arrays not needed for prediction (can be ~5MB each)
        result_state = {
            "params": np.asarray(self._result.params),
            "deviance": self._result.deviance,
            "iterations": self._result.iterations,
            "converged": self._result.converged,
            "nobs": self._result.nobs,
            "df_resid": self._result.df_resid,
            "df_model": self._result.df_model,
            "alpha": self._result.alpha,
            "l1_ratio": self._result.l1_ratio,
            "is_regularized": self._result.is_regularized,
            "penalty_type": self._result.penalty_type,
            "theta": self.__dict__.get("theta"),
            "theta_metadata": self.__dict__.get("theta_metadata"),
            "inference_status": self.__dict__.get("inference_status"),
            "optimizer_route": self.__dict__.get("optimizer_route"),
            "solver_status": self.__dict__.get("solver_status"),
            "step_halving_used": self.__dict__.get("step_halving_used"),
            "regularization_standardized": self.__dict__.get("_regularization_standardized", False),
        }

        # Extract builder state for prediction
        builder_state = None
        if self._builder is not None:
            builder_state = {
                "parsed_formula": self._builder._parsed_formula,
                "cat_encoding_cache": self._builder._cat_encoding_cache,
                "fitted_splines": self._builder._fitted_splines,
                "te_stats": getattr(self._builder, "_te_stats", {}),
                "fe_stats": getattr(self._builder, "_fe_stats", {}),
                "dtype": self._builder.dtype,
                "term_slots": getattr(self._builder, "_term_slots", []),
            }

        # Record basis implementation for each spline term
        spline_basis_impl = {}
        if self._builder is not None and hasattr(self._builder, "_fitted_splines"):
            for var_name, spline_term in self._builder._fitted_splines.items():
                if hasattr(spline_term, "spline_type"):
                    uses_ispline = spline_term.spline_type == "ms" or (
                        spline_term.spline_type == "bs"
                        and getattr(spline_term, "monotonicity", None) is not None
                    )
                    spline_basis_impl[var_name] = {
                        "spline_type": spline_term.spline_type,
                        "basis": "ispline" if uses_ispline else spline_term.spline_type,
                        "monotonicity": getattr(spline_term, "monotonicity", None),
                    }

        array_exposure_requires_override = bool(
            getattr(self, "_array_exposure_requires_prediction_override", False)
        )
        serializable_exposure_spec = (
            None if array_exposure_requires_override else self._exposure_spec
        )

        # Persist a prior-weights column spec so deserialized models keep
        # weighted diagnostics. Array weights, like array exposure, are fit-time
        # data the caller must re-supply, so only column-name specs are stored.
        weights_spec = getattr(self, "_weights_spec", None)
        serializable_weights_spec = weights_spec if isinstance(weights_spec, str) else None

        state = {
            "schema_version": self._SCHEMA_VERSION,
            "result_state": result_state,
            "feature_names": self.feature_names,
            "formula": self.formula,
            "family": self.family,
            "link": self.link,
            "builder_state": builder_state,
            "offset_spec": self._offset_spec,
            "weights_spec": serializable_weights_spec,
            # Raw array exposure is fit-time data, not reusable prediction
            # metadata. Persist only column specs; array-exposure models require
            # callers to supply prediction-time exposure explicitly after load.
            "exposure_spec": serializable_exposure_spec,
            "array_exposure_requires_prediction_override": array_exposure_requires_override,
            "smooth_results": self._smooth_results,
            "total_edf": self._total_edf,
            "gcv": self._gcv,
            "terms_dict": self._terms_dict,
            "interactions_spec": self._interactions_spec,
            "input_transforms": self._input_transforms,
            "complement_spec": self._complement_spec,
            "intercept_delta": float(self._intercept_delta),
            "intercept_delta_var": float(self._intercept_delta_var),
            "relevel_history": self.relevel_history,
            "basis_impl": spline_basis_impl,
        }

        return pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def from_bytes(cls, data: bytes) -> GLMModel:
        """
        Load a fitted model from bytes.

        Parameters
        ----------
        data : bytes
            Serialized model bytes from `to_bytes()`.

        Returns
        -------
        GLMModel
            Reconstructed fitted model ready for prediction.

        Examples
        --------
        >>> # Load from file
        >>> with open("model.bin", "rb") as f:
        ...     result = rs.GLMModel.from_bytes(f.read())
        >>>
        >>> # Make predictions
        >>> predictions = result.predict(new_data)
        """
        import pickle

        state = pickle.loads(data)

        # Fail loud on a schema this build cannot read, rather than silently
        # loading it and mis-handling fields (e.g. the pre-v4 exposure layout).
        sv = state.get("schema_version")
        if sv != cls._SCHEMA_VERSION:
            raise ValidationError(
                f"Cannot load model: serialized schema_version {sv!r} is not supported by "
                f"this RustyStats build (schema_version {cls._SCHEMA_VERSION}). Re-serialize "
                "the model with the current version."
            )

        result_state = state["result_state"]

        # Create a minimal result object that supports prediction
        result = _DeserializedResult(
            params=result_state["params"],
            deviance=result_state["deviance"],
            iterations=result_state["iterations"],
            converged=result_state["converged"],
            nobs=result_state["nobs"],
            df_resid=result_state["df_resid"],
            df_model=result_state["df_model"],
            alpha=result_state["alpha"],
            l1_ratio=result_state["l1_ratio"],
            is_regularized=result_state["is_regularized"],
            penalty_type=result_state["penalty_type"],
        )

        # Reconstruct builder if it was saved
        builder = None
        if state["builder_state"] is not None:
            builder = _DeserializedBuilder(state["builder_state"])

        offset_spec = state.get("offset_spec")
        exposure_spec = state.get("exposure_spec")
        array_exposure_requires_prediction_override = bool(
            state.get("array_exposure_requires_prediction_override", False)
        )

        model = cls(
            result=result,
            feature_names=state["feature_names"],
            formula=state["formula"],
            family=state["family"],
            link=state["link"],
            builder=builder,
            offset_spec=offset_spec,
            exposure_spec=exposure_spec,
            regularization_path_info=None,
            smooth_results=state["smooth_results"],
            total_edf=state["total_edf"],
            gcv=state["gcv"],
            terms_dict=state.get("terms_dict"),
            interactions_spec=state.get("interactions_spec"),
            input_transforms=state.get("input_transforms", []),
            complement_spec=state.get("complement_spec"),
            array_exposure_requires_prediction_override=array_exposure_requires_prediction_override,
            regularization_standardized=bool(
                result_state.get("regularization_standardized", False)
            ),
        )
        model.theta = result_state.get("theta")
        model.theta_metadata = result_state.get("theta_metadata")
        model.inference_status = result_state.get("inference_status")
        model.optimizer_route = result_state.get("optimizer_route")
        model.solver_status = result_state.get("solver_status")
        model.step_halving_used = result_state.get("step_halving_used")
        model._intercept_delta = float(state.get("intercept_delta", 0.0))
        model._intercept_delta_var = float(state.get("intercept_delta_var", 0.0))
        model._relevel_history = [dict(entry) for entry in state.get("relevel_history", [])]
        model._weights_spec = state.get("weights_spec")
        return model

    def __repr__(self) -> str:
        return (
            f"<GLMModel: {self.family} family, "
            f"{len(self.params)} parameters, "
            f"deviance={self.deviance:.2f}>"
        )


# =============================================================================
# Dict-based API
# =============================================================================

from rustystats.constants import (
    DEFAULT_N_PERMUTATIONS,
    DEFAULT_PRIOR_WEIGHT,
    DEFAULT_SPLINE_DEGREE,
)
from rustystats.interactions import (
    CategoricalTermSpec,
    ConstraintTermSpec,
    FrequencyEncodingTermSpec,
    IdentityTermSpec,
    InteractionTerm,
    ParsedFormula,
    TargetEncodingTermSpec,
)
from rustystats.splines import SplineTerm


def _validate_explicit_knots(
    var_name: str,
    knots: list | tuple,
    spec: dict[str, Any],
) -> list[float]:
    """Validate explicit knots and return as list[float].

    Raises ValidationError if knots are empty, unsorted, duplicated,
    or combined with df/k.
    """
    if spec.get("k") is not None or spec.get("df") is not None:
        raise ValidationError(
            f"Cannot specify both 'knots' and 'df'/'k' for '{var_name}'. "
            "Use either explicit knots or automatic knot placement, not both."
        )
    knots_list = list(knots)
    if len(knots_list) == 0:
        raise ValidationError(f"'knots' must be a non-empty sequence for '{var_name}'.")
    if knots_list != sorted(knots_list):
        raise ValidationError(
            f"'knots' must be sorted in ascending order for '{var_name}'. Got: {knots_list}"
        )
    if len(set(knots_list)) != len(knots_list):
        raise ValidationError(
            f"'knots' must contain unique values for '{var_name}'. Got duplicates in: {knots_list}"
        )
    return [float(v) for v in knots_list]


def _parse_term_spec(
    var_name: str,
    spec: dict[str, Any],
    categorical_vars: set[str],
    main_effects: list[str],
    spline_terms: list[SplineTerm],
    target_encoding_terms: list[TargetEncodingTermSpec],
    identity_terms: list[IdentityTermSpec],
    categorical_terms: list[CategoricalTermSpec],
    constraint_terms: list[ConstraintTermSpec],
    frequency_encoding_terms: list | None = None,
) -> None:
    """Parse a single term specification and add to appropriate lists."""
    # Valid keys for each term type
    VALID_KEYS = {
        "linear": {"type", "monotonicity"},
        "categorical": {"type", "levels"},
        "bs": {"type", "df", "k", "degree", "monotonicity", "knots", "boundary_knots"},
        "ns": {"type", "df", "k", "knots", "boundary_knots"},
        "ms": {"type", "df", "k", "degree", "monotonicity", "knots", "boundary_knots"},
        "target_encoding": {"type", "prior_weight", "n_permutations", "variable"},
        "frequency_encoding": {"type", "variable"},
        "expression": {"type", "expr", "monotonicity"},
    }

    term_type = spec.get("type", "linear")

    # Validate keys
    valid_keys = VALID_KEYS.get(term_type, set())
    unknown_keys = set(spec.keys()) - valid_keys
    if unknown_keys:
        # Check for common typos
        typo_suggestions = {
            "monoticity": "monotonicity",
            "montonicity": "monotonicity",
            "increaing": "increasing",
            "decreaing": "decreasing",
        }
        suggestions = []
        for key in unknown_keys:
            if key in typo_suggestions:
                suggestions.append(f"'{key}' (did you mean '{typo_suggestions[key]}'?)")
            else:
                suggestions.append(f"'{key}'")
        raise ValidationError(
            f"Unknown key(s) in term spec for '{var_name}': {', '.join(suggestions)}. "
            f"Valid keys for type='{term_type}' are: {sorted(valid_keys)}"
        )

    monotonicity = spec.get("monotonicity")  # "increasing" or "decreasing"

    if term_type == "linear":
        if monotonicity:
            # Constrained linear term
            constraint = "pos" if monotonicity == "increasing" else "neg"
            constraint_terms.append(
                ConstraintTermSpec(
                    var_name=var_name,
                    constraint=constraint,
                )
            )
        else:
            main_effects.append(var_name)

    elif term_type == "categorical":
        categorical_vars.add(var_name)
        levels = spec.get("levels")
        if levels:
            # Specific levels only
            categorical_terms.append(
                CategoricalTermSpec(
                    var_name=var_name,
                    levels=levels,
                )
            )
        else:
            main_effects.append(var_name)

    elif term_type == "bs":
        explicit_knots = spec.get("knots")
        user_boundary_knots = spec.get("boundary_knots")
        if explicit_knots is not None:
            knots_list = _validate_explicit_knots(var_name, explicit_knots, spec)
            degree = spec.get("degree", DEFAULT_SPLINE_DEGREE)
            implied_df = len(knots_list) + degree
            bk_tuple = tuple(user_boundary_knots) if user_boundary_knots is not None else None
            term = SplineTerm(
                var_name=var_name,
                spline_type="bs",
                df=implied_df,
                degree=degree,
                boundary_knots=bk_tuple,
                monotonicity=monotonicity,
            )
            term._computed_internal_knots = knots_list
            term._is_smooth = False
            if monotonicity:
                term._monotonic = True
            spline_terms.append(term)
        else:
            # Default to penalized smooth (k=DEFAULT_SPLINE_DF) if neither df nor k specified
            k = spec.get("k")
            df = spec.get("df")
            if df is None and k is None:
                df = DEFAULT_SPLINE_DF  # Default: penalized smooth
                is_penalized = True
            elif k is not None:
                df = k
                is_penalized = True
            else:
                is_penalized = False
            degree = spec.get("degree", DEFAULT_SPLINE_DEGREE)
            bk_tuple = tuple(user_boundary_knots) if user_boundary_knots is not None else None
            term = SplineTerm(
                var_name=var_name,
                spline_type="bs",
                df=df,
                degree=degree,
                boundary_knots=bk_tuple,
                monotonicity=monotonicity,
            )
            if is_penalized:
                term._is_smooth = True
            if monotonicity:
                term._monotonic = True
            spline_terms.append(term)

    elif term_type == "ns":
        if monotonicity:
            raise ValidationError(
                "Monotonicity constraints are not supported for natural splines (ns). "
                "Use type='bs' with monotonicity parameter instead for monotonic effects."
            )
        explicit_knots = spec.get("knots")
        user_boundary_knots = spec.get("boundary_knots")
        if explicit_knots is not None:
            knots_list = _validate_explicit_knots(var_name, explicit_knots, spec)
            implied_df = len(knots_list)
            bk_tuple = tuple(user_boundary_knots) if user_boundary_knots is not None else None
            term = SplineTerm(
                var_name=var_name,
                spline_type="ns",
                df=implied_df,
                boundary_knots=bk_tuple,
            )
            term._computed_internal_knots = knots_list
            term._is_smooth = False
            spline_terms.append(term)
        else:
            # Default to penalized smooth (k=DEFAULT_SPLINE_DF) if neither df nor k specified
            k = spec.get("k")
            df = spec.get("df")
            if df is None and k is None:
                df = DEFAULT_SPLINE_DF  # Default: penalized smooth
                is_penalized = True
            elif k is not None:
                df = k
                is_penalized = True
            else:
                is_penalized = False
            bk_tuple = tuple(user_boundary_knots) if user_boundary_knots is not None else None
            term = SplineTerm(
                var_name=var_name,
                spline_type="ns",
                df=df,
                boundary_knots=bk_tuple,
            )
            if is_penalized:
                term._is_smooth = True
            spline_terms.append(term)

    elif term_type == "ms":
        # Monotonic spline — uses I-spline basis via SplineTerm with spline_type="ms"
        # Default monotonicity to "increasing" if not specified
        mono = monotonicity or "increasing"
        explicit_knots = spec.get("knots")
        user_boundary_knots = spec.get("boundary_knots")
        if explicit_knots is not None:
            knots_list = _validate_explicit_knots(var_name, explicit_knots, spec)
            degree = spec.get("degree", DEFAULT_SPLINE_DEGREE)
            implied_df = len(knots_list) + degree
            bk_tuple = tuple(user_boundary_knots) if user_boundary_knots is not None else None
            term = SplineTerm(
                var_name=var_name,
                spline_type="ms",
                df=implied_df,
                degree=degree,
                boundary_knots=bk_tuple,
                monotonicity=mono,
            )
            term._computed_internal_knots = knots_list
            term._is_smooth = False
            term._monotonic = True
            spline_terms.append(term)
        else:
            k = spec.get("k")
            df = spec.get("df")
            if df is None and k is None:
                df = DEFAULT_SPLINE_DF
                is_penalized = True
            elif k is not None:
                df = k
                is_penalized = True
            else:
                is_penalized = False
            degree = spec.get("degree", DEFAULT_SPLINE_DEGREE)
            bk_tuple = tuple(user_boundary_knots) if user_boundary_knots is not None else None
            term = SplineTerm(
                var_name=var_name,
                spline_type="ms",
                df=df,
                degree=degree,
                boundary_knots=bk_tuple,
                monotonicity=mono,
            )
            if is_penalized:
                term._is_smooth = True
            term._monotonic = True
            spline_terms.append(term)

    elif term_type == "target_encoding":
        prior_weight = spec.get("prior_weight", DEFAULT_PRIOR_WEIGHT)
        n_permutations = spec.get("n_permutations", DEFAULT_N_PERMUTATIONS)
        # Single variable TE - use 'variable' key if provided
        # For TE interactions, use the interactions list with target_encoding: True
        actual_var = spec.get("variable", var_name)
        existing_te_vars = {te.var_name for te in target_encoding_terms}
        if actual_var not in existing_te_vars:
            target_encoding_terms.append(
                TargetEncodingTermSpec(
                    var_name=actual_var,
                    prior_weight=prior_weight,
                    n_permutations=n_permutations,
                )
            )

    elif term_type == "frequency_encoding":
        from rustystats.interactions import FrequencyEncodingTermSpec as FETermSpec

        if frequency_encoding_terms is None:
            raise ValidationError(
                f"frequency_encoding type not supported in this context. "
                f"Use formula string 'FE({var_name})' instead."
            )
        # Use 'variable' key if provided, otherwise use the dict key
        actual_var = spec.get("variable", var_name)
        frequency_encoding_terms.append(FETermSpec(var_name=actual_var))

    elif term_type == "expression":
        expr = spec.get("expr", var_name)
        if monotonicity:
            constraint = "pos" if monotonicity == "increasing" else "neg"
            constraint_terms.append(
                ConstraintTermSpec(
                    var_name=f"I({expr})",
                    constraint=constraint,
                )
            )
        else:
            identity_terms.append(IdentityTermSpec(expression=expr))

    else:
        raise ValidationError(f"Unknown term type: {term_type}")


def _parse_interaction_spec(
    interaction: dict[str, Any],
    interactions: list[InteractionTerm],
    categorical_vars: set[str],
    main_effects: list[str],
    spline_terms: list[SplineTerm],
    target_encoding_terms: list[TargetEncodingTermSpec],
    identity_terms: list[IdentityTermSpec],
    categorical_terms: list[CategoricalTermSpec],
    constraint_terms: list[ConstraintTermSpec],
    frequency_encoding_terms: list | None = None,
) -> None:
    """Parse an interaction specification.

    Supports two modes:
    1. Standard interaction: creates product terms (cat×cat, cat×cont, etc.)
    2. Encoding interaction: combines variables into single encoded value
       - target_encoding: True → TE(var1:var2:...)
       - frequency_encoding: True → FE(var1:var2:...)
    """
    # Reserved keys (not variable specs)
    RESERVED_KEYS = {
        "include_main",
        "target_encoding",
        "frequency_encoding",
        "prior_weight",
        "n_permutations",
    }

    include_main = interaction.get("include_main", False)
    is_te_interaction = interaction.get("target_encoding", False)
    is_fe_interaction = interaction.get("frequency_encoding", False)

    if is_te_interaction and is_fe_interaction:
        raise ValidationError(
            "Cannot specify both target_encoding and frequency_encoding for same interaction"
        )

    # Extract variable specs (everything except reserved keys)
    var_specs = {k: v for k, v in interaction.items() if k not in RESERVED_KEYS}

    if len(var_specs) < 2:
        raise ValidationError("Interaction must have at least 2 variables")

    # Helper: track categorical vars and optionally add main effects
    def _process_encoding_interaction() -> None:
        for var_name, spec in var_specs.items():
            if spec.get("type", "categorical") == "categorical":
                categorical_vars.add(var_name)

        if include_main:
            for var_name, spec in var_specs.items():
                _parse_term_spec(
                    var_name,
                    spec,
                    categorical_vars,
                    main_effects,
                    spline_terms,
                    target_encoding_terms,
                    identity_terms,
                    categorical_terms,
                    constraint_terms,
                    frequency_encoding_terms,
                )

    # Handle TE interaction: TE(var1:var2:...)
    if is_te_interaction:
        interaction_vars = list(var_specs.keys())
        target_encoding_terms.append(
            TargetEncodingTermSpec(
                var_name=":".join(interaction_vars),
                prior_weight=interaction.get("prior_weight", DEFAULT_PRIOR_WEIGHT),
                n_permutations=interaction.get("n_permutations", DEFAULT_N_PERMUTATIONS),
                interaction_vars=interaction_vars,
            )
        )
        _process_encoding_interaction()
        return

    # Handle FE interaction: FE(var1:var2:...)
    if is_fe_interaction:
        if frequency_encoding_terms is None:
            raise ValidationError("frequency_encoding interaction not supported in this context")
        interaction_vars = list(var_specs.keys())
        frequency_encoding_terms.append(
            FrequencyEncodingTermSpec(
                var_name=":".join(interaction_vars),
                interaction_vars=interaction_vars,
            )
        )
        _process_encoding_interaction()
        return

    # Standard interaction: product terms
    # Determine which factors are categorical, splines, or TE
    cat_factors = set()
    linear_factors = set()  # Factors explicitly typed as linear (no spline expansion)
    spline_factors = []
    te_factor_names = {}  # Maps original name -> TE(name) format

    for var_name, spec in var_specs.items():
        term_type = spec.get("type", "linear")

        if term_type == "linear":
            linear_factors.add(var_name)
        elif term_type == "categorical":
            cat_factors.add(var_name)
            categorical_vars.add(var_name)
        elif term_type in ("bs", "ns", "ms", "s"):
            explicit_knots = spec.get("knots")
            user_boundary_knots = spec.get("boundary_knots")
            # For s() smooth terms, use k parameter; for bs/ns/ms use df
            if explicit_knots is not None:
                knots_list = _validate_explicit_knots(var_name, explicit_knots, spec)
                degree = spec.get("degree", DEFAULT_SPLINE_DEGREE)
                spline_type_out = "bs" if term_type == "s" else term_type
                if spline_type_out in ("bs", "ms"):
                    implied_df = len(knots_list) + degree
                else:
                    implied_df = len(knots_list)
                bk_tuple = tuple(user_boundary_knots) if user_boundary_knots is not None else None
                monotonicity = spec.get("monotonicity")
                spline = SplineTerm(
                    var_name=var_name,
                    spline_type=spline_type_out,
                    df=implied_df,
                    degree=degree,
                    boundary_knots=bk_tuple,
                    monotonicity=monotonicity,
                )
                spline._computed_internal_knots = knots_list
                spline._is_smooth = False
            else:
                if term_type == "s":
                    df = spec.get("k", DEFAULT_SPLINE_DF)
                else:
                    df = spec.get("df", 5 if term_type in ("bs", "ms") else 4)
                degree = spec.get("degree", DEFAULT_SPLINE_DEGREE)
                monotonicity = spec.get("monotonicity")
                # For ms type, default monotonicity to "increasing"
                if term_type == "ms" and monotonicity is None:
                    monotonicity = "increasing"
                # Use unified bs with monotonicity parameter
                spline_type_out = "bs" if term_type == "s" else term_type
                bk_tuple = tuple(user_boundary_knots) if user_boundary_knots is not None else None
                spline = SplineTerm(
                    var_name=var_name,
                    spline_type=spline_type_out,
                    df=df,
                    degree=degree,
                    boundary_knots=bk_tuple,
                    monotonicity=monotonicity,
                )
                # Mark s() terms as smooth for penalized fitting
                if term_type == "s":
                    spline._is_smooth = True
                    if monotonicity:
                        spline._smooth_monotonicity = monotonicity
            spline_factors.append((var_name, spline))
        elif term_type == "target_encoding":
            prior_weight = spec.get("prior_weight", DEFAULT_PRIOR_WEIGHT)
            te_factor_names[var_name] = f"TE({var_name})"
            # TE in interaction - add to TE terms so encoding is available (if not already present)
            existing_te_vars = {te.var_name for te in target_encoding_terms}
            if var_name not in existing_te_vars:
                target_encoding_terms.append(
                    TargetEncodingTermSpec(
                        var_name=var_name,
                        prior_weight=prior_weight,
                    )
                )

    # Build factors list, renaming TE factors to TE(name) format
    factors = [te_factor_names.get(k, k) for k in var_specs]

    # Build interaction term - categorical_flags is a bool for each factor
    categorical_flags = [f in cat_factors for f in factors]

    interaction_term = InteractionTerm(
        factors=factors,
        categorical_flags=categorical_flags,
        force_linear=linear_factors if linear_factors else None,
    )
    interactions.append(interaction_term)

    # Add main effects if requested
    if include_main:
        for var_name, spec in var_specs.items():
            _parse_term_spec(
                var_name,
                spec,
                categorical_vars,
                main_effects,
                spline_terms,
                target_encoding_terms,
                identity_terms,
                categorical_terms,
                constraint_terms,
            )


def dict_to_parsed_formula(
    response: str,
    terms: dict[str, dict[str, Any]],
    interactions: list[dict[str, Any]] | None = None,
    intercept: bool = True,
) -> ParsedFormula:
    """
    Convert dict specification to ParsedFormula.

    Parameters
    ----------
    response : str
        Name of the response variable
    terms : dict
        Dictionary mapping variable names to term specifications
    interactions : list of dict, optional
        List of interaction specifications
    intercept : bool, default=True
        Whether to include an intercept

    Returns
    -------
    ParsedFormula
        Parsed formula object compatible with build_design_matrix
    """

    categorical_vars: set[str] = set()
    main_effects: list[str] = []
    spline_terms_list: list[SplineTerm] = []
    target_encoding_terms_list: list[TargetEncodingTermSpec] = []
    frequency_encoding_terms_list: list[FrequencyEncodingTermSpec] = []
    identity_terms_list: list[IdentityTermSpec] = []
    categorical_terms_list: list[CategoricalTermSpec] = []
    constraint_terms_list: list[ConstraintTermSpec] = []
    interaction_terms_list: list[InteractionTerm] = []

    # Parse main terms
    for var_name, spec in terms.items():
        _parse_term_spec(
            var_name,
            spec,
            categorical_vars,
            main_effects,
            spline_terms_list,
            target_encoding_terms_list,
            identity_terms_list,
            categorical_terms_list,
            constraint_terms_list,
            frequency_encoding_terms_list,
        )

    # Parse interactions
    if interactions:
        for interaction in interactions:
            _parse_interaction_spec(
                interaction,
                interaction_terms_list,
                categorical_vars,
                main_effects,
                spline_terms_list,
                target_encoding_terms_list,
                identity_terms_list,
                categorical_terms_list,
                constraint_terms_list,
                frequency_encoding_terms_list,
            )

    return ParsedFormula(
        response=response,
        main_effects=main_effects,
        interactions=interaction_terms_list,
        categorical_vars=categorical_vars,
        spline_terms=spline_terms_list,
        target_encoding_terms=target_encoding_terms_list,
        frequency_encoding_terms=frequency_encoding_terms_list,
        identity_terms=identity_terms_list,
        categorical_terms=categorical_terms_list,
        constraint_terms=constraint_terms_list,
        has_intercept=intercept,
    )


class FormulaGLMDict(_GLMBase):
    """
    GLM model with dict-based specification.

    Alternative to formula strings for programmatic model building.
    """

    def __init__(
        self,
        response: str,
        terms: dict[str, dict[str, Any]],
        data: pl.DataFrame,
        interactions: list[dict[str, Any]] | None = None,
        intercept: bool = True,
        family: str = "gaussian",
        link: str | None = None,
        var_power: float = 1.5,
        theta: float | str | None = None,
        exposure: str | np.ndarray | None = None,
        offset: str | np.ndarray | None = None,
        weights: str | np.ndarray | None = None,
        seed: int | None = None,
        complement: str | np.ndarray | GLMModel | None = None,
        input_transforms: list[dict[str, Any]] | None = None,
        allow_extended_tweedie: bool = False,
    ):
        self.response = response
        self.terms = terms
        self.interactions_spec = interactions
        self.intercept = intercept
        self._input_transforms = validate_input_transforms(
            input_transforms, data_schema=dict(data.schema)
        )
        self._compiled_input_transforms = compile_input_transforms(
            self._input_transforms,
            assume_validated=True,
        )
        if self._compiled_input_transforms:
            data = apply_input_transforms(data, self._compiled_input_transforms)
        self._owned_transformed_data = data if self._compiled_input_transforms else None
        # Store weak reference to data to allow garbage collection. When input
        # transforms are present, this is the transformed frame used for fitting.
        self._data_ref = weakref.ref(data)
        raw_family_base, _embedded_params = _split_embedded_family_param(family)
        if raw_family_base.lower() == "tweedie":
            family_base, tweedie_p = _parse_embedded_numeric_param(family, "p")
            embedded_theta = None
        elif raw_family_base.lower() in NEGBINOMIAL_ALIASES:
            family_base, embedded_theta = _parse_embedded_numeric_param(family, "theta")
            tweedie_p = None
        else:
            family_base = raw_family_base
            tweedie_p = None
            embedded_theta = None
        self.family = family_base.lower()
        self.link = link
        if tweedie_p is not None:
            if var_power != 1.5 and not np.isclose(var_power, tweedie_p):
                raise ValidationError(
                    f"Conflicting Tweedie variance powers: family={family!r} "
                    f"but var_power={var_power}."
                )
            var_power = tweedie_p
        if embedded_theta is not None:
            if (
                theta is not None
                and theta != "estimate"
                and not np.isclose(float(theta), embedded_theta)
            ):
                raise ValidationError(
                    f"Conflicting Negative Binomial theta values: family={family!r} "
                    f"but theta={theta}."
                )
            theta = embedded_theta
        self.var_power = var_power
        self.theta = theta
        # RS-ACT-006: opt-in flag for the extended Tweedie regimes (p outside
        # the default compound Poisson-Gamma interior 1 < p < 2). Defaults off;
        # the per-regime support rules are enforced in validate_glm_inputs.
        self.allow_extended_tweedie = bool(allow_extended_tweedie)
        self._exposure_spec = exposure
        self._offset_spec = offset
        self._weights_spec = weights
        self._seed = seed
        self._complement_spec = complement if isinstance(complement, str | GLMModel) else None
        self._complement_values = None  # Set by _process_complement

        # Build formula string for compatibility (used in results/diagnostics)
        self.formula = self._build_formula_string()

        # Convert dict to ParsedFormula
        parsed = dict_to_parsed_formula(
            response=response,
            terms=terms,
            interactions=interactions,
            intercept=intercept,
        )

        # RS-ACT-002: raw exposure (the logged rate denominator) and the
        # link-scale offset are independent — `exposure=` is logged into the fit
        # offset, `offset=` is added verbatim.
        raw_exposure = self._get_raw_exposure(exposure)
        if exposure is not None and not self._uses_log_link():
            raise ValidationError(
                "`exposure=` is only supported for log-link rate models "
                f"(got family={self.family!r}, link={self.link!r}). Use "
                "`offset=` for a link-scale adjustment on other links."
            )
        if (
            exposure is None
            and offset is not None
            and self._uses_log_link()
            and self._has_target_encoding()
        ):
            # A link-scale `offset` is not raw exposure, so target encoding falls
            # back to unweighted statistics. Point to `exposure=` for weighting.
            warnings.warn(
                "A link-scale `offset` is not raw exposure, so target encoding "
                "will use observation-weighted (unweighted) statistics. Pass "
                "`exposure=` for exposure-weighted target encoding.",
                UserWarning,
                stacklevel=3,
            )

        # Keep the validated raw exposure for fold-safe CV (RS-ACT-001b), where
        # each fold's exposure-weighted target encoding must use only its own
        # training exposure.
        self._raw_exposure = raw_exposure

        # Build design matrix using existing pipeline
        self._builder = InteractionBuilder(data)
        self.y, self.X, self.feature_names = self._builder.build_design_matrix_from_parsed(
            parsed, exposure=raw_exposure, seed=seed
        )
        self.n_obs = len(self.y)
        self.n_params = self.X.shape[1]

        # Fit-time link-scale offset = log(exposure) [if any] + offset [if any].
        user_offset = self._process_offset(offset)
        if raw_exposure is not None:
            log_exposure = np.log(raw_exposure)
            self.offset = log_exposure if user_offset is None else log_exposure + user_offset
        else:
            self.offset = user_offset
        self.weights = self._process_weights(weights)
        complement_link = self._process_complement(complement, raw_exposure)
        if complement_link is not None:
            if self.offset is not None:
                self.offset = self.offset + complement_link
            else:
                self.offset = complement_link

    def _build_formula_string(self) -> str:
        """Build a formula string representation for display purposes."""
        parts = [self.response, "~"]
        term_strs = []

        for var_name, spec in self.terms.items():
            term_type = spec.get("type", "linear")
            if term_type == "linear":
                term_strs.append(var_name)
            elif term_type == "categorical":
                term_strs.append(f"C({var_name})")
            elif term_type == "bs":
                knots = spec.get("knots")
                if knots is not None:
                    term_strs.append(f"bs({var_name}, knots=[{len(knots)}])")
                else:
                    df = spec.get("df", DEFAULT_SPLINE_DF)
                    term_strs.append(f"bs({var_name}, df={df})")
            elif term_type == "ns":
                knots = spec.get("knots")
                if knots is not None:
                    term_strs.append(f"ns({var_name}, knots=[{len(knots)}])")
                else:
                    df = spec.get("df", DEFAULT_SPLINE_DF)
                    term_strs.append(f"ns({var_name}, df={df})")
            elif term_type == "ms":
                mono = spec.get("monotonicity", "increasing")
                knots = spec.get("knots")
                if knots is not None:
                    term_strs.append(f"ms({var_name}, knots=[{len(knots)}], {mono})")
                else:
                    df = spec.get("df", DEFAULT_SPLINE_DF)
                    term_strs.append(f"ms({var_name}, df={df}, {mono})")
            elif term_type == "target_encoding":
                interaction = spec.get("interaction")
                if interaction:
                    term_strs.append(f"TE({':'.join(interaction)})")
                else:
                    term_strs.append(f"TE({var_name})")
            elif term_type == "frequency_encoding":
                term_strs.append(f"FE({var_name})")
            elif term_type == "expression":
                expr = spec.get("expr", var_name)
                term_strs.append(f"I({expr})")

        if not self.intercept:
            term_strs.insert(0, "0")

        parts.append(" + ".join(term_strs) if term_strs else "1")
        return " ".join(parts)

    def explore(
        self,
        categorical_factors: list[str] | None = None,
        continuous_factors: list[str] | None = None,
        n_bins: int = 10,
        rare_threshold_pct: float = 1.0,
        max_categorical_levels: int = 20,
        detect_interactions: bool = True,
        max_interaction_factors: int = 10,
    ) -> DataExploration:
        """
        Explore data before fitting the model.

        Parameters
        ----------
        categorical_factors : list of str, optional
            Names of categorical factors to analyze.
        continuous_factors : list of str, optional
            Names of continuous factors to analyze.
        n_bins : int, default=10
            Number of bins for continuous factors.
        rare_threshold_pct : float, default=1.0
            Threshold (%) below which categorical levels are grouped.
        max_categorical_levels : int, default=20
            Maximum categorical levels to show.
        detect_interactions : bool, default=True
            Whether to detect potential interactions.
        max_interaction_factors : int, default=10
            Maximum factors for interaction detection.

        Returns
        -------
        DataExploration
            Pre-fit exploration results with to_json() method.
        """
        from rustystats.diagnostics import explore_data

        data = self.data
        exposure_col = self._exposure_spec if isinstance(self._exposure_spec, str) else None
        if self._exposure_spec is not None and not isinstance(self._exposure_spec, str):
            exposure_values = np.asarray(self._exposure_spec, dtype=np.float64)
            if exposure_values.ndim != 1 or exposure_values.shape[0] != len(data):
                raise ValidationError(
                    "array exposure must be one-dimensional and match the training data length "
                    "to use explore()."
                )
            exposure_col = "__rustystats_exposure__"
            while exposure_col in data.columns:
                exposure_col = f"_{exposure_col}"
            data = data.with_columns(pl.Series(exposure_col, exposure_values))

        return explore_data(
            data=data,
            response=self.response,
            categorical_factors=categorical_factors,
            continuous_factors=continuous_factors,
            exposure=exposure_col,
            family=self.family,
            n_bins=n_bins,
            rare_threshold_pct=rare_threshold_pct,
            max_categorical_levels=max_categorical_levels,
            detect_interactions=detect_interactions,
            max_interaction_factors=max_interaction_factors,
        )

    def _resolve_negbinomial_theta(
        self, alpha: float, cv: int | None, regularization: str | None
    ) -> tuple[bool, float]:
        """Resolve the Negative Binomial theta contract (RS-ACT-010).

        Returns ``(estimate, theta)``. When ``estimate`` is True the profile MLE
        runs later and ``theta`` is an unused placeholder; otherwise ``theta`` is
        the fixed value to use. ``theta="estimate"`` requests profile-MLE
        estimation, which is only defined for the plain GLM path -- smooth,
        regularized, or sign-constrained models raise and must be given an
        explicit numeric theta. ``theta=None`` is rejected so Negative Binomial
        fits never silently choose either theta=1.0 or profile estimation.
        """
        spec = self.theta
        if spec is None:
            raise ValidationError(
                "Negative Binomial requires an explicit theta. Pass a positive numeric "
                "theta= for a fixed-dispersion fit, or theta='estimate' to opt in to "
                "profile-likelihood estimation on the plain GLM path."
            )
        if isinstance(spec, str):
            if spec != "estimate":
                raise ValidationError(
                    f"theta must be a positive number or 'estimate', got {spec!r}."
                )
        elif spec is not None:
            if spec <= 0:
                raise ValidationError(f"theta must be > 0 for negative binomial, got {spec}.")
            return False, float(spec)

        # spec is "estimate": estimation is plain-path only.
        unsupported = []
        if self._builder.get_smooth_terms()[0]:
            unsupported.append("smooth")
        if regularization is not None or cv is not None or alpha != 0.0:
            unsupported.append("regularized")
        nonneg, nonpos = _get_constraint_indices(self.feature_names)
        if nonneg or nonpos:
            unsupported.append("sign-constrained")
        if unsupported:
            raise ValidationError(
                "Negative Binomial theta estimation is not supported for "
                f"{'/'.join(unsupported)} models (the profile loop is defined only "
                "for the plain GLM path). Pass an explicit numeric theta= "
                "(e.g. theta=1.0) for these models."
            )
        return True, DEFAULT_NEGBINOMIAL_THETA  # placeholder; real theta from MLE

    def _inference_status_and_route(
        self,
        requested_alpha: float,
        requested_l1: float,
        cv: int | None,
        regularization: str | None,
        path_info: RegularizationPathInfo | None,
    ) -> tuple[str, str]:
        """Classify inference validity and record the optimizer route (RS-ACT-011).

        Ordinary standard errors / p-values / AIC / BIC are only valid for an
        unpenalized, unselected, unconstrained, non-smooth fit. Anything else gets
        a conservative non-``valid_standard`` status so the summary can suppress
        the usual significance machinery rather than present it as trustworthy.
        """
        smooth = bool(self._builder.get_smooth_terms()[0])
        nonneg, nonpos = _get_constraint_indices(self.feature_names)
        constrained = bool(nonneg or nonpos)
        used_cv = path_info is not None

        if used_cv:
            status = "naive_after_cv_selection"
        elif requested_alpha > 0:
            status = "naive_after_selection" if requested_l1 > 0 else "naive_after_regularization"
        elif smooth:
            status = "unavailable"
        elif constrained:
            status = "constrained_boundary"
        else:
            status = "valid_standard"

        if smooth:
            route = "gcv_penalized"
        elif regularization in ("lasso", "elastic_net") or (
            requested_alpha > 0 and requested_l1 > 0
        ):
            route = "coordinate_descent"
        else:
            route = "irls"
        return status, route

    def fit(
        self,
        alpha: float = 0.0,
        l1_ratio: float = 0.0,
        max_iter: int = DEFAULT_MAX_ITER,
        tol: float = DEFAULT_TOLERANCE,
        # Cross-validation based regularization path parameters
        cv: int | None = None,
        selection: str = "min",
        regularization: str | None = None,
        n_alphas: int = DEFAULT_N_ALPHAS,
        alpha_min_ratio: float = DEFAULT_ALPHA_MIN_RATIO,
        cv_seed: int | None = None,
        include_unregularized: bool = True,
        standardize: bool = True,
        verbose: bool = False,
        # Memory optimization
        store_design_matrix: bool = False,
    ) -> GLMModel:
        """
        Fit the GLM model, optionally with regularization.

        Parameters
        ----------
        alpha : float, default=0.0
            Regularization strength. Higher values = more shrinkage.
            Ignored if regularization is specified (uses CV to find optimal).

        l1_ratio : float, default=0.0
            Elastic Net mixing parameter (0=Ridge, 1=Lasso).
            Ignored if regularization is specified with type.

        max_iter : int, default=25
            Maximum IRLS iterations.
        tol : float, default=1e-8
            Convergence tolerance.

        cv : int, optional
            Number of cross-validation folds. Defaults to 5 if regularization is set.

        selection : str, default="min"
            CV selection method: "min" or "1se".

        regularization : str, optional
            Type: "ridge", "lasso", or "elastic_net". Triggers CV-based alpha selection.

        n_alphas : int, default=20
            Number of alpha values in CV path.

        alpha_min_ratio : float, default=0.0001
            Smallest alpha as ratio of alpha_max.

        cv_seed : int, optional
            Random seed for CV folds.

        include_unregularized : bool, default=True
            Include alpha=0 in CV comparison.

        standardize : bool, default=True
            For regularized fits, internally center/scale penalized design
            columns before applying the penalty, then report coefficients on
            the original scale. Set False for legacy raw-scale penalization.

        verbose : bool, default=False
            Print progress.

        Returns
        -------
        GLMModel
            Fitted model results.
        """
        is_negbinomial = is_negbinomial_family(self.family)
        # RS-ACT-011: capture the requested penalty before CV resolution rewrites
        # alpha/l1_ratio, so inference honesty reflects what the user asked for.
        requested_alpha, requested_l1 = alpha, l1_ratio

        # RS-ACT-010: resolve the Negative Binomial theta contract before any CV
        # work, so unsupported estimate combinations fail closed early and we
        # never silently fall back to theta=1.0.
        nb_estimate = False
        theta_metadata: dict | None = None
        if is_negbinomial:
            nb_estimate, theta = self._resolve_negbinomial_theta(alpha, cv, regularization)
        else:
            theta = (
                self.theta if isinstance(self.theta, (int, float)) else DEFAULT_NEGBINOMIAL_THETA
            )

        # RS-ACT-006: fail before CV/regularization path work starts. The final
        # core fit validates again after CV selects alpha, but unsupported
        # Tweedie regimes must not reach fold fitting or deviance scoring first.
        from rustystats.validation import validate_glm_inputs

        validate_glm_inputs(
            self.y,
            self.X,
            self.family,
            self.weights,
            self.offset,
            self.feature_names,
            is_exposure_offset=False,
            var_power=self.var_power,
            allow_extended_tweedie=self.allow_extended_tweedie,
        )

        # Handle CV-based regularization path (shared logic in _GLMBase)
        alpha, l1_ratio, path_info = self._resolve_cv_path(
            alpha,
            l1_ratio,
            max_iter,
            tol,
            cv,
            selection,
            regularization,
            n_alphas,
            alpha_min_ratio,
            cv_seed,
            include_unregularized,
            verbose,
            standardize,
        )

        if nb_estimate:
            # Profile-likelihood theta estimation (plain path only, RS-ACT-010).
            result, theta_metadata = _estimate_negbinomial(
                self.y,
                self.X,
                self.link,
                self.offset,
                self.weights,
                self.feature_names,
                max_iter=max_iter,
                tol=tol,
                store_design_matrix=store_design_matrix,
            )
            theta = theta_metadata["theta"]
            smooth_results = total_edf = gcv = None
        else:
            # Use shared core fitting logic
            result, smooth_results, total_edf, gcv = _fit_glm_core(
                self.y,
                self.X,
                self.family,
                self.link,
                self.var_power,
                theta,
                self.offset,
                self.weights,
                alpha,
                l1_ratio,
                max_iter,
                tol,
                self.feature_names,
                self._builder,
                fit_intercept=self.intercept,
                store_design_matrix=store_design_matrix,
                allow_extended_tweedie=self.allow_extended_tweedie,
                standardize=standardize,
            )
            if is_negbinomial:
                theta_metadata = {
                    "estimated": False,
                    "theta": float(theta),
                    "init_theta": None,
                    "theta_iterations": 0,
                    "theta_converged": None,
                    "theta_tol": None,
                    "max_theta_iter": None,
                    "glm_tol": float(tol),
                    "fallback_reason": None,
                }

        self._smooth_results = smooth_results
        self._total_edf = total_edf
        self._gcv = gcv

        result_family = _format_result_family(self.family, self.var_power, theta)

        results = _build_results(
            result,
            self.feature_names,
            self.formula,
            result_family,
            self.link,
            self._builder,
            self._offset_spec,
            self._exposure_spec,
            path_info,
            self._smooth_results,
            self._total_edf,
            self._gcv,
            terms_dict=self.terms,
            interactions_spec=self.interactions_spec,
            input_transforms=self._input_transforms,
            compiled_input_transforms=self._compiled_input_transforms,
            complement_spec=self._complement_spec,
            complement_values=self._complement_values,
            array_exposure_requires_prediction_override=(
                self._exposure_spec is not None and not isinstance(self._exposure_spec, str)
            ),
            regularization_standardized=bool(standardize and alpha > 0.0),
        )
        # RS-ACT-010: surface the theta actually used and its estimation provenance.
        results.theta = theta if is_negbinomial else None
        results.theta_metadata = theta_metadata
        # RS-ACT-011: honest inference status + solver status surfacing.
        results.inference_status, results.optimizer_route = self._inference_status_and_route(
            requested_alpha, requested_l1, cv, regularization, path_info
        )
        results.solver_status = getattr(result, "solver_status", "converged")
        results.step_halving_used = bool(getattr(result, "step_halving_used", False))
        # RS-ACT-004 backlog #1: carry the fitted prior-weights spec so
        # result.diagnostics() can auto-propagate it into weighted decile/lift
        # aggregates (mirrors how _exposure_spec is surfaced on the result).
        results._weights_spec = getattr(self, "_weights_spec", None)
        return results


def glm_dict(
    response: str,
    terms: dict[str, dict[str, Any]],
    data: pl.DataFrame | pl.LazyFrame,
    interactions: list[dict[str, Any]] | None = None,
    intercept: bool = True,
    family: str = "gaussian",
    link: str | None = None,
    var_power: float = 1.5,
    theta: float | str | None = None,
    exposure: str | np.ndarray | None = None,
    offset: str | np.ndarray | None = None,
    weights: str | np.ndarray | None = None,
    seed: int | None = None,
    complement: str | np.ndarray | GLMModel | None = None,
    input_transforms: list[dict[str, Any]] | None = None,
    allow_extended_tweedie: bool = False,
) -> FormulaGLMDict:
    """
    Create a GLM model from a dict specification.

    This is an alternative to the formula-based API for programmatic model building.

    Parameters
    ----------
    response : str
        Name of the response variable column.
    terms : dict
        Dictionary mapping variable names to term specifications.
        Each specification is a dict with 'type' and optional parameters:

        - ``{"type": "linear"}`` - continuous variable
        - ``{"type": "categorical"}`` - dummy encoding
        - ``{"type": "categorical", "levels": ["A", "B"]}`` - specific levels
        - ``{"type": "bs", "df": 5}`` - B-spline
        - ``{"type": "bs", "df": 5, "degree": 2}`` - quadratic B-spline
        - ``{"type": "ns", "df": 4}`` - natural spline
        - ``{"type": "bs", "df": 4, "monotonicity": "increasing"}`` - monotonic
        - ``{"type": "target_encoding"}`` - target encoding
        - ``{"type": "expression", "expr": "x**2"}`` - expression
        - ``{"type": "linear", "monotonicity": "increasing"}`` - constrained

    data : pl.DataFrame or pl.LazyFrame
        Polars DataFrame or LazyFrame containing the data. If a LazyFrame
        is passed, only the columns needed by the model are collected,
        enabling optimized reads from Parquet/CSV scans.
    interactions : list of dict, optional
        List of interaction specifications. Each is a dict with variable
        names as keys and their specs as values, plus 'include_main'.
    intercept : bool, default=True
        Whether to include an intercept.
    family : str, default="gaussian"
        Distribution family.
    link : str, optional
        Link function. If None, uses canonical link.
    var_power : float, default=1.5
        Variance power for Tweedie family.
    theta : float or {"estimate"}, optional
        Negative Binomial dispersion/shape. One of:

        - a positive number — used as a fixed theta (recorded as fixed);
        - ``"estimate"`` — profile-likelihood estimation, available only on the
          plain GLM path (smooth / regularized / sign-constrained NB fits raise
          unless a numeric theta is given);
        - unspecified / ``None`` — raises for ``family="negbinomial"``; there is
          no silent ``theta=1.0`` and no implicit estimation.

        NB results expose ``result.theta`` and ``result.theta_metadata``
        (estimated-vs-fixed, init theta, iterations, convergence, tolerances,
        fallback reason).
    exposure : str or array-like, optional
        Raw positive exposure (the rate denominator) for log-link rate models.
        Added to the linear predictor as ``log(exposure)`` and used as the
        denominator for exposure-weighted target encoding.
    offset : str or array-like, optional
        Link-scale additive offset, used as-is (a string names a column added
        verbatim on the link scale). Use ``exposure=`` for the rate denominator;
        an ``offset`` is never treated as raw exposure.
    weights : str or array-like, optional
        Prior weights.
    seed : int, optional
        Random seed for deterministic target encoding.
    complement : str, array-like, or GLMModel, optional
        Complement of credibility for lasso credibility. Values on
        the response scale (rates for log-link, probabilities for logit).
        When used with regularization (especially lasso), coefficients are
        shrunk toward the complement rather than toward zero. If str,
        column name in data. If GLMModel, predictions are computed and
        divided by exposure if applicable.
    input_transforms : list of dict, optional
        Deterministic input transforms (currently ``{"type": "lookup", ...}``)
        applied to the raw data before the design matrix is built, during fit,
        prediction, contributions, diagnostics, and serialization. Terms may
        reference a transform ``output`` column that is absent from the raw
        frame. See :func:`rustystats.validate_input_transforms` for the schema.
    allow_extended_tweedie : bool, default False
        Allow Tweedie power parameters outside the default ``1 < p < 2``
        compound-Poisson-Gamma range. By default such powers are rejected
        (RS-ACT-006); set this only when you intend an extended-support model.

    Returns
    -------
    FormulaGLMDict
        Model object. Call .fit() to fit the model.

    Examples
    --------
    >>> # Standard GLM
    >>> result = rs.glm_dict(
    ...     response="ClaimCount",
    ...     terms={
    ...         "VehAge": {"type": "linear"},
    ...         "DrivAge": {"type": "bs", "df": 5},
    ...         "Region": {"type": "categorical"},
    ...         "Brand": {"type": "target_encoding"},
    ...     },
    ...     data=data,
    ...     family="poisson",
    ...     exposure="Exposure",
    ... ).fit()

    >>> # LazyFrame: only needed columns are collected
    >>> lf = pl.scan_parquet("insurance.parquet")
    >>> result = rs.glm_dict(
    ...     response="ClaimCount",
    ...     terms={"VehAge": {"type": "linear"}, "Region": {"type": "categorical"}},
    ...     data=lf,
    ...     family="poisson",
    ...     exposure="Exposure",
    ... ).fit()

    >>> # Lasso credibility: shrink state model toward countrywide rates
    >>> state_result = rs.glm_dict(
    ...     response="ClaimCount",
    ...     terms={
    ...         "VehAge": {"type": "bs"},
    ...         "Region": {"type": "categorical"},
    ...     },
    ...     data=state_data,
    ...     family="poisson",
    ...     exposure="Exposure",
    ...     complement="countrywide_rate",
    ... ).fit(regularization="lasso")
    """
    # Resolve LazyFrame: select only needed columns, then collect
    needed = _extract_needed_columns(
        terms,
        response=response,
        interactions=interactions,
        input_transforms=input_transforms,
        offset=offset,
        weights=weights,
        exposure=exposure,
        complement=complement,
    )
    data = _collect_lazyframe(data, needed)

    return FormulaGLMDict(
        response=response,
        terms=terms,
        data=data,
        interactions=interactions,
        intercept=intercept,
        family=family,
        link=link,
        var_power=var_power,
        theta=theta,
        exposure=exposure,
        offset=offset,
        weights=weights,
        seed=seed,
        complement=complement,
        input_transforms=input_transforms,
        allow_extended_tweedie=allow_extended_tweedie,
    )
