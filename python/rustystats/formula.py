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


def is_negbinomial_family(family: str) -> bool:
    """Check if the family string refers to a Negative Binomial distribution."""
    return family.lower().split("(", 1)[0].strip() in NEGBINOMIAL_ALIASES


def is_tweedie_family(family: str) -> bool:
    """Check if the family string refers to a Tweedie distribution."""
    return family.lower().split("(", 1)[0].strip() == "tweedie"


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
    if response is not None:
        cols.add(response)

    for var_name, spec in terms.items():
        term_type = spec.get("type", "linear")
        if term_type == "expression":
            expr = spec["expr"]
            for token in re.findall(r"\b([A-Za-z_]\w*)\b", expr):
                cols.add(token)
        else:
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
                cols.add(key)

    if isinstance(offset, str):
        cols.add(offset)
    if isinstance(exposure, str):
        cols.add(exposure)
    if isinstance(weights, str):
        cols.add(weights)
    if isinstance(complement, str):
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
    store_design_matrix: bool = False,
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
    # (raw exposure validation happens there before log-transform)
    y, X, weights, offset = validate_glm_inputs(
        y, X, family, weights, offset, feature_names, is_exposure_offset=False
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
        )
        return result, smooth_results, total_edf, gcv

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
    is_exposure_offset: bool,
    exposure_spec: str | np.ndarray | None,
    path_info: RegularizationPathInfo | None,
    smooth_results: list[SmoothTermResult] | None,
    total_edf: float | None,
    gcv: float | None,
    terms_dict: dict[str, dict[str, Any]] | None = None,
    interactions_spec: list[dict[str, Any]] | None = None,
    complement_spec: str | GLMModel | None = None,
    complement_values: np.ndarray | None = None,
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
        offset_is_exposure=is_exposure_offset,
        exposure_spec=exposure_spec,
        regularization_path_info=path_info,
        smooth_results=smooth_results,
        total_edf=total_edf,
        gcv=gcv,
        terms_dict=terms_dict,
        interactions_spec=interactions_spec,
        complement_spec=complement_spec,
        complement_values=complement_values,
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
        """Check if model uses log link (explicit or canonical)."""
        if self.link == "log":
            return True
        if self.link is None and self.family in ("poisson", "quasipoisson", "negbinomial", "gamma"):
            return True
        return False

    def _process_offset(
        self,
        offset: str | np.ndarray | None,
        *,
        string_is_exposure: bool | None = None,
    ) -> np.ndarray | None:
        """Process offset specification, applying log for log-link families.

        For log-link families (Poisson, Gamma, etc.), exposure must be strictly
        positive before log-transform. Validation is done here on raw values.
        """
        from rustystats.exceptions import ValidationError

        if offset is None:
            return None

        if string_is_exposure is None:
            string_is_exposure = isinstance(offset, str) and self._uses_log_link()

        if isinstance(offset, str):
            offset_values = _get_column(self.data, offset)
            if string_is_exposure:
                # Validate raw exposure before log-transform
                n_invalid = np.sum(offset_values <= 0)
                if n_invalid > 0:
                    raise ValidationError(
                        f"Exposure '{offset}' must be strictly positive for {self.family} family with log link. "
                        f"Found {n_invalid} values <= 0. "
                        "Exposure represents the denominator (e.g., time, population) and cannot be zero or negative."
                    )
                offset_values = np.log(offset_values)
            return offset_values.astype(np.float64)
        else:
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
            if vals.ndim != 1 or vals.shape[0] != self.data.height:
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
        """True if any term or interaction requests target encoding."""
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
        offset: str | np.ndarray | None,
    ) -> np.ndarray | None:
        """Raw positive exposure for target encoding.

        Comes only from an explicit ``exposure`` or a legacy string offset under a
        log link -- never from a link-scale array offset (RS-ACT-002).
        """
        if exposure is not None:
            return self._resolve_exposure_values(exposure)
        if isinstance(offset, str) and self._uses_log_link():
            return self._resolve_exposure_values(offset)
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
    offset_is_exposure: bool,
) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    """Resolve a prediction-time offset to its raw and link-scale forms.

    Returns ``(raw, link_scale, column_name)`` or all-``None`` when no
    offset applies. ``raw`` and ``link_scale`` differ only for log-link
    exposure offsets (raw = exposure, link_scale = ``log(exposure)``);
    otherwise they share the same array.
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
        link = np.log(raw) if offset_is_exposure else raw
        return raw, link, offset_to_use
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
    if raw.ndim != 1 or raw.shape[0] != len(new_data):
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
    offset_is_exposure: bool,
    offset_spec_for_complement: str | np.ndarray | None,
    exposure_spec_for_complement: str | np.ndarray | None,
    link: str,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Resolve a prediction-time complement to its response and link scales.

    Returns ``(response_scale, link_scale)`` or both ``None`` when no
    complement applies. When the complement is a fitted ``GLMModel`` and the
    current model uses an exposure offset, the prior's response is divided by
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
        elif offset_is_exposure and isinstance(offset_spec_for_complement, str):
            exposure = new_data[offset_spec_for_complement].to_numpy().astype(np.float64)
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

    def __init__(
        self,
        result,
        feature_names: list[str],
        formula: str,
        family: str,
        link: str | None,
        builder: InteractionBuilder | None = None,
        offset_spec: str | np.ndarray | None = None,
        offset_is_exposure: bool = False,
        exposure_spec: str | np.ndarray | None = None,
        regularization_path_info: RegularizationPathInfo | None = None,
        smooth_results: list[SmoothTermResult] | None = None,
        total_edf: float | None = None,
        gcv: float | None = None,
        terms_dict: dict[str, dict[str, Any]] | None = None,
        interactions_spec: list[dict[str, Any]] | None = None,
        complement_spec: str | GLMModel | None = None,
        complement_values: np.ndarray | None = None,
    ):
        self._result = result
        self._is_deserialized = isinstance(result, _DeserializedResult)
        self._smooth_results = smooth_results
        self._total_edf = total_edf
        self._gcv = gcv
        self.feature_names = feature_names
        self.formula = formula
        self.family = family
        self._regularization_path_info = regularization_path_info
        self.link = link or get_default_link(family)
        self._builder = builder
        self._offset_spec = offset_spec
        self._offset_is_exposure = offset_is_exposure
        self._exposure_spec = exposure_spec
        self._terms_dict = terms_dict
        self._interactions_spec = interactions_spec
        self._complement_spec = complement_spec
        self._complement_values = complement_values

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying result object.

        This handles all properties and methods from PyGLMResults that are
        not explicitly defined on GLMModel (params, fittedvalues, deviance,
        bse, tvalues, pvalues, conf_int, resid_*, llf, aic, bic, scale,
        robust SEs, regularization properties, etc.).
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

    def aic(self) -> float:
        """Akaike Information Criterion.

        For penalized smooth models, use effective degrees of freedom instead of
        the raw basis-column count.
        """
        if self._total_edf is not None:
            return -2.0 * self._result.llf() + 2.0 * self._total_edf
        return self._result.aic()

    def bic(self) -> float:
        """Bayesian Information Criterion.

        For penalized smooth models, use effective degrees of freedom instead of
        the raw basis-column count.
        """
        if self._total_edf is not None:
            return -2.0 * self._result.llf() + self._total_edf * np.log(self._result.nobs)
        return self._result.bic()

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

        return pl.DataFrame(
            {
                "Feature": self.feature_names,
                "Estimate": self.params,
                "Std.Error": self.bse(),
                "z": self.tvalues(),
                "Pr(>|z|)": self.pvalues(),
                "Signif": self.significance_codes(),
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

        ci = self.conf_int()

        return pl.DataFrame(
            {
                "Feature": self.feature_names,
                "Relativity": np.exp(self.params),
                "CI_Lower": np.exp(ci[:, 0]),
                "CI_Upper": np.exp(ci[:, 1]),
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
            self._result,
            feature_names=self.feature_names,
            title=title,
            inference_status=getattr(self, "inference_status", None),
            solver_status=getattr(self, "solver_status", None),
            optimizer_route=getattr(self, "optimizer_route", None),
            effective_df=self._total_edf,
        )

        if self.has_complement:
            n_zeroed = int(np.sum(np.abs(self.params[1:]) < 1e-10))
            n_total = len(self.params) - 1
            result += (
                f"\nNote: Coefficients are deviations from the complement of credibility.\n"
                f"      {n_zeroed}/{n_total} non-intercept terms zeroed "
                f"(complement fully trusted).\n"
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
        >>> result = rs.glm_dict(response="ClaimNb", terms={"Age": {"type": "linear"}, "Region": {"type": "categorical"}}, data=data, family="poisson", offset="Exposure").fit()
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
            Offset for new data. If None and the model was fit with an offset
            column name, that column will be extracted from new_data.
            For Poisson/Gamma with log link, log() is auto-applied to exposure.
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
        >>> model = rs.glm_dict(response="ClaimNb", terms={"Age": {"type": "linear"}, "Region": {"type": "categorical"}}, data=data, family="poisson", offset="Exposure")
        >>> result = model.fit()
        >>>
        >>> # Predict on new data
        >>> predictions = result.predict(new_data)
        >>>
        >>> # Predict with custom offset
        >>> predictions = result.predict(new_data, offset=np.log(new_exposures))
        """
        if self._builder is None:
            raise PredictionError(
                "Cannot predict: model was not fitted with formula API. "
                "Use fittedvalues for training data predictions."
            )

        # Resolve LazyFrame: select only columns needed for prediction
        if self._terms_dict is not None:
            needed = _extract_needed_columns(
                terms=self._terms_dict,
                interactions=self._interactions_spec,
                offset=offset if offset is not None else self._offset_spec,
                exposure=exposure if exposure is not None else self._exposure_spec,
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
        if n_rows <= chunk_size:
            # Small input: skip slicing overhead, keep behavior identical to
            # the pre-chunking implementation.
            X_new = self._builder.transform_new_data(new_data)
            linear_pred = X_new @ params
            del X_new
        else:
            linear_pred = np.empty(n_rows, dtype=np.float64)
            for start in range(0, n_rows, chunk_size):
                stop = min(start + chunk_size, n_rows)
                chunk = new_data.slice(start, stop - start)
                X_chunk = self._builder.transform_new_data(chunk)
                # Write directly into the pre-allocated output slice; the
                # X_chunk reference is rebound on the next iteration so the
                # ~chunk_size × p matrix is freed before the next one is built.
                linear_pred[start:stop] = X_chunk @ params
                del X_chunk, chunk

        exposure_to_use = exposure if exposure is not None else self._exposure_spec
        exposure_link = None
        if exposure_to_use is not None:
            if self.link != "log":
                raise ValidationError("exposure= is only meaningful for log-link rate models.")
            exposure_link = _resolve_predict_exposure_link(new_data, exposure_to_use)
            linear_pred = linear_pred + exposure_link

        offset_is_exposure = self._offset_is_exposure and exposure_to_use is None
        _, offset_link, _ = _resolve_predict_offset(
            new_data, offset, self._offset_spec, offset_is_exposure
        )
        if offset_link is not None:
            linear_pred = linear_pred + offset_link

        _, complement_link = _resolve_predict_complement(
            new_data,
            complement,
            self._complement_spec,
            offset_is_exposure,
            offset if offset is not None else self._offset_spec,
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
            Override the offset used during fitting. ``str`` resolves a column
            in ``new_data``; arrays are used directly. For log-link models with
            an exposure offset, ``log()`` is applied automatically when a
            string is given.
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
        from rustystats.export_onnx import to_onnx

        return to_onnx(self, path=path, n_grid_points=n_grid_points, mode=mode)

    def to_bytes(self) -> bytes:
        """
        Serialize the fitted model to bytes for storage or transfer.

        The serialized model can be loaded with `GLMModel.from_bytes()`.
        All state needed for prediction is preserved, including:
        - Coefficients and feature names
        - Categorical encoding levels
        - Spline knot positions
        - Target encoding statistics
        - Family parameter metadata such as Negative Binomial theta

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

        state = {
            "schema_version": 3,
            "result_state": result_state,
            "feature_names": self.feature_names,
            "formula": self.formula,
            "family": self.family,
            "link": self.link,
            "builder_state": builder_state,
            "offset_spec": self._offset_spec,
            "offset_is_exposure": self._offset_is_exposure,
            "exposure_spec": self._exposure_spec,
            "smooth_results": self._smooth_results,
            "total_edf": self._total_edf,
            "gcv": self._gcv,
            "terms_dict": self._terms_dict,
            "interactions_spec": self._interactions_spec,
            "complement_spec": self._complement_spec,
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
        offset_is_exposure = state.get("offset_is_exposure", False)
        exposure_spec = state.get("exposure_spec")
        if exposure_spec is None and offset_is_exposure:
            exposure_spec = offset_spec
            offset_spec = None
            offset_is_exposure = False

        model = cls(
            result=result,
            feature_names=state["feature_names"],
            formula=state["formula"],
            family=state["family"],
            link=state["link"],
            builder=builder,
            offset_spec=offset_spec,
            offset_is_exposure=offset_is_exposure,
            exposure_spec=exposure_spec,
            regularization_path_info=None,
            smooth_results=state["smooth_results"],
            total_edf=state["total_edf"],
            gcv=state["gcv"],
            terms_dict=state.get("terms_dict"),
            interactions_spec=state.get("interactions_spec"),
            complement_spec=state.get("complement_spec"),
        )
        model.theta = result_state.get("theta")
        model.theta_metadata = result_state.get("theta_metadata")
        model.inference_status = result_state.get("inference_status")
        model.optimizer_route = result_state.get("optimizer_route")
        model.solver_status = result_state.get("solver_status")
        model.step_halving_used = result_state.get("step_halving_used")
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
    ):
        self.response = response
        self.terms = terms
        self.interactions_spec = interactions
        self.intercept = intercept
        # Store weak reference to data to allow garbage collection
        self._data_ref = weakref.ref(data)
        self.family = family.lower()
        self.link = link
        self.var_power = var_power
        self.theta = theta
        self._exposure_spec = exposure
        self._offset_spec = offset
        self._weights_spec = weights
        self._seed = seed
        # RS-ACT-002: a string offset under a log link has historically meant
        # raw exposure; preserve that meaning only when no explicit exposure= is
        # supplied.
        self._offset_is_legacy_exposure_alias = (
            exposure is None and isinstance(offset, str) and self._uses_log_link()
        )
        if self._offset_is_legacy_exposure_alias:
            # The legacy string offset *is* raw exposure; record it as such so
            # prediction, diagnostics, and serialization treat it uniformly.
            self._exposure_spec = offset
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

        # RS-ACT-002: keep model metadata split into raw exposure and link-scale
        # offset specs, while constructing a fit-only combined link-scale offset
        # for the Rust solver.
        raw_exposure = self._get_raw_exposure(exposure, offset)
        fit_offset = offset
        fit_offset_string_is_exposure: bool | None = None
        if exposure is not None:
            if not self._uses_log_link():
                raise ValidationError(
                    "`exposure=` is only supported for log-link rate models "
                    f"(got family={self.family!r}, link={self.link!r}). Use "
                    "`offset=` for a link-scale adjustment on other links."
                )
            if isinstance(exposure, str) and offset is None:
                fit_offset = exposure
                fit_offset_string_is_exposure = True
            else:
                user_offset = self._process_offset(offset, string_is_exposure=False)
                log_exposure = np.log(raw_exposure)
                fit_offset = log_exposure if user_offset is None else log_exposure + user_offset
            self._offset_spec = offset
        elif self._offset_is_legacy_exposure_alias:
            # Legacy offset="Exposure" is normalized to explicit exposure
            # metadata; the string is used only to produce the fit-time log
            # offset so fitted values remain identical to historical behavior.
            fit_offset = offset
            fit_offset_string_is_exposure = True
            self._offset_spec = None
        elif (
            offset is not None
            and not isinstance(offset, str)
            and self._uses_log_link()
            and self._has_target_encoding()
        ):
            # A link-scale array offset is not raw exposure (RS-ACT-002), so target
            # encoding falls back to unweighted statistics. Warn and point to
            # exposure= for exposure-weighted encoding.
            warnings.warn(
                "A link-scale array `offset` is not raw exposure, so target "
                "encoding will use observation-weighted (unweighted) statistics. "
                "Pass `exposure=` for exposure-weighted target encoding.",
                UserWarning,
                stacklevel=2,
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

        # Process offset, weights, and complement
        self.offset = self._process_offset(
            fit_offset, string_is_exposure=fit_offset_string_is_exposure
        )
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

        exposure_col = self._exposure_spec if isinstance(self._exposure_spec, str) else None

        return explore_data(
            data=self.data,
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
        the fixed value to use. ``theta="estimate"`` and an unspecified
        ``theta=None`` both request estimation, which is only defined for the
        plain GLM path -- smooth, regularized, or sign-constrained models raise
        and must be given an explicit numeric theta.
        """
        spec = self.theta
        if isinstance(spec, str):
            if spec != "estimate":
                raise ValidationError(
                    f"theta must be a positive number or 'estimate', got {spec!r}."
                )
        elif spec is not None:
            if spec <= 0:
                raise ValidationError(f"theta must be > 0 for negative binomial, got {spec}.")
            return False, float(spec)

        # spec is None or "estimate": estimation is plain-path only.
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
                store_design_matrix=store_design_matrix,
            )
            if is_negbinomial:
                theta_metadata = {"estimated": False, "theta": theta}

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
            False,
            self._exposure_spec,
            path_info,
            self._smooth_results,
            self._total_edf,
            self._gcv,
            terms_dict=self.terms,
            interactions_spec=self.interactions_spec,
            complement_spec=self._complement_spec,
            complement_values=self._complement_values,
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
    theta : float, optional
        Dispersion for Negative Binomial.
    exposure : str or array-like, optional
        Raw positive exposure (the rate denominator) for log-link rate models.
        Added to the linear predictor as ``log(exposure)`` and used as the
        denominator for exposure-weighted target encoding. Prefer this over
        ``offset="Exposure"`` (which remains accepted as a legacy alias).
    offset : str or array-like, optional
        Link-scale additive offset. A string offset for a log-link family is
        treated as raw exposure (a legacy alias for ``exposure=``); an array
        offset is used on the link scale as-is.
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
    ...     offset="Exposure",
    ... ).fit()

    >>> # LazyFrame: only needed columns are collected
    >>> lf = pl.scan_parquet("insurance.parquet")
    >>> result = rs.glm_dict(
    ...     response="ClaimCount",
    ...     terms={"VehAge": {"type": "linear"}, "Region": {"type": "categorical"}},
    ...     data=lf,
    ...     family="poisson",
    ...     offset="Exposure",
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
    ...     offset="Exposure",
    ...     complement="countrywide_rate",
    ... ).fit(regularization="lasso")
    """
    # Resolve LazyFrame: select only needed columns, then collect
    needed = _extract_needed_columns(
        terms,
        response=response,
        interactions=interactions,
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
    )
