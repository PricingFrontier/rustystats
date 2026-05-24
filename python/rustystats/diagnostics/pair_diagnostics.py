"""
Pair (interaction) diagnostics computation.

_PairDiagnosticsComputer handles per-pair surface-grid analysis for
user-specified interaction pairs, including:

- 2D surface grids (train + optional test, cell-aligned by train-derived edges)
- Block Wald significance (for fitted interactions)
- Block GVIF (Fox-Monette)
- Interaction representation detection from TermSlot
- Pre-fit interaction_strength (partial R² vs. multiplicative null)

All operations are no-refit: fitted interactions use the Wald-style block
chi-square on the existing covariance matrix without building a nested model.

The computer follows the same pattern as ``_FactorDiagnosticsComputer``:
it is constructed once on the train data and accepts ``data``-style kwargs
on each method call so the same instance serves train and test surfaces.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import polars as pl

from rustystats._rustystats import (
    aggregate_pair_cells_py as _rust_aggregate_pair_cells,
)
from rustystats._rustystats import (
    compute_correlation_and_vif_py as _rust_correlation_and_vif,
)
from rustystats._rustystats import (
    compute_factor_significance_batch_py as _rust_factor_significance_batch,
)
from rustystats._rustystats import (
    interaction_strength_from_codes_py as _rust_interaction_strength_from_codes,
)
from rustystats.constants import EPSILON
from rustystats.diagnostics.types import (
    FactorCoefficient,
    FactorSignificance,
    InteractionDiagnostics,
    InteractionExploration,
    SurfaceCell,
    SurfaceGrid,
)

# =============================================================================
# Defaults — pair-pipeline binning and cell caps.
# =============================================================================

DEFAULT_PAIR_CONT_CONT: tuple[int, int] = (6, 6)
DEFAULT_PAIR_CONT_CAT: tuple[int, int] = (8, 15)  # (continuous bins, top-K levels)
DEFAULT_PAIR_CAT_CAT: tuple[int, int] = (12, 12)
DEFAULT_PAIR_MAX_CELLS: int = 150

# Label for the residual bucket when a categorical side is top-K capped.
_OTHER_LABEL: str = "_Other"

# TermSlot.term_type values that represent an interaction (standard tensor
# product, target-encoded, or frequency-encoded). See diagnostics/interactions.py.
_INTERACTION_LIKE_TERM_TYPES: frozenset[str] = frozenset(
    {"interaction", "target_encoding", "frequency_encoding"}
)

# Reserved keys in dict-form interaction specs (do not treat as variable names).
_RESERVED_SPEC_KEYS: frozenset[str] = frozenset(
    {
        "factor1",
        "factor2",
        "include_main",
        "target_encoding",
        "frequency_encoding",
        "prior_weight",
    }
)


# =============================================================================
# Spec normalization and pair-type detection.
# =============================================================================


def _normalize_pair_spec(spec: Any) -> tuple[str, str]:
    """Coerce a user-supplied interaction spec to ``(factor1, factor2)``.

    Accepts ``{"factor1": ..., "factor2": ...}``, ``(a, b)``, ``[a, b]``, and
    the legacy fit-time form ``{"VehAge": {...}, "Region": {...}, ...}`` where
    reserved keys are stripped.
    """
    if isinstance(spec, dict):
        if "factor1" in spec and "factor2" in spec:
            return str(spec["factor1"]), str(spec["factor2"])
        variable_keys = [k for k in spec if k not in _RESERVED_SPEC_KEYS]
        if len(variable_keys) == 2:
            return str(variable_keys[0]), str(variable_keys[1])
        raise ValueError(f"Cannot extract two factors from interaction spec: keys={list(spec)}")
    if isinstance(spec, (tuple, list)) and len(spec) == 2:
        return str(spec[0]), str(spec[1])
    raise ValueError(f"Invalid interaction spec: {spec!r}")


def _detect_pair_type(data: pl.DataFrame, factor1: str, factor2: str) -> tuple[str, str, str]:
    """Determine ``pair_type`` and normalized factor ordering.

    Ordering: continuous always becomes ``factor1`` in the mixed case;
    for cat × cat, the higher-cardinality variable becomes ``factor1``.
    """
    is_num1 = data.schema[factor1].is_numeric()
    is_num2 = data.schema[factor2].is_numeric()

    if is_num1 and is_num2:
        return "continuous_x_continuous", factor1, factor2
    if not is_num1 and not is_num2:
        card1 = data[factor1].n_unique()
        card2 = data[factor2].n_unique()
        if card2 > card1:
            return "categorical_x_categorical", factor2, factor1
        return "categorical_x_categorical", factor1, factor2
    if is_num1:
        return "continuous_x_categorical", factor1, factor2
    return "continuous_x_categorical", factor2, factor1


def _find_termslot_for_pair(model: Any, factor1: str, factor2: str):
    """Search ``model._builder._term_slots`` for an interaction-like slot
    matching the unordered pair ``{factor1, factor2}``.
    """
    builder = getattr(model, "_builder", None)
    if builder is None:
        return None
    slots = getattr(builder, "_term_slots", None)
    if slots is None:
        return None
    target = frozenset({factor1, factor2})
    for slot in slots:
        if slot.term_type not in _INTERACTION_LIKE_TERM_TYPES:
            continue
        if len(slot.factors) < 2:
            continue
        if frozenset(slot.factors) == target:
            return slot
    return None


def _representation_from_slot(slot: Any) -> str | None:
    """Map a TermSlot to a representation tag for ``InteractionDiagnostics``."""
    if slot is None:
        return None
    if slot.term_type == "target_encoding":
        return "target_encoding"
    if slot.term_type == "frequency_encoding":
        return "frequency_encoding"
    if slot.term_type == "interaction":
        return "tensor_product"
    return None


# =============================================================================
# Binning helpers — train computes edges/levels; test reuses them.
# =============================================================================


def _bin_continuous_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Compute exposure-agnostic quantile bin edges from finite values."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.array([0.0, 1.0], dtype=np.float64)
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(finite, quantiles)
    edges = np.unique(edges).astype(np.float64)
    if edges.size < 2:
        # All values identical: degenerate single-bin grid.
        edges = np.array([edges[0], edges[0] + 1.0], dtype=np.float64)
    return edges


def _apply_continuous_edges(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Map continuous values to bin indices ``[0, len(edges)-2]``."""
    if edges.size < 2:
        return np.zeros(values.shape[0], dtype=np.int32)
    # np.digitize with the inner edges yields ``len(edges)-1`` bins.
    codes = np.digitize(values, edges[1:-1])
    codes = np.clip(codes, 0, edges.size - 2)
    return codes.astype(np.int32)


def _bin_categorical_levels(
    values: np.ndarray, top_k: int, exposure: np.ndarray | None = None
) -> list[str]:
    """Pick the top-``top_k`` levels by exposure (or row count) and append
    ``_Other``. Returned list has length ``top_k + 1`` (or fewer if the
    variable has < top_k unique levels — no ``_Other`` is added then).
    """
    str_vals = _to_string_array(values)
    if str_vals.size == 0:
        return [_OTHER_LABEL]
    unique, inverse = np.unique(str_vals, return_inverse=True)
    if exposure is not None and exposure.size == str_vals.size:
        weights = np.bincount(inverse, weights=exposure.astype(np.float64))
    else:
        weights = np.bincount(inverse)
    order = np.argsort(-weights, kind="stable")
    selected = unique[order[:top_k]].tolist()
    if len(unique) > top_k:
        selected.append(_OTHER_LABEL)
    return selected


def _apply_categorical_levels(values: np.ndarray | pl.Series, levels: list[str]) -> np.ndarray:
    """Map categorical string values to level indices.

    Uses polars' Enum cast for vectorized C-level lookup whenever the input
    is a ``pl.Series`` (which is the standard path from
    ``data[factor]`` and the explorer's pre-cached columns). Falls back to
    a NumPy dict lookup for plain-array inputs (used in helper unit tests).
    Matches the perf characteristics of the convention in
    ``api.py:_precompute_data_caches`` (api.py:317).

    Semantics:

    - If ``_Other`` is the last element of ``levels``, unseen values map to it.
    - Otherwise unseen values map to ``-1`` (the cell aggregator drops them).
    """
    if levels and levels[-1] == _OTHER_LABEL:
        selected = levels[:-1]
        other_code = len(selected)
        has_other = True
    else:
        selected = list(levels)
        other_code = -1
        has_other = False

    n_values = values.len() if isinstance(values, pl.Series) else np.asarray(values).shape[0]
    if not selected:
        return np.full(n_values, other_code, dtype=np.int32)

    # Fast path: polars input → vectorized C-level Enum cast.
    if isinstance(values, pl.Series):
        s = values.cast(pl.Utf8).fill_null("None")
        if has_other:
            enum_levels = selected + [_OTHER_LABEL]
            mapped = pl.select(
                pl.when(s.is_in(selected)).then(s).otherwise(pl.lit(_OTHER_LABEL)).alias("_v")
            ).get_column("_v")
            return (
                mapped.cast(pl.Enum(enum_levels))
                .to_physical()
                .to_numpy()
                .astype(np.int32, copy=False)
            )
        # No _Other bucket: unseen values become null → -1.
        casted = s.cast(pl.Enum(selected), strict=False)
        null_mask = casted.is_null().to_numpy()
        phys = casted.to_physical().to_numpy()
        # Replace nulls with -1 (np.where preserves int dtype when both args are int).
        return np.where(null_mask, -1, phys).astype(np.int32, copy=False)

    # Fallback: numpy/object array input (helper unit tests).
    str_vals = _to_string_array(values)
    level_to_code = {lvl: i for i, lvl in enumerate(selected)}
    out = np.empty(str_vals.shape[0], dtype=np.int32)
    for i, v in enumerate(str_vals):
        out[i] = level_to_code.get(v, other_code)
    return out


def _to_string_array(values: np.ndarray | pl.Series) -> np.ndarray:
    """Coerce arbitrary input to a ``numpy.ndarray[str]``.

    Handles polars Series (via ``.to_numpy()``), numpy arrays of any dtype,
    and Python lists. Nulls become the string ``"None"`` (consistent with
    the explorer's ``fill_null("None")`` convention in explorer.py:1153).
    """
    if hasattr(values, "to_numpy"):
        values = values.to_numpy()
    arr = np.asarray(values)
    if arr.dtype.kind == "O":
        return np.array(["None" if v is None else str(v) for v in arr], dtype=object)
    if arr.dtype.kind in {"U", "S"}:
        return arr.astype(object)
    return arr.astype(str).astype(object)


def _choose_bin_counts(
    pair_type: str,
    binning_cont_cont: tuple[int, int],
    binning_cont_cat: tuple[int, int],
    binning_cat_cat: tuple[int, int],
    max_total_cells: int,
    cardinality1: int | None,
    cardinality2: int | None,
) -> tuple[int, int]:
    """Pick (n1, n2) honoring the per-pair-type defaults and the global
    ``max_total_cells`` cap. Categorical-side top-K shrinks first.

    Returned values are the count of *selected* categorical levels — they
    do NOT include the ``_Other`` bucket. The ``_Other`` bucket adds one
    to the effective grid extent when ``cardinality > top_k``; this function
    factors that in when checking the cap.
    """

    def _effective_cat(top_k: int, card: int | None) -> int:
        """Effective grid extent for a categorical side, including ``_Other``."""
        if card is None:
            return top_k + 1  # conservative: assume _Other will be added
        if card > top_k:
            return top_k + 1
        return min(top_k, card)

    if pair_type == "continuous_x_continuous":
        n1, n2 = binning_cont_cont
        total = n1 * n2
        if total <= max_total_cells:
            return n1, n2
        # cont×cont: shrink both equally
        scale = (max_total_cells / total) ** 0.5
        return max(2, int(n1 * scale)), max(2, int(n2 * scale))

    if pair_type == "continuous_x_categorical":
        n_cont, n_cat = binning_cont_cat
        if cardinality2 is not None:
            n_cat = min(n_cat, cardinality2)
        while n_cont * _effective_cat(n_cat, cardinality2) > max_total_cells and n_cat > 2:
            n_cat -= 1
        return n_cont, n_cat

    # categorical × categorical
    top_k1, top_k2 = binning_cat_cat
    if cardinality1 is not None:
        top_k1 = min(top_k1, cardinality1)
    if cardinality2 is not None:
        top_k2 = min(top_k2, cardinality2)
    # Shrink the larger side first.
    while (
        _effective_cat(top_k1, cardinality1) * _effective_cat(top_k2, cardinality2)
        > max_total_cells
    ):
        if top_k1 >= top_k2 and top_k1 > 2:
            top_k1 -= 1
        elif top_k2 > 2:
            top_k2 -= 1
        else:
            break
    return top_k1, top_k2


def _bin_factor(
    data: pl.DataFrame,
    factor: str,
    side: str,  # "row" | "col"; affects which binning parameter is used
    pair_type: str,
    n_bin: int,
    exposure: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None, list[str] | None]:
    """Train-side binning of one factor of a pair.

    Returns ``(codes, edges, levels)`` where exactly one of ``edges`` /
    ``levels`` is non-None. ``codes`` is an int32 array of length n_rows.
    """
    is_categorical_side = (pair_type == "categorical_x_categorical") or (
        pair_type == "continuous_x_categorical" and side == "col"
    )
    values_series = data[factor]
    if is_categorical_side:
        levels = _bin_categorical_levels(values_series, n_bin, exposure)
        codes = _apply_categorical_levels(values_series, levels)
        return codes, None, levels
    values = np.asarray(values_series.to_numpy(), dtype=np.float64)
    edges = _bin_continuous_edges(values, n_bin)
    codes = _apply_continuous_edges(values, edges)
    return codes, edges, None


# =============================================================================
# Cell aggregation and surface-grid construction.
# =============================================================================


def _aggregate_cells(
    codes1: np.ndarray,
    codes2: np.ndarray,
    y: np.ndarray,
    exposure: np.ndarray,
    mu: np.ndarray | None,
    n_rows: int,
    n_cols: int,
) -> dict[tuple[int, int], tuple[int, float, float, float | None]]:
    """Aggregate ``(y, exposure, optional mu, count)`` by ``(code1, code2)``.

    Delegates to the Rust ``aggregate_pair_cells_py`` kernel
    (``crates/rustystats-core/src/diagnostics/exploration.rs``). The Rust
    path does a single allocation-light pass over the input arrays without
    holding the GIL — replacing what was previously three or four
    sequential ``np.bincount`` calls (each scanning ``y``, ``exposure``,
    ``mu`` separately) plus a Python-side dict assembly. Same algorithmic
    contract: dense ``Vec`` indexing when the level product is small,
    hashmap-backed sparse aggregation otherwise.

    Rows whose ``codes1 < 0`` or ``codes2 < 0`` (the "unseen, no _Other"
    sentinel from ``_apply_categorical_levels``) and rows whose codes
    exceed the grid extent are dropped — at the Python boundary here, then
    again defensively inside the Rust kernel.
    """
    # Drop invalid rows here so we never hand u32-typed arrays to Rust with
    # negative sentinel values (the int32 → uint32 cast would silently
    # wrap them otherwise).
    valid = (codes1 >= 0) & (codes2 >= 0) & (codes1 < n_rows) & (codes2 < n_cols)
    if not valid.all():
        codes1 = codes1[valid]
        codes2 = codes2[valid]
        y = y[valid]
        exposure = exposure[valid]
        if mu is not None:
            mu = mu[valid]

    if codes1.size == 0:
        return {}

    codes1_u32 = np.ascontiguousarray(codes1, dtype=np.uint32)
    codes2_u32 = np.ascontiguousarray(codes2, dtype=np.uint32)
    y_f64 = np.ascontiguousarray(y, dtype=np.float64)
    exp_f64 = np.ascontiguousarray(exposure, dtype=np.float64)
    mu_f64 = np.ascontiguousarray(mu, dtype=np.float64) if mu is not None else None

    rust_cells = _rust_aggregate_pair_cells(
        codes1_u32,
        int(n_rows),
        codes2_u32,
        int(n_cols),
        y_f64,
        exp_f64,
        mu_f64,
    )

    out: dict[tuple[int, int], tuple[int, float, float, float | None]] = {}
    if mu is None:
        for r, c, count, exp_sum, y_sum, _mu_sum in rust_cells:
            out[(int(r), int(c))] = (int(count), float(exp_sum), float(y_sum), None)
    else:
        for r, c, count, exp_sum, y_sum, mu_sum in rust_cells:
            out[(int(r), int(c))] = (
                int(count),
                float(exp_sum),
                float(y_sum),
                float(mu_sum),
            )
    return out


def _build_surface_grid(
    factor1: str,
    factor2: str,
    codes1: np.ndarray,
    codes2: np.ndarray,
    edges1: np.ndarray | None,
    levels1: list[str] | None,
    edges2: np.ndarray | None,
    levels2: list[str] | None,
    y: np.ndarray,
    exposure: np.ndarray,
    mu: np.ndarray | None,
) -> SurfaceGrid:
    """Materialize a SurfaceGrid from raw arrays + bin definitions."""
    n_rows = (edges1.size - 1) if edges1 is not None else len(levels1 or [])
    n_cols = (edges2.size - 1) if edges2 is not None else len(levels2 or [])
    if n_rows == 0:
        n_rows = 1
    if n_cols == 0:
        n_cols = 1

    aggregated = _aggregate_cells(codes1, codes2, y, exposure, mu, n_rows, n_cols)

    cells: list[SurfaceCell] = []
    for r in range(n_rows):
        for c in range(n_cols):
            agg = aggregated.get((r, c))
            if agg is None:
                continue
            n, exp, ys, ms = agg
            if exp > 0.0:
                actual = ys / exp
                predicted = (ms / exp) if ms is not None else None
                ae_ratio = (actual / predicted) if (predicted not in (None, 0.0)) else None
            else:
                actual = 0.0
                predicted = None
                ae_ratio = None
            cells.append(
                SurfaceCell(
                    r=r,
                    c=c,
                    n=n,
                    exposure=exp,
                    actual=actual,
                    predicted=predicted,
                    ae_ratio=ae_ratio,
                )
            )

    row_type = "quantile" if edges1 is not None else "levels"
    col_type = "quantile" if edges2 is not None else "levels"
    return SurfaceGrid(
        row_axis=factor1,
        col_axis=factor2,
        row_type=row_type,
        col_type=col_type,
        cells=cells,
        row_edges=edges1.tolist() if edges1 is not None else None,
        row_levels=list(levels1) if levels1 is not None else None,
        col_edges=edges2.tolist() if edges2 is not None else None,
        col_levels=list(levels2) if levels2 is not None else None,
    )


# =============================================================================
# Statistical quantities (no refit).
# =============================================================================


def _build_design_correlation_matrix(
    design_matrix: np.ndarray | None,
    epsilon: float = EPSILON,
) -> np.ndarray | None:
    """Build the regularized design-matrix correlation matrix ``R + ε·I``.

    Computed ONCE per diagnostics call (callers cache it and pass it to
    every ``_compute_block_gvif`` invocation) so the O(n·p²) standardize
    + ``Z.T @ Z`` work is paid a single time per call rather than per
    factor / per pair. Uses the existing Rust ``compute_correlation_and_vif``
    path so the standardization is done in Rust with no Python-side
    O(n·p) Z-matrix allocation — same path the per-column VIF computer
    already uses.
    """
    if design_matrix is None:
        return None
    X = np.asarray(design_matrix, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] <= 1 or X.shape[1] == 0:
        return None
    try:
        R, _vif = _rust_correlation_and_vif(np.ascontiguousarray(X), float(epsilon))
    except (ValueError, RuntimeError, np.linalg.LinAlgError):
        return None
    R_np = np.asarray(R, dtype=np.float64)
    p = R_np.shape[0]
    return R_np + epsilon * np.eye(p)


def _compute_block_gvif(
    correlation_matrix: np.ndarray | None,
    col_start: int,
    col_end: int,
) -> float | None:
    """Generalized VIF (Fox-Monette) on design columns ``[col_start, col_end)``.

    Takes a pre-computed regularized correlation matrix
    (see ``_build_design_correlation_matrix``) so the expensive
    standardization happens once per diagnostics call rather than per
    factor / per pair. The formula:

        GVIF = det(R_block) * det(R_complement) / det(R_full)

    Returns ``None`` when the matrix is missing, the block range is
    invalid, the complement is empty (single-block design), or any
    sub-determinant is non-positive (rank-deficient sub-block).
    """
    if correlation_matrix is None:
        return None
    R_reg = correlation_matrix
    p = R_reg.shape[0]
    if col_start < 0 or col_end > p or col_start >= col_end:
        return None

    complement = [i for i in range(p) if i < col_start or i >= col_end]
    if not complement:
        return None

    # Slice the block contiguously; only the complement needs np.ix_ since
    # it's a non-contiguous index set.
    R_block = R_reg[col_start:col_end, col_start:col_end]
    R_compl = R_reg[np.ix_(complement, complement)]

    try:
        sign_b, logdet_b = np.linalg.slogdet(R_block)
        sign_c, logdet_c = np.linalg.slogdet(R_compl)
        sign_f, logdet_f = np.linalg.slogdet(R_reg)
    except np.linalg.LinAlgError:
        return None
    if sign_b <= 0 or sign_c <= 0 or sign_f <= 0:
        return None
    log_gvif = logdet_b + logdet_c - logdet_f
    gvif = float(np.exp(log_gvif))
    if not np.isfinite(gvif) or gvif <= 0.0:
        return None
    return gvif


def _compute_block_significance(
    slot: Any,
    params: np.ndarray | None,
    bse: np.ndarray | None,
    bread: np.ndarray | None,
) -> FactorSignificance | None:
    """Block Wald chi² for the slot's column range — reuses the existing
    ``compute_factor_significance_batch`` Rust path with a single-block
    parameter index list.
    """
    if slot is None or params is None or bse is None or bread is None:
        return None
    indices = list(range(slot.col_start, slot.col_end))
    if not indices:
        return None
    params_arr = np.asarray(params, dtype=np.float64)
    bse_arr = np.asarray(bse, dtype=np.float64)
    bread_arr = np.asarray(bread, dtype=np.float64)
    if indices[-1] >= params_arr.size or indices[-1] >= bse_arr.size:
        return None
    try:
        results = _rust_factor_significance_batch([indices], params_arr, bse_arr, bread_arr)
    except (ValueError, RuntimeError, np.linalg.LinAlgError):
        return None
    if not results or results[0] is None:
        return None
    res = results[0]
    chi2 = float(res["chi2"])
    pvalue = float(res["pvalue"])
    return FactorSignificance(
        chi2=round(chi2, 2),
        p=round(pvalue, 4),
        dev_contrib=round(chi2, 2),  # asymptotic LRT-equivalence under H0
        dev_pct=None,
    )


def _extract_block_coefficients(
    slot: Any,
    params: np.ndarray | None,
    bse: np.ndarray | None,
    feature_names: list[str] | None,
    link: str | None,
) -> list[FactorCoefficient] | None:
    """Build per-column ``FactorCoefficient`` entries for the slot's range."""
    if slot is None or params is None or bse is None:
        return None
    params_arr = np.asarray(params, dtype=np.float64)
    bse_arr = np.asarray(bse, dtype=np.float64)
    coefs: list[FactorCoefficient] = []
    for offset, col_idx in enumerate(range(slot.col_start, slot.col_end)):
        if col_idx >= params_arr.size or col_idx >= bse_arr.size:
            continue
        est = float(params_arr[col_idx])
        se = float(bse_arr[col_idx])
        if se > 0.0 and np.isfinite(se):
            z = est / se
        else:
            z = 0.0
        # Two-sided standard-normal p-value. Interaction block entries don't
        # carry a standalone multiplicative relativity, so relativity stays
        # None even under a log link.
        p_val = math.erfc(abs(z) / math.sqrt(2.0))
        name = (
            slot.design_column_names[offset]
            if offset < len(slot.design_column_names)
            else (
                feature_names[col_idx]
                if feature_names and col_idx < len(feature_names)
                else f"col_{col_idx}"
            )
        )
        coefs.append(
            FactorCoefficient(
                term=name,
                estimate=round(est, 6),
                std_error=round(se, 6),
                z_value=round(z, 3),
                p_value=round(p_val, 4),
                relativity=None,
            )
        )
    return coefs if coefs else None


# =============================================================================
# Public computer.
# =============================================================================


class _PairDiagnosticsComputer:
    """Per-pair (interaction) diagnostics.

    Constructed once on train data; ``compute_pair_diagnostics`` and
    ``compute_pair_exploration`` are called per request.
    """

    def __init__(
        self,
        y: np.ndarray,
        mu: np.ndarray | None,
        exposure: np.ndarray | None,
        family: str,
        feature_names: list[str] | None,
        link: str | None = None,
    ) -> None:
        self.y = np.asarray(y, dtype=np.float64)
        self.mu = np.asarray(mu, dtype=np.float64) if mu is not None else None
        self.exposure = (
            np.asarray(exposure, dtype=np.float64)
            if exposure is not None
            else np.ones_like(self.y, dtype=np.float64)
        )
        self.family = (family or "gaussian").lower()
        self.feature_names = list(feature_names or [])
        self.link = link

    def compute_pair_diagnostics(
        self,
        pairs: list[Any],
        data: pl.DataFrame,
        model: Any = None,
        design_matrix: np.ndarray | None = None,
        bread_matrix: np.ndarray | None = None,
        params: np.ndarray | None = None,
        bse: np.ndarray | None = None,
        correlation_matrix: np.ndarray | None = None,
        test_data: pl.DataFrame | None = None,
        test_y: np.ndarray | None = None,
        test_mu: np.ndarray | None = None,
        test_exposure: np.ndarray | None = None,
        binning_cont_cont: tuple[int, int] = DEFAULT_PAIR_CONT_CONT,
        binning_cont_cat: tuple[int, int] = DEFAULT_PAIR_CONT_CAT,
        binning_cat_cat: tuple[int, int] = DEFAULT_PAIR_CAT_CAT,
        max_total_cells: int = DEFAULT_PAIR_MAX_CELLS,
    ) -> list[InteractionDiagnostics]:
        """Compute post-fit ``InteractionDiagnostics`` for each pair.

        ``correlation_matrix`` is the regularized design correlation matrix
        (built once per diagnostics call via
        ``_build_design_correlation_matrix``). When supplied it's used as
        the source for all block-GVIF computations. When ``None``, GVIF is
        derived on demand from ``design_matrix`` (slower path used only
        when this method is called outside the orchestrator).
        """
        # Lazy: build correlation matrix once if caller didn't supply it
        # (typical when called from outside the orchestrator).
        if correlation_matrix is None and design_matrix is not None:
            correlation_matrix = _build_design_correlation_matrix(design_matrix)

        # Per-call binning caches: a factor that appears in multiple pairs is
        # binned only once on each side (train/test). Key is
        # ``(factor, n_bin)``; n_bin disambiguates the bins-per-side choice
        # which can differ across pair_types for the same factor (e.g. age
        # gets 8 bins in cont×cat but 6 bins in cont×cont).
        train_bin_cache: dict[
            tuple[str, int], tuple[np.ndarray, np.ndarray | None, list[str] | None]
        ] = {}
        # Test-side cache key is ``(factor, id(edges or levels))`` — bins are
        # uniquely determined by the train-side edges/levels they reuse.
        test_bin_cache: dict[tuple[str, int], np.ndarray] = {}
        out: list[InteractionDiagnostics] = []
        for spec in pairs:
            f1_raw, f2_raw = _normalize_pair_spec(spec)
            for col in (f1_raw, f2_raw):
                if col not in data.columns:
                    raise ValueError(f"Interaction factor {col!r} is not a column in train data.")
            pair_type, f1, f2 = _detect_pair_type(data, f1_raw, f2_raw)

            card1 = data[f1].n_unique() if pair_type != "continuous_x_continuous" else None
            card2 = (
                data[f2].n_unique()
                if pair_type == "categorical_x_categorical"
                or (pair_type == "continuous_x_categorical")
                else None
            )
            n1, n2 = _choose_bin_counts(
                pair_type,
                binning_cont_cont,
                binning_cont_cat,
                binning_cat_cat,
                max_total_cells,
                card1,
                card2,
            )

            key1 = (f1, n1)
            cached = train_bin_cache.get(key1)
            if cached is not None:
                codes1, edges1, levels1 = cached
            else:
                codes1, edges1, levels1 = _bin_factor(data, f1, "row", pair_type, n1, self.exposure)
                train_bin_cache[key1] = (codes1, edges1, levels1)
            key2 = (f2, n2)
            cached = train_bin_cache.get(key2)
            if cached is not None:
                codes2, edges2, levels2 = cached
            else:
                codes2, edges2, levels2 = _bin_factor(data, f2, "col", pair_type, n2, self.exposure)
                train_bin_cache[key2] = (codes2, edges2, levels2)

            train_grid = _build_surface_grid(
                f1,
                f2,
                codes1,
                codes2,
                edges1,
                levels1,
                edges2,
                levels2,
                self.y,
                self.exposure,
                self.mu,
            )

            test_grid = None
            if test_data is not None and test_y is not None and test_exposure is not None:
                if f1 in test_data.columns and f2 in test_data.columns:
                    tkey1 = (f1, n1)
                    test_codes1 = test_bin_cache.get(tkey1)
                    if test_codes1 is None:
                        test_codes1 = self._apply_bins(test_data[f1], edges1, levels1)
                        test_bin_cache[tkey1] = test_codes1
                    tkey2 = (f2, n2)
                    test_codes2 = test_bin_cache.get(tkey2)
                    if test_codes2 is None:
                        test_codes2 = self._apply_bins(test_data[f2], edges2, levels2)
                        test_bin_cache[tkey2] = test_codes2
                    test_grid = _build_surface_grid(
                        f1,
                        f2,
                        test_codes1,
                        test_codes2,
                        edges1,
                        levels1,
                        edges2,
                        levels2,
                        np.asarray(test_y, dtype=np.float64),
                        np.asarray(test_exposure, dtype=np.float64),
                        np.asarray(test_mu, dtype=np.float64) if test_mu is not None else None,
                    )

            slot = _find_termslot_for_pair(model, f1, f2) if model is not None else None
            in_model = slot is not None
            representation = _representation_from_slot(slot)

            coefficients = (
                _extract_block_coefficients(slot, params, bse, self.feature_names, self.link)
                if in_model
                else None
            )
            significance = (
                _compute_block_significance(slot, params, bse, bread_matrix) if in_model else None
            )
            gvif = (
                _compute_block_gvif(correlation_matrix, slot.col_start, slot.col_end)
                if in_model
                else None
            )

            out.append(
                InteractionDiagnostics(
                    name=f"{f1}:{f2}",
                    factor1=f1,
                    factor2=f2,
                    pair_type=pair_type,
                    in_model=in_model,
                    representation=representation,
                    train_surface_grid=train_grid,
                    test_surface_grid=test_grid,
                    coefficients=coefficients,
                    significance=significance,
                    score_test=None,  # score test for unfitted pairs is a v2 follow-up
                    gvif=gvif,
                )
            )
        return out

    def compute_pair_exploration(
        self,
        pairs: list[Any],
        data: pl.DataFrame,
        binning_cont_cont: tuple[int, int] = DEFAULT_PAIR_CONT_CONT,
        binning_cont_cat: tuple[int, int] = DEFAULT_PAIR_CONT_CAT,
        binning_cat_cat: tuple[int, int] = DEFAULT_PAIR_CAT_CAT,
        max_total_cells: int = DEFAULT_PAIR_MAX_CELLS,
    ) -> list[InteractionExploration]:
        """Compute pre-fit ``InteractionExploration`` for each pair.

        No model is required. Cells carry only ``actual`` (no ``predicted``
        or ``ae_ratio``). ``interaction_strength`` is partial R² vs.
        multiplicative null.
        """
        out: list[InteractionExploration] = []
        for spec in pairs:
            f1_raw, f2_raw = _normalize_pair_spec(spec)
            for col in (f1_raw, f2_raw):
                if col not in data.columns:
                    raise ValueError(f"Interaction factor {col!r} is not a column in data.")
            pair_type, f1, f2 = _detect_pair_type(data, f1_raw, f2_raw)

            card1 = data[f1].n_unique() if pair_type != "continuous_x_continuous" else None
            card2 = (
                data[f2].n_unique()
                if pair_type
                in (
                    "categorical_x_categorical",
                    "continuous_x_categorical",
                )
                else None
            )
            n1, n2 = _choose_bin_counts(
                pair_type,
                binning_cont_cont,
                binning_cont_cat,
                binning_cat_cat,
                max_total_cells,
                card1,
                card2,
            )

            codes1, edges1, levels1 = _bin_factor(data, f1, "row", pair_type, n1, self.exposure)
            codes2, edges2, levels2 = _bin_factor(data, f2, "col", pair_type, n2, self.exposure)
            grid = _build_surface_grid(
                f1,
                f2,
                codes1,
                codes2,
                edges1,
                levels1,
                edges2,
                levels2,
                self.y,
                self.exposure,
                mu=None,  # pre-fit: no predictions
            )
            # Compute interaction strength via the same Rust function the
            # auto-detector uses (exploration.rs:367). No Python-side formula.
            n_levels1 = (edges1.size - 1) if edges1 is not None else len(levels1 or [])
            n_levels2 = (edges2.size - 1) if edges2 is not None else len(levels2 or [])
            strength = _rust_interaction_strength_from_codes(
                np.ascontiguousarray(codes1, dtype=np.uint32),
                int(n_levels1),
                np.ascontiguousarray(codes2, dtype=np.uint32),
                int(n_levels2),
                np.ascontiguousarray(self.y, dtype=np.float64),
                np.ascontiguousarray(self.exposure, dtype=np.float64),
                0,  # min_cell_count=0: include all non-empty cells
            )
            out.append(
                InteractionExploration(
                    name=f"{f1}:{f2}",
                    factor1=f1,
                    factor2=f2,
                    pair_type=pair_type,
                    surface_grid=grid,
                    interaction_strength=float(strength),
                )
            )
        return out

    @staticmethod
    def _apply_bins(
        series: pl.Series,
        edges: np.ndarray | None,
        levels: list[str] | None,
    ) -> np.ndarray:
        """Apply train-derived bin definitions to a test column."""
        if edges is not None:
            values = np.asarray(series.to_numpy(), dtype=np.float64)
            return _apply_continuous_edges(values, edges)
        if levels is not None:
            return _apply_categorical_levels(series, levels)
        raise ValueError("Either edges or levels must be provided.")
