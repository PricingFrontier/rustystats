"""Per-prediction contribution decomposition for fitted GLMs.

For any fitted GLMModel, ``predict_contributions`` decomposes each row's
prediction as ``base_value + sum(term_contributions) == linear predictor`` and
``inverse_link(linear predictor) == prediction``. Used by downstream tracing
(e.g. insurance pricing engines) to render a contribution ladder per row.

Design columns (spline bases, categorical dummies, target/frequency encoding,
interaction tensor products) are grouped back to their source term so the
ladder shows factor-level rows rather than basis-level rows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from rustystats.exceptions import PredictionError, ValidationError
from rustystats.interactions import TermSlot

if TYPE_CHECKING:
    import polars as pl

    from rustystats.formula import GLMModel


def compute_contributions(
    model: GLMModel,
    new_data: pl.DataFrame | pl.LazyFrame,
    *,
    offset: str | np.ndarray | None = None,
    exposure: str | np.ndarray | None = None,
    complement: str | np.ndarray | GLMModel | None = None,
    group_terms: bool = True,
    include_design_columns: bool = False,
    return_format: Literal["records", "dataframe"] = "records",
    validate: bool = True,
    atol: float = 1e-9,
    rtol: float = 1e-7,
) -> list[dict] | pl.DataFrame:
    """Decompose model predictions into per-term contributions.

    See :meth:`GLMModel.predict_contributions` for full documentation.
    """
    from rustystats.formula import (
        _collect_lazyframe,
        _extract_needed_columns,
        _resolve_predict_complement,
        _resolve_predict_exposure,
        _resolve_predict_offset,
        apply_inverse_link,
    )

    if model._builder is None:
        raise PredictionError(
            "Cannot compute contributions: model was not fitted with the formula API."
        )

    slots = list(model._builder._term_slots)

    if return_format not in ("records", "dataframe"):
        raise ValueError(f"return_format must be 'records' or 'dataframe', got {return_format!r}")

    exposure_to_use = exposure if exposure is not None else model._exposure_spec
    # Mirror predict(): an array exposure is fit-time data and must not be
    # silently reused as a prediction default (it would produce a wrong
    # contribution ladder whenever new_data is not row-aligned with the fit).
    if exposure is None and getattr(model, "_array_exposure_requires_prediction_override", False):
        raise PredictionError(
            "This model was fit with an array exposure, which is fit-time data and "
            "cannot be reused as a prediction default. Pass exposure= for the "
            "prediction data, or fit with exposure='<column>' so the column can be "
            "resolved from new_data."
        )
    offset_to_use = offset if offset is not None else model._offset_spec
    complement_to_use = complement if complement is not None else model._complement_spec

    if model._terms_dict is not None:
        needed = _extract_needed_columns(
            terms=model._terms_dict,
            interactions=model._interactions_spec,
            offset=offset_to_use,
            exposure=exposure_to_use,
            complement=complement_to_use,
        )
        new_data = _collect_lazyframe(new_data, needed)
    else:
        new_data = _collect_lazyframe(new_data, set())

    n_rows = len(new_data)
    params = np.asarray(model.params, dtype=np.float64)
    _validate_term_slots(slots, params.size)

    X_new = model._builder.transform_new_data(new_data)

    exposure_raw = None
    exposure_link = None
    exposure_name = None
    if exposure_to_use is not None:
        if model.link != "log":
            raise ValidationError("exposure= is only meaningful for log-link rate models.")
        exposure_raw, exposure_link, exposure_name = _resolve_predict_exposure(
            new_data, exposure_to_use
        )

    offset_raw, offset_link, offset_name = _resolve_predict_offset(
        new_data, offset, model._offset_spec
    )
    _, complement_link = _resolve_predict_complement(
        new_data,
        complement,
        model._complement_spec,
        exposure_to_use,
        model.link,
    )

    # --- base_value (per-row) and intercept routing ----------------------------
    intercept_slot = next((s for s in slots if s.term_type == "intercept"), None)
    has_complement = complement_link is not None

    if has_complement:
        # base = link(complement_for_row); intercept becomes a regular contribution row
        base_value = complement_link.astype(np.float64, copy=True)
        intercept_in_contribs = intercept_slot is not None
    elif intercept_slot is not None:
        base_value = np.full(n_rows, float(params[intercept_slot.col_start]))
        intercept_in_contribs = False
    else:
        base_value = np.zeros(n_rows, dtype=np.float64)
        intercept_in_contribs = False

    # --- per-term contributions ------------------------------------------------
    emitted_slots: list[TermSlot] = []
    contribs_per_slot: list[np.ndarray] = []
    feature_values_per_slot: list[Any] = []  # ndarray (scalar) or list (compound)

    for slot in slots:
        if slot.term_type == "intercept" and not intercept_in_contribs:
            continue
        if group_terms or slot.col_end - slot.col_start <= 1 or slot.term_type == "interaction":
            block = X_new[:, slot.col_start : slot.col_end]
            beta_block = params[slot.col_start : slot.col_end]
            contrib = block @ beta_block
            emitted_slots.append(slot)
            contribs_per_slot.append(contrib)
            feature_values_per_slot.append(_extract_feature_values(slot, new_data, X_new))
        else:
            # group_terms=False: expand each design column into its own row
            for j in range(slot.col_start, slot.col_end):
                col_name = slot.design_column_names[j - slot.col_start]
                child = TermSlot(
                    term_name=col_name,
                    term_type=f"{slot.term_type}_column",
                    factors=list(slot.factors),
                    col_start=j,
                    col_end=j + 1,
                    design_column_names=[col_name],
                    extra={"parent_term": slot.term_name, **slot.extra},
                )
                emitted_slots.append(child)
                contribs_per_slot.append(X_new[:, j] * params[j])
                feature_values_per_slot.append(X_new[:, j].astype(np.float64, copy=False))

    # --- exposure / offset contribution rows (synthetic; not in term_slots) ----
    # The two are distinguished by ``term_name``/``term_type`` so consumers
    # don't need to rely on positional order. After RS-ACT-002b the model
    # carries them on separate specs (``_exposure_spec`` vs ``_offset_spec``),
    # so a single ladder may contain both rows when callers pass both.
    if exposure_link is not None:
        exposure_slot = TermSlot(
            term_name="exposure",
            term_type="exposure",
            factors=[exposure_name] if exposure_name else [],
            col_start=-1,
            col_end=-1,
            design_column_names=[],
            extra={
                "raw": exposure_raw,
                "name": exposure_name,
            },
        )
        emitted_slots.append(exposure_slot)
        contribs_per_slot.append(exposure_link)
        feature_values_per_slot.append(exposure_raw)

    offset_slot: TermSlot | None = None
    if offset_link is not None:
        offset_slot = TermSlot(
            term_name="offset",
            term_type="offset",
            factors=[offset_name] if offset_name else [],
            col_start=-1,
            col_end=-1,
            design_column_names=[],
            extra={
                "raw": offset_raw,
                "name": offset_name,
            },
        )
        emitted_slots.append(offset_slot)
        contribs_per_slot.append(offset_link)
        feature_values_per_slot.append(offset_raw if offset_raw is not None else offset_link)

    # --- additivity validation -------------------------------------------------
    if contribs_per_slot:
        contribs_matrix = np.stack(contribs_per_slot, axis=1)  # (n_rows, n_terms)
    else:
        contribs_matrix = np.zeros((n_rows, 0), dtype=np.float64)

    sum_contribs = contribs_matrix.sum(axis=1) if contribs_matrix.size else np.zeros(n_rows)
    eta_from_contribs = base_value + sum_contribs
    mu_from_contribs = apply_inverse_link(eta_from_contribs, model.link)

    if validate:
        eta_actual = X_new @ params
        if exposure_link is not None:
            eta_actual = eta_actual + exposure_link
        if offset_link is not None:
            eta_actual = eta_actual + offset_link
        if complement_link is not None:
            eta_actual = eta_actual + complement_link
        mu_actual = apply_inverse_link(eta_actual, model.link)

        eta_delta = np.abs(eta_from_contribs - eta_actual)
        mu_delta = np.abs(mu_from_contribs - mu_actual)
        eta_tol = atol + rtol * np.abs(eta_actual)
        mu_tol = atol + rtol * np.abs(mu_actual)
        bad = (eta_delta > eta_tol) | (mu_delta > mu_tol)
        if bad.any():
            worst = int(np.argmax(eta_delta))
            raise PredictionError(
                f"predict_contributions failed additivity check at "
                f"{int(bad.sum())} of {n_rows} row(s). "
                f"Worst row {worst}: |base + sum(contribs) - eta| = "
                f"{float(eta_delta[worst]):.3e}, "
                f"|mu_from_contribs - mu_predict| = {float(mu_delta[worst]):.3e}. "
                f"This indicates a RustyStats bug; please report with the model "
                f"and input data."
            )

    output_space = "response" if model.link == "identity" else "linear_predictor"
    prediction_space = "response"

    if return_format == "records":
        return _build_records(
            model=model,
            slots=emitted_slots,
            contribs_matrix=contribs_matrix,
            feature_values_per_slot=feature_values_per_slot,
            base_value=base_value,
            sum_contribs=sum_contribs,
            eta_from_contribs=eta_from_contribs,
            mu_from_contribs=mu_from_contribs,
            output_space=output_space,
            prediction_space=prediction_space,
            X_new=X_new,
            params=params,
            include_design_columns=include_design_columns,
        )
    return _build_dataframe(
        model=model,
        slots=emitted_slots,
        contribs_matrix=contribs_matrix,
        feature_values_per_slot=feature_values_per_slot,
        base_value=base_value,
        sum_contribs=sum_contribs,
        eta_from_contribs=eta_from_contribs,
        mu_from_contribs=mu_from_contribs,
        output_space=output_space,
        prediction_space=prediction_space,
    )


def _validate_term_slots(slots: list[TermSlot], n_params: int) -> None:
    """Ensure serialized term-slot metadata can explain every coefficient."""
    if n_params == 0:
        return
    if not slots:
        raise PredictionError(
            "Cannot compute contributions: this model lacks term-slot metadata. "
            "Refit the model or report this as a RustyStats serialization bug."
        )

    covered = np.zeros(n_params, dtype=bool)
    for slot in slots:
        if slot.col_start < 0:
            continue
        if slot.col_start > slot.col_end or slot.col_end > n_params:
            raise PredictionError(
                "Cannot compute contributions: term-slot metadata is inconsistent "
                f"with model coefficients for term {slot.term_name!r}."
            )
        covered[slot.col_start : slot.col_end] = True

    if not covered.all():
        missing = np.flatnonzero(~covered).tolist()
        raise PredictionError(
            "Cannot compute contributions: term-slot metadata does not cover "
            f"coefficient index/indices {missing}."
        )


# ---------------------------------------------------------------------------
# Feature-value extraction (one entry per term, indexed by row)
# ---------------------------------------------------------------------------


def _extract_feature_values(slot: TermSlot, new_data: pl.DataFrame, X_new: np.ndarray) -> Any:
    """Return per-row feature values for a term.

    Returns an ndarray for scalar values; a list for compound (dict) values.
    """
    import polars as pl

    n = len(new_data)

    if slot.term_type == "intercept":
        return np.ones(n, dtype=np.int64)

    if slot.term_type in ("linear", "constraint"):
        # Use the underlying variable when available; fall back to the design
        # column for I()-wrapped constraints.
        var = slot.factors[0] if slot.factors else None
        if var is not None and var in new_data.columns:
            return new_data[var].to_numpy()
        return X_new[:, slot.col_start].astype(np.float64, copy=False)

    if slot.term_type == "categorical":
        return new_data[slot.factors[0]].cast(pl.Utf8).to_numpy()

    if slot.term_type in ("bs", "ns", "ms"):
        return new_data[slot.factors[0]].to_numpy()

    if slot.term_type in ("target_encoding", "frequency_encoding"):
        encoded = X_new[:, slot.col_start]
        ivars = slot.extra.get("interaction_vars")
        if ivars:
            raw_strs = new_data.select(
                pl.concat_str([pl.col(v).cast(pl.Utf8) for v in ivars], separator=":").alias(
                    "__rs_raw__"
                )
            )["__rs_raw__"].to_list()
        else:
            raw_strs = new_data[slot.extra["var_name"]].cast(pl.Utf8).to_list()
        return [{"raw": r, "encoded": float(e)} for r, e in zip(raw_strs, encoded)]

    if slot.term_type == "expression":
        values = X_new[:, slot.col_start]
        expr = slot.extra.get("expression", "")
        return [{"expr": expr, "value": float(v)} for v in values]

    if slot.term_type == "interaction":
        cat_flags = slot.extra.get("categorical_flags", [False] * len(slot.factors))
        factor_items: list[tuple[str, list]] = []
        for factor, is_cat in zip(slot.factors, cat_flags):
            key, values = _extract_interaction_factor_values(factor, bool(is_cat), new_data)
            factor_items.append((key, values))
        return [{key: values[i] for key, values in factor_items} for i in range(n)]

    if slot.term_type == "categorical_indicator":
        return new_data[slot.factors[0]].cast(pl.Utf8).to_numpy()

    if slot.term_type in ("offset", "exposure"):
        raw = slot.extra.get("raw")
        if raw is not None:
            return np.asarray(raw)
        return np.ones(n)

    # Fallback for ungrouped/_column expansions: emit the design-matrix value.
    if slot.col_start >= 0:
        return X_new[:, slot.col_start].astype(np.float64, copy=False)
    return np.zeros(n)


def _extract_interaction_factor_values(
    factor: str,
    is_categorical: bool,
    new_data: pl.DataFrame,
) -> tuple[str, list]:
    """Return display key and per-row values for one interaction factor."""
    import polars as pl

    if factor.startswith("TE(") and factor.endswith(")"):
        raw_factor = factor[3:-1]
        if raw_factor in new_data.columns:
            return raw_factor, new_data[raw_factor].cast(pl.Utf8).to_list()

        raw_parts = raw_factor.split(":")
        if raw_parts and all(part in new_data.columns for part in raw_parts):
            combined = new_data.select(
                pl.concat_str(
                    [pl.col(part).cast(pl.Utf8) for part in raw_parts],
                    separator=":",
                ).alias("__rs_te_factor__")
            )["__rs_te_factor__"].to_list()
            return raw_factor, combined

    if is_categorical:
        return factor, new_data[factor].cast(pl.Utf8).to_list()
    return factor, new_data[factor].to_list()


# ---------------------------------------------------------------------------
# Output formats
# ---------------------------------------------------------------------------


def _row_value(fv: Any, i: int) -> Any:
    """Index into a per-slot feature_values bag at row i."""
    if isinstance(fv, np.ndarray):
        v = fv[i]
        # Convert numpy scalars to native Python types for clean dict output
        if isinstance(v, np.floating):
            return float(v)
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.bool_):
            return bool(v)
        return v
    if isinstance(fv, list):
        return fv[i]
    return fv


def _build_records(
    *,
    model: GLMModel,
    slots: list[TermSlot],
    contribs_matrix: np.ndarray,
    feature_values_per_slot: list[Any],
    base_value: np.ndarray,
    sum_contribs: np.ndarray,
    eta_from_contribs: np.ndarray,
    mu_from_contribs: np.ndarray,
    output_space: str,
    prediction_space: str,
    X_new: np.ndarray,
    params: np.ndarray,
    include_design_columns: bool,
) -> list[dict]:
    n_rows = base_value.shape[0]
    n_terms = len(slots)

    if n_terms == 0:
        ranks = np.zeros((n_rows, 0), dtype=np.int64)
    else:
        # Rank by |contribution| descending; stable tie-break to keep slot order
        # for ties.
        order = np.argsort(-np.abs(contribs_matrix), axis=1, kind="stable")
        ranks = np.empty_like(order)
        for i in range(n_rows):
            ranks[i, order[i]] = np.arange(n_terms) + 1

    records: list[dict] = []
    for i in range(n_rows):
        row_contribs: list[dict] = []
        for k, slot in enumerate(slots):
            fv = feature_values_per_slot[k]
            entry = {
                "term": slot.term_name,
                "term_type": slot.term_type,
                "feature": slot.term_name,
                "feature_value": _row_value(fv, i),
                "contribution": float(contribs_matrix[i, k]),
                "rank": int(ranks[i, k]),
            }
            if include_design_columns and slot.col_start >= 0 and slot.col_end > slot.col_start:
                entry["design_columns"] = [
                    {
                        "column": slot.design_column_names[j - slot.col_start],
                        "basis_value": float(X_new[i, j]),
                        "coefficient": float(params[j]),
                        "contribution": float(X_new[i, j] * params[j]),
                    }
                    for j in range(slot.col_start, slot.col_end)
                ]
            row_contribs.append(entry)
        records.append(
            {
                "family": model.family,
                "link": model.link,
                "output_space": output_space,
                "prediction_space": prediction_space,
                "base_value": float(base_value[i]),
                "sum_contributions": float(sum_contribs[i]),
                "prediction_from_contributions": float(eta_from_contribs[i]),
                "prediction_value": float(mu_from_contribs[i]),
                "contributions": row_contribs,
            }
        )
    return records


def _build_dataframe(
    *,
    model: GLMModel,
    slots: list[TermSlot],
    contribs_matrix: np.ndarray,
    feature_values_per_slot: list[Any],
    base_value: np.ndarray,
    sum_contribs: np.ndarray,
    eta_from_contribs: np.ndarray,
    mu_from_contribs: np.ndarray,
    output_space: str,
    prediction_space: str,
) -> pl.DataFrame:
    """Long-format contribution dataframe (one row per (input_row, term))."""
    import polars as pl

    n_rows = base_value.shape[0]
    n_terms = len(slots)

    if n_terms == 0:
        return pl.DataFrame(
            {
                "row_index": np.arange(n_rows, dtype=np.int64),
                "base_value": base_value,
                "sum_contributions": sum_contribs,
                "prediction_from_contributions": eta_from_contribs,
                "prediction_value": mu_from_contribs,
                "output_space": [output_space] * n_rows,
                "prediction_space": [prediction_space] * n_rows,
                "family": [model.family] * n_rows,
                "link": [model.link] * n_rows,
            }
        )

    order = np.argsort(-np.abs(contribs_matrix), axis=1, kind="stable")
    ranks = np.empty_like(order)
    for i in range(n_rows):
        ranks[i, order[i]] = np.arange(n_terms) + 1

    # Stringify feature values per slot once, then tile across rows.
    feature_value_str_per_slot: list[list[str]] = []
    for fv in feature_values_per_slot:
        if isinstance(fv, np.ndarray):
            feature_value_str_per_slot.append([str(_row_value(fv, i)) for i in range(n_rows)])
        elif isinstance(fv, list):
            feature_value_str_per_slot.append([str(fv[i]) for i in range(n_rows)])
        else:
            feature_value_str_per_slot.append([str(fv)] * n_rows)

    # Flatten in (row, term) order
    row_index = np.repeat(np.arange(n_rows, dtype=np.int64), n_terms)
    term = [slots[k].term_name for _ in range(n_rows) for k in range(n_terms)]
    term_type = [slots[k].term_type for _ in range(n_rows) for k in range(n_terms)]
    feature_value = [
        feature_value_str_per_slot[k][i] for i in range(n_rows) for k in range(n_terms)
    ]
    contribution = contribs_matrix.reshape(-1)
    rank = ranks.reshape(-1)
    base_col = np.repeat(base_value, n_terms)
    sum_col = np.repeat(sum_contribs, n_terms)
    eta_col = np.repeat(eta_from_contribs, n_terms)
    pred_col = np.repeat(mu_from_contribs, n_terms)

    return pl.DataFrame(
        {
            "row_index": row_index,
            "term": term,
            "term_type": term_type,
            "feature": term,
            "feature_value": feature_value,
            "contribution": contribution,
            "rank": rank,
            "base_value": base_col,
            "sum_contributions": sum_col,
            "prediction_from_contributions": eta_col,
            "prediction_value": pred_col,
            "output_space": [output_space] * (n_rows * n_terms),
            "prediction_space": [prediction_space] * (n_rows * n_terms),
            "family": [model.family] * (n_rows * n_terms),
            "link": [model.link] * (n_rows * n_terms),
        }
    )
