"""Rate-table export for fitted RustyStats GLM models."""

from __future__ import annotations

import csv
import math
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from rustystats.exceptions import ValidationError

if TYPE_CHECKING:
    from rustystats.formula import GLMModel
    from rustystats.interactions import TermSlot


def to_rate_tables(
    model: GLMModel,
    path: str | Path | None = None,
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
    """Export a fitted GLM to concise resolved rate tables."""
    if style != "resolved":
        raise ValidationError(
            f"unsupported rate-table style {style!r}; only 'resolved' is supported."
        )
    if format not in ("dict", "stacked_csv"):
        raise ValidationError(
            f"unsupported rate-table format {format!r}; use 'dict' or 'stacked_csv'."
        )
    if spline_strategy not in ("unsupported", "grid", "basis"):
        raise ValidationError("spline_strategy must be 'unsupported', 'grid', or 'basis'.")
    if spline_interpolation != "linear":
        raise ValidationError("spline_interpolation must be 'linear' for spline grid export.")
    if spline_extrapolation not in ("clip", "raise"):
        raise ValidationError("spline_extrapolation must be 'clip' or 'raise'.")

    spline_grid_specs = {} if spline_grids is None else spline_grids
    artifact = _build_artifact(
        model,
        deployment=deployment,
        spline_strategy=spline_strategy,
        spline_grids=spline_grid_specs,
        spline_interpolation=spline_interpolation,
        spline_extrapolation=spline_extrapolation,
        include_components=include_components,
    )
    if format == "stacked_csv":
        if path is None:
            raise ValidationError("path is required when format='stacked_csv'.")
        _write_stacked_csv(artifact, path)
    return artifact


def _build_artifact(
    model: GLMModel,
    *,
    deployment: bool,
    spline_strategy: str,
    spline_grids: dict[str, list[float]],
    spline_interpolation: str,
    spline_extrapolation: str,
    include_components: bool,
) -> dict[str, Any]:
    params = np.asarray(model.params, dtype=np.float64)
    intercept = _intercept_eta(model, params)
    artifact: dict[str, Any] = {
        "schema_version": 1,
        "kind": "rustystats_rate_tables",
        "family": model.family,
        "link": model.link,
        "prediction_space": "response",
        "base": {
            "eta": intercept,
            "rel": _eta_to_rel(model, intercept),
        },
        "tables": [],
    }
    warnings: list[dict[str, str]] = []

    transform_by_output = {spec["output"]: spec for spec in getattr(model, "_input_transforms", [])}
    slots: list[TermSlot] = list(getattr(model._builder, "_term_slots", []))
    for slot in slots:
        if slot.term_type == "intercept":
            continue
        try:
            table = _build_slot_table(
                model,
                slot,
                params,
                transform_by_output,
                spline_strategy=spline_strategy,
                spline_grids=spline_grids,
                spline_interpolation=spline_interpolation,
                spline_extrapolation=spline_extrapolation,
                include_components=include_components,
            )
        except ValidationError:
            if deployment:
                raise
            warnings.append(
                {
                    "term": slot.term_name,
                    "message": "term could not be represented as a resolved deployment table",
                }
            )
            continue
        artifact["tables"].append(table)

    _make_table_names_unique(artifact["tables"])
    artifact["manifest"] = [_manifest_row(table) for table in artifact["tables"]]
    if warnings:
        artifact["warnings"] = warnings
    return artifact


def _build_slot_table(
    model: GLMModel,
    slot: TermSlot,
    params: np.ndarray,
    transform_by_output: dict[str, dict[str, Any]],
    *,
    spline_strategy: str,
    spline_grids: dict[str, list[float]],
    spline_interpolation: str,
    spline_extrapolation: str,
    include_components: bool,
) -> dict[str, Any]:
    if slot.term_type == "linear":
        factor = slot.factors[0] if slot.factors else slot.term_name
        transform = transform_by_output.get(factor)
        if transform is None:
            raise ValidationError(
                f"rate-table export for linear term {slot.term_name!r} requires a finite "
                "lookup transform or explicit grid."
            )
        if transform["output_dtype"] != "float64":
            raise ValidationError(
                f"linear term {slot.term_name!r} depends on non-numeric transform "
                f"{transform['name']!r}."
            )
        return _lookup_linear_table(model, slot, params, transform, include_components)

    if slot.term_type in ("categorical", "categorical_indicator"):
        factor = slot.factors[0] if slot.factors else slot.term_name
        transform = transform_by_output.get(factor)
        if transform is not None:
            if transform["output_dtype"] != "string":
                raise ValidationError(
                    f"categorical term {slot.term_name!r} depends on non-string transform "
                    f"{transform['name']!r}."
                )
            return _lookup_categorical_table(model, slot, params, transform, include_components)
        return _plain_categorical_table(model, slot, params, include_components)

    if slot.term_type in ("bs", "ns", "ms"):
        if spline_strategy == "unsupported":
            raise ValidationError(
                f"rate-table export does not support spline term {slot.term_name!r} "
                "with spline_strategy='unsupported'."
            )
        if spline_strategy == "grid":
            return _spline_grid_table(
                model,
                slot,
                params,
                spline_grids,
                interpolation=spline_interpolation,
                extrapolation=spline_extrapolation,
                include_components=include_components,
            )
        raise ValidationError(
            f"spline_strategy={spline_strategy!r} is not implemented for term {slot.term_name!r}."
        )

    if slot.term_type == "target_encoding":
        return _target_encoding_table(model, slot, params, include_components)

    if slot.term_type == "frequency_encoding":
        return _frequency_encoding_table(model, slot, params, include_components)

    raise ValidationError(
        f"rate-table export does not yet support {slot.term_type!r} term {slot.term_name!r}."
    )


def _lookup_linear_table(
    model: GLMModel,
    slot: TermSlot,
    params: np.ndarray,
    transform: dict[str, Any],
    include_components: bool,
) -> dict[str, Any]:
    coefficient = float(params[slot.col_start])
    columns = [*transform["sources"], transform["output"], "eta", "rel"]
    if include_components:
        columns.extend(["coefficient", "transform"])
    rows: list[list[Any]] = []
    for key, value in _sorted_lookup_pairs(transform):
        eta = coefficient * float(value)
        row = [*_display_key_values(key), float(value), eta, _eta_to_rel(model, eta)]
        if include_components:
            row.extend([coefficient, transform["name"]])
        rows.append(row)
    default_row = _lookup_default_row(
        model,
        transform,
        coefficient,
        include_components=include_components,
    )
    return {
        "name": _safe_table_name(slot.term_name),
        "term": slot.term_name,
        "kind": "lookup_linear",
        "sources": list(transform["sources"]),
        "columns": columns,
        "rows": rows,
        "default": default_row,
        "metadata": _lookup_policy_metadata(transform),
    }


def _lookup_categorical_table(
    model: GLMModel,
    slot: TermSlot,
    params: np.ndarray,
    transform: dict[str, Any],
    include_components: bool,
) -> dict[str, Any]:
    coef_by_level = _categorical_coefficients(slot, params)
    columns = [*transform["sources"], transform["output"], "eta", "rel"]
    if include_components:
        columns.extend(["design_column", "coefficient", "transform"])
    rows: list[list[Any]] = []
    for key, value in _sorted_lookup_pairs(transform):
        eta = float(coef_by_level.get(str(value), 0.0))
        row = [*_display_key_values(key), str(value), eta, _eta_to_rel(model, eta)]
        if include_components:
            design_col = _design_column_for_level(slot, str(value))
            row.extend([design_col, eta, transform["name"]])
        rows.append(row)
    default_row = _lookup_default_row(
        model,
        transform,
        coef_by_level,
        include_components=include_components,
        slot=slot,
    )
    return {
        "name": _safe_table_name(slot.term_name),
        "term": slot.term_name,
        "kind": "lookup_categorical",
        "sources": list(transform["sources"]),
        "columns": columns,
        "rows": rows,
        "default": default_row,
        "metadata": _lookup_policy_metadata(transform),
    }


def _plain_categorical_table(
    model: GLMModel,
    slot: TermSlot,
    params: np.ndarray,
    include_components: bool,
) -> dict[str, Any]:
    source = slot.factors[0] if slot.factors else slot.term_name
    coef_by_level = _categorical_coefficients(slot, params)
    levels = [str(level) for level in slot.extra.get("levels", [])]
    columns = [source, "eta", "rel"]
    if include_components:
        columns.extend(["design_column", "coefficient"])
    rows: list[list[Any]] = []
    for level in levels:
        eta = float(coef_by_level.get(level, 0.0))
        row = [level, eta, _eta_to_rel(model, eta)]
        if include_components:
            row.extend([_design_column_for_level(slot, level), eta])
        rows.append(row)
    return {
        "name": _safe_table_name(slot.term_name),
        "term": slot.term_name,
        "kind": "categorical",
        "sources": [source],
        "columns": columns,
        "rows": rows,
        "default": ["<unseen>", 0.0, _eta_to_rel(model, 0.0)] if levels else None,
    }


def _lookup_policy_metadata(transform: dict[str, Any]) -> dict[str, Any]:
    return {
        "on_unseen": transform["on_unseen"],
        "on_null": transform["on_null"],
        "default": transform.get("default"),
    }


def _lookup_default_marker(transform: dict[str, Any]) -> str | None:
    unseen_defaults = transform["on_unseen"] == "default"
    null_defaults = transform["on_null"] == "default"
    if unseen_defaults and null_defaults:
        return "<default>"
    if unseen_defaults:
        return "<unseen>"
    if null_defaults:
        return "<null>"
    return None


def _lookup_default_row(
    model: GLMModel,
    transform: dict[str, Any],
    effect: float | dict[str, float],
    *,
    include_components: bool,
    slot: TermSlot | None = None,
) -> list[Any] | None:
    marker = _lookup_default_marker(transform)
    if marker is None or transform.get("default") is None:
        return None
    default_value = transform["default"]
    if isinstance(effect, dict):
        level = str(default_value)
        eta = float(effect.get(level, 0.0))
        row: list[Any] = [
            *[marker for _ in transform["sources"]],
            level,
            eta,
            _eta_to_rel(model, eta),
        ]
        if include_components:
            row.extend(
                [
                    _design_column_for_level(slot, level) if slot is not None else None,
                    eta,
                    transform["name"],
                ]
            )
        return row
    numeric_default = float(default_value)
    eta = float(effect) * numeric_default
    row = [
        *[marker for _ in transform["sources"]],
        numeric_default,
        eta,
        _eta_to_rel(model, eta),
    ]
    if include_components:
        row.extend([float(effect), transform["name"]])
    return row


def _target_encoding_table(
    model: GLMModel,
    slot: TermSlot,
    params: np.ndarray,
    include_components: bool,
) -> dict[str, Any]:
    builder = model._builder
    stats_by_var = getattr(builder, "_te_stats", {})
    var_name = slot.extra.get("var_name")
    if var_name not in stats_by_var:
        raise ValidationError(f"target encoding stats missing for term {slot.term_name!r}.")
    stats = stats_by_var[var_name]
    level_stats = dict(stats["stats"])
    prior = float(stats["prior"])
    prior_weight = float(stats["prior_weight"])
    interaction_vars = stats.get("interaction_vars")
    if interaction_vars is not None and len(interaction_vars) >= 2:
        raise ValidationError(
            f"target encoding interaction term {slot.term_name!r} cannot be exported "
            "as a structured rate table until interaction stats store structured keys."
        )
    sources = list(interaction_vars or [var_name])
    encoded_name = slot.design_column_names[0]
    coefficient = float(params[slot.col_start])
    columns = [*sources, encoded_name, "eta", "rel"]
    if include_components:
        columns.extend(["coefficient", "prior", "prior_weight"])

    rows: list[list[Any]] = []
    for key, level_stat in _sorted_stats_items(level_stats):
        encoded = _target_encoded_value(level_stat, prior, prior_weight)
        eta = coefficient * encoded
        row = [*_split_encoded_key(key, len(sources)), encoded, eta, _eta_to_rel(model, eta)]
        if include_components:
            row.extend([coefficient, prior, prior_weight])
        rows.append(row)

    default_eta = coefficient * prior
    default_row = [
        *["<unseen>" for _ in sources],
        prior,
        default_eta,
        _eta_to_rel(model, default_eta),
    ]
    if include_components:
        default_row.extend([coefficient, prior, prior_weight])
    return {
        "name": _safe_table_name(slot.term_name),
        "term": slot.term_name,
        "kind": "target_encoding",
        "sources": sources,
        "columns": columns,
        "rows": rows,
        "default": default_row,
    }


def _frequency_encoding_table(
    model: GLMModel,
    slot: TermSlot,
    params: np.ndarray,
    include_components: bool,
) -> dict[str, Any]:
    builder = model._builder
    stats_by_var = getattr(builder, "_fe_stats", {})
    var_name = slot.extra.get("var_name")
    if var_name not in stats_by_var:
        raise ValidationError(f"frequency encoding stats missing for term {slot.term_name!r}.")
    stats = stats_by_var[var_name]
    level_counts = dict(stats["level_counts"])
    max_count = float(stats["max_count"])
    interaction_vars = stats.get("interaction_vars")
    if interaction_vars is not None and len(interaction_vars) >= 2:
        raise ValidationError(
            f"frequency encoding interaction term {slot.term_name!r} cannot be exported "
            "as a structured rate table until interaction stats store structured keys."
        )
    sources = list(interaction_vars or [var_name])
    encoded_name = slot.design_column_names[0]
    coefficient = float(params[slot.col_start])
    columns = [*sources, encoded_name, "eta", "rel"]
    if include_components:
        columns.extend(["coefficient", "count", "max_count"])

    rows: list[list[Any]] = []
    for key, count in _sorted_stats_items(level_counts):
        encoded = float(count) / max_count if max_count else 0.0
        eta = coefficient * encoded
        row = [*_split_encoded_key(key, len(sources)), encoded, eta, _eta_to_rel(model, eta)]
        if include_components:
            row.extend([coefficient, count, max_count])
        rows.append(row)

    default_eta = 0.0
    default_row = [
        *["<unseen>" for _ in sources],
        0.0,
        default_eta,
        _eta_to_rel(model, default_eta),
    ]
    if include_components:
        default_row.extend([coefficient, 0, max_count])
    return {
        "name": _safe_table_name(slot.term_name),
        "term": slot.term_name,
        "kind": "frequency_encoding",
        "sources": sources,
        "columns": columns,
        "rows": rows,
        "default": default_row,
    }


def _spline_grid_table(
    model: GLMModel,
    slot: TermSlot,
    params: np.ndarray,
    spline_grids: dict[str, list[float]],
    *,
    interpolation: str,
    extrapolation: str,
    include_components: bool,
) -> dict[str, Any]:
    var = slot.factors[0] if slot.factors else slot.term_name
    if var not in spline_grids:
        raise ValidationError(
            f"spline_strategy='grid' requires spline_grids[{var!r}] for term {slot.term_name!r}."
        )
    grid = _validate_spline_grid(var, spline_grids[var])
    fitted = getattr(model._builder, "_fitted_splines", {}).get(var)
    if fitted is None:
        raise ValidationError(f"fitted spline metadata missing for term {slot.term_name!r}.")

    basis, basis_names = fitted.transform(np.asarray(grid, dtype=np.float64))
    beta = params[slot.col_start : slot.col_end]
    if basis.shape[1] != beta.shape[0]:
        raise ValidationError(
            f"spline term {slot.term_name!r} basis width {basis.shape[1]} does not "
            f"match coefficient width {beta.shape[0]}."
        )
    eta_values = basis @ beta
    columns = [var, "eta", "rel"]
    if include_components:
        columns.extend([f"basis_{i + 1}" for i in range(basis.shape[1])])

    rows: list[list[Any]] = []
    for i, x in enumerate(grid):
        eta = float(eta_values[i])
        row: list[Any] = [float(x), eta, _eta_to_rel(model, eta)]
        if include_components:
            row.extend(float(v) for v in basis[i, :])
        rows.append(row)

    knot_info = fitted.get_knot_info() if hasattr(fitted, "get_knot_info") else {}
    return {
        "name": _safe_table_name(slot.term_name),
        "term": slot.term_name,
        "kind": "spline_grid",
        "sources": [var],
        "columns": columns,
        "rows": rows,
        "default": None,
        "metadata": {
            "approximation": True,
            "spline_strategy": "grid",
            "interpolation": interpolation,
            "extrapolation": extrapolation,
            "grid_min": float(grid[0]),
            "grid_max": float(grid[-1]),
            "grid_size": len(grid),
            "spline_type": slot.term_type,
            "basis_columns": list(basis_names),
            "knot_info": knot_info,
        },
    }


def _validate_spline_grid(var: str, raw_grid: list[float]) -> list[float]:
    if not isinstance(raw_grid, list) or len(raw_grid) < 2:
        raise ValidationError(f"spline grid for {var!r} must contain at least two values.")
    try:
        grid = [float(v) for v in raw_grid]
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"spline grid for {var!r} must contain numeric values.") from exc
    if not all(math.isfinite(v) for v in grid):
        raise ValidationError(f"spline grid for {var!r} must contain finite values.")
    if any(curr <= prev for prev, curr in pairwise(grid)):
        raise ValidationError(f"spline grid for {var!r} must be strictly increasing.")
    return grid


def _target_encoded_value(level_stat: Any, prior: float, prior_weight: float) -> float:
    total, count = level_stat
    return (float(total) + prior_weight * prior) / (float(count) + prior_weight)


def _sorted_stats_items(stats: dict[Any, Any]) -> list[tuple[str, Any]]:
    return sorted(((str(k), v) for k, v in stats.items()), key=lambda kv: kv[0])


def _split_encoded_key(key: str, width: int) -> list[str]:
    if width <= 1:
        return [key]
    parts = key.split(":")
    if len(parts) == width:
        return parts
    return [key, *[""] * (width - 1)]


def _categorical_coefficients(slot: TermSlot, params: np.ndarray) -> dict[str, float]:
    coefs: dict[str, float] = {}
    for name, coef in zip(
        slot.design_column_names,
        params[slot.col_start : slot.col_end],
        strict=True,
    ):
        level = _level_from_design_name(name)
        if level is not None:
            coefs[level] = float(coef)
    return coefs


def _level_from_design_name(name: str) -> str | None:
    marker = "[T."
    if marker not in name or not name.endswith("]"):
        return None
    return name.split(marker, 1)[1][:-1]


def _design_column_for_level(slot: TermSlot, level: str) -> str | None:
    for name in slot.design_column_names:
        if _level_from_design_name(name) == level:
            return name
    return None


def _sorted_lookup_pairs(transform: dict[str, Any]) -> list[tuple[list[Any], Any]]:
    pairs = list(zip(transform["keys"], transform["values"], strict=True))
    return sorted(pairs, key=lambda kv: tuple("" if v is None else str(v) for v in kv[0]))


def _display_key_values(key: list[Any]) -> list[Any]:
    return ["<null>" if value is None else value for value in key]


def _intercept_eta(model: GLMModel, params: np.ndarray) -> float:
    if model.feature_names and model.feature_names[0] == "Intercept":
        return float(params[0])
    return 0.0


def _eta_to_rel(model: GLMModel, eta: float) -> float | None:
    if model.link == "log":
        return float(math.exp(eta))
    return None


def _manifest_row(table: dict[str, Any]) -> dict[str, Any]:
    metadata = table.get("metadata", {})
    return {
        "name": table["name"],
        "term": table["term"],
        "kind": table["kind"],
        "sources": "|".join(table["sources"]),
        "row_count": len(table.get("rows", [])) + (1 if table.get("default") is not None else 0),
        "on_unseen": metadata.get("on_unseen", ""),
        "on_null": metadata.get("on_null", ""),
        "approximation": bool(metadata.get("approximation", False)),
        "interpolation": metadata.get("interpolation", ""),
        "extrapolation": metadata.get("extrapolation", ""),
        "grid_min": metadata.get("grid_min", ""),
        "grid_max": metadata.get("grid_max", ""),
    }


def _write_stacked_csv(artifact: dict[str, Any], path: str | Path) -> None:
    with Path(path).open("w", newline="") as f:
        writer = csv.writer(f)
        _write_block(
            writer,
            "_base",
            ["field", "value"],
            [
                ["schema_version", artifact["schema_version"]],
                ["family", artifact["family"]],
                ["link", artifact["link"]],
                ["base_eta", artifact["base"]["eta"]],
                ["base_rel", artifact["base"]["rel"]],
            ],
            first=True,
        )
        _write_block(
            writer,
            "_manifest",
            [
                "name",
                "term",
                "kind",
                "sources",
                "row_count",
                "on_unseen",
                "on_null",
                "approximation",
                "interpolation",
                "extrapolation",
                "grid_min",
                "grid_max",
            ],
            [
                [
                    row["name"],
                    row["term"],
                    row["kind"],
                    row["sources"],
                    row["row_count"],
                    row["on_unseen"],
                    row["on_null"],
                    row["approximation"],
                    row["interpolation"],
                    row["extrapolation"],
                    row["grid_min"],
                    row["grid_max"],
                ]
                for row in artifact.get("manifest", [])
            ],
        )
        for table in artifact["tables"]:
            rows = list(table.get("rows", []))
            if table.get("default") is not None:
                rows.append(table["default"])
            _write_block(writer, table["name"], list(table["columns"]), rows)
        if artifact.get("warnings"):
            _write_block(
                writer,
                "_warnings",
                ["term", "message"],
                [[row["term"], row["message"]] for row in artifact["warnings"]],
            )


def _write_block(
    writer: csv.writer,
    name: str,
    columns: list[str],
    rows: list[list[Any]],
    *,
    first: bool = False,
) -> None:
    if not first:
        writer.writerow([])
    writer.writerow([name])
    writer.writerow(columns)
    writer.writerows(rows)


def _safe_table_name(term: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in term).strip("_")
    return safe or "term"


def _make_table_names_unique(tables: list[dict[str, Any]]) -> None:
    # Global uniqueness: a generated ``{base}_{n}`` must not collide with a
    # genuine table whose own sanitized name is already ``{base}_{n}``.
    assigned: set[str] = set()
    for table in tables:
        base = table["name"]
        candidate = base
        count = 1
        while candidate in assigned:
            count += 1
            candidate = f"{base}_{count}"
        table["name"] = candidate
        assigned.add(candidate)
