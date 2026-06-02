"""Deterministic input transforms for dict-first GLM models.

The first supported transform is a vectorized lookup over one or more raw
source columns. Specs remain plain dictionaries for serialization; compiled
objects cache Polars lookup frames for fast repeated scoring.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import polars as pl

from rustystats.exceptions import PredictionError, ValidationError

_MATCH_ID = "__rustystats_input_transform_match__"


@dataclass(frozen=True)
class CompiledLookupTransform:
    """Compiled lookup transform.

    The lookup frame is rebuilt per application (see :func:`_build_lookup_frame`)
    because the source-key strings must be normalized against the *actual* dtype
    of each source column in the data being scored, exactly mirroring the
    ``pl.col(source).cast(Utf8)`` normalization applied to that column. Baking a
    dtype-independent frame at compile time silently mismatched numeric source
    columns (e.g. spec key ``10`` vs Polars ``Float64`` ``10.0`` -> ``"10.0"``).
    """

    spec: dict[str, Any]
    normalized_sources: tuple[str, ...]

    @property
    def name(self) -> str:
        return str(self.spec["name"])

    @property
    def sources(self) -> list[str]:
        return list(self.spec["sources"])

    @property
    def output(self) -> str:
        return str(self.spec["output"])


CompiledInputTransform = CompiledLookupTransform


def validate_input_transforms(
    specs: list[dict[str, Any]] | None,
    data_schema: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Validate and canonicalize transform specs.

    Parameters
    ----------
    specs:
        Input transform dictionaries. ``None`` is treated as an empty list.
    data_schema:
        Optional mapping of available input columns used for fit-time checks.
    """
    if specs is None:
        return []
    if not isinstance(specs, list):
        raise ValidationError("input_transforms must be a list of dictionaries.")

    available = set(data_schema.keys()) if data_schema is not None else None
    produced: set[str] = set()
    names: set[str] = set()
    outputs: set[str] = set()
    canonical: list[dict[str, Any]] = []

    for i, raw in enumerate(specs):
        if not isinstance(raw, dict):
            raise ValidationError(f"input_transforms[{i}] must be a dictionary.")
        transform_type = raw.get("type")
        if transform_type != "lookup":
            raise ValidationError(
                f"input_transforms[{i}] has unsupported type {transform_type!r}; "
                "only 'lookup' is supported."
            )

        name = raw.get("name")
        if not isinstance(name, str) or not name:
            raise ValidationError(f"input_transforms[{i}] requires a non-empty string 'name'.")
        if name in names:
            raise ValidationError(f"duplicate input transform name {name!r}.")
        names.add(name)

        sources = raw.get("sources")
        if (
            not isinstance(sources, list)
            or not sources
            or not all(isinstance(s, str) and s for s in sources)
        ):
            raise ValidationError(f"input transform {name!r} requires non-empty string sources.")

        output = raw.get("output")
        if not isinstance(output, str) or not output:
            raise ValidationError(f"input transform {name!r} requires a non-empty string output.")
        if output in outputs:
            raise ValidationError(f"duplicate input transform output {output!r}.")
        outputs.add(output)

        replace_existing = bool(raw.get("replace_existing", False))
        if not replace_existing and output in sources:
            raise ValidationError(
                f"input transform {name!r} output {output!r} collides with a source column; "
                "set replace_existing=True to allow replacement."
            )
        if available is not None:
            missing = [s for s in sources if s not in available and s not in produced]
            if missing:
                raise ValidationError(
                    f"input transform {name!r} requires missing source column(s): {missing}."
                )
            if not replace_existing and output in available:
                raise ValidationError(
                    f"input transform {name!r} output {output!r} already exists in data; "
                    "set replace_existing=True to allow replacement."
                )

        output_dtype = raw.get("output_dtype")
        if output_dtype not in ("float64", "string"):
            raise ValidationError(
                f"input transform {name!r} output_dtype must be 'float64' or 'string'."
            )

        keys = raw.get("keys")
        values = raw.get("values")
        if not isinstance(keys, list) or not isinstance(values, list):
            raise ValidationError(f"input transform {name!r} requires list 'keys' and 'values'.")
        if len(keys) != len(values):
            raise ValidationError(
                f"input transform {name!r} requires len(keys) == len(values); "
                f"got {len(keys)} and {len(values)}."
            )
        key_width = len(sources)
        canonical_keys: list[list[str | float | None]] = []
        seen_keys: set[tuple[tuple[str, Any], ...]] = set()
        for j, key in enumerate(keys):
            if not isinstance(key, list) or len(key) != key_width:
                raise ValidationError(
                    f"input transform {name!r} key {j} must be a list of length {key_width}."
                )
            raw_key = [_validate_key_value(name, v, j) for v in key]
            dedup_key = tuple(_dedup_key_value(v) for v in raw_key)
            if dedup_key in seen_keys:
                raise ValidationError(
                    f"input transform {name!r} contains duplicate key {raw_key!r}."
                )
            seen_keys.add(dedup_key)
            canonical_keys.append(list(raw_key))

        canonical_values = [_validate_value(name, output_dtype, v, "values") for v in values]

        on_unseen = raw.get("on_unseen", "default")
        if on_unseen not in ("default", "raise"):
            raise ValidationError(
                f"input transform {name!r} on_unseen must be 'default' or 'raise'."
            )
        on_null = raw.get("on_null", "default")
        if on_null not in ("default", "raise", "match"):
            raise ValidationError(
                f"input transform {name!r} on_null must be 'default', 'raise', or 'match'."
            )

        default = raw.get("default")
        if on_unseen == "default" or on_null == "default":
            if "default" not in raw:
                raise ValidationError(
                    f"input transform {name!r} requires default when unseen/null policy is default."
                )
            default = _validate_value(name, output_dtype, default, "default")
        elif "default" in raw and raw["default"] is not None:
            default = _validate_value(name, output_dtype, default, "default")
        else:
            default = None

        source_cast = raw.get("source_cast", "string")
        if source_cast != "string":
            raise ValidationError(
                f"input transform {name!r} source_cast={source_cast!r} is not supported; "
                "only 'string' is supported."
            )

        metadata = raw.get("metadata", {})
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise ValidationError(f"input transform {name!r} metadata must be a dictionary.")

        canonical.append(
            {
                "type": "lookup",
                "name": name,
                "sources": list(sources),
                "output": output,
                "output_dtype": output_dtype,
                "keys": canonical_keys,
                "values": canonical_values,
                "default": default,
                "on_unseen": on_unseen,
                "on_null": on_null,
                "source_cast": source_cast,
                "replace_existing": replace_existing,
                "metadata": dict(metadata),
            }
        )
        produced.add(output)

    return canonical


def compile_input_transforms(
    specs: list[dict[str, Any]] | None,
    *,
    assume_validated: bool = False,
) -> list[CompiledInputTransform]:
    """Compile canonical transform specs into reusable lookup frames."""
    canonical = list(specs) if assume_validated and specs is not None else []
    if not assume_validated:
        canonical = validate_input_transforms(specs)
    compiled: list[CompiledInputTransform] = []
    for spec in canonical:
        compiled.append(_compile_lookup(spec))
    return compiled


def input_transform_source_columns(specs: list[dict[str, Any]] | None) -> set[str]:
    """Return source columns required by transform specs."""
    canonical = validate_input_transforms(specs)
    return {source for spec in canonical for source in spec["sources"]}


def input_transform_output_columns(specs: list[dict[str, Any]] | None) -> set[str]:
    """Return columns produced by transform specs."""
    canonical = validate_input_transforms(specs)
    return {spec["output"] for spec in canonical}


def apply_input_transforms(
    data: pl.DataFrame | pl.LazyFrame,
    compiled_or_specs: list[CompiledInputTransform] | list[dict[str, Any]] | None,
) -> pl.DataFrame:
    """Apply transforms and return an eager Polars DataFrame."""
    if isinstance(data, pl.LazyFrame):
        data = data.collect()
    compiled = _ensure_compiled(compiled_or_specs)
    out = data
    for transform in compiled:
        out = _apply_lookup(out, transform)
    return out


def apply_input_transforms_lazy(
    lf: pl.LazyFrame,
    compiled_or_specs: list[CompiledInputTransform] | list[dict[str, Any]] | None,
) -> pl.LazyFrame:
    """Apply transforms to a LazyFrame.

    This returns a LazyFrame for API convenience, but intentionally routes via
    the eager implementation so null/unseen/collision errors are identical to
    prediction. Fully lazy scoring can be added later without weakening those
    semantics.
    """
    compiled = _ensure_compiled(compiled_or_specs)
    return apply_input_transforms(lf.collect(), compiled).lazy()


def _ensure_compiled(
    compiled_or_specs: list[CompiledInputTransform] | list[dict[str, Any]] | None,
) -> list[CompiledInputTransform]:
    if not compiled_or_specs:
        return []
    first = compiled_or_specs[0]
    if isinstance(first, CompiledLookupTransform):
        return list(compiled_or_specs)  # type: ignore[arg-type]
    return compile_input_transforms(compiled_or_specs)  # type: ignore[arg-type]


def _compile_lookup(spec: dict[str, Any]) -> CompiledLookupTransform:
    sources = list(spec["sources"])
    normalized_sources = tuple(
        _normalized_col_name(spec["name"], s, i) for i, s in enumerate(sources)
    )
    return CompiledLookupTransform(spec=dict(spec), normalized_sources=normalized_sources)


def _build_lookup_frame(
    data: pl.DataFrame,
    transform: CompiledLookupTransform,
    normalized_sources: list[str],
    match_id: str,
) -> pl.DataFrame:
    """Build the join frame, normalizing spec keys against each source's dtype.

    Each source-key column is cast to the *actual* dtype of its source column
    and then to ``Utf8``, identical to the ``pl.col(source).cast(Utf8)``
    normalization applied to the data side. This guarantees a numeric spec key
    (e.g. ``10``) matches a numeric source value (e.g. ``Float64`` ``10.0``)
    instead of silently falling through to the default.
    """
    spec = transform.spec
    keys = spec["keys"]
    n = len(keys)
    cols: dict[str, pl.Series] = {}
    for col_idx, (source, norm) in enumerate(
        zip(transform.sources, normalized_sources, strict=True)
    ):
        dtype = data.schema[source]
        raw_vals = [keys[row][col_idx] for row in range(n)]
        cols[norm] = pl.Series(norm, raw_vals).cast(dtype, strict=False).cast(pl.Utf8)
    cols[transform.output] = pl.Series(transform.output, list(spec["values"]))
    cols[match_id] = pl.Series(match_id, list(range(n)), dtype=pl.Int64)
    return pl.DataFrame(cols)


def _apply_lookup(data: pl.DataFrame, transform: CompiledLookupTransform) -> pl.DataFrame:
    spec = transform.spec
    _validate_sources_present(data, transform)

    output = transform.output
    if output in data.columns and not bool(spec["replace_existing"]):
        raise PredictionError(
            f"input transform {transform.name!r} output {output!r} already exists in data."
        )

    reserved = [*data.columns, output]
    normalized_sources: list[str] = []
    for base in transform.normalized_sources:
        normalized_sources.append(_unique_temp_name([*reserved, *normalized_sources], base))
    match_id = _unique_temp_name([*reserved, *normalized_sources], _MATCH_ID)
    lookup_frame = _build_lookup_frame(data, transform, normalized_sources, match_id)
    left = data
    if output in left.columns:
        left = left.drop(output)
    left = left.with_columns(_source_normalization_exprs(transform, normalized_sources))
    null_mask = _any_null_mask(left, normalized_sources)

    if spec["on_null"] == "raise" and bool(null_mask.any()):
        raise PredictionError(
            f"input transform {transform.name!r} received null source key(s) in "
            f"{spec['sources']!r}."
        )

    joined = left.join(
        lookup_frame,
        on=normalized_sources,
        how="left",
        maintain_order="left",
        nulls_equal=spec["on_null"] == "match",
    )

    missing_mask = joined[match_id].is_null()
    if spec["on_null"] == "default":
        unseen_mask = missing_mask & ~null_mask
        default_mask = missing_mask
    else:
        unseen_mask = missing_mask
        default_mask = missing_mask

    if spec["on_unseen"] == "raise" and bool(unseen_mask.any()):
        sample = _sample_bad_keys(joined, normalized_sources, unseen_mask)
        raise PredictionError(
            f"input transform {transform.name!r} received unseen key(s); sample={sample!r}."
        )

    if bool(default_mask.any()):
        joined = joined.with_columns(
            pl.when(pl.Series(default_mask))
            .then(pl.lit(spec["default"]))
            .otherwise(pl.col(output))
            .alias(output)
        )

    output_expr = _output_cast_expr(output, spec["output_dtype"])
    joined = joined.with_columns(output_expr)

    drop_cols = [match_id, *normalized_sources]
    return joined.drop([c for c in drop_cols if c in joined.columns])


def _source_normalization_exprs(
    transform: CompiledLookupTransform,
    normalized_sources: list[str],
) -> list[pl.Expr]:
    return [
        pl.col(source).cast(pl.Utf8).alias(normalized)
        for source, normalized in zip(transform.sources, normalized_sources, strict=True)
    ]


def _any_null_mask(data: pl.DataFrame, columns: list[str]) -> pl.Series:
    if not columns:
        return pl.Series([], dtype=pl.Boolean)
    mask = data[columns[0]].is_null()
    for col in columns[1:]:
        mask = mask | data[col].is_null()
    return mask


def _sample_bad_keys(
    data: pl.DataFrame,
    columns: list[str],
    mask: pl.Series,
    n: int = 5,
) -> list[list[Any]]:
    return data.filter(mask).select(columns).head(n).rows()


def _validate_sources_present(data: pl.DataFrame, transform: CompiledLookupTransform) -> None:
    missing = [source for source in transform.sources if source not in data.columns]
    if missing:
        raise PredictionError(
            f"input transform {transform.name!r} requires missing source column(s): {missing}."
        )


def _output_cast_expr(column: str, output_dtype: str) -> pl.Expr:
    if output_dtype == "float64":
        return pl.col(column).cast(pl.Float64).alias(column)
    return pl.col(column).cast(pl.Utf8).alias(column)


def _validate_key_value(name: str, value: Any, j: int) -> str | float | int | None:
    """Validate a raw spec-key element, preserving its type for dtype-aware matching."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValidationError(
            f"input transform {name!r} key {j} contains boolean {value!r}; "
            "use string or numeric key values."
        )
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValidationError(
                f"input transform {name!r} key {j} value {value!r} is not finite."
            )
        return value
    if isinstance(value, str):
        return value
    raise ValidationError(
        f"input transform {name!r} key {j} value {value!r} must be string, numeric, or null."
    )


def _dedup_key_value(value: Any) -> tuple[str, Any]:
    """Canonical comparison key so ``10`` and ``10.0`` collide but stay distinct from ``\"10\"``."""
    if value is None:
        return ("null", None)
    if isinstance(value, (int, float)):
        return ("num", float(value))
    return ("str", value)


def _validate_value(name: str, output_dtype: str, value: Any, field: str) -> float | str:
    if output_dtype == "float64":
        if isinstance(value, bool):
            raise ValidationError(
                f"input transform {name!r} {field} contains boolean {value!r}; expected float."
            )
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                f"input transform {name!r} {field} value {value!r} is not float64-compatible."
            ) from exc
        if not math.isfinite(result):
            raise ValidationError(
                f"input transform {name!r} {field} value {value!r} is not finite."
            )
        return result
    if value is None:
        raise ValidationError(
            f"input transform {name!r} {field} value cannot be null for output_dtype='string'."
        )
    return str(value)


def _normalized_col_name(transform_name: str, source: str, index: int) -> str:
    return f"__rustystats_it_{transform_name}_{index}_{source}__"


def _unique_temp_name(columns: list[str], base: str) -> str:
    existing = set(columns)
    name = base
    i = 0
    while name in existing:
        i += 1
        name = f"{base}_{i}"
    return name
