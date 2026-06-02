# Rate Table Export

`GLMModel.to_rate_tables()` exports a fitted dict-first GLM into concise
rating tables. This is intended for analyst review and simple table-lookup
deployment paths where an opaque scoring graph is not desirable.

```python
artifact = result.to_rate_tables(
    format="dict",
    style="resolved",
    deployment=True,
    spline_strategy="unsupported",
    spline_grids=None,
)
```

The default `style="resolved"` emits one table per rating effect. When a term
depends on an `input_transforms` lookup, the transform is incorporated into
the same table as the fitted coefficient or categorical relativity. Analysts
can inspect the raw factor value and see the applied `eta` and `rel` without
joining an encoding table to a parameter table.

This is the intended export path for frozen categorical effects from upstream
model compilers. Numeric frozen effects should be represented as lookup outputs
used by linear terms; grouped frozen categoricals should be represented as
string lookup outputs used by categorical terms. Both are emitted as resolved
raw-factor tables.

## Formats

### `format="dict"`

Returns an in-memory artifact:

```python
{
    "schema_version": 1,
    "kind": "rustystats_rate_tables",
    "family": "poisson",
    "link": "log",
    "base": {"eta": -3.2, "rel": 0.0408},
    "manifest": [...],
    "tables": [...],
}
```

### `format="stacked_csv"`

Writes one CSV file with titled blocks stacked vertically:

```python
result.to_rate_tables(path="rate_tables.csv", format="stacked_csv")
```

The first blocks are `_base` and `_manifest`, followed by one block per rating
effect. Each block has a title row, a header row, data rows, and a blank row
before the next block.

```text
_base
field,value
schema_version,1
family,poisson
link,log
base_eta,-3.2
base_rel,0.0408

brand_region_fts
brand,region,brand_region_fts,eta,rel
Ford,North,0.07,0.0651,1.0673
BMW,South,-0.03,-0.0279,0.9725
<default>,<default>,0.0,0.0,1.0
```

No writer dependency is required; the implementation uses Python's standard
`csv` module and the existing `numpy` / `polars` runtime dependencies.

Lookup tables include `on_unseen` and `on_null` metadata in the manifest. A
default row is emitted only when prediction would default that policy; strict
`"raise"` policies do not get a fallback row. Explicit null-match keys are
rendered as `<null>` so they are not confused with empty strings.

## Supported Terms

The v1 resolved exporter supports exact finite tables for:

| Term | Behavior |
|------|----------|
| Intercept | Exported once in `_base` as `eta` and, for log link, `rel` |
| Categorical | One raw-level table including baseline and optional unseen/default row |
| Selected categorical levels | One finite raw-level table for the requested levels |
| Numeric lookup + linear term | Raw lookup keys, lookup value, `eta`, and `rel` in one table |
| String lookup + categorical term | Raw lookup keys, group, `eta`, and `rel` in one table |
| Target encoding | Raw level, encoded value, `eta`, and `rel` |
| Frequency encoding | Raw level, encoded value, `eta`, and `rel` |

Unsupported deployment terms fail closed by default. This includes ordinary
unconstrained continuous terms without an explicit finite grid, expressions
without an explicit finite grid, and interaction encodings whose stored
metadata is not yet structured enough for a safe raw-factor table.

## Splines

Spline terms are not naturally finite rate tables. The default
`spline_strategy="unsupported"` raises a clear error when spline terms are
present.

Use `spline_strategy="grid"` with an explicit grid to export an approximated
ratebook table:

```python
artifact = result.to_rate_tables(
    format="stacked_csv",
    path="rate_tables.csv",
    spline_strategy="grid",
    spline_grids={"Age": [18, 21, 25, 30, 35, 40, 50, 60, 70]},
    spline_interpolation="linear",
    spline_extrapolation="clip",
)
```

The exported table contains one row per grid value with the exact RustyStats
spline contribution evaluated at that point:

```text
Age,eta,rel
18,-0.1200,0.8869
21,-0.0830,0.9203
25,-0.0200,0.9802
```

The `_manifest` block labels spline grid tables as approximations and records
the interpolation and extrapolation policy. The v1 exporter supports linear
interpolation metadata and `clip` or `raise` extrapolation metadata. The CSV
consumer is responsible for applying that policy between grid points.

`spline_strategy="basis"` is intentionally deferred. It would require the
consumer to evaluate RustyStats-compatible spline bases, so it is not a plain
ratebook table.

## Deployment Mode

With `deployment=True`, every emitted table must be directly scoreable by a
simple table-lookup engine under the selected options. Terms that cannot be
represented exactly are rejected rather than omitted or converted to
metadata-only tables.
