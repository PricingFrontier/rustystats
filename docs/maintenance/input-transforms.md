# Input Transforms Spec

Status: proposed
Scope: RustyStats dict-first `GLMModel` scoring and rate-table export contracts
Audience: maintainers implementing deployable GLM artifacts and upstream model compilers

This spec adds first-class deterministic input transforms to RustyStats so a
fitted `GLMModel` can score raw production data directly:

```python
mu = glm.predict(raw_df)
trace = glm.predict_contributions(raw_df)
```

The immediate motivating use case is destyler's frozen categorical effects.
Destyler currently compiles teacher-derived categorical main effects and
categorical combinations into derived columns such as `postcode_fts` and
`brand__region_fts`; callers then have to remember to run
`recipe.apply_grouping(df)` before calling `student.predict(...)`. That
preparation should live inside the deployable RustyStats object instead.

## Summary

Add an optional `input_transforms` field to `glm_dict` fits and to `GLMModel`.
RustyStats applies these transforms before building the design matrix during
fit, prediction, contribution tracing, diagnostics, calibration, and deployment
exports.

The public API scope is dict-first GLM only. Any internal objects still named
`FormulaGLMDict`, `ParsedFormula`, or similar are implementation details for
turning a dict specification into a design matrix; this spec does not introduce
or require a public formula-string API, a `gbm_dict` API, or any GBM/tree-model
API.

The first implementation only needs one generic transform type:

| Transform | Purpose | Example output |
| --- | --- | --- |
| `lookup` | Deterministically map one or more raw source columns to a derived string or numeric column | `brand_grp`, `postcode_fts`, `brand__region_fts` |

RustyStats must apply these maps; it must not learn them. Anything involving a
teacher model, functional ANOVA, credibility shrinkage, or response-informed
choice of lookup values remains outside RustyStats.

## Non-Goals

- No CatBoost, GBM, teacher, or functional-ANOVA dependency in RustyStats.
- No response-dependent fitting of frozen maps.
- No new target-encoding method. RustyStats target encoding remains the
  existing ordered/statistical encoder; frozen lookups are already-fitted
  deterministic data preparation.
- No hidden automatic categorical grouping. Upstream code must pass explicit
  maps.
- No change to GLM fitting mathematics, solver behavior, or coefficient
  interpretation.
- No formula-string, `gbm_dict`, or tree-model transform contract in the first
  implementation. The contract is `rs.glm_dict(...).fit()` -> `GLMModel`.

## Current Destyler Behavior

Destyler currently stores two kinds of deterministic preparation in its recipe:

1. Grouped categoricals:
   - source: one raw categorical column
   - output: `f"{feature}_grp"`
   - value: string group label such as `grp_1`
   - unseen default: `other`

2. Frozen categorical statistics:
   - source: one raw categorical column, or an all-categorical pair/triple
   - output: `f"{name}_fts"`
   - value: numeric teacher-derived purified effect
   - unseen default: `0.0`

For a multi-column frozen combination, destyler currently builds a composite
string key with `"|"`. RustyStats should not copy that serialization detail.
The RustyStats contract should store structured keys so raw levels containing
`"|"` cannot collide.

Conceptually:

```text
postcode -> postcode_fts
("Ford", "North") -> brand__region_fts
```

The GLM term then references the derived column as an ordinary linear term:

```python
terms = {
    "postcode_fts": {"type": "linear"},
    "brand__region_fts": {"type": "linear"},
}
```

The fitted contribution is `coefficient * frozen_value`. If the GLM coefficient
is near 1, the student uses the teacher-derived effect scale directly; otherwise
the GLM calibrates the frozen effect.

## Required User Contract

Given:

```python
input_transforms = [
    {
        "type": "lookup",
        "name": "postcode_frozen_effect",
        "sources": ["postcode"],
        "output": "postcode_fts",
        "output_dtype": "float64",
        "keys": [["AB1"], ["AB2"]],
        "values": [0.13, -0.04],
        "default": 0.0,
        "on_unseen": "default",
        "on_null": "default",
    },
    {
        "type": "lookup",
        "name": "brand_region_frozen_effect",
        "sources": ["brand", "region"],
        "output": "brand__region_fts",
        "output_dtype": "float64",
        "keys": [["Ford", "North"], ["BMW", "South"]],
        "values": [0.07, -0.03],
        "default": 0.0,
        "on_unseen": "default",
        "on_null": "default",
    },
]

glm = rs.glm_dict(
    response="ClaimNb",
    terms={
        "postcode_fts": {"type": "linear"},
        "brand__region_fts": {"type": "linear"},
        "age": {"type": "bs", "df": 4},
    },
    input_transforms=input_transforms,
    data=train_raw,
    family="poisson",
    exposure="Exposure",
).fit()

mu = glm.predict(score_raw)
```

`score_raw` only needs raw columns: `postcode`, `brand`, `region`, `age`,
`Exposure`. It does not need `postcode_fts` or `brand__region_fts`.

The same raw-data contract must hold for:

- `GLMModel.predict`
- `GLMModel.predict_contributions`
- `GLMModel.diagnostics`
- `GLMModel.calibration_summary`
- `GLMModel.fit_calibration`
- `GLMModel.to_bytes` / `GLMModel.from_bytes`
- full-mode PMML/ONNX export, where feasible
- rate-table export

## Transform Schema

### `lookup`

Required fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `type` | `"lookup"` | Transform kind |
| `name` | `str` | Stable transform identifier for reporting/debugging |
| `sources` | `list[str]` | One or more raw input columns, in key order |
| `output` | `str` | Derived column name added before design-matrix construction |
| `output_dtype` | `"float64"` or `"string"` | Output column dtype |
| `keys` | `list[list[str | null]]` | Structured source-key rows |
| `values` | `list[float | str]` | Lookup values aligned to `keys` |

Optional fields:

| Field | Default | Meaning |
| --- | --- | --- |
| `default` | required when `on_unseen="default"` | Value for unseen keys |
| `on_unseen` | `"default"` | `"default"` or `"raise"` |
| `on_null` | `"default"` | `"default"`, `"raise"`, or `"match"` |
| `source_cast` | `"string"` | How raw source values are normalized before lookup |
| `replace_existing` | `False` | Whether `output` may overwrite an existing column |
| `metadata` | `{}` | Audit-only information from upstream compilers |

Recommended metadata for destyler:

```python
{
    "producer": "destyler",
    "kind": "frozen_categorical_main",
    "teacher_family": "poisson",
    "teacher_depth": 2,
    "default_meaning": "centered_no_effect",
    "shrinkage": "buhlmann_median_mass",
}
```

### Key Normalization

RustyStats should normalize keys as tuples, not delimiter-joined strings.

Rules:

1. `sources` order is part of the key.
2. Raw values are cast according to `source_cast`.
3. The default `source_cast="string"` should match Polars UTF-8 casting for
   categorical and numeric source columns.
4. `null` handling is controlled by `on_null`:
   - `"default"`: null-containing keys use `default`.
   - `"raise"`: null-containing keys raise a prediction error.
   - `"match"`: `null` may appear in `keys` and match explicitly.
5. Duplicate normalized keys are invalid.
6. `len(keys) == len(values)` is required.
7. Every key row must have `len(sources)` elements.

## API Changes

### Dict API

Add `input_transforms`:

```python
rs.glm_dict(
    response,
    terms,
    data,
    interactions=None,
    input_transforms=None,
    ...
)
```

During `glm_dict` construction and the internal dict-model initialization:

1. Validate `input_transforms`.
2. Apply transforms to `data`.
3. Build the internal parsed design specification and design matrix from the
   transformed frame.
4. Store the validated transform specs on the fitted `GLMModel`.

The implementation may parse the dict specification before applying transforms,
but the design-matrix builder must see the transformed frame. If `data` is a
LazyFrame, the initial column projection must include transform source columns,
not only the term output columns.

### GLMModel

`GLMModel` should expose:

```python
glm.input_transforms
glm.prepare_input(raw_df)
```

`prepare_input` is useful for debugging and parity tests. It should return a
Polars `DataFrame` with derived columns added. It should not mutate the input.

`predict` and `predict_contributions` should call this internally before
building the design matrix. Public callers should not have to call it.

### Attaching After Fit

The core target path is fit-time `input_transforms`. A post-fit attachment API
is optional and should be conservative:

```python
glm2 = glm.with_input_transforms(input_transforms, validate_against=None)
```

If implemented, it must not mutate the original model. It should validate that
the transform outputs satisfy the model's required feature columns. This is a
convenience for upstream tools that already fitted on prepared data, but it is
not required for the first release.

## Prediction Semantics

For `GLMModel.predict(raw_df)`:

1. Collect LazyFrame columns needed by transforms, terms, interactions, offset,
   exposure, weights, and complement.
2. Apply transforms in declared order.
3. Build the design matrix from the transformed frame using the existing
   fitted `InteractionBuilder` state.
4. Compute `X @ beta`.
5. Add exposure, offset, and complement link-scale terms exactly as today.
6. Apply inverse link.

Transform order is significant. A later transform may consume the output of an
earlier transform. Cycles are invalid and should be caught by validation.

For `predict_contributions`, transforms do not create separate contribution
rows. They are input preparation. The derived GLM term receives the contribution
under the normal term name, for example:

```text
term = "brand__region_fts"
value = beta_brand_region * lookup_value
```

Future reporting may attach transform metadata to contribution rows, but the
additive decomposition should remain over GLM terms.

## Serialization

`GLMModel.to_bytes()` must include `input_transforms` in the serialized model
state. `GLMModel.from_bytes()` must restore them exactly.

The serialized state should bump schema version and include:

```python
{
    "schema_version": 4,
    "input_transforms": [...],
    ...
}
```

Compatibility rule:

- Older models without `input_transforms` load with an empty transform list.
- New models with transforms must fail clearly if loaded by an older version
  that cannot understand schema version 4.

The docs currently claim serialized models include everything needed for
prediction. After this feature, that claim must explicitly include deterministic
input transforms.

## Validation And Errors

Fail closed at fit time where possible.

Validation rules:

1. Unknown transform `type` raises `ValidationError`.
2. Missing required fields raise `ValidationError`.
3. Missing source columns in fit data raise `ValidationError`.
4. Missing source columns in prediction data raise `PredictionError`.
5. Duplicate output names raise unless they are the same as a source and
   `replace_existing=True`.
6. Output names that collide with existing raw columns raise unless
   `replace_existing=True`.
7. Duplicate normalized keys raise `ValidationError`.
8. `default` must match `output_dtype`.
9. Every value must match `output_dtype`.
10. A transform output consumed by `terms` or `interactions` must exist after
    transform application.
11. `on_unseen="raise"` should name the transform and key in the error.
12. `on_null="raise"` should name the transform and source columns in the error.

For destyler frozen statistics, the expected default is:

```python
default = 0.0
on_unseen = "default"
on_null = "default"
```

That means unseen levels/cells map to a centered no-effect prior.

## Export Semantics

### Rate Tables

RustyStats should support exporting a fitted GLM to a set of rate tables for
rating-plan deployment. For some pricing teams this may be the preferred
deployment artifact, not just an audit report, so the default output should be
compact, deterministic, and directly scoreable by a simple table-lookup engine:

```python
tables = glm.to_rate_tables(
    path="rate_tables.csv",
    format="stacked_csv",
    style="resolved",
    deployment=True,
    spline_strategy="unsupported",  # "unsupported", "grid", or "basis"
    spline_grids=None,
    spline_interpolation="linear",
    spline_extrapolation="clip",
    include_components=False,
)
```

This export is different from PMML/ONNX. It should produce concise,
human-auditable, machine-readable tables, not an opaque scoring graph and not a
normalized model-state dump. The default `style="resolved"` contract is:

- one logical table per rating effect
- rows keyed by raw source factor values wherever possible
- no separate "encoding table" plus "parameter table" for the same factor
- every row carries the final link-scale contribution and, for log-link models,
  the final multiplicative relativity
- transform internals may appear as row columns or table metadata, but analysts
  should not have to join multiple tables to answer "what relativity applies to
  this raw factor value?"

When `deployment=True`, every exported table must be directly usable for
scoring. Terms that cannot be represented as resolved finite tables under the
chosen options must fail loudly rather than falling back to metadata-only output.

A suggested compact top-level shape is:

```python
{
    "schema_version": 1,
    "kind": "rustystats_rate_tables",
    "family": "poisson",
    "link": "log",
    "prediction_space": "response",
    "base": {
        "eta": -3.2,
        "rel": 0.0408,
    },
    "tables": [...],
}
```

The same logical artifact should be exportable as:

- `format="dict"`: in-memory structured artifact.
- `format="stacked_csv"`: one dependency-free CSV file with each table written
  as a titled block, separated by blank rows.

For many pricing teams, `stacked_csv` may be the simplest deployment format:
easy to diff, easy to load, and it keeps the export dependency footprint at
the existing `numpy` / `polars` level.

`stacked_csv` structure:

```text
_base
field,value
schema_version,1
family,poisson
link,log
base_eta,-3.2
base_rel,0.0408

brand_region
brand,region,frozen_effect,eta,rel
Ford,North,0.07,0.0651,1.0673
BMW,South,-0.03,-0.0279,0.9725
<default>,<default>,0.0,0.0,1.0

postcode
postcode,group,eta,rel
AB1,grp_1,0.1133,1.1200
AB2,grp_7,-0.0513,0.9500
<unseen>,other,0.0,1.0
```

Each block starts with a title row containing only the table name, followed by a
normal CSV header row and then the rows for that table. Blocks are separated by
one blank row. The first blocks should be `_base` and `_manifest`, followed by
one block per rating effect, then optional `_warnings` / `_provenance` blocks.

The stacked CSV parser contract must be simple and explicit:

1. A non-empty single-cell row starts a new table block.
2. The following non-empty row is that table's header.
3. Subsequent rows belong to that table until the next blank row.
4. Table names must be deterministic and unique.
5. Values are ordinary CSV strings; loaders should cast columns according to
   `_manifest` metadata when typed values are required.

The manifest includes `on_unseen` and `on_null` for lookup-derived tables.
Explicit null-match keys are displayed as `<null>` so they are distinct from
empty-string source keys.

Tables should use a columnar row format by default. This avoids repeating field
names thousands of times while remaining easy to load into Polars, pandas,
SQL, or a pricing engine:

```python
{
    "name": "brand_region",
    "term": "brand__region_fts",
    "kind": "lookup_linear",
    "sources": ["brand", "region"],
    "columns": ["brand", "region", "frozen_effect", "eta", "rel"],
    "rows": [
        ["Ford", "North", 0.07, 0.0651, 1.0673],
        ["BMW", "South", -0.03, -0.0279, 0.9725],
    ],
    "default": ["<default>", "<default>", 0.0, 0.0, 1.0],
    "metadata": {"on_unseen": "default", "on_null": "default"},
}
```

For grouped categoricals, the grouping transform and the fitted categorical
parameterization should be resolved into the same raw-factor table:

```python
{
    "name": "postcode",
    "term": "postcode_grp",
    "kind": "lookup_categorical",
    "sources": ["postcode"],
    "columns": ["postcode", "group", "eta", "rel"],
    "rows": [
        ["AB1", "grp_1", 0.1133, 1.1200],
        ["AB2", "grp_7", -0.0513, 0.9500],
    ],
    "default": ["<unseen>", "other", 0.0, 1.0],
    "metadata": {"on_unseen": "default", "on_null": "raise"},
}
```

Input transforms must be incorporated inline:

1. Any GLM term that depends on a transform output should be exported as a
   resolved table over the transform's raw `sources`.
2. Numeric lookup outputs used as linear terms should emit raw key -> lookup
   value -> final contribution/relativity in one table.
3. String lookup outputs used as categorical terms should emit raw key -> group
   -> final contribution/relativity in one table.
4. The original transform spec may be included as compact table metadata when
   useful for deployment provenance, but it must not be the only place where the
   raw-key mapping appears.
5. A non-default `include_components=True` may add columns such as coefficient,
   design-column name, encoded value, or transform name. These columns must be in
   the same resolved table, not emitted as separate tables that need to be joined
   back together.

This is the primary ratebook path for frozen categorical effects. Frozen
high-cardinality categorical main effects and frozen categorical combinations
should be represented as lookup transforms and exported as resolved raw-factor
tables. They must not be split into separate "frozen encoding" and "coefficient"
tables.

Term export rules:

| Term type | Rate-table behavior |
| --- | --- |
| Intercept | Export once as `base.eta`; for log link also export `base.rel = exp(eta)` |
| Categorical | Export one resolved table over raw levels, including baseline with `eta=0` / `rel=1` |
| Linear | Export a resolved table only when a finite value set is known, such as a numeric lookup transform or explicit user grid; otherwise fail when `deployment=True` |
| Expression | Export a resolved table only with an explicit grid/value set; otherwise fail when `deployment=True` |
| Target/frequency encoding | Export raw level/cell -> encoded value -> final contribution/relativity in one table |
| Interactions | Export one resolved table over raw interaction keys when all dimensions have finite known levels; otherwise require an explicit grid/value set |
| Splines | See spline policy below |

Spline terms are not naturally finite rate tables. The default
`spline_strategy="unsupported"` should raise a clear unsupported-feature error
when splines are present. Silent omission or accidental discretization is not
allowed.

Supported spline strategies:

- `spline_strategy="grid"`: export an evaluation table on an explicit or
  generated grid, with the grid, interpolation policy, extrapolation policy,
  link-scale contribution, and relativity. This is an approximation for
  table-based deployment and must be labelled as such. The first
  implementation requires explicit user-supplied grids via `spline_grids`.
- `spline_strategy="basis"`: intentionally deferred exact spline metadata,
  knots, basis type, monotonicity metadata, and coefficients. This is not a
  plain ratebook table; a consumer must implement RustyStats-compatible basis
  evaluation.

The rate-table artifact must be self-contained for the chosen strategy. If a
term cannot be represented without changing scoring semantics, export must fail
loudly and name the unsupported term.

### PMML

Full-mode PMML export should emit lookup transforms as `DerivedField`s where
the PMML representation is tractable. If PMML cannot represent a transform
without changing semantics, export must raise a clear unsupported-feature error.

Scoring/design-matrix mode may ignore raw input transforms because the caller is
already expected to supply the design matrix.

### ONNX

Full-mode ONNX export should either encode lookup transforms with map/category
operators or raise a clear unsupported-feature error. Silent omission is not
allowed.

### Haute

Haute should be able to treat a transformed `.rsglm` exactly like any other
RustyStats GLM: load bytes, pass raw Polars data to `predict`, and receive
predictions. No destyler recipe should be required at scoring time.

## Implementation Plan

### Engineering Principles

1. Keep the public contract dict-first. Transform specs remain plain
   JSON-serializable dictionaries; compiled Python objects are internal caches.
2. Fail closed. Fitting, prediction, serialization, and export should raise
   explicit RustyStats errors rather than silently dropping transforms or
   approximating unsupported terms.
3. Use vectorized Polars operations for row-wise work. No Python loops over
   scoring rows are acceptable in production paths.
4. Preserve the existing prediction memory discipline. Input transforms must
   participate in the chunked prediction path instead of forcing a full
   materialized design matrix or derived-frame copy.
5. Add no dependencies beyond `numpy` and `polars`. The stacked CSV writer
   should use Python's standard `csv` module.
6. Keep artifacts deterministic: canonicalized transform specs, stable table
   names, stable row ordering, and reproducible serialized state.
7. Build narrow, composable modules. `formula.py` should orchestrate; transform
   and rate-table logic should live in dedicated modules.

### Phase 0: Integration Survey And Test Fixtures

Before changing behavior, identify and document the concrete touch points:

1. `python/rustystats/formula.py`
   - `glm_dict(...)` public signature.
   - `FormulaGLMDict` construction and fit-time LazyFrame projection.
   - `_extract_needed_columns(...)` and `_extract_model_needed_columns(...)`.
   - `GLMModel.__init__`, `predict`, `predict_contributions`,
     `_calibration_extract_arrays`, `to_bytes`, and `from_bytes`.
2. `python/rustystats/interactions.py`
   - `InteractionBuilder.build_design_matrix_from_parsed(...)` for fit.
   - `InteractionBuilder.transform_new_data(...)` for prediction and
     diagnostics.
   - Term-slot metadata used to resolve rate-table effects.
3. `python/rustystats/contributions.py` and diagnostics modules that call
   `transform_new_data(...)` directly.
4. `python/rustystats/export_pmml.py` and `python/rustystats/export_onnx.py`
   fail-closed checks for full raw-data export modes.

Add small fixtures up front:

1. Single-column numeric frozen effect.
2. Multi-column numeric frozen effect with a source value containing `"|"`.
3. Single-column string grouping used as a categorical term.
4. A model with a spline, to prove default rate-table export rejects it.
5. A LazyFrame fixture backed by a Parquet scan, to catch projection bugs.

### Phase 1: Input Transform Core

Add `python/rustystats/input_transforms.py` with a small public helper surface
and internal compiled classes:

```python
validate_input_transforms(specs, data_schema=None) -> list[dict]
compile_input_transforms(specs) -> list[CompiledInputTransform]
apply_input_transforms(data, compiled_or_specs) -> pl.DataFrame
apply_input_transforms_lazy(lf, compiled_or_specs) -> pl.LazyFrame
input_transform_source_columns(specs) -> set[str]
input_transform_output_columns(specs) -> set[str]
```

`apply_input_transforms_lazy()` may collect internally in v1 so that its
unseen/null/collision behavior is identical to eager prediction. A fully lazy
implementation should preserve those semantics before being used for streaming
scoring.

Implementation details:

1. Canonicalize specs at validation time: copy dictionaries, normalize field
   order, convert keys to structured lists, and validate all value dtypes.
2. Store compiled lookup state separately from serialized specs. A compiled
   lookup should hold a Polars lookup frame with normalized source columns, the
   output column, and a row-order column for deterministic exports.
3. Apply lookups using Polars joins on normalized source columns. Do not build
   delimiter-joined keys and do not iterate over scoring rows in Python.
4. Normalize source columns with expressions such as
   `pl.col(source).cast(pl.Utf8)` for the default `source_cast="string"`.
   Temporary normalized columns must use collision-resistant internal names and
   be dropped before returning.
5. Normalize spec keys through the same Polars casting path as incoming data.
   Do not use Python `str(...)` or tuple hashing as the canonical conversion.
6. Handle null matching explicitly. Use masks plus Polars `nulls_equal` so
   `on_null` semantics are stable and real string values cannot collide with an
   internal null sentinel.
7. Preserve row order after joins by using a temporary row index whenever the
   Polars operation may reorder rows.
8. Implement unseen/null handling as vectorized masks after the join:
   fill defaults, raise with a compact sample of bad keys, or allow explicit
   null matching when `on_null="match"`.
9. Validate transform dependency order. A transform may consume an earlier
   transform output; cycles, missing sources, and duplicate outputs fail at
   validation.
10. Keep `replace_existing=False` as the safe default. If replacement is allowed,
   make it explicit in both validation and returned metadata.

Performance requirements for this phase:

1. A single-source lookup should be within a small constant factor of an
   equivalent Polars left join.
2. Multi-source lookups must scale with Polars hash-join behavior, not with
   Python tuple-map row iteration.
3. Compiled lookup frames are built once per model load or fit, then reused for
   all predictions.

### Phase 2: Fit-Time Integration

Thread `input_transforms` through the dict API:

1. Add `input_transforms=None` to `glm_dict(...)` and `FormulaGLMDict`.
2. Validate and compile transforms before the design matrix is built.
3. Make LazyFrame projection transform-aware:
   - collect response, raw term columns not created by transforms, transform
     source columns, offset, exposure, weights, and complement columns
   - do not require transform output columns to exist in the raw LazyFrame
   - apply transforms after collection, before design-matrix construction
4. Apply transforms before
   `InteractionBuilder.build_design_matrix_from_parsed(...)`.
5. Store canonical specs and compiled transforms on the fitted `GLMModel`:
   - `glm.input_transforms` returns canonical dict specs
   - `glm._compiled_input_transforms` is an internal cache
6. Validate that every term and interaction source is available after transform
   application, and that the transformed fit frame has the expected row count.
7. Keep existing GLM fitting math untouched. The transform layer only changes
   which columns are present before design-matrix construction.

Release gate:

1. Fitting with transformed raw data must produce identical coefficients and
   predictions to fitting the same model on an explicitly pre-prepared frame.
2. LazyFrame and eager DataFrame fits must agree.

### Phase 3: Prediction, Contributions, And Diagnostics

Create one internal preparation path and route all scoring surfaces through it:

```python
GLMModel.prepare_input(raw_df) -> pl.DataFrame
GLMModel._prepare_prediction_data(raw_df, needed_columns, *, collect=True)
```

Implementation details:

1. `prepare_input` should be a debugging/public parity method. It returns a new
   DataFrame and never mutates caller-owned data.
2. `predict` should collect only needed raw columns, then apply transforms.
3. For large predictions, apply transforms inside the existing row-chunk loop
   before `InteractionBuilder.transform_new_data(chunk)`. This avoids holding
   both the full transformed frame and a large design matrix at once.
4. For small predictions, keep the current fast single-shot path.
5. `predict_contributions` should call the same preparation helper before
   building `X_new`; additivity validation must compare against transformed
   scoring, not the unprepared raw frame.
6. Calibration and diagnostics code paths that call `predict(...)` are covered
   automatically; paths that call `transform_new_data(...)` directly must call
   the preparation helper first.
7. Complement models require recursive needed-column extraction that includes
   their own input transforms.
8. Missing prediction-time source columns raise `PredictionError` naming the
   transform and missing columns.

Release gate:

1. Existing no-transform prediction benchmarks should not materially regress.
2. Transformed prediction should remain bounded by the existing design-matrix
   chunk byte budget plus the current chunk's transform columns.
3. `predict(raw_df)` and `predict(prepare_input(raw_df))` must agree.
4. `predict_contributions(raw_df, validate=True)` must pass for transformed
   models.

### Phase 4: Serialization And Compatibility

Update `GLMModel.to_bytes()` / `from_bytes()`:

1. Bump the schema version.
2. Store canonical `input_transforms` specs, not compiled lookup objects.
3. Recompile transforms after deserialization.
4. Older models without `input_transforms` load as an empty list.
5. Newer transform-bearing models must fail clearly in versions that do not
   support the schema.
6. Update serialization docs to say deterministic input transforms are part of
   the self-contained prediction artifact.

Release gate:

1. Serialized transformed models predict identically before and after
   round-trip.
2. Serialized no-transform models remain backward compatible.

### Phase 5: Rate-Table Export Core

Add `python/rustystats/rate_tables.py` and a thin
`GLMModel.to_rate_tables(...)` wrapper.

Core architecture:

1. Build an internal `RateTableArtifact` from the fitted model:
   - `_base` metadata with family, link, intercept/base `eta`, and base `rel`
   - `_manifest` with one row per rating effect
   - one compact table per resolved rating effect
   - optional `_warnings` and `_provenance`
2. Use `InteractionBuilder` term-slot metadata first. Fall back to
   feature-name parsing only where term slots do not carry enough information.
3. Resolve transformed terms inline:
   - numeric lookup + linear coefficient becomes raw keys, lookup value,
     `eta = coefficient * value`, and `rel`
   - string lookup + categorical coefficient becomes raw keys, group, category
     coefficient, `eta`, and `rel`
   - default policy rows are emitted in the same table only for defaulting
     `on_unseen` / `on_null` policies; strict `"raise"` policies are carried
     in metadata and must not get a fallback row
4. For plain categoricals, emit one table over raw levels including the
   baseline row.
5. For finite target/frequency encodings and finite interactions, emit one
   resolved table only when all source levels are known and the estimated row
   count is below a configurable guardrail.
6. Linear, expression, interaction, and spline terms without a finite known
   value set fail when `deployment=True`.
7. `spline_strategy="unsupported"` is the default and must reject spline terms.
8. Implement `spline_strategy="grid"` for explicit user-supplied grids with
   linear interpolation and explicit extrapolation metadata in the manifest.
9. Add `basis` only behind explicit tests and manifest metadata, because it
   requires a consumer to implement RustyStats-compatible basis evaluation.

Performance and artifact-size requirements:

1. Rate-table generation should use model metadata and transform lookup frames,
   not rescore the full training data.
2. Avoid cross-product expansion unless the resulting row count is explicitly
   bounded and useful for deployment.
3. The default artifact should not duplicate large transform maps in both
   metadata and rows.
4. Rows should be sorted deterministically by source columns, with default rows
   last, unless a future option requests source-order preservation.

### Phase 6: Stacked CSV Writer

Implement `format="dict"` first, then `format="stacked_csv"`:

1. `format="dict"` returns the in-memory `RateTableArtifact` as plain Python
   dictionaries and lists.
2. `format="stacked_csv"` writes one file using the standard `csv` module.
3. Write `_base`, `_manifest`, rating-effect blocks, then optional warning and
   provenance blocks.
4. Escape values using normal CSV rules. Do not hand-roll quoting.
5. Add a small parser helper in tests only, to verify the stacked CSV can round
   trip to the same logical blocks.

Release gate:

1. No extra dependencies are introduced.
2. The stacked CSV output is deterministic byte-for-byte for the same model and
   options.
3. An analyst can inspect a raw factor table and see the applied `eta` / `rel`
   without joining to another table.

### Phase 7: Export Interoperability

PMML and ONNX full raw-data exports should either support transforms or fail
closed:

1. Scoring/design-matrix export modes may keep requiring pre-built design
   matrices.
2. Full raw-data modes must detect `input_transforms`.
3. If a lookup transform cannot be represented exactly, raise a clear
   unsupported-feature error naming the transform and export mode.
4. Add support only when the exported representation can match RustyStats
   prediction semantics for nulls, unseen keys, defaults, and multi-column
   structured keys.

### Phase 8: Documentation, Benchmarks, And Release Readiness

Documentation:

1. Update `docs/api/dict-api.md` with fit and predict examples.
2. Update `docs/api/serialization.md` with the transform persistence contract.
3. Add rate-table export docs with the stacked CSV block format and deployment
   failure behavior.
4. Keep formula-string, `gbm_dict`, and tree-model APIs out of public docs for
   this feature.

Benchmarks:

1. Add `benchmarks/bench_input_transforms.py` for eager and LazyFrame lookups.
2. Add transformed prediction benchmarks for:
   - no-transform baseline
   - single-source lookup
   - multi-source lookup
   - chunked million-row prediction
3. Add rate-table export benchmarks for large categorical and lookup maps.

Release checklist:

1. Unit, integration, serialization, export, and destyler contract tests pass.
2. No-transform benchmark regression is investigated before merge.
3. Transform benchmarks are compared against direct Polars join baselines.
4. Memory profiling confirms chunked prediction does not materialize an
   unbounded transformed design matrix.
5. Public docs and error messages make deployment limitations explicit.

## Tests

Add unit tests for `input_transforms.py`:

1. Single-column numeric lookup produces expected `float64` output.
2. Multi-column numeric lookup uses structured tuple keys.
3. Source values containing `"|"` do not collide.
4. Unseen keys use `default`.
5. `on_unseen="raise"` raises.
6. Null keys follow `on_null`.
7. Duplicate normalized keys raise.
8. Output column collision raises unless `replace_existing=True`.
9. Transform application preserves row order.
10. Transform application does not mutate caller-owned DataFrames.

Add dict/model tests:

1. `glm_dict(..., input_transforms=...)` fits when terms reference derived
   columns absent from the raw frame.
2. `predict(raw_df)` equals `predict(prepared_df)` from the same model.
3. `predict_contributions(raw_df)` additivity holds.
4. `diagnostics(raw_df)` works with transformed terms.
5. `to_bytes` / `from_bytes` preserves predictions on raw data.
6. LazyFrame prediction collects transform source columns.
7. Chunked prediction applies transforms consistently.
8. No-transform models keep the existing prediction path and outputs.

Add rate-table export tests:

1. Categorical main effects export baseline and non-baseline relativities in one
   resolved table.
2. Numeric lookup-derived linear terms export raw keys, lookup values, and final
   relativities in one resolved table.
3. String lookup-derived categorical terms export raw keys, groups, and final
   grouped-level relativities in one resolved table.
4. Multi-column lookup keys remain structured in exported tables.
5. Splines raise by default with `spline_strategy="unsupported"`.
6. `spline_strategy="grid"` exports labelled approximate rows, manifest
   interpolation metadata, and manifest extrapolation metadata.
7. A rate-table artifact with input transforms can reproduce `predict(raw_df)`
   for supported term types.
8. No default export emits separate encoding and parameter tables for the same
   raw factor.
9. `format="stacked_csv"` writes `_base`, `_manifest`, and one titled block per
   rating effect, separated by blank rows.
10. Stacked CSV blocks can be parsed back into the same logical tables as
    `format="dict"`.
11. Stacked CSV output is deterministic byte-for-byte for the same model and
    options.
12. Unsupported deployment terms fail with clear term names rather than
    metadata-only fallback tables.

Add performance and dependency checks:

1. Benchmark no-transform prediction before and after integration.
2. Benchmark single-source and multi-source lookup application against direct
   Polars left joins.
3. Benchmark transformed chunked prediction on large data and record peak
   memory.
4. Verify no new runtime dependency is required beyond `numpy` and `polars`.

Add destyler contract tests in destyler:

```python
old = student.predict(recipe.apply_grouping(raw_df))
new = student_with_input_transforms.predict(raw_df)
np.testing.assert_allclose(old, new, rtol=1e-9, atol=1e-12)
```

Cover:

- grouped categorical main effects
- frozen high-cardinality categorical main effects
- frozen two-way categorical combinations
- unseen levels/cells
- serialized `.rsglm` round-trip

## Acceptance Criteria

The feature is complete when:

1. A RustyStats `GLMModel` can be fitted with terms that reference derived
   transform outputs not present in the raw training frame.
2. The fitted model can score raw data directly.
3. `predict`, `predict_contributions`, diagnostics, and calibration all use the
   same transform path.
4. Serialized models are self-contained.
5. Multi-column lookup keys are structured and delimiter-safe.
6. Export either preserves transforms or fails loudly.
7. Rate-table export includes input transforms inline in concise resolved tables
   or stacked CSV blocks, and either handles spline terms under an explicit
   strategy or rejects them clearly.
8. Destyler no longer needs to expose `recipe.apply_grouping(df)` as part of the
   production prediction path.
9. No additional runtime dependencies are introduced beyond `numpy` and
   `polars`.
10. Transformed prediction remains chunked and does not materialize an
    unbounded full design matrix.
11. Rate-table outputs are deterministic for the same model and options.

## Open Questions

1. Should transform metadata appear in `predict_contributions` rows, or only in
   separate model metadata?
2. Should `prepare_input` be public stable API or debugging API?
3. Should `replace_existing=True` be supported in v1, or should all transforms
   be required to write new columns?
4. Should string lookup outputs be represented as Polars `Utf8` only, or should
   categorical dtype be preserved when the input is categorical?
5. Should very large lookup maps use a compact binary payload separate from the
   pickle model state?
6. Should `to_rate_tables()` be limited to log-link pricing models in v1, or
   should identity/logit models export link-scale tables without relativities?
7. For spline grid export, should the grid be user-supplied only, or can
   RustyStats generate a default grid from fitted knots and training range?
8. Should the deployment default omit audit columns such as `eta`, coefficient,
   and encoded value when `rel` is available, or keep a minimal `eta` + `rel`
   pair for numerical traceability?
