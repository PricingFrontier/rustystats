# Destyler Methodology Support In RustyStats — Spec

Status: proposed
Scope: RustyStats only
Motivation: support a best-in-industry GBM-to-GLM workflow where a downstream
tool such as Destyler can render a full actuarial model card without
recomputing GLM diagnostics outside RustyStats.

RustyStats already has the right foundations:

- `result.diagnostics(..., base_predictions=...)` compares a fitted GLM against
  a base model.
- `FactorDiagnostics.actual_vs_expected` provides factor-level A/E bins.
- `FactorDiagnostics.train_test_bins` provides aligned train/test bin pairs.
- `partial_dependence` and `spline_info` expose fitted GLM effect shapes.
- User-specified pair diagnostics exist through `interactions=[...]`.

The remaining gap is not a new diagnostics system. It is a stronger diagnostics
contract: all report-ready A/E, factor, partial-dependence, base-model and
interaction artefacts must be available directly from RustyStats, with exposure
and prior weights handled consistently.

---

## Design Principles

1. **RustyStats owns GLM diagnostics methodology.**
   Downstream libraries must not re-bin factors, recompute A/E, infer exposure
   semantics, or rebuild model diagnostics from predictions except for display.

2. **Destyler owns GBM structure discovery and report narrative.**
   RustyStats should not know about "teacher decomposition", fANOVA, recipe
   selection, filing verdicts, or HTML layout.

3. **Emit raw actuarial data, not opinions.**
   RustyStats should emit rates, totals, intervals, fitted-term metadata,
   base-model overlays, train/test comparisons, significance, GVIF and score
   tests. Destyler can turn those into charts, warnings and committee language.

4. **Exposure and prior weights are distinct.**
   For any aggregate:

   - exposure is the rate denominator and log-link offset concept;
   - prior weights multiply the contribution of each row;
   - weighted exposure is `sum(w_i * exposure_i)`;
   - weighted actual is `sum(w_i * y_i)`;
   - weighted expected is `sum(w_i * mu_i)`;
   - A/E is `sum(w_i * y_i) / sum(w_i * mu_i)`.

5. **Backwards compatibility.**
   Existing fields stay valid. New fields are optional and appended to existing
   dataclasses where possible.

---

## RS-DST-001: Base Predictions By Role

### Problem

`diagnostics(base_predictions=...)` currently compares model predictions to a
base prediction column on `train_data`. In a GBM-to-GLM workflow, the important
comparison is usually holdout/OOT: raw GLM vs GBM teacher on the governed
evaluation role.

Destyler currently has to compute some held-out GBM-vs-GLM A/E views itself.
That is the wrong ownership boundary.

### Public API

Keep the existing argument:

```python
result.diagnostics(
    train_data=train,
    test_data=test,
    base_predictions="gbm_mu",
    exposure="Exposure",
    weights="weight",
)
```

Extend semantics:

- If `base_predictions` is a string, it is read from `train_data`.
- If `test_data` is supplied and contains the same column, compute the same base
  comparison for `test_data`.
- If `test_data` is supplied but does not contain the column, emit a warning and
  leave the test-side base comparison `None`.

Optional future-friendly extension:

```python
base_predictions={"train": "gbm_mu_train", "test": "gbm_mu_test"}
```

This can be added now if simple, but the string-column path is enough for
Destyler because it prepares the same teacher prediction column on both frames.

### Data Schema

Keep the existing field for compatibility:

```python
ModelDiagnostics.base_predictions_comparison: BasePredictionsComparison | None
```

Interpret it as the train-side comparison.

Add:

```python
@dataclass
class BasePredictionsByRole:
    train: BasePredictionsComparison | None
    test: BasePredictionsComparison | None = None
    ranking: str = "auto"
    prediction_basis: str = "response"

@dataclass
class ModelDiagnostics:
    ...
    base_predictions_by_role: BasePredictionsByRole | None = None
```

The `prediction_basis` is `"response"` because `base_predictions` must be on
the response scale: claim count, pure premium, probability, etc., not link
scale.

### Computation Rules

`BasePredictionsComparison` must use the same family loss, exposure ranking and
prior-weight semantics as the rest of diagnostics.

If weights are supplied:

- totals are `sum(w*y)`, `sum(w*mu_model)`, `sum(w*mu_base)`;
- exposure is `sum(w*exposure)`;
- Gini/lift use the same weighted logic as decile/lift diagnostics;
- uniform weights reproduce the unweighted result exactly.

### Tests

1. String `base_predictions` present on train only preserves current
   `base_predictions_comparison`.
2. Same column present on train and test populates
   `base_predictions_by_role.train` and `.test`.
3. Test column missing emits a warning, keeps train comparison, and does not
   fail.
4. Non-uniform prior weights change the train/test base totals exactly by
   `sum(w * value)`.
5. `w == 1` matches the unweighted path bit-for-bit within existing rounding.

---

## RS-DST-002: Base-Aware Factor A/E Bins

### Problem

`FactorDiagnostics.actual_vs_expected` gives actual and GLM expected by factor
bin. For Destyler's model card, each factor chart needs:

- actual line;
- GLM student line;
- GBM/base-model line;
- exposure bars;
- count bars.

RustyStats already computes the correct bins. It should add the base-model
overlay to those bins whenever `base_predictions` is supplied.

### Data Schema

Extend `ActualExpectedBin` with optional totals and base fields:

```python
@dataclass
class ActualExpectedBin:
    bin: str
    n: int
    exposure: float
    actual: float                 # actual rate = actual_total / exposure
    expected: float               # GLM rate = expected_total / exposure
    ae_ratio: float | None
    ae_ci: list[float]

    # NEW: report-ready totals and base overlay
    actual_total: float | None = None
    expected_total: float | None = None
    base_expected: float | None = None        # base rate
    base_expected_total: float | None = None
    base_ae_ratio: float | None = None
```

Extend `FactorBinPair`:

```python
@dataclass
class FactorBinPair:
    bin: str
    train_n: int
    train_exposure: float
    train_actual: float
    train_predicted: float
    train_ae_ratio: float | None
    test_n: int
    test_exposure: float
    test_actual: float | None = None
    test_predicted: float | None = None
    test_ae_ratio: float | None = None

    # NEW
    train_actual_total: float | None = None
    train_predicted_total: float | None = None
    train_base_predicted: float | None = None
    train_base_predicted_total: float | None = None
    train_base_ae_ratio: float | None = None
    test_actual_total: float | None = None
    test_predicted_total: float | None = None
    test_base_predicted: float | None = None
    test_base_predicted_total: float | None = None
    test_base_ae_ratio: float | None = None
```

Field meanings:

- `*_actual`, `*_predicted`, `*_base_predicted` are rates.
- `*_actual_total`, `*_predicted_total`, `*_base_predicted_total` are weighted
  totals.
- `n` / `train_n` / `test_n` are unweighted row counts.
- `exposure` / `train_exposure` / `test_exposure` are weighted exposure if
  prior weights are supplied.

### Computation Rules

For every factor bin:

```text
weighted_exposure = sum(w_i * exposure_i)
actual_total      = sum(w_i * y_i)
expected_total    = sum(w_i * mu_glm_i)
base_total        = sum(w_i * mu_base_i)

actual_rate       = actual_total / weighted_exposure
expected_rate     = expected_total / weighted_exposure
base_rate         = base_total / weighted_exposure

ae_ratio          = actual_total / expected_total
base_ae_ratio     = actual_total / base_total
```

If no `weights` are supplied, `w_i = 1`.

If no exposure is supplied, `exposure_i = 1`.

If no base predictions are supplied, all base fields are `None`.

### Tests

1. With no `base_predictions`, JSON output is backwards compatible and new base
   fields are `null` / omitted according to current serializer convention.
2. With base predictions, each factor bin contains `base_expected` and
   `base_ae_ratio`.
3. For a hand-computed three-bin fixture, all totals and rates match exact
   `sum(w * ...)` calculations.
4. Continuous factor test bins use train-side bin edges; categorical test bins
   use train-side level labels.
5. Empty test bins have `test_n=0`, `test_exposure=0`, and nullable actual /
   predicted / base values.

---

## RS-DST-003: Weighted Factor Diagnostics

### Problem

Weighted decile/lift diagnostics have been hardened, but factor A/E and
factor train/test bins must follow the same convention. For actuarial pricing,
factor charts are often model committee artefacts, so using unweighted bins
when the fit used prior weights is a methodological inconsistency.

### Required Change

Thread resolved prior weights into:

- `_FactorDiagnosticsComputer`;
- categorical A/E bins;
- continuous A/E bins;
- `compute_dataset_diagnostics().factor_diagnostics`;
- `compute_dataset_diagnostics().continuous_diagnostics`;
- `_build_factor_bin_pairs`;
- any Rust kernels used by those paths, or a Python weighted fallback if a Rust
  weighted kernel is not yet available.

### Acceptance Criteria

Every factor-level aggregate must obey:

```text
exposure = sum(w * exposure)
actual = sum(w * y) / sum(w * exposure)
predicted = sum(w * mu) / sum(w * exposure)
ae_ratio = sum(w * y) / sum(w * mu)
```

The unweighted path must be unchanged when `weights is None` or all weights are
one.

### Tests

1. Prior-weighted categorical A/E matches a hand-computed fixture.
2. Prior-weighted continuous A/E matches a hand-computed fixture.
3. Prior-weighted train/test bins match hand-computed train and test totals.
4. `weights=np.ones(n)` matches the unweighted output.
5. Explicit `weights=` overrides model-fitted weights.
6. Missing fitted weight column on test data emits the existing
   `test_weights_unavailable` warning and leaves test-side factor bins
   unweighted rather than reusing train-row weights.

---

## RS-DST-004: Report-Ready Partial Dependence And Knot Metadata

### Problem

RustyStats already emits `partial_dependence` and `spline_info`, but a
downstream report needs to join them reliably:

- plot the fitted GLM partial dependence curve;
- show whether the term is linear, spline, monotone spline, target encoding,
  frequency encoding, or categorical;
- draw vertical knot markers for splines;
- label whether predictions are response-scale or relativity-scale.

Destyler should not need to reverse-engineer spline metadata from feature names.

### Data Schema

Extend `PartialDependence`:

```python
@dataclass
class PartialDependence:
    variable: str
    variable_type: str
    grid_values: list[Any]
    predictions: list[float]
    relativities: list[float] | None
    shape: str
    recommendation: str

    # NEW
    term_type: str | None = None          # linear, bs, ns, ms, smooth, categorical, target_encoding, frequency_encoding
    prediction_scale: str = "response"    # response
    relativity_base: Any | None = None
    knots: list[float] | None = None
    boundary_knots: list[float] | None = None
    monotonicity: str | None = None       # increasing, decreasing, none, unknown
```

The existing top-level `spline_info` can stay. The new fields make each PD row
self-contained for plotting.

### Computation Rules

- Continuous spline terms must reuse fitted knots from the model builder, as
  the current PD implementation already does.
- `knots` are internal knots only.
- `boundary_knots` are min/max boundary knots used by the fitted basis.
- Monotone splines set `term_type="ms"` and include the monotonicity direction
  when available.
- Linear terms set `knots=None`.
- Target/frequency encoded categoricals set `term_type` accordingly; their
  `grid_values` are levels or encoded level groups, not arbitrary numeric grids.

### Tests

1. A fitted `bs(x, ...)` term emits PD with `term_type="bs"` and non-empty
   `knots`.
2. A fitted `ms(x, ...)` term emits `term_type="ms"` and monotonicity metadata.
3. A linear term emits no knots.
4. PD predictions are unchanged versus current implementation.
5. Serialized JSON contains enough fields to draw a curve with knot markers
   without consulting private model state.

---

## RS-DST-005: Encoding Diagnostics For Categorical Terms

### Problem

A GBM-to-GLM report must explain whether the GLM used:

- ordinary categorical treatment encoding;
- target encoding;
- frequency encoding;
- grouped categorical columns created upstream;
- target-encoded interactions.

RustyStats can see the fitted model representation. It should expose that
representation cleanly. It should not decide whether Destyler should have used
target encoding; that selection remains upstream.

### Data Schema

Add:

```python
@dataclass
class EncodingDiagnostics:
    name: str
    kind: str                      # categorical, target_encoding, frequency_encoding, grouped_categorical, unknown
    in_model: bool
    n_levels_train: int | None = None
    n_levels_test: int | None = None
    unseen_levels_test: int | None = None
    rare_levels_grouped: int | None = None
    interaction_order: int = 1
    source_factors: list[str] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

@dataclass
class ModelDiagnostics:
    ...
    encoding_diagnostics: list[EncodingDiagnostics] | None = None
```

### Computation Rules

- Use fitted `feature_names` and term slots, not substring matching.
- `TE(var)` maps to `kind="target_encoding"`.
- `FE(var)` maps to `kind="frequency_encoding"`.
- `C(var)[...]` maps to `kind="categorical"`.
- Interaction TE such as `TE(a:b)` maps to `interaction_order=2` and
  `source_factors=["a", "b"]`.
- A grouped categorical is not intrinsically detectable unless the grouped
  column is what RustyStats sees. If the factor name ends with a common suffix
  such as `_grp`, mark `kind="grouped_categorical"` only as a heuristic and
  add a note. Do not infer upstream grouping maps.

### Tests

1. Plain categorical term reports `kind="categorical"`.
2. Target encoded main effect reports `kind="target_encoding"`.
3. Frequency encoded main effect reports `kind="frequency_encoding"`.
4. Target encoded interaction reports the correct source factors and order.
5. Test data with unseen levels populates `unseen_levels_test`.

---

## RS-DST-006: Higher-Order Interaction Diagnostics

### Problem

Pair diagnostics exist. Destyler can discover and select three-way interactions
from the GBM teacher. If a 3-way interaction is fitted in the GLM, RustyStats
should expose at least block-level diagnostics for that fitted term.

Full 3D surfaces are useful but heavier. The minimum required RustyStats support
is block significance, coefficients, representation and GVIF for fitted
higher-order interactions.

### Public API

Extend `interactions=[...]` to accept length-3 specs:

```python
result.diagnostics(
    train_data=train,
    interactions=[
        ("VehAge", "BonusMalus"),
        ("VehAge", "BonusMalus", "Region"),
    ],
)
```

### Data Schema

Generalize the existing `InteractionDiagnostics` or add a parallel block type:

```python
@dataclass
class InteractionBlockDiagnostics:
    name: str                         # "a:b:c"
    factors: list[str]
    order: int
    in_model: bool
    representation: str | None        # tensor_product, target_encoding, frequency_encoding, unknown
    coefficients: list[FactorCoefficient] | None = None
    significance: FactorSignificance | None = None
    score_test: ScoreTestResult | None = None
    gvif: float | None = None
```

If we keep `InteractionDiagnostics`, add:

```python
factors: list[str]
order: int
```

and keep the current pair fields for backwards compatibility.

Optional later extension:

```python
@dataclass
class SurfaceCube:
    axes: list[str]
    axis_types: list[str]
    cells: list[SurfaceCubeCell]
```

Do not make 3D cube output mandatory for this phase. It can explode in sparse
categorical data and is more a visualization concern than a minimum actuarial
diagnostic.

### Computation Rules

- For fitted higher-order interactions, identify the term slot exactly.
- Emit coefficient block, Wald significance and block GVIF.
- For unfitted higher-order interactions, emit a score test only if the design
  expansion can be built safely under existing memory caps.
- If a 3-way request would exceed memory/cell caps, return a structured warning
  rather than silently omitting it.

### Tests

1. Fitted 3-way interaction reports `order=3`, `in_model=True`, coefficients and
   significance.
2. Target-encoded 3-way categorical interaction reports
   `representation="target_encoding"`.
3. Unfitted 3-way request either returns a score test or a clear
   `interaction_score_test_unavailable` warning.
4. Existing pair diagnostics JSON remains backwards compatible.

---

## RS-DST-007: Base-Aware Factor Train/Test Stability

### Problem

For GBM-to-GLM review, factor stability should answer:

- did the actual experience move from train to test?
- did the GLM move with it?
- did the GBM teacher move differently?
- is the student worse than the teacher for specific levels or bands?

`train_test_bins` is the right container, but it needs the base overlay from
RS-DST-002.

### Required Output

For each factor bin where base predictions are available, `train_test_bins`
must allow downstream code to compute:

```text
train_glm_error  = abs(train_ae_ratio - 1)
train_base_error = abs(train_base_ae_ratio - 1)
test_glm_error   = abs(test_ae_ratio - 1)
test_base_error  = abs(test_base_ae_ratio - 1)
student_minus_base_error = test_glm_error - test_base_error
```

Do not add verdict fields. Destyler can decide how to highlight bins.

### Tests

1. A bin where the base model is closer to A/E=1 has smaller base error from the
   emitted fields.
2. Train/test factor bins stay cell-aligned with and without base predictions.
3. Missing test base predictions leave test base fields null but keep GLM
   train/test fields populated.

---

## RS-DST-008: Documentation And Examples

### Required Documentation

Update:

- `docs/api/diagnostics.md`
- `docs/components/diagnostics.md`
- `docs/theory/diagnostics.md` if formulas are described there
- any relevant examples using `base_predictions`

Docs must explicitly state:

- `base_predictions` are response-scale predictions;
- with exposure, diagnostics rank by predicted rate under `ranking="auto"`;
- exposure is not a prior weight;
- prior weights multiply actual, expected and exposure totals;
- factor A/E base overlays are emitted only when base predictions are supplied;
- 3-way diagnostics are block diagnostics unless optional cube output is
  requested in a future phase.

### Example

```python
diag = glm.diagnostics(
    train_data=train.with_columns(pl.Series("gbm_mu", gbm_train)),
    test_data=test.with_columns(pl.Series("gbm_mu", gbm_test)),
    categorical_factors=["VehBrand", "VehGas", "Region"],
    continuous_factors=["VehAge", "DrivAge", "BonusMalus"],
    base_predictions="gbm_mu",
    exposure="Exposure",
    weights="policy_weight",
    ranking="auto",
)
```

The docs should show how to retrieve:

```python
diag.base_predictions_by_role.test
diag.factors[0].actual_vs_expected[0].base_expected
diag.factors[0].train_test_bins[0].test_base_predicted
diag.partial_dependence[0].knots
```

---

## Implementation Order

1. **RS-DST-001**: base predictions by train/test role.
2. **RS-DST-002**: base fields on factor A/E and train/test bins.
3. **RS-DST-003**: prior-weighted factor diagnostics.
4. **RS-DST-004**: self-contained PD + knot metadata.
5. **RS-DST-005**: encoding diagnostics.
6. **RS-DST-006**: higher-order interaction block diagnostics.
7. **RS-DST-008**: docs and examples.

Items 1-3 are the minimum needed for Destyler's report card factor charts.
Items 4-5 are needed for transparent GLM specification reporting.
Item 6 is important for complete 3-way interaction governance, but the first
Destyler report can still show teacher-discovered 3-way interactions from its
own decomposition before RustyStats implements 3-way GLM block diagnostics.

---

## Out Of Scope For RustyStats

These remain Destyler responsibilities:

- GBM/CatBoost training.
- fANOVA decomposition of the GBM teacher.
- selecting GLM terms from the teacher.
- deciding whether to use plain categorical, grouped categorical or target
  encoding.
- recording why a selected term came from a main effect, 2-way interaction or
  3-way interaction.
- HTML report rendering.
- promotion gates and model committee verdicts.
- golden-row parity across Destyler bundles.

RustyStats should provide the statistically correct GLM diagnostics that make
those downstream workflows defensible.
