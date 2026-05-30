# Destyler Methodology Diagnostics Implementation Plan

Status: implemented
Scope: RustyStats diagnostics, preserving existing public fields and fast paths

This plan implements `destyler-methodology-diagnostics-support.md` for two
related workflows:

- **Teacher-guided GLM validation**: Destyler trains a GBM teacher, uses it to
  guide a GLM build, then passes teacher predictions into RustyStats diagnostics
  to compare the GLM student against the teacher.
- **Challenger-vs-incumbent validation**: a modeller trains a new model and
  passes existing production or benchmark predictions into RustyStats diagnostics
  to compare the new model against the incumbent.

RustyStats should treat `base_predictions` as generic benchmark/reference
predictions. It should not care whether they came from a GBM teacher, a current
production GLM, a manual tariff, or another external model.

## Engineering Principles

1. **Additive compatibility**
   Keep all existing fields valid. Add optional fields to dataclasses rather
   than renaming or deleting current fields.

2. **One aggregate contract**
   For all reportable aggregates:

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

   If no prior weights are supplied, `w_i = 1`. If no exposure is supplied,
   `exposure_i = 1`.

3. **Preserve fast unweighted paths**
   Existing Rust batch kernels should remain the default for unweighted,
   no-base diagnostics. Weighted and base-aware paths may use vectorized Python
   fallbacks first, then graduate to Rust kernels if benchmarks require it.

4. **Response-scale benchmark predictions**
   `base_predictions` must be row-level response predictions, not link-scale
   values and not exposure-normalized rates. For frequency models with exposure,
   this means expected claim counts, not claim frequencies.

5. **RustyStats owns methodology, not narrative**
   Emit totals, rates, intervals, metadata, significance, GVIF, and score-test
   availability. Leave verdicts, model-committee language, HTML layout, GBM
   decomposition, and term-selection rationale to Destyler or other downstream
   tools.

## Implementation Stages

### 1. Schema Additions

- Add `BasePredictionsByRole`.
- Add optional total and base-overlay fields to `ActualExpectedBin`.
- Add optional total and base-overlay fields to `FactorBinPair`.
- Add optional plotting metadata to `PartialDependence`.
- Add `EncodingDiagnostics`.
- Add optional fields to `ModelDiagnostics`:
  - `base_predictions_by_role`
  - `encoding_diagnostics`
  - optionally `interaction_blocks` if higher-order interactions use a parallel
    block type.

Acceptance:

- Existing JSON fields remain present with the same names and meanings.
- New optional fields serialize as `null` when not populated.

### 2. Base Predictions By Role

- Keep `base_predictions="column"` compatibility.
- If `test_data` contains the same column, compute a test-side comparison.
- If `test_data` is present but the column is absent, emit a structured warning
  and leave the test comparison `None`.
- Add dictionary support:

  ```python
  base_predictions={"train": "gbm_oof_mu", "test": "gbm_test_mu"}
  ```

- Keep `ModelDiagnostics.base_predictions_comparison` as the train-side
  compatibility alias.

Acceptance:

- Train-only use behaves as before.
- Same-column train/test use populates `base_predictions_by_role.train` and
  `.test`.
- Dict input supports different train/test benchmark prediction columns.
- Missing test benchmark predictions warn without failing.

### 3. Weighted And Base-Aware Base Comparison

- Update base comparison to accept optional prior weights.
- Totals and A/E must use `sum(w * value)`.
- Gini must use the same weighted Gini helper used by lift/decile diagnostics.
- Keep unweighted results stable within existing rounding.

Acceptance:

- Non-uniform weights change totals exactly as hand-computed.
- `weights=np.ones(n)` matches unweighted output within current rounding.

### 4. Factor A/E Totals And Base Overlay

- Populate totals on `ActualExpectedBin`:
  - `actual_total`
  - `expected_total`
- When benchmark predictions are supplied, populate:
  - `base_expected`
  - `base_expected_total`
  - `base_ae_ratio`
- Keep `actual` and `expected` as rates.
- Preserve existing bin labels and ordering.

Acceptance:

- No-base output remains backwards compatible.
- Base-aware bins match hand-computed weighted totals and rates.

### 5. Weighted Factor Diagnostics

- Thread prior weights into:
  - `_FactorDiagnosticsComputer`
  - categorical A/E bins
  - continuous A/E bins
  - dataset categorical factor diagnostics
  - dataset continuous diagnostics
  - train/test factor-bin construction
- Do not reuse train-row array weights for test rows.

Acceptance:

- Categorical and continuous factor bins obey the aggregate contract.
- Missing fitted weight column on test data keeps the existing
  `test_weights_unavailable` warning behavior.

### 6. Base-Aware Train/Test Factor Stability

- Extend `FactorBinPair` population with train/test totals and benchmark fields.
- Continuous test bins continue using train-side bin edges.
- Categorical test bins continue using train-side labels.
- Empty test bins report `test_n=0`, `test_exposure=0`, and nullable
  rate/total fields.

Acceptance:

- Downstream code can compute GLM error, benchmark error, and
  student-minus-benchmark error per train/test bin.

### 7. Partial Dependence Metadata

- Populate:
  - `term_type`
  - `prediction_scale="response"`
  - `relativity_base`
  - `knots`
  - `boundary_knots`
  - `monotonicity`
- Source metadata from term slots and fitted spline info, not substring
  guessing.
- Keep prediction values unchanged.

Acceptance:

- Serialized PD rows are self-contained enough to draw curves with knot markers.

### 8. Encoding Diagnostics

- Add `encoding_diagnostics` built from term slots and fitted feature names.
- Detect:
  - ordinary categorical terms
  - target encoding
  - frequency encoding
  - target/frequency encoded interactions
  - heuristic grouped categoricals with a note
- Populate unseen test levels when `test_data` is supplied.

Acceptance:

- Main-effect and interaction encodings are explicit in diagnostics JSON.

### 9. Higher-Order Interaction Blocks

- Extend diagnostics `interactions=[...]` to accept length-3 specs.
- Emit block-level diagnostics for fitted higher-order interactions:
  - factors
  - order
  - representation
  - coefficients
  - Wald significance
  - GVIF
- For unfitted higher-order requests, emit score tests only when safe; otherwise
  emit a structured warning.
- Do not require 3D surface output in this phase.

Acceptance:

- Existing pair diagnostics JSON remains compatible.
- 3-way fitted interactions are visible as block diagnostics.

### 10. Documentation, Tests, And Benchmarks

- Update diagnostics docs and examples.
- Add hand-computed unit tests for weighted/base aggregates.
- Add compatibility tests for train-only `base_predictions`.
- Add train/test and dict-input tests.
- Add parity tests for `weights=np.ones(n)`.
- Add or extend benchmarks for unweighted, weighted, and base-aware diagnostics.

Acceptance:

- Focused test suite passes.
- Existing unweighted diagnostics performance is not materially regressed.

## Implementation Notes

- Prefer small shared helpers for weighted aggregate math so every diagnostic
  path uses the same formulas.
- Use additive dataclass fields to keep serialized consumers stable.
- Keep warnings structured and machine-readable.
- Measure before moving weighted/base aggregation into Rust kernels; vectorized
  Python is acceptable for the first implementation if the common no-base path
  remains fast.

## Verification

- `uv run ruff check python/rustystats/diagnostics tests/python/test_diagnostics.py tests/python/test_rate_ranked_diagnostics.py`
- `uv run pytest tests/python/test_rate_ranked_diagnostics.py::TestBasePredictionDiagnostics tests/python/test_diagnostics.py::TestDestylerMethodologyExtensions -q`
- `uv run pytest tests/python/test_diagnostics.py tests/python/test_rate_ranked_diagnostics.py -q`
