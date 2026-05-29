# Actuarial Methodology Hardening Spec

Status: proposed
Scope: RustyStats only
Audience: maintainers implementing actuarial/pricing-grade GLM workflows

This spec lists the RustyStats changes needed before the library should be
treated as the statistical engine for production insurance pricing workflows,
especially workflows that combine exposure offsets, target encoding,
regularization, splines, Tweedie/NB families, and model diagnostics.

The goal is not to change the modelling philosophy. The goal is to remove
silent methodological traps: target leakage, ambiguous exposure semantics,
incorrect family parameter propagation, diagnostics that rank exposure size
instead of risk, and solver/family edge cases that can make production evidence
look stronger than it is.

## Summary

| ID | Priority | Change | Required before production pricing? |
| --- | --- | --- | --- |
| RS-ACT-001 | P0 | Fold-safe CV for target encoding and weighted validation scoring | Yes, if CV or target encoding is used |
| RS-ACT-002 | P0 | Separate raw exposure from link-scale offset | Yes, if exposure, frequency rates, or exposure-weighted TE is used |
| RS-ACT-003 | P0/P1 | Pass Tweedie `var_power` and NB `theta` through smooth GLM | Yes, if smooth Tweedie/NB is used |
| RS-ACT-004 | P1 | Rank actuarial diagnostics by predicted rate when exposure is present | Yes for pricing diagnostics |
| RS-ACT-005 | P1 | Compute regularization alpha path from the GLM score | Strongly recommended |
| RS-ACT-006 | P1 | Tighten Tweedie support contract | Strongly recommended |
| RS-ACT-007 | P1 | Make IRLS step-halving and final fitted values robust | Strongly recommended |
| RS-ACT-008 | P2 | Label quasi-likelihood diagnostics correctly | Recommended |
| RS-ACT-009 | P1 | Add explicit calibration diagnostics and calibration primitives | Strongly recommended |
| RS-ACT-010 | P1 | Estimate or explicitly require Negative Binomial `theta` | Strongly recommended if NB is supported |
| RS-ACT-011 | P1 | Make constrained/penalized inference honest and solver-status aware | Strongly recommended |

P0 items should be fixed or explicitly blocked by runtime errors before any
pricing team relies on affected functionality. P1 items can be sequenced after
P0, but they are still part of the target production-grade statistical engine.

## Non-Goals

- No Atelier changes are included here.
- No Destyler pipeline changes are included here.
- No new actuarial UI workflow is included here.
- No GBM/CatBoost teacher calibration policy is included here. RustyStats
  should provide calibration primitives for GLM predictions and generic
  prediction arrays; Destyler should decide how those primitives are used in a
  GBM-to-GLM workflow.
- This spec does not require changing the core GLM API style or replacing the
  Rust backend.

## RS-ACT-001: Fold-Safe CV For Target Encoding

### Problem

`glm_dict` currently builds the design matrix before CV. If the design matrix
contains target-encoded columns, the encodings can use target information from
rows that later become validation folds. The CV path then slices the prebuilt
matrix. This leaks validation signal into model selection.

The validation score also uses an unweighted mean of unit deviances in parts of
the CV path. For actuarial pricing, model selection must score the same weighted
objective that the fit is optimizing.

Relevant code:

- `python/rustystats/formula.py`: `FormulaGLMDict.__init__` builds `self.X`
  before fitting.
- `python/rustystats/interactions.py`: target encoding is built from full
  `y`/exposure state.
- `python/rustystats/regularization_path.py`: CV slices a prebuilt `X`.
- `crates/rustystats/src/fitting_py.rs`: `fit_cv_path_py` operates on already
  materialized arrays.

### Required Semantics

When CV is requested:

1. The train/validation split must be created before any stateful transform is
   fit.
2. For each fold, target encoders must be fit using only the fold training rows.
3. Validation rows must be transformed using the fold-trained encoder state.
4. Unseen validation levels must map to the fold-training prior.
5. Exposure-weighted target encoding must use fold-training claims and
   fold-training raw exposure only.
6. Validation deviance must be scored with validation weights when weights are
   supplied.
7. Validation deviance must not include the regularization penalty.
8. CV fold fits must use the requested convergence settings by default.
   A faster approximate mode is acceptable only if it is explicit, named, and
   recorded in the result metadata.
9. The result must record enough metadata to explain the selected alpha:
   fold scores, mean score, standard error, selected alpha, scoring objective,
   convergence settings, and whether stateful fold transforms were used.

### API

No public API change is required for the final behavior. Existing calls such as
the following must become fold-safe:

```python
rs.glm_dict(
    response="claim_count",
    terms={"brand": {"type": "target_encoding"}},
    data=df,
    family="poisson",
    exposure="exposure",
    alpha="cv",
)
```

If implementation is staged, add a temporary fail-closed guard:

- If CV is requested and the parsed formula contains any target-encoded term or
  target-encoded interaction, raise a clear error until fold-safe transforms
  are implemented.
- Do not silently continue with prebuilt full-data encodings.

### Implementation Plan

Introduce a fold-aware design matrix path.

Option A, preferred:

- Split `InteractionBuilder` into a fit/transform API:
  - `fit_design_matrix_state(parsed, data, y, exposure, seed) -> state`
  - `transform_design_matrix(parsed, data, state) -> X, feature_names`
- The state should contain categorical level orderings, spline knots, target
  encoding priors/stats, expression metadata, and final feature names.
- Prediction should use the same transform state already stored on the fitted
  model.

Option B, acceptable first implementation:

- Add a private fold builder in `FormulaGLMDict` that rebuilds a complete
  `InteractionBuilder` per fold.
- Persist fold target-encoding state only for diagnostics.
- Keep production prediction using the final full-training design matrix state.

The Rust `fit_cv_path_py` function can remain for numeric, already-materialized
matrices with stateless transforms. Formula/dict models with target encoding
should run a Python-managed CV loop that calls the normal Rust fit function on
each fold-specific design matrix.

Validation scoring must use:

```text
score = sum_i w_i * deviance_i / sum_i w_i
```

where `w_i = 1` if no prior weights are supplied. Offsets and exposure enter
the model prediction, not the validation denominator, unless the user supplied
prior weights.

### Tests

Add tests covering:

1. CV plus target encoding does not use validation targets.
   - Build a small dataset where one rare level appears only in one fold.
   - Assert that the validation encoding for that level is the training prior,
     not the level's full-data target rate.
2. Fold-safe CV matches a hand-written manual fold calculation.
3. CV with target-encoded interactions is also fold-safe.
4. CV scoring changes when validation weights are changed.
5. CV with no target encoding continues to use the fast array path where safe.
6. CV defaults to strict convergence settings rather than hidden relaxed
   settings.
7. Approximate/fast CV mode, if implemented, records its relaxed settings.
8. The temporary fail-closed guard, if used, raises on target encoding plus CV.

## RS-ACT-002: Separate Raw Exposure From Link-Scale Offset

### Problem

RustyStats currently gives `offset` two meanings:

- A string offset for a log-link family is treated as raw exposure and logged.
- An array offset is treated as already link-scale and is not logged.

That distinction is valid for fitting, but target encoding and rate diagnostics
need raw positive exposure. Today this concretely breaks exposure-weighted
target encoding: a link-scale array offset is taken verbatim as the rate
denominator (`_get_raw_exposure` returns the array unchanged), so
`offset=np.log(exposure)` produces a `sum(log(exposure))` denominator and
degenerate encodings. Rate diagnostics do not currently mis-read an array
offset; they fall back to unit exposure. That still silently drops exposure
weighting, and once a genuine link-scale offset is separable from exposure,
diagnostics, `explore`, and complement-of-credibility division must source raw
exposure from `exposure`, never from a string `offset`. For example,
`offset=np.log(exposure)` is correct for fitting a Poisson frequency GLM, but
`np.log(exposure)` is not a valid denominator for exposure-weighted target
encoding.

Relevant code:

- `python/rustystats/formula.py`: `_process_offset`
- `python/rustystats/formula.py`: `_get_raw_exposure`
- `python/rustystats/formula.py`: prediction offset resolution
- `python/rustystats/interactions.py`: exposure-weighted target encoding
- `crates/rustystats-core/src/target_encoding/mod.rs`: exposure denominators

### Required Semantics

Add an explicit `exposure` concept.

Definitions:

- `exposure`: raw positive denominator for rate/frequency models.
- `offset`: arbitrary link-scale additive offset.
- For log-link rate models with exposure, the model linear predictor is:

```text
eta_total = eta_terms + log(exposure) + offset
mu = exp(eta_total)
```

- For non-log links, `exposure` must either be rejected or documented as
  diagnostic-only. The default should be to reject unsupported combinations.
- Exposure-weighted target encoding must use `exposure`, never `offset`.
- Rate-ranked diagnostics must use `mu / exposure` when exposure is present.

### Public API

Add `exposure` to model construction and prediction:

```python
rs.glm_dict(
    response,
    terms,
    data,
    family="poisson",
    exposure=None,
    offset=None,
    weights=None,
    ...
)

result.predict(new_data, exposure=None, offset=None)
result.model.predict(new_data, exposure=None, offset=None)
```

Backward compatibility:

1. Preserve `offset="Exposure"` for log-link families as a legacy alias for
   raw exposure when `exposure is None`.
2. Internally normalize that case to `exposure="Exposure"` and
   `offset=None`.
3. Preserve `offset=np.array(...)` as link-scale offset. Do not use it as raw
   exposure.
4. If an array offset is supplied and a target-encoded term requests
   exposure-weighted encoding, require `exposure=` or fall back to unweighted
   target encoding only with an explicit warning. Prefer raising in strict
   production mode.
5. If both `exposure` and `offset` are supplied, add both contributions.

### Validation

For log-link exposure models:

- Exposure must be finite.
- Exposure must be strictly positive.
- Exposure length must match the data length.
- Prediction-time exposure must be supplied if the model was fit with exposure
  and the new data does not contain the stored exposure column.

For target encoding:

- Exposure-weighted target encoding requires strictly positive exposure.
- If exposure is missing, use observation-weighted target encoding and mark
  `used_exposure_weighted=False`.
- If `offset` is an array and exposure is missing, do not assume
  `exp(offset)` is exposure. That would be wrong when the offset contains other
  link-scale adjustments.

### Storage And Serialization

Store both specs separately:

```python
_exposure_spec
_offset_spec
_offset_is_legacy_exposure_alias
```

Serialized models must include:

- `exposure_spec`
- `offset_spec`
- whether exposure contributed `log(exposure)` to the fit
- whether target encoding used exposure-weighted statistics

Existing serialized models with only `offset_spec` and
`offset_is_exposure=True` should load as legacy exposure models.

> **Superseded.** The `offset_is_exposure` flag, the `_offset_is_legacy_exposure_alias`
> field, and the `offset="Exposure"` → exposure normalization described in this
> section were later removed. A string `offset` is now always a verbatim
> link-scale column; raw exposure comes only from `exposure=`. Serialized
> payloads no longer carry `offset_is_exposure`, and legacy payloads are no
> longer migrated. This section is retained as a historical record of the
> RS-ACT-002 design.

### Tests

Add tests covering:

1. `exposure="Exposure"` and legacy `offset="Exposure"` produce identical
   fitted values for Poisson log-link models.
2. `offset=np.log(exposure), exposure=exposure` produces the same fit as
   `exposure=exposure`.
3. `offset=np.log(exposure)` without `exposure=` does not use
   exposure-weighted target encoding.
4. Exposure-weighted target encoding uses `sum(y) / sum(exposure)`, not
   `sum(y) / sum(offset)`.
5. Prediction with stored exposure column works.
6. Prediction with new exposure array works.
7. Prediction errors clearly when exposure is required but unavailable.
8. Serialization round-trips `exposure_spec` and `offset_spec`.

## RS-ACT-003: Smooth GLM Must Preserve Family Parameters

### Problem

The non-smooth GLM path accepts and passes through Tweedie `var_power` and
negative-binomial `theta`. The smooth GLM binding currently constructs Tweedie
and NB families with hard-coded defaults.

Relevant code:

- `python/rustystats/formula.py`: family parameter storage and fit dispatch
- `crates/rustystats/src/fitting_py.rs`: `fit_smooth_glm_unified_py`

### Required Semantics

Smooth and non-smooth fits must use the same family parameters.

For Tweedie:

```python
rs.glm_dict(..., family="tweedie", var_power=1.2)
```

must fit the smooth model with `var_power=1.2`, not `1.5`.

For negative binomial:

```python
rs.glm_dict(..., family="negbinomial", theta=3.0)
```

must fit the smooth model with `theta=3.0`, not `1.0`.

### Implementation Plan

1. Add `var_power` and `theta` to the PyO3 signature for
   `fit_smooth_glm_unified_py`.
2. Pass those values to `family_from_name`.
3. Update the Python smooth fit path to pass `self.var_power` and `self.theta`.
4. Ensure model summaries/reporting display the actual parameters used.

### Tests

Add tests covering:

1. Smooth Tweedie with non-default `var_power` passes the chosen value through
   to the smooth fit. Assert on family metadata and variance/covariance or
   smooth-vs-non-smooth agreement at the same `var_power`, not on a flaky
   coefficient difference from the default.
2. Smooth NB with non-default `theta` differs from default.
3. The returned family name/metadata records the chosen parameter for both
   Tweedie and NB.
4. Non-smooth and smooth dispatch receive the same parameter values.

## RS-ACT-004: Rate-Ranked Actuarial Diagnostics

### Problem

For exposure-offset frequency models, raw expected count `mu` combines risk and
exposure size. Ranking policies by `mu` can rank large exposure ahead of high
rate. Pricing diagnostics such as lift, calibration by decile, Lorenz curves,
and concentration charts usually need to sort by predicted risk rate when
exposure is present.

Some discrimination metrics already rate-rank when exposure is present. This
fix should harmonize those paths with decile/calibration tables and Python lift
charts so diagnostics are internally consistent.

Relevant code:

- `crates/rustystats-core/src/diagnostics/calibration.rs`
- `crates/rustystats-core/src/diagnostics/decile.rs`
- `crates/rustystats-core/src/diagnostics/factor_diagnostics.rs`

### Required Semantics

For models with raw exposure:

```text
risk_score_i = mu_i / exposure_i
```

Use `risk_score` for sorting/ranking. Continue aggregating actual and expected
counts for A/E:

```text
actual = sum(y_i)
expected = sum(mu_i)
exposure = sum(exposure_i)
actual_rate = actual / exposure
expected_rate = expected / exposure
ae_ratio = actual / expected
```

If prior weights are supplied, diagnostics must document whether aggregates are
weighted. The preferred production default is:

```text
actual = sum(w_i * y_i)
expected = sum(w_i * mu_i)
exposure = sum(w_i * exposure_i)
```

when weights represent portfolio importance. If weights represent frequency
counts or sampling weights, this is still the least surprising convention, but
the docs must be explicit.

For severity models without exposure, keep sorting by `mu`.

### API

Diagnostics should accept optional exposure explicitly or read it from the
fitted model metadata:

```python
result.diagnostics(exposure=None)
result.calibration_table(exposure=None)
result.lift_chart(exposure=None)
```

Add an escape hatch for advanced users:

```python
ranking="auto" | "mean" | "rate"
```

Defaults:

- `auto`: use rate when exposure is present, otherwise mean.
- `mean`: rank by `mu`.
- `rate`: require exposure and rank by `mu / exposure`.

### Tests

Add tests covering:

1. A dataset where raw `mu` and `mu/exposure` produce different decile order.
2. Calibration bins use rate ranking but aggregate counts.
3. Lorenz/lift charts use rate ranking when exposure exists.
4. Diagnostics without exposure preserve old ordering.
5. Weighted diagnostics produce expected weighted totals.

## RS-ACT-005: GLM-Score-Based Regularization Alpha Path

### Problem

The current alpha path can be based on centered raw `y` and does not fully
respect the family, link, offset, weights, or intercept-only solution. For GLM
lasso/elastic-net, the maximum alpha should come from the GLM score at the
null/intercept solution.

Relevant code:

- `python/rustystats/regularization_path.py`: `compute_alpha_max`
- `crates/rustystats/src/fitting_py.rs`: fallback alpha grid

### Required Semantics

For an elastic-net objective with L1 ratio `l1_ratio > 0`, compute:

```text
alpha_max = max_j |score_j| / l1_ratio
```

where `score_j` is the derivative of the unpenalized GLM loss with respect to
coefficient `j`, evaluated at the intercept-only solution including offsets and
weights. Exclude the intercept and any explicitly unpenalized columns.

The exact scaling must match the solver objective. If the solver divides the
loss by `sum(weights)` or `n`, the alpha path must use the same denominator.

For `l1_ratio == 0` ridge paths, `alpha_max` is not defined by an all-zero
solution. Use a documented ridge-specific grid.

### Implementation Plan

1. Add a helper that fits or computes the offset-aware null model.
2. Compute `mu0` from the active family/link and offset.
3. Compute per-feature score using the same weights and loss scaling as the
   solver.
4. Exclude intercept and unpenalized constrained columns if applicable.
5. Use this alpha max in both Python path generation and Rust fallback grids.

### Tests

Add tests covering:

1. At `alpha >= alpha_max`, all penalized coefficients are zero for lasso.
2. Offset changes `alpha_max` when exposure distribution changes.
3. Weights change `alpha_max`.
4. Gaussian identity matches RustyStats' own solver scaling, using raw weighted
   sums and no implicit `1/n` scaling.
5. Poisson log-link alpha max matches a finite-difference score check.

## RS-ACT-006: Tighten Tweedie Support Contract

### Problem

Actuarial pure premium Tweedie workflows usually require compound
Poisson-gamma behavior with:

```text
1 < p < 2
```

The current family code allows a broader set of powers. Some powers need
different support rules and edge-case handling, especially around zeros.

Relevant code:

- `crates/rustystats-core/src/families/tweedie.rs`
- `python/rustystats/validation.py`

### Required Semantics

Default public actuarial Tweedie support should be:

```text
1 < var_power < 2
y >= 0
mu > 0
```

If extended Tweedie powers remain available, they must be explicitly requested:

```python
rs.glm_dict(..., family="tweedie", var_power=2.5, allow_extended_tweedie=True)
```

and must validate the correct support for that regime. Do not silently apply
compound Tweedie assumptions outside their support.

Treat the out-of-range cases differently:

- `0 < var_power < 1` is mathematically invalid; no Tweedie distribution exists
  here. Reject it always, including under `allow_extended_tweedie=True`.
- `var_power <= 0`, `var_power == 1`, `var_power == 2`, and `var_power > 2`
  are valid distributions but outside the default compound Poisson-gamma
  interior. These require `allow_extended_tweedie=True` and must validate the
  support for that regime. In particular, `var_power >= 2` requires `y > 0`
  with no exact zeros, and `var_power == 2` is exactly Gamma.

### Tests

Add tests covering:

1. `0 < var_power < 1` errors always, even with
   `allow_extended_tweedie=True`.
2. `var_power <= 0`, `var_power == 1`, `var_power == 2`, and `var_power > 2`
   error by default but are accepted under `allow_extended_tweedie=True`.
3. `1 < var_power < 2` accepts zeros and positives.
4. `var_power == 2` routes to Gamma or errors clearly, and rejects exact zeros
   rather than returning infinite deviance.
5. Extended `var_power >= 2` rejects exact zeros because it requires `y > 0`.

## RS-ACT-007: IRLS Step-Halving And Final Fitted Values

### Problem

IRLS step-halving should not accept a final step that is worse than the
previous accepted deviance. Final fitted values should also be clamped through
the family rules consistently.

Relevant code:

- `crates/rustystats-core/src/solvers/irls.rs`

### Required Semantics

During each iteration:

1. Start from the previous accepted coefficients and deviance.
2. Try the full step.
3. If deviance worsens beyond a small relative tolerance, or becomes invalid
   through non-finite `eta`, `mu`, or deviance, halve the step.
4. Accept the best valid step. A step that leaves deviance unchanged within
   tolerance is normal convergence at the optimum and must be accepted, not
   treated as a failure. The convergence flag must derive from an accepted,
   non-worsening step; a worse-but-close step must not be reported as
   converged.
5. If no non-worsening step exists after the halving budget:
   - keep the previous accepted coefficients;
   - set convergence status to failed or step-halved-no-improvement;
   - expose a warning/status in Python results.

Final fitted values must use:

```text
mu = family.clamp_mu(link.inverse(eta))
```

not just `link.inverse(eta)`.

### Tests

Add tests covering:

1. A constructed case where the full step worsens deviance and the half step
   improves it.
2. A constructed case where no step improves deviance; previous coefficients
   are retained.
3. Final fitted values satisfy each family clamp domain.
4. Python results expose a useful convergence warning/status.

## RS-ACT-008: Quasi-Likelihood Diagnostics

### Problem

Quasi-Poisson and quasi-binomial models do not have ordinary full likelihoods
in the same sense as Poisson and binomial models. Reporting AIC/BIC/log
likelihood as if they were true likelihood diagnostics can mislead model
selection.

Relevant code:

- `crates/rustystats-core/src/families/quasi.rs`
- `python/rustystats/glm.py`

### Required Semantics

For quasi families:

- Label log-likelihood-like values as quasi-likelihood where appropriate.
- Return `NA`/`None` for AIC and BIC unless a documented quasi-information
  criterion is implemented.
- Summaries must not display quasi AIC/BIC as ordinary likelihood AIC/BIC.

### Tests

Add tests covering:

1. Quasi-Poisson summary does not label AIC/BIC as ordinary likelihood values.
2. Quasi-binomial summary behaves the same.
3. Serialization preserves quasi diagnostic metadata.

## RS-ACT-009: Explicit Calibration Diagnostics And Primitives

### Problem

Pricing models need to be checked for balance and calibration, not just
predictive deviance. A model can rank risks well but still over- or
under-predict the total premium/claims level, or be balanced overall but badly
miscalibrated by prediction band or important rating factor.

RustyStats should own the statistical primitives for calibration because they
apply directly to GLM predictions, diagnostics, and serialized model evidence.
However, RustyStats should not decide whether a GBM teacher, distilled GLM, or
product workflow is allowed to deploy a calibrated scorer. Those decisions
belong upstream in Destyler or downstream in the product/governance layer.

### Definitions

For response `y`, prediction `mu`, optional exposure `e`, and optional prior
weights `w`:

```text
actual = sum(w_i * y_i)
expected = sum(w_i * mu_i)
ae_ratio = actual / expected
global_factor = actual / expected
```

If no weights are supplied, use `w_i = 1`.

For exposure models, calibration tables should still aggregate expected counts
or costs with `mu`, but prediction bands should usually be formed by predicted
rate:

```text
risk_score_i = mu_i / e_i
```

### Required Semantics

RustyStats should provide:

1. Overall calibration diagnostics:
   - actual total
   - expected total
   - A/E ratio
   - calibration factor
   - weighted and unweighted row counts
2. Calibration by prediction band:
   - decile or configurable quantile bins
   - sort by `mu / exposure` when exposure is present and ranking is `auto`
   - report actual, expected, exposure, actual rate, expected rate, and A/E
3. Calibration by factor:
   - group by one or more rating variables
   - report actual, expected, exposure, and A/E
   - allow minimum exposure/count suppression for unstable cells
4. Calibration slope/intercept diagnostics:
   - fit a simple diagnostic regression of actuals on model predictions or
     model linear predictor where appropriate
   - report as diagnostics only, not as an automatic model rewrite
5. Explicit calibration objects:
   - `GlobalCalibration`
   - optional `IsotonicCalibration`

Calibration must never be applied silently after `.fit()`. It must be either:

- a diagnostic table;
- an explicit calibration object applied to predictions; or
- an explicit GLM relevel operation that changes the intercept.

### Public API

Add standalone functions for arrays:

```python
rs.calibration_summary(
    y,
    pred,
    exposure=None,
    weights=None,
    by=None,
    n_bins=10,
    ranking="auto",
)

rs.fit_global_calibration(y, pred, weights=None)
rs.fit_isotonic_calibration(y, pred, weights=None, increasing=True)
```

Add result/model methods:

```python
result.calibration_summary(
    data=None,
    exposure=None,
    weights=None,
    by=None,
    n_bins=10,
    ranking="auto",
)

result.fit_calibration(
    data,
    method="global",
    exposure=None,
    weights=None,
)
```

The calibration objects should support:

```python
cal.factor
cal.predict(pred)
cal.to_dict()
cal = rs.GlobalCalibration.from_dict(state)
```

For isotonic:

```python
cal.predict(pred)
cal.thresholds_
cal.values_
cal.to_dict()
```

### GLM Relevel Operation

For log-link GLMs, global calibration should be available as an intercept
relevel:

```python
releveled = result.relevel(
    data=calibration_df,
    exposure="Exposure",
    weights=None,
    inplace=False,
)
```

For a calibration factor:

```text
c = sum(w_i * y_i) / sum(w_i * mu_i)
```

the log-link intercept update is:

```text
intercept_new = intercept_old + log(c)
```

This is the same `global_factor` defined above and reduces to
`sum(y) / sum(mu)` when no weights are supplied. Exposure does not enter `c`
directly: under a log link, exposure is already inside `mu` through the
`log(exposure)` offset, so `sum(w_i * mu_i)` is the exposure-weighted expected
total and `c` is already exposure-correct. The weights here are prior weights
`w`, distinct from exposure. The `exposure=` argument to `relevel()` is used
only to compute `mu` on the calibration data through prediction, which is
required when the model was fit with exposure. It is never a denominator in
`c`.

This preserves the multiplicative GLM/rate-table structure. Relativities do
not change; only the model level changes.

For non-log links, `relevel(method="global")` should either:

- raise a clear error; or
- use a documented family/link-specific intercept solve.

The first implementation should raise for unsupported links.

### Isotonic Calibration

Isotonic calibration may be useful, but it changes the scorer from:

```text
GLM
```

to:

```text
GLM + monotone calibration map
```

Therefore:

1. It must be opt-in.
2. It must be serializable as a separate object.
3. It must be clearly represented in model metadata.
4. It must not be folded into GLM coefficients.
5. Export/parity tests must compare raw GLM predictions and calibrated
   predictions separately.

### Out-Of-Sample Discipline

The API should make it easy to use calibration correctly:

- Fit the model on training data.
- Choose model structure/regularization on validation data.
- Fit calibration on a separate calibration fold, or on out-of-fold
  predictions from cross-fitting.
- Report final calibration on untouched holdout/OOT data.

RustyStats should not enforce this entire workflow, but method docs and warning
messages should state that fitting calibration on the same rows used for model
fitting can overstate calibration quality.

### Storage And Serialization

If a `GLMModel` is releveled, store:

- original intercept
- new intercept
- calibration factor
- data fingerprint if provided by caller
- row count / total weight used to relevel
- timestamp or caller-supplied run id if the serialization layer already
  supports metadata

If a separate calibration object is attached or serialized, store:

- calibration method
- factor or isotonic thresholds/values
- whether predictions were calibrated on response scale
- whether exposure/rate ranking was used for diagnostics

### Tests

Add tests covering:

1. Overall A/E equals `sum(y) / sum(pred)`.
2. Weighted A/E equals `sum(w*y) / sum(w*pred)`.
3. Exposure models rank bins by `pred / exposure` when `ranking="auto"`.
4. Calibration by factor aggregates actual, expected, exposure, and A/E
   correctly.
5. `GlobalCalibration.predict(pred)` multiplies by the fitted factor.
6. Log-link `result.relevel()` updates only the intercept and makes calibration
   data balance exactly within numerical tolerance.
7. Relativities are unchanged after log-link intercept relevel.
8. Relevel errors clearly for unsupported links.
9. Isotonic calibration is monotone, serializable, and not applied implicitly.
10. Raw predictions and calibrated predictions are both accessible after
    calibration.
11. Relevel with non-trivial prior weights balances
    `sum(w * mu_new) == sum(w * y)` within tolerance and reduces to the
    unweighted case when `w == 1`.

## RS-ACT-010: Negative Binomial Theta Contract

### Problem

Negative Binomial models need a dispersion/shape parameter, usually called
`theta`. RustyStats has a Rust binding for automatic theta estimation, but the
dict/formula GLM path can still fall back to a default `theta=1.0` when the
caller does not provide one.

For actuarial pricing, silently using `theta=1.0` is too strong an assumption.
It can materially change fitted means, standard errors, deviance, and model
comparisons.

Relevant code:

- `python/rustystats/formula.py`: dict API stores `theta` as optional and
  falls back to the default when fitting.
- `crates/rustystats/src/fitting_py.rs`: `fit_negbinomial_py` implements
  automatic theta estimation.
- `python/rustystats/regularization_path.py`: regularization paths also fall
  back to the default theta.

### Required Semantics

For `family="negbinomial"`, the API must do one of the following:

1. Estimate `theta` by default when `theta is None`; or
2. Require the caller to provide `theta`; or
3. Require an explicit opt-in such as `theta="estimate"` or
   `estimate_theta=True`.

The preferred production behavior is:

```python
rs.glm_dict(..., family="negbinomial", theta="estimate")
```

with `theta="estimate"` eventually becoming the default if the estimation path
is stable for offsets, weights, regularization, and constraints.

If theta estimation is not supported for a specific combination, such as
regularized NB or constrained NB, the model should raise a clear error unless
the user supplies a fixed numeric theta.

### Metadata

Results must record:

- `theta`
- whether theta was estimated or fixed
- initial theta
- number of theta iterations
- theta convergence status
- theta tolerance
- any fallback reason

### Tests

Add tests covering:

1. `family="negbinomial"` without theta does not silently use `1.0`.
2. `theta="estimate"` estimates theta and records metadata.
3. Numeric `theta` is treated as fixed and recorded as fixed.
4. Offset and weights are respected in theta estimation.
5. Unsupported combinations raise clear errors.
6. Regularization path behavior is explicit: fixed theta only, estimated theta
   per fold, or fail-closed.

## RS-ACT-011: Honest Inference For Constrained And Penalized Fits

### Problem

RustyStats reports standard errors, confidence intervals, p-values, AIC, and
BIC through the regular result interface. These quantities have different
interpretations when a model is regularized, selected by CV, constrained, or
fit with monotonic spline penalties.

For a pricing engine, it is better to withhold or clearly label inference than
to display ordinary GLM p-values after lasso selection, elastic-net shrinkage,
monotonic projection, or constrained optimization.

This item also covers solver status: if a constrained or penalized fit reaches
a boundary, fails to improve under step-halving, or uses an approximate
optimization route, that must be visible in the result object and summary.

Relevant code:

- `python/rustystats/glm.py`: summary display of standard errors and p-values.
- `python/rustystats/formula.py`: result wrapper and regularization metadata.
- `crates/rustystats-core/src/inference/`: standard inference routines.
- `crates/rustystats-core/src/solvers/`: constrained and penalized solvers.

### Required Semantics

For ordinary unpenalized, unconstrained GLMs:

- model-based and robust inference can be shown normally.

For ridge, lasso, elastic-net, penalized splines, monotonic constraints, or
coefficient sign constraints:

- the result must expose `inference_status`;
- summaries must label p-values/standard errors as unavailable, approximate, or
  post-selection naive;
- ordinary significance stars must be hidden unless inference is valid for the
  fitted objective;
- AIC/BIC must be hidden or clearly labelled when the effective degrees of
  freedom are not ordinary parameter counts;
- boundary-active coefficients must be marked.

Recommended statuses:

```text
valid_standard
valid_robust
naive_after_regularization
naive_after_selection
naive_after_cv_selection
constrained_boundary
unavailable
```

CV/data-driven lambda selection is itself a selection layer, including pure
ridge. CV-selected fits must not fall through to `valid_standard`; they should
use `naive_after_cv_selection` or be folded explicitly into
`naive_after_selection`.

### Solver Status

Results should expose:

- convergence flag
- number of iterations
- final deviance
- whether step-halving was used
- whether any coefficient constraint is active
- whether the final step was accepted, rejected, or retained from the previous
  iteration
- optimizer route used, for example `irls`, `coordinate_descent`, `nnls`,
  `gcv_penalized`

### Tests

Add tests covering:

1. Unpenalized GLM summaries still show standard inference.
2. Lasso/elastic-net summaries do not present ordinary p-values as valid.
3. CV-selected models record selection metadata and inference caveats.
4. Constrained fits mark active boundary coefficients.
5. Penalized spline summaries use effective degrees of freedom where available.
6. Solver status records step-halving and convergence information.
7. Serialization preserves inference and solver-status metadata.

## Implementation Order

### Phase 1: Fail-Closed Safety

This phase can be done quickly and prevents silent misuse.

1. Add `exposure=` to the dict/formula construction path and normalize legacy
   string-offset exposure behavior.
2. Ensure array offsets are never treated as raw exposure for target encoding.
3. Add a guard that errors on CV plus target encoding until fold-safe CV is
   implemented.
4. Remove hidden relaxed CV convergence settings or make them explicit.
5. Add validation-weighted scoring to CV where possible.
6. Add tests for the guards and exposure semantics.

### Phase 2: Correct P0 Functionality

1. Implement fold-safe target encoding CV.
2. Thread exposure metadata through prediction, diagnostics, and serialization.
3. Pass `var_power` and `theta` through smooth GLM.
4. Add explicit Negative Binomial theta behavior.
5. Add full regression tests for P0/P1 items that can silently change fitted
   means or model selection.

### Phase 3: Production Diagnostics And Solver Hardening

1. Implement rate-ranked diagnostics.
2. Implement GLM-score-based alpha max.
3. Tighten Tweedie support.
4. Fix IRLS step acceptance and final clamping.
5. Correct quasi-likelihood reporting.
6. Add explicit calibration diagnostics and calibration primitives.
7. Add honest inference and solver-status metadata for constrained/penalized
   models.

## Acceptance Criteria

The work is complete when:

1. All new tests listed above pass.
2. Existing tests pass:

```bash
cargo test
uv run pytest tests/python/ -v
```

3. Documentation is updated:
   - `docs/api/dict-api.md` documents `exposure`.
   - `docs/components/target-encoding.md` explains fold-safe CV and exposure.
   - `docs/components/diagnostics.md` explains rate-ranked diagnostics.
   - `docs/api/diagnostics.md` documents calibration summaries and calibration
     objects.
   - `docs/theory/regularization.md` explains GLM-score alpha max.
   - `docs/theory/families.md` explains Tweedie support.
   - `docs/theory/families.md` explains fixed versus estimated NB theta.
   - `docs/api/results.md` explains inference caveats and solver status.
4. Serialized model compatibility tests cover old offset-as-exposure models.
5. Release notes include migration examples:

```python
# Preferred new spelling
rs.glm_dict(..., family="poisson", exposure="Exposure")

# Link-scale offset remains supported
rs.glm_dict(..., family="poisson", offset=np.log(exposure), exposure=exposure)

# Legacy spelling remains accepted for now
rs.glm_dict(..., family="poisson", offset="Exposure")
```

## Open Design Decisions

1. Should legacy `offset="Exposure"` emit a `FutureWarning`, or should it remain
   a permanent convenience alias for `exposure="Exposure"`?
2. Should strict production mode be a global option, a model argument, or a
   validation profile?
3. Should formula/dict CV always move to Python for fold-aware transforms, or
   should Rust receive fold indices and transform state in a later native path?
4. Should weights in diagnostics be treated as sampling weights, portfolio
   importance weights, or both with explicit modes?
5. Should extended Tweedie powers be supported at all in the public Python API?
6. Should `GlobalCalibration` live at `rs.GlobalCalibration`, under
   `rs.calibration`, or both?
7. Should log-link relevel be implemented as a new model object by default, or
   should `inplace=True` be allowed for advanced users?
8. Should `theta="estimate"` become the default for Negative Binomial, or should
   the API require an explicit opt-in?
9. Which inference quantities, if any, should be shown for regularized and
   constrained fits by default?
