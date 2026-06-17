# Multinomial Deferred Features - Native Support Spec

This spec covers the remaining native `multinomial_dict` features that were
explicitly deferred from the first multinomial implementation:

- multinomial target encoding,
- automatic smooth penalties and `k=` spline support,
- monotonicity constraints,
- lasso, elastic net, and cross-validation,
- PMML and ONNX export for `MultinomialModel`.

The goal is to bring multinomial choice models to the same engineering standard
as scalar `glm_dict` while preserving the statistical contracts of the native
baseline-category multinomial solver. These features are especially important
for insurance pricing teams modelling mutually exclusive product-tier outcomes.

---

## 1. Current State

The native multinomial path already supports:

- shared-covariate baseline-category logit,
- alternative-specific wide terms,
- unpenalized and ridge dense Newton fits,
- row weights and class weights,
- availability masks,
- class-specific utility offsets,
- calibration, diagnostics, scenarios, summaries, and serialization.

It currently rejects:

- `target_encoding` terms and target-encoded interactions,
- `bs`/`ns` terms with `k=` or omitted `df`/`knots`,
- `s`/`ms` smooth-term aliases,
- any `monotonicity` constraint,
- `regularization="lasso"`, `regularization="elastic_net"`, and `cv=...`,
- `to_pmml()` and `to_onnx()`.

The rejection points are mostly in `python/rustystats/multinomial.py`:

- `_validate_supported_term_spec`,
- `_validate_supported_terms`,
- `MultinomialDict.fit`,
- `MultinomialModel.to_pmml`,
- `MultinomialModel.to_onnx`,
- standalone guards in `export_pmml.py` and `export_onnx.py`.

---

## 2. Guiding Constraints

1. Reuse existing scalar infrastructure wherever it is semantically valid:
   - `InteractionBuilder`,
   - target/frequency encoding Rust helpers,
   - spline term parsing and transform state,
   - scalar regularization-path conventions,
   - PMML/ONNX exporter structure.
2. Do not reintroduce a fake scalar `Family` implementation for multinomial.
3. Preserve availability as a fit-time likelihood constraint.
4. Keep unpenalized predictions reference-invariant.
5. Document penalized, constrained, and target-encoded fits as reference-class
   dependent unless a later symmetric formulation is implemented.
6. Keep inference honest:
   - covariance is unavailable or explicitly naive for lasso/elastic net,
   - class-weighted inference remains labelled naive,
   - target encoding and CV must not leak validation labels.
7. Keep dense memory bounded:
   - no full `n x p x K` temporaries,
   - keep existing Hessian byte and parameter guards,
   - extend guards when smooth penalties or active-set solves add `q x q` state.
8. Every feature should fail closed for unsupported combinations rather than
   silently approximating a different model.

---

## 3. Dense Solver Scope and Escape Hatch

All deferred features must be designed against the current dense multinomial
solver ceiling:

```text
q = total estimated parameters
Hessian memory = q x q x 8 bytes
default max_dense_parameters = 5000
default hessian_memory_limit_bytes = 256 MiB
```

This is an explicit product decision for the next native feature set:

- the first implementation is for moderate-`q` pricing designs,
- high-cardinality target encoding should be parsimonious by default,
- lasso/elastic-net should not be advertised as a solution for 50k-level sparse
  one-hot designs while the solver still forms a dense Hessian,
- smooth `k=` support widens the dense basis and must remain under the same
  guards,
- CV multiplies fit cost and must fail early when candidate designs exceed the
  dense guard.

The future escape hatch is a sparse multinomial coordinate-descent path that
works from sparse `X` columns and probability/residual updates, glmnet-style,
without forming a dense `q x q` Hessian. That is out of scope for this spec, but
the API and docs must leave room for it. Until then, every regularized or smooth
multinomial feature should document that it is dense-solver bounded.

---

## 4. Feature Dependency Order

Recommended implementation sequence:

1. **Multinomial target encoding**
   - Needed before CV can be safely used with target-encoded rating factors.
2. **Lasso, elastic net, and CV**
   - Adds model selection and sparsity for high-cardinality pricing designs.
3. **Automatic smooth penalties and `k=` spline support**
   - Adds scalar-like smooth API with multinomial block penalties.
4. **Monotonicity constraints**
   - Builds on generalized constrained solver infrastructure.
5. **ONNX/PMML export**
   - Best done after the modelling surface and metadata are stable.

This ordering is not arbitrary: target encoding affects design construction and
CV leakage; lasso/CV affects solver metadata; smooth penalties and monotonicity
affect optimization; export depends on all feature metadata being reliable.

---

## 5. Multinomial Target Encoding

### 5.1 User-Facing API

Allow scalar-style target encoding terms:

```python
result = rs.multinomial_dict(
    response="tier",
    terms={
        "age": {"type": "linear"},
        "brand": {"type": "target_encoding", "prior_weight": 5.0},
        "region": {"type": "categorical"},
    },
    interactions=[
        {
            "brand": {"type": "categorical"},
            "region": {"type": "categorical"},
            "target_encoding": True,
            "prior_weight": 10.0,
        }
    ],
    data=train,
    classes=["none", "basic", "standard", "premium"],
    reference="none",
).fit()
```

The existing scalar keys should be supported:

- `prior_weight`,
- `n_permutations`,
- target-encoded interactions via `target_encoding: True`.

Do not add a separate multinomial-specific public term type unless the scalar
API cannot express the needed metadata.

### 5.2 Encoding Shape

A multinomial target-encoding term computes one smoothed rate per
non-reference class:

```text
TE(brand)[basic]
TE(brand)[standard]
TE(brand)[premium]
```

The default implementation should route those rates through the existing
alternative-specific class-specific tensor path:

```text
alternative_specific[row, class, TE(brand)] = TE(level(row), class)
coefficient per non-reference class = gamma_class
```

So one target-encoded factor contributes `K - 1` coefficients, not
`(K - 1)^2`. This is the diagonal/alternative-specific framing: the class `c`
utility gets the class `c` smoothed propensity for the row's level. It is the
right default for pricing because target encoding is meant to compress
high-cardinality factors, not re-expand them into a dense cross-class block.

A full shared-design block remains a possible future option, for example:

```python
{"brand": {"type": "target_encoding", "mode": "full_block"}}
```

but it should not be the default because it inflates `q` quickly and pushes the
model into the dense Hessian guard.

An even more parsimonious future option is a generic diagonal mode:

```text
utility_c += gamma * TE(level(row), c)
```

This contributes one coefficient total rather than `K - 1`. It is the closest
multinomial analogue to scalar target encoding's one-column-to-one-coefficient
shape. Keep class-specific diagonal routing as the first default because tier
pricing often benefits from letting each non-reference tier have its own
propensity sensitivity.

### 5.3 Encoding Formula

For a level `l` and non-reference class `c`, define:

```text
TE_lc =
  (sum_i w_i * 1[level_i = l] * 1[y_i = c]
   + prior_weight * pi_c)
  /
  (sum_i w_i * 1[level_i = l] * available_ic
   + prior_weight)
```

where:

- `w_i` is the row/prior weight supplied by the user,
- `available_ic` is the availability mask for class `c` on row `i`,
- `pi_c` is the weighted global class rate among rows where class `c` is
  available,
- class weights are not used in target-encoding statistics.

Rationale:

- availability should not turn unavailable alternatives into evidence of low
  demand;
- class weights are objective reweighting, not observed portfolio mix;
- row weights are legitimate exposure/credibility weights for the empirical
  class rates.

If availability is omitted, every class is available on every row.

If a level has no available rows for class `c`, use `pi_c`.

### 5.4 Implementation Path

Reuse the existing exposure-weighted target encoder once per non-reference
class instead of adding a new multiclass Rust kernel.

For class `c`, call the existing exposure-weighted ordered encoder with:

```text
claims_i   = w_i * 1[y_i = c]
exposure_i = w_i * available_ic
prior      = sum(claims_i) / sum(exposure_i)
```

This maps exactly to the formula above and inherits the already-tested
ordered/permutation prefix machinery in:

```text
crates/rustystats-core/src/target_encoding/mod.rs
crates/rustystats/src/target_encoding_py.rs
```

For target-encoded interactions, reuse the existing exposure-weighted
interaction helper:

```text
target_encode_interaction_with_exposure
```

with the same per-class `claims_i` and `exposure_i` mapping. Combined-key
multinomial TE interactions should be orchestration over the existing helper,
not a separate interaction encoder.

The implementation should be a thin Python/Rust orchestration loop over
non-reference classes plus state packaging for
`MultinomialTargetEncodingState`, not a net-new multiclass target-encoding
algorithm.

### 5.5 Training Leakage Policy

For final model fitting, use ordered/permutation target encoding analogous to
the scalar path:

1. For each permutation, maintain prefix weighted counts by level and class.
2. Encode each row from prefix counts that exclude the current row.
3. Average over `n_permutations`.
4. Fall back to global priors when a prefix has no available denominator.

Prediction uses full-training smoothed level statistics stored on the builder.
Unseen levels use global priors.

### 5.6 CV Leakage Policy

For cross-validation:

1. Build target-encoding statistics on each training fold only.
2. Transform validation rows using the fold-training statistics.
3. For training-fold rows, use ordered/permutation encoding within that fold.
4. Never use validation labels to encode validation rows.

The scalar `build_fold_design_matrices` pattern is the right model, but the
multinomial implementation needs class-vector statistics and availability-aware
denominators.

### 5.7 Implementation Notes

Add or extend:

```text
python/rustystats/interactions.py
python/rustystats/multinomial.py
python/rustystats/regularization_path.py
crates/rustystats/src/target_encoding_py.rs
crates/rustystats-core/src/target_encoding/mod.rs
```

Preferred shape:

- Introduce a `MultinomialTargetEncodingState` Python-side object or dict
  compatible with `InteractionBuilder` serialization.
- Reuse `target_encode_with_exposure` per non-reference class.
- Store:
  - class order,
  - reference class,
  - non-reference encoded classes,
  - `mode="alternative_specific_diagonal"`,
  - global priors,
  - per-level smoothed rates,
  - missing/unseen fallback policy,
  - prior weight and permutation count.
- Keep `transform_new_data()` as the prediction entry point.

Do not rewrite `InteractionBuilder`, and do not ask it to emit multinomial
target encoding as shared `X` columns. Use the clean ownership split:

- `InteractionBuilder`, or a parallel state object owned by it, fits,
  transforms, and serializes the per-class target-encoding statistics.
- The transformed values are returned to `multinomial.py` as per-row/per-class
  rate arrays.
- `multinomial.py` appends those arrays to the existing alternative-specific
  class-specific tensor path, alongside `_resolve_alternative_arrays`.
- The resulting columns use existing `_alternative_standardization`
  scale-only, per-class-block standardization, not shared
  `compute_standardization`.

### 5.8 Tests

Rust:

- per-class reuse of exposure-weighted TE matches a hand-computed small example,
- per-class reuse of `target_encode_interaction_with_exposure` matches a
  hand-computed combined-key interaction example,
- ordered encoding excludes the current row for each encoded class,
- availability-aware denominators ignore unavailable class rows,
- unseen levels use global priors.

Python:

- `multinomial_dict(... target_encoding ...)` fits and predicts,
- emitted coefficient rows show target encoding as alternative-class-specific,
- one TE term contributes `K - 1` coefficients, not `(K - 1)^2`,
- target-encoded columns are standardized through the alternative-specific
  scale-only path, not the shared-design standardization path,
- no leakage: duplicated high-cardinality labels do not get perfect training
  probabilities from their own row,
- availability changes TE values when a class is unavailable by segment,
- row weights change TE values,
- class weights do not change TE statistics but still change the fitted model,
- serialization round-trip preserves TE prediction values,
- CV with TE matches a manual fold-safe computation on a small dataset.

Acceptance:

- Target encoding no longer raises for supported terms.
- Validation labels are never used to encode validation rows.
- Prediction on unseen levels is deterministic and documented.

---

## 6. Lasso, Elastic Net, and Cross-Validation

### 6.1 User-Facing API

Support scalar-style calls:

```python
result = model.fit(regularization="lasso", alpha=0.1)

result = model.fit(
    regularization="elastic_net",
    alpha=0.1,
    l1_ratio=0.5,
)

result = model.fit(
    regularization="elastic_net",
    cv=5,
    selection="1se",
    n_alphas=50,
    alpha_min_ratio=1e-3,
    cv_seed=42,
)
```

The `MultinomialDict.fit` signature should align with scalar `glm_dict` for:

- `regularization`,
- `alpha`,
- `l1_ratio`,
- `cv`,
- `selection`,
- `n_alphas`,
- `alphas`,
- `alpha_min_ratio`,
- `cv_seed`,
- `max_iter`,
- `tol`,
- `standardize`.

If scalar names differ slightly, prefer the existing scalar names.

### 6.2 Penalty Definition

For flattened non-reference parameters `beta`, minimize:

```text
negative_log_likelihood(beta)
+ alpha * (1 - l1_ratio) / 2 * ||beta_pen||_2^2
+ alpha * l1_ratio * ||beta_pen||_1
```

Rules:

- intercept columns are never penalized,
- shared non-intercept coefficients are penalized,
- alternative generic coefficients are penalized,
- alternative class-specific coefficients are penalized,
- fixed smooth penalties, when present, are separate from elastic-net alpha,
- baseline lasso/elastic-net is reference-dependent and must be documented.

### 6.3 Solver

Keep dense Newton/ridge as-is for `l1_ratio == 0`.

For `l1_ratio > 0`, the first native implementation should be a dense
proximal-Newton solver for moderate-`q` models:

1. At current `beta`, compute objective, gradient, and Hessian using existing
   multinomial kernels.
2. Solve the quadratic plus L1 subproblem by coordinate descent on the dense
   Hessian:

   ```text
   min_d 0.5 d'Hd + g'd + alpha*l1_ratio*|beta + d|_1
   ```

3. Use active-set screening:
   - start with non-zero coefficients plus KKT violators,
   - periodically scan all penalized coefficients,
   - keep intercepts and unpenalized columns always active.
4. Line-search the true penalized objective after the proximal step.
5. Warm-start across alpha path values.

This reuses the dense Hessian already needed for Newton and avoids inventing a
separate `n x p x K` coordinate-descent probability update path.

This is deliberately **not** the final algorithm for very wide sparse rating
factor designs. The docs and errors must state that dense multinomial lasso/EN
is bounded by `max_dense_parameters` and `hessian_memory_limit_bytes`. For
50k-level sparse categoricals, the future implementation should be a sparse
glmnet-style coordinate descent over `X` columns and multinomial residuals,
without forming the dense Hessian.

### 6.4 KKT and Convergence

For penalized coefficients:

```text
beta_j != 0:  gradient_j + ridge_j + alpha*l1_ratio*sign(beta_j) ~= 0
beta_j == 0:  |gradient_j + ridge_j| <= alpha*l1_ratio + tolerance
```

For unpenalized coefficients:

```text
gradient_j ~= 0
```

Expose:

- `solver_status`,
- `warnings`,
- `regularization_type`,
- `alpha`,
- `l1_ratio`,
- `n_nonzero`,
- `regularization_path` for CV fits.

Covariance:

- direct lasso/elastic-net fits should return `covariance=None` initially,
- `coef_table()` should render `NaN` standard errors,
- `inference_status` should include `naive_after_regularization` and
  `covariance_skipped` or `covariance_unavailable`.

### 6.5 Alpha Grid

For `l1_ratio > 0`, compute `alpha_max` from the null/warm-start score:

```text
alpha_max = max_j |score_j| / l1_ratio
```

over penalized, standardized coefficients only.

The score must include:

- row weights,
- availability,
- offsets,
- alternative terms,
- class weights in the same way the fit objective sees them.

For ridge-only CV, keep scalar-compatible log-spaced alpha grids, but document
that `alpha_max` is a grid scale, not a sparsity threshold.

### 6.6 Cross-Validation

Use stratified folds over observed classes by default.

Fold construction must fail or retry if a training fold lacks any class with
positive effective weight. Do not let an empty-class fold reach the solver and
produce a confusing core error.

Validation score:

```text
weighted multinomial deviance on validation rows, excluding the penalty
```

Normalize by validation weight sum so scores are comparable across folds.

CV must use fold-local:

- target encoding,
- standardization,
- smooth lambda tuning, if smooth terms are present,
- class availability,
- offsets and weights.

Final model:

- refit on all training rows with the selected `alpha` and `l1_ratio`,
- use full-training preprocessing state,
- record CV metadata on the final `MultinomialModel`.

Dense guard policy:

- every alpha-grid candidate should be rejected before fold fitting if its
  design width/classes exceed dense guards,
- CV should report which candidate or design was rejected,
- fold-local target encoding must use the same diagonal/alternative-specific
  representation as final fitting.

### 6.7 Tests

Rust:

- L1 KKT conditions on a small dense problem,
- intercept is not penalized,
- alpha above `alpha_max` zeros all penalized coefficients,
- elastic net interpolates between ridge and lasso,
- warm-start path is equivalent to cold fits within tolerance.

Python:

- direct lasso produces sparse coefficients,
- direct elastic net fits and predicts finite probabilities,
- CV selects one alpha from the grid,
- `selection="1se"` chooses an alpha no less regularized than `selection="min"`
  according to the scalar convention,
- fold-safe target encoding works under CV,
- class weights and row weights affect CV scores,
- no covariance is exposed for lasso/elastic-net,
- reference dependence is documented and tested as not invariant.

Acceptance:

- `regularization="lasso"` and `"elastic_net"` no longer raise.
- `cv=...` no longer raises for supported regularization choices.
- CV results are reproducible with `cv_seed`.
- No validation target leakage is possible through target encoding.

---

## 7. Automatic Smooth Penalties and `k=` Splines

### 7.1 User-Facing API

Support scalar-style smooth spline specifications:

```python
terms = {
    "driver_age": {"type": "bs"},       # smooth, default k=10
    "vehicle_age": {"type": "bs", "k": 15},
    "tenure": {"type": "ns", "k": 8},
}
```

Rules:

- `df=` or explicit `knots=` remains a fixed unpenalized basis,
- omitted `df`/`knots` means penalized smooth with default `k=10`,
- `k=` means penalized smooth with that basis dimension,
- `s` and `ms` aliases should match scalar behavior only if their scalar
  semantics can be preserved exactly.

This spec does not require automatic search over `k`. It requires support for
the scalar meaning of `k`: user-selected maximum basis size plus automatic
penalty tuning.

### 7.2 Penalty Definition

For each smooth term `t` with basis columns `C_t` and penalty matrix `S_t`,
apply the same smoothness penalty independently to every non-reference class
block:

```text
0.5 * lambda_t * sum_k beta_kt' S_t beta_kt
```

There is no cross-class smoothness penalty in the first implementation.

The same `lambda_t` is used across non-reference classes for a term. Per-class
smooth lambdas are out of scope for the first smooth implementation because they
increase tuning dimensionality and make diagnostics harder to explain.

### 7.3 Lambda Tuning

Use multinomial GCV/AIC-style effective degrees of freedom:

```text
edf = trace(H_unpenalized * inv(H_unpenalized + P))
```

where:

- `H_unpenalized` is the multinomial Hessian at the penalized optimum,
- `P` is the block smooth penalty matrix,
- ridge/elastic-net penalties are not counted as smooth edf unless explicitly
  used for smoothing.

Initial tuning:

- single smooth term: re-implement a bounded one-dimensional search over
  `lambda` around full multinomial fits,
- multiple smooth terms: re-implement a coordinate search pattern around full
  multinomial fits.

Only the smooth primitives are reusable:

```text
crates/rustystats-core/src/splines/penalized.rs
```

Specifically reuse:

- `penalty_matrix`,
- block penalty assembly helpers where applicable,
- `compute_edf` / trace helpers,
- GCV/AIC/BIC score helpers if their normalization matches multinomial weights.

Do not call `fit_smooth_glm_full_matrix` from the multinomial implementation;
it is an IRLS-on-one-design scalar GLM entry point. For multinomial, each
candidate `lambda` requires a native multinomial fit with smooth penalties added
to the coupled multinomial Hessian.

Weighted GCV should use effective observation weight:

```text
gcv = deviance / (weight_sum - edf)^2 * weight_sum
```

Guard `weight_sum - edf` away from zero.

### 7.4 Solver Integration

Extend `MultinomialConfig` or add a separate smooth config with:

- smooth column ranges in shared `X`,
- per-term penalty matrices,
- tuned lambdas or fixed lambdas,
- lambda bounds,
- lambda method,
- edf output.

The existing Hessian assembly should add the smooth penalty matrix to the
corresponding class-block diagonals. Do not expand smooth penalties through
Python loops during every Newton iteration.

The EDF formula should use the full coupled multinomial Hessian, not a
per-class decoupled approximation. The smooth penalty matrix is block diagonal
by class, but the likelihood Hessian has off-diagonal class blocks through the
softmax.

### 7.5 Metadata

Expose on `MultinomialModel`:

- `smooth_terms`,
- `smooth_lambdas`,
- `smooth_edfs`,
- `total_edf`,
- `gcv`,
- term-level smooth diagnostics.

For AIC/BIC:

- unpenalized/fixed-basis models keep `n_params`,
- smooth models should use `total_edf` for diagnostics where scalar smooth
  models already do so,
- document any difference between parameter count and edf.

### 7.6 Tests

Rust:

- smooth penalty matrix is inserted in every non-reference class block,
- edf decreases as lambda increases,
- lambda tuning chooses finite positive lambdas,
- availability does not break Hessian/edf computation.

Python:

- `bs` with omitted `df` fits as a smooth term,
- `bs(k=15)` changes basis width and metadata,
- fixed `df` behavior is unchanged,
- smooth predictions are finite and probabilities sum to one,
- `smooth_lambdas` and `smooth_edfs` serialize,
- smooth diagnostics and summary render.

Acceptance:

- `k=` and default smooth spline terms no longer raise.
- Smooth metadata is available and stable after serialization.
- Dense memory guard accounts for smooth basis expansion and covariance/edf
  workspace.

---

## 8. Monotonicity Constraints

### 8.1 Statistical Contract

For multinomial models, `monotonicity` means monotonic utility/log-odds relative
to the reference class, not guaranteed monotonic class probabilities.

Example:

```python
terms = {
    "income": {"type": "linear", "monotonicity": "increasing"},
}
```

This means every non-reference class logit has a non-negative coefficient for
`income` relative to the reference class:

```text
eta_class - eta_reference is non-decreasing in income
```

Softmax coupling means the probability for a given non-reference class may still
move non-monotonically if other class utilities move faster. Documentation must
state this clearly.

### 8.2 Supported Terms

Initial support:

- `linear` terms,
- `expression` terms,
- fixed-df `bs` terms.

Reject:

- `ns` monotonicity, matching scalar behavior,
- smooth monotone `bs` until the mechanism is explicitly implemented,
- target-encoded monotonicity,
- categorical monotonicity,
- monotonic interactions unless they already have a scalar-equivalent contract.

Alternative-specific monotonicity is out of scope for the first pass. It should
be specified separately because class-specific price/richness terms have a more
natural per-alternative utility constraint than shared terms.

### 8.3 Constraint Mapping

For a constrained design column `j`:

- increasing: `beta_kj >= 0` for every non-reference class block `k`,
- decreasing: `beta_kj <= 0` for every non-reference class block `k`.

For monotone B-splines:

- fixed-df monotone `bs` may reuse scalar monotone basis/constraint
  construction,
- apply the generated sign constraints to every non-reference class block,
- preserve fitted knots and basis metadata for prediction.

Smooth monotone `bs` is a separate design fork. The scalar smooth-monotone path
uses exp-reparameterization rather than ordinary sign constraints, while
fixed-df monotone splines use sign/bound constraints. For multinomial, smooth
monotone support must choose one of:

- extend the scalar exp-reparameterization per class block and rederive the
  multinomial gradient/Hessian through the per-class Jacobian,
- introduce a SCOP-style monotone basis whose coefficients can use the same
  sign-constraint machinery as fixed-df splines.

Until that choice is made and tested, smooth + monotone should fail closed with
a clear error.

### 8.4 Solver Integration

Extend the multinomial solver with bound constraints:

```rust
pub nonneg_indices: Vec<usize>,
pub nonpos_indices: Vec<usize>,
```

where indices refer to the flattened multinomial parameter vector.

For ridge/Newton/smooth:

- use an active-set projected Newton method,
- solve free coefficients only when constraints are active,
- project trial steps before objective evaluation,
- check KKT conditions for active bounds.

This projected-Newton path applies to linear, expression, and fixed-df monotone
`bs` constraints. It does not by itself implement scalar-style smooth monotone
exp-reparameterization.

For lasso/elastic-net:

- combine soft-thresholding with projection,
- enforce bounds after each coordinate update,
- use KKT checks that include both L1 subgradient and bound constraints.

### 8.5 Tests

Rust:

- constrained optimum satisfies bounds,
- KKT conditions hold for active constraints,
- constraints that fight the data converge to boundary solutions,
- constraints combine with ridge and lasso.

Python:

- linear increasing/decreasing constraints produce coefficient signs in all
  non-reference blocks,
- fixed-df monotone `bs` produces monotone utility curves per non-reference
  class,
- smooth monotone `bs` fails closed until a mechanism is implemented,
- docs/test make clear probabilities are not guaranteed monotone,
- serialization preserves constraint metadata,
- CV folds receive the same constraints,
- unsupported monotonicity combinations fail with precise messages.

Acceptance:

- `monotonicity` no longer raises for supported shared terms.
- The solver returns constrained KKT-sane fits, not just clipped final
  coefficients.
- Documentation states the utility-level contract.

---

## 9. PMML and ONNX Export

### 9.1 Export Levels

Implement export in levels rather than trying to support every surface at once.

Level 1:

- shared-covariate models,
- no alternative-specific terms,
- no availability mask,
- no prediction-time offset,
- no calibration object baked in,
- terms already supported by scalar exporters where possible.

Level 1 is intentionally demo-grade for multinomial tier-conversion pricing,
because real models usually need alternative-specific price/richness terms.
Land it only if it keeps the graph/XML small and gives useful exporter
infrastructure.

Level 2:

- alternative-specific terms,
- availability columns,
- offsets,
- optional calibration intercept shifts,
- target encoding lookup state.

Level 2 is the first practically useful export level for insurance
tier-conversion models.

Level 3:

- smooth and monotone terms with piecewise-linear approximations matching scalar
  export behavior,
- target-encoded interactions,
- richer export metadata and parity tests.

Each level should fail closed for unsupported model state and list the offending
feature in the error message.

### 9.2 ONNX Contract

Scoring-mode ONNX should accept a pre-built design matrix first:

```text
input:  X_shared_without_intercept  shape (n, p_no_intercept)
output: probabilities              shape (n, K)
```

Graph:

1. MatMul shared design by coefficient matrix.
2. Add class intercept vector.
3. Insert reference-class zero logit in the correct class position.
4. Add optional calibration shifts if explicitly requested.
5. Softmax over classes.

Metadata:

- class labels,
- reference class,
- feature names,
- coefficient convention,
- rustystats version,
- unsupported feature flags.

Full-mode ONNX can then reuse scalar preprocessing graph pieces and extend them
to class-matrix logits.

### 9.3 PMML Contract

PMML should use a classification model with softmax normalization:

- one regression table or equivalent score expression per class,
- explicit reference-class table with zero intercept/coefs,
- `targetFieldName` set to the response,
- class labels in RustyStats class order.

PMML `RegressionModel` supports multinomial classification with
`normalizationMethod="softmax"` and one `RegressionTable` per category. The
current scalar exporter emits `GeneralRegressionModel`; multinomial export
should add a dedicated `RegressionModel` + softmax emitter rather than trying to
force the existing GRM path into a multinomial shape.

For scalar-compatible raw terms, reuse existing PMML feature classification and
derived-field construction where possible. For unsupported terms, raise
`ValidationError`.

### 9.4 Availability, Offsets, and Calibration

Availability:

- ONNX Level 2 may mask unavailable class logits with a large negative constant
  before Softmax.
- PMML support is optional unless a clean standards-compliant expression is
  available; otherwise fail closed.

Offsets:

- ONNX Level 2 may accept an offset matrix input and add it to logits.
- PMML should fail closed unless offsets are represented by named fields with a
  reliable expression.

Calibration:

- Do not silently bake calibration into export.
- Add explicit arguments:

```python
model.to_onnx(calibration=calibration)
model.to_pmml(calibration=calibration)
```

- Validate class order and reference before applying shifts.

### 9.5 Tests

ONNX:

- generated protobuf parses,
- metadata contains class/reference/feature names,
- scoring-mode logits/probabilities match RustyStats on numeric matrices,
- optional onnxruntime parity test if dependency is installed,
- unsupported model state fails clearly.

PMML:

- XML parses,
- target categories appear in class order,
- coefficients match `coef_table`,
- a lightweight PMML scorer test matches RustyStats for linear/categorical
  supported models,
- unsupported model state fails clearly.

Acceptance:

- `MultinomialModel.to_onnx()` and `.to_pmml()` no longer raise for supported
  models.
- Standalone `rs.to_onnx(model)` and `rs.to_pmml(model)` dispatch correctly.
- Unsupported combinations fail with precise, actionable messages.

---

## 10. Cross-Feature Interactions

### 10.1 Target Encoding + CV

This is mandatory support, not optional. Insurance pricing teams will naturally
cross-validate target-encoded high-cardinality factors.

Acceptance requires a manual K-fold parity test that recomputes fold-local
multiclass target encoding outside the public API and matches the reported CV
score within tolerance.

### 10.2 Target Encoding + Export

Target encoding export depends on serialized lookup state. Export should fail
closed until:

- full level stats,
- global priors,
- missing/unseen behavior,
- class order,
- reference class,
- alternative-specific diagonal TE mode

are all present in model state.

### 10.3 Smooth + Lasso/Elastic Net

Initial rule:

- smooth penalties and elastic-net penalties may coexist,
- smooth basis coefficients receive both smooth penalty and elastic-net penalty
  unless explicitly excluded by scalar precedent,
- edf calculations report smooth edf separately from L1 sparsity metadata.

If this proves statistically confusing, fail closed for `regularization` with
smooth terms in the first smooth PR and enable in a follow-up. Do not silently
drop either penalty.

### 10.4 Smooth + Monotonicity

Smooth + monotone is not automatically covered by either the smooth penalty
implementation or the fixed-df bound-constraint implementation. The first
monotonicity PR should fail closed for smooth monotone `bs` unless it also
chooses and implements one of the mechanisms in Section 8.3.

### 10.5 Monotonicity + Reference Class

Because constraints are relative to the reference class, changing the reference
changes the constrained parameterization. This is expected and must be
documented.

Tests should assert:

- constrained fits satisfy bounds under the chosen reference,
- releveling constrained models is unsupported unless the model is refit.

---

## 11. Serialization and Model State

Extend `MultinomialModel.to_bytes()` state with versioned fields:

- target-encoding state,
- smooth term metadata,
- smooth lambdas and edfs,
- constraint indices and user-facing monotonicity specs,
- regularization path metadata,
- selected alpha/l1 ratio/CV scores,
- export compatibility metadata if needed.

Rules:

- Increment schema version when adding non-optional state.
- Load older multinomial models with default empty metadata.
- Round-trip predictions must match before/after serialization for every new
  supported feature.

---

## 12. Documentation

Update:

```text
README.md
docs/api/dict-api.md
docs/api/serialization.md
docs/maintenance/performance.md
examples/tier_conversion_multinomial.py
CHANGELOG.md
```

Docs must include:

- target encoding shape and leakage policy,
- dense-solver scope for lasso/elastic-net and high-cardinality factors,
- utility-level monotonicity contract,
- smooth monotone fail-closed limitation until implemented,
- reference dependence of penalized/constrained fits,
- covariance/inference limitations for lasso/elastic net,
- CV scoring definition,
- smooth edf vs parameter count,
- export supported/unsupported feature matrix.

The example should eventually include:

- target-encoded high-cardinality factor,
- CV-selected elastic net,
- one smooth term,
- one monotone utility constraint,
- export only if the example model fits the supported export level.

---

## 13. Performance Requirements

Benchmarks should extend `benchmarks/bench_multinomial.py` with optional cases:

- high-cardinality target encoding,
- lasso/elastic-net alpha paths,
- CV with and without target encoding,
- smooth terms with `k=10` and `k=20`,
- constrained monotone fits,
- export generation time and artifact size.

Performance gates:

- target encoding should reuse the Rust exposure-weighted encoder per
  non-reference class,
- target-encoded interactions should reuse
  `target_encode_interaction_with_exposure` per non-reference class,
- CV should avoid rebuilding fold-invariant state unnecessarily,
- lasso path should warm-start,
- lasso/elastic-net should reject large dense `q` designs before fitting and
  document the future sparse-CD escape hatch,
- smooth lambda search should cap candidate fits,
- export should not materialize large prediction grids except spline
  approximation grids,
- existing quick benchmark should remain fast enough for local smoke runs.

---

## 14. Suggested PR Sequence

### PR A - Multinomial Target Encoding Core

Deliver:

- per-class reuse of exposure-weighted TE helpers,
- per-class reuse of exposure-weighted TE interaction helpers,
- availability-aware weighted statistics,
- ordered/permutation training encoding,
- alternative-specific diagonal TE tensor routing,
- explicit builder-state versus multinomial-tensor ownership split,
- prediction transform state,
- serialization,
- direct and public API tests.

### PR B - Fold-Safe Multinomial TE

Deliver:

- fold-local design construction,
- manual K-fold parity tests,
- CV-ready preprocessing state,
- no-leakage tests.

### PR C - Lasso/Elastic-Net Direct Fits

Deliver:

- dense proximal Newton solver for moderate-`q` designs,
- explicit dense-guard documentation and tests,
- KKT tests,
- standardization/back-transform,
- coefficient table behavior,
- inference-status updates.

### PR D - Multinomial CV

Deliver:

- alpha grid generation,
- stratified folds,
- validation deviance scoring,
- min/1se selection,
- path metadata,
- final refit.

### PR E - Smooth `k=` Support

Deliver:

- smooth metadata pass-through from `InteractionBuilder`,
- block smooth penalties,
- multinomial-specific lambda tuning loop,
- reuse of penalty/EDF primitives,
- edf/GCV metadata,
- serialization and docs.

### PR F - Monotonicity Constraints

Deliver:

- flattened bound-index mapping,
- constrained Newton/proximal solver support,
- fixed-df monotone spline support,
- smooth-monotone fail-closed behavior unless fully implemented,
- KKT tests,
- utility-contract docs.

### PR G - Export Level 1

Deliver:

- multinomial ONNX scoring mode,
- PMML `RegressionModel` softmax shared-covariate classification export,
- standalone dispatch,
- explicit fail-closed unsupported-feature checks.

### PR H - Export Level 2/3

Deliver:

- availability/offset/calibration handling where representable,
- alternative terms,
- target encoding lookups,
- smooth/monotone approximations,
- export parity tests.

---

## 15. Final Acceptance Checklist

The deferred feature set is complete when:

- `target_encoding` terms and interactions fit, predict, serialize, and pass
  fold-safe CV tests.
- `regularization="lasso"` and `"elastic_net"` fit directly and through CV.
- CV is stratified, reproducible, weight-aware, availability-aware, and scored
  by unpenalized validation deviance.
- `bs`/`ns` smooth terms with omitted `df`/`knots` and explicit `k=` work.
- Smooth lambdas, edf, and GCV metadata are exposed and serialized.
- Supported fixed-df monotonicity constraints enforce utility-level KKT
  conditions.
- Smooth monotone terms either fail closed or have their chosen mechanism fully
  tested.
- Probability-level monotonicity is not implied in docs.
- PMML/ONNX export works for documented supported multinomial models.
- Unsupported export combinations fail closed with actionable errors.
- Existing scalar `glm_dict` behavior and tests do not regress.
- Full Python and Rust test suites pass.
- Benchmark smoke runs stay bounded and predictable.

---

## 16. Resolved Decisions and Open Questions

Resolved in this spec:

1. Target encoding is probability-scale for the first implementation, matching
   scalar house convention. A log-odds encoding can be added later only as an
   explicit option.
2. Target encoding statistics do not incorporate class weights. Class weights
   affect fitting only.
3. Target encoding defaults to diagonal/alternative-specific routing, not a full
   shared `(K - 1) x (K - 1)` coefficient block. A one-coefficient generic
   diagonal mode can be considered later, but is not the first default.
4. Smooth lambdas are shared across class blocks per term.
5. Lasso/elastic-net uses a dense proximal-Newton implementation first and is
   explicitly moderate-`q`; the sparse-CD path is future work.
6. PMML multinomial export needs a `RegressionModel` softmax emitter, not the
   scalar `GeneralRegressionModel` emitter.
7. ONNX scoring mode should land before raw-data full mode.
8. `k=` means scalar-compatible basis-size support plus automatic lambda tuning,
   not automatic search over `k`.

Open questions for future specs:

1. Should monotonicity eventually support alternative-specific terms such as
   tier-specific price/richness?
2. Which mechanism should smooth monotone multinomial `bs` use:
   exp-reparameterization per class block or SCOP-style sign-constrained bases?
3. When should the sparse multinomial coordinate-descent solver be prioritized,
   and should it support only L1/EN or also very wide ridge?
4. Should full-block target encoding and/or one-coefficient generic diagonal
   target encoding be exposed as explicit expert modes?
