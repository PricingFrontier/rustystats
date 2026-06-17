# Multinomial Deferred Features - Implementation Plan

This plan implements `specs/multinomial-deferred-features-spec.md` to a
production standard. The target user is an insurance pricing team building
native multinomial tier-conversion models with high-cardinality rating factors,
smooth effects, regularization, constraints, and exportable scoring artifacts.

The work is intentionally staged. Each phase should leave the repository in a
reviewable, testable state and should preserve the scalar `glm_dict` path.

---

## 1. Engineering Principles

### 1.1 Non-Negotiables

- Reuse existing scalar and Rust infrastructure where it is semantically valid.
- Do not implement multinomial as a scalar `Family` variant.
- Do not rebuild target encoding, spline, standardization, PMML, or ONNX
  machinery when the existing implementation can be factored or orchestrated.
- Preserve availability as a fit-time likelihood and derivative constraint.
- Keep all unsupported combinations fail-closed with actionable
  `ValidationError` messages.
- Keep inference honest:
  - lasso and elastic net expose no covariance initially,
  - class-weighted inference remains labelled naive,
  - penalized and constrained fits are reference-class dependent.
- Bound memory before allocation. Dense multinomial features must respect:
  - `max_dense_parameters = 5000` by default,
  - `hessian_memory_limit_bytes = 256 MiB` by default,
  - covariance and EDF paths may need 2x to 3x `q * q * 8` peak memory.
- Do not parallelize outer class-pair Hessian loops over already-parallel
  `compute_xtwx_with_sparse_cache` calls.
- Every phase must add tests that would fail for the most likely wrong
  implementation, not only smoke tests.

### 1.2 Current Main Entry Points

Primary Python files:

```text
python/rustystats/multinomial.py
python/rustystats/interactions.py
python/rustystats/regularization_path.py
python/rustystats/formula.py
python/rustystats/export_pmml.py
python/rustystats/export_onnx.py
python/rustystats/_rustystats.pyi
```

Primary Rust files:

```text
crates/rustystats-core/src/solvers/multinomial.rs
crates/rustystats-core/src/solvers/irls.rs
crates/rustystats-core/src/target_encoding/mod.rs
crates/rustystats-core/src/splines/penalized.rs
crates/rustystats/src/multinomial_py.rs
crates/rustystats/src/target_encoding_py.rs
```

Primary tests:

```text
tests/python/test_multinomial.py
tests/python/test_export.py
tests/python/test_regularization_standardization.py
crates/rustystats-core/src/solvers/multinomial.rs
```

Add focused new test modules when a file becomes too large, for example:

```text
tests/python/test_multinomial_target_encoding.py
tests/python/test_multinomial_regularization.py
tests/python/test_multinomial_smooth.py
tests/python/test_multinomial_export.py
```

---

## 2. Stage 0 - Baseline and Branch Hygiene

Purpose: make later failures attributable to the implementation, not local
drift.

Tasks:

1. Record the current git status and do not overwrite unrelated user changes.
2. Run a focused baseline:

   ```bash
   cargo test -p rustystats-core solvers::multinomial
   uv run pytest tests/python/test_multinomial.py
   ```

3. If time permits, run:

   ```bash
   cargo test -p rustystats-core
   uv run pytest tests/python
   ```

4. Document pre-existing failures in the PR notes before modifying code.

Acceptance:

- Baseline failures, if any, are known.
- No feature work is mixed into baseline cleanup.

---

## 3. Stage 1 - Shared Multinomial Infrastructure

Purpose: add internal contracts that every deferred feature can share. This
stage should not expose new user-facing functionality.

### 3.1 Parameter Layout

The current multinomial solver already flattens parameters internally. Before
adding lasso, smooth penalties, monotonicity, and export, centralize that
mapping so future code does not duplicate index arithmetic.

Add an internal Rust layout helper in
`crates/rustystats-core/src/solvers/multinomial.rs` or a sibling module:

```rust
struct MultinomialParameterLayout {
    n_shared: usize,
    n_non_reference: usize,
    n_alt_generic: usize,
    n_alt_specific: usize,
    fit_intercept: bool,
}
```

Required methods:

- `len() -> usize`
- `shared(block, feature) -> usize`
- `alternative_generic(term) -> usize`
- `alternative_specific(block, term) -> usize`
- `shared_block_range(block) -> Range<usize>`
- `is_shared_intercept(idx) -> bool`
- `validate_index(idx)`

Rules:

- Intercept columns are recognized only in shared `X`.
- Intercept identity is pure flattening metadata and is Rust-owned through
  `is_shared_intercept(idx)`.
- Penalty eligibility masks are Python-owned design metadata, derived from
  `InteractionBuilder` output, and passed to Rust explicitly through
  `L1Penalty.mask` or future penalty metadata. Rust must not infer penalty
  masks from names or term semantics.
- Alternative generic coefficients have one coefficient total per term.
- Alternative class-specific coefficients have one coefficient per
  non-reference class and term.
- Reference-class coefficients are never estimated.

Pin the ownership direction:

- Python owns design semantics because `InteractionBuilder` creates feature
  names, term slots, spline column ranges, and transformed alternative tensors.
- Python derives concrete shared-column metadata from the builder:
  - penalized shared-column mask,
  - smooth term column ranges,
  - monotonicity column ranges and signs,
  - target-encoding tensor positions,
  - alternative generic/specific term positions.
- Python passes concrete masks/ranges/index sets to Rust.
- Rust owns only flattening arithmetic over `(p, K, alternative counts)` and
  maps Python-provided column metadata into flattened parameter indices.

Do not let Rust infer term semantics from names or duplicate
`InteractionBuilder` ordering. The first time Python reorders or expands a term,
Rust-side semantic inference will drift.

Tests:

- Layout length equals `n_params`.
- Every `coef_table()` row maps to exactly one flattened index.
- Shared intercept positions are identified by `is_shared_intercept(idx)` and
  Python-derived penalty masks exclude them.
- Alternative generic and class-specific blocks map to the same order used by
  Rust coefficient arrays.
- Python-derived masks/ranges map to the expected flattened Rust indices.

### 3.2 Dense Guard Helper

Create one preflight helper used by direct fit, CV, smooth EDF, and covariance:

```text
validate_dense_multinomial_size(q, *, operation, compute_covariance, compute_edf)
```

It should report:

- number of classes,
- shared columns,
- alternative generic terms,
- alternative specific terms,
- total `q`,
- single-Hessian memory,
- estimated peak memory for the requested operation,
- configured byte and parameter limits.

Use the helper before:

- direct Rust fit,
- each CV fold/candidate design,
- smooth EDF computation,
- covariance inversion,
- export paths that materialize dense matrices.

Do not loosen the default guard for lasso/elastic-net. These are dense-solver
features until a sparse coordinate-descent path exists.

### 3.3 Optimization Extension Points

Define one composable Newton/proximal step rather than forking the multinomial
iteration for each feature.

Add internal structs or equivalent configuration objects for:

```rust
struct QuadraticPenaltyBlock { indices: Vec<usize>, matrix: Array2<f64>, weight: f64 }
struct L1Penalty { alpha: f64, mask: Vec<bool> }
struct BoundConstraints { nonneg_indices: Vec<usize>, nonpos_indices: Vec<usize> }
```

The Stage 4 implementation should land the optimization path with all three
extension points present, even if smooth penalties and bounds are initially
empty:

- smooth/ridge-like quadratic penalties contribute objective, gradient, and
  Hessian terms;
- L1 contributes the proximal subproblem and KKT subgradient checks;
- bounds contribute projection and bound KKT checks;
- inert extension objects must reproduce the current unpenalized/ridge Newton
  behavior within tolerance.

Stages 6 and 7 should extend these extension objects. They should not fork a
second or third Newton loop.

Tests:

- Inert extension objects exactly match the existing Newton path.
- Quadratic-only penalties match current ridge behavior.
- L1-only and bound-only cases can be activated independently in small solver
  tests.

### 3.4 Model State Versioning

Increment `_SCHEMA_VERSION` only when a phase adds non-optional serialized
state. Older multinomial models must continue loading by defaulting new fields
to empty metadata.

Reserve versioned state keys for:

```text
target_encoding_state
regularization_path_info
smooth_metadata
constraint_metadata
export_metadata
```

Acceptance:

- Existing multinomial tests pass unchanged.
- Existing `MultinomialModel.from_bytes()` round trips still work.
- New layout tests pass.

---

## 4. Stage 2 - Multinomial Target Encoding, Direct Fits

Purpose: support target-encoded terms and target-encoded interactions for
ordinary direct fits, without CV yet.

### 4.1 Public Contract

Enable existing scalar syntax:

```python
terms={"brand": {"type": "target_encoding", "prior_weight": 5.0}}
interactions=[{"brand": {"type": "categorical"}, "region": {"type": "categorical"}, "target_encoding": True}]
```

Do not add a multinomial-specific public term unless the scalar term cannot
carry required metadata.

Default encoding shape:

```text
alternative_specific[row, class, term] = TE(level(row), class)
```

One target-encoded factor contributes `K - 1` class-specific coefficients, not
`(K - 1)^2`.

### 4.2 Ownership Split

This is the most important design point for this phase.

Use this ownership split:

- `InteractionBuilder` or a parallel state object owns TE term parsing,
  transform state, and serialization.
- Existing Rust TE helpers compute the actual per-class encodings.
- `multinomial.py` assembles the resulting arrays into the
  alternative-specific class-specific tensor.
- `_alternative_standardization()` standardizes these tensor columns for
  penalized fits.

Do not let `InteractionBuilder.build_design_matrix_from_parsed()` emit
multinomial TE columns into shared `X`.

### 4.3 Implementation Tasks

Python:

1. Add an internal `MultinomialTargetEncodingState` dataclass or dict wrapper.
   Store:
   - class order,
   - reference class,
   - non-reference encoded classes,
   - mode `"alternative_specific_diagonal"`,
   - per-term metadata,
   - global priors,
   - per-level smoothed stats,
   - missing/unseen fallback policy,
   - prior weights and permutation counts.
2. Add a helper that extracts target-encoding terms from the parsed formula
   without adding them to shared `X`.
3. Build a shared parsed formula with TE terms removed for the shared design
   matrix. Preserve non-TE terms and non-TE interactions exactly.
4. For each TE term and non-reference class `c`, call existing exposure-weighted
   helpers with:

   ```text
   claims_i   = row_weight_i * 1[y_i = c]
   exposure_i = row_weight_i * availability_i,c
   prior      = sum(claims_i) / sum(exposure_i)
   ```

5. For single-column TE, reuse `target_encode_with_exposure_py`.
6. For TE interactions, reuse `target_encode_interaction_with_exposure_py`.
7. Fill an `n x K x n_te_terms` tensor:
   - fill non-reference class slices with class-specific TE values,
   - leave reference class values harmlessly zero or unused,
   - append this tensor to existing `alternative_specific`.
8. Append stable feature names such as:

   ```text
   TE(brand)
   TE(brand:region)
   ```

9. Update prediction:
   - `MultinomialModel._prediction_alternative_arrays()` should transform new
     rows through stored TE state,
   - unseen levels use stored priors,
   - prediction must not require response labels.
10. Update serialization:
    - include TE state,
    - restore it in `from_bytes`,
    - preserve predictions after round trip.

Rust:

- Prefer no new core algorithm.
- Add only thin PyO3/state helpers if the existing exposed functions do not
  return enough state.
- Keep target encoding tests near the existing target encoding modules when
  they verify Rust helper behavior.

### 4.4 Validation and Fail-Closed Rules

Reject:

- `mode="full_block"` and generic diagonal mode until explicitly implemented,
- target-encoded monotonicity,
- target-encoded export until export phase supports lookup state,
- invalid weights, non-finite TE inputs, and rows where the observed class is
  unavailable.

### 4.5 Tests

Rust:

- `target_encode_with_exposure` per-class mapping matches a hand calculation.
- `target_encode_interaction_with_exposure` per-class mapping matches a hand
  calculation.
- Ordered encoding excludes the current row.
- Unseen levels use class priors.

Python:

- `multinomial_dict(... target_encoding ...)` fits and predicts.
- One TE term adds `K - 1` parameters.
- TE rows in `coef_table()` are alternative-class-specific.
- Availability changes TE denominators.
- Row weights change TE statistics.
- Class weights do not change TE statistics, but do change the fit objective.
- Penalized TE uses `_alternative_standardization()`, not shared
  `compute_standardization()`.
- Serialization round trip preserves `predict_proba`.

Acceptance:

- Direct fits with TE no longer raise.
- No validation label can enter prediction-time TE transforms.
- Scalar TE behavior is unchanged.

---

## 5. Stage 3 - Fold-Safe Target Encoding Preprocessing

Purpose: build the CV preprocessing foundation before enabling CV itself.

### 5.1 Fold Builder

Add a multinomial equivalent of scalar `build_fold_design_matrices`, but return
all fit inputs:

```text
FoldDesign:
  x_train
  x_val
  alternative_generic_train
  alternative_generic_val
  alternative_specific_train
  alternative_specific_val
  availability_train
  availability_val
  offset_train
  offset_val
  weights_train
  weights_val
  y_train
  y_val
  feature_names
  preprocessing_state
```

Rules:

- Fit TE state on fold-training rows only.
- Transform validation rows using fold-training TE stats.
- For fold-training rows, use ordered/permutation TE inside the fold.
- Reset fold-local mutable spline state, as scalar CV already does.
- Let per-fold categorical column counts differ if the fold-local builder does;
  each fold scores scalar validation deviance in its own column space.

### 5.2 Reuse Points

Reuse:

- scalar fold-safe TE design pattern in `regularization_path.py`,
- `InteractionBuilder.transform_new_data()`,
- `TargetEncodingTermSpec`,
- Rust target encoding helpers.

Do not:

- slice a full-data TE design for validation,
- reuse full-data TE states inside fold validation,
- cache fold designs across different fold seeds or class availability specs.

### 5.3 Tests

Python:

- Manual two-fold computation matches helper outputs.
- Validation labels do not affect validation encodings.
- Fold-local unseen levels fall back to fold-training priors.
- Fold-local availability changes class-specific TE values.
- Cached fold designs are deterministic for the same data, parsed formula, and
  seed.

Acceptance:

- Fold preprocessing exists and is tested before CV calls it.
- Direct fit behavior from Stage 2 is unchanged.

---

## 6. Stage 4 - Dense Lasso and Elastic Net, Direct Fits

Purpose: support direct non-ridge regularized multinomial fits for moderate
`q`, still without CV.

### 6.1 Solver Contract

Keep existing dense Newton/ridge behavior for `l1_ratio == 0`.

For `l1_ratio > 0`, implement dense proximal Newton:

1. Evaluate the smooth likelihood objective, gradient, and Hessian.
2. Add L2 part of elastic net to objective, gradient, and Hessian:

   ```text
   l2_alpha = alpha * (1 - l1_ratio)
   l1_alpha = alpha * l1_ratio
   ```

3. Do not add L1 to the gradient or Hessian.
4. Solve the quadratic plus L1 subproblem by coordinate descent on the dense
   Hessian.
5. Use active-set screening:
   - nonzero coefficients,
   - KKT violators,
   - all unpenalized coefficients.
6. Line-search the true penalized objective.
7. Stop on KKT conditions, not only step size.

This implementation must use the composable optimization extension points from
Stage 1:

- quadratic penalties active for elastic-net L2 and later smooth penalties,
- L1 prox active for lasso/elastic net,
- bound constraints present but inert until Stage 7.

Do not add a lasso-specific Newton loop that smooth penalties and monotonicity
will need to fork later.

### 6.2 Rust Implementation Tasks

In `crates/rustystats-core/src/solvers/multinomial.rs`:

1. Remove the current validation error for `l1_ratio != 0`.
2. Split ridge-only `alpha` usage into:
   - smooth differentiable L2 alpha,
   - non-smooth L1 alpha.
3. Add a flattened penalty mask by mapping Python-provided column masks through
   `MultinomialParameterLayout`:
   - intercepts false,
   - shared non-intercepts true,
   - alternative generic true,
   - alternative class-specific true.
4. Reuse the existing standardization plumbing:
   - shared `compute_standardization`/`solver_standardization`,
   - `_alternative_standardization()` for alternative tensors,
   - Rust standardization/back-transform helpers.
5. Apply L1 thresholds in standardized coefficient space. This is required for
   scale-invariant support selection. Back-transform coefficients before
   exposing `MultinomialModel` results.
6. Add helpers:

   ```rust
   fn penalized_objective(...)
   fn l1_norm(theta, penalty_mask)
   fn solve_proximal_newton_subproblem(...)
   fn check_l1_kkt(...)
   ```

7. Keep class-pair Hessian block assembly sequential over class pairs, because
   `compute_xtwx_with_sparse_cache` is already internally parallel.
8. Reuse nalgebra solves where full linear solves are still needed.
9. Add warm-start support internally if it does not complicate the direct API;
   otherwise land warm-start in the path/CV phase.

Python/PyO3:

1. Lift the Python direct-fit guards in `MultinomialDict.fit`:
   - the regularization guard that currently allows only `None`/`"ridge"`,
   - the `l1_ratio != 0.0` guard.
2. Keep the `cv is not None` guard in place until Stage 5.
3. Lift the Rust `l1_ratio != 0.0` validation guard in
   `validate_and_prepare`.
4. Normalize:

   ```text
   ridge       -> l1_ratio = 0.0
   lasso       -> l1_ratio = 1.0
   elastic_net -> supplied l1_ratio, default matching scalar convention
   ```

5. Keep covariance unavailable for L1 fits.
6. Set inference notes:
   - `naive_after_regularization`,
   - `naive_after_selection`,
   - `covariance_unavailable` or `covariance_skipped`.
7. Ensure `coef_table()` renders `NaN` standard errors gracefully.

### 6.3 Alpha Max

Add a multinomial alpha-max helper for L1 paths:

```text
alpha_max = max(abs(score_j)) / l1_ratio
```

over penalized standardized parameters only.

The score must include:

- shared terms,
- alternative generic terms,
- alternative class-specific terms,
- row weights,
- class weights as used by the fit objective,
- availability,
- offsets.

Use the helper in tests now, and in CV/path generation later.

### 6.4 Tests

Rust:

- L1 KKT conditions pass on a small dense problem.
- Intercepts are not penalized.
- A sufficiently large alpha zeros all penalized coefficients.
- Elastic net approaches ridge as `l1_ratio -> 0`.
- Elastic net approaches lasso as `l1_ratio -> 1`.
- Dense guard fires before allocating Hessian state.
- Rescaling one penalized feature leaves the selected support invariant after
  standardization/back-transform.

Python:

- Direct lasso produces at least one exact zero coefficient in a constructed
  problem.
- Direct elastic net fits and predicts finite probabilities.
- Rescaling a shared feature and an alternative-specific feature preserves the
  selected support under lasso/elastic net.
- `predict_proba` rows sum to one.
- `coef_table()` has `NaN` standard errors for L1 fits.
- Reference dependence is documented by a test that does not expect invariance.
- Existing ridge and unpenalized tests still pass.

Acceptance:

- `regularization="lasso"` and `"elastic_net"` direct fits no longer raise.
- Large dense designs fail early with useful messages.
- No scalar regularization behavior changes.

---

## 7. Stage 5 - Multinomial Regularization Paths and CV

Purpose: add alpha paths and fold-based model selection for ridge, lasso, and
elastic net.

### 7.1 Public API

Align `MultinomialDict.fit` with scalar names:

```python
fit(
    regularization="elastic_net",
    cv=5,
    selection="min" | "1se",
    n_alphas=...,
    alphas=...,
    alpha_min_ratio=...,
    cv_seed=...,
)
```

If scalar names differ, prefer the scalar names for consistency.

Lift the remaining `MultinomialDict.fit` `cv is not None` rejection guard in
this stage. The regularization and `l1_ratio` direct-fit guards should already
have been lifted in Stage 4.

### 7.2 Fold Construction

Use stratified folds over observed classes by default.

Rules:

- Every fold-training split must contain all observed classes with positive
  effective weight.
- Retry deterministic seeded fold construction a bounded number of times, then
  fail with a clear error.
- Preserve row weights and class weights in training and scoring.
- Do not let a bad fold reach the Rust empty-class validation.

### 7.3 Path Fitting

Create a multinomial-specific path helper rather than forcing the scalar
`fit_cv_regularization_path` to understand multinomial tensors.

Reusable scalar pieces:

- `generate_alpha_path`,
- `select_optimal_alpha`,
- `RegularizationPathInfo` shape if it is general enough,
- standardization helpers where dimensions match,
- fast non-TE path pattern from `fit_cv_regularization_path`,
- fold-safe TE path pattern from `fit_cv_te_regularization_path`.

New multinomial pieces:

- alpha-max over the flattened multinomial score,
- dense guard per fold/candidate,
- fold-local alternative arrays,
- fold-local target encoding,
- validation multinomial deviance scorer,
- final refit with full-data preprocessing.

Validation score:

```text
weighted validation multinomial deviance, excluding penalty
```

Normalize by validation weight sum.

### 7.4 Warm Starts

Within each fold:

- fit alphas from largest to smallest,
- pass the previous solution as the starting theta,
- keep preprocessing fixed for that fold across alphas,
- reset warm starts between folds.

For final refit:

- use the selected alpha,
- initialize from a nearby full-data path solution if available,
- otherwise use current intercept/null initialization.

Implementation note: the path still requires `n_alphas * cv` dense candidate
fits, but candidates warm-start within each fold and the final full-data refit
warms from the selected point on a full-data path. The dense preflight remains
the hard memory guard; warm starts reduce Newton iterations rather than the
number of candidate fits.

### 7.5 Metadata

Expose on `MultinomialModel`:

- selected alpha,
- selected `l1_ratio`,
- selection method,
- candidate alphas,
- mean validation deviances,
- standard errors,
- fold count,
- fold-safe target-encoding flag,
- failed candidate/fold diagnostics.

Keep metadata naming close to scalar `RegularizationPathInfo`.

### 7.6 Tests

Python:

- CV selects an alpha from the grid.
- `selection="1se"` chooses an alpha at least as regularized as `"min"`.
- CV is reproducible with `cv_seed`.
- Weighted validation deviance matches a manual small example.
- Fold-safe target encoding under CV matches manual fold encodings.
- Empty-class folds are retried or fail before Rust fit.
- `cv` with unsupported feature combinations fails closed.

Rust:

- Warm-start path equals cold fits within tolerance on a small problem.
- Alpha path KKT conditions hold per candidate.

Acceptance:

- `cv=...` works for ridge, lasso, and elastic net.
- TE + CV is fold-safe.
- CV cost is bounded by dense preflight checks and documented.

---

## 8. Stage 6 - Smooth `k=` Splines and Automatic Smooth Penalties

Purpose: support scalar-style smooth spline terms in multinomial models.

### 8.1 Supported Surface

Enable:

```python
terms={
    "driver_age": {"type": "bs"},
    "vehicle_age": {"type": "bs", "k": 15},
    "tenure": {"type": "ns", "k": 8},
}
```

Rules:

- `df=` and explicit `knots=` remain fixed unpenalized bases.
- Omitted `df`/`knots` means penalized smooth with default `k`.
- `k=` means basis size plus automatic lambda tuning, not search over `k`.
- `s` and `ms` aliases should be enabled only when they exactly match scalar
  semantics.

### 8.2 Reuse Points

Reuse:

- spline parsing and `SplineTerm` transform state,
- `crates/rustystats-core/src/splines/penalized.rs`,
- `penalty_matrix`,
- EDF/trace helpers if their normalization matches,
- GCV/AIC/BIC helper formulas where appropriate.

Do not call scalar `fit_smooth_glm_full_matrix`; multinomial lambda candidates
must call the native multinomial solver.

Note that the existing scalar `gcv_score(deviance, n, edf)` takes an integer
row count. Weighted multinomial GCV uses `weight_sum: f64`, so add a weighted
overload or compute the weighted formula inline rather than passing
`weight_sum` through the scalar signature.

### 8.3 Solver Changes

Add differentiable quadratic penalty support separate from elastic-net ridge:

```text
0.5 * sum_t lambda_t * beta_t' S_t beta_t
```

For each smooth term:

- use the shared `X` column range derived by Python from `InteractionBuilder`,
- apply the same `S_t` to every non-reference class block,
- keep no cross-class smooth penalty initially,
- add penalty contributions to objective, gradient, and Hessian.

Implementation guidance:

- Store smooth penalties as block descriptors, not as a full dense `q x q`
  matrix unless a helper explicitly needs dense form.
- During Hessian assembly, add each smooth penalty to the relevant diagonal
  class block after likelihood Hessian construction.
- Keep all smooth penalty loops outside row loops.

### 8.4 Lambda Tuning

Implement a multinomial-native tuning loop:

- one smooth term: bounded one-dimensional search,
- multiple smooth terms: coordinate search with capped iterations,
- candidate fits use warm starts where possible,
- each candidate uses dense preflight guards.

EDF:

```text
edf = trace(H_unpenalized * inv(H_unpenalized + P))
```

where `H_unpenalized` is the full coupled multinomial Hessian at the penalized
optimum. Do not approximate EDF as a sum of independent per-class scalar EDFs.

GCV:

```text
gcv = deviance / (weight_sum - edf)^2 * weight_sum
```

Guard `weight_sum - edf` away from zero.

Covariance and standard errors:

- If smooth/ridge-style penalized fits expose inverse-Hessian covariance, label
  the resulting standard errors as penalized-fit or naive, consistent with the
  existing ridge inference policy.
- Do not present penalized smooth standard errors as ordinary unpenalized MLE
  standard errors.
- L1 fits continue to expose no covariance initially.

### 8.5 Metadata and Serialization

Expose:

- `smooth_terms`,
- `smooth_lambdas`,
- `smooth_edfs`,
- `total_edf`,
- `gcv`,
- term-level smooth diagnostics.

Serialize:

- fitted knots,
- basis metadata,
- penalty matrices or enough state to rebuild them,
- lambdas,
- edfs,
- tuning method and convergence status.

### 8.6 Tests

Rust:

- Smooth penalty is inserted in every non-reference block.
- EDF decreases as lambda increases.
- Availability does not break Hessian/EDF.
- Penalty gradient/Hessian match finite differences on a small problem.

Python:

- `bs` with omitted `df` fits as a smooth term.
- `bs(k=15)` changes basis width and metadata.
- Fixed `df` behavior is unchanged.
- Smooth predictions are finite and sum to one.
- Smooth metadata survives serialization.
- Smooth summary/diagnostics render.
- Dense guard accounts for widened basis and EDF workspace.

Acceptance:

- `k=` and default smooth spline terms no longer raise.
- Smooth fits are native multinomial fits, not scalar IRLS calls.
- Existing fixed-basis spline behavior does not regress.

---

## 9. Stage 7 - Monotonicity Constraints

Purpose: support utility-level monotonicity for supported shared terms.

### 9.1 Supported Surface

Enable monotonicity for:

- linear terms,
- expression terms,
- fixed-df `bs` terms.

Reject:

- `ns` monotonicity,
- smooth monotone `bs`,
- target-encoded monotonicity,
- categorical monotonicity,
- monotonic interactions unless scalar-equivalent behavior is already clear,
- alternative-specific monotonicity in the first pass.

### 9.2 Constraint Mapping

For shared column `j`:

```text
increasing: beta_block,j >= 0 for every non-reference block
decreasing: beta_block,j <= 0 for every non-reference block
```

For fixed-df monotone `bs`:

- reuse scalar monotone basis/constraint construction if available,
- map generated sign constraints into every non-reference class block,
- preserve knots and basis metadata for prediction.

Python should derive constrained shared-column ranges from the builder and pass
the resulting flattened `nonneg_indices`/`nonpos_indices` to Rust. Rust should
not infer monotonicity from feature names.

Do not claim probability monotonicity. The contract is utility/logit
monotonicity relative to the reference class.

### 9.3 Solver Changes

Extend the flattened solver with bound constraints:

```rust
nonneg_indices: Vec<usize>
nonpos_indices: Vec<usize>
```

For Newton/ridge/smooth:

- implement active-set projected Newton,
- solve over free coefficients,
- project trial steps before evaluating the objective,
- check bound KKT conditions.

For lasso/elastic net:

- combine coordinate soft-thresholding with bound projection,
- enforce bounds after each coordinate update,
- include both L1 subgradient and bound multipliers in KKT checks.

Do not implement smooth monotone exp-reparameterization in this stage unless
the phase explicitly takes on the full gradient/Hessian derivation. Smooth
monotone terms should fail closed.

### 9.4 Tests

Rust:

- Bound-constrained optimum satisfies bounds.
- KKT conditions hold for active and inactive constraints.
- Data that fights a constraint converges to a boundary solution.
- Constraints combine with ridge and lasso.

Python:

- Linear increasing/decreasing signs hold in all non-reference blocks.
- Fixed-df monotone `bs` utility curves are monotone per non-reference class.
- Smooth monotone `bs` fails closed with a precise error.
- Serialization preserves constraint metadata.
- CV folds receive the same constraints.
- Docs/tests show probabilities need not be monotone.

Acceptance:

- Supported monotonicity no longer raises.
- The solver is constrained, not post-fit clipped.
- Reference dependence is documented.

---

## 10. Stage 8 - PMML and ONNX Export

Purpose: add fail-closed export for supported multinomial model states.

### 10.1 Export Level 1

Implement first:

- shared-covariate multinomial models,
- no alternative-specific terms,
- no availability mask,
- no prediction-time offset,
- no calibration baked in by default,
- terms already supported by scalar export infrastructure where possible.

This is useful infrastructure but not the final pricing export surface.

### 10.2 ONNX Scoring Mode

Start with scoring mode:

```text
input: pre-built shared design matrix without intercept
output: probabilities, shape (n, K)
```

Graph:

1. MatMul design by coefficient matrix.
2. Add class intercept vector.
3. Insert zero reference-class logit in the correct position.
4. Optionally add explicit calibration shifts.
5. Softmax over classes.

Reuse scalar ONNX serializer pieces where possible, especially metadata and
protobuf assembly. Add Rust graph builder support only for operations not
already exposed.

### 10.3 PMML RegressionModel Softmax

Add a dedicated PMML multinomial path. Do not force the scalar
`GeneralRegressionModel` emitter into multinomial shape.

Emit:

- PMML `RegressionModel`,
- `normalizationMethod="softmax"`,
- one `RegressionTable` per class,
- explicit zero table for the reference class,
- class labels in RustyStats class order.

Reuse scalar PMML feature classification and derived-field creation where
possible.

### 10.4 Export Levels 2 and 3

After Level 1:

Level 2:

- alternative-specific terms,
- availability masks where representable,
- offset matrix input or named offset fields,
- explicit calibration argument,
- target encoding lookup state.

Level 3:

- smooth and monotone approximations,
- target-encoded interactions,
- richer parity tests.

Each unsupported state must identify the first unsupported feature and suggest
a supported export level or scoring-mode alternative.

### 10.5 Tests

ONNX:

- generated bytes parse.
- metadata contains class labels, reference, feature names, and convention.
- scoring-mode probabilities match RustyStats on numeric matrices.
- optional `onnxruntime` parity if installed.
- unsupported model states fail clearly.

PMML:

- XML parses.
- `RegressionModel` and `normalizationMethod="softmax"` are present.
- classes appear in model order.
- coefficients match `coef_table()`.
- unsupported model states fail clearly.

Acceptance:

- `MultinomialModel.to_onnx()` and `.to_pmml()` work for documented supported
  states.
- Standalone `rs.to_onnx(model)` and `rs.to_pmml(model)` dispatch correctly.
- Scalar export tests continue to pass.

---

## 11. Stage 9 - Cross-Feature Integration

Purpose: verify feature combinations that pricing users will naturally try.

Required matrix:

```text
TE + CV
TE + ridge
TE + lasso/elastic_net
smooth + ridge
smooth + lasso/elastic_net, or fail closed with explicit rationale
monotonicity + ridge
monotonicity + lasso/elastic_net
monotonicity + CV
export + each supported metadata subset
```

Rules:

- Do not silently drop one penalty when two penalties are present.
- Do not silently export a simplified model.
- Do not report standard errors for selected L1 models.
- Do not claim reference invariance for penalized or constrained models.

Tests should include at least one realistic small tier-conversion dataset with:

- 4 classes,
- a high-cardinality TE factor,
- an alternative-specific price/richness term,
- a smooth age/tenure term,
- a monotone shared score,
- row weights,
- availability that removes one tier for a segment.

---

## 12. Stage 10 - Documentation, Examples, and Benchmarks

Update:

```text
README.md
docs/api/dict-api.md
docs/api/serialization.md
docs/maintenance/performance.md
examples/tier_conversion_multinomial.py
CHANGELOG.md
```

Docs must cover:

- target encoding shape and leakage policy,
- dense solver limits for lasso/elastic-net and smooth features,
- future sparse coordinate-descent escape hatch,
- utility-level monotonicity contract,
- smooth monotone fail-closed limitation,
- reference dependence of penalized/constrained fits,
- covariance limitations for L1 fits,
- CV scoring definition,
- smooth EDF versus parameter count,
- export supported-feature matrix.

Benchmark additions:

```text
benchmarks/bench_multinomial.py
```

Add optional benchmark cases:

- high-cardinality TE direct fit,
- TE + CV,
- lasso/elastic-net path,
- smooth `k=10` and `k=20`,
- monotone constrained fit,
- export generation time.

Benchmarks should assert bounded runtime or at least print stable metrics; they
should not become required long-running CI jobs unless explicitly configured.

---

## 13. Review and Quality Gates

Each phase must pass:

```bash
uv run --quiet ruff check python/rustystats tests/python
uv run --quiet ruff format --check python/rustystats tests/python
cargo fmt --check
cargo clippy -p rustystats-core -p rustystats --all-targets -- -D warnings
```

At minimum for feature PRs:

```bash
cargo test -p rustystats-core solvers::multinomial
uv run pytest tests/python/test_multinomial.py
```

Before declaring the whole feature set complete:

```bash
cargo test -p rustystats-core
cargo test -p rustystats
uv run pytest tests/python
```

Add targeted numerical tests before broad presentation tests:

- finite-difference gradient/Hessian checks,
- KKT checks,
- no-leakage checks,
- standardization/back-transform checks,
- serialization parity checks,
- export scoring parity checks.

---

## 14. Handoff Checklist for Future Agents

Before starting a phase:

1. Read this plan and `specs/multinomial-deferred-features-spec.md`.
2. Inspect current `git status --short`.
3. Find the current rejection point for the feature.
4. Identify the existing scalar/Rust helper to reuse.
5. Add or update the most specific failing test first.
6. Keep the implementation behind fail-closed validation until the full phase
   acceptance is met.

Before finishing a phase:

1. Run the targeted tests.
2. Run format/lint checks for touched languages.
3. Confirm scalar GLM behavior is unchanged.
4. Confirm serialization either round trips or older model loading is preserved.
5. Confirm docs and error messages match the implemented support surface.
6. Leave unsupported follow-up modes explicitly rejected, not partially wired.

---

## 15. Final Acceptance

The deferred feature set is complete when:

- Multinomial target encoding and TE interactions fit, predict, serialize, and
  pass fold-safe CV tests.
- Lasso and elastic net work for direct fits and CV.
- CV is reproducible, stratified, weighted, availability-aware, and scored by
  unpenalized validation deviance.
- Smooth `k=` terms fit through native multinomial smooth penalties.
- Smooth lambdas, EDF, and diagnostics serialize and render.
- Supported monotonicity constraints satisfy utility-level KKT conditions.
- Smooth monotone terms fail closed unless a full mechanism is implemented.
- PMML and ONNX export work for documented supported multinomial states.
- Unsupported export/model combinations fail closed with actionable errors.
- Full Python and Rust suites pass.
- Benchmark smoke runs remain bounded.
