# Native Multinomial Choice - Implementation Plan

This plan implements `specs/native-multinomial-choice.md` to a production
standard suitable for insurance pricing workflows. The goal is not just to make
the API work; it is to add a native multinomial path that is statistically
honest, performant for realistic tier-conversion models, and consistent with
RustyStats' existing Rust/Python architecture.

The plan is intentionally staged. Each stage should leave the repository in a
testable, reviewable state and should avoid destabilizing the scalar GLM path.

---

## 1. Guiding Constraints

### 1.1 Non-negotiables

- Do not implement multinomial as a scalar `Family` variant.
- Do not regress `glm_dict`, scalar regularization, splines, diagnostics, or
  serialization.
- Reuse `InteractionBuilder` for shared-covariate design construction instead
  of cloning design-matrix logic.
- Treat availability as a fit-time likelihood constraint, not a prediction-time
  mask.
- Guarantee reference-class invariance only for unpenalized fits.
- Document baseline ridge as reference-dependent.
- Fail loudly for unsupported scalar-only features.
- Add correctness tests before broad API polish.
- Add performance guards before exposing the dense Newton path.

### 1.2 Initial supported surface

Phase 1 supports:

- shared-covariate baseline-category multinomial logit,
- unpenalized dense Newton,
- ridge dense Newton,
- row weights,
- class weights,
- class availability,
- class-specific utility offsets,
- fixed shared-covariate design terms:
  - linear,
  - categorical,
  - expression,
  - fixed-degree `bs`,
  - fixed-degree `ns`,
  - frequency encoding,
  - standard interactions,
- `predict_proba`,
- `predict_log_proba`,
- `decision_function`,
- `predict`,
- `predict_top_k`,
- `coef_table`,
- `summary`,
- serialization,
- minimal multinomial diagnostics.

Phase 1 rejects:

- target encoding,
- automatic smooth penalties,
- lasso,
- elastic net,
- CV,
- monotonic constraints,
- complement / lasso credibility,
- exposure,
- PMML/ONNX export,
- alternative-specific terms.

---

## 2. Work Breakdown

### Stage 0 - Branch Prep and Baseline Checks

Purpose: establish that any later failures are caused by the multinomial work,
not pre-existing local drift.

Tasks:

1. Run the existing scalar test suite, or at least:
   - `cargo test -p rustystats-core`
   - `uv run pytest tests/python`
2. Record any pre-existing failures in the implementation PR notes.
3. Confirm generated/untracked files are unrelated before editing.
4. Add no code in this stage unless needed to make the baseline runnable.

Acceptance:

- Existing failures, if any, are documented.
- No scalar behavior has changed.

---

## 3. Stage 1 - Core Data Contracts

Purpose: add the type and validation foundation before writing the optimizer.

### 3.1 Rust module scaffold

Add:

```text
crates/rustystats-core/src/solvers/multinomial.rs
```

Export from:

```text
crates/rustystats-core/src/solvers/mod.rs
```

Define:

```rust
pub struct MultinomialConfig {
    pub max_iterations: usize,
    pub tolerance: f64,
    pub alpha: f64,
    pub l1_ratio: f64,
    pub fit_intercept: bool,
    pub standardize: bool,
    pub skip_covariance: bool,
    pub hessian_memory_limit_bytes: usize,
    pub max_dense_parameters: usize,
    pub verbose: bool,
}

pub struct MultinomialResult {
    pub coefficients: Array2<f64>,
    pub fitted_probabilities: Array2<f64>,
    pub linear_predictor: Array2<f64>,
    pub log_likelihood: f64,
    pub deviance: f64,
    pub null_deviance: f64,
    pub iterations: usize,
    pub converged: bool,
    pub covariance_unscaled: Option<Array2<f64>>,
    pub prior_weights: Array1<f64>,
    pub y_codes: Array1<usize>,
    pub reference_index: usize,
    pub warnings: Vec<String>,
    pub solver_status: String,
}
```

### 3.2 Validation helpers

Implement validation for:

- `n_classes >= 2`,
- `reference_index < n_classes`,
- `y_codes.len() == X.nrows()`,
- every `y_code < n_classes`,
- every supplied class has at least one observed row,
- `availability` shape is `n x K` if provided,
- observed class is available for every row,
- each row has at least one available class,
- `offset` shape is `n x K` if provided,
- `weights.len() == n` if provided,
- weights are finite and non-negative,
- total effective weight is positive,
- `l1_ratio == 0` for Phase 1 ridge support, except `alpha == 0`,
- dense Hessian guard passes.

Use RustyStats' existing error style: return structured `RustyStatsError` from
core and convert to clear `PyValueError` at the binding.

Python-layer validation in `python/rustystats/multinomial.py` should raise
`rustystats.ValidationError`, matching `glm_dict`; only the PyO3 binding maps
core errors to `PyValueError`.

### 3.3 Dense parameter guard

Compute:

```text
q = p * (K - 1)
hessian_bytes = q * q * 8
```

Reject if:

- `hessian_bytes > hessian_memory_limit_bytes`,
- `q > max_dense_parameters`.

Defaults:

- `hessian_memory_limit_bytes = 256 * 1024 * 1024`,
- `max_dense_parameters = 5000`.

These defaults are intentionally aligned: a 256 MB single-Hessian byte guard
would bind near `q = 5792`, so the parameter guard is set below that at `5000`
to provide a clearer compute-cost ceiling.

Error message must report:

- `p`,
- `K`,
- `q`,
- estimated Hessian MB,
- configured limit,
- suggested mitigations.

### 3.4 Tests

Rust tests:

- invalid class code fails,
- empty class fails,
- observed unavailable class fails,
- all-unavailable row fails,
- bad offset/availability/weight shapes fail,
- memory guard fails before allocation.

Acceptance:

- Core scaffolding compiles.
- Validation tests pass.
- No optimizer logic is needed yet.

---

## 4. Stage 2 - Softmax, Likelihood, Gradient, Hessian

Purpose: build numerically stable math primitives and prove them before adding
Newton iteration.

### 4.1 Masked softmax

Implement row-wise stable masked softmax:

- include only available classes in the denominator,
- set unavailable probabilities to `0.0`,
- handle reference class consistently with offsets,
- subtract row maximum over available logits,
- never return NaN/Inf for finite inputs.

Final returned probabilities should be exact zeros for unavailable classes.
For log-likelihood, clamp only the selected probability used in `log`.

### 4.2 Objective

Implement:

```text
ll = sum_i weight_i * log(p_i,y_i)
deviance = -2 * ll
ridge_penalty = 0.5 * alpha * sum(beta_j^2)
objective = -ll + ridge_penalty
```

Do not penalize intercept columns when `fit_intercept=True`.
`InteractionBuilder` places the intercept at design-column index 0 when present;
for multinomial coefficient blocks this means local feature index 0 in every
non-reference class block.

### 4.3 Gradient and Hessian

Use class-major flattened order:

```text
theta = [shared_beta_blocks..., future_alternative_gamma_block...]
```

For Phase 1, the gamma block is empty.

For every non-reference class `k`:

```text
grad[k] = X.T @ (weights * (p[:, k] - 1[y == k]))
```

For Hessian block `(k, l)`:

```text
H[k, l] = X.T @ diag(weights * p[:, k] * (1[k == l] - p[:, l])) @ X
```

Here `p` is the masked probability matrix. Unavailable classes contribute zero.

### 4.4 Efficient accumulation

Avoid building per-class diagonal matrices and avoid hand-rolling a new dense
weighted-Gram kernel for the Hessian. The scalar IRLS path already has the
kernel the multinomial Hessian needs:

- `compute_xtwx(x, w)` for `X.T @ diag(w) @ X`,
- `build_sparse_row_cache_if_beneficial(x)`,
- `SparseRowCache` and the sparse-aware normal-equation path used by scalar
  IRLS.

For every non-reference class-pair `(k, l)`, build a temporary weight vector:

```text
w_kl[i] = weights[i] * p[i, k] * (1[k == l] - p[i, l])
```

Then compute:

```text
H[k, l] = compute_xtwx(X, w_kl)
```

If the existing public `compute_xtwx` path is not enough to reuse the row cache,
add a small cached helper analogous to the existing cached `X'WX/X'Wz` helper
instead of rebuilding sparse row structure for every class-pair. The row cache
must be built once outside the Newton loop and reused.

Only compute upper-triangle class-pairs and mirror the block into the lower
triangle. For `K=4`, this is six weighted-Gram calls per Newton iteration:
three diagonal blocks and three off-diagonal blocks.

Performance rules:

- Build the sparse row cache once before the Newton loop when beneficial; it is
  weight-independent and can be reused for every class-pair and every iteration.
- Reuse the existing sparse-aware Gram code rather than duplicating parallel
  reduction logic.
- Iterate class-pairs sequentially. Each weighted-Gram call is already
  row-parallel through Rayon; parallelizing over class-pairs as well would
  oversubscribe the pool.
- Compute the gradient with a plain `X.T @ r_k` matrix-vector product, where
  `r_k = weights * (p[:, k] - 1[y == k])`. The row weight is already inside
  `r_k`; no WLS `X'Wz` helper is needed for correctness.
- Keep class-pair weight vectors and Hessian block buffers reusable across
  iterations.
- Do not allocate an `n x K` probability matrix during every iteration unless
  benchmarks show it is faster and memory-safe for small `K`.
- Compute final `fitted_probabilities` and `linear_predictor` only after
  convergence.

The single-pass row-accumulation design may have attractive cache behavior for
dense narrow designs, but Phase 1 should favor the existing sparse-aware kernel
because sparse categorical pricing designs are the target workload.

### 4.5 Tests

Rust tests:

- probabilities sum to one on all-available rows,
- unavailable probabilities are zero,
- probability rows sum to one over available classes,
- objective equals hand-computed tiny example,
- gradient matches finite differences,
- Hessian matches finite differences,
- Hessian is symmetric,
- availability changes gradient/Hessian versus all-available fit.

Acceptance:

- Math primitives pass finite-difference tests with tight tolerances.
- No Newton solver required yet.

---

## 5. Stage 3 - Dense Newton Solver

Purpose: fit unpenalized multinomial models robustly.

### 5.1 Initialization

Initial coefficient matrix:

- zeros for all non-reference classes,
- optionally initialize intercept columns from weighted class shares when
  all classes are available,
- for varying availability, zeros are acceptable; the null model fit handles
  intercept-only likelihood.

### 5.2 Newton step

Per iteration:

1. Evaluate objective, gradient, Hessian.
2. Solve `H step = grad`.
3. Propose `theta_new = theta - step`.
4. Apply step-halving until objective improves.
5. Track best accepted state.
6. Stop on:
   - relative objective improvement below tolerance,
   - max absolute gradient below tolerance-scaled threshold,
   - step norm below tolerance,
   - max iterations.

Use `nalgebra` for dense solves, consistent with scalar IRLS. Prefer Cholesky
when Hessian is positive definite; fallback to LU/QR only with clear warnings.

### 5.3 Separation and divergence

Detect:

- coefficient norm exceeding a configured threshold,
- repeated step-halving failures,
- probabilities saturating near 0/1 for many rows,
- singular or near-singular Hessian,
- max iterations with objective still improving.

Warnings/status:

- `converged`,
- `max_iterations`,
- `step_halving_no_improvement`,
- `singular_hessian`,
- `possible_separation`,
- `dense_guard_exceeded`.

Suggest ridge for instability/separation.

### 5.4 Null deviance

Implement null deviance with the same solver on an intercept-only design:

- closed-form weighted class shares only when every class is available for
  every row,
- otherwise fit intercept-only model with availability.

Use the same weights, class weights, offsets, and availability contract as the
full model. If offsets are present, include them in the null model.

### 5.5 Covariance

For unpenalized converged fits:

- covariance is inverse Hessian at optimum,
- shape is `q x q`,
- flattened order matches `coef_table`.

If `skip_covariance=True`, do not invert Hessian for covariance after the
coefficient solve.

Covariance inversion is also `O(q^3)` and needs extra `O(q^2)` memory. During
standard-error computation the Hessian, inverse, and nalgebra factorization
workspace can coexist, so peak memory may be roughly 2-3x the single Hessian
size. With the Phase 1 `max_dense_parameters=5000` guard, computing standard
errors can still dominate total runtime after the coefficient fit is done. The
Python API defaults to `compute_covariance=True` for consistency, but
implementation may auto-skip with a warning above a conservative `q` threshold,
or require callers to set `compute_covariance=False` for very wide dense fits.

### 5.6 Tests

Rust tests:

- intercept-only all-available model recovers weighted class shares,
- intercept-only varying-availability model differs from global class shares,
- objective decreases monotonically after accepted steps,
- solver recovers known coefficients on simulated data,
- unpenalized all-available results match `statsmodels.MNLogit` on predicted
  probabilities, log-likelihood, and coefficients up to reference convention,
- reference change preserves unpenalized probabilities,
- separation case warns or fails with ridge suggestion,
- covariance shape and diagonal sanity.

Acceptance:

- Unpenalized dense solver is correct and stable on small and medium synthetic
  datasets.

---

## 6. Stage 4 - Ridge and Standardization

Purpose: add the stabilization path needed for practical pricing data.

### 6.1 Ridge objective

Add ridge penalty to objective, gradient, and Hessian:

```text
objective += 0.5 * alpha * sum(beta_penalized^2)
grad[j] += alpha * beta[j]
H[j, j] += alpha
```

Do not penalize intercept columns.
The intercept exclusion is local to each class block: if `fit_intercept=True`,
feature index 0 is unpenalized in every non-reference class block.

### 6.2 Reference-dependence contract

Document and test:

- baseline ridge is reference-dependent,
- unpenalized fit is reference-invariant,
- future symmetric/sum-to-zero penalty is required for reference-invariant
  regularization.

### 6.3 Standardization

For `alpha > 0` and `standardize=True`:

- compute weighted center/scale in Python or Rust consistently with scalar GLM,
- use the same center/scale vector for each non-reference class block,
- compute scale-only standardization metadata for alternative-generic and
  alternative-class-specific tensors before calling Rust,
- for alternative-generic terms, compute one weighted scale per generic term
  across available row-class cells,
- for alternative-class-specific terms, compute one weighted scale per
  non-reference class/term block using rows where that class is available,
- do not standardize intercept columns; local feature index 0 is the intercept
  in every class block when `fit_intercept=True`,
- use scale-only if no intercept exists,
- back-transform coefficients and covariance before exposing results.

Coefficient back-transform can reuse the scalar p-vector logic per class block.
Covariance back-transform cannot call the scalar helper once on the flattened
`q x q` covariance because the scalar helper assumes a single intercept at
global column 0. Multinomial has one local intercept per non-reference class
block at global indices `0, p, 2p, ...`.

Use the block transform:

```text
theta_orig = (I_{K-1} kron T) theta_std
cov_orig   = (I_{K-1} kron T) cov_std (I_{K-1} kron T).T
```

where `T` is the scalar `p x p` standardization back-transform for one class
block. Implement this either as an explicit block-pair operation or by building
the Kronecker transform when `q` is small enough. Prefer block-pair operations
to avoid a second large dense matrix allocation.

Alternative-term coefficients use the same sparse affine transform as shared
coefficients. The high-level Python API passes zero centers and weighted scales
for alternative tensors, preserving sparse zeros while giving ridge a
scale-comparable coordinate system. The Rust core also accepts explicit
alternative centers/scales through the binding for direct callers.

### 6.4 Tests

Rust/Python tests:

- ridge shrinks coefficients,
- ridge improves convergence in a separation-like example,
- standardized ridge predictions are invariant to rescaling a feature,
- standardized ridge predictions are invariant to rescaling a generic
  alternative term,
- ridge works with both generic and class-specific alternative terms,
- baseline ridge changes when reference changes,
- covariance back-transform has correct dimensions and sanity.
- standardized covariance back-transform matches an equivalent externally
  standardized reference fit to tolerance; this test must catch incorrect
  one-shot use of the scalar covariance helper on the flattened matrix.

Acceptance:

- Ridge works, is honest about reference dependence, and does not break
  unpenalized behavior.

---

## 7. Stage 5 - PyO3 Binding

Purpose: expose the core solver safely to Python.

### 7.1 Add binding module

Add:

```text
crates/rustystats/src/multinomial_py.rs
```

Expose:

```rust
fit_multinomial_py(...)
PyMultinomialResults
```

Binding arguments:

- `y_codes: PyReadonlyArray1<i64>`,
- `x: PyReadonlyArray2<f64>`,
- `n_classes`,
- `reference_index`,
- `availability: Option<PyReadonlyArray2<bool>>`,
- `offset: Option<PyReadonlyArray2<f64>>`,
- `weights: Option<PyReadonlyArray1<f64>>`,
- `alpha`,
- `l1_ratio`,
- `max_iter`,
- `tol`,
- `fit_intercept`,
- `center`,
- `scale`,
- `skip_covariance`,
- `hessian_memory_limit_bytes`,
- `max_dense_parameters`,
- `store_design_matrix`.

Register in:

```text
crates/rustystats/src/lib.rs
```

Update:

```text
python/rustystats/_rustystats.pyi
```

### 7.2 Result getters

Expose:

- `params`,
- `fitted_probabilities`,
- `linear_predictor`,
- `cov_params_unscaled`,
- `log_likelihood`,
- `deviance`,
- `null_deviance`,
- `iterations`,
- `converged`,
- `solver_status`,
- `warnings`,
- `y_codes`,
- `prior_weights`.

### 7.3 Tests

Python smoke tests directly against `_rustystats.fit_multinomial_py`:

- small all-available fit,
- availability fit,
- bad input errors,
- result getter shapes,
- `skip_covariance=True` returns no covariance.

Acceptance:

- PyO3 boundary is thin and does not contain model orchestration logic.

---

## 8. Stage 6 - Python API and Design Reuse

Purpose: add the public `multinomial_dict` API in a way that fits the existing
package.

### 8.1 New module

Add:

```text
python/rustystats/multinomial.py
```

Export from:

```text
python/rustystats/__init__.py
```

Public constructor:

```python
def multinomial_dict(
    response: str,
    data: pl.DataFrame | pl.LazyFrame,
    *,
    terms: dict[str, dict[str, Any]] | None = None,
    shared_terms: dict[str, dict[str, Any]] | None = None,
    interactions: list[dict[str, Any]] | None = None,
    intercept: bool = True,
    classes: list[str] | None = None,
    reference: str | None = None,
    availability: dict[str, str | bool | np.ndarray] | np.ndarray | None = None,
    weights: str | np.ndarray | None = None,
    class_weights: dict[str, float] | None = None,
    offset: dict[str, str | np.ndarray] | np.ndarray | None = None,
    seed: int | None = None,
    input_transforms: list[dict[str, Any]] | None = None,
) -> MultinomialDict:
    ...
```

### 8.2 Parsing and design

Reuse:

- dict parsing helpers from `formula.py`,
- `InteractionBuilder` from `interactions.py`,
- `transform_new_data()` for prediction.

Reject unsupported terms before fitting:

- target encoding,
- smooth auto-penalty,
- monotonicity,
- complement,
- exposure,
- alternative-specific terms.

### 8.3 Class coding

Rules:

- explicit `classes` defines output order,
- Polars Enum/Categorical order if classes omitted,
- otherwise sorted stringified labels,
- explicit `reference`, default first class,
- observed labels must be in classes,
- empty supplied classes fail.

Store:

- `classes_`,
- `reference_`,
- `_class_to_code`,
- `_code_to_class`.

### 8.4 Availability and offsets

Resolve fit-time availability:

- dict class to bool/column/array,
- or `n x K` array,
- default all available.

Persist column-based availability specs for prediction. Raw arrays are fit-time
only and require prediction-time override if needed.

Resolve offsets similarly. Missing class offsets default to zero.

### 8.5 Fit method

`MultinomialDict.fit(...)`:

- validates options,
- builds design,
- computes weights and class weights,
- computes standardization for ridge if requested,
- computes alternative-term standardization for ridge if requested,
- calls `_rustystats.fit_multinomial_py`,
- wraps result in `MultinomialModel`,
- sets inference status.

Fold `class_weights` into the single row-weight vector before calling Rust:

```text
effective_weight[i] = row_weight[i] * class_weights[y_i]
```

If `class_weights` is present, set inference status to a naive/misspecified
weighted-likelihood label whenever standard errors are exposed.

### 8.6 Prediction

`MultinomialModel.predict_proba(...)`:

- collect needed LazyFrame columns,
- apply input transforms,
- chunk design matrix for large data,
- compute logits from `X @ params.T`,
- apply offsets and availability,
- stable masked softmax,
- return NumPy or Polars.

`predict(...)` returns class labels.

`predict_log_proba(...)` returns stable log probabilities.

`decision_function(...)` returns class logits/utilities, with the reference
column included by default.

`predict_top_k(...)` returns the top-k class labels and probabilities.

`tier_mix(...)` aggregates predicted probabilities by optional weights.

`coef_table(...)` is required in Phase 1. It is the surface used by `summary()`
and by inference-honesty labels for ridge and class-weighted fits.

### 8.7 Tests

Python API tests:

- constructor exported,
- signature works with keyword calls,
- class ordering and reference,
- predict shapes and label outputs,
- LazyFrame collection,
- categorical/fixed-spline/expression/frequency terms,
- unsupported features fail clearly,
- availability and offsets fit/predict,
- row/class weights,
- class weights set the naive/misspecified inference label,
- ridge standardization,
- unpenalized reference invariance.
- `predict_log_proba`, `decision_function`, `predict_top_k`, and `coef_table`
  work and preserve class order.

Acceptance:

- Public API works end to end for shared-covariate tier conversion.

---

## 9. Stage 7 - Serialization

Purpose: make fitted multinomial models deployable.

### 9.1 Pattern

Reuse the scalar model pattern:

- Python pickle,
- schema version,
- result state dict,
- builder state,
- no raw training data.

Do not build a Rust serializer.

### 9.2 Persisted state

Persist:

- schema version,
- classes,
- reference,
- feature names,
- coefficient matrix,
- formula/dict spec,
- fitted `InteractionBuilder` state,
- input transforms,
- availability spec,
- offset spec,
- weight spec when column-based,
- regularization metadata,
- inference status,
- solver status,
- warnings.

### 9.3 Tests

- round-trip preserves `predict_proba`,
- round-trip preserves class labels and ordering,
- array availability/offset behavior is explicit,
- schema mismatch fails clearly.

Acceptance:

- Serialized models can be loaded and scored without training data.

---

## 10. Stage 8 - Summary and Minimal Diagnostics

Purpose: make the MVP usable to pricing teams before full diagnostics.

### 10.1 Summary

Include:

- model type,
- classes,
- reference,
- observations,
- parameters,
- convergence,
- iterations,
- log-likelihood,
- deviance,
- null deviance,
- AIC/BIC,
- log loss,
- actual class mix,
- predicted class mix,
- coefficient table by class.

Parameter count:

```text
df_model = (K - 1) * p
```

Do not double-count `result.intercepts`; they are derived from `params`.

### 10.2 Minimal diagnostics

Add `MultinomialDiagnostics` with:

- weighted log loss,
- deviance,
- null deviance,
- AIC,
- BIC,
- confusion matrix,
- accuracy,
- top-2 accuracy,
- actual class mix,
- predicted class mix,
- class mix error.

Full calibration and factor diagnostics can be Phase 2.

### 10.3 Tests

- summary includes class/reference,
- AIC/BIC parameter count is correct,
- confusion matrix shape is `K x K`,
- class mix totals match weights,
- diagnostics JSON serializes.

Acceptance:

- A pricing user can fit, score, and inspect tier mix.

---

## 11. Stage 9 - Documentation and Examples

Purpose: make the feature discoverable and honest.

Update:

- README feature table,
- docs getting-started quickstart,
- docs/components or docs/api multinomial page,
- changelog.

Add example:

```text
examples/tier_conversion_multinomial.py
```

Example should show:

- quote-tier response,
- explicit classes,
- reference `"none"`,
- availability columns,
- `predict_proba`,
- tier mix,
- held-out log loss,
- tier price/richness `alternative_terms`,
- price-change scenarios.

Acceptance:

- Docs do not imply shared-covariate MVP can do tier-specific price elasticity.
- New examples must pass the repository's lint/format gates over `python/`,
  `tests/`, and `examples/`; keep `examples/tier_conversion_multinomial.py`
  ruff-clean.

---

## 12. Stage 10 - Performance Hardening

Purpose: keep the dense MVP fast and bounded.

### 12.1 Benchmarks

Add benchmark script:

```text
benchmarks/bench_multinomial.py
```

Benchmark dimensions:

- `n = 10k, 50k, 200k`,
- `p = 20, 100, 500`,
- `K = 3, 4, 8`,
- all-available vs availability mask,
- unpenalized vs ridge.

Metrics:

- fit wall time,
- peak RSS,
- iterations,
- Hessian MB,
- prediction throughput.

### 12.2 Hot-path checks

Profile:

- gradient/Hessian accumulation,
- softmax,
- dense solve,
- prediction chunking.

Optimize only after correctness:

- row-chunk size,
- Hessian buffer reuse,
- avoiding final probability allocation until return,
- symmetric Hessian fill,
- termwise prediction if design width is high.

### 12.3 Performance gates

Before marking Phase 1 done:

- small models fit in seconds,
- memory guard triggers before large allocations,
- prediction chunks respect memory budget,
- no avoidable full `n x p x K` temporary allocations.

Acceptance:

- Dense path is bounded, predictable, and documented.

---

## 13. Suggested PR Sequence

### PR 1 - Core contracts and validation

Files:

- `crates/rustystats-core/src/solvers/multinomial.rs`
- `crates/rustystats-core/src/solvers/mod.rs`

Deliver:

- structs,
- validation,
- memory guard,
- validation tests.

### PR 2 - Math primitives

Deliver:

- masked softmax,
- objective,
- gradient,
- Hessian,
- finite-difference tests.

### PR 3 - Unpenalized dense Newton

Deliver:

- Newton loop,
- step-halving,
- null deviance,
- covariance,
- reference-invariance tests.

### PR 4 - Ridge and standardization

Deliver:

- ridge objective/gradient/Hessian,
- standardization/back-transform,
- reference-dependence tests,
- separation stabilization tests.

### PR 5 - PyO3 binding

Deliver:

- `multinomial_py.rs`,
- result getters,
- `_rustystats.pyi`,
- direct binding tests.

### PR 6 - Python public API

Deliver:

- `python/rustystats/multinomial.py`,
- `multinomial_dict`,
- `MultinomialDict`,
- `MultinomialModel`,
- design reuse,
- prediction,
- Python API tests.

### PR 7 - Serialization, summary, diagnostics

Deliver:

- `to_bytes/from_bytes`,
- summary,
- minimal `MultinomialDiagnostics`,
- docs for inference status.

### PR 8 - Docs, examples, performance

Deliver:

- README/docs,
- example script,
- benchmarks,
- performance notes.

This sequence avoids one huge risky PR and gives reviewers clear seams:
contracts, math, solver, binding, public API, and polish.

---

## 14. Final Acceptance Checklist

The Phase 1 feature is complete when all are true:

- Existing scalar tests pass.
- Rust finite-difference tests pass.
- Unpenalized all-available fits match `statsmodels.MNLogit` on probabilities,
  log-likelihood, and coefficient convention.
- Unpenalized probabilities are reference-invariant.
- Baseline ridge is documented and tested as reference-dependent.
- Availability affects fit and prediction.
- Null deviance handles varying availability through fitting.
- Empty classes fail early.
- Separation produces useful status/warnings.
- Direct PyO3 and public Python API tests pass.
- Serialization round-trip preserves probabilities.
- Summary and diagnostics use correct parameter count.
- Dense memory/parameter guard works.
- Benchmarks show bounded memory and acceptable runtime.
- Docs include an insurance tier-conversion example and limitation note.
