# Native Multinomial Choice Support - Spec and Implementation Plan

RustyStats currently fits scalar-response GLMs. That is enough for binary
conversion, claim counts, severity, and pure premium, but it is not enough for
insurance product-tier conversion where a customer chooses exactly one outcome
from a competing set such as:

- no purchase
- basic
- standard
- premium

For that workflow, probabilities must be estimated jointly. A higher
probability for one tier necessarily lowers the probability mass available to
the others. Independent one-vs-rest binomial GLMs can be useful as a shortcut,
but they do not give pricing teams a coherent tier-mix model, calibrated class
probabilities, or a clean scenario engine for tier-specific price changes.

This spec adds **native baseline-category multinomial logit** support as a new
model path. It deliberately does not try to squeeze multinomial behavior into
the scalar `Family` trait.

---

## 1. Product Goal

Build a first-class native multinomial model for mutually exclusive insurance
choice outcomes.

The initial target use case is conversion across product tiers:

```python
result = rs.multinomial_dict(
    response="PurchasedTier",
    terms={
        "DriverAge": {"type": "bs", "df": 6},
        "Region": {"type": "categorical"},
        "VehicleValue": {"type": "linear"},
        "Channel": {"type": "categorical"},
    },
    data=quotes,
    classes=["none", "basic", "standard", "premium"],
    reference="none",
).fit()

probs = result.predict_proba(new_quotes)
tiers = result.predict(new_quotes)
mix = result.tier_mix(new_quotes)
```

The model should support the actuarial/pricing workflow:

- Estimate purchase probabilities for all tiers in one coherent probability
  vector per quote.
- Produce expected tier mix by portfolio segment.
- Compare scenarios by changing price or product attributes and recomputing
  class probabilities.
- Preserve the dict-first, programmatic API style already used by RustyStats.
- Reuse the existing design-matrix machinery where that machinery remains
  appropriate.
- Keep the Rust core pure and independently testable.

---

## 2. Core Decision

Implement this as a **new native multinomial path**, not as a scalar GLM family.

Current scalar GLM assumptions are pervasive:

- `Family` works with `Array1<f64>` responses and fitted means.
- `fit_glm_py` accepts one response vector.
- `GLMResults` stores one coefficient vector and one fitted value per row.
- `GLMModel.predict()` returns a one-dimensional prediction.

Native multinomial needs:

- a categorical response encoded as class indices,
- a coefficient matrix,
- a probability matrix,
- a softmax link,
- a block Hessian or equivalent optimizer,
- multiclass diagnostics,
- result objects whose primary prediction is `n_rows x n_classes`.

Trying to force that through the existing `Family` trait would either distort
the scalar API or leave hidden special cases throughout the codebase.

---

## 3. Model Specification

### 3.1 Baseline-Category Multinomial Logit

Let there be `K` classes. One class is selected as the reference class `r`.
For each non-reference class `k`, fit a linear predictor:

```text
eta[i, k] = X[i, :] dot beta[k, :] + offset[i, k]
eta[i, r] = 0
```

When a full class-specific utility offset is supplied, the reference class can
also receive an offset:

```text
eta[i, r] = offset[i, r]
```

Without an offset, the reference utility is zero. The reference class still has
no coefficient vector.

The class probabilities are:

```text
p[i, k] = exp(eta[i, k]) / sum_j exp(eta[i, j])
```

For numerical stability, evaluate softmax by subtracting the row maximum before
exponentiating.

The log-likelihood is:

```text
ll = sum_i weight[i] * log(p[i, y[i]])
```

The unpenalized objective minimized by the solver is:

```text
nll = -ll
```

The model deviance is:

```text
deviance = -2 * ll
```

The null deviance is the deviance of an intercept-only multinomial model using
the same class set, weights, and availability rules. When every class is
available on every row, the intercept-only solution is the weighted class-share
model. When availability varies by row, the null model must be fitted because
there is no single closed-form class-share solution.

### 3.2 Class Availability

Product tiers are not always available for every quote. The API should support
an optional availability matrix that forces unavailable alternatives to
probability zero.

```python
result = rs.multinomial_dict(
    response="PurchasedTier",
    terms=terms,
    data=quotes,
    classes=["none", "basic", "standard", "premium"],
    reference="none",
    availability={
        "none": True,
        "basic": "basic_available",
        "standard": "standard_available",
        "premium": "premium_available",
    },
).fit()
```

Rules:

- If `availability=None`, every class is available on every row.
- `availability` accepts a dict of class name to boolean scalar, column name, or
  one-dimensional boolean array.
- The observed response class must be available for its row.
- At least one class must be available for every row.
- Unavailable class probabilities are exactly zero.
- Softmax denominators include only available classes.
- Availability is used during fitting, not only during prediction. All
  log-likelihood, gradient, and Hessian calculations use the masked softmax.
- The reference class can be unavailable on some rows, but at least one class
  must remain available. This means the baseline utility is still a
  parametrization device, not a statement that the reference is always offered.

Availability support should be part of the native solver from the beginning,
even if the first public examples use all-available data. It is central to
pricing and product eligibility workflows.

### 3.3 Shared Covariates

The MVP supports row-level covariates shared by all classes:

```text
eta_basic    = X beta_basic
eta_standard = X beta_standard
eta_premium  = X beta_premium
eta_none     = 0
```

Each non-reference class gets its own coefficient vector for the same design
columns. A feature can therefore affect the relative log-odds of each tier
differently.

### 3.4 Alternative-Specific Covariates

Pricing teams eventually need tier-specific attributes:

- offered premium by tier,
- deductible by tier,
- limit by tier,
- plan richness,
- bundle discount,
- competitor price by tier,
- product availability or channel constraints.

The API should reserve room for this even if implementation lands after the
shared-covariate MVP.

Proposed wide-format API:

```python
result = rs.multinomial_dict(
    response="PurchasedTier",
    shared_terms={
        "DriverAge": {"type": "bs", "df": 6},
        "Region": {"type": "categorical"},
    },
    alternative_terms={
        "price": {
            "columns": {
                "basic": "price_basic",
                "standard": "price_standard",
                "premium": "price_premium",
            },
            "coefficient": "generic",
            "transform": "log",
        },
        "deductible": {
            "columns": {
                "basic": "deductible_basic",
                "standard": "deductible_standard",
                "premium": "deductible_premium",
            },
            "coefficient": "generic",
        },
    },
    data=quotes,
    classes=["none", "basic", "standard", "premium"],
    reference="none",
).fit()
```

The utility form becomes:

```text
U[i, k] = shared_X[i, :] dot beta[k, :]
          + alt_X[i, k, :] dot gamma
          + alt_offset[i, k]
```

Where `gamma` can be:

- `generic`: one coefficient shared across alternatives,
- `class_specific`: one coefficient per non-reference class,
- later: grouped constraints such as same coefficient for paid tiers only.

Phase 1 ships without public `alternative_terms`, but the result and solver
design should not make them hard to add.

---

## 4. Public API

### 4.1 Function

Add a new top-level constructor:

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

Notes:

- `terms=` is the ergonomic alias for `shared_terms=`.
- Passing both `terms` and `shared_terms` is an error.
- `alternative_terms` is intentionally not in the Phase 1 public signature.
  Reserve it for Phase 3 rather than adding an always-failing argument.
- `offset` is class-specific and lives on the utility/logit scale. It is not an
  exposure concept.
- An array `offset` has shape `n_rows x n_classes`; a dict `offset` maps class
  labels to columns or arrays. Missing class offsets default to zero.
- `weights` are row weights.
- `class_weights` multiply row weights by the observed response class weight.

### 4.2 Class Ordering and Reference

Class order is part of the model contract.

Rules:

- If `classes` is supplied, it defines output column order.
- If `classes` is omitted, derive classes deterministically from the response:
  preserve Polars Enum/Categorical ordering when present; otherwise sort unique
  stringified values.
- `reference` defaults to the first class in `classes`.
- Every observed response label must be present in `classes`.
- Every supplied class should have at least one observed row by default. A class
  with zero observations creates an intercept that wants to diverge to negative
  infinity; fail early unless a future explicit option keeps empty classes as
  prediction-only metadata.
- The reference class is included in `classes_`, prediction output, diagnostics,
  and serialization, but has no direct coefficient row.
- For unpenalized fits, changing the reference class must not change predicted
  probabilities, only coefficient parametrization.
- Baseline-category ridge/lasso/elastic-net penalties are generally
  reference-dependent because they shrink non-reference logits toward the chosen
  reference utility. If reference-invariant regularization is required, add a
  symmetric/sum-to-zero penalty as a later solver route rather than claiming
  invariance for baseline-penalized fits.

For pricing examples, documentation should recommend explicit classes and
explicit `reference="none"`.

### 4.3 Fit API

```python
result = model.fit(
    alpha: float = 0.0,
    l1_ratio: float = 0.0,
    regularization: str | None = None,
    cv: int | None = None,
    selection: str = "min",
    max_iter: int = 100,
    tol: float = 1e-8,
    standardize: bool = True,
    compute_covariance: bool = True,
    store_design_matrix: bool = False,
    verbose: bool = False,
)
```

Initial support:

- no penalty,
- ridge penalty,
- optional covariance for unpenalized and ridge fits.

Ridge in the MVP is useful for separation and unstable wide designs, but it is
not reference-invariant under the baseline parametrization. Summaries and docs
must state that unpenalized fits are the reference-invariant path; penalized
fits are conditional on the chosen reference unless a future symmetric penalty
is selected.

Deferred support:

- lasso,
- elastic net,
- CV regularization path,
- smooth-penalty GCV,
- multinomial target encoding.

If unsupported fit options are requested, raise a precise `ValidationError`
rather than silently falling back.

### 4.4 Result Object

Add `MultinomialModel` as the high-level Python result object.

Core attributes:

```python
result.classes_              # list[str], length K
result.reference_            # str
result.feature_names         # list[str], length p
result.params                # np.ndarray, shape (K - 1, p)
result.coef_matrix           # alias for params
result.intercepts            # np.ndarray, length K, reference intercept is 0
result.fitted_probabilities  # np.ndarray, shape (n, K)
result.linear_predictor      # np.ndarray, shape (n, K), reference column included
result.log_likelihood        # float
result.deviance              # float
result.null_deviance         # float
result.converged             # bool
result.iterations            # int
result.solver_status         # str
result.warnings              # list[str]
```

`result.intercepts` is a convenience view derived from the fitted intercept
column in `result.params` when `intercept=True`; prepend `0.0` for the reference
class. It is not a separately stored parameter and must not be double-counted in
degrees of freedom, AIC, or BIC. When `intercept=False`, return zeros or raise a
clear attribute error; pick one behavior and test it.

Prediction methods:

```python
result.predict_proba(new_data, return_format="numpy")
result.predict_log_proba(new_data, return_format="numpy")
result.decision_function(new_data, include_reference=True)
result.predict(new_data)
result.predict_top_k(new_data, k=2)
result.tier_mix(new_data, weights=None, return_format="dict")
```

Return-format rules:

- `return_format="numpy"` returns `np.ndarray` with columns in `classes_`.
- `return_format="polars"` returns a Polars DataFrame with columns like
  `prob_none`, `prob_basic`, `prob_standard`, `prob_premium`.
- `predict()` returns class labels, not integer codes.
- `predict_top_k()` returns labels and probabilities.

Coefficient access:

```python
result.coef_table(return_format="polars")
```

Rows should contain:

- class,
- feature,
- estimate,
- standard error when available,
- z value when available,
- p value when available,
- confidence interval when available,
- odds ratio versus reference for shared covariates when meaningful.

### 4.5 Summary

`result.summary()` should produce a compact table:

- model type,
- classes and reference,
- observations,
- parameters,
- method,
- regularization,
- iterations,
- convergence status,
- log-likelihood,
- deviance,
- null deviance,
- AIC/BIC,
- top-level metrics such as log loss and accuracy,
- coefficient blocks by class.

For AIC/BIC, the default parameter count is `(K - 1) * p`, where `p` is the
number of estimated shared-design columns in `params`, including the intercept
column when present. Do not add `result.intercepts` again; it is derived from
that same coefficient matrix. Future alternative-specific coefficient blocks
must add their estimated gamma parameters to this count.

For regularized fits, follow existing inference-honesty behavior: suppress or
label naive coefficient inference after selection or penalization.

---

## 5. Design Matrix Support

### 5.1 Reuse Existing Builder

The shared-covariate MVP should reuse `InteractionBuilder` in
`python/rustystats/interactions.py`, which is the existing design-matrix builder
used by `glm_dict`. The scalar orchestration in `formula.py` parses dict specs
and owns fitting control flow, but the reusable fitted design state lives on the
builder:

- linear terms,
- categorical terms,
- fixed-degree B-splines,
- fixed-degree natural splines,
- expression terms,
- standard interactions,
- frequency encoding,
- input transforms.

This is lighter than a full formula-layer rewrite. `InteractionBuilder` already
has `transform_new_data()`, fitted categorical levels, fitted spline knots,
frequency-encoding stats, term slots, and prediction-time column ordering. The
multinomial MVP mainly needs to:

- parse the dict spec through the same helpers used by scalar GLM,
- build `X` and feature names through `InteractionBuilder`,
- map the response labels to class codes in Python instead of treating the
  response as numeric `y`,
- reuse `transform_new_data()` for `predict_proba()`,
- reject response-dependent target encoding until a native multiclass encoder
  is implemented.

Any helper extraction should target the dict-parsing and builder-construction
boundary, not a wholesale rewrite of `formula.py`.

### 5.2 Initially Unsupported Term Behavior

The following should be rejected in the first native multinomial release unless
implemented explicitly:

- `target_encoding`,
- automatic penalized smooths (`bs`/`ns` without fixed `df`),
- monotonic smooth constraints,
- lasso credibility/complement,
- exposure,
- scalar-family diagnostics that assume one fitted mean.

Error messages should name the unsupported feature and the nearest supported
alternative.

Example:

```text
target_encoding is not yet supported for multinomial_dict. Use categorical or
frequency_encoding terms, or precompute multiclass encodings outside the model.
```

### 5.3 Multinomial Target Encoding - Later

Multiclass target encoding should be native rather than reusing the scalar mean
encoder. Proposed behavior:

- For a `target_encoding` term, emit one encoded column per non-reference class.
- Each encoded column is an ordered, smoothed class log-odds or class
  probability statistic.
- Use CatBoost-style ordered statistics for training rows.
- Store full training statistics for prediction rows.
- Respect the same fold boundaries used by CV.

Candidate encoding:

```text
TE_k(level) = log(
    (count_k(level) + prior_weight * prior_prob_k)
    /
    (count_ref(level) + prior_weight * prior_prob_ref)
)
```

This is interpretable as a prior-smoothed log-odds of class `k` against the
reference class. It also aligns with the baseline-category parametrization.

Open decision: probability columns may be numerically gentler than log-odds
columns for rare levels. This should be benchmarked before implementation.

---

## 6. Rust Core Architecture

### 6.1 New Module

Add:

```text
crates/rustystats-core/src/solvers/multinomial.rs
```

Export from:

```text
crates/rustystats-core/src/solvers/mod.rs
```

Core structs:

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
    pub coefficients: Array2<f64>,          // (k_minus_ref, p)
    pub fitted_probabilities: Array2<f64>,  // (n, k)
    pub linear_predictor: Array2<f64>,      // (n, k), reference included
    pub log_likelihood: f64,
    pub deviance: f64,
    pub null_deviance: f64,
    pub iterations: usize,
    pub converged: bool,
    pub covariance_unscaled: Option<Array2<f64>>, // flattened parameter order
    pub prior_weights: Array1<f64>,
    pub y_codes: Array1<usize>,
    pub reference_index: usize,
    pub warnings: Vec<String>,
    pub solver_status: String,
}
```

Do not store class labels in the Rust core. Labels are Python/API metadata.

### 6.2 Parameter Flattening

Rust solvers should flatten class coefficients in class-major order:

```text
theta = [
    beta[class_0_nonref, feature_0],
    beta[class_0_nonref, feature_1],
    ...
    beta[class_1_nonref, feature_0],
    ...
]
```

The Python result exposes a matrix, but covariance and Hessian internals use
the flattened order. `coef_table()` and diagnostics must use the same order.

Reserve the flattened ordering for future alternative-specific coefficients now
even though Phase 1 leaves that block empty:

```text
theta = [shared_beta_blocks..., alternative_gamma_block...]
```

That keeps covariance indexing and coefficient tables stable when Phase 3 adds
generic or class-specific price/product terms.

### 6.3 Newton / IRLS Solver

For the unpenalized and ridge MVP, use a dense Newton solver with step-halving.

Per iteration:

1. Compute logits and probabilities with stable masked softmax. If
   availability is supplied, unavailable class logits are excluded from the
   denominator and their probabilities are exactly zero.
2. Compute negative log-likelihood.
3. Compute gradient blocks:

```text
grad[k] = X.T @ (weights * (p[:, k] - 1[y == k]))
```

4. Compute Hessian blocks:

```text
H[k, l] = X.T @ diag(weights * p[:, k] * (1[k == l] - p[:, l])) @ X
```

Here `p` is always the fitted masked-softmax probability matrix. With
availability, unavailable classes have `p[:, k] = 0` for those rows, so their
gradient and Hessian contributions are zero. The observed class has already
been validated as available.

5. Add ridge diagonal to penalized parameters when `alpha > 0` and
   `l1_ratio == 0`.
6. Solve `H step = grad`.
7. Propose `theta_new = theta - step`.
8. Use step-halving until the penalized objective improves.
9. Stop when relative objective improvement or max absolute gradient is below
   tolerance.

Implementation notes:

- Reuse `nalgebra` for the dense solve, consistent with the scalar IRLS path.
- Accumulate gradient and Hessian in parallel by row chunks using Rayon.
- Maintain deterministic reductions where practical; tests should allow normal
  floating-point tolerance rather than byte equality for parallel summations.
- Clamp probabilities away from exact zero only for log evaluation, not for
  returned probabilities.
- Add explicit singular-Hessian diagnostics and ridge suggestions.
- Detect complete or quasi-complete separation: diverging coefficient norms,
  repeated step-halving without objective improvement, unobserved supplied
  classes, or probabilities saturating at machine precision. Warn clearly and
  suggest ridge regularization. A supplied class with zero observed responses
  should fail validation unless the user explicitly requests keeping empty
  classes for prediction metadata.

### 6.4 Memory Guard

The dense Hessian has size:

```text
q = p * (K - 1)
q x q x 8 bytes
```

Add a guard before allocation:

```text
if q * q * 8 > configurable_limit_bytes:
    raise "multinomial dense Hessian would require ...; use fewer columns,
    ridge regularization, or wait for the large-p solver"
```

Default guard: 256 MB and `max_dense_parameters=5000`, both configurable from
the Python layer later if needed. These defaults are intentionally aligned: a
256 MB single-Hessian byte guard binds near `q = 5792`, so the parameter guard
sits below that as a clearer compute-cost ceiling. Memory is not the only
concern: the dense solve is cubic in `q`, so a model that fits in memory can
still be too slow. Error messages should report both estimated Hessian memory
and `q`.

When covariance is requested, peak memory can be roughly 2-3x the single
Hessian size because the Hessian, inverse, and factorization workspace can
coexist. Very wide fits may need to skip covariance with a warning.

This prevents accidental crashes when a high-cardinality categorical produces a
wide design matrix, and it steers large pricing models toward a future
matrix-free or limited-memory solver.

### 6.5 Standardization

For regularized multinomial fits, standardize penalized shared-covariate columns
internally, matching the scalar regularization standardization behavior:

- intercept not standardized,
- constant columns not standardized,
- scale by weighted standard deviation,
- center when an intercept is present,
- scale-only when no intercept is present,
- back-transform coefficients and covariance before returning.

The same center/scale vector applies to each non-reference class coefficient
block.

### 6.6 Covariance and Inference

For unpenalized fits, the model-based covariance is the inverse Hessian at the
optimum.

For ridge fits:

- coefficient covariance is available as a naive penalized covariance,
- inference status should be marked as penalized/naive, consistent with scalar
  GLM behavior.

For class-weighted fits, model-based covariance is also naive/misspecified
unless a deliberate pseudo-likelihood inference story is added. Surface that in
`inference_status` rather than presenting ordinary standard errors as clean MLE
inference.

Robust covariance is deferred unless implemented deliberately. The multiclass
meat matrix differs from scalar Pearson residual machinery.

### 6.7 Later Large-Parameter Solver

Dense Newton is acceptable for the first native release when `K` is small and
the design is moderate. For larger pricing models, add a second solver route:

- L-BFGS or Newton-CG for unpenalized/ridge,
- proximal gradient or coordinate descent for lasso/elastic net,
- matrix-vector Hessian products instead of explicit dense Hessian.

Do not add this complexity to the first implementation unless dense Newton
blocks realistic examples.

---

## 7. PyO3 Binding Layer

Add:

```text
crates/rustystats/src/multinomial_py.rs
```

Expose:

```rust
#[pyfunction]
pub fn fit_multinomial_py(
    y_codes: PyReadonlyArray1<i64>,
    x: PyReadonlyArray2<f64>,
    n_classes: usize,
    reference_index: usize,
    availability: Option<PyReadonlyArray2<bool>>,
    offset: Option<PyReadonlyArray2<f64>>,
    weights: Option<PyReadonlyArray1<f64>>,
    alpha: f64,
    l1_ratio: f64,
    max_iter: usize,
    tol: f64,
    fit_intercept: bool,
    center: Option<PyReadonlyArray1<f64>>,
    scale: Option<PyReadonlyArray1<f64>>,
    skip_covariance: bool,
    hessian_memory_limit_bytes: usize,
    max_dense_parameters: usize,
    store_design_matrix: bool,
) -> PyResult<PyMultinomialResults>
```

The high-level Python API should expose `compute_covariance=True`, matching
`GLMModel.fit()`, and convert it to `skip_covariance=False` at the PyO3
boundary. Use `i64` for `y_codes` in the binding and stub because it matches
NumPy's default integer behavior.

Add `PyMultinomialResults` with NumPy getters:

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

Register the new functions/classes in:

```text
crates/rustystats/src/lib.rs
```

Update:

```text
python/rustystats/_rustystats.pyi
```

---

## 8. Python API Implementation

### 8.1 New Module

Add:

```text
python/rustystats/multinomial.py
```

Primary classes:

```python
class MultinomialDict:
    ...

class MultinomialModel:
    ...
```

Top-level constructor:

```python
def multinomial_dict(...):
    return MultinomialDict(...)
```

Export from:

```text
python/rustystats/__init__.py
```

### 8.2 Shared Design Builder Refactor

Reuse `InteractionBuilder` directly where possible, and extract only the
dict-parsing / builder-construction pieces that are currently coupled to
`FormulaGLMDict`.

Candidate helpers:

```python
build_design_from_dict(
    *,
    terms,
    interactions,
    data,
    response=None,
    input_transforms=None,
    seed=None,
    family_context="scalar" | "multinomial",
)
```

The helper should return:

- prepared data,
- response array or labels if requested,
- design matrix,
- feature names,
- builder object for prediction,
- fitted encoders/spline metadata,
- unsupported-feature diagnostics.

Keep this refactor scoped. Do not rewrite the whole scalar formula layer, and do
not duplicate `InteractionBuilder.transform_new_data()`.

### 8.3 Prediction

Prediction mirrors `GLMModel.predict()` but returns a probability matrix:

1. Resolve LazyFrame columns.
2. Apply input transforms.
3. Build `X_new` in chunks for wide/large data.
4. Compute logits for each non-reference class.
5. Insert reference logits.
6. Apply availability if prediction-time availability is configured.
7. Apply stable softmax.
8. Return NumPy or Polars output.

Prediction-time availability must be resolved from new data if it was specified
as columns at fit time.

### 8.4 Serialization

`MultinomialModel.to_bytes()` and `from_bytes()` should reuse the existing
Python serialization pattern: pickle a versioned state dict plus fitted builder
state. Do not build a separate Rust serializer. Use a multinomial-specific
schema-version namespace and result state shape.

Persist:

- schema version,
- classes,
- reference,
- feature names,
- coefficient matrix,
- formula/dict spec,
- fitted builder metadata,
- input transforms,
- availability spec,
- offset spec,
- training metadata,
- regularization metadata,
- solver status.

Do not serialize raw training data.

### 8.5 Releveling and Calibration

Scalar `relevel()` is a log-link multiplicative calibration and does not apply.

Multinomial calibration should be a separate feature:

- vector intercept calibration,
- temperature scaling,
- class-wise isotonic calibration,
- segment-level class-mix calibration.

Initial release:

- provide diagnostics,
- do not mutate fitted coefficients with calibration,
- raise a clear error for `relevel()`.

---

## 9. Diagnostics

Add a `MultinomialDiagnostics` path rather than forcing scalar
`ModelDiagnostics` to pretend probabilities are scalar means.

Initial method:

```python
diagnostics = result.diagnostics(
    train_data=quotes,
    test_data=holdout,
    categorical_factors=["Region", "Channel"],
    continuous_factors=["DriverAge", "VehicleValue"],
)
```

### 9.1 Model-Level Metrics

Report:

- weighted multiclass log loss,
- deviance,
- null deviance,
- McFadden pseudo R2,
- AIC,
- BIC,
- accuracy,
- top-2 accuracy,
- balanced accuracy,
- macro precision/recall/F1,
- per-class precision/recall/F1,
- confusion matrix,
- actual class mix,
- predicted class mix,
- class mix error by class.

Use the same estimated-parameter count as `summary()` for AIC/BIC:
`(K - 1) * p` for Phase 1 shared-covariate models, plus any future estimated
alternative-specific parameters.

### 9.2 Calibration

Calibration outputs:

- overall actual vs expected class counts,
- per-class one-vs-rest calibration curves,
- expected calibration error by class,
- multiclass expected calibration error using confidence bins,
- reliability by predicted winning class,
- tier-mix calibration by factor levels.

For insurance pricing, class-mix calibration is more important than raw
accuracy. The diagnostics should lead with actual/predicted tier mix.

### 9.3 Factor Diagnostics

For each factor/bin:

- exposure or row weight,
- actual class distribution,
- predicted class distribution,
- actual minus predicted by class,
- observed winning class,
- predicted winning class,
- chi-square style class-mix residual where meaningful.

This is the multiclass analogue of A/E factor diagnostics.

### 9.4 Scenario Diagnostics

Later, when alternative-specific price terms are supported, add:

```python
result.scenario(
    new_data,
    changes={
        "price_basic": 1.03,
        "price_standard": 1.01,
        "price_premium": 0.98,
    },
)
```

Outputs:

- base tier mix,
- scenario tier mix,
- probability deltas by class,
- expected premium/revenue if value columns are supplied,
- segment-level mix changes.

This should live in the multinomial result layer, not in the Rust core.

---

## 10. Export

Initial release can defer PMML/ONNX export, but it must fail explicitly:

```text
to_pmml does not yet support MultinomialModel.
to_onnx does not yet support MultinomialModel.
```

Later export requirements:

- coefficient matrix,
- softmax transform,
- class labels,
- categorical/spline preprocessing,
- availability behavior if representable,
- probability output columns.

PMML may support multinomial logistic regression directly. ONNX can represent
linear logits plus softmax.

---

## 11. Testing Plan

### 11.1 Rust Core Tests

Add tests under:

```text
crates/rustystats-core/src/solvers/multinomial.rs
crates/rustystats-core/tests/
```

Coverage:

- softmax rows sum to one,
- unavailable classes get zero probability,
- observed unavailable class raises,
- fitting with availability differs from fitting the same data while ignoring
  availability when availability is informative,
- gradient matches finite differences,
- Hessian matches finite differences,
- Hessian is symmetric,
- log-likelihood improves over iterations,
- step-halving handles a bad Newton step,
- intercept-only model recovers weighted class shares when all classes are
  available,
- intercept-only null deviance with varying availability is computed by fitting,
  not by global class shares,
- reference-class change preserves probabilities for unpenalized fits,
- baseline ridge is documented/tested as reference-dependent unless a future
  symmetric penalty is selected,
- ridge penalty shrinks coefficients,
- standardization preserves predictions under column rescaling,
- weighted fit matches row replication on small data,
- unobserved supplied classes fail early,
- complete/quasi-complete separation produces a clear warning or
  non-convergence status with a ridge suggestion,
- covariance dimensions and positive diagonal for well-posed data.

### 11.2 Python API Tests

Add:

```text
tests/python/test_multinomial.py
```

Coverage:

- `rs.multinomial_dict` is exported,
- classes and reference are preserved,
- string labels round-trip through prediction,
- `predict_proba()` shape and column order,
- Polars return format,
- `predict()` returns labels,
- LazyFrame column pruning works,
- fixed spline and categorical terms work,
- unsupported scalar-only terms raise clear errors,
- availability columns work at fit and prediction,
- availability affects fitted probabilities, not only prediction-time masking,
- class weights affect fit,
- class-weighted inference is labeled naive/misspecified when standard errors
  are exposed,
- serialization round-trip preserves probabilities,
- summary includes class/reference information,
- diagnostics returns class-mix metrics.

### 11.3 Reference Comparisons

Use `statsmodels.MNLogit` for unpenalized reference checks in dev tests:

- coefficients match up to reference-class/sign convention,
- predicted probabilities match,
- log-likelihood matches.

Use hand-computed tiny datasets for exact smoke tests so CI is not dependent on
subtle statsmodels convention differences.

### 11.4 Pricing Workflow Test

Create a synthetic quote-tier dataset:

- classes: none/basic/standard/premium,
- driver age,
- region,
- channel,
- tier-specific price columns,
- known utility coefficients.

For MVP shared-covariate tests, omit price terms or use shared price proxy.
For alternative-specific phase, verify:

- raising premium for a tier lowers that tier probability on average,
- total probabilities remain one,
- scenario tier mix moves in expected direction.

---

## 12. Implementation Phases

### Phase 0 - Spec and Scaffolding

- Add this spec.
- Create empty module placeholders if desired.
- Use `multinomial_dict` for Phase 1.
- Reserve `choice_dict` for Phase 3 alternative-specific choice models.
- Do not expose `alternative_terms` in the Phase 1 public signature; document it
  as planned.

### Phase 1 - Native Shared-Covariate MVP

Deliver:

- Rust dense Newton multinomial solver.
- PyO3 `fit_multinomial_py`.
- Python `multinomial_dict`.
- Shared covariate design matrix support.
- Classes/reference handling.
- Row weights and class weights.
- Availability matrix.
- Prediction methods.
- Basic summary.
- Serialization.
- Core tests and Python API tests.

Supported terms:

- linear,
- categorical,
- expression,
- fixed-degree `bs`,
- fixed-degree `ns`,
- frequency encoding,
- standard interactions.

Unsupported with clear errors:

- target encoding,
- automatic smooth penalties,
- lasso/elastic net,
- CV,
- monotonic constraints,
- alternative-specific terms,
- export.

### Phase 2 - Pricing-Grade Diagnostics

Deliver:

- multiclass diagnostics object,
- log loss/deviance/null deviance/AIC/BIC,
- confusion matrix,
- tier-mix actual vs predicted,
- class-wise calibration,
- factor-level class-mix diagnostics,
- train/test comparison,
- JSON output.

### Phase 3 - Alternative-Specific Choice Terms

Deliver:

- `alternative_terms` wide-format parser,
- generic alternative coefficient support,
- class-specific alternative coefficient support,
- prediction and scenario support,
- availability interactions,
- synthetic pricing workflow tests.

This is the phase that turns native multinomial into a practical pricing
scenario engine.

### Phase 4 - Regularization and CV

Deliver:

- ridge CV path,
- lasso/elastic net solver route,
- standardization with coefficient back-transform,
- multinomial alpha-max computation,
- fold-safe CV for future target encoding,
- inference-honesty labels.

Implementation options:

- dense Newton for ridge,
- proximal gradient or coordinate descent for lasso/elastic net,
- later L-BFGS/Newton-CG for large models.

### Phase 5 - Smooths, Target Encoding, Export

Deliver as separate tickets:

- multinomial smooth-penalty support,
- multinomial target encoding,
- ONNX export,
- PMML export,
- calibration objects,
- optional ordinal logit model.

---

## 13. Acceptance Criteria

The native feature is ready for first release when:

- A user can fit `none/basic/standard/premium` on a Polars DataFrame using
  `multinomial_dict`.
- `predict_proba()` returns stable probabilities whose rows sum to one.
- Unpenalized predicted probabilities are invariant to changing the reference
  class.
- Penalized baseline-category fits document their reference dependence unless a
  future symmetric penalty route is selected.
- Class availability is enforced in fit and predict.
- Varying class availability affects the fitted likelihood and probabilities,
  not only prediction-time output masking.
- Weighted fits behave like row replication on small examples.
- Unobserved supplied classes and separation-like fits fail or warn clearly with
  ridge guidance.
- Results match `statsmodels.MNLogit` on unpenalized all-available reference
  cases.
- Basic fixed-df splines, categoricals, expressions, and interactions work.
- Unsupported scalar-only features raise clear errors.
- The model serializes/deserializes and preserves predictions.
- Diagnostics include at least log loss, deviance, confusion matrix, and
  actual/predicted tier mix.
- README or docs include an insurance tier-conversion example.

---

## 14. Risks and Design Notes

### 14.1 IIA Assumption

Baseline multinomial logit has the independence of irrelevant alternatives
property. This may be imperfect for tiered insurance products where adjacent
tiers are closer substitutes than distant tiers.

Why still start here:

- It is the standard coherent baseline.
- It is interpretable.
- It supports pricing scenario work.
- It is much simpler than nested logit or mixed logit.
- It establishes the shared multiclass infrastructure needed by richer choice
  models later.

Future extension:

- nested logit for grouped alternatives,
- ordered logit for truly ordinal outcomes,
- mixed logit for random coefficients.

### 14.2 Tier Order

Insurance tiers are often ordered, but purchase choice is not always an ordinal
response problem:

- "none" is not naturally ordered with coverage tiers,
- price and eligibility can make choices non-monotone,
- tier names may encode product design rather than a single severity scale.

Therefore the first model should be nominal multinomial, not proportional-odds
ordinal regression. Ordinal models can be added later as separate constructors.

### 14.3 Dense Hessian Size

The dense Newton MVP is simple and statistically direct but can be memory-heavy.
High-cardinality categoricals can create large `p`, and the Hessian scales with
`(p * (K - 1))^2`.

Mitigations:

- memory guard,
- clear error messages,
- ridge suggestions,
- frequency encoding support,
- later large-parameter solver.

### 14.4 Monotonicity

Scalar monotonicity constraints do not translate cleanly to class probabilities.
A positive coefficient for `premium` versus `none` does not guarantee that
`P(premium)` is monotone when all class logits move together.

Initial policy:

- reject monotonic constraints in `multinomial_dict`,
- document why,
- later support logit-level sign constraints with clear wording.

### 14.5 Target Encoding Leakage

Multiclass target encoding is response-dependent and must respect folds,
permutations, class order, and reference class. It should be implemented
deliberately after the core solver, not rushed into the MVP.

---

## 15. Resolved Decisions and Remaining Questions

Resolved for Phase 1:

- Do not expose `alternative_terms` until Phase 3.
- Do not alias `choice_dict` to `multinomial_dict`; reserve `choice_dict` for
  alternative-specific choice models.
- If `classes=None`, preserve Polars Enum/Categorical ordering when available;
  otherwise sort stringified labels for reproducibility across data subsets.
- Include ridge as an MVP option for stabilization, but document that baseline
  ridge is reference-dependent. Unpenalized fits are the reference-invariant
  path.
- Use a default dense-Hessian memory guard of 256 MB plus
  `max_dense_parameters=5000`.
- Represent availability as an `n x K` boolean matrix in Phase 1; compact
  per-row available-class lists are a later optimization.
- Use long coefficient tables with one row per `(class, feature)` and optionally
  add a wide pivot convenience later.
- Add a new `MultinomialDiagnostics` dataclass rather than overloading scalar
  `ModelDiagnostics`.

Remaining questions:

1. Should Phase 1 expose knobs for the dense-Hessian memory/parameter guard, or
   keep them private until a user hits the limit?
2. Should a symmetric/sum-to-zero ridge penalty be designed as the Phase 4
   reference-invariant regularization route?
3. Should empty classes ever be allowed as prediction-only metadata, and if so
   what explicit opt-in flag should control that behavior?
4. How much diagnostics work belongs in Phase 1 versus Phase 2? A minimal
   tier-mix diagnostic is required for usefulness, but full calibration can be
   staged.

---

## 16. Recommended First Implementation Slice

The best first slice is:

1. Add `multinomial_dict` with shared terms only.
2. Implement dense Newton unpenalized plus ridge.
3. Support explicit `classes`, explicit `reference`, row weights, and
   availability.
4. Reuse fixed-design features: linear, categorical, expression, fixed splines,
   frequency encoding, interactions.
5. Add `predict_proba`, `predict`, `summary`, and serialization.
6. Add minimal diagnostics: log loss, deviance, confusion matrix, actual vs
   predicted class mix.

That slice is native, coherent, and useful for tier conversion. It also creates
the right foundation for alternative-specific price terms, which is where the
feature becomes especially valuable for pricing teams.

Important limitation: the shared-covariate MVP can model tier mix, but it cannot
answer tier-specific price-change scenarios such as "raise premium price by 3%"
unless price is represented in future `alternative_terms`. Phase 3 should be
co-designed early with pricing stakeholders even if it ships after Phase 1.
