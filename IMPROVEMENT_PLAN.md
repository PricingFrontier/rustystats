# RustyStats Comprehensive Improvement Plan

> Full codebase review completed. This document identifies architectural simplifications,
> methodology improvements, and code quality issues across all layers.
> Items are grouped by theme and prioritized for multi-week execution.

---

## Executive Summary

RustyStats is a well-structured GLM library with solid fundamentals. The core IRLS solver,
family/link trait system, and Rust-Python bridge are all functional and performant.
However, there are significant opportunities to **simplify** the codebase by reducing
duplication, consolidating parallel code paths, and pushing more logic into Rust.

**Key themes:**
1. **Duplicate code paths** — multiple functions that do nearly the same thing
2. **Python does too much** — orchestration logic that should be in Rust
3. **Constants scattered** — Rust and Python each have their own constants
4. **Result type bloat** — too many fields stored on result objects
5. **Solver consolidation** — 5+ entry points that could be 1-2

---

## 1. SOLVER CONSOLIDATION (High Priority)

### Problem
`solvers/irls.rs` exports 5 fitting functions with overlapping signatures:
- `fit_glm` (basic)
- `fit_glm_full` (with offset/weights)
- `fit_glm_warm_start` (with init coefficients)
- `fit_glm_regularized` (Ridge penalty)
- `fit_glm_regularized_warm` (Ridge + warm start)

These share ~90% of their code via `fit_glm_core`, but the caller has to pick the right
entry point. Additionally, `fit_glm_coordinate_descent` and `fit_smooth_glm_full_matrix`
are separate paths.

### Proposed Simplification
**Reduce to 2 entry points:**

```
fit_glm(y, x, family, link, config) -> IRLSResult
fit_glm_smooth(y, x, family, link, smooth_specs, config) -> SmoothGLMResult
```

Where `config` is a unified struct:

```rust
pub struct FitConfig {
    pub max_iterations: usize,
    pub tolerance: f64,
    pub min_weight: f64,
    pub offset: Option<Array1<f64>>,
    pub weights: Option<Array1<f64>>,
    pub init_coefficients: Option<Array1<f64>>,  // warm start
    pub regularization: RegularizationConfig,     // Ridge/Lasso/ElasticNet
    pub nonneg_indices: Vec<usize>,
    pub nonpos_indices: Vec<usize>,
    pub verbose: bool,
}
```

The single `fit_glm` internally dispatches:
- No penalty → standard IRLS
- Ridge only → modified IRLS with λI
- L1 component → coordinate descent

**Impact:** ~400 lines of duplicated code eliminated. Python binding layer simplifies from
5 conditional branches to 1 call.

### Files affected
- `crates/rustystats-core/src/solvers/irls.rs` — merge 5 functions into 1
- `crates/rustystats-core/src/solvers/mod.rs` — simplify re-exports
- `crates/rustystats/src/fitting_py.rs` — simplify `fit_glm_py` to build config + call
- `python/rustystats/formula.py` — simplify `_fit_glm_core`

---

## 2. RESULT TYPE SIMPLIFICATION (High Priority)

### Problem
`PyGLMResults` stores 15 fields including the full design matrix (`Array2<f64>`), the
full response vector, IRLS weights, and prior weights. For a model with 100k observations
and 50 parameters, the design matrix alone is ~40MB. This is stored redundantly —
the Python `GLMModel` also sometimes stores it.

Additionally, `SmoothGLMResult` is a separate struct with its own 15+ fields that
partially overlap with `IRLSResult`.

### Proposed Simplification
1. **Don't store design matrix in Rust result.** It's already available in Python (the
   caller created it). Only store it if explicitly requested.
2. **Merge `SmoothGLMResult` fields into `IRLSResult`** with optional smooth metadata:

```rust
pub struct IRLSResult {
    pub coefficients: Array1<f64>,
    pub fitted_values: Array1<f64>,
    pub linear_predictor: Array1<f64>,
    pub deviance: f64,
    pub iterations: usize,
    pub converged: bool,
    pub covariance_unscaled: Array2<f64>,
    pub irls_weights: Array1<f64>,
    pub family_name: String,
    pub penalty: Penalty,
    // Optional smooth metadata (None for standard GLMs)
    pub smooth: Option<SmoothMetadata>,
}

pub struct SmoothMetadata {
    pub lambdas: Vec<f64>,
    pub smooth_edfs: Vec<f64>,
    pub total_edf: f64,
    pub gcv: f64,
}
```

3. **`PyGLMResults` should lazily compute diagnostics** rather than storing all inputs.
   Store a reference to y/X only when needed (e.g., for residuals).

### Impact
- ~30% memory reduction for fitted models
- One result type instead of two
- Simpler Python wrapper (no separate `smooth_result_to_py` conversion)

### Files affected
- `crates/rustystats-core/src/solvers/irls.rs` — add `smooth` field
- `crates/rustystats-core/src/solvers/smooth_glm.rs` — return `IRLSResult`
- `crates/rustystats/src/results_py.rs` — remove design_matrix/y/prior_weights storage
- `crates/rustystats/src/fitting_py.rs` — simplify conversion

---

## 3. CONSTANTS CONSOLIDATION (Medium Priority)

### Problem
Constants are defined in **two places** with partially overlapping values:
- `crates/rustystats-core/src/constants.rs` (Rust)
- `python/rustystats/constants.py` (Python)

Some values differ between the two (e.g., Rust `DEFAULT_MAX_ITER = 25`, Python also
defines `DEFAULT_MAX_ITER = 25` but independently). This is a maintenance hazard.

### Proposed Simplification
**Python constants should be derived from Rust** where possible:
1. Expose Rust constants via a `constants_py` module in the bindings crate
2. Python `constants.py` imports from `_rustystats` and only adds Python-specific ones
3. Alternatively: keep Python constants but add a test that validates they match Rust

### Files affected
- `crates/rustystats/src/lib.rs` — add constants module
- `python/rustystats/constants.py` — import from Rust or add validation test
- `tests/python/` — add constant-sync test

---

## 4. PYTHON LAYER SIMPLIFICATION (High Priority)

### 4a. `formula.py` is too large (2044 lines)

This single file contains:
- `_GLMBase` (base class)
- `GLMModel` (result wrapper, ~300 lines)
- `FormulaGLMDict` (dict API class)
- `glm_dict()` (convenience function)
- `dict_to_parsed_formula()` (parser)
- `_parse_term_spec()` / `_parse_interaction_spec()` (parsing helpers)
- `_fit_glm_core()` / `_fit_with_smooth_penalties()` (fitting orchestration)
- `_build_results()` (result construction)
- `_DeserializedResult` / `_DeserializedBuilder` (serialization)

**Proposed split:**
- `formula/model.py` — `GLMModel` class
- `formula/dict_api.py` — `FormulaGLMDict`, `glm_dict`, parsers
- `formula/fitting.py` — `_fit_glm_core`, `_fit_with_smooth_penalties`
- `formula/serialization.py` — `_DeserializedResult`, `_DeserializedBuilder`

### 4b. `interactions.py` is too large (1823 lines)

Contains `InteractionBuilder` (the core design matrix builder), `ParsedFormula`,
and many dataclass specs. The `InteractionBuilder` class has too many responsibilities.

**Proposed split:**
- `interactions/types.py` — all dataclasses (`ParsedFormula`, `InteractionTerm`, etc.)
- `interactions/builder.py` — `InteractionBuilder` class
- `interactions/encoding.py` — target/frequency encoding column builders

### 4c. `_apply_inverse_link` duplicated in Python

`GLMModel._apply_inverse_link()` re-implements link function inverses in Python.
This should call the Rust link implementations for consistency.

### Files affected
- `python/rustystats/formula.py` — split into subpackage
- `python/rustystats/interactions.py` — split into subpackage

---

## 5. ndarray ↔ nalgebra CONVERSION OVERHEAD (Medium Priority)

### Problem
The codebase uses **two** linear algebra libraries:
- `ndarray` for storage and PyO3/NumPy interop
- `nalgebra` for actual linear algebra (Cholesky, LU, SVD)

Every linear algebra operation requires:
1. Convert `ndarray::Array2` → `nalgebra::DMatrix` (copy + transpose)
2. Do the operation
3. Convert back

This happens in `convert.rs`, `irls.rs` (`compute_xtwx_xtwz`), `gcv_optimizer.rs`
(`to_nalgebra`), and `smooth_glm.rs`.

### Proposed Simplification
**Option A: Use nalgebra throughout the core.** Only convert at the PyO3 boundary.
This eliminates all internal conversions.

**Option B: Use ndarray-linalg** (LAPACK bindings for ndarray) to avoid nalgebra entirely.
This eliminates the dual-library issue.

**Recommended: Option A.** nalgebra is already the workhorse. Make it the internal
representation and convert only in the `rustystats` (bindings) crate.

### Impact
- Eliminates ~200 lines of conversion code
- Removes ~30% of allocations in the solver hot path
- `convert.rs` becomes a thin boundary layer in the bindings crate

### Files affected
- `crates/rustystats-core/src/convert.rs` — simplify or move to bindings
- `crates/rustystats-core/src/solvers/irls.rs` — use DMatrix internally
- `crates/rustystats-core/src/solvers/gcv_optimizer.rs` — remove `to_nalgebra`
- `crates/rustystats-core/src/solvers/smooth_glm.rs` — use DMatrix internally

---

## 6. FAMILY DISPATCH VIA STRING MATCHING (Medium Priority)

### Problem
Family selection uses string matching in multiple places:
- `families_py::family_from_name()` in the bindings crate
- `PyGLMResults::get_family()` re-parses the family name string
- `coordinate_descent.rs` checks `family.name()` for initialization
- `irls.rs` checks `family.name()` for "Gamma"/"Tweedie" true-Hessian optimization

This is fragile — if someone changes a family's `name()` return value, multiple
places break silently.

### Proposed Simplification
1. Add a `FamilyType` enum to `rustystats-core`:
   ```rust
   pub enum FamilyType {
       Gaussian, Poisson, Binomial, Gamma,
       Tweedie(f64), QuasiPoisson, QuasiBinomial,
       NegativeBinomial(f64),
   }
   ```
2. `Family` trait gets `fn family_type(&self) -> FamilyType`
3. All dispatch uses `match` on the enum instead of string comparisons

### Impact
- Compile-time safety for family dispatch
- Eliminates string parsing bugs
- Cleaner code in solver hot paths

### Files affected
- `crates/rustystats-core/src/families/mod.rs` — add `FamilyType` enum
- `crates/rustystats-core/src/families/*.rs` — implement `family_type()`
- `crates/rustystats-core/src/solvers/irls.rs` — use enum matching
- `crates/rustystats/src/families_py.rs` — simplify dispatch

---

## 7. SCALE COMPUTATION DUPLICATION (Medium Priority)

### Problem
`PyGLMResults` recomputes `scale()` (dispersion parameter) every time any of these
methods is called: `bse()`, `tvalues()`, `pvalues()`, `conf_int()`,
`significance_codes()`. Each call independently calls `get_family()` (which does
string matching) and `estimate_dispersion_pearson()`.

### Proposed Simplification
Cache the scale value after first computation:
```rust
fn scale(&self) -> f64 {
    // Compute once, cache in a Cell<Option<f64>>
}
```

Or better: compute `scale`, `bse`, `tvalues`, `pvalues` all at once in a single
`summary_statistics()` method that returns a struct, avoiding redundant computation.

### Files affected
- `crates/rustystats/src/results_py.rs` — add caching or batch computation

---

## 8. SMOOTH GLM COMPLEXITY (Medium Priority)

### Problem
`smooth_glm.rs` (1147 lines) contains:
- `SmoothTermData` — data-oriented term spec
- `SmoothTermSpec` — column-range-oriented term spec
- `SmoothGLMResult` — full result struct (separate from `IRLSResult`)
- `SmoothGLMConfig` — config (separate from `IRLSConfig`)
- `solve_constrained_wls()` — monotonic WLS solver
- `fit_smooth_glm_full_matrix()` — main entry point
- `assemble_smooth_result_from_specs()` — result builder

There are two term representations (`SmoothTermData` and `SmoothTermSpec`) that
partially overlap.

### Proposed Simplification
1. Remove `SmoothTermData` — use `SmoothTermSpec` everywhere (it already has all needed info)
2. Merge `SmoothGLMResult` into `IRLSResult` with optional smooth metadata (see item 2)
3. Merge `SmoothGLMConfig` into the unified `FitConfig` (see item 1)

### Files affected
- `crates/rustystats-core/src/solvers/smooth_glm.rs`
- `crates/rustystats/src/fitting_py.rs`

---

## 9. PARALLEL COMPUTATION GRANULARITY (Low Priority)

### Problem
`coordinate_descent.rs` uses `rayon` parallel iterators for computing X'Wz and X'WX
even for small problems. The parallelism overhead for small matrices (e.g., p < 20,
n < 1000) likely exceeds the benefit.

Similarly, `irls.rs` parallelizes the IRLS weight computation loop even for small n.

### Proposed Simplification
Add a threshold: only parallelize when `n * p > PARALLEL_THRESHOLD` (e.g., 50,000).

```rust
const PARALLEL_THRESHOLD: usize = 50_000;

if n * p > PARALLEL_THRESHOLD {
    // rayon parallel
} else {
    // sequential
}
```

### Files affected
- `crates/rustystats-core/src/solvers/irls.rs`
- `crates/rustystats-core/src/solvers/coordinate_descent.rs`

---

## 10. CV PATH IMPROVEMENTS (Medium Priority)

### Problem
`fit_cv_path_py` in `fitting_py.rs`:
1. Uses hash-based fold assignment (not truly random, can have imbalanced folds)
2. Hardcodes `mu_val = lp_off.mapv(|x| x.clamp(-700.0, 700.0).exp())` which assumes
   log link — wrong for identity/logit links
3. The CV path logic is partially duplicated in `python/rustystats/regularization_path.py`

### Proposed Fix
1. Use proper stratified k-fold (or at least random shuffle with seed)
2. Apply the correct inverse link function based on the model's link
3. Consolidate CV path logic — either fully in Rust or fully in Python

### Files affected
- `crates/rustystats/src/fitting_py.rs` — fix fold assignment and inverse link
- `python/rustystats/regularization_path.py` — consolidate or remove

---

## 11. ERROR HANDLING IMPROVEMENTS (Low Priority)

### Problem
`exceptions.py` defines 7 exception types but they're inconsistently used:
- `wrap_fitting_error()` and `wrap_prediction_error()` exist but aren't called
  from most code paths
- Rust errors are caught with generic `.map_err(|e| PyValueError::new_err(...))`
  instead of mapping to specific Python exceptions

### Proposed Fix
1. Map `RustyStatsError` variants to specific Python exceptions in the bindings
2. Use `wrap_fitting_error()` in `fitting_py.rs`
3. Add `from RustyStatsError` conversion trait for PyO3

### Files affected
- `crates/rustystats/src/fitting_py.rs`
- `crates/rustystats-core/src/error.rs`

---

## 12. TEST COVERAGE GAPS (Medium Priority)

### Current test files (Python):
- `test_dict_api.py` (1634 lines) — comprehensive
- `test_diagnostics.py` (1268 lines) — good coverage
- `test_families.py` — family-specific tests
- `test_interactions.py` — interaction term tests
- `test_links.py` — link function tests
- `test_target_encoding.py` — TE tests
- `test_frequency_encoding.py` — FE tests

### Missing test coverage:
1. **No edge-case tests for coordinate descent** (Lasso/Elastic Net convergence)
2. **No tests for monotonic spline fitting** via the dict API
3. **No property-based tests** (e.g., coefficients under identity link ≈ OLS)
4. **No tests for prediction on new data with unseen categorical levels**
5. **Rust unit tests** exist in solver modules but are minimal — only basic
   Gaussian and Poisson cases
6. **No integration test** that validates against statsmodels for all families

### Proposed additions:
- `test_monotonic_splines.py` — monotonic constraint tests
- `test_edge_cases.py` — degenerate inputs, single-observation, all-same response
- `test_statsmodels_validation.py` — parity tests against statsmodels
- Expand Rust `#[cfg(test)]` modules in `irls.rs`, `coordinate_descent.rs`

---

## 13. SERIALIZATION SECURITY (Low Priority)

### Problem
`GLMModel.to_bytes()` / `from_bytes()` uses `pickle` which is inherently insecure.
Loading a pickle from an untrusted source can execute arbitrary code.

### Proposed Fix
Use a safe format (e.g., `msgpack`, `JSON + numpy save`, or custom binary format).
At minimum, add a warning in the docstring about pickle security.

### Files affected
- `python/rustystats/formula.py` — `GLMModel.to_bytes()` / `from_bytes()`

---

## 14. DOCUMENTATION IMPROVEMENTS (Low Priority)

### Issues found:
1. `README.md` references `FormulaGLM` class which appears to have been removed
   (replaced by `FormulaGLMDict`)
2. Docs mention formula string API (`"y ~ x1 + C(cat) + bs(x2, df=5)"`) but
   this interface appears to have been removed in favor of dict API only
3. `ARCHITECTURE_REVIEW.md` and `IMPLEMENTATION_GUIDE.md` exist but may be stale
4. Module-level docstrings in Rust are excellent — maintain this quality
5. Python docstrings are comprehensive — maintain this quality

### Files affected
- `README.md` — update to match current API
- `docs/` — audit for stale references

---

## 15. VERSION CONSISTENCY (Low Priority)

### Problem
Version numbers appear in 3 places:
- `pyproject.toml`: `version = "0.3.8"`
- `python/rustystats/__init__.py`: `__version__ = "0.3.8"`
- `crates/rustystats/Cargo.toml`: `version = "0.2.5"`
- `crates/rustystats-core/Cargo.toml`: `version = "0.1.0"`

The Rust crate versions are out of sync with the Python version.

### Proposed Fix
Use a single source of truth. Either:
- Derive all versions from `pyproject.toml` at build time
- Or keep them in sync manually with a CI check

---

## Execution Priority

### Week 1-2: Solver Consolidation + Result Simplification
- Items 1, 2 (solver merge, result type simplification)
- These are the highest-impact changes, touching the core architecture

### Week 3: Python Layer Refactoring
- Items 4a, 4b (split formula.py and interactions.py)
- Item 8 (smooth GLM simplification — feeds into solver consolidation)

### Week 4: Type Safety + Constants
- Item 6 (FamilyType enum)
- Item 3 (constants consolidation)
- Item 7 (scale computation caching)

### Week 5: Cross-cutting Quality
- Item 5 (ndarray ↔ nalgebra)
- Item 10 (CV path fixes)
- Item 12 (test coverage)

### Ongoing / As-needed:
- Items 9, 11, 13, 14, 15 (parallelism threshold, error handling,
  serialization, docs, versions)

---

## Metrics for Success

After implementing this plan, the codebase should achieve:
- **~30% fewer lines of code** in solver layer (from consolidation)
- **~40% fewer public API functions** in Rust core (simpler interface)
- **Zero string-based dispatch** in hot paths (type-safe family enum)
- **Single result type** for both standard and smooth GLMs
- **One fitting entry point** that handles all regularization types
- **Python files under 500 lines each** (from splitting large modules)
- **Full test parity** with statsmodels for all supported families
