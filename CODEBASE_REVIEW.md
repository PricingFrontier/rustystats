# Codebase Review: Dead Code, Redundancy & Problematic Fallbacks

> Generated from a full review of the Python package, Rust core crate, Rust Python bindings, constants, and error handling patterns.

---

## 1. Dead / Unused Python Code

### 1a. `TargetEncodingTerm` class — dead code

**File:** `python/rustystats/target_encoding.py` (lines 531–617)

This class is exported in `__init__.py` but never used internally. The dict API uses `TargetEncodingTermSpec` (from `interactions.py`) instead. `TargetEncodingTerm` has its own `fit_transform`/`transform` pattern that duplicates the `InteractionBuilder._build_target_encoding_columns` pipeline. It also contains a Pandas compatibility fallback (`data[self.var_name].values`) that's inconsistent with the rest of the codebase (Polars-only).

**Recommendation:** Remove `TargetEncodingTerm` from `target_encoding.py` and from `__init__.py` exports. If any users depend on it externally, deprecate first.

---

### 1b. `wrap_fitting_error` and `wrap_prediction_error` — never called

**File:** `python/rustystats/exceptions.py` (lines 116–209)

These helper functions are defined and exported in `__all__` but never called anywhere in the codebase. The Rust layer raises its own errors, and the Python layer uses the exception classes directly.

**Recommendation:** Remove both functions and their `__all__` entries.

---

### 1c. `_GLMBase` docstring references removed `FormulaGLM` class

**File:** `python/rustystats/formula.py` (lines 346–348)

The `_GLMBase` class and `_fit_glm_core` function both reference `FormulaGLM` in comments/docstrings, but this class has been removed. Only `FormulaGLMDict` exists now.

**Recommendation:** Update docstrings to remove references to the removed `FormulaGLM` class.

---

### 1d. `_apply_inverse_link` duplicated in two places

There are two independent implementations of inverse link dispatch:

- `python/rustystats/formula.py` (lines 1115–1128) — method on `GLMModel`
- `python/rustystats/regularization_path.py` (lines 120–145) — standalone function

Both do the same thing with slightly different fallback behavior (see finding 3a/3b below).

**Recommendation:** Consolidate into a single utility function imported by both.

---

## 2. Unused Constants

### 2a. `MU_MIN_POISSON`, `MU_MIN_GAMMA`, `MU_BOUNDS_BINOMIAL` — never imported outside `constants.py`

**File:** `python/rustystats/constants.py` (lines 102–104)

These numerical stability constants are defined and exported but never imported or used anywhere in the Python codebase. All mu-clamping happens in Rust now.

**Recommendation:** Remove them. They served a purpose when Python did the IRLS loop, but that's all in Rust now.

---

### 2b. `inversegaussian` entry in `DEFAULT_LINKS` — unsupported family

**File:** `python/rustystats/constants.py` (line 131)

```python
"inversegaussian": "inverse_squared",
```

There is no `InverseGaussian` family implementation, no `inverse_squared` link implementation, and no way to use this. The Rust `family_from_name` and `link_from_name` functions will error on both.

**Recommendation:** Remove this entry. It gives false confidence that inverse Gaussian is supported.

---

## 3. Fallbacks That Should Error Instead of Silently Doing Something Wrong

### 3a. `GLMModel._apply_inverse_link` silently defaults to identity for unknown links

**File:** `python/rustystats/formula.py` (lines 1126–1128)

```python
        else:
            # Default to identity
            return eta
```

If the model has a link that's not `identity`/`log`/`logit`/`inverse`, this silently returns raw eta without any transformation. This would produce completely wrong predictions.

**Recommendation:** Raise `ValidationError(f"Unknown link function '{link}' for prediction")`.

---

### 3b. `regularization_path._apply_inverse_link` silently defaults to log for unknown links

**File:** `python/rustystats/regularization_path.py` (lines 144–145)

```python
    # Default to log link for unknown
    return np.exp(eta)
```

Same problem, different default. An unknown link silently does `exp(eta)` here but `eta` in `formula.py`. Both should error.

**Recommendation:** Raise an error instead.

---

### 3c. `glm.py summary()` backward compat fallback for `is_regularized`

**File:** `python/rustystats/glm.py` (lines 94–100)

```python
    try:
        is_reg = result.is_regularized
        penalty_type = result.penalty_type if is_reg else "none"
    except AttributeError:
        # Older result objects may not have these attributes - this is expected
        is_reg = False
        penalty_type = "none"
```

The comment says "older result objects" — but all result objects now go through `PyGLMResults` which always has these attributes. The `_DeserializedResult` dataclass also defines them. This backward-compat `try/except` is dead code.

**Recommendation:** Remove the try/except, access the attributes directly.

---

### 3d. `_process_offset` silently applies log based on a heuristic

**File:** `python/rustystats/formula.py` (lines 383–385)

```python
            if self._uses_log_link():
                if np.all(offset_values > 0) and np.mean(offset_values) > 0.01:
                    offset_values = np.log(offset_values)
```

The `np.mean(offset_values) > 0.01` check is a heuristic to detect whether values are already log-transformed. If someone passes exposure values that happen to be very small (e.g., fractional years < 0.01), this would silently skip the log transform, producing wrong results with no warning.

**Recommendation:** This heuristic is fragile. Consider removing the mean check and always logging positive values for log-link models, or at minimum emit a warning when the heuristic kicks in.

---

## 4. Prediction Path Reimplements Rust in Python (Systematic Issue)

**The entire `transform_new_data` prediction path** in `interactions.py` reimplements in slow Python for-loops what the training path (`build_design_matrix_from_parsed`) does via Rust. This is the single biggest cleanup opportunity in the codebase.

### Training vs Prediction — side by side

| Operation | Training (Rust) | Prediction (Python for-loops) | Rust function to use |
|---|---|---|---|
| Categorical encoding | `encode_categorical_py` (line 348) | `_encode_categorical_new` (line 1546) | `encode_categorical_py` |
| Target encoding | `target_encode_with_exposure_py` (line 842) | `_encode_target_new` (lines 1748, 1759) | `apply_target_encoding_py` / `apply_exposure_weighted_target_encoding_py` |
| Frequency encoding | `frequency_encode_py` | `_encode_frequency_new` (line 1790) | `apply_frequency_encoding_py` |
| Cat × Cat interaction | `build_cat_cat_interaction_py` (line 587) | `_build_categorical_interaction_new` (line 1644) | `build_cat_cat_interaction_py` |
| Cat × Cont interaction | `build_cat_cont_interaction_py` (line 727) | `_build_mixed_interaction_new` (lines 1680, 1695) | `build_cat_cont_interaction_py` / `multiply_matrix_by_continuous_py` |

**Recommendation:** Replace all `_*_new` prediction methods with calls to the same Rust functions used during training. Remove the Python for-loop implementations.

### 4a. Unused Rust registrations (separate from above)

These functions are registered in `crates/rustystats/src/lib.rs` but have no Python-side caller and are not part of the prediction path issue:

| Rust function | Line in lib.rs | Status |
|---|---|---|
| `encode_categorical_indices_py` | 88 | Optimized path taking pre-computed indices instead of strings. Could be wired up as fast path when Polars provides categorical codes. |
| `ms_names_py` | 82 | Monotonic spline name generator. Never imported. |
| `compute_lorenz_curve_py` | 112 | Lorenz curve computation. Never imported (discrimination stats use Gini directly). |
| `fit_negbinomial_py` | 62 | Full NegBin with automatic theta estimation via profile likelihood. Python uses `fit_glm_py` with fixed theta instead. |

**Recommendation:** Remove `ms_names_py`, `compute_lorenz_curve_py`, and `fit_negbinomial_py` (or wire up `fit_negbinomial_py` if automatic theta is desired). Keep `encode_categorical_indices_py` for potential future optimization.

---

## 5. Redundant / Can Be Simplified

### 5a. `CVPathPoint` struct with `#[allow(dead_code)]`

**File:** `crates/rustystats/src/fitting_py.rs` (lines 246–247)

```rust
#[derive(Clone)]
#[allow(dead_code)]
struct CVPathPoint { alpha: f64, cv_deviance_mean: f64, cv_deviance_se: f64, n_nonzero: usize }
```

The `n_nonzero` field is populated but never read. The `#[allow(dead_code)]` annotation is masking this.

---

### 5b. Redundant `isinstance` branches in `target_encoding.py`

**File:** `python/rustystats/target_encoding.py` (e.g., lines 110–113)

```python
    if isinstance(categories, np.ndarray):
        categories = [str(x) for x in categories]
    else:
        categories = [str(x) for x in categories]
```

Both branches do the same thing. This pattern repeats in `apply_target_encoding`, `frequency_encode`, `apply_frequency_encoding`, and `target_encode_interaction` (5 occurrences total).

**Recommendation:** Simplify to `categories = [str(x) for x in categories]`.

---

### 5c. `_build_formula_string` uses wrong default df values

**File:** `python/rustystats/formula.py` (lines 1770–1773)

```python
            elif term_type == "bs":
                df = spec.get("df", 5)
            elif term_type == "ns":
                df = spec.get("df", 4)
```

The actual default in `_parse_term_spec` is `DEFAULT_SPLINE_DF` (which is 10), not 5 or 4. This means the formula string representation will show wrong df values when they're not explicitly set. Not a runtime error, but misleading in diagnostics output.

---

## 6. Summary of Actionable Items

| Priority | Item | Action |
|---|---|---|
| **High** | Prediction path reimplements Rust in Python (4) | Replace all `_*_new` methods with Rust calls, remove Python for-loops |
| **High** | Silent wrong-link fallbacks (3a, 3b) | Change to raise errors |
| **High** | Unused Rust registrations (4a) | Remove `ms_names_py`, `compute_lorenz_curve_py`, `fit_negbinomial_py` |
| **Medium** | `TargetEncodingTerm` class (1a) | Remove dead class |
| **Medium** | `wrap_fitting_error`/`wrap_prediction_error` (1b) | Remove dead functions |
| **Medium** | Unused constants `MU_MIN_*` (2a) | Remove |
| **Medium** | `inversegaussian` in DEFAULT_LINKS (2b) | Remove phantom entry |
| **Medium** | Backward compat `try/except` in `glm.py` (3c) | Remove |
| **Low** | Duplicate `_apply_inverse_link` (1d) | Consolidate |
| **Low** | Redundant isinstance branches (5b) | Simplify |
| **Low** | Wrong default df in formula string (5c) | Fix defaults |
| **Low** | Stale docstring references to `FormulaGLM` (1c) | Update |
| **Low** | `CVPathPoint.n_nonzero` dead field (5a) | Clean up |
