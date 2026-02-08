# RustyStats Comprehensive Architecture Review

**Date**: February 2026  
**Scope**: Full codebase review for top-tier coding standards  
**Focus**: Process simplification, modular design, and architectural improvements

---

## Executive Summary

RustyStats is a well-structured GLM library with a Rust core and Python bindings. The codebase demonstrates strong fundamentals but has accumulated complexity that can be simplified. This document outlines a multi-week improvement plan organized into prioritized work streams.

**Key Strengths:**
- Clean Rust/Python separation with PyO3 bindings
- Comprehensive feature set (8 families, splines, target encoding, diagnostics)
- Good test coverage across 17 test files
- Strong documentation in code comments

**Key Areas for Improvement:**
- Duplicated logic across Python wrappers
- Complex multi-step processes that could be streamlined
- Inconsistent abstraction levels between components
- Overly large files that mix concerns

---

## Part 1: Current Architecture Map

### 1.1 Layer Structure

```
┌─────────────────────────────────────────────────────────────┐
│                    Python API Layer                         │
│  formula.py (2606 lines) │ interactions.py (2049 lines)    │
│  diagnostics.py (4690 lines) │ target_encoding.py (615)    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  PyO3 Bindings Layer                        │
│  fitting_py.rs │ results_py.rs │ diagnostics_py.rs         │
│  splines_py.rs │ design_matrix_py.rs │ families_py.rs      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Rust Core Layer                          │
│  solvers/ │ families/ │ diagnostics/ │ splines/            │
│  links/ │ inference/ │ regularization/ │ design_matrix/    │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Component Inventory

| Component | Rust Core | PyO3 Bindings | Python Wrapper | Lines (approx) |
|-----------|-----------|---------------|----------------|----------------|
| **Solvers** | 6 files (161KB) | fitting_py.rs (484) | formula.py (2606) | ~4500 |
| **Families** | 8 files (75KB) | families_py.rs (14K) | families.py (14K) | ~2000 |
| **Diagnostics** | 10 files (170KB) | diagnostics_py.rs (25K) | diagnostics.py (4690) | ~6000 |
| **Splines** | 2 files (varies) | splines_py.rs (11K) | splines.py (574) | ~1500 |
| **Interactions** | 1 file | design_matrix_py.rs (6K) | interactions.py (2049) | ~2500 |
| **Target Encoding** | 1 file | target_encoding_py.rs (14K) | target_encoding.py (615) | ~1500 |

---

## Part 2: Process Analysis

### 2.1 Core Processes Identified

#### Process 1: Model Fitting Pipeline (Current: 7 steps)
```
1. Parse formula/dict → ParsedFormula
2. Extract columns from DataFrame
3. Build design matrix (encode categoricals, compute splines, target encode)
4. Process offset/weights
5. Call Rust fitting function
6. Handle smooth term lambda selection (if applicable)
7. Wrap result in GLMModel
```

**Complexity Issues:**
- Formula parsing happens in 3 places: Rust `parse_formula_py`, Python `parse_formula_interactions`, and inline parsing in `InteractionBuilder`
- Design matrix building has parallel code paths for formula API vs dict API
- Offset processing logic duplicated in `FormulaGLM` and `FormulaGLMDict`

#### Process 2: Prediction Pipeline (Current: 5 steps)
```
1. Validate new data has required columns
2. Re-encode categoricals using stored levels
3. Re-compute splines using stored knots
4. Apply target encoding using stored stats
5. Compute linear predictor and apply inverse link
```

**Complexity Issues:**
- `transform_new_data` in interactions.py is 200+ lines duplicating training logic
- Spline prediction requires manual knot tracking across SplineTerm instances
- TE prediction requires separate stats storage and application

#### Process 3: Diagnostics Pipeline (Current: 8 steps)
```
1. Compute residuals (Pearson, deviance, working)
2. Compute calibration metrics
3. Compute discrimination metrics (Gini, AUC, KS)
4. Compute factor-level diagnostics
5. Detect interactions
6. Compute VIF
7. Generate warnings
8. Assemble into ModelDiagnostics
```

**Complexity Issues:**
- diagnostics.py is 4690 lines - the largest file in the codebase
- Many dataclasses (30+) that could be simplified
- Duplicated metric computation for train vs test datasets

#### Process 4: Regularization Path (Current: 6 steps)
```
1. Compute alpha_max from data
2. Generate alpha path
3. Create CV folds
4. For each fold: fit models across alpha path
5. Aggregate CV deviances
6. Select optimal alpha (min or 1se)
```

**Complexity Issues:**
- Parallel implementation exists in both Python and Rust
- Python fallback path duplicates logic
- Warm starting logic scattered across functions

---

## Part 3: Identified Simplification Opportunities

### 3.1 HIGH PRIORITY - Structural Simplifications

#### S1: Unify Formula Parsing (5 steps → 2 steps)
**Current State:**
- Rust `parse_formula_py` returns a dict
- Python `parse_formula_interactions` converts to dataclasses
- `InteractionBuilder` re-parses some terms inline
- Dict API has separate parsing in `FormulaGLMDict.build_dict_design_matrix`
- Spline/TE term parsing duplicated in multiple places

**Proposed State:**
- Single Rust parser returns complete structured result
- Python layer only converts types, no re-parsing
- Dict API uses same parser with formula synthesis

**Effort**: 1 week

#### S2: Consolidate Design Matrix Building (4 parallel paths → 1 unified path)
**Current State:**
- `InteractionBuilder.build_design_matrix` for formula API
- `InteractionBuilder.build_dict_design_matrix` for dict API  
- `_DeserializedBuilder` for prediction
- `transform_new_data` for new data transformation

**Proposed State:**
- Single `DesignMatrixBuilder` class with modes: `fit`, `transform`
- Dict API synthesizes formula before building
- Prediction reuses same builder with stored state

**Effort**: 2 weeks

#### S3: Split diagnostics.py (1 file → 5 focused modules)
**Current State:**
- 4690 lines in single file
- 30+ dataclasses mixed with computation logic
- Train/test diagnostic computation duplicated

**Proposed State:**
```
diagnostics/
  __init__.py          # Public exports
  types.py             # All dataclasses (~400 lines)
  calibration.py       # Calibration metrics (~300 lines)
  discrimination.py    # Gini, AUC, lift (~300 lines)
  factors.py           # Factor-level diagnostics (~500 lines)
  computer.py          # DiagnosticsComputer class (~800 lines)
  exploration.py       # Pre-fit data exploration (~400 lines)
```

**Effort**: 1 week

### 3.2 MEDIUM PRIORITY - Process Simplifications

#### S4: Simplify Smooth GLM Fitting (3 entry points → 1)
**Current State:**
- `fit_smooth_glm_unified_py` in fitting_py.rs
- `_fit_with_smooth_penalties` in formula.py
- `fit_smooth_glm_full_matrix` in smooth_glm.rs

**Proposed State:**
- Single `fit_smooth_glm` entry point that handles all cases
- Monotonicity, multi-term, and simple cases use same code path
- GCV optimizer invoked conditionally

**Effort**: 1 week

#### S5: Unify Result Classes (3 result types → 1)
**Current State:**
- `PyGLMResults` (Rust)
- `GLMModel` (Python wrapper)
- `_DeserializedResult` (for loaded models)

**Proposed State:**
- Single `GLMResult` that works for all cases
- Serialization/deserialization as methods on same class
- Lazy computation of expensive properties

**Effort**: 1 week

#### S6: Consolidate Encoding Logic
**Current State:**
- Categorical encoding in `InteractionBuilder._get_categorical_encoding`
- TE encoding in `InteractionBuilder._encode_target_encoding_term`
- FE encoding in `InteractionBuilder._encode_frequency_encoding_term`
- Separate caches for each type

**Proposed State:**
- `EncodingManager` class managing all encodings
- Unified cache with type-tagged entries
- Single `encode(column, encoding_type, **kwargs)` method

**Effort**: 1 week

### 3.3 LOWER PRIORITY - Code Quality Improvements

#### S7: Extract Constants and Configuration
**Current State:**
- Magic numbers scattered (1e-10, 1e-8, 25, 10, etc.)
- Config objects defined in multiple places

**Proposed State:**
- `constants.py` for Python constants
- All config in `IRLSConfig`, `SmoothGLMConfig`, etc.
- Clear defaults documented in one place

**Effort**: 3 days

#### S8: Standardize Error Handling
**Current State:**
- Mix of `ValueError`, custom messages, and Rust errors
- Error context sometimes lost across boundary

**Proposed State:**
- `RustyStatsError` hierarchy in Python matching Rust
- Consistent error messages with actionable suggestions
- Error codes for programmatic handling

**Effort**: 3 days

#### S9: Improve Type Hints
**Current State:**
- Many `Any` types and missing hints
- Inconsistent use of `TYPE_CHECKING` imports

**Proposed State:**
- Full type coverage with mypy compliance
- Protocol classes for duck-typed interfaces
- Generic types where appropriate

**Effort**: 1 week

---

## Part 4: Proposed New Architecture

### 4.1 Simplified Layer Structure

```
┌─────────────────────────────────────────────────────────────┐
│                    Public Python API                         │
│  glm() │ glm_dict() │ GLMResult │ diagnostics()             │
│  TargetEncoder │ bs() │ ns()                                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Internal Python Layer                       │
│  DesignMatrixBuilder │ EncodingManager │ DiagnosticsComputer│
│  RegularizationPath │ SmoothFitter                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Rust Core (via PyO3)                       │
│  fit_glm │ parse_formula │ compute_diagnostics              │
│  encode_categorical │ target_encode │ spline_basis          │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 New Module Organization

```
python/rustystats/
├── __init__.py              # Public API exports
├── api.py                   # glm(), glm_dict() entry points
├── result.py                # GLMResult class (unified)
├── builder/
│   ├── __init__.py
│   ├── matrix.py            # DesignMatrixBuilder
│   ├── encoding.py          # EncodingManager
│   └── splines.py           # SplineManager
├── fitting/
│   ├── __init__.py
│   ├── standard.py          # Standard GLM fitting
│   ├── smooth.py            # Smooth/GAM fitting
│   └── regularized.py       # Regularization path
├── diagnostics/
│   ├── __init__.py
│   ├── types.py             # Dataclasses
│   ├── calibration.py
│   ├── discrimination.py
│   ├── factors.py
│   └── computer.py
├── families.py              # Family/link definitions
└── _rustystats.cpython*.so  # Rust extension
```

---

## Part 5: Implementation Roadmap

### Week 1-2: Foundation Work
- [x] **S3**: Split diagnostics.py into modules
- [x] **S7**: Extract constants and configuration
- [x] **S8**: Standardize error handling (exception hierarchy fully wired — 0 remaining ValueError/RuntimeError across all 11 modules; 5 tests updated)

### Week 3-4: Core Simplifications
- [x] **S1**: Unify formula parsing
- [x] **S4**: Simplify smooth GLM fitting (already unified — 3 Rust entry points collapsed to `fit_smooth_glm_full_matrix`)
- [x] **S5**: Unify result classes (refactored _DeserializedResult to dataclass, added _is_deserialized flag, removed dead code)

### Week 5-6: Major Refactoring
- [x] **S2**: Consolidate design matrix building (4 paths already collapsed to 2; extracted shared `_stack_columns`, decomposed predict-time interactions)
- [x] **S6**: Consolidate encoding logic (deduplicated `_build_target_encoding_columns` — 4 branches → 2 with shared stats dict and 3+ way loop)

### Week 7-8: Polish and Testing
- [x] **S9**: Improve type hints (all 16 missing return types fixed across formula.py, families.py, links.py, diagnostics; untyped params in _build_results and compute_diagnostics annotated)
- [ ] Update all tests for new structure
- [ ] Update documentation
- [ ] Performance benchmarking

---

## Part 6: Specific Code Issues

### 6.1 Files Needing Attention

| File | Lines | Issue | Action |
|------|-------|-------|--------|
| `diagnostics.py` | 4690 | Too large, mixed concerns | Split into 5 modules |
| `formula.py` | 2606 | Multiple classes, complex | Extract GLMModel, simplify |
| `interactions.py` | 2049 | Duplicated transform logic | Unify with builder pattern |
| `fitting_py.rs` | 484 | Multiple similar functions | Consolidate entry points |

### 6.2 Duplication Hotspots

1. **Offset Processing** (3 locations):
   - `FormulaGLM._process_offset`
   - `FormulaGLMDict._process_offset`
   - `_GLMBase._process_offset` (shared but still duplicated logic)

2. **Spline Term Parsing** (4 locations):
   - `parse_formula_interactions`
   - `parse_spline_factor`
   - Inline in `InteractionBuilder`
   - Rust `parse_formula_py`

3. **CV Fold Creation** (2 locations):
   - `regularization_path.create_cv_folds`
   - `fitting_py.fit_cv_path_py` (inline)

4. **Family/Link Resolution** (3 locations):
   - `families_py.family_from_name`
   - `formula.get_default_link`
   - Inline in various Python functions

### 6.3 Abstraction Level Inconsistencies

1. **SplineTerm** does too much:
   - Stores specification
   - Computes basis matrix
   - Tracks fitted knots
   - Computes penalty matrix
   
   **Better**: Separate `SplineSpec` (config) from `FittedSpline` (results)

2. **InteractionBuilder** mixes concerns:
   - DataFrame access
   - Column caching
   - Encoding computation
   - Interaction building
   - Prediction transformation
   
   **Better**: `DataExtractor`, `EncodingCache`, `InteractionComputer`, `Transformer`

3. **DiagnosticsComputer** has too many responsibilities:
   - Residual computation
   - Calibration metrics
   - Discrimination metrics
   - Factor analysis
   - Interaction detection
   - Warning generation
   
   **Better**: Focused computer classes composed together

---

## Part 7: Quick Wins (Can Do This Week)

### QW1: Add `__all__` exports to submodules
Many modules lack explicit exports, making the public API unclear.

### QW2: Remove dead code in interactions.py
Lines 855-2049 contain some unused helper functions and legacy code paths.

### QW3: Consolidate DEFAULT constants
Create single source of truth for:
- `DEFAULT_NEGBINOMIAL_THETA`
- Default IRLS settings
- Default regularization grid

### QW4: Fix inconsistent naming
- `var_power` vs `p` (Tweedie)
- `l1_ratio` vs `alpha_ratio`
- `cv` vs `n_folds`

### QW5: Add missing docstrings
Several Rust functions exposed to Python lack docstrings.

---

## Part 8: Metrics for Success

### Before Refactoring
- Largest Python file: 4690 lines (diagnostics.py)
- Formula parsing locations: 4
- Design matrix build paths: 4
- Result class types: 3
- Duplicated offset processing: 3

### After Refactoring (Target)
- Largest Python file: <1000 lines
- Formula parsing locations: 1
- Design matrix build paths: 1
- Result class types: 1
- Duplicated offset processing: 0

---

## Appendix A: File-by-File Notes

### Python Layer

**`formula.py`** (2606 lines)
- Contains `FormulaGLM`, `FormulaGLMDict`, `GLMModel`, `_GLMBase`
- `_fit_glm_core` and `_build_results` are good abstractions
- `GLMModel` has good delegation pattern but too many direct attributes

**`interactions.py`** (2049 lines)
- `InteractionBuilder` is the workhorse but overloaded
- Good use of Rust calls for heavy computation
- `transform_new_data` duplicates too much of `build_design_matrix`

**`diagnostics.py`** (4690 lines)
- 30+ dataclasses in one file
- Good separation of concerns with internal `_*Computer` classes
- Should be split into package

**`splines.py`** (574 lines)
- Clean, focused module
- Good Rust integration
- `SplineTerm` class could be simplified

**`target_encoding.py`** (615 lines)
- Well-organized with clear API
- Sklearn-style `TargetEncoder` is good pattern
- Could add validation for edge cases

### Rust Layer

**`solvers/irls.rs`** (1427 lines)
- Core IRLS well-implemented
- Good documentation
- Step-halving and constraints add complexity

**`solvers/smooth_glm.rs`** (1147 lines)
- Complex but necessary for GAMs
- `SmoothTermSpec` pattern is good
- Could use more helper functions

**`families/mod.rs`** (221 lines)
- Clean trait definition
- Good documentation
- Extension methods like `use_true_hessian_weights` well-designed

---

## Appendix B: Test Coverage Notes

17 test files covering major functionality:
- `test_dict_api.py` (61K) - comprehensive dict API tests
- `test_diagnostics.py` (42K) - good diagnostic coverage
- `test_frequency_encoding.py` (32K) - FE edge cases
- `test_interactions.py` (26K) - interaction building
- `test_smooth_fitting.py` (23K) - GAM fitting

**Gaps identified:**
- Limited error path testing
- No benchmark tests in test suite
- Serialization edge cases
- Multi-threading safety tests

---

## Next Steps

1. Review this document with team
2. Prioritize based on current pain points
3. Create feature branches for each S# item
4. Implement with test-first approach
5. Measure metrics after each phase

