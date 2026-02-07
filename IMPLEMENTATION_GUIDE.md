# RustyStats Implementation Guide

This document provides detailed implementation instructions for each improvement item identified in ARCHITECTURE_REVIEW.md.

---

## Week 1-2: Foundation Work

### S3: Split diagnostics.py into Modules

**Goal**: Break 4690-line file into 5 focused modules

#### Step 1: Create diagnostics package structure
```bash
mkdir -p python/rustystats/diagnostics
touch python/rustystats/diagnostics/__init__.py
```

#### Step 2: Extract dataclasses to types.py

Move all `@dataclass` definitions to `diagnostics/types.py`:
- `Percentiles`, `ResidualSummary`, `CalibrationBin`
- `LorenzPoint`, `ActualExpectedBin`, `ResidualPattern`
- `ContinuousFactorStats`, `CategoricalLevelStats`, `CategoricalFactorStats`
- `FactorSignificance`, `ScoreTestResult`, `FactorCoefficient`
- `FactorDiagnostics`, `InteractionCandidate`, `VIFResult`
- `CoefficientSummary`, `DevianceByLevel`, `FactorDeviance`
- `LiftDecile`, `LiftChart`, `PartialDependence`, `DecileMetrics`
- `FactorLevelMetrics`, `ContinuousBandMetrics`, `DatasetDiagnostics`
- `TrainTestComparison`, `ConvergenceDetails`, `DataExploration`
- `SmoothTermDiagnostics`, `ModelVsBaseDecile`, `BasePredictionsMetrics`
- `BasePredictionsComparison`, `ModelDiagnostics`

Also move helper functions:
- `_json_default`
- `_round_float`
- `_to_dict_recursive`
- `_extract_base_variable`

#### Step 3: Extract calibration.py

Move from diagnostics.py:
- `_CalibrationComputer` class
- Any calibration-specific helpers

#### Step 4: Extract discrimination.py

Move from diagnostics.py:
- `_DiscriminationComputer` class
- Lift chart computation logic

#### Step 5: Extract factors.py

Move factor-level diagnostic code:
- `_compute_factor_diagnostics` (if exists)
- Score test computation
- A/E by factor level logic

#### Step 6: Keep computer.py

Retain in main computer module:
- `_ResidualComputer`
- `DiagnosticsComputer`
- `compute_diagnostics` function
- `explore_data` function and `DataExplorer` class

#### Step 7: Update __init__.py exports

```python
# python/rustystats/diagnostics/__init__.py
from .types import (
    ModelDiagnostics,
    DataExploration,
    FactorDiagnostics,
    # ... all public types
)
from .computer import (
    DiagnosticsComputer,
    compute_diagnostics,
    explore_data,
    DataExplorer,
)
```

#### Step 8: Update imports in formula.py

```python
# Before
from rustystats.diagnostics import compute_diagnostics, ModelDiagnostics

# After (same - __init__.py re-exports)
from rustystats.diagnostics import compute_diagnostics, ModelDiagnostics
```

#### Verification
- Run `pytest tests/python/test_diagnostics.py`
- Ensure all exports work from `rustystats.diagnostics`

---

### S7: Extract Constants and Configuration

**Goal**: Single source of truth for magic numbers

#### Step 1: Create constants.py

```python
# python/rustystats/constants.py
"""
Central configuration and constants for RustyStats.
"""

# IRLS Algorithm Defaults
DEFAULT_MAX_ITER = 25
DEFAULT_TOLERANCE = 1e-8
DEFAULT_MIN_WEIGHT = 1e-10

# Regularization Defaults
DEFAULT_N_ALPHAS = 20
DEFAULT_ALPHA_MIN_RATIO = 0.0001
DEFAULT_CV_FOLDS = 5

# Spline Defaults
DEFAULT_SPLINE_DF = 10  # For penalized smooth
DEFAULT_SPLINE_DEGREE = 3
DEFAULT_LAMBDA_MIN = 1e-4
DEFAULT_LAMBDA_MAX = 1e6

# Target Encoding Defaults
DEFAULT_PRIOR_WEIGHT = 1.0
DEFAULT_N_PERMUTATIONS = 4

# Negative Binomial
DEFAULT_NEGBINOMIAL_THETA = 1.0

# Diagnostics
DEFAULT_N_CALIBRATION_BINS = 10
DEFAULT_RARE_THRESHOLD_PCT = 1.0
DEFAULT_MAX_CATEGORICAL_LEVELS = 20

# Numerical Stability
EPSILON = 1e-10
MU_MIN_POISSON = 1e-10
MU_MIN_GAMMA = 1e-10
MU_BOUNDS_BINOMIAL = (1e-10, 1 - 1e-10)
```

#### Step 2: Update imports across codebase

Replace magic numbers with constant imports:
```python
# Before
max_iter: int = 25

# After
from rustystats.constants import DEFAULT_MAX_ITER
max_iter: int = DEFAULT_MAX_ITER
```

---

### S8: Standardize Error Handling

**Goal**: Consistent error messages with actionable suggestions

#### Step 1: Create exceptions.py

```python
# python/rustystats/exceptions.py
"""
Custom exceptions for RustyStats with actionable error messages.
"""

class RustyStatsError(Exception):
    """Base exception for all RustyStats errors."""
    pass

class DesignMatrixError(RustyStatsError):
    """Error in design matrix construction or validation."""
    pass

class FittingError(RustyStatsError):
    """Error during model fitting."""
    pass

class PredictionError(RustyStatsError):
    """Error during prediction on new data."""
    pass

class EncodingError(RustyStatsError):
    """Error in categorical or target encoding."""
    pass

class FormulaError(RustyStatsError):
    """Error parsing formula string."""
    pass
```

#### Step 2: Wrap Rust errors with context

```python
# In fitting code
try:
    result = _fit_glm_rust(...)
except ValueError as e:
    if "singular" in str(e).lower():
        raise FittingError(
            f"Design matrix is singular (rank deficient). "
            f"This usually means:\n"
            f"  1. Perfect multicollinearity between predictors\n"
            f"  2. A categorical variable with only one level\n"
            f"  3. Duplicate columns in the design matrix\n"
            f"Run model.validate() to identify the specific issue.\n"
            f"Original error: {e}"
        ) from None
    raise
```

---

## Week 3-4: Core Simplifications

### S1: Unify Formula Parsing

**Goal**: Single Rust parser, Python only converts types

#### Current Flow (5 steps):
1. `parse_formula_py` (Rust) → dict
2. `parse_formula_interactions` (Python) → dataclasses
3. `InteractionBuilder` re-parses spline terms inline
4. Dict API has separate term parsing
5. Spline/TE parsing duplicated in helpers

#### Target Flow (2 steps):
1. `parse_formula_py` (Rust) → complete structured result
2. Python converts to dataclasses (no re-parsing)

#### Implementation

**Step 1**: Enhance Rust parser to return complete info

In `crates/rustystats-core/src/formula/mod.rs`, ensure parser returns:
- All spline parameters (df, k, degree, monotonicity, knots)
- All TE parameters (prior_weight, n_permutations)
- All categorical parameters (levels if specified)
- Constraint information (pos/neg)

**Step 2**: Update Python wrapper

```python
# interactions.py - simplified parse_formula_interactions
def parse_formula_interactions(formula: str) -> ParsedFormula:
    """Parse formula using Rust parser, convert to Python types."""
    parsed = _parse_formula_rust(formula)
    
    # Direct conversion - no re-parsing
    return ParsedFormula(
        response=parsed['response'],
        main_effects=parsed['main_effects'],
        interactions=[InteractionTerm(**i) for i in parsed['interactions']],
        categorical_vars=set(parsed['categorical_vars']),
        spline_terms=[SplineTerm(**s) for s in parsed['spline_terms']],
        target_encoding_terms=[TargetEncodingTermSpec(**t) for t in parsed['target_encoding_terms']],
        # ... rest of fields
        has_intercept=parsed['has_intercept'],
    )
```

**Step 3**: Remove redundant parsing functions

Delete or deprecate:
- `parse_spline_factor` in interactions.py (use parsed result)
- `parse_te_factor` in interactions.py (use parsed result)
- Inline parsing in `InteractionBuilder`

---

### S4: Simplify Smooth GLM Fitting

**Goal**: Single entry point for all smooth GLM cases

#### Current State:
- `fit_smooth_glm_unified_py` in fitting_py.rs
- `_fit_with_smooth_penalties` in formula.py
- Multiple paths for monotonic vs non-monotonic

#### Target State:
Single `fit_smooth_glm` that handles all cases

#### Implementation

**Step 1**: Consolidate Python-side smooth fitting

```python
# formula.py - simplified smooth fitting
def _fit_smooth(
    y: np.ndarray,
    X: np.ndarray,
    smooth_specs: List[SmoothTermSpec],
    family: str,
    link: str,
    offset: Optional[np.ndarray],
    weights: Optional[np.ndarray],
    config: SmoothConfig,
) -> Tuple[GLMResult, List[SmoothTermResult]]:
    """
    Unified smooth GLM fitting.
    
    Handles all cases:
    - Single smooth term
    - Multiple smooth terms
    - Monotonic constraints
    - Auto lambda selection via GCV
    """
    from rustystats._rustystats import fit_smooth_glm_unified_py
    
    # Build specs for Rust
    col_ranges = [(s.col_start, s.col_end) for s in smooth_specs]
    penalties = [s.penalty_matrix for s in smooth_specs]
    monotonicity = [s.monotonicity for s in smooth_specs]
    
    result, meta = fit_smooth_glm_unified_py(
        y, X, col_ranges, penalties,
        family, link, offset, weights,
        config.max_iter, config.tol,
        config.lambda_min, config.lambda_max,
        monotonicity,
    )
    
    smooth_results = [
        SmoothTermResult(
            variable=smooth_specs[i].variable,
            k=smooth_specs[i].k,
            edf=meta['smooth_edfs'][i],
            lambda_=meta['lambdas'][i],
            gcv=meta['gcv'],
        )
        for i in range(len(smooth_specs))
    ]
    
    return result, smooth_results
```

**Step 2**: Remove `_fit_with_smooth_penalties` 

Replace calls with unified `_fit_smooth`.

---

### S5: Unify Result Classes

**Goal**: Single `GLMResult` class for all use cases

#### Current State:
- `PyGLMResults` (Rust-side, exposed to Python)
- `GLMModel` (Python wrapper with formula context)
- `_DeserializedResult` (for loaded models)

#### Target State:
- `GLMResult` that wraps Rust result and handles all cases

#### Implementation

**Step 1**: Merge `_DeserializedResult` into `GLMModel`

```python
class GLMModel:
    """Unified GLM result class."""
    
    def __init__(
        self,
        result=None,  # PyGLMResults or None for deserialized
        coefficients: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        # ... other fields
        _is_deserialized: bool = False,
    ):
        if result is not None:
            self._result = result
            self._is_deserialized = False
        else:
            # Deserialized case - store coefficients directly
            self._coefficients = coefficients
            self._is_deserialized = True
        
        self.feature_names = feature_names
        # ... rest of init
    
    @property
    def params(self) -> np.ndarray:
        if self._is_deserialized:
            return self._coefficients
        return self._result.params
    
    @property
    def fittedvalues(self) -> np.ndarray:
        if self._is_deserialized:
            raise AttributeError(
                "fittedvalues not available on loaded models. "
                "Use predict() on new data instead."
            )
        return self._result.fittedvalues
```

**Step 2**: Update serialization to use unified class

```python
@classmethod
def from_bytes(cls, data: bytes) -> "GLMModel":
    """Load model from bytes."""
    state = _deserialize(data)
    return cls(
        result=None,
        coefficients=state['coefficients'],
        feature_names=state['feature_names'],
        # ... rest of state
        _is_deserialized=True,
    )
```

---

## Week 5-6: Major Refactoring

### S2: Consolidate Design Matrix Building

**Goal**: Single `DesignMatrixBuilder` with `fit` and `transform` modes

#### Current Paths:
1. `InteractionBuilder.build_design_matrix` (formula)
2. `InteractionBuilder.build_dict_design_matrix` (dict API)
3. `InteractionBuilder.transform_new_data` (prediction)
4. `_DeserializedBuilder` (loaded models)

#### Target: Unified Builder

```python
class DesignMatrixBuilder:
    """
    Unified design matrix construction.
    
    Modes:
    - fit: Build from formula/dict, store encoding state
    - transform: Apply stored encodings to new data
    """
    
    def __init__(self, data: Optional["pl.DataFrame"] = None):
        self._data = data
        self._state = BuilderState()
    
    def fit(
        self,
        spec: Union[str, Dict[str, Any]],
        y: Optional[np.ndarray] = None,
        exposure: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Build design matrix and store state for transform.
        
        Parameters
        ----------
        spec : str or dict
            Formula string or terms dict
        y : array, optional
            Response (needed for target encoding)
        exposure : array, optional
            Exposure (for TE rate computation)
        seed : int, optional
            Random seed for reproducibility
        
        Returns
        -------
        y : array
            Response vector
        X : array
            Design matrix
        names : list
            Feature names
        """
        if isinstance(spec, str):
            parsed = parse_formula(spec)
        else:
            parsed = self._parse_dict(spec)
        
        return self._build(parsed, y, exposure, seed, mode='fit')
    
    def transform(self, data: "pl.DataFrame") -> np.ndarray:
        """
        Transform new data using stored encodings.
        
        Parameters
        ----------
        data : DataFrame
            New data with same columns as training
        
        Returns
        -------
        X : array
            Design matrix for new data
        """
        if not self._state.is_fitted:
            raise ValueError("Builder not fitted. Call fit() first.")
        
        self._data = data
        _, X, _ = self._build(
            self._state.parsed_formula, 
            y=None, 
            exposure=None, 
            seed=None, 
            mode='transform'
        )
        return X
    
    def _build(
        self,
        parsed: ParsedFormula,
        y: Optional[np.ndarray],
        exposure: Optional[np.ndarray],
        seed: Optional[int],
        mode: str,
    ) -> Tuple[Optional[np.ndarray], np.ndarray, List[str]]:
        """Internal build logic for both fit and transform."""
        columns = []
        names = []
        
        # Intercept
        if parsed.has_intercept:
            columns.append(np.ones(len(self._data)))
            names.append("Intercept")
        
        # Process each term type
        for term in parsed.main_effects:
            col, name = self._process_term(term, y, exposure, seed, mode)
            columns.append(col)
            names.extend(name if isinstance(name, list) else [name])
        
        # ... interactions, etc.
        
        X = np.column_stack(columns) if columns else np.empty((len(self._data), 0))
        
        if mode == 'fit':
            self._state.parsed_formula = parsed
            self._state.is_fitted = True
            y_out = self._extract_response(parsed.response)
            return y_out, X, names
        else:
            return None, X, names
    
    def get_state(self) -> dict:
        """Get serializable state for model saving."""
        return self._state.to_dict()
    
    @classmethod
    def from_state(cls, state: dict) -> "DesignMatrixBuilder":
        """Restore builder from saved state."""
        builder = cls(data=None)
        builder._state = BuilderState.from_dict(state)
        return builder
```

---

### S6: Consolidate Encoding Logic

**Goal**: `EncodingManager` for all encoding types

```python
class EncodingManager:
    """
    Unified encoding management for categorical, TE, and FE.
    """
    
    def __init__(self):
        self._cache: Dict[str, EncodingResult] = {}
    
    def encode(
        self,
        values: np.ndarray,
        encoding_type: str,
        var_name: str,
        mode: str = 'fit',
        **kwargs,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Encode values with specified encoding type.
        
        Parameters
        ----------
        values : array
            Raw values to encode
        encoding_type : str
            'categorical', 'target', 'frequency'
        var_name : str
            Variable name for column naming
        mode : str
            'fit' (learn and encode) or 'transform' (apply learned)
        **kwargs
            Additional params (y, exposure for TE; prior_weight, etc.)
        
        Returns
        -------
        encoded : array
            Encoded values
        names : list
            Column names
        """
        cache_key = f"{var_name}_{encoding_type}"
        
        if mode == 'transform':
            if cache_key not in self._cache:
                raise ValueError(f"No encoding found for {var_name}")
            return self._apply_encoding(values, self._cache[cache_key])
        
        # Fit mode
        if encoding_type == 'categorical':
            result = self._fit_categorical(values, var_name, **kwargs)
        elif encoding_type == 'target':
            result = self._fit_target_encoding(values, var_name, **kwargs)
        elif encoding_type == 'frequency':
            result = self._fit_frequency_encoding(values, var_name, **kwargs)
        else:
            raise ValueError(f"Unknown encoding type: {encoding_type}")
        
        self._cache[cache_key] = result
        return result.encoded, result.names
    
    def _fit_categorical(self, values, var_name, drop_first=True, **kwargs):
        from rustystats._rustystats import encode_categorical_py
        encoding, names, indices, levels = encode_categorical_py(
            [str(v) for v in values], var_name, drop_first
        )
        return EncodingResult(
            encoded=encoding,
            names=names,
            levels=levels,
            indices=indices,
            encoding_type='categorical',
        )
    
    # ... similar for TE, FE
```

---

## Week 7-8: Polish

### S9: Type Hints

Add comprehensive type hints using:

```python
from typing import Protocol, TypeVar, Generic

class Fittable(Protocol):
    """Protocol for objects that can be fitted."""
    def fit(self, X: np.ndarray, y: np.ndarray) -> "Fittable": ...

class Transformable(Protocol):
    """Protocol for objects that can transform data."""
    def transform(self, X: np.ndarray) -> np.ndarray: ...

T = TypeVar('T', bound='GLMResult')

class GLMResult(Generic[T]):
    """Generic result with family-specific methods."""
    pass
```

---

## Testing Strategy

For each refactoring:

1. **Before changing**: Ensure all existing tests pass
2. **Add characterization tests**: Capture current behavior
3. **Refactor**: Make changes
4. **Verify**: All tests still pass
5. **Add new tests**: For simplified interfaces

```bash
# Run before any changes
pytest tests/python/ -v

# Run after each change
pytest tests/python/ -v --tb=short

# Run specific module tests
pytest tests/python/test_diagnostics.py -v
```

---

## Migration Notes

### For S3 (diagnostics split):
- Keep old imports working via `__init__.py` re-exports
- No API changes required for users

### For S5 (unified result):
- `GLMModel` API unchanged
- Internal `_DeserializedResult` removed
- Serialization format unchanged

### For S2 (unified builder):
- `InteractionBuilder` becomes thin wrapper
- `FormulaGLM` and `FormulaGLMDict` use same builder
- Prediction code simplified
