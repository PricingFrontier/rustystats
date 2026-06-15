"""
RustyStats: Fast Generalized Linear Models with a Rust Backend
==============================================================

A high-performance GLM library optimized for actuarial applications.

Quick Start
-----------
>>> import rustystats as rs
>>> import polars as pl
>>>
>>> # Load data
>>> data = pl.read_parquet("insurance.parquet")
>>>
>>> # Fit a Poisson GLM using the dict API
>>> result = rs.glm_dict(
...     response="ClaimNb",
...     terms={
...         "VehPower": {"type": "linear"},
...         "VehAge": {"type": "linear"},
...         "Area": {"type": "categorical"},
...         "Region": {"type": "categorical"},
...     },
...     data=data,
...     family="poisson",
...     exposure="Exposure",
... ).fit()
>>>
>>> print(result.summary())

Available Families
------------------
- **gaussian**: Continuous data, constant variance (linear regression)
- **poisson**: Count data, variance = mean (claim frequency)
- **binomial**: Binary/proportion data (logistic regression)
- **gamma**: Positive continuous, variance ∝ mean² (claim severity)
- **tweedie**: Mixed zeros and positives, variance = μ^p (pure premium)
- **quasipoisson**: Overdispersed count data
- **quasibinomial**: Overdispersed binary data
- **negbinomial**: Overdispersed counts with auto θ estimation

Available Link Functions
------------------------
- **identity**: η = μ (default for Gaussian)
- **log**: η = log(μ) (default for Poisson, Gamma)
- **logit**: η = log(μ/(1-μ)) (default for Binomial)

For Actuaries
-------------
- **Claim Frequency**: Use Poisson family with log link
- **Claim Severity**: Use Gamma family with log link
- **Claim Occurrence**: Use Binomial family with logit link
- **Pure Premium**: Use Tweedie family with var_power=1.5
"""

# Version of the package (read from installed package metadata)
try:
    from importlib.metadata import version

    __version__ = version("rustystats")
except Exception:
    __version__ = "0.0.0"

# Import the Rust extension module
# This contains the fast implementations
# Import Python wrappers
from rustystats import families, links
from rustystats._rustystats import (
    BinomialFamily,
    GammaFamily,
    # Families
    GaussianFamily,
    # GLM results type
    GLMResults,
    # Link functions
    IdentityLink,
    LogitLink,
    LogLink,
    PoissonFamily,
    TweedieFamily,
)
from rustystats._rustystats import (
    # Spline functions (raw Rust)
    bs_py as _bs_rust,  # noqa: F401
)
from rustystats._rustystats import (
    ns_py as _ns_rust,  # noqa: F401
)

# Calibration primitives (RS-ACT-009 / PR11)
from rustystats.calibration import (
    GlobalCalibration,
    IsotonicCalibration,
    calibration_summary,
    fit_global_calibration,
    fit_isotonic_calibration,
)

# Model diagnostics
from rustystats.diagnostics import (
    DataExploration,
    DataExplorer,
    DiagnosticsComputer,
    ModelDiagnostics,
    compute_diagnostics,
    explore_data,
)

# Exceptions
from rustystats.exceptions import (
    ConvergenceError,
    DesignMatrixError,
    EncodingError,
    FittingError,
    PredictionError,
    RustyStatsError,
    SerializationError,
    ValidationError,
)
from rustystats.export_onnx import to_onnx

# Model export (PMML / ONNX)
from rustystats.export_pmml import to_pmml

# Dict-based API (the primary API)
from rustystats.formula import FormulaGLMDict, GLMModel, glm_dict
from rustystats.glm import summary, summary_relativities

# Deterministic input transforms (frozen lookups applied before the design matrix)
from rustystats.input_transforms import (
    apply_input_transforms,
    compile_input_transforms,
    validate_input_transforms,
)
from rustystats.multinomial import (
    MultinomialDiagnostics,
    MultinomialDict,
    MultinomialModel,
    multinomial_dict,
)

# Rate-table export (resolved deployment tables), sibling to to_pmml / to_onnx
from rustystats.rate_tables import to_rate_tables

# IRLS residual / working-response helper (e.g. for link-scale boosting loops)
from rustystats.residuals import working_response_weights

# Penalized spline utilities (for GAMs with automatic smoothness selection)
from rustystats.smooth import compute_edf, difference_matrix, gcv_score, penalty_matrix

# Spline basis functions (for non-linear continuous effects)
from rustystats.splines import SplineTerm, bs, bs_names, ns, ns_names

# Target encoding (CatBoost-style ordered target statistics)
from rustystats.target_encoding import (
    FrequencyEncoder,
    TargetEncoder,
    apply_frequency_encoding,
    apply_target_encoding,
    # Frequency encoding (CatBoost Counter CTR)
    frequency_encode,
    target_encode,
    # Target encoding for interactions
    target_encode_interaction,
)

# Input validation (for advanced users who want to pre-validate)
from rustystats.validation import validate_glm_inputs, validate_residual_inputs

# What gets exported when someone does `from rustystats import *`
__all__ = [
    # Version
    "__version__",
    # Dict-based API (primary interface)
    "GLMModel",
    "GLMResults",
    "glm_dict",
    "FormulaGLMDict",
    "MultinomialDiagnostics",
    "MultinomialDict",
    "MultinomialModel",
    "multinomial_dict",
    "summary",
    "summary_relativities",
    # Spline functions
    "bs",
    "ns",
    "bs_names",
    "ns_names",
    "SplineTerm",
    # Penalized spline utilities (GAMs)
    "penalty_matrix",
    "difference_matrix",
    "gcv_score",
    "compute_edf",
    # Target encoding (CatBoost-style)
    "target_encode",
    "apply_target_encoding",
    "TargetEncoder",
    # Frequency encoding
    "frequency_encode",
    "apply_frequency_encoding",
    "FrequencyEncoder",
    # Target encoding for interactions
    "target_encode_interaction",
    # Sub-modules
    "families",
    "links",
    # Model diagnostics
    "compute_diagnostics",
    "ModelDiagnostics",
    "DiagnosticsComputer",
    "explore_data",
    "DataExploration",
    "DataExplorer",
    # Calibration primitives (RS-ACT-009)
    "calibration_summary",
    "fit_global_calibration",
    "fit_isotonic_calibration",
    "GlobalCalibration",
    "IsotonicCalibration",
    # Direct access to classes (for convenience)
    "IdentityLink",
    "LogLink",
    "LogitLink",
    "GaussianFamily",
    "PoissonFamily",
    "BinomialFamily",
    "GammaFamily",
    "TweedieFamily",
    # Model export
    "to_pmml",
    "to_onnx",
    "to_rate_tables",
    # Deterministic input transforms
    "validate_input_transforms",
    "compile_input_transforms",
    "apply_input_transforms",
    # Exceptions
    "RustyStatsError",
    "DesignMatrixError",
    "FittingError",
    "ConvergenceError",
    "PredictionError",
    "EncodingError",
    "ValidationError",
    "SerializationError",
    # Input validation
    "validate_glm_inputs",
    "validate_residual_inputs",
    # IRLS residual helper
    "working_response_weights",
]
