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
>>> # Fit a Poisson GLM using the formula API
>>> result = rs.glm(
...     formula="ClaimNb ~ VehPower + VehAge + C(Area) + C(Region)",
...     data=data,
...     family="poisson",
...     offset="Exposure"
... ).fit()
>>>
>>> print(result.summary())
>>> print(result.coef_table())

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

Formula Syntax
--------------
- Main effects: ``x1``, ``x2``, ``C(cat)`` (categorical)
- Interactions: ``x1*x2`` (main + interaction), ``x1:x2`` (interaction only)
- Splines: ``bs(x, df=5)``, ``ns(x, df=4)``, ``bs(x, df=5, monotonicity='increasing')``
- Target encoding: ``TE(brand)`` for high-cardinality categoricals

For Actuaries
-------------
- **Claim Frequency**: Use Poisson family with log link
- **Claim Severity**: Use Gamma family with log link  
- **Claim Occurrence**: Use Binomial family with logit link
- **Pure Premium**: Use Tweedie family with var_power=1.5
"""

# Version of the package (must match pyproject.toml)
__version__ = "0.3.8"

# Import the Rust extension module
# This contains the fast implementations
from rustystats._rustystats import (
    # Link functions
    IdentityLink,
    LogLink,
    LogitLink,
    # Families
    GaussianFamily,
    PoissonFamily,
    BinomialFamily,
    GammaFamily,
    TweedieFamily,
    # GLM results type
    GLMResults,
    # Spline functions (raw Rust)
    bs_py as _bs_rust,
    ns_py as _ns_rust,
)

# Import Python wrappers
from rustystats import families
from rustystats import links
from rustystats.glm import summary, summary_relativities

# Formula-based API (the primary API)
from rustystats.formula import glm, FormulaGLM, GLMModel

# Dict-based API (alternative to formula strings)
from rustystats.formula import glm_dict, FormulaGLMDict

# Spline basis functions (for non-linear continuous effects)
from rustystats.splines import bs, ns, bs_names, ns_names, SplineTerm

# Penalized spline utilities (for GAMs with automatic smoothness selection)
from rustystats.smooth import penalty_matrix, difference_matrix, gcv_score, compute_edf

# Target encoding (CatBoost-style ordered target statistics)
from rustystats.target_encoding import (
    target_encode,
    apply_target_encoding,
    TargetEncoder,
    TargetEncodingTerm,
    # Frequency encoding (CatBoost Counter CTR)
    frequency_encode,
    apply_frequency_encoding,
    FrequencyEncoder,
    # Target encoding for interactions
    target_encode_interaction,
)

# Model diagnostics
from rustystats.diagnostics import (
    compute_diagnostics,
    ModelDiagnostics,
    DiagnosticsComputer,
    explore_data,
    DataExploration,
    DataExplorer,
)

# Exceptions
from rustystats.exceptions import (
    RustyStatsError,
    DesignMatrixError,
    FittingError,
    ConvergenceError,
    PredictionError,
    EncodingError,
    FormulaError,
    ValidationError,
    SerializationError,
)

# What gets exported when someone does `from rustystats import *`
__all__ = [
    # Version
    "__version__",
    # Formula-based API (primary interface)
    "glm",
    "FormulaGLM",
    "GLMModel",
    "GLMResults",
    # Dict-based API
    "glm_dict",
    "FormulaGLMDict",
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
    "TargetEncodingTerm",
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
    # Direct access to classes (for convenience)
    "IdentityLink",
    "LogLink",
    "LogitLink",
    "GaussianFamily",
    "PoissonFamily",
    "BinomialFamily",
    "GammaFamily",
    "TweedieFamily",
    # Exceptions
    "RustyStatsError",
    "DesignMatrixError",
    "FittingError",
    "ConvergenceError",
    "PredictionError",
    "EncodingError",
    "FormulaError",
    "ValidationError",
    "SerializationError",
]
