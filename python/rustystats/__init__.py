"""
RustyStats: Fast Generalized Linear Models with a Rust Backend
==============================================================

A statsmodels-compatible GLM library optimized for actuarial applications.

Quick Start
-----------
>>> import rustystats as rs
>>> import numpy as np
>>>
>>> # Create some example data
>>> np.random.seed(42)
>>> n = 100
>>> X = np.column_stack([np.ones(n), np.random.randn(n)])
>>> y = np.random.poisson(lam=np.exp(0.5 + 0.3 * X[:, 1]))
>>>
>>> # Fit a Poisson GLM
>>> result = rs.fit_glm(y, X, family="poisson")
>>> print(f"Coefficients: {result.params}")
>>> print(f"Converged: {result.converged}")

Available Families
------------------
- **gaussian**: Continuous data, constant variance (linear regression)
- **poisson**: Count data, variance = mean (claim frequency)
- **binomial**: Binary/proportion data (logistic regression)
- **gamma**: Positive continuous, variance ∝ mean² (claim severity)
- **tweedie**: Mixed zeros and positives, variance = μ^p (pure premium)

Available Link Functions
------------------------
- **identity**: η = μ (default for Gaussian)
- **log**: η = log(μ) (default for Poisson, Gamma)
- **logit**: η = log(μ/(1-μ)) (default for Binomial)

For Actuaries
-------------
This library is designed with actuarial applications in mind:

- **Claim Frequency**: Use Poisson family with log link
- **Claim Severity**: Use Gamma family with log link  
- **Claim Occurrence**: Use Binomial family with logit link
- **Pure Premium**: Use Tweedie family with var_power=1.5

All code is heavily documented to help non-programmers understand
what's happening under the hood.

Example: Claim Frequency Model
------------------------------
>>> # Simulate claim data
>>> n = 1000
>>> age = np.random.uniform(18, 70, n)
>>> exposure = np.random.uniform(0.5, 1.0, n)  # Policy years
>>> 
>>> # True model: log(claims) = -2 + 0.02 * age + log(exposure)
>>> true_rate = np.exp(-2 + 0.02 * age) * exposure
>>> claims = np.random.poisson(true_rate)
>>>
>>> # Fit model (with exposure as offset)
>>> X = np.column_stack([np.ones(n), age])
>>> result = rs.fit_glm(claims, X, family="poisson")
>>> print(f"Intercept: {result.params[0]:.3f}")  # Should be ≈ -2
>>> print(f"Age effect: {result.params[1]:.4f}")  # Should be ≈ 0.02
"""

# Version of the package
__version__ = "0.1.0"

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
    # GLM fitting
    GLMResults,
    fit_glm_py as _fit_glm_rust,
)

# Import Python wrappers (these provide the nice API)
from rustystats import families
from rustystats import links
from rustystats import glm as glm_module  # Import as module for backward compat
from rustystats.glm import GLM, fit_glm, summary, summary_relativities, predict

# Formula-based API (works with DataFrames)
from rustystats.formula import glm, FormulaGLM, FormulaGLMResults

# Variable selection utilities
from rustystats.selection import (
    regularization_path,
    lasso_path,
    cv_glm,
    cv_lasso,
    cv_ridge,
    cv_elasticnet,
    RegularizationPath,
    CVResult,
)

# What gets exported when someone does `from rustystats import *`
__all__ = [
    # Version
    "__version__",
    # Main API
    "GLM",
    "fit_glm",
    "predict",
    "glm",  # Formula-based API
    "glm_module",  # The glm module (for rs.glm_module.predict compatibility)
    "GLMResults",
    "FormulaGLM",
    "FormulaGLMResults",
    "summary",
    "summary_relativities",
    # Sub-modules
    "families",
    "links",
    # Variable selection
    "regularization_path",
    "lasso_path",
    "cv_glm",
    "cv_lasso",
    "cv_ridge",
    "cv_elasticnet",
    "RegularizationPath",
    "CVResult",
    # Direct access to classes (for convenience)
    "IdentityLink",
    "LogLink",
    "LogitLink",
    "GaussianFamily",
    "PoissonFamily",
    "BinomialFamily",
    "GammaFamily",
    "TweedieFamily",
]
