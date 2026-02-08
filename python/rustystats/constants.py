"""
Central configuration and constants for RustyStats.

This module provides a single source of truth for all default values
and magic numbers used throughout the library.
"""

__all__ = [
    # IRLS
    "DEFAULT_MAX_ITER",
    "DEFAULT_TOLERANCE",
    # Regularization
    "DEFAULT_N_ALPHAS",
    "DEFAULT_ALPHA_MIN_RATIO",
    "DEFAULT_CV_FOLDS",
    # Splines
    "DEFAULT_SPLINE_DF",
    "DEFAULT_SPLINE_DEGREE",
    "DEFAULT_LAMBDA_MIN",
    "DEFAULT_LAMBDA_MAX",
    "DEFAULT_N_LAMBDA",
    # Target Encoding
    "DEFAULT_PRIOR_WEIGHT",
    "DEFAULT_N_PERMUTATIONS",
    # Negative Binomial
    "DEFAULT_NEGBINOMIAL_THETA",
    "DEFAULT_THETA_TOL",
    "DEFAULT_MAX_THETA_ITER",
    # Diagnostics
    "DEFAULT_N_CALIBRATION_BINS",
    "DEFAULT_N_FACTOR_BINS",
    "DEFAULT_RARE_THRESHOLD_PCT",
    "DEFAULT_MAX_CATEGORICAL_LEVELS",
    "DEFAULT_MAX_INTERACTION_FACTORS",
    # Numerical Stability
    "EPSILON",
    # Validation Thresholds
    "ZERO_VARIANCE_THRESHOLD",
    "CONDITION_NUMBER_THRESHOLD",
    "DEFAULT_CORRELATION_THRESHOLD",
    # Regularization Internals
    "L1_RATIO_MIN_CLAMP",
    "ALPHA_MAX_FLOOR",
    "DEFAULT_ELASTIC_NET_L1_RATIO",
    "DEFAULT_CV_SEED",
    # Links and Aliases
    "DEFAULT_LINKS",
    "NEGBINOMIAL_ALIASES",
]

# =============================================================================
# IRLS Algorithm Defaults
# =============================================================================
DEFAULT_MAX_ITER = 25
DEFAULT_TOLERANCE = 1e-8

# =============================================================================
# Regularization Path Defaults
# =============================================================================
DEFAULT_N_ALPHAS = 20
DEFAULT_ALPHA_MIN_RATIO = 0.0001
DEFAULT_CV_FOLDS = 5

# =============================================================================
# Spline Defaults
# =============================================================================
DEFAULT_SPLINE_DF = 10  # For penalized smooth (s() syntax)
DEFAULT_SPLINE_DEGREE = 3
DEFAULT_LAMBDA_MIN = 1e-1
DEFAULT_LAMBDA_MAX = 1e3
DEFAULT_N_LAMBDA = 6

# =============================================================================
# Target Encoding Defaults
# =============================================================================
DEFAULT_PRIOR_WEIGHT = 1.0
DEFAULT_N_PERMUTATIONS = 4

# =============================================================================
# Negative Binomial
# =============================================================================
DEFAULT_NEGBINOMIAL_THETA = 1.0
DEFAULT_THETA_TOL = 1e-5
DEFAULT_MAX_THETA_ITER = 10

# =============================================================================
# Diagnostics
# =============================================================================
DEFAULT_N_CALIBRATION_BINS = 10
DEFAULT_N_FACTOR_BINS = 10
DEFAULT_RARE_THRESHOLD_PCT = 1.0
DEFAULT_MAX_CATEGORICAL_LEVELS = 20
DEFAULT_MAX_INTERACTION_FACTORS = 10

# =============================================================================
# Numerical Stability
# =============================================================================
EPSILON = 1e-10

# =============================================================================
# Validation Thresholds
# =============================================================================
ZERO_VARIANCE_THRESHOLD = 1e-10
CONDITION_NUMBER_THRESHOLD = 1e10
DEFAULT_CORRELATION_THRESHOLD = 0.999

# =============================================================================
# Regularization Internals
# =============================================================================
L1_RATIO_MIN_CLAMP = 1e-3     # Floor for l1_ratio in alpha_max computation
ALPHA_MAX_FLOOR = 1e-4        # Minimum alpha_max to avoid degenerate paths
DEFAULT_ELASTIC_NET_L1_RATIO = 0.5
DEFAULT_CV_SEED = 42

# =============================================================================
# Canonical Default Links
# =============================================================================
DEFAULT_LINKS = {
    "gaussian": "identity",
    "poisson": "log",
    "quasipoisson": "log",
    "binomial": "logit",
    "gamma": "log",
    "tweedie": "log",
    "negbinomial": "log",
    "negativebinomial": "log",
}

# =============================================================================
# Family Aliases
# =============================================================================
NEGBINOMIAL_ALIASES = frozenset({
    "negbinomial",
    "negativebinomial", 
    "negative_binomial",
    "nb",
})
