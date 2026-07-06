"""
Central configuration and constants for RustyStats.

This module provides a single source of truth for all default values
and magic numbers used throughout the library.
"""

__all__ = [
    # IRLS
    "DEFAULT_MAX_ITER",
    "DEFAULT_MONOTONE_SMOOTH_MAX_ITER",
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
    # Diagnostics Thresholds
    "SIGNIFICANCE_THRESHOLD",
    "OVERFITTING_GINI_GAP_THRESHOLD",
    "CALIBRATION_AE_LOWER",
    "CALIBRATION_AE_UPPER",
    "FACTOR_AE_DIFF_THRESHOLD",
    "OVERDISPERSION_SEVERE",
    "OVERDISPERSION_MODERATE",
    "OVERDISPERSION_MILD",
    # Partial Dependence Shape Detection
    "PD_FLAT_RELATIVE_RANGE",
    "PD_MONOTONIC_THRESHOLD",
    "PD_CURVATURE_RELATIVE_THRESHOLD",
    "PD_STEP_FUNCTION_RATIO",
]

# =============================================================================
# IRLS Algorithm Defaults
# =============================================================================
DEFAULT_MAX_ITER = 25
DEFAULT_TOLERANCE = 1e-8

# Default TOTAL iteration budget for the monotone smooth path: warm start plus
# all inner PIRLS iterations across lambda updates (scam-style nested solve).
# Matches the previous effective ceiling (10 outer cycles x 200 inner).
DEFAULT_MONOTONE_SMOOTH_MAX_ITER = 2000

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
L1_RATIO_MIN_CLAMP = 1e-3  # Floor for l1_ratio in alpha_max computation
ALPHA_MAX_FLOOR = 1e-4  # Minimum alpha_max to avoid degenerate paths
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
    "quasibinomial": "logit",
    "gamma": "log",
    "tweedie": "log",
    "negbinomial": "log",
    "negativebinomial": "log",
}

# =============================================================================
# Family Aliases
# =============================================================================
NEGBINOMIAL_ALIASES = frozenset(
    {
        "negbinomial",
        "negativebinomial",
        "negative_binomial",
        "nb",
    }
)

# =============================================================================
# Diagnostics Thresholds
# =============================================================================
# Coefficient significance threshold for p-values (alpha = 0.05 standard).
SIGNIFICANCE_THRESHOLD = 0.05

# Minimum train-test Gini gap that flags overfitting risk.
OVERFITTING_GINI_GAP_THRESHOLD = 0.03

# Acceptable A/E ratio range; outside this triggers calibration_drift flag.
CALIBRATION_AE_LOWER = 0.95
CALIBRATION_AE_UPPER = 1.05

# Train/test factor-level A/E divergence flagged as unstable above this absolute difference.
FACTOR_AE_DIFF_THRESHOLD = 0.1

# Pearson chi-square dispersion thresholds for over-dispersion severity classification.
OVERDISPERSION_SEVERE = 5.0
OVERDISPERSION_MODERATE = 2.0
OVERDISPERSION_MILD = 1.5

# =============================================================================
# Partial Dependence Shape Detection Thresholds
# =============================================================================
# Below this fraction of pred_mean, partial-dependence variation is "flat".
PD_FLAT_RELATIVE_RANGE = 0.05

# Fraction of grid steps that must move in the same direction to call a curve "monotonic".
PD_MONOTONIC_THRESHOLD = 0.8

# Curvature threshold (relative to pred_range) above which a monotonic curve is "non-linear".
PD_CURVATURE_RELATIVE_THRESHOLD = 0.1

# Fraction of total range concentrated in a single step → step-function shape.
PD_STEP_FUNCTION_RATIO = 0.4
