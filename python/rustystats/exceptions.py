"""
Custom exceptions for RustyStats with actionable error messages.

This module provides a hierarchy of exceptions that give users clear
guidance on how to resolve issues.
"""

__all__ = [
    "BatchSourceChangedError",
    "ConvergenceError",
    "DesignMatrixError",
    "DesignTooWideError",
    "EncodingError",
    "FittingError",
    "LearnedStateTooLargeError",
    "MemoryPlanExceededError",
    "OutOfCoreOptionError",
    "OutOfCoreResultError",
    "PredictionError",
    "RustyStatsError",
    "SerializationError",
    "SpillError",
    "UnseenLabelError",
    "ValidationError",
]


class RustyStatsError(Exception):
    """Base exception for all RustyStats errors."""

    pass


class DesignMatrixError(RustyStatsError):
    """
    Error in design matrix construction or validation.

    Common causes:
    - Singular matrix (perfect multicollinearity)
    - Missing values in predictors
    - Categorical variable with only one level
    - Duplicate columns
    """

    pass


class FittingError(RustyStatsError):
    """
    Error during model fitting.

    Common causes:
    - IRLS failed to converge
    - Numerical overflow/underflow
    - Invalid family/link combination
    - Separation in logistic regression
    """

    pass


class ConvergenceError(FittingError):
    """
    Model fitting failed to converge within max iterations.

    Try:
    - Increasing max_iter
    - Scaling predictors
    - Using a different starting point
    - Checking for separation (binomial)
    """

    pass


class PredictionError(RustyStatsError):
    """
    Error during prediction on new data.

    Common causes:
    - New data has different columns than training
    - Unseen categorical levels (check warnings)
    - Missing required columns
    """

    pass


class EncodingError(RustyStatsError):
    """
    Error in categorical or target encoding.

    Common causes:
    - Empty categories
    - Target values with wrong dtype
    - Mismatched array lengths
    """

    pass


class UnseenLabelError(EncodingError):
    """
    A value was not among the categories learned from the same column.

    In an out-of-core fit the categories are learned in one pass and applied
    in later ones, so this means the source changed between passes.
    """

    pass


class LearnedStateTooLargeError(RustyStatsError):
    """
    Learned state would exceed its byte budget.

    Raised before the state is allocated. Try:
    - Raising max_learned_state_bytes (or max_bytes)
    - Reducing the cardinality of the named column
    """

    pass


class OutOfCoreResultError(RustyStatsError):
    """
    A result method needs per-row training values that an out-of-core fit does not keep.

    Out-of-core results hold coefficients, covariances and fit statistics, not
    fitted values, residuals or the design. Predict on the data instead.
    """

    pass


class ValidationError(RustyStatsError):
    """
    Input validation error.

    Common causes:
    - Invalid parameter values
    - Incompatible array shapes
    - Wrong data types
    """

    pass


class OutOfCoreOptionError(ValidationError):
    """
    An out-of-core option, or a model, that an out-of-core fit cannot honour.

    Raised before the source is read. The message names the option.
    """

    pass


class BatchSourceChangedError(RustyStatsError):
    """
    A BatchSource gave different rows on a later pass than on the first.

    The message names the pass and the column whose fingerprint changed, or the
    dtype or row-count difference. Every pass must yield the same rows in the same
    order.
    """

    pass


class DesignTooWideError(DesignMatrixError):
    """
    The design would have more columns than max_design_columns allows.

    The message names p, the limit and the term that crossed it.
    """

    pass


class MemoryPlanExceededError(RustyStatsError):
    """
    A phase of an out-of-core fit would need more memory than max_bytes allows.

    The message names the phase, its planned peak and the limit. Try raising
    max_bytes or lowering encode_rows.
    """

    pass


class SpillError(RustyStatsError):
    """
    A spill file could not be written, read or removed.

    The message names the path and the operating-system error.
    """

    pass


class SerializationError(RustyStatsError):
    """
    Error during model serialization or deserialization.

    Common causes:
    - Corrupted model file
    - Version mismatch
    - Missing required fields
    """

    pass
