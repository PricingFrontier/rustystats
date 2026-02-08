"""
Custom exceptions for RustyStats with actionable error messages.

This module provides a hierarchy of exceptions that give users clear
guidance on how to resolve issues.
"""

__all__ = [
    "RustyStatsError",
    "DesignMatrixError",
    "FittingError",
    "ConvergenceError",
    "PredictionError",
    "EncodingError",
    "ValidationError",
    "SerializationError",
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


class ValidationError(RustyStatsError):
    """
    Input validation error.
    
    Common causes:
    - Invalid parameter values
    - Incompatible array shapes
    - Wrong data types
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
