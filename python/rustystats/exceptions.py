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
    "FormulaError",
    "ValidationError",
    "SerializationError",
    "wrap_fitting_error",
    "wrap_prediction_error",
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


class FormulaError(RustyStatsError):
    """
    Error parsing formula string.
    
    Check:
    - Matching parentheses
    - Valid variable names
    - Correct function syntax (C(), TE(), bs(), etc.)
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


# Helper functions for wrapping errors with context

def wrap_fitting_error(original_error: Exception, context: str = "") -> FittingError:
    """
    Wrap a low-level error with fitting context.
    
    Parameters
    ----------
    original_error : Exception
        The original exception
    context : str
        Additional context about what was being attempted
        
    Returns
    -------
    FittingError
        Wrapped exception with actionable message
    """
    msg = str(original_error).lower()
    
    if "singular" in msg or "rank" in msg:
        return FittingError(
            f"Design matrix is singular (rank deficient). "
            f"This usually means:\n"
            f"  1. Perfect multicollinearity between predictors\n"
            f"  2. A categorical variable with only one level\n"
            f"  3. Duplicate columns in the design matrix\n"
            f"{context}\n"
            f"Original error: {original_error}"
        )
    
    if "converge" in msg or "iteration" in msg:
        return ConvergenceError(
            f"Model failed to converge. Try:\n"
            f"  1. Increasing max_iter (current default: 25)\n"
            f"  2. Scaling continuous predictors\n"
            f"  3. Adding regularization (alpha > 0)\n"
            f"{context}\n"
            f"Original error: {original_error}"
        )
    
    if "overflow" in msg or "underflow" in msg or "inf" in msg or "nan" in msg:
        return FittingError(
            f"Numerical instability detected. Try:\n"
            f"  1. Scaling predictors to similar ranges\n"
            f"  2. Checking for extreme values in data\n"
            f"  3. Using a different link function\n"
            f"{context}\n"
            f"Original error: {original_error}"
        )
    
    # Generic fitting error
    return FittingError(
        f"Model fitting failed. {context}\n"
        f"Original error: {original_error}"
    )


def wrap_prediction_error(original_error: Exception, context: str = "") -> PredictionError:
    """
    Wrap a low-level error with prediction context.
    
    Parameters
    ----------
    original_error : Exception
        The original exception
    context : str
        Additional context about what was being attempted
        
    Returns
    -------
    PredictionError
        Wrapped exception with actionable message
    """
    msg = str(original_error).lower()
    
    if "column" in msg or "key" in msg or "not found" in msg:
        return PredictionError(
            f"Missing column in prediction data. "
            f"Ensure new data has all columns used during training.\n"
            f"{context}\n"
            f"Original error: {original_error}"
        )
    
    if "shape" in msg or "dimension" in msg:
        return PredictionError(
            f"Shape mismatch in prediction data. "
            f"Check that feature dimensions match training.\n"
            f"{context}\n"
            f"Original error: {original_error}"
        )
    
    return PredictionError(
        f"Prediction failed. {context}\n"
        f"Original error: {original_error}"
    )
