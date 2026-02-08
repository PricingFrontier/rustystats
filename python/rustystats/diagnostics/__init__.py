"""
RustyStats Model Diagnostics Package
=====================================

This package provides comprehensive model diagnostics for assessing GLM quality.

Features:
- Overall model fit statistics
- Calibration metrics (A/E ratios, calibration curves)
- Discrimination metrics (Gini, lift, Lorenz curve)
- Per-factor diagnostics (for both fitted and unfitted factors)
- Interaction detection
- JSON export for LLM consumption

Usage:
------
>>> result = rs.glm_dict(response="y", terms={"x1": {"type": "linear"}, "region": {"type": "categorical"}}, data=data, family="poisson").fit()
>>> diagnostics = result.diagnostics(
...     data=data,
...     categorical_factors=["region", "brand"],
...     continuous_factors=["age", "income"]
... )
>>> print(diagnostics.to_json())
"""

# Import from types module (extracted dataclasses)
from rustystats.diagnostics.types import (
    # Utility functions
    _json_default,
    _round_float,
    _to_dict_recursive,
    _extract_base_variable,
    # Basic types
    Percentiles,
    ResidualSummary,
    CalibrationBin,
    LorenzPoint,
    ActualExpectedBin,
    ResidualPattern,
    # Factor statistics
    ContinuousFactorStats,
    CategoricalLevelStats,
    CategoricalFactorStats,
    FactorSignificance,
    ScoreTestResult,
    FactorCoefficient,
    FactorDiagnostics,
    # Interaction and VIF
    InteractionCandidate,
    VIFResult,
    # Coefficient and deviance
    CoefficientSummary,
    DevianceByLevel,
    FactorDeviance,
    # Lift and calibration
    LiftDecile,
    LiftChart,
    PartialDependence,
    DecileMetrics,
    FactorLevelMetrics,
    ContinuousBandMetrics,
    # Dataset diagnostics
    DatasetDiagnostics,
    TrainTestComparison,
    ConvergenceDetails,
    # Smooth terms
    SmoothTermDiagnostics,
    # Base predictions
    ModelVsBaseDecile,
    BasePredictionsMetrics,
    BasePredictionsComparison,
    # Data exploration
    DataExploration,
    # Main output
    ModelDiagnostics,
)

# Import computation components
from rustystats.diagnostics.components import (
    _ResidualComputer,
    _CalibrationComputer,
    _DiscriminationComputer,
)

# Import main computation orchestrator
from rustystats.diagnostics.computer import DiagnosticsComputer

# Import pre-fit data exploration
from rustystats.diagnostics.explorer import DataExplorer, explore_data

# Import top-level API functions
from rustystats.diagnostics.api import compute_diagnostics, _compute_smooth_term_diagnostics

__all__ = [
    # Types
    "Percentiles",
    "ResidualSummary",
    "CalibrationBin",
    "LorenzPoint",
    "ActualExpectedBin",
    "ResidualPattern",
    "ContinuousFactorStats",
    "CategoricalLevelStats",
    "CategoricalFactorStats",
    "FactorSignificance",
    "ScoreTestResult",
    "FactorCoefficient",
    "FactorDiagnostics",
    "InteractionCandidate",
    "VIFResult",
    "CoefficientSummary",
    "DevianceByLevel",
    "FactorDeviance",
    "LiftDecile",
    "LiftChart",
    "PartialDependence",
    "DecileMetrics",
    "FactorLevelMetrics",
    "ContinuousBandMetrics",
    "DatasetDiagnostics",
    "TrainTestComparison",
    "ConvergenceDetails",
    "SmoothTermDiagnostics",
    "ModelVsBaseDecile",
    "BasePredictionsMetrics",
    "BasePredictionsComparison",
    "DataExploration",
    "ModelDiagnostics",
    # Computer classes and functions
    "DiagnosticsComputer",
    "DataExplorer",
    "compute_diagnostics",
    "explore_data",
]
