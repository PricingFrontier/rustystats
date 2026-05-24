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
# Import top-level API functions
from rustystats.diagnostics.api import (  # noqa: F401
    _compute_smooth_term_diagnostics,
    compute_diagnostics,
)

# Import computation components
from rustystats.diagnostics.components import (
    _CalibrationComputer,  # noqa: F401
    _DiscriminationComputer,  # noqa: F401
    _ResidualComputer,  # noqa: F401
)

# Import main computation orchestrator
from rustystats.diagnostics.computer import DiagnosticsComputer

# Import pre-fit data exploration
from rustystats.diagnostics.explorer import DataExplorer, explore_data
from rustystats.diagnostics.types import (
    ActualExpectedBin,
    BasePredictionsComparison,
    BasePredictionsMetrics,
    CalibrationBin,
    CategoricalFactorStats,
    CategoricalLevelStats,
    # Coefficient and deviance
    CoefficientSummary,
    ContinuousBandMetrics,
    # Factor statistics
    ContinuousFactorStats,
    ConvergenceDetails,
    # Data exploration
    DataExploration,
    # Dataset diagnostics
    DatasetDiagnostics,
    DecileMetrics,
    DevianceByLevel,
    FactorBinPair,
    FactorCoefficient,
    FactorDeviance,
    FactorDiagnostics,
    FactorLevelMetrics,
    FactorSignificance,
    # Interaction and VIF
    InteractionCandidate,
    InteractionDiagnostics,
    InteractionExploration,
    LiftChart,
    # Lift and calibration
    LiftDecile,
    LorenzPoint,
    # Main output
    ModelDiagnostics,
    # Base predictions
    ModelVsBaseDecile,
    PartialDependence,
    # Basic types
    Percentiles,
    ResidualPattern,
    ResidualSummary,
    ScoreTestResult,
    # Smooth terms
    SmoothTermDiagnostics,
    SurfaceCell,
    SurfaceGrid,
    TrainTestComparison,
    VIFResult,
    _extract_base_variable,  # noqa: F401
    # Utility functions
    _json_default,  # noqa: F401
    _round_float,  # noqa: F401
    _to_dict_recursive,  # noqa: F401
)

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
    "FactorBinPair",
    "InteractionCandidate",
    "InteractionDiagnostics",
    "InteractionExploration",
    "SurfaceCell",
    "SurfaceGrid",
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
