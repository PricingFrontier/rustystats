"""
Data types for RustyStats diagnostics.

This module contains all dataclasses used in diagnostics output.
These types are extracted from the original diagnostics module for better organization.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union

import numpy as np

__all__ = [
    # Utility functions
    "_json_default",
    "_round_float",
    "_to_dict_recursive",
    "_extract_base_variable",
    # Basic types
    "Percentiles",
    "ResidualSummary",
    "CalibrationBin",
    "LorenzPoint",
    "ActualExpectedBin",
    "ResidualPattern",
    # Factor statistics
    "ContinuousFactorStats",
    "CategoricalLevelStats",
    "CategoricalFactorStats",
    "FactorSignificance",
    "ScoreTestResult",
    "FactorCoefficient",
    "FactorDiagnostics",
    # Interaction and VIF
    "InteractionCandidate",
    "VIFResult",
    # Coefficient and deviance
    "CoefficientSummary",
    "DevianceByLevel",
    "FactorDeviance",
    # Lift and calibration
    "LiftDecile",
    "LiftChart",
    "PartialDependence",
    "DecileMetrics",
    "FactorLevelMetrics",
    "ContinuousBandMetrics",
    # Dataset diagnostics
    "DatasetDiagnostics",
    "TrainTestComparison",
    "ConvergenceDetails",
    # Smooth terms
    "SmoothTermDiagnostics",
    # Base predictions
    "ModelVsBaseDecile",
    "BasePredictionsMetrics",
    "BasePredictionsComparison",
    # Data exploration
    "DataExploration",
    # Main output
    "ModelDiagnostics",
]


# =============================================================================
# Utility Functions
# =============================================================================

def _json_default(obj: Any) -> Any:
    """Handle special types for JSON serialization."""
    if isinstance(obj, float):
        if math.isnan(obj):
            return None
        if math.isinf(obj):
            return None
    if hasattr(obj, '__dict__'):
        return obj.__dict__
    return str(obj)


def _round_float(x: float, decimals: int = 4) -> float:
    """Round float for token-efficient JSON output."""
    if x == 0:
        return 0.0
    if abs(x) >= 100:
        return round(x, 2)
    elif abs(x) >= 1:
        return round(x, 4)
    else:
        return round(x, 6)


def _to_dict_recursive(obj) -> Any:
    """Recursively convert dataclasses and handle special values."""
    if isinstance(obj, dict):
        return {k: _to_dict_recursive(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_dict_recursive(v) for v in obj]
    elif isinstance(obj, SmoothTermDiagnostics):
        return obj.to_dict()
    elif hasattr(obj, '__dataclass_fields__'):
        result = {}
        for field_name in obj.__dataclass_fields__:
            value = getattr(obj, field_name)
            result[field_name] = _to_dict_recursive(value)
        return result
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return _round_float(obj)
    elif isinstance(obj, np.ndarray):
        return [_to_dict_recursive(v) for v in obj.tolist()]
    elif isinstance(obj, np.floating):
        return _round_float(float(obj))
    elif isinstance(obj, np.integer):
        return int(obj)
    else:
        return obj


def _extract_base_variable(feature_name: str) -> str:
    """Extract base variable name from a feature name.
    
    Examples:
        'BonusMalus' -> 'BonusMalus'
        'I(BonusMalus ** 2)' -> 'BonusMalus'
        'bs(age, 1/4)' -> 'age'
        'C(Region)[T.A]' -> 'Region'
    """
    name = feature_name.strip()
    
    match = re.match(r'pos\((.+)\)$', name)
    if match:
        return _extract_base_variable(match.group(1))
    
    match = re.match(r'C\(([^)]+)\)\[', name)
    if match:
        return match.group(1).strip()
    
    match = re.match(r'ms\(([^,)]+)', name)
    if match:
        return match.group(1).strip()
    
    match = re.match(r'(?:bs|ns|s)\(([^,)]+)', name)
    if match:
        return match.group(1).strip()
    
    match = re.match(r'I\(([a-zA-Z_][a-zA-Z0-9_]*)\s*\*\*', name)
    if match:
        return match.group(1).strip()
    
    match = re.match(r'(?:np\.)?(?:log|sqrt|exp|abs)\(([^)]+)\)', name)
    if match:
        return match.group(1).strip()
    
    if ':' in name:
        return name.split(':')[0].strip()
    
    return name


# =============================================================================
# Basic Types
# =============================================================================

@dataclass
class Percentiles:
    """Percentile values for a continuous variable (compact array format)."""
    values: List[float]  # [p1, p5, p10, p25, p50, p75, p90, p95, p99]
    
    @classmethod
    def from_values(cls, p1, p5, p10, p25, p50, p75, p90, p95, p99) -> "Percentiles":
        return cls(values=[p1, p5, p10, p25, p50, p75, p90, p95, p99])


@dataclass
class ResidualSummary:
    """Summary statistics for residuals."""
    mean: float
    std: float
    skewness: float


@dataclass
class CalibrationBin:
    """A single bin in the calibration curve."""
    bin_index: int
    predicted_lower: float
    predicted_upper: float
    predicted_mean: float
    actual_mean: float
    actual_expected_ratio: float
    count: int
    exposure: float
    actual_sum: float
    predicted_sum: float
    ae_confidence_interval_lower: float
    ae_confidence_interval_upper: float


@dataclass
class LorenzPoint:
    """A point on the Lorenz curve."""
    cumulative_exposure_pct: float
    cumulative_actual_pct: float
    cumulative_predicted_pct: float


@dataclass
class ActualExpectedBin:
    """A/E statistics for a single bin."""
    bin: str
    n: int
    exposure: float
    actual: float
    expected: float
    ae_ratio: float
    ae_ci: List[float]  # [lower, upper]


@dataclass
class ResidualPattern:
    """Residual pattern analysis for a factor."""
    resid_corr: float
    var_explained: float


# =============================================================================
# Factor Statistics
# =============================================================================

@dataclass
class ContinuousFactorStats:
    """Univariate statistics for a continuous factor."""
    mean: float
    std: float
    min: float
    max: float
    missing_count: int
    percentiles: List[float]  # [p1, p5, p10, p25, p50, p75, p90, p95, p99]


@dataclass
class CategoricalLevelStats:
    """Statistics for a categorical level."""
    level: str
    count: int
    percentage: float


@dataclass
class CategoricalFactorStats:
    """Distribution statistics for a categorical factor."""
    n_levels: int
    n_rare_levels: int
    rare_level_total_pct: float


@dataclass
class FactorSignificance:
    """Statistical significance tests for a factor."""
    chi2: Optional[float]
    p: Optional[float]
    dev_contrib: Optional[float]
    dev_pct: Optional[float] = None


@dataclass
class ScoreTestResult:
    """Rao's score test result for an unfitted factor."""
    statistic: float
    df: int
    pvalue: float
    significant: bool
    expected_dev_pct: Optional[float] = None


@dataclass
class FactorCoefficient:
    """Coefficient for a factor term."""
    term: str
    estimate: float
    std_error: float
    z_value: float
    p_value: float
    relativity: Optional[float]


@dataclass
class FactorDiagnostics:
    """Complete diagnostics for a single factor."""
    name: str
    factor_type: str
    in_model: bool
    transform: Optional[str]
    coefficients: Optional[List[FactorCoefficient]]
    actual_vs_expected: List[ActualExpectedBin]
    residual_pattern: ResidualPattern
    univariate: Optional[Union[ContinuousFactorStats, CategoricalFactorStats]] = None
    significance: Optional[FactorSignificance] = None
    score_test: Optional[ScoreTestResult] = None
    relative_importance: Optional[float] = None


# =============================================================================
# Interaction and VIF
# =============================================================================

@dataclass
class InteractionCandidate:
    """A potential interaction between two factors."""
    factor1: str
    factor2: str
    interaction_strength: float
    pvalue: float
    n_cells: int
    current_terms: Optional[List[str]] = None
    recommendation: Optional[str] = None


@dataclass
class VIFResult:
    """Variance Inflation Factor for a design matrix column."""
    feature: str
    vif: float
    severity: str  # "none", "moderate", "severe", "expected"
    collinear_with: Optional[List[str]] = None


# =============================================================================
# Coefficient and Deviance
# =============================================================================

@dataclass
class CoefficientSummary:
    """Summary of a coefficient for interpretation."""
    feature: str
    estimate: float
    std_error: float
    z_value: float
    p_value: float
    significant: bool
    relativity: Optional[float]
    relativity_ci: Optional[List[float]]


@dataclass
class DevianceByLevel:
    """Deviance contribution for a factor level."""
    level: str
    n: int
    deviance: float
    deviance_pct: float
    mean_deviance: float
    ae_ratio: float
    problem: bool


@dataclass
class FactorDeviance:
    """Deviance breakdown by factor levels."""
    factor: str
    total_deviance: float
    levels: List[DevianceByLevel]
    problem_levels: List[str]


# =============================================================================
# Lift and Calibration
# =============================================================================

@dataclass
class LiftDecile:
    """Lift statistics for a single decile."""
    decile: int
    n: int
    exposure: float
    actual: float
    predicted: float
    ae_ratio: float
    cumulative_actual_pct: float
    cumulative_predicted_pct: float
    lift: float
    cumulative_lift: float


@dataclass
class LiftChart:
    """Full lift chart with all deciles."""
    deciles: List[LiftDecile]
    gini: float
    ks_statistic: float
    ks_decile: int
    weak_deciles: List[int]


@dataclass
class PartialDependence:
    """Partial dependence for a variable."""
    variable: str
    variable_type: str
    grid_values: List[Any]
    predictions: List[float]
    relativities: Optional[List[float]]
    shape: str
    recommendation: str


@dataclass
class DecileMetrics:
    """Metrics for a single decile in calibration analysis."""
    decile: int
    n: int
    exposure: float
    actual: float
    predicted: float
    ae_ratio: float


@dataclass
class FactorLevelMetrics:
    """Metrics for a single factor level."""
    level: str
    n: int
    exposure: float
    actual: float
    predicted: float
    ae_ratio: float
    residual_mean: float


@dataclass
class ContinuousBandMetrics:
    """Metrics for a continuous variable band."""
    band: int
    range_min: float
    range_max: float
    midpoint: float
    n: int
    exposure: float
    actual: float
    predicted: float
    ae_ratio: float
    partial_dep: float
    residual_mean: float


# =============================================================================
# Dataset Diagnostics
# =============================================================================

@dataclass
class DatasetDiagnostics:
    """Comprehensive diagnostics for a single dataset (train or test)."""
    dataset: str
    n_obs: int
    total_exposure: float
    total_actual: float
    total_predicted: float
    loss: float
    deviance: float
    log_likelihood: float
    aic: float
    gini: float
    auc: float
    ae_ratio: float
    ae_by_decile: List[DecileMetrics]
    factor_diagnostics: Dict[str, List[FactorLevelMetrics]]
    continuous_diagnostics: Dict[str, List[ContinuousBandMetrics]]


@dataclass
class TrainTestComparison:
    """Train metrics and optional test comparison."""
    train: DatasetDiagnostics
    test: Optional[DatasetDiagnostics] = None
    gini_gap: Optional[float] = None
    ae_ratio_diff: Optional[float] = None
    decile_comparison: Optional[List[Dict[str, Any]]] = None
    factor_divergence: Optional[Dict[str, List[Dict[str, Any]]]] = None
    overfitting_risk: bool = False
    calibration_drift: bool = False
    unstable_factors: List[str] = field(default_factory=list)


@dataclass
class ConvergenceDetails:
    """Details about model convergence."""
    max_iterations_allowed: int
    iterations_used: int
    converged: bool
    reason: str


# =============================================================================
# Smooth Terms
# =============================================================================

@dataclass
class SmoothTermDiagnostics:
    """Diagnostics for a smooth term (penalized spline)."""
    variable: str
    k: int
    edf: float
    lambda_: float
    gcv: float
    ref_df: float
    chi2: float
    p_value: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "variable": self.variable,
            "k": self.k,
            "edf": round(self.edf, 2),
            "lambda": round(self.lambda_, 4),
            "gcv": round(self.gcv, 4),
            "ref_df": round(self.ref_df, 2),
            "chi2": round(self.chi2, 2),
            "p_value": round(self.p_value, 4),
        }


# =============================================================================
# Base Predictions Comparison
# =============================================================================

@dataclass
class ModelVsBaseDecile:
    """Metrics for comparing model vs base predictions by decile."""
    decile: int
    n: int
    exposure: float
    actual: float
    model_predicted: float
    base_predicted: float
    model_ae_ratio: float
    base_ae_ratio: float
    model_base_ratio_mean: float


@dataclass
class BasePredictionsMetrics:
    """Metrics for base predictions."""
    total_predicted: float
    ae_ratio: float
    loss: float
    gini: float
    auc: float


@dataclass
class BasePredictionsComparison:
    """Comparison between model predictions and base predictions."""
    model_metrics: BasePredictionsMetrics
    base_metrics: BasePredictionsMetrics
    model_vs_base_deciles: List[ModelVsBaseDecile]
    model_better_deciles: int
    base_better_deciles: int
    loss_improvement_pct: float
    gini_improvement: float
    auc_improvement: float


# =============================================================================
# Data Exploration
# =============================================================================

@dataclass
class DataExploration:
    """Pre-fit data exploration results."""
    data_summary: Dict[str, Any]
    factor_stats: List[Dict[str, Any]]
    missing_values: Dict[str, Any]
    univariate_tests: List[Dict[str, Any]]
    correlations: Dict[str, Any]
    cramers_v: Dict[str, Any]
    vif: List[Dict[str, Any]]
    zero_inflation: Dict[str, Any]
    overdispersion: Dict[str, Any]
    interaction_candidates: List[InteractionCandidate]
    response_stats: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return _to_dict_recursive(self)
    
    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=_json_default)


# =============================================================================
# Main Diagnostics Output
# =============================================================================

@dataclass
class ModelDiagnostics:
    """Complete model diagnostics output."""
    model_summary: Dict[str, Any]
    train_test: TrainTestComparison
    calibration: Dict[str, Any]
    residual_summary: Dict[str, ResidualSummary]
    factors: List[FactorDiagnostics]
    interaction_candidates: List[InteractionCandidate]
    model_comparison: Dict[str, float]
    warnings: List[Dict[str, str]]
    vif: Optional[List[VIFResult]] = None
    smooth_terms: Optional[List[SmoothTermDiagnostics]] = None
    coefficient_summary: Optional[List[CoefficientSummary]] = None
    factor_deviance: Optional[List[FactorDeviance]] = None
    lift_chart: Optional[LiftChart] = None
    partial_dependence: Optional[List[PartialDependence]] = None
    overdispersion: Optional[Dict[str, Any]] = None
    spline_info: Optional[Dict[str, Dict[str, Any]]] = None
    base_predictions_comparison: Optional[BasePredictionsComparison] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return _to_dict_recursive(self)
    
    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=_json_default)
