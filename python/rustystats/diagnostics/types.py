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
from typing import Any

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
    # Pair diagnostics (user-supplied interactions=[...])
    "SurfaceCell",
    "SurfaceGrid",
    "InteractionDiagnostics",
    "InteractionExploration",
    "FactorBinPair",
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
    "BasePredictionsByRole",
    "EncodingDiagnostics",
    "InteractionBlockDiagnostics",
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
    if hasattr(obj, "__dict__"):
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
    elif hasattr(obj, "__dataclass_fields__"):
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

    match = re.match(r"pos\((.+)\)$", name)
    if match:
        return _extract_base_variable(match.group(1))

    match = re.match(r"C\(([^)]+)\)\[", name)
    if match:
        return match.group(1).strip()

    match = re.match(r"ms\(([^,)]+)", name)
    if match:
        return match.group(1).strip()

    match = re.match(r"(?:bs|ns|s)\(([^,)]+)", name)
    if match:
        return match.group(1).strip()

    match = re.match(r"I\(([a-zA-Z_][a-zA-Z0-9_]*)\s*\*\*", name)
    if match:
        return match.group(1).strip()

    match = re.match(r"(?:np\.)?(?:log|sqrt|exp|abs)\(([^)]+)\)", name)
    if match:
        return match.group(1).strip()

    if ":" in name:
        return name.split(":")[0].strip()

    return name


# =============================================================================
# Basic Types
# =============================================================================


@dataclass
class Percentiles:
    """Percentile values for a continuous variable (compact array format)."""

    values: list[float]  # [p1, p5, p10, p25, p50, p75, p90, p95, p99]

    @classmethod
    def from_values(cls, p1, p5, p10, p25, p50, p75, p90, p95, p99) -> Percentiles:
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
    ae_ratio: float | None
    ae_ci: list[float]  # [lower, upper]
    actual_total: float | None = None
    expected_total: float | None = None
    base_expected: float | None = None
    base_expected_total: float | None = None
    base_ae_ratio: float | None = None


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
    percentiles: list[float]  # [p1, p5, p10, p25, p50, p75, p90, p95, p99]


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

    chi2: float | None
    p: float | None
    dev_contrib: float | None
    dev_pct: float | None = None


@dataclass
class ScoreTestResult:
    """Rao's score test result for an unfitted factor."""

    statistic: float
    df: int
    pvalue: float
    significant: bool
    expected_dev_pct: float | None = None


@dataclass
class FactorCoefficient:
    """Coefficient for a factor term."""

    term: str
    estimate: float
    std_error: float
    z_value: float
    p_value: float
    relativity: float | None


@dataclass
class FactorDiagnostics:
    """Complete diagnostics for a single factor."""

    name: str
    factor_type: str
    in_model: bool
    transform: str | None
    coefficients: list[FactorCoefficient] | None
    actual_vs_expected: list[ActualExpectedBin]
    residual_pattern: ResidualPattern
    univariate: ContinuousFactorStats | CategoricalFactorStats | None = None
    significance: FactorSignificance | None = None
    score_test: ScoreTestResult | None = None
    relative_importance: float | None = None
    gvif: float | None = None
    train_test_bins: list[FactorBinPair] | None = None


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
    current_terms: list[str] | None = None
    recommendation: str | None = None


@dataclass
class VIFResult:
    """Variance Inflation Factor for a design matrix column."""

    feature: str
    vif: float
    severity: str  # "none", "moderate", "severe", "expected"
    collinear_with: list[str] | None = None


# =============================================================================
# Pair Diagnostics
# =============================================================================


@dataclass
class SurfaceCell:
    """One cell of the binned 2D interaction surface."""

    r: int
    c: int
    n: int
    exposure: float
    actual: float
    predicted: float | None = None
    ae_ratio: float | None = None


@dataclass
class SurfaceGrid:
    """Binned 2D surface for an interaction pair."""

    row_axis: str
    col_axis: str
    row_type: str  # "quantile" | "levels"
    col_type: str
    cells: list[SurfaceCell]
    row_edges: list[float] | None = None
    row_levels: list[str] | None = None
    col_edges: list[float] | None = None
    col_levels: list[str] | None = None


@dataclass
class InteractionDiagnostics:
    """Diagnostics for a single user-specified interaction pair (post-fit)."""

    name: str
    factor1: str
    factor2: str
    pair_type: (
        str  # "continuous_x_continuous" | "continuous_x_categorical" | "categorical_x_categorical"
    )
    in_model: bool
    representation: str | None  # "tensor_product" | "target_encoding" | "frequency_encoding" | None
    train_surface_grid: SurfaceGrid
    test_surface_grid: SurfaceGrid | None = None
    coefficients: list[FactorCoefficient] | None = None
    significance: FactorSignificance | None = None
    score_test: ScoreTestResult | None = None
    gvif: float | None = None


@dataclass
class InteractionBlockDiagnostics:
    """Block diagnostics for fitted or requested higher-order interactions."""

    name: str
    factors: list[str]
    order: int
    in_model: bool
    representation: str | None
    coefficients: list[FactorCoefficient] | None = None
    significance: FactorSignificance | None = None
    score_test: ScoreTestResult | None = None
    gvif: float | None = None


@dataclass
class InteractionExploration:
    """Pre-fit data summary for a single user-specified interaction pair."""

    name: str
    factor1: str
    factor2: str
    pair_type: str
    surface_grid: SurfaceGrid
    interaction_strength: float


@dataclass
class FactorBinPair:
    """Train/test pair for one bin/level of a factor (cell-aligned by bin label)."""

    bin: str
    train_n: int
    train_exposure: float
    train_actual: float
    train_predicted: float
    train_ae_ratio: float | None
    test_n: int
    test_exposure: float
    test_actual: float | None = None
    test_predicted: float | None = None
    test_ae_ratio: float | None = None
    train_actual_total: float | None = None
    train_predicted_total: float | None = None
    train_base_predicted: float | None = None
    train_base_predicted_total: float | None = None
    train_base_ae_ratio: float | None = None
    test_actual_total: float | None = None
    test_predicted_total: float | None = None
    test_base_predicted: float | None = None
    test_base_predicted_total: float | None = None
    test_base_ae_ratio: float | None = None


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
    conf_int: list[float] | None = None
    relativity: float | None = None
    relativity_ci: list[float] | None = None
    robust_std_error: float | None = None
    robust_z_value: float | None = None
    robust_p_value: float | None = None
    robust_significant: bool | None = None


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
    levels: list[DevianceByLevel]
    problem_levels: list[str]


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

    deciles: list[LiftDecile]
    gini: float
    ks_statistic: float
    ks_decile: int
    weak_deciles: list[int]


@dataclass
class PartialDependence:
    """Partial dependence for a variable."""

    variable: str
    variable_type: str
    grid_values: list[Any]
    predictions: list[float]
    relativities: list[float] | None
    shape: str
    recommendation: str
    term_type: str | None = None
    prediction_scale: str = "response"
    relativity_base: Any | None = None
    knots: list[float] | None = None
    boundary_knots: list[float] | None = None
    monotonicity: str | None = None


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
    actual_total: float | None = None
    predicted_total: float | None = None
    base_predicted: float | None = None
    base_predicted_total: float | None = None
    base_ae_ratio: float | None = None


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
    actual_total: float | None = None
    predicted_total: float | None = None
    base_predicted: float | None = None
    base_predicted_total: float | None = None
    base_ae_ratio: float | None = None


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
    aic: float | None
    bic: float | None
    gini: float
    auc: float
    ae_ratio: float
    ae_by_decile: list[DecileMetrics]
    factor_diagnostics: dict[str, list[FactorLevelMetrics]]
    continuous_diagnostics: dict[str, list[ContinuousBandMetrics]]
    log_likelihood_label: str = "log_likelihood"
    is_quasi_likelihood: bool = False


@dataclass
class TrainTestComparison:
    """Train metrics and optional test comparison."""

    train: DatasetDiagnostics
    test: DatasetDiagnostics | None = None
    gini_gap: float | None = None
    ae_ratio_diff: float | None = None
    decile_comparison: list[dict[str, Any]] | None = None
    factor_divergence: dict[str, list[dict[str, Any]]] | None = None
    overfitting_risk: bool = False
    calibration_drift: bool = False
    unstable_factors: list[str] = field(default_factory=list)


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

    def to_dict(self) -> dict[str, Any]:
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
    total_actual: float | None = None
    total_exposure: float | None = None


@dataclass
class BasePredictionsComparison:
    """Comparison between model predictions and base predictions."""

    model_metrics: BasePredictionsMetrics
    base_metrics: BasePredictionsMetrics
    model_vs_base_deciles: list[ModelVsBaseDecile]
    model_better_deciles: int
    base_better_deciles: int
    loss_improvement_pct: float
    gini_improvement: float
    auc_improvement: float


@dataclass
class BasePredictionsByRole:
    """Base/benchmark prediction comparison split by dataset role."""

    train: BasePredictionsComparison | None
    test: BasePredictionsComparison | None = None
    ranking: str = "auto"
    prediction_basis: str = "response"


@dataclass
class EncodingDiagnostics:
    """Representation diagnostics for categorical and encoded terms."""

    name: str
    kind: str
    in_model: bool
    n_levels_train: int | None = None
    n_levels_test: int | None = None
    unseen_levels_test: int | None = None
    rare_levels_grouped: int | None = None
    interaction_order: int = 1
    source_factors: list[str] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# =============================================================================
# Data Exploration
# =============================================================================


@dataclass
class DataExploration:
    """Pre-fit data exploration results."""

    data_summary: dict[str, Any]
    factor_stats: list[dict[str, Any]]
    missing_values: dict[str, Any]
    univariate_tests: list[dict[str, Any]]
    correlations: dict[str, Any]
    cramers_v: dict[str, Any]
    vif: list[dict[str, Any]]
    zero_inflation: dict[str, Any]
    overdispersion: dict[str, Any]
    interaction_candidates: list[InteractionCandidate]
    response_stats: dict[str, Any]
    interactions: list[InteractionExploration] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _to_dict_recursive(self)

    def to_json(self, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=_json_default)


# =============================================================================
# Main Diagnostics Output
# =============================================================================


@dataclass
class ModelDiagnostics:
    """Complete model diagnostics output."""

    model_summary: dict[str, Any]
    train_test: TrainTestComparison
    calibration: dict[str, Any]
    residual_summary: dict[str, ResidualSummary]
    factors: list[FactorDiagnostics]
    interaction_candidates: list[InteractionCandidate]
    model_comparison: dict[str, float]
    warnings: list[dict[str, str]]
    vif: list[VIFResult] | None = None
    smooth_terms: list[SmoothTermDiagnostics] | None = None
    coefficient_summary: list[CoefficientSummary] | None = None
    factor_deviance: list[FactorDeviance] | None = None
    lift_chart: LiftChart | None = None
    partial_dependence: list[PartialDependence] | None = None
    overdispersion: dict[str, Any] | None = None
    spline_info: dict[str, dict[str, Any]] | None = None
    base_predictions_by_role: BasePredictionsByRole | None = None
    encoding_diagnostics: list[EncodingDiagnostics] | None = None
    interaction_blocks: list[InteractionBlockDiagnostics] = field(default_factory=list)
    interactions: list[InteractionDiagnostics] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _to_dict_recursive(self)

    def to_json(self, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=_json_default)
