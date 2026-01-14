"""
Model Diagnostics for RustyStats GLM
=====================================

This module provides comprehensive model diagnostics for assessing GLM quality.

Features:
- Overall model fit statistics
- Calibration metrics (A/E ratios, calibration curves)
- Discrimination metrics (Gini, lift, Lorenz curve)
- Per-factor diagnostics (for both fitted and unfitted factors)
- Interaction detection
- JSON export for LLM consumption

Usage:
------
>>> result = rs.glm("y ~ x1 + C(region)", data, family="poisson").fit()
>>> diagnostics = result.diagnostics(
...     data=data,
...     categorical_factors=["region", "brand"],
...     continuous_factors=["age", "income"]
... )
>>> print(diagnostics.to_json())
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from functools import cached_property

# Import Rust diagnostics functions
from rustystats._rustystats import (
    compute_calibration_curve_py as _rust_calibration_curve,
    compute_discrimination_stats_py as _rust_discrimination_stats,
    compute_ae_continuous_py as _rust_ae_continuous,
    compute_ae_categorical_py as _rust_ae_categorical,
    compute_loss_metrics_py as _rust_loss_metrics,
    compute_lorenz_curve_py as _rust_lorenz_curve,
    hosmer_lemeshow_test_py as _rust_hosmer_lemeshow,
    compute_fit_statistics_py as _rust_fit_statistics,
    compute_dataset_metrics_py as _rust_dataset_metrics,
    compute_residual_summary_py as _rust_residual_summary,
    compute_residual_pattern_py as _rust_residual_pattern,
    compute_pearson_residuals_py as _rust_pearson_residuals,
    compute_deviance_residuals_py as _rust_deviance_residuals,
    compute_null_deviance_py as _rust_null_deviance,
    compute_unit_deviance_py as _rust_unit_deviance,
    # Statistical distribution CDFs (replaces scipy.stats)
    chi2_cdf_py as _chi2_cdf,
    t_cdf_py as _t_cdf,
    f_cdf_py as _f_cdf,
)
from typing import Optional, List, Dict, Any, Union, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import polars as pl


# =============================================================================
# Data Classes for Diagnostics Structure
# =============================================================================

@dataclass
class Percentiles:
    """Percentile values for a continuous variable (compact array format).
    
    Token optimization: stored as [p1, p5, p10, p25, p50, p75, p90, p95, p99]
    """
    values: List[float]  # [p1, p5, p10, p25, p50, p75, p90, p95, p99]
    
    @classmethod
    def from_values(cls, p1, p5, p10, p25, p50, p75, p90, p95, p99) -> "Percentiles":
        return cls(values=[p1, p5, p10, p25, p50, p75, p90, p95, p99])


@dataclass
class ResidualSummary:
    """Summary statistics for residuals (compressed: mean, std, skewness only)."""
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
    """A/E statistics for a single bin.
    
    For count models, actual and expected are frequencies (per-exposure rates).
    """
    bin: str  # bin label or range
    n: int  # count of observations
    exposure: float  # total exposure in bin
    actual: float  # actual frequency = sum(y) / exposure
    expected: float  # expected frequency = sum(mu) / exposure
    ae_ratio: float  # actual/expected ratio
    ae_ci: List[float]  # [lower, upper] confidence interval


@dataclass
class ResidualPattern:
    """Residual pattern analysis for a factor (compressed)."""
    resid_corr: float  # correlation_with_residuals
    var_explained: float  # residual_variance_explained


@dataclass
class ContinuousFactorStats:
    """Univariate statistics for a continuous factor (compact format)."""
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
    """Distribution statistics for a categorical factor (compressed: no levels array)."""
    n_levels: int
    n_rare_levels: int
    rare_level_total_pct: float


@dataclass
class FactorSignificance:
    """Statistical significance tests for a factor (compressed field names)."""
    chi2: Optional[float]  # Wald chi-square test statistic
    p: Optional[float]  # p-value for Wald test
    dev_contrib: Optional[float]  # Drop-in-deviance if term removed


@dataclass
class FactorCoefficient:
    """Coefficient for a factor term.
    
    For categorical factors, each level (except base) has a coefficient.
    For continuous factors with splines, each basis function has a coefficient.
    """
    term: str  # e.g., "C(region)[T.B]" or "bs(age, 2/4)"
    estimate: float
    std_error: float
    z_value: float
    p_value: float
    relativity: Optional[float]  # exp(coef) for log-link


@dataclass
class FactorDiagnostics:
    """Complete diagnostics for a single factor."""
    name: str
    factor_type: str  # "continuous" or "categorical"
    in_model: bool
    transform: Optional[str]  # transformation applied (e.g. "bs(age, df=4)")
    coefficients: Optional[List[FactorCoefficient]]  # fitted coefficients for this factor
    univariate: Union[ContinuousFactorStats, CategoricalFactorStats]
    actual_vs_expected: List[ActualExpectedBin]
    residual_pattern: ResidualPattern
    significance: Optional[FactorSignificance] = None


@dataclass
class InteractionCandidate:
    """A potential interaction between two factors."""
    factor1: str
    factor2: str
    interaction_strength: float
    pvalue: float
    n_cells: int
    current_terms: Optional[List[str]] = None  # How factors currently appear in model
    recommendation: Optional[str] = None  # Suggested action


@dataclass
class VIFResult:
    """Variance Inflation Factor for a design matrix column."""
    feature: str
    vif: float
    severity: str  # "none", "moderate", "severe"
    collinear_with: Optional[List[str]] = None  # Features it's most correlated with


@dataclass
class CoefficientSummary:
    """Summary of a coefficient for agent interpretation.
    
    Token optimization: removed 'recommendation' (usually null) and 'impact' (derivable).
    """
    feature: str
    estimate: float
    std_error: float
    z_value: float
    p_value: float
    significant: bool
    relativity: Optional[float]  # exp(coef) for log-link
    relativity_ci: Optional[List[float]]  # [lower, upper]


@dataclass
class DevianceByLevel:
    """Deviance contribution for a factor level."""
    level: str
    n: int
    deviance: float
    deviance_pct: float  # Percentage of total deviance
    mean_deviance: float  # Per-observation deviance
    ae_ratio: float
    problem: bool  # True if this level is problematic


@dataclass
class FactorDeviance:
    """Deviance breakdown by factor levels."""
    factor: str
    total_deviance: float
    levels: List[DevianceByLevel]
    problem_levels: List[str]  # Levels with high deviance contribution


@dataclass
class LiftDecile:
    """Lift statistics for a single decile.
    
    For count models, actual and predicted are frequencies (per-exposure rates).
    """
    decile: int  # 1-10
    n: int
    exposure: float
    actual: float  # actual frequency = sum(y) / exposure
    predicted: float  # predicted frequency = sum(mu) / exposure
    ae_ratio: float
    cumulative_actual_pct: float
    cumulative_predicted_pct: float
    lift: float  # actual_rate / overall_rate
    cumulative_lift: float


@dataclass
class LiftChart:
    """Full lift chart with all deciles."""
    deciles: List[LiftDecile]
    gini: float
    ks_statistic: float
    ks_decile: int  # Decile where max separation occurs
    weak_deciles: List[int]  # Deciles with poor discrimination


@dataclass
class PartialDependence:
    """Partial dependence for a variable."""
    variable: str
    variable_type: str  # "continuous" or "categorical"
    grid_values: List[Any]  # x-axis values
    predictions: List[float]  # Mean prediction at each grid value
    relativities: Optional[List[float]]  # exp(predictions) for log-link
    std_errors: Optional[List[float]]  # Standard errors of predictions
    shape: str  # "linear", "monotonic", "u_shaped", "complex"
    recommendation: str  # e.g., "Consider spline" or "Linear effect adequate"


@dataclass
class DecileMetrics:
    """Metrics for a single decile in calibration analysis.
    
    For count models, actual and predicted are frequencies (per-exposure rates).
    """
    decile: int
    n: int
    exposure: float
    actual: float  # actual frequency = sum(y) / exposure
    predicted: float  # predicted frequency = sum(mu) / exposure
    ae_ratio: float


@dataclass
class FactorLevelMetrics:
    """Metrics for a single factor level.
    
    For count models, actual and predicted are frequencies (per-exposure rates).
    """
    level: str
    n: int
    exposure: float
    actual: float  # actual frequency = sum(y) / exposure
    predicted: float  # predicted frequency = sum(mu) / exposure
    ae_ratio: float
    residual_mean: float


@dataclass
class ContinuousBandMetrics:
    """Metrics for a continuous variable band.
    
    For count models, actual and predicted are frequencies (per-exposure rates).
    """
    band: int
    range_min: float
    range_max: float
    midpoint: float
    n: int
    exposure: float
    actual: float  # actual frequency = sum(y) / exposure
    predicted: float  # predicted frequency = sum(mu) / exposure
    ae_ratio: float
    partial_dep: float  # Marginal effect at midpoint
    residual_mean: float  # Mean deviance residual for this band


@dataclass
class DatasetDiagnostics:
    """Comprehensive diagnostics for a single dataset (train or test).
    
    Includes family deviance loss (same as GBM loss functions like
    Poisson NLL, Gamma deviance) and AIC for model comparison.
    """
    dataset: str  # "train" or "test"
    n_obs: int
    total_exposure: float
    total_actual: float
    total_predicted: float
    
    # PRIMARY LOSS METRIC - USE THIS for model comparison
    loss: float  # Family-appropriate per-obs loss (Poisson deviance, NB deviance, etc.)
    
    # Fit statistics
    deviance: float  # Total deviance (sum of unit deviances)
    log_likelihood: float
    aic: float
    
    # Discrimination
    gini: float
    auc: float
    
    # Overall calibration
    ae_ratio: float
    
    # A/E by decile (10 buckets sorted by predicted value)
    ae_by_decile: List[DecileMetrics]
    
    # Factor-level diagnostics (keyed by factor name)
    factor_diagnostics: Dict[str, List[FactorLevelMetrics]]
    
    # Continuous variable diagnostics (keyed by variable name)
    continuous_diagnostics: Dict[str, List[ContinuousBandMetrics]]


@dataclass
class TrainTestComparison:
    """Train metrics and optional test comparison."""
    
    # Train diagnostics (always present)
    train: DatasetDiagnostics
    
    # Test diagnostics (None if no test data provided)
    test: Optional[DatasetDiagnostics] = None
    
    # Comparison metrics (None if no test data)
    gini_gap: Optional[float] = None
    ae_ratio_diff: Optional[float] = None
    decile_comparison: Optional[List[Dict[str, Any]]] = None
    factor_divergence: Optional[Dict[str, List[Dict[str, Any]]]] = None
    
    # Flags (False if no test data)
    overfitting_risk: bool = False
    calibration_drift: bool = False
    unstable_factors: List[str] = field(default_factory=list)


# TrainTestMetrics removed - use DatasetDiagnostics instead


@dataclass
class ConvergenceDetails:
    """Details about model convergence."""
    max_iterations_allowed: int
    iterations_used: int
    converged: bool
    reason: str  # "converged", "max_iterations_reached", "gradient_tolerance", etc.


@dataclass
class DataExploration:
    """Pre-fit data exploration results."""
    
    # Data summary
    data_summary: Dict[str, Any]
    
    # Factor statistics
    factor_stats: List[Dict[str, Any]]
    
    # Missing value analysis
    missing_values: Dict[str, Any]
    
    # Univariate significance tests (each factor vs response)
    univariate_tests: List[Dict[str, Any]]
    
    # Correlation matrix for continuous factors
    correlations: Dict[str, Any]
    
    # Cramér's V matrix for categorical factors
    cramers_v: Dict[str, Any]
    
    # Variance inflation factors (multicollinearity)
    vif: List[Dict[str, Any]]
    
    # Zero inflation check (for count data)
    zero_inflation: Dict[str, Any]
    
    # Overdispersion check
    overdispersion: Dict[str, Any]
    
    # Interaction candidates
    interaction_candidates: List[InteractionCandidate]
    
    # Response distribution
    response_stats: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return _to_dict_recursive(self)
    
    def to_json(self, indent: Optional[int] = None) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=_json_default)


@dataclass
class ModelDiagnostics:
    """Complete model diagnostics output."""
    
    # Model metadata
    model_summary: Dict[str, Any]
    
    # Train/test metrics - SINGLE SOURCE OF TRUTH for loss, aic, gini, etc.
    train_test: TrainTestComparison
    
    # Calibration (A/E ratio, problem deciles)
    calibration: Dict[str, Any]
    
    # Residual summary
    residual_summary: Dict[str, ResidualSummary]
    
    # Per-factor diagnostics
    factors: List[FactorDiagnostics]
    
    # Interaction candidates
    interaction_candidates: List[InteractionCandidate]
    
    # Model comparison vs null
    model_comparison: Dict[str, float]
    
    # Warnings
    warnings: List[Dict[str, str]]
    
    # VIF / Multicollinearity scores
    vif: Optional[List[VIFResult]] = None
    
    # Coefficient summary with interpretations
    coefficient_summary: Optional[List[CoefficientSummary]] = None
    
    # Deviance breakdown by factor level
    factor_deviance: Optional[List[FactorDeviance]] = None
    
    # Full lift chart
    lift_chart: Optional[LiftChart] = None
    
    # Partial dependence plots
    partial_dependence: Optional[List[PartialDependence]] = None
    
    # Overdispersion diagnostics (for count/binomial data)
    overdispersion: Optional[Dict[str, Any]] = None
    
    # Spline knot information
    spline_info: Optional[Dict[str, Dict[str, Any]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, handling nested dataclasses."""
        return _to_dict_recursive(self)
    
    def to_json(self, indent: Optional[int] = None) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=_json_default)


def _json_default(obj):
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
    # Use fewer decimals for large numbers, more for small
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
    elif hasattr(obj, '__dataclass_fields__'):
        return {k: _to_dict_recursive(v) for k, v in asdict(obj).items()}
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


# =============================================================================
# Focused Diagnostic Components
# =============================================================================
#
# Each component handles a specific type of diagnostic computation.
# DiagnosticsComputer coordinates these components to produce unified output.
# =============================================================================

class _ResidualComputer:
    """Computes and caches residuals."""
    
    def __init__(self, y: np.ndarray, mu: np.ndarray, family: str, exposure: np.ndarray):
        self.y = y
        self.mu = mu
        self.family = family
        self.exposure = exposure
        self._pearson = None
        self._deviance = None
        self._null_dev = None
    
    @property
    def pearson(self) -> np.ndarray:
        if self._pearson is None:
            self._pearson = np.asarray(_rust_pearson_residuals(self.y, self.mu, self.family))
        return self._pearson
    
    @property
    def deviance(self) -> np.ndarray:
        if self._deviance is None:
            self._deviance = np.asarray(_rust_deviance_residuals(self.y, self.mu, self.family))
        return self._deviance
    
    @property
    def null_deviance(self) -> float:
        if self._null_dev is None:
            self._null_dev = _rust_null_deviance(self.y, self.family, self.exposure)
        return self._null_dev
    
    def unit_deviance(self, y: np.ndarray, mu: np.ndarray) -> np.ndarray:
        return np.asarray(_rust_unit_deviance(y, mu, self.family))


class _CalibrationComputer:
    """Computes calibration metrics."""
    
    def __init__(self, y: np.ndarray, mu: np.ndarray, exposure: np.ndarray):
        self.y = y
        self.mu = mu
        self.exposure = exposure
    
    def compute(self, n_bins: int = 10) -> Dict[str, Any]:
        actual_total = float(np.sum(self.y))
        predicted_total = float(np.sum(self.mu))
        exposure_total = float(np.sum(self.exposure))
        ae_ratio = actual_total / predicted_total if predicted_total > 0 else float('nan')
        
        bins = self._compute_bins(n_bins)
        hl_stat, hl_pvalue = self._hosmer_lemeshow(n_bins)
        
        # Compressed format: only include problem deciles (A/E outside [0.9, 1.1])
        problem_deciles = [
            {
                "decile": b.bin_index,
                "ae": round(b.actual_expected_ratio, 2),
                "n": b.count,
                "ae_ci": [round(b.ae_confidence_interval_lower, 2), round(b.ae_confidence_interval_upper, 2)],
            }
            for b in bins
            if b.actual_expected_ratio < 0.9 or b.actual_expected_ratio > 1.1
        ]
        
        return {
            "ae_ratio": round(ae_ratio, 3),
            "hl_pvalue": round(hl_pvalue, 4) if not np.isnan(hl_pvalue) else None,
            "problem_deciles": problem_deciles,
        }
    
    def _compute_bins(self, n_bins: int) -> List[CalibrationBin]:
        rust_bins = _rust_calibration_curve(self.y, self.mu, self.exposure, n_bins)
        return [
            CalibrationBin(
                bin_index=b["bin_index"], predicted_lower=b["predicted_lower"],
                predicted_upper=b["predicted_upper"], predicted_mean=b["predicted_mean"],
                actual_mean=b["actual_mean"], actual_expected_ratio=b["actual_expected_ratio"],
                count=b["count"], exposure=b["exposure"], actual_sum=b["actual_sum"],
                predicted_sum=b["predicted_sum"], ae_confidence_interval_lower=b["ae_ci_lower"],
                ae_confidence_interval_upper=b["ae_ci_upper"],
            )
            for b in rust_bins
        ]
    
    def _hosmer_lemeshow(self, n_bins: int) -> tuple:
        result = _rust_hosmer_lemeshow(self.y, self.mu, n_bins)
        return result["chi2_statistic"], result["pvalue"]


class _DiscriminationComputer:
    """Computes discrimination metrics."""
    
    def __init__(self, y: np.ndarray, mu: np.ndarray, exposure: np.ndarray):
        self.y = y
        self.mu = mu
        self.exposure = exposure
    
    def compute(self) -> Dict[str, Any]:
        stats = _rust_discrimination_stats(self.y, self.mu, self.exposure)
        # Removed lorenz_curve - Gini coefficient provides sufficient discrimination info
        return {
            "gini": round(stats["gini"], 3),
            "auc": round(stats["auc"], 3),
            "ks": round(stats["ks_statistic"], 3),
            "lift_10pct": round(stats["lift_at_10pct"], 3),
            "lift_20pct": round(stats["lift_at_20pct"], 3),
        }


# =============================================================================
# Main Diagnostics Computation
# =============================================================================

class DiagnosticsComputer:
    """
    Computes comprehensive model diagnostics.
    
    Coordinates focused component classes to produce unified diagnostics output.
    All results are cached for efficiency.
    """
    
    def __init__(
        self,
        y: np.ndarray,
        mu: np.ndarray,
        linear_predictor: np.ndarray,
        family: str,
        n_params: int,
        deviance: float,
        exposure: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        var_power: float = 1.5,
        theta: float = 1.0,
        null_deviance: Optional[float] = None,
    ):
        self.y = np.asarray(y, dtype=np.float64)
        self.mu = np.asarray(mu, dtype=np.float64)
        self.linear_predictor = np.asarray(linear_predictor, dtype=np.float64)
        self.family = family.lower()
        self.n_params = n_params
        self.deviance = deviance
        self._null_deviance_override = null_deviance  # From model result
        self.exposure = np.asarray(exposure, dtype=np.float64) if exposure is not None else np.ones_like(y)
        self.feature_names = feature_names or []
        self.var_power = var_power
        self.theta = theta
        
        self.n_obs = len(y)
        self.df_resid = self.n_obs - n_params
        
        # Initialize focused components
        self._residuals = _ResidualComputer(self.y, self.mu, self.family, self.exposure)
        self._calibration = _CalibrationComputer(self.y, self.mu, self.exposure)
        self._discrimination = _DiscriminationComputer(self.y, self.mu, self.exposure)
    
    @property
    def pearson_residuals(self) -> np.ndarray:
        return self._residuals.pearson
    
    @property
    def deviance_residuals(self) -> np.ndarray:
        return self._residuals.deviance
    
    @property
    def null_deviance(self) -> float:
        # Use override from model if provided, otherwise compute
        if self._null_deviance_override is not None:
            return self._null_deviance_override
        return self._residuals.null_deviance
    
    def _compute_unit_deviance(self, y: np.ndarray, mu: np.ndarray) -> np.ndarray:
        return self._residuals.unit_deviance(y, mu)
    
    def _compute_loss(self, y: np.ndarray, mu: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
        unit_dev = self._compute_unit_deviance(y, mu)
        if weights is not None:
            return np.average(unit_dev, weights=weights)
        return np.mean(unit_dev)
    
    def compute_fit_statistics(self) -> Dict[str, float]:
        """Compute overall fit statistics using Rust backend."""
        return _rust_fit_statistics(
            self.y, self.mu, self.deviance, self.null_deviance, self.n_params, self.family
        )
    
    def compute_loss_metrics(self) -> Dict[str, float]:
        """Compute various loss metrics using Rust backend."""
        rust_loss = _rust_loss_metrics(self.y, self.mu, self.family)
        return {
            "loss": rust_loss["family_loss"],  # Primary metric for model comparison
            "mse": rust_loss["mse"],
            "mae": rust_loss["mae"],
            "rmse": rust_loss["rmse"],
        }
    
    def compute_calibration(self, n_bins: int = 10) -> Dict[str, Any]:
        """Compute calibration metrics using focused component."""
        return self._calibration.compute(n_bins)
    
    def compute_discrimination(self) -> Optional[Dict[str, Any]]:
        """Compute discrimination metrics using focused component."""
        return self._discrimination.compute()
    
    def compute_residual_summary(self) -> Dict[str, ResidualSummary]:
        """Compute residual summary statistics using Rust backend (compressed)."""
        def summarize(resid: np.ndarray) -> ResidualSummary:
            stats = _rust_residual_summary(resid)
            return ResidualSummary(
                mean=round(stats["mean"], 2),
                std=round(stats["std"], 2),
                skewness=round(stats["skewness"], 1),
            )
        
        return {
            "pearson": summarize(self.pearson_residuals),
            "deviance": summarize(self.deviance_residuals),
        }
    
    def compute_factor_diagnostics(
        self,
        data: "pl.DataFrame",
        categorical_factors: List[str],
        continuous_factors: List[str],
        result=None,  # GLMResults for significance tests
        n_bins: int = 10,
        rare_threshold_pct: float = 1.0,
        max_categorical_levels: int = 20,
    ) -> List[FactorDiagnostics]:
        """Compute diagnostics for each specified factor."""
        factors = []
        
        # Process categorical factors
        for name in categorical_factors:
            if name not in data.columns:
                raise ValueError(f"Categorical factor '{name}' not found in data columns: {list(data.columns)}")
            
            values = data[name].to_numpy().astype(str)
            in_model = any(name in fn for fn in self.feature_names)
            
            # Univariate stats (compressed: no levels array, info is in actual_vs_expected)
            unique, counts = np.unique(values, return_counts=True)
            total = len(values)
            percentages = [100.0 * c / total for c in counts]
            
            n_rare = sum(1 for pct in percentages if pct < rare_threshold_pct)
            rare_pct = sum(pct for pct in percentages if pct < rare_threshold_pct)
            
            univariate = CategoricalFactorStats(
                n_levels=len(unique),
                n_rare_levels=n_rare,
                rare_level_total_pct=round(rare_pct, 2),
            )
            
            # A/E by level
            ae_bins = self._compute_ae_categorical(
                values, rare_threshold_pct, max_categorical_levels
            )
            
            # Residual pattern
            resid_pattern = self._compute_residual_pattern_categorical(values)
            
            # Factor significance (only for factors in model)
            significance = self.compute_factor_significance(name, result) if in_model and result else None
            
            # Extract coefficients for this factor
            coefficients = self._get_factor_coefficients(name, result) if in_model and result else None
            
            factors.append(FactorDiagnostics(
                name=name,
                factor_type="categorical",
                in_model=in_model,
                transform=self._get_transformation(name),
                coefficients=coefficients,
                univariate=univariate,
                actual_vs_expected=ae_bins,
                residual_pattern=resid_pattern,
                significance=significance,
            ))
        
        # Process continuous factors
        for name in continuous_factors:
            if name not in data.columns:
                raise ValueError(f"Continuous factor '{name}' not found in data columns: {list(data.columns)}")
            
            values = data[name].to_numpy().astype(np.float64)
            in_model = any(name in fn for fn in self.feature_names)
            
            # Univariate stats - batch percentile calculation
            valid_mask = ~np.isnan(values) & ~np.isinf(values)
            valid = values[valid_mask]
            
            if len(valid) > 0:
                # Single batched percentile call (much faster)
                # Token optimization: compact array [p1, p5, p10, p25, p50, p75, p90, p95, p99]
                pcts = np.percentile(valid, [1, 5, 10, 25, 50, 75, 90, 95, 99])
                univariate = ContinuousFactorStats(
                    mean=float(np.mean(valid)),
                    std=float(np.std(valid)),
                    min=float(np.min(valid)),
                    max=float(np.max(valid)),
                    missing_count=int(np.sum(~valid_mask)),
                    percentiles=[round(float(p), 2) for p in pcts],
                )
            else:
                univariate = ContinuousFactorStats(
                    mean=float('nan'), std=float('nan'), min=float('nan'), max=float('nan'),
                    missing_count=len(values), percentiles=[]
                )
            
            # A/E by quantile bins
            ae_bins = self._compute_ae_continuous(values, n_bins)
            
            # Residual pattern
            resid_pattern = self._compute_residual_pattern_continuous(values, n_bins)
            
            # Factor significance (only for factors in model)
            significance = self.compute_factor_significance(name, result) if in_model and result else None
            
            # Extract coefficients for this factor
            coefficients = self._get_factor_coefficients(name, result) if in_model and result else None
            
            factors.append(FactorDiagnostics(
                name=name,
                factor_type="continuous",
                in_model=in_model,
                transform=self._get_transformation(name),
                coefficients=coefficients,
                univariate=univariate,
                actual_vs_expected=ae_bins,
                residual_pattern=resid_pattern,
                significance=significance,
            ))
        
        return factors
    
    def _get_transformation(self, name: str) -> Optional[str]:
        """Find transformation for a factor in the model."""
        for fn in self.feature_names:
            if name in fn and fn != name:
                return fn
        return None
    
    def _get_factor_terms(self, name: str) -> List[str]:
        """Get all model terms that include this factor."""
        return [fn for fn in self.feature_names if name in fn]
    
    def _get_factor_coefficients(self, name: str, result) -> Optional[List[FactorCoefficient]]:
        """Extract coefficients for all terms involving this factor."""
        if result is None or not hasattr(result, 'params'):
            return None
        
        try:
            # Get params as array
            params = result.params
            if callable(params):
                params = params()
            if hasattr(params, 'tolist'):
                params = params.tolist() if hasattr(params, 'tolist') else list(params)
            
            feature_names = result.feature_names if hasattr(result, 'feature_names') else self.feature_names
            
            # Get standard errors if available (may be method or property)
            bse = None
            if hasattr(result, 'bse'):
                bse = result.bse
                if callable(bse):
                    bse = bse()
            elif hasattr(result, 'std_errors'):
                bse = result.std_errors
                if callable(bse):
                    bse = bse()
            
            # Get p-values if available
            pvalues = None
            if hasattr(result, 'pvalues'):
                pvalues = result.pvalues
                if callable(pvalues):
                    pvalues = pvalues()
            
            # Check if log-link for relativity calculation
            link = result.link if hasattr(result, 'link') else self.link
            is_log_link = link in ('log', 'Log')
            
            coefficients = []
            for i, fn in enumerate(feature_names):
                # Check if this term involves the factor (but not interactions)
                if name in fn and ':' not in fn and fn != 'Intercept':
                    coef = float(params[i])
                    se = float(bse[i]) if bse is not None else 0.0
                    z_val = coef / se if se > 0 else 0.0
                    p_val = float(pvalues[i]) if pvalues is not None else (2 * (1 - min(0.9999, abs(z_val) / 4)))
                    
                    rel = float(np.exp(coef)) if is_log_link else None
                    
                    coefficients.append(FactorCoefficient(
                        term=fn,
                        estimate=round(coef, 6),
                        std_error=round(se, 6),
                        z_value=round(z_val, 3),
                        p_value=round(p_val, 4),
                        relativity=round(rel, 4) if rel else None,
                    ))
            
            return coefficients if coefficients else None
        except Exception as e:
            raise RuntimeError(f"Failed to extract coefficient table: {e}") from e
    
    def compute_factor_significance(
        self,
        name: str,
        result,  # GLMResults or FormulaGLMResults
    ) -> Optional[FactorSignificance]:
        """
        Compute significance tests for a factor in the model.
        
        Returns Wald chi-square test and deviance contribution.
        """
        if not hasattr(result, 'params') or not hasattr(result, 'bse'):
            return None
        
        # Find indices of parameters related to this factor
        param_indices = []
        for i, fn in enumerate(self.feature_names):
            if name in fn and fn != 'Intercept':
                param_indices.append(i)
        
        if not param_indices:
            return None
        
        try:
            params = np.asarray(result.params)
            bse = np.asarray(result.bse())
            
            # Wald chi-square: sum of (coef/se)^2 for all related parameters
            wald_chi2 = 0.0
            for idx in param_indices:
                if bse[idx] > 0:
                    wald_chi2 += (params[idx] / bse[idx]) ** 2
            
            # Degrees of freedom = number of parameters for this term
            df = len(param_indices)
            
            # P-value from chi-square distribution (using Rust CDF)
            wald_pvalue = 1 - _chi2_cdf(wald_chi2, float(df)) if df > 0 else 1.0
            
            # Deviance contribution: approximate using sum of z^2 (scaled)
            # This is an approximation; true drop-in-deviance requires refitting
            deviance_contribution = float(wald_chi2)  # Approximate
            
            return FactorSignificance(
                chi2=round(float(wald_chi2), 2),
                p=round(float(wald_pvalue), 4),
                dev_contrib=round(deviance_contribution, 2),
            )
        except Exception as e:
            # Re-raise to surface bugs - factor significance computation shouldn't fail silently
            raise RuntimeError(f"Failed to compute factor significance for '{factor_name}': {e}") from e
    
    def _compute_ae_continuous(self, values: np.ndarray, n_bins: int) -> List[ActualExpectedBin]:
        """Compute A/E for continuous factor using Rust backend (compact format)."""
        rust_bins = _rust_ae_continuous(values, self.y, self.mu, self.exposure, n_bins, self.family)
        # Filter out empty bins (count=0)
        non_empty_bins = [b for b in rust_bins if b["count"] > 0]
        return [
            ActualExpectedBin(
                bin=b["bin_label"],
                n=b["count"],
                exposure=round(b["exposure"], 2),
                actual=round(b["actual_sum"] / b["exposure"], 6) if b["exposure"] > 0 else 0.0,
                expected=round(b["predicted_sum"] / b["exposure"], 6) if b["exposure"] > 0 else 0.0,
                ae_ratio=round(b["actual_expected_ratio"], 2),
                ae_ci=[round(b["ae_ci_lower"], 2), round(b["ae_ci_upper"], 2)],
            )
            for b in non_empty_bins
        ]
    
    def _compute_ae_categorical(
        self,
        values: np.ndarray,
        rare_threshold_pct: float,
        max_levels: int,
    ) -> List[ActualExpectedBin]:
        """Compute A/E for categorical factor using Rust backend (compact format)."""
        levels = [str(v) for v in values]
        rust_bins = _rust_ae_categorical(levels, self.y, self.mu, self.exposure, 
                                          rare_threshold_pct, max_levels, self.family)
        return [
            ActualExpectedBin(
                bin=b["bin_label"],
                n=b["count"],
                exposure=round(b["exposure"], 2),
                actual=round(b["actual_sum"] / b["exposure"], 6) if b["exposure"] > 0 else 0.0,
                expected=round(b["predicted_sum"] / b["exposure"], 6) if b["exposure"] > 0 else 0.0,
                ae_ratio=round(b["actual_expected_ratio"], 2),
                ae_ci=[round(b["ae_ci_lower"], 2), round(b["ae_ci_upper"], 2)],
            )
            for b in rust_bins
        ]
    
    def _compute_residual_pattern_continuous(
        self,
        values: np.ndarray,
        n_bins: int,
    ) -> ResidualPattern:
        """Compute residual pattern using Rust backend (compressed: no mean_by_bin)."""
        valid_mask = ~np.isnan(values) & ~np.isinf(values)
        
        if not np.any(valid_mask):
            return ResidualPattern(resid_corr=0.0, var_explained=0.0)
        
        result = _rust_residual_pattern(values, self.pearson_residuals, n_bins)
        corr = result["correlation_with_residuals"]
        corr_val = float(corr) if not np.isnan(corr) else 0.0
        
        return ResidualPattern(
            resid_corr=round(corr_val, 4),
            var_explained=round(corr_val ** 2, 6),
        )
    
    def _compute_residual_pattern_categorical(self, values: np.ndarray) -> ResidualPattern:
        """Compute residual pattern for categorical factor (compressed)."""
        # Use pandas groupby for vectorized computation (faster than Python loop)
        import pandas as pd
        
        df = pd.DataFrame({'level': values, 'resid': self.pearson_residuals})
        
        # Compute group means in one vectorized operation
        group_stats = df.groupby('level')['resid'].agg(['mean', 'count'])
        level_means = group_stats['mean'].values
        level_counts = group_stats['count'].values
        
        # Compute eta-squared (variance explained)
        overall_mean = np.mean(self.pearson_residuals)
        ss_total = np.sum((self.pearson_residuals - overall_mean) ** 2)
        ss_between = np.sum(level_counts * (level_means - overall_mean) ** 2)
        
        eta_squared = ss_between / ss_total if ss_total > 0 else 0.0
        mean_abs_resid = np.mean(np.abs(level_means))
        
        return ResidualPattern(
            resid_corr=round(float(mean_abs_resid), 4),
            var_explained=round(float(eta_squared), 6),
        )
    
    def _linear_trend_test(self, x: np.ndarray, y: np.ndarray) -> tuple:
        """Simple linear regression trend test."""
        n = len(x)
        if n < 3:
            return float('nan'), float('nan')
        
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        ss_xx = np.sum((x - x_mean) ** 2)
        ss_xy = np.sum((x - x_mean) * (y - y_mean))
        
        if ss_xx == 0:
            return 0.0, 1.0
        
        slope = ss_xy / ss_xx
        
        # Residuals from regression
        y_pred = y_mean + slope * (x - x_mean)
        ss_res = np.sum((y - y_pred) ** 2)
        
        df = n - 2
        mse = ss_res / df if df > 0 else 0
        se_slope = np.sqrt(mse / ss_xx) if mse > 0 and ss_xx > 0 else float('nan')
        
        if np.isnan(se_slope) or se_slope == 0:
            return slope, float('nan')
        
        t_stat = slope / se_slope
        
        # P-value from t-distribution (using Rust CDF)
        pvalue = 2 * (1 - _t_cdf(abs(t_stat), float(df)))
        
        return slope, pvalue
    
    def detect_interactions(
        self,
        data: "pl.DataFrame",
        factor_names: List[str],
        max_factors: int = 10,
        min_correlation: float = 0.01,
        max_candidates: int = 5,
        min_cell_count: int = 30,
    ) -> List[InteractionCandidate]:
        """Detect potential interactions using greedy residual-based approach."""
        # First, rank factors by residual association
        factor_scores = []
        
        for name in factor_names:
            if name not in data.columns:
                raise ValueError(f"Factor '{name}' not found in data columns: {list(data.columns)}")
            
            values = data[name].to_numpy()
            
            # Check if categorical or continuous
            if values.dtype == object or str(values.dtype).startswith('str'):
                score = self._compute_eta_squared(values.astype(str))
            else:
                values = values.astype(np.float64)
                valid_mask = ~np.isnan(values) & ~np.isinf(values)
                if np.sum(valid_mask) < 10:
                    continue
                score = abs(np.corrcoef(values[valid_mask], self.pearson_residuals[valid_mask])[0, 1])
            
            if score >= min_correlation:
                factor_scores.append((name, score))
        
        # Sort and take top factors
        factor_scores.sort(key=lambda x: -x[1])
        top_factors = [name for name, _ in factor_scores[:max_factors]]
        
        if len(top_factors) < 2:
            return []
        
        # Check pairwise interactions
        candidates = []
        
        for i in range(len(top_factors)):
            for j in range(i + 1, len(top_factors)):
                name1, name2 = top_factors[i], top_factors[j]
                
                values1 = data[name1].to_numpy()
                values2 = data[name2].to_numpy()
                
                # Discretize both factors
                bins1 = self._discretize(values1, 5)
                bins2 = self._discretize(values2, 5)
                
                # Compute interaction strength
                candidate = self._compute_interaction_strength(
                    name1, bins1, name2, bins2, min_cell_count
                )
                
                if candidate is not None:
                    # Add current_terms and recommendation
                    terms1 = self._get_factor_terms(name1)
                    terms2 = self._get_factor_terms(name2)
                    candidate.current_terms = terms1 + terms2 if (terms1 or terms2) else None
                    
                    # Generate recommendation based on current terms and factor types
                    candidate.recommendation = self._generate_interaction_recommendation(
                        name1, name2, terms1, terms2, values1, values2
                    )
                    candidates.append(candidate)
        
        # Sort by strength and return top candidates
        candidates.sort(key=lambda x: -x.interaction_strength)
        return candidates[:max_candidates]
    
    def _generate_interaction_recommendation(
        self,
        name1: str,
        name2: str,
        terms1: List[str],
        terms2: List[str],
        values1: np.ndarray,
        values2: np.ndarray,
    ) -> str:
        """Generate a recommendation for how to model an interaction."""
        is_cat1 = values1.dtype == object or str(values1.dtype).startswith('str')
        is_cat2 = values2.dtype == object or str(values2.dtype).startswith('str')
        
        # Check if factors have spline/polynomial terms
        has_spline1 = any('bs(' in t or 'ns(' in t or 's(' in t for t in terms1)
        has_spline2 = any('bs(' in t or 'ns(' in t or 's(' in t for t in terms2)
        has_poly1 = any('I(' in t and '**' in t for t in terms1)
        has_poly2 = any('I(' in t and '**' in t for t in terms2)
        
        if is_cat1 and is_cat2:
            return f"Consider C({name1}):C({name2}) interaction term"
        elif is_cat1 and not is_cat2:
            if has_spline2:
                return f"Consider C({name1}):{name2} or separate splines by {name1} level"
            else:
                return f"Consider C({name1}):{name2} interaction term"
        elif not is_cat1 and is_cat2:
            if has_spline1:
                return f"Consider {name1}:C({name2}) or separate splines by {name2} level"
            else:
                return f"Consider {name1}:C({name2}) interaction term"
        else:
            # Both continuous
            if has_spline1 or has_spline2 or has_poly1 or has_poly2:
                return f"Consider {name1}:{name2} or tensor product spline"
            else:
                return f"Consider {name1}:{name2} interaction or joint spline"
    
    def _compute_eta_squared(self, categories: np.ndarray) -> float:
        """Compute eta-squared for categorical association with residuals."""
        unique_levels = np.unique(categories)
        overall_mean = np.mean(self.pearson_residuals)
        ss_total = np.sum((self.pearson_residuals - overall_mean) ** 2)
        
        if ss_total == 0:
            return 0.0
        
        ss_between = 0.0
        for level in unique_levels:
            mask = categories == level
            level_resid = self.pearson_residuals[mask]
            level_mean = np.mean(level_resid)
            ss_between += len(level_resid) * (level_mean - overall_mean) ** 2
        
        return ss_between / ss_total
    
    def _discretize(self, values: np.ndarray, n_bins: int) -> np.ndarray:
        """Discretize values into bins."""
        if values.dtype == object or str(values.dtype).startswith('str'):
            # Categorical - map to integers
            unique_vals = np.unique(values)
            mapping = {v: i for i, v in enumerate(unique_vals)}
            return np.array([mapping[v] for v in values])
        else:
            # Continuous - quantile bins
            values = values.astype(np.float64)
            valid_mask = ~np.isnan(values) & ~np.isinf(values)
            
            if not np.any(valid_mask):
                return np.zeros(len(values), dtype=int)
            
            quantiles = np.percentile(values[valid_mask], np.linspace(0, 100, n_bins + 1))
            bins = np.digitize(values, quantiles[1:-1])
            bins[~valid_mask] = n_bins  # Invalid values in separate bin
            return bins
    
    def _compute_interaction_strength(
        self,
        name1: str,
        bins1: np.ndarray,
        name2: str,
        bins2: np.ndarray,
        min_cell_count: int,
    ) -> Optional[InteractionCandidate]:
        """Compute interaction strength between two discretized factors."""
        # Create interaction cells
        cell_ids = bins1 * 1000 + bins2  # Unique cell ID
        unique_cells = np.unique(cell_ids)
        
        # Filter cells with sufficient data
        valid_cells = []
        cell_residuals = []
        
        for cell_id in unique_cells:
            mask = cell_ids == cell_id
            if np.sum(mask) >= min_cell_count:
                valid_cells.append(cell_id)
                cell_residuals.append(self.pearson_residuals[mask])
        
        if len(valid_cells) < 4:
            return None
        
        # Compute variance explained by cells
        all_resid = np.concatenate(cell_residuals)
        overall_mean = np.mean(all_resid)
        ss_total = np.sum((all_resid - overall_mean) ** 2)
        
        if ss_total == 0:
            return None
        
        ss_model = sum(
            len(r) * (np.mean(r) - overall_mean) ** 2
            for r in cell_residuals
        )
        
        r_squared = ss_model / ss_total
        
        # F-test p-value
        df_model = len(valid_cells) - 1
        df_resid = len(all_resid) - len(valid_cells)
        
        if df_model > 0 and df_resid > 0:
            f_stat = (ss_model / df_model) / ((ss_total - ss_model) / df_resid)
            
            # P-value from F-distribution (using Rust CDF)
            pvalue = 1 - _f_cdf(f_stat, float(df_model), float(df_resid))
        else:
            pvalue = float('nan')
        
        return InteractionCandidate(
            factor1=name1,
            factor2=name2,
            interaction_strength=float(r_squared),
            pvalue=float(pvalue),
            n_cells=len(valid_cells),
        )
    
    def compute_model_comparison(self) -> Dict[str, float]:
        """Compute model comparison statistics vs null model."""
        null_dev = self.null_deviance
        
        # Likelihood ratio test
        lr_chi2 = null_dev - self.deviance
        lr_df = self.n_params - 1
        
        # P-value from chi-square distribution (using Rust CDF)
        lr_pvalue = 1 - _chi2_cdf(lr_chi2, float(lr_df)) if lr_df > 0 else float('nan')
        
        deviance_reduction_pct = 100 * (1 - self.deviance / null_dev) if null_dev > 0 else 0
        
        # AIC improvement
        null_aic = null_dev + 2  # Null model has 1 parameter
        model_aic = self.deviance + 2 * self.n_params
        aic_improvement = null_aic - model_aic
        
        return {
            "likelihood_ratio_chi2": float(lr_chi2),
            "likelihood_ratio_df": lr_df,
            "likelihood_ratio_pvalue": float(lr_pvalue),
            "deviance_reduction_pct": float(deviance_reduction_pct),
            "aic_improvement": float(aic_improvement),
        }
    
    def generate_warnings(
        self,
        fit_stats: Dict[str, float],
        calibration: Dict[str, Any],
        factors: List[FactorDiagnostics],
        family: str = "",
    ) -> List[Dict[str, str]]:
        """Generate warnings based on diagnostics."""
        warnings = []
        
        # NegBin-specific warnings
        family_lower = family.lower() if family else ""
        if family_lower.startswith("negativebinomial"):
            # Regularization warning
            warnings.append({
                "type": "negbinomial_regularization",
                "message": "Negative binomial fitting applies minimum ridge regularization (alpha=1e-6) for numerical stability. Coefficient bias is negligible but inference is approximate."
            })
            
            # Large theta warning (essentially Poisson)
            if "theta=" in family:
                try:
                    theta_str = family.split("theta=")[1].rstrip(")")
                    theta = float(theta_str)
                    if theta >= 100:
                        warnings.append({
                            "type": "negbinomial_large_theta",
                            "message": f"Estimated theta={theta:.1f} is very large, suggesting minimal overdispersion. Consider using Poisson instead for simpler interpretation."
                        })
                    elif theta <= 0.1:
                        warnings.append({
                            "type": "negbinomial_small_theta",
                            "message": f"Estimated theta={theta:.4f} is very small, indicating severe overdispersion. Check for missing covariates or consider zero-inflated models."
                        })
                except (ValueError, IndexError) as e:
                    # Theta parsing failed - this is a bug in family string formatting
                    raise RuntimeError(f"Failed to parse theta from family string '{family}': {e}") from e
        
        # High dispersion warning
        dispersion = fit_stats.get("dispersion", 1.0)
        if dispersion > 1.5:
            warnings.append({
                "type": "high_dispersion",
                "message": f"Dispersion {dispersion:.2f} suggests overdispersion. Consider quasipoisson or negbinomial."
            })
        
        # Poor overall calibration
        ae_ratio = calibration.get("ae_ratio", 1.0)
        if abs(ae_ratio - 1.0) > 0.05:
            direction = "over" if ae_ratio < 1 else "under"
            warnings.append({
                "type": "poor_calibration",
                "message": f"Model {direction}-predicts overall (A/E = {ae_ratio:.3f})."
            })
        
        # Token optimization: skip per-decile warnings (problem_deciles in calibration has this info)
        
        # Factors with high residual correlation (not in model)
        for factor in factors:
            if not factor.in_model:
                r2 = factor.residual_pattern.var_explained
                if r2 > 0.02:
                    warnings.append({
                        "type": "missing_factor",
                        "message": f"Factor '{factor.name}' not in model but explains {100*r2:.1f}% of residual variance."
                    })
        
        return warnings
    
    # =========================================================================
    # NEW: Enhanced diagnostics for agentic workflows
    # =========================================================================
    
    def compute_vif(
        self,
        X: np.ndarray,
        feature_names: List[str],
        threshold_moderate: float = 5.0,
        threshold_severe: float = 10.0,
    ) -> List[VIFResult]:
        """
        Compute Variance Inflation Factors for design matrix columns.
        
        Uses correlation matrix inverse for O(k³) complexity instead of
        O(k × n × k²) for k features and n observations.
        
        VIF detects multicollinearity which can cause:
        - Unstable coefficient estimates
        - Inflated standard errors
        - Failed matrix inversions (like VehPower + bs(VehPower, df=4))
        
        Parameters
        ----------
        X : np.ndarray
            Design matrix (n_obs, n_features)
        feature_names : list of str
            Names of features in X
        threshold_moderate : float
            VIF above this indicates moderate multicollinearity
        threshold_severe : float
            VIF above this indicates severe multicollinearity
            
        Returns
        -------
        list of VIFResult
            VIF for each feature, sorted by VIF (highest first)
        """
        n_obs, n_features = X.shape
        results = []
        
        # Skip intercept column if present
        has_intercept = feature_names and feature_names[0] == "Intercept"
        start_idx = 1 if has_intercept else 0
        
        if n_features - start_idx <= 1:
            # Only one feature (besides intercept), VIF = 1
            for i in range(start_idx, n_features):
                results.append(VIFResult(
                    feature=feature_names[i] if i < len(feature_names) else f"X{i}",
                    vif=1.0, severity="none", collinear_with=None
                ))
            return results
        
        # Extract non-intercept columns
        X_no_int = X[:, start_idx:]
        names_no_int = feature_names[start_idx:] if feature_names else [f"X{i}" for i in range(start_idx, n_features)]
        k = X_no_int.shape[1]
        
        # Fast VIF via correlation matrix inverse
        # VIF_j = diag((R^{-1}))_j where R is correlation matrix
        try:
            # Center and scale columns (standardize to get correlation matrix)
            means = np.mean(X_no_int, axis=0)
            stds = np.std(X_no_int, axis=0, ddof=0)
            stds[stds == 0] = 1.0  # Avoid division by zero
            X_std = (X_no_int - means) / stds
            
            # Correlation matrix R = X'X / n
            R = (X_std.T @ X_std) / n_obs
            
            # Add small regularization for numerical stability
            R += np.eye(k) * 1e-10
            
            # VIF = diagonal of R^{-1}
            R_inv = np.linalg.inv(R)
            vif_values = np.diag(R_inv)
            
            # Also compute correlation matrix for finding collinear pairs
            corr_matrix = R - np.eye(k) * 1e-10  # Remove regularization for reporting
            
        except np.linalg.LinAlgError as e:
            raise RuntimeError(
                f"VIF computation failed: design matrix is singular. "
                f"This indicates severe multicollinearity - some columns are exact linear "
                f"combinations of others. Check for duplicate or constant columns."
            ) from e
        
        # Build results
        for i in range(k):
            feature_name = names_no_int[i] if i < len(names_no_int) else f"X{i}"
            vif = vif_values[i]
            
            # Determine severity
            if np.isnan(vif) or np.isinf(vif) or vif > 100:
                severity = "severe"
                vif = 999.0 if np.isnan(vif) or np.isinf(vif) else vif
            elif vif > threshold_severe:
                severity = "severe"
            elif vif > threshold_moderate:
                severity = "moderate"
            else:
                severity = "none"
            
            # Find most correlated features (only for problematic ones)
            collinear_with = None
            if severity != "none":
                correlations = []
                for j in range(k):
                    if j != i:
                        corr = corr_matrix[i, j]
                        if not np.isnan(corr) and abs(corr) > 0.5:
                            correlations.append((names_no_int[j], abs(corr)))
                correlations.sort(key=lambda x: -x[1])
                collinear_with = [c[0] for c in correlations[:3]]  # Top 3
            
            results.append(VIFResult(
                feature=feature_name,
                vif=round(float(vif), 2),
                severity=severity,
                collinear_with=collinear_with if collinear_with else None,
            ))
        
        # Sort by VIF (highest first) and filter to only problematic ones
        results.sort(key=lambda x: -x.vif if not np.isnan(x.vif) else 0)
        # Token optimization: only return severe/moderate VIF (agent doesn't need "none")
        return [r for r in results if r.severity in ("severe", "moderate")]
    
    def compute_coefficient_summary(
        self,
        result,  # GLMResults or FormulaGLMResults
        link: str = "log",
    ) -> List[CoefficientSummary]:
        """
        Compute coefficient summary with interpretations for agent use.
        
        Token-optimized compact format with shortened field names.
        Agent can infer impact from z-value sign and relativity magnitude.
        
        Returns
        -------
        list of CoefficientSummary
            Summary for each coefficient, sorted by absolute z-value
        """
        params = np.asarray(result.params)
        bse = np.asarray(result.bse())
        tvalues = np.asarray(result.tvalues())
        pvalues = np.asarray(result.pvalues())
        ci = np.asarray(result.conf_int(0.05))
        
        feature_names = self.feature_names if self.feature_names else [f"X{i}" for i in range(len(params))]
        
        summaries = []
        for i, name in enumerate(feature_names):
            coef_val = float(params[i])
            se_val = float(bse[i])
            z_val = float(tvalues[i])
            p_val = float(pvalues[i])
            
            # Relativity for log-link models
            rel = None
            rel_ci = None
            if link == "log":
                rel = round(float(np.exp(coef_val)), 4)
                rel_ci = [round(float(np.exp(ci[i, 0])), 4), round(float(np.exp(ci[i, 1])), 4)]
            
            summaries.append(CoefficientSummary(
                feature=name,
                estimate=round(coef_val, 6),
                std_error=round(se_val, 6),
                z_value=round(z_val, 3),
                p_value=round(p_val, 4),
                significant=p_val < 0.05,
                relativity=rel,
                relativity_ci=rel_ci,
            ))
        
        # Sort by absolute z-value (most significant first), but keep Intercept at end
        intercept = [s for s in summaries if s.feature == "Intercept"]
        others = [s for s in summaries if s.feature != "Intercept"]
        others.sort(key=lambda x: -abs(x.z_value))
        return others + intercept
    
    def compute_factor_deviance(
        self,
        data: "pl.DataFrame",
        categorical_factors: List[str],
    ) -> List[FactorDeviance]:
        """
        Compute deviance breakdown by factor level.
        
        Uses Rust backend for fast groupby aggregation on large datasets.
        
        Identifies which categorical levels are driving poor fit,
        helping the agent pinpoint problem areas.
        
        Returns
        -------
        list of FactorDeviance
            Deviance breakdown for each categorical factor
        """
        from rustystats._rustystats import compute_factor_deviance_py as _rust_factor_deviance
        
        results = []
        for factor_name in categorical_factors:
            if factor_name not in data.columns:
                continue
            
            values = [str(v) for v in data[factor_name].to_list()]
            
            # Call Rust for fast computation
            rust_result = _rust_factor_deviance(
                factor_name,
                values,
                self.y,
                self.mu,
                self.family,
                getattr(self, 'var_power', 1.5),
                getattr(self, 'theta', 1.0),
            )
            
            # Convert Rust result to Python dataclasses
            levels = [
                DevianceByLevel(
                    level=level["level"],
                    n=level["count"],
                    deviance=round(level["deviance"], 2),
                    deviance_pct=round(level["deviance_pct"], 2),
                    mean_deviance=round(level["mean_deviance"], 4),
                    ae_ratio=round(level["ae_ratio"], 3) if not np.isnan(level["ae_ratio"]) else None,
                    problem=level["is_problem"],
                )
                for level in rust_result["levels"]
            ]
            
            results.append(FactorDeviance(
                factor=factor_name,
                total_deviance=round(rust_result["total_deviance"], 2),
                levels=levels,
                problem_levels=rust_result["problem_levels"],
            ))
        
        return results
    
    def compute_lift_chart(self, n_deciles: int = 10) -> LiftChart:
        """
        Compute full lift chart with all deciles.
        
        Shows where the model discriminates well vs poorly,
        helping the agent identify risk bands needing attention.
        
        Returns
        -------
        LiftChart
            Complete lift chart with discrimination metrics
        """
        # Sort by predicted values
        sort_idx = np.argsort(self.mu)
        y_sorted = self.y[sort_idx]
        mu_sorted = self.mu[sort_idx]
        exp_sorted = self.exposure[sort_idx]
        
        # Overall rate
        overall_rate = np.sum(self.y) / np.sum(self.exposure)
        
        # Compute deciles
        n = len(self.y)
        decile_size = n // n_deciles
        
        deciles = []
        cumulative_actual = 0
        cumulative_predicted = 0
        total_actual = np.sum(self.y)
        total_predicted = np.sum(self.mu)
        
        max_ks = 0
        ks_decile = 1
        weak_deciles = []
        
        for d in range(n_deciles):
            start = d * decile_size
            end = (d + 1) * decile_size if d < n_deciles - 1 else n
            
            y_d = y_sorted[start:end]
            mu_d = mu_sorted[start:end]
            exp_d = exp_sorted[start:end]
            
            actual = float(np.sum(y_d))
            predicted = float(np.sum(mu_d))
            exposure = float(np.sum(exp_d))
            n_d = len(y_d)
            
            ae_ratio = actual / predicted if predicted > 0 else float('nan')
            
            cumulative_actual += actual
            cumulative_predicted += predicted
            
            cum_actual_pct = 100 * cumulative_actual / total_actual if total_actual > 0 else 0
            cum_pred_pct = 100 * cumulative_predicted / total_predicted if total_predicted > 0 else 0
            
            # Lift: rate in this decile / overall rate
            decile_rate = actual / exposure if exposure > 0 else 0
            lift = decile_rate / overall_rate if overall_rate > 0 else 1.0
            
            # Cumulative lift
            cum_rate = cumulative_actual / np.sum(exp_sorted[:end]) if np.sum(exp_sorted[:end]) > 0 else 0
            cum_lift = cum_rate / overall_rate if overall_rate > 0 else 1.0
            
            # KS statistic
            ks = abs(cum_actual_pct - cum_pred_pct)
            if ks > max_ks:
                max_ks = ks
                ks_decile = d + 1
            
            # Weak deciles: poor A/E or lift close to 1
            if abs(ae_ratio - 1.0) > 0.2 or (d < 3 and lift > 0.8) or (d > 6 and lift < 1.2):
                weak_deciles.append(d + 1)
            
            predicted_rate = predicted / exposure if exposure > 0 else 0
            deciles.append(LiftDecile(
                decile=d + 1,
                n=n_d,
                exposure=round(exposure, 2),
                actual=round(decile_rate, 6),
                predicted=round(predicted_rate, 6),
                ae_ratio=round(ae_ratio, 3) if not np.isnan(ae_ratio) else None,
                cumulative_actual_pct=round(cum_actual_pct, 2),
                cumulative_predicted_pct=round(cum_pred_pct, 2),
                lift=round(lift, 3),
                cumulative_lift=round(cum_lift, 3),
            ))
        
        # Compute Gini
        gini = 2 * max_ks / 100  # Approximate from KS
        stats = _rust_discrimination_stats(self.y, self.mu, self.exposure)
        gini = float(stats["gini"])
        
        return LiftChart(
            deciles=deciles,
            gini=round(gini, 3),
            ks_statistic=round(max_ks, 2),
            ks_decile=ks_decile,
            weak_deciles=weak_deciles,
        )
    
    def compute_partial_dependence(
        self,
        data: "pl.DataFrame",
        result,  # GLMResults with predict capability
        continuous_factors: List[str],
        categorical_factors: List[str],
        link: str = "log",
        n_grid: int = 20,
    ) -> List[PartialDependence]:
        """
        Compute partial dependence for each variable.
        
        Shows the marginal effect shape, helping the agent decide
        between linear, spline, or banding approaches.
        
        Returns
        -------
        list of PartialDependence
            Partial dependence for each variable
        """
        results = []
        
        # Continuous variables
        for var in continuous_factors:
            if var not in data.columns:
                continue
            
            values = data[var].to_numpy().astype(np.float64)
            valid_mask = ~np.isnan(values) & ~np.isinf(values)
            valid_values = values[valid_mask]
            
            if len(valid_values) < 10:
                continue
            
            # Create grid
            grid = np.linspace(np.percentile(valid_values, 1), 
                              np.percentile(valid_values, 99), n_grid)
            
            predictions = []
            for g in grid:
                # Mean prediction if we set this variable to g
                # Use the coefficient to approximate partial effect
                var_idx = None
                for i, name in enumerate(self.feature_names):
                    if var == name or var in name:
                        var_idx = i
                        break
                
                if var_idx is not None:
                    # Linear approximation using coefficient
                    coef = result.params[var_idx]
                    base_pred = np.mean(self.mu)
                    if link == "log":
                        pred = base_pred * np.exp(coef * (g - np.mean(valid_values)))
                    else:
                        pred = base_pred + coef * (g - np.mean(valid_values))
                    predictions.append(float(pred))
                else:
                    predictions.append(float(np.mean(self.mu)))
            
            # Analyze shape
            shape, recommendation = self._analyze_pd_shape(grid, predictions, link)
            
            # Convert to relativities for log-link
            relativities = None
            if link == "log" and predictions:
                base = predictions[len(predictions)//2]
                relativities = [p/base if base > 0 else 1.0 for p in predictions]
            
            results.append(PartialDependence(
                variable=var,
                variable_type="continuous",
                grid_values=[round(float(g), 4) for g in grid],
                predictions=[round(p, 6) for p in predictions],
                relativities=[round(r, 4) for r in relativities] if relativities else None,
                std_errors=None,  # Would need bootstrap for this
                shape=shape,
                recommendation=recommendation,
            ))
        
        # Categorical variables
        for var in categorical_factors:
            if var not in data.columns:
                continue
            
            values = data[var].to_numpy().astype(str)
            unique_levels = np.unique(values)
            
            grid_values = list(unique_levels)
            predictions = []
            
            for level in unique_levels:
                mask = values == level
                if np.any(mask):
                    predictions.append(float(np.mean(self.mu[mask])))
                else:
                    predictions.append(float(np.mean(self.mu)))
            
            # Analyze categorical effect
            if len(predictions) > 1:
                max_pred = max(predictions)
                min_pred = min(predictions)
                range_ratio = max_pred / min_pred if min_pred > 0 else float('inf')
                
                if range_ratio > 2:
                    shape = "high_variation"
                    recommendation = "Keep as categorical - significant level differences"
                elif range_ratio > 1.2:
                    shape = "moderate_variation"
                    recommendation = "Categorical appropriate, consider grouping similar levels"
                else:
                    shape = "low_variation"
                    recommendation = "Consider removing - little variation across levels"
            else:
                shape = "single_level"
                recommendation = "Cannot assess with single level"
            
            relativities = None
            if link == "log" and predictions:
                base = predictions[0]  # First level as base
                relativities = [p/base if base > 0 else 1.0 for p in predictions]
            
            results.append(PartialDependence(
                variable=var,
                variable_type="categorical",
                grid_values=grid_values,
                predictions=[round(p, 6) for p in predictions],
                relativities=[round(r, 4) for r in relativities] if relativities else None,
                std_errors=None,
                shape=shape,
                recommendation=recommendation,
            ))
        
        return results
    
    def _analyze_pd_shape(
        self, 
        grid: np.ndarray, 
        predictions: List[float],
        link: str,
    ) -> tuple:
        """Analyze partial dependence shape and provide recommendation."""
        if len(predictions) < 3:
            return "insufficient_data", "Need more data points"
        
        preds = np.array(predictions)
        
        # Compute differences
        diffs = np.diff(preds)
        
        # Check monotonicity
        increasing = np.sum(diffs > 0)
        decreasing = np.sum(diffs < 0)
        n_diffs = len(diffs)
        
        # Analyze curvature
        second_diffs = np.diff(diffs)
        curvature = np.mean(np.abs(second_diffs))
        
        # Relative range
        pred_range = np.max(preds) - np.min(preds)
        pred_mean = np.mean(preds)
        relative_range = pred_range / pred_mean if pred_mean > 0 else 0
        
        if relative_range < 0.05:
            return "flat", "May not need in model - negligible effect"
        
        if increasing >= n_diffs * 0.8:
            if curvature < pred_range * 0.1:
                return "linear_increasing", "Linear effect adequate"
            else:
                return "monotonic_increasing", "Consider spline for non-linearity"
        
        if decreasing >= n_diffs * 0.8:
            if curvature < pred_range * 0.1:
                return "linear_decreasing", "Linear effect adequate"
            else:
                return "monotonic_decreasing", "Consider spline for non-linearity"
        
        # Check for U-shape
        mid = len(preds) // 2
        left_trend = np.mean(diffs[:mid]) if mid > 0 else 0
        right_trend = np.mean(diffs[mid:]) if mid < len(diffs) else 0
        
        if left_trend < 0 and right_trend > 0:
            return "u_shaped", "Use spline (df=4+) or polynomial"
        if left_trend > 0 and right_trend < 0:
            return "inverted_u", "Use spline (df=4+) or polynomial"
        
        # Check for step function
        max_jump = np.max(np.abs(diffs))
        if max_jump > pred_range * 0.4:
            return "step_function", "Consider banding/categorical transformation"
        
        return "complex", "Use spline (df=5+) to capture non-linearity"
    
    def compute_dataset_diagnostics(
        self,
        y: np.ndarray,
        mu: np.ndarray,
        exposure: np.ndarray,
        data: "pl.DataFrame",
        categorical_factors: List[str],
        continuous_factors: List[str],
        dataset_name: str,
        result=None,
        n_bands: int = 10,
    ) -> DatasetDiagnostics:
        """
        Compute comprehensive diagnostics for a single dataset.
        
        Parameters
        ----------
        y : np.ndarray
            Actual response values
        mu : np.ndarray
            Predicted values
        exposure : np.ndarray
            Exposure weights
        data : pl.DataFrame
            DataFrame with factor columns
        categorical_factors : list of str
            Names of categorical factors
        continuous_factors : list of str
            Names of continuous factors
        dataset_name : str
            "train" or "test"
        result : GLMResults, optional
            Model results for partial dependence
        n_bands : int
            Number of bands for continuous variables
            
        Returns
        -------
        DatasetDiagnostics
        """
        n_obs = len(y)
        total_exposure = float(np.sum(exposure))
        total_actual = float(np.sum(y))
        total_predicted = float(np.sum(mu))
        
        # Family deviance metrics (same as GBM loss) using Rust backend
        dataset_metrics = _rust_dataset_metrics(y, mu, self.family, self.n_params)
        deviance = float(dataset_metrics["deviance"])
        mean_deviance = float(dataset_metrics["mean_deviance"])
        log_likelihood = float(dataset_metrics["log_likelihood"])
        aic_val = float(dataset_metrics["aic"])
        
        # Discrimination metrics
        stats = _rust_discrimination_stats(y, mu, exposure)
        gini = float(stats["gini"])
        auc = float(stats["auc"])
        
        # Overall A/E
        ae_ratio = total_actual / total_predicted if total_predicted > 0 else float('nan')
        
        # A/E by decile (sorted by predicted value)
        ae_by_decile = self._compute_ae_by_decile(y, mu, exposure, n_deciles=10)
        
        # Factor-level diagnostics
        factor_diag = {}
        for factor in categorical_factors:
            if factor in data.columns:
                factor_diag[factor] = self._compute_factor_level_metrics(
                    y, mu, exposure, data[factor].to_numpy().astype(str)
                )
        
        # Continuous variable diagnostics
        continuous_diag = {}
        for var in continuous_factors:
            if var in data.columns:
                values = data[var].to_numpy().astype(np.float64)
                continuous_diag[var] = self._compute_continuous_band_metrics(
                    y, mu, exposure, values, result, var, n_bands
                )
        
        return DatasetDiagnostics(
            dataset=dataset_name,
            n_obs=n_obs,
            total_exposure=round(total_exposure, 2),
            total_actual=round(total_actual, 2),
            total_predicted=round(total_predicted, 2),
            loss=round(mean_deviance, 6),
            deviance=round(deviance, 2),
            log_likelihood=round(log_likelihood, 2),
            aic=round(aic_val, 2),
            gini=round(gini, 4),
            auc=round(auc, 4),
            ae_ratio=round(ae_ratio, 4),
            ae_by_decile=ae_by_decile,
            factor_diagnostics=factor_diag,
            continuous_diagnostics=continuous_diag,
        )
    
    def _compute_ae_by_decile(
        self,
        y: np.ndarray,
        mu: np.ndarray,
        exposure: np.ndarray,
        n_deciles: int = 10,
    ) -> List[DecileMetrics]:
        """Compute A/E by decile sorted by predicted value."""
        # Sort by predicted values
        sort_idx = np.argsort(mu)
        y_sorted = y[sort_idx]
        mu_sorted = mu[sort_idx]
        exp_sorted = exposure[sort_idx]
        
        n = len(y)
        decile_size = n // n_deciles
        
        deciles = []
        for d in range(n_deciles):
            start = d * decile_size
            end = (d + 1) * decile_size if d < n_deciles - 1 else n
            
            y_d = y_sorted[start:end]
            mu_d = mu_sorted[start:end]
            exp_d = exp_sorted[start:end]
            
            actual = float(np.sum(y_d))
            predicted = float(np.sum(mu_d))
            exp_sum = float(np.sum(exp_d))
            ae = actual / predicted if predicted > 0 else float('nan')
            
            actual_freq = actual / exp_sum if exp_sum > 0 else 0.0
            predicted_freq = predicted / exp_sum if exp_sum > 0 else 0.0
            deciles.append(DecileMetrics(
                decile=d + 1,
                n=len(y_d),
                exposure=round(exp_sum, 2),
                actual=round(actual_freq, 6),
                predicted=round(predicted_freq, 6),
                ae_ratio=round(ae, 4) if not np.isnan(ae) else None,
            ))
        
        return deciles
    
    def _compute_factor_level_metrics(
        self,
        y: np.ndarray,
        mu: np.ndarray,
        exposure: np.ndarray,
        factor_values: np.ndarray,
    ) -> List[FactorLevelMetrics]:
        """Compute metrics for each level of a categorical factor."""
        unique_levels = np.unique(factor_values)
        # Use deviance residuals for consistency with continuous band metrics
        residuals = np.asarray(_rust_deviance_residuals(y, mu, self.family))
        
        metrics = []
        for level in unique_levels:
            mask = factor_values == level
            n = int(np.sum(mask))
            
            if n == 0:
                continue
            
            actual = float(np.sum(y[mask]))
            predicted = float(np.sum(mu[mask]))
            exp_sum = float(np.sum(exposure[mask]))
            ae = actual / predicted if predicted > 0 else float('nan')
            resid_mean = float(np.mean(residuals[mask]))
            
            actual_freq = actual / exp_sum if exp_sum > 0 else 0.0
            predicted_freq = predicted / exp_sum if exp_sum > 0 else 0.0
            metrics.append(FactorLevelMetrics(
                level=str(level),
                n=n,
                exposure=round(exp_sum, 2),
                actual=round(actual_freq, 6),
                predicted=round(predicted_freq, 6),
                ae_ratio=round(ae, 4) if not np.isnan(ae) else None,
                residual_mean=round(resid_mean, 6),
            ))
        
        # Sort by exposure (largest first)
        metrics.sort(key=lambda x: -x.exposure)
        return metrics
    
    def _compute_continuous_band_metrics(
        self,
        y: np.ndarray,
        mu: np.ndarray,
        exposure: np.ndarray,
        values: np.ndarray,
        result,
        var_name: str,
        n_bands: int = 10,
    ) -> List[ContinuousBandMetrics]:
        """Compute metrics for bands of a continuous variable."""
        # Remove NaN/Inf
        valid_mask = ~np.isnan(values) & ~np.isinf(values)
        
        if np.sum(valid_mask) < n_bands:
            return []
        
        # Use quantile bands
        percentiles = np.linspace(0, 100, n_bands + 1)
        edges = np.percentile(values[valid_mask], percentiles)
        edges = np.unique(edges)  # Remove duplicates
        
        if len(edges) < 2:
            return []
        
        metrics = []
        # Compute deviance residuals for consistency with categorical diagnostics
        deviance_resids = np.asarray(_rust_deviance_residuals(y, mu, self.family))
        
        for i in range(len(edges) - 1):
            lower, upper = edges[i], edges[i + 1]
            
            if i == len(edges) - 2:
                mask = valid_mask & (values >= lower) & (values <= upper)
            else:
                mask = valid_mask & (values >= lower) & (values < upper)
            
            n = int(np.sum(mask))
            if n == 0:
                continue
            
            actual = float(np.sum(y[mask]))
            predicted = float(np.sum(mu[mask]))
            exp_sum = float(np.sum(exposure[mask]))
            ae = actual / predicted if predicted > 0 else float('nan')
            midpoint = (lower + upper) / 2
            
            # Partial dependence at midpoint
            partial_dep = float(np.mean(mu[mask]))
            
            # Mean deviance residual for this band
            resid_mean = float(np.mean(deviance_resids[mask]))
            
            actual_freq = actual / exp_sum if exp_sum > 0 else 0.0
            predicted_freq = predicted / exp_sum if exp_sum > 0 else 0.0
            metrics.append(ContinuousBandMetrics(
                band=i + 1,
                range_min=round(float(lower), 4),
                range_max=round(float(upper), 4),
                midpoint=round(float(midpoint), 4),
                n=n,
                exposure=round(exp_sum, 2),
                actual=round(actual_freq, 6),
                predicted=round(predicted_freq, 6),
                ae_ratio=round(ae, 4) if not np.isnan(ae) else None,
                partial_dep=round(partial_dep, 6),
                residual_mean=round(resid_mean, 6),
            ))
        
        return metrics
    
    def compute_train_test_comparison(
        self,
        train_data: "pl.DataFrame",
        test_data: "pl.DataFrame",
        y_train: np.ndarray,
        mu_train: np.ndarray,
        exposure_train: np.ndarray,
        y_test: np.ndarray,
        mu_test: np.ndarray,
        exposure_test: np.ndarray,
        categorical_factors: List[str],
        continuous_factors: List[str],
        result=None,
    ) -> TrainTestComparison:
        """
        Compute comprehensive train vs test comparison with flags.
        
        Returns
        -------
        TrainTestComparison
            Complete comparison with per-set diagnostics and flags
        """
        # Compute diagnostics for each dataset
        train_diag = self.compute_dataset_diagnostics(
            y_train, mu_train, exposure_train, train_data,
            categorical_factors, continuous_factors, "train", result
        )
        test_diag = self.compute_dataset_diagnostics(
            y_test, mu_test, exposure_test, test_data,
            categorical_factors, continuous_factors, "test", result
        )
        
        # Comparison metrics
        gini_gap = train_diag.gini - test_diag.gini
        ae_ratio_diff = abs(train_diag.ae_ratio - test_diag.ae_ratio)
        
        # Decile comparison
        decile_comparison = []
        for i in range(min(len(train_diag.ae_by_decile), len(test_diag.ae_by_decile))):
            train_d = train_diag.ae_by_decile[i]
            test_d = test_diag.ae_by_decile[i]
            decile_comparison.append({
                "decile": i + 1,
                "train_ae": train_d.ae_ratio,
                "test_ae": test_d.ae_ratio,
                "ae_diff": round(abs((train_d.ae_ratio or 0) - (test_d.ae_ratio or 0)), 4),
            })
        
        # Factor-level divergence
        factor_divergence = {}
        unstable_factors_list = []
        
        for factor in categorical_factors:
            if factor in train_diag.factor_diagnostics and factor in test_diag.factor_diagnostics:
                train_levels = {m.level: m for m in train_diag.factor_diagnostics[factor]}
                test_levels = {m.level: m for m in test_diag.factor_diagnostics[factor]}
                
                divergent = []
                for level in set(train_levels.keys()) | set(test_levels.keys()):
                    train_ae = train_levels.get(level, FactorLevelMetrics(level, 0, 0, 0, 0, None, 0)).ae_ratio
                    test_ae = test_levels.get(level, FactorLevelMetrics(level, 0, 0, 0, 0, None, 0)).ae_ratio
                    
                    if train_ae is not None and test_ae is not None:
                        diff = abs(train_ae - test_ae)
                        if diff > 0.1:
                            divergent.append({
                                "level": level,
                                "train_ae": train_ae,
                                "test_ae": test_ae,
                                "ae_diff": round(diff, 4),
                            })
                            unstable_factors_list.append(f"{factor}[{level}]")
                
                if divergent:
                    factor_divergence[factor] = divergent
        
        # Flags for agent
        overfitting_risk = gini_gap > 0.03
        calibration_drift = test_diag.ae_ratio < 0.95 or test_diag.ae_ratio > 1.05
        
        return TrainTestComparison(
            train=train_diag,
            test=test_diag,
            gini_gap=round(gini_gap, 4),
            ae_ratio_diff=round(ae_ratio_diff, 4),
            decile_comparison=decile_comparison,
            factor_divergence=factor_divergence,
            overfitting_risk=overfitting_risk,
            calibration_drift=calibration_drift,
            unstable_factors=unstable_factors_list,
        )


# =============================================================================
# Pre-Fit Data Exploration
# =============================================================================

class DataExplorer:
    """
    Explores data before model fitting.
    
    This class provides pre-fit analysis including:
    - Factor statistics (univariate distributions)
    - Interaction detection based on response variable
    - Response distribution analysis
    
    Unlike DiagnosticsComputer, this does NOT require a fitted model.
    """
    
    def __init__(
        self,
        y: np.ndarray,
        exposure: Optional[np.ndarray] = None,
        family: str = "poisson",
    ):
        """
        Initialize the data explorer.
        
        Parameters
        ----------
        y : np.ndarray
            Response variable.
        exposure : np.ndarray, optional
            Exposure or weights.
        family : str, default="poisson"
            Family hint for appropriate statistics.
        """
        self.y = np.asarray(y, dtype=np.float64)
        self.exposure = np.asarray(exposure, dtype=np.float64) if exposure is not None else np.ones_like(self.y)
        self.family = family.lower()
        self.n_obs = len(y)
    
    def compute_response_stats(self) -> Dict[str, Any]:
        """Compute response variable statistics."""
        y_rate = self.y / self.exposure
        
        stats = {
            "n_observations": self.n_obs,
            "total_exposure": float(np.sum(self.exposure)),
            "total_response": float(np.sum(self.y)),
            "mean_response": float(np.mean(self.y)),
            "mean_rate": float(np.mean(y_rate)),
            "std_rate": float(np.std(y_rate)),
            "min": float(np.min(self.y)),
            "max": float(np.max(self.y)),
            "zeros_count": int(np.sum(self.y == 0)),
            "zeros_pct": float(100 * np.sum(self.y == 0) / self.n_obs),
        }
        
        # Add percentiles
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            stats[f"p{p}"] = float(np.percentile(y_rate, p))
        
        return stats
    
    def compute_factor_stats(
        self,
        data: "pl.DataFrame",
        categorical_factors: List[str],
        continuous_factors: List[str],
        n_bins: int = 10,
        rare_threshold_pct: float = 1.0,
        max_categorical_levels: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Compute univariate statistics for each factor.
        
        Returns statistics and actual/expected rates by level/bin.
        """
        factors = []
        
        # Continuous factors
        for name in continuous_factors:
            if name not in data.columns:
                raise ValueError(f"Continuous factor '{name}' not found in data columns: {list(data.columns)}")
            
            values = data[name].to_numpy().astype(np.float64)
            valid_mask = ~np.isnan(values) & ~np.isinf(values)
            valid_values = values[valid_mask]
            
            if len(valid_values) == 0:
                continue
            
            # Univariate stats
            stats = {
                "name": name,
                "type": "continuous",
                "mean": float(np.mean(valid_values)),
                "std": float(np.std(valid_values)),
                "min": float(np.min(valid_values)),
                "max": float(np.max(valid_values)),
                "missing_count": int(np.sum(~valid_mask)),
                "missing_pct": float(100 * np.sum(~valid_mask) / len(values)),
            }
            
            # Response by quantile bins
            quantiles = np.percentile(valid_values, np.linspace(0, 100, n_bins + 1))
            bins_data = []
            bin_rates = []
            thin_cells = []
            total_exposure = np.sum(self.exposure)
            
            for i in range(n_bins):
                if i == n_bins - 1:
                    bin_mask = (values >= quantiles[i]) & (values <= quantiles[i + 1])
                else:
                    bin_mask = (values >= quantiles[i]) & (values < quantiles[i + 1])
                
                if not np.any(bin_mask):
                    continue
                
                y_bin = self.y[bin_mask]
                exp_bin = self.exposure[bin_mask]
                bin_exposure = float(np.sum(exp_bin))
                rate = float(np.sum(y_bin) / bin_exposure) if bin_exposure > 0 else 0
                
                bins_data.append({
                    "bin_index": i,
                    "bin_lower": float(quantiles[i]),
                    "bin_upper": float(quantiles[i + 1]),
                    "count": int(np.sum(bin_mask)),
                    "exposure": bin_exposure,
                    "response_sum": float(np.sum(y_bin)),
                    "response_rate": rate,
                })
                bin_rates.append(rate)
                
                # Check for thin cells (< 1% exposure)
                if bin_exposure / total_exposure < 0.01:
                    thin_cells.append(i)
            
            stats["response_by_bin"] = bins_data
            
            # Compute shape recommendation
            if len(bin_rates) >= 3:
                shape_hint = self._compute_shape_hint(bin_rates)
            else:
                shape_hint = {"shape": "insufficient_data", "recommendation": "linear"}
            
            stats["modeling_hints"] = {
                "shape": shape_hint["shape"],
                "recommendation": shape_hint["recommendation"],
                "thin_cells": thin_cells if thin_cells else None,
                "thin_cell_warning": f"Bins {thin_cells} have <1% exposure" if thin_cells else None,
            }
            
            factors.append(stats)
        
        # Categorical factors
        for name in categorical_factors:
            if name not in data.columns:
                raise ValueError(f"Categorical factor '{name}' not found in data columns: {list(data.columns)}")
            
            values = data[name].to_numpy().astype(str)
            unique_levels = np.unique(values)
            
            # Sort levels by exposure
            level_exposures = []
            for level in unique_levels:
                mask = values == level
                exp = np.sum(self.exposure[mask])
                level_exposures.append((level, exp))
            level_exposures.sort(key=lambda x: -x[1])
            
            total_exposure = np.sum(self.exposure)
            
            # Build level stats
            levels_data = []
            other_mask = np.zeros(len(values), dtype=bool)
            
            for i, (level, exp) in enumerate(level_exposures):
                pct = 100 * exp / total_exposure
                
                if pct < rare_threshold_pct or i >= max_categorical_levels - 1:
                    other_mask |= (values == level)
                else:
                    mask = values == level
                    y_level = self.y[mask]
                    exp_level = self.exposure[mask]
                    
                    levels_data.append({
                        "level": level,
                        "count": int(np.sum(mask)),
                        "exposure": float(np.sum(exp_level)),
                        "exposure_pct": float(pct),
                        "response_sum": float(np.sum(y_level)),
                        "response_rate": float(np.sum(y_level) / np.sum(exp_level)) if np.sum(exp_level) > 0 else 0,
                    })
            
            # Add "Other" if needed
            if np.any(other_mask):
                y_other = self.y[other_mask]
                exp_other = self.exposure[other_mask]
                levels_data.append({
                    "level": "_Other",
                    "count": int(np.sum(other_mask)),
                    "exposure": float(np.sum(exp_other)),
                    "exposure_pct": float(100 * np.sum(exp_other) / total_exposure),
                    "response_sum": float(np.sum(y_other)),
                    "response_rate": float(np.sum(y_other) / np.sum(exp_other)) if np.sum(exp_other) > 0 else 0,
                })
            
            # Compute modeling hints for categorical
            main_levels = [l for l in levels_data if l["level"] != "_Other"]
            
            # Suggested base level: highest exposure among non-Other levels
            suggested_base = main_levels[0]["level"] if main_levels else None
            
            # Check for thin cells
            thin_levels = [l["level"] for l in main_levels if l["exposure_pct"] < 1.0]
            
            # Check if ordinal (levels are numeric or follow A-Z pattern)
            ordinal_hint = self._detect_ordinal_pattern(unique_levels)
            
            stats = {
                "name": name,
                "type": "categorical",
                "n_levels": len(unique_levels),
                "n_levels_shown": len(levels_data),
                "levels": levels_data,
                "modeling_hints": {
                    "suggested_base_level": suggested_base,
                    "ordinal": ordinal_hint["is_ordinal"],
                    "ordinal_pattern": ordinal_hint["pattern"],
                    "thin_levels": thin_levels if thin_levels else None,
                    "thin_level_warning": f"Levels {thin_levels} have <1% exposure" if thin_levels else None,
                },
            }
            factors.append(stats)
        
        return factors
    
    def _compute_shape_hint(self, bin_rates: List[float]) -> Dict[str, str]:
        """Analyze binned response rates to suggest transformation."""
        n = len(bin_rates)
        if n < 3:
            return {"shape": "insufficient_data", "recommendation": "linear"}
        
        # Check monotonicity
        diffs = [bin_rates[i+1] - bin_rates[i] for i in range(n-1)]
        increasing = sum(1 for d in diffs if d > 0)
        decreasing = sum(1 for d in diffs if d < 0)
        
        # Strong monotonic pattern
        if increasing >= n - 2:
            return {"shape": "monotonic_increasing", "recommendation": "linear or log"}
        if decreasing >= n - 2:
            return {"shape": "monotonic_decreasing", "recommendation": "linear or log"}
        
        # Check for U-shape or inverted U
        mid = n // 2
        left_trend = sum(diffs[:mid])
        right_trend = sum(diffs[mid:])
        
        if left_trend < 0 and right_trend > 0:
            return {"shape": "u_shaped", "recommendation": "spline or polynomial"}
        if left_trend > 0 and right_trend < 0:
            return {"shape": "inverted_u", "recommendation": "spline or polynomial"}
        
        # Check for step function (large jump)
        max_diff = max(abs(d) for d in diffs)
        avg_rate = sum(bin_rates) / n
        if max_diff > avg_rate * 0.5:
            return {"shape": "step_function", "recommendation": "banding or categorical"}
        
        # Non-linear but no clear pattern
        variance = sum((r - avg_rate)**2 for r in bin_rates) / n
        if variance > avg_rate * 0.1:
            return {"shape": "non_linear", "recommendation": "spline"}
        
        return {"shape": "flat", "recommendation": "may not need in model"}
    
    def _detect_ordinal_pattern(self, levels: np.ndarray) -> Dict[str, Any]:
        """Detect if categorical levels follow an ordinal pattern."""
        levels_str = [str(l) for l in levels]
        
        # Check for numeric levels
        try:
            numeric = [float(l) for l in levels_str]
            return {"is_ordinal": True, "pattern": "numeric"}
        except ValueError:
            pass
        
        # Check for single letter A-Z pattern
        if all(len(l) == 1 and l.isalpha() for l in levels_str):
            return {"is_ordinal": True, "pattern": "alphabetic"}
        
        # Check for common ordinal patterns
        ordinal_patterns = [
            (["low", "medium", "high"], "low_medium_high"),
            (["small", "medium", "large"], "size"),
            (["young", "middle", "old"], "age"),
            (["1", "2", "3", "4", "5"], "numeric_string"),
        ]
        
        levels_lower = [l.lower() for l in levels_str]
        for pattern, name in ordinal_patterns:
            if all(p in levels_lower for p in pattern):
                return {"is_ordinal": True, "pattern": name}
        
        # Check for prefix + number pattern (e.g., "Region1", "Region2")
        import re
        if all(re.match(r'^[A-Za-z]+\d+$', l) for l in levels_str):
            return {"is_ordinal": True, "pattern": "prefix_numeric"}
        
        return {"is_ordinal": False, "pattern": None}
    
    def compute_univariate_tests(
        self,
        data: "pl.DataFrame",
        categorical_factors: List[str],
        continuous_factors: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Compute univariate significance tests for each factor vs response.
        
        For continuous factors: Pearson correlation + F-test from simple regression
        For categorical factors: ANOVA F-test (eta-squared based)
        """
        results = []
        y_rate = self.y / self.exposure
        
        for name in continuous_factors:
            if name not in data.columns:
                raise ValueError(f"Continuous factor '{name}' not found in data columns: {list(data.columns)}")
            
            values = data[name].to_numpy().astype(np.float64)
            valid_mask = ~np.isnan(values) & ~np.isinf(values)
            
            if np.sum(valid_mask) < 10:
                continue  # Skip factors with insufficient valid data (expected behavior)
            
            x_valid = values[valid_mask]
            y_valid = y_rate[valid_mask]
            w_valid = self.exposure[valid_mask]
            
            # Weighted correlation
            x_mean = np.average(x_valid, weights=w_valid)
            y_mean = np.average(y_valid, weights=w_valid)
            
            cov_xy = np.sum(w_valid * (x_valid - x_mean) * (y_valid - y_mean)) / np.sum(w_valid)
            std_x = np.sqrt(np.sum(w_valid * (x_valid - x_mean) ** 2) / np.sum(w_valid))
            std_y = np.sqrt(np.sum(w_valid * (y_valid - y_mean) ** 2) / np.sum(w_valid))
            
            corr = cov_xy / (std_x * std_y) if std_x > 0 and std_y > 0 else 0.0
            
            # F-test from regression
            n = len(x_valid)
            r2 = corr ** 2
            f_stat = (r2 / 1) / ((1 - r2) / (n - 2)) if r2 < 1 and n > 2 else 0
            
            # P-value from F-distribution (using Rust CDF)
            pvalue = 1 - _f_cdf(f_stat, 1.0, float(n - 2)) if n > 2 else 1.0
            
            results.append({
                "factor": name,
                "type": "continuous",
                "test": "correlation_f_test",
                "correlation": float(corr),
                "r_squared": float(r2),
                "f_statistic": float(f_stat),
                "pvalue": float(pvalue),
                "significant_01": pvalue < 0.01 if not np.isnan(pvalue) else False,
                "significant_05": pvalue < 0.05 if not np.isnan(pvalue) else False,
            })
        
        for name in categorical_factors:
            if name not in data.columns:
                raise ValueError(f"Categorical factor '{name}' not found in data columns: {list(data.columns)}")
            
            values = data[name].to_numpy().astype(str)
            
            # ANOVA: eta-squared and F-test
            eta_sq = self._compute_eta_squared_response(values)
            
            unique_levels = np.unique(values)
            k = len(unique_levels)
            n = len(values)
            
            if k > 1 and n > k:
                f_stat = (eta_sq / (k - 1)) / ((1 - eta_sq) / (n - k)) if eta_sq < 1 else 0
                
                # P-value from F-distribution (using Rust CDF)
                pvalue = 1 - _f_cdf(f_stat, float(k - 1), float(n - k))
            else:
                f_stat = 0.0
                pvalue = 1.0
            
            results.append({
                "factor": name,
                "type": "categorical",
                "test": "anova_f_test",
                "n_levels": k,
                "eta_squared": float(eta_sq),
                "f_statistic": float(f_stat),
                "pvalue": float(pvalue),
                "significant_01": pvalue < 0.01 if not np.isnan(pvalue) else False,
                "significant_05": pvalue < 0.05 if not np.isnan(pvalue) else False,
            })
        
        # Sort by p-value (most significant first)
        results.sort(key=lambda x: x["pvalue"] if not np.isnan(x["pvalue"]) else 1.0)
        return results
    
    def compute_correlations(
        self,
        data: "pl.DataFrame",
        continuous_factors: List[str],
    ) -> Dict[str, Any]:
        """
        Compute pairwise correlations between continuous factors.
        
        Returns correlation matrix and flags for high correlations.
        """
        valid_factors = [f for f in continuous_factors if f in data.columns]
        
        if len(valid_factors) < 2:
            return {"factors": valid_factors, "matrix": [], "high_correlations": []}
        
        # Build matrix of valid values
        arrays = []
        for name in valid_factors:
            arr = data[name].to_numpy().astype(np.float64)
            arrays.append(arr)
        
        X = np.column_stack(arrays)
        
        # Handle missing values - use pairwise complete observations
        n_factors = len(valid_factors)
        corr_matrix = np.eye(n_factors)
        
        for i in range(n_factors):
            for j in range(i + 1, n_factors):
                xi, xj = X[:, i], X[:, j]
                valid = ~np.isnan(xi) & ~np.isnan(xj) & ~np.isinf(xi) & ~np.isinf(xj)
                
                if np.sum(valid) > 2:
                    corr = np.corrcoef(xi[valid], xj[valid])[0, 1]
                    corr_matrix[i, j] = corr
                    corr_matrix[j, i] = corr
                else:
                    corr_matrix[i, j] = float('nan')
                    corr_matrix[j, i] = float('nan')
        
        # Find high correlations (|r| > 0.7)
        high_corrs = []
        for i in range(n_factors):
            for j in range(i + 1, n_factors):
                r = corr_matrix[i, j]
                if not np.isnan(r) and abs(r) > 0.7:
                    high_corrs.append({
                        "factor1": valid_factors[i],
                        "factor2": valid_factors[j],
                        "correlation": float(r),
                        "severity": "high" if abs(r) > 0.9 else "moderate",
                    })
        
        high_corrs.sort(key=lambda x: -abs(x["correlation"]))
        
        return {
            "factors": valid_factors,
            "matrix": corr_matrix.tolist(),
            "high_correlations": high_corrs,
        }
    
    def compute_vif(
        self,
        data: "pl.DataFrame",
        continuous_factors: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Compute Variance Inflation Factors for multicollinearity detection.
        
        VIF > 5 indicates moderate multicollinearity
        VIF > 10 indicates severe multicollinearity
        """
        valid_factors = [f for f in continuous_factors if f in data.columns]
        
        if len(valid_factors) < 2:
            return [{"factor": f, "vif": 1.0, "severity": "none"} for f in valid_factors]
        
        # Build design matrix
        arrays = []
        for name in valid_factors:
            arr = data[name].to_numpy().astype(np.float64)
            arrays.append(arr)
        
        X = np.column_stack(arrays)
        
        # Remove rows with any NaN/Inf
        valid_rows = np.all(~np.isnan(X) & ~np.isinf(X), axis=1)
        X = X[valid_rows]
        
        if len(X) < len(valid_factors) + 1:
            return [{"factor": f, "vif": float('nan'), "severity": "unknown"} for f in valid_factors]
        
        # Standardize
        X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-10)
        
        results = []
        for i, name in enumerate(valid_factors):
            # Regress factor i on all others
            y = X[:, i]
            others = np.delete(X, i, axis=1)
            
            # Add intercept
            others_with_int = np.column_stack([np.ones(len(others)), others])
            
            try:
                # OLS: beta = (X'X)^-1 X'y
                beta = np.linalg.lstsq(others_with_int, y, rcond=None)[0]
                y_pred = others_with_int @ beta
                
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                vif = 1 / (1 - r2) if r2 < 1 else float('inf')
            except Exception as e:
                # Re-raise - VIF computation shouldn't fail silently
                raise RuntimeError(f"Failed to compute VIF for '{name}': {e}") from e
            
            if np.isnan(vif) or np.isinf(vif):
                severity = "unknown"
            elif vif > 10:
                severity = "severe"
            elif vif > 5:
                severity = "moderate"
            else:
                severity = "none"
            
            results.append({
                "factor": name,
                "vif": float(vif) if not np.isinf(vif) else 999.0,
                "severity": severity,
            })
        
        results.sort(key=lambda x: -x["vif"] if not np.isnan(x["vif"]) else 0)
        return results
    
    def compute_missing_values(
        self,
        data: "pl.DataFrame",
        categorical_factors: List[str],
        continuous_factors: List[str],
    ) -> Dict[str, Any]:
        """
        Analyze missing values across all factors.
        """
        all_factors = categorical_factors + continuous_factors
        factor_missing = []
        total_rows = len(data)
        
        for name in all_factors:
            if name not in data.columns:
                raise ValueError(f"Factor '{name}' not found in data columns: {list(data.columns)}")
            
            col = data[name]
            n_missing = col.null_count()
            pct_missing = 100.0 * n_missing / total_rows if total_rows > 0 else 0
            
            factor_missing.append({
                "factor": name,
                "n_missing": int(n_missing),
                "pct_missing": float(pct_missing),
                "severity": "high" if pct_missing > 10 else ("moderate" if pct_missing > 1 else "none"),
            })
        
        factor_missing.sort(key=lambda x: -x["pct_missing"])
        
        # Count rows with any missing
        any_missing = 0
        for name in all_factors:
            if name in data.columns:
                any_missing += data[name].null_count()
        
        return {
            "total_rows": total_rows,
            "factors_with_missing": [f for f in factor_missing if f["n_missing"] > 0],
            "n_complete_rows": total_rows - sum(f["n_missing"] for f in factor_missing),
            "summary": "No missing values" if all(f["n_missing"] == 0 for f in factor_missing) else "Missing values present",
        }
    
    def compute_zero_inflation(self) -> Dict[str, Any]:
        """
        Check for zero inflation in count data.
        
        Compares observed zeros to expected zeros under Poisson assumption.
        """
        y = self.y
        n = len(y)
        
        observed_zeros = int(np.sum(y == 0))
        observed_zero_pct = 100.0 * observed_zeros / n if n > 0 else 0
        
        # Expected zeros under Poisson: P(Y=0) = exp(-lambda) where lambda = mean
        mean_y = np.mean(y)
        if mean_y > 0:
            expected_zero_pct = 100.0 * np.exp(-mean_y)
            excess_zeros = observed_zero_pct - expected_zero_pct
        else:
            expected_zero_pct = 100.0
            excess_zeros = 0.0
        
        # Severity assessment
        if excess_zeros > 20:
            severity = "severe"
            recommendation = "Consider zero-inflated model (ZIP, ZINB)"
        elif excess_zeros > 10:
            severity = "moderate"
            recommendation = "Consider zero-inflated or hurdle model"
        elif excess_zeros > 5:
            severity = "mild"
            recommendation = "Monitor; may need zero-inflated model"
        else:
            severity = "none"
            recommendation = "Standard Poisson/NegBin likely adequate"
        
        return {
            "observed_zeros": observed_zeros,
            "observed_zero_pct": float(observed_zero_pct),
            "expected_zero_pct_poisson": float(expected_zero_pct),
            "excess_zero_pct": float(excess_zeros),
            "severity": severity,
            "recommendation": recommendation,
        }
    
    def compute_overdispersion(self) -> Dict[str, Any]:
        """
        Check for overdispersion in count data.
        
        Compares variance to mean (Poisson assumes Var = Mean).
        """
        y = self.y
        exposure = self.exposure
        
        # Compute rate
        rate = y / exposure
        
        # Weighted mean and variance
        total_exp = np.sum(exposure)
        mean_rate = np.sum(y) / total_exp
        
        # Variance of rates (exposure-weighted)
        var_rate = np.sum(exposure * (rate - mean_rate) ** 2) / total_exp
        
        # For Poisson with exposure, expected variance of rate is mean_rate / exposure
        # Aggregate expected variance
        expected_var = mean_rate * np.sum(1.0 / exposure * exposure) / total_exp  # = mean_rate
        
        # Dispersion ratio
        if expected_var > 0:
            dispersion_ratio = var_rate / expected_var
        else:
            dispersion_ratio = 1.0
        
        # Also compute using counts directly
        mean_count = np.mean(y)
        var_count = np.var(y, ddof=1)
        count_dispersion = var_count / mean_count if mean_count > 0 else 1.0
        
        # Severity assessment
        if count_dispersion > 5:
            severity = "severe"
            recommendation = "Use Negative Binomial or QuasiPoisson"
        elif count_dispersion > 2:
            severity = "moderate"
            recommendation = "Consider Negative Binomial or QuasiPoisson"
        elif count_dispersion > 1.5:
            severity = "mild"
            recommendation = "Monitor; Poisson may underestimate standard errors"
        else:
            severity = "none"
            recommendation = "Poisson assumption reasonable"
        
        return {
            "mean_count": float(mean_count),
            "var_count": float(var_count),
            "dispersion_ratio": float(count_dispersion),
            "severity": severity,
            "recommendation": recommendation,
        }
    
    def compute_cramers_v(
        self,
        data: "pl.DataFrame",
        categorical_factors: List[str],
    ) -> Dict[str, Any]:
        """
        Compute Cramér's V matrix for categorical factor pairs.
        
        Cramér's V measures association between categorical variables (0 to 1).
        """
        valid_factors = [f for f in categorical_factors if f in data.columns]
        
        if len(valid_factors) < 2:
            return {"factors": valid_factors, "matrix": [], "high_associations": []}
        
        n_factors = len(valid_factors)
        v_matrix = np.eye(n_factors)
        
        for i in range(n_factors):
            for j in range(i + 1, n_factors):
                v = self._compute_cramers_v_pair(
                    data[valid_factors[i]].to_numpy(),
                    data[valid_factors[j]].to_numpy(),
                )
                v_matrix[i, j] = v
                v_matrix[j, i] = v
        
        # Find high associations (V > 0.3)
        high_assoc = []
        for i in range(n_factors):
            for j in range(i + 1, n_factors):
                v = v_matrix[i, j]
                if not np.isnan(v) and v > 0.3:
                    high_assoc.append({
                        "factor1": valid_factors[i],
                        "factor2": valid_factors[j],
                        "cramers_v": float(v),
                        "severity": "high" if v > 0.5 else "moderate",
                    })
        
        high_assoc.sort(key=lambda x: -x["cramers_v"])
        
        return {
            "factors": valid_factors,
            "matrix": v_matrix.tolist(),
            "high_associations": high_assoc,
        }
    
    def _compute_cramers_v_pair(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute Cramér's V for a pair of categorical variables."""
        # Build contingency table
        x_str = x.astype(str)
        y_str = y.astype(str)
        
        x_cats = np.unique(x_str)
        y_cats = np.unique(y_str)
        
        r, k = len(x_cats), len(y_cats)
        if r < 2 or k < 2:
            return 0.0
        
        # Count frequencies
        contingency = np.zeros((r, k))
        for i, xc in enumerate(x_cats):
            for j, yc in enumerate(y_cats):
                contingency[i, j] = np.sum((x_str == xc) & (y_str == yc))
        
        n = contingency.sum()
        if n == 0:
            return 0.0
        
        # Chi-squared statistic
        row_sums = contingency.sum(axis=1, keepdims=True)
        col_sums = contingency.sum(axis=0, keepdims=True)
        expected = row_sums * col_sums / n
        
        # Handle zero expected values explicitly (don't suppress warnings)
        if np.any(expected == 0):
            # Zero expected values indicate empty cells - raise error for actuarial transparency
            raise ValueError(
                f"Cramér's V calculation has zero expected frequencies. "
                f"This indicates empty cells in the contingency table between factors. "
                f"Check data quality or reduce number of factor levels."
            )
        chi2 = np.sum((contingency - expected) ** 2 / expected)
        
        # Cramér's V
        min_dim = min(r - 1, k - 1)
        if min_dim == 0 or n == 0:
            return 0.0
        
        v = np.sqrt(chi2 / (n * min_dim))
        return float(v)
    
    def detect_interactions(
        self,
        data: "pl.DataFrame",
        factor_names: List[str],
        max_factors: int = 10,
        min_effect_size: float = 0.01,
        max_candidates: int = 5,
        min_cell_count: int = 30,
    ) -> List[InteractionCandidate]:
        """
        Detect potential interactions using response-based analysis.
        
        This identifies factors whose combined effect on the response
        differs from their individual effects, suggesting an interaction.
        """
        # First, rank factors by their effect on response variance
        factor_scores = []
        
        for name in factor_names:
            if name not in data.columns:
                raise ValueError(f"Factor '{name}' not found in data columns: {list(data.columns)}")
            
            values = data[name].to_numpy()
            
            # Compute eta-squared (variance explained)
            if values.dtype == object or str(values.dtype).startswith('str'):
                score = self._compute_eta_squared_response(values.astype(str))
            else:
                values = values.astype(np.float64)
                valid_mask = ~np.isnan(values) & ~np.isinf(values)
                if np.sum(valid_mask) < 10:
                    continue
                # Bin continuous variables
                bins = self._discretize(values, 5)
                score = self._compute_eta_squared_response(bins.astype(str))
            
            if score >= min_effect_size:
                factor_scores.append((name, score))
        
        # Sort and take top factors
        factor_scores.sort(key=lambda x: -x[1])
        top_factors = [name for name, _ in factor_scores[:max_factors]]
        
        if len(top_factors) < 2:
            return []
        
        # Check pairwise interactions
        candidates = []
        
        for i in range(len(top_factors)):
            for j in range(i + 1, len(top_factors)):
                name1, name2 = top_factors[i], top_factors[j]
                
                values1 = data[name1].to_numpy()
                values2 = data[name2].to_numpy()
                
                # Discretize both factors
                bins1 = self._discretize(values1, 5)
                bins2 = self._discretize(values2, 5)
                
                # Compute interaction strength
                candidate = self._compute_interaction_strength_response(
                    name1, bins1, name2, bins2, min_cell_count
                )
                
                if candidate is not None:
                    candidates.append(candidate)
        
        # Sort by strength and return top candidates
        candidates.sort(key=lambda x: -x.interaction_strength)
        return candidates[:max_candidates]
    
    def _compute_eta_squared_response(self, categories: np.ndarray) -> float:
        """Compute eta-squared for categorical association with response."""
        y_rate = self.y / self.exposure
        unique_levels = np.unique(categories)
        overall_mean = np.average(y_rate, weights=self.exposure)
        
        ss_total = np.sum(self.exposure * (y_rate - overall_mean) ** 2)
        
        if ss_total == 0:
            return 0.0
        
        ss_between = 0.0
        for level in unique_levels:
            mask = categories == level
            level_rate = y_rate[mask]
            level_exp = self.exposure[mask]
            level_mean = np.average(level_rate, weights=level_exp)
            ss_between += np.sum(level_exp) * (level_mean - overall_mean) ** 2
        
        return ss_between / ss_total
    
    def _discretize(self, values: np.ndarray, n_bins: int) -> np.ndarray:
        """Discretize values into bins."""
        if values.dtype == object or str(values.dtype).startswith('str'):
            unique_vals = np.unique(values)
            mapping = {v: i for i, v in enumerate(unique_vals)}
            return np.array([mapping[v] for v in values])
        else:
            values = values.astype(np.float64)
            valid_mask = ~np.isnan(values) & ~np.isinf(values)
            
            if not np.any(valid_mask):
                return np.zeros(len(values), dtype=int)
            
            quantiles = np.percentile(values[valid_mask], np.linspace(0, 100, n_bins + 1))
            bins = np.digitize(values, quantiles[1:-1])
            bins[~valid_mask] = n_bins
            return bins
    
    def _compute_interaction_strength_response(
        self,
        name1: str,
        bins1: np.ndarray,
        name2: str,
        bins2: np.ndarray,
        min_cell_count: int,
    ) -> Optional[InteractionCandidate]:
        """Compute interaction strength based on response variance."""
        y_rate = self.y / self.exposure
        
        # Create interaction cells
        cell_ids = bins1 * 1000 + bins2
        unique_cells = np.unique(cell_ids)
        
        # Filter cells with sufficient data
        valid_cells = []
        cell_rates = []
        cell_weights = []
        
        for cell_id in unique_cells:
            mask = cell_ids == cell_id
            if np.sum(mask) >= min_cell_count:
                valid_cells.append(cell_id)
                cell_rates.append(y_rate[mask])
                cell_weights.append(self.exposure[mask])
        
        if len(valid_cells) < 4:
            return None
        
        # Compute variance explained by cells
        all_rates = np.concatenate(cell_rates)
        all_weights = np.concatenate(cell_weights)
        overall_mean = np.average(all_rates, weights=all_weights)
        
        ss_total = np.sum(all_weights * (all_rates - overall_mean) ** 2)
        
        if ss_total == 0:
            return None
        
        ss_model = sum(
            np.sum(w) * (np.average(r, weights=w) - overall_mean) ** 2
            for r, w in zip(cell_rates, cell_weights)
        )
        
        r_squared = ss_model / ss_total
        
        # F-test p-value
        df_model = len(valid_cells) - 1
        df_resid = len(all_rates) - len(valid_cells)
        
        if df_model > 0 and df_resid > 0:
            f_stat = (ss_model / df_model) / ((ss_total - ss_model) / df_resid)
            
            # P-value from F-distribution (using Rust CDF)
            pvalue = 1 - _f_cdf(f_stat, float(df_model), float(df_resid))
        else:
            pvalue = float('nan')
        
        return InteractionCandidate(
            factor1=name1,
            factor2=name2,
            interaction_strength=float(r_squared),
            pvalue=float(pvalue),
            n_cells=len(valid_cells),
        )


def explore_data(
    data: "pl.DataFrame",
    response: str,
    categorical_factors: Optional[List[str]] = None,
    continuous_factors: Optional[List[str]] = None,
    exposure: Optional[str] = None,
    family: str = "poisson",
    n_bins: int = 10,
    rare_threshold_pct: float = 1.0,
    max_categorical_levels: int = 20,
    detect_interactions: bool = True,
    max_interaction_factors: int = 10,
) -> DataExploration:
    """
    Explore data before model fitting.
    
    This function provides pre-fit analysis including factor statistics
    and interaction detection without requiring a fitted model.
    
    Results are automatically saved to 'analysis/exploration.json'.
    
    Parameters
    ----------
    data : pl.DataFrame
        Data to explore.
    response : str
        Name of the response variable column.
    categorical_factors : list of str, optional
        Names of categorical factors to analyze.
    continuous_factors : list of str, optional
        Names of continuous factors to analyze.
    exposure : str, optional
        Name of the exposure/weights column.
    family : str, default="poisson"
        Expected family (for appropriate statistics).
    n_bins : int, default=10
        Number of bins for continuous factors.
    rare_threshold_pct : float, default=1.0
        Threshold (%) below which categorical levels are grouped.
    max_categorical_levels : int, default=20
        Maximum categorical levels to show.
    detect_interactions : bool, default=True
        Whether to detect potential interactions.
    max_interaction_factors : int, default=10
        Maximum factors for interaction detection.
    
    Returns
    -------
    DataExploration
        Pre-fit exploration results with to_json() method.
    
    Examples
    --------
    >>> import rustystats as rs
    >>> 
    >>> # Explore data before fitting
    >>> exploration = rs.explore_data(
    ...     data=data,
    ...     response="ClaimNb",
    ...     categorical_factors=["Region", "VehBrand"],
    ...     continuous_factors=["Age", "VehPower"],
    ...     exposure="Exposure",
    ...     family="poisson",
    ... )
    >>> 
    >>> # View interaction candidates
    >>> for ic in exploration.interaction_candidates:
    ...     print(f"{ic.factor1} x {ic.factor2}: {ic.interaction_strength:.3f}")
    >>> 
    >>> # Export as JSON
    >>> print(exploration.to_json())
    """
    categorical_factors = list(dict.fromkeys(categorical_factors or []))  # Dedupe preserving order
    continuous_factors = list(dict.fromkeys(continuous_factors or []))  # Dedupe preserving order
    
    # Extract response and exposure
    y = data[response].to_numpy().astype(np.float64)
    exp = data[exposure].to_numpy().astype(np.float64) if exposure else None
    
    # Create explorer
    explorer = DataExplorer(y=y, exposure=exp, family=family)
    
    # Compute statistics
    response_stats = explorer.compute_response_stats()
    
    factor_stats = explorer.compute_factor_stats(
        data=data,
        categorical_factors=categorical_factors,
        continuous_factors=continuous_factors,
        n_bins=n_bins,
        rare_threshold_pct=rare_threshold_pct,
        max_categorical_levels=max_categorical_levels,
    )
    
    # Univariate significance tests
    univariate_tests = explorer.compute_univariate_tests(
        data=data,
        categorical_factors=categorical_factors,
        continuous_factors=continuous_factors,
    )
    
    # Correlations between continuous factors
    correlations = explorer.compute_correlations(
        data=data,
        continuous_factors=continuous_factors,
    )
    
    # VIF for multicollinearity
    vif = explorer.compute_vif(
        data=data,
        continuous_factors=continuous_factors,
    )
    
    # Missing value analysis
    missing_values = explorer.compute_missing_values(
        data=data,
        categorical_factors=categorical_factors,
        continuous_factors=continuous_factors,
    )
    
    # Cramér's V for categorical pairs
    cramers_v = explorer.compute_cramers_v(
        data=data,
        categorical_factors=categorical_factors,
    )
    
    # Zero inflation check (for count data)
    zero_inflation = explorer.compute_zero_inflation()
    
    # Overdispersion check
    overdispersion = explorer.compute_overdispersion()
    
    # Interaction detection
    interaction_candidates = []
    if detect_interactions and len(categorical_factors) + len(continuous_factors) >= 2:
        all_factors = categorical_factors + continuous_factors
        interaction_candidates = explorer.detect_interactions(
            data=data,
            factor_names=all_factors,
            max_factors=max_interaction_factors,
            min_effect_size=0.001,  # Lower threshold to catch more interactions
        )
    
    # Data summary
    data_summary = {
        "n_rows": len(data),
        "n_columns": len(data.columns),
        "response_column": response,
        "exposure_column": exposure,
        "n_categorical_factors": len(categorical_factors),
        "n_continuous_factors": len(continuous_factors),
    }
    
    result = DataExploration(
        data_summary=data_summary,
        factor_stats=factor_stats,
        missing_values=missing_values,
        univariate_tests=univariate_tests,
        correlations=correlations,
        cramers_v=cramers_v,
        vif=vif,
        zero_inflation=zero_inflation,
        overdispersion=overdispersion,
        interaction_candidates=interaction_candidates,
        response_stats=response_stats,
    )
    
    # Auto-save JSON to analysis folder
    import os
    os.makedirs("analysis", exist_ok=True)
    with open("analysis/exploration.json", "w") as f:
        f.write(result.to_json(indent=2))
    
    return result


# =============================================================================
# Post-Fit Model Diagnostics
# =============================================================================

def compute_diagnostics(
    result,  # GLMResults or FormulaGLMResults
    train_data: "pl.DataFrame",
    categorical_factors: Optional[List[str]] = None,
    continuous_factors: Optional[List[str]] = None,
    n_calibration_bins: int = 10,
    n_factor_bins: int = 10,
    rare_threshold_pct: float = 1.0,
    max_categorical_levels: int = 20,
    detect_interactions: bool = True,
    max_interaction_factors: int = 10,
    # Test data for overfitting detection (response/exposure auto-inferred from model)
    test_data: Optional["pl.DataFrame"] = None,
    # Design matrix for VIF calculation
    design_matrix: Optional[np.ndarray] = None,
    # Control which enhanced diagnostics to compute
    compute_vif: bool = True,
    compute_coefficients: bool = True,
    compute_deviance_by_level: bool = True,
    compute_lift: bool = True,
    compute_partial_dep: bool = True,
    # Legacy parameters (deprecated, use train_data instead)
    data: Optional["pl.DataFrame"] = None,
    test_response: Optional[str] = None,
    test_exposure: Optional[str] = None,
) -> ModelDiagnostics:
    """
    Compute comprehensive model diagnostics.
    
    Results are automatically saved to 'analysis/diagnostics.json'.
    
    Parameters
    ----------
    result : GLMResults or FormulaGLMResults
        Fitted model results.
    train_data : pl.DataFrame
        Training data used for fitting.
    categorical_factors : list of str, optional
        Names of categorical factors to analyze.
    continuous_factors : list of str, optional
        Names of continuous factors to analyze.
    n_calibration_bins : int, default=10
        Number of bins for calibration curve.
    n_factor_bins : int, default=10
        Number of quantile bins for continuous factors.
    rare_threshold_pct : float, default=1.0
        Threshold (%) below which categorical levels are grouped into "Other".
    max_categorical_levels : int, default=20
        Maximum number of categorical levels to show (rest grouped to "Other").
    detect_interactions : bool, default=True
        Whether to detect potential interactions.
    max_interaction_factors : int, default=10
        Maximum number of factors to consider for interaction detection.
    test_data : pl.DataFrame, optional
        Test/holdout data for overfitting detection. Response and exposure
        columns are automatically inferred from the model's formula.
    design_matrix : np.ndarray, optional
        Design matrix X for VIF calculation. If not provided, VIF is skipped.
    compute_vif : bool, default=True
        Whether to compute VIF/multicollinearity scores (train-only).
    compute_coefficients : bool, default=True
        Whether to compute coefficient summary with interpretations (train-only).
    compute_deviance_by_level : bool, default=True
        Whether to compute deviance breakdown by factor level.
    compute_lift : bool, default=True
        Whether to compute full lift chart.
    compute_partial_dep : bool, default=True
        Whether to compute partial dependence plots.
    
    Returns
    -------
    ModelDiagnostics
        Complete diagnostics object with to_json() method.
        
        Fields for agentic workflows:
        - vif: VIF scores for detecting multicollinearity (train-only)
        - coefficient_summary: Coefficient interpretations (train-only)
        - factor_deviance: Deviance breakdown by categorical levels
        - lift_chart: Full lift chart showing all deciles
        - partial_dependence: Marginal effect shapes for each variable
        - train_test: Comprehensive train vs test comparison with flags:
            - overfitting_risk: True if gini_gap > 0.03
            - calibration_drift: True if test A/E outside [0.95, 1.05]
            - unstable_factors: Factors where train/test A/E differ by > 0.1
    
    Examples
    --------
    >>> result = rs.glm("ClaimNb ~ Age + C(Region)", data, family="poisson", offset="Exposure").fit()
    >>> diagnostics = result.diagnostics(
    ...     train_data=train_data,
    ...     test_data=test_data,
    ...     categorical_factors=["Region", "VehBrand"],
    ...     continuous_factors=["Age", "VehPower"],
    ... )
    >>> 
    >>> # Check overfitting flags
    >>> if diagnostics.train_test and diagnostics.train_test.overfitting_risk:
    ...     print("Warning: Overfitting detected!")
    """
    # Support legacy 'data' parameter
    if train_data is None and data is not None:
        train_data = data
    
    # Deduplicate factors while preserving order
    categorical_factors = list(dict.fromkeys(categorical_factors or []))
    continuous_factors = list(dict.fromkeys(continuous_factors or []))
    # Remove any overlap (a factor can't be both categorical and continuous)
    continuous_factors = [f for f in continuous_factors if f not in categorical_factors]
    
    # Extract what we need from result
    # ALWAYS re-predict on train_data using result.predict() to get consistent encoding
    # This is critical for TE() which uses LOO encoding during fit but full encoding for predict
    # Using fittedvalues would give artificially high train loss due to LOO handicap
    formula_parts = result.formula.split('~') if hasattr(result, 'formula') else []
    response_col_temp = formula_parts[0].strip() if formula_parts else None
    
    if response_col_temp and response_col_temp in train_data.columns:
        y = train_data[response_col_temp].to_numpy().astype(np.float64)
        mu = np.asarray(result.predict(train_data), dtype=np.float64)
        lp = np.log(mu) if np.all(mu > 0) else mu
    else:
        # Fallback to fitted values if we can't determine response column
        mu = np.asarray(result.fittedvalues, dtype=np.float64)
        response_resid = np.asarray(result.resid_response(), dtype=np.float64)
        y = mu + response_resid
        lp = np.asarray(result.linear_predictor, dtype=np.float64)
    
    # Require essential attributes - fail loudly if missing
    if not hasattr(result, 'family'):
        raise ValueError("Result object missing 'family' attribute")
    if not hasattr(result, 'link'):
        raise ValueError("Result object missing 'link' attribute")
    if not hasattr(result, 'feature_names'):
        raise ValueError("Result object missing 'feature_names' attribute")
    
    family = result.family
    link = result.link
    n_params = len(result.params)
    deviance = result.deviance
    feature_names = result.feature_names
    
    # Auto-infer response and exposure column names from formula
    response_col = None
    exposure_col = None
    if hasattr(result, 'formula') and result.formula:
        # Parse response from formula (left side of ~)
        formula_parts = result.formula.split('~')
        if len(formula_parts) >= 1:
            response_col = formula_parts[0].strip()
    if hasattr(result, '_offset_spec') and isinstance(result._offset_spec, str):
        exposure_col = result._offset_spec
    
    # Legacy override if explicitly provided
    if test_response:
        response_col = test_response
    if test_exposure:
        exposure_col = test_exposure
    
    # Get exposure from training data
    exposure = None
    if exposure_col and exposure_col in train_data.columns:
        exposure = train_data[exposure_col].to_numpy().astype(np.float64)
    
    # Extract family parameters
    var_power = 1.5
    theta = 1.0
    if "tweedie" in family.lower():
        # Try to parse var_power from family string
        import re
        match = re.search(r'p=(\d+\.?\d*)', family)
        if match:
            var_power = float(match.group(1))
    if "negbinomial" in family.lower() or "negativebinomial" in family.lower():
        import re
        match = re.search(r'theta=(\d+\.?\d*)', family)
        if match:
            theta = float(match.group(1))
    
    # Get null deviance from model result (more accurate than recomputing)
    null_deviance = None
    if hasattr(result, 'null_deviance'):
        null_deviance = result.null_deviance() if callable(result.null_deviance) else result.null_deviance
    
    # Create computer
    computer = DiagnosticsComputer(
        y=y,
        mu=mu,
        linear_predictor=lp,
        family=family,
        n_params=n_params,
        deviance=deviance,
        exposure=exposure,
        feature_names=feature_names,
        var_power=var_power,
        theta=theta,
        null_deviance=null_deviance,
    )
    
    # Compute diagnostics
    calibration = computer.compute_calibration(n_calibration_bins)
    residual_summary = computer.compute_residual_summary()
    
    factors = computer.compute_factor_diagnostics(
        data=train_data,
        categorical_factors=categorical_factors,
        continuous_factors=continuous_factors,
        result=result,  # Pass result for significance tests
        n_bins=n_factor_bins,
        rare_threshold_pct=rare_threshold_pct,
        max_categorical_levels=max_categorical_levels,
    )
    
    # Interaction detection
    interaction_candidates = []
    if detect_interactions and len(categorical_factors) + len(continuous_factors) >= 2:
        all_factors = categorical_factors + continuous_factors
        interaction_candidates = computer.detect_interactions(
            data=train_data,
            factor_names=all_factors,
            max_factors=max_interaction_factors,
        )
    
    model_comparison = computer.compute_model_comparison()
    
    # Always compute train_test - this is the single source of truth for metrics
    exposure_train = computer.exposure
    train_diag = computer.compute_dataset_diagnostics(
        y, mu, exposure_train, train_data,
        categorical_factors, continuous_factors, "train", result
    )
    
    # Generate warnings (use train_diag for fit stats)
    fit_stats_for_warnings = {
        "deviance": train_diag.deviance,
        "aic": train_diag.aic,
        "log_likelihood": train_diag.log_likelihood,
    }
    warnings = computer.generate_warnings(fit_stats_for_warnings, calibration, factors, family=family)
    
    # =========================================================================
    # NEW: Enhanced diagnostics for agentic workflows
    # =========================================================================
    
    # VIF / Multicollinearity
    # Token optimization: VIF array already contains all info, no separate warnings needed
    vif_results = None
    if compute_vif and design_matrix is not None:
        vif_results = computer.compute_vif(design_matrix, feature_names)
    
    # Coefficient summary
    coef_summary = None
    if compute_coefficients:
        coef_summary = computer.compute_coefficient_summary(result, link=link)
        # Token optimization: skip weak_predictors warning (agent can infer from sig=False + rel~1.0)
    
    # Deviance by factor level
    factor_dev = None
    if compute_deviance_by_level and categorical_factors:
        factor_dev = computer.compute_factor_deviance(train_data, categorical_factors)
        # Add warnings for problem levels
        for fd in factor_dev:
            if fd.problem_levels:
                warnings.append({
                    "type": "problem_factor_levels",
                    "message": f"Factor '{fd.factor}' has problem levels with poor fit: "
                              f"{', '.join(fd.problem_levels[:5])}{'...' if len(fd.problem_levels) > 5 else ''}"
                })
    
    # Lift chart
    lift_chart = None
    if compute_lift:
        lift_chart = computer.compute_lift_chart(n_deciles=10)
        # Add warnings for weak discrimination
        if lift_chart.weak_deciles:
            warnings.append({
                "type": "weak_discrimination",
                "message": f"Model has weak discrimination in deciles: {lift_chart.weak_deciles}. "
                          f"Consider adding features or interactions to improve separation."
            })
    
    # Partial dependence
    partial_dep = None
    if compute_partial_dep and (continuous_factors or categorical_factors):
        partial_dep = computer.compute_partial_dependence(
            data=train_data,
            result=result,
            continuous_factors=continuous_factors,
            categorical_factors=categorical_factors,
            link=link,
        )
        # Add recommendations for non-linear effects
        for pd in partial_dep:
            if pd.shape in ("u_shaped", "inverted_u", "complex") and "spline" in pd.recommendation.lower():
                warnings.append({
                    "type": "nonlinear_effect",
                    "message": f"Variable '{pd.variable}' shows {pd.shape} pattern. {pd.recommendation}"
                })
    
    # Build train_test (train is always present, test is optional)
    train_test = TrainTestComparison(train=train_diag)
    
    if test_data is not None and response_col is not None:
        # Get test response
        if response_col not in test_data.columns:
            raise ValueError(f"Response column '{response_col}' not found in test_data")
        y_test = test_data[response_col].to_numpy().astype(np.float64)
        
        # Get test predictions
        if not hasattr(result, 'predict'):
            raise ValueError("Model does not support prediction on new data")
        mu_test = result.predict(test_data)
        
        # Get test exposure
        exposure_test = np.ones(len(y_test))
        if exposure_col and exposure_col in test_data.columns:
            exposure_test = test_data[exposure_col].to_numpy().astype(np.float64)
        
        # Compute test diagnostics
        test_diag = computer.compute_dataset_diagnostics(
            y_test, mu_test, exposure_test, test_data,
            categorical_factors, continuous_factors, "test", result
        )
        
        # Compute comparison metrics
        gini_gap = train_diag.gini - test_diag.gini
        ae_ratio_diff = abs(train_diag.ae_ratio - test_diag.ae_ratio)
        
        # Decile comparison
        decile_comparison = []
        for i in range(min(len(train_diag.ae_by_decile), len(test_diag.ae_by_decile))):
            train_d = train_diag.ae_by_decile[i]
            test_d = test_diag.ae_by_decile[i]
            decile_comparison.append({
                "decile": i + 1,
                "train_ae": train_d.ae_ratio,
                "test_ae": test_d.ae_ratio,
                "ae_diff": round(abs((train_d.ae_ratio or 0) - (test_d.ae_ratio or 0)), 4),
            })
        
        # Factor divergence
        factor_divergence = {}
        unstable_factors_list = []
        for factor in categorical_factors:
            if factor in train_diag.factor_diagnostics and factor in test_diag.factor_diagnostics:
                train_levels = {m.level: m for m in train_diag.factor_diagnostics[factor]}
                test_levels = {m.level: m for m in test_diag.factor_diagnostics[factor]}
                divergent = []
                for level in set(train_levels.keys()) | set(test_levels.keys()):
                    tr_ae = train_levels.get(level, FactorLevelMetrics(level, 0, 0, 0, 0, None, 0)).ae_ratio
                    te_ae = test_levels.get(level, FactorLevelMetrics(level, 0, 0, 0, 0, None, 0)).ae_ratio
                    if tr_ae is not None and te_ae is not None:
                        diff = abs(tr_ae - te_ae)
                        if diff > 0.1:
                            divergent.append({"level": level, "train_ae": tr_ae, "test_ae": te_ae, "ae_diff": round(diff, 4)})
                            unstable_factors_list.append(f"{factor}[{level}]")
                if divergent:
                    factor_divergence[factor] = divergent
        
        # Flags
        overfitting_risk = gini_gap > 0.03
        calibration_drift = test_diag.ae_ratio < 0.95 or test_diag.ae_ratio > 1.05
        
        train_test = TrainTestComparison(
            train=train_diag,
            test=test_diag,
            gini_gap=round(gini_gap, 4),
            ae_ratio_diff=round(ae_ratio_diff, 4),
            decile_comparison=decile_comparison,
            factor_divergence=factor_divergence,
            overfitting_risk=overfitting_risk,
            calibration_drift=calibration_drift,
            unstable_factors=unstable_factors_list,
        )
        
        # Add warnings based on flags
        if overfitting_risk:
            warnings.append({
                "type": "overfitting",
                "message": f"Overfitting detected: Train Gini={train_diag.gini:.3f}, "
                          f"Test Gini={test_diag.gini:.3f} (gap={gini_gap:.3f}). "
                          f"Consider reducing model complexity or using regularization."
            })
        if calibration_drift:
            warnings.append({
                "type": "calibration_drift",
                "message": f"Calibration drift: Test A/E={test_diag.ae_ratio:.3f} "
                          f"(outside [0.95, 1.05]). Model may not generalize well."
            })
        if unstable_factors_list:
            warnings.append({
                "type": "unstable_factors",
                "message": f"Unstable factor levels (train/test A/E differ by >0.1): "
                          f"{', '.join(unstable_factors_list[:10])}"
                          f"{'...' if len(unstable_factors_list) > 10 else ''}"
            })
    
    # Extract convergence info - require these attributes
    if not hasattr(result, 'converged'):
        raise ValueError("Result object missing 'converged' attribute")
    if not hasattr(result, 'iterations'):
        raise ValueError("Result object missing 'iterations' attribute")
    if not hasattr(result, 'formula'):
        raise ValueError("Result object missing 'formula' attribute")
    
    converged = result.converged
    iterations = result.iterations
    
    # Model summary
    model_summary = {
        "formula": result.formula,
        "family": family,
        "link": link,
        "n_obs": computer.n_obs,
        "n_params": n_params,
        "df_resid": computer.df_resid,
        "converged": converged,
        "iterations": iterations,
    }
    
    # Add regularization info if present (concise for LLM parsing)
    if hasattr(result, 'alpha') and result.alpha > 0:
        reg_type = getattr(result, 'regularization_type', None)
        if reg_type is None:
            l1 = getattr(result, 'l1_ratio', 0)
            reg_type = "lasso" if l1 >= 1 else "ridge" if l1 <= 0 else "elastic_net"
        model_summary["regularization"] = {
            "type": reg_type,
            "alpha": round(result.alpha, 6),
            "l1_ratio": round(getattr(result, 'l1_ratio', 0), 2),
        }
        # Add CV info if available
        if hasattr(result, 'cv_deviance') and result.cv_deviance is not None:
            model_summary["regularization"]["cv_deviance"] = round(result.cv_deviance, 6)
            model_summary["regularization"]["cv_folds"] = getattr(result, 'n_cv_folds', None)
            model_summary["regularization"]["selection"] = getattr(result, 'cv_selection_method', None)
    
    # Compute overdispersion (for Poisson/Binomial families)
    overdispersion_result = None
    family_lower = family.lower()
    if any(f in family_lower for f in ["poisson", "binomial", "negativebinomial"]):
        # Model-based dispersion: Pearson chi-squared / df_resid
        pearson_chi2 = result.pearson_chi2() if hasattr(result, 'pearson_chi2') else None
        df_resid = computer.df_resid
        
        if pearson_chi2 is not None and df_resid > 0:
            pearson_dispersion = pearson_chi2 / df_resid
            
            # Also compute raw dispersion from data (Var/Mean for counts)
            mean_count = float(np.mean(y))
            var_count = float(np.var(y, ddof=1)) if len(y) > 1 else 0.0
            raw_dispersion = var_count / mean_count if mean_count > 0 else 1.0
            
            # Severity based on Pearson dispersion (more reliable)
            if pearson_dispersion > 5:
                severity = "severe"
                recommendation = "Use Negative Binomial or QuasiPoisson"
            elif pearson_dispersion > 2:
                severity = "moderate"
                recommendation = "Consider Negative Binomial or QuasiPoisson"
            elif pearson_dispersion > 1.5:
                severity = "mild"
                recommendation = "Monitor; Poisson may underestimate standard errors"
            else:
                severity = "none"
                recommendation = "Poisson assumption appears reasonable"
            
            overdispersion_result = {
                "pearson_dispersion": round(pearson_dispersion, 4),
                "pearson_chi2": round(pearson_chi2, 2),
                "df_resid": df_resid,
                "raw_dispersion": round(raw_dispersion, 4),
                "mean_count": round(mean_count, 4),
                "var_count": round(var_count, 4),
                "severity": severity,
                "recommendation": recommendation,
            }
            
            # Add warning if overdispersed
            if pearson_dispersion > 1.5:
                warnings.append({
                    "type": "overdispersion",
                    "message": f"Overdispersion detected (φ={pearson_dispersion:.2f}). {recommendation}"
                })
    
    # Get spline knot information if available
    spline_info = None
    if hasattr(result, '_builder') and hasattr(result._builder, 'get_spline_info'):
        spline_info = result._builder.get_spline_info()
        if not spline_info:  # Empty dict -> None
            spline_info = None
    
    diagnostics = ModelDiagnostics(
        model_summary=model_summary,
        train_test=train_test,
        calibration=calibration,
        residual_summary=residual_summary,
        factors=factors,
        interaction_candidates=interaction_candidates,
        model_comparison=model_comparison,
        warnings=warnings,
        vif=vif_results,
        coefficient_summary=coef_summary,
        factor_deviance=factor_dev,
        lift_chart=lift_chart,
        partial_dependence=partial_dep,
        overdispersion=overdispersion_result,
        spline_info=spline_info,
    )
    
    # Auto-save JSON to analysis folder
    import os
    os.makedirs("analysis", exist_ok=True)
    with open("analysis/diagnostics.json", "w") as f:
        f.write(diagnostics.to_json(indent=2))
    
    return diagnostics
