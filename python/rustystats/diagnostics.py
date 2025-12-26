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
    compute_residual_summary_py as _rust_residual_summary,
    compute_residual_pattern_py as _rust_residual_pattern,
    compute_pearson_residuals_py as _rust_pearson_residuals,
    compute_deviance_residuals_py as _rust_deviance_residuals,
    compute_null_deviance_py as _rust_null_deviance,
    compute_unit_deviance_py as _rust_unit_deviance,
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
    """Percentile values for a continuous variable."""
    p1: float
    p5: float
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float
    p95: float
    p99: float


@dataclass
class ResidualSummary:
    """Summary statistics for residuals."""
    mean: float
    std: float
    min: float
    max: float
    skewness: float
    kurtosis: float
    percentiles: Percentiles


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
    """A/E statistics for a single bin or categorical level."""
    bin_index: int
    bin_label: str
    bin_lower: Optional[float]
    bin_upper: Optional[float]
    count: int
    exposure: float
    actual_sum: float
    predicted_sum: float
    actual_mean: float
    predicted_mean: float
    actual_expected_ratio: float
    loss: float
    ae_confidence_interval_lower: float
    ae_confidence_interval_upper: float


@dataclass
class ResidualPattern:
    """Residual pattern analysis for a factor."""
    correlation_with_residuals: float
    mean_residual_by_bin: List[float]
    trend_slope: float
    trend_pvalue: float
    residual_variance_explained: float


@dataclass
class ContinuousFactorStats:
    """Univariate statistics for a continuous factor."""
    mean: float
    std: float
    min: float
    max: float
    missing_count: int
    percentiles: Percentiles


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
    levels: List[CategoricalLevelStats]
    n_rare_levels: int
    rare_level_total_pct: float


@dataclass
class FactorDiagnostics:
    """Complete diagnostics for a single factor."""
    name: str
    factor_type: str  # "continuous" or "categorical"
    in_model: bool
    transformation: Optional[str]
    univariate_stats: Union[ContinuousFactorStats, CategoricalFactorStats]
    actual_vs_expected: List[ActualExpectedBin]
    residual_pattern: ResidualPattern


@dataclass
class InteractionCandidate:
    """A potential interaction between two factors."""
    factor1: str
    factor2: str
    interaction_strength: float
    pvalue: float
    n_cells: int


@dataclass
class DataExploration:
    """Pre-fit data exploration results."""
    
    # Data summary
    data_summary: Dict[str, Any]
    
    # Factor statistics
    factor_stats: List[Dict[str, Any]]
    
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
    
    # Model summary
    model_summary: Dict[str, Any]
    
    # Fit statistics
    fit_statistics: Dict[str, float]
    
    # Loss metrics
    loss_metrics: Dict[str, float]
    
    # Calibration
    calibration: Dict[str, Any]
    
    # Discrimination (only for applicable models)
    discrimination: Optional[Dict[str, Any]]
    
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
        return obj
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    else:
        return obj


# =============================================================================
# Main Diagnostics Computation
# =============================================================================

class DiagnosticsComputer:
    """
    Computes comprehensive model diagnostics.
    
    This class is instantiated with fitted model results and computes
    all diagnostics on demand with caching for efficiency.
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
    ):
        self.y = np.asarray(y, dtype=np.float64)
        self.mu = np.asarray(mu, dtype=np.float64)
        self.linear_predictor = np.asarray(linear_predictor, dtype=np.float64)
        self.family = family.lower()
        self.n_params = n_params
        self.deviance = deviance
        self.exposure = np.asarray(exposure, dtype=np.float64) if exposure is not None else np.ones_like(y)
        self.feature_names = feature_names or []
        self.var_power = var_power
        self.theta = theta
        
        self.n_obs = len(y)
        self.df_resid = self.n_obs - n_params
        
        # Cached computations
        self._pearson_residuals: Optional[np.ndarray] = None
        self._deviance_residuals: Optional[np.ndarray] = None
        self._null_deviance: Optional[float] = None
    
    @property
    def pearson_residuals(self) -> np.ndarray:
        """Pearson residuals using Rust backend."""
        if self._pearson_residuals is None:
            self._pearson_residuals = np.asarray(_rust_pearson_residuals(self.y, self.mu, self.family))
        return self._pearson_residuals
    
    @property
    def deviance_residuals(self) -> np.ndarray:
        """Deviance residuals using Rust backend."""
        if self._deviance_residuals is None:
            self._deviance_residuals = np.asarray(_rust_deviance_residuals(self.y, self.mu, self.family))
        return self._deviance_residuals
    
    @property
    def null_deviance(self) -> float:
        """Null deviance using Rust backend."""
        if self._null_deviance is None:
            self._null_deviance = _rust_null_deviance(self.y, self.family, self.exposure)
        return self._null_deviance
    
    def _compute_unit_deviance(self, y: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """Compute unit deviance using Rust backend."""
        return np.asarray(_rust_unit_deviance(y, mu, self.family))
    
    def _compute_loss(self, y: np.ndarray, mu: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
        """Compute family-specific loss."""
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
        weighted_loss = np.average(self._compute_unit_deviance(self.y, self.mu), weights=self.exposure)
        return {
            "family_deviance_loss": rust_loss["family_loss"],
            "mse": rust_loss["mse"],
            "mae": rust_loss["mae"],
            "rmse": rust_loss["rmse"],
            "weighted_loss": weighted_loss,
        }
    
    def compute_calibration(self, n_bins: int = 10) -> Dict[str, Any]:
        """Compute calibration metrics using Rust backend."""
        actual_total = float(np.sum(self.y))
        predicted_total = float(np.sum(self.mu))
        exposure_total = float(np.sum(self.exposure))
        ae_ratio = actual_total / predicted_total if predicted_total > 0 else float('nan')
        
        bins = self._compute_calibration_bins(n_bins)
        hl_stat, hl_pvalue = self._hosmer_lemeshow_test(n_bins)
        
        return {
            "actual_expected_ratio": ae_ratio,
            "actual_total": actual_total,
            "predicted_total": predicted_total,
            "exposure_total": exposure_total,
            "calibration_error": abs(1 - ae_ratio) if not np.isnan(ae_ratio) else float('nan'),
            "by_decile": [asdict(b) for b in bins],
            "hosmer_lemeshow_statistic": hl_stat,
            "hosmer_lemeshow_pvalue": hl_pvalue,
        }
    
    def _compute_calibration_bins(self, n_bins: int) -> List[CalibrationBin]:
        """Compute calibration bins using Rust backend."""
        rust_bins = _rust_calibration_curve(self.y, self.mu, self.exposure, n_bins)
        return [
            CalibrationBin(
                bin_index=b["bin_index"],
                predicted_lower=b["predicted_lower"],
                predicted_upper=b["predicted_upper"],
                predicted_mean=b["predicted_mean"],
                actual_mean=b["actual_mean"],
                actual_expected_ratio=b["actual_expected_ratio"],
                count=b["count"],
                exposure=b["exposure"],
                actual_sum=b["actual_sum"],
                predicted_sum=b["predicted_sum"],
                ae_confidence_interval_lower=b["ae_ci_lower"],
                ae_confidence_interval_upper=b["ae_ci_upper"],
            )
            for b in rust_bins
        ]
    
    def _hosmer_lemeshow_test(self, n_bins: int = 10) -> tuple:
        """Compute Hosmer-Lemeshow test using Rust backend."""
        result = _rust_hosmer_lemeshow(self.y, self.mu, n_bins)
        return result["chi2_statistic"], result["pvalue"]
    
    def compute_discrimination(self) -> Optional[Dict[str, Any]]:
        """Compute discrimination metrics using Rust backend."""
        stats = _rust_discrimination_stats(self.y, self.mu, self.exposure)
        lorenz_points = _rust_lorenz_curve(self.y, self.mu, self.exposure, 20)
        
        return {
            "gini_coefficient": stats["gini"],
            "auc": stats["auc"],
            "ks_statistic": stats["ks_statistic"],
            "lift_at_10pct": stats["lift_at_10pct"],
            "lift_at_20pct": stats["lift_at_20pct"],
            "lorenz_curve": lorenz_points,
        }
    
    def compute_residual_summary(self) -> Dict[str, ResidualSummary]:
        """Compute residual summary statistics using Rust backend."""
        def summarize(resid: np.ndarray) -> ResidualSummary:
            stats = _rust_residual_summary(resid)
            return ResidualSummary(
                mean=stats["mean"],
                std=stats["std"],
                min=stats["min"],
                max=stats["max"],
                skewness=stats["skewness"],
                kurtosis=stats["kurtosis"],
                percentiles=Percentiles(
                    p1=stats["p1"], p5=stats["p5"], p10=stats["p10"],
                    p25=stats["p25"], p50=stats["p50"], p75=stats["p75"],
                    p90=stats["p90"], p95=stats["p95"], p99=stats["p99"],
                ),
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
        n_bins: int = 10,
        rare_threshold_pct: float = 1.0,
        max_categorical_levels: int = 20,
    ) -> List[FactorDiagnostics]:
        """Compute diagnostics for each specified factor."""
        factors = []
        
        # Process categorical factors
        for name in categorical_factors:
            if name not in data.columns:
                continue
            
            values = data[name].to_numpy().astype(str)
            in_model = any(name in fn for fn in self.feature_names)
            
            # Univariate stats
            unique, counts = np.unique(values, return_counts=True)
            total = len(values)
            levels = [CategoricalLevelStats(
                level=str(u),
                count=int(c),
                percentage=100.0 * c / total
            ) for u, c in sorted(zip(unique, counts), key=lambda x: -x[1])]
            
            n_rare = sum(1 for l in levels if l.percentage < rare_threshold_pct)
            rare_pct = sum(l.percentage for l in levels if l.percentage < rare_threshold_pct)
            
            univariate = CategoricalFactorStats(
                n_levels=len(unique),
                levels=levels[:max_categorical_levels],
                n_rare_levels=n_rare,
                rare_level_total_pct=rare_pct,
            )
            
            # A/E by level
            ae_bins = self._compute_ae_categorical(
                values, rare_threshold_pct, max_categorical_levels
            )
            
            # Residual pattern
            resid_pattern = self._compute_residual_pattern_categorical(values)
            
            factors.append(FactorDiagnostics(
                name=name,
                factor_type="categorical",
                in_model=in_model,
                transformation=self._get_transformation(name),
                univariate_stats=univariate,
                actual_vs_expected=ae_bins,
                residual_pattern=resid_pattern,
            ))
        
        # Process continuous factors
        for name in continuous_factors:
            if name not in data.columns:
                continue
            
            values = data[name].to_numpy().astype(np.float64)
            in_model = any(name in fn for fn in self.feature_names)
            
            # Univariate stats - batch percentile calculation
            valid_mask = ~np.isnan(values) & ~np.isinf(values)
            valid = values[valid_mask]
            
            if len(valid) > 0:
                # Single batched percentile call (much faster)
                pcts = np.percentile(valid, [1, 5, 10, 25, 50, 75, 90, 95, 99])
                percentiles = Percentiles(
                    p1=float(pcts[0]), p5=float(pcts[1]), p10=float(pcts[2]),
                    p25=float(pcts[3]), p50=float(pcts[4]), p75=float(pcts[5]),
                    p90=float(pcts[6]), p95=float(pcts[7]), p99=float(pcts[8]),
                )
                univariate = ContinuousFactorStats(
                    mean=float(np.mean(valid)),
                    std=float(np.std(valid)),
                    min=float(pcts[0]),  # Reuse min from percentiles
                    max=float(np.max(valid)),
                    missing_count=int(np.sum(~valid_mask)),
                    percentiles=percentiles,
                )
            else:
                nan = float('nan')
                percentiles = Percentiles(p1=nan, p5=nan, p10=nan, p25=nan, p50=nan, p75=nan, p90=nan, p95=nan, p99=nan)
                univariate = ContinuousFactorStats(mean=nan, std=nan, min=nan, max=nan, missing_count=len(values), percentiles=percentiles)
            
            # A/E by quantile bins
            ae_bins = self._compute_ae_continuous(values, n_bins)
            
            # Residual pattern
            resid_pattern = self._compute_residual_pattern_continuous(values, n_bins)
            
            factors.append(FactorDiagnostics(
                name=name,
                factor_type="continuous",
                in_model=in_model,
                transformation=self._get_transformation(name),
                univariate_stats=univariate,
                actual_vs_expected=ae_bins,
                residual_pattern=resid_pattern,
            ))
        
        return factors
    
    def _get_transformation(self, name: str) -> Optional[str]:
        """Find transformation for a factor in the model."""
        for fn in self.feature_names:
            if name in fn and fn != name:
                return fn
        return None
    
    def _compute_ae_continuous(self, values: np.ndarray, n_bins: int) -> List[ActualExpectedBin]:
        """Compute A/E for continuous factor using Rust backend."""
        rust_bins = _rust_ae_continuous(values, self.y, self.mu, self.exposure, n_bins, self.family)
        return [
            ActualExpectedBin(
                bin_index=b["bin_index"],
                bin_label=b["bin_label"],
                bin_lower=b.get("bin_lower"),
                bin_upper=b.get("bin_upper"),
                count=b["count"],
                exposure=b["exposure"],
                actual_sum=b["actual_sum"],
                predicted_sum=b["predicted_sum"],
                actual_mean=b["actual_mean"],
                predicted_mean=b["predicted_mean"],
                actual_expected_ratio=b["actual_expected_ratio"],
                loss=b["loss"],
                ae_confidence_interval_lower=b["ae_ci_lower"],
                ae_confidence_interval_upper=b["ae_ci_upper"],
            )
            for b in rust_bins
        ]
    
    def _compute_ae_categorical(
        self,
        values: np.ndarray,
        rare_threshold_pct: float,
        max_levels: int,
    ) -> List[ActualExpectedBin]:
        """Compute A/E for categorical factor using Rust backend."""
        levels = [str(v) for v in values]
        rust_bins = _rust_ae_categorical(levels, self.y, self.mu, self.exposure, 
                                          rare_threshold_pct, max_levels, self.family)
        return [
            ActualExpectedBin(
                bin_index=b["bin_index"],
                bin_label=b["bin_label"],
                bin_lower=None,
                bin_upper=None,
                count=b["count"],
                exposure=b["exposure"],
                actual_sum=b["actual_sum"],
                predicted_sum=b["predicted_sum"],
                actual_mean=b["actual_mean"],
                predicted_mean=b["predicted_mean"],
                actual_expected_ratio=b["actual_expected_ratio"],
                loss=b["loss"],
                ae_confidence_interval_lower=b["ae_ci_lower"],
                ae_confidence_interval_upper=b["ae_ci_upper"],
            )
            for b in rust_bins
        ]
    
    def _compute_residual_pattern_continuous(
        self,
        values: np.ndarray,
        n_bins: int,
    ) -> ResidualPattern:
        """Compute residual pattern using Rust backend."""
        valid_mask = ~np.isnan(values) & ~np.isinf(values)
        
        if not np.any(valid_mask):
            return ResidualPattern(
                correlation_with_residuals=float('nan'),
                mean_residual_by_bin=[],
                trend_slope=float('nan'),
                trend_pvalue=float('nan'),
                residual_variance_explained=float('nan'),
            )
        
        result = _rust_residual_pattern(values, self.pearson_residuals, n_bins)
        corr = result["correlation_with_residuals"]
        mean_bins = [b["mean_residual"] for b in result["mean_residual_by_bin"]]
        
        return ResidualPattern(
            correlation_with_residuals=float(corr) if not np.isnan(corr) else 0.0,
            mean_residual_by_bin=mean_bins,
            trend_slope=float('nan'),
            trend_pvalue=float('nan'),
            residual_variance_explained=corr ** 2 if not np.isnan(corr) else 0.0,
        )
    
    def _compute_residual_pattern_categorical(self, values: np.ndarray) -> ResidualPattern:
        """Compute residual pattern for categorical factor."""
        unique_levels = np.unique(values)
        
        # Mean residual by level
        level_means = []
        overall_mean = np.mean(self.pearson_residuals)
        ss_total = np.sum((self.pearson_residuals - overall_mean) ** 2)
        ss_between = 0.0
        
        for level in unique_levels:
            mask = values == level
            level_resid = self.pearson_residuals[mask]
            level_mean = np.mean(level_resid)
            level_means.append(float(level_mean))
            ss_between += len(level_resid) * (level_mean - overall_mean) ** 2
        
        # Eta-squared
        eta_squared = ss_between / ss_total if ss_total > 0 else 0.0
        
        # Mean absolute residual as proxy for correlation
        mean_abs_resid = np.mean(np.abs(level_means))
        
        return ResidualPattern(
            correlation_with_residuals=float(mean_abs_resid),
            mean_residual_by_bin=level_means[:20],  # Limit to 20 levels
            trend_slope=float('nan'),  # Not applicable
            trend_pvalue=float('nan'),
            residual_variance_explained=float(eta_squared),
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
        
        try:
            from scipy.stats import t
            pvalue = 2 * (1 - t.cdf(abs(t_stat), df))
        except ImportError:
            pvalue = float('nan')
        
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
                continue
            
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
                    candidates.append(candidate)
        
        # Sort by strength and return top candidates
        candidates.sort(key=lambda x: -x.interaction_strength)
        return candidates[:max_candidates]
    
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
            
            try:
                from scipy.stats import f
                pvalue = 1 - f.cdf(f_stat, df_model, df_resid)
            except ImportError:
                pvalue = float('nan')
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
        
        try:
            from scipy.stats import chi2
            lr_pvalue = 1 - chi2.cdf(lr_chi2, lr_df) if lr_df > 0 else float('nan')
        except ImportError:
            lr_pvalue = float('nan')
        
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
    ) -> List[Dict[str, str]]:
        """Generate warnings based on diagnostics."""
        warnings = []
        
        # High dispersion warning
        dispersion = fit_stats.get("dispersion_pearson", 1.0)
        if dispersion > 1.5:
            warnings.append({
                "type": "high_dispersion",
                "message": f"Dispersion {dispersion:.2f} suggests overdispersion. Consider quasipoisson or negbinomial."
            })
        
        # Poor overall calibration
        ae_ratio = calibration.get("actual_expected_ratio", 1.0)
        if abs(ae_ratio - 1.0) > 0.05:
            direction = "over" if ae_ratio < 1 else "under"
            warnings.append({
                "type": "poor_overall_calibration",
                "message": f"Model {direction}-predicts overall (A/E = {ae_ratio:.3f})."
            })
        
        # Extreme calibration bins
        for bin in calibration.get("by_decile", []):
            if isinstance(bin, dict):
                ae = bin.get("actual_expected_ratio", 1.0)
                if ae is not None and abs(ae - 1.0) > 0.3:
                    warnings.append({
                        "type": "poor_bin_calibration",
                        "message": f"Decile {bin.get('bin_index', '?')} has A/E = {ae:.2f}."
                    })
        
        # Factors with high residual correlation (not in model)
        for factor in factors:
            if not factor.in_model:
                corr = factor.residual_pattern.correlation_with_residuals
                r2 = factor.residual_pattern.residual_variance_explained
                if r2 > 0.02:
                    warnings.append({
                        "type": "missing_factor",
                        "message": f"Factor '{factor.name}' not in model but explains {100*r2:.1f}% of residual variance."
                    })
        
        return warnings


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
                continue
            
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
            
            for i in range(n_bins):
                if i == n_bins - 1:
                    bin_mask = (values >= quantiles[i]) & (values <= quantiles[i + 1])
                else:
                    bin_mask = (values >= quantiles[i]) & (values < quantiles[i + 1])
                
                if not np.any(bin_mask):
                    continue
                
                y_bin = self.y[bin_mask]
                exp_bin = self.exposure[bin_mask]
                
                bins_data.append({
                    "bin_index": i,
                    "bin_lower": float(quantiles[i]),
                    "bin_upper": float(quantiles[i + 1]),
                    "count": int(np.sum(bin_mask)),
                    "exposure": float(np.sum(exp_bin)),
                    "response_sum": float(np.sum(y_bin)),
                    "response_rate": float(np.sum(y_bin) / np.sum(exp_bin)) if np.sum(exp_bin) > 0 else 0,
                })
            
            stats["response_by_bin"] = bins_data
            factors.append(stats)
        
        # Categorical factors
        for name in categorical_factors:
            if name not in data.columns:
                continue
            
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
            
            stats = {
                "name": name,
                "type": "categorical",
                "n_levels": len(unique_levels),
                "n_levels_shown": len(levels_data),
                "levels": levels_data,
            }
            factors.append(stats)
        
        return factors
    
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
                continue
            
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
            
            try:
                from scipy.stats import f
                pvalue = 1 - f.cdf(f_stat, df_model, df_resid)
            except ImportError:
                pvalue = float('nan')
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
    categorical_factors = categorical_factors or []
    continuous_factors = continuous_factors or []
    
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
    
    # Interaction detection
    interaction_candidates = []
    if detect_interactions and len(categorical_factors) + len(continuous_factors) >= 2:
        all_factors = categorical_factors + continuous_factors
        interaction_candidates = explorer.detect_interactions(
            data=data,
            factor_names=all_factors,
            max_factors=max_interaction_factors,
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
    
    return DataExploration(
        data_summary=data_summary,
        factor_stats=factor_stats,
        interaction_candidates=interaction_candidates,
        response_stats=response_stats,
    )


# =============================================================================
# Post-Fit Model Diagnostics
# =============================================================================

def compute_diagnostics(
    result,  # GLMResults or FormulaGLMResults
    data: "pl.DataFrame",
    categorical_factors: Optional[List[str]] = None,
    continuous_factors: Optional[List[str]] = None,
    n_calibration_bins: int = 10,
    n_factor_bins: int = 10,
    rare_threshold_pct: float = 1.0,
    max_categorical_levels: int = 20,
    detect_interactions: bool = True,
    max_interaction_factors: int = 10,
) -> ModelDiagnostics:
    """
    Compute comprehensive model diagnostics.
    
    Parameters
    ----------
    result : GLMResults or FormulaGLMResults
        Fitted model results.
    data : pl.DataFrame
        Original data used for fitting.
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
    
    Returns
    -------
    ModelDiagnostics
        Complete diagnostics object with to_json() method.
    """
    categorical_factors = categorical_factors or []
    continuous_factors = continuous_factors or []
    
    # Extract what we need from result
    # Get y from the residuals (y = mu + response_residuals)
    mu = np.asarray(result.fittedvalues, dtype=np.float64)
    response_resid = np.asarray(result.resid_response(), dtype=np.float64)
    y = mu + response_resid
    
    lp = np.asarray(result.linear_predictor, dtype=np.float64)
    family = result.family if hasattr(result, 'family') else "unknown"
    n_params = len(result.params)
    deviance = result.deviance
    feature_names = result.feature_names if hasattr(result, 'feature_names') else []
    
    # Try to get exposure from data if weights column exists
    exposure = None
    
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
    )
    
    # Compute all diagnostics
    fit_stats = computer.compute_fit_statistics()
    loss_metrics = computer.compute_loss_metrics()
    calibration = computer.compute_calibration(n_calibration_bins)
    discrimination = computer.compute_discrimination()
    residual_summary = computer.compute_residual_summary()
    
    factors = computer.compute_factor_diagnostics(
        data=data,
        categorical_factors=categorical_factors,
        continuous_factors=continuous_factors,
        n_bins=n_factor_bins,
        rare_threshold_pct=rare_threshold_pct,
        max_categorical_levels=max_categorical_levels,
    )
    
    # Interaction detection
    interaction_candidates = []
    if detect_interactions and len(categorical_factors) + len(continuous_factors) >= 2:
        all_factors = categorical_factors + continuous_factors
        interaction_candidates = computer.detect_interactions(
            data=data,
            factor_names=all_factors,
            max_factors=max_interaction_factors,
        )
    
    model_comparison = computer.compute_model_comparison()
    warnings = computer.generate_warnings(fit_stats, calibration, factors)
    
    # Model summary
    model_summary = {
        "family": family,
        "link": result.link if hasattr(result, 'link') else "unknown",
        "n_observations": computer.n_obs,
        "n_parameters": n_params,
        "degrees_of_freedom_residual": computer.df_resid,
        "converged": result.converged if hasattr(result, 'converged') else True,
        "iterations": result.iterations if hasattr(result, 'iterations') else 0,
    }
    
    return ModelDiagnostics(
        model_summary=model_summary,
        fit_statistics=fit_stats,
        loss_metrics=loss_metrics,
        calibration=calibration,
        discrimination=discrimination,
        residual_summary=residual_summary,
        factors=factors,
        interaction_candidates=interaction_candidates,
        model_comparison=model_comparison,
        warnings=warnings,
    )
