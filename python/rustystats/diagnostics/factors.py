"""
Factor-level diagnostic computations.

_FactorDiagnosticsComputer handles per-factor analysis including:
- Actual vs Expected (A/E) by level/bin
- Residual patterns
- Factor significance (Wald chi-square)
- Score tests for unfitted factors
- Coefficient extraction
"""

from __future__ import annotations

import re
from typing import Optional, List, TYPE_CHECKING

import numpy as np

from rustystats._rustystats import (
    compute_ae_continuous_py as _rust_ae_continuous,
    compute_ae_categorical_py as _rust_ae_categorical,
    compute_residual_pattern_py as _rust_residual_pattern,
    chi2_cdf_py as _chi2_cdf,
    t_cdf_py as _t_cdf,
    score_test_continuous_py as _rust_score_test_continuous,
    score_test_categorical_py as _rust_score_test_categorical,
)

from rustystats.diagnostics.types import (
    ActualExpectedBin,
    ResidualPattern,
    ContinuousFactorStats,
    CategoricalFactorStats,
    FactorSignificance,
    ScoreTestResult,
    FactorCoefficient,
    FactorDiagnostics,
)

from rustystats.diagnostics.utils import validate_factor_in_data
from rustystats.exceptions import FittingError
from rustystats.constants import (
    EPSILON,
    DEFAULT_N_FACTOR_BINS,
    DEFAULT_RARE_THRESHOLD_PCT,
    DEFAULT_MAX_CATEGORICAL_LEVELS,
)

if TYPE_CHECKING:
    import polars as pl


class _FactorDiagnosticsComputer:
    """Computes per-factor diagnostics for fitted GLM models.
    
    Requires arrays from the parent DiagnosticsComputer: y, mu, exposure,
    pearson_residuals, feature_names, and family string.
    """
    
    def __init__(
        self,
        y: np.ndarray,
        mu: np.ndarray,
        exposure: np.ndarray,
        pearson_residuals: np.ndarray,
        feature_names: List[str],
        family: str,
    ):
        self.y = y
        self.mu = mu
        self.exposure = exposure
        self.pearson_residuals = pearson_residuals
        self.feature_names = feature_names
        self.family = family
    
    def compute_factor_diagnostics(
        self,
        data: "pl.DataFrame",
        categorical_factors: List[str],
        continuous_factors: List[str],
        result=None,
        n_bins: int = DEFAULT_N_FACTOR_BINS,
        rare_threshold_pct: float = DEFAULT_RARE_THRESHOLD_PCT,
        max_categorical_levels: int = DEFAULT_MAX_CATEGORICAL_LEVELS,
        design_matrix: Optional[np.ndarray] = None,
        bread_matrix: Optional[np.ndarray] = None,
        irls_weights: Optional[np.ndarray] = None,
    ) -> List[FactorDiagnostics]:
        """Compute diagnostics for each specified factor.
        
        For unfitted factors, computes Rao's score test if design_matrix, 
        bread_matrix, and irls_weights are provided.
        """
        factors = []
        
        can_compute_score_test = (
            design_matrix is not None and 
            bread_matrix is not None and 
            irls_weights is not None
        )
        
        # Process categorical factors
        for name in categorical_factors:
            validate_factor_in_data(name, data, "Categorical factor")
            
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
            
            ae_bins = self._compute_ae_categorical(
                values, rare_threshold_pct, max_categorical_levels
            )
            resid_pattern = self._compute_residual_pattern_categorical(values)
            significance = self.compute_factor_significance(name, result) if in_model and result else None
            coefficients = self._get_factor_coefficients(name, result) if in_model and result else None
            
            score_test = None
            if not in_model and can_compute_score_test:
                score_test = self._compute_score_test_categorical(
                    values, design_matrix, bread_matrix, irls_weights
                )
            
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
                score_test=score_test,
            ))
        
        # Process continuous factors
        for name in continuous_factors:
            validate_factor_in_data(name, data, "Continuous factor")
            
            values = data[name].to_numpy().astype(np.float64)
            in_model = any(name in fn for fn in self.feature_names)
            
            # Univariate stats - batch percentile calculation
            valid_mask = ~np.isnan(values) & ~np.isinf(values)
            valid = values[valid_mask]
            
            if len(valid) > 0:
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
            
            ae_bins = self._compute_ae_continuous(values, n_bins)
            resid_pattern = self._compute_residual_pattern_continuous(values, n_bins)
            significance = self.compute_factor_significance(name, result) if in_model and result else None
            coefficients = self._get_factor_coefficients(name, result) if in_model and result else None
            
            score_test = None
            if not in_model and can_compute_score_test:
                score_test = self._compute_score_test_continuous(
                    values, design_matrix, bread_matrix, irls_weights
                )
            
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
                score_test=score_test,
            ))
        
        return factors
    
    def _get_transformation(self, name: str) -> Optional[str]:
        """Find transformation for a factor in the model.
        
        Prioritizes actual transforms (splines, TE, C) over interaction terms.
        """
        # Priority 1: Spline transforms - bs(name, ...), ns(name, ...), s(name, ...), ms(name, ...)
        spline_pattern = re.compile(rf'^(?:bs|ns|s|ms)\({re.escape(name)}[,)]')
        for fn in self.feature_names:
            if spline_pattern.match(fn):
                return fn
        
        # Priority 2: Target encoding - TE(name)
        te_pattern = f"TE({name})"
        for fn in self.feature_names:
            if fn == te_pattern or fn.startswith(f"TE({name})"):
                return fn
        
        # Priority 3: Categorical encoding - C(name)[...]
        cat_pattern = f"C({name})"
        for fn in self.feature_names:
            if fn.startswith(cat_pattern):
                return fn
        
        # Priority 4: Other transforms (I(...), log, sqrt, etc.) - but NOT interactions
        for fn in self.feature_names:
            if name in fn and fn != name and ':' not in fn:
                return fn
        
        # Priority 5: Interactions (only if nothing else found)
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
            params = result.params
            if callable(params):
                params = params()
            if hasattr(params, 'tolist'):
                params = params.tolist() if hasattr(params, 'tolist') else list(params)
            
            feature_names = result.feature_names if hasattr(result, 'feature_names') else self.feature_names
            
            bse = None
            if hasattr(result, 'bse'):
                bse = result.bse
                if callable(bse):
                    bse = bse()
            elif hasattr(result, 'std_errors'):
                bse = result.std_errors
                if callable(bse):
                    bse = bse()
            
            pvalues = None
            if hasattr(result, 'pvalues'):
                pvalues = result.pvalues
                if callable(pvalues):
                    pvalues = pvalues()
            
            link = result.link if hasattr(result, 'link') else None
            is_log_link = link in ('log', 'Log')
            
            coefficients = []
            for i, fn in enumerate(feature_names):
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
            raise FittingError(f"Failed to extract coefficient table: {e}") from e
    
    def compute_factor_significance(
        self,
        name: str,
        result,
    ) -> Optional[FactorSignificance]:
        """
        Compute significance tests for a factor in the model.
        
        Returns Wald chi-square test and deviance contribution.
        """
        if not hasattr(result, 'params') or not hasattr(result, 'bse'):
            return None
        
        param_indices = []
        for i, fn in enumerate(self.feature_names):
            if name in fn and fn != 'Intercept':
                param_indices.append(i)
        
        if not param_indices:
            return None
        
        try:
            params = np.asarray(result.params)
            bse = np.asarray(result.bse())
            
            wald_chi2 = 0.0
            for idx in param_indices:
                if bse[idx] > 0:
                    wald_chi2 += (params[idx] / bse[idx]) ** 2
            
            df = len(param_indices)
            wald_pvalue = 1 - _chi2_cdf(wald_chi2, float(df)) if df > 0 else 1.0
            deviance_contribution = float(wald_chi2)
            
            return FactorSignificance(
                chi2=round(float(wald_chi2), 2),
                p=round(float(wald_pvalue), 4),
                dev_contrib=round(deviance_contribution, 2),
            )
        except Exception as e:
            raise FittingError(f"Failed to compute factor significance for '{name}': {e}") from e
    
    def _compute_ae_continuous(self, values: np.ndarray, n_bins: int) -> List[ActualExpectedBin]:
        """Compute A/E for continuous factor using Rust backend (compact format)."""
        rust_bins = _rust_ae_continuous(values, self.y, self.mu, self.exposure, n_bins, self.family)
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
        unique_levels = np.unique(values)
        level_means = np.empty(len(unique_levels))
        level_counts = np.empty(len(unique_levels))
        for i, level in enumerate(unique_levels):
            mask = values == level
            level_counts[i] = np.sum(mask)
            level_means[i] = np.mean(self.pearson_residuals[mask])
        
        overall_mean = np.mean(self.pearson_residuals)
        ss_total = np.sum((self.pearson_residuals - overall_mean) ** 2)
        ss_between = np.sum(level_counts * (level_means - overall_mean) ** 2)
        
        eta_squared = ss_between / ss_total if ss_total > 0 else 0.0
        mean_abs_resid = np.mean(np.abs(level_means))
        
        return ResidualPattern(
            resid_corr=round(float(mean_abs_resid), 4),
            var_explained=round(float(eta_squared), 6),
        )
    
    def _compute_score_test_continuous(
        self,
        values: np.ndarray,
        design_matrix: np.ndarray,
        bread_matrix: np.ndarray,
        irls_weights: np.ndarray,
    ) -> Optional[ScoreTestResult]:
        """Compute Rao's score test for a continuous unfitted factor."""
        try:
            valid_mask = ~np.isnan(values) & ~np.isinf(values)
            z = values.copy()
            if not np.all(valid_mask):
                z[~valid_mask] = np.mean(values[valid_mask]) if np.any(valid_mask) else 0.0
            
            result = _rust_score_test_continuous(
                z, design_matrix, self.y, self.mu, irls_weights, bread_matrix, self.family
            )
            
            return ScoreTestResult(
                statistic=round(result["statistic"], 2),
                df=result["df"],
                pvalue=round(result["pvalue"], 4),
                significant=result["significant"],
            )
        except Exception as e:
            import warnings
            warnings.warn(
                f"Score test computation failed for continuous factor: {e}. "
                "This may indicate numerical issues with the design matrix or IRLS weights.",
                RuntimeWarning
            )
            return None
    
    def _compute_score_test_categorical(
        self,
        values: np.ndarray,
        design_matrix: np.ndarray,
        bread_matrix: np.ndarray,
        irls_weights: np.ndarray,
    ) -> Optional[ScoreTestResult]:
        """Compute Rao's score test for a categorical unfitted factor.
        
        Uses target encoding (CatBoost-style): computes the mean target value
        for each level and tests this as a single continuous variable (df=1).
        """
        try:
            unique_levels = np.unique(values)
            if len(unique_levels) < 2:
                return None
            
            if self.exposure is not None:
                rates = self.y / np.maximum(self.exposure, EPSILON)
            else:
                rates = self.y
            
            level_means = {}
            for level in unique_levels:
                mask = values == level
                if np.sum(mask) > 0:
                    level_means[level] = np.mean(rates[mask])
                else:
                    level_means[level] = np.mean(rates)
            
            z = np.array([level_means[v] for v in values], dtype=np.float64)
            
            valid_mask = np.isfinite(z)
            if not np.all(valid_mask):
                z = z.copy()
                z[~valid_mask] = np.mean(z[valid_mask]) if np.any(valid_mask) else 0.0
            
            result = _rust_score_test_continuous(
                z, design_matrix, self.y, self.mu, irls_weights, bread_matrix, self.family
            )
            
            return ScoreTestResult(
                statistic=round(result["statistic"], 2),
                df=result["df"],
                pvalue=round(result["pvalue"], 4),
                significant=result["significant"],
            )
        except Exception as e:
            import warnings
            warnings.warn(
                f"Score test computation failed for categorical factor: {e}. "
                "This may indicate numerical issues with the target encoding or design matrix.",
                RuntimeWarning
            )
            return None
    
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
        
        y_pred = y_mean + slope * (x - x_mean)
        ss_res = np.sum((y - y_pred) ** 2)
        
        df = n - 2
        mse = ss_res / df if df > 0 else 0
        se_slope = np.sqrt(mse / ss_xx) if mse > 0 and ss_xx > 0 else float('nan')
        
        if np.isnan(se_slope) or se_slope == 0:
            return slope, float('nan')
        
        t_stat = slope / se_slope
        pvalue = 2 * (1 - _t_cdf(abs(t_stat), float(df)))
        
        return slope, pvalue
