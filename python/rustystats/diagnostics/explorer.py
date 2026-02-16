"""
Pre-fit data exploration.

DataExplorer provides analysis before model fitting, including:
- Factor statistics (univariate distributions)
- Interaction detection based on response variable
- Response distribution analysis
- Correlation and multicollinearity checks

Unlike DiagnosticsComputer, this does NOT require a fitted model.
"""

from __future__ import annotations

import json
import os
from typing import Any, TYPE_CHECKING

import numpy as np

from rustystats._rustystats import (
    f_cdf_py as _f_cdf,
    factorize_strings_py as _factorize_strings,
)

from rustystats.diagnostics.types import (
    _json_default,
    _to_dict_recursive,
    InteractionCandidate,
    DataExploration,
)

from rustystats.diagnostics.utils import discretize, validate_factor_in_data
from rustystats.exceptions import FittingError, ValidationError
from rustystats.constants import (
    EPSILON,
    DEFAULT_N_FACTOR_BINS,
    DEFAULT_RARE_THRESHOLD_PCT,
    DEFAULT_MAX_CATEGORICAL_LEVELS,
    DEFAULT_MAX_INTERACTION_FACTORS,
)

if TYPE_CHECKING:
    import polars as pl


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
        n_bins: int = DEFAULT_N_FACTOR_BINS,
        rare_threshold_pct: float = DEFAULT_RARE_THRESHOLD_PCT,
        max_categorical_levels: int = DEFAULT_MAX_CATEGORICAL_LEVELS,
        cat_column_cache: Optional[Dict[str, np.ndarray]] = None,
        cat_unique_cache: Optional[Dict[str, tuple]] = None,
        cont_column_cache: Optional[Dict[str, np.ndarray]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Compute univariate statistics for each factor.
        
        Returns statistics and actual/expected rates by level/bin.
        """
        factors = []
        
        # Continuous factors
        for name in continuous_factors:
            validate_factor_in_data(name, data, "Continuous factor")
            
            values = cont_column_cache[name] if cont_column_cache and name in cont_column_cache else data[name].to_numpy().astype(np.float64)
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
            validate_factor_in_data(name, data, "Categorical factor")
            
            values = cat_column_cache[name] if cat_column_cache and name in cat_column_cache else data[name].to_numpy().astype(str)
            if cat_unique_cache and name in cat_unique_cache:
                unique_levels, inverse = cat_unique_cache[name]
            else:
                unique_levels, inverse = np.unique(values, return_inverse=True)
            k = len(unique_levels)
            
            # Vectorized aggregation with np.bincount
            counts = np.bincount(inverse, minlength=k)
            exp_by_level = np.bincount(inverse, weights=self.exposure, minlength=k)
            y_by_level = np.bincount(inverse, weights=self.y, minlength=k)
            total_exposure = np.sum(self.exposure)
            
            # Sort levels by exposure (descending)
            sort_idx = np.argsort(-exp_by_level)
            
            # Build level stats
            levels_data = []
            other_y = 0.0
            other_exp = 0.0
            other_count = 0
            
            for rank, idx in enumerate(sort_idx):
                exp = float(exp_by_level[idx])
                pct = 100 * exp / total_exposure
                
                if pct < rare_threshold_pct or rank >= max_categorical_levels - 1:
                    other_y += float(y_by_level[idx])
                    other_exp += exp
                    other_count += int(counts[idx])
                else:
                    levels_data.append({
                        "level": str(unique_levels[idx]),
                        "count": int(counts[idx]),
                        "exposure": exp,
                        "exposure_pct": float(pct),
                        "response_sum": float(y_by_level[idx]),
                        "response_rate": float(y_by_level[idx] / exp) if exp > 0 else 0,
                    })
            
            # Add "Other" if needed
            if other_count > 0:
                levels_data.append({
                    "level": "_Other",
                    "count": other_count,
                    "exposure": other_exp,
                    "exposure_pct": float(100 * other_exp / total_exposure),
                    "response_sum": other_y,
                    "response_rate": float(other_y / other_exp) if other_exp > 0 else 0,
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
        cat_column_cache: Optional[Dict[str, np.ndarray]] = None,
        cat_unique_cache: Optional[Dict[str, tuple]] = None,
        cont_column_cache: Optional[Dict[str, np.ndarray]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Compute univariate significance tests for each factor vs response.
        
        For continuous factors: Pearson correlation + F-test from simple regression
        For categorical factors: ANOVA F-test (eta-squared based)
        """
        results = []
        y_rate = self.y / self.exposure
        
        for name in continuous_factors:
            validate_factor_in_data(name, data, "Continuous factor")
            
            values = cont_column_cache[name] if cont_column_cache and name in cont_column_cache else data[name].to_numpy().astype(np.float64)
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
            validate_factor_in_data(name, data, "Categorical factor")
            
            values = cat_column_cache[name] if cat_column_cache and name in cat_column_cache else data[name].to_numpy().astype(str)
            
            # ANOVA: eta-squared and F-test
            eta_sq = self._compute_eta_squared_response(values)
            
            if cat_unique_cache and name in cat_unique_cache:
                unique_levels, _ = cat_unique_cache[name]
            else:
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
        cont_column_cache: Optional[Dict[str, np.ndarray]] = None,
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
            arr = cont_column_cache[name] if cont_column_cache and name in cont_column_cache else data[name].to_numpy().astype(np.float64)
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
        cont_column_cache: Optional[Dict[str, np.ndarray]] = None,
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
            arr = cont_column_cache[name] if cont_column_cache and name in cont_column_cache else data[name].to_numpy().astype(np.float64)
            arrays.append(arr)
        
        X = np.column_stack(arrays)
        
        # Remove rows with any NaN/Inf
        valid_rows = np.all(~np.isnan(X) & ~np.isinf(X), axis=1)
        X = X[valid_rows]
        
        if len(X) < len(valid_factors) + 1:
            return [{"factor": f, "vif": float('nan'), "severity": "unknown"} for f in valid_factors]
        
        # Standardize
        X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + EPSILON)
        
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
                raise FittingError(f"Failed to compute VIF for '{name}': {e}") from e
            
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
            validate_factor_in_data(name, data)
            
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
        cat_column_cache: Optional[Dict[str, np.ndarray]] = None,
        cat_unique_cache: Optional[Dict[str, tuple]] = None,
    ) -> Dict[str, Any]:
        """
        Compute Cramér's V matrix for categorical factor pairs.
        
        Cramér's V measures association between categorical variables (0 to 1).
        """
        valid_factors = [f for f in categorical_factors if f in data.columns]
        
        if len(valid_factors) < 2:
            return {"factors": valid_factors, "matrix": [], "high_associations": []}
        
        # Pre-fetch arrays and unique/inverse for all factors
        _arrays = {}
        _uniq_inv = {}
        for name in valid_factors:
            if cat_column_cache and name in cat_column_cache:
                _arrays[name] = cat_column_cache[name]
            else:
                _arrays[name] = data[name].to_numpy().astype(str)
            if cat_unique_cache and name in cat_unique_cache:
                _uniq_inv[name] = cat_unique_cache[name]
            else:
                _uniq_inv[name] = np.unique(_arrays[name], return_inverse=True)
        
        n_factors = len(valid_factors)
        v_matrix = np.eye(n_factors)
        
        for i in range(n_factors):
            for j in range(i + 1, n_factors):
                v = self._compute_cramers_v_pair_fast(
                    _uniq_inv[valid_factors[i]],
                    _uniq_inv[valid_factors[j]],
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
    
    def _compute_cramers_v_pair_fast(self, x_uniq_inv: tuple, y_uniq_inv: tuple) -> float:
        """Compute Cramér's V from pre-computed (unique, inverse) tuples.
        
        Uses vectorized contingency table via combined integer encoding.
        """
        x_cats, x_inv = x_uniq_inv
        y_cats, y_inv = y_uniq_inv
        
        r, k = len(x_cats), len(y_cats)
        if r < 2 or k < 2:
            return 0.0
        
        # Build contingency table vectorized: encode (x_inv, y_inv) as single int
        combined = x_inv * k + y_inv
        flat_counts = np.bincount(combined, minlength=r * k)
        contingency = flat_counts.reshape(r, k).astype(np.float64)
        
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
            raise ValidationError(
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

    def _compute_cramers_v_pair(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute Cramér's V for a pair of categorical variables (fallback).
        
        Uses vectorized contingency table via combined integer encoding.
        """
        x_str = x.astype(str)
        y_str = y.astype(str)
        
        x_cats, x_inv = np.unique(x_str, return_inverse=True)
        y_cats, y_inv = np.unique(y_str, return_inverse=True)
        
        r, k = len(x_cats), len(y_cats)
        if r < 2 or k < 2:
            return 0.0
        
        # Build contingency table vectorized: encode (x_inv, y_inv) as single int
        combined = x_inv * k + y_inv
        flat_counts = np.bincount(combined, minlength=r * k)
        contingency = flat_counts.reshape(r, k).astype(np.float64)
        
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
            raise ValidationError(
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
        cat_column_cache: Optional[Dict[str, np.ndarray]] = None,
        cont_column_cache: Optional[Dict[str, np.ndarray]] = None,
    ) -> List[InteractionCandidate]:
        """
        Detect potential interactions using response-based analysis.
        
        This identifies factors whose combined effect on the response
        differs from their individual effects, suggesting an interaction.
        """
        # First, rank factors by their effect on response variance
        factor_scores = []
        
        for name in factor_names:
            validate_factor_in_data(name, data)
            
            # Use cached arrays when available
            if cat_column_cache and name in cat_column_cache:
                score = self._compute_eta_squared_response(cat_column_cache[name])
            elif cont_column_cache and name in cont_column_cache:
                values = cont_column_cache[name]
                valid_mask = ~np.isnan(values) & ~np.isinf(values)
                if np.sum(valid_mask) < 10:
                    continue
                bins = self._discretize(values, 5)
                score = self._compute_eta_squared_response(bins.astype(str))
            else:
                values = data[name].to_numpy()
                if values.dtype == object or str(values.dtype).startswith('str'):
                    score = self._compute_eta_squared_response(values.astype(str))
                else:
                    values = values.astype(np.float64)
                    valid_mask = ~np.isnan(values) & ~np.isinf(values)
                    if np.sum(valid_mask) < 10:
                        continue
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
        """Compute eta-squared for categorical association with response.
        
        Uses np.bincount for O(n) aggregation instead of per-level masking.
        """
        y_rate = self.y / self.exposure
        _unique_levels, inverse = np.unique(categories, return_inverse=True)
        k = len(_unique_levels)
        overall_mean = np.average(y_rate, weights=self.exposure)
        
        ss_total = np.sum(self.exposure * (y_rate - overall_mean) ** 2)
        
        if ss_total == 0:
            return 0.0
        
        # Weighted mean per level: sum(exposure * rate) / sum(exposure) = sum(y) / sum(exposure)
        level_y = np.bincount(inverse, weights=self.y, minlength=k)
        level_exp = np.bincount(inverse, weights=self.exposure, minlength=k)
        level_means = np.divide(level_y, level_exp, out=np.zeros(k), where=level_exp > 0)
        ss_between = float(np.sum(level_exp * (level_means - overall_mean) ** 2))
        
        return ss_between / ss_total
    
    def _discretize(self, values: np.ndarray, n_bins: int) -> np.ndarray:
        """Discretize values into bins."""
        return discretize(values, n_bins)
    
    def _compute_interaction_strength_response(
        self,
        name1: str,
        bins1: np.ndarray,
        name2: str,
        bins2: np.ndarray,
        min_cell_count: int,
    ) -> Optional[InteractionCandidate]:
        """Compute interaction strength based on response variance.
        
        Uses np.bincount for O(n) aggregation instead of per-cell masking.
        """
        y_rate = self.y / self.exposure
        
        # Create interaction cells
        cell_ids = bins1 * 1000 + bins2
        unique_cells, inverse = np.unique(cell_ids, return_inverse=True)
        k = len(unique_cells)
        
        # Vectorized aggregation
        cell_counts = np.bincount(inverse, minlength=k)
        cell_y_sums = np.bincount(inverse, weights=self.y, minlength=k)
        cell_exp_sums = np.bincount(inverse, weights=self.exposure, minlength=k)
        
        # Filter cells with sufficient data
        valid_mask = cell_counts >= min_cell_count
        n_valid_cells = int(np.sum(valid_mask))
        if n_valid_cells < 4:
            return None
        
        valid_counts = cell_counts[valid_mask]
        valid_y = cell_y_sums[valid_mask]
        valid_exp = cell_exp_sums[valid_mask]
        valid_means = np.divide(valid_y, valid_exp, out=np.zeros_like(valid_y), where=valid_exp > 0)
        
        # Build combined index for valid observations (vectorized)
        obs_valid = valid_mask[inverse]
        all_rates = y_rate[obs_valid]
        all_weights = self.exposure[obs_valid]
        
        overall_mean = np.average(all_rates, weights=all_weights)
        ss_total = np.sum(all_weights * (all_rates - overall_mean) ** 2)
        
        if ss_total == 0:
            return None
        
        ss_model = float(np.sum(valid_exp * (valid_means - overall_mean) ** 2))
        r_squared = ss_model / ss_total
        
        # F-test p-value
        df_model = n_valid_cells - 1
        df_resid = len(all_rates) - n_valid_cells
        
        if df_model > 0 and df_resid > 0:
            f_stat = (ss_model / df_model) / ((ss_total - ss_model) / df_resid)
            pvalue = 1 - _f_cdf(f_stat, float(df_model), float(df_resid))
        else:
            pvalue = float('nan')
        
        return InteractionCandidate(
            factor1=name1,
            factor2=name2,
            interaction_strength=float(r_squared),
            pvalue=float(pvalue),
            n_cells=n_valid_cells,
        )


def explore_data(
    data: "pl.DataFrame",
    response: str,
    categorical_factors: Optional[List[str]] = None,
    continuous_factors: Optional[List[str]] = None,
    exposure: Optional[str] = None,
    family: str = "poisson",
    n_bins: int = DEFAULT_N_FACTOR_BINS,
    rare_threshold_pct: float = DEFAULT_RARE_THRESHOLD_PCT,
    max_categorical_levels: int = DEFAULT_MAX_CATEGORICAL_LEVELS,
    detect_interactions: bool = True,
    max_interaction_factors: int = DEFAULT_MAX_INTERACTION_FACTORS,
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
    
    # Pre-extract columns once to avoid repeated .to_numpy().astype() calls
    # Uses Rust HashMap-based factorize for O(n) encoding instead of O(n log n) np.unique
    _cat_cache = {}
    _cat_unique_cache = {}
    for name in categorical_factors:
        if name in data.columns:
            str_list = data[name].cast(str).to_list()
            levels, codes = _factorize_strings(str_list)
            str_vals = np.array(str_list)
            _cat_cache[name] = str_vals
            _cat_unique_cache[name] = (np.array(levels), codes)
    _cont_cache = {}
    for name in continuous_factors:
        if name in data.columns:
            _cont_cache[name] = data[name].to_numpy().astype(np.float64)
    
    # Compute statistics
    response_stats = explorer.compute_response_stats()
    
    factor_stats = explorer.compute_factor_stats(
        data=data,
        categorical_factors=categorical_factors,
        continuous_factors=continuous_factors,
        n_bins=n_bins,
        rare_threshold_pct=rare_threshold_pct,
        max_categorical_levels=max_categorical_levels,
        cat_column_cache=_cat_cache,
        cat_unique_cache=_cat_unique_cache,
        cont_column_cache=_cont_cache,
    )
    
    # Univariate significance tests
    univariate_tests = explorer.compute_univariate_tests(
        data=data,
        categorical_factors=categorical_factors,
        continuous_factors=continuous_factors,
        cat_column_cache=_cat_cache,
        cat_unique_cache=_cat_unique_cache,
        cont_column_cache=_cont_cache,
    )
    
    # Correlations between continuous factors
    correlations = explorer.compute_correlations(
        data=data,
        continuous_factors=continuous_factors,
        cont_column_cache=_cont_cache,
    )
    
    # VIF for multicollinearity
    vif = explorer.compute_vif(
        data=data,
        continuous_factors=continuous_factors,
        cont_column_cache=_cont_cache,
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
        cat_column_cache=_cat_cache,
        cat_unique_cache=_cat_unique_cache,
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
            cat_column_cache=_cat_cache,
            cont_column_cache=_cont_cache,
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
    os.makedirs("analysis", exist_ok=True)
    with open("analysis/exploration.json", "w") as f:
        f.write(result.to_json(indent=2))
    
    return result
