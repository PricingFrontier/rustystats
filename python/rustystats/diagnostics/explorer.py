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

import os
from typing import TYPE_CHECKING, Any

import numpy as np

from rustystats._rustystats import (
    compute_correlation_and_vif_py as _compute_correlation_and_vif,
)
from rustystats._rustystats import (
    compute_cramers_v_matrix_from_codes_py as _compute_cramers_v_matrix_from_codes,
)
from rustystats._rustystats import (
    detect_exploratory_interactions_py as _detect_exploratory_interactions,
)
from rustystats._rustystats import (
    f_cdf_py as _f_cdf,
)
from rustystats.constants import (
    DEFAULT_MAX_CATEGORICAL_LEVELS,
    DEFAULT_MAX_INTERACTION_FACTORS,
    DEFAULT_N_FACTOR_BINS,
    DEFAULT_RARE_THRESHOLD_PCT,
    EPSILON,
)
from rustystats.diagnostics.types import (
    DataExploration,
    InteractionCandidate,
)
from rustystats.diagnostics.utils import discretize, validate_factor_in_data
from rustystats.exceptions import FittingError, ValidationError

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
        exposure: np.ndarray | None = None,
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
        self.exposure = (
            np.asarray(exposure, dtype=np.float64) if exposure is not None else np.ones_like(self.y)
        )
        self.family = family.lower()
        self.n_obs = len(y)

    def compute_response_stats(self) -> dict[str, Any]:
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
        data: pl.DataFrame,
        categorical_factors: list[str],
        continuous_factors: list[str],
        n_bins: int = DEFAULT_N_FACTOR_BINS,
        rare_threshold_pct: float = DEFAULT_RARE_THRESHOLD_PCT,
        max_categorical_levels: int = DEFAULT_MAX_CATEGORICAL_LEVELS,
        cat_column_cache: dict[str, np.ndarray | None] | None = None,
        cat_unique_cache: dict[str, tuple | None] | None = None,
        cont_column_cache: dict[str, np.ndarray | None] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Compute univariate statistics for each factor.

        Returns statistics and actual/expected rates by level/bin.
        """
        factors = []

        # Continuous factors
        for name in continuous_factors:
            validate_factor_in_data(name, data, "Continuous factor")

            values = (
                cont_column_cache[name]
                if cont_column_cache and name in cont_column_cache
                else data[name].to_numpy().astype(np.float64)
            )
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

            valid_bin_ids = np.searchsorted(quantiles, valid_values, side="right") - 1
            valid_bin_ids = np.clip(valid_bin_ids, 0, n_bins - 1)
            valid_exposure = self.exposure[valid_mask]
            valid_y = self.y[valid_mask]

            bin_counts = np.bincount(valid_bin_ids, minlength=n_bins)
            bin_exposures = np.bincount(valid_bin_ids, weights=valid_exposure, minlength=n_bins)
            bin_responses = np.bincount(valid_bin_ids, weights=valid_y, minlength=n_bins)

            for i in range(n_bins):
                count = int(bin_counts[i])
                if count == 0:
                    continue

                bin_exposure = float(bin_exposures[i])
                response_sum = float(bin_responses[i])
                rate = float(response_sum / bin_exposure) if bin_exposure > 0 else 0

                bins_data.append(
                    {
                        "bin_index": i,
                        "bin_lower": float(quantiles[i]),
                        "bin_upper": float(quantiles[i + 1]),
                        "count": count,
                        "exposure": bin_exposure,
                        "response_sum": response_sum,
                        "response_rate": rate,
                    }
                )
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

            if cat_unique_cache and name in cat_unique_cache:
                unique_levels, inverse = cat_unique_cache[name]
            else:
                values = (
                    cat_column_cache[name]
                    if cat_column_cache and name in cat_column_cache
                    else data[name].to_numpy().astype(str)
                )
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
                    levels_data.append(
                        {
                            "level": str(unique_levels[idx]),
                            "count": int(counts[idx]),
                            "exposure": exp,
                            "exposure_pct": float(pct),
                            "response_sum": float(y_by_level[idx]),
                            "response_rate": float(y_by_level[idx] / exp) if exp > 0 else 0,
                        }
                    )

            # Add "Other" if needed
            if other_count > 0:
                levels_data.append(
                    {
                        "level": "_Other",
                        "count": other_count,
                        "exposure": other_exp,
                        "exposure_pct": float(100 * other_exp / total_exposure),
                        "response_sum": other_y,
                        "response_rate": float(other_y / other_exp) if other_exp > 0 else 0,
                    }
                )

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
                    "thin_level_warning": f"Levels {thin_levels} have <1% exposure"
                    if thin_levels
                    else None,
                },
            }
            factors.append(stats)

        return factors

    def _compute_shape_hint(self, bin_rates: list[float]) -> dict[str, str]:
        """Analyze binned response rates to suggest transformation."""
        n = len(bin_rates)
        if n < 3:
            return {"shape": "insufficient_data", "recommendation": "linear"}

        # Check monotonicity
        diffs = [bin_rates[i + 1] - bin_rates[i] for i in range(n - 1)]
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
        variance = sum((r - avg_rate) ** 2 for r in bin_rates) / n
        if variance > avg_rate * 0.1:
            return {"shape": "non_linear", "recommendation": "spline"}

        return {"shape": "flat", "recommendation": "may not need in model"}

    def _detect_ordinal_pattern(self, levels: np.ndarray) -> dict[str, Any]:
        """Detect if categorical levels follow an ordinal pattern."""
        levels_str = [str(l) for l in levels]

        # Check for numeric levels
        try:
            [float(l) for l in levels_str]
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

        if all(re.match(r"^[A-Za-z]+\d+$", l) for l in levels_str):
            return {"is_ordinal": True, "pattern": "prefix_numeric"}

        return {"is_ordinal": False, "pattern": None}

    def compute_univariate_tests(
        self,
        data: pl.DataFrame,
        categorical_factors: list[str],
        continuous_factors: list[str],
        cat_column_cache: dict[str, np.ndarray | None] | None = None,
        cat_unique_cache: dict[str, tuple | None] | None = None,
        cont_column_cache: dict[str, np.ndarray | None] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Compute univariate significance tests for each factor vs response.

        For continuous factors: Pearson correlation + F-test from simple regression
        For categorical factors: ANOVA F-test (eta-squared based)
        """
        results = []
        y_rate = self.y / self.exposure

        for name in continuous_factors:
            validate_factor_in_data(name, data, "Continuous factor")

            values = (
                cont_column_cache[name]
                if cont_column_cache and name in cont_column_cache
                else data[name].to_numpy().astype(np.float64)
            )
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
            r2 = corr**2
            f_stat = (r2 / 1) / ((1 - r2) / (n - 2)) if r2 < 1 and n > 2 else 0

            # P-value from F-distribution (using Rust CDF)
            pvalue = 1 - _f_cdf(f_stat, 1.0, float(n - 2)) if n > 2 else 1.0

            results.append(
                {
                    "factor": name,
                    "type": "continuous",
                    "test": "correlation_f_test",
                    "correlation": float(corr),
                    "r_squared": float(r2),
                    "f_statistic": float(f_stat),
                    "pvalue": float(pvalue),
                    "significant_01": pvalue < 0.01 if not np.isnan(pvalue) else False,
                    "significant_05": pvalue < 0.05 if not np.isnan(pvalue) else False,
                }
            )

        for name in categorical_factors:
            validate_factor_in_data(name, data, "Categorical factor")

            if cat_unique_cache and name in cat_unique_cache:
                unique_levels, inverse = cat_unique_cache[name]
                eta_sq = self._compute_eta_squared_response_codes(inverse, len(unique_levels))
            else:
                values = (
                    cat_column_cache[name]
                    if cat_column_cache and name in cat_column_cache
                    else data[name].to_numpy().astype(str)
                )
                unique_levels, inverse = np.unique(values, return_inverse=True)
                eta_sq = self._compute_eta_squared_response_codes(inverse, len(unique_levels))
            k = len(unique_levels)
            n = len(inverse)

            if k > 1 and n > k:
                f_stat = (eta_sq / (k - 1)) / ((1 - eta_sq) / (n - k)) if eta_sq < 1 else 0

                # P-value from F-distribution (using Rust CDF)
                pvalue = 1 - _f_cdf(f_stat, float(k - 1), float(n - k))
            else:
                f_stat = 0.0
                pvalue = 1.0

            results.append(
                {
                    "factor": name,
                    "type": "categorical",
                    "test": "anova_f_test",
                    "n_levels": k,
                    "eta_squared": float(eta_sq),
                    "f_statistic": float(f_stat),
                    "pvalue": float(pvalue),
                    "significant_01": pvalue < 0.01 if not np.isnan(pvalue) else False,
                    "significant_05": pvalue < 0.05 if not np.isnan(pvalue) else False,
                }
            )

        # Sort by p-value (most significant first)
        results.sort(key=lambda x: x["pvalue"] if not np.isnan(x["pvalue"]) else 1.0)
        return results

    def compute_correlations(
        self,
        data: pl.DataFrame,
        continuous_factors: list[str],
        cont_column_cache: dict[str, np.ndarray | None] | None = None,
    ) -> dict[str, Any]:
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
            arr = (
                cont_column_cache[name]
                if cont_column_cache and name in cont_column_cache
                else data[name].to_numpy().astype(np.float64)
            )
            arrays.append(arr)

        X = np.column_stack(arrays)

        n_factors = len(valid_factors)
        if np.all(np.isfinite(X)):
            corr_matrix, _ = _compute_correlation_and_vif(
                np.ascontiguousarray(X, dtype=np.float64), EPSILON, 0
            )
            corr_matrix = np.asarray(corr_matrix, dtype=np.float64)
            np.fill_diagonal(corr_matrix, 1.0)
        else:
            # Preserve pairwise-complete semantics when missing or infinite
            # values are present; the Rust fast path intentionally handles the
            # common dense/finite case.
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
                        corr_matrix[i, j] = float("nan")
                        corr_matrix[j, i] = float("nan")

        # Find high correlations (|r| > 0.7)
        high_corrs = []
        for i in range(n_factors):
            for j in range(i + 1, n_factors):
                r = corr_matrix[i, j]
                if not np.isnan(r) and abs(r) > 0.7:
                    high_corrs.append(
                        {
                            "factor1": valid_factors[i],
                            "factor2": valid_factors[j],
                            "correlation": float(r),
                            "severity": "high" if abs(r) > 0.9 else "moderate",
                        }
                    )

        high_corrs.sort(key=lambda x: -abs(x["correlation"]))

        return {
            "factors": valid_factors,
            "matrix": corr_matrix.tolist(),
            "high_correlations": high_corrs,
        }

    def compute_vif(
        self,
        data: pl.DataFrame,
        continuous_factors: list[str],
        cont_column_cache: dict[str, np.ndarray | None] | None = None,
    ) -> list[dict[str, Any]]:
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
            arr = (
                cont_column_cache[name]
                if cont_column_cache and name in cont_column_cache
                else data[name].to_numpy().astype(np.float64)
            )
            arrays.append(arr)

        X = np.column_stack(arrays)

        # Remove rows with any NaN/Inf
        valid_rows = np.all(~np.isnan(X) & ~np.isinf(X), axis=1)
        X = X[valid_rows]

        if len(X) < len(valid_factors) + 1:
            return [
                {"factor": f, "vif": float("nan"), "severity": "unknown"} for f in valid_factors
            ]

        zero_variance = np.std(X, axis=0) <= EPSILON

        try:
            _, vif_values = _compute_correlation_and_vif(
                np.ascontiguousarray(X, dtype=np.float64), EPSILON, 0
            )
            vif_values = np.asarray(vif_values, dtype=np.float64)
        except Exception as e:
            raise FittingError(f"Failed to compute VIF: {e}") from e

        results = []
        for i, name in enumerate(valid_factors):
            vif = 1.0 if zero_variance[i] else float(vif_values[i])

            if np.isnan(vif) or np.isinf(vif):
                severity = "unknown"
            elif vif > 10:
                severity = "severe"
            elif vif > 5:
                severity = "moderate"
            else:
                severity = "none"

            results.append(
                {
                    "factor": name,
                    "vif": float(vif) if not np.isinf(vif) else 999.0,
                    "severity": severity,
                }
            )

        results.sort(key=lambda x: -x["vif"] if not np.isnan(x["vif"]) else 0)
        return results

    def compute_missing_values(
        self,
        data: pl.DataFrame,
        categorical_factors: list[str],
        continuous_factors: list[str],
    ) -> dict[str, Any]:
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

            factor_missing.append(
                {
                    "factor": name,
                    "n_missing": int(n_missing),
                    "pct_missing": float(pct_missing),
                    "severity": "high"
                    if pct_missing > 10
                    else ("moderate" if pct_missing > 1 else "none"),
                }
            )

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
            "summary": "No missing values"
            if all(f["n_missing"] == 0 for f in factor_missing)
            else "Missing values present",
        }

    def compute_zero_inflation(self) -> dict[str, Any]:
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

    def compute_overdispersion(self) -> dict[str, Any]:
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
            var_rate / expected_var
        else:
            pass

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
        data: pl.DataFrame,
        categorical_factors: list[str],
        cat_column_cache: dict[str, np.ndarray | None] | None = None,
        cat_unique_cache: dict[str, tuple | None] | None = None,
    ) -> dict[str, Any]:
        """
        Compute Cramér's V matrix for categorical factor pairs.

        Cramér's V measures association between categorical variables (0 to 1).
        """
        valid_factors = [f for f in categorical_factors if f in data.columns]

        if len(valid_factors) < 2:
            return {"factors": valid_factors, "matrix": [], "high_associations": []}

        # Pre-fetch per-column unique/inverse codes; Rust owns the pairwise
        # contingency aggregation.
        _uniq_inv = {}
        for name in valid_factors:
            if cat_unique_cache and name in cat_unique_cache:
                _uniq_inv[name] = cat_unique_cache[name]
            else:
                values = (
                    cat_column_cache[name]
                    if cat_column_cache and name in cat_column_cache
                    else data[name].to_numpy().astype(str)
                )
                _uniq_inv[name] = np.unique(values, return_inverse=True)

        codes_list = [
            np.ascontiguousarray(_uniq_inv[name][1], dtype=np.uint32) for name in valid_factors
        ]
        n_levels = [len(_uniq_inv[name][0]) for name in valid_factors]

        try:
            v_matrix = np.asarray(
                _compute_cramers_v_matrix_from_codes(codes_list, n_levels),
                dtype=np.float64,
            )
        except ValueError as e:
            raise ValidationError(str(e)) from e

        n_factors = len(valid_factors)

        # Find high associations (V > 0.3)
        high_assoc = []
        for i in range(n_factors):
            for j in range(i + 1, n_factors):
                v = v_matrix[i, j]
                if not np.isnan(v) and v > 0.3:
                    high_assoc.append(
                        {
                            "factor1": valid_factors[i],
                            "factor2": valid_factors[j],
                            "cramers_v": float(v),
                            "severity": "high" if v > 0.5 else "moderate",
                        }
                    )

        high_assoc.sort(key=lambda x: -x["cramers_v"])

        return {
            "factors": valid_factors,
            "matrix": v_matrix.tolist(),
            "high_associations": high_assoc,
        }

    def _compute_cramers_v_pair_fast(self, x_uniq_inv: tuple, y_uniq_inv: tuple) -> float:
        """Compute Cramér's V from pre-computed ``np.unique`` tuples."""
        x_cats, x_inv = x_uniq_inv
        y_cats, y_inv = y_uniq_inv

        codes = [
            np.ascontiguousarray(x_inv, dtype=np.uint32),
            np.ascontiguousarray(y_inv, dtype=np.uint32),
        ]
        try:
            matrix = _compute_cramers_v_matrix_from_codes(
                codes,
                [len(x_cats), len(y_cats)],
            )
        except ValueError as e:
            raise ValidationError(str(e)) from e

        return float(matrix[0][1])

    def detect_interactions(
        self,
        data: pl.DataFrame,
        factor_names: list[str],
        max_factors: int = 10,
        min_effect_size: float = 0.01,
        max_candidates: int = 5,
        min_cell_count: int = 30,
        cat_column_cache: dict[str, np.ndarray | None] | None = None,
        cat_unique_cache: dict[str, tuple | None] | None = None,
        cont_column_cache: dict[str, np.ndarray | None] | None = None,
    ) -> list[InteractionCandidate]:
        """
        Detect potential interactions using response-based analysis.

        This identifies factors whose combined effect on the response
        differs from their individual effects, suggesting an interaction.
        """
        encoded_names: list[str] = []
        encoded_codes: list[np.ndarray] = []
        encoded_n_levels: list[int] = []

        for name in factor_names:
            validate_factor_in_data(name, data)

            if cat_unique_cache and name in cat_unique_cache:
                levels, inverse = cat_unique_cache[name]
                codes = np.ascontiguousarray(inverse, dtype=np.uint32)
                n_levels = len(levels)
            elif cat_column_cache and name in cat_column_cache:
                levels, inverse = np.unique(cat_column_cache[name], return_inverse=True)
                codes = np.ascontiguousarray(inverse, dtype=np.uint32)
                n_levels = len(levels)
            elif cont_column_cache and name in cont_column_cache:
                values = cont_column_cache[name]
                valid_mask = ~np.isnan(values) & ~np.isinf(values)
                if np.sum(valid_mask) < 10:
                    continue
                bins = self._discretize(values, 5)
                codes = np.ascontiguousarray(bins, dtype=np.uint32)
                n_levels = int(np.max(codes)) + 1 if len(codes) else 0
            else:
                values = data[name].to_numpy()
                if values.dtype.kind in ("O", "U", "S"):
                    levels, inverse = np.unique(values.astype(str), return_inverse=True)
                    codes = np.ascontiguousarray(inverse, dtype=np.uint32)
                    n_levels = len(levels)
                else:
                    values = values.astype(np.float64)
                    valid_mask = ~np.isnan(values) & ~np.isinf(values)
                    if np.sum(valid_mask) < 10:
                        continue
                    bins = self._discretize(values, 5)
                    codes = np.ascontiguousarray(bins, dtype=np.uint32)
                    n_levels = int(np.max(codes)) + 1 if len(codes) else 0

            if n_levels < 2:
                continue
            encoded_names.append(name)
            encoded_codes.append(codes)
            encoded_n_levels.append(n_levels)

        if len(encoded_names) < 2:
            return []

        raw_candidates = _detect_exploratory_interactions(
            np.ascontiguousarray(self.y, dtype=np.float64),
            np.ascontiguousarray(self.exposure, dtype=np.float64),
            encoded_names,
            encoded_codes,
            encoded_n_levels,
            max_factors,
            min_effect_size,
            max_candidates,
            min_cell_count,
        )
        return [InteractionCandidate(**candidate) for candidate in raw_candidates]

    def _compute_eta_squared_response(self, categories: np.ndarray) -> float:
        """Compute eta-squared for categorical association with response.

        Uses np.bincount for O(n) aggregation instead of per-level masking.
        """
        _unique_levels, inverse = np.unique(categories, return_inverse=True)
        return self._compute_eta_squared_response_codes(inverse, len(_unique_levels))

    def _compute_eta_squared_response_codes(self, inverse: np.ndarray, n_levels: int) -> float:
        """Compute eta-squared from pre-factorized level codes."""
        if n_levels == 0:
            return 0.0

        y_rate = self.y / self.exposure
        inverse = np.asarray(inverse, dtype=np.int64)
        overall_mean = np.average(y_rate, weights=self.exposure)

        ss_total = np.sum(self.exposure * (y_rate - overall_mean) ** 2)

        if ss_total == 0:
            return 0.0

        # Weighted mean per level: sum(exposure * rate) / sum(exposure) = sum(y) / sum(exposure)
        level_y = np.bincount(inverse, weights=self.y, minlength=n_levels)
        level_exp = np.bincount(inverse, weights=self.exposure, minlength=n_levels)
        level_means = np.divide(level_y, level_exp, out=np.zeros(n_levels), where=level_exp > 0)
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
    ) -> InteractionCandidate | None:
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

        cell_counts[valid_mask]
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
            pvalue = float("nan")

        return InteractionCandidate(
            factor1=name1,
            factor2=name2,
            interaction_strength=float(r_squared),
            pvalue=float(pvalue),
            n_cells=n_valid_cells,
        )


def explore_data(
    data: pl.DataFrame,
    response: str,
    categorical_factors: list[str | None] | None = None,
    continuous_factors: list[str | None] | None = None,
    exposure: str | None = None,
    family: str = "poisson",
    n_bins: int = DEFAULT_N_FACTOR_BINS,
    rare_threshold_pct: float = DEFAULT_RARE_THRESHOLD_PCT,
    max_categorical_levels: int = DEFAULT_MAX_CATEGORICAL_LEVELS,
    detect_interactions: bool = True,
    max_interaction_factors: int = DEFAULT_MAX_INTERACTION_FACTORS,
    interactions: list | None = None,
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
        Whether to detect potential interactions (top-K auto-discovery
        producing ``interaction_candidates``).
    max_interaction_factors : int, default=10
        Maximum factors for interaction detection.
    interactions : list, optional
        Explicit list of variable pairs for per-pair pre-fit exploration.
        Each entry: ``{"factor1": ..., "factor2": ...}``, ``(a, b)``, or
        ``[a, b]``. Populates ``DataExploration.interactions`` with a
        ``SurfaceGrid`` of observed cell rates plus a partial-R² style
        ``interaction_strength``. Independent of ``detect_interactions=``.

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
    # Use per-column pl.Enum factorization to avoid global Categorical string-cache leakage.
    import polars as pl

    _cat_cache = {}
    _cat_unique_cache = {}
    for name in categorical_factors:
        if name in data.columns:
            values = data[name].cast(pl.Utf8).fill_null("None")
            level_list = values.unique().sort().to_list()
            levels = np.array(level_list, dtype=object)
            codes = (
                values.cast(pl.Enum(level_list)).to_physical().to_numpy().astype(np.uint32)
                if level_list
                else np.array([], dtype=np.uint32)
            )
            _cat_unique_cache[name] = (
                levels,
                np.ascontiguousarray(codes, dtype=np.uint32),
            )
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

    # Interaction detection (top-K auto-discovery)
    interaction_candidates = []
    if detect_interactions and len(categorical_factors) + len(continuous_factors) >= 2:
        all_factors = categorical_factors + continuous_factors
        interaction_candidates = explorer.detect_interactions(
            data=data,
            factor_names=all_factors,
            max_factors=max_interaction_factors,
            min_effect_size=0.001,  # Lower threshold to catch more interactions
            cat_column_cache=_cat_cache,
            cat_unique_cache=_cat_unique_cache,
            cont_column_cache=_cont_cache,
        )

    # User-specified interaction exploration (per-pair surface grids + strength)
    user_interactions: list = []
    if interactions:
        from rustystats.diagnostics.pair_diagnostics import _PairDiagnosticsComputer

        pair_computer = _PairDiagnosticsComputer(
            y=y,
            mu=None,
            exposure=exp,
            family=family,
            feature_names=None,
            link=None,
        )
        user_interactions = pair_computer.compute_pair_exploration(
            pairs=list(interactions),
            data=data,
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
        interactions=user_interactions,
    )

    # Auto-save JSON to analysis folder
    os.makedirs("analysis", exist_ok=True)
    with open("analysis/exploration.json", "w") as f:
        f.write(result.to_json(indent=2))

    return result
