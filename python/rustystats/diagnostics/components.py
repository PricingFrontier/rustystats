"""
Focused diagnostic component classes.

Each component handles a specific type of diagnostic computation.
DiagnosticsComputer coordinates these components to produce unified output.
"""

from __future__ import annotations

from typing import List, Dict, Any

import numpy as np

from rustystats._rustystats import (
    compute_calibration_curve_py as _rust_calibration_curve,
    compute_discrimination_stats_py as _rust_discrimination_stats,
    compute_pearson_residuals_py as _rust_pearson_residuals,
    compute_deviance_residuals_py as _rust_deviance_residuals,
    compute_null_deviance_py as _rust_null_deviance,
    compute_unit_deviance_py as _rust_unit_deviance,
    hosmer_lemeshow_test_py as _rust_hosmer_lemeshow,
)

from rustystats.diagnostics.types import CalibrationBin
from rustystats.constants import DEFAULT_N_CALIBRATION_BINS


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
    
    def compute(self, n_bins: int = DEFAULT_N_CALIBRATION_BINS) -> Dict[str, Any]:
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
