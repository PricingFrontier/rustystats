"""
Focused diagnostic component classes.

Each component handles a specific type of diagnostic computation.
DiagnosticsComputer coordinates these components to produce unified output.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rustystats._rustystats import (
    compute_deviance_residuals_py as _rust_deviance_residuals,
)
from rustystats._rustystats import (
    compute_discrimination_stats_py as _rust_discrimination_stats,
)
from rustystats._rustystats import (
    compute_null_deviance_py as _rust_null_deviance,
)
from rustystats._rustystats import (
    compute_pearson_residuals_py as _rust_pearson_residuals,
)
from rustystats._rustystats import (
    compute_unit_deviance_py as _rust_unit_deviance,
)
from rustystats._rustystats import (
    hosmer_lemeshow_test_py as _rust_hosmer_lemeshow,
)
from rustystats.constants import DEFAULT_N_CALIBRATION_BINS
from rustystats.diagnostics.types import CalibrationBin
from rustystats.exceptions import ValidationError


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

    def compute(
        self, n_bins: int = DEFAULT_N_CALIBRATION_BINS, ranking: str = "auto"
    ) -> dict[str, Any]:
        # Route through the single weighted A/E primitive (RS-ACT-009): even
        # the legacy no-weight call path now resolves the overall ratio through
        # the same `_overall_ae` helper that `rs.calibration_summary` uses,
        # so the two stay numerically identical.
        from rustystats.calibration import _overall_ae

        overall = _overall_ae(self.y, self.mu, weights=None)
        ae_ratio = overall["ae_ratio"]

        bins = self._compute_bins(n_bins, ranking=ranking)
        _hl_stat, hl_pvalue = self._hosmer_lemeshow(n_bins)

        # Compressed format: only include problem deciles (A/E outside [0.9, 1.1])
        problem_deciles = [
            {
                "decile": b.bin_index,
                "ae": round(b.actual_expected_ratio, 2),
                "n": b.count,
                "ae_ci": [
                    round(b.ae_confidence_interval_lower, 2),
                    round(b.ae_confidence_interval_upper, 2),
                ],
            }
            for b in bins
            if b.actual_expected_ratio < 0.9 or b.actual_expected_ratio > 1.1
        ]

        return {
            "ae_ratio": round(ae_ratio, 3),
            "hl_pvalue": round(hl_pvalue, 4) if not np.isnan(hl_pvalue) else None,
            "problem_deciles": problem_deciles,
        }

    def _compute_bins(self, n_bins: int, ranking: str = "auto") -> list[CalibrationBin]:
        has_exposure = self.exposure is not None and not np.allclose(self.exposure, 1.0)
        safe_exposure = np.where(self.exposure > 0.0, self.exposure, 1.0)
        if ranking == "auto":
            score = (
                np.where(self.exposure > 0.0, self.mu / safe_exposure, self.mu)
                if has_exposure
                else self.mu
            )
        elif ranking == "mean":
            score = self.mu
        elif ranking == "rate":
            if not has_exposure:
                raise ValidationError("ranking='rate' requires exposure to be supplied.")
            score = np.where(self.exposure > 0.0, self.mu / safe_exposure, self.mu)
        else:
            raise ValidationError(f"ranking must be 'auto', 'mean', or 'rate', got {ranking!r}.")

        order = np.argsort(score, kind="stable")
        y_sorted = self.y[order]
        mu_sorted = self.mu[order]
        exposure_sorted = self.exposure[order]
        score_sorted = score[order]
        n = len(y_sorted)
        bins: list[CalibrationBin] = []
        for b in range(n_bins):
            start = b * n // n_bins
            end = (b + 1) * n // n_bins
            if start >= end:
                continue
            y_b = y_sorted[start:end]
            mu_b = mu_sorted[start:end]
            exp_b = exposure_sorted[start:end]
            score_b = score_sorted[start:end]
            actual_sum = float(np.sum(y_b))
            predicted_sum = float(np.sum(mu_b))
            exposure_sum = float(np.sum(exp_b))
            ae = actual_sum / predicted_sum if predicted_sum > 0 else np.nan
            if predicted_sum > 0.0 and np.isfinite(ae):
                ae_se = np.sqrt(max(actual_sum, 0.0)) / predicted_sum
                ae_ci_lower = max(0.0, ae - 1.96 * ae_se)
                ae_ci_upper = ae + 1.96 * ae_se
            else:
                ae_ci_lower = float("nan")
                ae_ci_upper = float("nan")
            rust_bin = {
                "bin_index": b + 1,
                "predicted_lower": float(score_b[0]),
                "predicted_upper": float(score_b[-1]),
                "predicted_mean": float(np.mean(score_b)),
                "actual_mean": float(actual_sum / exposure_sum if exposure_sum > 0 else np.nan),
                "actual_expected_ratio": float(ae),
                "count": int(end - start),
                "exposure": exposure_sum,
                "actual_sum": actual_sum,
                "predicted_sum": predicted_sum,
                "ae_ci_lower": float(ae_ci_lower),
                "ae_ci_upper": float(ae_ci_upper),
            }
            bins.append(rust_bin)
        rust_bins = bins
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

    def _hosmer_lemeshow(self, n_bins: int) -> tuple:
        result = _rust_hosmer_lemeshow(self.y, self.mu, n_bins)
        return result["chi2_statistic"], result["pvalue"]


class _DiscriminationComputer:
    """Computes discrimination metrics."""

    def __init__(self, y: np.ndarray, mu: np.ndarray, exposure: np.ndarray):
        self.y = y
        self.mu = mu
        self.exposure = exposure

    def compute(self) -> dict[str, Any]:
        stats = _rust_discrimination_stats(self.y, self.mu, self.exposure)
        # Removed lorenz_curve - Gini coefficient provides sufficient discrimination info
        return {
            "gini": round(stats["gini"], 3),
            "auc": round(stats["auc"], 3),
            "ks": round(stats["ks_statistic"], 3),
            "lift_10pct": round(stats["lift_at_10pct"], 3),
            "lift_20pct": round(stats["lift_at_20pct"], 3),
        }
