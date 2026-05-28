"""Calibration primitives for actuarial GLM workflows (RS-ACT-009 / PR11).

These are *primitives*. They never silently fold into a fitted ``GLMModel``:

* :func:`calibration_summary` returns a dict of overall, per-bin and optional
  per-factor calibration aggregates.
* :func:`fit_global_calibration` and :class:`GlobalCalibration` materialise a
  scalar multiplicative calibration factor ``c = Σ(w·y) / Σ(w·μ)``.
* :func:`fit_isotonic_calibration` and :class:`IsotonicCalibration` build a
  monotone (PAV) calibration map; opt-in only, serialised separately, and
  *never* folded into GLM coefficients.

For log-link GLMs a global calibration is also exposed as ``result.relevel()``:
it shifts the intercept by ``log(c)`` and leaves every other coefficient
bit-identical, so relativities (``exp(β_j)``) are preserved.

.. note::

   Fitting a calibration on the same rows used to fit the underlying model
   overstates calibration quality. Prefer a held-out calibration fold or
   out-of-fold predictions; report final calibration on untouched holdout
   data.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from rustystats.diagnostics.computer import rank_sort_idx
from rustystats.exceptions import ValidationError

__all__ = [
    "calibration_summary",
    "fit_global_calibration",
    "fit_isotonic_calibration",
    "GlobalCalibration",
    "IsotonicCalibration",
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _as_1d(x: Any, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValidationError(f"{name} must be 1-D, got shape {arr.shape}")
    return arr


def _resolve_weights(weights: Any, n: int) -> np.ndarray | None:
    if weights is None:
        return None
    w = _as_1d(weights, "weights")
    if w.shape[0] != n:
        raise ValidationError(f"weights length {w.shape[0]} does not match y length {n}")
    if not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValidationError("weights must be finite and non-negative")
    return w


def _resolve_exposure(exposure: Any, n: int) -> np.ndarray | None:
    if exposure is None:
        return None
    e = _as_1d(exposure, "exposure")
    if e.shape[0] != n:
        raise ValidationError(f"exposure length {e.shape[0]} does not match y length {n}")
    if not np.all(np.isfinite(e)) or np.any(e <= 0.0):
        raise ValidationError("exposure must be finite and positive")
    return e


def _overall_ae(
    y: np.ndarray, mu: np.ndarray, weights: np.ndarray | None
) -> dict[str, float | int]:
    """Single source of truth for overall A/E (weighted when supplied)."""
    if weights is None:
        actual = float(np.sum(y))
        expected = float(np.sum(mu))
        total_weight = float(y.shape[0])
    else:
        actual = float(np.sum(weights * y))
        expected = float(np.sum(weights * mu))
        total_weight = float(np.sum(weights))
    ae_ratio = actual / expected if expected != 0.0 else float("nan")
    return {
        "actual": actual,
        "expected": expected,
        "ae_ratio": ae_ratio,
        "n_obs": int(y.shape[0]),
        "total_weight": total_weight,
    }


def _validate_response_pred(y: Any, pred: Any) -> tuple[np.ndarray, np.ndarray]:
    y_arr = _as_1d(y, "y")
    mu_arr = _as_1d(pred, "pred")
    if y_arr.shape != mu_arr.shape:
        raise ValidationError(f"y length {y_arr.shape[0]} and pred length {mu_arr.shape[0]} differ")
    if not np.all(np.isfinite(y_arr)):
        raise ValidationError("y must contain only finite values")
    if not np.all(np.isfinite(mu_arr)):
        raise ValidationError("pred must contain only finite values")
    return y_arr, mu_arr


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------


def _compute_bins(
    y: np.ndarray,
    mu: np.ndarray,
    exposure: np.ndarray | None,
    weights: np.ndarray | None,
    n_bins: int,
    ranking: str,
) -> list[dict[str, Any]]:
    """Quantile-bin observations on the risk score and aggregate per bin.

    Bin size is set by *exposure when present* (so each bin holds an equal
    slice of portfolio exposure) and otherwise by row count. Aggregates are
    on the count scale (``Σy``, ``Σμ``, ``Σexposure``); weighted variants use
    ``Σw·y / Σw·μ / Σw·exposure`` to stay consistent with the overall A/E.
    """
    if n_bins <= 0 or y.shape[0] == 0:
        return []
    sort_idx = rank_sort_idx(mu, exposure, ranking=ranking)

    if exposure is not None:
        size_arr = exposure
    elif weights is not None:
        size_arr = weights
    else:
        size_arr = np.ones_like(y)

    total = float(np.sum(size_arr))
    if total <= 0.0:
        return []
    target = total / n_bins

    bins: list[dict[str, Any]] = []
    start = 0
    cum = 0.0
    bin_idx = 0
    n = y.shape[0]
    for pos, idx in enumerate(sort_idx):
        cum += float(size_arr[idx])
        is_last = pos == n - 1
        remaining_bins = n_bins - bin_idx - 1
        remaining_obs = n - pos - 1
        if is_last or (
            cum >= target * 0.99 and remaining_bins > 0 and remaining_obs >= remaining_bins
        ):
            members = sort_idx[start : pos + 1]
            bins.append(_aggregate_bin(y, mu, exposure, weights, members, bin_idx))
            bin_idx += 1
            start = pos + 1
            cum = 0.0
    return bins


def _aggregate_bin(
    y: np.ndarray,
    mu: np.ndarray,
    exposure: np.ndarray | None,
    weights: np.ndarray | None,
    members: np.ndarray,
    bin_index: int,
) -> dict[str, Any]:
    if weights is None:
        actual = float(np.sum(y[members]))
        expected = float(np.sum(mu[members]))
    else:
        actual = float(np.sum(weights[members] * y[members]))
        expected = float(np.sum(weights[members] * mu[members]))
    if exposure is None:
        exposure_sum = float(members.size)
    elif weights is None:
        exposure_sum = float(np.sum(exposure[members]))
    else:
        exposure_sum = float(np.sum(weights[members] * exposure[members]))
    ae_ratio = actual / expected if expected != 0.0 else float("nan")
    # Score-space stats (rate when exposure present, else raw mu).
    if exposure is not None:
        rate = mu[members] / exposure[members]
    else:
        rate = mu[members]
    return {
        "bin_index": int(bin_index),
        "count": int(members.size),
        "actual": actual,
        "expected": expected,
        "ae_ratio": ae_ratio,
        "exposure": exposure_sum,
        "predicted_rate_min": float(np.min(rate)),
        "predicted_rate_max": float(np.max(rate)),
        "predicted_rate_mean": float(np.mean(rate)),
        "actual_rate": actual / exposure_sum if exposure_sum > 0.0 else float("nan"),
        "expected_rate": expected / exposure_sum if exposure_sum > 0.0 else float("nan"),
    }


# ---------------------------------------------------------------------------
# Per-factor aggregation
# ---------------------------------------------------------------------------


def _by_factor(
    y: np.ndarray,
    mu: np.ndarray,
    exposure: np.ndarray | None,
    weights: np.ndarray | None,
    by: Mapping[str, Sequence[Any]] | None,
    min_exposure: float,
) -> dict[str, list[dict[str, Any]]] | None:
    if by is None:
        return None
    out: dict[str, list[dict[str, Any]]] = {}
    n = y.shape[0]
    for factor_name, values in by.items():
        arr = np.asarray(values)
        if arr.shape[0] != n:
            raise ValidationError(f"by[{factor_name!r}] length {arr.shape[0]} != y length {n}")
        rows: list[dict[str, Any]] = []
        # Preserve sorted level order for determinism; np.unique sorts already.
        # Use object-safe unique via pandas-free path: numpy works on strings/ints.
        unique = np.unique(arr)
        for level in unique:
            mask = arr == level
            members = np.flatnonzero(mask)
            if members.size == 0:
                continue
            row = _aggregate_bin(y, mu, exposure, weights, members, bin_index=0)
            # Replace bin_index with the level identifier.
            row.pop("bin_index", None)
            row["level"] = level.item() if isinstance(level, np.generic) else level
            row["suppressed"] = bool(row["exposure"] < min_exposure)
            rows.append(row)
        out[factor_name] = rows
    return out


# ---------------------------------------------------------------------------
# Public API: calibration_summary
# ---------------------------------------------------------------------------


def calibration_summary(
    y: Any,
    pred: Any,
    *,
    exposure: Any = None,
    weights: Any = None,
    by: Mapping[str, Sequence[Any]] | None = None,
    n_bins: int = 10,
    ranking: str = "auto",
    min_exposure: float = 0.0,
) -> dict[str, Any]:
    """Compute overall, per-bin and optional per-factor calibration aggregates.

    Parameters
    ----------
    y, pred : array-like, shape (n,)
        Response and predictions on the response scale (``μ``).
    exposure : array-like, optional
        Raw positive exposure. When supplied, bins are rate-ranked
        (``μ/exposure``) under ``ranking="auto"`` and bin aggregates include
        actual/expected/exposure rates.
    weights : array-like, optional
        Prior weights. When supplied, overall and bin aggregates use
        ``Σw·y / Σw·μ``; calibration objects fitted from this primitive use the
        same convention.
    by : mapping {factor_name: array-like}, optional
        For each named factor, aggregates per level (``actual``, ``expected``,
        ``exposure``, ``ae_ratio``, ``suppressed``).
    n_bins : int, default 10
        Number of equal-(exposure|weight|row) bins.
    ranking : {"auto", "mean", "rate"}, default "auto"
        See :func:`rustystats.diagnostics.computer.rank_sort_idx`.
    min_exposure : float, default 0.0
        Per-factor levels whose aggregated exposure is below this threshold
        are marked ``suppressed=True``.

    Returns
    -------
    dict with keys ``overall``, ``bins``, ``by_factor`` (or ``None`` when
    ``by=None``), ``ranking``, ``has_exposure``.

    Notes
    -----
    Fitting calibration on the same rows used to fit a model overstates
    calibration quality. Prefer a held-out calibration fold.
    """
    y_arr, mu_arr = _validate_response_pred(y, pred)
    exposure_arr = _resolve_exposure(exposure, y_arr.shape[0])
    weights_arr = _resolve_weights(weights, y_arr.shape[0])
    if ranking == "rate" and exposure_arr is None:
        raise ValidationError("ranking='rate' requires exposure to be supplied.")

    overall = _overall_ae(y_arr, mu_arr, weights_arr)
    bins = _compute_bins(y_arr, mu_arr, exposure_arr, weights_arr, n_bins, ranking)
    factor_rows = _by_factor(y_arr, mu_arr, exposure_arr, weights_arr, by, min_exposure)

    return {
        "overall": overall,
        "bins": bins,
        "by_factor": factor_rows,
        "ranking": ranking,
        "has_exposure": exposure_arr is not None,
    }


# ---------------------------------------------------------------------------
# GlobalCalibration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GlobalCalibration:
    """Scalar multiplicative calibration map ``y_hat = factor * pred``.

    ``factor = Σ(w·y) / Σ(w·μ)`` when fitted via :func:`fit_global_calibration`.
    For log-link GLMs, the same factor is what :meth:`GLMModel.relevel` applies
    to the intercept; see that method for in-place rate-table updates.
    """

    factor: float
    n_obs: int = 0
    total_weight: float = 0.0
    method: str = "global"

    def predict(self, pred: Any) -> np.ndarray:
        """Apply the calibration map: ``factor * pred`` (element-wise)."""
        arr = np.asarray(pred, dtype=np.float64)
        return self.factor * arr

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "factor": float(self.factor),
            "n_obs": int(self.n_obs),
            "total_weight": float(self.total_weight),
        }

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> GlobalCalibration:
        method = str(state.get("method", "global"))
        if method != "global":
            raise ValidationError(
                f"GlobalCalibration.from_dict expected method='global', got {method!r}"
            )
        return cls(
            factor=float(state["factor"]),
            n_obs=int(state.get("n_obs", 0)),
            total_weight=float(state.get("total_weight", 0.0)),
            method=method,
        )


def fit_global_calibration(y: Any, pred: Any, *, weights: Any = None) -> GlobalCalibration:
    """Fit a :class:`GlobalCalibration` with ``factor = Σ(w·y) / Σ(w·μ)``.

    Raises ``ValidationError`` when ``Σ(w·pred)`` is zero or non-finite. See
    the module docstring for the in-sample-optimism caveat.
    """
    y_arr, mu_arr = _validate_response_pred(y, pred)
    weights_arr = _resolve_weights(weights, y_arr.shape[0])
    overall = _overall_ae(y_arr, mu_arr, weights_arr)
    factor = overall["ae_ratio"]
    if not np.isfinite(factor):
        raise ValidationError(
            f"global calibration factor is non-finite (Σpred={overall['expected']!r})."
        )
    return GlobalCalibration(
        factor=float(factor),
        n_obs=int(overall["n_obs"]),
        total_weight=float(overall["total_weight"]),
    )


# ---------------------------------------------------------------------------
# IsotonicCalibration (Pool-Adjacent-Violators with weights)
# ---------------------------------------------------------------------------


def _pav_increasing(sorted_y: np.ndarray, sorted_w: np.ndarray) -> np.ndarray:
    """Pool-Adjacent-Violators with per-observation weights, increasing.

    Returns block means aligned with ``sorted_y``: positions inside a merged
    block all share the block's weighted mean.
    """
    n = sorted_y.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.float64)

    block_start = np.empty(n, dtype=np.int64)
    sum_wy = np.empty(n, dtype=np.float64)
    sum_w = np.empty(n, dtype=np.float64)
    mean = np.empty(n, dtype=np.float64)
    top = -1

    for i in range(n):
        top += 1
        block_start[top] = i
        wy = sorted_w[i] * sorted_y[i]
        sw = sorted_w[i]
        m = wy / sw if sw > 0.0 else sorted_y[i]
        sum_wy[top] = wy
        sum_w[top] = sw
        mean[top] = m
        # Merge while the previous block violates monotonicity.
        while top > 0 and mean[top - 1] > mean[top]:
            sum_wy[top - 1] += sum_wy[top]
            sum_w[top - 1] += sum_w[top]
            mean[top - 1] = (
                sum_wy[top - 1] / sum_w[top - 1] if sum_w[top - 1] > 0.0 else mean[top - 1]
            )
            top -= 1

    # Expand block means back to one value per row.
    fitted = np.empty(n, dtype=np.float64)
    for b in range(top + 1):
        start = block_start[b]
        end = block_start[b + 1] if b + 1 <= top else n
        fitted[start:end] = mean[b]
    return fitted


@dataclass(frozen=True)
class IsotonicCalibration:
    """Monotone calibration map fit by the Pool-Adjacent-Violators algorithm.

    Predictions for inputs between knots are linearly interpolated; inputs
    outside ``[thresholds_[0], thresholds_[-1]]`` are clamped to the nearest
    knot (matching :func:`numpy.interp` defaults).
    """

    thresholds_: np.ndarray  # sklearn-style trailing underscore
    values_: np.ndarray
    increasing: bool = True
    n_obs: int = 0
    total_weight: float = 0.0
    method: str = "isotonic"

    def predict(self, pred: Any) -> np.ndarray:
        arr = np.asarray(pred, dtype=np.float64)
        return np.interp(arr, self.thresholds_, self.values_)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "thresholds": [float(x) for x in self.thresholds_],
            "values": [float(x) for x in self.values_],
            "increasing": bool(self.increasing),
            "n_obs": int(self.n_obs),
            "total_weight": float(self.total_weight),
        }

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> IsotonicCalibration:
        method = str(state.get("method", "isotonic"))
        if method != "isotonic":
            raise ValidationError(
                f"IsotonicCalibration.from_dict expected method='isotonic', got {method!r}"
            )
        return cls(
            thresholds_=np.asarray(state["thresholds"], dtype=np.float64),
            values_=np.asarray(state["values"], dtype=np.float64),
            increasing=bool(state.get("increasing", True)),
            n_obs=int(state.get("n_obs", 0)),
            total_weight=float(state.get("total_weight", 0.0)),
            method=method,
        )


def fit_isotonic_calibration(
    y: Any,
    pred: Any,
    *,
    weights: Any = None,
    increasing: bool = True,
) -> IsotonicCalibration:
    """Fit a monotone calibration map ``pred -> y_hat`` using PAV.

    Decreasing fits are obtained by negating both the input ordering and the
    fitted values; the returned object always exposes ``thresholds_`` ascending.
    """
    y_arr, mu_arr = _validate_response_pred(y, pred)
    weights_arr = _resolve_weights(weights, y_arr.shape[0])
    if weights_arr is None:
        weights_arr = np.ones_like(y_arr)

    # For decreasing, flip the sign of pred so the increasing PAV builds the
    # right shape, then flip back. Equivalently we could sort descending and
    # PAV-increasing on -y; either way the returned thresholds_/values_ stay
    # ascending in pred-space, with values_ non-increasing.
    if increasing:
        signed_pred = mu_arr
    else:
        signed_pred = -mu_arr

    order = np.argsort(signed_pred, kind="stable")
    sorted_y = y_arr[order]
    sorted_w = weights_arr[order]
    fitted_sorted = _pav_increasing(sorted_y, sorted_w)

    # Recover original-space pred values; when ``increasing=False`` the signed
    # sort is ascending in ``-pred`` (i.e. descending in ``pred``), so reverse
    # to keep ``thresholds_`` ascending in ``pred``.
    sorted_pred = mu_arr[order]
    if not increasing:
        sorted_pred = sorted_pred[::-1]
        fitted_sorted = fitted_sorted[::-1]

    # Deduplicate equal pred values: keep the weighted average of fitted
    # values inside ties (within a PAV block, all `fitted_sorted` entries are
    # equal; across ties at boundary, prefer the rightmost so monotonicity
    # holds).
    unique_pred, first_idx = np.unique(sorted_pred, return_index=True)
    last_idx = np.r_[first_idx[1:] - 1, sorted_pred.shape[0] - 1]
    unique_values = fitted_sorted[last_idx]

    # Defensive monotonicity enforcement (handles ties at PAV block boundaries
    # that could leave the dedup'd values slightly out of order).
    if increasing:
        unique_values = np.maximum.accumulate(unique_values)
    else:
        unique_values = np.minimum.accumulate(unique_values)

    total_weight = float(np.sum(weights_arr))
    return IsotonicCalibration(
        thresholds_=unique_pred.astype(np.float64),
        values_=unique_values.astype(np.float64),
        increasing=bool(increasing),
        n_obs=int(y_arr.shape[0]),
        total_weight=total_weight,
    )
