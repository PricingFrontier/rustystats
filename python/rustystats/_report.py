"""Assemble the validation report behind ``GLMModel.validation_report`` and ``html_report``.

The report is one JSON-safe dictionary: what a reviewer needs to judge a fitted GLM, in and out
of sample. ``_html_report.report_html`` draws it and computes nothing itself, so a page can be
rebuilt from a report saved as JSON.

Most of the statistics come from ``GLMModel.diagnostics``. This module adds what a reviewer also
needs and diagnostics does not hold: one set of bands per factor, the same for every dataset,
with a standard deviation behind each A/E; fitted effects with confidence bands; the variance
check; influence; drift; and a list of review flags with the thresholds that raised them.
"""

from __future__ import annotations

import datetime as dt
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise
from statistics import NormalDist
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from rustystats._rustystats import (
    compute_deviance_residuals_py as _rust_deviance_residuals,
)
from rustystats._rustystats import (
    compute_null_deviance_py as _rust_null_deviance,
)
from rustystats.diagnostics.api import (
    _extract_response_and_predictions,
    _extract_test_arrays,
    _iter_design_matrix_chunks,
    _parse_family_params,
    _resolve_offset_and_response,
    _resolve_test_weights,
    _resolve_weights,
)
from rustystats.diagnostics.computer import rank_sort_idx
from rustystats.diagnostics.pair_diagnostics import _OTHER_LABEL
from rustystats.exceptions import RustyStatsError, ValidationError

if TYPE_CHECKING:
    from rustystats.formula import GLMModel

__all__ = ["REPORT_VERSION", "DEFAULT_THRESHOLDS", "build_report"]

REPORT_VERSION = 1

DEFAULT_THRESHOLDS: dict[str, float] = {
    "ae_train": 0.02,
    "ae_test": 0.05,
    "factor_ae": 0.10,
    "factor_min_mass": 0.01,
    "factor_z": 3.0,
    "decile_ae": 0.05,
    "near_miss_z": 2.0,
    "psi_warn": 0.10,
    "psi_critical": 0.25,
    "train_gap": 0.05,
    "unseen_mass": 0.01,
    "dispersion": 1.2,
    "gvif": 10.0,
    "term_p": 0.05,
    "interaction_ae": 0.15,
}
THRESHOLD_DESCRIPTIONS: dict[str, str] = {
    "ae_train": "overall A/E on the train data within 1 ± value",
    "ae_test": "overall A/E on the test data within 1 ± value",
    "factor_ae": "factor bands whose A/E is off by more than value",
    "factor_min_mass": "only bands holding at least this share of the mass",
    "factor_z": (
        "only bands and deciles whose A - E is at least value standard deviations from 0 "
        "(None: the tolerance alone)"
    ),
    "decile_ae": "fitted-value deciles whose A/E is off by more than value (and |z| ≥ factor_z)",
    "near_miss_z": (
        "bands and deciles off by more than their tolerance with z ≥ value (below factor_z) "
        "are listed as near misses"
    ),
    "psi_warn": "PSI per factor above value: warning",
    "psi_critical": "PSI per factor above value: critical",
    "train_gap": "drop in deviance explained from train to test",
    "unseen_mass": "mass share of categorical values the train data did not have",
    "dispersion": "variance-to-mean dispersion above value on a family that fixes it at 1",
    "gvif": "variance inflation per coefficient, GVIF^(1/df), above value",
    "term_p": "fitted factors whose joint test has a p-value above value",
    "interaction_ae": "interaction cells whose A/E is off by more than value (and |z| ≥ factor_z)",
}

_CATEGORICAL_TERMS = frozenset(
    {"categorical", "target_encoding", "frequency_encoding", "categorical_indicator"}
)
_VALID_INFERENCE = frozenset({"valid_standard", "valid_robust"})
_MAX_LEVELS = 30
_MISSING_LEVEL = "None"  # as the explorer and design matrix
_DECILES = 10
_FIT_BANDS = 20
_HISTOGRAM_BINS = 40
_LORENZ_POINTS = 101
_QQ_POINTS = 199
_PSI_FLOOR = 1e-4
_MAX_COUNT = 5
_TOP_INFLUENCE = 10
_TOP_CORRELATIONS = 15
_MAX_CORRELATION_COLUMNS = 2000
_INFLUENCE_FLOPS = 2e11  # n·p² beyond which leverage is left out
_EFFECT_POINTS_PER_BAND = 4
_MAX_KEY = 250
_KEY_CHARS = re.compile(r"[^A-Za-z0-9_]")
_Z_95 = 1.959964
# What rebuilding a design from a grid or the train rows can raise; the report then says the
# chart is unavailable. Anything else is a bug and should surface.
_EFFECT_ERRORS = (RustyStatsError, pl.exceptions.PolarsError, ValueError)


# --------------------------------------------------------------------------- small helpers ---


def resolve_thresholds(thresholds: dict[str, Any] | None) -> dict[str, float | None]:
    """The defaults with ``thresholds`` applied; ``None`` turns a check off."""
    out: dict[str, float | None] = dict(DEFAULT_THRESHOLDS)
    if thresholds is None:
        return out
    if not isinstance(thresholds, dict):
        raise ValidationError("thresholds must be a dict of threshold name to value.")
    unknown = sorted(set(thresholds) - set(DEFAULT_THRESHOLDS))
    if unknown:
        raise ValidationError(
            f"unknown threshold(s) {unknown}; known: {sorted(DEFAULT_THRESHOLDS)}"
        )
    for key, value in thresholds.items():
        if value is None:
            out[key] = None
            continue
        v = float(value)
        if not math.isfinite(v) or v < 0:
            raise ValidationError(
                f"threshold {key!r} must be finite and non-negative, got {value!r}"
            )
        out[key] = v
    warn, crit = out["psi_warn"], out["psi_critical"]
    if warn is not None and crit is not None and warn > crit:
        raise ValidationError("psi_warn must not exceed psi_critical.")
    return out


def json_safe(value: Any) -> Any:
    """``value`` as plain JSON: numpy scalars to Python, non-finite floats to ``None``."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        x = float(value)
        return x if math.isfinite(x) else None
    if value is None or isinstance(value, str):
        return value
    return str(value)


def _ratio(num: float, den: float) -> float | None:
    return float(num / den) if den > 0 else None


def _attr(model: Any, name: str) -> Any:
    """An attribute of the model, called if callable; ``None`` when a loaded model lacks it."""
    try:
        value = getattr(model, name)
        return value() if callable(value) else value
    except (AttributeError, RustyStatsError):
        return None


def _family(model: Any) -> tuple[str, float, float]:
    """``(kind, variance power, theta)`` of the model's family."""
    low = str(model.family).lower()
    power, theta = _parse_family_params(str(model.family))
    for prefix, kind in (
        ("quasipoisson", "poisson"),
        ("poisson", "poisson"),
        ("quasibinomial", "binomial"),
        ("binomial", "binomial"),
        ("gamma", "gamma"),
        ("tweedie", "tweedie"),
        ("negbin", "negbin"),
        ("negativebinomial", "negbin"),
    ):
        if low.startswith(prefix):
            return kind, power, theta
    return "gaussian", power, theta


def _variance(kind: str, mu: np.ndarray, power: float, theta: float) -> np.ndarray:
    """The family's variance function V(μ), per row."""
    if kind == "poisson":
        v = np.maximum(mu, 0.0)
    elif kind == "gamma":
        v = mu * mu
    elif kind == "tweedie":
        v = np.power(np.maximum(mu, 0.0), power)
    elif kind == "binomial":
        p = np.clip(mu, 0.0, 1.0)
        v = p * (1.0 - p)
    elif kind == "negbin":
        v = mu + mu * mu / theta
    else:
        v = np.ones_like(mu)
    return np.asarray(v, dtype=np.float64)


def _offset(
    model: Any, data: pl.DataFrame, *, exposure: Any = None, train: bool = False
) -> np.ndarray | None:
    """What prediction adds to the linear predictor beyond exposure: offset and complement.

    An array complement is fit-time data that prediction does not reuse, so it applies to the
    train rows only.
    """
    from rustystats.formula import (
        _resolve_predict_complement,
        _resolve_predict_offset,
        apply_link,
    )

    needs_prepared = model._prediction_aux_needs_prepared_data(
        offset=None, exposure=exposure, complement=None
    )
    frame = model.prepare_input(data) if needs_prepared else data
    _, offset, _ = _resolve_predict_offset(frame, None, model._offset_spec)
    stored_exposure = exposure if exposure is not None else model._exposure_spec
    _, complement = _resolve_predict_complement(
        frame, None, model._complement_spec, stored_exposure, model.link
    )
    fitted = model._complement_values
    if complement is None and train and fitted is not None and fitted.shape == (data.height,):
        complement = apply_link(fitted, model.link)
    parts = [part for part in (offset, complement) if part is not None]
    return np.sum(parts, axis=0) if parts else None


# --------------------------------------------------------------------------- datasets --------


@dataclass
class _Scored:
    """One dataset with its per-row arrays."""

    label: str
    data: pl.DataFrame
    y: np.ndarray
    mu: np.ndarray
    exposure: np.ndarray
    weight: np.ndarray
    offset: np.ndarray | None
    has_exposure: bool

    @property
    def n(self) -> int:
        return int(self.y.shape[0])

    @property
    def mass(self) -> np.ndarray:
        return self.weight * self.exposure

    @property
    def rate(self) -> np.ndarray:
        safe = np.where(self.exposure > 0, self.exposure, 1.0)
        return self.mu / safe


def _unit_deviance(y: np.ndarray, mu: np.ndarray, family: str) -> np.ndarray:
    r = np.asarray(_rust_deviance_residuals(y, mu, family), dtype=np.float64)
    return r * r


def _null_deviance(d: _Scored, family: str, link: str) -> float:
    """The deviance of the best constant on ``d``'s rows, with their exposure and offset."""
    offset = d.offset
    if d.has_exposure:
        with np.errstate(divide="ignore"):
            log_exposure = np.log(d.exposure)
        offset = log_exposure if offset is None else offset + log_exposure
    return float(_rust_null_deviance(d.y, family, weights=d.weight, offset=offset, link=link))


def _concentration(
    actual: np.ndarray, score: np.ndarray, mass: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """The Lorenz curve of ``actual`` by ascending ``score``; tied scores form one segment."""
    keep = mass > 0
    if not keep.any():
        return None
    uniques, inverse = np.unique(score[keep], return_inverse=True)
    a = np.bincount(inverse, weights=actual[keep], minlength=uniques.size)
    m = np.bincount(inverse, weights=mass[keep], minlength=uniques.size)
    if a.sum() <= 0 or m.sum() <= 0:
        return None
    x = np.concatenate([[0.0], np.cumsum(m) / m.sum()])
    y = np.concatenate([[0.0], np.cumsum(a) / a.sum()])
    return x, y


def _gini(actual: np.ndarray, score: np.ndarray, mass: np.ndarray) -> float | None:
    """Concentration Gini of ``actual`` ranked by ``score``, ``mass`` on the x-axis."""
    curve = _concentration(actual, score, mass)
    if curve is None:
        return None
    x, y = curve
    return float(1.0 - np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1])))


def _deciles(d: _Scored, variance: np.ndarray, phi: float) -> list[dict[str, Any]]:
    """Deciles of predicted rate, the same partition diagnostics uses (equal rows)."""
    order = rank_sort_idx(d.mu, d.exposure, has_exposure=d.has_exposure, ranking="auto")
    size = d.n // _DECILES
    total_mass = float(d.mass.sum())
    out = []
    for k in range(_DECILES):
        idx = order[k * size : (k + 1) * size if k < _DECILES - 1 else d.n]
        out.append(_aggregate(d, idx, variance, phi, total_mass) | {"band": k + 1})
    return out


def _aggregate(
    d: _Scored, idx: np.ndarray, variance: np.ndarray, phi: float, total_mass: float
) -> dict[str, Any]:
    w = d.weight[idx]
    mass = float(np.sum(w * d.exposure[idx]))
    actual = float(np.sum(w * d.y[idx]))
    expected = float(np.sum(w * d.mu[idx]))
    var = float(phi * np.sum(w * variance[idx]))
    return _band_numbers(int(idx.size), mass, actual, expected, var, total_mass)


def _band_numbers(
    rows: int, mass: float, actual: float, expected: float, var: float, total_mass: float
) -> dict[str, Any]:
    sd = math.sqrt(var) if var > 0 else None
    return {
        "rows": rows,
        "mass": mass,
        "mass_share": _ratio(mass, total_mass),
        "actual_total": actual,
        "expected_total": expected,
        "actual": _ratio(actual, mass),
        "predicted": _ratio(expected, mass),
        "ae": _ratio(actual, expected),
        "sd": sd / expected if sd is not None and expected > 0 else None,
        "z": (actual - expected) / sd if sd is not None else None,
    }


def _fit_bands(d: _Scored, variance: np.ndarray, phi: float) -> list[dict[str, Any]]:
    """Equal-row bands of fitted rate: A/E and the variance check in each."""
    order = rank_sort_idx(d.mu, d.exposure, has_exposure=d.has_exposure, ranking="auto")
    size = max(d.n // _FIT_BANDS, 1)
    total_mass = float(d.mass.sum())
    rate = d.rate
    bands = []
    for k in range(_FIT_BANDS):
        idx = order[k * size : (k + 1) * size if k < _FIT_BANDS - 1 else d.n]
        if idx.size == 0:
            continue
        b = _aggregate(d, idx, variance, phi, total_mass)
        w = d.weight[idx]
        spread = float(np.sum(w * (d.y[idx] - d.mu[idx]) ** 2))
        var = float(np.sum(w * variance[idx]))
        b |= {
            "band": k + 1,
            "fitted_mean": b["predicted"],
            "fitted_low": float(rate[idx].min()),
            "fitted_high": float(rate[idx].max()),
            "dispersion": _ratio(spread, var),
        }
        bands.append(b)
    return bands


def _histogram(values: np.ndarray, mass: np.ndarray, edges: np.ndarray) -> list[float]:
    clipped = np.clip(values, edges[0], edges[-1])
    counts, _ = np.histogram(clipped, bins=edges, weights=mass)
    return [float(c) for c in counts]


def _rate_edges(rate: np.ndarray) -> tuple[np.ndarray, bool]:
    live = rate[np.isfinite(rate)]
    if live.size == 0:
        return np.array([0.0, 1.0]), False
    lo, hi = np.quantile(live, [0.001, 0.999])
    if not hi > lo:
        hi = lo + (abs(lo) * 0.1 or 1.0)
    if lo > 0 and hi / lo > 3:
        return np.geomspace(lo, hi, _HISTOGRAM_BINS + 1), True
    return np.linspace(lo, hi, _HISTOGRAM_BINS + 1), False


def _count_distribution(d: _Scored, kind: str, theta: float) -> dict[str, Any] | None:
    """Observed and expected counts of each response value, for a count family."""
    if kind not in ("poisson", "negbin"):
        return None
    if not np.all(d.y >= 0) or not np.all(np.equal(np.mod(d.y, 1), 0)):
        return None
    top = min(_MAX_COUNT, int(d.y.max()) + 1)
    mu = np.maximum(d.mu, 1e-300)
    pmf = np.exp(-mu) if kind == "poisson" else (theta / (theta + mu)) ** theta
    observed, expected = [], []
    tail = np.ones_like(mu)
    for k in range(top):
        observed.append(float(np.sum(d.weight[d.y == k])))
        expected.append(float(np.sum(d.weight * pmf)))
        tail = tail - pmf
        pmf = (
            pmf * mu / (k + 1)
            if kind == "poisson"
            else pmf * (k + theta) / (k + 1) * mu / (theta + mu)
        )
    observed.append(float(np.sum(d.weight[d.y >= top])))
    expected.append(float(np.sum(d.weight * np.maximum(tail, 0.0))))
    return {"top": top, "observed": observed, "expected": expected}


def _total_z(actual: float, expected: float, variance: float) -> float | None:
    """How many standard deviations a dataset's total sits from its expected value."""
    return (actual - expected) / math.sqrt(variance) if variance > 0 else None


def _curves(
    d: _Scored, variance: np.ndarray, phi: float, rate_bins: tuple[np.ndarray, bool]
) -> dict[str, Any]:
    """Calibration, ranking and spread of one dataset's predictions."""
    mass = d.mass
    rate = d.rate
    curve = _concentration(d.weight * d.y, rate, mass)
    grid = np.linspace(0.0, 1.0, _LORENZ_POINTS)
    edges, is_log = rate_bins
    quantiles = (1, 5, 25, 50, 75, 95, 99)
    return {
        "deciles": _deciles(d, variance, phi),
        "fit_bands": _fit_bands(d, variance, phi),
        "lorenz": (
            None
            if curve is None
            else {"x": grid.tolist(), "y": np.interp(grid, curve[0], curve[1]).tolist()}
        ),
        "rate_histogram": {
            "edges": [float(e) for e in edges],
            "mass": _histogram(rate, mass, edges),
            "log": is_log,
        },
        "rate_percentiles": {
            str(q): float(v) for q, v in zip(quantiles, np.percentile(rate, quantiles))
        },
    }


def _dataset(
    d: _Scored,
    diag_row: Any,
    model: Any,
    kind: str,
    power: float,
    theta: float,
    phi: float,
    rate_bins: tuple[np.ndarray, bool],
) -> dict[str, Any]:
    family = str(model.family)
    unit = _unit_deviance(d.y, d.mu, family)
    deviance = float(np.sum(d.weight * unit))
    null = _null_deviance(d, family, model.link)
    variance = _variance(kind, d.mu, power, theta)
    mass = d.mass
    actual, expected = float(np.sum(d.weight * d.y)), float(np.sum(d.weight * d.mu))
    safe = np.where(d.exposure > 0, d.exposure, 1.0)
    perfect = _gini(d.weight * d.y, d.y / safe, mass)
    gini = _gini(d.weight * d.y, d.rate, mass)
    resid = np.sign(d.y - d.mu) * np.sqrt(unit)
    live = resid[np.isfinite(resid)]
    lo, hi = np.quantile(live, [0.005, 0.995]) if live.size else (-1.0, 1.0)
    if not hi > lo:
        hi = lo + 1.0
    edges = np.linspace(lo, hi, _HISTOGRAM_BINS + 1)
    qq = None
    if kind in ("gaussian", "gamma"):
        scale = _attr(model, "scale")
        scale = float(scale) if scale and scale > 0 else 1.0
        p = (np.arange(_QQ_POINTS) + 0.5) / _QQ_POINTS
        normal = NormalDist()
        qq = {
            "theoretical": [normal.inv_cdf(float(v)) for v in p],
            "sample": [float(v) for v in np.quantile(live / math.sqrt(scale), p)],
        }
    return {
        "label": d.label,
        "rows": d.n,
        "metrics": {
            "mass": float(mass.sum()),
            "actual": actual,
            "expected": expected,
            "ae": _ratio(actual, expected),
            "ae_z": _total_z(actual, expected, phi * float(np.sum(d.weight * variance))),
            "mean_actual": _ratio(actual, float(mass.sum())),
            "mean_predicted": _ratio(expected, float(mass.sum())),
            "deviance": deviance,
            "mean_deviance": deviance / float(d.weight.sum()) if d.weight.sum() > 0 else None,
            "null_deviance": null,
            "deviance_explained": 1.0 - deviance / null if null > 0 else None,
            "gini": gini,
            "gini_normalised": gini / perfect if gini is not None and perfect else None,
            "auc": float(diag_row.auc),
            "log_likelihood": diag_row.log_likelihood,
            "log_likelihood_label": diag_row.log_likelihood_label,
            "aic": diag_row.aic,
            "bic": diag_row.bic,
            "is_quasi_likelihood": bool(diag_row.is_quasi_likelihood),
        },
        **_curves(d, variance, phi, rate_bins),
        "residual_histogram": {
            "edges": [float(e) for e in edges],
            "mass": _histogram(live, d.weight[np.isfinite(resid)], edges),
        },
        "qq": qq,
        "count_distribution": _count_distribution(d, kind, theta),
    }


# --------------------------------------------------------------------------- factors ---------


def _model_factors(model: Any, data: pl.DataFrame) -> tuple[list[str], list[str]]:
    """The model's own factors as (categorical, continuous) names, from its term slots."""
    formula = str(getattr(model, "formula", "") or "")
    excluded = {formula.split("~")[0].strip()} if "~" in formula else set()
    for spec in ("_exposure_spec", "_weights_spec", "_offset_spec"):
        value = getattr(model, spec, None)
        if isinstance(value, str):
            excluded.add(value)
    builder = getattr(model, "_builder", None)
    slots = getattr(builder, "_term_slots", None) or []
    cats: list[str] = []
    conts: list[str] = []
    for slot in slots:
        if slot.term_type == "intercept":
            continue
        for factor in slot.factors:
            if factor in excluded or factor not in data.columns:
                continue
            categorical = slot.term_type in _CATEGORICAL_TERMS or (
                len(slot.factors) > 1 and not data.schema[factor].is_numeric()
            )
            if categorical and factor not in cats:
                cats.append(factor)
                if factor in conts:
                    conts.remove(factor)
            elif not categorical and factor not in cats and factor not in conts:
                conts.append(factor)
    return cats, conts


def _level_labels(train: Any, feature: str) -> list[str]:
    return [str(b.level) for b in train.factor_diagnostics.get(feature, [])]


def _band_defs(feature: str, ftype: str, train: Any) -> dict[str, Any] | None:
    """The bands of a factor, the same for every dataset, from the train diagnostics."""
    if ftype == "categorical":
        labels = _level_labels(train, feature)
        if not labels:
            return None
        shown = labels[:_MAX_LEVELS]
        out = {"kind": "levels", "levels": shown, "other": len(labels) > len(shown)}
        out["labels"] = shown + (["other levels"] if out["other"] else [])
        return out
    bands = sorted(train.continuous_diagnostics.get(feature, []), key=lambda b: b.band)
    if not bands:
        return None
    edges = [float(bands[0].range_min)] + [float(b.range_max) for b in bands]
    return {
        "kind": "edges",
        "edges": edges,
        "mid": [float(b.midpoint) for b in bands],
        "partial_dep": [float(b.partial_dep) for b in bands],
        "labels": [f"{_short_num(a)}–{_short_num(b)}" for a, b in pairwise(edges)],
    }


def _short_num(x: float) -> str:
    return f"{x:.4g}"


def _string_column(series: pl.Series) -> np.ndarray:
    return series.cast(pl.String).fill_null(_MISSING_LEVEL).to_numpy()


def _codes(series: pl.Series, defs: dict[str, Any]) -> np.ndarray:
    """The band of every row: -1 for a missing value or a level the train data did not have."""
    if defs["kind"] == "levels":
        codes = {lv: i for i, lv in enumerate(defs["levels"])}
        other = len(defs["levels"]) if defs["other"] else -1
        uniques, inverse = np.unique(_string_column(series).astype(str), return_inverse=True)
        table = np.array([codes.get(u, other) for u in uniques], dtype=np.int64)
        return table[inverse]
    values = series.cast(pl.Float64).to_numpy()
    edges = np.asarray(defs["edges"], dtype=np.float64)
    idx = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(edges) - 2)
    return np.where(np.isfinite(values), idx, -1)


def _band_rows(
    codes: np.ndarray, n: int, d: _Scored, variance: np.ndarray, phi: float
) -> list[dict[str, Any]]:
    live = codes >= 0
    c = codes[live]
    w = d.weight[live]
    rows = np.bincount(c, minlength=n)
    mass = np.bincount(c, weights=w * d.exposure[live], minlength=n)
    actual = np.bincount(c, weights=w * d.y[live], minlength=n)
    expected = np.bincount(c, weights=w * d.mu[live], minlength=n)
    var = np.bincount(c, weights=phi * w * variance[live], minlength=n)
    total = float(mass.sum())
    return [
        _band_numbers(
            int(rows[i]),
            float(mass[i]),
            float(actual[i]),
            float(expected[i]),
            float(var[i]),
            total,
        )
        for i in range(n)
    ]


def _psi(first: list[float], second: list[float]) -> float | None:
    a, b = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    if a.sum() <= 0 or b.sum() <= 0:
        return None
    p = np.maximum(a / a.sum(), _PSI_FLOOR)
    q = np.maximum(b / b.sum(), _PSI_FLOOR)
    return float(np.sum((q - p) * np.log(q / p)))


def _unseen_mass(codes: np.ndarray, d: _Scored) -> float:
    """The mass of rows no band holds: levels the train data did not have, or missing."""
    return float(d.mass[codes < 0].sum())


def _main_columns(model: Any, feature: str) -> list[int]:
    """The design columns of the single-factor terms of ``feature``."""
    slots = getattr(getattr(model, "_builder", None), "_term_slots", None) or []
    out: list[int] = []
    for slot in slots:
        if slot.term_type != "intercept" and list(slot.factors) == [feature]:
            out += list(range(slot.col_start, slot.col_end))
    return out


def _inference_valid(model: Any) -> bool:
    status = getattr(model, "inference_status", None)
    return (status is None or status in _VALID_INFERENCE) and (
        _attr(model, "get_bread_matrix") is not None
    )


def _frame_on(template: pl.DataFrame, columns: dict[str, pl.Series]) -> pl.DataFrame:
    n = next(iter(columns.values())).len()
    frame = pl.concat([template] * n)
    return frame.with_columns(*columns.values())


def _effect_scale(link: str) -> tuple[str, bool]:
    if link == "log":
        return "relativity", True
    if link == "logit":
        return "odds multiplier", True
    return "effect", False


def _design(model: Any, frame: pl.DataFrame) -> np.ndarray:
    """The model's design matrix for ``frame``."""
    prepare = getattr(model, "prepare_input", None)
    data = prepare(frame) if prepare else model._prepare_prediction_data(frame, None, None)
    return np.asarray(model._builder.transform_new_data(data), dtype=np.float64)


def _surface_effect(
    model: Any,
    data: pl.DataFrame,
    columns: dict[str, pl.Series],
    design_cols: list[int],
    ref: dict[str, pl.Series],
    cov: np.ndarray | None,
    mult: bool,
    beta: np.ndarray,
) -> dict[str, list[float | None]] | None:
    """The effect of ``design_cols`` at each row of ``columns`` against the row ``ref``."""
    template = data.slice(0, 1)
    try:
        x = _design(model, _frame_on(template, columns))
        x0 = _design(model, _frame_on(template, ref))
    except _EFFECT_ERRORS:
        return None
    delta = x[:, design_cols] - x0[:, design_cols]
    eta = delta @ beta[design_cols]
    transform = np.exp if mult else (lambda v: v)
    out: dict[str, list[float | None]] = {"value": [float(v) for v in transform(eta)]}
    if cov is not None:
        se = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", delta, cov, delta), 0.0))
        out["lo"] = [float(v) for v in transform(eta - _Z_95 * se)]
        out["hi"] = [float(v) for v in transform(eta + _Z_95 * se)]
    return out


def _effect(
    model: Any,
    feature: str,
    ftype: str,
    data: pl.DataFrame,
    defs: dict[str, Any],
    mass: np.ndarray,
    cov_full: np.ndarray | None,
    *,
    link: str,
    beta: np.ndarray,
) -> dict[str, Any] | None:
    """A factor's fitted effect relative to a reference value, with a 95% band when valid."""
    cols = _main_columns(model, feature)
    if not cols:
        return None
    kind, mult = _effect_scale(link)
    dtype = data.schema[feature]
    cov = cov_full[np.ix_(cols, cols)] if cov_full is not None else None
    if ftype == "categorical":
        levels = list(defs["levels"])
        as_type = data[feature].cast(pl.String)
        ref_level = _reference_level(model, cols, data, feature, levels)
        grid = pl.Series(feature, levels, dtype=pl.String)
        ref = pl.Series(feature, [ref_level], dtype=pl.String)
        if dtype != pl.String:
            grid = _cast_levels(grid, data[feature], as_type)
            ref = _cast_levels(ref, data[feature], as_type)
            if grid is None or ref is None:
                return None
        got = _surface_effect(model, data, {feature: grid}, cols, {feature: ref}, cov, mult, beta)
        if got is None:
            return None
        return {"kind": kind, "multiplicative": mult, "base": ref_level, "x": levels, **got}
    edges = np.asarray(defs["edges"], dtype=np.float64)
    values = data[feature].cast(pl.Float64).to_numpy()
    finite = np.isfinite(values)
    centre = float(np.average(values[finite], weights=mass[finite])) if mass[finite].sum() else 0.0
    # The curve covers the central 99% of the exposure: splines and linear terms run away in the
    # sparse far tail, and that would set the scale of the whole chart.
    low, high = _weighted_quantiles(values[finite], mass[finite], (0.005, 0.995))
    points = sorted(
        {low, high}
        | {
            float(np.clip(v, low, high))
            for a, b in pairwise(edges)
            for v in np.linspace(a, b, _EFFECT_POINTS_PER_BAND + 1)
        }
    )
    grid = pl.Series(feature, points, dtype=pl.Float64)
    ref = pl.Series(feature, [centre], dtype=pl.Float64)
    got = _surface_effect(model, data, {feature: grid}, cols, {feature: ref}, cov, mult, beta)
    if got is None:
        return None
    return {"kind": kind, "multiplicative": mult, "base": centre, "x": points, **got}


def _weighted_quantiles(
    values: np.ndarray, weights: np.ndarray, probs: tuple[float, float]
) -> tuple[float, float]:
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    if cumulative.size == 0 or cumulative[-1] <= 0:
        return 0.0, 0.0
    at = np.searchsorted(cumulative, [p * cumulative[-1] for p in probs])
    picked = values[order][np.minimum(at, values.size - 1)]
    return float(picked[0]), float(picked[1])


def _cast_levels(levels: pl.Series, column: pl.Series, as_string: pl.Series) -> pl.Series | None:
    """``levels`` (strings) in the dtype of ``column``, by matching their string form."""
    lookup = pl.DataFrame({"s": as_string, "v": column}).unique(subset="s", keep="first")
    joined = pl.DataFrame({"s": levels}).join(lookup, on="s", how="left", maintain_order="left")
    if joined["v"].null_count():
        return None
    return joined["v"].alias(levels.name)


def _reference_level(
    model: Any, cols: list[int], data: pl.DataFrame, feature: str, levels: list[str]
) -> str:
    """The level that the model's coding sets to zero; failing that, the first level."""
    names = list(model.feature_names)
    fitted = " ".join(names[c] for c in cols)
    return next((lv for lv in levels if f"[T.{lv}]" not in fitted), levels[0])


def _features(
    model: Any,
    diag: Any,
    scored: list[_Scored],
    variance: list[np.ndarray],
    phi: float,
) -> list[dict[str, Any]]:
    train = diag.train_test.train
    cov = None
    if _inference_valid(model):
        bread = _attr(model, "get_bread_matrix")
        scale = _attr(model, "scale")
        if bread is not None:
            cov = np.asarray(bread, dtype=np.float64) * (float(scale) if scale else 1.0)
    pdp = {p.variable: p for p in diag.partial_dependence or []}
    fits = []
    for factor in diag.factors:
        defs = _band_defs(factor.name, factor.factor_type, train)
        if defs is None or factor.name not in scored[0].data.columns:
            continue
        n = len(defs["labels"])
        bands = []
        unseen: list[float] = []
        missing = []
        for d, v in zip(scored, variance):
            codes = _codes(d.data[factor.name], defs)
            bands.append(_band_rows(codes, n, d, v, phi))
            unseen.append(_unseen_mass(codes, d))
            miss = d.data[factor.name].is_null().to_numpy() | (codes < 0)
            missing.append(
                {
                    "rows": int(miss.sum()),
                    "mass": float(d.mass[miss].sum()),
                    "mass_share": _ratio(float(d.mass[miss].sum()), float(d.mass.sum())),
                }
            )
        psi = None
        if len(scored) > 1:
            psi = _psi(
                [b["mass"] for b in bands[0]] + [unseen[0]],
                [b["mass"] for b in bands[1]] + [unseen[1]],
            )
        effect = None
        if factor.in_model:
            effect = _effect(
                model,
                factor.name,
                factor.factor_type,
                scored[0].data,
                defs,
                scored[0].mass,
                cov,
                link=model.link,
                beta=np.asarray(model.params, dtype=np.float64),
            )
        pd = pdp.get(factor.name)
        fits.append(
            {
                "feature": factor.name,
                "type": factor.factor_type,
                "in_model": bool(factor.in_model),
                "transform": factor.transform,
                "relative_importance": factor.relative_importance,
                "significance": _plain(factor.significance),
                "score_test": _plain(factor.score_test),
                "gvif": factor.gvif,
                "df": len(_main_columns(model, factor.name)),
                "residual_pattern": _plain(factor.residual_pattern),
                "coefficients": [_plain(c) for c in factor.coefficients or []],
                "labels": defs["labels"],
                "edges": defs.get("edges"),
                "mid": defs.get("mid"),
                "bands": bands,
                "missing": missing,
                "psi": psi,
                "effect": effect,
                "partial_dependence": _plain(pd) if pd is not None else None,
                "in_interactions": [],
            }
        )
    fits.sort(
        key=lambda f: (
            not f["in_model"],
            -(f["relative_importance"] or 0.0),
            -((f.get("score_test") or {}).get("statistic") or 0.0),
        )
    )
    return fits


def _plain(value: Any) -> Any:
    """A diagnostics dataclass as a plain dict, without the rounding ``to_dict`` applies."""
    if value is None:
        return None
    if hasattr(value, "__dataclass_fields__"):
        return {k: _plain(getattr(value, k)) for k in value.__dataclass_fields__}
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


# --------------------------------------------------------------------------- interactions ----


def _axis_codes(series: pl.Series, grid: Any, row: bool) -> tuple[np.ndarray, list[str], int]:
    """The surface cell index along one axis, as the pair diagnostics bin it."""
    edges = grid.row_edges if row else grid.col_edges
    levels = grid.row_levels if row else grid.col_levels
    if edges is not None:
        e = np.asarray(edges, dtype=np.float64)
        n = e.size - 1
        values = series.cast(pl.Float64).to_numpy()
        codes = np.clip(np.digitize(values, e[1:-1]), 0, n - 1)
        labels = [f"{_short_num(a)}–{_short_num(b)}" for a, b in pairwise(e)]
        return np.where(np.isfinite(values), codes, -1), labels, n
    names = [str(v) for v in levels or []]
    lookup = {lv: i for i, lv in enumerate(names)}
    other = lookup.get(_OTHER_LABEL, -1)
    uniques, inverse = np.unique(_string_column(series).astype(str), return_inverse=True)
    table = np.array([lookup.get(u, other) for u in uniques], dtype=np.int64)
    return table[inverse], names, len(names)


def _grid_points(grid: Any, row: bool) -> list[Any]:
    edges = grid.row_edges if row else grid.col_edges
    if edges is not None:
        e = np.asarray(edges, dtype=np.float64)
        return [float(v) for v in (e[:-1] + e[1:]) / 2]
    return [str(v) for v in (grid.row_levels if row else grid.col_levels) or []]


def _interaction_slot(model: Any, factors: list[str]) -> Any | None:
    slots = getattr(getattr(model, "_builder", None), "_term_slots", None) or []
    want = set(factors)
    return next(
        (s for s in slots if len(s.factors) >= 2 and set(s.factors) == want),
        None,
    )


def _pair_effect(
    model: Any,
    data: pl.DataFrame,
    grid: Any,
    slot: Any | None,
    r_pts: list[Any],
    c_pts: list[Any],
    *,
    link: str,
    beta: np.ndarray,
) -> dict[str, Any] | None:
    if slot is None or not r_pts or not c_pts:
        return None
    r_name, c_name = grid.row_axis, grid.col_axis
    if r_name not in data.columns or c_name not in data.columns:
        return None
    r_series: list[Any] = []
    c_series: list[Any] = []
    for r in r_pts:
        for c in c_pts:
            r_series.append(r)
            c_series.append(c)
    r_col = pl.Series(r_name, r_series)
    c_col = pl.Series(c_name, c_series)
    r_ref = pl.Series(r_name, [r_pts[0]])
    c_ref = pl.Series(c_name, [c_pts[0]])
    kind, mult = _effect_scale(link)
    cols = list(range(slot.col_start, slot.col_end))
    columns: dict[str, pl.Series] = {}
    refs: dict[str, pl.Series] = {}
    try:
        for name, col, ref_col in ((r_name, r_col, r_ref), (c_name, c_col, c_ref)):
            dtype = data.schema[name]
            columns[name] = col.cast(dtype) if col.dtype != dtype else col
            refs[name] = ref_col.cast(dtype) if ref_col.dtype != dtype else ref_col
    except pl.exceptions.PolarsError:
        return None
    got = _surface_effect(model, data, columns, cols, refs, None, mult, beta)
    if got is None:
        return None
    values = np.asarray(got["value"], dtype=np.float64).reshape(len(r_pts), len(c_pts))
    return {"kind": kind, "multiplicative": mult, "values": values.tolist()}


def _interactions(
    model: Any, diag: Any, scored: list[_Scored], variance: list[np.ndarray], phi: float
) -> list[dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for inter in diag.interactions or []:
        if inter.name in out:
            continue
        grid = inter.train_surface_grid
        data = scored[0].data
        if grid.row_axis not in data.columns or grid.col_axis not in data.columns:
            continue
        r_names = _axis_codes(data[grid.row_axis], grid, True)
        c_names = _axis_codes(data[grid.col_axis], grid, False)
        n_r, n_c = r_names[2], c_names[2]
        cells_by_ds = []
        for d, v in zip(scored, variance):
            rc, _, _ = _axis_codes(d.data[grid.row_axis], grid, True)
            cc, _, _ = _axis_codes(d.data[grid.col_axis], grid, False)
            codes = np.where((rc >= 0) & (cc >= 0), rc * n_c + cc, -1)
            cells = _band_rows(codes, n_r * n_c, d, v, phi)
            cells_by_ds.append(cells)
        slot = _interaction_slot(model, [inter.factor1, inter.factor2])
        effect = _pair_effect(
            model,
            data,
            grid,
            slot,
            _grid_points(grid, True),
            _grid_points(grid, False),
            link=model.link,
            beta=np.asarray(model.params, dtype=np.float64),
        )
        out[inter.name] = {
            "name": inter.name,
            "factor1": inter.factor1,
            "factor2": inter.factor2,
            "pair_type": inter.pair_type,
            "in_model": bool(inter.in_model),
            "representation": inter.representation,
            "row_axis": grid.row_axis,
            "col_axis": grid.col_axis,
            "rows": r_names[1],
            "cols": c_names[1],
            "n_rows": n_r,
            "n_cols": n_c,
            "cells": cells_by_ds,
            "effect": effect,
            "significance": _plain(inter.significance),
            "score_test": _plain(inter.score_test),
            "gvif": inter.gvif,
            "coefficients": [_plain(c) for c in inter.coefficients or []],
        }
    return list(out.values())


# --------------------------------------------------------------------------- fit section -----


def _influence(model: Any, d: _Scored, kind: str, power: float, theta: float) -> dict[str, Any]:
    """Leverage and Cook's distance from the fit's unscaled covariance."""
    bread = _attr(model, "get_bread_matrix")
    if bread is None:
        return {"available": False, "reason": "the model does not hold its covariance matrix"}
    bread = np.asarray(bread, dtype=np.float64)
    p = bread.shape[0]
    if d.n * p * p > _INFLUENCE_FLOPS:
        return {
            "available": False,
            "reason": f"left out on {d.n:,} rows × {p} coefficients to keep the report quick",
        }
    irls = _attr(model, "get_irls_weights")
    if irls is None:
        return {"available": False, "reason": "the model does not hold its working weights"}
    working = np.asarray(irls, dtype=np.float64)
    try:
        chunks = [
            np.einsum("ij,ij->i", np.asarray(x, dtype=np.float64) @ bread, x)
            for x in _iter_design_matrix_chunks(model, d.data)
        ]
        h = np.clip(np.concatenate(chunks) * working, 0.0, 0.9999)
    except _EFFECT_ERRORS as exc:
        return {"available": False, "reason": f"the design matrix could not be rebuilt: {exc}"}
    v = np.maximum(_variance(kind, d.mu, power, theta), 1e-300)
    r = np.sqrt(d.weight) * (d.y - d.mu) / np.sqrt(v)
    scale = _attr(model, "scale")
    phi = float(scale) if scale and scale > 0 else 1.0
    cooks = r * r * h / (p * phi * (1.0 - h) ** 2)
    top = np.argsort(-cooks, kind="stable")[:_TOP_INFLUENCE]
    return {
        "available": True,
        "mean_leverage": float(h.mean()),
        "max_leverage": float(h.max()),
        "leverage_rule": 2.0 * p / d.n,
        "high_leverage_rows": int(np.sum(h > 2.0 * p / d.n)),
        "max_cooks": float(cooks.max()),
        "cooks_rule": 4.0 / d.n,
        "influential_rows": int(np.sum(cooks > 4.0 / d.n)),
        "top": [
            {
                "row": int(i),
                "y": float(d.y[i]),
                "mu": float(d.mu[i]),
                "exposure": float(d.exposure[i]),
                "leverage": float(h[i]),
                "pearson": float(r[i]),
                "cooks": float(cooks[i]),
            }
            for i in top
        ],
    }


def _top_correlations(
    cov: np.ndarray, names: list[str], owner: np.ndarray, intercept: np.ndarray
) -> list[dict[str, Any]]:
    """The largest correlations between coefficient estimates of different terms."""
    if cov.shape[0] > _MAX_CORRELATION_COLUMNS:
        return []
    sd = np.sqrt(np.maximum(np.diag(cov), 1e-300))
    corr = cov / np.outer(sd, sd)
    mask = np.triu(np.ones_like(corr, dtype=bool), 1)
    mask &= ~(intercept[:, None] | intercept[None, :])
    mask &= owner[:, None] != owner[None, :]
    i, j = np.where(mask)
    if i.size == 0:
        return []
    order = np.argsort(-np.abs(corr[i, j]), kind="stable")[:_TOP_CORRELATIONS]
    return [
        {"a": names[int(i[k])], "b": names[int(j[k])], "corr": float(corr[i[k], j[k]])}
        for k in order
    ]


def _term_owners(model: Any, n: int) -> tuple[np.ndarray, np.ndarray]:
    """For each design column, the index of its term and whether it is the intercept."""
    slots = getattr(getattr(model, "_builder", None), "_term_slots", None) or []
    owner = np.full(n, -1)
    intercept = np.zeros(n, dtype=bool)
    for s, slot in enumerate(slots):
        owner[slot.col_start : slot.col_end] = s
        if slot.term_type == "intercept":
            intercept[slot.col_start : slot.col_end] = True
    return owner, intercept


def _correlations(model: Any) -> list[dict[str, Any]]:
    bread = _attr(model, "get_bread_matrix")
    if bread is None:
        return []
    cov = np.asarray(bread, dtype=np.float64)
    owner, intercept = _term_owners(model, cov.shape[0])
    return _top_correlations(cov, list(model.feature_names), owner, intercept)


def _terms(model: Any) -> list[dict[str, Any]]:
    slots = getattr(getattr(model, "_builder", None), "_term_slots", None) or []
    return [
        {
            "term": slot.term_name,
            "type": slot.term_type,
            "factors": list(slot.factors),
            "columns": int(slot.col_end - slot.col_start),
        }
        for slot in slots
    ]


def _fit_section(model: Any, diag: Any, kind: str) -> dict[str, Any]:
    summary = dict(diag.model_summary)
    fd = _attr(model, "fit_diagnostics") or {}
    trace = getattr(model, "__dict__", {}).get("solver_trace") or ()
    return {
        "summary": summary,
        "inference_status": getattr(model, "inference_status", None),
        "inference_valid": _inference_valid(model),
        "optimizer_route": getattr(model, "optimizer_route", None) or fd.get("optimizer_route"),
        "solver_status": fd.get("solver_status"),
        "fit_diagnostics": {k: v for k, v in fd.items() if k != "target_encoding"},
        "warnings": list(_attr(model, "warnings") or []),
        "solver_trace": [list(step) for step in trace],
        "boundary_active": summary.get("boundary_active_coefficients") or [],
        "calibration": dict(diag.calibration or {}),
        "residual_summary": _plain(diag.residual_summary),
        "model_comparison": dict(diag.model_comparison or {}),
        "overdispersion": _plain(diag.overdispersion),
        "smooth_terms": [s.to_dict() for s in diag.smooth_terms or []],
        "spline_info": _plain(diag.spline_info),
        "regularization_path": _attr(model, "regularization_path"),
        "cv_profile": _attr(model, "cv_profile"),
        "vif": [_plain(v) for v in diag.vif or []],
        "kind": kind,
    }


# --------------------------------------------------------------------------- review ----------


def _flag(
    check: str,
    status: str,
    message: str,
    *,
    value: Any = None,
    threshold: Any = None,
    dataset: str | None = None,
    feature: str | None = None,
    cls: str | None = None,
) -> dict[str, Any]:
    return {
        "check": check,
        "status": status,
        "message": message,
        "value": value,
        "threshold": threshold,
        "dataset": dataset,
        "feature": feature,
        "class": cls,
    }


def _review_bands(
    flags: list[dict[str, Any]],
    report: dict[str, Any],
    th: dict[str, float | None],
    index: int,
    cls: str | None = None,
) -> None:
    ftol, fmin, fz, nz = th["factor_ae"], th["factor_min_mass"], th["factor_z"], th["near_miss_z"]
    ds = report["datasets"][index]
    label = ds["label"]
    out_of_sample = index > 0 or len(report["datasets"]) == 1
    if ftol is None:
        return
    start = len(flags)
    covered, total_mass = 0.0, 0.0
    for feature in report["features"]:
        bands = feature["bands"][index]
        total = sum(b["mass"] for b in bands)
        for b in bands:
            b.pop("flag", None)
        eligible = [b for b in bands if total > 0 and b["mass"] / total >= (fmin or 0.0)]
        covered += sum(b["mass"] for b in eligible)
        total_mass += total
        bad = [
            b
            for b in eligible
            if b["ae"] is not None
            and abs(b["ae"] - 1.0) > ftol
            and (fz is None or (b["z"] is not None and abs(b["z"]) >= fz))
        ]
        near = [
            b
            for b in eligible
            if b not in bad
            and nz is not None
            and out_of_sample
            and b["ae"] is not None
            and abs(b["ae"] - 1.0) > ftol
            and b["z"] is not None
            and abs(b["z"]) >= nz
        ]
        labels = feature["labels"]
        named = {id(b): labels[i] for i, b in enumerate(bands)}
        for b in bad:
            b["flag"] = "warning"
        for b in near:
            b["flag"] = "near_miss"
            flags.append(
                _flag(
                    "factor_ae",
                    "info",
                    f"near miss: {named[id(b)]} A/E {b['ae']:.3f}, z {b['z']:+.1f}",
                    value=b["ae"],
                    threshold=ftol,
                    dataset=label,
                    feature=feature["feature"],
                    cls=cls,
                )
            )
        if bad:
            worst = max(bad, key=lambda b: abs(b["ae"] - 1.0))
            z = f", z {worst['z']:+.1f}" if worst["z"] is not None else ""
            flags.append(
                _flag(
                    "factor_ae",
                    "warning",
                    f"{len(bad)} band(s) off by more than {ftol:g}"
                    + (f" and {fz:g} standard deviations" if fz is not None else "")
                    + f" (worst: {named[id(worst)]}, A/E {worst['ae']:.3f}{z})",
                    value=len(bad),
                    threshold=ftol,
                    dataset=label,
                    feature=feature["feature"],
                    cls=cls,
                )
            )
    if not any(f["status"] == "warning" for f in flags[start:]):
        share = covered / total_mass if total_mass else 0.0
        flags.append(
            _flag(
                "factor_ae",
                "pass",
                f"no factor band off by more than {ftol:g}"
                + (f" and {fz:g} standard deviations" if fz is not None else "")
                + f"; the bands checked hold {share:.1%} of the mass",
                threshold=ftol,
                dataset=label,
                cls=cls,
            )
        )


def _review_overall(
    flags: list[dict[str, Any]],
    ds: dict[str, Any],
    tol: float | None,
    fz: float | None,
    cls: str | None = None,
) -> None:
    """Overall A/E: past its tolerance and, like a band, past ``factor_z`` standard deviations."""
    ae, z = ds["metrics"]["ae"], ds["metrics"].get("ae_z")
    if tol is None or ae is None:
        return
    outside = abs(ae - 1.0) > tol
    credible = fz is None or z is None or abs(z) >= fz
    tail = f" (z {z:+.1f})" if z is not None else ""
    flags.append(
        _flag(
            "overall_ae",
            "warning" if outside and credible else "pass",
            f"overall A/E {ae:.4f} is {'outside' if outside else 'within'} 1 ± {tol:g}{tail}",
            value=ae,
            threshold=tol,
            dataset=ds["label"],
            cls=cls,
        )
    )


def _review_deciles(
    flags: list[dict[str, Any]],
    ds: dict[str, Any],
    th: dict[str, float | None],
    out_of_sample: bool,
    cls: str | None = None,
) -> None:
    tol, fz, nz = th["decile_ae"], th["factor_z"], th["near_miss_z"]
    if tol is None:
        return
    label = ds["label"]
    start = len(flags)
    for b in ds["deciles"]:
        ae, z = b.get("ae"), b.get("z")
        if ae is None or abs(ae - 1.0) <= tol:
            continue
        text = f"decile {b['band']} A/E {ae:.3f}" + (f" (z {z:+.1f})" if z is not None else "")
        if fz is None or (z is not None and abs(z) >= fz):
            flags.append(
                _flag("decile_ae", "warning", text, value=ae, threshold=tol, dataset=label, cls=cls)
            )
        elif nz is not None and z is not None and abs(z) >= nz and out_of_sample:
            flags.append(
                _flag(
                    "decile_ae",
                    "info",
                    "near miss: " + text,
                    value=ae,
                    threshold=tol,
                    dataset=label,
                    cls=cls,
                )
            )
    if not any(f["status"] == "warning" for f in flags[start:]):
        flags.append(
            _flag(
                "decile_ae",
                "pass",
                f"no fitted-value decile off by more than {tol:g}"
                + (f" and {fz:g} standard deviations" if fz is not None else ""),
                threshold=tol,
                dataset=label,
                cls=cls,
            )
        )


def _review_interactions(
    flags: list[dict[str, Any]],
    report: dict[str, Any],
    th: dict[str, float | None],
    cls: str | None = None,
) -> None:
    tol, fz, fmin = th["interaction_ae"], th["factor_z"], th["factor_min_mass"]
    if tol is None:
        return
    index = len(report["datasets"]) - 1
    label = report["datasets"][index]["label"]
    for inter in report["interactions"]:
        cells = inter["cells"][index]
        total = sum(c["mass"] for c in cells)
        off = [
            (i, c)
            for i, c in enumerate(cells)
            if total > 0
            and c["mass"] / total >= (fmin or 0.0)
            and c["ae"] is not None
            and abs(c["ae"] - 1.0) > tol
            and (fz is None or (c["z"] is not None and abs(c["z"]) >= fz))
        ]
        if off:
            worst_i, worst = max(off, key=lambda t: abs(t[1]["ae"] - 1.0))
            r, c = divmod(worst_i, inter["n_cols"])
            status = "info" if inter["in_model"] else "warning"
            flags.append(
                _flag(
                    "interaction_ae",
                    status,
                    f"{len(off)} cell(s) off by more than {tol:g}; worst "
                    f"{inter['rows'][r]} × {inter['cols'][c]}, A/E {worst['ae']:.3f}",
                    value=len(off),
                    threshold=tol,
                    dataset=label,
                    feature=inter["name"],
                    cls=cls,
                )
            )


def _review_fit(
    flags: list[dict[str, Any]],
    report: dict[str, Any],
    th: dict[str, float | None],
    dispersion_ratio: float,
    fixed_scale: bool,
    pearson: float | None,
) -> None:
    fit = report["fit"]
    summary = fit["summary"]
    if summary.get("converged") is False:
        flags.append(
            _flag(
                "converged",
                "critical",
                f"the fit did not converge in {summary.get('iterations')} iterations",
            )
        )
    else:
        flags.append(
            _flag(
                "converged",
                "pass",
                f"converged in {summary.get('iterations')} iterations",
                value=summary.get("iterations"),
            )
        )
    if not fit["inference_valid"]:
        status = fit["inference_status"] or "unavailable"
        flags.append(
            _flag(
                "inference",
                "info",
                f"standard errors are not valid as usual ({status}); "
                "confidence bands and p-values are left out or naive",
            )
        )
    tol = th["dispersion"]
    if fixed_scale and tol is not None:
        over = dispersion_ratio > tol
        extra = (
            f"; the row-level Pearson φ is {pearson:.2f}, inflated by rows with small expected values"
            if pearson is not None and pearson > dispersion_ratio * 1.5
            else ""
        )
        flags.append(
            _flag(
                "dispersion",
                "warning" if over else "pass",
                f"variance-to-mean dispersion {dispersion_ratio:.3f}{extra}"
                + ("; standard errors are too narrow, consider a quasi family" if over else ""),
                value=dispersion_ratio,
                threshold=tol,
            )
        )
    gtol, ptol = th["gvif"], th["term_p"]
    for f in report["features"]:
        if not f["in_model"]:
            st = f.get("score_test")
            if st and st.get("significant"):
                gain = st.get("expected_dev_pct")
                flags.append(
                    _flag(
                        "score_test",
                        "info",
                        f"not in the model but its score test is significant (p "
                        f"{st['pvalue']:.3g}"
                        + (f", about {gain:.2f}% of the deviance" if gain is not None else "")
                        + ")",
                        value=st.get("pvalue"),
                        feature=f["feature"],
                    )
                )
            continue
        sig = f.get("significance") or {}
        if (
            ptol is not None
            and fit["inference_valid"]
            and sig.get("p") is not None
            and sig["p"] > ptol
        ):
            flags.append(
                _flag(
                    "term_p",
                    "info",
                    f"its coefficients are not jointly significant (p {sig['p']:.3g})",
                    value=sig["p"],
                    threshold=ptol,
                    feature=f["feature"],
                )
            )
        df = max(int(f.get("df") or 1), 1)
        if gtol is not None and f.get("gvif") is not None and f["gvif"] ** (1.0 / df) > gtol:
            adjusted = f["gvif"] ** (1.0 / df)
            own = bool(f["in_interactions"])
            flags.append(
                _flag(
                    "gvif",
                    "info" if own else "warning",
                    f"GVIF^(1/df) {adjusted:.1f}"
                    + (", inflated by an interaction it is part of" if own else ""),
                    value=adjusted,
                    threshold=gtol,
                    feature=f["feature"],
                )
            )
    for item in fit["boundary_active"][:20]:
        name = item.get("feature") if isinstance(item, dict) else item
        flags.append(
            _flag("boundary", "info", f"coefficient {name} is held at a constraint boundary")
        )
    for warning in report.get("diagnostic_warnings") or []:
        if warning["type"] in _WARNINGS_NOTED:
            flags.append(_flag("diagnostics", "info", warning["message"]))


# Passed through as notes; the rest are replaced by a credibility-aware check or by the page.
_WARNINGS_NOTED = frozenset(
    {
        "robust_se_unavailable",
        "coefficient_inference_unavailable",
        "test_weights_unavailable",
        "base_predictions_unavailable",
        "interaction_score_test_unavailable",
        "undersmoothed",
        "insignificant_smooth",
        "negbinomial_small_theta",
        "negbinomial_large_theta",
        "negbinomial_regularization",
    }
)


def _review_holdout(
    flags: list[dict[str, Any]], datasets: list[dict[str, Any]], th: dict[str, float | None]
) -> None:
    if len(datasets) == 1:
        flags.append(
            _flag(
                "holdout_data",
                "info",
                "train data only: every metric is in-sample. Pass test_data for out-of-sample "
                "evidence",
                dataset=datasets[0]["label"],
            )
        )
    else:
        train, test = datasets[0], datasets[-1]
        first, later = train["metrics"]["deviance_explained"], test["metrics"]["deviance_explained"]
        if later is not None:
            flags.append(
                _flag(
                    "beats_null",
                    "critical" if later <= 0 else "pass",
                    f"deviance explained on {test['label']}: {later:.4f}",
                    value=later,
                    threshold=0.0,
                    dataset=test["label"],
                )
            )
        gap = th["train_gap"]
        if gap is not None and first is not None and later is not None:
            drop = first - later
            flags.append(
                _flag(
                    "train_gap",
                    "warning" if drop > gap else "pass",
                    f"deviance explained drops by {drop:.4f} from {train['label']} to "
                    f"{test['label']}",
                    value=drop,
                    threshold=gap,
                    dataset=test["label"],
                )
            )


def _review_effects(flags: list[dict[str, Any]], features: list[dict[str, Any]]) -> None:
    """A fitted factor with no effect curve is a chart the page could not draw: say so."""
    for f in features:
        if f["in_model"] and f["df"] > 0 and f["effect"] is None:
            flags.append(
                _flag(
                    "effect",
                    "info",
                    "its fitted effect could not be evaluated on a grid, so it is not drawn",
                    feature=f["feature"],
                )
            )


def _review_population(
    flags: list[dict[str, Any]],
    features: list[dict[str, Any]],
    datasets: list[dict[str, Any]],
    th: dict[str, float | None],
) -> None:
    warn, crit = th["psi_warn"], th["psi_critical"]
    if len(datasets) > 1:
        for f in features:
            psi = f.get("psi")
            if psi is None or (warn is None and crit is None):
                continue
            status = (
                "critical"
                if crit is not None and psi > crit
                else "warning"
                if warn is not None and psi > warn
                else "pass"
            )
            flags.append(
                _flag(
                    "psi",
                    status,
                    f"PSI {psi:.4f} from {datasets[0]['label']} to {datasets[-1]['label']}",
                    value=psi,
                    threshold=warn if status != "critical" else crit,
                    dataset=datasets[-1]["label"],
                    feature=f["feature"],
                )
            )
    unseen = th["unseen_mass"]
    if unseen is not None and len(datasets) > 1:
        for f in features:
            if f["type"] != "categorical":
                continue
            m = f["missing"][-1]
            if not m["rows"]:
                continue
            share = m["mass_share"] or 0.0
            flags.append(
                _flag(
                    "unseen_values",
                    "warning" if share > unseen else "info",
                    f"{m['rows']:,} row(s) with a missing level or one the train data did not "
                    f"have, {share:.2%} of the mass",
                    value=share,
                    threshold=unseen,
                    dataset=datasets[-1]["label"],
                    feature=f["feature"],
                )
            )


def _review(
    report: dict[str, Any],
    th: dict[str, float | None],
    dispersion_ratio: float,
    fixed_scale: bool,
    pearson: float | None,
) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    datasets = report["datasets"]
    for i, ds in enumerate(datasets):
        tol = th["ae_train"] if i == 0 else th["ae_test"]
        _review_overall(flags, ds, tol, th["factor_z"])
        _review_deciles(flags, ds, th, i > 0 or len(datasets) == 1)
        _review_bands(flags, report, th, i)
    _review_holdout(flags, datasets, th)
    _review_population(flags, report["features"], datasets, th)
    _review_interactions(flags, report, th)
    _review_effects(flags, report["features"])
    _review_fit(flags, report, th, dispersion_ratio, fixed_scale, pearson)
    return flags


def metric_writer(out: dict[str, float]) -> Callable[[str, Any], None]:
    """A function that adds a finite metric to ``out`` under a valid, unique name."""

    def put(key: str, value: Any) -> None:
        if value is None or isinstance(value, bool):
            return
        v = float(value)
        if not math.isfinite(v):
            return
        key = _KEY_CHARS.sub("_", key)[:_MAX_KEY]
        base, i = key, 2
        while key in out:
            suffix = f"_{i}"
            key = base[: _MAX_KEY - len(suffix)] + suffix
            i += 1
        out[key] = v

    return put


def _summary_metrics(report: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    put = metric_writer(out)
    for ds in report["datasets"]:
        m, label = ds["metrics"], ds["label"]
        for name in (
            "deviance",
            "null_deviance",
            "deviance_explained",
            "gini",
            "gini_normalised",
            "auc",
            "ae",
            "aic",
            "bic",
            "mass",
        ):
            put(f"{label}_{name}", m.get(name))
        put(f"{label}_rows", ds["rows"])
    summary = report["fit"]["summary"]
    put("n_params", summary.get("n_params"))
    put("iterations", summary.get("iterations"))
    put("dispersion", report["fit"]["dispersion"])
    flags = report["review"]
    put("review_warnings", sum(f["status"] == "warning" for f in flags))
    put("review_critical", sum(f["status"] == "critical" for f in flags))
    return out


# --------------------------------------------------------------------------- entry point -----


def _for_rows(override: Any, data: pl.DataFrame) -> Any:
    """An exposure override if it applies to ``data``: a column, or an array of its length."""
    if override is None or isinstance(override, str):
        return override
    return override if np.shape(override) == (data.height,) else None


def check_arguments(train_data: Any, test_data: Any, labels: tuple[str, str]) -> None:
    """Refuse what neither kind of report can use."""
    if len(labels) != 2 or len({str(label) for label in labels}) != 2:
        raise ValidationError("labels must name the two datasets, with different names.")
    if not isinstance(train_data, pl.DataFrame):
        raise ValidationError("train_data must be a polars DataFrame.")
    if test_data is not None and not isinstance(test_data, pl.DataFrame):
        raise ValidationError("test_data must be a polars DataFrame.")


def build_report(
    model: GLMModel,
    train_data: pl.DataFrame,
    test_data: pl.DataFrame | None = None,
    *,
    name: str | None = None,
    thresholds: dict[str, Any] | None = None,
    categorical_factors: list[str] | None = None,
    continuous_factors: list[str] | None = None,
    interactions: list[Any] | None = None,
    base_predictions: str | dict[str, str] | None = None,
    n_factor_bins: int = 10,
    exposure: str | np.ndarray | None = None,
    weights: str | np.ndarray | None = None,
    labels: tuple[str, str] = ("train", "test"),
) -> dict[str, Any]:
    """The report for ``GLMModel.validation_report``; see that method."""
    th = resolve_thresholds(thresholds)
    check_arguments(train_data, test_data, labels)
    cats, conts = _model_factors(model, train_data)
    for extra, target in ((categorical_factors, cats), (continuous_factors, conts)):
        for column in extra or []:
            if column not in train_data.columns:
                raise ValidationError(f"factor {column!r} is not a column of train_data.")
            if column not in target:
                target.append(column)
    diag = model.diagnostics(
        train_data=train_data,
        test_data=test_data,
        categorical_factors=cats,
        continuous_factors=conts,
        n_factor_bins=n_factor_bins,
        detect_interactions=True,
        interactions=interactions,
        include_fitted_interactions=True,
        base_predictions=base_predictions,  # type: ignore[arg-type]  # a dict is accepted too
        exposure=exposure,
        weights=weights,
        # An out-of-core fit keeps no rows to compute them from.
        compute_robust_se=getattr(model, "fit_mode", None) != "out_of_core",
    )
    kind, power, theta = _family(model)
    response, exposure_col, exposure_arr = _resolve_offset_and_response(model, train_data, exposure)
    y, mu, _ = _extract_response_and_predictions(model, train_data, exposure)
    n = y.shape[0]
    has_exposure = exposure_arr is not None
    train_weight = _resolve_weights(model, train_data, weights)
    scored = [
        _Scored(
            labels[0],
            train_data,
            y,
            mu,
            exposure_arr if exposure_arr is not None else np.ones(n),
            train_weight if train_weight is not None else np.ones(n),
            _offset(model, train_data, exposure=exposure, train=True),
            has_exposure,
        )
    ]
    if test_data is not None:
        warnings_sink: list[dict[str, str]] = []
        y_t, mu_t, exp_t, _ = _extract_test_arrays(
            test_data, model, response, exposure_col, "auto", exposure
        )
        if y_t is None or mu_t is None or exp_t is None:
            raise ValidationError("The model has no response column to score test_data with.")
        wt = _resolve_test_weights(model, test_data, weights, warnings_sink)
        scored.append(
            _Scored(
                labels[1],
                test_data,
                y_t,
                mu_t,
                exp_t,
                wt if wt is not None else np.ones(y_t.shape[0]),
                _offset(model, test_data, exposure=_for_rows(exposure, test_data)),
                has_exposure,
            )
        )
    variance = [_variance(kind, d.mu, power, theta) for d in scored]
    train = scored[0]
    spread = float(np.sum(train.weight * (train.y - train.mu) ** 2))
    denom = float(np.sum(train.weight * variance[0]))
    ratio = spread / denom if denom > 0 else 1.0
    fixed_scale = kind in ("poisson", "binomial") and not str(model.family).lower().startswith(
        "quasi"
    )
    phi = 1.0 if kind == "binomial" and fixed_scale else ratio
    if fixed_scale or kind == "negbin":
        phi = max(phi, 1.0)
    rate_bins = _rate_edges(train.rate)
    tt = diag.train_test
    datasets = [_dataset(scored[0], tt.train, model, kind, power, theta, phi, rate_bins)]
    if len(scored) > 1 and tt.test is not None:
        datasets.append(_dataset(scored[1], tt.test, model, kind, power, theta, phi, rate_bins))
    features = _features(model, diag, scored, variance, phi)
    inter = _interactions(model, diag, scored, variance, phi)
    for pair in inter:
        if not pair["in_model"]:
            continue
        for f in features:
            if f["feature"] in (pair["factor1"], pair["factor2"]):
                f["in_interactions"].append(pair["name"])
    fit = _fit_section(model, diag, kind)
    summary = fit["summary"]
    pearson = None
    if diag.overdispersion:
        pearson = diag.overdispersion.get("pearson_dispersion")
    fit |= {
        "dispersion": ratio,
        "credibility_dispersion": phi,
        "fixed_scale": fixed_scale,
        "pearson_dispersion": summary.get("scale_pearson") if pearson is None else pearson,
        "base_rate": _base_rate(model),
        "influence": _influence(model, train, kind, power, theta),
        "coefficient_correlations": _correlations(model),
        "terms": _terms(model),
    }
    report: dict[str, Any] = {
        "report_version": REPORT_VERSION,
        "name": name,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "model": {
            "formula": summary.get("formula"),
            "family": str(model.family),
            "family_kind": kind,
            "link": model.link,
            "n_obs": summary.get("n_obs"),
            "n_params": summary.get("n_params"),
            "df_resid": summary.get("df_resid"),
            "response": response,
            "exposure_column": exposure_col,
            "weights_column": _column_name(model, "_weights_spec"),
            "offset_column": _column_name(model, "_offset_spec"),
            "deserialized": bool(getattr(model, "_is_deserialized", False)),
            "rustystats_version": _version(),
            "terms_dict": _plain(_attr(model, "terms_dict")),
            "interactions_spec": _plain(_attr(model, "interactions_spec")),
        },
        "datasets": datasets,
        "features": features,
        "interactions": inter,
        "interaction_candidates": [_plain(c) for c in diag.interaction_candidates or []],
        "coefficients": [_plain(c) for c in diag.coefficient_summary or []],
        "encodings": [_plain(c) for c in diag.encoding_diagnostics or []],
        "factor_deviance": [_plain(c) for c in diag.factor_deviance or []],
        "base_predictions": _plain(diag.base_predictions_by_role),
        "fit": fit,
        "thresholds": th,
        "threshold_descriptions": THRESHOLD_DESCRIPTIONS,
        "diagnostic_warnings": list(diag.warnings or []),
        "caveat": (
            "The report describes the data passed to it; it cannot verify which rows were used "
            "to fit the model and which were held out."
        ),
    }
    report["review"] = _review(report, th, ratio, fixed_scale, pearson)
    report["summary_metrics"] = _summary_metrics(report)
    return json_safe(report)


def _base_rate(model: Any) -> float | None:
    params = _attr(model, "params")
    if params is None or not len(params):
        return None
    names = list(getattr(model, "feature_names", []) or [])
    if "Intercept" not in names:
        return None
    intercept = float(np.asarray(params, dtype=np.float64)[names.index("Intercept")])
    return math.exp(intercept) if model.link == "log" else None


def _column_name(model: Any, attr: str) -> str | None:
    value = getattr(model, attr, None)
    return value if isinstance(value, str) else None


def _version() -> str:
    from rustystats import __version__

    return str(__version__)
