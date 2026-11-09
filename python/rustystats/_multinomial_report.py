"""The validation report of a multinomial model (``MultinomialModel.validation_report``).

A multinomial fit is reviewed class by class. Each class gets a one-vs-rest view with the shape
of a GLM report (datasets, factors, interactions, coefficients), so ``report_html`` draws it with
the same charts and the review applies the same checks. Beside the views the report holds what
belongs to the model as a whole: the deviance against the class shares, accuracy, the confusion
and expected-count matrices, the class mix, and each factor's joint test across the classes.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from itertools import pairwise
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from rustystats._report import (
    _CATEGORICAL_TERMS,
    _MAX_LEVELS,
    _VALID_INFERENCE,
    REPORT_VERSION,
    THRESHOLD_DESCRIPTIONS,
    _attr,
    _axis_codes,
    _band_rows,
    _codes,
    _column_name,
    _curves,
    _effect,
    _flag,
    _gini,
    _grid_points,
    _interaction_slot,
    _main_columns,
    _model_factors,
    _null_deviance,
    _pair_effect,
    _plain,
    _psi,
    _rate_edges,
    _ratio,
    _review_bands,
    _review_deciles,
    _review_effects,
    _review_holdout,
    _review_interactions,
    _review_overall,
    _review_population,
    _Scored,
    _short_num,
    _string_column,
    _term_owners,
    _terms,
    _top_correlations,
    _total_z,
    _unit_deviance,
    _unseen_mass,
    _variance,
    _version,
    check_arguments,
    json_safe,
    metric_writer,
    resolve_thresholds,
)
from rustystats._rustystats import chi2_cdf_py as _chi2_cdf
from rustystats.diagnostics.pair_diagnostics import (
    _bin_categorical_levels,
    _bin_continuous_edges,
    _normalize_pair_spec,
)
from rustystats.exceptions import ValidationError

if TYPE_CHECKING:
    from rustystats.multinomial import MultinomialModel

__all__ = ["build_multinomial_report"]

_PAIR_BINS = 5
_PAIR_LEVELS = 8


@dataclass
class _Frame:
    """One dataset of a multinomial report with its per-row arrays."""

    label: str
    frame: pl.DataFrame
    probabilities: np.ndarray
    y_codes: np.ndarray
    weight: np.ndarray

    def scored(self, k: int) -> _Scored:
        """The dataset as a one-vs-rest problem for class ``k``."""
        n = self.y_codes.shape[0]
        return _Scored(
            self.label,
            self.frame,
            (self.y_codes == k).astype(np.float64),
            self.probabilities[:, k],
            np.ones(n),
            self.weight,
            None,
            False,
        )


def _factors(
    model: Any,
    data: pl.DataFrame,
    categorical: list[str] | None,
    continuous: list[str] | None,
) -> tuple[list[str], list[str]]:
    cats, conts = _model_factors(model, data)
    # Target encodings live outside the shared design, so they have no term slot.
    for name, spec in (model._terms_dict or {}).items():
        if name in data.columns and name not in cats and name not in conts:
            kind = spec.get("type")
            numeric = data.schema[name].is_numeric() and kind not in _CATEGORICAL_TERMS
            (conts if numeric else cats).append(name)
    for extra, target in ((categorical, cats), (continuous, conts)):
        for column in extra or []:
            if column not in data.columns:
                raise ValidationError(f"factor {column!r} is not a column of train_data.")
            if column not in target:
                target.append(column)
    return cats, conts


def _band_defs(name: str, ftype: str, frame: pl.DataFrame, weight: np.ndarray, n_bins: int) -> dict:
    """The bands of a factor, set on the train data and shared by every dataset and class."""
    if ftype == "categorical":
        uniques, inverse = np.unique(_string_column(frame[name]).astype(str), return_inverse=True)
        mass = np.bincount(inverse, weights=weight, minlength=uniques.size)
        order = np.argsort(-mass, kind="stable")
        levels = [str(uniques[i]) for i in order]
        shown = levels[:_MAX_LEVELS]
        other = len(levels) > len(shown)
        return {
            "kind": "levels",
            "levels": shown,
            "other": other,
            "labels": shown + (["other levels"] if other else []),
        }
    edges = _bin_continuous_edges(frame[name].cast(pl.Float64).to_numpy(), n_bins)
    return {
        "kind": "edges",
        "edges": [float(e) for e in edges],
        "mid": [float(v) for v in (edges[:-1] + edges[1:]) / 2],
        "labels": [f"{_short_num(float(a))}–{_short_num(float(b))}" for a, b in pairwise(edges)],
    }


def _wald(beta: np.ndarray, cov: np.ndarray) -> tuple[float, float] | None:
    """The joint Wald statistic of ``beta`` and its p-value."""
    try:
        chi2 = float(beta @ np.linalg.solve(cov, beta))
    except np.linalg.LinAlgError:
        return None
    if not math.isfinite(chi2) or chi2 < 0:
        return None
    return chi2, max(0.0, 1.0 - float(_chi2_cdf(chi2, float(beta.size))))


def _joint_tests(model: Any, names: list[str], cov: np.ndarray | None) -> dict[str, dict]:
    """Each fitted factor's joint test across the classes, and its share of the total."""
    if cov is None:
        return {}
    blocks, p = model.params.shape
    out: dict[str, dict[str, Any]] = {}
    for name in names:
        cols = _main_columns(model, name)
        if not cols:
            continue
        idx = np.array([b * p + c for b in range(blocks) for c in cols], dtype=np.intp)
        got = _wald(np.asarray(model.params, dtype=np.float64).ravel()[idx], cov[np.ix_(idx, idx)])
        if got is not None:
            out[name] = {"chi2": got[0], "p": got[1], "dev_contrib": got[0], "df": int(idx.size)}
    total = sum(t["chi2"] for t in out.values())
    for test in out.values():
        test["share"] = test["chi2"] / total * 100.0 if total > 0 else None
    return out


def _coefficient(row: dict[str, Any], valid: bool) -> dict[str, Any]:
    lo, hi, p = row["ci_lower"], row["ci_upper"], row["p_value"]
    has_interval = math.isfinite(lo) and math.isfinite(hi)
    return {
        "feature": row["feature"],
        "estimate": row["estimate"],
        "std_error": row["std_error"],
        "z_value": row["z"],
        "p_value": p,
        "significant": bool(valid and math.isfinite(p) and p < 0.05),
        "conf_int": [lo, hi] if has_interval else None,
        "relativity": float(np.exp(row["estimate"])),
        "relativity_ci": [float(np.exp(lo)), float(np.exp(hi))] if has_interval else None,
        "robust_std_error": None,
    }


def _term_coefficients(
    model: Any, cols: list[int], rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    names = {model.feature_names[c] for c in cols}
    return [
        {
            "term": c["feature"],
            "estimate": c["estimate"],
            "std_error": c["std_error"],
            "z_value": c["z_value"],
            "p_value": c["p_value"],
            "relativity": c["relativity"],
        }
        for c in rows
        if c["feature"] in names
    ]


def _dataset_summary(data: _Frame, row: Any, classes: list[str]) -> dict[str, Any]:
    """The metrics of a dataset as a whole, across the classes."""
    from rustystats.multinomial import _share_based_null_deviance

    null = _share_based_null_deviance(row.actual_class_counts, row.total_weight)
    k = len(classes)
    expected = np.zeros((k, k))
    for c in range(k):
        mask = data.y_codes == c
        expected[c] = (data.probabilities[mask] * data.weight[mask, None]).sum(axis=0)
    return {
        "label": data.label,
        "rows": row.nobs,
        "metrics": {
            "mass": row.total_weight,
            "deviance": row.deviance,
            "null_deviance": null,
            "deviance_explained": 1.0 - row.deviance / null if null else None,
            "log_likelihood": -0.5 * row.deviance,
            "log_loss": row.log_loss,
            "accuracy": row.accuracy,
            "top_2_accuracy": row.top_2_accuracy,
            "balanced_accuracy": row.balanced_accuracy,
            "macro_precision": row.macro_precision,
            "macro_recall": row.macro_recall,
            "macro_f1": row.macro_f1,
            "ece": row.multiclass_expected_calibration_error,
        },
        "class_matrix": {
            "classes": classes,
            "confusion": row.confusion_matrix.tolist(),
            "expected": expected.tolist(),
        },
        "class_mix": {
            "actual": dict(row.actual_class_mix),
            "predicted": dict(row.predicted_class_mix),
        },
    }


def _class_dataset(
    d: _Scored,
    variance: np.ndarray,
    rate_bins: tuple[np.ndarray, bool],
    row: Any,
    label: str,
) -> dict[str, Any]:
    """One dataset as a one-vs-rest problem for one class."""
    w = d.weight
    deviance = float(np.sum(w * _unit_deviance(d.y, d.mu, "binomial")))
    null = _null_deviance(d, "binomial", "logit")
    mass = float(w.sum())
    actual, expected = float(np.sum(w * d.y)), float(np.sum(w * d.mu))
    gini = _gini(w * d.y, d.mu, w)
    perfect = _gini(w * d.y, d.y, w)
    stats = row.per_class_metrics[label]
    return {
        "label": d.label,
        "rows": d.n,
        "metrics": {
            "mass": mass,
            "actual": actual,
            "expected": expected,
            "ae": _ratio(actual, expected),
            "ae_z": _total_z(actual, expected, float(np.sum(w * variance))),
            "mean_actual": _ratio(actual, mass),
            "mean_predicted": _ratio(expected, mass),
            "deviance": deviance,
            "null_deviance": null,
            "deviance_explained": 1.0 - deviance / null if null > 0 else None,
            "gini": gini,
            "gini_normalised": gini / perfect if gini is not None and perfect else None,
            "precision": stats["precision"],
            "recall": stats["recall"],
            "f1": stats["f1"],
            "ece": row.expected_calibration_error_by_class[label],
        },
        **_curves(d, variance, 1.0, rate_bins),
        "residual_histogram": None,
        "qq": None,
        "count_distribution": None,
    }


def _pair_specs(model: Any, requested: list[Any] | None) -> list[tuple[str, str, bool]]:
    """The pairs to draw: the fitted two-way terms first, then those requested."""
    slots = getattr(getattr(model, "_builder", None), "_term_slots", None) or []
    fitted = [tuple(s.factors) for s in slots if len(s.factors) == 2]
    known = {frozenset(pair) for pair in fitted}
    out = [(a, b, True) for a, b in fitted]
    seen = set(known)
    for spec in requested or []:
        a, b = _normalize_pair_spec(spec)
        if frozenset((a, b)) not in seen:
            seen.add(frozenset((a, b)))
            out.append((a, b, False))
    return out


def _axis(frame: pl.DataFrame, name: str, cats: list[str], weight: np.ndarray) -> dict[str, Any]:
    series = frame[name]
    if name not in cats and series.dtype.is_numeric():
        edges = _bin_continuous_edges(series.cast(pl.Float64).to_numpy(), _PAIR_BINS)
        return {"edges": [float(e) for e in edges], "levels": None}
    return {"edges": None, "levels": _bin_categorical_levels(series, _PAIR_LEVELS, weight)}


def _pair_definitions(
    model: Any,
    frame: pl.DataFrame,
    weight: np.ndarray,
    specs: list[tuple[str, str, bool]],
    cats: list[str],
) -> list[dict[str, Any]]:
    out = []
    for a, b, in_model in specs:
        if a not in frame.columns or b not in frame.columns:
            continue
        rows, cols = _axis(frame, a, cats, weight), _axis(frame, b, cats, weight)
        grid = SimpleNamespace(
            row_axis=a,
            col_axis=b,
            row_edges=rows["edges"],
            row_levels=rows["levels"],
            col_edges=cols["edges"],
            col_levels=cols["levels"],
        )
        out.append({"name": f"{a}:{b}", "grid": grid, "in_model": in_model, "a": a, "b": b})
    return out


def _class_pairs(
    model: Any,
    data: pl.DataFrame,
    pairs: list[dict[str, Any]],
    scored: list[_Scored],
    variance: list[np.ndarray],
    beta: np.ndarray | None,
    coefficients: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out = []
    for pair in pairs:
        grid = pair["grid"]
        rr, r_labels, n_r = _axis_codes(scored[0].data[grid.row_axis], grid, True)
        cc, c_labels, n_c = _axis_codes(scored[0].data[grid.col_axis], grid, False)
        cells = []
        for i, (d, v) in enumerate(zip(scored, variance)):
            if i == 0:
                r_codes, c_codes = rr, cc
            else:
                r_codes, _, _ = _axis_codes(d.data[grid.row_axis], grid, True)
                c_codes, _, _ = _axis_codes(d.data[grid.col_axis], grid, False)
            codes = np.where((r_codes >= 0) & (c_codes >= 0), r_codes * n_c + c_codes, -1)
            cells.append(_band_rows(codes, n_r * n_c, d, v, 1.0))
        slot = _interaction_slot(model, [pair["a"], pair["b"]]) if pair["in_model"] else None
        effect = None
        if beta is not None:
            effect = _pair_effect(
                model,
                data,
                grid,
                slot,
                _grid_points(grid, True),
                _grid_points(grid, False),
                link="logit",
                beta=beta,
            )
        cols = list(range(slot.col_start, slot.col_end)) if slot is not None else []
        out.append(
            {
                "name": pair["name"],
                "factor1": pair["a"],
                "factor2": pair["b"],
                "pair_type": "fitted interaction" if pair["in_model"] else "requested pair",
                "in_model": pair["in_model"],
                "representation": None,
                "row_axis": grid.row_axis,
                "col_axis": grid.col_axis,
                "rows": r_labels,
                "cols": c_labels,
                "n_rows": n_r,
                "n_cols": n_c,
                "cells": cells,
                "effect": effect,
                "significance": None,
                "score_test": None,
                "gvif": None,
                "coefficients": _term_coefficients(model, cols, coefficients),
            }
        )
    return out


def _fit_section(
    model: Any,
    diag: Any,
    cov: np.ndarray | None,
    valid: bool,
    formula: str,
) -> dict[str, Any]:
    n_blocks, p = model.params.shape
    df_resid = diag.nobs * (len(model.classes_) - 1) - model.n_params
    summary: dict[str, Any] = {
        "formula": formula,
        "family": "multinomial",
        "link": "softmax",
        "n_obs": diag.nobs,
        "n_params": model.n_params,
        "df_resid": df_resid,
        "converged": bool(model.converged),
        "iterations": int(model.iterations),
    }
    if model.alpha > 0:
        summary["regularization"] = {
            "type": model.regularization_type,
            "alpha": model.alpha,
            "l1_ratio": model.l1_ratio,
            "cv_deviance": model.cv_deviance,
        }
    correlations: list[dict[str, Any]] = []
    if cov is not None:
        owner, intercept = _term_owners(model, p)
        size = n_blocks * p
        labels = [
            f"{model.classes_[i]} · {name}"
            for i in _non_reference(model)
            for name in model.feature_names
        ]
        correlations = _top_correlations(
            cov[:size, :size], labels, np.tile(owner, n_blocks), np.tile(intercept, n_blocks)
        )
    return {
        "summary": summary,
        "inference_status": model.inference_status,
        "inference_valid": valid,
        "optimizer_route": model.solver_name,
        "solver_status": model.solver_status,
        "fit_diagnostics": {},
        "warnings": list(model.warnings),
        "solver_trace": [],
        "boundary_active": list(model.active_constraints),
        "calibration": {},
        "residual_summary": None,
        "model_comparison": {},
        "overdispersion": None,
        "smooth_terms": _attr(model, "smooth_terms") or [],
        "spline_info": None,
        "regularization_path": _attr(model, "regularization_path"),
        "cv_profile": _attr(model, "cv_profile"),
        "vif": [],
        "kind": "multinomial",
        "dispersion": None,
        "credibility_dispersion": 1.0,
        "fixed_scale": True,
        "pearson_dispersion": None,
        "base_rate": None,
        "influence": {"available": False, "reason": "not computed for multinomial models"},
        "coefficient_correlations": correlations,
        "terms": _terms(model),
    }


def _non_reference(model: Any) -> list[int]:
    return [i for i, label in enumerate(model.classes_) if label != model.reference_]


def _review(report: dict[str, Any], th: dict[str, float | None]) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    fit = report["fit"]
    summary = fit["summary"]
    if summary["converged"]:
        flags.append(
            _flag(
                "converged",
                "pass",
                f"converged in {summary['iterations']} iterations",
                value=summary["iterations"],
            )
        )
    else:
        flags.append(
            _flag(
                "converged",
                "critical",
                f"the fit did not converge in {summary['iterations']} iterations",
            )
        )
    if not fit["inference_valid"]:
        flags.append(
            _flag(
                "inference",
                "info",
                f"standard errors are not valid as usual ({fit['inference_status']}); "
                "confidence bands and p-values are left out or naive",
            )
        )
    n_datasets = len(report["datasets"])
    for view in report["classes"]:
        label = view["class"]
        for i, ds in enumerate(view["datasets"]):
            tol = th["ae_train"] if i == 0 else th["ae_test"]
            _review_overall(flags, ds, tol, th["factor_z"], cls=label)
            _review_deciles(flags, ds, th, i > 0 or n_datasets == 1, cls=label)
            _review_bands(flags, view, th, i, cls=label)
        _review_interactions(flags, view, th, cls=label)
    # Effects are per class, but the reference class has none; read a class that does.
    first = next(v for v in report["classes"] if not v["is_reference"])
    _review_effects(flags, first["features"])
    _review_holdout(flags, report["datasets"], th)
    _review_population(flags, first["features"], report["datasets"], th)
    ptol = th["term_p"]
    if ptol is not None and fit["inference_valid"]:
        for f in first["features"]:
            sig = f.get("significance") or {}
            if f["in_model"] and sig.get("p") is not None and sig["p"] > ptol:
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
    for item in fit["boundary_active"][:20]:
        flags.append(
            _flag("boundary", "info", f"constraint active on {item.get('feature', 'a term')}")
        )
    for warning in fit["warnings"]:
        flags.append(_flag("diagnostics", "info", str(warning)))
    return flags


def _summary_metrics(report: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    put = metric_writer(out)
    for ds in report["datasets"]:
        m, label = ds["metrics"], ds["label"]
        for name in (
            "deviance",
            "null_deviance",
            "deviance_explained",
            "log_loss",
            "accuracy",
            "top_2_accuracy",
            "balanced_accuracy",
            "macro_f1",
            "mass",
        ):
            put(f"{label}_{name}", m.get(name))
        put(f"{label}_rows", ds["rows"])
    for view in report["classes"]:
        for ds in view["datasets"]:
            for name in ("ae", "gini"):
                put(f"{ds['label']}_{name}_{view['class']}", ds["metrics"].get(name))
    put("n_params", report["fit"]["summary"]["n_params"])
    put("iterations", report["fit"]["summary"]["iterations"])
    flags = report["review"]
    put("review_warnings", sum(f["status"] == "warning" for f in flags))
    put("review_critical", sum(f["status"] == "critical" for f in flags))
    return out


def build_multinomial_report(
    model: MultinomialModel,
    train_data: pl.DataFrame,
    test_data: pl.DataFrame | None = None,
    *,
    name: str | None = None,
    thresholds: dict[str, Any] | None = None,
    categorical_factors: list[str] | None = None,
    continuous_factors: list[str] | None = None,
    interactions: list[Any] | None = None,
    n_factor_bins: int = 10,
    weights: str | None = None,
    labels: tuple[str, str] = ("train", "test"),
) -> dict[str, Any]:
    """The report for ``MultinomialModel.validation_report``; see that method."""
    from rustystats.multinomial import _DeserializedMultinomialResult

    th = resolve_thresholds(thresholds)
    check_arguments(train_data, test_data, labels)
    cats, conts = _factors(model, train_data, categorical_factors, continuous_factors)
    diag = model.diagnostics(
        train_data, test_data, categorical_factors=cats, continuous_factors=conts, weights=weights
    )
    specs = _pair_specs(model, interactions)
    # Columns the pairs need beyond the factors reviewed.
    extra = {f for a, b, _ in specs for f in (a, b)} - set(cats) - set(conts)
    extra_cats = [f for f in extra if f in train_data.columns]
    frames = []
    for label, data in zip(labels, (train_data, test_data)):
        if data is None:
            continue
        frame, probabilities, y_codes, weight = model._diagnostic_arrays_from_data(
            data,
            categorical_factors=cats + extra_cats,
            continuous_factors=conts,
            weights_override=weights,
        )
        frames.append(_Frame(label, frame, probabilities, y_codes, weight))
    rows = [diag.train] + ([diag.test] if diag.test is not None else [])
    classes = list(model.classes_)
    valid_status = all(part in _VALID_INFERENCE for part in str(model.inference_status).split("+"))
    raw_cov = model._covariance()
    valid = valid_status and raw_cov is not None
    cov = raw_cov if valid else None
    # The joint tests rank the factors even when the standard errors are only naive.
    tests = _joint_tests(model, cats + conts, raw_cov)
    train = frames[0]
    names = [(f, "categorical") for f in cats] + [(f, "continuous") for f in conts]
    names = [(f, t) for f, t in names if f in train.frame.columns]
    defs = {f: _band_defs(f, t, train.frame, train.weight, n_factor_bins) for f, t in names}
    codes = [{f: _codes(m.frame[f], defs[f]) for f, _ in names} for m in frames]
    pairs = _pair_definitions(model, train.frame, train.weight, specs, cats)
    in_pairs: dict[str, list[str]] = {}
    for pair in pairs:
        if pair["in_model"]:
            in_pairs.setdefault(pair["a"], []).append(pair["name"])
            in_pairs.setdefault(pair["b"], []).append(pair["name"])
    coefficient_rows = model.coef_table("records")
    blocks = {classes[i]: b for b, i in enumerate(_non_reference(model))}
    p = model.params.shape[1]
    views = []
    for k, label in enumerate(classes):
        block = blocks.get(label)
        scored = [m.scored(k) for m in frames]
        variance = [_variance("binomial", d.mu, 1.5, 1.0) for d in scored]
        rate_bins = _rate_edges(scored[0].mu)
        beta = None if block is None else np.asarray(model.params[block], dtype=np.float64)
        class_cov = None
        if cov is not None and block is not None:
            class_cov = cov[block * p : (block + 1) * p, block * p : (block + 1) * p]
        coefficients = [
            _coefficient(row, valid)
            for row in coefficient_rows
            if row["class"] == label and row["coefficient_type"] != "alternative_generic"
        ]
        features = []
        for f, ftype in names:
            bands = [
                _band_rows(codes[i][f], len(defs[f]["labels"]), d, v, 1.0)
                for i, (d, v) in enumerate(zip(scored, variance))
            ]
            missing = []
            for i, d in enumerate(scored):
                miss = d.data[f].is_null().to_numpy() | (codes[i][f] < 0)
                missing.append(
                    {
                        "rows": int(miss.sum()),
                        "mass": float(d.mass[miss].sum()),
                        "mass_share": _ratio(float(d.mass[miss].sum()), float(d.mass.sum())),
                    }
                )
            psi = None
            if len(frames) > 1:
                psi = _psi(
                    [b["mass"] for b in bands[0]] + [_unseen_mass(codes[0][f], scored[0])],
                    [b["mass"] for b in bands[1]] + [_unseen_mass(codes[1][f], scored[1])],
                )
            cols = _main_columns(model, f)
            effect = None
            if beta is not None and cols and f in train_data.columns:
                effect = _effect(
                    model,
                    f,
                    ftype,
                    train_data,
                    defs[f],
                    train.weight,
                    class_cov,
                    link="logit",
                    beta=beta,
                )
            test = tests.get(f)
            features.append(
                {
                    "feature": f,
                    "type": ftype,
                    "in_model": bool(cols) or f in (model._terms_dict or {}),
                    "transform": None,
                    "relative_importance": test["share"] if test else None,
                    "significance": (
                        {"chi2": test["chi2"], "p": test["p"] if valid else None} if test else None
                    ),
                    "score_test": None,
                    "gvif": None,
                    "df": len(cols),
                    "residual_pattern": None,
                    "coefficients": _term_coefficients(model, cols, coefficients),
                    "labels": defs[f]["labels"],
                    "edges": defs[f].get("edges"),
                    "mid": defs[f].get("mid"),
                    "bands": bands,
                    "missing": missing,
                    "psi": psi,
                    "effect": effect,
                    "partial_dependence": None,
                    "in_interactions": in_pairs.get(f, []),
                }
            )
        features.sort(
            key=lambda f: (not f["in_model"], -(f["relative_importance"] or 0.0), f["feature"])
        )
        views.append(
            {
                "class": label,
                "is_reference": block is None,
                "datasets": [
                    _class_dataset(d, v, rate_bins, row, label)
                    for d, v, row in zip(scored, variance, rows)
                ],
                "features": features,
                "interactions": _class_pairs(
                    model, train_data, pairs, scored, variance, beta, coefficients
                ),
                "coefficients": coefficients,
            }
        )
    terms = [slot["term"] for slot in _terms(model) if slot["type"] != "intercept"]
    formula = f"{model.response} ~ {' + '.join(terms)}"
    fit = _fit_section(model, diag, cov, valid, formula)
    datasets = [_dataset_summary(m, row, classes) for m, row in zip(frames, rows)]
    datasets[0]["metrics"] |= {"aic": diag.aic, "bic": diag.bic}
    # Both deviances from the rows and weights the report was given.
    null, deviance = datasets[0]["metrics"]["null_deviance"], datasets[0]["metrics"]["deviance"]
    fit["summary"]["null_deviance"] = null
    df = model.n_params - (len(classes) - 1)
    if null is not None and df > 0 and null >= deviance:
        fit["model_comparison"] = {
            "likelihood_ratio_chi2": null - deviance,
            "likelihood_ratio_df": df,
            "likelihood_ratio_pvalue": max(0.0, 1.0 - float(_chi2_cdf(null - deviance, df))),
        }
    report: dict[str, Any] = {
        "report_version": REPORT_VERSION,
        "name": name,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "model": {
            "formula": formula,
            "family": "multinomial",
            "family_kind": "multinomial",
            "link": "softmax",
            "multiclass": True,
            "classes": classes,
            "reference": model.reference_,
            "n_obs": diag.nobs,
            "n_params": model.n_params,
            "df_resid": fit["summary"]["df_resid"],
            "response": model.response,
            "exposure_column": None,
            "weights_column": weights or _column_name(model, "_weights_spec"),
            "offset_column": None,
            "deserialized": isinstance(model._result, _DeserializedMultinomialResult),
            "rustystats_version": _version(),
            "terms_dict": _plain(model._terms_dict),
            "interactions_spec": _plain(model._interactions_spec),
        },
        "datasets": datasets,
        "classes": views,
        "alternative_coefficients": [
            _coefficient(row, valid)
            for row in coefficient_rows
            if row["coefficient_type"] == "alternative_generic"
        ],
        "interaction_candidates": [],
        "encodings": [],
        "factor_deviance": [],
        "base_predictions": None,
        "fit": fit,
        "thresholds": th,
        "threshold_descriptions": THRESHOLD_DESCRIPTIONS,
        "diagnostic_warnings": [],
        "caveat": (
            "The report describes the data passed to it; it cannot verify which rows were used "
            "to fit the model and which were held out."
        ),
    }
    report["review"] = _review(report, th)
    report["summary_metrics"] = _summary_metrics(report)
    return json_safe(report)
