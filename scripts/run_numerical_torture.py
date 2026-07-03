#!/usr/bin/env python3
"""Run deterministic numerical torture scenarios."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import rustystats as rs

ROOT = Path(__file__).resolve().parents[1]
SCENARIO_FILE = ROOT / "specs" / "numerical_torture_scenarios.json"


def _finite_model_result(result, data: pl.DataFrame, **predict_kwargs) -> dict[str, Any]:
    params = np.asarray(result.params, dtype=np.float64)
    pred = np.asarray(result.predict(data, **predict_kwargs), dtype=np.float64)
    if not np.all(np.isfinite(params)):
        raise AssertionError("non-finite fitted coefficients")
    if not np.all(np.isfinite(pred)):
        raise AssertionError("non-finite predictions")
    return {
        "params_max_abs": float(np.max(np.abs(params))) if params.size else 0.0,
        "prediction_min": float(np.min(pred)) if pred.size else 0.0,
        "prediction_max": float(np.max(pred)) if pred.size else 0.0,
    }


def zero_weight_rejection() -> dict[str, Any]:
    data = pl.DataFrame({"y": [1.0, 2.0, 3.0], "x": [0.0, 1.0, 2.0], "w": [0.0, 0.0, 0.0]})
    rs.glm_dict(
        response="y",
        terms={"x": {"type": "linear"}},
        data=data,
        family="poisson",
        weights="w",
    ).fit()
    raise AssertionError("zero-weight fit unexpectedly succeeded")


def poisson_exposure_extremes() -> dict[str, Any]:
    x = np.linspace(-1.0, 1.0, 48)
    exposure = np.geomspace(1e-6, 1e6, x.size)
    rate = np.exp(-9.0 + 0.15 * x)
    y = np.floor(rate * exposure + (np.arange(x.size) % 3 == 0)).astype(float)
    data = pl.DataFrame({"y": y, "x": x, "exposure": exposure})
    result = rs.glm_dict(
        response="y",
        terms={"x": {"type": "linear"}},
        data=data,
        family="poisson",
        exposure="exposure",
    ).fit(max_iter=100, tol=1e-10)
    return _finite_model_result(result, data)


def gamma_boundary_rejection() -> dict[str, Any]:
    data = pl.DataFrame({"y": [0.0, 1.2, 2.5, 3.1], "x": [0.0, 1.0, 2.0, 3.0]})
    rs.glm_dict(response="y", terms={"x": {"type": "linear"}}, data=data, family="gamma").fit()
    raise AssertionError("gamma boundary fit unexpectedly succeeded")


def tweedie_boundary_fit() -> dict[str, Any]:
    x = np.linspace(-1.5, 1.5, 80)
    mu = np.exp(0.2 + 0.25 * x)
    y = mu * (1.0 + 0.05 * np.sin(np.arange(x.size)))
    y[::7] = 0.0
    data = pl.DataFrame({"y": y, "x": x})
    result = rs.glm_dict(
        response="y",
        terms={"x": {"type": "linear"}},
        data=data,
        family="tweedie",
        var_power=1.5,
    ).fit(max_iter=100, tol=1e-10)
    return _finite_model_result(result, data)


def negbinomial_theta_extreme_fit() -> dict[str, Any]:
    x = np.linspace(-2.0, 2.0, 72)
    mu = np.exp(0.1 + 0.18 * x)
    y = np.floor(mu + (np.arange(x.size) % 5 == 0)).astype(float)
    data = pl.DataFrame({"y": y, "x": x})
    result = rs.glm_dict(
        response="y",
        terms={"x": {"type": "linear"}},
        data=data,
        family="negbinomial",
        theta=0.25,
    ).fit(max_iter=100, tol=1e-10)
    return _finite_model_result(result, data)


def rare_unseen_category_prediction() -> dict[str, Any]:
    data = pl.DataFrame(
        {
            "y": [0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 3.0, 0.0],
            "x": np.linspace(-1.0, 1.0, 8),
            "brand": ["A", "A", "B", "B", "C", "C", "RARE", "A"],
        }
    )
    result = rs.glm_dict(
        response="y",
        terms={"x": {"type": "linear"}, "brand": {"type": "categorical"}},
        data=data,
        family="poisson",
    ).fit(max_iter=100, tol=1e-10)
    pred_data = pl.DataFrame({"x": [-0.5, 0.5], "brand": ["RARE", "UNSEEN"]})
    pred = np.asarray(result.predict(pred_data), dtype=np.float64)
    if not np.all(np.isfinite(pred)):
        raise AssertionError("unseen-category predictions were not finite")
    return {"prediction_min": float(np.min(pred)), "prediction_max": float(np.max(pred))}


def singular_gaussian_fit() -> dict[str, Any]:
    x = np.linspace(-2.0, 2.0, 64)
    data = pl.DataFrame({"y": 1.0 + 0.3 * x, "x1": x, "x2": 2.0 * x})
    try:
        result = rs.glm_dict(
            response="y",
            terms={"x1": {"type": "linear"}, "x2": {"type": "linear"}},
            data=data,
            family="gaussian",
        ).fit(max_iter=100, tol=1e-10, compute_covariance=False)
    except Exception as exc:
        message = str(exc).lower()
        if "singular" in message or "multicollinearity" in message:
            return {"controlled_error": exc.__class__.__name__, "message": str(exc)}
        raise
    return _finite_model_result(result, data)


def binomial_perfect_separation_status() -> dict[str, Any]:
    x = np.linspace(-3.0, 3.0, 80)
    y = (x > 0.0).astype(float)
    data = pl.DataFrame({"y": y, "x": x})
    try:
        result = rs.glm_dict(
            response="y",
            terms={"x": {"type": "linear"}},
            data=data,
            family="binomial",
        ).fit(max_iter=50, tol=1e-10, compute_covariance=False)
    except Exception as exc:
        message = str(exc).lower()
        if "conver" in message or "singular" in message or "separation" in message:
            return {"controlled_error": exc.__class__.__name__, "message": str(exc)}
        raise
    return _finite_model_result(result, data)


ACTION_MAP: dict[str, Callable[[], dict[str, Any]]] = {
    "zero_weight_rejection": zero_weight_rejection,
    "poisson_exposure_extremes": poisson_exposure_extremes,
    "gamma_boundary_rejection": gamma_boundary_rejection,
    "tweedie_boundary_fit": tweedie_boundary_fit,
    "negbinomial_theta_extreme_fit": negbinomial_theta_extreme_fit,
    "rare_unseen_category_prediction": rare_unseen_category_prediction,
    "singular_gaussian_fit": singular_gaussian_fit,
    "binomial_perfect_separation_status": binomial_perfect_separation_status,
}


def load_scenarios(path: Path = SCENARIO_FILE) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("numerical torture scenario schema_version must be 1")
    scenarios = data.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("numerical torture scenario list must be non-empty")
    return scenarios


def run_scenario(scenario: dict[str, Any]) -> dict[str, Any]:
    action_name = scenario["action"]
    action = ACTION_MAP.get(action_name)
    if action is None:
        return {"id": scenario["id"], "status": "failed", "error": f"unknown action {action_name}"}

    expected = scenario["expected"]
    try:
        details = action()
    except Exception as exc:
        if expected == "validation_error" and isinstance(exc, rs.ValidationError):
            required = scenario.get("require_message")
            if required and required.lower() not in str(exc).lower():
                return {
                    "id": scenario["id"],
                    "status": "failed",
                    "error": f"message {str(exc)!r} did not contain {required!r}",
                }
            return {"id": scenario["id"], "status": "passed", "exception": exc.__class__.__name__}
        return {
            "id": scenario["id"],
            "status": "failed",
            "exception": exc.__class__.__name__,
            "error": str(exc),
        }

    if expected == "validation_error":
        return {"id": scenario["id"], "status": "failed", "error": "expected validation error"}
    return {"id": scenario["id"], "status": "passed", "details": details}


def run_all(scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [run_scenario(scenario) for scenario in scenarios]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-file", type=Path, default=SCENARIO_FILE)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    scenarios = load_scenarios(args.scenario_file)
    results = run_all(scenarios)
    report = {"schema_version": 1, "scenario_count": len(results), "results": results}
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    failures = [result for result in results if result["status"] != "passed"]
    if failures:
        print("Numerical torture suite failed:")
        for failure in failures:
            print(f" - {failure['id']}: {failure.get('error') or failure.get('exception')}")
        return 1

    print(f"Numerical torture suite passed for {len(results)} scenarios.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
