#!/usr/bin/env python3
"""Deep performance and memory smoke scenarios for high-assurance CI."""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl
import rustystats as rs
from rustystats._rustystats import fit_multinomial_py


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument(
        "--scenario",
        action="append",
        choices=[
            "all",
            "poisson_categorical_exposure",
            "smooth_poisson_bs",
            "multinomial_native_masked",
        ],
        default=[],
    )
    return parser.parse_args()


def _rss_mb() -> float:
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0)
    except Exception:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return usage / (1024.0 * 1024.0)
        return usage / 1024.0


def _budget(config: dict, key: str) -> float:
    return float(config[key])


def _load_baseline(path: Path) -> dict:
    baseline = json.loads(path.read_text(encoding="utf-8"))
    if baseline.get("schema_version") != 1:
        raise ValueError("baseline schema_version must be 1")
    return baseline


def _poisson_categorical_exposure(config: dict) -> dict:
    n = int(config["n"])
    rng = np.random.default_rng(20260701)
    age = rng.uniform(18.0, 80.0, n)
    vehicle_power = rng.uniform(50.0, 220.0, n)
    region = rng.choice(["A", "B", "C", "D", "E", "F", "G"], n)
    exposure = rng.uniform(0.1, 2.5, n)
    eta = -3.0 + 0.015 * age + 0.002 * vehicle_power + 0.25 * (region == "A")
    claims = rng.poisson(np.exp(eta) * exposure).astype(float)
    data = pl.DataFrame(
        {
            "claims": claims,
            "age": age,
            "vehicle_power": vehicle_power,
            "region": region,
            "exposure": exposure,
        }
    )

    rss_before = _rss_mb()
    start = perf_counter()
    model = rs.glm_dict(
        response="claims",
        terms={
            "age": {"type": "linear"},
            "vehicle_power": {"type": "linear"},
            "region": {"type": "categorical"},
        },
        data=data,
        family="poisson",
        exposure="exposure",
    ).fit(compute_covariance=False)
    fit_seconds = perf_counter() - start
    rss_after_fit = _rss_mb()

    start = perf_counter()
    predictions = np.asarray(model.predict(data), dtype=float)
    predict_seconds = perf_counter() - start
    rss_after_predict = _rss_mb()
    return _scenario_result(
        "poisson_categorical_exposure",
        config,
        fit_seconds,
        predict_seconds,
        max(rss_after_fit, rss_after_predict) - rss_before,
        bool(model.converged),
        bool(np.all(np.isfinite(predictions))),
        {"n": n},
    )


def _smooth_poisson_bs(config: dict) -> dict:
    n = int(config["n"])
    rng = np.random.default_rng(20260702)
    x = rng.uniform(0.0, 5.0, n)
    region = rng.choice(["urban", "rural", "coastal"], n)
    eta = 0.15 + 0.55 * np.sin(x) + 0.20 * (region == "urban")
    y = rng.poisson(np.exp(eta)).astype(float)
    data = pl.DataFrame({"y": y, "x": x, "region": region})

    rss_before = _rss_mb()
    start = perf_counter()
    model = rs.glm_dict(
        response="y",
        terms={
            "x": {"type": "bs", "k": int(config["k"])},
            "region": {"type": "categorical"},
        },
        data=data,
        family="poisson",
    ).fit()
    fit_seconds = perf_counter() - start
    rss_after_fit = _rss_mb()

    start = perf_counter()
    predictions = np.asarray(model.predict(data), dtype=float)
    predict_seconds = perf_counter() - start
    rss_after_predict = _rss_mb()
    return _scenario_result(
        "smooth_poisson_bs",
        config,
        fit_seconds,
        predict_seconds,
        max(rss_after_fit, rss_after_predict) - rss_before,
        bool(model.converged),
        bool(np.all(np.isfinite(predictions))),
        {"n": n, "k": int(config["k"])},
    )


def _masked_softmax(logits: np.ndarray, availability: np.ndarray) -> np.ndarray:
    masked = np.where(availability, logits, -np.inf)
    max_eta = masked.max(axis=1, keepdims=True)
    exp_eta = np.where(availability, np.exp(masked - max_eta), 0.0)
    return exp_eta / exp_eta.sum(axis=1, keepdims=True)


def _sample_classes(probabilities: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    cumulative = np.cumsum(probabilities, axis=1)
    draws = rng.random(probabilities.shape[0])
    return (draws[:, None] > cumulative).sum(axis=1).astype(np.int64)


def _multinomial_native_masked(config: dict) -> dict:
    n_rows = int(config["n_rows"])
    n_features = int(config["n_features"])
    n_classes = int(config["n_classes"])
    rng = np.random.default_rng(20260703)
    x = rng.normal(size=(n_rows, n_features)).astype(np.float64)
    x[:, 0] = 1.0
    beta = rng.normal(scale=0.20, size=(n_classes - 1, n_features))
    logits = np.zeros((n_rows, n_classes), dtype=np.float64)
    logits[:, 1:] = x @ beta.T
    availability = rng.random((n_rows, n_classes)) > 0.12
    availability[:, 0] = True
    probabilities = _masked_softmax(logits, availability)
    y = _sample_classes(probabilities, rng)
    availability[np.arange(n_rows), y] = True

    rss_before = _rss_mb()
    start = perf_counter()
    result = fit_multinomial_py(
        y,
        x,
        n_classes,
        0,
        availability=availability,
        alpha=float(config["alpha"]),
        max_iter=80,
        tol=1e-8,
        skip_covariance=True,
        hessian_memory_limit_bytes=int(config["hessian_memory_limit_bytes"]),
        max_dense_parameters=int(config["max_dense_parameters"]),
    )
    fit_seconds = perf_counter() - start
    rss_after_fit = _rss_mb()

    start = perf_counter()
    scored_logits = np.zeros((n_rows, n_classes), dtype=np.float64)
    scored_logits[:, 1:] = x @ result.params.T
    scored = _masked_softmax(scored_logits, availability)
    predict_seconds = perf_counter() - start
    rss_after_predict = _rss_mb()
    finite = bool(np.all(np.isfinite(scored)) and np.allclose(scored.sum(axis=1), 1.0, atol=1e-10))
    return _scenario_result(
        "multinomial_native_masked",
        config,
        fit_seconds,
        predict_seconds,
        max(rss_after_fit, rss_after_predict) - rss_before,
        bool(result.converged),
        finite,
        {
            "n_rows": n_rows,
            "n_features": n_features,
            "n_classes": n_classes,
            "iterations": int(result.iterations),
        },
    )


def _scenario_result(
    name: str,
    config: dict,
    fit_seconds: float,
    predict_seconds: float,
    rss_delta_mb: float,
    converged: bool,
    finite_predictions: bool,
    dimensions: dict,
) -> dict:
    passed = (
        converged
        and finite_predictions
        and fit_seconds <= _budget(config, "max_fit_seconds")
        and predict_seconds <= _budget(config, "max_predict_seconds")
        and rss_delta_mb <= _budget(config, "max_rss_delta_mb")
    )
    return {
        "scenario": name,
        "dimensions": dimensions,
        "fit_seconds": round(fit_seconds, 6),
        "predict_seconds": round(predict_seconds, 6),
        "rss_peak_delta_mb": round(rss_delta_mb, 3),
        "max_fit_seconds": _budget(config, "max_fit_seconds"),
        "max_predict_seconds": _budget(config, "max_predict_seconds"),
        "max_rss_delta_mb": _budget(config, "max_rss_delta_mb"),
        "converged": converged,
        "finite_predictions": finite_predictions,
        "passed": passed,
    }


SCENARIOS = {
    "poisson_categorical_exposure": _poisson_categorical_exposure,
    "smooth_poisson_bs": _smooth_poisson_bs,
    "multinomial_native_masked": _multinomial_native_masked,
}


def main() -> int:
    args = _parse_args()
    baseline = _load_baseline(args.baseline)
    requested = args.scenario or ["all"]
    names = sorted(SCENARIOS) if "all" in requested else requested
    scenarios_config = baseline.get("scenarios", {})

    results = []
    for name in names:
        if name not in SCENARIOS:
            raise ValueError(f"unknown scenario: {name}")
        config = scenarios_config.get(name)
        if not isinstance(config, dict):
            raise ValueError(f"baseline missing config for scenario: {name}")
        result = SCENARIOS[name](config)
        results.append(result)
        print(
            f"{name}: fit={result['fit_seconds']:.3f}/{result['max_fit_seconds']:.3f}s "
            f"predict={result['predict_seconds']:.3f}/{result['max_predict_seconds']:.3f}s "
            f"rss_delta={result['rss_peak_delta_mb']:.1f}/{result['max_rss_delta_mb']:.1f}MB "
            f"passed={result['passed']}"
        )

    report = {
        "schema_version": 1,
        "scenario_count": len(results),
        "results": results,
        "passed": all(result["passed"] for result in results),
    }
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
