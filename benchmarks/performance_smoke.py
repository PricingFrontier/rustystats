#!/usr/bin/env python3
"""Small performance smoke for CI and scheduled high-assurance runs."""

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


def _load_baseline(path: Path | None) -> dict:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _budget(env_name: str, baseline: dict, key: str, default: float) -> float:
    return float(os.environ.get(env_name, baseline.get(key, default)))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def _rss_mb() -> float:
    """Current resident set size in MB, using psutil when available."""
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0)
    except Exception:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KB; macOS reports bytes. CI is Linux, but keep the
        # fallback portable enough for local release evidence.
        if sys.platform == "darwin":
            return usage / (1024.0 * 1024.0)
        return usage / 1024.0


def main() -> int:
    args = _parse_args()
    baseline = _load_baseline(args.baseline)
    n = int(os.environ.get("RUSTYSTATS_PERF_SMOKE_N", baseline.get("n", "20000")))
    fit_budget = _budget("RUSTYSTATS_PERF_SMOKE_FIT_SECONDS", baseline, "max_fit_seconds", 20.0)
    predict_budget = _budget(
        "RUSTYSTATS_PERF_SMOKE_PREDICT_SECONDS",
        baseline,
        "max_predict_seconds",
        3.0,
    )
    rss_delta_budget = _budget(
        "RUSTYSTATS_PERF_SMOKE_RSS_DELTA_MB",
        baseline,
        "max_rss_delta_mb",
        512.0,
    )

    rng = np.random.default_rng(20260630)
    age = rng.uniform(18.0, 80.0, n)
    vehicle_power = rng.uniform(50.0, 220.0, n)
    region = rng.choice(["A", "B", "C", "D", "E"], n)
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
    predictions = model.predict(data)
    predict_seconds = perf_counter() - start
    rss_after_predict = _rss_mb()
    rss_peak_delta = max(rss_after_fit, rss_after_predict) - rss_before

    result = {
        "scenario": "poisson_categorical_exposure_smoke",
        "n": n,
        "fit_seconds": round(fit_seconds, 6),
        "predict_seconds": round(predict_seconds, 6),
        "rss_before_mb": round(rss_before, 3),
        "rss_after_fit_mb": round(rss_after_fit, 3),
        "rss_after_predict_mb": round(rss_after_predict, 3),
        "rss_peak_delta_mb": round(rss_peak_delta, 3),
        "max_fit_seconds": fit_budget,
        "max_predict_seconds": predict_budget,
        "max_rss_delta_mb": rss_delta_budget,
        "converged": bool(model.converged),
    }

    print(f"fit_seconds={fit_seconds:.3f} budget={fit_budget:.3f} n={n}")
    print(f"predict_seconds={predict_seconds:.3f} budget={predict_budget:.3f} n={n}")
    print(f"rss_peak_delta_mb={rss_peak_delta:.1f} budget={rss_delta_budget:.1f} n={n}")
    if args.json_output is not None:
        args.json_output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    if not model.converged:
        print("Model did not converge.")
        return 1
    if not np.all(np.isfinite(predictions)):
        print("Predictions contain non-finite values.")
        return 1
    if fit_seconds > fit_budget:
        print("Fit time exceeded performance smoke budget.")
        return 1
    if predict_seconds > predict_budget:
        print("Predict time exceeded performance smoke budget.")
        return 1
    if rss_peak_delta > rss_delta_budget:
        print("RSS growth exceeded performance smoke memory budget.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
