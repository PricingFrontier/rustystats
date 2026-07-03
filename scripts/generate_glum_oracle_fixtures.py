#!/usr/bin/env python3
"""Generate archived glum oracle fixtures for regularized Gaussian GLMs."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import glum
import numpy as np
from glum import GeneralizedLinearRegressor

ROOT = Path(__file__).resolve().parents[1]
ORACLE_DIR = ROOT / "tests" / "oracles" / "glum"


def _round_array(values: np.ndarray) -> list[float]:
    return [float(f"{v:.16g}") for v in np.asarray(values, dtype=np.float64)]


def _data_columns(n: int = 100) -> dict[str, list[float]]:
    x1 = np.linspace(-1.5, 1.8, n)
    x2 = np.sin(np.linspace(0.0, 4.0, n))
    x3 = np.cos(np.linspace(0.0, 2.0, n))
    y = 1.0 + 0.7 * x1 - 0.25 * x2 + 0.05 * np.cos(np.arange(n))
    return {
        "y": _round_array(y),
        "x1": _round_array(x1),
        "x2": _round_array(x2),
        "x3": _round_array(x3),
    }


def _kkt_residual(
    x_design: np.ndarray, y: np.ndarray, beta: np.ndarray, alpha: float, l1_ratio: float
) -> float:
    gradient = x_design.T @ (x_design @ beta - y)
    l1 = alpha * l1_ratio
    l2 = alpha * (1.0 - l1_ratio)
    residuals = [abs(float(gradient[0]))]
    for j in range(1, len(beta)):
        if abs(beta[j]) > 1e-10:
            residuals.append(abs(float(gradient[j] + l2 * beta[j] + l1 * np.sign(beta[j]))))
        else:
            residuals.append(max(0.0, abs(float(gradient[j])) - l1))
    return max(residuals)


def _fixture(case_id: str, alpha: float, l1_ratio: float, penalty_name: str) -> dict:
    columns = _data_columns()
    y = np.asarray(columns["y"], dtype=np.float64)
    x = np.column_stack(
        [
            np.asarray(columns["x1"], dtype=np.float64),
            np.asarray(columns["x2"], dtype=np.float64),
            np.asarray(columns["x3"], dtype=np.float64),
        ]
    )
    x_design = np.column_stack([np.ones(len(y)), x])
    # Glum minimizes the average loss, while RustyStats' regularized Gaussian
    # path uses the unnormalized normal-equation scale. alpha / n aligns the
    # two objective definitions exactly when standardize=False.
    glum_alpha = alpha / len(y)
    result = GeneralizedLinearRegressor(
        alpha=glum_alpha,
        l1_ratio=l1_ratio,
        family="normal",
        link="identity",
        fit_intercept=True,
        max_iter=5000,
        gradient_tol=1e-12,
        step_size_tol=1e-12,
        scale_predictors=False,
    )
    result.fit(x, y)
    params = np.r_[result.intercept_, result.coef_]
    fitted = x_design @ params
    return {
        "schema_version": 1,
        "case_id": case_id,
        "oracle": "glum",
        "oracle_version": glum.__version__,
        "data": {"columns": columns},
        "model": {
            "response": "y",
            "terms": {
                "x1": {"type": "linear"},
                "x2": {"type": "linear"},
                "x3": {"type": "linear"},
            },
            "family": "gaussian",
            "alpha": alpha,
            "glum_alpha": glum_alpha,
            "l1_ratio": l1_ratio,
            "standardize": False,
            "fit_kwargs": {
                "max_iter": 1000,
                "tol": 1e-12,
                "compute_covariance": False,
                "standardize": False,
            },
        },
        "expected": {
            "penalty": penalty_name,
            "params": _round_array(params),
            "fittedvalues": _round_array(fitted),
            "nonzero_count": int(np.count_nonzero(np.abs(params[1:]) > 1e-10)),
            "kkt_max_abs": float(f"{_kkt_residual(x_design, y, params, alpha, l1_ratio):.16g}"),
        },
        "tolerances": {
            "params_atol": 5e-7,
            "params_rtol": 5e-7,
            "prediction_atol": 5e-7,
            "prediction_rtol": 5e-7,
            "kkt_atol": 2e-6,
        },
    }


def build_fixtures() -> dict[str, dict]:
    return {
        "gaussian_ridge.json": _fixture(
            "glum-gaussian-ridge",
            alpha=0.5,
            l1_ratio=0.0,
            penalty_name="ridge",
        ),
        "gaussian_lasso.json": _fixture(
            "glum-gaussian-lasso",
            alpha=0.5,
            l1_ratio=1.0,
            penalty_name="lasso",
        ),
        "gaussian_elastic_net.json": _fixture(
            "glum-gaussian-elastic-net",
            alpha=0.2,
            l1_ratio=0.5,
            penalty_name="elastic_net",
        ),
    }


def _write_fixtures(directory: Path, fixtures: dict[str, dict]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, fixture in fixtures.items():
        (directory / name).write_text(
            json.dumps(fixture, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accept", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    fixtures = build_fixtures()
    if args.accept:
        _write_fixtures(ORACLE_DIR, fixtures)
        print(f"Wrote {len(fixtures)} glum oracle fixtures to {ORACLE_DIR.relative_to(ROOT)}.")
        return 0

    output_dir = args.output_dir or Path(tempfile.mkdtemp(prefix="rustystats-glum-oracles-"))
    if args.output_dir and output_dir.exists():
        shutil.rmtree(output_dir)
    _write_fixtures(output_dir, fixtures)

    if args.check:
        failures = []
        for name in fixtures:
            expected = ORACLE_DIR / name
            generated = output_dir / name
            if not expected.is_file():
                failures.append(f"missing checked-in fixture {expected.relative_to(ROOT)}")
            elif expected.read_text(encoding="utf-8") != generated.read_text(encoding="utf-8"):
                failures.append(f"fixture drift: {expected.relative_to(ROOT)}")
        if failures:
            print("Glum oracle fixture generation check failed:")
            for failure in failures:
                print(f" - {failure}")
            print(f"Generated comparison fixtures in {output_dir}")
            return 1
        print(f"Glum oracle fixture generation check passed for {len(fixtures)} fixtures.")
        return 0

    print(f"Wrote comparison fixtures to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
